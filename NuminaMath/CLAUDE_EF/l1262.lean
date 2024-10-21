import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1262_126254

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem max_omega_value (ω φ : ℝ) :
  ω > 0 →
  |φ| ≤ π / 2 →
  f ω φ (-π / 4) = 0 →
  (∀ x : ℝ, f ω φ (π / 4 + x) = f ω φ (π / 4 - x)) →
  (∀ x y : ℝ, π / 18 < x → x < y → y < 5 * π / 36 → f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y) →
  (∀ ω' : ℝ, ω' > ω → ¬(
    ω' > 0 ∧
    ∃ φ' : ℝ, |φ'| ≤ π / 2 ∧
    f ω' φ' (-π / 4) = 0 ∧
    (∀ x : ℝ, f ω' φ' (π / 4 + x) = f ω' φ' (π / 4 - x)) ∧
    (∀ x y : ℝ, π / 18 < x → x < y → y < 5 * π / 36 → f ω' φ' x < f ω' φ' y ∨ f ω' φ' x > f ω' φ' y)
  )) →
  ω = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1262_126254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_calculation_l1262_126289

noncomputable section

def top_diameter : ℝ := 10
def bottom_diameter : ℝ := 14
def frustum_height : ℝ := 8

def top_radius : ℝ := top_diameter / 2
def bottom_radius : ℝ := bottom_diameter / 2

def frustum_volume (r1 r2 h : ℝ) : ℝ := (Real.pi * h * (r1^2 + r1*r2 + r2^2)) / 3

theorem frustum_volume_calculation :
  frustum_volume top_radius bottom_radius frustum_height = 872 * Real.pi / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_calculation_l1262_126289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_in_still_water_l1262_126280

/-- Represents the speed of a swimmer in still water and the speed of the stream -/
structure SwimmerSpeed where
  stillWater : ℝ
  stream : ℝ

/-- Proves that given the downstream and upstream distances traveled in 7 hours,
    the swimmer's speed in still water is 8 km/h -/
theorem swimmer_speed_in_still_water
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (time : ℝ)
  (h_downstream : downstream_distance = 91)
  (h_upstream : upstream_distance = 21)
  (h_time : time = 7)
  (s : SwimmerSpeed)
  (h_downstream_eq : downstream_distance = (s.stillWater + s.stream) * time)
  (h_upstream_eq : upstream_distance = (s.stillWater - s.stream) * time) :
  s.stillWater = 8 := by
  sorry

#check swimmer_speed_in_still_water

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_in_still_water_l1262_126280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_F_l1262_126292

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d  -- Arithmetic sequence property
  h3 : (a 2) ^ 2 = a 1 * a 6     -- Geometric sequence condition
  h4 : a 10 = -17

/-- Sum of first n terms of the arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- The fraction we want to minimize -/
noncomputable def F (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  S seq n / 2^n

/-- The theorem stating the minimum value of F -/
theorem min_value_of_F (seq : ArithmeticSequence) :
  ∃ n : ℕ, F seq n = -1/2 ∧ ∀ m : ℕ, F seq m ≥ -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_F_l1262_126292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nut_mixture_weights_l1262_126236

/-- Represents the weight of each type of nut in the mixture -/
structure NutMixture where
  almonds : ℝ
  walnuts : ℝ
  cashews : ℝ
  pistachios : ℝ

/-- Calculates the total weight of the mixture -/
def total_weight (mix : NutMixture) : ℝ :=
  mix.almonds + mix.walnuts + mix.cashews + mix.pistachios

/-- Theorem stating the weights of nuts in the mixture -/
theorem nut_mixture_weights :
  ∃ (mix : NutMixture),
    -- Ratio of almonds : walnuts : cashews is 5 : 3 : 2
    mix.almonds / mix.walnuts = 5 / 3 ∧
    mix.almonds / mix.cashews = 5 / 2 ∧
    -- Ratio of pistachios to almonds is 1 : 4
    mix.pistachios / mix.almonds = 1 / 4 ∧
    -- Total weight is 300 pounds
    total_weight mix = 300 ∧
    -- Approximate weights of each nut type
    (abs (mix.almonds - 133.35) < 0.01 ∧
     abs (mix.walnuts - 80.01) < 0.01 ∧
     abs (mix.cashews - 53.34) < 0.01 ∧
     abs (mix.pistachios - 33.34) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nut_mixture_weights_l1262_126236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_satisfying_conditions_l1262_126241

theorem ordered_pairs_satisfying_conditions :
  ∀ a b : ℕ,
  a > 0 → b > 0 →
  (∃ k : ℤ, k > 0 ∧ (a^3 * b - 1) = k * (a + 1)) ∧
  (∃ m : ℤ, m > 0 ∧ (b^3 * a + 1) = m * (b - 1)) →
  ((a = 1 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_satisfying_conditions_l1262_126241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_pi_sixth_monotone_increasing_theta_range_l1262_126210

-- Define the function f
noncomputable def f (x θ : ℝ) : ℝ := x^2 + 2*x*Real.sin θ - 1

-- Define the domain of x
def domain : Set ℝ := {x | -Real.sqrt 3 / 2 ≤ x ∧ x ≤ 1 / 2}

-- Theorem 1: Minimum value when θ = π/6
theorem min_value_at_pi_sixth :
  ∃ (min : ℝ), min = -5/4 ∧ ∀ x ∈ domain, f x (π/6) ≥ min :=
sorry

-- Theorem 2: Range of θ for monotonically increasing f
theorem monotone_increasing_theta_range :
  ∀ θ, θ ∈ Set.Icc 0 (2*π) →
    (∀ x y, x ∈ domain → y ∈ domain → x < y → f x θ < f y θ) ↔
    θ ∈ Set.Icc (π/3) (2*π/3) ∪ Set.Icc (7*π/6) (11*π/6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_pi_sixth_monotone_increasing_theta_range_l1262_126210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_term_at_6_or_7_l1262_126234

noncomputable def a (n : ℕ) : ℝ := (n + 1 : ℝ) * (7/8) ^ n

theorem max_term_at_6_or_7 :
  ∃ k : ℕ, k ∈ ({6, 7} : Set ℕ) ∧ ∀ n : ℕ, a n ≤ a k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_term_at_6_or_7_l1262_126234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_filling_time_l1262_126257

/-- Represents the time taken to fill the tank -/
noncomputable def T : ℝ := 30

/-- Rate at which pipe A fills the tank -/
noncomputable def rate_A : ℝ := 1 / 60

/-- Rate at which pipe B fills the tank -/
noncomputable def rate_B : ℝ := 1 / 40

/-- Combined rate of pipes A and B -/
noncomputable def combined_rate : ℝ := rate_A + rate_B

/-- Theorem stating that the tank is filled completely -/
theorem tank_filling_time :
  (T / 2) * rate_B + (T / 2) * combined_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_filling_time_l1262_126257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_points_range_l1262_126215

open Real

-- Define the functions g and h
def g (a x : ℝ) : ℝ := a - x^3
noncomputable def h (x : ℝ) : ℝ := 3 * log x

-- Define the interval [1/e, e]
def interval : Set ℝ := { x | 1/exp 1 ≤ x ∧ x ≤ exp 1 }

-- Theorem statement
theorem symmetrical_points_range (a : ℝ) :
  (∃ x ∈ interval, g a x = -h x) → 1 ≤ a ∧ a ≤ exp 1 ^ 3 - 3 := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma interval_nonempty : Set.Nonempty interval := by
  sorry

lemma h_continuous_on_interval : ContinuousOn h interval := by
  sorry

lemma g_continuous_on_interval (a : ℝ) : ContinuousOn (g a) interval := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_points_range_l1262_126215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chelsea_victory_condition_l1262_126275

/-- The total number of shots in the archery contest -/
def total_shots : ℕ := 120

/-- The number of shots taken so far (halfway through the contest) -/
def shots_taken : ℕ := 60

/-- Chelsea's lead at the halfway point -/
def chelsea_lead : ℕ := 60

/-- The minimum score Chelsea gets on each shot -/
def chelsea_min_score : ℕ := 5

/-- The maximum score the opponent can get on each shot -/
def opponent_max_score : ℕ := 10

/-- The score for a bullseye -/
def bullseye_score : ℕ := 10

/-- The number of bullseyes Chelsea needs to shoot in the remaining 60 shots -/
def required_bullseyes : ℕ := 49

theorem chelsea_victory_condition :
  ∀ chelsea_current_score : ℕ,
  chelsea_current_score + bullseye_score * required_bullseyes + 
  chelsea_min_score * (total_shots - shots_taken - required_bullseyes) >
  chelsea_current_score - chelsea_lead + opponent_max_score * (total_shots - shots_taken) ∧
  ∀ n : ℕ, n < required_bullseyes →
    chelsea_current_score + bullseye_score * n + 
    chelsea_min_score * (total_shots - shots_taken - n) ≤
    chelsea_current_score - chelsea_lead + opponent_max_score * (total_shots - shots_taken) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chelsea_victory_condition_l1262_126275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_sum_l1262_126291

/-- Two 3x3 matrices that are inverses of each other -/
def A (x y z w : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, 2, y],
    ![3, 3, 4],
    ![z, 6, w]]

def B (m n p q : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![-6, m, -12],
    ![n, -14, p],
    ![3, q, 5]]

/-- Theorem stating that the sum of elements equals 49 -/
theorem inverse_matrices_sum (x y z w m n p q : ℝ) :
  A x y z w * B m n p q = 1 →
  x + y + z + w + m + n + p + q = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_sum_l1262_126291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_profit_is_200_l1262_126262

/-- Calculates the overall profit from selling a refrigerator and a mobile phone -/
def calculate_overall_profit (refrigerator_cost mobile_cost : ℕ)
  (refrigerator_loss_percent mobile_profit_percent : ℚ) : ℤ :=
  let refrigerator_loss := (refrigerator_cost : ℚ) * refrigerator_loss_percent / 100
  let refrigerator_selling_price := refrigerator_cost - refrigerator_loss.floor
  let mobile_profit := (mobile_cost : ℚ) * mobile_profit_percent / 100
  let mobile_selling_price := mobile_cost + mobile_profit.ceil
  let total_cost := refrigerator_cost + mobile_cost
  let total_selling_price := refrigerator_selling_price + mobile_selling_price
  (total_selling_price : ℤ) - (total_cost : ℤ)

/-- Proves that the overall profit is 200 given the specific conditions -/
theorem overall_profit_is_200 :
  calculate_overall_profit 15000 8000 4 10 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_profit_is_200_l1262_126262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_opposite_A_is_V_l1262_126297

/-- Represents a face of the cube -/
inductive Face : Type
| T | P | X | E | V | A | F

/-- Represents the cube with its faces -/
structure Cube where
  faces : Finset Face
  different : ∀ f₁ f₂, f₁ ∈ faces → f₂ ∈ faces → f₁ ≠ f₂
  complete : faces.card = 6

/-- Represents the adjacency relationship between faces -/
def adjacent (c : Cube) (f₁ f₂ : Face) : Prop := 
  f₁ ∈ c.faces ∧ f₂ ∈ c.faces ∧ f₁ ≠ f₂

/-- The main theorem to prove -/
theorem cube_face_opposite_A_is_V (c : Cube) 
  (h1 : adjacent c Face.T Face.E)
  (h2 : adjacent c Face.T Face.A)
  (h3 : adjacent c Face.X Face.P)
  (h4 : adjacent c Face.X Face.F)
  (h5 : adjacent c Face.E Face.A)
  (h6 : adjacent c Face.E Face.V)
  (h7 : adjacent c Face.A Face.V) :
  ∃ (f : Face), f ∈ c.faces ∧ f = Face.V ∧ 
  (∀ (g : Face), g ∈ c.faces → adjacent c Face.A g → ¬adjacent c f g) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_opposite_A_is_V_l1262_126297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_proper_divisors_432_l1262_126203

def is_proper_divisor (d n : ℕ) : Prop := d ∣ n ∧ d ≠ n

def sum_proper_divisors (n : ℕ) : ℕ := 
  (Finset.filter (λ d => d ∣ n ∧ d ≠ n) (Finset.range n)).sum id

theorem sum_proper_divisors_432 : sum_proper_divisors 432 = 808 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_proper_divisors_432_l1262_126203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_quotient_l1262_126299

theorem smallest_quotient (S : Set ℝ) (hS : S = {-1, 2, -3, 0, 5}) :
  (∀ a b, a ∈ S → b ∈ S → b ≠ 0 → a / b ≥ -5) ∧ 
  (∃ a b, a ∈ S ∧ b ∈ S ∧ b ≠ 0 ∧ a / b = -5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_quotient_l1262_126299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_2_statement_4_l1262_126217

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (lies_in : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (plane_intersect : Plane → Plane → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Statement ②
theorem statement_2 
  (α β γ : Plane) (m n : Line) :
  plane_parallel α β →
  plane_intersect α γ m →
  plane_intersect β γ n →
  parallel m n :=
by
  sorry

-- Statement ④
theorem statement_4 
  (α β : Plane) (m n : Line) :
  plane_intersect α β m →
  parallel n m →
  ¬ lies_in n α →
  ¬ lies_in n β →
  (line_parallel_plane n α ∧ line_parallel_plane n β) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_2_statement_4_l1262_126217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1262_126263

/-- The distance between the foci of a hyperbola given by the equation 9x^2 - 36x - 16y^2 - 64y = 144 -/
theorem hyperbola_foci_distance : 
  let hyperbola := {(x, y) : ℝ × ℝ | 9*x^2 - 36*x - 16*y^2 - 64*y = 144}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ hyperbola ∧ f₂ ∈ hyperbola ∧ 
    ∀ p : ℝ × ℝ, p ∈ hyperbola → 
      |dist p f₁ - dist p f₂| = (2 * Real.sqrt 1189) / (6 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1262_126263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_5672_399201_l1262_126293

noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 0.5 then ⌊x⌋ else ⌈x⌉

theorem round_5672_399201 :
  round_to_nearest 5672.399201 = 5672 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_5672_399201_l1262_126293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_average_speed_l1262_126243

/-- Calculates the average speed of a trip with two segments -/
noncomputable def averageSpeed (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) : ℝ :=
  (distance1 + distance2) / (distance1 / speed1 + distance2 / speed2)

theorem car_trip_average_speed :
  let totalDistance : ℝ := 100  -- Assume total distance is 100 units for simplicity
  let distance1 : ℝ := 0.75 * totalDistance
  let distance2 : ℝ := 0.25 * totalDistance
  let speed1 : ℝ := 60
  let speed2 : ℝ := 20
  averageSpeed distance1 speed1 distance2 speed2 = 40 := by
  sorry

#eval show String from "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_average_speed_l1262_126243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fourth_number_l1262_126279

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_ones : ones ≤ 9

/-- The sum of the digits of a two-digit number -/
def digitSum (n : TwoDigitNumber) : Nat := n.tens + n.ones

/-- The value of a two-digit number -/
def value (n : TwoDigitNumber) : Nat := 10 * n.tens + n.ones

theorem smallest_fourth_number :
  ∀ (fourth : TwoDigitNumber),
  fourth.tens > 6 →
  (let first : TwoDigitNumber := ⟨2, 3, by simp [Nat.le_refl], by simp [Nat.le_refl]⟩
   let second : TwoDigitNumber := ⟨4, 5, by simp [Nat.le_refl], by simp [Nat.le_refl]⟩
   let third : TwoDigitNumber := ⟨3, 6, by simp [Nat.le_refl], by simp [Nat.le_refl]⟩
   digitSum first + digitSum second + digitSum third + digitSum fourth =
   (value first + value second + value third + value fourth) / 4) →
  value fourth ≥ 80 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fourth_number_l1262_126279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_air_quality_exercise_relation_l1262_126246

-- Define the data structure
structure AirQualityData :=
  (excellent_low : ℕ) (excellent_mid : ℕ) (excellent_high : ℕ)
  (good_low : ℕ) (good_mid : ℕ) (good_high : ℕ)
  (mild_low : ℕ) (mild_mid : ℕ) (mild_high : ℕ)
  (moderate_low : ℕ) (moderate_mid : ℕ) (moderate_high : ℕ)

def total_days (data : AirQualityData) : ℕ :=
  data.excellent_low + data.excellent_mid + data.excellent_high +
  data.good_low + data.good_mid + data.good_high +
  data.mild_low + data.mild_mid + data.mild_high +
  data.moderate_low + data.moderate_mid + data.moderate_high

def prob_excellent (data : AirQualityData) : ℚ :=
  (data.excellent_low + data.excellent_mid + data.excellent_high) / (total_days data)

def prob_good (data : AirQualityData) : ℚ :=
  (data.good_low + data.good_mid + data.good_high) / (total_days data)

def prob_mild (data : AirQualityData) : ℚ :=
  (data.mild_low + data.mild_mid + data.mild_high) / (total_days data)

def prob_moderate (data : AirQualityData) : ℚ :=
  (data.moderate_low + data.moderate_mid + data.moderate_high) / (total_days data)

def avg_exercise (data : AirQualityData) : ℚ :=
  (100 * (data.excellent_low + data.good_low + data.mild_low + data.moderate_low) +
   300 * (data.excellent_mid + data.good_mid + data.mild_mid + data.moderate_mid) +
   500 * (data.excellent_high + data.good_high + data.mild_high + data.moderate_high)) /
  (total_days data)

def k_squared (a b c d : ℕ) (total : ℕ) : ℚ :=
  (total * (a * d - b * c)^2) /
  ((a + b) * (c + d) * (a + c) * (b + d))

theorem air_quality_exercise_relation
  (data : AirQualityData)
  (h_total : total_days data = 100)
  (h_data : data = {
    excellent_low := 2, excellent_mid := 16, excellent_high := 25,
    good_low := 5, good_mid := 10, good_high := 12,
    mild_low := 6, mild_mid := 7, mild_high := 8,
    moderate_low := 7, moderate_mid := 2, moderate_high := 0
  }) :
  prob_excellent data + prob_good data + prob_mild data + prob_moderate data = 1 ∧
  avg_exercise data = 350 ∧
  k_squared 33 37 22 8 100 > 3.841 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_air_quality_exercise_relation_l1262_126246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l1262_126283

/-- The volume of a pyramid with a rectangular base and given edge length -/
noncomputable def pyramid_volume (base_length base_width edge_length : ℝ) : ℝ :=
  (1/3) * base_length * base_width * 
  Real.sqrt (edge_length^2 - (1/4) * (base_length^2 + base_width^2))

/-- Theorem: The volume of a pyramid with a 7 × 9 rectangular base and edge length of 15 
    from the apex to each corner of the base is equal to 84√10 -/
theorem pyramid_volume_specific : 
  pyramid_volume 7 9 15 = 84 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l1262_126283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_quadratic_l1262_126221

theorem divisibility_by_quadratic (a b : ℤ) (k : ℕ) (h : ¬(3 ∣ k)) :
  ∃ m : ℤ, (a + b)^(2*k) + a^(2*k) + b^(2*k) = m * (a^2 + a*b + b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_quadratic_l1262_126221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_root_l1262_126258

theorem smallest_cube_root (m n : ℕ) (r : ℝ) : 
  (∀ k < n, ¬ ∃ s : ℝ, s > 0 ∧ s < 1/2000 ∧ ((k : ℝ) + s)^3 = (k^3 : ℝ)) →
  r > 0 →
  r < 1/2000 →
  ((n : ℝ) + r)^3 = m →
  n = 26 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_root_l1262_126258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ACBP_is_five_l1262_126255

noncomputable section

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 3

-- Define the circle C
def circle_C (x y a : ℝ) : Prop := (x - a)^2 + (y - 5)^2 = 10

-- Define point P
def point_P : ℝ × ℝ := (-1/2, 5/2)

-- Define the area of quadrilateral ACBP
noncomputable def area_ACBP (A B C : ℝ × ℝ) : ℝ := 
  let (x_p, y_p) := point_P
  let (x_c, y_c) := C
  let PC := ((x_p - x_c)^2 + (y_p - y_c)^2)^(1/2)
  let AB := ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2)
  (1/2) * PC * AB

-- Theorem statement
theorem area_ACBP_is_five 
  (A B C : ℝ × ℝ) 
  (h_A : line_l A.1 A.2 ∧ circle_C A.1 A.2 C.1)
  (h_B : line_l B.1 B.2 ∧ circle_C B.1 B.2 C.1)
  (h_C : circle_C C.1 C.2 C.1)
  (h_P : ∃ (l₁ l₂ : ℝ × ℝ → Prop), (l₁ A ∧ l₁ point_P) ∧ (l₂ B ∧ l₂ point_P)) :
  area_ACBP A B C = 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ACBP_is_five_l1262_126255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_to_mater_ratio_l1262_126238

/-- The cost of Lightning McQueen in dollars -/
noncomputable def lightning_cost : ℚ := 140000

/-- The cost of Mater as a percentage of Lightning McQueen's cost -/
noncomputable def mater_percentage : ℚ := 10

/-- The cost of Sally McQueen in dollars -/
noncomputable def sally_cost : ℚ := 42000

/-- The cost of Mater in dollars -/
noncomputable def mater_cost : ℚ := (mater_percentage / 100) * lightning_cost

/-- The theorem stating the ratio of Sally McQueen's cost to Mater's cost -/
theorem sally_to_mater_ratio : sally_cost / mater_cost = 3 := by
  -- Expand the definitions
  unfold sally_cost mater_cost mater_percentage lightning_cost
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_to_mater_ratio_l1262_126238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l1262_126272

/-- The curve y = x³ - √3x + 3/5 -/
noncomputable def curve (x : ℝ) : ℝ := x^3 - Real.sqrt 3 * x + 3/5

/-- The derivative of the curve -/
noncomputable def curve_derivative (x : ℝ) : ℝ := 3 * x^2 - Real.sqrt 3

/-- The angle of inclination of the tangent line at a point on the curve -/
noncomputable def angle_of_inclination (x : ℝ) : ℝ := Real.arctan (curve_derivative x)

/-- The theorem stating the range of the angle of inclination -/
theorem angle_of_inclination_range :
  ∀ x : ℝ, (0 ≤ angle_of_inclination x ∧ angle_of_inclination x < π/2) ∨
           (2*π/3 ≤ angle_of_inclination x ∧ angle_of_inclination x < π) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l1262_126272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_age_at_23_feet_l1262_126232

/-- A tree grows linearly from an initial height at a constant rate per year. -/
structure TreeGrowth where
  initial_age : ℕ
  initial_height : ℕ
  growth_rate : ℕ

/-- The height of the tree after a given number of years. -/
def TreeGrowth.height (t : TreeGrowth) (years : ℕ) : ℕ :=
  t.initial_height + t.growth_rate * years

/-- The age of the tree when it reaches a given height. -/
def TreeGrowth.age_at_height (t : TreeGrowth) (target_height : ℕ) : ℕ :=
  t.initial_age + (target_height - t.initial_height) / t.growth_rate

theorem tree_age_at_23_feet :
  let t : TreeGrowth := { initial_age := 1, initial_height := 5, growth_rate := 3 }
  t.age_at_height 23 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_age_at_23_feet_l1262_126232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l1262_126248

-- Define a structure for a right triangle with natural number side lengths
structure RightTriangle where
  a : ℕ  -- First leg
  b : ℕ  -- Second leg
  c : ℕ  -- Hypotenuse
  is_right : a^2 + b^2 = c^2  -- Pythagorean theorem
  coprime_ab : Nat.Coprime a b
  coprime_ac : Nat.Coprime a c
  coprime_bc : Nat.Coprime b c

-- Theorem statement
theorem right_triangle_properties (t : RightTriangle) : 
  Odd t.c ∧ (Odd t.a ∧ Even t.b ∨ Even t.a ∧ Odd t.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l1262_126248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1262_126233

/-- An arithmetic sequence where the sum of the second and fourth terms is 12 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = a n + r

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) 
  (h_sum : a 2 + a 4 = 12) : 
  a 3 + a 5 = 12 := by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1262_126233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_l1262_126204

theorem unique_divisor : ∃! n : ℕ, n > 1 ∧ 200 % n = 2 ∧ 398 % n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_l1262_126204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parallelogram_side_ratio_l1262_126220

/-- A parallelogram with the property that the area of the quadrilateral formed by its internal angle bisectors is one-third of its own area. -/
structure SpecialParallelogram where
  -- Adjacent sides of the parallelogram
  a : ℝ
  b : ℝ
  -- Angles of the parallelogram
  α : ℝ
  β : ℝ
  -- Conditions
  a_pos : 0 < a
  b_pos : 0 < b
  α_pos : 0 < α
  β_pos : 0 < β
  angle_sum : α + β = π / 2
  area_ratio : (a - b)^2 * Real.sin (2 * α) = 2 * a * b * Real.sin (2 * α) / 3

/-- The ratio of adjacent sides in a SpecialParallelogram is (4 + √13) / 3. -/
theorem special_parallelogram_side_ratio (p : SpecialParallelogram) : 
  p.a / p.b = (4 + Real.sqrt 13) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parallelogram_side_ratio_l1262_126220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_size_l1262_126296

/-- Represents the number of students in different categories -/
structure StudentCounts where
  numericMethods : ℕ
  automaticControl : ℕ
  both : ℕ

/-- Calculates the total number of students in the faculty given the counts of second-year students -/
def totalStudents (counts : StudentCounts) (secondYearPercentage : ℚ) : ℕ :=
  let secondYearTotal := counts.numericMethods + counts.automaticControl - counts.both
  Nat.ceil ((secondYearTotal : ℚ) / secondYearPercentage)

/-- Theorem stating that given the conditions, the total number of students is approximately 661 -/
theorem faculty_size (counts : StudentCounts) 
  (h1 : counts.numericMethods = 240)
  (h2 : counts.automaticControl = 423)
  (h3 : counts.both = 134)
  (h4 : secondYearPercentage = 4/5) :
  totalStudents counts secondYearPercentage = 661 := by
  sorry

#eval totalStudents ⟨240, 423, 134⟩ (4/5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_size_l1262_126296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1262_126287

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → (x + x⁻¹) * f y = f (x * y) + f (y / x)

/-- The theorem stating the form of functions satisfying the equation -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
  ∃ c₁ c₂ : ℝ, ∀ x : ℝ, x > 0 → f x = c₁ * x + c₂ / x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1262_126287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_range_l1262_126290

noncomputable def f (a k x : ℝ) : ℝ := a^x + (k-1)*a^(-x) + k^2

theorem odd_function_and_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, f a 0 (-x) = -f a 0 x) ∧
  (f a 0 1 > 0 → ∀ t, (∀ x, f a 0 (x^2 + x) + f a 0 (t - 2*x) > 0) ↔ t > 1/4) :=
by
  sorry

#check odd_function_and_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_range_l1262_126290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1262_126256

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, (2 : ℝ)^x > 0) ↔ (∃ x₀ : ℝ, (2 : ℝ)^x₀ ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1262_126256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_l1262_126288

/-- Represents a chessboard square --/
inductive Square
| Black
| White

/-- Represents a modified 8x8 chessboard with two opposite corners removed --/
def ModifiedChessboard : Type := Fin 62

/-- The color of a square on the modified chessboard --/
def squareColor : ModifiedChessboard → Square :=
  sorry

/-- A domino covers exactly two adjacent squares --/
def Domino : Type := Prod ModifiedChessboard ModifiedChessboard

/-- A tiling of the modified chessboard --/
def Tiling : Type := List Domino

/-- Checks if a tiling is valid (covers all squares exactly once) --/
def isValidTiling (t : Tiling) : Prop :=
  sorry

/-- The number of black squares on the modified chessboard --/
def numBlackSquares : Nat := 30

/-- The number of white squares on the modified chessboard --/
def numWhiteSquares : Nat := 32

/-- Main theorem: It's impossible to tile the modified chessboard with dominoes --/
theorem impossible_tiling : ¬∃ (t : Tiling), isValidTiling t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_l1262_126288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_water_rafting_vests_l1262_126202

def additional_life_vests_needed 
  (total_people : ℕ) 
  (students : ℕ) 
  (vests_on_hand : ℕ) 
  (student_vests_percent : ℚ) : ℕ :=
  total_people - 
  (Nat.floor (student_vests_percent * students) + vests_on_hand)

theorem white_water_rafting_vests 
  (total_people : ℕ) 
  (students : ℕ) 
  (instructors : ℕ) 
  (vests_on_hand : ℕ) 
  (student_vests_percent : ℚ) :
  total_people = students + instructors →
  students = 40 →
  instructors = 10 →
  vests_on_hand = 20 →
  student_vests_percent = 1/5 →
  additional_life_vests_needed total_people students vests_on_hand student_vests_percent = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_water_rafting_vests_l1262_126202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l1262_126261

theorem smallest_difference (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : a * b - 5 * a + 2 * b = 102) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * y - 5 * x + 2 * y = 102 ∧ |Int.ofNat x - Int.ofNat y| ≤ |Int.ofNat a - Int.ofNat b| ∧ |Int.ofNat x - Int.ofNat y| = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l1262_126261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statements_l1262_126214

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  (Real.sqrt 3 / 2) * Real.sin (ω * x) - Real.cos (ω * x / 2)^2 + 1/2

theorem incorrect_statements 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) :
  (∀ x ∈ Set.Ioo 0 π, f ω x ≠ 0 → 0 < ω ∧ ω < 1/6) ∧
  (∀ T > 0, (∀ x, |f ω (x + T)| = |f ω x|) → T = π → ω = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statements_l1262_126214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_l1262_126276

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x + 1)

-- Define the decreasing intervals
def decreasing_intervals : Set (Set ℝ) := {{x | x < -1}, {x | x > -1}}

-- State the theorem
theorem f_decreasing_intervals :
  ∀ (S : Set ℝ), S ∈ decreasing_intervals →
  ∀ (x y : ℝ), x ∈ S → y ∈ S → x < y → f y < f x := by
  sorry

#check f_decreasing_intervals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_l1262_126276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1262_126253

-- Define the function h as noncomputable
noncomputable def h (x : ℝ) : ℝ := (2 * x - 5) / (x - 3)

-- State the theorem about the domain of h
theorem domain_of_h : 
  {x : ℝ | ∃ y, h x = y} = {x : ℝ | x < 3 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1262_126253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_is_focus_l1262_126260

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the directrix
def directrix : ℝ := -1

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the circle
def circleEq (center : ℝ × ℝ) (r : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = r^2

-- Define the fixed point Q
def Q : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem fixed_point_is_focus :
  ∀ (P : PointOnParabola),
  ∃ (r : ℝ),
  (circleEq (P.x, P.y) r (directrix, P.y)) ∧ (circleEq (P.x, P.y) r Q) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_is_focus_l1262_126260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersections_design_possible_l1262_126219

/-- Represents a straight line in a plane -/
structure Line where
  -- Add necessary fields
  mk :: -- Empty structure for now

/-- Represents an intersection between two lines -/
structure Intersection where
  line1 : Line
  line2 : Line

/-- Represents the type of crossing at an intersection -/
inductive CrossingType
| Over
| Under

/-- A country with roads and intersections -/
structure Country where
  roads : Set Line
  intersections : Set Intersection
  crossing_type : Intersection → CrossingType

/-- All roads in the country are straight -/
axiom roads_are_straight (c : Country) : ∀ r, r ∈ c.roads → True

/-- Any two roads intersect -/
axiom roads_intersect (c : Country) (r1 r2 : Line) : 
  r1 ∈ c.roads → r2 ∈ c.roads → r1 ≠ r2 → ∃ i, i ∈ c.intersections ∧ i.line1 = r1 ∧ i.line2 = r2

/-- At every intersection, two roads meet -/
axiom two_roads_per_intersection (c : Country) (i : Intersection) : 
  i ∈ c.intersections → i.line1 ≠ i.line2

/-- Traveling along a road results in alternating over and under crossings -/
def alternating_crossings (c : Country) (r : Line) : Prop :=
  ∀ i j, i ∈ c.intersections → j ∈ c.intersections → 
    (i.line1 = r ∨ i.line2 = r) → 
    (j.line1 = r ∨ j.line2 = r) → 
    i ≠ j → 
    c.crossing_type i ≠ c.crossing_type j

/-- It is possible to design intersections with alternating crossings -/
theorem intersections_design_possible (c : Country) : 
  ∃ crossing_type : Intersection → CrossingType, 
    ∀ r, r ∈ c.roads → alternating_crossings { c with crossing_type := crossing_type } r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersections_design_possible_l1262_126219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_face_area_ratio_l1262_126226

/-- Represents a rectangle with length-to-width ratio of 2:1 -/
structure ClockFaceRectangle where
  width : ℝ
  length : ℝ
  ratio_constraint : length = 2 * width

/-- Represents the area of the central rectangle -/
noncomputable def central_rectangle_area (r : ClockFaceRectangle) : ℝ :=
  r.width * r.width

/-- Represents the area of a corner triangle -/
noncomputable def corner_triangle_area (r : ClockFaceRectangle) : ℝ :=
  (1 / 4) * r.width * r.width

theorem clock_face_area_ratio (r : ClockFaceRectangle) :
  central_rectangle_area r / corner_triangle_area r = 4 := by
  sorry

#check clock_face_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_face_area_ratio_l1262_126226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1262_126265

/-- Given an ellipse with eccentricity √3/2 and a line passing through its right vertex -/
theorem ellipse_line_intersection (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  let e := (3 : ℝ).sqrt / 2
  let G := {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  let A := (a, 0)
  ∃ (C : ℝ × ℝ) (l : Set (ℝ × ℝ)),
    -- The eccentricity of G is √3/2
    e = Real.sqrt (a^2 - b^2) / a →
    -- A is on G
    A ∈ G →
    -- C is on G and l
    C ∈ G ∧ C ∈ l →
    -- A is on l
    A ∈ l →
    -- B is the top vertex of G
    let B := (0, b)
    -- The circle with diameter AC passes through B
    (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 0 →
    -- The standard equation of G
    G = {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1} ∧
    -- The equation of line l
    (l = {p : ℝ × ℝ | p.1 + 2 * p.2 = 2} ∨ l = {p : ℝ × ℝ | 3 * p.1 - 10 * p.2 = 6}) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1262_126265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l1262_126251

theorem max_true_statements (x y : ℝ) (hx : x < 0) (hy : y > 0) :
  ∃ (true_statements : List Bool), 
    (true_statements.length = 4) ∧ 
    (∀ (i : Nat), i < 5 → 
      match i with
      | 0 => (1 / x < 1 / y) = true_statements[0]!
      | 1 => (x^2 > y^2) = true_statements[1]!
      | 2 => (x < y) = true_statements[2]!
      | 3 => (x < 0) = true_statements[3]!
      | 4 => (y > 0) = true_statements[4]!
      | _ => False
    ) ∧
    (¬∃ (better_true_statements : List Bool), 
      (better_true_statements.length > true_statements.length) ∧
      (∀ (i : Nat), i < 5 → 
        match i with
        | 0 => (1 / x < 1 / y) = better_true_statements[0]!
        | 1 => (x^2 > y^2) = better_true_statements[1]!
        | 2 => (x < y) = better_true_statements[2]!
        | 3 => (x < 0) = better_true_statements[3]!
        | 4 => (y > 0) = better_true_statements[4]!
        | _ => False
      )) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l1262_126251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_statement_C_l1262_126230

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the properties and relations
variable (parallel : Line → Line → Prop)
variable (parallelPlane : Plane → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (inPlane : Line → Plane → Prop)

-- Define the non-coincidence property
variable (nonCoincident : Line → Line → Prop)
variable (nonCoincidentPlane : Plane → Plane → Prop)

-- Theorem for statement A
theorem statement_A 
  (a b : Line) (α : Plane)
  (h1 : nonCoincident a b)
  (h2 : parallel a b) 
  (h3 : subset b α) :
  ∃ (S : Set Line), (∀ l ∈ S, inPlane l α ∧ parallel a l) ∧ Set.Infinite S :=
sorry

-- Theorem for statement C
theorem statement_C 
  (a : Line) (α β : Plane)
  (h1 : nonCoincidentPlane α β)
  (h2 : parallelPlane α β) 
  (h3 : subset a α) :
  parallelLinePlane a β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_statement_C_l1262_126230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_wood_weight_l1262_126247

/-- The side length of the equilateral triangle in inches -/
def triangle_side : ℝ := 4

/-- The weight of the equilateral triangle in ounces -/
def triangle_weight : ℝ := 18

/-- The side length of the square in inches -/
def square_side : ℝ := 6

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

/-- The area of a square with side length s -/
def square_area (s : ℝ) : ℝ := s^2

/-- The weight of the square piece of wood in ounces -/
noncomputable def square_weight : ℝ := 
  (square_area square_side / equilateral_triangle_area triangle_side) * triangle_weight

/-- Theorem stating that the weight of the square piece of wood is approximately 93.5 ounces -/
theorem square_wood_weight : 
  ∃ ε > 0, abs (square_weight - 93.5) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_wood_weight_l1262_126247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_salary_problem_l1262_126249

theorem workshop_salary_problem (total_workers : ℕ) (technicians : ℕ) 
  (avg_salary_all : ℚ) (avg_salary_tech : ℚ) :
  total_workers = 22 →
  technicians = 7 →
  avg_salary_all = 850 →
  avg_salary_tech = 1000 →
  let non_techs := total_workers - technicians
  let total_salary := avg_salary_all * total_workers
  let tech_salary := avg_salary_tech * technicians
  let non_tech_salary := total_salary - tech_salary
  let avg_salary_non_tech := non_tech_salary / non_techs
  avg_salary_non_tech = 780 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_salary_problem_l1262_126249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_self_correlated_curves_l1262_126208

/-- Definition of a self-correlated curve -/
def is_self_correlated_curve (Γ : Set (ℝ × ℝ)) : Prop :=
  ∃ M : ℝ × ℝ, ∀ P ∈ Γ, ∃ Q ∈ Γ, Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) * 
                                Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) = 1

/-- Definition of an ellipse -/
def is_ellipse (Γ : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ 
  Γ = {P : ℝ × ℝ | (P.1 - c)^2 / a^2 + P.2^2 / b^2 = 1}

/-- Definition of a hyperbola -/
def is_hyperbola (Γ : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ 
  Γ = {P : ℝ × ℝ | (P.1 - c)^2 / a^2 - P.2^2 / b^2 = 1}

theorem self_correlated_curves :
  (∀ Γ : Set (ℝ × ℝ), is_ellipse Γ → is_self_correlated_curve Γ) ∧
  (¬ ∃ Γ : Set (ℝ × ℝ), is_hyperbola Γ ∧ is_self_correlated_curve Γ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_self_correlated_curves_l1262_126208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l1262_126295

/-- Helper function to calculate the area of a triangle -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := 
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

/-- Given an ellipse E with eccentricity √3/2 and right focus F, 
    where AF has slope 2√3/3 and A is at (0, -2), 
    prove the equation of E and the equation of line l 
    that maximizes the area of triangle OPQ. -/
theorem ellipse_and_max_area_line 
  (E : Set (ℝ × ℝ)) 
  (a b c : ℝ) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) 
  (h_a_gt_b : a > b) 
  (h_E_eq : E = {(x, y) | x^2/a^2 + y^2/b^2 = 1}) 
  (h_ecc : c/a = Real.sqrt 3/2) 
  (F : ℝ × ℝ) 
  (h_F_focus : F.1 > 0 ∧ F.2 = 0) 
  (h_AF_slope : (0 - F.2) / (-2 - F.1) = 2 * Real.sqrt 3/3) 
  (A : ℝ × ℝ) 
  (h_A : A = (0, -2)) 
  (O : ℝ × ℝ) 
  (h_O : O = (0, 0)) :
  (E = {(x, y) | x^2/4 + y^2 = 1}) ∧ 
  (∃ (k : ℝ), 
    (k = Real.sqrt 7/2 ∨ k = -Real.sqrt 7/2) ∧
    (∀ (l : Set (ℝ × ℝ)), 
      l = {(x, y) | y = k*x - 2} → 
      (∀ (P Q : ℝ × ℝ), P ∈ E ∧ Q ∈ E ∧ P ∈ l ∧ Q ∈ l →
        ∀ (l' : Set (ℝ × ℝ)) (P' Q' : ℝ × ℝ), 
          P' ∈ E ∧ Q' ∈ E ∧ P' ∈ l' ∧ Q' ∈ l' →
          area_triangle O P Q ≥ area_triangle O P' Q'))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l1262_126295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_at_neg_four_thirds_l1262_126268

/-- The quadratic function f(x) = -3x^2 - 8x + 18 -/
def f (x : ℝ) := -3 * x^2 - 8 * x + 18

/-- The maximum point of f(x) -/
noncomputable def max_point : ℝ := -4/3

/-- The maximum value of f(x) -/
noncomputable def max_value : ℝ := 70/3

/-- Theorem stating that f(x) has a maximum value at x = -4/3 -/
theorem f_has_max_at_neg_four_thirds :
  f max_point = max_value ∧ ∀ x : ℝ, f x ≤ max_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_at_neg_four_thirds_l1262_126268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_take_home_pay_l1262_126285

/-- Represents the income in thousands of dollars -/
noncomputable def income : ℝ → ℝ := λ x => 1000 * x

/-- Calculates the tax rate as a percentage based on the income -/
noncomputable def taxRate : ℝ → ℝ := λ x => 2 * x

/-- Calculates the tax amount based on the income -/
noncomputable def tax : ℝ → ℝ := λ x => (taxRate x / 100) * income x

/-- Calculates the take-home pay based on the income -/
noncomputable def takeHomePay : ℝ → ℝ := λ x => income x - tax x

/-- The income that maximizes take-home pay -/
def maxTakeHomePayIncome : ℝ := 25

theorem max_take_home_pay :
  ∀ x : ℝ, takeHomePay maxTakeHomePayIncome ≥ takeHomePay x := by
  sorry

#eval maxTakeHomePayIncome

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_take_home_pay_l1262_126285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l1262_126266

-- Define the points
noncomputable def A : ℝ × ℝ := (-1/2, 0)
noncomputable def B : ℝ × ℝ := (0, 1)
noncomputable def A' : ℝ × ℝ := (1/2, 0)

-- State the theorem
theorem reflected_ray_equation : 
  -- A' is the reflection of A across the y-axis
  (A'.1 = -A.1 ∧ A'.2 = A.2) →
  -- B is on the y-axis
  B.1 = 0 →
  -- The equation of the line through B and A' is 2x + y - 1 = 0
  ∀ (x y : ℝ), (x = B.1 ∧ y = B.2) ∨ (x = A'.1 ∧ y = A'.2) → 2*x + y - 1 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l1262_126266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_count_l1262_126244

theorem book_pages_count : ℕ := by
  -- Total number of pages in the book
  let total_pages : ℕ := 240

  -- Pages read on day 1
  let day1_read (x : ℕ) : ℕ := x / 6 + 10

  -- Pages remaining after day 1
  let day1_remaining (x : ℕ) : ℕ := x - day1_read x

  -- Pages read on day 2
  let day2_read (x : ℕ) : ℕ := day1_remaining x / 5 + 20

  -- Pages remaining after day 2
  let day2_remaining (x : ℕ) : ℕ := day1_remaining x - day2_read x

  -- Pages read on day 3
  let day3_read (x : ℕ) : ℕ := day2_remaining x / 4 + 25

  -- Pages remaining after day 3
  let day3_remaining (x : ℕ) : ℕ := day2_remaining x - day3_read x

  -- Theorem statement
  have pages_count : day3_remaining total_pages = 74 := by
    sorry

  exact total_pages


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_count_l1262_126244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_radar_motion_l1262_126284

/-- Represents the motion of a car tracked by a radar --/
structure CarRadarSystem where
  v : ℝ  -- Speed of the car
  h : ℝ  -- Distance of radar from the road
  (h_pos : h > 0)  -- Assumption that h is positive

/-- Angular velocity as a function of time --/
noncomputable def angular_velocity_t (sys : CarRadarSystem) (t : ℝ) : ℝ :=
  (sys.v / sys.h) / (1 + (sys.v^2 * t^2) / sys.h^2)

/-- Angular velocity as a function of angular displacement --/
noncomputable def angular_velocity_α (sys : CarRadarSystem) (α : ℝ) : ℝ :=
  (sys.v / sys.h) * (Real.cos α)^2

/-- Angular acceleration as a function of time --/
noncomputable def angular_acceleration_t (sys : CarRadarSystem) (t : ℝ) : ℝ :=
  -2 * (sys.v / sys.h)^2 * (sys.v * t / sys.h) / (1 + (sys.v^2 * t^2 / sys.h^2))^2

/-- Angular acceleration as a function of angular displacement --/
noncomputable def angular_acceleration_α (sys : CarRadarSystem) (α : ℝ) : ℝ :=
  -2 * (sys.v / sys.h)^2 * Real.sin α * (Real.cos α)^3

theorem car_radar_motion (sys : CarRadarSystem) :
  (∀ t, angular_velocity_t sys t = (sys.v / sys.h) / (1 + (sys.v^2 * t^2) / sys.h^2)) ∧
  (∀ α, angular_velocity_α sys α = (sys.v / sys.h) * (Real.cos α)^2) ∧
  (∀ t, angular_acceleration_t sys t = -2 * (sys.v / sys.h)^2 * (sys.v * t / sys.h) / (1 + (sys.v^2 * t^2 / sys.h^2))^2) ∧
  (∀ α, angular_acceleration_α sys α = -2 * (sys.v / sys.h)^2 * Real.sin α * (Real.cos α)^3) ∧
  (∃ α_max, α_max = -π/6 ∧ ∀ α, |angular_acceleration_α sys α| ≤ |angular_acceleration_α sys α_max|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_radar_motion_l1262_126284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_area_ratio_l1262_126239

theorem hexagon_circle_area_ratio : 
  ∀ (r : ℝ), r > 0 → 
  (3 * Real.sqrt 3 / 2) * r^2 / (Real.pi * r^2) = 3 * Real.sqrt 3 / (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_area_ratio_l1262_126239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_alpha_l1262_126259

theorem tan_pi_4_plus_alpha (α : ℝ) 
  (h : Real.cos (π / 2 + α) = 2 * Real.cos (π - α)) : 
  Real.tan (π / 4 + α) = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_alpha_l1262_126259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l1262_126250

-- Define the basic shapes as propositions
def IsRectangle (x : Type) : Prop := sorry
def IsParallelogram (x : Type) : Prop := sorry
def IsTriangle (x : Type) : Prop := sorry

-- Define the premises and conclusion
axiom major_premise : ∀ x, IsRectangle x → IsParallelogram x
axiom minor_premise : ∀ x, IsTriangle x → ¬IsParallelogram x
axiom conclusion : ∀ x, IsTriangle x → ¬IsRectangle x

-- Define a syllogism
structure Syllogism :=
  (major : ∀ x, IsRectangle x → IsParallelogram x)
  (minor : ∀ x, IsTriangle x → ¬IsParallelogram x)
  (concl : ∀ x, IsTriangle x → ¬IsRectangle x)

-- Theorem to prove
theorem minor_premise_identification (s : Syllogism) : s.minor = minor_premise := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l1262_126250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_properties_l1262_126267

/-- A convex polyhedron with volume and surface area -/
structure ConvexPolyhedron (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  volume : ℝ
  surface_area : ℝ

/-- A sphere that can be inscribed in a convex polyhedron -/
structure InscribedSphere (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] extends ConvexPolyhedron α where
  radius : ℝ

/-- Two nested convex polyhedra -/
structure NestedPolyhedra (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  outer : ConvexPolyhedron α
  inner : ConvexPolyhedron α

/-- Main theorem about convex polyhedra and inscribed spheres -/
theorem convex_polyhedron_properties 
  {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α]
  (P : ConvexPolyhedron α) 
  (S : InscribedSphere α) 
  (N : NestedPolyhedra α) : 
  (P.volume / P.surface_area ≥ S.radius / 3) ∧ 
  (∃ (r : ℝ), r = P.volume / P.surface_area ∧ ∃ (S' : InscribedSphere α), S'.radius = r) ∧
  (3 * N.outer.volume / N.outer.surface_area ≥ N.inner.volume / N.inner.surface_area) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_properties_l1262_126267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_two_quadratic_polynomials_l1262_126281

/-- A set of complex numbers with specific properties -/
structure ComplexSet (n : ℕ) where
  S : Set ℂ
  card_S : Fintype S
  card_eq_n : Fintype.card S = n
  n_ge_9 : n ≥ 9
  real_count : Fintype.card {z ∈ S | z.im = 0} = n - 3

/-- A quadratic polynomial function over complex numbers -/
def QuadraticPolynomial := ℂ → ℂ

/-- The property that a quadratic polynomial maps a set to itself -/
def MapsSelfTo (f : QuadraticPolynomial) (S : Set ℂ) : Prop :=
  (∀ z ∈ S, f z ∈ S) ∧ (∀ w ∈ S, ∃ z ∈ S, f z = w)

/-- The main theorem statement -/
theorem at_most_two_quadratic_polynomials (n : ℕ) (CS : ComplexSet n) :
    ∃ (f g : QuadraticPolynomial),
      MapsSelfTo f CS.S ∧ MapsSelfTo g CS.S ∧
      ∀ (h : QuadraticPolynomial), MapsSelfTo h CS.S → h = f ∨ h = g :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_two_quadratic_polynomials_l1262_126281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1262_126231

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focal distance of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

/-- Checks if a point is on the hyperbola -/
def is_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Checks if a point is on the circle with diameter equal to focal distance -/
def is_on_focal_circle (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 + p.y^2 = (focal_distance h)^2 / 4

/-- Checks if the foot of perpendicular from a point to x-axis is midpoint of OF₂ -/
def perpendicular_foot_is_midpoint (h : Hyperbola) (p : Point) : Prop :=
  p.x = focal_distance h / 2

theorem hyperbola_eccentricity (h : Hyperbola) (m : Point)
    (h_on_hyp : is_on_hyperbola h m)
    (h_on_circle : is_on_focal_circle h m)
    (h_perp : perpendicular_foot_is_midpoint h m)
    (h_first_quadrant : m.x > 0 ∧ m.y > 0) :
    eccentricity h = Real.sqrt 3 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1262_126231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_sum_l1262_126286

/-- The original rational function -/
noncomputable def f (x : ℝ) : ℝ := (x^3 + 6*x^2 + 11*x + 6) / (x + 1)

/-- The theorem stating the existence of A, B, C, and D satisfying the required conditions -/
theorem simplification_and_sum :
  ∃ (A B C D : ℝ),
    (∀ x : ℝ, x ≠ D → f x = A*x^2 + B*x + C) ∧
    (∀ x : ℝ, x ≠ D ↔ (x + 1 ≠ 0)) ∧
    A + B + C + D = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_sum_l1262_126286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equality_condition_l1262_126298

variable (m n p q : ℝ)

def f (x : ℝ) : ℝ := m * x + n
def g (x : ℝ) : ℝ := p * x + q

theorem composition_equality_condition :
  (∀ x, f m n (g p q x) = g p q (f m n x)) ↔ n * (1 - p) = q * (1 - m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equality_condition_l1262_126298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_uniqueness_l1262_126216

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the conditions for triangle construction
def sss (a b c : ℝ) : Prop := ∃ (t : Triangle), t.a = a ∧ t.b = b ∧ t.c = c
def sas (a b γ : ℝ) : Prop := ∃ (t : Triangle), t.a = a ∧ t.b = b ∧ t.γ = γ
def asa (α b γ : ℝ) : Prop := ∃ (t : Triangle), t.α = α ∧ t.b = b ∧ t.γ = γ
def ssa (a b α : ℝ) : Prop := ∃ (t : Triangle), t.a = a ∧ t.b = b ∧ t.α = α

-- Theorem stating that SSS, SAS, and ASA uniquely determine a triangle, but SSA does not
theorem triangle_construction_uniqueness :
  (∀ a b c, sss a b c → ∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.c = c) ∧
  (∀ a b γ, sas a b γ → ∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.γ = γ) ∧
  (∀ α b γ, asa α b γ → ∃! t : Triangle, t.α = α ∧ t.b = b ∧ t.γ = γ) ∧
  ¬(∀ a b α, ssa a b α → ∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.α = α) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_uniqueness_l1262_126216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_fourth_l1262_126271

theorem tan_alpha_minus_pi_fourth (α : ℝ) 
  (h1 : Real.cos α = -4/5) 
  (h2 : α ∈ Set.Ioo (-Real.pi) 0) : 
  Real.tan (α - Real.pi/4) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_fourth_l1262_126271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_two_three_l1262_126228

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 5 / (2^x) - Real.log x / Real.log 2

-- Theorem statement
theorem root_in_interval_two_three :
  ∃ (r : ℝ), r ∈ Set.Ioo 2 3 ∧ f r = 0 :=
by
  sorry

#check root_in_interval_two_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_two_three_l1262_126228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fan_airflow_in_week_l1262_126209

/-- Calculates the total airflow created by a fan in one week -/
theorem fan_airflow_in_week 
  (airflow_rate : ℝ) 
  (operation_time_minutes : ℝ) 
  (days_in_week : ℕ) 
  (h1 : airflow_rate = 10) 
  (h2 : operation_time_minutes = 10) 
  (h3 : days_in_week = 7) : 
  airflow_rate * operation_time_minutes * 60 * (days_in_week : ℝ) = 42000 := by
  sorry

#check fan_airflow_in_week

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fan_airflow_in_week_l1262_126209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_power_of_two_l1262_126229

theorem exists_sum_power_of_two (n : ℕ) (A B : Finset ℕ) 
  (h1 : A ⊆ Finset.range (n + 1)) 
  (h2 : B ⊆ Finset.range (n + 1)) 
  (h3 : (A.card + B.card : ℕ) ≥ n + 2) :
  ∃ (a b : ℕ), a ∈ A ∧ b ∈ B ∧ ∃ (k : ℕ), a + b = 2^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_power_of_two_l1262_126229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_current_age_l1262_126201

/-- Andrew's age in years -/
def andrew_age : ℕ := sorry

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℕ := sorry

/-- Andrew's grandfather's age is fifteen times Andrew's age -/
axiom age_relation : grandfather_age = 15 * andrew_age

/-- Andrew's grandfather was 70 years old when Andrew was born -/
axiom age_difference : grandfather_age - andrew_age = 70

/-- Theorem: Andrew's current age is 5 years -/
theorem andrew_current_age : andrew_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_current_age_l1262_126201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_food_percentage_l1262_126211

-- Define the restaurant's budget allocation
noncomputable def restaurant_budget (total : ℝ) : ℝ × ℝ := 
  let rent := (1/4) * total
  let remaining := total - rent
  let food_and_beverages := (1/4) * remaining
  (rent, food_and_beverages)

-- Theorem statement
theorem restaurant_food_percentage :
  ∀ (total : ℝ), total > 0 →
  let (_, food_and_beverages) := restaurant_budget total
  (food_and_beverages / total) * 100 = 18.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_food_percentage_l1262_126211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l1262_126200

theorem sin_cos_difference (θ : ℝ) (h1 : θ ∈ Set.Ioo 0 Real.pi) (h2 : Real.sin θ + Real.cos θ = 1/5) :
  Real.sin θ - Real.cos θ = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l1262_126200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_9_range_l1262_126274

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3^(-x) else x^2

def solution_set : Set ℝ :=
  {x | f x > 9}

theorem f_greater_than_9_range :
  solution_set = Set.Ioi 3 ∪ Set.Iic (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_9_range_l1262_126274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1262_126207

theorem perpendicular_vectors (a b : ℝ × ℝ) (lambda : ℝ) : 
  a = (1, 3) → b = (3, 4) → 
  (a.1 - lambda * b.1, a.2 - lambda * b.2) • (b.1 - a.1, b.2 - a.2) = 0 → 
  lambda = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1262_126207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_problem_l1262_126282

theorem race_problem (donovan_lap_time michael_laps_to_pass : ℝ) 
  (h1 : donovan_lap_time = 48)
  (h2 : michael_laps_to_pass = 6.000000000000002) :
  ∃ michael_lap_time : ℝ,
    michael_lap_time * (1 + 1 / michael_laps_to_pass) = donovan_lap_time ∧
    abs (michael_lap_time - 41.14285714285716) < 0.00000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_problem_l1262_126282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_l1262_126277

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (x + θ) + Real.sqrt 3 * Real.cos (x + θ)

theorem theta_value (θ : ℝ) 
  (h1 : θ ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : ∀ x, f θ x = f θ (-x)) : 
  θ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_l1262_126277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersecting_radius_l1262_126225

/-- Circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Given two circles A and B, where A has center (0, 0) and radius 16,
    and B has center (0, b) and radius |a|, the maximum possible radius
    of their intersecting region is 16. -/
theorem max_intersecting_radius (a b : ℝ) : 
  let circle_a : Circle := { center := (0, 0), radius := 16 }
  let circle_b : Circle := { center := (0, b), radius := |a| }
  ∃ (r : ℝ), r ≤ 16 ∧ 
    ∀ (r' : ℝ), (∃ (x y : ℝ), (x - 0)^2 + (y - 0)^2 ≤ 16^2 ∧ 
                               (x - 0)^2 + (y - b)^2 ≤ a^2) → 
                r' ≤ r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersecting_radius_l1262_126225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l1262_126270

theorem count_integers_satisfying_inequality : 
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ |3*x + 4| ≤ 10) ∧ S.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l1262_126270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_one_is_local_minimum_l1262_126245

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- State the theorem
theorem negative_one_is_local_minimum : 
  ∃ δ > 0, ∀ x : ℝ, |x - (-1)| < δ → f x ≥ f (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_one_is_local_minimum_l1262_126245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_pace_cycling_l1262_126294

/-- Represents the cycling speed in miles per minute -/
noncomputable def cycling_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Proves that given a constant cycling pace where 3 miles takes 24 minutes, 
    cycling 4 miles will take 32 minutes -/
theorem constant_pace_cycling 
  (d₁ : ℝ) (t₁ : ℝ) (d₂ : ℝ) (t₂ : ℝ)
  (h₁ : d₁ = 3) (h₂ : t₁ = 24) (h₃ : d₂ = 4)
  (h₄ : cycling_speed d₁ t₁ = cycling_speed d₂ t₂) :
  t₂ = 32 := by
  sorry

#check constant_pace_cycling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_pace_cycling_l1262_126294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_13_l1262_126205

theorem sum_remainder_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_13_l1262_126205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_111_11_is_7_75_l1262_126273

/-- Represents a binary digit (0 or 1) -/
inductive BinaryDigit
| zero : BinaryDigit
| one : BinaryDigit

/-- Converts a BinaryDigit to its decimal value -/
def binaryDigitToDecimal (d : BinaryDigit) : ℚ :=
  match d with
  | BinaryDigit.zero => 0
  | BinaryDigit.one => 1

/-- Represents a binary number with 3 digits before the decimal point and 2 after -/
structure BinaryNumber :=
  (d2 d1 d0 d_1 d_2 : BinaryDigit)

/-- Converts a BinaryNumber to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : ℚ :=
  (binaryDigitToDecimal b.d2) * 2^2 +
  (binaryDigitToDecimal b.d1) * 2^1 +
  (binaryDigitToDecimal b.d0) * 2^0 +
  (binaryDigitToDecimal b.d_1) * 2^(-1 : ℤ) +
  (binaryDigitToDecimal b.d_2) * 2^(-2 : ℤ)

theorem binary_111_11_is_7_75 :
  binaryToDecimal { d2 := BinaryDigit.one,
                    d1 := BinaryDigit.one,
                    d0 := BinaryDigit.one,
                    d_1 := BinaryDigit.one,
                    d_2 := BinaryDigit.one } = 31/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_111_11_is_7_75_l1262_126273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equipment_purchase_solution_l1262_126242

/-- Represents the purchase problem for school equipment --/
structure EquipmentPurchase where
  price_ratio : ℝ  -- Price ratio of A to B
  budget_a : ℝ     -- Budget for A-type equipment
  budget_b : ℝ     -- Budget for B-type equipment
  extra_units : ℕ  -- Extra units of A that can be bought compared to B
  total_units : ℕ  -- Total units to be purchased
  min_ratio : ℝ    -- Minimum ratio of A to B units

/-- Theorem stating the solution to the equipment purchase problem --/
theorem equipment_purchase_solution (ep : EquipmentPurchase)
  (h_price_ratio : ep.price_ratio = 1.2)
  (h_budget_a : ep.budget_a = 30000)
  (h_budget_b : ep.budget_b = 15000)
  (h_extra_units : ep.extra_units = 4)
  (h_total_units : ep.total_units = 50)
  (h_min_ratio : ep.min_ratio = 1/3) :
  ∃ (price_b price_a : ℝ) (cost_function : ℝ → ℝ) (min_cost : ℝ),
    price_b = 2500 ∧
    price_a = 3000 ∧
    (∀ a : ℝ, cost_function a = 500 * a + 125000) ∧
    min_cost = 131500 ∧
    (∀ a : ℝ, a ≥ ep.min_ratio * (ep.total_units - a) → cost_function a ≥ min_cost) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equipment_purchase_solution_l1262_126242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_power_of_seven_l1262_126212

def number_of_prime_factors (n : ℕ) : ℕ :=
  sorry

theorem prime_factors_power_of_seven (x : ℕ) : 
  (∀ n : ℕ, n = 4^11 * 7^x * 11^2 → (number_of_prime_factors n) = 29) → x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_power_of_seven_l1262_126212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_theorem_l1262_126240

/-- The line l: 3x - 4y + 1 = 0 -/
def l (x y : ℝ) : ℝ := 3 * x - 4 * y + 1

/-- Distance from a point (x, y) to the line l -/
noncomputable def dist_to_l (x y : ℝ) : ℝ := 
  |l x y| / Real.sqrt (3^2 + (-4)^2)

/-- The x-coordinate of point A -/
def ax : ℝ := -2

/-- The y-coordinate of point A -/
def ay : ℝ := 0

/-- The x-coordinate of point B -/
def bx : ℝ := 4

theorem equidistant_points_theorem (a : ℝ) :
  dist_to_l ax ay = dist_to_l bx a → a = 2 ∨ a = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_theorem_l1262_126240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1262_126227

noncomputable def f (x : ℝ) : ℝ := (x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1) / (x^3 - 9*x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x < -3 ∨ (-3 < x ∧ x < 0) ∨ (0 < x ∧ x < 3) ∨ 3 < x} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1262_126227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_l1262_126206

/-- Two lines in 2D space -/
structure Line2D where
  origin : ℝ × ℝ
  direction : ℝ × ℝ

/-- A point lies on a line if it can be represented by the line's parameterization -/
def onLine (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p = (l.origin.1 + t * l.direction.1, l.origin.2 + t * l.direction.2)

/-- The intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line2D) : ℝ × ℝ :=
  (2/13, 69/13)

theorem line_intersection :
  let l1 : Line2D := ⟨(2, -3), (-1, 4)⟩
  let l2 : Line2D := ⟨(-1, 6), (5, -7)⟩
  let p : ℝ × ℝ := intersectionPoint l1 l2
  (onLine p l1 ∧ onLine p l2) ∧
  ∀ q : ℝ × ℝ, (onLine q l1 ∧ onLine q l2) → q = p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_l1262_126206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_line_l1262_126224

/-- The distance from a point (x,y) to the line ax + y + 1 = 0 -/
noncomputable def distToLine (x y a : ℝ) : ℝ :=
  abs (a * x + y + 1) / Real.sqrt (a^2 + 1)

/-- Theorem: If points A(-3,-4) and B(6,3) are equidistant from the line ax + y + 1 = 0,
    then a = -1/3 or a = -7/9 -/
theorem equidistant_points_line (a : ℝ) :
  distToLine (-3) (-4) a = distToLine 6 3 a →
  a = -1/3 ∨ a = -7/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_line_l1262_126224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l1262_126252

/-- The function f(x) defined as cos(√3x + φ) -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ)

/-- The theorem stating that if f(x) + f'(x) is odd and 0 < φ < π, then φ = π/6 -/
theorem phi_value (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi) :
  (∀ x, f φ x + (deriv (f φ)) x = -(f φ (-x) + (deriv (f φ)) (-x))) →
  φ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l1262_126252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l1262_126218

/-- An arithmetic sequence with sum property -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  sum_property : a 1 + a 3 + a 5 = 3
  arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- Theorem: If a₁ + a₃ + a₅ = 3 for an arithmetic sequence, then S₅ = 5 -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) : S seq 5 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l1262_126218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l1262_126235

theorem cubic_root_sum (m n p : ℕ) (h : m > 0 ∧ n > 0 ∧ p > 0) : 
  (∃ x : ℝ, 27 * x^3 - 12 * x^2 - 12 * x - 4 = 0 ∧ 
   x = (m^(1/3 : ℝ) + n^(1/3 : ℝ) + 2) / p) → 
  m + n + p = 90 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l1262_126235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_pair_properties_l1262_126278

/-- Two vectors in a plane with given magnitudes and angle between them -/
structure VectorPair where
  a : ℝ × ℝ
  b : ℝ × ℝ
  angle : ℝ
  mag_a : ℝ
  mag_b : ℝ
  h_angle : angle = 2 * Real.pi / 3
  h_mag_a : mag_a = 4
  h_mag_b : mag_b = 2

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem vector_pair_properties (vp : VectorPair) :
  let a := vp.a
  let b := vp.b
  (dot_product (a.1 - 2 * b.1, a.2 - 2 * b.2) (a.1 + b.1, a.2 + b.2) = 12) ∧
  (magnitude (3 * a.1 - 4 * b.1, 3 * a.2 - 4 * b.2) = 4 * Real.sqrt 19) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_pair_properties_l1262_126278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l1262_126269

def my_sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 4) ∧
  (a 4 = 40) ∧
  ∀ n > 1, a n = (1 : ℚ) / 4 * (a (n-1) + a (n+1))

theorem fifth_term_value (a : ℕ → ℚ) (h : my_sequence a) : a 5 = 2236 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l1262_126269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painter_work_days_l1262_126222

/-- The number of work-days required for a given number of painters to complete a job,
    assuming all painters work at the same rate. -/
noncomputable def work_days (painters : ℝ) (initial_painters : ℝ) (initial_days : ℝ) : ℝ :=
  (initial_painters * initial_days) / painters

theorem painter_work_days :
  let initial_painters : ℝ := 8
  let initial_days : ℝ := 0.75
  let new_painters : ℝ := 5
  work_days new_painters initial_painters initial_days = 1.2 := by
  -- Unfold the definitions
  unfold work_days
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painter_work_days_l1262_126222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heartsuit_sum_l1262_126264

noncomputable def heartsuit (x : ℝ) : ℝ := (x + x^3) / 2

theorem heartsuit_sum : heartsuit 1 + heartsuit (-1) + heartsuit 2 = 5 := by
  -- Expand the definition of heartsuit
  unfold heartsuit
  -- Simplify the expressions
  simp
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heartsuit_sum_l1262_126264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_sum_exists_l1262_126213

theorem zero_sum_exists (m : ℕ) (S : Finset ℤ) :
  S.card = 2 * m + 1 →
  (∀ x : ℤ, x ∈ S → |x| ≤ 2 * m - 1) →
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_sum_exists_l1262_126213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_property_l1262_126223

-- Define a triangle ABC in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define the squared distance between two points
def sqDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Theorem statement
theorem centroid_property (t : Triangle) (P : ℝ × ℝ) :
  let G := centroid t
  sqDistance P t.A + sqDistance P t.B + sqDistance P t.C =
  3 * sqDistance P G + (sqDistance G t.A + sqDistance G t.B + sqDistance G t.C) / 2 := by
  sorry

#check centroid_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_property_l1262_126223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_circle_with_three_points_l1262_126237

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- The unit square -/
def UnitSquare : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- A set of 51 points in the unit square -/
def RandomPoints : Set Point :=
  {p : Point | p ∈ UnitSquare ∧ (∃ (S : Finset Point), ↑S ⊆ UnitSquare ∧ S.card = 51 ∧ p ∈ S)}

/-- A point is inside a circle -/
def InsideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- The main theorem -/
theorem existence_of_circle_with_three_points :
  ∃ (c : Circle), c.radius = 1/7 ∧ (∃ (S : Finset Point), ↑S ⊆ RandomPoints ∧ S.card ≥ 3 ∧ ∀ p ∈ S, InsideCircle p c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_circle_with_three_points_l1262_126237
