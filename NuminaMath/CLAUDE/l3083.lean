import Mathlib

namespace objective_function_range_l3083_308393

-- Define the constraint set
def ConstraintSet (x y : ℝ) : Prop :=
  x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ -1

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ := 3*x - y

-- State the theorem
theorem objective_function_range :
  ∀ x y : ℝ, ConstraintSet x y →
  ∃ z_min z_max : ℝ, z_min = -3/2 ∧ z_max = 6 ∧
  z_min ≤ ObjectiveFunction x y ∧ ObjectiveFunction x y ≤ z_max :=
sorry

end objective_function_range_l3083_308393


namespace max_xy_value_l3083_308345

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 168 := by
  sorry

end max_xy_value_l3083_308345


namespace min_value_2a_plus_b_l3083_308300

theorem min_value_2a_plus_b (a b : ℝ) (h : Real.log a + Real.log b = Real.log (a + 2*b)) :
  (∀ x y : ℝ, Real.log x + Real.log y = Real.log (x + 2*y) → 2*x + y ≥ 2*a + b) ∧ (∃ x y : ℝ, Real.log x + Real.log y = Real.log (x + 2*y) ∧ 2*x + y = 9) :=
by sorry

end min_value_2a_plus_b_l3083_308300


namespace function_properties_l3083_308378

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (2 * ω * x - Real.pi / 6) + 2 * (Real.cos (ω * x))^2 - 1

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x, f ω (x + Real.pi / ω) = f ω x) : 
  ω = 1 ∧ 
  (∀ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x ≥ -Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x = 1) ∧
  (∃ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x = -Real.sqrt 3 / 2) :=
by sorry

end function_properties_l3083_308378


namespace vitamin_a_weekly_pills_l3083_308386

/-- Calculates the number of pills needed for a week's supply of Vitamin A -/
def weekly_vitamin_pills (daily_recommended : ℕ) (mg_per_pill : ℕ) : ℕ :=
  (daily_recommended / mg_per_pill) * 7

/-- Theorem stating that 28 pills are needed for a week's supply of Vitamin A -/
theorem vitamin_a_weekly_pills :
  weekly_vitamin_pills 200 50 = 28 := by
  sorry

#eval weekly_vitamin_pills 200 50

end vitamin_a_weekly_pills_l3083_308386


namespace abc_fraction_value_l3083_308392

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 1 / 3)
  (h2 : b * c / (b + c) = 1 / 4)
  (h3 : c * a / (c + a) = 1 / 5) :
  a * b * c / (a * b + b * c + c * a) = 1 / 6 := by
  sorry

end abc_fraction_value_l3083_308392


namespace sarah_scored_135_l3083_308348

def sarahs_score (greg_score sarah_score : ℕ) : Prop :=
  greg_score + 50 = sarah_score ∧ (greg_score + sarah_score) / 2 = 110

theorem sarah_scored_135 :
  ∃ (greg_score : ℕ), sarahs_score greg_score 135 :=
by sorry

end sarah_scored_135_l3083_308348


namespace distance_before_meeting_l3083_308383

/-- The distance between two boats one minute before they meet -/
theorem distance_before_meeting (v1 v2 d : ℝ) (hv1 : v1 = 4) (hv2 : v2 = 20) (hd : d = 20) :
  let t := d / (v1 + v2)  -- Time to meet
  let distance_per_minute := (v1 + v2) / 60
  (t - 1/60) * (v1 + v2) = 0.4
  := by sorry

end distance_before_meeting_l3083_308383


namespace line_curve_hyperbola_l3083_308357

variable (a b : ℝ)

theorem line_curve_hyperbola (h1 : a ≠ 0) (h2 : b ≠ 0) :
  ∃ (x y : ℝ), (a * x - y + b = 0) ∧ (b * x^2 + a * y^2 = a * b) →
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧ ∀ (x y : ℝ), x^2 / A - y^2 / B = 1 :=
sorry

end line_curve_hyperbola_l3083_308357


namespace envelope_count_l3083_308390

/-- Proves that the number of envelopes sent is 850, given the weight of one envelope and the total weight. -/
theorem envelope_count (envelope_weight : ℝ) (total_weight_kg : ℝ) : 
  envelope_weight = 8.5 →
  total_weight_kg = 7.225 →
  (total_weight_kg * 1000) / envelope_weight = 850 := by
  sorry

end envelope_count_l3083_308390


namespace derivative_positive_implies_increasing_l3083_308307

open Set

theorem derivative_positive_implies_increasing
  {f : ℝ → ℝ} {I : Set ℝ} (hI : IsOpen I) (hf : DifferentiableOn ℝ f I)
  (h : ∀ x ∈ I, deriv f x > 0) :
  StrictMonoOn f I :=
sorry

end derivative_positive_implies_increasing_l3083_308307


namespace f_g_2_equals_22_l3083_308377

-- Define the functions g and f
def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x - 2

-- State the theorem
theorem f_g_2_equals_22 : f (g 2) = 22 := by
  sorry

end f_g_2_equals_22_l3083_308377


namespace perpendicular_planes_theorem_l3083_308344

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpLine : Line → Line → Prop)
variable (perpLinePlane : Line → Plane → Prop)
variable (perpPlane : Plane → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the theorem
theorem perpendicular_planes_theorem 
  (a b : Line) (α β : Plane) : 
  (∀ (l1 l2 : Line) (p1 p2 : Plane), 
    (perpLine l1 l2 ∧ perpLinePlane l1 p1 → ¬(parallelLinePlane l2 p1)) ∧
    (perpPlane p1 p2 ∧ parallelLinePlane l1 p1 → ¬(perpLinePlane l1 p2)) ∧
    (perpLinePlane l1 p2 ∧ perpPlane p1 p2 → ¬(parallelLinePlane l1 p1))) →
  (perpLine a b ∧ perpLinePlane a α ∧ perpLinePlane b β → perpPlane α β) :=
sorry

end perpendicular_planes_theorem_l3083_308344


namespace train_arrival_interval_l3083_308367

def train_interval (passengers_per_hour : ℕ) (passengers_left : ℕ) (passengers_taken : ℕ) : ℕ :=
  60 / (passengers_per_hour / (passengers_left + passengers_taken))

theorem train_arrival_interval :
  train_interval 6240 200 320 = 5 := by
  sorry

end train_arrival_interval_l3083_308367


namespace age_ratio_proof_l3083_308306

/-- Proves that given the conditions about A's and B's ages, the ratio between A's age 4 years hence and B's age 4 years ago is 3:1 -/
theorem age_ratio_proof (x : ℕ) (h1 : 5 * x > 4) (h2 : 3 * x > 4) : 
  (5 * x + 4) / (3 * x - 4) = 3 := by
  sorry

end age_ratio_proof_l3083_308306


namespace cats_asleep_l3083_308313

theorem cats_asleep (total : ℕ) (awake : ℕ) (h1 : total = 98) (h2 : awake = 6) :
  total - awake = 92 := by
  sorry

end cats_asleep_l3083_308313


namespace compote_level_reduction_l3083_308363

theorem compote_level_reduction (V : ℝ) (h : V > 0) :
  let initial_level := V
  let level_after_third := 3/4 * V
  let volume_of_remaining_peaches := 1/6 * V
  let final_level := level_after_third - volume_of_remaining_peaches
  (level_after_third - final_level) / level_after_third = 2/9 := by
  sorry

end compote_level_reduction_l3083_308363


namespace least_months_to_double_amount_l3083_308376

/-- The amount owed after t months -/
def amount_owed (initial_amount : ℝ) (interest_rate : ℝ) (t : ℕ) : ℝ :=
  initial_amount * (1 + interest_rate) ^ t

/-- The theorem stating that 25 is the least number of months to double the borrowed amount -/
theorem least_months_to_double_amount : 
  let initial_amount : ℝ := 1500
  let interest_rate : ℝ := 0.03
  let double_amount : ℝ := 2 * initial_amount
  ∀ t : ℕ, t < 25 → amount_owed initial_amount interest_rate t ≤ double_amount ∧
  amount_owed initial_amount interest_rate 25 > double_amount :=
by sorry

end least_months_to_double_amount_l3083_308376


namespace garrick_nickels_count_l3083_308371

/-- The number of cents in a dime -/
def dime_value : ℕ := 10

/-- The number of cents in a quarter -/
def quarter_value : ℕ := 25

/-- The number of cents in a nickel -/
def nickel_value : ℕ := 5

/-- The number of cents in a penny -/
def penny_value : ℕ := 1

/-- The number of dimes Cindy tossed -/
def cindy_dimes : ℕ := 5

/-- The number of quarters Eric flipped -/
def eric_quarters : ℕ := 3

/-- The number of pennies Ivy dropped -/
def ivy_pennies : ℕ := 60

/-- The total amount of money in the pond in cents -/
def total_cents : ℕ := 200

/-- The number of nickels Garrick threw into the pond -/
def garrick_nickels : ℕ := (total_cents - (cindy_dimes * dime_value + eric_quarters * quarter_value + ivy_pennies * penny_value)) / nickel_value

theorem garrick_nickels_count : garrick_nickels = 3 := by
  sorry

end garrick_nickels_count_l3083_308371


namespace trig_identity_l3083_308373

/-- For any angle α, sin²(α) + cos²(30° + α) + sin(α)cos(30° + α) = 3/4 -/
theorem trig_identity (α : Real) : 
  (Real.sin α)^2 + (Real.cos (π/6 + α))^2 + (Real.sin α) * (Real.cos (π/6 + α)) = 3/4 := by
  sorry

end trig_identity_l3083_308373


namespace heat_of_neutralization_instruments_l3083_308399

-- Define the set of available instruments
inductive Instrument
  | Balance
  | MeasuringCylinder
  | Beaker
  | Burette
  | Thermometer
  | TestTube
  | AlcoholLamp

-- Define the requirements for the heat of neutralization experiment
structure ExperimentRequirements where
  needsWeighing : Bool
  needsHeating : Bool
  reactionContainer : Instrument
  volumeMeasurementTool : Instrument
  temperatureMeasurementTool : Instrument

-- Define the correct set of instruments
def correctInstruments : Set Instrument :=
  {Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer}

-- Define the heat of neutralization experiment requirements
def heatOfNeutralizationRequirements : ExperimentRequirements :=
  { needsWeighing := false
  , needsHeating := false
  , reactionContainer := Instrument.Beaker
  , volumeMeasurementTool := Instrument.MeasuringCylinder
  , temperatureMeasurementTool := Instrument.Thermometer
  }

-- Theorem statement
theorem heat_of_neutralization_instruments :
  correctInstruments = 
    { i : Instrument | i = heatOfNeutralizationRequirements.volumeMeasurementTool ∨
                       i = heatOfNeutralizationRequirements.reactionContainer ∨
                       i = heatOfNeutralizationRequirements.temperatureMeasurementTool } :=
by sorry

end heat_of_neutralization_instruments_l3083_308399


namespace election_votes_l3083_308302

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (excess_percent : ℚ) : 
  total_votes = 5720 →
  invalid_percent = 1/5 →
  excess_percent = 3/20 →
  ∃ (a_votes b_votes : ℕ),
    (a_votes : ℚ) + b_votes = total_votes * (1 - invalid_percent) ∧
    (a_votes : ℚ) = b_votes + total_votes * excess_percent ∧
    b_votes = 1859 := by
sorry

end election_votes_l3083_308302


namespace door_height_is_eight_l3083_308391

/-- Represents the dimensions of a rectangular door and a pole. -/
structure DoorPole where
  pole_length : ℝ
  door_width : ℝ
  door_height : ℝ
  door_diagonal : ℝ

/-- The conditions of the door and pole problem. -/
def door_pole_conditions (d : DoorPole) : Prop :=
  d.pole_length = d.door_width + 4 ∧
  d.pole_length = d.door_height + 2 ∧
  d.pole_length = d.door_diagonal ∧
  d.door_diagonal^2 = d.door_width^2 + d.door_height^2

/-- The theorem stating that under the given conditions, the door height is 8 feet. -/
theorem door_height_is_eight (d : DoorPole) 
  (h : door_pole_conditions d) : d.door_height = 8 := by
  sorry

end door_height_is_eight_l3083_308391


namespace flight_duration_sum_l3083_308382

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes, accounting for day change -/
def timeDiffMinutes (t1 t2 : Time) : ℕ :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  if totalMinutes2 < totalMinutes1 then
    (24 * 60 - totalMinutes1) + totalMinutes2
  else
    totalMinutes2 - totalMinutes1

theorem flight_duration_sum (departure : Time) (arrival : Time) (h m : ℕ) :
  departure.hours = 17 ∧ departure.minutes = 30 ∧
  arrival.hours = 2 ∧ arrival.minutes = 15 ∧
  0 < m ∧ m < 60 ∧
  timeDiffMinutes departure arrival + 3 * 60 = h * 60 + m →
  h + m = 56 := by
  sorry

end flight_duration_sum_l3083_308382


namespace ten_by_ten_grid_triangles_l3083_308331

/-- Represents a square grid -/
structure SquareGrid :=
  (size : ℕ)

/-- Counts the number of triangles formed by drawing a diagonal in a square grid -/
def countTriangles (grid : SquareGrid) : ℕ :=
  (grid.size + 1) * (grid.size + 1) - (grid.size + 1)

/-- Theorem: In a 10 × 10 square grid with one diagonal drawn, 110 triangles are formed -/
theorem ten_by_ten_grid_triangles :
  countTriangles { size := 10 } = 110 := by
  sorry

end ten_by_ten_grid_triangles_l3083_308331


namespace second_bill_overdue_months_l3083_308317

/-- Calculates the number of months a bill is overdue given the total amount owed and the conditions of three bills -/
def months_overdue (total_owed : ℚ) (bill1_amount : ℚ) (bill1_interest_rate : ℚ) (bill1_months : ℕ)
                   (bill2_amount : ℚ) (bill2_fee : ℚ)
                   (bill3_fee1 : ℚ) (bill3_fee2 : ℚ) : ℕ :=
  let bill1_total := bill1_amount + bill1_amount * bill1_interest_rate * bill1_months
  let bill3_total := bill3_fee1 + bill3_fee2
  let bill2_overdue := total_owed - bill1_total - bill3_total
  Nat.ceil (bill2_overdue / bill2_fee)

/-- The number of months the second bill is overdue is 18 -/
theorem second_bill_overdue_months :
  months_overdue 1234 200 (1/10) 2 130 50 40 80 = 18 := by
  sorry

end second_bill_overdue_months_l3083_308317


namespace charity_amount_l3083_308309

/-- The amount of money raised from the rubber duck race -/
def money_raised (small_price medium_price large_price : ℚ) 
  (small_qty medium_qty large_qty : ℕ) : ℚ :=
  small_price * small_qty + medium_price * medium_qty + large_price * large_qty

/-- Theorem stating the total amount raised for charity -/
theorem charity_amount : 
  money_raised 2 3 5 150 221 185 = 1888 := by sorry

end charity_amount_l3083_308309


namespace modular_inverse_5_mod_23_l3083_308319

theorem modular_inverse_5_mod_23 : ∃ (a : ℤ), 5 * a ≡ 1 [ZMOD 23] ∧ a = 14 := by
  sorry

end modular_inverse_5_mod_23_l3083_308319


namespace floor_coverage_l3083_308387

/-- A type representing a rectangular floor -/
structure RectangularFloor where
  m : ℕ
  n : ℕ
  h_m : m > 3
  h_n : n > 3

/-- A predicate that determines if a floor can be fully covered by 2x4 tiles -/
def canBeCovered (floor : RectangularFloor) : Prop :=
  floor.m % 2 = 0 ∧ floor.n % 2 = 0

/-- Theorem stating that a rectangular floor can be fully covered by 2x4 tiles 
    if and only if both dimensions are even -/
theorem floor_coverage (floor : RectangularFloor) :
  canBeCovered floor ↔ (floor.m % 2 = 0 ∧ floor.n % 2 = 0) := by
  sorry

#check floor_coverage

end floor_coverage_l3083_308387


namespace profit_increase_l3083_308370

theorem profit_increase (profit_1995 : ℝ) : 
  let profit_1996 := profit_1995 * 1.1
  let profit_1997 := profit_1995 * 1.3200000000000001
  (profit_1997 / profit_1996 - 1) * 100 = 20 := by sorry

end profit_increase_l3083_308370


namespace sphere_only_orientation_independent_l3083_308323

-- Define the types of 3D objects we're considering
inductive Object3D
  | Cube
  | RegularTetrahedron
  | RegularTriangularPyramid
  | Sphere

-- Define a function that determines if an object's orthographic projections are orientation-independent
def hasOrientationIndependentProjections (obj : Object3D) : Prop :=
  match obj with
  | Object3D.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_only_orientation_independent :
  ∀ (obj : Object3D), hasOrientationIndependentProjections obj ↔ obj = Object3D.Sphere :=
by sorry

end sphere_only_orientation_independent_l3083_308323


namespace quadratic_root_condition_l3083_308341

/-- 
If the quadratic function f(x) = -x^2 - 2x + m has a root, 
then m is greater than or equal to 1.
-/
theorem quadratic_root_condition (m : ℝ) : 
  (∃ x, -x^2 - 2*x + m = 0) → m ≥ 1 := by
  sorry

end quadratic_root_condition_l3083_308341


namespace fan_work_time_theorem_l3083_308333

/-- Represents the fan's properties and operation --/
structure Fan where
  airflow_rate : ℝ  -- liters per second
  work_time : ℝ     -- minutes per day
  total_airflow : ℝ -- liters per week

/-- Theorem stating the relationship between fan operation and total airflow --/
theorem fan_work_time_theorem (f : Fan) (h1 : f.airflow_rate = 10) 
  (h2 : f.total_airflow = 42000) : 
  f.work_time = 10 ↔ f.total_airflow = 7 * f.work_time * 60 * f.airflow_rate := by
  sorry

#check fan_work_time_theorem

end fan_work_time_theorem_l3083_308333


namespace sum_of_squares_zero_implies_both_zero_l3083_308389

theorem sum_of_squares_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end sum_of_squares_zero_implies_both_zero_l3083_308389


namespace quadratic_roots_sum_inverse_squares_l3083_308339

theorem quadratic_roots_sum_inverse_squares (a b c k : ℝ) (kr ks : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : a * kr^2 + k * c * kr + b = 0) 
  (h4 : a * ks^2 + k * c * ks + b = 0) 
  (h5 : kr ≠ 0) (h6 : ks ≠ 0) : 
  1 / kr^2 + 1 / ks^2 = (k^2 * c^2 - 2 * a * b) / b^2 := by
  sorry

end quadratic_roots_sum_inverse_squares_l3083_308339


namespace exists_function_satisfying_condition_l3083_308361

theorem exists_function_satisfying_condition : ∃ f : ℕ → ℕ, 
  (∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1) ∧ 
  f 2019 = 2019 := by
  sorry

end exists_function_satisfying_condition_l3083_308361


namespace rectangle_areas_sum_l3083_308380

theorem rectangle_areas_sum : 
  let width := 3
  let lengths := [1, 8, 27, 64, 125, 216]
  let areas := lengths.map (λ l => width * l)
  areas.sum = 1323 := by
  sorry

end rectangle_areas_sum_l3083_308380


namespace cube_surface_area_l3083_308374

/-- The surface area of a cube with edge length 2a cm is 24a² cm² -/
theorem cube_surface_area (a : ℝ) (h : a > 0) : 
  6 * (2 * a) ^ 2 = 24 * a ^ 2 := by
  sorry

#check cube_surface_area

end cube_surface_area_l3083_308374


namespace filter_kit_price_calculation_l3083_308324

/-- The price of a camera lens filter kit -/
def filter_kit_price (price1 price2 price3 : ℝ) (discount : ℝ) : ℝ :=
  let total_individual := 2 * price1 + 2 * price2 + price3
  total_individual * (1 - discount)

/-- Theorem stating the price of the filter kit -/
theorem filter_kit_price_calculation :
  filter_kit_price 16.45 14.05 19.50 0.08 = 74.06 := by
  sorry

end filter_kit_price_calculation_l3083_308324


namespace range_of_x_l3083_308316

theorem range_of_x (x : ℝ) : 2 * x + 1 ≤ 0 → x ≤ -1/2 := by
  sorry

end range_of_x_l3083_308316


namespace cloth_square_cutting_l3083_308356

/-- Proves that a 29 cm by 40 cm cloth can be cut into at most 280 squares of 4 square centimeters each. -/
theorem cloth_square_cutting (cloth_width : ℕ) (cloth_length : ℕ) 
  (square_area : ℕ) (max_squares : ℕ) : 
  cloth_width = 29 → 
  cloth_length = 40 → 
  square_area = 4 → 
  max_squares = 280 → 
  (cloth_width / 2) * (cloth_length / 2) ≤ max_squares :=
by
  sorry

#check cloth_square_cutting

end cloth_square_cutting_l3083_308356


namespace intersection_sum_l3083_308347

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 3*x + 1
def g (x y : ℝ) : Prop := x + 3*y = 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = p.2 ∧ g p.1 p.2}

-- Theorem statement
theorem intersection_sum :
  ∃ (p₁ p₂ p₃ : ℝ × ℝ),
    p₁ ∈ intersection_points ∧
    p₂ ∈ intersection_points ∧
    p₃ ∈ intersection_points ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    p₁.1 + p₂.1 + p₃.1 = 0 ∧
    p₁.2 + p₂.2 + p₃.2 = 3 :=
sorry

end intersection_sum_l3083_308347


namespace stones_for_hall_l3083_308328

/-- Calculates the number of stones required to pave a rectangular hall --/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  let hall_area := hall_length * hall_width * 100
  let stone_area := stone_length * stone_width
  (hall_area / stone_area).ceil.toNat

/-- Theorem stating that 4,500 stones are required to pave the given hall --/
theorem stones_for_hall : stones_required 72 30 6 8 = 4500 := by
  sorry

end stones_for_hall_l3083_308328


namespace second_train_speed_l3083_308394

/-- Given two trains starting from the same station, traveling in the same direction
    on parallel tracks for 8 hours, with one train moving at 11 mph and ending up
    160 miles behind the other train, prove that the speed of the second train is 31 mph. -/
theorem second_train_speed (v : ℝ) : 
  v > 0 → -- The speed of the second train is positive
  (v * 8 - 11 * 8 = 160) → -- Distance difference after 8 hours
  v = 31 :=
by sorry

end second_train_speed_l3083_308394


namespace fraction_transformation_l3083_308342

theorem fraction_transformation (x : ℝ) (h : x ≠ 1) : -1 / (1 - x) = 1 / (x - 1) := by
  sorry

end fraction_transformation_l3083_308342


namespace unique_integer_triples_l3083_308366

theorem unique_integer_triples : 
  {(a, b, c) : ℕ × ℕ × ℕ | 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≤ b ∧ b ≤ c ∧
    a + b + c + a*b + b*c + c*a = a*b*c + 1} 
  = {(2, 5, 8), (3, 4, 13)} := by sorry

end unique_integer_triples_l3083_308366


namespace cube_difference_l3083_308369

theorem cube_difference (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 - n^2 = 43) :
  m^3 - n^3 = 1387 := by
  sorry

end cube_difference_l3083_308369


namespace inscribed_sphere_surface_area_l3083_308304

/-- The surface area of a sphere inscribed in a cube with edge length 2 is 4π. -/
theorem inscribed_sphere_surface_area (cube_edge : ℝ) (h : cube_edge = 2) :
  let sphere_radius := cube_edge / 2
  4 * Real.pi * sphere_radius ^ 2 = 4 * Real.pi := by
  sorry

end inscribed_sphere_surface_area_l3083_308304


namespace unique_number_with_sum_of_largest_divisors_3333_l3083_308360

/-- The largest divisor of a natural number is the number itself -/
def largest_divisor (n : ℕ) : ℕ := n

/-- The second largest divisor of an even natural number is half of the number -/
def second_largest_divisor (n : ℕ) : ℕ := n / 2

/-- The property that the sum of the two largest divisors of n is 3333 -/
def sum_of_largest_divisors_is_3333 (n : ℕ) : Prop :=
  largest_divisor n + second_largest_divisor n = 3333

theorem unique_number_with_sum_of_largest_divisors_3333 :
  ∀ n : ℕ, sum_of_largest_divisors_is_3333 n → n = 2222 :=
by sorry

end unique_number_with_sum_of_largest_divisors_3333_l3083_308360


namespace square_and_cube_roots_l3083_308375

-- Define square root
def is_square_root (x y : ℝ) : Prop := y^2 = x

-- Define cube root
def is_cube_root (x y : ℝ) : Prop := y^3 = x

-- Define self square root
def is_self_square_root (x : ℝ) : Prop := x^2 = x

theorem square_and_cube_roots :
  (∃ y : ℝ, y < 0 ∧ is_square_root 2 y) ∧
  (is_cube_root (-1) (-1)) ∧
  (is_square_root 100 10) ∧
  (∀ x : ℝ, is_self_square_root x ↔ (x = 0 ∨ x = 1)) :=
by sorry

end square_and_cube_roots_l3083_308375


namespace simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l3083_308362

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := 2 * x^2 - 3 * x - y + x * y

-- Theorem 1: Simplification of 2A - 3B
theorem simplify_2A_minus_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7 * x + 7 * y - 11 * x * y :=
by sorry

-- Theorem 2: Value of 2A - 3B under specific conditions
theorem value_2A_minus_3B_specific (x y : ℝ) 
  (h1 : x + y = 6/7) (h2 : x * y = -1) :
  2 * A x y - 3 * B x y = 17 :=
by sorry

-- Theorem 3: Value of 2A - 3B when independent of y
theorem value_2A_minus_3B_independent (x : ℝ) 
  (h : ∀ y : ℝ, 2 * A x y - 3 * B x y = 2 * A x 0 - 3 * B x 0) :
  2 * A x 0 - 3 * B x 0 = 49/11 :=
by sorry

end simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l3083_308362


namespace frog_jump_probability_l3083_308320

/-- Represents a jump of the frog -/
structure Jump where
  direction : ℝ × ℝ
  length : ℝ
  random_direction : Bool

/-- Represents the frog's journey -/
structure FrogJourney where
  jumps : List Jump
  final_position : ℝ × ℝ

/-- The probability of the frog's final position being within 1 meter of the start -/
def probability_within_one_meter (journey : FrogJourney) : ℝ :=
  sorry

/-- Theorem stating the probability of the frog's final position being within 1 meter of the start -/
theorem frog_jump_probability :
  ∀ (journey : FrogJourney),
    journey.jumps.length = 4 ∧
    (∀ jump ∈ journey.jumps, jump.length = 1 ∧ jump.random_direction) →
    probability_within_one_meter journey = 1/5 :=
  sorry

end frog_jump_probability_l3083_308320


namespace sqrt_x_minus_one_squared_l3083_308346

theorem sqrt_x_minus_one_squared (x : ℝ) (h : |2 - x| = 2 + |x|) : 
  Real.sqrt ((x - 1)^2) = 1 - x := by sorry

end sqrt_x_minus_one_squared_l3083_308346


namespace unique_solution_condition_l3083_308334

theorem unique_solution_condition (j : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 6) = -52 + j * x) ↔ 
  (j = -14 + 4 * Real.sqrt 21 ∨ j = -14 - 4 * Real.sqrt 21) :=
sorry

end unique_solution_condition_l3083_308334


namespace quadratic_inequality_equivalence_l3083_308381

theorem quadratic_inequality_equivalence (x : ℝ) :
  x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end quadratic_inequality_equivalence_l3083_308381


namespace f_definition_l3083_308354

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_definition (x : ℝ) : f x = 2 - x :=
  sorry

end f_definition_l3083_308354


namespace coins_percentage_of_dollar_l3083_308330

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of pennies in Samantha's purse -/
def num_pennies : ℕ := 2

/-- The number of nickels in Samantha's purse -/
def num_nickels : ℕ := 1

/-- The number of dimes in Samantha's purse -/
def num_dimes : ℕ := 3

/-- The number of quarters in Samantha's purse -/
def num_quarters : ℕ := 2

/-- The total value of coins in Samantha's purse as a percentage of a dollar -/
theorem coins_percentage_of_dollar :
  (num_pennies * penny_value + num_nickels * nickel_value +
   num_dimes * dime_value + num_quarters * quarter_value) * 100 / 100 = 87 := by
  sorry

end coins_percentage_of_dollar_l3083_308330


namespace derivative_at_one_l3083_308352

open Real

noncomputable def f (x : ℝ) (f'1 : ℝ) : ℝ := 2 * x * f'1 + log x

theorem derivative_at_one (f'1 : ℝ) :
  (∀ x > 0, f x f'1 = 2 * x * f'1 + log x) →
  deriv (f · f'1) 1 = -1 :=
by sorry

end derivative_at_one_l3083_308352


namespace children_left_l3083_308312

theorem children_left (total_guests : ℕ) (men : ℕ) (stayed : ℕ) :
  total_guests = 50 ∧ 
  men = 15 ∧ 
  stayed = 43 →
  (total_guests / 2 : ℕ) + men + ((total_guests - (total_guests / 2 + men)) - 
    (total_guests - stayed - men / 5)) = 4 := by
  sorry

end children_left_l3083_308312


namespace correlation_identification_l3083_308349

-- Define the relationships
def age_wealth_relation : Type := Unit
def point_coordinates_relation : Type := Unit
def apple_climate_relation : Type := Unit
def tree_diameter_height_relation : Type := Unit

-- Define the concept of correlation
def has_correlation (relation : Type) : Prop := sorry

-- Define the concept of deterministic relationship
def is_deterministic (relation : Type) : Prop := sorry

-- Theorem statement
theorem correlation_identification :
  (has_correlation age_wealth_relation) ∧
  (has_correlation apple_climate_relation) ∧
  (has_correlation tree_diameter_height_relation) ∧
  (is_deterministic point_coordinates_relation) ∧
  (¬ has_correlation point_coordinates_relation) := by sorry

end correlation_identification_l3083_308349


namespace fraction_sum_equals_two_l3083_308396

theorem fraction_sum_equals_two (a b : ℝ) (h : a ≠ b) : 
  (2 * a) / (a - b) + (2 * b) / (b - a) = 2 := by
  sorry

end fraction_sum_equals_two_l3083_308396


namespace variance_invariant_under_translation_mutually_exclusive_events_l3083_308385

-- Define a dataset as a list of real numbers
def Dataset := List Real

-- Define variance function
noncomputable def variance (data : Dataset) : Real := sorry

-- Define a function to add a constant to each element of a dataset
def addConstant (data : Dataset) (c : Real) : Dataset := sorry

-- Statement 1: Variance remains unchanged after adding a constant
theorem variance_invariant_under_translation (data : Dataset) (c : Real) :
  variance (addConstant data c) = variance data := by sorry

-- Define a type for students
inductive Student
| Boy
| Girl

-- Define a function to create a group of students
def createGroup (numBoys numGirls : Nat) : List Student := sorry

-- Define a function to select n students from a group
def selectStudents (group : List Student) (n : Nat) : List (List Student) := sorry

-- Define predicates for the events
def atLeastOneGirl (selection : List Student) : Prop := sorry
def allBoys (selection : List Student) : Prop := sorry

-- Statement 2: "At least 1 girl" and "all boys" are mutually exclusive when selecting 2 from 3 boys and 2 girls
theorem mutually_exclusive_events :
  let group := createGroup 3 2
  let selections := selectStudents group 2
  ∀ selection ∈ selections, ¬(atLeastOneGirl selection ∧ allBoys selection) := by sorry

end variance_invariant_under_translation_mutually_exclusive_events_l3083_308385


namespace percent_of_number_l3083_308338

theorem percent_of_number (x : ℝ) : (26 / 100) * x = 93.6 → x = 360 := by
  sorry

end percent_of_number_l3083_308338


namespace parallel_vectors_imply_x_equals_two_l3083_308397

def a : Fin 2 → ℝ := ![1, 1]
def b (x : ℝ) : Fin 2 → ℝ := ![2, x]

theorem parallel_vectors_imply_x_equals_two :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ (a + b x) = k • (4 • b x - 2 • a)) → x = 2 := by
  sorry

end parallel_vectors_imply_x_equals_two_l3083_308397


namespace negative_two_x_times_three_y_l3083_308322

theorem negative_two_x_times_three_y (x y : ℝ) : -2 * x * 3 * y = -6 * x * y := by
  sorry

end negative_two_x_times_three_y_l3083_308322


namespace average_speed_two_hours_l3083_308315

/-- The average speed of a car over two hours, given its speeds in each hour -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 90 → speed2 = 75 → (speed1 + speed2) / 2 = 82.5 := by
  sorry

end average_speed_two_hours_l3083_308315


namespace min_value_of_f_l3083_308365

/-- The function f(x) = -(x-1)³ + 12x + a - 1 -/
def f (x a : ℝ) : ℝ := -(x-1)^3 + 12*x + a - 1

/-- The interval [a, b] -/
def closed_interval (a b : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ b}

theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ closed_interval (-2) 2, ∀ y ∈ closed_interval (-2) 2, f y a ≤ f x a) ∧
  (∃ x ∈ closed_interval (-2) 2, f x a = 20) →
  (∃ x ∈ closed_interval (-2) 2, f x a = -7 ∧ ∀ y ∈ closed_interval (-2) 2, -7 ≤ f y a) :=
by sorry

end min_value_of_f_l3083_308365


namespace sports_club_players_l3083_308303

/-- The number of players in a sports club with three games: kabaddi, kho kho, and badminton -/
theorem sports_club_players (kabaddi kho_kho_only badminton both_kabaddi_kho_kho both_kabaddi_badminton both_kho_kho_badminton all_three : ℕ) 
  (h1 : kabaddi = 20)
  (h2 : kho_kho_only = 50)
  (h3 : badminton = 25)
  (h4 : both_kabaddi_kho_kho = 15)
  (h5 : both_kabaddi_badminton = 10)
  (h6 : both_kho_kho_badminton = 5)
  (h7 : all_three = 3) :
  kabaddi + kho_kho_only + badminton - both_kabaddi_kho_kho - both_kabaddi_badminton - both_kho_kho_badminton + all_three = 68 := by
  sorry


end sports_club_players_l3083_308303


namespace quadratic_equation_solution_l3083_308318

theorem quadratic_equation_solution (a : ℝ) : 
  ((-1)^2 - 2*(-1) + a = 0) → 
  (3^2 - 2*3 + a = 0) ∧ 
  (∀ x : ℝ, x^2 - 2*x + a = 0 → (x = -1 ∨ x = 3)) :=
by sorry

end quadratic_equation_solution_l3083_308318


namespace units_digit_of_33_power_l3083_308305

theorem units_digit_of_33_power (n : ℕ) : (33 ^ (33 * (22 ^ 22))) % 10 = 1 := by
  sorry

end units_digit_of_33_power_l3083_308305


namespace fraction_subtraction_l3083_308327

theorem fraction_subtraction : (18 : ℚ) / 42 - 2 / 9 = 13 / 63 := by
  sorry

end fraction_subtraction_l3083_308327


namespace margarita_ricciana_difference_l3083_308340

/-- Ricciana's running distance in feet -/
def ricciana_run : ℝ := 20

/-- Ricciana's jumping distance in feet -/
def ricciana_jump : ℝ := 4

/-- Margarita's running distance in feet -/
def margarita_run : ℝ := 18

/-- Calculates Margarita's jumping distance in feet -/
def margarita_jump : ℝ := 2 * ricciana_jump - 1

/-- Calculates Ricciana's total distance (run + jump) in feet -/
def ricciana_total : ℝ := ricciana_run + ricciana_jump

/-- Calculates Margarita's total distance (run + jump) in feet -/
def margarita_total : ℝ := margarita_run + margarita_jump

/-- Proves that Margarita ran and jumped 1 foot farther than Ricciana -/
theorem margarita_ricciana_difference : margarita_total - ricciana_total = 1 := by
  sorry

end margarita_ricciana_difference_l3083_308340


namespace probability_multiple_of_15_l3083_308321

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A five-digit number without repeating digits -/
def FiveDigitNumber := {n : Finset Nat // n.card = 5 ∧ n ⊆ digits}

/-- The set of all possible five-digit numbers -/
def allNumbers : Finset FiveDigitNumber := sorry

/-- Predicate to check if a number is a multiple of 15 -/
def isMultipleOf15 (n : FiveDigitNumber) : Prop := sorry

/-- The set of five-digit numbers that are multiples of 15 -/
def multiplesOf15 : Finset FiveDigitNumber := sorry

/-- The probability of drawing a multiple of 15 -/
def probabilityMultipleOf15 : ℚ := (multiplesOf15.card : ℚ) / (allNumbers.card : ℚ)

theorem probability_multiple_of_15 : probabilityMultipleOf15 = 1 / 5 := by sorry

end probability_multiple_of_15_l3083_308321


namespace train_distance_l3083_308379

/-- Proves that a train traveling at 3 m/s for 9 seconds covers 27 meters -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 3 → time = 9 → distance = speed * time → distance = 27 :=
by sorry

end train_distance_l3083_308379


namespace cookie_boxes_theorem_l3083_308301

theorem cookie_boxes_theorem (n : ℕ) : 
  (∃ (m a : ℕ), 
    m = n - 11 ∧ 
    a = n - 2 ∧ 
    m ≥ 1 ∧ 
    a ≥ 1 ∧ 
    m + a < n) → 
  n = 12 := by
  sorry

end cookie_boxes_theorem_l3083_308301


namespace binary_1010011_conversion_l3083_308353

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its hexadecimal representation -/
def decimal_to_hex (n : ℕ) : String :=
  let rec aux (m : ℕ) : List Char :=
    if m = 0 then []
    else
      let digit := m % 16
      let char := if digit < 10 then Char.ofNat (digit + 48) else Char.ofNat (digit + 55)
      char :: aux (m / 16)
  String.mk (aux n).reverse

/-- The binary number 1010011₂ -/
def binary_1010011 : List Bool := [true, true, false, false, true, false, true]

theorem binary_1010011_conversion :
  (binary_to_decimal binary_1010011 = 83) ∧
  (decimal_to_hex (binary_to_decimal binary_1010011) = "53") := by
  sorry

end binary_1010011_conversion_l3083_308353


namespace quadratic_translation_l3083_308355

-- Define the original function
def f (x : ℝ) : ℝ := x^2 + 3

-- Define the transformed function
def g (x : ℝ) : ℝ := (x + 2)^2 + 1

-- Theorem statement
theorem quadratic_translation (x : ℝ) : 
  g x = f (x + 2) - 2 := by sorry

end quadratic_translation_l3083_308355


namespace hyperbola_equation_l3083_308364

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- focal distance
  e : ℝ  -- eccentricity

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Theorem: For a hyperbola with given properties, prove its standard equation -/
theorem hyperbola_equation (h : Hyperbola) 
  (h_b : h.b = 12)
  (h_e : h.e = 5/4)
  (h_foci : h.c^2 = h.a^2 + h.b^2)
  (x y : ℝ) :
  standard_equation h x y ↔ x^2 / 64 - y^2 / 36 = 1 :=
by sorry

end hyperbola_equation_l3083_308364


namespace forum_questions_per_hour_l3083_308368

/-- Proves that given the conditions of the forum, the average number of questions posted by each user per hour is 3 -/
theorem forum_questions_per_hour (members : ℕ) (total_posts_per_day : ℕ) 
  (h1 : members = 200)
  (h2 : total_posts_per_day = 57600) : 
  (total_posts_per_day / (24 * members)) / 4 = 3 := by
  sorry

#check forum_questions_per_hour

end forum_questions_per_hour_l3083_308368


namespace product_of_two_primes_not_prime_l3083_308398

/-- A number is prime if it's greater than 1 and its only positive divisors are 1 and itself -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem product_of_two_primes_not_prime (a b : ℤ) :
  isPrime (Int.natAbs (a * b)) → ¬(isPrime (Int.natAbs a) ∧ isPrime (Int.natAbs b)) := by
  sorry

end product_of_two_primes_not_prime_l3083_308398


namespace second_number_value_l3083_308332

theorem second_number_value (x : ℚ) 
  (sum_condition : 2*x + x + (2/3)*x + (1/2)*x = 330) : x = 46 := by
  sorry

end second_number_value_l3083_308332


namespace max_white_pieces_l3083_308325

/-- Represents the color of a piece -/
inductive Color
| Black
| White

/-- Represents the circle of pieces -/
def Circle := List Color

/-- The initial configuration of the circle -/
def initial_circle : Circle :=
  [Color.Black, Color.Black, Color.Black, Color.Black, Color.White]

/-- Applies the rules to place new pieces and remove old ones -/
def apply_rules (c : Circle) : Circle :=
  sorry

/-- Counts the number of white pieces in the circle -/
def count_white (c : Circle) : Nat :=
  sorry

/-- Theorem stating that the maximum number of white pieces is 3 -/
theorem max_white_pieces (c : Circle) : 
  count_white (apply_rules c) ≤ 3 :=
sorry

end max_white_pieces_l3083_308325


namespace vector_sum_l3083_308350

theorem vector_sum (x : ℝ) : 
  (⟨-3, 4, -2⟩ : ℝ × ℝ × ℝ) + (⟨5, -3, x⟩ : ℝ × ℝ × ℝ) = ⟨2, 1, x - 2⟩ := by
sorry

end vector_sum_l3083_308350


namespace bears_in_stock_before_shipment_l3083_308326

/-- The number of bears in a new shipment -/
def new_shipment : ℕ := 10

/-- The number of bears on each shelf -/
def bears_per_shelf : ℕ := 9

/-- The number of shelves used -/
def shelves_used : ℕ := 3

/-- The number of bears in stock before the new shipment -/
def bears_before_shipment : ℕ := shelves_used * bears_per_shelf - new_shipment

theorem bears_in_stock_before_shipment :
  bears_before_shipment = 17 := by
  sorry

end bears_in_stock_before_shipment_l3083_308326


namespace work_time_for_c_l3083_308310

/-- The time it takes for worker c to complete the work alone, given the combined work rates of pairs of workers. -/
theorem work_time_for_c (a b c : ℝ) 
  (h1 : a + b = 1/4)   -- a and b can do the work in 4 days
  (h2 : b + c = 1/6)   -- b and c can do the work in 6 days
  (h3 : c + a = 1/3) : -- c and a can do the work in 3 days
  1/c = 8 := by sorry

end work_time_for_c_l3083_308310


namespace canoe_row_probability_l3083_308337

-- Define the probability of each oar working
def p_left_works : ℚ := 3/5
def p_right_works : ℚ := 3/5

-- Define the event of being able to row the canoe
def can_row : ℚ := 
  p_left_works * p_right_works +  -- both oars work
  p_left_works * (1 - p_right_works) +  -- left works, right breaks
  (1 - p_left_works) * p_right_works  -- left breaks, right works

-- Theorem statement
theorem canoe_row_probability : can_row = 21/25 := by
  sorry

end canoe_row_probability_l3083_308337


namespace negation_of_universal_proposition_l3083_308388

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end negation_of_universal_proposition_l3083_308388


namespace power_function_through_point_l3083_308308

theorem power_function_through_point (a : ℝ) : (fun x : ℝ => x^a) 2 = 16 → a = 4 := by
  sorry

end power_function_through_point_l3083_308308


namespace initial_lives_proof_l3083_308395

/-- Represents the number of lives Kaleb had initially -/
def initial_lives : ℕ := 98

/-- Represents the number of lives Kaleb lost -/
def lives_lost : ℕ := 25

/-- Represents the number of lives Kaleb had remaining -/
def remaining_lives : ℕ := 73

/-- Theorem stating that the initial number of lives equals the sum of remaining lives and lives lost -/
theorem initial_lives_proof : initial_lives = remaining_lives + lives_lost := by
  sorry

end initial_lives_proof_l3083_308395


namespace imaginary_part_of_product_l3083_308372

/-- The imaginary part of (1 - i)(2 + 4i) is 2, where i is the imaginary unit. -/
theorem imaginary_part_of_product : Complex.im ((1 - Complex.I) * (2 + 4 * Complex.I)) = 2 := by
  sorry

end imaginary_part_of_product_l3083_308372


namespace systematicSamplingExample_l3083_308343

/-- Calculates the number of groups for systematic sampling -/
def systematicSamplingGroups (totalStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  totalStudents / sampleSize

/-- Theorem stating that for 600 students and a sample size of 20, 
    the number of groups for systematic sampling is 30 -/
theorem systematicSamplingExample : 
  systematicSamplingGroups 600 20 = 30 := by
  sorry

end systematicSamplingExample_l3083_308343


namespace corn_acreage_l3083_308311

theorem corn_acreage (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : beans_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end corn_acreage_l3083_308311


namespace sheila_hourly_rate_l3083_308384

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_long_day : ℕ
  days_long : ℕ
  hours_short_day : ℕ
  days_short : ℕ
  weekly_earnings : ℕ

/-- Calculate hourly rate given a work schedule --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (schedule.hours_long_day * schedule.days_long + 
                              schedule.hours_short_day * schedule.days_short)

/-- Sheila's specific work schedule --/
def sheila_schedule : WorkSchedule := {
  hours_long_day := 8,
  days_long := 3,
  hours_short_day := 6,
  days_short := 2,
  weekly_earnings := 252
}

/-- Theorem: Sheila's hourly rate is $7 --/
theorem sheila_hourly_rate : hourly_rate sheila_schedule = 7 := by
  sorry


end sheila_hourly_rate_l3083_308384


namespace unique_correct_ranking_l3083_308336

/-- Represents the participants in the long jump competition -/
inductive Participant
| Decimals
| Elementary
| Xiaohua
| Xiaoyuan
| Exploration

/-- Represents the ranking of participants -/
def Ranking := Participant → Fin 5

/-- Checks if a ranking satisfies all the given conditions -/
def satisfies_conditions (r : Ranking) : Prop :=
  (r Participant.Decimals < r Participant.Elementary) ∧
  (r Participant.Xiaohua > r Participant.Xiaoyuan) ∧
  (r Participant.Exploration > r Participant.Elementary) ∧
  (r Participant.Elementary < r Participant.Xiaohua) ∧
  (r Participant.Xiaoyuan > r Participant.Exploration)

/-- The correct ranking of participants -/
def correct_ranking : Ranking :=
  fun p => match p with
    | Participant.Decimals => 0
    | Participant.Elementary => 1
    | Participant.Exploration => 2
    | Participant.Xiaoyuan => 3
    | Participant.Xiaohua => 4

/-- Theorem stating that the correct_ranking is the unique ranking that satisfies all conditions -/
theorem unique_correct_ranking :
  satisfies_conditions correct_ranking ∧
  ∀ (r : Ranking), satisfies_conditions r → r = correct_ranking :=
sorry

end unique_correct_ranking_l3083_308336


namespace constant_is_arithmetic_l3083_308358

def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a m

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem constant_is_arithmetic :
  ∀ a : ℕ → ℝ, is_constant_sequence a → is_arithmetic_sequence a :=
by
  sorry

end constant_is_arithmetic_l3083_308358


namespace unique_solution_condition_l3083_308329

theorem unique_solution_condition (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 + 1 ∧ p.2 = 4*p.1 + k) ↔ k = 0 :=
by sorry

end unique_solution_condition_l3083_308329


namespace coordinate_sum_theorem_l3083_308359

/-- Given a function f where f(3) = 4, the sum of the coordinates of the point (x, y) 
    satisfying 4y = 2f(2x) + 7 is equal to 5.25. -/
theorem coordinate_sum_theorem (f : ℝ → ℝ) (hf : f 3 = 4) :
  ∃ (x y : ℝ), 4 * y = 2 * f (2 * x) + 7 ∧ x + y = 5.25 := by
  sorry

end coordinate_sum_theorem_l3083_308359


namespace all_propositions_false_l3083_308351

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define relationships between lines
variable (parallel : Line → Line → Prop)
variable (coplanar : Line → Line → Prop)
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem all_propositions_false :
  (∀ a b c : Line,
    (parallel a b ∧ ¬coplanar a c) → ¬coplanar b c) ∧
  (∀ a b c : Line,
    (coplanar a b ∧ ¬coplanar b c) → ¬coplanar a c) ∧
  (∀ a b c : Line,
    (¬coplanar a b ∧ coplanar a c) → ¬coplanar b c) ∧
  (∀ a b c : Line,
    (¬coplanar a b ∧ ¬intersect b c) → ¬intersect a c) →
  False :=
sorry

end all_propositions_false_l3083_308351


namespace chocolate_bars_left_l3083_308335

theorem chocolate_bars_left (initial_bars : ℕ) (people : ℕ) (given_to_mother : ℕ) (eaten : ℕ) : 
  initial_bars = 20 →
  people = 5 →
  given_to_mother = 3 →
  eaten = 2 →
  (initial_bars / people / 2 * people) - given_to_mother - eaten = 5 :=
by sorry

end chocolate_bars_left_l3083_308335


namespace planes_with_parallel_lines_are_parallel_or_intersecting_l3083_308314

/-- Two planes in 3D space -/
structure Plane3D where
  -- Add necessary fields here
  
/-- A straight line in 3D space -/
structure Line3D where
  -- Add necessary fields here

/-- Predicate to check if a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate to check if two lines are parallel -/
def lines_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate to check if two planes are parallel -/
def planes_parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Predicate to check if two planes are intersecting -/
def planes_intersecting (p1 p2 : Plane3D) : Prop :=
  sorry

theorem planes_with_parallel_lines_are_parallel_or_intersecting 
  (p1 p2 : Plane3D) (l1 l2 : Line3D) 
  (h1 : line_in_plane l1 p1) 
  (h2 : line_in_plane l2 p2) 
  (h3 : lines_parallel l1 l2) : 
  planes_parallel p1 p2 ∨ planes_intersecting p1 p2 :=
sorry

end planes_with_parallel_lines_are_parallel_or_intersecting_l3083_308314
