import Mathlib

namespace NUMINAMATH_CALUDE_solve_system_l459_45960

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20) 
  (eq2 : 6 * p + 5 * q = 29) : 
  q = -25 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l459_45960


namespace NUMINAMATH_CALUDE_movie_session_duration_l459_45990

/-- Represents the start time of a movie session -/
structure SessionTime where
  hour : ℕ
  minute : ℕ
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents the duration of a movie session -/
structure SessionDuration where
  hours : ℕ
  minutes : ℕ
  m_valid : minutes < 60

/-- Checks if a given SessionTime is consistent with the known session times -/
def is_consistent (st : SessionTime) (duration : SessionDuration) : Prop :=
  let next_session := SessionTime.mk 
    ((st.hour + duration.hours + (st.minute + duration.minutes) / 60) % 24)
    ((st.minute + duration.minutes) % 60)
    sorry
    sorry
  (st.hour = 12 ∧ next_session.hour = 13) ∨
  (st.hour = 13 ∧ next_session.hour = 14) ∨
  (st.hour = 23 ∧ next_session.hour = 24) ∨
  (st.hour = 24 ∧ next_session.hour = 1)

theorem movie_session_duration : 
  ∃ (start : SessionTime) (duration : SessionDuration),
    duration.hours = 1 ∧ 
    duration.minutes = 50 ∧
    is_consistent start duration ∧
    (∀ (other_duration : SessionDuration),
      is_consistent start other_duration → 
      other_duration = duration) := by
  sorry

end NUMINAMATH_CALUDE_movie_session_duration_l459_45990


namespace NUMINAMATH_CALUDE_locus_characterization_l459_45914

/-- The locus of points equidistant from A(4, 1) and the y-axis -/
def locus_equation (x y : ℝ) : Prop :=
  (y - 1)^2 = 16 * (x - 2)

/-- A point P(x, y) is equidistant from A(4, 1) and the y-axis -/
def is_equidistant (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 1)^2 = x^2

theorem locus_characterization (x y : ℝ) :
  is_equidistant x y ↔ locus_equation x y := by sorry

end NUMINAMATH_CALUDE_locus_characterization_l459_45914


namespace NUMINAMATH_CALUDE_books_initially_l459_45907

/-- Given that Paul bought some books and ended up with a certain total, 
    this theorem proves how many books he had initially. -/
theorem books_initially (bought : ℕ) (total_after : ℕ) (h : bought = 101) (h' : total_after = 151) :
  total_after - bought = 50 := by
  sorry

end NUMINAMATH_CALUDE_books_initially_l459_45907


namespace NUMINAMATH_CALUDE_cycle_reappearance_l459_45968

theorem cycle_reappearance (letter_cycle_length digit_cycle_length : ℕ) 
  (h1 : letter_cycle_length = 7)
  (h2 : digit_cycle_length = 4) :
  Nat.lcm letter_cycle_length digit_cycle_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_cycle_reappearance_l459_45968


namespace NUMINAMATH_CALUDE_power_comparison_l459_45915

theorem power_comparison (h1 : 2 > 1) (h2 : -1.1 > -1.2) : 2^(-1.1) > 2^(-1.2) := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l459_45915


namespace NUMINAMATH_CALUDE_multiplier_is_three_l459_45972

theorem multiplier_is_three : ∃ m : ℤ, m = 3 ∧ ∀ x : ℤ, x = 13 → m * x = (26 - x) + 26 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_is_three_l459_45972


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l459_45924

/-- A geometric sequence is a sequence where the ratio between consecutive terms is constant. -/
def IsGeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- Given a geometric sequence b where b₂ * b₃ * b₄ = 8, prove that b₃ = 2 -/
theorem geometric_sequence_product (b : ℕ → ℝ) (h_geo : IsGeometricSequence b) 
    (h_prod : b 2 * b 3 * b 4 = 8) : b 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l459_45924


namespace NUMINAMATH_CALUDE_bus_driver_regular_rate_l459_45906

/-- Calculates the regular hourly rate for a bus driver given their total hours worked,
    total compensation, and overtime policy. -/
def calculate_regular_rate (total_hours : ℕ) (total_compensation : ℚ) : ℚ :=
  let regular_hours := min total_hours 40
  let overtime_hours := total_hours - regular_hours
  let rate := total_compensation / (regular_hours + 1.75 * overtime_hours)
  rate

/-- Theorem stating that given the specific conditions of the bus driver's work week,
    their regular hourly rate is $16. -/
theorem bus_driver_regular_rate :
  calculate_regular_rate 54 1032 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_regular_rate_l459_45906


namespace NUMINAMATH_CALUDE_tims_garden_fence_length_l459_45975

/-- The perimeter of an irregular pentagon with given side lengths -/
def pentagon_perimeter (a b c d e : ℝ) : ℝ := a + b + c + d + e

/-- Theorem: The perimeter of Tim's garden fence -/
theorem tims_garden_fence_length :
  pentagon_perimeter 28 32 25 35 39 = 159 := by
  sorry

end NUMINAMATH_CALUDE_tims_garden_fence_length_l459_45975


namespace NUMINAMATH_CALUDE_problem_statement_l459_45986

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) := Real.log x - x + 2

theorem problem_statement :
  (∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 1) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), x ≥ 1 → m * f x ≥ (x - 1) / (x + 1)) ↔ m ≥ 1/2) ∧
  (∀ (α : ℝ), 0 < α ∧ α < Real.pi/2 →
    ((0 < α ∧ α < Real.pi/4 → f (Real.tan α) < -Real.cos (2*α)) ∧
     (α = Real.pi/4 → f (Real.tan α) = -Real.cos (2*α)) ∧
     (Real.pi/4 < α ∧ α < Real.pi/2 → f (Real.tan α) > -Real.cos (2*α)))) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l459_45986


namespace NUMINAMATH_CALUDE_remainder_product_mod_twelve_l459_45939

theorem remainder_product_mod_twelve : (1425 * 1427 * 1429) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_product_mod_twelve_l459_45939


namespace NUMINAMATH_CALUDE_counterfeit_coins_l459_45981

def bags : List Nat := [18, 19, 21, 23, 25, 34]

structure Distribution where
  xiaocong : List Nat
  xiaomin : List Nat
  counterfeit : Nat

def isValidDistribution (d : Distribution) : Prop :=
  d.xiaocong.length = 3 ∧
  d.xiaomin.length = 2 ∧
  d.xiaocong.sum = 2 * d.xiaomin.sum ∧
  d.counterfeit ∈ bags ∧
  d.xiaocong.sum + d.xiaomin.sum + d.counterfeit = bags.sum

theorem counterfeit_coins (d : Distribution) :
  isValidDistribution d → d.counterfeit = 23 := by
  sorry

end NUMINAMATH_CALUDE_counterfeit_coins_l459_45981


namespace NUMINAMATH_CALUDE_lollipop_reimbursement_l459_45916

/-- Given that Sarah bought 12 lollipops for 3 dollars and shared one-quarter with Julie,
    prove that Julie reimbursed Sarah 75 cents. -/
theorem lollipop_reimbursement (total_lollipops : ℕ) (total_cost : ℚ) (share_fraction : ℚ) :
  total_lollipops = 12 →
  total_cost = 3 →
  share_fraction = 1/4 →
  (share_fraction * total_lollipops : ℚ) * (total_cost / total_lollipops) * 100 = 75 := by
  sorry

#check lollipop_reimbursement

end NUMINAMATH_CALUDE_lollipop_reimbursement_l459_45916


namespace NUMINAMATH_CALUDE_tie_record_score_difference_l459_45937

/-- The league record average score per player per round -/
def league_record : ℕ := 287

/-- The number of players in a team -/
def players_per_team : ℕ := 4

/-- The number of rounds in a season -/
def rounds_per_season : ℕ := 10

/-- The total score of George's team after 9 rounds -/
def team_score_9_rounds : ℕ := 10440

/-- The minimum average score needed per player in the final round to tie the record -/
def min_avg_score_final_round : ℕ := (league_record * players_per_team * rounds_per_season - team_score_9_rounds) / players_per_team

/-- The difference between the league record average and the minimum average score needed -/
def score_difference : ℕ := league_record - min_avg_score_final_round

theorem tie_record_score_difference : score_difference = 27 := by
  sorry

end NUMINAMATH_CALUDE_tie_record_score_difference_l459_45937


namespace NUMINAMATH_CALUDE_min_squares_to_exceed_10000_l459_45902

/-- Represents the squaring operation on a calculator --/
def square (x : ℕ) : ℕ := x * x

/-- Represents n iterations of squaring, starting from x --/
def iterate_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => square (iterate_square x n)

/-- The theorem to be proved --/
theorem min_squares_to_exceed_10000 :
  (∃ n : ℕ, iterate_square 5 n > 10000) ∧
  (∀ n : ℕ, iterate_square 5 n > 10000 → n ≥ 3) ∧
  (iterate_square 5 3 > 10000) :=
sorry

end NUMINAMATH_CALUDE_min_squares_to_exceed_10000_l459_45902


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l459_45922

theorem distance_from_origin_to_point :
  let x : ℝ := 8
  let y : ℝ := 15
  Real.sqrt (x^2 + y^2) = 17 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l459_45922


namespace NUMINAMATH_CALUDE_profit_per_meter_of_cloth_l459_45985

/-- Profit per meter of cloth calculation -/
theorem profit_per_meter_of_cloth
  (total_meters : ℕ)
  (selling_price : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : total_meters = 45)
  (h2 : selling_price = 4500)
  (h3 : cost_price_per_meter = 86) :
  (selling_price - total_meters * cost_price_per_meter) / total_meters = 14 := by
sorry

end NUMINAMATH_CALUDE_profit_per_meter_of_cloth_l459_45985


namespace NUMINAMATH_CALUDE_abs_2x_minus_7_not_positive_l459_45931

theorem abs_2x_minus_7_not_positive (x : ℚ) : |2*x - 7| ≤ 0 ↔ x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_2x_minus_7_not_positive_l459_45931


namespace NUMINAMATH_CALUDE_smaller_exterior_angle_implies_obtuse_l459_45977

-- Define a triangle
structure Triangle where
  -- We don't need to specify the exact properties of a triangle here

-- Define the property of having an exterior angle smaller than its adjacent interior angle
def has_smaller_exterior_angle (t : Triangle) : Prop :=
  ∃ (exterior_angle interior_angle : ℝ), exterior_angle < interior_angle

-- Define an obtuse triangle
def is_obtuse (t : Triangle) : Prop :=
  ∃ (angle : ℝ), angle > Real.pi / 2

-- State the theorem
theorem smaller_exterior_angle_implies_obtuse (t : Triangle) :
  has_smaller_exterior_angle t → is_obtuse t :=
sorry

end NUMINAMATH_CALUDE_smaller_exterior_angle_implies_obtuse_l459_45977


namespace NUMINAMATH_CALUDE_combined_liquid_fraction_l459_45998

/-- Represents the capacity of a beaker -/
structure Beaker where
  capacity : ℝ
  filled : ℝ
  density : ℝ

/-- The problem setup -/
def problemSetup : Prop := ∃ (small large third : Beaker),
  -- Small beaker conditions
  small.filled = (1/2) * small.capacity ∧
  small.density = 1.025 ∧
  -- Large beaker conditions
  large.capacity = 5 * small.capacity ∧
  large.filled = (1/5) * large.capacity ∧
  large.density = 1 ∧
  -- Third beaker conditions
  third.capacity = (1/2) * large.capacity ∧
  third.filled = (3/4) * third.capacity ∧
  third.density = 0.85

/-- The theorem to prove -/
theorem combined_liquid_fraction (h : problemSetup) :
  ∃ (small large third : Beaker),
  (large.filled + small.filled + third.filled) / large.capacity = 27/40 := by
  sorry

end NUMINAMATH_CALUDE_combined_liquid_fraction_l459_45998


namespace NUMINAMATH_CALUDE_max_removal_is_seven_l459_45944

def yellow_marbles : ℕ := 8
def red_marbles : ℕ := 7
def black_marbles : ℕ := 5

def total_marbles : ℕ := yellow_marbles + red_marbles + black_marbles

def is_valid_removal (n : ℕ) : Prop :=
  ∀ (y r b : ℕ),
    y + r + b = total_marbles - n →
    y ≤ yellow_marbles ∧ r ≤ red_marbles ∧ b ≤ black_marbles →
    (y ≥ 4 ∧ (r ≥ 3 ∨ b ≥ 3)) ∨
    (r ≥ 4 ∧ (y ≥ 3 ∨ b ≥ 3)) ∨
    (b ≥ 4 ∧ (y ≥ 3 ∨ r ≥ 3))

theorem max_removal_is_seven :
  is_valid_removal 7 ∧ ¬is_valid_removal 8 := by sorry

end NUMINAMATH_CALUDE_max_removal_is_seven_l459_45944


namespace NUMINAMATH_CALUDE_sandwich_non_filler_percentage_l459_45997

/-- Given a sandwich weighing 180 grams with 45 grams of fillers,
    prove that the percentage of the sandwich that is not filler is 75%. -/
theorem sandwich_non_filler_percentage
  (total_weight : ℝ)
  (filler_weight : ℝ)
  (h1 : total_weight = 180)
  (h2 : filler_weight = 45) :
  (total_weight - filler_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_non_filler_percentage_l459_45997


namespace NUMINAMATH_CALUDE_rectangular_field_length_l459_45963

/-- Proves the length of a rectangular field given specific conditions -/
theorem rectangular_field_length : ∀ w : ℝ,
  w > 0 →  -- width is positive
  (w + 10) * w = 171 →  -- area equation
  w + 10 = 19 :=  -- length equation
by
  sorry

#check rectangular_field_length

end NUMINAMATH_CALUDE_rectangular_field_length_l459_45963


namespace NUMINAMATH_CALUDE_jack_remaining_money_l459_45919

def calculate_remaining_money (initial_amount : ℝ) 
  (sparkling_water_bottles : ℕ) (sparkling_water_cost : ℝ)
  (still_water_multiplier : ℕ) (still_water_cost : ℝ)
  (cheddar_cheese_pounds : ℝ) (cheddar_cheese_cost : ℝ)
  (swiss_cheese_pounds : ℝ) (swiss_cheese_cost : ℝ) : ℝ :=
  let sparkling_water_total := sparkling_water_bottles * sparkling_water_cost
  let still_water_total := (sparkling_water_bottles * still_water_multiplier) * still_water_cost
  let cheddar_cheese_total := cheddar_cheese_pounds * cheddar_cheese_cost
  let swiss_cheese_total := swiss_cheese_pounds * swiss_cheese_cost
  let total_cost := sparkling_water_total + still_water_total + cheddar_cheese_total + swiss_cheese_total
  initial_amount - total_cost

theorem jack_remaining_money :
  calculate_remaining_money 150 4 3 3 2.5 2 8.5 1.5 11 = 74.5 := by
  sorry

end NUMINAMATH_CALUDE_jack_remaining_money_l459_45919


namespace NUMINAMATH_CALUDE_robins_gum_problem_l459_45956

theorem robins_gum_problem (initial_gum : ℝ) (total_gum : ℕ) (h1 : initial_gum = 18.0) (h2 : total_gum = 62) :
  (total_gum : ℝ) - initial_gum = 44 := by
  sorry

end NUMINAMATH_CALUDE_robins_gum_problem_l459_45956


namespace NUMINAMATH_CALUDE_polynomial_equation_sum_l459_45905

theorem polynomial_equation_sum (a b c d : ℤ) :
  (∀ x : ℤ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 + x^2 + 8*x - 12) →
  a + b + c + d = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_sum_l459_45905


namespace NUMINAMATH_CALUDE_congruence_problem_l459_45957

theorem congruence_problem (x : ℤ) : 
  (5 * x + 9) % 18 = 3 → (3 * x + 14) % 18 = 14 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l459_45957


namespace NUMINAMATH_CALUDE_complex_equality_implies_a_value_l459_45962

theorem complex_equality_implies_a_value (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (2 + Complex.I)
  Complex.re z = Complex.im z → a = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_implies_a_value_l459_45962


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l459_45970

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, r > 0 ∧ 30 * r = a ∧ a * r = 9/4) : 
  a = 15 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l459_45970


namespace NUMINAMATH_CALUDE_no_integer_n_for_real_nth_power_of_complex_l459_45966

theorem no_integer_n_for_real_nth_power_of_complex : 
  ¬ ∃ (n : ℤ), (Complex.I + n : ℂ)^5 ∈ Set.range Complex.ofReal := by sorry

end NUMINAMATH_CALUDE_no_integer_n_for_real_nth_power_of_complex_l459_45966


namespace NUMINAMATH_CALUDE_sandy_work_hours_l459_45940

/-- Sandy's work schedule -/
structure WorkSchedule where
  total_hours : ℕ
  num_days : ℕ
  hours_per_day : ℕ
  equal_hours : total_hours = num_days * hours_per_day

/-- Theorem: Sandy worked 9 hours per day -/
theorem sandy_work_hours (schedule : WorkSchedule)
  (h1 : schedule.total_hours = 45)
  (h2 : schedule.num_days = 5) :
  schedule.hours_per_day = 9 := by
  sorry

end NUMINAMATH_CALUDE_sandy_work_hours_l459_45940


namespace NUMINAMATH_CALUDE_max_silver_tokens_l459_45901

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth --/
structure Booth where
  inputRed : ℕ
  inputBlue : ℕ
  outputSilver : ℕ
  outputRed : ℕ
  outputBlue : ℕ

/-- The initial token count --/
def initialTokens : TokenCount :=
  { red := 100, blue := 50, silver := 0 }

/-- The first exchange booth --/
def booth1 : Booth :=
  { inputRed := 3, inputBlue := 0, outputSilver := 1, outputRed := 0, outputBlue := 2 }

/-- The second exchange booth --/
def booth2 : Booth :=
  { inputRed := 0, inputBlue := 4, outputSilver := 1, outputRed := 2, outputBlue := 0 }

/-- Predicate to check if an exchange is possible --/
def canExchange (tokens : TokenCount) (booth : Booth) : Prop :=
  tokens.red ≥ booth.inputRed ∧ tokens.blue ≥ booth.inputBlue

/-- The final token count after all possible exchanges --/
noncomputable def finalTokens : TokenCount :=
  sorry

/-- Theorem stating that the maximum number of silver tokens is 103 --/
theorem max_silver_tokens : finalTokens.silver = 103 := by
  sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l459_45901


namespace NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l459_45954

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  /-- The base angle of the isosceles triangle -/
  baseAngle : ℝ
  /-- The altitude to the base of the isosceles triangle -/
  altitude : ℝ
  /-- The base of the isosceles triangle -/
  base : ℝ

/-- Theorem stating that an isosceles triangle is not uniquely determined by one angle and the altitude to one of its sides -/
theorem isosceles_triangle_not_unique (α : ℝ) (h : ℝ) : 
  ∃ t1 t2 : IsoscelesTriangle, t1.baseAngle = α ∧ t1.altitude = h ∧ 
  t2.baseAngle = α ∧ t2.altitude = h ∧ t1 ≠ t2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l459_45954


namespace NUMINAMATH_CALUDE_stream_speed_l459_45949

/-- Proves that the speed of the stream is 6 kmph given the conditions --/
theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 18 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 6 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l459_45949


namespace NUMINAMATH_CALUDE_total_apples_is_200_l459_45935

/-- The number of apples Kayla picked -/
def kayla_apples : ℕ := 40

/-- The number of apples Kylie picked -/
def kylie_apples : ℕ := 4 * kayla_apples

/-- The total number of apples picked by Kayla and Kylie -/
def total_apples : ℕ := kayla_apples + kylie_apples

/-- Theorem stating that the total number of apples picked is 200 -/
theorem total_apples_is_200 : total_apples = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_200_l459_45935


namespace NUMINAMATH_CALUDE_largest_angle_right_triangle_l459_45928

theorem largest_angle_right_triangle (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 90) (h3 : b / c = 7 / 2) : max a (max b c) = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_right_triangle_l459_45928


namespace NUMINAMATH_CALUDE_model_price_increase_l459_45927

theorem model_price_increase (original_price : ℚ) (original_quantity : ℕ) (new_quantity : ℕ) 
  (h1 : original_price = 45 / 100)
  (h2 : original_quantity = 30)
  (h3 : new_quantity = 27) :
  let total_saved := original_price * original_quantity
  let new_price := total_saved / new_quantity
  new_price = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_model_price_increase_l459_45927


namespace NUMINAMATH_CALUDE_sachins_age_l459_45934

theorem sachins_age (sachin rahul : ℝ) 
  (h1 : rahul = sachin + 7)
  (h2 : sachin / rahul = 7 / 9) :
  sachin = 24.5 := by
sorry

end NUMINAMATH_CALUDE_sachins_age_l459_45934


namespace NUMINAMATH_CALUDE_F_opposite_A_l459_45946

/-- Represents a face of a cube --/
inductive CubeFace
| A | B | C | D | E | F

/-- Represents the position of a face relative to face A in the net --/
inductive Position
| Left | Above | Right | Below | NotAttached

/-- Describes the layout of faces in the cube net --/
def net_layout : CubeFace → Position
| CubeFace.B => Position.Left
| CubeFace.C => Position.Above
| CubeFace.D => Position.Right
| CubeFace.E => Position.Below
| CubeFace.F => Position.NotAttached
| CubeFace.A => Position.NotAttached  -- A's position relative to itself is not relevant

/-- Determines if two faces are opposite in the folded cube --/
def are_opposite (f1 f2 : CubeFace) : Prop := sorry

/-- Theorem stating that face F is opposite to face A when the net is folded --/
theorem F_opposite_A : are_opposite CubeFace.F CubeFace.A := by
  sorry

end NUMINAMATH_CALUDE_F_opposite_A_l459_45946


namespace NUMINAMATH_CALUDE_fraction_division_evaluate_fraction_l459_45982

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  a / (c / d) = (a * d) / c :=
by sorry

theorem evaluate_fraction :
  (4 : ℚ) / ((8 : ℚ) / 13) = 13 / 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_evaluate_fraction_l459_45982


namespace NUMINAMATH_CALUDE_perpendicular_vector_l459_45967

/-- Given three points A, B, C in ℝ³ and a vector a, if a is perpendicular to both AB and AC,
    then a = (1, 1, 1) -/
theorem perpendicular_vector (A B C a : ℝ × ℝ × ℝ) :
  A = (0, 2, 3) →
  B = (-2, 1, 6) →
  C = (1, -1, 5) →
  a.2.2 = 1 →
  (a.1 * (B.1 - A.1) + a.2.1 * (B.2.1 - A.2.1) + a.2.2 * (B.2.2 - A.2.2) = 0) →
  (a.1 * (C.1 - A.1) + a.2.1 * (C.2.1 - A.2.1) + a.2.2 * (C.2.2 - A.2.2) = 0) →
  a = (1, 1, 1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_l459_45967


namespace NUMINAMATH_CALUDE_egg_count_problem_l459_45908

theorem egg_count_problem (count_sum : ℕ) (error_sum : ℤ) (actual_count : ℕ) : 
  count_sum = 3162 →
  (∃ (e1 e2 e3 : ℤ), (e1 = 1 ∨ e1 = -1) ∧ 
                     (e2 = 10 ∨ e2 = -10) ∧ 
                     (e3 = 100 ∨ e3 = -100) ∧ 
                     error_sum = e1 + e2 + e3) →
  7 * actual_count + error_sum = count_sum →
  actual_count = 439 := by
sorry

end NUMINAMATH_CALUDE_egg_count_problem_l459_45908


namespace NUMINAMATH_CALUDE_storks_and_birds_l459_45974

theorem storks_and_birds (initial_storks initial_birds new_birds : ℕ) :
  initial_storks = 6 →
  initial_birds = 2 →
  new_birds = 3 →
  initial_storks - (initial_birds + new_birds) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_and_birds_l459_45974


namespace NUMINAMATH_CALUDE_geometric_sequence_range_l459_45953

theorem geometric_sequence_range (a₁ a₂ a₃ a₄ : ℝ) :
  (∃ q : ℝ, a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q) →
  (0 < a₁ ∧ a₁ < 1) →
  (1 < a₂ ∧ a₂ < 2) →
  (2 < a₃ ∧ a₃ < 4) →
  (2 * Real.sqrt 2 < a₄ ∧ a₄ < 16) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_range_l459_45953


namespace NUMINAMATH_CALUDE_sin_60_abs_5_pi_sqrt2_equality_l459_45938

theorem sin_60_abs_5_pi_sqrt2_equality : 
  2 * Real.sin (π / 3) + |-5| - (π - Real.sqrt 2) ^ 0 = Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_60_abs_5_pi_sqrt2_equality_l459_45938


namespace NUMINAMATH_CALUDE_y_range_l459_45942

theorem y_range (a b y : ℝ) (h1 : a + b = 2) (h2 : b ≤ 2) (h3 : y - a^2 - 2*a + 2 = 0) :
  y ≥ -2 := by
sorry

end NUMINAMATH_CALUDE_y_range_l459_45942


namespace NUMINAMATH_CALUDE_arctan_sum_property_l459_45996

theorem arctan_sum_property (a b c : ℝ) 
  (h : Real.arctan a + Real.arctan b + Real.arctan c + π / 2 = 0) : 
  (a * b + b * c + c * a = 1) ∧ (a + b + c < a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_property_l459_45996


namespace NUMINAMATH_CALUDE_shorter_leg_of_right_triangle_with_hypotenuse_65_l459_45943

-- Define a right triangle with integer side lengths
def RightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0

theorem shorter_leg_of_right_triangle_with_hypotenuse_65 :
  ∃ (a b : ℕ), RightTriangle a b 65 ∧ a ≤ b ∧ a = 25 :=
by sorry

end NUMINAMATH_CALUDE_shorter_leg_of_right_triangle_with_hypotenuse_65_l459_45943


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l459_45903

def is_prime (n : ℕ) : Prop := sorry

def is_square (n : ℕ) : Prop := sorry

def has_prime_factor_less_than (n m : ℕ) : Prop := sorry

theorem smallest_non_prime_non_square_no_small_factors : 
  (∀ k < 4087, k > 0 → is_prime k ∨ is_square k ∨ has_prime_factor_less_than k 60) ∧
  ¬ is_prime 4087 ∧
  ¬ is_square 4087 ∧
  ¬ has_prime_factor_less_than 4087 60 := by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l459_45903


namespace NUMINAMATH_CALUDE_basketball_lineups_l459_45900

def total_players : ℕ := 12
def players_per_lineup : ℕ := 5
def point_guards_per_lineup : ℕ := 1

def number_of_lineups : ℕ :=
  total_players * (Nat.choose (total_players - 1) (players_per_lineup - 1))

theorem basketball_lineups :
  number_of_lineups = 3960 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineups_l459_45900


namespace NUMINAMATH_CALUDE_newton_sports_club_membership_ratio_l459_45933

/-- Proves that given the average ages of female and male members, and the overall average age,
    the ratio of female to male members is 8:17 --/
theorem newton_sports_club_membership_ratio
  (avg_age_female : ℝ)
  (avg_age_male : ℝ)
  (avg_age_all : ℝ)
  (h_female : avg_age_female = 45)
  (h_male : avg_age_male = 20)
  (h_all : avg_age_all = 28)
  : ∃ (f m : ℝ), f > 0 ∧ m > 0 ∧ f / m = 8 / 17 := by
  sorry

end NUMINAMATH_CALUDE_newton_sports_club_membership_ratio_l459_45933


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l459_45988

theorem absolute_value_inequality_solution (x : ℝ) :
  (|2*x - 3| < 5) ↔ (-1 < x ∧ x < 4) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l459_45988


namespace NUMINAMATH_CALUDE_cube_sum_greater_than_product_sufficient_l459_45917

theorem cube_sum_greater_than_product_sufficient (x y z : ℝ) : 
  x + y + z > 0 → x^3 + y^3 + z^3 > 3*x*y*z := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_greater_than_product_sufficient_l459_45917


namespace NUMINAMATH_CALUDE_quadrilateral_classification_l459_45994

/-
  Definitions:
  - a, b, c, d: vectors representing sides AB, BC, CD, DA of quadrilateral ABCD
  - m, n: real numbers
-/

variable (a b c d : ℝ × ℝ)
variable (m n : ℝ)

/-- A quadrilateral is a rectangle if its adjacent sides are perpendicular and opposite sides are equal -/
def is_rectangle (a b c d : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0 ∧
  b.1 * c.1 + b.2 * c.2 = 0 ∧
  c.1 * d.1 + c.2 * d.2 = 0 ∧
  d.1 * a.1 + d.2 * a.2 = 0 ∧
  a.1^2 + a.2^2 = c.1^2 + c.2^2 ∧
  b.1^2 + b.2^2 = d.1^2 + d.2^2

/-- A quadrilateral is an isosceles trapezoid if it has one pair of parallel sides and the other pair of equal length -/
def is_isosceles_trapezoid (a b c d : ℝ × ℝ) : Prop :=
  (a.1 * d.2 - a.2 * d.1 = b.1 * c.2 - b.2 * c.1) ∧
  (a.1^2 + a.2^2 = c.1^2 + c.2^2) ∧
  (a.1 * d.2 - a.2 * d.1 ≠ 0 ∨ b.1 * c.2 - b.2 * c.1 ≠ 0)

theorem quadrilateral_classification (h1 : a.1 * b.1 + a.2 * b.2 = m) 
                                     (h2 : b.1 * c.1 + b.2 * c.2 = m)
                                     (h3 : c.1 * d.1 + c.2 * d.2 = n)
                                     (h4 : d.1 * a.1 + d.2 * a.2 = n) :
  (m = n → is_rectangle a b c d) ∧
  (m ≠ n → is_isosceles_trapezoid a b c d) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_classification_l459_45994


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l459_45913

theorem triangle_angle_measure (A B C : Real) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) :
  C = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l459_45913


namespace NUMINAMATH_CALUDE_reciprocal_of_fraction_difference_l459_45979

theorem reciprocal_of_fraction_difference : 
  (((2 : ℚ) / 5 - (3 : ℚ) / 4)⁻¹ : ℚ) = -(20 : ℚ) / 7 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_fraction_difference_l459_45979


namespace NUMINAMATH_CALUDE_reflection_line_equation_l459_45991

/-- The equation of the reflection line for a triangle. -/
def reflection_line (D E F D' E' F' : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

/-- Theorem stating the equation of the reflection line for the given triangle. -/
theorem reflection_line_equation
  (D : ℝ × ℝ) (E : ℝ × ℝ) (F : ℝ × ℝ)
  (D' : ℝ × ℝ) (E' : ℝ × ℝ) (F' : ℝ × ℝ)
  (hD : D = (1, 2)) (hE : E = (6, 3)) (hF : F = (-3, 4))
  (hD' : D' = (1, -2)) (hE' : E' = (6, -3)) (hF' : F' = (-3, -4)) :
  reflection_line D E F D' E' F' = {p : ℝ × ℝ | p.2 = 0} :=
sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l459_45991


namespace NUMINAMATH_CALUDE_yeri_change_correct_l459_45984

def calculate_change (num_candies : ℕ) (candy_cost : ℕ) (num_chocolates : ℕ) (chocolate_cost : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_candies * candy_cost + num_chocolates * chocolate_cost)

theorem yeri_change_correct : 
  calculate_change 5 120 3 350 2500 = 850 := by
  sorry

end NUMINAMATH_CALUDE_yeri_change_correct_l459_45984


namespace NUMINAMATH_CALUDE_ratio_x_to_y_is_eight_l459_45923

theorem ratio_x_to_y_is_eight (x y : ℝ) (h : y = 0.125 * x) : x / y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_is_eight_l459_45923


namespace NUMINAMATH_CALUDE_sequence_sum_2017_l459_45925

theorem sequence_sum_2017 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → S n = 2 * n - 1) →
  (∀ n : ℕ, n > 0 → S n = S (n - 1) + a n) →
  a 2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_2017_l459_45925


namespace NUMINAMATH_CALUDE_f_composition_of_one_l459_45929

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_composition_of_one (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x + 2) : f (f (f 1)) = 53 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_one_l459_45929


namespace NUMINAMATH_CALUDE_kelly_initial_apples_l459_45964

/-- The number of apples Kelly needs to pick -/
def apples_to_pick : ℕ := 49

/-- The total number of apples Kelly will have after picking -/
def total_apples : ℕ := 105

/-- The initial number of apples Kelly had -/
def initial_apples : ℕ := total_apples - apples_to_pick

theorem kelly_initial_apples :
  initial_apples = 56 :=
sorry

end NUMINAMATH_CALUDE_kelly_initial_apples_l459_45964


namespace NUMINAMATH_CALUDE_fred_initial_cards_l459_45973

/-- The number of baseball cards Keith bought from Fred -/
def cards_bought : ℕ := 22

/-- The number of baseball cards Fred has now -/
def cards_remaining : ℕ := 18

/-- The initial number of baseball cards Fred had -/
def initial_cards : ℕ := cards_bought + cards_remaining

theorem fred_initial_cards : initial_cards = 40 := by
  sorry

end NUMINAMATH_CALUDE_fred_initial_cards_l459_45973


namespace NUMINAMATH_CALUDE_sin_graph_shift_l459_45971

theorem sin_graph_shift (x : ℝ) :
  2 * Real.sin (3 * x - π / 5) = 2 * Real.sin (3 * (x - π / 15)) :=
by sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l459_45971


namespace NUMINAMATH_CALUDE_circle_fits_in_triangle_l459_45921

theorem circle_fits_in_triangle (a b c : ℝ) (S : ℝ) : 
  a = 3 ∧ b = 4 ∧ c = 5 → S = 25 / 8 →
  ∃ (r R : ℝ), r = (a + b - c) / 2 ∧ S = π * R^2 ∧ R < r := by
  sorry

end NUMINAMATH_CALUDE_circle_fits_in_triangle_l459_45921


namespace NUMINAMATH_CALUDE_smallest_multiple_of_2_3_4_5_7_l459_45920

theorem smallest_multiple_of_2_3_4_5_7 : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(2 ∣ m ∧ 3 ∣ m ∧ 4 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m)) ∧ 
  (2 ∣ n ∧ 3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) :=
by
  use 420
  sorry

#eval 420 % 2
#eval 420 % 3
#eval 420 % 4
#eval 420 % 5
#eval 420 % 7

end NUMINAMATH_CALUDE_smallest_multiple_of_2_3_4_5_7_l459_45920


namespace NUMINAMATH_CALUDE_smallest_y_coordinate_l459_45987

theorem smallest_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_coordinate_l459_45987


namespace NUMINAMATH_CALUDE_project_assignment_count_l459_45993

/-- The number of ways to assign projects to teams --/
def assign_projects (total_projects : ℕ) (num_teams : ℕ) (max_for_one_team : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of assignments for the given conditions --/
theorem project_assignment_count :
  assign_projects 5 3 2 = 130 :=
sorry

end NUMINAMATH_CALUDE_project_assignment_count_l459_45993


namespace NUMINAMATH_CALUDE_proposition_1_proposition_3_l459_45958

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the lines and planes
variable (l m n : Line)
variable (α β γ : Plane)

-- Assume the lines and planes are distinct
variable (h_distinct_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
variable (h_distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Proposition 1
theorem proposition_1 :
  parallel α β → contains α l → line_parallel_plane l β :=
by sorry

-- Proposition 3
theorem proposition_3 :
  ¬contains α m → contains α n → line_parallel m n → line_parallel_plane m α :=
by sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_3_l459_45958


namespace NUMINAMATH_CALUDE_rachel_reading_homework_l459_45961

theorem rachel_reading_homework (literature_pages : ℕ) (additional_reading_pages : ℕ) 
  (h1 : literature_pages = 10) 
  (h2 : additional_reading_pages = 6) : 
  literature_pages + additional_reading_pages = 16 := by
  sorry

end NUMINAMATH_CALUDE_rachel_reading_homework_l459_45961


namespace NUMINAMATH_CALUDE_remainder_proof_l459_45945

theorem remainder_proof : (7 * 10^20 + 2^20) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l459_45945


namespace NUMINAMATH_CALUDE_roses_per_girl_l459_45965

/-- Proves that each girl planted 3 roses given the conditions of the problem -/
theorem roses_per_girl (total_students : ℕ) (total_plants : ℕ) (birches : ℕ) 
  (h1 : total_students = 24)
  (h2 : total_plants = 24)
  (h3 : birches = 6)
  (h4 : birches * 3 = total_students - (total_students - birches * 3)) :
  (total_plants - birches) / (total_students - birches * 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_roses_per_girl_l459_45965


namespace NUMINAMATH_CALUDE_regular_polyhedra_coloring_l459_45950

structure RegularPolyhedron where
  edges : ℕ
  vertexDegree : ℕ
  faceEdges : ℕ

def isGoodColoring (p : RegularPolyhedron) (redEdges : ℕ) : Prop :=
  redEdges ≤ p.edges * (p.vertexDegree - 1) / p.vertexDegree

def isCompletelyGoodColoring (p : RegularPolyhedron) (redEdges : ℕ) : Prop :=
  isGoodColoring p redEdges ∧ redEdges < p.edges

def maxGoodColoring (p : RegularPolyhedron) : ℕ :=
  p.edges * (p.vertexDegree - 1) / p.vertexDegree

def maxCompletelyGoodColoring (p : RegularPolyhedron) : ℕ :=
  min (maxGoodColoring p) (p.edges - 1)

def tetrahedron : RegularPolyhedron := ⟨6, 3, 3⟩
def cube : RegularPolyhedron := ⟨12, 3, 4⟩
def octahedron : RegularPolyhedron := ⟨12, 4, 3⟩
def dodecahedron : RegularPolyhedron := ⟨30, 3, 5⟩
def icosahedron : RegularPolyhedron := ⟨30, 5, 3⟩

theorem regular_polyhedra_coloring :
  (maxGoodColoring tetrahedron = maxCompletelyGoodColoring tetrahedron) ∧
  (maxGoodColoring cube = maxCompletelyGoodColoring cube) ∧
  (maxGoodColoring dodecahedron = maxCompletelyGoodColoring dodecahedron) ∧
  (maxGoodColoring octahedron ≠ maxCompletelyGoodColoring octahedron) ∧
  (maxGoodColoring icosahedron ≠ maxCompletelyGoodColoring icosahedron) := by
  sorry

end NUMINAMATH_CALUDE_regular_polyhedra_coloring_l459_45950


namespace NUMINAMATH_CALUDE_yellow_preference_l459_45909

theorem yellow_preference (total_students : ℕ) (total_girls : ℕ) 
  (h1 : total_students = 30)
  (h2 : total_girls = 18)
  (h3 : total_students / 2 = total_students - (total_students / 2 + total_girls / 3)) :
  total_students - (total_students / 2 + total_girls / 3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_yellow_preference_l459_45909


namespace NUMINAMATH_CALUDE_train_length_calculation_l459_45976

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 ∧ 
  crossing_time = 30 ∧ 
  bridge_length = 255 →
  (train_speed * 1000 / 3600) * crossing_time - bridge_length = 120 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l459_45976


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l459_45936

-- Define the triangle and equation conditions
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

def are_equation_roots (a b c : ℝ) : Prop :=
  ∃ (x : ℝ), a * x^2 - b * x + c = 0 ∧ (x = a ∨ x = b)

-- Main theorem
theorem triangle_side_sum_range (a b c : ℝ) :
  is_valid_triangle a b c → are_equation_roots a b c → a < b →
  7/8 < a + b - c ∧ a + b - c < Real.sqrt 5 - 1 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_sum_range_l459_45936


namespace NUMINAMATH_CALUDE_weight_of_calcium_hydride_l459_45995

/-- The atomic weight of calcium in g/mol -/
def Ca_weight : ℝ := 40.08

/-- The atomic weight of hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- The molecular weight of calcium hydride (CaH2) in g/mol -/
def CaH2_weight : ℝ := Ca_weight + 2 * H_weight

/-- The number of moles of calcium hydride -/
def moles : ℝ := 6

/-- Theorem: The weight of 6 moles of calcium hydride (CaH2) is 252.576 grams -/
theorem weight_of_calcium_hydride : moles * CaH2_weight = 252.576 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_calcium_hydride_l459_45995


namespace NUMINAMATH_CALUDE_equation_solutions_l459_45941

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 13*x - 8) + 1 / (x^2 + 3*x - 8) + 1 / (x^2 - 15*x - 8) = 0)} = {8, 1, -1, -8} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l459_45941


namespace NUMINAMATH_CALUDE_zuzka_structure_bounds_l459_45911

/-- A structure made of cubes -/
structure CubeStructure where
  base : Nat
  layers : Nat
  third_layer : Nat
  total : Nat

/-- The conditions of Zuzka's cube structure -/
def zuzka_structure (s : CubeStructure) : Prop :=
  s.base = 16 ∧ 
  s.layers ≥ 3 ∧ 
  s.third_layer = 2 ∧
  s.total = s.base + (s.layers - 1) * s.third_layer + (s.total - s.base - s.third_layer)

/-- The theorem stating the range of possible total cubes -/
theorem zuzka_structure_bounds (s : CubeStructure) :
  zuzka_structure s → 22 ≤ s.total ∧ s.total ≤ 27 :=
by
  sorry


end NUMINAMATH_CALUDE_zuzka_structure_bounds_l459_45911


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l459_45912

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    S_n = (1/2)3^(n+1) - a, prove that a = 3/2 -/
theorem geometric_sequence_sum (n : ℕ) (a_n : ℕ → ℝ) (S : ℕ → ℝ) (a : ℝ) :
  (∀ k, S k = (1/2) * 3^(k+1) - a) →
  (∀ k, a_n (k+1) = S (k+1) - S k) →
  (∀ k, a_n (k+2) * a_n k = (a_n (k+1))^2) →
  a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l459_45912


namespace NUMINAMATH_CALUDE_gcd_factorial_bound_l459_45959

theorem gcd_factorial_bound (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p > q) :
  Nat.gcd (Nat.factorial p - 1) (Nat.factorial q - 1) ≤ p^(5/3) := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_bound_l459_45959


namespace NUMINAMATH_CALUDE_square_sum_from_means_l459_45910

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 18) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 92) : 
  x^2 + y^2 = 1112 := by sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l459_45910


namespace NUMINAMATH_CALUDE_yoojung_initial_candies_l459_45947

/-- The number of candies Yoojung gave to her older sister -/
def candies_to_older_sister : ℕ := 7

/-- The number of candies Yoojung gave to her younger sister -/
def candies_to_younger_sister : ℕ := 6

/-- The number of candies Yoojung had left over -/
def candies_left_over : ℕ := 15

/-- The initial number of candies Yoojung had -/
def initial_candies : ℕ := candies_to_older_sister + candies_to_younger_sister + candies_left_over

theorem yoojung_initial_candies : initial_candies = 28 := by
  sorry

end NUMINAMATH_CALUDE_yoojung_initial_candies_l459_45947


namespace NUMINAMATH_CALUDE_fifteen_point_seven_billion_in_scientific_notation_l459_45978

-- Define the number of billions
def billions : ℝ := 15.7

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.57 * (10 ^ 9)

-- Theorem statement
theorem fifteen_point_seven_billion_in_scientific_notation :
  billions * (10 ^ 9) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_fifteen_point_seven_billion_in_scientific_notation_l459_45978


namespace NUMINAMATH_CALUDE_inequality_proof_l459_45930

theorem inequality_proof (x m : ℝ) (a b c : ℝ) :
  (∀ x, |x - 3| + |x - m| ≥ 2*m) →
  a > 0 → b > 0 → c > 0 → a + b + c = 1 →
  (∃ m_max : ℝ, m_max = 1 ∧ 
    (∀ m', (∀ x, |x - 3| + |x - m'| ≥ 2*m') → m' ≤ m_max)) ∧
  (4*a^2 + 9*b^2 + c^2 ≥ 36/49) ∧
  (4*a^2 + 9*b^2 + c^2 = 36/49 ↔ a = 9/49 ∧ b = 4/49 ∧ c = 36/49) :=
by sorry


end NUMINAMATH_CALUDE_inequality_proof_l459_45930


namespace NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l459_45932

theorem sum_reciprocal_lower_bound (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0) 
  (h_sum : a₁ + a₂ + a₃ + a₄ = 1) : 
  1/a₁ + 1/a₂ + 1/a₃ + 1/a₄ ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l459_45932


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l459_45926

theorem smallest_prime_divisor_of_sum : ∃ (p : ℕ), p.Prime ∧ p ∣ (3^19 + 11^23) ∧ ∀ (q : ℕ), q.Prime → q ∣ (3^19 + 11^23) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l459_45926


namespace NUMINAMATH_CALUDE_product_213_16_l459_45948

theorem product_213_16 : 213 * 16 = 3408 := by
  sorry

end NUMINAMATH_CALUDE_product_213_16_l459_45948


namespace NUMINAMATH_CALUDE_problem_solution_l459_45918

theorem problem_solution (x y : ℝ) 
  (h1 : 5 + x = 3 - y) 
  (h2 : 2 + y = 6 + x) : 
  5 - x = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l459_45918


namespace NUMINAMATH_CALUDE_number_of_bs_l459_45955

/-- Represents the number of students earning each grade in a philosophy class. -/
structure GradeDistribution where
  total : ℕ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the grade distribution satisfies the given conditions. -/
def isValidDistribution (g : GradeDistribution) : Prop :=
  g.total = 40 ∧
  g.a = 0.5 * g.b ∧
  g.c = 2 * g.b ∧
  g.a + g.b + g.c = g.total

/-- Theorem stating the number of B's in the class. -/
theorem number_of_bs (g : GradeDistribution) 
  (h : isValidDistribution g) : g.b = 40 / 3.5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_bs_l459_45955


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l459_45904

/-- Given a geometric sequence {a_n}, prove that if a_1 + a_3 = 20 and a_2 + a_4 = 40, then a_3 + a_5 = 80 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_sum1 : a 1 + a 3 = 20) (h_sum2 : a 2 + a 4 = 40) : 
  a 3 + a 5 = 80 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l459_45904


namespace NUMINAMATH_CALUDE_one_tetrahedron_formed_l459_45999

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the set of available triangles -/
def AvailableTriangles : Finset Triangle := sorry

/-- Checks if a set of four triangles can form a tetrahedron -/
def CanFormTetrahedron (t1 t2 t3 t4 : Triangle) : Prop := sorry

/-- Counts the number of tetrahedrons that can be formed -/
def CountTetrahedrons (triangles : Finset Triangle) : ℕ := sorry

/-- The main theorem stating that exactly one tetrahedron can be formed -/
theorem one_tetrahedron_formed :
  CountTetrahedrons AvailableTriangles = 1 := by sorry

end NUMINAMATH_CALUDE_one_tetrahedron_formed_l459_45999


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l459_45989

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 3 + 3 * a 8 + a 13 = 120 → a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l459_45989


namespace NUMINAMATH_CALUDE_jack_apples_l459_45992

/-- Calculates the remaining apples after a series of sales and a gift --/
def remaining_apples (initial : ℕ) (sale1_percent : ℕ) (sale2_percent : ℕ) (sale3_percent : ℕ) (gift : ℕ) : ℕ :=
  let after_sale1 := initial - initial * sale1_percent / 100
  let after_sale2 := after_sale1 - after_sale1 * sale2_percent / 100
  let after_sale3 := after_sale2 - (after_sale2 * sale3_percent / 100)
  after_sale3 - gift

/-- Theorem stating that given the specific conditions, Jack ends up with 75 apples --/
theorem jack_apples : remaining_apples 150 30 20 10 1 = 75 := by
  sorry

end NUMINAMATH_CALUDE_jack_apples_l459_45992


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l459_45980

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 7*x - 1

-- Define the roots
noncomputable def a : ℝ := 3 + Real.sqrt 8
noncomputable def b : ℝ := 3 - Real.sqrt 8

-- Theorem statement
theorem root_sum_reciprocal : 
  f a = 0 ∧ f b = 0 ∧ (∀ x, f x = 0 → b ≤ x ∧ x ≤ a) → a / b + b / a = 34 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l459_45980


namespace NUMINAMATH_CALUDE_sphere_roll_coplanar_l459_45969

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a rectangular box -/
structure RectangularBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the transformation of a point on a sphere's surface after rolling -/
def sphereRoll (s : Sphere) (b : RectangularBox) (p : Point3D) : Point3D :=
  sorry

/-- States that four points lie in the same plane -/
def coplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem sphere_roll_coplanar (s : Sphere) (b : RectangularBox) (X : Point3D) :
  let X₁ := sphereRoll s b X
  let X₂ := sphereRoll s b X₁
  let X₃ := sphereRoll s b X₂
  coplanar X X₁ X₂ X₃ :=
sorry

end NUMINAMATH_CALUDE_sphere_roll_coplanar_l459_45969


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l459_45952

theorem ratio_equation_solution (a b c : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 15 - 4 * a - c)
  (h3 : c = a + 2) :
  a = 13 / 9 := by
sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l459_45952


namespace NUMINAMATH_CALUDE_curve_S_properties_l459_45951

-- Define the curve S
def S (x : ℝ) : ℝ := x^3 - 6*x^2 - x + 6

-- Define the derivative of S
def S' (x : ℝ) : ℝ := 3*x^2 - 12*x - 1

-- Define the point P
def P : ℝ × ℝ := (2, -12)

theorem curve_S_properties :
  -- 1. P is the point where the tangent line has the smallest slope
  (∀ x : ℝ, S' P.1 ≤ S' x) ∧
  -- 2. S is symmetric about P
  (∀ x : ℝ, S (P.1 + x) - P.2 = -(S (P.1 - x) - P.2)) :=
sorry

end NUMINAMATH_CALUDE_curve_S_properties_l459_45951


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l459_45983

theorem sqrt_sum_problem (x : ℝ) (h_pos : x > 0) (h_eq : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l459_45983
