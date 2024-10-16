import Mathlib

namespace NUMINAMATH_CALUDE_point_p_coordinates_l2373_237391

/-- Given a linear function and points A and P, proves that P satisfies the conditions of the problem -/
theorem point_p_coordinates (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => -3/2 * t - 3
  let A : ℝ × ℝ := (-5, 0)
  let B : ℝ × ℝ := (f⁻¹ 0, 0)
  let P : ℝ × ℝ := (x, y)
  (P.2 = f P.1) →  -- P lies on the linear function
  (P ≠ B) →        -- P does not coincide with B
  (abs ((A.1 - B.1) * P.2) / 2 = 6) →  -- Area of triangle ABP is 6
  ((x = -14/3 ∧ y = 4) ∨ (x = 2/3 ∧ y = -4)) :=
by sorry

end NUMINAMATH_CALUDE_point_p_coordinates_l2373_237391


namespace NUMINAMATH_CALUDE_problem_statement_l2373_237337

theorem problem_statement (a b : ℝ) (h : |a - 1| + Real.sqrt (b + 2) = 0) : 
  (a + b) ^ 2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2373_237337


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2373_237385

/-- Given two vectors a and b in ℝ², if a is parallel to b, then the magnitude of b is 2√5. -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, 2) → 
  b.1 = -2 → 
  ∃ (t : ℝ), a = t • b → 
  ‖b‖ = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2373_237385


namespace NUMINAMATH_CALUDE_second_polygon_sides_l2373_237316

/-- Given two regular polygons with equal perimeters, where one polygon has 50 sides
    and each of its sides is three times as long as each side of the other polygon,
    the number of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 → n > 0 → 50 * (3 * s) = n * s → n = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l2373_237316


namespace NUMINAMATH_CALUDE_kiera_envelopes_l2373_237360

theorem kiera_envelopes (blue : ℕ) (yellow : ℕ) (green : ℕ) 
  (h1 : blue = 14)
  (h2 : yellow = blue - 6)
  (h3 : green = 3 * yellow) :
  blue + yellow + green = 46 := by
  sorry

end NUMINAMATH_CALUDE_kiera_envelopes_l2373_237360


namespace NUMINAMATH_CALUDE_circle_ratio_theorem_l2373_237300

noncomputable def circle_ratio : ℝ :=
  let r1 : ℝ := Real.sqrt 2
  let r2 : ℝ := 2
  let d : ℝ := Real.sqrt 3 + 1
  let common_area : ℝ := (7 * Real.pi - 6 * (Real.sqrt 3 + 1)) / 6
  let inscribed_radius : ℝ := (Real.sqrt 2 + 1 - Real.sqrt 3) / 2
  let inscribed_area : ℝ := Real.pi * inscribed_radius ^ 2
  inscribed_area / common_area

theorem circle_ratio_theorem : circle_ratio = 
  (3 * Real.pi * (3 + Real.sqrt 2 - Real.sqrt 3 - Real.sqrt 6)) / 
  (7 * Real.pi - 6 * (Real.sqrt 3 + 1)) := by sorry

end NUMINAMATH_CALUDE_circle_ratio_theorem_l2373_237300


namespace NUMINAMATH_CALUDE_inequality_solution_l2373_237312

-- Define the function f(x) = 1/√(x+1)
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x + 1)

-- Define the solution set
def solution_set : Set ℝ := Set.Icc 0 (1/2)

-- Define the inequality
def inequality (l k x : ℝ) : Prop := 
  1 - l * x ≤ f x ∧ f x ≤ 1 - k * x

-- Theorem statement
theorem inequality_solution (l k : ℝ) : 
  (∀ x ∈ solution_set, inequality l k x) ↔ (l = 1/2 ∧ k = 2 - 2 * Real.sqrt 6 / 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2373_237312


namespace NUMINAMATH_CALUDE_range_of_m_l2373_237334

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |3 - x| + |5 + x| > m) ↔ m < 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2373_237334


namespace NUMINAMATH_CALUDE_max_volume_rect_frame_l2373_237380

/-- Represents the dimensions of a rectangular frame. -/
structure RectFrame where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular frame. -/
def volume (frame : RectFrame) : ℝ :=
  frame.length * frame.width * frame.height

/-- Calculates the perimeter of the base of a rectangular frame. -/
def basePerimeter (frame : RectFrame) : ℝ :=
  2 * (frame.length + frame.width)

/-- Calculates the total length of steel bar used for a rectangular frame. -/
def totalBarLength (frame : RectFrame) : ℝ :=
  basePerimeter frame + 4 * frame.height

/-- Theorem: The maximum volume of a rectangular frame enclosed by an 18m steel bar,
    where the ratio of length to width is 2:1, is equal to the correct maximum volume. -/
theorem max_volume_rect_frame :
  ∃ (frame : RectFrame),
    frame.length = 2 * frame.width ∧
    totalBarLength frame = 18 ∧
    ∀ (other : RectFrame),
      other.length = 2 * other.width →
      totalBarLength other = 18 →
      volume frame ≥ volume other :=
by sorry


end NUMINAMATH_CALUDE_max_volume_rect_frame_l2373_237380


namespace NUMINAMATH_CALUDE_committee_meeting_arrangements_l2373_237303

/-- Represents a school in the club --/
structure School :=
  (members : Nat)

/-- Represents the club with its schools --/
structure Club :=
  (schools : List School)
  (total_members : Nat)

/-- Represents the committee meeting arrangement --/
structure CommitteeMeeting :=
  (host : School)
  (first_non_host : School)
  (second_non_host : School)
  (host_reps : Nat)
  (first_non_host_reps : Nat)
  (second_non_host_reps : Nat)

/-- The number of ways to arrange a committee meeting --/
def arrange_committee_meeting (club : Club) : Nat :=
  sorry

/-- Theorem stating the number of possible committee meeting arrangements --/
theorem committee_meeting_arrangements (club : Club) :
  club.schools.length = 3 ∧
  club.total_members = 18 ∧
  (∀ s ∈ club.schools, s.members = 6) →
  arrange_committee_meeting club = 5400 :=
sorry

end NUMINAMATH_CALUDE_committee_meeting_arrangements_l2373_237303


namespace NUMINAMATH_CALUDE_cyclist_time_is_pi_over_five_l2373_237355

/-- Represents the problem of a cyclist riding on a highway strip -/
def CyclistProblem (width : ℝ) (length : ℝ) (large_semicircle_distance : ℝ) (speed : ℝ) : Prop :=
  width = 40 ∧ 
  length = 5280 ∧ 
  large_semicircle_distance = 528 ∧ 
  speed = 5

/-- Calculates the time taken for the cyclist to cover the entire strip -/
noncomputable def cycleTime (width : ℝ) (length : ℝ) (large_semicircle_distance : ℝ) (speed : ℝ) : ℝ :=
  (Real.pi * length) / (speed * width)

/-- Theorem stating that the time taken is π/5 hours -/
theorem cyclist_time_is_pi_over_five 
  (width : ℝ) (length : ℝ) (large_semicircle_distance : ℝ) (speed : ℝ) 
  (h : CyclistProblem width length large_semicircle_distance speed) : 
  cycleTime width length large_semicircle_distance speed = Real.pi / 5 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_time_is_pi_over_five_l2373_237355


namespace NUMINAMATH_CALUDE_golden_ratio_properties_l2373_237362

theorem golden_ratio_properties (x y : ℝ) 
  (hx : x^2 = x + 1) 
  (hy : y^2 = y + 1) 
  (hxy : x ≠ y) : 
  (x + y = 1) ∧ (x^5 + y^5 = 11) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_properties_l2373_237362


namespace NUMINAMATH_CALUDE_toms_weekly_distance_l2373_237356

/-- Represents Tom's weekly exercise schedule --/
structure ExerciseSchedule where
  monday_run_morning : Real
  monday_run_evening : Real
  wednesday_run_morning : Real
  wednesday_run_evening : Real
  friday_run_first : Real
  friday_run_second : Real
  friday_run_third : Real
  tuesday_cycle_morning : Real
  tuesday_cycle_evening : Real
  thursday_cycle_morning : Real
  thursday_cycle_evening : Real

/-- Calculates the total distance Tom runs and cycles in a week --/
def total_distance (schedule : ExerciseSchedule) : Real :=
  schedule.monday_run_morning + schedule.monday_run_evening +
  schedule.wednesday_run_morning + schedule.wednesday_run_evening +
  schedule.friday_run_first + schedule.friday_run_second + schedule.friday_run_third +
  schedule.tuesday_cycle_morning + schedule.tuesday_cycle_evening +
  schedule.thursday_cycle_morning + schedule.thursday_cycle_evening

/-- Tom's actual exercise schedule --/
def toms_schedule : ExerciseSchedule :=
  { monday_run_morning := 6
  , monday_run_evening := 4
  , wednesday_run_morning := 5.25
  , wednesday_run_evening := 5
  , friday_run_first := 3
  , friday_run_second := 4.5
  , friday_run_third := 2
  , tuesday_cycle_morning := 10
  , tuesday_cycle_evening := 8
  , thursday_cycle_morning := 7
  , thursday_cycle_evening := 12
  }

/-- Theorem stating that Tom's total weekly distance is 66.75 miles --/
theorem toms_weekly_distance : total_distance toms_schedule = 66.75 := by
  sorry


end NUMINAMATH_CALUDE_toms_weekly_distance_l2373_237356


namespace NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_n_l2373_237373

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Statement of the theorem
theorem exists_fibonacci_divisible_by_n (n : ℕ) (hn : n > 0) : 
  ∃ m : ℕ, m > 0 ∧ n ∣ fib m :=
sorry

end NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_n_l2373_237373


namespace NUMINAMATH_CALUDE_facebook_group_messages_l2373_237377

/-- Calculates the total number of messages sent in a week by remaining members of a Facebook group after some members were removed. -/
theorem facebook_group_messages
  (initial_members : ℕ)
  (removed_members : ℕ)
  (messages_per_day : ℕ)
  (days_in_week : ℕ)
  (h1 : initial_members = 150)
  (h2 : removed_members = 20)
  (h3 : messages_per_day = 50)
  (h4 : days_in_week = 7)
  : (initial_members - removed_members) * messages_per_day * days_in_week = 45500 :=
by sorry

end NUMINAMATH_CALUDE_facebook_group_messages_l2373_237377


namespace NUMINAMATH_CALUDE_timmy_calories_needed_l2373_237398

/-- Represents the number of calories in an orange -/
def calories_per_orange : ℕ := 80

/-- Represents the cost of an orange in cents -/
def cost_per_orange : ℕ := 120

/-- Represents Timmy's initial amount of money in cents -/
def initial_money : ℕ := 1000

/-- Represents the amount of money Timmy has left after buying oranges in cents -/
def money_left : ℕ := 400

/-- Calculates the number of calories Timmy needs to get -/
def calories_needed : ℕ := 
  ((initial_money - money_left) / cost_per_orange) * calories_per_orange

theorem timmy_calories_needed : calories_needed = 400 := by
  sorry

end NUMINAMATH_CALUDE_timmy_calories_needed_l2373_237398


namespace NUMINAMATH_CALUDE_complex_square_fourth_quadrant_l2373_237336

theorem complex_square_fourth_quadrant :
  let z : ℂ := 2 - I
  let w : ℂ := z^2
  (w.re > 0) ∧ (w.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_square_fourth_quadrant_l2373_237336


namespace NUMINAMATH_CALUDE_base_of_equation_l2373_237379

theorem base_of_equation (e : ℕ) (h : e = 35) :
  ∃ b : ℚ, b^e * (1/4)^18 = 1/(2*(10^35)) ∧ b = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_base_of_equation_l2373_237379


namespace NUMINAMATH_CALUDE_total_glue_blobs_is_96_l2373_237305

/-- Represents a layer in the pyramid --/
structure Layer where
  size : Nat
  deriving Repr

/-- Calculates the number of internal glue blobs within a layer --/
def internalGlueBlobs (layer : Layer) : Nat :=
  2 * layer.size * (layer.size - 1)

/-- Calculates the number of glue blobs between two adjacent layers --/
def interlayerGlueBlobs (upper : Layer) (lower : Layer) : Nat :=
  upper.size * upper.size * 4

/-- The pyramid structure --/
def pyramid : List Layer := [
  { size := 4 },
  { size := 3 },
  { size := 2 },
  { size := 1 }
]

/-- Theorem: The total number of glue blobs in the pyramid is 96 --/
theorem total_glue_blobs_is_96 : 
  (pyramid.map internalGlueBlobs).sum + 
  (List.zipWith interlayerGlueBlobs pyramid.tail pyramid).sum = 96 := by
  sorry

#eval (pyramid.map internalGlueBlobs).sum + 
      (List.zipWith interlayerGlueBlobs pyramid.tail pyramid).sum

end NUMINAMATH_CALUDE_total_glue_blobs_is_96_l2373_237305


namespace NUMINAMATH_CALUDE_unique_pair_solution_l2373_237335

theorem unique_pair_solution (m n : ℕ) (h1 : m < n) 
  (h2 : ∃ k : ℕ, m^2 + 1 = k * n) (h3 : ∃ l : ℕ, n^2 + 1 = l * m) : 
  m = 1 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_solution_l2373_237335


namespace NUMINAMATH_CALUDE_sum_of_xyz_equals_six_l2373_237386

theorem sum_of_xyz_equals_six (a b : ℝ) (x y z : ℤ) : 
  a^2 = 9/36 → 
  b^2 = (1 + Real.sqrt 3)^2 / 8 → 
  a < 0 → 
  b > 0 → 
  (a - b)^2 = (x : ℝ) * Real.sqrt y / z → 
  x + y + z = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_xyz_equals_six_l2373_237386


namespace NUMINAMATH_CALUDE_infinite_special_numbers_l2373_237368

theorem infinite_special_numbers :
  ∃ (seq : ℕ → ℕ), 
    (∀ i, ∃ n, seq i = n) ∧
    (∀ i j, i < j → seq i < seq j) ∧
    (∀ i, ∀ p : ℕ, Prime p → p ∣ (seq i)^2 + 3 →
      ∃ k : ℕ, k^2 < seq i ∧ p ∣ k^2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_infinite_special_numbers_l2373_237368


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l2373_237381

theorem max_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hax : a^x = 6) (hby : b^y = 6) 
  (hab : a + b = 2 * Real.sqrt 6) : 
  (∀ x' y' a' b' : ℝ, 
    a' > 1 → b' > 1 → 
    a'^x' = 6 → b'^y' = 6 → 
    a' + b' = 2 * Real.sqrt 6 → 
    1/x + 1/y ≥ 1/x' + 1/y') ∧ 
  (∃ x₀ y₀ a₀ b₀ : ℝ, 
    a₀ > 1 ∧ b₀ > 1 ∧ 
    a₀^x₀ = 6 ∧ b₀^y₀ = 6 ∧ 
    a₀ + b₀ = 2 * Real.sqrt 6 ∧ 
    1/x₀ + 1/y₀ = 1) :=
by sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l2373_237381


namespace NUMINAMATH_CALUDE_exponential_monotonicity_l2373_237354

theorem exponential_monotonicity (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > 1) : 
  c^a > c^b := by
  sorry

end NUMINAMATH_CALUDE_exponential_monotonicity_l2373_237354


namespace NUMINAMATH_CALUDE_mystery_number_proof_l2373_237346

theorem mystery_number_proof : ∃ x : ℝ, x * 6 = 72 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_mystery_number_proof_l2373_237346


namespace NUMINAMATH_CALUDE_train_problem_l2373_237370

/-- The speed of the freight train in km/h given the conditions of the train problem -/
def freight_train_speed : ℝ := by sorry

theorem train_problem (passenger_length freight_length : ℝ) (passing_time : ℝ) (speed_ratio : ℚ) :
  passenger_length = 200 →
  freight_length = 280 →
  passing_time = 18 →
  speed_ratio = 5 / 3 →
  freight_train_speed = 36 := by sorry

end NUMINAMATH_CALUDE_train_problem_l2373_237370


namespace NUMINAMATH_CALUDE_client_phones_dropped_off_kevins_phone_repair_problem_l2373_237331

theorem client_phones_dropped_off (initial_phones : ℕ) (repaired_phones : ℕ) (phones_per_person : ℕ) : ℕ :=
  let remaining_phones := initial_phones - repaired_phones
  let total_phones_to_repair := 2 * phones_per_person
  total_phones_to_repair - remaining_phones

theorem kevins_phone_repair_problem :
  client_phones_dropped_off 15 3 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_client_phones_dropped_off_kevins_phone_repair_problem_l2373_237331


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_sequence_l2373_237366

def integer_sequence : List Int := List.range 10 |>.map (λ x => x - 5)

theorem arithmetic_mean_of_sequence (seq : List Int := integer_sequence) :
  seq.sum / seq.length = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_sequence_l2373_237366


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2373_237342

/-- Configuration of triangles ABC and CDE --/
structure TriangleConfig where
  -- Angles in triangle ABC
  angle_A : ℝ
  angle_B : ℝ
  -- Angle y in triangle CDE
  angle_y : ℝ
  -- Assertions about the configuration
  angle_A_eq : angle_A = 50
  angle_B_eq : angle_B = 70
  right_angle_E : True  -- Represents the right angle at E
  angle_C_eq : True  -- Represents that angle at C is same in both triangles

/-- Theorem stating that in the given configuration, y = 30° --/
theorem triangle_angle_calculation (config : TriangleConfig) : config.angle_y = 30 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_calculation_l2373_237342


namespace NUMINAMATH_CALUDE_two_triangle_range_l2373_237382

theorem two_triangle_range (A B C : ℝ) (a b c : ℝ) :
  A = Real.pi / 3 →  -- 60 degrees in radians
  a = Real.sqrt 3 →
  b = x →
  (∃ (x : ℝ), ∀ B, 
    Real.pi / 3 < B ∧ B < 2 * Real.pi / 3 →  -- 60° < B < 120°
    Real.sin B = x / 2 →
    x > Real.sqrt 3 ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_two_triangle_range_l2373_237382


namespace NUMINAMATH_CALUDE_sports_club_members_l2373_237394

/-- A sports club with members who play badminton, tennis, both, or neither. -/
structure SportsClub where
  badminton : ℕ  -- Number of members who play badminton
  tennis : ℕ     -- Number of members who play tennis
  both : ℕ       -- Number of members who play both badminton and tennis
  neither : ℕ    -- Number of members who play neither badminton nor tennis

/-- The total number of members in the sports club -/
def SportsClub.totalMembers (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - club.both + club.neither

/-- Theorem stating that the total number of members in the given sports club is 35 -/
theorem sports_club_members (club : SportsClub)
    (h1 : club.badminton = 15)
    (h2 : club.tennis = 18)
    (h3 : club.neither = 5)
    (h4 : club.both = 3) :
    club.totalMembers = 35 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l2373_237394


namespace NUMINAMATH_CALUDE_area_bounded_by_parabola_and_x_axis_l2373_237378

/-- The area of the figure bounded by y = 2x - x^2 and y = 0 is 4/3 square units. -/
theorem area_bounded_by_parabola_and_x_axis : 
  let f (x : ℝ) := 2 * x - x^2
  ∫ x in (0)..(2), max 0 (f x) = 4/3 := by sorry

end NUMINAMATH_CALUDE_area_bounded_by_parabola_and_x_axis_l2373_237378


namespace NUMINAMATH_CALUDE_four_integer_average_l2373_237365

theorem four_integer_average (a b c d : ℕ+) : 
  (a + b : ℚ) / 2 = 35 →
  c ≤ 130 →
  d ≤ 130 →
  (a + b + c + d : ℚ) / 4 = 50.25 :=
by sorry

end NUMINAMATH_CALUDE_four_integer_average_l2373_237365


namespace NUMINAMATH_CALUDE_tony_purchase_cost_l2373_237318

/-- Calculates the total cost of Tony's purchases given the specified conditions --/
def total_cost (lego_price : ℝ) (sword_price_eur : ℝ) (dough_price_gbp : ℝ)
                (day1_discount : ℝ) (day2_discount : ℝ) (sales_tax : ℝ)
                (eur_to_usd_day1 : ℝ) (gbp_to_usd_day1 : ℝ)
                (eur_to_usd_day2 : ℝ) (gbp_to_usd_day2 : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the total cost is $1560.83 given the problem conditions --/
theorem tony_purchase_cost :
  let lego_price := 250
  let sword_price_eur := 100
  let dough_price_gbp := 30
  let day1_discount := 0.2
  let day2_discount := 0.1
  let sales_tax := 0.05
  let eur_to_usd_day1 := 1 / 0.85
  let gbp_to_usd_day1 := 1 / 0.75
  let eur_to_usd_day2 := 1 / 0.84
  let gbp_to_usd_day2 := 1 / 0.74
  total_cost lego_price sword_price_eur dough_price_gbp
             day1_discount day2_discount sales_tax
             eur_to_usd_day1 gbp_to_usd_day1
             eur_to_usd_day2 gbp_to_usd_day2 = 1560.83 :=
by
  sorry

end NUMINAMATH_CALUDE_tony_purchase_cost_l2373_237318


namespace NUMINAMATH_CALUDE_parallelogram_exists_l2373_237364

/-- Represents a cell in the grid -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents the grid and its blue cells -/
structure Grid where
  n : Nat
  blue_cells : Finset Cell

/-- Predicate to check if four cells form a parallelogram -/
def is_parallelogram (c1 c2 c3 c4 : Cell) : Prop :=
  (c2.x - c1.x = c4.x - c3.x) ∧ (c2.y - c1.y = c4.y - c3.y)

/-- Main theorem: In an n x n grid with 2n blue cells, there exist 4 blue cells forming a parallelogram -/
theorem parallelogram_exists (g : Grid) (h1 : g.blue_cells.card = 2 * g.n) :
  ∃ c1 c2 c3 c4 : Cell, c1 ∈ g.blue_cells ∧ c2 ∈ g.blue_cells ∧ c3 ∈ g.blue_cells ∧ c4 ∈ g.blue_cells ∧
    is_parallelogram c1 c2 c3 c4 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_exists_l2373_237364


namespace NUMINAMATH_CALUDE_coefficient_is_negative_seven_l2373_237320

-- Define the expression
def expression (x : ℝ) : ℝ := 5 * (x - 6) + 6 * (9 - 3 * x^2 + 3 * x) - 10 * (3 * x - 2)

-- Define the coefficient of x
def coefficient_of_x (f : ℝ → ℝ) : ℝ :=
  (f 1 - f 0)

-- Theorem statement
theorem coefficient_is_negative_seven :
  coefficient_of_x expression = -7 := by sorry

end NUMINAMATH_CALUDE_coefficient_is_negative_seven_l2373_237320


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l2373_237310

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ m n : ℤ, x^2 + b*x + 2023 = (x + m) * (x + n)) → b ≥ 136 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l2373_237310


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2373_237350

theorem trigonometric_identity (α : ℝ) : 
  4 * Real.sin (2 * α - 3/2 * Real.pi) * Real.sin (Real.pi/6 + 2 * α) * Real.sin (Real.pi/6 - 2 * α) = Real.cos (6 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2373_237350


namespace NUMINAMATH_CALUDE_heat_engine_efficiencies_l2373_237387

/-- Heat engine efficiencies problem -/
theorem heat_engine_efficiencies
  (η₀ η₁ η₂ Q₁₂ Q₁₃ Q₃₄ α : ℝ)
  (h₀ : η₀ = 1 - Q₃₄ / Q₁₂)
  (h₁ : η₁ = 1 - Q₁₃ / Q₁₂)
  (h₂ : η₂ = 1 - Q₃₄ / Q₁₃)
  (h₃ : η₂ = (η₀ - η₁) / (1 - η₁))
  (h₄ : η₁ < η₀)
  (h₅ : η₂ < η₀)
  (h₆ : η₀ < 1)
  (h₇ : η₁ < 1)
  (h₈ : η₁ = (1 - 0.01 * α) * η₀) :
  η₂ = α / (100 - (100 - α) * η₀) := by
  sorry

end NUMINAMATH_CALUDE_heat_engine_efficiencies_l2373_237387


namespace NUMINAMATH_CALUDE_factorial_divisibility_l2373_237395

theorem factorial_divisibility (n : ℕ) (M : ℕ) (h : Nat.factorial 100 = 12^n * M) 
  (h_max : ∀ k : ℕ, Nat.factorial 100 = 12^k * M → k ≤ n) : 
  (2 ∣ M) ∧ ¬(3 ∣ M) := by
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l2373_237395


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l2373_237390

theorem min_value_of_sum_of_roots (x : ℝ) :
  ∃ (y : ℝ), ∀ (x : ℝ),
    Real.sqrt (5 * x^2 - 16 * x + 16) + Real.sqrt (5 * x^2 - 18 * x + 29) ≥ y ∧
    ∃ (z : ℝ), Real.sqrt (5 * z^2 - 16 * z + 16) + Real.sqrt (5 * z^2 - 18 * z + 29) = y :=
by
  use Real.sqrt 29
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l2373_237390


namespace NUMINAMATH_CALUDE_pairwise_disjoint_sequences_l2373_237323

def largest_prime_power_divisor (n : ℕ) : ℕ := sorry

theorem pairwise_disjoint_sequences 
  (n : Fin 10000 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → n i ≠ n j) 
  (h_distinct_lpd : ∀ i j, i ≠ j → 
    largest_prime_power_divisor (n i) ≠ largest_prime_power_divisor (n j)) :
  ∃ a : Fin 10000 → ℤ, ∀ i j k l, i ≠ j → 
    a i + k * n i ≠ a j + l * n j :=
sorry

end NUMINAMATH_CALUDE_pairwise_disjoint_sequences_l2373_237323


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2373_237317

theorem quadratic_root_value (a : ℝ) : (1 : ℝ)^2 + a * 1 + 4 = 0 → a = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2373_237317


namespace NUMINAMATH_CALUDE_rhonda_marbles_l2373_237311

/-- Given that Amon and Rhonda have a total of 215 marbles, and Amon has 55 more marbles than Rhonda,
    prove that Rhonda has 80 marbles. -/
theorem rhonda_marbles (total : ℕ) (difference : ℕ) (rhonda : ℕ) : 
  total = 215 → difference = 55 → total = rhonda + (rhonda + difference) → rhonda = 80 := by
  sorry

end NUMINAMATH_CALUDE_rhonda_marbles_l2373_237311


namespace NUMINAMATH_CALUDE_range_of_P_l2373_237375

theorem range_of_P (x y : ℝ) (h : x^2/3 + y^2 = 1) :
  2 ≤ |2*x + y - 4| + |4 - x - 2*y| ∧ |2*x + y - 4| + |4 - x - 2*y| ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_range_of_P_l2373_237375


namespace NUMINAMATH_CALUDE_function_properties_l2373_237324

def f (x m : ℝ) : ℝ := |x - m| - |x + 3 * m|

theorem function_properties (m : ℝ) (h : m > 0) :
  (m = 1 → {x : ℝ | f x m ≥ 1} = {x : ℝ | x ≤ -3/2}) ∧
  ({m : ℝ | ∀ x t : ℝ, f x m < |2 + t| + |t - 1|} = {m : ℝ | 0 < m ∧ m < 3/4}) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2373_237324


namespace NUMINAMATH_CALUDE_max_product_sum_2024_l2373_237348

theorem max_product_sum_2024 : 
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144) ∧ 
  (∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2024_l2373_237348


namespace NUMINAMATH_CALUDE_volume_of_large_cube_l2373_237338

/-- Given a cube with surface area 96 cm², prove that 8 such cubes form a larger cube with volume 512 cm³ -/
theorem volume_of_large_cube (small_cube : Real → Real → Real → Real) 
  (h1 : ∀ x, small_cube x x x = 96) -- surface area of small cube is 96
  (h2 : ∀ x y z, small_cube x y z = 6 * x * y) -- definition of surface area for a cube
  (large_cube : Real → Real → Real → Real)
  (h3 : ∀ x, large_cube x x x = 8 * small_cube (x/2) (x/2) (x/2)) -- large cube is made of 8 small cubes
  : ∃ x, large_cube x x x = 512 :=
sorry

end NUMINAMATH_CALUDE_volume_of_large_cube_l2373_237338


namespace NUMINAMATH_CALUDE_additive_is_odd_l2373_237389

/-- A function satisfying the given additive property -/
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- Theorem stating that an additive function is odd -/
theorem additive_is_odd (f : ℝ → ℝ) (h : is_additive f) : 
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_additive_is_odd_l2373_237389


namespace NUMINAMATH_CALUDE_system_solution_unique_l2373_237309

theorem system_solution_unique (x y : ℚ) : 
  (2 * x - 3 * y = 1) ∧ ((2 + x) / 3 = (y + 1) / 4) ↔ (x = -3 ∧ y = -7/3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2373_237309


namespace NUMINAMATH_CALUDE_apple_percentage_after_removal_l2373_237351

/-- Represents a bowl of fruit with apples and oranges -/
structure FruitBowl where
  apples : ℕ
  oranges : ℕ

/-- Calculates the percentage of apples in a fruit bowl -/
def applePercentage (bowl : FruitBowl) : ℚ :=
  (bowl.apples : ℚ) / ((bowl.apples + bowl.oranges) : ℚ) * 100

theorem apple_percentage_after_removal :
  let initialBowl : FruitBowl := { apples := 12, oranges := 23 }
  let removedOranges : ℕ := 15
  let finalBowl : FruitBowl := { apples := initialBowl.apples, oranges := initialBowl.oranges - removedOranges }
  applePercentage finalBowl = 60 := by
  sorry

end NUMINAMATH_CALUDE_apple_percentage_after_removal_l2373_237351


namespace NUMINAMATH_CALUDE_functional_equation_equivalence_l2373_237332

theorem functional_equation_equivalence (f : ℝ → ℝ) :
  (∀ x y, f (x + y) = f x + f y) ↔ (∀ x y, f (x + y + x * y) = f x + f y + f (x * y)) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_equivalence_l2373_237332


namespace NUMINAMATH_CALUDE_system_solution_l2373_237396

-- Define the system of equations
def equation1 (x y : ℚ) : Prop := (2 * x - 3) / (3 * x - y) = 3 / 5
def equation2 (x y : ℚ) : Prop := x^2 + y = 7

-- Define the solution set
def solution_set : Set (ℚ × ℚ) := {(-2/3, 47/9), (3, 4)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℚ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2373_237396


namespace NUMINAMATH_CALUDE_product_of_base6_digits_7891_l2373_237301

/-- The base 6 representation of a natural number -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- The product of a list of natural numbers -/
def listProduct (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 6 representation of 7891 is 0 -/
theorem product_of_base6_digits_7891 :
  listProduct (toBase6 7891) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_base6_digits_7891_l2373_237301


namespace NUMINAMATH_CALUDE_chess_team_arrangements_l2373_237358

def num_boys : ℕ := 3
def num_girls : ℕ := 2

def arrange_chess_team (boys : ℕ) (girls : ℕ) : ℕ :=
  (girls.factorial) * (boys.factorial)

theorem chess_team_arrangements :
  arrange_chess_team num_boys num_girls = 12 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_arrangements_l2373_237358


namespace NUMINAMATH_CALUDE_divisors_of_30_l2373_237313

/-- The number of integer divisors (positive and negative) of 30 -/
def number_of_divisors_of_30 : ℕ :=
  (Finset.filter (· ∣ 30) (Finset.range 31)).card * 2

/-- Theorem stating that the number of integer divisors of 30 is 16 -/
theorem divisors_of_30 : number_of_divisors_of_30 = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_30_l2373_237313


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2373_237308

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2373_237308


namespace NUMINAMATH_CALUDE_play_area_calculation_l2373_237319

/-- Calculates the area of a rectangular play area given specific fencing conditions. -/
theorem play_area_calculation (total_posts : ℕ) (post_spacing : ℕ) (extra_posts_long_side : ℕ) : 
  total_posts = 24 → 
  post_spacing = 5 → 
  extra_posts_long_side = 6 → 
  ∃ (short_side_posts long_side_posts : ℕ),
    short_side_posts + extra_posts_long_side = long_side_posts ∧
    2 * short_side_posts + 2 * long_side_posts - 4 = total_posts ∧
    (short_side_posts - 1) * post_spacing * (long_side_posts - 1) * post_spacing = 675 :=
by sorry

end NUMINAMATH_CALUDE_play_area_calculation_l2373_237319


namespace NUMINAMATH_CALUDE_some_number_less_than_two_l2373_237367

theorem some_number_less_than_two (x y : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + y < 9)
  (h6 : x = 7) : 
  y < 2 := by
sorry

end NUMINAMATH_CALUDE_some_number_less_than_two_l2373_237367


namespace NUMINAMATH_CALUDE_number_of_divisors_of_36_l2373_237339

theorem number_of_divisors_of_36 : Finset.card (Nat.divisors 36) = 9 := by sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_36_l2373_237339


namespace NUMINAMATH_CALUDE_calculation_proof_l2373_237399

theorem calculation_proof : 17 * (17/18) + 35 * (35/36) = 50 + 1/12 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2373_237399


namespace NUMINAMATH_CALUDE_determine_original_prices_l2373_237388

/-- Represents a purchase of products A and B -/
structure Purchase where
  quantityA : ℕ
  quantityB : ℕ
  totalPrice : ℕ

/-- Represents the store's pricing system -/
structure Store where
  priceA : ℕ
  priceB : ℕ

/-- Checks if a purchase is consistent with the store's pricing -/
def isPurchaseConsistent (s : Store) (p : Purchase) : Prop :=
  s.priceA * p.quantityA + s.priceB * p.quantityB = p.totalPrice

/-- The theorem stating that given the purchase data, we can determine the original prices -/
theorem determine_original_prices 
  (p1 p2 : Purchase)
  (h1 : p1.quantityA = 6 ∧ p1.quantityB = 5 ∧ p1.totalPrice = 1140)
  (h2 : p2.quantityA = 3 ∧ p2.quantityB = 7 ∧ p2.totalPrice = 1110) :
  ∃ (s : Store), 
    s.priceA = 90 ∧ 
    s.priceB = 120 ∧ 
    isPurchaseConsistent s p1 ∧ 
    isPurchaseConsistent s p2 :=
  sorry

end NUMINAMATH_CALUDE_determine_original_prices_l2373_237388


namespace NUMINAMATH_CALUDE_equal_non_overlapping_areas_l2373_237372

-- Define two congruent triangles
def Triangle : Type := ℝ × ℝ × ℝ

-- Define a function to calculate the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the hexagon formed by the intersection
def Hexagon : Type := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

-- Define a function to calculate the area of a hexagon
def hexagon_area (h : Hexagon) : ℝ := sorry

-- Define the overlapping triangles and their intersection
def triangles_overlap (t1 t2 : Triangle) (h : Hexagon) : Prop :=
  ∃ (a1 a2 : ℝ), 
    area t1 = hexagon_area h + a1 ∧
    area t2 = hexagon_area h + a2

-- Theorem statement
theorem equal_non_overlapping_areas 
  (t1 t2 : Triangle) 
  (h : Hexagon) 
  (congruent : area t1 = area t2) 
  (overlap : triangles_overlap t1 t2 h) : 
  ∃ (a : ℝ), 
    area t1 = hexagon_area h + a ∧ 
    area t2 = hexagon_area h + a := 
by sorry

end NUMINAMATH_CALUDE_equal_non_overlapping_areas_l2373_237372


namespace NUMINAMATH_CALUDE_point_P_in_fourth_quadrant_l2373_237343

def point_P : ℝ × ℝ := (8, -3)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_P_in_fourth_quadrant :
  in_fourth_quadrant point_P := by
  sorry

end NUMINAMATH_CALUDE_point_P_in_fourth_quadrant_l2373_237343


namespace NUMINAMATH_CALUDE_olivia_savings_account_l2373_237330

/-- The compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem olivia_savings_account :
  let principal : ℝ := 5000
  let rate : ℝ := 0.07
  let time : ℕ := 15
  let final_amount := compound_interest principal rate time
  ∃ ε > 0, |final_amount - 13795.15| < ε :=
by sorry

end NUMINAMATH_CALUDE_olivia_savings_account_l2373_237330


namespace NUMINAMATH_CALUDE_swift_stream_pump_l2373_237397

/-- The SwiftStream pump problem -/
theorem swift_stream_pump (pump_rate : ℝ) (time : ℝ) (volume : ℝ) : 
  pump_rate = 500 → time = 1/2 → volume = pump_rate * time → volume = 250 := by
  sorry

end NUMINAMATH_CALUDE_swift_stream_pump_l2373_237397


namespace NUMINAMATH_CALUDE_intersection_complement_equal_set_l2373_237325

def M : Set ℝ := {-1, 0, 1, 3}
def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement_equal_set : M ∩ (Set.univ \ N) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_set_l2373_237325


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2373_237327

theorem mod_equivalence_unique_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 4897 [ZMOD 9] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2373_237327


namespace NUMINAMATH_CALUDE_sequence_sum_50_l2373_237353

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ := 
  (List.range n).map a |>.sum

theorem sequence_sum_50 (a : ℕ → ℕ) : 
  a 1 = 7 ∧ (∀ n : ℕ, a n + a (n + 1) = 20) → sequence_sum a 50 = 500 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_50_l2373_237353


namespace NUMINAMATH_CALUDE_min_colors_theorem_l2373_237306

theorem min_colors_theorem : ∃ (f : Fin 2013 → Fin 3), 
  (∀ i j : Fin 2013, f i = f j → ¬(((i.val + 1) * (j.val + 1)) % 2014 = 0)) ∧
  (∀ n : ℕ, n < 3 → ¬∃ (g : Fin 2013 → Fin n), 
    ∀ i j : Fin 2013, g i = g j → ¬(((i.val + 1) * (j.val + 1)) % 2014 = 0)) :=
sorry

end NUMINAMATH_CALUDE_min_colors_theorem_l2373_237306


namespace NUMINAMATH_CALUDE_both_brothers_selected_probability_l2373_237383

theorem both_brothers_selected_probability 
  (prob_X : ℚ) 
  (prob_Y : ℚ) 
  (h1 : prob_X = 1 / 3) 
  (h2 : prob_Y = 2 / 7) : 
  prob_X * prob_Y = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_both_brothers_selected_probability_l2373_237383


namespace NUMINAMATH_CALUDE_unique_positive_solution_num_positive_solutions_correct_l2373_237341

/-- The polynomial function f(x) = x^11 + 8x^10 + 15x^9 + 1000x^8 - 1200x^7 -/
def f (x : ℝ) : ℝ := x^11 + 8*x^10 + 15*x^9 + 1000*x^8 - 1200*x^7

/-- The number of positive real solutions to the equation f(x) = 0 -/
def num_positive_solutions : ℕ := 1

theorem unique_positive_solution : 
  ∃! (x : ℝ), x > 0 ∧ f x = 0 :=
sorry

theorem num_positive_solutions_correct : 
  (∃! (x : ℝ), x > 0 ∧ f x = 0) ↔ num_positive_solutions = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_num_positive_solutions_correct_l2373_237341


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2373_237349

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_P : ℝ := 30.97
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms for each element
def num_Al : ℕ := 1
def num_P : ℕ := 1
def num_O : ℕ := 4

-- Define the molecular weight calculation function
def molecular_weight (w_Al w_P w_O : ℝ) (n_Al n_P n_O : ℕ) : ℝ :=
  w_Al * n_Al + w_P * n_P + w_O * n_O

-- Theorem statement
theorem compound_molecular_weight :
  molecular_weight atomic_weight_Al atomic_weight_P atomic_weight_O num_Al num_P num_O = 121.95 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2373_237349


namespace NUMINAMATH_CALUDE_endpoint_sum_l2373_237384

/-- Given a line segment with one endpoint at (10, -5) and its midpoint,
    when scaled by a factor of 2 along each axis, results in the point (12, -18),
    prove that the sum of the coordinates of the other endpoint is -11. -/
theorem endpoint_sum (x y : ℝ) : 
  (10 + x) / 2 = 6 ∧ (-5 + y) / 2 = -9 → x + y = -11 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_l2373_237384


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2373_237392

/-- The area of the shaded region in a grid composed of three rectangles minus a triangle --/
theorem shaded_area_calculation (bottom_height bottom_width middle_height middle_width top_height top_width triangle_base triangle_height : ℕ) 
  (h_bottom : bottom_height = 3 ∧ bottom_width = 5)
  (h_middle : middle_height = 4 ∧ middle_width = 7)
  (h_top : top_height = 5 ∧ top_width = 12)
  (h_triangle : triangle_base = 12 ∧ triangle_height = 5) :
  (bottom_height * bottom_width + middle_height * middle_width + top_height * top_width) - (triangle_base * triangle_height / 2) = 73 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2373_237392


namespace NUMINAMATH_CALUDE_tempo_insurance_premium_l2373_237304

/-- Calculate the premium amount for a tempo insurance --/
theorem tempo_insurance_premium 
  (original_value : ℝ) 
  (insurance_extent : ℝ) 
  (premium_rate : ℝ) 
  (h1 : original_value = 14000)
  (h2 : insurance_extent = 5/7)
  (h3 : premium_rate = 3/100) : 
  original_value * insurance_extent * premium_rate = 300 := by
  sorry

end NUMINAMATH_CALUDE_tempo_insurance_premium_l2373_237304


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2373_237352

theorem max_value_of_expression (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  2 * x * y + y * z + 2 * z * x ≤ 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2373_237352


namespace NUMINAMATH_CALUDE_min_text_length_for_symbol_occurrence_l2373_237322

theorem min_text_length_for_symbol_occurrence : 
  ∃ (x : ℕ), (19 : ℝ) * (21 : ℝ) / 200 < (x : ℝ) ∧ (x : ℝ) < (19 : ℝ) * (11 : ℝ) / 100 ∧
  ∀ (L : ℕ), L < 19 → ¬∃ (y : ℕ), (L : ℝ) * (21 : ℝ) / 200 < (y : ℝ) ∧ (y : ℝ) < (L : ℝ) * (11 : ℝ) / 100 :=
by sorry

end NUMINAMATH_CALUDE_min_text_length_for_symbol_occurrence_l2373_237322


namespace NUMINAMATH_CALUDE_agnes_twice_jane_age_l2373_237359

/-- The number of years until Agnes is twice as old as Jane -/
def years_until_double_age (agnes_age : ℕ) (jane_age : ℕ) : ℕ :=
  (agnes_age - 2 * jane_age) / (2 - 1)

/-- Theorem stating that it will take 13 years for Agnes to be twice as old as Jane -/
theorem agnes_twice_jane_age (agnes_current_age jane_current_age : ℕ) 
  (h1 : agnes_current_age = 25) 
  (h2 : jane_current_age = 6) : 
  years_until_double_age agnes_current_age jane_current_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_agnes_twice_jane_age_l2373_237359


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2373_237333

-- Define the quadratic inequality and its solution set
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop := (1 - a) * x^2 - 4*x + 6 > 0
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}

-- State the theorem
theorem quadratic_inequality_theorem (a : ℝ) :
  (∀ x, x ∈ solution_set a ↔ quadratic_inequality a x) →
  (a = 3) ∧
  (∀ x, 2*x^2 + (2-a)*x - a > 0 ↔ x < -1 ∨ x > 1) ∧
  (∀ b, (∀ x, a*x^2 + b*x + 3 ≥ 0) ↔ -6 ≤ b ∧ b ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2373_237333


namespace NUMINAMATH_CALUDE_book_weight_l2373_237326

theorem book_weight (total_weight : ℝ) (num_books : ℕ) (h1 : total_weight = 42) (h2 : num_books = 14) :
  total_weight / num_books = 3 := by
sorry

end NUMINAMATH_CALUDE_book_weight_l2373_237326


namespace NUMINAMATH_CALUDE_alex_growth_rate_l2373_237363

/-- Alex's growth rate problem -/
theorem alex_growth_rate :
  let required_height : ℚ := 54
  let current_height : ℚ := 48
  let growth_rate_upside_down : ℚ := 1 / 12
  let hours_upside_down_per_month : ℚ := 2
  let months_in_year : ℕ := 12
  let height_difference := required_height - current_height
  let growth_from_hanging := growth_rate_upside_down * hours_upside_down_per_month * months_in_year
  let natural_growth := height_difference - growth_from_hanging
  natural_growth / months_in_year = 1 / 3 := by
sorry


end NUMINAMATH_CALUDE_alex_growth_rate_l2373_237363


namespace NUMINAMATH_CALUDE_swap_three_of_eight_eq_112_l2373_237371

/-- The number of ways to select and swap 3 people out of 8 in a row --/
def swap_three_of_eight : ℕ :=
  Nat.choose 8 3 * 2

/-- Theorem stating that swapping 3 out of 8 people results in 112 different arrangements --/
theorem swap_three_of_eight_eq_112 : swap_three_of_eight = 112 := by
  sorry

end NUMINAMATH_CALUDE_swap_three_of_eight_eq_112_l2373_237371


namespace NUMINAMATH_CALUDE_matrix_determinant_l2373_237376

theorem matrix_determinant : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; -3, 5]
  Matrix.det A = 29 := by
sorry

end NUMINAMATH_CALUDE_matrix_determinant_l2373_237376


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2373_237347

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = -1

/-- The asymptote equations -/
def asymptotes (x y : ℝ) : Prop := x + 2*y = 0 ∨ x - 2*y = 0

/-- Theorem: The asymptotes of the given hyperbola are x ± 2y = 0 -/
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola x y → asymptotes x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2373_237347


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_odds_l2373_237307

theorem largest_of_five_consecutive_odds (n : ℤ) : 
  (n % 2 = 1) → 
  (n * (n + 2) * (n + 4) * (n + 6) * (n + 8) = 93555) → 
  (n + 8 = 19) := by
  sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_odds_l2373_237307


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2373_237369

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 4
  f 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2373_237369


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l2373_237344

theorem quadratic_root_sum (p q : ℝ) : 
  (∃ x : ℂ, x^2 + p*x + q = 0 ∧ x = 1 + Complex.I) → p + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l2373_237344


namespace NUMINAMATH_CALUDE_cafe_chairs_minimum_l2373_237315

theorem cafe_chairs_minimum (indoor_tables outdoor_tables : ℕ)
  (indoor_min indoor_max outdoor_min outdoor_max : ℕ)
  (total_customers indoor_customers : ℕ) :
  indoor_tables = 9 →
  outdoor_tables = 11 →
  indoor_min = 6 →
  indoor_max = 10 →
  outdoor_min = 3 →
  outdoor_max = 5 →
  total_customers = 35 →
  indoor_customers = 18 →
  indoor_min ≤ indoor_max →
  outdoor_min ≤ outdoor_max →
  indoor_customers ≤ total_customers →
  (∀ t, t ≤ indoor_tables → indoor_min ≤ t * indoor_min) →
  (∀ t, t ≤ outdoor_tables → outdoor_min ≤ t * outdoor_min) →
  87 ≤ indoor_tables * indoor_min + outdoor_tables * outdoor_min :=
by
  sorry

#check cafe_chairs_minimum

end NUMINAMATH_CALUDE_cafe_chairs_minimum_l2373_237315


namespace NUMINAMATH_CALUDE_percentage_commutation_l2373_237374

theorem percentage_commutation (n : ℝ) (h : 0.20 * 0.10 * n = 12) : 0.10 * 0.20 * n = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l2373_237374


namespace NUMINAMATH_CALUDE_fuchsia_to_mauve_amount_correct_l2373_237393

/-- Represents the composition of paint in parts --/
structure PaintComposition where
  red : ℚ
  blue : ℚ

/-- The amount of fuchsia paint being changed to mauve paint --/
def fuchsia_amount : ℚ := 106.68

/-- The composition of fuchsia paint --/
def fuchsia : PaintComposition := { red := 5, blue := 3 }

/-- The composition of mauve paint --/
def mauve : PaintComposition := { red := 3, blue := 5 }

/-- The amount of blue paint added to change fuchsia to mauve --/
def blue_added : ℚ := 26.67

/-- Theorem stating that the calculated amount of fuchsia paint is correct --/
theorem fuchsia_to_mauve_amount_correct :
  fuchsia_amount * (fuchsia.blue / (fuchsia.red + fuchsia.blue)) + blue_added =
  fuchsia_amount * (mauve.blue / (mauve.red + mauve.blue)) := by
  sorry

end NUMINAMATH_CALUDE_fuchsia_to_mauve_amount_correct_l2373_237393


namespace NUMINAMATH_CALUDE_printer_task_pages_l2373_237302

theorem printer_task_pages : ∀ (P : ℕ),
  (P / 60 + (P / 60 + 3) = P / 24) →
  (P = 360) :=
by
  sorry

#check printer_task_pages

end NUMINAMATH_CALUDE_printer_task_pages_l2373_237302


namespace NUMINAMATH_CALUDE_specific_systematic_sample_l2373_237345

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (sampleSize : ℕ) (firstNumber : ℕ) (nthItem : ℕ) : ℕ :=
  let k := totalItems / sampleSize
  firstNumber + k * (nthItem - 1)

/-- Theorem for the specific systematic sampling problem -/
theorem specific_systematic_sample :
  systematicSample 1000 50 15 40 = 795 := by
  sorry

end NUMINAMATH_CALUDE_specific_systematic_sample_l2373_237345


namespace NUMINAMATH_CALUDE_smallest_in_S_l2373_237357

def S : Set Int := {0, -17, 4, 3, -2}

theorem smallest_in_S : ∀ x ∈ S, -17 ≤ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_in_S_l2373_237357


namespace NUMINAMATH_CALUDE_optimal_strategy_probability_l2373_237329

/-- Represents the color of a hat -/
inductive HatColor
| Red
| Blue

/-- A strategy for guessing hat colors -/
def Strategy := (n : Nat) → (Vector HatColor n) → Vector Bool n

/-- The probability of all prisoners guessing correctly given a strategy -/
def SuccessProbability (n : Nat) (s : Strategy) : ℚ :=
  sorry

/-- Theorem stating that the maximum success probability is 1/2 -/
theorem optimal_strategy_probability (n : Nat) :
  ∀ s : Strategy, SuccessProbability n s ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_probability_l2373_237329


namespace NUMINAMATH_CALUDE_village_population_panic_l2373_237328

theorem village_population_panic (original_population : ℕ) (final_population : ℕ) 
  (h1 : original_population = 7600)
  (h2 : final_population = 5130) :
  let remaining_after_initial := original_population - original_population / 10
  let left_during_panic := remaining_after_initial - final_population
  (left_during_panic : ℚ) / remaining_after_initial * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_village_population_panic_l2373_237328


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l2373_237340

theorem complex_number_coordinates (a : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a - 2 : ℂ) + (a + 1) * i = b * i) (h3 : b ≠ 0) :
  (a - 3 * i) / (2 - i) = 7/5 - 4/5 * i :=
sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l2373_237340


namespace NUMINAMATH_CALUDE_simplify_expression_l2373_237314

theorem simplify_expression (w : ℝ) : 2*w + 3 - 4*w - 5 + 6*w + 7 - 8*w - 9 = -4*w - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2373_237314


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2373_237321

theorem fraction_sum_equality : (7 : ℚ) / 10 + (3 : ℚ) / 100 + (9 : ℚ) / 1000 = 739 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2373_237321


namespace NUMINAMATH_CALUDE_stream_speed_l2373_237361

/-- Given a boat's travel times and distances, prove the speed of the stream --/
theorem stream_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) 
  (h1 : downstream_distance = 100) 
  (h2 : downstream_time = 4)
  (h3 : upstream_distance = 75)
  (h4 : upstream_time = 15) :
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = downstream_distance / downstream_time ∧
    boat_speed - stream_speed = upstream_distance / upstream_time ∧
    stream_speed = 10 := by
  sorry

#check stream_speed

end NUMINAMATH_CALUDE_stream_speed_l2373_237361
