import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_of_product_l870_87084

theorem units_digit_of_product (n : ℕ) : 
  (2^2021 * 5^2022 * 7^2023) % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l870_87084


namespace NUMINAMATH_CALUDE_pillar_height_at_F_is_10_l870_87003

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a regular hexagon with pillars -/
structure HexagonWithPillars where
  sideLength : ℝ
  vertexA : Point3D
  heightA : ℝ
  heightB : ℝ
  heightC : ℝ

/-- Calculates the height of the pillar at vertex F -/
def pillarHeightAtF (h : HexagonWithPillars) : ℝ :=
  sorry

/-- Theorem: The height of the pillar at vertex F is 10 meters -/
theorem pillar_height_at_F_is_10 (h : HexagonWithPillars) 
  (h_side : h.sideLength = 8)
  (h_vertexA : h.vertexA = ⟨3, 3 * Real.sqrt 3, 0⟩)
  (h_heightA : h.heightA = 15)
  (h_heightB : h.heightB = 10)
  (h_heightC : h.heightC = 12) :
  pillarHeightAtF h = 10 :=
by sorry

#check pillar_height_at_F_is_10

end NUMINAMATH_CALUDE_pillar_height_at_F_is_10_l870_87003


namespace NUMINAMATH_CALUDE_f_is_quadratic_l870_87042

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + x - 2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l870_87042


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l870_87085

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_intersect : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (α β : Plane) (m n : Line)
  (h_distinct_planes : α ≠ β)
  (h_planes_parallel : parallel α β)
  (h_m_parallel_α : line_parallel_plane m α)
  (h_n_intersect_m : line_intersect n m)
  (h_n_not_in_β : ¬ line_in_plane n β) :
  line_parallel_plane n β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l870_87085


namespace NUMINAMATH_CALUDE_car_rental_cost_per_mile_l870_87089

/-- Represents a car rental plan with an initial fee and a per-mile cost. -/
structure RentalPlan where
  initialFee : ℝ
  costPerMile : ℝ

/-- The total cost of a rental plan for a given number of miles. -/
def totalCost (plan : RentalPlan) (miles : ℝ) : ℝ :=
  plan.initialFee + plan.costPerMile * miles

theorem car_rental_cost_per_mile :
  let plan1 : RentalPlan := { initialFee := 65, costPerMile := x }
  let plan2 : RentalPlan := { initialFee := 0, costPerMile := 0.60 }
  let miles : ℝ := 325
  totalCost plan1 miles = totalCost plan2 miles →
  x = 0.40 := by
sorry

end NUMINAMATH_CALUDE_car_rental_cost_per_mile_l870_87089


namespace NUMINAMATH_CALUDE_abc_value_l870_87046

def is_valid_abc (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > b ∧ b > c ∧
  (10 * a + b) + (10 * b + a) = 55 ∧
  1300 < 222 * (a + b + c) ∧ 222 * (a + b + c) < 1400

theorem abc_value :
  ∀ a b c : ℕ, is_valid_abc a b c → a = 3 ∧ b = 2 ∧ c = 1 :=
sorry

end NUMINAMATH_CALUDE_abc_value_l870_87046


namespace NUMINAMATH_CALUDE_farm_animals_l870_87095

theorem farm_animals (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 20 + 3 * (cows + chickens)) → 
  (cows = 20 + chickens) :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_l870_87095


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l870_87036

/-- A batch of products -/
structure Batch where
  good : ℕ
  defective : ℕ
  h_good : good > 2
  h_defective : defective > 2

/-- A sample of two items from a batch -/
structure Sample (b : Batch) where
  good : Fin 3
  defective : Fin 3
  h_sum : good.val + defective.val = 2

/-- Event: At least one defective product in the sample -/
def at_least_one_defective (s : Sample b) : Prop :=
  s.defective.val ≥ 1

/-- Event: All products in the sample are good -/
def all_good (s : Sample b) : Prop :=
  s.good.val = 2

/-- The main theorem: "At least one defective" and "All good" are mutually exclusive -/
theorem mutually_exclusive_events (b : Batch) :
  ∀ (s : Sample b), ¬(at_least_one_defective s ∧ all_good s) :=
by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l870_87036


namespace NUMINAMATH_CALUDE_extremum_at_one_decreasing_when_a_geq_two_monotonicity_when_a_lt_two_l870_87041

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := (2 - a) * x - 2 * Real.log x

def f_deriv (x : ℝ) : ℝ := 2 - a - 2 / x

theorem extremum_at_one (h : f_deriv a 1 = 0) : a = 0 := by sorry

theorem decreasing_when_a_geq_two (h : a ≥ 2) : 
  ∀ x > 0, f_deriv a x < 0 := by sorry

theorem monotonicity_when_a_lt_two (h : a < 2) :
  (∀ x ∈ Set.Ioo 0 (2 / (2 - a)), f_deriv a x < 0) ∧
  (∀ x ∈ Set.Ioi (2 / (2 - a)), f_deriv a x > 0) := by sorry

end NUMINAMATH_CALUDE_extremum_at_one_decreasing_when_a_geq_two_monotonicity_when_a_lt_two_l870_87041


namespace NUMINAMATH_CALUDE_jimmy_payment_l870_87082

/-- Represents the cost of a pizza in dollars -/
def pizza_cost : ℕ := 12

/-- Represents the delivery charge in dollars for distances over 1 km -/
def delivery_charge : ℕ := 2

/-- Represents the distance threshold in meters for applying delivery charge -/
def distance_threshold : ℕ := 1000

/-- Represents the number of pizzas delivered to the park -/
def park_pizzas : ℕ := 3

/-- Represents the distance to the park in meters -/
def park_distance : ℕ := 100

/-- Represents the number of pizzas delivered to the building -/
def building_pizzas : ℕ := 2

/-- Represents the distance to the building in meters -/
def building_distance : ℕ := 2000

/-- Calculates the total amount Jimmy got paid for the pizzas -/
def total_amount : ℕ :=
  (park_pizzas + building_pizzas) * pizza_cost +
  (if building_distance > distance_threshold then building_pizzas * delivery_charge else 0)

theorem jimmy_payment : total_amount = 64 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_payment_l870_87082


namespace NUMINAMATH_CALUDE_equal_share_of_candles_total_divisible_by_four_l870_87069

/- Define the number of candles for each person -/
def ambika_candles : ℕ := 4
def aniyah_candles : ℕ := 6 * ambika_candles
def bree_candles : ℕ := 2 * aniyah_candles
def caleb_candles : ℕ := bree_candles + (bree_candles / 2)

/- Define the total number of candles -/
def total_candles : ℕ := ambika_candles + aniyah_candles + bree_candles + caleb_candles

/- The theorem to prove -/
theorem equal_share_of_candles : total_candles / 4 = 37 := by
  sorry

/- Additional helper theorem to show the total is divisible by 4 -/
theorem total_divisible_by_four : total_candles % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_of_candles_total_divisible_by_four_l870_87069


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l870_87050

theorem geometric_series_first_term 
  (a r : ℝ) 
  (sum_condition : a / (1 - r) = 30)
  (sum_squares_condition : a^2 / (1 - r^2) = 120) :
  a = 240 / 7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l870_87050


namespace NUMINAMATH_CALUDE_binomial_18_4_l870_87066

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_4_l870_87066


namespace NUMINAMATH_CALUDE_mitch_weekend_to_weekday_ratio_l870_87053

/-- Represents Mitch's work schedule and earnings --/
structure MitchSchedule where
  weekdayHours : ℕ  -- Hours worked per weekday
  weekendHours : ℕ  -- Hours worked per weekend day
  weekdayRate : ℚ   -- Hourly rate for weekdays
  totalEarnings : ℚ -- Total weekly earnings

/-- Calculates the ratio of weekend rate to weekday rate --/
def weekendToWeekdayRatio (schedule : MitchSchedule) : ℚ :=
  let totalWeekdayHours := schedule.weekdayHours * 5
  let totalWeekendHours := schedule.weekendHours * 2
  let weekdayEarnings := schedule.weekdayRate * totalWeekdayHours
  let weekendEarnings := schedule.totalEarnings - weekdayEarnings
  let weekendRate := weekendEarnings / totalWeekendHours
  weekendRate / schedule.weekdayRate

/-- Theorem stating that Mitch's weekend to weekday rate ratio is 2:1 --/
theorem mitch_weekend_to_weekday_ratio :
  let schedule : MitchSchedule := {
    weekdayHours := 5,
    weekendHours := 3,
    weekdayRate := 3,
    totalEarnings := 111
  }
  weekendToWeekdayRatio schedule = 2 := by
  sorry

end NUMINAMATH_CALUDE_mitch_weekend_to_weekday_ratio_l870_87053


namespace NUMINAMATH_CALUDE_square_b_minus_d_l870_87077

theorem square_b_minus_d (a b c d : ℤ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 3) : 
  (b - d)^2 = 25 := by sorry

end NUMINAMATH_CALUDE_square_b_minus_d_l870_87077


namespace NUMINAMATH_CALUDE_stratified_sampling_seniors_l870_87010

theorem stratified_sampling_seniors (total_students : ℕ) (senior_students : ℕ) (sample_size : ℕ)
  (h1 : total_students = 4500)
  (h2 : senior_students = 1500)
  (h3 : sample_size = 300)
  (h4 : senior_students ≤ total_students) :
  (senior_students * sample_size) / total_students = 100 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_seniors_l870_87010


namespace NUMINAMATH_CALUDE_sum_inequality_l870_87068

theorem sum_inequality (t1 t2 t3 t4 t5 : ℝ) :
  (1 - t1) * Real.exp t1 +
  (1 - t2) * Real.exp (t1 + t2) +
  (1 - t3) * Real.exp (t1 + t2 + t3) +
  (1 - t4) * Real.exp (t1 + t2 + t3 + t4) +
  (1 - t5) * Real.exp (t1 + t2 + t3 + t4 + t5) ≤ Real.exp (Real.exp (Real.exp (Real.exp 1))) := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l870_87068


namespace NUMINAMATH_CALUDE_raven_age_is_55_l870_87092

-- Define the current ages
def phoebe_age : ℕ := 10
def raven_age : ℕ := 55

-- Define the conditions
def condition1 : Prop := raven_age + 5 = 4 * (phoebe_age + 5)
def condition2 : Prop := phoebe_age = 10
def condition3 : Prop := ∃ sam_age : ℕ, sam_age = 2 * ((raven_age + 3) - (phoebe_age + 3))

-- Theorem statement
theorem raven_age_is_55 : 
  condition1 ∧ condition2 ∧ condition3 → raven_age = 55 :=
by sorry

end NUMINAMATH_CALUDE_raven_age_is_55_l870_87092


namespace NUMINAMATH_CALUDE_sick_days_per_year_l870_87099

/-- Represents the number of hours in a workday -/
def hoursPerDay : ℕ := 8

/-- Represents the number of hours remaining after using half of the allotment -/
def remainingHours : ℕ := 80

/-- Theorem stating that the number of sick days per year is 20 -/
theorem sick_days_per_year :
  ∀ (sickDays vacationDays : ℕ),
  sickDays = vacationDays →
  sickDays + vacationDays = 2 * (remainingHours / hoursPerDay) →
  sickDays = 20 := by sorry

end NUMINAMATH_CALUDE_sick_days_per_year_l870_87099


namespace NUMINAMATH_CALUDE_jake_sister_weight_ratio_l870_87005

/-- Represents the weight ratio problem of Jake and his sister -/
theorem jake_sister_weight_ratio :
  let jake_present_weight : ℕ := 108
  let total_weight : ℕ := 156
  let weight_loss : ℕ := 12
  let jake_new_weight : ℕ := jake_present_weight - weight_loss
  let sister_weight : ℕ := total_weight - jake_new_weight
  (jake_new_weight : ℚ) / sister_weight = 8 / 5 :=
by sorry

end NUMINAMATH_CALUDE_jake_sister_weight_ratio_l870_87005


namespace NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l870_87006

/-- Given a rectangle with perimeter 60 and length 3 times its width, 
    and a triangle with height 36, if their areas are equal, 
    then the base of the triangle (which is also one side of the rectangle) is 9.375. -/
theorem rectangle_triangle_equal_area (w : ℝ) (x : ℝ) : 
  (2 * (w + 3*w) = 60) →  -- Rectangle perimeter is 60
  (w * (3*w) = (1/2) * 36 * x) →  -- Rectangle and triangle have equal area
  x = 9.375 := by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l870_87006


namespace NUMINAMATH_CALUDE_target_hit_probability_l870_87078

theorem target_hit_probability (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.6) (h_B : prob_B = 0.5) : 
  1 - (1 - prob_A) * (1 - prob_B) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l870_87078


namespace NUMINAMATH_CALUDE_initial_nickels_correct_l870_87043

/-- The number of nickels Mike had initially -/
def initial_nickels : ℕ := 87

/-- The number of nickels Mike's dad borrowed -/
def borrowed_nickels : ℕ := 75

/-- The number of nickels Mike was left with -/
def remaining_nickels : ℕ := 12

/-- Theorem stating that the initial number of nickels is correct -/
theorem initial_nickels_correct : initial_nickels = borrowed_nickels + remaining_nickels := by
  sorry

end NUMINAMATH_CALUDE_initial_nickels_correct_l870_87043


namespace NUMINAMATH_CALUDE_work_completion_time_l870_87079

/-- Given that:
  * p can complete the work in 20 days
  * p and q work together for 2 days
  * After 2 days of working together, 0.7 of the work is left
  Prove that q can complete the work alone in 10 days -/
theorem work_completion_time (p_time q_time : ℝ) (h1 : p_time = 20) 
  (h2 : 2 * (1 / p_time + 1 / q_time) = 0.3) : q_time = 10 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l870_87079


namespace NUMINAMATH_CALUDE_a_plus_b_value_m_range_l870_87017

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 < 0}

-- Theorem 1
theorem a_plus_b_value :
  ∀ a b : ℝ, A a b = {x | -1 ≤ x ∧ x ≤ 4} → a + b = -7 :=
sorry

-- Theorem 2
theorem m_range (a b : ℝ) :
  A a b = {x | -1 ≤ x ∧ x ≤ 4} →
  (∀ x : ℝ, x ∈ A a b → x ∉ B m) →
  m ≤ -3 ∨ m ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_a_plus_b_value_m_range_l870_87017


namespace NUMINAMATH_CALUDE_major_premise_is_false_l870_87030

/-- A plane in 3D space -/
structure Plane3D where
  -- Define plane properties here
  
/-- A line in 3D space -/
structure Line3D where
  -- Define line properties here

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines when two lines are skew -/
def skew_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines when two lines are perpendicular -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Theorem stating that the major premise is false -/
theorem major_premise_is_false :
  ¬ ∀ (l : Line3D) (p : Plane3D) (l_in_p : Line3D),
    parallel_line_plane l p →
    line_in_plane l_in_p p →
    parallel_lines l l_in_p :=
  sorry

end NUMINAMATH_CALUDE_major_premise_is_false_l870_87030


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_140_l870_87002

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_140 :
  rectangle_area 1225 10 = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_140_l870_87002


namespace NUMINAMATH_CALUDE_exists_integers_satisfying_inequality_l870_87044

theorem exists_integers_satisfying_inequality :
  ∃ (A B : ℤ), (0.999 : ℝ) < (A : ℝ) + (B : ℝ) * Real.sqrt 2 ∧ (A : ℝ) + (B : ℝ) * Real.sqrt 2 < 1 :=
by sorry

end NUMINAMATH_CALUDE_exists_integers_satisfying_inequality_l870_87044


namespace NUMINAMATH_CALUDE_intersection_area_greater_than_half_l870_87018

/-- Two identical rectangles with sides a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ
  pos_a : 0 < a
  pos_b : 0 < b

/-- The configuration of two intersecting rectangles -/
structure IntersectingRectangles where
  rect : Rectangle
  intersection_points : ℕ
  eight_intersections : intersection_points = 8

/-- The area of intersection of two rectangles -/
def intersectionArea (ir : IntersectingRectangles) : ℝ := sorry

/-- The area of a single rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.a * r.b

/-- Theorem: The area of intersection is greater than half the area of each rectangle -/
theorem intersection_area_greater_than_half (ir : IntersectingRectangles) :
  intersectionArea ir > (1/2) * rectangleArea ir.rect :=
sorry

end NUMINAMATH_CALUDE_intersection_area_greater_than_half_l870_87018


namespace NUMINAMATH_CALUDE_lcm_problem_l870_87048

theorem lcm_problem (m : ℕ) (h1 : m > 0) (h2 : Nat.lcm 30 m = 90) (h3 : Nat.lcm m 45 = 180) : m = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l870_87048


namespace NUMINAMATH_CALUDE_wendy_shoes_left_l870_87040

theorem wendy_shoes_left (total : ℕ) (given_away : ℕ) (h1 : total = 33) (h2 : given_away = 14) :
  total - given_away = 19 := by
  sorry

end NUMINAMATH_CALUDE_wendy_shoes_left_l870_87040


namespace NUMINAMATH_CALUDE_percent_problem_l870_87047

theorem percent_problem : ∃ x : ℝ, (1 / 100) * x = 123.56 ∧ x = 12356 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l870_87047


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l870_87063

theorem infinitely_many_solutions
  (a b c k : ℤ)
  (D : ℤ)
  (hD : D = b^2 - 4*a*c)
  (hD_pos : D > 0)
  (hD_nonsquare : ∀ m : ℤ, D ≠ m^2)
  (hk : k ≠ 0)
  (h_solution : ∃ (x₀ y₀ : ℤ), a*x₀^2 + b*x₀*y₀ + c*y₀^2 = k) :
  ∃ (S : Set (ℤ × ℤ)), (Set.Infinite S) ∧ (∀ (x y : ℤ), (x, y) ∈ S → a*x^2 + b*x*y + c*y^2 = k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l870_87063


namespace NUMINAMATH_CALUDE_complex_symmetry_product_l870_87034

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  (z₁.im = -z₂.im) → (z₁.re = z₂.re) → (z₁ ≠ 1 + I) → z₁ * z₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_symmetry_product_l870_87034


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_5_or_8_l870_87051

/-- The set of digits excluding 0, 5, and 8 -/
def ValidFirstDigits : Finset ℕ := {1, 2, 3, 4, 6, 7, 9}

/-- The set of digits excluding 5 and 8 -/
def ValidOtherDigits : Finset ℕ := {0, 1, 2, 3, 4, 6, 7, 9}

/-- The number of four-digit numbers -/
def TotalFourDigitNumbers : ℕ := 9000

/-- The number of four-digit numbers without 5 or 8 -/
def NumbersWithout5Or8 : ℕ := Finset.card ValidFirstDigits * Finset.card ValidOtherDigits ^ 3

theorem four_digit_numbers_with_5_or_8 :
  TotalFourDigitNumbers - NumbersWithout5Or8 = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_5_or_8_l870_87051


namespace NUMINAMATH_CALUDE_blue_tetrahedron_volume_l870_87011

/-- Represents a cube with alternating colored corners -/
structure ColoredCube where
  sideLength : ℝ
  alternatingColors : Bool

/-- Calculates the volume of the tetrahedron formed by similarly colored vertices -/
def tetrahedronVolume (cube : ColoredCube) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem blue_tetrahedron_volume (cube : ColoredCube) 
  (h1 : cube.sideLength = 8) 
  (h2 : cube.alternatingColors = true) : 
  tetrahedronVolume cube = 512 / 3 := by
    sorry

end NUMINAMATH_CALUDE_blue_tetrahedron_volume_l870_87011


namespace NUMINAMATH_CALUDE_expression_evaluation_l870_87062

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y^2) :
  (x - 1 / x^2) * (y + 2 / y) = 2 * x^(5/2) - 1 / x :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l870_87062


namespace NUMINAMATH_CALUDE_tinplate_allocation_l870_87070

theorem tinplate_allocation (total_tinplates : ℕ) 
  (bodies_per_tinplate : ℕ) (bottoms_per_tinplate : ℕ) 
  (bodies_to_bottoms_ratio : ℚ) :
  total_tinplates = 36 →
  bodies_per_tinplate = 25 →
  bottoms_per_tinplate = 40 →
  bodies_to_bottoms_ratio = 1/2 →
  ∃ (bodies_tinplates bottoms_tinplates : ℕ),
    bodies_tinplates + bottoms_tinplates = total_tinplates ∧
    bodies_tinplates * bodies_per_tinplate * 2 = bottoms_tinplates * bottoms_per_tinplate ∧
    bodies_tinplates = 16 ∧
    bottoms_tinplates = 20 :=
by sorry

end NUMINAMATH_CALUDE_tinplate_allocation_l870_87070


namespace NUMINAMATH_CALUDE_fraction_comparison_l870_87060

def first_numerator : ℕ := 100^99
def first_denominator : ℕ := 9777777  -- 97...7 with 7 digits

def second_numerator : ℕ := 55555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555  -- 100 digits of 5
def second_denominator : ℕ := 55555  -- 5 digits of 5

theorem fraction_comparison :
  (first_numerator : ℚ) / first_denominator < (second_numerator : ℚ) / second_denominator :=
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l870_87060


namespace NUMINAMATH_CALUDE_unique_n_value_l870_87072

theorem unique_n_value : ∃! n : ℕ, 
  50 ≤ n ∧ n ≤ 120 ∧ 
  ∃ k : ℕ, n = 8 * k ∧
  n % 7 = 5 ∧
  n % 6 = 3 ∧
  n = 208 := by
sorry

end NUMINAMATH_CALUDE_unique_n_value_l870_87072


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l870_87035

theorem point_in_second_quadrant (A B C : ℝ) : 
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  A + B + C = π →    -- A, B, C are angles of a triangle
  Real.cos B - Real.sin A < 0 ∧ Real.sin B - Real.cos A > 0 := by
sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l870_87035


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l870_87057

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l870_87057


namespace NUMINAMATH_CALUDE_article_selling_price_l870_87073

theorem article_selling_price (cost_price : ℝ) (selling_price : ℝ) : 
  (selling_price - cost_price = cost_price - 448) → 
  (768 = 1.2 * cost_price) → 
  selling_price = 832 := by
sorry

end NUMINAMATH_CALUDE_article_selling_price_l870_87073


namespace NUMINAMATH_CALUDE_computer_pricing_l870_87022

/-- Proves that if a selling price of $2240 yields a 40% profit on cost, 
    then a selling price of $2560 yields a 60% profit on the same cost. -/
theorem computer_pricing (cost : ℝ) 
  (h1 : 2240 = cost + 0.4 * cost) 
  (h2 : 2560 = cost + 0.6 * cost) : 
  2240 = cost * 1.4 ∧ 2560 = cost * 1.6 := by
  sorry

#check computer_pricing

end NUMINAMATH_CALUDE_computer_pricing_l870_87022


namespace NUMINAMATH_CALUDE_meter_to_jumps_conversion_l870_87061

/-- Conversion between different units of measurement -/
theorem meter_to_jumps_conversion
  (a p q r s t u v : ℚ)
  (hop_to_skip : a * 1 = p)
  (jump_to_hop : q = r * 1)
  (skip_to_leap : s * 1 = t)
  (leap_to_meter : u * 1 = v)
  (h_nonzero : p ≠ 0 ∧ v ≠ 0 ∧ t ≠ 0 ∧ r ≠ 0) :
  1 = (u * s * a * q) / (p * v * t * r) :=
by sorry

end NUMINAMATH_CALUDE_meter_to_jumps_conversion_l870_87061


namespace NUMINAMATH_CALUDE_pencil_count_l870_87075

theorem pencil_count (num_pens : ℕ) (max_students : ℕ) (num_pencils : ℕ) : 
  num_pens = 640 →
  max_students = 40 →
  num_pens % max_students = 0 →
  num_pencils % max_students = 0 →
  ∃ k : ℕ, num_pencils = 40 * k :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l870_87075


namespace NUMINAMATH_CALUDE_unique_n_divisibility_l870_87065

theorem unique_n_divisibility : ∃! n : ℕ, 
  0 < n ∧ n < 1000 ∧ 
  (∃ k₁ k₂ k₃ : ℕ, 
    345564 - n = 13 * k₁ ∧ 
    345564 - n = 17 * k₂ ∧ 
    345564 - n = 19 * k₃) :=
by sorry

end NUMINAMATH_CALUDE_unique_n_divisibility_l870_87065


namespace NUMINAMATH_CALUDE_parabola_equation_l870_87076

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in its general form ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- Function to calculate the greatest common divisor of six integers -/
def gcd6 (a b c d e f : ℤ) : ℤ := sorry

/-- Theorem stating the equation of the parabola with given focus and directrix -/
theorem parabola_equation (focus : Point) (directrix : Line) : 
  focus.x = 2 ∧ focus.y = 4 ∧ 
  directrix.a = 4 ∧ directrix.b = 5 ∧ directrix.c = 20 → 
  ∃ (p : Parabola), 
    p.a = 25 ∧ p.b = -40 ∧ p.c = 16 ∧ p.d = 0 ∧ p.e = 0 ∧ p.f = 0 ∧ 
    p.a > 0 ∧ 
    gcd6 (abs p.a) (abs p.b) (abs p.c) (abs p.d) (abs p.e) (abs p.f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l870_87076


namespace NUMINAMATH_CALUDE_congruence_sufficient_not_necessary_for_equal_area_l870_87019

-- Define the property of two triangles being congruent
def are_congruent (t1 t2 : Triangle) : Prop := sorry

-- Define the property of two triangles having equal area
def have_equal_area (t1 t2 : Triangle) : Prop := sorry

-- Theorem stating that congruence is sufficient but not necessary for equal area
theorem congruence_sufficient_not_necessary_for_equal_area :
  (∀ t1 t2 : Triangle, are_congruent t1 t2 → have_equal_area t1 t2) ∧
  (∃ t1 t2 : Triangle, have_equal_area t1 t2 ∧ ¬are_congruent t1 t2) := by sorry

end NUMINAMATH_CALUDE_congruence_sufficient_not_necessary_for_equal_area_l870_87019


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l870_87015

open Set

theorem intersection_of_M_and_N :
  let U : Type := ℝ
  let M : Set U := {x | x < 1}
  let N : Set U := {x | 0 < x ∧ x < 2}
  M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l870_87015


namespace NUMINAMATH_CALUDE_square_diagonal_length_l870_87056

theorem square_diagonal_length (perimeter : ℝ) (diagonal : ℝ) : 
  perimeter = 40 → diagonal = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_length_l870_87056


namespace NUMINAMATH_CALUDE_largest_three_digit_sum_l870_87031

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The sum of YXX + YX + ZY given X, Y, and Z -/
def sum (X Y Z : Digit) : ℕ :=
  111 * Y.val + 12 * X.val + 10 * Z.val

/-- Predicate to check if three digits are distinct -/
def distinct (X Y Z : Digit) : Prop :=
  X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z

theorem largest_three_digit_sum :
  ∃ (X Y Z : Digit), distinct X Y Z ∧ 
    sum X Y Z ≤ 999 ∧
    ∀ (A B C : Digit), distinct A B C → sum A B C ≤ sum X Y Z :=
by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_sum_l870_87031


namespace NUMINAMATH_CALUDE_male_students_count_l870_87009

theorem male_students_count (total : ℕ) (difference : ℕ) (male : ℕ) (female : ℕ) : 
  total = 1443 →
  difference = 141 →
  male = female + difference →
  total = male + female →
  male = 792 := by
sorry

end NUMINAMATH_CALUDE_male_students_count_l870_87009


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l870_87055

-- Define the polynomial expression
def f (d : ℝ) : ℝ := -(5 - d) * (d + 2 * (5 - d))

-- Define the expanded form of the polynomial
def expanded_form (d : ℝ) : ℝ := -d^2 + 15*d - 50

-- Theorem statement
theorem sum_of_coefficients :
  (∀ d, f d = expanded_form d) →
  (-1 : ℝ) + 15 + (-50) = -36 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l870_87055


namespace NUMINAMATH_CALUDE_pattern_c_cannot_fold_l870_87080

/-- Represents a pattern of squares with folding lines -/
structure SquarePattern where
  squares : Finset (ℝ × ℝ)  -- Set of coordinates for squares
  foldLines : Finset ((ℝ × ℝ) × (ℝ × ℝ))  -- Set of folding lines

/-- Represents the set of all possible patterns -/
def AllPatterns : Finset SquarePattern := sorry

/-- Predicate to check if a pattern can be folded into a cube without overlap -/
def canFoldIntoCube (p : SquarePattern) : Prop := sorry

/-- The specific Pattern C -/
def PatternC : SquarePattern := sorry

/-- Theorem stating that Pattern C is the only pattern that cannot be folded into a cube -/
theorem pattern_c_cannot_fold :
  PatternC ∈ AllPatterns ∧
  ¬(canFoldIntoCube PatternC) ∧
  ∀ p ∈ AllPatterns, p ≠ PatternC → canFoldIntoCube p :=
sorry

end NUMINAMATH_CALUDE_pattern_c_cannot_fold_l870_87080


namespace NUMINAMATH_CALUDE_square_intersection_perimeter_ratio_l870_87096

/-- Given a square with vertices (-a, 0), (a, 0), (a, 2a), (-a, 2a) intersected by the line y = 2x,
    the ratio of the perimeter of one of the resulting congruent quadrilaterals to a is 5 + √5. -/
theorem square_intersection_perimeter_ratio (a : ℝ) (a_pos : a > 0) :
  let square_vertices := [(-a, 0), (a, 0), (a, 2*a), (-a, 2*a)]
  let intersecting_line := (fun x : ℝ => 2*x)
  let quadrilateral_perimeter := 
    (a + 2*a + 2*a + Real.sqrt (a^2 + (2*a)^2))
  quadrilateral_perimeter / a = 5 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_intersection_perimeter_ratio_l870_87096


namespace NUMINAMATH_CALUDE_original_milk_cost_is_three_l870_87038

/-- The original cost of a gallon of whole milk -/
def original_milk_cost : ℝ := 3

/-- The current price of a gallon of whole milk -/
def current_milk_price : ℝ := 2

/-- The discount on a box of cereal -/
def cereal_discount : ℝ := 1

/-- The total savings from buying 3 gallons of milk and 5 boxes of cereal -/
def total_savings : ℝ := 8

/-- Theorem stating that the original cost of a gallon of whole milk is $3 -/
theorem original_milk_cost_is_three :
  original_milk_cost = 3 ∧
  current_milk_price = 2 ∧
  cereal_discount = 1 ∧
  total_savings = 8 ∧
  3 * (original_milk_cost - current_milk_price) + 5 * cereal_discount = total_savings :=
by sorry

end NUMINAMATH_CALUDE_original_milk_cost_is_three_l870_87038


namespace NUMINAMATH_CALUDE_number_calculation_l870_87094

theorem number_calculation (x : ℝ) : (0.8 * 90 = 0.7 * x + 30) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l870_87094


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l870_87054

theorem shirt_price_reduction (original_price : ℝ) (h1 : original_price > 0) : 
  let sale_price := 0.70 * original_price
  let final_price := 0.63 * original_price
  ∃ markdown_percent : ℝ, 
    markdown_percent = 10 ∧ 
    final_price = sale_price * (1 - markdown_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l870_87054


namespace NUMINAMATH_CALUDE_quadratic_minimum_l870_87029

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - x + 3

-- Theorem statement
theorem quadratic_minimum :
  ∀ x : ℝ, f x ≥ 11/4 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l870_87029


namespace NUMINAMATH_CALUDE_distribution_of_slots_l870_87001

theorem distribution_of_slots (n : ℕ) (k : ℕ) :
  n = 6 →
  k = 3 →
  (Nat.choose (n - 1) (k - 1) : ℕ) = 10 :=
by sorry

end NUMINAMATH_CALUDE_distribution_of_slots_l870_87001


namespace NUMINAMATH_CALUDE_compound_interest_principal_l870_87071

/-- Given a principal amount and an annual compound interest rate,
    prove that the principal amount is approximately 5967.79 if it grows
    to 8000 after 2 years and 9261 after 3 years under compound interest. -/
theorem compound_interest_principal (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 8000)
  (h2 : P * (1 + r)^3 = 9261) :
  ∃ ε > 0, |P - 5967.79| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l870_87071


namespace NUMINAMATH_CALUDE_students_present_l870_87049

def total_students : ℕ := 28
def absent_fraction : ℚ := 2 / 7

theorem students_present (total : ℕ) (absent_frac : ℚ) :
  total = total_students →
  absent_frac = absent_fraction →
  (total : ℚ) - (absent_frac * total) = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_present_l870_87049


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l870_87064

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 6 > 0} = {x : ℝ | x < -3 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l870_87064


namespace NUMINAMATH_CALUDE_sally_has_hundred_l870_87081

/-- Sally's current amount of money -/
def sally_money : ℕ := sorry

/-- The condition that if Sally had $20 less, she would have $80 -/
axiom sally_condition : sally_money - 20 = 80

/-- Theorem: Sally has $100 -/
theorem sally_has_hundred : sally_money = 100 := by sorry

end NUMINAMATH_CALUDE_sally_has_hundred_l870_87081


namespace NUMINAMATH_CALUDE_isosceles_triangle_removal_l870_87025

/-- Given a square with isosceles right triangles removed from each corner to form a rectangle,
    if the diagonal of the resulting rectangle is 15 units,
    then the combined area of the four removed triangles is 112.5 square units. -/
theorem isosceles_triangle_removal (r s : ℝ) : 
  r > 0 → s > 0 →  -- r and s are positive real numbers
  (r + s)^2 + (r - s)^2 = 15^2 →  -- diagonal of resulting rectangle is 15
  2 * r * s = 112.5  -- combined area of four removed triangles
  := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_removal_l870_87025


namespace NUMINAMATH_CALUDE_helen_cookies_l870_87045

/-- The number of raisin cookies Helen baked this morning -/
def raisin_cookies : ℕ := 231

/-- The difference between chocolate chip cookies and raisin cookies -/
def cookie_difference : ℕ := 25

/-- The number of chocolate chip cookies Helen baked this morning -/
def choc_chip_cookies : ℕ := raisin_cookies + cookie_difference

theorem helen_cookies : choc_chip_cookies = 256 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l870_87045


namespace NUMINAMATH_CALUDE_equation_solution_l870_87098

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = 4 ∧ x₂ = -6) ∧ 
  (∀ x : ℝ, 2 * (x + 1)^2 - 49 = 1 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l870_87098


namespace NUMINAMATH_CALUDE_largest_common_divisor_m_squared_minus_n_squared_plus_two_l870_87083

theorem largest_common_divisor_m_squared_minus_n_squared_plus_two
  (m n : ℤ) (h : n < m) :
  ∃ (k : ℤ), m^2 - n^2 + 2 = 2 * k ∧
  ∀ (d : ℤ), (∀ (a b : ℤ), b < a → ∃ (l : ℤ), a^2 - b^2 + 2 = d * l) → d ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_m_squared_minus_n_squared_plus_two_l870_87083


namespace NUMINAMATH_CALUDE_dollar_op_five_negative_two_l870_87052

-- Define the $ operation
def dollar_op (c d : Int) : Int := c * (d + 1) + c * d

-- Theorem statement
theorem dollar_op_five_negative_two :
  dollar_op 5 (-2) = -15 := by
  sorry

end NUMINAMATH_CALUDE_dollar_op_five_negative_two_l870_87052


namespace NUMINAMATH_CALUDE_kiwi_lemon_equality_l870_87007

/-- Represents the distribution of fruits in Tania's baskets -/
structure FruitBaskets where
  total_fruits : Nat
  mangoes : Nat
  pears : Nat
  pawpaws : Nat
  lemons : Nat
  num_baskets : Nat

/-- Tania's fruit baskets satisfy the given conditions -/
def tania_baskets : FruitBaskets :=
  { total_fruits := 58
  , mangoes := 18
  , pears := 10
  , pawpaws := 12
  , lemons := 9
  , num_baskets := 5
  }

/-- The number of kiwis equals the number of lemons in the last two baskets -/
theorem kiwi_lemon_equality (b : FruitBaskets) (h : b = tania_baskets) :
  b.total_fruits - b.mangoes - b.pears - b.pawpaws - b.lemons = b.lemons :=
by sorry

end NUMINAMATH_CALUDE_kiwi_lemon_equality_l870_87007


namespace NUMINAMATH_CALUDE_pascal_all_even_rows_l870_87039

/-- Returns true if a row in Pascal's triangle consists of all even numbers except for the 1s at each end -/
def isAllEvenExceptEnds (row : ℕ) : Bool := sorry

/-- Counts the number of rows in Pascal's triangle from row 2 to row 30 (inclusive) that consist of all even numbers except for the 1s at each end -/
def countAllEvenRows : ℕ := sorry

theorem pascal_all_even_rows : countAllEvenRows = 4 := by sorry

end NUMINAMATH_CALUDE_pascal_all_even_rows_l870_87039


namespace NUMINAMATH_CALUDE_girls_average_score_l870_87020

-- Define the variables
def num_girls : ℝ := 1
def num_boys : ℝ := 1.8 * num_girls
def class_average : ℝ := 75
def girls_score_ratio : ℝ := 1.2

-- Theorem statement
theorem girls_average_score :
  ∃ (girls_score : ℝ),
    girls_score * num_girls + (girls_score / girls_score_ratio) * num_boys = 
    class_average * (num_girls + num_boys) ∧
    girls_score = 84 := by
  sorry

end NUMINAMATH_CALUDE_girls_average_score_l870_87020


namespace NUMINAMATH_CALUDE_xyz_bounds_l870_87033

-- Define the problem
theorem xyz_bounds (x y z a : ℝ) (ha : a > 0) 
  (h1 : x + y + z = a) (h2 : x^2 + y^2 + z^2 = a^2 / 2) :
  (0 ≤ x ∧ x ≤ 2*a/3) ∧ (0 ≤ y ∧ y ≤ 2*a/3) ∧ (0 ≤ z ∧ z ≤ 2*a/3) := by
  sorry

end NUMINAMATH_CALUDE_xyz_bounds_l870_87033


namespace NUMINAMATH_CALUDE_charitable_distribution_boy_amount_l870_87000

def charitable_distribution (initial_pennies : ℕ) 
  (farmer_pennies : ℕ) (beggar_pennies : ℕ) (boy_pennies : ℕ) : Prop :=
  initial_pennies = 42 ∧
  farmer_pennies = initial_pennies / 2 + 1 ∧
  beggar_pennies = (initial_pennies - farmer_pennies) / 2 + 2 ∧
  boy_pennies = initial_pennies - farmer_pennies - beggar_pennies - 1

theorem charitable_distribution_boy_amount :
  ∀ (initial_pennies farmer_pennies beggar_pennies boy_pennies : ℕ),
  charitable_distribution initial_pennies farmer_pennies beggar_pennies boy_pennies →
  boy_pennies = 7 :=
by sorry

end NUMINAMATH_CALUDE_charitable_distribution_boy_amount_l870_87000


namespace NUMINAMATH_CALUDE_caravan_feet_head_difference_l870_87058

theorem caravan_feet_head_difference : 
  let num_hens : ℕ := 50
  let num_goats : ℕ := 45
  let num_camels : ℕ := 8
  let num_keepers : ℕ := 15
  let feet_per_hen : ℕ := 2
  let feet_per_goat : ℕ := 4
  let feet_per_camel : ℕ := 4
  let feet_per_keeper : ℕ := 2
  let total_heads : ℕ := num_hens + num_goats + num_camels + num_keepers
  let total_feet : ℕ := num_hens * feet_per_hen + num_goats * feet_per_goat + 
                        num_camels * feet_per_camel + num_keepers * feet_per_keeper
  total_feet - total_heads = 224 := by
sorry

end NUMINAMATH_CALUDE_caravan_feet_head_difference_l870_87058


namespace NUMINAMATH_CALUDE_max_rooks_on_chessboard_l870_87024

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a rook placement on a chessboard --/
structure RookPlacement :=
  (board : Chessboard)
  (num_rooks : ℕ)

/-- Predicate to check if a rook placement satisfies the condition --/
def satisfies_condition (placement : RookPlacement) : Prop :=
  ∀ (removed : ℕ), removed < placement.num_rooks →
    ∃ (square : ℕ × ℕ), 
      square.1 ≤ placement.board.size ∧ 
      square.2 ≤ placement.board.size ∧
      (∀ (rook : ℕ × ℕ), rook ≠ removed → 
        (rook.1 ≠ square.1 ∧ rook.2 ≠ square.2))

/-- The main theorem --/
theorem max_rooks_on_chessboard :
  ∃ (placement : RookPlacement),
    placement.board.size = 10 ∧
    placement.num_rooks = 81 ∧
    satisfies_condition placement ∧
    (∀ (other_placement : RookPlacement),
      other_placement.board.size = 10 →
      satisfies_condition other_placement →
      other_placement.num_rooks ≤ 81) :=
sorry

end NUMINAMATH_CALUDE_max_rooks_on_chessboard_l870_87024


namespace NUMINAMATH_CALUDE_combined_tax_rate_l870_87027

theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.3) 
  (h2 : mindy_rate = 0.2) 
  (h3 : income_ratio = 3) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.225 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l870_87027


namespace NUMINAMATH_CALUDE_inequality_proof_l870_87004

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + a^2) ≥ Real.sqrt 2 * (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l870_87004


namespace NUMINAMATH_CALUDE_chickens_and_rabbits_l870_87087

theorem chickens_and_rabbits (x y : ℕ) : 
  (x + y = 35 ∧ 2*x + 4*y = 94) ↔ 
  (x + y = 35 ∧ x * 2 + y * 4 = 94) := by sorry

end NUMINAMATH_CALUDE_chickens_and_rabbits_l870_87087


namespace NUMINAMATH_CALUDE_rental_car_distance_l870_87059

theorem rental_car_distance (fixed_fee : ℝ) (per_km_charge : ℝ) (total_bill : ℝ) (km_travelled : ℝ) : 
  fixed_fee = 45 →
  per_km_charge = 0.12 →
  total_bill = 74.16 →
  total_bill = fixed_fee + per_km_charge * km_travelled →
  km_travelled = 243 := by
sorry

end NUMINAMATH_CALUDE_rental_car_distance_l870_87059


namespace NUMINAMATH_CALUDE_anton_card_difference_l870_87013

/-- Given that Anton has three times as many cards as Heike, Ann has the same number of cards as Heike, 
    and Ann has 60 cards, prove that Anton has 120 more cards than Ann. -/
theorem anton_card_difference (heike_cards : ℕ) (ann_cards : ℕ) (anton_cards : ℕ) 
    (h1 : anton_cards = 3 * heike_cards)
    (h2 : ann_cards = heike_cards)
    (h3 : ann_cards = 60) : 
  anton_cards - ann_cards = 120 := by
  sorry

end NUMINAMATH_CALUDE_anton_card_difference_l870_87013


namespace NUMINAMATH_CALUDE_original_lemon_price_was_eight_l870_87014

/-- The problem of determining the original lemon price --/
def lemon_price_problem (original_lemon_price : ℚ) : Prop :=
  let lemon_price_increase : ℚ := 4
  let grape_price_increase : ℚ := lemon_price_increase / 2
  let original_grape_price : ℚ := 7
  let num_lemons : ℕ := 80
  let num_grapes : ℕ := 140
  let total_revenue : ℚ := 2220
  let new_lemon_price : ℚ := original_lemon_price + lemon_price_increase
  let new_grape_price : ℚ := original_grape_price + grape_price_increase
  (num_lemons : ℚ) * new_lemon_price + (num_grapes : ℚ) * new_grape_price = total_revenue

/-- Theorem stating that the original lemon price was 8 --/
theorem original_lemon_price_was_eight :
  lemon_price_problem 8 := by
  sorry

end NUMINAMATH_CALUDE_original_lemon_price_was_eight_l870_87014


namespace NUMINAMATH_CALUDE_brandon_application_theorem_l870_87067

/-- The number of businesses Brandon can still apply to -/
def businesses_can_apply (total : ℕ) (fired : ℕ) (quit : ℕ) (x : ℕ) (y : ℕ) : ℕ :=
  total - (fired + quit - x) + y

theorem brandon_application_theorem (x y : ℕ) :
  businesses_can_apply 72 36 24 x y = 12 + x + y := by
  sorry

end NUMINAMATH_CALUDE_brandon_application_theorem_l870_87067


namespace NUMINAMATH_CALUDE_student_cannot_enter_finals_l870_87026

/-- Represents the competition structure and student's performance -/
structure Competition where
  total_rounds : ℕ
  required_specified : ℕ
  required_creative : ℕ
  min_selected_for_award : ℕ
  rounds_for_finals : ℕ
  specified_selected : ℕ
  total_specified : ℕ
  creative_selected : ℕ
  total_creative : ℕ
  prob_increase : ℚ

/-- Calculates the probability of winning the "Skillful Hands Award" in one round -/
def prob_win_award (c : Competition) : ℚ :=
  sorry

/-- Calculates the expected number of times winning the award in all rounds after intensive training -/
def expected_wins_after_training (c : Competition) : ℚ :=
  sorry

/-- Main theorem: The student cannot enter the finals -/
theorem student_cannot_enter_finals (c : Competition) 
  (h1 : c.total_rounds = 5)
  (h2 : c.required_specified = 2)
  (h3 : c.required_creative = 2)
  (h4 : c.min_selected_for_award = 3)
  (h5 : c.rounds_for_finals = 4)
  (h6 : c.specified_selected = 4)
  (h7 : c.total_specified = 5)
  (h8 : c.creative_selected = 3)
  (h9 : c.total_creative = 5)
  (h10 : c.prob_increase = 1/10) :
  prob_win_award c = 33/50 ∧ expected_wins_after_training c < 4 :=
sorry

end NUMINAMATH_CALUDE_student_cannot_enter_finals_l870_87026


namespace NUMINAMATH_CALUDE_hotel_rate_problem_l870_87032

-- Define the flat rate for the first night and the nightly rate for additional nights
variable (f : ℝ) -- Flat rate for the first night
variable (n : ℝ) -- Nightly rate for additional nights

-- Define Alice's stay
def alice_stay : ℝ := f + 4 * n

-- Define Bob's stay
def bob_stay : ℝ := f + 9 * n

-- State the theorem
theorem hotel_rate_problem (h1 : alice_stay = 245) (h2 : bob_stay = 470) : f = 65 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rate_problem_l870_87032


namespace NUMINAMATH_CALUDE_chess_tournament_games_l870_87074

/-- The number of games played in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 15 players where each player plays every other player once, 
    the total number of games played is 105. -/
theorem chess_tournament_games : num_games 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l870_87074


namespace NUMINAMATH_CALUDE_counterexample_prime_plus_two_l870_87097

theorem counterexample_prime_plus_two :
  ∃ n : ℕ, Nat.Prime n ∧ ¬(Nat.Prime (n + 2)) :=
sorry

end NUMINAMATH_CALUDE_counterexample_prime_plus_two_l870_87097


namespace NUMINAMATH_CALUDE_range_of_m_for_subset_l870_87028

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x : ℝ | -1 < x ∧ x < m}

-- The main theorem
theorem range_of_m_for_subset (h : ∀ m : ℝ, (∀ x : ℝ, x ∈ A → x ∈ B m) ∧ 
  ¬(∀ x : ℝ, x ∈ B m → x ∈ A)) : 
  {m : ℝ | A ⊆ B m} = {m : ℝ | m > 3} := by
  sorry


end NUMINAMATH_CALUDE_range_of_m_for_subset_l870_87028


namespace NUMINAMATH_CALUDE_prisoners_puzzle_solution_l870_87086

-- Define the hair colors
inductive HairColor
| Blonde
| Red
| Brunette

-- Define the prisoners
inductive Prisoner
| P1
| P2
| P3
| P4
| P5

-- Define the ladies
structure Lady where
  name : String
  hairColor : HairColor

-- Define the statement of a prisoner
structure Statement where
  prisoner : Prisoner
  ownLady : Lady
  neighborLadies : List HairColor

-- Define the truthfulness of a prisoner
inductive Truthfulness
| AlwaysTruth
| AlwaysLie
| Variable

-- Define the problem setup
def prisonerSetup : List (Prisoner × Truthfulness) := 
  [(Prisoner.P1, Truthfulness.AlwaysTruth),
   (Prisoner.P2, Truthfulness.AlwaysLie),
   (Prisoner.P3, Truthfulness.AlwaysTruth),
   (Prisoner.P4, Truthfulness.AlwaysLie),
   (Prisoner.P5, Truthfulness.Variable)]

-- Define the statements of the prisoners
def prisonerStatements : List Statement := 
  [{ prisoner := Prisoner.P1, 
     ownLady := { name := "Anna", hairColor := HairColor.Blonde },
     neighborLadies := [HairColor.Blonde] },
   { prisoner := Prisoner.P2,
     ownLady := { name := "Brynhild", hairColor := HairColor.Red },
     neighborLadies := [HairColor.Brunette, HairColor.Brunette] },
   { prisoner := Prisoner.P3,
     ownLady := { name := "Clotilde", hairColor := HairColor.Red },
     neighborLadies := [HairColor.Red, HairColor.Red] },
   { prisoner := Prisoner.P4,
     ownLady := { name := "Gudrun", hairColor := HairColor.Red },
     neighborLadies := [HairColor.Brunette, HairColor.Brunette] },
   { prisoner := Prisoner.P5,
     ownLady := { name := "Johanna", hairColor := HairColor.Brunette },
     neighborLadies := [HairColor.Brunette, HairColor.Blonde] }]

-- Define the correct solution
def correctSolution : List Lady := 
  [{ name := "Anna", hairColor := HairColor.Blonde },
   { name := "Brynhild", hairColor := HairColor.Red },
   { name := "Clotilde", hairColor := HairColor.Red },
   { name := "Gudrun", hairColor := HairColor.Red },
   { name := "Johanna", hairColor := HairColor.Brunette }]

-- Theorem statement
theorem prisoners_puzzle_solution :
  ∀ (solution : List Lady),
  (∀ p ∈ prisonerSetup, 
   ∀ s ∈ prisonerStatements,
   p.1 = s.prisoner →
   (p.2 = Truthfulness.AlwaysTruth → 
    (s.ownLady ∈ solution ∧ 
     ∀ c ∈ s.neighborLadies, ∃ l ∈ solution, l.hairColor = c)) ∧
   (p.2 = Truthfulness.AlwaysLie → 
    (s.ownLady ∉ solution ∨ 
     ∃ c ∈ s.neighborLadies, ∀ l ∈ solution, l.hairColor ≠ c))) →
  solution = correctSolution :=
sorry

end NUMINAMATH_CALUDE_prisoners_puzzle_solution_l870_87086


namespace NUMINAMATH_CALUDE_simone_apple_fraction_l870_87023

theorem simone_apple_fraction (x : ℚ) : 
  (16 * x + 15 * (1 / 3 : ℚ) = 13) → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simone_apple_fraction_l870_87023


namespace NUMINAMATH_CALUDE_muffin_cost_muffin_cost_proof_l870_87008

/-- The cost of a muffin given the conditions of Francis and Kiera's breakfast -/
theorem muffin_cost : ℝ → Prop := fun m =>
  let fruit_cup_cost : ℝ := 3
  let francis_breakfast : ℝ := 2 * m + 2 * fruit_cup_cost
  let kiera_breakfast : ℝ := 2 * m + fruit_cup_cost
  let total_cost : ℝ := 17
  francis_breakfast + kiera_breakfast = total_cost → m = 2

/-- Proof of the muffin cost theorem -/
theorem muffin_cost_proof : ∃ m : ℝ, muffin_cost m :=
  sorry

end NUMINAMATH_CALUDE_muffin_cost_muffin_cost_proof_l870_87008


namespace NUMINAMATH_CALUDE_triangle_properties_l870_87037

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a / Real.cos t.A = t.c / (2 - Real.cos t.C) ∧
  t.b = 4 ∧
  t.c = 3 ∧
  (1/2) * t.a * t.b * Real.sin t.C = 3

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.a = 2 ∧ 3 * Real.sin t.C + 4 * Real.cos t.C = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l870_87037


namespace NUMINAMATH_CALUDE_tunnel_length_tunnel_length_specific_l870_87090

/-- Calculates the length of a tunnel given train specifications and travel time -/
theorem tunnel_length (train_length : ℝ) (train_speed_kmh : ℝ) (travel_time_min : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let travel_time_s := travel_time_min * 60
  let total_distance := train_speed_ms * travel_time_s
  let tunnel_length_m := total_distance - train_length
  let tunnel_length_km := tunnel_length_m / 1000
  tunnel_length_km

/-- The length of the tunnel is 1.7 km given the specified conditions -/
theorem tunnel_length_specific : tunnel_length 100 72 1.5 = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_tunnel_length_specific_l870_87090


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l870_87093

theorem polynomial_remainder_theorem (f : ℝ → ℝ) (a b c p q r l m n : ℝ) 
  (h_abc : a * b * c ≠ 0)
  (h_rem1 : ∀ x, ∃ k, f x = k * (x - a) * (x - b) + p * x + l)
  (h_rem2 : ∀ x, ∃ k, f x = k * (x - b) * (x - c) + q * x + m)
  (h_rem3 : ∀ x, ∃ k, f x = k * (x - c) * (x - a) + r * x + n) :
  l * (1/a - 1/b) + m * (1/b - 1/c) + n * (1/c - 1/a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l870_87093


namespace NUMINAMATH_CALUDE_quadrilateral_angle_sum_l870_87091

structure Quadrilateral where
  diagonals_intersect : Bool
  intersection_not_on_side : Bool

def sum_of_angles (q : Quadrilateral) : ℝ :=
  if q.diagonals_intersect ∧ q.intersection_not_on_side then 720 else 0

theorem quadrilateral_angle_sum (q : Quadrilateral) 
  (h1 : q.diagonals_intersect = true) 
  (h2 : q.intersection_not_on_side = true) : 
  sum_of_angles q = 720 := by
  sorry

#check quadrilateral_angle_sum

end NUMINAMATH_CALUDE_quadrilateral_angle_sum_l870_87091


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l870_87088

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  1259 = 23 * q + r ∧ 
  q > 0 ∧ 
  r > 0 ∧
  ∀ (q' r' : ℕ), (1259 = 23 * q' + r' ∧ q' > 0 ∧ r' > 0) → q' - r' ≤ q - r ∧ 
  q - r = 37 := by
  sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l870_87088


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l870_87016

theorem greatest_of_three_consecutive_integers (n : ℤ) :
  n + 2 = 8 → (n < n + 1 ∧ n + 1 < n + 2) → n + 2 = max n (max (n + 1) (n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l870_87016


namespace NUMINAMATH_CALUDE_work_completion_time_l870_87021

theorem work_completion_time (x : ℝ) 
  (h1 : x > 0) 
  (h2 : 1/x + 1/18 = 1/6) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l870_87021


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l870_87012

-- Define set A
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3*a) < 0}

-- Theorem for the first part of the problem
theorem subset_condition (a : ℝ) : 
  A ⊆ (A ∩ B a) ↔ 4/3 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem for the second part of the problem
theorem disjoint_condition (a : ℝ) :
  A ∩ B a = ∅ ↔ a ≤ 2/3 ∨ a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l870_87012
