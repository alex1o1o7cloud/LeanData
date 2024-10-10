import Mathlib

namespace broccoli_production_increase_l1554_155474

def broccoli_production_difference (this_year_production : ℕ) 
  (last_year_side_length : ℕ) : Prop :=
  this_year_production = 1600 ∧
  last_year_side_length * last_year_side_length < this_year_production ∧
  (last_year_side_length + 1) * (last_year_side_length + 1) = this_year_production ∧
  this_year_production - (last_year_side_length * last_year_side_length) = 79

theorem broccoli_production_increase : 
  ∃ (last_year_side_length : ℕ), broccoli_production_difference 1600 last_year_side_length :=
sorry

end broccoli_production_increase_l1554_155474


namespace soy_sauce_bottles_l1554_155408

/-- Represents the amount of soy sauce in ounces -/
def OuncesPerBottle : ℕ := 16

/-- Represents the number of ounces in a cup -/
def OuncesPerCup : ℕ := 8

/-- Represents the amount of soy sauce needed for recipe 1 in cups -/
def Recipe1Cups : ℕ := 2

/-- Represents the amount of soy sauce needed for recipe 2 in cups -/
def Recipe2Cups : ℕ := 1

/-- Represents the amount of soy sauce needed for recipe 3 in cups -/
def Recipe3Cups : ℕ := 3

/-- Calculates the total number of cups needed for all recipes -/
def TotalCups : ℕ := Recipe1Cups + Recipe2Cups + Recipe3Cups

/-- Calculates the total number of ounces needed for all recipes -/
def TotalOunces : ℕ := TotalCups * OuncesPerCup

/-- Calculates the number of bottles needed -/
def BottlesNeeded : ℕ := (TotalOunces + OuncesPerBottle - 1) / OuncesPerBottle

theorem soy_sauce_bottles : BottlesNeeded = 3 := by
  sorry

end soy_sauce_bottles_l1554_155408


namespace julie_school_year_hours_l1554_155489

/-- Calculates the required weekly hours for Julie to earn a target amount during the school year,
    given her summer work details and school year duration. -/
theorem julie_school_year_hours
  (summer_weeks : ℕ)
  (summer_hours_per_week : ℕ)
  (summer_earnings : ℚ)
  (school_year_weeks : ℕ)
  (school_year_target : ℚ)
  (h1 : summer_weeks = 10)
  (h2 : summer_hours_per_week = 60)
  (h3 : summer_earnings = 7500)
  (h4 : school_year_weeks = 50)
  (h5 : school_year_target = 7500) :
  (school_year_target / (summer_earnings / (summer_weeks * summer_hours_per_week))) / school_year_weeks = 12 := by
  sorry

end julie_school_year_hours_l1554_155489


namespace smallest_positive_equivalent_angle_l1554_155490

theorem smallest_positive_equivalent_angle (α : ℝ) : 
  (α > 0 ∧ α < 360 ∧ ∃ k : ℤ, α = 400 - 360 * k) → α = 40 := by
  sorry

end smallest_positive_equivalent_angle_l1554_155490


namespace number_of_students_l1554_155413

def total_pencils : ℕ := 195
def pencils_per_student : ℕ := 3

theorem number_of_students : 
  total_pencils / pencils_per_student = 65 := by
  sorry

end number_of_students_l1554_155413


namespace sandy_payment_l1554_155496

def amount_paid (football_cost baseball_cost change : ℚ) : ℚ :=
  football_cost + baseball_cost + change

theorem sandy_payment (football_cost baseball_cost change : ℚ) 
  (h1 : football_cost = 9.14)
  (h2 : baseball_cost = 6.81)
  (h3 : change = 4.05) :
  amount_paid football_cost baseball_cost change = 20 :=
by sorry

end sandy_payment_l1554_155496


namespace least_integer_square_64_more_than_double_l1554_155487

theorem least_integer_square_64_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 64 ∧ ∀ y : ℤ, y^2 = 2*y + 64 → x ≤ y :=
by sorry

end least_integer_square_64_more_than_double_l1554_155487


namespace family_ages_sum_l1554_155427

theorem family_ages_sum (a b c d : ℕ) (e : ℕ) : 
  a + b + c + d = 114 →  -- Sum of ages 5 years ago plus 20
  e = d - 14 →           -- Age difference between daughter and daughter-in-law
  a + b + c + e + 20 = 120 := by
sorry

end family_ages_sum_l1554_155427


namespace line_slope_intercept_product_l1554_155438

/-- For a line y = mx + b with slope m = 2 and y-intercept b = -3, the product mb is less than -3. -/
theorem line_slope_intercept_product (m b : ℝ) : m = 2 ∧ b = -3 → m * b < -3 := by
  sorry

end line_slope_intercept_product_l1554_155438


namespace intersection_point_of_lines_l1554_155480

theorem intersection_point_of_lines :
  ∃! p : ℝ × ℝ, 
    2 * p.1 + p.2 - 7 = 0 ∧
    p.1 + 2 * p.2 - 5 = 0 ∧
    p = (3, 1) := by
  sorry

end intersection_point_of_lines_l1554_155480


namespace train_length_l1554_155461

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 9 → ∃ length : ℝ, abs (length - 299.97) < 0.01 := by
  sorry

end train_length_l1554_155461


namespace slope_sum_constant_l1554_155483

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the line l passing through (2, 0)
def l (k x y : ℝ) : Prop := y = k * (x - 2)

-- Define point A
def A : ℝ × ℝ := (-3, 0)

-- Define the theorem
theorem slope_sum_constant 
  (k k₁ k₂ x₁ y₁ x₂ y₂ : ℝ) 
  (hM : C x₁ y₁ ∧ l k x₁ y₁) 
  (hN : C x₂ y₂ ∧ l k x₂ y₂) 
  (hk₁ : k₁ = y₁ / (x₁ + 3)) 
  (hk₂ : k₂ = y₂ / (x₂ + 3)) :
  k / k₁ + k / k₂ = -1/2 := by sorry

end slope_sum_constant_l1554_155483


namespace imaginary_part_of_z_l1554_155412

theorem imaginary_part_of_z (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  Complex.im z = -2 := by
  sorry

end imaginary_part_of_z_l1554_155412


namespace cubic_meter_to_cubic_centimeters_l1554_155401

/-- Theorem: One cubic meter is equal to 1,000,000 cubic centimeters -/
theorem cubic_meter_to_cubic_centimeters :
  ∀ (m cm : ℝ), m = 100 * cm → m^3 = 1000000 * cm^3 := by
  sorry

end cubic_meter_to_cubic_centimeters_l1554_155401


namespace kelly_glue_bottles_l1554_155484

theorem kelly_glue_bottles (students : ℕ) (paper_per_student : ℕ) (added_paper : ℕ) (final_supplies : ℕ) :
  students = 8 →
  paper_per_student = 3 →
  added_paper = 5 →
  final_supplies = 20 →
  ∃ (initial_supplies : ℕ) (glue_bottles : ℕ),
    initial_supplies = students * paper_per_student + glue_bottles ∧
    initial_supplies / 2 + added_paper = final_supplies ∧
    glue_bottles = 6 :=
by sorry

end kelly_glue_bottles_l1554_155484


namespace total_apples_calculation_l1554_155462

/-- The number of apples given to each person -/
def apples_per_person : ℝ := 15.0

/-- The number of people who received apples -/
def number_of_people : ℝ := 3.0

/-- The total number of apples given -/
def total_apples : ℝ := apples_per_person * number_of_people

theorem total_apples_calculation : total_apples = 45.0 := by
  sorry

end total_apples_calculation_l1554_155462


namespace martha_cards_l1554_155479

/-- The number of cards Martha has at the end of the process -/
def final_cards (initial : ℕ) (multiplier : ℕ) (given_away : ℕ) : ℕ :=
  initial + multiplier * initial - given_away

/-- Theorem stating that Martha ends up with 1479 cards -/
theorem martha_cards : final_cards 423 3 213 = 1479 := by
  sorry

end martha_cards_l1554_155479


namespace cat_walking_distance_l1554_155428

/-- The distance a cat walks given resistance time, walking rate, and total time -/
theorem cat_walking_distance (resistance_time walking_rate total_time : ℕ) : 
  resistance_time = 20 →
  walking_rate = 8 →
  total_time = 28 →
  (total_time - resistance_time) * walking_rate = 64 := by
  sorry

end cat_walking_distance_l1554_155428


namespace decreasing_power_function_l1554_155468

theorem decreasing_power_function (m : ℝ) : 
  (m^2 - m - 1 > 0) ∧ (m^2 - 2*m - 3 < 0) ↔ m = 2 := by
  sorry

end decreasing_power_function_l1554_155468


namespace extreme_values_and_max_l1554_155411

/-- The function f(x) with parameters a, b, and c -/
def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

/-- The derivative of f(x) with respect to x -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_max (a b c : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (a = -3 ∧ b = 4) ∧
  (c = -2 → ∀ x ∈ Set.Icc 0 3, f a b c x ≤ -7) :=
sorry

end extreme_values_and_max_l1554_155411


namespace nine_people_four_houses_l1554_155451

-- Define the relationship between people, houses, and time
def paint_time (people : ℕ) (houses : ℕ) : ℚ :=
  let rate := (8 : ℚ) * 12 / 3  -- Rate derived from the given condition
  rate * houses / people

-- Theorem statement
theorem nine_people_four_houses :
  paint_time 9 4 = 128 / 9 := by
  sorry

end nine_people_four_houses_l1554_155451


namespace fruit_count_l1554_155431

theorem fruit_count (apples pears tangerines : ℕ) : 
  apples = 45 →
  pears = apples - 21 →
  pears = tangerines - 18 →
  tangerines = 42 := by
sorry

end fruit_count_l1554_155431


namespace sqrt_difference_equals_2sqrt2_l1554_155471

theorem sqrt_difference_equals_2sqrt2 :
  Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3) = 2 * Real.sqrt 2 := by
  sorry

end sqrt_difference_equals_2sqrt2_l1554_155471


namespace rectangle_length_l1554_155478

theorem rectangle_length (P b l A : ℝ) : 
  P / b = 5 → 
  A = 216 → 
  P = 2 * (l + b) → 
  A = l * b → 
  l = 18 :=
by sorry

end rectangle_length_l1554_155478


namespace roberts_score_l1554_155426

/-- Proves that Robert's score is 94 given the conditions of the problem -/
theorem roberts_score (total_students : ℕ) (first_19_avg : ℚ) (new_avg : ℚ) : 
  total_students = 20 → 
  first_19_avg = 74 → 
  new_avg = 75 → 
  (total_students - 1) * first_19_avg + 94 = total_students * new_avg :=
by sorry

end roberts_score_l1554_155426


namespace new_year_weather_probability_l1554_155433

theorem new_year_weather_probability :
  let n : ℕ := 5  -- number of days
  let k : ℕ := 2  -- desired number of clear days
  let p : ℚ := 3/5  -- probability of snow (complement of 60%)

  -- probability of exactly k clear days out of n days
  (n.choose k : ℚ) * p^(n - k) * (1 - p)^k = 1080/3125 :=
by
  sorry

end new_year_weather_probability_l1554_155433


namespace remaining_water_l1554_155421

/-- Calculates the remaining amount of water after an experiment -/
theorem remaining_water (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end remaining_water_l1554_155421


namespace tank_capacity_l1554_155475

theorem tank_capacity (x : ℝ) 
  (h1 : (5/6 : ℝ) * x - 15 = (2/3 : ℝ) * x) : x = 90 := by
  sorry

end tank_capacity_l1554_155475


namespace infinite_series_sum_l1554_155414

/-- The sum of the infinite series Σ(n=1 to ∞) [(3n - 2) / (n(n+1)(n+3))] is equal to 2/21 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n - 2) / (n * (n + 1) * (n + 3))) = 2 / 21 := by
  sorry

end infinite_series_sum_l1554_155414


namespace product_upper_bound_l1554_155424

theorem product_upper_bound (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : x * y + y * z + z * x = 1) : 
  x * z < 1/2 ∧ ∀ ε > 0, ∃ x' y' z' : ℝ, x' ≥ y' ∧ y' ≥ z' ∧ x' * y' + y' * z' + z' * x' = 1 ∧ x' * z' > 1/2 - ε :=
sorry

end product_upper_bound_l1554_155424


namespace mixed_sample_2_prob_is_0_19_expected_tests_plan3_is_2_3756_l1554_155453

-- Define the probability of an animal having the disease
def disease_prob : ℝ := 0.1

-- Define the probability of a mixed sample of 2 animals testing positive
def mixed_sample_2_prob : ℝ := 1 - (1 - disease_prob)^2

-- Define the probability of a mixed sample of 4 animals testing negative
def mixed_sample_4_neg_prob : ℝ := (1 - disease_prob)^4

-- Define the expected number of tests for Plan 3 (mixing all 4 samples)
def expected_tests_plan3 : ℝ := 1 * mixed_sample_4_neg_prob + 5 * (1 - mixed_sample_4_neg_prob)

-- Theorem 1: Probability of positive test for mixed sample of 2 animals
theorem mixed_sample_2_prob_is_0_19 : mixed_sample_2_prob = 0.19 := by sorry

-- Theorem 2: Expected number of tests for Plan 3
theorem expected_tests_plan3_is_2_3756 : expected_tests_plan3 = 2.3756 := by sorry

end mixed_sample_2_prob_is_0_19_expected_tests_plan3_is_2_3756_l1554_155453


namespace number_divided_by_0_08_equals_12_5_l1554_155422

theorem number_divided_by_0_08_equals_12_5 (x : ℝ) : x / 0.08 = 12.5 → x = 1 := by
  sorry

end number_divided_by_0_08_equals_12_5_l1554_155422


namespace min_value_inequality_l1554_155410

theorem min_value_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c) * ((a + b + d)⁻¹ + (a + c + d)⁻¹ + (b + c + d)⁻¹) ≥ (9 : ℝ) / 2 := by
  sorry

end min_value_inequality_l1554_155410


namespace mirror_16_is_8_l1554_155434

/-- Represents a time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Calculates the mirror image of a given time -/
def mirrorTime (t : Time) : Time :=
  { hour := (24 - t.hour) % 24,
    minute := (60 - t.minute) % 60,
    h_valid := by sorry
    m_valid := by sorry }

/-- Theorem: The mirror image of 16:00 is 08:00 -/
theorem mirror_16_is_8 :
  let t : Time := ⟨16, 0, by norm_num, by norm_num⟩
  mirrorTime t = ⟨8, 0, by norm_num, by norm_num⟩ := by sorry

end mirror_16_is_8_l1554_155434


namespace smaller_integer_proof_l1554_155437

theorem smaller_integer_proof (x y : ℤ) : 
  y = 5 * x + 2 →  -- One integer is 2 more than 5 times the other
  y - x = 26 →     -- The difference between the two integers is 26
  x = 6            -- The smaller integer is 6
:= by sorry

end smaller_integer_proof_l1554_155437


namespace comic_book_frames_l1554_155436

theorem comic_book_frames (frames_per_page : ℝ) (pages : ℝ) 
  (h1 : frames_per_page = 143.0) 
  (h2 : pages = 11.0) : 
  frames_per_page * pages = 1573.0 := by
sorry

end comic_book_frames_l1554_155436


namespace prime_factors_of_factorial_30_l1554_155415

theorem prime_factors_of_factorial_30 : 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 10 := by
  sorry

end prime_factors_of_factorial_30_l1554_155415


namespace smallest_number_divisible_by_13_after_subtraction_l1554_155458

theorem smallest_number_divisible_by_13_after_subtraction (N : ℕ) : 
  (∃ k : ℕ, N - 10 = 13 * k) → N ≥ 23 :=
by sorry

end smallest_number_divisible_by_13_after_subtraction_l1554_155458


namespace distance_between_points_l1554_155429

theorem distance_between_points : 
  let p₁ : ℝ × ℝ := (3, 4)
  let p₂ : ℝ × ℝ := (8, -6)
  Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = 5 * Real.sqrt 5 := by
  sorry

end distance_between_points_l1554_155429


namespace opposite_of_negative_five_l1554_155417

theorem opposite_of_negative_five : 
  ∃ x : ℤ, (x + (-5) = 0 ∧ x = 5) :=
by
  sorry

end opposite_of_negative_five_l1554_155417


namespace mountain_climb_theorem_l1554_155439

/-- Represents the mountain climbing scenario -/
structure MountainClimb where
  x : ℝ  -- Height of the mountain in meters
  male_speed : ℝ  -- Speed of male team
  female_speed : ℝ  -- Speed of female team

/-- The main theorem about the mountain climbing scenario -/
theorem mountain_climb_theorem (mc : MountainClimb) 
  (h1 : mc.x / (mc.x - 600) = mc.male_speed / mc.female_speed)  -- Condition when male team reaches summit
  (h2 : mc.male_speed / mc.female_speed = 3 / 2)  -- Speed ratio
  : mc.male_speed / mc.female_speed = 3 / 2  -- 1. Speed ratio is 3:2
  ∧ mc.x = 1800  -- 2. Mountain height is 1800 meters
  ∧ ∀ b : ℝ, b > 0 → b / mc.male_speed < (600 - b) / mc.female_speed → b < 360  -- 3. Point B is less than 360 meters from summit
  := by sorry

end mountain_climb_theorem_l1554_155439


namespace f_bounds_l1554_155442

/-- A function that represents the minimum size of the largest subfamily 
    that doesn't contain a union for n mutually distinct sets -/
noncomputable def f (n : ℕ) : ℝ :=
  Real.sqrt (2 * n) - 1

/-- Theorem stating the bounds for the function f -/
theorem f_bounds (n : ℕ) : 
  Real.sqrt (2 * n) - 1 ≤ f n ∧ f n ≤ 2 * Real.sqrt n + 1 := by
  sorry

#check f_bounds

end f_bounds_l1554_155442


namespace management_subcommittee_count_l1554_155402

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid subcommittees -/
def validSubcommittees (totalMembers managers subcommitteeSize minManagers : ℕ) : ℕ :=
  choose totalMembers subcommitteeSize -
  (choose (totalMembers - managers) subcommitteeSize +
   choose managers 1 * choose (totalMembers - managers) (subcommitteeSize - 1))

theorem management_subcommittee_count :
  validSubcommittees 12 5 5 2 = 596 := by sorry

end management_subcommittee_count_l1554_155402


namespace sum_of_squares_l1554_155400

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/7 := by
  sorry

end sum_of_squares_l1554_155400


namespace constant_ratio_problem_l1554_155493

/-- Given a constant ratio between (2x - 5) and (y + 20), and the condition that y = 6 when x = 7,
    prove that x = 499/52 when y = 21 -/
theorem constant_ratio_problem (k : ℚ) :
  (∀ x y : ℚ, (2 * x - 5) / (y + 20) = k) →
  ((2 * 7 - 5) / (6 + 20) = k) →
  ∃ x : ℚ, (2 * x - 5) / (21 + 20) = k ∧ x = 499 / 52 :=
by sorry

end constant_ratio_problem_l1554_155493


namespace f_value_at_7_minus_a_l1554_155416

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 2 else -Real.log x / Real.log 3

-- Define a as the value where f(a) = -2
noncomputable def a : ℝ :=
  Real.exp (-2 * Real.log 3)

-- Theorem statement
theorem f_value_at_7_minus_a :
  f (7 - a) = -7/4 :=
sorry

end f_value_at_7_minus_a_l1554_155416


namespace six_points_theorem_l1554_155459

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  -- This is a simplified representation
  is_convex : Bool

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  -- Simplified representation of a line
  point1 : Point
  point2 : Point

/-- Calculates the vector between two points -/
def vector (p1 p2 : Point) : Point :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

/-- Checks if a point is on the side of a polygon -/
def is_on_side (p : Point) (poly : ConvexPolygon) : Prop :=
  sorry -- Define the condition for a point to be on the side of the polygon

/-- Calculates the distance between a line and a point -/
def distance_line_point (l : Line) (p : Point) : ℝ :=
  sorry -- Define the distance calculation

theorem six_points_theorem (H : ConvexPolygon) (a : ℝ) 
    (h1 : 0 < a) (h2 : a < 1) :
  ∃ (A1 A2 A3 A4 A5 A6 : Point),
    is_on_side A1 H ∧ is_on_side A2 H ∧ is_on_side A3 H ∧
    is_on_side A4 H ∧ is_on_side A5 H ∧ is_on_side A6 H ∧
    A1 ≠ A2 ∧ A2 ≠ A3 ∧ A3 ≠ A4 ∧ A4 ≠ A5 ∧ A5 ≠ A6 ∧ A6 ≠ A1 ∧
    vector A1 A2 = vector A5 A4 ∧
    vector A1 A2 = vector (Point.mk 0 0) (Point.mk (a * (A6.x - A3.x)) (a * (A6.y - A3.y))) ∧
    distance_line_point (Line.mk A1 A2) A3 = distance_line_point (Line.mk A5 A4) A3 :=
by
  sorry


end six_points_theorem_l1554_155459


namespace vacation_pictures_count_l1554_155432

def zoo_pictures : ℕ := 150
def aquarium_pictures : ℕ := 210
def museum_pictures : ℕ := 90
def amusement_park_pictures : ℕ := 120

def zoo_deletion_percentage : ℚ := 25 / 100
def aquarium_deletion_percentage : ℚ := 15 / 100
def amusement_park_deletion : ℕ := 20
def museum_addition : ℕ := 30

theorem vacation_pictures_count :
  ⌊(zoo_pictures : ℚ) * (1 - zoo_deletion_percentage)⌋ +
  ⌊(aquarium_pictures : ℚ) * (1 - aquarium_deletion_percentage)⌋ +
  (museum_pictures + museum_addition) +
  (amusement_park_pictures - amusement_park_deletion) = 512 := by
  sorry

end vacation_pictures_count_l1554_155432


namespace remainder_7n_mod_4_l1554_155470

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l1554_155470


namespace root_difference_nonnegative_root_difference_l1554_155455

theorem root_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → |r₁ - r₂| = Real.sqrt (b^2 - 4*a*c) / a :=
by sorry

theorem nonnegative_root_difference :
  let eq := fun x : ℝ ↦ x^2 + 40*x + 300
  ∃ r₁ r₂ : ℝ, eq r₁ = 0 ∧ eq r₂ = 0 ∧ |r₁ - r₂| = 20 :=
by sorry

end root_difference_nonnegative_root_difference_l1554_155455


namespace rohan_house_rent_percentage_l1554_155405

/-- Rohan's monthly expenses and savings -/
structure RohanFinances where
  salary : ℝ
  food_percent : ℝ
  entertainment_percent : ℝ
  conveyance_percent : ℝ
  savings : ℝ
  house_rent_percent : ℝ

/-- Theorem stating that Rohan spends 20% of his salary on house rent -/
theorem rohan_house_rent_percentage 
  (rf : RohanFinances)
  (h_salary : rf.salary = 7500)
  (h_food : rf.food_percent = 40)
  (h_entertainment : rf.entertainment_percent = 10)
  (h_conveyance : rf.conveyance_percent = 10)
  (h_savings : rf.savings = 1500)
  (h_total : rf.food_percent + rf.entertainment_percent + rf.conveyance_percent + 
             rf.house_rent_percent + (rf.savings / rf.salary * 100) = 100) :
  rf.house_rent_percent = 20 := by
  sorry

end rohan_house_rent_percentage_l1554_155405


namespace fifth_number_correct_l1554_155435

/-- The function that generates the 5th number on the n-th row of the array -/
def fifthNumber (n : ℕ) : ℚ :=
  (n - 1) * (n - 2) * (n - 3) * (3 * n + 8) / 24

/-- The theorem stating that for n > 5, the 5th number on the n-th row
    of the given array is equal to (n-1)(n-2)(n-3)(3n + 8) / 24 -/
theorem fifth_number_correct (n : ℕ) (h : n > 5) :
  fifthNumber n = (n - 1) * (n - 2) * (n - 3) * (3 * n + 8) / 24 := by
  sorry

end fifth_number_correct_l1554_155435


namespace number_division_remainder_l1554_155465

theorem number_division_remainder (N : ℕ) : 
  N % 5 = 0 ∧ N / 5 = 2 → N % 4 = 2 := by
  sorry

end number_division_remainder_l1554_155465


namespace exists_triangular_numbers_ratio_two_to_one_l1554_155492

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: There exist two triangular numbers with a ratio of 2:1 -/
theorem exists_triangular_numbers_ratio_two_to_one :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ triangular_number m = 2 * triangular_number n :=
by
  sorry

end exists_triangular_numbers_ratio_two_to_one_l1554_155492


namespace solution_is_correct_l1554_155498

def is_valid_triple (a m n : ℕ) : Prop :=
  a ≥ 2 ∧ m ≥ 2 ∧ ∃ k : ℕ, a^n + 203 = k * (a^m + 1)

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(a, m, n) | 
    (∃ k : ℕ, a = 2 ∧ m = 2 ∧ n = 4*k + 1) ∨
    (∃ k : ℕ, a = 2 ∧ m = 3 ∧ n = 6*k + 2) ∨
    (∃ k : ℕ, a = 2 ∧ m = 4 ∧ n = 8*k + 8) ∨
    (∃ k : ℕ, a = 2 ∧ m = 6 ∧ n = 12*k + 9) ∨
    (∃ k : ℕ, a = 3 ∧ m = 2 ∧ n = 4*k + 3) ∨
    (∃ k : ℕ, a = 4 ∧ m = 2 ∧ n = 4*k + 4) ∨
    (∃ k : ℕ, a = 5 ∧ m = 2 ∧ n = 4*k + 1) ∨
    (∃ k : ℕ, a = 8 ∧ m = 2 ∧ n = 4*k + 3) ∨
    (∃ k : ℕ, a = 10 ∧ m = 2 ∧ n = 4*k + 2) ∨
    (∃ k m : ℕ, a = 203 ∧ n = (2*k + 1)*m + 1)}

theorem solution_is_correct :
  ∀ a m n : ℕ, is_valid_triple a m n ↔ (a, m, n) ∈ solution_set :=
sorry

end solution_is_correct_l1554_155498


namespace binomial_coefficient_x6_in_expansion_1_plus_x_8_l1554_155420

theorem binomial_coefficient_x6_in_expansion_1_plus_x_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (1 : ℕ)^(8 - k) * 1^k) = 28 := by
  sorry

end binomial_coefficient_x6_in_expansion_1_plus_x_8_l1554_155420


namespace incorrect_inequality_for_all_reals_l1554_155450

theorem incorrect_inequality_for_all_reals : 
  ¬(∀ x : ℝ, x + (1 / x) ≥ 2 * Real.sqrt (x * (1 / x))) :=
sorry

end incorrect_inequality_for_all_reals_l1554_155450


namespace words_per_page_l1554_155497

theorem words_per_page (total_pages : ℕ) (word_congruence : ℕ) (max_words_per_page : ℕ)
  (h1 : total_pages = 195)
  (h2 : word_congruence = 221)
  (h3 : max_words_per_page = 120)
  (h4 : ∃ (words_per_page : ℕ), 
    (total_pages * words_per_page) % 251 = word_congruence ∧ 
    words_per_page ≤ max_words_per_page) :
  ∃ (words_per_page : ℕ), words_per_page = 41 ∧ 
    (total_pages * words_per_page) % 251 = word_congruence ∧
    words_per_page ≤ max_words_per_page :=
by sorry

end words_per_page_l1554_155497


namespace jessica_quarters_problem_l1554_155430

/-- The number of quarters Jessica's sister gave her -/
def quarters_given (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem jessica_quarters_problem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 8) 
  (h2 : final = 11) : 
  quarters_given initial final = 3 := by
  sorry

end jessica_quarters_problem_l1554_155430


namespace triangle_angle_proof_l1554_155445

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a^2 - (b - c)^2 = b * c →
  Real.cos A * Real.cos B = (Real.sin A + Real.cos C) / 2 →
  A = π / 3 ∧ B = π / 6 :=
by sorry

end triangle_angle_proof_l1554_155445


namespace athlete_subgrid_exists_l1554_155488

/-- Represents a grid of athletes -/
def AthleteGrid := Fin 5 → Fin 49 → Bool

/-- Theorem: In any 5x49 grid of athletes, there exists a 3x3 subgrid of the same gender -/
theorem athlete_subgrid_exists (grid : AthleteGrid) :
  ∃ (i j : Fin 3) (r c : Fin 5),
    (∀ x y, x < i → y < j → grid (r + x) (c + y) = grid r c) :=
  sorry

end athlete_subgrid_exists_l1554_155488


namespace dans_age_l1554_155443

theorem dans_age : ∃ x : ℕ, (x + 18 = 5 * (x - 6)) ∧ (x = 12) := by sorry

end dans_age_l1554_155443


namespace initial_bales_count_l1554_155404

theorem initial_bales_count (added_bales current_total : ℕ) 
  (h1 : added_bales = 26)
  (h2 : current_total = 54)
  : current_total - added_bales = 28 := by
  sorry

end initial_bales_count_l1554_155404


namespace no_solution_for_equation_l1554_155481

theorem no_solution_for_equation : ¬∃ (x : ℝ), 1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5) := by
  sorry

end no_solution_for_equation_l1554_155481


namespace min_sugar_amount_l1554_155446

theorem min_sugar_amount (f s : ℕ) : 
  (f ≥ 9 + s / 2) → 
  (f ≤ 3 * s) → 
  (∃ (f : ℕ), f ≥ 9 + s / 2 ∧ f ≤ 3 * s) → 
  s ≥ 4 := by
sorry

end min_sugar_amount_l1554_155446


namespace wire_average_length_l1554_155485

theorem wire_average_length :
  let total_wires : ℕ := 6
  let third_wires : ℕ := total_wires / 3
  let remaining_wires : ℕ := total_wires - third_wires
  let avg_length_third : ℝ := 70
  let avg_length_remaining : ℝ := 85
  let total_length : ℝ := (third_wires : ℝ) * avg_length_third + (remaining_wires : ℝ) * avg_length_remaining
  let overall_avg_length : ℝ := total_length / (total_wires : ℝ)
  overall_avg_length = 80 := by
sorry

end wire_average_length_l1554_155485


namespace intersection_points_l1554_155447

/-- A periodic function with period 2 that equals x^2 on [-1, 1] -/
noncomputable def f : ℝ → ℝ := sorry

/-- The number of intersection points between f and |log₅(x)| -/
def num_intersections : ℕ := sorry

theorem intersection_points :
  (∀ x, f (x + 2) = f x) ∧ 
  (∀ x ∈ Set.Icc (-1) 1, f x = x^2) →
  num_intersections = 5 := by sorry

end intersection_points_l1554_155447


namespace circle_radius_in_triangle_l1554_155407

/-- Triangle DEF with specified side lengths -/
structure Triangle where
  de : ℝ
  df : ℝ
  ef : ℝ
  h_de : de = 64
  h_df : df = 64
  h_ef : ef = 72

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Tangency relation between circle and line segment -/
def IsTangent (c : Circle) (a b : ℝ × ℝ) : Prop := sorry

/-- External tangency relation between two circles -/
def IsExternallyTangent (c1 c2 : Circle) : Prop := sorry

/-- A circle is inside a triangle -/
def IsInside (c : Circle) (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem circle_radius_in_triangle (t : Triangle) (r s : Circle) :
  t.de = 64 →
  t.df = 64 →
  t.ef = 72 →
  r.radius = 20 →
  IsTangent r (0, 0) (t.df, 0) →
  IsTangent r (t.ef, 0) (0, 0) →
  IsExternallyTangent s r →
  IsTangent s (0, 0) (t.de, 0) →
  IsTangent s (t.ef, 0) (0, 0) →
  IsInside s t →
  s.radius = 52 - 4 * Real.sqrt 41 := by
  sorry

end circle_radius_in_triangle_l1554_155407


namespace max_sum_of_digits_l1554_155477

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem max_sum_of_digits (A B C D : ℕ) : 
  is_digit A → is_digit B → is_digit C → is_digit D →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (C + D) % 2 = 0 →
  (A + B) % (C + D) = 0 →
  A + B ≤ 16 :=
by sorry

end max_sum_of_digits_l1554_155477


namespace unique_seven_l1554_155449

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if all numbers in the grid are unique and between 1 and 9 -/
def valid_numbers (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 9 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

/-- Check if the sum of each row, column, and diagonal is 18 -/
def sum_18 (g : Grid) : Prop :=
  (∀ i, g i 0 + g i 1 + g i 2 = 18) ∧  -- rows
  (∀ j, g 0 j + g 1 j + g 2 j = 18) ∧  -- columns
  (g 0 0 + g 1 1 + g 2 2 = 18) ∧       -- main diagonal
  (g 0 2 + g 1 1 + g 2 0 = 18)         -- other diagonal

/-- The main theorem -/
theorem unique_seven (g : Grid) 
  (h1 : valid_numbers g) 
  (h2 : sum_18 g) 
  (h3 : g 0 0 = 6) 
  (h4 : g 2 2 = 1) : 
  ∃! (i j : Fin 3), g i j = 7 :=
sorry

end unique_seven_l1554_155449


namespace double_inequality_solution_l1554_155452

theorem double_inequality_solution (x : ℝ) : 
  (4 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 9 * x + 3) ↔ 
  (x > 3 + 2 * Real.sqrt 2 ∧ x < 5.5 + Real.sqrt 32.25) :=
sorry

end double_inequality_solution_l1554_155452


namespace negation_of_existence_negation_of_proposition_l1554_155499

theorem negation_of_existence (P : ℝ → Prop) :
  (¬∃ x, P x) ↔ (∀ x, ¬P x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end negation_of_existence_negation_of_proposition_l1554_155499


namespace shannon_stones_l1554_155491

/-- The number of heart-shaped stones Shannon wants in each bracelet -/
def stones_per_bracelet : ℝ := 8.0

/-- The number of bracelets Shannon can make -/
def number_of_bracelets : ℕ := 6

/-- The total number of heart-shaped stones Shannon brought -/
def total_stones : ℝ := stones_per_bracelet * (number_of_bracelets : ℝ)

theorem shannon_stones :
  total_stones = 48.0 := by sorry

end shannon_stones_l1554_155491


namespace unique_a_value_l1554_155476

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

theorem unique_a_value : 
  ∃! a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ (Set.univ \ B)) ∧ 
            (∀ x : ℝ, x ∈ (Set.univ \ B) → a - 1 < x ∧ x < a + 1) := by
  sorry

end unique_a_value_l1554_155476


namespace max_popsicles_lucy_can_buy_l1554_155448

theorem max_popsicles_lucy_can_buy (lucy_money : ℝ) (popsicle_price : ℝ) :
  lucy_money = 19.23 →
  popsicle_price = 1.60 →
  ∃ n : ℕ, n * popsicle_price ≤ lucy_money ∧
    ∀ m : ℕ, m * popsicle_price ≤ lucy_money → m ≤ n ∧
    n = 12 :=
by sorry

end max_popsicles_lucy_can_buy_l1554_155448


namespace minimum_additional_marbles_lisa_additional_marbles_l1554_155469

theorem minimum_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := (num_friends * (num_friends + 1)) / 2
  if required_marbles > initial_marbles then
    required_marbles - initial_marbles
  else
    0

theorem lisa_additional_marbles :
  minimum_additional_marbles 12 40 = 38 := by
  sorry

end minimum_additional_marbles_lisa_additional_marbles_l1554_155469


namespace complex_multiplication_l1554_155467

theorem complex_multiplication (i : ℂ) : i * i = -1 →
  (1/2 : ℂ) + (Real.sqrt 3/2 : ℂ) * i * ((Real.sqrt 3/2 : ℂ) + (1/2 : ℂ) * i) = i := by
  sorry

end complex_multiplication_l1554_155467


namespace fraction_equation_solution_l1554_155406

theorem fraction_equation_solution (n : ℚ) : 
  2 / (n + 2) + 3 / (n + 2) + 2 * n / (n + 2) = 4 → n = -3/2 := by
  sorry

end fraction_equation_solution_l1554_155406


namespace max_gcd_bn_l1554_155456

def b (n : ℕ) : ℚ := (15^n - 1) / 14

theorem max_gcd_bn (n : ℕ) : Nat.gcd (Nat.floor (b n)) (Nat.floor (b (n + 1))) = 1 := by
  sorry

end max_gcd_bn_l1554_155456


namespace fraction_equality_l1554_155472

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (2 * x + 3 * y) / (x - 2 * y) = 3) : 
  (x + 2 * y) / (2 * x - y) = 11 / 17 := by
sorry

end fraction_equality_l1554_155472


namespace right_triangle_other_leg_l1554_155418

theorem right_triangle_other_leg 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- right triangle condition
  (h_a : a = 9)                -- one leg is 9 cm
  (h_c : c = 15)               -- hypotenuse is 15 cm
  : b = 12 := by               -- prove other leg is 12 cm
  sorry

end right_triangle_other_leg_l1554_155418


namespace surprise_shop_revenue_loss_l1554_155440

/-- Calculates the potential revenue loss for a shop during Christmas holiday closures over multiple years. -/
def potential_revenue_loss (days_closed : ℕ) (daily_revenue : ℕ) (years : ℕ) : ℕ :=
  days_closed * daily_revenue * years

/-- Proves that the total potential revenue lost by the "Surprise" shop during 6 years of Christmas holiday closures is $90,000. -/
theorem surprise_shop_revenue_loss :
  potential_revenue_loss 3 5000 6 = 90000 := by
  sorry

end surprise_shop_revenue_loss_l1554_155440


namespace rational_solutions_quadratic_l1554_155464

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + 9 * k = 0) ↔ k = 4 :=
sorry

end rational_solutions_quadratic_l1554_155464


namespace units_digit_of_difference_is_seven_l1554_155466

-- Define a three-digit number
def ThreeDigitNumber (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

-- Define the relationship between hundreds and units digits
def HundredsUnitsRelation (a c : ℕ) : Prop :=
  a = c - 3

-- Define the original number
def OriginalNumber (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

-- Define the reversed number
def ReversedNumber (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a

-- Theorem: The units digit of the difference is 7
theorem units_digit_of_difference_is_seven 
  (a b c : ℕ) 
  (h1 : ThreeDigitNumber a b c) 
  (h2 : HundredsUnitsRelation a c) : 
  (OriginalNumber a b c - ReversedNumber a b c) % 10 = 7 := by
  sorry

end units_digit_of_difference_is_seven_l1554_155466


namespace car_distance_difference_l1554_155486

/-- Calculates the distance traveled by a car given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the problem of two cars traveling at different speeds -/
theorem car_distance_difference 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (time : ℝ) 
  (h1 : speed_A = 60) 
  (h2 : speed_B = 45) 
  (h3 : time = 5) : 
  distance speed_A time - distance speed_B time = 75 := by
sorry

end car_distance_difference_l1554_155486


namespace quadratic_root_property_l1554_155460

theorem quadratic_root_property (a : ℝ) : 
  a^2 + 2*a - 3 = 0 → 2*a^2 + 4*a = 6 := by
  sorry

end quadratic_root_property_l1554_155460


namespace mans_age_percentage_l1554_155419

/-- Given a man's age satisfying certain conditions, prove that his present age is 125% of what it was 10 years ago. -/
theorem mans_age_percentage (present_age : ℕ) (future_age : ℕ) (past_age : ℕ) : 
  present_age = 50 ∧ 
  present_age = (5 : ℚ) / 6 * future_age ∧ 
  present_age = past_age + 10 →
  (present_age : ℚ) / past_age = 5 / 4 := by
sorry


end mans_age_percentage_l1554_155419


namespace most_accurate_reading_l1554_155403

def scale_start : ℝ := 10.25
def scale_end : ℝ := 10.5
def arrow_position : ℝ := 10.3  -- Approximate position based on the problem description

def options : List ℝ := [10.05, 10.15, 10.25, 10.3, 10.6]

theorem most_accurate_reading :
  scale_start < arrow_position ∧ 
  arrow_position < scale_end ∧
  |arrow_position - 10.3| < |arrow_position - ((scale_start + scale_end) / 2)| →
  (options.filter (λ x => x ≥ scale_start ∧ x ≤ scale_end)).argmin (λ x => |x - arrow_position|) = some 10.3 := by
  sorry

end most_accurate_reading_l1554_155403


namespace double_reflection_of_D_l1554_155482

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflects a point across the line y = x + 1 -/
def reflect_y_eq_x_plus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)  -- Translate down by 1
  let p'' := (p'.2, p'.1)   -- Reflect across y = x
  (p''.1, p''.2 + 1)        -- Translate back up by 1

/-- The main theorem -/
theorem double_reflection_of_D (D : ℝ × ℝ) (h : D = (4, 1)) :
  reflect_y_eq_x_plus_1 (reflect_x D) = (-2, 5) := by
  sorry


end double_reflection_of_D_l1554_155482


namespace visit_neither_country_l1554_155494

theorem visit_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 60 → iceland = 35 → norway = 23 → both = 31 →
  total - (iceland + norway - both) = 33 := by
sorry

end visit_neither_country_l1554_155494


namespace negation_equivalence_l1554_155454

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^4 - x^3 + x^2 + 5 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^4 - x₀^3 + x₀^2 + 5 > 0) := by
  sorry

end negation_equivalence_l1554_155454


namespace log_roll_volume_l1554_155457

theorem log_roll_volume (log_length : ℝ) (large_radius small_radius : ℝ) :
  log_length = 10 ∧ 
  large_radius = 3 ∧ 
  small_radius = 1 →
  let path_radius := large_radius + small_radius
  let cross_section_area := π * large_radius^2 + π * path_radius^2 / 2 - π * small_radius^2 / 2
  cross_section_area * log_length = 155 * π :=
by sorry

end log_roll_volume_l1554_155457


namespace pizza_payment_difference_l1554_155495

/-- Pizza sharing problem -/
theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let plain_cost : ℚ := 12
  let mushroom_cost : ℚ := 3
  let pepperoni_cost : ℚ := 5
  let bob_slices : ℕ := 8
  let anne_slices : ℕ := 3
  let total_cost : ℚ := plain_cost + mushroom_cost + pepperoni_cost
  let bob_cost : ℚ := total_cost - (anne_slices : ℚ) * (plain_cost / total_slices)
  let anne_cost : ℚ := (anne_slices : ℚ) * (plain_cost / total_slices)
  bob_cost - anne_cost = 14 :=
by sorry

end pizza_payment_difference_l1554_155495


namespace gcd_of_polynomials_l1554_155409

theorem gcd_of_polynomials (a : ℤ) (h : ∃ k : ℤ, a = 720 * k) :
  Int.gcd (a^2 + 8*a + 18) (a + 6) = 6 := by
  sorry

end gcd_of_polynomials_l1554_155409


namespace disprove_seventh_power_conjecture_l1554_155425

theorem disprove_seventh_power_conjecture :
  144^7 + 110^7 + 84^7 + 27^7 = 206^7 := by
  sorry

end disprove_seventh_power_conjecture_l1554_155425


namespace solution_count_condition_condition_implies_solution_count_l1554_155423

/-- The system of equations has three or two solutions if and only if a = ±1 or a = ±√2 -/
theorem solution_count_condition (a : ℝ) : 
  (∃ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) ∧
  (∀ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1 → 
    (x = 0 ∨ x ≠ 0) ∧ (y = 0 ∨ y ≠ 0)) →
  (a = 1 ∨ a = -1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by sorry

/-- If a = ±1 or a = ±√2, then the system has three or two solutions -/
theorem condition_implies_solution_count (a : ℝ) 
  (h : a = 1 ∨ a = -1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :
  (∃ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) ∧
  (∀ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1 → 
    (x = 0 ∨ x ≠ 0) ∧ (y = 0 ∨ y ≠ 0)) :=
by sorry

end solution_count_condition_condition_implies_solution_count_l1554_155423


namespace factorization_proof_l1554_155441

theorem factorization_proof (x y : ℝ) : x^2*y - 2*x*y^2 + y^3 = y*(x-y)^2 := by
  sorry

end factorization_proof_l1554_155441


namespace determinant_inequality_solution_l1554_155473

-- Define the determinant
def det (a b c d : ℝ) : ℝ := |a * d - b * c|

-- Define the logarithm base sqrt(2)
noncomputable def log_sqrt2 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 2)

-- Define the solution set
def solution_set : Set ℝ := {x | x ∈ (Set.Ioo 0 1) ∪ (Set.Ioo 1 2)}

-- State the theorem
theorem determinant_inequality_solution :
  {x : ℝ | log_sqrt2 (det 1 11 1 x) < 0} = solution_set :=
by sorry

end determinant_inequality_solution_l1554_155473


namespace fraction_simplification_l1554_155463

theorem fraction_simplification (a : ℝ) (h : a ≠ 0) : (a - 1) / a + 1 / a = 1 := by
  sorry

end fraction_simplification_l1554_155463


namespace garden_length_is_fifty_l1554_155444

/-- Represents a rectangular garden with a given width and length. -/
structure RectangularGarden where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangular garden. -/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.width + g.length)

/-- Theorem: A rectangular garden with length twice its width and perimeter 150 yards has a length of 50 yards. -/
theorem garden_length_is_fifty {g : RectangularGarden} 
  (h1 : g.length = 2 * g.width) 
  (h2 : perimeter g = 150) : 
  g.length = 50 := by
  sorry

end garden_length_is_fifty_l1554_155444
