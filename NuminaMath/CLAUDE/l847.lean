import Mathlib

namespace NUMINAMATH_CALUDE_percentage_equality_l847_84778

theorem percentage_equality (x : ℝ) : 
  (15 / 100) * 75 = (x / 100) * 450 → x = 2.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_equality_l847_84778


namespace NUMINAMATH_CALUDE_cake_baking_fraction_l847_84708

theorem cake_baking_fraction (total_cakes : ℕ) 
  (h1 : total_cakes = 60) 
  (initially_baked : ℕ) 
  (h2 : initially_baked = total_cakes / 2) 
  (first_day_baked : ℕ) 
  (h3 : first_day_baked = (total_cakes - initially_baked) / 2) 
  (second_day_baked : ℕ) 
  (h4 : total_cakes - initially_baked - first_day_baked - second_day_baked = 10) :
  (second_day_baked : ℚ) / (total_cakes - initially_baked - first_day_baked) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cake_baking_fraction_l847_84708


namespace NUMINAMATH_CALUDE_calculation_proof_l847_84779

theorem calculation_proof : (24 / (8 + 2 - 5)) * 7 = 33.6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l847_84779


namespace NUMINAMATH_CALUDE_min_value_of_z_l847_84726

theorem min_value_of_z (x y : ℝ) : 
  ∃ (m : ℝ), m = 8 ∧ ∀ (x y : ℝ), 3*x^2 + 4*y^2 + 12*x - 8*y + 3*x*y + 30 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_z_l847_84726


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l847_84759

theorem quadratic_always_positive : ∀ x : ℝ, 3 * x^2 - 2 * x + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l847_84759


namespace NUMINAMATH_CALUDE_acid_dilution_l847_84788

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution results in a 25% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.40 →
  added_water = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l847_84788


namespace NUMINAMATH_CALUDE_apple_cost_problem_l847_84716

theorem apple_cost_problem (l q : ℝ) : 
  (30 * l + 3 * q = 168) →
  (30 * l + 6 * q = 186) →
  (∀ k, k ≤ 30 → k * l = k * 5) →
  20 * l = 100 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_problem_l847_84716


namespace NUMINAMATH_CALUDE_basketball_team_selection_l847_84784

def team_size : ℕ := 15
def captain_count : ℕ := 2
def lineup_size : ℕ := 5

theorem basketball_team_selection :
  (team_size.choose captain_count) * 
  (team_size - captain_count).factorial / (team_size - captain_count - lineup_size).factorial = 16201200 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l847_84784


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l847_84701

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 3*a - 1 = 0) → 
  (b^3 - 3*b - 1 = 0) → 
  (c^3 - 3*c - 1 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -9 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l847_84701


namespace NUMINAMATH_CALUDE_train_length_l847_84795

/-- Given a train that crosses a signal post in 40 seconds and takes 2 minutes to cross a 1.8 kilometer
    long bridge at constant speed, the length of the train is 900 meters. -/
theorem train_length (signal_time : ℝ) (bridge_time : ℝ) (bridge_length : ℝ) :
  signal_time = 40 →
  bridge_time = 120 →
  bridge_length = 1800 →
  ∃ (train_length : ℝ) (speed : ℝ),
    train_length = speed * signal_time ∧
    train_length + bridge_length = speed * bridge_time ∧
    train_length = 900 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l847_84795


namespace NUMINAMATH_CALUDE_greatest_missed_problems_to_pass_l847_84739

/-- The number of problems on the test -/
def total_problems : ℕ := 50

/-- The minimum percentage required to pass the test -/
def passing_percentage : ℚ := 85 / 100

/-- The greatest number of problems that can be missed while still passing -/
def max_missed_problems : ℕ := 7

theorem greatest_missed_problems_to_pass :
  max_missed_problems = 
    (total_problems - Int.floor (passing_percentage * total_problems : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_missed_problems_to_pass_l847_84739


namespace NUMINAMATH_CALUDE_conference_games_l847_84787

/-- The number of games in a complete season for a sports conference --/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * inter_division_games
  (total_teams * games_per_team) / 2

/-- Theorem stating the number of games in the specific conference setup --/
theorem conference_games : total_games 16 8 3 2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_l847_84787


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l847_84709

/-- A geometric sequence with common ratio q > 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  (4 * (a 2016)^2 - 8 * (a 2016) + 3 = 0) →
  (4 * (a 2017)^2 - 8 * (a 2017) + 3 = 0) →
  a 2018 + a 2019 = 18 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l847_84709


namespace NUMINAMATH_CALUDE_valid_choices_count_l847_84750

/-- Represents a point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle in the plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a straight line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The set of 9 points created by the intersection of two lines and two circles -/
def intersection_points : Finset Point := sorry

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if three points lie on the same circle -/
def on_same_circle (p q r : Point) (c1 c2 : Circle) : Prop := sorry

/-- The number of ways to choose 4 points from the intersection points
    such that no 3 of them are collinear or on the same circle -/
def valid_choices : ℕ := sorry

theorem valid_choices_count :
  valid_choices = 114 :=
sorry

end NUMINAMATH_CALUDE_valid_choices_count_l847_84750


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l847_84794

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 5 }

/-- Theorem: The maximum number of soap boxes that can fit in the carton is 300 -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 300 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l847_84794


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l847_84770

theorem negation_of_existential_proposition :
  (¬ ∃ a : ℝ, a ≥ -1 ∧ Real.log (Real.exp n + 1) > 1/2) ↔
  (∀ a : ℝ, a ≥ -1 → Real.log (Real.exp n + 1) ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l847_84770


namespace NUMINAMATH_CALUDE_ratio_transformation_l847_84764

theorem ratio_transformation (y : ℚ) : 
  (2 + y : ℚ) / (3 + y) = 4 / 5 → (2 + y = 4 ∧ 3 + y = 5) := by
  sorry

#check ratio_transformation

end NUMINAMATH_CALUDE_ratio_transformation_l847_84764


namespace NUMINAMATH_CALUDE_f_10_equals_756_l847_84773

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

theorem f_10_equals_756 : f 10 = 756 := by sorry

end NUMINAMATH_CALUDE_f_10_equals_756_l847_84773


namespace NUMINAMATH_CALUDE_roger_has_more_candy_l847_84748

-- Define the number of bags for Sandra and Roger
def sandra_bags : ℕ := 2
def roger_bags : ℕ := 2

-- Define the number of candies in each of Sandra's bags
def sandra_candy_per_bag : ℕ := 6

-- Define the number of candies in Roger's bags
def roger_candy_bag1 : ℕ := 11
def roger_candy_bag2 : ℕ := 3

-- Calculate the total number of candies for Sandra and Roger
def sandra_total : ℕ := sandra_bags * sandra_candy_per_bag
def roger_total : ℕ := roger_candy_bag1 + roger_candy_bag2

-- State the theorem
theorem roger_has_more_candy : roger_total = sandra_total + 2 := by
  sorry

end NUMINAMATH_CALUDE_roger_has_more_candy_l847_84748


namespace NUMINAMATH_CALUDE_integral_sin_cos_l847_84744

theorem integral_sin_cos : 
  ∫ x in (0)..(2*Real.pi/3), (1 + Real.sin x) / (1 + Real.cos x + Real.sin x) = Real.pi/3 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin_cos_l847_84744


namespace NUMINAMATH_CALUDE_train_length_calculation_l847_84732

/-- Calculate the length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60 →
  person_speed = 6 →
  passing_time = 6 →
  (train_speed + person_speed) * passing_time * (1000 / 3600) = 110.04 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l847_84732


namespace NUMINAMATH_CALUDE_N_is_composite_l847_84724

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Nat.Prime N := by
  sorry

end NUMINAMATH_CALUDE_N_is_composite_l847_84724


namespace NUMINAMATH_CALUDE_bad_carrots_count_bad_carrots_problem_l847_84747

theorem bad_carrots_count (nancy_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : ℕ :=
  let total_carrots := nancy_carrots + mom_carrots
  total_carrots - good_carrots

theorem bad_carrots_problem :
  bad_carrots_count 38 47 71 = 14 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_bad_carrots_problem_l847_84747


namespace NUMINAMATH_CALUDE_evelyn_family_without_daughters_l847_84760

/-- Represents the family structure of Evelyn and her descendants -/
structure EvelynFamily where
  daughters : ℕ
  granddaughters : ℕ
  daughters_with_daughters : ℕ
  daughters_per_mother : ℕ

/-- The actual family structure of Evelyn -/
def evelyn_family : EvelynFamily :=
  { daughters := 8,
    granddaughters := 36 - 8,
    daughters_with_daughters := (36 - 8) / 7,
    daughters_per_mother := 7 }

/-- The number of Evelyn's daughters and granddaughters who have no daughters -/
def women_without_daughters (f : EvelynFamily) : ℕ :=
  (f.daughters - f.daughters_with_daughters) + f.granddaughters

theorem evelyn_family_without_daughters :
  women_without_daughters evelyn_family = 32 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_family_without_daughters_l847_84760


namespace NUMINAMATH_CALUDE_johns_number_l847_84721

def is_two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem johns_number 
  (t m j d : ℕ) 
  (h1 : is_two_digit_prime t)
  (h2 : is_two_digit_prime m)
  (h3 : is_two_digit_prime j)
  (h4 : is_two_digit_prime d)
  (h5 : t ≠ m ∧ t ≠ j ∧ t ≠ d ∧ m ≠ j ∧ m ≠ d ∧ j ≠ d)
  (h6 : t + j = 26)
  (h7 : m + d = 32)
  (h8 : j + d = 34)
  (h9 : t + d = 36) : 
  j = 13 := by sorry

end NUMINAMATH_CALUDE_johns_number_l847_84721


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l847_84740

theorem tan_alpha_plus_pi_fourth (α : Real) (h : Real.tan α = Real.sqrt 3) :
  Real.tan (α + π/4) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l847_84740


namespace NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l847_84717

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the property of line m passing through focus and intersecting the parabola
def line_intersects_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A ∧ parabola B ∧ 
  (A.1 - focus.1) * (B.2 - focus.2) = (B.1 - focus.1) * (A.2 - focus.2)

-- Define the condition |AF| + |BF| = 10
def distance_sum_condition (A B : ℝ × ℝ) : Prop :=
  Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2) +
  Real.sqrt ((B.1 - focus.1)^2 + (B.2 - focus.2)^2) = 10

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (A B : ℝ × ℝ) 
  (h1 : line_intersects_parabola A B) 
  (h2 : distance_sum_condition A B) : 
  (A.1 + B.1) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l847_84717


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l847_84792

-- Problem 1
theorem problem_1 : (1) - 6 - 13 + (-24) = -43 := by sorry

-- Problem 2
theorem problem_2 : (-6) / (3/7) * (-7) = 98 := by sorry

-- Problem 3
theorem problem_3 : (2/3 - 1/12 - 1/15) * (-60) = -31 := by sorry

-- Problem 4
theorem problem_4 : -1^4 - 1/6 * (2 - (-3)^2) = 1/6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l847_84792


namespace NUMINAMATH_CALUDE_sqrt_450_simplified_l847_84771

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplified_l847_84771


namespace NUMINAMATH_CALUDE_harry_blue_weights_l847_84745

/-- Represents the weight configuration on a gym bar -/
structure WeightConfig where
  blue_weight : ℕ  -- Weight of each blue weight in pounds
  green_weight : ℕ  -- Weight of each green weight in pounds
  num_green : ℕ  -- Number of green weights
  bar_weight : ℕ  -- Weight of the bar in pounds
  total_weight : ℕ  -- Total weight in pounds

/-- Calculates the number of blue weights given a weight configuration -/
def num_blue_weights (config : WeightConfig) : ℕ :=
  (config.total_weight - config.bar_weight - config.num_green * config.green_weight) / config.blue_weight

/-- Theorem stating that Harry's configuration results in 4 blue weights -/
theorem harry_blue_weights :
  let config : WeightConfig := {
    blue_weight := 2,
    green_weight := 3,
    num_green := 5,
    bar_weight := 2,
    total_weight := 25
  }
  num_blue_weights config = 4 := by sorry

end NUMINAMATH_CALUDE_harry_blue_weights_l847_84745


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l847_84754

-- Problem 1
theorem problem_1 (a b c : ℚ) : (-6 * a^2 * b^5 * c) / (-2 * a * b^2)^2 = 3/2 * b * c := by sorry

-- Problem 2
theorem problem_2 (m n : ℚ) : (-3*m - 2*n) * (3*m + 2*n) = -9*m^2 - 12*m*n - 4*n^2 := by sorry

-- Problem 3
theorem problem_3 (x y : ℚ) (h : y ≠ 0) : ((x - 2*y)^2 - (x - 2*y)*(x + 2*y)) / (2*y) = -2*x + 4*y := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l847_84754


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l847_84702

theorem equal_roots_quadratic (k B : ℝ) : 
  k = 1 → (∃ x : ℝ, 2 * k * x^2 + B * x + 2 = 0 ∧ 
    ∀ y : ℝ, 2 * k * y^2 + B * y + 2 = 0 → y = x) → 
  B = 4 ∨ B = -4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l847_84702


namespace NUMINAMATH_CALUDE_jordan_oreos_l847_84727

theorem jordan_oreos (jordan : ℕ) (james : ℕ) : 
  james = 2 * jordan + 3 →
  jordan + james = 36 →
  jordan = 11 := by
sorry

end NUMINAMATH_CALUDE_jordan_oreos_l847_84727


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l847_84720

theorem complex_magnitude_theorem (a b : ℂ) (x : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = x - 6 * Complex.I →
  x = 3 * Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l847_84720


namespace NUMINAMATH_CALUDE_yuna_has_most_points_l847_84776

-- Define the point totals for each person
def yoongi_points : ℕ := 7
def jungkook_points : ℕ := 6
def yuna_points : ℕ := 9
def yoojung_points : ℕ := 8

-- Theorem stating that Yuna has the largest number of points
theorem yuna_has_most_points :
  yuna_points ≥ yoongi_points ∧
  yuna_points ≥ jungkook_points ∧
  yuna_points ≥ yoojung_points :=
by sorry

end NUMINAMATH_CALUDE_yuna_has_most_points_l847_84776


namespace NUMINAMATH_CALUDE_range_sum_of_bounds_l847_84729

open Set Real

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

theorem range_sum_of_bounds (a b : ℝ) :
  (∀ y ∈ Set.range h, a < y ∧ y ≤ b) ∧
  (∀ ε > 0, ∃ y ∈ Set.range h, y < a + ε) ∧
  (b ∈ Set.range h) →
  a + b = 1 := by sorry

end NUMINAMATH_CALUDE_range_sum_of_bounds_l847_84729


namespace NUMINAMATH_CALUDE_function_negative_on_interval_l847_84743

/-- The function f(x) = x^2 + mx - 1 is negative on [m, m+1] iff m is in (-√2/2, 0) -/
theorem function_negative_on_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), x^2 + m*x - 1 < 0) ↔ 
  m ∈ Set.Ioo (-(Real.sqrt 2)/2) 0 :=
sorry

end NUMINAMATH_CALUDE_function_negative_on_interval_l847_84743


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_three_l847_84767

theorem sum_of_fractions_geq_three (a b c : ℝ) (h : a * b * c = 1) :
  (1 + a + a * b) / (1 + b + a * b) +
  (1 + b + b * c) / (1 + c + b * c) +
  (1 + c + a * c) / (1 + a + a * c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_three_l847_84767


namespace NUMINAMATH_CALUDE_sin_15_cos_15_equals_quarter_l847_84791

theorem sin_15_cos_15_equals_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_equals_quarter_l847_84791


namespace NUMINAMATH_CALUDE_ellipse_condition_l847_84707

/-- Represents an ellipse equation with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  (5 - m > 0) ∧ (m + 3 > 0) ∧ (5 - m ≠ m + 3)

/-- The condition -3 < m < 5 -/
def condition (m : ℝ) : Prop :=
  -3 < m ∧ m < 5

theorem ellipse_condition (m : ℝ) :
  (is_ellipse m → condition m) ∧ 
  ¬(condition m → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l847_84707


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l847_84736

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h_different_lines : m ≠ n)
  (h_different_planes : α ≠ β)
  (h_n_subset_β : subset n β)
  (h_m_parallel_n : parallel m n)
  (h_m_perp_α : perpendicular m α) :
  perpendicular_planes α β := by
  sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l847_84736


namespace NUMINAMATH_CALUDE_fraction_division_multiplication_l847_84753

theorem fraction_division_multiplication :
  (3 : ℚ) / 7 / 4 * 2 = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_multiplication_l847_84753


namespace NUMINAMATH_CALUDE_power_inequality_l847_84718

theorem power_inequality (n : ℕ) (hn : n > 3) : n^(n+1) > (n+1)^n := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l847_84718


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l847_84734

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 4 * x * y) :
  1 / x + 1 / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l847_84734


namespace NUMINAMATH_CALUDE_factorization_difference_l847_84793

theorem factorization_difference (a b : ℤ) : 
  (∀ y : ℤ, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) → 
  a - b = -7 := by
sorry

end NUMINAMATH_CALUDE_factorization_difference_l847_84793


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l847_84731

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ := sorry

/-- The extended quadrilateral formed by extending the sides of the original quadrilateral -/
noncomputable def extendedQuadrilateral (q : Quadrilateral) : Quadrilateral := sorry

/-- Theorem: The area of the extended quadrilateral is five times the area of the original quadrilateral -/
theorem extended_quadrilateral_area (q : Quadrilateral) :
  area (extendedQuadrilateral q) = 5 * area q := by sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l847_84731


namespace NUMINAMATH_CALUDE_other_factor_is_five_l847_84758

def w : ℕ := 120

theorem other_factor_is_five :
  ∀ (product : ℕ),
  (∃ (k : ℕ), product = 936 * w * k) →
  (∃ (m : ℕ), product = 2^5 * 3^3 * m) →
  (∀ (x : ℕ), x < w → ¬(∃ (y : ℕ), 936 * x * y = product)) →
  (∃ (n : ℕ), product = 936 * w * 5 * n) :=
by sorry

end NUMINAMATH_CALUDE_other_factor_is_five_l847_84758


namespace NUMINAMATH_CALUDE_young_in_specific_sample_l847_84789

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  young_population : ℕ
  sample_size : ℕ

/-- Calculates the number of young people in a stratified sample -/
def young_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.young_population) / s.total_population

/-- Theorem stating the number of young people in the specific stratified sample -/
theorem young_in_specific_sample :
  let s : StratifiedSample := {
    total_population := 108,
    young_population := 51,
    sample_size := 36
  }
  young_in_sample s = 17 := by
  sorry

end NUMINAMATH_CALUDE_young_in_specific_sample_l847_84789


namespace NUMINAMATH_CALUDE_fedya_deposit_l847_84783

theorem fedya_deposit (n : ℕ) (hn : 0 < n ∧ n < 30) : 
  (∃ (x : ℕ), x * (100 - n) = 847 * 100) → 
  (∃ (x : ℕ), x * (100 - n) = 847 * 100 ∧ x = 1100) :=
by sorry

end NUMINAMATH_CALUDE_fedya_deposit_l847_84783


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l847_84723

theorem students_taking_neither_music_nor_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (both : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 20) 
  (h3 : art = 20) 
  (h4 : both = 10) : 
  total - (music + art - both) = 470 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l847_84723


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l847_84746

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m) → 
    n ≤ m) ∧
  n = 210 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l847_84746


namespace NUMINAMATH_CALUDE_carries_text_messages_l847_84761

/-- The number of text messages Carrie sends to her brother on Saturday -/
def saturday_messages : ℕ := 5

/-- The number of text messages Carrie sends to her brother on Sunday -/
def sunday_messages : ℕ := 5

/-- The number of text messages Carrie sends to her brother on each weekday -/
def weekday_messages : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks we are considering -/
def weeks : ℕ := 4

/-- The total number of text messages Carrie sends to her brother over 4 full weeks -/
def total_messages : ℕ := 80

theorem carries_text_messages :
  saturday_messages + sunday_messages +
  weekday_messages * weekdays_per_week * weeks = total_messages := by
  sorry

end NUMINAMATH_CALUDE_carries_text_messages_l847_84761


namespace NUMINAMATH_CALUDE_apple_problem_l847_84705

/-- Represents the types of Red Fuji apples --/
inductive AppleType
  | A
  | B

/-- Represents the purchase and selling prices for each apple type --/
def price (t : AppleType) : ℕ × ℕ :=
  match t with
  | AppleType.A => (28, 42)
  | AppleType.B => (22, 34)

/-- Represents the total number of apples purchased in the first batch --/
def totalApples : ℕ := 30

/-- Represents the total cost of the first batch of apples --/
def totalCost : ℕ := 720

/-- Represents the maximum number of apples to be purchased in the second batch --/
def maxApples : ℕ := 80

/-- Represents the maximum cost allowed for the second batch --/
def maxCost : ℕ := 2000

/-- Represents the initial daily sales of type B apples at original price --/
def initialSales : ℕ := 4

/-- Represents the increase in daily sales for every 1 yuan price reduction --/
def salesIncrease : ℕ := 2

/-- Represents the target daily profit for type B apples --/
def targetProfit : ℕ := 90

theorem apple_problem :
  ∃ (x y : ℕ),
    x + y = totalApples ∧
    x * (price AppleType.A).1 + y * (price AppleType.B).1 = totalCost ∧
    ∃ (m : ℕ),
      m ≤ maxApples ∧
      m * (price AppleType.A).1 + (maxApples - m) * (price AppleType.B).1 ≤ maxCost ∧
      ∀ (k : ℕ),
        k ≤ maxApples →
        k * (price AppleType.A).1 + (maxApples - k) * (price AppleType.B).1 ≤ maxCost →
        m * ((price AppleType.A).2 - (price AppleType.A).1) +
          (maxApples - m) * ((price AppleType.B).2 - (price AppleType.B).1) ≥
        k * ((price AppleType.A).2 - (price AppleType.A).1) +
          (maxApples - k) * ((price AppleType.B).2 - (price AppleType.B).1) ∧
    ∃ (a : ℕ),
      (initialSales + salesIncrease * a) * ((price AppleType.B).2 - a - (price AppleType.B).1) = targetProfit :=
by
  sorry


end NUMINAMATH_CALUDE_apple_problem_l847_84705


namespace NUMINAMATH_CALUDE_flags_on_circular_track_l847_84737

/-- The number of flags needed on a circular track -/
def num_flags (track_length : ℕ) (flag_interval : ℕ) : ℕ :=
  (track_length / flag_interval) + 1

/-- Theorem: 5 flags are needed for a 400m track with 90m intervals -/
theorem flags_on_circular_track :
  num_flags 400 90 = 5 := by
  sorry

end NUMINAMATH_CALUDE_flags_on_circular_track_l847_84737


namespace NUMINAMATH_CALUDE_amount_received_by_B_l847_84781

/-- Theorem: Given a total amount of 1440, if A receives 1/3 as much as B, and B receives 1/4 as much as C, then B receives 202.5. -/
theorem amount_received_by_B (total : ℝ) (a b c : ℝ) : 
  total = 1440 →
  a = (1/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  b = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_amount_received_by_B_l847_84781


namespace NUMINAMATH_CALUDE_negation_of_proposition_l847_84763

theorem negation_of_proposition (A B : Set α) :
  ¬(∀ x, x ∈ A ∩ B → x ∈ A ∨ x ∈ B) ↔ (∃ x, x ∉ A ∩ B ∧ x ∉ A ∧ x ∉ B) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l847_84763


namespace NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l847_84796

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l847_84796


namespace NUMINAMATH_CALUDE_x_plus_two_equals_seven_implies_x_equals_five_l847_84790

theorem x_plus_two_equals_seven_implies_x_equals_five :
  ∀ x : ℝ, x + 2 = 7 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_two_equals_seven_implies_x_equals_five_l847_84790


namespace NUMINAMATH_CALUDE_marc_total_spending_l847_84752

/-- The total amount spent by Marc on his purchases -/
def total_spent (model_car_price : ℕ) (paint_bottle_price : ℕ) (paintbrush_price : ℕ) 
  (model_car_quantity : ℕ) (paint_bottle_quantity : ℕ) (paintbrush_quantity : ℕ) : ℕ :=
  model_car_price * model_car_quantity + 
  paint_bottle_price * paint_bottle_quantity + 
  paintbrush_price * paintbrush_quantity

/-- Theorem stating that Marc's total spending is $160 -/
theorem marc_total_spending :
  total_spent 20 10 2 5 5 5 = 160 := by
  sorry

end NUMINAMATH_CALUDE_marc_total_spending_l847_84752


namespace NUMINAMATH_CALUDE_sin_cos_sum_formula_l847_84730

theorem sin_cos_sum_formula (α β : ℝ) : 
  Real.sin α * Real.sin β - Real.cos α * Real.cos β = - Real.cos (α + β) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_formula_l847_84730


namespace NUMINAMATH_CALUDE_jackson_decorations_given_l847_84703

/-- Given that Mrs. Jackson has 4 boxes of Christmas decorations with 15 decorations in each box
    and she used 35 decorations, prove that she gave 25 decorations to her neighbor. -/
theorem jackson_decorations_given (boxes : ℕ) (decorations_per_box : ℕ) (used_decorations : ℕ)
    (h1 : boxes = 4)
    (h2 : decorations_per_box = 15)
    (h3 : used_decorations = 35) :
    boxes * decorations_per_box - used_decorations = 25 := by
  sorry

end NUMINAMATH_CALUDE_jackson_decorations_given_l847_84703


namespace NUMINAMATH_CALUDE_student_arrangements_l847_84715

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def adjacent_arrangement (n : ℕ) : ℕ := 2 * factorial (n - 1)

def non_adjacent_arrangement (n : ℕ) : ℕ := factorial (n - 2) * (n * (n - 1))

def special_arrangement (n : ℕ) : ℕ := 
  factorial n - 3 * factorial (n - 1) + 2 * factorial (n - 2)

theorem student_arrangements :
  adjacent_arrangement 5 = 48 ∧
  non_adjacent_arrangement 5 = 72 ∧
  special_arrangement 5 = 60 := by
  sorry

#eval adjacent_arrangement 5
#eval non_adjacent_arrangement 5
#eval special_arrangement 5

end NUMINAMATH_CALUDE_student_arrangements_l847_84715


namespace NUMINAMATH_CALUDE_hat_cloak_color_probability_l847_84711

/-- The number of possible hat colors for sixth-graders -/
def num_hat_colors : ℕ := 2

/-- The number of possible cloak colors for seventh-graders -/
def num_cloak_colors : ℕ := 3

/-- The total number of possible color combinations -/
def total_combinations : ℕ := num_hat_colors * num_cloak_colors

/-- The number of combinations where hat and cloak colors are different -/
def different_color_combinations : ℕ := num_hat_colors * (num_cloak_colors - 1)

/-- The probability of hat and cloak colors being different -/
def prob_different_colors : ℚ := different_color_combinations / total_combinations

theorem hat_cloak_color_probability :
  prob_different_colors = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_hat_cloak_color_probability_l847_84711


namespace NUMINAMATH_CALUDE_cubic_line_bounded_area_l847_84775

/-- The area bounded by a cubic function and a line -/
noncomputable def boundedArea (a b c d p q α β γ : ℝ) : ℝ :=
  |a| / 12 * (γ - α)^3 * |2*β - γ - α|

/-- Theorem stating the area bounded by a cubic function and a line -/
theorem cubic_line_bounded_area
  (a b c d p q α β γ : ℝ)
  (h_a : a ≠ 0)
  (h_order : α < β ∧ β < γ)
  (h_cubic : ∀ x, a*x^3 + b*x^2 + c*x + d = p*x + q → x = α ∨ x = β ∨ x = γ) :
  ∃ A, A = boundedArea a b c d p q α β γ ∧
    A = |∫ (x : ℝ) in α..γ, (a*x^3 + b*x^2 + c*x + d) - (p*x + q)| :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_line_bounded_area_l847_84775


namespace NUMINAMATH_CALUDE_mansion_rooms_less_than_55_l847_84728

/-- Represents the number of rooms with a specific type of bouquet -/
structure BouquetRooms where
  roses : ℕ
  carnations : ℕ
  chrysanthemums : ℕ

/-- Represents the number of rooms with combinations of bouquets -/
structure OverlapRooms where
  carnations_chrysanthemums : ℕ
  chrysanthemums_roses : ℕ
  carnations_roses : ℕ

theorem mansion_rooms_less_than_55 (b : BouquetRooms) (o : OverlapRooms) 
    (h1 : b.roses = 30)
    (h2 : b.carnations = 20)
    (h3 : b.chrysanthemums = 10)
    (h4 : o.carnations_chrysanthemums = 2)
    (h5 : o.chrysanthemums_roses = 3)
    (h6 : o.carnations_roses = 4) :
    b.roses + b.carnations + b.chrysanthemums - 
    o.carnations_chrysanthemums - o.chrysanthemums_roses - o.carnations_roses < 55 := by
  sorry


end NUMINAMATH_CALUDE_mansion_rooms_less_than_55_l847_84728


namespace NUMINAMATH_CALUDE_repeating_decimal_length_1_221_l847_84798

theorem repeating_decimal_length_1_221 : ∃ n : ℕ, n > 0 ∧ n = 48 ∧ ∀ k : ℕ, (10^k - 1) % 221 = 0 ↔ n ∣ k := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_length_1_221_l847_84798


namespace NUMINAMATH_CALUDE_quadratic_roots_l847_84749

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 1
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l847_84749


namespace NUMINAMATH_CALUDE_cubic_from_quadratic_roots_l847_84755

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots r and s,
    this theorem states the form of the cubic equation
    with roots r^2 + br + a and s^2 + bs + a. -/
theorem cubic_from_quadratic_roots (a b c r s : ℝ) : 
  (a * r^2 + b * r + c = 0) →
  (a * s^2 + b * s + c = 0) →
  (r ≠ s) →
  ∃ (p q : ℝ), 
    (x^3 + (b^2 - a*b^2 + 4*a^3 - 2*a*c)/a^2 * x^2 + p*x + q = 0) ↔ 
    (x = r^2 + b*r + a ∨ x = s^2 + b*s + a ∨ x = -(r^2 + b*r + a + s^2 + b*s + a)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_from_quadratic_roots_l847_84755


namespace NUMINAMATH_CALUDE_only_negative_one_squared_is_negative_l847_84714

theorem only_negative_one_squared_is_negative :
  ((-1 : ℝ)^0 < 0 ∨ |-1| < 0 ∨ Real.sqrt 1 < 0 ∨ -(1^2) < 0) ∧
  ((-1 : ℝ)^0 ≥ 0 ∧ |-1| ≥ 0 ∧ Real.sqrt 1 ≥ 0) ∧
  (-(1^2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_one_squared_is_negative_l847_84714


namespace NUMINAMATH_CALUDE_probability_standard_bulb_l847_84700

/-- Probability of a bulb being from factory 1 -/
def p_factory1 : ℝ := 0.45

/-- Probability of a bulb being from factory 2 -/
def p_factory2 : ℝ := 0.40

/-- Probability of a bulb being from factory 3 -/
def p_factory3 : ℝ := 0.15

/-- Probability of a bulb from factory 1 being standard -/
def p_standard_factory1 : ℝ := 0.70

/-- Probability of a bulb from factory 2 being standard -/
def p_standard_factory2 : ℝ := 0.80

/-- Probability of a bulb from factory 3 being standard -/
def p_standard_factory3 : ℝ := 0.81

/-- The probability of purchasing a standard bulb from the store -/
theorem probability_standard_bulb :
  p_factory1 * p_standard_factory1 + 
  p_factory2 * p_standard_factory2 + 
  p_factory3 * p_standard_factory3 = 0.7565 := by
  sorry

end NUMINAMATH_CALUDE_probability_standard_bulb_l847_84700


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l847_84797

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 2 * x) = 5 → x = 21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l847_84797


namespace NUMINAMATH_CALUDE_ceiling_equation_solution_l847_84765

theorem ceiling_equation_solution :
  ∃! b : ℝ, b + ⌈b⌉ = 21.5 ∧ b = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_equation_solution_l847_84765


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l847_84774

theorem arithmetic_square_root_of_nine : ∃ x : ℝ, x ≥ 0 ∧ x^2 = 9 ∧ ∀ y : ℝ, y ≥ 0 ∧ y^2 = 9 → y = x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l847_84774


namespace NUMINAMATH_CALUDE_equation_solution_l847_84735

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 - ⌊x₁⌋ = 2019) ∧ 
    (x₂^2 - ⌊x₂⌋ = 2019) ∧ 
    (x₁ = -Real.sqrt 1974) ∧ 
    (x₂ = Real.sqrt 2064) ∧ 
    (∀ (x : ℝ), x^2 - ⌊x⌋ = 2019 → x = x₁ ∨ x = x₂) :=
by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l847_84735


namespace NUMINAMATH_CALUDE_symmetry_about_x_axis_l847_84722

/-- A point in the 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis for two points. -/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The theorem stating that (2, -3) is symmetric to (2, 3) with respect to the x-axis. -/
theorem symmetry_about_x_axis :
  symmetricAboutXAxis (Point.mk 2 3) (Point.mk 2 (-3)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_x_axis_l847_84722


namespace NUMINAMATH_CALUDE_jeanne_initial_tickets_l847_84757

/-- The cost of all three attractions in tickets -/
def total_cost : ℕ := 13

/-- The number of additional tickets Jeanne needs to buy -/
def additional_tickets : ℕ := 8

/-- Jeanne's initial number of tickets -/
def initial_tickets : ℕ := total_cost - additional_tickets

theorem jeanne_initial_tickets : initial_tickets = 5 := by
  sorry

end NUMINAMATH_CALUDE_jeanne_initial_tickets_l847_84757


namespace NUMINAMATH_CALUDE_num_closed_lockers_l847_84786

/-- The number of lockers and students -/
def n : ℕ := 100

/-- A locker is open if and only if its number is a perfect square -/
def is_open (k : ℕ) : Prop := ∃ m : ℕ, k = m^2

/-- The number of perfect squares less than or equal to n -/
def num_perfect_squares (n : ℕ) : ℕ := (n.sqrt : ℕ)

/-- The main theorem: The number of closed lockers is equal to
    the total number of lockers minus the number of perfect squares -/
theorem num_closed_lockers : 
  n - (num_perfect_squares n) = 90 := by sorry

end NUMINAMATH_CALUDE_num_closed_lockers_l847_84786


namespace NUMINAMATH_CALUDE_paulas_travel_time_fraction_l847_84777

theorem paulas_travel_time_fraction (luke_bus_time : ℕ) (total_travel_time : ℕ) 
  (h1 : luke_bus_time = 70)
  (h2 : total_travel_time = 504) :
  ∃ f : ℚ, 
    f = 3/5 ∧ 
    (luke_bus_time + 5 * luke_bus_time + 2 * (f * luke_bus_time) : ℚ) = total_travel_time :=
by sorry

end NUMINAMATH_CALUDE_paulas_travel_time_fraction_l847_84777


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l847_84719

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 3 * Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = Real.sqrt 91 ∧ θ = Real.arctan (3 * Real.sqrt 3 / 8) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l847_84719


namespace NUMINAMATH_CALUDE_shaded_square_area_l847_84738

/-- Represents a figure with five squares and two right-angled triangles -/
structure GeometricFigure where
  square1 : ℝ
  square2 : ℝ
  square3 : ℝ
  square4 : ℝ
  square5 : ℝ

/-- The theorem stating the area of the shaded square -/
theorem shaded_square_area (fig : GeometricFigure) 
  (h1 : fig.square1 = 5)
  (h2 : fig.square2 = 8)
  (h3 : fig.square3 = 32) :
  fig.square5 = 45 := by
  sorry


end NUMINAMATH_CALUDE_shaded_square_area_l847_84738


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l847_84762

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l847_84762


namespace NUMINAMATH_CALUDE_product_is_three_l847_84780

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1/3

/-- The product of the repeating decimal 0.333... and 9 --/
def product : ℚ := repeating_third * 9

/-- Theorem stating that the product of 0.333... and 9 is equal to 3 --/
theorem product_is_three : product = 3 := by sorry

end NUMINAMATH_CALUDE_product_is_three_l847_84780


namespace NUMINAMATH_CALUDE_cyclist_speed_l847_84742

theorem cyclist_speed (speed : ℝ) : 
  (speed ≥ 0) →                           -- Non-negative speed
  (5 * speed + 5 * speed = 50) →          -- Total distance after 5 hours
  (speed = 5) :=                          -- Speed of each cyclist
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_l847_84742


namespace NUMINAMATH_CALUDE_horner_V3_value_l847_84741

-- Define the polynomial coefficients
def a : List ℤ := [12, 35, -8, 79, 6, 5, 3]

-- Define Horner's method for a single step
def horner_step (v : ℤ) (x : ℤ) (a : ℤ) : ℤ := v * x + a

-- Define the function to compute V_3 using Horner's method
def compute_V3 (coeffs : List ℤ) (x : ℤ) : ℤ :=
  let v0 := coeffs.reverse.head!
  let v1 := horner_step v0 x (coeffs.reverse.tail!.head!)
  let v2 := horner_step v1 x (coeffs.reverse.tail!.tail!.head!)
  horner_step v2 x (coeffs.reverse.tail!.tail!.tail!.head!)

-- State the theorem
theorem horner_V3_value :
  compute_V3 a (-4) = -57 := by sorry

end NUMINAMATH_CALUDE_horner_V3_value_l847_84741


namespace NUMINAMATH_CALUDE_fixed_deposit_equation_l847_84751

theorem fixed_deposit_equation (x : ℝ) : 
  (∀ (interest_rate deposit_tax_rate final_amount : ℝ),
    interest_rate = 0.0198 →
    deposit_tax_rate = 0.20 →
    final_amount = 1300 →
    x + interest_rate * x * (1 - deposit_tax_rate) = final_amount) :=
by sorry

end NUMINAMATH_CALUDE_fixed_deposit_equation_l847_84751


namespace NUMINAMATH_CALUDE_number_of_grey_stones_l847_84704

/-- Given a collection of stones with specific properties, prove the number of grey stones. -/
theorem number_of_grey_stones 
  (total_stones : ℕ) 
  (white_stones : ℕ) 
  (green_stones : ℕ) 
  (h1 : total_stones = 100)
  (h2 : white_stones = 60)
  (h3 : green_stones = 60)
  (h4 : white_stones > total_stones - white_stones)
  (h5 : (white_stones : ℚ) / (total_stones - white_stones) = (grey_stones : ℚ) / green_stones) :
  grey_stones = 90 :=
by
  sorry

#check number_of_grey_stones

end NUMINAMATH_CALUDE_number_of_grey_stones_l847_84704


namespace NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l847_84769

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n > 0) →
  (a 4 * a 14 = 8) →
  (∀ x y : ℝ, 2 * a 7 + a 11 ≥ x + y → x * y ≤ 16) :=
by sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l847_84769


namespace NUMINAMATH_CALUDE_inequality_with_gcd_l847_84725

theorem inequality_with_gcd (a b : ℕ) (h1 : a > b) (h2 : (a - b) ∣ (a^2 + b)) :
  (a + 1) / (b + 1 : ℝ) ≤ Nat.gcd a b + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_with_gcd_l847_84725


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l847_84713

theorem smallest_angle_solution (x : ℝ) : 
  (∀ y ∈ Set.Ioo 0 x, 8 * Real.sin y * Real.cos y ^ 6 - 8 * Real.sin y ^ 6 * Real.cos y ≠ 2) ∧ 
  (8 * Real.sin x * Real.cos x ^ 6 - 8 * Real.sin x ^ 6 * Real.cos x = 2) → 
  x = π / 16 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l847_84713


namespace NUMINAMATH_CALUDE_complex_number_subtraction_l847_84772

theorem complex_number_subtraction : (5 * Complex.I) - (2 + 2 * Complex.I) = -2 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_subtraction_l847_84772


namespace NUMINAMATH_CALUDE_direct_proportion_function_l847_84768

/-- A function that is directly proportional to 2x+3 and passes through the point (1, -5) -/
def f (x : ℝ) : ℝ := -2 * x - 3

theorem direct_proportion_function :
  (∃ k : ℝ, ∀ x, f x = k * (2 * x + 3)) ∧
  f 1 = -5 ∧
  (∀ x, f x = -2 * x - 3) ∧
  (f (5/2) = 2) := by
  sorry

#check direct_proportion_function

end NUMINAMATH_CALUDE_direct_proportion_function_l847_84768


namespace NUMINAMATH_CALUDE_square_equation_solutions_l847_84785

theorem square_equation_solutions (n : ℝ) :
  ∃ (x y : ℝ), x ≠ y ∧
  (n - (2 * n + 1) / 2)^2 = ((n + 1) - (2 * n + 1) / 2)^2 ∧
  (x = n - (2 * n + 1) / 2 ∧ y = (n + 1) - (2 * n + 1) / 2) ∨
  (x = n - (2 * n + 1) / 2 ∧ y = -((n + 1) - (2 * n + 1) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_square_equation_solutions_l847_84785


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l847_84710

theorem condition_necessary_not_sufficient :
  (∀ a : ℝ, a < -1 → 1/a > -1) ∧
  (∃ a : ℝ, 1/a > -1 ∧ ¬(a < -1)) := by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l847_84710


namespace NUMINAMATH_CALUDE_biff_wifi_cost_l847_84712

/-- Proves the hourly cost of WiFi for Biff to break even on a 3-hour bus trip -/
theorem biff_wifi_cost (ticket : ℝ) (snacks : ℝ) (headphones : ℝ) (hourly_rate : ℝ) 
  (trip_duration : ℝ) :
  ticket = 11 →
  snacks = 3 →
  headphones = 16 →
  hourly_rate = 12 →
  trip_duration = 3 →
  ∃ (wifi_cost : ℝ),
    wifi_cost = 2 ∧
    trip_duration * hourly_rate = ticket + snacks + headphones + trip_duration * wifi_cost :=
by sorry

end NUMINAMATH_CALUDE_biff_wifi_cost_l847_84712


namespace NUMINAMATH_CALUDE_polynomial_existence_l847_84733

theorem polynomial_existence (n : ℕ+) :
  ∃ (f g : Polynomial ℤ), (f * (X + 1) ^ (2 ^ n.val) + g * (X ^ (2 ^ n.val) + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_existence_l847_84733


namespace NUMINAMATH_CALUDE_employee_distribution_percentage_difference_l847_84782

theorem employee_distribution_percentage_difference :
  let total_degrees : ℝ := 360
  let manufacturing_degrees : ℝ := 162
  let sales_degrees : ℝ := 108
  let research_degrees : ℝ := 54
  let admin_degrees : ℝ := 36
  let manufacturing_percent := (manufacturing_degrees / total_degrees) * 100
  let sales_percent := (sales_degrees / total_degrees) * 100
  let research_percent := (research_degrees / total_degrees) * 100
  let admin_percent := (admin_degrees / total_degrees) * 100
  let max_percent := max manufacturing_percent (max sales_percent (max research_percent admin_percent))
  let min_percent := min manufacturing_percent (min sales_percent (min research_percent admin_percent))
  (max_percent - min_percent) = 35 := by
  sorry

end NUMINAMATH_CALUDE_employee_distribution_percentage_difference_l847_84782


namespace NUMINAMATH_CALUDE_problem_statement_l847_84766

theorem problem_statement (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x ≠ y)
  (h : (y * z - x^2) / (1 - x) = (x * z - y^2) / (1 - y)) :
  (y * z - x^2) / (1 - x) = x + y + z := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l847_84766


namespace NUMINAMATH_CALUDE_haley_marbles_l847_84706

/-- The number of boys who love to play marbles -/
def num_marble_boys : ℕ := 13

/-- The number of marbles each boy receives -/
def marbles_per_boy : ℕ := 2

/-- The total number of marbles Haley has -/
def total_marbles : ℕ := num_marble_boys * marbles_per_boy

theorem haley_marbles : total_marbles = 26 := by
  sorry

end NUMINAMATH_CALUDE_haley_marbles_l847_84706


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l847_84799

/-- The number of ways to place 4 different balls into 4 different boxes --/
def placeBalls (emptyBoxes : Nat) : Nat :=
  if emptyBoxes = 1 then 144
  else if emptyBoxes = 2 then 84
  else 0

theorem ball_placement_theorem :
  (placeBalls 1 = 144) ∧ (placeBalls 2 = 84) := by
  sorry

#eval placeBalls 1  -- Expected output: 144
#eval placeBalls 2  -- Expected output: 84

end NUMINAMATH_CALUDE_ball_placement_theorem_l847_84799


namespace NUMINAMATH_CALUDE_inequality_solution_l847_84756

theorem inequality_solution (b : ℝ) (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → ∃ a : ℝ, x * |x - a| + b < 0) :
  ((-1 ≤ b ∧ b < 2 * Real.sqrt 2 - 3 →
    ∃ a : ℝ, a ∈ Set.Ioo (1 + b) (2 * Real.sqrt (-b))) ∧
   (b < -1 →
    ∃ a : ℝ, a ∈ Set.Ioo (1 + b) (1 - b))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l847_84756
