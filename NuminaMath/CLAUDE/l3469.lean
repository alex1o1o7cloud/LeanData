import Mathlib

namespace NUMINAMATH_CALUDE_project_completion_l3469_346915

/-- Represents the work rate of the men (amount of work done per day per person) -/
def work_rate : ℝ := 1

/-- Represents the total amount of work in the project -/
def total_work : ℝ := 1

/-- The number of days it takes the original group to complete the project -/
def original_days : ℕ := 40

/-- The number of days it takes the reduced group to complete the project -/
def reduced_days : ℕ := 50

/-- The number of men removed from the original group -/
def men_removed : ℕ := 5

theorem project_completion (M : ℕ) : 
  (M : ℝ) * work_rate * original_days = total_work ∧
  ((M : ℝ) - men_removed) * work_rate * reduced_days = total_work →
  M = 25 := by
sorry

end NUMINAMATH_CALUDE_project_completion_l3469_346915


namespace NUMINAMATH_CALUDE_range_of_a_l3469_346987

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0
def q (x a : ℝ) : Prop := x^2 - (2*a - 1)*x + a*(a - 1) ≥ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (0 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3469_346987


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l3469_346982

/-- An isosceles triangle with specific measurements -/
structure IsoscelesTriangle where
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The inradius of the triangle -/
  inradius : ℝ
  /-- One of the angles of the triangle in degrees -/
  angle : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : Bool
  /-- The perimeter is 20 cm -/
  perimeterIs20 : perimeter = 20
  /-- The inradius is 2.5 cm -/
  inradiusIs2_5 : inradius = 2.5
  /-- One angle is 40 degrees -/
  angleIs40 : angle = 40
  /-- The triangle is confirmed to be isosceles -/
  isIsoscelesTrue : isIsosceles = true

/-- The area of the isosceles triangle is 25 cm² -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) : 
  t.inradius * (t.perimeter / 2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l3469_346982


namespace NUMINAMATH_CALUDE_cube_difference_positive_l3469_346990

theorem cube_difference_positive {a b : ℝ} (h : a > b) : a^3 - b^3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_positive_l3469_346990


namespace NUMINAMATH_CALUDE_lcm_problem_l3469_346929

theorem lcm_problem (a b c : ℕ) : 
  lcm a b = 24 → lcm b c = 28 → 
  ∃ (m : ℕ), m = lcm a c ∧ ∀ (n : ℕ), n = lcm a c → m ≤ n := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l3469_346929


namespace NUMINAMATH_CALUDE_fraction_calculation_l3469_346945

theorem fraction_calculation (x y : ℚ) (hx : x = 4/6) (hy : y = 8/12) :
  (6*x + 8*y) / (48*x*y) = 7/16 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3469_346945


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3469_346912

theorem complex_fraction_evaluation :
  (⌈(19 : ℚ) / 7 - ⌈(35 : ℚ) / 19⌉⌉) / (⌈(35 : ℚ) / 7 + ⌈(7 * 19 : ℚ) / 35⌉⌉) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3469_346912


namespace NUMINAMATH_CALUDE_statements_c_and_d_are_correct_l3469_346999

theorem statements_c_and_d_are_correct :
  (∀ a b c : ℝ, c^2 > 0 → a*c^2 > b*c^2 → a > b) ∧
  (∀ a b m : ℝ, a > b → b > 0 → m > 0 → (b+m)/(a+m) > b/a) :=
by sorry

end NUMINAMATH_CALUDE_statements_c_and_d_are_correct_l3469_346999


namespace NUMINAMATH_CALUDE_cube_sum_equality_l3469_346911

theorem cube_sum_equality (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a^3 + 10) / a^2 = (b^3 + 10) / b^2 ∧
  (b^3 + 10) / b^2 = (c^3 + 10) / c^2 →
  a^3 + b^3 + c^3 = 1301 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l3469_346911


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l3469_346939

/-- The number of ways to distribute indistinguishable objects into distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of new ice cream flavors -/
def new_flavors : ℕ := distribute 6 5

theorem ice_cream_flavors : new_flavors = 210 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l3469_346939


namespace NUMINAMATH_CALUDE_school_play_chairs_l3469_346930

theorem school_play_chairs (chairs_per_row : ℕ) (unoccupied_seats : ℕ) (occupied_seats : ℕ) :
  chairs_per_row = 20 →
  unoccupied_seats = 10 →
  occupied_seats = 790 →
  (occupied_seats + unoccupied_seats) / chairs_per_row = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_school_play_chairs_l3469_346930


namespace NUMINAMATH_CALUDE_frustum_center_height_for_specific_pyramid_l3469_346949

/-- Represents a rectangular pyramid with a parallel cut -/
structure CutPyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ
  volume_ratio : ℝ  -- ratio of smaller pyramid to whole pyramid

/-- Calculate the distance from the center of the frustum's circumsphere to the base -/
def frustum_center_height (p : CutPyramid) : ℝ :=
  sorry

/-- The main theorem -/
theorem frustum_center_height_for_specific_pyramid :
  let p : CutPyramid := {
    base_length := 15,
    base_width := 20,
    height := 30,
    volume_ratio := 1/9
  }
  abs (frustum_center_height p - 25.73) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_frustum_center_height_for_specific_pyramid_l3469_346949


namespace NUMINAMATH_CALUDE_problem_solution_l3469_346969

theorem problem_solution (x y z : ℚ) : 
  x / (y + 1) = 4 / 5 → 
  3 * z = 2 * x + y → 
  y = 10 → 
  z = 46 / 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3469_346969


namespace NUMINAMATH_CALUDE_average_of_combined_sets_l3469_346901

theorem average_of_combined_sets (m n : ℕ) (a b : ℝ) :
  let sum_m := m * a
  let sum_n := n * b
  (sum_m + sum_n) / (m + n) = (a * m + b * n) / (m + n) := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_sets_l3469_346901


namespace NUMINAMATH_CALUDE_bill_bouquets_theorem_l3469_346927

/-- Represents the number of roses in a bouquet Bill buys -/
def roses_per_bouquet_buy : ℕ := 7

/-- Represents the number of roses in a bouquet Bill sells -/
def roses_per_bouquet_sell : ℕ := 5

/-- Represents the price of a bouquet (both buying and selling) -/
def price_per_bouquet : ℕ := 20

/-- Represents the target profit -/
def target_profit : ℕ := 1000

/-- Calculates the number of bouquets Bill needs to buy to earn the target profit -/
def bouquets_to_buy : ℕ :=
  let bouquets_per_operation := roses_per_bouquet_sell
  let profit_per_operation := price_per_bouquet * bouquets_per_operation - price_per_bouquet * roses_per_bouquet_buy / roses_per_bouquet_sell
  let operations_needed := target_profit / profit_per_operation
  operations_needed * roses_per_bouquet_buy / roses_per_bouquet_sell

theorem bill_bouquets_theorem : bouquets_to_buy = 125 := by
  sorry

end NUMINAMATH_CALUDE_bill_bouquets_theorem_l3469_346927


namespace NUMINAMATH_CALUDE_wen_family_science_fair_cost_l3469_346992

theorem wen_family_science_fair_cost : ∀ (x : ℝ),
  x > 0 →
  0.7 * x = 7 →
  let student_ticket := 0.6 * x
  let regular_ticket := x
  let senior_ticket := 0.7 * x
  3 * student_ticket + regular_ticket + senior_ticket = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_wen_family_science_fair_cost_l3469_346992


namespace NUMINAMATH_CALUDE_black_squares_in_37th_row_l3469_346910

/-- Represents the number of squares in the nth row of a stair-step figure -/
def num_squares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of black squares in the nth row of a stair-step figure -/
def num_black_squares (n : ℕ) : ℕ := (num_squares n - 1) / 2

theorem black_squares_in_37th_row :
  num_black_squares 37 = 36 := by sorry

end NUMINAMATH_CALUDE_black_squares_in_37th_row_l3469_346910


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3469_346964

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 1) :
  {x : ℝ | x^2 - (a + 1)*x + a < 0} = {x : ℝ | a < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3469_346964


namespace NUMINAMATH_CALUDE_angstadt_seniors_l3469_346946

/-- Mr. Angstadt's class enrollment problem -/
theorem angstadt_seniors (total_students : ℕ) 
  (stats_percent geometry_percent : ℚ)
  (stats_calc_overlap geometry_calc_overlap : ℚ)
  (stats_senior_percent geometry_senior_percent calc_senior_percent : ℚ)
  (h1 : total_students = 240)
  (h2 : stats_percent = 45/100)
  (h3 : geometry_percent = 35/100)
  (h4 : stats_calc_overlap = 10/100)
  (h5 : geometry_calc_overlap = 5/100)
  (h6 : stats_senior_percent = 90/100)
  (h7 : geometry_senior_percent = 60/100)
  (h8 : calc_senior_percent = 80/100) :
  ∃ (senior_count : ℕ), senior_count = 161 := by
sorry


end NUMINAMATH_CALUDE_angstadt_seniors_l3469_346946


namespace NUMINAMATH_CALUDE_system_solution_l3469_346918

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, x - 3*y = k + 2 ∧ x - y = 4 ∧ 3*x + y = -8) → k = 12 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3469_346918


namespace NUMINAMATH_CALUDE_jan_cable_purchase_l3469_346926

theorem jan_cable_purchase (section_length : ℕ) (sections_on_hand : ℕ) : 
  section_length = 25 →
  sections_on_hand = 15 →
  (4 : ℚ) * sections_on_hand = 3 * (2 * sections_on_hand) →
  (4 : ℚ) * sections_on_hand * section_length = 1000 := by
  sorry

end NUMINAMATH_CALUDE_jan_cable_purchase_l3469_346926


namespace NUMINAMATH_CALUDE_complex_z_value_l3469_346932

theorem complex_z_value : ∃ z : ℂ, z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) ∧ 
  z = Complex.mk (1/2) (-Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_z_value_l3469_346932


namespace NUMINAMATH_CALUDE_inequality_proof_l3469_346962

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3469_346962


namespace NUMINAMATH_CALUDE_inverse_g_at_negative_seven_sixty_four_l3469_346928

open Real

noncomputable def g (x : ℝ) : ℝ := (x^5 - 1) / 4

theorem inverse_g_at_negative_seven_sixty_four :
  g⁻¹ (-7/64) = (9/16)^(1/5) :=
by sorry

end NUMINAMATH_CALUDE_inverse_g_at_negative_seven_sixty_four_l3469_346928


namespace NUMINAMATH_CALUDE_martha_exceptional_savings_l3469_346980

/-- Represents Martha's savings over a week -/
def MarthaSavings (daily_allowance : ℚ) (regular_fraction : ℚ) (exceptional_fraction : ℚ) : ℚ :=
  6 * (daily_allowance * regular_fraction) + (daily_allowance * exceptional_fraction)

/-- Theorem stating the fraction Martha saved on the exceptional day -/
theorem martha_exceptional_savings :
  ∀ (daily_allowance : ℚ),
  daily_allowance = 12 →
  MarthaSavings daily_allowance (1/2) (1/4) = 39 :=
by
  sorry


end NUMINAMATH_CALUDE_martha_exceptional_savings_l3469_346980


namespace NUMINAMATH_CALUDE_prop2_prop4_l3469_346974

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Theorem for proposition 2
theorem prop2 (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → plane_perpendicular α β := by sorry

-- Theorem for proposition 4
theorem prop4 (m : Line) (α β : Plane) :
  perpendicular m α → plane_parallel α β → perpendicular m β := by sorry

end NUMINAMATH_CALUDE_prop2_prop4_l3469_346974


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3469_346902

/-- A quadratic form ax^2 + bxy + cy^2 is a perfect square if and only if its discriminant b^2 - 4ac is zero. -/
def is_perfect_square (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- If 4x^2 + mxy + 25y^2 is a perfect square, then m = ±20. -/
theorem perfect_square_condition (m : ℝ) :
  is_perfect_square 4 m 25 → m = 20 ∨ m = -20 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3469_346902


namespace NUMINAMATH_CALUDE_probability_of_vowel_in_four_consecutive_letters_l3469_346977

/-- Represents the English alphabet --/
def Alphabet : Finset Char := sorry

/-- Represents the vowels in the English alphabet --/
def Vowels : Finset Char := sorry

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of vowels --/
def vowel_count : ℕ := 5

/-- The number of possible sets of 4 consecutive letters --/
def consecutive_sets : ℕ := 23

/-- The number of sets of 4 consecutive letters without a vowel --/
def sets_without_vowel : ℕ := 5

/-- Theorem: The probability of selecting at least one vowel when choosing 4 consecutive letters at random from the English alphabet is 18/23 --/
theorem probability_of_vowel_in_four_consecutive_letters :
  (consecutive_sets - sets_without_vowel : ℚ) / consecutive_sets = 18 / 23 :=
sorry

end NUMINAMATH_CALUDE_probability_of_vowel_in_four_consecutive_letters_l3469_346977


namespace NUMINAMATH_CALUDE_ball_distribution_count_l3469_346968

theorem ball_distribution_count :
  let n_balls : ℕ := 4
  let n_boxes : ℕ := 3
  let ways_per_ball : ℕ := n_boxes
  n_boxes ^ n_balls = 81 :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_count_l3469_346968


namespace NUMINAMATH_CALUDE_max_different_ages_l3469_346947

/-- The maximum number of different integer ages within one standard deviation of the average -/
theorem max_different_ages (average_age : ℝ) (std_dev : ℝ) : average_age = 31 → std_dev = 9 →
  (Set.Icc (average_age - std_dev) (average_age + std_dev) ∩ Set.range (Int.cast : ℤ → ℝ)).ncard = 19 := by
  sorry

end NUMINAMATH_CALUDE_max_different_ages_l3469_346947


namespace NUMINAMATH_CALUDE_vector_sum_l3469_346953

def vector1 : ℝ × ℝ × ℝ := (4, -9, 2)
def vector2 : ℝ × ℝ × ℝ := (-1, 16, 5)

theorem vector_sum : 
  (vector1.1 + vector2.1, vector1.2.1 + vector2.2.1, vector1.2.2 + vector2.2.2) = (3, 7, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_l3469_346953


namespace NUMINAMATH_CALUDE_jeff_scores_mean_l3469_346967

def jeff_scores : List ℝ := [90, 93, 85, 97, 92, 88]

theorem jeff_scores_mean : 
  (jeff_scores.sum / jeff_scores.length : ℝ) = 90.8333 := by
  sorry

end NUMINAMATH_CALUDE_jeff_scores_mean_l3469_346967


namespace NUMINAMATH_CALUDE_multiply_72514_9999_l3469_346970

theorem multiply_72514_9999 : 72514 * 9999 = 725067486 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72514_9999_l3469_346970


namespace NUMINAMATH_CALUDE_apple_difference_l3469_346998

/-- An apple eating contest with six students -/
structure AppleContest where
  students : Nat
  max_apples : Nat
  min_apples : Nat

/-- The properties of the given apple eating contest -/
def given_contest : AppleContest :=
  { students := 6
  , max_apples := 6
  , min_apples := 1 }

/-- Theorem stating the difference between max and min apples eaten -/
theorem apple_difference (contest : AppleContest) (h1 : contest = given_contest) :
  contest.max_apples - contest.min_apples = 5 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l3469_346998


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3469_346986

theorem angle_measure_proof (x : ℝ) : 
  (x + (3 * x - 2) = 90) → x = 23 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3469_346986


namespace NUMINAMATH_CALUDE_watermelon_slices_l3469_346917

/-- The number of slices in a watermelon, given the number of seeds per slice and the total number of seeds. -/
def number_of_slices (black_seeds_per_slice : ℕ) (white_seeds_per_slice : ℕ) (total_seeds : ℕ) : ℕ :=
  total_seeds / (black_seeds_per_slice + white_seeds_per_slice)

/-- Theorem stating that the number of slices is 40 given the conditions of the problem. -/
theorem watermelon_slices :
  number_of_slices 20 20 1600 = 40 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_slices_l3469_346917


namespace NUMINAMATH_CALUDE_inequality_infimum_l3469_346907

theorem inequality_infimum (m : ℝ) : 
  (∃ (a b : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 0 1 → x^3 - m ≤ a*x + b ∧ a*x + b ≤ x^3 + m) →
  m ≥ Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_CALUDE_inequality_infimum_l3469_346907


namespace NUMINAMATH_CALUDE_final_score_is_94_l3469_346993

/-- Represents the scoring system for a choir competition -/
structure ScoringSystem where
  songContentWeight : Real
  singingSkillsWeight : Real
  spiritWeight : Real
  weightSum : songContentWeight + singingSkillsWeight + spiritWeight = 1

/-- Represents the scores of a participating team -/
structure TeamScores where
  songContent : Real
  singingSkills : Real
  spirit : Real

/-- Calculates the final score given a scoring system and team scores -/
def calculateFinalScore (system : ScoringSystem) (scores : TeamScores) : Real :=
  system.songContentWeight * scores.songContent +
  system.singingSkillsWeight * scores.singingSkills +
  system.spiritWeight * scores.spirit

theorem final_score_is_94 (system : ScoringSystem) (scores : TeamScores)
    (h1 : system.songContentWeight = 0.3)
    (h2 : system.singingSkillsWeight = 0.4)
    (h3 : system.spiritWeight = 0.3)
    (h4 : scores.songContent = 90)
    (h5 : scores.singingSkills = 94)
    (h6 : scores.spirit = 98) :
    calculateFinalScore system scores = 94 := by
  sorry


end NUMINAMATH_CALUDE_final_score_is_94_l3469_346993


namespace NUMINAMATH_CALUDE_perfect_square_analysis_l3469_346914

theorem perfect_square_analysis :
  (∃ (x : ℕ), 8^2050 = x^2) ∧
  (∃ (x : ℕ), 9^2048 = x^2) ∧
  (∀ (x : ℕ), 10^2051 ≠ x^2) ∧
  (∃ (x : ℕ), 11^2052 = x^2) ∧
  (∃ (x : ℕ), 12^2050 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_analysis_l3469_346914


namespace NUMINAMATH_CALUDE_adams_age_l3469_346985

theorem adams_age (adam_age eve_age : ℕ) : 
  adam_age = eve_age - 5 →
  eve_age + 1 = 3 * (adam_age - 4) →
  adam_age = 9 := by
sorry

end NUMINAMATH_CALUDE_adams_age_l3469_346985


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l3469_346940

/-- The volume of a cylinder formed by rotating a rectangle about its shorter side. -/
theorem cylinder_volume_from_rectangle (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) :
  let r := w / (2 * Real.pi)
  (Real.pi * r^2 * h) = 1000 / Real.pi → h = 10 ∧ w = 20 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l3469_346940


namespace NUMINAMATH_CALUDE_square_neq_four_implies_neq_two_l3469_346961

theorem square_neq_four_implies_neq_two (a : ℝ) :
  (a^2 ≠ 4 → a ≠ 2) ∧ ¬(∀ a : ℝ, a ≠ 2 → a^2 ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_square_neq_four_implies_neq_two_l3469_346961


namespace NUMINAMATH_CALUDE_social_media_to_phone_ratio_l3469_346951

/-- Represents the daily phone usage in hours -/
def daily_phone_usage : ℝ := 8

/-- Represents the weekly social media usage in hours -/
def weekly_social_media : ℝ := 28

/-- Represents the number of days in a week -/
def days_in_week : ℝ := 7

/-- Theorem stating that the ratio of daily social media usage to total daily phone usage is 1:2 -/
theorem social_media_to_phone_ratio :
  (weekly_social_media / days_in_week) / daily_phone_usage = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_social_media_to_phone_ratio_l3469_346951


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3469_346921

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 + x - 4*m = 0) ↔ m ≥ -1/16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3469_346921


namespace NUMINAMATH_CALUDE_arithmetic_seq_property_l3469_346916

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a where a₂ + a₆ = 2, prove that a₄ = 1 -/
theorem arithmetic_seq_property (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_seq a) (h_sum : a 2 + a 6 = 2) : 
  a 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_property_l3469_346916


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3469_346900

def polynomial (x : ℝ) : ℝ := 5*x^8 + 2*x^7 - 3*x^4 + 4*x^3 - 5*x^2 + 6*x - 20

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, polynomial x = q x * divisor x + 1404 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3469_346900


namespace NUMINAMATH_CALUDE_largest_four_digit_mod_5_3_l3469_346972

theorem largest_four_digit_mod_5_3 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 5 = 3 → n ≤ 9998 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_mod_5_3_l3469_346972


namespace NUMINAMATH_CALUDE_find_heaviest_and_lightest_in_13_weighings_l3469_346920

/-- Represents a coin with a unique weight -/
structure Coin where
  weight : ℕ

/-- Represents the result of weighing two coins -/
inductive WeighResult
  | Left  : WeighResult  -- left coin is heavier
  | Right : WeighResult  -- right coin is heavier
  | Equal : WeighResult  -- coins have equal weight

/-- A function that simulates weighing two coins -/
def weigh (a b : Coin) : WeighResult :=
  if a.weight > b.weight then WeighResult.Left
  else if a.weight < b.weight then WeighResult.Right
  else WeighResult.Equal

/-- Theorem stating that it's possible to find the heaviest and lightest coins in 13 weighings -/
theorem find_heaviest_and_lightest_in_13_weighings
  (coins : List Coin)
  (h_distinct : ∀ i j, i ≠ j → (coins.get i).weight ≠ (coins.get j).weight)
  (h_count : coins.length = 10) :
  ∃ (heaviest lightest : Coin) (steps : List (Coin × Coin)),
    heaviest ∈ coins ∧
    lightest ∈ coins ∧
    (∀ c ∈ coins, c.weight ≤ heaviest.weight) ∧
    (∀ c ∈ coins, c.weight ≥ lightest.weight) ∧
    steps.length ≤ 13 ∧
    (∀ step ∈ steps, ∃ a b, step = (a, b) ∧ a ∈ coins ∧ b ∈ coins) :=
by sorry

end NUMINAMATH_CALUDE_find_heaviest_and_lightest_in_13_weighings_l3469_346920


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3469_346922

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3469_346922


namespace NUMINAMATH_CALUDE_forgotten_angle_measure_l3469_346935

/-- The sum of interior angles of a polygon with n sides --/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The sum of all but one interior angle of the polygon --/
def partial_sum : ℝ := 2017

/-- The measure of the forgotten angle --/
def forgotten_angle : ℝ := 143

theorem forgotten_angle_measure :
  ∃ (n : ℕ), n > 3 ∧ sum_interior_angles n = partial_sum + forgotten_angle :=
sorry

end NUMINAMATH_CALUDE_forgotten_angle_measure_l3469_346935


namespace NUMINAMATH_CALUDE_race_finishing_orders_l3469_346976

/-- The number of possible finishing orders for a race with n participants and no ties -/
def racePermutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of racers -/
def numRacers : ℕ := 4

theorem race_finishing_orders :
  racePermutations numRacers = 24 :=
by sorry

end NUMINAMATH_CALUDE_race_finishing_orders_l3469_346976


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3469_346919

theorem triangle_angle_proof (a b : ℝ) (B : ℝ) (A : ℝ) : 
  a = 4 → b = 4 * Real.sqrt 2 → B = π/4 → A = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3469_346919


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3469_346934

theorem quadratic_equal_roots (a b c : ℝ) 
  (h1 : b ≠ c) 
  (h2 : ∃ x : ℝ, (b - c) * x^2 + (a - b) * x + (c - a) = 0 ∧ 
       ((b - c) * (2 * x) + (a - b) = 0)) : 
  c = (a + b) / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3469_346934


namespace NUMINAMATH_CALUDE_inequality_solution_l3469_346931

theorem inequality_solution : 
  {x : ℕ | 3 * x - 2 < 7} = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3469_346931


namespace NUMINAMATH_CALUDE_probability_53_sundays_in_leap_year_l3469_346954

/-- A leap year has 366 days -/
def leapYearDays : ℕ := 366

/-- A week has 7 days -/
def daysInWeek : ℕ := 7

/-- A leap year has 52 weeks and 2 extra days -/
def leapYearWeeks : ℕ := 52
def leapYearExtraDays : ℕ := 2

/-- The probability of a randomly chosen leap year having 53 Sundays -/
def probabilityOf53Sundays : ℚ := 1 / 7

theorem probability_53_sundays_in_leap_year :
  probabilityOf53Sundays = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_53_sundays_in_leap_year_l3469_346954


namespace NUMINAMATH_CALUDE_bee_path_distance_l3469_346909

open Complex

-- Define ω as e^(πi/4)
noncomputable def ω : ℂ := exp (I * Real.pi / 4)

-- Define the path of the bee
noncomputable def z : ℂ := 1 + 2 * ω + 3 * ω^2 + 4 * ω^3 + 5 * ω^4 + 6 * ω^5 + 7 * ω^6

-- Theorem stating the distance from P₀ to P₇
theorem bee_path_distance : abs z = Real.sqrt (25 - 7 * Real.sqrt 2 / 2) := by sorry

end NUMINAMATH_CALUDE_bee_path_distance_l3469_346909


namespace NUMINAMATH_CALUDE_whitney_bookmarks_l3469_346979

/-- Proves that Whitney bought 2 bookmarks given the conditions of the problem --/
theorem whitney_bookmarks :
  ∀ (initial_amount : ℕ) 
    (poster_cost notebook_cost bookmark_cost : ℕ)
    (posters_bought notebooks_bought : ℕ)
    (amount_left : ℕ),
  initial_amount = 2 * 20 →
  poster_cost = 5 →
  notebook_cost = 4 →
  bookmark_cost = 2 →
  posters_bought = 2 →
  notebooks_bought = 3 →
  amount_left = 14 →
  ∃ (bookmarks_bought : ℕ),
    initial_amount = 
      poster_cost * posters_bought + 
      notebook_cost * notebooks_bought + 
      bookmark_cost * bookmarks_bought + 
      amount_left ∧
    bookmarks_bought = 2 :=
by sorry

end NUMINAMATH_CALUDE_whitney_bookmarks_l3469_346979


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_equation_l3469_346997

theorem geometric_progression_ratio_equation 
  (x y z r : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
  (h_geometric : ∃ a : ℝ, a ≠ 0 ∧ 
    x * (y - z) = a ∧ 
    y * (z - x) = a * r ∧ 
    z * (y - x) = a * r^2) : 
  r^2 - r + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_equation_l3469_346997


namespace NUMINAMATH_CALUDE_second_number_is_thirty_l3469_346944

theorem second_number_is_thirty
  (a b c : ℝ)
  (sum_eq_98 : a + b + c = 98)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 5 / 8)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c) :
  b = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_thirty_l3469_346944


namespace NUMINAMATH_CALUDE_div_decimal_equals_sixty_l3469_346957

theorem div_decimal_equals_sixty : (0.24 : ℚ) / (0.004 : ℚ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_div_decimal_equals_sixty_l3469_346957


namespace NUMINAMATH_CALUDE_remainder_3_153_mod_8_l3469_346905

theorem remainder_3_153_mod_8 : 3^153 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_153_mod_8_l3469_346905


namespace NUMINAMATH_CALUDE_shooting_probability_theorem_l3469_346971

def shooting_probability (accuracy_A accuracy_B : ℝ) : ℝ × ℝ :=
  let prob_both_two := accuracy_A * accuracy_A * accuracy_B * accuracy_B
  let prob_at_least_three := prob_both_two + 
    accuracy_A * accuracy_A * accuracy_B * (1 - accuracy_B) +
    accuracy_A * (1 - accuracy_A) * accuracy_B * accuracy_B
  (prob_both_two, prob_at_least_three)

theorem shooting_probability_theorem :
  shooting_probability 0.4 0.6 = (0.0576, 0.1824) := by
  sorry

end NUMINAMATH_CALUDE_shooting_probability_theorem_l3469_346971


namespace NUMINAMATH_CALUDE_absolute_value_negative_2023_l3469_346908

theorem absolute_value_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_negative_2023_l3469_346908


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3469_346975

def U : Set Nat := {1,2,3,4,5}
def A : Set Nat := {2,3,4}
def B : Set Nat := {1,4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {1,4,5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3469_346975


namespace NUMINAMATH_CALUDE_lunch_average_price_proof_l3469_346963

theorem lunch_average_price_proof (total_price : ℝ) (num_people : ℕ) (gratuity_rate : ℝ) 
  (h1 : total_price = 207)
  (h2 : num_people = 15)
  (h3 : gratuity_rate = 0.15) :
  (total_price / (1 + gratuity_rate)) / num_people = 12 := by
  sorry

end NUMINAMATH_CALUDE_lunch_average_price_proof_l3469_346963


namespace NUMINAMATH_CALUDE_cos_225_degrees_l3469_346991

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l3469_346991


namespace NUMINAMATH_CALUDE_problem_statement_l3469_346965

theorem problem_statement :
  (∃ m : ℝ, |m| + m = 0 ∧ m ≥ 0) ∧
  (∃ a b : ℝ, |a - b| = b - a ∧ b ≤ a) ∧
  (∀ a b : ℝ, a^5 + b^5 = 0 → a + b = 0) ∧
  (∃ a b : ℝ, a + b = 0 ∧ a / b ≠ -1) ∧
  (∀ a b c : ℚ, |a| / a + |b| / b + |c| / c = 1 → |a * b * c| / (a * b * c) = -1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3469_346965


namespace NUMINAMATH_CALUDE_ladder_slide_l3469_346906

-- Define the ladder and wall setup
def ladder_length : ℝ := 25
def initial_distance : ℝ := 7
def top_slide : ℝ := 4

-- Theorem statement
theorem ladder_slide :
  let initial_height : ℝ := Real.sqrt (ladder_length^2 - initial_distance^2)
  let new_height : ℝ := initial_height - top_slide
  let new_distance : ℝ := Real.sqrt (ladder_length^2 - new_height^2)
  new_distance - initial_distance = 8 := by
  sorry

end NUMINAMATH_CALUDE_ladder_slide_l3469_346906


namespace NUMINAMATH_CALUDE_students_speaking_both_languages_l3469_346923

theorem students_speaking_both_languages (total : ℕ) (english : ℕ) (japanese : ℕ) (neither : ℕ) :
  total = 50 →
  english = 36 →
  japanese = 20 →
  neither = 8 →
  ∃ (both : ℕ), both = 14 ∧
    total = english + japanese - both + neither :=
by sorry

end NUMINAMATH_CALUDE_students_speaking_both_languages_l3469_346923


namespace NUMINAMATH_CALUDE_solve_for_t_l3469_346952

theorem solve_for_t (p : ℝ) (t : ℝ) 
  (h1 : 5 = p * 3^t) 
  (h2 : 45 = p * 9^t) : 
  t = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l3469_346952


namespace NUMINAMATH_CALUDE_binomial_plus_ten_l3469_346955

theorem binomial_plus_ten : (Nat.choose 9 5) + 10 = 136 := by sorry

end NUMINAMATH_CALUDE_binomial_plus_ten_l3469_346955


namespace NUMINAMATH_CALUDE_proportion_solution_l3469_346966

theorem proportion_solution (x : ℚ) : (2 : ℚ) / 5 = (4 : ℚ) / 3 / x → x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3469_346966


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l3469_346924

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l3469_346924


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3469_346973

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |2*y - 44| + |y - 24| = |3*y - 66| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3469_346973


namespace NUMINAMATH_CALUDE_misread_weight_correction_l3469_346984

theorem misread_weight_correction (n : ℕ) (incorrect_avg correct_avg misread_weight : ℝ) :
  n = 20 ∧ 
  incorrect_avg = 58.4 ∧ 
  correct_avg = 58.7 ∧ 
  misread_weight = 56 →
  ∃ correct_weight : ℝ,
    correct_weight = 62 ∧
    n * correct_avg = (n - 1) * incorrect_avg + correct_weight ∧
    n * incorrect_avg = (n - 1) * incorrect_avg + misread_weight :=
by sorry

end NUMINAMATH_CALUDE_misread_weight_correction_l3469_346984


namespace NUMINAMATH_CALUDE_tissue_cost_theorem_l3469_346994

/-- Calculates the total cost of tissues given the number of boxes, packs per box,
    tissues per pack, and cost per tissue. -/
def totalCost (boxes : ℕ) (packsPerBox : ℕ) (tissuesPerPack : ℕ) (costPerTissue : ℚ) : ℚ :=
  (boxes * packsPerBox * tissuesPerPack : ℚ) * costPerTissue

/-- Proves that the total cost of 10 boxes of tissues, with 20 packs per box,
    100 tissues per pack, and 5 cents per tissue, is $1000. -/
theorem tissue_cost_theorem :
  totalCost 10 20 100 (5 / 100) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_tissue_cost_theorem_l3469_346994


namespace NUMINAMATH_CALUDE_truck_sand_problem_l3469_346941

/-- The amount of sand remaining on a truck after making several stops --/
def sandRemaining (initialSand : ℕ) (sandLostAtStops : List ℕ) : ℕ :=
  initialSand - sandLostAtStops.sum

/-- Theorem: A truck with 1050 pounds of sand that loses 32, 67, 45, and 54 pounds at four stops will have 852 pounds remaining --/
theorem truck_sand_problem :
  let initialSand : ℕ := 1050
  let sandLostAtStops : List ℕ := [32, 67, 45, 54]
  sandRemaining initialSand sandLostAtStops = 852 := by
  sorry

#eval sandRemaining 1050 [32, 67, 45, 54]

end NUMINAMATH_CALUDE_truck_sand_problem_l3469_346941


namespace NUMINAMATH_CALUDE_gcd_18_30_42_l3469_346938

theorem gcd_18_30_42 : Nat.gcd 18 (Nat.gcd 30 42) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_42_l3469_346938


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1320_l3469_346933

theorem sum_of_largest_and_smallest_prime_factors_of_1320 :
  ∃ (smallest largest : Nat),
    smallest.Prime ∧
    largest.Prime ∧
    smallest ∣ 1320 ∧
    largest ∣ 1320 ∧
    (∀ p : Nat, p.Prime → p ∣ 1320 → p ≥ smallest) ∧
    (∀ p : Nat, p.Prime → p ∣ 1320 → p ≤ largest) ∧
    smallest + largest = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1320_l3469_346933


namespace NUMINAMATH_CALUDE_andrew_grapes_purchase_l3469_346996

theorem andrew_grapes_purchase (x : ℝ) : 
  x ≥ 0 →
  54 * x + 62 * 10 = 1376 →
  x = 14 := by
sorry

end NUMINAMATH_CALUDE_andrew_grapes_purchase_l3469_346996


namespace NUMINAMATH_CALUDE_smallest_result_l3469_346942

def S : Finset ℕ := {2, 4, 6, 8, 10, 12}

def process (a b c : ℕ) : ℕ := (a + b) * c

def valid_choice (a b c : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∃ (a b c : ℕ), valid_choice a b c ∧
  process a b c = 20 ∧
  ∀ (x y z : ℕ), valid_choice x y z → process x y z ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_result_l3469_346942


namespace NUMINAMATH_CALUDE_intersection_minimizes_sum_of_distances_l3469_346904

/-- Given a triangle ABC, construct equilateral triangles ABC₁, ACB₁, and BCA₁ externally --/
def constructExternalTriangles (A B C : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- Compute the intersection point of lines AA₁, BB₁, and CC₁ --/
def intersectionPoint (A B C : ℝ × ℝ) (A₁ B₁ C₁ : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Compute the sum of distances from a point to the vertices of a triangle --/
def sumOfDistances (P A B C : ℝ × ℝ) : ℝ := sorry

/-- The main theorem: The intersection point minimizes the sum of distances --/
theorem intersection_minimizes_sum_of_distances (A B C : ℝ × ℝ) :
  let (A₁, B₁, C₁) := constructExternalTriangles A B C
  let O := intersectionPoint A B C A₁ B₁ C₁
  ∀ P : ℝ × ℝ, sumOfDistances O A B C ≤ sumOfDistances P A B C :=
sorry

end NUMINAMATH_CALUDE_intersection_minimizes_sum_of_distances_l3469_346904


namespace NUMINAMATH_CALUDE_sock_order_ratio_l3469_346983

theorem sock_order_ratio (red_socks green_socks : ℕ) (price_green : ℝ) :
  red_socks = 5 →
  (red_socks * (3 * price_green) + green_socks * price_green) * 1.8 =
    green_socks * (3 * price_green) + red_socks * price_green →
  green_socks = 18 :=
by sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l3469_346983


namespace NUMINAMATH_CALUDE_expansion_coefficient_implies_a_equals_one_l3469_346958

-- Define the binomial expansion coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := sorry

-- Define the function for the coefficient of x^(-3) in the expansion
def coefficient_x_neg_3 (a : ℝ) : ℝ :=
  (binomial_coefficient 7 2) * (2^2) * (a^5)

-- State the theorem
theorem expansion_coefficient_implies_a_equals_one :
  coefficient_x_neg_3 1 = 84 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_implies_a_equals_one_l3469_346958


namespace NUMINAMATH_CALUDE_range_of_m_l3469_346925

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x m : ℝ) : Prop := x^2 + x + m - m^2 > 0

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (m > 1) →
  (∀ x : ℝ, (¬(p x) → ¬(q x m)) ∧ ∃ y : ℝ, ¬(p y) ∧ (q y m)) →
  m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3469_346925


namespace NUMINAMATH_CALUDE_probability_at_least_one_cherry_plum_probability_at_least_one_cherry_plum_proof_l3469_346936

/-- The probability of selecting at least one cherry plum cutting -/
theorem probability_at_least_one_cherry_plum 
  (total_cuttings : ℕ) 
  (cherry_plum_cuttings : ℕ) 
  (plum_cuttings : ℕ) 
  (selected_cuttings : ℕ)
  (h1 : total_cuttings = 20)
  (h2 : cherry_plum_cuttings = 8)
  (h3 : plum_cuttings = 12)
  (h4 : selected_cuttings = 3)
  (h5 : total_cuttings = cherry_plum_cuttings + plum_cuttings) : 
  ℚ :=
46/57

theorem probability_at_least_one_cherry_plum_proof 
  (total_cuttings : ℕ) 
  (cherry_plum_cuttings : ℕ) 
  (plum_cuttings : ℕ) 
  (selected_cuttings : ℕ)
  (h1 : total_cuttings = 20)
  (h2 : cherry_plum_cuttings = 8)
  (h3 : plum_cuttings = 12)
  (h4 : selected_cuttings = 3)
  (h5 : total_cuttings = cherry_plum_cuttings + plum_cuttings) : 
  probability_at_least_one_cherry_plum total_cuttings cherry_plum_cuttings plum_cuttings selected_cuttings h1 h2 h3 h4 h5 = 46/57 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_cherry_plum_probability_at_least_one_cherry_plum_proof_l3469_346936


namespace NUMINAMATH_CALUDE_cube_frame_construction_l3469_346989

/-- Represents a cube -/
structure Cube where
  edge_length : ℝ

/-- Represents a wire -/
structure Wire where
  length : ℝ

/-- Represents the number of cuts needed to construct a cube frame -/
def num_cuts_needed (c : Cube) (w : Wire) : ℕ := sorry

theorem cube_frame_construction (c : Cube) (w : Wire) 
  (h1 : c.edge_length = 10)
  (h2 : w.length = 120) :
  ¬ (num_cuts_needed c w = 0) ∧ (num_cuts_needed c w = 3) := by sorry

end NUMINAMATH_CALUDE_cube_frame_construction_l3469_346989


namespace NUMINAMATH_CALUDE_negation_of_universal_real_proposition_l3469_346943

theorem negation_of_universal_real_proposition :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_real_proposition_l3469_346943


namespace NUMINAMATH_CALUDE_abc_product_equals_k_l3469_346978

theorem abc_product_equals_k (a b c k : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → k ≠ 0 →
  a ≠ b → b ≠ c → a ≠ c →
  (a + k / b = b + k / c) → (b + k / c = c + k / a) →
  |a * b * c| = |k| := by
sorry

end NUMINAMATH_CALUDE_abc_product_equals_k_l3469_346978


namespace NUMINAMATH_CALUDE_quadratic_two_roots_condition_l3469_346950

theorem quadratic_two_roots_condition (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + c = x₁ + 2 ∧ x₂^2 - 3*x₂ + c = x₂ + 2) ↔ c < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_condition_l3469_346950


namespace NUMINAMATH_CALUDE_not_necessarily_square_l3469_346988

/-- A quadrilateral with four sides and two diagonals -/
structure Quadrilateral :=
  (side1 side2 side3 side4 diagonal1 diagonal2 : ℝ)

/-- Predicate to check if a quadrilateral has 4 equal sides and 2 equal diagonals -/
def has_equal_sides_and_diagonals (q : Quadrilateral) : Prop :=
  q.side1 = q.side2 ∧ q.side2 = q.side3 ∧ q.side3 = q.side4 ∧
  q.diagonal1 = q.diagonal2 ∧
  q.side1 ≠ q.diagonal1

/-- Predicate to check if a quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop :=
  q.side1 = q.side2 ∧ q.side2 = q.side3 ∧ q.side3 = q.side4 ∧
  q.diagonal1 = q.diagonal2 ∧
  q.diagonal1 = q.side1 * Real.sqrt 2

/-- Theorem stating that a quadrilateral with 4 equal sides and 2 equal diagonals
    is not necessarily a square -/
theorem not_necessarily_square :
  ∃ q : Quadrilateral, has_equal_sides_and_diagonals q ∧ ¬is_square q :=
sorry


end NUMINAMATH_CALUDE_not_necessarily_square_l3469_346988


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3469_346913

theorem min_value_quadratic :
  ∃ (m : ℝ), (∀ x : ℝ, 4 * x^2 + 8 * x + 12 ≥ m) ∧
  (∃ x : ℝ, 4 * x^2 + 8 * x + 12 = m) ∧
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3469_346913


namespace NUMINAMATH_CALUDE_stock_value_change_l3469_346995

theorem stock_value_change (initial_value : ℝ) (h : initial_value > 0) :
  let day1_value := initial_value * 0.85
  let day2_value := day1_value * 1.4
  (day2_value - initial_value) / initial_value = 0.19 := by
sorry

end NUMINAMATH_CALUDE_stock_value_change_l3469_346995


namespace NUMINAMATH_CALUDE_valid_draws_count_l3469_346960

/-- Represents the number of cards for each color --/
def cards_per_color : ℕ := 5

/-- Represents the number of colors --/
def num_colors : ℕ := 3

/-- Represents the number of cards drawn --/
def cards_drawn : ℕ := 4

/-- Represents the total number of cards --/
def total_cards : ℕ := cards_per_color * num_colors

/-- Represents the set of all possible draws --/
def all_draws : Set (Fin total_cards) := sorry

/-- Predicate to check if a draw contains all colors --/
def has_all_colors (draw : Fin total_cards) : Prop := sorry

/-- Predicate to check if a draw has all different letters --/
def has_different_letters (draw : Fin total_cards) : Prop := sorry

/-- The number of valid draws --/
def num_valid_draws : ℕ := sorry

theorem valid_draws_count : num_valid_draws = 360 := by sorry

end NUMINAMATH_CALUDE_valid_draws_count_l3469_346960


namespace NUMINAMATH_CALUDE_football_equipment_cost_l3469_346959

/-- The cost of equipment for a football team -/
theorem football_equipment_cost : 
  let num_players : ℕ := 16
  let jersey_cost : ℚ := 25
  let shorts_cost : ℚ := 15.20
  let socks_cost : ℚ := 6.80
  let total_cost : ℚ := num_players * (jersey_cost + shorts_cost + socks_cost)
  total_cost = 752 := by sorry

end NUMINAMATH_CALUDE_football_equipment_cost_l3469_346959


namespace NUMINAMATH_CALUDE_whitney_purchase_cost_is_445_62_l3469_346981

/-- Calculates the total cost of Whitney's purchase given the specified conditions --/
def whitneyPurchaseCost : ℝ :=
  let whaleBookCount : ℕ := 15
  let fishBookCount : ℕ := 12
  let sharkBookCount : ℕ := 5
  let magazineCount : ℕ := 8
  let whaleBookPrice : ℝ := 14
  let fishBookPrice : ℝ := 13
  let sharkBookPrice : ℝ := 10
  let magazinePrice : ℝ := 3
  let fishBookDiscount : ℝ := 0.1
  let salesTaxRate : ℝ := 0.05

  let whaleBooksCost := whaleBookCount * whaleBookPrice
  let fishBooksCost := fishBookCount * fishBookPrice * (1 - fishBookDiscount)
  let sharkBooksCost := sharkBookCount * sharkBookPrice
  let magazinesCost := magazineCount * magazinePrice

  let totalBeforeTax := whaleBooksCost + fishBooksCost + sharkBooksCost + magazinesCost
  let salesTax := totalBeforeTax * salesTaxRate
  let totalCost := totalBeforeTax + salesTax

  totalCost

/-- Theorem stating that Whitney's total purchase cost is $445.62 --/
theorem whitney_purchase_cost_is_445_62 : whitneyPurchaseCost = 445.62 := by
  sorry


end NUMINAMATH_CALUDE_whitney_purchase_cost_is_445_62_l3469_346981


namespace NUMINAMATH_CALUDE_family_gathering_handshakes_count_l3469_346956

/-- Represents the number of unique handshakes at a family gathering with twins and triplets -/
def familyGatheringHandshakes : ℕ :=
  let twin_sets := 12
  let triplet_sets := 5
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let first_twin_sets := 4
  let first_twins := first_twin_sets * 2

  -- Handshakes among twins
  let twin_handshakes := (twins * (twins - 2)) / 2

  -- Handshakes among triplets
  let triplet_handshakes := (triplets * (triplets - 3)) / 2

  -- Handshakes between first 4 sets of twins and triplets
  let first_twin_triplet_handshakes := first_twins * (triplets / 3)

  twin_handshakes + triplet_handshakes + first_twin_triplet_handshakes

/-- The total number of unique handshakes at the family gathering is 394 -/
theorem family_gathering_handshakes_count :
  familyGatheringHandshakes = 394 := by
  sorry

end NUMINAMATH_CALUDE_family_gathering_handshakes_count_l3469_346956


namespace NUMINAMATH_CALUDE_set_difference_equals_interval_l3469_346948

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | x ≠ 1 ∧ x / (x - 1) ≤ 0}

-- Define the open interval (-1, 0)
def openInterval : Set ℝ := {x | -1 < x ∧ x < 0}

-- Theorem statement
theorem set_difference_equals_interval : M \ N = openInterval := by
  sorry

end NUMINAMATH_CALUDE_set_difference_equals_interval_l3469_346948


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l3469_346937

/-- Represents the speed of a man rowing in different water conditions -/
structure RowingSpeed where
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the speed of the man rowing upstream given his rowing speeds in still water and downstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the man's speed in still water is 20 kmph and downstream is 33 kmph, his upstream speed is 7 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 20) 
  (h2 : s.downstream = 33) : 
  upstreamSpeed s = 7 := by
  sorry

#eval upstreamSpeed { stillWater := 20, downstream := 33 }

end NUMINAMATH_CALUDE_upstream_speed_calculation_l3469_346937


namespace NUMINAMATH_CALUDE_polynomial_division_l3469_346903

theorem polynomial_division (x : ℝ) :
  x^5 + 3*x^4 - 28*x^3 + 15*x^2 - 21*x + 8 =
  (x - 3) * (x^4 + 6*x^3 - 10*x^2 - 15*x - 66) + (-100) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l3469_346903
