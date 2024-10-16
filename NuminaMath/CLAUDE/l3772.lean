import Mathlib

namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3772_377282

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (a b : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a x * b x = k

theorem inverse_variation_problem (a b : ℝ → ℝ) 
  (h1 : VaryInversely a b) 
  (h2 : a 1 = 1500) 
  (h3 : b 1 = 0.25) 
  (h4 : a 2 = 3000) : 
  b 2 = 0.125 := by
sorry


end NUMINAMATH_CALUDE_inverse_variation_problem_l3772_377282


namespace NUMINAMATH_CALUDE_unique_solution_l3772_377271

theorem unique_solution : ∃! (m n : ℕ), 
  m > 0 ∧ n > 0 ∧ 14 * m * n = 55 - 7 * m - 2 * n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3772_377271


namespace NUMINAMATH_CALUDE_factors_of_34650_l3772_377222

theorem factors_of_34650 : Nat.card (Nat.divisors 34650) = 72 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_34650_l3772_377222


namespace NUMINAMATH_CALUDE_four_step_staircase_l3772_377226

/-- The number of ways to climb a staircase with n steps -/
def climbStairs (n : ℕ) : ℕ := sorry

/-- Theorem: There are exactly 8 ways to climb a staircase with 4 steps -/
theorem four_step_staircase : climbStairs 4 = 8 := by sorry

end NUMINAMATH_CALUDE_four_step_staircase_l3772_377226


namespace NUMINAMATH_CALUDE_father_son_ages_new_age_ratio_l3772_377218

/-- Given the ratio of a father's age to his son's age and their age product, 
    prove the father's age, son's age, and their combined income. -/
theorem father_son_ages (father_son_ratio : ℚ) (age_product : ℕ) (income_percentage : ℚ) :
  father_son_ratio = 7/3 →
  age_product = 756 →
  income_percentage = 2/5 →
  ∃ (father_age son_age : ℕ) (combined_income : ℚ),
    father_age = 42 ∧
    son_age = 18 ∧
    combined_income = 105 ∧
    (father_age : ℚ) / son_age = father_son_ratio ∧
    father_age * son_age = age_product ∧
    (father_age : ℚ) = income_percentage * combined_income :=
by sorry

/-- Given the father's and son's ages, prove their new age ratio after 6 years. -/
theorem new_age_ratio (father_age son_age : ℕ) (years : ℕ) :
  father_age = 42 →
  son_age = 18 →
  years = 6 →
  ∃ (new_ratio : ℚ),
    new_ratio = 2/1 ∧
    new_ratio = (father_age + years : ℚ) / (son_age + years) :=
by sorry

end NUMINAMATH_CALUDE_father_son_ages_new_age_ratio_l3772_377218


namespace NUMINAMATH_CALUDE_set_difference_proof_l3772_377263

def A : Set Int := {-1, 1, 3, 5, 7, 9}
def B : Set Int := {-1, 5, 7}

theorem set_difference_proof : A \ B = {1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_set_difference_proof_l3772_377263


namespace NUMINAMATH_CALUDE_initial_bees_calculation_l3772_377298

/-- Calculates the initial number of bees given the daily hatch rate, daily loss rate,
    number of days, and final number of bees. -/
def initialBees (hatchRate dailyLoss : ℕ) (days : ℕ) (finalBees : ℕ) : ℕ :=
  finalBees - (hatchRate - dailyLoss) * days

theorem initial_bees_calculation 
  (hatchRate dailyLoss days finalBees : ℕ) 
  (hatchRate_pos : hatchRate > dailyLoss) :
  initialBees hatchRate dailyLoss days finalBees = 
    finalBees - (hatchRate - dailyLoss) * days := by
  sorry

#eval initialBees 3000 900 7 27201

end NUMINAMATH_CALUDE_initial_bees_calculation_l3772_377298


namespace NUMINAMATH_CALUDE_suv_coupe_price_ratio_l3772_377230

theorem suv_coupe_price_ratio 
  (coupe_price : ℝ) 
  (commission_rate : ℝ) 
  (total_commission : ℝ) 
  (h1 : coupe_price = 30000)
  (h2 : commission_rate = 0.02)
  (h3 : total_commission = 1800)
  (h4 : ∃ x : ℝ, commission_rate * (coupe_price + x * coupe_price) = total_commission) :
  ∃ x : ℝ, x * coupe_price = 2 * coupe_price := by
sorry

end NUMINAMATH_CALUDE_suv_coupe_price_ratio_l3772_377230


namespace NUMINAMATH_CALUDE_penthouse_units_l3772_377256

theorem penthouse_units (total_floors : ℕ) (regular_units : ℕ) (penthouse_floors : ℕ) (total_units : ℕ)
  (h1 : total_floors = 23)
  (h2 : regular_units = 12)
  (h3 : penthouse_floors = 2)
  (h4 : total_units = 256) :
  (total_units - (total_floors - penthouse_floors) * regular_units) / penthouse_floors = 2 := by
  sorry

end NUMINAMATH_CALUDE_penthouse_units_l3772_377256


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l3772_377243

theorem sum_of_squares_and_products (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x^2 + y^2 + z^2 = 52 → x*y + y*z + z*x = 24 → x + y + z = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l3772_377243


namespace NUMINAMATH_CALUDE_contractor_absence_solution_l3772_377268

/-- Represents the problem of calculating a contractor's absence days --/
def ContractorAbsenceProblem (total_days : ℕ) (daily_pay : ℚ) (daily_fine : ℚ) (total_received : ℚ) : Prop :=
  ∃ (absent_days : ℕ),
    (absent_days ≤ total_days) ∧
    (daily_pay * (total_days - absent_days : ℚ) - daily_fine * (absent_days : ℚ) = total_received)

/-- Theorem stating the solution to the contractor absence problem --/
theorem contractor_absence_solution :
  ContractorAbsenceProblem 30 25 7.5 425 → ∃ (absent_days : ℕ), absent_days = 10 := by
  sorry

#check contractor_absence_solution

end NUMINAMATH_CALUDE_contractor_absence_solution_l3772_377268


namespace NUMINAMATH_CALUDE_tangent_plane_parallel_to_given_plane_l3772_377276

-- Define the elliptic paraboloid
def elliptic_paraboloid (x y : ℝ) : ℝ := 2 * x^2 + 4 * y^2

-- Define the plane
def plane (x y z : ℝ) : ℝ := 8 * x - 32 * y - 2 * z + 3

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ × ℝ := (1, -2, 18)

-- Define the tangent plane at the point of tangency
def tangent_plane (x y z : ℝ) : ℝ := 4 * x - 16 * y - z - 18

theorem tangent_plane_parallel_to_given_plane :
  let (x₀, y₀, z₀) := point_of_tangency
  ∃ (k : ℝ), k ≠ 0 ∧
    (∀ x y z, tangent_plane x y z = k * plane x y z) ∧
    z₀ = elliptic_paraboloid x₀ y₀ :=
by sorry

end NUMINAMATH_CALUDE_tangent_plane_parallel_to_given_plane_l3772_377276


namespace NUMINAMATH_CALUDE_lcm_fraction_evenness_l3772_377233

theorem lcm_fraction_evenness (x y z : ℕ+) :
  ∃ (k : ℕ), k > 0 ∧ k % 2 = 0 ∧
  (Nat.lcm x.val y.val + Nat.lcm y.val z.val) / Nat.lcm x.val z.val = k ∧
  ∀ (n : ℕ), n > 0 → n % 2 = 0 →
    ∃ (a b c : ℕ+), (Nat.lcm a.val b.val + Nat.lcm b.val c.val) / Nat.lcm a.val c.val = n :=
by sorry

end NUMINAMATH_CALUDE_lcm_fraction_evenness_l3772_377233


namespace NUMINAMATH_CALUDE_geometric_sequence_from_arithmetic_l3772_377287

/-- Given three positive numbers in arithmetic sequence summing to 12,
    if adding 1, 4, and 11 to these numbers respectively results in terms b₂, b₃, and b₄
    of a geometric sequence, then the general term of this sequence is bₙ = 2ⁿ -/
theorem geometric_sequence_from_arithmetic (a d : ℝ) (h1 : 0 < a - d ∧ 0 < a ∧ 0 < a + d)
  (h2 : (a - d) + a + (a + d) = 12)
  (h3 : ∃ r : ℝ, (a - d + 1) * r = a + 4 ∧ (a + 4) * r = a + d + 11) :
  ∃ b : ℕ → ℝ, (∀ n : ℕ, b (n + 1) = 2 * b n) ∧ b 2 = a - d + 1 ∧ b 3 = a + 4 ∧ b 4 = a + d + 11 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_from_arithmetic_l3772_377287


namespace NUMINAMATH_CALUDE_initial_strawberry_weight_l3772_377269

/-- The initial total weight of strawberries collected by Marco and his dad -/
def initial_total (marco_weight dad_weight lost_weight : ℕ) : ℕ :=
  marco_weight + dad_weight + lost_weight

/-- Proof that the initial total weight of strawberries is 36 pounds -/
theorem initial_strawberry_weight :
  ∀ (marco_weight dad_weight lost_weight : ℕ),
    marco_weight = 12 →
    dad_weight = 16 →
    lost_weight = 8 →
    initial_total marco_weight dad_weight lost_weight = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_strawberry_weight_l3772_377269


namespace NUMINAMATH_CALUDE_future_ratio_years_l3772_377205

/-- Represents the ages and time in the problem -/
structure AgeData where
  vimal_initial : ℕ  -- Vimal's age 6 years ago
  saroj_initial : ℕ  -- Saroj's age 6 years ago
  years_passed : ℕ   -- Years passed since the initial ratio

/-- The conditions of the problem -/
def problem_conditions (data : AgeData) : Prop :=
  data.vimal_initial * 5 = data.saroj_initial * 6 ∧  -- Initial ratio 6:5
  data.saroj_initial + 6 = 16 ∧                      -- Saroj's current age is 16
  (data.vimal_initial + 6 + 4) * 10 = (data.saroj_initial + 6 + 4) * 11  -- Future ratio 11:10

/-- The theorem to be proved -/
theorem future_ratio_years (data : AgeData) :
  problem_conditions data → data.years_passed = 4 := by
  sorry


end NUMINAMATH_CALUDE_future_ratio_years_l3772_377205


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l3772_377248

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1050 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l3772_377248


namespace NUMINAMATH_CALUDE_rowing_current_velocity_l3772_377258

/-- Proves that the velocity of the current is 2 kmph given the conditions of the rowing problem. -/
theorem rowing_current_velocity
  (still_water_speed : ℝ)
  (distance : ℝ)
  (total_time : ℝ)
  (h1 : still_water_speed = 8)
  (h2 : distance = 7.5)
  (h3 : total_time = 2)
  : ∃ v : ℝ, v = 2 ∧ 
    (distance / (still_water_speed + v) + distance / (still_water_speed - v) = total_time) :=
by sorry

end NUMINAMATH_CALUDE_rowing_current_velocity_l3772_377258


namespace NUMINAMATH_CALUDE_circle_configuration_exists_l3772_377231

/-- A configuration of numbers in circles -/
structure CircleConfiguration where
  numbers : Fin 9 → ℕ
  consecutive : ∀ i j : Fin 9, i.val < j.val → numbers i < numbers j
  contains_six : ∃ i : Fin 9, numbers i = 6

/-- The lines connecting the circles -/
inductive Line
  | Line1 : Line
  | Line2 : Line
  | Line3 : Line
  | Line4 : Line
  | Line5 : Line
  | Line6 : Line

/-- The endpoints of each line -/
def lineEndpoints : Line → Fin 9 × Fin 9
  | Line.Line1 => (⟨0, by norm_num⟩, ⟨1, by norm_num⟩)
  | Line.Line2 => (⟨1, by norm_num⟩, ⟨2, by norm_num⟩)
  | Line.Line3 => (⟨2, by norm_num⟩, ⟨3, by norm_num⟩)
  | Line.Line4 => (⟨3, by norm_num⟩, ⟨4, by norm_num⟩)
  | Line.Line5 => (⟨4, by norm_num⟩, ⟨5, by norm_num⟩)
  | Line.Line6 => (⟨5, by norm_num⟩, ⟨0, by norm_num⟩)

/-- The sum of numbers on a line -/
def lineSum (config : CircleConfiguration) (line : Line) : ℕ :=
  let (a, b) := lineEndpoints line
  config.numbers a + config.numbers b

/-- The theorem statement -/
theorem circle_configuration_exists :
  ∃ config : CircleConfiguration, ∀ line : Line, lineSum config line = 23 := by
  sorry


end NUMINAMATH_CALUDE_circle_configuration_exists_l3772_377231


namespace NUMINAMATH_CALUDE_s_4_equals_14916_l3772_377259

-- Define s(n) as a function that attaches the first n perfect squares
def s (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

-- Theorem to prove
theorem s_4_equals_14916 : s 4 = 14916 := by
  sorry

end NUMINAMATH_CALUDE_s_4_equals_14916_l3772_377259


namespace NUMINAMATH_CALUDE_check_max_value_l3772_377235

theorem check_max_value (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →  -- x is a two-digit number
  (10 ≤ y ∧ y ≤ 99) →  -- y is a two-digit number
  (100 * x + y) - (100 * y + x) = 2061 →  -- difference between correct and incorrect amounts
  x ≤ 78 :=
by sorry

end NUMINAMATH_CALUDE_check_max_value_l3772_377235


namespace NUMINAMATH_CALUDE_money_ratio_l3772_377216

/-- Represents the money of each person -/
structure Money where
  natasha : ℚ
  carla : ℚ
  cosima : ℚ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.natasha = 60 ∧
  m.carla = 2 * m.cosima ∧
  (7/5) * (m.natasha + m.carla + m.cosima) - (m.natasha + m.carla + m.cosima) = 36

/-- The theorem to prove -/
theorem money_ratio (m : Money) : 
  problem_conditions m → m.natasha / m.carla = 3 / 1 := by
  sorry


end NUMINAMATH_CALUDE_money_ratio_l3772_377216


namespace NUMINAMATH_CALUDE_complex_cube_root_sum_l3772_377247

theorem complex_cube_root_sum (a b : ℤ) (z : ℂ) : 
  z = a + b * Complex.I ∧ z^3 = 2 + 11 * Complex.I → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_sum_l3772_377247


namespace NUMINAMATH_CALUDE_max_piece_length_l3772_377246

def rope_lengths : List Nat := [48, 72, 120, 144]

def min_pieces : Nat := 5

def is_valid_piece_length (len : Nat) : Bool :=
  rope_lengths.all (fun rope => rope % len = 0 ∧ rope / len ≥ min_pieces)

theorem max_piece_length :
  ∃ (max_len : Nat), max_len = 8 ∧
    is_valid_piece_length max_len ∧
    ∀ (len : Nat), len > max_len → ¬is_valid_piece_length len :=
by sorry

end NUMINAMATH_CALUDE_max_piece_length_l3772_377246


namespace NUMINAMATH_CALUDE_distance_one_fourth_from_perigee_l3772_377219

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ
  apogee : ℝ

/-- Calculates the distance from the focus to a point on the major axis of an elliptical orbit -/
def distanceFromFocus (orbit : EllipticalOrbit) (fraction : ℝ) : ℝ :=
  let majorAxis := orbit.apogee + orbit.perigee
  let centerToFocus := Real.sqrt ((majorAxis / 2) ^ 2 - orbit.perigee ^ 2)
  let distanceFromPerigee := fraction * majorAxis
  distanceFromPerigee

/-- Theorem: For an elliptical orbit with perigee 3 AU and apogee 15 AU,
    the distance from the focus to a point 1/4 of the way from perigee to apogee
    along the major axis is 4.5 AU -/
theorem distance_one_fourth_from_perigee (orbit : EllipticalOrbit)
    (h1 : orbit.perigee = 3)
    (h2 : orbit.apogee = 15) :
    distanceFromFocus orbit (1/4) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_distance_one_fourth_from_perigee_l3772_377219


namespace NUMINAMATH_CALUDE_right_triangle_side_relation_l3772_377207

theorem right_triangle_side_relation (a b c x : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b = c * x →
  1 < x ∧ x ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_relation_l3772_377207


namespace NUMINAMATH_CALUDE_history_science_books_l3772_377253

def total_books : ℕ := 120
def school_books : ℕ := 25
def sports_books : ℕ := 35

theorem history_science_books : total_books - (school_books + sports_books) = 60 := by
  sorry

end NUMINAMATH_CALUDE_history_science_books_l3772_377253


namespace NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l3772_377277

theorem cubic_equation_sum_of_cubes :
  ∃ (r s t : ℝ),
    (∀ x : ℝ, (x - Real.rpow 17 (1/3 : ℝ)) * (x - Real.rpow 37 (1/3 : ℝ)) * (x - Real.rpow 57 (1/3 : ℝ)) = -1/2 ↔ x = r ∨ x = s ∨ x = t) →
    r^3 + s^3 + t^3 = 107.5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l3772_377277


namespace NUMINAMATH_CALUDE_max_ab_value_l3772_377227

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 4) :
  ab ≤ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 4 ∧ a₀*b₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l3772_377227


namespace NUMINAMATH_CALUDE_fourth_game_score_l3772_377224

def game_scores (game1 game2 game3 game4 total : ℕ) : Prop :=
  game1 = 10 ∧ game2 = 14 ∧ game3 = 6 ∧ game1 + game2 + game3 + game4 = total

theorem fourth_game_score (game1 game2 game3 game4 total : ℕ) :
  game_scores game1 game2 game3 game4 total → total = 40 → game4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_game_score_l3772_377224


namespace NUMINAMATH_CALUDE_slide_boys_count_l3772_377252

theorem slide_boys_count (initial_boys : ℕ) (additional_boys : ℕ) 
  (h1 : initial_boys = 22) 
  (h2 : additional_boys = 13) : 
  initial_boys + additional_boys = 35 := by
  sorry

end NUMINAMATH_CALUDE_slide_boys_count_l3772_377252


namespace NUMINAMATH_CALUDE_students_behind_yoongi_count_l3772_377283

/-- The number of students in the line. -/
def total_students : ℕ := 20

/-- Jungkook's position in the line. -/
def jungkook_position : ℕ := 3

/-- The number of students between Jungkook and Yoongi. -/
def students_between : ℕ := 5

/-- Yoongi's position in the line. -/
def yoongi_position : ℕ := jungkook_position + students_between + 1

/-- The number of students behind Yoongi. -/
def students_behind_yoongi : ℕ := total_students - yoongi_position

theorem students_behind_yoongi_count : students_behind_yoongi = 11 := by sorry

end NUMINAMATH_CALUDE_students_behind_yoongi_count_l3772_377283


namespace NUMINAMATH_CALUDE_inequalities_hold_l3772_377255

theorem inequalities_hold (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) : 
  (a - d > b - c) ∧ (a * d^2 > b * c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l3772_377255


namespace NUMINAMATH_CALUDE_parabola_with_directrix_x_eq_1_l3772_377292

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- The standard equation of a parabola represents the set of points (x, y) that satisfy the parabola's definition. -/
def standard_equation (p : Parabola) : (ℝ × ℝ) → Prop :=
  sorry

theorem parabola_with_directrix_x_eq_1 (p : Parabola) (h : p.directrix = 1) :
  standard_equation p = fun (x, y) ↦ y^2 = -4*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_with_directrix_x_eq_1_l3772_377292


namespace NUMINAMATH_CALUDE_negation_equivalence_l3772_377295

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3772_377295


namespace NUMINAMATH_CALUDE_smallest_multiple_36_with_digit_sum_multiple_9_l3772_377273

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

theorem smallest_multiple_36_with_digit_sum_multiple_9 :
  ∃ (k : ℕ), k > 0 ∧ 36 * k = 36 ∧
  (∀ m : ℕ, m > 0 ∧ m < k → ¬(∃ n : ℕ, 36 * m = 36 * n ∧ 9 ∣ sumOfDigits (36 * n))) ∧
  (9 ∣ sumOfDigits 36) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_36_with_digit_sum_multiple_9_l3772_377273


namespace NUMINAMATH_CALUDE_marcy_votes_l3772_377217

theorem marcy_votes (joey_votes : ℕ) (barry_votes : ℕ) (marcy_votes : ℕ) : 
  joey_votes = 8 →
  barry_votes = 2 * (joey_votes + 3) →
  marcy_votes = 3 * barry_votes →
  marcy_votes = 66 := by
  sorry

end NUMINAMATH_CALUDE_marcy_votes_l3772_377217


namespace NUMINAMATH_CALUDE_binomial_square_condition_l3772_377261

theorem binomial_square_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l3772_377261


namespace NUMINAMATH_CALUDE_minimum_value_a2_plus_b2_l3772_377213

theorem minimum_value_a2_plus_b2 (a b : ℝ) : 
  (∃ k : ℕ, (20 : ℝ) = k * a^3 * b^3 ∧ Nat.choose 6 k * a^(6-k) * b^k = (20 : ℝ) * a^(6-k) * b^k) → 
  a^2 + b^2 ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀^2 + b₀^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_a2_plus_b2_l3772_377213


namespace NUMINAMATH_CALUDE_greatest_x_value_l3772_377260

theorem greatest_x_value : ∃ (x : ℤ), (∀ (y : ℤ), 2.134 * (10 : ℝ) ^ y < 21000 → y ≤ x) ∧ 2.134 * (10 : ℝ) ^ x < 21000 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3772_377260


namespace NUMINAMATH_CALUDE_maggies_earnings_proof_l3772_377201

/-- Calculates Maggie's earnings from selling magazine subscriptions -/
def maggies_earnings (price_per_subscription : ℕ) 
                     (parents_subscriptions : ℕ)
                     (grandfather_subscriptions : ℕ)
                     (neighbor1_subscriptions : ℕ) : ℕ :=
  let total_subscriptions := parents_subscriptions + 
                             grandfather_subscriptions + 
                             neighbor1_subscriptions + 
                             (2 * neighbor1_subscriptions)
  price_per_subscription * total_subscriptions

theorem maggies_earnings_proof : 
  maggies_earnings 5 4 1 2 = 55 := by
  sorry

#eval maggies_earnings 5 4 1 2

end NUMINAMATH_CALUDE_maggies_earnings_proof_l3772_377201


namespace NUMINAMATH_CALUDE_ryan_final_tokens_l3772_377214

def token_calculation (initial_tokens : ℕ) : ℕ :=
  let after_pacman := initial_tokens - (2 * initial_tokens / 3)
  let after_candy_crush := after_pacman - (after_pacman / 2)
  let after_skiball := after_candy_crush - 7
  let after_friend_borrowed := after_skiball - 5
  let after_friend_returned := after_friend_borrowed + 8
  let after_parents_bought := after_friend_returned + (10 * 7)
  after_parents_bought - 3

theorem ryan_final_tokens : 
  token_calculation 36 = 75 := by sorry

end NUMINAMATH_CALUDE_ryan_final_tokens_l3772_377214


namespace NUMINAMATH_CALUDE_function_properties_l3772_377288

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 - a*x + 1)

theorem function_properties (a : ℝ) :
  (∀ x y : ℝ, y = f a 0 → 3*x + y - 1 = 0 → x = 0) →
  a = 4 ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1-ε) (-1+ε), f 4 (-1) ≥ f 4 x) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3-ε) (3+ε), f 4 3 ≤ f 4 x) ∧
  f 4 (-1) = 6 / Real.exp 1 ∧
  f 4 3 = -2 * Real.exp 3 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3772_377288


namespace NUMINAMATH_CALUDE_divisors_of_1728_power_1728_l3772_377229

theorem divisors_of_1728_power_1728 :
  ∃! n : ℕ, n = (Finset.filter
    (fun d => (Finset.filter (fun x => x ∣ d) (Finset.range (d + 1))).card = 1728)
    (Finset.filter (fun x => x ∣ 1728^1728) (Finset.range (1728^1728 + 1)))).card :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_1728_power_1728_l3772_377229


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3772_377211

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 600 → s^3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3772_377211


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l3772_377297

/-- Systematic sampling selection function -/
def systematicSample (initialSelection : ℕ) (interval : ℕ) (groupNumber : ℕ) : ℕ :=
  initialSelection + interval * (groupNumber - 1)

/-- Theorem for the systematic sampling problem -/
theorem systematic_sampling_problem (totalStudents : ℕ) (sampleSize : ℕ) (interval : ℕ) 
    (initialSelection : ℕ) (targetGroupStart : ℕ) (targetGroupEnd : ℕ) :
    totalStudents = 800 →
    sampleSize = 50 →
    interval = 16 →
    initialSelection = 7 →
    targetGroupStart = 65 →
    targetGroupEnd = 80 →
    systematicSample initialSelection interval 
      ((targetGroupStart - 1) / interval + 1) = 71 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_problem_l3772_377297


namespace NUMINAMATH_CALUDE_max_polygon_size_no_parallel_sides_l3772_377257

/-- A type representing a point on a circle -/
structure CirclePoint where
  angle : ℝ
  -- Assuming angle is in radians and normalized to [0, 2π)

/-- The number of points marked on the circle -/
def num_points : ℕ := 2012

/-- The set of all points on the circle -/
def circle_points : Finset CirclePoint :=
  sorry

/-- Predicate to check if two line segments are parallel -/
def are_parallel (p1 p2 p3 p4 : CirclePoint) : Prop :=
  sorry

/-- Predicate to check if a set of points forms a convex polygon -/
def is_convex_polygon (points : Finset CirclePoint) : Prop :=
  sorry

/-- The main theorem -/
theorem max_polygon_size_no_parallel_sides :
  ∃ (points : Finset CirclePoint),
    points.card = 1509 ∧
    is_convex_polygon points ∧
    (∀ (p1 p2 p3 p4 : CirclePoint),
      p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points →
      p1 ≠ p2 → p3 ≠ p4 → ¬(are_parallel p1 p2 p3 p4)) ∧
    (∀ (larger_set : Finset CirclePoint),
      larger_set.card > 1509 →
      is_convex_polygon larger_set →
      (∃ (q1 q2 q3 q4 : CirclePoint),
        q1 ∈ larger_set ∧ q2 ∈ larger_set ∧ q3 ∈ larger_set ∧ q4 ∈ larger_set ∧
        q1 ≠ q2 ∧ q3 ≠ q4 ∧ are_parallel q1 q2 q3 q4)) :=
by sorry


end NUMINAMATH_CALUDE_max_polygon_size_no_parallel_sides_l3772_377257


namespace NUMINAMATH_CALUDE_units_digit_product_l3772_377245

theorem units_digit_product : (5^2 + 1) * (5^3 + 1) * (5^23 + 1) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l3772_377245


namespace NUMINAMATH_CALUDE_painters_work_days_l3772_377285

/-- Represents the time taken to complete a job given a number of painters -/
def time_to_complete (num_painters : ℕ) (work_days : ℚ) : ℚ := num_painters * work_days

/-- Proves that if 6 painters can finish a job in 2 work-days, 
    then 4 painters will take 3 work-days to finish the same job -/
theorem painters_work_days (initial_painters : ℕ) (initial_days : ℚ) 
  (new_painters : ℕ) : 
  initial_painters = 6 → initial_days = 2 → new_painters = 4 →
  time_to_complete new_painters (3 : ℚ) = time_to_complete initial_painters initial_days :=
by
  sorry

end NUMINAMATH_CALUDE_painters_work_days_l3772_377285


namespace NUMINAMATH_CALUDE_flag_arrangements_count_l3772_377296

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def M : ℕ :=
  /- Number of ways to choose 11 positions out of 13 for red flags -/
  let red_positions := Nat.choose 13 11
  /- Number of ways to place the divider between flagpoles -/
  let divider_positions := 13
  /- Total number of arrangements -/
  let total_arrangements := red_positions * divider_positions
  /- Number of invalid arrangements (where one pole gets no flag) -/
  let invalid_arrangements := 2 * red_positions
  /- Final number of valid arrangements -/
  total_arrangements - invalid_arrangements

/-- Theorem stating that M is equal to 858 -/
theorem flag_arrangements_count : M = 858 := by sorry

end NUMINAMATH_CALUDE_flag_arrangements_count_l3772_377296


namespace NUMINAMATH_CALUDE_tribe_leadership_combinations_l3772_377274

theorem tribe_leadership_combinations (n : ℕ) (h : n = 15) : 
  (n) *                             -- Choose the chief
  (Nat.choose (n - 1) 2) *          -- Choose 2 supporting chiefs
  (Nat.choose (n - 3) 2) *          -- Choose 2 inferior officers for chief A
  (Nat.choose (n - 5) 2) *          -- Choose 2 assistants for A's officers
  (Nat.choose (n - 7) 2) *          -- Choose 2 inferior officers for chief B
  (Nat.choose (n - 9) 2) *          -- Choose 2 assistants for B's officers
  (Nat.choose (n - 11) 2) *         -- Choose 2 assistants for B's officers
  (Nat.choose (n - 13) 2) = 400762320000 := by
sorry

end NUMINAMATH_CALUDE_tribe_leadership_combinations_l3772_377274


namespace NUMINAMATH_CALUDE_experiment_is_conditional_control_l3772_377294

-- Define the types of control experiments
inductive ControlType
  | Blank
  | Standard
  | Mutual
  | Conditional

-- Define the components of a culture medium
structure CultureMedium where
  urea : Bool
  nitrate : Bool
  otherComponents : Set String

-- Define an experimental group
structure ExperimentalGroup where
  medium : CultureMedium

-- Define the experiment
structure Experiment where
  groupA : ExperimentalGroup
  groupB : ExperimentalGroup
  sameOtherConditions : Bool

def isConditionalControl (exp : Experiment) : Prop :=
  exp.groupA.medium.urea = true ∧
  exp.groupA.medium.nitrate = false ∧
  exp.groupB.medium.urea = true ∧
  exp.groupB.medium.nitrate = true ∧
  exp.groupA.medium.otherComponents = exp.groupB.medium.otherComponents ∧
  exp.sameOtherConditions = true

theorem experiment_is_conditional_control (exp : Experiment) 
  (h1 : exp.groupA.medium.urea = true)
  (h2 : exp.groupA.medium.nitrate = false)
  (h3 : exp.groupB.medium.urea = true)
  (h4 : exp.groupB.medium.nitrate = true)
  (h5 : exp.groupA.medium.otherComponents = exp.groupB.medium.otherComponents)
  (h6 : exp.sameOtherConditions = true) :
  isConditionalControl exp :=
by sorry

end NUMINAMATH_CALUDE_experiment_is_conditional_control_l3772_377294


namespace NUMINAMATH_CALUDE_machine_production_l3772_377264

/-- The number of shirts produced by a machine in a given time -/
def shirts_produced (shirts_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  shirts_per_minute * minutes

/-- Theorem: A machine that produces 6 shirts per minute, operating for 12 minutes, will produce 72 shirts -/
theorem machine_production :
  shirts_produced 6 12 = 72 := by
  sorry

end NUMINAMATH_CALUDE_machine_production_l3772_377264


namespace NUMINAMATH_CALUDE_dollar_cube_difference_l3772_377215

/-- The dollar operation: a $ b = (a + b)² + ab -/
def dollar (a b : ℝ) : ℝ := (a + b)^2 + a * b

/-- Theorem: For any real numbers x and y, (x - y)³ $ (y - x)³ = -(x - y)⁶ -/
theorem dollar_cube_difference (x y : ℝ) : 
  dollar ((x - y)^3) ((y - x)^3) = -((x - y)^6) := by
  sorry

end NUMINAMATH_CALUDE_dollar_cube_difference_l3772_377215


namespace NUMINAMATH_CALUDE_parabola_and_slope_theorem_l3772_377221

-- Define the parabola E
def E (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line that intersects E
def intersecting_line (m : ℝ) (x y : ℝ) : Prop := x = m*y + 3

-- Define points A and B as the intersection points
def A (p m : ℝ) : ℝ × ℝ := sorry
def B (p m : ℝ) : ℝ × ℝ := sorry

-- Define the dot product condition
def dot_product_condition (p m : ℝ) : Prop :=
  let a := A p m
  let b := B p m
  (a.1 * b.1 + a.2 * b.2) = 6

-- Define point C
def C : ℝ × ℝ := (-3, 0)

-- Define slopes k₁ and k₂
def k₁ (p m : ℝ) : ℝ := sorry
def k₂ (p m : ℝ) : ℝ := sorry

theorem parabola_and_slope_theorem (p m : ℝ) :
  E p (A p m).1 (A p m).2 ∧
  E p (B p m).1 (B p m).2 ∧
  intersecting_line m (A p m).1 (A p m).2 ∧
  intersecting_line m (B p m).1 (B p m).2 ∧
  dot_product_condition p m →
  (p = 1/2) ∧ 
  (1 / (k₁ p m)^2 + 1 / (k₂ p m)^2 - 2*m^2 = 24) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_slope_theorem_l3772_377221


namespace NUMINAMATH_CALUDE_inequality_proof_l3772_377209

theorem inequality_proof (a b c d x y : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) (hd : d > 1)
  (h1 : a^x + b^y = (a^2 + b^2)^x)
  (h2 : c^x + d^y = 2^y * (c*d)^(y/2)) :
  x < y := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3772_377209


namespace NUMINAMATH_CALUDE_train_speed_l3772_377220

/-- The speed of a train given its length, time to cross a person, and the person's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) :
  train_length = 1200 →
  crossing_time = 71.99424046076314 →
  man_speed = 3 →
  ∃ (train_speed : ℝ), 
    (abs (train_speed - 63.00468) < 0.00001) ∧ 
    (train_speed * 1000 / 3600 - man_speed * 1000 / 3600) * crossing_time = train_length :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l3772_377220


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l3772_377249

theorem largest_n_for_equation : 
  ∃ (x y z : ℕ+), 8^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 10 ∧ 
  ∀ (n : ℕ+), n > 8 → ¬∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 10 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l3772_377249


namespace NUMINAMATH_CALUDE_box_two_three_neg_two_l3772_377270

-- Define the box operation for integers a, b, and c
def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

-- Theorem statement
theorem box_two_three_neg_two :
  box 2 3 (-2) = 107 / 9 := by sorry

end NUMINAMATH_CALUDE_box_two_three_neg_two_l3772_377270


namespace NUMINAMATH_CALUDE_base_is_six_l3772_377244

/-- The sum of all single-digit numbers in base b, including 0 and twice the largest single-digit number -/
def sum_digits (b : ℕ) : ℚ :=
  (b^2 + 3*b - 4) / 2

/-- Representation of 107 in base b -/
def base_b_107 (b : ℕ) : ℕ :=
  b^2 + 7

theorem base_is_six (b : ℕ) (h_pos : b > 0) :
  sum_digits b = base_b_107 b → b = 6 :=
by sorry

end NUMINAMATH_CALUDE_base_is_six_l3772_377244


namespace NUMINAMATH_CALUDE_square_of_sum_of_squares_is_sum_of_squares_l3772_377284

def is_sum_of_two_squares (x : ℕ) : Prop :=
  ∃ (a b : ℕ), x = a^2 + b^2 ∧ a > 0 ∧ b > 0

theorem square_of_sum_of_squares_is_sum_of_squares (n : ℕ) :
  (is_sum_of_two_squares (n - 1) ∧ 
   is_sum_of_two_squares n ∧ 
   is_sum_of_two_squares (n + 1)) →
  (is_sum_of_two_squares (n^2 - 1) ∧ 
   is_sum_of_two_squares (n^2) ∧ 
   is_sum_of_two_squares (n^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_square_of_sum_of_squares_is_sum_of_squares_l3772_377284


namespace NUMINAMATH_CALUDE_sign_sum_theorem_l3772_377228

theorem sign_sum_theorem (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  let sign_sum := x / |x| + y / |y| + z / |z| + w / |w| + (x * y * z * w) / |x * y * z * w|
  sign_sum = 5 ∨ sign_sum = 1 ∨ sign_sum = -1 ∨ sign_sum = -5 := by
  sorry

end NUMINAMATH_CALUDE_sign_sum_theorem_l3772_377228


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3772_377202

/-- A quadratic function f(x) = ax² + bx + c with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The value of the quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_coefficient (f : QuadraticFunction) 
  (vertex_x : f.eval 2 = 5)
  (point : f.eval 1 = 2) : 
  f.a = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3772_377202


namespace NUMINAMATH_CALUDE_blue_balls_removed_l3772_377241

theorem blue_balls_removed (total_balls : Nat) (initial_blue : Nat) (final_probability : Rat) :
  total_balls = 15 →
  initial_blue = 7 →
  final_probability = 1/3 →
  ∃ (removed : Nat), removed = 3 ∧
    (initial_blue - removed : Rat) / (total_balls - removed : Rat) = final_probability :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_removed_l3772_377241


namespace NUMINAMATH_CALUDE_problem_solution_l3772_377223

theorem problem_solution (x y : ℕ+) :
  (x : ℚ) / (Nat.gcd x.val y.val : ℚ) + (y : ℚ) / (Nat.gcd x.val y.val : ℚ) = 18 ∧
  Nat.lcm x.val y.val = 975 →
  x = 75 ∧ y = 195 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3772_377223


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_necessary_not_sufficient_condition_l3772_377291

-- Define the sets A and B
def A : Set ℝ := {x | (x - 5) / (x - 2) ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Part 1
theorem complement_intersection_theorem :
  (Aᶜ ∩ (B 2)ᶜ) = {x | x ≤ 1 ∨ x > 5} := by sorry

-- Part 2
theorem necessary_not_sufficient_condition :
  (∀ x, x ∈ A → x ∈ B a) ∧ ¬(∀ x, x ∈ B a → x ∈ A) →
  3 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_necessary_not_sufficient_condition_l3772_377291


namespace NUMINAMATH_CALUDE_total_weekly_sleep_time_l3772_377234

/-- Represents the sleep patterns of animals -/
structure SleepPattern where
  evenDaySleep : ℕ
  oddDaySleep : ℕ

/-- Calculates the total weekly sleep for an animal given its sleep pattern -/
def weeklyTotalSleep (pattern : SleepPattern) : ℕ :=
  3 * pattern.evenDaySleep + 4 * pattern.oddDaySleep

/-- The sleep pattern of a cougar -/
def cougarSleep : SleepPattern :=
  { evenDaySleep := 4, oddDaySleep := 6 }

/-- The sleep pattern of a zebra -/
def zebraSleep : SleepPattern :=
  { evenDaySleep := cougarSleep.evenDaySleep + 2,
    oddDaySleep := cougarSleep.oddDaySleep + 2 }

/-- Theorem stating the total weekly sleep time for both animals -/
theorem total_weekly_sleep_time :
  weeklyTotalSleep cougarSleep + weeklyTotalSleep zebraSleep = 86 := by
  sorry


end NUMINAMATH_CALUDE_total_weekly_sleep_time_l3772_377234


namespace NUMINAMATH_CALUDE_problem_solution_l3772_377204

-- Define A and B as functions of a and b
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 + 6 * a * b + 8 * a

-- Theorem for the three parts of the problem
theorem problem_solution :
  (∀ a b : ℝ, 2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a) ∧
  (2 * A (-2) 1 - B (-2) 1 = 54) ∧
  (∀ a b : ℝ, (∀ a' : ℝ, 2 * A a' b - B a' b = 2 * A a b - B a b) ↔ b = -1/2) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l3772_377204


namespace NUMINAMATH_CALUDE_flavoring_corn_ratio_comparison_l3772_377200

/-- Standard formulation ratio of flavoring to corn syrup to water -/
def standard_ratio : Fin 3 → ℚ
  | 0 => 1
  | 1 => 12
  | 2 => 30

/-- Sport formulation contains 2 ounces of corn syrup -/
def sport_corn_syrup : ℚ := 2

/-- Sport formulation contains 30 ounces of water -/
def sport_water : ℚ := 30

/-- Sport formulation ratio of flavoring to water is half that of standard formulation -/
def sport_flavoring_water_ratio : ℚ := (standard_ratio 0) / (standard_ratio 2) / 2

/-- Calculate the amount of flavoring in sport formulation -/
def sport_flavoring : ℚ := sport_water * sport_flavoring_water_ratio

/-- Ratio of flavoring to corn syrup in sport formulation -/
def sport_flavoring_corn_ratio : ℚ := sport_flavoring / sport_corn_syrup

/-- Ratio of flavoring to corn syrup in standard formulation -/
def standard_flavoring_corn_ratio : ℚ := (standard_ratio 0) / (standard_ratio 1)

/-- Main theorem: The ratio of (flavoring to corn syrup in sport formulation) to 
    (flavoring to corn syrup in standard formulation) is 3 -/
theorem flavoring_corn_ratio_comparison : 
  sport_flavoring_corn_ratio / standard_flavoring_corn_ratio = 3 := by
  sorry

end NUMINAMATH_CALUDE_flavoring_corn_ratio_comparison_l3772_377200


namespace NUMINAMATH_CALUDE_min_value_theorem_l3772_377232

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_seq : x * Real.log 2 + y * Real.log 2 = 2 * Real.log (Real.sqrt 2)) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a * Real.log 2 + b * Real.log 2 = 2 * Real.log (Real.sqrt 2) →
  1 / x + 9 / y ≤ 1 / a + 9 / b :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3772_377232


namespace NUMINAMATH_CALUDE_toy_price_reduction_l3772_377265

theorem toy_price_reduction :
  ∃! x : ℕ, 1 ≤ x ∧ x ≤ 12 ∧
  (∃ y : ℕ, 1 ≤ y ∧ y ≤ 100 ∧ (13 - x) * y = 781) ∧
  (∀ z : ℕ, z > x → ¬∃ y : ℕ, 1 ≤ y ∧ y ≤ 100 ∧ (13 - z) * y = 781) :=
by sorry

end NUMINAMATH_CALUDE_toy_price_reduction_l3772_377265


namespace NUMINAMATH_CALUDE_train_crossing_time_l3772_377262

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross : ℝ) :
  train_length = 110 →
  train_speed_kmh = 144 →
  time_to_cross = train_length / (train_speed_kmh * (1000 / 3600)) →
  time_to_cross = 2.75 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3772_377262


namespace NUMINAMATH_CALUDE_floor_sqrt_18_squared_l3772_377251

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_18_squared_l3772_377251


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3772_377272

theorem exponent_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3772_377272


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l3772_377239

/-- Calculates the time taken for a monkey to climb a tree given the tree height and climbing behavior -/
def monkey_climb_time (tree_height : ℕ) (hop_up : ℕ) (slip_back : ℕ) : ℕ := sorry

/-- Theorem stating that for the given conditions, the monkey takes 15 hours to reach the top -/
theorem monkey_climb_theorem :
  monkey_climb_time 51 7 4 = 15 := by sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l3772_377239


namespace NUMINAMATH_CALUDE_set_operations_l3772_377280

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | (x - 1) / (x - 6) < 0}

theorem set_operations :
  (A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 6}) ∧
  (A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8}) ∧
  ((Aᶜ ∩ B) = {x : ℝ | 1 < x ∧ x < 2}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l3772_377280


namespace NUMINAMATH_CALUDE_three_digit_numbers_equation_l3772_377242

theorem three_digit_numbers_equation : 
  ∃! (A B : ℕ), 
    100 ≤ A ∧ A < 1000 ∧
    100 ≤ B ∧ B < 1000 ∧
    1000 * A + B = 3 * A * B := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_equation_l3772_377242


namespace NUMINAMATH_CALUDE_oblique_square_area_l3772_377203

/-- Represents a square in an oblique projection drawing as a parallelogram -/
structure ObliqueSquare where
  parallelogram_side : ℝ

/-- The possible areas of the original square given its oblique projection -/
def possible_areas (os : ObliqueSquare) : Set ℝ :=
  {16, 64}

/-- 
Given a square represented as a parallelogram in an oblique projection drawing,
if one side of the parallelogram is 4, then the area of the original square
is either 16 or 64.
-/
theorem oblique_square_area (os : ObliqueSquare) 
  (h : os.parallelogram_side = 4) : 
  ∀ a ∈ possible_areas os, a = 16 ∨ a = 64 := by
  sorry

end NUMINAMATH_CALUDE_oblique_square_area_l3772_377203


namespace NUMINAMATH_CALUDE_point_distance_implies_k_value_l3772_377236

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - 5/2 * y + 1 = 0

-- Define the two points on the line
def point1 (m n : ℝ) : Prop := line_equation m n
def point2 (m n k : ℝ) : Prop := line_equation (m + 1/2) (n + 1/k)

-- Define the condition that the y-coordinate of the second point is 1 unit more than the first
def y_difference (n k : ℝ) : Prop := n + 1/k = n + 1

-- Theorem statement
theorem point_distance_implies_k_value (m n k : ℝ) :
  point1 m n → point2 m n k → y_difference n k → k = 1 := by sorry

end NUMINAMATH_CALUDE_point_distance_implies_k_value_l3772_377236


namespace NUMINAMATH_CALUDE_barbara_age_when_16_l3772_377290

theorem barbara_age_when_16 (mike_current_age barbara_current_age : ℕ) : 
  mike_current_age = 16 →
  barbara_current_age = mike_current_age / 2 →
  ∃ (mike_future_age : ℕ), mike_future_age = 24 ∧ mike_future_age - (mike_current_age - barbara_current_age) = 16 := by
  sorry

end NUMINAMATH_CALUDE_barbara_age_when_16_l3772_377290


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3772_377240

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = -1 - Complex.I → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3772_377240


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3772_377293

theorem quadratic_form_sum (x : ℝ) : ∃ (a h k : ℝ),
  (5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) ∧ (a + h + k = -5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3772_377293


namespace NUMINAMATH_CALUDE_smallest_parabola_coefficient_l3772_377208

theorem smallest_parabola_coefficient (a b c : ℝ) : 
  (∃ (x y : ℝ), y = a * (x - 3/4)^2 - 25/16) →  -- vertex condition
  (∀ (x y : ℝ), y = a * x^2 + b * x + c) →      -- parabola equation
  a > 0 →                                       -- a is positive
  ∃ (n : ℚ), a + b + c = n →                    -- sum is rational
  ∀ (a' : ℝ), (∃ (b' c' : ℝ) (n' : ℚ), 
    (∀ (x y : ℝ), y = a' * x^2 + b' * x + c') ∧ 
    a' > 0 ∧ 
    a' + b' + c' = n') → 
  a ≤ a' →
  a = 41 := by
sorry

end NUMINAMATH_CALUDE_smallest_parabola_coefficient_l3772_377208


namespace NUMINAMATH_CALUDE_sum_power_inequality_l3772_377206

theorem sum_power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^a * b^b + a^b * b^a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_power_inequality_l3772_377206


namespace NUMINAMATH_CALUDE_triangle_distance_inequality_l3772_377250

/-- Given a triangle ABC with an internal point P, prove the inequality involving distances from P to vertices and sides. -/
theorem triangle_distance_inequality 
  (x y z p q r : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hp : p ≥ 0) (hq : q ≥ 0) (hr : r ≥ 0) 
  (h_internal : x + y > z ∧ y + z > x ∧ z + x > y) : 
  x * y * z ≥ (q + r) * (r + p) * (p + q) := by
  sorry

end NUMINAMATH_CALUDE_triangle_distance_inequality_l3772_377250


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l3772_377254

theorem quadratic_root_sum (a b : ℤ) : 
  (∃ x : ℝ, x^2 + a*x + b = 0 ∧ x = Real.sqrt (7 - 4 * Real.sqrt 3)) →
  a + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l3772_377254


namespace NUMINAMATH_CALUDE_chessboard_coverage_l3772_377212

/-- Represents a chessboard square --/
inductive Square
| Black
| White

/-- Represents a 2x1 tile --/
structure Tile :=
  (first : Square)
  (second : Square)

/-- Represents a chessboard --/
def Chessboard := List (List Square)

/-- Creates a standard 8x8 chessboard --/
def standardChessboard : Chessboard :=
  sorry

/-- Removes two squares of different colors from the chessboard --/
def removeSquares (board : Chessboard) (pos1 pos2 : Nat × Nat) : Chessboard :=
  sorry

/-- Checks if a given tile placement is valid on the board --/
def isValidPlacement (board : Chessboard) (tile : Tile) (pos : Nat × Nat) : Bool :=
  sorry

/-- Theorem: A chessboard with two squares of different colors removed can always be covered with 2x1 tiles --/
theorem chessboard_coverage (board : Chessboard) (pos1 pos2 : Nat × Nat) :
  let removedBoard := removeSquares standardChessboard pos1 pos2
  ∃ (tilePlacements : List (Tile × (Nat × Nat))),
    (∀ (placement : Tile × (Nat × Nat)), placement ∈ tilePlacements →
      isValidPlacement removedBoard placement.fst placement.snd) ∧
    (∀ (square : Nat × Nat), square ≠ pos1 ∧ square ≠ pos2 →
      ∃ (placement : Tile × (Nat × Nat)), placement ∈ tilePlacements ∧
        (placement.snd = square ∨ (placement.snd.1 + 1, placement.snd.2) = square)) :=
  sorry


end NUMINAMATH_CALUDE_chessboard_coverage_l3772_377212


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3772_377225

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution (w : ℂ) :
  w - 1 = (1 + w) * i → w = i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3772_377225


namespace NUMINAMATH_CALUDE_problem_statement_l3772_377279

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : Real.exp a * (1 - Real.log b) = 1) :
  (1 < b ∧ b < Real.exp 1) ∧
  (b - a > 1) ∧
  (a > Real.log b) ∧
  (Real.exp a - Real.log b > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3772_377279


namespace NUMINAMATH_CALUDE_tenth_occurrence_shift_l3772_377299

/-- Represents the number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- Calculates the shift for the nth occurrence of a letter -/
def shift (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2 + 1

/-- Theorem: The 10th occurrence of a letter is replaced by the letter 13 positions to its right -/
theorem tenth_occurrence_shift :
  shift 10 % alphabet_size = 13 :=
sorry

end NUMINAMATH_CALUDE_tenth_occurrence_shift_l3772_377299


namespace NUMINAMATH_CALUDE_min_power_cycles_mod1024_l3772_377281

/-- A power cycle is a set of nonnegative integer powers of an integer a. -/
def PowerCycle (a : ℤ) : Set ℤ :=
  {k : ℤ | ∃ n : ℕ, k = a ^ n}

/-- A set of power cycles covers all odd integers modulo 1024 if for any odd integer n,
    there exists a power cycle in the set and an integer k in that cycle
    such that n ≡ k (mod 1024). -/
def CoverAllOddMod1024 (S : Set (Set ℤ)) : Prop :=
  ∀ n : ℤ, Odd n → ∃ C ∈ S, ∃ k ∈ C, n ≡ k [ZMOD 1024]

/-- The theorem states that the minimum number of power cycles required
    to cover all odd integers modulo 1024 is 10. -/
theorem min_power_cycles_mod1024 :
  ∃ S : Set (Set ℤ),
    (∀ C ∈ S, ∃ a : ℤ, C = PowerCycle a) ∧
    CoverAllOddMod1024 S ∧
    S.ncard = 10 ∧
    ∀ T : Set (Set ℤ),
      (∀ C ∈ T, ∃ a : ℤ, C = PowerCycle a) →
      CoverAllOddMod1024 T →
      T.ncard ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_min_power_cycles_mod1024_l3772_377281


namespace NUMINAMATH_CALUDE_max_snacks_is_14_l3772_377286

/-- Represents the maximum number of snacks that can be purchased with a given budget and pricing options. -/
def max_snacks (budget : ℕ) (single_price : ℕ) (pack4_price : ℕ) (pack6_price : ℕ) : ℕ :=
  sorry

/-- Theorem: Given the specific pricing and budget, the maximum number of snacks is 14. -/
theorem max_snacks_is_14 :
  max_snacks 20 2 6 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_snacks_is_14_l3772_377286


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l3772_377237

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l3772_377237


namespace NUMINAMATH_CALUDE_anya_vanya_catchup_l3772_377267

/-- Represents the speeds and catch-up times in the Anya-Vanya problem -/
structure AnyaVanyaProblem where
  anya_speed : ℝ
  vanya_speed : ℝ
  original_catch_up_time : ℝ

/-- The conditions of the problem -/
def problem_conditions (p : AnyaVanyaProblem) : Prop :=
  p.anya_speed > 0 ∧ 
  p.vanya_speed > p.anya_speed ∧
  p.original_catch_up_time > 0 ∧
  (2 * p.vanya_speed - p.anya_speed) * p.original_catch_up_time = 
    3 * (p.vanya_speed - p.anya_speed) * (p.original_catch_up_time / 3)

/-- The theorem to be proved -/
theorem anya_vanya_catchup (p : AnyaVanyaProblem) 
  (h : problem_conditions p) : 
  (2 * p.vanya_speed - p.anya_speed / 2) * (p.original_catch_up_time / 7) = 
  (p.vanya_speed - p.anya_speed) * p.original_catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_anya_vanya_catchup_l3772_377267


namespace NUMINAMATH_CALUDE_second_derivative_y_l3772_377238

noncomputable def x (t : ℝ) : ℝ := Real.log t
noncomputable def y (t : ℝ) : ℝ := Real.sin (2 * t)

theorem second_derivative_y (t : ℝ) (h : t > 0) :
  (deriv^[2] (y ∘ (x⁻¹))) (x t) = -4 * t^2 * Real.sin (2 * t) + 2 * t * Real.cos (2 * t) :=
by sorry

end NUMINAMATH_CALUDE_second_derivative_y_l3772_377238


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l3772_377289

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : distributeBalls 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l3772_377289


namespace NUMINAMATH_CALUDE_blue_marbles_count_l3772_377275

theorem blue_marbles_count (blue yellow : ℕ) : 
  (blue : ℚ) / yellow = 8 / 5 →
  (blue - 12 : ℚ) / (yellow + 21) = 1 / 3 →
  blue = 24 := by
sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l3772_377275


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3772_377278

/-- Represents a number in a given base with two identical digits. -/
def two_digit_number (digit : ℕ) (base : ℕ) : ℕ :=
  digit * base + digit

/-- Checks if a number is valid in a given base. -/
def is_valid_in_base (n : ℕ) (base : ℕ) : Prop :=
  n < base

theorem smallest_dual_base_representation :
  ∃ (C D : ℕ),
    is_valid_in_base C 6 ∧
    is_valid_in_base D 8 ∧
    two_digit_number C 6 = 63 ∧
    two_digit_number D 8 = 63 ∧
    (∀ (C' D' : ℕ),
      is_valid_in_base C' 6 →
      is_valid_in_base D' 8 →
      two_digit_number C' 6 = two_digit_number D' 8 →
      63 ≤ two_digit_number C' 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3772_377278


namespace NUMINAMATH_CALUDE_copy_pages_theorem_l3772_377210

def cost_per_5_pages : ℚ := 7
def pages_per_5 : ℚ := 5
def total_money : ℚ := 1500  -- in cents

def pages_copied : ℕ := 1071

theorem copy_pages_theorem :
  ⌊(total_money / cost_per_5_pages) * pages_per_5⌋ = pages_copied :=
by sorry

end NUMINAMATH_CALUDE_copy_pages_theorem_l3772_377210


namespace NUMINAMATH_CALUDE_no_snow_probability_l3772_377266

theorem no_snow_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 2/3)
  (h2 : p2 = 2/3)
  (h3 : p3 = 3/5)
  (h_independent : True)  -- Representing independence of events
  : (1 - p1) * (1 - p2) * (1 - p3) = 2/45 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l3772_377266
