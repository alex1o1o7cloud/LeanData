import Mathlib

namespace NUMINAMATH_CALUDE_soccer_team_lineups_l4093_409387

/-- The number of possible lineups for a soccer team -/
def number_of_lineups (total_players : ℕ) (goalkeeper : ℕ) (defenders : ℕ) (others : ℕ) : ℕ :=
  total_players * (Nat.choose (total_players - 1) defenders) * (Nat.choose (total_players - 1 - defenders) others)

/-- Theorem: The number of possible lineups for a soccer team of 18 players,
    with 1 goalkeeper, 4 defenders, and 4 other players is 30,544,200 -/
theorem soccer_team_lineups :
  number_of_lineups 18 1 4 4 = 30544200 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_lineups_l4093_409387


namespace NUMINAMATH_CALUDE_sin_140_cos_50_plus_sin_130_cos_40_eq_1_l4093_409347

theorem sin_140_cos_50_plus_sin_130_cos_40_eq_1 :
  Real.sin (140 * π / 180) * Real.cos (50 * π / 180) +
  Real.sin (130 * π / 180) * Real.cos (40 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_140_cos_50_plus_sin_130_cos_40_eq_1_l4093_409347


namespace NUMINAMATH_CALUDE_prob_e_value_l4093_409384

-- Define the probability measure
variable (p : Set α → ℝ)

-- Define events e and f
variable (e f : Set α)

-- Define the conditions
variable (h1 : p f = 75)
variable (h2 : p (e ∩ f) = 75)
variable (h3 : p f / p e = 3)

-- Theorem statement
theorem prob_e_value : p e = 25 := by
  sorry

end NUMINAMATH_CALUDE_prob_e_value_l4093_409384


namespace NUMINAMATH_CALUDE_flight_savings_l4093_409337

/-- Calculates the savings by choosing the cheaper flight between two airlines with given prices and discounts -/
theorem flight_savings (delta_price united_price : ℝ) (delta_discount united_discount : ℝ) :
  delta_price = 850 →
  united_price = 1100 →
  delta_discount = 0.20 →
  united_discount = 0.30 →
  let delta_final := delta_price * (1 - delta_discount)
  let united_final := united_price * (1 - united_discount)
  min delta_final united_final = delta_final →
  united_final - delta_final = 90 := by
sorry


end NUMINAMATH_CALUDE_flight_savings_l4093_409337


namespace NUMINAMATH_CALUDE_total_marbles_l4093_409397

/-- Represents the number of marbles of each color -/
structure MarbleCollection where
  yellow : ℕ
  purple : ℕ
  orange : ℕ

/-- The ratio of marbles (yellow:purple:orange) -/
def marble_ratio : MarbleCollection := ⟨2, 4, 6⟩

/-- The number of orange marbles -/
def orange_count : ℕ := 18

/-- Theorem stating the total number of marbles -/
theorem total_marbles (c : MarbleCollection) 
  (h1 : c.yellow * marble_ratio.purple = c.purple * marble_ratio.yellow)
  (h2 : c.yellow * marble_ratio.orange = c.orange * marble_ratio.yellow)
  (h3 : c.orange = orange_count) : 
  c.yellow + c.purple + c.orange = 36 := by
  sorry


end NUMINAMATH_CALUDE_total_marbles_l4093_409397


namespace NUMINAMATH_CALUDE_mean_of_car_counts_l4093_409368

theorem mean_of_car_counts : 
  let counts : List ℕ := [30, 14, 14, 21, 25]
  (counts.sum / counts.length : ℚ) = 20.8 := by sorry

end NUMINAMATH_CALUDE_mean_of_car_counts_l4093_409368


namespace NUMINAMATH_CALUDE_painting_time_with_break_l4093_409334

/-- The time it takes Doug and Dave to paint a room together, including a break -/
theorem painting_time_with_break (doug_time dave_time break_time : ℝ) 
  (h_doug : doug_time = 4)
  (h_dave : dave_time = 6)
  (h_break : break_time = 2) : 
  ∃ s : ℝ, s = 22 / 5 ∧ 
  (1 / doug_time + 1 / dave_time) * (s - break_time) = 1 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_with_break_l4093_409334


namespace NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l4093_409329

/-- Represents the number of apples of each color in the basket -/
structure Basket :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Represents the number of apples taken from the basket -/
structure ApplesTaken :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Checks if the condition for stopping is met -/
def stopCondition (taken : ApplesTaken) : Prop :=
  taken.green < taken.yellow ∧ taken.yellow < taken.red

/-- The initial state of the basket -/
def initialBasket : Basket :=
  { green := 11, yellow := 14, red := 19 }

/-- Theorem stating the maximum number of yellow apples that can be taken -/
theorem max_yellow_apples (taken : ApplesTaken) 
  (h : taken.yellow ≤ initialBasket.yellow) 
  (h_stop : ¬stopCondition taken) : 
  taken.yellow ≤ 14 :=
sorry

/-- Theorem stating the maximum total number of apples that can be taken -/
theorem max_total_apples (taken : ApplesTaken) 
  (h_green : taken.green ≤ initialBasket.green)
  (h_yellow : taken.yellow ≤ initialBasket.yellow)
  (h_red : taken.red ≤ initialBasket.red)
  (h_stop : ¬stopCondition taken) :
  taken.green + taken.yellow + taken.red ≤ 42 :=
sorry

end NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l4093_409329


namespace NUMINAMATH_CALUDE_oil_price_reduction_l4093_409353

/-- Represents the price reduction problem for oil -/
theorem oil_price_reduction (original_price : ℝ) (original_quantity : ℝ) : 
  (original_price * original_quantity = 684) →
  (0.8 * original_price * (original_quantity + 4) = 684) →
  (0.8 * original_price = 34.20) :=
by sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l4093_409353


namespace NUMINAMATH_CALUDE_linear_function_value_at_negative_two_l4093_409350

/-- A linear function passing through a given point -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x

theorem linear_function_value_at_negative_two 
  (k : ℝ) 
  (h : linear_function k 2 = 4) : 
  linear_function k (-2) = -4 := by
sorry

end NUMINAMATH_CALUDE_linear_function_value_at_negative_two_l4093_409350


namespace NUMINAMATH_CALUDE_sum_of_smaller_radii_eq_original_radius_l4093_409321

/-- A triangle with an inscribed circle and three smaller triangles formed by tangents -/
structure InscribedCircleTriangle where
  /-- The radius of the circle inscribed in the original triangle -/
  r : ℝ
  /-- The radius of the circle inscribed in the first smaller triangle -/
  r₁ : ℝ
  /-- The radius of the circle inscribed in the second smaller triangle -/
  r₂ : ℝ
  /-- The radius of the circle inscribed in the third smaller triangle -/
  r₃ : ℝ
  /-- Ensure all radii are positive -/
  r_pos : r > 0
  r₁_pos : r₁ > 0
  r₂_pos : r₂ > 0
  r₃_pos : r₃ > 0

/-- The sum of the radii of the inscribed circles in the smaller triangles
    equals the radius of the inscribed circle in the original triangle -/
theorem sum_of_smaller_radii_eq_original_radius (t : InscribedCircleTriangle) :
  t.r₁ + t.r₂ + t.r₃ = t.r := by
  sorry

end NUMINAMATH_CALUDE_sum_of_smaller_radii_eq_original_radius_l4093_409321


namespace NUMINAMATH_CALUDE_container_capacity_proof_l4093_409354

/-- The capacity of a container in liters, given the number of portions and volume per portion in milliliters. -/
def container_capacity (portions : ℕ) (ml_per_portion : ℕ) : ℚ :=
  (portions * ml_per_portion : ℚ) / 1000

/-- Proves that a container with 10 portions of 200 ml each has a capacity of 2 liters. -/
theorem container_capacity_proof :
  container_capacity 10 200 = 2 := by
  sorry

#eval container_capacity 10 200

end NUMINAMATH_CALUDE_container_capacity_proof_l4093_409354


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l4093_409356

theorem quadratic_root_problem (a : ℝ) : 
  (2^2 + 2 - a = 0) → 
  (∃ x : ℝ, x ≠ 2 ∧ x^2 + x - a = 0) → 
  ((-3)^2 + (-3) - a = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l4093_409356


namespace NUMINAMATH_CALUDE_exactly_three_false_l4093_409369

-- Define the type for statements
inductive Statement
| one : Statement
| two : Statement
| three : Statement
| four : Statement

-- Define a function to evaluate the truth value of a statement
def evaluate : Statement → (Statement → Bool) → Bool
| Statement.one, f => (f Statement.one && ¬f Statement.two && ¬f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && f Statement.two && ¬f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && ¬f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && ¬f Statement.two && ¬f Statement.three && f Statement.four)
| Statement.two, f => (¬f Statement.one && f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && f Statement.two && ¬f Statement.three && f Statement.four) ||
                      (f Statement.one && ¬f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (f Statement.one && ¬f Statement.two && ¬f Statement.three && f Statement.four) ||
                      (¬f Statement.one && f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (f Statement.one && f Statement.two && ¬f Statement.three && ¬f Statement.four)
| Statement.three, f => (¬f Statement.one && ¬f Statement.two && f Statement.three && ¬f Statement.four)
| Statement.four, f => (f Statement.one && f Statement.two && f Statement.three && f Statement.four)

-- Theorem statement
theorem exactly_three_false :
  ∃ (f : Statement → Bool),
    (∀ s, evaluate s f = f s) ∧
    (f Statement.one = false ∧
     f Statement.two = false ∧
     f Statement.three = true ∧
     f Statement.four = false) :=
by sorry

end NUMINAMATH_CALUDE_exactly_three_false_l4093_409369


namespace NUMINAMATH_CALUDE_marble_distribution_theorem_l4093_409313

/-- The number of ways to distribute marbles to students under specific conditions -/
def marbleDistributionWays : ℕ := 3150

/-- The total number of marbles -/
def totalMarbles : ℕ := 12

/-- The number of red marbles -/
def redMarbles : ℕ := 3

/-- The number of blue marbles -/
def blueMarbles : ℕ := 4

/-- The number of green marbles -/
def greenMarbles : ℕ := 5

/-- The total number of students -/
def totalStudents : ℕ := 12

theorem marble_distribution_theorem :
  marbleDistributionWays = 3150 ∧
  totalMarbles = redMarbles + blueMarbles + greenMarbles ∧
  totalStudents = totalMarbles ∧
  ∃ (distribution : Fin totalStudents → Fin 3),
    (∃ (i j : Fin totalStudents), i ≠ j ∧ distribution i = distribution j) ∧
    (∃ (k : Fin totalStudents), distribution k = 2) :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_theorem_l4093_409313


namespace NUMINAMATH_CALUDE_ratio_sum_last_number_l4093_409304

theorem ratio_sum_last_number (a b c : ℕ) : 
  a + b + c = 1000 → 
  5 * b = a → 
  4 * b = c → 
  c = 400 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_last_number_l4093_409304


namespace NUMINAMATH_CALUDE_test_question_count_l4093_409320

/-- Given a test with the following properties:
  * The test is worth 100 points
  * There are 2-point and 4-point questions
  * There are 30 two-point questions
  Prove that the total number of questions is 40 -/
theorem test_question_count (total_points : ℕ) (two_point_count : ℕ) :
  total_points = 100 →
  two_point_count = 30 →
  ∃ (four_point_count : ℕ),
    total_points = 2 * two_point_count + 4 * four_point_count ∧
    two_point_count + four_point_count = 40 :=
by sorry

end NUMINAMATH_CALUDE_test_question_count_l4093_409320


namespace NUMINAMATH_CALUDE_work_completion_theorem_l4093_409323

theorem work_completion_theorem 
  (total_work : ℕ) 
  (days_group1 : ℕ) 
  (men_group1 : ℕ) 
  (days_group2 : ℕ) : 
  men_group1 * days_group1 = total_work → 
  total_work = days_group2 * (total_work / days_group2) → 
  men_group1 = 10 → 
  days_group1 = 35 → 
  days_group2 = 50 → 
  total_work / days_group2 = 7 :=
by
  sorry

#check work_completion_theorem

end NUMINAMATH_CALUDE_work_completion_theorem_l4093_409323


namespace NUMINAMATH_CALUDE_angle_theta_value_l4093_409315

theorem angle_theta_value (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (20 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 25 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_theta_value_l4093_409315


namespace NUMINAMATH_CALUDE_simplify_expression_l4093_409325

theorem simplify_expression : 
  ((9 * 10^8) * 2^2) / (3 * 2^3 * 10^3) = 150000 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l4093_409325


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l4093_409310

theorem degree_to_radian_conversion :
  ((-300 : ℝ) * (π / 180)) = -(5 / 3 : ℝ) * π :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l4093_409310


namespace NUMINAMATH_CALUDE_sum_of_ages_l4093_409340

/-- Given that in 5 years Nacho will be three times older than Divya, 
    and Divya is currently 5 years old, prove that the sum of their 
    current ages is 30 years. -/
theorem sum_of_ages (nacho_age divya_age : ℕ) : 
  divya_age = 5 → 
  nacho_age + 5 = 3 * (divya_age + 5) → 
  nacho_age + divya_age = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l4093_409340


namespace NUMINAMATH_CALUDE_cube_volume_after_increase_l4093_409395

theorem cube_volume_after_increase (surface_area : ℝ) (increase_factor : ℝ) : 
  surface_area = 864 → increase_factor = 1.5 → 
  (increase_factor * (surface_area / 6).sqrt) ^ 3 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_after_increase_l4093_409395


namespace NUMINAMATH_CALUDE_definite_integral_equals_ln3_minus_ln2_plus_1_l4093_409396

theorem definite_integral_equals_ln3_minus_ln2_plus_1 :
  let a : ℝ := 2 * Real.arctan (1 / 3)
  let b : ℝ := 2 * Real.arctan (1 / 2)
  let f (x : ℝ) := 1 / (Real.sin x * (1 - Real.sin x))
  ∫ x in a..b, f x = Real.log 3 - Real.log 2 + 1 := by sorry

end NUMINAMATH_CALUDE_definite_integral_equals_ln3_minus_ln2_plus_1_l4093_409396


namespace NUMINAMATH_CALUDE_bahs_equal_to_yahs_l4093_409344

/-- Conversion rate from bahs to rahs -/
def bah_to_rah : ℚ := 16 / 10

/-- Conversion rate from rahs to yahs -/
def rah_to_yah : ℚ := 15 / 9

/-- The number of yahs we want to convert -/
def target_yahs : ℚ := 1500

theorem bahs_equal_to_yahs : 
  (target_yahs / (bah_to_rah * rah_to_yah) : ℚ) = 562.5 := by sorry

end NUMINAMATH_CALUDE_bahs_equal_to_yahs_l4093_409344


namespace NUMINAMATH_CALUDE_exists_k_no_roots_l4093_409322

/-- A homogeneous real polynomial of degree 2 -/
def HomogeneousPolynomial2 (a b c : ℝ) (x y : ℝ) : ℝ :=
  a * x^2 + b * x * y + c * y^2

/-- A homogeneous real polynomial of degree 3 -/
noncomputable def HomogeneousPolynomial3 (x y : ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem exists_k_no_roots
  (a b c : ℝ)
  (h_pos : b^2 < 4*a*c) :
  ∃ k : ℝ, k > 0 ∧
    ∀ x y : ℝ, x^2 + y^2 < k →
      HomogeneousPolynomial2 a b c x y = HomogeneousPolynomial3 x y →
        x = 0 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_exists_k_no_roots_l4093_409322


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4093_409372

theorem min_value_quadratic (x y : ℝ) : 2 * x^2 + 3 * y^2 - 8 * x + 6 * y + 25 ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4093_409372


namespace NUMINAMATH_CALUDE_jeff_total_hours_l4093_409360

/-- Represents Jeff's weekly schedule --/
structure JeffSchedule where
  facebook_hours_per_day : ℕ
  weekend_work_ratio : ℕ
  twitter_hours_per_weekend_day : ℕ
  instagram_hours_per_weekday : ℕ
  weekday_work_ratio : ℕ

/-- Calculates Jeff's total hours spent on work, Twitter, and Instagram in a week --/
def total_hours (schedule : JeffSchedule) : ℕ :=
  let weekend_work_hours := 2 * (schedule.facebook_hours_per_day / schedule.weekend_work_ratio)
  let weekday_work_hours := 5 * (4 * (schedule.facebook_hours_per_day + schedule.instagram_hours_per_weekday))
  let twitter_hours := 2 * schedule.twitter_hours_per_weekend_day
  let instagram_hours := 5 * schedule.instagram_hours_per_weekday
  weekend_work_hours + weekday_work_hours + twitter_hours + instagram_hours

/-- Theorem stating Jeff's total hours in a week --/
theorem jeff_total_hours : 
  ∀ (schedule : JeffSchedule),
    schedule.facebook_hours_per_day = 3 ∧
    schedule.weekend_work_ratio = 3 ∧
    schedule.twitter_hours_per_weekend_day = 2 ∧
    schedule.instagram_hours_per_weekday = 1 ∧
    schedule.weekday_work_ratio = 4 →
    total_hours schedule = 91 := by
  sorry

end NUMINAMATH_CALUDE_jeff_total_hours_l4093_409360


namespace NUMINAMATH_CALUDE_expression_equality_l4093_409388

theorem expression_equality : 7^2 - 2*(5) + 4^2 / 2 = 47 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l4093_409388


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l4093_409381

theorem isosceles_right_triangle_hypotenuse (a c : ℝ) :
  a > 0 →  -- Ensure positive side length
  c > 0 →  -- Ensure positive hypotenuse length
  c^2 = 2 * a^2 →  -- Pythagorean theorem for isosceles right triangle
  2 * a + c = 4 + 4 * Real.sqrt 2 →  -- Perimeter condition
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l4093_409381


namespace NUMINAMATH_CALUDE_martha_router_time_l4093_409362

theorem martha_router_time (x : ℝ) 
  (router_time : x > 0)
  (hold_time : ℝ)
  (hold_time_def : hold_time = 6 * x)
  (yelling_time : ℝ)
  (yelling_time_def : yelling_time = 3 * x)
  (total_time : x + hold_time + yelling_time = 100) :
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_martha_router_time_l4093_409362


namespace NUMINAMATH_CALUDE_amount_saved_christine_savings_l4093_409380

/-- Calculates the amount saved by a salesperson given their commission rate, total sales, and allocation for personal needs. -/
theorem amount_saved 
  (commission_rate : ℚ) 
  (total_sales : ℚ) 
  (personal_needs_allocation : ℚ) : ℚ :=
  let commission_earned := commission_rate * total_sales
  let amount_for_personal_needs := personal_needs_allocation * commission_earned
  commission_earned - amount_for_personal_needs

/-- Proves that given the specific conditions, Christine saved $1152. -/
theorem christine_savings : 
  amount_saved (12/100) 24000 (60/100) = 1152 := by
  sorry

end NUMINAMATH_CALUDE_amount_saved_christine_savings_l4093_409380


namespace NUMINAMATH_CALUDE_simplify_expression_l4093_409378

theorem simplify_expression : 3000 * (3000 ^ 3000) + 3000 * (3000 ^ 3000) = 2 * 3000 ^ 3001 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4093_409378


namespace NUMINAMATH_CALUDE_inequality_solution_l4093_409398

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x + 2) ≥ 3 / (x - 2) + 7 / 4) ↔ (x ∈ Set.Ioo (-2) 2 ∪ Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l4093_409398


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l4093_409331

theorem polygon_sides_from_angle_sum :
  ∀ n : ℕ,
  (n - 2) * 180 = 720 →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l4093_409331


namespace NUMINAMATH_CALUDE_eighth_term_value_l4093_409390

/-- An increasing sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a n + a (n + 1))

theorem eighth_term_value 
  (a : ℕ → ℕ) 
  (h_seq : RecurrenceSequence a) 
  (h_seventh : a 7 = 120) : 
  a 8 = 194 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_value_l4093_409390


namespace NUMINAMATH_CALUDE_geometry_theorem_l4093_409361

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Axioms
axiom different_lines {l1 l2 : Line} : l1 ≠ l2
axiom different_planes {p1 p2 : Plane} : p1 ≠ p2

-- Theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane) :
  (perpendicular m n ∧ perpendicularLP m α ∧ ¬subset n α → parallel n α) ∧
  (perpendicular m n ∧ perpendicularLP m α ∧ perpendicularLP n β → perpendicularPP α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorem_l4093_409361


namespace NUMINAMATH_CALUDE_binomial_9_5_l4093_409300

theorem binomial_9_5 : Nat.choose 9 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_5_l4093_409300


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l4093_409314

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define the midpoint relationship
def is_midpoint (mx my px py : ℝ) : Prop :=
  mx = (px + 4) / 2 ∧ my = py / 2

-- Theorem statement
theorem trajectory_of_midpoint :
  ∀ (x y : ℝ), 
    (∃ (x1 y1 : ℝ), on_ellipse x1 y1 ∧ is_midpoint x y x1 y1) →
    (x - 2)^2 + 4*y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l4093_409314


namespace NUMINAMATH_CALUDE_range_of_a_range_of_t_l4093_409365

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Statement for the range of a
theorem range_of_a : 
  {a : ℝ | ∃ x, a ≥ f x} = {a : ℝ | a ≥ -5/2} := by sorry

-- Statement for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ -t^2 - 5/2*t - 1} = 
  {t : ℝ | t ≥ 1/2 ∨ t ≤ -3} := by sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_t_l4093_409365


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l4093_409367

theorem factorization_of_quadratic (a : ℚ) : 2 * a^2 - 4 * a = 2 * a * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l4093_409367


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4093_409335

theorem sum_of_squares_of_roots (q r s : ℝ) : 
  (3 * q^3 - 4 * q^2 + 6 * q + 15 = 0) →
  (3 * r^3 - 4 * r^2 + 6 * r + 15 = 0) →
  (3 * s^3 - 4 * s^2 + 6 * s + 15 = 0) →
  q^2 + r^2 + s^2 = -20/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4093_409335


namespace NUMINAMATH_CALUDE_notebook_distribution_l4093_409349

theorem notebook_distribution (C : ℕ) (N : ℕ) 
  (h1 : N = C^2 / 8)
  (h2 : N = 8 * (C / 2))
  : N = 512 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l4093_409349


namespace NUMINAMATH_CALUDE_range_of_x_when_proposition_false_l4093_409316

theorem range_of_x_when_proposition_false :
  (∀ a : ℝ, -1 ≤ a ∧ a ≤ 3 → ∀ x : ℝ, a * x^2 - (2*a - 1) * x + (3 - a) ≥ 0) →
  ∀ x : ℝ, ((-1 ≤ x ∧ x ≤ 0) ∨ (5/3 ≤ x ∧ x ≤ 4)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_when_proposition_false_l4093_409316


namespace NUMINAMATH_CALUDE_scooter_initial_price_l4093_409327

/-- The initial purchase price of a scooter, given the repair cost, selling price, and gain percentage. -/
theorem scooter_initial_price (repair_cost selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : repair_cost = 200)
  (h2 : selling_price = 1400)
  (h3 : gain_percent = 40) :
  ∃ (initial_price : ℝ), 
    selling_price = (1 + gain_percent / 100) * (initial_price + repair_cost) ∧ 
    initial_price = 800 := by
  sorry

end NUMINAMATH_CALUDE_scooter_initial_price_l4093_409327


namespace NUMINAMATH_CALUDE_smallest_candy_count_l4093_409385

theorem smallest_candy_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 9) % 7 = 0 ∧
  (∀ m : ℕ, m ≥ 100 ∧ m ≤ 999 ∧ (m + 7) % 9 = 0 ∧ (m - 9) % 7 = 0 → m ≥ n) ∧
  n = 128 :=
by sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l4093_409385


namespace NUMINAMATH_CALUDE_system_solutions_l4093_409317

def is_solution (x y z : ℤ) : Prop :=
  x^2 + y^2 + 2*x + 6*y = -5 ∧
  x^2 + z^2 + 2*x - 4*z = 8 ∧
  y^2 + z^2 + 6*y - 4*z = -3

theorem system_solutions :
  is_solution 1 (-2) (-1) ∧
  is_solution 1 (-2) 5 ∧
  is_solution 1 (-4) (-1) ∧
  is_solution 1 (-4) 5 ∧
  is_solution (-3) (-2) (-1) ∧
  is_solution (-3) (-2) 5 ∧
  is_solution (-3) (-4) (-1) ∧
  is_solution (-3) (-4) 5 :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_l4093_409317


namespace NUMINAMATH_CALUDE_basketball_shot_probability_l4093_409328

theorem basketball_shot_probability (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < 1) (hbb : b < 1) (hcb : c < 1) (sum_prob : a + b + c = 1) (expected_value : 3*a + 2*b = 2) :
  (2/a + 1/(3*b)) ≥ 16/3 := by
sorry

end NUMINAMATH_CALUDE_basketball_shot_probability_l4093_409328


namespace NUMINAMATH_CALUDE_basketball_team_size_l4093_409332

theorem basketball_team_size (total_points : ℕ) (min_score : ℕ) (max_score : ℕ) :
  total_points = 100 →
  min_score = 7 →
  max_score = 23 →
  ∃ (team_size : ℕ) (scores : List ℕ),
    team_size = 12 ∧
    scores.length = team_size ∧
    scores.sum = total_points ∧
    (∀ s ∈ scores, min_score ≤ s ∧ s ≤ max_score) :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_size_l4093_409332


namespace NUMINAMATH_CALUDE_intersection_parallel_line_equation_l4093_409319

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line. -/
theorem intersection_parallel_line_equation 
  (l₁ : ℝ → ℝ → Prop) 
  (l₂ : ℝ → ℝ → Prop)
  (l_parallel : ℝ → ℝ → Prop)
  (result_line : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 2 * x - 3 * y + 2 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 3 * x - 4 * y - 2 = 0)
  (h_parallel : ∀ x y, l_parallel x y ↔ 4 * x - 2 * y + 7 = 0)
  (h_result : ∀ x y, result_line x y ↔ 2 * x - y - 18 = 0) :
  ∃ (x₀ y₀ : ℝ), 
    (l₁ x₀ y₀ ∧ l₂ x₀ y₀) ∧ 
    (∃ (k : ℝ), ∀ x y, result_line x y ↔ l_parallel (x - x₀) (y - y₀)) ∧
    result_line x₀ y₀ := by
  sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_equation_l4093_409319


namespace NUMINAMATH_CALUDE_negative_64_to_two_thirds_power_l4093_409342

theorem negative_64_to_two_thirds_power (x : ℝ) : x = (-64)^(2/3) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_negative_64_to_two_thirds_power_l4093_409342


namespace NUMINAMATH_CALUDE_broccoli_carrot_calorie_ratio_l4093_409307

/-- The number of calories in a pound of carrots -/
def carrot_calories : ℕ := 51

/-- The number of pounds of carrots Tom eats -/
def carrot_pounds : ℕ := 1

/-- The number of pounds of broccoli Tom eats -/
def broccoli_pounds : ℕ := 2

/-- The total number of calories Tom ate -/
def total_calories : ℕ := 85

/-- The number of calories in a pound of broccoli -/
def broccoli_calories : ℚ := (total_calories - carrot_calories * carrot_pounds) / broccoli_pounds

theorem broccoli_carrot_calorie_ratio :
  broccoli_calories / carrot_calories = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_broccoli_carrot_calorie_ratio_l4093_409307


namespace NUMINAMATH_CALUDE_help_sign_white_area_l4093_409352

/-- Represents the dimensions of a rectangular sign -/
structure SignDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a letter painted with 1-unit wide strokes -/
def letterArea (letter : Char) : ℕ :=
  match letter with
  | 'H' => 13
  | 'E' => 9
  | 'L' => 8
  | 'P' => 10
  | _ => 0

/-- Calculates the total area of a word painted with 1-unit wide strokes -/
def wordArea (word : String) : ℕ :=
  word.toList.map letterArea |> List.sum

/-- Theorem: The white area of the sign with "HELP" painted is 35 square units -/
theorem help_sign_white_area (sign : SignDimensions) 
  (h1 : sign.width = 15) 
  (h2 : sign.height = 5) : 
  sign.width * sign.height - wordArea "HELP" = 35 := by
  sorry

end NUMINAMATH_CALUDE_help_sign_white_area_l4093_409352


namespace NUMINAMATH_CALUDE_max_pens_sold_is_226_l4093_409303

/-- Represents the store's promotional sale --/
structure PromotionalSale where
  penProfit : ℕ            -- Profit per pen in yuan
  teddyBearCost : ℕ        -- Cost of teddy bear in yuan
  pensPerPackage : ℕ       -- Number of pens in a promotional package
  totalProfit : ℕ          -- Total profit from the promotion in yuan

/-- Calculates the maximum number of pens sold during the promotional sale --/
def maxPensSold (sale : PromotionalSale) : ℕ :=
  sorry

/-- Theorem stating that for the given promotional sale conditions, 
    the maximum number of pens sold is 226 --/
theorem max_pens_sold_is_226 :
  let sale : PromotionalSale := {
    penProfit := 9
    teddyBearCost := 2
    pensPerPackage := 4
    totalProfit := 1922
  }
  maxPensSold sale = 226 := by
  sorry

end NUMINAMATH_CALUDE_max_pens_sold_is_226_l4093_409303


namespace NUMINAMATH_CALUDE_dow_jones_decrease_l4093_409363

theorem dow_jones_decrease (initial_value end_value : ℝ) : 
  (end_value = initial_value * 0.98) → 
  (end_value = 8722) → 
  (initial_value = 8900) := by
sorry

end NUMINAMATH_CALUDE_dow_jones_decrease_l4093_409363


namespace NUMINAMATH_CALUDE_pineapples_theorem_l4093_409302

/-- Calculates the number of fresh pineapples left in a store. -/
def fresh_pineapples_left (initial : ℕ) (sold : ℕ) (rotten : ℕ) : ℕ :=
  initial - sold - rotten

/-- Proves that the number of fresh pineapples left is 29. -/
theorem pineapples_theorem :
  fresh_pineapples_left 86 48 9 = 29 := by
  sorry

end NUMINAMATH_CALUDE_pineapples_theorem_l4093_409302


namespace NUMINAMATH_CALUDE_john_twice_james_age_john_twice_james_age_proof_l4093_409308

/-- Proves that John will be twice as old as James in 15 years -/
theorem john_twice_james_age : ℕ → Prop :=
  fun years_until_twice_age : ℕ =>
    let john_current_age : ℕ := 39
    let james_brother_age : ℕ := 16
    let age_difference_james_brother : ℕ := 4
    let james_current_age : ℕ := james_brother_age - age_difference_james_brother
    let john_age_3_years_ago : ℕ := john_current_age - 3
    let james_age_in_future : ℕ → ℕ := fun x => james_current_age + x
    ∃ x : ℕ, john_age_3_years_ago = 2 * (james_age_in_future x) →
    (john_current_age + years_until_twice_age = 2 * (james_current_age + years_until_twice_age)) →
    years_until_twice_age = 15

/-- Proof of the theorem -/
theorem john_twice_james_age_proof : john_twice_james_age 15 := by
  sorry

end NUMINAMATH_CALUDE_john_twice_james_age_john_twice_james_age_proof_l4093_409308


namespace NUMINAMATH_CALUDE_cubic_equation_has_real_root_l4093_409355

theorem cubic_equation_has_real_root (a b : ℝ) : 
  ∃ x : ℝ, a * x^3 + a * x + b = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_has_real_root_l4093_409355


namespace NUMINAMATH_CALUDE_binomial_7_choose_4_l4093_409326

theorem binomial_7_choose_4 : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_choose_4_l4093_409326


namespace NUMINAMATH_CALUDE_kabadi_players_count_l4093_409373

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 15

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 25

/-- The number of people who play both kabadi and kho kho -/
def both_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := 35

theorem kabadi_players_count : 
  kabadi_players = total_players - kho_kho_only + both_players :=
by
  sorry

#check kabadi_players_count

end NUMINAMATH_CALUDE_kabadi_players_count_l4093_409373


namespace NUMINAMATH_CALUDE_power_difference_value_l4093_409370

theorem power_difference_value (x m n : ℝ) (hm : x^m = 6) (hn : x^n = 3) : x^(m-n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_value_l4093_409370


namespace NUMINAMATH_CALUDE_hotel_supplies_theorem_l4093_409399

/-- The greatest number of bathrooms that can be stocked identically with given supplies -/
def max_bathrooms (toilet_paper soap towels shower_gel : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd (Nat.gcd toilet_paper soap) towels) shower_gel

/-- Theorem stating that the maximum number of bathrooms that can be stocked
    with the given supplies is 6 -/
theorem hotel_supplies_theorem :
  max_bathrooms 36 18 24 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hotel_supplies_theorem_l4093_409399


namespace NUMINAMATH_CALUDE_no_twelve_consecutive_primes_in_ap_l4093_409359

theorem no_twelve_consecutive_primes_in_ap (a d : ℕ) (h_d : d < 2000) :
  ¬ ∀ k : Fin 12, Nat.Prime (a + k.val * d) := by
  sorry

end NUMINAMATH_CALUDE_no_twelve_consecutive_primes_in_ap_l4093_409359


namespace NUMINAMATH_CALUDE_first_day_over_500_l4093_409379

/-- Represents the number of markers Liam has on a given day -/
def markers (day : ℕ) : ℕ :=
  if day = 1 then 5
  else if day = 2 then 10
  else 5 * 3^(day - 2)

/-- The day of the week as a number from 1 to 7 -/
def dayOfWeek (day : ℕ) : ℕ :=
  (day - 1) % 7 + 1

theorem first_day_over_500 :
  ∃ d : ℕ, markers d > 500 ∧ 
    ∀ k < d, markers k ≤ 500 ∧
    dayOfWeek d = 6 :=
  sorry

end NUMINAMATH_CALUDE_first_day_over_500_l4093_409379


namespace NUMINAMATH_CALUDE_expected_urns_with_one_marble_value_l4093_409389

/-- The number of urns -/
def n : ℕ := 7

/-- The number of marbles -/
def m : ℕ := 5

/-- The probability that a specific urn has exactly one marble -/
def p : ℚ := m * (n - 1)^(m - 1) / n^m

/-- The expected number of urns with exactly one marble -/
def expected_urns_with_one_marble : ℚ := n * p

theorem expected_urns_with_one_marble_value : 
  expected_urns_with_one_marble = 6480 / 2401 := by sorry

end NUMINAMATH_CALUDE_expected_urns_with_one_marble_value_l4093_409389


namespace NUMINAMATH_CALUDE_line_segment_ratios_l4093_409393

/-- Given points A, B, C on a straight line with AC : BC = m : n,
    prove the ratios AC : AB and BC : AB -/
theorem line_segment_ratios
  (A B C : ℝ) -- Points on a real line
  (m n : ℕ) -- Natural numbers for the ratio
  (h_line : (A ≤ B ∧ B ≤ C) ∨ (A ≤ C ∧ C ≤ B) ∨ (B ≤ A ∧ A ≤ C)) -- Points are on a line
  (h_ratio : |C - A| / |C - B| = m / n) : -- Given ratio
  (∃ (r₁ r₂ : ℚ),
    (r₁ = m / (m + n) ∧ r₂ = n / (m + n)) ∨
    (r₁ = m / (n - m) ∧ r₂ = n / (n - m)) ∨
    (m = n ∧ r₁ = 1 / 2 ∧ r₂ = 1 / 2)) ∧
  (|A - C| / |A - B| = r₁ ∧ |B - C| / |A - B| = r₂) :=
sorry

end NUMINAMATH_CALUDE_line_segment_ratios_l4093_409393


namespace NUMINAMATH_CALUDE_probability_theorem_l4093_409312

def total_containers : ℕ := 14
def dry_soil_containers : ℕ := 6
def selected_containers : ℕ := 5
def desired_dry_containers : ℕ := 3

def probability_dry_soil : ℚ :=
  (Nat.choose dry_soil_containers desired_dry_containers *
   Nat.choose (total_containers - dry_soil_containers) (selected_containers - desired_dry_containers)) /
  Nat.choose total_containers selected_containers

theorem probability_theorem :
  probability_dry_soil = 560 / 2002 :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l4093_409312


namespace NUMINAMATH_CALUDE_circular_fields_area_comparison_l4093_409306

theorem circular_fields_area_comparison :
  ∀ (r1 r2 : ℝ),
  r1 > 0 → r2 > 0 →
  r2 / r1 = 10 / 4 →
  (π * r2^2 - π * r1^2) / (π * r1^2) * 100 = 525 :=
by
  sorry

end NUMINAMATH_CALUDE_circular_fields_area_comparison_l4093_409306


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4093_409346

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 2 = 1 →
  a 3 + a 4 = 9 →
  a 5 + a 6 = 81 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4093_409346


namespace NUMINAMATH_CALUDE_range_of_f_l4093_409386

def f (x : ℤ) : ℤ := x^2 - 2*x

def domain : Set ℤ := {x | -2 ≤ x ∧ x ≤ 4}

theorem range_of_f : {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3, 8} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l4093_409386


namespace NUMINAMATH_CALUDE_extremum_implies_not_monotonic_l4093_409377

open Set
open Function

-- Define a real-valued function on R
variable (f : ℝ → ℝ)

-- Define differentiability
variable (h_diff : Differentiable ℝ f)

-- Define the existence of an extremum
variable (h_extremum : ∃ x₀ : ℝ, IsLocalExtremum ℝ f x₀)

-- Theorem statement
theorem extremum_implies_not_monotonic :
  ¬(StrictMono f ∨ StrictAnti f) :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_not_monotonic_l4093_409377


namespace NUMINAMATH_CALUDE_xy_sum_zero_l4093_409339

theorem xy_sum_zero (x y : ℝ) :
  (x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) = 1 →
  x + y = 0 ∧ ∀ z, ((x + Real.sqrt (x^2 + 1)) * (z + Real.sqrt (z^2 + 1)) = 1 → x + z = 0) :=
by sorry

end NUMINAMATH_CALUDE_xy_sum_zero_l4093_409339


namespace NUMINAMATH_CALUDE_product_xy_equals_sqrt_30_6_l4093_409357

/-- Represents a parallelogram EFGH with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ

/-- The product of x and y in the parallelogram EFGH -/
def product_xy (p : Parallelogram) : ℝ → ℝ → ℝ := fun x y => x * y

/-- Theorem: The product of x and y in the given parallelogram is √(30.6) -/
theorem product_xy_equals_sqrt_30_6 (p : Parallelogram) 
  (h1 : p.EF = 54)
  (h2 : ∀ x, p.FG x = 8 * x^2 + 2)
  (h3 : ∀ y, p.GH y = 5 * y^2 + 20)
  (h4 : p.HE = 38) :
  ∃ x y, product_xy p x y = Real.sqrt 30.6 := by
  sorry

#check product_xy_equals_sqrt_30_6

end NUMINAMATH_CALUDE_product_xy_equals_sqrt_30_6_l4093_409357


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l4093_409333

theorem smallest_number_in_sequence (x y z t : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  t = (y + z) / 3 →
  (x + y + z + t) / 4 = 220 →
  x = 2640 / 43 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l4093_409333


namespace NUMINAMATH_CALUDE_ariels_female_fish_l4093_409358

/-- Given that Ariel has 45 fish in total and 2/3 of the fish are male,
    prove that the number of female fish is 15. -/
theorem ariels_female_fish :
  ∀ (total_fish : ℕ) (male_fraction : ℚ),
    total_fish = 45 →
    male_fraction = 2/3 →
    (total_fish : ℚ) * (1 - male_fraction) = 15 :=
by sorry

end NUMINAMATH_CALUDE_ariels_female_fish_l4093_409358


namespace NUMINAMATH_CALUDE_expression_factorization_l4093_409309

theorem expression_factorization (x : ℝ) : 
  (15 * x^3 + 80 * x - 5) - (-4 * x^3 + 4 * x - 5) = 19 * x * (x^2 + 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l4093_409309


namespace NUMINAMATH_CALUDE_divisible_by_six_and_inductive_step_l4093_409392

theorem divisible_by_six_and_inductive_step (n : ℕ) :
  6 ∣ (n * (n + 1) * (2 * n + 1)) ∧
  (∀ k : ℕ, (k + 1) * ((k + 1) + 1) * (2 * (k + 1) + 1) = k * (k + 1) * (2 * k + 1) + 6 * (k + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_six_and_inductive_step_l4093_409392


namespace NUMINAMATH_CALUDE_pages_read_difference_l4093_409375

theorem pages_read_difference (beatrix_pages : ℕ) (cristobal_pages : ℕ) : 
  beatrix_pages = 704 → 
  cristobal_pages = 15 + 3 * beatrix_pages → 
  cristobal_pages - beatrix_pages = 1423 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_difference_l4093_409375


namespace NUMINAMATH_CALUDE_function_periodic_l4093_409324

/-- A function satisfying certain symmetry properties is periodic -/
theorem function_periodic (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 3) = f (3 - x))
  (h2 : ∀ x : ℝ, f (x + 11) = f (11 - x)) :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x :=
sorry

end NUMINAMATH_CALUDE_function_periodic_l4093_409324


namespace NUMINAMATH_CALUDE_solution_set_f_positive_solution_set_f_leq_g_l4093_409305

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 + (4 - m) * x - 6 * m
def g (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - x - m

-- Part 1: Solution set of f(x) > 0 when m = 1
theorem solution_set_f_positive (x : ℝ) :
  f 1 x > 0 ↔ x < -2 ∨ x > 1 := by sorry

-- Part 2: Solution set of f(x) ≤ g(x) when m > 0
theorem solution_set_f_leq_g (m : ℝ) (x : ℝ) (h : m > 0) :
  f m x ≤ g m x ↔ -5 ≤ x ∧ x ≤ m := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_solution_set_f_leq_g_l4093_409305


namespace NUMINAMATH_CALUDE_bowling_team_weight_l4093_409366

/-- Given a bowling team with the following properties:
  * 7 original players
  * Original average weight of 103 kg
  * 2 new players join
  * One new player weighs 60 kg
  * New average weight is 99 kg
  Prove that the other new player weighs 110 kg -/
theorem bowling_team_weight (original_players : ℕ) (original_avg : ℝ) 
  (new_players : ℕ) (known_new_weight : ℝ) (new_avg : ℝ) :
  original_players = 7 ∧ 
  original_avg = 103 ∧ 
  new_players = 2 ∧ 
  known_new_weight = 60 ∧ 
  new_avg = 99 →
  ∃ x : ℝ, x = 110 ∧ 
    (original_players * original_avg + known_new_weight + x) / 
    (original_players + new_players) = new_avg :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_weight_l4093_409366


namespace NUMINAMATH_CALUDE_colin_average_time_l4093_409394

/-- Represents Colin's running times for each mile -/
def colinTimes : List ℕ := [6, 5, 5, 4]

/-- The number of miles Colin ran -/
def totalMiles : ℕ := colinTimes.length

/-- Calculates the average time per mile -/
def averageTime : ℚ := (colinTimes.sum : ℚ) / totalMiles

theorem colin_average_time :
  averageTime = 5 := by sorry

end NUMINAMATH_CALUDE_colin_average_time_l4093_409394


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l4093_409371

theorem simplify_product_of_square_roots (x : ℝ) (h : x > 0) :
  Real.sqrt (5 * 2 * x) * Real.sqrt (x^3 * 5^3) = 25 * x^2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l4093_409371


namespace NUMINAMATH_CALUDE_ellipse_slope_theorem_l4093_409376

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define points A and B as endpoints of minor axis
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (0, 1)

-- Define line l passing through (0,1)
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 1 = k * x

-- Define points C and D on the ellipse and line l
def C (k : ℝ) : ℝ × ℝ := sorry
def D (k : ℝ) : ℝ × ℝ := sorry

-- Define slopes k₁ and k₂
def k₁ (k : ℝ) : ℝ := sorry
def k₂ (k : ℝ) : ℝ := sorry

theorem ellipse_slope_theorem (k : ℝ) :
  (∀ x y, ellipse x y → line_l k x y → (x, y) = C k ∨ (x, y) = D k) →
  k₁ k / k₂ k = 2 →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_slope_theorem_l4093_409376


namespace NUMINAMATH_CALUDE_smallest_third_altitude_nine_is_achievable_l4093_409311

/-- Represents a triangle with altitudes --/
structure TriangleWithAltitudes where
  /-- The lengths of the three altitudes --/
  altitudes : Fin 3 → ℝ
  /-- At least two altitudes are positive --/
  two_positive : ∃ (i j : Fin 3), i ≠ j ∧ altitudes i > 0 ∧ altitudes j > 0

/-- The proposition to be proved --/
theorem smallest_third_altitude 
  (t : TriangleWithAltitudes) 
  (h1 : t.altitudes 0 = 6) 
  (h2 : t.altitudes 1 = 18) 
  (h3 : ∃ (n : ℕ), t.altitudes 2 = n) :
  t.altitudes 2 ≥ 9 := by
sorry

/-- The proposition that 9 is achievable --/
theorem nine_is_achievable : 
  ∃ (t : TriangleWithAltitudes), 
    t.altitudes 0 = 6 ∧ 
    t.altitudes 1 = 18 ∧ 
    t.altitudes 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_smallest_third_altitude_nine_is_achievable_l4093_409311


namespace NUMINAMATH_CALUDE_sum_of_1006th_row_is_20112_l4093_409338

/-- Calculates the sum of numbers in the nth row of the pattern -/
def row_sum (n : ℕ) : ℕ := n * (3 * n - 1) / 2

/-- The theorem states that the sum of numbers in the 1006th row equals 20112 -/
theorem sum_of_1006th_row_is_20112 : row_sum 1006 = 20112 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_1006th_row_is_20112_l4093_409338


namespace NUMINAMATH_CALUDE_expression_equals_one_l4093_409391

theorem expression_equals_one (x : ℝ) 
  (h1 : x^3 + 2*x + 1 ≠ 0) 
  (h2 : x^3 - 2*x - 1 ≠ 0) : 
  ((((x+2)^2 * (x^2-x+2)^2) / (x^3+2*x+1)^2)^3 * 
   (((x-2)^2 * (x^2+x+2)^2) / (x^3-2*x-1)^2)^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l4093_409391


namespace NUMINAMATH_CALUDE_correct_number_increase_l4093_409351

theorem correct_number_increase : 
  ∀ (a b c d : ℕ), 
    (a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9) →
    (a + (b + 1) * c - d = 36) ∧
    (¬(a + 1 + b * c - d = 36)) ∧
    (¬(a + b * (c + 1) - d = 36)) ∧
    (¬(a + b * c - (d + 1) = 36)) :=
by sorry

end NUMINAMATH_CALUDE_correct_number_increase_l4093_409351


namespace NUMINAMATH_CALUDE_difference_not_arithmetic_for_k_ge_4_l4093_409336

/-- Two geometric sequences -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The difference sequence -/
def difference_sequence (a b : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n - b n

/-- Arithmetic sequence with non-zero common difference -/
def is_arithmetic_with_nonzero_diff (c : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, c (n + 1) - c n = d

theorem difference_not_arithmetic_for_k_ge_4 (a b : ℕ → ℝ) (k : ℕ) 
  (h1 : geometric_sequence a)
  (h2 : geometric_sequence b)
  (h3 : k ≥ 4) :
  ¬ is_arithmetic_with_nonzero_diff (difference_sequence a b) :=
sorry

end NUMINAMATH_CALUDE_difference_not_arithmetic_for_k_ge_4_l4093_409336


namespace NUMINAMATH_CALUDE_reverse_square_digits_l4093_409345

theorem reverse_square_digits : ∃! n : ℕ, n > 0 ∧
  (n^2 % 100 = 10 * ((n+1)^2 % 10) + ((n+1)^2 / 10 % 10)) ∧
  ((n+1)^2 % 100 = 10 * (n^2 % 10) + (n^2 / 10 % 10)) :=
sorry

end NUMINAMATH_CALUDE_reverse_square_digits_l4093_409345


namespace NUMINAMATH_CALUDE_sin_45_degrees_l4093_409374

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l4093_409374


namespace NUMINAMATH_CALUDE_circle_center_sum_l4093_409382

/-- Given a circle with equation x^2 + y^2 = 6x + 8y + 15, 
    prove that the sum of the x and y coordinates of its center is 7. -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 6*x + 8*y + 15 → 
  ∃ (h k : ℝ), (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = (x^2 + y^2 - 6*x - 8*y - 15)) ∧ 
               h + k = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l4093_409382


namespace NUMINAMATH_CALUDE_tara_dad_attendance_l4093_409348

/-- The number of games Tara played each year -/
def games_per_year : ℕ := 20

/-- The number of games Tara's dad attended in the second year -/
def games_attended_second_year : ℕ := 14

/-- The difference in games attended between the first and second year -/
def games_difference : ℕ := 4

/-- The percentage of games Tara's dad attended in the first year -/
def attendance_percentage : ℚ := 90

theorem tara_dad_attendance :
  (games_attended_second_year + games_difference) / games_per_year * 100 = attendance_percentage := by
  sorry

end NUMINAMATH_CALUDE_tara_dad_attendance_l4093_409348


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l4093_409318

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l4093_409318


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l4093_409341

/-- Given a rectangular parallelepiped with dimensions l, w, and h, if the shortest distances
    from an interior diagonal to the edges it does not meet are 2√5, 30/√13, and 15/√10,
    then the volume of the parallelepiped is 750. -/
theorem parallelepiped_volume (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) : 
  (l * w / Real.sqrt (l^2 + w^2) = 2 * Real.sqrt 5) →
  (h * w / Real.sqrt (h^2 + w^2) = 30 / Real.sqrt 13) →
  (h * l / Real.sqrt (h^2 + l^2) = 15 / Real.sqrt 10) →
  l * w * h = 750 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l4093_409341


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_negative_four_condition_for_b_subset_complement_a_l4093_409343

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 5}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

-- Theorem for the first part of the problem
theorem intersection_and_union_when_a_is_negative_four :
  (A ∩ B (-4)) = {x | 1/2 ≤ x ∧ x < 2} ∧
  (A ∪ B (-4)) = {x | -2 < x ∧ x ≤ 3} := by sorry

-- Theorem for the second part of the problem
theorem condition_for_b_subset_complement_a (a : ℝ) :
  (B a ∩ Aᶜ = B a) ↔ a ≥ -1/4 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_negative_four_condition_for_b_subset_complement_a_l4093_409343


namespace NUMINAMATH_CALUDE_find_y_value_l4093_409364

theorem find_y_value : ∃ y : ℚ, 
  (1/4 : ℚ) * ((y + 8) + (7*y + 4) + (3*y + 9) + (4*y + 5)) = 6*y - 10 → y = 22/3 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l4093_409364


namespace NUMINAMATH_CALUDE_mary_overtime_rate_increase_l4093_409301

/-- Calculates the percentage increase in overtime rate compared to regular rate -/
def overtime_rate_increase (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℚ) (max_earnings : ℚ) : ℚ :=
  let overtime_hours := max_hours - regular_hours
  let regular_earnings := regular_hours * regular_rate
  let overtime_earnings := max_earnings - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  ((overtime_rate - regular_rate) / regular_rate) * 100

/-- The percentage increase in overtime rate for Mary's work schedule -/
theorem mary_overtime_rate_increase :
  overtime_rate_increase 80 20 8 760 = 25 := by
  sorry

end NUMINAMATH_CALUDE_mary_overtime_rate_increase_l4093_409301


namespace NUMINAMATH_CALUDE_flight_speed_l4093_409383

/-- Given a flight distance of 256 miles and a flight time of 8 hours,
    prove that the speed is 32 miles per hour. -/
theorem flight_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 256) 
    (h2 : time = 8) 
    (h3 : speed = distance / time) : speed = 32 := by
  sorry

end NUMINAMATH_CALUDE_flight_speed_l4093_409383


namespace NUMINAMATH_CALUDE_larry_wins_prob_l4093_409330

/-- The probability of Larry winning the game --/
def larry_wins : ℚ :=
  let larry_prob : ℚ := 3/4  -- Larry's probability of knocking the bottle off
  let julius_prob : ℚ := 1/4 -- Julius's probability of knocking the bottle off
  let max_rounds : ℕ := 5    -- Maximum number of rounds

  -- Probability of Larry winning in the 1st round
  let p1 : ℚ := larry_prob

  -- Probability of Larry winning in the 3rd round
  let p3 : ℚ := (1 - larry_prob) * julius_prob * larry_prob

  -- Probability of Larry winning in the 5th round
  let p5 : ℚ := ((1 - larry_prob) * julius_prob)^2 * larry_prob

  -- Total probability of Larry winning
  p1 + p3 + p5

/-- Theorem stating that the probability of Larry winning is 825/1024 --/
theorem larry_wins_prob : larry_wins = 825/1024 := by
  sorry

end NUMINAMATH_CALUDE_larry_wins_prob_l4093_409330
