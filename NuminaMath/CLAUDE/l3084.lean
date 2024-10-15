import Mathlib

namespace NUMINAMATH_CALUDE_xyz_value_l3084_308484

theorem xyz_value (a b c x y z : ℂ) 
  (eq1 : a = (b + c) / (x - 2))
  (eq2 : b = (c + a) / (y - 2))
  (eq3 : c = (a + b) / (z - 2))
  (sum_prod : x * y + y * z + z * x = 67)
  (sum : x + y + z = 2010) :
  x * y * z = -5892 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3084_308484


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3084_308462

theorem binomial_coefficient_equality (n s : ℕ) (h : s > 0) :
  (n.choose s) = (n * (n - 1).choose (s - 1)) / s :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3084_308462


namespace NUMINAMATH_CALUDE_min_value_of_f_l3084_308448

/-- The function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := (x^2 - 1) * (x^2 + a*x + b)

/-- The theorem stating the minimum value of f(x) -/
theorem min_value_of_f (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (4 - x)) →
  (∃ x₀ : ℝ, ∀ x : ℝ, f a b x₀ ≤ f a b x) ∧
  (∃ x₁ : ℝ, f a b x₁ = -16) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3084_308448


namespace NUMINAMATH_CALUDE_division_remainder_l3084_308487

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 760 →
  divisor = 36 →
  quotient = 21 →
  dividend = divisor * quotient + remainder →
  remainder = 4 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3084_308487


namespace NUMINAMATH_CALUDE_point_translation_l3084_308432

/-- Given a point A(-5, 6) in a Cartesian coordinate system, 
    moving it 5 units right and 6 units up results in point A₁(0, 12) -/
theorem point_translation :
  let A : ℝ × ℝ := (-5, 6)
  let right_shift : ℝ := 5
  let up_shift : ℝ := 6
  let A₁ : ℝ × ℝ := (A.1 + right_shift, A.2 + up_shift)
  A₁ = (0, 12) := by
sorry

end NUMINAMATH_CALUDE_point_translation_l3084_308432


namespace NUMINAMATH_CALUDE_specific_competition_scores_l3084_308406

/-- Represents a mathematics competition with the given scoring rules. -/
structure MathCompetition where
  total_problems : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  unattempted_points : ℤ

/-- Calculates the number of different total scores possible in the competition. -/
def countPossibleScores (comp : MathCompetition) : ℕ := sorry

/-- The specific competition described in the problem. -/
def specificCompetition : MathCompetition :=
  { total_problems := 30
  , correct_points := 4
  , incorrect_points := -1
  , unattempted_points := 0 }

/-- Theorem stating that the number of different total scores in the specific competition is 145. -/
theorem specific_competition_scores :
  countPossibleScores specificCompetition = 145 := by sorry

end NUMINAMATH_CALUDE_specific_competition_scores_l3084_308406


namespace NUMINAMATH_CALUDE_geometric_progression_and_quadratic_vertex_l3084_308453

/-- Given that a, b, c, d are in geometric progression and the vertex of y = x^2 - 2x + 3 is (b, c),
    prove that a + d = 9/2 -/
theorem geometric_progression_and_quadratic_vertex 
  (a b c d : ℝ) 
  (h1 : ∃ (q : ℝ), q ≠ 0 ∧ b = a * q ∧ c = b * q ∧ d = c * q) 
  (h2 : b^2 - 2*b + 3 = c) : 
  a + d = 9/2 := by sorry

end NUMINAMATH_CALUDE_geometric_progression_and_quadratic_vertex_l3084_308453


namespace NUMINAMATH_CALUDE_always_positive_l3084_308493

theorem always_positive (x y : ℝ) : 3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_l3084_308493


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3084_308475

theorem gcd_of_powers_of_two : Nat.gcd (2^1005 - 1) (2^1016 - 1) = 2^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3084_308475


namespace NUMINAMATH_CALUDE_ryan_has_twenty_more_l3084_308490

/-- The number of stickers each person has -/
structure StickerCount where
  karl : ℕ
  ryan : ℕ
  ben : ℕ

/-- The conditions of the sticker problem -/
def StickerProblem (s : StickerCount) : Prop :=
  s.karl = 25 ∧
  s.ryan > s.karl ∧
  s.ben = s.ryan - 10 ∧
  s.karl + s.ryan + s.ben = 105

/-- The theorem stating Ryan has 20 more stickers than Karl -/
theorem ryan_has_twenty_more (s : StickerCount) 
  (h : StickerProblem s) : s.ryan - s.karl = 20 := by
  sorry

#check ryan_has_twenty_more

end NUMINAMATH_CALUDE_ryan_has_twenty_more_l3084_308490


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l3084_308460

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ := by
  -- Define the number of sides in an octagon
  let n : ℕ := 8

  -- Define the sum of interior angles for an n-sided polygon
  let sum_interior_angles : ℝ := 180 * (n - 2)

  -- Define the measure of one interior angle
  let one_angle : ℝ := sum_interior_angles / n

  -- Prove that one_angle equals 135
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l3084_308460


namespace NUMINAMATH_CALUDE_equation_solution_l3084_308499

theorem equation_solution : 
  ∀ x : ℝ, x > 0 → (x^(Real.log x / Real.log 10) = x^5 / 10000 ↔ x = 10 ∨ x = 10000) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3084_308499


namespace NUMINAMATH_CALUDE_moving_circle_theorem_l3084_308471

-- Define the circle and its properties
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_point : center.1^2 + center.2^2 = (center.1 - 1)^2 + center.2^2
  tangent_to_line : (center.1 + 1)^2 = center.1^2 + center.2^2

-- Define the trajectory
def trajectory (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the condition for two points on the trajectory
def trajectory_points_condition (A B : ℝ × ℝ) : Prop :=
  trajectory A ∧ trajectory B ∧ A ≠ B ∧
  (A.2 / A.1) * (B.2 / B.1) = 1

-- The main theorem
theorem moving_circle_theorem (C : MovingCircle) :
  (∀ p : ℝ × ℝ, p = C.center → trajectory p) ∧
  (∀ A B : ℝ × ℝ, trajectory_points_condition A B →
    ∃ k : ℝ, B.2 - A.2 = k * (B.1 - A.1) ∧ A.2 = k * (A.1 + 4)) :=
sorry

end NUMINAMATH_CALUDE_moving_circle_theorem_l3084_308471


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l3084_308441

theorem line_through_parabola_vertex :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  ∀ b : ℝ, b ∈ s ↔ 
    (∃ x y : ℝ, y = 2*x + b ∧ 
               y = x^2 + b^2 - 1 ∧ 
               ∀ x' : ℝ, x'^2 + b^2 - 1 ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l3084_308441


namespace NUMINAMATH_CALUDE_smallest_divisible_by_prime_main_result_l3084_308435

def consecutive_even_product (n : ℕ) : ℕ :=
  Finset.prod (Finset.range (n/2 + 1)) (λ i => 2 * i)

theorem smallest_divisible_by_prime (p : ℕ) (hp : Nat.Prime p) :
  (∀ m : ℕ, m < 2 * p → ¬(p ∣ consecutive_even_product m)) ∧
  (p ∣ consecutive_even_product (2 * p)) :=
sorry

theorem main_result : 
  ∀ n : ℕ, n < 63994 → ¬(31997 ∣ consecutive_even_product n) ∧
  31997 ∣ consecutive_even_product 63994 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_prime_main_result_l3084_308435


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_eleven_l3084_308496

-- Define the equation that x satisfies
def satisfies_equation (x : ℝ) : Prop :=
  x^2 + 4*x + 4/x + 1/x^2 = 34

-- Define the condition that x can be written as a + √b
def can_be_written_as_a_plus_sqrt_b (x : ℝ) : Prop :=
  ∃ (a b : ℕ), x = a + Real.sqrt b ∧ a > 0 ∧ b > 0

-- State the theorem
theorem sum_of_a_and_b_is_eleven :
  ∀ x : ℝ, satisfies_equation x → can_be_written_as_a_plus_sqrt_b x →
  ∃ (a b : ℕ), x = a + Real.sqrt b ∧ a > 0 ∧ b > 0 ∧ a + b = 11 :=
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_eleven_l3084_308496


namespace NUMINAMATH_CALUDE_cold_drink_recipe_l3084_308431

theorem cold_drink_recipe (tea_per_drink : ℚ) (total_mixture : ℚ) (total_lemonade : ℚ)
  (h1 : tea_per_drink = 1/4)
  (h2 : total_mixture = 18)
  (h3 : total_lemonade = 15) :
  (total_lemonade / ((total_mixture - total_lemonade) / tea_per_drink)) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_cold_drink_recipe_l3084_308431


namespace NUMINAMATH_CALUDE_fraction_addition_l3084_308498

theorem fraction_addition (c : ℝ) : (4 + 3 * c) / 7 + 2 = (18 + 3 * c) / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3084_308498


namespace NUMINAMATH_CALUDE_number_equation_solution_l3084_308412

theorem number_equation_solution : 
  ∃ x : ℝ, x - (1002 / 20.04) = 2984 ∧ x = 3034 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3084_308412


namespace NUMINAMATH_CALUDE_min_a_for_increasing_cubic_l3084_308418

/-- Given a function f(x) = x^3 + ax that is increasing on [1, +∞), 
    the minimum value of a is -3. -/
theorem min_a_for_increasing_cubic (a : ℝ) : 
  (∀ x ≥ 1, Monotone (fun x => x^3 + a*x)) → a ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_min_a_for_increasing_cubic_l3084_308418


namespace NUMINAMATH_CALUDE_point_movement_to_origin_l3084_308424

theorem point_movement_to_origin (a b : ℝ) :
  (2 * a - 2 = 0) ∧ (-3 * b - 3 = 0) →
  (2 * a = 2) ∧ (-3 * b = 3) :=
by sorry

end NUMINAMATH_CALUDE_point_movement_to_origin_l3084_308424


namespace NUMINAMATH_CALUDE_salary_solution_l3084_308489

def salary_problem (J F M A May : ℕ) : Prop :=
  (J + F + M + A) / 4 = 8000 ∧
  (F + M + A + May) / 4 = 8700 ∧
  J = 3700 ∧
  May = 6500

theorem salary_solution :
  ∀ J F M A May : ℕ,
    salary_problem J F M A May →
    May = 6500 := by
  sorry

end NUMINAMATH_CALUDE_salary_solution_l3084_308489


namespace NUMINAMATH_CALUDE_max_value_of_f_l3084_308407

def f (x : ℝ) : ℝ := -3 * x^2 + 8

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x₀ : ℝ), f x₀ = M) ∧ M = 8 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3084_308407


namespace NUMINAMATH_CALUDE_abs_plus_a_neg_implies_b_sq_lt_a_sq_but_not_conversely_l3084_308472

theorem abs_plus_a_neg_implies_b_sq_lt_a_sq_but_not_conversely :
  ∃ (a b : ℝ), (abs b + a < 0 → b^2 < a^2) ∧
  ¬(∀ (a b : ℝ), b^2 < a^2 → abs b + a < 0) :=
by sorry

end NUMINAMATH_CALUDE_abs_plus_a_neg_implies_b_sq_lt_a_sq_but_not_conversely_l3084_308472


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3084_308414

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y - 1 = 0 ↔ 2*x - 4*y + 5 = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3084_308414


namespace NUMINAMATH_CALUDE_line_up_arrangement_count_l3084_308476

/-- The number of different arrangements of 5 students (2 boys and 3 girls) where only two girls are adjacent. -/
def arrangement_count : ℕ := 24

/-- The total number of students in the line-up. -/
def total_students : ℕ := 5

/-- The number of boys in the line-up. -/
def num_boys : ℕ := 2

/-- The number of girls in the line-up. -/
def num_girls : ℕ := 3

/-- The number of adjacent girls in each arrangement. -/
def adjacent_girls : ℕ := 2

theorem line_up_arrangement_count :
  arrangement_count = 24 ∧
  total_students = num_boys + num_girls ∧
  num_boys = 2 ∧
  num_girls = 3 ∧
  adjacent_girls = 2 :=
sorry

end NUMINAMATH_CALUDE_line_up_arrangement_count_l3084_308476


namespace NUMINAMATH_CALUDE_no_solution_exists_l3084_308469

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the properties of the function f
def StrictlyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- State the theorem
theorem no_solution_exists (f : ℝ → ℝ) 
  (h1 : StrictlyDecreasing f)
  (h2 : ∀ x ∈ PositiveReals, f x * f (f x + 3 / (2 * x)) = 1/4) :
  ¬ ∃ x ∈ PositiveReals, f x + 3 * x = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3084_308469


namespace NUMINAMATH_CALUDE_parabola_directrix_l3084_308456

/-- 
Given a parabola y² = 2px with intersection point (4, 0), 
prove that its directrix has the equation x = -4 
-/
theorem parabola_directrix (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Equation of the parabola
  (0^2 = 2*p*4) →            -- Intersection point (4, 0)
  (x = -4) →                 -- Equation of the directrix
  True := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3084_308456


namespace NUMINAMATH_CALUDE_number_of_divisors_3960_l3084_308477

theorem number_of_divisors_3960 : Nat.card (Nat.divisors 3960) = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_3960_l3084_308477


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l3084_308455

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (h1 : total_students = 450)
  (h2 : boys = 320)
  (h3 : soccer_players = 250)
  (h4 : (86 : ℚ) / 100 * soccer_players = ↑(boys_playing_soccer))
  (boys_playing_soccer : ℕ) :
  total_students - boys - (soccer_players - boys_playing_soccer) = 95 :=
by sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l3084_308455


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3084_308430

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3084_308430


namespace NUMINAMATH_CALUDE_tabitha_honey_per_cup_l3084_308459

/-- Proves that Tabitha adds 1 serving of honey per cup of tea -/
theorem tabitha_honey_per_cup :
  ∀ (cups_per_night : ℕ) 
    (container_ounces : ℕ) 
    (servings_per_ounce : ℕ) 
    (nights : ℕ),
  cups_per_night = 2 →
  container_ounces = 16 →
  servings_per_ounce = 6 →
  nights = 48 →
  (container_ounces * servings_per_ounce) / (cups_per_night * nights) = 1 :=
by
  sorry

#check tabitha_honey_per_cup

end NUMINAMATH_CALUDE_tabitha_honey_per_cup_l3084_308459


namespace NUMINAMATH_CALUDE_angle_ratio_theorem_l3084_308442

theorem angle_ratio_theorem (α : Real) (m : Real) :
  m < 0 →
  let P : Real × Real := (4 * m, -3 * m)
  (P.1 / (Real.sqrt (P.1^2 + P.2^2)) = -4/5) →
  (P.2 / (Real.sqrt (P.1^2 + P.2^2)) = 3/5) →
  (2 * (P.2 / (Real.sqrt (P.1^2 + P.2^2))) + (P.1 / (Real.sqrt (P.1^2 + P.2^2)))) /
  ((P.2 / (Real.sqrt (P.1^2 + P.2^2))) - (P.1 / (Real.sqrt (P.1^2 + P.2^2)))) = 2/7 := by
sorry

end NUMINAMATH_CALUDE_angle_ratio_theorem_l3084_308442


namespace NUMINAMATH_CALUDE_certain_amount_proof_l3084_308409

/-- The interest rate per annum -/
def interest_rate : ℚ := 8 / 100

/-- The time period for the first amount in years -/
def time1 : ℚ := 25 / 2

/-- The time period for the second amount in years -/
def time2 : ℚ := 4

/-- The first principal amount in Rs -/
def principal1 : ℚ := 160

/-- The second principal amount (the certain amount) in Rs -/
def principal2 : ℚ := 500

/-- Simple interest formula -/
def simple_interest (p r t : ℚ) : ℚ := p * r * t

theorem certain_amount_proof :
  simple_interest principal1 interest_rate time1 = simple_interest principal2 interest_rate time2 :=
sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l3084_308409


namespace NUMINAMATH_CALUDE_bridge_length_l3084_308411

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 ∧ 
  train_speed_kmh = 54 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 320 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3084_308411


namespace NUMINAMATH_CALUDE_equation_solution_l3084_308427

theorem equation_solution : ∃! x : ℝ, 45 - (28 - (37 - (x - 17))) = 56 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3084_308427


namespace NUMINAMATH_CALUDE_value_of_y_l3084_308482

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 24) : y = 120 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l3084_308482


namespace NUMINAMATH_CALUDE_andys_walk_distance_l3084_308492

/-- Proves the distance between Andy's house and the market given his walking routes --/
theorem andys_walk_distance (house_to_school : ℝ) (school_to_park : ℝ) (total_distance : ℝ)
  (h1 : house_to_school = 50)
  (h2 : school_to_park = 25)
  (h3 : total_distance = 345) :
  total_distance - (2 * house_to_school + school_to_park + school_to_park / 2) = 195 := by
  sorry


end NUMINAMATH_CALUDE_andys_walk_distance_l3084_308492


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3084_308422

theorem arithmetic_geometric_mean_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3084_308422


namespace NUMINAMATH_CALUDE_fence_savings_weeks_l3084_308401

theorem fence_savings_weeks (fence_cost : ℕ) (grandparents_gift : ℕ) (aunt_gift : ℕ) (cousin_gift : ℕ) (weekly_earnings : ℕ) :
  fence_cost = 800 →
  grandparents_gift = 120 →
  aunt_gift = 80 →
  cousin_gift = 20 →
  weekly_earnings = 20 →
  ∃ (weeks : ℕ), weeks = 29 ∧ fence_cost = grandparents_gift + aunt_gift + cousin_gift + weeks * weekly_earnings :=
by sorry

end NUMINAMATH_CALUDE_fence_savings_weeks_l3084_308401


namespace NUMINAMATH_CALUDE_sector_arc_length_l3084_308491

/-- Given a sector with central angle 120° and area 300π cm², its arc length is 20π cm. -/
theorem sector_arc_length (θ : ℝ) (S : ℝ) (l : ℝ) : 
  θ = 120 * π / 180 → 
  S = 300 * π → 
  l = 20 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3084_308491


namespace NUMINAMATH_CALUDE_peppers_total_weight_l3084_308449

theorem peppers_total_weight (green : Real) (red : Real) (yellow : Real) (jalapeno : Real) (habanero : Real)
  (h1 : green = 1.45)
  (h2 : red = 0.68)
  (h3 : yellow = 1.6)
  (h4 : jalapeno = 2.25)
  (h5 : habanero = 3.2) :
  green + red + yellow + jalapeno + habanero = 9.18 := by
  sorry

end NUMINAMATH_CALUDE_peppers_total_weight_l3084_308449


namespace NUMINAMATH_CALUDE_total_employees_is_100_l3084_308465

/-- The ratio of employees in groups A, B, and C -/
def group_ratio : Fin 3 → ℕ
  | 0 => 5  -- Group A
  | 1 => 4  -- Group B
  | 2 => 1  -- Group C

/-- The total sample size -/
def sample_size : ℕ := 20

/-- The probability of selecting both person A and person B in group C -/
def prob_select_two : ℚ := 1 / 45

theorem total_employees_is_100 :
  ∀ (total : ℕ),
  (∃ (group_C_size : ℕ),
    /- Group C size is 1/10 of the total -/
    group_C_size = total / 10 ∧
    /- The probability of selecting 2 from group C matches the given probability -/
    (group_C_size.choose 2 : ℚ) / total.choose 2 = prob_select_two ∧
    /- The sample size for group C is 2 -/
    group_C_size * sample_size / total = 2) →
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_total_employees_is_100_l3084_308465


namespace NUMINAMATH_CALUDE_fraction_equality_l3084_308494

theorem fraction_equality : (18 : ℚ) / (0.5 * 106) = 18 / 53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3084_308494


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_3_l3084_308474

theorem smallest_four_digit_mod_8_3 : ∀ n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧ n % 8 = 3 → n ≥ 1003 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_3_l3084_308474


namespace NUMINAMATH_CALUDE_multiply_123_32_125_l3084_308439

theorem multiply_123_32_125 : 123 * 32 * 125 = 492000 := by
  sorry

end NUMINAMATH_CALUDE_multiply_123_32_125_l3084_308439


namespace NUMINAMATH_CALUDE_count_numbers_with_at_most_two_digits_is_2034_l3084_308454

/-- Counts the number of positive integers less than 100000 with at most two different digits. -/
def count_numbers_with_at_most_two_digits : ℕ :=
  let single_digit_count : ℕ := 45
  let two_digits_with_zero_count : ℕ := 117
  let two_digits_without_zero_count : ℕ := 1872
  single_digit_count + two_digits_with_zero_count + two_digits_without_zero_count

/-- The count of positive integers less than 100000 with at most two different digits is 2034. -/
theorem count_numbers_with_at_most_two_digits_is_2034 :
  count_numbers_with_at_most_two_digits = 2034 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_at_most_two_digits_is_2034_l3084_308454


namespace NUMINAMATH_CALUDE_rectangle_equal_diagonals_l3084_308402

-- Define a rectangle
def is_rectangle (A B C D : Point) : Prop := sorry

-- Define equal diagonals
def equal_diagonals (A B C D : Point) : Prop := sorry

-- Theorem statement
theorem rectangle_equal_diagonals (A B C D : Point) :
  is_rectangle A B C D → equal_diagonals A B C D := by sorry

end NUMINAMATH_CALUDE_rectangle_equal_diagonals_l3084_308402


namespace NUMINAMATH_CALUDE_linear_inequality_solution_l3084_308410

theorem linear_inequality_solution (x : ℝ) : 3 * (x + 1) > 9 ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_l3084_308410


namespace NUMINAMATH_CALUDE_daryl_crate_loading_problem_l3084_308425

theorem daryl_crate_loading_problem :
  let crate_capacity : ℕ := 20
  let num_crates : ℕ := 15
  let total_capacity : ℕ := crate_capacity * num_crates
  let num_nail_bags : ℕ := 4
  let nail_bag_weight : ℕ := 5
  let num_hammer_bags : ℕ := 12
  let hammer_bag_weight : ℕ := 5
  let num_plank_bags : ℕ := 10
  let plank_bag_weight : ℕ := 30
  let total_nail_weight : ℕ := num_nail_bags * nail_bag_weight
  let total_hammer_weight : ℕ := num_hammer_bags * hammer_bag_weight
  let total_plank_weight : ℕ := num_plank_bags * plank_bag_weight
  let total_item_weight : ℕ := total_nail_weight + total_hammer_weight + total_plank_weight
  total_item_weight - total_capacity = 80 :=
by sorry

end NUMINAMATH_CALUDE_daryl_crate_loading_problem_l3084_308425


namespace NUMINAMATH_CALUDE_two_distinct_roots_root_three_implies_sum_l3084_308403

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : Prop :=
  x^2 + 2*m*x + m^2 - 2 = 0

-- Part 1: The equation always has two distinct real roots
theorem two_distinct_roots :
  ∀ m : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ :=
sorry

-- Part 2: If 3 is a root, then 2m^2 + 12m + 2043 = 2029
theorem root_three_implies_sum (m : ℝ) :
  quadratic_equation m 3 → 2*m^2 + 12*m + 2043 = 2029 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_root_three_implies_sum_l3084_308403


namespace NUMINAMATH_CALUDE_second_player_wins_l3084_308420

/-- A game played on a circle with 2n + 1 equally spaced points. -/
structure CircleGame where
  n : ℕ
  h : n ≥ 2

/-- A player in the game. -/
inductive Player
  | First
  | Second

/-- A strategy for a player. -/
def Strategy (g : CircleGame) := List (Fin (2 * g.n + 1)) → Fin (2 * g.n + 1)

/-- Predicate to check if all remaining triangles are obtuse. -/
def AllTrianglesObtuse (g : CircleGame) (remaining : List (Fin (2 * g.n + 1))) : Prop :=
  sorry

/-- Predicate to check if a strategy is winning for a player. -/
def IsWinningStrategy (g : CircleGame) (p : Player) (s : Strategy g) : Prop :=
  sorry

/-- Theorem stating that the second player has a winning strategy. -/
theorem second_player_wins (g : CircleGame) :
  ∃ (s : Strategy g), IsWinningStrategy g Player.Second s :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l3084_308420


namespace NUMINAMATH_CALUDE_four_valid_start_days_l3084_308467

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Counts the number of occurrences of a specific weekday in a 30-day month starting from a given day -/
def countWeekday (start : Weekday) (target : Weekday) : Nat :=
  sorry

/-- Checks if Tuesdays and Fridays are equal in number for a given starting day -/
def hasSameTuesdaysAndFridays (start : Weekday) : Bool :=
  countWeekday start Weekday.Tuesday = countWeekday start Weekday.Friday

/-- The set of all weekdays -/
def allWeekdays : List Weekday :=
  [Weekday.Monday, Weekday.Tuesday, Weekday.Wednesday, Weekday.Thursday, 
   Weekday.Friday, Weekday.Saturday, Weekday.Sunday]

/-- The main theorem stating that exactly 4 weekdays satisfy the condition -/
theorem four_valid_start_days :
  (allWeekdays.filter hasSameTuesdaysAndFridays).length = 4 :=
  sorry

end NUMINAMATH_CALUDE_four_valid_start_days_l3084_308467


namespace NUMINAMATH_CALUDE_root_product_theorem_l3084_308451

theorem root_product_theorem (a b m p : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ∃ r, ((a + 1/b)^2 - p*(a + 1/b) + r = 0) ∧ 
       ((b + 1/a)^2 - p*(b + 1/a) + r = 0) → 
  r = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3084_308451


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3084_308446

theorem fraction_to_decimal (n : ℕ) (d : ℕ) (h : d = 5^4 * 2) :
  (n : ℚ) / d = 47 / d → (n : ℚ) / d = 0.0376 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3084_308446


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3084_308404

/-- Represents a digit in a given base --/
def IsDigit (d : ℕ) (base : ℕ) : Prop := d < base

/-- Converts a two-digit number in a given base to base 10 --/
def ToBase10 (d : ℕ) (base : ℕ) : ℕ := base * d + d

/-- The problem statement --/
theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (c d : ℕ),
    IsDigit c 6 ∧
    IsDigit d 8 ∧
    ToBase10 c 6 = n ∧
    ToBase10 d 8 = n ∧
    (∀ (m : ℕ) (c' d' : ℕ),
      IsDigit c' 6 →
      IsDigit d' 8 →
      ToBase10 c' 6 = m →
      ToBase10 d' 8 = m →
      n ≤ m) ∧
    n = 63 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3084_308404


namespace NUMINAMATH_CALUDE_line_equation_through_midpoint_l3084_308438

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point is on the x-axis -/
def Point.onXAxis (p : Point) : Prop :=
  p.y = 0

/-- Check if a point is on the y-axis -/
def Point.onYAxis (p : Point) : Prop :=
  p.x = 0

/-- Check if a point is the midpoint of two other points -/
def Point.isMidpointOf (m p q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

theorem line_equation_through_midpoint (m p q : Point) (l : Line) :
  m = Point.mk 1 (-2) →
  p.onXAxis →
  q.onYAxis →
  m.isMidpointOf p q →
  p.onLine l →
  q.onLine l →
  m.onLine l →
  l = Line.mk 2 (-1) (-4) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_midpoint_l3084_308438


namespace NUMINAMATH_CALUDE_smallest_k_for_difference_property_l3084_308416

theorem smallest_k_for_difference_property (n : ℕ) (hn : n ≥ 1) :
  let k := n^2 + 2
  ∀ (S : Finset ℝ), S.card ≥ k →
    ∃ (x y : ℝ), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧
      (|x - y| < 1 / n ∨ |x - y| > n) ∧
    ∀ (m : ℕ), m < k →
      ∃ (T : Finset ℝ), T.card = m ∧
        ∀ (a b : ℝ), a ∈ T ∧ b ∈ T ∧ a ≠ b →
          |a - b| ≥ 1 / n ∧ |a - b| ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_difference_property_l3084_308416


namespace NUMINAMATH_CALUDE_john_money_left_l3084_308486

/-- Proves that John has $65 left after giving money to his parents -/
theorem john_money_left (initial_amount : ℚ) : 
  initial_amount = 200 →
  initial_amount - (3/8 * initial_amount + 3/10 * initial_amount) = 65 := by
  sorry

end NUMINAMATH_CALUDE_john_money_left_l3084_308486


namespace NUMINAMATH_CALUDE_fashion_show_models_l3084_308445

/-- The number of bathing suit sets each model wears -/
def bathing_suit_sets : ℕ := 2

/-- The number of evening wear sets each model wears -/
def evening_wear_sets : ℕ := 3

/-- The time in minutes for one runway walk -/
def runway_walk_time : ℕ := 2

/-- The total runway time for the show in minutes -/
def total_runway_time : ℕ := 60

/-- The number of models in the fashion show -/
def number_of_models : ℕ := 6

theorem fashion_show_models :
  (bathing_suit_sets + evening_wear_sets) * runway_walk_time * number_of_models = total_runway_time :=
by sorry

end NUMINAMATH_CALUDE_fashion_show_models_l3084_308445


namespace NUMINAMATH_CALUDE_max_a_value_l3084_308400

-- Define the events A and B
def event_A (x y a : ℝ) : Prop := x^2 + y^2 ≤ a ∧ a > 0

def event_B (x y : ℝ) : Prop :=
  x - y + 1 ≥ 0 ∧ 5*x - 2*y - 4 ≤ 0 ∧ 2*x + y + 2 ≥ 0

-- Define the conditional probability P(B|A) = 1
def conditional_probability_is_one (a : ℝ) : Prop :=
  ∀ x y, event_A x y a → event_B x y

-- Theorem statement
theorem max_a_value :
  ∃ a_max : ℝ, a_max = 1/2 ∧
  (∀ a : ℝ, conditional_probability_is_one a → a ≤ a_max) ∧
  conditional_probability_is_one a_max :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l3084_308400


namespace NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l3084_308483

/-- Calculates the interest rate for the first part of an investment given the total amount,
    the amount in the first part, the interest rate for the second part, and the total profit. -/
def calculate_first_interest_rate (total_amount : ℕ) (first_part : ℕ) (second_interest_rate : ℕ) (total_profit : ℕ) : ℚ :=
  let second_part := total_amount - first_part
  let second_part_profit := (second_part * second_interest_rate) / 100
  let first_part_profit := total_profit - second_part_profit
  (first_part_profit * 100) / first_part

theorem first_interest_rate_is_ten_percent :
  calculate_first_interest_rate 80000 70000 20 9000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l3084_308483


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3084_308461

/-- A regular polygon with perimeter 49 and side length 7 has 7 sides. -/
theorem regular_polygon_sides (p : ℕ) (s : ℕ) (h1 : p = 49) (h2 : s = 7) :
  p / s = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3084_308461


namespace NUMINAMATH_CALUDE_adjacent_xue_rong_rong_arrangements_l3084_308413

def num_bing_dung_dung : ℕ := 4
def num_xue_rong_rong : ℕ := 3

def adjacent_arrangements (n_bdd : ℕ) (n_xrr : ℕ) : ℕ :=
  2 * (n_bdd + 2).factorial * (n_bdd + 1)

theorem adjacent_xue_rong_rong_arrangements :
  adjacent_arrangements num_bing_dung_dung num_xue_rong_rong = 960 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_xue_rong_rong_arrangements_l3084_308413


namespace NUMINAMATH_CALUDE_cubic_function_property_l3084_308458

/-- Given a cubic function f(x) = ax³ - bx + 5 where a and b are real numbers,
    if f(-3) = -1, then f(3) = 11. -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^3 - b * x + 5)
    (h2 : f (-3) = -1) : f 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3084_308458


namespace NUMINAMATH_CALUDE_roundness_of_eight_million_l3084_308423

/-- Roundness of a positive integer is the sum of exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 8,000,000 is 15 -/
theorem roundness_of_eight_million :
  roundness 8000000 = 15 := by sorry

end NUMINAMATH_CALUDE_roundness_of_eight_million_l3084_308423


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3084_308470

theorem solve_linear_equation (x : ℚ) (h : x + 2*x + 3*x + 4*x = 5) : x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3084_308470


namespace NUMINAMATH_CALUDE_pole_length_l3084_308440

theorem pole_length (pole_length : ℝ) (gate_height : ℝ) (gate_width : ℝ) : 
  gate_width = 3 →
  pole_length = gate_height + 1 →
  pole_length^2 = gate_height^2 + gate_width^2 →
  pole_length = 5 := by
sorry

end NUMINAMATH_CALUDE_pole_length_l3084_308440


namespace NUMINAMATH_CALUDE_middle_three_sum_is_15_l3084_308478

/-- Represents a card with a color and a number. -/
inductive Card
| green (n : Nat)
| purple (n : Nat)

/-- Checks if a stack of cards satisfies the given conditions. -/
def validStack (stack : List Card) : Prop :=
  let greenCards := [1, 2, 3, 4, 5, 6]
  let purpleCards := [4, 5, 6, 7, 8]
  (∀ c ∈ stack, match c with
    | Card.green n => n ∈ greenCards
    | Card.purple n => n ∈ purpleCards) ∧
  (stack.length = 11) ∧
  (∀ i, i % 2 = 0 → match stack.get? i with
    | some (Card.green _) => True
    | _ => False) ∧
  (∀ i, i % 2 = 1 → match stack.get? i with
    | some (Card.purple _) => True
    | _ => False) ∧
  (∀ i, i + 1 < stack.length →
    match stack.get? i, stack.get? (i + 1) with
    | some (Card.green m), some (Card.purple n) => n % m = 0 ∧ n > m
    | some (Card.purple n), some (Card.green m) => n % m = 0 ∧ n > m
    | _, _ => True)

/-- The sum of the numbers on the middle three cards in a valid stack is 15. -/
theorem middle_three_sum_is_15 (stack : List Card) :
  validStack stack →
  (match stack.get? 4, stack.get? 5, stack.get? 6 with
   | some (Card.purple n1), some (Card.green n2), some (Card.purple n3) =>
     n1 + n2 + n3 = 15
   | _, _, _ => False) :=
by sorry

end NUMINAMATH_CALUDE_middle_three_sum_is_15_l3084_308478


namespace NUMINAMATH_CALUDE_arithmetic_mean_relation_l3084_308434

theorem arithmetic_mean_relation (n : ℕ) (d : ℕ) (h1 : d > 0) :
  let seq := List.range d
  let arithmetic_mean := (n * d + (d * (d - 1)) / 2) / d
  let largest := n + d - 1
  arithmetic_mean = 5 * n →
  largest / arithmetic_mean = 9 / 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_relation_l3084_308434


namespace NUMINAMATH_CALUDE_line_slope_value_l3084_308447

/-- Given a line l passing through points A(3, m+1) and B(4, 2m+1) with slope π/4, prove that m = 1 -/
theorem line_slope_value (m : ℝ) : 
  (∃ l : Set (ℝ × ℝ), 
    (3, m + 1) ∈ l ∧ 
    (4, 2*m + 1) ∈ l ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = Real.pi / 4)) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_value_l3084_308447


namespace NUMINAMATH_CALUDE_highlighter_count_after_increase_l3084_308419

/-- Calculates the final number of highlighters after accounting for broken and borrowed ones, 
    and applying a 25% increase. -/
theorem highlighter_count_after_increase 
  (pink yellow blue green purple : ℕ)
  (broken_pink broken_yellow broken_blue : ℕ)
  (borrowed_green borrowed_purple : ℕ)
  (h1 : pink = 18)
  (h2 : yellow = 14)
  (h3 : blue = 11)
  (h4 : green = 8)
  (h5 : purple = 7)
  (h6 : broken_pink = 3)
  (h7 : broken_yellow = 2)
  (h8 : broken_blue = 1)
  (h9 : borrowed_green = 1)
  (h10 : borrowed_purple = 2) :
  let remaining := (pink - broken_pink) + (yellow - broken_yellow) + (blue - broken_blue) +
                   (green - borrowed_green) + (purple - borrowed_purple)
  let increase := (remaining * 25) / 100
  (remaining + increase) = 61 :=
by sorry

end NUMINAMATH_CALUDE_highlighter_count_after_increase_l3084_308419


namespace NUMINAMATH_CALUDE_base8_calculation_l3084_308495

-- Define a function to convert base 8 to natural numbers
def base8ToNat (x : ℕ) : ℕ := sorry

-- Define a function to convert natural numbers to base 8
def natToBase8 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_calculation : 
  natToBase8 ((base8ToNat 452 - base8ToNat 126) + base8ToNat 237) = 603 := by
  sorry

end NUMINAMATH_CALUDE_base8_calculation_l3084_308495


namespace NUMINAMATH_CALUDE_range_of_a_l3084_308426

/-- Curve C1 in polar coordinates -/
def C1 (ρ θ a : ℝ) : Prop := ρ * (Real.sqrt 2 * Real.cos θ - Real.sin θ) = a

/-- Curve C2 in parametric form -/
def C2 (x y θ : ℝ) : Prop := x = Real.sin θ + Real.cos θ ∧ y = 1 + Real.sin (2 * θ)

/-- C1 in rectangular coordinates -/
def C1_rect (x y a : ℝ) : Prop := Real.sqrt 2 * x - y - a = 0

/-- C2 in rectangular coordinates -/
def C2_rect (x y : ℝ) : Prop := y = x^2 ∧ x ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)

/-- The main theorem -/
theorem range_of_a (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    C1_rect x₁ y₁ a ∧ C2_rect x₁ y₁ ∧
    C1_rect x₂ y₂ a ∧ C2_rect x₂ y₂) ↔
  a ∈ Set.Icc (-1/2) 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3084_308426


namespace NUMINAMATH_CALUDE_lighter_person_weight_l3084_308473

/-- Given two people with a total weight of 88 kg, where one person is 4 kg heavier than the other,
    prove that the weight of the lighter person is 42 kg. -/
theorem lighter_person_weight (total_weight : ℝ) (weight_difference : ℝ) (lighter_weight : ℝ) : 
  total_weight = 88 → weight_difference = 4 → 
  lighter_weight + (lighter_weight + weight_difference) = total_weight →
  lighter_weight = 42 := by
  sorry

#check lighter_person_weight

end NUMINAMATH_CALUDE_lighter_person_weight_l3084_308473


namespace NUMINAMATH_CALUDE_negation_of_p_l3084_308437

-- Define the proposition p
def p : Prop := ∀ a : ℝ, a ≥ 0 → ∃ x : ℝ, x^2 + a*x + 1 = 0

-- State the theorem
theorem negation_of_p : 
  ¬p ↔ ∃ a : ℝ, a ≥ 0 ∧ ¬∃ x : ℝ, x^2 + a*x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_p_l3084_308437


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_real_solution_l3084_308405

/-- The quadratic inequality -/
def quadratic_inequality (k x : ℝ) : Prop := k * x^2 - 2 * x + 6 * k < 0

/-- The solution set for part 1 -/
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2

/-- The solution set for part 2 -/
def solution_set_2 : Set ℝ := Set.univ

theorem quadratic_inequality_solution (k : ℝ) :
  (k ≠ 0 ∧ ∀ x, quadratic_inequality k x ↔ solution_set_1 x) → k = -2/5 :=
sorry

theorem quadratic_inequality_real_solution (k : ℝ) :
  (k ≠ 0 ∧ ∀ x, quadratic_inequality k x ↔ x ∈ solution_set_2) → 
  k < 0 ∧ k < -Real.sqrt 6 / 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_real_solution_l3084_308405


namespace NUMINAMATH_CALUDE_angle_measure_from_coordinates_l3084_308408

/-- Given an acute angle α and a point A on its terminal side with coordinates (2sin 3, -2cos 3),
    prove that α = 3 - π/2 --/
theorem angle_measure_from_coordinates (α : Real) (A : Real × Real) :
  α > 0 ∧ α < π/2 →  -- α is acute
  A.1 = 2 * Real.sin 3 ∧ A.2 = -2 * Real.cos 3 →  -- coordinates of A
  α = 3 - π/2 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_from_coordinates_l3084_308408


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l3084_308429

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [3, 2, 4]
def base1 : Nat := 9

def num2 : List Nat := [2, 1, 5]
def base2 : Nat := 6

-- State the theorem
theorem base_conversion_subtraction :
  (to_base_10 num1 base1) - (to_base_10 num2 base2) = 182 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l3084_308429


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3084_308428

theorem trigonometric_equation_solution (x : ℝ) : 
  2 * (Real.sin x ^ 6 + Real.cos x ^ 6) - 3 * (Real.sin x ^ 4 + Real.cos x ^ 4) = Real.cos (2 * x) →
  ∃ k : ℤ, x = π / 2 * (2 * ↑k + 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3084_308428


namespace NUMINAMATH_CALUDE_range_of_m_for_nonempty_solution_l3084_308444

theorem range_of_m_for_nonempty_solution (m : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) → -5 ≤ m ∧ m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_nonempty_solution_l3084_308444


namespace NUMINAMATH_CALUDE_x_not_greater_than_one_l3084_308443

theorem x_not_greater_than_one (x : ℝ) (h : |x - 1| + x = 1) : x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_x_not_greater_than_one_l3084_308443


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l3084_308497

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l3084_308497


namespace NUMINAMATH_CALUDE_new_students_average_age_l3084_308479

theorem new_students_average_age
  (initial_count : ℕ)
  (initial_avg : ℝ)
  (new_count : ℕ)
  (new_avg_increase : ℝ)
  (h1 : initial_count = 10)
  (h2 : initial_avg = 14)
  (h3 : new_count = 5)
  (h4 : new_avg_increase = 1) :
  let total_initial := initial_count * initial_avg
  let total_new := (initial_count + new_count) * (initial_avg + new_avg_increase)
  let new_students_total := total_new - total_initial
  new_students_total / new_count = 17 := by
sorry

end NUMINAMATH_CALUDE_new_students_average_age_l3084_308479


namespace NUMINAMATH_CALUDE_double_earnings_cars_needed_l3084_308481

/-- Represents the earnings and sales of a car salesman -/
structure CarSalesman where
  baseSalary : ℕ
  commissionPerCar : ℕ
  marchEarnings : ℕ

/-- Calculates the number of cars needed to be sold to reach a target earning -/
def carsNeededForTarget (s : CarSalesman) (targetEarnings : ℕ) : ℕ :=
  ((targetEarnings - s.baseSalary) / s.commissionPerCar : ℕ)

/-- Theorem: A car salesman needs to sell 15 cars in April to double his March earnings -/
theorem double_earnings_cars_needed (s : CarSalesman) 
    (h1 : s.baseSalary = 1000)
    (h2 : s.commissionPerCar = 200)
    (h3 : s.marchEarnings = 2000) : 
  carsNeededForTarget s (2 * s.marchEarnings) = 15 := by
  sorry

#eval carsNeededForTarget ⟨1000, 200, 2000⟩ 4000

end NUMINAMATH_CALUDE_double_earnings_cars_needed_l3084_308481


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3084_308480

theorem complex_equation_solution (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (2 : ℂ) + a * i = (b * i - 1) * i) : 
  (a : ℂ) + b * i = -1 - 2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3084_308480


namespace NUMINAMATH_CALUDE_circle_equation_with_tangent_line_l3084_308457

/-- The equation of a circle with center (a, b) and radius r is (x-a)^2 + (y-b)^2 = r^2 -/
def CircleEquation (a b r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

/-- A line is tangent to a circle if the distance from the center of the circle to the line equals the radius of the circle -/
def IsTangentLine (a b r : ℝ) (m n c : ℝ) : Prop :=
  r = |m*a + n*b + c| / Real.sqrt (m^2 + n^2)

/-- The theorem stating that (x-2)^2 + (y+1)^2 = 8 is the equation of the circle with center (2, -1) tangent to the line x + y = 5 -/
theorem circle_equation_with_tangent_line :
  ∀ x y : ℝ,
  CircleEquation 2 (-1) (Real.sqrt 8) x y ↔ 
  IsTangentLine 2 (-1) (Real.sqrt 8) 1 1 (-5) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_tangent_line_l3084_308457


namespace NUMINAMATH_CALUDE_city_population_change_l3084_308433

theorem city_population_change (n : ℕ) : 
  (0.85 * (n + 1500) : ℚ).floor = n - 50 → n = 8833 := by
  sorry

end NUMINAMATH_CALUDE_city_population_change_l3084_308433


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_for_not_p_sufficient_not_necessary_for_q_l3084_308436

-- Define the sets A and B
def A : Set ℝ := {x | 6 + 5*x - x^2 > 0}
def B (a : ℝ) : Set ℝ := {x | (x - (1-a)) * (x - (1+a)) > 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- Statement 1: A ∩ (ℝ\B) when a = 2
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x : ℝ | -1 < x ∧ x ≤ 3} :=
sorry

-- Statement 2: Range of a where ¬p is sufficient but not necessary for q
theorem range_of_a_for_not_p_sufficient_not_necessary_for_q :
  {a : ℝ | 0 < a ∧ a < 2} =
  {a : ℝ | ∀ x, ¬(p x) → q a x ∧ ∃ y, q a y ∧ p y} :=
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_for_not_p_sufficient_not_necessary_for_q_l3084_308436


namespace NUMINAMATH_CALUDE_emily_candy_duration_l3084_308488

/-- The number of days Emily's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  (neighbors_candy + sister_candy) / daily_consumption

/-- Theorem stating that Emily's candy will last 2 days -/
theorem emily_candy_duration :
  candy_duration 5 13 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_candy_duration_l3084_308488


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3084_308464

theorem negation_of_universal_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, 3 * x - 5 > 0) ↔ (∃ x ∈ S, 3 * x - 5 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3084_308464


namespace NUMINAMATH_CALUDE_parallelogram_area_l3084_308450

theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : θ = π/4) :
  a * b * Real.sin θ = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3084_308450


namespace NUMINAMATH_CALUDE_winter_break_probability_l3084_308466

/-- The probability of getting exactly k successes in n independent trials,
    where each trial has probability p of success. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The number of days in the winter break -/
def num_days : ℕ := 5

/-- The probability of clear weather on each day -/
def prob_clear : ℝ := 0.4

/-- The desired number of clear days -/
def desired_clear_days : ℕ := 2

theorem winter_break_probability :
  binomial_probability num_days desired_clear_days prob_clear = 216 / 625 := by
  sorry

end NUMINAMATH_CALUDE_winter_break_probability_l3084_308466


namespace NUMINAMATH_CALUDE_f_zero_f_odd_f_range_l3084_308415

noncomputable section

variable (f : ℝ → ℝ)

-- Define the properties of f
axiom add_hom : ∀ x y : ℝ, f (x + y) = f x + f y
axiom pos_map_pos : ∀ x : ℝ, x > 0 → f x > 0
axiom f_neg_one : f (-1) = -2

-- Theorem statements
theorem f_zero : f 0 = 0 := by sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem f_range : ∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f x ∈ Set.Icc (-4) 2 := by sorry

end

end NUMINAMATH_CALUDE_f_zero_f_odd_f_range_l3084_308415


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l3084_308468

theorem sum_first_150_remainder (n : Nat) (h : n = 150) : 
  (n * (n + 1) / 2) % 11325 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l3084_308468


namespace NUMINAMATH_CALUDE_solution_to_sqrt_equation_l3084_308485

theorem solution_to_sqrt_equation :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
sorry

end NUMINAMATH_CALUDE_solution_to_sqrt_equation_l3084_308485


namespace NUMINAMATH_CALUDE_faster_train_speed_l3084_308452

/-- Given two trains moving in the same direction, prove that the speed of the faster train is 90 kmph -/
theorem faster_train_speed 
  (speed_difference : ℝ) 
  (faster_train_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : speed_difference = 36) 
  (h2 : faster_train_length = 435) 
  (h3 : crossing_time = 29) : 
  ∃ (faster_speed slower_speed : ℝ), 
    faster_speed - slower_speed = speed_difference ∧ 
    faster_train_length / crossing_time * 3.6 = speed_difference ∧
    faster_speed = 90 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l3084_308452


namespace NUMINAMATH_CALUDE_elimination_tournament_sequences_l3084_308417

def team_size : ℕ := 7

/-- The number of possible sequences in the elimination tournament -/
def elimination_sequences (n : ℕ) : ℕ :=
  2 * (Nat.choose (2 * n - 1) (n - 1))

/-- The theorem stating the number of possible sequences for the given problem -/
theorem elimination_tournament_sequences :
  elimination_sequences team_size = 3432 := by
  sorry

end NUMINAMATH_CALUDE_elimination_tournament_sequences_l3084_308417


namespace NUMINAMATH_CALUDE_phi_equality_l3084_308463

-- Define the set M_φ
def M_phi (φ : ℕ → ℕ) : Set (ℕ → ℤ) :=
  {f | ∀ x, f x > f (φ x)}

-- State the theorem
theorem phi_equality (φ₁ φ₂ : ℕ → ℕ) :
  M_phi φ₁ = M_phi φ₂ → M_phi φ₁ ≠ ∅ → φ₁ = φ₂ := by
  sorry

end NUMINAMATH_CALUDE_phi_equality_l3084_308463


namespace NUMINAMATH_CALUDE_books_from_first_shop_l3084_308421

/-- 
Proves that the number of books bought from the first shop is 65, given:
- Total cost of books from first shop is 1150
- 50 books were bought from the second shop for 920
- The average price per book is 18
-/
theorem books_from_first_shop : 
  ∀ (x : ℕ), 
  (1150 + 920 : ℚ) / (x + 50 : ℚ) = 18 → x = 65 := by
sorry

end NUMINAMATH_CALUDE_books_from_first_shop_l3084_308421
