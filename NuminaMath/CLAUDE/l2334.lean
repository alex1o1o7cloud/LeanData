import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2334_233426

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| > 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2334_233426


namespace NUMINAMATH_CALUDE_matrix_cube_property_l2334_233480

theorem matrix_cube_property (a b c : ℂ) : 
  let M : Matrix (Fin 3) (Fin 3) ℂ := !![a, b, c; b, c, a; c, a, b]
  (M^3 = 1) → (a*b*c = -1) → (a^3 + b^3 + c^3 = 4) := by
sorry

end NUMINAMATH_CALUDE_matrix_cube_property_l2334_233480


namespace NUMINAMATH_CALUDE_sphere_in_cube_volume_l2334_233448

/-- The volume of a sphere inscribed in a cube of edge length 2 -/
theorem sphere_in_cube_volume :
  let cube_edge : ℝ := 2
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (4 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cube_volume_l2334_233448


namespace NUMINAMATH_CALUDE_remainder_theorem_l2334_233487

theorem remainder_theorem (x y u v : ℤ) 
  (x_pos : 0 < x) (y_pos : 0 < y) 
  (division : x = u * y + v) (rem_bound : 0 ≤ v ∧ v < y) : 
  (x + y * u^2 + 3 * v) % y = (4 * v) % y := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2334_233487


namespace NUMINAMATH_CALUDE_book_cost_l2334_233438

theorem book_cost (total_paid : ℕ) (change : ℕ) (pen_cost : ℕ) (ruler_cost : ℕ) 
  (h1 : total_paid = 50)
  (h2 : change = 20)
  (h3 : pen_cost = 4)
  (h4 : ruler_cost = 1) :
  total_paid - change - (pen_cost + ruler_cost) = 25 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l2334_233438


namespace NUMINAMATH_CALUDE_roses_ratio_l2334_233435

/-- Proves that the ratio of roses given to Susan's daughter to the total number of roses in the bouquet is 1:2 -/
theorem roses_ratio (total : ℕ) (vase : ℕ) (daughter : ℕ) : 
  total = 3 * 12 →
  total = vase + daughter →
  vase = 18 →
  12 = (2/3) * vase →
  daughter / total = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_roses_ratio_l2334_233435


namespace NUMINAMATH_CALUDE_translation_result_l2334_233473

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  ⟨p.x + t.dx, p.y + t.dy⟩

theorem translation_result :
  let A : Point := ⟨-3, 2⟩
  let t : Translation := ⟨3, -2⟩
  applyTranslation A t = ⟨0, 0⟩ := by sorry

end NUMINAMATH_CALUDE_translation_result_l2334_233473


namespace NUMINAMATH_CALUDE_line_through_points_l2334_233458

/-- Theorem: Line passing through specific points with given conditions -/
theorem line_through_points (k x y : ℚ) : 
  (k + 4) / 4 = k →  -- slope condition
  x - y = 2 →        -- condition on x and y
  k - x = 3 →        -- condition on k and x
  k = 4/3 ∧ x = -5/3 ∧ y = -11/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2334_233458


namespace NUMINAMATH_CALUDE_jeep_distance_calculation_l2334_233408

theorem jeep_distance_calculation (initial_time : ℝ) (speed : ℝ) (time_factor : ℝ) :
  initial_time = 7 →
  speed = 40 →
  time_factor = 3 / 2 →
  (speed * (time_factor * initial_time)) = 420 :=
by sorry

end NUMINAMATH_CALUDE_jeep_distance_calculation_l2334_233408


namespace NUMINAMATH_CALUDE_range_a_theorem_l2334_233472

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀ + (a - 1) * x₀ + 1 < 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a > 3 ∨ (a ≥ -1 ∧ a ≤ 1)

-- State the theorem
theorem range_a_theorem (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a :=
by sorry

end NUMINAMATH_CALUDE_range_a_theorem_l2334_233472


namespace NUMINAMATH_CALUDE_sum_15_27_in_base4_l2334_233437

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_15_27_in_base4 :
  toBase4 (15 + 27) = [2, 2, 2] :=
sorry

end NUMINAMATH_CALUDE_sum_15_27_in_base4_l2334_233437


namespace NUMINAMATH_CALUDE_range_of_t_when_p_range_of_t_when_p_xor_q_l2334_233410

-- Define the propositions
def p (t : ℝ) : Prop := ∀ x, x^2 + 2*x + 2*t - 4 ≠ 0

def q (t : ℝ) : Prop := 
  t ≠ 4 ∧ t ≠ 2 ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  ∀ x y, x^2 / (4 - t) + y^2 / (t - 2) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

-- Theorem 1
theorem range_of_t_when_p (t : ℝ) : p t → t > 5/2 := by sorry

-- Theorem 2
theorem range_of_t_when_p_xor_q (t : ℝ) : 
  (p t ∨ q t) ∧ ¬(p t ∧ q t) → (2 < t ∧ t ≤ 5/2) ∨ t ≥ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_t_when_p_range_of_t_when_p_xor_q_l2334_233410


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2334_233498

theorem quadratic_equation_solution (a b : ℝ) : 
  (a * 1^2 + b * 1 + 2 = 0) → (2023 - a - b = 2025) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2334_233498


namespace NUMINAMATH_CALUDE_sum_of_imaginary_parts_zero_l2334_233413

theorem sum_of_imaginary_parts_zero (z : ℂ) : 
  (z^2 - 2*z = -1 + Complex.I) → 
  (∃ z₁ z₂ : ℂ, (z₁^2 - 2*z₁ = -1 + Complex.I) ∧ 
                (z₂^2 - 2*z₂ = -1 + Complex.I) ∧ 
                (z₁ ≠ z₂) ∧
                (Complex.im z₁ + Complex.im z₂ = 0)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_imaginary_parts_zero_l2334_233413


namespace NUMINAMATH_CALUDE_max_k_value_l2334_233414

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 4 = k^2 * (x^2/y^2 + 2 + y^2/x^2) + k^3 * (x/y + y/x)) :
  k ≤ 4 * Real.sqrt 2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l2334_233414


namespace NUMINAMATH_CALUDE_profit_per_meter_cloth_l2334_233467

theorem profit_per_meter_cloth (meters_sold : ℕ) (selling_price : ℕ) (cost_price_per_meter : ℕ) 
  (h1 : meters_sold = 66)
  (h2 : selling_price = 660)
  (h3 : cost_price_per_meter = 5) :
  (selling_price - meters_sold * cost_price_per_meter) / meters_sold = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_per_meter_cloth_l2334_233467


namespace NUMINAMATH_CALUDE_basketball_win_calculation_l2334_233482

/-- Proves the number of games a basketball team needs to win to achieve a specific win percentage -/
theorem basketball_win_calculation (total_games : ℕ) (first_games : ℕ) (first_wins : ℕ) (remaining_games : ℕ) 
  (target_percentage : ℚ) (h1 : total_games = first_games + remaining_games) 
  (h2 : total_games = 100) (h3 : first_games = 45) (h4 : first_wins = 30) 
  (h5 : remaining_games = 55) (h6 : target_percentage = 65 / 100) : 
  ∃ (x : ℕ), (first_wins + x : ℚ) / total_games = target_percentage ∧ x = 35 := by
sorry

end NUMINAMATH_CALUDE_basketball_win_calculation_l2334_233482


namespace NUMINAMATH_CALUDE_fraction_not_simplifiable_l2334_233444

theorem fraction_not_simplifiable (n : ℕ) : ¬ ∃ (d : ℤ), d > 1 ∧ d ∣ (21 * n + 4) ∧ d ∣ (14 * n + 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_simplifiable_l2334_233444


namespace NUMINAMATH_CALUDE_calculation_proof_l2334_233481

theorem calculation_proof : 17 * (17/18) + 35 * (35/36) = 50 + 1/12 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2334_233481


namespace NUMINAMATH_CALUDE_bucket_calculation_reduced_capacity_buckets_l2334_233457

/-- Given a tank that requires a certain number of buckets to fill and a reduction in bucket capacity,
    calculate the new number of buckets required to fill the tank. -/
theorem bucket_calculation (original_buckets : ℕ) (capacity_reduction : ℚ) : 
  original_buckets / capacity_reduction = original_buckets * (1 / capacity_reduction) :=
by sorry

/-- Prove that 105 buckets are required when the original number of buckets is 42
    and the capacity is reduced to two-fifths. -/
theorem reduced_capacity_buckets : 
  let original_buckets : ℕ := 42
  let capacity_reduction : ℚ := 2 / 5
  original_buckets / capacity_reduction = 105 :=
by sorry

end NUMINAMATH_CALUDE_bucket_calculation_reduced_capacity_buckets_l2334_233457


namespace NUMINAMATH_CALUDE_negation_equivalence_l2334_233405

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ + 1) * Real.exp x₀ > 1) ↔
  (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2334_233405


namespace NUMINAMATH_CALUDE_triangle_area_problem_l2334_233495

/-- Given a triangle ABC with area 15 and a point D on AB such that AD:DB = 3:2,
    if there exist points E on BC and F on CA forming triangle ABE and quadrilateral DBEF
    with equal areas, then the area of triangle ABE is 9. -/
theorem triangle_area_problem (A B C D E F : ℝ × ℝ) : 
  let triangle_area (P Q R : ℝ × ℝ) := abs ((P.1 - R.1) * (Q.2 - R.2) - (Q.1 - R.1) * (P.2 - R.2)) / 2
  triangle_area A B C = 15 →
  D.1 = (3 * B.1 + 2 * A.1) / 5 ∧ D.2 = (3 * B.2 + 2 * A.2) / 5 →
  E.1 = B.1 ∧ E.2 ≤ B.2 ∧ E.2 ≥ C.2 →
  F.1 ≥ C.1 ∧ F.1 ≤ A.1 ∧ F.2 = C.2 →
  triangle_area A B E = triangle_area D B E + triangle_area D E F →
  triangle_area A B E = 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l2334_233495


namespace NUMINAMATH_CALUDE_min_value_trig_function_l2334_233436

theorem min_value_trig_function (x : ℝ) : 
  Real.sin x ^ 4 + Real.cos x ^ 4 + (1 / Real.cos x) ^ 4 + (1 / Real.sin x) ^ 4 ≥ 17 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_function_l2334_233436


namespace NUMINAMATH_CALUDE_aunt_gift_amount_l2334_233461

def birthday_money_problem (grandmother_gift aunt_gift uncle_gift total_money game_cost games_bought remaining_money : ℕ) : Prop :=
  grandmother_gift = 20 ∧
  uncle_gift = 30 ∧
  total_money = 125 ∧
  game_cost = 35 ∧
  games_bought = 3 ∧
  remaining_money = 20 ∧
  total_money = grandmother_gift + aunt_gift + uncle_gift ∧
  total_money = game_cost * games_bought + remaining_money

theorem aunt_gift_amount :
  ∀ (grandmother_gift aunt_gift uncle_gift total_money game_cost games_bought remaining_money : ℕ),
    birthday_money_problem grandmother_gift aunt_gift uncle_gift total_money game_cost games_bought remaining_money →
    aunt_gift = 75 := by
  sorry

end NUMINAMATH_CALUDE_aunt_gift_amount_l2334_233461


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_special_terms_l2334_233452

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_with_special_terms :
  ∃ (a : ℕ → ℤ) (d : ℤ),
    is_arithmetic_sequence a d ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    a 9 = (a 2) ^ 3 ∧
    (∃ n, a n = (a 2) ^ 2) ∧
    (∃ m, a m = (a 2) ^ 4) →
    a 1 = -24 ∧ a 2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_special_terms_l2334_233452


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2334_233422

def A : Set ℝ := {1, 2, 3, 4}

def B : Set ℝ := {x : ℝ | ∃ y ∈ A, y = 2 * x}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2334_233422


namespace NUMINAMATH_CALUDE_solve_for_s_l2334_233493

theorem solve_for_s (m : ℝ) (s : ℝ) 
  (h1 : 5 = m * (3 ^ s)) 
  (h2 : 45 = m * (9 ^ s)) : 
  s = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_s_l2334_233493


namespace NUMINAMATH_CALUDE_line_parameterization_l2334_233419

/-- Given a line y = 2x - 30 parameterized by (x, y) = (f(t), 20t - 10), prove that f(t) = 10t + 10 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 20 * t - 10 = 2 * f t - 30) → 
  (∀ t : ℝ, f t = 10 * t + 10) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l2334_233419


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l2334_233439

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x

theorem f_derivative_at_one :
  deriv f 1 = 2 * Real.log 2 + 1 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l2334_233439


namespace NUMINAMATH_CALUDE_total_sums_attempted_l2334_233447

theorem total_sums_attempted (correct : ℕ) (wrong : ℕ) : 
  correct = 25 →
  wrong = 2 * correct →
  correct + wrong = 75 := by
sorry

end NUMINAMATH_CALUDE_total_sums_attempted_l2334_233447


namespace NUMINAMATH_CALUDE_bus_walk_distance_difference_l2334_233474

/-- Craig's route home from school -/
structure Route where
  busA : ℝ
  walk1 : ℝ
  busB : ℝ
  walk2 : ℝ
  busC : ℝ
  walk3 : ℝ

/-- Calculate the total bus distance -/
def totalBusDistance (r : Route) : ℝ :=
  r.busA + r.busB + r.busC

/-- Calculate the total walking distance -/
def totalWalkDistance (r : Route) : ℝ :=
  r.walk1 + r.walk2 + r.walk3

/-- Craig's actual route -/
def craigsRoute : Route :=
  { busA := 1.25
  , walk1 := 0.35
  , busB := 2.68
  , walk2 := 0.47
  , busC := 3.27
  , walk3 := 0.21 }

/-- Theorem: The difference between total bus distance and total walking distance is 6.17 miles -/
theorem bus_walk_distance_difference :
  totalBusDistance craigsRoute - totalWalkDistance craigsRoute = 6.17 := by
  sorry

end NUMINAMATH_CALUDE_bus_walk_distance_difference_l2334_233474


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l2334_233441

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x > 0) (hn : n > 1) :
  (1 + x)^n > 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l2334_233441


namespace NUMINAMATH_CALUDE_total_stickers_l2334_233431

/-- Given the following conditions:
    - There are 10 stickers on a page originally
    - There are 22 pages of stickers
    - 3 stickers are missing from each page
    Prove that the total number of stickers is 154 -/
theorem total_stickers (original_stickers : ℕ) (pages : ℕ) (missing_stickers : ℕ)
  (h1 : original_stickers = 10)
  (h2 : pages = 22)
  (h3 : missing_stickers = 3) :
  (original_stickers - missing_stickers) * pages = 154 :=
by sorry

end NUMINAMATH_CALUDE_total_stickers_l2334_233431


namespace NUMINAMATH_CALUDE_eunji_gymnastics_count_l2334_233488

/-- Represents the position of a student in a rectangular arrangement -/
structure StudentPosition where
  leftColumn : Nat
  rightColumn : Nat
  frontRow : Nat
  backRow : Nat

/-- Calculates the total number of students in a rectangular arrangement -/
def totalStudents (pos : StudentPosition) : Nat :=
  let totalColumns := pos.leftColumn + pos.rightColumn - 1
  let totalRows := pos.frontRow + pos.backRow - 1
  totalColumns * totalRows

/-- Theorem: Given Eunji's position, the total number of students is 441 -/
theorem eunji_gymnastics_count :
  let eunjiPosition : StudentPosition := {
    leftColumn := 8,
    rightColumn := 14,
    frontRow := 7,
    backRow := 15
  }
  totalStudents eunjiPosition = 441 := by
  sorry

end NUMINAMATH_CALUDE_eunji_gymnastics_count_l2334_233488


namespace NUMINAMATH_CALUDE_holiday_duration_l2334_233433

theorem holiday_duration (total_rain_days : ℕ) (sunny_mornings : ℕ) (sunny_afternoons : ℕ)
  (h1 : total_rain_days = 7)
  (h2 : sunny_mornings = 5)
  (h3 : sunny_afternoons = 6) :
  ∃ (total_days : ℕ), total_days = 9 ∧ total_days ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_holiday_duration_l2334_233433


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2334_233404

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) :
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2334_233404


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2334_233499

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2334_233499


namespace NUMINAMATH_CALUDE_last_episode_length_correct_l2334_233445

/-- Represents the duration of a TV series viewing session -/
structure SeriesViewing where
  episodeLengths : List Nat
  breakLength : Nat
  totalTime : Nat

/-- Calculates the length of the last episode given the viewing details -/
def lastEpisodeLength (s : SeriesViewing) : Nat :=
  s.totalTime
    - (s.episodeLengths.sum + s.breakLength * s.episodeLengths.length)

theorem last_episode_length_correct (s : SeriesViewing) :
  s.episodeLengths = [58, 62, 65, 71, 79] ∧
  s.breakLength = 12 ∧
  s.totalTime = 9 * 60 →
  lastEpisodeLength s = 145 := by
  sorry

#eval lastEpisodeLength {
  episodeLengths := [58, 62, 65, 71, 79],
  breakLength := 12,
  totalTime := 9 * 60
}

end NUMINAMATH_CALUDE_last_episode_length_correct_l2334_233445


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l2334_233462

/-- Theorem: Given a car's speed of 145 km/h in the first hour and an average speed of 102.5 km/h over two hours, the speed in the second hour is 60 km/h. -/
theorem car_speed_second_hour (speed_first_hour : ℝ) (average_speed : ℝ) (speed_second_hour : ℝ) :
  speed_first_hour = 145 →
  average_speed = 102.5 →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_second_hour = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l2334_233462


namespace NUMINAMATH_CALUDE_equationA_is_linear_l2334_233456

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and at least one of a or b is non-zero. --/
def IsLinearInTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation x/2 + 3y = 2 --/
def EquationA (x y : ℝ) : Prop := x/2 + 3*y = 2

theorem equationA_is_linear : IsLinearInTwoVariables EquationA := by
  sorry


end NUMINAMATH_CALUDE_equationA_is_linear_l2334_233456


namespace NUMINAMATH_CALUDE_smallest_chocolate_beverage_volume_l2334_233454

/-- Represents the ratio of milk to syrup in the chocolate beverage -/
def milk_syrup_ratio : ℚ := 5 / 2

/-- Volume of milk in each bottle (in liters) -/
def milk_bottle_volume : ℚ := 2

/-- Volume of syrup in each bottle (in liters) -/
def syrup_bottle_volume : ℚ := 14 / 10

/-- Finds the smallest number of whole bottles of milk and syrup that satisfy the ratio -/
def find_smallest_bottles : ℕ × ℕ := (7, 4)

/-- Calculates the total volume of the chocolate beverage -/
def total_volume (bottles : ℕ × ℕ) : ℚ :=
  milk_bottle_volume * bottles.1 + syrup_bottle_volume * bottles.2

/-- Theorem stating that the smallest volume of chocolate beverage that can be made
    using only whole bottles of milk and syrup is 19.6 L -/
theorem smallest_chocolate_beverage_volume :
  total_volume (find_smallest_bottles) = 196 / 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_chocolate_beverage_volume_l2334_233454


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2334_233417

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 3 ∧ 
  (∀ (x y z : ℝ), (x + y + z)^2 ≤ n * (x^2 + y^2 + z^2)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), (x + y + z)^2 > m * (x^2 + y^2 + z^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2334_233417


namespace NUMINAMATH_CALUDE_expression_evaluation_l2334_233402

theorem expression_evaluation (x : ℝ) (h : x < 2) :
  Real.sqrt ((x - 2) / (1 - (x - 3) / (x - 2))) = (2 - x) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2334_233402


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2334_233479

theorem consecutive_odd_numbers_sum (a b c d : ℕ) : 
  (∃ x : ℕ, a = 2*x + 1 ∧ b = 2*x + 3 ∧ c = 2*x + 5 ∧ d = 2*x + 7) →  -- Consecutive odd numbers
  a + b + c + d = 112 →  -- Sum is 112
  b = 27  -- Second smallest is 27
:= by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2334_233479


namespace NUMINAMATH_CALUDE_square_difference_l2334_233416

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2334_233416


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l2334_233425

def seven_digit_number (n : ℕ) : ℕ := 7010000 + n * 1000 + 864

theorem divisibility_by_eleven (n : ℕ) :
  (seven_digit_number n) % 11 = 0 ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l2334_233425


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l2334_233423

theorem regular_polygon_interior_angle_sum :
  ∀ n : ℕ,
  n > 2 →
  (360 : ℝ) / n = 20 →
  (n - 2) * 180 = 2880 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l2334_233423


namespace NUMINAMATH_CALUDE_first_number_is_seven_l2334_233442

/-- A sequence of 8 numbers where each number starting from the third
    is the sum of the two previous numbers. -/
def FibonacciLikeSequence (a : Fin 8 → ℕ) : Prop :=
  ∀ i : Fin 8, i.val ≥ 2 → a i = a (Fin.sub i 1) + a (Fin.sub i 2)

/-- Theorem stating that if the 5th number is 53 and the 8th number is 225
    in a Fibonacci-like sequence of 8 numbers, then the 1st number is 7. -/
theorem first_number_is_seven
  (a : Fin 8 → ℕ)
  (h_seq : FibonacciLikeSequence a)
  (h_fifth : a 4 = 53)
  (h_eighth : a 7 = 225) :
  a 0 = 7 := by
  sorry


end NUMINAMATH_CALUDE_first_number_is_seven_l2334_233442


namespace NUMINAMATH_CALUDE_specific_solid_surface_area_l2334_233440

/-- A solid with specific dimensions -/
structure Solid where
  front_length : ℝ
  front_width : ℝ
  left_length : ℝ
  left_width : ℝ
  top_radius : ℝ

/-- The surface area of the solid -/
def surface_area (s : Solid) : ℝ := sorry

/-- Theorem stating the surface area of the specific solid -/
theorem specific_solid_surface_area :
  ∀ s : Solid,
    s.front_length = 4 ∧
    s.front_width = 2 ∧
    s.left_length = 4 ∧
    s.left_width = 2 ∧
    s.top_radius = 2 →
    surface_area s = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_specific_solid_surface_area_l2334_233440


namespace NUMINAMATH_CALUDE_tony_puzzle_time_l2334_233420

/-- Calculates the total time spent solving puzzles given the time for a warm-up puzzle
    and the number and relative duration of additional puzzles. -/
def total_puzzle_time (warm_up_time : ℕ) (num_additional_puzzles : ℕ) (additional_puzzle_factor : ℕ) : ℕ :=
  warm_up_time + num_additional_puzzles * (warm_up_time * additional_puzzle_factor)

/-- Proves that given the specific conditions of Tony's puzzle-solving session,
    the total time spent is 70 minutes. -/
theorem tony_puzzle_time :
  total_puzzle_time 10 2 3 = 70 := by
  sorry

end NUMINAMATH_CALUDE_tony_puzzle_time_l2334_233420


namespace NUMINAMATH_CALUDE_pet_store_spiders_l2334_233427

theorem pet_store_spiders (initial_birds initial_puppies initial_cats : ℕ)
  (initial_spiders : ℕ) (sold_birds adopted_puppies loose_spiders : ℕ)
  (total_left : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_cats = 5 →
  sold_birds = initial_birds / 2 →
  adopted_puppies = 3 →
  loose_spiders = 7 →
  total_left = 25 →
  total_left = (initial_birds - sold_birds) + (initial_puppies - adopted_puppies) +
               initial_cats + (initial_spiders - loose_spiders) →
  initial_spiders = 15 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_spiders_l2334_233427


namespace NUMINAMATH_CALUDE_triangle_trig_max_l2334_233415

open Real

theorem triangle_trig_max (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  π/4 < B ∧ B < π/2 ∧
  a * cos B - b * cos A = (3/5) * c →
  ∃ (max_val : ℝ), max_val = -512 ∧ 
    ∀ x, x = tan (2*B) * (tan A)^3 → x ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_triangle_trig_max_l2334_233415


namespace NUMINAMATH_CALUDE_fuchsia_to_mauve_amount_correct_l2334_233459

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

end NUMINAMATH_CALUDE_fuchsia_to_mauve_amount_correct_l2334_233459


namespace NUMINAMATH_CALUDE_product_value_l2334_233486

def product_term (n : ℕ) : ℚ :=
  (n * (n + 2)) / ((n + 1) * (n + 1))

def product_sequence : ℕ → ℚ
  | 0 => 1
  | n + 1 => product_sequence n * product_term (n + 1)

theorem product_value : product_sequence 98 = 50 / 99 := by
  sorry

end NUMINAMATH_CALUDE_product_value_l2334_233486


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2334_233451

/-- Three points in 3D space are collinear if they all lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t s : ℝ), p2 = p1 + t • (p3 - p1) ∧ p3 = p1 + s • (p3 - p1)

/-- If the points (2,a,b), (a,3,b), and (a,b,4) are collinear, then a + b = 6. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

#check collinear_points_sum

end NUMINAMATH_CALUDE_collinear_points_sum_l2334_233451


namespace NUMINAMATH_CALUDE_two_cubic_feet_to_cubic_inches_l2334_233406

/-- Converts cubic feet to cubic inches -/
def cubic_feet_to_cubic_inches (cf : ℝ) : ℝ := cf * (12^3)

/-- Theorem stating that 2 cubic feet equals 3456 cubic inches -/
theorem two_cubic_feet_to_cubic_inches : 
  cubic_feet_to_cubic_inches 2 = 3456 := by
  sorry

end NUMINAMATH_CALUDE_two_cubic_feet_to_cubic_inches_l2334_233406


namespace NUMINAMATH_CALUDE_calories_in_one_bar_l2334_233476

/-- The number of calories in 11 candy bars -/
def total_calories : ℕ := 341

/-- The number of candy bars -/
def num_bars : ℕ := 11

/-- The number of calories in one candy bar -/
def calories_per_bar : ℕ := total_calories / num_bars

theorem calories_in_one_bar : calories_per_bar = 31 := by
  sorry

end NUMINAMATH_CALUDE_calories_in_one_bar_l2334_233476


namespace NUMINAMATH_CALUDE_arrangements_count_l2334_233492

/-- The number of students in the row -/
def n : ℕ := 7

/-- The number of positions where A and B can be placed with one person in between -/
def positions : ℕ := 5

/-- The number of ways to arrange the remaining students -/
def remaining_arrangements : ℕ := Nat.factorial (n - 3)

/-- The number of ways A and B can switch places -/
def ab_switch : ℕ := 2

/-- The total number of arrangements for 7 students standing in a row,
    where there must be one person standing between students A and B -/
def total_arrangements : ℕ := positions * remaining_arrangements * ab_switch

theorem arrangements_count : total_arrangements = 1200 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l2334_233492


namespace NUMINAMATH_CALUDE_cars_meet_time_l2334_233475

/-- Two cars meet on a highway -/
theorem cars_meet_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 333)
  (h2 : speed1 = 54) (h3 : speed2 = 57) :
  (highway_length / (speed1 + speed2) : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cars_meet_time_l2334_233475


namespace NUMINAMATH_CALUDE_genetically_modified_microorganisms_percentage_l2334_233412

/-- Represents the budget allocation for Megatech Corporation --/
structure BudgetAllocation where
  microphotonics : ℝ
  homeElectronics : ℝ
  foodAdditives : ℝ
  industrialLubricants : ℝ
  basicAstrophysicsDegrees : ℝ

/-- Theorem stating the percentage allocated to genetically modified microorganisms --/
theorem genetically_modified_microorganisms_percentage 
  (budget : BudgetAllocation)
  (h1 : budget.microphotonics = 13)
  (h2 : budget.homeElectronics = 24)
  (h3 : budget.foodAdditives = 15)
  (h4 : budget.industrialLubricants = 8)
  (h5 : budget.basicAstrophysicsDegrees = 39.6) :
  100 - (budget.microphotonics + budget.homeElectronics + budget.foodAdditives + 
         budget.industrialLubricants + (budget.basicAstrophysicsDegrees / 360 * 100)) = 29 := by
  sorry

end NUMINAMATH_CALUDE_genetically_modified_microorganisms_percentage_l2334_233412


namespace NUMINAMATH_CALUDE_older_females_count_l2334_233429

/-- Represents the population of a town divided into equal groups -/
structure TownPopulation where
  total : ℕ
  num_groups : ℕ
  h_positive : 0 < num_groups

/-- Calculates the size of each group in the town -/
def group_size (town : TownPopulation) : ℕ :=
  town.total / town.num_groups

/-- Theorem: In a town with 1000 people divided into 5 equal groups,
    the number of people in each group is 200 -/
theorem older_females_count (town : TownPopulation)
    (h_total : town.total = 1000)
    (h_groups : town.num_groups = 5) :
    group_size town = 200 := by
  sorry

#eval group_size ⟨1000, 5, by norm_num⟩

end NUMINAMATH_CALUDE_older_females_count_l2334_233429


namespace NUMINAMATH_CALUDE_base6_divisibility_l2334_233471

/-- Converts a base 6 number to decimal --/
def base6ToDecimal (a b c d : ℕ) : ℕ :=
  a * 6^3 + b * 6^2 + c * 6 + d

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 19 * k

theorem base6_divisibility :
  isDivisibleBy19 (base6ToDecimal 4 5 5 2) ∧
  ∀ x : ℕ, x < 5 → ¬isDivisibleBy19 (base6ToDecimal 4 5 x 2) :=
by sorry

end NUMINAMATH_CALUDE_base6_divisibility_l2334_233471


namespace NUMINAMATH_CALUDE_quiz_winning_probability_l2334_233443

-- Define the quiz parameters
def num_questions : ℕ := 4
def num_choices : ℕ := 3
def min_correct : ℕ := 3

-- Define the probability of guessing one question correctly
def prob_correct : ℚ := 1 / num_choices

-- Define the probability of guessing one question incorrectly
def prob_incorrect : ℚ := 1 - prob_correct

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the probability of winning
def prob_winning : ℚ :=
  (binomial num_questions num_questions) * (prob_correct ^ num_questions) +
  (binomial num_questions min_correct) * (prob_correct ^ min_correct) * (prob_incorrect ^ (num_questions - min_correct))

-- Theorem statement
theorem quiz_winning_probability :
  prob_winning = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_quiz_winning_probability_l2334_233443


namespace NUMINAMATH_CALUDE_fraction_change_l2334_233450

theorem fraction_change (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (2*x + 2*y) / (2*x * 2*y) = (1/2) * ((x + y) / (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_change_l2334_233450


namespace NUMINAMATH_CALUDE_arrangement_theorems_l2334_233424

/-- The number of men in the group -/
def num_men : ℕ := 6

/-- The number of women in the group -/
def num_women : ℕ := 4

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- Calculate the number of arrangements with no two women next to each other -/
def arrangements_no_adjacent_women : ℕ := sorry

/-- Calculate the number of arrangements with Man A not first and Man B not last -/
def arrangements_a_not_first_b_not_last : ℕ := sorry

/-- Calculate the number of arrangements with fixed order of Men A, B, and C -/
def arrangements_fixed_abc : ℕ := sorry

/-- Calculate the number of arrangements with Man A to the left of Man B -/
def arrangements_a_left_of_b : ℕ := sorry

theorem arrangement_theorems :
  (arrangements_no_adjacent_women = num_men.factorial * (num_women.choose (num_men + 1))) ∧
  (arrangements_a_not_first_b_not_last = total_people.factorial - 2 * (total_people - 1).factorial + (total_people - 2).factorial) ∧
  (arrangements_fixed_abc = total_people.factorial / 6) ∧
  (arrangements_a_left_of_b = total_people.factorial / 2) := by sorry

end NUMINAMATH_CALUDE_arrangement_theorems_l2334_233424


namespace NUMINAMATH_CALUDE_time_after_elapsed_hours_l2334_233468

def hours_elapsed : ℕ := 2023
def starting_time : ℕ := 3
def clock_hours : ℕ := 12

theorem time_after_elapsed_hours :
  (starting_time + hours_elapsed) % clock_hours = 10 :=
by sorry

end NUMINAMATH_CALUDE_time_after_elapsed_hours_l2334_233468


namespace NUMINAMATH_CALUDE_notebook_marker_cost_l2334_233403

theorem notebook_marker_cost (notebook_cost marker_cost : ℝ) 
  (h1 : 3 * notebook_cost + 2 * marker_cost = 7.20)
  (h2 : 2 * notebook_cost + 3 * marker_cost = 6.90) :
  notebook_cost + marker_cost = 2.82 := by
  sorry

end NUMINAMATH_CALUDE_notebook_marker_cost_l2334_233403


namespace NUMINAMATH_CALUDE_potato_bag_weight_l2334_233432

theorem potato_bag_weight (original_weight : ℝ) : 
  (original_weight / (original_weight / 2) = 36) → original_weight = 648 :=
by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l2334_233432


namespace NUMINAMATH_CALUDE_fraction_less_than_one_l2334_233485

theorem fraction_less_than_one (a b : ℝ) (h1 : a > b) (h2 : b > 0) : b / a < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_one_l2334_233485


namespace NUMINAMATH_CALUDE_beach_probability_l2334_233411

theorem beach_probability (total : ℕ) (sunglasses : ℕ) (caps : ℕ) (cap_and_sunglasses_prob : ℚ) :
  total = 100 →
  sunglasses = 70 →
  caps = 60 →
  cap_and_sunglasses_prob = 2/3 →
  (cap_and_sunglasses_prob * caps : ℚ) / sunglasses = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_beach_probability_l2334_233411


namespace NUMINAMATH_CALUDE_salad_vegetables_count_l2334_233469

theorem salad_vegetables_count :
  ∀ (cucumbers tomatoes total : ℕ),
  cucumbers = 70 →
  tomatoes = 3 * cucumbers →
  total = cucumbers + tomatoes →
  total = 280 :=
by
  sorry

end NUMINAMATH_CALUDE_salad_vegetables_count_l2334_233469


namespace NUMINAMATH_CALUDE_shooting_probabilities_l2334_233478

-- Define the probabilities
def prob_A : ℚ := 1/2
def prob_B : ℚ := 1/3

-- Define the event of hitting the target exactly twice
def hit_twice : ℚ := prob_A * prob_B

-- Define the event of hitting the target at least once
def hit_at_least_once : ℚ := 1 - (1 - prob_A) * (1 - prob_B)

-- Theorem to prove
theorem shooting_probabilities :
  (hit_twice = 1/6) ∧ (hit_at_least_once = 1 - 1/2 * 2/3) :=
sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l2334_233478


namespace NUMINAMATH_CALUDE_min_rooms_for_departments_l2334_233421

def minRooms (d1 d2 d3 : Nat) : Nat :=
  let gcd := Nat.gcd (Nat.gcd d1 d2) d3
  (d1 / gcd) + (d2 / gcd) + (d3 / gcd)

theorem min_rooms_for_departments :
  minRooms 72 58 24 = 77 := by
  sorry

end NUMINAMATH_CALUDE_min_rooms_for_departments_l2334_233421


namespace NUMINAMATH_CALUDE_total_shells_l2334_233465

theorem total_shells (morning_shells afternoon_shells : ℕ) 
  (h1 : morning_shells = 292)
  (h2 : afternoon_shells = 324) :
  morning_shells + afternoon_shells = 616 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_l2334_233465


namespace NUMINAMATH_CALUDE_tan_negative_55_6_pi_l2334_233446

theorem tan_negative_55_6_pi : Real.tan (-55/6 * Real.pi) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_55_6_pi_l2334_233446


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2334_233463

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₀ + a₁ + a₂ + a₃ = 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2334_233463


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l2334_233409

/-- The number of people sitting around the table -/
def total_people : ℕ := 11

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 3

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 3

/-- The probability of math majors sitting consecutively -/
def consecutive_math_prob : ℚ := 1 / 42

theorem math_majors_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := Nat.factorial (total_people - math_majors) * Nat.factorial math_majors
  (favorable_arrangements : ℚ) / total_arrangements = consecutive_math_prob := by
  sorry

#check math_majors_consecutive_probability

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l2334_233409


namespace NUMINAMATH_CALUDE_sports_club_members_l2334_233460

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

end NUMINAMATH_CALUDE_sports_club_members_l2334_233460


namespace NUMINAMATH_CALUDE_x_minus_y_equals_half_l2334_233434

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {2, 0, x}
def B (x y : ℝ) : Set ℝ := {1/x, |x|, y/x}

-- State the theorem
theorem x_minus_y_equals_half (x y : ℝ) : A x = B x y → x - y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_half_l2334_233434


namespace NUMINAMATH_CALUDE_probability_king_or_queen_l2334_233428

-- Define the structure of a standard deck
structure StandardDeck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)

-- Define the properties of a standard deck
def is_standard_deck (d : StandardDeck) : Prop :=
  d.total_cards = 52 ∧
  d.num_ranks = 13 ∧
  d.num_suits = 4 ∧
  d.num_kings = 4 ∧
  d.num_queens = 4

-- Theorem statement
theorem probability_king_or_queen (d : StandardDeck) 
  (h : is_standard_deck d) : 
  (d.num_kings + d.num_queens : ℚ) / d.total_cards = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_king_or_queen_l2334_233428


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2334_233497

theorem polynomial_simplification (x : ℝ) :
  (2 * x^13 + 3 * x^12 - 4 * x^9 + 5 * x^7) + 
  (8 * x^11 - 2 * x^9 + 3 * x^7 + 6 * x^4 - 7 * x + 9) + 
  (x^13 + 4 * x^12 + x^11 + 9 * x^9) = 
  3 * x^13 + 7 * x^12 + 9 * x^11 + 3 * x^9 + 8 * x^7 + 6 * x^4 - 7 * x + 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2334_233497


namespace NUMINAMATH_CALUDE_odd_function_iff_a_b_zero_l2334_233418

def f (x a b : ℝ) : ℝ := x * abs (x - a) + b

theorem odd_function_iff_a_b_zero (a b : ℝ) :
  (∀ x, f x a b = -f (-x) a b) ↔ a^2 + b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_iff_a_b_zero_l2334_233418


namespace NUMINAMATH_CALUDE_simplify_fraction_l2334_233400

theorem simplify_fraction : (140 : ℚ) / 9800 * 35 = 1 / 70 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2334_233400


namespace NUMINAMATH_CALUDE_min_value_theorem_l2334_233449

theorem min_value_theorem (a b : ℝ) (h1 : a > b) 
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x_0 : ℝ, a * x_0^2 + 2 * x_0 + b = 0) :
  (∀ a b : ℝ, 2 * a^2 + b^2 ≥ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, 2 * a^2 + b^2 = 2 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2334_233449


namespace NUMINAMATH_CALUDE_kantana_chocolates_l2334_233484

/-- The number of chocolates Kantana buys for herself and her sister every Saturday -/
def regular_chocolates : ℕ := 3

/-- The number of additional chocolates Kantana bought for Charlie on the last Saturday -/
def additional_chocolates : ℕ := 10

/-- The number of Saturdays in a month -/
def saturdays_in_month : ℕ := 4

/-- The total number of chocolates Kantana bought for the month -/
def total_chocolates : ℕ := (saturdays_in_month - 1) * regular_chocolates + 
                            (regular_chocolates + additional_chocolates)

theorem kantana_chocolates : total_chocolates = 22 := by
  sorry

end NUMINAMATH_CALUDE_kantana_chocolates_l2334_233484


namespace NUMINAMATH_CALUDE_toll_formula_correct_l2334_233496

/-- Represents the toll formula for a truck crossing a bridge -/
def toll_formula (x : ℕ) : ℚ := 0.50 + 0.30 * x

/-- Represents an 18-wheel truck with 2 wheels on its front axle and 4 wheels on each other axle -/
def eighteen_wheel_truck : ℕ := 5

theorem toll_formula_correct : 
  toll_formula eighteen_wheel_truck = 2 := by sorry

end NUMINAMATH_CALUDE_toll_formula_correct_l2334_233496


namespace NUMINAMATH_CALUDE_whole_number_between_fractions_l2334_233407

theorem whole_number_between_fractions (M : ℤ) : 
  (5 < (M : ℚ) / 4) ∧ ((M : ℚ) / 4 < 5.5) → M = 21 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_fractions_l2334_233407


namespace NUMINAMATH_CALUDE_optimal_discount_order_l2334_233483

def book_price : ℚ := 30
def flat_discount : ℚ := 5
def percentage_discount : ℚ := 0.25

def price_flat_then_percent : ℚ := (book_price - flat_discount) * (1 - percentage_discount)
def price_percent_then_flat : ℚ := book_price * (1 - percentage_discount) - flat_discount

theorem optimal_discount_order :
  price_percent_then_flat < price_flat_then_percent ∧
  price_flat_then_percent - price_percent_then_flat = 125 / 100 := by
  sorry

end NUMINAMATH_CALUDE_optimal_discount_order_l2334_233483


namespace NUMINAMATH_CALUDE_committee_size_lower_bound_l2334_233453

/-- A structure representing a committee with its meeting details -/
structure Committee where
  total_meetings : ℕ
  members_per_meeting : ℕ
  total_members : ℕ

/-- The property that no two people have met more than once -/
def no_repeated_meetings (c : Committee) : Prop :=
  c.total_meetings * (c.members_per_meeting.choose 2) ≤ c.total_members.choose 2

/-- The theorem to be proved -/
theorem committee_size_lower_bound (c : Committee) 
  (h1 : c.total_meetings = 40)
  (h2 : c.members_per_meeting = 10)
  (h3 : no_repeated_meetings c) :
  c.total_members > 60 := by
  sorry

end NUMINAMATH_CALUDE_committee_size_lower_bound_l2334_233453


namespace NUMINAMATH_CALUDE_grant_total_earnings_l2334_233470

/-- The total amount Grant made from selling his baseball gear -/
def total_amount : ℝ :=
  let baseball_cards := 25
  let baseball_bat := 10
  let baseball_glove_original := 30
  let baseball_glove_discount := 0.20
  let baseball_glove := baseball_glove_original * (1 - baseball_glove_discount)
  let baseball_cleats := 10
  let num_cleats := 2
  baseball_cards + baseball_bat + baseball_glove + (baseball_cleats * num_cleats)

/-- Theorem stating that the total amount Grant made is $79 -/
theorem grant_total_earnings : total_amount = 79 := by
  sorry

end NUMINAMATH_CALUDE_grant_total_earnings_l2334_233470


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2334_233455

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2334_233455


namespace NUMINAMATH_CALUDE_sum_of_root_products_l2334_233477

theorem sum_of_root_products (p q r s : ℂ) : 
  (4 * p^4 - 8 * p^3 + 12 * p^2 - 16 * p + 9 = 0) →
  (4 * q^4 - 8 * q^3 + 12 * q^2 - 16 * q + 9 = 0) →
  (4 * r^4 - 8 * r^3 + 12 * r^2 - 16 * r + 9 = 0) →
  (4 * s^4 - 8 * s^3 + 12 * s^2 - 16 * s + 9 = 0) →
  p * q + p * r + p * s + q * r + q * s + r * s = -3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_root_products_l2334_233477


namespace NUMINAMATH_CALUDE_kittens_found_on_monday_l2334_233464

def solve_cat_problem (initial_cats : ℕ) (tuesday_cats : ℕ) (adoptions : ℕ) (cats_per_adoption : ℕ) (final_cats : ℕ) : ℕ :=
  initial_cats + tuesday_cats - (adoptions * cats_per_adoption) - final_cats

theorem kittens_found_on_monday :
  solve_cat_problem 20 1 3 2 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_kittens_found_on_monday_l2334_233464


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l2334_233466

theorem unique_solution_factorial_equation :
  ∃! (n : ℕ), n > 0 ∧ (n + 2).factorial - (n + 1).factorial - n.factorial = n^2 + n^4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l2334_233466


namespace NUMINAMATH_CALUDE_inequality_solution_l2334_233430

/-- Given an inequality ax^2 - 3x + 2 > 0 with solution set {x | x < 1 or x > b},
    where b > 1 and a > 0, prove that a = 1, b = 2, and the solution set for
    x^2 - 3x + 2 > 0 is {x | 1 < x < 2}. -/
theorem inequality_solution (a b : ℝ) 
    (h1 : ∀ x, a * x^2 - 3*x + 2 > 0 ↔ x < 1 ∨ x > b)
    (h2 : b > 1) 
    (h3 : a > 0) : 
    a = 1 ∧ b = 2 ∧ (∀ x, x^2 - 3*x + 2 > 0 ↔ 1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2334_233430


namespace NUMINAMATH_CALUDE_impossible_coverage_l2334_233489

/-- Represents a 5x5 board --/
def Board := Fin 5 → Fin 5 → Bool

/-- Represents a stromino (3x1 rectangle) --/
structure Stromino where
  start_row : Fin 5
  start_col : Fin 5
  is_horizontal : Bool

/-- Checks if a stromino is valid (fits within the board) --/
def is_valid_stromino (s : Stromino) : Bool :=
  if s.is_horizontal then
    s.start_col < 3
  else
    s.start_row < 3

/-- Counts how many strominos cover a given square --/
def count_coverage (board : Board) (strominos : List Stromino) (row col : Fin 5) : Nat :=
  sorry

/-- Checks if a given arrangement of strominos is valid --/
def is_valid_arrangement (strominos : List Stromino) : Bool :=
  sorry

/-- The main theorem stating that it's impossible to cover the board with 16 strominos --/
theorem impossible_coverage : ¬ ∃ (strominos : List Stromino),
  strominos.length = 16 ∧
  is_valid_arrangement strominos ∧
  ∀ (row col : Fin 5),
    let coverage := count_coverage (λ _ _ => true) strominos row col
    coverage = 1 ∨ coverage = 2 :=
  sorry

end NUMINAMATH_CALUDE_impossible_coverage_l2334_233489


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_P_l2334_233494

/-- If the terminal side of angle α passes through point P(a, 2a) where a < 0, then cos(α) = -√5/5 -/
theorem cos_alpha_for_point_P (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : ∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = a ∧ r * (Real.sin α) = 2*a) : 
  Real.cos α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_P_l2334_233494


namespace NUMINAMATH_CALUDE_smallest_integer_solution_unique_smallest_solution_l2334_233491

theorem smallest_integer_solution (x : ℤ) : (10 - 5 * x < -18) ↔ x ≥ 6 :=
  sorry

theorem unique_smallest_solution : ∃! x : ℤ, (10 - 5 * x < -18) ∧ ∀ y : ℤ, (10 - 5 * y < -18) → x ≤ y :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_unique_smallest_solution_l2334_233491


namespace NUMINAMATH_CALUDE_chrome_users_l2334_233401

theorem chrome_users (total : ℕ) (angle : ℕ) (chrome_users : ℕ) : 
  total = 530 → angle = 216 → chrome_users = 318 →
  (chrome_users : ℚ) / total * 360 = angle := by
  sorry

end NUMINAMATH_CALUDE_chrome_users_l2334_233401


namespace NUMINAMATH_CALUDE_dog_count_l2334_233490

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  sit : ℕ
  stay : ℕ
  rollOver : ℕ
  sitStay : ℕ
  stayRollOver : ℕ
  sitRollOver : ℕ
  allThree : ℕ
  none : ℕ
  stayRollOverPlayDead : ℕ

/-- The total number of dogs in the training center -/
def totalDogs (d : DogTricks) : ℕ := sorry

/-- Theorem stating the total number of dogs in the training center -/
theorem dog_count (d : DogTricks) 
  (h1 : d.sit = 60)
  (h2 : d.stay = 35)
  (h3 : d.rollOver = 40)
  (h4 : d.sitStay = 22)
  (h5 : d.stayRollOver = 15)
  (h6 : d.sitRollOver = 20)
  (h7 : d.allThree = 10)
  (h8 : d.none = 10)
  (h9 : d.stayRollOverPlayDead = 5)
  (h10 : d.stayRollOverPlayDead ≤ d.stayRollOver) :
  totalDogs d = 98 := by
  sorry

end NUMINAMATH_CALUDE_dog_count_l2334_233490
