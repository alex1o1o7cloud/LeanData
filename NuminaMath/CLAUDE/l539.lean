import Mathlib

namespace NUMINAMATH_CALUDE_xiao_li_commute_l539_53972

/-- Xiao Li's commute problem -/
theorem xiao_li_commute
  (distance : ℝ)
  (walk_late : ℝ)
  (bike_early : ℝ)
  (bike_speed_ratio : ℝ)
  (bike_breakdown_distance : ℝ)
  (h_distance : distance = 4.5)
  (h_walk_late : walk_late = 5 / 60)
  (h_bike_early : bike_early = 10 / 60)
  (h_bike_speed_ratio : bike_speed_ratio = 1.5)
  (h_bike_breakdown_distance : bike_breakdown_distance = 1.5) :
  ∃ (walk_speed bike_speed min_run_speed : ℝ),
    walk_speed = 6 ∧
    bike_speed = 9 ∧
    min_run_speed = 7.2 ∧
    distance / walk_speed - walk_late = distance / bike_speed + bike_early ∧
    bike_speed = bike_speed_ratio * walk_speed ∧
    bike_breakdown_distance / bike_speed +
      (distance - bike_breakdown_distance) / min_run_speed ≤
        distance / bike_speed + bike_early - 5 / 60 :=
by sorry

end NUMINAMATH_CALUDE_xiao_li_commute_l539_53972


namespace NUMINAMATH_CALUDE_cube_root_equals_square_root_l539_53920

theorem cube_root_equals_square_root :
  ∀ x : ℝ, (x ^ (1/3) = x ^ (1/2)) → x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equals_square_root_l539_53920


namespace NUMINAMATH_CALUDE_rectangle_area_from_diagonal_l539_53960

/-- Theorem: Area of a rectangle with length thrice its width and diagonal x -/
theorem rectangle_area_from_diagonal (x : ℝ) (h : x > 0) : 
  ∃ w l : ℝ, w > 0 ∧ l = 3 * w ∧ w^2 + l^2 = x^2 ∧ w * l = (3/10) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_from_diagonal_l539_53960


namespace NUMINAMATH_CALUDE_two_digit_reverse_sqrt_l539_53918

theorem two_digit_reverse_sqrt (n x y : ℕ) : 
  (x > y) →
  (2 * n = x + y) →
  (10 ≤ n ∧ n < 100) →
  (∃ (a b : ℕ), n = 10 * a + b ∧ a < 10 ∧ b < 10) →
  (∃ (k : ℕ), k * k = x * y) →
  (∃ (a b : ℕ), n = 10 * a + b ∧ k = 10 * b + a) →
  (x - y = 66) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sqrt_l539_53918


namespace NUMINAMATH_CALUDE_ratio_problem_l539_53967

theorem ratio_problem (ratio_percent : ℚ) (first_part : ℚ) (second_part : ℚ) :
  ratio_percent = 200 / 3 →
  first_part = 2 →
  first_part / second_part = ratio_percent / 100 →
  second_part = 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l539_53967


namespace NUMINAMATH_CALUDE_problem_solution_l539_53946

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem problem_solution (a b : ℝ) :
  (A ∪ B a b = Set.univ) ∧ 
  (A ∩ B a b = Set.Ioc 3 4) →
  a = -3 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l539_53946


namespace NUMINAMATH_CALUDE_peters_change_l539_53953

/-- Calculates the change left after Peter buys glasses -/
theorem peters_change (small_price large_price total_money small_count large_count : ℕ) : 
  small_price = 3 →
  large_price = 5 →
  total_money = 50 →
  small_count = 8 →
  large_count = 5 →
  total_money - (small_price * small_count + large_price * large_count) = 1 := by
sorry

end NUMINAMATH_CALUDE_peters_change_l539_53953


namespace NUMINAMATH_CALUDE_triangle_max_sin2A_tan2C_l539_53935

theorem triangle_max_sin2A_tan2C (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are positive sides of the triangle
  0 < a ∧ 0 < b ∧ 0 < c →
  -- -c cosB is the arithmetic mean of √2a cosB and √2b cosA
  -c * Real.cos B = (Real.sqrt 2 * a * Real.cos B + Real.sqrt 2 * b * Real.cos A) / 2 →
  -- Maximum value of sin2A•tan²C
  ∃ (max : Real), ∀ (A' B' C' : Real) (a' b' c' : Real),
    0 < A' ∧ 0 < B' ∧ 0 < C' ∧ A' + B' + C' = π →
    0 < a' ∧ 0 < b' ∧ 0 < c' →
    -c' * Real.cos B' = (Real.sqrt 2 * a' * Real.cos B' + Real.sqrt 2 * b' * Real.cos A') / 2 →
    Real.sin (2 * A') * (Real.tan C')^2 ≤ max ∧
    max = 3 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_sin2A_tan2C_l539_53935


namespace NUMINAMATH_CALUDE_circle_area_l539_53938

/-- The area of the circle described by the polar equation r = 3 cos θ - 4 sin θ is 25π/4 -/
theorem circle_area (θ : Real) (r : Real → Real) :
  (r = fun θ ↦ 3 * Real.cos θ - 4 * Real.sin θ) →
  (∀ θ, ∃ x y : Real, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) →
  (∃ c : Real × Real, ∃ radius : Real, ∀ x y : Real,
    (x - c.1)^2 + (y - c.2)^2 = radius^2 ↔ ∃ θ : Real, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) →
  (π * (5/2)^2 : Real) = 25*π/4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_l539_53938


namespace NUMINAMATH_CALUDE_inequality_transformation_l539_53951

theorem inequality_transformation (a b : ℝ) (h : a < b) : -a/3 > -b/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l539_53951


namespace NUMINAMATH_CALUDE_three_digit_number_puzzle_l539_53955

theorem three_digit_number_puzzle :
  ∀ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 →
    a + b + c = 10 →
    b = a + c →
    100 * c + 10 * b + a = 100 * a + 10 * b + c + 99 →
    100 * a + 10 * b + c = 253 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_puzzle_l539_53955


namespace NUMINAMATH_CALUDE_total_gum_pieces_l539_53994

theorem total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) 
  (h1 : packages = 27) (h2 : pieces_per_package = 18) : 
  packages * pieces_per_package = 486 := by
  sorry

end NUMINAMATH_CALUDE_total_gum_pieces_l539_53994


namespace NUMINAMATH_CALUDE_cookies_eaten_yesterday_l539_53909

/-- Given the number of cookies eaten today and the difference between today and yesterday,
    calculate the number of cookies eaten yesterday. -/
def cookies_yesterday (today : ℕ) (difference : ℕ) : ℕ :=
  today - difference

/-- Theorem stating that given 140 cookies eaten today and 30 fewer yesterday,
    the number of cookies eaten yesterday was 110. -/
theorem cookies_eaten_yesterday :
  cookies_yesterday 140 30 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_yesterday_l539_53909


namespace NUMINAMATH_CALUDE_light_travel_distance_l539_53901

/-- The speed of light in miles per second -/
def speed_of_light : ℝ := 186282

/-- The number of seconds light travels -/
def travel_time : ℝ := 500

/-- The conversion factor from miles to kilometers -/
def mile_to_km : ℝ := 1.609

/-- The distance light travels in kilometers -/
def light_distance : ℝ := speed_of_light * travel_time * mile_to_km

theorem light_travel_distance :
  ∃ ε > 0, |light_distance - 1.498e8| < ε :=
sorry

end NUMINAMATH_CALUDE_light_travel_distance_l539_53901


namespace NUMINAMATH_CALUDE_students_walking_home_l539_53992

theorem students_walking_home (bus_fraction car_fraction scooter_fraction : ℚ)
  (h1 : bus_fraction = 2/5)
  (h2 : car_fraction = 1/5)
  (h3 : scooter_fraction = 1/8)
  (h4 : bus_fraction + car_fraction + scooter_fraction < 1) :
  1 - (bus_fraction + car_fraction + scooter_fraction) = 11/40 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l539_53992


namespace NUMINAMATH_CALUDE_area_regular_octagon_in_circle_l539_53996

/-- The area of a regular octagon inscribed in a circle with area 256π -/
theorem area_regular_octagon_in_circle (circle_area : ℝ) (octagon_area : ℝ) : 
  circle_area = 256 * Real.pi → 
  octagon_area = 8 * (1/2 * (circle_area / Real.pi) * Real.sin (Real.pi / 4)) → 
  octagon_area = 512 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_area_regular_octagon_in_circle_l539_53996


namespace NUMINAMATH_CALUDE_abc_mod_nine_l539_53985

theorem abc_mod_nine (a b c : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9)
  (h1 : (a + 2*b + 3*c) % 9 = 0)
  (h2 : (2*a + 3*b + c) % 9 = 3)
  (h3 : (3*a + b + 2*c) % 9 = 8) :
  (a * b * c) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_abc_mod_nine_l539_53985


namespace NUMINAMATH_CALUDE_line_segment_intersection_condition_l539_53924

/-- A line in 2D space defined by the equation ax + y + 2 = 0 -/
structure Line2D where
  a : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Checks if a line intersects a line segment -/
def intersects (l : Line2D) (p q : Point2D) : Prop :=
  sorry

/-- The theorem to be proved -/
theorem line_segment_intersection_condition (l : Line2D) (p q : Point2D) :
  p = Point2D.mk (-2) 1 →
  q = Point2D.mk 3 2 →
  intersects l p q →
  l.a ∈ Set.Ici (3/2) ∪ Set.Iic (-4/3) :=
sorry

end NUMINAMATH_CALUDE_line_segment_intersection_condition_l539_53924


namespace NUMINAMATH_CALUDE_number_multiplied_by_five_thirds_l539_53911

theorem number_multiplied_by_five_thirds : ∃ x : ℚ, (5 : ℚ) / 3 * x = 45 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_five_thirds_l539_53911


namespace NUMINAMATH_CALUDE_group_size_is_eight_l539_53971

/-- The number of people in a group, given certain weight conditions -/
def number_of_people : ℕ :=
  let weight_increase_per_person : ℕ := 5
  let weight_difference : ℕ := 75 - 35
  weight_difference / weight_increase_per_person

theorem group_size_is_eight :
  number_of_people = 8 :=
by
  -- Proof goes here
  sorry

#eval number_of_people  -- Should output 8

end NUMINAMATH_CALUDE_group_size_is_eight_l539_53971


namespace NUMINAMATH_CALUDE_min_value_theorem_l539_53916

theorem min_value_theorem (a : ℝ) (h : 8 * a^2 + 7 * a + 6 = 5) :
  ∃ (m : ℝ), (∀ x, 8 * x^2 + 7 * x + 6 = 5 → 3 * x + 2 ≥ m) ∧ (∃ y, 8 * y^2 + 7 * y + 6 = 5 ∧ 3 * y + 2 = m) ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l539_53916


namespace NUMINAMATH_CALUDE_john_cannot_achieve_goal_l539_53925

/-- Represents John's quiz scores throughout the year -/
structure QuizScores where
  total : Nat
  goal_percentage : Rat
  taken : Nat
  high_scores : Nat

/-- Checks if it's possible to achieve the goal given the current scores -/
def can_achieve_goal (scores : QuizScores) : Prop :=
  ∃ (remaining_high_scores : Nat),
    remaining_high_scores ≤ scores.total - scores.taken ∧
    (scores.high_scores + remaining_high_scores : Rat) / scores.total ≥ scores.goal_percentage

/-- John's actual quiz scores -/
def john_scores : QuizScores :=
  { total := 60
  , goal_percentage := 9/10
  , taken := 40
  , high_scores := 32 }

/-- Theorem stating that John cannot achieve his goal -/
theorem john_cannot_achieve_goal :
  ¬(can_achieve_goal john_scores) := by
  sorry

end NUMINAMATH_CALUDE_john_cannot_achieve_goal_l539_53925


namespace NUMINAMATH_CALUDE_expense_increase_percentage_l539_53928

theorem expense_increase_percentage (monthly_salary : ℝ) (initial_savings_rate : ℝ) (new_savings : ℝ) :
  monthly_salary = 6500 →
  initial_savings_rate = 0.20 →
  new_savings = 260 →
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_savings - new_savings
  expense_increase / initial_expenses = 0.20 := by sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_l539_53928


namespace NUMINAMATH_CALUDE_sample_is_reading_time_data_l539_53944

/-- Represents a resident in the study -/
structure Resident where
  id : Nat
  readingTime : ℝ

/-- Represents the statistical study -/
structure ReadingStudy where
  population : Finset Resident
  sampleSize : Nat
  sample : Finset Resident

/-- Definition of a valid sample in the reading study -/
def validSample (study : ReadingStudy) : Prop :=
  study.sample.card = study.sampleSize ∧
  study.sample ⊆ study.population

/-- The main theorem about the sample definition -/
theorem sample_is_reading_time_data (study : ReadingStudy)
    (h_pop_size : study.population.card = 5000)
    (h_sample_size : study.sampleSize = 200)
    (h_valid_sample : validSample study) :
    ∃ (sample_data : Finset ℝ),
      sample_data = study.sample.image Resident.readingTime ∧
      sample_data.card = study.sampleSize :=
  sorry


end NUMINAMATH_CALUDE_sample_is_reading_time_data_l539_53944


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_quadratic_equation_coefficients_l539_53900

/-- Given a quadratic equation 2x^2 - 1 = 6x, prove its general form and coefficients --/
theorem quadratic_equation_general_form :
  ∀ x : ℝ, (2 * x^2 - 1 = 6 * x) ↔ (2 * x^2 - 6 * x - 1 = 0) :=
by sorry

/-- Prove the coefficients of the general form ax^2 + bx + c = 0 --/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = 2 * x^2 - 6 * x - 1) ∧ 
    (a = 2 ∧ b = -6 ∧ c = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_quadratic_equation_coefficients_l539_53900


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l539_53950

theorem trigonometric_inequality (x : ℝ) (n m : ℕ) 
  (h1 : 0 < x) (h2 : x < Real.pi / 2) (h3 : n > m) :
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤ 3 * |Real.sin x ^ m - Real.cos x ^ m| := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l539_53950


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l539_53982

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 - Complex.I) ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l539_53982


namespace NUMINAMATH_CALUDE_incorrect_statement_is_false_l539_53906

/-- Represents the method used for separation and counting of bacteria. -/
inductive SeparationMethod
| DilutionPlating
| StreakPlate

/-- Represents a biotechnology practice. -/
structure BiotechPractice where
  soil_bacteria_method : SeparationMethod
  fruit_vinegar_air : Bool
  nitrite_detection : Bool
  dna_extraction : Bool

/-- The correct biotechnology practices. -/
def correct_practices : BiotechPractice := {
  soil_bacteria_method := SeparationMethod.DilutionPlating,
  fruit_vinegar_air := true,
  nitrite_detection := true,
  dna_extraction := true
}

/-- The statement to be proven false. -/
def incorrect_statement : BiotechPractice := {
  soil_bacteria_method := SeparationMethod.StreakPlate,
  fruit_vinegar_air := true,
  nitrite_detection := true,
  dna_extraction := true
}

/-- Theorem stating that the incorrect statement is indeed incorrect. -/
theorem incorrect_statement_is_false : incorrect_statement ≠ correct_practices := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_is_false_l539_53906


namespace NUMINAMATH_CALUDE_least_number_with_divisibility_property_l539_53957

theorem least_number_with_divisibility_property : ∃ k : ℕ, 
  k > 0 ∧ 
  (k / 23 = k % 47 + 13) ∧
  (∀ m : ℕ, m > 0 → m < k → m / 23 ≠ m % 47 + 13) ∧
  k = 576 :=
sorry

end NUMINAMATH_CALUDE_least_number_with_divisibility_property_l539_53957


namespace NUMINAMATH_CALUDE_log_expression_equality_l539_53929

theorem log_expression_equality : 
  4 * Real.log 3 / Real.log 2 - Real.log (81 / 4) / Real.log 2 - (5 : ℝ) ^ (Real.log 3 / Real.log 5) + Real.log (Real.sqrt 3) / Real.log 9 = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l539_53929


namespace NUMINAMATH_CALUDE_fifth_equation_in_pattern_l539_53991

theorem fifth_equation_in_pattern : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 4 → 
    (List.range n).sum + (List.range n).sum.succ = n^2) →
  (List.range 5).sum + (List.range 5).sum.succ = 81 :=
sorry

end NUMINAMATH_CALUDE_fifth_equation_in_pattern_l539_53991


namespace NUMINAMATH_CALUDE_sequence_formula_l539_53954

-- Define the sequence and its sum
def S (n : ℕ) : ℕ := 2 * n^2 + n

-- Define the nth term of the sequence
def a (n : ℕ) : ℕ := 4 * n - 1

-- Theorem statement
theorem sequence_formula (n : ℕ) : S n - S (n-1) = a n :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l539_53954


namespace NUMINAMATH_CALUDE_l₃_equation_min_distance_l₁_l₄_l539_53983

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x + y = 3
def l₂ (x y : ℝ) : Prop := x - y = 0
def l₄ (x y : ℝ) (m : ℝ) : Prop := 4 * x + 2 * y + m^2 + 1 = 0

-- Define the intersection point A
def A : ℝ × ℝ := (1, 1)

-- Theorem for the equation of l₃
theorem l₃_equation : 
  ∃ (l₃ : ℝ → ℝ → Prop), 
    (l₃ (A.1) (A.2)) ∧ 
    (∀ x y, l₃ x y ↔ x - 2*y + 1 = 0) ∧
    (∀ x y, l₁ x y → (y - A.2 = -1/2 * (x - A.1) ↔ l₃ x y)) :=
sorry

-- Theorem for the minimum distance between l₁ and l₄
theorem min_distance_l₁_l₄ :
  ∃ (d : ℝ), 
    d = 7 * Real.sqrt 5 / 10 ∧
    (∀ x y m, l₁ x y → l₄ x y m → 
      (x - 0)^2 + (y - 0)^2 ≥ d^2) :=
sorry

end NUMINAMATH_CALUDE_l₃_equation_min_distance_l₁_l₄_l539_53983


namespace NUMINAMATH_CALUDE_randy_blocks_left_l539_53912

/-- Calculates the number of blocks Randy has left after a series of actions. -/
def blocks_left (initial : ℕ) (used : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - used - given_away + bought

/-- Proves that Randy has 70 blocks left after his actions. -/
theorem randy_blocks_left : 
  blocks_left 78 19 25 36 = 70 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_left_l539_53912


namespace NUMINAMATH_CALUDE_least_x_for_divisibility_l539_53908

theorem least_x_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(3 ∣ 1894 * y)) ∧ (3 ∣ 1894 * x) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_x_for_divisibility_l539_53908


namespace NUMINAMATH_CALUDE_son_age_is_30_l539_53968

/-- The age difference between the man and his son -/
def age_difference : ℕ := 32

/-- The present age of the son -/
def son_age : ℕ := 30

/-- The present age of the man -/
def man_age : ℕ := son_age + age_difference

theorem son_age_is_30 :
  (man_age = son_age + age_difference) ∧
  (man_age + 2 = 2 * (son_age + 2)) →
  son_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_son_age_is_30_l539_53968


namespace NUMINAMATH_CALUDE_point_on_circle_l539_53939

theorem point_on_circle (t : ℝ) :
  let x := (2 - t^2) / (2 + t^2)
  let y := 3*t / (2 + t^2)
  x^2 + y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_point_on_circle_l539_53939


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l539_53937

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem largest_four_digit_divisible_by_6 :
  ∀ n : ℕ, is_four_digit n → divisible_by_6 n → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l539_53937


namespace NUMINAMATH_CALUDE_omega_roots_quadratic_equation_l539_53981

theorem omega_roots_quadratic_equation :
  ∀ (ω : ℂ) (α β : ℂ),
    ω^5 = 1 →
    ω ≠ 1 →
    α = ω + ω^2 →
    β = ω^3 + ω^4 →
    ∃ (a b : ℝ), ∀ (x : ℂ), x = α ∨ x = β → x^2 + a*x + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_omega_roots_quadratic_equation_l539_53981


namespace NUMINAMATH_CALUDE_algebraic_identities_l539_53997

theorem algebraic_identities (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hab : a ≠ b) :
  (a / (a - b) + b / (b - a) = 1) ∧
  (a^2 / (b^2 * c) * (-b * c^2 / (2 * a)) / (a / b) = -c) :=
sorry

end NUMINAMATH_CALUDE_algebraic_identities_l539_53997


namespace NUMINAMATH_CALUDE_max_enclosed_area_l539_53914

/-- Represents the side length of the square garden -/
def s : ℕ := 49

/-- Represents the non-shared side length of the rectangular garden -/
def x : ℕ := 2

/-- The total perimeter of both gardens combined -/
def total_perimeter : ℕ := 200

/-- The maximum area that can be enclosed -/
def max_area : ℕ := 2499

/-- Theorem stating the maximum area that can be enclosed given the constraints -/
theorem max_enclosed_area :
  (4 * s + 2 * x = total_perimeter) → 
  (∀ s' x' : ℕ, (4 * s' + 2 * x' = total_perimeter) → (s' * s' + s' * x' ≤ max_area)) →
  (s * s + s * x = max_area) := by
  sorry

end NUMINAMATH_CALUDE_max_enclosed_area_l539_53914


namespace NUMINAMATH_CALUDE_fifth_power_sum_equality_l539_53980

theorem fifth_power_sum_equality : ∃! m : ℕ+, m.val ^ 5 = 144 ^ 5 + 91 ^ 5 + 56 ^ 5 + 19 ^ 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_equality_l539_53980


namespace NUMINAMATH_CALUDE_abs_neg_two_and_half_l539_53934

theorem abs_neg_two_and_half : |(-5/2 : ℚ)| = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_and_half_l539_53934


namespace NUMINAMATH_CALUDE_helmet_sales_and_pricing_l539_53964

/-- Helmet sales and pricing problem -/
theorem helmet_sales_and_pricing
  (march_sales : ℕ)
  (may_sales : ℕ)
  (cost_price : ℝ)
  (initial_price : ℝ)
  (initial_monthly_sales : ℕ)
  (price_sensitivity : ℝ)
  (target_profit : ℝ)
  (h_march_sales : march_sales = 256)
  (h_may_sales : may_sales = 400)
  (h_cost_price : cost_price = 30)
  (h_initial_price : initial_price = 40)
  (h_initial_monthly_sales : initial_monthly_sales = 600)
  (h_price_sensitivity : price_sensitivity = 10)
  (h_target_profit : target_profit = 10000)
  :
  ∃ (r : ℝ) (actual_price : ℝ),
    r > 0 ∧
    r = 0.25 ∧
    actual_price = 50 ∧
    march_sales * (1 + r)^2 = may_sales ∧
    (actual_price - cost_price) * (initial_monthly_sales - price_sensitivity * (actual_price - initial_price)) = target_profit ∧
    actual_price ≥ initial_price :=
by sorry

end NUMINAMATH_CALUDE_helmet_sales_and_pricing_l539_53964


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l539_53952

theorem sum_of_fractions_equals_one (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) : 
  1 / (1 + a + a * b) + 1 / (1 + b + b * c) + 1 / (1 + c + c * a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l539_53952


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l539_53959

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of being monotonically increasing on [0,+∞)
def IsIncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Define the set of x satisfying f(2x-1) < f(3)
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (2*x - 1) < f 3}

-- State the theorem
theorem solution_set_is_open_interval
  (f : ℝ → ℝ) (h1 : IsEven f) (h2 : IsIncreasingOnNonnegative f) :
  SolutionSet f = Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l539_53959


namespace NUMINAMATH_CALUDE_constant_term_zero_l539_53917

theorem constant_term_zero (x : ℝ) (x_pos : x > 0) : 
  (∃ k : ℕ, k ≤ 10 ∧ (10 - k) / 2 - k = 0) → False :=
sorry

end NUMINAMATH_CALUDE_constant_term_zero_l539_53917


namespace NUMINAMATH_CALUDE_total_textbook_cost_l539_53915

/-- The total cost of textbooks given specific pricing conditions -/
theorem total_textbook_cost : 
  ∀ (sale_price : ℕ) (online_total : ℕ) (sale_count online_count bookstore_count : ℕ),
    sale_price = 10 →
    online_total = 40 →
    sale_count = 5 →
    online_count = 2 →
    bookstore_count = 3 →
    sale_count * sale_price + online_total + bookstore_count * online_total = 210 :=
by sorry

end NUMINAMATH_CALUDE_total_textbook_cost_l539_53915


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l539_53907

/-- Calculates the cost of paving a rectangular floor -/
def calculate_paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a 5.5m x 4m room at 850 Rs/m² is 18700 Rs -/
theorem paving_cost_calculation :
  calculate_paving_cost 5.5 4 850 = 18700 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l539_53907


namespace NUMINAMATH_CALUDE_even_function_implies_A_equals_one_l539_53913

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x + 1)(x - A) -/
def f (A : ℝ) : ℝ → ℝ := λ x ↦ (x + 1) * (x - A)

/-- If f(x) = (x + 1)(x - A) is an even function, then A = 1 -/
theorem even_function_implies_A_equals_one :
  IsEven (f A) → A = 1 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_A_equals_one_l539_53913


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l539_53995

/-- A polynomial with real coefficients -/
def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem sum_of_coefficients (a b c d : ℝ) :
  g a b c d (1 + Complex.I) = 0 →
  g a b c d (3 * Complex.I) = 0 →
  a + b + c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l539_53995


namespace NUMINAMATH_CALUDE_vector_addition_l539_53969

theorem vector_addition (a b : Fin 2 → ℝ) 
  (ha : a = ![2, 1]) 
  (hb : b = ![1, 3]) : 
  a + b = ![3, 4] := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l539_53969


namespace NUMINAMATH_CALUDE_parabola_vertex_l539_53976

/-- The vertex of a parabola defined by y = -2x^2 + 3 is (0, 3) -/
theorem parabola_vertex :
  let f : ℝ → ℝ := λ x => -2 * x^2 + 3
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 3 ∧ ∀ x : ℝ, f x ≤ f p.1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l539_53976


namespace NUMINAMATH_CALUDE_train_stop_time_l539_53904

/-- Proves that a train with given speeds stops for 18 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 30)
  (h2 : speed_with_stops = 21) : ℝ :=
by
  -- Define the stop time in minutes
  let stop_time : ℝ := 18
  
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_train_stop_time_l539_53904


namespace NUMINAMATH_CALUDE_store_B_cheapest_l539_53999

/-- Represents a store with its pricing strategy -/
structure Store :=
  (name : String)
  (basePrice : ℕ)
  (discountStrategy : ℕ → ℕ)

/-- Calculates the cost of buying balls from a store -/
def cost (s : Store) (balls : ℕ) : ℕ :=
  s.discountStrategy balls

/-- Store A's discount strategy -/
def storeAStrategy (balls : ℕ) : ℕ :=
  let freeBalls := (balls / 10) * 2
  (balls - freeBalls) * 25

/-- Store B's discount strategy -/
def storeBStrategy (balls : ℕ) : ℕ :=
  balls * (25 - 5)

/-- Store C's discount strategy -/
def storeCStrategy (balls : ℕ) : ℕ :=
  let totalSpent := balls * 25
  let cashback := (totalSpent / 200) * 30
  totalSpent - cashback

/-- The three stores -/
def storeA : Store := ⟨"A", 25, storeAStrategy⟩
def storeB : Store := ⟨"B", 25, storeBStrategy⟩
def storeC : Store := ⟨"C", 25, storeCStrategy⟩

/-- The theorem to prove -/
theorem store_B_cheapest : 
  cost storeB 60 < cost storeA 60 ∧ cost storeB 60 < cost storeC 60 := by
  sorry

end NUMINAMATH_CALUDE_store_B_cheapest_l539_53999


namespace NUMINAMATH_CALUDE_area_relationship_l539_53998

/-- Triangle with sides 13, 14, and 15 inscribed in a circle -/
structure InscribedTriangle where
  -- Define the sides of the triangle
  a : ℝ := 13
  b : ℝ := 14
  c : ℝ := 15
  -- Define the areas of non-triangular regions
  A : ℝ
  B : ℝ
  C : ℝ
  -- C is the largest area
  hC_largest : C ≥ A ∧ C ≥ B

/-- The relationship between areas A, B, C, and the triangle area -/
theorem area_relationship (t : InscribedTriangle) : t.A + t.B + 84 = t.C := by
  sorry

end NUMINAMATH_CALUDE_area_relationship_l539_53998


namespace NUMINAMATH_CALUDE_function_problem_l539_53942

/-- Given a function f(x) = x / (ax + b) where a ≠ 0, f(4) = 4/3, and f(x) = x has a unique solution,
    prove that f(x) = 2x / (x + 2) and f[f(-3)] = 3/2 -/
theorem function_problem (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ x / (a * x + b)
  (f 4 = 4 / 3) →
  (∃! x, f x = x) →
  (∀ x, f x = 2 * x / (x + 2)) ∧
  (f (f (-3)) = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_function_problem_l539_53942


namespace NUMINAMATH_CALUDE_brooke_has_eight_customers_l539_53936

/-- Represents Brooke's milk and butter business --/
structure MilkBusiness where
  num_cows : ℕ
  milk_price : ℚ
  butter_price : ℚ
  milk_per_cow : ℕ
  milk_per_customer : ℕ
  total_revenue : ℚ

/-- Calculates the number of customers in Brooke's milk business --/
def calculate_customers (business : MilkBusiness) : ℕ :=
  let total_milk := business.num_cows * business.milk_per_cow
  total_milk / business.milk_per_customer

/-- Theorem stating that Brooke has 8 customers --/
theorem brooke_has_eight_customers :
  let brooke_business : MilkBusiness := {
    num_cows := 12,
    milk_price := 3,
    butter_price := 3/2,
    milk_per_cow := 4,
    milk_per_customer := 6,
    total_revenue := 144
  }
  calculate_customers brooke_business = 8 := by
  sorry

end NUMINAMATH_CALUDE_brooke_has_eight_customers_l539_53936


namespace NUMINAMATH_CALUDE_evaluate_expression_l539_53933

theorem evaluate_expression (x : ℝ) (y : ℝ) (h1 : x = 5) (h2 : y = 2 * x) : 
  y * (y - 2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l539_53933


namespace NUMINAMATH_CALUDE_each_child_gets_twenty_cookies_l539_53945

/-- Represents the cookie distribution problem in Everlee's family -/
def cookie_distribution (total_cookies : ℕ) (num_adults : ℕ) (num_children : ℕ) : ℕ :=
  let adults_share := total_cookies / 3
  let remaining_cookies := total_cookies - adults_share
  remaining_cookies / num_children

/-- Theorem stating that each child gets 20 cookies -/
theorem each_child_gets_twenty_cookies :
  cookie_distribution 120 2 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_each_child_gets_twenty_cookies_l539_53945


namespace NUMINAMATH_CALUDE_line_slope_l539_53947

theorem line_slope (x y : ℝ) : 
  (3 * x - Real.sqrt 3 * y + 1 = 0) → 
  (∃ m : ℝ, y = m * x + (-1 / Real.sqrt 3) ∧ m = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_line_slope_l539_53947


namespace NUMINAMATH_CALUDE_no_valid_a_l539_53948

theorem no_valid_a : ∀ a : ℝ, a > 0 → ∃ x : ℝ, 
  |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_a_l539_53948


namespace NUMINAMATH_CALUDE_odot_two_four_l539_53919

def odot (a b : ℝ) : ℝ := 5 * a + 2 * b

theorem odot_two_four : odot 2 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_odot_two_four_l539_53919


namespace NUMINAMATH_CALUDE_tom_charges_twelve_l539_53978

/-- Represents Tom's lawn mowing business --/
structure LawnBusiness where
  gas_cost : ℕ
  lawns_mowed : ℕ
  extra_income : ℕ
  total_profit : ℕ

/-- Calculates the price per lawn given Tom's business details --/
def price_per_lawn (b : LawnBusiness) : ℚ :=
  (b.total_profit + b.gas_cost - b.extra_income) / b.lawns_mowed

/-- Theorem stating that Tom charges $12 per lawn --/
theorem tom_charges_twelve (tom : LawnBusiness) 
  (h1 : tom.gas_cost = 17)
  (h2 : tom.lawns_mowed = 3)
  (h3 : tom.extra_income = 10)
  (h4 : tom.total_profit = 29) : 
  price_per_lawn tom = 12 := by
  sorry


end NUMINAMATH_CALUDE_tom_charges_twelve_l539_53978


namespace NUMINAMATH_CALUDE_compare_a_b_fraction_inequality_l539_53910

-- Problem 1
theorem compare_a_b (m n : ℝ) :
  (m^2 + 1) * (n^2 + 4) ≥ (m * n + 2)^2 := by sorry

-- Problem 2
theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) (h5 : e > 0) :
  e / (a - c) < e / (b - d) := by sorry

end NUMINAMATH_CALUDE_compare_a_b_fraction_inequality_l539_53910


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l539_53922

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : t.A = 60 * π / 180)  -- A = 60°
  (h2 : t.a = 3)             -- a = 3
  (h3 : t.A + t.B + t.C = π) -- Sum of angles in a triangle
  (h4 : t.a / Real.sin t.A = t.b / Real.sin t.B)  -- Law of Sines
  (h5 : t.b / Real.sin t.B = t.c / Real.sin t.C)  -- Law of Sines
  : (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l539_53922


namespace NUMINAMATH_CALUDE_profit_percentage_theorem_l539_53973

theorem profit_percentage_theorem (selling_price purchase_price : ℝ) 
  (h1 : selling_price > 0) 
  (h2 : purchase_price > 0) 
  (h3 : selling_price > purchase_price) :
  let original_profit_percentage := (selling_price - purchase_price) / purchase_price * 100
  let new_purchase_price := purchase_price * 0.95
  let new_profit_percentage := (selling_price - new_purchase_price) / new_purchase_price * 100
  (new_profit_percentage - original_profit_percentage = 15) → 
  (original_profit_percentage = 185) := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_theorem_l539_53973


namespace NUMINAMATH_CALUDE_problem_statement_l539_53903

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (∃ (max : ℝ), max = 1/3 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 1 → x*y + y*z + z*x ≤ max) ∧ 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ 1/2) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l539_53903


namespace NUMINAMATH_CALUDE_inequality_proof_l539_53984

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2)/(x^2 + y*z) + (y^2 + 2*z^2 + 2*x^2)/(y^2 + z*x) + (z^2 + 2*x^2 + 2*y^2)/(z^2 + x*y) > 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l539_53984


namespace NUMINAMATH_CALUDE_equal_areas_in_circle_configuration_l539_53989

/-- Given a circle with radius R and four smaller circles with radius r = R/2 drawn through its center and touching it, 
    the area of the region not covered by the smaller circles (black region) 
    is equal to the sum of the areas of the overlapping regions of the smaller circles (gray regions). -/
theorem equal_areas_in_circle_configuration (R : ℝ) (h : R > 0) : 
  ∃ (black_area gray_area : ℝ),
    black_area = R^2 * π - 4 * (R/2)^2 * π ∧
    gray_area = 4 * ((R/2)^2 * π - (R/2)^2 * π / 3) ∧
    black_area = gray_area :=
by sorry

end NUMINAMATH_CALUDE_equal_areas_in_circle_configuration_l539_53989


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l539_53905

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l539_53905


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l539_53990

/-- An isosceles triangle with congruent sides of length 8 cm and perimeter 27 cm has a base of length 11 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base : ℝ),
  base > 0 →
  8 > 0 →
  8 + 8 + base = 27 →
  base = 11 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l539_53990


namespace NUMINAMATH_CALUDE_distance_from_point_to_x_axis_l539_53970

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distance_to_x_axis (x y : ℝ) : ℝ :=
  |y|

theorem distance_from_point_to_x_axis :
  distance_to_x_axis 3 (-4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_x_axis_l539_53970


namespace NUMINAMATH_CALUDE_square_root_of_25_l539_53979

theorem square_root_of_25 : 
  {x : ℝ | x^2 = 25} = {5, -5} := by sorry

end NUMINAMATH_CALUDE_square_root_of_25_l539_53979


namespace NUMINAMATH_CALUDE_gathering_attendance_l539_53930

theorem gathering_attendance (empty_chairs : ℕ) 
  (h1 : empty_chairs = 9)
  (h2 : ∃ (total_chairs seated_people total_people : ℕ),
    empty_chairs = total_chairs / 3 ∧
    seated_people = 2 * total_chairs / 3 ∧
    seated_people = 3 * total_people / 5) :
  ∃ (total_people : ℕ), total_people = 30 :=
by sorry

end NUMINAMATH_CALUDE_gathering_attendance_l539_53930


namespace NUMINAMATH_CALUDE_multiply_polynomial_equality_l539_53958

theorem multiply_polynomial_equality (x : ℝ) : 
  (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomial_equality_l539_53958


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l539_53966

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 40 * x - 72 = (a * x + b)^2 + c) →
  a * b = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l539_53966


namespace NUMINAMATH_CALUDE_digit_count_theorem_l539_53963

/-- The number of n-digit positive integers -/
def nDigitNumbers (n : ℕ) : ℕ := 9 * 10^(n-1)

/-- The total number of digits needed to write all natural numbers from 1 to 10^n (not including 10^n) -/
def totalDigits (n : ℕ) : ℚ := n * 10^n - (10^n - 1) / 9

theorem digit_count_theorem (n : ℕ) (h : n > 0) :
  (∀ k : ℕ, k > 0 → k ≤ n → nDigitNumbers k = 9 * 10^(k-1)) ∧
  totalDigits n = n * 10^n - (10^n - 1) / 9 :=
sorry

end NUMINAMATH_CALUDE_digit_count_theorem_l539_53963


namespace NUMINAMATH_CALUDE_sum_of_squares_l539_53965

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 131) → (a + b + c = 20) → (a^2 + b^2 + c^2 = 138) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l539_53965


namespace NUMINAMATH_CALUDE_quadratic_minimum_l539_53943

theorem quadratic_minimum (f : ℝ → ℝ) (h : f = λ x => (x - 1)^2 + 3) : 
  ∀ x, f x ≥ 3 ∧ ∃ x₀, f x₀ = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l539_53943


namespace NUMINAMATH_CALUDE_correct_sticker_distribution_l539_53921

/-- Represents the number of stickers Miss Walter has and distributes -/
structure StickerDistribution where
  gold : Nat
  silver : Nat
  bronze : Nat
  students : Nat

/-- Calculates the number of stickers each student receives -/
def stickersPerStudent (sd : StickerDistribution) : Nat :=
  (sd.gold + sd.silver + sd.bronze) / sd.students

/-- Theorem stating the correct number of stickers each student receives -/
theorem correct_sticker_distribution :
  ∀ sd : StickerDistribution,
    sd.gold = 50 →
    sd.silver = 2 * sd.gold →
    sd.bronze = sd.silver - 20 →
    sd.students = 5 →
    stickersPerStudent sd = 46 := by
  sorry

end NUMINAMATH_CALUDE_correct_sticker_distribution_l539_53921


namespace NUMINAMATH_CALUDE_point_A_in_first_quadrant_l539_53961

-- Define the Cartesian coordinate system
def CartesianCoordinate := ℝ × ℝ

-- Define the point A
def A : CartesianCoordinate := (1, 2)

-- Define the first quadrant
def FirstQuadrant (p : CartesianCoordinate) : Prop :=
  p.1 > 0 ∧ p.2 > 0

-- Theorem statement
theorem point_A_in_first_quadrant : FirstQuadrant A := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_first_quadrant_l539_53961


namespace NUMINAMATH_CALUDE_li_ying_final_score_l539_53956

/-- Calculate the final score in a quiz where correct answers earn points and incorrect answers deduct points. -/
def calculate_final_score (correct_points : ℤ) (incorrect_points : ℤ) (num_correct : ℕ) (num_incorrect : ℕ) : ℤ :=
  correct_points * num_correct - incorrect_points * num_incorrect

/-- Theorem stating that Li Ying's final score in the safety knowledge quiz is 45 points. -/
theorem li_ying_final_score :
  let correct_points : ℤ := 5
  let incorrect_points : ℤ := 3
  let num_correct : ℕ := 12
  let num_incorrect : ℕ := 5
  calculate_final_score correct_points incorrect_points num_correct num_incorrect = 45 := by
  sorry

#eval calculate_final_score 5 3 12 5

end NUMINAMATH_CALUDE_li_ying_final_score_l539_53956


namespace NUMINAMATH_CALUDE_bear_weight_gain_l539_53931

def bear_weight_problem (total_weight : ℝ) 
  (berry_fraction : ℝ) (insect_fraction : ℝ) 
  (acorn_multiplier : ℝ) (honey_multiplier : ℝ) 
  (salmon_fraction : ℝ) : Prop :=
  let berry_weight := berry_fraction * total_weight
  let insect_weight := insect_fraction * total_weight
  let acorn_weight := acorn_multiplier * berry_weight
  let honey_weight := honey_multiplier * insect_weight
  let gained_weight := berry_weight + insect_weight + acorn_weight + honey_weight
  gained_weight = total_weight →
  total_weight - gained_weight = 0 →
  total_weight - (berry_weight + insect_weight + acorn_weight + honey_weight) = 0

theorem bear_weight_gain :
  bear_weight_problem 1200 (1/5) (1/10) 2 3 (1/4) →
  0 = 0 := by sorry

end NUMINAMATH_CALUDE_bear_weight_gain_l539_53931


namespace NUMINAMATH_CALUDE_football_league_analysis_l539_53940

structure Team :=
  (avg_goals_conceded : ℝ)
  (std_dev_goals : ℝ)

def team1 : Team := ⟨1.5, 1.1⟩
def team2 : Team := ⟨2.1, 0.4⟩

def better_defense (t1 t2 : Team) : Prop :=
  t1.avg_goals_conceded < t2.avg_goals_conceded

def more_stable_defense (t1 t2 : Team) : Prop :=
  t1.std_dev_goals < t2.std_dev_goals

def inconsistent_defense (t : Team) : Prop :=
  t.std_dev_goals > 1.0

def rarely_concedes_no_goals (t : Team) : Prop :=
  t.avg_goals_conceded > 2.0 ∧ t.std_dev_goals < 0.5

theorem football_league_analysis :
  (better_defense team1 team2) ∧
  (more_stable_defense team2 team1) ∧
  (inconsistent_defense team1) ∧
  ¬(rarely_concedes_no_goals team2) :=
by sorry

end NUMINAMATH_CALUDE_football_league_analysis_l539_53940


namespace NUMINAMATH_CALUDE_basketball_free_throw_probability_l539_53977

theorem basketball_free_throw_probability :
  ∀ p : ℝ,
  0 ≤ p ∧ p ≤ 1 →
  (1 - p^2 = 16/25) →
  p = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_basketball_free_throw_probability_l539_53977


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l539_53987

/-- Given two complementary angles with measures in the ratio of 3:1, 
    their positive difference is 45 degrees. -/
theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a = 3 * b →   -- ratio of angles is 3:1
  |a - b| = 45  -- positive difference is 45 degrees
:= by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l539_53987


namespace NUMINAMATH_CALUDE_cross_product_equals_l539_53993

def vector1 : ℝ × ℝ × ℝ := (3, -4, 5)
def vector2 : ℝ × ℝ × ℝ := (-2, 7, 1)

def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, c) := v1
  let (d, e, f) := v2
  (b * f - c * e, c * d - a * f, a * e - b * d)

theorem cross_product_equals : cross_product vector1 vector2 = (-39, -13, 13) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_equals_l539_53993


namespace NUMINAMATH_CALUDE_functional_equation_solution_l539_53986

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f (x - y) + 4 * x * y) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x^2 + c :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l539_53986


namespace NUMINAMATH_CALUDE_three_digit_number_appended_to_1220_l539_53975

theorem three_digit_number_appended_to_1220 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (1220000 + n) % 2014 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_three_digit_number_appended_to_1220_l539_53975


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_l539_53988

theorem power_of_two_plus_one (b m n : ℕ) 
  (h1 : b > 1) 
  (h2 : m ≠ n) 
  (h3 : ∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) : 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_l539_53988


namespace NUMINAMATH_CALUDE_alcohol_dilution_l539_53962

theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 15 →
  initial_percentage = 26 →
  added_water = 5 →
  let alcohol_volume := initial_volume * (initial_percentage / 100)
  let total_volume := initial_volume + added_water
  let final_percentage := (alcohol_volume / total_volume) * 100
  final_percentage = 19.5 := by
    sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l539_53962


namespace NUMINAMATH_CALUDE_solve_lollipop_problem_l539_53923

def lollipop_problem (alison henry diane emily : ℕ) (daily_rate : ℝ) : Prop :=
  henry = alison + 30 ∧
  alison = 60 ∧
  alison * 2 = diane ∧
  emily = 50 ∧
  emily + 10 = diane ∧
  daily_rate = 1.5 ∧
  ∃ (days : ℕ), days = 4 ∧
    (let total := alison + henry + diane + emily
     let first_day := 45
     let rec consumed (n : ℕ) : ℝ :=
       if n = 0 then 0
       else if n = 1 then first_day
       else consumed (n - 1) * daily_rate
     consumed days > total ∧ consumed (days - 1) ≤ total)

theorem solve_lollipop_problem :
  ∃ (alison henry diane emily : ℕ) (daily_rate : ℝ),
    lollipop_problem alison henry diane emily daily_rate :=
by
  sorry

end NUMINAMATH_CALUDE_solve_lollipop_problem_l539_53923


namespace NUMINAMATH_CALUDE_blackboard_multiplication_l539_53927

theorem blackboard_multiplication (a b : ℕ) (n : ℕ+) : 
  (100 ≤ a ∧ a ≤ 999) →
  (100 ≤ b ∧ b ≤ 999) →
  10000 * a + b = n * (a * b) →
  n = 73 := by sorry

end NUMINAMATH_CALUDE_blackboard_multiplication_l539_53927


namespace NUMINAMATH_CALUDE_solution_inequality_l539_53926

theorem solution_inequality (x : ℝ) (h : x = 1.8) : x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_inequality_l539_53926


namespace NUMINAMATH_CALUDE_cricket_run_rate_proof_l539_53941

/-- Calculates the required run rate for the remaining overs in a cricket game. -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let first_runs := (first_run_rate * first_overs : ℚ).floor
  let remaining_runs := target - first_runs
  (remaining_runs : ℚ) / remaining_overs

/-- Proves that the required run rate for the remaining 40 overs is 6.5 -/
theorem cricket_run_rate_proof :
  required_run_rate 50 10 (32/10) 292 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_proof_l539_53941


namespace NUMINAMATH_CALUDE_min_sum_on_parabola_l539_53974

theorem min_sum_on_parabola :
  ∀ n m : ℕ,
  m = 19 * n^2 - 98 * n →
  102 ≤ m + n :=
by sorry

end NUMINAMATH_CALUDE_min_sum_on_parabola_l539_53974


namespace NUMINAMATH_CALUDE_min_value_of_sum_l539_53902

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + 2 * x + y = 8) :
  x + y ≥ 2 * Real.sqrt 10 - 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ + 2 * x₀ + y₀ = 8 ∧ x₀ + y₀ = 2 * Real.sqrt 10 - 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l539_53902


namespace NUMINAMATH_CALUDE_rabbit_population_solution_l539_53932

/-- Represents the rabbit population in a park --/
structure RabbitPopulation where
  yesterday : ℕ
  brown : ℕ
  white : ℕ
  male : ℕ
  female : ℕ

/-- Conditions for the rabbit population problem --/
def rabbitProblem (pop : RabbitPopulation) : Prop :=
  -- Today's total is triple yesterday's
  pop.brown + pop.white = 3 * pop.yesterday
  -- 13 + 7 = 1/3 of brown rabbits
  ∧ 20 = pop.brown / 3
  -- White rabbits relation to brown
  ∧ pop.white = pop.brown / 2 - 2
  -- Male to female ratio is 5:3
  ∧ 5 * pop.female = 3 * pop.male
  -- Total rabbits is sum of male and female
  ∧ pop.male + pop.female = pop.brown + pop.white

/-- Theorem stating the solution to the rabbit population problem --/
theorem rabbit_population_solution :
  ∃ (pop : RabbitPopulation),
    rabbitProblem pop ∧ 
    pop.brown = 60 ∧ 
    pop.white = 28 ∧ 
    pop.male = 55 ∧ 
    pop.female = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_population_solution_l539_53932


namespace NUMINAMATH_CALUDE_danny_fish_tank_theorem_l539_53949

/-- The number of fish remaining after selling some from Danny's fish tank. -/
def remaining_fish (initial_guppies initial_angelfish initial_tiger_sharks initial_oscar_fish
                    sold_guppies sold_angelfish sold_tiger_sharks sold_oscar_fish : ℕ) : ℕ :=
  (initial_guppies - sold_guppies) +
  (initial_angelfish - sold_angelfish) +
  (initial_tiger_sharks - sold_tiger_sharks) +
  (initial_oscar_fish - sold_oscar_fish)

/-- Theorem stating the number of remaining fish in Danny's tank. -/
theorem danny_fish_tank_theorem :
  remaining_fish 94 76 89 58 30 48 17 24 = 198 := by
  sorry

end NUMINAMATH_CALUDE_danny_fish_tank_theorem_l539_53949
