import Mathlib

namespace NUMINAMATH_CALUDE_billy_apple_ratio_l4148_414868

/-- The number of apples Billy ate in a week -/
def total_apples : ℕ := 20

/-- The number of apples Billy ate on Monday -/
def monday_apples : ℕ := 2

/-- The number of apples Billy ate on Wednesday -/
def wednesday_apples : ℕ := 9

/-- The number of apples Billy ate on Friday -/
def friday_apples : ℕ := monday_apples / 2

/-- The number of apples Billy ate on Thursday -/
def thursday_apples : ℕ := 4 * friday_apples

/-- The number of apples Billy ate on Tuesday -/
def tuesday_apples : ℕ := total_apples - (monday_apples + wednesday_apples + thursday_apples + friday_apples)

/-- The ratio of apples eaten on Tuesday to Monday -/
def tuesday_to_monday_ratio : ℚ := tuesday_apples / monday_apples

theorem billy_apple_ratio : tuesday_to_monday_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_apple_ratio_l4148_414868


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l4148_414837

-- Define the quadratic equation
def quadratic_equation (m n x : ℝ) : Prop := x^2 + m*x + n = 0

-- Define the condition for two real roots
def has_two_real_roots (m n : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m n x₁ ∧ quadratic_equation m n x₂

-- Define the condition for negative roots
def has_negative_roots (m n : ℝ) : Prop := ∀ x : ℝ, quadratic_equation m n x → x < 0

-- Define the inequality
def inequality (m n t : ℝ) : Prop := t ≤ (m-1)^2 + (n-1)^2 + (m-n)^2

-- Theorem statement
theorem quadratic_equation_properties :
  ∀ m n : ℝ, has_two_real_roots m n →
  (∃ t : ℝ, (n = 3 - m ∧ has_negative_roots m n) → 2 ≤ m ∧ m < 3) ∧
  (∃ t_max : ℝ, t_max = 9/8 ∧ ∀ t : ℝ, inequality m n t → t ≤ t_max) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l4148_414837


namespace NUMINAMATH_CALUDE_jolene_bicycle_fundraising_l4148_414840

/-- Proves that Jolene raises enough money to buy the bicycle with some extra --/
theorem jolene_bicycle_fundraising (
  bicycle_cost : ℕ)
  (babysitting_families : ℕ)
  (babysitting_rate : ℕ)
  (car_washing_neighbors : ℕ)
  (car_washing_rate : ℕ)
  (dog_walking_count : ℕ)
  (dog_walking_rate : ℕ)
  (cash_gift : ℕ)
  (h1 : bicycle_cost = 250)
  (h2 : babysitting_families = 4)
  (h3 : babysitting_rate = 30)
  (h4 : car_washing_neighbors = 5)
  (h5 : car_washing_rate = 12)
  (h6 : dog_walking_count = 3)
  (h7 : dog_walking_rate = 15)
  (h8 : cash_gift = 40) :
  let total_raised := babysitting_families * babysitting_rate +
                      car_washing_neighbors * car_washing_rate +
                      dog_walking_count * dog_walking_rate +
                      cash_gift
  ∃ (extra : ℕ), total_raised = 265 ∧ total_raised > bicycle_cost ∧ extra = total_raised - bicycle_cost ∧ extra = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_jolene_bicycle_fundraising_l4148_414840


namespace NUMINAMATH_CALUDE_expression_value_l4148_414804

theorem expression_value : 50 * (50 - 5) - (50 * 50 - 5) = -245 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4148_414804


namespace NUMINAMATH_CALUDE_deborah_has_no_oranges_l4148_414874

-- Define the initial conditions
def initial_oranges : ℝ := 55.0
def susan_oranges : ℝ := 35.0
def final_oranges : ℝ := 90.0

-- Define Deborah's oranges as a variable
def deborah_oranges : ℝ := sorry

-- Theorem statement
theorem deborah_has_no_oranges :
  initial_oranges + susan_oranges + deborah_oranges = final_oranges →
  deborah_oranges = 0 := by sorry

end NUMINAMATH_CALUDE_deborah_has_no_oranges_l4148_414874


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l4148_414803

/-- The number of participants in the chess tournament. -/
def n : ℕ := 19

/-- The number of matches played in a round-robin tournament. -/
def matches_played (x : ℕ) : ℕ := x * (x - 1) / 2

/-- The number of matches played after three players dropped out. -/
def matches_after_dropout (x : ℕ) : ℕ := (x - 3) * (x - 4) / 2

/-- Theorem stating that the number of participants in the chess tournament is 19. -/
theorem chess_tournament_participants :
  matches_played n - matches_after_dropout n = 130 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l4148_414803


namespace NUMINAMATH_CALUDE_cos_double_angle_problem_l4148_414819

theorem cos_double_angle_problem (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) → 
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_problem_l4148_414819


namespace NUMINAMATH_CALUDE_min_value_inequality_l4148_414838

theorem min_value_inequality (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a * b * c = 1) :
  1 / (a^2 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l4148_414838


namespace NUMINAMATH_CALUDE_quadratic_properties_l4148_414897

def f (x : ℝ) := -2 * x^2 + 4 * x + 3

theorem quadratic_properties :
  (∀ x y, x < y → f x > f y) ∧
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (f 1 = 5) ∧
  (∀ x, x > 1 → ∀ y, y > x → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l4148_414897


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l4148_414866

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the line passing through the focus with slope k
def line (k x y : ℝ) : Prop := y = k*(x - 2)

-- Define the intersection points
def intersection (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ line k p.1 p.2}

-- Define the condition AF = 2FB
def point_condition (A B : ℝ × ℝ) : Prop :=
  (4 - A.1, -A.2) = (2*(B.1 - 4), 2*B.2)

-- Theorem statement
theorem parabola_intersection_theorem (k : ℝ) :
  ∃ (A B : ℝ × ℝ), A ∈ intersection k ∧ B ∈ intersection k ∧
  point_condition A B → |k| = 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l4148_414866


namespace NUMINAMATH_CALUDE_distinct_solutions_condition_l4148_414831

theorem distinct_solutions_condition (a x y : ℝ) : 
  x ≠ y → x = a - y^2 → y = a - x^2 → a > 3/4 := by
  sorry

end NUMINAMATH_CALUDE_distinct_solutions_condition_l4148_414831


namespace NUMINAMATH_CALUDE_multiple_of_1897_l4148_414822

theorem multiple_of_1897 (n : ℕ) : 1897 ∣ (2903^n - 803^n - 464^n + 261^n) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_1897_l4148_414822


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_monotonic_decreasing_on_open_interval_l4148_414826

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 2) ↔ (f_derivative x < 0) :=
by sorry

-- Main theorem
theorem f_monotonic_decreasing_on_open_interval :
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_monotonic_decreasing_on_open_interval_l4148_414826


namespace NUMINAMATH_CALUDE_remaining_numbers_average_l4148_414896

theorem remaining_numbers_average (n₁ n₂ n₃ n₄ n₅ n₆ : ℝ) : 
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆) / 6 = 2.5 →
  (n₁ + n₂) / 2 = 1.1 →
  (n₃ + n₄) / 2 = 1.4 →
  (n₅ + n₆) / 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_remaining_numbers_average_l4148_414896


namespace NUMINAMATH_CALUDE_max_correct_answers_l4148_414870

theorem max_correct_answers (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) :
  total_questions = 25 →
  correct_score = 5 →
  incorrect_score = -3 →
  total_score = 57 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct ≤ 12 ∧
    ∀ (c i u : ℕ),
      c + i + u = total_questions →
      correct_score * c + incorrect_score * i = total_score →
      c ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l4148_414870


namespace NUMINAMATH_CALUDE_fraction_comparison_l4148_414892

theorem fraction_comparison (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l4148_414892


namespace NUMINAMATH_CALUDE_vector_angle_condition_l4148_414809

/-- Given two vectors a and b in R², if the angle between them is acute,
    then the second component of b satisfies the given conditions. -/
theorem vector_angle_condition (m : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (1, m)
  -- The angle between a and b is acute
  (0 < (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) ∧ 
   (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) < 1) →
  m > -1/2 ∧ m ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_vector_angle_condition_l4148_414809


namespace NUMINAMATH_CALUDE_remainder_8_900_mod_29_l4148_414813

theorem remainder_8_900_mod_29 : (8 : Nat)^900 % 29 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8_900_mod_29_l4148_414813


namespace NUMINAMATH_CALUDE_find_number_l4148_414818

theorem find_number (x : ℝ) : ((4 * x - 28) / 7 + 12 = 36) → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l4148_414818


namespace NUMINAMATH_CALUDE_marketing_specialization_percentage_l4148_414880

theorem marketing_specialization_percentage
  (initial_finance : Real)
  (increased_finance : Real)
  (marketing_after_increase : Real)
  (h1 : initial_finance = 88)
  (h2 : increased_finance = 90)
  (h3 : marketing_after_increase = 43.333333333333336)
  (h4 : increased_finance - initial_finance = 2) :
  initial_finance + marketing_after_increase + 2 = 45.333333333333336 + 88 := by
  sorry

end NUMINAMATH_CALUDE_marketing_specialization_percentage_l4148_414880


namespace NUMINAMATH_CALUDE_hannahs_farm_animals_l4148_414899

/-- The total number of animals on Hannah's farm -/
def total_animals (num_pigs : ℕ) : ℕ :=
  let num_cows := 2 * num_pigs - 3
  let num_goats := num_cows + 6
  num_pigs + num_cows + num_goats

/-- Theorem stating the total number of animals on Hannah's farm -/
theorem hannahs_farm_animals :
  total_animals 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_farm_animals_l4148_414899


namespace NUMINAMATH_CALUDE_train_speed_l4148_414820

/-- Calculate the speed of a train given its length, platform length, and time to cross -/
theorem train_speed (train_length platform_length : ℝ) (time : ℝ) : 
  train_length = 250 →
  platform_length = 520 →
  time = 50.395968322534195 →
  ∃ (speed : ℝ), abs (speed - 54.99) < 0.01 ∧ 
    speed = (train_length + platform_length) / time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l4148_414820


namespace NUMINAMATH_CALUDE_product_expansion_l4148_414841

theorem product_expansion (x y : ℝ) :
  (3 * x + 4) * (2 * x + 6 * y + 7) = 6 * x^2 + 18 * x * y + 29 * x + 24 * y + 28 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l4148_414841


namespace NUMINAMATH_CALUDE_power_division_rule_l4148_414859

theorem power_division_rule (x : ℝ) (h : x ≠ 0) : x^10 / x^5 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l4148_414859


namespace NUMINAMATH_CALUDE_expression_simplification_l4148_414858

theorem expression_simplification (x : ℝ) (h : x = 1) : 
  (4 + (4 + x^2) / x) / ((x + 2) / x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4148_414858


namespace NUMINAMATH_CALUDE_football_team_right_handed_count_l4148_414847

theorem football_team_right_handed_count (total_players throwers : ℕ) : 
  total_players = 70 →
  throwers = 37 →
  (total_players - throwers) % 3 = 0 →
  (throwers + (total_players - throwers) * 2 / 3 = 59) :=
by
  sorry

end NUMINAMATH_CALUDE_football_team_right_handed_count_l4148_414847


namespace NUMINAMATH_CALUDE_cubic_roots_theorem_l4148_414801

theorem cubic_roots_theorem (a b c : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x - c = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a = 1 ∧ b = -2 ∧ c = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_theorem_l4148_414801


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l4148_414873

theorem salary_increase_percentage
  (total_employees : ℕ)
  (travel_allowance_percentage : ℚ)
  (no_increase_count : ℕ)
  (h1 : total_employees = 480)
  (h2 : travel_allowance_percentage = 1/5)
  (h3 : no_increase_count = 336) :
  (total_employees - no_increase_count - (travel_allowance_percentage * total_employees)) / total_employees = 1/10 :=
by sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l4148_414873


namespace NUMINAMATH_CALUDE_angle_subtraction_theorem_l4148_414845

/-- Represents an angle in degrees and minutes -/
structure AngleDM where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Addition of angles in degrees and minutes -/
def add_angles (a b : AngleDM) : AngleDM :=
  let total_minutes := a.minutes + b.minutes
  let carry_degrees := total_minutes / 60
  { degrees := a.degrees + b.degrees + carry_degrees,
    minutes := total_minutes % 60,
    valid := by sorry }

/-- Subtraction of angles in degrees and minutes -/
def sub_angles (a b : AngleDM) : AngleDM :=
  let total_minutes := a.degrees * 60 + a.minutes - (b.degrees * 60 + b.minutes)
  { degrees := total_minutes / 60,
    minutes := total_minutes % 60,
    valid := by sorry }

/-- The main theorem to prove -/
theorem angle_subtraction_theorem :
  sub_angles { degrees := 72, minutes := 24, valid := by sorry }
              { degrees := 28, minutes := 36, valid := by sorry } =
  { degrees := 43, minutes := 48, valid := by sorry } := by sorry

end NUMINAMATH_CALUDE_angle_subtraction_theorem_l4148_414845


namespace NUMINAMATH_CALUDE_tangent_slope_points_l4148_414800

theorem tangent_slope_points (x y : ℝ) : 
  y = x^3 ∧ (3 * x^2 = 3) ↔ (x = -1 ∧ y = -1) ∨ (x = 1 ∧ y = 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_points_l4148_414800


namespace NUMINAMATH_CALUDE_f_properties_l4148_414856

open Real

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

theorem f_properties (a : ℝ) :
  (f_deriv a 1 = 1) →
  (a = 2) ∧
  (∃ m b : ℝ, m = 9 ∧ b = 3 ∧ ∀ x y : ℝ, y = f a x → m*(-1) - y + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4148_414856


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4148_414839

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) + 2 * Complex.I * z = (4 : ℂ) - 6 * Complex.I * z ∧ z = Complex.I / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4148_414839


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l4148_414885

/-- The perimeter of an equilateral triangle with side length 13/12 meters is 3.25 meters. -/
theorem equilateral_triangle_perimeter :
  let side_length : ℚ := 13/12
  let perimeter : ℚ := 3 * side_length
  perimeter = 13/4 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l4148_414885


namespace NUMINAMATH_CALUDE_circle_radius_is_zero_l4148_414860

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -2)

/-- Theorem stating that the radius of the circle is 0 -/
theorem circle_radius_is_zero :
  ∀ x y : ℝ, circle_equation x y →
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_zero_l4148_414860


namespace NUMINAMATH_CALUDE_mixture_replacement_solution_l4148_414836

/-- Represents the mixture replacement problem -/
def MixtureReplacement (initial_A : ℝ) (initial_ratio_A : ℝ) (initial_ratio_B : ℝ) 
                       (final_ratio_A : ℝ) (final_ratio_B : ℝ) : Prop :=
  let initial_B := initial_A * initial_ratio_B / initial_ratio_A
  let replaced_amount := 
    (final_ratio_B * initial_A - final_ratio_A * initial_B) / 
    (final_ratio_A + final_ratio_B)
  replaced_amount = 40

/-- Theorem stating the solution to the mixture replacement problem -/
theorem mixture_replacement_solution :
  MixtureReplacement 32 4 1 2 3 := by
  sorry

end NUMINAMATH_CALUDE_mixture_replacement_solution_l4148_414836


namespace NUMINAMATH_CALUDE_larger_number_proof_l4148_414802

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365) 
  (h2 : L = 4 * S + 15) : 
  L = 1815 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l4148_414802


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l4148_414833

theorem power_fraction_simplification :
  (16 ^ 10 * 8 ^ 6) / (4 ^ 22) = 2 ^ 14 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l4148_414833


namespace NUMINAMATH_CALUDE_P_in_third_quadrant_iff_m_less_than_two_l4148_414854

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the third quadrant -/
def is_in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The point P with coordinates (-1, -2+m) -/
def P (m : ℝ) : Point :=
  ⟨-1, -2+m⟩

/-- Theorem stating the condition for P to be in the third quadrant -/
theorem P_in_third_quadrant_iff_m_less_than_two (m : ℝ) :
  is_in_third_quadrant (P m) ↔ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_P_in_third_quadrant_iff_m_less_than_two_l4148_414854


namespace NUMINAMATH_CALUDE_samuel_homework_time_l4148_414852

theorem samuel_homework_time (sarah_time : Real) (time_difference : Nat) : 
  sarah_time = 1.3 → time_difference = 48 → 
  ⌊sarah_time * 60 - time_difference⌋ = 30 := by
  sorry

end NUMINAMATH_CALUDE_samuel_homework_time_l4148_414852


namespace NUMINAMATH_CALUDE_remainder_problem_l4148_414884

theorem remainder_problem (d r : ℤ) : 
  d > 1 →
  1223 % d = r →
  1625 % d = r →
  2513 % d = r →
  d - r = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4148_414884


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l4148_414879

-- Define a point in 2D space
def point : ℝ × ℝ := (-8, 2)

-- Define the second quadrant
def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem point_in_second_quadrant :
  second_quadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l4148_414879


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l4148_414842

theorem similar_triangles_leg_sum (a b c d e f : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  a^2 + b^2 = c^2 →
  d^2 + e^2 = f^2 →
  (1/2) * a * b = 24 →
  (1/2) * d * e = 600 →
  c = 13 →
  (a / d)^2 = (b / e)^2 →
  d + e = 85 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l4148_414842


namespace NUMINAMATH_CALUDE_percentage_votes_against_l4148_414821

/-- Given a total number of votes and the difference between votes in favor and against,
    calculate the percentage of votes against the proposal. -/
theorem percentage_votes_against (total_votes : ℕ) (favor_minus_against : ℕ) 
    (h1 : total_votes = 340)
    (h2 : favor_minus_against = 68) : 
    (total_votes - favor_minus_against) / 2 / total_votes * 100 = 40 := by
  sorry

#check percentage_votes_against

end NUMINAMATH_CALUDE_percentage_votes_against_l4148_414821


namespace NUMINAMATH_CALUDE_pythagorean_consecutive_naturals_l4148_414815

theorem pythagorean_consecutive_naturals :
  ∀ x y z : ℕ, y = x + 1 → z = x + 2 →
  (z^2 = y^2 + x^2 ↔ x = 3 ∧ y = 4 ∧ z = 5) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_consecutive_naturals_l4148_414815


namespace NUMINAMATH_CALUDE_geometric_sequence_term_count_l4148_414808

theorem geometric_sequence_term_count (a₁ q aₙ : ℚ) (n : ℕ) :
  a₁ = 1/2 →
  q = 1/2 →
  aₙ = 1/32 →
  aₙ = a₁ * q^(n-1) →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_term_count_l4148_414808


namespace NUMINAMATH_CALUDE_f_composition_comparison_f_inverse_solutions_l4148_414853

noncomputable section

def f (x : ℝ) : ℝ :=
  if x < 1 then -2 * x + 1 else x^2 - 2 * x

theorem f_composition_comparison : f (f (-3)) > f (f 3) := by sorry

theorem f_inverse_solutions (x : ℝ) :
  f x = 1 ↔ x = 0 ∨ x = 1 + Real.sqrt 2 := by sorry

end

end NUMINAMATH_CALUDE_f_composition_comparison_f_inverse_solutions_l4148_414853


namespace NUMINAMATH_CALUDE_square_sum_given_squared_sum_and_product_l4148_414863

theorem square_sum_given_squared_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 49) 
  (h2 : x * y = 10) : 
  x^2 + y^2 = 29 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_squared_sum_and_product_l4148_414863


namespace NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l4148_414861

/-- The speed of a train excluding stoppages, given its speed including stoppages and stop duration. -/
theorem train_speed_excluding_stoppages 
  (speed_with_stops : ℝ) 
  (stop_duration : ℝ) 
  (h1 : speed_with_stops = 32) 
  (h2 : stop_duration = 20) : 
  speed_with_stops * 60 / (60 - stop_duration) = 48 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l4148_414861


namespace NUMINAMATH_CALUDE_unique_modular_solution_l4148_414844

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l4148_414844


namespace NUMINAMATH_CALUDE_polygon_sides_l4148_414895

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 5 * 360 → n = 12 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l4148_414895


namespace NUMINAMATH_CALUDE_seating_probability_is_two_sevenths_l4148_414886

/-- The number of boys to be seated -/
def num_boys : ℕ := 5

/-- The number of girls to be seated -/
def num_girls : ℕ := 6

/-- The total number of chairs -/
def total_chairs : ℕ := 11

/-- The probability of seating boys and girls with the given condition -/
def seating_probability : ℚ :=
  2 / 7

/-- Theorem stating that the probability of seating boys and girls
    such that there are no more boys than girls at any point is 2/7 -/
theorem seating_probability_is_two_sevenths :
  seating_probability = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_seating_probability_is_two_sevenths_l4148_414886


namespace NUMINAMATH_CALUDE_factor_expression_l4148_414891

theorem factor_expression (b : ℝ) : 26 * b^2 + 78 * b = 26 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4148_414891


namespace NUMINAMATH_CALUDE_function_value_at_inverse_l4148_414817

/-- Given a function f(x) = kx + 2/x^3 - 3 where k is a real number,
    if f(ln 6) = 1, then f(ln(1/6)) = -7 -/
theorem function_value_at_inverse (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ k * x + 2 / x^3 - 3
  f (Real.log 6) = 1 → f (Real.log (1/6)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_inverse_l4148_414817


namespace NUMINAMATH_CALUDE_milan_phone_bill_l4148_414887

/-- Calculates the number of minutes billed given the total bill, monthly fee, and cost per minute -/
def minutes_billed (total_bill monthly_fee cost_per_minute : ℚ) : ℚ :=
  (total_bill - monthly_fee) / cost_per_minute

/-- Proves that given the specified conditions, the number of minutes billed is 178 -/
theorem milan_phone_bill :
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let cost_per_minute : ℚ := 0.12
  minutes_billed total_bill monthly_fee cost_per_minute = 178 := by
  sorry

end NUMINAMATH_CALUDE_milan_phone_bill_l4148_414887


namespace NUMINAMATH_CALUDE_xyz_value_l4148_414810

variables (x y z : ℝ)

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
                   (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 6 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l4148_414810


namespace NUMINAMATH_CALUDE_number_difference_l4148_414827

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 24581) 
  (b_div_12 : ∃ k : ℕ, b = 12 * k) 
  (a_times_10 : a * 10 = b) : 
  b - a = 20801 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l4148_414827


namespace NUMINAMATH_CALUDE_grocery_payment_proof_l4148_414848

def grocery_cost (soup_cans bread_loaves cereal_boxes milk_gallons apples cookie_bags olive_oil : ℕ)
  (soup_price bread_price cereal_price milk_price apple_price cookie_price oil_price : ℕ) : ℕ :=
  soup_cans * soup_price + bread_loaves * bread_price + cereal_boxes * cereal_price +
  milk_gallons * milk_price + apples * apple_price + cookie_bags * cookie_price + olive_oil * oil_price

def min_bills_needed (total_cost bill_value : ℕ) : ℕ :=
  (total_cost + bill_value - 1) / bill_value

theorem grocery_payment_proof :
  let total_cost := grocery_cost 6 3 4 2 7 5 1 2 5 3 4 1 3 8
  min_bills_needed total_cost 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_grocery_payment_proof_l4148_414848


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l4148_414828

-- Define the universal set U
def U : Set ℝ := {x | x > 1}

-- Define set A
def A : Set ℝ := {x | x > 2}

-- State the theorem
theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l4148_414828


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_sum_10_l4148_414883

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_first_year_after_2010_with_sum_10 (year : ℕ) : Prop :=
  year > 2010 ∧
  sum_of_digits year = 10 ∧
  ∀ y, 2010 < y ∧ y < year → sum_of_digits y ≠ 10

theorem first_year_after_2010_with_sum_10 :
  is_first_year_after_2010_with_sum_10 2017 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_sum_10_l4148_414883


namespace NUMINAMATH_CALUDE_simplify_expression_l4148_414806

theorem simplify_expression : 0.72 * 0.43 + 0.12 * 0.34 = 0.3504 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4148_414806


namespace NUMINAMATH_CALUDE_silver_car_percentage_l4148_414876

/-- Calculates the percentage of silver cars in a car dealership's inventory after a new shipment. -/
theorem silver_car_percentage
  (initial_cars : ℕ)
  (initial_silver_percentage : ℚ)
  (new_shipment : ℕ)
  (new_non_silver_percentage : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percentage = 1/10)
  (h3 : new_shipment = 80)
  (h4 : new_non_silver_percentage = 1/4)
  : ∃ (result : ℚ), abs (result - 53333/100000) < 1/10000 ∧
    result = (initial_silver_percentage * initial_cars + (1 - new_non_silver_percentage) * new_shipment) / (initial_cars + new_shipment) :=
by sorry

end NUMINAMATH_CALUDE_silver_car_percentage_l4148_414876


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l4148_414850

theorem floor_expression_equals_eight (n : ℕ) (h : n = 2009) :
  ⌊((n + 1)^3 / ((n - 1) * n : ℝ)) - ((n - 1)^3 / (n * (n + 1) : ℝ))⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l4148_414850


namespace NUMINAMATH_CALUDE_jewelry_sales_problem_l4148_414882

/-- Represents the jewelry sales problem --/
theorem jewelry_sales_problem 
  (necklace_price bracelet_price earring_price ensemble_price : ℚ)
  (bracelets_sold earrings_sold ensembles_sold : ℕ)
  (total_revenue : ℚ)
  (h1 : necklace_price = 25)
  (h2 : bracelet_price = 15)
  (h3 : earring_price = 10)
  (h4 : ensemble_price = 45)
  (h5 : bracelets_sold = 10)
  (h6 : earrings_sold = 20)
  (h7 : ensembles_sold = 2)
  (h8 : total_revenue = 565) :
  ∃ (necklaces_sold : ℕ), 
    necklace_price * necklaces_sold + 
    bracelet_price * bracelets_sold + 
    earring_price * earrings_sold + 
    ensemble_price * ensembles_sold = total_revenue ∧
    necklaces_sold = 5 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_sales_problem_l4148_414882


namespace NUMINAMATH_CALUDE_oil_bill_problem_l4148_414812

/-- The oil bill problem -/
theorem oil_bill_problem (jan_bill feb_bill : ℝ) 
  (h1 : feb_bill / jan_bill = 5 / 4)
  (h2 : (feb_bill + 45) / jan_bill = 3 / 2) :
  jan_bill = 180 := by
  sorry

end NUMINAMATH_CALUDE_oil_bill_problem_l4148_414812


namespace NUMINAMATH_CALUDE_power_function_through_point_l4148_414816

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * (x ^ b)

-- Define the theorem
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l4148_414816


namespace NUMINAMATH_CALUDE_gcd_b_81_is_3_l4148_414811

theorem gcd_b_81_is_3 (a b : ℤ) : 
  (∃ (x : ℝ), x^2 = 2 ∧ (1 + x)^2012 = a + b * x) → Nat.gcd b.natAbs 81 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_b_81_is_3_l4148_414811


namespace NUMINAMATH_CALUDE_barbed_wire_height_l4148_414865

theorem barbed_wire_height (area : ℝ) (cost_per_meter : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) :
  area = 3136 →
  cost_per_meter = 1 →
  gate_width = 1 →
  num_gates = 2 →
  total_cost = 666 →
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let wire_length := perimeter - (↑num_gates * gate_width)
  let wire_cost := wire_length * cost_per_meter
  let height := (total_cost - wire_cost) / wire_length
  height = 2 := by sorry

end NUMINAMATH_CALUDE_barbed_wire_height_l4148_414865


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l4148_414872

/-- The total number of pencils after adding more -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Proof that the total number of pencils is 72 -/
theorem pencils_in_drawer : total_pencils 27 45 = 72 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l4148_414872


namespace NUMINAMATH_CALUDE_inequality_proof_l4148_414824

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b)/(a + b) + b^2 * (b - c)/(b + c) + c^2 * (c - a)/(c + a) ≥ 0 ∧
  (a^2 * (a - b)/(a + b) + b^2 * (b - c)/(b + c) + c^2 * (c - a)/(c + a) = 0 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4148_414824


namespace NUMINAMATH_CALUDE_cube_space_diagonal_l4148_414875

theorem cube_space_diagonal (surface_area : ℝ) (h : surface_area = 64) :
  let side_length := Real.sqrt (surface_area / 6)
  let space_diagonal := side_length * Real.sqrt 3
  space_diagonal = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_space_diagonal_l4148_414875


namespace NUMINAMATH_CALUDE_graph_x_squared_minus_y_squared_l4148_414857

/-- The graph of x^2 - y^2 = 0 consists of two intersecting lines in the real plane -/
theorem graph_x_squared_minus_y_squared (x y : ℝ) :
  x^2 - y^2 = 0 ↔ (y = x ∨ y = -x) :=
sorry

end NUMINAMATH_CALUDE_graph_x_squared_minus_y_squared_l4148_414857


namespace NUMINAMATH_CALUDE_division_simplification_l4148_414871

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  6 * x^3 * y^2 / (-3 * x * y) = -2 * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l4148_414871


namespace NUMINAMATH_CALUDE_coffee_thermoses_count_l4148_414881

-- Define the conversion factor from gallons to pints
def gallons_to_pints : ℚ := 8

-- Define the total amount of coffee in gallons
def total_coffee_gallons : ℚ := 9/2

-- Define the number of thermoses Genevieve drank
def thermoses_consumed : ℕ := 3

-- Define the amount of coffee Genevieve consumed in pints
def coffee_consumed_pints : ℕ := 6

-- Theorem to prove
theorem coffee_thermoses_count : 
  (total_coffee_gallons * gallons_to_pints) / (coffee_consumed_pints / thermoses_consumed) = 18 := by
  sorry

end NUMINAMATH_CALUDE_coffee_thermoses_count_l4148_414881


namespace NUMINAMATH_CALUDE_min_abs_sum_of_quadratic_roots_l4148_414862

theorem min_abs_sum_of_quadratic_roots : ∃ (α β : ℝ), 
  (∀ y : ℝ, y^2 - 6*y + 5 = 0 ↔ y = α ∨ y = β) ∧
  (∀ x : ℝ, |x - α| + |x - β| ≥ 4) ∧
  (∃ x : ℝ, |x - α| + |x - β| = 4) := by
sorry

end NUMINAMATH_CALUDE_min_abs_sum_of_quadratic_roots_l4148_414862


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4148_414825

theorem complex_equation_solution (c d x : ℂ) (i : ℂ) : 
  c * d = x - 5 * i → 
  Complex.abs c = 3 →
  Complex.abs d = Real.sqrt 50 →
  x = 5 * Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4148_414825


namespace NUMINAMATH_CALUDE_r_value_when_n_is_three_l4148_414846

theorem r_value_when_n_is_three :
  let n : ℕ := 3
  let s : ℕ := 2^n - 1
  let r : ℕ := 2^s + s
  r = 135 := by sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_three_l4148_414846


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l4148_414849

/-- Represents the earnings of investors a, b, and c -/
structure Earnings where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Calculates the total earnings of a, b, and c -/
def total_earnings (e : Earnings) : ℚ :=
  e.a + e.b + e.c

/-- Theorem stating the total earnings given the investment and return ratios -/
theorem total_earnings_theorem (x y : ℚ) (h1 : x > 0) (h2 : y > 0) : 
  let e := Earnings.mk (18*x*y) (20*x*y) (20*x*y)
  2*x*y = 120 → total_earnings e = 3480 := by
  sorry

#check total_earnings_theorem

end NUMINAMATH_CALUDE_total_earnings_theorem_l4148_414849


namespace NUMINAMATH_CALUDE_martin_distance_l4148_414843

/-- The distance traveled by Martin -/
def distance : ℝ := 72.0

/-- Martin's driving speed in miles per hour -/
def speed : ℝ := 12.0

/-- Time taken for Martin's journey in hours -/
def time : ℝ := 6.0

/-- Theorem stating that the distance Martin traveled is equal to his speed multiplied by the time taken -/
theorem martin_distance : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_martin_distance_l4148_414843


namespace NUMINAMATH_CALUDE_art_supplies_theorem_l4148_414835

def art_supplies_problem (total_spent canvas_cost paint_cost_ratio easel_cost : ℚ) : Prop :=
  let paint_cost := canvas_cost * paint_cost_ratio
  let other_items_cost := canvas_cost + paint_cost + easel_cost
  let paintbrush_cost := total_spent - other_items_cost
  paintbrush_cost = 15

theorem art_supplies_theorem :
  art_supplies_problem 90 40 (1/2) 15 := by
  sorry

end NUMINAMATH_CALUDE_art_supplies_theorem_l4148_414835


namespace NUMINAMATH_CALUDE_binomial_n_equals_10_l4148_414867

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a binomial distribution with p = 0.8 and variance 1.6, n = 10 -/
theorem binomial_n_equals_10 :
  ∀ X : BinomialRV, X.p = 0.8 → variance X = 1.6 → X.n = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_n_equals_10_l4148_414867


namespace NUMINAMATH_CALUDE_seventh_observation_value_l4148_414894

theorem seventh_observation_value
  (n : ℕ) -- number of initial observations
  (initial_avg : ℚ) -- initial average
  (new_avg : ℚ) -- new average after adding one observation
  (h1 : n = 6) -- there are 6 initial observations
  (h2 : initial_avg = 15) -- the initial average is 15
  (h3 : new_avg = initial_avg - 1) -- the new average is decreased by 1
  : (n + 1) * new_avg - n * initial_avg = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_observation_value_l4148_414894


namespace NUMINAMATH_CALUDE_find_original_number_l4148_414898

/-- A four-digit number is between 1000 and 9999 inclusive -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem find_original_number (N : ℕ) (h1 : FourDigitNumber N) (h2 : N - 3 - 57 = 1819) : N = 1879 := by
  sorry

end NUMINAMATH_CALUDE_find_original_number_l4148_414898


namespace NUMINAMATH_CALUDE_translation_office_staff_count_l4148_414888

/-- The number of people working at a translation office -/
def translation_office_staff : ℕ :=
  let english_only : ℕ := 8
  let german_only : ℕ := 8
  let russian_only : ℕ := 8
  let english_german : ℕ := 1
  let german_russian : ℕ := 2
  let english_russian : ℕ := 3
  let all_three : ℕ := 1
  english_only + german_only + russian_only + english_german + german_russian + english_russian + all_three

/-- Theorem stating the number of people working at the translation office -/
theorem translation_office_staff_count : translation_office_staff = 31 := by
  sorry

end NUMINAMATH_CALUDE_translation_office_staff_count_l4148_414888


namespace NUMINAMATH_CALUDE_square_root_squared_l4148_414877

theorem square_root_squared (a : ℝ) (ha : 0 ≤ a) : (Real.sqrt a) ^ 2 = a := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_l4148_414877


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4148_414805

/-- Given two parallel vectors a = (-3, 2) and b = (1, x), prove that x = -2/3 -/
theorem parallel_vectors_x_value (x : ℚ) : 
  let a : ℚ × ℚ := (-3, 2)
  let b : ℚ × ℚ := (1, x)
  (∃ (k : ℚ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2) → x = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4148_414805


namespace NUMINAMATH_CALUDE_fraction_simplification_positive_integer_solutions_l4148_414851

-- Problem 1
theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x)) = x^2 / (x - 1) := by
  sorry

-- Problem 2
def inequality_system (x : ℝ) : Prop :=
  (2*x + 1) / 3 - (5*x - 1) / 2 < 1 ∧ 5*x - 1 < 3*(x + 2)

theorem positive_integer_solutions :
  {x : ℕ | inequality_system x} = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_positive_integer_solutions_l4148_414851


namespace NUMINAMATH_CALUDE_square_screen_diagonal_l4148_414855

theorem square_screen_diagonal (d : ℝ) : 
  d > 0 → 
  (d / Real.sqrt 2) ^ 2 = 20 ^ 2 + 42 → 
  d = Real.sqrt 884 := by
  sorry

end NUMINAMATH_CALUDE_square_screen_diagonal_l4148_414855


namespace NUMINAMATH_CALUDE_point_on_line_l4148_414830

/-- For any point (m,n) on the line y = 2x + 1, 2m - n = -1 -/
theorem point_on_line (m n : ℝ) : n = 2 * m + 1 → 2 * m - n = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l4148_414830


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_l4148_414814

-- Define the two lines
def l₁ (x y : ℝ) : Prop := x + 8*y + 7 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := x + y + 1 = 0

theorem intersection_and_parallel_line :
  -- Part 1: Prove that (1, -1) is the intersection point
  (l₁ 1 (-1) ∧ l₂ 1 (-1)) ∧
  -- Part 2: Prove that x + y = 0 is the equation of the line passing through
  -- the intersection point and parallel to x + y + 1 = 0
  (∃ c : ℝ, ∀ x y : ℝ, (l₁ x y ∧ l₂ x y) → x + y = c) ∧
  (∀ x y : ℝ, (x + y = 0) ↔ (∃ k : ℝ, x = 1 + k ∧ y = -1 - k ∧ parallel_line (1 + k) (-1 - k))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_line_l4148_414814


namespace NUMINAMATH_CALUDE_triangle_cannot_have_two_right_angles_l4148_414893

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ

-- Define properties of a triangle
def Triangle.sumOfAngles (t : Triangle) : ℝ := t.angles 0 + t.angles 1 + t.angles 2

-- Define a right angle
def rightAngle : ℝ := 90

-- Theorem: A triangle cannot have two right angles
theorem triangle_cannot_have_two_right_angles (t : Triangle) :
  (t.angles 0 = rightAngle ∧ t.angles 1 = rightAngle) →
  t.sumOfAngles ≠ 180 :=
sorry

end NUMINAMATH_CALUDE_triangle_cannot_have_two_right_angles_l4148_414893


namespace NUMINAMATH_CALUDE_empire_state_building_race_l4148_414889

-- Define the total number of steps
def total_steps : ℕ := 1576

-- Define the total time in seconds
def total_time_seconds : ℕ := 11 * 60 + 57

-- Define the function to calculate steps per minute
def steps_per_minute (steps : ℕ) (time_seconds : ℕ) : ℚ :=
  (steps : ℚ) / ((time_seconds : ℚ) / 60)

-- Theorem statement
theorem empire_state_building_race :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |steps_per_minute total_steps total_time_seconds - 130| < ε :=
sorry

end NUMINAMATH_CALUDE_empire_state_building_race_l4148_414889


namespace NUMINAMATH_CALUDE_sequence_difference_theorem_l4148_414807

theorem sequence_difference_theorem (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, a (n + 1) ≤ 2 * n) 
  (h3 : ∀ n : ℕ, a n < a (n + 1)) :
  ∀ n : ℕ, ∃ p q : ℕ, a p - a q = n :=
by sorry

end NUMINAMATH_CALUDE_sequence_difference_theorem_l4148_414807


namespace NUMINAMATH_CALUDE_brad_age_l4148_414890

/-- Given the ages and relationships between Jaymee, Shara, and Brad, prove Brad's age -/
theorem brad_age (shara_age : ℕ) (jaymee_age : ℕ) (brad_age : ℕ) : 
  shara_age = 10 →
  jaymee_age = 2 * shara_age + 2 →
  brad_age = (shara_age + jaymee_age) / 2 - 3 →
  brad_age = 13 :=
by sorry

end NUMINAMATH_CALUDE_brad_age_l4148_414890


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4148_414834

theorem complex_magnitude_problem (a b c : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : Complex.abs (a + b + c) = 1)
  (h5 : Complex.abs (a - b) = Complex.abs (a - c))
  (h6 : b ≠ c) :
  Complex.abs (a + b) * Complex.abs (a + c) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4148_414834


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4148_414832

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4148_414832


namespace NUMINAMATH_CALUDE_rectangle_area_l4148_414829

/-- Proves that a rectangle with a perimeter of 176 inches and a length 8 inches more than its width has an area of 1920 square inches. -/
theorem rectangle_area (w : ℝ) (l : ℝ) : 
  (2 * l + 2 * w = 176) →  -- Perimeter condition
  (l = w + 8) →            -- Length-width relation
  (l * w = 1920)           -- Area to be proved
  := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l4148_414829


namespace NUMINAMATH_CALUDE_reptile_house_count_l4148_414864

/-- The number of animals in the Rain Forest exhibit -/
def rain_forest_animals : ℕ := 7

/-- The number of animals in the Reptile House -/
def reptile_house_animals : ℕ := 3 * rain_forest_animals - 5

theorem reptile_house_count : reptile_house_animals = 16 := by
  sorry

end NUMINAMATH_CALUDE_reptile_house_count_l4148_414864


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l4148_414878

theorem pencil_pen_cost (pencil_cost pen_cost : ℝ) 
  (h1 : 3 * pencil_cost + 4 * pen_cost = 5.20)
  (h2 : 4 * pencil_cost + 3 * pen_cost = 4.90) : 
  pencil_cost + 3 * pen_cost = 3.1857 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l4148_414878


namespace NUMINAMATH_CALUDE_twenty_two_students_remain_l4148_414823

/-- The number of remaining students after some leave early -/
def remaining_students (total_groups : ℕ) (students_per_group : ℕ) (students_who_left : ℕ) : ℕ :=
  total_groups * students_per_group - students_who_left

/-- Theorem stating that given 3 groups of 8 students with 2 leaving early, 22 students remain -/
theorem twenty_two_students_remain :
  remaining_students 3 8 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_twenty_two_students_remain_l4148_414823


namespace NUMINAMATH_CALUDE_total_amount_l4148_414869

/-- The ratio of money distribution among w, x, y, and z -/
structure MoneyDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  ratio_condition : x = 0.7 * w ∧ y = 0.5 * w ∧ z = 0.3 * w

/-- The problem statement -/
theorem total_amount (d : MoneyDistribution) (h : d.y = 90) :
  d.w + d.x + d.y + d.z = 450 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_l4148_414869
