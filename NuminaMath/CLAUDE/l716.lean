import Mathlib

namespace NUMINAMATH_CALUDE_shelter_cats_l716_71613

theorem shelter_cats (total : ℕ) (tuna : ℕ) (chicken : ℕ) (both : ℕ) 
  (h1 : total = 75)
  (h2 : tuna = 18)
  (h3 : chicken = 55)
  (h4 : both = 10) :
  total - (tuna + chicken - both) = 12 :=
by sorry

end NUMINAMATH_CALUDE_shelter_cats_l716_71613


namespace NUMINAMATH_CALUDE_expression_evaluation_l716_71675

theorem expression_evaluation : 
  1 / 2^2 + ((2 / 3^3 * (3 / 2)^2) + 4^(1/2)) - 8 / (4^2 - 3^2) = 107/84 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l716_71675


namespace NUMINAMATH_CALUDE_cubic_sum_from_linear_and_quadratic_sum_l716_71665

theorem cubic_sum_from_linear_and_quadratic_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_from_linear_and_quadratic_sum_l716_71665


namespace NUMINAMATH_CALUDE_f_definition_f_max_min_on_interval_l716_71604

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x => 2 / (x - 1)

-- Theorem for the function definition
theorem f_definition (x : ℝ) (h : x ≠ 1) : 
  f ((x - 1) / (x + 1)) = -x - 1 := by sorry

-- Theorem for the maximum and minimum values
theorem f_max_min_on_interval : 
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc 2 6, f x ≤ max ∧ min ≤ f x) ∧ 
  (∃ x₁ ∈ Set.Icc 2 6, f x₁ = max) ∧ 
  (∃ x₂ ∈ Set.Icc 2 6, f x₂ = min) ∧ 
  max = 2 ∧ min = 2/5 := by sorry

end NUMINAMATH_CALUDE_f_definition_f_max_min_on_interval_l716_71604


namespace NUMINAMATH_CALUDE_video_games_expense_is_11_l716_71668

def total_allowance : ℚ := 60

def books_fraction : ℚ := 1/4
def snacks_fraction : ℚ := 1/6
def toys_fraction : ℚ := 2/5

def video_games_expense : ℚ := total_allowance - (books_fraction * total_allowance + snacks_fraction * total_allowance + toys_fraction * total_allowance)

theorem video_games_expense_is_11 : video_games_expense = 11 := by
  sorry

end NUMINAMATH_CALUDE_video_games_expense_is_11_l716_71668


namespace NUMINAMATH_CALUDE_only_set_D_is_right_triangle_l716_71600

-- Define a function to check if three numbers form a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the sets of line segments
def set_A : (ℝ × ℝ × ℝ) := (3, 5, 7)
def set_B : (ℝ × ℝ × ℝ) := (4, 6, 8)
def set_C : (ℝ × ℝ × ℝ) := (5, 7, 9)
def set_D : (ℝ × ℝ × ℝ) := (6, 8, 10)

-- Theorem stating that only set D forms a right triangle
theorem only_set_D_is_right_triangle :
  ¬(is_right_triangle set_A.1 set_A.2.1 set_A.2.2) ∧
  ¬(is_right_triangle set_B.1 set_B.2.1 set_B.2.2) ∧
  ¬(is_right_triangle set_C.1 set_C.2.1 set_C.2.2) ∧
  (is_right_triangle set_D.1 set_D.2.1 set_D.2.2) :=
by sorry


end NUMINAMATH_CALUDE_only_set_D_is_right_triangle_l716_71600


namespace NUMINAMATH_CALUDE_arithmetic_geometric_general_term_l716_71606

-- Define the arithmetic-geometric sequence
def arithmetic_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

-- Define the conditions
def conditions (a : ℕ → ℝ) : Prop :=
  a 2 = 6 ∧ 6 * a 1 + a 3 = 30

-- Theorem statement
theorem arithmetic_geometric_general_term (a : ℕ → ℝ) :
  arithmetic_geometric_seq a → conditions a →
  (∀ n : ℕ, a n = 3 * 3^(n - 1)) ∨ (∀ n : ℕ, a n = 2 * 2^(n - 1)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_general_term_l716_71606


namespace NUMINAMATH_CALUDE_bike_price_l716_71657

theorem bike_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 ∧ upfront_percentage = 20 ∧ upfront_payment = (upfront_percentage / 100) * total_price →
  total_price = 1200 :=
by sorry

end NUMINAMATH_CALUDE_bike_price_l716_71657


namespace NUMINAMATH_CALUDE_rebecca_eggs_marbles_difference_l716_71655

theorem rebecca_eggs_marbles_difference : 
  ∀ (eggs marbles : ℕ), 
  eggs = 20 → marbles = 6 → eggs - marbles = 14 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_marbles_difference_l716_71655


namespace NUMINAMATH_CALUDE_smallest_angle_of_triangle_l716_71699

theorem smallest_angle_of_triangle (y : ℝ) (h : y + 40 + 70 = 180) :
  min (min 40 70) y = 40 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_of_triangle_l716_71699


namespace NUMINAMATH_CALUDE_book_cost_theorem_l716_71622

theorem book_cost_theorem (selling_price_1 selling_price_2 : ℝ) 
  (h1 : selling_price_1 = 340)
  (h2 : selling_price_2 = 350)
  (h3 : ∃ (profit : ℝ), selling_price_1 = cost + profit ∧ 
                         selling_price_2 = cost + (1.05 * profit)) :
  cost = 140 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_theorem_l716_71622


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l716_71648

theorem complex_number_quadrant (z : ℂ) (h : z * (2 + Complex.I) = 3 - Complex.I) :
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l716_71648


namespace NUMINAMATH_CALUDE_true_discount_is_36_l716_71640

/-- Given a banker's discount and sum due, calculate the true discount -/
def true_discount (BD : ℚ) (SD : ℚ) : ℚ :=
  BD / (1 + BD / SD)

/-- Theorem stating that for the given banker's discount and sum due, the true discount is 36 -/
theorem true_discount_is_36 :
  true_discount 42 252 = 36 := by
  sorry

end NUMINAMATH_CALUDE_true_discount_is_36_l716_71640


namespace NUMINAMATH_CALUDE_inequality_solution_set_l716_71633

theorem inequality_solution_set (x : ℝ) : (x - 2) * (3 - x) > 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l716_71633


namespace NUMINAMATH_CALUDE_ceiling_sqrt_156_l716_71601

theorem ceiling_sqrt_156 : ⌈Real.sqrt 156⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_156_l716_71601


namespace NUMINAMATH_CALUDE_trigonometric_equation_l716_71695

theorem trigonometric_equation (α : Real) 
  (h : (5 * Real.sin α - Real.cos α) / (Real.cos α + Real.sin α) = 1) : 
  Real.tan α = 1/2 ∧ 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) + Real.sin α * Real.cos α = 17/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l716_71695


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l716_71630

theorem sin_cos_sum_equals_half :
  Real.sin (13 * π / 180) * Real.cos (343 * π / 180) +
  Real.cos (13 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l716_71630


namespace NUMINAMATH_CALUDE_complex_equation_sum_l716_71670

theorem complex_equation_sum (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (2 + b * Complex.I) / (1 - Complex.I) = a * Complex.I →
  a + b = 4 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l716_71670


namespace NUMINAMATH_CALUDE_group_size_proof_l716_71696

theorem group_size_proof (n : ℕ) (W : ℝ) : 
  (W + 25) / n - W / n = 2.5 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l716_71696


namespace NUMINAMATH_CALUDE_smallest_num_neighbors_correct_l716_71673

/-- The number of points on the circumference of the circle -/
def num_points : ℕ := 2005

/-- The maximum angle (in degrees) that a chord can subtend at the center for two points to be considered neighbors -/
def max_angle : ℝ := 10

/-- Definition of the smallest number of pairs of neighbors function -/
def smallest_num_neighbors (n : ℕ) (θ : ℝ) : ℕ :=
  25 * (Nat.choose 57 2) + 10 * (Nat.choose 58 2)

/-- Theorem stating that the smallest number of pairs of neighbors for the given conditions is correct -/
theorem smallest_num_neighbors_correct :
  smallest_num_neighbors num_points max_angle =
  25 * (Nat.choose 57 2) + 10 * (Nat.choose 58 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_num_neighbors_correct_l716_71673


namespace NUMINAMATH_CALUDE_ratio_inequality_l716_71618

theorem ratio_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 2*b + 3*c)^2 / (a^2 + 2*b^2 + 3*c^2) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_inequality_l716_71618


namespace NUMINAMATH_CALUDE_carly_backstroke_practice_days_l716_71645

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of weeks in a month -/
def weeksInMonth : ℕ := 4

/-- Represents the total hours Carly practices swimming in a month -/
def totalPracticeHours : ℕ := 96

/-- Represents the hours Carly practices butterfly stroke per day -/
def butterflyHoursPerDay : ℕ := 3

/-- Represents the days Carly practices butterfly stroke per week -/
def butterflyDaysPerWeek : ℕ := 4

/-- Represents the hours Carly practices backstroke per day -/
def backstrokeHoursPerDay : ℕ := 2

/-- Theorem stating that Carly practices backstroke 6 days a week -/
theorem carly_backstroke_practice_days :
  ∃ (backstrokeDaysPerWeek : ℕ),
    backstrokeDaysPerWeek * backstrokeHoursPerDay * weeksInMonth +
    butterflyDaysPerWeek * butterflyHoursPerDay * weeksInMonth = totalPracticeHours ∧
    backstrokeDaysPerWeek = 6 :=
by sorry

end NUMINAMATH_CALUDE_carly_backstroke_practice_days_l716_71645


namespace NUMINAMATH_CALUDE_condensed_milk_higher_caloric_value_l716_71659

theorem condensed_milk_higher_caloric_value (a b c : ℝ) : 
  (3*a + 4*b + 2*c > 2*a + 3*b + 4*c) → 
  (3*a + 4*b + 2*c > 4*a + 2*b + 3*c) → 
  b > c := by
sorry

end NUMINAMATH_CALUDE_condensed_milk_higher_caloric_value_l716_71659


namespace NUMINAMATH_CALUDE_high_school_ten_season_games_l716_71639

/-- Represents a basketball conference -/
structure BasketballConference where
  teamCount : ℕ
  intraConferenceGamesPerPair : ℕ
  nonConferenceGamesPerTeam : ℕ

/-- Calculates the total number of games in a season for a given basketball conference -/
def totalSeasonGames (conf : BasketballConference) : ℕ :=
  let intraConferenceGames := conf.teamCount.choose 2 * conf.intraConferenceGamesPerPair
  let nonConferenceGames := conf.teamCount * conf.nonConferenceGamesPerTeam
  intraConferenceGames + nonConferenceGames

/-- The High School Ten basketball conference -/
def highSchoolTen : BasketballConference :=
  { teamCount := 10
  , intraConferenceGamesPerPair := 2
  , nonConferenceGamesPerTeam := 6 }

theorem high_school_ten_season_games :
  totalSeasonGames highSchoolTen = 150 := by
  sorry

end NUMINAMATH_CALUDE_high_school_ten_season_games_l716_71639


namespace NUMINAMATH_CALUDE_nadia_mistakes_l716_71678

/-- Calculates the number of mistakes made by a piano player given their error rate, playing speed, and duration of play. -/
def calculate_mistakes (mistakes_per_block : ℕ) (notes_per_block : ℕ) (notes_per_minute : ℕ) (minutes_played : ℕ) : ℕ :=
  let total_notes := notes_per_minute * minutes_played
  let num_blocks := total_notes / notes_per_block
  num_blocks * mistakes_per_block

/-- Theorem stating that under the given conditions, Nadia will make 36 mistakes on average when playing for 8 minutes. -/
theorem nadia_mistakes :
  calculate_mistakes 3 40 60 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nadia_mistakes_l716_71678


namespace NUMINAMATH_CALUDE_solve_equations_l716_71623

theorem solve_equations :
  (∃ x : ℝ, 4 * x = 2 * x + 6 ∧ x = 3) ∧
  (∃ x : ℝ, 3 * x + 5 = 6 * x - 1 ∧ x = 2) ∧
  (∃ x : ℝ, 3 * x - 2 * (x - 1) = 2 + 3 * (4 - x) ∧ x = 3) ∧
  (∃ x : ℝ, (x - 3) / 5 - (x + 4) / 2 = -2 ∧ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l716_71623


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l716_71643

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (-1, x) (1, 2) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l716_71643


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l716_71698

theorem quadratic_roots_difference_squared :
  ∀ a b : ℝ,
  (6 * a^2 + 13 * a - 28 = 0) →
  (6 * b^2 + 13 * b - 28 = 0) →
  (a - b)^2 = 841 / 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l716_71698


namespace NUMINAMATH_CALUDE_prob_equal_prob_first_value_prob_second_value_l716_71628

/-- Represents the number of classes -/
def total_classes : ℕ := 10

/-- Represents the specific class we're interested in (Class 5) -/
def target_class : ℕ := 5

/-- The probability of drawing the target class first -/
def prob_first : ℚ := 1 / total_classes

/-- The probability of drawing the target class second -/
def prob_second : ℚ := 1 / total_classes

/-- Theorem stating that the probabilities of drawing the target class first and second are equal -/
theorem prob_equal : prob_first = prob_second := by sorry

/-- Theorem stating that the probability of drawing the target class first is 1/10 -/
theorem prob_first_value : prob_first = 1 / 10 := by sorry

/-- Theorem stating that the probability of drawing the target class second is 1/10 -/
theorem prob_second_value : prob_second = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_prob_equal_prob_first_value_prob_second_value_l716_71628


namespace NUMINAMATH_CALUDE_action_figures_removed_l716_71653

theorem action_figures_removed (initial : ℕ) (added : ℕ) (final : ℕ) : 
  initial = 15 → added = 2 → final = 10 → initial + added - final = 7 := by
sorry

end NUMINAMATH_CALUDE_action_figures_removed_l716_71653


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l716_71625

-- Define the inverse relationship between y and x^2
def inverse_relation (k : ℝ) (x y : ℝ) : Prop := y = k / (x^2)

-- Theorem statement
theorem inverse_variation_problem (k : ℝ) :
  (inverse_relation k 1 8) →
  (inverse_relation k 4 0.5) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l716_71625


namespace NUMINAMATH_CALUDE_f_neg_five_l716_71621

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 + 1

-- State the theorem
theorem f_neg_five (a b : ℝ) (h : f a b 5 = 7) : f a b (-5) = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_five_l716_71621


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l716_71688

/-- Given a group of 6 persons where one person is replaced by a new person weighing 79.8 kg,
    and the average weight increases by 1.8 kg, prove that the replaced person weighed 69 kg. -/
theorem weight_of_replaced_person
  (initial_count : ℕ)
  (new_person_weight : ℝ)
  (average_increase : ℝ)
  (h1 : initial_count = 6)
  (h2 : new_person_weight = 79.8)
  (h3 : average_increase = 1.8) :
  ∃ (replaced_weight : ℝ),
    replaced_weight = 69 ∧
    new_person_weight = replaced_weight + (initial_count : ℝ) * average_increase :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l716_71688


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l716_71693

theorem arc_length_of_sector (θ : Real) (r : Real) (L : Real) : 
  θ = 120 → r = 3/2 → L = θ / 360 * (2 * Real.pi * r) → L = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l716_71693


namespace NUMINAMATH_CALUDE_undefined_values_sum_l716_71691

theorem undefined_values_sum (f : ℝ → ℝ) (h : f = λ x => 5*x / (3*x^2 - 9*x + 6)) : 
  ∃ C D : ℝ, (3*C^2 - 9*C + 6 = 0) ∧ (3*D^2 - 9*D + 6 = 0) ∧ (C + D = 3) := by
  sorry

end NUMINAMATH_CALUDE_undefined_values_sum_l716_71691


namespace NUMINAMATH_CALUDE_greatest_K_inequality_l716_71626

theorem greatest_K_inequality : 
  ∃ (K : ℝ), K = 16 ∧ 
  (∀ (u v w : ℝ), u > 0 → v > 0 → w > 0 → u^2 > 4*v*w → 
    (u^2 - 4*v*w)^2 > K*(2*v^2 - u*w)*(2*w^2 - u*v)) ∧
  (∀ (K' : ℝ), K' > K → 
    ∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ u^2 > 4*v*w ∧ 
      (u^2 - 4*v*w)^2 ≤ K'*(2*v^2 - u*w)*(2*w^2 - u*v)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_K_inequality_l716_71626


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l716_71684

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 25) 
  (h2 : failed_english = 35) 
  (h3 : failed_both = 40) : 
  100 - (failed_hindi + failed_english - failed_both) = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l716_71684


namespace NUMINAMATH_CALUDE_constant_term_expansion_l716_71617

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = (x - 2 + 1/x)^4) ∧ 
  (∃ c : ℝ, ∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ 
  (∃ c : ℝ, ∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ c = 70 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l716_71617


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l716_71689

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - 2*x + m = 0

-- Theorem statement
theorem quadratic_roots_theorem (m : ℝ) (h : m < 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m) ∧
  (quadratic_equation (-1) m → m = -3 ∧ quadratic_equation 3 m) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l716_71689


namespace NUMINAMATH_CALUDE_ladder_problem_l716_71619

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 8.5)
  (h2 : height = 7.5) :
  ∃ base : ℝ, base = 4 ∧ base^2 + height^2 = ladder_length^2 :=
sorry

end NUMINAMATH_CALUDE_ladder_problem_l716_71619


namespace NUMINAMATH_CALUDE_shopping_cost_other_goods_l716_71609

def tuna_packs : ℕ := 5
def tuna_price : ℚ := 2
def water_bottles : ℕ := 4
def water_price : ℚ := 3/2
def discount_rate : ℚ := 1/10
def paid_after_discount : ℚ := 56
def conversion_rate : ℚ := 3/2

theorem shopping_cost_other_goods :
  let total_cost := paid_after_discount / (1 - discount_rate)
  let tuna_water_cost := tuna_packs * tuna_price + water_bottles * water_price
  let other_goods_local := total_cost - tuna_water_cost
  let other_goods_home := other_goods_local / conversion_rate
  other_goods_home = 30.81 := by sorry

end NUMINAMATH_CALUDE_shopping_cost_other_goods_l716_71609


namespace NUMINAMATH_CALUDE_alien_abduction_percentage_l716_71612

/-- The number of people initially abducted by the alien -/
def initial_abducted : ℕ := 200

/-- The number of people taken away after returning some -/
def taken_away : ℕ := 40

/-- The number of people left on Earth after returning some and taking away others -/
def left_on_earth : ℕ := 160

/-- The percentage of people returned by the alien -/
def percentage_returned : ℚ := (left_on_earth : ℚ) / (initial_abducted : ℚ) * 100

theorem alien_abduction_percentage :
  percentage_returned = 80 := by sorry

end NUMINAMATH_CALUDE_alien_abduction_percentage_l716_71612


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l716_71666

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 12 = 0 → y = x) ↔ 
  (m = -10 ∨ m = 14) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l716_71666


namespace NUMINAMATH_CALUDE_min_value_fraction_min_value_achieved_l716_71652

theorem min_value_fraction (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_sum : a + b + 2 * c = 2) : 
  (a + b) / (a * b * c) ≥ 8 := by
  sorry

theorem min_value_achieved : ∃ a b c : ℝ, 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + 2 * c = 2 ∧
  (a + b) / (a * b * c) = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_min_value_achieved_l716_71652


namespace NUMINAMATH_CALUDE_shortest_ant_path_l716_71680

/-- Represents a grid of square tiles -/
structure TileGrid where
  rows : ℕ
  columns : ℕ
  tileSize : ℝ

/-- Represents the path of an ant on a tile grid -/
def antPath (grid : TileGrid) : ℝ :=
  grid.tileSize * (grid.rows + grid.columns - 2)

/-- Theorem stating the shortest path for an ant on a 5x3 grid with tile size 10 -/
theorem shortest_ant_path :
  let grid : TileGrid := ⟨5, 3, 10⟩
  antPath grid = 80 := by
  sorry

#check shortest_ant_path

end NUMINAMATH_CALUDE_shortest_ant_path_l716_71680


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l716_71620

theorem quadratic_function_properties (a c : ℕ+) (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (a : ℝ) * x^2 + 2 * x + c
  (f 1 = 5) →
  (6 < f 2 ∧ f 2 < 11) →
  (∀ x ∈ Set.Icc (1/2 : ℝ) (3/2 : ℝ), f x - 2 * m * x ≤ 1) →
  (a = 1 ∧ c = 2 ∧ m ≥ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l716_71620


namespace NUMINAMATH_CALUDE_g_range_l716_71682

noncomputable def g (x : ℝ) : ℝ := 
  (Real.cos x ^ 3 + 7 * Real.cos x ^ 2 + 2 * Real.cos x + 3 * Real.sin x ^ 2 - 14) / (Real.cos x - 2)

theorem g_range : 
  ∀ x : ℝ, Real.cos x ≠ 2 → 
  (∃ y ∈ Set.Icc (1/2 : ℝ) (25/2 : ℝ), g x = y) ∧ 
  (∀ y : ℝ, g x = y → y ∈ Set.Icc (1/2 : ℝ) (25/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_g_range_l716_71682


namespace NUMINAMATH_CALUDE_inequality_range_l716_71616

-- Define the inequality
def inequality (x a : ℝ) : Prop :=
  x^2 - (a + 1) * x + a ≤ 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x : ℝ | inequality x a}

-- Define the interval [-4, 3]
def interval : Set ℝ :=
  {x : ℝ | -4 ≤ x ∧ x ≤ 3}

-- Statement of the theorem
theorem inequality_range :
  (∀ a : ℝ, solution_set a ⊆ interval) →
  ∀ a : ℝ, -4 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l716_71616


namespace NUMINAMATH_CALUDE_monotonicity_condition_positivity_condition_l716_71651

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the interval [5, 20]
def I : Set ℝ := Set.Icc 5 20

-- Part I: Monotonicity condition
theorem monotonicity_condition (k : ℝ) :
  (∀ x ∈ I, ∀ y ∈ I, x ≤ y → f k x ≤ f k y) ∨
  (∀ x ∈ I, ∀ y ∈ I, x ≤ y → f k x ≥ f k y) ↔
  k ∈ Set.Iic 40 ∪ Set.Ici 160 :=
sorry

-- Part II: Positivity condition
theorem positivity_condition (k : ℝ) :
  (∀ x ∈ I, f k x > 0) ↔ k < 92/5 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_condition_positivity_condition_l716_71651


namespace NUMINAMATH_CALUDE_initial_orange_balloons_l716_71658

theorem initial_orange_balloons (blue_balloons : ℕ) (lost_orange_balloons : ℕ) (remaining_orange_balloons : ℕ) : 
  blue_balloons = 4 → 
  lost_orange_balloons = 2 → 
  remaining_orange_balloons = 7 → 
  remaining_orange_balloons + lost_orange_balloons = 9 :=
by sorry

end NUMINAMATH_CALUDE_initial_orange_balloons_l716_71658


namespace NUMINAMATH_CALUDE_square_root_of_four_l716_71637

theorem square_root_of_four :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l716_71637


namespace NUMINAMATH_CALUDE_martin_crayon_boxes_l716_71676

theorem martin_crayon_boxes
  (crayons_per_box : ℕ)
  (total_crayons : ℕ)
  (h1 : crayons_per_box = 7)
  (h2 : total_crayons = 56) :
  total_crayons / crayons_per_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_martin_crayon_boxes_l716_71676


namespace NUMINAMATH_CALUDE_expression_evaluation_l716_71663

theorem expression_evaluation (a : ℝ) : 
  let x : ℝ := 2*a + 6
  (x - 2*a + 4) = 10 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l716_71663


namespace NUMINAMATH_CALUDE_largest_two_digit_divisible_by_six_ending_in_four_l716_71661

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_six_ending_in_four :
  ∃ (max : ℕ), 
    is_two_digit max ∧ 
    max % 6 = 0 ∧ 
    ends_in_four max ∧
    ∀ (n : ℕ), is_two_digit n → n % 6 = 0 → ends_in_four n → n ≤ max :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_divisible_by_six_ending_in_four_l716_71661


namespace NUMINAMATH_CALUDE_sum_equals_product_integer_pairs_l716_71632

theorem sum_equals_product_integer_pairs :
  ∀ x y : ℤ, x + y = x * y ↔ (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) := by
sorry

end NUMINAMATH_CALUDE_sum_equals_product_integer_pairs_l716_71632


namespace NUMINAMATH_CALUDE_function_value_inequality_l716_71610

theorem function_value_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) (h2 : ∀ x, deriv f x > 1) :
  f 3 > f 1 + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_inequality_l716_71610


namespace NUMINAMATH_CALUDE_tan_four_fifths_alpha_l716_71654

theorem tan_four_fifths_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 2 * Real.sqrt 3 * (Real.cos α) ^ 2 - Real.sin (2 * α) + 2 - Real.sqrt 3 = 0) : 
  Real.tan (4 / 5 * α) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_four_fifths_alpha_l716_71654


namespace NUMINAMATH_CALUDE_parabola_vertex_l716_71634

/-- A parabola is defined by the equation y^2 + 10y + 4x + 9 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 10*y + 4*x + 9 = 0

/-- The vertex of a parabola is the point where it turns -/
def is_vertex (x y : ℝ) : Prop :=
  ∀ t : ℝ, parabola_equation (x + t) (y + t) → t = 0

/-- The vertex of the parabola y^2 + 10y + 4x + 9 = 0 is the point (4, -5) -/
theorem parabola_vertex : is_vertex 4 (-5) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l716_71634


namespace NUMINAMATH_CALUDE_carter_cards_l716_71656

/-- Given that Marcus has 210 baseball cards and 58 more than Carter, 
    prove that Carter has 152 baseball cards. -/
theorem carter_cards (marcus_cards : ℕ) (difference : ℕ) (carter_cards : ℕ) 
  (h1 : marcus_cards = 210)
  (h2 : marcus_cards = carter_cards + difference)
  (h3 : difference = 58) : 
  carter_cards = 152 := by
  sorry

end NUMINAMATH_CALUDE_carter_cards_l716_71656


namespace NUMINAMATH_CALUDE_greatest_three_digit_odd_non_divisible_l716_71672

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_even_integers_up_to (n : ℕ) : ℕ :=
  let k := n / 2
  k * (k + 1)

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem greatest_three_digit_odd_non_divisible :
  ∀ n : ℕ,
    is_three_digit n →
    n % 2 = 1 →
    ¬(factorial n % sum_of_even_integers_up_to n = 0) →
    n ≤ 999 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_odd_non_divisible_l716_71672


namespace NUMINAMATH_CALUDE_toms_calculation_l716_71687

theorem toms_calculation (x y z : ℝ) 
  (h1 : (x + y) - z = 8) 
  (h2 : (x + y) + z = 20) : 
  x + y = 14 := by
sorry

end NUMINAMATH_CALUDE_toms_calculation_l716_71687


namespace NUMINAMATH_CALUDE_cake_eaten_after_four_trips_l716_71697

/-- The fraction of cake eaten after n trips, where on each trip 1/3 of the remaining cake is eaten -/
def cakeEaten (n : ℕ) : ℚ :=
  1 - (2/3)^n

/-- The theorem stating that after 4 trips, 40/81 of the cake is eaten -/
theorem cake_eaten_after_four_trips :
  cakeEaten 4 = 40/81 := by sorry

end NUMINAMATH_CALUDE_cake_eaten_after_four_trips_l716_71697


namespace NUMINAMATH_CALUDE_electronic_shop_price_l716_71644

def smartphone_price : ℝ := 300

def personal_computer_price (smartphone_price : ℝ) : ℝ :=
  smartphone_price + 500

def advanced_tablet_price (smartphone_price personal_computer_price : ℝ) : ℝ :=
  smartphone_price + personal_computer_price

def total_price (smartphone_price personal_computer_price advanced_tablet_price : ℝ) : ℝ :=
  smartphone_price + personal_computer_price + advanced_tablet_price

def discounted_price (total_price : ℝ) : ℝ :=
  total_price * 0.9

def final_price (discounted_price : ℝ) : ℝ :=
  discounted_price * 1.05

theorem electronic_shop_price :
  final_price (discounted_price (total_price smartphone_price 
    (personal_computer_price smartphone_price) 
    (advanced_tablet_price smartphone_price (personal_computer_price smartphone_price)))) = 2079 := by
  sorry

end NUMINAMATH_CALUDE_electronic_shop_price_l716_71644


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l716_71615

-- Define the concept of a line in 3D space
structure Line3D where
  -- Add appropriate fields to represent a line in 3D space
  -- This is a simplified representation
  dummy : Unit

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Add appropriate definition
  sorry

-- Define what it means for two lines to not intersect
def do_not_intersect (l1 l2 : Line3D) : Prop :=
  -- Add appropriate definition
  sorry

-- Theorem statement
theorem skew_lines_sufficient_not_necessary :
  (∀ l1 l2 : Line3D, are_skew l1 l2 → do_not_intersect l1 l2) ∧
  (∃ l1 l2 : Line3D, do_not_intersect l1 l2 ∧ ¬are_skew l1 l2) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l716_71615


namespace NUMINAMATH_CALUDE_correct_commutative_transformation_l716_71635

-- Define the commutative property of addition
axiom commutative_add (a b : ℝ) : a + b = b + a

-- Define the associative property of addition
axiom associative_add (a b c : ℝ) : (a + b) + c = a + (b + c)

-- State the theorem
theorem correct_commutative_transformation :
  4 + (-6) + 3 = (-6) + 4 + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_commutative_transformation_l716_71635


namespace NUMINAMATH_CALUDE_joint_purchase_popularity_l716_71667

/-- Represents the benefits of joint purchases -/
structure JointPurchaseBenefits where
  cost_savings : ℝ
  information_sharing : ℝ

/-- Represents the drawbacks of joint purchases -/
structure JointPurchaseDrawbacks where
  risks : ℝ
  transactional_costs : ℝ

/-- Represents factors affecting joint purchases -/
structure JointPurchaseFactors where
  benefits : JointPurchaseBenefits
  drawbacks : JointPurchaseDrawbacks
  proximity_to_stores : ℝ
  delivery_cost_savings : ℝ

/-- Determines if joint purchases are popular based on given factors -/
def joint_purchases_popular (factors : JointPurchaseFactors) : Prop :=
  factors.benefits.cost_savings + factors.benefits.information_sharing >
  factors.drawbacks.risks

/-- Determines if joint purchases are popular among neighbors based on given factors -/
def joint_purchases_popular_neighbors (factors : JointPurchaseFactors) : Prop :=
  factors.delivery_cost_savings >
  factors.drawbacks.transactional_costs + factors.proximity_to_stores

theorem joint_purchase_popularity
  (factors_countries factors_neighbors : JointPurchaseFactors)
  (h1 : joint_purchases_popular factors_countries)
  (h2 : ¬joint_purchases_popular_neighbors factors_neighbors) :
  (factors_countries.benefits.cost_savings + factors_countries.benefits.information_sharing >
   factors_countries.drawbacks.risks) ∧
  (factors_neighbors.drawbacks.transactional_costs + factors_neighbors.proximity_to_stores ≥
   factors_neighbors.delivery_cost_savings) :=
by sorry

end NUMINAMATH_CALUDE_joint_purchase_popularity_l716_71667


namespace NUMINAMATH_CALUDE_sundae_price_l716_71683

theorem sundae_price 
  (ice_cream_bars : ℕ) 
  (sundaes : ℕ) 
  (total_price : ℚ) 
  (ice_cream_price : ℚ) :
  ice_cream_bars = 225 →
  sundaes = 125 →
  total_price = 200 →
  ice_cream_price = 0.6 →
  (total_price - ice_cream_bars * ice_cream_price) / sundaes = 0.52 :=
by sorry

end NUMINAMATH_CALUDE_sundae_price_l716_71683


namespace NUMINAMATH_CALUDE_mike_picked_64_peaches_l716_71642

/-- Calculates the number of peaches Mike picked from the orchard -/
def peaches_picked (initial : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_away)

/-- Theorem: Given the initial conditions, Mike picked 64 peaches from the orchard -/
theorem mike_picked_64_peaches (initial : ℕ) (given_away : ℕ) (final : ℕ)
    (h1 : initial = 34)
    (h2 : given_away = 12)
    (h3 : final = 86) :
  peaches_picked initial given_away final = 64 := by
  sorry

#eval peaches_picked 34 12 86

end NUMINAMATH_CALUDE_mike_picked_64_peaches_l716_71642


namespace NUMINAMATH_CALUDE_postal_code_arrangements_l716_71624

/-- The number of possible arrangements of four distinct digits -/
def fourDigitArrangements : ℕ := 24

/-- The set of digits used in the postal code -/
def postalCodeDigits : Finset ℕ := {2, 3, 5, 8}

/-- Theorem: The number of arrangements of four distinct digits equals 24 -/
theorem postal_code_arrangements :
  Finset.card (Finset.powersetCard 4 postalCodeDigits) = fourDigitArrangements :=
by sorry

end NUMINAMATH_CALUDE_postal_code_arrangements_l716_71624


namespace NUMINAMATH_CALUDE_divisibility_cycle_l716_71611

theorem divisibility_cycle (x y z : ℕ+) : 
  (∃ a : ℕ+, y + 1 = a * x) ∧ 
  (∃ b : ℕ+, z + 1 = b * y) ∧ 
  (∃ c : ℕ+, x + 1 = c * z) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨
   (x = 1 ∧ y = 1 ∧ z = 2) ∨
   (x = 1 ∧ y = 2 ∧ z = 3) ∨
   (x = 3 ∧ y = 5 ∧ z = 4)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_cycle_l716_71611


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l716_71605

theorem modulo_eleven_residue : (310 + 6 * 45 + 8 * 154 + 3 * 23) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l716_71605


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l716_71629

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b) ∨ (b = c) ∨ (a = c)
  sumOfAngles : a + b + c = 180

-- Define the condition of angle ratio
def angleRatio (t : IsoscelesTriangle) : Prop :=
  (t.a = 2 * t.b) ∨ (t.b = 2 * t.c) ∨ (t.c = 2 * t.a)

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (t : IsoscelesTriangle) 
  (h : angleRatio t) : 
  (t.a = 90 ∨ t.b = 90 ∨ t.c = 90) ∨ 
  (t.a = 36 ∨ t.b = 36 ∨ t.c = 36) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l716_71629


namespace NUMINAMATH_CALUDE_complex_fraction_equals_one_minus_i_l716_71607

theorem complex_fraction_equals_one_minus_i : 
  let i : ℂ := Complex.I
  2 / (1 + i) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_one_minus_i_l716_71607


namespace NUMINAMATH_CALUDE_final_number_bound_l716_71685

/-- A function that represents the process of replacing two numbers with their arithmetic mean. -/
def replace (numbers : List ℝ) : List ℝ :=
  sorry

/-- The theorem stating that the final number is not less than 1/n. -/
theorem final_number_bound (n : ℕ) (h : n > 0) :
  ∃ (process : ℕ → List ℝ), 
    (process 0 = List.replicate n 1) ∧ 
    (∀ k, process (k + 1) = replace (process k)) ∧
    (∃ m, (process m).length = 1 ∧ 
      ∀ x ∈ process m, x ≥ 1 / n) :=
  sorry

end NUMINAMATH_CALUDE_final_number_bound_l716_71685


namespace NUMINAMATH_CALUDE_f_range_l716_71641

-- Define the function
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- State the theorem
theorem f_range : 
  Set.range f = { y | y ≥ 2 } := by sorry

end NUMINAMATH_CALUDE_f_range_l716_71641


namespace NUMINAMATH_CALUDE_inequality_proof_l716_71694

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) : 
  (a * b ≤ 1/8) ∧ (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 6 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l716_71694


namespace NUMINAMATH_CALUDE_sqrt_two_squared_times_three_to_fourth_l716_71614

theorem sqrt_two_squared_times_three_to_fourth : Real.sqrt (2^2 * 3^4) = 18 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_times_three_to_fourth_l716_71614


namespace NUMINAMATH_CALUDE_circle_properties_l716_71679

/-- Given a circle with equation x^2 + y^2 - 4x + 2y + 4 = 0, 
    its radius is 1 and its center coordinates are (2, -1) -/
theorem circle_properties : 
  ∃ (r : ℝ) (x₀ y₀ : ℝ), 
    (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 4 = 0 ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
    r = 1 ∧ x₀ = 2 ∧ y₀ = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l716_71679


namespace NUMINAMATH_CALUDE_shaded_region_area_l716_71627

/-- The area of a shaded region formed by the intersection of two circles -/
theorem shaded_region_area (r : ℝ) (h : r = 5) : 
  (2 * (π * r^2 / 4) - r^2) = (50 * π - 100) / 4 := by
  sorry

#check shaded_region_area

end NUMINAMATH_CALUDE_shaded_region_area_l716_71627


namespace NUMINAMATH_CALUDE_direction_vector_of_line_l716_71671

/-- Given a line with equation y = -1/2 * x + 1, prove that (2, -1) is a valid direction vector. -/
theorem direction_vector_of_line (x y : ℝ) :
  y = -1/2 * x + 1 →
  ∃ (t : ℝ), (x + 2*t, y - t) = (x, y) :=
by sorry

end NUMINAMATH_CALUDE_direction_vector_of_line_l716_71671


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l716_71681

def f (x : ℝ) := x^3 - x - 1

theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l716_71681


namespace NUMINAMATH_CALUDE_nancy_purchase_cost_l716_71686

/-- The cost of a set of crystal beads in dollars -/
def crystal_cost : ℕ := 9

/-- The cost of a set of metal beads in dollars -/
def metal_cost : ℕ := 10

/-- The number of crystal bead sets purchased -/
def crystal_sets : ℕ := 1

/-- The number of metal bead sets purchased -/
def metal_sets : ℕ := 2

/-- The total cost of the purchase in dollars -/
def total_cost : ℕ := crystal_cost * crystal_sets + metal_cost * metal_sets

theorem nancy_purchase_cost : total_cost = 29 := by
  sorry

end NUMINAMATH_CALUDE_nancy_purchase_cost_l716_71686


namespace NUMINAMATH_CALUDE_square_on_hypotenuse_l716_71608

theorem square_on_hypotenuse (a b : ℝ) (ha : a = 9) (hb : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a * b) / (a + b)
  s = 120 / 37 := by sorry

end NUMINAMATH_CALUDE_square_on_hypotenuse_l716_71608


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l716_71662

theorem ratio_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l716_71662


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_hyperbola_l716_71677

theorem no_integer_solutions_for_hyperbola : 
  ¬∃ (x y : ℤ), x^2 - y^2 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_hyperbola_l716_71677


namespace NUMINAMATH_CALUDE_equal_squares_on_8x7_board_l716_71690

/-- Represents a rectangular board with alternating light and dark squares. -/
structure AlternatingBoard :=
  (rows : Nat)
  (columns : Nat)

/-- Counts the number of dark squares on the board. -/
def count_dark_squares (board : AlternatingBoard) : Nat :=
  (board.rows / 2) * ((board.columns + 1) / 2) + 
  ((board.rows + 1) / 2) * (board.columns / 2)

/-- Counts the number of light squares on the board. -/
def count_light_squares (board : AlternatingBoard) : Nat :=
  ((board.rows + 1) / 2) * ((board.columns + 1) / 2) + 
  (board.rows / 2) * (board.columns / 2)

/-- Theorem stating that for an 8x7 alternating board, the number of dark squares equals the number of light squares. -/
theorem equal_squares_on_8x7_board :
  let board : AlternatingBoard := ⟨8, 7⟩
  count_dark_squares board = count_light_squares board := by
  sorry

#eval count_dark_squares ⟨8, 7⟩
#eval count_light_squares ⟨8, 7⟩

end NUMINAMATH_CALUDE_equal_squares_on_8x7_board_l716_71690


namespace NUMINAMATH_CALUDE_math_reading_difference_l716_71674

def reading_homework : ℕ := 4
def math_homework : ℕ := 7

theorem math_reading_difference : math_homework - reading_homework = 3 := by
  sorry

end NUMINAMATH_CALUDE_math_reading_difference_l716_71674


namespace NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l716_71664

theorem prop_a_necessary_not_sufficient (h : ℝ) (h_pos : h > 0) :
  (∀ a b : ℝ, (|a - 1| < h ∧ |b - 1| < h) → |a - b| < 2*h) ∧
  (∃ a b : ℝ, |a - b| < 2*h ∧ ¬(|a - 1| < h ∧ |b - 1| < h)) :=
by sorry

end NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l716_71664


namespace NUMINAMATH_CALUDE_square_diff_over_seventy_l716_71647

theorem square_diff_over_seventy : (535^2 - 465^2) / 70 = 1000 := by sorry

end NUMINAMATH_CALUDE_square_diff_over_seventy_l716_71647


namespace NUMINAMATH_CALUDE_range_of_a_l716_71660

theorem range_of_a (x a : ℝ) : 
  x > 2 → a ≤ x + 2 / (x - 2) → ∃ s : ℝ, s = 2 + 2 * Real.sqrt 2 ∧ IsLUB {a | ∃ x > 2, a ≤ x + 2 / (x - 2)} s :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l716_71660


namespace NUMINAMATH_CALUDE_systematic_sampling_distribution_l716_71603

/-- Represents a building in the summer camp -/
inductive Building
| A
| B
| C

/-- Calculates the number of students selected from each building using systematic sampling -/
def systematic_sampling (total_students : ℕ) (sample_size : ℕ) (start : ℕ) : Building → ℕ :=
  λ b =>
    match b with
    | Building.A => sorry
    | Building.B => sorry
    | Building.C => sorry

theorem systematic_sampling_distribution :
  let total_students := 400
  let sample_size := 50
  let start := 5
  (systematic_sampling total_students sample_size start Building.A = 25) ∧
  (systematic_sampling total_students sample_size start Building.B = 12) ∧
  (systematic_sampling total_students sample_size start Building.C = 13) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_distribution_l716_71603


namespace NUMINAMATH_CALUDE_golden_ratio_bounds_l716_71692

theorem golden_ratio_bounds : 
  let φ := (Real.sqrt 5 - 1) / 2
  0.6 < φ ∧ φ < 0.7 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_bounds_l716_71692


namespace NUMINAMATH_CALUDE_milk_sharing_problem_l716_71638

/-- Given a total amount of milk and a difference between two people's consumption,
    calculate the amount consumed by the person drinking more. -/
def calculate_larger_share (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

/-- Proof that given 2100 ml of milk shared between two people,
    where one drinks 200 ml more than the other,
    the person drinking more consumes 1150 ml. -/
theorem milk_sharing_problem :
  calculate_larger_share 2100 200 = 1150 := by
  sorry

end NUMINAMATH_CALUDE_milk_sharing_problem_l716_71638


namespace NUMINAMATH_CALUDE_right_angled_triangle_l716_71631

theorem right_angled_triangle (A B C : ℝ) (h : A + B + C = π) 
  (eq : (Real.cos A) / 20 + (Real.cos B) / 21 + (Real.cos C) / 29 = 29 / 420) : 
  C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l716_71631


namespace NUMINAMATH_CALUDE_all_propositions_imply_target_l716_71646

theorem all_propositions_imply_target : ∀ (p q r : Prop),
  (p ∧ q ∧ r → (p → q) ∨ r) ∧
  (¬p ∧ q ∧ ¬r → (p → q) ∨ r) ∧
  (p ∧ ¬q ∧ r → (p → q) ∨ r) ∧
  (¬p ∧ ¬q ∧ ¬r → (p → q) ∨ r) :=
by sorry

#check all_propositions_imply_target

end NUMINAMATH_CALUDE_all_propositions_imply_target_l716_71646


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l716_71669

-- Define the quadratic function f
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

-- Theorem statement
theorem quadratic_function_properties :
  (∀ x, f x ≥ 1) ∧  -- Minimum value is 1
  (f 0 = 3) ∧ (f 2 = 3) ∧  -- f(0) = f(2) = 3
  (∀ x, f x = 2 * x^2 - 4 * x + 3) ∧  -- Expression of f(x)
  (∀ a, (0 < a ∧ a < 1/3) ↔ 
    (∃ x y, 3*a ≤ x ∧ x < y ∧ y ≤ a+1 ∧ f x > f y ∧ 
    ∃ z, x < z ∧ z < y ∧ f z < f y)) ∧  -- Non-monotonic condition
  (∀ m, m < -1 ↔ 
    (∀ x, -1 ≤ x ∧ x ≤ 1 → f x > 2*x + 2*m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l716_71669


namespace NUMINAMATH_CALUDE_smallest_section_area_l716_71649

/-- The area of the smallest circular section of a sphere circumscribed around a cube --/
theorem smallest_section_area (cube_edge : ℝ) (h : cube_edge = 4) : 
  let sphere_radius : ℝ := cube_edge * Real.sqrt 3 / 2
  let midpoint_to_center : ℝ := cube_edge * Real.sqrt 2 / 2
  let section_radius : ℝ := Real.sqrt (sphere_radius^2 - midpoint_to_center^2)
  π * section_radius^2 = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_smallest_section_area_l716_71649


namespace NUMINAMATH_CALUDE_social_gathering_handshakes_l716_71636

theorem social_gathering_handshakes (n : ℕ) (h : n = 8) : 
  let total_people := 2 * n
  let handshakes_per_person := total_people - 2
  (total_people * handshakes_per_person) / 2 = 112 := by
sorry

end NUMINAMATH_CALUDE_social_gathering_handshakes_l716_71636


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l716_71650

/-- The circle C -/
def circle_C (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 100

/-- Point A -/
def point_A : ℝ × ℝ := (4, 0)

/-- The trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

/-- Theorem: The trajectory of the center of a moving circle that is tangent to circle C
    and passes through point A is described by the equation x²/25 + y²/9 = 1 -/
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 → 
      (∃ (x'' y'' : ℝ), circle_C x'' y'' ∧ (x' - x'')^2 + (y' - y'')^2 = 0)) ∧
    ((x - point_A.1)^2 + (y - point_A.2)^2 = r^2)) →
  trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l716_71650


namespace NUMINAMATH_CALUDE_kevin_distance_after_six_hops_l716_71602

/-- Kevin's hopping journey on a number line -/
def kevin_hop (total_distance : ℚ) (first_hop_fraction : ℚ) (subsequent_hop_fraction : ℚ) (num_hops : ℕ) : ℚ :=
  let first_hop := first_hop_fraction * total_distance
  let remaining_distance := total_distance - first_hop
  let subsequent_hops := remaining_distance * (1 - (1 - subsequent_hop_fraction) ^ (num_hops - 1))
  first_hop + subsequent_hops

/-- The theorem stating the distance Kevin has hopped after six hops -/
theorem kevin_distance_after_six_hops :
  kevin_hop 2 (1/4) (2/3) 6 = 1071/243 := by
  sorry

end NUMINAMATH_CALUDE_kevin_distance_after_six_hops_l716_71602
