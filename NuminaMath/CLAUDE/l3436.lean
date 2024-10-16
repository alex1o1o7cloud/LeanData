import Mathlib

namespace NUMINAMATH_CALUDE_jane_work_days_l3436_343600

theorem jane_work_days (john_rate : ℚ) (total_days : ℕ) (jane_stop_days : ℕ) :
  john_rate = 1/20 →
  total_days = 10 →
  jane_stop_days = 5 →
  ∃ jane_rate : ℚ,
    (5 * (john_rate + jane_rate) + 5 * john_rate = 1) ∧
    (jane_rate = 1/10) :=
by sorry

end NUMINAMATH_CALUDE_jane_work_days_l3436_343600


namespace NUMINAMATH_CALUDE_spaceship_age_conversion_l3436_343619

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let hundreds := octal / 100
  let tens := (octal / 10) % 10
  let ones := octal % 10
  hundreds * 64 + tens * 8 + ones

/-- Theorem: The octal number 367₈ is equal to 247 in base 10 --/
theorem spaceship_age_conversion : octal_to_decimal 367 = 247 := by
  sorry

#eval octal_to_decimal 367  -- Should output 247

end NUMINAMATH_CALUDE_spaceship_age_conversion_l3436_343619


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3436_343675

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 5 + a 6 = 20) : 
  (a 4 + a 7) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3436_343675


namespace NUMINAMATH_CALUDE_wood_length_proof_l3436_343623

/-- The original length of a piece of wood, given the length sawed off and the remaining length. -/
def original_length (sawed_off : ℝ) (remaining : ℝ) : ℝ := sawed_off + remaining

/-- Theorem stating that the original length of the wood is 0.41 meters. -/
theorem wood_length_proof :
  let sawed_off : ℝ := 0.33
  let remaining : ℝ := 0.08
  original_length sawed_off remaining = 0.41 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_proof_l3436_343623


namespace NUMINAMATH_CALUDE_eccentricity_for_one_and_nine_l3436_343634

/-- The eccentricity of a curve given two positive numbers -/
def eccentricity_of_curve (x y : ℝ) : Set ℝ :=
  let a := (x + y) / 2
  let b := Real.sqrt (x * y)
  let e₁ := Real.sqrt (a - b) / Real.sqrt a
  let e₂ := Real.sqrt (a + b) / Real.sqrt a
  {e₁, e₂}

/-- Theorem: The eccentricity of the curve for numbers 1 and 9 -/
theorem eccentricity_for_one_and_nine :
  eccentricity_of_curve 1 9 = {Real.sqrt 10 / 5, 2 * Real.sqrt 10 / 5} :=
by sorry

end NUMINAMATH_CALUDE_eccentricity_for_one_and_nine_l3436_343634


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l3436_343616

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l3436_343616


namespace NUMINAMATH_CALUDE_radiator_water_fraction_l3436_343649

/-- The fraction of water remaining after n replacements in a radiator -/
def water_fraction (radiator_capacity : ℚ) (replacement_volume : ℚ) (n : ℕ) : ℚ :=
  (1 - replacement_volume / radiator_capacity) ^ n

theorem radiator_water_fraction :
  let radiator_capacity : ℚ := 20
  let replacement_volume : ℚ := 5
  let num_replacements : ℕ := 5
  water_fraction radiator_capacity replacement_volume num_replacements = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_radiator_water_fraction_l3436_343649


namespace NUMINAMATH_CALUDE_ninth_term_value_l3436_343693

/-- An arithmetic sequence with specific conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧  -- Definition of arithmetic sequence
  (a 5 + a 7 = 16) ∧                         -- Given condition
  (a 3 = 4)                                  -- Given condition

/-- Theorem stating the value of the 9th term -/
theorem ninth_term_value (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_value_l3436_343693


namespace NUMINAMATH_CALUDE_smallest_x_properties_l3436_343657

/-- The smallest integer with 18 positive factors that is divisible by both 18 and 24 -/
def smallest_x : ℕ := 288

/-- The number of positive factors of smallest_x -/
def factor_count : ℕ := 18

theorem smallest_x_properties :
  (∃ (factors : Finset ℕ), factors.card = factor_count ∧ 
    ∀ d ∈ factors, d ∣ smallest_x) ∧
  18 ∣ smallest_x ∧
  24 ∣ smallest_x ∧
  ∀ y : ℕ, y < smallest_x →
    ¬(∃ (factors : Finset ℕ), factors.card = factor_count ∧
      ∀ d ∈ factors, d ∣ y ∧ 18 ∣ y ∧ 24 ∣ y) :=
by
  sorry

#eval smallest_x

end NUMINAMATH_CALUDE_smallest_x_properties_l3436_343657


namespace NUMINAMATH_CALUDE_triangle_areas_sum_l3436_343612

/-- Given a rectangle and two triangles with specific properties, prove that the combined area of the triangles is 108 cm² -/
theorem triangle_areas_sum (rectangle_length rectangle_width : ℝ)
  (triangle1_area_factor : ℝ)
  (triangle2_base triangle2_base_height_sum : ℝ)
  (h_rectangle_length : rectangle_length = 6)
  (h_rectangle_width : rectangle_width = 4)
  (h_rectangle_triangle1_ratio : (rectangle_length * rectangle_width) / (5 * triangle1_area_factor) = 2 / 5)
  (h_triangle2_base : triangle2_base = 8)
  (h_triangle2_sum : triangle2_base + (triangle2_base_height_sum - triangle2_base) = 20)
  (h_triangle_ratio : (triangle2_base * (triangle2_base_height_sum - triangle2_base)) / (10 * triangle1_area_factor) = 3 / 5) :
  5 * triangle1_area_factor + (triangle2_base * (triangle2_base_height_sum - triangle2_base)) / 2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_triangle_areas_sum_l3436_343612


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3436_343621

/-- Given two vectors a and b in ℝ², where a is perpendicular to b,
    prove that the angle between (a - b) and b is 150°. -/
theorem angle_between_vectors (a b : ℝ × ℝ) (h_a : a = (Real.sqrt 3, 1))
    (h_b : b.2 = -3) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
    let diff := (a.1 - b.1, a.2 - b.2)
    Real.arccos ((diff.1 * b.1 + diff.2 * b.2) / 
      (Real.sqrt (diff.1^2 + diff.2^2) * Real.sqrt (b.1^2 + b.2^2))) =
    150 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l3436_343621


namespace NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_2_mod_17_l3436_343685

theorem largest_four_digit_negative_congruent_to_2_mod_17 :
  ∀ x : ℤ, -9999 ≤ x ∧ x < -999 ∧ x ≡ 2 [ZMOD 17] → x ≤ -1001 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_2_mod_17_l3436_343685


namespace NUMINAMATH_CALUDE_apples_left_proof_l3436_343689

/-- The number of apples left when the farmer's children got home -/
def apples_left (num_children : ℕ) (apples_per_child : ℕ) (children_who_ate : ℕ) 
  (apples_eaten_per_child : ℕ) (apples_sold : ℕ) : ℕ :=
  num_children * apples_per_child - (children_who_ate * apples_eaten_per_child + apples_sold)

/-- Theorem stating the number of apples left when the farmer's children got home -/
theorem apples_left_proof : 
  apples_left 5 15 2 4 7 = 60 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_proof_l3436_343689


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3436_343650

-- Define the polynomial
def p (x : ℝ) : ℝ := 42 * x^3 - 35 * x^2 + 10 * x - 1

-- Define the roots
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

-- State the theorem
theorem root_sum_reciprocal :
  p a = 0 ∧ p b = 0 ∧ p c = 0 ∧   -- a, b, c are roots of p
  0 < a ∧ a < 1 ∧                 -- a is between 0 and 1
  0 < b ∧ b < 1 ∧                 -- b is between 0 and 1
  0 < c ∧ c < 1 ∧                 -- c is between 0 and 1
  a ≠ b ∧ b ≠ c ∧ a ≠ c →         -- roots are distinct
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 2.875 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3436_343650


namespace NUMINAMATH_CALUDE_balloon_theorem_l3436_343630

/-- Represents a person's balloon collection -/
structure BalloonCollection where
  count : ℕ
  cost : ℕ

/-- Calculates the total number of balloons from a list of balloon collections -/
def totalBalloons (collections : List BalloonCollection) : ℕ :=
  collections.map (·.count) |>.sum

/-- Calculates the total cost of balloons from a list of balloon collections -/
def totalCost (collections : List BalloonCollection) : ℕ :=
  collections.map (fun c => c.count * c.cost) |>.sum

theorem balloon_theorem (fred sam mary susan tom : BalloonCollection)
    (h1 : fred = ⟨5, 3⟩)
    (h2 : sam = ⟨6, 4⟩)
    (h3 : mary = ⟨7, 5⟩)
    (h4 : susan = ⟨4, 6⟩)
    (h5 : tom = ⟨10, 2⟩) :
    let collections := [fred, sam, mary, susan, tom]
    totalBalloons collections = 32 ∧ totalCost collections = 118 := by
  sorry

end NUMINAMATH_CALUDE_balloon_theorem_l3436_343630


namespace NUMINAMATH_CALUDE_min_fleet_size_10x10_l3436_343620

/-- A ship is a figure made up of unit squares connected by common edges -/
def Ship : Type := Unit

/-- A fleet is a set of ships where no two ships contain squares that share a common vertex -/
def Fleet : Type := Set Ship

/-- The size of the grid -/
def gridSize : ℕ := 10

/-- The minimum number of squares in a fleet to which no new ship can be added -/
def minFleetSize (n : ℕ) : ℕ :=
  if n % 3 = 0 then (n / 3) ^ 2
  else (n / 3 + 1) ^ 2

theorem min_fleet_size_10x10 :
  minFleetSize gridSize = 16 := by sorry

end NUMINAMATH_CALUDE_min_fleet_size_10x10_l3436_343620


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3436_343681

/-- Given simple interest, principal, and time, calculate the interest rate in paise per rupee per month -/
theorem interest_rate_calculation (simple_interest principal time : ℚ) 
  (h1 : simple_interest = 4.8)
  (h2 : principal = 8)
  (h3 : time = 12) :
  (simple_interest / (principal * time)) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3436_343681


namespace NUMINAMATH_CALUDE_original_group_size_l3436_343602

theorem original_group_size 
  (total_work : ℝ) 
  (original_days : ℕ) 
  (remaining_days : ℕ) 
  (absent_men : ℕ) :
  let original_work_rate := total_work / original_days
  let remaining_work_rate := total_work / remaining_days
  original_days = 10 ∧ 
  remaining_days = 12 ∧ 
  absent_men = 5 →
  ∃ (original_size : ℕ),
    original_size * original_work_rate = (original_size - absent_men) * remaining_work_rate ∧
    original_size = 25 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l3436_343602


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3436_343644

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the foci
def focus1 : ℝ × ℝ := sorry
def focus2 : ℝ × ℝ := sorry

-- Define points P and Q on the ellipse
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define the line passing through F₁, P, and Q
def line_through_F1PQ : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter :
  ellipse P.1 P.2 ∧ ellipse Q.1 Q.2 ∧ 
  P ∈ line_through_F1PQ ∧ Q ∈ line_through_F1PQ ∧ focus1 ∈ line_through_F1PQ →
  (dist P Q + dist Q focus2 + dist P focus2 = 20) :=
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3436_343644


namespace NUMINAMATH_CALUDE_new_students_average_age_l3436_343660

/-- Proves that the average age of new students is 32 years given the problem conditions. -/
theorem new_students_average_age
  (original_average : ℕ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℕ)
  (h1 : original_average = 40)
  (h2 : original_strength = 12)
  (h3 : new_students = 12)
  (h4 : average_decrease = 4) :
  (original_average * original_strength + new_students * 32) / (original_strength + new_students) =
  original_average - average_decrease :=
by sorry


end NUMINAMATH_CALUDE_new_students_average_age_l3436_343660


namespace NUMINAMATH_CALUDE_common_tangents_count_l3436_343654

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 9 = 0

-- Define the function to count common tangent lines
def count_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle1 circle2 = 3 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l3436_343654


namespace NUMINAMATH_CALUDE_range_positive_iff_l3436_343615

/-- The quadratic function f(x) = ax^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

/-- The range of f is a subset of positive real numbers -/
def range_subset_positive (a : ℝ) : Prop :=
  ∀ x, f a x > 0

/-- The necessary and sufficient condition for the range of f to be a subset of positive real numbers -/
theorem range_positive_iff (a : ℝ) :
  range_subset_positive a ↔ 0 ≤ a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_range_positive_iff_l3436_343615


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_eight_consecutive_l3436_343655

theorem three_digit_divisible_by_eight_consecutive : ∃ n : ℕ, 
  100 ≤ n ∧ n ≤ 999 ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 8 → k ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_eight_consecutive_l3436_343655


namespace NUMINAMATH_CALUDE_teal_color_perception_l3436_343695

theorem teal_color_perception (total : ℕ) (kinda_blue : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 150)
  (h2 : kinda_blue = 90)
  (h3 : both = 45)
  (h4 : neither = 25) :
  ∃ kinda_green : ℕ, kinda_green = 80 ∧ 
  kinda_green = total - (kinda_blue - both) - neither :=
by sorry

end NUMINAMATH_CALUDE_teal_color_perception_l3436_343695


namespace NUMINAMATH_CALUDE_petyas_run_l3436_343680

theorem petyas_run (V D : ℝ) (hV : V > 0) (hD : D > 0) : 
  D / (2 * 1.25 * V) + D / (2 * 0.8 * V) > D / V := by
  sorry

end NUMINAMATH_CALUDE_petyas_run_l3436_343680


namespace NUMINAMATH_CALUDE_days_to_watch_all_episodes_l3436_343687

-- Define the number of episodes for each season type
def regular_season_episodes : ℕ := 22
def third_season_episodes : ℕ := 24
def last_season_episodes : ℕ := regular_season_episodes + 4

-- Define the duration of episodes for different seasons
def early_episode_duration : ℚ := 1/2
def later_episode_duration : ℚ := 3/4

-- Define John's daily watching time
def daily_watching_time : ℚ := 2

-- Define the total number of seasons
def total_seasons : ℕ := 10

-- Define a function to calculate the total viewing time
def total_viewing_time : ℚ :=
  let early_seasons_episodes : ℕ := 2 * regular_season_episodes + third_season_episodes
  let early_seasons_time : ℚ := early_seasons_episodes * early_episode_duration
  let later_seasons_episodes : ℕ := (total_seasons - 3) * regular_season_episodes + last_season_episodes
  let later_seasons_time : ℚ := later_seasons_episodes * later_episode_duration
  early_seasons_time + later_seasons_time

-- Theorem statement
theorem days_to_watch_all_episodes :
  ⌈total_viewing_time / daily_watching_time⌉ = 77 := by sorry

end NUMINAMATH_CALUDE_days_to_watch_all_episodes_l3436_343687


namespace NUMINAMATH_CALUDE_max_n_for_int_polynomial_l3436_343686

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The property that P(aᵢ) = i for all 1 ≤ i ≤ n -/
def SatisfiesProperty (P : IntPolynomial) (n : ℕ) : Prop :=
  ∃ (a : ℕ → ℤ), ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → P.eval (a i) = i

/-- The theorem stating the maximum n for which the property holds -/
theorem max_n_for_int_polynomial (P : IntPolynomial) (h : P.degree = 2022) :
    (∃ n : ℕ, SatisfiesProperty P n ∧ ∀ m : ℕ, SatisfiesProperty P m → m ≤ n) ∧
    (∃ n : ℕ, n = 2022 ∧ SatisfiesProperty P n) :=
  sorry

end NUMINAMATH_CALUDE_max_n_for_int_polynomial_l3436_343686


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_l3436_343699

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

theorem least_three_digit_7_heavy : 
  (∀ m : ℕ, 100 ≤ m ∧ m < 104 → ¬ is_7_heavy m) ∧ 
  is_7_heavy 104 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_l3436_343699


namespace NUMINAMATH_CALUDE_student_correct_answers_l3436_343640

/-- Represents a test score calculation system -/
structure TestScore where
  totalQuestions : ℕ
  score : ℤ
  correctAnswers : ℕ
  incorrectAnswers : ℕ

/-- Theorem: Given the conditions, prove that the student answered 91 questions correctly -/
theorem student_correct_answers
  (test : TestScore)
  (h1 : test.totalQuestions = 100)
  (h2 : test.score = test.correctAnswers - 2 * test.incorrectAnswers)
  (h3 : test.score = 73)
  (h4 : test.correctAnswers + test.incorrectAnswers = test.totalQuestions) :
  test.correctAnswers = 91 := by
  sorry


end NUMINAMATH_CALUDE_student_correct_answers_l3436_343640


namespace NUMINAMATH_CALUDE_square_perimeter_doubled_l3436_343601

theorem square_perimeter_doubled (area : ℝ) (h : area = 900) : 
  let side_length := Real.sqrt area
  let initial_perimeter := 4 * side_length
  let doubled_perimeter := 2 * initial_perimeter
  doubled_perimeter = 240 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_doubled_l3436_343601


namespace NUMINAMATH_CALUDE_problem_solution_l3436_343638

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + x^3 / y^2 + y^3 / x^2 + y = 590 + 5/9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3436_343638


namespace NUMINAMATH_CALUDE_angle_of_inclination_x_eq_neg_one_l3436_343676

-- Define a vertical line
def vertical_line (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 = a}

-- Define the angle of inclination for a vertical line
def angle_of_inclination_vertical (l : Set (ℝ × ℝ)) : ℝ := 90

-- Theorem statement
theorem angle_of_inclination_x_eq_neg_one :
  angle_of_inclination_vertical (vertical_line (-1)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_x_eq_neg_one_l3436_343676


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3436_343611

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (((43 - 2*x) ^ (1/4) : ℝ) + ((37 + 2*x) ^ (1/4) : ℝ) = 4) ↔ (x = -19 ∨ x = 21) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3436_343611


namespace NUMINAMATH_CALUDE_largest_term_binomial_sequence_l3436_343692

theorem largest_term_binomial_sequence (k : ℕ) :
  k ≤ 1992 →
  k * Nat.choose 1992 k ≤ 997 * Nat.choose 1992 997 :=
by sorry

end NUMINAMATH_CALUDE_largest_term_binomial_sequence_l3436_343692


namespace NUMINAMATH_CALUDE_equation_solution_l3436_343627

theorem equation_solution (x : ℝ) : 
  Real.sqrt (5 * x - 4) + 15 / Real.sqrt (5 * x - 4) = 8 → x = 29/5 ∨ x = 13/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3436_343627


namespace NUMINAMATH_CALUDE_range_of_a_l3436_343633

noncomputable def f (a x : ℝ) : ℝ := a / x - Real.exp (-x)

theorem range_of_a (a : ℝ) :
  (∃ p q : ℝ, p < q ∧ 
    (∀ x : ℝ, x > 0 → (f a x ≤ 0 ↔ p ≤ x ∧ x ≤ q))) →
  0 < a ∧ a < 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3436_343633


namespace NUMINAMATH_CALUDE_xy_positive_iff_fraction_positive_solution_set_inequality_l3436_343651

-- Statement A
theorem xy_positive_iff_fraction_positive (x y : ℝ) :
  x * y > 0 ↔ x / y > 0 :=
sorry

-- Statement D
theorem solution_set_inequality (x : ℝ) :
  (x + 1) * (2 - x) < 0 ↔ x < -1 ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_xy_positive_iff_fraction_positive_solution_set_inequality_l3436_343651


namespace NUMINAMATH_CALUDE_cube_surface_area_l3436_343669

theorem cube_surface_area (d : ℝ) (h : d = 8 * Real.sqrt 2) : 
  6 * (d / Real.sqrt 2) ^ 2 = 384 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3436_343669


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3436_343691

theorem quadratic_equal_roots (x : ℝ) : 
  (∃ r : ℝ, (x^2 + 2*x + 1 = 0) ↔ (x = r ∧ x = r)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3436_343691


namespace NUMINAMATH_CALUDE_equal_probability_for_all_positions_l3436_343628

/-- Represents a lottery draw with n tickets, where one is winning. -/
structure LotteryDraw (n : ℕ) where
  tickets : Fin n → Bool
  winning_exists : ∃ t, tickets t = true
  only_one_winning : ∀ t₁ t₂, tickets t₁ = true → tickets t₂ = true → t₁ = t₂

/-- The probability of drawing the winning ticket in any position of a sequence of n draws. -/
def winning_probability (n : ℕ) (pos : Fin n) (draw : LotteryDraw n) : ℚ :=
  1 / n

/-- Theorem stating that the probability of drawing the winning ticket is equal for all positions in a sequence of 5 draws. -/
theorem equal_probability_for_all_positions (draw : LotteryDraw 5) :
    ∀ pos₁ pos₂ : Fin 5, winning_probability 5 pos₁ draw = winning_probability 5 pos₂ draw :=
  sorry

end NUMINAMATH_CALUDE_equal_probability_for_all_positions_l3436_343628


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3436_343641

theorem ratio_x_to_y (x y : ℝ) (h : y = x * (1 - 0.8333333333333334)) :
  x / y = 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3436_343641


namespace NUMINAMATH_CALUDE_average_growth_rate_correct_l3436_343639

/-- The average monthly growth rate of profit from March to May -/
def average_growth_rate : ℝ := 0.2

/-- The profit in March -/
def profit_march : ℝ := 5000

/-- The profit in May -/
def profit_may : ℝ := 7200

/-- The number of months between March and May -/
def months_between : ℕ := 2

/-- Theorem stating that the average monthly growth rate is correct -/
theorem average_growth_rate_correct : 
  profit_march * (1 + average_growth_rate) ^ months_between = profit_may := by
  sorry

end NUMINAMATH_CALUDE_average_growth_rate_correct_l3436_343639


namespace NUMINAMATH_CALUDE_opposite_to_83_is_84_l3436_343679

/-- Represents a circle divided into 100 equal arcs with numbers assigned to each arc. -/
structure NumberedCircle where
  /-- The assignment of numbers to arcs, represented as a function from arc index to number. -/
  number_assignment : Fin 100 → Fin 100
  /-- The assignment is a bijection (each number is used exactly once). -/
  bijective : Function.Bijective number_assignment

/-- Checks if numbers less than k are evenly distributed on both sides of the diameter through k. -/
def evenlyDistributed (c : NumberedCircle) (k : Fin 100) : Prop :=
  ∀ (i : Fin 100), c.number_assignment i < k →
    (∃ (j : Fin 100), c.number_assignment j < k ∧ (i + 50) % 100 = j)

/-- The main theorem stating that if numbers are evenly distributed for all k,
    then the number opposite to 83 is 84. -/
theorem opposite_to_83_is_84 (c : NumberedCircle) 
    (h : ∀ (k : Fin 100), evenlyDistributed c k) :
    ∃ (i : Fin 100), c.number_assignment i = 83 ∧ c.number_assignment ((i + 50) % 100) = 84 := by
  sorry

end NUMINAMATH_CALUDE_opposite_to_83_is_84_l3436_343679


namespace NUMINAMATH_CALUDE_intersection_A_B_l3436_343605

def A : Set ℝ := {x | -3 ≤ 2*x - 1 ∧ 2*x - 1 < 3}
def B : Set ℝ := {x | ∃ k : ℤ, x = 2*k + 1}

theorem intersection_A_B : A ∩ B = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3436_343605


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3436_343688

/-- Given a circle and a line of symmetry, this theorem proves the equation of the symmetric circle. -/
theorem symmetric_circle_equation (x y : ℝ) :
  (x - 3)^2 + (y + 4)^2 = 2 →  -- Original circle equation
  x + y = 0 →               -- Line of symmetry
  (x - 4)^2 + (y + 3)^2 = 2 -- Symmetric circle equation
:= by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3436_343688


namespace NUMINAMATH_CALUDE_original_height_is_100_l3436_343646

/-- The rebound factor of the ball -/
def rebound_factor : ℝ := 0.5

/-- The total travel distance when the ball touches the floor for the third time -/
def total_distance : ℝ := 250

/-- Calculates the total travel distance for a ball dropped from height h -/
def calculate_total_distance (h : ℝ) : ℝ :=
  h + 2 * h * rebound_factor + 2 * h * rebound_factor^2

/-- Theorem stating that the original height is 100 cm -/
theorem original_height_is_100 :
  ∃ h : ℝ, h > 0 ∧ calculate_total_distance h = total_distance ∧ h = 100 :=
sorry

end NUMINAMATH_CALUDE_original_height_is_100_l3436_343646


namespace NUMINAMATH_CALUDE_f_zero_eq_zero_l3436_343661

-- Define the function f
def f : ℝ → ℝ := fun x => sorry

-- State the theorem
theorem f_zero_eq_zero :
  (∀ x : ℝ, f (x + 1) = x^2 + 2*x + 1) →
  f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_f_zero_eq_zero_l3436_343661


namespace NUMINAMATH_CALUDE_sin_plus_cos_eq_one_solutions_l3436_343643

theorem sin_plus_cos_eq_one_solutions (x : Real) : 
  0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0 ∨ x = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_eq_one_solutions_l3436_343643


namespace NUMINAMATH_CALUDE_composite_representation_l3436_343608

theorem composite_representation (n : ℕ) (h1 : n > 3) (h2 : ¬ Nat.Prime n) :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ n = a * b + b * c + c * a + 1 := by
  sorry

end NUMINAMATH_CALUDE_composite_representation_l3436_343608


namespace NUMINAMATH_CALUDE_pieces_after_n_divisions_no_2009_pieces_l3436_343690

/-- Represents the number of pieces after n divisions -/
def num_pieces (n : ℕ) : ℕ := 3 * n + 1

/-- Theorem stating the number of pieces after n divisions -/
theorem pieces_after_n_divisions (n : ℕ) :
  num_pieces n = 3 * n + 1 := by sorry

/-- Theorem stating that it's impossible to have 2009 pieces -/
theorem no_2009_pieces :
  ¬ ∃ (n : ℕ), num_pieces n = 2009 := by sorry

end NUMINAMATH_CALUDE_pieces_after_n_divisions_no_2009_pieces_l3436_343690


namespace NUMINAMATH_CALUDE_walter_bus_time_l3436_343647

def wake_up_time : Nat := 6 * 60 + 30
def leave_time : Nat := 7 * 60 + 30
def return_time : Nat := 16 * 60 + 30
def num_classes : Nat := 7
def class_duration : Nat := 45
def lunch_duration : Nat := 40
def additional_time : Nat := 150

def total_away_time : Nat := return_time - leave_time
def school_time : Nat := num_classes * class_duration + lunch_duration + additional_time

theorem walter_bus_time :
  total_away_time - school_time = 35 := by
  sorry

end NUMINAMATH_CALUDE_walter_bus_time_l3436_343647


namespace NUMINAMATH_CALUDE_original_sequence_reappearance_l3436_343631

/-- The cycle length of the letter sequence -/
def letter_cycle_length : ℕ := 8

/-- The cycle length of the digit sequence -/
def digit_cycle_length : ℕ := 5

/-- The line number where the original sequence reappears -/
def reappearance_line : ℕ := 40

theorem original_sequence_reappearance :
  Nat.lcm letter_cycle_length digit_cycle_length = reappearance_line :=
by sorry

end NUMINAMATH_CALUDE_original_sequence_reappearance_l3436_343631


namespace NUMINAMATH_CALUDE_mixed_number_multiplication_problem_solution_l3436_343658

theorem mixed_number_multiplication (a b c d : ℚ) :
  (a + b / c) * (1 / d) = (a * c + b) / (c * d) :=
by sorry

theorem problem_solution : 2 + 4/5 * (1/5) = 14/25 :=
by sorry

end NUMINAMATH_CALUDE_mixed_number_multiplication_problem_solution_l3436_343658


namespace NUMINAMATH_CALUDE_max_achievable_grade_l3436_343642

theorem max_achievable_grade (test1_score test2_score test3_score : ℝ)
  (test1_weight test2_weight test3_weight test4_weight : ℝ)
  (max_extra_credit : ℝ) (target_grade : ℝ) :
  test1_score = 95 ∧ test2_score = 80 ∧ test3_score = 90 ∧
  test1_weight = 0.25 ∧ test2_weight = 0.3 ∧ test3_weight = 0.25 ∧ test4_weight = 0.2 ∧
  max_extra_credit = 5 ∧ target_grade = 93 →
  let current_weighted_grade := test1_score * test1_weight + test2_score * test2_weight + test3_score * test3_weight
  let max_fourth_test_score := 100 + max_extra_credit
  let max_achievable_grade := current_weighted_grade + max_fourth_test_score * test4_weight
  max_achievable_grade < target_grade ∧ max_achievable_grade = 91.25 :=
by sorry

end NUMINAMATH_CALUDE_max_achievable_grade_l3436_343642


namespace NUMINAMATH_CALUDE_tennis_ball_distribution_l3436_343636

theorem tennis_ball_distribution (total : Nat) (containers : Nat) : 
  total = 100 → containers = 5 → (total / 2) / containers = 10 := by
  sorry

end NUMINAMATH_CALUDE_tennis_ball_distribution_l3436_343636


namespace NUMINAMATH_CALUDE_divide_multiply_result_l3436_343603

theorem divide_multiply_result (x : ℝ) (h : x = 4.5) : (x / 6) * 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_divide_multiply_result_l3436_343603


namespace NUMINAMATH_CALUDE_probability_theorem_l3436_343667

/-- Parallelogram with given vertices -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The specific parallelogram ABCD from the problem -/
def ABCD : Parallelogram :=
  { A := (9, 4)
    B := (3, -2)
    C := (-3, -2)
    D := (3, 4) }

/-- Probability of a point in the parallelogram being not above the x-axis -/
def probability_not_above_x_axis (p : Parallelogram) : ℚ :=
  1/2

/-- Theorem stating the probability for the given parallelogram -/
theorem probability_theorem :
  probability_not_above_x_axis ABCD = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l3436_343667


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3436_343670

/-- A geometric sequence with positive terms where a_1 and a_{99} are roots of x^2 - 10x + 16 = 0 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) ∧
  a 1 * a 99 = 16 ∧
  a 1 + a 99 = 10

theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 20 * a 50 * a 80 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3436_343670


namespace NUMINAMATH_CALUDE_game_cost_calculation_l3436_343673

theorem game_cost_calculation (total_earned : ℕ) (blade_cost : ℕ) (num_games : ℕ) 
  (h1 : total_earned = 69)
  (h2 : blade_cost = 24)
  (h3 : num_games = 9)
  (h4 : total_earned ≥ blade_cost) :
  (total_earned - blade_cost) / num_games = 5 := by
  sorry

end NUMINAMATH_CALUDE_game_cost_calculation_l3436_343673


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3436_343622

theorem polynomial_remainder (x : ℝ) : 
  let p : ℝ → ℝ := λ x => x^5 - 2*x^3 + 4*x + 5
  p 2 = 29 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3436_343622


namespace NUMINAMATH_CALUDE_noras_muffin_sales_l3436_343635

/-- Represents the muffin fundraising problem -/
def muffin_fundraiser (target : ℕ) (muffins_per_pack : ℕ) (packs_per_case : ℕ) (price_per_muffin : ℕ) : Prop :=
  ∃ (cases : ℕ),
    cases * (packs_per_case * (muffins_per_pack * price_per_muffin)) = target

/-- Theorem stating the solution to Nora's muffin fundraising problem -/
theorem noras_muffin_sales : muffin_fundraiser 120 4 3 2 :=
  sorry

end NUMINAMATH_CALUDE_noras_muffin_sales_l3436_343635


namespace NUMINAMATH_CALUDE_number_problem_l3436_343696

theorem number_problem (x : ℚ) : x - (3/5) * x = 64 → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3436_343696


namespace NUMINAMATH_CALUDE_larry_stickers_l3436_343614

theorem larry_stickers (initial : ℕ) (lost : ℕ) (gained : ℕ) 
  (h1 : initial = 193) 
  (h2 : lost = 6) 
  (h3 : gained = 12) : 
  initial - lost + gained = 199 := by
  sorry

end NUMINAMATH_CALUDE_larry_stickers_l3436_343614


namespace NUMINAMATH_CALUDE_min_translation_for_symmetry_l3436_343626

/-- The minimum translation that makes sin(3x + π/6) symmetric about y-axis -/
theorem min_translation_for_symmetry :
  let f (x : ℝ) := Real.sin (3 * x + π / 6)
  ∃ (m : ℝ), m > 0 ∧
    (∀ (x : ℝ), f (x - m) = f (-x - m) ∨ f (x + m) = f (-x + m)) ∧
    (∀ (m' : ℝ), m' > 0 → 
      (∀ (x : ℝ), f (x - m') = f (-x - m') ∨ f (x + m') = f (-x + m')) →
      m ≤ m') ∧
    m = π / 9 := by
  sorry

end NUMINAMATH_CALUDE_min_translation_for_symmetry_l3436_343626


namespace NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l3436_343678

-- Define the base 10 number
def base_10_num : ℕ := 3500

-- Define the base 7 representation
def base_7_repr : List ℕ := [1, 3, 1, 3, 0]

-- Function to convert a list of digits in base 7 to a natural number
def to_nat (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

-- Theorem stating the equivalence
theorem base_10_to_base_7_conversion :
  base_10_num = to_nat base_7_repr :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l3436_343678


namespace NUMINAMATH_CALUDE_reference_city_stores_l3436_343664

/-- The number of stores in the reference city -/
def stores : ℕ := sorry

/-- The number of hospitals in the reference city -/
def hospitals : ℕ := 500

/-- The number of schools in the reference city -/
def schools : ℕ := 200

/-- The number of police stations in the reference city -/
def police_stations : ℕ := 20

/-- The total number of buildings in the new city -/
def new_city_buildings : ℕ := 2175

theorem reference_city_stores :
  stores / 2 + 2 * hospitals + (schools - 50) + (police_stations + 5) = new_city_buildings →
  stores = 2000 := by
  sorry

end NUMINAMATH_CALUDE_reference_city_stores_l3436_343664


namespace NUMINAMATH_CALUDE_frank_fence_length_l3436_343683

/-- Given a rectangular yard with one side of 40 feet and an area of 320 square feet,
    the perimeter minus one side is 56 feet. -/
theorem frank_fence_length :
  ∀ (length width : ℝ),
    length = 40 →
    length * width = 320 →
    2 * width + length = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_frank_fence_length_l3436_343683


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l3436_343671

theorem unique_root_quadratic (k : ℝ) : 
  (∃! a : ℝ, (k^2 - 9) * a^2 - 2 * (k + 1) * a + 1 = 0) → 
  (k = 3 ∨ k = -3 ∨ k = -5) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l3436_343671


namespace NUMINAMATH_CALUDE_fraction_equals_121_l3436_343694

theorem fraction_equals_121 : (1100^2 : ℚ) / (260^2 - 240^2) = 121 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_121_l3436_343694


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_algebraic_simplification_l3436_343684

-- Part 1: Quadratic equation
theorem quadratic_equation_solution (x : ℝ) : 
  2 * x^2 - 3 * x + 1 = 0 ↔ x = 1/2 ∨ x = 1 := by sorry

-- Part 2: Algebraic simplification
theorem algebraic_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  ((a^2 - b^2) / (a^2 - 2*a*b + b^2) + a / (b - a)) / (b^2 / (a^2 - a*b)) = a / b := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_algebraic_simplification_l3436_343684


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3436_343629

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) →  -- a_n is a geometric sequence
  a 7 * a 11 = 6 →                           -- a_7 * a_11 = 6
  a 4 + a 14 = 5 →                           -- a_4 + a_14 = 5
  (a 20 / a 10 = 3/2 ∨ a 20 / a 10 = 2/3) :=  -- a_20 / a_10 is either 3/2 or 2/3
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3436_343629


namespace NUMINAMATH_CALUDE_complex_fraction_equation_l3436_343659

theorem complex_fraction_equation (y : ℚ) : 
  3 + 1 / (1 + 1 / (3 + 3 / (4 + y))) = 169 / 53 → y = -605 / 119 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equation_l3436_343659


namespace NUMINAMATH_CALUDE_ac_length_l3436_343606

/-- Given a line segment AB of length 4 with a point C on it, prove that if AC is the mean
    proportional between AB and BC, then the length of AC is 2√5 - 2. -/
theorem ac_length (AB : ℝ) (C : ℝ) (hAB : AB = 4) (hC : 0 ≤ C ∧ C ≤ AB) 
  (hMean : C^2 = AB * (AB - C)) : C = 2 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_l3436_343606


namespace NUMINAMATH_CALUDE_robbers_river_crossing_impossibility_l3436_343674

theorem robbers_river_crossing_impossibility :
  ∀ (n : ℕ) (trips : ℕ → ℕ → Prop),
    n = 40 →
    (∀ i j, i < n → j < n → i ≠ j → (trips i j ∨ trips j i)) →
    (∀ i j k, i < n → j < n → k < n → i ≠ j → j ≠ k → i ≠ k →
      ¬(trips i j ∧ trips i k)) →
    False :=
by
  sorry

end NUMINAMATH_CALUDE_robbers_river_crossing_impossibility_l3436_343674


namespace NUMINAMATH_CALUDE_expression_value_l3436_343662

theorem expression_value (x y z : ℝ) 
  (h1 : (1 / (y + z)) + (1 / (x + z)) + (1 / (x + y)) = 5)
  (h2 : x + y + z = 2) :
  (x / (y + z)) + (y / (x + z)) + (z / (x + y)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3436_343662


namespace NUMINAMATH_CALUDE_min_squares_sum_l3436_343672

theorem min_squares_sum (n : ℕ) (h1 : n < 8) (h2 : ∃ a : ℕ, 3 * n + 1 = a ^ 2) :
  (∃ k : ℕ, (∃ (x y z : ℕ), n + 1 = x^2 + y^2 + z^2) ∧
            (∀ m : ℕ, m < k → ¬∃ (a b c : ℕ), n + 1 = a^2 + b^2 + c^2 ∧ 
              (∀ i : ℕ, i > m → c^2 = 0))) ∧
  (∀ k : ℕ, (∃ (x y z : ℕ), n + 1 = x^2 + y^2 + z^2) ∧
            (∀ m : ℕ, m < k → ¬∃ (a b c : ℕ), n + 1 = a^2 + b^2 + c^2 ∧ 
              (∀ i : ℕ, i > m → c^2 = 0)) → k ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_squares_sum_l3436_343672


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3436_343625

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given that the man is 30 years older than his son and the son's current age is 28 years. -/
theorem mans_age_to_sons_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 28 →
    man_age = son_age + 30 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3436_343625


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_abc_l3436_343624

def is_valid_abc (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = 2) ∧ (n % 10 = 7)

def largest_abc : ℕ := 297
def smallest_abc : ℕ := 207

theorem sum_of_largest_and_smallest_abc :
  is_valid_abc largest_abc ∧
  is_valid_abc smallest_abc ∧
  (∀ n : ℕ, is_valid_abc n → smallest_abc ≤ n ∧ n ≤ largest_abc) ∧
  largest_abc + smallest_abc = 504 :=
sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_abc_l3436_343624


namespace NUMINAMATH_CALUDE_village_assistants_selection_l3436_343613

theorem village_assistants_selection (n m k : ℕ) : 
  n = 10 → m = 3 → k = 2 →
  (Nat.choose (n - 1) m) - (Nat.choose (n - k - 1) m) = 49 := by
  sorry

end NUMINAMATH_CALUDE_village_assistants_selection_l3436_343613


namespace NUMINAMATH_CALUDE_eds_walking_speed_l3436_343668

/-- Proves that Ed's walking speed is 4 km/h given the specified conditions -/
theorem eds_walking_speed (total_distance : ℝ) (sandys_speed : ℝ) (sandys_distance : ℝ) (time_difference : ℝ) :
  total_distance = 52 →
  sandys_speed = 6 →
  sandys_distance = 36 →
  time_difference = 2 →
  ∃ (eds_speed : ℝ), eds_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_eds_walking_speed_l3436_343668


namespace NUMINAMATH_CALUDE_expected_other_marbles_is_two_l3436_343610

/-- Represents the distribution of marble colors in Percius's collection -/
structure MarbleCollection where
  clear_percent : ℚ
  black_percent : ℚ
  other_percent : ℚ
  sum_to_one : clear_percent + black_percent + other_percent = 1

/-- Calculates the expected number of marbles of a certain color when taking a sample -/
def expected_marbles (collection : MarbleCollection) (sample_size : ℕ) (color_percent : ℚ) : ℚ :=
  color_percent * sample_size

/-- Theorem: The expected number of other-colored marbles in a sample of 5 is 2 -/
theorem expected_other_marbles_is_two (collection : MarbleCollection) 
    (h1 : collection.clear_percent = 2/5)
    (h2 : collection.black_percent = 1/5) :
    expected_marbles collection 5 collection.other_percent = 2 := by
  sorry

#eval expected_marbles ⟨2/5, 1/5, 2/5, by norm_num⟩ 5 (2/5)

end NUMINAMATH_CALUDE_expected_other_marbles_is_two_l3436_343610


namespace NUMINAMATH_CALUDE_flower_pot_cost_difference_l3436_343665

theorem flower_pot_cost_difference (n : Nat) (largest_cost difference total_cost : ℚ) : 
  n = 6 ∧ 
  largest_cost = 175/100 ∧ 
  difference = 15/100 ∧
  total_cost = (n : ℚ) * largest_cost - ((n - 1) * n / 2 : ℚ) * difference →
  total_cost = 825/100 := by
sorry

end NUMINAMATH_CALUDE_flower_pot_cost_difference_l3436_343665


namespace NUMINAMATH_CALUDE_lukes_trip_time_l3436_343618

/-- Calculates the total trip time for Luke's journey to London --/
theorem lukes_trip_time :
  let bus_time : ℚ := 75 / 60
  let walk_time : ℚ := 15 / 60
  let wait_time : ℚ := 2 * walk_time
  let train_time : ℚ := 6
  bus_time + walk_time + wait_time + train_time = 8 :=
by sorry

end NUMINAMATH_CALUDE_lukes_trip_time_l3436_343618


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3436_343652

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h_arithmetic : (a + b) / 2 = 5/2) (h_geometric : Real.sqrt (a * b) = Real.sqrt 6) :
  let e := Real.sqrt (1 + b^2 / a^2)
  e = Real.sqrt 13 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3436_343652


namespace NUMINAMATH_CALUDE_p_plus_q_value_l3436_343609

theorem p_plus_q_value (p q : ℝ) 
  (hp : p^3 - 12*p^2 + 25*p - 75 = 0)
  (hq : 10*q^3 - 75*q^2 - 375*q + 3750 = 0) : 
  p + q = -5/2 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_value_l3436_343609


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3436_343656

theorem product_of_three_numbers (x y z : ℚ) : 
  x + y + z = 190 ∧ 
  8 * x = y - 7 ∧ 
  8 * x = z + 11 ∧
  x ≤ y ∧ 
  x ≤ z →
  x * y * z = (97 * 215 * 161) / 108 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3436_343656


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3436_343666

theorem perfect_square_condition (x : ℤ) : 
  (∃ k : ℤ, x^2 - 14*x - 256 = k^2) ↔ (x = 15 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3436_343666


namespace NUMINAMATH_CALUDE_hidden_digit_problem_l3436_343698

theorem hidden_digit_problem :
  ∃! (x : ℕ), x ≠ 0 ∧ x < 10 ∧ ((10 * x + x) + (10 * x + x) + 1) * x = 100 * x + 10 * x + x :=
by
  sorry

end NUMINAMATH_CALUDE_hidden_digit_problem_l3436_343698


namespace NUMINAMATH_CALUDE_tangent_and_cosine_relations_l3436_343604

theorem tangent_and_cosine_relations (θ : Real) (h : Real.tan θ = 2) :
  (Real.tan (π / 4 - θ) = -1 / 3) ∧ (Real.cos (2 * θ) = -3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_cosine_relations_l3436_343604


namespace NUMINAMATH_CALUDE_no_natural_number_with_digit_product_6552_l3436_343677

theorem no_natural_number_with_digit_product_6552 :
  ¬ ∃ n : ℕ, (n.digits 10).prod = 6552 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_number_with_digit_product_6552_l3436_343677


namespace NUMINAMATH_CALUDE_f_properties_l3436_343682

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x + 5 else -2 * x + 8

theorem f_properties :
  (f 2 = 4) ∧
  (f (f (-1)) = 0) ∧
  (∀ x, f x ≥ 4 ↔ -1 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3436_343682


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3436_343637

/-- Proves that for a parallelogram with area 450 sq m and altitude twice the base, the base length is 15 m -/
theorem parallelogram_base_length 
  (area : ℝ) 
  (base : ℝ) 
  (altitude : ℝ) 
  (h1 : area = 450) 
  (h2 : altitude = 2 * base) 
  (h3 : area = base * altitude) : 
  base = 15 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3436_343637


namespace NUMINAMATH_CALUDE_equation_equivalent_to_line_segments_l3436_343697

def satisfies_equation (x y : ℝ) : Prop :=
  3 * |x - 1| + 2 * |y + 2| = 6

def within_rectangle (x y : ℝ) : Prop :=
  -1 ≤ x ∧ x ≤ 3 ∧ -5 ≤ y ∧ y ≤ 1

def on_line_segments (x y : ℝ) : Prop :=
  (3*x + 2*y = 5 ∨ -3*x + 2*y = -1 ∨ 3*x - 2*y = 13 ∨ -3*x - 2*y = 7) ∧ within_rectangle x y

theorem equation_equivalent_to_line_segments :
  ∀ x y : ℝ, satisfies_equation x y ↔ on_line_segments x y :=
sorry

end NUMINAMATH_CALUDE_equation_equivalent_to_line_segments_l3436_343697


namespace NUMINAMATH_CALUDE_felipe_build_time_l3436_343632

/-- Represents the time taken by each person to build their house, including break time. -/
structure BuildTime where
  felipe : ℝ
  emilio : ℝ
  carlos : ℝ

/-- Represents the break time taken by each person during construction. -/
structure BreakTime where
  felipe : ℝ
  emilio : ℝ
  carlos : ℝ

/-- The theorem stating Felipe's total build time is 27 months given the problem conditions. -/
theorem felipe_build_time (bt : BuildTime) (brt : BreakTime) : bt.felipe = 27 :=
  by
  have h1 : bt.felipe = bt.emilio / 2 := sorry
  have h2 : bt.carlos = bt.felipe + bt.emilio := sorry
  have h3 : bt.felipe + bt.emilio + bt.carlos = 10.5 * 12 := sorry
  have h4 : brt.felipe = 6 := sorry
  have h5 : brt.emilio = 2 * brt.felipe := sorry
  have h6 : brt.carlos = brt.emilio / 2 := sorry
  have h7 : bt.felipe + brt.felipe = 27 := sorry
  sorry

#check felipe_build_time

end NUMINAMATH_CALUDE_felipe_build_time_l3436_343632


namespace NUMINAMATH_CALUDE_min_n_value_l3436_343653

theorem min_n_value (m n : ℝ) : 
  (∃ x : ℝ, x^2 + (m - 2023) * x + (n - 1) = 0 ∧ 
   ∀ y : ℝ, y^2 + (m - 2023) * y + (n - 1) = 0 → y = x) → 
  n ≥ 1 ∧ ∃ m₀ : ℝ, ∃ x₀ : ℝ, x₀^2 + (m₀ - 2023) * x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_n_value_l3436_343653


namespace NUMINAMATH_CALUDE_det_A_squared_minus_2A_l3436_343645

/-- Given a 2x2 matrix A, prove that det(A^2 - 2A) = 25 -/
theorem det_A_squared_minus_2A (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A = ![![1, 3], ![2, 1]]) : 
  Matrix.det (A ^ 2 - 2 • A) = 25 := by
sorry

end NUMINAMATH_CALUDE_det_A_squared_minus_2A_l3436_343645


namespace NUMINAMATH_CALUDE_f_value_at_3_l3436_343648

theorem f_value_at_3 (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^7 + a*x^5 + b*x - 5) 
  (h2 : f (-3) = 5) : f 3 = -15 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l3436_343648


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l3436_343607

theorem infinite_solutions_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l3436_343607


namespace NUMINAMATH_CALUDE_difference_in_amounts_l3436_343663

/-- Represents the three products A, B, and C --/
inductive Product
| A
| B
| C

/-- The initial price of a product --/
def initialPrice (p : Product) : ℝ :=
  match p with
  | Product.A => 100
  | Product.B => 150
  | Product.C => 200

/-- The price increase percentage for a product --/
def priceIncrease (p : Product) : ℝ :=
  match p with
  | Product.A => 0.10
  | Product.B => 0.15
  | Product.C => 0.20

/-- The quantity bought after price increase as a fraction of initial quantity --/
def quantityAfterIncrease (p : Product) : ℝ :=
  match p with
  | Product.A => 0.90
  | Product.B => 0.85
  | Product.C => 0.80

/-- The discount percentage --/
def discount : ℝ := 0.05

/-- The additional quantity bought on discount day as a fraction of initial quantity --/
def additionalQuantity (p : Product) : ℝ :=
  match p with
  | Product.A => 0.10
  | Product.B => 0.15
  | Product.C => 0.20

/-- The total amount paid on the price increase day --/
def amountOnIncreaseDay : ℝ :=
  (initialPrice Product.A * (1 + priceIncrease Product.A) * quantityAfterIncrease Product.A) +
  (initialPrice Product.B * (1 + priceIncrease Product.B) * quantityAfterIncrease Product.B) +
  (initialPrice Product.C * (1 + priceIncrease Product.C) * quantityAfterIncrease Product.C)

/-- The total amount paid on the discount day --/
def amountOnDiscountDay : ℝ :=
  (initialPrice Product.A * (1 - discount) * (1 + additionalQuantity Product.A)) +
  (initialPrice Product.B * (1 - discount) * (1 + additionalQuantity Product.B)) +
  (initialPrice Product.C * (1 - discount) * (1 + additionalQuantity Product.C))

/-- The theorem stating the difference in amounts paid --/
theorem difference_in_amounts : amountOnIncreaseDay - amountOnDiscountDay = 58.75 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_amounts_l3436_343663


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_with_two_even_two_odd_l3436_343617

/-- A function that checks if a number has two even and two odd digits -/
def hasTwoEvenTwoOddDigits (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.filter (·.mod 2 = 0)).length = 2 ∧ 
  (digits.filter (·.mod 2 = 1)).length = 2

/-- The smallest positive four-digit number divisible by 9 with two even and two odd digits -/
def smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd : ℕ := 1089

theorem smallest_four_digit_divisible_by_9_with_two_even_two_odd :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n.mod 9 = 0 ∧ hasTwoEvenTwoOddDigits n →
    smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd ≤ n) ∧
  1000 ≤ smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd ∧
  smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd < 10000 ∧
  smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd.mod 9 = 0 ∧
  hasTwoEvenTwoOddDigits smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_with_two_even_two_odd_l3436_343617
