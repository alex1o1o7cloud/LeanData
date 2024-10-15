import Mathlib

namespace NUMINAMATH_CALUDE_sequence_property_l1547_154708

theorem sequence_property (a : ℕ → ℝ) 
  (h_pos : ∀ n : ℕ, n ≥ 1 → a n > 0)
  (h_ineq : ∀ n : ℕ, n ≥ 1 → (a (n + 1))^2 + a n * a (n + 2) ≤ a n + a (n + 2)) :
  a 21022 ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l1547_154708


namespace NUMINAMATH_CALUDE_guessing_game_theorem_l1547_154714

/-- The guessing game between Banana and Corona -/
def GuessGame (n k : ℕ) : Prop :=
  n > 0 ∧ k > 0 ∧ k ≤ 2^n

/-- Corona can determine x in finitely many turns -/
def CanDetermine (n k : ℕ) : Prop :=
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ n → ∃ t : ℕ, t > 0 ∧ (∀ y : ℕ, 1 ≤ y ∧ y ≤ n → y = x)

/-- The main theorem: Corona can determine x iff k ≤ 2^(n-1) -/
theorem guessing_game_theorem (n k : ℕ) :
  GuessGame n k → (CanDetermine n k ↔ k ≤ 2^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_guessing_game_theorem_l1547_154714


namespace NUMINAMATH_CALUDE_sequence_problem_l1547_154757

theorem sequence_problem (a : ℕ → ℤ) (h1 : a 5 = 14) (h2 : ∀ n : ℕ, a (n + 1) - a n = n + 1) : a 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1547_154757


namespace NUMINAMATH_CALUDE_sum_of_ages_l1547_154790

/-- Represents the ages of Xavier and Yasmin -/
structure Ages where
  xavier : ℕ
  yasmin : ℕ

/-- The current ages of Xavier and Yasmin satisfy the given conditions -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.xavier = 2 * ages.yasmin ∧ ages.xavier + 6 = 30

/-- Theorem: The sum of Xavier's and Yasmin's current ages is 36 -/
theorem sum_of_ages (ages : Ages) (h : satisfies_conditions ages) : 
  ages.xavier + ages.yasmin = 36 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_ages_l1547_154790


namespace NUMINAMATH_CALUDE_opposite_of_two_thirds_l1547_154761

theorem opposite_of_two_thirds :
  (-(2 : ℚ) / 3) = (-1 : ℚ) * (2 : ℚ) / 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_two_thirds_l1547_154761


namespace NUMINAMATH_CALUDE_union_of_sets_l1547_154732

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem union_of_sets :
  ∃ x : ℝ, (B x ∩ A x = {9}) → (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1547_154732


namespace NUMINAMATH_CALUDE_cosine_inequality_equivalence_l1547_154706

theorem cosine_inequality_equivalence (x : Real) : 
  (∃ k : Int, (-π/6 + 2*π*k < x ∧ x < π/6 + 2*π*k) ∨ 
              (2*π/3 + 2*π*k < x ∧ x < 4*π/3 + 2*π*k)) ↔ 
  (Real.cos (2*x) - 4*Real.cos (π/4)*Real.cos (5*π/12)*Real.cos x + 
   Real.cos (5*π/6) + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_equivalence_l1547_154706


namespace NUMINAMATH_CALUDE_knife_sharpening_problem_l1547_154776

/-- Represents the pricing structure for knife sharpening -/
structure KnifeSharpeningPrices where
  first_knife : ℕ → ℕ
  next_three_knives : ℕ → ℕ
  remaining_knives : ℕ → ℕ

/-- Calculates the total cost of sharpening knives -/
def total_cost (prices : KnifeSharpeningPrices) (num_knives : ℕ) : ℕ :=
  prices.first_knife 1 +
  prices.next_three_knives (min 3 (num_knives - 1)) +
  prices.remaining_knives (max 0 (num_knives - 4))

/-- Theorem stating that given the pricing structure and total cost, the number of knives is 9 -/
theorem knife_sharpening_problem (prices : KnifeSharpeningPrices) 
  (h1 : prices.first_knife 1 = 5)
  (h2 : ∀ n : ℕ, n ≤ 3 → prices.next_three_knives n = 4 * n)
  (h3 : ∀ n : ℕ, prices.remaining_knives n = 3 * n)
  (h4 : total_cost prices 9 = 32) :
  ∃ (n : ℕ), total_cost prices n = 32 ∧ n = 9 := by
  sorry


end NUMINAMATH_CALUDE_knife_sharpening_problem_l1547_154776


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_equation_solution_expression_evaluation_l1547_154736

-- Part 1: System of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (x - 4 < 2 * (x - 1) ∧ (1 + 2 * x) / 3 ≥ x) ↔ -2 < x ∧ x ≤ 1 := by sorry

-- Part 2: Equation solution
theorem equation_solution :
  ∃! x : ℝ, (x - 2) / (x - 3) = 2 - 1 / (3 - x) ∧ x = 3 := by sorry

-- Part 3: Expression simplification and evaluation
theorem expression_evaluation (x : ℝ) (h : x = 3) :
  (1 - 1 / (x + 2)) / ((x^2 - 1) / (x + 2)) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_equation_solution_expression_evaluation_l1547_154736


namespace NUMINAMATH_CALUDE_teresa_bought_six_hardcovers_l1547_154726

/-- The number of hardcover books Teresa bought -/
def num_hardcovers : ℕ := sorry

/-- The number of paperback books Teresa bought -/
def num_paperbacks : ℕ := sorry

/-- The total number of books Teresa bought -/
def total_books : ℕ := 12

/-- The cost of a hardcover book -/
def hardcover_cost : ℕ := 30

/-- The cost of a paperback book -/
def paperback_cost : ℕ := 18

/-- The total amount Teresa spent -/
def total_spent : ℕ := 288

/-- Theorem stating that Teresa bought 6 hardcover books -/
theorem teresa_bought_six_hardcovers :
  num_hardcovers = 6 ∧
  num_hardcovers ≥ 4 ∧
  num_hardcovers + num_paperbacks = total_books ∧
  num_hardcovers * hardcover_cost + num_paperbacks * paperback_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_teresa_bought_six_hardcovers_l1547_154726


namespace NUMINAMATH_CALUDE_solution_set_f_g_has_zero_condition_l1547_154701

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |3 + a|

-- Part I
theorem solution_set_f (x : ℝ) : 
  f 3 x > 6 ↔ x < -4 ∨ x > 2 := by sorry

-- Part II
theorem g_has_zero_condition (a : ℝ) :
  (∃ x, g a x = 0) → a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_g_has_zero_condition_l1547_154701


namespace NUMINAMATH_CALUDE_min_a_for_inequality_l1547_154788

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_inequality_l1547_154788


namespace NUMINAMATH_CALUDE_harmonic_arithmetic_mean_inequality_l1547_154730

theorem harmonic_arithmetic_mean_inequality {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let h := 3 / ((1 / a) + (1 / b) + (1 / c))
  let m := (a + b + c) / 3
  h ≤ m ∧ (h = m ↔ a = b ∧ b = c) := by
  sorry

#check harmonic_arithmetic_mean_inequality

end NUMINAMATH_CALUDE_harmonic_arithmetic_mean_inequality_l1547_154730


namespace NUMINAMATH_CALUDE_sqrt_1936_div_11_l1547_154772

theorem sqrt_1936_div_11 : Real.sqrt 1936 / 11 = 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_1936_div_11_l1547_154772


namespace NUMINAMATH_CALUDE_age_difference_l1547_154770

/-- Given that B is currently 42 years old, and in 10 years A will be twice as old as B was 10 years ago, prove that A is 12 years older than B. -/
theorem age_difference (A B : ℕ) : B = 42 → A + 10 = 2 * (B - 10) → A - B = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1547_154770


namespace NUMINAMATH_CALUDE_stratified_sampling_second_year_l1547_154723

/-- The number of students in the second year of high school -/
def second_year_students : ℕ := 750

/-- The probability of a student being selected in the stratified sampling -/
def selection_probability : ℚ := 2 / 100

/-- The number of students to be drawn from the second year -/
def students_drawn : ℕ := 15

/-- Theorem stating that the number of students drawn from the second year
    is equal to the product of the total number of second-year students
    and the selection probability -/
theorem stratified_sampling_second_year :
  (second_year_students : ℚ) * selection_probability = students_drawn := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_year_l1547_154723


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l1547_154785

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 2187 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l1547_154785


namespace NUMINAMATH_CALUDE_inequality_solution_l1547_154746

theorem inequality_solution (x : ℝ) :
  x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 →
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔
  (x < 1 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1547_154746


namespace NUMINAMATH_CALUDE_union_equals_reals_l1547_154711

-- Define sets A and B
def A : Set ℝ := {x | Real.log x > 0}
def B : Set ℝ := {x | x ≤ 1}

-- State the theorem
theorem union_equals_reals : A ∪ B = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_union_equals_reals_l1547_154711


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1547_154712

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a₁ d : ℝ, ∀ n : ℕ, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 5 + a 10 = 12 →
  3 * a 7 + a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1547_154712


namespace NUMINAMATH_CALUDE_power_ranger_stickers_l1547_154773

theorem power_ranger_stickers (total : ℕ) (difference : ℕ) (first_box : ℕ) : 
  total = 58 → difference = 12 → first_box + (first_box + difference) = total → first_box = 23 := by
  sorry

end NUMINAMATH_CALUDE_power_ranger_stickers_l1547_154773


namespace NUMINAMATH_CALUDE_remainder_theorem_l1547_154705

theorem remainder_theorem (n : ℤ) (h : n % 50 = 23) : (3 * n - 5) % 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1547_154705


namespace NUMINAMATH_CALUDE_friday_dressing_time_l1547_154766

/-- Represents the dressing times for each day of the week -/
structure DressingTimes where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the average dressing time for the week -/
def weeklyAverage (times : DressingTimes) : ℚ :=
  (times.monday + times.tuesday + times.wednesday + times.thursday + times.friday) / 5

/-- The old average dressing time -/
def oldAverage : ℚ := 3

/-- The given dressing times for Monday through Thursday -/
def givenTimes : DressingTimes := {
  monday := 2
  tuesday := 4
  wednesday := 3
  thursday := 4
  friday := 0  -- We'll solve for this
}

theorem friday_dressing_time :
  ∃ (fridayTime : ℕ),
    let newTimes := { givenTimes with friday := fridayTime }
    weeklyAverage newTimes = oldAverage ∧ fridayTime = 2 := by
  sorry


end NUMINAMATH_CALUDE_friday_dressing_time_l1547_154766


namespace NUMINAMATH_CALUDE_dice_configuration_dots_l1547_154718

/-- Represents a single die face -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the configuration of four dice -/
structure DiceConfiguration :=
  (faceA : DieFace)
  (faceB : DieFace)
  (faceC : DieFace)
  (faceD : DieFace)

/-- Counts the number of dots on a die face -/
def dotCount (face : DieFace) : Nat :=
  match face with
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

/-- The specific configuration of dice in the problem -/
def problemConfiguration : DiceConfiguration :=
  { faceA := DieFace.three
  , faceB := DieFace.five
  , faceC := DieFace.six
  , faceD := DieFace.five }

theorem dice_configuration_dots :
  dotCount problemConfiguration.faceA = 3 ∧
  dotCount problemConfiguration.faceB = 5 ∧
  dotCount problemConfiguration.faceC = 6 ∧
  dotCount problemConfiguration.faceD = 5 := by
  sorry

end NUMINAMATH_CALUDE_dice_configuration_dots_l1547_154718


namespace NUMINAMATH_CALUDE_complex_square_expansion_l1547_154775

theorem complex_square_expansion (x y c : ℝ) : 
  (x + Complex.I * y + c)^2 = x^2 + c^2 - y^2 + 2*c*x + Complex.I * (2*x*y + 2*c*y) := by
  sorry

end NUMINAMATH_CALUDE_complex_square_expansion_l1547_154775


namespace NUMINAMATH_CALUDE_square_of_arithmetic_mean_geq_product_l1547_154796

theorem square_of_arithmetic_mean_geq_product (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : c = (a + b) / 2) : c^2 ≥ a * b := by
  sorry

end NUMINAMATH_CALUDE_square_of_arithmetic_mean_geq_product_l1547_154796


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_28_l1547_154767

/-- The sum of n consecutive positive integers starting from a -/
def sumConsecutive (n : ℕ) (a : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Predicate to check if a sequence of n consecutive integers starting from a sums to 28 -/
def isValidSequence (n : ℕ) (a : ℕ) : Prop :=
  a > 0 ∧ sumConsecutive n a = 28

theorem largest_consecutive_sum_28 :
  (∃ a : ℕ, isValidSequence 7 a) ∧
  (∀ n : ℕ, n > 7 → ¬∃ a : ℕ, isValidSequence n a) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_28_l1547_154767


namespace NUMINAMATH_CALUDE_broccoli_area_l1547_154791

/-- Represents the garden and broccoli production --/
structure BroccoliGarden where
  last_year_side : ℝ
  this_year_side : ℝ
  broccoli_increase : ℕ
  this_year_count : ℕ

/-- The conditions of the broccoli garden problem --/
def broccoli_conditions (g : BroccoliGarden) : Prop :=
  g.broccoli_increase = 79 ∧
  g.this_year_count = 1600 ∧
  g.this_year_side ^ 2 = g.this_year_count ∧
  g.this_year_side ^ 2 - g.last_year_side ^ 2 = g.broccoli_increase

/-- The theorem stating that each broccoli takes 1 square foot --/
theorem broccoli_area (g : BroccoliGarden) 
  (h : broccoli_conditions g) : 
  g.this_year_side ^ 2 / g.this_year_count = 1 := by
  sorry


end NUMINAMATH_CALUDE_broccoli_area_l1547_154791


namespace NUMINAMATH_CALUDE_industrial_lubricants_budget_percentage_l1547_154741

theorem industrial_lubricants_budget_percentage
  (total_degrees : ℝ)
  (microphotonics_percent : ℝ)
  (home_electronics_percent : ℝ)
  (food_additives_percent : ℝ)
  (genetically_modified_microorganisms_percent : ℝ)
  (basic_astrophysics_degrees : ℝ)
  (h1 : total_degrees = 360)
  (h2 : microphotonics_percent = 14)
  (h3 : home_electronics_percent = 19)
  (h4 : food_additives_percent = 10)
  (h5 : genetically_modified_microorganisms_percent = 24)
  (h6 : basic_astrophysics_degrees = 90) :
  let total_percent : ℝ := 100
  let basic_astrophysics_percent : ℝ := (basic_astrophysics_degrees / total_degrees) * total_percent
  let known_sectors_percent : ℝ := microphotonics_percent + home_electronics_percent + 
                                   food_additives_percent + genetically_modified_microorganisms_percent
  let industrial_lubricants_percent : ℝ := total_percent - known_sectors_percent - basic_astrophysics_percent
  industrial_lubricants_percent = 8 := by
  sorry

end NUMINAMATH_CALUDE_industrial_lubricants_budget_percentage_l1547_154741


namespace NUMINAMATH_CALUDE_balanced_132_l1547_154738

def is_balanced (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit number
  (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧  -- All digits are different
  n = (10 * (n / 100) + (n / 10) % 10) +
      (10 * (n / 100) + n % 10) +
      (10 * ((n / 10) % 10) + n / 100) +
      (10 * ((n / 10) % 10) + n % 10) +
      (10 * (n % 10) + n / 100) +
      (10 * (n % 10) + (n / 10) % 10)  -- Sum of all possible two-digit numbers

theorem balanced_132 : is_balanced 132 := by
  sorry

end NUMINAMATH_CALUDE_balanced_132_l1547_154738


namespace NUMINAMATH_CALUDE_helga_wrote_250_articles_l1547_154784

/-- Represents Helga's work schedule and article writing capacity --/
structure HelgaWorkSchedule where
  articles_per_half_hour : ℕ
  regular_hours_per_day : ℕ
  regular_days_per_week : ℕ
  extra_hours_thursday : ℕ
  extra_hours_friday : ℕ

/-- Calculates the total number of articles Helga wrote in a week --/
def total_articles_in_week (schedule : HelgaWorkSchedule) : ℕ :=
  let articles_per_hour := schedule.articles_per_half_hour * 2
  let regular_articles := articles_per_hour * schedule.regular_hours_per_day * schedule.regular_days_per_week
  let extra_articles := articles_per_hour * (schedule.extra_hours_thursday + schedule.extra_hours_friday)
  regular_articles + extra_articles

/-- Theorem stating that Helga wrote 250 articles in the given week --/
theorem helga_wrote_250_articles : 
  ∀ (schedule : HelgaWorkSchedule), 
    schedule.articles_per_half_hour = 5 ∧ 
    schedule.regular_hours_per_day = 4 ∧ 
    schedule.regular_days_per_week = 5 ∧ 
    schedule.extra_hours_thursday = 2 ∧ 
    schedule.extra_hours_friday = 3 →
    total_articles_in_week schedule = 250 := by
  sorry

end NUMINAMATH_CALUDE_helga_wrote_250_articles_l1547_154784


namespace NUMINAMATH_CALUDE_unique_solution_mn_l1547_154707

theorem unique_solution_mn (m n : ℕ+) 
  (h1 : (n^4 : ℕ) ∣ 2*m^5 - 1)
  (h2 : (m^4 : ℕ) ∣ 2*n^5 + 1) :
  m = 1 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mn_l1547_154707


namespace NUMINAMATH_CALUDE_cone_height_l1547_154768

/-- Given a cone with slant height 13 cm and lateral area 65π cm², prove its height is 12 cm -/
theorem cone_height (s : ℝ) (l : ℝ) (h : ℝ) : 
  s = 13 → l = 65 * Real.pi → l = Real.pi * s * (l / (Real.pi * s)) → h^2 + (l / (Real.pi * s))^2 = s^2 → h = 12 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l1547_154768


namespace NUMINAMATH_CALUDE_election_votes_theorem_l1547_154783

/-- Proves that in an election with two candidates, if the first candidate got 60% of the votes
    and the second candidate got 240 votes, then the total number of votes was 600. -/
theorem election_votes_theorem (total_votes : ℕ) (first_candidate_percentage : ℚ) 
    (second_candidate_votes : ℕ) : 
    first_candidate_percentage = 60 / 100 →
    second_candidate_votes = 240 →
    (1 - first_candidate_percentage) * total_votes = second_candidate_votes →
    total_votes = 600 := by
  sorry

#check election_votes_theorem

end NUMINAMATH_CALUDE_election_votes_theorem_l1547_154783


namespace NUMINAMATH_CALUDE_calligraphy_class_equation_l1547_154778

theorem calligraphy_class_equation (x : ℕ+) 
  (h1 : 6 * x = planned + 7)
  (h2 : 5 * x = planned - 13)
  : 6 * x - 7 = 5 * x + 13 := by
  sorry

end NUMINAMATH_CALUDE_calligraphy_class_equation_l1547_154778


namespace NUMINAMATH_CALUDE_range_of_m_l1547_154755

-- Define the inequality system
def inequality_system (x m : ℝ) : Prop :=
  x + 5 < 5*x + 1 ∧ x - m > 1

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x > 1

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∀ x, inequality_system x m ↔ solution_set x) →
  m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1547_154755


namespace NUMINAMATH_CALUDE_power_of_three_squared_to_fourth_l1547_154797

theorem power_of_three_squared_to_fourth : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_squared_to_fourth_l1547_154797


namespace NUMINAMATH_CALUDE_intersection_point_properties_l1547_154729

/-- Point P where the given lines intersect -/
def P : ℝ × ℝ := (1, 1)

/-- Line perpendicular to 3x + 4y - 15 = 0 passing through P -/
def l₁ (x y : ℝ) : Prop := 4 * x - 3 * y - 1 = 0

/-- Line with equal intercepts passing through P -/
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0

theorem intersection_point_properties :
  (∃ (x y : ℝ), 2 * x - y - 1 = 0 ∧ x - 2 * y + 1 = 0 ∧ (x, y) = P) ∧
  (∀ (x y : ℝ), l₁ x y ↔ (4 * x - 3 * y - 1 = 0 ∧ (x, y) = P ∨ (x, y) ≠ P)) ∧
  (∀ (x y : ℝ), l₂ x y ↔ (x + y - 2 = 0 ∧ (x, y) = P ∨ (x, y) ≠ P)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_properties_l1547_154729


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1547_154792

/-- Proves that given the conditions of the problem, the principal amount must be 700 --/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_rate : R > 0) : 
  (P * (R + 2) * 4) / 100 = (P * R * 4) / 100 + 56 → P = 700 := by
  sorry

#check simple_interest_problem

end NUMINAMATH_CALUDE_simple_interest_problem_l1547_154792


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1547_154786

theorem greatest_divisor_with_remainders : Nat.gcd (1657 - 6) (2037 - 5) = 127 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1547_154786


namespace NUMINAMATH_CALUDE_muffin_count_l1547_154762

/-- Given a ratio of doughnuts to cookies to muffins and the number of doughnuts,
    calculate the number of muffins. -/
def calculate_muffins (doughnut_ratio : ℕ) (cookie_ratio : ℕ) (muffin_ratio : ℕ) (num_doughnuts : ℕ) : ℕ :=
  (num_doughnuts / doughnut_ratio) * muffin_ratio

/-- Theorem stating that given the ratio 5:3:1 for doughnuts:cookies:muffins
    and 50 doughnuts, there are 10 muffins. -/
theorem muffin_count : calculate_muffins 5 3 1 50 = 10 := by
  sorry

#eval calculate_muffins 5 3 1 50

end NUMINAMATH_CALUDE_muffin_count_l1547_154762


namespace NUMINAMATH_CALUDE_circle_ellipse_tangent_l1547_154744

-- Define the circle M
def circle_M (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*m*x - 3 = 0

-- Define the ellipse C
def ellipse_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1

-- Define the focus F
def focus_F (c : ℝ) : ℝ × ℝ :=
  (-c, 0)

-- Define the line l
def line_l (c : ℝ) (x y : ℝ) : Prop :=
  x = -c

theorem circle_ellipse_tangent (m a c : ℝ) :
  m < 0 →
  (∀ x y, circle_M m x y → (x - 1)^2 + y^2 = 4) →
  (∃ x y, circle_M m x y ∧ line_l c x y) →
  (∀ x y, circle_M m x y ∧ line_l c x y → (x - 1)^2 + y^2 = 4) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_ellipse_tangent_l1547_154744


namespace NUMINAMATH_CALUDE_max_blocks_fit_l1547_154756

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the box dimensions -/
def box : Dimensions := ⟨5, 4, 6⟩

/-- Represents the block dimensions -/
def block : Dimensions := ⟨3, 3, 2⟩

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ := 6

theorem max_blocks_fit :
  (volume box) / (volume block) ≥ max_blocks ∧
  max_blocks * (volume block) ≤ volume box ∧
  max_blocks * block.length ≤ box.length + block.length - 1 ∧
  max_blocks * block.width ≤ box.width + block.width - 1 ∧
  max_blocks * block.height ≤ box.height + block.height - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l1547_154756


namespace NUMINAMATH_CALUDE_cats_remaining_l1547_154709

theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) 
  (h1 : siamese = 13) 
  (h2 : house = 5) 
  (h3 : sold = 10) : 
  siamese + house - sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l1547_154709


namespace NUMINAMATH_CALUDE_identical_sequences_l1547_154748

/-- Given two sequences of n real numbers where the first is strictly increasing,
    and their element-wise sum is strictly increasing,
    prove that the sequences are identical. -/
theorem identical_sequences
  (n : ℕ)
  (a b : Fin n → ℝ)
  (h_a_increasing : ∀ i j : Fin n, i < j → a i < a j)
  (h_sum_increasing : ∀ i j : Fin n, i < j → a i + b i < a j + b j) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_identical_sequences_l1547_154748


namespace NUMINAMATH_CALUDE_greatest_common_factor_90_135_180_l1547_154704

theorem greatest_common_factor_90_135_180 : Nat.gcd 90 (Nat.gcd 135 180) = 45 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_90_135_180_l1547_154704


namespace NUMINAMATH_CALUDE_cans_per_bag_l1547_154774

theorem cans_per_bag (total_bags : ℕ) (total_cans : ℕ) (h1 : total_bags = 9) (h2 : total_cans = 72) :
  total_cans / total_bags = 8 :=
by sorry

end NUMINAMATH_CALUDE_cans_per_bag_l1547_154774


namespace NUMINAMATH_CALUDE_fraction_reduction_l1547_154739

theorem fraction_reduction (n : ℤ) :
  (∃ k : ℤ, n = 7 * k + 1) ↔
  (∃ m : ℤ, 4 * n + 3 = 7 * m) ∧ (∃ l : ℤ, 5 * n + 2 = 7 * l) :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l1547_154739


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1547_154713

-- Define the quadratic function f
def f : ℝ → ℝ := λ x => x^2 - x + 1

-- State the theorem
theorem quadratic_function_properties :
  (f 0 = 1) ∧
  (∀ x, f (x + 1) - f x = 2 * x) ∧
  (f = λ x => x^2 - x + 1) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≥ 3/4) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3/4) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l1547_154713


namespace NUMINAMATH_CALUDE_average_headcount_spring_terms_l1547_154793

def spring_02_03 : ℕ := 10900
def spring_03_04 : ℕ := 10500
def spring_04_05 : ℕ := 10700
def spring_05_06 : ℕ := 11300

def total_headcount : ℕ := spring_02_03 + spring_03_04 + spring_04_05 + spring_05_06
def num_terms : ℕ := 4

theorem average_headcount_spring_terms :
  (total_headcount : ℚ) / num_terms = 10850 := by sorry

end NUMINAMATH_CALUDE_average_headcount_spring_terms_l1547_154793


namespace NUMINAMATH_CALUDE_river_flow_speed_l1547_154749

/-- Proves that the speed of river flow is 2 km/hr given the conditions of the boat journey -/
theorem river_flow_speed (distance : ℝ) (boat_speed : ℝ) (total_time : ℝ) :
  distance = 48 →
  boat_speed = 6 →
  total_time = 18 →
  ∃ (river_speed : ℝ),
    river_speed > 0 ∧
    (distance / (boat_speed - river_speed) + distance / (boat_speed + river_speed) = total_time) ∧
    river_speed = 2 := by
  sorry


end NUMINAMATH_CALUDE_river_flow_speed_l1547_154749


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1547_154759

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≥ 5) →
  (∀ x ∈ Set.Icc 1 2, x^2 ≤ a) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x^2 ≤ a) → (a ≥ 5)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1547_154759


namespace NUMINAMATH_CALUDE_exists_polygon_9_exists_polygon_8_l1547_154747

/-- A polygon is represented as a list of points in the plane -/
def Polygon := List (ℝ × ℝ)

/-- Check if three points are collinear -/
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

/-- Property: each side of the polygon lies on a line containing at least one additional vertex -/
def satisfiesProperty (poly : Polygon) : Prop :=
  ∀ i j : Fin poly.length,
    i ≠ j →
    ∃ k : Fin poly.length, k ≠ i ∧ k ≠ j ∧
      collinear (poly.get i) (poly.get j) (poly.get k)

/-- Theorem: There exists a polygon with at most 9 vertices satisfying the property -/
theorem exists_polygon_9 : ∃ poly : Polygon, poly.length ≤ 9 ∧ satisfiesProperty poly :=
  sorry

/-- Theorem: There exists a polygon with at most 8 vertices satisfying the property -/
theorem exists_polygon_8 : ∃ poly : Polygon, poly.length ≤ 8 ∧ satisfiesProperty poly :=
  sorry

end NUMINAMATH_CALUDE_exists_polygon_9_exists_polygon_8_l1547_154747


namespace NUMINAMATH_CALUDE_water_depth_in_specific_tank_l1547_154737

/-- Represents a horizontal cylindrical water tank -/
structure CylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the possible depths of water in a cylindrical tank given the surface area -/
def water_depth (tank : CylindricalTank) (surface_area : ℝ) : Set ℝ :=
  { h : ℝ | ∃ (c : ℝ), 
    c = surface_area / tank.length ∧ 
    c = 2 * Real.sqrt (tank.diameter * h - h^2) ∧ 
    0 < h ∧ 
    h < tank.diameter }

/-- Theorem stating the possible depths of water in a specific cylindrical tank -/
theorem water_depth_in_specific_tank : 
  let tank : CylindricalTank := { length := 12, diameter := 8 }
  let surface_area := 48
  water_depth tank surface_area = {4 - 2 * Real.sqrt 3, 4 + 2 * Real.sqrt 3} :=
by
  sorry

end NUMINAMATH_CALUDE_water_depth_in_specific_tank_l1547_154737


namespace NUMINAMATH_CALUDE_increase_in_average_age_increase_in_average_age_is_two_l1547_154798

/-- The increase in average age when replacing two men with two women in a group -/
theorem increase_in_average_age : ℝ :=
  let initial_group_size : ℕ := 8
  let replaced_men_ages : List ℝ := [20, 24]
  let women_average_age : ℝ := 30
  let total_age_increase : ℝ := 2 * women_average_age - replaced_men_ages.sum
  total_age_increase / initial_group_size

/-- Proof that the increase in average age is 2 years -/
theorem increase_in_average_age_is_two :
  increase_in_average_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_average_age_increase_in_average_age_is_two_l1547_154798


namespace NUMINAMATH_CALUDE_prob_odd_after_removal_on_die_l1547_154787

/-- Represents a standard die face with its number of dots -/
inductive DieFace
| one
| two
| three
| four
| five
| six

/-- Calculates the number of ways to choose 2 dots from n dots -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the probability of choosing 2 dots from a face with n dots -/
def prob_choose_from_face (n : ℕ) : ℚ := (choose_two n : ℚ) / (choose_two 21 : ℚ)

/-- Determines if a face will have an odd number of dots after removing 2 dots -/
def odd_after_removal (face : DieFace) : Bool :=
  match face with
  | DieFace.one => false
  | DieFace.two => false
  | DieFace.three => false
  | DieFace.four => true
  | DieFace.five => false
  | DieFace.six => true

/-- Calculates the probability of getting an odd number of dots on a specific face after removal -/
def prob_odd_after_removal (face : DieFace) : ℚ :=
  if odd_after_removal face then
    match face with
    | DieFace.four => prob_choose_from_face 4
    | DieFace.six => prob_choose_from_face 6
    | _ => 0
  else 0

/-- The main theorem to prove -/
theorem prob_odd_after_removal_on_die : 
  (1 / 6 : ℚ) * (prob_odd_after_removal DieFace.one + 
                 prob_odd_after_removal DieFace.two + 
                 prob_odd_after_removal DieFace.three + 
                 prob_odd_after_removal DieFace.four + 
                 prob_odd_after_removal DieFace.five + 
                 prob_odd_after_removal DieFace.six) = 1 / 60 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_after_removal_on_die_l1547_154787


namespace NUMINAMATH_CALUDE_recurring_fraction_equality_l1547_154734

-- Define the recurring decimal 0.812812...
def recurring_812 : ℚ := 812 / 999

-- Define the recurring decimal 2.406406...
def recurring_2406 : ℚ := 2404 / 999

-- Theorem statement
theorem recurring_fraction_equality : 
  recurring_812 / recurring_2406 = 203 / 601 := by
  sorry

end NUMINAMATH_CALUDE_recurring_fraction_equality_l1547_154734


namespace NUMINAMATH_CALUDE_extra_minutes_per_A_is_correct_l1547_154799

/-- The number of extra minutes earned for each A grade -/
def extra_minutes_per_A : ℕ := 2

/-- The normal recess time in minutes -/
def normal_recess : ℕ := 20

/-- The number of A grades -/
def num_A : ℕ := 10

/-- The number of B grades -/
def num_B : ℕ := 12

/-- The number of C grades -/
def num_C : ℕ := 14

/-- The number of D grades -/
def num_D : ℕ := 5

/-- The total recess time in minutes -/
def total_recess : ℕ := 47

theorem extra_minutes_per_A_is_correct :
  extra_minutes_per_A * num_A + num_B - num_D = total_recess - normal_recess :=
by sorry

end NUMINAMATH_CALUDE_extra_minutes_per_A_is_correct_l1547_154799


namespace NUMINAMATH_CALUDE_odd_polynomial_sum_zero_main_theorem_l1547_154717

/-- Definition of an even polynomial -/
def is_even_polynomial (p : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, p y = p (-y)

/-- Definition of an odd polynomial -/
def is_odd_polynomial (p : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, p y = -p (-y)

/-- Theorem: For an odd polynomial, the sum of values at opposite points is zero -/
theorem odd_polynomial_sum_zero (A : ℝ → ℝ) (h : is_odd_polynomial A) :
  A 3 + A (-3) = 0 :=
by sorry

/-- Main theorem to be proved -/
theorem main_theorem :
  ∃ (A : ℝ → ℝ), is_odd_polynomial A ∧ A 3 + A (-3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_odd_polynomial_sum_zero_main_theorem_l1547_154717


namespace NUMINAMATH_CALUDE_find_a_value_l1547_154735

def A : Set ℝ := {0, 1, 2}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 3}

theorem find_a_value :
  ∀ a : ℝ, (A ∩ B a = {1}) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l1547_154735


namespace NUMINAMATH_CALUDE_base_four_of_85_l1547_154760

def base_four_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_four_of_85 :
  base_four_representation 85 = [1, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_four_of_85_l1547_154760


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1547_154710

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : ∀ (x y : ℝ), 
  x + y - 4 = 0 → 
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧ 
  ∀ (x' y' : ℝ), x' + y' - 4 = 0 → 
  Real.sqrt (x' ^ 2 + y' ^ 2) ≥ d := by
  sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l1547_154710


namespace NUMINAMATH_CALUDE_lottery_cheating_suspicion_l1547_154794

/-- The number of balls in the lottery -/
def total_balls : ℕ := 45

/-- The number of winning balls in each draw -/
def winning_balls : ℕ := 6

/-- The probability of no repetition in a single draw -/
def p : ℚ := (total_balls - winning_balls).choose winning_balls / total_balls.choose winning_balls

/-- The suspicion threshold -/
def threshold : ℚ := 1 / 100

theorem lottery_cheating_suspicion :
  p ^ 6 < threshold ∧ p ^ 5 ≥ threshold :=
sorry

end NUMINAMATH_CALUDE_lottery_cheating_suspicion_l1547_154794


namespace NUMINAMATH_CALUDE_min_value_theorem_l1547_154715

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y - 9 = 0) :
  2/y + 1/x ≥ 1 ∧ ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + 2*y' - 9 = 0 ∧ 2/y' + 1/x' = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1547_154715


namespace NUMINAMATH_CALUDE_quadratic_equation_magnitude_l1547_154754

theorem quadratic_equation_magnitude (z : ℂ) : 
  z^2 - 12*z + 157 = 0 → ∃! r : ℝ, (Complex.abs z = r ∧ r = Real.sqrt 157) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_magnitude_l1547_154754


namespace NUMINAMATH_CALUDE_approximation_accuracy_l1547_154781

/-- The actual number of students --/
def actual_number : ℕ := 76500

/-- The approximate number in scientific notation --/
def approximate_number : ℝ := 7.7 * 10^4

/-- Definition of accuracy to thousands place --/
def accurate_to_thousands (x y : ℝ) : Prop :=
  ∃ k : ℤ, x = k * 1000 ∧ |y - x| < 500

/-- Theorem stating the approximation is accurate to the thousands place --/
theorem approximation_accuracy :
  accurate_to_thousands (↑actual_number) approximate_number :=
sorry

end NUMINAMATH_CALUDE_approximation_accuracy_l1547_154781


namespace NUMINAMATH_CALUDE_jamie_calculation_l1547_154745

theorem jamie_calculation (x : ℝ) : (x / 8 + 20 = 28) → (x * 8 - 20 = 492) := by
  sorry

end NUMINAMATH_CALUDE_jamie_calculation_l1547_154745


namespace NUMINAMATH_CALUDE_cos_shift_right_l1547_154716

theorem cos_shift_right (x : ℝ) :
  let f (x : ℝ) := Real.cos (2 * x)
  let shift := π / 12
  let g (x : ℝ) := f (x - shift)
  g x = Real.cos (2 * x - π / 6) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_right_l1547_154716


namespace NUMINAMATH_CALUDE_norma_laundry_problem_l1547_154782

/-- The number of T-shirts Norma left in the washer -/
def t_shirts_left : ℕ := 9

/-- The number of sweaters Norma left in the washer -/
def sweaters_left : ℕ := 2 * t_shirts_left

/-- The total number of clothes Norma left in the washer -/
def total_left : ℕ := t_shirts_left + sweaters_left

/-- The number of sweaters Norma found when she returned -/
def sweaters_found : ℕ := 3

/-- The number of T-shirts Norma found when she returned -/
def t_shirts_found : ℕ := 3 * t_shirts_left

/-- The total number of clothes Norma found when she returned -/
def total_found : ℕ := sweaters_found + t_shirts_found

/-- The number of missing items -/
def missing_items : ℕ := total_left - total_found

theorem norma_laundry_problem : missing_items = 15 := by
  sorry

end NUMINAMATH_CALUDE_norma_laundry_problem_l1547_154782


namespace NUMINAMATH_CALUDE_real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l1547_154750

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 2) (m^2 - 1)

-- Theorem for real number condition
theorem real_number_condition (m : ℝ) : (z m).im = 0 ↔ m = 1 ∨ m = -1 := by sorry

-- Theorem for imaginary number condition
theorem imaginary_number_condition (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ 1 ∧ m ≠ -1 := by sorry

-- Theorem for pure imaginary number condition
theorem pure_imaginary_number_condition (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l1547_154750


namespace NUMINAMATH_CALUDE_bianca_drawing_time_l1547_154769

theorem bianca_drawing_time (school_time home_time total_time : ℕ) : 
  school_time = 22 → total_time = 41 → home_time = total_time - school_time → home_time = 19 :=
by sorry

end NUMINAMATH_CALUDE_bianca_drawing_time_l1547_154769


namespace NUMINAMATH_CALUDE_cake_surface_area_change_l1547_154763

/-- The change in surface area of a cylindrical cake after removing a cube from its top center -/
theorem cake_surface_area_change 
  (h : ℝ) -- height of the cylinder
  (r : ℝ) -- radius of the cylinder
  (s : ℝ) -- side length of the cube
  (h_pos : h > 0)
  (r_pos : r > 0)
  (s_pos : s > 0)
  (h_val : h = 5)
  (r_val : r = 2)
  (s_val : s = 1)
  (h_ge_s : h ≥ s) -- ensure the cube fits in the cylinder
  : (2 * π * r * s + s^2) - s^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_cake_surface_area_change_l1547_154763


namespace NUMINAMATH_CALUDE_pickle_problem_l1547_154764

/-- Pickle problem theorem -/
theorem pickle_problem (jars : ℕ) (cucumbers : ℕ) (initial_vinegar : ℕ) 
  (pickles_per_cucumber : ℕ) (vinegar_per_jar : ℕ) (remaining_vinegar : ℕ) :
  jars = 4 →
  cucumbers = 10 →
  initial_vinegar = 100 →
  pickles_per_cucumber = 6 →
  vinegar_per_jar = 10 →
  remaining_vinegar = 60 →
  (initial_vinegar - remaining_vinegar) / vinegar_per_jar = jars →
  (cucumbers * pickles_per_cucumber) / jars = 15 :=
by sorry

end NUMINAMATH_CALUDE_pickle_problem_l1547_154764


namespace NUMINAMATH_CALUDE_ratio_calculation_l1547_154765

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 ∧ B / C = 1/5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by
sorry

end NUMINAMATH_CALUDE_ratio_calculation_l1547_154765


namespace NUMINAMATH_CALUDE_dandelion_seed_percentage_l1547_154740

def num_sunflowers : ℕ := 6
def num_dandelions : ℕ := 8
def seeds_per_sunflower : ℕ := 9
def seeds_per_dandelion : ℕ := 12

def total_seeds : ℕ := num_sunflowers * seeds_per_sunflower + num_dandelions * seeds_per_dandelion
def dandelion_seeds : ℕ := num_dandelions * seeds_per_dandelion

theorem dandelion_seed_percentage :
  (dandelion_seeds : ℚ) / total_seeds * 100 = 64 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_seed_percentage_l1547_154740


namespace NUMINAMATH_CALUDE_exam_correct_answers_l1547_154743

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  total_questions : Nat
  correct_score : Int
  wrong_score : Int

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  total_score : Int

/-- Calculates the number of correctly answered questions. -/
def correct_answers (result : ExamResult) : Nat :=
  sorry

/-- Theorem stating that for the given exam conditions, 
    the number of correct answers is 42. -/
theorem exam_correct_answers 
  (e : Exam) 
  (r : ExamResult) 
  (h1 : e.total_questions = 80) 
  (h2 : e.correct_score = 4) 
  (h3 : e.wrong_score = -1) 
  (h4 : r.exam = e) 
  (h5 : r.total_score = 130) : 
  correct_answers r = 42 := by
  sorry

end NUMINAMATH_CALUDE_exam_correct_answers_l1547_154743


namespace NUMINAMATH_CALUDE_complex_number_location_l1547_154795

theorem complex_number_location (a : ℝ) (h : 0 < a ∧ a < 1) :
  let z : ℂ := Complex.mk a (a - 1)
  Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1547_154795


namespace NUMINAMATH_CALUDE_min_value_f_range_of_a_inequality_ln_exp_l1547_154789

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := x * Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 3

-- Statement 1
theorem min_value_f (t : ℝ) (h : t > 0) :
  (∀ x ∈ Set.Icc t (t + 2), f x ≥ (-1 / Real.exp 1) ∧ (t < 1 / Real.exp 1 → f x > t * Real.log t)) ∧
  (∀ x ∈ Set.Icc t (t + 2), f x ≥ t * Real.log t ∧ (t ≥ 1 / Real.exp 1 → f x > -1 / Real.exp 1)) :=
sorry

-- Statement 2
theorem range_of_a (a : ℝ) :
  (∀ x > 0, 2 * f x ≥ g a x) ↔ a ≤ 4 :=
sorry

-- Statement 3
theorem inequality_ln_exp (x : ℝ) (h : x > 0) :
  Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end NUMINAMATH_CALUDE_min_value_f_range_of_a_inequality_ln_exp_l1547_154789


namespace NUMINAMATH_CALUDE_sticker_pages_l1547_154753

theorem sticker_pages (stickers_per_page : ℕ) (total_stickers : ℕ) (h1 : stickers_per_page = 10) (h2 : total_stickers = 220) :
  total_stickers / stickers_per_page = 22 := by
  sorry

end NUMINAMATH_CALUDE_sticker_pages_l1547_154753


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1547_154720

/-- 
If a quadratic equation x² - 2x + m = 0 has two equal real roots,
then m = 1.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ 
   (∀ y : ℝ, y^2 - 2*y + m = 0 → y = x)) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1547_154720


namespace NUMINAMATH_CALUDE_halfway_to_end_time_l1547_154779

/-- Two cars traveling in opposite directions with constant speeds --/
structure TwoCars where
  speed : ℝ
  distance : ℝ

/-- The state of the cars at a given time --/
def carState (cars : TwoCars) (t : ℝ) : ℝ × ℝ :=
  (cars.speed * t, cars.distance - cars.speed * t)

/-- The condition that after 1 hour, one car is halfway to the other --/
def halfwayAfterOneHour (cars : TwoCars) : Prop :=
  cars.speed = (cars.distance / 2)

/-- The time when one car is halfway between the other car and its starting point --/
def halfwayToEnd (cars : TwoCars) : ℝ :=
  2

/-- The main theorem --/
theorem halfway_to_end_time {cars : TwoCars} (h : halfwayAfterOneHour cars) :
  let (x, y) := carState cars (halfwayToEnd cars)
  x = (cars.distance - y) / 2 := by
  sorry


end NUMINAMATH_CALUDE_halfway_to_end_time_l1547_154779


namespace NUMINAMATH_CALUDE_zeros_of_g_l1547_154728

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x + b
def g (a b x : ℝ) : ℝ := b * x^2 - a * x

-- State the theorem
theorem zeros_of_g (a b : ℝ) (h : f a b 2 = 0) :
  (g a b 0 = 0) ∧ (g a b (-1/2) = 0) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_g_l1547_154728


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l1547_154727

theorem cubic_root_equation_solution :
  ∀ x : ℝ, (Real.rpow (17 * x - 1) (1/3) + Real.rpow (11 * x + 1) (1/3) = 2 * Real.rpow x (1/3)) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l1547_154727


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1547_154771

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ := 27

/-- A convex polygon with 9 sides -/
structure ConvexNonagon where
  sides : ℕ
  is_convex : Bool
  side_count_eq_9 : sides = 9

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals (n : ConvexNonagon) : num_diagonals_nonagon = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1547_154771


namespace NUMINAMATH_CALUDE_balloon_count_impossible_l1547_154722

theorem balloon_count_impossible : ¬∃ (b g : ℕ), 3 * (b + g) = 100 := by
  sorry

#check balloon_count_impossible

end NUMINAMATH_CALUDE_balloon_count_impossible_l1547_154722


namespace NUMINAMATH_CALUDE_win_sector_area_l1547_154719

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/8) :
  p * π * r^2 = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l1547_154719


namespace NUMINAMATH_CALUDE_coin_combination_theorem_l1547_154751

/-- Represents the number of different coin values obtainable -/
def different_values (num_five_cent : ℕ) (num_ten_cent : ℕ) : ℕ :=
  29 - num_five_cent

/-- Theorem stating that given 15 coins with 22 different obtainable values, there must be 8 10-cent coins -/
theorem coin_combination_theorem :
  ∀ (num_five_cent num_ten_cent : ℕ),
    num_five_cent + num_ten_cent = 15 →
    different_values num_five_cent num_ten_cent = 22 →
    num_ten_cent = 8 := by
  sorry

#check coin_combination_theorem

end NUMINAMATH_CALUDE_coin_combination_theorem_l1547_154751


namespace NUMINAMATH_CALUDE_ethan_reading_pages_l1547_154777

theorem ethan_reading_pages : 
  ∀ (total_pages saturday_morning sunday_pages pages_left saturday_night : ℕ),
  total_pages = 360 →
  saturday_morning = 40 →
  sunday_pages = 2 * (saturday_morning + saturday_night) →
  pages_left = 210 →
  total_pages = saturday_morning + saturday_night + sunday_pages + pages_left →
  saturday_night = 10 := by
sorry

end NUMINAMATH_CALUDE_ethan_reading_pages_l1547_154777


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1547_154780

theorem min_value_of_expression (a : ℝ) (b : ℝ) (h : b > 0) :
  ∃ (min : ℝ), min = 2 * (1 - Real.log 2)^2 ∧
  ∀ (x y : ℝ) (hy : y > 0), ((1/2 * Real.exp x - Real.log (2*y))^2 + (x - y)^2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1547_154780


namespace NUMINAMATH_CALUDE_total_birds_l1547_154758

-- Define the number of geese
def geese : ℕ := 58

-- Define the number of ducks
def ducks : ℕ := 37

-- Theorem stating the total number of birds
theorem total_birds : geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_l1547_154758


namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l1547_154731

-- Define the radius of the sun in kilometers
def sun_radius_km : ℝ := 696000

-- Define the conversion factor from kilometers to meters
def km_to_m : ℝ := 1000

-- Theorem to prove
theorem sun_radius_scientific_notation :
  sun_radius_km * km_to_m = 6.96 * (10 ^ 8) :=
by sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l1547_154731


namespace NUMINAMATH_CALUDE_luka_aubrey_age_difference_l1547_154703

structure Person where
  name : String
  age : ℕ

structure Dog where
  name : String
  age : ℕ

def age_difference (p1 p2 : Person) : ℤ :=
  p1.age - p2.age

theorem luka_aubrey_age_difference 
  (luka aubrey : Person) 
  (max : Dog) 
  (h1 : max.age = luka.age - 4)
  (h2 : aubrey.age = 8)
  (h3 : max.age = 6) : 
  age_difference luka aubrey = 2 := by
sorry

end NUMINAMATH_CALUDE_luka_aubrey_age_difference_l1547_154703


namespace NUMINAMATH_CALUDE_trapezoid_area_l1547_154702

/-- Given an outer equilateral triangle with area 16, an inner equilateral triangle
    with area 1, and three congruent trapezoids between them, the area of one
    trapezoid is 5. -/
theorem trapezoid_area (outer_area inner_area : ℝ) (num_trapezoids : ℕ) :
  outer_area = 16 →
  inner_area = 1 →
  num_trapezoids = 3 →
  (outer_area - inner_area) / num_trapezoids = 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1547_154702


namespace NUMINAMATH_CALUDE_triangle_properties_l1547_154721

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  c = a * (Real.cos B + Real.sqrt 3 * Real.sin B) →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 →
  a = 1 →
  A = π / 6 ∧ a + b + c = Real.sqrt 3 + 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1547_154721


namespace NUMINAMATH_CALUDE_farm_animal_problem_l1547_154724

/-- Prove that the number of cows is 9 given the conditions of the farm animal problem -/
theorem farm_animal_problem :
  ∃ (num_cows : ℕ),
    let num_chickens : ℕ := 8
    let num_ducks : ℕ := 3
    let total_legs : ℕ := 4 * num_cows + 2 * num_chickens + 2 * num_ducks
    let total_heads : ℕ := num_cows + num_chickens + 2 * num_ducks
    total_legs = 18 + 2 * total_heads ∧ num_cows = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_animal_problem_l1547_154724


namespace NUMINAMATH_CALUDE_polynomial_problem_l1547_154733

theorem polynomial_problem (n : ℕ) (p : ℝ → ℝ) : 
  (∀ k : ℕ, k ≤ n → p (2 * k) = 0) →
  (∀ k : ℕ, k < n → p (2 * k + 1) = 2) →
  p (2 * n + 1) = -30 →
  (∃ c : ℝ → ℝ, ∀ x, p x = c x * x * (x - 2)^2 * (x - 4)) →
  (n = 2 ∧ ∀ x, p x = -2/3 * x * (x - 2)^2 * (x - 4)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_problem_l1547_154733


namespace NUMINAMATH_CALUDE_solution_system_equations_l1547_154752

theorem solution_system_equations (x y z : ℝ) : 
  x^2 + y^2 = 6*z ∧ 
  y^2 + z^2 = 6*x ∧ 
  z^2 + x^2 = 6*y → 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 3 ∧ y = 3 ∧ z = 3) := by
sorry

end NUMINAMATH_CALUDE_solution_system_equations_l1547_154752


namespace NUMINAMATH_CALUDE_distance_ratio_l1547_154742

/-- Yan's travel scenario --/
structure TravelScenario where
  w : ℝ  -- Yan's walking speed
  x : ℝ  -- Distance from Yan to his office
  y : ℝ  -- Distance from Yan to the concert hall
  h_positive : w > 0 ∧ x > 0 ∧ y > 0  -- Positive distances and speed

/-- The time taken for both travel options is equal --/
def equal_time (s : TravelScenario) : Prop :=
  s.y / s.w = (s.x / s.w + (s.x + s.y) / (5 * s.w))

/-- The theorem stating the ratio of distances --/
theorem distance_ratio (s : TravelScenario) 
  (h_equal_time : equal_time s) : 
  s.x / s.y = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_distance_ratio_l1547_154742


namespace NUMINAMATH_CALUDE_equation_solutions_l1547_154700

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 - Real.sqrt 11 ∧ x₂ = 2 + Real.sqrt 11 ∧
    x₁^2 - 4*x₁ - 7 = 0 ∧ x₂^2 - 4*x₂ - 7 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧
    (x₁ - 3)^2 + 2*(x₁ - 3) = 0 ∧ (x₂ - 3)^2 + 2*(x₂ - 3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1547_154700


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1547_154725

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

/-- The derivative of the cubic curve -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

/-- The x-coordinate of the point of tangency -/
def x₀ : ℝ := 2

/-- The y-coordinate of the point of tangency -/
def y₀ : ℝ := -3

/-- The slope of the tangent line -/
def m : ℝ := f' x₀

theorem tangent_line_equation :
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -3*x + 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1547_154725
