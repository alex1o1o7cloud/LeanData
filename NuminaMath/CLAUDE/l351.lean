import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l351_35108

/-- Theorem: Solution of quadratic inequality ax^2 + bx + c < 0 -/
theorem quadratic_inequality_solution 
  (a b c : ℝ) (h : a ≠ 0) :
  let x1 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x2 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a > 0 → {x : ℝ | x1 < x ∧ x < x2} = {x : ℝ | a*x^2 + b*x + c < 0}) ∧
  (a < 0 → {x : ℝ | x < x1 ∨ x2 < x} = {x : ℝ | a*x^2 + b*x + c < 0}) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_l351_35108


namespace NUMINAMATH_CALUDE_series_convergence_l351_35159

/-- The infinite series ∑(k=1 to ∞) [k(k+1)/(2*3^k)] converges to 3/2 -/
theorem series_convergence : 
  ∑' k, (k * (k + 1) : ℝ) / (2 * 3^k) = 3/2 := by sorry

end NUMINAMATH_CALUDE_series_convergence_l351_35159


namespace NUMINAMATH_CALUDE_square_of_3y_plus_4_when_y_is_neg_2_l351_35135

theorem square_of_3y_plus_4_when_y_is_neg_2 :
  let y : ℤ := -2
  (3 * y + 4)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_square_of_3y_plus_4_when_y_is_neg_2_l351_35135


namespace NUMINAMATH_CALUDE_perfect_square_function_characterization_l351_35157

/-- A function from positive naturals to positive naturals -/
def PositiveNatFunction := ℕ+ → ℕ+

/-- The property that (m + g(n))(g(m) + n) is a perfect square for all m, n -/
def IsPerfectSquareProperty (g : PositiveNatFunction) : Prop :=
  ∀ m n : ℕ+, ∃ k : ℕ+, (m + g n) * (g m + n) = k * k

/-- The main theorem stating that if g satisfies the perfect square property,
    then it must be of the form g(n) = n + c for some constant c -/
theorem perfect_square_function_characterization (g : PositiveNatFunction) 
    (h : IsPerfectSquareProperty g) :
    ∃ c : ℕ, ∀ n : ℕ+, g n = n + c := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_function_characterization_l351_35157


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l351_35197

/-- The number of unique arrangements of n distinct beads on a bracelet, 
    considering rotational and reflectional symmetry -/
def bracelet_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of unique arrangements of 8 distinct beads 
    on a bracelet, considering rotational and reflectional symmetry, is 2520 -/
theorem eight_bead_bracelet_arrangements :
  bracelet_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l351_35197


namespace NUMINAMATH_CALUDE_average_wage_calculation_l351_35190

/-- Calculates the average wage per day paid by a contractor given the number of workers and their wages. -/
theorem average_wage_calculation
  (male_workers female_workers child_workers : ℕ)
  (male_wage female_wage child_wage : ℚ)
  (h_male : male_workers = 20)
  (h_female : female_workers = 15)
  (h_child : child_workers = 5)
  (h_male_wage : male_wage = 35)
  (h_female_wage : female_wage = 20)
  (h_child_wage : child_wage = 8) :
  (male_workers * male_wage + female_workers * female_wage + child_workers * child_wage) /
  (male_workers + female_workers + child_workers : ℚ) = 26 := by
  sorry

end NUMINAMATH_CALUDE_average_wage_calculation_l351_35190


namespace NUMINAMATH_CALUDE_work_completion_time_l351_35195

/-- The time it takes for A and B to complete a work together, given their individual work rates -/
def time_to_complete_together (rate_b rate_a : ℚ) : ℚ :=
  1 / (rate_a + rate_b)

/-- The proposition that A and B complete the work in 4 days under the given conditions -/
theorem work_completion_time 
  (rate_b : ℚ) -- B's work rate
  (rate_a : ℚ) -- A's work rate
  (h1 : rate_b = 1 / 12) -- B completes the work in 12 days
  (h2 : rate_a = 2 * rate_b) -- A works twice as fast as B
  : time_to_complete_together rate_b rate_a = 4 := by
  sorry

#eval time_to_complete_together (1/12) (1/6)

end NUMINAMATH_CALUDE_work_completion_time_l351_35195


namespace NUMINAMATH_CALUDE_find_a_l351_35131

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {1, 2, a^2 + 2*a - 3}

-- Define set A
def A (a : ℝ) : Set ℝ := {|a - 2|, 2}

-- Define the complement of A with respect to U
def complementA (a : ℝ) : Set ℝ := (U a) \ (A a)

-- Theorem statement
theorem find_a : ∃ a : ℝ, (U a = {1, 2, a^2 + 2*a - 3}) ∧ 
                          (A a = {|a - 2|, 2}) ∧ 
                          (complementA a = {0}) ∧ 
                          (a = 1) := by
  sorry

end NUMINAMATH_CALUDE_find_a_l351_35131


namespace NUMINAMATH_CALUDE_sue_age_l351_35168

theorem sue_age (total_age kate_age maggie_age sue_age : ℕ) : 
  total_age = 48 → kate_age = 19 → maggie_age = 17 → 
  total_age = kate_age + maggie_age + sue_age →
  sue_age = 12 := by
sorry

end NUMINAMATH_CALUDE_sue_age_l351_35168


namespace NUMINAMATH_CALUDE_polynomial_roots_l351_35112

theorem polynomial_roots (AT TB : ℝ) (h1 : AT + TB = 15) (h2 : AT * TB = 36) :
  ∃ (p : ℝ → ℝ), p = (fun x ↦ x^2 - 20*x + 75) ∧ 
  p (AT + 5) = 0 ∧ p TB = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l351_35112


namespace NUMINAMATH_CALUDE_equation_solution_l351_35139

theorem equation_solution : 
  ∀ x : ℝ, |2001*x - 2001| = 2001 ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l351_35139


namespace NUMINAMATH_CALUDE_project_completion_time_l351_35196

theorem project_completion_time 
  (workers_initial : ℕ) 
  (days_initial : ℕ) 
  (workers_new : ℕ) 
  (h1 : workers_initial = 60) 
  (h2 : days_initial = 3) 
  (h3 : workers_new = 30) :
  workers_initial * days_initial = workers_new * (2 * days_initial) :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l351_35196


namespace NUMINAMATH_CALUDE_letter_puzzle_solutions_l351_35115

/-- A function that checks if a number is a single digit (1 to 9) -/
def isSingleDigit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- A function that checks if two numbers are distinct -/
def areDistinct (a b : ℕ) : Prop := a ≠ b

/-- A function that checks if a number is a two-digit number -/
def isTwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that constructs a two-digit number from two single digits -/
def twoDigitConstruct (b a : ℕ) : ℕ := 10 * b + a

/-- The main theorem stating the only solutions to A^B = BA -/
theorem letter_puzzle_solutions :
  ∀ A B : ℕ,
  isSingleDigit A →
  isSingleDigit B →
  areDistinct A B →
  isTwoDigitNumber (twoDigitConstruct B A) →
  twoDigitConstruct B A ≠ B * A →
  A^B = twoDigitConstruct B A →
  ((A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3)) :=
by sorry

end NUMINAMATH_CALUDE_letter_puzzle_solutions_l351_35115


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l351_35109

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => x^2 - 36*x + 323
  let solution_set := {x : ℝ | f x ≤ 5}
  let lower_bound := 18 - Real.sqrt 6
  let upper_bound := 18 + Real.sqrt 6
  solution_set = Set.Icc lower_bound upper_bound := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l351_35109


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_attained_min_value_is_1215_l351_35119

theorem min_value_sum (x y z : ℕ+) (h : x^3 + y^3 + z^3 - 3*x*y*z = 607) :
  ∀ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 607 → x + 2*y + 3*z ≤ a + 2*b + 3*c :=
by sorry

theorem min_value_attained (x y z : ℕ+) (h : x^3 + y^3 + z^3 - 3*x*y*z = 607) :
  ∃ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 607 ∧ a + 2*b + 3*c = 1215 :=
by sorry

theorem min_value_is_1215 :
  ∃ (x y z : ℕ+), x^3 + y^3 + z^3 - 3*x*y*z = 607 ∧
  (∀ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 607 → x + 2*y + 3*z ≤ a + 2*b + 3*c) ∧
  x + 2*y + 3*z = 1215 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_attained_min_value_is_1215_l351_35119


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l351_35174

-- Problem 1
theorem problem_one : |2 - Real.sqrt 3| - 2^0 - Real.sqrt 12 = 1 - 3 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_two : (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + (2 * Real.sqrt 3 + 1)^2 = 14 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l351_35174


namespace NUMINAMATH_CALUDE_johnson_family_seating_l351_35136

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def total_arrangements (n : ℕ) : ℕ := factorial n

def alternating_arrangements (n : ℕ) : ℕ := 2 * factorial n * factorial n

theorem johnson_family_seating (boys girls : ℕ) (h : boys = 4 ∧ girls = 4) :
  total_arrangements (boys + girls) - alternating_arrangements boys =
  39168 := by sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l351_35136


namespace NUMINAMATH_CALUDE_unsold_books_percentage_l351_35158

def initial_stock : ℕ := 700
def monday_sales : ℕ := 50
def tuesday_sales : ℕ := 82
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

theorem unsold_books_percentage :
  let total_sales := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales
  let unsold_books := initial_stock - total_sales
  (unsold_books : ℚ) / initial_stock * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_percentage_l351_35158


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_when_f_geq_3_l351_35161

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

-- Theorem 1
theorem min_value_when_a_is_one :
  ∀ x ∈ Set.Ioo 0 (Real.exp 1), f 1 x ≥ f 1 1 ∧ f 1 1 = 1 := by sorry

-- Theorem 2
theorem range_of_a_when_f_geq_3 :
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f a x ≥ 3) → a ≥ Real.exp 2 := by sorry

end

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_when_f_geq_3_l351_35161


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l351_35149

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {-3} → a = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l351_35149


namespace NUMINAMATH_CALUDE_b_is_criminal_l351_35176

-- Define the set of suspects
inductive Suspect : Type
  | A | B | C | D

-- Define a function to represent the statements of each suspect
def statement (s : Suspect) (criminal : Suspect) : Prop :=
  match s with
  | Suspect.A => criminal ≠ Suspect.A
  | Suspect.B => criminal = Suspect.C
  | Suspect.C => criminal = Suspect.A ∨ criminal = Suspect.B
  | Suspect.D => criminal = Suspect.C

-- Define a function to check if a statement is true given the actual criminal
def is_true_statement (s : Suspect) (criminal : Suspect) : Prop :=
  statement s criminal

-- Theorem stating that B is the criminal
theorem b_is_criminal :
  ∃ (criminal : Suspect),
    criminal = Suspect.B ∧
    (∃ (t1 t2 l1 l2 : Suspect),
      t1 ≠ t2 ∧ l1 ≠ l2 ∧
      t1 ≠ l1 ∧ t1 ≠ l2 ∧ t2 ≠ l1 ∧ t2 ≠ l2 ∧
      is_true_statement t1 criminal ∧
      is_true_statement t2 criminal ∧
      ¬is_true_statement l1 criminal ∧
      ¬is_true_statement l2 criminal) :=
by
  sorry


end NUMINAMATH_CALUDE_b_is_criminal_l351_35176


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l351_35187

theorem absolute_value_inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x + 1| - |x - 2| ≥ a^2 - 4*a) ↔ a ∈ Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l351_35187


namespace NUMINAMATH_CALUDE_right_triangle_angle_bisector_square_area_l351_35153

/-- Given a right triangle where the bisector of the right angle cuts the hypotenuse
    into segments of lengths a and b, the area of the square whose side is this bisector
    is equal to 2a²b² / (a² + b²). -/
theorem right_triangle_angle_bisector_square_area
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let bisector_length := Real.sqrt (2 * a^2 * b^2 / (a^2 + b^2))
  (bisector_length)^2 = 2 * a^2 * b^2 / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_bisector_square_area_l351_35153


namespace NUMINAMATH_CALUDE_committee_selection_l351_35141

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l351_35141


namespace NUMINAMATH_CALUDE_initial_toothbrushes_l351_35163

/-- The number of toothbrushes given away in January -/
def january : ℕ := 53

/-- The number of toothbrushes given away in February -/
def february : ℕ := 67

/-- The number of toothbrushes given away in March -/
def march : ℕ := 46

/-- The difference between the busiest and slowest month -/
def difference : ℕ := 36

/-- The number of toothbrushes given away in April (equal to May) -/
def april_may : ℕ := february - difference

/-- The total number of toothbrushes Dr. Banks had initially -/
def total_toothbrushes : ℕ := january + february + march + 2 * april_may

theorem initial_toothbrushes : total_toothbrushes = 228 := by
  sorry

end NUMINAMATH_CALUDE_initial_toothbrushes_l351_35163


namespace NUMINAMATH_CALUDE_remove_32_toothpicks_eliminates_triangles_l351_35120

/-- A triangular figure constructed with toothpicks -/
structure TriangularFigure where
  toothpicks : ℕ
  triangles : ℕ
  horizontal_toothpicks : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (f : TriangularFigure) : ℕ :=
  f.horizontal_toothpicks

/-- Theorem stating that removing 32 toothpicks is sufficient to eliminate all triangles 
    in a specific triangular figure -/
theorem remove_32_toothpicks_eliminates_triangles (f : TriangularFigure) 
  (h1 : f.toothpicks = 42)
  (h2 : f.triangles > 35)
  (h3 : f.horizontal_toothpicks = 32) :
  min_toothpicks_to_remove f = 32 := by
  sorry

end NUMINAMATH_CALUDE_remove_32_toothpicks_eliminates_triangles_l351_35120


namespace NUMINAMATH_CALUDE_steve_pie_difference_l351_35113

/-- Represents a baker's weekly pie production --/
structure BakerProduction where
  pies_per_day : ℕ
  apple_pie_days : ℕ
  cherry_pie_days : ℕ

/-- Calculates the difference between apple pies and cherry pies baked in a week --/
def pie_difference (bp : BakerProduction) : ℕ :=
  bp.pies_per_day * bp.apple_pie_days - bp.pies_per_day * bp.cherry_pie_days

/-- Theorem stating the difference in pie production for Steve's bakery --/
theorem steve_pie_difference :
  ∀ (bp : BakerProduction),
    bp.pies_per_day = 12 →
    bp.apple_pie_days = 3 →
    bp.cherry_pie_days = 2 →
    pie_difference bp = 12 := by
  sorry

end NUMINAMATH_CALUDE_steve_pie_difference_l351_35113


namespace NUMINAMATH_CALUDE_fraction_addition_l351_35140

theorem fraction_addition (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  3 / x + 2 / y = (3 * y + 2 * x) / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l351_35140


namespace NUMINAMATH_CALUDE_path_count_l351_35194

/-- The number of paths on a grid from A to B satisfying specific conditions -/
def number_of_paths : ℕ :=
  Nat.choose 8 4

/-- Theorem stating that the number of paths is 70 -/
theorem path_count : number_of_paths = 70 := by
  sorry

end NUMINAMATH_CALUDE_path_count_l351_35194


namespace NUMINAMATH_CALUDE_fractional_equation_transformation_l351_35117

theorem fractional_equation_transformation (x : ℝ) :
  (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3) ↔ (x - 2 = 3 * (2 * x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_transformation_l351_35117


namespace NUMINAMATH_CALUDE_factor_implies_k_value_l351_35123

theorem factor_implies_k_value (k : ℚ) :
  (∀ x, (3 * x + 4) ∣ (9 * x^3 + k * x^2 + 16 * x + 64)) →
  k = -12 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_k_value_l351_35123


namespace NUMINAMATH_CALUDE_second_applicant_revenue_l351_35172

/-- Represents the financial details of a job applicant -/
structure Applicant where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for the company from an applicant -/
def netGain (a : Applicant) : ℕ :=
  a.revenue - a.salary - (a.trainingMonths * a.trainingCostPerMonth) - (a.salary * a.hiringBonusPercent / 100)

/-- The theorem to prove -/
theorem second_applicant_revenue
  (first : Applicant)
  (second : Applicant)
  (h1 : first.salary = 42000)
  (h2 : first.revenue = 93000)
  (h3 : first.trainingMonths = 3)
  (h4 : first.trainingCostPerMonth = 1200)
  (h5 : first.hiringBonusPercent = 0)
  (h6 : second.salary = 45000)
  (h7 : second.trainingMonths = 0)
  (h8 : second.trainingCostPerMonth = 0)
  (h9 : second.hiringBonusPercent = 1)
  (h10 : netGain second = netGain first + 850) :
  second.revenue = 93700 := by
  sorry

end NUMINAMATH_CALUDE_second_applicant_revenue_l351_35172


namespace NUMINAMATH_CALUDE_chess_probability_l351_35162

theorem chess_probability (prob_A_win prob_draw : ℝ) 
  (h1 : prob_A_win = 0.3)
  (h2 : prob_draw = 0.5) :
  prob_A_win + prob_draw = 0.8 := by
sorry

end NUMINAMATH_CALUDE_chess_probability_l351_35162


namespace NUMINAMATH_CALUDE_spinner_direction_final_direction_is_west_l351_35169

-- Define the possible directions
inductive Direction
  | North
  | South
  | East
  | West

-- Define a function to rotate a direction
def rotate (d : Direction) (revolutions : ℚ) : Direction :=
  sorry

-- State the theorem
theorem spinner_direction (initial : Direction) 
  (clockwise : ℚ) (counterclockwise : ℚ) : Direction :=
  by
  -- Assume the initial direction is south
  have h1 : initial = Direction.South := by sorry
  -- Assume clockwise rotation is 3½ revolutions
  have h2 : clockwise = 7/2 := by sorry
  -- Assume counterclockwise rotation is 1¾ revolutions
  have h3 : counterclockwise = 7/4 := by sorry
  -- Prove that the final direction is west
  sorry

-- The main theorem
theorem final_direction_is_west :
  spinner_direction Direction.South (7/2) (7/4) = Direction.West :=
  by sorry

end NUMINAMATH_CALUDE_spinner_direction_final_direction_is_west_l351_35169


namespace NUMINAMATH_CALUDE_yeast_growth_30_minutes_l351_35186

/-- The number of yeast cells after a given number of 5-minute intervals -/
def yeast_population (initial_population : ℕ) (intervals : ℕ) : ℕ :=
  initial_population * 2^intervals

/-- Theorem: After 30 minutes (6 intervals), the yeast population will be 3200 -/
theorem yeast_growth_30_minutes :
  yeast_population 50 6 = 3200 := by
  sorry

end NUMINAMATH_CALUDE_yeast_growth_30_minutes_l351_35186


namespace NUMINAMATH_CALUDE_sequence_with_special_sums_l351_35105

theorem sequence_with_special_sums : ∃ (seq : Fin 20 → ℝ), 
  (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧ 
  (Finset.sum Finset.univ seq < 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_special_sums_l351_35105


namespace NUMINAMATH_CALUDE_cafeteria_pies_l351_35101

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) :
  initial_apples = 250 →
  handed_out = 33 →
  apples_per_pie = 7 →
  (initial_apples - handed_out) / apples_per_pie = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l351_35101


namespace NUMINAMATH_CALUDE_block_height_is_75_l351_35127

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the properties of the cubes cut from the block -/
structure CubeProperties where
  sideLength : ℝ
  count : ℕ

/-- Checks if the given dimensions and cube properties satisfy the problem conditions -/
def satisfiesConditions (block : BlockDimensions) (cube : CubeProperties) : Prop :=
  block.length = 15 ∧
  block.width = 30 ∧
  cube.count = 10 ∧
  (cube.sideLength ∣ block.length) ∧
  (cube.sideLength ∣ block.width) ∧
  (cube.sideLength ∣ block.height) ∧
  block.length * block.width * block.height = cube.sideLength ^ 3 * cube.count

theorem block_height_is_75 (block : BlockDimensions) (cube : CubeProperties) :
  satisfiesConditions block cube → block.height = 75 := by
  sorry

end NUMINAMATH_CALUDE_block_height_is_75_l351_35127


namespace NUMINAMATH_CALUDE_wrong_observation_value_l351_35122

theorem wrong_observation_value (n : ℕ) (initial_mean corrected_mean true_value : ℝ) : 
  n = 50 →
  initial_mean = 36 →
  corrected_mean = 36.54 →
  true_value = 48 →
  (n : ℝ) * corrected_mean - (n : ℝ) * initial_mean = true_value - (n : ℝ) * initial_mean + (n : ℝ) * corrected_mean - (n : ℝ) * initial_mean →
  true_value - ((n : ℝ) * corrected_mean - (n : ℝ) * initial_mean) = 21 := by
  sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l351_35122


namespace NUMINAMATH_CALUDE_refuel_cost_is_950_l351_35182

/-- Calculates the total cost to refuel a fleet of planes --/
def total_refuel_cost (small_plane_count : ℕ) (large_plane_count : ℕ) (special_plane_count : ℕ)
  (small_tank_size : ℝ) (large_tank_size_factor : ℝ) (special_tank_size : ℝ)
  (regular_fuel_cost : ℝ) (special_fuel_cost : ℝ)
  (regular_service_fee : ℝ) (special_service_fee : ℝ) : ℝ :=
  let large_tank_size := small_tank_size * (1 + large_tank_size_factor)
  let regular_fuel_volume := small_plane_count * small_tank_size + large_plane_count * large_tank_size
  let regular_fuel_cost := regular_fuel_volume * regular_fuel_cost
  let special_fuel_cost := special_plane_count * special_tank_size * special_fuel_cost
  let regular_service_cost := (small_plane_count + large_plane_count) * regular_service_fee
  let special_service_cost := special_plane_count * special_service_fee
  regular_fuel_cost + special_fuel_cost + regular_service_cost + special_service_cost

/-- The total cost to refuel all five planes is $950 --/
theorem refuel_cost_is_950 :
  total_refuel_cost 2 2 1 60 0.5 200 0.5 1 100 200 = 950 := by
  sorry

end NUMINAMATH_CALUDE_refuel_cost_is_950_l351_35182


namespace NUMINAMATH_CALUDE_new_people_calculation_l351_35167

/-- The number of new people who moved into the town -/
def new_people : ℕ := 580

/-- The original population of the town -/
def original_population : ℕ := 780

/-- The number of people who moved out -/
def people_moved_out : ℕ := 400

/-- The population after 4 years -/
def final_population : ℕ := 60

/-- The number of years that passed -/
def years_passed : ℕ := 4

theorem new_people_calculation :
  (((original_population - people_moved_out + new_people : ℚ) / 2^years_passed) : ℚ) = final_population := by
  sorry

end NUMINAMATH_CALUDE_new_people_calculation_l351_35167


namespace NUMINAMATH_CALUDE_triangle_area_from_square_areas_l351_35152

theorem triangle_area_from_square_areas (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2) * a * b = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_square_areas_l351_35152


namespace NUMINAMATH_CALUDE_expression_evaluation_l351_35114

theorem expression_evaluation : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l351_35114


namespace NUMINAMATH_CALUDE_equation_solution_l351_35166

theorem equation_solution : ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l351_35166


namespace NUMINAMATH_CALUDE_time_after_317h_58m_30s_l351_35147

def hours_to_12hour_clock (h : ℕ) : ℕ :=
  h % 12

def add_time (start_hour start_minute start_second : ℕ) 
             (add_hours add_minutes add_seconds : ℕ) : ℕ × ℕ × ℕ :=
  let total_seconds := start_second + add_seconds
  let total_minutes := start_minute + add_minutes + total_seconds / 60
  let total_hours := start_hour + add_hours + total_minutes / 60
  (hours_to_12hour_clock total_hours, total_minutes % 60, total_seconds % 60)

theorem time_after_317h_58m_30s : 
  let (A, B, C) := add_time 3 0 0 317 58 30
  A + B + C = 96 := by sorry

end NUMINAMATH_CALUDE_time_after_317h_58m_30s_l351_35147


namespace NUMINAMATH_CALUDE_sum_of_dot_products_l351_35199

/-- Given three points A, B, C on a plane, prove that the sum of their vector dot products is -25 -/
theorem sum_of_dot_products (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let CA := (A.1 - C.1, A.2 - C.2)
  (AB.1^2 + AB.2^2 = 3^2) →
  (BC.1^2 + BC.2^2 = 4^2) →
  (CA.1^2 + CA.2^2 = 5^2) →
  (AB.1 * BC.1 + AB.2 * BC.2) + (BC.1 * CA.1 + BC.2 * CA.2) + (CA.1 * AB.1 + CA.2 * AB.2) = -25 :=
by
  sorry


end NUMINAMATH_CALUDE_sum_of_dot_products_l351_35199


namespace NUMINAMATH_CALUDE_janet_earnings_theorem_l351_35132

/-- Calculates Janet's earnings per hour based on the number of posts checked and payment rates. -/
def janet_earnings_per_hour (text_posts image_posts video_posts : ℕ) 
  (text_rate image_rate video_rate : ℚ) : ℚ :=
  text_posts * text_rate + image_posts * image_rate + video_posts * video_rate

/-- Proves that Janet's earnings per hour equal $69.50 given the specified conditions. -/
theorem janet_earnings_theorem : 
  janet_earnings_per_hour 150 80 20 0.25 0.30 0.40 = 69.50 := by
  sorry

end NUMINAMATH_CALUDE_janet_earnings_theorem_l351_35132


namespace NUMINAMATH_CALUDE_sum_squares_and_inverses_bound_l351_35173

theorem sum_squares_and_inverses_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^2 + 1/a^2 + b^2 + 1/b^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_and_inverses_bound_l351_35173


namespace NUMINAMATH_CALUDE_ram_weight_increase_l351_35164

theorem ram_weight_increase (ram_initial : ℝ) (shyam_initial : ℝ) 
  (h_ratio : ram_initial / shyam_initial = 4 / 5)
  (h_total_new : ram_initial * (1 + x / 100) + shyam_initial * 1.19 = 82.8)
  (h_total_increase : ram_initial * (1 + x / 100) + shyam_initial * 1.19 = (ram_initial + shyam_initial) * 1.15)
  : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ram_weight_increase_l351_35164


namespace NUMINAMATH_CALUDE_equation_solutions_l351_35183

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 25 = 0 ↔ x = 5 ∨ x = -5) ∧
  (∀ x : ℝ, (2*x - 1)^3 = -8 ↔ x = -1/2) ∧
  (∀ x : ℝ, 4*(x + 1)^2 = 8 ↔ x = -1 - Real.sqrt 2 ∨ x = -1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l351_35183


namespace NUMINAMATH_CALUDE_rice_division_l351_35170

theorem rice_division (total_pounds : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) : 
  total_pounds = 25 / 2 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_pounds * ounces_per_pound) / num_containers = 50 := by
  sorry

end NUMINAMATH_CALUDE_rice_division_l351_35170


namespace NUMINAMATH_CALUDE_sqrt_four_equals_two_l351_35100

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_two_l351_35100


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l351_35179

theorem systematic_sampling_probability 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_population = 120) 
  (h2 : sample_size = 20) :
  (sample_size : ℚ) / total_population = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l351_35179


namespace NUMINAMATH_CALUDE_range_of_increasing_function_l351_35103

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem range_of_increasing_function (f : ℝ → ℝ) (h : increasing_function f) :
  {m : ℝ | f (2 - m) < f (m^2)} = {m : ℝ | m < -2 ∨ m > 1} := by
  sorry

end NUMINAMATH_CALUDE_range_of_increasing_function_l351_35103


namespace NUMINAMATH_CALUDE_average_permutation_sum_l351_35134

def permutation_sum (b : Fin 8 → Fin 8) : ℕ :=
  |b 0 - b 1| + |b 2 - b 3| + |b 4 - b 5| + |b 6 - b 7|

def all_permutations : Finset (Fin 8 → Fin 8) :=
  Finset.univ.filter (λ b ↦ Function.Bijective b)

theorem average_permutation_sum :
  (Finset.sum all_permutations permutation_sum) / all_permutations.card = 672 := by
  sorry

end NUMINAMATH_CALUDE_average_permutation_sum_l351_35134


namespace NUMINAMATH_CALUDE_vector_collinear_same_direction_l351_35138

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- Two vectors have the same direction if one is a positive scalar multiple of the other -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ b = (k * a.1, k * a.2)

/-- Main theorem: If vectors a = (-1, x) and b = (-x, 2) are collinear and have the same direction, then x = √2 -/
theorem vector_collinear_same_direction (x : ℝ) :
  let a : ℝ × ℝ := (-1, x)
  let b : ℝ × ℝ := (-x, 2)
  collinear a b → same_direction a b → x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinear_same_direction_l351_35138


namespace NUMINAMATH_CALUDE_equidecomposable_transitivity_l351_35106

-- Define the concept of a polygon
def Polygon : Type := sorry

-- Define the concept of equidecomposability between two polygons
def equidecomposable (P Q : Polygon) : Prop := sorry

-- Theorem statement
theorem equidecomposable_transitivity (P Q R : Polygon) :
  equidecomposable P R → equidecomposable Q R → equidecomposable P Q := by
  sorry

end NUMINAMATH_CALUDE_equidecomposable_transitivity_l351_35106


namespace NUMINAMATH_CALUDE_total_remaining_students_l351_35118

def calculate_remaining_students (initial_a initial_b initial_c new_a new_b new_c transfer_rate_a transfer_rate_b transfer_rate_c : ℕ) : ℕ :=
  let total_a := initial_a + new_a
  let total_b := initial_b + new_b
  let total_c := initial_c + new_c
  let remaining_a := total_a - (total_a * transfer_rate_a / 100)
  let remaining_b := total_b - (total_b * transfer_rate_b / 100)
  let remaining_c := total_c - (total_c * transfer_rate_c / 100)
  remaining_a + remaining_b + remaining_c

theorem total_remaining_students :
  calculate_remaining_students 160 145 130 20 25 15 30 25 20 = 369 :=
by sorry

end NUMINAMATH_CALUDE_total_remaining_students_l351_35118


namespace NUMINAMATH_CALUDE_dance_attendance_problem_l351_35193

/-- Represents the number of different dance pairs at a school dance. -/
def total_pairs : ℕ := 430

/-- Represents the number of boys the first girl danced with. -/
def first_girl_partners : ℕ := 12

/-- Calculates the number of boys a girl danced with based on her position. -/
def partners_for_girl (girl_position : ℕ) : ℕ :=
  first_girl_partners + girl_position - 1

/-- Calculates the total number of dance pairs for a given number of girls. -/
def sum_of_pairs (num_girls : ℕ) : ℕ :=
  (num_girls * (2 * first_girl_partners + num_girls - 1)) / 2

/-- Represents the problem of finding the number of girls and boys at the dance. -/
theorem dance_attendance_problem :
  ∃ (num_girls num_boys : ℕ),
    num_girls > 0 ∧
    num_boys = partners_for_girl num_girls ∧
    sum_of_pairs num_girls = total_pairs ∧
    num_girls = 20 ∧
    num_boys = 31 := by
  sorry

end NUMINAMATH_CALUDE_dance_attendance_problem_l351_35193


namespace NUMINAMATH_CALUDE_composition_value_l351_35107

/-- Given two functions f and g, and a composition condition, prove that d equals 18 -/
theorem composition_value (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = 5*x + c)
  (hg : ∀ x, g x = c*x + 3)
  (hcomp : ∀ x, f (g x) = 15*x + d) :
  d = 18 := by
  sorry

end NUMINAMATH_CALUDE_composition_value_l351_35107


namespace NUMINAMATH_CALUDE_max_d_is_one_l351_35148

def a (n : ℕ) : ℕ := 105 + n^2 + 3*n

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_is_one :
  ∀ n : ℕ, n ≥ 1 → d n ≤ 1 ∧ ∃ m : ℕ, m ≥ 1 ∧ d m = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_d_is_one_l351_35148


namespace NUMINAMATH_CALUDE_cistern_depth_is_correct_l351_35142

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  total_wet_area : ℝ

/-- Calculates the total wet surface area of a cistern --/
def wet_surface_area (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem stating that for a cistern with given dimensions and wet surface area, the depth is 1.25 m --/
theorem cistern_depth_is_correct (c : Cistern) 
    (h1 : c.length = 6)
    (h2 : c.width = 4)
    (h3 : c.total_wet_area = 49)
    (h4 : wet_surface_area c = c.total_wet_area) :
    c.depth = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_cistern_depth_is_correct_l351_35142


namespace NUMINAMATH_CALUDE_root_between_roots_l351_35111

theorem root_between_roots (a b c r s : ℝ) 
  (ha : a ≠ 0) 
  (hc : c ≠ 0) 
  (hr : a * r^2 + b * r + c = 0) 
  (hs : -a * s^2 + b * s + c = 0) : 
  ∃ t, (t > min r s ∧ t < max r s) ∧ (a / 2) * t^2 + b * t + c = 0 :=
sorry

end NUMINAMATH_CALUDE_root_between_roots_l351_35111


namespace NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l351_35150

/-- The floor number on which Vasya lives -/
def vasya_floor (petya_steps : ℕ) (vasya_steps : ℕ) : ℕ :=
  1 + vasya_steps / (petya_steps / 2)

/-- Theorem stating that Vasya lives on the 5th floor -/
theorem vasya_lives_on_fifth_floor :
  vasya_floor 36 72 = 5 := by
  sorry

#eval vasya_floor 36 72

end NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l351_35150


namespace NUMINAMATH_CALUDE_cassidy_grades_below_B_l351_35144

/-- The number of grades below B that Cassidy received -/
def grades_below_B : ℕ := sorry

/-- The base grounding period in days -/
def base_grounding : ℕ := 14

/-- The additional grounding days for each grade below B -/
def extra_days_per_grade : ℕ := 3

/-- The total grounding period in days -/
def total_grounding : ℕ := 26

theorem cassidy_grades_below_B :
  grades_below_B * extra_days_per_grade + base_grounding = total_grounding ∧
  grades_below_B = 4 := by sorry

end NUMINAMATH_CALUDE_cassidy_grades_below_B_l351_35144


namespace NUMINAMATH_CALUDE_g_value_l351_35129

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem g_value (f g : ℝ → ℝ) (h1 : is_odd f) (h2 : is_even g)
  (h3 : f (-1) + g 1 = 2) (h4 : f 1 + g (-1) = 4) : g 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_value_l351_35129


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l351_35184

theorem arithmetic_calculations : 
  (8 / (-2) - (-4) * (-3) = 8) ∧ 
  ((-2)^3 / 4 * (5 - (-3)^2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l351_35184


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l351_35198

theorem fruit_basket_problem :
  let oranges : ℕ := 15
  let peaches : ℕ := 9
  let pears : ℕ := 18
  let bananas : ℕ := 12
  let apples : ℕ := 24
  Nat.gcd oranges (Nat.gcd peaches (Nat.gcd pears (Nat.gcd bananas apples))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l351_35198


namespace NUMINAMATH_CALUDE_tetrahedron_volume_lower_bound_l351_35178

-- Define a tetrahedron type
structure Tetrahedron where
  -- The volume of the tetrahedron
  volume : ℝ
  -- The distances between opposite edges
  d₁ : ℝ
  d₂ : ℝ
  d₃ : ℝ
  -- Ensure all distances are positive
  d₁_pos : d₁ > 0
  d₂_pos : d₂ > 0
  d₃_pos : d₃ > 0

-- State the theorem
theorem tetrahedron_volume_lower_bound (t : Tetrahedron) : 
  t.volume ≥ (1/3) * t.d₁ * t.d₂ * t.d₃ := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_lower_bound_l351_35178


namespace NUMINAMATH_CALUDE_middle_share_is_forty_l351_35137

/-- Represents the distribution of marbles among three people -/
structure MarbleDistribution where
  total : ℕ
  ratio1 : ℕ
  ratio2 : ℕ
  ratio3 : ℕ

/-- Calculates the number of marbles for the person with the middle ratio -/
def middleShare (d : MarbleDistribution) : ℕ :=
  d.total * d.ratio2 / (d.ratio1 + d.ratio2 + d.ratio3)

/-- Theorem: In a distribution of 120 marbles with ratio 4:5:6, the middle share is 40 -/
theorem middle_share_is_forty : 
  let d : MarbleDistribution := ⟨120, 4, 5, 6⟩
  middleShare d = 40 := by sorry


end NUMINAMATH_CALUDE_middle_share_is_forty_l351_35137


namespace NUMINAMATH_CALUDE_white_marbles_count_l351_35126

theorem white_marbles_count (total : ℕ) (black red green : ℕ) 
  (h_total : total = 60)
  (h_black : black = 32)
  (h_red : red = 10)
  (h_green : green = 5)
  (h_sum : total = black + red + green + (total - (black + red + green))) :
  total - (black + red + green) = 13 := by
  sorry

end NUMINAMATH_CALUDE_white_marbles_count_l351_35126


namespace NUMINAMATH_CALUDE_sine_cosine_power_inequality_l351_35189

theorem sine_cosine_power_inequality (n m : ℕ) (hn : n > 0) (hm : m > 0) (hnm : n > m) :
  ∀ x : ℝ, 0 < x ∧ x < π / 2 →
    2 * |Real.sin x ^ n - Real.cos x ^ n| ≤ 3 * |Real.sin x ^ m - Real.cos x ^ m| :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_power_inequality_l351_35189


namespace NUMINAMATH_CALUDE_multiply_469158_and_9999_l351_35191

theorem multiply_469158_and_9999 : 469158 * 9999 = 4691176842 := by
  sorry

end NUMINAMATH_CALUDE_multiply_469158_and_9999_l351_35191


namespace NUMINAMATH_CALUDE_polynomial_factorization_l351_35151

theorem polynomial_factorization (a b c : ℝ) :
  2*a*(b - c)^3 + 3*b*(c - a)^3 + 2*c*(a - b)^3 = (a - b)*(b - c)*(c - a)*(5*b - c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l351_35151


namespace NUMINAMATH_CALUDE_x_value_proof_l351_35130

theorem x_value_proof (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x - y^2 = 3) (h4 : x^2 + y^4 = 13) : 
  x = (3 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l351_35130


namespace NUMINAMATH_CALUDE_intersection_complement_when_a_is_two_union_equality_iff_a_leq_two_l351_35185

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for part I
theorem intersection_complement_when_a_is_two :
  M ∩ (Set.univ \ N 2) = {x | -2 ≤ x ∧ x < 3} := by sorry

-- Theorem for part II
theorem union_equality_iff_a_leq_two (a : ℝ) :
  M ∪ N a = M ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_a_is_two_union_equality_iff_a_leq_two_l351_35185


namespace NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_l351_35160

-- Define the function f
def f (b c x : ℝ) : ℝ := x * abs x + b * x + c

-- Proposition 1
theorem prop_1 (b : ℝ) : 
  ∀ x, f b 0 (-x) = -(f b 0 x) := by sorry

-- Proposition 2
theorem prop_2 (c : ℝ) (h : c > 0) : 
  ∃! x, f 0 c x = 0 := by sorry

-- Proposition 3
theorem prop_3 (b c : ℝ) : 
  ∀ x, f b c (-x) = 2 * c - f b c x := by sorry

end NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_l351_35160


namespace NUMINAMATH_CALUDE_largest_number_value_l351_35143

theorem largest_number_value (a b c : ℕ) : 
  a < b ∧ b < c ∧
  a + b + c = 80 ∧
  c = b + 9 ∧
  b = a + 4 ∧
  a * b = 525 →
  c = 34 := by
sorry

end NUMINAMATH_CALUDE_largest_number_value_l351_35143


namespace NUMINAMATH_CALUDE_jack_shoe_time_proof_l351_35124

/-- The time it takes Jack to put on his shoes -/
def jack_shoe_time : ℝ := 4

/-- The time it takes Jack to help one toddler with their shoes -/
def toddler_shoe_time (j : ℝ) : ℝ := j + 3

/-- The total time for Jack and two toddlers to get ready -/
def total_time (j : ℝ) : ℝ := j + 2 * (toddler_shoe_time j)

theorem jack_shoe_time_proof :
  total_time jack_shoe_time = 18 :=
by sorry

end NUMINAMATH_CALUDE_jack_shoe_time_proof_l351_35124


namespace NUMINAMATH_CALUDE_scramble_word_count_l351_35188

/-- The number of letters in the extended Kobish alphabet -/
def alphabet_size : ℕ := 21

/-- The maximum length of a word -/
def max_word_length : ℕ := 4

/-- Calculates the number of words of a given length that contain the letter B at least once -/
def words_with_b (length : ℕ) : ℕ :=
  alphabet_size^length - (alphabet_size - 1)^length

/-- The total number of valid words in the Scramble language -/
def total_valid_words : ℕ :=
  words_with_b 1 + words_with_b 2 + words_with_b 3 + words_with_b 4

theorem scramble_word_count : total_valid_words = 35784 := by
  sorry

end NUMINAMATH_CALUDE_scramble_word_count_l351_35188


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l351_35102

theorem cyclist_speed_ratio :
  ∀ (v₁ v₂ : ℝ), v₁ > v₂ → v₁ > 0 → v₂ > 0 →
  (v₁ + v₂ = 25) →
  (v₁ - v₂ = 10 / 3) →
  (v₁ / v₂ = 17 / 13) := by
sorry

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l351_35102


namespace NUMINAMATH_CALUDE_intersection_M_N_l351_35110

def M : Set ℝ := {x | 3 * x - 6 ≥ 0}
def N : Set ℝ := {x | x^2 < 16}

theorem intersection_M_N : M ∩ N = Set.Icc 2 4 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l351_35110


namespace NUMINAMATH_CALUDE_cubic_root_sum_l351_35104

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I + 2 : ℂ) ^ 3 + a * (Complex.I + 2) + b = 0 → a + b = 20 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l351_35104


namespace NUMINAMATH_CALUDE_A_subset_A_inter_B_iff_l351_35128

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- State the theorem
theorem A_subset_A_inter_B_iff (a : ℝ) : 
  (A a).Nonempty → (A a ⊆ A a ∩ B ↔ 6 ≤ a ∧ a ≤ 9) := by sorry

end NUMINAMATH_CALUDE_A_subset_A_inter_B_iff_l351_35128


namespace NUMINAMATH_CALUDE_slope_angle_of_y_equals_1_l351_35154

-- Define a line parallel to the x-axis
def parallel_to_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y

-- Define the slope angle of a line
def slope_angle (f : ℝ → ℝ) : ℝ := sorry

-- Theorem: The slope angle of the line y = 1 is 0
theorem slope_angle_of_y_equals_1 :
  let f : ℝ → ℝ := λ x => 1
  parallel_to_x_axis f ∧ slope_angle f = 0 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_y_equals_1_l351_35154


namespace NUMINAMATH_CALUDE_nathan_daily_hours_l351_35145

/-- Proves that Nathan played 3 hours per day given the conditions of the problem -/
theorem nathan_daily_hours : ∃ x : ℕ, 
  (14 * x + 5 * 7 = 77) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_nathan_daily_hours_l351_35145


namespace NUMINAMATH_CALUDE_batsman_average_l351_35133

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℝ) :
  total_innings = 12 →
  last_innings_score = 92 →
  average_increase = 2 →
  (((total_innings - 1 : ℝ) * ((total_innings * (70 : ℝ) - last_innings_score : ℝ) / total_innings) + last_innings_score) / total_innings) - 
  ((total_innings * (70 : ℝ) - last_innings_score : ℝ) / total_innings) = average_increase →
  (((total_innings - 1 : ℝ) * ((total_innings * (70 : ℝ) - last_innings_score : ℝ) / total_innings) + last_innings_score) / total_innings) = 70 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_l351_35133


namespace NUMINAMATH_CALUDE_deepak_age_l351_35177

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age -/
theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (rahul_future_age : ℕ) 
  (years_ahead : ℕ) :
  rahul_ratio = 4 →
  deepak_ratio = 3 →
  rahul_future_age = 38 →
  years_ahead = 6 →
  ∃ (x : ℕ), rahul_ratio * x + years_ahead = rahul_future_age ∧ 
             deepak_ratio * x = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l351_35177


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l351_35156

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l351_35156


namespace NUMINAMATH_CALUDE_jeans_discount_rates_l351_35155

def regular_price_moose : ℝ := 20
def regular_price_fox : ℝ := 15
def regular_price_pony : ℝ := 18

def num_moose : ℕ := 2
def num_fox : ℕ := 3
def num_pony : ℕ := 2

def total_savings : ℝ := 12.48

def sum_all_rates : ℝ := 0.32
def sum_fox_pony_rates : ℝ := 0.20

def discount_rate_moose : ℝ := 0.12
def discount_rate_fox : ℝ := 0.0533
def discount_rate_pony : ℝ := 0.1467

theorem jeans_discount_rates :
  (discount_rate_moose + discount_rate_fox + discount_rate_pony = sum_all_rates) ∧
  (discount_rate_fox + discount_rate_pony = sum_fox_pony_rates) ∧
  (num_moose * discount_rate_moose * regular_price_moose +
   num_fox * discount_rate_fox * regular_price_fox +
   num_pony * discount_rate_pony * regular_price_pony = total_savings) :=
by sorry

end NUMINAMATH_CALUDE_jeans_discount_rates_l351_35155


namespace NUMINAMATH_CALUDE_carly_nail_trimming_l351_35165

/-- Calculates the total number of nails trimmed by a pet groomer --/
def total_nails_trimmed (total_dogs : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) : ℕ :=
  let four_legged_dogs := total_dogs - three_legged_dogs
  let nails_per_four_legged_dog := 4 * nails_per_paw
  let nails_per_three_legged_dog := 3 * nails_per_paw
  four_legged_dogs * nails_per_four_legged_dog + three_legged_dogs * nails_per_three_legged_dog

theorem carly_nail_trimming :
  total_nails_trimmed 11 3 4 = 164 := by
  sorry

end NUMINAMATH_CALUDE_carly_nail_trimming_l351_35165


namespace NUMINAMATH_CALUDE_insufficient_info_to_determine_C_l351_35171

/-- A line in the xy-plane defined by the equation x = 8y + C -/
structure Line where
  C : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  p.x = 8 * p.y + l.C

theorem insufficient_info_to_determine_C 
  (m n : ℝ) (l : Line) :
  let p1 : Point := ⟨m, n⟩
  let p2 : Point := ⟨m + 2, n + 0.25⟩
  p1.on_line l ∧ p2.on_line l →
  ∃ (C' : ℝ), C' ≠ l.C ∧ 
    (⟨m, n⟩ : Point).on_line ⟨C'⟩ ∧ 
    (⟨m + 2, n + 0.25⟩ : Point).on_line ⟨C'⟩ :=
sorry

end NUMINAMATH_CALUDE_insufficient_info_to_determine_C_l351_35171


namespace NUMINAMATH_CALUDE_quadratic_equation_with_root_difference_l351_35116

theorem quadratic_equation_with_root_difference (c : ℝ) : 
  (∃ (r₁ r₂ : ℝ), 2 * r₁^2 + 5 * r₁ = c ∧ 
                   2 * r₂^2 + 5 * r₂ = c ∧ 
                   r₂ = r₁ + 5.5) → 
  c = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_root_difference_l351_35116


namespace NUMINAMATH_CALUDE_point_positions_l351_35121

def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 2*x + 4*y - 4

def point_M : ℝ × ℝ := (2, -4)
def point_N : ℝ × ℝ := (-2, 1)

theorem point_positions :
  circle_equation point_M.1 point_M.2 < 0 ∧ 
  circle_equation point_N.1 point_N.2 > 0 := by
sorry

end NUMINAMATH_CALUDE_point_positions_l351_35121


namespace NUMINAMATH_CALUDE_number_2018_in_equation_31_l351_35175

def first_term (n : ℕ) : ℕ := 2 * n^2

theorem number_2018_in_equation_31 :
  ∃ k : ℕ, k ≥ first_term 31 ∧ k ≤ first_term 32 ∧ k = 2018 :=
by sorry

end NUMINAMATH_CALUDE_number_2018_in_equation_31_l351_35175


namespace NUMINAMATH_CALUDE_largest_number_with_13_matchsticks_has_digit_sum_9_l351_35192

/-- Represents the number of matchsticks needed to form each digit --/
def matchsticks_per_digit : Fin 10 → ℕ
| 0 => 6
| 1 => 2
| 2 => 5
| 3 => 5
| 4 => 4
| 5 => 5
| 6 => 6
| 7 => 3
| 8 => 7
| 9 => 6

/-- Represents a number as a list of digits --/
def Number := List (Fin 10)

/-- Calculates the sum of digits in a number --/
def sum_of_digits (n : Number) : ℕ :=
  n.foldl (fun acc d => acc + d.val) 0

/-- Calculates the total number of matchsticks used to form a number --/
def matchsticks_used (n : Number) : ℕ :=
  n.foldl (fun acc d => acc + matchsticks_per_digit d) 0

/-- Checks if a number is valid (uses exactly 13 matchsticks) --/
def is_valid_number (n : Number) : Prop :=
  matchsticks_used n = 13

/-- Compares two numbers lexicographically --/
def number_gt (a b : Number) : Prop :=
  match a, b with
  | [], [] => False
  | _ :: _, [] => True
  | [], _ :: _ => False
  | x :: xs, y :: ys => x > y ∨ (x = y ∧ number_gt xs ys)

/-- The main theorem to be proved --/
theorem largest_number_with_13_matchsticks_has_digit_sum_9 :
  ∃ (n : Number), is_valid_number n ∧
    (∀ (m : Number), is_valid_number m → number_gt n m ∨ n = m) ∧
    sum_of_digits n = 9 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_13_matchsticks_has_digit_sum_9_l351_35192


namespace NUMINAMATH_CALUDE_ternary_57_has_four_digits_l351_35180

/-- Converts a natural number to its ternary (base 3) representation -/
def to_ternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The theorem stating that the ternary representation of 57 has exactly 4 digits -/
theorem ternary_57_has_four_digits : (to_ternary 57).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ternary_57_has_four_digits_l351_35180


namespace NUMINAMATH_CALUDE_product_equality_l351_35181

theorem product_equality (x y : ℝ) 
  (h : (x + Real.sqrt (1 + y^2)) * (y + Real.sqrt (1 + x^2)) = 1) :
  (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l351_35181


namespace NUMINAMATH_CALUDE_divisible_by_nine_l351_35125

theorem divisible_by_nine (k : ℕ+) : 
  (9 : ℤ) ∣ (3 * (2 + 7^(k : ℕ))) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l351_35125


namespace NUMINAMATH_CALUDE_coffee_shop_sales_l351_35146

theorem coffee_shop_sales (teas lattes extra : ℕ) : 
  teas = 6 → lattes = 32 → 4 * teas + extra = lattes → extra = 8 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_sales_l351_35146
