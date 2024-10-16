import Mathlib

namespace NUMINAMATH_CALUDE_train_crossing_time_l354_35465

/-- Calculates the time for a train to cross a signal pole given its length, 
    the length of a platform it crosses, and the time it takes to cross the platform. -/
theorem train_crossing_time (train_length platform_length : ℝ) (platform_crossing_time : ℝ) 
  (h1 : train_length = 150)
  (h2 : platform_length = 175)
  (h3 : platform_crossing_time = 39) :
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l354_35465


namespace NUMINAMATH_CALUDE_robbery_participants_l354_35438

theorem robbery_participants :
  -- Define variables for each suspect's guilt
  (A B V G : Prop) →
  -- Condition 1
  (¬G → (B ∧ ¬A)) →
  -- Condition 2
  (V → (¬A ∧ ¬B)) →
  -- Condition 3
  (G → B) →
  -- Condition 4
  (B → (A ∨ V)) →
  -- Conclusion: Alexey, Boris, and Grigory are guilty, Veniamin is innocent
  (A ∧ B ∧ G ∧ ¬V) :=
by sorry

end NUMINAMATH_CALUDE_robbery_participants_l354_35438


namespace NUMINAMATH_CALUDE_equal_probability_sums_l354_35496

/-- A standard die with faces labeled 1 to 6 -/
def StandardDie := Fin 6

/-- The number of dice being rolled -/
def numDice : ℕ := 9

/-- The sum we're comparing to -/
def compareSum : ℕ := 15

/-- The sum we're proving has the same probability -/
def targetSum : ℕ := 48

/-- A function to calculate the probability of a specific sum occurring when rolling n dice -/
noncomputable def probabilityOfSum (n : ℕ) (sum : ℕ) : ℝ := sorry

theorem equal_probability_sums :
  probabilityOfSum numDice compareSum = probabilityOfSum numDice targetSum :=
sorry

end NUMINAMATH_CALUDE_equal_probability_sums_l354_35496


namespace NUMINAMATH_CALUDE_youtube_video_dislikes_l354_35464

theorem youtube_video_dislikes :
  let initial_likes : ℕ := 5000
  let initial_dislikes : ℕ := (initial_likes / 3) + 50
  let likes_increase : ℕ := 2000
  let dislikes_increase : ℕ := 400
  let new_likes : ℕ := initial_likes + likes_increase
  let new_dislikes : ℕ := initial_dislikes + dislikes_increase
  let doubled_new_likes : ℕ := 2 * new_likes
  doubled_new_likes - new_dislikes = 11983 ∧ new_dislikes = 2017 :=
by sorry


end NUMINAMATH_CALUDE_youtube_video_dislikes_l354_35464


namespace NUMINAMATH_CALUDE_grid_paths_equals_choose_l354_35491

/-- The number of paths from (0,0) to (m,n) in a grid, moving only right or up -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

/-- Theorem: The number of paths in an m × n grid is (m+n) choose n -/
theorem grid_paths_equals_choose (m n : ℕ) : 
  gridPaths m n = Nat.choose (m + n) n := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_equals_choose_l354_35491


namespace NUMINAMATH_CALUDE_work_completion_theorem_l354_35437

theorem work_completion_theorem (work : ℝ) (days1 days2 : ℝ) (men1 : ℕ) (men2 : ℕ) :
  work = men1 * days1 ∧ work = men2 * days2 ∧ men1 = 14 ∧ days1 = 25 ∧ days2 = 17.5 →
  men2 = 20 := by
sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l354_35437


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l354_35421

/-- Given that x and y are positive real numbers, x² and y vary inversely,
    and y = 25 when x = 3, prove that x = √3/4 when y = 1200. -/
theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ k : ℝ, ∀ x y, x * x * y = k)
  (h_initial : 3 * 3 * 25 = 9 * 25) :
  y = 1200 → x = Real.sqrt 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l354_35421


namespace NUMINAMATH_CALUDE_count_a_in_sentence_l354_35475

def sentence : String := "Happy Teachers'Day!"

theorem count_a_in_sentence : 
  (sentence.toList.filter (· = 'a')).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_a_in_sentence_l354_35475


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l354_35466

theorem vector_subtraction_and_scalar_multiplication :
  let v₁ : Fin 2 → ℝ := ![3, -8]
  let v₂ : Fin 2 → ℝ := ![2, -6]
  v₁ - 5 • v₂ = ![-7, 22] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l354_35466


namespace NUMINAMATH_CALUDE_christine_savings_l354_35440

/-- Calculates the amount saved by a salesperson given their commission rate, total sales, and personal needs allocation percentage. -/
def amount_saved (commission_rate : ℚ) (total_sales : ℚ) (personal_needs_percent : ℚ) : ℚ :=
  let total_commission := commission_rate * total_sales
  let personal_needs := personal_needs_percent * total_commission
  total_commission - personal_needs

/-- Proves that given the specific conditions, the amount saved is $1152. -/
theorem christine_savings : 
  amount_saved (12/100) 24000 (60/100) = 1152 := by
  sorry

end NUMINAMATH_CALUDE_christine_savings_l354_35440


namespace NUMINAMATH_CALUDE_linear_system_solution_l354_35404

/-- Given a system of linear equations and conditions, prove the range of m and its integer values -/
theorem linear_system_solution (m x y : ℝ) : 
  (2 * x + y = 1 + 2 * m) → 
  (x + 2 * y = 2 - m) → 
  (x + y > 0) → 
  (m > -3) ∧ 
  (((2 * m + 1) * x - 2 * m < 1) → 
   (x > 1) → 
   (m = -2 ∨ m = -1)) := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l354_35404


namespace NUMINAMATH_CALUDE_negation_equivalence_l354_35444

-- Define the proposition for "at least one of a, b, c is positive"
def atLeastOnePositive (a b c : ℝ) : Prop := a > 0 ∨ b > 0 ∨ c > 0

-- Define the proposition for "a, b, c are all non-positive"
def allNonPositive (a b c : ℝ) : Prop := a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0

-- Theorem stating the negation equivalence
theorem negation_equivalence (a b c : ℝ) : 
  ¬(atLeastOnePositive a b c) ↔ allNonPositive a b c :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l354_35444


namespace NUMINAMATH_CALUDE_finance_club_probability_l354_35454

theorem finance_club_probability (total_students : ℕ) (interested_fraction : ℚ) 
  (h1 : total_students = 20)
  (h2 : interested_fraction = 3/4) :
  let interested_students := (interested_fraction * total_students).num
  let not_interested_students := total_students - interested_students
  1 - (not_interested_students / total_students) * ((not_interested_students - 1) / (total_students - 1)) = 18/19 := by
sorry

end NUMINAMATH_CALUDE_finance_club_probability_l354_35454


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l354_35499

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l354_35499


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l354_35481

/-- A line y = 3x + d is tangent to the parabola y^2 = 12x if and only if d = 1 -/
theorem line_tangent_to_parabola (d : ℝ) :
  (∃ x y : ℝ, y = 3 * x + d ∧ y^2 = 12 * x ∧
    ∀ x' y' : ℝ, y' = 3 * x' + d → y'^2 ≤ 12 * x') ↔ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l354_35481


namespace NUMINAMATH_CALUDE_fraction_equality_l354_35497

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l354_35497


namespace NUMINAMATH_CALUDE_prop_p_and_q_false_iff_a_gt_1_l354_35434

-- Define the propositions p and q
def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → a^x > a^y

def q (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, Real.log (a*x^2 - x + a) = y

-- State the theorem
theorem prop_p_and_q_false_iff_a_gt_1 :
  ∀ a : ℝ, (¬(p a ∧ q a)) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_prop_p_and_q_false_iff_a_gt_1_l354_35434


namespace NUMINAMATH_CALUDE_four_sacks_filled_l354_35431

/-- Calculates the number of sacks filled given the total pieces of wood and capacity per sack -/
def sacks_filled (total_wood : ℕ) (wood_per_sack : ℕ) : ℕ :=
  total_wood / wood_per_sack

/-- Theorem: Given 80 pieces of wood and sacks that can hold 20 pieces each, 4 sacks will be filled -/
theorem four_sacks_filled : sacks_filled 80 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_sacks_filled_l354_35431


namespace NUMINAMATH_CALUDE_unique_pair_exists_l354_35445

theorem unique_pair_exists (n : ℕ) : ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_exists_l354_35445


namespace NUMINAMATH_CALUDE_max_value_implies_a_l354_35408

def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

theorem max_value_implies_a (a : ℝ) :
  (∀ x, a - 2 ≤ x ∧ x ≤ a + 1 → f x ≤ 3) ∧
  (∃ x, a - 2 ≤ x ∧ x ≤ a + 1 ∧ f x = 3) →
  a = 0 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l354_35408


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l354_35463

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : a * b = 6)
  (h2 : a * c = 8)
  (h3 : b * c = 12)
  : Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l354_35463


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l354_35489

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_theorem :
  can_form_triangle 8 6 4 ∧
  ¬can_form_triangle 2 4 6 ∧
  ¬can_form_triangle 14 6 7 ∧
  ¬can_form_triangle 2 3 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l354_35489


namespace NUMINAMATH_CALUDE_subtract_sqrt_25_equals_negative_2_l354_35406

theorem subtract_sqrt_25_equals_negative_2 : 3 - Real.sqrt 25 = -2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_sqrt_25_equals_negative_2_l354_35406


namespace NUMINAMATH_CALUDE_max_y_diff_intersection_points_l354_35435

/-- The maximum difference between the y-coordinates of the intersection points
    of y = 4 - 2x^2 + x^3 and y = 2 + x^2 + x^3 is 4√2/9. -/
theorem max_y_diff_intersection_points :
  let f (x : ℝ) := 4 - 2 * x^2 + x^3
  let g (x : ℝ) := 2 + x^2 + x^3
  let intersection_points := {x : ℝ | f x = g x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ intersection_points ∧ x₂ ∈ intersection_points ∧
    |f x₁ - f x₂| = 4 * Real.sqrt 2 / 9 ∧
    ∀ (y₁ y₂ : ℝ), y₁ ∈ intersection_points → y₂ ∈ intersection_points →
      |f y₁ - f y₂| ≤ 4 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_y_diff_intersection_points_l354_35435


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l354_35483

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((2 * Complex.I - 5) / (2 - Complex.I)) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l354_35483


namespace NUMINAMATH_CALUDE_island_perimeter_calculation_l354_35401

/-- The perimeter of a rectangular island -/
def island_perimeter (width : ℝ) (length : ℝ) : ℝ :=
  2 * (width + length)

/-- Theorem: The perimeter of a rectangular island with width 4 miles and length 7 miles is 22 miles -/
theorem island_perimeter_calculation :
  island_perimeter 4 7 = 22 := by
  sorry

end NUMINAMATH_CALUDE_island_perimeter_calculation_l354_35401


namespace NUMINAMATH_CALUDE_dresser_shirts_count_l354_35492

/-- Given a dresser with pants and shirts in the ratio of 7:10, 
    and 14 pants, prove that there are 20 shirts. -/
theorem dresser_shirts_count (pants_count : ℕ) (ratio_pants : ℕ) (ratio_shirts : ℕ) :
  pants_count = 14 →
  ratio_pants = 7 →
  ratio_shirts = 10 →
  (pants_count : ℚ) / ratio_pants * ratio_shirts = 20 := by
  sorry

end NUMINAMATH_CALUDE_dresser_shirts_count_l354_35492


namespace NUMINAMATH_CALUDE_students_in_front_of_yuna_l354_35450

/-- Given a line of students with Yuna somewhere in the line, this theorem
    proves the number of students in front of Yuna. -/
theorem students_in_front_of_yuna 
  (total_students : ℕ) 
  (students_behind_yuna : ℕ) 
  (h1 : total_students = 25)
  (h2 : students_behind_yuna = 9) :
  total_students - (students_behind_yuna + 1) = 15 :=
by sorry

end NUMINAMATH_CALUDE_students_in_front_of_yuna_l354_35450


namespace NUMINAMATH_CALUDE_rectangle_max_area_l354_35480

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) :
  2 * x + 2 * y = 40 → x * y ≤ 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l354_35480


namespace NUMINAMATH_CALUDE_equation_equality_l354_35416

theorem equation_equality (x y z : ℝ) (h : x / y = 3 / z) : 9 * y^2 = x^2 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l354_35416


namespace NUMINAMATH_CALUDE_range_of_m_l354_35498

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a line in the form 2x + y + m = 0 -/
def Line (m : ℝ) (p : Point) : Prop :=
  2 * p.x + p.y + m = 0

/-- Defines when two points are on opposite sides of a line -/
def OppositesSides (m : ℝ) (p1 p2 : Point) : Prop :=
  (2 * p1.x + p1.y + m) * (2 * p2.x + p2.y + m) < 0

/-- The main theorem -/
theorem range_of_m (p1 p2 : Point) (h : OppositesSides m p1 p2) 
  (h1 : p1 = ⟨1, 3⟩) (h2 : p2 = ⟨-4, -2⟩) : 
  -5 < m ∧ m < 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l354_35498


namespace NUMINAMATH_CALUDE_set_operations_l354_35412

def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | x^2 + 18 < 11*x}

theorem set_operations :
  (Set.compl (A ∩ B) = {x | x < 3 ∨ x ≥ 6}) ∧
  ((Set.compl B ∪ A) = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l354_35412


namespace NUMINAMATH_CALUDE_pennies_spent_l354_35423

/-- Given that Sam initially had 98 pennies and now has 5 pennies left,
    prove that the number of pennies Sam spent is 93. -/
theorem pennies_spent (initial : Nat) (left : Nat) (spent : Nat)
    (h1 : initial = 98)
    (h2 : left = 5)
    (h3 : spent = initial - left) :
  spent = 93 := by
  sorry

end NUMINAMATH_CALUDE_pennies_spent_l354_35423


namespace NUMINAMATH_CALUDE_solve_for_m_l354_35411

theorem solve_for_m (m : ℝ) (h1 : m ≠ 0) :
  (∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 12)) →
  m = 12 := by
sorry

end NUMINAMATH_CALUDE_solve_for_m_l354_35411


namespace NUMINAMATH_CALUDE_inequality_range_l354_35495

theorem inequality_range (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) :
  (∀ k > 0, Real.log (Real.sin θ)^2 - Real.log (Real.cos θ)^2 ≤ k * Real.cos (2 * θ)) ↔
  θ ∈ Set.Ioo 0 (Real.pi / 4) ∪ Set.Icc (3 * Real.pi / 4) Real.pi ∪ 
      Set.Ioo Real.pi (5 * Real.pi / 4) ∪ Set.Icc (7 * Real.pi / 4) (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l354_35495


namespace NUMINAMATH_CALUDE_uma_income_l354_35451

/-- Represents the income and expenditure of a person -/
structure Person where
  income : ℝ
  expenditure : ℝ

/-- The problem setup -/
def problem_setup (uma bala : Person) : Prop :=
  -- Income ratio
  uma.income / bala.income = 8 / 7 ∧
  -- Expenditure ratio
  uma.expenditure / bala.expenditure = 7 / 6 ∧
  -- Savings
  uma.income - uma.expenditure = 2000 ∧
  bala.income - bala.expenditure = 2000

/-- The theorem to prove -/
theorem uma_income (uma bala : Person) :
  problem_setup uma bala → uma.income = 8000 / 7.5 := by
  sorry

end NUMINAMATH_CALUDE_uma_income_l354_35451


namespace NUMINAMATH_CALUDE_min_score_on_last_two_l354_35471

/-- The number of tests Shauna takes -/
def num_tests : ℕ := 5

/-- The maximum score possible on each test -/
def max_score : ℕ := 120

/-- The desired average score across all tests -/
def target_average : ℕ := 95

/-- Shauna's scores on the first three tests -/
def first_three_scores : Fin 3 → ℕ
  | 0 => 86
  | 1 => 112
  | 2 => 91

/-- The sum of Shauna's scores on the first three tests -/
def sum_first_three : ℕ := (first_three_scores 0) + (first_three_scores 1) + (first_three_scores 2)

/-- The theorem stating the minimum score needed on one of the last two tests -/
theorem min_score_on_last_two (score : ℕ) :
  (sum_first_three + score + max_score = target_average * num_tests) ∧
  (∀ s, s < score → sum_first_three + s + max_score < target_average * num_tests) →
  score = 66 := by
  sorry

end NUMINAMATH_CALUDE_min_score_on_last_two_l354_35471


namespace NUMINAMATH_CALUDE_digit_789_of_7_29_l354_35476

def decimal_representation_7_29 : List ℕ :=
  [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7]

def repeating_period : ℕ := 28

theorem digit_789_of_7_29 : 
  (decimal_representation_7_29[(789 % repeating_period) - 1]) = 6 := by sorry

end NUMINAMATH_CALUDE_digit_789_of_7_29_l354_35476


namespace NUMINAMATH_CALUDE_parabola_directrix_l354_35427

/-- Given a parabola with equation y = ax^2 and directrix y = 1, prove that a = -1/4 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Condition 1: Equation of the parabola
  (∃ y : ℝ, y = 1 ∧ ∀ x : ℝ, y ≠ a * x^2) →  -- Condition 2: Equation of the directrix
  a = -1/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l354_35427


namespace NUMINAMATH_CALUDE_chicken_coops_count_l354_35468

theorem chicken_coops_count (chickens_per_coop : ℕ) (total_chickens : ℕ) 
  (h1 : chickens_per_coop = 60) 
  (h2 : total_chickens = 540) : 
  total_chickens / chickens_per_coop = 9 := by
  sorry

end NUMINAMATH_CALUDE_chicken_coops_count_l354_35468


namespace NUMINAMATH_CALUDE_purely_imaginary_modulus_l354_35474

/-- Given a complex number z = (a + 3i) / (1 + 2i) where a is real,
    if z is purely imaginary, then |z| = 3 -/
theorem purely_imaginary_modulus (a : ℝ) :
  let z : ℂ := (a + 3 * Complex.I) / (1 + 2 * Complex.I)
  (∃ b : ℝ, z = b * Complex.I) → Complex.abs z = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_modulus_l354_35474


namespace NUMINAMATH_CALUDE_part_one_part_two_l354_35452

-- Define the linear function f
def f (a b x : ℝ) : ℝ := a * x + b

-- Define the function g
def g (m x : ℝ) : ℝ := (x + m) * (4 * x + 1)

-- Theorem for part (I)
theorem part_one (a b : ℝ) :
  (∀ x y, x < y → f a b x < f a b y) →
  (∀ x, f a b (f a b x) = 16 * x + 5) →
  a = 4 ∧ b = 1 := by sorry

-- Theorem for part (II)
theorem part_two (m : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y → g m x < g m y) →
  m ≥ -9/4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l354_35452


namespace NUMINAMATH_CALUDE_path_area_and_cost_l354_35439

def field_length : ℝ := 95
def field_width : ℝ := 55
def path_width : ℝ := 2.5
def cost_per_sqm : ℝ := 2

def total_length : ℝ := field_length + 2 * path_width
def total_width : ℝ := field_width + 2 * path_width

def total_area : ℝ := total_length * total_width
def field_area : ℝ := field_length * field_width
def path_area : ℝ := total_area - field_area

def construction_cost : ℝ := path_area * cost_per_sqm

theorem path_area_and_cost :
  path_area = 775 ∧ construction_cost = 1550 := by sorry

end NUMINAMATH_CALUDE_path_area_and_cost_l354_35439


namespace NUMINAMATH_CALUDE_point_on_x_axis_l354_35400

theorem point_on_x_axis (x : ℝ) : 
  (x^2 + 2 + 9 = 12) → (x = 1 ∨ x = -1) := by
  sorry

#check point_on_x_axis

end NUMINAMATH_CALUDE_point_on_x_axis_l354_35400


namespace NUMINAMATH_CALUDE_max_sum_is_27_l354_35448

/-- Represents the arrangement of numbers in the grid -/
structure Arrangement where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {2, 5, 8, 11, 14}

/-- Checks if an arrangement is valid according to the problem conditions -/
def isValidArrangement (arr : Arrangement) : Prop :=
  (arr.a ∈ availableNumbers) ∧
  (arr.b ∈ availableNumbers) ∧
  (arr.c ∈ availableNumbers) ∧
  (arr.d ∈ availableNumbers) ∧
  (arr.e ∈ availableNumbers) ∧
  (arr.f ∈ availableNumbers) ∧
  (arr.a + arr.b + arr.e = arr.c + arr.d + arr.f) ∧
  (arr.a + arr.c = arr.b + arr.d) ∧
  (arr.a + arr.c = arr.e + arr.f)

/-- The theorem to be proven -/
theorem max_sum_is_27 :
  ∀ (arr : Arrangement), isValidArrangement arr →
  (arr.a + arr.b + arr.e ≤ 27 ∧ arr.c + arr.d + arr.f ≤ 27) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_is_27_l354_35448


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l354_35461

/-- Inverse proportion function passing through (2,1) has k = 2 -/
theorem inverse_proportion_k_value (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = k / x) ∧ f 2 = 1) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l354_35461


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l354_35490

theorem absolute_value_equation_solution :
  ∃! n : ℝ, |2 * n + 8| = 3 * n - 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l354_35490


namespace NUMINAMATH_CALUDE_min_product_positive_reals_l354_35453

theorem min_product_positive_reals (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  x ≤ 2 * (y + z) →
  y ≤ 2 * (x + z) →
  z ≤ 2 * (x + y) →
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 1 → 
    a ≤ 2 * (b + c) → b ≤ 2 * (a + c) → c ≤ 2 * (a + b) →
    x * y * z ≤ a * b * c →
  x * y * z = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_min_product_positive_reals_l354_35453


namespace NUMINAMATH_CALUDE_square_circle_union_area_l354_35410

/-- The area of the union of a square and a circle with specific dimensions -/
theorem square_circle_union_area (square_side : ℝ) (circle_radius : ℝ) : 
  square_side = 8 →
  circle_radius = 12 →
  (square_side ^ 2) + (Real.pi * circle_radius ^ 2) - (Real.pi * circle_radius ^ 2 / 4) = 64 + 108 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l354_35410


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l354_35487

theorem square_circle_area_ratio :
  ∀ (r : ℝ) (s₁ s₂ : ℝ),
  r > 0 → s₁ > 0 → s₂ > 0 →
  2 * π * r = 4 * s₁ →  -- Circle and first square have same perimeter
  2 * r = s₂ * Real.sqrt 2 →  -- Diameter of circle is diagonal of second square
  (s₂^2) / (s₁^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l354_35487


namespace NUMINAMATH_CALUDE_line_property_l354_35459

/-- Given a line passing through points (2, -1) and (-1, 6), prove that 3m - 2b = -19 where m is the slope and b is the y-intercept -/
theorem line_property (m b : ℚ) : 
  (∀ (x y : ℚ), (x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 6) → y = m * x + b) →
  3 * m - 2 * b = -19 := by
  sorry

end NUMINAMATH_CALUDE_line_property_l354_35459


namespace NUMINAMATH_CALUDE_unknown_interest_rate_l354_35407

/-- Proves that given the conditions of the problem, the unknown interest rate is 6% -/
theorem unknown_interest_rate (total : ℚ) (part1 : ℚ) (part2 : ℚ) (rate1 : ℚ) (rate2 : ℚ) (yearly_income : ℚ) :
  total = 2600 →
  part1 = 1600 →
  part2 = total - part1 →
  rate1 = 5 / 100 →
  yearly_income = 140 →
  yearly_income = part1 * rate1 + part2 * rate2 →
  rate2 = 6 / 100 := by
sorry

#eval (6 : ℚ) / 100

end NUMINAMATH_CALUDE_unknown_interest_rate_l354_35407


namespace NUMINAMATH_CALUDE_number_equation_solution_l354_35470

theorem number_equation_solution :
  ∀ x : ℝ, 35 + 3 * x^2 = 89 → x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_number_equation_solution_l354_35470


namespace NUMINAMATH_CALUDE_expansion_properties_l354_35424

def binomial_sum (n : ℕ) : ℕ := 2^n

theorem expansion_properties (x : ℝ) :
  let n : ℕ := 8
  let binomial_sum_diff : ℕ := 128
  let largest_coeff_term : ℝ := 70 * x^4
  let x_power_7_term : ℝ := -56 * x^7
  (binomial_sum n - binomial_sum 7 = binomial_sum_diff) ∧
  (∀ k, 0 ≤ k ∧ k ≤ n → |(-1)^k * (n.choose k) * x^(2*n - 3*k)| ≤ |largest_coeff_term|) ∧
  ((-1)^3 * (n.choose 3) * x^(2*n - 3*3) = x_power_7_term) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l354_35424


namespace NUMINAMATH_CALUDE_boat_against_stream_distance_l354_35426

/-- The distance a boat travels against the stream in one hour -/
def distance_against_stream (downstream_distance : ℝ) (still_water_speed : ℝ) : ℝ :=
  still_water_speed - (downstream_distance - still_water_speed)

/-- Theorem: Given a boat that travels 13 km downstream in one hour with a still water speed of 9 km/hr,
    the distance it travels against the stream in one hour is 5 km. -/
theorem boat_against_stream_distance :
  distance_against_stream 13 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_against_stream_distance_l354_35426


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l354_35479

/-- Given a line and a circle, prove that if the line is tangent to the circle, then the constant a in the circle equation equals 2 + √5. -/
theorem line_tangent_to_circle (t θ : ℝ) (a : ℝ) (h_a : a > 0) :
  let line : ℝ × ℝ → Prop := λ p => ∃ t, p.1 = 1 - t ∧ p.2 = 2 * t
  let circle : ℝ × ℝ → Prop := λ p => ∃ θ, p.1 = Real.cos θ ∧ p.2 = Real.sin θ + a
  (∀ p, line p → ¬ circle p) ∧ (∃ p, line p ∧ (∀ ε > 0, ∃ q, circle q ∧ dist p q < ε)) →
  a = 2 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l354_35479


namespace NUMINAMATH_CALUDE_counterexample_exists_l354_35417

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l354_35417


namespace NUMINAMATH_CALUDE_inequality_proof_l354_35425

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l354_35425


namespace NUMINAMATH_CALUDE_car_distance_theorem_l354_35484

/-- Calculates the total distance traveled by a car with increasing speed over a given number of hours -/
def totalDistance (initialDistance : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  hours * (2 * initialDistance + (hours - 1) * speedIncrease) / 2

theorem car_distance_theorem :
  totalDistance 55 2 12 = 792 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l354_35484


namespace NUMINAMATH_CALUDE_red_other_side_probability_l354_35455

structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

def total_cards : ℕ := 9
def black_both_sides : ℕ := 4
def black_red : ℕ := 2
def red_both_sides : ℕ := 3

def is_red (side : Bool) : Prop := side = true

theorem red_other_side_probability :
  let cards : List Card := 
    (List.replicate black_both_sides ⟨false, false⟩) ++
    (List.replicate black_red ⟨false, true⟩) ++
    (List.replicate red_both_sides ⟨true, true⟩)
  let total_red_sides := red_both_sides * 2 + black_red
  let red_both_sides_count := red_both_sides * 2
  (red_both_sides_count : ℚ) / total_red_sides = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_red_other_side_probability_l354_35455


namespace NUMINAMATH_CALUDE_same_tribe_adjacent_l354_35446

-- Define the tribes
inductive Tribe
| Human
| Dwarf
| Elf
| Goblin

-- Define a function to check if two tribes can sit next to each other
def canSitTogether (t1 t2 : Tribe) : Prop :=
  match t1, t2 with
  | Tribe.Human, Tribe.Goblin => False
  | Tribe.Goblin, Tribe.Human => False
  | Tribe.Elf, Tribe.Dwarf => False
  | Tribe.Dwarf, Tribe.Elf => False
  | _, _ => True

-- Define the theorem
theorem same_tribe_adjacent
  (arrangement : Fin 1991 → Tribe)
  (round_table : ∀ i j : Fin 1991, canSitTogether (arrangement i) (arrangement j)) :
  ∃ i : Fin 1991, arrangement i = arrangement (i + 1) :=
sorry

end NUMINAMATH_CALUDE_same_tribe_adjacent_l354_35446


namespace NUMINAMATH_CALUDE_matchstick_subtraction_theorem_l354_35405

/-- Represents a collection of matchsticks -/
structure MatchstickSet :=
  (count : ℕ)

/-- Represents a Roman numeral -/
inductive RomanNumeral
  | I
  | V
  | X
  | L
  | C
  | D
  | M

/-- Function to determine if a given number of matchsticks can form a Roman numeral -/
def can_form_roman_numeral (m : MatchstickSet) (r : RomanNumeral) : Prop :=
  match r with
  | RomanNumeral.I => m.count ≥ 1
  | RomanNumeral.V => m.count ≥ 2
  | _ => false  -- For simplicity, we only consider I and V in this problem

/-- The main theorem to prove -/
theorem matchstick_subtraction_theorem :
  ∀ (initial : MatchstickSet),
    initial.count = 10 →
    ∃ (removed : MatchstickSet) (remaining : MatchstickSet),
      removed.count = 7 ∧
      remaining.count = initial.count - removed.count ∧
      can_form_roman_numeral remaining RomanNumeral.I ∧
      can_form_roman_numeral remaining RomanNumeral.V :=
sorry

end NUMINAMATH_CALUDE_matchstick_subtraction_theorem_l354_35405


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l354_35462

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    if a_5 = 2S_4 + 3 and a_6 = 2S_5 + 3, then q = 3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- Definition of sum of geometric sequence
  a 5 = 2 * S 4 + 3 →
  a 6 = 2 * S 5 + 3 →
  q = 3 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l354_35462


namespace NUMINAMATH_CALUDE_total_egg_collection_l354_35419

/-- The number of dozen eggs collected by each person -/
structure EggCollection where
  benjamin : ℚ
  carla : ℚ
  trisha : ℚ
  david : ℚ
  emily : ℚ

/-- The conditions of the egg collection problem -/
def eggCollectionConditions (e : EggCollection) : Prop :=
  e.benjamin = 6 ∧
  e.carla = 3 * e.benjamin ∧
  e.trisha = e.benjamin - 4 ∧
  e.david = 2 * e.trisha ∧
  e.david = e.carla / 2 ∧
  e.emily = 3/4 * e.david ∧
  e.emily = e.trisha + e.trisha / 2

/-- The theorem stating that the total number of dozen eggs collected is 33 -/
theorem total_egg_collection (e : EggCollection) 
  (h : eggCollectionConditions e) : 
  e.benjamin + e.carla + e.trisha + e.david + e.emily = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_egg_collection_l354_35419


namespace NUMINAMATH_CALUDE_pool_depth_relationship_l354_35414

/-- The depth of Sarah's pool in feet -/
def sarahs_pool_depth : ℝ := 5

/-- The depth of John's pool in feet -/
def johns_pool_depth : ℝ := 15

/-- Theorem stating the relationship between John's and Sarah's pool depths -/
theorem pool_depth_relationship : 
  johns_pool_depth = 2 * sarahs_pool_depth + 5 ∧ sarahs_pool_depth = 5 := by
  sorry

end NUMINAMATH_CALUDE_pool_depth_relationship_l354_35414


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l354_35460

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  ∃ (k : ℤ), (n - 1) + n + (n + 1) + (n + 2) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l354_35460


namespace NUMINAMATH_CALUDE_part1_solution_part2_solution_l354_35418

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x + (1-a)

-- Part 1
theorem part1_solution :
  {x : ℝ | f 4 x ≥ 7} = {x : ℝ | x ≥ 5 ∨ x ≤ -2} := by sorry

-- Part 2
theorem part2_solution :
  (∀ x, x > -1 → f a x > 0) → a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_part1_solution_part2_solution_l354_35418


namespace NUMINAMATH_CALUDE_neighbor_dog_ate_five_chickens_l354_35447

/-- The number of chickens eaten by the neighbor's dog -/
def chickens_eaten (initial : ℕ) (final : ℕ) : ℕ :=
  2 * initial + 6 - final

theorem neighbor_dog_ate_five_chickens : chickens_eaten 4 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_dog_ate_five_chickens_l354_35447


namespace NUMINAMATH_CALUDE_inequality_equivalence_l354_35441

theorem inequality_equivalence (x : ℝ) : 
  x * Real.log (x^2 + x + 1) / Real.log (1/10) > 0 ↔ x < -1 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l354_35441


namespace NUMINAMATH_CALUDE_four_digit_integer_proof_l354_35428

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ := ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℤ := (n / 1000 : ℤ) - (n % 10 : ℤ)

theorem four_digit_integer_proof (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : digit_sum n = 17)
  (h3 : middle_digits_sum n = 8)
  (h4 : thousands_minus_units n = 3)
  (h5 : n % 7 = 0) :
  n = 6443 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_integer_proof_l354_35428


namespace NUMINAMATH_CALUDE_max_value_a_l354_35442

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 2 * d)
  (h4 : d < 100) :
  a ≤ 1179 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 1179 ∧
    a' < 2 * b' ∧
    b' < 3 * c' ∧
    c' < 2 * d' ∧
    d' < 100 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l354_35442


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l354_35429

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h1 : a 1 + a 5 = 10)
  (h2 : a 4 = 7)
  (h3 : ∃ d, arithmetic_sequence a d) :
  ∃ d, arithmetic_sequence a d ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l354_35429


namespace NUMINAMATH_CALUDE_solution_x_equals_two_l354_35443

theorem solution_x_equals_two : 
  ∃ x : ℝ, x = 2 ∧ 7 * x - 14 = 0 := by
sorry

end NUMINAMATH_CALUDE_solution_x_equals_two_l354_35443


namespace NUMINAMATH_CALUDE_circle_center_and_radius_prove_center_and_radius_l354_35494

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 4*x = 0

-- Define the center of the circle
def center : ℝ × ℝ := (-2, 0)

-- Define the radius of the circle
def radius : ℝ := 2

-- Theorem statement
theorem circle_center_and_radius :
  ∀ (x y : ℝ), circle_equation x y ↔ (x + 2)^2 + y^2 = 4 :=
by sorry

-- Prove that the center and radius are correct
theorem prove_center_and_radius :
  (∀ (x y : ℝ), circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_prove_center_and_radius_l354_35494


namespace NUMINAMATH_CALUDE_good_numbers_l354_35415

def isGoodNumber (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), ∀ k : Fin n, ∃ m : ℕ, k.val + 1 + a k = m^2

theorem good_numbers :
  isGoodNumber 13 ∧
  isGoodNumber 15 ∧
  isGoodNumber 17 ∧
  isGoodNumber 19 ∧
  ¬isGoodNumber 11 := by sorry

end NUMINAMATH_CALUDE_good_numbers_l354_35415


namespace NUMINAMATH_CALUDE_simplify_expression_l354_35485

theorem simplify_expression (n : ℕ) :
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+4)) = 29 / 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l354_35485


namespace NUMINAMATH_CALUDE_crosswalk_height_l354_35478

/-- Represents a parallelogram with given dimensions -/
structure Parallelogram where
  side1 : ℝ  -- Length of one side
  side2 : ℝ  -- Length of adjacent side
  base : ℝ   -- Length of base parallel to side1
  height1 : ℝ -- Height perpendicular to side1
  height2 : ℝ -- Height perpendicular to side2

/-- The area of a parallelogram can be calculated two ways -/
axiom area_equality (p : Parallelogram) : p.side1 * p.height1 = p.side2 * p.height2

/-- Theorem stating the height of the parallelogram perpendicular to the 80-foot side -/
theorem crosswalk_height (p : Parallelogram) 
    (h1 : p.side1 = 60)
    (h2 : p.side2 = 80)
    (h3 : p.base = 30)
    (h4 : p.height1 = 60) :
    p.height2 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_crosswalk_height_l354_35478


namespace NUMINAMATH_CALUDE_min_cards_for_even_product_l354_35430

def is_even (n : Nat) : Bool := n % 2 = 0

theorem min_cards_for_even_product :
  ∀ (S : Finset Nat),
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 16) →
  (Finset.card S = 16) →
  (∃ (T : Finset Nat), T ⊆ S ∧ Finset.card T = 9 ∧ ∃ n ∈ T, is_even n) ∧
  (∀ (U : Finset Nat), U ⊆ S → Finset.card U < 9 → ∀ n ∈ U, ¬is_even n) :=
by sorry

end NUMINAMATH_CALUDE_min_cards_for_even_product_l354_35430


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l354_35457

theorem tan_theta_minus_pi_fourth (θ : Real) : 
  (∃ (x y : Real), x = 2 ∧ y = 3 ∧ Real.tan θ = y / x) → 
  Real.tan (θ - Real.pi / 4) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l354_35457


namespace NUMINAMATH_CALUDE_square_land_area_l354_35467

/-- The area of a square land plot with side length 32 units is 1024 square units. -/
theorem square_land_area (side_length : ℝ) (h : side_length = 32) : 
  side_length * side_length = 1024 := by
  sorry

end NUMINAMATH_CALUDE_square_land_area_l354_35467


namespace NUMINAMATH_CALUDE_sleeves_weight_addition_l354_35473

theorem sleeves_weight_addition (raw_squat : ℝ) (wrap_percentage : ℝ) (wrap_sleeve_difference : ℝ) 
  (h1 : raw_squat = 600)
  (h2 : wrap_percentage = 0.25)
  (h3 : wrap_sleeve_difference = 120) :
  let squat_with_wraps := raw_squat + wrap_percentage * raw_squat
  let squat_with_sleeves := squat_with_wraps - wrap_sleeve_difference
  squat_with_sleeves - raw_squat = 30 := by
sorry

end NUMINAMATH_CALUDE_sleeves_weight_addition_l354_35473


namespace NUMINAMATH_CALUDE_min_value_of_function_l354_35433

theorem min_value_of_function (x y : ℝ) : 2*x^2 + 3*y^2 + 8*x - 6*y + 5*x*y + 36 ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l354_35433


namespace NUMINAMATH_CALUDE_percentage_decrease_in_z_l354_35432

/-- Given positive real numbers x and z, and a real number q, if x and (z+10) are inversely 
    proportional, and x increases by q%, then the percentage decrease in z is q(z+10)/(100+q)% -/
theorem percentage_decrease_in_z (x z q : ℝ) (hx : x > 0) (hz : z > 0) (hq : q ≠ -100) :
  (∃ k : ℝ, k > 0 ∧ x * (z + 10) = k) →
  let x' := x * (1 + q / 100)
  let z' := (100 / (100 + q)) * (z + 10) - 10
  (z - z') / z * 100 = q * (z + 10) / (100 + q) := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_in_z_l354_35432


namespace NUMINAMATH_CALUDE_simplify_power_expression_l354_35493

theorem simplify_power_expression (x : ℝ) : (3 * (2 * x)^5)^4 = 84934656 * x^20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l354_35493


namespace NUMINAMATH_CALUDE_fundraising_problem_l354_35402

/-- Represents a student with their fundraising goal -/
structure Student where
  name : String
  goal : ℕ

/-- Represents a day's fundraising activity -/
structure FundraisingDay where
  income : ℕ
  expense : ℕ

/-- The fundraising problem -/
theorem fundraising_problem 
  (students : List Student)
  (collective_goal : ℕ)
  (fundraising_days : List FundraisingDay)
  (h1 : students.length = 8)
  (h2 : collective_goal = 3500)
  (h3 : fundraising_days.length = 5)
  (h4 : students.map Student.goal = [350, 450, 500, 550, 600, 650, 450, 550])
  (h5 : fundraising_days.map FundraisingDay.income = [800, 950, 500, 700, 550])
  (h6 : fundraising_days.map FundraisingDay.expense = [100, 150, 50, 75, 100]) :
  (students.map Student.goal = [350, 450, 500, 550, 600, 650, 450, 550]) ∧
  ((fundraising_days.map (λ d => d.income - d.expense)).sum + 3975 = collective_goal + (students.map Student.goal).sum) :=
sorry

end NUMINAMATH_CALUDE_fundraising_problem_l354_35402


namespace NUMINAMATH_CALUDE_hostel_mess_expenditure_l354_35449

/-- The original daily expenditure of a hostel mess given certain conditions -/
theorem hostel_mess_expenditure 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (expense_increase : ℕ) 
  (avg_expense_decrease : ℕ) 
  (h1 : initial_students = 35)
  (h2 : new_students = 7)
  (h3 : expense_increase = 42)
  (h4 : avg_expense_decrease = 1) : 
  ∃ (original_expenditure : ℕ), original_expenditure = 420 :=
by sorry

end NUMINAMATH_CALUDE_hostel_mess_expenditure_l354_35449


namespace NUMINAMATH_CALUDE_range_of_a_l354_35422

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| > 2 → |x| > a) ∧ 
  (∃ x : ℝ, |x| > a ∧ |x + 1| ≤ 2) → 
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l354_35422


namespace NUMINAMATH_CALUDE_f_4_equals_40_l354_35409

-- Define the function f
def f (x : ℝ) (a b c : ℝ) : ℝ := 2 * a * x + b * x + c

-- State the theorem
theorem f_4_equals_40 
  (h1 : f 1 a b c = 10) 
  (h2 : f 2 a b c = 20) : 
  f 4 a b c = 40 := by
  sorry

end NUMINAMATH_CALUDE_f_4_equals_40_l354_35409


namespace NUMINAMATH_CALUDE_zaras_goats_l354_35403

theorem zaras_goats (cows sheep : ℕ) (groups : ℕ) (animals_per_group : ℕ) (goats : ℕ) : 
  cows = 24 → 
  sheep = 7 → 
  groups = 3 → 
  animals_per_group = 48 → 
  goats = groups * animals_per_group - (cows + sheep) → 
  goats = 113 := by
  sorry

end NUMINAMATH_CALUDE_zaras_goats_l354_35403


namespace NUMINAMATH_CALUDE_equation_proof_l354_35458

theorem equation_proof : 
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l354_35458


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l354_35486

theorem arctan_tan_difference (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) :
  Real.arctan (Real.tan x - 2 * Real.tan y) = 25 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l354_35486


namespace NUMINAMATH_CALUDE_card_selection_two_suits_l354_35477

theorem card_selection_two_suits (deck_size : ℕ) (suits : ℕ) (cards_per_suit : ℕ) 
  (selection_size : ℕ) (h1 : deck_size = suits * cards_per_suit) 
  (h2 : suits = 4) (h3 : cards_per_suit = 13) (h4 : selection_size = 3) : 
  (suits.choose 2) * (cards_per_suit.choose 2 * cards_per_suit.choose 1 + 
   cards_per_suit.choose 1 * cards_per_suit.choose 2) = 12168 :=
by sorry

end NUMINAMATH_CALUDE_card_selection_two_suits_l354_35477


namespace NUMINAMATH_CALUDE_teachers_students_arrangement_l354_35488

def num_students : ℕ := 4
def num_teachers : ℕ := 3

-- Function to calculate permutations
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

-- Theorem statement
theorem teachers_students_arrangement :
  permutations num_teachers num_teachers * permutations num_students num_students = 144 :=
by sorry

end NUMINAMATH_CALUDE_teachers_students_arrangement_l354_35488


namespace NUMINAMATH_CALUDE_polynomial_not_factorable_l354_35472

theorem polynomial_not_factorable : ¬ ∃ (a b c d : ℤ),
  ∀ (x : ℝ), x^4 + 3*x^3 + 6*x^2 + 9*x + 12 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_factorable_l354_35472


namespace NUMINAMATH_CALUDE_unique_solution_l354_35420

def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - p.2 = 1 ∧ p.1 + 4 * p.2 = 5}

theorem unique_solution : solution_set = {(1, 1)} := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l354_35420


namespace NUMINAMATH_CALUDE_xiao_ming_correct_count_l354_35413

/-- Represents a math problem with a given answer --/
structure MathProblem where
  given_answer : Int
  correct_answer : Int

/-- Checks if a math problem is answered correctly --/
def is_correct (problem : MathProblem) : Bool :=
  problem.given_answer = problem.correct_answer

/-- Counts the number of correctly answered problems --/
def count_correct (problems : List MathProblem) : Nat :=
  (problems.filter is_correct).length

/-- The list of math problems Xiao Ming solved --/
def xiao_ming_problems : List MathProblem := [
  { given_answer := 0, correct_answer := -4 },
  { given_answer := -4, correct_answer := 0 },
  { given_answer := -4, correct_answer := -4 }
]

theorem xiao_ming_correct_count :
  count_correct xiao_ming_problems = 1 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_correct_count_l354_35413


namespace NUMINAMATH_CALUDE_tile_count_theorem_l354_35469

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledRectangle where
  length : ℕ
  width : ℕ
  diagonal_tiles : ℕ

/-- The condition that one side is twice the length of the other. -/
def double_side (rect : TiledRectangle) : Prop :=
  rect.length = 2 * rect.width

/-- The number of tiles on the diagonals. -/
def diagonal_count (rect : TiledRectangle) : ℕ :=
  rect.diagonal_tiles

/-- The total number of tiles covering the floor. -/
def total_tiles (rect : TiledRectangle) : ℕ :=
  rect.length * rect.width

/-- The main theorem stating the problem. -/
theorem tile_count_theorem (rect : TiledRectangle) :
  double_side rect → diagonal_count rect = 49 → total_tiles rect = 50 := by
  sorry


end NUMINAMATH_CALUDE_tile_count_theorem_l354_35469


namespace NUMINAMATH_CALUDE_abc_value_l354_35456

theorem abc_value (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 39) 
  (h3 : a + b + c = 10) : 
  a * b * c = -150 + 15 * Real.sqrt 69 := by
  sorry

end NUMINAMATH_CALUDE_abc_value_l354_35456


namespace NUMINAMATH_CALUDE_probability_highest_is_four_value_l354_35436

def number_of_balls : ℕ := 5
def balls_drawn : ℕ := 3

def probability_highest_is_four : ℚ :=
  (Nat.choose (number_of_balls - 2) (balls_drawn - 1)) / (Nat.choose number_of_balls balls_drawn)

theorem probability_highest_is_four_value : 
  probability_highest_is_four = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_highest_is_four_value_l354_35436


namespace NUMINAMATH_CALUDE_age_ratio_problem_l354_35482

theorem age_ratio_problem (ali_age yusaf_age umar_age : ℕ) : 
  ali_age = 8 →
  ali_age = yusaf_age + 3 →
  ∃ k : ℕ, umar_age = k * yusaf_age →
  umar_age = 10 →
  umar_age / yusaf_age = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l354_35482
