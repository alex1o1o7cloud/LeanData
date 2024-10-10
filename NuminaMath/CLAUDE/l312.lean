import Mathlib

namespace triangle_reconstruction_uniqueness_l312_31252

/-- A triangle in a 2D plane --/
structure Triangle :=
  (A B C : EuclideanSpace ℝ (Fin 2))

/-- The circumcenter of a triangle --/
def circumcenter (t : Triangle) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- Reflect a point across a line defined by two points --/
def reflect_point (p q r : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- Given three points that are reflections of a fourth point across the sides of a triangle,
    reconstruct the original triangle --/
def reconstruct_triangle (O1 O2 O3 : EuclideanSpace ℝ (Fin 2)) : Triangle :=
  sorry

theorem triangle_reconstruction_uniqueness 
  (t : Triangle) 
  (O : EuclideanSpace ℝ (Fin 2)) 
  (hO : O = circumcenter t) 
  (O1 : EuclideanSpace ℝ (Fin 2)) 
  (hO1 : O1 = reflect_point O t.B t.C) 
  (O2 : EuclideanSpace ℝ (Fin 2)) 
  (hO2 : O2 = reflect_point O t.C t.A) 
  (O3 : EuclideanSpace ℝ (Fin 2)) 
  (hO3 : O3 = reflect_point O t.A t.B) :
  reconstruct_triangle O1 O2 O3 = t :=
sorry

end triangle_reconstruction_uniqueness_l312_31252


namespace parabola_shift_l312_31231

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift amount
def shift : ℝ := 1

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x + shift)^2

-- Theorem statement
theorem parabola_shift :
  ∀ x y : ℝ, y = original_parabola (x + shift) ↔ y = shifted_parabola x :=
by sorry

end parabola_shift_l312_31231


namespace problem_solution_l312_31256

theorem problem_solution (t s x : ℝ) : 
  t = 15 * s^2 → t = 3.75 → x = s / 2 → s = 0.5 ∧ x = 0.25 := by
  sorry

end problem_solution_l312_31256


namespace problem_solution_l312_31235

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + x^3 / y^2 + y^3 / x^2 + y = 590 + 5/9 := by
sorry

end problem_solution_l312_31235


namespace first_player_advantage_l312_31245

/-- Represents a chocolate bar game state -/
structure ChocolateBar :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- The result of a game -/
structure GameResult :=
  (first_player_pieces : ℕ)
  (second_player_pieces : ℕ)

/-- A strategy for playing the game -/
def Strategy := ChocolateBar → Player → ChocolateBar

/-- Play the game with a given strategy -/
def play_game (initial : ChocolateBar) (strategy : Strategy) : GameResult :=
  sorry

/-- The optimal strategy for the first player -/
def optimal_strategy : Strategy :=
  sorry

/-- Theorem stating that the first player can get at least 6 more pieces -/
theorem first_player_advantage (initial : ChocolateBar) :
  initial.rows = 9 ∧ initial.cols = 6 →
  let result := play_game initial optimal_strategy
  result.first_player_pieces ≥ result.second_player_pieces + 6 :=
by sorry

end first_player_advantage_l312_31245


namespace smallest_solution_of_equation_l312_31274

theorem smallest_solution_of_equation (x : ℝ) :
  x^4 - 54*x^2 + 441 = 0 → x ≥ -Real.sqrt 33 :=
by sorry

end smallest_solution_of_equation_l312_31274


namespace words_with_vowels_count_l312_31268

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I'}
def vowels : Finset Char := {'A', 'E', 'I'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := consonants.card ^ word_length

theorem words_with_vowels_count :
  total_words - words_without_vowels = 29643 :=
sorry

end words_with_vowels_count_l312_31268


namespace fraction_unchanged_l312_31248

theorem fraction_unchanged (x y : ℝ) : 
  (2*x) * (2*y) / ((2*x)^2 - (2*y)^2) = x * y / (x^2 - y^2) :=
by sorry

end fraction_unchanged_l312_31248


namespace parallel_iff_a_eq_neg_one_l312_31285

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a*y - 2 = 0

-- Define parallelism for these lines
def parallel (a : ℝ) : Prop := ∀ x y, l₁ x y ↔ ∃ k, l₂ a (x + k) (y + k)

-- State the theorem
theorem parallel_iff_a_eq_neg_one :
  ∀ a : ℝ, parallel a ↔ a = -1 := by sorry

end parallel_iff_a_eq_neg_one_l312_31285


namespace curve_is_two_lines_l312_31234

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop := x^2 - x*y - 2*y^2 = 0

/-- The curve represents two straight lines -/
theorem curve_is_two_lines : 
  ∃ (a b c d : ℝ), ∀ (x y : ℝ), 
    curve_equation x y ↔ (a*x + b*y = 0 ∧ c*x + d*y = 0) :=
sorry

end curve_is_two_lines_l312_31234


namespace f_decreasing_implies_a_range_l312_31237

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x - a * x^2 else a^x

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/2) 1 ∧ a ≠ 1 :=
sorry

end f_decreasing_implies_a_range_l312_31237


namespace harold_bought_three_doughnuts_l312_31283

def harold_doughnuts (harold_coffee : ℕ) (harold_total : ℚ) 
  (melinda_doughnuts : ℕ) (melinda_coffee : ℕ) (melinda_total : ℚ) 
  (doughnut_price : ℚ) : Prop :=
  ∃ (coffee_price : ℚ),
    (doughnut_price * 3 + coffee_price * harold_coffee = harold_total) ∧
    (doughnut_price * melinda_doughnuts + coffee_price * melinda_coffee = melinda_total)

theorem harold_bought_three_doughnuts :
  harold_doughnuts 4 4.91 5 6 7.59 0.45 :=
sorry

end harold_bought_three_doughnuts_l312_31283


namespace train_problem_solution_l312_31209

/-- Represents the day of the week -/
inductive Day
  | Saturday
  | Monday

/-- Represents a date in a month -/
structure Date where
  day : Day
  number : Nat

/-- Represents a train car -/
structure TrainCar where
  number : Nat
  seat : Nat

/-- The problem setup -/
def TrainProblem (d1 d2 : Date) (car : TrainCar) : Prop :=
  d1.day = Day.Saturday ∧
  d2.day = Day.Monday ∧
  d2.number = car.number ∧
  car.seat < car.number ∧
  d1.number > car.number ∧
  d1.number ≠ d2.number ∧
  car.number < 10

theorem train_problem_solution :
  ∀ (d1 d2 : Date) (car : TrainCar),
    TrainProblem d1 d2 car →
    car.number = 2 ∧ car.seat = 1 :=
by
  sorry


end train_problem_solution_l312_31209


namespace inequality_system_solution_l312_31208

theorem inequality_system_solution (x : ℝ) :
  (2 * x + 1 > x + 3 ∧ 2 * x - 4 < x) ↔ (2 < x ∧ x < 4) := by
  sorry

end inequality_system_solution_l312_31208


namespace distance_to_reflection_over_y_axis_l312_31243

/-- Given a point P at (3, 5), prove that the distance between P and its reflection over the y-axis is 6 -/
theorem distance_to_reflection_over_y_axis :
  let P : ℝ × ℝ := (3, 5)
  let P' : ℝ × ℝ := (-P.1, P.2)
  Real.sqrt ((P'.1 - P.1)^2 + (P'.2 - P.2)^2) = 6 := by sorry

end distance_to_reflection_over_y_axis_l312_31243


namespace collinear_points_theorem_l312_31244

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_collinear (v w : V) : Prop := ∃ (r : ℝ), v = r • w

theorem collinear_points_theorem 
  (e₁ e₂ : V) 
  (h_noncollinear : ¬ is_collinear e₁ e₂)
  (k : ℝ)
  (AB CB CD : V)
  (h_AB : AB = e₁ - k • e₂)
  (h_CB : CB = 2 • e₁ + e₂)
  (h_CD : CD = 3 • e₁ - e₂)
  (h_collinear : is_collinear AB (CD - CB)) :
  k = 2 := by sorry

end collinear_points_theorem_l312_31244


namespace school_average_age_l312_31212

theorem school_average_age (total_students : ℕ) (boys_avg_age girls_avg_age : ℚ) (num_girls : ℕ) :
  total_students = 632 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  num_girls = 158 →
  let num_boys := total_students - num_girls
  let total_age := boys_avg_age * num_boys + girls_avg_age * num_girls
  total_age / total_students = 11.75 := by
  sorry

end school_average_age_l312_31212


namespace rhombus_existence_and_uniqueness_l312_31214

/-- Represents a rhombus -/
structure Rhombus where
  side : ℝ
  diag1 : ℝ
  diag2 : ℝ
  angle : ℝ

/-- Given the sum of diagonals and an opposite angle, a unique rhombus can be determined -/
theorem rhombus_existence_and_uniqueness 
  (diag_sum : ℝ) 
  (opp_angle : ℝ) 
  (h_pos : diag_sum > 0) 
  (h_angle : 0 < opp_angle ∧ opp_angle < π) :
  ∃! r : Rhombus, r.diag1 + r.diag2 = diag_sum ∧ r.angle = opp_angle :=
sorry

end rhombus_existence_and_uniqueness_l312_31214


namespace sum_of_abc_l312_31269

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30) (hac : a * c = 60) (hbc : b * c = 90) :
  a + b + c = 11 * Real.sqrt 5 := by
  sorry

end sum_of_abc_l312_31269


namespace pairing_possibility_l312_31296

/-- Represents a pairing of children -/
structure Pairing :=
  (boys : ℕ)   -- Number of boy-boy pairs
  (girls : ℕ)  -- Number of girl-girl pairs
  (mixed : ℕ)  -- Number of boy-girl pairs

/-- Represents a group of children that can be arranged in different pairings -/
structure ChildrenGroup :=
  (to_museum : Pairing)
  (from_museum : Pairing)
  (total_boys : ℕ)
  (total_girls : ℕ)

/-- The theorem to be proved -/
theorem pairing_possibility (group : ChildrenGroup) 
  (h1 : group.to_museum.boys = 3 * group.to_museum.girls)
  (h2 : group.from_museum.boys = 4 * group.from_museum.girls)
  (h3 : group.total_boys = 2 * group.to_museum.boys + group.to_museum.mixed)
  (h4 : group.total_girls = 2 * group.to_museum.girls + group.to_museum.mixed)
  (h5 : group.total_boys = 2 * group.from_museum.boys + group.from_museum.mixed)
  (h6 : group.total_girls = 2 * group.from_museum.girls + group.from_museum.mixed) :
  ∃ (new_pairing : Pairing), 
    new_pairing.boys = 7 * new_pairing.girls ∧ 
    2 * new_pairing.boys + 2 * new_pairing.girls + new_pairing.mixed = group.total_boys + group.total_girls :=
sorry

end pairing_possibility_l312_31296


namespace solution_when_m_3_no_solution_conditions_l312_31225

-- Define the fractional equation
def fractional_equation (x m : ℝ) : Prop :=
  (3 - 2*x) / (x - 2) - (m*x - 2) / (2 - x) = -1

-- Theorem 1: When m = 3, the solution is x = 1/2
theorem solution_when_m_3 :
  ∃ x : ℝ, fractional_equation x 3 ∧ x = 1/2 :=
sorry

-- Theorem 2: The equation has no solution when m = 1 or m = 3/2
theorem no_solution_conditions :
  (∀ x : ℝ, ¬ fractional_equation x 1) ∧
  (∀ x : ℝ, ¬ fractional_equation x (3/2)) :=
sorry

end solution_when_m_3_no_solution_conditions_l312_31225


namespace square_difference_2019_l312_31272

theorem square_difference_2019 (a b : ℕ) (h1 : 2019 = a^2 - b^2) (h2 : a < 1000) : a = 338 := by
  sorry

end square_difference_2019_l312_31272


namespace cupcakes_frosted_in_ten_minutes_l312_31265

def mark_rate : ℚ := 1 / 15
def julia_rate : ℚ := 1 / 40
def total_time : ℚ := 10 * 60  -- 10 minutes in seconds

theorem cupcakes_frosted_in_ten_minutes : 
  ⌊(mark_rate + julia_rate) * total_time⌋ = 55 := by sorry

end cupcakes_frosted_in_ten_minutes_l312_31265


namespace range_of_a_l312_31287

-- Define the function f(x) = x^2 - 2x + a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

-- State the theorem
theorem range_of_a (h : ∀ x ∈ Set.Icc 2 3, f a x > 0) : a > 0 := by
  sorry


end range_of_a_l312_31287


namespace action_figure_cost_l312_31259

theorem action_figure_cost (current : ℕ) (total : ℕ) (cost : ℕ) : 
  current = 7 → total = 16 → cost = 72 → 
  (cost : ℚ) / ((total : ℚ) - (current : ℚ)) = 8 := by sorry

end action_figure_cost_l312_31259


namespace smallest_n_for_sqrt_18n_integer_l312_31240

theorem smallest_n_for_sqrt_18n_integer (n : ℕ) : 
  (∀ k : ℕ, 0 < k → k < 2 → ¬ ∃ m : ℕ, m^2 = 18 * k) ∧ 
  (∃ m : ℕ, m^2 = 18 * 2) → 
  n = 2 → 
  (∃ m : ℕ, m^2 = 18 * n) ∧ 
  (∀ k : ℕ, 0 < k → k < n → ¬ ∃ m : ℕ, m^2 = 18 * k) := by
  sorry

end smallest_n_for_sqrt_18n_integer_l312_31240


namespace decimal_to_octal_conversion_l312_31250

/-- Converts a natural number to its octal representation -/
def toOctal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: toOctal (n / 8)

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 521

/-- The expected octal representation -/
def expectedOctal : List ℕ := [1, 1, 0, 1]

theorem decimal_to_octal_conversion :
  toOctal decimalNumber = expectedOctal := by
  sorry

end decimal_to_octal_conversion_l312_31250


namespace tan_alpha_2_implies_fraction_equals_3_5_l312_31254

theorem tan_alpha_2_implies_fraction_equals_3_5 (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 3/5 := by
  sorry

end tan_alpha_2_implies_fraction_equals_3_5_l312_31254


namespace square_root_problem_l312_31255

theorem square_root_problem (x y : ℝ) 
  (h1 : (x - 1) = 9) 
  (h2 : (2 * x + y + 7) = 8) : 
  (7 - x - y) = 16 := by
  sorry

end square_root_problem_l312_31255


namespace systematic_sampling_theorem_l312_31264

def population_size : Nat := 1000
def num_groups : Nat := 10
def sample_size : Nat := 10

def systematic_sample (x : Nat) : List Nat :=
  List.range num_groups |>.map (fun k => (x + 33 * k) % 100)

def last_two_digits (n : Nat) : Nat := n % 100

theorem systematic_sampling_theorem :
  (systematic_sample 24 = [24, 57, 90, 23, 56, 89, 22, 55, 88, 21]) ∧
  (∀ x : Nat, x < population_size →
    (∃ n ∈ systematic_sample x, last_two_digits n = 87) →
    x ∈ [21, 22, 23, 54, 55, 56, 87, 88, 89, 90]) :=
by sorry

end systematic_sampling_theorem_l312_31264


namespace age_difference_l312_31227

/-- Given three people a, b, and c, prove that a is 2 years older than b -/
theorem age_difference (a b c : ℕ) : 
  b = 2 * c →
  a + b + c = 22 →
  b = 8 →
  a - b = 2 :=
by sorry

end age_difference_l312_31227


namespace ryan_overall_score_l312_31271

def first_test_questions : ℕ := 30
def first_test_score : ℚ := 85 / 100

def second_test_math_questions : ℕ := 20
def second_test_math_score : ℚ := 95 / 100
def second_test_science_questions : ℕ := 15
def second_test_science_score : ℚ := 80 / 100

def third_test_questions : ℕ := 15
def third_test_score : ℚ := 65 / 100

theorem ryan_overall_score :
  let total_questions := first_test_questions + second_test_math_questions + second_test_science_questions + third_test_questions
  let correct_answers := (first_test_questions : ℚ) * first_test_score +
                         (second_test_math_questions : ℚ) * second_test_math_score +
                         (second_test_science_questions : ℚ) * second_test_science_score +
                         (third_test_questions : ℚ) * third_test_score
  correct_answers / (total_questions : ℚ) = 8281 / 10000 :=
by sorry

end ryan_overall_score_l312_31271


namespace sequence_sum_property_l312_31258

/-- Given a positive sequence {a_n}, prove that a_n = 2n - 1 for all positive integers n,
    where S_n = (a_n + 1)^2 / 4 is the sum of the first n terms. -/
theorem sequence_sum_property (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, S n = (a n + 1)^2 / 4) →
  ∀ n, a n = 2 * n - 1 := by
sorry

end sequence_sum_property_l312_31258


namespace seventh_term_is_29_3_l312_31207

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℚ
  -- Common difference
  d : ℚ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Sixth term is 8
  sixth_term : a + 5*d = 8

/-- The seventh term of the arithmetic sequence is 29/3 -/
theorem seventh_term_is_29_3 (seq : ArithmeticSequence) : seq.a + 6*seq.d = 29/3 := by
  sorry


end seventh_term_is_29_3_l312_31207


namespace no_linear_factor_l312_31247

/-- The polynomial p(x,y,z) = x^2-y^2+2yz-z^2+2x-y-3z has no linear factor with integer coefficients. -/
theorem no_linear_factor (x y z : ℤ) : 
  ¬ ∃ (a b c d : ℤ), (a*x + b*y + c*z + d) ∣ (x^2 - y^2 + 2*y*z - z^2 + 2*x - y - 3*z) :=
sorry

end no_linear_factor_l312_31247


namespace quadratic_always_nonnegative_implies_a_in_range_l312_31253

theorem quadratic_always_nonnegative_implies_a_in_range (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + a ≥ 0) → a ∈ Set.Icc 0 4 := by sorry

end quadratic_always_nonnegative_implies_a_in_range_l312_31253


namespace rectangular_solid_diagonal_l312_31267

theorem rectangular_solid_diagonal 
  (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 24) 
  (h2 : a + b + c = 6) : 
  Real.sqrt (a^2 + b^2 + c^2) = 2 * Real.sqrt 3 := by
sorry

end rectangular_solid_diagonal_l312_31267


namespace cone_base_radius_l312_31224

theorem cone_base_radius (S : ℝ) (r : ℝ) : 
  S = 9 * Real.pi → -- Surface area is 9π cm²
  S = 3 * Real.pi * r^2 → -- Surface area formula for a cone with semicircular lateral surface
  r = Real.sqrt 3 := by
sorry

end cone_base_radius_l312_31224


namespace max_vector_sum_on_circle_l312_31216

def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

def point_on_circle (p : ℝ × ℝ) : Prop := circle_C p.1 p.2

theorem max_vector_sum_on_circle (A B : ℝ × ℝ) :
  point_on_circle A →
  point_on_circle B →
  ‖(A.1 - B.1, A.2 - B.2)‖ = 2 * Real.sqrt 3 →
  ∃ (max : ℝ), max = 8 ∧ ∀ (A' B' : ℝ × ℝ),
    point_on_circle A' →
    point_on_circle B' →
    ‖(A'.1 - B'.1, A'.2 - B'.2)‖ = 2 * Real.sqrt 3 →
    ‖(A'.1 + B'.1, A'.2 + B'.2)‖ ≤ max :=
by sorry

end max_vector_sum_on_circle_l312_31216


namespace problem_solution_l312_31266

theorem problem_solution :
  (12 / 60 = 0.2) ∧
  (0.2 = 4 / 20) ∧
  (0.2 = 20 / 100) := by
sorry

end problem_solution_l312_31266


namespace nested_sqrt_evaluation_l312_31290

theorem nested_sqrt_evaluation :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by sorry

end nested_sqrt_evaluation_l312_31290


namespace probability_specific_selection_l312_31238

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 3

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 6

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 7

/-- The total number of articles of clothing in the drawer -/
def total_articles : ℕ := num_shirts + num_shorts + num_socks

/-- The number of articles to be selected -/
def num_selected : ℕ := 4

/-- The probability of selecting one shirt, two pairs of shorts, and one pair of socks -/
theorem probability_specific_selection :
  (Nat.choose num_shirts 1 * Nat.choose num_shorts 2 * Nat.choose num_socks 1) /
  (Nat.choose total_articles num_selected) = 63 / 364 :=
sorry

end probability_specific_selection_l312_31238


namespace car_stop_once_probability_l312_31262

/-- The probability of a car stopping once at three traffic lights. -/
theorem car_stop_once_probability
  (pA pB pC : ℝ)
  (hA : pA = 1/3)
  (hB : pB = 1/2)
  (hC : pC = 2/3)
  : (1 - pA) * pB * pC + pA * (1 - pB) * pC + pA * pB * (1 - pC) = 7/18 := by
  sorry

end car_stop_once_probability_l312_31262


namespace existence_of_three_numbers_with_same_product_last_digit_l312_31282

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem existence_of_three_numbers_with_same_product_last_digit :
  ∃ (a b c : ℕ), 
    (lastDigit a ≠ lastDigit b) ∧ 
    (lastDigit b ≠ lastDigit c) ∧ 
    (lastDigit a ≠ lastDigit c) ∧
    (lastDigit (a * b) = lastDigit (b * c)) ∧
    (lastDigit (b * c) = lastDigit (a * c)) :=
by
  -- The proof would go here
  sorry

end existence_of_three_numbers_with_same_product_last_digit_l312_31282


namespace difference_10_6_l312_31210

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  prop1 : a 5 * a 7 = 6
  prop2 : a 2 + a 10 = 5

/-- The difference between the 10th and 6th terms is either 2 or -2 -/
theorem difference_10_6 (seq : ArithmeticSequence) : 
  seq.a 10 - seq.a 6 = 2 ∨ seq.a 10 - seq.a 6 = -2 := by
  sorry

end difference_10_6_l312_31210


namespace complex_square_eq_neg100_minus_64i_l312_31277

theorem complex_square_eq_neg100_minus_64i (z : ℂ) :
  z^2 = -100 - 64*I ↔ z = 4 - 8*I ∨ z = -4 + 8*I := by
  sorry

end complex_square_eq_neg100_minus_64i_l312_31277


namespace prob_at_least_one_expired_l312_31275

def total_bottles : ℕ := 10
def expired_bottles : ℕ := 3
def selected_bottles : ℕ := 3

def probability_at_least_one_expired : ℚ := 17/24

theorem prob_at_least_one_expired :
  (1 : ℚ) - (Nat.choose (total_bottles - expired_bottles) selected_bottles : ℚ) / 
  (Nat.choose total_bottles selected_bottles : ℚ) = probability_at_least_one_expired := by
  sorry

end prob_at_least_one_expired_l312_31275


namespace apples_left_proof_l312_31299

/-- The number of apples left when the farmer's children got home -/
def apples_left (num_children : ℕ) (apples_per_child : ℕ) (children_who_ate : ℕ) 
  (apples_eaten_per_child : ℕ) (apples_sold : ℕ) : ℕ :=
  num_children * apples_per_child - (children_who_ate * apples_eaten_per_child + apples_sold)

/-- Theorem stating the number of apples left when the farmer's children got home -/
theorem apples_left_proof : 
  apples_left 5 15 2 4 7 = 60 := by
  sorry

end apples_left_proof_l312_31299


namespace billy_wednesday_apples_l312_31241

/-- The number of apples Billy ate in a week -/
def total_apples : ℕ := 20

/-- The number of apples Billy ate on Monday -/
def monday_apples : ℕ := 2

/-- The number of apples Billy ate on Tuesday -/
def tuesday_apples : ℕ := 2 * monday_apples

/-- The number of apples Billy ate on Friday -/
def friday_apples : ℕ := monday_apples / 2

/-- The number of apples Billy ate on Thursday -/
def thursday_apples : ℕ := 4 * friday_apples

/-- The number of apples Billy ate on Wednesday -/
def wednesday_apples : ℕ := total_apples - (monday_apples + tuesday_apples + thursday_apples + friday_apples)

theorem billy_wednesday_apples :
  wednesday_apples = 9 := by sorry

end billy_wednesday_apples_l312_31241


namespace museum_ticket_cost_l312_31222

def regular_ticket_cost : ℝ := 10

theorem museum_ticket_cost :
  let discounted_ticket := 0.7 * regular_ticket_cost
  let full_price_ticket := regular_ticket_cost
  let total_spent := 44
  2 * discounted_ticket + 3 * full_price_ticket = total_spent :=
by sorry

end museum_ticket_cost_l312_31222


namespace value_of_a_l312_31279

theorem value_of_a (a b c : ℤ) 
  (h1 : a + b = c) 
  (h2 : b + c = 7) 
  (h3 : c = 5) : 
  a = 3 := by
sorry

end value_of_a_l312_31279


namespace correct_remaining_money_l312_31228

def remaining_money (olivia_initial : ℕ) (nigel_initial : ℕ) (num_passes : ℕ) (cost_per_pass : ℕ) : ℕ :=
  olivia_initial + nigel_initial - num_passes * cost_per_pass

theorem correct_remaining_money :
  remaining_money 112 139 6 28 = 83 := by
  sorry

end correct_remaining_money_l312_31228


namespace two_different_expressions_equal_seven_l312_31292

/-- An arithmetic expression using digits of 4 and basic operations -/
inductive Expr
  | four : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression -/
def eval : Expr → ℚ
  | Expr.four => 4
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of 4's used in an expression -/
def count_fours : Expr → ℕ
  | Expr.four => 1
  | Expr.add e1 e2 => count_fours e1 + count_fours e2
  | Expr.sub e1 e2 => count_fours e1 + count_fours e2
  | Expr.mul e1 e2 => count_fours e1 + count_fours e2
  | Expr.div e1 e2 => count_fours e1 + count_fours e2

/-- Check if two expressions are equivalent under commutative and associative properties -/
def are_equivalent : Expr → Expr → Prop := sorry

theorem two_different_expressions_equal_seven :
  ∃ (e1 e2 : Expr),
    eval e1 = 7 ∧
    eval e2 = 7 ∧
    count_fours e1 = 4 ∧
    count_fours e2 = 4 ∧
    ¬(are_equivalent e1 e2) :=
  sorry

end two_different_expressions_equal_seven_l312_31292


namespace hyperbola_equation_correct_l312_31291

/-- Represents a hyperbola with given asymptotes and passing through a specific point -/
def is_correct_hyperbola (a b : ℝ) : Prop :=
  -- The equation of the hyperbola
  (∀ x y : ℝ, (3 * y^2 / 4) - (x^2 / 3) = 1 ↔ b * y^2 - a * x^2 = a * b) ∧
  -- The asymptotes are y = ±(2/3)x
  (a / b = 3 / 2) ∧
  -- The hyperbola passes through the point (√6, 2)
  (3 * 2^2 / 4 - Real.sqrt 6^2 / 3 = 1)

/-- The standard equation of the hyperbola satisfies the given conditions -/
theorem hyperbola_equation_correct :
  ∃ a b : ℝ, is_correct_hyperbola a b :=
sorry

end hyperbola_equation_correct_l312_31291


namespace least_positive_integer_with_remainders_l312_31220

theorem least_positive_integer_with_remainders : ∃! n : ℕ,
  (n > 0) ∧
  (n % 11 = 10) ∧
  (n % 12 = 11) ∧
  (n % 13 = 12) ∧
  (n % 14 = 13) ∧
  (n % 15 = 14) ∧
  (n % 16 = 15) ∧
  (∀ m : ℕ, m > 0 ∧ 
    (m % 11 = 10) ∧
    (m % 12 = 11) ∧
    (m % 13 = 12) ∧
    (m % 14 = 13) ∧
    (m % 15 = 14) ∧
    (m % 16 = 15) → m ≥ n) ∧
  n = 720719 :=
by sorry

end least_positive_integer_with_remainders_l312_31220


namespace nina_bought_two_card_packs_l312_31289

def num_toys : ℕ := 3
def toy_price : ℕ := 10
def num_shirts : ℕ := 5
def shirt_price : ℕ := 6
def card_pack_price : ℕ := 5
def total_spent : ℕ := 70

theorem nina_bought_two_card_packs :
  (num_toys * toy_price + num_shirts * shirt_price + 2 * card_pack_price = total_spent) := by
  sorry

end nina_bought_two_card_packs_l312_31289


namespace bryden_receives_amount_l312_31281

/-- The face value of a state quarter in dollars -/
def quarterValue : ℚ := 1 / 4

/-- The number of quarters Bryden has -/
def brydenQuarters : ℕ := 5

/-- The percentage multiplier offered by the collector -/
def collectorMultiplier : ℚ := 25

/-- The total amount Bryden will receive in dollars -/
def brydenReceives : ℚ := brydenQuarters * quarterValue * collectorMultiplier

theorem bryden_receives_amount : brydenReceives = 125 / 4 := by sorry

end bryden_receives_amount_l312_31281


namespace min_value_theorem_l312_31273

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + 2 * y = 3) :
  (x^2 + 3 * y) / (x * y) ≥ 2 * Real.sqrt 2 + 1 :=
by sorry

end min_value_theorem_l312_31273


namespace complex_exp_conversion_l312_31217

theorem complex_exp_conversion :
  (Complex.exp (13 * Real.pi * Complex.I / 4)) * (Complex.ofReal (Real.sqrt 2)) = -1 - Complex.I :=
by sorry

end complex_exp_conversion_l312_31217


namespace car_wash_solution_l312_31257

/-- Represents the car wash problem --/
structure CarWash where
  car_price : ℕ
  truck_price : ℕ
  suv_price : ℕ
  total_raised : ℕ
  num_suvs : ℕ
  num_cars : ℕ

/-- The solution to the car wash problem --/
def solve_car_wash (cw : CarWash) : ℕ :=
  (cw.total_raised - cw.car_price * cw.num_cars - cw.suv_price * cw.num_suvs) / cw.truck_price

/-- Theorem stating the solution to the specific problem --/
theorem car_wash_solution :
  let cw : CarWash := {
    car_price := 5,
    truck_price := 6,
    suv_price := 7,
    total_raised := 100,
    num_suvs := 5,
    num_cars := 7
  }
  solve_car_wash cw = 5 := by
  sorry

end car_wash_solution_l312_31257


namespace refrigerator_profit_percentage_l312_31295

/-- Calculates the profit percentage for a refrigerator sale --/
theorem refrigerator_profit_percentage 
  (discounted_price : ℝ) 
  (discount_rate : ℝ) 
  (transport_cost : ℝ) 
  (installation_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : discounted_price = 13500) 
  (h2 : discount_rate = 0.20) 
  (h3 : transport_cost = 125) 
  (h4 : installation_cost = 250) 
  (h5 : selling_price = 18975) : 
  ∃ (profit_percentage : ℝ), abs (profit_percentage - 36.73) < 0.01 := by
  sorry

end refrigerator_profit_percentage_l312_31295


namespace trigonometric_identity_l312_31249

theorem trigonometric_identity (α β : Real) 
  (h : (Real.sin α)^2 / (Real.cos β)^2 + (Real.cos α)^2 / (Real.sin β)^2 = 4) :
  (Real.cos β)^2 / (Real.sin α)^2 + (Real.sin β)^2 / (Real.cos α)^2 = -1 := by
  sorry

end trigonometric_identity_l312_31249


namespace nested_custom_op_equals_two_l312_31298

/-- Custom operation [a,b,c] defined as (a+b)/c where c ≠ 0 -/
def custom_op (a b c : ℚ) : ℚ := (a + b) / c

/-- Theorem stating that [[72,18,90],[4,2,6],[12,6,18]] = 2 -/
theorem nested_custom_op_equals_two :
  custom_op (custom_op 72 18 90) (custom_op 4 2 6) (custom_op 12 6 18) = 2 := by
  sorry

end nested_custom_op_equals_two_l312_31298


namespace value_of_a_l312_31211

theorem value_of_a (a : ℝ) : (0.005 * a = 0.80) → (a = 160) := by sorry

end value_of_a_l312_31211


namespace eleventh_number_with_digit_sum_13_l312_31297

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nthNumberWithDigitSum13 (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 11th number with digit sum 13 is 166 -/
theorem eleventh_number_with_digit_sum_13 : nthNumberWithDigitSum13 11 = 166 := by sorry

end eleventh_number_with_digit_sum_13_l312_31297


namespace milk_remainder_l312_31226

theorem milk_remainder (initial_milk : ℚ) (given_away : ℚ) (remainder : ℚ) : 
  initial_milk = 4 → given_away = 7/3 → remainder = initial_milk - given_away → remainder = 5/3 := by
  sorry

end milk_remainder_l312_31226


namespace a_range_l312_31263

def f (a : ℝ) (x : ℝ) : ℝ := x * |x - a|

theorem a_range (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ici 3 → x₂ ∈ Set.Ici 3 → x₁ ≠ x₂ → 
    (f a x₁ - f a x₂) / (x₁ - x₂) > 0) → 
  a ∈ Set.Iic 3 := by
sorry

end a_range_l312_31263


namespace necessary_but_not_sufficient_l312_31218

/-- The quadratic equation x^2 + ax + a = 0 has no real roots -/
def has_no_real_roots (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a*x + a ≠ 0

/-- The condition 0 ≤ a ≤ 4 is necessary but not sufficient for x^2 + ax + a = 0 to have no real roots -/
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, has_no_real_roots a → 0 ≤ a ∧ a ≤ 4) ∧
  (∃ a : ℝ, 0 ≤ a ∧ a ≤ 4 ∧ ¬(has_no_real_roots a)) :=
sorry

end necessary_but_not_sufficient_l312_31218


namespace min_children_for_all_colors_l312_31219

/-- Represents the distribution of pencils among children -/
structure PencilDistribution where
  total_pencils : ℕ
  num_colors : ℕ
  pencils_per_color : ℕ
  num_children : ℕ
  pencils_per_child : ℕ

/-- Theorem stating the minimum number of children to select to guarantee all colors -/
theorem min_children_for_all_colors (d : PencilDistribution) 
  (h1 : d.total_pencils = 24)
  (h2 : d.num_colors = 4)
  (h3 : d.pencils_per_color = 6)
  (h4 : d.num_children = 6)
  (h5 : d.pencils_per_child = 4)
  (h6 : d.total_pencils = d.num_colors * d.pencils_per_color)
  (h7 : d.total_pencils = d.num_children * d.pencils_per_child) :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), m < n → ¬(∀ (selection : Finset (Fin d.num_children)), 
    selection.card = m → 
    (∃ (colors : Finset (Fin d.num_colors)), colors.card = d.num_colors ∧
      ∀ (c : Fin d.num_colors), c ∈ colors → 
        ∃ (child : Fin d.num_children), child ∈ selection ∧ 
          ∃ (pencil : Fin d.pencils_per_child), pencil.val < d.pencils_per_child ∧
            (child.val * d.pencils_per_child + pencil.val) % d.num_colors = c.val))) ∧
  (∀ (selection : Finset (Fin d.num_children)), 
    selection.card = n → 
    (∃ (colors : Finset (Fin d.num_colors)), colors.card = d.num_colors ∧
      ∀ (c : Fin d.num_colors), c ∈ colors → 
        ∃ (child : Fin d.num_children), child ∈ selection ∧ 
          ∃ (pencil : Fin d.pencils_per_child), pencil.val < d.pencils_per_child ∧
            (child.val * d.pencils_per_child + pencil.val) % d.num_colors = c.val)) :=
by sorry

end min_children_for_all_colors_l312_31219


namespace linear_equation_equivalence_l312_31230

theorem linear_equation_equivalence (x y : ℝ) :
  (2 * x - y = 3) ↔ (y = 2 * x - 3) := by sorry

end linear_equation_equivalence_l312_31230


namespace triple_composition_identity_implies_identity_l312_31293

theorem triple_composition_identity_implies_identity 
  (f : ℝ → ℝ) (hf : Continuous f) (h : ∀ x, f (f (f x)) = x) : 
  ∀ x, f x = x := by
  sorry

end triple_composition_identity_implies_identity_l312_31293


namespace greene_nursery_yellow_carnations_l312_31278

theorem greene_nursery_yellow_carnations :
  let total_flowers : ℕ := 6284
  let red_roses : ℕ := 1491
  let white_roses : ℕ := 1768
  let yellow_carnations : ℕ := total_flowers - (red_roses + white_roses)
  yellow_carnations = 3025 := by
  sorry

end greene_nursery_yellow_carnations_l312_31278


namespace average_growth_rate_correct_l312_31236

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

end average_growth_rate_correct_l312_31236


namespace largest_four_digit_divisible_by_5_6_2_l312_31205

theorem largest_four_digit_divisible_by_5_6_2 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ 5 ∣ n ∧ 6 ∣ n → n ≤ 9990 :=
by
  sorry

end largest_four_digit_divisible_by_5_6_2_l312_31205


namespace keys_for_52_phones_l312_31213

/-- Represents the warehouse setup and the task of retrieving phones -/
structure WarehouseSetup where
  total_cabinets : ℕ
  boxes_per_cabinet : ℕ
  phones_per_box : ℕ
  phones_to_retrieve : ℕ

/-- Calculates the minimum number of keys required to retrieve the specified number of phones -/
def min_keys_required (setup : WarehouseSetup) : ℕ :=
  let boxes_needed := (setup.phones_to_retrieve + setup.phones_per_box - 1) / setup.phones_per_box
  let cabinets_needed := (boxes_needed + setup.boxes_per_cabinet - 1) / setup.boxes_per_cabinet
  boxes_needed + cabinets_needed + 1

/-- The theorem stating that for the given setup, 9 keys are required -/
theorem keys_for_52_phones :
  let setup : WarehouseSetup := {
    total_cabinets := 8,
    boxes_per_cabinet := 4,
    phones_per_box := 10,
    phones_to_retrieve := 52
  }
  min_keys_required setup = 9 := by
  sorry

end keys_for_52_phones_l312_31213


namespace lcm_12_35_l312_31239

theorem lcm_12_35 : Nat.lcm 12 35 = 420 := by
  sorry

end lcm_12_35_l312_31239


namespace min_value_a_plus_2b_l312_31203

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → x + 2*y ≥ 3 + 2*Real.sqrt 2 :=
by sorry

end min_value_a_plus_2b_l312_31203


namespace unique_white_bucket_count_l312_31229

/-- Represents a bucket with its color and water content -/
structure Bucket :=
  (color : Bool)  -- true for red, false for white
  (water : ℕ)

/-- Represents a move where water is added to a pair of buckets -/
structure Move :=
  (red_bucket : ℕ)
  (white_bucket : ℕ)
  (water_added : ℕ)

/-- The main theorem statement -/
theorem unique_white_bucket_count
  (red_count : ℕ)
  (white_count : ℕ)
  (moves : List Move)
  (h_red_count : red_count = 100)
  (h_all_non_empty : ∀ b : Bucket, b.water > 0)
  (h_equal_water : ∀ m : Move, ∃ b1 b2 : Bucket,
    b1.color = true ∧ b2.color = false ∧
    b1.water = b2.water) :
  white_count = 100 := by
  sorry

end unique_white_bucket_count_l312_31229


namespace vasya_shirt_day_l312_31223

structure TennisTournament where
  participants : ℕ
  days : ℕ
  matches_per_day : ℕ
  petya_shirt_day : ℕ
  petya_shirt_rank : ℕ
  vasya_shirt_rank : ℕ

def tournament : TennisTournament :=
  { participants := 20
  , days := 19
  , matches_per_day := 10
  , petya_shirt_day := 11
  , petya_shirt_rank := 11
  , vasya_shirt_rank := 15
  }

theorem vasya_shirt_day (t : TennisTournament) (h1 : t = tournament) :
  t.petya_shirt_day + (t.vasya_shirt_rank - t.petya_shirt_rank) = 15 :=
by
  sorry

#check vasya_shirt_day

end vasya_shirt_day_l312_31223


namespace train_platform_time_l312_31270

/-- Given a train of length 1500 meters that crosses a tree in 100 seconds,
    calculate the time taken to pass a platform of length 500 meters. -/
theorem train_platform_time (train_length platform_length tree_crossing_time : ℝ)
    (h1 : train_length = 1500)
    (h2 : platform_length = 500)
    (h3 : tree_crossing_time = 100) :
    (train_length + platform_length) / (train_length / tree_crossing_time) = 400/3 := by
  sorry

#eval (1500 + 500) / (1500 / 100) -- Should output approximately 133.33333333

end train_platform_time_l312_31270


namespace lives_lost_l312_31233

theorem lives_lost (starting_lives ending_lives : ℕ) 
  (h1 : starting_lives = 98)
  (h2 : ending_lives = 73) :
  starting_lives - ending_lives = 25 := by
  sorry

end lives_lost_l312_31233


namespace ones_digit_of_power_l312_31284

theorem ones_digit_of_power (x : ℕ) : (2^3)^x = 4096 → (3^(x^3)) % 10 = 1 := by
  sorry

end ones_digit_of_power_l312_31284


namespace scarves_difference_formula_l312_31200

/-- Calculates the difference in scarves produced between a normal day and a tiring day. -/
def scarves_difference (h : ℝ) : ℝ :=
  let s := 3 * h
  let normal_day := s * h
  let tiring_day := (s - 2) * (h - 3)
  normal_day - tiring_day

/-- Theorem stating that the difference in scarves produced is 11h - 6. -/
theorem scarves_difference_formula (h : ℝ) :
  scarves_difference h = 11 * h - 6 := by
  sorry

end scarves_difference_formula_l312_31200


namespace intersection_point_of_g_and_inverse_l312_31280

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 4*x^2 + 10*x + 20

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-2, -2) := by
  sorry

end intersection_point_of_g_and_inverse_l312_31280


namespace intersection_points_properties_l312_31201

/-- The curve equation -/
def curve (x : ℝ) : ℝ := x^2 - 5*x + 4

/-- The line equation -/
def line (p : ℝ) : ℝ := p

/-- Theorem stating the properties of the intersection points -/
theorem intersection_points_properties (a b p : ℝ) : 
  (curve a = line p) ∧ 
  (curve b = line p) ∧ 
  (a ≠ b) ∧
  (a^4 + b^4 = 1297) →
  (a = 6 ∧ b = -1) ∨ (a = -1 ∧ b = 6) := by
  sorry

#check intersection_points_properties

end intersection_points_properties_l312_31201


namespace inequality_proof_l312_31232

theorem inequality_proof (x : ℝ) (h1 : 3/2 ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end inequality_proof_l312_31232


namespace min_value_of_reciprocal_sum_l312_31288

/-- An arithmetic sequence of positive terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n, a n > 0) →
  a 1 + a 2015 = 2 →
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 2) →
  ∃ m : ℝ, m = 1/a 2 + 1/a 2014 ∧ m ≥ 2 ∧ ∀ z, z = 1/a 2 + 1/a 2014 → z ≥ m :=
sorry

end min_value_of_reciprocal_sum_l312_31288


namespace quadratic_inequality_solution_l312_31286

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) →
  a + b = -1 := by
sorry

end quadratic_inequality_solution_l312_31286


namespace final_cat_count_l312_31202

/-- Represents the number of cats of each breed -/
structure CatInventory where
  siamese : ℕ
  house : ℕ
  persian : ℕ
  sphynx : ℕ

/-- Calculates the total number of cats -/
def totalCats (inventory : CatInventory) : ℕ :=
  inventory.siamese + inventory.house + inventory.persian + inventory.sphynx

/-- Represents a sale event -/
structure SaleEvent where
  siamese : ℕ
  house : ℕ
  persian : ℕ
  sphynx : ℕ

/-- Applies a sale event to the inventory -/
def applySale (inventory : CatInventory) (sale : SaleEvent) : CatInventory where
  siamese := inventory.siamese - sale.siamese
  house := inventory.house - sale.house
  persian := inventory.persian - sale.persian
  sphynx := inventory.sphynx - sale.sphynx

/-- Adds new cats to the inventory -/
def addNewCats (inventory : CatInventory) (newSiamese newPersian : ℕ) : CatInventory where
  siamese := inventory.siamese + newSiamese
  house := inventory.house
  persian := inventory.persian + newPersian
  sphynx := inventory.sphynx

theorem final_cat_count (initialInventory : CatInventory)
    (sale1 sale2 : SaleEvent) (newSiamese newPersian : ℕ)
    (h1 : initialInventory = CatInventory.mk 12 20 8 18)
    (h2 : sale1 = SaleEvent.mk 6 4 5 0)
    (h3 : sale2 = SaleEvent.mk 0 15 0 10)
    (h4 : newSiamese = 5)
    (h5 : newPersian = 3) :
    totalCats (addNewCats (applySale (applySale initialInventory sale1) sale2) newSiamese newPersian) = 26 := by
  sorry


end final_cat_count_l312_31202


namespace max_distance_between_inscribed_squares_max_distance_is_5_sqrt_2_l312_31260

/-- The maximum distance between vertices of two squares, where the smaller square
    (perimeter 24) is inscribed in the larger square (perimeter 32) and rotated such that
    one of its vertices lies on the midpoint of one side of the larger square. -/
theorem max_distance_between_inscribed_squares : ℝ :=
  let inner_perimeter : ℝ := 24
  let outer_perimeter : ℝ := 32
  let inner_side : ℝ := inner_perimeter / 4
  let outer_side : ℝ := outer_perimeter / 4
  5 * Real.sqrt 2

/-- Proof that the maximum distance between vertices of the inscribed squares is 5√2. -/
theorem max_distance_is_5_sqrt_2 (inner_perimeter outer_perimeter : ℝ)
  (h1 : inner_perimeter = 24)
  (h2 : outer_perimeter = 32)
  (h3 : ∃ (v : ℝ × ℝ), v.1 = outer_perimeter / 8 ∧ v.2 = 0) :
  max_distance_between_inscribed_squares = 5 * Real.sqrt 2 :=
by
  sorry

end max_distance_between_inscribed_squares_max_distance_is_5_sqrt_2_l312_31260


namespace prob_10_or_9_prob_at_least_7_l312_31221

/-- Represents the probabilities of hitting different rings in a shooting event -/
structure ShootingProbabilities where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ
  ring7 : ℝ
  below7 : ℝ

/-- The probabilities sum to 1 -/
axiom prob_sum_to_one (p : ShootingProbabilities) : 
  p.ring10 + p.ring9 + p.ring8 + p.ring7 + p.below7 = 1

/-- All probabilities are non-negative -/
axiom prob_non_negative (p : ShootingProbabilities) : 
  p.ring10 ≥ 0 ∧ p.ring9 ≥ 0 ∧ p.ring8 ≥ 0 ∧ p.ring7 ≥ 0 ∧ p.below7 ≥ 0

/-- Given probabilities for a shooting event -/
def shooter_probs : ShootingProbabilities := {
  ring10 := 0.1,
  ring9 := 0.2,
  ring8 := 0.3,
  ring7 := 0.3,
  below7 := 0.1
}

/-- Theorem: The probability of hitting the 10 or 9 ring is 0.3 -/
theorem prob_10_or_9 : shooter_probs.ring10 + shooter_probs.ring9 = 0.3 := by sorry

/-- Theorem: The probability of hitting at least the 7 ring is 0.9 -/
theorem prob_at_least_7 : 1 - shooter_probs.below7 = 0.9 := by sorry

end prob_10_or_9_prob_at_least_7_l312_31221


namespace fencing_calculation_l312_31204

/-- Represents a rectangular field with fencing on three sides -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- Calculates the required fencing for a given field -/
def required_fencing (field : FencedField) : ℝ :=
  field.length + 2 * field.width

theorem fencing_calculation (field : FencedField) 
  (h1 : field.area = 680)
  (h2 : field.uncovered_side = 34)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  required_fencing field = 74 := by
  sorry

#check fencing_calculation

end fencing_calculation_l312_31204


namespace alternating_squares_sum_l312_31206

theorem alternating_squares_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 221 := by
  sorry

end alternating_squares_sum_l312_31206


namespace quadratic_problem_l312_31215

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the properties of the quadratic function and its value at x = 5 -/
theorem quadratic_problem (a b c : ℝ) :
  (∀ x, f a b c x ≥ 4) →  -- Minimum value is 4
  (f a b c 2 = 4) →  -- Minimum occurs at x = 2
  (f a b c 0 = -8) →  -- Passes through (0, -8)
  (f a b c 5 = 31) :=  -- Passes through (5, 31)
by sorry

end quadratic_problem_l312_31215


namespace equation_solution_l312_31294

theorem equation_solution : ∃ x : ℝ, 45 * x = 0.6 * 900 ∧ x = 12 := by
  sorry

end equation_solution_l312_31294


namespace recover_sequence_l312_31246

/-- A sequence of six positive integers in arithmetic progression. -/
def ArithmeticSequence : Type := Fin 6 → ℕ+

/-- The given sequence with one number omitted and one miscopied. -/
def GivenSequence : Fin 5 → ℕ+ := ![113, 137, 149, 155, 173]

/-- The correct sequence. -/
def CorrectSequence : ArithmeticSequence := ![113, 125, 137, 149, 161, 173]

/-- Checks if a sequence is in arithmetic progression. -/
def isArithmeticProgression (s : ArithmeticSequence) : Prop :=
  ∃ d : ℕ+, ∀ i : Fin 5, s (i + 1) = s i + d

/-- Checks if a sequence matches the given sequence except for one miscopied number. -/
def matchesGivenSequence (s : ArithmeticSequence) : Prop :=
  ∃ j : Fin 5, ∀ i : Fin 5, i ≠ j → s i = GivenSequence i

theorem recover_sequence :
  isArithmeticProgression CorrectSequence ∧
  matchesGivenSequence CorrectSequence :=
sorry

end recover_sequence_l312_31246


namespace truck_transport_time_l312_31251

theorem truck_transport_time (total_time : ℝ) (first_truck_portion : ℝ) (actual_time : ℝ)
  (h1 : total_time = 6)
  (h2 : first_truck_portion = 3/5)
  (h3 : actual_time = 12) :
  ∃ (t1 t2 : ℝ),
    ((t1 = 10 ∧ t2 = 15) ∨ (t1 = 12 ∧ t2 = 12)) ∧
    (1 / t1 + 1 / t2 = 1 / total_time) ∧
    (first_truck_portion / t1 + (1 - first_truck_portion) / t2 = 1 / actual_time) := by
  sorry

end truck_transport_time_l312_31251


namespace geometric_sequence_common_ratio_l312_31261

/-- Proves that for an increasing geometric sequence with a_3 = 8 and S_3 = 14,
    the common ratio is 2. -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)  -- The geometric sequence
  (h_incr : ∀ n, a n < a (n + 1))  -- The sequence is increasing
  (h_a3 : a 3 = 8)  -- Third term is 8
  (h_S3 : (a 1) + (a 2) + (a 3) = 14)  -- Sum of first 3 terms is 14
  : ∃ q : ℝ, (∀ n, a (n + 1) = q * a n) ∧ q = 2 :=
by sorry

end geometric_sequence_common_ratio_l312_31261


namespace ratio_closest_to_ten_l312_31276

theorem ratio_closest_to_ten : 
  let r := (10^3000 + 10^3004) / (10^3001 + 10^3003)
  ∀ n : ℤ, n ≠ 10 → |r - 10| < |r - n| := by
  sorry

end ratio_closest_to_ten_l312_31276


namespace janet_flight_cost_l312_31242

/-- The cost of flying between two cities -/
def flying_cost (distance : ℝ) (cost_per_km : ℝ) (booking_fee : ℝ) : ℝ :=
  distance * cost_per_km + booking_fee

/-- Theorem: The cost for Janet to fly from City D to City E is $720 -/
theorem janet_flight_cost : 
  flying_cost 4750 0.12 150 = 720 := by
  sorry

end janet_flight_cost_l312_31242
