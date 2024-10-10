import Mathlib

namespace percent_increase_in_sales_l3933_393373

def sales_last_year : ℝ := 320
def sales_this_year : ℝ := 480

theorem percent_increase_in_sales :
  (sales_this_year - sales_last_year) / sales_last_year * 100 = 50 := by
  sorry

end percent_increase_in_sales_l3933_393373


namespace perfect_square_condition_l3933_393395

theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n^2 - 19*n + 95 = k^2) ↔ (n = 5 ∨ n = 14) := by
sorry

end perfect_square_condition_l3933_393395


namespace program_result_l3933_393300

def double_n_times (initial : ℕ) (n : ℕ) : ℕ :=
  initial * (2^n)

theorem program_result :
  double_n_times 1 6 = 64 := by
  sorry

end program_result_l3933_393300


namespace unique_quadruple_existence_l3933_393359

theorem unique_quadruple_existence : 
  ∃! (a b c d : ℝ), 
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a^2 + b^2 + c^2 + d^2 = 4 ∧
    (a + b + c + d) * (a^2 + b^2 + c^2 + d^2) = 8 := by
  sorry

end unique_quadruple_existence_l3933_393359


namespace inequality_preservation_l3933_393308

theorem inequality_preservation (m n : ℝ) (h : m > n) : m / 4 > n / 4 := by
  sorry

end inequality_preservation_l3933_393308


namespace marie_erasers_l3933_393380

/-- Given that Marie starts with 95.0 erasers and buys 42.0 erasers, 
    prove that she ends up with 137.0 erasers. -/
theorem marie_erasers : 
  let initial_erasers : ℝ := 95.0
  let bought_erasers : ℝ := 42.0
  let final_erasers : ℝ := initial_erasers + bought_erasers
  final_erasers = 137.0 := by
  sorry

end marie_erasers_l3933_393380


namespace solve_for_x_l3933_393383

theorem solve_for_x (x y : ℚ) (h1 : x / y = 8 / 3) (h2 : y = 21) : x = 56 := by
  sorry

end solve_for_x_l3933_393383


namespace guide_is_native_l3933_393324

-- Define the two tribes
inductive Tribe
| Native
| Alien

-- Define a function to represent whether a statement is true or false
def isTruthful (t : Tribe) (s : Prop) : Prop :=
  match t with
  | Tribe.Native => s
  | Tribe.Alien => ¬s

-- Define the guide's statement
def guideStatement (encounteredTribe : Tribe) : Prop :=
  isTruthful encounteredTribe (encounteredTribe = Tribe.Native)

-- Theorem: The guide must be a native
theorem guide_is_native :
  ∀ (guideTribe : Tribe),
    (∀ (encounteredTribe : Tribe),
      isTruthful guideTribe (guideStatement encounteredTribe)) →
    guideTribe = Tribe.Native :=
by sorry


end guide_is_native_l3933_393324


namespace triangle_ABC_is_obtuse_angled_l3933_393363

/-- Triangle ABC is obtuse-angled given the specified angle conditions -/
theorem triangle_ABC_is_obtuse_angled (A B C : ℝ) 
  (h1 : A + B = 141)
  (h2 : C + B = 165)
  (h3 : A + B + C = 180) : 
  ∃ (angle : ℝ), angle > 90 ∧ (angle = A ∨ angle = B ∨ angle = C) := by
  sorry

end triangle_ABC_is_obtuse_angled_l3933_393363


namespace daffodil_stamps_count_l3933_393336

theorem daffodil_stamps_count 
  (rooster_stamps : ℕ) 
  (daffodil_stamps : ℕ) 
  (h1 : rooster_stamps = 2) 
  (h2 : rooster_stamps - daffodil_stamps = 0) : 
  daffodil_stamps = 2 :=
by sorry

end daffodil_stamps_count_l3933_393336


namespace arcsin_arccos_pi_sixth_l3933_393335

theorem arcsin_arccos_pi_sixth : 
  Real.arcsin (1/2) = π/6 ∧ Real.arccos (Real.sqrt 3/2) = π/6 := by
  sorry

end arcsin_arccos_pi_sixth_l3933_393335


namespace intersection_of_sets_l3933_393334

theorem intersection_of_sets : 
  let A : Set Int := {-1, 0, 1}
  let B : Set Int := {0, 1, 2, 3}
  A ∩ B = {0, 1} := by
sorry

end intersection_of_sets_l3933_393334


namespace min_k_plus_l_l3933_393369

theorem min_k_plus_l (k l : ℕ+) (h : 120 * k = l ^ 3) : 
  ∀ (k' l' : ℕ+), 120 * k' = l' ^ 3 → k + l ≤ k' + l' :=
by sorry

end min_k_plus_l_l3933_393369


namespace quadratic_min_bound_l3933_393399

theorem quadratic_min_bound (p q α β : ℝ) (n : ℤ) (h : ℝ → ℝ) :
  (∀ x, h x = x^2 + p*x + q) →
  h α = 0 →
  h β = 0 →
  α ≠ β →
  (n : ℝ) < α →
  α < β →
  β < (n + 1 : ℝ) →
  min (h n) (h (n + 1)) < (1/4 : ℝ) := by
  sorry

end quadratic_min_bound_l3933_393399


namespace distance_before_collision_l3933_393382

/-- Theorem: Distance between boats 3 minutes before collision -/
theorem distance_before_collision
  (river_current : ℝ)
  (boat1_speed : ℝ)
  (boat2_speed : ℝ)
  (initial_distance : ℝ)
  (h1 : river_current = 2)
  (h2 : boat1_speed = 5)
  (h3 : boat2_speed = 25)
  (h4 : initial_distance = 20) :
  let relative_speed := (boat1_speed - river_current) + (boat2_speed - river_current)
  let time_before_collision : ℝ := 3 / 60
  let distance_covered := relative_speed * time_before_collision
  initial_distance - distance_covered = 1.3 := by
  sorry

#check distance_before_collision

end distance_before_collision_l3933_393382


namespace algebraic_expression_equality_l3933_393364

theorem algebraic_expression_equality (x y : ℝ) : 
  x - 2*y + 8 = 18 → 3*x - 6*y + 4 = 34 := by
  sorry

end algebraic_expression_equality_l3933_393364


namespace division_remainder_l3933_393394

theorem division_remainder : ∃ (q r : ℕ), 1620 = (1620 - 1365) * q + r ∧ r < (1620 - 1365) ∧ r = 90 := by
  sorry

end division_remainder_l3933_393394


namespace focus_directrix_distance_l3933_393341

/-- The parabola equation y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The directrix of the parabola y^2 = 4x -/
def directrix (x : ℝ) : Prop := x = -1

/-- The distance from the focus to the directrix of the parabola y^2 = 4x is 2 -/
theorem focus_directrix_distance : 
  ∃ (d : ℝ), d = 2 ∧ d = |focus.1 - (-1)| :=
sorry

end focus_directrix_distance_l3933_393341


namespace good_number_exists_l3933_393311

/-- A function that checks if two numbers have the same digits (possibly in different order) --/
def sameDigits (a b : ℕ) : Prop := sorry

/-- A function that checks if a number is a four-digit number --/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem good_number_exists : ∃ n : ℕ, 
  isFourDigit n ∧ 
  n % 11 = 0 ∧ 
  sameDigits n (3 * n) ∧
  n = 2475 := by sorry

end good_number_exists_l3933_393311


namespace simplify_expression_1_simplify_expression_2_l3933_393375

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = 3 * a^2 * b - a * b^2 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) :
  7 * x + 2 * (x^2 - 2) - 4 * (1/2 * x^2 - x + 3) = 11 * x - 16 := by
  sorry

end simplify_expression_1_simplify_expression_2_l3933_393375


namespace colored_spanning_tree_existence_l3933_393353

/-- A colored edge in a graph -/
inductive ColoredEdge
  | Red
  | Green
  | Blue

/-- A graph with colored edges -/
structure ColoredGraph (V : Type) where
  edges : V → V → Option ColoredEdge

/-- A spanning tree of a graph -/
def SpanningTree (V : Type) := V → V → Prop

/-- Count the number of edges of a specific color in a spanning tree -/
def CountEdges (V : Type) (t : SpanningTree V) (c : ColoredEdge) : ℕ := sorry

/-- The main theorem -/
theorem colored_spanning_tree_existence
  (V : Type)
  (G : ColoredGraph V)
  (n : ℕ)
  (r v b : ℕ)
  (h_connected : sorry)  -- G is connected
  (h_vertex_count : sorry)  -- G has n+1 vertices
  (h_sum : r + v + b = n)
  (h_red_tree : ∃ t : SpanningTree V, CountEdges V t ColoredEdge.Red = r)
  (h_green_tree : ∃ t : SpanningTree V, CountEdges V t ColoredEdge.Green = v)
  (h_blue_tree : ∃ t : SpanningTree V, CountEdges V t ColoredEdge.Blue = b) :
  ∃ t : SpanningTree V,
    CountEdges V t ColoredEdge.Red = r ∧
    CountEdges V t ColoredEdge.Green = v ∧
    CountEdges V t ColoredEdge.Blue = b :=
  sorry

end colored_spanning_tree_existence_l3933_393353


namespace f_sum_theorem_l3933_393368

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_sum_theorem (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : ∀ x, f (x + 3) = -f x)
  (h_f1 : f 1 = -1) : 
  f 5 + f 13 = -2 := by
sorry

end f_sum_theorem_l3933_393368


namespace nth_prime_upper_bound_l3933_393319

def nth_prime (n : ℕ) : ℕ := sorry

theorem nth_prime_upper_bound (n : ℕ) : nth_prime n ≤ 2^(2^(n-1)) := by sorry

end nth_prime_upper_bound_l3933_393319


namespace total_trophies_in_three_years_l3933_393385

theorem total_trophies_in_three_years :
  let michael_current_trophies : ℕ := 30
  let michael_trophy_increase : ℕ := 100
  let jack_trophy_multiplier : ℕ := 10
  let michael_future_trophies : ℕ := michael_current_trophies + michael_trophy_increase
  let jack_future_trophies : ℕ := jack_trophy_multiplier * michael_current_trophies
  michael_future_trophies + jack_future_trophies = 430 := by
sorry

end total_trophies_in_three_years_l3933_393385


namespace twins_age_product_difference_l3933_393320

theorem twins_age_product_difference (current_age : ℕ) : 
  current_age = 2 → (current_age + 1)^2 - current_age^2 = 5 := by
  sorry

end twins_age_product_difference_l3933_393320


namespace social_logistics_turnover_scientific_notation_l3933_393340

/-- Given that one trillion is 10^12, prove that 347.6 trillion yuan is equal to 3.476 × 10^14 yuan -/
theorem social_logistics_turnover_scientific_notation :
  let trillion : ℝ := 10^12
  347.6 * trillion = 3.476 * 10^14 := by
  sorry

end social_logistics_turnover_scientific_notation_l3933_393340


namespace inverse_f_at_2_l3933_393323

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inverse_f_at_2 :
  ∃ (f_inv : ℝ → ℝ),
    (∀ x ≥ 0, f_inv (f x) = x) ∧
    (∀ y ≥ -1, f (f_inv y) = y) ∧
    f_inv 2 = Real.sqrt 3 := by
  sorry

end inverse_f_at_2_l3933_393323


namespace polynomial_simplification_l3933_393325

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 :=
by sorry

end polynomial_simplification_l3933_393325


namespace interest_rate_calculation_l3933_393301

/-- Proves that the rate of interest is 8% given the problem conditions --/
theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) (rate : ℝ) :
  principal = 1100 →
  interest_paid = 704 →
  interest_paid = principal * rate * rate / 100 →
  rate = 8 :=
by
  sorry

#check interest_rate_calculation

end interest_rate_calculation_l3933_393301


namespace specific_value_calculation_l3933_393303

theorem specific_value_calculation : ∀ (x : ℕ), x = 11 → x + 3 + 5 = 19 := by
  sorry

end specific_value_calculation_l3933_393303


namespace triangle_area_specific_l3933_393370

/-- The area of a triangle given the coordinates of its vertices -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℤ) : ℚ :=
  (1 / 2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- Theorem: The area of a triangle with vertices at (-1,-1), (2,3), and (-4,0) is 8.5 square units -/
theorem triangle_area_specific : triangleArea (-1) (-1) 2 3 (-4) 0 = 17/2 := by
  sorry

end triangle_area_specific_l3933_393370


namespace simplify_and_evaluate_l3933_393391

theorem simplify_and_evaluate (x : ℝ) (h : x = 1 / (3 + 2 * Real.sqrt 2)) :
  ((1 - x)^2 / (x - 1)) + (Real.sqrt (x^2 + 4 - 4*x) / (x - 2)) = 1 - 2 * Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l3933_393391


namespace floor_difference_l3933_393386

theorem floor_difference : ⌊(-2.7 : ℝ)⌋ - ⌊(4.5 : ℝ)⌋ = -7 := by
  sorry

end floor_difference_l3933_393386


namespace infinite_geometric_series_first_term_l3933_393342

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h1 : r = 1 / 4)
  (h2 : S = 50)
  (h3 : S = a / (1 - r)) :
  a = 75 / 2 := by
  sorry

end infinite_geometric_series_first_term_l3933_393342


namespace max_intersection_points_l3933_393314

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 20

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 10

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := 8550

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersection_points :
  (num_x_points.choose 2) * (num_y_points.choose 2) = max_intersections :=
sorry

end max_intersection_points_l3933_393314


namespace julio_lime_cost_l3933_393317

/-- Represents the number of days Julio makes mocktails -/
def days : ℕ := 30

/-- Represents the amount of lime juice used per mocktail in tablespoons -/
def juice_per_mocktail : ℚ := 1

/-- Represents the amount of lime juice that can be squeezed from one lime in tablespoons -/
def juice_per_lime : ℚ := 2

/-- Represents the number of limes sold for $1.00 -/
def limes_per_dollar : ℚ := 3

/-- Calculates the total cost of limes for Julio's mocktails over the given number of days -/
def lime_cost (d : ℕ) (j_mocktail j_lime l_dollar : ℚ) : ℚ :=
  (d * j_mocktail / j_lime) / l_dollar

/-- Theorem stating that Julio will spend $5.00 on limes after 30 days -/
theorem julio_lime_cost : 
  lime_cost days juice_per_mocktail juice_per_lime limes_per_dollar = 5 := by
  sorry

end julio_lime_cost_l3933_393317


namespace joey_study_time_l3933_393393

/-- Calculates the total study time for Joey's SAT exam -/
def total_study_time (weekday_hours_per_night : ℕ) (weekday_nights : ℕ) 
  (weekend_hours_per_day : ℕ) (weekend_days : ℕ) (weeks_until_exam : ℕ) : ℕ :=
  ((weekday_hours_per_night * weekday_nights + weekend_hours_per_day * weekend_days) 
    * weeks_until_exam)

/-- Proves that Joey will spend 96 hours studying for his SAT exam -/
theorem joey_study_time : 
  total_study_time 2 5 3 2 6 = 96 := by
  sorry

end joey_study_time_l3933_393393


namespace vector_magnitude_l3933_393345

theorem vector_magnitude (a b : ℝ × ℝ) :
  a = (2, 1) →
  a.1 * b.1 + a.2 * b.2 = 10 →
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = 50 →
  b.1^2 + b.2^2 = 25 := by
  sorry

end vector_magnitude_l3933_393345


namespace reciprocal_of_two_thirds_l3933_393356

theorem reciprocal_of_two_thirds : 
  (2 : ℚ) / 3 * (3 : ℚ) / 2 = 1 := by sorry

end reciprocal_of_two_thirds_l3933_393356


namespace power_of_seven_expansion_l3933_393357

theorem power_of_seven_expansion : 7^3 - 3*(7^2) + 3*7 - 1 = 216 := by
  sorry

end power_of_seven_expansion_l3933_393357


namespace square_of_85_l3933_393376

theorem square_of_85 : (85 : ℕ)^2 = 7225 := by
  sorry

end square_of_85_l3933_393376


namespace rice_weight_qualification_l3933_393371

def weight_range (x : ℝ) : Prop := 9.9 ≤ x ∧ x ≤ 10.1

theorem rice_weight_qualification :
  ¬(weight_range 9.09) ∧
  weight_range 9.99 ∧
  weight_range 10.01 ∧
  weight_range 10.09 :=
by sorry

end rice_weight_qualification_l3933_393371


namespace carpet_cost_calculation_carpet_cost_result_l3933_393318

/-- Calculate the cost of a carpet with increased dimensions -/
theorem carpet_cost_calculation (breadth_1 : Real) (length_ratio : Real) 
  (length_increase : Real) (breadth_increase : Real) (rate : Real) : Real :=
  let length_1 := breadth_1 * length_ratio
  let breadth_2 := breadth_1 * (1 + breadth_increase)
  let length_2 := length_1 * (1 + length_increase)
  let area_2 := breadth_2 * length_2
  area_2 * rate

/-- The cost of the carpet with specified dimensions and rate -/
theorem carpet_cost_result : 
  carpet_cost_calculation 6 1.44 0.4 0.25 45 = 4082.4 := by
  sorry

end carpet_cost_calculation_carpet_cost_result_l3933_393318


namespace cube_root_nine_inequality_false_l3933_393339

theorem cube_root_nine_inequality_false : 
  ¬(∀ n : ℤ, (n : ℝ) < (9 : ℝ)^(1/3) ∧ (9 : ℝ)^(1/3) < (n : ℝ) + 1 → n = 3) :=
by sorry

end cube_root_nine_inequality_false_l3933_393339


namespace integer_fraction_pairs_l3933_393388

def is_integer_fraction (m n : ℕ+) : Prop :=
  ∃ k : ℤ, (n.val ^ 3 + 1 : ℤ) = k * (m.val * n.val - 1)

def solution_set : Set (ℕ+ × ℕ+) :=
  {(1, 2), (1, 3), (2, 1), (2, 2), (2, 5), (3, 1), (3, 5), (5, 2), (5, 3)}

theorem integer_fraction_pairs :
  {p : ℕ+ × ℕ+ | is_integer_fraction p.1 p.2} = solution_set :=
sorry

end integer_fraction_pairs_l3933_393388


namespace test_score_calculation_l3933_393321

/-- Calculates the total score for a test given the total number of problems,
    points for correct answers, points deducted for wrong answers,
    and the number of wrong answers. -/
def calculateScore (totalProblems : ℕ) (pointsPerCorrect : ℕ) (pointsPerWrong : ℕ) (wrongAnswers : ℕ) : ℤ :=
  (totalProblems - wrongAnswers : ℤ) * pointsPerCorrect - wrongAnswers * pointsPerWrong

/-- Theorem stating that for a test with 25 problems, 4 points for each correct answer,
    1 point deducted for each wrong answer, and 3 wrong answers, the total score is 85. -/
theorem test_score_calculation :
  calculateScore 25 4 1 3 = 85 := by
  sorry

end test_score_calculation_l3933_393321


namespace pentagon_area_sum_l3933_393352

/-- Represents a pentagon with vertices F, G, H, I, J -/
structure Pentagon :=
  (F G H I J : Point)

/-- The area of the pentagon -/
def area (p : Pentagon) : ℝ := sorry

/-- Condition that the pentagon is constructed from 10 line segments of length 3 -/
def is_valid_pentagon (p : Pentagon) : Prop := sorry

theorem pentagon_area_sum (p : Pentagon) (a b : ℕ) :
  is_valid_pentagon p →
  area p = Real.sqrt a + Real.sqrt b →
  a + b = 29 := by sorry

end pentagon_area_sum_l3933_393352


namespace panda_babies_born_l3933_393392

/-- The number of panda babies born in a zoo with given conditions -/
theorem panda_babies_born (total_pandas : ℕ) (pregnancy_rate : ℚ) : 
  total_pandas = 16 →
  pregnancy_rate = 1/4 →
  (total_pandas / 2 : ℚ) * pregnancy_rate * 1 = 2 := by
  sorry

end panda_babies_born_l3933_393392


namespace set_membership_implies_a_values_l3933_393327

def A (a : ℝ) : Set ℝ := {2, 1-a, a^2-a+2}

theorem set_membership_implies_a_values (a : ℝ) :
  4 ∈ A a → a = -3 ∨ a = 2 := by
  sorry

end set_membership_implies_a_values_l3933_393327


namespace library_visitors_average_l3933_393328

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (sundays_in_month : ℕ) (h1 : sunday_visitors = 140) 
  (h2 : other_day_visitors = 80) (h3 : days_in_month = 30) (h4 : sundays_in_month = 4) :
  (sunday_visitors * sundays_in_month + other_day_visitors * (days_in_month - sundays_in_month)) / 
  days_in_month = 88 := by
  sorry

#check library_visitors_average

end library_visitors_average_l3933_393328


namespace chessboard_impossible_l3933_393366

/-- Represents a 6x6 chessboard filled with numbers -/
def Chessboard := Fin 6 → Fin 6 → Fin 36

/-- The sum of numbers from 1 to 36 -/
def total_sum : Nat := (36 * 37) / 2

/-- The required sum for each row, column, and diagonal -/
def required_sum : Nat := total_sum / 6

/-- Checks if a number appears exactly once on the chessboard -/
def appears_once (board : Chessboard) (n : Fin 36) : Prop :=
  ∃! (i j : Fin 6), board i j = n

/-- Checks if a row has the required sum -/
def row_sum_correct (board : Chessboard) (i : Fin 6) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun j => (board i j).val + 1) = required_sum

/-- Checks if a column has the required sum -/
def col_sum_correct (board : Chessboard) (j : Fin 6) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun i => (board i j).val + 1) = required_sum

/-- Checks if a northeast diagonal has the required sum -/
def diag_sum_correct (board : Chessboard) (k : Fin 6) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun i =>
    (board i ((i.val - k.val + 6) % 6 : Fin 6)).val + 1) = required_sum

/-- The main theorem stating that it's impossible to fill the chessboard with the given conditions -/
theorem chessboard_impossible : ¬∃ (board : Chessboard),
  (∀ n : Fin 36, appears_once board n) ∧
  (∀ i : Fin 6, row_sum_correct board i) ∧
  (∀ j : Fin 6, col_sum_correct board j) ∧
  (∀ k : Fin 6, diag_sum_correct board k) :=
sorry

end chessboard_impossible_l3933_393366


namespace sum_of_s_coordinates_l3933_393377

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle defined by four points -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Given a rectangle PQRS with P and R as diagonally opposite corners,
    proves that the sum of coordinates of S is 8 -/
theorem sum_of_s_coordinates (rect : Rectangle) : 
  rect.P = Point.mk (-3) (-2) →
  rect.R = Point.mk 9 1 →
  rect.Q = Point.mk 2 (-5) →
  rect.S.x + rect.S.y = 8 := by
  sorry


end sum_of_s_coordinates_l3933_393377


namespace sin_sum_of_complex_exponentials_l3933_393365

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  (Complex.exp (Complex.I * θ) = (4 : ℝ) / 5 + (3 : ℝ) / 5 * Complex.I) →
  (Complex.exp (Complex.I * φ) = -(5 : ℝ) / 13 + (12 : ℝ) / 13 * Complex.I) →
  Real.sin (θ + φ) = (33 : ℝ) / 65 := by
  sorry

end sin_sum_of_complex_exponentials_l3933_393365


namespace remaining_balance_l3933_393346

def house_price : ℝ := 100000

def down_payment_percentage : ℝ := 0.20

def parents_contribution_percentage : ℝ := 0.30

theorem remaining_balance (hp : ℝ) (dp : ℝ) (pc : ℝ) : 
  hp * (1 - dp) * (1 - pc) = 56000 :=
by
  sorry

#check remaining_balance house_price down_payment_percentage parents_contribution_percentage

end remaining_balance_l3933_393346


namespace wage_decrease_hours_increase_l3933_393316

theorem wage_decrease_hours_increase (W H : ℝ) (W_new H_new : ℝ) :
  W > 0 → H > 0 →
  W_new = 0.8 * W →
  W * H = W_new * H_new →
  (H_new - H) / H = 0.25 :=
sorry

end wage_decrease_hours_increase_l3933_393316


namespace instantaneous_velocity_at_3_seconds_l3933_393360

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := -1 + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by sorry

end instantaneous_velocity_at_3_seconds_l3933_393360


namespace tire_sale_price_l3933_393322

/-- Calculates the sale price of a tire given the number of tires, total savings, and original price. -/
def sale_price (num_tires : ℕ) (total_savings : ℚ) (original_price : ℚ) : ℚ :=
  original_price - (total_savings / num_tires)

/-- Theorem stating that the sale price of each tire is $75 given the problem conditions. -/
theorem tire_sale_price :
  let num_tires : ℕ := 4
  let total_savings : ℚ := 36
  let original_price : ℚ := 84
  sale_price num_tires total_savings original_price = 75 := by
sorry

end tire_sale_price_l3933_393322


namespace triangle_perimeter_strict_l3933_393302

theorem triangle_perimeter_strict (a b x : ℝ) : 
  a = 12 → b = 25 → a > 0 → b > 0 → x > 0 → 
  a + b > x → a + x > b → b + x > a → 
  a + b + x > 50 := by sorry

end triangle_perimeter_strict_l3933_393302


namespace largest_number_proof_l3933_393389

def is_valid_expression (expr : ℕ → ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    (a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 8 ∧ e = 8 ∧ f = 8) ∧
    (∀ n, expr n = n ∨ expr n = a ∨ expr n = b ∨ expr n = c ∨ expr n = d ∨ expr n = e ∨ expr n = f ∨
      ∃ (x y : ℕ), (expr n = expr x + expr y ∨ expr n = expr x - expr y ∨ 
                    expr n = expr x * expr y ∨ expr n = expr x / expr y ∨ 
                    expr n = expr x ^ expr y))

def largest_expression : ℕ → ℕ :=
  fun n => 3^(3^(3^(8^(8^8))))

theorem largest_number_proof :
  (is_valid_expression largest_expression) ∧
  (∀ expr, is_valid_expression expr → ∀ n, expr n ≤ largest_expression n) :=
by sorry

end largest_number_proof_l3933_393389


namespace stream_speed_l3933_393310

/-- Proves that the speed of a stream is 4 km/hr given the conditions of the boat problem -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : boat_speed = 24)
  (h2 : distance = 168) (h3 : time = 6)
  (h4 : distance = (boat_speed + (distance / time - boat_speed)) * time) : 
  distance / time - boat_speed = 4 := by
  sorry

end stream_speed_l3933_393310


namespace central_angle_from_arc_length_l3933_393358

/-- Given a circle with radius 2 and arc length 4, prove that the central angle is 2 radians -/
theorem central_angle_from_arc_length (r : ℝ) (l : ℝ) (h1 : r = 2) (h2 : l = 4) :
  l / r = 2 := by
  sorry

end central_angle_from_arc_length_l3933_393358


namespace complex_square_of_one_plus_i_l3933_393307

theorem complex_square_of_one_plus_i :
  ∀ z : ℂ, (z.re = 1 ∧ z.im = 1) → z^2 = 2*I :=
by sorry

end complex_square_of_one_plus_i_l3933_393307


namespace multiplication_problem_l3933_393348

theorem multiplication_problem (x : ℝ) : 4 * x = 60 → 8 * x = 120 := by
  sorry

end multiplication_problem_l3933_393348


namespace small_triangle_perimeter_l3933_393331

/-- Given a triangle with perimeter 11 and three trapezoids formed by cuts parallel to its sides
    with perimeters 5, 7, and 9, the perimeter of the small triangle formed after the cuts is 10. -/
theorem small_triangle_perimeter (original_perimeter : ℝ) (trapezoid1_perimeter trapezoid2_perimeter trapezoid3_perimeter : ℝ)
    (h1 : original_perimeter = 11)
    (h2 : trapezoid1_perimeter = 5)
    (h3 : trapezoid2_perimeter = 7)
    (h4 : trapezoid3_perimeter = 9) :
    trapezoid1_perimeter + trapezoid2_perimeter + trapezoid3_perimeter = original_perimeter + 10 := by
  sorry

end small_triangle_perimeter_l3933_393331


namespace complex_fraction_simplification_l3933_393354

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 5 * i) / (2 + 7 * i) = Complex.mk (-29/53) (-31/53) :=
by sorry

end complex_fraction_simplification_l3933_393354


namespace sum_of_zeros_is_14_l3933_393306

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the vertex of the original parabola
def vertex : ℝ × ℝ := (3, 4)

-- Define the transformed parabola after rotation and translation
def transformed_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 1

-- Define the zeros of the transformed parabola
def p : ℝ := 6
def q : ℝ := 8

theorem sum_of_zeros_is_14 : p + q = 14 := by
  sorry

end sum_of_zeros_is_14_l3933_393306


namespace circle_point_x_value_l3933_393305

theorem circle_point_x_value (x : ℝ) :
  let center : ℝ × ℝ := ((21 - (-3)) / 2 + (-3), 0)
  let radius : ℝ := (21 - (-3)) / 2
  (x - center.1) ^ 2 + (12 - center.2) ^ 2 = radius ^ 2 →
  x = 9 :=
by sorry

end circle_point_x_value_l3933_393305


namespace tangerines_remaining_l3933_393338

/-- The number of tangerines remaining in Yuna's house after Yoo-jung ate some. -/
def remaining_tangerines (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem stating that the number of remaining tangerines is 9. -/
theorem tangerines_remaining :
  remaining_tangerines 12 3 = 9 := by
  sorry

end tangerines_remaining_l3933_393338


namespace final_price_percentage_l3933_393367

/-- Given a suggested retail price, store discount, and additional discount,
    calculates the percentage of the original price paid. -/
def percentage_paid (suggested_retail_price : ℝ) (store_discount : ℝ) (additional_discount : ℝ) : ℝ :=
  (1 - store_discount) * (1 - additional_discount) * 100

/-- Theorem stating that with a 20% store discount and 10% additional discount,
    the final price paid is 72% of the suggested retail price. -/
theorem final_price_percentage (suggested_retail_price : ℝ) 
  (h1 : suggested_retail_price > 0)
  (h2 : store_discount = 0.2)
  (h3 : additional_discount = 0.1) :
  percentage_paid suggested_retail_price store_discount additional_discount = 72 := by
  sorry

end final_price_percentage_l3933_393367


namespace cos_120_degrees_l3933_393374

theorem cos_120_degrees : Real.cos (2 * π / 3) = -1/2 := by
  sorry

end cos_120_degrees_l3933_393374


namespace preimage_of_3_1_l3933_393378

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2*p.2, 2*p.1 - p.2)

-- Theorem statement
theorem preimage_of_3_1 :
  ∃ (p : ℝ × ℝ), f p = (3, 1) ∧ p = (1, 1) :=
sorry

end preimage_of_3_1_l3933_393378


namespace problem_solution_l3933_393397

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - (a+2)*x + 4

-- Define the function g
def g (m x : ℝ) : ℝ := m*x + 5 - 2*m

theorem problem_solution :
  -- Part 1
  (∀ a : ℝ, 
    (a < 2 → {x : ℝ | f a x ≤ 4 - 2*a} = {x : ℝ | a ≤ x ∧ x ≤ 2}) ∧
    (a = 2 → {x : ℝ | f a x ≤ 4 - 2*a} = {x : ℝ | x = 2}) ∧
    (a > 2 → {x : ℝ | f a x ≤ 4 - 2*a} = {x : ℝ | 2 ≤ x ∧ x ≤ a})) ∧
  -- Part 2
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 4 → f a x + a + 1 ≥ 0) → a ≤ 4) ∧
  -- Part 3
  (∀ m : ℝ, 
    (∀ x₁ : ℝ, x₁ ∈ Set.Icc 1 4 → 
      ∃ x₂ : ℝ, x₂ ∈ Set.Icc 1 4 ∧ f 2 x₁ = g m x₂) →
    m ≤ -5/2 ∨ m ≥ 5) :=
by sorry

end problem_solution_l3933_393397


namespace penny_remaining_money_l3933_393350

/-- Calculates the remaining money after Penny's shopping trip --/
def remaining_money (initial_amount : ℚ) (sock_price : ℚ) (sock_quantity : ℕ)
  (hat_price : ℚ) (hat_quantity : ℕ) (scarf_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_cost := sock_price * sock_quantity + hat_price * hat_quantity + scarf_price
  let discounted_cost := total_cost * (1 - discount_rate)
  initial_amount - discounted_cost

/-- Theorem stating that Penny has $14 left after her purchases --/
theorem penny_remaining_money :
  remaining_money 50 4 3 10 2 8 (1/10) = 14 := by
  sorry

end penny_remaining_money_l3933_393350


namespace arithmetic_sequence_common_difference_l3933_393396

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 10)
  (h_a12 : a 12 = 31) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry


end arithmetic_sequence_common_difference_l3933_393396


namespace trapezoid_AB_length_l3933_393387

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  AB : ℝ
  -- Length of side CD
  CD : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- Condition: The ratio of areas is 5:2
  area_ratio_condition : area_ratio = 5 / 2
  -- Condition: The sum of AB and CD is 280
  sum_sides : AB + CD = 280

/-- Theorem stating that under given conditions, AB = 200 -/
theorem trapezoid_AB_length (t : Trapezoid) : t.AB = 200 := by
  sorry


end trapezoid_AB_length_l3933_393387


namespace unique_point_perpendicular_segments_l3933_393351

/-- Given a non-zero real number α, there exists a unique point P in the coordinate plane
    such that for every line through P intersecting the parabola y = αx² in two distinct points A and B,
    the segments OA and OB are perpendicular (where O is the origin). -/
theorem unique_point_perpendicular_segments (α : ℝ) (h : α ≠ 0) :
  ∃! P : ℝ × ℝ, ∀ (A B : ℝ × ℝ),
    (A.2 = α * A.1^2) →
    (B.2 = α * B.1^2) →
    (∃ t : ℝ, A.1 + t * (P.1 - A.1) = B.1 ∧ A.2 + t * (P.2 - A.2) = B.2) →
    (A ≠ B) →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    P = (0, 1 / α) :=
sorry

end unique_point_perpendicular_segments_l3933_393351


namespace first_day_over_200_acorns_l3933_393384

/-- Represents the number of acorns Mark has on a given day -/
def acorns (k : ℕ) : ℕ := 5 * 5^k - 2 * k

/-- Represents the day of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Converts a natural number to a day of the week -/
def toDay (n : ℕ) : Day :=
  match n % 7 with
  | 0 => Day.Monday
  | 1 => Day.Tuesday
  | 2 => Day.Wednesday
  | 3 => Day.Thursday
  | 4 => Day.Friday
  | 5 => Day.Saturday
  | _ => Day.Sunday

theorem first_day_over_200_acorns :
  ∀ k : ℕ, k < 3 → acorns k ≤ 200 ∧
  acorns 3 > 200 ∧
  toDay 3 = Day.Thursday :=
sorry

end first_day_over_200_acorns_l3933_393384


namespace quadratic_inequality_range_l3933_393332

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x + 1/2 ≥ 0) → 
  (k > 0 ∧ k ≤ 4) :=
by sorry

end quadratic_inequality_range_l3933_393332


namespace hyperbolic_to_linear_transformation_l3933_393309

theorem hyperbolic_to_linear_transformation (x y a b : ℝ) (h : 1 / y = a + b / x) :
  1 / y = a + b * (1 / x) := by sorry

end hyperbolic_to_linear_transformation_l3933_393309


namespace fruit_sales_revenue_l3933_393372

theorem fruit_sales_revenue : 
  let original_lemon_price : ℝ := 8
  let original_grape_price : ℝ := 7
  let lemon_price_increase : ℝ := 4
  let grape_price_increase : ℝ := lemon_price_increase / 2
  let num_lemons : ℕ := 80
  let num_grapes : ℕ := 140
  let new_lemon_price : ℝ := original_lemon_price + lemon_price_increase
  let new_grape_price : ℝ := original_grape_price + grape_price_increase
  let total_revenue : ℝ := (↑num_lemons * new_lemon_price) + (↑num_grapes * new_grape_price)
  total_revenue = 2220 := by
sorry

end fruit_sales_revenue_l3933_393372


namespace trouser_sale_price_l3933_393315

theorem trouser_sale_price (original_price : ℝ) (discount_percentage : ℝ) 
  (h1 : original_price = 100)
  (h2 : discount_percentage = 70) : 
  original_price * (1 - discount_percentage / 100) = 30 := by
  sorry

end trouser_sale_price_l3933_393315


namespace cubic_equation_product_l3933_393337

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016)
  (h₄ : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃))
  (h₅ : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -671/336 := by
sorry

end cubic_equation_product_l3933_393337


namespace complement_A_in_U_l3933_393326

-- Define the sets U and A
def U : Set ℝ := Set.Icc 0 1
def A : Set ℝ := Set.Ico 0 1

-- State the theorem
theorem complement_A_in_U : (U \ A) = {1} := by sorry

end complement_A_in_U_l3933_393326


namespace min_focal_chord_length_is_2p_l3933_393344

/-- Represents a parabola defined by the equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- The minimum length of focal chords for a given parabola -/
def min_focal_chord_length (par : Parabola) : ℝ := 2 * par.p

/-- Theorem stating that the minimum length of focal chords is 2p -/
theorem min_focal_chord_length_is_2p (par : Parabola) :
  min_focal_chord_length par = 2 * par.p := by sorry

end min_focal_chord_length_is_2p_l3933_393344


namespace percentage_difference_l3933_393381

theorem percentage_difference (x : ℝ) : x = 35 → (0.8 * 170) - (x / 100 * 300) = 31 := by
  sorry

end percentage_difference_l3933_393381


namespace stratified_sample_is_proportional_l3933_393362

/-- Represents the number of students in each grade and the sample size -/
structure School :=
  (total : ℕ)
  (freshmen : ℕ)
  (sophomores : ℕ)
  (seniors : ℕ)
  (sample_size : ℕ)

/-- Represents the number of students sampled from each grade -/
structure Sample :=
  (freshmen : ℕ)
  (sophomores : ℕ)
  (seniors : ℕ)

/-- Calculates the proportional sample size for a given grade -/
def proportional_sample (grade_size : ℕ) (school : School) : ℕ :=
  (grade_size * school.sample_size) / school.total

/-- Checks if a sample is proportionally correct -/
def is_proportional_sample (school : School) (sample : Sample) : Prop :=
  sample.freshmen = proportional_sample school.freshmen school ∧
  sample.sophomores = proportional_sample school.sophomores school ∧
  sample.seniors = proportional_sample school.seniors school

/-- Theorem: The stratified sample is proportional for the given school -/
theorem stratified_sample_is_proportional (school : School)
  (h1 : school.total = 900)
  (h2 : school.freshmen = 300)
  (h3 : school.sophomores = 200)
  (h4 : school.seniors = 400)
  (h5 : school.sample_size = 45)
  (h6 : school.total = school.freshmen + school.sophomores + school.seniors) :
  is_proportional_sample school ⟨15, 10, 20⟩ := by
  sorry

end stratified_sample_is_proportional_l3933_393362


namespace more_freshmen_than_sophomores_l3933_393347

theorem more_freshmen_than_sophomores :
  ∀ (total juniors not_sophomores seniors freshmen sophomores : ℕ),
  total = 800 →
  juniors = (22 * total) / 100 →
  not_sophomores = (75 * total) / 100 →
  seniors = 160 →
  freshmen + sophomores + juniors + seniors = total →
  sophomores = total - not_sophomores →
  freshmen - sophomores = 64 :=
by sorry

end more_freshmen_than_sophomores_l3933_393347


namespace rectangular_prism_diagonal_l3933_393398

theorem rectangular_prism_diagonal (length width height : ℝ) :
  length = 24 ∧ width = 16 ∧ height = 12 →
  Real.sqrt (length^2 + width^2 + height^2) = 4 * Real.sqrt 61 := by
  sorry

end rectangular_prism_diagonal_l3933_393398


namespace inequality_proof_l3933_393330

theorem inequality_proof (m n : ℕ) (h : m < Real.sqrt 2 * n) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * n^2)) :=
by sorry

end inequality_proof_l3933_393330


namespace parabola_focus_l3933_393349

/-- The parabola defined by y = 2x² -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The theorem stating that (0, -1/8) is the focus of the parabola y = 2x² -/
theorem parabola_focus :
  ∃ (f : Focus), f.x = 0 ∧ f.y = -1/8 ∧
  ∀ (x y : ℝ), parabola x y →
    (x - f.x)^2 + (y - f.y)^2 = (y + 1/8)^2 :=
sorry

end parabola_focus_l3933_393349


namespace tan_arccos_three_fifths_l3933_393312

theorem tan_arccos_three_fifths : Real.tan (Real.arccos (3/5)) = 4/3 := by
  sorry

end tan_arccos_three_fifths_l3933_393312


namespace kelly_travel_days_l3933_393313

/-- Kelly's vacation details -/
structure VacationSchedule where
  total_days : ℕ
  initial_travel : ℕ
  grandparents : ℕ
  brother : ℕ
  to_sister_travel : ℕ
  sister : ℕ
  final_travel : ℕ

/-- The number of days Kelly spent traveling between her Grandparents' house and her brother's house -/
def days_between_grandparents_and_brother (schedule : VacationSchedule) : ℕ :=
  schedule.total_days - (schedule.initial_travel + schedule.grandparents + 
    schedule.brother + schedule.to_sister_travel + schedule.sister + schedule.final_travel)

/-- Theorem stating that Kelly spent 1 day traveling between her Grandparents' and brother's houses -/
theorem kelly_travel_days (schedule : VacationSchedule) 
  (h1 : schedule.total_days = 21)  -- Three weeks
  (h2 : schedule.initial_travel = 1)
  (h3 : schedule.grandparents = 5)
  (h4 : schedule.brother = 5)
  (h5 : schedule.to_sister_travel = 2)
  (h6 : schedule.sister = 5)
  (h7 : schedule.final_travel = 2) :
  days_between_grandparents_and_brother schedule = 1 := by
  sorry

end kelly_travel_days_l3933_393313


namespace complex_power_difference_l3933_393343

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^12 - (1 - i)^12 = 0 := by
  sorry

end complex_power_difference_l3933_393343


namespace cos_two_x_value_l3933_393329

theorem cos_two_x_value (x : ℝ) (h : Real.sin (-x) = Real.sqrt 3 / 2) : 
  Real.cos (2 * x) = -(1 / 2) := by
  sorry

end cos_two_x_value_l3933_393329


namespace exists_perpendicular_angles_not_equal_or_180_l3933_393333

/-- Two angles in 3D space with perpendicular sides -/
structure PerpendicularAngles where
  α : Real
  β : Real
  perp_sides : Bool

/-- Predicate for angles being equal or summing to 180° -/
def equal_or_sum_180 (angles : PerpendicularAngles) : Prop :=
  angles.α = angles.β ∨ angles.α + angles.β = 180

/-- Theorem stating the existence of perpendicular angles that don't satisfy the condition -/
theorem exists_perpendicular_angles_not_equal_or_180 :
  ∃ (angles : PerpendicularAngles), angles.perp_sides ∧ ¬(equal_or_sum_180 angles) :=
sorry

end exists_perpendicular_angles_not_equal_or_180_l3933_393333


namespace bridge_building_time_l3933_393355

/-- If 60 workers can build a bridge in 8 days, then 40 workers can build the same bridge in 12 days, given that all workers work at the same rate. -/
theorem bridge_building_time 
  (work : ℝ) -- Total amount of work required to build the bridge
  (rate : ℝ) -- Rate of work per worker per day
  (h1 : work = 60 * rate * 8) -- 60 workers complete the bridge in 8 days
  (h2 : rate > 0) -- Workers have a positive work rate
  : work = 40 * rate * 12 := by
  sorry

end bridge_building_time_l3933_393355


namespace simplify_fraction_product_l3933_393379

theorem simplify_fraction_product : (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end simplify_fraction_product_l3933_393379


namespace dress_price_difference_l3933_393390

theorem dress_price_difference (P : ℝ) (h : P - 0.15 * P = 68) :
  P - (68 + 0.25 * 68) = -5 := by
  sorry

end dress_price_difference_l3933_393390


namespace leftover_value_calculation_l3933_393361

/-- Calculates the value of leftover coins after making complete rolls --/
def leftover_value (quarters_per_roll dimes_per_roll toledo_quarters toledo_dimes brian_quarters brian_dimes : ℕ) : ℚ :=
  let total_quarters := toledo_quarters + brian_quarters
  let total_dimes := toledo_dimes + brian_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * (1 / 4) + (leftover_dimes : ℚ) * (1 / 10)

theorem leftover_value_calculation :
  leftover_value 30 50 95 172 137 290 = 17/10 := by
  sorry

end leftover_value_calculation_l3933_393361


namespace triangle_third_side_length_l3933_393304

theorem triangle_third_side_length (a b c : ℕ) (h1 : a = 5) (h2 : b = 2) (h3 : Odd c) : 
  (a + b > c ∧ a + c > b ∧ b + c > a) → c = 5 :=
by sorry

end triangle_third_side_length_l3933_393304
