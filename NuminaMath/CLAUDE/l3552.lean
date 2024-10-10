import Mathlib

namespace max_profit_at_18_profit_maximized_at_18_l3552_355285

-- Define the profit function
def profit (x : ℝ) : ℝ := -0.5 * x^2 + 18 * x - 20

-- Theorem statement
theorem max_profit_at_18 :
  ∃ (x_max : ℝ), x_max > 0 ∧ 
  (∀ (x : ℝ), x > 0 → profit x ≤ profit x_max) ∧
  x_max = 18 ∧ profit x_max = 142 := by
  sorry

-- Additional theorem to show that 18 is indeed the maximizer
theorem profit_maximized_at_18 :
  ∀ (x : ℝ), x > 0 → profit x ≤ profit 18 := by
  sorry

end max_profit_at_18_profit_maximized_at_18_l3552_355285


namespace complex_equation_solution_l3552_355219

theorem complex_equation_solution (z : ℂ) : 
  (Complex.I * z = 4 + 3 * Complex.I) → (z = 3 - 4 * Complex.I) :=
by sorry

end complex_equation_solution_l3552_355219


namespace total_tips_proof_l3552_355265

/-- Calculates the total tips earned over 3 days for a food truck --/
def total_tips (tips_per_customer : ℚ) (friday_customers : ℕ) (sunday_customers : ℕ) : ℚ :=
  let saturday_customers := 3 * friday_customers
  tips_per_customer * (friday_customers + saturday_customers + sunday_customers)

/-- Proves that the total tips earned over 3 days is $296.00 --/
theorem total_tips_proof :
  total_tips 2 28 36 = 296 := by
  sorry

end total_tips_proof_l3552_355265


namespace students_taking_both_languages_l3552_355226

theorem students_taking_both_languages (total : ℕ) (french : ℕ) (german : ℕ) (neither : ℕ) :
  total = 94 →
  french = 41 →
  german = 22 →
  neither = 40 →
  ∃ (both : ℕ), both = 9 ∧ total = french + german - both + neither :=
by sorry

end students_taking_both_languages_l3552_355226


namespace equation_solution_l3552_355220

theorem equation_solution : ∃! x : ℝ, -2 * x^2 = (4*x + 2) / (x + 4) :=
  sorry

end equation_solution_l3552_355220


namespace retail_price_calculation_l3552_355222

/-- Proves that the retail price of a machine is $120 given the specified conditions -/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  wholesale_price = 90 →
  discount_rate = 0.1 →
  profit_rate = 0.2 →
  ∃ (retail_price : ℝ),
    retail_price = 120 ∧
    wholesale_price * (1 + profit_rate) = retail_price * (1 - discount_rate) :=
by
  sorry

end retail_price_calculation_l3552_355222


namespace isabelle_bubble_bath_amount_l3552_355267

/-- Represents the configuration of a hotel --/
structure HotelConfig where
  double_suites : Nat
  couple_rooms : Nat
  single_rooms : Nat
  family_rooms : Nat
  double_suite_capacity : Nat
  couple_room_capacity : Nat
  single_room_capacity : Nat
  family_room_capacity : Nat
  bubble_bath_per_guest : Nat

/-- Calculates the total bubble bath needed for a given hotel configuration --/
def total_bubble_bath (config : HotelConfig) : Nat :=
  (config.double_suites * config.double_suite_capacity +
   config.couple_rooms * config.couple_room_capacity +
   config.single_rooms * config.single_room_capacity +
   config.family_rooms * config.family_room_capacity) *
  config.bubble_bath_per_guest

/-- The specific hotel configuration from the problem --/
def isabelle_hotel : HotelConfig :=
  { double_suites := 5
  , couple_rooms := 13
  , single_rooms := 14
  , family_rooms := 3
  , double_suite_capacity := 4
  , couple_room_capacity := 2
  , single_room_capacity := 1
  , family_room_capacity := 6
  , bubble_bath_per_guest := 25
  }

/-- Theorem stating that the total bubble bath needed for Isabelle's hotel is 1950 ml --/
theorem isabelle_bubble_bath_amount :
  total_bubble_bath isabelle_hotel = 1950 := by
  sorry

end isabelle_bubble_bath_amount_l3552_355267


namespace arrange_teachers_and_students_eq_24_l3552_355228

/-- The number of ways to arrange 2 teachers and 4 students in a row -/
def arrange_teachers_and_students : ℕ :=
  /- Two teachers must be in the middle -/
  let teacher_arrangements : ℕ := 2

  /- One specific student (A) cannot be at either end -/
  let student_A_positions : ℕ := 2

  /- Remaining three students can be arranged in the remaining positions -/
  let other_student_arrangements : ℕ := 6

  /- Total number of arrangements -/
  teacher_arrangements * student_A_positions * other_student_arrangements

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrange_teachers_and_students_eq_24 :
  arrange_teachers_and_students = 24 := by
  sorry

end arrange_teachers_and_students_eq_24_l3552_355228


namespace no_130_consecutive_numbers_with_900_divisors_l3552_355235

theorem no_130_consecutive_numbers_with_900_divisors :
  ¬ ∃ (n : ℕ), ∀ (k : ℕ), k ∈ Finset.range 130 →
    (Nat.divisors (n + k)).card = 900 :=
sorry

end no_130_consecutive_numbers_with_900_divisors_l3552_355235


namespace quadratic_binomial_square_l3552_355223

theorem quadratic_binomial_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 50*x + c = (x - a)^2) → c = 625 := by
  sorry

end quadratic_binomial_square_l3552_355223


namespace course_selection_theorem_l3552_355258

/-- The number of ways for students to select courses --/
def selectCourses (numCourses numStudents coursesPerStudent : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of selection methods --/
theorem course_selection_theorem :
  selectCourses 4 3 2 = 114 := by
  sorry

end course_selection_theorem_l3552_355258


namespace parallel_vectors_solution_l3552_355284

def a (x : ℝ) : Fin 3 → ℝ := ![x, 4, 1]
def b (y : ℝ) : Fin 3 → ℝ := ![-2, y, -1]

theorem parallel_vectors_solution (x y : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ a x = k • b y) → x = 2 ∧ y = -4 := by
  sorry

end parallel_vectors_solution_l3552_355284


namespace smallest_valid_purchase_l3552_355259

def is_valid_purchase (n : ℕ) : Prop :=
  n % 12 = 0 ∧ n % 10 = 0 ∧ n % 9 = 0 ∧ n % 8 = 0 ∧
  n % 18 = 0 ∧ n % 24 = 0 ∧ n % 20 = 0 ∧ n % 30 = 0

theorem smallest_valid_purchase :
  ∃ (n : ℕ), is_valid_purchase n ∧ ∀ (m : ℕ), is_valid_purchase m → n ≤ m :=
by
  sorry

end smallest_valid_purchase_l3552_355259


namespace union_M_N_l3552_355212

def M : Set ℝ := {x | x ≥ -1}
def N : Set ℝ := {x | 2 - x^2 ≥ 0}

theorem union_M_N : M ∪ N = {x : ℝ | x ≥ -Real.sqrt 2} := by sorry

end union_M_N_l3552_355212


namespace jasons_lawn_cutting_l3552_355268

/-- The number of lawns Jason can cut in 8 hours, given that it takes 30 minutes to cut one lawn -/
theorem jasons_lawn_cutting (time_per_lawn : ℕ) (total_time_hours : ℕ) : 
  time_per_lawn = 30 → total_time_hours = 8 → (total_time_hours * 60) / time_per_lawn = 16 := by
  sorry

end jasons_lawn_cutting_l3552_355268


namespace sum_of_special_numbers_l3552_355217

theorem sum_of_special_numbers :
  ∀ A B : ℤ,
  (A = -3 - (-5)) →
  (B = 2 + (-2)) →
  A + B = 2 := by
sorry

end sum_of_special_numbers_l3552_355217


namespace regular_implies_all_equal_regular_implies_rotational_symmetry_rotational_symmetry_implies_regular_regular_implies_topologically_regular_l3552_355266

-- Define a structure for a polyhedron
structure Polyhedron where
  vertices : Set Point
  edges : Set (Point × Point)
  faces : Set (Set Point)

-- Define properties of a regular polyhedron
def is_regular (P : Polyhedron) : Prop := sorry

-- Define equality of geometric elements
def all_elements_equal (P : Polyhedron) : Prop := sorry

-- Define rotational symmetry property
def has_rotational_symmetry (P : Polyhedron) : Prop := sorry

-- Define topological regularity
def is_topologically_regular (P : Polyhedron) : Prop := sorry

-- Theorem 1
theorem regular_implies_all_equal (P : Polyhedron) :
  is_regular P → all_elements_equal P := by sorry

-- Theorem 2
theorem regular_implies_rotational_symmetry (P : Polyhedron) :
  is_regular P → has_rotational_symmetry P := by sorry

-- Theorem 3
theorem rotational_symmetry_implies_regular (P : Polyhedron) :
  has_rotational_symmetry P → is_regular P := by sorry

-- Theorem 4
theorem regular_implies_topologically_regular (P : Polyhedron) :
  is_regular P → is_topologically_regular P := by sorry

end regular_implies_all_equal_regular_implies_rotational_symmetry_rotational_symmetry_implies_regular_regular_implies_topologically_regular_l3552_355266


namespace variance_of_transformed_data_l3552_355232

-- Define a type for our dataset
def Dataset := List ℝ

-- Define the variance of a dataset
noncomputable def variance (X : Dataset) : ℝ := sorry

-- Define the transformation function
def transform (X : Dataset) : Dataset := X.map (λ x => 2 * x - 5)

-- Theorem statement
theorem variance_of_transformed_data (X : Dataset) :
  variance X = 1/2 → variance (transform X) = 2 := by sorry

end variance_of_transformed_data_l3552_355232


namespace worker_b_completion_time_l3552_355277

/-- Given workers A, B, and C, and their work rates, prove that B can complete the work alone in 5 days -/
theorem worker_b_completion_time 
  (total_work : ℝ) 
  (rate_a : ℝ) (rate_b : ℝ) (rate_c : ℝ) 
  (time_a : ℝ) (time_b : ℝ) (time_c : ℝ) (time_abc : ℝ) 
  (h1 : rate_a = total_work / time_a)
  (h2 : rate_b = total_work / time_b)
  (h3 : rate_c = total_work / time_c)
  (h4 : rate_a + rate_b + rate_c = total_work / time_abc)
  (h5 : time_a = 4)
  (h6 : time_c = 20)
  (h7 : time_abc = 2)
  (h8 : total_work > 0) :
  time_b = 5 := by
  sorry

end worker_b_completion_time_l3552_355277


namespace bellas_bistro_purchase_l3552_355279

/-- The cost of a sandwich at Bella's Bistro -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda at Bella's Bistro -/
def soda_cost : ℕ := 1

/-- The number of sandwiches to be purchased -/
def num_sandwiches : ℕ := 6

/-- The number of sodas to be purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase at Bella's Bistro -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem bellas_bistro_purchase :
  total_cost = 29 := by
  sorry

end bellas_bistro_purchase_l3552_355279


namespace complex_magnitude_equals_one_l3552_355256

theorem complex_magnitude_equals_one : ∀ (z : ℂ), z = (2 * Complex.I + 1) / (Complex.I - 2) → Complex.abs z = 1 := by
  sorry

end complex_magnitude_equals_one_l3552_355256


namespace sawyer_coaching_fee_l3552_355295

/-- Calculate the total coaching fee for Sawyer --/
theorem sawyer_coaching_fee :
  let start_date : Nat := 1  -- January 1
  let end_date : Nat := 307  -- November 3
  let daily_fee : ℚ := 39
  let discount_days : Nat := 50
  let discount_rate : ℚ := 0.1

  let full_price_days : Nat := min discount_days (end_date - start_date + 1)
  let discounted_days : Nat := (end_date - start_date + 1) - full_price_days
  let discounted_fee : ℚ := daily_fee * (1 - discount_rate)

  let total_fee : ℚ := (full_price_days : ℚ) * daily_fee + (discounted_days : ℚ) * discounted_fee

  total_fee = 10967.7 := by
    sorry

end sawyer_coaching_fee_l3552_355295


namespace probability_no_three_consecutive_ones_sum_of_fraction_parts_l3552_355254

/-- Recurrence relation for sequences without three consecutive 1s -/
def b : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => b (n + 2) + b (n + 1) + b n

/-- The probability of a 12-element binary sequence not containing three consecutive 1s -/
theorem probability_no_three_consecutive_ones : 
  (b 12 : ℚ) / 2^12 = 927 / 4096 := by sorry

/-- The sum of numerator and denominator of the probability fraction -/
theorem sum_of_fraction_parts : 927 + 4096 = 5023 := by sorry

end probability_no_three_consecutive_ones_sum_of_fraction_parts_l3552_355254


namespace orange_distribution_ratio_l3552_355269

/-- Proves the ratio of oranges given to the brother to the total number of oranges --/
theorem orange_distribution_ratio :
  let total_oranges : ℕ := 12
  let friend_oranges : ℕ := 2
  ∀ brother_fraction : ℚ,
    (1 / 4 : ℚ) * ((1 : ℚ) - brother_fraction) * total_oranges = friend_oranges →
    (brother_fraction * total_oranges : ℚ) / total_oranges = 1 / 3 := by
  sorry

end orange_distribution_ratio_l3552_355269


namespace sam_goal_impossible_l3552_355286

theorem sam_goal_impossible (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (a_grades : ℕ) :
  total_quizzes = 60 →
  goal_percentage = 85 / 100 →
  completed_quizzes = 40 →
  a_grades = 26 →
  ¬∃ (remaining_non_a : ℕ), 
    (a_grades + (total_quizzes - completed_quizzes - remaining_non_a) : ℚ) / total_quizzes ≥ goal_percentage :=
by sorry

end sam_goal_impossible_l3552_355286


namespace pauls_rate_l3552_355221

/-- The number of cars Paul and Jack can service in a day -/
def total_cars (paul_rate : ℝ) : ℝ := 8 * (paul_rate + 3)

/-- Theorem stating Paul's rate of changing oil in cars per hour -/
theorem pauls_rate : ∃ (paul_rate : ℝ), total_cars paul_rate = 40 ∧ paul_rate = 2 := by
  sorry

end pauls_rate_l3552_355221


namespace anya_andrea_erasers_l3552_355208

theorem anya_andrea_erasers : 
  ∀ (andrea_erasers : ℕ) (anya_multiplier : ℕ),
    andrea_erasers = 4 →
    anya_multiplier = 4 →
    anya_multiplier * andrea_erasers - andrea_erasers = 12 := by
  sorry

end anya_andrea_erasers_l3552_355208


namespace candy_distribution_l3552_355270

/-- Calculates the number of candy pieces each student receives -/
def candy_per_student (total : ℕ) (reserved : ℕ) (students : ℕ) : ℕ :=
  (total - reserved) / students

/-- Proves that each student receives 6 pieces of candy -/
theorem candy_distribution (total : ℕ) (reserved : ℕ) (students : ℕ) 
  (h1 : total = 344) 
  (h2 : reserved = 56) 
  (h3 : students = 43) : 
  candy_per_student total reserved students = 6 := by
  sorry

end candy_distribution_l3552_355270


namespace triangle_problem_l3552_355233

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a > b →
  a = 5 →
  c = 6 →
  Real.sin B = 3/5 →
  (b = Real.sqrt 13 ∧ 
   Real.sin A = (3 * Real.sqrt 13) / 13) ∧
  Real.sin (2*A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end triangle_problem_l3552_355233


namespace inequality_solution_set_l3552_355288

-- Define the inequality
def inequality (x : ℝ) : Prop := |2*x - 1| < 1

-- Define the solution set
def solution_set : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end inequality_solution_set_l3552_355288


namespace david_math_homework_time_l3552_355280

/-- Given David's homework times, prove he spent 15 minutes on math. -/
theorem david_math_homework_time :
  ∀ (total_time spelling_time reading_time math_time : ℕ),
    total_time = 60 →
    spelling_time = 18 →
    reading_time = 27 →
    math_time = total_time - spelling_time - reading_time →
    math_time = 15 := by
  sorry

end david_math_homework_time_l3552_355280


namespace triangle_area_approximation_l3552_355260

theorem triangle_area_approximation (α β : Real) (k l m : Real) :
  α = π / 6 →
  β = π / 4 →
  k = 3 →
  l = 2 →
  m = 4 →
  let γ : Real := π - α - β
  let S := ((k * Real.sin α + l * Real.sin β + m * Real.sin γ) ^ 2) / (2 * Real.sin α * Real.sin β * Real.sin γ)
  |S - 67| < 0.5 := by
sorry

end triangle_area_approximation_l3552_355260


namespace intersection_count_l3552_355227

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 15

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 10

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := (num_x_points.choose 2) * (num_y_points.choose 2)

/-- Theorem stating the maximum number of intersection points -/
theorem intersection_count :
  max_intersections = 4725 := by sorry

end intersection_count_l3552_355227


namespace park_trees_l3552_355206

theorem park_trees (blackbirds_per_tree : ℕ) (magpies : ℕ) (total_birds : ℕ) :
  blackbirds_per_tree = 3 →
  magpies = 13 →
  total_birds = 34 →
  ∃ trees : ℕ, trees * blackbirds_per_tree + magpies = total_birds ∧ trees = 7 :=
by
  sorry

end park_trees_l3552_355206


namespace tree_height_difference_l3552_355213

/-- Given three trees with specific height relationships, prove the difference between half the height of the tallest tree and the height of the middle-sized tree. -/
theorem tree_height_difference (tallest middle smallest : ℝ) : 
  tallest = 108 →
  smallest = 12 →
  smallest = (1/4) * middle →
  (tallest / 2) - middle = 6 :=
by
  sorry

end tree_height_difference_l3552_355213


namespace trajectory_circle_fixed_points_l3552_355289

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y^2 = 4*x) ∨ (x < 0 ∧ y = 0)

/-- The distance condition for point M -/
def distance_condition (x y : ℝ) : Prop :=
  ((x - 1)^2 + y^2)^(1/2) = x + 1

/-- The line passing through F(1,0) and intersecting the trajectory -/
def intersecting_line (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y + 1

/-- The circle with diameter AB -/
def circle_AB (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 + 4 * y = 4

/-- The main theorem -/
theorem trajectory_circle_fixed_points :
  ∀ x y m,
  trajectory x y →
  distance_condition x y →
  intersecting_line m x y →
  (circle_AB (-1) 0 ∧ circle_AB 3 0) :=
by sorry

end trajectory_circle_fixed_points_l3552_355289


namespace largest_number_l3552_355276

def A : ℕ := 27

def B (A : ℕ) : ℕ := A + 7

def C (B : ℕ) : ℕ := B - 9

def D (C : ℕ) : ℕ := 2 * C

theorem largest_number (A B C D : ℕ) (hA : A = 27) (hB : B = A + 7) (hC : C = B - 9) (hD : D = 2 * C) :
  D = max A (max B (max C D)) := by
  sorry

end largest_number_l3552_355276


namespace volume_ratio_l3552_355263

theorem volume_ratio (A B C : ℝ) 
  (h1 : 2 * A = B + C) 
  (h2 : 5 * B = A + C) : 
  C / (A + B) = 1 := by sorry

end volume_ratio_l3552_355263


namespace reduce_piles_to_zero_reduce_table_to_zero_l3552_355234

/-- Represents the state of three piles of stones -/
structure ThreePiles :=
  (pile1 pile2 pile3 : Nat)

/-- Represents the state of an 8x5 table of natural numbers -/
def Table := Fin 8 → Fin 5 → Nat

/-- Allowed operations on three piles of stones -/
inductive PileOperation
  | removeOne : PileOperation
  | doubleOne : Fin 3 → PileOperation

/-- Allowed operations on the table -/
inductive TableOperation
  | doubleColumn : Fin 5 → TableOperation
  | subtractRow : Fin 8 → TableOperation

/-- Applies a pile operation to a ThreePiles state -/
def applyPileOp (s : ThreePiles) (op : PileOperation) : ThreePiles :=
  match op with
  | PileOperation.removeOne => ⟨s.pile1 - 1, s.pile2 - 1, s.pile3 - 1⟩
  | PileOperation.doubleOne i =>
      match i with
      | 0 => ⟨s.pile1 * 2, s.pile2, s.pile3⟩
      | 1 => ⟨s.pile1, s.pile2 * 2, s.pile3⟩
      | 2 => ⟨s.pile1, s.pile2, s.pile3 * 2⟩

/-- Applies a table operation to a Table state -/
def applyTableOp (t : Table) (op : TableOperation) : Table :=
  match op with
  | TableOperation.doubleColumn j => fun i k => if k = j then t i k * 2 else t i k
  | TableOperation.subtractRow i => fun j k => if j = i then t j k - 1 else t j k

/-- Theorem stating that any ThreePiles state can be reduced to zero -/
theorem reduce_piles_to_zero (s : ThreePiles) :
  ∃ (ops : List PileOperation), (ops.foldl applyPileOp s).pile1 = 0 ∧
                                (ops.foldl applyPileOp s).pile2 = 0 ∧
                                (ops.foldl applyPileOp s).pile3 = 0 :=
  sorry

/-- Theorem stating that any Table state can be reduced to zero -/
theorem reduce_table_to_zero (t : Table) :
  ∃ (ops : List TableOperation), ∀ i j, (ops.foldl applyTableOp t) i j = 0 :=
  sorry

end reduce_piles_to_zero_reduce_table_to_zero_l3552_355234


namespace riku_stickers_comparison_l3552_355287

/-- The number of stickers Kristoff has -/
def kristoff_stickers : ℕ := 85

/-- The number of stickers Riku has -/
def riku_stickers : ℕ := 2210

/-- The number of times Riku has more stickers than Kristoff -/
def times_more_stickers : ℚ := riku_stickers / kristoff_stickers

/-- Theorem stating that Riku has 26 times more stickers than Kristoff -/
theorem riku_stickers_comparison : times_more_stickers = 26 := by
  sorry

end riku_stickers_comparison_l3552_355287


namespace linear_function_value_l3552_355262

theorem linear_function_value (k b : ℝ) :
  ((-1 : ℝ) * k + b = 1) →
  (2 * k + b = -2) →
  (1 : ℝ) * k + b = -1 := by
sorry

end linear_function_value_l3552_355262


namespace pipe_filling_time_l3552_355224

/-- Given two pipes A and B that fill a tank, where:
    - Pipe A fills the tank in t minutes
    - Pipe B fills the tank 3 times as fast as Pipe A
    - Both pipes together fill the tank in 3 minutes
    Then, Pipe A takes 12 minutes to fill the tank alone. -/
theorem pipe_filling_time (t : ℝ) 
  (hA : t > 0)  -- Pipe A's filling time is positive
  (hB : t / 3 > 0)  -- Pipe B's filling time is positive
  (h_both : 1 / t + 1 / (t / 3) = 1 / 3)  -- Combined filling rate equals 1/3
  : t = 12 := by
  sorry


end pipe_filling_time_l3552_355224


namespace hexagonangulo_19_requires_59_l3552_355207

/-- A hexagonângulo is a shape formed by triangles -/
structure Hexagonangulo where
  triangles : ℕ
  perimeter : ℕ

/-- Calculates the number of unit triangles needed to form a triangle of given side length -/
def trianglesInLargerTriangle (side : ℕ) : ℕ := side^2

/-- Constructs a hexagonângulo with given perimeter using unit triangles -/
def constructHexagonangulo (p : ℕ) : Hexagonangulo :=
  { triangles := 
      4 * trianglesInLargerTriangle 2 + 
      3 * trianglesInLargerTriangle 3 + 
      1 * trianglesInLargerTriangle 4,
    perimeter := p }

/-- Theorem: A hexagonângulo with perimeter 19 requires 59 unit triangles -/
theorem hexagonangulo_19_requires_59 : 
  (constructHexagonangulo 19).triangles = 59 := by sorry

end hexagonangulo_19_requires_59_l3552_355207


namespace simplify_fraction_l3552_355297

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end simplify_fraction_l3552_355297


namespace cube_inequality_iff_inequality_l3552_355274

theorem cube_inequality_iff_inequality (a b : ℝ) : a^3 > b^3 ↔ a > b := by sorry

end cube_inequality_iff_inequality_l3552_355274


namespace gold_coin_distribution_l3552_355215

theorem gold_coin_distribution (x y : ℕ) (h1 : x + y = 25) :
  ∃ k : ℕ, x^2 - y^2 = k * (x - y) → k = 25 := by
sorry

end gold_coin_distribution_l3552_355215


namespace subtract_negatives_l3552_355261

theorem subtract_negatives : (-7) - (-5) = -2 := by
  sorry

end subtract_negatives_l3552_355261


namespace h_min_neg_l3552_355209

-- Define the functions f, g, and h
variable (f g : ℝ → ℝ)
variable (a b : ℝ)

def h (x : ℝ) := a * f x + b * g x + 2

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_odd : ∀ x, g (-x) = -g x

-- Define the maximum value of h on (0, +∞)
axiom h_max : ∀ x > 0, h x ≤ 5

-- State the theorem to be proved
theorem h_min_neg : (∀ x < 0, h x ≥ -1) := by sorry

end h_min_neg_l3552_355209


namespace product_abcd_is_zero_l3552_355282

theorem product_abcd_is_zero 
  (a b c d : ℤ) 
  (eq1 : 2*a + 3*b + 5*c + 7*d = 34)
  (eq2 : 3*(d + c) = b)
  (eq3 : 3*b + c = a)
  (eq4 : c - 1 = d) :
  a * b * c * d = 0 :=
by sorry

end product_abcd_is_zero_l3552_355282


namespace wire_length_equals_49_l3552_355281

/-- The total length of a wire cut into two pieces forming a square and a regular octagon -/
def wire_length (square_side : ℝ) : ℝ :=
  4 * square_side

theorem wire_length_equals_49 (square_side : ℝ) (h1 : square_side = 7) :
  let octagon_side := (3 * wire_length square_side) / (8 * 4)
  let square_area := square_side ^ 2
  let octagon_area := 2 * (1 + Real.sqrt 2) * octagon_side ^ 2
  square_area = octagon_area →
  wire_length square_side = 49 := by sorry

end wire_length_equals_49_l3552_355281


namespace substitution_elimination_l3552_355292

theorem substitution_elimination (x y : ℝ) : 
  (y = x - 5 ∧ 3*x - y = 8) → (3*x - x + 5 = 8) := by
  sorry

end substitution_elimination_l3552_355292


namespace new_rectangle_area_l3552_355296

theorem new_rectangle_area (x y : ℝ) (h : 0 < x ∧ x ≤ y) :
  let base := Real.sqrt (x^2 + y^2) + y
  let altitude := Real.sqrt (x^2 + y^2) - y
  base * altitude = x^2 := by
  sorry

end new_rectangle_area_l3552_355296


namespace sum_x_y_equals_eight_l3552_355245

theorem sum_x_y_equals_eight (x y : ℝ) 
  (h1 : |x| + x + y = 14)
  (h2 : x + |y| - y = 10)
  (h3 : |x| - |y| + x - y = 8) :
  x + y = 8 := by
  sorry

end sum_x_y_equals_eight_l3552_355245


namespace max_pieces_is_nine_l3552_355299

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 15

/-- The size of a small piece in inches -/
def small_piece_size : ℕ := 5

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := (large_cake_size * large_cake_size) / (small_piece_size * small_piece_size)

theorem max_pieces_is_nine : max_pieces = 9 := by
  sorry

end max_pieces_is_nine_l3552_355299


namespace sugar_for_muffins_l3552_355201

/-- Given a recipe that requires 3 cups of sugar for 24 muffins,
    calculate the number of cups of sugar needed for 72 muffins. -/
theorem sugar_for_muffins (recipe_muffins : ℕ) (recipe_sugar : ℕ) (target_muffins : ℕ) :
  recipe_muffins = 24 →
  recipe_sugar = 3 →
  target_muffins = 72 →
  (target_muffins * recipe_sugar) / recipe_muffins = 9 :=
by
  sorry

#check sugar_for_muffins

end sugar_for_muffins_l3552_355201


namespace min_gumballs_for_four_same_color_is_13_l3552_355248

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- The minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFourSameColor (machine : GumballMachine) : Nat :=
  13

/-- Theorem stating that for the given gumball machine configuration,
    the minimum number of gumballs needed to guarantee four of the same color is 13 -/
theorem min_gumballs_for_four_same_color_is_13 (machine : GumballMachine)
    (h : machine = { red := 12, white := 15, blue := 10, green := 7 }) :
    minGumballsForFourSameColor machine = 13 := by
  sorry

end min_gumballs_for_four_same_color_is_13_l3552_355248


namespace triangle_inequalities_l3552_355214

theorem triangle_inequalities (A B C : ℝ) (h_triangle : A + B + C = π) (h_obtuse : A > π/2) :
  (1 + Real.sin (A/2) + Real.sin (B/2) + Real.sin (C/2) < Real.cos (A/2) + Real.cos (B/2) + Real.cos (C/2)) ∧
  (1 - Real.cos A + Real.sin B + Real.sin C < Real.sin A + Real.cos B + Real.cos C) :=
by sorry

end triangle_inequalities_l3552_355214


namespace sum_of_primes_less_than_20_is_77_l3552_355275

def is_prime (n : ℕ) : Prop := sorry

def sum_of_primes_less_than_20 : ℕ := sorry

theorem sum_of_primes_less_than_20_is_77 :
  sum_of_primes_less_than_20 = 77 := by sorry

end sum_of_primes_less_than_20_is_77_l3552_355275


namespace rational_solutions_quadratic_l3552_355211

theorem rational_solutions_quadratic (k : ℕ+) :
  (∃ x : ℚ, 2 * k * x^2 + 36 * x + 3 * k = 0) ↔ k = 6 := by
  sorry

end rational_solutions_quadratic_l3552_355211


namespace min_value_theorem_l3552_355210

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 8) :
  (2/x + 3/y) ≥ 25/8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 8 ∧ 2/x₀ + 3/y₀ = 25/8 := by
  sorry

end min_value_theorem_l3552_355210


namespace scooter_profit_l3552_355290

theorem scooter_profit (original_cost repair_cost profit_percentage : ℝ) : 
  repair_cost = 0.1 * original_cost → 
  repair_cost = 500 → 
  profit_percentage = 0.2 → 
  original_cost * profit_percentage = 1000 := by
sorry

end scooter_profit_l3552_355290


namespace arcsin_of_one_l3552_355264

theorem arcsin_of_one (π : Real) : Real.arcsin 1 = π / 2 := by
  sorry

end arcsin_of_one_l3552_355264


namespace roots_reciprocal_sum_l3552_355218

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 5 = 0 → x₂^2 - 2*x₂ - 5 = 0 → 1/x₁ + 1/x₂ = -2/5 := by
  sorry

end roots_reciprocal_sum_l3552_355218


namespace systematic_sampling_result_l3552_355239

/-- Represents a systematic sampling result -/
structure SystematicSample where
  first : Nat
  interval : Nat
  size : Nat

/-- Generates a sequence of numbers using systematic sampling -/
def generateSequence (sample : SystematicSample) : List Nat :=
  List.range sample.size |>.map (fun i => sample.first + i * sample.interval)

/-- Checks if a sequence is within the given range -/
def isWithinRange (seq : List Nat) (maxVal : Nat) : Prop :=
  seq.all (· ≤ maxVal)

theorem systematic_sampling_result :
  let classSize : Nat := 50
  let sampleSize : Nat := 5
  let result : List Nat := [5, 15, 25, 35, 45]
  ∃ (sample : SystematicSample),
    sample.size = sampleSize ∧
    sample.interval = classSize / sampleSize ∧
    generateSequence sample = result ∧
    isWithinRange result classSize :=
by sorry

end systematic_sampling_result_l3552_355239


namespace quadratic_root_relation_l3552_355251

/-- Given two quadratic equations, where the roots of the second are each three less than
    the roots of the first, this theorem proves that the constant term of the second
    equation is 3.5. -/
theorem quadratic_root_relation (d : ℝ) :
  (∃ r s : ℝ, r + s = 2 ∧ r * s = 1/2 ∧ 
   ∀ x : ℝ, 4 * x^2 - 8 * x + 2 = 0 ↔ (x = r ∨ x = s)) →
  (∃ e : ℝ, ∀ x : ℝ, x^2 + d * x + e = 0 ↔ (x = r - 3 ∨ x = s - 3)) →
  ∃ e : ℝ, e = 3.5 := by
sorry

end quadratic_root_relation_l3552_355251


namespace root_sum_theorem_l3552_355205

theorem root_sum_theorem (a b : ℝ) 
  (h1 : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1^2 + a*r1 + b = 0) ∧ (r2^2 + a*r2 + b = 0))
  (h2 : ∃ s1 s2 : ℝ, s1 ≠ s2 ∧ (s1^2 + b*s1 + a = 0) ∧ (s2^2 + b*s2 + a = 0))
  (h3 : ∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ 
    ((t1^2 + a*t1 + b) * (t1^2 + b*t1 + a) = 0) ∧
    ((t2^2 + a*t2 + b) * (t2^2 + b*t2 + a) = 0) ∧
    ((t3^2 + a*t3 + b) * (t3^2 + b*t3 + a) = 0)) :
  t1 + t2 + t3 = -2 := by sorry

end root_sum_theorem_l3552_355205


namespace two_different_pitchers_l3552_355216

-- Define the type for pitchers
structure Pitcher :=
  (shape : ℕ)
  (color : ℕ)

-- Define the theorem
theorem two_different_pitchers 
  (pitchers : Set Pitcher) 
  (h1 : ∃ (a b : Pitcher), a ∈ pitchers ∧ b ∈ pitchers ∧ a.shape ≠ b.shape)
  (h2 : ∃ (c d : Pitcher), c ∈ pitchers ∧ d ∈ pitchers ∧ c.color ≠ d.color) :
  ∃ (x y : Pitcher), x ∈ pitchers ∧ y ∈ pitchers ∧ x.shape ≠ y.shape ∧ x.color ≠ y.color :=
sorry

end two_different_pitchers_l3552_355216


namespace cricket_team_size_l3552_355257

theorem cricket_team_size :
  ∀ (n : ℕ) (initial_avg final_avg : ℝ),
  initial_avg = 29 →
  final_avg = 26 →
  (n * final_avg = (n - 2) * (initial_avg - 1) + (initial_avg + 3) + initial_avg) →
  n = 5 :=
by
  sorry

end cricket_team_size_l3552_355257


namespace monitor_length_is_14_l3552_355250

/-- Represents the dimensions of a rectangular monitor. -/
structure Monitor where
  width : ℝ
  length : ℝ
  circumference : ℝ

/-- The circumference of a rectangle is equal to twice the sum of its length and width. -/
def circumference_formula (m : Monitor) : Prop :=
  m.circumference = 2 * (m.length + m.width)

/-- Theorem: A monitor with width 9 cm and circumference 46 cm has a length of 14 cm. -/
theorem monitor_length_is_14 :
  ∃ (m : Monitor), m.width = 9 ∧ m.circumference = 46 ∧ circumference_formula m → m.length = 14 :=
by
  sorry


end monitor_length_is_14_l3552_355250


namespace coordinates_of_P_l3552_355241

-- Define points M and N
def M : ℝ × ℝ := (3, 2)
def N : ℝ × ℝ := (-5, -5)

-- Define vector from M to N
def MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

-- Define point P
def P : ℝ × ℝ := (x, y) where
  x : ℝ := sorry
  y : ℝ := sorry

-- Define vector from M to P
def MP : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)

-- Theorem statement
theorem coordinates_of_P :
  MP = (1/2 : ℝ) • MN → P = (-1, -3/2) := by sorry

end coordinates_of_P_l3552_355241


namespace factorization_3y_squared_minus_12_l3552_355271

theorem factorization_3y_squared_minus_12 (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) := by
  sorry

end factorization_3y_squared_minus_12_l3552_355271


namespace function_determination_l3552_355278

/-- Given a function f(x) = a^x + k, if f(1) = 3 and f(0) = 2, then f(x) = 2^x + 1 -/
theorem function_determination (a k : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a^x + k) 
  (h2 : f 1 = 3) 
  (h3 : f 0 = 2) : 
  ∀ x, f x = 2^x + 1 := by
sorry

end function_determination_l3552_355278


namespace ice_water_masses_l3552_355252

/-- Proof of initial ice and water masses in a cylindrical vessel --/
theorem ice_water_masses
  (S : ℝ) (ρw ρi : ℝ) (hf Δh : ℝ)
  (h_S : S = 15)
  (h_ρw : ρw = 1)
  (h_ρi : ρi = 0.9)
  (h_hf : hf = 115)
  (h_Δh : Δh = 5) :
  ∃ (m_ice m_water : ℝ),
    m_ice = 675 ∧
    m_water = 1050 ∧
    m_ice / ρi - m_ice / ρw = S * Δh ∧
    m_water = ρw * S * hf - m_ice :=
by sorry

end ice_water_masses_l3552_355252


namespace distance_sum_constant_l3552_355230

theorem distance_sum_constant (a b x : ℝ) (h : 0 ≤ x ∧ x ≤ 50) :
  |x - a| + |x - b| = 50 :=
by
  sorry

#check distance_sum_constant

end distance_sum_constant_l3552_355230


namespace sugar_recipe_reduction_l3552_355225

theorem sugar_recipe_reduction :
  let original_recipe : ℚ := 27/4  -- 6 3/4 cups
  let reduced_recipe : ℚ := (1/3) * original_recipe
  reduced_recipe = 9/4  -- 2 1/4 cups
  := by sorry

end sugar_recipe_reduction_l3552_355225


namespace pony_discount_rate_l3552_355255

/-- Represents the discount rate for Fox jeans -/
def F : ℝ := sorry

/-- Represents the discount rate for Pony jeans -/
def P : ℝ := sorry

/-- Regular price of Fox jeans -/
def fox_price : ℝ := 15

/-- Regular price of Pony jeans -/
def pony_price : ℝ := 20

/-- Number of Fox jeans purchased -/
def fox_count : ℕ := 3

/-- Number of Pony jeans purchased -/
def pony_count : ℕ := 2

/-- Total savings -/
def total_savings : ℝ := 9

/-- Sum of discount rates -/
def discount_sum : ℝ := 22

theorem pony_discount_rate :
  F + P = discount_sum ∧
  (fox_count * fox_price * F / 100 + pony_count * pony_price * P / 100 = total_savings) →
  P = 18 := by sorry

end pony_discount_rate_l3552_355255


namespace augmented_matrix_of_system_l3552_355283

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * x + 5 * y + 6 = 0
def equation2 (x y : ℝ) : Prop := 4 * x - 3 * y - 7 = 0

-- Define the augmented matrix
def augmented_matrix : Matrix (Fin 2) (Fin 3) ℝ :=
  !![3, 5, -6;
     4, -3, 7]

-- Theorem statement
theorem augmented_matrix_of_system :
  ∀ (x y : ℝ), equation1 x y ∧ equation2 x y →
  augmented_matrix = !![3, 5, -6; 4, -3, 7] := by
  sorry

end augmented_matrix_of_system_l3552_355283


namespace remainder_polynomial_l3552_355291

theorem remainder_polynomial (p : ℝ → ℝ) (h1 : p 2 = 7) (h2 : p 5 = 8) (h3 : p 0 = 6) :
  ∃ q : ℝ → ℝ, ∀ x, p x = q x * (x - 2) * (x - 5) + ((1/3) * x + 19/3) := by
sorry

end remainder_polynomial_l3552_355291


namespace problem_solution_l3552_355244

theorem problem_solution (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  w = 0.5 := by
  sorry

end problem_solution_l3552_355244


namespace sum_of_absolute_values_l3552_355294

theorem sum_of_absolute_values (a b : ℤ) : 
  (abs a = 2023) → (abs b = 2022) → (a > b) → ((a + b = 1) ∨ (a + b = 4045)) :=
by
  sorry

end sum_of_absolute_values_l3552_355294


namespace exponent_division_l3552_355237

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^4 / x = x^3 := by
  sorry

end exponent_division_l3552_355237


namespace gcf_75_45_l3552_355202

theorem gcf_75_45 : Nat.gcd 75 45 = 15 := by
  sorry

end gcf_75_45_l3552_355202


namespace triangle_inequality_l3552_355240

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a + b) * Real.sqrt (a * b) + (a + c) * Real.sqrt (a * c) + (b + c) * Real.sqrt (b * c) ≥ (a + b + c)^2 / 2 := by
  sorry

end triangle_inequality_l3552_355240


namespace girl_boy_ratio_l3552_355247

/-- Represents the number of students in the class -/
def total_students : ℕ := 28

/-- Represents the difference between the number of girls and boys -/
def girl_boy_difference : ℕ := 4

/-- Theorem stating that the ratio of girls to boys is 4:3 -/
theorem girl_boy_ratio :
  ∃ (girls boys : ℕ),
    girls + boys = total_students ∧
    girls = boys + girl_boy_difference ∧
    girls * 3 = boys * 4 :=
by sorry

end girl_boy_ratio_l3552_355247


namespace diamond_value_l3552_355293

/-- Given a digit d, this function returns the value of d3 in base 5 -/
def base5_value (d : ℕ) : ℕ := d * 5 + 3

/-- Given a digit d, this function returns the value of d2 in base 6 -/
def base6_value (d : ℕ) : ℕ := d * 6 + 2

/-- The theorem states that the digit d satisfying d3 in base 5 equals d2 in base 6 is 1 -/
theorem diamond_value :
  ∃ (d : ℕ), d < 10 ∧ base5_value d = base6_value d ∧ d = 1 :=
sorry

end diamond_value_l3552_355293


namespace apple_ratio_is_one_to_two_l3552_355249

/-- Represents the number of golden delicious apples needed for one pint of cider -/
def golden_delicious_per_pint : ℕ := 20

/-- Represents the number of pink lady apples needed for one pint of cider -/
def pink_lady_per_pint : ℕ := 40

/-- Represents the number of farmhands -/
def num_farmhands : ℕ := 6

/-- Represents the number of apples a farmhand can pick per hour -/
def apples_per_hour : ℕ := 240

/-- Represents the number of hours worked -/
def hours_worked : ℕ := 5

/-- Represents the number of pints of cider that can be made -/
def pints_of_cider : ℕ := 120

/-- Theorem stating that the ratio of golden delicious apples to pink lady apples gathered is 1:2 -/
theorem apple_ratio_is_one_to_two :
  (golden_delicious_per_pint * pints_of_cider) / (pink_lady_per_pint * pints_of_cider) = 1 / 2 :=
by sorry

end apple_ratio_is_one_to_two_l3552_355249


namespace cafe_tables_theorem_l3552_355204

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- Calculates the number of tables needed given the number of people and people per table --/
def tablesNeeded (people : Nat) (peoplePerTable : Nat) : Nat :=
  people / peoplePerTable

theorem cafe_tables_theorem (seatingCapacity : Nat) (peoplePerTable : Nat) :
  seatingCapacity = 312 ∧ peoplePerTable = 3 →
  tablesNeeded (base7ToBase10 seatingCapacity) peoplePerTable = 52 := by
  sorry

#eval base7ToBase10 312  -- Should output 156
#eval tablesNeeded 156 3  -- Should output 52

end cafe_tables_theorem_l3552_355204


namespace fraction_comparison_l3552_355246

theorem fraction_comparison (a b c d : ℝ) (h1 : a/b < c/d) (h2 : b > d) (h3 : d > 0) :
  (a+c)/(b+d) < (1/2) * (a/b + c/d) := by
  sorry

end fraction_comparison_l3552_355246


namespace consecutive_binomial_ratio_l3552_355231

theorem consecutive_binomial_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k+1) : ℚ) = 1/3 ∧
  (n.choose (k+1) : ℚ) / (n.choose (k+2) : ℚ) = 3/5 →
  n + k = 8 := by
sorry

end consecutive_binomial_ratio_l3552_355231


namespace max_points_is_168_l3552_355298

/-- Represents the number of cards of each color chosen by Vasya -/
structure CardChoice where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total points for a given card choice -/
def calculatePoints (choice : CardChoice) : ℕ :=
  choice.red + 2 * choice.red * choice.blue + 3 * choice.blue * choice.yellow

/-- Theorem: The maximum number of points Vasya can earn is 168 -/
theorem max_points_is_168 : 
  ∃ (choice : CardChoice), 
    choice.red + choice.blue + choice.yellow = 15 ∧ 
    choice.red ≤ 15 ∧ choice.blue ≤ 15 ∧ choice.yellow ≤ 15 ∧
    calculatePoints choice = 168 ∧
    ∀ (other : CardChoice), 
      other.red + other.blue + other.yellow = 15 → 
      other.red ≤ 15 ∧ other.blue ≤ 15 ∧ other.yellow ≤ 15 →
      calculatePoints other ≤ 168 := by
  sorry


end max_points_is_168_l3552_355298


namespace charlotte_overall_score_l3552_355272

/-- Charlotte's test scores -/
def charlotte_scores : Fin 3 → ℚ
  | 0 => 60 / 100
  | 1 => 75 / 100
  | 2 => 85 / 100

/-- Number of problems in each test -/
def test_problems : Fin 3 → ℕ
  | 0 => 15
  | 1 => 20
  | 2 => 25

/-- Total number of problems in the combined test -/
def total_problems : ℕ := 60

/-- Charlotte's overall score on the combined test -/
def overall_score : ℚ := (charlotte_scores 0 * test_problems 0 +
                          charlotte_scores 1 * test_problems 1 +
                          charlotte_scores 2 * test_problems 2) / total_problems

theorem charlotte_overall_score :
  overall_score = 75 / 100 := by sorry

end charlotte_overall_score_l3552_355272


namespace third_range_is_56_prove_third_range_l3552_355253

/-- The minimum possible range of scores -/
def min_range : ℕ := 30

/-- The first given range -/
def range1 : ℕ := 18

/-- The second given range -/
def range2 : ℕ := 26

/-- The theorem stating that the third range is 56 -/
theorem third_range_is_56 : ℕ :=
  min_range + range2

/-- The main theorem to prove -/
theorem prove_third_range :
  third_range_is_56 = 56 :=
by sorry

end third_range_is_56_prove_third_range_l3552_355253


namespace increase_decrease_calculation_l3552_355200

theorem increase_decrease_calculation (initial : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) : 
  initial = 80 → 
  increase_percent = 150 → 
  decrease_percent = 20 → 
  (initial * (1 + increase_percent / 100)) * (1 - decrease_percent / 100) = 160 := by
sorry

end increase_decrease_calculation_l3552_355200


namespace mango_rate_is_75_l3552_355273

/-- The rate of mangoes per kg given the purchase details -/
def mango_rate (apple_weight : ℕ) (apple_rate : ℕ) (mango_weight : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - apple_weight * apple_rate) / mango_weight

/-- Theorem stating that the rate of mangoes is 75 per kg -/
theorem mango_rate_is_75 :
  mango_rate 8 70 9 1235 = 75 := by
  sorry

#eval mango_rate 8 70 9 1235

end mango_rate_is_75_l3552_355273


namespace point_on_line_l3552_355243

/-- Given a line L with equation Ax + By + C = 0 that can be rewritten as A(x - x₀) + B(y - y₀) = 0,
    prove that the point (x₀, y₀) lies on the line L. -/
theorem point_on_line (A B C x₀ y₀ : ℝ) :
  (∀ x y, A * x + B * y + C = 0 ↔ A * (x - x₀) + B * (y - y₀) = 0) →
  A * x₀ + B * y₀ + C = 0 := by
sorry

end point_on_line_l3552_355243


namespace tuna_weight_l3552_355238

/-- A fish market scenario where we need to determine the weight of each tuna. -/
theorem tuna_weight (total_customers : ℕ) (num_tuna : ℕ) (pounds_per_customer : ℕ) (unserved_customers : ℕ) :
  total_customers = 100 →
  num_tuna = 10 →
  pounds_per_customer = 25 →
  unserved_customers = 20 →
  (total_customers - unserved_customers) * pounds_per_customer / num_tuna = 200 := by
sorry

end tuna_weight_l3552_355238


namespace max_consecutive_sum_is_six_l3552_355236

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The target sum -/
def target_sum : ℕ := 21

/-- The property that n consecutive integers sum to the target -/
def sum_to_target (n : ℕ) : Prop :=
  sum_first_n n = target_sum

/-- The maximum number of consecutive positive integers that sum to the target -/
def max_consecutive_sum : ℕ := 6

theorem max_consecutive_sum_is_six :
  (sum_to_target max_consecutive_sum) ∧
  (∀ k : ℕ, k > max_consecutive_sum → ¬(sum_to_target k)) :=
sorry

end max_consecutive_sum_is_six_l3552_355236


namespace unique_solution_xy_l3552_355229

theorem unique_solution_xy : ∃! (x y : ℝ), 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - y = (x - 3) + (y - 3)) ∧ 
  x = 1 ∧ y = 6 := by
  sorry

end unique_solution_xy_l3552_355229


namespace inequality_problem_l3552_355242

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a * c < b * d := by
  sorry

end inequality_problem_l3552_355242


namespace exam_average_l3552_355203

theorem exam_average (total_candidates : ℕ) (passed_candidates : ℕ) (passed_avg : ℝ) (failed_avg : ℝ) :
  total_candidates = 120 →
  passed_candidates = 100 →
  passed_avg = 39 →
  failed_avg = 15 →
  let failed_candidates := total_candidates - passed_candidates
  let total_marks := passed_candidates * passed_avg + failed_candidates * failed_avg
  let overall_avg := total_marks / total_candidates
  overall_avg = 35 := by
sorry

end exam_average_l3552_355203
