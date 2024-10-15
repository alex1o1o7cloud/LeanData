import Mathlib

namespace NUMINAMATH_CALUDE_lowest_possible_score_l2947_294779

/-- Represents a set of test scores -/
structure TestScores where
  scores : List ℕ
  deriving Repr

/-- Calculates the average of a list of numbers -/
def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

/-- Checks if a number is within a given range -/
def inRange (n : ℕ) (lower upper : ℕ) : Prop :=
  lower ≤ n ∧ n ≤ upper

theorem lowest_possible_score 
  (first_three : TestScores)
  (h1 : first_three.scores = [82, 90, 88])
  (h2 : first_three.scores.length = 3)
  (total_tests : ℕ)
  (h3 : total_tests = 6)
  (desired_average : ℚ)
  (h4 : desired_average = 85)
  (range_lower range_upper : ℕ)
  (h5 : range_lower = 70 ∧ range_upper = 85)
  (max_score : ℕ)
  (h6 : max_score = 100) :
  ∃ (remaining : TestScores),
    remaining.scores.length = 3 ∧
    (∃ (score : ℕ), score ∈ remaining.scores ∧ inRange score range_lower range_upper) ∧
    (∃ (lowest : ℕ), lowest ∈ remaining.scores ∧ lowest = 65) ∧
    average (first_three.scores ++ remaining.scores) = desired_average ∧
    (∀ (s : ℕ), s ∈ (first_three.scores ++ remaining.scores) → s ≤ max_score) :=
by sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l2947_294779


namespace NUMINAMATH_CALUDE_sum_of_three_square_roots_inequality_l2947_294780

theorem sum_of_three_square_roots_inequality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) : 
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_square_roots_inequality_l2947_294780


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l2947_294738

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base : ℝ) (height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 12 cm and height 10 cm is 120 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 12 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l2947_294738


namespace NUMINAMATH_CALUDE_real_pair_existence_l2947_294710

theorem real_pair_existence :
  (∃ (u v : ℝ), (∃ (q : ℚ), u + v = q) ∧ 
    (∀ (n : ℕ), n ≥ 2 → ∀ (q : ℚ), u^n + v^n ≠ q)) ∧
  (¬ ∃ (u v : ℝ), (∀ (q : ℚ), u + v ≠ q) ∧ 
    (∀ (n : ℕ), n ≥ 2 → ∃ (q : ℚ), u^n + v^n = q)) :=
by sorry

end NUMINAMATH_CALUDE_real_pair_existence_l2947_294710


namespace NUMINAMATH_CALUDE_smallest_x_value_solution_exists_l2947_294717

theorem smallest_x_value (x : ℝ) : 
  (x^2 - x - 72) / (x - 9) = 3 / (x + 6) → x ≥ -9 :=
by
  sorry

theorem solution_exists : 
  ∃ x : ℝ, (x^2 - x - 72) / (x - 9) = 3 / (x + 6) ∧ x = -9 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_solution_exists_l2947_294717


namespace NUMINAMATH_CALUDE_marble_bag_problem_l2947_294785

theorem marble_bag_problem (total_marbles : ℕ) (red_marbles : ℕ) 
  (probability_non_red : ℚ) : 
  red_marbles = 12 → 
  probability_non_red = 36 / 49 → 
  (((total_marbles - red_marbles : ℚ) / total_marbles) ^ 2 = probability_non_red) → 
  total_marbles = 84 := by
  sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l2947_294785


namespace NUMINAMATH_CALUDE_min_bricks_needed_l2947_294725

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of the parallelepiped -/
structure ParallelepipedDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The theorem statement -/
theorem min_bricks_needed
  (brick : BrickDimensions)
  (parallelepiped : ParallelepipedDimensions)
  (h1 : brick.length = 22)
  (h2 : brick.width = 11)
  (h3 : brick.height = 6)
  (h4 : parallelepiped.length = 5 * parallelepiped.height / 4)
  (h5 : parallelepiped.width = 3 * parallelepiped.height / 2)
  (h6 : parallelepiped.length % brick.length = 0)
  (h7 : parallelepiped.width % brick.width = 0)
  (h8 : parallelepiped.height % brick.height = 0) :
  (parallelepiped.length / brick.length) *
  (parallelepiped.width / brick.width) *
  (parallelepiped.height / brick.height) = 13200 := by
  sorry

end NUMINAMATH_CALUDE_min_bricks_needed_l2947_294725


namespace NUMINAMATH_CALUDE_rectangle_properties_l2947_294790

theorem rectangle_properties (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196) :
  (2 * (x + y) = 50) ∧ 
  (∃ k : ℤ, (x + 5) * (y + 5) - (x - 2) * (y - 2) = 7 * k) ∧
  (x = y + 5 → ∃ a b : ℕ, a * b = (x + 5) * (y + 5) ∧ (a = x ∨ b = x) ∧ (a = y + 5 ∨ b = y + 5)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_properties_l2947_294790


namespace NUMINAMATH_CALUDE_triangle_area_inequalities_l2947_294754

/-- The area of a triangle ABC with sides a and b is less than or equal to both
    (1/2)(a² - ab + b²) and ((a + b)/(2√2))² -/
theorem triangle_area_inequalities (a b : ℝ) (hpos : 0 < a ∧ 0 < b) :
  let area := (1/2) * a * b * Real.sin C
  ∃ C, 0 ≤ C ∧ C ≤ π ∧
    area ≤ (1/2) * (a^2 - a*b + b^2) ∧
    area ≤ ((a + b)/(2 * Real.sqrt 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_inequalities_l2947_294754


namespace NUMINAMATH_CALUDE_carpet_fit_l2947_294718

theorem carpet_fit (carpet_area : ℝ) (cut_length : ℝ) (room_area : ℝ) : 
  carpet_area = 169 →
  cut_length = 2 →
  room_area = (Real.sqrt carpet_area) * (Real.sqrt carpet_area - cut_length) →
  room_area = 143 := by
sorry

end NUMINAMATH_CALUDE_carpet_fit_l2947_294718


namespace NUMINAMATH_CALUDE_angle_between_m_and_n_l2947_294764

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (9, 12)
def c : ℝ × ℝ := (4, -3)
def m : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)
def n : ℝ × ℝ := (a.1 + c.1, a.2 + c.2)

theorem angle_between_m_and_n :
  Real.arccos ((m.1 * n.1 + m.2 * n.2) / (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2))) = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_m_and_n_l2947_294764


namespace NUMINAMATH_CALUDE_labourer_fine_problem_l2947_294708

/-- Calculates the fine per day of absence for a labourer --/
def calculate_fine_per_day (total_days : ℕ) (daily_wage : ℚ) (total_received : ℚ) (days_absent : ℕ) : ℚ :=
  let days_worked := total_days - days_absent
  let total_earned := days_worked * daily_wage
  (total_earned - total_received) / days_absent

/-- Theorem stating the fine per day of absence for the given problem --/
theorem labourer_fine_problem :
  calculate_fine_per_day 25 2 (37 + 1/2) 5 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_labourer_fine_problem_l2947_294708


namespace NUMINAMATH_CALUDE_road_length_for_given_conditions_l2947_294745

/-- Calculates the length of a road given the number of trees, space between trees, and space occupied by each tree. -/
def road_length (num_trees : ℕ) (space_between : ℕ) (tree_space : ℕ) : ℕ :=
  (num_trees * tree_space) + ((num_trees - 1) * space_between)

/-- Theorem stating that for 11 trees, with 14 feet between each tree, and each tree taking 1 foot of space, the road length is 151 feet. -/
theorem road_length_for_given_conditions :
  road_length 11 14 1 = 151 := by
  sorry

end NUMINAMATH_CALUDE_road_length_for_given_conditions_l2947_294745


namespace NUMINAMATH_CALUDE_frac_repeating_block_length_l2947_294730

/-- The least number of digits in a repeating block of the decimal expansion of 7/13 -/
def repeating_block_length : ℕ := 6

/-- 7/13 is a rational number -/
def frac : ℚ := 7 / 13

theorem frac_repeating_block_length : 
  ∃ (n : ℕ) (k : ℕ+) (a b : ℕ), 
    frac * 10^n = (a : ℚ) + (b : ℚ) / (10^repeating_block_length - 1) ∧
    b < 10^repeating_block_length - 1 ∧
    ∀ m < repeating_block_length, 
      ¬∃ (c d : ℕ), frac * 10^n = (c : ℚ) + (d : ℚ) / (10^m - 1) ∧ d < 10^m - 1 :=
sorry

end NUMINAMATH_CALUDE_frac_repeating_block_length_l2947_294730


namespace NUMINAMATH_CALUDE_rectangle_width_l2947_294792

theorem rectangle_width (area : ℝ) (length width : ℝ) : 
  area = 63 →
  width = length - 2 →
  area = length * width →
  width = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l2947_294792


namespace NUMINAMATH_CALUDE_crabapple_recipients_count_l2947_294746

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of different sequences of crabapple recipients in a week -/
def crabapple_sequences : ℕ := num_students * (num_students - 1) * (num_students - 2)

/-- Theorem stating the number of different sequences of crabapple recipients -/
theorem crabapple_recipients_count :
  crabapple_sequences = 2730 :=
by sorry

end NUMINAMATH_CALUDE_crabapple_recipients_count_l2947_294746


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2947_294721

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3}

theorem union_of_M_and_N :
  M ∪ N = {1, 2, 3} :=
by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2947_294721


namespace NUMINAMATH_CALUDE_no_linear_term_in_product_l2947_294727

theorem no_linear_term_in_product (m : ℚ) : 
  (∀ x : ℚ, (x - 2) * (x^2 + m*x + 1) = x^3 + (m-2)*x^2 + 0*x + (-2)) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_in_product_l2947_294727


namespace NUMINAMATH_CALUDE_special_matrix_exists_iff_even_l2947_294723

/-- A matrix with elements from {-1, 0, 1} -/
def SpecialMatrix (n : ℕ) := Matrix (Fin n) (Fin n) (Fin 3)

/-- The sum of elements in a row of a SpecialMatrix -/
def rowSum (A : SpecialMatrix n) (i : Fin n) : ℤ := sorry

/-- The sum of elements in a column of a SpecialMatrix -/
def colSum (A : SpecialMatrix n) (j : Fin n) : ℤ := sorry

/-- All row and column sums are distinct -/
def distinctSums (A : SpecialMatrix n) : Prop :=
  ∀ i j i' j', (i ≠ i' ∨ j ≠ j') → 
    (rowSum A i ≠ rowSum A i' ∧ 
     rowSum A i ≠ colSum A j' ∧ 
     colSum A j ≠ rowSum A i' ∧ 
     colSum A j ≠ colSum A j')

theorem special_matrix_exists_iff_even (n : ℕ) :
  (∃ A : SpecialMatrix n, distinctSums A) ↔ (∃ k : ℕ, n = 2 * k) :=
sorry

end NUMINAMATH_CALUDE_special_matrix_exists_iff_even_l2947_294723


namespace NUMINAMATH_CALUDE_macaroons_remaining_l2947_294736

/-- The number of red macaroons initially baked -/
def initial_red : ℕ := 50

/-- The number of green macaroons initially baked -/
def initial_green : ℕ := 40

/-- The number of green macaroons eaten -/
def green_eaten : ℕ := 15

/-- The number of red macaroons eaten is twice the number of green macaroons eaten -/
def red_eaten : ℕ := 2 * green_eaten

/-- The total number of remaining macaroons -/
def remaining_macaroons : ℕ := (initial_red - red_eaten) + (initial_green - green_eaten)

theorem macaroons_remaining :
  remaining_macaroons = 45 := by
  sorry

end NUMINAMATH_CALUDE_macaroons_remaining_l2947_294736


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2947_294758

theorem quadratic_inequality_solution (a m : ℝ) : 
  (∀ x : ℝ, (a * x^2 + 6 * x - a^2 < 0) ↔ (x < 1 ∨ x > m)) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2947_294758


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2947_294752

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2947_294752


namespace NUMINAMATH_CALUDE_max_product_given_sum_l2947_294743

theorem max_product_given_sum (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 40 → ∀ x y : ℝ, x > 0 → y > 0 → x + y = 40 → x * y ≤ a * b → a * b ≤ 400 := by
  sorry

end NUMINAMATH_CALUDE_max_product_given_sum_l2947_294743


namespace NUMINAMATH_CALUDE_rectangle_area_l2947_294776

/-- Given a rectangular plot where the length is thrice the breadth and the breadth is 30 meters,
    prove that the area is 2700 square meters. -/
theorem rectangle_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 30 →
  length = 3 * breadth →
  area = length * breadth →
  area = 2700 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2947_294776


namespace NUMINAMATH_CALUDE_find_number_l2947_294722

theorem find_number (G N : ℕ) (h1 : G = 129) (h2 : N % G = 9) (h3 : 2206 % G = 13) : N = 2202 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2947_294722


namespace NUMINAMATH_CALUDE_final_answer_calculation_l2947_294732

theorem final_answer_calculation (chosen_number : ℕ) (h : chosen_number = 800) : 
  (chosen_number / 5 : ℚ) - 154 = 6 := by
  sorry

end NUMINAMATH_CALUDE_final_answer_calculation_l2947_294732


namespace NUMINAMATH_CALUDE_age_difference_l2947_294748

theorem age_difference (frank_age john_age : ℕ) : 
  (frank_age + 4 = 16) → 
  (john_age + 3 = 2 * (frank_age + 3)) → 
  (john_age - frank_age = 15) :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l2947_294748


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2947_294755

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * s = 4 * Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2947_294755


namespace NUMINAMATH_CALUDE_stating_batsman_average_increase_l2947_294751

/-- 
Represents a batsman's scoring record.
-/
structure BatsmanRecord where
  inningsPlayed : ℕ
  totalRuns : ℕ
  average : ℚ

/-- 
Calculates the increase in average given the batsman's record before and after an inning.
-/
def averageIncrease (before after : BatsmanRecord) : ℚ :=
  after.average - before.average

/-- 
Theorem stating that given a batsman's score of 85 runs in the 17th inning
and an average of 37 runs after the 17th inning, the increase in the batsman's average is 3 runs.
-/
theorem batsman_average_increase :
  ∀ (before : BatsmanRecord),
    before.inningsPlayed = 16 →
    (BatsmanRecord.mk 17 (before.totalRuns + 85) 37).average - before.average = 3 := by
  sorry

end NUMINAMATH_CALUDE_stating_batsman_average_increase_l2947_294751


namespace NUMINAMATH_CALUDE_circle_properties_l2947_294772

-- Define the circle C
def Circle (t s r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - t)^2 + (p.2 - s)^2 = r^2}

-- Define the condition for independence of x₀ and y₀
def IndependentSum (t s r a b : ℝ) : Prop :=
  ∀ (x₀ y₀ : ℝ), (x₀, y₀) ∈ Circle t s r →
    ∃ (k : ℝ), |x₀ - y₀ + a| + |x₀ - y₀ + b| = k

-- Main theorem
theorem circle_properties
  (t s r a b : ℝ)
  (h_r : r > 0)
  (h_ab : a ≠ b)
  (h_ind : IndependentSum t s r a b) :
  (|a - b| = 2 * Real.sqrt 2 * r →
    ∃ (m n : ℝ), ∀ (x y : ℝ), (x, y) ∈ Circle t s r → m * x + n * y = 1) ∧
  (|a - b| = 2 * Real.sqrt 2 →
    r ≤ 1 ∧ ∃ (t₀ s₀ : ℝ), r = 1 ∧ (t₀, s₀) ∈ Circle t s r) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l2947_294772


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_implication_l2947_294788

theorem sufficient_not_necessary_implication (p q : Prop) :
  (p → q) ∧ ¬(q → p) → (¬q → ¬p) ∧ ¬(¬p → ¬q) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_implication_l2947_294788


namespace NUMINAMATH_CALUDE_eraser_cost_l2947_294713

theorem eraser_cost (total_cartons : ℕ) (total_cost : ℕ) (pencil_cost : ℕ) (pencil_cartons : ℕ) :
  total_cartons = 100 →
  total_cost = 360 →
  pencil_cost = 6 →
  pencil_cartons = 20 →
  (total_cost - pencil_cost * pencil_cartons) / (total_cartons - pencil_cartons) = 3 := by
sorry

end NUMINAMATH_CALUDE_eraser_cost_l2947_294713


namespace NUMINAMATH_CALUDE_duck_pond_problem_l2947_294763

theorem duck_pond_problem (initial_ducks : ℕ) (final_ducks : ℕ) 
  (h1 : initial_ducks = 320)
  (h2 : final_ducks = 140) : 
  ∃ (F : ℚ),
    F = 1/6 ∧
    final_ducks = (initial_ducks * 3/4 * (1 - F) * 0.7).floor := by
  sorry

end NUMINAMATH_CALUDE_duck_pond_problem_l2947_294763


namespace NUMINAMATH_CALUDE_set_A_membership_l2947_294793

def A : Set ℝ := {x | 2 * x - 3 < 0}

theorem set_A_membership : 1 ∈ A ∧ 2 ∉ A := by
  sorry

end NUMINAMATH_CALUDE_set_A_membership_l2947_294793


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_inequality_l2947_294741

theorem negation_of_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x - 1| ≥ 2) ↔ (∃ x : ℝ, |x - 1| < 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_inequality_l2947_294741


namespace NUMINAMATH_CALUDE_min_value_of_f_l2947_294783

def f (x : ℝ) : ℝ := x^2 - 8*x + 15

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x₀ : ℝ), f x₀ = m) ∧ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2947_294783


namespace NUMINAMATH_CALUDE_sophie_donuts_l2947_294753

/-- The number of donuts left for Sophie after buying boxes and giving some away. -/
def donuts_left (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_given : ℕ) (donuts_given : ℕ) : ℕ :=
  (total_boxes - boxes_given) * donuts_per_box - donuts_given

/-- Theorem stating that Sophie has 30 donuts left. -/
theorem sophie_donuts :
  donuts_left 4 12 1 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sophie_donuts_l2947_294753


namespace NUMINAMATH_CALUDE_initial_playtime_l2947_294711

/-- Proof of initial daily playtime in a game scenario -/
theorem initial_playtime (initial_days : ℕ) (initial_completion_percent : ℚ)
  (remaining_days : ℕ) (remaining_hours_per_day : ℕ) :
  initial_days = 14 →
  initial_completion_percent = 2/5 →
  remaining_days = 12 →
  remaining_hours_per_day = 7 →
  ∃ (x : ℚ),
    x * initial_days = initial_completion_percent * (x * initial_days + remaining_days * remaining_hours_per_day) ∧
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_playtime_l2947_294711


namespace NUMINAMATH_CALUDE_correct_calculation_result_l2947_294704

theorem correct_calculation_result : ∃ x : ℕ, (40 + x = 52) ∧ (20 * x = 240) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_l2947_294704


namespace NUMINAMATH_CALUDE_basil_plant_selling_price_l2947_294731

/-- Proves that the selling price per basil plant is $5.00 given the costs and net profit --/
theorem basil_plant_selling_price 
  (seed_cost : ℝ) 
  (soil_cost : ℝ) 
  (num_plants : ℕ) 
  (net_profit : ℝ) 
  (h1 : seed_cost = 2)
  (h2 : soil_cost = 8)
  (h3 : num_plants = 20)
  (h4 : net_profit = 90) :
  (net_profit + seed_cost + soil_cost) / num_plants = 5 := by
  sorry

#check basil_plant_selling_price

end NUMINAMATH_CALUDE_basil_plant_selling_price_l2947_294731


namespace NUMINAMATH_CALUDE_marble_sculpture_first_week_cut_l2947_294729

/-- Proves that the percentage of marble cut away in the first week is 30% --/
theorem marble_sculpture_first_week_cut (
  original_weight : ℝ)
  (second_week_cut : ℝ)
  (third_week_cut : ℝ)
  (final_weight : ℝ)
  (h1 : original_weight = 250)
  (h2 : second_week_cut = 20)
  (h3 : third_week_cut = 25)
  (h4 : final_weight = 105)
  : ∃ (first_week_cut : ℝ),
    first_week_cut = 30 ∧
    final_weight = original_weight * 
      (1 - first_week_cut / 100) * 
      (1 - second_week_cut / 100) * 
      (1 - third_week_cut / 100) := by
  sorry


end NUMINAMATH_CALUDE_marble_sculpture_first_week_cut_l2947_294729


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_stamp_books_l2947_294728

theorem largest_common_divisor_of_stamp_books : ∃ (n : ℕ), n > 0 ∧ n ∣ 900 ∧ n ∣ 1200 ∧ n ∣ 1500 ∧ ∀ (m : ℕ), m > n → ¬(m ∣ 900 ∧ m ∣ 1200 ∧ m ∣ 1500) := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_stamp_books_l2947_294728


namespace NUMINAMATH_CALUDE_advertisement_arrangements_l2947_294795

theorem advertisement_arrangements : ℕ := by
  -- Define the total number of advertisements
  let total_ads : ℕ := 6
  -- Define the number of commercial advertisements
  let commercial_ads : ℕ := 4
  -- Define the number of public service advertisements
  let public_service_ads : ℕ := 2
  -- Define the condition that public service ads must be at the beginning and end
  let public_service_at_ends : Prop := true

  -- The theorem to prove
  have : (public_service_at_ends ∧ 
          total_ads = commercial_ads + public_service_ads) → 
         (Nat.factorial public_service_ads * Nat.factorial commercial_ads = 48) := by
    sorry

  -- The final statement
  exact 48

end NUMINAMATH_CALUDE_advertisement_arrangements_l2947_294795


namespace NUMINAMATH_CALUDE_divide_by_sqrt_two_l2947_294726

theorem divide_by_sqrt_two : 2 / Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_sqrt_two_l2947_294726


namespace NUMINAMATH_CALUDE_jack_morning_emails_indeterminate_l2947_294747

/-- Represents the number of emails received at different times of the day -/
structure EmailCount where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- Defines the properties of Jack's email counts -/
def jack_email_properties (e : EmailCount) : Prop :=
  e.afternoon = 5 ∧ 
  e.evening = 8 ∧ 
  e.afternoon + e.evening = 13

/-- Theorem stating that Jack's morning email count cannot be uniquely determined -/
theorem jack_morning_emails_indeterminate :
  ∃ e1 e2 : EmailCount, 
    jack_email_properties e1 ∧ 
    jack_email_properties e2 ∧ 
    e1.morning ≠ e2.morning :=
sorry

end NUMINAMATH_CALUDE_jack_morning_emails_indeterminate_l2947_294747


namespace NUMINAMATH_CALUDE_intersection_point_l2947_294740

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 2) / 4 = (y - 1) / (-3) ∧ (y - 1) / (-3) = (z + 3) / (-2)

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  3 * x - y + 4 * z = 0

/-- The theorem stating that (6, -2, -5) is the unique point of intersection -/
theorem intersection_point : ∃! (x y z : ℝ), line x y z ∧ plane x y z ∧ x = 6 ∧ y = -2 ∧ z = -5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2947_294740


namespace NUMINAMATH_CALUDE_sam_has_six_balloons_l2947_294712

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := total_balloons - fred_balloons - mary_balloons

theorem sam_has_six_balloons : sam_balloons = 6 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_six_balloons_l2947_294712


namespace NUMINAMATH_CALUDE_counterexamples_count_l2947_294749

def sumOfDigits (n : ℕ) : ℕ := sorry

def hasNoZeroDigit (n : ℕ) : Prop := sorry

def isPrime (n : ℕ) : Prop := sorry

theorem counterexamples_count :
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, sumOfDigits n = 5 ∧ hasNoZeroDigit n ∧ ¬isPrime n) ∧
    (∀ n ∉ S, ¬(sumOfDigits n = 5 ∧ hasNoZeroDigit n ∧ ¬isPrime n)) ∧
    Finset.card S = 6 := by sorry

end NUMINAMATH_CALUDE_counterexamples_count_l2947_294749


namespace NUMINAMATH_CALUDE_first_number_is_thirty_l2947_294774

theorem first_number_is_thirty (x y : ℝ) 
  (sum_eq : x + y = 50) 
  (diff_eq : 2 * (x - y) = 20) : 
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_first_number_is_thirty_l2947_294774


namespace NUMINAMATH_CALUDE_february_production_l2947_294715

/-- Represents the monthly carrot cake production sequence -/
def carrotCakeSequence : ℕ → ℕ
| 0 => 19  -- October (0-indexed)
| n + 1 => carrotCakeSequence n + 2

/-- Theorem stating that the 5th term (February) of the sequence is 27 -/
theorem february_production : carrotCakeSequence 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_february_production_l2947_294715


namespace NUMINAMATH_CALUDE_max_perimeter_rectangle_l2947_294706

/-- Represents a rectangular enclosure -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The perimeter of the rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: Maximum perimeter of a rectangle with given constraints -/
theorem max_perimeter_rectangle : 
  ∃ (r : Rectangle), 
    area r = 8000 ∧ 
    r.width ≥ 50 ∧
    ∀ (r' : Rectangle), area r' = 8000 ∧ r'.width ≥ 50 → perimeter r' ≤ perimeter r ∧
    r.length = 100 ∧ 
    r.width = 80 ∧ 
    perimeter r = 360 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_rectangle_l2947_294706


namespace NUMINAMATH_CALUDE_pauls_lawn_mowing_earnings_l2947_294786

/-- 
Given that:
1. Paul's total money is the sum of money from mowing lawns and $28 from weed eating
2. Paul spends $9 per week
3. Paul's money lasts for 8 weeks

Prove that Paul made $44 mowing lawns.
-/
theorem pauls_lawn_mowing_earnings :
  ∀ (M : ℕ), -- M represents the amount Paul made mowing lawns
  (M + 28 = 9 * 8) → -- Total money equals weekly spending times number of weeks
  M = 44 := by
sorry

end NUMINAMATH_CALUDE_pauls_lawn_mowing_earnings_l2947_294786


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_count_l2947_294762

/-- Represents the characteristics of girls in a school -/
structure SchoolGirls where
  total : ℕ
  blueEyedBlondes : ℕ
  brunettes : ℕ
  brownEyed : ℕ

/-- Calculates the number of brown-eyed brunettes -/
def brownEyedBrunettes (s : SchoolGirls) : ℕ :=
  s.brownEyed - (s.total - s.brunettes - s.blueEyedBlondes)

/-- Theorem stating the number of brown-eyed brunettes -/
theorem brown_eyed_brunettes_count (s : SchoolGirls) 
  (h1 : s.total = 60)
  (h2 : s.blueEyedBlondes = 20)
  (h3 : s.brunettes = 35)
  (h4 : s.brownEyed = 25) :
  brownEyedBrunettes s = 20 := by
  sorry

#eval brownEyedBrunettes { total := 60, blueEyedBlondes := 20, brunettes := 35, brownEyed := 25 }

end NUMINAMATH_CALUDE_brown_eyed_brunettes_count_l2947_294762


namespace NUMINAMATH_CALUDE_camp_provisions_duration_l2947_294798

/-- Represents the camp provisions problem -/
theorem camp_provisions_duration (initial_men_1 initial_men_2 : ℕ) 
  (initial_days_1 initial_days_2 : ℕ) (additional_men : ℕ) 
  (consumption_rate : ℚ) (days_before_supply : ℕ) 
  (supply_men supply_days : ℕ) : 
  initial_men_1 = 800 →
  initial_men_2 = 200 →
  initial_days_1 = 20 →
  initial_days_2 = 10 →
  additional_men = 200 →
  consumption_rate = 3/2 →
  days_before_supply = 10 →
  supply_men = 300 →
  supply_days = 15 →
  ∃ (remaining_days : ℚ), 
    remaining_days > 7.30 ∧ 
    remaining_days < 7.32 ∧
    remaining_days = 
      (initial_men_1 * initial_days_1 + initial_men_2 * initial_days_2 - 
       (initial_men_1 + initial_men_2 + additional_men * consumption_rate) * days_before_supply +
       supply_men * supply_days) / 
      (initial_men_1 + initial_men_2 + additional_men * consumption_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_camp_provisions_duration_l2947_294798


namespace NUMINAMATH_CALUDE_smallest_result_l2947_294702

def number_set : Finset ℕ := {3, 4, 7, 11, 13, 14}

def is_prime_greater_than_10 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n > 10

def valid_triple (a b c : ℕ) : Prop :=
  a ∈ number_set ∧ b ∈ number_set ∧ c ∈ number_set ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (is_prime_greater_than_10 a ∨ is_prime_greater_than_10 b ∨ is_prime_greater_than_10 c)

def process_result (a b c : ℕ) : ℕ :=
  (a + b) * c

theorem smallest_result :
  ∀ a b c : ℕ, valid_triple a b c →
    77 ≤ min (process_result a b c) (min (process_result a c b) (process_result b c a)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_l2947_294702


namespace NUMINAMATH_CALUDE_same_function_l2947_294701

theorem same_function (x : ℝ) : (x^3 + x) / (x^2 + 1) = x := by
  sorry

end NUMINAMATH_CALUDE_same_function_l2947_294701


namespace NUMINAMATH_CALUDE_original_prices_l2947_294767

-- Define the sale prices and discount rates
def book_sale_price : ℚ := 8
def book_discount_rate : ℚ := 1 / 8
def pen_sale_price : ℚ := 4
def pen_discount_rate : ℚ := 1 / 5

-- Theorem statement
theorem original_prices :
  (book_sale_price / book_discount_rate = 64) ∧
  (pen_sale_price / pen_discount_rate = 20) :=
by sorry

end NUMINAMATH_CALUDE_original_prices_l2947_294767


namespace NUMINAMATH_CALUDE_function_property_implies_k_equals_8_l2947_294703

/-- Given a function f: ℝ → ℝ satisfying certain properties, prove that k = 8 -/
theorem function_property_implies_k_equals_8 (f : ℝ → ℝ) (k : ℝ) 
  (h1 : f 1 = 1)
  (h2 : ∀ x y, f (x + y) = f x + f y + k * x * y - 2)
  (h3 : f 7 = 163) :
  k = 8 := by
  sorry

end NUMINAMATH_CALUDE_function_property_implies_k_equals_8_l2947_294703


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2947_294714

theorem complex_equation_solution (z : ℂ) :
  (1 + Complex.I) * z = -2 * Complex.I →
  z = -1 - Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2947_294714


namespace NUMINAMATH_CALUDE_smallest_sequence_sum_l2947_294742

theorem smallest_sequence_sum : ∃ (A B C D : ℕ),
  (A > 0 ∧ B > 0 ∧ C > 0) ∧  -- A, B, C are positive integers
  (∃ (r : ℚ), C - B = B - A ∧ C = B * r ∧ D = C * r) ∧  -- arithmetic and geometric sequences
  (C : ℚ) / B = 7 / 4 ∧  -- C/B = 7/4
  (∀ (A' B' C' D' : ℕ),
    (A' > 0 ∧ B' > 0 ∧ C' > 0) →
    (∃ (r' : ℚ), C' - B' = B' - A' ∧ C' = B' * r' ∧ D' = C' * r') →
    (C' : ℚ) / B' = 7 / 4 →
    A + B + C + D ≤ A' + B' + C' + D') ∧
  A + B + C + D = 97 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sequence_sum_l2947_294742


namespace NUMINAMATH_CALUDE_divisible_by_sixteen_l2947_294796

theorem divisible_by_sixteen (n : ℕ) : ∃ k : ℤ, (2*n - 1)^3 - (2*n)^2 + 2*n + 1 = 16 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_sixteen_l2947_294796


namespace NUMINAMATH_CALUDE_expected_sum_of_marbles_l2947_294766

-- Define the set of marbles
def marbles : Finset ℕ := Finset.range 7

-- Define the function to calculate the sum of two marbles
def marbleSum (pair : Finset ℕ) : ℕ := Finset.sum pair id

-- Define the set of all possible pairs of marbles
def marblePairs : Finset (Finset ℕ) := marbles.powerset.filter (fun s => s.card = 2)

-- Statement of the theorem
theorem expected_sum_of_marbles :
  (Finset.sum marblePairs marbleSum) / marblePairs.card = 52 / 7 := by
sorry

end NUMINAMATH_CALUDE_expected_sum_of_marbles_l2947_294766


namespace NUMINAMATH_CALUDE_square_minus_product_plus_triple_l2947_294799

theorem square_minus_product_plus_triple (x y : ℝ) :
  x - y + 3 = 0 → x^2 - x*y + 3*y = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_triple_l2947_294799


namespace NUMINAMATH_CALUDE_stock_value_change_l2947_294735

theorem stock_value_change (initial_value : ℝ) (day1_decrease : ℝ) (day2_increase : ℝ) :
  day1_decrease = 0.2 →
  day2_increase = 0.3 →
  (1 - day1_decrease) * (1 + day2_increase) = 1.04 := by
  sorry

end NUMINAMATH_CALUDE_stock_value_change_l2947_294735


namespace NUMINAMATH_CALUDE_factorial_simplification_l2947_294794

theorem factorial_simplification : (15 : ℕ).factorial / ((12 : ℕ).factorial + 3 * (10 : ℕ).factorial) = 2669 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l2947_294794


namespace NUMINAMATH_CALUDE_blue_spools_count_l2947_294739

/-- The number of spools needed to make one beret -/
def spools_per_beret : ℕ := 3

/-- The number of red yarn spools -/
def red_spools : ℕ := 12

/-- The number of black yarn spools -/
def black_spools : ℕ := 15

/-- The total number of berets that can be made -/
def total_berets : ℕ := 11

/-- The number of blue yarn spools -/
def blue_spools : ℕ := total_berets * spools_per_beret - (red_spools + black_spools)

theorem blue_spools_count : blue_spools = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_spools_count_l2947_294739


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l2947_294734

/-- Given a rectangle with sides L and W, where one side is measured 14% in excess
    and the other side is measured x% in deficit, resulting in an 8.3% error in the calculated area,
    prove that x = 5. -/
theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) (h_pos_L : L > 0) (h_pos_W : W > 0) : 
  (1.14 * L) * ((1 - 0.01 * x) * W) = 1.083 * (L * W) → x = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l2947_294734


namespace NUMINAMATH_CALUDE_log_equation_solution_l2947_294773

theorem log_equation_solution :
  ∃ x : ℝ, (Real.log x - 4 * Real.log 5 = -3) ∧ (x = 0.625) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2947_294773


namespace NUMINAMATH_CALUDE_pond_A_has_more_fish_l2947_294724

-- Define the capture-recapture estimation function
def estimateFishPopulation (totalSecondCatch : ℕ) (totalMarkedReleased : ℕ) (markedInSecondCatch : ℕ) : ℚ :=
  (totalSecondCatch * totalMarkedReleased : ℚ) / markedInSecondCatch

-- Define the parameters for each pond
def pondAMarkedFish : ℕ := 8
def pondBMarkedFish : ℕ := 16
def fishCaught : ℕ := 200
def fishMarked : ℕ := 200

-- Theorem statement
theorem pond_A_has_more_fish :
  estimateFishPopulation fishCaught fishMarked pondAMarkedFish >
  estimateFishPopulation fishCaught fishMarked pondBMarkedFish :=
by
  sorry

end NUMINAMATH_CALUDE_pond_A_has_more_fish_l2947_294724


namespace NUMINAMATH_CALUDE_omega_range_l2947_294777

open Real

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := cos (ω * x + φ)

theorem omega_range (ω φ α : ℝ) :
  ω > 0 →
  f ω φ α = 0 →
  deriv (f ω φ) α > 0 →
  (∀ x ∈ Set.Icc α (π + α), ¬ IsLocalMin (f ω φ) x) →
  ω ∈ Set.Ioo 1 (3/2) :=
sorry

end NUMINAMATH_CALUDE_omega_range_l2947_294777


namespace NUMINAMATH_CALUDE_rebus_solution_l2947_294733

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_distinct (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

theorem rebus_solution (a p m i r : ℕ) :
  is_digit a ∧ is_digit p ∧ is_digit m ∧ is_digit i ∧ is_digit r ∧
  are_distinct a p m i r ∧
  (10 * a + p) ^ m = 100 * m + 10 * i + r →
  a = 1 ∧ p = 6 ∧ m = 2 ∧ i = 5 ∧ r = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_solution_l2947_294733


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l2947_294797

/-- The volume of a sphere inscribed in a cube with edge length 8 inches -/
theorem inscribed_sphere_volume (π : ℝ) :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = 256 * π / 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l2947_294797


namespace NUMINAMATH_CALUDE_leila_cake_problem_l2947_294760

theorem leila_cake_problem (monday : ℕ) (friday : ℕ) (saturday : ℕ) : 
  friday = 9 →
  saturday = 3 * monday →
  monday + friday + saturday = 33 →
  monday = 6 := by
sorry

end NUMINAMATH_CALUDE_leila_cake_problem_l2947_294760


namespace NUMINAMATH_CALUDE_orange_harvest_days_l2947_294789

def sacks_per_day : ℕ := 4
def total_sacks : ℕ := 56

theorem orange_harvest_days : 
  total_sacks / sacks_per_day = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_days_l2947_294789


namespace NUMINAMATH_CALUDE_equivalent_proposition_and_truth_l2947_294720

theorem equivalent_proposition_and_truth :
  (∀ x : ℝ, x > 1 → (x - 1) * (x + 3) > 0) ↔
  (∀ x : ℝ, (x - 1) * (x + 3) ≤ 0 → x ≤ 1) ∧
  (∀ x : ℝ, x > 1 → (x - 1) * (x + 3) > 0) ∧
  (∀ x : ℝ, (x - 1) * (x + 3) ≤ 0 → x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_proposition_and_truth_l2947_294720


namespace NUMINAMATH_CALUDE_spatial_quadrilateral_angle_sum_l2947_294700

-- Define a spatial quadrilateral
structure SpatialQuadrilateral :=
  (A B C D : Real)

-- State the theorem
theorem spatial_quadrilateral_angle_sum 
  (q : SpatialQuadrilateral) : q.A + q.B + q.C + q.D ≤ 360 := by
  sorry

end NUMINAMATH_CALUDE_spatial_quadrilateral_angle_sum_l2947_294700


namespace NUMINAMATH_CALUDE_average_sales_is_84_l2947_294778

/-- Sales data for each month -/
def sales : List Int := [120, 80, -20, 100, 140]

/-- Number of months -/
def num_months : Nat := 5

/-- Theorem: The average sales per month is 84 dollars -/
theorem average_sales_is_84 : (sales.sum / num_months : Int) = 84 := by
  sorry

end NUMINAMATH_CALUDE_average_sales_is_84_l2947_294778


namespace NUMINAMATH_CALUDE_union_M_complement_N_l2947_294775

universe u

def U : Finset ℕ := {0, 1, 2, 3, 4, 5}
def M : Finset ℕ := {0, 3, 5}
def N : Finset ℕ := {1, 4, 5}

theorem union_M_complement_N : M ∪ (U \ N) = {0, 2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_union_M_complement_N_l2947_294775


namespace NUMINAMATH_CALUDE_min_pieces_is_3n_plus_1_l2947_294716

/-- A rectangular sheet of paper with holes -/
structure PerforatedSheet :=
  (n : ℕ)  -- number of holes
  (noOverlap : Bool)  -- holes do not overlap
  (parallelSides : Bool)  -- holes' sides are parallel to sheet edges

/-- The minimum number of rectangular pieces a perforated sheet can be divided into -/
def minPieces (sheet : PerforatedSheet) : ℕ :=
  3 * sheet.n + 1

/-- Theorem: The minimum number of rectangular pieces is 3n + 1 -/
theorem min_pieces_is_3n_plus_1 (sheet : PerforatedSheet) 
  (h1 : sheet.noOverlap = true) 
  (h2 : sheet.parallelSides = true) : 
  minPieces sheet = 3 * sheet.n + 1 := by
  sorry

end NUMINAMATH_CALUDE_min_pieces_is_3n_plus_1_l2947_294716


namespace NUMINAMATH_CALUDE_hotel_revenue_l2947_294744

theorem hotel_revenue
  (total_rooms : ℕ)
  (single_room_cost double_room_cost : ℕ)
  (single_rooms_booked : ℕ)
  (h_total : total_rooms = 260)
  (h_single_cost : single_room_cost = 35)
  (h_double_cost : double_room_cost = 60)
  (h_single_booked : single_rooms_booked = 64) :
  single_room_cost * single_rooms_booked +
  double_room_cost * (total_rooms - single_rooms_booked) = 14000 := by
sorry

end NUMINAMATH_CALUDE_hotel_revenue_l2947_294744


namespace NUMINAMATH_CALUDE_arithmetic_progression_square_l2947_294769

/-- An arithmetic progression containing two natural numbers and the square of the smaller one also contains the square of the larger one. -/
theorem arithmetic_progression_square (a b : ℕ) (d : ℚ) (n m : ℤ) :
  a < b →
  b = a + n * d →
  a^2 = a + m * d →
  ∃ k : ℤ, b^2 = a + k * d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_square_l2947_294769


namespace NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l2947_294787

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_integer_prime_quadratic :
  ∃ (x : ℤ), (∀ y : ℤ, is_prime (Int.natAbs (5 * y^2 - 42 * y + 8)) → y ≤ x) ∧
             is_prime (Int.natAbs (5 * x^2 - 42 * x + 8)) ∧
             x = 5 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l2947_294787


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2947_294761

theorem trigonometric_identity (α β : Real) (h : α + β = Real.pi / 3) :
  Real.sin α ^ 2 + Real.sin α * Real.sin β + Real.sin β ^ 2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2947_294761


namespace NUMINAMATH_CALUDE_other_number_proof_l2947_294750

/-- Given two positive integers with specific HCF and LCM, prove that if one is 24, the other is 156 -/
theorem other_number_proof (A B : ℕ+) : 
  Nat.gcd A B = 12 →
  Nat.lcm A B = 312 →
  A = 24 →
  B = 156 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l2947_294750


namespace NUMINAMATH_CALUDE_max_value_and_sum_l2947_294756

theorem max_value_and_sum (x y z v w : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0 ∧ v > 0 ∧ w > 0) 
  (heq : 4 * x^2 + y^2 + z^2 + v^2 + w^2 = 8080) :
  (∃ (M : ℝ), ∀ (x' y' z' v' w' : ℝ), 
    x' > 0 → y' > 0 → z' > 0 → v' > 0 → w' > 0 →
    4 * x'^2 + y'^2 + z'^2 + v'^2 + w'^2 = 8080 →
    x' * z' + 4 * y' * z' + 6 * z' * v' + 14 * z' * w' ≤ M ∧
    M = 60480 * Real.sqrt 249) ∧
  (∃ (x_M y_M z_M v_M w_M : ℝ),
    x_M > 0 ∧ y_M > 0 ∧ z_M > 0 ∧ v_M > 0 ∧ w_M > 0 ∧
    4 * x_M^2 + y_M^2 + z_M^2 + v_M^2 + w_M^2 = 8080 ∧
    x_M * z_M + 4 * y_M * z_M + 6 * z_M * v_M + 14 * z_M * w_M = 60480 * Real.sqrt 249 ∧
    60480 * Real.sqrt 249 + x_M + y_M + z_M + v_M + w_M = 280 + 60600 * Real.sqrt 249) := by
  sorry

end NUMINAMATH_CALUDE_max_value_and_sum_l2947_294756


namespace NUMINAMATH_CALUDE_triangle_area_l2947_294705

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if b = 1, c = √3, and angle C = 2π/3, then the area of the triangle is √3/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 1 → c = Real.sqrt 3 → C = 2 * Real.pi / 3 →
  (1/2) * b * c * Real.sin C = Real.sqrt 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2947_294705


namespace NUMINAMATH_CALUDE_domain_of_f_l2947_294759

noncomputable def f (x : ℝ) := Real.log (x - 1) + Real.sqrt (2 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l2947_294759


namespace NUMINAMATH_CALUDE_sum_multiple_special_property_l2947_294782

def is_sum_multiple (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ m = (n / 100 + (n / 10) % 10 + n % 10) ∧ n % m = 0

def digit_sum (n : ℕ) : ℕ :=
  n / 100 + (n / 10) % 10 + n % 10

def F (n : ℕ) : ℕ :=
  max (n / 100 * 10 + (n / 10) % 10) (max (n / 100 * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

def G (n : ℕ) : ℕ :=
  min (n / 100 * 10 + (n / 10) % 10) (min (n / 100 * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

theorem sum_multiple_special_property :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧
    is_sum_multiple n ∧
    digit_sum n = 12 ∧
    n / 100 > (n / 10) % 10 ∧ (n / 10) % 10 > n % 10 ∧
    (F n + G n) % 16 = 0} =
  {732, 372, 516, 156} := by
  sorry

end NUMINAMATH_CALUDE_sum_multiple_special_property_l2947_294782


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2947_294757

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 3) : 150 * (1 / x) = 400 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2947_294757


namespace NUMINAMATH_CALUDE_intersection_intercept_sum_l2947_294771

/-- Given two lines that intersect at (3, 6), prove their y-intercepts sum to -9 -/
theorem intersection_intercept_sum (c d : ℝ) : 
  (3 = 2 * 6 + c) →  -- First line passes through (3, 6)
  (6 = 2 * 3 + d) →  -- Second line passes through (3, 6)
  c + d = -9 := by
sorry

end NUMINAMATH_CALUDE_intersection_intercept_sum_l2947_294771


namespace NUMINAMATH_CALUDE_original_denominator_problem_l2947_294765

theorem original_denominator_problem (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (11 : ℚ) / (d + 8) = 2 / 5 →
  d = 39 / 2 :=
by sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l2947_294765


namespace NUMINAMATH_CALUDE_rectangular_prism_layers_l2947_294737

theorem rectangular_prism_layers (prism_volume : ℕ) (block_volume : ℕ) (blocks_per_layer : ℕ) (h1 : prism_volume = 252) (h2 : block_volume = 1) (h3 : blocks_per_layer = 36) : 
  (prism_volume / (blocks_per_layer * block_volume) : ℕ) = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_layers_l2947_294737


namespace NUMINAMATH_CALUDE_garden_area_proof_l2947_294707

theorem garden_area_proof (total_posts : ℕ) (post_spacing : ℕ) :
  total_posts = 24 →
  post_spacing = 6 →
  ∃ (short_posts long_posts : ℕ),
    short_posts + long_posts = total_posts / 2 ∧
    long_posts = 3 * short_posts ∧
    (short_posts - 1) * post_spacing * (long_posts - 1) * post_spacing = 576 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_proof_l2947_294707


namespace NUMINAMATH_CALUDE_josh_ribbon_shortage_l2947_294784

/-- Calculates the shortage of ribbon for gift wrapping --/
def ribbon_shortage (total_ribbon : ℝ) (num_gifts : ℕ) 
  (wrap_per_gift : ℝ) (bow_per_gift : ℝ) (tag_per_gift : ℝ) (trim_per_gift : ℝ) : ℝ :=
  let required_ribbon := num_gifts * (wrap_per_gift + bow_per_gift + tag_per_gift + trim_per_gift)
  required_ribbon - total_ribbon

/-- Proves that Josh is short by 7.5 yards of ribbon --/
theorem josh_ribbon_shortage : 
  ribbon_shortage 18 6 2 1.5 0.25 0.5 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_josh_ribbon_shortage_l2947_294784


namespace NUMINAMATH_CALUDE_adjusted_ratio_equals_three_halves_l2947_294781

theorem adjusted_ratio_equals_three_halves :
  (2^2003 * 3^2005) / 6^2004 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_adjusted_ratio_equals_three_halves_l2947_294781


namespace NUMINAMATH_CALUDE_sqrt_122_between_integers_product_l2947_294719

theorem sqrt_122_between_integers_product : ∃ (n : ℕ), 
  (n : ℝ) < Real.sqrt 122 ∧ 
  Real.sqrt 122 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 132 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_122_between_integers_product_l2947_294719


namespace NUMINAMATH_CALUDE_rainfall_problem_l2947_294709

/-- Rainfall problem statement -/
theorem rainfall_problem (day1 day2 day3 normal_avg this_year_total : ℝ) 
  (h1 : day1 = 26)
  (h2 : day3 = day2 - 12)
  (h3 : normal_avg = 140)
  (h4 : this_year_total = normal_avg - 58)
  (h5 : this_year_total = day1 + day2 + day3) :
  day2 = 34 := by
sorry

end NUMINAMATH_CALUDE_rainfall_problem_l2947_294709


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2947_294768

theorem quadratic_inequality (x : ℝ) : x^2 + 9*x + 8 < 0 ↔ -8 < x ∧ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2947_294768


namespace NUMINAMATH_CALUDE_largest_common_term_l2947_294791

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem largest_common_term :
  ∃ (n m : ℕ),
    179 = arithmetic_sequence 3 8 n ∧
    179 = arithmetic_sequence 5 9 m ∧
    179 ≤ 200 ∧
    ∀ (k : ℕ), k > 179 →
      k ≤ 200 →
      (∀ (p q : ℕ), k ≠ arithmetic_sequence 3 8 p ∨ k ≠ arithmetic_sequence 5 9 q) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l2947_294791


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_h_geq_one_l2947_294770

/-- A quadratic function of the form y = (x - h)^2 + 3 -/
def quadratic_function (h : ℝ) (x : ℝ) : ℝ := (x - h)^2 + 3

/-- The derivative of the quadratic function -/
def quadratic_derivative (h : ℝ) (x : ℝ) : ℝ := 2 * (x - h)

theorem quadratic_decreasing_implies_h_geq_one (h : ℝ) :
  (∀ x < 1, quadratic_derivative h x < 0) → h ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_h_geq_one_l2947_294770
