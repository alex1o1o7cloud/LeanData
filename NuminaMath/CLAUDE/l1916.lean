import Mathlib

namespace NUMINAMATH_CALUDE_carries_babysitting_earnings_l1916_191641

/-- Carrie's babysitting earnings problem -/
theorem carries_babysitting_earnings 
  (iphone_cost : ℕ) 
  (trade_in_value : ℕ) 
  (weeks_to_work : ℕ) 
  (h1 : iphone_cost = 800)
  (h2 : trade_in_value = 240)
  (h3 : weeks_to_work = 7) :
  (iphone_cost - trade_in_value) / weeks_to_work = 80 :=
by sorry

end NUMINAMATH_CALUDE_carries_babysitting_earnings_l1916_191641


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_18_div_5_l1916_191630

/-- The equation has no solutions if and only if k = 18/5 -/
theorem no_solution_iff_k_eq_18_div_5 :
  let v1 : Fin 2 → ℝ := ![1, 3]
  let v2 : Fin 2 → ℝ := ![5, -9]
  let v3 : Fin 2 → ℝ := ![4, 0]
  let v4 : Fin 2 → ℝ := ![-2, k]
  (∀ t s : ℝ, v1 + t • v2 ≠ v3 + s • v4) ↔ k = 18/5 := by
sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_18_div_5_l1916_191630


namespace NUMINAMATH_CALUDE_truth_values_of_p_and_q_l1916_191637

theorem truth_values_of_p_and_q (hp_and_q : ¬(p ∧ q)) (hnot_p_or_q : ¬p ∨ q) :
  ¬p ∧ (q ∨ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_truth_values_of_p_and_q_l1916_191637


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1916_191656

theorem unique_n_satisfying_conditions : ∃! n : ℤ,
  50 < n ∧ n < 120 ∧
  n % 8 = 0 ∧
  n % 9 = 5 ∧
  n % 7 = 3 ∧
  n = 104 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1916_191656


namespace NUMINAMATH_CALUDE_trapezoid_area_l1916_191663

-- Define the rectangle ABCD
structure Rectangle :=
  (AB : ℝ)
  (AD : ℝ)

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the trapezoid AFCB
structure Trapezoid :=
  (AB : ℝ)
  (FC : ℝ)
  (AD : ℝ)

-- Define the problem setup
def setup (rect : Rectangle) (circ : Circle) : Prop :=
  rect.AB = 32 ∧
  rect.AD = 40 ∧
  -- Circle is tangent to AB and AD
  circ.radius ≤ min rect.AB rect.AD ∧
  -- E is on BC and BE = 1
  1 ≤ rect.AB - circ.radius

-- Theorem statement
theorem trapezoid_area 
  (rect : Rectangle) 
  (circ : Circle) 
  (trap : Trapezoid) 
  (h : setup rect circ) :
  trap.AB = rect.AB ∧ 
  trap.AD = rect.AD ∧
  trap.FC = 27 → 
  (trap.AB + trap.FC) * trap.AD / 2 = 1180 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1916_191663


namespace NUMINAMATH_CALUDE_incorrect_addition_theorem_l1916_191626

/-- Represents a 6-digit number as a list of digits -/
def SixDigitNumber := List Nat

/-- Checks if a number is a valid 6-digit number -/
def isValidSixDigitNumber (n : SixDigitNumber) : Prop :=
  n.length = 6 ∧ n.all (λ d => d < 10)

/-- Converts a 6-digit number to its integer value -/
def toInt (n : SixDigitNumber) : Nat :=
  n.foldl (λ acc d => acc * 10 + d) 0

/-- Replaces all occurrences of one digit with another in a number -/
def replaceDigit (n : SixDigitNumber) (d e : Nat) : SixDigitNumber :=
  n.map (λ x => if x = d then e else x)

theorem incorrect_addition_theorem :
  ∃ (A B : SixDigitNumber) (d e : Nat),
    isValidSixDigitNumber A ∧
    isValidSixDigitNumber B ∧
    d < 10 ∧
    e < 10 ∧
    toInt A + toInt B ≠ 1061835 ∧
    toInt (replaceDigit A d e) + toInt (replaceDigit B d e) = 1061835 ∧
    d + e = 1 :=
  sorry

end NUMINAMATH_CALUDE_incorrect_addition_theorem_l1916_191626


namespace NUMINAMATH_CALUDE_polyhedron_diagonals_l1916_191649

/-- A polyhedron with the given properties -/
structure Polyhedron :=
  (num_vertices : ℕ)
  (edges_per_vertex : ℕ)

/-- The number of interior diagonals in a polyhedron -/
def interior_diagonals (p : Polyhedron) : ℕ :=
  (p.num_vertices * (p.num_vertices - 1 - p.edges_per_vertex)) / 2

/-- Theorem: A polyhedron with 15 vertices and 6 edges per vertex has 60 interior diagonals -/
theorem polyhedron_diagonals :
  ∀ (p : Polyhedron), p.num_vertices = 15 ∧ p.edges_per_vertex = 6 →
  interior_diagonals p = 60 :=
by sorry

end NUMINAMATH_CALUDE_polyhedron_diagonals_l1916_191649


namespace NUMINAMATH_CALUDE_cylinder_inscribed_sphere_tangent_spheres_l1916_191646

theorem cylinder_inscribed_sphere_tangent_spheres 
  (cylinder_radius : ℝ) 
  (cylinder_height : ℝ) 
  (large_sphere_radius : ℝ) 
  (small_sphere_radius : ℝ) :
  cylinder_radius = 15 →
  cylinder_height = 16 →
  large_sphere_radius = Real.sqrt (cylinder_radius^2 + (cylinder_height/2)^2) →
  large_sphere_radius = small_sphere_radius + Real.sqrt ((cylinder_height/2 + small_sphere_radius)^2 + (2*small_sphere_radius*Real.sqrt 3/3)^2) →
  small_sphere_radius = (15 * Real.sqrt 37 - 75) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_inscribed_sphere_tangent_spheres_l1916_191646


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l1916_191607

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 252) 
  (h2 : a + b + c = 22) : 
  a*b + b*c + c*a = 116 := by
sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l1916_191607


namespace NUMINAMATH_CALUDE_cube_side_length_l1916_191602

theorem cube_side_length (s₂ : ℝ) : 
  s₂ > 0 →
  (6 * s₂^2) / (6 * 1^2) = 36 →
  s₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_cube_side_length_l1916_191602


namespace NUMINAMATH_CALUDE_trick_deck_cost_l1916_191680

theorem trick_deck_cost (tom_decks : ℕ) (friend_decks : ℕ) (total_spent : ℕ) 
  (h1 : tom_decks = 3)
  (h2 : friend_decks = 5)
  (h3 : total_spent = 64) :
  (total_spent : ℚ) / (tom_decks + friend_decks : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_trick_deck_cost_l1916_191680


namespace NUMINAMATH_CALUDE_evaluate_expression_l1916_191673

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/3) (hz : z = -6) :
  x^2 * y^3 * z^2 = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1916_191673


namespace NUMINAMATH_CALUDE_first_reduction_percentage_l1916_191659

theorem first_reduction_percentage (x : ℝ) : 
  (1 - 0.7) * (1 - x / 100) = 1 - 0.775 → x = 25 := by sorry

end NUMINAMATH_CALUDE_first_reduction_percentage_l1916_191659


namespace NUMINAMATH_CALUDE_point_placement_result_l1916_191698

theorem point_placement_result (x : ℕ) : ∃ x > 0, 9 * x - 8 = 82 := by
  sorry

#check point_placement_result

end NUMINAMATH_CALUDE_point_placement_result_l1916_191698


namespace NUMINAMATH_CALUDE_rhombuses_in_5x5_grid_l1916_191687

/-- Represents a grid of equilateral triangles -/
structure TriangleGrid where
  rows : Nat
  cols : Nat

/-- Counts the number of rhombuses in a triangle grid -/
def count_rhombuses (grid : TriangleGrid) : Nat :=
  sorry

/-- Theorem: In a 5x5 grid of equilateral triangles, there are 30 rhombuses -/
theorem rhombuses_in_5x5_grid :
  let grid : TriangleGrid := { rows := 5, cols := 5 }
  count_rhombuses grid = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombuses_in_5x5_grid_l1916_191687


namespace NUMINAMATH_CALUDE_product_of_squared_fractions_l1916_191634

theorem product_of_squared_fractions : (1/3 * 9)^2 * (1/27 * 81)^2 * (1/243 * 729)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squared_fractions_l1916_191634


namespace NUMINAMATH_CALUDE_max_a_value_l1916_191676

theorem max_a_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ a : ℝ, m^2 - a*m*n + 2*n^2 ≥ 0 → a ≤ 2*Real.sqrt 2) ∧
  ∃ a : ℝ, a = 2*Real.sqrt 2 ∧ m^2 - a*m*n + 2*n^2 ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1916_191676


namespace NUMINAMATH_CALUDE_team_total_catch_l1916_191691

/-- Represents the number of days in the fishing competition -/
def competition_days : ℕ := 5

/-- Represents Jackson's daily catch -/
def jackson_daily_catch : ℕ := 6

/-- Represents Jonah's daily catch -/
def jonah_daily_catch : ℕ := 4

/-- Represents George's daily catch -/
def george_daily_catch : ℕ := 8

/-- Theorem stating the total catch of the team during the competition -/
theorem team_total_catch : 
  competition_days * (jackson_daily_catch + jonah_daily_catch + george_daily_catch) = 90 := by
  sorry

end NUMINAMATH_CALUDE_team_total_catch_l1916_191691


namespace NUMINAMATH_CALUDE_problem_solution_l1916_191694

/-- Represents the box of electronic products -/
structure Box where
  total : ℕ
  first_class : ℕ
  second_class : ℕ

/-- The probability of drawing a first-class product only on the third draw without replacement -/
def prob_first_class_third_draw (b : Box) : ℚ :=
  (b.second_class : ℚ) / b.total *
  ((b.second_class - 1) : ℚ) / (b.total - 1) *
  (b.first_class : ℚ) / (b.total - 2)

/-- The expected number of first-class products in n draws with replacement -/
def expected_first_class (b : Box) (n : ℕ) : ℚ :=
  (n : ℚ) * (b.first_class : ℚ) / b.total

/-- The box described in the problem -/
def problem_box : Box := { total := 5, first_class := 3, second_class := 2 }

theorem problem_solution :
  prob_first_class_third_draw problem_box = 1 / 10 ∧
  expected_first_class problem_box 10 = 6 := by
  sorry

#eval prob_first_class_third_draw problem_box
#eval expected_first_class problem_box 10

end NUMINAMATH_CALUDE_problem_solution_l1916_191694


namespace NUMINAMATH_CALUDE_john_finishes_at_305_l1916_191654

/-- Represents time in minutes since midnight -/
def Time := ℕ

/-- Converts hours and minutes to Time -/
def toTime (hours minutes : ℕ) : Time :=
  hours * 60 + minutes

/-- The time John starts working -/
def startTime : Time := toTime 9 0

/-- The time John finishes the fourth task -/
def fourthTaskEndTime : Time := toTime 13 0

/-- The number of tasks John completes -/
def totalTasks : ℕ := 6

/-- The number of tasks completed before the first break -/
def tasksBeforeBreak : ℕ := 1

/-- The duration of each break in minutes -/
def breakDuration : ℕ := 10

/-- Calculates the time John finishes all tasks -/
noncomputable def calculateEndTime : Time := sorry

theorem john_finishes_at_305 :
  calculateEndTime = toTime 15 5 := by sorry

end NUMINAMATH_CALUDE_john_finishes_at_305_l1916_191654


namespace NUMINAMATH_CALUDE_simplify_fraction_l1916_191645

theorem simplify_fraction : 8 * (15 / 4) * (-45 / 50) = -12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1916_191645


namespace NUMINAMATH_CALUDE_shaded_square_covering_all_columns_l1916_191618

def shaded_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => shaded_sequence n + (2 * (n + 1) - 1)

def column_position (n : ℕ) : ℕ :=
  (shaded_sequence n - 1) % 10 + 1

def all_columns_covered (n : ℕ) : Prop :=
  ∀ k : ℕ, k ∈ Finset.range 10 → ∃ i : ℕ, i ≤ n ∧ column_position i = k + 1

theorem shaded_square_covering_all_columns :
  all_columns_covered 20 ∧ ∀ m : ℕ, m < 20 → ¬ all_columns_covered m :=
sorry

end NUMINAMATH_CALUDE_shaded_square_covering_all_columns_l1916_191618


namespace NUMINAMATH_CALUDE_intersection_point_integer_coordinates_l1916_191616

theorem intersection_point_integer_coordinates (m : ℕ+) : 
  (∃ x y : ℤ, 17 * x + 7 * y = 1000 ∧ y = m * x + 2) ↔ m = 68 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_integer_coordinates_l1916_191616


namespace NUMINAMATH_CALUDE_second_drawn_number_l1916_191614

def systematicSampling (totalStudents : ℕ) (sampleSize : ℕ) (firstDrawn : ℕ) : ℕ → ℕ :=
  fun n => firstDrawn + (totalStudents / sampleSize) * (n - 1)

theorem second_drawn_number
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (firstDrawn : ℕ)
  (h1 : totalStudents = 500)
  (h2 : sampleSize = 50)
  (h3 : firstDrawn = 3) :
  systematicSampling totalStudents sampleSize firstDrawn 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_second_drawn_number_l1916_191614


namespace NUMINAMATH_CALUDE_replacement_paint_intensity_l1916_191670

theorem replacement_paint_intensity
  (original_intensity : ℝ)
  (new_mixture_intensity : ℝ)
  (fraction_replaced : ℝ)
  (replacement_intensity : ℝ)
  (h1 : original_intensity = 50)
  (h2 : new_mixture_intensity = 40)
  (h3 : fraction_replaced = 1 / 3)
  (h4 : (1 - fraction_replaced) * original_intensity + fraction_replaced * replacement_intensity = new_mixture_intensity) :
  replacement_intensity = 20 := by
sorry

end NUMINAMATH_CALUDE_replacement_paint_intensity_l1916_191670


namespace NUMINAMATH_CALUDE_dance_theorem_l1916_191619

/-- Represents a dance function with boys and girls -/
structure DanceFunction where
  boys : ℕ
  girls : ℕ
  first_boy_dances : ℕ
  last_boy_dances_all : Prop

/-- The relationship between boys and girls in the dance function -/
def dance_relationship (df : DanceFunction) : Prop :=
  df.boys = df.girls - df.first_boy_dances + 1

theorem dance_theorem (df : DanceFunction) 
  (h1 : df.first_boy_dances = 6)
  (h2 : df.last_boy_dances_all)
  : df.boys = df.girls - 5 := by
  sorry

end NUMINAMATH_CALUDE_dance_theorem_l1916_191619


namespace NUMINAMATH_CALUDE_A_equals_set_l1916_191623

def A : Set ℤ := {x | -1 < |x - 1| ∧ |x - 1| < 2}

theorem A_equals_set : A = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_A_equals_set_l1916_191623


namespace NUMINAMATH_CALUDE_two_propositions_correct_l1916_191650

-- Define the original proposition
def original (x : ℝ) : Prop := x = 3 → x^2 - 7*x + 12 = 0

-- Define the converse proposition
def converse (x : ℝ) : Prop := x^2 - 7*x + 12 = 0 → x = 3

-- Define the inverse proposition
def inverse (x : ℝ) : Prop := x ≠ 3 → x^2 - 7*x + 12 ≠ 0

-- Define the contrapositive proposition
def contrapositive (x : ℝ) : Prop := x^2 - 7*x + 12 ≠ 0 → x ≠ 3

-- Theorem stating that exactly two propositions are correct
theorem two_propositions_correct :
  (∃! (n : ℕ), n = 2 ∧
    (∀ (x : ℝ), original x) ∧
    (∀ (x : ℝ), contrapositive x) ∧
    ¬(∀ (x : ℝ), converse x) ∧
    ¬(∀ (x : ℝ), inverse x)) :=
sorry

end NUMINAMATH_CALUDE_two_propositions_correct_l1916_191650


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1916_191639

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 3*x - 4 < 0} = Set.Ioo (-4 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1916_191639


namespace NUMINAMATH_CALUDE_coupons_used_proof_l1916_191684

/-- Calculates the total number of coupons used in a store's promotion --/
def total_coupons_used (initial_stock : ℝ) (books_sold : ℝ) (coupons_per_book : ℝ) : ℝ :=
  (initial_stock - books_sold) * coupons_per_book

/-- Proves that the total number of coupons used is 80.0 --/
theorem coupons_used_proof (initial_stock : ℝ) (books_sold : ℝ) (coupons_per_book : ℝ)
  (h1 : initial_stock = 40.0)
  (h2 : books_sold = 20.0)
  (h3 : coupons_per_book = 4.0) :
  total_coupons_used initial_stock books_sold coupons_per_book = 80.0 := by
  sorry

end NUMINAMATH_CALUDE_coupons_used_proof_l1916_191684


namespace NUMINAMATH_CALUDE_two_digit_fraction_problem_l1916_191644

theorem two_digit_fraction_problem :
  ∃ (A B : ℕ), 
    (10 ≤ A ∧ A ≤ 99) ∧ 
    (10 ≤ B ∧ B ≤ 99) ∧ 
    (A - 5 : ℚ) / A + 4 / B = 1 ∧
    (∀ A' : ℕ, (10 ≤ A' ∧ A' ≤ 99) → (A' - 5 : ℚ) / A' + 4 / B = 1 → A ≤ A') ∧
    (∀ B' : ℕ, (10 ≤ B' ∧ B' ≤ 99) → (A - 5 : ℚ) / A + 4 / B' = 1 → B' ≤ B) ∧
    A = 15 ∧ B = 76 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_fraction_problem_l1916_191644


namespace NUMINAMATH_CALUDE_infinite_square_free_triples_l1916_191600

/-- A positive integer is square-free if it's not divisible by any perfect square greater than 1 -/
def IsSquareFree (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → k * k ∣ n → k = 1

/-- The set of positive integers n for which n, n+1, and n+2 are all square-free -/
def SquareFreeTriples : Set ℕ :=
  {n : ℕ | n > 0 ∧ IsSquareFree n ∧ IsSquareFree (n + 1) ∧ IsSquareFree (n + 2)}

/-- The set of positive integers n for which n, n+1, and n+2 are all square-free is infinite -/
theorem infinite_square_free_triples : Set.Infinite SquareFreeTriples :=
sorry

end NUMINAMATH_CALUDE_infinite_square_free_triples_l1916_191600


namespace NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_is_9_64_l1916_191605

/-- The number of coin tosses -/
def n : ℕ := 10

/-- The probability of getting heads on a single toss -/
def p : ℚ := 1/2

/-- The number of ways to arrange k heads in n tosses without consecutive heads -/
def non_consecutive_heads (n k : ℕ) : ℕ := Nat.choose (n - k + 1) k

/-- The total number of favorable outcomes -/
def total_favorable_outcomes (n : ℕ) : ℕ :=
  (List.range (n/2 + 1)).map (non_consecutive_heads n) |>.sum

/-- The probability of not having consecutive heads in n fair coin tosses -/
def prob_no_consecutive_heads (n : ℕ) : ℚ :=
  (total_favorable_outcomes n : ℚ) / 2^n

/-- The main theorem -/
theorem prob_no_consecutive_heads_10_is_9_64 :
  prob_no_consecutive_heads n = 9/64 := by sorry

end NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_is_9_64_l1916_191605


namespace NUMINAMATH_CALUDE_elevator_trips_l1916_191640

def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def capacity : ℕ := 190

def is_valid_trip (trip : List ℕ) : Bool :=
  trip.sum ≤ capacity

def min_trips (masses : List ℕ) (capacity : ℕ) : ℕ :=
  sorry

theorem elevator_trips :
  min_trips masses capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_elevator_trips_l1916_191640


namespace NUMINAMATH_CALUDE_augmented_matrix_sum_l1916_191664

/-- Given a system of linear equations represented by an augmented matrix,
    prove that the sum of certain parameters equals 3. -/
theorem augmented_matrix_sum (m n : ℝ) : 
  (∃ (x y : ℝ), 2 * x = m ∧ n * x + y = 2 ∧ x = 1 ∧ y = 1) →
  m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_augmented_matrix_sum_l1916_191664


namespace NUMINAMATH_CALUDE_f_properties_l1916_191642

def f (x : ℝ) := x^3 + 2*x^2 - 4*x + 5

theorem f_properties :
  (f (-2) = 13) ∧
  (HasDerivAt f 0 (-2)) ∧
  (∀ x ∈ Set.Icc (-3) 0, f x ≤ 13) ∧
  (∃ x ∈ Set.Icc (-3) 0, f x = 13) ∧
  (∀ x ∈ Set.Icc (-3) 0, f x ≥ 5) ∧
  (∃ x ∈ Set.Icc (-3) 0, f x = 5) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1916_191642


namespace NUMINAMATH_CALUDE_smallest_sum_is_381_l1916_191689

/-- A permutation of the digits 1 to 6 -/
def Digit6Perm := Fin 6 → Fin 6

/-- Checks if a permutation is valid (bijective) -/
def isValidPerm (p : Digit6Perm) : Prop :=
  Function.Bijective p

/-- Converts a permutation to two 3-digit numbers -/
def permToNumbers (p : Digit6Perm) : ℕ × ℕ :=
  ((p 0 + 1) * 100 + (p 1 + 1) * 10 + (p 2 + 1),
   (p 3 + 1) * 100 + (p 4 + 1) * 10 + (p 5 + 1))

/-- Sums the two numbers obtained from a permutation -/
def sumFromPerm (p : Digit6Perm) : ℕ :=
  let (n1, n2) := permToNumbers p
  n1 + n2

/-- The main theorem stating that 381 is the smallest possible sum -/
theorem smallest_sum_is_381 :
  ∀ p : Digit6Perm, isValidPerm p → sumFromPerm p ≥ 381 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_is_381_l1916_191689


namespace NUMINAMATH_CALUDE_fewer_children_than_adults_l1916_191622

theorem fewer_children_than_adults : 
  ∀ (children seniors : ℕ),
  58 + children + seniors = 127 →
  seniors = 2 * children →
  58 - children = 35 := by
sorry

end NUMINAMATH_CALUDE_fewer_children_than_adults_l1916_191622


namespace NUMINAMATH_CALUDE_probability_all_white_balls_l1916_191660

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 7

theorem probability_all_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 8 / 6435 :=
sorry

end NUMINAMATH_CALUDE_probability_all_white_balls_l1916_191660


namespace NUMINAMATH_CALUDE_quadruple_solution_l1916_191662

theorem quadruple_solution (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (eq1 : a * b + c * d = 8)
  (eq2 : a * b * c * d = 8 + a + b + c + d) :
  a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 := by
sorry

end NUMINAMATH_CALUDE_quadruple_solution_l1916_191662


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1916_191604

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x
  (f 0 = 0) ∧ (f 3 = 0) ∧ (∀ x : ℝ, f x = 0 → x = 0 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1916_191604


namespace NUMINAMATH_CALUDE_pencil_carton_cost_pencil_carton_cost_proof_l1916_191675

/-- The cost of a carton of pencils given the following conditions:
  1. Erasers cost 3 dollars per carton
  2. Total order is 100 cartons
  3. Total order cost is 360 dollars
  4. The order includes 20 cartons of pencils -/
theorem pencil_carton_cost : ℝ :=
  let eraser_cost : ℝ := 3
  let total_cartons : ℕ := 100
  let total_cost : ℝ := 360
  let pencil_cartons : ℕ := 20
  6

/-- Proof that the cost of a carton of pencils is 6 dollars -/
theorem pencil_carton_cost_proof :
  pencil_carton_cost = 6 := by sorry

end NUMINAMATH_CALUDE_pencil_carton_cost_pencil_carton_cost_proof_l1916_191675


namespace NUMINAMATH_CALUDE_student_marks_average_l1916_191678

theorem student_marks_average (P C M : ℕ) (h : P + C + M = P + 150) :
  (C + M) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_average_l1916_191678


namespace NUMINAMATH_CALUDE_junior_rabbit_toys_l1916_191669

def toys_per_rabbit (num_rabbits : ℕ) (monday_toys : ℕ) : ℕ :=
  let wednesday_toys := 2 * monday_toys
  let friday_toys := 4 * monday_toys
  let saturday_toys := wednesday_toys / 2
  let total_toys := monday_toys + wednesday_toys + friday_toys + saturday_toys
  total_toys / num_rabbits

theorem junior_rabbit_toys :
  toys_per_rabbit 16 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_junior_rabbit_toys_l1916_191669


namespace NUMINAMATH_CALUDE_geometric_relations_l1916_191610

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Axioms
axiom different_lines {m n : Line} : m ≠ n
axiom different_planes {α β : Plane} : α ≠ β

-- Theorem
theorem geometric_relations 
  (m n : Line) (α β : Plane) :
  (perpendicular_plane m α ∧ perpendicular m n → 
    parallel_plane n α ∨ contains α n) ∧
  (parallel_planes α β ∧ perpendicular_plane n α ∧ parallel_plane m β → 
    perpendicular m n) ∧
  (parallel_plane m α ∧ perpendicular_plane n β ∧ perpendicular m n → 
    ¬(perpendicular_planes α β)) ∧
  (parallel_plane m α ∧ perpendicular_plane n β ∧ parallel m n → 
    perpendicular_planes α β) :=
by sorry


end NUMINAMATH_CALUDE_geometric_relations_l1916_191610


namespace NUMINAMATH_CALUDE_waitress_income_fraction_l1916_191671

theorem waitress_income_fraction (S : ℚ) : 
  let first_week_salary := S
  let first_week_tips := (11 / 4) * S
  let second_week_salary := (5 / 4) * S
  let second_week_tips := (7 / 3) * second_week_salary
  let total_salary := first_week_salary + second_week_salary
  let total_tips := first_week_tips + second_week_tips
  let total_income := total_salary + total_tips
  (total_tips / total_income) = 68 / 95 := by
  sorry

end NUMINAMATH_CALUDE_waitress_income_fraction_l1916_191671


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l1916_191688

theorem green_shirt_pairs 
  (total_students : ℕ) 
  (blue_shirts : ℕ) 
  (yellow_shirts : ℕ) 
  (green_shirts : ℕ) 
  (total_pairs : ℕ) 
  (blue_blue_pairs : ℕ) :
  total_students = 200 →
  blue_shirts = 70 →
  yellow_shirts = 80 →
  green_shirts = 50 →
  total_pairs = 100 →
  blue_blue_pairs = 30 →
  total_students = blue_shirts + yellow_shirts + green_shirts →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 25 := by
  sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l1916_191688


namespace NUMINAMATH_CALUDE_tom_finishes_30_min_after_anna_l1916_191667

/-- Represents the race scenario with given parameters -/
structure RaceScenario where
  distance : ℝ
  anna_speed : ℝ
  tom_speed : ℝ

/-- Calculates the finish time difference between Tom and Anna -/
def finishTimeDifference (race : RaceScenario) : ℝ :=
  race.distance * (race.tom_speed - race.anna_speed)

/-- Theorem stating that in the given race scenario, Tom finishes 30 minutes after Anna -/
theorem tom_finishes_30_min_after_anna :
  let race : RaceScenario := {
    distance := 15,
    anna_speed := 7,
    tom_speed := 9
  }
  finishTimeDifference race = 30 := by sorry

end NUMINAMATH_CALUDE_tom_finishes_30_min_after_anna_l1916_191667


namespace NUMINAMATH_CALUDE_problem_statement_l1916_191655

theorem problem_statement (x : ℝ) (h : x + 3 = 10) : 5 * x + 15 = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1916_191655


namespace NUMINAMATH_CALUDE_stacys_current_height_l1916_191643

/-- Prove Stacy's current height given the conditions of the problem -/
theorem stacys_current_height 
  (S J M S' J' M' : ℕ) 
  (h1 : S = 50)
  (h2 : S' = J' + 6)
  (h3 : J' = J + 1)
  (h4 : M' = M + 2 * (J' - J))
  (h5 : S + J + M = 128)
  (h6 : S' + J' + M' = 140) :
  S' = 59 := by
  sorry

end NUMINAMATH_CALUDE_stacys_current_height_l1916_191643


namespace NUMINAMATH_CALUDE_max_value_theorem_l1916_191636

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + 3 * b^2)) ≤ a^2 + 3 * b^2) ∧
  (∃ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + 3 * b^2)) = a^2 + 3 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1916_191636


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1916_191615

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 2*x - 8 = 0) :
  x^3 - 2*x^2 - 8*x + 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1916_191615


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l1916_191679

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2024 - 1) (2^2000 - 1) = 2^24 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l1916_191679


namespace NUMINAMATH_CALUDE_prime_factorization_sum_l1916_191632

theorem prime_factorization_sum (a b c : ℕ+) : 
  2^(a : ℕ) * 3^(b : ℕ) * 5^(c : ℕ) = 36000 → 3*(a : ℕ) + 4*(b : ℕ) + 6*(c : ℕ) = 41 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_sum_l1916_191632


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l1916_191638

theorem quadratic_equation_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   k * x₁^2 - Real.sqrt (2 * k + 1) * x₁ + 1 = 0 ∧
   k * x₂^2 - Real.sqrt (2 * k + 1) * x₂ + 1 = 0) ∧
  k ≠ 0 →
  -1/2 ≤ k ∧ k < 1/2 ∧ k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l1916_191638


namespace NUMINAMATH_CALUDE_cu_cn2_formation_l1916_191668

-- Define the chemical species
inductive Species
| HCN
| CuSO4
| CuCN2
| H2SO4

-- Define a type for chemical reactions
structure Reaction where
  reactants : List (Species × ℕ)
  products : List (Species × ℕ)

-- Define the balanced equation
def balancedEquation : Reaction :=
  { reactants := [(Species.HCN, 2), (Species.CuSO4, 1)]
  , products := [(Species.CuCN2, 1), (Species.H2SO4, 1)] }

-- Define the initial amounts of reactants
def initialHCN : ℕ := 2
def initialCuSO4 : ℕ := 1

-- Theorem statement
theorem cu_cn2_formation
  (reaction : Reaction)
  (hreaction : reaction = balancedEquation)
  (hHCN : initialHCN = 2)
  (hCuSO4 : initialCuSO4 = 1) :
  ∃ (amount : ℕ), amount = 1 ∧ 
  (Species.CuCN2, amount) ∈ reaction.products :=
sorry

end NUMINAMATH_CALUDE_cu_cn2_formation_l1916_191668


namespace NUMINAMATH_CALUDE_function_value_at_three_l1916_191628

/-- Given a positive real number and a function satisfying certain conditions,
    prove that the function evaluated at 3 equals 1/3. -/
theorem function_value_at_three (x : ℝ) (f : ℝ → ℝ) 
    (h1 : x > 0)
    (h2 : x + 17 = 60 * f x)
    (h3 : x = 3) : 
  f 3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l1916_191628


namespace NUMINAMATH_CALUDE_triangle_inequality_equality_condition_l1916_191629

theorem triangle_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x^2 + y^2 = z^2) : 
  x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) ≤ (2 + 3 * Real.sqrt 2) * x * y * z :=
sorry

theorem equality_condition (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x^2 + y^2 = z^2) : 
  x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) = (2 + 3 * Real.sqrt 2) * x * y * z ↔ x = y :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_equality_condition_l1916_191629


namespace NUMINAMATH_CALUDE_continuous_bounded_function_theorem_l1916_191661

theorem continuous_bounded_function_theorem (f : ℝ → ℝ) 
  (hcont : Continuous f) 
  (hbound : ∃ M, ∀ x, |f x| ≤ M) 
  (heq : ∀ x y, (f x)^2 - (f y)^2 = f (x + y) * f (x - y)) :
  ∃ a b : ℝ, ∀ x, f x = b * Real.sin (π * x / (2 * a)) := by
sorry

end NUMINAMATH_CALUDE_continuous_bounded_function_theorem_l1916_191661


namespace NUMINAMATH_CALUDE_pet_store_problem_l1916_191682

/-- The number of ways to choose pets for Emily, John, and Lucy -/
def pet_store_combinations (num_puppies num_kittens num_rabbits : ℕ) : ℕ :=
  num_puppies * num_kittens * num_rabbits * 6

/-- Theorem: Given 20 puppies, 10 kittens, and 12 rabbits, there are 14400 ways for
    Emily, John, and Lucy to buy pets, ensuring they all get different types of pets. -/
theorem pet_store_problem : pet_store_combinations 20 10 12 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_problem_l1916_191682


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1916_191647

theorem root_sum_theorem (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → 
  b^2 - 5*b + 6 = 0 → 
  a^3 + a^4*b^2 + a^2*b^4 + b^3 + a*b^3 + b*a^3 = 683 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1916_191647


namespace NUMINAMATH_CALUDE_add_45_minutes_to_10_20_l1916_191631

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := (totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, sorry⟩

theorem add_45_minutes_to_10_20 :
  addMinutes ⟨10, 20, sorry⟩ 45 = ⟨11, 5, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_add_45_minutes_to_10_20_l1916_191631


namespace NUMINAMATH_CALUDE_square_perimeter_l1916_191651

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 625) (h2 : side * side = area) :
  4 * side = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1916_191651


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1916_191612

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 4) + 6 = x^2 + x - 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1916_191612


namespace NUMINAMATH_CALUDE_isosceles_triangles_perimeter_l1916_191633

-- Define the points
variable (P Q R S : ℝ × ℝ)

-- Define the distances between points
def PQ : ℝ := sorry
def PR : ℝ := sorry
def QR : ℝ := sorry
def PS : ℝ := sorry
def SR : ℝ := sorry

-- Define x
def x : ℝ := sorry

-- State the theorem
theorem isosceles_triangles_perimeter (P Q R S : ℝ × ℝ) (PQ PR QR PS SR x : ℝ) :
  PQ = PR →                           -- Triangle PQR is isosceles
  PS = SR →                           -- Triangle PRS is isosceles
  PS = x →                            -- PS = x
  SR = x →                            -- SR = x
  PQ + QR + PR = 22 →                 -- Perimeter of Triangle PQR is 22
  PR + PS + SR = 22 →                 -- Perimeter of Triangle PRS is 22
  PQ + QR + SR + PS = 24 →            -- Perimeter of quadrilateral PQRS is 24
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_perimeter_l1916_191633


namespace NUMINAMATH_CALUDE_sum_of_three_circles_l1916_191657

-- Define the values for triangles and circles
variable (triangle : ℝ)
variable (circle : ℝ)

-- Define the conditions
axiom condition1 : 3 * triangle + 2 * circle = 21
axiom condition2 : 2 * triangle + 3 * circle = 19

-- Theorem to prove
theorem sum_of_three_circles : 3 * circle = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_circles_l1916_191657


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1916_191699

-- Define the vertices of the quadrilateral
def v1 : ℝ × ℝ := (0, 0)
def v2 : ℝ × ℝ := (4, 3)
def v3 : ℝ × ℝ := (7, 0)
def v4 : ℝ × ℝ := (4, 4)

-- Define the quadrilateral as a list of vertices
def quadrilateral : List (ℝ × ℝ) := [v1, v2, v3, v4]

-- Function to calculate the area of a quadrilateral using its vertices
def quadrilateralArea (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the area of the given quadrilateral is 3.5
theorem quadrilateral_area : quadrilateralArea quadrilateral = 3.5 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1916_191699


namespace NUMINAMATH_CALUDE_factory_production_l1916_191625

/-- Given a factory that produces a certain number of toys per week and workers
    that work a certain number of days per week, calculate the number of toys
    produced each day (rounded down). -/
def toysPerDay (toysPerWeek : ℕ) (daysWorked : ℕ) : ℕ :=
  toysPerWeek / daysWorked

/-- Theorem stating that for a factory producing 6400 toys per week with workers
    working 3 days a week, the number of toys produced each day is 2133. -/
theorem factory_production :
  toysPerDay 6400 3 = 2133 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l1916_191625


namespace NUMINAMATH_CALUDE_article_percentage_loss_l1916_191611

theorem article_percentage_loss 
  (selling_price : ℝ) 
  (selling_price_with_gain : ℝ) 
  (gain_percentage : ℝ) :
  selling_price = 136 →
  selling_price_with_gain = 192 →
  gain_percentage = 20 →
  let cost_price := selling_price_with_gain / (1 + gain_percentage / 100)
  let loss := cost_price - selling_price
  let percentage_loss := (loss / cost_price) * 100
  percentage_loss = 15 := by
sorry

end NUMINAMATH_CALUDE_article_percentage_loss_l1916_191611


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l1916_191672

/-- The largest perimeter of a triangle with two sides of 7 and 8 units, and the third side being an integer --/
theorem largest_triangle_perimeter : 
  ∀ x : ℤ, 
  (7 : ℝ) + 8 > x ∧ 
  (7 : ℝ) + x > 8 ∧ 
  8 + x > 7 →
  (∀ y : ℤ, 
    (7 : ℝ) + 8 > y ∧ 
    (7 : ℝ) + y > 8 ∧ 
    8 + y > 7 →
    7 + 8 + x ≥ 7 + 8 + y) →
  7 + 8 + x = 29 :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l1916_191672


namespace NUMINAMATH_CALUDE_fraction_simplification_l1916_191695

theorem fraction_simplification (a b m : ℝ) (hb : b ≠ 0) (hm : m ≠ 0) :
  (a * m) / (b * m) = a / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1916_191695


namespace NUMINAMATH_CALUDE_passes_through_point1_passes_through_point2_unique_line_l1916_191606

/-- A line passing through two points (-1, 0) and (0, 2) -/
def line (x y : ℝ) : Prop := y = 2 * x + 2

/-- The line passes through the point (-1, 0) -/
theorem passes_through_point1 : line (-1) 0 := by sorry

/-- The line passes through the point (0, 2) -/
theorem passes_through_point2 : line 0 2 := by sorry

/-- The equation y = 2x + 2 represents the unique line passing through (-1, 0) and (0, 2) -/
theorem unique_line : ∀ (x y : ℝ), (y = 2 * x + 2) ↔ line x y := by sorry

end NUMINAMATH_CALUDE_passes_through_point1_passes_through_point2_unique_line_l1916_191606


namespace NUMINAMATH_CALUDE_conversion_theorem_l1916_191653

-- Define conversion rates
def meters_per_km : ℝ := 1000
def minutes_per_hour : ℝ := 60

-- Problem 1: Convert 70 kilometers and 50 meters to kilometers
def problem1 (km : ℝ) (m : ℝ) : Prop :=
  km + m / meters_per_km = 70.05

-- Problem 2: Convert 3.6 hours to hours and minutes
def problem2 (h : ℝ) : Prop :=
  ∃ (whole_hours : ℕ) (minutes : ℕ),
    h = whole_hours + (minutes : ℝ) / minutes_per_hour ∧
    whole_hours = 3 ∧
    minutes = 36

theorem conversion_theorem :
  problem1 70 50 ∧ problem2 3.6 := by sorry

end NUMINAMATH_CALUDE_conversion_theorem_l1916_191653


namespace NUMINAMATH_CALUDE_f_minimum_and_inequality_l1916_191635

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_minimum_and_inequality :
  (∃ (x : ℝ), x > 0 ∧ f x = -1 / Real.exp 1) ∧
  (∀ (x : ℝ), x > 0 → Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x)) := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_and_inequality_l1916_191635


namespace NUMINAMATH_CALUDE_solve_for_k_l1916_191686

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x^3 - 2 * x + 4
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + x^2 - 5 * x - k

-- State the theorem
theorem solve_for_k : ∃ k : ℝ, f 3 - g k 3 = 14 ∧ k = -17 := by sorry

end NUMINAMATH_CALUDE_solve_for_k_l1916_191686


namespace NUMINAMATH_CALUDE_choose_two_from_four_l1916_191648

theorem choose_two_from_four (n : ℕ) (h : n = 4) : Nat.choose n 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_four_l1916_191648


namespace NUMINAMATH_CALUDE_rational_function_value_l1916_191677

/-- A rational function with specific properties -/
def rational_function (p q : ℝ → ℝ) : Prop :=
  (∃ k : ℝ, ∀ x, p x = k * x) ∧  -- p is linear
  (∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c) ∧  -- q is quadratic
  p 0 / q 0 = 0 ∧  -- passes through (0,0)
  p 4 / q 4 = 2 ∧  -- passes through (4,2)
  q (-4) = 0 ∧  -- vertical asymptote at x = -4
  q 1 = 0  -- vertical asymptote at x = 1

theorem rational_function_value (p q : ℝ → ℝ) :
  rational_function p q → p (-1) / q (-1) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l1916_191677


namespace NUMINAMATH_CALUDE_train_length_l1916_191609

/-- Calculates the length of a train given its speed and the time and distance it takes to cross a bridge -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (1000 / 3600) →
  bridge_length = 150 →
  crossing_time = 20 →
  (train_speed * crossing_time) - bridge_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1916_191609


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1916_191692

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (6 * x) + Real.sin (10 * x) = 2 * Real.sin (8 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1916_191692


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1916_191696

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ), 
  (X^3 + 2*X^2 - 3 : Polynomial ℝ) = (X^2 + 2) * q + r ∧ 
  r.degree < (X^2 + 2).degree ∧
  r = -2*X - 7 :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1916_191696


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1916_191665

/-- The proposition "If m > 0, then the equation x^2 + x - m = 0 has real roots" -/
def original_proposition (m : ℝ) : Prop :=
  m > 0 → ∃ x : ℝ, x^2 + x - m = 0

/-- The contrapositive of the original proposition -/
def contrapositive (m : ℝ) : Prop :=
  (¬∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0

/-- Theorem stating that the contrapositive is equivalent to the expected form -/
theorem contrapositive_equivalence :
  ∀ m : ℝ, contrapositive m ↔ (¬∃ x : ℝ, x^2 + x - m = 0 → m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1916_191665


namespace NUMINAMATH_CALUDE_problem_solution_l1916_191601

theorem problem_solution (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12) 
  (h2 : a^2 + b^2 + c^2 = a*b + b*c + c*a) : 
  a + b^2 + c^3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1916_191601


namespace NUMINAMATH_CALUDE_tangent_slope_circle_l1916_191658

/-- Slope of the tangent line to a circle -/
theorem tangent_slope_circle (center_x center_y tangent_x tangent_y : ℝ) :
  center_x = 2 →
  center_y = 3 →
  tangent_x = 7 →
  tangent_y = 8 →
  (tangent_y - center_y) / (tangent_x - center_x) = 1 →
  -(((tangent_y - center_y) / (tangent_x - center_x))⁻¹) = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_circle_l1916_191658


namespace NUMINAMATH_CALUDE_subcommittee_formation_ways_l1916_191613

def senate_committee_ways (total_republicans : ℕ) (total_democrats : ℕ) 
                          (subcommittee_republicans : ℕ) (subcommittee_democrats : ℕ) : ℕ :=
  Nat.choose total_republicans subcommittee_republicans * 
  Nat.choose total_democrats subcommittee_democrats

theorem subcommittee_formation_ways : 
  senate_committee_ways 10 8 4 3 = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_ways_l1916_191613


namespace NUMINAMATH_CALUDE_teds_overall_correct_percentage_l1916_191685

theorem teds_overall_correct_percentage
  (t : ℝ) -- total number of problems
  (h_t_pos : t > 0) -- ensure t is positive
  (independent_solving : ℝ := 0.4 * t) -- 40% of problems solved independently
  (collaborative_solving : ℝ := 0.6 * t) -- 60% of problems solved collaboratively
  (ned_independent_correct : ℝ := 0.7 * independent_solving) -- Ned's correct answers for independent solving
  (ned_overall_correct : ℝ := 0.82 * t) -- Ned's overall correct answers
  (ted_independent_correct : ℝ := 0.85 * independent_solving) -- Ted's correct answers for independent solving
  : (ted_independent_correct + (ned_overall_correct - ned_independent_correct)) / t = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_teds_overall_correct_percentage_l1916_191685


namespace NUMINAMATH_CALUDE_chucks_team_lead_l1916_191620

/-- Represents a team in the basketball match -/
inductive Team
| ChucksTeam
| YellowTeam

/-- Represents a quarter in the basketball match -/
inductive Quarter
| First
| Second
| Third
| Fourth

/-- Calculates the score for a given team in a given quarter -/
def quarterScore (team : Team) (quarter : Quarter) : ℤ :=
  match team, quarter with
  | Team.ChucksTeam, Quarter.First => 23
  | Team.ChucksTeam, Quarter.Second => 18
  | Team.ChucksTeam, Quarter.Third => 19
  | Team.ChucksTeam, Quarter.Fourth => 17
  | Team.YellowTeam, Quarter.First => 24
  | Team.YellowTeam, Quarter.Second => 19
  | Team.YellowTeam, Quarter.Third => 14
  | Team.YellowTeam, Quarter.Fourth => 16

/-- Points gained from technical fouls -/
def technicalFoulPoints (team : Team) : ℤ :=
  match team with
  | Team.ChucksTeam => 3
  | Team.YellowTeam => 2

/-- Calculates the total score for a team -/
def totalScore (team : Team) : ℤ :=
  quarterScore team Quarter.First +
  quarterScore team Quarter.Second +
  quarterScore team Quarter.Third +
  quarterScore team Quarter.Fourth +
  technicalFoulPoints team

/-- The main theorem stating Chuck's Team's lead -/
theorem chucks_team_lead :
  totalScore Team.ChucksTeam - totalScore Team.YellowTeam = 5 := by
  sorry


end NUMINAMATH_CALUDE_chucks_team_lead_l1916_191620


namespace NUMINAMATH_CALUDE_swing_slide_wait_time_difference_l1916_191627

theorem swing_slide_wait_time_difference :
  let swingKids : ℕ := 6
  let slideKids : ℕ := 4 * swingKids
  let swingWaitTime1 : ℝ := 3.5 * 60  -- 3.5 minutes in seconds
  let slideWaitTime1 : ℝ := 45  -- 45 seconds
  let rounds : ℕ := 3
  
  let swingTotalWait : ℝ := swingKids * (swingWaitTime1 * (1 - 2^rounds) / (1 - 2))
  let slideTotalWait : ℝ := slideKids * (slideWaitTime1 * (1 - 2^rounds) / (1 - 2))
  
  swingTotalWait - slideTotalWait = 1260
  := by sorry

end NUMINAMATH_CALUDE_swing_slide_wait_time_difference_l1916_191627


namespace NUMINAMATH_CALUDE_specific_normal_distribution_mean_l1916_191652

/-- A normal distribution with given properties -/
structure NormalDistribution where
  μ : ℝ  -- arithmetic mean
  σ : ℝ  -- standard deviation
  value_2sd_below : ℝ  -- value 2 standard deviations below the mean

/-- Theorem stating the properties of the specific normal distribution -/
theorem specific_normal_distribution_mean 
  (d : NormalDistribution) 
  (h1 : d.σ = 2.3)
  (h2 : d.value_2sd_below = 11.6)
  (h3 : d.value_2sd_below = d.μ - 2 * d.σ) : 
  d.μ = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_specific_normal_distribution_mean_l1916_191652


namespace NUMINAMATH_CALUDE_inscribed_square_area_is_2275_l1916_191621

/-- A right triangle with an inscribed square -/
structure InscribedSquareTriangle where
  /-- The length of side XY of the triangle -/
  xy : ℝ
  /-- The length of side ZQ of the triangle -/
  zq : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- Condition ensuring the square fits in the triangle -/
  square_fits : square_side ≤ min xy zq

/-- The area of the inscribed square in the triangle -/
def inscribed_square_area (t : InscribedSquareTriangle) : ℝ := t.square_side ^ 2

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area_is_2275 
  (t : InscribedSquareTriangle) 
  (h1 : t.xy = 35) 
  (h2 : t.zq = 65) : 
  inscribed_square_area t = 2275 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_is_2275_l1916_191621


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l1916_191693

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + m = 0 ∧ x = 1) → m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l1916_191693


namespace NUMINAMATH_CALUDE_min_ballots_proof_l1916_191608

/-- Represents the number of candidates for each position -/
def candidates : List Nat := [3, 4, 5]

/-- Represents the requirement that each candidate must appear under each number
    an equal number of times -/
def equal_appearance (ballots : Nat) : Prop :=
  ∀ n ∈ candidates, ballots % n = 0

/-- The minimum number of different ballots required -/
def min_ballots : Nat := 5

/-- Theorem stating that the minimum number of ballots satisfying the equal appearance
    requirement is 5 -/
theorem min_ballots_proof :
  (∀ k : Nat, k < min_ballots → ¬(equal_appearance k)) ∧
  (equal_appearance min_ballots) :=
sorry

end NUMINAMATH_CALUDE_min_ballots_proof_l1916_191608


namespace NUMINAMATH_CALUDE_B_age_is_18_l1916_191666

/-- Given three people A, B, and C with the following conditions:
  1. A is two years older than B
  2. B is twice as old as C
  3. The sum of their ages is 47
  Prove that B is 18 years old -/
theorem B_age_is_18 (A B C : ℕ) 
  (h1 : A = B + 2)
  (h2 : B = 2 * C)
  (h3 : A + B + C = 47) :
  B = 18 := by
  sorry

end NUMINAMATH_CALUDE_B_age_is_18_l1916_191666


namespace NUMINAMATH_CALUDE_blender_sunday_price_l1916_191681

/-- The Sunday price of a blender after applying discounts -/
theorem blender_sunday_price (original_price : ℝ) (regular_discount : ℝ) (sunday_discount : ℝ) :
  original_price = 250 →
  regular_discount = 0.60 →
  sunday_discount = 0.25 →
  original_price * (1 - regular_discount) * (1 - sunday_discount) = 75 := by
sorry

end NUMINAMATH_CALUDE_blender_sunday_price_l1916_191681


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l1916_191624

/-- Given two 2D vectors a and b, if a + b is parallel to a - b, 
    then the first component of a is -4/3. -/
theorem vector_parallel_condition (a b : ℝ × ℝ) :
  a.1 = m ∧ a.2 = 2 ∧ b = (2, -3) ∧ 
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (a - b)) →
  m = -4/3 := by sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l1916_191624


namespace NUMINAMATH_CALUDE_solve_equation_l1916_191697

theorem solve_equation : ∃ x : ℚ, 5 * (x - 9) = 7 * (3 - 3 * x) + 10 ∧ x = 38 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1916_191697


namespace NUMINAMATH_CALUDE_only_one_satisfies_property_l1916_191674

theorem only_one_satisfies_property : ∃! (n : ℕ), 
  n > 0 ∧ 
  (∀ (a : ℤ), Odd a → (a^2 : ℤ) ≤ n → a ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_only_one_satisfies_property_l1916_191674


namespace NUMINAMATH_CALUDE_sum_200_consecutive_integers_l1916_191603

theorem sum_200_consecutive_integers (n : ℕ) : 
  (n = 2000200000 ∨ n = 3000300000 ∨ n = 4000400000 ∨ n = 5000500000 ∨ n = 6000600000) →
  ¬∃ k : ℕ, n = (200 * (k + 100)) + 10050 := by
  sorry

end NUMINAMATH_CALUDE_sum_200_consecutive_integers_l1916_191603


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1916_191690

theorem binomial_expansion_coefficient (a : ℝ) : 
  (Nat.choose 6 3 : ℝ) * a^3 * 2^3 = 5/2 → a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1916_191690


namespace NUMINAMATH_CALUDE_min_shift_for_symmetry_l1916_191683

theorem min_shift_for_symmetry (m : ℝ) : 
  m > 0 ∧ 
  (∀ x : ℝ, 2 * Real.sin (x + m + π/3) = 2 * Real.sin (-x + m + π/3)) →
  m ≥ π/6 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_for_symmetry_l1916_191683


namespace NUMINAMATH_CALUDE_simplify_fraction_l1916_191617

theorem simplify_fraction : (5^3 + 5^5) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1916_191617
