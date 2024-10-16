import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1747_174728

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence definition
  r > 1 →  -- increasing sequence
  a 1 + a 3 + a 5 = 21 →  -- given condition
  a 3 = 6 →  -- given condition
  a 5 + a 7 + a 9 = 84 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1747_174728


namespace NUMINAMATH_CALUDE_local_minimum_implies_a_eq_neg_two_monotone_increasing_implies_a_nonneg_l1747_174796

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.exp x

theorem local_minimum_implies_a_eq_neg_two (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 1| < δ → f a x ≥ f a 1) → a = -2 := by sorry

theorem monotone_increasing_implies_a_nonneg (a : ℝ) :
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f a x < f a y) → a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_local_minimum_implies_a_eq_neg_two_monotone_increasing_implies_a_nonneg_l1747_174796


namespace NUMINAMATH_CALUDE_quadratic_roots_shift_l1747_174700

theorem quadratic_roots_shift (a b c : ℝ) (ha : a ≠ 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (y : ℝ) := a * y^2 + (b - 2*a) * y + (a - b + c)
  ∀ (x y : ℝ), f x = 0 ∧ g y = 0 → y = x + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_shift_l1747_174700


namespace NUMINAMATH_CALUDE_dans_initial_money_l1747_174736

/-- The amount of money Dan spent on the candy bar -/
def candy_bar_cost : ℝ := 1.00

/-- The amount of money Dan has left after buying the candy bar -/
def money_left : ℝ := 2.00

/-- Dan's initial amount of money -/
def initial_money : ℝ := candy_bar_cost + money_left

theorem dans_initial_money : initial_money = 3.00 := by sorry

end NUMINAMATH_CALUDE_dans_initial_money_l1747_174736


namespace NUMINAMATH_CALUDE_yan_position_ratio_l1747_174739

/-- Yan's position between home and stadium -/
structure Position where
  home_dist : ℝ     -- distance from home
  stadium_dist : ℝ  -- distance to stadium
  home_dist_nonneg : 0 ≤ home_dist
  stadium_dist_nonneg : 0 ≤ stadium_dist

/-- Yan's walking speed -/
def walking_speed : ℝ := 1

/-- Yan's cycling speed -/
def cycling_speed : ℝ := 4 * walking_speed

theorem yan_position_ratio (pos : Position) : 
  (pos.home_dist / pos.stadium_dist = 3 / 5) ↔ 
  (pos.stadium_dist / walking_speed = 
   pos.home_dist / walking_speed + (pos.home_dist + pos.stadium_dist) / cycling_speed) := by
  sorry

end NUMINAMATH_CALUDE_yan_position_ratio_l1747_174739


namespace NUMINAMATH_CALUDE_max_third_term_arithmetic_sequence_greatest_third_term_l1747_174763

theorem max_third_term_arithmetic_sequence (a d : ℕ) (h1 : 0 < a) (h2 : 0 < d) 
  (h3 : a + (a + d) + (a + 2*d) + (a + 3*d) = 52) : 
  ∀ (x y : ℕ), 0 < x → 0 < y → 
  x + (x + y) + (x + 2*y) + (x + 3*y) = 52 → 
  x + 2*y ≤ a + 2*d := by
sorry

theorem greatest_third_term : 
  ∃ (a d : ℕ), 0 < a ∧ 0 < d ∧ 
  a + (a + d) + (a + 2*d) + (a + 3*d) = 52 ∧
  a + 2*d = 17 ∧
  (∀ (x y : ℕ), 0 < x → 0 < y → 
   x + (x + y) + (x + 2*y) + (x + 3*y) = 52 → 
   x + 2*y ≤ 17) := by
sorry

end NUMINAMATH_CALUDE_max_third_term_arithmetic_sequence_greatest_third_term_l1747_174763


namespace NUMINAMATH_CALUDE_scalene_triangle_bisector_inequality_l1747_174732

/-- Given a scalene triangle with longest angle bisector l₁, shortest angle bisector l₂, and area S,
    prove that l₁² > √3 S > l₂². -/
theorem scalene_triangle_bisector_inequality (a b c : ℝ) (h_scalene : a > b ∧ b > c ∧ c > 0) :
  ∃ (l₁ l₂ S : ℝ), l₁ > 0 ∧ l₂ > 0 ∧ S > 0 ∧
  (∀ l : ℝ, (l > 0 ∧ l ≠ l₁ ∧ l ≠ l₂) → (l < l₁ ∧ l > l₂)) ∧
  S = (1/2) * b * c * Real.sin ((2/3) * Real.pi) ∧
  l₁^2 > Real.sqrt 3 * S ∧ Real.sqrt 3 * S > l₂^2 :=
by sorry

end NUMINAMATH_CALUDE_scalene_triangle_bisector_inequality_l1747_174732


namespace NUMINAMATH_CALUDE_chemistry_textbook_weight_l1747_174721

/-- The weight of Kelly's chemistry textbook in pounds -/
def chemistry_weight : ℝ := sorry

/-- The weight of Kelly's geometry textbook in pounds -/
def geometry_weight : ℝ := 0.625

theorem chemistry_textbook_weight :
  chemistry_weight = geometry_weight + 6.5 ∧ chemistry_weight = 7.125 := by sorry

end NUMINAMATH_CALUDE_chemistry_textbook_weight_l1747_174721


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_eq_five_l1747_174717

/-- The number of zeroes to the right of the decimal point and before the first non-zero digit
    in the decimal representation of 1/(2^3 * 5^6) -/
def zeros_before_first_nonzero : ℕ :=
  let denominator := 2^3 * 5^6
  let decimal_places := 6  -- log_10(denominator)
  decimal_places - 1

theorem zeros_before_first_nonzero_eq_five :
  zeros_before_first_nonzero = 5 := by
  sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_eq_five_l1747_174717


namespace NUMINAMATH_CALUDE_sin_15_75_simplification_l1747_174748

theorem sin_15_75_simplification : 2 * Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_75_simplification_l1747_174748


namespace NUMINAMATH_CALUDE_expected_subtree_size_ten_vertices_l1747_174702

-- Define a type for rooted trees
structure RootedTree where
  vertices : Nat
  root : Nat

-- Define a function to represent the expected subtree size
def expectedSubtreeSize (t : RootedTree) : ℚ :=
  sorry

-- Theorem statement
theorem expected_subtree_size_ten_vertices :
  ∀ t : RootedTree,
  t.vertices = 10 →
  expectedSubtreeSize t = 7381 / 2520 :=
by sorry

end NUMINAMATH_CALUDE_expected_subtree_size_ten_vertices_l1747_174702


namespace NUMINAMATH_CALUDE_square_root_division_problem_l1747_174743

theorem square_root_division_problem : ∃ x : ℝ, (Real.sqrt 5184) / x = 4 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_root_division_problem_l1747_174743


namespace NUMINAMATH_CALUDE_father_age_l1747_174701

/-- Represents the ages of a family -/
structure FamilyAges where
  yy : ℕ
  cousin : ℕ
  mother : ℕ
  father : ℕ

/-- Defines the conditions of the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.yy = ages.cousin + 3 ∧
  ages.father = ages.mother + 4 ∧
  ages.yy + ages.cousin + ages.mother + ages.father = 95 ∧
  (ages.yy - 8) + (ages.cousin - 8) + (ages.mother - 8) + (ages.father - 8) = 65

/-- The theorem to be proved -/
theorem father_age (ages : FamilyAges) :
  problem_conditions ages → ages.father = 42 := by
  sorry

end NUMINAMATH_CALUDE_father_age_l1747_174701


namespace NUMINAMATH_CALUDE_group_arrangements_eq_40_l1747_174783

/-- The number of ways to divide 2 teachers and 6 students into two groups,
    each consisting of 1 teacher and 3 students. -/
def group_arrangements : ℕ :=
  (Nat.choose 2 1) * (Nat.choose 6 3)

/-- Theorem stating that the number of group arrangements is 40. -/
theorem group_arrangements_eq_40 : group_arrangements = 40 := by
  sorry

end NUMINAMATH_CALUDE_group_arrangements_eq_40_l1747_174783


namespace NUMINAMATH_CALUDE_class_average_weight_l1747_174725

theorem class_average_weight (students_a : ℕ) (students_b : ℕ) (avg_weight_a : ℝ) (avg_weight_b : ℝ)
  (h1 : students_a = 36)
  (h2 : students_b = 44)
  (h3 : avg_weight_a = 40)
  (h4 : avg_weight_b = 35) :
  let total_students := students_a + students_b
  let total_weight := students_a * avg_weight_a + students_b * avg_weight_b
  total_weight / total_students = 37.25 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l1747_174725


namespace NUMINAMATH_CALUDE_infinitely_many_primes_with_large_primitive_root_l1747_174793

theorem infinitely_many_primes_with_large_primitive_root (n : ℕ) (hn : n > 0) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ p ∈ S,
    Nat.Prime p ∧ ∀ m ∈ Finset.range n, ∃ x, x^2 ≡ m [MOD p] :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_with_large_primitive_root_l1747_174793


namespace NUMINAMATH_CALUDE_yoe_speed_calculation_l1747_174731

/-- Yoe's speed in miles per hour -/
def yoe_speed : ℝ := 40

/-- Teena's speed in miles per hour -/
def teena_speed : ℝ := 55

/-- Initial distance between Teena and Yoe in miles (Teena behind) -/
def initial_distance : ℝ := 7.5

/-- Time elapsed in hours -/
def time_elapsed : ℝ := 1.5

/-- Final distance between Teena and Yoe in miles (Teena ahead) -/
def final_distance : ℝ := 15

theorem yoe_speed_calculation : 
  yoe_speed = (teena_speed * time_elapsed - initial_distance - final_distance) / time_elapsed :=
by sorry

end NUMINAMATH_CALUDE_yoe_speed_calculation_l1747_174731


namespace NUMINAMATH_CALUDE_pattern_two_odd_one_even_l1747_174749

/-- A box containing 100 balls numbered from 1 to 100 -/
def Box := Finset (Fin 100)

/-- The set of odd-numbered balls in the box -/
def OddBalls (box : Box) : Finset (Fin 100) :=
  box.filter (fun n => n % 2 = 1)

/-- The set of even-numbered balls in the box -/
def EvenBalls (box : Box) : Finset (Fin 100) :=
  box.filter (fun n => n % 2 = 0)

/-- A selection pattern of 3 balls -/
structure SelectionPattern :=
  (first second third : Bool)

/-- The probability of selecting an odd-numbered ball first -/
def ProbFirstOdd (pattern : SelectionPattern) : ℚ :=
  if pattern.first then 2/3 else 1/3

theorem pattern_two_odd_one_even
  (box : Box)
  (h_box_size : box.card = 100)
  (h_prob_first_odd : ∃ pattern : SelectionPattern, ProbFirstOdd pattern = 2/3) :
  ∃ pattern : SelectionPattern,
    pattern.first ≠ pattern.second ∨ pattern.first ≠ pattern.third ∨ pattern.second ≠ pattern.third :=
sorry

end NUMINAMATH_CALUDE_pattern_two_odd_one_even_l1747_174749


namespace NUMINAMATH_CALUDE_tetrahedron_divides_space_l1747_174792

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  faces : Fin 4 → Plane
  edges : Fin 6 → Line
  vertices : Fin 4 → Point

/-- The number of regions formed by the planes of a tetrahedron's faces -/
def num_regions (t : Tetrahedron) : ℕ := 15

/-- Theorem stating that the planes of a tetrahedron's faces divide space into 15 regions -/
theorem tetrahedron_divides_space (t : Tetrahedron) : 
  num_regions t = 15 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_divides_space_l1747_174792


namespace NUMINAMATH_CALUDE_employee_payment_percentage_l1747_174787

theorem employee_payment_percentage (total_payment y_payment x_payment : ℝ) :
  total_payment = 770 →
  y_payment = 350 →
  x_payment + y_payment = total_payment →
  x_payment / y_payment = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_employee_payment_percentage_l1747_174787


namespace NUMINAMATH_CALUDE_johns_grass_height_l1747_174755

/-- The height to which John cuts his grass -/
def cut_height : ℝ := 2

/-- The monthly growth rate of the grass in inches -/
def growth_rate : ℝ := 0.5

/-- The maximum height of the grass before cutting in inches -/
def max_height : ℝ := 4

/-- The number of times John cuts his grass per year -/
def cuts_per_year : ℕ := 3

/-- The number of months between each cutting -/
def months_between_cuts : ℕ := 4

theorem johns_grass_height :
  cut_height + growth_rate * months_between_cuts = max_height :=
sorry

end NUMINAMATH_CALUDE_johns_grass_height_l1747_174755


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1747_174753

-- Define the inequality function
def f (x : ℝ) : ℝ := x^2 - 2*x - 8

-- Define the solution set
def solution_set : Set ℝ := {x | x ≤ -2 ∨ x ≥ 4}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1747_174753


namespace NUMINAMATH_CALUDE_subset_implies_all_elements_in_l1747_174751

theorem subset_implies_all_elements_in : 
  ∀ (A B : Set α), A.Nonempty → B.Nonempty → A ⊆ B → ∀ x ∈ A, x ∈ B := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_all_elements_in_l1747_174751


namespace NUMINAMATH_CALUDE_factorization_problems_l1747_174723

theorem factorization_problems :
  (∀ a : ℝ, 18 * a^2 - 32 = 2 * (3*a + 4) * (3*a - 4)) ∧
  (∀ x y : ℝ, y - 6*x*y + 9*x^2*y = y * (1 - 3*x)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l1747_174723


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1747_174789

theorem simplify_complex_fraction :
  (1 / ((2 / (Real.sqrt 5 + 2)) + (3 / (Real.sqrt 7 - 2)))) =
  ((2 * Real.sqrt 5 + Real.sqrt 7 + 2) / (23 + 4 * Real.sqrt 35)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1747_174789


namespace NUMINAMATH_CALUDE_jakes_third_test_score_l1747_174703

/-- Given Jake's test scores, prove he scored 65 in the third test -/
theorem jakes_third_test_score :
  -- Define the number of tests
  let num_tests : ℕ := 4
  -- Define the average score
  let average_score : ℚ := 75
  -- Define the score of the first test
  let first_test_score : ℕ := 80
  -- Define the score difference between second and first tests
  let second_test_difference : ℕ := 10
  -- Define the condition that third and fourth test scores are equal
  ∀ (third_test_score fourth_test_score : ℕ),
    -- Total score equals average multiplied by number of tests
    (first_test_score + (first_test_score + second_test_difference) + third_test_score + fourth_test_score : ℚ) = num_tests * average_score →
    -- Third and fourth test scores are equal
    third_test_score = fourth_test_score →
    -- Prove that the third test score is 65
    third_test_score = 65 := by
  sorry

end NUMINAMATH_CALUDE_jakes_third_test_score_l1747_174703


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1747_174733

theorem systematic_sampling_theorem (total_workers : ℕ) (sample_size : ℕ) (start_num : ℕ) (interval_start : ℕ) (interval_end : ℕ) : 
  total_workers = 840 →
  sample_size = 42 →
  start_num = 21 →
  interval_start = 421 →
  interval_end = 720 →
  (interval_end - interval_start + 1) / (total_workers / sample_size) = 15 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1747_174733


namespace NUMINAMATH_CALUDE_marble_jar_problem_l1747_174734

theorem marble_jar_problem (total_marbles : ℕ) : 
  (∃ (marbles_per_person : ℕ), 
    total_marbles = 18 * marbles_per_person ∧ 
    total_marbles = 20 * (marbles_per_person - 1)) → 
  total_marbles = 180 := by
sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l1747_174734


namespace NUMINAMATH_CALUDE_truck_toll_calculation_l1747_174794

/-- Calculate the toll for a truck based on its number of axles -/
def toll (x : ℕ) : ℚ := 2.5 + 0.5 * (x - 2)

/-- Calculate the number of axles for a truck given its wheel configuration -/
def axle_count (total_wheels front_wheels other_axle_wheels : ℕ) : ℕ :=
  1 + (total_wheels - front_wheels) / other_axle_wheels

theorem truck_toll_calculation (total_wheels front_wheels other_axle_wheels : ℕ) 
  (h1 : total_wheels = 18)
  (h2 : front_wheels = 2)
  (h3 : other_axle_wheels = 4) :
  toll (axle_count total_wheels front_wheels other_axle_wheels) = 4 := by
  sorry

end NUMINAMATH_CALUDE_truck_toll_calculation_l1747_174794


namespace NUMINAMATH_CALUDE_number_wall_value_l1747_174711

/-- Represents a simplified Number Wall with four bottom values and a top value --/
structure NumberWall where
  bottom_left : ℕ
  bottom_mid_left : ℕ
  bottom_mid_right : ℕ
  bottom_right : ℕ
  top : ℕ

/-- The Number Wall is valid if it follows the construction rules --/
def is_valid_number_wall (w : NumberWall) : Prop :=
  ∃ (mid_left mid_right : ℕ),
    w.bottom_left + w.bottom_mid_left = mid_left ∧
    w.bottom_mid_left + w.bottom_mid_right = mid_right ∧
    w.bottom_mid_right + w.bottom_right = w.top - mid_left ∧
    mid_left + mid_right = w.top

theorem number_wall_value (w : NumberWall) 
    (h : is_valid_number_wall w)
    (h1 : w.bottom_mid_left = 6)
    (h2 : w.bottom_mid_right = 10)
    (h3 : w.bottom_right = 9)
    (h4 : w.top = 64) :
  w.bottom_left = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_value_l1747_174711


namespace NUMINAMATH_CALUDE_always_quadratic_in_x_l1747_174776

theorem always_quadratic_in_x (k : ℝ) :
  ∃ a b c : ℝ, a ≠ 0 ∧
  ∀ x : ℝ, (k^2 + 1) * x^2 - (k * x - 8) - 1 = a * x^2 + b * x + c :=
by sorry

end NUMINAMATH_CALUDE_always_quadratic_in_x_l1747_174776


namespace NUMINAMATH_CALUDE_distinct_roots_of_f_l1747_174765

-- Define the function
def f (x : ℝ) : ℝ := (x - 5) * (x + 3)^2

-- Theorem statement
theorem distinct_roots_of_f :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ f r₁ = 0 ∧ f r₂ = 0 ∧ ∀ x, f x = 0 → x = r₁ ∨ x = r₂ :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_of_f_l1747_174765


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1747_174715

theorem no_integer_solutions (n k m l : ℕ) : 
  l ≥ 2 → 4 ≤ k → k ≤ n - 4 → Nat.choose n k ≠ m^l := by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1747_174715


namespace NUMINAMATH_CALUDE_solution_set_equality_l1747_174712

open Set

-- Define the solution set
def solutionSet : Set ℝ := {x | |2*x + 1| > 3}

-- State the theorem
theorem solution_set_equality : solutionSet = Iio (-2) ∪ Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l1747_174712


namespace NUMINAMATH_CALUDE_extremum_point_of_f_l1747_174714

def f (x : ℝ) := x^2 + 1

theorem extremum_point_of_f :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_extremum_point_of_f_l1747_174714


namespace NUMINAMATH_CALUDE_min_lcm_a_c_l1747_174775

theorem min_lcm_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 24) (h2 : Nat.lcm b c = 18) :
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 12 ∧ 
    (∀ (x y : ℕ), Nat.lcm x b = 24 → Nat.lcm b y = 18 → Nat.lcm x y ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_min_lcm_a_c_l1747_174775


namespace NUMINAMATH_CALUDE_high_school_math_club_payment_l1747_174777

theorem high_school_math_club_payment (A : Nat) : 
  A < 10 → (2 * 100 + A * 10 + 3) % 3 = 0 → A = 1 ∨ A = 4 := by
  sorry

end NUMINAMATH_CALUDE_high_school_math_club_payment_l1747_174777


namespace NUMINAMATH_CALUDE_hall_length_proof_l1747_174704

/-- Proves that a hall with given dimensions and mat cost has a specific length -/
theorem hall_length_proof (width height mat_cost_per_sqm total_cost : ℝ) 
  (h_width : width = 15)
  (h_height : height = 5)
  (h_mat_cost : mat_cost_per_sqm = 40)
  (h_total_cost : total_cost = 38000) :
  ∃ (length : ℝ), 
    length = 32 ∧ 
    total_cost = mat_cost_per_sqm * (length * width + 2 * length * height + 2 * width * height) :=
by sorry

end NUMINAMATH_CALUDE_hall_length_proof_l1747_174704


namespace NUMINAMATH_CALUDE_four_hearts_probability_l1747_174771

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
| Hearts | Diamonds | Clubs | Spades

/-- Represents the rank of a card -/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function that maps a card index to its suit -/
def card_to_suit : Fin 52 → Suit := sorry

/-- A function that maps a card index to its rank -/
def card_to_rank : Fin 52 → Rank := sorry

/-- The number of hearts in a standard deck -/
def hearts_count : Nat := 13

/-- Theorem: The probability of drawing four hearts as the top four cards from a standard 52-card deck is 286/108290 -/
theorem four_hearts_probability (d : Deck) : 
  (hearts_count * (hearts_count - 1) * (hearts_count - 2) * (hearts_count - 3)) / 
  (d.cards.card * (d.cards.card - 1) * (d.cards.card - 2) * (d.cards.card - 3)) = 286 / 108290 :=
sorry

end NUMINAMATH_CALUDE_four_hearts_probability_l1747_174771


namespace NUMINAMATH_CALUDE_johnny_signature_dish_count_l1747_174722

/-- Represents the number of times Johnny makes his signature crab dish in a day -/
def signature_dish_count : ℕ := sorry

/-- The amount of crab meat used in each signature dish (in pounds) -/
def crab_meat_per_dish : ℚ := 3/2

/-- The price of crab meat per pound (in dollars) -/
def crab_meat_price : ℕ := 8

/-- The total amount Johnny spends on crab meat in a week (in dollars) -/
def weekly_spending : ℕ := 1920

/-- The number of days Johnny's restaurant is closed in a week -/
def closed_days : ℕ := 3

/-- The number of days Johnny's restaurant is open in a week -/
def open_days : ℕ := 7 - closed_days

theorem johnny_signature_dish_count :
  signature_dish_count = 40 :=
by sorry

end NUMINAMATH_CALUDE_johnny_signature_dish_count_l1747_174722


namespace NUMINAMATH_CALUDE_product_zero_l1747_174730

theorem product_zero (b : ℤ) (h : b = 3) : 
  (b - 13) * (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * 
  (b - 6) * (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_zero_l1747_174730


namespace NUMINAMATH_CALUDE_probability_twelve_rolls_last_l1747_174709

/-- The probability of getting the same number on the 12th roll as on the 11th roll,
    given that all previous pairs of consecutive rolls were different. -/
theorem probability_twelve_rolls_last (d : ℕ) (h : d = 6) : 
  (((d - 1) / d) ^ 10 * (1 / d) : ℚ) = 9765625 / 362797056 := by
  sorry

end NUMINAMATH_CALUDE_probability_twelve_rolls_last_l1747_174709


namespace NUMINAMATH_CALUDE_garden_area_increase_l1747_174798

/-- Given a rectangular garden with dimensions 60 feet by 20 feet,
    prove that changing it to a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rectangle_length : ℝ := 60
  let rectangle_width : ℝ := 20
  let rectangle_perimeter : ℝ := 2 * (rectangle_length + rectangle_width)
  let square_side : ℝ := rectangle_perimeter / 4
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let square_area : ℝ := square_side ^ 2
  square_area - rectangle_area = 400 := by
sorry

end NUMINAMATH_CALUDE_garden_area_increase_l1747_174798


namespace NUMINAMATH_CALUDE_quadratic_vertex_on_x_axis_l1747_174705

/-- The quadratic function -x^2 + 4x + t has its vertex on the x-axis if and only if t = -4 -/
theorem quadratic_vertex_on_x_axis (t : ℝ) : 
  (∃ x : ℝ, ∀ y : ℝ, y = -x^2 + 4*x + t → y = 0) ↔ t = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_on_x_axis_l1747_174705


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l1747_174720

theorem no_positive_integer_solution (m n : ℕ+) : 4 * m * (m + 1) ≠ n * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l1747_174720


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1747_174758

-- Problem 1
theorem problem_1 : -9 / 3 + (1 / 2 - 2 / 3) * 12 - |(-4)^3| = -69 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : 2 * (a^2 + 2*b^2) - 3 * (2*a^2 - b^2) = -4*a^2 + 7*b^2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1747_174758


namespace NUMINAMATH_CALUDE_hexagon_probability_l1747_174795

/-- Represents a hexagonal checkerboard -/
structure HexBoard :=
  (total_hexagons : ℕ)
  (side_length : ℕ)

/-- Calculates the number of hexagons on the perimeter of the board -/
def perimeter_hexagons (board : HexBoard) : ℕ :=
  6 * board.side_length - 6

/-- Calculates the number of hexagons not on the perimeter of the board -/
def inner_hexagons (board : HexBoard) : ℕ :=
  board.total_hexagons - perimeter_hexagons board

/-- Theorem: The probability of a randomly chosen hexagon not touching the outer edge -/
theorem hexagon_probability (board : HexBoard) 
  (h1 : board.total_hexagons = 91)
  (h2 : board.side_length = 5) :
  (inner_hexagons board : ℚ) / board.total_hexagons = 67 / 91 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_probability_l1747_174795


namespace NUMINAMATH_CALUDE_complex_expression_equals_nine_l1747_174708

theorem complex_expression_equals_nine :
  (0.4 + 8 * (5 - 0.8 * (5 / 8)) - 5 / (2 + 1 / 2)) /
  ((1 + 7 / 8) * 8 - (8.9 - 2.6 / (2 / 3))) * (34 + 2 / 5) * 90 = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_nine_l1747_174708


namespace NUMINAMATH_CALUDE_valid_sequences_of_length_20_l1747_174773

/-- Counts valid binary sequences of given length -/
def countValidSequences (n : ℕ) : ℕ :=
  if n < 3 then 0
  else if n = 3 then 1
  else countValidSequences (n - 4) + 2 * countValidSequences (n - 5) + countValidSequences (n - 6)

/-- Theorem stating the number of valid sequences of length 20 -/
theorem valid_sequences_of_length_20 :
  countValidSequences 20 = 86 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_of_length_20_l1747_174773


namespace NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l1747_174782

/-- Represents the probability of not winning any ticket when buying n tickets -/
def prob_no_win (total : ℕ) (rate : ℚ) (n : ℕ) : ℚ :=
  (1 - rate) ^ n

theorem lottery_not_guaranteed_win (total : ℕ) (rate : ℚ) (n : ℕ) 
  (h_total : total = 100000)
  (h_rate : rate = 1 / 1000)
  (h_n : n = 2000) :
  prob_no_win total rate n > 0 := by
  sorry

#check lottery_not_guaranteed_win

end NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l1747_174782


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l1747_174747

theorem triangle_abc_theorem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are sides opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  Real.cos (2 * A) + 2 * Real.sin (π + B) ^ 2 + 2 * Real.cos (π / 2 + C) ^ 2 - 1 = 2 * Real.sin B * Real.sin C →
  -- Given side lengths
  b = 4 ∧ c = 5 →
  -- Conclusions
  A = π / 3 ∧ Real.sin B = 2 * Real.sqrt 7 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l1747_174747


namespace NUMINAMATH_CALUDE_decimal_point_problem_l1747_174797

theorem decimal_point_problem : ∃ x : ℝ, x > 0 ∧ 1000 * x = 9 * (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l1747_174797


namespace NUMINAMATH_CALUDE_new_cards_count_l1747_174724

theorem new_cards_count (cards_per_page : ℕ) (old_cards : ℕ) (total_pages : ℕ) : 
  cards_per_page = 3 → old_cards = 16 → total_pages = 8 → 
  total_pages * cards_per_page - old_cards = 8 := by
  sorry

end NUMINAMATH_CALUDE_new_cards_count_l1747_174724


namespace NUMINAMATH_CALUDE_two_tangents_from_three_zero_l1747_174744

/-- The curve y = x^2 - 2x -/
def curve (x : ℝ) : ℝ := x^2 - 2*x

/-- Condition for two tangents to exist from a point (a, b) to the curve -/
def two_tangents_condition (a b : ℝ) : Prop :=
  a^2 - 2*a - b > 0

/-- Theorem stating that (3, 0) satisfies the two tangents condition -/
theorem two_tangents_from_three_zero :
  two_tangents_condition 3 0 := by
  sorry

end NUMINAMATH_CALUDE_two_tangents_from_three_zero_l1747_174744


namespace NUMINAMATH_CALUDE_leak_empty_time_proof_l1747_174784

/-- Represents the time it takes for a pipe to fill a tank -/
def fill_time_no_leak : ℝ := 8

/-- Represents the time it takes for a pipe to fill a tank with a leak -/
def fill_time_with_leak : ℝ := 12

/-- Represents the time it takes for the leak to empty a full tank -/
def leak_empty_time : ℝ := 24

/-- Theorem stating that given the fill times with and without a leak, 
    the time for the leak to empty the tank is 24 hours -/
theorem leak_empty_time_proof : 
  ∀ (fill_rate : ℝ) (leak_rate : ℝ),
  fill_rate = 1 / fill_time_no_leak →
  fill_rate - leak_rate = 1 / fill_time_with_leak →
  1 / leak_rate = leak_empty_time :=
by sorry

end NUMINAMATH_CALUDE_leak_empty_time_proof_l1747_174784


namespace NUMINAMATH_CALUDE_intersection_quadrilateral_perimeter_bounds_l1747_174750

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  a : ℝ
  a_pos : 0 < a

/-- A quadrilateral formed by the intersection of a plane and a regular tetrahedron -/
structure IntersectionQuadrilateral (t : RegularTetrahedron) where
  perimeter : ℝ

/-- The theorem stating that the perimeter of the intersection quadrilateral
    is bounded between 2a and 3a -/
theorem intersection_quadrilateral_perimeter_bounds
  (t : RegularTetrahedron) (q : IntersectionQuadrilateral t) :
  2 * t.a ≤ q.perimeter ∧ q.perimeter ≤ 3 * t.a :=
sorry

end NUMINAMATH_CALUDE_intersection_quadrilateral_perimeter_bounds_l1747_174750


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l1747_174770

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l1747_174770


namespace NUMINAMATH_CALUDE_negation_of_existential_inequality_l1747_174740

theorem negation_of_existential_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_inequality_l1747_174740


namespace NUMINAMATH_CALUDE_circle_radius_is_one_l1747_174788

-- Define the polar equation of the circle
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Define the Cartesian equation of the circle
def cartesian_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Theorem statement
theorem circle_radius_is_one :
  ∀ ρ θ x y : ℝ,
  polar_equation ρ θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  cartesian_equation x y →
  1 = (x^2 + y^2).sqrt :=
by sorry


end NUMINAMATH_CALUDE_circle_radius_is_one_l1747_174788


namespace NUMINAMATH_CALUDE_no_zero_root_l1747_174738

-- Define the three equations
def equation1 (x : ℝ) : Prop := 4 * x^2 - 4 = 36
def equation2 (x : ℝ) : Prop := (2*x + 1)^2 = (x + 2)^2
def equation3 (x : ℝ) : Prop := (x^2 - 9 : ℝ) = x + 2

-- Theorem statement
theorem no_zero_root :
  (∀ x : ℝ, equation1 x → x ≠ 0) ∧
  (∀ x : ℝ, equation2 x → x ≠ 0) ∧
  (∀ x : ℝ, equation3 x → x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_no_zero_root_l1747_174738


namespace NUMINAMATH_CALUDE_sets_partition_integers_l1747_174799

theorem sets_partition_integers (A B : Set ℤ) 
  (h1 : A ∪ B = (Set.univ : Set ℤ))
  (h2 : ∀ x : ℤ, x ∈ A → x - 1 ∈ B)
  (h3 : ∀ x y : ℤ, x ∈ B → y ∈ B → x + y ∈ A) :
  A = {x : ℤ | ∃ k : ℤ, x = 2 * k} ∧ 
  B = {x : ℤ | ∃ k : ℤ, x = 2 * k + 1} :=
by sorry

end NUMINAMATH_CALUDE_sets_partition_integers_l1747_174799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l1747_174779

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_b (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_of_b b n + b (n + 1)

theorem arithmetic_sequence_m_value
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a3 : a 3 = 5)
  (h_a9 : a 9 = 17)
  (h_sum_b : ∀ n : ℕ, sum_of_b b n = 3^n - 1)
  (h_relation : ∃ m : ℕ, m > 0 ∧ 1 + a m = b 4) :
  ∃ m : ℕ, m > 0 ∧ 1 + a m = b 4 ∧ m = 27 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l1747_174779


namespace NUMINAMATH_CALUDE_conical_container_height_l1747_174742

theorem conical_container_height (d : ℝ) (n : ℕ) (h r : ℝ) : 
  d = 64 ∧ n = 4 ∧ (π * d^2 / 4) = n * (π * r * (d / 2)) ∧ h^2 + r^2 = (d / 2)^2 
  → h = 8 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_conical_container_height_l1747_174742


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l1747_174780

theorem sqrt_difference_inequality : 
  let a := Real.sqrt 2023 - Real.sqrt 2022
  let b := Real.sqrt 2022 - Real.sqrt 2021
  let c := Real.sqrt 2021 - Real.sqrt 2020
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l1747_174780


namespace NUMINAMATH_CALUDE_inequality_proof_l1747_174713

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1747_174713


namespace NUMINAMATH_CALUDE_commercial_length_l1747_174785

theorem commercial_length (original_length : ℝ) : 
  (original_length * 0.7 = 21) → original_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_commercial_length_l1747_174785


namespace NUMINAMATH_CALUDE_largest_selected_is_57_l1747_174710

/-- Represents the systematic sampling of students. -/
structure StudentSampling where
  total_students : Nat
  first_selected : Nat
  second_selected : Nat

/-- Calculates the sample interval based on the first two selected numbers. -/
def sample_interval (s : StudentSampling) : Nat :=
  s.second_selected - s.first_selected

/-- Calculates the number of selected students. -/
def num_selected (s : StudentSampling) : Nat :=
  s.total_students / sample_interval s

/-- Calculates the largest selected number. -/
def largest_selected (s : StudentSampling) : Nat :=
  s.first_selected + (sample_interval s) * (num_selected s - 1)

/-- Theorem stating that the largest selected number is 57 for the given conditions. -/
theorem largest_selected_is_57 (s : StudentSampling) 
    (h1 : s.total_students = 60)
    (h2 : s.first_selected = 3)
    (h3 : s.second_selected = 9) : 
  largest_selected s = 57 := by
  sorry

#eval largest_selected { total_students := 60, first_selected := 3, second_selected := 9 }

end NUMINAMATH_CALUDE_largest_selected_is_57_l1747_174710


namespace NUMINAMATH_CALUDE_circle_common_chord_l1747_174727

variables (a b x y : ℝ)

-- Define the first circle
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*a*x = 0

-- Define the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*b*y = 0

-- Define the resulting circle
def resultCircle (x y : ℝ) : Prop := (a^2 + b^2)*(x^2 + y^2) - 2*a*b*(b*x + a*y) = 0

-- Theorem statement
theorem circle_common_chord (hb : b ≠ 0) :
  ∃ (x y : ℝ), circle1 a x y ∧ circle2 b x y →
  resultCircle a b x y ∧
  ∀ (x' y' : ℝ), resultCircle a b x' y' →
    ∃ (t : ℝ), x' = x + t*(y - x) ∧ y' = y + t*(x - y) :=
sorry

end NUMINAMATH_CALUDE_circle_common_chord_l1747_174727


namespace NUMINAMATH_CALUDE_savings_ratio_l1747_174741

/-- Represents the number of cans collected from different sources -/
structure CanCollection where
  home : ℕ
  grandparents : ℕ
  neighbor : ℕ
  office : ℕ

/-- Calculates the total number of cans collected -/
def total_cans (c : CanCollection) : ℕ :=
  c.home + c.grandparents + c.neighbor + c.office

/-- Represents the problem setup -/
structure RecyclingProblem where
  collection : CanCollection
  price_per_can : ℚ
  savings_amount : ℚ

/-- Main theorem: The ratio of savings to total amount collected is 1:2 -/
theorem savings_ratio (p : RecyclingProblem)
  (h1 : p.collection.home = 12)
  (h2 : p.collection.grandparents = 3 * p.collection.home)
  (h3 : p.collection.neighbor = 46)
  (h4 : p.collection.office = 250)
  (h5 : p.price_per_can = 1/4)
  (h6 : p.savings_amount = 43) :
  p.savings_amount / (p.price_per_can * total_cans p.collection) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_savings_ratio_l1747_174741


namespace NUMINAMATH_CALUDE_meghan_weight_conversion_l1747_174764

/-- Given a base b, this function calculates the value of a number represented as 451 in base b -/
def value_in_base_b (b : ℕ) : ℕ := 4 * b^2 + 5 * b + 1

/-- Given a base b, this function calculates the value of a number represented as 127 in base 2b -/
def value_in_base_2b (b : ℕ) : ℕ := 1 * (2*b)^2 + 2 * (2*b) + 7

/-- Theorem stating that if a number is represented as 451 in base b and 127 in base 2b, 
    then it is equal to 175 in base 10 -/
theorem meghan_weight_conversion (b : ℕ) : 
  value_in_base_b b = value_in_base_2b b → value_in_base_b b = 175 := by
  sorry

#eval value_in_base_b 6  -- Should output 175
#eval value_in_base_2b 6  -- Should also output 175

end NUMINAMATH_CALUDE_meghan_weight_conversion_l1747_174764


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1747_174766

theorem profit_percentage_calculation (cost_price selling_price : ℝ) :
  cost_price = 240 →
  selling_price = 288 →
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1747_174766


namespace NUMINAMATH_CALUDE_average_equation_solution_l1747_174761

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((x + 6) + (6*x + 2) + (2*x + 7)) = 4*x - 7 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l1747_174761


namespace NUMINAMATH_CALUDE_opposite_of_negative_sqrt_two_l1747_174790

theorem opposite_of_negative_sqrt_two : -(-Real.sqrt 2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_sqrt_two_l1747_174790


namespace NUMINAMATH_CALUDE_magic_card_profit_100_l1747_174729

/-- Calculates the profit from selling a Magic card that has tripled in value --/
def magic_card_profit (purchase_price : ℝ) : ℝ :=
  3 * purchase_price - purchase_price

theorem magic_card_profit_100 :
  magic_card_profit 100 = 200 := by
  sorry

#eval magic_card_profit 100

end NUMINAMATH_CALUDE_magic_card_profit_100_l1747_174729


namespace NUMINAMATH_CALUDE_sqrt_equation_implies_value_l1747_174726

theorem sqrt_equation_implies_value (a b : ℝ) :
  (Real.sqrt (a - 2 * b + 4) + (a + b - 5) ^ 2 = 0) →
  (4 * Real.sqrt a - Real.sqrt 24 / Real.sqrt b = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_implies_value_l1747_174726


namespace NUMINAMATH_CALUDE_function_inequality_l1747_174767

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, HasDerivAt f (f' x) x) 
  (h' : ∀ x, f' x > f x) : f (Real.log 2022) > 2022 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1747_174767


namespace NUMINAMATH_CALUDE_max_value_expression_l1747_174762

theorem max_value_expression (a b c d : ℝ) 
  (ha : -8.5 ≤ a ∧ a ≤ 8.5)
  (hb : -8.5 ≤ b ∧ b ≤ 8.5)
  (hc : -8.5 ≤ c ∧ c ≤ 8.5)
  (hd : -8.5 ≤ d ∧ d ≤ 8.5) :
  (∀ x y z w, -8.5 ≤ x ∧ x ≤ 8.5 ∧ 
              -8.5 ≤ y ∧ y ≤ 8.5 ∧ 
              -8.5 ≤ z ∧ z ≤ 8.5 ∧ 
              -8.5 ≤ w ∧ w ≤ 8.5 → 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 306) ∧
  (∃ x y z w, -8.5 ≤ x ∧ x ≤ 8.5 ∧ 
              -8.5 ≤ y ∧ y ≤ 8.5 ∧ 
              -8.5 ≤ z ∧ z ≤ 8.5 ∧ 
              -8.5 ≤ w ∧ w ≤ 8.5 ∧ 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 306) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1747_174762


namespace NUMINAMATH_CALUDE_rectangle_area_is_six_l1747_174718

/-- Represents a square within the rectangle ABCD -/
structure Square where
  side_length : ℝ
  area : ℝ
  area_eq : area = side_length ^ 2

/-- The rectangle ABCD containing three squares -/
structure Rectangle where
  squares : Fin 3 → Square
  non_overlapping : ∀ i j, i ≠ j → (squares i).area + (squares j).area ≤ area
  shaded_square_area : (squares 0).area = 1
  area : ℝ

/-- The theorem stating that the area of rectangle ABCD is 6 square inches -/
theorem rectangle_area_is_six (rect : Rectangle) : rect.area = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_six_l1747_174718


namespace NUMINAMATH_CALUDE_no_common_root_with_specific_values_l1747_174786

theorem no_common_root_with_specific_values : ¬ ∃ (P₁ P₂ : ℤ → ℤ) (a b : ℤ),
  (∀ x, ∃ (c : ℤ), P₁ x = c) ∧  -- P₁ has integer coefficients
  (∀ x, ∃ (c : ℤ), P₂ x = c) ∧  -- P₂ has integer coefficients
  a < 0 ∧                       -- a is strictly negative
  P₁ a = 0 ∧                    -- a is a root of P₁
  P₂ a = 0 ∧                    -- a is a root of P₂
  b > 0 ∧                       -- b is positive
  P₁ b = 2007 ∧                 -- P₁ evaluates to 2007 at b
  P₂ b = 2008                   -- P₂ evaluates to 2008 at b
  := by sorry

end NUMINAMATH_CALUDE_no_common_root_with_specific_values_l1747_174786


namespace NUMINAMATH_CALUDE_special_function_at_one_l1747_174716

/-- A function satisfying certain properties on positive real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f (1 / x) = x * f x) ∧
  (∀ x > 0, ∀ y > 0, f x + f y = x + y + f (x * y))

/-- The value of f(1) for a function satisfying the special properties -/
theorem special_function_at_one (f : ℝ → ℝ) (h : special_function f) : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_one_l1747_174716


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_ratio_l1747_174768

/-- The ratio of areas between an inscribed hexagon with side length s/2 
    and an outer hexagon with side length s is 1/4 -/
theorem inscribed_hexagon_area_ratio (s : ℝ) (h : s > 0) : 
  (3 * Real.sqrt 3 * (s/2)^2 / 2) / (3 * Real.sqrt 3 * s^2 / 2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_ratio_l1747_174768


namespace NUMINAMATH_CALUDE_parabola_sum_is_line_l1747_174745

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola -/
def original : QuadraticFunction :=
  { a := 3, b := 4, c := -5 }

/-- Reflects a quadratic function about the x-axis -/
def reflect (f : QuadraticFunction) : QuadraticFunction :=
  { a := -f.a, b := -f.b, c := -f.c }

/-- Translates a quadratic function horizontally -/
def translate (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := f.b - 2 * f.a * d
  , c := f.a * d^2 - f.b * d + f.c }

/-- Adds two quadratic functions -/
def add (f g : QuadraticFunction) : QuadraticFunction :=
  { a := f.a + g.a
  , b := f.b + g.b
  , c := f.c + g.c }

/-- Theorem stating that the sum of the translated original parabola and its reflected and translated version is a non-horizontal line -/
theorem parabola_sum_is_line :
  let f := translate original 4
  let g := translate (reflect original) (-6)
  let sum := add f g
  sum.a = 0 ∧ sum.b ≠ 0 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_is_line_l1747_174745


namespace NUMINAMATH_CALUDE_grid_constant_l1747_174752

/-- A function representing the assignment of positive integers to grid points -/
def GridAssignment := ℤ → ℤ → ℕ+

/-- The condition that each value is the arithmetic mean of its neighbors -/
def is_arithmetic_mean (f : GridAssignment) : Prop :=
  ∀ x y : ℤ, (f x y : ℚ) = ((f (x-1) y + f (x+1) y + f x (y-1) + f x (y+1)) : ℚ) / 4

/-- The main theorem: if a grid assignment satisfies the arithmetic mean condition,
    then it is constant across the entire grid -/
theorem grid_constant (f : GridAssignment) (h : is_arithmetic_mean f) :
  ∀ x y x' y' : ℤ, f x y = f x' y' :=
sorry

end NUMINAMATH_CALUDE_grid_constant_l1747_174752


namespace NUMINAMATH_CALUDE_complex_modulus_l1747_174706

theorem complex_modulus (z : ℂ) (h : z + 2*Complex.I - 3 = 3 - 3*Complex.I) : 
  Complex.abs z = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1747_174706


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l1747_174772

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define point H
def H : ℝ × ℝ := (1, -1)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 3/2)^2 = 25/4

-- Theorem statement
theorem tangent_circle_equation :
  ∃ (A B : ℝ × ℝ),
    (parabola A.1 A.2) ∧
    (parabola B.1 B.2) ∧
    (∃ (m₁ m₂ : ℝ),
      (A.2 - H.2 = m₁ * (A.1 - H.1)) ∧
      (B.2 - H.2 = m₂ * (B.1 - H.1)) ∧
      (∀ (x y : ℝ), parabola x y → m₁ * (x - A.1) + A.2 ≥ y) ∧
      (∀ (x y : ℝ), parabola x y → m₂ * (x - B.1) + B.2 ≥ y)) →
    ∀ (x y : ℝ), circle_equation x y ↔ 
      ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ x = t * A.1 + (1 - t) * B.1 ∧ y = t * A.2 + (1 - t) * B.2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l1747_174772


namespace NUMINAMATH_CALUDE_least_number_with_special_property_l1747_174757

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by another number -/
def is_divisible_by (n m : ℕ) : Prop := sorry

/-- The least positive integer whose digits add to a multiple of 27 yet the number itself is not a multiple of 27 -/
theorem least_number_with_special_property : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), sum_of_digits n = 27 * k) ∧ 
  ¬(is_divisible_by n 27) ∧
  (∀ (m : ℕ), m < n → 
    ((∃ (k : ℕ), sum_of_digits m = 27 * k) → (is_divisible_by m 27))) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_special_property_l1747_174757


namespace NUMINAMATH_CALUDE_unique_prime_triplet_l1747_174756

theorem unique_prime_triplet :
  ∀ p q r : ℕ,
    Prime p → Prime q → Prime r →
    p + q = r →
    ∃ n : ℕ, (r - p) * (q - p) - 27 * p = n^2 →
    p = 2 ∧ q = 29 ∧ r = 31 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triplet_l1747_174756


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l1747_174719

theorem employee_pay_percentage (total_pay y_pay : ℝ) (h1 : total_pay = 570) (h2 : y_pay = 259.09) :
  let x_pay := total_pay - y_pay
  (x_pay / y_pay) * 100 = 120.03 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l1747_174719


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l1747_174769

theorem sum_of_roots_zero (a b c x y : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq1 : a^3 + a*x + y = 0)
  (h_eq2 : b^3 + b*x + y = 0)
  (h_eq3 : c^3 + c*x + y = 0) :
  a + b + c = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l1747_174769


namespace NUMINAMATH_CALUDE_max_pages_copied_l1747_174754

-- Define the cost per page in cents
def cost_per_page : ℕ := 3

-- Define the budget in dollars
def budget : ℕ := 15

-- Define the function to calculate the number of pages
def pages_copied (cost : ℕ) (budget : ℕ) : ℕ :=
  (budget * 100) / cost

-- Theorem statement
theorem max_pages_copied :
  pages_copied cost_per_page budget = 500 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_copied_l1747_174754


namespace NUMINAMATH_CALUDE_propositions_correctness_l1747_174774

theorem propositions_correctness :
  (∃ a b, a > b ∧ b > 0 ∧ (1 / a ≥ 1 / b)) ∧
  (∀ a b, a > b ∧ b > 0 → a^2 - a > b^2 - b) ∧
  (∃ a b, a > b ∧ b > 0 ∧ a^3 ≤ b^3) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ 2*a + b = 1 → 
    (∀ x y, x > 0 ∧ y > 0 ∧ 2*x + y = 1 → a^2 + b^2 ≤ x^2 + y^2) ∧
    a^2 + b^2 = 1/9) :=
by sorry

end NUMINAMATH_CALUDE_propositions_correctness_l1747_174774


namespace NUMINAMATH_CALUDE_bankers_discount_example_l1747_174735

/-- Calculates the banker's discount given the face value and true discount of a bill -/
def bankers_discount (face_value : ℚ) (true_discount : ℚ) : ℚ :=
  let present_value := face_value - true_discount
  (true_discount * face_value) / present_value

/-- Theorem: Given a bill with face value 2460 and true discount 360, the banker's discount is 422 -/
theorem bankers_discount_example : bankers_discount 2460 360 = 422 := by
  sorry

#eval bankers_discount 2460 360

end NUMINAMATH_CALUDE_bankers_discount_example_l1747_174735


namespace NUMINAMATH_CALUDE_amy_school_year_hours_l1747_174778

/-- Calculates the required weekly hours for Amy's school year work --/
def school_year_weekly_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) : ℕ :=
  let hourly_wage := summer_earnings / (summer_weeks * summer_hours_per_week)
  let total_hours_needed := school_year_target / hourly_wage
  total_hours_needed / school_year_weeks

/-- Theorem stating that Amy needs to work 15 hours per week during the school year --/
theorem amy_school_year_hours : 
  school_year_weekly_hours 8 40 3200 32 4800 = 15 := by
  sorry

end NUMINAMATH_CALUDE_amy_school_year_hours_l1747_174778


namespace NUMINAMATH_CALUDE_problem_statement_l1747_174791

theorem problem_statement (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < 0) (h4 : 0 < c) :
  (a * b > a * c) ∧ (a * b > b * c) ∧ (a + c < b + c) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1747_174791


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1747_174760

def U : Finset ℕ := {1,2,3,4,5,6,7}
def A : Finset ℕ := {1,3,5}
def B : Finset ℕ := {2,5,7}

theorem complement_union_theorem : 
  (U \ (A ∪ B)) = {4,6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1747_174760


namespace NUMINAMATH_CALUDE_cargo_transport_possible_l1747_174707

/-- Represents the cargo transportation problem -/
structure CargoTransport where
  totalCargo : ℕ
  bagCapacity : ℕ
  truckCapacity : ℕ
  maxTrips : ℕ

/-- Checks if the cargo can be transported within the given number of trips -/
def canTransport (ct : CargoTransport) : Prop :=
  ∃ (trips : ℕ), trips ≤ ct.maxTrips ∧ trips * ct.truckCapacity ≥ ct.totalCargo

/-- Theorem stating that 36 tons of cargo can be transported in 11 trips or fewer -/
theorem cargo_transport_possible : 
  canTransport ⟨36, 1, 4, 11⟩ := by
  sorry

#check cargo_transport_possible

end NUMINAMATH_CALUDE_cargo_transport_possible_l1747_174707


namespace NUMINAMATH_CALUDE_brick_width_correct_l1747_174759

/-- The width of a brick used to build a wall with given dimensions -/
def brick_width : ℝ :=
  let wall_length : ℝ := 800  -- 8 m in cm
  let wall_height : ℝ := 600  -- 6 m in cm
  let wall_thickness : ℝ := 22.5
  let brick_count : ℕ := 1600
  let brick_length : ℝ := 100
  let brick_height : ℝ := 6
  11.25

/-- Theorem stating that the calculated brick width is correct -/
theorem brick_width_correct :
  let wall_length : ℝ := 800  -- 8 m in cm
  let wall_height : ℝ := 600  -- 6 m in cm
  let wall_thickness : ℝ := 22.5
  let brick_count : ℕ := 1600
  let brick_length : ℝ := 100
  let brick_height : ℝ := 6
  wall_length * wall_height * wall_thickness = 
    brick_count * (brick_length * brick_width * brick_height) :=
by
  sorry

#eval brick_width

end NUMINAMATH_CALUDE_brick_width_correct_l1747_174759


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l1747_174781

theorem beef_weight_before_processing (weight_after : ℝ) (percentage_lost : ℝ) 
  (h1 : weight_after = 546)
  (h2 : percentage_lost = 35) : 
  weight_after / (1 - percentage_lost / 100) = 840 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l1747_174781


namespace NUMINAMATH_CALUDE_assign_four_providers_from_twentyfive_l1747_174746

/-- The number of ways to assign different service providers to children -/
def assignProviders (totalProviders : ℕ) (children : ℕ) : ℕ :=
  (List.range children).foldl (fun acc i => acc * (totalProviders - i)) 1

/-- Theorem: Assigning 4 different service providers to 4 children from 25 providers -/
theorem assign_four_providers_from_twentyfive :
  assignProviders 25 4 = 303600 := by
  sorry

end NUMINAMATH_CALUDE_assign_four_providers_from_twentyfive_l1747_174746


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l1747_174737

theorem sequence_sum_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n ≥ 2, a n + 2 * S n * S (n - 1) = 0) →
  S 5 = 1 / 11 →
  a 1 = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l1747_174737
