import Mathlib

namespace negation_of_forall_positive_square_plus_one_l4118_411833

theorem negation_of_forall_positive_square_plus_one :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end negation_of_forall_positive_square_plus_one_l4118_411833


namespace exam_score_problem_l4118_411881

theorem exam_score_problem (mean : ℝ) (high_score : ℝ) (std_dev : ℝ) :
  mean = 74 ∧ high_score = 98 ∧ high_score = mean + 3 * std_dev →
  mean - 2 * std_dev = 58 := by
  sorry

end exam_score_problem_l4118_411881


namespace hyperbola_equation_l4118_411867

noncomputable section

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 3 * x

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -Real.sqrt 3

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 2 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define points A and B
def A : ℝ × ℝ := (-Real.sqrt 3, 2)
def B : ℝ × ℝ := (-Real.sqrt 3, -2)

-- Define the property of equilateral triangle
def is_equilateral_triangle (F A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2 ∧
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, hyperbola a b x y ↔ parabola x y) →
  (∀ x, directrix x → ∃ y, hyperbola a b x y) →
  (∀ x y, asymptote x y → hyperbola a b x y) →
  is_equilateral_triangle focus A B →
  (∀ x y, hyperbola a b x y ↔ x^2 - y^2 / 2 = 1) :=
sorry

end

end hyperbola_equation_l4118_411867


namespace only_negative_four_less_than_negative_three_l4118_411831

theorem only_negative_four_less_than_negative_three :
  let numbers : List ℝ := [-4, -2.8, 0, |-4|]
  ∀ x ∈ numbers, x < -3 ↔ x = -4 := by
  sorry

end only_negative_four_less_than_negative_three_l4118_411831


namespace distribute_cards_count_l4118_411817

/-- The number of ways to distribute 6 cards into 3 envelopes -/
def distribute_cards : ℕ :=
  let n_cards := 6
  let n_envelopes := 3
  let cards_per_envelope := 2
  let ways_to_place_1_and_2 := n_envelopes
  let remaining_cards := n_cards - cards_per_envelope
  let ways_to_distribute_remaining := 6  -- This is a given fact from the problem
  ways_to_place_1_and_2 * ways_to_distribute_remaining

/-- Theorem stating that the number of ways to distribute the cards is 18 -/
theorem distribute_cards_count : distribute_cards = 18 := by
  sorry

end distribute_cards_count_l4118_411817


namespace guppy_angelfish_ratio_l4118_411836

/-- Proves that the ratio of guppies to angelfish is 2:1 given the conditions -/
theorem guppy_angelfish_ratio :
  let goldfish : ℕ := 8
  let angelfish : ℕ := goldfish + 4
  let total_fish : ℕ := 44
  let guppies : ℕ := total_fish - (goldfish + angelfish)
  (guppies : ℚ) / angelfish = 2 / 1 := by
sorry

end guppy_angelfish_ratio_l4118_411836


namespace tangent_line_at_point_one_l4118_411859

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2*x + 2

theorem tangent_line_at_point_one :
  (f' 1 = 4) ∧
  (∀ x y : ℝ, y = f 1 → (4*x - y - 3 = 0 ↔ y - f 1 = f' 1 * (x - 1))) :=
sorry

end tangent_line_at_point_one_l4118_411859


namespace evans_books_in_eight_years_l4118_411812

def books_six_years_ago : ℕ := 500
def books_reduced : ℕ := 100
def books_given_away_fraction : ℚ := 1/2
def books_replaced_fraction : ℚ := 1/4
def books_increase_fraction : ℚ := 3/2
def books_gifted : ℕ := 30

theorem evans_books_in_eight_years :
  let current_books := books_six_years_ago - books_reduced
  let books_after_giving_away := (current_books : ℚ) * (1 - books_given_away_fraction)
  let books_after_replacing := books_after_giving_away + books_after_giving_away * books_replaced_fraction
  let final_books := books_after_replacing + books_after_replacing * books_increase_fraction + books_gifted
  final_books = 655 := by sorry

end evans_books_in_eight_years_l4118_411812


namespace max_correct_answers_l4118_411856

theorem max_correct_answers (total_questions : Nat) (correct_score : Int) (incorrect_score : Int)
  (john_score : Int) (min_attempted : Nat) :
  total_questions = 25 →
  correct_score = 4 →
  incorrect_score = -3 →
  john_score = 52 →
  min_attempted = 20 →
  ∃ (correct incorrect unanswered : Nat),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_score + incorrect * incorrect_score = john_score ∧
    correct + incorrect ≥ min_attempted ∧
    correct ≤ 17 ∧
    ∀ (c : Nat), c > 17 →
      ¬(∃ (i u : Nat), c + i + u = total_questions ∧
        c * correct_score + i * incorrect_score = john_score ∧
        c + i ≥ min_attempted) :=
by sorry

end max_correct_answers_l4118_411856


namespace cards_given_to_friends_l4118_411895

theorem cards_given_to_friends (initial_cards : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 13 → remaining_cards = 4 → initial_cards - remaining_cards = 9 := by
  sorry

end cards_given_to_friends_l4118_411895


namespace abs_leq_two_necessary_not_sufficient_l4118_411828

theorem abs_leq_two_necessary_not_sufficient :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x| ≤ 2) ∧
  (∃ x : ℝ, |x| ≤ 2 ∧ ¬(0 ≤ x ∧ x ≤ 2)) :=
by sorry

end abs_leq_two_necessary_not_sufficient_l4118_411828


namespace intersection_A_B_l4118_411835

-- Define set A
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 ≥ 4}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by
  sorry

end intersection_A_B_l4118_411835


namespace total_sides_l4118_411803

/-- The number of dice each person brought -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 6

/-- The number of people who brought dice -/
def num_people : ℕ := 2

/-- Theorem: The total number of sides on all dice is 48 -/
theorem total_sides : num_people * num_dice * sides_per_die = 48 := by
  sorry

end total_sides_l4118_411803


namespace production_scaling_l4118_411897

theorem production_scaling (x z : ℝ) (h : x > 0) :
  let production (n : ℝ) := n * n * n * (2 / n)
  production x = 2 * x^2 →
  production z = 2 * z^3 / x :=
by sorry

end production_scaling_l4118_411897


namespace crackers_distribution_l4118_411827

theorem crackers_distribution
  (initial_crackers : ℕ)
  (num_friends : ℕ)
  (remaining_crackers : ℕ)
  (h1 : initial_crackers = 15)
  (h2 : num_friends = 5)
  (h3 : remaining_crackers = 10)
  (h4 : num_friends > 0) :
  (initial_crackers - remaining_crackers) / num_friends = 1 :=
by sorry

end crackers_distribution_l4118_411827


namespace april_initial_roses_l4118_411876

/-- The number of roses April started with, given the price per rose, 
    the number of roses left, and the total earnings from selling roses. -/
def initial_roses (price_per_rose : ℕ) (roses_left : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings / price_per_rose) + roses_left

/-- Theorem stating that April started with 13 roses -/
theorem april_initial_roses : 
  initial_roses 4 4 36 = 13 := by
  sorry

end april_initial_roses_l4118_411876


namespace mouse_cheese_distance_l4118_411868

/-- The point where the mouse starts getting farther from the cheese -/
def mouse_turn_point : ℚ × ℚ := (-33/17, 285/17)

/-- The location of the cheese -/
def cheese_location : ℚ × ℚ := (9, 15)

/-- The equation of the line the mouse is running on: y = -4x + 9 -/
def mouse_path (x : ℚ) : ℚ := -4 * x + 9

theorem mouse_cheese_distance :
  let (a, b) := mouse_turn_point
  -- The point is on the mouse's path
  (mouse_path a = b) ∧
  -- The line from the cheese to the point is perpendicular to the mouse's path
  ((b - 15) / (a - 9) = 1 / 4) ∧
  -- The sum of the coordinates is 252/17
  (a + b = 252 / 17) := by sorry

end mouse_cheese_distance_l4118_411868


namespace candy_distribution_l4118_411854

/-- Candy distribution problem -/
theorem candy_distribution (tabitha stan julie carlos : ℕ) : 
  tabitha = 22 →
  stan = 13 →
  julie = tabitha / 2 →
  tabitha + stan + julie + carlos = 72 →
  carlos / stan = 2 :=
by
  sorry

end candy_distribution_l4118_411854


namespace lydias_flowering_plants_fraction_l4118_411832

theorem lydias_flowering_plants_fraction (total_plants : ℕ) 
  (flowering_percentage : ℚ) (flowers_per_plant : ℕ) (total_flowers_on_porch : ℕ) :
  total_plants = 80 →
  flowering_percentage = 2/5 →
  flowers_per_plant = 5 →
  total_flowers_on_porch = 40 →
  (total_flowers_on_porch / flowers_per_plant) / (flowering_percentage * total_plants) = 1/4 := by
sorry

end lydias_flowering_plants_fraction_l4118_411832


namespace sum_20_225_base7_l4118_411857

/-- Represents a number in base 7 --/
def Base7 : Type := ℕ

/-- Converts a natural number to its base 7 representation --/
def toBase7 (n : ℕ) : Base7 := sorry

/-- Adds two numbers in base 7 --/
def addBase7 (a b : Base7) : Base7 := sorry

/-- Theorem: The sum of 20₇ and 225₇ in base 7 is 245₇ --/
theorem sum_20_225_base7 :
  addBase7 (toBase7 20) (toBase7 225) = toBase7 245 := by sorry

end sum_20_225_base7_l4118_411857


namespace not_mapping_A_to_B_l4118_411891

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {y | 1 ≤ y ∧ y ≤ 4}

def f (x : ℝ) : ℝ := 4 - x^2

theorem not_mapping_A_to_B :
  ¬(∀ x ∈ A, f x ∈ B) :=
by sorry

end not_mapping_A_to_B_l4118_411891


namespace six_digit_divisibility_l4118_411890

theorem six_digit_divisibility (a b c : ℕ) 
  (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  ∃ k : ℤ, (a * 100000 + b * 10000 + c * 1000 + a * 100 + b * 10 + c : ℤ) = 1001 * k :=
sorry

end six_digit_divisibility_l4118_411890


namespace sqrt_product_plus_one_l4118_411889

theorem sqrt_product_plus_one : 
  Real.sqrt ((21:ℝ) * 20 * 19 * 18 + 1) = 379 := by
  sorry

end sqrt_product_plus_one_l4118_411889


namespace percent_increase_proof_l4118_411872

def original_lines : ℕ := 5600 - 1600
def increased_lines : ℕ := 5600
def line_increase : ℕ := 1600

theorem percent_increase_proof :
  (line_increase : ℝ) / (original_lines : ℝ) * 100 = 40 := by
  sorry

end percent_increase_proof_l4118_411872


namespace quadratic_distinct_roots_range_l4118_411837

theorem quadratic_distinct_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) ↔ 
  (a < 1 ∧ a ≠ 0) :=
sorry

end quadratic_distinct_roots_range_l4118_411837


namespace fraction_problem_l4118_411847

theorem fraction_problem (a b : ℤ) : 
  (a + 2 : ℚ) / b = 4 / 7 →
  (a : ℚ) / (b - 2) = 14 / 25 →
  ∃ (k : ℤ), k ≠ 0 ∧ k * a = 6 ∧ k * b = 11 :=
by sorry

end fraction_problem_l4118_411847


namespace only_exponential_has_multiplicative_property_l4118_411823

-- Define the property that we're looking for
def HasMultiplicativeProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (x + y) = f x * f y

-- Define the types of functions we're considering
class FunctionType (f : ℝ → ℝ) where
  isPower : Prop
  isLogarithmic : Prop
  isExponential : Prop
  isLinear : Prop

-- Theorem stating that only exponential functions have the multiplicative property
theorem only_exponential_has_multiplicative_property (f : ℝ → ℝ) [FunctionType f] :
  HasMultiplicativeProperty f ↔ FunctionType.isExponential f := by
  sorry

-- Note: The proof is omitted as per the instructions

end only_exponential_has_multiplicative_property_l4118_411823


namespace pairball_playing_time_l4118_411873

theorem pairball_playing_time (total_time : ℕ) (num_children : ℕ) (players_per_game : ℕ) :
  total_time = 90 ∧ 
  num_children = 6 ∧ 
  players_per_game = 2 →
  (total_time * players_per_game) / num_children = 30 := by
sorry

end pairball_playing_time_l4118_411873


namespace cistern_leak_time_l4118_411887

/-- Proves that if a cistern can be filled by pipe A in 16 hours and both pipes A and B together fill the cistern in 80.00000000000001 hours, then pipe B alone can leak out the full cistern in 80 hours. -/
theorem cistern_leak_time (fill_time_A : ℝ) (fill_time_both : ℝ) (leak_time_B : ℝ) : 
  fill_time_A = 16 →
  fill_time_both = 80.00000000000001 →
  (1 / fill_time_A) - (1 / leak_time_B) = 1 / fill_time_both →
  leak_time_B = 80 := by
  sorry

end cistern_leak_time_l4118_411887


namespace monochromatic_sequence_exists_l4118_411850

def S (n : ℕ) : ℕ := (n * (n^2 + 5)) / 6

def is_valid_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i < n - 1, a i < a (i + 1)) ∧
  (∀ i < n - 2, a (i + 1) - a i ≤ a (i + 2) - a (i + 1))

theorem monochromatic_sequence_exists (n : ℕ) (h : n ≥ 2) :
  ∀ c : ℕ → Bool,
  ∃ a : ℕ → ℕ, ∃ color : Bool,
    (∀ i < n, a i ≤ S n) ∧
    (∀ i < n, c (a i) = color) ∧
    is_valid_sequence a n :=
sorry

end monochromatic_sequence_exists_l4118_411850


namespace circle_diameter_l4118_411878

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 9 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 6 := by
  sorry

end circle_diameter_l4118_411878


namespace student_age_problem_l4118_411820

theorem student_age_problem (total_students : Nat) (avg_age : Nat) 
  (group1_count : Nat) (group1_avg : Nat)
  (group2_count : Nat) (group2_avg : Nat)
  (group3_count : Nat) (group3_avg : Nat) :
  total_students = 25 →
  avg_age = 16 →
  group1_count = 7 →
  group1_avg = 15 →
  group2_count = 12 →
  group2_avg = 16 →
  group3_count = 5 →
  group3_avg = 18 →
  group1_count + group2_count + group3_count = total_students - 1 →
  (total_students * avg_age) - (group1_count * group1_avg + group2_count * group2_avg + group3_count * group3_avg) = 13 := by
  sorry

end student_age_problem_l4118_411820


namespace smallest_integer_with_given_remainders_l4118_411839

theorem smallest_integer_with_given_remainders :
  let x : ℕ := 167
  (∀ y : ℕ, y > 0 →
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → y ≥ x) ∧
  (x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7) := by
  sorry

end smallest_integer_with_given_remainders_l4118_411839


namespace bottle_caps_per_box_l4118_411896

theorem bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℕ) 
  (h1 : total_caps = 316) (h2 : num_boxes = 79) : 
  total_caps / num_boxes = 4 := by
  sorry

end bottle_caps_per_box_l4118_411896


namespace total_candies_l4118_411807

/-- The total number of candies Linda and Chloe have together is 62, 
    given that Linda has 34 candies and Chloe has 28 candies. -/
theorem total_candies (linda_candies chloe_candies : ℕ) 
  (h1 : linda_candies = 34) 
  (h2 : chloe_candies = 28) : 
  linda_candies + chloe_candies = 62 := by
  sorry

end total_candies_l4118_411807


namespace residue_calculation_l4118_411870

theorem residue_calculation : (230 * 15 - 20 * 9 + 5) % 17 = 0 := by
  sorry

end residue_calculation_l4118_411870


namespace figurine_cost_l4118_411849

theorem figurine_cost (tv_count : ℕ) (tv_price : ℕ) (figurine_count : ℕ) (total_spent : ℕ) :
  tv_count = 5 →
  tv_price = 50 →
  figurine_count = 10 →
  total_spent = 260 →
  (total_spent - tv_count * tv_price) / figurine_count = 1 :=
by sorry

end figurine_cost_l4118_411849


namespace original_number_proof_l4118_411809

theorem original_number_proof (x : ℝ) (h : 1 + 1/x = 9/4) : x = 4/5 := by
  sorry

end original_number_proof_l4118_411809


namespace min_days_to_plant_100_trees_l4118_411886

def trees_planted (n : ℕ) : ℕ := 2 * (2^n - 1)

theorem min_days_to_plant_100_trees :
  (∃ n : ℕ, trees_planted n ≥ 100) ∧
  (∀ n : ℕ, trees_planted n ≥ 100 → n ≥ 6) ∧
  trees_planted 6 ≥ 100 :=
sorry

end min_days_to_plant_100_trees_l4118_411886


namespace larger_number_is_84_l4118_411816

theorem larger_number_is_84 (a b : ℕ+) (h1 : Nat.gcd a b = 84) (h2 : Nat.lcm a b = 21) (h3 : b = 4 * a) :
  b = 84 := by
  sorry

end larger_number_is_84_l4118_411816


namespace no_solutions_for_equation_l4118_411801

theorem no_solutions_for_equation : 
  ¬∃ (a b : ℕ+), 
    a ≥ b ∧ 
    a * b + 125 = 30 * Nat.lcm a b + 24 * Nat.gcd a b + a % b :=
by sorry

end no_solutions_for_equation_l4118_411801


namespace line_equation_from_intercepts_l4118_411815

theorem line_equation_from_intercepts (x y : ℝ) :
  (x = -2 ∧ y = 0) ∨ (x = 0 ∧ y = 3) → 3 * x - 2 * y + 6 = 0 := by
  sorry

end line_equation_from_intercepts_l4118_411815


namespace prop_logic_l4118_411860

theorem prop_logic (p q : Prop) (h : ¬(¬p ∨ ¬q)) : (p ∧ q) ∧ (p ∨ q) := by
  sorry

end prop_logic_l4118_411860


namespace geometric_sequence_expression_zero_l4118_411863

/-- For a geometric sequence, the product of terms equidistant from the ends is constant -/
def geometric_sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a 1 * a n = a 2 * a (n - 1)

/-- The expression (a₁aₙ)² - a₂a₄aₙ₋₁aₙ₋₃ equals zero for any geometric sequence -/
theorem geometric_sequence_expression_zero (a : ℕ → ℝ) (n : ℕ) 
  (h : geometric_sequence_property a) : 
  (a 1 * a n)^2 - (a 2 * a 4 * a (n-1) * a (n-3)) = 0 := by
  sorry

end geometric_sequence_expression_zero_l4118_411863


namespace books_sum_l4118_411888

/-- The number of books Sam has -/
def sam_books : ℕ := 110

/-- The number of books Joan has -/
def joan_books : ℕ := 102

/-- The total number of books Sam and Joan have together -/
def total_books : ℕ := sam_books + joan_books

theorem books_sum :
  total_books = 212 :=
by sorry

end books_sum_l4118_411888


namespace school_classrooms_l4118_411842

theorem school_classrooms (total_students : ℕ) (desks_type1 : ℕ) (desks_type2 : ℕ) :
  total_students = 400 →
  desks_type1 = 30 →
  desks_type2 = 25 →
  ∃ (num_classrooms : ℕ),
    num_classrooms > 0 ∧
    (num_classrooms / 3) * desks_type1 + (2 * num_classrooms / 3) * desks_type2 = total_students ∧
    num_classrooms = 15 := by
  sorry

end school_classrooms_l4118_411842


namespace system_solution_proof_l4118_411810

theorem system_solution_proof :
  let x₁ : ℝ := 4
  let x₂ : ℝ := 3
  let x₃ : ℝ := 5
  (x₁ + 2 * x₂ = 10) ∧
  (3 * x₁ + 2 * x₂ + x₃ = 23) ∧
  (x₂ + 2 * x₃ = 13) := by
  sorry

end system_solution_proof_l4118_411810


namespace system_solution_l4118_411893

theorem system_solution (x y : ℝ) : 
  x * y * (x^2 + y^2) = 78 ∧ x^4 + y^4 = 97 ↔ 
  ((x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = -3 ∧ y = -2) ∨ (x = -2 ∧ y = -3)) :=
by sorry

end system_solution_l4118_411893


namespace simplify_expression_l4118_411884

theorem simplify_expression (x y : ℝ) : 7*x + 3*y + 4 - 2*x + 9 + 5*y = 5*x + 8*y + 13 := by
  sorry

end simplify_expression_l4118_411884


namespace water_in_mixture_l4118_411869

theorem water_in_mixture (water_parts syrup_parts total_volume : ℚ) 
  (h1 : water_parts = 5)
  (h2 : syrup_parts = 2)
  (h3 : total_volume = 3) : 
  (water_parts * total_volume) / (water_parts + syrup_parts) = 15 / 7 := by
  sorry

end water_in_mixture_l4118_411869


namespace first_discount_percentage_l4118_411825

/-- Proves that the first discount percentage is 10% for an article with a given price and two successive discounts -/
theorem first_discount_percentage
  (normal_price : ℝ)
  (first_discount : ℝ)
  (second_discount : ℝ)
  (h1 : normal_price = 174.99999999999997)
  (h2 : first_discount = 0.1)
  (h3 : second_discount = 0.2)
  : first_discount = 0.1 := by
  sorry

end first_discount_percentage_l4118_411825


namespace factorial_divisibility_power_of_two_l4118_411883

theorem factorial_divisibility_power_of_two (n : ℕ) : 
  (∃ k : ℕ, n = 2^k) ↔ (n.factorial % 2^(n-1) = 0) := by
  sorry

end factorial_divisibility_power_of_two_l4118_411883


namespace stu_book_count_l4118_411834

theorem stu_book_count (stu_books : ℕ) (albert_books : ℕ) : 
  albert_books = 4 * stu_books →
  stu_books + albert_books = 45 →
  stu_books = 9 := by
sorry

end stu_book_count_l4118_411834


namespace marks_interest_earned_l4118_411806

/-- Calculates the interest earned on an investment with annual compound interest -/
def interestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- The interest earned on Mark's investment -/
theorem marks_interest_earned :
  let principal : ℝ := 1500
  let rate : ℝ := 0.02
  let years : ℕ := 8
  abs (interestEarned principal rate years - 257.49) < 0.01 := by
  sorry

end marks_interest_earned_l4118_411806


namespace history_book_cost_l4118_411846

/-- Given the conditions of a book purchase, this theorem proves the cost of each history book. -/
theorem history_book_cost (total_books : ℕ) (math_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 90 →
  math_book_cost = 4 →
  total_price = 397 →
  math_books = 53 →
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 :=
by sorry

end history_book_cost_l4118_411846


namespace total_legs_on_farm_l4118_411829

/-- The number of legs for each animal type -/
def duck_legs : ℕ := 2
def dog_legs : ℕ := 4

/-- The farm composition -/
def total_animals : ℕ := 11
def num_ducks : ℕ := 6
def num_dogs : ℕ := total_animals - num_ducks

/-- The theorem to prove -/
theorem total_legs_on_farm : 
  num_ducks * duck_legs + num_dogs * dog_legs = 32 := by sorry

end total_legs_on_farm_l4118_411829


namespace johns_final_push_time_l4118_411899

/-- The time of John's final push in a race, given specific conditions --/
theorem johns_final_push_time (john_initial_lag : ℝ) (john_speed : ℝ) (steve_speed : ℝ) (john_final_lead : ℝ)
  (h1 : john_initial_lag = 15)
  (h2 : john_speed = 4.2)
  (h3 : steve_speed = 3.7)
  (h4 : john_final_lead = 2) :
  (john_initial_lag + john_final_lead) / john_speed = 17 / 4.2 := by
  sorry

end johns_final_push_time_l4118_411899


namespace inverse_function_parameter_l4118_411819

/-- Given a function f and its inverse, find the value of b -/
theorem inverse_function_parameter (f : ℝ → ℝ) (b : ℝ) : 
  (∀ x, f x = 1 / (2 * x + b)) →
  (∀ x, f⁻¹ x = (2 - 3 * x) / (3 * x)) →
  b = -2 := by
sorry

end inverse_function_parameter_l4118_411819


namespace arithmetic_sequence_sum_l4118_411877

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 5 + a 8 = 15 → a 3 + a 7 = 10 := by
  sorry

end arithmetic_sequence_sum_l4118_411877


namespace identity_function_proof_l4118_411802

theorem identity_function_proof (f : ℝ → ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 0 1) →
  (∀ x ∈ Set.Icc 0 1, f (2 * x - f x) = x) →
  (∀ x ∈ Set.Icc 0 1, f x = x) := by
sorry

end identity_function_proof_l4118_411802


namespace addington_average_temperature_l4118_411814

/-- The average of the daily low temperatures in Addington from September 15th, 2008 through September 19th, 2008, inclusive, is 42.4 degrees Fahrenheit. -/
theorem addington_average_temperature : 
  let temperatures : List ℝ := [40, 47, 45, 41, 39]
  (temperatures.sum / temperatures.length : ℝ) = 42.4 := by
  sorry

end addington_average_temperature_l4118_411814


namespace kitchen_tiles_l4118_411811

theorem kitchen_tiles (kitchen_length kitchen_width tile_area : ℝ) 
  (h1 : kitchen_length = 52)
  (h2 : kitchen_width = 79)
  (h3 : tile_area = 7.5) : 
  ⌈(kitchen_length * kitchen_width) / tile_area⌉ = 548 := by
  sorry

end kitchen_tiles_l4118_411811


namespace pitcher_juice_distribution_l4118_411841

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) : 
  let total_juice := (2/3) * C
  let cups := 6
  let juice_per_cup := total_juice / cups
  (juice_per_cup / C) * 100 = 100/9 := by sorry

end pitcher_juice_distribution_l4118_411841


namespace perpendicular_to_parallel_is_perpendicular_l4118_411821

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_parallel_is_perpendicular
  (α β : Plane) (m n : Line)
  (h1 : α ≠ β)
  (h2 : m ≠ n)
  (h3 : perpendicular_line_plane m β)
  (h4 : parallel_line_plane n β) :
  perpendicular_lines m n :=
sorry

end perpendicular_to_parallel_is_perpendicular_l4118_411821


namespace x_range_l4118_411830

theorem x_range (x : ℝ) (h : ∀ a > 0, x^2 < 1 + a) : -1 ≤ x ∧ x ≤ 1 := by
  sorry

end x_range_l4118_411830


namespace cafeteria_red_apples_l4118_411882

theorem cafeteria_red_apples :
  ∀ (red_apples green_apples students_wanting_fruit extra_apples : ℕ),
    green_apples = 15 →
    students_wanting_fruit = 5 →
    extra_apples = 16 →
    red_apples + green_apples = students_wanting_fruit + extra_apples →
    red_apples = 6 :=
by
  sorry

end cafeteria_red_apples_l4118_411882


namespace sam_exchange_probability_l4118_411880

/-- Represents the vending machine and Sam's purchasing scenario -/
structure VendingMachine where
  num_toys : Nat
  toy_prices : List Rat
  favorite_toy_price : Rat
  sam_quarters : Nat
  sam_bill : Nat

/-- Calculates the probability of Sam needing to exchange his bill -/
def probability_need_exchange (vm : VendingMachine) : Rat :=
  1 - (Nat.factorial 7 : Rat) / (Nat.factorial vm.num_toys : Rat)

/-- Theorem stating the probability of Sam needing to exchange his bill -/
theorem sam_exchange_probability (vm : VendingMachine) :
  vm.num_toys = 10 ∧
  vm.toy_prices = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5] ∧
  vm.favorite_toy_price = 4 ∧
  vm.sam_quarters = 12 ∧
  vm.sam_bill = 20 →
  probability_need_exchange vm = 719 / 720 := by
  sorry

#eval probability_need_exchange {
  num_toys := 10,
  toy_prices := [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
  favorite_toy_price := 4,
  sam_quarters := 12,
  sam_bill := 20
}

end sam_exchange_probability_l4118_411880


namespace circle_properties_l4118_411865

/-- Given a circle with circumference 36 cm, prove its radius, diameter, and area -/
theorem circle_properties (C : ℝ) (h : C = 36) :
  ∃ (r d A : ℝ),
    r = 18 / Real.pi ∧
    d = 36 / Real.pi ∧
    A = 324 / Real.pi ∧
    C = 2 * Real.pi * r ∧
    d = 2 * r ∧
    A = Real.pi * r^2 := by
  sorry

end circle_properties_l4118_411865


namespace sum_of_coordinates_l4118_411805

/-- Given a point A with coordinates (m, n) that are (-3, 2) with respect to the origin, 
    prove that m + n = -1 -/
theorem sum_of_coordinates (m n : ℝ) (h : (m, n) = (-3, 2)) : m + n = -1 := by
  sorry

end sum_of_coordinates_l4118_411805


namespace medium_size_can_be_rational_l4118_411826

-- Define the popcorn sizes
structure PopcornSize where
  name : String
  amount : Nat
  price : Nat

-- Define the customer's preferences
structure CustomerPreferences where
  budget : Nat
  wantsDrink : Bool
  preferBalancedMeal : Bool

-- Define the utility function
def utility (choice : PopcornSize) (prefs : CustomerPreferences) : Nat :=
  sorry

-- Define the theorem
theorem medium_size_can_be_rational (small medium large : PopcornSize) 
  (prefs : CustomerPreferences) : 
  small.name = "small" → 
  small.amount = 50 → 
  small.price = 200 →
  medium.name = "medium" → 
  medium.amount = 70 → 
  medium.price = 400 →
  large.name = "large" → 
  large.amount = 130 → 
  large.price = 500 →
  prefs.budget = 500 →
  prefs.wantsDrink = true →
  prefs.preferBalancedMeal = true →
  ∃ (drink_price : Nat), 
    utility medium prefs + utility (PopcornSize.mk "drink" 0 drink_price) prefs ≥ 
    max (utility small prefs) (utility large prefs) :=
  sorry


end medium_size_can_be_rational_l4118_411826


namespace tshirt_socks_price_difference_l4118_411874

/-- The price difference between a t-shirt and socks -/
theorem tshirt_socks_price_difference 
  (jeans_price t_shirt_price socks_price : ℝ) 
  (h1 : jeans_price = 2 * t_shirt_price) 
  (h2 : jeans_price = 30) 
  (h3 : socks_price = 5) : 
  t_shirt_price - socks_price = 10 := by
sorry

end tshirt_socks_price_difference_l4118_411874


namespace game_cost_calculation_l4118_411822

theorem game_cost_calculation (initial_amount : ℕ) (spent_amount : ℕ) (num_games : ℕ) :
  initial_amount = 42 →
  spent_amount = 10 →
  num_games = 4 →
  num_games > 0 →
  ∃ (game_cost : ℕ), game_cost * num_games = initial_amount - spent_amount ∧ game_cost = 8 :=
by sorry

end game_cost_calculation_l4118_411822


namespace tangent_line_condition_l4118_411838

theorem tangent_line_condition (a : ℝ) : 
  (∃ (x : ℝ), a * x + 1 - a = x^2 ∧ ∀ (y : ℝ), y ≠ x → a * y + 1 - a ≠ y^2) ↔ |a| = 2 :=
sorry

end tangent_line_condition_l4118_411838


namespace half_of_a_l4118_411853

theorem half_of_a (a : ℝ) : (1 / 2) * a = a / 2 := by
  sorry

end half_of_a_l4118_411853


namespace original_number_proof_l4118_411885

theorem original_number_proof (x : ℝ) : 3 * (2 * x + 9) = 57 → x = 5 := by
  sorry

end original_number_proof_l4118_411885


namespace valid_n_set_l4118_411855

theorem valid_n_set (n : ℕ) : (∃ a b : ℤ, n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n ∈ ({0, 1, 2} : Set ℕ) := by
  sorry

end valid_n_set_l4118_411855


namespace square_of_binomial_l4118_411848

theorem square_of_binomial (m n : ℝ) : (3*m - n)^2 = 9*m^2 - 6*m*n + n^2 := by
  sorry

end square_of_binomial_l4118_411848


namespace abs_neg_six_l4118_411866

theorem abs_neg_six : |(-6 : ℤ)| = 6 := by sorry

end abs_neg_six_l4118_411866


namespace odd_function_and_inequality_l4118_411840

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + a * x^2) + 2 * x)

theorem odd_function_and_inequality (a m : ℝ) : 
  (∀ x, f a x = -f a (-x)) ∧ 
  (∀ x, f a (2 * m - m * Real.sin x) + f a (Real.cos x)^2 ≥ 0) →
  a = 4 ∧ m ≥ 0 := by sorry

end odd_function_and_inequality_l4118_411840


namespace population_growth_prediction_l4118_411800

/-- Theorem: Population Growth and Prediction --/
theorem population_growth_prediction
  (initial_population : ℝ)
  (current_population : ℝ)
  (future_population : ℝ)
  (h1 : current_population = 3 * initial_population)
  (h2 : future_population = 1.4 * current_population)
  (h3 : future_population = 16800)
  : initial_population = 4000 := by
  sorry

end population_growth_prediction_l4118_411800


namespace square_sum_given_diff_and_product_l4118_411898

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 5) 
  (h2 : x * y = 4) : 
  x^2 + y^2 = 33 := by
sorry

end square_sum_given_diff_and_product_l4118_411898


namespace highest_y_coordinate_zero_is_highest_y_l4118_411813

theorem highest_y_coordinate (x y : ℝ) : 
  (x - 4)^2 / 25 + y^2 / 49 = 0 → y ≤ 0 :=
by sorry

theorem zero_is_highest_y (x y : ℝ) : 
  (x - 4)^2 / 25 + y^2 / 49 = 0 → ∃ (x₀ y₀ : ℝ), (x₀ - 4)^2 / 25 + y₀^2 / 49 = 0 ∧ y₀ = 0 :=
by sorry

end highest_y_coordinate_zero_is_highest_y_l4118_411813


namespace negative_one_less_than_abs_neg_two_fifths_l4118_411871

theorem negative_one_less_than_abs_neg_two_fifths : -1 < |-2/5| := by
  sorry

end negative_one_less_than_abs_neg_two_fifths_l4118_411871


namespace estimate_larger_than_actual_l4118_411861

theorem estimate_larger_than_actual (x y z : ℝ) 
  (hxy : x > y) (hy : y > 0) (hz : z > 0) : 
  (x + z) - (y - z) > x - y := by
  sorry

end estimate_larger_than_actual_l4118_411861


namespace prob_same_length_hexagon_l4118_411808

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of elements in T -/
def total_elements : ℕ := num_sides + num_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := 17 / 35

theorem prob_same_length_hexagon :
  (num_sides * (num_sides - 1) + num_diagonals * (num_diagonals - 1)) /
  (total_elements * (total_elements - 1)) = prob_same_length :=
sorry

end prob_same_length_hexagon_l4118_411808


namespace unique_number_not_in_range_l4118_411843

theorem unique_number_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, x ≠ -d/c → f x = (a*x + b)/(c*x + d))
  (h11 : f 11 = 11)
  (h41 : f 41 = 41)
  (hinv : ∀ x, x ≠ -d/c → f (f x) = x) :
  ∃! y, ∀ x, f x ≠ y ∧ y = a/12 :=
sorry

end unique_number_not_in_range_l4118_411843


namespace min_value_of_x_l4118_411851

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 3 + (1/3) * Real.log x) : x ≥ 3 * Real.sqrt 3 := by
  sorry

end min_value_of_x_l4118_411851


namespace parabola_point_distance_l4118_411818

/-- A point on a parabola with a specific distance to the focus has a specific distance to the y-axis -/
theorem parabola_point_distance (P : ℝ × ℝ) : 
  (P.2)^2 = 8 * P.1 →  -- P is on the parabola y^2 = 8x
  ((P.1 - 2)^2 + P.2^2)^(1/2 : ℝ) = 6 →  -- Distance from P to focus (2, 0) is 6
  P.1 = 4 :=  -- Distance from P to y-axis is 4
by sorry

end parabola_point_distance_l4118_411818


namespace max_value_of_a_l4118_411844

theorem max_value_of_a (a : ℝ) : 
  (∀ x > 1, a - x + Real.log (x * (x + 1)) ≤ 0) →
  a ≤ (1 + Real.sqrt 3) / 2 - Real.log ((3 / 2) + Real.sqrt 3) :=
by sorry

end max_value_of_a_l4118_411844


namespace ball_ratio_l4118_411875

theorem ball_ratio (total : ℕ) (blue red : ℕ) (green : ℕ := 3 * blue) 
  (yellow : ℕ := total - (blue + red + green)) 
  (h1 : total = 36) (h2 : blue = 6) (h3 : red = 4) :
  yellow / red = 2 :=
by sorry

end ball_ratio_l4118_411875


namespace cosine_identity_proof_l4118_411892

theorem cosine_identity_proof : 2 * (Real.cos (15 * π / 180))^2 - Real.cos (30 * π / 180) = 1 := by
  sorry

end cosine_identity_proof_l4118_411892


namespace distance_to_line_is_sqrt_17_l4118_411894

/-- The distance from a point to a line in 3D space --/
def distance_point_to_line (point : ℝ × ℝ × ℝ) (line_point1 line_point2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating that the distance from (2, 0, -1) to the line passing through (1, 3, 1) and (3, -1, 5) is √17 --/
theorem distance_to_line_is_sqrt_17 :
  distance_point_to_line (2, 0, -1) (1, 3, 1) (3, -1, 5) = Real.sqrt 17 := by
  sorry

end distance_to_line_is_sqrt_17_l4118_411894


namespace problem_solution_l4118_411804

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = (1 + Real.sqrt 33) / 2 := by
sorry

end problem_solution_l4118_411804


namespace factor_x_squared_minus_64_l4118_411879

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end factor_x_squared_minus_64_l4118_411879


namespace bug_return_probability_l4118_411864

/-- Probability of returning to the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | 1 => 0
  | (n + 2) => (1 / 3) * (1 - Q (n + 1)) + (1 / 3) * Q n

/-- The probability of returning to the starting vertex on the tenth move is 34817/59049 -/
theorem bug_return_probability :
  Q 10 = 34817 / 59049 := by
  sorry

end bug_return_probability_l4118_411864


namespace calculation_proof_l4118_411862

theorem calculation_proof : 1525 + 140 / 70 - 225 = 1302 := by
  sorry

end calculation_proof_l4118_411862


namespace sum_of_special_primes_is_prime_l4118_411852

theorem sum_of_special_primes_is_prime (P Q : ℕ+) (h1 : Nat.Prime P)
  (h2 : Nat.Prime Q) (h3 : Nat.Prime (P - Q)) (h4 : Nat.Prime (P + Q)) :
  Nat.Prime (P + Q + (P - Q) + Q) :=
sorry

end sum_of_special_primes_is_prime_l4118_411852


namespace number_exceeding_fraction_l4118_411858

theorem number_exceeding_fraction : ∃ x : ℚ, x = (3/8) * x + 15 ∧ x = 24 := by
  sorry

end number_exceeding_fraction_l4118_411858


namespace limit_x_to_x_as_x_approaches_zero_l4118_411824

theorem limit_x_to_x_as_x_approaches_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |x^x - 1| < ε :=
sorry

end limit_x_to_x_as_x_approaches_zero_l4118_411824


namespace base_conversion_l4118_411845

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem base_conversion :
  (binary_to_decimal [true, true, false, true, false, true]) = 43 ∧
  (decimal_to_base7 85) = [1, 5, 1] :=
sorry

end base_conversion_l4118_411845
