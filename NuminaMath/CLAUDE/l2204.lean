import Mathlib

namespace steven_fruit_difference_l2204_220487

/-- The number of apples Steven has -/
def steven_apples : ℕ := 19

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 15

/-- The difference between Steven's apples and peaches -/
def apple_peach_difference : ℕ := steven_apples - steven_peaches

theorem steven_fruit_difference : apple_peach_difference = 4 := by
  sorry

end steven_fruit_difference_l2204_220487


namespace x_power_four_plus_reciprocal_l2204_220466

theorem x_power_four_plus_reciprocal (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x^2) = 2 → x^4 + (1/x^4) = 2 := by
sorry

end x_power_four_plus_reciprocal_l2204_220466


namespace A_on_axes_l2204_220479

def A (a : ℝ) : ℝ × ℝ := (a - 3, a^2 - 4)

theorem A_on_axes :
  (∀ a : ℝ, (A a).2 = 0 → (A a = (-1, 0) ∨ A a = (-5, 0))) ∧
  (∀ a : ℝ, (A a).1 = 0 → A a = (0, 5)) := by
  sorry

end A_on_axes_l2204_220479


namespace mini_quiz_multiple_choice_count_l2204_220422

/-- The number of ways to answer 3 true-false questions where all answers cannot be the same -/
def truefalse_combinations : ℕ := 6

/-- The number of answer choices for each multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- The total number of ways to write the answer key -/
def total_combinations : ℕ := 96

/-- Proves that the number of multiple-choice questions is 2 -/
theorem mini_quiz_multiple_choice_count :
  ∃ (n : ℕ), truefalse_combinations * multiple_choice_options ^ n = total_combinations ∧ n = 2 := by
sorry

end mini_quiz_multiple_choice_count_l2204_220422


namespace intersection_of_A_and_B_l2204_220438

def A : Set ℤ := {1, 2}
def B : Set ℤ := {-1, 1, 4}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end intersection_of_A_and_B_l2204_220438


namespace inequality_proof_l2204_220471

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 := by
  sorry

end inequality_proof_l2204_220471


namespace fathers_with_full_time_jobs_l2204_220414

theorem fathers_with_full_time_jobs 
  (total_parents : ℝ) 
  (mothers_ratio : ℝ) 
  (mothers_full_time_ratio : ℝ) 
  (no_full_time_ratio : ℝ) 
  (h1 : mothers_ratio = 0.6) 
  (h2 : mothers_full_time_ratio = 5/6) 
  (h3 : no_full_time_ratio = 0.2) : 
  (total_parents * (1 - mothers_ratio) * 3/4) = 
  (total_parents * (1 - no_full_time_ratio) - total_parents * mothers_ratio * mothers_full_time_ratio) := by
sorry

end fathers_with_full_time_jobs_l2204_220414


namespace bank_withdrawal_total_l2204_220486

theorem bank_withdrawal_total (x y : ℕ) : 
  x / 20 + y / 20 = 30 → x + y = 600 := by
  sorry

end bank_withdrawal_total_l2204_220486


namespace remainder_of_9876543210_div_101_l2204_220441

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 68 := by
  sorry

end remainder_of_9876543210_div_101_l2204_220441


namespace basketball_tryouts_l2204_220411

theorem basketball_tryouts (girls boys called_back : ℕ) 
  (h1 : girls = 42)
  (h2 : boys = 80)
  (h3 : called_back = 25) :
  girls + boys - called_back = 97 := by
sorry

end basketball_tryouts_l2204_220411


namespace biased_coin_flip_l2204_220406

theorem biased_coin_flip (h : ℝ) : 
  0 < h → h < 1 →
  (4 : ℝ) * h * (1 - h)^3 = 6 * h^2 * (1 - h)^2 →
  (6 : ℝ) * (2/5)^2 * (3/5)^2 = 216/625 :=
by sorry

end biased_coin_flip_l2204_220406


namespace prob_same_school_adjacent_l2204_220463

/-- The number of students from the first school -/
def students_school1 : ℕ := 2

/-- The number of students from the second school -/
def students_school2 : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := students_school1 + students_school2

/-- The probability that students from the same school will be standing next to each other -/
def probability_same_school_adjacent : ℚ := 4/5

/-- Theorem stating that the probability of students from the same school standing next to each other is 4/5 -/
theorem prob_same_school_adjacent :
  probability_same_school_adjacent = 4/5 := by sorry

end prob_same_school_adjacent_l2204_220463


namespace ellipse_hyperbola_conditions_l2204_220481

/-- Represents the equation (x^2)/(2m) - (y^2)/(m-6) = 1 as an ellipse with foci on the y-axis -/
def proposition_p (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧
  (∀ x y : ℝ, x^2 / (2*m) - y^2 / (m-6) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)

/-- Represents the equation (x^2)/(m+1) + (y^2)/(m-1) = 1 as a hyperbola -/
def proposition_q (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / (m+1) + y^2 / (m-1) = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1)

/-- Theorem stating the conditions for proposition_p and proposition_q -/
theorem ellipse_hyperbola_conditions (m : ℝ) :
  (proposition_p m ↔ 0 < m ∧ m < 2) ∧
  (¬proposition_q m ↔ m ≤ -1 ∨ m ≥ 1) :=
sorry

end ellipse_hyperbola_conditions_l2204_220481


namespace necessary_but_not_sufficient_l2204_220467

-- Define the sets M and P
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def P : Set ℝ := {x : ℝ | x ≤ -1}

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x ∈ M ∩ P → x ∈ M ∪ P) ∧
  ¬(∀ x : ℝ, x ∈ M ∪ P → x ∈ M ∩ P) :=
sorry

end necessary_but_not_sufficient_l2204_220467


namespace units_digit_of_7_power_2023_l2204_220495

theorem units_digit_of_7_power_2023 : 7^2023 % 10 = 3 := by
  sorry

end units_digit_of_7_power_2023_l2204_220495


namespace probability_red_or_blue_specific_l2204_220493

/-- Represents the probability of drawing a red or blue marble after a previous draw -/
def probability_red_or_blue (red blue yellow : ℕ) : ℚ :=
  let total := red + blue + yellow
  let p_yellow := yellow / total
  let p_not_yellow := 1 - p_yellow
  let p_red_or_blue_after_yellow := (red + blue) / (total - 1)
  let p_red_or_blue_after_not_yellow := (red + blue) / total
  p_yellow * p_red_or_blue_after_yellow + p_not_yellow * p_red_or_blue_after_not_yellow

/-- Theorem stating the probability of drawing a red or blue marble
    after a previous draw from a bag with 4 red, 3 blue, and 6 yellow marbles -/
theorem probability_red_or_blue_specific :
  probability_red_or_blue 4 3 6 = 91 / 169 := by
  sorry

end probability_red_or_blue_specific_l2204_220493


namespace kitten_weight_l2204_220447

/-- Given the weights of a kitten and two dogs satisfying certain conditions,
    prove that the kitten weighs 6 pounds. -/
theorem kitten_weight (x y z : ℝ) 
  (h1 : x + y + z = 36)
  (h2 : x + z = 2*y)
  (h3 : x + y = z) :
  x = 6 := by
  sorry

end kitten_weight_l2204_220447


namespace ashley_wedding_champagne_servings_l2204_220457

/-- The number of servings in one bottle of champagne for Ashley's wedding toast. -/
def servings_per_bottle (guests : ℕ) (glasses_per_guest : ℕ) (total_bottles : ℕ) : ℕ :=
  (guests * glasses_per_guest) / total_bottles

/-- Theorem stating that there are 6 servings in one bottle of champagne for Ashley's wedding toast. -/
theorem ashley_wedding_champagne_servings :
  servings_per_bottle 120 2 40 = 6 := by
  sorry

end ashley_wedding_champagne_servings_l2204_220457


namespace book_cost_l2204_220419

/-- If two identical books cost $36 in total, then eight of these books will cost $144. -/
theorem book_cost (two_books_cost : ℕ) (h : two_books_cost = 36) : 
  (8 * (two_books_cost / 2) = 144) :=
sorry

end book_cost_l2204_220419


namespace abc_product_l2204_220423

theorem abc_product (a b c : ℝ) 
  (h1 : a - b = 4)
  (h2 : a^2 + b^2 = 18)
  (h3 : a + b + c = 8) :
  a * b * c = 92 - 50 * Real.sqrt 5 := by
  sorry

end abc_product_l2204_220423


namespace remainder_492381_div_6_l2204_220434

theorem remainder_492381_div_6 : 492381 % 6 = 3 := by
  sorry

end remainder_492381_div_6_l2204_220434


namespace cupcakes_left_l2204_220492

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := (5 * dozen) / 2

/-- The initial number of people in the class -/
def initial_class_size : ℕ := 27 + 1 + 1

/-- The number of students absent -/
def absent_students : ℕ := 3

/-- The actual number of people present in the class -/
def class_size : ℕ := initial_class_size - absent_students

theorem cupcakes_left : cupcakes_brought - class_size = 4 := by
  sorry

end cupcakes_left_l2204_220492


namespace rectangular_field_width_l2204_220491

/-- The width of a rectangular field, given its length and a relationship between length and width. -/
def field_width (length : ℝ) (length_width_relation : ℝ → ℝ → Prop) : ℝ :=
  13.5

/-- Theorem stating that the width of a rectangular field is 13.5 meters, given specific conditions. -/
theorem rectangular_field_width :
  let length := 24
  let length_width_relation := λ l w => l = 2 * w - 3
  field_width length length_width_relation = 13.5 := by
  sorry

end rectangular_field_width_l2204_220491


namespace number_times_one_fourth_squared_l2204_220432

theorem number_times_one_fourth_squared (x : ℝ) : x * (1/4)^2 = 4^3 → x = 1024 := by
  sorry

end number_times_one_fourth_squared_l2204_220432


namespace roses_to_sister_l2204_220418

-- Define the initial number of roses
def initial_roses : ℕ := 20

-- Define the number of roses given to mother
def roses_to_mother : ℕ := 6

-- Define the number of roses given to grandmother
def roses_to_grandmother : ℕ := 9

-- Define the number of roses Ian kept for himself
def roses_kept : ℕ := 1

-- Theorem to prove
theorem roses_to_sister : 
  initial_roses - (roses_to_mother + roses_to_grandmother + roses_kept) = 4 := by
  sorry

end roses_to_sister_l2204_220418


namespace binomial_distribution_unique_parameters_l2204_220451

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- Variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_unique_parameters
  (X : BinomialDistribution)
  (h_expectation : expectation X = 8)
  (h_variance : variance X = 1.6) :
  X.n = 10 ∧ X.p = 0.8 := by
  sorry

end binomial_distribution_unique_parameters_l2204_220451


namespace distance_after_eight_hours_l2204_220427

/-- The distance between two trains after a given time -/
def distance_between_trains (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed2 - speed1) * time

/-- Theorem: The distance between two trains after 8 hours -/
theorem distance_after_eight_hours :
  distance_between_trains 11 31 8 = 160 := by
  sorry

end distance_after_eight_hours_l2204_220427


namespace glove_ratio_for_43_participants_l2204_220477

/-- The ratio of the minimum number of gloves needed to the number of participants -/
def glove_ratio (participants : ℕ) : ℚ :=
  2

theorem glove_ratio_for_43_participants :
  glove_ratio 43 = 2 := by
  sorry

end glove_ratio_for_43_participants_l2204_220477


namespace janet_video_game_lives_l2204_220468

theorem janet_video_game_lives : ∀ initial_lives : ℕ,
  initial_lives - 23 + 46 = 70 → initial_lives = 47 :=
by
  sorry

end janet_video_game_lives_l2204_220468


namespace polar_to_cartesian_conversion_l2204_220480

theorem polar_to_cartesian_conversion :
  let r : ℝ := 2
  let θ : ℝ := 2 * Real.pi / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -1) ∧ (y = Real.sqrt 3) := by
  sorry

end polar_to_cartesian_conversion_l2204_220480


namespace sqrt_simplification_l2204_220499

theorem sqrt_simplification : Real.sqrt 32 + Real.sqrt 8 - Real.sqrt 50 = Real.sqrt 2 := by
  sorry

end sqrt_simplification_l2204_220499


namespace tax_rate_as_percent_l2204_220494

/-- Given a tax rate of $82 per $100.00, prove that the tax rate expressed as a percent is 82%. -/
theorem tax_rate_as_percent (tax_amount : ℝ) (base_amount : ℝ) :
  tax_amount = 82 ∧ base_amount = 100 →
  (tax_amount / base_amount) * 100 = 82 := by
sorry

end tax_rate_as_percent_l2204_220494


namespace divisibility_by_twelve_l2204_220404

theorem divisibility_by_twelve (n : ℕ) : 
  (713 * 10 + n ≥ 1000) ∧ 
  (713 * 10 + n < 10000) ∧ 
  (713 * 10 + n) % 12 = 0 ↔ 
  n = 4 := by
sorry

end divisibility_by_twelve_l2204_220404


namespace ellipse_equation_slope_product_sum_of_squares_l2204_220478

/-- An ellipse with eccentricity √2/2 and foci on the unit circle -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (a^2 - b^2) / a^2 = 1/2
  h4 : a^2 - b^2 = 1

/-- A point on the ellipse -/
structure PointOnEllipse (e : SpecialEllipse) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation (e : SpecialEllipse) :
  ∀ (p : PointOnEllipse e), p.x^2 / 2 + p.y^2 = 1 := by sorry

theorem slope_product (e : SpecialEllipse) (p q : PointOnEllipse e) 
  (hp : p.x ≠ 0) (hq : q.x ≠ 0) :
  (p.y / p.x) * (q.y / q.x) = -1/2 := by sorry

theorem sum_of_squares (e : SpecialEllipse) (p q : PointOnEllipse e) :
  p.x^2 + p.y^2 + q.x^2 + q.y^2 = 3 := by sorry

end ellipse_equation_slope_product_sum_of_squares_l2204_220478


namespace number_wall_top_value_l2204_220421

/-- Represents a number wall pyramid --/
structure NumberWall :=
  (bottom_left : ℕ)
  (bottom_middle : ℕ)
  (bottom_right : ℕ)

/-- Calculates the value at the top of the number wall pyramid --/
def top_value (wall : NumberWall) : ℕ :=
  let m := wall.bottom_left + wall.bottom_middle
  let n := wall.bottom_middle + wall.bottom_right
  let left_mid := wall.bottom_left + m
  let right_mid := m + n
  let left_top := left_mid + right_mid
  let right_top := right_mid + wall.bottom_right
  2 * (left_top + right_top)

/-- Theorem stating that the top value of the given number wall is 320 --/
theorem number_wall_top_value :
  ∃ (wall : NumberWall), wall.bottom_left = 20 ∧ wall.bottom_middle = 34 ∧ wall.bottom_right = 44 ∧ top_value wall = 320 :=
sorry

end number_wall_top_value_l2204_220421


namespace unique_solution_l2204_220474

/-- The product of all digits of a positive integer -/
def digit_product (n : ℕ+) : ℕ := sorry

/-- Theorem stating that 12 is the only positive integer solution -/
theorem unique_solution : 
  ∃! (x : ℕ+), digit_product x = x^2 - 10*x - 22 :=
by
  sorry

end unique_solution_l2204_220474


namespace probability_no_adjacent_seating_l2204_220475

def num_chairs : ℕ := 9
def num_people : ℕ := 4

def total_arrangements (n m : ℕ) : ℕ :=
  (n - 1) * (n - 2) * (n - 3)

def favorable_arrangements (n m : ℕ) : ℕ :=
  (n - m + 1) * m

theorem probability_no_adjacent_seating :
  (favorable_arrangements num_chairs num_people : ℚ) / 
  (total_arrangements num_chairs num_people : ℚ) = 1 / 14 := by
  sorry

end probability_no_adjacent_seating_l2204_220475


namespace exam_score_calculation_l2204_220476

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℕ) :
  total_questions = 60 →
  correct_answers = 36 →
  total_marks = 120 →
  ∃ (score_per_correct : ℕ),
    score_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧
    score_per_correct = 4 :=
by sorry

end exam_score_calculation_l2204_220476


namespace three_integers_divisibility_l2204_220439

theorem three_integers_divisibility (x y z : ℕ+) :
  (x ∣ y + z) ∧ (y ∣ x + z) ∧ (z ∣ x + y) →
  (∃ a : ℕ+, (x = a ∧ y = a ∧ z = a) ∨
             (x = a ∧ y = a ∧ z = 2 * a) ∨
             (x = a ∧ y = 2 * a ∧ z = 3 * a)) :=
by sorry

end three_integers_divisibility_l2204_220439


namespace sequence_characterization_l2204_220462

/-- An infinite sequence of positive integers -/
def Sequence := ℕ → ℕ

/-- The property that the sequence is strictly increasing -/
def StrictlyIncreasing (a : Sequence) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The property that no three terms in the sequence sum to another term -/
def NoThreeSum (a : Sequence) : Prop :=
  ∀ i j k : ℕ, a i + a j ≠ a k

/-- The property that infinitely many terms of the sequence are of the form 2k - 1 -/
def InfinitelyManyOdd (a : Sequence) : Prop :=
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ a k = 2 * k - 1

/-- The main theorem: any sequence satisfying the given properties must be aₙ = 2n - 1 -/
theorem sequence_characterization (a : Sequence)
  (h1 : StrictlyIncreasing a)
  (h2 : NoThreeSum a)
  (h3 : InfinitelyManyOdd a) :
  ∀ n : ℕ, a n = 2 * n - 1 :=
by sorry

end sequence_characterization_l2204_220462


namespace max_triangle_side_length_l2204_220407

theorem max_triangle_side_length (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Three different side lengths
  a + b + c = 24 →         -- Perimeter is 24
  a < b + c ∧ b < a + c ∧ c < a + b →  -- Triangle inequality
  a ≤ 11 ∧ b ≤ 11 ∧ c ≤ 11 :=
by sorry

end max_triangle_side_length_l2204_220407


namespace unique_set_l2204_220450

def is_valid_set (A : Finset ℤ) : Prop :=
  A.card = 4 ∧
  ∀ (subset : Finset ℤ), subset ⊆ A → subset.card = 3 →
    (subset.sum id) ∈ ({-1, 5, 3, 8} : Finset ℤ)

theorem unique_set :
  ∃! (A : Finset ℤ), is_valid_set A ∧ A = {-3, 0, 2, 6} :=
sorry

end unique_set_l2204_220450


namespace root_of_fifth_unity_l2204_220409

theorem root_of_fifth_unity {p q r s t m : ℂ} (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + t = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + t * m + p = 0) :
  m^5 = 1 :=
by sorry

end root_of_fifth_unity_l2204_220409


namespace geometric_sequence_common_ratio_l2204_220413

theorem geometric_sequence_common_ratio
  (x : ℝ)
  (h : ∃ r : ℝ, (x + Real.log 2 / Real.log 27) * r = x + Real.log 2 / Real.log 9 ∧
                (x + Real.log 2 / Real.log 9) * r = x + Real.log 2 / Real.log 3) :
  ∃ r : ℝ, r = 3 ∧
    (x + Real.log 2 / Real.log 27) * r = x + Real.log 2 / Real.log 9 ∧
    (x + Real.log 2 / Real.log 9) * r = x + Real.log 2 / Real.log 3 :=
by
  sorry

end geometric_sequence_common_ratio_l2204_220413


namespace range_of_f_l2204_220429

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -1 ≤ y ∧ y ≤ 3} := by
  sorry

end range_of_f_l2204_220429


namespace bridge_length_specific_bridge_length_l2204_220426

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the specific bridge length problem -/
theorem specific_bridge_length : 
  bridge_length 200 60 25 = 216.75 := by
  sorry

end bridge_length_specific_bridge_length_l2204_220426


namespace intersection_in_first_quadrant_l2204_220489

theorem intersection_in_first_quadrant (k : ℝ) : 
  (∃ x y : ℝ, 
    y = k * x + 2 * k + 1 ∧ 
    y = -1/2 * x + 2 ∧ 
    x > 0 ∧ 
    y > 0) ↔ 
  -1/6 < k ∧ k < 1/2 :=
sorry

end intersection_in_first_quadrant_l2204_220489


namespace line_passes_through_point_l2204_220448

/-- Given that the midpoint of (k, 0) and (b, 0) is (-1, 0),
    prove that the line y = kx + b passes through (1, -2) -/
theorem line_passes_through_point
  (k b : ℝ) -- k and b are real numbers
  (h : (k + b) / 2 = -1) -- midpoint condition
  : k * 1 + b = -2 := by -- line passes through (1, -2)
  sorry

end line_passes_through_point_l2204_220448


namespace height_on_hypotenuse_of_right_triangle_l2204_220458

theorem height_on_hypotenuse_of_right_triangle (a b h c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → (1/2) * a * b = (1/2) * c * h → h = 4.8 := by
  sorry

end height_on_hypotenuse_of_right_triangle_l2204_220458


namespace exponential_function_property_l2204_220485

theorem exponential_function_property (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  ∀ (x₁ x₂ : ℝ), (fun x ↦ a^x) (x₁ + x₂) = (fun x ↦ a^x) x₁ * (fun x ↦ a^x) x₂ := by
  sorry

end exponential_function_property_l2204_220485


namespace linear_system_solution_l2204_220416

/-- Given a system of linear equations and a condition on its solution, prove the value of k. -/
theorem linear_system_solution (x y k : ℝ) : 
  3 * x + 2 * y = k + 1 →
  2 * x + 3 * y = k →
  x + y = 3 →
  k = 7 := by sorry

end linear_system_solution_l2204_220416


namespace quadratic_equation_roots_l2204_220412

theorem quadratic_equation_roots (c : ℝ) :
  (2 + Real.sqrt 3 : ℝ) ^ 2 - 4 * (2 + Real.sqrt 3) + c = 0 →
  (2 - Real.sqrt 3 : ℝ) ^ 2 - 4 * (2 - Real.sqrt 3) + c = 0 ∧ c = 1 := by
  sorry

end quadratic_equation_roots_l2204_220412


namespace tangent_product_simplification_l2204_220440

theorem tangent_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end tangent_product_simplification_l2204_220440


namespace tournament_prize_orderings_l2204_220454

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := 5

/-- Represents the number of possible outcomes for each game -/
def outcomes_per_game : ℕ := 2

/-- Theorem stating the number of possible prize orderings in the tournament -/
theorem tournament_prize_orderings :
  (outcomes_per_game ^ num_games : ℕ) = 32 := by sorry

end tournament_prize_orderings_l2204_220454


namespace range_of_b_for_two_intersection_points_l2204_220444

/-- The range of b for which there are exactly two points P satisfying the given conditions -/
theorem range_of_b_for_two_intersection_points (b : ℝ) : 
  (∃! (P₁ P₂ : ℝ × ℝ), 
    P₁ ≠ P₂ ∧ 
    (P₁.1 + Real.sqrt 3 * P₁.2 = b) ∧ 
    (P₂.1 + Real.sqrt 3 * P₂.2 = b) ∧ 
    ((P₁.1 - 4)^2 + P₁.2^2 = 4 * (P₁.1^2 + P₁.2^2)) ∧
    ((P₂.1 - 4)^2 + P₂.2^2 = 4 * (P₂.1^2 + P₂.2^2))) ↔ 
  (-20/3 < b ∧ b < 4) :=
sorry

end range_of_b_for_two_intersection_points_l2204_220444


namespace mans_speed_against_current_l2204_220461

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speedAgainstCurrent (speedWithCurrent : ℝ) (currentSpeed : ℝ) : ℝ :=
  speedWithCurrent - 2 * currentSpeed

/-- Theorem stating that given the specific speeds mentioned in the problem, 
    the man's speed against the current is 10 km/hr. -/
theorem mans_speed_against_current :
  speedAgainstCurrent 15 2.5 = 10 := by
  sorry

#eval speedAgainstCurrent 15 2.5

end mans_speed_against_current_l2204_220461


namespace sin_225_degrees_l2204_220465

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_225_degrees_l2204_220465


namespace fuel_station_problem_l2204_220436

/-- Represents the problem of determining the number of mini-vans filled up at a fuel station. -/
theorem fuel_station_problem (service_cost truck_count mini_van_tank truck_tank_ratio total_cost fuel_cost : ℚ) :
  service_cost = 210/100 →
  fuel_cost = 70/100 →
  truck_count = 2 →
  mini_van_tank = 65 →
  truck_tank_ratio = 220/100 →
  total_cost = 3472/10 →
  ∃ (mini_van_count : ℚ),
    mini_van_count = 3 ∧
    mini_van_count * (service_cost + mini_van_tank * fuel_cost) +
    truck_count * (service_cost + (mini_van_tank * truck_tank_ratio) * fuel_cost) = total_cost :=
by sorry

end fuel_station_problem_l2204_220436


namespace max_beads_in_pile_l2204_220405

/-- Represents a pile of beads -/
structure BeadPile :=
  (size : ℕ)
  (has_lighter_bead : Bool)

/-- Represents a balance scale measurement -/
inductive Measurement
  | Balanced
  | Unbalanced

/-- A function that performs a measurement on a subset of beads -/
def perform_measurement (subset_size : ℕ) : Measurement :=
  sorry

/-- A function that represents the algorithm to find the lighter bead -/
def find_lighter_bead (pile : BeadPile) (max_measurements : ℕ) : Bool :=
  sorry

/-- Theorem stating the maximum number of beads in the pile -/
theorem max_beads_in_pile :
  ∀ (pile : BeadPile),
    pile.has_lighter_bead →
    (∃ (algorithm : BeadPile → ℕ → Bool),
      (∀ p, algorithm p 2 = find_lighter_bead p 2) →
      algorithm pile 2 = true) →
    pile.size ≤ 7 :=
sorry

end max_beads_in_pile_l2204_220405


namespace fraction_subtraction_complex_fraction_division_l2204_220428

-- Define a and b as real numbers
variable (a b : ℝ)

-- Assumption that a ≠ b
variable (h : a ≠ b)

-- First theorem
theorem fraction_subtraction : (b / (a - b)) - (a / (a - b)) = -1 := by sorry

-- Second theorem
theorem complex_fraction_division : 
  ((a^2 - a*b) / a^2) / ((a / b) - (b / a)) = b / (a + b) := by sorry

end fraction_subtraction_complex_fraction_division_l2204_220428


namespace highest_frequency_count_l2204_220431

theorem highest_frequency_count (total_sample : ℕ) (num_groups : ℕ) 
  (cumulative_freq_seven : ℚ) (a : ℕ) (r : ℕ) : 
  total_sample = 100 →
  num_groups = 10 →
  cumulative_freq_seven = 79/100 →
  r > 1 →
  a + a * r + a * r^2 = total_sample - (cumulative_freq_seven * total_sample).num →
  (∃ (max_freq : ℕ), max_freq = max a (max (a * r) (a * r^2)) ∧ max_freq = 12) :=
sorry

end highest_frequency_count_l2204_220431


namespace power_of_seven_mod_twelve_l2204_220408

theorem power_of_seven_mod_twelve : 7^145 % 12 = 7 := by
  sorry

end power_of_seven_mod_twelve_l2204_220408


namespace relationship_fg_l2204_220459

noncomputable def f (x : ℝ) := Real.exp x + x - 2
noncomputable def g (x : ℝ) := Real.log x + x^2 - 3

theorem relationship_fg (a b : ℝ) (h1 : f a = 0) (h2 : g b = 0) :
  g a < 0 ∧ 0 < f b := by sorry

end relationship_fg_l2204_220459


namespace cyclic_sum_inequality_l2204_220470

theorem cyclic_sum_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_3 : x + y + z = 3) :
  x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < 3 + x*y + y*z + z*x := by
  sorry

end cyclic_sum_inequality_l2204_220470


namespace g_composition_of_three_l2204_220403

-- Define the function g
def g (x : ℝ) : ℝ := 7 * x - 3

-- State the theorem
theorem g_composition_of_three : g (g (g 3)) = 858 := by
  sorry

end g_composition_of_three_l2204_220403


namespace equation_transformation_correct_l2204_220435

theorem equation_transformation_correct (x : ℝ) :
  (x + 1) / 2 - 1 = (2 * x - 1) / 3 ↔ 3 * (x + 1) - 6 = 2 * (2 * x - 1) := by
  sorry

end equation_transformation_correct_l2204_220435


namespace triangle_equation_solution_l2204_220452

theorem triangle_equation_solution (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a < b + c) :
  let p := (a + b + c) / 2
  let x := a * b * c / (2 * Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  b * Real.sqrt (x^2 - c^2) + c * Real.sqrt (x^2 - b^2) = a * x := by
sorry

end triangle_equation_solution_l2204_220452


namespace max_value_2x_minus_y_l2204_220456

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0) 
  (h2 : y + 1 ≥ 0) 
  (h3 : x + y + 1 ≤ 0) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (x' y' : ℝ), 
    x' - y' + 1 ≥ 0 → y' + 1 ≥ 0 → x' + y' + 1 ≤ 0 → 2*x' - y' ≤ m :=
by sorry

end max_value_2x_minus_y_l2204_220456


namespace face_value_of_shares_l2204_220443

/-- Calculates the face value of shares given investment details -/
theorem face_value_of_shares
  (investment : ℝ)
  (quoted_price : ℝ)
  (dividend_rate : ℝ)
  (annual_income : ℝ)
  (h1 : investment = 4940)
  (h2 : quoted_price = 9.5)
  (h3 : dividend_rate = 0.14)
  (h4 : annual_income = 728)
  : ∃ (face_value : ℝ),
    face_value = 10 ∧
    annual_income = (investment / quoted_price) * (dividend_rate * face_value) :=
by sorry

end face_value_of_shares_l2204_220443


namespace melissas_fabric_l2204_220437

/-- The amount of fabric Melissa has given her work hours and dress requirements -/
theorem melissas_fabric (fabric_per_dress : ℝ) (hours_per_dress : ℝ) (total_work_hours : ℝ) :
  fabric_per_dress = 4 →
  hours_per_dress = 3 →
  total_work_hours = 42 →
  (total_work_hours / hours_per_dress) * fabric_per_dress = 56 := by
  sorry

end melissas_fabric_l2204_220437


namespace power_division_l2204_220498

theorem power_division (n : ℕ) : (16^3018) / 8 = 2^9032 := by
  sorry

end power_division_l2204_220498


namespace dot_product_range_l2204_220496

/-- Given a fixed point M(0, 4) and a point P(x, y) on the circle x^2 + y^2 = 4,
    the dot product of MP⃗ and OP⃗ is bounded between -4 and 12. -/
theorem dot_product_range (x y : ℝ) : 
  x^2 + y^2 = 4 → 
  -4 ≤ x * x + y * y - 4 * y ∧ x * x + y * y - 4 * y ≤ 12 := by
  sorry

end dot_product_range_l2204_220496


namespace orange_cost_calculation_l2204_220449

/-- Given the cost of 5 dozen oranges, calculate the cost of 8 dozen oranges at the same rate -/
theorem orange_cost_calculation (cost_five_dozen : ℝ) : cost_five_dozen = 42 →
  (8 : ℝ) * (cost_five_dozen / 5) = 67.2 := by
  sorry

end orange_cost_calculation_l2204_220449


namespace decimal_sum_to_fraction_l2204_220473

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 + 0.000012 = 3858 / 15625 := by
  sorry

end decimal_sum_to_fraction_l2204_220473


namespace expression_evaluation_l2204_220415

theorem expression_evaluation : 
  (121 * (1/13 - 1/17) + 169 * (1/17 - 1/11) + 289 * (1/11 - 1/13)) / 
  (11 * (1/13 - 1/17) + 13 * (1/17 - 1/11) + 17 * (1/11 - 1/13)) = 41 := by
  sorry

end expression_evaluation_l2204_220415


namespace min_cuts_for_3inch_to_1inch_cube_l2204_220483

/-- Represents a three-dimensional cube -/
structure Cube where
  side_length : ℕ

/-- Represents a cut on a cube -/
inductive Cut
  | plane : Cut

/-- The minimum number of cuts required to divide a cube into smaller cubes -/
def min_cuts (original : Cube) (target : Cube) : ℕ := sorry

/-- The number of smaller cubes that can be created from a larger cube -/
def num_smaller_cubes (original : Cube) (target : Cube) : ℕ := 
  (original.side_length / target.side_length) ^ 3

theorem min_cuts_for_3inch_to_1inch_cube : 
  let original := Cube.mk 3
  let target := Cube.mk 1
  min_cuts original target = 6 ∧ 
  num_smaller_cubes original target = 27 := by sorry

end min_cuts_for_3inch_to_1inch_cube_l2204_220483


namespace octal_245_equals_decimal_165_l2204_220442

/-- Converts an octal number to decimal --/
def octal_to_decimal (a b c : ℕ) : ℕ := c * 8^2 + b * 8^1 + a * 8^0

/-- Proves that 245 in octal is equal to 165 in decimal --/
theorem octal_245_equals_decimal_165 : octal_to_decimal 5 4 2 = 165 := by
  sorry

end octal_245_equals_decimal_165_l2204_220442


namespace grocer_coffee_stock_l2204_220488

theorem grocer_coffee_stock (initial_stock : ℝ) : 
  initial_stock > 0 →
  0.30 * initial_stock + 0.60 * 100 = 0.36 * (initial_stock + 100) →
  initial_stock = 400 := by
sorry

end grocer_coffee_stock_l2204_220488


namespace smallest_four_digit_divisible_by_53_l2204_220433

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end smallest_four_digit_divisible_by_53_l2204_220433


namespace polygon_contains_center_l2204_220484

/-- A convex polygon type -/
structure ConvexPolygon where
  area : ℝ
  isConvex : Bool

/-- A circle type -/
structure Circle where
  radius : ℝ

/-- Predicate to check if a polygon is inside a circle -/
def isInside (p : ConvexPolygon) (c : Circle) : Prop :=
  sorry

/-- Predicate to check if a polygon contains the center of a circle -/
def containsCenter (p : ConvexPolygon) (c : Circle) : Prop :=
  sorry

/-- Theorem statement -/
theorem polygon_contains_center (p : ConvexPolygon) (c : Circle) :
  p.area = 7 ∧ p.isConvex = true ∧ c.radius = 2 ∧ isInside p c → containsCenter p c :=
sorry

end polygon_contains_center_l2204_220484


namespace jerrys_average_increase_l2204_220453

theorem jerrys_average_increase (initial_average : ℝ) (fourth_test_score : ℝ) : 
  initial_average = 85 → fourth_test_score = 93 → 
  (4 * (initial_average + 2) = 3 * initial_average + fourth_test_score) := by
sorry

end jerrys_average_increase_l2204_220453


namespace consecutive_integers_product_plus_one_is_square_l2204_220410

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  ∃ m : ℤ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m ^ 2 := by
  sorry

end consecutive_integers_product_plus_one_is_square_l2204_220410


namespace polynomial_coefficients_sum_l2204_220445

theorem polynomial_coefficients_sum 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h : ∀ x, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) : 
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1) ∧ (a₀ + a₂ + a₄ + a₆ = 365) := by
  sorry

end polynomial_coefficients_sum_l2204_220445


namespace gcd_8675309_7654321_l2204_220455

theorem gcd_8675309_7654321 : Nat.gcd 8675309 7654321 = 36 := by
  sorry

end gcd_8675309_7654321_l2204_220455


namespace typing_time_calculation_l2204_220420

theorem typing_time_calculation (original_speed : ℕ) (speed_reduction : ℕ) (document_length : ℕ) :
  original_speed = 212 →
  speed_reduction = 40 →
  document_length = 3440 →
  (document_length : ℚ) / ((original_speed - speed_reduction) : ℚ) = 20 := by
  sorry

end typing_time_calculation_l2204_220420


namespace range_of_a_l2204_220402

-- Define propositions P and Q
def P (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (P a ∨ Q a) ∧ ¬(P a ∧ Q a) →
  ((-1 < a ∧ a ≤ 1) ∨ a ≥ 2) :=
sorry

end range_of_a_l2204_220402


namespace unique_integer_solution_l2204_220460

theorem unique_integer_solution :
  ∃! (x y z : ℤ), x^2 + y^2 + z^2 = x^2 * y^2 :=
by sorry

end unique_integer_solution_l2204_220460


namespace group_selection_problem_l2204_220482

theorem group_selection_problem (n : ℕ) (k : ℕ) : n = 30 ∧ k = 3 → Nat.choose n k = 4060 := by
  sorry

end group_selection_problem_l2204_220482


namespace sugar_spilled_correct_l2204_220401

/-- The amount of sugar Pamela spilled on the floor -/
def sugar_spilled (original : ℝ) (left : ℝ) : ℝ := original - left

/-- Theorem stating that the amount of sugar spilled is correct -/
theorem sugar_spilled_correct (original left : ℝ) 
  (h1 : original = 9.8)
  (h2 : left = 4.6) : 
  sugar_spilled original left = 5.2 := by
  sorry

end sugar_spilled_correct_l2204_220401


namespace rational_equation_solution_l2204_220490

theorem rational_equation_solution :
  ∃ x : ℝ, x ≠ 2 ∧ x ≠ (4/5 : ℝ) ∧
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 20*x - 40)/(5*x - 4) = -5 ∧
  x = -3 := by
  sorry

end rational_equation_solution_l2204_220490


namespace triangle_longest_side_l2204_220400

theorem triangle_longest_side (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (ratio : a / 5 = b / 6 ∧ b / 6 = c / 7)
  (perimeter : a + b + c = 720) :
  c = 280 := by
sorry

end triangle_longest_side_l2204_220400


namespace square_sum_implies_fourth_power_sum_l2204_220472

theorem square_sum_implies_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 7) : 
  r^4 + 1/r^4 = 23 := by
sorry

end square_sum_implies_fourth_power_sum_l2204_220472


namespace reflected_ray_passes_through_C_l2204_220430

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the reflected ray equation
def reflected_ray_equation (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Theorem statement
theorem reflected_ray_passes_through_C : 
  ∃ C : ℝ × ℝ, C.1 = 1 ∧ C.2 = 4 ∧ reflected_ray_equation C.1 C.2 := by sorry

end reflected_ray_passes_through_C_l2204_220430


namespace prime_sum_theorem_l2204_220425

theorem prime_sum_theorem (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p + q = r → 1 < p → p < q → p = 2 := by
sorry

end prime_sum_theorem_l2204_220425


namespace inequality_proof_l2204_220424

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a + b + c) / 3 - (a * b * c) ^ (1/3) ≤ max ((a^(1/2) - b^(1/2))^2) (max ((b^(1/2) - c^(1/2))^2) ((c^(1/2) - a^(1/2))^2)) :=
by sorry

end inequality_proof_l2204_220424


namespace fraction_addition_l2204_220469

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end fraction_addition_l2204_220469


namespace largest_n_for_trig_inequality_l2204_220446

theorem largest_n_for_trig_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≤ Real.sqrt n / 2) ∧
  (∀ (m : ℕ), m > n → ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m > Real.sqrt m / 2) ∧
  n = 8 := by
  sorry

end largest_n_for_trig_inequality_l2204_220446


namespace projection_v_onto_w_l2204_220417

def v : Fin 2 → ℝ := ![3, -1]
def w : Fin 2 → ℝ := ![4, 2]

theorem projection_v_onto_w :
  (((v • w) / (w • w)) • w) = ![2, 1] := by sorry

end projection_v_onto_w_l2204_220417


namespace trophy_cost_l2204_220497

def total_cost (a b : ℕ) : ℚ := (a * 1000 + 999 + b) / 10

theorem trophy_cost (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : (a * 1000 + 999 + b) % 8 = 0) 
  (h4 : (a + 9 + 9 + 9 + b) % 9 = 0) : 
  (total_cost a b) / 72 = 11.11 := by
  sorry

end trophy_cost_l2204_220497


namespace fraction_equality_l2204_220464

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : a / b = (2 * a) / (2 * b) := by
  sorry

end fraction_equality_l2204_220464
