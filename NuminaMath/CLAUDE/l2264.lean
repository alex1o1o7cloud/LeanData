import Mathlib

namespace triangle_square_ratio_l2264_226489

/-- The ratio of the combined area to the combined perimeter of an equilateral triangle and a square -/
theorem triangle_square_ratio : 
  let triangle_side : ℝ := 10
  let triangle_altitude : ℝ := triangle_side * (Real.sqrt 3 / 2)
  let square_side : ℝ := triangle_altitude / 2
  let triangle_area : ℝ := (1 / 2) * triangle_side * triangle_altitude
  let square_area : ℝ := square_side ^ 2
  let combined_area : ℝ := triangle_area + square_area
  let triangle_perimeter : ℝ := 3 * triangle_side
  let square_perimeter : ℝ := 4 * square_side
  let combined_perimeter : ℝ := triangle_perimeter + square_perimeter
  combined_area / combined_perimeter = (25 * Real.sqrt 3 + 18.75) / (30 + 10 * Real.sqrt 3) :=
by
  sorry


end triangle_square_ratio_l2264_226489


namespace super_rare_snake_price_multiplier_l2264_226407

/-- Represents the snake selling scenario --/
structure SnakeScenario where
  num_snakes : Nat
  eggs_per_snake : Nat
  regular_price : Nat
  total_revenue : Nat

/-- Calculates the price multiplier of the super rare snake --/
def super_rare_price_multiplier (scenario : SnakeScenario) : Nat :=
  let total_eggs := scenario.num_snakes * scenario.eggs_per_snake
  let num_regular_snakes := total_eggs - 1
  let regular_revenue := num_regular_snakes * scenario.regular_price
  let super_rare_price := scenario.total_revenue - regular_revenue
  super_rare_price / scenario.regular_price

/-- Theorem stating the price multiplier of the super rare snake --/
theorem super_rare_snake_price_multiplier :
  let scenario := SnakeScenario.mk 3 2 250 2250
  super_rare_price_multiplier scenario = 4 := by
  sorry

end super_rare_snake_price_multiplier_l2264_226407


namespace chloe_total_score_l2264_226434

/-- Calculates the total score for a two-level game with treasures and level completion bonuses. -/
def totalScore (
  level1Points : ℕ
  ) (level1Bonus : ℕ
  ) (level2Points : ℕ
  ) (level2Bonus : ℕ
  ) (level1Treasures : ℕ
  ) (level2Treasures : ℕ
  ) : ℕ :=
  level1Points * level1Treasures + level1Bonus +
  level2Points * level2Treasures + level2Bonus

theorem chloe_total_score :
  totalScore 9 15 11 20 6 3 = 122 := by
  sorry

end chloe_total_score_l2264_226434


namespace polygon_sides_l2264_226488

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : sum_interior_angles = 1260 → (n - 2) * 180 = sum_interior_angles → n = 9 := by
  sorry

end polygon_sides_l2264_226488


namespace f_properties_l2264_226433

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x + 1

theorem f_properties (a b c : ℝ) 
  (h : ∀ x, a * f x + b * f (x - c) = 1) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 3) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 3) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 2) ∧
  (b * Real.cos c / a = -1) := by
  sorry

end f_properties_l2264_226433


namespace harry_routine_duration_is_90_l2264_226426

/-- Harry's morning routine duration --/
def harry_routine_duration : ℕ :=
  let coffee_bagel_time := 15
  let dog_walking_time := 20
  let exercise_time := 25
  let reading_eating_time := 2 * coffee_bagel_time
  coffee_bagel_time + dog_walking_time + exercise_time + reading_eating_time

/-- Theorem stating that Harry's morning routine takes 90 minutes --/
theorem harry_routine_duration_is_90 : harry_routine_duration = 90 := by
  sorry

end harry_routine_duration_is_90_l2264_226426


namespace janice_purchase_problem_l2264_226430

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents50 : ℕ
  dollar150 : ℕ
  dollar250 : ℕ
  dollar350 : ℕ

/-- The problem statement -/
theorem janice_purchase_problem (items : ItemCounts) : 
  (items.cents50 + items.dollar150 + items.dollar250 + items.dollar350 = 50) →
  (50 * items.cents50 + 150 * items.dollar150 + 250 * items.dollar250 + 350 * items.dollar350 = 10000) →
  items.cents50 = 5 := by
  sorry

#check janice_purchase_problem

end janice_purchase_problem_l2264_226430


namespace neither_sufficient_nor_necessary_l2264_226409

/-- Predicate for the equation x^2 - mx + n = 0 having two positive roots -/
def has_two_positive_roots (m n : ℝ) : Prop :=
  m^2 - 4*n ≥ 0 ∧ m > 0 ∧ n > 0

/-- Predicate for the curve mx^2 + ny^2 = 1 being an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0

theorem neither_sufficient_nor_necessary :
  ¬(∀ m n : ℝ, has_two_positive_roots m n → is_ellipse m n) ∧
  ¬(∀ m n : ℝ, is_ellipse m n → has_two_positive_roots m n) :=
sorry

end neither_sufficient_nor_necessary_l2264_226409


namespace max_areas_theorem_l2264_226452

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk :=
  (n : ℕ)  -- number of pairs of radii

/-- The maximum number of non-overlapping areas in a divided disk -/
def max_areas (disk : DividedDisk) : ℕ :=
  4 * disk.n + 4

/-- Theorem stating the maximum number of non-overlapping areas -/
theorem max_areas_theorem (disk : DividedDisk) :
  max_areas disk = 4 * disk.n + 4 :=
sorry

end max_areas_theorem_l2264_226452


namespace abs_eq_sqrt_square_l2264_226491

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end abs_eq_sqrt_square_l2264_226491


namespace davids_english_marks_l2264_226439

/-- Calculates the marks in English given the marks in other subjects and the average -/
def marks_in_english (math physics chemistry biology : ℕ) (average : ℚ) : ℚ :=
  5 * average - (math + physics + chemistry + biology)

/-- Proves that David's marks in English are 72 -/
theorem davids_english_marks :
  marks_in_english 60 35 62 84 (62.6) = 72 := by
  sorry

end davids_english_marks_l2264_226439


namespace modular_congruence_solution_l2264_226401

theorem modular_congruence_solution : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -4376 [ZMOD 10] ∧ n = 4 := by
  sorry

end modular_congruence_solution_l2264_226401


namespace f_max_min_on_interval_l2264_226427

-- Define the function f(x) = -x^3 + 3x^2
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the interval [-2, 2]
def interval : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 20 ∧ min = 0 := by
  sorry

end f_max_min_on_interval_l2264_226427


namespace inequality_solution_l2264_226481

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then
    {x | x ≤ -1}
  else if a > 0 then
    {x | x ≥ 2/a ∨ x ≤ -1}
  else if -2 < a ∧ a < 0 then
    {x | 2/a ≤ x ∧ x ≤ -1}
  else if a = -2 then
    {x | x = -1}
  else
    {x | -1 ≤ x ∧ x ≤ 2/a}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | a*x^2 - 2 ≥ 2*x - a*x} = solution_set a :=
sorry

end inequality_solution_l2264_226481


namespace phone_answer_probability_l2264_226467

/-- The probability of answering the phone on the first ring -/
def p1 : ℝ := 0.1

/-- The probability of answering the phone on the second ring -/
def p2 : ℝ := 0.2

/-- The probability of answering the phone on the third ring -/
def p3 : ℝ := 0.25

/-- The probability of answering the phone on the fourth ring -/
def p4 : ℝ := 0.25

/-- The theorem stating that the probability of answering the phone before the fifth ring is 0.8 -/
theorem phone_answer_probability : p1 + p2 + p3 + p4 = 0.8 := by
  sorry

end phone_answer_probability_l2264_226467


namespace line_intercept_l2264_226460

/-- A line with slope 2 and y-intercept m passing through (1,1) has m = -1 -/
theorem line_intercept (m : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + m)  -- Equation of the line
  → 1 = 2 * 1 + m             -- Line passes through (1,1)
  → m = -1                    -- The y-intercept is -1
:= by sorry

end line_intercept_l2264_226460


namespace trajectory_of_c_l2264_226464

/-- The trajectory of point C in a triangle ABC, where A(-5, 0) and B(5, 0) are fixed points,
    and the product of slopes of AC and BC is -1/2 --/
theorem trajectory_of_c (x y : ℝ) (h : x ≠ 5 ∧ x ≠ -5) :
  (y / (x + 5)) * (y / (x - 5)) = -1/2 →
  x^2 / 25 + y^2 / (25/2) = 1 := by sorry

end trajectory_of_c_l2264_226464


namespace binary_110101_to_hex_35_l2264_226449

def binary_to_decimal (b : List Bool) : ℕ :=
  List.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0 b

def decimal_to_hexadecimal (n : ℕ) : String :=
  let rec aux (m : ℕ) (acc : String) : String :=
    if m = 0 then
      if acc.isEmpty then "0" else acc
    else
      let digit := m % 16
      let hex_digit := if digit < 10 then 
        Char.toString (Char.ofNat (digit + 48))
      else
        Char.toString (Char.ofNat (digit + 55))
      aux (m / 16) (hex_digit ++ acc)
  aux n ""

def binary_110101 : List Bool := [true, true, false, true, false, true]

theorem binary_110101_to_hex_35 :
  decimal_to_hexadecimal (binary_to_decimal binary_110101) = "35" :=
by sorry

end binary_110101_to_hex_35_l2264_226449


namespace negation_equivalence_l2264_226471

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define a function to get the unit digit of a number
def unitDigit (n : ℕ) : ℕ := n % 10

-- The original proposition
def originalProposition : Prop :=
  ∀ x ∈ PositiveIntegers, unitDigit (x^2) ≠ 2

-- The negation of the original proposition
def negationProposition : Prop :=
  ∃ x ∈ PositiveIntegers, unitDigit (x^2) = 2

-- The theorem to prove
theorem negation_equivalence : ¬originalProposition ↔ negationProposition :=
sorry

end negation_equivalence_l2264_226471


namespace wendy_initial_bags_l2264_226423

/-- The number of points earned per recycled bag -/
def points_per_bag : ℕ := 5

/-- The number of bags Wendy didn't recycle -/
def unrecycled_bags : ℕ := 2

/-- The total points Wendy would have earned if she recycled all bags -/
def total_possible_points : ℕ := 45

/-- The number of bags Wendy initially had -/
def initial_bags : ℕ := 11

theorem wendy_initial_bags :
  points_per_bag * (initial_bags - unrecycled_bags) = total_possible_points :=
by sorry

end wendy_initial_bags_l2264_226423


namespace vowel_count_l2264_226482

theorem vowel_count (total_alphabets : ℕ) (num_vowels : ℕ) (h1 : total_alphabets = 20) (h2 : num_vowels = 5) :
  total_alphabets / num_vowels = 4 := by
  sorry

end vowel_count_l2264_226482


namespace unique_ages_l2264_226436

/-- Represents the ages of Abe, Beth, and Charlie -/
structure Ages where
  abe : ℕ
  beth : ℕ
  charlie : ℕ

/-- Checks if the given ages satisfy all the conditions -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.abe + ages.beth = 45 ∧
  ages.abe - ages.beth = 9 ∧
  (ages.abe - 7) + (ages.beth - 7) = 31 ∧
  ages.charlie - ages.abe = 5 ∧
  ages.charlie + ages.beth = 56

/-- Theorem stating that the ages 27, 18, and 38 are the unique solution -/
theorem unique_ages : ∃! ages : Ages, satisfiesConditions ages ∧ 
  ages.abe = 27 ∧ ages.beth = 18 ∧ ages.charlie = 38 := by
  sorry

end unique_ages_l2264_226436


namespace vertex_x_coordinate_l2264_226442

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def passes_through (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y

-- Theorem statement
theorem vertex_x_coordinate 
  (a b c : ℝ) 
  (h1 : passes_through (quadratic a b c) 2 5)
  (h2 : passes_through (quadratic a b c) 8 5)
  (h3 : passes_through (quadratic a b c) 10 16) :
  ∃ (x_vertex : ℝ), x_vertex = 5 ∧ 
    ∀ (x : ℝ), quadratic a b c x ≥ quadratic a b c x_vertex :=
sorry

end vertex_x_coordinate_l2264_226442


namespace lyndee_friends_l2264_226422

theorem lyndee_friends (total_pieces : ℕ) (lyndee_ate : ℕ) (friend_ate : ℕ) : 
  total_pieces = 11 → lyndee_ate = 1 → friend_ate = 2 → 
  (total_pieces - lyndee_ate) / friend_ate = 5 :=
by
  sorry

end lyndee_friends_l2264_226422


namespace scale_model_height_emilys_model_height_l2264_226469

/-- Given an obelisk with height h and base area A, and a scale model with base area a,
    the height of the scale model is h * √(a/A) -/
theorem scale_model_height
  (h : ℝ) -- height of the original obelisk
  (A : ℝ) -- base area of the original obelisk
  (a : ℝ) -- base area of the scale model
  (h_pos : h > 0)
  (A_pos : A > 0)
  (a_pos : a > 0) :
  h * Real.sqrt (a / A) = (h * Real.sqrt a) / Real.sqrt A :=
by sorry

/-- The height of Emily's scale model obelisk is 5√10 meters -/
theorem emilys_model_height
  (h : ℝ) -- height of the original obelisk
  (A : ℝ) -- base area of the original obelisk
  (a : ℝ) -- base area of the scale model
  (h_eq : h = 50)
  (A_eq : A = 25)
  (a_eq : a = 0.025) :
  h * Real.sqrt (a / A) = 5 * Real.sqrt 10 :=
by sorry

end scale_model_height_emilys_model_height_l2264_226469


namespace product_of_two_digit_numbers_is_8670_l2264_226478

theorem product_of_two_digit_numbers_is_8670 : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8670 ∧
  8670 = 8670 := by
sorry

end product_of_two_digit_numbers_is_8670_l2264_226478


namespace debugging_time_l2264_226498

theorem debugging_time (total_hours : ℝ) (flow_chart_fraction : ℝ) (coding_fraction : ℝ)
  (h1 : total_hours = 48)
  (h2 : flow_chart_fraction = 1/4)
  (h3 : coding_fraction = 3/8)
  (h4 : flow_chart_fraction + coding_fraction < 1) :
  total_hours * (1 - flow_chart_fraction - coding_fraction) = 18 :=
by sorry

end debugging_time_l2264_226498


namespace set_intersection_theorem_l2264_226496

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1)}

def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem set_intersection_theorem : N ∩ (U \ M) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end set_intersection_theorem_l2264_226496


namespace author_earnings_proof_l2264_226444

/-- Calculates the author's earnings from book sales -/
def author_earnings (paper_copies : ℕ) (paper_price : ℚ) (paper_percentage : ℚ)
                    (hard_copies : ℕ) (hard_price : ℚ) (hard_percentage : ℚ) : ℚ :=
  let paper_total := paper_copies * paper_price
  let hard_total := hard_copies * hard_price
  paper_total * paper_percentage + hard_total * hard_percentage

/-- Proves that the author's earnings are $1,104 given the specified conditions -/
theorem author_earnings_proof :
  author_earnings 32000 0.20 0.06 15000 0.40 0.12 = 1104 := by
  sorry

#eval author_earnings 32000 (20 / 100) (6 / 100) 15000 (40 / 100) (12 / 100)

end author_earnings_proof_l2264_226444


namespace abc_is_cube_l2264_226459

theorem abc_is_cube (a b c : ℕ) : 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (∀ n : ℕ, 2 ≤ n ∧ n ≤ 100 → ¬(n ∣ a) ∧ ¬(n ∣ b) ∧ ¬(n ∣ c)) →
  (∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z →
    (∀ n : ℕ, 2 ≤ n ∧ n ≤ 100 → ¬(n ∣ x) ∧ ¬(n ∣ y) ∧ ¬(n ∣ z)) →
    a ≤ x ∧ b ≤ y ∧ c ≤ z) →
  ∃ k : ℕ, a * b * c = k^3 :=
by sorry

end abc_is_cube_l2264_226459


namespace rectangle_dimension_difference_l2264_226477

theorem rectangle_dimension_difference (L B D : ℝ) : 
  L - B = D →
  2 * (L + B) = 246 →
  L * B = 3650 →
  D^2 = 29729 := by sorry

end rectangle_dimension_difference_l2264_226477


namespace trip_cost_equalization_l2264_226479

/-- Given three people who shared expenses on a trip, this theorem calculates
    the amount two of them must pay the third to equalize the costs. -/
theorem trip_cost_equalization
  (A B C : ℝ)  -- Amounts paid by LeRoy, Bernardo, and Carlos respectively
  (h1 : A < B) (h2 : B < C)  -- Ordering of the amounts
  : (2 * C - A - B) / 3 = 
    ((A + B + C) / 3 - A) + ((A + B + C) / 3 - B) :=
by sorry


end trip_cost_equalization_l2264_226479


namespace rhombus_diagonals_l2264_226497

-- Define the rhombus
structure Rhombus where
  perimeter : ℝ
  diagonal_difference : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

-- State the theorem
theorem rhombus_diagonals (r : Rhombus) :
  r.perimeter = 100 ∧ r.diagonal_difference = 34 →
  r.diagonal1 = 14 ∧ r.diagonal2 = 48 := by
  sorry


end rhombus_diagonals_l2264_226497


namespace group_size_proof_l2264_226465

theorem group_size_proof : 
  ∀ n : ℕ, 
  (n : ℝ) * (n : ℝ) = 9801 → 
  n = 99 :=
by
  sorry

end group_size_proof_l2264_226465


namespace toys_cost_price_gained_l2264_226410

/-- Calculates the number of toys' cost price gained in a sale --/
theorem toys_cost_price_gained
  (num_toys : ℕ)
  (total_selling_price : ℕ)
  (cost_price_per_toy : ℕ)
  (h1 : num_toys = 18)
  (h2 : total_selling_price = 16800)
  (h3 : cost_price_per_toy = 800) :
  (total_selling_price - num_toys * cost_price_per_toy) / cost_price_per_toy = 3 :=
by sorry

end toys_cost_price_gained_l2264_226410


namespace factor_polynomial_l2264_226435

theorem factor_polynomial (x : ℝ) : 
  x^2 - 6*x + 9 - 64*x^4 = (8*x^2 + x - 3)*(-8*x^2 + x - 3) := by
sorry

end factor_polynomial_l2264_226435


namespace passage_uses_deductive_reasoning_l2264_226400

/-- Represents a statement in the chain of reasoning --/
inductive Statement
| NamesNotCorrect
| LanguageNotCorrect
| ThingsNotDoneSuccessfully
| RitualsAndMusicNotFlourish
| PunishmentsNotProper
| PeopleConfused

/-- Represents the chain of reasoning in the passage --/
def reasoning_chain : List (Statement × Statement) :=
  [(Statement.NamesNotCorrect, Statement.LanguageNotCorrect),
   (Statement.LanguageNotCorrect, Statement.ThingsNotDoneSuccessfully),
   (Statement.ThingsNotDoneSuccessfully, Statement.RitualsAndMusicNotFlourish),
   (Statement.RitualsAndMusicNotFlourish, Statement.PunishmentsNotProper),
   (Statement.PunishmentsNotProper, Statement.PeopleConfused)]

/-- Definition of deductive reasoning --/
def is_deductive_reasoning (chain : List (Statement × Statement)) : Prop :=
  ∀ (premise conclusion : Statement), 
    (premise, conclusion) ∈ chain → 
    (∃ (general_premise : Statement), 
      (general_premise, premise) ∈ chain ∧ (general_premise, conclusion) ∈ chain)

/-- Theorem stating that the reasoning in the passage is deductive --/
theorem passage_uses_deductive_reasoning : 
  is_deductive_reasoning reasoning_chain :=
sorry

end passage_uses_deductive_reasoning_l2264_226400


namespace candy_distribution_l2264_226446

theorem candy_distribution (num_children : ℕ) : 
  (3 * num_children + 12 = 5 * num_children - 10) → num_children = 11 := by
sorry

end candy_distribution_l2264_226446


namespace playground_children_count_l2264_226490

theorem playground_children_count :
  ∀ (girls boys : ℕ),
    girls = 28 →
    boys = 35 →
    girls + boys = 63 :=
by
  sorry

end playground_children_count_l2264_226490


namespace average_rate_of_change_x_squared_l2264_226486

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- Define the interval
def a : ℝ := 1
def b : ℝ := 2

-- State the theorem
theorem average_rate_of_change_x_squared :
  (f b - f a) / (b - a) = 3 := by
  sorry

end average_rate_of_change_x_squared_l2264_226486


namespace triangle_area_rational_l2264_226415

theorem triangle_area_rational
  (m n p q : ℚ)
  (hm : m > 0)
  (hn : n > 0)
  (hp : p > 0)
  (hq : q > 0)
  (hmn : m > n)
  (hpq : p > q)
  (a : ℚ := m * n * (p^2 + q^2))
  (b : ℚ := p * q * (m^2 + n^2))
  (c : ℚ := (m * q + n * p) * (m * p - n * q)) :
  ∃ (t : ℚ), t = m * n * p * q * (m * q + n * p) * (m * p - n * q) :=
by sorry

end triangle_area_rational_l2264_226415


namespace arctans_sum_to_pi_l2264_226438

theorem arctans_sum_to_pi : 
  Real.arctan (1/3) + Real.arctan (3/8) + Real.arctan (8/3) = π := by
  sorry

end arctans_sum_to_pi_l2264_226438


namespace complex_equidistant_points_l2264_226405

theorem complex_equidistant_points (z : ℂ) : 
  (Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
   Complex.abs (z - 2) = Complex.abs (z + 2*I)) ↔ 
  z = -1 + I :=
by sorry

end complex_equidistant_points_l2264_226405


namespace line_plane_relationship_l2264_226474

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Defines when a line is contained in a plane -/
def Line3D.containedIn (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Defines when a line is parallel to a plane -/
def Line3D.parallelTo (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Defines when a line intersects a plane -/
def Line3D.intersects (l : Line3D) (p : Plane3D) : Prop := sorry

/-- The main theorem -/
theorem line_plane_relationship (a : Line3D) (α : Plane3D) :
  ¬(a.containedIn α) → (a.parallelTo α ∨ a.intersects α) := by
  sorry

end line_plane_relationship_l2264_226474


namespace fifth_term_is_negative_ten_l2264_226441

/-- A sequence satisfying the given conditions -/
def special_sequence (a : ℕ+ → ℤ) : Prop :=
  a 1 = -2 ∧ ∀ m n : ℕ+, a (m + n) = a m + a n

/-- The theorem stating that the 5th term of the sequence is -10 -/
theorem fifth_term_is_negative_ten (a : ℕ+ → ℤ) (h : special_sequence a) : 
  a 5 = -10 := by sorry

end fifth_term_is_negative_ten_l2264_226441


namespace figure_dimension_proof_l2264_226420

theorem figure_dimension_proof (x : ℝ) : 
  let square_area := (3 * x)^2
  let rectangle_area := 2 * x * 6 * x
  let triangle_area := (1 / 2) * (3 * x) * (2 * x)
  let total_area := square_area + rectangle_area + triangle_area
  total_area = 1000 → x = (5 * Real.sqrt 15) / 3 := by
sorry

end figure_dimension_proof_l2264_226420


namespace first_friend_slices_l2264_226421

def burgers : ℕ := 5
def friends : ℕ := 4
def slices_per_burger : ℕ := 2
def second_friend_slices : ℕ := 2
def third_friend_slices : ℕ := 3
def fourth_friend_slices : ℕ := 3
def era_slices : ℕ := 1

theorem first_friend_slices : 
  burgers * slices_per_burger - 
  (second_friend_slices + third_friend_slices + fourth_friend_slices + era_slices) = 1 := by
  sorry

end first_friend_slices_l2264_226421


namespace laundry_ratio_l2264_226499

def wednesday_loads : ℕ := 6
def thursday_loads : ℕ := 2 * wednesday_loads
def saturday_loads : ℕ := wednesday_loads / 3
def total_loads : ℕ := 26

def friday_loads : ℕ := total_loads - (wednesday_loads + thursday_loads + saturday_loads)

theorem laundry_ratio :
  (friday_loads : ℚ) / thursday_loads = 1 / 2 := by
  sorry

end laundry_ratio_l2264_226499


namespace sugar_solution_replacement_l2264_226453

theorem sugar_solution_replacement (original_sugar_percent : ℝ) 
                                   (second_sugar_percent : ℝ) 
                                   (final_sugar_percent : ℝ) :
  original_sugar_percent = 10 →
  second_sugar_percent = 26.000000000000007 →
  final_sugar_percent = 14 →
  ∃ (x : ℝ), x = 0.25 ∧ 
             (1 - x) * (original_sugar_percent / 100) + 
             x * (second_sugar_percent / 100) = 
             final_sugar_percent / 100 :=
by sorry

end sugar_solution_replacement_l2264_226453


namespace root_difference_implies_k_value_l2264_226462

theorem root_difference_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0 ∧ 
   (r+2)^2 - k*(r+2) + 10 = 0 ∧ (s+2)^2 - k*(s+2) + 10 = 0) → k = 2 :=
by sorry

end root_difference_implies_k_value_l2264_226462


namespace tree_growth_l2264_226454

theorem tree_growth (initial_circumference : ℝ) (annual_increase : ℝ) (target_circumference : ℝ) :
  initial_circumference = 10 ∧ 
  annual_increase = 3 ∧ 
  target_circumference = 90 →
  ∃ x : ℝ, x > 80 / 3 ∧ initial_circumference + x * annual_increase > target_circumference :=
by sorry

end tree_growth_l2264_226454


namespace sara_balloons_l2264_226428

/-- Given that Sara initially had 31 red balloons and gave away 24 red balloons,
    prove that she is left with 7 red balloons. -/
theorem sara_balloons (initial_red : Nat) (given_away : Nat) (remaining : Nat) 
    (h1 : initial_red = 31)
    (h2 : given_away = 24)
    (h3 : remaining = initial_red - given_away) : 
  remaining = 7 := by
  sorry

end sara_balloons_l2264_226428


namespace problem_1_problem_2_problem_3_problem_4_l2264_226473

-- Problem 1
theorem problem_1 : 23 + (-16) - (-7) = 14 := by sorry

-- Problem 2
theorem problem_2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by sorry

-- Problem 3
theorem problem_3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -10/3 := by sorry

-- Problem 4
theorem problem_4 : -1^4 - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2264_226473


namespace smallest_integer_with_remainders_l2264_226448

theorem smallest_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 6 = 3) ∧ 
  (x % 8 = 2) ∧ 
  (∀ y : ℕ, y > 0 ∧ y % 6 = 3 ∧ y % 8 = 2 → x ≤ y) ∧
  (x = 33) := by
sorry

end smallest_integer_with_remainders_l2264_226448


namespace cubic_factorization_l2264_226412

theorem cubic_factorization (x : ℝ) : x^3 - 4*x^2 + 4*x = x*(x-2)^2 := by
  sorry

end cubic_factorization_l2264_226412


namespace binomial_26_7_l2264_226484

theorem binomial_26_7 (h1 : Nat.choose 24 5 = 42504)
                      (h2 : Nat.choose 24 6 = 134596)
                      (h3 : Nat.choose 24 7 = 346104) :
  Nat.choose 26 7 = 657800 := by
  sorry

end binomial_26_7_l2264_226484


namespace survey_problem_l2264_226492

theorem survey_problem (A B : ℕ) : 
  (A * 20 = B * 50) →                   -- 20% of A equals 50% of B (students who read both books)
  (A - A * 20 / 100) - (B - B * 50 / 100) = 150 →  -- Difference between those who read only A and only B
  (A - A * 20 / 100) + (B - B * 50 / 100) + (A * 20 / 100) = 300 :=  -- Total number of students
by sorry

end survey_problem_l2264_226492


namespace ariel_current_age_l2264_226495

/-- Represents a person with birth year and fencing information -/
structure Person where
  birth_year : Nat
  fencing_start_year : Nat
  years_fencing : Nat

/-- Calculate the current year based on fencing information -/
def current_year (p : Person) : Nat :=
  p.fencing_start_year + p.years_fencing

/-- Calculate the age of a person in a given year -/
def age_in_year (p : Person) (year : Nat) : Nat :=
  year - p.birth_year

/-- Ariel's information -/
def ariel : Person :=
  { birth_year := 1992
  , fencing_start_year := 2006
  , years_fencing := 16 }

/-- Theorem: Ariel's current age is 30 years old -/
theorem ariel_current_age :
  age_in_year ariel (current_year ariel) = 30 := by
  sorry

end ariel_current_age_l2264_226495


namespace construction_possible_l2264_226418

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- Check if a line is tangent to a circle -/
def tangent_to_circle (l : Line) (c : Circle) : Prop :=
  sorry

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  sorry

/-- Check if a point lies on a circle -/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  sorry

/-- The main theorem -/
theorem construction_possible 
  (E F G H : ℝ × ℝ) : 
  ∃ (e f g : Line) (k : Circle),
    perpendicular e f ∧
    tangent_to_circle e k ∧
    tangent_to_circle f k ∧
    point_on_line E g ∧
    point_on_line F g ∧
    point_on_circle G k ∧
    point_on_circle H k ∧
    point_on_line G g ∧
    point_on_line H g :=
  sorry

end construction_possible_l2264_226418


namespace work_completion_time_l2264_226402

/-- The number of laborers originally employed by the contractor -/
def original_laborers : ℚ := 17.5

/-- The number of absent laborers -/
def absent_laborers : ℕ := 7

/-- The number of days it took the remaining laborers to complete the work -/
def actual_days : ℕ := 10

/-- The original number of days the work was supposed to be completed in -/
def original_days : ℚ := (original_laborers - absent_laborers : ℚ) * actual_days / original_laborers

theorem work_completion_time : original_days = 6 := by sorry

end work_completion_time_l2264_226402


namespace margo_walk_l2264_226419

theorem margo_walk (outbound_speed return_speed : ℝ) (total_time : ℝ) 
  (h1 : outbound_speed = 5)
  (h2 : return_speed = 3)
  (h3 : total_time = 1) :
  let distance := (outbound_speed * return_speed * total_time) / (outbound_speed + return_speed)
  2 * distance = 3.75 := by
  sorry

end margo_walk_l2264_226419


namespace max_rows_with_unique_letters_l2264_226455

theorem max_rows_with_unique_letters : ∃ (m : ℕ),
  (∀ (n : ℕ), n > m → ¬∃ (table : Fin n → Fin 8 → Fin 4),
    ∀ (i j : Fin n), i ≠ j →
      (∃! (k : Fin 8), table i k = table j k) ∨
      (∀ (k : Fin 8), table i k ≠ table j k)) ∧
  (∃ (table : Fin m → Fin 8 → Fin 4),
    ∀ (i j : Fin m), i ≠ j →
      (∃! (k : Fin 8), table i k = table j k) ∨
      (∀ (k : Fin 8), table i k ≠ table j k)) ∧
  m = 28 :=
by sorry


end max_rows_with_unique_letters_l2264_226455


namespace square_area_error_l2264_226457

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * 1.05
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.1025 := by
sorry

end square_area_error_l2264_226457


namespace brads_running_speed_l2264_226414

/-- Proof of Brad's running speed given the conditions of the problem -/
theorem brads_running_speed 
  (total_distance : ℝ) 
  (maxwells_speed : ℝ) 
  (time_until_meeting : ℝ) 
  (brad_delay : ℝ) 
  (h1 : total_distance = 24) 
  (h2 : maxwells_speed = 4) 
  (h3 : time_until_meeting = 3) 
  (h4 : brad_delay = 1) : 
  (total_distance - maxwells_speed * time_until_meeting) / (time_until_meeting - brad_delay) = 6 := by
  sorry

#check brads_running_speed

end brads_running_speed_l2264_226414


namespace train_length_calculation_l2264_226476

/-- Calculates the length of a train given the speeds of a jogger and the train,
    the initial distance between them, and the time it takes for the train to pass the jogger. -/
theorem train_length_calculation (jogger_speed train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ)
    (h1 : jogger_speed = 9 / 3.6) -- Convert 9 km/hr to m/s
    (h2 : train_speed = 45 / 3.6) -- Convert 45 km/hr to m/s
    (h3 : initial_distance = 240)
    (h4 : passing_time = 35) :
    train_speed * passing_time - jogger_speed * passing_time - initial_distance = 110 := by
  sorry

#check train_length_calculation

end train_length_calculation_l2264_226476


namespace journey_speed_l2264_226451

theorem journey_speed (d : ℝ) (v : ℝ) :
  d > 0 ∧ v > 0 ∧
  3 * d = 1.5 ∧
  d / 5 + d / 10 + d / v = 11 / 60 →
  v = 15 := by
sorry

end journey_speed_l2264_226451


namespace cuboid_faces_at_vertex_l2264_226417

/-- A cuboid is a three-dimensional geometric shape -/
structure Cuboid where

/-- A vertex is a point where edges of a geometric shape meet -/
structure Vertex where

/-- Represents a face of a geometric shape -/
structure Face where

/-- The number of faces meeting at a vertex of a cuboid -/
def faces_at_vertex (c : Cuboid) (v : Vertex) : ℕ := sorry

/-- Theorem stating that the number of faces meeting at a vertex of a cuboid is 3 -/
theorem cuboid_faces_at_vertex (c : Cuboid) (v : Vertex) : 
  faces_at_vertex c v = 3 := by sorry

end cuboid_faces_at_vertex_l2264_226417


namespace f_opens_upwards_f_passes_through_origin_f_satisfies_conditions_l2264_226416

/-- A quadratic function that opens upwards and passes through (0,1) -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The graph of f opens upwards -/
theorem f_opens_upwards : ∀ x y : ℝ, x < y → f ((x + y) / 2) < (f x + f y) / 2 := by sorry

/-- f passes through the point (0,1) -/
theorem f_passes_through_origin : f 0 = 1 := by sorry

/-- f is a quadratic function satisfying the given conditions -/
theorem f_satisfies_conditions : 
  (∀ x y : ℝ, x < y → f ((x + y) / 2) < (f x + f y) / 2) ∧ 
  f 0 = 1 := by sorry

end f_opens_upwards_f_passes_through_origin_f_satisfies_conditions_l2264_226416


namespace demand_decrease_with_price_increase_l2264_226403

theorem demand_decrease_with_price_increase (P Q : ℝ) (P_new Q_new : ℝ) :
  P > 0 → Q > 0 →
  P_new = 1.5 * P →
  P_new * Q_new = 1.2 * P * Q →
  Q_new = 0.8 * Q :=
by
  sorry

#check demand_decrease_with_price_increase

end demand_decrease_with_price_increase_l2264_226403


namespace remainder_67_power_67_plus_67_mod_68_l2264_226483

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end remainder_67_power_67_plus_67_mod_68_l2264_226483


namespace calculate_expression_l2264_226424

theorem calculate_expression : (-3)^2 + 2017^0 - Real.sqrt 18 * Real.sin (π/4) = 7 := by
  sorry

end calculate_expression_l2264_226424


namespace exhibition_solution_l2264_226413

/-- The number of paintings contributed by each grade in a school exhibition --/
structure PaintingExhibition where
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  fourth_grade : ℕ

/-- Properties of the painting exhibition --/
def ValidExhibition (e : PaintingExhibition) : Prop :=
  e.first_grade = 20 ∧
  e.second_grade = 45 ∧
  e.third_grade = e.first_grade + e.second_grade - 17 ∧
  e.fourth_grade = 2 * e.third_grade - 36

/-- Theorem stating the correct number of paintings for third and fourth grades --/
theorem exhibition_solution (e : PaintingExhibition) (h : ValidExhibition e) :
  e.third_grade = 48 ∧ e.fourth_grade = 60 := by
  sorry


end exhibition_solution_l2264_226413


namespace class_gpa_theorem_l2264_226440

/-- The grade point average (GPA) of a class, given the GPAs of two subgroups -/
def classGPA (fraction1 : ℚ) (gpa1 : ℚ) (fraction2 : ℚ) (gpa2 : ℚ) : ℚ :=
  fraction1 * gpa1 + fraction2 * gpa2

/-- Theorem: The GPA of a class where one-third has GPA 60 and two-thirds has GPA 66 is 64 -/
theorem class_gpa_theorem :
  classGPA (1/3) 60 (2/3) 66 = 64 := by
  sorry

end class_gpa_theorem_l2264_226440


namespace supreme_sports_package_channels_l2264_226432

/-- Represents the changes in Larry's cable package --/
structure CablePackage where
  initial : ℕ
  removed : ℕ
  replaced : ℕ
  reduced : ℕ
  sportsAdded : ℕ
  final : ℕ

/-- Calculates the number of channels in the supreme sports package --/
def supremeSportsPackage (cp : CablePackage) : ℕ :=
  cp.final - (cp.initial - cp.removed + cp.replaced - cp.reduced + cp.sportsAdded)

/-- Theorem stating that the supreme sports package contains 7 channels --/
theorem supreme_sports_package_channels : 
  ∀ (cp : CablePackage), 
    cp.initial = 150 ∧ 
    cp.removed = 20 ∧ 
    cp.replaced = 12 ∧ 
    cp.reduced = 10 ∧ 
    cp.sportsAdded = 8 ∧ 
    cp.final = 147 → 
    supremeSportsPackage cp = 7 := by
  sorry

#eval supremeSportsPackage ⟨150, 20, 12, 10, 8, 147⟩

end supreme_sports_package_channels_l2264_226432


namespace sandys_age_l2264_226458

/-- Proves that Sandy's age is 42 given the conditions -/
theorem sandys_age (sandy_age molly_age : ℕ) : 
  molly_age = sandy_age + 12 → 
  sandy_age * 9 = molly_age * 7 →
  sandy_age = 42 := by
sorry

end sandys_age_l2264_226458


namespace one_twentieth_of_80_l2264_226447

-- Define the given condition (even though it's mathematically incorrect)
def one_ninth_of_60 : ℚ := 5

-- Define the function to calculate a fraction of a number
def fraction_of (numerator denominator number : ℚ) : ℚ :=
  (numerator / denominator) * number

-- State the theorem
theorem one_twentieth_of_80 :
  fraction_of 1 20 80 = 4 :=
sorry

end one_twentieth_of_80_l2264_226447


namespace min_ratio_two_digit_integers_l2264_226485

theorem min_ratio_two_digit_integers (x y : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ y ≥ 10 ∧ y ≤ 99 →  -- x and y are two-digit integers
  (x + y) / 2 = 65 →                   -- mean is 65
  x > 50 ∧ y > 50 →                    -- x and y are both greater than 50
  ∀ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (a + b) / 2 = 65 ∧ a > 50 ∧ b > 50 →
  (a : ℚ) / b ≥ (51 : ℚ) / 79 →
  (x : ℚ) / y ≥ (51 : ℚ) / 79 := by
sorry

end min_ratio_two_digit_integers_l2264_226485


namespace sequential_draw_probability_l2264_226404

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of cards of each suit in a standard deck -/
def cardsPerSuit : ℕ := 13

/-- The probability of drawing a club, then a diamond, then a heart in order from a standard deck -/
def sequentialDrawProbability : ℚ :=
  (cardsPerSuit : ℚ) / standardDeckSize *
  (cardsPerSuit : ℚ) / (standardDeckSize - 1) *
  (cardsPerSuit : ℚ) / (standardDeckSize - 2)

theorem sequential_draw_probability :
  sequentialDrawProbability = 2197 / 132600 := by
  sorry

end sequential_draw_probability_l2264_226404


namespace calculate_expression_l2264_226443

theorem calculate_expression : 
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 := by
  sorry

end calculate_expression_l2264_226443


namespace fraction_sum_equality_l2264_226494

theorem fraction_sum_equality (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_sum : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 
  1 / (q - r) + 1 / (r - p) + 1 / (p - q) - 1 := by
  sorry

end fraction_sum_equality_l2264_226494


namespace remaining_distance_l2264_226406

theorem remaining_distance (total_distance driven_distance : ℕ) 
  (h1 : total_distance = 1200)
  (h2 : driven_distance = 384) :
  total_distance - driven_distance = 816 := by
  sorry

end remaining_distance_l2264_226406


namespace counterexample_to_goldbach_like_conjecture_l2264_226411

theorem counterexample_to_goldbach_like_conjecture :
  (∃ n : ℕ, n > 5 ∧ Odd n ∧ ¬∃ (a b c : ℕ), Odd a ∧ Odd b ∧ Odd c ∧ n = a + b + c) ∨
  (∃ n : ℕ, n > 5 ∧ Even n ∧ ¬∃ (a b c : ℕ), Odd a ∧ Odd b ∧ Odd c ∧ n = a + b + c) →
  ¬(∀ n : ℕ, n > 5 → ∃ (a b c : ℕ), Odd a ∧ Odd b ∧ Odd c ∧ n = a + b + c) :=
by sorry

end counterexample_to_goldbach_like_conjecture_l2264_226411


namespace tylers_dogs_l2264_226487

/-- Proves that Tyler had 15 dogs initially, given the conditions of the problem -/
theorem tylers_dogs : ∀ (initial_dogs : ℕ), 
  (initial_dogs * 5 = 75) → initial_dogs = 15 := by
  sorry

end tylers_dogs_l2264_226487


namespace sqrt_of_sqrt_16_l2264_226480

theorem sqrt_of_sqrt_16 : ∃ (x : ℝ), x^2 = 16 ∧ (∀ y : ℝ, y^2 = x → y = 2 ∨ y = -2) := by
  sorry

end sqrt_of_sqrt_16_l2264_226480


namespace cindy_marbles_l2264_226429

theorem cindy_marbles (initial_marbles : Nat) (friends : Nat) (marbles_per_friend : Nat) : 
  initial_marbles = 500 → friends = 4 → marbles_per_friend = 80 →
  4 * (initial_marbles - friends * marbles_per_friend) = 720 := by
  sorry

end cindy_marbles_l2264_226429


namespace some_employees_not_team_leaders_l2264_226475

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Employee : U → Prop)
variable (TeamLeader : U → Prop)
variable (MeetsDeadlines : U → Prop)

-- State the theorem
theorem some_employees_not_team_leaders
  (h1 : ∃ x, Employee x ∧ ¬MeetsDeadlines x)
  (h2 : ∀ x, TeamLeader x → MeetsDeadlines x) :
  ∃ x, Employee x ∧ ¬TeamLeader x :=
by
  sorry

end some_employees_not_team_leaders_l2264_226475


namespace line_direction_vector_l2264_226445

/-- The direction vector of a parameterized line -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

/-- The point on the line at t = 0 -/
def initial_point (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

theorem line_direction_vector :
  let line (t : ℝ) : ℝ × ℝ := 
    (7 - 25 / Real.sqrt 41 * t, 3 - 20 / Real.sqrt 41 * t)
  let y (x : ℝ) : ℝ := (4 * x - 7) / 5
  ∀ x ≤ 7, 
    let point := (x, y x)
    let distance := Real.sqrt ((x - 7)^2 + (y x - 3)^2)
    (∃ t, point = line t ∧ distance = t) →
    direction_vector line = (-25 / Real.sqrt 41, -20 / Real.sqrt 41) :=
by sorry

end line_direction_vector_l2264_226445


namespace equation_solution_l2264_226425

theorem equation_solution (x : ℝ) (h : x ≠ -1) :
  (x^2 + 2*x + 3) / (x + 1) = x + 3 ↔ x = 0 := by
sorry

end equation_solution_l2264_226425


namespace product_of_integers_l2264_226450

theorem product_of_integers (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 30)
  (h3 : 1 / p + 1 / q + 1 / r + 420 / (p * q * r) = 1) :
  p * q * r = 576 := by
  sorry

end product_of_integers_l2264_226450


namespace line_slope_range_l2264_226470

/-- Given two points A and B, and a line l that intersects line segment AB,
    prove that the range of possible slopes for line l is (-∞,-4] ∪ [1/2,+∞). -/
theorem line_slope_range (a : ℝ) :
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (4, 0)
  let l := {(x, y) : ℝ × ℝ | a * x + y - 2 * a + 1 = 0}
  let intersects := ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    (∃ (x y : ℝ), (x, y) = (1 - t) • A + t • B ∧ (x, y) ∈ l)
  intersects →
  (a ≤ -4 ∨ a ≥ 1/2) :=
sorry

end line_slope_range_l2264_226470


namespace track_length_is_480_l2264_226437

/-- Represents the circular track and the runners' properties -/
structure Track :=
  (length : ℝ)
  (brenda_speed : ℝ)
  (sally_speed : ℝ)

/-- The conditions of the problem -/
def problem_conditions (track : Track) : Prop :=
  ∃ (t1 t2 : ℝ),
    -- First meeting
    track.brenda_speed * t1 = 120 ∧
    track.sally_speed * t1 = track.length / 2 - 120 ∧
    -- Second meeting
    track.sally_speed * (t1 + t2) = track.length / 2 + 60 ∧
    track.brenda_speed * (t1 + t2) = track.length / 2 - 60 ∧
    -- Constant speeds
    track.brenda_speed > 0 ∧
    track.sally_speed > 0

/-- The theorem stating that the track length is 480 meters -/
theorem track_length_is_480 :
  ∃ (track : Track), problem_conditions track ∧ track.length = 480 :=
sorry

end track_length_is_480_l2264_226437


namespace inverse_A_times_B_l2264_226493

theorem inverse_A_times_B : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 1]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; -1, 5]
  A⁻¹ * B = !![1/2, 1; -1/2, 5] := by sorry

end inverse_A_times_B_l2264_226493


namespace find_y_l2264_226463

theorem find_y (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 := by
  sorry

end find_y_l2264_226463


namespace job_completion_time_work_completion_time_jose_raju_l2264_226408

/-- The time required for two workers to complete a job together, given their individual completion times -/
theorem job_completion_time (jose_time raju_time : ℝ) (jose_time_pos : jose_time > 0) (raju_time_pos : raju_time > 0) :
  (1 / jose_time + 1 / raju_time)⁻¹ = (jose_time * raju_time) / (jose_time + raju_time) :=
by sorry

theorem work_completion_time_jose_raju :
  let jose_time : ℝ := 10
  let raju_time : ℝ := 40
  (1 / jose_time + 1 / raju_time)⁻¹ = 8 :=
by sorry

end job_completion_time_work_completion_time_jose_raju_l2264_226408


namespace largest_two_prime_product_digit_product_l2264_226456

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def digit_product (n : ℕ) : ℕ := 
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

theorem largest_two_prime_product_digit_product : 
  ∃ m d e : ℕ,
    is_prime d ∧ 
    2 ≤ d ∧ d ≤ 5 ∧
    e = d + 10 ∧
    is_prime e ∧
    m = d * e ∧
    (∀ m' d' e' : ℕ, 
      is_prime d' ∧ 
      2 ≤ d' ∧ d' ≤ 5 ∧ 
      e' = d' + 10 ∧ 
      is_prime e' ∧ 
      m' = d' * e' → 
      m' ≤ m) ∧
    digit_product m = 27 :=
sorry

end largest_two_prime_product_digit_product_l2264_226456


namespace exists_special_subset_l2264_226461

/-- Definition of arithmetic mean -/
def arithmetic_mean (S : Finset ℕ) : ℚ :=
  (S.sum id) / S.card

/-- Definition of perfect power -/
def is_perfect_power (n : ℚ) : Prop :=
  ∃ (a : ℚ) (k : ℕ), k > 1 ∧ n = a ^ k

/-- Main theorem -/
theorem exists_special_subset :
  ∃ (A : Finset ℕ), A.card = 2022 ∧
    ∀ (B : Finset ℕ), B ⊆ A →
      is_perfect_power (arithmetic_mean B) :=
sorry

end exists_special_subset_l2264_226461


namespace greatest_x_value_l2264_226466

theorem greatest_x_value (x : ℤ) : 
  (2.13 * (10 : ℝ)^(x : ℝ) < 2100) ∧ 
  (∀ y : ℤ, y > x → 2.13 * (10 : ℝ)^(y : ℝ) ≥ 2100) → 
  x = 2 :=
by sorry

end greatest_x_value_l2264_226466


namespace handshakes_in_room_l2264_226468

/-- Represents the number of handshakes in a room with specific friendship conditions -/
def number_of_handshakes (total_people : ℕ) (friends : ℕ) (strangers : ℕ) : ℕ :=
  -- Handshakes between friends and strangers
  friends * strangers +
  -- Handshakes among strangers who know no one
  (strangers - 5).choose 2 +
  -- Handshakes between strangers who know one person and those who know no one
  5 * (strangers - 5)

/-- Theorem stating the number of handshakes in the given scenario -/
theorem handshakes_in_room (total_people : ℕ) (friends : ℕ) (strangers : ℕ) 
  (h1 : total_people = 40)
  (h2 : friends = 25)
  (h3 : strangers = 15)
  (h4 : friends + strangers = total_people) :
  number_of_handshakes total_people friends strangers = 345 := by
  sorry

#eval number_of_handshakes 40 25 15

end handshakes_in_room_l2264_226468


namespace represent_383_l2264_226431

/-- Given a number of hundreds, tens, and ones, calculate the represented number. -/
def representedNumber (hundreds tens ones : ℕ) : ℕ :=
  100 * hundreds + 10 * tens + ones

/-- Prove that 3 hundreds, 8 tens, and 3 ones represent the number 383. -/
theorem represent_383 : representedNumber 3 8 3 = 383 := by
  sorry

end represent_383_l2264_226431


namespace special_triangle_DE_length_l2264_226472

/-- Triangle ABC with given side lengths and DE parallel to BC containing the incenter -/
structure SpecialTriangle where
  -- Side lengths
  AB : ℝ
  AC : ℝ
  BC : ℝ
  -- Points D and E
  D : ℝ × ℝ
  E : ℝ × ℝ
  -- Condition that DE is parallel to BC
  DE_parallel_BC : Bool
  -- Condition that DE contains the incenter
  DE_contains_incenter : Bool

/-- The length of DE in the special triangle -/
def length_DE (t : SpecialTriangle) : ℝ := sorry

/-- Theorem stating the length of DE in the special case -/
theorem special_triangle_DE_length :
  ∀ t : SpecialTriangle,
  t.AB = 28 ∧ t.AC = 29 ∧ t.BC = 26 ∧ t.DE_parallel_BC ∧ t.DE_contains_incenter →
  length_DE t = 806 / 57 := by sorry

end special_triangle_DE_length_l2264_226472
