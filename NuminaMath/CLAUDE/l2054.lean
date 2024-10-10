import Mathlib

namespace determinant_max_value_l2054_205414

theorem determinant_max_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a * b - 1 ≥ x * y - 1) →
  a * b - 1 = 0 :=
sorry

end determinant_max_value_l2054_205414


namespace virus_spread_l2054_205430

/-- The average number of computers infected by one computer in each round -/
def average_infection_rate : ℝ := 8

/-- The number of infected computers after two rounds -/
def infected_after_two_rounds : ℕ := 81

/-- The number of infected computers after three rounds -/
def infected_after_three_rounds : ℕ := 729

theorem virus_spread :
  (1 + average_infection_rate + average_infection_rate ^ 2 = infected_after_two_rounds) ∧
  ((1 + average_infection_rate) ^ 3 > 700) := by
  sorry

end virus_spread_l2054_205430


namespace younger_person_age_l2054_205496

theorem younger_person_age (y e : ℕ) : 
  e = y + 20 →                  -- The ages differ by 20 years
  e - 8 = 5 * (y - 8) →         -- 8 years ago, elder was 5 times younger's age
  y = 13                        -- The younger person's age is 13
  := by sorry

end younger_person_age_l2054_205496


namespace graph_translation_up_one_unit_l2054_205454

/-- Represents a vertical translation of a function --/
def verticalTranslation (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := fun x ↦ f x + k

/-- The original quadratic function --/
def originalFunction : ℝ → ℝ := fun x ↦ x^2

theorem graph_translation_up_one_unit :
  verticalTranslation originalFunction 1 = fun x ↦ x^2 + 1 := by
  sorry

end graph_translation_up_one_unit_l2054_205454


namespace valid_combination_exists_l2054_205431

/-- Represents a combination of cards -/
structure CardCombination where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Checks if a card combination is valid according to the given conditions -/
def isValidCombination (c : CardCombination) : Prop :=
  c.red + c.blue + c.green = 20 ∧
  c.red ≥ 2 ∧
  c.blue ≥ 3 ∧
  c.green ≥ 1 ∧
  3 * c.red + 5 * c.blue + 7 * c.green = 84

/-- There exists a valid card combination that satisfies all conditions -/
theorem valid_combination_exists : ∃ c : CardCombination, isValidCombination c := by
  sorry

#check valid_combination_exists

end valid_combination_exists_l2054_205431


namespace distance_AB_is_750_l2054_205405

/-- The distance between two points A and B -/
def distance_AB : ℝ := 750

/-- The speed of person B in meters per minute -/
def speed_B : ℝ := 50

/-- The time it takes for A to catch up with B when moving in the same direction (in minutes) -/
def time_same_direction : ℝ := 30

/-- The time it takes for A and B to meet when moving towards each other (in minutes) -/
def time_towards_each_other : ℝ := 6

/-- The theorem stating that the distance between A and B is 750 meters -/
theorem distance_AB_is_750 : distance_AB = 750 :=
  sorry

end distance_AB_is_750_l2054_205405


namespace hyperbola_parabola_intersection_l2054_205435

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x = y^2 / 10 + 5 / 2

/-- The focus of the parabola -/
def parabola_focus : ℝ × ℝ := (5, 0)

/-- The directrix of the parabola is the x-axis -/
def parabola_directrix (x : ℝ) : ℝ × ℝ := (x, 0)

theorem hyperbola_parabola_intersection :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    hyperbola x₁ y₁ ∧ parabola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧ parabola x₂ y₂ ∧
    (x₁, y₁) ≠ (x₂, y₂) :=
  sorry

end hyperbola_parabola_intersection_l2054_205435


namespace triangle_abc_properties_l2054_205436

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  b * Real.sin (2 * A) = Real.sqrt 3 * a * Real.sin B →
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 →
  b / c = 3 * Real.sqrt 3 / 4 →
  A = π / 6 ∧ a = Real.sqrt 7 := by
  sorry

end triangle_abc_properties_l2054_205436


namespace truncated_pyramid_diagonal_l2054_205446

/-- Regular truncated quadrilateral pyramid -/
structure TruncatedPyramid where
  height : ℝ
  lower_base_side : ℝ
  upper_base_side : ℝ

/-- Diagonal of a truncated pyramid -/
def diagonal (p : TruncatedPyramid) : ℝ :=
  sorry

/-- Theorem: The diagonal of the specified truncated pyramid is 6 -/
theorem truncated_pyramid_diagonal :
  let p : TruncatedPyramid := ⟨2, 5, 3⟩
  diagonal p = 6 := by sorry

end truncated_pyramid_diagonal_l2054_205446


namespace geometric_sequence_first_term_value_l2054_205464

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term_value
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (t : ℝ)
  (h1 : a 1 = t)
  (h2 : geometric_sequence a)
  (h3 : ∀ n : ℕ+, a (n + 1) = 2 * S n + 1)
  : t = 1 := by
  sorry

end geometric_sequence_first_term_value_l2054_205464


namespace solve_baguette_problem_l2054_205469

def baguette_problem (batches_per_day : ℕ) (baguettes_per_batch : ℕ) 
  (sold_after_first : ℕ) (sold_after_second : ℕ) (left_at_end : ℕ) : Prop :=
  let total_baguettes := batches_per_day * baguettes_per_batch
  let sold_first_two := sold_after_first + sold_after_second
  let sold_after_third := total_baguettes - sold_first_two - left_at_end
  sold_after_third = 49

theorem solve_baguette_problem : 
  baguette_problem 3 48 37 52 6 := by sorry

end solve_baguette_problem_l2054_205469


namespace fraction_equality_l2054_205442

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 :=
by
  sorry

end fraction_equality_l2054_205442


namespace polynomial_divisibility_l2054_205478

theorem polynomial_divisibility (A B : ℝ) : 
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^103 + A*x + B = 0) → A + B = -1 := by
  sorry

end polynomial_divisibility_l2054_205478


namespace construct_numbers_l2054_205482

/-- Given a natural number n, construct it using only the number 8, 
    arithmetic operations, and exponentiation -/
def construct_number (n : ℕ) : ℚ :=
  match n with
  | 1 => (8 / 8) ^ (8 / 8) * (8 / 8)
  | 2 => 8 / 8 + 8 / 8
  | 3 => (8 + 8 + 8) / 8
  | 4 => 8 / 8 + 8 / 8 + 8 / 8 + 8 / 8
  | 8 => 8
  | _ => 0  -- Default case, not all numbers are constructed

/-- Theorem stating that we can construct the numbers 1, 2, 3, 4, and 8
    using only the number 8, arithmetic operations, and exponentiation -/
theorem construct_numbers : 
  (construct_number 1 = 1) ∧ 
  (construct_number 2 = 2) ∧ 
  (construct_number 3 = 3) ∧ 
  (construct_number 4 = 4) ∧ 
  (construct_number 8 = 8) := by
  sorry


end construct_numbers_l2054_205482


namespace hannah_stocking_stuffers_l2054_205495

/-- The number of candy canes per stocking -/
def candy_canes : Nat := 4

/-- The number of beanie babies per stocking -/
def beanie_babies : Nat := 2

/-- The number of books per stocking -/
def books : Nat := 1

/-- The number of kids Hannah has -/
def num_kids : Nat := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stuffers : Nat := (candy_canes + beanie_babies + books) * num_kids

theorem hannah_stocking_stuffers :
  total_stuffers = 21 := by sorry

end hannah_stocking_stuffers_l2054_205495


namespace exponent_rule_l2054_205445

theorem exponent_rule (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_rule_l2054_205445


namespace bertha_family_without_children_l2054_205484

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of Bertha's daughters and granddaughters without daughters -/
def daughters_without_children (f : BerthaFamily) : ℕ :=
  f.total_descendants - f.daughters_with_children

/-- Theorem stating the number of Bertha's daughters and granddaughters without daughters -/
theorem bertha_family_without_children (f : BerthaFamily) 
  (h1 : f.daughters = 5)
  (h2 : f.total_descendants = 25)
  (h3 : f.daughters_with_children * 5 = f.total_descendants - f.daughters) :
  daughters_without_children f = 21 := by
  sorry

#check bertha_family_without_children

end bertha_family_without_children_l2054_205484


namespace multiply_by_nine_l2054_205438

theorem multiply_by_nine (A B : ℕ) (h1 : 1 ≤ A ∧ A ≤ 9) (h2 : B ≤ 9) :
  (10 * A + B) * 9 = ((10 * A + B) - (A + 1)) * 10 + (10 - B) := by
  sorry

end multiply_by_nine_l2054_205438


namespace log_equation_solution_l2054_205410

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 2 + Real.log x / Real.log 8 = 5 → x = 2^(15/4) := by
  sorry

end log_equation_solution_l2054_205410


namespace joan_football_games_l2054_205479

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The total number of football games Joan went to -/
def total_games : ℕ := 13

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := total_games - games_this_year

theorem joan_football_games : games_last_year = 9 := by
  sorry

end joan_football_games_l2054_205479


namespace dennis_rocks_l2054_205494

theorem dennis_rocks (initial_rocks : ℕ) : 
  (initial_rocks / 2 + 2 = 7) → initial_rocks = 10 := by
sorry

end dennis_rocks_l2054_205494


namespace trapezoid_area_is_400_l2054_205485

-- Define the trapezoid and square properties
def trapezoid_base1 : ℝ := 50
def trapezoid_base2 : ℝ := 30
def num_trapezoids : ℕ := 4
def outer_square_area : ℝ := 2500

-- Theorem statement
theorem trapezoid_area_is_400 :
  let outer_square_side : ℝ := trapezoid_base1
  let inner_square_side : ℝ := trapezoid_base2
  let inner_square_area : ℝ := inner_square_side ^ 2
  let total_trapezoid_area : ℝ := outer_square_area - inner_square_area
  let single_trapezoid_area : ℝ := total_trapezoid_area / num_trapezoids
  single_trapezoid_area = 400 :=
by sorry

end trapezoid_area_is_400_l2054_205485


namespace gwen_recycled_amount_l2054_205473

-- Define the recycling rate
def recycling_rate : ℕ := 3

-- Define the points earned
def points_earned : ℕ := 6

-- Define the amount recycled by friends
def friends_recycled : ℕ := 13

-- Theorem to prove
theorem gwen_recycled_amount : 
  ∃ (gwen_amount : ℕ), 
    (gwen_amount + friends_recycled) / recycling_rate = points_earned ∧
    gwen_amount = 5 := by
  sorry

end gwen_recycled_amount_l2054_205473


namespace triangle_problem_l2054_205491

open Real

theorem triangle_problem (a b c A B C : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sine_law : a / sin A = b / sin B ∧ b / sin B = c / sin C)
  (h_condition : Real.sqrt 3 * b * sin C = c * cos B + c) : 
  B = π / 3 ∧ 
  (b^2 = a * c → 2 / tan A + 1 / tan C = 2 * Real.sqrt 3 / 3) := by
  sorry

end triangle_problem_l2054_205491


namespace min_multiplications_proof_twelve_numbers_multiplications_l2054_205463

/-- The minimum number of multiplications needed to multiply n numbers -/
def min_multiplications (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

/-- Theorem stating that the minimum number of multiplications for n ≥ 2 numbers is n-1 -/
theorem min_multiplications_proof (n : ℕ) (h : n ≥ 2) :
  min_multiplications n = n - 1 := by
  sorry

/-- Corollary for the specific case of 12 numbers -/
theorem twelve_numbers_multiplications :
  min_multiplications 12 = 11 := by
  sorry

end min_multiplications_proof_twelve_numbers_multiplications_l2054_205463


namespace data_entry_team_size_l2054_205490

theorem data_entry_team_size :
  let rudy_speed := 64
  let joyce_speed := 76
  let gladys_speed := 91
  let lisa_speed := 80
  let mike_speed := 89
  let team_average := 80
  let total_speed := rudy_speed + joyce_speed + gladys_speed + lisa_speed + mike_speed
  (total_speed / team_average : ℚ) = 5 := by sorry

end data_entry_team_size_l2054_205490


namespace point_on_axes_l2054_205423

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The coordinate axes in 2D space -/
def CoordinateAxes : Set Point2D :=
  {p : Point2D | p.x = 0 ∨ p.y = 0}

/-- Theorem: If xy = 0, then P(x,y) is located on the coordinate axes -/
theorem point_on_axes (p : Point2D) (h : p.x * p.y = 0) : p ∈ CoordinateAxes := by
  sorry

end point_on_axes_l2054_205423


namespace problem_statement_l2054_205434

theorem problem_statement (x y : ℝ) (h : (x + y - 2020) * (2023 - x - y) = 2) :
  (x + y - 2020)^2 * (2023 - x - y)^2 = 4 := by
  sorry

end problem_statement_l2054_205434


namespace age_ratio_is_four_thirds_l2054_205439

-- Define the current ages of Arun and Deepak
def arun_current_age : ℕ := 26 - 6
def deepak_current_age : ℕ := 15

-- Define the ratio of their ages
def age_ratio : ℚ := arun_current_age / deepak_current_age

-- Theorem to prove
theorem age_ratio_is_four_thirds : age_ratio = 4/3 := by
  sorry

end age_ratio_is_four_thirds_l2054_205439


namespace division_of_fractions_calculate_fraction_division_l2054_205448

theorem division_of_fractions (a b c : ℚ) (hb : b ≠ 0) :
  a / (c / b) = (a * b) / c :=
by sorry

theorem calculate_fraction_division :
  (4 : ℚ) / (5 / 7) = 28 / 5 :=
by sorry

end division_of_fractions_calculate_fraction_division_l2054_205448


namespace four_propositions_two_correct_l2054_205459

theorem four_propositions_two_correct :
  (∀ A B : Set α, A ∩ B = A → A ⊆ B) ∧
  (∀ a : ℝ, (∃ x y : ℝ, a * x + y + 1 = 0 ∧ x - y + 1 = 0 ∧ (∀ x' y' : ℝ, a * x' + y' + 1 = 0 → x' - y' + 1 = 0 → (x, y) ≠ (x', y'))) → a = 1) ∧
  ¬(∀ p q : Prop, p ∨ q → p ∧ q) ∧
  ¬(∀ a b m : ℝ, a < b → a * m^2 < b * m^2) :=
by sorry

end four_propositions_two_correct_l2054_205459


namespace rectangle_measurement_error_l2054_205455

theorem rectangle_measurement_error (x : ℝ) : 
  (1 + x / 100) * (1 - 4 / 100) = 100.8 / 100 → x = 5 := by
  sorry

end rectangle_measurement_error_l2054_205455


namespace product_from_lcm_and_gcd_l2054_205444

theorem product_from_lcm_and_gcd (a b : ℤ) : 
  lcm a b = 36 → gcd a b = 6 → a * b = 216 := by sorry

end product_from_lcm_and_gcd_l2054_205444


namespace least_value_of_x_l2054_205424

theorem least_value_of_x (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) 
  (h3 : ∃ (k : ℕ), k > 0 ∧ x = 11 * p * k ∧ Nat.Prime k ∧ Even k) : 
  x ≥ 44 ∧ ∃ (x₀ : ℕ), x₀ ≥ 44 ∧ 
    (∃ (p₀ : ℕ), Nat.Prime p₀ ∧ ∃ (k₀ : ℕ), k₀ > 0 ∧ x₀ = 11 * p₀ * k₀ ∧ Nat.Prime k₀ ∧ Even k₀) :=
by sorry

end least_value_of_x_l2054_205424


namespace brothers_selection_probability_l2054_205419

theorem brothers_selection_probability
  (prob_X_initial : ℚ) (prob_Y_initial : ℚ)
  (prob_X_interview : ℚ) (prob_X_test : ℚ)
  (prob_Y_interview : ℚ) (prob_Y_test : ℚ)
  (h1 : prob_X_initial = 1 / 7)
  (h2 : prob_Y_initial = 2 / 5)
  (h3 : prob_X_interview = 3 / 4)
  (h4 : prob_X_test = 4 / 9)
  (h5 : prob_Y_interview = 5 / 8)
  (h6 : prob_Y_test = 7 / 10) :
  prob_X_initial * prob_X_interview * prob_X_test *
  prob_Y_initial * prob_Y_interview * prob_Y_test = 7 / 840 := by
  sorry

end brothers_selection_probability_l2054_205419


namespace max_original_points_l2054_205420

/-- Represents a rectangular matrix of points on a grid -/
structure RectMatrix where
  rows : ℕ
  cols : ℕ

/-- The maximum grid size -/
def maxGridSize : ℕ := 19

/-- The number of additional points -/
def additionalPoints : ℕ := 45

/-- Checks if a rectangular matrix fits within the maximum grid size -/
def fitsInGrid (rect : RectMatrix) : Prop :=
  rect.rows ≤ maxGridSize ∧ rect.cols ≤ maxGridSize

/-- Checks if a rectangular matrix can be expanded by adding the additional points -/
def canBeExpanded (small rect : RectMatrix) : Prop :=
  (rect.rows - small.rows) * (rect.cols - small.cols) = additionalPoints

/-- The theorem stating the maximum number of points in the original matrix -/
theorem max_original_points : 
  ∃ (small large : RectMatrix), 
    fitsInGrid small ∧ 
    fitsInGrid large ∧
    canBeExpanded small large ∧
    (small.rows = large.rows ∨ small.cols = large.cols) ∧
    small.rows * small.cols = 285 ∧
    ∀ (other : RectMatrix), 
      fitsInGrid other → 
      (∃ (expanded : RectMatrix), 
        fitsInGrid expanded ∧ 
        canBeExpanded other expanded ∧ 
        (other.rows = expanded.rows ∨ other.cols = expanded.cols)) →
      other.rows * other.cols ≤ 285 :=
sorry

end max_original_points_l2054_205420


namespace unique_common_term_l2054_205417

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem unique_common_term : ∀ n : ℕ, x n = y n → x n = 1 := by
  sorry

end unique_common_term_l2054_205417


namespace probability_ace_of_hearts_fifth_l2054_205465

-- Define the number of cards in a standard deck
def standard_deck_size : ℕ := 52

-- Define a function to calculate the probability of a specific card in a specific position
def probability_specific_card_in_position (deck_size : ℕ) : ℚ :=
  1 / deck_size

-- Theorem statement
theorem probability_ace_of_hearts_fifth : 
  probability_specific_card_in_position standard_deck_size = 1 / 52 := by
  sorry

end probability_ace_of_hearts_fifth_l2054_205465


namespace solution_306_is_valid_l2054_205492

def is_valid_solution (a b c : Nat) : Prop :=
  a ≠ 0 ∧ 
  b = 0 ∧ 
  c ≠ 0 ∧ 
  a ≠ c ∧
  1995 * (a * 100 + c) = 1995 * a * 100 + 1995 * c

theorem solution_306_is_valid : is_valid_solution 3 0 6 := by
  sorry

#check solution_306_is_valid

end solution_306_is_valid_l2054_205492


namespace cans_collected_l2054_205403

/-- The total number of cans collected by six people -/
def total_cans (solomon juwan levi gaby michelle sarah : ℕ) : ℕ :=
  solomon + juwan + levi + gaby + michelle + sarah

/-- Theorem stating the total number of cans collected by six people -/
theorem cans_collected :
  ∀ (solomon juwan levi gaby michelle sarah : ℕ),
    solomon = 66 →
    solomon = 3 * juwan →
    levi = juwan / 2 →
    gaby = (5 * solomon) / 2 →
    michelle = gaby / 3 →
    sarah = gaby - levi - 6 →
    total_cans solomon juwan levi gaby michelle sarah = 467 := by
  sorry

end cans_collected_l2054_205403


namespace ceiling_neg_sqrt_64_over_9_l2054_205449

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end ceiling_neg_sqrt_64_over_9_l2054_205449


namespace diophantine_equation_solutions_composite_sum_l2054_205450

theorem diophantine_equation_solutions :
  {(m, n) : ℕ × ℕ | 5 * m + 8 * n = 120} = {(24, 0), (16, 5), (8, 10), (0, 15)} := by sorry

theorem composite_sum :
  ∀ (a b c : ℕ+), c > 1 → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c →
  (∃ d : ℕ, 1 < d ∧ d < a + c ∧ (a + c) % d = 0) ∨
  (∃ d : ℕ, 1 < d ∧ d < b + c ∧ (b + c) % d = 0) := by sorry

end diophantine_equation_solutions_composite_sum_l2054_205450


namespace sequence_property_l2054_205443

/-- The sum of the first n terms of the sequence {a_n} -/
def S (a : ℕ+ → ℕ) (n : ℕ+) : ℕ := (Finset.range n.val).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem stating that if S_n = 2a_n - 2 for all n, then a_n = 2^n for all n -/
theorem sequence_property (a : ℕ+ → ℕ) 
    (h : ∀ n : ℕ+, S a n = 2 * a n - 2) : 
    ∀ n : ℕ+, a n = 2^n.val := by
  sorry

end sequence_property_l2054_205443


namespace inequality_solution_and_abs_inequality_l2054_205416

def f (x : ℝ) := |x - 1|

theorem inequality_solution_and_abs_inequality (a b : ℝ) :
  (∀ x, f x + f (x + 4) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3) ∧
  (|a| < 1 → |b| < 1 → a ≠ 0 → f (a * b) > |a| * f (b / a)) := by
  sorry

end inequality_solution_and_abs_inequality_l2054_205416


namespace halfway_between_one_sixth_and_one_fourth_l2054_205426

theorem halfway_between_one_sixth_and_one_fourth : 
  (1/6 : ℚ) / 2 + (1/4 : ℚ) / 2 = 5/24 := by sorry

end halfway_between_one_sixth_and_one_fourth_l2054_205426


namespace cyclist_return_speed_l2054_205474

/-- Proves that given the conditions of the cyclist's trip, the average speed for the return trip is 9 miles per hour. -/
theorem cyclist_return_speed (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) : 
  total_distance = 36 →
  speed1 = 12 →
  speed2 = 10 →
  total_time = 7.3 →
  (total_distance / speed1 + total_distance / speed2 + total_distance / 9 = total_time) := by
sorry

end cyclist_return_speed_l2054_205474


namespace salary_restoration_l2054_205402

theorem salary_restoration (original_salary : ℝ) (reduced_salary : ℝ) : 
  reduced_salary = original_salary * (1 - 0.5) →
  reduced_salary * 2 = original_salary :=
by
  sorry

end salary_restoration_l2054_205402


namespace unique_invariant_quadratic_l2054_205421

/-- A quadratic equation that remains unchanged when its roots are used as coefficients. -/
def InvariantQuadratic (p q : ℤ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧
  ∃ (x y : ℝ),
    x^2 + p*x + q = 0 ∧
    y^2 + p*y + q = 0 ∧
    x ≠ y ∧
    x^2 + y*x + (x*y) = 0 ∧
    p = -(x + y) ∧
    q = x * y

theorem unique_invariant_quadratic :
  ∀ (p q : ℤ), InvariantQuadratic p q → p = 1 ∧ q = -2 :=
sorry

end unique_invariant_quadratic_l2054_205421


namespace first_marvelous_monday_l2054_205458

/-- Represents a date with a year, month, and day. -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a day of the week. -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns true if the given date is a Monday. -/
def isMonday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Prop :=
  sorry

/-- Returns true if the given date is the fifth Monday of its month. -/
def isFifthMonday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Prop :=
  sorry

/-- Returns true if date d1 is strictly after date d2. -/
def isAfter (d1 d2 : Date) : Prop :=
  sorry

theorem first_marvelous_monday 
  (schoolStartDate : Date)
  (h1 : schoolStartDate.year = 2023)
  (h2 : schoolStartDate.month = 9)
  (h3 : schoolStartDate.day = 11)
  (h4 : isMonday schoolStartDate schoolStartDate DayOfWeek.Monday) :
  ∃ (marvelousMonday : Date), 
    marvelousMonday.year = 2023 ∧ 
    marvelousMonday.month = 10 ∧ 
    marvelousMonday.day = 30 ∧
    isFifthMonday marvelousMonday schoolStartDate DayOfWeek.Monday ∧
    isAfter marvelousMonday schoolStartDate ∧
    ∀ (d : Date), 
      isFifthMonday d schoolStartDate DayOfWeek.Monday → 
      isAfter d schoolStartDate → 
      ¬(isAfter d marvelousMonday) :=
by sorry

end first_marvelous_monday_l2054_205458


namespace indeterminate_divisor_l2054_205452

theorem indeterminate_divisor (x y : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) →
  (∃ m : ℤ, x + 7 = y * m + 12) →
  ¬ (∃! y : ℤ, ∃ m : ℤ, x + 7 = y * m + 12) :=
sorry

end indeterminate_divisor_l2054_205452


namespace building_floors_l2054_205406

theorem building_floors (total_height : ℝ) (regular_floor_height : ℝ) (extra_height : ℝ) :
  total_height = 61 ∧
  regular_floor_height = 3 ∧
  extra_height = 0.5 →
  ∃ (n : ℕ), n = 20 ∧
    total_height = regular_floor_height * (n - 2 : ℝ) + (regular_floor_height + extra_height) * 2 :=
by
  sorry


end building_floors_l2054_205406


namespace anthony_pencils_count_l2054_205418

/-- Given Anthony's initial pencils and Kathryn's gift, calculate Anthony's total pencils -/
def anthonyTotalPencils (initialPencils giftedPencils : ℕ) : ℕ :=
  initialPencils + giftedPencils

/-- Theorem: Anthony's total pencils is 65 given the initial conditions -/
theorem anthony_pencils_count :
  anthonyTotalPencils 9 56 = 65 := by
  sorry

end anthony_pencils_count_l2054_205418


namespace cubic_root_inequality_l2054_205475

theorem cubic_root_inequality (p q x : ℝ) : x^3 + p*x + q = 0 → 4*q*x ≤ p^2 := by
  sorry

end cubic_root_inequality_l2054_205475


namespace pythagorean_triple_product_divisible_by_six_l2054_205493

theorem pythagorean_triple_product_divisible_by_six (A B C : ℤ) : 
  A^2 + B^2 = C^2 → (6 ∣ A * B) := by
sorry

end pythagorean_triple_product_divisible_by_six_l2054_205493


namespace remainder_of_P_div_Q_l2054_205447

/-- P(x) is a polynomial defined as x^(6n) + x^(5n) + x^(4n) + x^(3n) + x^(2n) + x^n + 1 -/
def P (x n : ℕ) : ℕ := x^(6*n) + x^(5*n) + x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1

/-- Q(x) is a polynomial defined as x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 -/
def Q (x : ℕ) : ℕ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

/-- Theorem stating that the remainder of P(x) divided by Q(x) is 7 when n is a multiple of 7 -/
theorem remainder_of_P_div_Q (x n : ℕ) (h : ∃ k, n = 7 * k) :
  P x n % Q x = 7 := by sorry

end remainder_of_P_div_Q_l2054_205447


namespace students_to_add_l2054_205471

theorem students_to_add (current_students : ℕ) (teachers : ℕ) (h1 : current_students = 1049) (h2 : teachers = 9) :
  ∃ (students_to_add : ℕ), 
    students_to_add = 4 ∧
    (current_students + students_to_add) % teachers = 0 ∧
    ∀ (n : ℕ), n < students_to_add → (current_students + n) % teachers ≠ 0 :=
by
  sorry

end students_to_add_l2054_205471


namespace tg_ctg_equation_solution_l2054_205422

theorem tg_ctg_equation_solution (x : ℝ) :
  (∀ n : ℤ, x ≠ (n : ℝ) * π / 2) →
  (Real.tan x ^ 4 + (1 / Real.tan x) ^ 4 = (82 / 9) * (Real.tan x * Real.tan (2 * x) + 1) * Real.cos (2 * x)) ↔
  ∃ k : ℤ, x = π / 6 * ((3 * k : ℝ) + 1) ∨ x = π / 6 * ((3 * k : ℝ) - 1) :=
by sorry

end tg_ctg_equation_solution_l2054_205422


namespace apple_water_bottle_difference_l2054_205457

theorem apple_water_bottle_difference (total_bottles : ℕ) (water_bottles : ℕ) (apple_bottles : ℕ) : 
  total_bottles = 54 →
  water_bottles = 2 * 12 →
  apple_bottles = total_bottles - water_bottles →
  apple_bottles - water_bottles = 6 := by
  sorry

end apple_water_bottle_difference_l2054_205457


namespace total_worksheets_l2054_205415

theorem total_worksheets (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) (problems_left : ℕ) :
  problems_per_worksheet = 4 →
  graded_worksheets = 8 →
  problems_left = 32 →
  graded_worksheets + (problems_left / problems_per_worksheet) = 16 :=
by sorry

end total_worksheets_l2054_205415


namespace perry_vs_phil_l2054_205412

-- Define the number of games won by each player
def phil_games : ℕ := 12
def charlie_games : ℕ := phil_games - 3
def dana_games : ℕ := charlie_games + 2
def perry_games : ℕ := dana_games + 5

-- Theorem statement
theorem perry_vs_phil : perry_games = phil_games + 4 := by
  sorry

end perry_vs_phil_l2054_205412


namespace parallelogram_properties_l2054_205407

-- Define the points
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Define the vector operation
def vectorOp : ℝ × ℝ := (3 * AB.1 - 2 * AC.1 + BC.1, 3 * AB.2 - 2 * AC.2 + BC.2)

-- Define point D
def D : ℝ × ℝ := (A.1 + BC.1, A.2 + BC.2)

theorem parallelogram_properties :
  vectorOp = (0, 2) ∧ D = (2, -1) := by
  sorry


end parallelogram_properties_l2054_205407


namespace a_gt_b_gt_c_l2054_205483

theorem a_gt_b_gt_c : 3^44 > 4^33 ∧ 4^33 > 5^22 := by
  sorry

end a_gt_b_gt_c_l2054_205483


namespace square_divisibility_l2054_205441

theorem square_divisibility (a b : ℕ+) (h : (a * b + 1) ∣ (a ^ 2 + b ^ 2)) :
  ∃ k : ℕ, (a ^ 2 + b ^ 2) / (a * b + 1) = k ^ 2 := by
  sorry

end square_divisibility_l2054_205441


namespace inequality_proof_l2054_205470

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / Real.sqrt (x * y) ≤ x / y + y / x :=
by sorry

end inequality_proof_l2054_205470


namespace equation_is_quadratic_l2054_205409

/-- A quadratic equation is of the form ax^2 + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 = 3x - 2 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem equation_is_quadratic : is_quadratic_equation f := by
  sorry

end equation_is_quadratic_l2054_205409


namespace mixed_numbers_sum_range_l2054_205486

theorem mixed_numbers_sum_range : 
  let a : ℚ := 3 + 1 / 9
  let b : ℚ := 4 + 1 / 3
  let c : ℚ := 6 + 1 / 21
  let sum : ℚ := a + b + c
  13.5 < sum ∧ sum < 14 := by
sorry

end mixed_numbers_sum_range_l2054_205486


namespace inequality_and_maximum_l2054_205408

theorem inequality_and_maximum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 3) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 3) ∧
  (c = a * b → ∀ c', c' = a * b → c' ≤ 1) := by
  sorry

end inequality_and_maximum_l2054_205408


namespace complete_collection_probability_l2054_205440

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def uncollected_stickers : ℕ := 6
def collected_stickers : ℕ := 12

theorem complete_collection_probability :
  (Nat.choose uncollected_stickers uncollected_stickers * Nat.choose collected_stickers (selected_stickers - uncollected_stickers)) /
  (Nat.choose total_stickers selected_stickers) = 5 / 442 := by
  sorry

end complete_collection_probability_l2054_205440


namespace big_eighteen_soccer_league_games_l2054_205497

/-- Calculates the number of games in a soccer league with specific rules --/
def soccer_league_games (num_divisions : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let intra_games := num_divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_games := num_divisions * teams_per_division * (num_divisions - 1) * teams_per_division * inter_division_games
  (intra_games + inter_games) / 2

/-- The Big Eighteen Soccer League schedule theorem --/
theorem big_eighteen_soccer_league_games : 
  soccer_league_games 3 6 3 2 = 351 := by
  sorry

end big_eighteen_soccer_league_games_l2054_205497


namespace max_value_fraction_l2054_205432

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^2 / (x^2 + y^2 + x*y) ≤ 4/3 := by
  sorry

end max_value_fraction_l2054_205432


namespace point_on_x_axis_l2054_205460

/-- If a point P(m, m-3) lies on the x-axis, then its coordinates are (3, 0) -/
theorem point_on_x_axis (m : ℝ) :
  (m : ℝ) = m ∧ (m - 3 : ℝ) = 0 → (m : ℝ) = 3 ∧ (m - 3 : ℝ) = 0 :=
by sorry

end point_on_x_axis_l2054_205460


namespace point_B_coordinates_l2054_205472

/-- Given that point A(m+2, m) lies on the y-axis, prove that point B(m+5, m-1) has coordinates (3, -3) -/
theorem point_B_coordinates (m : ℝ) 
  (h_A_on_y_axis : m + 2 = 0) : 
  (m + 5, m - 1) = (3, -3) := by
  sorry

end point_B_coordinates_l2054_205472


namespace complex_magnitude_l2054_205467

theorem complex_magnitude (a b : ℝ) (h : (1 + 2*a*Complex.I) * Complex.I = 1 - b*Complex.I) :
  Complex.abs (a + b*Complex.I) = Real.sqrt 5 / 2 := by
  sorry

end complex_magnitude_l2054_205467


namespace fifth_term_of_arithmetic_sequence_l2054_205487

/-- Given an arithmetic sequence of 20 terms with first term 2 and last term 59,
    prove that the 5th term is 14. -/
theorem fifth_term_of_arithmetic_sequence :
  ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
  a 0 = 2 →                            -- first term is 2
  a 19 = 59 →                          -- last term (20th term) is 59
  a 4 = 14 :=                          -- 5th term (index 4) is 14
by
  sorry

end fifth_term_of_arithmetic_sequence_l2054_205487


namespace max_basketballs_proof_l2054_205488

/-- The maximum number of basketballs that can be purchased given the constraints -/
def max_basketballs : ℕ := 26

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 40

/-- The cost of each basketball in dollars -/
def basketball_cost : ℕ := 80

/-- The cost of each soccer ball in dollars -/
def soccer_ball_cost : ℕ := 50

/-- The total budget in dollars -/
def total_budget : ℕ := 2800

theorem max_basketballs_proof :
  (∀ x : ℕ, 
    x ≤ total_balls ∧ 
    (basketball_cost * x + soccer_ball_cost * (total_balls - x) ≤ total_budget) →
    x ≤ max_basketballs) ∧
  (basketball_cost * max_basketballs + soccer_ball_cost * (total_balls - max_basketballs) ≤ total_budget) :=
sorry

end max_basketballs_proof_l2054_205488


namespace linear_equation_condition_l2054_205462

theorem linear_equation_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ k b : ℝ, (a - 6) * x - y^(a - 6) = k * y + b) → a = 7 :=
by sorry

end linear_equation_condition_l2054_205462


namespace probability_theorem_l2054_205427

-- Define the total number of students
def total_students : ℕ := 20

-- Define the fraction of students interested in the career
def interested_fraction : ℚ := 4 / 5

-- Define the number of interested students
def interested_students : ℕ := (interested_fraction * total_students).num.toNat

-- Define the function to calculate the probability
def probability_at_least_one_interested : ℚ :=
  1 - (total_students - interested_students) * (total_students - interested_students - 1) /
      (total_students * (total_students - 1))

-- Theorem statement
theorem probability_theorem :
  probability_at_least_one_interested = 92 / 95 :=
sorry

end probability_theorem_l2054_205427


namespace fraction_subtraction_theorem_l2054_205499

theorem fraction_subtraction_theorem : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end fraction_subtraction_theorem_l2054_205499


namespace students_liking_both_subjects_l2054_205477

theorem students_liking_both_subjects 
  (total_students : ℕ) 
  (art_students : ℕ) 
  (science_students : ℕ) 
  (h1 : total_students = 45)
  (h2 : art_students = 42)
  (h3 : science_students = 40)
  (h4 : art_students ≤ total_students)
  (h5 : science_students ≤ total_students) :
  art_students + science_students - total_students = 37 := by
sorry

end students_liking_both_subjects_l2054_205477


namespace combine_like_terms_l2054_205456

theorem combine_like_terms (x y : ℝ) : 3 * x * y - 6 * x * y + (-2 * x * y) = -5 * x * y := by
  sorry

end combine_like_terms_l2054_205456


namespace equation_solution_l2054_205476

theorem equation_solution : 
  ∃! n : ℚ, (1 : ℚ) / (n + 2) + (3 : ℚ) / (n + 2) + n / (n + 2) = 4 ∧ n = -4/3 := by
  sorry

end equation_solution_l2054_205476


namespace major_axis_length_eccentricity_l2054_205480

/-- Definition of the ellipse E -/
def ellipse_E (x y : ℝ) : Prop := y^2 / 4 + x^2 / 3 = 1

/-- F₁ and F₂ are the foci of the ellipse E -/
axiom foci_on_ellipse : ∃ F₁ F₂ : ℝ × ℝ, ellipse_E F₁.1 F₁.2 ∧ ellipse_E F₂.1 F₂.2

/-- Point P lies on the ellipse E -/
axiom P_on_ellipse : ∃ P : ℝ × ℝ, ellipse_E P.1 P.2

/-- The length of the major axis of ellipse E is 4 -/
theorem major_axis_length : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  (∀ x y : ℝ, ellipse_E x y ↔ (x/a)^2 + (y/b)^2 = 1) ∧ 
  max a b = 2 :=
sorry

/-- The eccentricity of ellipse E is 1/2 -/
theorem eccentricity : ∃ e : ℝ, e = 1/2 ∧
  ∃ c a : ℝ, c^2 = 4 - 3 ∧ a^2 = 3 ∧ e = c/a :=
sorry

end major_axis_length_eccentricity_l2054_205480


namespace parallelogram_slant_height_l2054_205404

/-- Given a rectangle and a shape composed of an isosceles triangle and a parallelogram,
    prove that the slant height of the parallelogram is approximately 8.969 inches
    when the areas are equal. -/
theorem parallelogram_slant_height (rectangle_length rectangle_width triangle_base triangle_height parallelogram_base parallelogram_height : ℝ) 
  (h_rectangle_length : rectangle_length = 5)
  (h_rectangle_width : rectangle_width = 24)
  (h_triangle_base : triangle_base = 12)
  (h_parallelogram_base : parallelogram_base = 12)
  (h_equal_heights : triangle_height = parallelogram_height)
  (h_equal_areas : rectangle_length * rectangle_width = 
    (1/2 * triangle_base * triangle_height) + (parallelogram_base * parallelogram_height)) :
  ∃ (slant_height : ℝ), abs (slant_height - 8.969) < 0.001 ∧ 
    slant_height^2 = parallelogram_height^2 + (parallelogram_base/2)^2 :=
by sorry

end parallelogram_slant_height_l2054_205404


namespace problem_solution_l2054_205429

theorem problem_solution (x y z a b c : ℝ) 
  (h1 : x * y = 2 * a) 
  (h2 : x * z = 3 * b) 
  (h3 : y * z = 4 * c) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^2 + y^2 + z^2 = (3*a*b)/(2*c) + (8*a*c)/(3*b) + (6*b*c)/a ∧ 
  x*y*z = 2 * Real.sqrt (6*a*b*c) := by
  sorry

end problem_solution_l2054_205429


namespace initial_apples_count_l2054_205489

theorem initial_apples_count (initial_apples : ℕ) : 
  initial_apples - 2 + (8 - 2 * 2) + (15 - (2 / 3 * 15)) = 14 → 
  initial_apples = 7 := by
sorry

end initial_apples_count_l2054_205489


namespace game_probability_specific_case_l2054_205401

def game_probability (total_rounds : ℕ) 
  (alex_prob : ℚ) (chelsea_prob : ℚ) (mel_prob : ℚ)
  (alex_wins : ℕ) (chelsea_wins : ℕ) (mel_wins : ℕ) : ℚ :=
  (alex_prob ^ alex_wins) * 
  (chelsea_prob ^ chelsea_wins) * 
  (mel_prob ^ mel_wins) * 
  (Nat.choose total_rounds alex_wins).choose chelsea_wins

theorem game_probability_specific_case : 
  game_probability 8 (5/12) (1/3) (1/4) 3 4 1 = 625/9994 := by
  sorry

end game_probability_specific_case_l2054_205401


namespace fibonacci_triangle_isosceles_l2054_205433

def fibonacci_set : Set ℕ := {2, 3, 5, 8, 13, 21, 34, 55, 89, 144}

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem fibonacci_triangle_isosceles :
  ∀ a b c : ℕ,
    a ∈ fibonacci_set →
    b ∈ fibonacci_set →
    c ∈ fibonacci_set →
    is_triangle a b c →
    is_isosceles a b c :=
by sorry

end fibonacci_triangle_isosceles_l2054_205433


namespace initial_milk_water_ratio_l2054_205498

/-- Proves that the initial ratio of milk to water in a mixture is 3:2, given specific conditions -/
theorem initial_milk_water_ratio
  (total_initial : ℝ)
  (water_added : ℝ)
  (milk : ℝ)
  (water : ℝ)
  (h1 : total_initial = 165)
  (h2 : water_added = 66)
  (h3 : milk + water = total_initial)
  (h4 : milk / (water + water_added) = 3 / 4)
  : milk / water = 3 / 2 := by
  sorry

end initial_milk_water_ratio_l2054_205498


namespace sum_remainder_l2054_205451

theorem sum_remainder (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
sorry

end sum_remainder_l2054_205451


namespace integer_solutions_of_system_l2054_205425

theorem integer_solutions_of_system : 
  ∀ x y z t : ℤ, 
    (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
    ((x, y, z, t) = (1, 0, 3, 1) ∨ 
     (x, y, z, t) = (-1, 0, -3, -1) ∨ 
     (x, y, z, t) = (3, 1, 1, 0) ∨ 
     (x, y, z, t) = (-3, -1, -1, 0)) :=
by sorry

end integer_solutions_of_system_l2054_205425


namespace fraction_problem_l2054_205468

theorem fraction_problem (x : ℚ) : x = 3/5 ↔ (2/5 * 300 : ℚ) - (x * 125) = 45 := by
  sorry

end fraction_problem_l2054_205468


namespace power_tower_mod_1000_l2054_205413

theorem power_tower_mod_1000 : 5^(5^(5^5)) ≡ 125 [ZMOD 1000] := by
  sorry

end power_tower_mod_1000_l2054_205413


namespace existence_of_n_l2054_205437

theorem existence_of_n (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h_cd : c * d = 1) :
  ∃ n : ℤ, (a * b : ℝ) ≤ (n : ℝ)^2 ∧ (n : ℝ)^2 ≤ (a + c) * (b + d) := by
  sorry

end existence_of_n_l2054_205437


namespace roberts_expenses_l2054_205481

theorem roberts_expenses (total : ℝ) (machinery : ℝ) (cash_percentage : ℝ) 
  (h1 : total = 250)
  (h2 : machinery = 125)
  (h3 : cash_percentage = 0.1)
  : total - machinery - (cash_percentage * total) = 100 := by
  sorry

end roberts_expenses_l2054_205481


namespace amalie_coin_spending_l2054_205466

/-- Proof that Amalie spends 3/4 of her coins on toys -/
theorem amalie_coin_spending :
  ∀ (elsa_coins amalie_coins : ℕ),
    -- The ratio of Elsa's coins to Amalie's coins is 10:45
    elsa_coins * 45 = amalie_coins * 10 →
    -- The total number of coins they have is 440
    elsa_coins + amalie_coins = 440 →
    -- Amalie remains with 90 coins after spending
    ∃ (spent_coins : ℕ),
      spent_coins ≤ amalie_coins ∧
      amalie_coins - spent_coins = 90 →
    -- The fraction of coins Amalie spends on toys is 3/4
    (spent_coins : ℚ) / amalie_coins = 3 / 4 :=
by
  sorry

end amalie_coin_spending_l2054_205466


namespace emily_fishing_total_weight_l2054_205411

theorem emily_fishing_total_weight :
  let trout_count : ℕ := 4
  let catfish_count : ℕ := 3
  let bluegill_count : ℕ := 5
  let trout_weight : ℚ := 2
  let catfish_weight : ℚ := 1.5
  let bluegill_weight : ℚ := 2.5
  let total_weight : ℚ := trout_count * trout_weight + catfish_count * catfish_weight + bluegill_count * bluegill_weight
  total_weight = 25 := by sorry

end emily_fishing_total_weight_l2054_205411


namespace final_short_bushes_count_l2054_205461

/-- The number of short bushes in the park -/
def initial_short_bushes : ℕ := 37

/-- The number of tall trees in the park -/
def tall_trees : ℕ := 30

/-- The number of short bushes to be planted -/
def new_short_bushes : ℕ := 20

/-- The total number of short bushes after planting -/
def total_short_bushes : ℕ := initial_short_bushes + new_short_bushes

theorem final_short_bushes_count : total_short_bushes = 57 := by
  sorry

end final_short_bushes_count_l2054_205461


namespace congruence_problem_l2054_205400

theorem congruence_problem (x : ℤ) : 
  (3 * x + 7) % 16 = 5 → (4 * x + 3) % 16 = 11 := by
  sorry

end congruence_problem_l2054_205400


namespace pet_shop_solution_l2054_205428

/-- Represents the pet shop inventory --/
structure PetShop where
  kittens : ℕ
  hamsters : ℕ
  birds : ℕ
  puppies : ℕ

/-- The initial state of the pet shop --/
def initial_state : PetShop :=
  { kittens := 45,
    hamsters := 30,
    birds := 60,
    puppies := 15 }

/-- The final state of the pet shop after changes --/
def final_state : PetShop :=
  { kittens := initial_state.kittens,
    hamsters := initial_state.hamsters,
    birds := initial_state.birds + 10,
    puppies := initial_state.puppies - 5 }

/-- Theorem stating the correctness of the solution --/
theorem pet_shop_solution :
  (initial_state.kittens + initial_state.hamsters + initial_state.birds + initial_state.puppies = 150) ∧
  (3 * initial_state.hamsters = 2 * initial_state.kittens) ∧
  (initial_state.birds = initial_state.hamsters + 30) ∧
  (4 * initial_state.puppies = initial_state.birds) ∧
  (final_state.kittens + final_state.hamsters + final_state.birds + final_state.puppies = 155) ∧
  (final_state.kittens = 45) ∧
  (final_state.hamsters = 30) ∧
  (final_state.birds = 70) ∧
  (final_state.puppies = 10) := by
  sorry


end pet_shop_solution_l2054_205428


namespace prime_square_remainders_mod_180_l2054_205453

theorem prime_square_remainders_mod_180 :
  ∃! (s : Finset Nat), 
    (∀ r ∈ s, r < 180) ∧ 
    (∀ p : Nat, Prime p → p > 5 → (p^2 % 180) ∈ s) ∧ 
    s.card = 2 := by
  sorry

end prime_square_remainders_mod_180_l2054_205453
