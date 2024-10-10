import Mathlib

namespace cosine_amplitude_l1174_117407

theorem cosine_amplitude (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∀ x, a * Real.cos (b * x - c) ≤ 3) ∧
  (∃ x, a * Real.cos (b * x - c) = 3) ∧
  (∀ x, a * Real.cos (b * x - c) = a * Real.cos (b * (x + 2 * Real.pi) - c)) →
  a = 3 :=
sorry

end cosine_amplitude_l1174_117407


namespace charles_initial_skittles_l1174_117442

theorem charles_initial_skittles (current_skittles taken_skittles : ℕ) 
  (h1 : current_skittles = 18)
  (h2 : taken_skittles = 7) :
  current_skittles + taken_skittles = 25 := by
  sorry

end charles_initial_skittles_l1174_117442


namespace power_zero_l1174_117479

theorem power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end power_zero_l1174_117479


namespace revenue_difference_l1174_117457

/-- Mr. Banks' number of investments -/
def banks_investments : ℕ := 8

/-- Revenue from each of Mr. Banks' investments -/
def banks_revenue_per_investment : ℕ := 500

/-- Ms. Elizabeth's number of investments -/
def elizabeth_investments : ℕ := 5

/-- Revenue from each of Ms. Elizabeth's investments -/
def elizabeth_revenue_per_investment : ℕ := 900

/-- The difference in total revenue between Ms. Elizabeth and Mr. Banks -/
theorem revenue_difference : 
  elizabeth_investments * elizabeth_revenue_per_investment - 
  banks_investments * banks_revenue_per_investment = 500 := by
  sorry

end revenue_difference_l1174_117457


namespace fourth_group_number_l1174_117471

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (secondNumber : ℕ) : ℕ → ℕ :=
  fun groupIndex => secondNumber + (groupIndex - 2) * (totalStudents / sampleSize)

theorem fourth_group_number
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (secondNumber : ℕ)
  (h1 : totalStudents = 60)
  (h2 : sampleSize = 5)
  (h3 : secondNumber = 16) :
  systematicSample totalStudents sampleSize secondNumber 4 = 40 := by
  sorry

end fourth_group_number_l1174_117471


namespace condition_necessary_not_sufficient_l1174_117462

theorem condition_necessary_not_sufficient (b : ℝ) (hb : b ≠ 0) :
  (∃ a : ℝ, a > b ∧ Real.log (a - b) ≤ 0) ∧
  (∀ a : ℝ, Real.log (a - b) > 0 → a > b) :=
sorry

end condition_necessary_not_sufficient_l1174_117462


namespace bch_unique_product_l1174_117421

/-- Represents a letter of the alphabet -/
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J | K | L | M
| N | O | P | Q | R | S | T | U | V | W | X | Y | Z

/-- Assigns a numerical value to each letter -/
def letterValue : Letter → Nat
| Letter.A => 1
| Letter.B => 2
| Letter.C => 3
| Letter.D => 4
| Letter.E => 5
| Letter.F => 6
| Letter.G => 7
| Letter.H => 8
| Letter.I => 9
| Letter.J => 10
| Letter.K => 11
| Letter.L => 12
| Letter.M => 13
| Letter.N => 14
| Letter.O => 15
| Letter.P => 16
| Letter.Q => 17
| Letter.R => 18
| Letter.S => 19
| Letter.T => 20
| Letter.U => 21
| Letter.V => 22
| Letter.W => 23
| Letter.X => 24
| Letter.Y => 25
| Letter.Z => 26

/-- Calculates the product of a three-letter list -/
def productOfList (a b c : Letter) : Nat :=
  letterValue a * letterValue b * letterValue c

/-- Checks if three letters are in alphabetical order -/
def isAlphabeticalOrder (a b c : Letter) : Prop :=
  letterValue a ≤ letterValue b ∧ letterValue b ≤ letterValue c

/-- Main theorem: BCH is the only other three-letter list with product equal to BDF -/
theorem bch_unique_product :
  ∀ (x y z : Letter),
    x ≠ y ∧ y ≠ z ∧ x ≠ z →
    isAlphabeticalOrder x y z →
    productOfList x y z = productOfList Letter.B Letter.D Letter.F →
    x = Letter.B ∧ y = Letter.C ∧ z = Letter.H :=
by sorry


end bch_unique_product_l1174_117421


namespace target_walmart_knife_ratio_l1174_117467

/-- Represents the number of tools in a multitool -/
structure Multitool where
  screwdrivers : ℕ
  knives : ℕ
  other_tools : ℕ

/-- The Walmart multitool -/
def walmart : Multitool :=
  { screwdrivers := 1
    knives := 3
    other_tools := 2 }

/-- The Target multitool -/
def target (k : ℕ) : Multitool :=
  { screwdrivers := 1
    knives := k
    other_tools := 4 }  -- 3 files + 1 pair of scissors

/-- Total number of tools in a multitool -/
def total_tools (m : Multitool) : ℕ :=
  m.screwdrivers + m.knives + m.other_tools

/-- The theorem to prove -/
theorem target_walmart_knife_ratio :
    ∃ k : ℕ, 
      total_tools (target k) = total_tools walmart + 5 ∧ 
      k = 2 * walmart.knives := by
  sorry


end target_walmart_knife_ratio_l1174_117467


namespace inequality_proof_l1174_117401

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (1+a)/(1-a) + (1+b)/(1-b) + (1+c)/(1-c) ≤ 2*(b/a + c/b + a/c) := by
  sorry

end inequality_proof_l1174_117401


namespace probability_ratio_l1174_117431

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of different numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The number of cards with each number -/
def cards_per_number : ℕ := 5

/-- The probability of drawing five cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The probability of drawing four cards with one number and one card with a different number -/
def q : ℚ := (2250 : ℚ) / (Nat.choose total_cards cards_drawn)

theorem probability_ratio :
  q / p = 225 := by sorry

end probability_ratio_l1174_117431


namespace quadratic_inequality_solution_l1174_117475

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ -1 < x ∧ x < 3) → a + b = -1 := by
  sorry

end quadratic_inequality_solution_l1174_117475


namespace sequence_conclusions_l1174_117436

def sequence_property (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1)^2 - a (n + 1) = a n)

theorem sequence_conclusions (a : ℕ → ℝ) (h : sequence_property a) :
  (∀ n ≥ 2, a n > 1) ∧
  ((0 < a 1 ∧ a 1 < 2) → (∀ n, a n < a (n + 1))) ∧
  (a 1 > 2 → ∀ n ≥ 2, 2 < a n ∧ a n < a 1) := by
  sorry

end sequence_conclusions_l1174_117436


namespace solutions_satisfy_system_system_implies_solutions_l1174_117429

/-- The system of equations we want to solve -/
def system (x y z : ℝ) : Prop :=
  x + y + z = 2 ∧ x^2 + y^2 + z^2 = 26 ∧ x^3 + y^3 + z^3 = 38

/-- The set of solutions to our system -/
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(1, 4, -3), (1, -3, 4), (4, 1, -3), (4, -3, 1), (-3, 1, 4), (-3, 4, 1)}

/-- Theorem stating that the solutions satisfy the system -/
theorem solutions_satisfy_system : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → system x y z := by
  sorry

/-- Theorem stating that any solution to the system is in our set of solutions -/
theorem system_implies_solutions : ∀ (x y z : ℝ), system x y z → (x, y, z) ∈ solutions := by
  sorry

end solutions_satisfy_system_system_implies_solutions_l1174_117429


namespace herd_division_l1174_117437

theorem herd_division (total : ℕ) 
  (h1 : (1 : ℚ) / 3 * total + (1 : ℚ) / 6 * total + (1 : ℚ) / 9 * total + 12 = total) : 
  total = 54 := by
  sorry

end herd_division_l1174_117437


namespace max_npn_value_l1174_117413

def is_two_digit_same_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ (n / 10 = n % 10)

def is_one_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def is_three_digit_npn (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

theorem max_npn_value :
  ∀ (mm m npn : ℕ),
    is_two_digit_same_digits mm →
    is_one_digit m →
    is_three_digit_npn npn →
    mm * m = npn →
    npn ≤ 729 :=
sorry

end max_npn_value_l1174_117413


namespace minimal_poster_area_l1174_117483

theorem minimal_poster_area (m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) (h_m_ge_n : m ≥ n) : 
  ∃ (posters : Finset (ℕ × ℕ)), 
    (Finset.card posters = m * n) ∧ 
    (∀ (k l : ℕ), (k, l) ∈ posters → 1 ≤ k ∧ k ≤ m ∧ 1 ≤ l ∧ l ≤ n) →
    (minimal_area : ℕ) = m * (n * (n + 1) / 2) :=
by sorry

end minimal_poster_area_l1174_117483


namespace equation_solution_l1174_117477

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (5 * x)^4 = (10 * x)^3 → x = 8/5 := by
  sorry

end equation_solution_l1174_117477


namespace arithmetic_sequence_property_l1174_117488

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 13 + a 15 = 20) :
  a 10 - (1/5) * a 12 = 4 :=
sorry

end arithmetic_sequence_property_l1174_117488


namespace train_length_train_length_is_120_l1174_117403

/-- The length of a train that overtakes a motorbike -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) 
  (h1 : train_speed = 100) 
  (h2 : motorbike_speed = 64) 
  (h3 : overtake_time = 12) : ℝ :=
let train_speed_ms := train_speed * 1000 / 3600
let motorbike_speed_ms := motorbike_speed * 1000 / 3600
let relative_speed := train_speed_ms - motorbike_speed_ms
120

/-- The length of the train is 120 meters -/
theorem train_length_is_120 (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) 
  (h1 : train_speed = 100) 
  (h2 : motorbike_speed = 64) 
  (h3 : overtake_time = 12) : 
  train_length train_speed motorbike_speed overtake_time h1 h2 h3 = 120 := by
sorry

end train_length_train_length_is_120_l1174_117403


namespace min_value_of_fraction_l1174_117455

theorem min_value_of_fraction (x a b : ℝ) (hx : 0 < x ∧ x < 1) (ha : a > 0) (hb : b > 0) :
  ∃ m : ℝ, m = (a + b)^2 ∧ ∀ y : ℝ, 0 < y ∧ y < 1 → 1 / (y^a * (1 - y)^b) ≥ m := by
  sorry

end min_value_of_fraction_l1174_117455


namespace complement_union_theorem_l1174_117446

def U : Set Nat := {1,2,3,4,5}
def A : Set Nat := {2,3,4}
def B : Set Nat := {1,4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {1,4,5} := by sorry

end complement_union_theorem_l1174_117446


namespace ice_cream_sales_l1174_117433

theorem ice_cream_sales (chocolate : ℕ) (mango : ℕ) 
  (h1 : chocolate = 50) 
  (h2 : mango = 54) : 
  chocolate + mango - (chocolate * 3 / 5 + mango * 2 / 3) = 38 := by
  sorry

end ice_cream_sales_l1174_117433


namespace odd_function_property_l1174_117497

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_period : ∀ x, f (x + 2) = -f x) : 
  f (-2) = 0 := by
sorry

end odd_function_property_l1174_117497


namespace medals_award_ways_l1174_117458

-- Define the total number of sprinters
def total_sprinters : ℕ := 10

-- Define the number of Canadian sprinters
def canadian_sprinters : ℕ := 4

-- Define the number of non-Canadian sprinters
def non_canadian_sprinters : ℕ := total_sprinters - canadian_sprinters

-- Define the number of medals
def num_medals : ℕ := 3

-- Function to calculate the number of ways to award medals
def ways_to_award_medals : ℕ := 
  -- Case 1: No Canadians get a medal
  (non_canadian_sprinters * (non_canadian_sprinters - 1) * (non_canadian_sprinters - 2)) + 
  -- Case 2: Exactly one Canadian gets a medal
  (canadian_sprinters * num_medals * (non_canadian_sprinters) * (non_canadian_sprinters - 1))

-- Theorem statement
theorem medals_award_ways : 
  ways_to_award_medals = 360 := by sorry

end medals_award_ways_l1174_117458


namespace total_students_third_and_fourth_grade_l1174_117400

theorem total_students_third_and_fourth_grade 
  (third_grade : ℕ) 
  (difference : ℕ) 
  (h1 : third_grade = 203)
  (h2 : difference = 125) :
  third_grade + (third_grade + difference) = 531 := by
  sorry

end total_students_third_and_fourth_grade_l1174_117400


namespace sum_of_squares_representation_l1174_117448

theorem sum_of_squares_representation (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), (x : ℤ) = a^2 + b^2 ∧ (y : ℤ) = c^2 + d^2 ∧ (z : ℤ) = a * c + b * d := by
  sorry

end sum_of_squares_representation_l1174_117448


namespace mean_of_added_numbers_l1174_117447

theorem mean_of_added_numbers (original_numbers : List ℝ) (x y z : ℝ) :
  original_numbers.length = 7 →
  original_numbers.sum / original_numbers.length = 75 →
  (original_numbers.sum + x + y + z) / (original_numbers.length + 3) = 90 →
  (x + y + z) / 3 = 125 := by
sorry

end mean_of_added_numbers_l1174_117447


namespace earnings_difference_l1174_117443

/-- Proves the difference in earnings between Evan and Markese -/
theorem earnings_difference : 
  ∀ (E : ℕ), 
  E > 16 →  -- Evan earned more than Markese
  E + 16 = 37 →  -- Their combined earnings
  E - 16 = 5  -- The difference in earnings
  := by sorry

end earnings_difference_l1174_117443


namespace sqrt_7_simplest_l1174_117423

-- Define the concept of simplest square root
def is_simplest_sqrt (x : ℝ) : Prop :=
  x > 0 ∧ ∀ y : ℝ, y > 0 → y^2 = x → y = Real.sqrt x

-- Define the set of square roots to compare
def sqrt_set : Set ℝ := {Real.sqrt 24, Real.sqrt (1/3), Real.sqrt 7, Real.sqrt 0.2}

-- Theorem statement
theorem sqrt_7_simplest :
  ∀ x ∈ sqrt_set, x ≠ Real.sqrt 7 → ¬(is_simplest_sqrt x) ∧ is_simplest_sqrt (Real.sqrt 7) :=
by sorry

end sqrt_7_simplest_l1174_117423


namespace point_in_fourth_quadrant_m_range_l1174_117492

-- Define the point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (m + 3, m - 1)

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant_m_range :
  ∀ m : ℝ, in_fourth_quadrant (P m) ↔ -3 < m ∧ m < 1 :=
sorry

end point_in_fourth_quadrant_m_range_l1174_117492


namespace b_spending_percentage_l1174_117439

/-- Proves that B spends 85% of his salary given the specified conditions -/
theorem b_spending_percentage (total_salary : ℝ) (a_salary : ℝ) (a_spending_rate : ℝ) 
  (h1 : total_salary = 2000)
  (h2 : a_salary = 1500)
  (h3 : a_spending_rate = 0.95)
  (h4 : a_salary * (1 - a_spending_rate) = (total_salary - a_salary) * (1 - b_spending_rate)) :
  b_spending_rate = 0.85 := by
  sorry

#check b_spending_percentage

end b_spending_percentage_l1174_117439


namespace valid_draws_count_l1174_117419

/-- Represents the number of cards for each color --/
def cards_per_color : ℕ := 5

/-- Represents the number of colors --/
def num_colors : ℕ := 3

/-- Represents the number of cards drawn --/
def cards_drawn : ℕ := 4

/-- Represents the total number of cards --/
def total_cards : ℕ := cards_per_color * num_colors

/-- Represents the set of all possible draws --/
def all_draws : Set (Fin total_cards) := sorry

/-- Predicate to check if a draw contains all colors --/
def has_all_colors (draw : Fin total_cards) : Prop := sorry

/-- Predicate to check if a draw has all different letters --/
def has_different_letters (draw : Fin total_cards) : Prop := sorry

/-- The number of valid draws --/
def num_valid_draws : ℕ := sorry

theorem valid_draws_count : num_valid_draws = 360 := by sorry

end valid_draws_count_l1174_117419


namespace hyperbola_eq_theorem_l1174_117463

/-- A hyperbola with the given properties -/
structure Hyperbola where
  -- The hyperbola is centered at the origin
  center_origin : True
  -- The foci are on the coordinate axes
  foci_on_axes : True
  -- One of the asymptotes has the equation y = (1/2)x
  asymptote_eq : ∀ x y : ℝ, y = (1/2) * x
  -- The point (2, 2) lies on the hyperbola
  point_on_hyperbola : True

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (y^2 / 3) - (x^2 / 12) = 1

/-- Theorem stating the equation of the hyperbola with the given properties -/
theorem hyperbola_eq_theorem (h : Hyperbola) :
  ∀ x y : ℝ, hyperbola_equation h x y :=
sorry

end hyperbola_eq_theorem_l1174_117463


namespace special_gp_common_ratio_l1174_117409

/-- A geometric progression with positive terms where any term minus the next term 
    equals half the sum of the next two terms. -/
structure SpecialGeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : 0 < a
  r_pos : 0 < r
  special_property : ∀ n : ℕ, a * r^n - a * r^(n+1) = (1/2) * (a * r^(n+1) + a * r^(n+2))

/-- The common ratio of a special geometric progression is (√17 - 3) / 2. -/
theorem special_gp_common_ratio (gp : SpecialGeometricProgression) : 
  gp.r = (Real.sqrt 17 - 3) / 2 := by
  sorry

end special_gp_common_ratio_l1174_117409


namespace algebra_to_calculus_ratio_l1174_117489

/-- Represents the number of years Devin taught each subject and in total -/
structure TeachingYears where
  calculus : ℕ
  algebra : ℕ
  statistics : ℕ
  total : ℕ

/-- Defines the conditions of Devin's teaching career -/
def devin_teaching (y : TeachingYears) : Prop :=
  y.calculus = 4 ∧
  y.statistics = 5 * y.algebra ∧
  y.total = y.calculus + y.algebra + y.statistics ∧
  y.total = 52

/-- Theorem stating the ratio of Algebra to Calculus teaching years -/
theorem algebra_to_calculus_ratio (y : TeachingYears) 
  (h : devin_teaching y) : y.algebra / y.calculus = 2 := by
  sorry

end algebra_to_calculus_ratio_l1174_117489


namespace polynomial_value_constraint_l1174_117459

theorem polynomial_value_constraint (P : ℤ → ℤ) (a b c d : ℤ) :
  (∃ (n : ℤ → ℤ), P = n) →
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  P a = 1979 →
  P b = 1979 →
  P c = 1979 →
  P d = 1979 →
  ∀ (x : ℤ), P x ≠ 3958 :=
by sorry

end polynomial_value_constraint_l1174_117459


namespace impossible_digit_product_after_increment_l1174_117469

def digit_product (n : ℕ) : ℕ := sorry

theorem impossible_digit_product_after_increment :
  ∀ N : ℕ,
  N > 0 →
  digit_product N = 20 →
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 24) ∧
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 25) ∧
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 30) ∧
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 40) ∧
  ¬(∃ M : ℕ, M = N + 1 ∧ digit_product M = 35) :=
by sorry

end impossible_digit_product_after_increment_l1174_117469


namespace largest_integer_satisfying_inequality_l1174_117410

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (3 - 5 * x > 22) → x ≤ -4 ∧ (3 - 5 * (-4) > 22) :=
by sorry

end largest_integer_satisfying_inequality_l1174_117410


namespace sin_pi_half_minus_x_is_even_l1174_117422

/-- The function f(x) = sin(π/2 - x) is even, implying symmetry about the y-axis -/
theorem sin_pi_half_minus_x_is_even :
  ∀ x : ℝ, Real.sin (π / 2 - x) = Real.sin (π / 2 - (-x)) :=
by sorry

end sin_pi_half_minus_x_is_even_l1174_117422


namespace subset_implies_m_eq_two_l1174_117430

/-- The set A of solutions to the quadratic equation x^2 + 3x + 2 = 0 -/
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}

/-- The set B of solutions to the quadratic equation x^2 + (m+1)x + m = 0 -/
def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

/-- Theorem stating that if A is a subset of B, then m must equal 2 -/
theorem subset_implies_m_eq_two : A ⊆ B 2 := by
  sorry

end subset_implies_m_eq_two_l1174_117430


namespace inequality_solution_set_l1174_117486

theorem inequality_solution_set : ∀ x : ℝ, 
  (2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5) ↔ (5 / 2 < x ∧ x ≤ 14 / 5) := by
  sorry

end inequality_solution_set_l1174_117486


namespace cube_volume_from_surface_area_l1174_117478

/-- Given a cube with surface area 294 square centimeters, its volume is 343 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  s > 0 →
  6 * s^2 = 294 →
  s^3 = 343 := by
sorry

end cube_volume_from_surface_area_l1174_117478


namespace find_S_l1174_117425

theorem find_S : ∃ S : ℝ, (1/3 * 1/8 * S = 1/4 * 1/6 * 120) ∧ (S = 120) := by
  sorry

end find_S_l1174_117425


namespace number_multiple_problem_l1174_117470

theorem number_multiple_problem (A B k : ℕ) 
  (sum_cond : A + B = 77)
  (bigger_cond : A = 42)
  (multiple_cond : 6 * B = k * A) :
  k = 5 := by sorry

end number_multiple_problem_l1174_117470


namespace marcella_shoes_theorem_l1174_117472

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_pairs_remaining (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

/-- Theorem stating that with 50 initial pairs and 15 individual shoes lost,
    the maximum number of complete pairs remaining is 35. -/
theorem marcella_shoes_theorem :
  max_pairs_remaining 50 15 = 35 := by
  sorry

end marcella_shoes_theorem_l1174_117472


namespace no_valid_labeling_l1174_117456

/-- Represents a labeling of the hexagon vertices and center -/
def Labeling := Fin 7 → Fin 7

/-- The sum of labels on a line through the center -/
def lineSum (l : Labeling) (i j : Fin 7) : ℕ :=
  l i + l 6 + l j  -- Assuming index 6 represents the center J

/-- Checks if a labeling is valid according to the problem conditions -/
def isValidLabeling (l : Labeling) : Prop :=
  (Function.Injective l) ∧ 
  (lineSum l 0 4 = lineSum l 1 5) ∧
  (lineSum l 1 5 = lineSum l 2 3) ∧
  (lineSum l 2 3 = lineSum l 0 4)

/-- The main theorem: there are no valid labelings -/
theorem no_valid_labeling : 
  ¬ ∃ l : Labeling, isValidLabeling l :=
sorry

end no_valid_labeling_l1174_117456


namespace any_nonzero_to_zero_power_l1174_117465

theorem any_nonzero_to_zero_power (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end any_nonzero_to_zero_power_l1174_117465


namespace rectangle_horizontal_length_l1174_117493

theorem rectangle_horizontal_length 
  (perimeter : ℝ) 
  (h v : ℝ) 
  (perimeter_eq : perimeter = 2 * h + 2 * v) 
  (vertical_shorter : v = h - 3) 
  (perimeter_value : perimeter = 54) : h = 15 := by
  sorry

end rectangle_horizontal_length_l1174_117493


namespace fifth_grade_total_is_144_l1174_117484

/-- The number of students in the fifth grade of Longteng Primary School --/
def fifth_grade_total : ℕ :=
  let class1 : ℕ := 42
  let class2 : ℕ := (class1 * 6) / 7
  let class3 : ℕ := (class2 * 5) / 6
  let class4 : ℕ := (class3 * 12) / 10
  class1 + class2 + class3 + class4

theorem fifth_grade_total_is_144 : fifth_grade_total = 144 := by
  sorry

end fifth_grade_total_is_144_l1174_117484


namespace geometric_sequence_sum_problem_l1174_117434

/-- Represents the sum of the first k terms of a geometric sequence -/
noncomputable def S (a₁ q : ℝ) (k : ℕ) : ℝ :=
  if q = 1 then k * a₁ else a₁ * (1 - q^k) / (1 - q)

theorem geometric_sequence_sum_problem
  (a₁ q : ℝ)
  (h_pos : ∀ n : ℕ, 0 < a₁ * q^n)
  (h_Sn : S a₁ q n = 2)
  (h_S3n : S a₁ q (3*n) = 14) :
  S a₁ q (4*n) = 30 :=
sorry

end geometric_sequence_sum_problem_l1174_117434


namespace art_show_ratio_l1174_117453

theorem art_show_ratio (total_painted : ℚ) (sold : ℚ) 
  (h1 : total_painted = 180.5)
  (h2 : sold = 76.3) :
  (total_painted - sold) / sold = 1042 / 763 := by
  sorry

end art_show_ratio_l1174_117453


namespace line_chart_drawing_method_l1174_117416

/-- Represents a point on a grid -/
structure GridPoint where
  x : ℝ
  y : ℝ

/-- Represents a line chart -/
structure LineChart where
  points : List GridPoint
  unit_length : ℝ
  quantity : ℝ → ℝ

/-- The method of drawing a line chart -/
def draw_line_chart (chart : LineChart) : Prop :=
  ∃ (plotted_points : List GridPoint) (connected_points : List GridPoint),
    plotted_points = chart.points ∧
    connected_points = chart.points ∧
    (∀ p ∈ chart.points, p.y = chart.quantity (p.x * chart.unit_length))

theorem line_chart_drawing_method (chart : LineChart) :
  draw_line_chart chart ↔
  (∃ (plotted_points : List GridPoint) (connected_points : List GridPoint),
    plotted_points = chart.points ∧
    connected_points = chart.points) :=
sorry

end line_chart_drawing_method_l1174_117416


namespace common_external_tangents_parallel_l1174_117468

/-- Two circles with equal radii -/
structure EqualRadiiCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  radius : ℝ

/-- A line representing a common external tangent -/
structure CommonExternalTangent where
  slope : ℝ
  intercept : ℝ

/-- The line connecting the centers of two circles -/
def centerLine (c : EqualRadiiCircles) : ℝ × ℝ → Prop :=
  fun p => ∃ t : ℝ, p = (1 - t) • c.center1 + t • c.center2

/-- Two lines are parallel -/
def parallel (l1 l2 : ℝ × ℝ → Prop) : Prop :=
  ∀ p q : ℝ × ℝ, l1 p → l1 q → l2 p → l2 q → 
    (p.1 - q.1) * (p.2 - q.2) = (p.1 - q.1) * (p.2 - q.2)

theorem common_external_tangents_parallel (c : EqualRadiiCircles) 
  (t1 t2 : CommonExternalTangent) : 
  parallel (fun p => p.2 = t1.slope * p.1 + t1.intercept) 
           (fun p => p.2 = t2.slope * p.1 + t2.intercept) ∧
  parallel (fun p => p.2 = t1.slope * p.1 + t1.intercept) 
           (centerLine c) := by
  sorry

end common_external_tangents_parallel_l1174_117468


namespace tank_capacity_is_33_l1174_117408

/-- Represents the capacity of a water tank with specific filling conditions. -/
def tank_capacity (initial_fraction : ℚ) (added_water : ℕ) (leak_rate : ℕ) (fill_time : ℕ) : ℚ :=
  let total_leak := (leak_rate * fill_time : ℚ)
  let total_added := (added_water : ℚ) + total_leak
  total_added / (1 - initial_fraction)

/-- Theorem stating that under given conditions, the tank capacity is 33 gallons. -/
theorem tank_capacity_is_33 :
  tank_capacity (1/3) 16 2 3 = 33 := by
  sorry

end tank_capacity_is_33_l1174_117408


namespace equation_proof_l1174_117411

theorem equation_proof : 225 + 2 * 15 * 4 + 16 = 361 := by
  sorry

end equation_proof_l1174_117411


namespace product_of_roots_l1174_117417

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 18 → 
  ∃ y : ℝ, (y + 3) * (y - 4) = 18 ∧ x * y = -30 := by
  sorry

end product_of_roots_l1174_117417


namespace average_weight_problem_l1174_117496

/-- The average weight problem -/
theorem average_weight_problem 
  (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84)
  (h2 : A = 78)
  (h3 : E = D + 6)
  (h4 : (B + C + D + E) / 4 = 79) :
  (A + B + C + D) / 4 = 80 := by
  sorry

end average_weight_problem_l1174_117496


namespace fathers_age_twice_marikas_l1174_117487

/-- Marika's age in 2006 -/
def marika_age_2006 : ℕ := 10

/-- The year of Marika's 10th birthday -/
def birth_year : ℕ := 2006

/-- The ratio of father's age to Marika's age in 2006 -/
def age_ratio_2006 : ℕ := 5

/-- The year when father's age will be twice Marika's age -/
def target_year : ℕ := 2036

theorem fathers_age_twice_marikas : 
  target_year = birth_year + (age_ratio_2006 - 2) * marika_age_2006 := by
  sorry

end fathers_age_twice_marikas_l1174_117487


namespace shoe_pair_difference_l1174_117460

theorem shoe_pair_difference (ellie_shoes riley_shoes : ℕ) : 
  ellie_shoes = 8 →
  riley_shoes < ellie_shoes →
  ellie_shoes + riley_shoes = 13 →
  ellie_shoes - riley_shoes = 3 :=
by
  sorry

end shoe_pair_difference_l1174_117460


namespace prop2_prop4_l1174_117445

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Theorem for proposition 2
theorem prop2 (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → plane_perpendicular α β := by sorry

-- Theorem for proposition 4
theorem prop4 (m : Line) (α β : Plane) :
  perpendicular m α → plane_parallel α β → perpendicular m β := by sorry

end prop2_prop4_l1174_117445


namespace containers_used_is_three_l1174_117427

/-- The number of posters that can be printed with the initial amount of ink -/
def initial_posters : ℕ := 60

/-- The number of posters that can be printed after losing one container of ink -/
def remaining_posters : ℕ := 45

/-- The number of posters that can be printed with one container of ink -/
def posters_per_container : ℕ := initial_posters - remaining_posters

/-- The number of containers used to print the remaining posters -/
def containers_used : ℕ := remaining_posters / posters_per_container

theorem containers_used_is_three :
  containers_used = 3 := by sorry

end containers_used_is_three_l1174_117427


namespace total_amount_paid_l1174_117451

def ticket_cost : ℕ := 44
def num_people : ℕ := 3
def service_fee : ℕ := 18

theorem total_amount_paid : 
  ticket_cost * num_people + service_fee = 150 := by
  sorry

end total_amount_paid_l1174_117451


namespace circle_properties_l1174_117432

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 4

-- Define the line that contains the center of C
def center_line (x y : ℝ) : Prop :=
  2 * x - y + 2 = 0

-- Define the intersecting line
def intersecting_line (a x y : ℝ) : Prop :=
  x = a * y + 3

-- Theorem statement
theorem circle_properties :
  -- Given conditions
  (circle_C 1 0) ∧ 
  (circle_C (-1) 2) ∧
  (∃ (x₀ y₀ : ℝ), circle_C x₀ y₀ ∧ center_line x₀ y₀) →
  -- Conclusions
  (∀ (x y : ℝ), circle_C x y ↔ (x + 1)^2 + y^2 = 4) ∧
  (∃ (a : ℝ), (a = Real.sqrt 15 ∨ a = -Real.sqrt 15) ∧
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
      intersecting_line a x₁ y₁ ∧ intersecting_line a x₂ y₂ ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) :=
by sorry


end circle_properties_l1174_117432


namespace prob_two_consecutive_sum_four_l1174_117424

-- Define a 3-sided die
def Die := Fin 3

-- Define the sum of two dice
def diceSum (d1 d2 : Die) : ℕ := d1.val + d2.val + 2

-- Define the probability of getting a sum of 4 on a single roll
def probSumFour : ℚ := 1 / 3

-- Theorem statement
theorem prob_two_consecutive_sum_four :
  (probSumFour * probSumFour : ℚ) = 1 / 9 := by sorry

end prob_two_consecutive_sum_four_l1174_117424


namespace translation_theorem_l1174_117485

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point left by a given amount -/
def translateLeft (p : Point) (amount : ℝ) : Point :=
  { x := p.x - amount, y := p.y }

/-- Translates a point down by a given amount -/
def translateDown (p : Point) (amount : ℝ) : Point :=
  { x := p.x, y := p.y - amount }

theorem translation_theorem :
  let p := Point.mk (-4) 3
  let p' := translateDown (translateLeft p 2) 2
  p'.x = -6 ∧ p'.y = 1 := by sorry

end translation_theorem_l1174_117485


namespace quadratic_solution_property_l1174_117474

theorem quadratic_solution_property (k : ℚ) : 
  (∃ a b : ℚ, 
    (5 * a^2 + 4 * a + k = 0) ∧ 
    (5 * b^2 + 4 * b + k = 0) ∧ 
    (|a - b| = a^2 + b^2)) ↔ 
  (k = 3/5 ∨ k = -12/5) := by
sorry

end quadratic_solution_property_l1174_117474


namespace sin_inequality_l1174_117490

open Real

theorem sin_inequality (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) :
  let f := fun θ => (sin θ)^3 / (2 * θ - sin (2 * θ))
  f α > f β := by
sorry

end sin_inequality_l1174_117490


namespace christine_walking_distance_l1174_117406

/-- Given Christine's walking speed and time spent walking, calculate the distance she wandered. -/
theorem christine_walking_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (h1 : speed = 4) 
  (h2 : time = 5) : 
  speed * time = 20 := by
sorry

end christine_walking_distance_l1174_117406


namespace intersection_A_B_complement_A_complement_A_union_B_l1174_117494

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- State the theorems to be proved
theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 7} := by sorry

theorem complement_A : (Set.univ \ A) = {x | x < 3 ∨ 7 ≤ x} := by sorry

theorem complement_A_union_B : (Set.univ \ (A ∪ B)) = {x | x ≤ 2 ∨ 10 ≤ x} := by sorry

end intersection_A_B_complement_A_complement_A_union_B_l1174_117494


namespace original_triangle_area_l1174_117420

theorem original_triangle_area
  (original_area : ℝ)
  (new_area : ℝ)
  (h1 : new_area = 32)
  (h2 : new_area = 4 * original_area) :
  original_area = 8 :=
by sorry

end original_triangle_area_l1174_117420


namespace birds_in_tree_l1174_117414

theorem birds_in_tree (initial_birds : ℕ) : 
  initial_birds + 21 = 35 → initial_birds = 14 := by sorry

end birds_in_tree_l1174_117414


namespace classroom_books_l1174_117452

/-- The total number of books in a classroom -/
def total_books (num_children : ℕ) (books_per_child : ℕ) (teacher_books : ℕ) : ℕ :=
  num_children * books_per_child + teacher_books

/-- Theorem: The total number of books in the classroom is 202 -/
theorem classroom_books : 
  total_books 15 12 22 = 202 := by
  sorry

end classroom_books_l1174_117452


namespace number_division_problem_l1174_117498

theorem number_division_problem (x : ℚ) : (x / 5 = 75 + x / 6) → x = 2250 := by
  sorry

end number_division_problem_l1174_117498


namespace number_exists_l1174_117435

theorem number_exists : ∃ N : ℝ, 2.5 * N = 199.99999999999997 := by
  sorry

end number_exists_l1174_117435


namespace penny_frog_count_l1174_117499

/-- The number of tree frogs Penny counted -/
def tree_frogs : ℕ := 55

/-- The number of poison frogs Penny counted -/
def poison_frogs : ℕ := 10

/-- The number of wood frogs Penny counted -/
def wood_frogs : ℕ := 13

/-- The total number of frogs Penny counted -/
def total_frogs : ℕ := tree_frogs + poison_frogs + wood_frogs

theorem penny_frog_count : total_frogs = 78 := by
  sorry

end penny_frog_count_l1174_117499


namespace zoo_count_l1174_117441

theorem zoo_count (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 200) 
  (h2 : total_legs = 522) : 
  ∃ (birds mammals : ℕ), 
    birds + mammals = total_animals ∧ 
    2 * birds + 4 * mammals = total_legs ∧ 
    birds = 139 := by
  sorry

end zoo_count_l1174_117441


namespace jimin_class_size_l1174_117481

/-- The number of students in Jimin's class -/
def total_students : ℕ := 45

/-- The number of students who like Korean -/
def korean_fans : ℕ := 38

/-- The number of students who like math -/
def math_fans : ℕ := 39

/-- The number of students who like both Korean and math -/
def both_fans : ℕ := 32

/-- There is no student who does not like both Korean and math -/
axiom no_other_students : total_students = korean_fans + math_fans - both_fans

theorem jimin_class_size :
  total_students = 45 :=
sorry

end jimin_class_size_l1174_117481


namespace min_corners_8x8_grid_l1174_117495

/-- Represents a square grid --/
structure Grid :=
  (size : Nat)

/-- Represents a seven-cell corner --/
structure SevenCellCorner

/-- The number of cells in a seven-cell corner --/
def SevenCellCorner.cells : Nat := 7

/-- Checks if a given number of seven-cell corners can fit in the grid --/
def can_fit (g : Grid) (n : Nat) : Prop :=
  g.size * g.size ≥ n * SevenCellCorner.cells

/-- Checks if after clipping n seven-cell corners, no more can be clipped --/
def no_more_corners (g : Grid) (n : Nat) : Prop :=
  can_fit g n ∧ ¬can_fit g (n + 1)

/-- The main theorem: The minimum number of seven-cell corners that can be clipped from an 8x8 grid such that no more can be clipped is 3 --/
theorem min_corners_8x8_grid :
  ∃ (n : Nat), n = 3 ∧ no_more_corners (Grid.mk 8) n ∧ ∀ m < n, ¬no_more_corners (Grid.mk 8) m :=
sorry

end min_corners_8x8_grid_l1174_117495


namespace fencing_cost_is_1634_l1174_117438

/-- Represents the cost of fencing for a single side -/
structure SideCost where
  length : ℝ
  costPerFoot : ℝ

/-- Calculates the total cost of fencing for an irregular four-sided plot -/
def totalFencingCost (sideA sideB sideC sideD : SideCost) : ℝ :=
  sideA.length * sideA.costPerFoot +
  sideB.length * sideB.costPerFoot +
  sideC.length * sideC.costPerFoot +
  sideD.length * sideD.costPerFoot

/-- Theorem stating that the total fencing cost for the given plot is 1634 -/
theorem fencing_cost_is_1634 :
  let sideA : SideCost := { length := 8, costPerFoot := 58 }
  let sideB : SideCost := { length := 5, costPerFoot := 62 }
  let sideC : SideCost := { length := 6, costPerFoot := 64 }
  let sideD : SideCost := { length := 7, costPerFoot := 68 }
  totalFencingCost sideA sideB sideC sideD = 1634 := by
  sorry

end fencing_cost_is_1634_l1174_117438


namespace circles_contained_l1174_117482

theorem circles_contained (r R d : ℝ) (hr : r = 1) (hR : R = 5) (hd : d = 3) :
  d < R - r ∧ d + r < R :=
sorry

end circles_contained_l1174_117482


namespace franks_breakfast_cost_l1174_117491

/-- The cost of Frank's breakfast shopping -/
def breakfast_cost (bun_price : ℚ) (bun_quantity : ℕ) (milk_price : ℚ) (milk_quantity : ℕ) (egg_price_multiplier : ℕ) : ℚ :=
  bun_price * bun_quantity + milk_price * milk_quantity + (milk_price * egg_price_multiplier)

/-- Theorem stating that Frank's breakfast shopping costs $11 -/
theorem franks_breakfast_cost :
  breakfast_cost 0.1 10 2 2 3 = 11 := by
  sorry

end franks_breakfast_cost_l1174_117491


namespace small_bottles_sold_percentage_l1174_117440

theorem small_bottles_sold_percentage
  (initial_small : ℕ)
  (initial_big : ℕ)
  (big_sold_percentage : ℚ)
  (total_remaining : ℕ)
  (h1 : initial_small = 6000)
  (h2 : initial_big = 15000)
  (h3 : big_sold_percentage = 12 / 100)
  (h4 : total_remaining = 18540)
  (h5 : ∃ (x : ℚ), 
    initial_small - (x * initial_small) + 
    initial_big - (big_sold_percentage * initial_big) = total_remaining) :
  ∃ (x : ℚ), x = 11 / 100 ∧ 
    initial_small - (x * initial_small) + 
    initial_big - (big_sold_percentage * initial_big) = total_remaining :=
by sorry

end small_bottles_sold_percentage_l1174_117440


namespace bacteria_growth_l1174_117464

/-- The time in minutes for the bacteria population to double -/
def doubling_time : ℝ := 6

/-- The initial population of bacteria -/
def initial_population : ℕ := 1000

/-- The total time of growth in minutes -/
def total_time : ℝ := 53.794705707972525

/-- The final population of bacteria after the given time -/
def final_population : ℕ := 495451

theorem bacteria_growth :
  let num_doublings : ℝ := total_time / doubling_time
  let theoretical_population : ℝ := initial_population * (2 ^ num_doublings)
  ⌊theoretical_population⌋ = final_population :=
sorry

end bacteria_growth_l1174_117464


namespace football_equipment_cost_l1174_117418

/-- The cost of equipment for a football team -/
theorem football_equipment_cost : 
  let num_players : ℕ := 16
  let jersey_cost : ℚ := 25
  let shorts_cost : ℚ := 15.20
  let socks_cost : ℚ := 6.80
  let total_cost : ℚ := num_players * (jersey_cost + shorts_cost + socks_cost)
  total_cost = 752 := by sorry

end football_equipment_cost_l1174_117418


namespace cosine_sine_power_equation_l1174_117466

theorem cosine_sine_power_equation (x : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  (Real.cos x)^8 + (Real.sin x)^8 = 97 / 128 ↔ x = Real.pi / 12 ∨ x = 5 * Real.pi / 12 := by
  sorry

end cosine_sine_power_equation_l1174_117466


namespace largest_divisor_of_a_pow_25_minus_a_l1174_117449

theorem largest_divisor_of_a_pow_25_minus_a : 
  ∃ (n : ℕ), n = 2730 ∧ 
  (∀ (a : ℤ), (a^25 - a) % n = 0) ∧
  (∀ (m : ℕ), m > n → ∃ (a : ℤ), (a^25 - a) % m ≠ 0) :=
sorry

end largest_divisor_of_a_pow_25_minus_a_l1174_117449


namespace f_composition_value_l1174_117412

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else (1/2) ^ x

theorem f_composition_value : f (f (1/27)) = 8 := by
  sorry

end f_composition_value_l1174_117412


namespace variance_best_stability_measure_l1174_117405

/-- A performance measure is a function that takes a list of real numbers (representing performance data) and returns a real number. -/
def PerformanceMeasure := (List ℝ) → ℝ

/-- Average of a list of real numbers -/
def average : PerformanceMeasure := sorry

/-- Median of a list of real numbers -/
def median : PerformanceMeasure := sorry

/-- Mode of a list of real numbers -/
def mode : PerformanceMeasure := sorry

/-- Variance of a list of real numbers -/
def variance : PerformanceMeasure := sorry

/-- A measure is considered stable if it reflects the spread of the data -/
def reflectsSpread (m : PerformanceMeasure) : Prop := sorry

/-- Theorem stating that variance is the measure that best reflects the stability of performance -/
theorem variance_best_stability_measure :
  reflectsSpread variance ∧
  ¬reflectsSpread average ∧
  ¬reflectsSpread median ∧
  ¬reflectsSpread mode :=
sorry

end variance_best_stability_measure_l1174_117405


namespace h_equation_l1174_117480

theorem h_equation (x : ℝ) (h : ℝ → ℝ) :
  (4 * x^4 + 11 * x^3 + h x = 10 * x^3 - x^2 + 4 * x - 7) →
  h x = -4 * x^4 - x^3 - x^2 + 4 * x - 7 := by
sorry

end h_equation_l1174_117480


namespace ratio_change_after_addition_l1174_117426

theorem ratio_change_after_addition : 
  ∀ (a b : ℕ), 
    (a : ℚ) / b = 2 / 3 →
    b - a = 8 →
    (a + 4 : ℚ) / (b + 4) = 5 / 7 := by
  sorry

end ratio_change_after_addition_l1174_117426


namespace triangle_BC_length_l1174_117473

/-- Triangle ABC with given properties -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  AM : ℝ  -- Median from A to midpoint of BC
  area : ℝ
  BC : ℝ

/-- The triangle satisfies the given conditions -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.AB = 6 ∧ t.AC = 8 ∧ t.AM = 5 ∧ t.area = 24

/-- Theorem: If a triangle satisfies the given conditions, its BC side length is 10 -/
theorem triangle_BC_length (t : Triangle) (h : satisfies_conditions t) : t.BC = 10 := by
  sorry

end triangle_BC_length_l1174_117473


namespace systematic_sampling_eighth_group_l1174_117461

/-- Systematic sampling function -/
def systematicSample (totalEmployees : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun n => firstSample + (n - 1) * (totalEmployees / sampleSize)

theorem systematic_sampling_eighth_group 
  (totalEmployees : ℕ) 
  (sampleSize : ℕ) 
  (firstSample : ℕ) :
  totalEmployees = 200 →
  sampleSize = 40 →
  firstSample = 22 →
  systematicSample totalEmployees sampleSize firstSample 8 = 37 :=
by
  sorry

#eval systematicSample 200 40 22 8

end systematic_sampling_eighth_group_l1174_117461


namespace temperature_difference_is_8_l1174_117428

-- Define the temperatures
def temp_top : ℝ := -9
def temp_foot : ℝ := -1

-- Define the temperature difference
def temp_difference : ℝ := temp_foot - temp_top

-- Theorem statement
theorem temperature_difference_is_8 : temp_difference = 8 := by
  sorry

end temperature_difference_is_8_l1174_117428


namespace trig_equation_solution_l1174_117402

theorem trig_equation_solution (z : ℝ) :
  5 * (Real.sin (2 * z))^4 - 4 * (Real.sin (2 * z))^2 * (Real.cos (2 * z))^2 - (Real.cos (2 * z))^4 + 4 * Real.cos (4 * z) = 0 →
  (∃ k : ℤ, z = π / 8 * (2 * ↑k + 1)) ∨ (∃ n : ℤ, z = π / 6 * (3 * ↑n + 1) ∨ z = π / 6 * (3 * ↑n - 1)) := by
sorry

end trig_equation_solution_l1174_117402


namespace lcm_of_5_8_10_27_l1174_117454

theorem lcm_of_5_8_10_27 : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 27)) = 1080 := by
  sorry

end lcm_of_5_8_10_27_l1174_117454


namespace truck_speed_theorem_l1174_117476

/-- The average speed of Truck X in miles per hour -/
def truck_x_speed : ℝ := 47

/-- The average speed of Truck Y in miles per hour -/
def truck_y_speed : ℝ := 53

/-- The initial distance between Truck X and Truck Y in miles -/
def initial_distance : ℝ := 13

/-- The time it takes for Truck Y to overtake and get ahead of Truck X in hours -/
def overtake_time : ℝ := 3

/-- The distance Truck Y is ahead of Truck X after overtaking in miles -/
def final_distance : ℝ := 5

theorem truck_speed_theorem :
  truck_x_speed * overtake_time + initial_distance + final_distance = truck_y_speed * overtake_time :=
by sorry

end truck_speed_theorem_l1174_117476


namespace algebraic_expression_simplification_l1174_117415

theorem algebraic_expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (a + 1) / a / (a - (1 + 2 * a^2) / (3 * a)) = Real.sqrt 3 := by
  sorry

end algebraic_expression_simplification_l1174_117415


namespace absolute_value_equation_solution_l1174_117444

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |2*y - 44| + |y - 24| = |3*y - 66| := by
  sorry

end absolute_value_equation_solution_l1174_117444


namespace speed_equivalence_l1174_117404

/-- Proves that a speed of 12/36 m/s is equivalent to 1.2 km/h -/
theorem speed_equivalence : ∀ (x : ℚ), x = 12 / 36 → x * (3600 / 1000) = 1.2 := by
  sorry

end speed_equivalence_l1174_117404


namespace tomatoes_needed_fried_green_tomatoes_l1174_117450

theorem tomatoes_needed (slices_per_tomato : ℕ) (slices_per_meal : ℕ) (people : ℕ) : ℕ :=
  let total_slices := slices_per_meal * people
  total_slices / slices_per_tomato

theorem fried_green_tomatoes :
  tomatoes_needed 8 20 8 = 20 := by
  sorry

end tomatoes_needed_fried_green_tomatoes_l1174_117450
