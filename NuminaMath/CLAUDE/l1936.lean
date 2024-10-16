import Mathlib

namespace NUMINAMATH_CALUDE_ampersand_composition_l1936_193660

def ampersand_right (x : ℝ) : ℝ := 9 - x

def ampersand_left (x : ℝ) : ℝ := x - 9

theorem ampersand_composition : ampersand_left (ampersand_right 10) = -10 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_composition_l1936_193660


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1936_193693

theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-2 : ℝ) 1 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 - b * x + a < 0} = Set.Ioo (-1 : ℝ) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1936_193693


namespace NUMINAMATH_CALUDE_room_problem_l1936_193619

theorem room_problem (boys girls : ℕ) : 
  boys = 3 * girls ∧ 
  (boys - 4) = 5 * (girls - 4) →
  boys + girls = 32 :=
by sorry

end NUMINAMATH_CALUDE_room_problem_l1936_193619


namespace NUMINAMATH_CALUDE_range_of_a_given_inequalities_and_unique_solution_l1936_193667

theorem range_of_a_given_inequalities_and_unique_solution :
  ∀ a : ℝ,
  (∃! x : ℤ, (2 * ↑x - 7 < 0 ∧ ↑x - a > 0)) →
  (2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_inequalities_and_unique_solution_l1936_193667


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1936_193600

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 32 →
  a 4 + a 5 = 55 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1936_193600


namespace NUMINAMATH_CALUDE_infinite_hyperbolas_l1936_193692

/-- A hyperbola with asymptotes 2x ± 3y = 0 -/
def Hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), 4 * x^2 - 9 * y^2 = k ∧ k ≠ 0

/-- The set of all hyperbolas with asymptotes 2x ± 3y = 0 -/
def HyperbolaSet : Set ℝ :=
  {k : ℝ | Hyperbola k}

/-- Theorem stating that there are infinitely many hyperbolas with asymptotes 2x ± 3y = 0 -/
theorem infinite_hyperbolas : Set.Infinite HyperbolaSet := by
  sorry

end NUMINAMATH_CALUDE_infinite_hyperbolas_l1936_193692


namespace NUMINAMATH_CALUDE_james_final_amounts_l1936_193689

def calculate_final_amounts (initial_gold : ℕ) (tax_rate : ℚ) (divorce_loss : ℚ) 
  (investment_percentage : ℚ) (stock_gain : ℕ) (exchange_rates : List ℚ) : ℕ × ℕ × ℕ :=
  sorry

theorem james_final_amounts :
  let initial_gold : ℕ := 60
  let tax_rate : ℚ := 1/10
  let divorce_loss : ℚ := 1/2
  let investment_percentage : ℚ := 1/4
  let stock_gain : ℕ := 1
  let exchange_rates : List ℚ := [5, 7, 3]
  let (silver_bars, remaining_gold, stock_investment) := 
    calculate_final_amounts initial_gold tax_rate divorce_loss investment_percentage stock_gain exchange_rates
  silver_bars = 99 ∧ remaining_gold = 3 ∧ stock_investment = 6 :=
by sorry

end NUMINAMATH_CALUDE_james_final_amounts_l1936_193689


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l1936_193665

noncomputable def g (a c : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then 2 * a * x + 4
  else if -3 ≤ x ∧ x ≤ 3 then x^2 - 7
  else 3 * x - c

def IsContinuous (f : ℝ → ℝ) : Prop :=
  ∀ x₀ : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x₀| < δ → |f x - f x₀| < ε

theorem continuous_piecewise_function_sum (a c : ℝ) :
  IsContinuous (g a c) → a + c = -34/3 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l1936_193665


namespace NUMINAMATH_CALUDE_diana_reading_time_l1936_193601

/-- The number of hours Diana read this week -/
def hours_read : ℝ := 12

/-- The initial reward rate in minutes per hour -/
def initial_rate : ℝ := 30

/-- The percentage increase in the reward rate -/
def rate_increase : ℝ := 0.2

/-- The total increase in video game time due to the raise in minutes -/
def total_increase : ℝ := 72

theorem diana_reading_time :
  hours_read * initial_rate * rate_increase = total_increase := by
  sorry

end NUMINAMATH_CALUDE_diana_reading_time_l1936_193601


namespace NUMINAMATH_CALUDE_factorization_left_to_right_l1936_193610

/-- Factorization from left to right for x^2 - 1 -/
theorem factorization_left_to_right :
  ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_left_to_right_l1936_193610


namespace NUMINAMATH_CALUDE_geometric_sequence_with_unit_modulus_ratio_l1936_193687

theorem geometric_sequence_with_unit_modulus_ratio (α : ℝ) : 
  let a : ℕ → ℂ := λ n => Complex.cos (n * α) + Complex.I * Complex.sin (n * α)
  ∃ r : ℂ, (∀ n : ℕ, a (n + 1) = r * a n) ∧ Complex.abs r = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_with_unit_modulus_ratio_l1936_193687


namespace NUMINAMATH_CALUDE_triangle_with_60_degree_angle_l1936_193613

/-- In a triangle with sides 4, 2√3, and 2 + 2√2, one of the angles is 60°. -/
theorem triangle_with_60_degree_angle :
  ∃ (a b c : ℝ) (α β γ : ℝ),
    a = 4 ∧ 
    b = 2 * Real.sqrt 3 ∧ 
    c = 2 + 2 * Real.sqrt 2 ∧
    α + β + γ = π ∧
    a^2 = b^2 + c^2 - 2*b*c*Real.cos α ∧
    b^2 = a^2 + c^2 - 2*a*c*Real.cos β ∧
    c^2 = a^2 + b^2 - 2*a*b*Real.cos γ ∧
    β = π/3 :=
by sorry


end NUMINAMATH_CALUDE_triangle_with_60_degree_angle_l1936_193613


namespace NUMINAMATH_CALUDE_percentage_problem_l1936_193631

theorem percentage_problem (p : ℝ) : 
  (0.65 * 40 = p / 100 * 60 + 23) → p = 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1936_193631


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1936_193650

theorem real_part_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.re z = -1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1936_193650


namespace NUMINAMATH_CALUDE_find_d_l1936_193641

theorem find_d (A B C D : ℝ) : 
  (A + B + C) / 3 = 130 →
  (A + B + C + D) / 4 = 126 →
  D = 114 := by
sorry

end NUMINAMATH_CALUDE_find_d_l1936_193641


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1936_193628

/-- 
Given an arithmetic sequence {aₙ} with sum of first n terms Sₙ,
prove that if S₈ = 8a₅ - 4, then the common difference of the sequence is 1.
-/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- The arithmetic sequence
  (S : ℕ → ℝ) -- The sum function
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- Arithmetic sequence condition
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) -- Sum formula for arithmetic sequence
  (h_given : S 8 = 8 * a 5 - 4) -- Given condition
  : a 2 - a 1 = 1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1936_193628


namespace NUMINAMATH_CALUDE_balloon_radius_ratio_l1936_193627

theorem balloon_radius_ratio :
  ∀ (r_L r_S : ℝ),
    (4 / 3 : ℝ) * Real.pi * r_L ^ 3 = 450 * Real.pi →
    (4 / 3 : ℝ) * Real.pi * r_S ^ 3 = 36 * Real.pi →
    r_S / r_L = Real.rpow 2 (1/3) / 5 :=
by sorry

end NUMINAMATH_CALUDE_balloon_radius_ratio_l1936_193627


namespace NUMINAMATH_CALUDE_exterior_angle_theorem_l1936_193637

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)
  (sum_eq_180 : a + b + c = 180)

-- Define the problem
theorem exterior_angle_theorem (t : Triangle) 
  (h1 : t.a = 45)
  (h2 : t.b = 30) :
  180 - t.c = 75 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_theorem_l1936_193637


namespace NUMINAMATH_CALUDE_lines_perpendicular_l1936_193659

/-- Two lines with slopes that are roots of x^2 - mx - 1 = 0 are perpendicular --/
theorem lines_perpendicular (m : ℝ) (k₁ k₂ : ℝ) : 
  k₁^2 - m*k₁ - 1 = 0 → k₂^2 - m*k₂ - 1 = 0 → k₁ * k₂ = -1 := by
  sorry

#check lines_perpendicular

end NUMINAMATH_CALUDE_lines_perpendicular_l1936_193659


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1936_193646

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 3) :
  let original := (3 / (x - 1) - x - 1) / ((x^2 - 4*x + 4) / (x - 1))
  let simplified := -(x + 2) / (x - 2)
  original = simplified ∧ simplified = -5 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1936_193646


namespace NUMINAMATH_CALUDE_special_polynomial_form_l1936_193629

/-- A polynomial of two variables satisfying specific conditions -/
structure SpecialPolynomial where
  P : ℝ → ℝ → ℝ
  n : ℕ+
  homogeneous : ∀ (t x y : ℝ), P (t * x) (t * y) = t ^ n.val * P x y
  sum_condition : ∀ (a b c : ℝ), P (a + b) c + P (b + c) a + P (c + a) b = 0
  normalization : P 1 0 = 1

/-- The theorem stating the form of the special polynomial -/
theorem special_polynomial_form (sp : SpecialPolynomial) :
  ∃ (n : ℕ+), ∀ (x y : ℝ), sp.P x y = (x - 2 * y) * (x + y) ^ (n.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_form_l1936_193629


namespace NUMINAMATH_CALUDE_intersection_when_a_is_three_possible_values_of_a_l1936_193635

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 3 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x < 1 ∨ x > 6}

-- Theorem 1
theorem intersection_when_a_is_three :
  A 3 ∩ (Set.univ \ B) = {x | 1 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem possible_values_of_a (a : ℝ) :
  a > 0 ∧ A a ∩ B = ∅ → 0 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_three_possible_values_of_a_l1936_193635


namespace NUMINAMATH_CALUDE_hash_difference_l1936_193661

/-- Custom operation # defined as x#y = xy + 2x -/
def hash (x y : ℤ) : ℤ := x * y + 2 * x

/-- Theorem stating that (5#3) - (3#5) = 4 -/
theorem hash_difference : hash 5 3 - hash 3 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l1936_193661


namespace NUMINAMATH_CALUDE_competition_winners_l1936_193647

theorem competition_winners (total_winners : Nat) (total_score : Nat) 
  (first_place_score : Nat) (second_place_score : Nat) (third_place_score : Nat) :
  total_winners = 5 →
  total_score = 94 →
  first_place_score = 20 →
  second_place_score = 19 →
  third_place_score = 18 →
  ∃ (first_place_winners second_place_winners third_place_winners : Nat),
    first_place_winners = 1 ∧
    second_place_winners = 2 ∧
    third_place_winners = 2 ∧
    first_place_winners + second_place_winners + third_place_winners = total_winners ∧
    first_place_winners * first_place_score + 
    second_place_winners * second_place_score + 
    third_place_winners * third_place_score = total_score :=
by sorry

end NUMINAMATH_CALUDE_competition_winners_l1936_193647


namespace NUMINAMATH_CALUDE_rupert_candles_l1936_193607

/-- Given that Peter has 10 candles on his cake and Rupert is 3.5 times older than Peter,
    prove that Rupert's cake will have 35 candles. -/
theorem rupert_candles (peter_candles : ℕ) (age_ratio : ℚ) : ℕ :=
  by
  -- Define Peter's candles
  have h1 : peter_candles = 10 := by sorry
  -- Define the age ratio between Rupert and Peter
  have h2 : age_ratio = 3.5 := by sorry
  -- Calculate Rupert's candles
  have h3 : ↑peter_candles * age_ratio = 35 := by sorry
  -- Prove that Rupert's candles equal 35
  exact 35

end NUMINAMATH_CALUDE_rupert_candles_l1936_193607


namespace NUMINAMATH_CALUDE_multiply_y_value_l1936_193638

theorem multiply_y_value (x y : ℝ) (h1 : ∃ (n : ℝ), 5 * x = n * y) 
  (h2 : x * y ≠ 0) (h3 : (1/5 * x) / (1/6 * y) = 0.7200000000000001) : 
  ∃ (n : ℝ), 5 * x = n * y ∧ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_multiply_y_value_l1936_193638


namespace NUMINAMATH_CALUDE_frequency_calculation_l1936_193688

theorem frequency_calculation (sample_size : ℕ) (frequency_rate : ℚ) (h1 : sample_size = 1000) (h2 : frequency_rate = 0.4) :
  (sample_size : ℚ) * frequency_rate = 400 := by
  sorry

end NUMINAMATH_CALUDE_frequency_calculation_l1936_193688


namespace NUMINAMATH_CALUDE_lennon_reimbursement_l1936_193615

/-- Calculates the total reimbursement for a sales rep given daily mileage and reimbursement rate -/
def calculate_reimbursement (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) (friday : ℕ) (rate : ℚ) : ℚ :=
  (monday + tuesday + wednesday + thursday + friday : ℚ) * rate

/-- Proves that the total reimbursement for Lennon's mileage is $36 -/
theorem lennon_reimbursement :
  calculate_reimbursement 18 26 20 20 16 (36/100) = 36 := by
  sorry

end NUMINAMATH_CALUDE_lennon_reimbursement_l1936_193615


namespace NUMINAMATH_CALUDE_groupD_correct_l1936_193685

/-- Represents a group of Chinese words -/
structure WordGroup :=
  (words : List String)

/-- Checks if a word is correctly written -/
def isCorrectlyWritten (word : String) : Prop :=
  sorry -- Implementation details omitted

/-- Checks if all words in a group are correctly written -/
def allWordsCorrect (group : WordGroup) : Prop :=
  ∀ word ∈ group.words, isCorrectlyWritten word

/-- The four given groups of words -/
def groupA : WordGroup :=
  ⟨["萌孽", "青鸾", "契合", "苦思冥想", "情深意笃", "骇人听闻"]⟩

def groupB : WordGroup :=
  ⟨["斒斓", "彭觞", "木楔", "虚与委蛇", "肆无忌惮", "殒身不恤"]⟩

def groupC : WordGroup :=
  ⟨["青睐", "气概", "编辑", "呼天抢地", "轻歌慢舞", "长歌当哭"]⟩

def groupD : WordGroup :=
  ⟨["缧绁", "剌谬", "陷阱", "伶仃孤苦", "运筹帷幄", "作壁上观"]⟩

/-- Theorem stating that group D is the only group with all words correctly written -/
theorem groupD_correct :
  allWordsCorrect groupD ∧
  ¬allWordsCorrect groupA ∧
  ¬allWordsCorrect groupB ∧
  ¬allWordsCorrect groupC :=
sorry

end NUMINAMATH_CALUDE_groupD_correct_l1936_193685


namespace NUMINAMATH_CALUDE_chi_square_greater_than_critical_expected_volleyball_recipients_correct_l1936_193645

-- Define the total number of students
def total_students : ℕ := 200

-- Define the number of male and female students
def male_students : ℕ := 100
def female_students : ℕ := 100

-- Define the number of students in Group A (volleyball)
def group_a_total : ℕ := 96

-- Define the number of male students in Group A
def group_a_male : ℕ := 36

-- Define the critical value for α = 0.001
def critical_value : ℚ := 10828 / 1000

-- Define the chi-square statistic
def chi_square : ℚ := 11538 / 1000

-- Define the number of male students selected for stratified sampling
def stratified_sample : ℕ := 25

-- Define the number of students selected for gifts
def gift_recipients : ℕ := 3

-- Define the expected number of volleyball players among gift recipients
def expected_volleyball_recipients : ℚ := 621 / 575

-- Theorem 1: The chi-square value is greater than the critical value
theorem chi_square_greater_than_critical : chi_square > critical_value := by sorry

-- Theorem 2: The expected number of volleyball players among gift recipients is correct
theorem expected_volleyball_recipients_correct : 
  expected_volleyball_recipients = 621 / 575 := by sorry

end NUMINAMATH_CALUDE_chi_square_greater_than_critical_expected_volleyball_recipients_correct_l1936_193645


namespace NUMINAMATH_CALUDE_earliest_solution_l1936_193679

/-- The temperature model as a function of time -/
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 60

/-- The theorem stating that 12.5 is the earliest non-negative solution -/
theorem earliest_solution :
  ∀ t : ℝ, t ≥ 0 → temperature t = 85 → t ≥ 12.5 := by sorry

end NUMINAMATH_CALUDE_earliest_solution_l1936_193679


namespace NUMINAMATH_CALUDE_sin_double_angle_circle_l1936_193657

theorem sin_double_angle_circle (α : Real) :
  let P : ℝ × ℝ := (1, 2)
  let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
  (P.1^2 + P.2^2 = r^2) →  -- Point P is on the circle
  (P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α) →  -- P is on the terminal side of α
  Real.sin (2 * α) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_circle_l1936_193657


namespace NUMINAMATH_CALUDE_rock_mist_distance_l1936_193694

/-- The distance from the city to Sky Falls in miles -/
def distance_to_sky_falls : ℝ := 8

/-- The factor by which Rock Mist Mountains are farther from the city than Sky Falls -/
def rock_mist_factor : ℝ := 50

/-- The distance from the city to Rock Mist Mountains in miles -/
def distance_to_rock_mist : ℝ := distance_to_sky_falls * rock_mist_factor

theorem rock_mist_distance : distance_to_rock_mist = 400 := by
  sorry

end NUMINAMATH_CALUDE_rock_mist_distance_l1936_193694


namespace NUMINAMATH_CALUDE_binomial_coefficient_modulo_prime_l1936_193663

theorem binomial_coefficient_modulo_prime (p : ℕ) (hp : Nat.Prime p) : 
  (Nat.choose (2 * p) p) ≡ 2 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_modulo_prime_l1936_193663


namespace NUMINAMATH_CALUDE_die_faces_count_l1936_193614

-- Define the probability of all five dice showing the same number
def probability : ℝ := 0.0007716049382716049

-- Define the number of dice
def num_dice : ℕ := 5

-- Theorem: The number of faces on each die is 10
theorem die_faces_count : 
  ∃ (n : ℕ), n > 0 ∧ (1 : ℝ) / n ^ num_dice = probability ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_die_faces_count_l1936_193614


namespace NUMINAMATH_CALUDE_sequence_properties_l1936_193662

def S (n : ℕ) : ℤ := n^2 - 9*n

def a (n : ℕ) : ℤ := 2*n - 10

theorem sequence_properties :
  (∀ n : ℕ, S (n+1) - S n = a (n+1)) ∧
  (∃! k : ℕ, k > 0 ∧ 5 < a k ∧ a k < 8 ∧ k = 8) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1936_193662


namespace NUMINAMATH_CALUDE_rectangle_area_l1936_193686

theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 5 →
  length = 2 * width →
  area = length * width →
  area = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1936_193686


namespace NUMINAMATH_CALUDE_adult_ticket_price_l1936_193602

/-- Proves that the price of an adult ticket is $15 given the conditions of the problem -/
theorem adult_ticket_price (total_cost : ℕ) (child_ticket_price : ℕ) (num_children : ℕ) :
  total_cost = 720 →
  child_ticket_price = 8 →
  num_children = 15 →
  ∃ (adult_ticket_price : ℕ),
    adult_ticket_price * (num_children + 25) + child_ticket_price * num_children = total_cost ∧
    adult_ticket_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l1936_193602


namespace NUMINAMATH_CALUDE_tree_height_difference_l1936_193691

/-- The height of the birch tree in feet -/
def birch_height : ℚ := 49/4

/-- The height of the pine tree in feet -/
def pine_height : ℚ := 37/2

/-- The difference in height between the pine tree and the birch tree -/
def height_difference : ℚ := pine_height - birch_height

theorem tree_height_difference : height_difference = 25/4 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l1936_193691


namespace NUMINAMATH_CALUDE_quinary_324_equals_binary_1011001_l1936_193696

/-- Converts a quinary (base-5) number to decimal --/
def quinary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldr (fun ⟨i, d⟩ acc => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to binary --/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem quinary_324_equals_binary_1011001 :
  decimal_to_binary (quinary_to_decimal [4, 2, 3]) = [1, 0, 1, 1, 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_quinary_324_equals_binary_1011001_l1936_193696


namespace NUMINAMATH_CALUDE_sum_of_squares_l1936_193690

theorem sum_of_squares (a b : ℝ) : (a^2 + b^2) * (a^2 + b^2 - 4) - 5 = 0 → a^2 + b^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1936_193690


namespace NUMINAMATH_CALUDE_calculate_expression_l1936_193621

theorem calculate_expression : 
  2 * Real.tan (60 * π / 180) - (-2023)^(0 : ℝ) + (1/2)^(-1 : ℝ) + |Real.sqrt 3 - 1| = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1936_193621


namespace NUMINAMATH_CALUDE_quilt_block_shaded_half_l1936_193622

/-- Represents a square quilt block divided into a 4x4 grid -/
structure QuiltBlock where
  grid_size : Nat
  full_shaded : Nat
  half_shaded : Nat

/-- Calculates the fraction of shaded area in a quilt block -/
def shaded_fraction (qb : QuiltBlock) : Rat :=
  (qb.full_shaded + qb.half_shaded / 2) / (qb.grid_size * qb.grid_size)

theorem quilt_block_shaded_half :
  ∀ qb : QuiltBlock,
    qb.grid_size = 4 →
    qb.full_shaded = 6 →
    qb.half_shaded = 4 →
    shaded_fraction qb = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quilt_block_shaded_half_l1936_193622


namespace NUMINAMATH_CALUDE_profit_increase_calculation_l1936_193669

/-- Proves that given a 40% increase followed by a 20% decrease, 
    a final increase that results in an overall 68% increase must be a 50% increase. -/
theorem profit_increase_calculation (P : ℝ) (h : P > 0) : 
  let april_profit := 1.40 * P
  let may_profit := 0.80 * april_profit
  let june_profit := 1.68 * P
  (june_profit / may_profit - 1) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_calculation_l1936_193669


namespace NUMINAMATH_CALUDE_max_two_digit_composite_relatively_prime_l1936_193699

/-- A number is two-digit if it's between 10 and 99 inclusive -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A number is composite if it has a factor other than 1 and itself -/
def isComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- Two numbers are relatively prime if their greatest common divisor is 1 -/
def areRelativelyPrime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The set of numbers satisfying our conditions -/
def validSet (S : Finset ℕ) : Prop :=
  ∀ n ∈ S, isTwoDigit n ∧ isComposite n ∧
  ∀ m ∈ S, m ≠ n → areRelativelyPrime m n

theorem max_two_digit_composite_relatively_prime :
  (∃ S : Finset ℕ, validSet S ∧ S.card = 4) ∧
  ∀ T : Finset ℕ, validSet T → T.card ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_two_digit_composite_relatively_prime_l1936_193699


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l1936_193634

theorem muffin_banana_price_ratio :
  ∀ (m b : ℝ),
  (3 * m + 5 * b > 0) →
  (4 * m + 10 * b = 3 * m + 5 * b + 12) →
  (m / b = 2) :=
by sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l1936_193634


namespace NUMINAMATH_CALUDE_xiaoqiang_games_l1936_193682

/-- Represents a participant in the chess tournament -/
inductive Participant
  | Jia
  | Yi
  | Bing
  | Ding
  | Xiaoqiang

/-- The number of games played by each participant -/
def games_played (p : Participant) : ℕ :=
  match p with
  | Participant.Jia => 4
  | Participant.Yi => 3
  | Participant.Bing => 2
  | Participant.Ding => 1
  | Participant.Xiaoqiang => 2  -- This is what we want to prove

/-- The total number of games played in the tournament -/
def total_games : ℕ := 10  -- (5 choose 2) = 10

theorem xiaoqiang_games :
  games_played Participant.Xiaoqiang = 2 :=
by sorry

end NUMINAMATH_CALUDE_xiaoqiang_games_l1936_193682


namespace NUMINAMATH_CALUDE_mean_of_additional_numbers_l1936_193608

theorem mean_of_additional_numbers
  (original_count : Nat)
  (original_mean : ℝ)
  (new_count : Nat)
  (new_mean : ℝ)
  (h1 : original_count = 7)
  (h2 : original_mean = 72)
  (h3 : new_count = 9)
  (h4 : new_mean = 80) :
  let x_plus_y := new_count * new_mean - original_count * original_mean
  let mean_x_y := x_plus_y / 2
  mean_x_y = 108 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_additional_numbers_l1936_193608


namespace NUMINAMATH_CALUDE_age_difference_l1936_193649

theorem age_difference (a b c : ℕ) : 
  b = 12 →
  b = 2 * c →
  a + b + c = 32 →
  a = b + 2 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1936_193649


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l1936_193681

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l1936_193681


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l1936_193643

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a number is the first prime after 7 consecutive non-primes
def isFirstPrimeAfter7NonPrimes (p : ℕ) : Prop :=
  isPrime p ∧
  ∀ k : ℕ, k ∈ Finset.range 7 → ¬isPrime (p - k - 1) ∧
  ∀ q : ℕ, q < p → isFirstPrimeAfter7NonPrimes q → False

-- State the theorem
theorem smallest_prime_after_seven_nonprimes :
  isFirstPrimeAfter7NonPrimes 97 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l1936_193643


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l1936_193676

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ is_perfect_square (n + reverse_digits n)

theorem two_digit_reverse_sum_square :
  {n : ℕ | satisfies_condition n} = {29, 38, 47, 56, 65, 74, 83, 92} :=
by sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l1936_193676


namespace NUMINAMATH_CALUDE_sum_x_y_l1936_193605

theorem sum_x_y (x y : ℝ) 
  (h1 : |x| + x + y - 2 = 14)
  (h2 : x + |y| - y + 3 = 20) : 
  x + y = 31/5 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_l1936_193605


namespace NUMINAMATH_CALUDE_chloe_profit_l1936_193626

/-- Calculates the profit from selling chocolate-dipped strawberries -/
def calculate_profit (buy_price_per_dozen : ℕ) (sell_price_per_half_dozen : ℕ) (dozens_sold : ℕ) : ℕ :=
  let cost := buy_price_per_dozen * dozens_sold
  let revenue := sell_price_per_half_dozen * 2 * dozens_sold
  revenue - cost

/-- Proves that Chloe's profit is $500 given the specified conditions -/
theorem chloe_profit :
  calculate_profit 50 30 50 = 500 := by
  sorry

end NUMINAMATH_CALUDE_chloe_profit_l1936_193626


namespace NUMINAMATH_CALUDE_target_miss_probability_l1936_193618

theorem target_miss_probability (p_I p_II p_III : ℝ) 
  (h_I : p_I = 0.35) 
  (h_II : p_II = 0.30) 
  (h_III : p_III = 0.25) : 
  1 - (p_I + p_II + p_III) = 0.10 := by
  sorry

end NUMINAMATH_CALUDE_target_miss_probability_l1936_193618


namespace NUMINAMATH_CALUDE_quaternary_2132_equals_septenary_314_l1936_193633

/-- Converts a quaternary (base 4) number represented as a list of digits to decimal (base 10) -/
def quaternary_to_decimal (digits : List Nat) : Nat := sorry

/-- Converts a decimal (base 10) number to septenary (base 7) represented as a list of digits -/
def decimal_to_septenary (n : Nat) : List Nat := sorry

/-- Theorem stating that the quaternary number 2132 is equal to the septenary number 314 -/
theorem quaternary_2132_equals_septenary_314 :
  decimal_to_septenary (quaternary_to_decimal [2, 1, 3, 2]) = [3, 1, 4] := by sorry

end NUMINAMATH_CALUDE_quaternary_2132_equals_septenary_314_l1936_193633


namespace NUMINAMATH_CALUDE_square_of_sum_l1936_193695

theorem square_of_sum (x y : ℝ) 
  (h1 : x * (2 * x + y) = 36)
  (h2 : y * (2 * x + y) = 72) : 
  (2 * x + y)^2 = 108 := by
sorry

end NUMINAMATH_CALUDE_square_of_sum_l1936_193695


namespace NUMINAMATH_CALUDE_multiples_of_seven_between_50_and_150_l1936_193642

theorem multiples_of_seven_between_50_and_150 :
  (Finset.filter (fun n => 50 ≤ 7 * n ∧ 7 * n ≤ 150) (Finset.range (150 / 7 + 1))).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_seven_between_50_and_150_l1936_193642


namespace NUMINAMATH_CALUDE_possible_values_of_C_l1936_193625

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The first number in the problem -/
def number1 (A B : Digit) : ℕ := 9100000 + 10000 * A.val + 300 + 10 * B.val + 2

/-- The second number in the problem -/
def number2 (A B C : Digit) : ℕ := 6000000 + 100000 * A.val + 10000 * B.val + 400 + 50 + 10 * C.val + 2

/-- Theorem stating the possible values of C -/
theorem possible_values_of_C :
  ∀ (A B C : Digit),
    (∃ k : ℕ, number1 A B = 3 * k) →
    (∃ m : ℕ, number2 A B C = 5 * m) →
    C.val = 0 ∨ C.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_C_l1936_193625


namespace NUMINAMATH_CALUDE_sequence_sum_l1936_193698

def geometric_sequence (a : ℕ → ℚ) (r : ℚ) :=
  ∀ n, a (n + 1) = a n * r

theorem sequence_sum (a : ℕ → ℚ) (r : ℚ) :
  geometric_sequence a r →
  a 6 = 4 →
  a 7 = 1 →
  a 4 + a 5 = 80 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1936_193698


namespace NUMINAMATH_CALUDE_inequality_theorem_largest_c_k_zero_largest_c_k_two_l1936_193668

theorem inequality_theorem :
  ∀ (k : ℝ), 
  (∃ (c_k : ℝ), c_k > 0 ∧ 
    (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 
      (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c_k * (x + y + z)^k)) ↔ 
  (0 ≤ k ∧ k ≤ 2) :=
sorry

theorem largest_c_k_zero :
  ∀ (c : ℝ), c > 0 →
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c) →
  c ≤ 1 :=
sorry

theorem largest_c_k_two :
  ∀ (c : ℝ), c > 0 →
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c * (x + y + z)^2) →
  c ≤ 8/9 :=
sorry

end NUMINAMATH_CALUDE_inequality_theorem_largest_c_k_zero_largest_c_k_two_l1936_193668


namespace NUMINAMATH_CALUDE_problem_solution_l1936_193639

noncomputable section

def f (x : ℝ) : ℝ := Real.log x

def g (m n x : ℝ) : ℝ := m * (x + n) / (x + 1)

def tangent_perpendicular (n : ℝ) : Prop :=
  let f' : ℝ → ℝ := λ x => 1 / x
  let g' : ℝ → ℝ := λ x => (1 - n) / ((x + 1) ^ 2)
  f' 1 * g' 1 = -1

def inequality_holds (m n : ℝ) : Prop :=
  ∀ x > 0, |f x| ≥ |g m n x|

theorem problem_solution :
  (∃ n : ℝ, tangent_perpendicular n ∧ n = 5) ∧
  (∃ n : ℝ, ∃ m : ℝ, m > 0 ∧ inequality_holds m n ∧ n = -1 ∧
    (∀ m' > 0, inequality_holds m' n → m' ≤ m) ∧ m = 2) := by sorry

end

end NUMINAMATH_CALUDE_problem_solution_l1936_193639


namespace NUMINAMATH_CALUDE_triangle_equilateral_l1936_193624

theorem triangle_equilateral (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^4 = b^4 + c^4 - b^2*c^2)
  (h5 : b^4 = c^4 + a^4 - a^2*c^2) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l1936_193624


namespace NUMINAMATH_CALUDE_original_selling_price_l1936_193636

theorem original_selling_price 
  (P : ℝ) -- Original purchase price
  (S : ℝ) -- Original selling price
  (S_new : ℝ) -- New selling price
  (h1 : S = 1.1 * P) -- Original selling price is 110% of purchase price
  (h2 : S_new = 1.17 * P) -- New selling price based on 10% lower purchase and 30% profit
  (h3 : S_new - S = 63) -- Difference between new and original selling price is $63
  : S = 990 := by
sorry

end NUMINAMATH_CALUDE_original_selling_price_l1936_193636


namespace NUMINAMATH_CALUDE_expected_value_is_eight_l1936_193609

/-- The number of marbles in the bag -/
def n : ℕ := 7

/-- The set of marble numbers -/
def marbles : Finset ℕ := Finset.range n

/-- The sum of all possible pairs of marbles -/
def sum_of_pairs : ℕ := (marbles.powerset.filter (fun s => s.card = 2)).sum (fun s => s.sum id)

/-- The number of ways to choose 2 marbles out of n -/
def num_combinations : ℕ := n.choose 2

/-- The expected value of the sum of two randomly drawn marbles -/
def expected_value : ℚ := (sum_of_pairs : ℚ) / num_combinations

theorem expected_value_is_eight : expected_value = 8 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_eight_l1936_193609


namespace NUMINAMATH_CALUDE_prob_two_defective_consignment_l1936_193664

/-- Represents a consignment of picture tubes -/
structure Consignment where
  total : ℕ
  defective : ℕ
  h_defective_le_total : defective ≤ total

/-- Calculates the probability of selecting two defective tubes without replacement -/
def prob_two_defective (c : Consignment) : ℚ :=
  (c.defective : ℚ) / (c.total : ℚ) * ((c.defective - 1) : ℚ) / ((c.total - 1) : ℚ)

theorem prob_two_defective_consignment :
  let c : Consignment := ⟨20, 5, by norm_num⟩
  prob_two_defective c = 1 / 19 := by sorry

end NUMINAMATH_CALUDE_prob_two_defective_consignment_l1936_193664


namespace NUMINAMATH_CALUDE_insufficient_evidence_l1936_193673

/-- Represents the data from a 2x2 contingency table --/
structure ContingencyTable :=
  (irregular_disease : Nat)
  (irregular_no_disease : Nat)
  (regular_disease : Nat)
  (regular_no_disease : Nat)

/-- Represents the result of a statistical test --/
inductive TestResult
  | Significant
  | NotSignificant

/-- Performs a statistical test on the contingency table data --/
def statisticalTest (data : ContingencyTable) : TestResult :=
  sorry

/-- Theorem stating that the given survey data does not provide sufficient evidence
    for a relationship between stomach diseases and living habits --/
theorem insufficient_evidence (survey_data : ContingencyTable) 
  (h1 : survey_data.irregular_disease = 5)
  (h2 : survey_data.irregular_no_disease = 15)
  (h3 : survey_data.regular_disease = 40)
  (h4 : survey_data.regular_no_disease = 10) :
  statisticalTest survey_data = TestResult.NotSignificant :=
sorry

end NUMINAMATH_CALUDE_insufficient_evidence_l1936_193673


namespace NUMINAMATH_CALUDE_minimum_sum_l1936_193611

theorem minimum_sum (x y z : ℝ) (h : (4 / x) + (2 / y) + (1 / z) = 1) :
  x + 8 * y + 4 * z ≥ 64 ∧
  (x + 8 * y + 4 * z = 64 ↔ x = 16 ∧ y = 4 ∧ z = 4) := by
  sorry

end NUMINAMATH_CALUDE_minimum_sum_l1936_193611


namespace NUMINAMATH_CALUDE_partition_uniqueness_l1936_193603

/-- An arithmetic progression is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticProgression (a : ℤ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℤ, a (n + 1) = a n + d

/-- A set of integers X can be partitioned into N disjoint increasing
    arithmetic progressions. -/
def CanBePartitioned (X : Set ℤ) (N : ℕ) : Prop :=
  ∃ (partitions : Fin N → Set ℤ),
    (∀ i : Fin N, ∃ a : ℤ → ℤ, ArithmeticProgression a ∧ partitions i = Set.range a) ∧
    (∀ i j : Fin N, i ≠ j → partitions i ∩ partitions j = ∅) ∧
    (⋃ i : Fin N, partitions i) = X

/-- X cannot be partitioned into fewer than N arithmetic progressions. -/
def MinimalPartition (X : Set ℤ) (N : ℕ) : Prop :=
  CanBePartitioned X N ∧ ∀ k < N, ¬CanBePartitioned X k

/-- The partition of X into N arithmetic progressions is unique. -/
def UniquePartition (X : Set ℤ) (N : ℕ) : Prop :=
  ∀ p₁ p₂ : Fin N → Set ℤ,
    (∀ i : Fin N, ∃ a : ℤ → ℤ, ArithmeticProgression a ∧ p₁ i = Set.range a) →
    (∀ i : Fin N, ∃ a : ℤ → ℤ, ArithmeticProgression a ∧ p₂ i = Set.range a) →
    (∀ i j : Fin N, i ≠ j → p₁ i ∩ p₁ j = ∅) →
    (∀ i j : Fin N, i ≠ j → p₂ i ∩ p₂ j = ∅) →
    (⋃ i : Fin N, p₁ i) = X →
    (⋃ i : Fin N, p₂ i) = X →
    ∀ i : Fin N, ∃ j : Fin N, p₁ i = p₂ j

theorem partition_uniqueness (X : Set ℤ) :
  (∀ N : ℕ, MinimalPartition X N → (N = 2 → UniquePartition X N) ∧ (N = 3 → ¬UniquePartition X N)) := by
  sorry

end NUMINAMATH_CALUDE_partition_uniqueness_l1936_193603


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l1936_193655

theorem solve_system_of_equations (a b m : ℤ) 
  (eq1 : a - b = 6)
  (eq2 : 2 * a + b = m)
  (opposite : a + b = 0) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l1936_193655


namespace NUMINAMATH_CALUDE_expenditure_is_negative_l1936_193604

/-- Represents the recording of a monetary transaction -/
inductive MonetaryRecord
| Income (amount : ℤ)
| Expenditure (amount : ℤ)

/-- Converts a MonetaryRecord to its signed integer representation -/
def toSignedAmount (record : MonetaryRecord) : ℤ :=
  match record with
  | MonetaryRecord.Income a => a
  | MonetaryRecord.Expenditure a => -a

theorem expenditure_is_negative (income_amount expenditure_amount : ℤ) 
  (h : toSignedAmount (MonetaryRecord.Income income_amount) = income_amount) :
  toSignedAmount (MonetaryRecord.Expenditure expenditure_amount) = -expenditure_amount := by
  sorry

end NUMINAMATH_CALUDE_expenditure_is_negative_l1936_193604


namespace NUMINAMATH_CALUDE_leadership_combinations_count_l1936_193623

def tribe_size : ℕ := 15
def num_supporting_chiefs : ℕ := 3
def num_inferior_officers_per_chief : ℕ := 2

def leadership_combinations : ℕ := 
  tribe_size * 
  (tribe_size - 1) * 
  (tribe_size - 2) * 
  (tribe_size - 3) * 
  Nat.choose (tribe_size - 4) num_inferior_officers_per_chief *
  Nat.choose (tribe_size - 6) num_inferior_officers_per_chief *
  Nat.choose (tribe_size - 8) num_inferior_officers_per_chief

theorem leadership_combinations_count : leadership_combinations = 19320300 := by
  sorry

end NUMINAMATH_CALUDE_leadership_combinations_count_l1936_193623


namespace NUMINAMATH_CALUDE_probability_3_successes_in_7_trials_value_l1936_193648

/-- The probability of getting exactly 3 successes in 7 independent trials,
    where the probability of success in each trial is 3/7. -/
def probability_3_successes_in_7_trials : ℚ :=
  (Nat.choose 7 3 : ℚ) * (3/7)^3 * (4/7)^4

/-- Theorem stating that the probability of getting exactly 3 successes
    in 7 independent trials, where the probability of success in each trial
    is 3/7, is equal to 242112/823543. -/
theorem probability_3_successes_in_7_trials_value :
  probability_3_successes_in_7_trials = 242112/823543 := by
  sorry

end NUMINAMATH_CALUDE_probability_3_successes_in_7_trials_value_l1936_193648


namespace NUMINAMATH_CALUDE_new_average_age_l1936_193644

theorem new_average_age (n : ℕ) (initial_avg : ℝ) (new_person_age : ℝ) : 
  n = 9 → initial_avg = 14 → new_person_age = 34 → 
  ((n : ℝ) * initial_avg + new_person_age) / ((n : ℝ) + 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l1936_193644


namespace NUMINAMATH_CALUDE_determinant_equality_l1936_193683

theorem determinant_equality (x y z w : ℝ) : 
  x * w - y * z = 7 → (x + z) * w - (y + 2 * w) * z = 7 - w * z := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l1936_193683


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1936_193612

/-- Given a sphere with surface area 256π cm², its volume is 2048/3 π cm³. -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * π * r^2 = 256 * π → (4 / 3) * π * r^3 = (2048 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1936_193612


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l1936_193677

-- Define a function f that satisfies the symmetry condition
def f (x : ℝ) : ℝ := sorry

-- State the symmetry condition
axiom symmetry_condition : ∀ x, f x = f (4 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem statement
theorem axis_of_symmetry :
  is_axis_of_symmetry 2 :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l1936_193677


namespace NUMINAMATH_CALUDE_smallest_k_for_divisibility_property_l1936_193653

theorem smallest_k_for_divisibility_property (n : ℕ) :
  let M := Finset.range n
  (∃ k : ℕ, k > 0 ∧
    (∀ S : Finset ℕ, S ⊆ M → S.card = k →
      ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b) ∧
    (∀ k' : ℕ, k' < k →
      ∃ S : Finset ℕ, S ⊆ M ∧ S.card = k' ∧
        ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(a ∣ b))) →
  let k := ⌈(n : ℚ) / 2⌉₊ + 1
  (∀ S : Finset ℕ, S ⊆ M → S.card = k →
    ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b) ∧
  (∀ k' : ℕ, k' < k →
    ∃ S : Finset ℕ, S ⊆ M ∧ S.card = k' ∧
      ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(a ∣ b)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_divisibility_property_l1936_193653


namespace NUMINAMATH_CALUDE_caroline_lassi_production_caroline_lassi_production_proof_l1936_193697

/-- Given that Caroline can make 7 lassis from 3 mangoes, 
    prove that she can make 35 lassis from 15 mangoes. -/
theorem caroline_lassi_production : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mangoes_small lassis_small mangoes_large lassis_large =>
    mangoes_small = 3 ∧ 
    lassis_small = 7 ∧ 
    mangoes_large = 15 ∧
    lassis_large = 35 ∧
    (mangoes_large * lassis_small = mangoes_small * lassis_large) →
    lassis_large = (mangoes_large * lassis_small) / mangoes_small

theorem caroline_lassi_production_proof : 
  caroline_lassi_production 3 7 15 35 := by
  sorry

end NUMINAMATH_CALUDE_caroline_lassi_production_caroline_lassi_production_proof_l1936_193697


namespace NUMINAMATH_CALUDE_parabola_properties_l1936_193656

/-- A parabola with equation y^2 = 2px and focus at (1,0) -/
structure Parabola where
  p : ℝ
  focus_x : ℝ
  focus_y : ℝ
  h_focus : (focus_x, focus_y) = (1, 0)

/-- The value of p for the parabola -/
def p_value (par : Parabola) : ℝ := par.p

/-- The equation of the directrix for the parabola -/
def directrix_equation (par : Parabola) : ℝ → Prop := fun x ↦ x = -1

theorem parabola_properties (par : Parabola) :
  p_value par = 2 ∧ directrix_equation par = fun x ↦ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1936_193656


namespace NUMINAMATH_CALUDE_lower_profit_percentage_l1936_193675

/-- Proves that given an article with a cost price of $800, if the profit at 18% is $72 more than the profit at another percentage, then that other percentage is 9%. -/
theorem lower_profit_percentage (cost_price : ℝ) (higher_percentage lower_percentage : ℝ) : 
  cost_price = 800 →
  higher_percentage = 18 →
  (higher_percentage / 100) * cost_price = (lower_percentage / 100) * cost_price + 72 →
  lower_percentage = 9 := by
  sorry

end NUMINAMATH_CALUDE_lower_profit_percentage_l1936_193675


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l1936_193666

theorem min_value_of_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 :=
by sorry

theorem equality_achieved : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = 2023 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l1936_193666


namespace NUMINAMATH_CALUDE_complex_arithmetic_l1936_193606

/-- Given complex numbers B, Q, R, and T, prove that 2(B - Q + R + T) = 18 + 10i -/
theorem complex_arithmetic (B Q R T : ℂ) 
  (hB : B = 3 + 2*I)
  (hQ : Q = -5)
  (hR : R = -2*I)
  (hT : T = 1 + 5*I) :
  2 * (B - Q + R + T) = 18 + 10*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l1936_193606


namespace NUMINAMATH_CALUDE_lawn_mowing_donation_percentage_l1936_193670

/-- Proves that the percentage of lawn mowing proceeds donated to the shelter is 100% --/
theorem lawn_mowing_donation_percentage
  (carwash_earnings : ℝ)
  (carwash_donation_rate : ℝ)
  (bake_sale_earnings : ℝ)
  (bake_sale_donation_rate : ℝ)
  (lawn_mowing_earnings : ℝ)
  (total_donation : ℝ)
  (h1 : carwash_earnings = 100)
  (h2 : carwash_donation_rate = 0.9)
  (h3 : bake_sale_earnings = 80)
  (h4 : bake_sale_donation_rate = 0.75)
  (h5 : lawn_mowing_earnings = 50)
  (h6 : total_donation = 200)
  (h7 : total_donation = carwash_earnings * carwash_donation_rate +
                         bake_sale_earnings * bake_sale_donation_rate +
                         lawn_mowing_earnings * (lawn_mowing_donation / lawn_mowing_earnings)) :
  lawn_mowing_donation / lawn_mowing_earnings = 1 :=
by
  sorry

#check lawn_mowing_donation_percentage

end NUMINAMATH_CALUDE_lawn_mowing_donation_percentage_l1936_193670


namespace NUMINAMATH_CALUDE_circle_y_is_eleven_l1936_193652

/-- Represents the configuration of numbers in the circles. -/
structure CircleConfig where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  x : ℤ
  y : ℤ

/-- The conditions given in the problem. -/
def satisfiesConditions (config : CircleConfig) : Prop :=
  config.a + config.b + config.x = 30 ∧
  config.c + config.d + config.y = 30 ∧
  config.a + config.b + config.c + config.d = 40 ∧
  config.x + config.y + config.c + config.b = 40 ∧
  config.x = 9

/-- The theorem stating that if the conditions are satisfied, Y must be 11. -/
theorem circle_y_is_eleven (config : CircleConfig) 
  (h : satisfiesConditions config) : config.y = 11 := by
  sorry


end NUMINAMATH_CALUDE_circle_y_is_eleven_l1936_193652


namespace NUMINAMATH_CALUDE_smallest_taxicab_number_is_smallest_l1936_193680

/-- The smallest positive integer that can be expressed as the sum of two cubes in two different ways -/
def smallest_taxicab_number : ℕ := 1729

/-- A function that checks if a number can be expressed as the sum of two cubes in two different ways -/
def is_taxicab (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a < c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a^3 + b^3 = n ∧ c^3 + d^3 = n ∧ (a, b) ≠ (c, d)

/-- Theorem stating that smallest_taxicab_number is indeed the smallest taxicab number -/
theorem smallest_taxicab_number_is_smallest :
  is_taxicab smallest_taxicab_number ∧
  ∀ m : ℕ, m < smallest_taxicab_number → ¬is_taxicab m :=
sorry

end NUMINAMATH_CALUDE_smallest_taxicab_number_is_smallest_l1936_193680


namespace NUMINAMATH_CALUDE_weaver_output_increase_l1936_193640

theorem weaver_output_increase (first_day_output : ℝ) (total_days : ℕ) (total_output : ℝ) :
  first_day_output = 5 ∧ total_days = 30 ∧ total_output = 390 →
  ∃ (daily_increase : ℝ),
    daily_increase = 16/29 ∧
    total_output = total_days * first_day_output + (total_days * (total_days - 1) / 2) * daily_increase :=
by sorry

end NUMINAMATH_CALUDE_weaver_output_increase_l1936_193640


namespace NUMINAMATH_CALUDE_projectile_max_height_l1936_193617

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 120 * t + 36

/-- The time at which the maximum height occurs -/
def t_max : ℝ := 3

/-- The maximum height reached by the projectile -/
def h_max : ℝ := 216

theorem projectile_max_height :
  (∀ t, h t ≤ h_max) ∧ h t_max = h_max := by sorry

end NUMINAMATH_CALUDE_projectile_max_height_l1936_193617


namespace NUMINAMATH_CALUDE_factorial_squared_ge_power_l1936_193678

theorem factorial_squared_ge_power (n : ℕ) (h : n ≥ 1) : (n!)^2 ≥ n^n := by
  sorry

end NUMINAMATH_CALUDE_factorial_squared_ge_power_l1936_193678


namespace NUMINAMATH_CALUDE_train_length_l1936_193658

/-- The length of a train given specific conditions --/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 200 →
  passing_time = 41 →
  (train_speed - jogger_speed) * passing_time - initial_distance = 210 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1936_193658


namespace NUMINAMATH_CALUDE_no_solution_iff_m_equals_one_l1936_193671

theorem no_solution_iff_m_equals_one :
  ∀ m : ℝ, (∀ x : ℝ, x ≠ 3 → ((3 - 2*x) / (x - 3) - (m*x - 2) / (3 - x) ≠ -1)) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_equals_one_l1936_193671


namespace NUMINAMATH_CALUDE_no_integer_square_root_product_l1936_193672

theorem no_integer_square_root_product (n1 n2 : ℤ) : 
  (n1 : ℚ) / n2 = 3 / 4 →
  n1 + n2 = 21 →
  n2 > n1 →
  ¬ ∃ (n3 : ℤ), n1 * n2 = n3^2 := by
sorry

end NUMINAMATH_CALUDE_no_integer_square_root_product_l1936_193672


namespace NUMINAMATH_CALUDE_f_one_geq_25_l1936_193630

/-- A function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The property that f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f m x ≤ f m y

theorem f_one_geq_25 (m : ℝ) (h : is_increasing_on_interval m) : f m 1 ≥ 25 :=
sorry

end NUMINAMATH_CALUDE_f_one_geq_25_l1936_193630


namespace NUMINAMATH_CALUDE_ascending_order_abc_l1936_193620

theorem ascending_order_abc :
  let a := Real.log 5 / Real.log 0.6
  let b := 2 ^ (4/5 : ℝ)
  let c := Real.sin 1
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ascending_order_abc_l1936_193620


namespace NUMINAMATH_CALUDE_tetrahedron_volume_formula_l1936_193632

/-- A tetrahedron with its properties -/
structure Tetrahedron where
  S : ℝ  -- Surface area
  R : ℝ  -- Radius of inscribed sphere
  V : ℝ  -- Volume

/-- Theorem: The volume of a tetrahedron is one-third the product of its surface area and the radius of its inscribed sphere -/
theorem tetrahedron_volume_formula (t : Tetrahedron) : t.V = (1/3) * t.S * t.R := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_formula_l1936_193632


namespace NUMINAMATH_CALUDE_divisibility_problem_l1936_193654

theorem divisibility_problem (N : ℕ) : 
  N % 44 = 0 → N % 39 = 15 → N / 44 = 3 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1936_193654


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l1936_193674

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (d : ℚ), ∀ (n : ℕ), a (n + 1) = a n + d

theorem arithmetic_sequence_common_ratio
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  ∃ (q : ℚ), q = 1/2 ∧ ∀ (n : ℕ), a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l1936_193674


namespace NUMINAMATH_CALUDE_complex_cube_equation_l1936_193616

theorem complex_cube_equation (a b : ℕ+) :
  (↑a + ↑b * Complex.I) ^ 3 = (2 : ℂ) + 11 * Complex.I →
  ↑a + ↑b * Complex.I = (2 : ℂ) + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_equation_l1936_193616


namespace NUMINAMATH_CALUDE_continued_fraction_equation_solution_l1936_193684

def continued_fraction (a : ℕ → ℕ) (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (a n : ℚ)⁻¹ + continued_fraction a (n-1)

def left_side (n : ℕ) : ℚ :=
  1 - continued_fraction (fun i => i + 1) n

def right_side (x : ℕ → ℕ) (n : ℕ) : ℚ :=
  continued_fraction x n

theorem continued_fraction_equation_solution (n : ℕ) (h : n ≥ 2) :
  ∃! x : ℕ → ℕ, left_side n = right_side x n ∧
    x 1 = 1 ∧ x 2 = 1 ∧ ∀ i, 3 ≤ i → i ≤ n → x i = i :=
sorry

end NUMINAMATH_CALUDE_continued_fraction_equation_solution_l1936_193684


namespace NUMINAMATH_CALUDE_complex_fraction_magnitude_l1936_193651

theorem complex_fraction_magnitude (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_magnitude_l1936_193651
