import Mathlib

namespace min_value_theorem_equality_condition_l1937_193743

theorem min_value_theorem (a : ℝ) (h : a > 2) : a + 4 / (a - 2) ≥ 6 :=
sorry

theorem equality_condition (a : ℝ) (h : a > 2) : 
  ∃ a₀ > 2, a₀ + 4 / (a₀ - 2) = 6 :=
sorry

end min_value_theorem_equality_condition_l1937_193743


namespace supplementary_angles_ratio_l1937_193777

theorem supplementary_angles_ratio (angle1 angle2 : ℝ) : 
  angle1 + angle2 = 180 →  -- angles are supplementary
  angle1 = 4 * angle2 →    -- angles are in ratio 4:1
  angle2 = 36 :=           -- smaller angle is 36°
by sorry

end supplementary_angles_ratio_l1937_193777


namespace phantom_needs_more_money_l1937_193759

/-- The amount of additional money Phantom needs to buy printer inks -/
def additional_money_needed (initial_money : ℚ) 
  (black_price red_price yellow_price blue_price magenta_price cyan_price : ℚ)
  (black_count red_count yellow_count blue_count magenta_count cyan_count : ℕ)
  (tax_rate : ℚ) : ℚ :=
  let subtotal := black_price * black_count + red_price * red_count + 
                  yellow_price * yellow_count + blue_price * blue_count + 
                  magenta_price * magenta_count + cyan_price * cyan_count
  let total_cost := subtotal + subtotal * tax_rate
  total_cost - initial_money

theorem phantom_needs_more_money :
  additional_money_needed 50 12 16 14 17 15 18 3 4 3 2 2 1 (5/100) = 185.20 := by
  sorry

end phantom_needs_more_money_l1937_193759


namespace monomial_sum_l1937_193722

/-- Given two monomials of the same type, prove their sum -/
theorem monomial_sum (m n : ℕ) : 
  (2 : ℤ) * X^m * Y^3 + (-5 : ℤ) * X^1 * Y^(n+1) = (-3 : ℤ) * X^1 * Y^3 :=
by sorry

end monomial_sum_l1937_193722


namespace odd_implies_symmetric_abs_not_vice_versa_l1937_193741

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The graph of |f(x)| is symmetric about the y-axis if |f(-x)| = |f(x)| for all x ∈ ℝ -/
def IsSymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, |f (-x)| = |f x|

/-- If f is odd, then |f(x)| is symmetric about the y-axis, but not vice versa -/
theorem odd_implies_symmetric_abs_not_vice_versa :
  (∃ f : ℝ → ℝ, IsOdd f → IsSymmetricAboutYAxis f) ∧
  (∃ g : ℝ → ℝ, IsSymmetricAboutYAxis g ∧ ¬IsOdd g) := by
  sorry

end odd_implies_symmetric_abs_not_vice_versa_l1937_193741


namespace trampoline_jumps_l1937_193738

theorem trampoline_jumps (ronald_jumps : ℕ) (rupert_extra_jumps : ℕ) : 
  ronald_jumps = 157 → rupert_extra_jumps = 86 → 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 := by
sorry

end trampoline_jumps_l1937_193738


namespace meeting_arrangements_l1937_193781

/-- Represents the number of schools -/
def num_schools : ℕ := 4

/-- Represents the number of members per school -/
def members_per_school : ℕ := 6

/-- Represents the total number of members -/
def total_members : ℕ := num_schools * members_per_school

/-- Represents the number of representatives from the host school -/
def host_representatives : ℕ := 1

/-- Represents the number of non-host schools that send representatives -/
def non_host_schools : ℕ := 2

/-- Represents the number of representatives from each non-host school -/
def non_host_representatives : ℕ := 2

/-- Theorem stating the number of ways to arrange the meeting -/
theorem meeting_arrangements : 
  (num_schools) * (members_per_school.choose host_representatives) * 
  ((num_schools - 1).choose non_host_schools) * 
  ((members_per_school.choose non_host_representatives) ^ non_host_schools) = 16200 := by
  sorry


end meeting_arrangements_l1937_193781


namespace fraction_equality_l1937_193723

theorem fraction_equality : (2 : ℚ) / 5 - (1 : ℚ) / 7 = 1 / ((35 : ℚ) / 9) := by sorry

end fraction_equality_l1937_193723


namespace units_digit_of_n_l1937_193717

/-- Given two natural numbers m and n, where mn = 34^8 and m has a units digit of 4,
    prove that the units digit of n is 4. -/
theorem units_digit_of_n (m n : ℕ) 
  (h1 : m * n = 34^8)
  (h2 : m % 10 = 4) : 
  n % 10 = 4 := by sorry

end units_digit_of_n_l1937_193717


namespace common_material_choices_eq_120_l1937_193750

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways two students can choose 2 materials each from 6 materials,
    such that they have exactly 1 material in common -/
def commonMaterialChoices : ℕ :=
  choose 6 1 * choose 5 2

theorem common_material_choices_eq_120 : commonMaterialChoices = 120 := by
  sorry


end common_material_choices_eq_120_l1937_193750


namespace polygon_diagonals_l1937_193765

theorem polygon_diagonals (n : ℕ+) : 
  (∃ n, n * (n - 3) / 2 = 2 ∨ n * (n - 3) / 2 = 54) ∧ 
  (∀ n, n * (n - 3) / 2 ≠ 21 ∧ n * (n - 3) / 2 ≠ 32 ∧ n * (n - 3) / 2 ≠ 63) :=
by sorry

end polygon_diagonals_l1937_193765


namespace divisible_by_24_l1937_193746

theorem divisible_by_24 (n : ℤ) : ∃ k : ℤ, n^4 + 6*n^3 + 11*n^2 + 6*n = 24*k := by
  sorry

end divisible_by_24_l1937_193746


namespace linear_equation_solution_l1937_193709

theorem linear_equation_solution (x₁ y₁ x₂ y₂ : ℤ) :
  x₁ = 1 ∧ y₁ = -2 ∧ x₂ = -1 ∧ y₂ = -4 →
  x₁ - y₁ = 3 ∧ x₂ - y₂ = 3 :=
by sorry

end linear_equation_solution_l1937_193709


namespace tax_free_amount_correct_l1937_193766

/-- The tax-free amount for goods purchased in country B -/
def tax_free_amount : ℝ := 600

/-- The total value of goods purchased -/
def total_value : ℝ := 1720

/-- The tax rate applied to the excess amount -/
def tax_rate : ℝ := 0.07

/-- The amount of tax paid -/
def tax_paid : ℝ := 78.4

/-- Theorem stating that the tax-free amount satisfies the given conditions -/
theorem tax_free_amount_correct : 
  tax_rate * (total_value - tax_free_amount) = tax_paid := by
  sorry

end tax_free_amount_correct_l1937_193766


namespace base8_4532_equals_2394_l1937_193757

-- Define a function to convert a base 8 number to base 10
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- State the theorem
theorem base8_4532_equals_2394 :
  base8ToBase10 [2, 3, 5, 4] = 2394 := by
  sorry

end base8_4532_equals_2394_l1937_193757


namespace fixed_point_of_parabola_l1937_193735

theorem fixed_point_of_parabola (t : ℝ) : 
  5 * (3 : ℝ)^2 + t * 3 - 3 * t = 45 := by sorry

end fixed_point_of_parabola_l1937_193735


namespace circle_symmetry_orthogonality_l1937_193710

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

-- Define the symmetry line
def symmetry_line (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

-- Define the orthogonality condition
def orthogonal (x1 y1 x2 y2 : ℝ) : Prop := x1*x2 + y1*y2 = 0

theorem circle_symmetry_orthogonality :
  ∃ (m : ℝ) (x1 y1 x2 y2 : ℝ),
    curve x1 y1 ∧ curve x2 y2 ∧
    (∃ (x0 y0 : ℝ), symmetry_line m x0 y0 ∧ 
      (x1 - x0)^2 + (y1 - y0)^2 = (x2 - x0)^2 + (y2 - y0)^2) ∧
    orthogonal x1 y1 x2 y2 →
    m = -1 ∧ y2 - y1 = -(x2 - x1) := by sorry

end circle_symmetry_orthogonality_l1937_193710


namespace smallest_number_with_remainders_l1937_193725

theorem smallest_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 7 = 2 ∧
  n % 11 = 2 ∧
  n % 13 = 2 ∧
  n % 17 = 3 ∧
  n % 23 = 0 ∧
  n % 5 = 0 ∧
  (∀ m : ℕ, m > 0 ∧
    m % 7 = 2 ∧
    m % 11 = 2 ∧
    m % 13 = 2 ∧
    m % 17 = 3 ∧
    m % 23 = 0 ∧
    m % 5 = 0 → m ≥ n) ∧
  n = 391410 :=
by sorry

end smallest_number_with_remainders_l1937_193725


namespace largest_digit_divisible_by_six_l1937_193727

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (57890 + N).mod 6 = 0 → N ≤ 4 :=
by
  sorry

end largest_digit_divisible_by_six_l1937_193727


namespace essay_completion_time_l1937_193728

-- Define the essay parameters
def essay_length : ℕ := 1200
def initial_speed : ℕ := 400
def initial_duration : ℕ := 2
def subsequent_speed : ℕ := 200

-- Theorem statement
theorem essay_completion_time :
  let initial_words := initial_speed * initial_duration
  let remaining_words := essay_length - initial_words
  let subsequent_duration := remaining_words / subsequent_speed
  initial_duration + subsequent_duration = 4 := by
  sorry

end essay_completion_time_l1937_193728


namespace greatest_common_divisor_under_60_l1937_193718

theorem greatest_common_divisor_under_60 : 
  ∃ (n : ℕ), n = 45 ∧ 
  n ∣ 540 ∧ 
  n < 60 ∧ 
  n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 → m < 60 → m ∣ 180 → m ≤ n :=
by sorry

end greatest_common_divisor_under_60_l1937_193718


namespace place_face_difference_46_4_l1937_193711

/-- The place value of a digit in a two-digit number -/
def placeValue (n : ℕ) (d : ℕ) : ℕ :=
  if n ≥ 10 ∧ n < 100 ∧ d = n / 10 then d * 10 else 0

/-- The face value of a digit -/
def faceValue (d : ℕ) : ℕ := d

/-- The difference between place value and face value for a digit in a two-digit number -/
def placeFaceDifference (n : ℕ) (d : ℕ) : ℕ :=
  placeValue n d - faceValue d

theorem place_face_difference_46_4 : 
  placeFaceDifference 46 4 = 36 := by
  sorry

end place_face_difference_46_4_l1937_193711


namespace cricketer_average_score_l1937_193747

theorem cricketer_average_score (score1 score2 : ℝ) (matches1 matches2 : ℕ) 
  (h1 : score1 = 20)
  (h2 : score2 = 30)
  (h3 : matches1 = 2)
  (h4 : matches2 = 3) :
  let total_matches := matches1 + matches2
  let total_score := score1 * matches1 + score2 * matches2
  total_score / total_matches = 26 := by
sorry

end cricketer_average_score_l1937_193747


namespace floor_sum_example_l1937_193737

theorem floor_sum_example : ⌊(23.8 : ℝ)⌋ + ⌊(-23.8 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l1937_193737


namespace simplify_expression_l1937_193705

theorem simplify_expression (b : ℝ) (h : b ≠ 1) :
  1 - (1 / (1 + b / (1 - b))) = b := by sorry

end simplify_expression_l1937_193705


namespace jacob_younger_than_michael_l1937_193731

/-- Represents the age difference between Michael and Jacob -/
def age_difference (jacob_age michael_age : ℕ) : ℕ := michael_age - jacob_age

/-- Proves that Jacob is 14 years younger than Michael given the problem conditions -/
theorem jacob_younger_than_michael :
  ∀ (jacob_age michael_age : ℕ),
    (jacob_age < michael_age) →                        -- Jacob is younger than Michael
    (michael_age + 9 = 2 * (jacob_age + 9)) →          -- 9 years from now, Michael will be twice as old as Jacob
    (jacob_age + 4 = 9) →                              -- Jacob will be 9 years old in 4 years
    age_difference jacob_age michael_age = 14 :=        -- The age difference is 14 years
by
  sorry  -- Proof omitted


end jacob_younger_than_michael_l1937_193731


namespace circle_area_with_diameter_10_l1937_193774

/-- The area of a circle with diameter 10 centimeters is 25π square centimeters. -/
theorem circle_area_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end circle_area_with_diameter_10_l1937_193774


namespace cube_surface_area_equal_volume_l1937_193773

theorem cube_surface_area_equal_volume (a b c : ℝ) (h1 : a = 12) (h2 : b = 4) (h3 : c = 18) :
  let prism_volume := a * b * c
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = 864 := by
  sorry

end cube_surface_area_equal_volume_l1937_193773


namespace expression_evaluation_l1937_193785

theorem expression_evaluation (a : ℤ) (h : a = -1) : 
  (2*a + 1) * (2*a - 1) - 4*a*(a - 1) = -5 := by
  sorry

end expression_evaluation_l1937_193785


namespace quadratic_sequence_inconsistency_l1937_193716

def isQuadraticSequence (seq : List ℤ) : Prop :=
  let firstDiffs := List.zipWith (·-·) seq.tail seq
  let secondDiffs := List.zipWith (·-·) firstDiffs.tail firstDiffs
  secondDiffs.all (· = secondDiffs.head!)

def findInconsistentTerm (seq : List ℤ) : Option ℤ :=
  let firstDiffs := List.zipWith (·-·) seq.tail seq
  let secondDiffs := List.zipWith (·-·) firstDiffs.tail firstDiffs
  if h : secondDiffs.all (· = secondDiffs.head!) then
    none
  else
    some (seq[secondDiffs.findIndex (· ≠ secondDiffs.head!) + 1]!)

theorem quadratic_sequence_inconsistency 
  (seq : List ℤ) 
  (hseq : seq = [2107, 2250, 2402, 2574, 2738, 2920, 3094, 3286]) : 
  ¬isQuadraticSequence seq ∧ findInconsistentTerm seq = some 2574 :=
sorry

end quadratic_sequence_inconsistency_l1937_193716


namespace largest_four_digit_congruent_to_seven_mod_nineteen_l1937_193742

theorem largest_four_digit_congruent_to_seven_mod_nineteen :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 19 = 7 → n ≤ 9982 :=
by sorry

end largest_four_digit_congruent_to_seven_mod_nineteen_l1937_193742


namespace quadratic_roots_l1937_193753

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = -1 ∧ 
  (∀ x : ℝ, x^2 + x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_roots_l1937_193753


namespace max_value_of_expression_l1937_193775

theorem max_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n < 0) (h : 1/m + 1/n = 1) :
  ∃ (x : ℝ), ∀ (m' n' : ℝ), m' > 0 → n' < 0 → 1/m' + 1/n' = 1 → 4*m' + n' ≤ x ∧ 4*m + n = x :=
sorry

end max_value_of_expression_l1937_193775


namespace complement_of_A_union_B_l1937_193789

-- Define the set of integers
def U : Set Int := Set.univ

-- Define set A
def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}

-- Define set B
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}

-- Define the set of integers divisible by 3
def DivisibleBy3 : Set Int := {x | ∃ k : Int, x = 3 * k}

-- Theorem statement
theorem complement_of_A_union_B (x : Int) : 
  x ∈ (U \ (A ∪ B)) ↔ x ∈ DivisibleBy3 :=
sorry

end complement_of_A_union_B_l1937_193789


namespace sum_of_series_in_base7_l1937_193707

/-- Converts a number from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ :=
  sorry

/-- Converts a number from base 7 to base 10 -/
def fromBase7 (n : ℕ) : ℕ :=
  sorry

/-- Computes the sum of an arithmetic series -/
def arithmeticSeriesSum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem sum_of_series_in_base7 :
  let last_term := fromBase7 33
  let sum := arithmeticSeriesSum last_term
  toBase7 sum = 606 := by sorry

end sum_of_series_in_base7_l1937_193707


namespace student_subtraction_problem_l1937_193726

theorem student_subtraction_problem (x : ℝ) (h : x = 155) :
  ∃! y : ℝ, 2 * x - y = 110 ∧ y = 200 := by
sorry

end student_subtraction_problem_l1937_193726


namespace age_difference_l1937_193779

theorem age_difference (alvin_age simon_age : ℕ) (h1 : alvin_age = 30) (h2 : simon_age = 10) :
  alvin_age / 2 - simon_age = 5 := by
  sorry

end age_difference_l1937_193779


namespace function_max_min_sum_l1937_193796

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (Real.log x) / (Real.log a)

theorem function_max_min_sum (a : ℝ) :
  a > 0 ∧ a ≠ 1 →
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 2, f a x ≤ max) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = max) ∧
    (∀ x ∈ Set.Icc 1 2, min ≤ f a x) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = min) ∧
    max + min = (Real.log 2) / (Real.log a) + 6) →
  a = 2 :=
sorry

end function_max_min_sum_l1937_193796


namespace library_book_distribution_l1937_193703

/-- The number of books bought for each grade -/
def BookDistribution := Fin 4 → ℕ

/-- The total number of books bought -/
def total_books (d : BookDistribution) : ℕ :=
  d 0 + d 1 + d 2 + d 3

theorem library_book_distribution :
  ∃ (d : BookDistribution),
    d 0 = 37 ∧
    d 1 = 39 ∧
    d 2 = 43 ∧
    d 3 = 28 ∧
    d 1 + d 2 + d 3 = 110 ∧
    d 0 + d 2 + d 3 = 108 ∧
    d 0 + d 1 + d 3 = 104 ∧
    d 0 + d 1 + d 2 = 119 ∧
    total_books d = 147 :=
by
  sorry

end library_book_distribution_l1937_193703


namespace heather_block_distribution_l1937_193733

/-- Given an initial number of blocks, the number of blocks shared, and the number of friends,
    calculate the number of blocks each friend receives when distributing the remaining blocks equally. -/
def blocks_per_friend (initial_blocks : ℕ) (shared_blocks : ℕ) (num_friends : ℕ) : ℕ :=
  (initial_blocks - shared_blocks) / num_friends

/-- Theorem stating that given 258 initial blocks, after sharing 129 blocks and
    distributing the remainder equally among 6 friends, each friend receives 21 blocks. -/
theorem heather_block_distribution :
  blocks_per_friend 258 129 6 = 21 := by
  sorry

end heather_block_distribution_l1937_193733


namespace middle_number_bounds_l1937_193713

theorem middle_number_bounds (a b c : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : a + b + c = 10) (h4 : a - c = 3) : 7/3 < b ∧ b < 13/3 := by
  sorry

end middle_number_bounds_l1937_193713


namespace perfect_square_minus_seven_l1937_193700

theorem perfect_square_minus_seven (k : ℕ+) : 
  ∃ (n m : ℕ+), n * 2^k.val - 7 = m^2 := by
  sorry

end perfect_square_minus_seven_l1937_193700


namespace student_count_l1937_193771

theorem student_count (avg_age_students : ℝ) (teacher_age : ℕ) (new_avg_age : ℝ)
  (h1 : avg_age_students = 14)
  (h2 : teacher_age = 65)
  (h3 : new_avg_age = 15) :
  ∃ n : ℕ, n * avg_age_students + teacher_age = (n + 1) * new_avg_age ∧ n = 50 :=
by sorry

end student_count_l1937_193771


namespace shoe_production_facts_l1937_193762

/-- The daily production cost function -/
def C (n : ℕ) : ℝ := 4000 + 50 * n

/-- The selling price per pair of shoes -/
def sellingPrice : ℝ := 90

/-- All produced shoes are sold out -/
axiom all_sold : ∀ n : ℕ, n > 0 → ∃ revenue : ℝ, revenue = sellingPrice * n

/-- The profit function -/
def P (n : ℕ) : ℝ := sellingPrice * n - C n

theorem shoe_production_facts :
  (C 1000 = 54000) ∧
  (∃ n : ℕ, C n = 48000 ∧ n = 880) ∧
  (∀ n : ℕ, P n = 40 * n - 4000) ∧
  (∃ min_n : ℕ, min_n = 100 ∧ ∀ n : ℕ, n ≥ min_n → P n ≥ 0) :=
by sorry

end shoe_production_facts_l1937_193762


namespace minimize_sum_of_number_and_square_l1937_193760

/-- The function representing the sum of a number and its square -/
def f (x : ℝ) : ℝ := x + x^2

/-- The theorem stating that -1/2 minimizes the function f -/
theorem minimize_sum_of_number_and_square :
  ∀ x : ℝ, f (-1/2) ≤ f x :=
by
  sorry

end minimize_sum_of_number_and_square_l1937_193760


namespace peach_problem_l1937_193730

theorem peach_problem (steven jake jill hanna lucy : ℕ) : 
  steven = 19 →
  jake = steven - 12 →
  jake = 3 * jill →
  hanna = jake + 3 →
  lucy = hanna + 5 →
  lucy + jill = 17 :=
by
  sorry

end peach_problem_l1937_193730


namespace multiply_negative_two_l1937_193786

theorem multiply_negative_two : 3 * (-2) = -6 := by
  sorry

end multiply_negative_two_l1937_193786


namespace cos_difference_formula_l1937_193768

theorem cos_difference_formula (a b : ℝ) 
  (h1 : Real.sin a + Real.sin b = 1)
  (h2 : Real.cos a + Real.cos b = 3/2) : 
  Real.cos (a - b) = 5/8 := by sorry

end cos_difference_formula_l1937_193768


namespace soup_bins_calculation_l1937_193752

def total_bins : ℚ := 75/100
def vegetable_bins : ℚ := 12/100
def pasta_bins : ℚ := 1/2

theorem soup_bins_calculation : 
  total_bins - (vegetable_bins + pasta_bins) = 13/100 := by
  sorry

end soup_bins_calculation_l1937_193752


namespace constant_segment_shadow_ratio_l1937_193708

-- Define a structure for a segment and its shadow
structure SegmentWithShadow where
  segment_length : ℝ
  shadow_length : ℝ

-- Define the fixed conditions (lines and projection direction)
axiom fixed_conditions : Prop

-- Define the theorem
theorem constant_segment_shadow_ratio 
  (s1 s2 : SegmentWithShadow) 
  (h : fixed_conditions) : 
  s1.segment_length / s1.shadow_length = s2.segment_length / s2.shadow_length :=
sorry

end constant_segment_shadow_ratio_l1937_193708


namespace company_employee_increase_l1937_193749

theorem company_employee_increase (jan_employees dec_employees : ℝ) 
  (h_jan : jan_employees = 426.09)
  (h_dec : dec_employees = 490) :
  let increase := dec_employees - jan_employees
  let percentage_increase := (increase / jan_employees) * 100
  ∃ ε > 0, |percentage_increase - 15| < ε :=
by sorry

end company_employee_increase_l1937_193749


namespace rectangle_area_l1937_193740

theorem rectangle_area (x : ℝ) (h1 : (x + 5) * (2 * (x + 10)) = 3 * x * (x + 10)) (h2 : x > 0) :
  x * (x + 10) = 200 := by
sorry

end rectangle_area_l1937_193740


namespace f_is_odd_and_piecewise_l1937_193772

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f defined piecewise -/
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x * (x + 2) else -x^2 + 2*x

theorem f_is_odd_and_piecewise :
  OddFunction f ∧ (∀ x < 0, f x = x * (x + 2)) → ∀ x > 0, f x = -x^2 + 2*x := by
  sorry

end f_is_odd_and_piecewise_l1937_193772


namespace mutually_exclusive_events_l1937_193755

-- Define the sample space for two coin tosses
inductive CoinToss
  | HH  -- Two heads
  | HT  -- Head then tail
  | TH  -- Tail then head
  | TT  -- Two tails

-- Define the event "At least one head"
def atLeastOneHead (outcome : CoinToss) : Prop :=
  outcome = CoinToss.HH ∨ outcome = CoinToss.HT ∨ outcome = CoinToss.TH

-- Define the event "Both tosses are tails"
def bothTails (outcome : CoinToss) : Prop :=
  outcome = CoinToss.TT

-- Theorem stating that "Both tosses are tails" is mutually exclusive to "At least one head"
theorem mutually_exclusive_events :
  ∀ (outcome : CoinToss), ¬(atLeastOneHead outcome ∧ bothTails outcome) :=
by sorry

end mutually_exclusive_events_l1937_193755


namespace dans_remaining_limes_l1937_193776

/-- Given that Dan initially had 9 limes and gave away 4 limes, prove that he now has 5 limes. -/
theorem dans_remaining_limes (initial_limes : ℕ) (given_away : ℕ) (h1 : initial_limes = 9) (h2 : given_away = 4) :
  initial_limes - given_away = 5 := by
  sorry

end dans_remaining_limes_l1937_193776


namespace basketball_shooting_frequency_l1937_193787

/-- Given a basketball player who made 90 total shots with 63 successful shots,
    prove that the shooting frequency is equal to 0.7. -/
theorem basketball_shooting_frequency :
  let total_shots : ℕ := 90
  let successful_shots : ℕ := 63
  let shooting_frequency := (successful_shots : ℚ) / total_shots
  shooting_frequency = 0.7 := by sorry

end basketball_shooting_frequency_l1937_193787


namespace unique_solution_l1937_193784

theorem unique_solution : ∃! x : ℚ, x * 8 / 3 - (2 + 3) * 2 = 6 := by sorry

end unique_solution_l1937_193784


namespace bouquet_lilies_percentage_l1937_193798

theorem bouquet_lilies_percentage (F : ℚ) (F_pos : F > 0) : 
  let purple_flowers := (7 / 10) * F
  let purple_tulips := (1 / 2) * purple_flowers
  let yellow_flowers := F - purple_flowers
  let yellow_lilies := (2 / 3) * yellow_flowers
  let total_lilies := (purple_flowers - purple_tulips) + yellow_lilies
  (total_lilies / F) * 100 = 55 := by sorry

end bouquet_lilies_percentage_l1937_193798


namespace log_equation_range_l1937_193720

theorem log_equation_range (a : ℝ) :
  (∃ y : ℝ, y = Real.log (5 - a) / Real.log (a - 2)) ↔ (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) :=
sorry

end log_equation_range_l1937_193720


namespace flag_design_count_l1937_193795

/-- The number of school colors -/
def num_colors : ℕ := 3

/-- The number of horizontal stripes on the flag -/
def num_horizontal_stripes : ℕ := 3

/-- The number of options for the vertical stripe (3 colors + no stripe) -/
def vertical_stripe_options : ℕ := num_colors + 1

/-- The total number of possible flag designs -/
def total_flag_designs : ℕ := num_colors ^ num_horizontal_stripes * vertical_stripe_options

theorem flag_design_count :
  total_flag_designs = 108 :=
sorry

end flag_design_count_l1937_193795


namespace table_tennis_games_l1937_193791

theorem table_tennis_games (total_games : ℕ) 
  (petya_games : ℕ) (kolya_games : ℕ) (vasya_games : ℕ) : 
  petya_games = total_games / 2 →
  kolya_games = total_games / 3 →
  vasya_games = total_games / 5 →
  petya_games + kolya_games + vasya_games ≤ total_games →
  (∃ (games_between_petya_kolya : ℕ), 
    games_between_petya_kolya ≤ 1 ∧
    petya_games + kolya_games + vasya_games + games_between_petya_kolya = total_games) →
  total_games = 30 := by
sorry

end table_tennis_games_l1937_193791


namespace quadratic_real_roots_l1937_193780

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by sorry

end quadratic_real_roots_l1937_193780


namespace soda_cans_for_euros_l1937_193754

/-- The number of cans of soda that can be purchased for E euros, given that S cans can be purchased for Q quarters and 1 euro is worth 5 quarters. -/
theorem soda_cans_for_euros (S Q E : ℚ) (h1 : S > 0) (h2 : Q > 0) (h3 : E > 0) :
  (S / Q) * (5 * E) = (5 * S * E) / Q := by
  sorry

#check soda_cans_for_euros

end soda_cans_for_euros_l1937_193754


namespace quilt_patch_cost_l1937_193767

/-- The total cost of patches for a quilt with given dimensions and pricing structure -/
theorem quilt_patch_cost (quilt_length quilt_width patch_area : ℕ)
  (first_batch_size first_batch_price : ℕ) : 
  quilt_length = 16 →
  quilt_width = 20 →
  patch_area = 4 →
  first_batch_size = 10 →
  first_batch_price = 10 →
  (quilt_length * quilt_width) % patch_area = 0 →
  (first_batch_size * first_batch_price) + 
  ((quilt_length * quilt_width / patch_area - first_batch_size) * (first_batch_price / 2)) = 450 :=
by sorry

end quilt_patch_cost_l1937_193767


namespace expression_value_l1937_193734

theorem expression_value : 3 * (24 + 7)^2 - (24^2 + 7^2) = 2258 := by
  sorry

end expression_value_l1937_193734


namespace matchsticks_100th_stage_l1937_193764

/-- Represents the number of matchsticks in a stage of the pattern -/
def matchsticks (n : ℕ) : ℕ := 4 + (n - 1) * 4

/-- Proves that the 100th stage of the pattern contains 400 matchsticks -/
theorem matchsticks_100th_stage : matchsticks 100 = 400 := by
  sorry

end matchsticks_100th_stage_l1937_193764


namespace cube_difference_divisibility_l1937_193736

theorem cube_difference_divisibility (a b : ℤ) :
  ∃ k : ℤ, (2*a + 1)^3 - (2*b + 1)^3 + 8 = 16 * k := by
  sorry

end cube_difference_divisibility_l1937_193736


namespace next_simultaneous_visit_l1937_193719

def visit_interval_1 : ℕ := 6
def visit_interval_2 : ℕ := 8
def visit_interval_3 : ℕ := 9

theorem next_simultaneous_visit :
  Nat.lcm (Nat.lcm visit_interval_1 visit_interval_2) visit_interval_3 = 72 := by
  sorry

end next_simultaneous_visit_l1937_193719


namespace train_speed_calculation_l1937_193745

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed_calculation (bridge_length : ℝ) (train_length : ℝ) (time : ℝ) : 
  bridge_length = 650 →
  train_length = 200 →
  time = 17 →
  (bridge_length + train_length) / time = 50 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l1937_193745


namespace pure_imaginary_equation_l1937_193751

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is nonzero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_equation (z : ℂ) (a : ℝ) 
  (h1 : IsPureImaginary z) 
  (h2 : (1 + Complex.I) * z = 1 - a * Complex.I) : 
  a = 1 := by
  sorry

end pure_imaginary_equation_l1937_193751


namespace even_function_sum_l1937_193712

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- Define the property of an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc b 3, f a x = f a (-x)) →
  is_even (f a) →
  a + b = -3 :=
sorry

end even_function_sum_l1937_193712


namespace root_difference_implies_k_value_l1937_193739

theorem root_difference_implies_k_value (k : ℝ) : 
  (∀ r s : ℝ, r^2 + k*r + 12 = 0 ∧ s^2 + k*s + 12 = 0 → 
    (r+3)^2 - k*(r+3) + 12 = 0 ∧ (s+3)^2 - k*(s+3) + 12 = 0) →
  k = 3 := by
sorry

end root_difference_implies_k_value_l1937_193739


namespace board_cut_theorem_l1937_193732

theorem board_cut_theorem (total_length shorter_length longer_length : ℝ) :
  total_length = 20 ∧
  total_length = shorter_length + longer_length ∧
  2 * shorter_length = longer_length + 4 →
  shorter_length = 8 := by
sorry

end board_cut_theorem_l1937_193732


namespace future_age_calculation_l1937_193794

theorem future_age_calculation (nora_current_age terry_current_age : ℕ) 
  (h1 : nora_current_age = 10)
  (h2 : terry_current_age = 30) :
  ∃ (years_future : ℕ), terry_current_age + years_future = 4 * nora_current_age ∧ years_future = 10 :=
by sorry

end future_age_calculation_l1937_193794


namespace exists_special_sequence_l1937_193729

/-- An integer sequence satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℤ) (m : ℕ) : Prop :=
  (a 0 = 1) ∧ 
  (a 1 = 337) ∧ 
  (∀ n : ℕ, n ≥ 1 → (a (n+1) * a (n-1) - a n ^ 2) + 3 * (a (n+1) + a (n-1) - 2 * a n) / 4 = m) ∧
  (∀ n : ℕ, ∃ k : ℤ, (a n + 1) * (2 * a n + 1) / 6 = k ^ 2)

/-- Theorem stating the existence of a natural number m and a sequence satisfying the conditions -/
theorem exists_special_sequence : ∃ (m : ℕ) (a : ℕ → ℤ), SpecialSequence a m := by
  sorry

end exists_special_sequence_l1937_193729


namespace rectangular_field_dimensions_l1937_193701

theorem rectangular_field_dimensions : ∃ m : ℝ, m > 3 ∧ (3*m + 8)*(m - 3) = 100 := by
  sorry

end rectangular_field_dimensions_l1937_193701


namespace tan_alpha_value_l1937_193769

theorem tan_alpha_value (α : Real) (h_obtuse : π / 2 < α ∧ α < π) 
  (h_eq : (Real.sin α - 3 * Real.cos α) / (Real.cos α - Real.sin α) = Real.tan (2 * α)) :
  Real.tan α = 2 - Real.sqrt 7 := by
  sorry

end tan_alpha_value_l1937_193769


namespace equation_two_roots_l1937_193724

-- Define the equation
def equation (x k : ℂ) : Prop :=
  x / (x + 3) + x / (x + 4) = k * x

-- Define the set of valid k values
def valid_k_values : Set ℂ :=
  {0, 7/12, Complex.I, -Complex.I}

-- Theorem statement
theorem equation_two_roots (k : ℂ) :
  (∃! (r₁ r₂ : ℂ), r₁ ≠ r₂ ∧ equation r₁ k ∧ equation r₂ k) ↔ k ∈ valid_k_values :=
sorry

end equation_two_roots_l1937_193724


namespace problem_solution_l1937_193704

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := a > 0 ∧ ∃ c, c > 0 ∧ c < a ∧
  ∀ x y, x^2/2 + y^2/a = 1 → y^2 ≥ c*(1 - x^2/2)

-- Define the main theorem
theorem problem_solution (a : ℝ) 
  (h1 : ¬(q a))
  (h2 : p a ∨ q a) :
  (-1 < a ∧ a ≤ 0) ∧
  ((a = 1 → ∀ x y, (a+1)*x^2 + (1-a)*y^2 = (a+1)*(1-a) → y = 0) ∧
   (-1 < a ∧ a < 0 → ∃ b c, b > c ∧ c > 0 ∧
     ∀ x y, (a+1)*x^2 + (1-a)*y^2 = (a+1)*(1-a) →
       x^2/b^2 + y^2/c^2 = 1) ∧
   (a = 0 → ∀ x y, (a+1)*x^2 + (1-a)*y^2 = (a+1)*(1-a) →
     x^2 + y^2 = 1)) := by
  sorry

end problem_solution_l1937_193704


namespace second_student_wrong_answers_second_student_wrong_answers_value_l1937_193744

theorem second_student_wrong_answers 
  (total_questions : Nat) 
  (hannah_correct : Nat) 
  (hannah_highest_score : Bool) : Nat :=
  let second_student_correct := hannah_correct - 1
  let second_student_wrong := total_questions - second_student_correct
  second_student_wrong

#check second_student_wrong_answers

theorem second_student_wrong_answers_value :
  second_student_wrong_answers 40 39 true = 2 := by sorry

end second_student_wrong_answers_second_student_wrong_answers_value_l1937_193744


namespace factor_implies_s_value_l1937_193702

theorem factor_implies_s_value (m s : ℤ) : 
  (∃ k : ℤ, m^2 - s*m - 24 = (m - 8) * k) → s = 5 := by
  sorry

end factor_implies_s_value_l1937_193702


namespace cubic_extrema_opposite_signs_l1937_193714

/-- A cubic function with coefficients p and q -/
def cubic_function (p q : ℝ) (x : ℝ) : ℝ := x^3 + p*x + q

/-- The derivative of the cubic function -/
def cubic_derivative (p : ℝ) (x : ℝ) : ℝ := 3*x^2 + p

/-- Condition for opposite signs of extremum points -/
def opposite_signs_condition (p q : ℝ) : Prop := 
  (q/2)^2 + (p/3)^3 < 0 ∧ p < 0

theorem cubic_extrema_opposite_signs 
  (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    cubic_derivative p x₁ = 0 ∧ 
    cubic_derivative p x₂ = 0 ∧ 
    cubic_function p q x₁ * cubic_function p q x₂ < 0) ↔ 
  opposite_signs_condition p q :=
sorry

end cubic_extrema_opposite_signs_l1937_193714


namespace expression_evaluation_l1937_193721

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 8) + 2 = -x^4 + 3*x^3 - 5*x^2 + 8*x + 2 := by
  sorry

end expression_evaluation_l1937_193721


namespace hyperbola_asymptotes_l1937_193761

/-- The hyperbola and parabola equations -/
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

def parabola (x y : ℝ) : Prop := y^2 = 8 * Real.sqrt 2 * x

/-- The right focus of the hyperbola coincides with the focus of the parabola -/
axiom focus_coincide : ∃ (x₀ y₀ : ℝ), 
  (x₀ = 2 * Real.sqrt 2 ∧ y₀ = 0) ∧
  (∀ x y b, hyperbola x y b → (x - x₀)^2 + y^2 = (2 * Real.sqrt 2)^2)

/-- The theorem stating that the asymptotes of the hyperbola are y = ±x -/
theorem hyperbola_asymptotes : 
  ∃ b, ∀ x y, hyperbola x y b → (y = x ∨ y = -x) ∨ (x^2 > y^2) := by sorry

end hyperbola_asymptotes_l1937_193761


namespace purple_balls_count_l1937_193797

theorem purple_balls_count (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 60 + purple)
  (h2 : white = 22)
  (h3 : green = 18)
  (h4 : yellow = 17)
  (h5 : red = 3)
  (h6 : (white + green + yellow : ℚ) / total = 95/100) : 
  purple = 0 := by
sorry

end purple_balls_count_l1937_193797


namespace max_value_expression_l1937_193758

theorem max_value_expression (x y : ℝ) : 
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ 5 := by
  sorry

end max_value_expression_l1937_193758


namespace roots_of_quadratic_sum_of_fourth_powers_l1937_193790

theorem roots_of_quadratic_sum_of_fourth_powers (α β : ℝ) : 
  α^2 - 2*α - 8 = 0 → β^2 - 2*β - 8 = 0 → 3*α^4 + 4*β^4 = 1232 := by
  sorry

end roots_of_quadratic_sum_of_fourth_powers_l1937_193790


namespace mikeys_leaves_theorem_l1937_193763

/-- Given an initial number of leaves and the remaining number of leaves,
    calculate the number of leaves that blew away. -/
def leaves_blown_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that for Mikey's specific case, 
    the number of leaves blown away is 244. -/
theorem mikeys_leaves_theorem :
  leaves_blown_away 356 112 = 244 := by
  sorry

end mikeys_leaves_theorem_l1937_193763


namespace max_sum_after_pyramid_addition_l1937_193706

/-- Represents a polyhedron with faces, edges, and vertices -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Represents the result of adding a pyramid to a face of a polyhedron -/
structure PyramidAddition where
  newFaces : ℕ
  newEdges : ℕ
  newVertices : ℕ

/-- Calculates the sum of faces, edges, and vertices after adding a pyramid -/
def sumAfterPyramidAddition (p : Polyhedron) (pa : PyramidAddition) : ℕ :=
  (p.faces - 1 + pa.newFaces) + (p.edges + pa.newEdges) + (p.vertices + pa.newVertices)

/-- The pentagonal prism -/
def pentagonalPrism : Polyhedron :=
  { faces := 7, edges := 15, vertices := 10 }

/-- Adding a pyramid to a pentagonal face -/
def pentagonalFaceAddition : PyramidAddition :=
  { newFaces := 5, newEdges := 5, newVertices := 1 }

/-- Adding a pyramid to a quadrilateral face -/
def quadrilateralFaceAddition : PyramidAddition :=
  { newFaces := 4, newEdges := 4, newVertices := 1 }

/-- Theorem: The maximum sum of faces, edges, and vertices after adding a pyramid is 42 -/
theorem max_sum_after_pyramid_addition :
  (max 
    (sumAfterPyramidAddition pentagonalPrism pentagonalFaceAddition)
    (sumAfterPyramidAddition pentagonalPrism quadrilateralFaceAddition)) = 42 := by
  sorry

end max_sum_after_pyramid_addition_l1937_193706


namespace equation_roots_and_ellipse_condition_l1937_193793

theorem equation_roots_and_ellipse_condition (m n : ℝ) : 
  ¬(((m^2 - 4*n ≥ 0 ∧ m > 0 ∧ n > 0) → (m > 0 ∧ n > 0 ∧ m ≠ n)) ∧ 
    ((m > 0 ∧ n > 0 ∧ m ≠ n) → (m^2 - 4*n ≥ 0 ∧ m > 0 ∧ n > 0))) :=
by sorry

end equation_roots_and_ellipse_condition_l1937_193793


namespace distance_to_y_axis_l1937_193748

/-- Theorem: For a point P with coordinates (x, -6), if the distance from the x-axis to P
    is half the distance from the y-axis to P, then the distance from the y-axis to P is 12 units. -/
theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -6)
  let distance_to_x_axis := |P.2|
  let distance_to_y_axis := |P.1|
  distance_to_x_axis = (1/2 : ℝ) * distance_to_y_axis →
  distance_to_y_axis = 12 := by
  sorry

end distance_to_y_axis_l1937_193748


namespace kelsey_ekon_difference_l1937_193792

/-- The number of videos watched by three friends. -/
def total_videos : ℕ := 411

/-- The number of videos watched by Kelsey. -/
def kelsey_videos : ℕ := 160

/-- The number of videos watched by Uma. -/
def uma_videos : ℕ := (total_videos - kelsey_videos + 17) / 2

/-- The number of videos watched by Ekon. -/
def ekon_videos : ℕ := uma_videos - 17

/-- Theorem stating the difference in videos watched between Kelsey and Ekon. -/
theorem kelsey_ekon_difference :
  kelsey_videos - ekon_videos = 43 :=
by sorry

end kelsey_ekon_difference_l1937_193792


namespace difference_of_odd_squares_divisible_by_eight_l1937_193799

theorem difference_of_odd_squares_divisible_by_eight (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 2 * k + 1) 
  (hb : ∃ m : ℤ, b = 2 * m + 1) : 
  ∃ n : ℤ, a ^ 2 - b ^ 2 = 8 * n := by
  sorry

end difference_of_odd_squares_divisible_by_eight_l1937_193799


namespace bean_region_probability_l1937_193756

noncomputable def probability_bean_region : ℝ :=
  let total_area := (1 - 0) * ((Real.exp 1 + 1) - 0)
  let specific_area := ∫ x in (0)..(1), (Real.exp x + 1) - (Real.exp 1 + 1)
  specific_area / total_area

theorem bean_region_probability : probability_bean_region = 1 / (Real.exp 1 + 1) := by
  sorry

end bean_region_probability_l1937_193756


namespace solution_difference_l1937_193778

theorem solution_difference (r s : ℝ) : 
  ((6 * r - 18) / (r^2 - 4*r - 21) = r - 3) →
  ((6 * s - 18) / (s^2 - 4*s - 21) = s - 3) →
  r ≠ s →
  r > s →
  r - s = 4 := by
sorry

end solution_difference_l1937_193778


namespace inverse_proportion_graph_l1937_193782

/-- Given that point A(2,4) lies on the graph of y = k/x, prove that (4,2) also lies on the graph
    while (-2,4), (2,-4), and (-4,2) do not. -/
theorem inverse_proportion_graph (k : ℝ) (h : k ≠ 0) : 
  (4 : ℝ) = k / 2 →  -- Point A(2,4) lies on the graph
  (2 : ℝ) = k / 4 ∧  -- Point (4,2) lies on the graph
  (4 : ℝ) ≠ k / (-2) ∧  -- Point (-2,4) does not lie on the graph
  (-4 : ℝ) ≠ k / 2 ∧  -- Point (2,-4) does not lie on the graph
  (2 : ℝ) ≠ k / (-4) :=  -- Point (-4,2) does not lie on the graph
by
  sorry


end inverse_proportion_graph_l1937_193782


namespace store_change_calculation_l1937_193715

def payment : ℕ := 20
def num_items : ℕ := 3
def item_cost : ℕ := 2

theorem store_change_calculation :
  payment - (num_items * item_cost) = 14 := by
  sorry

end store_change_calculation_l1937_193715


namespace fast_clock_next_correct_time_l1937_193770

/-- Represents a clock that gains time uniformly --/
structure FastClock where
  /-- The rate at which the clock gains time, in minutes per day --/
  gain_rate : ℝ
  /-- The gain rate is positive but less than 60 minutes per day --/
  gain_rate_bounds : 0 < gain_rate ∧ gain_rate < 60

/-- Represents a specific date and time --/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Function to check if two DateTimes are equal --/
def DateTime.eq (dt1 dt2 : DateTime) : Prop :=
  dt1.year = dt2.year ∧ 
  dt1.month = dt2.month ∧ 
  dt1.day = dt2.day ∧ 
  dt1.hour = dt2.hour ∧ 
  dt1.minute = dt2.minute

/-- Function to calculate when the clock will next show the correct time --/
def next_correct_time (c : FastClock) (start : DateTime) (overlap : DateTime) : DateTime :=
  sorry

/-- Theorem stating when the clock will next show the correct time --/
theorem fast_clock_next_correct_time (c : FastClock) :
  let start := DateTime.mk 1982 1 1 0 0
  let overlap := DateTime.mk 1982 1 1 13 5
  let result := DateTime.mk 1984 5 13 12 0
  DateTime.eq (next_correct_time c start overlap) result := by
  sorry

end fast_clock_next_correct_time_l1937_193770


namespace modular_arithmetic_problems_l1937_193783

theorem modular_arithmetic_problems :
  (∃ k : ℕ, 19^10 = 6 * k + 1) ∧
  (∃ m : ℕ, 19^14 = 70 * m + 11) ∧
  (∃ n : ℕ, 17^9 = 48 * n + 17) ∧
  (∃ p : ℕ, 14^(14^14) = 100 * p + 36) := by
sorry

end modular_arithmetic_problems_l1937_193783


namespace video_likes_dislikes_ratio_l1937_193788

theorem video_likes_dislikes_ratio :
  ∀ (initial_dislikes : ℕ),
    (initial_dislikes + 1000 = 2600) →
    (initial_dislikes : ℚ) / 3000 = 8 / 15 := by
  sorry

end video_likes_dislikes_ratio_l1937_193788
