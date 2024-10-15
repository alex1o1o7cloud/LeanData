import Mathlib

namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1870_187018

theorem simplify_trig_expression :
  (Real.sin (35 * π / 180))^2 - 1/2 = 
  -2 * (Real.cos (10 * π / 180) * Real.cos (80 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1870_187018


namespace NUMINAMATH_CALUDE_baseball_stats_l1870_187096

-- Define the total number of hits and the number of each type of hit
def total_hits : ℕ := 45
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 6

-- Define the number of singles
def singles : ℕ := total_hits - (home_runs + triples + doubles)

-- Define the percentage of singles
def singles_percentage : ℚ := (singles : ℚ) / (total_hits : ℚ) * 100

-- Theorem to prove
theorem baseball_stats :
  singles = 34 ∧ singles_percentage = 75.56 := by
  sorry

end NUMINAMATH_CALUDE_baseball_stats_l1870_187096


namespace NUMINAMATH_CALUDE_division_theorem_l1870_187074

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = 176 → 
  divisor = 19 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder → 
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l1870_187074


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l1870_187059

theorem sine_cosine_inequality (x a b : ℝ) :
  (Real.sin x + a * Real.cos x) * (Real.sin x + b * Real.cos x) ≤ 1 + ((a + b) / 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l1870_187059


namespace NUMINAMATH_CALUDE_percentage_problem_l1870_187021

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 400) : 1.2 * x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1870_187021


namespace NUMINAMATH_CALUDE_point_B_coordinate_l1870_187035

def point_A : ℝ := -1

theorem point_B_coordinate (point_B : ℝ) (h : |point_B - point_A| = 3) :
  point_B = 2 ∨ point_B = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinate_l1870_187035


namespace NUMINAMATH_CALUDE_pet_store_puppies_sold_l1870_187037

/-- Proves that the number of puppies sold is 1, given the conditions of the pet store problem. -/
theorem pet_store_puppies_sold :
  let kittens_sold : ℕ := 2
  let kitten_price : ℕ := 6
  let puppy_price : ℕ := 5
  let total_earnings : ℕ := 17
  let puppies_sold : ℕ := (total_earnings - kittens_sold * kitten_price) / puppy_price
  puppies_sold = 1 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_sold_l1870_187037


namespace NUMINAMATH_CALUDE_lcm_inequality_l1870_187085

/-- For any two positive integers n and m where n > m, 
    the sum of the least common multiples of (m,n) and (m+1,n+1) 
    is greater than or equal to (2nm)/√(n-m). -/
theorem lcm_inequality (m n : ℕ) (h1 : 0 < m) (h2 : m < n) :
  Nat.lcm m n + Nat.lcm (m + 1) (n + 1) ≥ 
  (2 * n * m : ℝ) / Real.sqrt (n - m : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_lcm_inequality_l1870_187085


namespace NUMINAMATH_CALUDE_group_composition_l1870_187068

/-- Proves that in a group of 300 people, where the number of men is twice the number of women,
    and the number of women is 3 times the number of children, the number of children is 30. -/
theorem group_composition (total : ℕ) (children : ℕ) (women : ℕ) (men : ℕ) 
    (h1 : total = 300)
    (h2 : men = 2 * women)
    (h3 : women = 3 * children)
    (h4 : total = children + women + men) : 
  children = 30 := by
sorry

end NUMINAMATH_CALUDE_group_composition_l1870_187068


namespace NUMINAMATH_CALUDE_square_plus_one_nonzero_l1870_187033

theorem square_plus_one_nonzero : ∀ x : ℝ, x^2 + 1 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_square_plus_one_nonzero_l1870_187033


namespace NUMINAMATH_CALUDE_propositions_truth_values_l1870_187044

def proposition1 : Prop := (100 % 10 = 0) ∧ (100 % 5 = 0)

def proposition2 : Prop := (3^2 - 9 = 0) ∨ ((-3)^2 - 9 = 0)

def proposition3 : Prop := ¬(2^2 - 9 = 0)

theorem propositions_truth_values :
  proposition1 ∧ proposition2 ∧ ¬proposition3 :=
sorry

end NUMINAMATH_CALUDE_propositions_truth_values_l1870_187044


namespace NUMINAMATH_CALUDE_texas_integrated_school_student_count_l1870_187009

theorem texas_integrated_school_student_count 
  (initial_classes : ℕ) 
  (students_per_class : ℕ) 
  (additional_classes : ℕ) : 
  initial_classes = 15 → 
  students_per_class = 20 → 
  additional_classes = 5 → 
  (initial_classes + additional_classes) * students_per_class = 400 := by
sorry

end NUMINAMATH_CALUDE_texas_integrated_school_student_count_l1870_187009


namespace NUMINAMATH_CALUDE_b_8_equals_162_l1870_187012

/-- Given sequences {aₙ} and {bₙ} satisfying the specified conditions, b₈ equals 162 -/
theorem b_8_equals_162 (a b : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 1 → (a n) * (a (n + 1)) = 3^n)
  (h3 : ∀ n : ℕ, n ≥ 1 → (a n) + (a (n + 1)) = b n)
  (h4 : ∀ n : ℕ, n ≥ 1 → (a n)^2 - (b n) * (a n) + 3^n = 0) :
  b 8 = 162 := by
  sorry

end NUMINAMATH_CALUDE_b_8_equals_162_l1870_187012


namespace NUMINAMATH_CALUDE_min_prime_divisor_of_quadratic_l1870_187098

theorem min_prime_divisor_of_quadratic : 
  ∀ p : ℕ, Prime p → (∃ n : ℕ, p ∣ (n^2 + 7*n + 23)) → p ≥ 11 :=
by sorry

end NUMINAMATH_CALUDE_min_prime_divisor_of_quadratic_l1870_187098


namespace NUMINAMATH_CALUDE_impossible_c_nine_l1870_187006

/-- An obtuse triangle with sides a, b, and c -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_obtuse : (a^2 + b^2 < c^2) ∨ (a^2 + c^2 < b^2) ∨ (b^2 + c^2 < a^2)
  h_triangle_inequality : (a + b > c) ∧ (a + c > b) ∧ (b + c > a)
  h_positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The theorem stating that c = 9 is impossible for the given obtuse triangle -/
theorem impossible_c_nine (t : ObtuseTriangle) (h1 : t.a = 6) (h2 : t.b = 8) : t.c ≠ 9 := by
  sorry

#check impossible_c_nine

end NUMINAMATH_CALUDE_impossible_c_nine_l1870_187006


namespace NUMINAMATH_CALUDE_bid_probabilities_theorem_l1870_187003

/-- Represents the probability of winning a bid for a project -/
structure BidProbability where
  value : ℝ
  is_probability : 0 ≤ value ∧ value ≤ 1

/-- Represents the probabilities of winning bids for three projects -/
structure ProjectProbabilities where
  a : BidProbability
  b : BidProbability
  c : BidProbability
  a_gt_b : a.value > b.value
  c_eq_quarter : c.value = 1/4

/-- The main theorem stating the properties of the bid probabilities -/
theorem bid_probabilities_theorem (p : ProjectProbabilities) : 
  p.a.value * p.b.value * p.c.value = 1/24 ∧
  1 - (1 - p.a.value) * (1 - p.b.value) * (1 - p.c.value) = 3/4 →
  p.a.value = 1/2 ∧ p.b.value = 1/3 ∧
  p.a.value * p.b.value * (1 - p.c.value) + 
  p.a.value * (1 - p.b.value) * p.c.value + 
  (1 - p.a.value) * p.b.value * p.c.value = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_bid_probabilities_theorem_l1870_187003


namespace NUMINAMATH_CALUDE_point_outside_circle_l1870_187032

theorem point_outside_circle (a b : ℝ) (i : ℂ) : 
  i * i = -1 → 
  Complex.I = i →
  a + b * i = (2 + i) / (1 - i) → 
  a^2 + b^2 > 2 := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1870_187032


namespace NUMINAMATH_CALUDE_current_at_12_ohms_l1870_187070

/-- A battery with voltage 48V and current-resistance relationship I = 48 / R -/
structure Battery where
  voltage : ℝ
  current : ℝ → ℝ
  resistance : ℝ
  h_voltage : voltage = 48
  h_current : ∀ R, current R = 48 / R

/-- When resistance is 12Ω, the current is 4A -/
theorem current_at_12_ohms (b : Battery) (h : b.resistance = 12) : 
  b.current b.resistance = 4 := by
  sorry

end NUMINAMATH_CALUDE_current_at_12_ohms_l1870_187070


namespace NUMINAMATH_CALUDE_some_students_not_fraternity_members_l1870_187046

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (Honest : U → Prop)
variable (FraternityMember : U → Prop)

-- Define the given conditions
axiom some_students_not_honest : ∃ x, Student x ∧ ¬Honest x
axiom all_fraternity_members_honest : ∀ x, FraternityMember x → Honest x

-- Theorem to prove
theorem some_students_not_fraternity_members : 
  ∃ x, Student x ∧ ¬FraternityMember x :=
sorry

end NUMINAMATH_CALUDE_some_students_not_fraternity_members_l1870_187046


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_6_l1870_187038

theorem circle_area_with_diameter_6 (π : ℝ) (h : π > 0) :
  let diameter : ℝ := 6
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 9 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_6_l1870_187038


namespace NUMINAMATH_CALUDE_cost_per_book_l1870_187073

theorem cost_per_book (total_books : ℕ) (total_spent : ℕ) (h1 : total_books = 14) (h2 : total_spent = 224) :
  total_spent / total_books = 16 := by
sorry

end NUMINAMATH_CALUDE_cost_per_book_l1870_187073


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1870_187081

theorem largest_digit_divisible_by_six :
  ∃ (M : ℕ), M < 10 ∧ 
  (∀ (n : ℕ), n < 10 → 6 ∣ (3190 * 10 + n) → n ≤ M) ∧
  (6 ∣ (3190 * 10 + M)) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1870_187081


namespace NUMINAMATH_CALUDE_discount_calculation_l1870_187077

/-- Proves that a product with given cost and original prices, sold at a specific profit margin, 
    results in a particular discount percentage. -/
theorem discount_calculation (cost_price original_price : ℝ) 
  (profit_margin : ℝ) (discount_percentage : ℝ) : 
  cost_price = 200 → 
  original_price = 300 → 
  profit_margin = 0.05 →
  discount_percentage = 0.7 →
  (original_price * discount_percentage - cost_price) / cost_price = profit_margin :=
by
  sorry

#check discount_calculation

end NUMINAMATH_CALUDE_discount_calculation_l1870_187077


namespace NUMINAMATH_CALUDE_max_value_theorem_l1870_187051

theorem max_value_theorem (x : ℝ) (h : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 6)) / x ≤ 36 / (2 * Real.sqrt 3 + Real.sqrt (2 * Real.sqrt 6)) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1870_187051


namespace NUMINAMATH_CALUDE_collinear_points_theorem_l1870_187049

/-- Given vectors OA, OB, OC in R², if A, B, C are collinear, then the x-coordinate of OA is 18 -/
theorem collinear_points_theorem (k : ℝ) : 
  let OA : Fin 2 → ℝ := ![k, 12]
  let OB : Fin 2 → ℝ := ![4, 5]
  let OC : Fin 2 → ℝ := ![10, 8]
  (∃ (t : ℝ), (OC - OB) = t • (OA - OB)) → k = 18 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_theorem_l1870_187049


namespace NUMINAMATH_CALUDE_compare_powers_l1870_187048

theorem compare_powers : 2^2023 * 7^2023 < 3^2023 * 5^2023 := by
  sorry

end NUMINAMATH_CALUDE_compare_powers_l1870_187048


namespace NUMINAMATH_CALUDE_min_value_bn_Sn_l1870_187093

def S (n : ℕ) : ℚ := n / (n + 1)

def b (n : ℕ) : ℤ := n - 8

theorem min_value_bn_Sn :
  ∃ (min : ℚ), min = -4 ∧
  ∀ (n : ℕ), n ≥ 1 → (b n : ℚ) * S n ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_bn_Sn_l1870_187093


namespace NUMINAMATH_CALUDE_student_number_factor_l1870_187011

theorem student_number_factor (x f : ℝ) : 
  x = 110 → x * f - 220 = 110 → f = 3 := by
  sorry

end NUMINAMATH_CALUDE_student_number_factor_l1870_187011


namespace NUMINAMATH_CALUDE_joe_pocket_transfer_l1870_187034

/-- Represents the money transfer problem with Joe's pockets --/
def MoneyTransferProblem (total initial_left transfer_amount : ℚ) : Prop :=
  let initial_right := total - initial_left
  let after_quarter_left := initial_left - (initial_left / 4)
  let after_quarter_right := initial_right + (initial_left / 4)
  let final_left := after_quarter_left - transfer_amount
  let final_right := after_quarter_right + transfer_amount
  (total = 200) ∧ 
  (initial_left = 160) ∧ 
  (final_left = final_right) ∧
  (transfer_amount > 0)

theorem joe_pocket_transfer : 
  ∃ (transfer_amount : ℚ), MoneyTransferProblem 200 160 transfer_amount ∧ transfer_amount = 20 := by
  sorry

end NUMINAMATH_CALUDE_joe_pocket_transfer_l1870_187034


namespace NUMINAMATH_CALUDE_y_coordinates_descending_l1870_187004

/-- Given a line y = -2x + b and three points on this line, prove that the y-coordinates are in descending order as x increases. -/
theorem y_coordinates_descending 
  (b : ℝ) 
  (y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = 4 + b) 
  (h2 : y₂ = 2 + b) 
  (h3 : y₃ = -2 + b) : 
  y₃ < y₂ ∧ y₂ < y₁ := by
sorry

end NUMINAMATH_CALUDE_y_coordinates_descending_l1870_187004


namespace NUMINAMATH_CALUDE_oil_quantity_function_correct_l1870_187063

/-- Represents the remaining oil quantity in liters after t minutes -/
def Q (t : ℝ) : ℝ := 20 - 0.2 * t

/-- The initial oil quantity in liters -/
def initial_quantity : ℝ := 20

/-- The outflow rate in liters per minute -/
def outflow_rate : ℝ := 0.2

theorem oil_quantity_function_correct : 
  ∀ t : ℝ, t ≥ 0 → Q t = initial_quantity - outflow_rate * t :=
sorry

end NUMINAMATH_CALUDE_oil_quantity_function_correct_l1870_187063


namespace NUMINAMATH_CALUDE_max_value_of_equation_l1870_187087

theorem max_value_of_equation (x : ℝ) : 
  (x^2 - x - 30) / (x - 5) = 2 / (x + 6) → x ≤ Real.sqrt 38 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_equation_l1870_187087


namespace NUMINAMATH_CALUDE_direct_proportion_information_needed_l1870_187019

/-- A structure representing a direct proportion between x and y -/
structure DirectProportion where
  k : ℝ  -- Constant of proportionality
  y : ℝ → ℝ  -- Function mapping x to y
  prop : ∀ x, y x = k * x  -- Property of direct proportion

/-- The number of pieces of information needed to determine a direct proportion -/
def informationNeeded : ℕ := 2

/-- Theorem stating that exactly 2 pieces of information are needed to determine a direct proportion -/
theorem direct_proportion_information_needed :
  ∀ (dp : DirectProportion), informationNeeded = 2 :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_information_needed_l1870_187019


namespace NUMINAMATH_CALUDE_star_def_star_diff_neg_star_special_case_l1870_187084

-- Define the ☆ operation
def star (a b : ℝ) : ℝ := 3 * a + b

-- Theorem 1: Definition of ☆ operation
theorem star_def (a b : ℝ) : star a b = 3 * a + b := by sorry

-- Theorem 2: If a < b, then a☆b - b☆a < 0
theorem star_diff_neg {a b : ℝ} (h : a < b) : star a b - star b a < 0 := by sorry

-- Theorem 3: If a☆(-2b) = 4, then [3(a-b)]☆(3a+b) = 16
theorem star_special_case {a b : ℝ} (h : star a (-2*b) = 4) : 
  star (3*(a-b)) (3*a+b) = 16 := by sorry

end NUMINAMATH_CALUDE_star_def_star_diff_neg_star_special_case_l1870_187084


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1870_187057

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : a 2 = 9) 
  (h2 : a 5 = 33) : 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d ∧ d = 8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1870_187057


namespace NUMINAMATH_CALUDE_wendys_brother_candy_prove_wendys_brother_candy_l1870_187056

/-- Wendy's candy problem -/
theorem wendys_brother_candy : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (wendys_boxes : ℕ) (pieces_per_box : ℕ) (total_pieces : ℕ) (brothers_pieces : ℕ) =>
    wendys_boxes * pieces_per_box + brothers_pieces = total_pieces →
    wendys_boxes = 2 →
    pieces_per_box = 3 →
    total_pieces = 12 →
    brothers_pieces = 6

/-- Proof of Wendy's candy problem -/
theorem prove_wendys_brother_candy : wendys_brother_candy 2 3 12 6 := by
  sorry

end NUMINAMATH_CALUDE_wendys_brother_candy_prove_wendys_brother_candy_l1870_187056


namespace NUMINAMATH_CALUDE_second_exam_study_time_l1870_187010

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  study_time : ℝ
  test_score : ℝ
  inverse_relation : study_time * test_score = study_time * test_score

/-- Theorem stating the required study time for the second exam -/
theorem second_exam_study_time 
  (first_exam : StudyScoreRelation)
  (h : first_exam.study_time = 6 ∧ first_exam.test_score = 60)
  (second_exam : StudyScoreRelation)
  (average_score : ℝ)
  (h_average : average_score = 90)
  (h_total_score : first_exam.test_score + second_exam.test_score = 2 * average_score) :
  second_exam.study_time = 3 ∧ second_exam.test_score = 120 := by
  sorry

#check second_exam_study_time

end NUMINAMATH_CALUDE_second_exam_study_time_l1870_187010


namespace NUMINAMATH_CALUDE_parentheses_placement_count_l1870_187001

/-- A sequence of prime numbers -/
def primeSequence : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- The operation of placing parentheses in the expression -/
def parenthesesPlacement (seq : List Nat) : Nat :=
  2^(seq.length - 2)

/-- Theorem stating the number of different values obtained by placing parentheses -/
theorem parentheses_placement_count :
  parenthesesPlacement primeSequence = 256 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_placement_count_l1870_187001


namespace NUMINAMATH_CALUDE_train_speed_l1870_187060

/-- Given a train and a platform, calculate the speed of the train -/
theorem train_speed (train_length platform_length time : ℝ) 
  (h1 : train_length = 150)
  (h2 : platform_length = 250)
  (h3 : time = 8) :
  (train_length + platform_length) / time = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1870_187060


namespace NUMINAMATH_CALUDE_marty_painting_combinations_l1870_187002

/-- The number of available colors -/
def num_colors : ℕ := 5

/-- The number of available painting methods -/
def num_methods : ℕ := 4

/-- The number of restricted combinations (white paint with spray) -/
def num_restricted : ℕ := 1

theorem marty_painting_combinations :
  (num_colors - 1) * num_methods + (num_methods - 1) = 19 := by
  sorry

end NUMINAMATH_CALUDE_marty_painting_combinations_l1870_187002


namespace NUMINAMATH_CALUDE_factorization_x12_minus_729_l1870_187023

theorem factorization_x12_minus_729 (x : ℝ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3*x^2 + 9) * (x^2 - 3) * (x^4 + 3*x^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x12_minus_729_l1870_187023


namespace NUMINAMATH_CALUDE_right_triangle_circumscribed_circle_perimeter_l1870_187095

theorem right_triangle_circumscribed_circle_perimeter 
  (r : ℝ) (h : ℝ) (a b : ℝ) :
  r = 4 →
  h = 26 →
  a^2 + b^2 = h^2 →
  a * b = 4 * (a + b + h) →
  a + b + h = 60 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_circumscribed_circle_perimeter_l1870_187095


namespace NUMINAMATH_CALUDE_equation_positive_root_l1870_187094

theorem equation_positive_root (x m : ℝ) : 
  (2 / (x - 2) = 1 - m / (x - 2)) → 
  (x > 0) → 
  (m = -2) := by
sorry

end NUMINAMATH_CALUDE_equation_positive_root_l1870_187094


namespace NUMINAMATH_CALUDE_wrong_number_difference_l1870_187027

/-- The number of elements in the set of numbers --/
def n : ℕ := 10

/-- The original average of the numbers --/
def original_average : ℚ := 402/10

/-- The correct average of the numbers --/
def correct_average : ℚ := 403/10

/-- The second wrongly copied number --/
def wrong_second : ℕ := 13

/-- The correct second number --/
def correct_second : ℕ := 31

/-- Theorem stating the difference between the wrongly copied number and the actual number --/
theorem wrong_number_difference (first_wrong : ℚ) (first_actual : ℚ) 
  (h1 : first_wrong > first_actual)
  (h2 : n * original_average = (n - 2) * correct_average + first_wrong + wrong_second)
  (h3 : n * correct_average = (n - 2) * correct_average + first_actual + correct_second) :
  first_wrong - first_actual = 19 := by
  sorry

end NUMINAMATH_CALUDE_wrong_number_difference_l1870_187027


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_5_minus_5_3_l1870_187082

theorem least_prime_factor_of_5_5_minus_5_3 :
  Nat.minFac (5^5 - 5^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_5_minus_5_3_l1870_187082


namespace NUMINAMATH_CALUDE_expansion_without_x3_x2_implies_m_plus_n_eq_neg_4_l1870_187008

theorem expansion_without_x3_x2_implies_m_plus_n_eq_neg_4 
  (m n : ℝ) 
  (h1 : (1 + m) = 0)
  (h2 : (-3*m + n) = 0) :
  m + n = -4 := by
  sorry

end NUMINAMATH_CALUDE_expansion_without_x3_x2_implies_m_plus_n_eq_neg_4_l1870_187008


namespace NUMINAMATH_CALUDE_extreme_points_count_f_nonnegative_range_l1870_187099

/-- The function f(x) defined on (-1, +∞) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * (x^2 - x)

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 1 / (x + 1) + 2 * a * x - a

/-- Theorem about the number of extreme points of f(x) -/
theorem extreme_points_count (a : ℝ) : 
  (a < 0 → ∃! x, x > -1 ∧ f' a x = 0) ∧ 
  (0 ≤ a ∧ a ≤ 8/9 → ∀ x > -1, f' a x ≠ 0) ∧
  (a > 8/9 → ∃ x y, x > -1 ∧ y > -1 ∧ x ≠ y ∧ f' a x = 0 ∧ f' a y = 0) :=
sorry

/-- Theorem about the range of a for which f(x) ≥ 0 when x > 0 -/
theorem f_nonnegative_range : 
  {a : ℝ | ∀ x > 0, f a x ≥ 0} = Set.Icc 0 1 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_count_f_nonnegative_range_l1870_187099


namespace NUMINAMATH_CALUDE_centroid_quadrilateral_area_l1870_187022

-- Define the square ABCD
structure Square :=
  (sideLength : ℝ)

-- Define a point P inside the square
structure PointInSquare :=
  (distanceAP : ℝ)
  (distanceBP : ℝ)

-- Define the quadrilateral formed by centroids
structure CentroidQuadrilateral :=
  (diagonalLength : ℝ)

-- Define the theorem
theorem centroid_quadrilateral_area
  (s : Square)
  (p : PointInSquare)
  (q : CentroidQuadrilateral)
  (h1 : s.sideLength = 30)
  (h2 : p.distanceAP = 12)
  (h3 : p.distanceBP = 26)
  (h4 : q.diagonalLength = 20) :
  q.diagonalLength * q.diagonalLength / 2 = 200 :=
sorry

end NUMINAMATH_CALUDE_centroid_quadrilateral_area_l1870_187022


namespace NUMINAMATH_CALUDE_bookstore_inventory_calculation_l1870_187080

/-- Represents the inventory of a bookstore -/
structure BookInventory where
  fiction : ℕ
  nonFiction : ℕ
  children : ℕ

/-- Represents the sales figures for a day -/
structure DailySales where
  inStoreFiction : ℕ
  inStoreNonFiction : ℕ
  inStoreChildren : ℕ
  online : ℕ

/-- Calculate the total number of books in the inventory -/
def totalBooks (inventory : BookInventory) : ℕ :=
  inventory.fiction + inventory.nonFiction + inventory.children

/-- Calculate the total in-store sales -/
def totalInStoreSales (sales : DailySales) : ℕ :=
  sales.inStoreFiction + sales.inStoreNonFiction + sales.inStoreChildren

theorem bookstore_inventory_calculation 
  (initialInventory : BookInventory)
  (saturdaySales : DailySales)
  (sundayInStoreSalesMultiplier : ℕ)
  (sundayOnlineSalesIncrease : ℕ)
  (newShipment : ℕ)
  (h1 : totalBooks initialInventory = 743)
  (h2 : initialInventory.fiction = 520)
  (h3 : initialInventory.nonFiction = 123)
  (h4 : initialInventory.children = 100)
  (h5 : totalInStoreSales saturdaySales = 37)
  (h6 : saturdaySales.inStoreFiction = 15)
  (h7 : saturdaySales.inStoreNonFiction = 12)
  (h8 : saturdaySales.inStoreChildren = 10)
  (h9 : saturdaySales.online = 128)
  (h10 : sundayInStoreSalesMultiplier = 2)
  (h11 : sundayOnlineSalesIncrease = 34)
  (h12 : newShipment = 160)
  : totalBooks initialInventory - 
    (totalInStoreSales saturdaySales + saturdaySales.online) - 
    (sundayInStoreSalesMultiplier * totalInStoreSales saturdaySales + 
     saturdaySales.online + sundayOnlineSalesIncrease) + 
    newShipment = 502 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_inventory_calculation_l1870_187080


namespace NUMINAMATH_CALUDE_commute_days_is_22_l1870_187064

/-- Represents the commuting options for a day -/
inductive CommuteOption
  | MorningCarEveningBike
  | MorningBikeEveningCar
  | BothCar

/-- Represents the commute data over a period of days -/
structure CommuteData where
  totalDays : ℕ
  morningCar : ℕ
  eveningBike : ℕ
  totalCarCommutes : ℕ

/-- The commute data satisfies the given conditions -/
def validCommuteData (data : CommuteData) : Prop :=
  data.morningCar = 10 ∧
  data.eveningBike = 12 ∧
  data.totalCarCommutes = 14

theorem commute_days_is_22 (data : CommuteData) (h : validCommuteData data) :
  data.totalDays = 22 := by
  sorry

#check commute_days_is_22

end NUMINAMATH_CALUDE_commute_days_is_22_l1870_187064


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1870_187088

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → (x, y) ∈ foci ∨ (x, y) ∉ foci) ∧
  (∃ a b c : ℝ, a^2 = 4 ∧ b^2 = 12 ∧ c^2 = a^2 + b^2 ∧ eccentricity = c / a) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1870_187088


namespace NUMINAMATH_CALUDE_equal_area_intersection_sum_l1870_187029

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (a b c d : Point) : ℚ :=
  (1/2) * abs (a.x * b.y + b.x * c.y + c.x * d.y + d.x * a.y
             - (b.x * a.y + c.x * b.y + d.x * c.y + a.x * d.y))

/-- Checks if a fraction is in its lowest terms -/
def isLowestTerms (p q : ℤ) : Prop :=
  ∀ (d : ℤ), d > 1 → ¬(d ∣ p ∧ d ∣ q)

/-- Main theorem -/
theorem equal_area_intersection_sum (p q r s : ℤ) :
  let a := Point.mk 0 0
  let b := Point.mk 1 3
  let c := Point.mk 4 4
  let d := Point.mk 5 0
  let intersectionPoint := Point.mk (p/q) (r/s)
  quadrilateralArea a b intersectionPoint d = quadrilateralArea b c d intersectionPoint →
  isLowestTerms p q →
  isLowestTerms r s →
  p + q + r + s = 200 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_intersection_sum_l1870_187029


namespace NUMINAMATH_CALUDE_cupboard_sale_percentage_l1870_187042

def cost_price : ℝ := 6875
def additional_amount : ℝ := 1650
def profit_percentage : ℝ := 12

theorem cupboard_sale_percentage (selling_price : ℝ) 
  (h1 : selling_price + additional_amount = cost_price * (1 + profit_percentage / 100)) :
  (cost_price - selling_price) / cost_price * 100 = profit_percentage := by
sorry

end NUMINAMATH_CALUDE_cupboard_sale_percentage_l1870_187042


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1870_187067

theorem quadratic_real_roots (a : ℝ) : 
  a > 1 → ∃ x : ℝ, x^2 - (2*a + 1)*x + a^2 = 0 :=
by
  sorry

#check quadratic_real_roots

end NUMINAMATH_CALUDE_quadratic_real_roots_l1870_187067


namespace NUMINAMATH_CALUDE_no_4digit_square_abba_palindromes_l1870_187089

/-- A function that checks if a number is a 4-digit square --/
def is_4digit_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, m * m = n

/-- A function that checks if a number is a palindrome with two different middle digits (abba form) --/
def is_abba_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    a ≠ b ∧
    n = a * 1000 + b * 100 + b * 10 + a

/-- The main theorem stating that there are no 4-digit squares that are abba palindromes --/
theorem no_4digit_square_abba_palindromes :
  ¬ ∃ n : ℕ, is_4digit_square n ∧ is_abba_palindrome n :=
sorry

end NUMINAMATH_CALUDE_no_4digit_square_abba_palindromes_l1870_187089


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l1870_187020

theorem inequality_system_solutions :
  let S : Set (ℝ × ℝ) := {(x, y) | 
    x^4 + 8*x^3*y + 16*x^2*y^2 + 16 ≤ 8*x^2 + 32*x*y ∧
    y^4 + 64*x^2*y^2 + 10*y^2 + 25 ≤ 16*x*y^3 + 80*x*y}
  S = {(2/Real.sqrt 11, 5/Real.sqrt 11), 
       (-2/Real.sqrt 11, -5/Real.sqrt 11),
       (2/Real.sqrt 3, 1/Real.sqrt 3), 
       (-2/Real.sqrt 3, -1/Real.sqrt 3)} := by
  sorry


end NUMINAMATH_CALUDE_inequality_system_solutions_l1870_187020


namespace NUMINAMATH_CALUDE_angle_ABH_measure_l1870_187050

/-- A regular octagon ABCDEFGH -/
structure RegularOctagon where
  -- Define the octagon (we don't need to specify all vertices, just declare it's regular)
  vertices : Fin 8 → ℝ × ℝ
  is_regular : True  -- We assume it's regular without specifying the conditions

/-- The measure of an angle in a regular octagon -/
def regular_octagon_angle : ℝ := 135

/-- Angle ABH in the regular octagon -/
def angle_ABH (octagon : RegularOctagon) : ℝ :=
  22.5

/-- Theorem: In a regular octagon ABCDEFGH, the measure of angle ABH is 22.5° -/
theorem angle_ABH_measure (octagon : RegularOctagon) :
  angle_ABH octagon = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_angle_ABH_measure_l1870_187050


namespace NUMINAMATH_CALUDE_basketball_match_children_l1870_187065

/-- Calculates the number of children at a basketball match given the total number of spectators,
    the number of men, and the ratio of children to women. -/
def number_of_children (total : ℕ) (men : ℕ) (child_to_woman_ratio : ℕ) : ℕ :=
  let non_men := total - men
  let women := non_men / (child_to_woman_ratio + 1)
  child_to_woman_ratio * women

/-- Theorem stating that given the specific conditions of the basketball match,
    the number of children is 2500. -/
theorem basketball_match_children :
  number_of_children 10000 7000 5 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_basketball_match_children_l1870_187065


namespace NUMINAMATH_CALUDE_min_study_tools_l1870_187075

theorem min_study_tools (n : ℕ) : n^3 ≥ 366 ∧ (n-1)^3 < 366 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_study_tools_l1870_187075


namespace NUMINAMATH_CALUDE_cats_problem_l1870_187092

/-- The number of cats owned by the certain person -/
def person_cats (melanie_cats : ℕ) (annie_cats : ℕ) : ℕ :=
  3 * annie_cats

theorem cats_problem (melanie_cats : ℕ) (annie_cats : ℕ) 
  (h1 : melanie_cats = 2 * annie_cats)
  (h2 : melanie_cats = 60) :
  person_cats melanie_cats annie_cats = 90 := by
  sorry

end NUMINAMATH_CALUDE_cats_problem_l1870_187092


namespace NUMINAMATH_CALUDE_function_properties_monotone_interval_l1870_187043

def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem function_properties (a b : ℝ) :
  f a b 1 = 4 ∧ 
  (3 * a * (-2)^2 + 2 * b * (-2) = 0) →
  a = 1 ∧ b = 3 :=
sorry

def g (x : ℝ) : ℝ := x^3 + 3 * x^2

theorem monotone_interval (m : ℝ) :
  (∀ x ∈ Set.Ioo m (m + 1), MonotoneOn g (Set.Ioo m (m + 1))) →
  m ≤ -3 ∨ m ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_function_properties_monotone_interval_l1870_187043


namespace NUMINAMATH_CALUDE_success_permutations_l1870_187061

/-- The number of letters in the word "SUCCESS" -/
def total_letters : ℕ := 7

/-- The number of times 'S' appears in "SUCCESS" -/
def s_count : ℕ := 3

/-- The number of times 'C' appears in "SUCCESS" -/
def c_count : ℕ := 2

/-- The number of times 'U' appears in "SUCCESS" -/
def u_count : ℕ := 1

/-- The number of times 'E' appears in "SUCCESS" -/
def e_count : ℕ := 1

/-- The number of unique arrangements of the letters in "SUCCESS" -/
def success_arrangements : ℕ := 420

theorem success_permutations :
  Nat.factorial total_letters / (Nat.factorial s_count * Nat.factorial c_count * Nat.factorial u_count * Nat.factorial e_count) = success_arrangements :=
by sorry

end NUMINAMATH_CALUDE_success_permutations_l1870_187061


namespace NUMINAMATH_CALUDE_min_n_for_S_gt_1020_l1870_187028

def S (n : ℕ) : ℕ := 2^(n+1) - 2 - n

theorem min_n_for_S_gt_1020 : ∀ k : ℕ, k ≥ 10 ↔ S k > 1020 :=
sorry

end NUMINAMATH_CALUDE_min_n_for_S_gt_1020_l1870_187028


namespace NUMINAMATH_CALUDE_find_x_l1870_187054

theorem find_x : ∃ x : ℝ, 3 * x = (26 - x) + 18 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1870_187054


namespace NUMINAMATH_CALUDE_polygon_with_900_degree_sum_is_heptagon_l1870_187007

theorem polygon_with_900_degree_sum_is_heptagon :
  ∀ n : ℕ, n ≥ 3 → (n - 2) * 180 = 900 → n = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_900_degree_sum_is_heptagon_l1870_187007


namespace NUMINAMATH_CALUDE_f_inequality_l1870_187052

-- Define f as a differentiable function on ℝ
variable (f : ℝ → ℝ)

-- State the condition that f'(x) - f(x) < 0 for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv f) x - f x < 0)

-- Define e as the mathematical constant e
noncomputable def e : ℝ := Real.exp 1

-- State the theorem to be proved
theorem f_inequality : e * f 2015 > f 2016 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_l1870_187052


namespace NUMINAMATH_CALUDE_num_correct_statements_is_zero_l1870_187040

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Two vectors are parallel -/
def parallel (v1 v2 : Vector3D) : Prop := sorry

/-- A vector is a unit vector -/
def is_unit_vector (v : Vector3D) : Prop := sorry

/-- Two vectors are collinear -/
def collinear (v1 v2 : Vector3D) : Prop := sorry

/-- The zero vector -/
def zero_vector : Vector3D := ⟨0, 0, 0⟩

/-- Theorem: The number of correct statements is 0 -/
theorem num_correct_statements_is_zero : 
  ¬(∀ (v1 v2 : Vector3D) (p : Point3D), is_unit_vector v1 → is_unit_vector v2 → v1.x = p.x ∧ v1.y = p.y ∧ v1.z = p.z → v2.x = p.x ∧ v2.y = p.y ∧ v2.z = p.z) ∧ 
  ¬(∀ (A B C D : Point3D), parallel ⟨B.x - A.x, B.y - A.y, B.z - A.z⟩ ⟨D.x - C.x, D.y - C.y, D.z - C.z⟩ → 
    ∃ (t : ℝ), C = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y), A.z + t * (B.z - A.z)⟩) ∧
  ¬(∀ (a b c : Vector3D), parallel a b → parallel b c → b ≠ zero_vector → parallel a c) ∧
  ¬(∀ (v1 v2 : Vector3D) (A B C D : Point3D), 
    collinear v1 v2 → 
    v1.x = B.x - A.x ∧ v1.y = B.y - A.y ∧ v1.z = B.z - A.z →
    v2.x = D.x - C.x ∧ v2.y = D.y - C.y ∧ v2.z = D.z - C.z →
    A ≠ C → B ≠ D) :=
by sorry

end NUMINAMATH_CALUDE_num_correct_statements_is_zero_l1870_187040


namespace NUMINAMATH_CALUDE_sum_of_decimals_l1870_187025

theorem sum_of_decimals : 0.001 + 1.01 + 0.11 = 1.121 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l1870_187025


namespace NUMINAMATH_CALUDE_circle_areas_sum_l1870_187079

/-- The sum of the areas of an infinite series of circles with radii following
    the geometric sequence 1/√(2^(n-1)) is equal to 2π. -/
theorem circle_areas_sum : 
  let radius (n : ℕ) := (1 : ℝ) / Real.sqrt (2 ^ (n - 1))
  let area (n : ℕ) := Real.pi * (radius n) ^ 2
  (∑' n, area n) = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_areas_sum_l1870_187079


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l1870_187016

theorem candidate_vote_difference (total_votes : ℝ) (candidate_percentage : ℝ) : 
  total_votes = 10000.000000000002 →
  candidate_percentage = 0.4 →
  (total_votes * (1 - candidate_percentage) - total_votes * candidate_percentage) = 2000 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l1870_187016


namespace NUMINAMATH_CALUDE_find_x_value_l1870_187072

theorem find_x_value (x : ℝ) (hx : x ≠ 0) 
  (h : x = (1/x) * (-x) + 3) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l1870_187072


namespace NUMINAMATH_CALUDE_hyperbola_iff_mn_positive_l1870_187066

-- Define the condition for a curve to be a hyperbola
def is_hyperbola (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), m * x^2 - n * y^2 = 1 ↔ (x / a)^2 - (y / b)^2 = 1

-- State the theorem
theorem hyperbola_iff_mn_positive (m n : ℝ) :
  is_hyperbola m n ↔ m * n > 0 := by sorry

end NUMINAMATH_CALUDE_hyperbola_iff_mn_positive_l1870_187066


namespace NUMINAMATH_CALUDE_irrational_product_l1870_187053

-- Define the property of being irrational
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Define the property of being rational
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem irrational_product (x : ℝ) : 
  IsIrrational x → 
  IsRational ((x - 2) * (x + 6)) → 
  IsIrrational ((x + 2) * (x - 6)) := by
  sorry

end NUMINAMATH_CALUDE_irrational_product_l1870_187053


namespace NUMINAMATH_CALUDE_max_value_of_f_l1870_187090

-- Define the function we want to maximize
def f (x : ℤ) : ℝ := 5 - |6 * x - 80|

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℤ), f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1870_187090


namespace NUMINAMATH_CALUDE_perpendicular_iff_a_eq_neg_five_or_one_l1870_187091

def line1 (a : ℝ) (x y : ℝ) : Prop := (2*a + 1)*x + (a + 5)*y - 6 = 0

def line2 (a : ℝ) (x y : ℝ) : Prop := (a + 5)*x + (a - 4)*y + 1 = 0

def perpendicular (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, line1 a x1 y1 ∧ line2 a x2 y2 →
    (2*a + 1)*(a + 5) + (a + 5)*(a - 4) = 0

theorem perpendicular_iff_a_eq_neg_five_or_one :
  ∀ a : ℝ, perpendicular a ↔ a = -5 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_iff_a_eq_neg_five_or_one_l1870_187091


namespace NUMINAMATH_CALUDE_clock_equivalent_hours_l1870_187069

theorem clock_equivalent_hours : 
  ∃ n : ℕ, n > 5 ∧ 
           n * n - n ≡ 0 [MOD 12] ∧ 
           ∀ m : ℕ, m > 5 ∧ m < n → ¬(m * m - m ≡ 0 [MOD 12]) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_clock_equivalent_hours_l1870_187069


namespace NUMINAMATH_CALUDE_train_length_l1870_187055

theorem train_length (v : ℝ) (L : ℝ) : 
  v > 0 → -- The train's speed is positive
  (L + 120) / 60 = v → -- It takes 60 seconds to pass through a 120m tunnel
  L / 20 = v → -- It takes 20 seconds to be completely inside the tunnel
  L = 60 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1870_187055


namespace NUMINAMATH_CALUDE_product_equals_zero_l1870_187039

theorem product_equals_zero (b : ℤ) (h : b = 3) :
  (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) * 
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l1870_187039


namespace NUMINAMATH_CALUDE_calculate_product_l1870_187015

theorem calculate_product : 500 * 1986 * 0.3972 * 100 = 20 * 1986^2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_product_l1870_187015


namespace NUMINAMATH_CALUDE_deduction_is_three_l1870_187036

/-- Calculates the deduction per idle day for a worker --/
def calculate_deduction_per_idle_day (total_days : ℕ) (pay_rate : ℕ) (total_payment : ℕ) (idle_days : ℕ) : ℕ :=
  let working_days := total_days - idle_days
  let total_earnings := working_days * pay_rate
  (total_earnings - total_payment) / idle_days

/-- Theorem: Given the conditions, the deduction per idle day is 3 --/
theorem deduction_is_three :
  calculate_deduction_per_idle_day 60 20 280 40 = 3 := by
  sorry

#eval calculate_deduction_per_idle_day 60 20 280 40

end NUMINAMATH_CALUDE_deduction_is_three_l1870_187036


namespace NUMINAMATH_CALUDE_two_distinct_roots_implies_k_values_l1870_187047

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| + 1
def g (k : ℝ) (x : ℝ) : ℝ := k * x

-- State the theorem
theorem two_distinct_roots_implies_k_values (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g k x₁ ∧ f x₂ = g k x₂) →
  (k = 1/2 ∨ k = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_roots_implies_k_values_l1870_187047


namespace NUMINAMATH_CALUDE_whatsapp_messages_l1870_187014

theorem whatsapp_messages (monday tuesday wednesday thursday : ℕ) :
  monday = 300 →
  tuesday = 200 →
  thursday = 2 * wednesday →
  monday + tuesday + wednesday + thursday = 2000 →
  wednesday - tuesday = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_whatsapp_messages_l1870_187014


namespace NUMINAMATH_CALUDE_a_5_value_l1870_187026

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 = -9 →
  a 7 = -1 →
  a 5 = -3 := by
sorry

end NUMINAMATH_CALUDE_a_5_value_l1870_187026


namespace NUMINAMATH_CALUDE_max_value_chord_intersection_l1870_187058

theorem max_value_chord_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ (2*a*x + b*y = 2) ∧
   ∃ (x1 y1 x2 y2 : ℝ), x1^2 + y1^2 = 4 ∧ x2^2 + y2^2 = 4 ∧
   2*a*x1 + b*y1 = 2 ∧ 2*a*x2 + b*y2 = 2 ∧
   (x1 - x2)^2 + (y1 - y2)^2 = 12) →
  (∀ c : ℝ, c ≤ (9 * Real.sqrt 2) / 8 ∨ ∃ d : ℝ, d > c ∧ d = a * Real.sqrt (1 + 2*b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_chord_intersection_l1870_187058


namespace NUMINAMATH_CALUDE_min_omega_value_l1870_187024

open Real

/-- Given a function f(x) = 2sin(ωx + φ) where ω > 0, 
    if the graph is symmetrical about the line x = π/3 and f(π/12) = 0, 
    then the minimum value of ω is 2. -/
theorem min_omega_value (ω φ : ℝ) (hω : ω > 0) :
  (∀ x, 2 * sin (ω * x + φ) = 2 * sin (ω * (2 * π/3 - x) + φ)) →
  2 * sin (ω * π/12 + φ) = 0 →
  ω ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_value_l1870_187024


namespace NUMINAMATH_CALUDE_typing_time_l1870_187062

/-- Proves that given Tom's typing speed and page length, it takes 50 minutes to type 10 pages -/
theorem typing_time (typing_speed : ℕ) (words_per_page : ℕ) (pages : ℕ) : 
  typing_speed = 90 → words_per_page = 450 → pages = 10 → 
  (pages * words_per_page) / typing_speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_l1870_187062


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1870_187071

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {3, 7, 9}
def B : Set ℕ := {1, 9}

theorem intersection_complement_equality : A ∩ (U \ B) = {3, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1870_187071


namespace NUMINAMATH_CALUDE_quadratic_roots_implies_composite_l1870_187013

/-- A number is composite if it's the product of two integers each greater than 1 -/
def IsComposite (n : ℕ) : Prop :=
  ∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ n = p * q

/-- The roots of the quadratic x^2 + ax + b + 1 are positive integers -/
def HasPositiveIntegerRoots (a b : ℤ) : Prop :=
  ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c^2 + a*c + b + 1 = 0 ∧ d^2 + a*d + b + 1 = 0

/-- If x^2 + ax + b + 1 has positive integer roots, then a^2 + b^2 is composite -/
theorem quadratic_roots_implies_composite (a b : ℤ) :
  HasPositiveIntegerRoots a b → IsComposite (Int.natAbs (a^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_implies_composite_l1870_187013


namespace NUMINAMATH_CALUDE_paint_calculation_l1870_187076

/-- The amount of paint Joe uses given the initial amount and usage fractions -/
def paint_used (initial : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) : ℚ :=
  let first_week := first_week_fraction * initial
  let remaining := initial - first_week
  let second_week := second_week_fraction * remaining
  first_week + second_week

/-- Theorem stating that given 360 gallons of paint, if 2/3 is used in the first week
    and 1/5 of the remainder is used in the second week, the total amount of paint used is 264 gallons -/
theorem paint_calculation :
  paint_used 360 (2/3) (1/5) = 264 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l1870_187076


namespace NUMINAMATH_CALUDE_gcd_1337_382_l1870_187030

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1337_382_l1870_187030


namespace NUMINAMATH_CALUDE_sunny_lead_second_race_l1870_187045

/-- Represents a runner in the races -/
structure Runner where
  speed : ℝ

/-- Represents the race conditions -/
structure RaceConditions where
  raceLength : ℝ
  sunnyLeadFirstRace : ℝ
  sunnyStartBehind : ℝ
  windyDelay : ℝ

/-- Calculate the lead of Sunny at the end of the second race -/
def calculateSunnyLead (sunny : Runner) (windy : Runner) (conditions : RaceConditions) : ℝ :=
  sorry

/-- Theorem stating that Sunny finishes 56.25 meters ahead in the second race -/
theorem sunny_lead_second_race (sunny : Runner) (windy : Runner) (conditions : RaceConditions) :
  conditions.raceLength = 400 ∧
  conditions.sunnyLeadFirstRace = 50 ∧
  conditions.sunnyStartBehind = 50 ∧
  conditions.windyDelay = 10 →
  calculateSunnyLead sunny windy conditions = 56.25 :=
by
  sorry

end NUMINAMATH_CALUDE_sunny_lead_second_race_l1870_187045


namespace NUMINAMATH_CALUDE_mysoon_ornament_collection_l1870_187041

theorem mysoon_ornament_collection :
  ∀ (O : ℕ), 
    (O / 6 + 10 : ℕ) = (O / 3 : ℕ) * 2 →  -- Condition 1 and 2 combined
    (O / 3 : ℕ) = O / 3 →                 -- Condition 3
    O = 20 := by
  sorry

end NUMINAMATH_CALUDE_mysoon_ornament_collection_l1870_187041


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_negative_55_l1870_187097

theorem alpha_plus_beta_equals_negative_55 :
  ∀ α β : ℝ, 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 50*x + 621) / (x^2 + 75*x - 2016)) →
  α + β = -55 :=
by sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_negative_55_l1870_187097


namespace NUMINAMATH_CALUDE_grasshopper_can_return_to_start_l1870_187005

/-- Represents the position of the grasshopper on a 2D plane -/
structure Position where
  x : Int
  y : Int

/-- Represents a single jump of the grasshopper -/
structure Jump where
  distance : Nat
  direction : Nat  -- 0: right, 1: up, 2: left, 3: down

/-- Applies a jump to a position -/
def applyJump (pos : Position) (jump : Jump) : Position :=
  match jump.direction % 4 with
  | 0 => ⟨pos.x + jump.distance, pos.y⟩
  | 1 => ⟨pos.x, pos.y + jump.distance⟩
  | 2 => ⟨pos.x - jump.distance, pos.y⟩
  | _ => ⟨pos.x, pos.y - jump.distance⟩

/-- Generates the nth jump -/
def nthJump (n : Nat) : Jump :=
  ⟨n, n - 1⟩

/-- Theorem: The grasshopper can return to the starting point -/
theorem grasshopper_can_return_to_start :
  ∃ (jumps : List Jump), 
    let finalPos := jumps.foldl applyJump ⟨0, 0⟩
    finalPos.x = 0 ∧ finalPos.y = 0 :=
  sorry


end NUMINAMATH_CALUDE_grasshopper_can_return_to_start_l1870_187005


namespace NUMINAMATH_CALUDE_rectangle_diagonal_intersection_l1870_187083

/-- Given a rectangle with opposite vertices at (2, -3) and (14, 9),
    prove that its diagonals intersect at the point (8, 3). -/
theorem rectangle_diagonal_intersection :
  let a : ℝ × ℝ := (2, -3)
  let c : ℝ × ℝ := (14, 9)
  let midpoint : ℝ × ℝ := ((a.1 + c.1) / 2, (a.2 + c.2) / 2)
  midpoint = (8, 3) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_intersection_l1870_187083


namespace NUMINAMATH_CALUDE_new_person_weight_l1870_187086

theorem new_person_weight (initial_total : ℝ) (h1 : initial_total > 0) : 
  let initial_avg := initial_total / 5
  let new_avg := initial_avg + 4
  let new_total := new_avg * 5
  new_total - (initial_total - 50) = 70 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1870_187086


namespace NUMINAMATH_CALUDE_systematic_sampling_524_l1870_187078

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  populationSize : Nat
  samplingInterval : Nat

/-- Checks if the sampling interval divides the population size evenly -/
def SystematicSampling.isValidInterval (s : SystematicSampling) : Prop :=
  s.populationSize % s.samplingInterval = 0

theorem systematic_sampling_524 :
  ∃ (s : SystematicSampling), s.populationSize = 524 ∧ s.samplingInterval = 4 ∧ s.isValidInterval :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_524_l1870_187078


namespace NUMINAMATH_CALUDE_snakes_not_hiding_l1870_187017

/-- Given a cage with snakes, some of which are hiding, calculate the number of snakes not hiding. -/
theorem snakes_not_hiding (total_snakes hiding_snakes : ℕ) 
  (h1 : total_snakes = 95)
  (h2 : hiding_snakes = 64) :
  total_snakes - hiding_snakes = 31 := by
  sorry

end NUMINAMATH_CALUDE_snakes_not_hiding_l1870_187017


namespace NUMINAMATH_CALUDE_four_Z_three_l1870_187031

-- Define the Z operation
def Z (x y : ℤ) : ℤ := x^2 - 3*x*y + y^2

-- Theorem to prove
theorem four_Z_three : Z 4 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_three_l1870_187031


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1870_187000

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_of_A_in_U : 
  Set.compl A = Set.Icc (-1 : ℝ) (3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1870_187000
