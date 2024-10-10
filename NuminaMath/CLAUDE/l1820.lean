import Mathlib

namespace quadratic_inequality_range_l1820_182058

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - 3 < 0) ↔ k ∈ Set.Ioc (-3) 0 :=
sorry

end quadratic_inequality_range_l1820_182058


namespace divisible_by_nine_l1820_182019

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem divisible_by_nine (N : ℕ) : 
  sum_of_digits N = sum_of_digits (5 * N) → N % 9 = 0 := by sorry

end divisible_by_nine_l1820_182019


namespace artist_painted_thirteen_pictures_l1820_182045

/-- The number of pictures painted by an artist over three months -/
def total_pictures (june july august : ℕ) : ℕ := june + july + august

/-- Theorem stating that the artist painted 13 pictures in total -/
theorem artist_painted_thirteen_pictures : 
  total_pictures 2 2 9 = 13 := by sorry

end artist_painted_thirteen_pictures_l1820_182045


namespace perpendicular_vector_with_sum_condition_l1820_182082

/-- Given two parallel lines l and m with direction vector (4, 3),
    prove that (-6, 8) is perpendicular to their direction vector
    and its components sum to 2. -/
theorem perpendicular_vector_with_sum_condition :
  let direction_vector : ℝ × ℝ := (4, 3)
  let perpendicular_vector : ℝ × ℝ := (-6, 8)
  (direction_vector.1 * perpendicular_vector.1 + direction_vector.2 * perpendicular_vector.2 = 0) ∧
  (perpendicular_vector.1 + perpendicular_vector.2 = 2) := by
  sorry


end perpendicular_vector_with_sum_condition_l1820_182082


namespace unique_root_condition_l1820_182062

theorem unique_root_condition (k : ℝ) : 
  (∃! x : ℝ, (x / (x + 3) + x / (x + 4) = k * x)) ↔ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) := by
  sorry

end unique_root_condition_l1820_182062


namespace music_talent_sample_l1820_182024

/-- Represents the number of students selected in a stratified sampling -/
def stratified_sample (total_population : ℕ) (group_size : ℕ) (sample_size : ℕ) : ℕ :=
  (group_size * sample_size) / total_population

/-- Proves that in a stratified sampling of 40 students from a population of 100 students,
    where 40 students have music talent, the number of music-talented students selected is 16 -/
theorem music_talent_sample :
  stratified_sample 100 40 40 = 16 := by
  sorry

end music_talent_sample_l1820_182024


namespace quadratic_two_distinct_roots_l1820_182033

theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k = 0 ∧ y^2 - 2*y + k = 0) → k < 1 :=
by sorry

end quadratic_two_distinct_roots_l1820_182033


namespace positive_integers_relation_l1820_182025

theorem positive_integers_relation (a b : ℕ) : 
  a > 0 → b > 0 → (a, b) ≠ (1, 1) → (a * b - 1) ∣ (a^2 + b^2) → a^2 + b^2 = 5 * a * b - 5 := by
  sorry

end positive_integers_relation_l1820_182025


namespace add_twenty_four_thirty_six_l1820_182031

theorem add_twenty_four_thirty_six : 24 + 36 = 60 := by
  sorry

end add_twenty_four_thirty_six_l1820_182031


namespace prob_sum_three_is_one_over_216_l1820_182084

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_die : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The target sum we're aiming for -/
def target_sum : ℕ := 3

/-- The probability of rolling a sum of 3 with three fair six-sided dice -/
def prob_sum_three : ℚ := prob_single_die ^ num_dice

theorem prob_sum_three_is_one_over_216 : 
  prob_sum_three = 1 / 216 := by sorry

end prob_sum_three_is_one_over_216_l1820_182084


namespace oranges_per_bag_l1820_182065

theorem oranges_per_bag (total_oranges : ℕ) (num_bags : ℕ) (h1 : total_oranges = 1035) (h2 : num_bags = 45) (h3 : total_oranges % num_bags = 0) : 
  total_oranges / num_bags = 23 := by
sorry

end oranges_per_bag_l1820_182065


namespace stratified_sampling_female_count_stratified_sampling_female_count_correct_l1820_182092

theorem stratified_sampling_female_count 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 75) 
  (h2 : male_employees = 30) 
  (h3 : sample_size = 20) : ℕ :=
let female_employees := total_employees - male_employees
let sample_ratio := sample_size / total_employees
let female_sample := (female_employees : ℚ) * sample_ratio
12

theorem stratified_sampling_female_count_correct 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 75) 
  (h2 : male_employees = 30) 
  (h3 : sample_size = 20) : 
  stratified_sampling_female_count total_employees male_employees sample_size h1 h2 h3 = 12 := by
sorry

end stratified_sampling_female_count_stratified_sampling_female_count_correct_l1820_182092


namespace max_sum_of_goods_l1820_182004

theorem max_sum_of_goods (m n : ℕ) : m > 0 ∧ n > 0 ∧ 5 * m + 17 * n = 203 → m + n ≤ 31 := by
  sorry

end max_sum_of_goods_l1820_182004


namespace charlotte_tuesday_poodles_l1820_182086

/-- Represents the schedule for Charlotte's dog walking --/
structure DogWalkingSchedule where
  poodles_monday : ℕ
  chihuahuas_monday : ℕ
  labradors_wednesday : ℕ
  poodle_time : ℕ
  chihuahua_time : ℕ
  labrador_time : ℕ
  total_time : ℕ

/-- Calculates the number of poodles Charlotte can walk on Tuesday --/
def poodles_tuesday (s : DogWalkingSchedule) : ℕ :=
  let monday_time := s.poodles_monday * s.poodle_time + s.chihuahuas_monday * s.chihuahua_time
  let wednesday_time := s.labradors_wednesday * s.labrador_time
  let tuesday_time := s.total_time - monday_time - wednesday_time - s.chihuahuas_monday * s.chihuahua_time
  tuesday_time / s.poodle_time

/-- Theorem stating that Charlotte can walk 4 poodles on Tuesday --/
theorem charlotte_tuesday_poodles (s : DogWalkingSchedule) 
  (h1 : s.poodles_monday = 4)
  (h2 : s.chihuahuas_monday = 2)
  (h3 : s.labradors_wednesday = 4)
  (h4 : s.poodle_time = 2)
  (h5 : s.chihuahua_time = 1)
  (h6 : s.labrador_time = 3)
  (h7 : s.total_time = 32) :
  poodles_tuesday s = 4 := by
  sorry


end charlotte_tuesday_poodles_l1820_182086


namespace salary_percentage_difference_l1820_182063

theorem salary_percentage_difference (a b : ℝ) (h : b = 1.25 * a) :
  (b - a) / b = 0.2 := by
  sorry

end salary_percentage_difference_l1820_182063


namespace four_digit_number_theorem_l1820_182085

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem four_digit_number_theorem (n : ℕ) : 
  is_four_digit n ∧
  (∃ k : ℕ, n + 1 = 15 * k) ∧
  (∃ m : ℕ, n - 3 = 38 * m) ∧
  (∃ l : ℕ, n + reverse_digits n = 10 * l) →
  n = 1409 ∨ n = 1979 :=
by sorry

end four_digit_number_theorem_l1820_182085


namespace quadratic_solution_l1820_182054

theorem quadratic_solution : ∃! x : ℝ, x > 0 ∧ 3 * x^2 + 13 * x - 10 = 0 :=
by
  use 2/3
  sorry

end quadratic_solution_l1820_182054


namespace cricket_game_solution_l1820_182046

def cricket_game (initial_run_rate : ℝ) (required_rate : ℝ) (total_target : ℝ) : Prop :=
  ∃ (initial_overs : ℝ),
    initial_overs > 0 ∧
    initial_overs < 50 ∧
    initial_overs + 40 = 50 ∧
    initial_run_rate * initial_overs + required_rate * 40 = total_target

theorem cricket_game_solution :
  cricket_game 3.2 5.5 252 → ∃ (initial_overs : ℝ), initial_overs = 10 := by
  sorry

end cricket_game_solution_l1820_182046


namespace isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l1820_182097

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 25 cm has a base of 11 cm. -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruent_side := 7
    let perimeter := 25
    (2 * congruent_side + base = perimeter) →
    base = 11

-- The proof is omitted
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 11 := by
  sorry

end isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l1820_182097


namespace matrix_power_2020_l1820_182051

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 3, 1]

theorem matrix_power_2020 :
  A ^ 2020 = !![1, 0; 6060, 1] := by sorry

end matrix_power_2020_l1820_182051


namespace standardDeviation_best_stability_measure_l1820_182090

-- Define the type for crop yields
def CropYield := ℝ

-- Define a list of crop yields
def YieldList := List CropYield

-- Define statistical measures
def mean (yields : YieldList) : ℝ := sorry
def standardDeviation (yields : YieldList) : ℝ := sorry
def maximum (yields : YieldList) : ℝ := sorry
def median (yields : YieldList) : ℝ := sorry

-- Define a measure of stability
def stabilityMeasure : (YieldList → ℝ) → Prop := sorry

-- Theorem statement
theorem standardDeviation_best_stability_measure :
  ∀ (yields : YieldList),
    stabilityMeasure standardDeviation ∧
    ¬stabilityMeasure mean ∧
    ¬stabilityMeasure maximum ∧
    ¬stabilityMeasure median :=
  sorry

end standardDeviation_best_stability_measure_l1820_182090


namespace students_playing_neither_sport_l1820_182016

theorem students_playing_neither_sport (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ)
  (h_total : total = 38)
  (h_football : football = 26)
  (h_tennis : tennis = 20)
  (h_both : both = 17) :
  total - (football + tennis - both) = 9 :=
by sorry

end students_playing_neither_sport_l1820_182016


namespace parallelogram_to_triangle_impossibility_l1820_182074

theorem parallelogram_to_triangle_impossibility (a : ℝ) (h : a > 0) :
  ¬ (a + a > 2*a ∧ a + 2*a > a ∧ 2*a + a > a) :=
by sorry

end parallelogram_to_triangle_impossibility_l1820_182074


namespace quadratic_root_sum_equality_l1820_182009

theorem quadratic_root_sum_equality (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h₁ : b₁^2 - 4*c₁ = 1)
  (h₂ : b₂^2 - 4*c₂ = 4)
  (h₃ : b₃^2 - 4*c₃ = 9) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁^2 + b₁*x₁ + c₁ = 0) ∧
    (y₁^2 + b₁*y₁ + c₁ = 0) ∧
    (x₂^2 + b₂*x₂ + c₂ = 0) ∧
    (y₂^2 + b₂*y₂ + c₂ = 0) ∧
    (x₃^2 + b₃*x₃ + c₃ = 0) ∧
    (y₃^2 + b₃*y₃ + c₃ = 0) ∧
    (x₁ + x₂ + y₃ = y₁ + y₂ + x₃) := by
  sorry

end quadratic_root_sum_equality_l1820_182009


namespace cube_difference_positive_l1820_182049

theorem cube_difference_positive {a b : ℝ} (h : a > b) : a^3 - b^3 > 0 := by
  sorry

end cube_difference_positive_l1820_182049


namespace multiplication_equality_l1820_182034

theorem multiplication_equality : 500 * 3986 * 0.3986 * 5 = 0.25 * 3986^2 := by
  sorry

end multiplication_equality_l1820_182034


namespace age_difference_theorem_l1820_182036

/-- Represents a two-digit number --/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≤ 9
  h_ones : ones ≤ 9
  h_not_zero : tens ≠ 0

/-- The value of a two-digit number --/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem age_difference_theorem (jack bill : TwoDigitNumber)
    (h_reversed : jack.tens = bill.ones ∧ jack.ones = bill.tens)
    (h_future : jack.value + 6 = 3 * (bill.value + 6)) :
    jack.value - bill.value = 36 := by
  sorry

end age_difference_theorem_l1820_182036


namespace inequality_holds_l1820_182073

theorem inequality_holds (a b c : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : b > c) :
  a / c^2 > b / c^2 := by
  sorry

end inequality_holds_l1820_182073


namespace lansing_new_students_average_l1820_182026

/-- The average number of new students per school in Lansing -/
def average_new_students_per_school (total_schools : Float) (total_new_students : Float) : Float :=
  total_new_students / total_schools

/-- Theorem: The average number of new students per school in Lansing is 9.88 -/
theorem lansing_new_students_average :
  let total_schools : Float := 25.0
  let total_new_students : Float := 247.0
  average_new_students_per_school total_schools total_new_students = 9.88 := by
  sorry

end lansing_new_students_average_l1820_182026


namespace fruitBaskets_eq_96_l1820_182015

/-- The number of ways to choose k items from n identical items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of fruit baskets with at least 3 pieces of fruit,
    given 7 apples and 12 oranges -/
def fruitBaskets : ℕ :=
  let totalBaskets := (choose 7 0 + choose 7 1 + choose 7 2 + choose 7 3 +
                       choose 7 4 + choose 7 5 + choose 7 6 + choose 7 7) *
                      (choose 12 0 + choose 12 1 + choose 12 2 + choose 12 3 +
                       choose 12 4 + choose 12 5 + choose 12 6 + choose 12 7 +
                       choose 12 8 + choose 12 9 + choose 12 10 + choose 12 11 +
                       choose 12 12)
  let invalidBaskets := choose 7 0 * choose 12 0 +
                        choose 7 0 * choose 12 1 +
                        choose 7 0 * choose 12 2 +
                        choose 7 1 * choose 12 0 +
                        choose 7 1 * choose 12 1 +
                        choose 7 2 * choose 12 0
  totalBaskets - invalidBaskets

theorem fruitBaskets_eq_96 : fruitBaskets = 96 := by sorry

end fruitBaskets_eq_96_l1820_182015


namespace product_pass_rate_l1820_182061

/-- The pass rate of a product going through two independent processing steps -/
def pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

/-- Theorem stating that the pass rate of a product going through two independent processing steps
    with defect rates a and b is (1-a)·(1-b) -/
theorem product_pass_rate (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  pass_rate a b = (1 - a) * (1 - b) := by
  sorry

end product_pass_rate_l1820_182061


namespace range_of_a_l1820_182012

-- Define the propositions p and q
def p (x : ℝ) : Prop := 1 ≤ x ∧ x < 3
def q (x a : ℝ) : Prop := x^2 - a*x ≤ x - a

-- Define the range of a
def a_range (a : ℝ) : Prop := 1 ≤ a ∧ a < 3

-- State the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a))) →
  (∀ a : ℝ, a_range a) :=
sorry

end range_of_a_l1820_182012


namespace k_equals_nine_l1820_182047

/-- Two circles centered at the origin with specific points and distances -/
structure TwoCircles where
  -- Radius of the larger circle
  R : ℝ
  -- Radius of the smaller circle
  r : ℝ
  -- Point P on the larger circle
  P : ℝ × ℝ
  -- Point S on the smaller circle
  S : ℝ × ℝ
  -- Distance QR
  QR : ℝ
  -- Conditions
  P_on_larger : P.1^2 + P.2^2 = R^2
  S_on_smaller : S.1^2 + S.2^2 = r^2
  P_coords : P = (5, 12)
  S_coords : S = (0, S.2)
  QR_value : QR = 4

/-- The theorem stating that k (the y-coordinate of S) equals 9 -/
theorem k_equals_nine (c : TwoCircles) : c.S.2 = 9 := by
  sorry

end k_equals_nine_l1820_182047


namespace polynomial_floor_property_l1820_182000

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The floor function -/
noncomputable def floor : ℝ → ℤ := sorry

/-- The property that P(⌊x⌋) = ⌊P(x)⌋ for all real x -/
def HasFloorProperty (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, P (↑(floor x)) = ↑(floor (P x))

/-- The main theorem -/
theorem polynomial_floor_property (P : RealPolynomial) :
  HasFloorProperty P → ∃ k : ℤ, ∀ x : ℝ, P x = x + k := by sorry

end polynomial_floor_property_l1820_182000


namespace selection_problem_l1820_182089

theorem selection_problem (total : Nat) (translation_capable : Nat) (software_capable : Nat) 
  (both_capable : Nat) (to_select : Nat) (for_translation : Nat) (for_software : Nat) :
  total = 8 →
  translation_capable = 5 →
  software_capable = 4 →
  both_capable = 1 →
  to_select = 5 →
  for_translation = 3 →
  for_software = 2 →
  (Nat.choose (translation_capable - 1) for_translation * 
   Nat.choose (software_capable - 1) for_software) +
  (Nat.choose (translation_capable - 1) (for_translation - 1) * 
   Nat.choose software_capable for_software) +
  (Nat.choose translation_capable for_translation * 
   Nat.choose (software_capable - 1) (for_software - 1)) = 42 := by
  sorry

#check selection_problem

end selection_problem_l1820_182089


namespace same_color_probability_problem_die_l1820_182091

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (golden : ℕ)
  (total : ℕ)
  (h_total : red + green + blue + golden = total)

/-- The probability of rolling the same color on two identical colored dice -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.red^2 + d.green^2 + d.blue^2 + d.golden^2 : ℚ) / d.total^2

/-- The specific 30-sided die described in the problem -/
def problem_die : ColoredDie :=
  { red := 6
  , green := 8
  , blue := 10
  , golden := 6
  , total := 30
  , h_total := by simp }

/-- Theorem stating the probability of rolling the same color on two problem_die is 59/225 -/
theorem same_color_probability_problem_die :
  same_color_probability problem_die = 59 / 225 := by
  sorry

end same_color_probability_problem_die_l1820_182091


namespace min_value_A_over_C_l1820_182018

theorem min_value_A_over_C (x A C : ℝ) (hx : x > 0) (hA : A > 0) (hC : C > 0)
  (h1 : x^2 + 1/x^2 = A) (h2 : x - 1/x = C) (h3 : C = Real.sqrt 3) :
  A / C ≥ 5 * Real.sqrt 3 / 3 := by
sorry

end min_value_A_over_C_l1820_182018


namespace rectangle_ratio_l1820_182057

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0)
  (h4 : s + 2*y = 3*s) -- outer square side length
  (h5 : x + s = 3*s) -- outer square side length
  (h6 : (3*s)^2 = 9*s^2) -- area relation
  : x / y = 2 :=
by sorry

end rectangle_ratio_l1820_182057


namespace first_month_sale_is_5400_l1820_182056

/-- Calculates the sale in the first month given the sales for the next 5 months and the average sale for 6 months -/
def first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale2 + sale3 + sale4 + sale5 + sale6)

/-- Theorem stating that the sale in the first month is 5400 given the specific sales figures -/
theorem first_month_sale_is_5400 :
  first_month_sale 9000 6300 7200 4500 1200 5600 = 5400 := by
  sorry

#eval first_month_sale 9000 6300 7200 4500 1200 5600

end first_month_sale_is_5400_l1820_182056


namespace smallest_positive_integer_congruence_l1820_182011

theorem smallest_positive_integer_congruence (x : ℕ) : x = 29 ↔ 
  x > 0 ∧
  (5 * x) % 20 = 25 % 20 ∧
  (3 * x + 1) % 7 = 4 % 7 ∧
  (2 * x - 3) % 13 = x % 13 ∧
  ∀ y : ℕ, y > 0 → 
    ((5 * y) % 20 = 25 % 20 ∧
     (3 * y + 1) % 7 = 4 % 7 ∧
     (2 * y - 3) % 13 = y % 13) → 
    x ≤ y :=
by sorry

end smallest_positive_integer_congruence_l1820_182011


namespace max_sum_ab_l1820_182094

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Given four distinct digits A, B, C, D, where (A+B)/(C+D) is an integer
    and C+D > 1, the maximum value of A+B is 15 -/
theorem max_sum_ab (A B C D : Digit) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_integer : ∃ k : ℕ+, k * (C.val + D.val) = A.val + B.val)
  (h_cd_gt_one : C.val + D.val > 1) :
  A.val + B.val ≤ 15 :=
sorry

end max_sum_ab_l1820_182094


namespace davids_chemistry_marks_l1820_182002

theorem davids_chemistry_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (biology : ℕ)
  (average : ℚ)
  (h1 : english = 96)
  (h2 : mathematics = 95)
  (h3 : physics = 82)
  (h4 : biology = 92)
  (h5 : average = 90.4)
  (h6 : (english + mathematics + physics + biology + chemistry : ℚ) / 5 = average) :
  chemistry = 87 :=
by sorry

end davids_chemistry_marks_l1820_182002


namespace translation_result_l1820_182055

def point_translation (x y dx dy : ℝ) : ℝ × ℝ :=
  (x + dx, y - dy)

theorem translation_result :
  point_translation (-2) 3 3 1 = (1, 2) := by
  sorry

end translation_result_l1820_182055


namespace ferris_wheel_theorem_l1820_182099

/-- The number of people who can ride a Ferris wheel at the same time -/
def ferris_wheel_capacity (seats : ℕ) (people_per_seat : ℕ) : ℕ :=
  seats * people_per_seat

/-- Theorem: The capacity of a Ferris wheel with 2 seats and 2 people per seat is 4 -/
theorem ferris_wheel_theorem : ferris_wheel_capacity 2 2 = 4 := by
  sorry

end ferris_wheel_theorem_l1820_182099


namespace distribute_six_balls_three_boxes_l1820_182078

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 7 := by
  sorry

end distribute_six_balls_three_boxes_l1820_182078


namespace museum_ticket_fraction_l1820_182052

def total_amount : ℚ := 180
def sandwich_fraction : ℚ := 1/5
def book_fraction : ℚ := 1/2
def leftover_amount : ℚ := 24

theorem museum_ticket_fraction :
  let spent_amount := total_amount - leftover_amount
  let sandwich_cost := sandwich_fraction * total_amount
  let book_cost := book_fraction * total_amount
  let museum_ticket_cost := spent_amount - sandwich_cost - book_cost
  museum_ticket_cost / total_amount = 1/6 := by sorry

end museum_ticket_fraction_l1820_182052


namespace red_balls_count_l1820_182071

/-- The number of blue balls in the bin -/
def blue_balls : ℕ := 7

/-- The amount won when drawing a blue ball -/
def blue_win : ℤ := 3

/-- The amount lost when drawing a red ball -/
def red_loss : ℤ := 1

/-- The expected value of the game -/
def expected_value : ℚ := 1

/-- The number of red balls in the bin -/
def red_balls : ℕ := sorry

theorem red_balls_count : red_balls = 7 := by sorry

end red_balls_count_l1820_182071


namespace complex_multiplication_l1820_182006

theorem complex_multiplication (i : ℂ) : i * i = -1 → (1 + i) * (1 - i) = 1 := by
  sorry

end complex_multiplication_l1820_182006


namespace tourist_tax_calculation_l1820_182095

/-- Represents the tax system in Country B -/
structure TaxSystem where
  taxFreeLimit : ℝ
  bracket1Rate : ℝ
  bracket1Limit : ℝ
  bracket2Rate : ℝ
  bracket2Limit : ℝ
  bracket3Rate : ℝ
  electronicsRate : ℝ
  luxuryRate : ℝ
  studentDiscount : ℝ

/-- Represents a purchase made by a tourist -/
structure Purchase where
  totalValue : ℝ
  electronicsValue : ℝ
  luxuryValue : ℝ
  educationalValue : ℝ
  hasStudentID : Bool

def calculateTax (system : TaxSystem) (purchase : Purchase) : ℝ :=
  sorry

theorem tourist_tax_calculation (system : TaxSystem) (purchase : Purchase) :
  system.taxFreeLimit = 600 ∧
  system.bracket1Rate = 0.12 ∧
  system.bracket1Limit = 1000 ∧
  system.bracket2Rate = 0.18 ∧
  system.bracket2Limit = 1500 ∧
  system.bracket3Rate = 0.25 ∧
  system.electronicsRate = 0.05 ∧
  system.luxuryRate = 0.10 ∧
  system.studentDiscount = 0.05 ∧
  purchase.totalValue = 2100 ∧
  purchase.electronicsValue = 900 ∧
  purchase.luxuryValue = 820 ∧
  purchase.educationalValue = 380 ∧
  purchase.hasStudentID = true
  →
  calculateTax system purchase = 304 :=
by sorry

end tourist_tax_calculation_l1820_182095


namespace triangle_ratio_theorem_l1820_182048

theorem triangle_ratio_theorem (A B C : Real) (hTriangle : A + B + C = PI) 
  (hCondition : 3 * Real.sin B * Real.cos C = Real.sin C * (1 - 3 * Real.cos B)) : 
  Real.sin C / Real.sin A = 3 := by
  sorry

end triangle_ratio_theorem_l1820_182048


namespace initial_packages_l1820_182042

theorem initial_packages (cupcakes_per_package : ℕ) (eaten_cupcakes : ℕ) (remaining_cupcakes : ℕ) :
  cupcakes_per_package = 4 →
  eaten_cupcakes = 5 →
  remaining_cupcakes = 7 →
  (eaten_cupcakes + remaining_cupcakes) / cupcakes_per_package = 3 :=
by
  sorry

end initial_packages_l1820_182042


namespace football_club_balance_l1820_182040

/-- Represents the balance and transactions of a football club --/
structure FootballClub where
  initialBalance : ℝ
  playersSold : ℕ
  sellingPrice : ℝ
  playerAPrice : ℝ
  playerBPrice : ℝ
  playerCPrice : ℝ
  playerDPrice : ℝ
  eurToUsd : ℝ
  gbpToUsd : ℝ
  jpyToUsd : ℝ

/-- Calculates the final balance of the football club after transactions --/
def finalBalance (club : FootballClub) : ℝ :=
  club.initialBalance +
  club.playersSold * club.sellingPrice -
  (club.playerAPrice * club.eurToUsd +
   club.playerBPrice * club.gbpToUsd +
   club.playerCPrice * club.jpyToUsd +
   club.playerDPrice * club.eurToUsd)

/-- Theorem stating that the final balance of the football club is 71.4 million USD --/
theorem football_club_balance (club : FootballClub)
  (h1 : club.initialBalance = 100)
  (h2 : club.playersSold = 2)
  (h3 : club.sellingPrice = 10)
  (h4 : club.playerAPrice = 12)
  (h5 : club.playerBPrice = 8)
  (h6 : club.playerCPrice = 1000)
  (h7 : club.playerDPrice = 9)
  (h8 : club.eurToUsd = 1.3)
  (h9 : club.gbpToUsd = 1.6)
  (h10 : club.jpyToUsd = 0.0085) :
  finalBalance club = 71.4 := by
  sorry

end football_club_balance_l1820_182040


namespace triangle_area_is_one_third_of_square_l1820_182093

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in 2D space -/
structure Square where
  bottomLeft : Point
  topRight : Point

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ :=
  (s.topRight.x - s.bottomLeft.x) * (s.topRight.y - s.bottomLeft.y)

/-- Main theorem: The area of the triangle formed by the line and the bottom of the square
    is 1/3 of the total square area -/
theorem triangle_area_is_one_third_of_square (s : Square)
  (p1 p2 : Point)
  (h1 : s.bottomLeft = ⟨2, 1⟩)
  (h2 : s.topRight = ⟨5, 4⟩)
  (h3 : p1 = ⟨2, 3⟩)
  (h4 : p2 = ⟨5, 1⟩) :
  triangleArea p1 p2 s.bottomLeft / squareArea s = 1/3 := by
  sorry


end triangle_area_is_one_third_of_square_l1820_182093


namespace largest_prime_factor_of_3913_l1820_182066

theorem largest_prime_factor_of_3913 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3913 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3913 → q ≤ p :=
by sorry

end largest_prime_factor_of_3913_l1820_182066


namespace five_digit_palindromes_count_l1820_182070

/-- A function that returns the number of positive five-digit palindromic integers -/
def count_five_digit_palindromes : ℕ :=
  9 * 10 * 10

/-- Theorem stating that the number of positive five-digit palindromic integers is 900 -/
theorem five_digit_palindromes_count :
  count_five_digit_palindromes = 900 := by
  sorry

#eval count_five_digit_palindromes

end five_digit_palindromes_count_l1820_182070


namespace instantaneous_velocity_at_3_l1820_182028

/-- The motion equation of an object -/
def s (t : ℝ) : ℝ := t^3 + t^2 - 1

/-- The velocity function derived from the motion equation -/
def v (t : ℝ) : ℝ := 3 * t^2 + 2 * t

theorem instantaneous_velocity_at_3 : v 3 = 33 := by
  sorry

end instantaneous_velocity_at_3_l1820_182028


namespace intersection_line_equation_l1820_182087

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 13 = 0

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (P Q : ℝ × ℝ),
  circle1 P.1 P.2 ∧ circle1 Q.1 Q.2 ∧
  circle2 P.1 P.2 ∧ circle2 Q.1 Q.2 ∧
  P ≠ Q →
  line P.1 P.2 ∧ line Q.1 Q.2 :=
by sorry

end intersection_line_equation_l1820_182087


namespace lcm_factor_problem_l1820_182081

theorem lcm_factor_problem (A B : ℕ+) (h : Nat.gcd A B = 25) (hA : A = 350) 
  (hlcm : ∃ x : ℕ+, Nat.lcm A B = 25 * 13 * x) : 
  ∃ x : ℕ+, Nat.lcm A B = 25 * 13 * x ∧ x = 14 := by
  sorry

end lcm_factor_problem_l1820_182081


namespace lebesgue_decomposition_l1820_182050

variable (E : Type) [MeasurableSpace E]
variable (μ ν : Measure E)

/-- Lebesgue decomposition theorem -/
theorem lebesgue_decomposition :
  ∃ (f : E → ℝ) (D : Set E),
    MeasurableSet D ∧
    (∀ x, 0 ≤ f x) ∧
    Measurable f ∧
    ν D = 0 ∧
    (∀ (B : Set E), MeasurableSet B →
      μ B = ∫ x in B, f x ∂ν + μ (B ∩ D)) ∧
    (∀ (g : E → ℝ) (C : Set E),
      MeasurableSet C →
      (∀ x, 0 ≤ g x) →
      Measurable g →
      ν C = 0 →
      (∀ (B : Set E), MeasurableSet B →
        μ B = ∫ x in B, g x ∂ν + μ (B ∩ C)) →
      (μ (D Δ C) = 0 ∧ ν {x | f x ≠ g x} = 0)) :=
sorry

end lebesgue_decomposition_l1820_182050


namespace mary_candy_count_l1820_182075

theorem mary_candy_count (megan_candy : ℕ) (mary_multiplier : ℕ) (mary_additional : ℕ)
  (h1 : megan_candy = 5)
  (h2 : mary_multiplier = 3)
  (h3 : mary_additional = 10) :
  megan_candy * mary_multiplier + mary_additional = 25 :=
by sorry

end mary_candy_count_l1820_182075


namespace perpendicular_lines_a_values_l1820_182037

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 * a = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (2 * a - 1) * x + a * y = 0

-- Define perpendicularity
def perpendicular (a : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, line1 a x₁ y₁ ∧ line2 a x₂ y₂ →
    (x₁ - x₂) * (y₁ - y₂) = 0

-- Theorem statement
theorem perpendicular_lines_a_values :
  ∀ a : ℝ, perpendicular a → a = 0 ∨ a = 1 := by
  sorry

end perpendicular_lines_a_values_l1820_182037


namespace magnitude_BC_l1820_182022

/-- Given two points A and C in ℝ², and a vector AB, prove that the magnitude of BC is √29 -/
theorem magnitude_BC (A C B : ℝ × ℝ) (h1 : A = (2, -1)) (h2 : C = (0, 2)) 
  (h3 : B.1 - A.1 = 3 ∧ B.2 - A.2 = 5) : 
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = Real.sqrt 29 := by
  sorry

#check magnitude_BC

end magnitude_BC_l1820_182022


namespace fill_time_with_leak_l1820_182080

/-- Time taken to fill a tank with two pipes and a leak -/
theorem fill_time_with_leak (pipe1_time pipe2_time : ℝ) (leak_fraction : ℝ) : 
  pipe1_time = 20 →
  pipe2_time = 30 →
  leak_fraction = 1/3 →
  (1 / ((1 / pipe1_time + 1 / pipe2_time) * (1 - leak_fraction))) = 18 :=
by sorry

end fill_time_with_leak_l1820_182080


namespace sufficient_not_necessary_l1820_182067

/-- The quadratic function f(x) with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 9

/-- Predicate indicating if f has a root for a given m -/
def has_root (m : ℝ) : Prop := ∃ x, f m x = 0

theorem sufficient_not_necessary :
  (∀ m, m > 7 → has_root m) ∧ 
  (∃ m, has_root m ∧ m ≤ 7) := by sorry

end sufficient_not_necessary_l1820_182067


namespace sun_division_l1820_182041

theorem sun_division (x y z : ℚ) : 
  (∀ r : ℚ, y = (45/100) * r → z = (30/100) * r) →  -- For each rupee x gets, y gets 45 paisa and z gets 30 paisa
  y = 54 →                                          -- Y's share is Rs. 54
  x + y + z = 210                                   -- Total amount is Rs. 210
  := by sorry

end sun_division_l1820_182041


namespace elevator_weight_average_l1820_182023

theorem elevator_weight_average (initial_people : Nat) (initial_avg_weight : ℝ) (new_person_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 156 →
  new_person_weight = 121 →
  let total_weight := initial_people * initial_avg_weight + new_person_weight
  let new_people_count := initial_people + 1
  let new_avg_weight := total_weight / new_people_count
  new_avg_weight = 151 := by
sorry

end elevator_weight_average_l1820_182023


namespace polynomial_simplification_l1820_182072

theorem polynomial_simplification (x : ℝ) :
  (3*x^2 - 2*x + 5)*(x - 2) - (x - 2)*(2*x^2 + 5*x - 8) + (2*x - 3)*(x - 2)*(x + 4) = 3*x^3 - 8*x^2 + 5*x - 2 := by
  sorry

end polynomial_simplification_l1820_182072


namespace framed_painting_ratio_l1820_182068

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_width : ℝ
  painting_height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting -/
def framed_dimensions (fp : FramedPainting) : (ℝ × ℝ) :=
  (fp.painting_width + 2 * fp.side_frame_width,
   fp.painting_height + 6 * fp.side_frame_width)

/-- Calculates the area of the framed painting -/
def framed_area (fp : FramedPainting) : ℝ :=
  let (w, h) := framed_dimensions fp
  w * h

/-- Calculates the area of the painting -/
def painting_area (fp : FramedPainting) : ℝ :=
  fp.painting_width * fp.painting_height

/-- Theorem stating the ratio of smaller to larger dimension of the framed painting -/
theorem framed_painting_ratio (fp : FramedPainting) 
  (h1 : fp.painting_width = 20)
  (h2 : fp.painting_height = 30)
  (h3 : framed_area fp = 3 * painting_area fp) :
  let (w, h) := framed_dimensions fp
  w / h = 1 / 2 := by
  sorry

end framed_painting_ratio_l1820_182068


namespace loan_interest_time_l1820_182083

/-- Given two loans and their interest rates, calculate the time needed to reach a specific total interest. -/
theorem loan_interest_time (loan1 loan2 rate1 rate2 total_interest : ℚ) : 
  loan1 = 1000 →
  loan2 = 1400 →
  rate1 = 3 / 100 →
  rate2 = 5 / 100 →
  total_interest = 350 →
  ∃ (time : ℚ), time * (loan1 * rate1 + loan2 * rate2) = total_interest ∧ time = 7 / 2 := by
  sorry

#check loan_interest_time

end loan_interest_time_l1820_182083


namespace largest_angle_in_special_triangle_l1820_182021

theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    b = (3/2) * a →
    c = 2 * a →
    a + b + c = 180 →
    max a (max b c) = 80 :=
by sorry

end largest_angle_in_special_triangle_l1820_182021


namespace remainder_55_57_mod_7_l1820_182079

theorem remainder_55_57_mod_7 : (55 * 57) % 7 = 6 := by
  sorry

end remainder_55_57_mod_7_l1820_182079


namespace spade_operation_result_l1820_182005

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_operation_result : spade 3 (spade 5 (spade 8 12)) = 2 := by sorry

end spade_operation_result_l1820_182005


namespace extra_time_at_reduced_speed_l1820_182014

theorem extra_time_at_reduced_speed 
  (usual_time : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : usual_time = 72.00000000000001)
  (h2 : speed_ratio = 0.75) : 
  (usual_time / speed_ratio) - usual_time = 24 := by
sorry

end extra_time_at_reduced_speed_l1820_182014


namespace calculate_expression_l1820_182001

theorem calculate_expression : (2200 - 2090)^2 / (144 + 25) = 64 := by
  sorry

end calculate_expression_l1820_182001


namespace prime_power_composite_and_divisor_l1820_182030

theorem prime_power_composite_and_divisor (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  let q := (4^p - 1) / 3
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ q = a * b) ∧ (q ∣ 2^(q - 1) - 1) := by
  sorry

end prime_power_composite_and_divisor_l1820_182030


namespace quadratic_equation_properties_l1820_182008

/-- Given a quadratic equation x^2 + (2k-1)x + k^2 - k = 0 where x = 2 is one of the roots,
    prove that it has two distinct real roots and the value of -2k^2 - 6k - 5 is -1 -/
theorem quadratic_equation_properties (k : ℝ) :
  (∃ x : ℝ, x^2 + (2*k - 1)*x + k^2 - k = 0) →
  (2^2 + (2*k - 1)*2 + k^2 - k = 0) →
  (∃ x y : ℝ, x ≠ y ∧ x^2 + (2*k - 1)*x + k^2 - k = 0 ∧ y^2 + (2*k - 1)*y + k^2 - k = 0) ∧
  (-2*k^2 - 6*k - 5 = -1) := by
sorry

end quadratic_equation_properties_l1820_182008


namespace january_savings_l1820_182029

def savings_challenge (initial_savings : ℚ) : ℕ → ℚ
  | 0 => initial_savings
  | n + 1 => 2 * savings_challenge initial_savings n

theorem january_savings (may_savings : ℚ) :
  may_savings = 160 →
  ∃ (initial_savings : ℚ),
    savings_challenge initial_savings 4 = may_savings ∧
    initial_savings = 10 :=
by sorry

end january_savings_l1820_182029


namespace square_of_sum_eleven_five_l1820_182038

theorem square_of_sum_eleven_five : 11^2 + 2*(11*5) + 5^2 = 256 := by
  sorry

end square_of_sum_eleven_five_l1820_182038


namespace nonagon_diagonal_intersections_l1820_182017

/-- The number of vertices in a regular nonagon -/
def n : ℕ := 9

/-- The number of distinct intersection points of diagonals in the interior of a regular nonagon -/
def intersection_points (n : ℕ) : ℕ := Nat.choose n 4

/-- Theorem: The number of distinct intersection points of diagonals in the interior of a regular nonagon is 126 -/
theorem nonagon_diagonal_intersections :
  intersection_points n = 126 := by sorry

end nonagon_diagonal_intersections_l1820_182017


namespace expression_not_constant_l1820_182044

theorem expression_not_constant : 
  ¬∀ (x y : ℝ), x ≠ -1 → x ≠ 3 → y ≠ -1 → y ≠ 3 → 
  (3*x^2 + 4*x - 5) / ((x+1)*(x-3)) - (8 + x) / ((x+1)*(x-3)) = 
  (3*y^2 + 4*y - 5) / ((y+1)*(y-3)) - (8 + y) / ((y+1)*(y-3)) :=
by sorry

end expression_not_constant_l1820_182044


namespace evaluate_expression_l1820_182053

theorem evaluate_expression : (3 + 2) * (3^2 + 2^2) * (3^4 + 2^4) = 6255 := by
  sorry

end evaluate_expression_l1820_182053


namespace solve_equation_l1820_182010

theorem solve_equation : ∃ x : ℝ, 3 * x = (36 - x) + 16 ∧ x = 13 := by
  sorry

end solve_equation_l1820_182010


namespace quadratic_integer_root_l1820_182098

theorem quadratic_integer_root (b : ℤ) : 
  (∃ x : ℤ, x^2 + 4*x + b = 0) ↔ (b = -12 ∨ b = -5 ∨ b = 3 ∨ b = 4) := by
  sorry

end quadratic_integer_root_l1820_182098


namespace average_sour_candies_is_correct_l1820_182059

/-- The number of people in the group -/
def num_people : ℕ := 4

/-- The number of sour candies Wendy's brother has -/
def brother_sour_candies : ℕ := 4

/-- The number of sour candies Wendy has -/
def wendy_sour_candies : ℕ := 5

/-- The number of sour candies their cousin has -/
def cousin_sour_candies : ℕ := 1

/-- The number of sour candies their uncle has -/
def uncle_sour_candies : ℕ := 3

/-- The total number of sour candies -/
def total_sour_candies : ℕ := brother_sour_candies + wendy_sour_candies + cousin_sour_candies + uncle_sour_candies

/-- The average number of sour candies per person -/
def average_sour_candies : ℚ := total_sour_candies / num_people

theorem average_sour_candies_is_correct : average_sour_candies = 13/4 := by
  sorry

end average_sour_candies_is_correct_l1820_182059


namespace union_necessary_not_sufficient_for_complement_l1820_182096

universe u

variable {U : Type u}
variable (A B : Set U)

theorem union_necessary_not_sufficient_for_complement :
  (∀ (A B : Set U), B = Aᶜ → A ∪ B = Set.univ) ∧
  (∃ (A B : Set U), A ∪ B = Set.univ ∧ B ≠ Aᶜ) :=
sorry

end union_necessary_not_sufficient_for_complement_l1820_182096


namespace peters_cucumbers_l1820_182076

/-- The problem of Peter's grocery shopping -/
theorem peters_cucumbers 
  (initial_amount : ℕ)
  (potato_kilos potato_price : ℕ)
  (tomato_kilos tomato_price : ℕ)
  (banana_kilos banana_price : ℕ)
  (cucumber_price : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 500)
  (h2 : potato_kilos = 6)
  (h3 : potato_price = 2)
  (h4 : tomato_kilos = 9)
  (h5 : tomato_price = 3)
  (h6 : banana_kilos = 3)
  (h7 : banana_price = 5)
  (h8 : cucumber_price = 4)
  (h9 : remaining_amount = 426)
  : ∃ (cucumber_kilos : ℕ), 
    initial_amount - 
    (potato_kilos * potato_price + 
     tomato_kilos * tomato_price + 
     banana_kilos * banana_price + 
     cucumber_kilos * cucumber_price) = remaining_amount ∧ 
    cucumber_kilos = 5 := by
  sorry

end peters_cucumbers_l1820_182076


namespace parabola_perpendicular_range_l1820_182035

/-- Given a parabola y² = x + 4 and points A(0,2), B(m² - 4, m), and C(x₀² - 4, x₀) where B and C are on the parabola and AB ⊥ BC, 
    the y-coordinate of C (x₀) satisfies: x₀ ≤ 2 - 2√2 or x₀ ≥ 2 + 2√2 -/
theorem parabola_perpendicular_range (m x₀ : ℝ) : 
  (m ^ 2 - 4 ≥ 0) →  -- B is on or above the x-axis
  (x₀ ^ 2 - 4 ≥ 0) →  -- C is on or above the x-axis
  ((m - 2) / (m ^ 2 - 4) * (x₀ - m) / (x₀ ^ 2 - m ^ 2) = -1) →  -- AB ⊥ BC
  (x₀ ≤ 2 - 2 * Real.sqrt 2 ∨ x₀ ≥ 2 + 2 * Real.sqrt 2) :=
by sorry


end parabola_perpendicular_range_l1820_182035


namespace triangle_side_length_l1820_182069

/-- Prove that in a triangle ABC with specific properties, the length of side a is 3√2 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  let f := λ x : ℝ => (Real.cos x, -1) • (Real.cos x + Real.sqrt 3 * Real.sin x, -3/2) - 2
  (f A = 1/2) →
  (2 * a = b + c) →
  (b * c / 2 = 9) →
  (a = 3 * Real.sqrt 2) := by
  sorry

end triangle_side_length_l1820_182069


namespace range_of_a_l1820_182088

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 4*x + a = 0

-- Define the set of valid a values
def valid_a : Set ℝ := {a | a < Real.exp 1 ∨ a > 4}

-- Theorem statement
theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) → a ∈ valid_a := by
  sorry

end range_of_a_l1820_182088


namespace problem_1_l1820_182064

theorem problem_1 : (-8) + (-7) - (-6) + 9 = 0 := by
  sorry

end problem_1_l1820_182064


namespace lower_variance_less_volatile_l1820_182003

/-- Represents a shooter's performance --/
structure ShooterPerformance where
  average_score : ℝ
  variance : ℝ
  num_shots : ℕ

/-- Defines volatility based on variance --/
def less_volatile (a b : ShooterPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two shooters with the same average score but different variances,
    the shooter with the lower variance has less volatile performance --/
theorem lower_variance_less_volatile (a b : ShooterPerformance) 
  (h1 : a.average_score = b.average_score)
  (h2 : a.variance ≠ b.variance)
  (h3 : a.num_shots = b.num_shots)
  : less_volatile (if a.variance < b.variance then a else b) (if a.variance > b.variance then a else b) :=
by
  sorry

end lower_variance_less_volatile_l1820_182003


namespace distance_between_polar_points_l1820_182077

/-- Given two points P and Q in polar coordinates, where the difference of their angles is π/3,
    prove that the distance between them is 8√10. -/
theorem distance_between_polar_points (α β : Real) :
  let P : Real × Real := (4, α)
  let Q : Real × Real := (12, β)
  α - β = π / 3 →
  let distance := Real.sqrt ((12 * Real.cos β - 4 * Real.cos α)^2 + (12 * Real.sin β - 4 * Real.sin α)^2)
  distance = 8 * Real.sqrt 10 := by
  sorry

end distance_between_polar_points_l1820_182077


namespace students_on_south_side_l1820_182039

theorem students_on_south_side (total : ℕ) (difference : ℕ) (south : ℕ) : 
  total = 41 → difference = 3 → south = total / 2 + difference / 2 → south = 22 := by
  sorry

end students_on_south_side_l1820_182039


namespace bayswater_volleyball_club_members_l1820_182007

theorem bayswater_volleyball_club_members : 
  let knee_pad_cost : ℕ := 6
  let jersey_cost : ℕ := knee_pad_cost + 7
  let member_cost : ℕ := 2 * (knee_pad_cost + jersey_cost)
  let total_expenditure : ℕ := 3120
  total_expenditure / member_cost = 82 :=
by sorry

end bayswater_volleyball_club_members_l1820_182007


namespace macy_running_goal_l1820_182060

/-- Calculates the remaining miles to run given a weekly goal, daily run distance, and number of days run. -/
def remaining_miles (weekly_goal : ℕ) (daily_run : ℕ) (days_run : ℕ) : ℕ :=
  weekly_goal - daily_run * days_run

/-- Proves that given a weekly goal of 24 miles and a daily run of 3 miles, the remaining distance to run after 6 days is 6 miles. -/
theorem macy_running_goal :
  remaining_miles 24 3 6 = 6 := by
  sorry

#eval remaining_miles 24 3 6

end macy_running_goal_l1820_182060


namespace simplify_expression_l1820_182013

theorem simplify_expression (x : ℝ) : 2*x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -4*x - 4 := by
  sorry

end simplify_expression_l1820_182013


namespace max_sum_of_product_60_l1820_182020

theorem max_sum_of_product_60 (a b c : ℕ) (h : a * b * c = 60) :
  a + b + c ≤ 62 := by
  sorry

end max_sum_of_product_60_l1820_182020


namespace ferry_problem_l1820_182032

/-- The ferry problem -/
theorem ferry_problem (speed_p speed_q : ℝ) (time_p : ℝ) (distance_q : ℝ) :
  speed_p = 8 →
  time_p = 3 →
  speed_q = speed_p + 4 →
  distance_q = 2 * speed_p * time_p →
  distance_q / speed_q - time_p = 1 := by
  sorry

end ferry_problem_l1820_182032


namespace modulus_of_complex_number_l1820_182043

theorem modulus_of_complex_number (z : ℂ) : z = 2 / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_complex_number_l1820_182043


namespace opposite_of_2023_l1820_182027

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by
sorry

end opposite_of_2023_l1820_182027
