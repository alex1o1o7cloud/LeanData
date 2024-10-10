import Mathlib

namespace mod_equivalence_unique_solution_l2821_282189

theorem mod_equivalence_unique_solution : 
  ∃! n : ℕ, n ≤ 6 ∧ n ≡ -4752 [ZMOD 7] := by sorry

end mod_equivalence_unique_solution_l2821_282189


namespace complex_product_pure_imaginary_l2821_282153

theorem complex_product_pure_imaginary (x : ℝ) : 
  let z₁ : ℂ := 1 - I
  let z₂ : ℂ := -1 - x * I
  (z₁ * z₂).re = 0 → x = -1 := by sorry

end complex_product_pure_imaginary_l2821_282153


namespace kendra_shirts_theorem_l2821_282160

/-- The number of shirts Kendra needs for a three-week period -/
def shirts_needed : ℕ :=
  let school_shirts := 5  -- 5 weekdays
  let club_shirts := 3    -- 3 days a week
  let saturday_shirts := 3 -- 1 for workout, 1 for art class, 1 for rest of the day
  let sunday_shirts := 3   -- 1 for church, 1 for volunteer work, 1 for rest of the day
  let weekly_shirts := school_shirts + club_shirts + saturday_shirts + sunday_shirts
  let weeks := 3
  weekly_shirts * weeks

/-- Theorem stating that Kendra needs 42 shirts for a three-week period -/
theorem kendra_shirts_theorem : shirts_needed = 42 := by
  sorry

end kendra_shirts_theorem_l2821_282160


namespace equal_expressions_l2821_282195

theorem equal_expressions (x y z : ℤ) :
  x + 2 * y * z = (x + y) * (x + 2 * z) ↔ x + y + 2 * z = 1 ∨ x = 0 :=
by sorry

end equal_expressions_l2821_282195


namespace fifteen_buses_needed_l2821_282184

/-- Given the number of students, bus capacity, and pre-reserved bus seats,
    calculate the total number of buses needed. -/
def total_buses_needed (total_students : ℕ) (bus_capacity : ℕ) (pre_reserved_seats : ℕ) : ℕ :=
  let remaining_students := total_students - pre_reserved_seats
  let new_buses := (remaining_students + bus_capacity - 1) / bus_capacity
  new_buses + 1

/-- Theorem stating that 15 buses are needed for the given conditions. -/
theorem fifteen_buses_needed :
  total_buses_needed 635 45 20 = 15 := by
  sorry

#eval total_buses_needed 635 45 20

end fifteen_buses_needed_l2821_282184


namespace hexagonal_prism_edge_sum_specific_l2821_282191

/-- Calculates the sum of lengths of all edges of a regular hexagonal prism -/
def hexagonal_prism_edge_sum (base_side_length : ℝ) (height : ℝ) : ℝ :=
  2 * (6 * base_side_length) + 6 * height

theorem hexagonal_prism_edge_sum_specific : 
  hexagonal_prism_edge_sum 6 11 = 138 := by
  sorry

end hexagonal_prism_edge_sum_specific_l2821_282191


namespace evaluate_expression_l2821_282100

theorem evaluate_expression : 6 - 8 * (9 - 2^3 + 12/3) * 5 = -194 := by
  sorry

end evaluate_expression_l2821_282100


namespace sum_of_coefficients_l2821_282198

theorem sum_of_coefficients (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 15625 := by
sorry

end sum_of_coefficients_l2821_282198


namespace total_dangerous_animals_l2821_282159

def crocodiles : ℕ := 22
def alligators : ℕ := 23
def vipers : ℕ := 5

theorem total_dangerous_animals : crocodiles + alligators + vipers = 50 := by
  sorry

end total_dangerous_animals_l2821_282159


namespace oldest_child_daily_cheese_is_two_l2821_282128

/-- The number of string cheeses Kelly's oldest child wants every day. -/
def oldest_child_daily_cheese : ℕ := 
  let days_per_week : ℕ := 5
  let total_weeks : ℕ := 4
  let cheeses_per_package : ℕ := 30
  let packages_needed : ℕ := 2
  let youngest_child_daily_cheese : ℕ := 1
  let total_days : ℕ := days_per_week * total_weeks
  let total_cheeses : ℕ := packages_needed * cheeses_per_package
  let youngest_total_cheeses : ℕ := youngest_child_daily_cheese * total_days
  let oldest_total_cheeses : ℕ := total_cheeses - youngest_total_cheeses
  oldest_total_cheeses / total_days

theorem oldest_child_daily_cheese_is_two : oldest_child_daily_cheese = 2 := by
  sorry

end oldest_child_daily_cheese_is_two_l2821_282128


namespace remainder_theorem_l2821_282125

theorem remainder_theorem (N : ℤ) : 
  (N % 779 = 47) → (N % 19 = 9) := by
sorry

end remainder_theorem_l2821_282125


namespace sufficient_not_necessary_condition_l2821_282105

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a < b ∧ b < 0)) := by
  sorry

end sufficient_not_necessary_condition_l2821_282105


namespace binary_51_l2821_282155

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Theorem: The binary representation of 51 is [true, true, false, false, true, true] -/
theorem binary_51 : toBinary 51 = [true, true, false, false, true, true] := by
  sorry

#eval toBinary 51

end binary_51_l2821_282155


namespace carol_rectangle_width_l2821_282175

/-- Given two rectangles with equal areas, where one has a length of 5 inches
    and the other has dimensions of 2 inches by 60 inches,
    prove that the width of the first rectangle is 24 inches. -/
theorem carol_rectangle_width (w : ℝ) :
  (5 * w = 2 * 60) → w = 24 :=
by sorry

end carol_rectangle_width_l2821_282175


namespace max_area_right_triangle_l2821_282166

theorem max_area_right_triangle (c : ℝ) (h : c = 8) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x^2 + y^2 = c^2 →
  (1/2) * x * y ≤ (1/2) * a * b ∧
  (1/2) * a * b = 16 := by
sorry

end max_area_right_triangle_l2821_282166


namespace nba_player_age_distribution_l2821_282120

theorem nba_player_age_distribution (total_players : ℕ) 
  (h1 : total_players = 1000)
  (h2 : (2 : ℚ) / 5 * total_players = (players_25_to_35 : ℕ))
  (h3 : (3 : ℚ) / 8 * total_players = (players_over_35 : ℕ)) :
  total_players - (players_25_to_35 + players_over_35) = 225 :=
by sorry

end nba_player_age_distribution_l2821_282120


namespace equation_solution_range_l2821_282149

theorem equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ (x + m) / (x - 2) - 3 = (x - 1) / (2 - x)) ↔ 
  (m ≥ -5 ∧ m ≠ -3) :=
sorry

end equation_solution_range_l2821_282149


namespace arithmetic_sequence_problem_l2821_282157

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 3 + a 8 = 10 →              -- given condition
  3 * a 5 + a 7 = 20 :=         -- conclusion to prove
by
  sorry

end arithmetic_sequence_problem_l2821_282157


namespace square_reassembly_l2821_282112

/-- Given two squares with side lengths a and b (where a > b), 
    they can be cut and reassembled into a single square with side length √(a² + b²) -/
theorem square_reassembly (a b : ℝ) (h : a > b) (h' : a > 0) (h'' : b > 0) :
  ∃ (new_side : ℝ), 
    new_side = Real.sqrt (a^2 + b^2) ∧ 
    new_side^2 = a^2 + b^2 :=
by sorry

end square_reassembly_l2821_282112


namespace entrance_exam_correct_answers_l2821_282192

theorem entrance_exam_correct_answers 
  (total_questions : ℕ) 
  (correct_marks : ℤ) 
  (wrong_marks : ℤ) 
  (total_score : ℤ) : 
  total_questions = 70 → 
  correct_marks = 3 → 
  wrong_marks = -1 → 
  total_score = 38 → 
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_marks + (total_questions - correct_answers) * wrong_marks = total_score ∧ 
    correct_answers = 27 := by
  sorry

end entrance_exam_correct_answers_l2821_282192


namespace ordered_numbers_count_l2821_282146

/-- Counts the numbers from 000 to 999 with digits in non-decreasing or non-increasing order -/
def count_ordered_numbers : ℕ :=
  let non_decreasing := Nat.choose 12 9
  let non_increasing := Nat.choose 12 9
  let double_counted := 10  -- Numbers with all identical digits
  non_decreasing + non_increasing - double_counted

/-- The count of numbers from 000 to 999 with digits in non-decreasing or non-increasing order is 430 -/
theorem ordered_numbers_count : count_ordered_numbers = 430 := by
  sorry

end ordered_numbers_count_l2821_282146


namespace horner_first_step_for_f_l2821_282170

def f (x : ℝ) : ℝ := 0.5 * x^6 + 4 * x^5 - x^4 + 3 * x^3 - 5 * x

def horner_first_step (a₆ a₅ : ℝ) (x : ℝ) : ℝ := a₆ * x + a₅

theorem horner_first_step_for_f :
  horner_first_step 0.5 4 3 = 5.5 :=
sorry

end horner_first_step_for_f_l2821_282170


namespace min_value_theorem_l2821_282181

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y / x = 1) :
  1 / x + x / y ≥ 4 ∧ (1 / x + x / y = 4 ↔ y = x^2) := by
  sorry

end min_value_theorem_l2821_282181


namespace sum_plus_even_count_equals_1811_l2821_282117

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_plus_even_count_equals_1811 :
  sum_of_integers 10 60 + count_even_integers 10 60 = 1811 := by
  sorry

end sum_plus_even_count_equals_1811_l2821_282117


namespace jacket_price_after_discounts_l2821_282163

theorem jacket_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 20 →
  discount1 = 0.25 →
  discount2 = 0.40 →
  original_price * (1 - discount1) * (1 - discount2) = 9 := by
sorry

end jacket_price_after_discounts_l2821_282163


namespace voting_change_l2821_282169

theorem voting_change (total_members : ℕ) 
  (h_total : total_members = 400)
  (initial_for initial_against : ℕ) 
  (h_initial_sum : initial_for + initial_against = total_members)
  (h_initial_reject : initial_against > initial_for)
  (second_for second_against : ℕ) 
  (h_second_sum : second_for + second_against = total_members)
  (h_second_pass : second_for > second_against)
  (h_margin : second_for - second_against = 3 * (initial_against - initial_for))
  (h_proportion : second_for = (10 * initial_against) / 9) :
  second_for - initial_for = 48 := by
sorry

end voting_change_l2821_282169


namespace union_of_M_and_P_l2821_282197

-- Define the sets M and P
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def P : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

-- State the theorem
theorem union_of_M_and_P : M ∪ P = {x : ℝ | -2 ≤ x ∧ x < 2} := by sorry

end union_of_M_and_P_l2821_282197


namespace quadratic_roots_not_integers_l2821_282196

/-- 
Given a quadratic polynomial p(x) = ax² + bx + c where a, b, and c are odd integers,
if the roots x₁ and x₂ exist, they cannot both be integers.
-/
theorem quadratic_roots_not_integers 
  (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c)
  (hroots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :
  ¬∃ (y₁ y₂ : ℤ), (y₁ : ℝ) = x₁ ∧ (y₂ : ℝ) = x₂ :=
sorry

end quadratic_roots_not_integers_l2821_282196


namespace square_triangle_perimeter_ratio_l2821_282101

/-- Given a square with side length s_s and an equilateral triangle with side length s_t,
    if their perimeters are equal, then the ratio of s_t to s_s is 4/3. -/
theorem square_triangle_perimeter_ratio (s_s s_t : ℝ) (h : s_s > 0) (h' : s_t > 0) :
  4 * s_s = 3 * s_t → s_t / s_s = 4 / 3 := by
  sorry


end square_triangle_perimeter_ratio_l2821_282101


namespace min_throws_correct_l2821_282156

/-- The probability of hitting the target on a single throw -/
def p : ℝ := 0.6

/-- The desired minimum probability of hitting the target at least once -/
def min_prob : ℝ := 0.9

/-- The function that calculates the probability of hitting the target at least once in n throws -/
def prob_hit_at_least_once (n : ℕ) : ℝ := 1 - (1 - p)^n

/-- The minimum number of throws needed to exceed the desired probability -/
def min_throws : ℕ := 3

theorem min_throws_correct :
  (∀ k < min_throws, prob_hit_at_least_once k ≤ min_prob) ∧
  prob_hit_at_least_once min_throws > min_prob :=
sorry

end min_throws_correct_l2821_282156


namespace cubic_equation_no_negative_roots_l2821_282194

theorem cubic_equation_no_negative_roots :
  ∀ x : ℝ, x < 0 → x^3 - 9*x^2 + 23*x - 15 ≠ 0 := by
sorry

end cubic_equation_no_negative_roots_l2821_282194


namespace square_sum_equation_solutions_l2821_282151

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The property that a number is the sum of two squares -/
def isSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ x y : ℕ, x^2 + y^2 = n

/-- The main theorem -/
theorem square_sum_equation_solutions :
  (∃ k : ℕ+, 
    (∃ a b c : ℕ+, a^2 + b^2 + c^2 = k * a * b * c) ∧ 
    (∀ n : ℕ, ∃ a_n b_n c_n : ℕ+,
      a_n^2 + b_n^2 + c_n^2 = k * a_n * b_n * c_n ∧
      isSumOfTwoSquares (a_n * b_n) ∧
      isSumOfTwoSquares (b_n * c_n) ∧
      isSumOfTwoSquares (c_n * a_n))) ↔
  (k = 1 ∨ k = 3) := by
  sorry

end square_sum_equation_solutions_l2821_282151


namespace sin_cos_equation_solution_range_l2821_282131

theorem sin_cos_equation_solution_range :
  let f : ℝ → ℝ → ℝ := λ x a => Real.sin x ^ 2 + 2 * Real.cos x + a
  ∀ a : ℝ, (∃ x : ℝ, f x a = 0) ↔ a ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

end sin_cos_equation_solution_range_l2821_282131


namespace admission_fee_problem_l2821_282183

/-- Admission fee problem -/
theorem admission_fee_problem (child_fee : ℚ) (total_people : ℕ) (total_amount : ℚ) 
  (num_children : ℕ) (num_adults : ℕ) :
  child_fee = 3/2 →
  total_people = 2200 →
  total_amount = 5050 →
  num_children = 700 →
  num_adults = 1500 →
  num_children + num_adults = total_people →
  ∃ adult_fee : ℚ, 
    adult_fee * num_adults + child_fee * num_children = total_amount ∧
    adult_fee = 8/3 :=
by sorry

end admission_fee_problem_l2821_282183


namespace plane_perpendicular_parallel_transitive_l2821_282150

/-- A structure representing a 3D space with planes and perpendicularity/parallelism relations -/
structure Space3D where
  Plane : Type
  perpendicular : Plane → Plane → Prop
  parallel : Plane → Plane → Prop

/-- The main theorem to be proved -/
theorem plane_perpendicular_parallel_transitive 
  (S : Space3D) (α β γ : S.Plane) : 
  S.perpendicular α β → S.parallel α γ → S.perpendicular β γ := by
  sorry

/-- Helper lemma: If two planes are parallel, they are not perpendicular -/
lemma parallel_not_perpendicular 
  (S : Space3D) (p q : S.Plane) :
  S.parallel p q → ¬S.perpendicular p q := by
  sorry

/-- Helper lemma: Perpendicularity is symmetric -/
lemma perpendicular_symmetric 
  (S : Space3D) (p q : S.Plane) :
  S.perpendicular p q → S.perpendicular q p := by
  sorry

/-- Helper lemma: Parallelism is symmetric -/
lemma parallel_symmetric 
  (S : Space3D) (p q : S.Plane) :
  S.parallel p q → S.parallel q p := by
  sorry

end plane_perpendicular_parallel_transitive_l2821_282150


namespace race_start_distances_l2821_282143

-- Define the start distances
def start_A_B : ℝ := 50
def start_B_C : ℝ := 157.89473684210532

-- Theorem statement
theorem race_start_distances :
  let start_A_C := start_A_B + start_B_C
  start_A_C = 207.89473684210532 := by sorry

end race_start_distances_l2821_282143


namespace linear_equation_solution_l2821_282172

theorem linear_equation_solution (x y a : ℝ) : 
  x = 1 → y = 2 → x + a = 3 * y - 2 → a = 3 := by sorry

end linear_equation_solution_l2821_282172


namespace power_sum_equality_l2821_282187

theorem power_sum_equality : (3^2)^2 + (2^3)^3 = 593 := by
  sorry

end power_sum_equality_l2821_282187


namespace candy_cost_proof_l2821_282199

def candy_problem (num_packs : ℕ) (total_paid : ℕ) (change : ℕ) : Prop :=
  let total_cost : ℕ := total_paid - change
  let cost_per_pack : ℕ := total_cost / num_packs
  cost_per_pack = 3

theorem candy_cost_proof :
  candy_problem 3 20 11 := by
  sorry

end candy_cost_proof_l2821_282199


namespace solve_equation_l2821_282158

theorem solve_equation : 
  let x := 70 / (8 - 3/4)
  x = 280/29 := by sorry

end solve_equation_l2821_282158


namespace jim_caught_two_fish_l2821_282126

def fish_problem (ben judy billy susie jim : ℕ) : Prop :=
  ben = 4 ∧
  judy = 1 ∧
  billy = 3 ∧
  susie = 5 ∧
  ∃ (thrown_back : ℕ), thrown_back = 3 ∧
  ∃ (total_filets : ℕ), total_filets = 24 ∧
  (ben + judy + billy + susie + jim - thrown_back) * 2 = total_filets

theorem jim_caught_two_fish :
  ∀ ben judy billy susie jim : ℕ,
  fish_problem ben judy billy susie jim →
  jim = 2 :=
by sorry

end jim_caught_two_fish_l2821_282126


namespace sum_of_abc_l2821_282133

theorem sum_of_abc (a b c : ℝ) 
  (h1 : a^2 - 2*b = -2) 
  (h2 : b^2 + 6*c = 7) 
  (h3 : c^2 - 8*a = -31) : 
  a + b + c = 2 := by
sorry

end sum_of_abc_l2821_282133


namespace hyperbola_iff_m_negative_l2821_282118

/-- A conic section in the xy-plane -/
structure ConicSection where
  equation : ℝ → ℝ → Prop

/-- A hyperbola in the xy-plane -/
structure Hyperbola extends ConicSection

/-- The specific conic section given by the equation x^2 + my^2 = 1 -/
def specific_conic (m : ℝ) : ConicSection where
  equation := fun x y => x^2 + m*y^2 = 1

theorem hyperbola_iff_m_negative (m : ℝ) :
  ∃ (h : Hyperbola), h.equation = (specific_conic m).equation ↔ m < 0 :=
sorry

end hyperbola_iff_m_negative_l2821_282118


namespace square_side_length_l2821_282185

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 361 → side * side = area → side = 19 := by
  sorry

end square_side_length_l2821_282185


namespace amithab_january_expenditure_l2821_282171

def january_expenditure (avg_jan_jun avg_feb_jul july_expenditure : ℝ) : ℝ :=
  6 * avg_feb_jul - 6 * avg_jan_jun + july_expenditure

theorem amithab_january_expenditure :
  january_expenditure 4200 4250 1500 = 1200 := by
  sorry

end amithab_january_expenditure_l2821_282171


namespace inequality_implication_l2821_282180

theorem inequality_implication (x y z : ℝ) (h : x^2 + x*y + x*z < 0) : y^2 > 4*x*z := by
  sorry

end inequality_implication_l2821_282180


namespace set_equality_implies_sum_l2821_282121

def A (x y : ℝ) : Set ℝ := {x, y / x, 1}
def B (x y : ℝ) : Set ℝ := {x^2, x + y, 0}

theorem set_equality_implies_sum (x y : ℝ) (h : A x y = B x y) : x^2023 + y^2024 = -1 := by
  sorry

end set_equality_implies_sum_l2821_282121


namespace polygon_division_theorem_l2821_282108

/-- A polygon is a closed planar figure with straight sides -/
structure Polygon where
  sides : ℕ
  is_closed : Bool
  is_planar : Bool

/-- Represents a division of a polygon into shapes -/
structure PolygonDivision (P : Polygon) (n : ℕ) (shape : Type) where
  num_divisions : ℕ
  is_valid : Bool

/-- Given a polygon that can be divided into 100 rectangles but not 99,
    it cannot be divided into 100 triangles -/
theorem polygon_division_theorem (P : Polygon) 
  (h1 : ∃ (d : PolygonDivision P 100 Rectangle), d.is_valid)
  (h2 : ¬ ∃ (d : PolygonDivision P 99 Rectangle), d.is_valid) :
  ¬ ∃ (d : PolygonDivision P 100 Triangle), d.is_valid :=
by sorry

end polygon_division_theorem_l2821_282108


namespace area_of_region_R_l2821_282109

/-- Represents a rhombus ABCD -/
structure Rhombus where
  sideLength : ℝ
  angleB : ℝ

/-- Represents the region R inside the rhombus -/
def RegionR (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of region R in the rhombus -/
noncomputable def areaR (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of region R in the specific rhombus -/
theorem area_of_region_R : 
  let r : Rhombus := { sideLength := 3, angleB := 150 * π / 180 }
  ∃ ε > 0, |areaR r - 0.873| < ε :=
sorry

end area_of_region_R_l2821_282109


namespace min_quotient_base12_number_l2821_282142

/-- Represents a digit in base 12, ranging from 1 to 10 (in base 10) -/
def Digit12 := {d : ℕ // 1 ≤ d ∧ d ≤ 10}

/-- Converts a base 12 number to base 10 -/
def toBase10 (a b c : Digit12) : ℕ :=
  144 * a.val + 12 * b.val + c.val

/-- Calculates the sum of digits in base 10 -/
def digitSum (a b c : Digit12) : ℕ :=
  a.val + b.val + c.val

/-- The main theorem stating the minimum quotient -/
theorem min_quotient_base12_number :
  ∀ a b c : Digit12,
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (toBase10 a b c : ℚ) / (digitSum a b c) ≥ 24.5 :=
sorry

end min_quotient_base12_number_l2821_282142


namespace unique_amazing_rectangle_l2821_282127

/-- An amazing rectangle is a rectangle where the area is equal to three times its perimeter,
    one side is double the other, and both sides are positive integers. -/
structure AmazingRectangle where
  width : ℕ+
  length : ℕ+
  is_double : length = 2 * width
  is_amazing : width * length = 3 * (2 * (width + length))

/-- Theorem stating that there exists only one amazing rectangle and its area is 162. -/
theorem unique_amazing_rectangle :
  (∃! r : AmazingRectangle, True) ∧
  (∀ r : AmazingRectangle, r.width * r.length = 162) := by
  sorry


end unique_amazing_rectangle_l2821_282127


namespace roots_product_zero_l2821_282162

theorem roots_product_zero (a b c d : ℝ) : 
  (a^2 + 57*a + 1 = 0) →
  (b^2 + 57*b + 1 = 0) →
  (c^2 - 57*c + 1 = 0) →
  (d^2 - 57*d + 1 = 0) →
  (a + c) * (b + c) * (a - d) * (b - d) = 0 := by
sorry

end roots_product_zero_l2821_282162


namespace min_z_value_l2821_282119

theorem min_z_value (x y z : ℤ) (sum_eq : x + y + z = 100) (ineq : x < y ∧ y < 2*z) : 
  ∀ w : ℤ, (∃ a b : ℤ, a + b + w = 100 ∧ a < b ∧ b < 2*w) → w ≥ 21 := by
  sorry

#check min_z_value

end min_z_value_l2821_282119


namespace no_real_roots_l2821_282124

theorem no_real_roots : ¬∃ (x : ℝ), x^2 - 4*x + 8 = 0 := by
  sorry

end no_real_roots_l2821_282124


namespace greatest_divisor_with_remainders_l2821_282182

theorem greatest_divisor_with_remainders : 
  Nat.gcd (450 - 60) (Nat.gcd (330 - 15) (Nat.gcd (675 - 45) (725 - 25))) = 5 := by
  sorry

end greatest_divisor_with_remainders_l2821_282182


namespace factory_produces_160_crayons_in_4_hours_l2821_282145

/-- Represents a crayon factory with given specifications -/
structure CrayonFactory where
  num_colors : ℕ
  crayons_per_color_per_box : ℕ
  boxes_per_hour : ℕ

/-- Calculates the total number of crayons produced in a given number of hours -/
def total_crayons_produced (factory : CrayonFactory) (hours : ℕ) : ℕ :=
  factory.num_colors * factory.crayons_per_color_per_box * factory.boxes_per_hour * hours

/-- Theorem stating that a factory with given specifications produces 160 crayons in 4 hours -/
theorem factory_produces_160_crayons_in_4_hours 
  (factory : CrayonFactory) 
  (h1 : factory.num_colors = 4) 
  (h2 : factory.crayons_per_color_per_box = 2) 
  (h3 : factory.boxes_per_hour = 5) : 
  total_crayons_produced factory 4 = 160 := by
  sorry

end factory_produces_160_crayons_in_4_hours_l2821_282145


namespace select_questions_theorem_l2821_282107

/-- The number of ways to select 3 questions from a set of questions with the given conditions -/
def select_questions (multiple_choice : ℕ) (fill_in_blank : ℕ) (open_ended : ℕ) : ℕ :=
  let total_questions := multiple_choice + fill_in_blank + open_ended
  let one_each := Nat.choose multiple_choice 1 * Nat.choose fill_in_blank 1 * Nat.choose open_ended 1
  let two_multiple_one_open := Nat.choose multiple_choice 2 * Nat.choose open_ended 1
  let one_multiple_two_open := Nat.choose multiple_choice 1 * Nat.choose open_ended 2
  one_each + two_multiple_one_open + one_multiple_two_open

theorem select_questions_theorem :
  select_questions 12 4 6 = 864 := by
  sorry

end select_questions_theorem_l2821_282107


namespace triangle_inequality_l2821_282103

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  let s := (a + b + c) / 2
  (a * b) / (s - c) + (b * c) / (s - a) + (c * a) / (s - b) ≥ 4 * s := by
  sorry

end triangle_inequality_l2821_282103


namespace specific_classroom_seats_l2821_282152

/-- Represents a tiered classroom with increasing seats per row -/
structure TieredClassroom where
  rows : ℕ
  firstRowSeats : ℕ
  seatIncrease : ℕ

/-- Calculates the number of seats in the nth row -/
def seatsInRow (c : TieredClassroom) (n : ℕ) : ℕ :=
  c.firstRowSeats + (n - 1) * c.seatIncrease

/-- Calculates the total number of seats in the classroom -/
def totalSeats (c : TieredClassroom) : ℕ :=
  (c.firstRowSeats + seatsInRow c c.rows) * c.rows / 2

/-- Theorem stating the total number of seats in the specific classroom configuration -/
theorem specific_classroom_seats :
  let c : TieredClassroom := { rows := 22, firstRowSeats := 22, seatIncrease := 2 }
  totalSeats c = 946 := by sorry

end specific_classroom_seats_l2821_282152


namespace garden_area_l2821_282165

/-- A rectangular garden with specific walking measurements -/
structure Garden where
  length : ℝ
  width : ℝ
  length_walk : length * 30 = 1200
  perimeter_walk : (2 * length + 2 * width) * 12 = 1200

/-- The area of the garden is 400 square meters -/
theorem garden_area (g : Garden) : g.length * g.width = 400 := by
  sorry

end garden_area_l2821_282165


namespace geometric_sequence_fifth_term_l2821_282190

/-- A geometric sequence with specified terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_2 : a 2 = 2)
  (h_3 : a 3 = -4) :
  a 5 = -16 := by
sorry

end geometric_sequence_fifth_term_l2821_282190


namespace investment_interest_proof_l2821_282137

/-- Calculates the simple interest earned on an investment. -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_interest_proof (total_investment : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (investment1 : ℝ) (time : ℝ) 
  (h1 : total_investment = 15000)
  (h2 : rate1 = 0.06)
  (h3 : rate2 = 0.075)
  (h4 : investment1 = 8200)
  (h5 : time = 1)
  (h6 : investment1 ≤ total_investment) :
  simple_interest investment1 rate1 time + 
  simple_interest (total_investment - investment1) rate2 time = 1002 := by
  sorry

#check investment_interest_proof

end investment_interest_proof_l2821_282137


namespace hyperbola_eccentricity_l2821_282104

/-- The eccentricity of a hyperbola with the given conditions is √5 -/
theorem hyperbola_eccentricity (a b c : ℝ) (P Q F : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  F = (c, 0) →
  (P.1^2 / a^2) - (P.2^2 / b^2) = 1 →
  (Q.1 - c/3)^2 + Q.2^2 = b^2/9 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4 * ((Q.1 - F.1)^2 + (Q.2 - F.2)^2) →
  (P.1 - F.1) * (Q.1 - c/3) + (P.2 - F.2) * Q.2 = 0 →
  c^2 = a^2 + b^2 →
  c / a = Real.sqrt 5 := by
sorry

end hyperbola_eccentricity_l2821_282104


namespace rats_meet_on_day_10_l2821_282140

/-- The thickness of the wall in feet -/
def wall_thickness : ℕ := 1000

/-- The initial drilling speed of both rats in feet per day -/
def initial_speed : ℕ := 1

/-- The function representing the total distance drilled by both rats after n days -/
def total_distance (n : ℕ) : ℚ :=
  (2^n - 1) + 2 * (1 - (1/2)^n)

/-- The theorem stating that the rats meet on the 10th day -/
theorem rats_meet_on_day_10 :
  total_distance 9 < wall_thickness ∧ total_distance 10 ≥ wall_thickness :=
sorry

end rats_meet_on_day_10_l2821_282140


namespace exclusive_or_implication_l2821_282135

theorem exclusive_or_implication :
  let statement1 := ¬p ∧ ¬q
  let statement2 := ¬p ∧ q
  let statement3 := p ∧ ¬q
  let statement4 := p ∧ q
  let exclusive_condition := ¬(p ∧ q)
  (statement1 → exclusive_condition) ∧
  (statement2 → exclusive_condition) ∧
  (statement3 → exclusive_condition) ∧
  ¬(statement4 → exclusive_condition) := by
  sorry

end exclusive_or_implication_l2821_282135


namespace machining_defective_rate_l2821_282136

theorem machining_defective_rate 
  (p1 p2 p3 : ℚ) 
  (h1 : p1 = 1 / 70)
  (h2 : p2 = 1 / 69)
  (h3 : p3 = 1 / 68)
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - p1) * (1 - p2) * (1 - p3) = 3 / 70 :=
sorry

end machining_defective_rate_l2821_282136


namespace swimmers_arrangement_count_l2821_282167

/-- The number of swimmers -/
def num_swimmers : ℕ := 6

/-- The number of arrangements when A is leftmost -/
def arrangements_A_leftmost : ℕ := (num_swimmers - 1) * (Nat.factorial (num_swimmers - 2))

/-- The number of arrangements when B is leftmost -/
def arrangements_B_leftmost : ℕ := (num_swimmers - 2) * (Nat.factorial (num_swimmers - 2))

/-- The total number of arrangements -/
def total_arrangements : ℕ := arrangements_A_leftmost + arrangements_B_leftmost

theorem swimmers_arrangement_count :
  total_arrangements = 216 :=
sorry

end swimmers_arrangement_count_l2821_282167


namespace pencils_used_l2821_282188

theorem pencils_used (initial : ℕ) (current : ℕ) (h1 : initial = 94) (h2 : current = 91) :
  initial - current = 3 := by
  sorry

end pencils_used_l2821_282188


namespace f_second_derivative_positive_l2821_282102

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

def f_domain : Set ℝ := {x : ℝ | x > 0}

noncomputable def f'' (x : ℝ) : ℝ := 2 + 4 / x^2

theorem f_second_derivative_positive :
  {x ∈ f_domain | f'' x > 0} = f_domain :=
sorry

end f_second_derivative_positive_l2821_282102


namespace clothing_price_proof_l2821_282164

/-- Given the following conditions:
    - Total spent on 7 pieces of clothing is $610
    - One piece costs $49
    - Another piece costs $81
    - The remaining pieces all cost the same
    - The price of the remaining pieces is a multiple of 5
    Prove that each of the remaining pieces costs $96 -/
theorem clothing_price_proof (total_spent : ℕ) (total_pieces : ℕ) (price1 : ℕ) (price2 : ℕ) (price_other : ℕ) :
  total_spent = 610 →
  total_pieces = 7 →
  price1 = 49 →
  price2 = 81 →
  (total_spent - price1 - price2) % (total_pieces - 2) = 0 →
  price_other % 5 = 0 →
  price_other * (total_pieces - 2) + price1 + price2 = total_spent →
  price_other = 96 := by
  sorry

#eval 96 * 5 + 49 + 81  -- Should output 610

end clothing_price_proof_l2821_282164


namespace calculation_proof_l2821_282113

theorem calculation_proof : (1/2)⁻¹ + (Real.sqrt 2)^2 - 4 * |(-(1/2))| = 2 := by
  sorry

end calculation_proof_l2821_282113


namespace sin_equality_solution_l2821_282141

theorem sin_equality_solution (x : Real) (h1 : x ∈ Set.Icc 0 (2 * Real.pi)) 
  (h2 : Real.sin x = Real.sin (Real.arcsin (2/3) - Real.arcsin (-1/3))) : 
  x = Real.arcsin ((4 * Real.sqrt 2 + Real.sqrt 5) / 9) ∨ 
  x = Real.pi - Real.arcsin ((4 * Real.sqrt 2 + Real.sqrt 5) / 9) := by
  sorry

end sin_equality_solution_l2821_282141


namespace determine_absolute_b_l2821_282168

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the polynomial g(x)
def g (a b c : ℤ) (x : ℂ) : ℂ :=
  a * x^5 + b * x^4 + c * x^3 + c * x^2 + b * x + a

-- State the theorem
theorem determine_absolute_b (a b c : ℤ) : 
  g a b c (3 + i) = 0 →
  Int.gcd a b = 1 ∧ Int.gcd a c = 1 ∧ Int.gcd b c = 1 →
  |b| = 66 := by
  sorry

end determine_absolute_b_l2821_282168


namespace square_less_than_power_of_three_l2821_282114

theorem square_less_than_power_of_three (n : ℕ) (h : n ≥ 3) : (n + 1)^2 < 3^n := by
  sorry

end square_less_than_power_of_three_l2821_282114


namespace simplify_sqrt_expression_l2821_282123

theorem simplify_sqrt_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (4 + ((x^6 - 3*x^3 + 2) / (3*x^3))^2) = (x^6 - 3*x^3 + 2) / (3*x^3) := by
  sorry

end simplify_sqrt_expression_l2821_282123


namespace least_common_multiple_first_ten_l2821_282106

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧
  (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ≤ 10 → k > 0 → m % k = 0) → m ≥ 2520) :=
by sorry

end least_common_multiple_first_ten_l2821_282106


namespace least_product_of_three_primes_over_50_l2821_282144

theorem least_product_of_three_primes_over_50 :
  ∃ (p q r : Nat),
    Prime p ∧ Prime q ∧ Prime r ∧
    p > 50 ∧ q > 50 ∧ r > 50 ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    p * q * r = 190847 ∧
    ∀ (a b c : Nat),
      Prime a → Prime b → Prime c →
      a > 50 → b > 50 → c > 50 →
      a ≠ b → a ≠ c → b ≠ c →
      a * b * c ≥ 190847 :=
by
  sorry

end least_product_of_three_primes_over_50_l2821_282144


namespace ratio_composition_l2821_282134

theorem ratio_composition (a b c : ℚ) 
  (h1 : a / b = 2 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 2 / 15 := by
sorry

end ratio_composition_l2821_282134


namespace january_oil_bill_l2821_282139

theorem january_oil_bill (feb_bill jan_bill : ℚ) : 
  (feb_bill / jan_bill = 3 / 2) → 
  ((feb_bill + 10) / jan_bill = 5 / 3) → 
  jan_bill = 60 := by
sorry

end january_oil_bill_l2821_282139


namespace m_minus_n_values_l2821_282132

theorem m_minus_n_values (m n : ℤ) 
  (hm : |m| = 5)
  (hn : |n| = 7)
  (hmn_neg : m + n < 0) :
  m - n = 12 ∨ m - n = 2 := by
  sorry

end m_minus_n_values_l2821_282132


namespace radius_vector_coordinates_l2821_282122

/-- Given a point M with coordinates (-2, 5, 0) in a rectangular coordinate system,
    prove that the coordinates of its radius vector OM are equal to (-2, 5, 0). -/
theorem radius_vector_coordinates (M : ℝ × ℝ × ℝ) (h : M = (-2, 5, 0)) :
  M = (-2, 5, 0) := by sorry

end radius_vector_coordinates_l2821_282122


namespace unique_valid_square_l2821_282116

/-- A perfect square less than 100 with ones digit 5, 6, or 7 -/
def ValidSquare (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k^2 ∧ n < 100 ∧ (n % 10 = 5 ∨ n % 10 = 6 ∨ n % 10 = 7)

/-- There is exactly one perfect square less than 100 with ones digit 5, 6, or 7 -/
theorem unique_valid_square : ∃! (n : ℕ), ValidSquare n :=
sorry

end unique_valid_square_l2821_282116


namespace cost_calculation_l2821_282130

/-- The total cost of buying bread and drinks -/
def total_cost (a b : ℝ) : ℝ := 2 * a + 3 * b

/-- Theorem: The total cost of buying 2 pieces of bread at 'a' yuan each 
    and 3 bottles of drink at 'b' yuan each is equal to 2a+3b yuan -/
theorem cost_calculation (a b : ℝ) : 
  total_cost a b = 2 * a + 3 * b := by sorry

end cost_calculation_l2821_282130


namespace odd_increasing_nonneg_implies_increasing_neg_l2821_282186

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is increasing on a set S if f(x) ≤ f(y) for all x, y ∈ S with x ≤ y -/
def IsIncreasingOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

theorem odd_increasing_nonneg_implies_increasing_neg
  (f : ℝ → ℝ) (h_odd : IsOdd f) (h_incr_nonneg : IsIncreasingOn f (Set.Ici 0)) :
  IsIncreasingOn f (Set.Iic 0) :=
sorry

end odd_increasing_nonneg_implies_increasing_neg_l2821_282186


namespace unique_fish_count_l2821_282129

/-- The number of unique fish owned by four friends given specific conditions -/
theorem unique_fish_count :
  let micah_fish : ℕ := 7
  let kenneth_fish : ℕ := 3 * micah_fish
  let matthias_fish : ℕ := kenneth_fish - 15
  let gabrielle_fish : ℕ := 2 * (micah_fish + kenneth_fish + matthias_fish)
  let micah_matthias_shared : ℕ := 4
  let kenneth_gabrielle_shared : ℕ := 6
  (micah_fish + kenneth_fish + matthias_fish + gabrielle_fish) - 
  (micah_matthias_shared + kenneth_gabrielle_shared) = 92 :=
by sorry

end unique_fish_count_l2821_282129


namespace parabola_vertex_coordinates_l2821_282179

/-- The vertex coordinates of the parabola y = x^2 + 2x - 2 are (-1, -3) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x => x^2 + 2*x - 2
  ∃ (h : ℝ → ℝ), (∀ x, f x = h (x + 1) - 3) ∧ (∀ x, h x ≥ 0) :=
sorry

end parabola_vertex_coordinates_l2821_282179


namespace point_in_fourth_quadrant_l2821_282176

/-- A point P(x, y) is in the fourth quadrant if x > 0 and y < 0 -/
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The x-coordinate of point P -/
def x_coord (x : ℝ) : ℝ := 2 * x + 6

/-- The y-coordinate of point P -/
def y_coord (x : ℝ) : ℝ := 5 * x

theorem point_in_fourth_quadrant (x : ℝ) :
  in_fourth_quadrant (x_coord x) (y_coord x) ↔ -3 < x ∧ x < 0 := by
  sorry

end point_in_fourth_quadrant_l2821_282176


namespace tickets_for_pesos_l2821_282193

/-- Given that T tickets cost R dollars and 10 pesos is worth 40 dollars,
    this theorem proves that the number of tickets that can be purchased
    for P pesos is 4PT/R. -/
theorem tickets_for_pesos (T R P : ℝ) (h1 : T > 0) (h2 : R > 0) (h3 : P > 0) :
  let dollars_per_peso : ℝ := 40 / 10
  let pesos_in_dollars : ℝ := P * dollars_per_peso
  let tickets_per_dollar : ℝ := T / R
  tickets_per_dollar * pesos_in_dollars = 4 * P * T / R :=
by sorry

end tickets_for_pesos_l2821_282193


namespace textbook_recycling_savings_scientific_notation_l2821_282174

theorem textbook_recycling_savings_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    (31680000000 : ℝ) = a * (10 : ℝ) ^ n ∧
    a = 3.168 ∧ n = 10 := by
  sorry

end textbook_recycling_savings_scientific_notation_l2821_282174


namespace complex_number_magnitude_product_l2821_282154

theorem complex_number_magnitude_product (z₁ z₂ : ℂ) : 
  Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂ := by sorry

end complex_number_magnitude_product_l2821_282154


namespace square_root_729_l2821_282173

theorem square_root_729 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 729) : x = 27 := by
  sorry

end square_root_729_l2821_282173


namespace arithmetic_sequence_reaches_negative_27_l2821_282111

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_reaches_negative_27 :
  ∃ n : ℕ, arithmetic_sequence 1 (-2) n = -27 ∧ n = 15 := by
sorry

end arithmetic_sequence_reaches_negative_27_l2821_282111


namespace triangle_returns_after_six_rotations_l2821_282138

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents a rotation around a point by a given angle -/
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Performs a single rotation of the triangle around one of its vertices -/
def rotateTriangle (t : Triangle) (vertex : Fin 3) : Triangle := sorry

/-- Performs six successive rotations of the triangle -/
def sixRotations (t : Triangle) : Triangle := sorry

/-- Theorem stating that after six rotations, the triangle returns to its original position -/
theorem triangle_returns_after_six_rotations (t : Triangle) : 
  sixRotations t = t := by sorry

end triangle_returns_after_six_rotations_l2821_282138


namespace simplify_algebraic_expression_l2821_282115

theorem simplify_algebraic_expression (a b : ℝ) (h : b ≠ 0) :
  (14 * a^3 * b^2 - 7 * a * b^2) / (7 * a * b^2) = 2 * a^2 - 1 :=
by sorry

end simplify_algebraic_expression_l2821_282115


namespace power_sum_sequence_l2821_282110

theorem power_sum_sequence (a b x y : ℝ) 
  (eq1 : a*x + b*y = 3)
  (eq2 : a*x^2 + b*y^2 = 7)
  (eq3 : a*x^3 + b*y^3 = 16)
  (eq4 : a*x^4 + b*y^4 = 42) :
  a*x^5 + b*y^5 = 20 := by
sorry

end power_sum_sequence_l2821_282110


namespace second_polygon_sides_l2821_282178

/-- Given two regular polygons with the same perimeter, where the first has 45 sides
    and a side length three times as long as the second, prove that the second polygon
    has 135 sides. -/
theorem second_polygon_sides (p1 p2 : ℕ) (s : ℝ) : 
  p1 = 45 →                          -- The first polygon has 45 sides
  p1 * (3 * s) = p2 * s →            -- Both polygons have the same perimeter
  p2 = 135 := by sorry

end second_polygon_sides_l2821_282178


namespace simplify_expression_l2821_282177

theorem simplify_expression (n : ℕ) : 
  (3^(n+4) - 3*(3^n) + 3^(n+2)) / (3*(3^(n+3))) = 29 / 27 := by
  sorry

end simplify_expression_l2821_282177


namespace intersection_empty_iff_a_greater_than_neg_four_l2821_282148

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + (a+2)*x + 1 = 0}
def B : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem intersection_empty_iff_a_greater_than_neg_four (a : ℝ) :
  A a ∩ B = ∅ ↔ a > -4 := by sorry

end intersection_empty_iff_a_greater_than_neg_four_l2821_282148


namespace function_equation_solution_l2821_282147

theorem function_equation_solution (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f (f x) = x * f x - a * x) →
  (∃ x y : ℝ, f x ≠ f y) →
  (∃ t : ℝ, f t = a) →
  (a = 0 ∨ a = -1) :=
by sorry

end function_equation_solution_l2821_282147


namespace fate_region_is_correct_l2821_282161

def f (x : ℝ) := x^2 + 3*x + 2
def g (x : ℝ) := 2*x + 3

def is_fate_function (f g : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, |f x - g x| ≤ 1

def fate_region (f g : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | |f x - g x| ≤ 1}

theorem fate_region_is_correct :
  fate_region f g = Set.union (Set.Icc (-2) (-1)) (Set.Icc 0 1) :=
by sorry

end fate_region_is_correct_l2821_282161
