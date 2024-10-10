import Mathlib

namespace paper_I_maximum_mark_l3571_357112

theorem paper_I_maximum_mark :
  ∀ (max_mark : ℕ) (passing_percentage : ℚ) (scored_marks failed_by : ℕ),
    passing_percentage = 52 / 100 →
    scored_marks = 45 →
    failed_by = 35 →
    (scored_marks + failed_by : ℚ) = passing_percentage * max_mark →
    max_mark = 154 :=
by
  sorry

end paper_I_maximum_mark_l3571_357112


namespace polynomial_not_factorizable_l3571_357133

theorem polynomial_not_factorizable :
  ¬ ∃ (f : Polynomial ℝ) (g : Polynomial ℝ),
    (∀ (x y : ℝ), (f.eval x) * (g.eval y) = x^200 * y^200 + 1) :=
sorry

end polynomial_not_factorizable_l3571_357133


namespace range_of_a_in_first_quadrant_l3571_357161

-- Define a complex number z with real part a and imaginary part (a-1)
def z (a : ℝ) : ℂ := Complex.mk a (a - 1)

-- Define what it means for a complex number to be in the first quadrant
def in_first_quadrant (w : ℂ) : Prop := 0 < w.re ∧ 0 < w.im

-- State the theorem
theorem range_of_a_in_first_quadrant :
  ∀ a : ℝ, in_first_quadrant (z a) ↔ a > 1 := by sorry

end range_of_a_in_first_quadrant_l3571_357161


namespace four_points_plane_count_l3571_357152

/-- A set of four points in three-dimensional space -/
structure FourPoints where
  points : Fin 4 → ℝ × ℝ × ℝ

/-- Predicate to check if three points are collinear -/
def are_collinear (p q r : ℝ × ℝ × ℝ) : Prop := sorry

/-- Predicate to check if no three points in a set of four points are collinear -/
def no_three_collinear (fp : FourPoints) : Prop :=
  ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬(are_collinear (fp.points i) (fp.points j) (fp.points k))

/-- The number of planes determined by four points -/
def num_planes (fp : FourPoints) : ℕ := sorry

/-- Theorem: Given four points in space where no three points are collinear, 
    the number of planes these points can determine is either 1 or 4 -/
theorem four_points_plane_count (fp : FourPoints) 
  (h : no_three_collinear fp) : 
  num_planes fp = 1 ∨ num_planes fp = 4 := by sorry

end four_points_plane_count_l3571_357152


namespace particular_innings_number_l3571_357115

/-- Represents the statistics of a cricket player -/
structure CricketStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after adding runs -/
def newAverage (stats : CricketStats) (newRuns : ℕ) : ℚ :=
  (stats.totalRuns + newRuns) / (stats.innings + 1)

theorem particular_innings_number
  (initialStats : CricketStats)
  (h1 : initialStats.innings = 16)
  (h2 : newAverage initialStats 112 = initialStats.average + 6)
  (h3 : newAverage initialStats 112 = 16) :
  initialStats.innings + 1 = 17 := by
  sorry

end particular_innings_number_l3571_357115


namespace average_difference_l3571_357107

theorem average_difference (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 80 + 15) / 3 + 5 → x = 40 := by
  sorry

end average_difference_l3571_357107


namespace greatest_three_digit_odd_non_divisible_l3571_357185

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_even_integers_up_to (n : ℕ) : ℕ :=
  let k := n / 2
  k * (k + 1)

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem greatest_three_digit_odd_non_divisible :
  ∀ n : ℕ,
    is_three_digit n →
    n % 2 = 1 →
    ¬(factorial n % sum_of_even_integers_up_to n = 0) →
    n ≤ 999 :=
by sorry

end greatest_three_digit_odd_non_divisible_l3571_357185


namespace train_length_calculation_l3571_357132

/-- Two trains of equal length passing each other -/
def train_passing_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) : Prop :=
  faster_speed > slower_speed ∧
  faster_speed = 75 ∧
  slower_speed = 60 ∧
  passing_time = 45 ∧
  ∃ (train_length : ℝ),
    train_length = (faster_speed - slower_speed) * passing_time * (5 / 18) / 2

theorem train_length_calculation (faster_speed slower_speed passing_time : ℝ) :
  train_passing_problem faster_speed slower_speed passing_time →
  ∃ (train_length : ℝ), train_length = 93.75 := by
  sorry

#check train_length_calculation

end train_length_calculation_l3571_357132


namespace true_discount_is_36_l3571_357194

/-- Given a banker's discount and sum due, calculate the true discount -/
def true_discount (BD : ℚ) (SD : ℚ) : ℚ :=
  BD / (1 + BD / SD)

/-- Theorem stating that for the given banker's discount and sum due, the true discount is 36 -/
theorem true_discount_is_36 :
  true_discount 42 252 = 36 := by
  sorry

end true_discount_is_36_l3571_357194


namespace tan_four_fifths_alpha_l3571_357197

theorem tan_four_fifths_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 2 * Real.sqrt 3 * (Real.cos α) ^ 2 - Real.sin (2 * α) + 2 - Real.sqrt 3 = 0) : 
  Real.tan (4 / 5 * α) = Real.sqrt 3 := by
sorry

end tan_four_fifths_alpha_l3571_357197


namespace find_S_l3571_357108

-- Define the relationship between R, S, and T
def relationship (c R S T : ℝ) : Prop :=
  R = c * (S / T)

-- Define the theorem
theorem find_S (c : ℝ) :
  relationship c (4/3) (3/7) (9/14) →
  relationship c (Real.sqrt 98) S (Real.sqrt 32) →
  S = 28 := by
  sorry


end find_S_l3571_357108


namespace unique_solution_system_l3571_357100

theorem unique_solution_system (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (2 * x₁ = x₅^2 - 23) ∧
  (4 * x₂ = x₁^2 + 7) ∧
  (6 * x₃ = x₂^2 + 14) ∧
  (8 * x₄ = x₃^2 + 23) ∧
  (10 * x₅ = x₄^2 + 34) →
  x₁ = 1 ∧ x₂ = 2 ∧ x₃ = 3 ∧ x₄ = 4 ∧ x₅ = 5 :=
by sorry

#check unique_solution_system

end unique_solution_system_l3571_357100


namespace average_of_squares_first_11_even_l3571_357104

/-- The first 11 consecutive even numbers -/
def first_11_even_numbers : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

/-- The average of squares of the first 11 consecutive even numbers -/
theorem average_of_squares_first_11_even : 
  (first_11_even_numbers.map (λ x => x^2)).sum / first_11_even_numbers.length = 184 := by
  sorry

#eval (first_11_even_numbers.map (λ x => x^2)).sum / first_11_even_numbers.length

end average_of_squares_first_11_even_l3571_357104


namespace rectangle_ratio_l3571_357170

-- Define the side length of the small squares
def small_square_side : ℝ := sorry

-- Define the side length of the large square
def large_square_side : ℝ := 3 * small_square_side

-- Define the length of the rectangle
def rectangle_length : ℝ := large_square_side

-- Define the width of the rectangle
def rectangle_width : ℝ := small_square_side

-- Theorem stating that the ratio of rectangle's length to width is 3
theorem rectangle_ratio : rectangle_length / rectangle_width = 3 := by sorry

end rectangle_ratio_l3571_357170


namespace quadratic_roots_property_l3571_357158

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
sorry

end quadratic_roots_property_l3571_357158


namespace charity_event_selection_l3571_357105

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of students -/
def total_students : ℕ := 10

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of students excluding A and B -/
def remaining_students : ℕ := total_students - 2

theorem charity_event_selection :
  choose total_students selected_students - choose remaining_students selected_students = 140 := by
  sorry

end charity_event_selection_l3571_357105


namespace six_hamburgers_left_over_l3571_357109

/-- Given a restaurant that made hamburgers and served some, calculate the number left over. -/
def hamburgers_left_over (made served : ℕ) : ℕ := made - served

/-- Prove that when 9 hamburgers are made and 3 are served, 6 are left over. -/
theorem six_hamburgers_left_over :
  hamburgers_left_over 9 3 = 6 := by
  sorry

end six_hamburgers_left_over_l3571_357109


namespace book_cost_theorem_l3571_357199

theorem book_cost_theorem (selling_price_1 selling_price_2 : ℝ) 
  (h1 : selling_price_1 = 340)
  (h2 : selling_price_2 = 350)
  (h3 : ∃ (profit : ℝ), selling_price_1 = cost + profit ∧ 
                         selling_price_2 = cost + (1.05 * profit)) :
  cost = 140 := by
  sorry

end book_cost_theorem_l3571_357199


namespace coles_fence_payment_l3571_357156

/-- Calculates Cole's payment for fencing his backyard -/
theorem coles_fence_payment
  (side_length : ℝ)
  (back_length : ℝ)
  (fence_cost_per_foot : ℝ)
  (back_neighbor_contribution_ratio : ℝ)
  (left_neighbor_contribution_ratio : ℝ)
  (h1 : side_length = 9)
  (h2 : back_length = 18)
  (h3 : fence_cost_per_foot = 3)
  (h4 : back_neighbor_contribution_ratio = 1/2)
  (h5 : left_neighbor_contribution_ratio = 1/3) :
  side_length * 2 + back_length * fence_cost_per_foot -
  (back_length * back_neighbor_contribution_ratio * fence_cost_per_foot +
   side_length * left_neighbor_contribution_ratio * fence_cost_per_foot) = 72 :=
by sorry

end coles_fence_payment_l3571_357156


namespace special_triangle_relation_l3571_357139

/-- Represents a triangle with angles A, B, C and parts C₁, C₂, C₃ -/
structure SpecialTriangle where
  A : Real
  B : Real
  C₁ : Real
  C₂ : Real
  C₃ : Real
  ang_sum : A + B + C₁ + C₂ + C₃ = 180
  B_gt_A : B > A
  C₂_largest : C₂ ≥ C₁ ∧ C₂ ≥ C₃
  C₂_between : C₁ + C₂ + C₃ = C₁ + C₃ + C₂

/-- The main theorem stating the relationship between angles and parts -/
theorem special_triangle_relation (t : SpecialTriangle) : t.C₁ - t.C₃ = t.B - t.A := by
  sorry

end special_triangle_relation_l3571_357139


namespace arithmetic_sequence_eighth_term_l3571_357145

/-- An arithmetic sequence (b_n) with given first three terms -/
def arithmetic_sequence (x : ℝ) (n : ℕ) : ℝ :=
  x^2 + (n - 1) * x

theorem arithmetic_sequence_eighth_term (x : ℝ) :
  arithmetic_sequence x 8 = 2 * x^2 + 7 * x := by
  sorry

end arithmetic_sequence_eighth_term_l3571_357145


namespace volume_of_four_cubes_l3571_357101

theorem volume_of_four_cubes (edge_length : ℝ) (num_boxes : ℕ) : 
  edge_length = 5 → num_boxes = 4 → num_boxes * (edge_length ^ 3) = 500 := by
  sorry

end volume_of_four_cubes_l3571_357101


namespace product_of_distances_to_asymptotes_l3571_357111

/-- Represents a hyperbola with equation y²/2 - x²/b = 1 -/
structure Hyperbola where
  b : ℝ
  h_b_pos : b > 0

/-- A point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : y^2 / 2 - x^2 / h.b = 1

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The distance from a point to an asymptote of the hyperbola -/
def distance_to_asymptote (h : Hyperbola) (p : PointOnHyperbola h) : ℝ := sorry

/-- The theorem stating the product of distances to asymptotes -/
theorem product_of_distances_to_asymptotes (h : Hyperbola) 
  (h_ecc : eccentricity h = 2) (p : PointOnHyperbola h) : 
  (distance_to_asymptote h p) * (distance_to_asymptote h p) = 3/2 := 
sorry

end product_of_distances_to_asymptotes_l3571_357111


namespace max_cookies_without_ingredients_l3571_357124

/-- Given a set of cookies with specific ingredient distributions, 
    prove the maximum number of cookies without any of the ingredients. -/
theorem max_cookies_without_ingredients (total_cookies : ℕ) 
    (h_total : total_cookies = 48)
    (h_choc_chips : (total_cookies / 2 : ℕ) = 24)
    (h_peanut_butter : (total_cookies * 3 / 4 : ℕ) = 36)
    (h_white_choc : (total_cookies / 3 : ℕ) = 16)
    (h_coconut : (total_cookies / 8 : ℕ) = 6) :
    ∃ (max_without : ℕ), max_without ≤ 12 ∧ 
    max_without = total_cookies - (total_cookies * 3 / 4 : ℕ) :=
by sorry

end max_cookies_without_ingredients_l3571_357124


namespace expression_evaluation_l3571_357155

theorem expression_evaluation :
  Real.sqrt 8 + (1/2)⁻¹ - 2 * Real.sin (45 * π / 180) - abs (1 - Real.sqrt 2) = 3 := by
  sorry

end expression_evaluation_l3571_357155


namespace binary_101101_to_octal_55_l3571_357116

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert decimal to octal
def decimal_to_octal (n : Nat) : List Nat :=
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

-- Theorem statement
theorem binary_101101_to_octal_55 :
  let binary := [true, false, true, true, false, true]
  decimal_to_octal (binary_to_decimal binary) = [5, 5] := by sorry

end binary_101101_to_octal_55_l3571_357116


namespace special_number_not_perfect_square_l3571_357144

/-- A number composed of exactly 100 zeros, 100 ones, and 100 twos -/
def special_number : ℕ :=
  -- We don't need to define the exact number, just its properties
  sorry

/-- The sum of digits of the special number -/
def sum_of_digits : ℕ := 300

/-- Theorem: The special number is not a perfect square -/
theorem special_number_not_perfect_square :
  ∀ n : ℕ, n ^ 2 ≠ special_number := by
  sorry

end special_number_not_perfect_square_l3571_357144


namespace calculate_expression_l3571_357129

theorem calculate_expression : (-3)^2 - (1/5)⁻¹ - Real.sqrt 8 * Real.sqrt 2 + (-2)^0 = 1 := by
  sorry

end calculate_expression_l3571_357129


namespace divisibility_cycle_l3571_357177

theorem divisibility_cycle (x y z : ℕ+) : 
  (∃ a : ℕ+, y + 1 = a * x) ∧ 
  (∃ b : ℕ+, z + 1 = b * y) ∧ 
  (∃ c : ℕ+, x + 1 = c * z) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨
   (x = 1 ∧ y = 1 ∧ z = 2) ∨
   (x = 1 ∧ y = 2 ∧ z = 3) ∨
   (x = 3 ∧ y = 5 ∧ z = 4)) :=
by sorry

end divisibility_cycle_l3571_357177


namespace arithmetic_sequence_sum_max_l3571_357141

/-- An arithmetic sequence -/
def ArithmeticSequence := ℕ+ → ℝ

/-- Sum of the first n terms of an arithmetic sequence -/
def SumOfTerms (a : ArithmeticSequence) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_sum_max (a : ArithmeticSequence) :
  (SumOfTerms a 10 > 0) →
  (SumOfTerms a 11 = 0) →
  (∀ n : ℕ+, ∃ k : ℕ+, SumOfTerms a n ≤ SumOfTerms a k) →
  ∃ k : ℕ+, (k = 5 ∨ k = 6) ∧ 
    (∀ n : ℕ+, SumOfTerms a n ≤ SumOfTerms a k) :=
by sorry

end arithmetic_sequence_sum_max_l3571_357141


namespace expression_evaluation_l3571_357188

theorem expression_evaluation (a : ℝ) : 
  let x : ℝ := 2*a + 6
  (x - 2*a + 4) = 10 := by sorry

end expression_evaluation_l3571_357188


namespace rebecca_eggs_marbles_difference_l3571_357191

theorem rebecca_eggs_marbles_difference : 
  ∀ (eggs marbles : ℕ), 
  eggs = 20 → marbles = 6 → eggs - marbles = 14 := by
  sorry

end rebecca_eggs_marbles_difference_l3571_357191


namespace marcella_lost_shoes_l3571_357163

/-- Given the initial number of shoe pairs and the final number of matching pairs,
    calculate the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (final_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * final_pairs

/-- Theorem stating that Marcella lost 10 individual shoes. -/
theorem marcella_lost_shoes : shoes_lost 23 18 = 10 := by
  sorry

end marcella_lost_shoes_l3571_357163


namespace jimmys_bet_l3571_357102

/-- Represents a fan with equally spaced blades -/
structure Fan where
  num_blades : ℕ
  revolutions_per_second : ℝ

/-- Represents a bullet shot -/
structure Bullet where
  shot_time : ℝ
  speed : ℝ

/-- Predicate that determines if a bullet can hit all blades of a fan -/
def can_hit_all_blades (f : Fan) (b : Bullet) : Prop :=
  ∃ t : ℝ, ∀ i : Fin f.num_blades, 
    ∃ k : ℤ, b.shot_time + (i : ℝ) * (1 / f.num_blades) = t + k / f.revolutions_per_second

/-- Theorem stating that for a fan with 4 blades rotating at 50 revolutions per second,
    there exists a bullet that can hit all blades -/
theorem jimmys_bet : 
  ∃ b : Bullet, can_hit_all_blades ⟨4, 50⟩ b :=
sorry

end jimmys_bet_l3571_357102


namespace closest_integer_to_thirteen_minus_sqrt_thirteen_l3571_357143

theorem closest_integer_to_thirteen_minus_sqrt_thirteen : 
  ∃ (n : ℤ), ∀ (m : ℤ), |13 - Real.sqrt 13 - n| ≤ |13 - Real.sqrt 13 - m| → n = 9 := by
  sorry

end closest_integer_to_thirteen_minus_sqrt_thirteen_l3571_357143


namespace safe_password_l3571_357131

def digits : List Nat := [6, 2, 5]

def largest_number (digits : List Nat) : Nat :=
  100 * (digits.maximum?.getD 0) + 
  10 * (digits.filter (· ≠ digits.maximum?.getD 0)).maximum?.getD 0 + 
  (digits.filter (· ∉ [digits.maximum?.getD 0, (digits.filter (· ≠ digits.maximum?.getD 0)).maximum?.getD 0])).sum

def smallest_number (digits : List Nat) : Nat :=
  100 * (digits.minimum?.getD 0) + 
  10 * (digits.filter (· ≠ digits.minimum?.getD 0)).minimum?.getD 0 + 
  (digits.filter (· ∉ [digits.minimum?.getD 0, (digits.filter (· ≠ digits.minimum?.getD 0)).minimum?.getD 0])).sum

theorem safe_password : 
  largest_number digits + smallest_number digits = 908 := by
  sorry

end safe_password_l3571_357131


namespace square_diff_over_seventy_l3571_357179

theorem square_diff_over_seventy : (535^2 - 465^2) / 70 = 1000 := by sorry

end square_diff_over_seventy_l3571_357179


namespace arrangement_theorem_l3571_357190

def num_boys : ℕ := 3
def num_girls : ℕ := 4
def total_people : ℕ := num_boys + num_girls

def arrange_condition1 : ℕ := sorry

def arrange_condition2 : ℕ := sorry

def arrange_condition3 : ℕ := sorry

def arrange_condition4 : ℕ := sorry

theorem arrangement_theorem :
  arrange_condition1 = 2160 ∧
  arrange_condition2 = 720 ∧
  arrange_condition3 = 144 ∧
  arrange_condition4 = 720 :=
sorry

end arrangement_theorem_l3571_357190


namespace floor_negative_seven_fourths_l3571_357183

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end floor_negative_seven_fourths_l3571_357183


namespace language_spoken_by_three_scientists_l3571_357160

/-- Represents a scientist at the conference -/
structure Scientist where
  id : Nat
  languages : Finset String
  languages_bound : languages.card ≤ 3

/-- The set of all scientists at the conference -/
def Scientists : Finset Scientist :=
  sorry

/-- The number of scientists at the conference -/
axiom scientists_count : Scientists.card = 9

/-- No scientist speaks more than 3 languages -/
axiom max_languages (s : Scientist) : s.languages.card ≤ 3

/-- Among any three scientists, there are two who speak a common language -/
axiom common_language_exists (s1 s2 s3 : Scientist) :
  s1 ∈ Scientists → s2 ∈ Scientists → s3 ∈ Scientists →
  ∃ (l : String), (l ∈ s1.languages ∧ l ∈ s2.languages) ∨
                  (l ∈ s1.languages ∧ l ∈ s3.languages) ∨
                  (l ∈ s2.languages ∧ l ∈ s3.languages)

/-- There exists a language spoken by at least three scientists -/
theorem language_spoken_by_three_scientists :
  ∃ (l : String), (Scientists.filter (fun s => l ∈ s.languages)).card ≥ 3 :=
sorry

end language_spoken_by_three_scientists_l3571_357160


namespace jericho_debt_ratio_l3571_357128

theorem jericho_debt_ratio :
  ∀ (jericho_money annika_debt manny_debt : ℚ),
    2 * jericho_money = 60 →
    annika_debt = 14 →
    jericho_money - annika_debt - manny_debt = 9 →
    manny_debt / annika_debt = 1 / 2 := by
  sorry

end jericho_debt_ratio_l3571_357128


namespace wages_comparison_l3571_357120

theorem wages_comparison (erica robin charles : ℝ) 
  (h1 : robin = 1.3 * erica) 
  (h2 : charles = 1.23076923076923077 * robin) : 
  charles = 1.6 * erica := by
sorry

end wages_comparison_l3571_357120


namespace angle_A_is_60_degrees_b_and_c_are_2_l3571_357136

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions given in the problem
def satisfiesCondition1 (t : Triangle) : Prop :=
  t.a^2 - t.c^2 = t.b^2 - t.b * t.c

def satisfiesCondition2 (t : Triangle) : Prop :=
  t.a = 2 ∧ t.b + t.c = 4

-- Theorem 1: If the first condition is satisfied, then angle A is 60°
theorem angle_A_is_60_degrees (t : Triangle) (h : satisfiesCondition1 t) :
  t.A = 60 * (π / 180) := by sorry

-- Theorem 2: If both conditions are satisfied, then b = 2 and c = 2
theorem b_and_c_are_2 (t : Triangle) (h1 : satisfiesCondition1 t) (h2 : satisfiesCondition2 t) :
  t.b = 2 ∧ t.c = 2 := by sorry

end angle_A_is_60_degrees_b_and_c_are_2_l3571_357136


namespace car_price_difference_car_price_difference_proof_l3571_357168

/-- The price difference between a new car and an old car, given specific conditions --/
theorem car_price_difference : ℝ → Prop :=
  fun price_difference =>
    ∃ (old_car_price : ℝ),
      -- New car costs $30,000
      let new_car_price : ℝ := 30000
      -- Down payment is 25% of new car price
      let down_payment : ℝ := 0.25 * new_car_price
      -- Old car sold at 80% of original price
      let old_car_sale_price : ℝ := 0.8 * old_car_price
      -- After selling old car and making down payment, $4000 more is needed
      old_car_sale_price + down_payment + 4000 = new_car_price ∧
      -- Price difference is the difference between new and old car prices
      price_difference = new_car_price - old_car_price ∧
      -- The price difference is $6875
      price_difference = 6875

/-- Proof of the car price difference theorem --/
theorem car_price_difference_proof : car_price_difference 6875 := by
  sorry

end car_price_difference_car_price_difference_proof_l3571_357168


namespace point_in_second_quadrant_l3571_357172

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  second_quadrant (-5 : ℝ) (4 : ℝ) :=
by sorry

end point_in_second_quadrant_l3571_357172


namespace arbelos_equal_segments_l3571_357147

/-- Arbelos type representing the geometric figure --/
structure Arbelos where
  -- Define necessary components of an arbelos
  -- (placeholder for actual definition)

/-- Point type representing a point in the plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing a line in the plane --/
structure Line where
  -- Define necessary components of a line
  -- (placeholder for actual definition)

/-- Function to check if a point is inside an arbelos --/
def isInsideArbelos (a : Arbelos) (p : Point) : Prop :=
  -- Define the condition for a point to be inside an arbelos
  sorry

/-- Function to check if two lines make equal angles with a given line --/
def makeEqualAngles (l1 l2 base : Line) : Prop :=
  -- Define the condition for two lines to make equal angles with a base line
  sorry

/-- Function to get the segment cut by an arbelos on a line --/
def segmentCutByArbelos (a : Arbelos) (l : Line) : ℝ :=
  -- Define how to calculate the segment cut by an arbelos on a line
  sorry

/-- Theorem statement --/
theorem arbelos_equal_segments 
  (a : Arbelos) (ac : Line) (d : Point) (l1 l2 : Line) :
  isInsideArbelos a d →
  makeEqualAngles l1 l2 ac →
  segmentCutByArbelos a l1 = segmentCutByArbelos a l2 :=
sorry

end arbelos_equal_segments_l3571_357147


namespace unique_students_count_l3571_357134

theorem unique_students_count (orchestra band choir : ℕ) 
  (orchestra_band orchestra_choir band_choir all_three : ℕ) :
  orchestra = 25 →
  band = 40 →
  choir = 30 →
  orchestra_band = 5 →
  orchestra_choir = 6 →
  band_choir = 4 →
  all_three = 2 →
  orchestra + band + choir - (orchestra_band + orchestra_choir + band_choir) + all_three = 82 :=
by sorry

end unique_students_count_l3571_357134


namespace distance_to_parabola_directrix_l3571_357121

/-- The distance from a point to the directrix of a parabola -/
def distance_to_directrix (a : ℝ) (P : ℝ × ℝ) : ℝ :=
  |P.1 + a|

/-- The parabola equation -/
def is_parabola (x y : ℝ) (a : ℝ) : Prop :=
  y^2 = -4*a*x

theorem distance_to_parabola_directrix :
  ∃ (a : ℝ), 
    is_parabola (-2) 4 a ∧ 
    distance_to_directrix a (-2, 4) = 4 :=
sorry

end distance_to_parabola_directrix_l3571_357121


namespace exam_results_l3571_357146

theorem exam_results (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 30)
  (h2 : failed_english = 42)
  (h3 : failed_both = 28) :
  100 - (failed_hindi + failed_english - failed_both) = 56 := by
  sorry

end exam_results_l3571_357146


namespace g_fifty_l3571_357140

/-- A function g satisfying g(xy) = xg(y) for all real x and y, and g(1) = 40 -/
def g : ℝ → ℝ := sorry

/-- The property that g(xy) = xg(y) for all real x and y -/
axiom g_prop (x y : ℝ) : g (x * y) = x * g y

/-- The property that g(1) = 40 -/
axiom g_one : g 1 = 40

/-- Theorem: g(50) = 2000 -/
theorem g_fifty : g 50 = 2000 := by sorry

end g_fifty_l3571_357140


namespace journey_distance_l3571_357171

/-- Calculates the total distance of a journey with multiple parts and a detour -/
theorem journey_distance (speed1 speed2 speed3 : ℝ) 
                         (time1 time2 time3 : ℝ) 
                         (detour_distance : ℝ) : 
  speed1 = 40 →
  speed2 = 50 →
  speed3 = 30 →
  time1 = 1.5 →
  time2 = 1 →
  time3 = 2.25 →
  detour_distance = 10 →
  speed1 * time1 + speed2 * time2 + detour_distance + speed3 * time3 = 187.5 := by
  sorry

#check journey_distance

end journey_distance_l3571_357171


namespace ellipse_and_chord_l3571_357114

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about an ellipse and a chord -/
theorem ellipse_and_chord 
  (C : Ellipse) 
  (h_ecc : C.a * C.a - C.b * C.b = (C.a * C.a) / 4) 
  (h_point : (2 : ℝ) * (2 : ℝ) / (C.a * C.a) + (-3 : ℝ) * (-3 : ℝ) / (C.b * C.b) = 1)
  (M : Point) 
  (h_M : M.x = -1 ∧ M.y = 2) :
  (∃ (D : Ellipse), D.a * D.a = 16 ∧ D.b * D.b = 12) ∧
  (∃ (l : Line), l.a = 3 ∧ l.b = -8 ∧ l.c = 19) := by
  sorry

end ellipse_and_chord_l3571_357114


namespace map_scale_l3571_357162

/-- Given a map where 10 cm represents 50 km, prove that 15 cm represents 75 km -/
theorem map_scale (scale : ℝ → ℝ) : 
  (scale 10 = 50) → (scale 15 = 75) := by
  sorry

end map_scale_l3571_357162


namespace arithmetic_geometric_equivalence_l3571_357130

def is_arithmetic_seq (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_seq (b : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, q > 1 ∧ ∀ n, b (n + 1) = b n * q

def every_term_in (b a : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m, b n = a m

theorem arithmetic_geometric_equivalence
  (a b : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_seq a d →
  is_geometric_seq b →
  a 1 = b 1 →
  a 2 = b 2 →
  (d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5) →
  every_term_in b a :=
sorry

end arithmetic_geometric_equivalence_l3571_357130


namespace fractional_equation_solution_range_l3571_357135

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, (2 * x - m) / (x + 1) = 1 ∧ x < 0) ↔ (m < -1 ∧ m ≠ -2) :=
by sorry

end fractional_equation_solution_range_l3571_357135


namespace power_of_product_l3571_357125

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by sorry

end power_of_product_l3571_357125


namespace square_area_from_rectangles_l3571_357127

/-- Given a square divided into 5 identical rectangles, where each rectangle has a perimeter of 120
    and a length that is 5 times its width, the area of the original square is 2500 -/
theorem square_area_from_rectangles (perimeter width length : ℝ) : 
  perimeter = 120 →
  length = 5 * width →
  2 * (length + width) = perimeter →
  (5 * width)^2 = 2500 :=
by sorry

end square_area_from_rectangles_l3571_357127


namespace testes_mice_most_suitable_for_meiosis_l3571_357118

/-- Represents different types of biological materials --/
inductive BiologicalMaterial
  | FertilizedEggsAscaris
  | TestesMice
  | SpermLocusts
  | BloodChickens

/-- Represents different types of cell division --/
inductive CellDivision
  | Mitosis
  | Meiosis
  | Amitosis

/-- Defines the property of a biological material undergoing a specific type of cell division --/
def undergoes (m : BiologicalMaterial) (d : CellDivision) : Prop := sorry

/-- Defines the property of a biological material being suitable for observing a specific type of cell division --/
def suitableForObserving (m : BiologicalMaterial) (d : CellDivision) : Prop := sorry

/-- Defines the property of a biological material producing a large number of cells --/
def producesLargeNumberOfCells (m : BiologicalMaterial) : Prop := sorry

/-- Theorem stating that testes of mice are the most suitable material for observing meiosis among the given options --/
theorem testes_mice_most_suitable_for_meiosis :
  suitableForObserving BiologicalMaterial.TestesMice CellDivision.Meiosis ∧
  ¬suitableForObserving BiologicalMaterial.FertilizedEggsAscaris CellDivision.Meiosis ∧
  ¬suitableForObserving BiologicalMaterial.SpermLocusts CellDivision.Meiosis ∧
  ¬suitableForObserving BiologicalMaterial.BloodChickens CellDivision.Meiosis :=
by sorry

end testes_mice_most_suitable_for_meiosis_l3571_357118


namespace min_value_expression_l3571_357159

theorem min_value_expression (x : ℝ) :
  ∃ (min : ℝ), min = -1640.25 ∧
  ∀ y : ℝ, (15 - y) * (12 - y) * (15 + y) * (12 + y) ≥ min :=
by sorry

end min_value_expression_l3571_357159


namespace total_car_production_l3571_357169

theorem total_car_production (north_america : ℕ) (europe : ℕ) 
  (h1 : north_america = 3884) (h2 : europe = 2871) : 
  north_america + europe = 6755 := by
  sorry

end total_car_production_l3571_357169


namespace prop_a_necessary_not_sufficient_l3571_357189

theorem prop_a_necessary_not_sufficient (h : ℝ) (h_pos : h > 0) :
  (∀ a b : ℝ, (|a - 1| < h ∧ |b - 1| < h) → |a - b| < 2*h) ∧
  (∃ a b : ℝ, |a - b| < 2*h ∧ ¬(|a - 1| < h ∧ |b - 1| < h)) :=
by sorry

end prop_a_necessary_not_sufficient_l3571_357189


namespace point_sqrt_6_away_from_origin_l3571_357123

-- Define a point on the number line
def Point := ℝ

-- Define the distance function
def distance (p : Point) : ℝ := |p|

-- State the theorem
theorem point_sqrt_6_away_from_origin (M : Point) 
  (h : distance M = Real.sqrt 6) : M = Real.sqrt 6 ∨ M = -Real.sqrt 6 := by
  sorry

end point_sqrt_6_away_from_origin_l3571_357123


namespace tangent_line_slope_l3571_357165

/-- The value of a for which the tangent line to y = ax - ln(x+1) at (0,0) is y = 2x -/
theorem tangent_line_slope (a : ℝ) : 
  (∀ x y : ℝ, y = a * x - Real.log (x + 1)) →
  (∃ m : ℝ, ∀ x y : ℝ, y = m * x ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → 
      |((a * h - Real.log (h + 1)) / h) - m| < ε)) →
  m = 2 →
  a = 3 :=
by sorry

end tangent_line_slope_l3571_357165


namespace function_expression_l3571_357117

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_expression (f : ℝ → ℝ) :
  is_periodic f 2 →
  is_even f →
  (∀ x ∈ Set.Icc 2 3, f x = x) →
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| :=
by sorry

end function_expression_l3571_357117


namespace probability_no_consecutive_ones_l3571_357154

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def valid_sequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

theorem probability_no_consecutive_ones (n : ℕ) :
  n = 12 →
  (valid_sequences n : ℚ) / (total_sequences n : ℚ) = 377 / 4096 := by
  sorry

#eval valid_sequences 12
#eval total_sequences 12

end probability_no_consecutive_ones_l3571_357154


namespace total_profit_calculation_l3571_357166

/-- Given the capital ratios and R's share of the profit, calculate the total profit -/
theorem total_profit_calculation (P Q R : ℕ) (r_profit : ℕ) 
  (h1 : 4 * P = 6 * Q)
  (h2 : 6 * Q = 10 * R)
  (h3 : r_profit = 900) : 
  4650 = (31 * r_profit) / 6 :=
by sorry

end total_profit_calculation_l3571_357166


namespace least_subtraction_for_divisibility_problem_solution_l3571_357110

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃! x : ℕ, x < d ∧ (n - x) % d = 0 :=
by sorry

theorem problem_solution :
  let n := 42739
  let d := 15
  (least_subtraction_for_divisibility n d (by norm_num)).choose = 4 :=
by sorry

end least_subtraction_for_divisibility_problem_solution_l3571_357110


namespace action_figures_removed_l3571_357196

theorem action_figures_removed (initial : ℕ) (added : ℕ) (final : ℕ) : 
  initial = 15 → added = 2 → final = 10 → initial + added - final = 7 := by
sorry

end action_figures_removed_l3571_357196


namespace lines_parallel_iff_a_eq_neg_one_l3571_357126

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The first line ax + 3y + 3 = 0 -/
def line1 (a : ℝ) : Line :=
  { a := a, b := 3, c := 3 }

/-- The second line x + (a-2)y + l = 0 -/
def line2 (a l : ℝ) : Line :=
  { a := 1, b := a - 2, c := l }

/-- Theorem stating that the lines are parallel if and only if a = -1 -/
theorem lines_parallel_iff_a_eq_neg_one (a l : ℝ) :
  parallel (line1 a) (line2 a l) ↔ a = -1 := by
  sorry

end lines_parallel_iff_a_eq_neg_one_l3571_357126


namespace min_value_expression_l3571_357153

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 3 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 →
  (2 * x^2 + 1) / (x * y) - 2 ≥ min_val ∧
  (2 * a^2 + 1) / (a * b) - 2 = min_val ↔ a = (Real.sqrt 3 - 1) / 2 :=
sorry

end min_value_expression_l3571_357153


namespace six_digit_numbers_with_two_zeros_l3571_357157

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers with fewer than two zeros -/
def fewer_than_two_zeros : ℕ := 826686

/-- The number of 6-digit numbers with at least two zeros -/
def at_least_two_zeros : ℕ := total_six_digit_numbers - fewer_than_two_zeros

theorem six_digit_numbers_with_two_zeros :
  at_least_two_zeros = 73314 := by
  sorry

end six_digit_numbers_with_two_zeros_l3571_357157


namespace carter_cards_l3571_357192

/-- Given that Marcus has 210 baseball cards and 58 more than Carter, 
    prove that Carter has 152 baseball cards. -/
theorem carter_cards (marcus_cards : ℕ) (difference : ℕ) (carter_cards : ℕ) 
  (h1 : marcus_cards = 210)
  (h2 : marcus_cards = carter_cards + difference)
  (h3 : difference = 58) : 
  carter_cards = 152 := by
  sorry

end carter_cards_l3571_357192


namespace nap_time_calculation_l3571_357173

/-- Calculates the remaining time for a nap given flight duration and time spent on activities -/
def remaining_nap_time (
  flight_hours : ℕ
) (
  flight_minutes : ℕ
) (
  reading_hours : ℕ
) (
  movie_hours : ℕ
) (
  dinner_minutes : ℕ
) (
  radio_minutes : ℕ
) (
  game_hours : ℕ
) (
  game_minutes : ℕ
) : ℕ :=
  let total_flight_minutes := flight_hours * 60 + flight_minutes
  let total_activity_minutes := 
    reading_hours * 60 + 
    movie_hours * 60 + 
    dinner_minutes + 
    radio_minutes + 
    game_hours * 60 + 
    game_minutes
  (total_flight_minutes - total_activity_minutes) / 60

theorem nap_time_calculation : 
  remaining_nap_time 11 20 2 4 30 40 1 10 = 3 := by
  sorry

end nap_time_calculation_l3571_357173


namespace f_definition_f_max_min_on_interval_l3571_357181

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x => 2 / (x - 1)

-- Theorem for the function definition
theorem f_definition (x : ℝ) (h : x ≠ 1) : 
  f ((x - 1) / (x + 1)) = -x - 1 := by sorry

-- Theorem for the maximum and minimum values
theorem f_max_min_on_interval : 
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc 2 6, f x ≤ max ∧ min ≤ f x) ∧ 
  (∃ x₁ ∈ Set.Icc 2 6, f x₁ = max) ∧ 
  (∃ x₂ ∈ Set.Icc 2 6, f x₂ = min) ∧ 
  max = 2 ∧ min = 2/5 := by sorry

end f_definition_f_max_min_on_interval_l3571_357181


namespace circle_equation_l3571_357151

/-- The general equation of a circle with specific properties -/
theorem circle_equation (x y : ℝ) : 
  ∃ (h k : ℝ), 
    (k = -4 * h) ∧ 
    ((3 - h)^2 + (-2 - k)^2 = (3 + (-2) - 1)^2) ∧
    (∀ (a b : ℝ), (a + b - 1 = 0) → ((a - h)^2 + (b - k)^2 ≥ (3 + (-2) - 1)^2)) →
    x^2 + y^2 - 2*x + 8*y + 9 = 0 := by
  sorry

end circle_equation_l3571_357151


namespace probability_inner_circle_l3571_357187

/-- The probability of a random point from a circle with radius 3 falling within a concentric circle with radius 1.5 -/
theorem probability_inner_circle (outer_radius inner_radius : ℝ) 
  (h_outer : outer_radius = 3)
  (h_inner : inner_radius = 1.5) :
  (π * inner_radius^2) / (π * outer_radius^2) = 1/4 := by
  sorry

end probability_inner_circle_l3571_357187


namespace sum_of_squares_l3571_357184

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_seven : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = -6/7 := by
  sorry

end sum_of_squares_l3571_357184


namespace max_profit_at_16_l3571_357103

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (k b x : ℝ) : ℝ := k * x + b

/-- Represents the daily profit as a function of selling price -/
def daily_profit (k b x : ℝ) : ℝ := (x - 12) * (sales_quantity k b x)

theorem max_profit_at_16 (k b : ℝ) :
  sales_quantity k b 15 = 50 →
  sales_quantity k b 17 = 30 →
  (∀ x, 12 ≤ x → x ≤ 18 → daily_profit k b x ≤ daily_profit k b 16) ∧
  daily_profit k b 16 = 160 :=
sorry

end max_profit_at_16_l3571_357103


namespace arrangement_inequality_l3571_357164

-- Define the arrangement function
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Define the set of valid x values
def valid_x : Set ℕ := {x | 3 ≤ x ∧ x ≤ 7}

-- State the theorem
theorem arrangement_inequality (x : ℕ) (h1 : 2 < x) (h2 : x ≤ 9) :
  A 9 x > 6 * A 9 (x - 2) ↔ x ∈ valid_x :=
sorry

end arrangement_inequality_l3571_357164


namespace therapy_cost_difference_l3571_357137

/-- Represents the pricing scheme of a psychologist -/
structure PricingScheme where
  firstHourCost : ℕ
  additionalHourCost : ℕ
  first_hour_more_expensive : firstHourCost > additionalHourCost

/-- Theorem: Given the conditions, the difference in cost between the first hour
    and each additional hour is $30 -/
theorem therapy_cost_difference (p : PricingScheme) 
  (five_hour_cost : p.firstHourCost + 4 * p.additionalHourCost = 400)
  (three_hour_cost : p.firstHourCost + 2 * p.additionalHourCost = 252) :
  p.firstHourCost - p.additionalHourCost = 30 := by
  sorry

end therapy_cost_difference_l3571_357137


namespace fans_with_all_items_l3571_357167

/-- The maximum capacity of the stadium --/
def stadium_capacity : ℕ := 3000

/-- The interval at which t-shirts are given --/
def tshirt_interval : ℕ := 50

/-- The interval at which caps are given --/
def cap_interval : ℕ := 25

/-- The interval at which wristbands are given --/
def wristband_interval : ℕ := 60

/-- Theorem stating that the number of fans receiving all three items is 10 --/
theorem fans_with_all_items : 
  (stadium_capacity / (Nat.lcm tshirt_interval (Nat.lcm cap_interval wristband_interval))) = 10 := by
  sorry

end fans_with_all_items_l3571_357167


namespace no_solutions_for_equation_l3571_357175

theorem no_solutions_for_equation : ¬∃ (x : Fin 8 → ℝ), 
  (2 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + 
  (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + (x 7)^2 = 1/9 := by
  sorry

end no_solutions_for_equation_l3571_357175


namespace fraction_simplification_l3571_357119

theorem fraction_simplification :
  (1/2 + 1/3) / (3/4 - 1/5) = 50/33 := by sorry

end fraction_simplification_l3571_357119


namespace chalk_inventory_theorem_l3571_357149

-- Define the types of chalk
inductive ChalkType
  | Regular
  | Unusual
  | Excellent

-- Define the store's chalk inventory
structure ChalkInventory where
  regular : ℕ
  unusual : ℕ
  excellent : ℕ

def initial_ratio : Fin 3 → ℕ
  | 0 => 3  -- Regular
  | 1 => 4  -- Unusual
  | 2 => 6  -- Excellent

def new_ratio : Fin 3 → ℕ
  | 0 => 2  -- Regular
  | 1 => 5  -- Unusual
  | 2 => 8  -- Excellent

theorem chalk_inventory_theorem (initial : ChalkInventory) (final : ChalkInventory) :
  -- Initial ratio condition
  initial.regular * initial_ratio 1 = initial.unusual * initial_ratio 0 ∧
  initial.regular * initial_ratio 2 = initial.excellent * initial_ratio 0 ∧
  -- New ratio condition
  final.regular * new_ratio 1 = final.unusual * new_ratio 0 ∧
  final.regular * new_ratio 2 = final.excellent * new_ratio 0 ∧
  -- Excellent chalk increase condition
  final.excellent = initial.excellent * 180 / 100 ∧
  -- Regular chalk decrease condition
  initial.regular - final.regular ≤ 10 ∧
  -- Total initial packs
  initial.regular + initial.unusual + initial.excellent = 390 :=
by sorry

end chalk_inventory_theorem_l3571_357149


namespace enthalpy_change_proof_l3571_357142

-- Define the sum of standard formation enthalpies for products
def sum_enthalpy_products : ℝ := -286.0 - 297.0

-- Define the sum of standard formation enthalpies for reactants
def sum_enthalpy_reactants : ℝ := -20.17

-- Define Hess's Law
def hess_law (products reactants : ℝ) : ℝ := products - reactants

-- Theorem statement
theorem enthalpy_change_proof :
  hess_law sum_enthalpy_products sum_enthalpy_reactants = -1125.66 := by
  sorry

end enthalpy_change_proof_l3571_357142


namespace collinear_vectors_x_value_l3571_357193

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (-1, x) (1, 2) → x = -2 := by
  sorry

end collinear_vectors_x_value_l3571_357193


namespace numbers_close_to_zero_not_set_l3571_357113

-- Define the criteria for a set
structure SetCriteria where
  definiteness : Bool
  distinctness : Bool
  unorderedness : Bool

-- Define a predicate for whether a collection can form a set
def canFormSet (c : SetCriteria) : Bool :=
  c.definiteness ∧ c.distinctness ∧ c.unorderedness

-- Define the property of being "close to 0"
def closeToZero (ε : ℝ) (x : ℝ) : Prop := abs x < ε

-- Theorem stating that "Numbers close to 0" cannot form a set
theorem numbers_close_to_zero_not_set : 
  ∃ ε > 0, ¬∃ (S : Set ℝ), (∀ x ∈ S, closeToZero ε x) ∧ 
  (canFormSet ⟨true, true, true⟩) :=
sorry

end numbers_close_to_zero_not_set_l3571_357113


namespace quadratic_function_properties_l3571_357182

-- Define the quadratic function f
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

-- Theorem statement
theorem quadratic_function_properties :
  (∀ x, f x ≥ 1) ∧  -- Minimum value is 1
  (f 0 = 3) ∧ (f 2 = 3) ∧  -- f(0) = f(2) = 3
  (∀ x, f x = 2 * x^2 - 4 * x + 3) ∧  -- Expression of f(x)
  (∀ a, (0 < a ∧ a < 1/3) ↔ 
    (∃ x y, 3*a ≤ x ∧ x < y ∧ y ≤ a+1 ∧ f x > f y ∧ 
    ∃ z, x < z ∧ z < y ∧ f z < f y)) ∧  -- Non-monotonic condition
  (∀ m, m < -1 ↔ 
    (∀ x, -1 ≤ x ∧ x ≤ 1 → f x > 2*x + 2*m + 1)) :=
by sorry

end quadratic_function_properties_l3571_357182


namespace yoongi_behind_count_l3571_357106

/-- Given a line of students, calculate the number of students behind a specific position. -/
def studentsBehindinLine (totalStudents : ℕ) (position : ℕ) : ℕ :=
  totalStudents - position

theorem yoongi_behind_count :
  let totalStudents : ℕ := 20
  let jungkookPosition : ℕ := 3
  let yoongiPosition : ℕ := jungkookPosition + 1
  studentsBehindinLine totalStudents yoongiPosition = 16 := by
  sorry

end yoongi_behind_count_l3571_357106


namespace reporters_not_covering_politics_l3571_357122

theorem reporters_not_covering_politics 
  (local_politics_coverage : Real) 
  (non_local_politics_ratio : Real) 
  (h1 : local_politics_coverage = 0.12)
  (h2 : non_local_politics_ratio = 0.4) :
  1 - (local_politics_coverage / (1 - non_local_politics_ratio)) = 0.8 := by
sorry

end reporters_not_covering_politics_l3571_357122


namespace smallest_num_neighbors_correct_l3571_357186

/-- The number of points on the circumference of the circle -/
def num_points : ℕ := 2005

/-- The maximum angle (in degrees) that a chord can subtend at the center for two points to be considered neighbors -/
def max_angle : ℝ := 10

/-- Definition of the smallest number of pairs of neighbors function -/
def smallest_num_neighbors (n : ℕ) (θ : ℝ) : ℕ :=
  25 * (Nat.choose 57 2) + 10 * (Nat.choose 58 2)

/-- Theorem stating that the smallest number of pairs of neighbors for the given conditions is correct -/
theorem smallest_num_neighbors_correct :
  smallest_num_neighbors num_points max_angle =
  25 * (Nat.choose 57 2) + 10 * (Nat.choose 58 2) :=
by sorry

end smallest_num_neighbors_correct_l3571_357186


namespace condensed_milk_higher_caloric_value_l3571_357198

theorem condensed_milk_higher_caloric_value (a b c : ℝ) : 
  (3*a + 4*b + 2*c > 2*a + 3*b + 4*c) → 
  (3*a + 4*b + 2*c > 4*a + 2*b + 3*c) → 
  b > c := by
sorry

end condensed_milk_higher_caloric_value_l3571_357198


namespace remainder_problem_l3571_357138

theorem remainder_problem (x : ℤ) : 
  x % 82 = 5 → (x + 13) % 41 = 18 := by
sorry

end remainder_problem_l3571_357138


namespace systematic_sampling_distribution_l3571_357180

/-- Represents a building in the summer camp -/
inductive Building
| A
| B
| C

/-- Calculates the number of students selected from each building using systematic sampling -/
def systematic_sampling (total_students : ℕ) (sample_size : ℕ) (start : ℕ) : Building → ℕ :=
  λ b =>
    match b with
    | Building.A => sorry
    | Building.B => sorry
    | Building.C => sorry

theorem systematic_sampling_distribution :
  let total_students := 400
  let sample_size := 50
  let start := 5
  (systematic_sampling total_students sample_size start Building.A = 25) ∧
  (systematic_sampling total_students sample_size start Building.B = 12) ∧
  (systematic_sampling total_students sample_size start Building.C = 13) :=
by sorry

end systematic_sampling_distribution_l3571_357180


namespace system_of_equations_range_l3571_357174

theorem system_of_equations_range (x y m : ℝ) : 
  x + 2*y = 1 - m →
  2*x + y = 3 →
  x + y > 0 →
  m < 4 := by
sorry

end system_of_equations_range_l3571_357174


namespace f_range_l3571_357195

-- Define the function
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- State the theorem
theorem f_range : 
  Set.range f = { y | y ≥ 2 } := by sorry

end f_range_l3571_357195


namespace intersection_of_A_and_B_l3571_357148

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l3571_357148


namespace moving_circle_trajectory_l3571_357176

/-- The circle C -/
def circle_C (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 100

/-- Point A -/
def point_A : ℝ × ℝ := (4, 0)

/-- The trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

/-- Theorem: The trajectory of the center of a moving circle that is tangent to circle C
    and passes through point A is described by the equation x²/25 + y²/9 = 1 -/
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 → 
      (∃ (x'' y'' : ℝ), circle_C x'' y'' ∧ (x' - x'')^2 + (y' - y'')^2 = 0)) ∧
    ((x - point_A.1)^2 + (y - point_A.2)^2 = r^2)) →
  trajectory x y :=
by sorry

end moving_circle_trajectory_l3571_357176


namespace alien_abduction_percentage_l3571_357178

/-- The number of people initially abducted by the alien -/
def initial_abducted : ℕ := 200

/-- The number of people taken away after returning some -/
def taken_away : ℕ := 40

/-- The number of people left on Earth after returning some and taking away others -/
def left_on_earth : ℕ := 160

/-- The percentage of people returned by the alien -/
def percentage_returned : ℚ := (left_on_earth : ℚ) / (initial_abducted : ℚ) * 100

theorem alien_abduction_percentage :
  percentage_returned = 80 := by sorry

end alien_abduction_percentage_l3571_357178


namespace candy_pack_cost_l3571_357150

theorem candy_pack_cost (cory_has : ℝ) (cory_needs : ℝ) (num_packs : ℕ) :
  cory_has = 20 →
  cory_needs = 78 →
  num_packs = 2 →
  (cory_has + cory_needs) / num_packs = 49 := by
  sorry

end candy_pack_cost_l3571_357150
