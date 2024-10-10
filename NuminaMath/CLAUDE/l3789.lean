import Mathlib

namespace coefficient_x_squared_in_expansion_l3789_378909

/-- The coefficient of x^2 in the expansion of (1+x)^7(1-x) -/
def coefficient_x_squared : ℤ := 14

/-- The expansion of (1+x)^7(1-x) -/
def expansion (x : ℝ) : ℝ := (1 + x)^7 * (1 - x)

theorem coefficient_x_squared_in_expansion :
  (∃ f : ℝ → ℝ, ∃ g : ℝ → ℝ, expansion = λ x => coefficient_x_squared * x^2 + x * f x + g x) :=
sorry

end coefficient_x_squared_in_expansion_l3789_378909


namespace base_9_8_conversion_l3789_378923

/-- Represents a number in a given base -/
def BaseRepresentation (base : ℕ) (tens_digit : ℕ) (ones_digit : ℕ) : ℕ :=
  base * tens_digit + ones_digit

theorem base_9_8_conversion : 
  ∃ (n : ℕ) (C D : ℕ), 
    C < 9 ∧ D < 9 ∧ D < 8 ∧ 
    n = BaseRepresentation 9 C D ∧
    n = BaseRepresentation 8 D C ∧
    n = 71 := by
  sorry

end base_9_8_conversion_l3789_378923


namespace initial_books_count_l3789_378918

def books_sold : ℕ := 26
def books_left : ℕ := 7

theorem initial_books_count :
  books_sold + books_left = 33 := by sorry

end initial_books_count_l3789_378918


namespace income_comparison_l3789_378905

theorem income_comparison (tim mary juan : ℝ) 
  (h1 : mary = 1.6 * tim) 
  (h2 : mary = 0.8 * juan) : 
  tim = 0.5 * juan := by
  sorry

end income_comparison_l3789_378905


namespace final_selling_price_l3789_378975

/-- Calculate the final selling price of three items with given costs, profit/loss percentages, discount, and tax. -/
theorem final_selling_price (cycle_cost scooter_cost motorbike_cost : ℚ)
  (cycle_loss_percent scooter_profit_percent motorbike_profit_percent : ℚ)
  (discount_percent tax_percent : ℚ) :
  let cycle_price := cycle_cost * (1 - cycle_loss_percent)
  let scooter_price := scooter_cost * (1 + scooter_profit_percent)
  let motorbike_price := motorbike_cost * (1 + motorbike_profit_percent)
  let total_price := cycle_price + scooter_price + motorbike_price
  let discounted_price := total_price * (1 - discount_percent)
  let final_price := discounted_price * (1 + tax_percent)
  cycle_cost = 2300 ∧
  scooter_cost = 12000 ∧
  motorbike_cost = 25000 ∧
  cycle_loss_percent = 0.30 ∧
  scooter_profit_percent = 0.25 ∧
  motorbike_profit_percent = 0.15 ∧
  discount_percent = 0.10 ∧
  tax_percent = 0.05 →
  final_price = 41815.20 := by
sorry

end final_selling_price_l3789_378975


namespace fraction_reduction_l3789_378919

theorem fraction_reduction (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (4*x - 4*y) / (4*x * 4*y) = (1/4) * ((x - y) / (x * y)) :=
by sorry

end fraction_reduction_l3789_378919


namespace recreation_spending_comparison_l3789_378998

theorem recreation_spending_comparison (W : ℝ) : 
  let last_week_recreation := 0.15 * W
  let this_week_wages := 0.8 * W
  let this_week_recreation := 0.5 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 267 := by
sorry

end recreation_spending_comparison_l3789_378998


namespace range_of_a_l3789_378997

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_ineq : f (a + 1) ≤ f 4) :
  -5 ≤ a ∧ a ≤ 3 :=
by sorry

end range_of_a_l3789_378997


namespace smallest_number_satisfying_conditions_l3789_378935

theorem smallest_number_satisfying_conditions : 
  ∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → ¬((m + 3) % 5 = 0 ∧ (m - 3) % 6 = 0)) ∧
    (n + 3) % 5 = 0 ∧ 
    (n - 3) % 6 = 0 ∧
    n = 27 := by
  sorry

end smallest_number_satisfying_conditions_l3789_378935


namespace wednesday_savings_l3789_378900

/-- Represents Donny's savings throughout the week -/
structure WeekSavings where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Calculates the total savings before Thursday -/
def total_savings (s : WeekSavings) : ℕ :=
  s.monday + s.tuesday + s.wednesday

theorem wednesday_savings (s : WeekSavings) 
  (h1 : s.monday = 15)
  (h2 : s.tuesday = 28)
  (h3 : total_savings s / 2 = 28) : 
  s.wednesday = 13 := by
sorry

end wednesday_savings_l3789_378900


namespace smallest_angle_proof_l3789_378953

def AP : ℝ := 2

noncomputable def smallest_angle (x : ℝ) : ℝ :=
  Real.arctan (Real.sqrt 2 / 4)

theorem smallest_angle_proof (x : ℝ) : 
  smallest_angle x = Real.arctan (Real.sqrt 2 / 4) :=
sorry

end smallest_angle_proof_l3789_378953


namespace vowel_initial_probability_theorem_l3789_378951

/-- The number of students in the class -/
def total_students : ℕ := 26

/-- The number of vowels (including 'Y') -/
def vowel_count : ℕ := 6

/-- The probability of selecting a student with vowel initials -/
def vowel_initial_probability : ℚ := 3 / 13

/-- Theorem stating the probability of selecting a student with vowel initials -/
theorem vowel_initial_probability_theorem :
  (vowel_count : ℚ) / total_students = vowel_initial_probability := by
  sorry

end vowel_initial_probability_theorem_l3789_378951


namespace neil_fraction_of_packs_l3789_378999

def total_marbles : ℕ := 400
def marbles_per_pack : ℕ := 10
def packs_kept_by_leo : ℕ := 25
def fraction_to_manny : ℚ := 1/4

theorem neil_fraction_of_packs (total_packs : ℕ) (packs_given_away : ℕ) 
  (packs_to_manny : ℕ) (packs_to_neil : ℕ) :
  total_packs = total_marbles / marbles_per_pack →
  packs_given_away = total_packs - packs_kept_by_leo →
  packs_to_manny = ⌊(fraction_to_manny * packs_given_away : ℚ)⌋ →
  packs_to_neil = packs_given_away - packs_to_manny →
  (packs_to_neil : ℚ) / packs_given_away = 4/5 := by
  sorry

end neil_fraction_of_packs_l3789_378999


namespace foreign_student_percentage_l3789_378924

theorem foreign_student_percentage 
  (total_students : ℕ) 
  (new_foreign_students : ℕ) 
  (future_foreign_students : ℕ) :
  total_students = 1800 →
  new_foreign_students = 200 →
  future_foreign_students = 740 →
  (↑(future_foreign_students - new_foreign_students) / ↑total_students : ℚ) = 30 / 100 := by
sorry

end foreign_student_percentage_l3789_378924


namespace remaining_distance_l3789_378957

-- Define the total distance to the concert
def total_distance : ℕ := 78

-- Define the distance already driven
def distance_driven : ℕ := 32

-- Theorem to prove the remaining distance
theorem remaining_distance : total_distance - distance_driven = 46 := by
  sorry

end remaining_distance_l3789_378957


namespace triangle_properties_l3789_378996

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define geometric elements
def angleBisector (t : Triangle) : Fin 3 → Set (ℝ × ℝ) := sorry
def median (t : Triangle) : Fin 3 → Set (ℝ × ℝ) := sorry
def altitude (t : Triangle) : Fin 3 → Set (ℝ × ℝ) := sorry

-- Define properties
def isInside (t : Triangle) (s : Set (ℝ × ℝ)) : Prop := sorry
def isRightTriangle (t : Triangle) : Prop := sorry
def isLine (s : Set (ℝ × ℝ)) : Prop := sorry
def isRay (s : Set (ℝ × ℝ)) : Prop := sorry
def isLineSegment (s : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem triangle_properties :
  ∃ (t : Triangle),
    (¬ (∀ i : Fin 3, isInside t (angleBisector t i) ∧ isInside t (median t i) ∧ isInside t (altitude t i))) ∧
    (isRightTriangle t → ∃ i j : Fin 3, i ≠ j ∧ altitude t i ≠ altitude t j) ∧
    (∃ i : Fin 3, isInside t (altitude t i)) ∧
    (¬ (∀ i : Fin 3, isLine (altitude t i) ∧ isRay (angleBisector t i) ∧ isLineSegment (median t i))) :=
by sorry

end triangle_properties_l3789_378996


namespace rectangle_circle_area_ratio_l3789_378946

theorem rectangle_circle_area_ratio (w : ℝ) (r : ℝ) (h1 : w > 0) (h2 : r > 0) :
  let l := 2 * w
  let rectangle_perimeter := 2 * l + 2 * w
  let circle_circumference := 2 * Real.pi * r
  rectangle_perimeter = circle_circumference →
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
    sorry

end rectangle_circle_area_ratio_l3789_378946


namespace max_min_product_l3789_378931

theorem max_min_product (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 12) (h_prod_sum : a * b + b * c + c * a = 35) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 3 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 3 :=
sorry

end max_min_product_l3789_378931


namespace election_votes_theorem_l3789_378969

theorem election_votes_theorem (emily_votes : ℕ) (emily_fraction : ℚ) (dexter_fraction : ℚ) :
  emily_votes = 48 →
  emily_fraction = 4 / 15 →
  dexter_fraction = 1 / 3 →
  ∃ (total_votes : ℕ),
    (emily_votes : ℚ) / total_votes = emily_fraction ∧
    total_votes = 180 :=
by sorry

end election_votes_theorem_l3789_378969


namespace discount_calculation_l3789_378988

/-- Proves that a 20% discount followed by a 15% discount on an item 
    originally priced at 450 results in a final price of 306 -/
theorem discount_calculation (original_price : ℝ) (first_discount second_discount final_price : ℝ) :
  original_price = 450 ∧ 
  first_discount = 20 ∧ 
  second_discount = 15 ∧ 
  final_price = 306 →
  original_price * (1 - first_discount / 100) * (1 - second_discount / 100) = final_price :=
by sorry

end discount_calculation_l3789_378988


namespace bread_per_sandwich_proof_l3789_378985

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := 6

/-- The number of pieces of bread per sandwich -/
def bread_per_sandwich : ℕ := 2

theorem bread_per_sandwich_proof :
  saturday_sandwiches * bread_per_sandwich + sunday_sandwiches * bread_per_sandwich = total_bread :=
by sorry

end bread_per_sandwich_proof_l3789_378985


namespace square_sum_from_means_l3789_378921

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 110) : 
  x^2 + y^2 = 1380 := by
  sorry

end square_sum_from_means_l3789_378921


namespace evaluate_64_to_5_6_l3789_378981

theorem evaluate_64_to_5_6 : (64 : ℝ) ^ (5/6) = 32 := by
  sorry

end evaluate_64_to_5_6_l3789_378981


namespace conical_funnel_area_l3789_378912

/-- The area of cardboard required for a conical funnel -/
theorem conical_funnel_area (slant_height : ℝ) (base_circumference : ℝ) 
  (h1 : slant_height = 6)
  (h2 : base_circumference = 6 * Real.pi) : 
  (1 / 2 : ℝ) * base_circumference * slant_height = 18 * Real.pi := by
  sorry

end conical_funnel_area_l3789_378912


namespace mark_fish_problem_l3789_378955

/-- Calculates the total number of young fish given the number of tanks, 
    pregnant fish per tank, and young per fish. -/
def total_young_fish (num_tanks : ℕ) (fish_per_tank : ℕ) (young_per_fish : ℕ) : ℕ :=
  num_tanks * fish_per_tank * young_per_fish

/-- Theorem stating that with 3 tanks, 4 pregnant fish per tank, and 20 young per fish, 
    the total number of young fish is 240. -/
theorem mark_fish_problem :
  total_young_fish 3 4 20 = 240 := by
  sorry

end mark_fish_problem_l3789_378955


namespace multiply_by_six_l3789_378948

theorem multiply_by_six (x : ℚ) (h : x / 11 = 2) : 6 * x = 132 := by
  sorry

end multiply_by_six_l3789_378948


namespace ruth_apples_l3789_378980

/-- The number of apples Ruth ends up with after a series of events -/
def final_apples (initial : ℕ) (shared : ℕ) (gift : ℕ) : ℕ :=
  let remaining := initial - shared
  let after_sister := remaining - remaining / 2
  after_sister + gift

/-- Theorem stating that Ruth ends up with 105 apples -/
theorem ruth_apples : final_apples 200 5 7 = 105 := by
  sorry

end ruth_apples_l3789_378980


namespace right_rectangular_prism_volume_l3789_378906

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 36) 
  (h2 : a * c = 54) 
  (h3 : b * c = 72) : 
  a * b * c = 648 := by
sorry

end right_rectangular_prism_volume_l3789_378906


namespace problem_solution_l3789_378944

theorem problem_solution (m n : ℝ) (h1 : m + 1/m = -4) (h2 : n + 1/n = -4) (h3 : m ≠ n) : 
  m * (n + 1) + n = -3 := by
sorry

end problem_solution_l3789_378944


namespace range_of_x_inequality_l3789_378952

theorem range_of_x_inequality (x : ℝ) : 
  (∀ (a b : ℝ), a ≠ 0 → |a + b| + |a - b| ≥ |a| * |x - 2|) ↔ 0 ≤ x ∧ x ≤ 4 := by
  sorry

end range_of_x_inequality_l3789_378952


namespace snickers_count_l3789_378968

theorem snickers_count (total : ℕ) (mars : ℕ) (butterfingers : ℕ) (snickers : ℕ) : 
  total = 12 → mars = 2 → butterfingers = 7 → total = mars + butterfingers + snickers → snickers = 3 := by
  sorry

end snickers_count_l3789_378968


namespace simplified_ratio_of_stickers_l3789_378959

theorem simplified_ratio_of_stickers (kate_stickers : ℕ) (jenna_stickers : ℕ) 
  (h1 : kate_stickers = 21) (h2 : jenna_stickers = 12) : 
  (kate_stickers / Nat.gcd kate_stickers jenna_stickers : ℚ) / 
  (jenna_stickers / Nat.gcd kate_stickers jenna_stickers : ℚ) = 7 / 4 := by
  sorry

end simplified_ratio_of_stickers_l3789_378959


namespace divisibility_implies_equality_l3789_378990

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ 0 → ∃ q : ℤ, a - k^n = (b - k) * q) →
  a = b^n := by sorry

end divisibility_implies_equality_l3789_378990


namespace intercept_ratio_l3789_378954

/-- Given two lines with the same y-intercept (0, b) where b ≠ 0,
    if the first line has slope 12 and x-intercept (s, 0),
    and the second line has slope 8 and x-intercept (t, 0),
    then s/t = 2/3 -/
theorem intercept_ratio (b s t : ℝ) (hb : b ≠ 0)
  (h1 : 0 = 12 * s + b) (h2 : 0 = 8 * t + b) : s / t = 2 / 3 := by
  sorry

end intercept_ratio_l3789_378954


namespace purely_imaginary_complex_number_l3789_378915

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = (a^2 - 4*a + 3) + Complex.I * (a - 1)) → a = 3 := by
  sorry

end purely_imaginary_complex_number_l3789_378915


namespace fiscal_revenue_scientific_notation_l3789_378926

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

/-- Rounds a ScientificNotation to a specified number of significant figures -/
def roundToSignificantFigures (sn : ScientificNotation) (figures : ℕ) : ScientificNotation := sorry

/-- The fiscal revenue in yuan -/
def fiscalRevenue : ℝ := 1073 * 10^9

theorem fiscal_revenue_scientific_notation :
  roundToSignificantFigures (toScientificNotation fiscalRevenue) 2 =
  ScientificNotation.mk 1.07 11 (by norm_num) :=
sorry

end fiscal_revenue_scientific_notation_l3789_378926


namespace amicable_pairs_theorem_l3789_378979

/-- Sum of divisors of a number -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Two numbers are amicable if the sum of proper divisors of each equals the other number -/
def is_amicable_pair (m n : ℕ) : Prop :=
  sum_of_divisors m = m + n ∧ sum_of_divisors n = m + n

/-- The main theorem stating that the given pairs are amicable -/
theorem amicable_pairs_theorem :
  let pair1_1 := 3^3 * 5 * 7 * 71
  let pair1_2 := 3^3 * 5 * 17 * 31
  let pair2_1 := 3^2 * 5 * 13 * 79 * 29
  let pair2_2 := 3^2 * 5 * 13 * 11 * 199
  is_amicable_pair pair1_1 pair1_2 ∧ is_amicable_pair pair2_1 pair2_2 := by
  sorry

end amicable_pairs_theorem_l3789_378979


namespace evaluate_expression_l3789_378916

theorem evaluate_expression : (120 : ℚ) / 6 * 2 / 3 = 40 / 3 := by
  sorry

end evaluate_expression_l3789_378916


namespace average_study_time_difference_l3789_378947

def average_difference (differences : List Int) : ℚ :=
  (differences.sum : ℚ) / differences.length

theorem average_study_time_difference 
  (differences : List Int) 
  (h1 : differences.length = 5) :
  average_difference differences = 
    (differences.sum : ℚ) / 5 := by sorry

end average_study_time_difference_l3789_378947


namespace distinct_values_of_combination_sum_l3789_378967

theorem distinct_values_of_combination_sum :
  ∃ (S : Finset ℕ), 
    (∀ r : ℕ, r + 1 ≤ 10 ∧ 17 - r ≤ 10 → 
      (Nat.choose 10 (r + 1) + Nat.choose 10 (17 - r)) ∈ S) ∧
    Finset.card S = 2 :=
sorry

end distinct_values_of_combination_sum_l3789_378967


namespace parallel_transitive_perpendicular_to_plane_parallel_l3789_378925

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicularity relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Axiom for transitivity of parallelism
axiom parallel_trans {a b c : Line} : parallel a b → parallel b c → parallel a c

-- Axiom for perpendicular lines to a plane being parallel
axiom perpendicular_parallel {a b : Line} {γ : Plane} : 
  perpendicular a γ → perpendicular b γ → parallel a b

-- Theorem 1: If a∥b and b∥c, then a∥c
theorem parallel_transitive {a b c : Line} : 
  parallel a b → parallel b c → parallel a c :=
by sorry

-- Theorem 2: If a⊥γ and b⊥γ, then a∥b
theorem perpendicular_to_plane_parallel {a b : Line} {γ : Plane} : 
  perpendicular a γ → perpendicular b γ → parallel a b :=
by sorry

end parallel_transitive_perpendicular_to_plane_parallel_l3789_378925


namespace multiplication_problem_solution_l3789_378917

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the multiplication problem -/
structure MultiplicationProblem where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D
  equation : A.val * 100 + B.val * 10 + C.val = C.val * 100 + C.val * 10 + A.val

theorem multiplication_problem_solution (p : MultiplicationProblem) : p.A.val + p.C.val = 10 := by
  sorry

end multiplication_problem_solution_l3789_378917


namespace congruence_problem_l3789_378974

theorem congruence_problem (a b n : ℤ) : 
  a ≡ 25 [ZMOD 60] →
  b ≡ 85 [ZMOD 60] →
  150 ≤ n →
  n ≤ 241 →
  (a - b ≡ n [ZMOD 60]) ↔ (n = 180 ∨ n = 240) :=
by sorry

end congruence_problem_l3789_378974


namespace dvd_pack_cost_l3789_378929

theorem dvd_pack_cost (total_amount : ℕ) (num_packs : ℕ) (cost_per_pack : ℕ) :
  total_amount = 104 →
  num_packs = 4 →
  cost_per_pack = total_amount / num_packs →
  cost_per_pack = 26 := by
  sorry

end dvd_pack_cost_l3789_378929


namespace regression_lines_intersection_l3789_378945

-- Define the regression line type
def RegressionLine := ℝ → ℝ

-- Define the property that a regression line passes through a point
def passes_through (l : RegressionLine) (p : ℝ × ℝ) : Prop :=
  l p.1 = p.2

-- Theorem statement
theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : passes_through l₁ (s, t))
  (h₂ : passes_through l₂ (s, t)) :
  ∃ p : ℝ × ℝ, p = (s, t) ∧ passes_through l₁ p ∧ passes_through l₂ p :=
sorry

end regression_lines_intersection_l3789_378945


namespace weight_solution_l3789_378976

def weight_problem (A B C D E : ℝ) : Prop :=
  let avg_ABC := (A + B + C) / 3
  let avg_ABCD := (A + B + C + D) / 4
  let avg_BCDE := (B + C + D + E) / 4
  avg_ABC = 50 ∧ 
  avg_ABCD = 53 ∧ 
  E = D + 3 ∧ 
  avg_BCDE = 51 →
  A = 8

theorem weight_solution :
  ∀ A B C D E : ℝ, weight_problem A B C D E :=
by sorry

end weight_solution_l3789_378976


namespace equation_solution_l3789_378920

theorem equation_solution : 
  ∃ x : ℝ, 0.3 * x + (0.4 * 0.5) = 0.26 ∧ x = 0.2 := by
sorry

end equation_solution_l3789_378920


namespace quadratic_distinct_roots_condition_l3789_378961

/-- 
Given a quadratic equation (k-1)x^2 + 4x + 1 = 0, this theorem states that
for the equation to have two distinct real roots, k must be less than 5 and not equal to 1.
-/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k - 1) * x₁^2 + 4 * x₁ + 1 = 0 ∧ 
    (k - 1) * x₂^2 + 4 * x₂ + 1 = 0) ↔ 
  (k < 5 ∧ k ≠ 1) :=
sorry

end quadratic_distinct_roots_condition_l3789_378961


namespace child_workers_count_l3789_378949

/-- Represents the number of workers of each type and their daily wages --/
structure WorkforceData where
  male_workers : ℕ
  female_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the number of child workers given the workforce data --/
def calculate_child_workers (data : WorkforceData) : ℕ :=
  let total_workers := data.male_workers + data.female_workers
  let total_wage := data.male_workers * data.male_wage + data.female_workers * data.female_wage
  let x := (data.average_wage * total_workers - total_wage) / (data.average_wage - data.child_wage)
  x

/-- Theorem stating that the number of child workers is 5 given the specific workforce data --/
theorem child_workers_count (data : WorkforceData) 
  (h1 : data.male_workers = 20)
  (h2 : data.female_workers = 15)
  (h3 : data.male_wage = 35)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 26) :
  calculate_child_workers data = 5 := by
  sorry

end child_workers_count_l3789_378949


namespace only_one_and_two_satisfy_property_l3789_378907

/-- A function that checks if a number has n-1 digits of 1 and one digit of 7 -/
def has_n_minus_1_ones_and_one_seven (x : ℕ) (n : ℕ) : Prop := sorry

/-- A function that generates all numbers with n-1 digits of 1 and one digit of 7 -/
def numbers_with_n_minus_1_ones_and_one_seven (n : ℕ) : Set ℕ := sorry

theorem only_one_and_two_satisfy_property :
  ∀ n : ℕ, (∀ x ∈ numbers_with_n_minus_1_ones_and_one_seven n, Nat.Prime x) ↔ (n = 1 ∨ n = 2) :=
sorry

end only_one_and_two_satisfy_property_l3789_378907


namespace polar_to_cartesian_circle_l3789_378962

/-- The polar equation r = 1 / (sin θ + cos θ) represents a circle in Cartesian coordinates -/
theorem polar_to_cartesian_circle :
  ∃ (h k r : ℝ), ∀ (x y : ℝ),
    (∃ (θ : ℝ), x = (1 / (Real.sin θ + Real.cos θ)) * Real.cos θ ∧
                 y = (1 / (Real.sin θ + Real.cos θ)) * Real.sin θ) →
    (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end polar_to_cartesian_circle_l3789_378962


namespace gain_percent_problem_l3789_378991

/-- 
If the cost price of 50 articles is equal to the selling price of 25 articles, 
then the gain percent is 100%.
-/
theorem gain_percent_problem (C S : ℝ) (hpos : C > 0) : 
  50 * C = 25 * S → (S - C) / C * 100 = 100 :=
by
  sorry

#check gain_percent_problem

end gain_percent_problem_l3789_378991


namespace triangle_theorem_l3789_378964

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 + t.b^2 - t.c^2 = Real.sqrt 3 * t.a * t.b) 
  (h2 : 0 < t.A ∧ t.A ≤ 2 * Real.pi / 3) :
  t.C = Real.pi / 6 ∧ 
  let m := 2 * (Real.cos (t.A / 2))^2 - Real.sin t.B - 1
  ∀ x, m = x → -1 ≤ x ∧ x < 1/2 := by
sorry

end triangle_theorem_l3789_378964


namespace hidden_numbers_average_l3789_378965

/-- A card with two numbers -/
structure Card where
  visible : ℕ
  hidden : ℕ

/-- The problem setup -/
def problem_setup (cards : Fin 3 → Card) : Prop :=
  -- The sums of the numbers on each card are the same
  (∃ s : ℕ, ∀ i : Fin 3, (cards i).visible + (cards i).hidden = s) ∧
  -- Visible numbers are 81, 52, and 47
  (cards 0).visible = 81 ∧ (cards 1).visible = 52 ∧ (cards 2).visible = 47 ∧
  -- Hidden numbers are all prime
  (∀ i : Fin 3, Nat.Prime (cards i).hidden) ∧
  -- All numbers are different
  (∀ i j : Fin 3, i ≠ j → (cards i).visible ≠ (cards j).visible ∧ 
                         (cards i).hidden ≠ (cards j).hidden ∧
                         (cards i).visible ≠ (cards j).hidden)

/-- The theorem to prove -/
theorem hidden_numbers_average (cards : Fin 3 → Card) 
  (h : problem_setup cards) : 
  (cards 0).hidden + (cards 1).hidden + (cards 2).hidden = 119 := by
  sorry

#check hidden_numbers_average

end hidden_numbers_average_l3789_378965


namespace real_y_condition_l3789_378908

theorem real_y_condition (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 6 * x * y + 2 * x + 2 = 0) ↔ 
  (x ≤ (1 - Real.sqrt 7) / 3 ∨ x ≥ (1 + Real.sqrt 7) / 3) :=
by sorry

end real_y_condition_l3789_378908


namespace student_average_age_l3789_378971

theorem student_average_age (n : ℕ) (teacher_age : ℕ) (new_average : ℝ) :
  n = 50 ∧ teacher_age = 65 ∧ new_average = 15 →
  (n : ℝ) * (((n : ℝ) * new_average - teacher_age) / n) + teacher_age = (n + 1 : ℝ) * new_average →
  ((n : ℝ) * new_average - teacher_age) / n = 14 := by
  sorry

end student_average_age_l3789_378971


namespace no_integer_solution_l3789_378933

theorem no_integer_solution : ¬ ∃ (a b c d : ℤ),
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
  sorry

end no_integer_solution_l3789_378933


namespace sphere_wedge_volume_l3789_378986

/-- Given a sphere with circumference 18π inches, cut into 8 congruent wedges,
    the volume of one wedge is 121.5π cubic inches. -/
theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) :
  circumference = 18 * Real.pi →
  num_wedges = 8 →
  (1 / num_wedges : ℝ) * (4 / 3 * Real.pi * (circumference / (2 * Real.pi))^3) = 121.5 * Real.pi :=
by sorry

end sphere_wedge_volume_l3789_378986


namespace inequality_solution_range_l3789_378983

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 2| + |x + 2| ≤ a^2 - 3*a) ↔ (a ≥ 4 ∨ a ≤ -1) := by
  sorry

end inequality_solution_range_l3789_378983


namespace range_of_a_for_p_range_of_a_for_p_or_q_and_not_p_and_q_l3789_378938

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x ≥ 1 ∧ 4^x + 2^(x+1) - 7 - a < 0

-- Theorem for part 1
theorem range_of_a_for_p :
  {a : ℝ | p a} = {a : ℝ | 0 ≤ a ∧ a < 4} :=
sorry

-- Theorem for part 2
theorem range_of_a_for_p_or_q_and_not_p_and_q :
  {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)} = {a : ℝ | (0 ≤ a ∧ a ≤ 1) ∨ a ≥ 4} :=
sorry

end range_of_a_for_p_range_of_a_for_p_or_q_and_not_p_and_q_l3789_378938


namespace arithmetic_progression_equality_l3789_378956

theorem arithmetic_progression_equality (n : ℕ) 
  (a b : Fin n → ℕ+) 
  (h_n : n ≥ 2018)
  (h_distinct : ∀ i j : Fin n, i ≠ j → (a i ≠ a j ∧ b i ≠ b j))
  (h_bound : ∀ i : Fin n, a i ≤ 5*n ∧ b i ≤ 5*n)
  (h_arithmetic : ∃ d : ℚ, ∀ i j : Fin n, (a j : ℚ) / (b j : ℚ) - (a i : ℚ) / (b i : ℚ) = (j : ℚ) - (i : ℚ) * d) :
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) := by
sorry

end arithmetic_progression_equality_l3789_378956


namespace calculate_expression_l3789_378963

theorem calculate_expression : 24 / (-6) * (3/2) / (-4/3) = 9/2 := by
  sorry

end calculate_expression_l3789_378963


namespace remainder_theorem_l3789_378970

-- Define the polynomial q(x)
def q (A B C x : ℝ) : ℝ := A * x^5 - B * x^3 + C * x - 2

-- Theorem statement
theorem remainder_theorem (A B C : ℝ) :
  (q A B C 2 = -6) → (q A B C (-2) = 2) := by
  sorry

end remainder_theorem_l3789_378970


namespace sector_angle_when_arc_equals_radius_l3789_378994

theorem sector_angle_when_arc_equals_radius (r : ℝ) (θ : ℝ) :
  r > 0 → r * θ = r → θ = 1 := by sorry

end sector_angle_when_arc_equals_radius_l3789_378994


namespace dc_length_l3789_378972

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def conditions (q : Quadrilateral) : Prop :=
  let dist := λ p₁ p₂ : ℝ × ℝ => Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  let sinAngle := λ p₁ p₂ p₃ : ℝ × ℝ => 
    let v1 := (p₂.1 - p₁.1, p₂.2 - p₁.2)
    let v2 := (p₃.1 - p₁.1, p₃.2 - p₁.2)
    (v1.1 * v2.2 - v1.2 * v2.1) / (dist p₁ p₂ * dist p₁ p₃)
  dist q.A q.B = 30 ∧
  (q.A.1 - q.D.1) * (q.B.1 - q.D.1) + (q.A.2 - q.D.2) * (q.B.2 - q.D.2) = 0 ∧
  sinAngle q.B q.A q.D = 4/5 ∧
  sinAngle q.B q.C q.D = 1/5

-- State the theorem
theorem dc_length (q : Quadrilateral) (h : conditions q) : 
  dist q.D q.C = 48 * Real.sqrt 6 := by
  sorry

end dc_length_l3789_378972


namespace sum_not_arithmetic_l3789_378936

/-- An infinite arithmetic progression -/
def arithmetic_progression (a d : ℝ) : ℕ → ℝ := λ n => a + (n - 1) * d

/-- An infinite geometric progression -/
def geometric_progression (b q : ℝ) : ℕ → ℝ := λ n => b * q^(n - 1)

/-- The sum of an arithmetic and a geometric progression -/
def sum_progression (a d b q : ℝ) : ℕ → ℝ :=
  λ n => arithmetic_progression a d n + geometric_progression b q n

theorem sum_not_arithmetic (a d b q : ℝ) (hq : q ≠ 1) :
  ¬ ∃ (A D : ℝ), ∀ n : ℕ, sum_progression a d b q n = A + (n - 1) * D :=
sorry

end sum_not_arithmetic_l3789_378936


namespace leo_current_weight_l3789_378950

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 80

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 140 - leo_weight

/-- The combined weight of Leo and Kendra in pounds -/
def combined_weight : ℝ := 140

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = combined_weight) ∧
  (leo_weight = 80) :=
by sorry

end leo_current_weight_l3789_378950


namespace earth_inhabitable_fraction_l3789_378928

theorem earth_inhabitable_fraction :
  let water_fraction : ℚ := 2/3
  let land_fraction : ℚ := 1 - water_fraction
  let inhabitable_land_fraction : ℚ := 1/3
  (1 - water_fraction) * inhabitable_land_fraction = 1/9 :=
by
  sorry

end earth_inhabitable_fraction_l3789_378928


namespace elizabeth_revenue_is_900_l3789_378901

/-- Represents the revenue and investment data for Mr. Banks and Ms. Elizabeth -/
structure InvestmentData where
  banks_investments : ℕ
  banks_revenue_per_investment : ℕ
  elizabeth_investments : ℕ
  elizabeth_total_revenue_difference : ℕ

/-- Calculates Ms. Elizabeth's revenue per investment given the investment data -/
def elizabeth_revenue_per_investment (data : InvestmentData) : ℕ :=
  (data.banks_investments * data.banks_revenue_per_investment + data.elizabeth_total_revenue_difference) / data.elizabeth_investments

/-- Theorem stating that Ms. Elizabeth's revenue per investment is $900 given the problem conditions -/
theorem elizabeth_revenue_is_900 (data : InvestmentData)
  (h1 : data.banks_investments = 8)
  (h2 : data.banks_revenue_per_investment = 500)
  (h3 : data.elizabeth_investments = 5)
  (h4 : data.elizabeth_total_revenue_difference = 500) :
  elizabeth_revenue_per_investment data = 900 := by
  sorry

end elizabeth_revenue_is_900_l3789_378901


namespace equation_solutions_l3789_378982

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1^2 = 2 ∧ x2^2 = 2 ∧ x1 = Real.sqrt 2 ∧ x2 = -Real.sqrt 2) ∧
  (∃ x1 x2 : ℝ, 4*x1^2 - 1 = 0 ∧ 4*x2^2 - 1 = 0 ∧ x1 = 1/2 ∧ x2 = -1/2) ∧
  (∃ x1 x2 : ℝ, (x1-1)^2 - 4 = 0 ∧ (x2-1)^2 - 4 = 0 ∧ x1 = 3 ∧ x2 = -1) ∧
  (∃ x1 x2 : ℝ, 12*(3-x1)^2 - 48 = 0 ∧ 12*(3-x2)^2 - 48 = 0 ∧ x1 = 1 ∧ x2 = 5) := by
  sorry

end equation_solutions_l3789_378982


namespace harmonic_sum_number_bounds_harmonic_number_digit_sum_even_l3789_378987

/-- Represents a three-digit natural number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Checks if a number is a sum number -/
def is_sum_number (n : ThreeDigitNumber) : Prop :=
  n.hundreds = n.tens + n.units

/-- Checks if a number is a harmonic number -/
def is_harmonic_number (n : ThreeDigitNumber) : Prop :=
  n.hundreds = n.tens^2 - n.units^2

/-- Checks if a number is a harmonic sum number -/
def is_harmonic_sum_number (n : ThreeDigitNumber) : Prop :=
  is_sum_number n ∧ is_harmonic_number n

/-- Converts a ThreeDigitNumber to its numeric value -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

theorem harmonic_sum_number_bounds (n : ThreeDigitNumber) :
  is_harmonic_sum_number n → 110 ≤ to_nat n ∧ to_nat n ≤ 954 := by
  sorry

theorem harmonic_number_digit_sum_even (n : ThreeDigitNumber) :
  is_harmonic_number n → Even (n.hundreds + n.tens + n.units) := by
  sorry

end harmonic_sum_number_bounds_harmonic_number_digit_sum_even_l3789_378987


namespace diamond_two_three_l3789_378973

-- Define the diamond operation
def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

-- Theorem statement
theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end diamond_two_three_l3789_378973


namespace suzannes_book_pages_l3789_378904

theorem suzannes_book_pages : 
  ∀ (pages_monday pages_tuesday pages_left : ℕ),
    pages_monday = 15 →
    pages_tuesday = pages_monday + 16 →
    pages_left = 18 →
    pages_monday + pages_tuesday + pages_left = 64 :=
by
  sorry

end suzannes_book_pages_l3789_378904


namespace quadrilateral_area_l3789_378984

/-- The area of a quadrilateral ABCD with given diagonal and offsets -/
theorem quadrilateral_area (BD AC : ℝ) (offset_A offset_C : ℝ) :
  BD = 28 →
  offset_A = 8 →
  offset_C = 2 →
  (1/2 * BD * offset_A) + (1/2 * BD * offset_C) = 140 :=
by sorry

end quadrilateral_area_l3789_378984


namespace triple_angle_bracket_ten_l3789_378960

def divisor_sum (n : ℕ) : ℕ :=
  sorry

def angle_bracket (n : ℕ) : ℕ :=
  sorry

theorem triple_angle_bracket_ten : angle_bracket (angle_bracket (angle_bracket 10)) = 0 := by
  sorry

end triple_angle_bracket_ten_l3789_378960


namespace pigeon_problem_solution_l3789_378903

/-- Represents the number of pigeons on the branches and under the tree -/
structure PigeonCount where
  onBranches : ℕ
  underTree : ℕ

/-- The conditions of the pigeon problem -/
def satisfiesPigeonConditions (p : PigeonCount) : Prop :=
  (p.underTree - 1 = (p.onBranches + 1) / 3) ∧
  (p.onBranches - 1 = p.underTree + 1)

/-- The theorem stating the solution to the pigeon problem -/
theorem pigeon_problem_solution :
  ∃ (p : PigeonCount), satisfiesPigeonConditions p ∧ p.onBranches = 7 ∧ p.underTree = 5 := by
  sorry


end pigeon_problem_solution_l3789_378903


namespace rajas_income_l3789_378992

theorem rajas_income (household_percent : ℝ) (clothes_percent : ℝ) (medicines_percent : ℝ)
  (transportation_percent : ℝ) (entertainment_percent : ℝ) (savings : ℝ) (income : ℝ) :
  household_percent = 0.45 →
  clothes_percent = 0.12 →
  medicines_percent = 0.08 →
  transportation_percent = 0.15 →
  entertainment_percent = 0.10 →
  savings = 5000 →
  household_percent * income + clothes_percent * income + medicines_percent * income +
    transportation_percent * income + entertainment_percent * income + savings = income →
  income = 50000 := by
  sorry

end rajas_income_l3789_378992


namespace intersection_P_Q_l3789_378943

-- Define set P
def P : Set ℝ := {x | x * (x - 3) < 0}

-- Define set Q
def Q : Set ℝ := {x | |x| < 2}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = Set.Ioo 0 2 := by
  sorry

end intersection_P_Q_l3789_378943


namespace sqrt_equation_solution_l3789_378966

theorem sqrt_equation_solution (x : ℝ) (h : x > 6) :
  Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3 ↔ x ≥ 18 := by
  sorry

end sqrt_equation_solution_l3789_378966


namespace linear_equation_exponents_l3789_378989

/-- A function to represent the linearity of an equation in two variables -/
def is_linear_two_var (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (m n c : ℝ), ∀ x y, f x y = m * x + n * y + c

/-- The main theorem -/
theorem linear_equation_exponents :
  ∀ a b : ℝ,
  (is_linear_two_var (λ x y => x^(a-3) + y^(b-1))) →
  (a = 4 ∧ b = 2) :=
by sorry

end linear_equation_exponents_l3789_378989


namespace fib_100_div_5_l3789_378913

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The Fibonacci sequence modulo 5 repeats every 20 terms -/
axiom fib_mod_5_period : ∀ n : ℕ, fib (n + 20) % 5 = fib n % 5

theorem fib_100_div_5 : 5 ∣ fib 100 := by
  sorry

end fib_100_div_5_l3789_378913


namespace f_2023_of_5_eq_57_l3789_378978

def f (x : ℚ) : ℚ := (2 + x) / (1 - 2 * x)

def f_n : ℕ → ℚ → ℚ
  | 0, x => x
  | n + 1, x => f (f_n n x)

theorem f_2023_of_5_eq_57 : f_n 2023 5 = 57 := by
  sorry

end f_2023_of_5_eq_57_l3789_378978


namespace a_minus_b_pow_2014_l3789_378939

theorem a_minus_b_pow_2014 (a b : ℝ) 
  (ha : a^3 - 6*a^2 + 15*a = 9) 
  (hb : b^3 - 3*b^2 + 6*b = -1) : 
  (a - b)^2014 = 1 := by sorry

end a_minus_b_pow_2014_l3789_378939


namespace solve_equation_l3789_378930

theorem solve_equation (x : ℤ) (h : 9873 + x = 13200) : x = 3327 := by
  sorry

end solve_equation_l3789_378930


namespace removed_to_total_ratio_is_one_to_two_l3789_378902

/-- Represents the number of bricks in a course -/
def bricks_per_course : ℕ := 400

/-- Represents the initial number of courses -/
def initial_courses : ℕ := 3

/-- Represents the number of courses added -/
def added_courses : ℕ := 2

/-- Represents the total number of bricks after removal -/
def total_bricks_after_removal : ℕ := 1800

/-- Theorem stating that the ratio of removed bricks to total bricks in the last course is 1:2 -/
theorem removed_to_total_ratio_is_one_to_two :
  let total_courses := initial_courses + added_courses
  let expected_total_bricks := total_courses * bricks_per_course
  let removed_bricks := expected_total_bricks - total_bricks_after_removal
  let last_course_bricks := bricks_per_course
  (removed_bricks : ℚ) / (last_course_bricks : ℚ) = 1 / 2 :=
by sorry

end removed_to_total_ratio_is_one_to_two_l3789_378902


namespace correct_calculation_l3789_378940

theorem correct_calculation (x : ℝ) : (x / 7 = 49) → (x * 6 = 2058) := by
  sorry

end correct_calculation_l3789_378940


namespace probability_ratio_l3789_378942

/-- The number of balls -/
def num_balls : ℕ := 24

/-- The number of bins -/
def num_bins : ℕ := 6

/-- The probability of the first distribution (6-6-3-3-3-3) -/
noncomputable def p : ℝ := sorry

/-- The probability of the second distribution (4-4-4-4-4-4) -/
noncomputable def q : ℝ := sorry

/-- Theorem stating that the ratio of probabilities p and q is 12 -/
theorem probability_ratio : p / q = 12 := by sorry

end probability_ratio_l3789_378942


namespace stratified_sample_theorem_l3789_378937

/-- Represents a stratified sampling scenario in a high school -/
structure StratifiedSample where
  total_students : ℕ
  liberal_arts_students : ℕ
  sample_size : ℕ

/-- Calculates the expected number of liberal arts students in the sample -/
def expected_liberal_arts_in_sample (s : StratifiedSample) : ℕ :=
  (s.liberal_arts_students * s.sample_size) / s.total_students

/-- Theorem stating the expected number of liberal arts students in the sample -/
theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_students = 1000)
  (h2 : s.liberal_arts_students = 200)
  (h3 : s.sample_size = 100) :
  expected_liberal_arts_in_sample s = 20 := by
  sorry

#eval expected_liberal_arts_in_sample { total_students := 1000, liberal_arts_students := 200, sample_size := 100 }

end stratified_sample_theorem_l3789_378937


namespace final_bacteria_count_l3789_378922

def initial_count : ℕ := 30
def start_time : ℕ := 0  -- 10:00 AM represented as 0 minutes
def end_time : ℕ := 30   -- 10:30 AM represented as 30 minutes
def growth_interval : ℕ := 5  -- population triples every 5 minutes
def death_interval : ℕ := 15  -- 10% die every 15 minutes

def growth_factor : ℚ := 3
def survival_rate : ℚ := 0.9  -- 90% survival rate (10% die)

def number_of_growth_periods (t : ℕ) : ℕ := t / growth_interval

def number_of_death_periods (t : ℕ) : ℕ := t / death_interval

def bacteria_count (t : ℕ) : ℚ :=
  initial_count *
  growth_factor ^ (number_of_growth_periods t) *
  survival_rate ^ (number_of_death_periods t)

theorem final_bacteria_count :
  bacteria_count end_time = 17694 := by sorry

end final_bacteria_count_l3789_378922


namespace fish_tank_water_calculation_l3789_378977

theorem fish_tank_water_calculation (initial_water : ℝ) (added_water : ℝ) : 
  initial_water = 7.75 → added_water = 7 → initial_water + added_water = 14.75 := by
  sorry

end fish_tank_water_calculation_l3789_378977


namespace childrens_book_weight_l3789_378934

-- Define the weight of a comic book
def comic_book_weight : ℝ := 0.8

-- Define the total weight of all books
def total_weight : ℝ := 10.98

-- Define the number of comic books
def num_comic_books : ℕ := 9

-- Define the number of children's books
def num_children_books : ℕ := 7

-- Theorem to prove
theorem childrens_book_weight :
  (total_weight - (num_comic_books : ℝ) * comic_book_weight) / num_children_books = 0.54 := by
  sorry

end childrens_book_weight_l3789_378934


namespace biggest_number_l3789_378910

theorem biggest_number (jungkook yoongi yuna : ℚ) : 
  jungkook = 6 / 3 → yoongi = 4 → yuna = 5 → 
  max (max jungkook yoongi) yuna = 5 := by
sorry

end biggest_number_l3789_378910


namespace ones_digit_of_11_to_46_l3789_378995

theorem ones_digit_of_11_to_46 : (11^46 : ℕ) % 10 = 1 := by
  sorry

end ones_digit_of_11_to_46_l3789_378995


namespace bianca_books_total_l3789_378932

theorem bianca_books_total (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 8)
  (h2 : mystery_shelves = 5)
  (h3 : picture_shelves = 4) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 72 :=
by sorry

end bianca_books_total_l3789_378932


namespace expression_evaluation_l3789_378958

theorem expression_evaluation :
  let a : ℤ := -2
  3 * a * (2 * a^2 - 4 * a + 3) - 2 * a^2 * (3 * a + 4) = -98 := by
  sorry

end expression_evaluation_l3789_378958


namespace min_value_ab_minus_cd_l3789_378911

theorem min_value_ab_minus_cd (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 9)
  (h5 : a^2 + b^2 + c^2 + d^2 = 21) :
  a * b - c * d ≥ 2 := by
  sorry

end min_value_ab_minus_cd_l3789_378911


namespace sophia_age_in_eight_years_l3789_378993

/-- Represents the ages of individuals in the problem -/
structure Ages where
  jeremy : ℕ
  sebastian : ℕ
  isabella : ℕ
  sophia : ℕ
  lucas : ℕ
  olivia : ℕ
  ethan : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.jeremy + ages.sebastian + ages.isabella + ages.sophia + ages.lucas + ages.olivia + ages.ethan + 42 = 495) ∧
  (ages.sebastian = ages.jeremy + 4) ∧
  (ages.isabella = ages.sebastian - 3) ∧
  (ages.sophia = 2 * ages.lucas) ∧
  (ages.lucas = ages.jeremy - 5) ∧
  (ages.olivia = ages.isabella) ∧
  (ages.ethan = ages.olivia / 2) ∧
  (ages.jeremy + ages.sebastian + ages.isabella + 6 = 150) ∧
  (ages.jeremy = 40)

/-- The theorem to be proved -/
theorem sophia_age_in_eight_years (ages : Ages) :
  problem_conditions ages → ages.sophia + 8 = 78 := by
  sorry


end sophia_age_in_eight_years_l3789_378993


namespace parabola_directrix_equation_l3789_378941

/-- The equation of the directrix of a parabola passing through a point on a circle -/
theorem parabola_directrix_equation (y : ℝ) (p : ℝ) : 
  (1^2 - 4*1 + y^2 = 0) →  -- Point P(1, y) lies on the circle
  (p > 0) →                -- p is positive
  (1^2 = -2*p*y) →         -- Parabola passes through P(1, y)
  (-(p : ℝ) = Real.sqrt 3 / 12) := by
  sorry

end parabola_directrix_equation_l3789_378941


namespace strudel_price_calculation_l3789_378914

/-- Calculates the final price of a strudel after two 50% increases and a 50% decrease -/
def finalPrice (initialPrice : ℝ) : ℝ :=
  initialPrice * 1.5 * 1.5 * 0.5

/-- Theorem stating that the final price of a strudel is 90 rubles -/
theorem strudel_price_calculation :
  finalPrice 80 = 90 := by
  sorry

end strudel_price_calculation_l3789_378914


namespace smallest_base_for_100_l3789_378927

theorem smallest_base_for_100 : 
  ∃ (b : ℕ), b = 5 ∧ 
  (∀ (x : ℕ), x < b → ¬(x^2 ≤ 100 ∧ 100 < x^3)) ∧
  (5^2 ≤ 100 ∧ 100 < 5^3) := by
  sorry

end smallest_base_for_100_l3789_378927
