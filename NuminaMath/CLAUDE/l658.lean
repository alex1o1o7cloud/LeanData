import Mathlib

namespace technician_salary_l658_65805

/-- The average salary of technicians in a workshop --/
theorem technician_salary (total_workers : ℝ) (total_avg_salary : ℝ) 
  (num_technicians : ℝ) (non_tech_avg_salary : ℝ) :
  total_workers = 21.11111111111111 →
  total_avg_salary = 1000 →
  num_technicians = 10 →
  non_tech_avg_salary = 820 →
  (total_workers * total_avg_salary - (total_workers - num_technicians) * non_tech_avg_salary) 
    / num_technicians = 1200 := by
  sorry

end technician_salary_l658_65805


namespace slope_of_line_l658_65859

/-- The slope of a line given by the equation 4y = 5x - 20 is 5/4 -/
theorem slope_of_line (x y : ℝ) : 4 * y = 5 * x - 20 → (∃ b : ℝ, y = (5/4) * x + b) := by
  sorry

end slope_of_line_l658_65859


namespace power_inequality_l658_65802

theorem power_inequality (x y a : ℝ) (hx : 0 < x) (hy : x < y) (hy1 : y < 1) (ha : 0 < a) (ha1 : a < 1) :
  x^a < y^a := by
  sorry

end power_inequality_l658_65802


namespace compare_function_values_l658_65849

/-- Given a quadratic function f(x) = x^2 - bx + c with specific properties,
    prove that f(b^x) ≤ f(c^x) for all real x. -/
theorem compare_function_values (b c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 - b*x + c) 
    (h2 : ∀ x, f (1 - x) = f (1 + x)) (h3 : f 0 = 3) : 
    ∀ x, f (b^x) ≤ f (c^x) := by
  sorry

end compare_function_values_l658_65849


namespace ratio_xy_system_l658_65893

theorem ratio_xy_system (x y t : ℝ) 
  (eq1 : 2 * x + 5 * y = 6 * t) 
  (eq2 : 3 * x - y = t) : 
  x / y = 11 / 16 := by
sorry

end ratio_xy_system_l658_65893


namespace book_price_reduction_l658_65816

theorem book_price_reduction (original_price : ℝ) : 
  original_price = 20 → 
  (original_price * (1 - 0.25) * (1 - 0.40) = 9) := by
  sorry

end book_price_reduction_l658_65816


namespace smallest_of_five_consecutive_odds_with_product_l658_65876

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def consecutive_odds (a b c d e : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ is_odd c ∧ is_odd d ∧ is_odd e ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2 ∧ e = d + 2

theorem smallest_of_five_consecutive_odds_with_product (a b c d e : ℕ) :
  consecutive_odds a b c d e →
  a * b * c * d * e = 135135 →
  a = 7 :=
sorry

end smallest_of_five_consecutive_odds_with_product_l658_65876


namespace bailey_chew_toys_l658_65854

theorem bailey_chew_toys (dog_treats rawhide_bones credit_cards items_per_charge : ℕ) 
  (h1 : dog_treats = 8)
  (h2 : rawhide_bones = 10)
  (h3 : credit_cards = 4)
  (h4 : items_per_charge = 5) :
  credit_cards * items_per_charge - (dog_treats + rawhide_bones) = 2 := by
  sorry

end bailey_chew_toys_l658_65854


namespace tip_percentage_is_twenty_percent_l658_65838

def appetizer_cost : ℚ := 8
def entree_cost : ℚ := 20
def wine_cost : ℚ := 3
def dessert_cost : ℚ := 6
def discount_ratio : ℚ := (1/2)
def total_spent : ℚ := 38

def full_cost : ℚ := appetizer_cost + entree_cost + 2 * wine_cost + dessert_cost
def discounted_cost : ℚ := appetizer_cost + entree_cost * (1 - discount_ratio) + 2 * wine_cost + dessert_cost
def tip_amount : ℚ := total_spent - discounted_cost

theorem tip_percentage_is_twenty_percent :
  (tip_amount / full_cost) * 100 = 20 := by sorry

end tip_percentage_is_twenty_percent_l658_65838


namespace period_of_cos_3x_l658_65861

theorem period_of_cos_3x :
  let f : ℝ → ℝ := λ x => Real.cos (3 * x)
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, 0 < q ∧ q < p → ∃ x : ℝ, f (x + q) ≠ f x :=
by
  sorry

end period_of_cos_3x_l658_65861


namespace catch_fraction_l658_65847

theorem catch_fraction (joe_catches derek_catches tammy_catches : ℕ) : 
  joe_catches = 23 →
  derek_catches = 2 * joe_catches - 4 →
  tammy_catches = 30 →
  (tammy_catches : ℚ) / derek_catches = 5 / 7 := by
sorry

end catch_fraction_l658_65847


namespace telescope_visual_range_increase_l658_65878

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 90)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 200 / 3 := by
  sorry

end telescope_visual_range_increase_l658_65878


namespace f_is_odd_f_def_nonneg_f_neg_one_eq_neg_two_l658_65806

/-- An odd function f defined on ℝ with f(x) = x(1+x) for x ≥ 0 -/
def f : ℝ → ℝ :=
  sorry

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
  sorry

theorem f_def_nonneg : ∀ x : ℝ, x ≥ 0 → f x = x * (1 + x) :=
  sorry

theorem f_neg_one_eq_neg_two : f (-1) = -2 :=
  sorry

end f_is_odd_f_def_nonneg_f_neg_one_eq_neg_two_l658_65806


namespace ratio_of_part_to_whole_l658_65844

theorem ratio_of_part_to_whole (N : ℝ) : 
  (1 / 1) * (1 / 3) * (2 / 5) * N = 10 →
  (40 / 100) * N = 120 →
  (10 : ℝ) / ((1 / 3) * (2 / 5) * N) = 1 / 4 := by
  sorry

end ratio_of_part_to_whole_l658_65844


namespace mean_median_difference_is_03_l658_65856

/-- Represents a frequency histogram bin -/
structure HistogramBin where
  lowerBound : ℝ
  upperBound : ℝ
  frequency : ℕ

/-- Calculates the median of a dataset represented by a frequency histogram -/
def calculateMedian (histogram : List HistogramBin) (totalStudents : ℕ) : ℝ :=
  sorry

/-- Calculates the mean of a dataset represented by a frequency histogram -/
def calculateMean (histogram : List HistogramBin) (totalStudents : ℕ) : ℝ :=
  sorry

theorem mean_median_difference_is_03 (histogram : List HistogramBin) : 
  let totalStudents := 20
  let h := [
    ⟨0, 1, 4⟩,
    ⟨2, 3, 2⟩,
    ⟨4, 5, 6⟩,
    ⟨6, 7, 3⟩,
    ⟨8, 9, 5⟩
  ]
  calculateMean h totalStudents - calculateMedian h totalStudents = 0.3 := by
  sorry

end mean_median_difference_is_03_l658_65856


namespace digit_difference_1250_l658_65855

/-- The number of digits in the base-b representation of a positive integer n -/
def num_digits (n : ℕ) (b : ℕ) : ℕ :=
  Nat.log b n + 1

/-- The theorem stating the difference in number of digits between base-4 and base-9 representations of 1250 -/
theorem digit_difference_1250 :
  num_digits 1250 4 - num_digits 1250 9 = 2 := by
  sorry

end digit_difference_1250_l658_65855


namespace count_numbers_with_at_least_two_zeros_l658_65897

/-- The number of digits in the numbers we're considering -/
def n : ℕ := 6

/-- The total number of n-digit numbers -/
def total_n_digit_numbers : ℕ := 9 * 10^(n-1)

/-- The number of n-digit numbers with no zeros -/
def numbers_with_no_zeros : ℕ := 9^n

/-- The number of n-digit numbers with exactly one zero -/
def numbers_with_one_zero : ℕ := n * 9^(n-1)

/-- The number of n-digit numbers with at least two zeros -/
def numbers_with_at_least_two_zeros : ℕ :=
  total_n_digit_numbers - numbers_with_no_zeros - numbers_with_one_zero

theorem count_numbers_with_at_least_two_zeros :
  numbers_with_at_least_two_zeros = 14265 := by
  sorry

end count_numbers_with_at_least_two_zeros_l658_65897


namespace vertical_line_equation_l658_65812

/-- A line passing through the point (-2,1) with an undefined slope (vertical line) has the equation x + 2 = 0 -/
theorem vertical_line_equation : 
  ∀ (l : Set (ℝ × ℝ)), 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x = -2) → 
  (-2, 1) ∈ l → 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x + 2 = 0) :=
by sorry

end vertical_line_equation_l658_65812


namespace band_formation_proof_l658_65833

/-- Represents the number of columns in the rectangular formation -/
def n : ℕ := 14

/-- The total number of band members -/
def total_members : ℕ := n * (n + 7)

/-- The side length of the square formation -/
def square_side : ℕ := 17

theorem band_formation_proof :
  -- Square formation condition
  total_members = square_side ^ 2 + 5 ∧
  -- Rectangular formation condition
  total_members = n * (n + 7) ∧
  -- Maximum number of members
  total_members = 294 ∧
  -- No larger n satisfies the conditions
  ∀ m : ℕ, m > n → ¬(∃ k : ℕ, m * (m + 7) = k ^ 2 + 5) :=
by sorry

end band_formation_proof_l658_65833


namespace square_sum_given_conditions_l658_65879

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end square_sum_given_conditions_l658_65879


namespace anya_lost_games_l658_65836

/-- Represents a girl in the table tennis game -/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- Represents the number of games played by each girl -/
def games_played (g : Girl) : ℕ :=
  match g with
  | .Anya => 4
  | .Bella => 6
  | .Valya => 7
  | .Galya => 10
  | .Dasha => 11

/-- The total number of games played -/
def total_games : ℕ := 19

/-- Theorem stating that Anya lost in specific games -/
theorem anya_lost_games :
  ∃ (lost_games : List ℕ),
    lost_games = [4, 8, 12, 16] ∧
    (∀ g ∈ lost_games, g ≤ total_games) ∧
    (∀ g ∈ lost_games, ∃ i, g = 4 * i) ∧
    lost_games.length = games_played Girl.Anya := by
  sorry

end anya_lost_games_l658_65836


namespace regular_polygon_with_45_degree_exterior_angle_has_8_sides_l658_65889

/-- A regular polygon with an exterior angle of 45° has 8 sides. -/
theorem regular_polygon_with_45_degree_exterior_angle_has_8_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 0 →
    exterior_angle = 45 →
    (360 : ℝ) / exterior_angle = n →
    n = 8 := by
  sorry

end regular_polygon_with_45_degree_exterior_angle_has_8_sides_l658_65889


namespace joe_rounding_threshold_l658_65868

/-- A grade is a nonnegative rational number -/
def Grade := { x : ℚ // 0 ≤ x }

/-- Joe's rounding function -/
noncomputable def joeRound (x : Grade) : ℕ :=
  sorry

/-- The smallest rational number M such that any grade x ≥ M gets rounded to at least 90 -/
def M : ℚ := 805 / 9

theorem joe_rounding_threshold :
  ∀ (x : Grade), joeRound x ≥ 90 ↔ x.val ≥ M :=
sorry

end joe_rounding_threshold_l658_65868


namespace solve_for_a_l658_65891

theorem solve_for_a (a b : ℝ) (h1 : b/a = 4) (h2 : b = 24 - 4*a) : a = 3 := by
  sorry

end solve_for_a_l658_65891


namespace base7_divisible_by_19_unique_x_divisible_by_19_x_is_4_l658_65828

/-- Converts a base 7 number of the form 52x4 to decimal --/
def base7ToDecimal (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

/-- The digit x in 52x4₇ makes the number divisible by 19 --/
theorem base7_divisible_by_19 :
  ∃ x : ℕ, x < 7 ∧ isDivisibleBy19 (base7ToDecimal x) :=
sorry

/-- The digit x in 52x4₇ that makes the number divisible by 19 is unique --/
theorem unique_x_divisible_by_19 :
  ∃! x : ℕ, x < 7 ∧ isDivisibleBy19 (base7ToDecimal x) :=
sorry

/-- The digit x in 52x4₇ that makes the number divisible by 19 is 4 --/
theorem x_is_4 :
  ∃ x : ℕ, x = 4 ∧ x < 7 ∧ isDivisibleBy19 (base7ToDecimal x) :=
sorry

end base7_divisible_by_19_unique_x_divisible_by_19_x_is_4_l658_65828


namespace fraction_equality_implies_values_l658_65822

theorem fraction_equality_implies_values (A B : ℚ) :
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ 5 ∧ x^2 - 7*x + 10 ≠ 0 →
    (B*x - 7) / (x^2 - 7*x + 10) = A / (x - 2) + 5 / (x - 5)) →
  A = -3/5 ∧ B = 22/5 ∧ A + B = 19/5 := by
sorry

end fraction_equality_implies_values_l658_65822


namespace missy_watch_time_l658_65898

/-- The total time Missy spends watching TV, given the number of reality shows,
    the duration of each reality show, and the duration of the cartoon. -/
def total_watch_time (num_reality_shows : ℕ) (reality_show_duration : ℕ) (cartoon_duration : ℕ) : ℕ :=
  num_reality_shows * reality_show_duration + cartoon_duration

/-- Theorem stating that Missy spends 150 minutes watching TV. -/
theorem missy_watch_time :
  total_watch_time 5 28 10 = 150 := by
  sorry

end missy_watch_time_l658_65898


namespace perpendicular_vectors_k_value_l658_65852

/-- Given two vectors a and b in ℝ², where a = (1,2) and b = (2,k),
    if 2a + b is perpendicular to a, then k = -6. -/
theorem perpendicular_vectors_k_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, k)
  (2 • a + b) • a = 0 → k = -6 :=
by sorry

end perpendicular_vectors_k_value_l658_65852


namespace sum_and_interval_l658_65896

theorem sum_and_interval : 
  let sum := 3 + 1/6 + 4 + 3/8 + 6 + 1/12
  sum = 13.625 ∧ 13.5 < sum ∧ sum < 14 := by
  sorry

end sum_and_interval_l658_65896


namespace largest_package_size_l658_65823

theorem largest_package_size (ming_markers : ℕ) (catherine_markers : ℕ)
  (h1 : ming_markers = 72)
  (h2 : catherine_markers = 48) :
  ∃ (package_size : ℕ),
    package_size ∣ ming_markers ∧
    package_size ∣ catherine_markers ∧
    ∀ (n : ℕ), n ∣ ming_markers → n ∣ catherine_markers → n ≤ package_size :=
by
  sorry

end largest_package_size_l658_65823


namespace grocery_shopping_problem_l658_65815

theorem grocery_shopping_problem (initial_budget : ℚ) (bread_cost : ℚ) (candy_cost : ℚ) 
  (h1 : initial_budget = 32)
  (h2 : bread_cost = 3)
  (h3 : candy_cost = 2) : 
  let remaining_after_bread_candy := initial_budget - (bread_cost + candy_cost)
  let turkey_cost := (1 / 3) * remaining_after_bread_candy
  initial_budget - (bread_cost + candy_cost + turkey_cost) = 18 := by
sorry

end grocery_shopping_problem_l658_65815


namespace systematic_sampling_questionnaire_B_l658_65814

/-- Systematic sampling problem -/
theorem systematic_sampling_questionnaire_B 
  (total_pool : ℕ) 
  (sample_size : ℕ) 
  (first_selected : ℕ) 
  (questionnaire_B_start : ℕ) 
  (questionnaire_B_end : ℕ) : 
  total_pool = 960 → 
  sample_size = 32 → 
  first_selected = 9 → 
  questionnaire_B_start = 461 → 
  questionnaire_B_end = 761 → 
  (Finset.filter (fun n => 
    questionnaire_B_start ≤ (first_selected + (n - 1) * (total_pool / sample_size)) ∧ 
    (first_selected + (n - 1) * (total_pool / sample_size)) ≤ questionnaire_B_end
  ) (Finset.range (sample_size + 1))).card = 10 := by
  sorry


end systematic_sampling_questionnaire_B_l658_65814


namespace negation_of_universal_statement_l658_65843

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x > 0 → x ≥ 1) ↔ (∃ x : ℝ, x > 0 ∧ x < 1) :=
by sorry

end negation_of_universal_statement_l658_65843


namespace sequence_problem_l658_65882

/-- Given a geometric sequence {a_n} and an arithmetic sequence {b_n} satisfying certain conditions,
    this theorem proves the general formulas for both sequences and the minimum n for which
    the sum of their first n terms exceeds 100. -/
theorem sequence_problem (a b : ℕ → ℝ) (n : ℕ) : 
  (∀ k, a (k + 1) = a k * (a 2 / a 1)) →  -- geometric sequence condition
  (∀ k, b (k + 1) - b k = b 2 - b 1) →   -- arithmetic sequence condition
  a 1 = 1 →
  b 1 = 1 →
  a 1 ≠ a 2 →
  a 1 + b 3 = 2 * a 2 →  -- a₁, a₂, b₃ form an arithmetic sequence
  b 1 * b 4 = (a 2)^2 →  -- b₁, a₂, b₄ form a geometric sequence
  (∀ k, a k = 2^(k-1)) ∧ 
  (∀ k, b k = k) ∧
  (n = 7 ∧ (2^n - 1 + n * (n + 1) / 2 > 100) ∧ 
   ∀ m < n, (2^m - 1 + m * (m + 1) / 2 ≤ 100)) :=
by sorry


end sequence_problem_l658_65882


namespace certain_event_three_people_two_groups_l658_65858

theorem certain_event_three_people_two_groups : 
  ∀ (group1 group2 : Finset Nat), 
  (group1 ∪ group2).card = 3 → 
  group1 ∩ group2 = ∅ → 
  group1 ≠ ∅ → 
  group2 ≠ ∅ → 
  (group1.card = 2 ∨ group2.card = 2) :=
sorry

end certain_event_three_people_two_groups_l658_65858


namespace max_dimes_in_piggy_banks_l658_65830

/-- Represents the number of coins a piggy bank can hold -/
def PiggyBankCapacity : ℕ := 100

/-- Represents the total number of coins in two piggy banks -/
def TotalCoins : ℕ := 200

/-- Represents the total value of coins in cents -/
def TotalValue : ℕ := 1200

/-- Represents the value of a dime in cents -/
def DimeValue : ℕ := 10

/-- Represents the value of a penny in cents -/
def PennyValue : ℕ := 1

/-- Theorem stating the maximum number of dimes that can be held in the piggy banks -/
theorem max_dimes_in_piggy_banks :
  ∃ (d : ℕ), d ≤ 111 ∧
  d * DimeValue + (TotalCoins - d) * PennyValue = TotalValue ∧
  (∀ (x : ℕ), x > d →
    x * DimeValue + (TotalCoins - x) * PennyValue ≠ TotalValue) :=
by sorry

#check max_dimes_in_piggy_banks

end max_dimes_in_piggy_banks_l658_65830


namespace classroom_to_total_ratio_is_one_to_four_l658_65811

/-- Given a class of students with some on the playground and some in the classroom,
    prove that the ratio of students in the classroom to total students is 1:4. -/
theorem classroom_to_total_ratio_is_one_to_four
  (total_students : ℕ)
  (playground_students : ℕ)
  (classroom_students : ℕ)
  (playground_girls : ℕ)
  (h1 : total_students = 20)
  (h2 : total_students = playground_students + classroom_students)
  (h3 : playground_girls = 10)
  (h4 : playground_girls = (2 : ℚ) / 3 * playground_students) :
  (classroom_students : ℚ) / total_students = 1 / 4 := by
  sorry

end classroom_to_total_ratio_is_one_to_four_l658_65811


namespace cylinder_height_equals_sphere_surface_area_l658_65873

/-- Given a sphere of radius 6 cm and a right circular cylinder with equal height and diameter,
    if their surface areas are equal, then the height of the cylinder is 6√2 cm. -/
theorem cylinder_height_equals_sphere_surface_area (r h : ℝ) : 
  r = 6 →  -- radius of sphere is 6 cm
  h = 2 * r →  -- height of cylinder equals its diameter
  4 * Real.pi * r^2 = 2 * Real.pi * r * h →  -- surface areas are equal
  h = 6 * Real.sqrt 2 := by
  sorry

#check cylinder_height_equals_sphere_surface_area

end cylinder_height_equals_sphere_surface_area_l658_65873


namespace scores_mode_and_median_l658_65877

def scores : List ℕ := [80, 85, 85, 85, 90, 90, 90, 90, 95]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem scores_mode_and_median :
  mode scores = 90 ∧ median scores = 90 := by sorry

end scores_mode_and_median_l658_65877


namespace inverse_of_composed_linear_functions_l658_65820

/-- Given two functions p and q, we define r as their composition and prove its inverse -/
theorem inverse_of_composed_linear_functions 
  (p q r : ℝ → ℝ)
  (hp : ∀ x, p x = 4 * x - 7)
  (hq : ∀ x, q x = 3 * x + 2)
  (hr : ∀ x, r x = p (q x))
  : (∀ x, r x = 12 * x + 1) ∧ 
    (∀ x, Function.invFun r x = (x - 1) / 12) := by
  sorry


end inverse_of_composed_linear_functions_l658_65820


namespace crayons_in_drawer_l658_65829

theorem crayons_in_drawer (initial_crayons : ℕ) :
  (initial_crayons + 3 = 10) → initial_crayons = 7 := by
  sorry

end crayons_in_drawer_l658_65829


namespace two_books_from_different_genres_l658_65826

/-- The number of ways to choose two books from different genres -/
def choose_two_books (mystery : ℕ) (fantasy : ℕ) (biography : ℕ) : ℕ :=
  mystery * fantasy + mystery * biography + fantasy * biography

/-- Theorem: Given 5 mystery novels, 3 fantasy novels, and 2 biographies,
    the number of ways to choose 2 books from different genres is 31 -/
theorem two_books_from_different_genres :
  choose_two_books 5 3 2 = 31 := by
  sorry

end two_books_from_different_genres_l658_65826


namespace residue_problem_l658_65863

theorem residue_problem : (195 * 13 - 25 * 8 + 5) % 17 = 3 := by
  sorry

end residue_problem_l658_65863


namespace special_polyhedron_properties_l658_65864

structure Polyhedron where
  convex : Bool
  flat_faces : Bool
  symmetry_planes : Nat
  vertices : Nat
  edges_per_vertex : Nat
  vertex_types : List (Nat × List Nat)

def special_polyhedron : Polyhedron :=
{
  convex := true,
  flat_faces := true,
  symmetry_planes := 2,
  vertices := 8,
  edges_per_vertex := 3,
  vertex_types := [
    (2, [1, 1, 1]),
    (4, [1, 1, 2]),
    (2, [2, 2, 3])
  ]
}

theorem special_polyhedron_properties (K : Polyhedron) 
  (h : K = special_polyhedron) : 
  ∃ (surface_area volume : ℝ), 
    surface_area = 13.86 ∧ 
    volume = 2.946 :=
sorry

end special_polyhedron_properties_l658_65864


namespace weight_loss_challenge_l658_65825

theorem weight_loss_challenge (W : ℝ) (W_pos : W > 0) : 
  let weight_after_loss := W * 0.9
  let weight_with_clothes := weight_after_loss * 1.02
  let measured_loss_percentage := (W - weight_with_clothes) / W * 100
  measured_loss_percentage = 8.2 := by
sorry

end weight_loss_challenge_l658_65825


namespace min_detectors_for_ship_detection_l658_65842

/-- Represents a cell on the board -/
structure Cell :=
  (x : Fin 7)
  (y : Fin 7)

/-- Represents a 2x2 ship placement on the board -/
structure Ship :=
  (topLeft : Cell)

/-- Represents a detector placement on the board -/
structure Detector :=
  (position : Cell)

/-- A function that determines if a ship occupies a given cell -/
def shipOccupies (s : Ship) (c : Cell) : Prop :=
  s.topLeft.x ≤ c.x ∧ c.x < s.topLeft.x + 2 ∧
  s.topLeft.y ≤ c.y ∧ c.y < s.topLeft.y + 2

/-- A function that determines if a detector can detect a ship -/
def detectorDetects (d : Detector) (s : Ship) : Prop :=
  shipOccupies s d.position

/-- The main theorem stating that 16 detectors are sufficient and necessary -/
theorem min_detectors_for_ship_detection :
  ∃ (detectors : Finset Detector),
    (detectors.card = 16) ∧
    (∀ (s : Ship), ∃ (d : Detector), d ∈ detectors ∧ detectorDetects d s) ∧
    (∀ (detectors' : Finset Detector),
      detectors'.card < 16 →
      ∃ (s : Ship), ∀ (d : Detector), d ∈ detectors' → ¬detectorDetects d s) :=
by sorry

end min_detectors_for_ship_detection_l658_65842


namespace equal_goldfish_after_six_months_l658_65808

/-- Number of goldfish Brent has after n months -/
def brent_goldfish (n : ℕ) : ℕ := 2 * 4^n

/-- Number of goldfish Gretel has after n months -/
def gretel_goldfish (n : ℕ) : ℕ := 162 * 3^n

/-- The number of months it takes for Brent and Gretel to have the same number of goldfish -/
def months_to_equal_goldfish : ℕ := 6

/-- Theorem stating that after 'months_to_equal_goldfish' months, 
    Brent and Gretel have the same number of goldfish -/
theorem equal_goldfish_after_six_months : 
  brent_goldfish months_to_equal_goldfish = gretel_goldfish months_to_equal_goldfish :=
by sorry

end equal_goldfish_after_six_months_l658_65808


namespace nancy_spent_95_40_l658_65832

/-- The total amount Nancy spends on beads -/
def total_spent (crystal_price metal_price : ℚ) (crystal_sets metal_sets : ℕ) 
  (crystal_discount metal_tax : ℚ) : ℚ :=
  let crystal_cost := crystal_price * crystal_sets
  let metal_cost := metal_price * metal_sets
  let discounted_crystal := crystal_cost * (1 - crystal_discount)
  let taxed_metal := metal_cost * (1 + metal_tax)
  discounted_crystal + taxed_metal

/-- Theorem: Nancy spends $95.40 on beads -/
theorem nancy_spent_95_40 : 
  total_spent 12 15 3 4 (1/10) (1/20) = 95.4 := by
  sorry

end nancy_spent_95_40_l658_65832


namespace common_rest_days_1000_l658_65899

/-- Represents the work-rest cycle of a person -/
structure WorkCycle where
  workDays : ℕ
  restDays : ℕ

/-- Calculates the number of common rest days for two people within a given number of days -/
def commonRestDays (cycleA cycleB : WorkCycle) (totalDays : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of common rest days for Person A and Person B -/
theorem common_rest_days_1000 :
  let cycleA := WorkCycle.mk 3 1
  let cycleB := WorkCycle.mk 7 3
  commonRestDays cycleA cycleB 1000 = 100 := by
  sorry

end common_rest_days_1000_l658_65899


namespace extreme_value_condition_l658_65853

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - x - a * x

theorem extreme_value_condition (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) ∨
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≥ f a 1) ↔
  a = -1/2 := by
  sorry

end extreme_value_condition_l658_65853


namespace tangent_line_to_circle_l658_65886

/-- A line y = kx is tangent to the circle x^2 + y^2 - 6x + 8 = 0 at a point in the fourth quadrant -/
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ),
    y = k * x ∧
    x^2 + y^2 - 6*x + 8 = 0 ∧
    x > 0 ∧ y < 0 ∧
    ∀ (x' y' : ℝ), y' = k * x' → (x' - x)^2 + (y' - y)^2 > 0

theorem tangent_line_to_circle (k : ℝ) :
  is_tangent k → k = -Real.sqrt 2 / 4 :=
sorry

end tangent_line_to_circle_l658_65886


namespace smallest_lychee_count_correct_l658_65827

/-- The smallest number of lychees satisfying the distribution condition -/
def smallest_lychee_count : ℕ := 839

/-- Checks if a number satisfies the lychee distribution condition -/
def satisfies_condition (x : ℕ) : Prop :=
  ∀ n : ℕ, 3 ≤ n → n ≤ 8 → x % n = n - 1

theorem smallest_lychee_count_correct :
  satisfies_condition smallest_lychee_count ∧
  ∀ y : ℕ, y < smallest_lychee_count → ¬(satisfies_condition y) :=
by sorry

end smallest_lychee_count_correct_l658_65827


namespace mackenzie_bought_three_new_cds_l658_65865

/-- Represents the price of a new CD -/
def new_cd_price : ℚ := 127.92 - 2 * 9.99

/-- Represents the number of new CDs Mackenzie bought -/
def mackenzie_new_cds : ℚ := (133.89 - 8 * 9.99) / (127.92 - 2 * 9.99) * 6

theorem mackenzie_bought_three_new_cds : 
  ⌊mackenzie_new_cds⌋ = 3 := by sorry

end mackenzie_bought_three_new_cds_l658_65865


namespace fair_coin_999th_flip_l658_65874

/-- A fair coin is a coin that has equal probability of landing heads or tails. -/
def FairCoin : Type := Unit

/-- A sequence of coin flips. -/
def CoinFlips (n : ℕ) := Fin n → Bool

/-- The probability of an event occurring in a fair coin flip. -/
def prob (event : Bool → Prop) : ℚ := sorry

theorem fair_coin_999th_flip (c : FairCoin) (flips : CoinFlips 1000) :
  prob (λ result => result = true) = 1/2 := by sorry

end fair_coin_999th_flip_l658_65874


namespace unique_n_for_prime_ones_and_seven_l658_65841

def has_n_minus_one_ones_and_one_seven (n : ℕ) (x : ℕ) : Prop :=
  ∃ k : ℕ, k < n ∧ x = (10^n - 1) / 9 + 6 * 10^k

theorem unique_n_for_prime_ones_and_seven :
  ∃! n : ℕ, n > 0 ∧ ∀ x : ℕ, has_n_minus_one_ones_and_one_seven n x → Nat.Prime x :=
by sorry

end unique_n_for_prime_ones_and_seven_l658_65841


namespace strategies_conversion_l658_65866

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8^i)) 0

/-- The number of strategies in base 8 -/
def strategies_base8 : List Nat := [2, 3, 4]

theorem strategies_conversion :
  base8_to_base10 strategies_base8 = 282 := by
  sorry

end strategies_conversion_l658_65866


namespace f_properties_l658_65872

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x)

theorem f_properties :
  (∀ x ∈ Set.Icc (-5 : ℝ) 5, f (-1) x ≤ 37) ∧
  (∃ x ∈ Set.Icc (-5 : ℝ) 5, f (-1) x = 37) ∧
  (∀ x ∈ Set.Icc (-5 : ℝ) 5, 1 ≤ f (-1) x) ∧
  (∃ x ∈ Set.Icc (-5 : ℝ) 5, f (-1) x = 1) ∧
  (∀ a : ℝ, is_monotonic_on (f a) (-5) 5 ↔ a ≤ -5 ∨ a ≥ 5) :=
by sorry

end f_properties_l658_65872


namespace max_ratio_two_digit_mean_50_l658_65884

theorem max_ratio_two_digit_mean_50 :
  ∀ x y : ℕ,
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  (x + y) / 2 = 50 →
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  (a + b) / 2 = 50 →
  x / y ≤ 9 :=
by sorry

end max_ratio_two_digit_mean_50_l658_65884


namespace town_budget_ratio_l658_65840

theorem town_budget_ratio (total_budget education public_spaces : ℕ) 
  (h1 : total_budget = 32000000)
  (h2 : education = 12000000)
  (h3 : public_spaces = 4000000) :
  (total_budget - education - public_spaces) * 2 = total_budget :=
by sorry

end town_budget_ratio_l658_65840


namespace log_identity_l658_65880

theorem log_identity : Real.log 16 / Real.log 4 - (Real.log 3 / Real.log 2) * (Real.log 2 / Real.log 3) = 1 := by
  sorry

end log_identity_l658_65880


namespace grapes_purchased_l658_65887

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℕ := 9

/-- The price of mangoes per kg -/
def mango_price : ℕ := 45

/-- The total amount paid to the shopkeeper -/
def total_paid : ℕ := 965

/-- The amount of grapes purchased in kg -/
def grape_amount : ℕ := 8

theorem grapes_purchased : 
  grape_price * grape_amount + mango_price * mango_amount = total_paid :=
by sorry

end grapes_purchased_l658_65887


namespace algebraic_expression_value_l658_65883

/-- Given that:
  - a and b are opposite numbers
  - c and d are reciprocals
  - The distance from point m to the origin is 5
Prove that m^2 - 100a - 99b - bcd + |cd - 2| = -74 -/
theorem algebraic_expression_value 
  (a b c d m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : m^2 = 25) : 
  m^2 - 100*a - 99*b - b*c*d + |c*d - 2| = -74 := by
  sorry

end algebraic_expression_value_l658_65883


namespace complement_of_A_l658_65800

def U : Set Int := {-1, 0, 1, 2}

def A : Set Int := {x : Int | x^2 < 2}

theorem complement_of_A : (U \ A) = {2} := by sorry

end complement_of_A_l658_65800


namespace f_ratio_is_integer_l658_65803

/-- Sequence a_n defined recursively -/
def a (r s : ℕ+) : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => r * a r s (n + 1) + s * a r s n

/-- Product f_n of the first n terms of a_n -/
def f (r s : ℕ+) : ℕ → ℕ
  | 0 => 1
  | (n + 1) => f r s n * a r s (n + 1)

/-- Main theorem: f_n / (f_k * f_(n-k)) is an integer for 0 < k < n -/
theorem f_ratio_is_integer (r s : ℕ+) (n k : ℕ) (h1 : 0 < k) (h2 : k < n) :
  ∃ m : ℕ, f r s n = m * (f r s k * f r s (n - k)) := by
  sorry

end f_ratio_is_integer_l658_65803


namespace ceiling_of_3_7_l658_65813

theorem ceiling_of_3_7 : ⌈(3.7 : ℝ)⌉ = 4 := by sorry

end ceiling_of_3_7_l658_65813


namespace cubic_equation_real_root_l658_65839

theorem cubic_equation_real_root (K : ℝ) : 
  ∃ x : ℝ, x = K^3 * (x - 1) * (x - 2) * (x - 3) :=
sorry

end cubic_equation_real_root_l658_65839


namespace election_winner_votes_l658_65848

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  winner_percentage = 54/100 →
  vote_difference = 288 →
  ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference →
  ↑total_votes * winner_percentage = 1944 :=
by sorry

end election_winner_votes_l658_65848


namespace square_to_rectangle_area_increase_l658_65807

theorem square_to_rectangle_area_increase (a : ℝ) (h : a > 0) :
  let square_area := a ^ 2
  let rectangle_length := a * 1.4
  let rectangle_breadth := a * 1.3
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area - square_area = 0.82 * square_area := by sorry

end square_to_rectangle_area_increase_l658_65807


namespace frank_problems_per_type_is_30_l658_65867

/-- The number of math problems composed by Bill -/
def bill_problems : ℕ := 20

/-- The number of math problems composed by Ryan -/
def ryan_problems : ℕ := 2 * bill_problems

/-- The number of math problems composed by Frank -/
def frank_problems : ℕ := 3 * ryan_problems

/-- The number of different types of math problems -/
def problem_types : ℕ := 4

/-- The number of problems of each type that Frank composes -/
def frank_problems_per_type : ℕ := frank_problems / problem_types

theorem frank_problems_per_type_is_30 :
  frank_problems_per_type = 30 := by sorry

end frank_problems_per_type_is_30_l658_65867


namespace at_most_12_moves_for_9_l658_65819

/-- A move is defined as reversing the order of any block of consecutive increasing or decreasing numbers -/
def is_valid_move (perm : List Nat) (start finish : Nat) : Prop :=
  start < finish ∧ finish ≤ perm.length ∧
  (∀ i, start < i ∧ i < finish → perm[i-1]! < perm[i]! ∨ perm[i-1]! > perm[i]!)

/-- The function that counts the minimum number of moves needed to sort a permutation -/
def min_moves_to_sort (perm : List Nat) : Nat :=
  sorry

/-- Theorem stating that at most 12 moves are needed to sort any permutation of numbers from 1 to 9 -/
theorem at_most_12_moves_for_9 :
  ∀ perm : List Nat, perm.Nodup → perm.length = 9 → (∀ n, n ∈ perm ↔ 1 ≤ n ∧ n ≤ 9) →
  min_moves_to_sort perm ≤ 12 :=
by sorry

end at_most_12_moves_for_9_l658_65819


namespace min_value_of_function_min_value_is_achievable_l658_65888

theorem min_value_of_function (x : ℝ) : x^2 + 6 / (x^2 + 1) ≥ 2 * Real.sqrt 6 - 1 := by
  sorry

theorem min_value_is_achievable : ∃ x : ℝ, x^2 + 6 / (x^2 + 1) = 2 * Real.sqrt 6 - 1 := by
  sorry

end min_value_of_function_min_value_is_achievable_l658_65888


namespace prob_one_pilot_hits_l658_65817

/-- The probability that exactly one of two independent events occurs,
    given their individual probabilities of occurrence. -/
def prob_exactly_one (p_a p_b : ℝ) : ℝ := p_a * (1 - p_b) + (1 - p_a) * p_b

/-- The probability of pilot A hitting the target -/
def p_a : ℝ := 0.4

/-- The probability of pilot B hitting the target -/
def p_b : ℝ := 0.5

/-- Theorem: The probability that exactly one pilot hits the target is 0.5 -/
theorem prob_one_pilot_hits : prob_exactly_one p_a p_b = 0.5 := by
  sorry

end prob_one_pilot_hits_l658_65817


namespace last_three_digits_of_7_to_105_l658_65834

theorem last_three_digits_of_7_to_105 : 7^105 ≡ 783 [ZMOD 1000] := by
  sorry

end last_three_digits_of_7_to_105_l658_65834


namespace students_in_grades_2_and_3_l658_65824

theorem students_in_grades_2_and_3 (boys_grade_2 girls_grade_2 : ℕ) 
  (h1 : boys_grade_2 = 20)
  (h2 : girls_grade_2 = 11)
  (h3 : ∀ x, x = boys_grade_2 + girls_grade_2 → 2 * x = students_grade_3) :
  boys_grade_2 + girls_grade_2 + students_grade_3 = 93 :=
by
  sorry

#check students_in_grades_2_and_3

end students_in_grades_2_and_3_l658_65824


namespace unique_coefficients_sum_l658_65835

theorem unique_coefficients_sum : 
  let y : ℝ := Real.sqrt ((Real.sqrt 75 / 3) - 5/2)
  ∃! (a b c : ℕ+), 
    y^100 = 3*y^98 + 15*y^96 + 12*y^94 - 2*y^50 + (a : ℝ)*y^46 + (b : ℝ)*y^44 + (c : ℝ)*y^40 ∧
    a + b + c = 66 := by sorry

end unique_coefficients_sum_l658_65835


namespace quadratic_equation_result_l658_65845

theorem quadratic_equation_result : 
  ∀ y : ℝ, (6 * y^2 + 5 = 2 * y + 10) → (12 * y - 5)^2 = 133 := by
  sorry

end quadratic_equation_result_l658_65845


namespace rectangle_diagonal_l658_65885

/-- Given a rectangle with breadth b and length l, if its perimeter is 5 times its breadth
    and its area is 216 sq. cm, then its diagonal is 6√13 cm. -/
theorem rectangle_diagonal (b l : ℝ) (h1 : 2 * (l + b) = 5 * b) (h2 : l * b = 216) :
  Real.sqrt (l^2 + b^2) = 6 * Real.sqrt 13 := by
  sorry

end rectangle_diagonal_l658_65885


namespace clothing_calculation_l658_65857

/-- Calculates the remaining pieces of clothing after donations and disposal --/
def remaining_clothing (initial : ℕ) (first_donation : ℕ) (disposal : ℕ) : ℕ :=
  initial - (first_donation + 3 * first_donation) - disposal

/-- Theorem stating the remaining pieces of clothing --/
theorem clothing_calculation :
  remaining_clothing 100 5 15 = 65 := by
  sorry

end clothing_calculation_l658_65857


namespace inequalities_from_sum_positive_l658_65851

theorem inequalities_from_sum_positive (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a+2)*(b+2) > a*b) := by
  sorry

end inequalities_from_sum_positive_l658_65851


namespace g_composition_fixed_points_l658_65810

def g (x : ℝ) : ℝ := x^2 - 4*x + 1

theorem g_composition_fixed_points :
  {x : ℝ | g (g x) = g x} =
  {x : ℝ | x = 2 + Real.sqrt ((11 + 2*Real.sqrt 21)/2) ∨
           x = 2 - Real.sqrt ((11 + 2*Real.sqrt 21)/2) ∨
           x = 2 + Real.sqrt ((11 - 2*Real.sqrt 21)/2) ∨
           x = 2 - Real.sqrt ((11 - 2*Real.sqrt 21)/2)} :=
by sorry

end g_composition_fixed_points_l658_65810


namespace expected_draws_for_given_balls_l658_65850

/-- Represents the number of balls of each color in the bag -/
structure BallCount where
  red : ℕ
  yellow : ℕ

/-- Calculates the expected number of balls drawn until two different colors are drawn -/
def expectedDraws (balls : BallCount) : ℚ :=
  sorry

/-- The theorem stating that the expected number of draws is 5/2 for the given ball configuration -/
theorem expected_draws_for_given_balls :
  expectedDraws { red := 3, yellow := 2 } = 5/2 := by sorry

end expected_draws_for_given_balls_l658_65850


namespace sum_of_roots_squared_difference_sum_of_roots_specific_equation_l658_65875

theorem sum_of_roots_squared_difference (a c : ℝ) :
  let f := fun x : ℝ => (x - a)^2 - c
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∀ x y : ℝ, f x = 0 → f y = 0 → x + y = 2*a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f := fun x : ℝ => (x - 5)^2 - 9
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∀ x y : ℝ, f x = 0 → f y = 0 → x + y = 10) :=
by sorry

end sum_of_roots_squared_difference_sum_of_roots_specific_equation_l658_65875


namespace unique_solution_condition_l658_65837

/-- The function representing y = 2x^2 --/
def f (x : ℝ) : ℝ := 2 * x^2

/-- The function representing y = 4x + c --/
def g (c : ℝ) (x : ℝ) : ℝ := 4 * x + c

/-- The condition for two identical solutions --/
def has_two_identical_solutions (c : ℝ) : Prop :=
  ∃! x : ℝ, f x = g c x ∧ ∀ y : ℝ, f y = g c y → y = x

theorem unique_solution_condition (c : ℝ) :
  has_two_identical_solutions c ↔ c = -2 := by
  sorry

end unique_solution_condition_l658_65837


namespace power_inequality_l658_65860

theorem power_inequality (a b : ℕ) (ha : a > 1) (hb : b > 2) :
  a^b + 1 ≥ b * (a + 1) ∧ (a^b + 1 = b * (a + 1) ↔ a = 2 ∧ b = 3) := by
  sorry

end power_inequality_l658_65860


namespace definite_integral_quarter_circle_l658_65809

theorem definite_integral_quarter_circle (f : ℝ → ℝ) :
  (∫ x in (0 : ℝ)..(Real.sqrt 2), f x) = π / 2 :=
by
  sorry

end definite_integral_quarter_circle_l658_65809


namespace complex_fraction_simplification_l658_65890

/-- Given that i² = -1, prove that (3 - 2i) / (4 + 5i) = 2/41 - 23/41 * i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (3 - 2*i) / (4 + 5*i) = 2/41 - 23/41 * i :=
by sorry

end complex_fraction_simplification_l658_65890


namespace sam_football_games_l658_65871

/-- The number of football games Sam went to this year -/
def games_this_year : ℕ := 43 - 29

/-- Theorem stating that Sam went to 14 football games this year -/
theorem sam_football_games : games_this_year = 14 := by
  sorry

end sam_football_games_l658_65871


namespace group_frequency_l658_65862

theorem group_frequency (sample_capacity : ℕ) (group_frequency : ℚ) : 
  sample_capacity = 20 → group_frequency = 0.25 → 
  (sample_capacity : ℚ) * group_frequency = 5 := by
  sorry

end group_frequency_l658_65862


namespace perpendicular_planes_theorem_l658_65895

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_theorem 
  (α β γ : Plane) (m n : Line)
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h_diff_lines : m ≠ n)
  (h_n_perp_α : perp n α)
  (h_n_perp_β : perp n β)
  (h_m_perp_α : perp m α) :
  perp m β :=
sorry

end perpendicular_planes_theorem_l658_65895


namespace quadratic_equal_roots_l658_65881

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - (k + 1) * x + 2 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - (k + 1) * y + 2 = 0 → y = x) ↔ 
  (k = 2 * Real.sqrt 6 - 1 ∨ k = -2 * Real.sqrt 6 - 1) := by
sorry

end quadratic_equal_roots_l658_65881


namespace union_of_given_sets_l658_65804

theorem union_of_given_sets :
  let A : Set Int := {-3, 1, 2}
  let B : Set Int := {0, 1, 2, 3}
  A ∪ B = {-3, 0, 1, 2, 3} := by
sorry

end union_of_given_sets_l658_65804


namespace pebble_distribution_correct_l658_65831

/-- The number of friends who received pebbles from Janice --/
def num_friends : ℕ := 17

/-- The total weight of pebbles in grams --/
def total_weight : ℕ := 36000

/-- The weight of a small pebble in grams --/
def small_pebble_weight : ℕ := 200

/-- The weight of a large pebble in grams --/
def large_pebble_weight : ℕ := 300

/-- The number of small pebbles given to each friend --/
def small_pebbles_per_friend : ℕ := 3

/-- The number of large pebbles given to each friend --/
def large_pebbles_per_friend : ℕ := 5

/-- Theorem stating that the number of friends who received pebbles is correct --/
theorem pebble_distribution_correct : 
  num_friends * (small_pebbles_per_friend * small_pebble_weight + 
                 large_pebbles_per_friend * large_pebble_weight) ≤ total_weight ∧
  (num_friends + 1) * (small_pebbles_per_friend * small_pebble_weight + 
                       large_pebbles_per_friend * large_pebble_weight) > total_weight :=
by sorry

end pebble_distribution_correct_l658_65831


namespace opposite_of_sqrt_difference_l658_65894

theorem opposite_of_sqrt_difference : -(Real.sqrt 2 - Real.sqrt 3) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end opposite_of_sqrt_difference_l658_65894


namespace farm_animals_count_l658_65821

/-- Represents the number of animals on a farm --/
structure FarmAnimals where
  cows : ℕ
  dogs : ℕ
  cats : ℕ
  sheep : ℕ
  chickens : ℕ

/-- Calculates the total number of animals on the farm --/
def totalAnimals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.dogs + farm.cats + farm.sheep + farm.chickens

/-- Represents the initial state of the farm --/
def initialFarm : FarmAnimals where
  cows := 120
  dogs := 18
  cats := 6
  sheep := 0
  chickens := 288

/-- Applies the changes to the farm as described in the problem --/
def applyChanges (farm : FarmAnimals) : FarmAnimals :=
  let soldCows := farm.cows / 4
  let soldDogs := farm.dogs * 3 / 5
  let remainingDogs := farm.dogs - soldDogs + soldDogs  -- Sell and add back equal number
  { cows := farm.cows - soldCows,
    dogs := remainingDogs,
    cats := farm.cats,
    sheep := remainingDogs / 2,
    chickens := farm.chickens * 3 / 2 }  -- 50% increase

theorem farm_animals_count :
  totalAnimals (applyChanges initialFarm) = 555 :=
by sorry

end farm_animals_count_l658_65821


namespace probability_system_failure_correct_l658_65892

/-- The probability of at least one component failing in a system of m identical components -/
def probability_system_failure (m : ℕ) (P : ℝ) : ℝ :=
  1 - (1 - P)^m

/-- Theorem: The probability of at least one component failing in a system of m identical components
    with individual failure probability P is 1-(1-P)^m -/
theorem probability_system_failure_correct (m : ℕ) (P : ℝ) 
    (h1 : 0 ≤ P) (h2 : P ≤ 1) :
  probability_system_failure m P = 1 - (1 - P)^m :=
by
  sorry

end probability_system_failure_correct_l658_65892


namespace pen_pencil_cost_difference_l658_65869

/-- The cost difference between a pen and a pencil -/
def cost_difference (pen_cost pencil_cost : ℝ) : ℝ := pen_cost - pencil_cost

/-- The total cost of a pen and a pencil -/
def total_cost (pen_cost pencil_cost : ℝ) : ℝ := pen_cost + pencil_cost

theorem pen_pencil_cost_difference :
  ∀ (pen_cost : ℝ),
    pencil_cost = 2 →
    total_cost pen_cost pencil_cost = 13 →
    pen_cost > pencil_cost →
    cost_difference pen_cost pencil_cost = 9 :=
by
  sorry

end pen_pencil_cost_difference_l658_65869


namespace decimal_sum_to_fraction_l658_65818

theorem decimal_sum_to_fraction :
  (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 : ℚ) = 12345 / 160000 := by
  sorry

end decimal_sum_to_fraction_l658_65818


namespace square_plus_reciprocal_square_l658_65846

theorem square_plus_reciprocal_square (m : ℝ) (h : m + 1/m = 6) :
  m^2 + 1/m^2 + 4 = 38 := by
  sorry

end square_plus_reciprocal_square_l658_65846


namespace division_simplification_l658_65801

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  -6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y := by
  sorry

end division_simplification_l658_65801


namespace fishing_theorem_l658_65870

/-- The total number of fish caught by Leo and Agrey -/
def total_fish (leo_fish : ℕ) (agrey_fish : ℕ) : ℕ :=
  leo_fish + agrey_fish

/-- Theorem: Given Leo caught 40 fish and Agrey caught 20 more fish than Leo,
    the total number of fish they caught together is 100. -/
theorem fishing_theorem :
  let leo_fish : ℕ := 40
  let agrey_fish : ℕ := leo_fish + 20
  total_fish leo_fish agrey_fish = 100 := by
sorry

end fishing_theorem_l658_65870
