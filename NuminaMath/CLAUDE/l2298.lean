import Mathlib

namespace certain_number_proof_l2298_229880

theorem certain_number_proof :
  let first_number : ℝ := 15
  let certain_number : ℝ := (0.4 * first_number) - (0.8 * 5)
  certain_number = 2 := by
sorry

end certain_number_proof_l2298_229880


namespace chris_video_game_cost_l2298_229870

def video_game_cost (hourly_rate : ℕ) (hours_worked : ℕ) (candy_cost : ℕ) (leftover : ℕ) : ℕ :=
  hourly_rate * hours_worked - candy_cost - leftover

theorem chris_video_game_cost :
  video_game_cost 8 9 5 7 = 60 := by
  sorry

end chris_video_game_cost_l2298_229870


namespace min_values_l2298_229829

-- Define the logarithm function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the condition from the problem
def condition (x y : ℝ) : Prop := lg (3 * x) + lg y = lg (x + y + 1)

-- Theorem statement
theorem min_values {x y : ℝ} (h : condition x y) :
  (∀ a b : ℝ, condition a b → x * y ≤ a * b) ∧
  (∀ c d : ℝ, condition c d → x + y ≤ c + d) :=
by sorry

end min_values_l2298_229829


namespace locus_of_A_is_hyperbola_l2298_229822

/-- Triangle ABC with special properties -/
structure SpecialTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- H is the orthocenter
  H : ℝ × ℝ
  -- G is the centroid
  G : ℝ × ℝ
  -- B and C are fixed points
  B_fixed : B.1 = -a ∧ B.2 = 0
  C_fixed : C.1 = a ∧ C.2 = 0
  -- Midpoint of HG lies on BC
  HG_midpoint_on_BC : ∃ m : ℝ, (H.1 + G.1) / 2 = m ∧ (H.2 + G.2) / 2 = 0
  -- G is the centroid
  G_is_centroid : G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3
  -- H is the orthocenter
  H_is_orthocenter : (A.1 - B.1) * (H.1 - C.1) + (A.2 - B.2) * (H.2 - C.2) = 0 ∧
                     (B.1 - C.1) * (H.1 - A.1) + (B.2 - C.2) * (H.2 - A.2) = 0

/-- The locus of A in a special triangle is a hyperbola -/
theorem locus_of_A_is_hyperbola (t : SpecialTriangle) : 
  t.A.1^2 - t.A.2^2/3 = a^2 := by sorry

end locus_of_A_is_hyperbola_l2298_229822


namespace probability_neither_correct_l2298_229855

theorem probability_neither_correct (P_A P_B P_AB : ℝ) 
  (h1 : P_A = 0.75)
  (h2 : P_B = 0.70)
  (h3 : P_AB = 0.65)
  (h4 : 0 ≤ P_A ∧ P_A ≤ 1)
  (h5 : 0 ≤ P_B ∧ P_B ≤ 1)
  (h6 : 0 ≤ P_AB ∧ P_AB ≤ 1) :
  1 - (P_A + P_B - P_AB) = 0.20 := by
  sorry

#check probability_neither_correct

end probability_neither_correct_l2298_229855


namespace equation_solution_l2298_229844

theorem equation_solution : ∃ N : ℝ,
  (∃ e₁ e₂ : ℝ, 2 * |2 - e₁| = N ∧ 2 * |2 - e₂| = N ∧ e₁ + e₂ = 4) →
  N = 0 :=
by sorry

end equation_solution_l2298_229844


namespace derangement_probability_five_l2298_229825

/-- The number of derangements of n elements -/
def derangement (n : ℕ) : ℕ := sorry

/-- The probability of a derangement of n elements -/
def derangementProbability (n : ℕ) : ℚ :=
  (derangement n : ℚ) / (Nat.factorial n)

theorem derangement_probability_five :
  derangementProbability 5 = 11 / 30 := by sorry

end derangement_probability_five_l2298_229825


namespace money_duration_l2298_229878

def lawn_money : ℕ := 9
def weed_eating_money : ℕ := 18
def weekly_spending : ℕ := 3

theorem money_duration : 
  (lawn_money + weed_eating_money) / weekly_spending = 9 := by
  sorry

end money_duration_l2298_229878


namespace sally_balloons_count_l2298_229837

/-- The number of blue balloons Joan initially has -/
def initial_balloons : ℕ := 9

/-- The number of blue balloons Joan gives to Jessica -/
def balloons_given_away : ℕ := 2

/-- The number of blue balloons Joan has after all transactions -/
def final_balloons : ℕ := 12

/-- The number of blue balloons Sally gives to Joan -/
def balloons_from_sally : ℕ := 5

theorem sally_balloons_count : 
  initial_balloons + balloons_from_sally - balloons_given_away = final_balloons :=
by sorry

end sally_balloons_count_l2298_229837


namespace triangle_theorem_triangle_range_theorem_l2298_229838

noncomputable section

def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (A + B + C = Real.pi) ∧ 
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem triangle_theorem 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_equation : (Real.cos B - 2 * Real.cos A) / (2 * a - b) = Real.cos C / c) :
  a / b = 2 := 
sorry

theorem triangle_range_theorem 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_equation : (Real.cos B - 2 * Real.cos A) / (2 * a - b) = Real.cos C / c)
  (h_obtuse : A > Real.pi / 2)
  (h_c : c = 3) :
  Real.sqrt 3 < b ∧ b < 3 := 
sorry

end triangle_theorem_triangle_range_theorem_l2298_229838


namespace exists_48_good_perfect_square_l2298_229817

/-- A number is k-good if it can be split into two parts y and z where y = k * z -/
def is_k_good (k : ℕ) (n : ℕ) : Prop :=
  ∃ y z : ℕ, y * (10^(Nat.log 10 z + 1)) + z = n ∧ y = k * z

/-- The main theorem: there exists a 48-good perfect square -/
theorem exists_48_good_perfect_square : ∃ n : ℕ, is_k_good 48 n ∧ ∃ m : ℕ, n = m^2 :=
sorry

end exists_48_good_perfect_square_l2298_229817


namespace lunch_percentage_l2298_229841

theorem lunch_percentage (total_students : ℕ) (total_students_pos : total_students > 0) :
  let boys := (6 : ℚ) / 10 * total_students
  let girls := (4 : ℚ) / 10 * total_students
  let boys_lunch := (60 : ℚ) / 100 * boys
  let girls_lunch := (40 : ℚ) / 100 * girls
  let total_lunch := boys_lunch + girls_lunch
  (total_lunch / total_students) * 100 = 52 := by
  sorry

end lunch_percentage_l2298_229841


namespace ambiguous_decomposition_l2298_229892

def M : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 4 * k - 3}

def isSimple (n : ℕ) : Prop :=
  n ∈ M ∧ ∀ a b : ℕ, a ∈ M → b ∈ M → a * b = n → (a = 1 ∨ b = 1)

theorem ambiguous_decomposition : ∃ n : ℕ,
  n ∈ M ∧ (∃ a b c d : ℕ,
    isSimple a ∧ isSimple b ∧ isSimple c ∧ isSimple d ∧
    a * b = n ∧ c * d = n ∧ (a ≠ c ∨ b ≠ d)) :=
sorry

end ambiguous_decomposition_l2298_229892


namespace rectangle_golden_ratio_l2298_229812

/-- A rectangle with sides x and y, where x > y, can be cut in half parallel to the longer side
    to produce scaled-down versions of the original if and only if x/y = √2 -/
theorem rectangle_golden_ratio (x y : ℝ) (h : x > y) (h' : x > 0) (h'' : y > 0) :
  (x / 2 : ℝ) / y = x / y ↔ x / y = Real.sqrt 2 := by
sorry

end rectangle_golden_ratio_l2298_229812


namespace points_in_quadrant_I_l2298_229826

theorem points_in_quadrant_I (x y : ℝ) : 
  y > -x + 6 ∧ y > 3*x - 2 → x > 0 ∧ y > 0 := by
sorry

end points_in_quadrant_I_l2298_229826


namespace multiple_optimal_solutions_l2298_229833

/-- The feasible region defined by the given linear constraints -/
def FeasibleRegion (x y : ℝ) : Prop :=
  2 * x - y + 2 ≥ 0 ∧ x - 3 * y + 1 ≤ 0 ∧ x + y - 2 ≤ 0

/-- The objective function z -/
def ObjectiveFunction (a x y : ℝ) : ℝ := a * x - y

/-- The theorem stating that a = 1/3 results in multiple optimal solutions -/
theorem multiple_optimal_solutions :
  ∃ (a : ℝ), a > 0 ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    FeasibleRegion x₁ y₁ ∧ FeasibleRegion x₂ y₂ ∧
    ObjectiveFunction a x₁ y₁ = ObjectiveFunction a x₂ y₂ ∧
    (∀ (x y : ℝ), FeasibleRegion x y → ObjectiveFunction a x y ≤ ObjectiveFunction a x₁ y₁)) ∧
  a = 1/3 :=
sorry

end multiple_optimal_solutions_l2298_229833


namespace problem_solution_l2298_229843

theorem problem_solution : 
  (2 * (Real.sqrt 3 - Real.sqrt 5) + 3 * (Real.sqrt 3 + Real.sqrt 5) = 5 * Real.sqrt 3 + Real.sqrt 5) ∧
  (-1^2 - |1 - Real.sqrt 3| + (8 : Real)^(1/3) - (-3) * Real.sqrt 9 = 11 - Real.sqrt 3) := by
  sorry

end problem_solution_l2298_229843


namespace square_division_perimeter_counterexample_l2298_229839

theorem square_division_perimeter_counterexample :
  ∃ (s : ℚ), 
    s > 0 ∧ 
    (∃ (w h : ℚ), w > 0 ∧ h > 0 ∧ w + h = s ∧ (2 * (w + h)).isInt) ∧ 
    ¬(4 * s).isInt :=
by sorry

end square_division_perimeter_counterexample_l2298_229839


namespace arithmetic_geometric_sum_ratio_l2298_229862

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℝ

/-- Sum of the first n terms of an arithmetic-geometric sequence -/
def sum_n (seq : ArithmeticGeometricSequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum seq.a

/-- Theorem: For an arithmetic-geometric sequence, if s_6 : s_3 = 1 : 2, then s_9 : s_3 = 3/4 -/
theorem arithmetic_geometric_sum_ratio 
  (seq : ArithmeticGeometricSequence) 
  (h : sum_n seq 6 / sum_n seq 3 = 1 / 2) : 
  sum_n seq 9 / sum_n seq 3 = 3 / 4 := by
  sorry

end arithmetic_geometric_sum_ratio_l2298_229862


namespace books_sold_l2298_229882

/-- Given Paul's initial and final number of books, prove that he sold 42 books. -/
theorem books_sold (initial_books final_books : ℕ) 
  (h1 : initial_books = 108) 
  (h2 : final_books = 66) : 
  initial_books - final_books = 42 := by
  sorry

#check books_sold

end books_sold_l2298_229882


namespace wall_width_proof_l2298_229810

/-- Given a rectangular wall and a square mirror, if the mirror's area is half the wall's area,
    prove that the wall's width is 68 inches. -/
theorem wall_width_proof (wall_length wall_width mirror_side : ℝ) : 
  wall_length = 85.76470588235294 →
  mirror_side = 54 →
  (mirror_side * mirror_side) = (wall_length * wall_width) / 2 →
  wall_width = 68 := by sorry

end wall_width_proof_l2298_229810


namespace prob_two_red_in_three_draws_l2298_229850

def total_balls : ℕ := 8
def red_balls : ℕ := 3
def white_balls : ℕ := 5

def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem prob_two_red_in_three_draws :
  (prob_white * prob_red * prob_red) + (prob_red * prob_white * prob_red) = 45 / 256 := by
  sorry

end prob_two_red_in_three_draws_l2298_229850


namespace park_trees_l2298_229899

/-- The number of walnut trees in the park after planting -/
def total_trees (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem: The park will have 77 walnut trees after planting -/
theorem park_trees : total_trees 33 44 = 77 := by
  sorry

end park_trees_l2298_229899


namespace max_intersections_12_6_l2298_229814

/-- The maximum number of intersection points in the first quadrant 
    given the number of points on x and y axes -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersections for 12 x-axis points
    and 6 y-axis points -/
theorem max_intersections_12_6 :
  max_intersections 12 6 = 990 := by
  sorry

#eval max_intersections 12 6

end max_intersections_12_6_l2298_229814


namespace place_four_men_five_women_l2298_229872

/-- The number of ways to place men and women into groups -/
def placeInGroups (numMen numWomen : ℕ) : ℕ :=
  let twoGroup := numMen * numWomen
  let threeGroup := (numMen - 1) * (numWomen.choose 2)
  let fourGroup := 1  -- As all remaining people form this group
  twoGroup * threeGroup * fourGroup

/-- Theorem stating the number of ways to place 4 men and 5 women into specific groups -/
theorem place_four_men_five_women :
  placeInGroups 4 5 = 360 := by
  sorry

#eval placeInGroups 4 5

end place_four_men_five_women_l2298_229872


namespace min_value_2x_plus_y_l2298_229898

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * (x + y) = 5 * x + y) :
  ∀ z : ℝ, 2 * x + y ≥ 9 ∧ (∃ x₀ y₀ : ℝ, 2 * x₀ + y₀ = 9 ∧ x₀ > 0 ∧ y₀ > 0 ∧ x₀ * (x₀ + y₀) = 5 * x₀ + y₀) :=
by sorry

end min_value_2x_plus_y_l2298_229898


namespace smallest_resolvable_debt_is_correct_l2298_229895

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 240

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = p * pig_value + g * goat_value

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 80

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, 0 < d → d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
by sorry

end smallest_resolvable_debt_is_correct_l2298_229895


namespace equation_solution_l2298_229803

theorem equation_solution :
  ∃ y : ℝ, (6 * y / (y + 2) - 4 / (y + 2) = 2 / (y + 2)) ∧ y = 1 := by
  sorry

end equation_solution_l2298_229803


namespace min_sum_arithmetic_sequence_l2298_229868

/-- An arithmetic sequence with a_n = 2n - 19 -/
def a (n : ℕ) : ℤ := 2 * n - 19

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n^2 - 18 * n

theorem min_sum_arithmetic_sequence :
  ∃ k : ℕ, k > 0 ∧ 
  (∀ n : ℕ, n > 0 → S n ≥ S k) ∧
  S k = -81 := by
  sorry

end min_sum_arithmetic_sequence_l2298_229868


namespace pen_sales_problem_l2298_229811

theorem pen_sales_problem (d : ℕ) : 
  (96 + 44 * d) / (d + 1) = 48 → d = 12 := by
  sorry

end pen_sales_problem_l2298_229811


namespace mr_green_potato_yield_l2298_229869

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (garden.length : ℝ) * step_length * (garden.width : ℝ) * step_length * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 18 25
  let step_length := 2.5
  let yield_per_sqft := 0.5
  expected_potato_yield garden step_length yield_per_sqft = 1406.25 := by
  sorry


end mr_green_potato_yield_l2298_229869


namespace shirt_discount_price_l2298_229883

theorem shirt_discount_price (original_price discount_percentage : ℝ) 
  (h1 : original_price = 80)
  (h2 : discount_percentage = 15) : 
  original_price * (1 - discount_percentage / 100) = 68 := by
  sorry

end shirt_discount_price_l2298_229883


namespace city_population_ratio_l2298_229884

theorem city_population_ratio (x y z : ℕ) (hxy : ∃ k : ℕ, x = k * y) (hyz : y = 2 * z) (hxz : x = 14 * z) :
  x / y = 7 :=
by sorry

end city_population_ratio_l2298_229884


namespace g_2_4_neg1_eq_neg7_div_3_l2298_229893

/-- The function g as defined in the problem -/
def g (a b c : ℚ) : ℚ := (a + b - c) / (a - b + c)

/-- Theorem stating that g(2, 4, -1) = -7/3 -/
theorem g_2_4_neg1_eq_neg7_div_3 : g 2 4 (-1) = -7/3 := by
  sorry

end g_2_4_neg1_eq_neg7_div_3_l2298_229893


namespace restaurant_bill_calculation_l2298_229807

/-- Restaurant bill calculation -/
theorem restaurant_bill_calculation
  (appetizer_cost : ℝ)
  (num_entrees : ℕ)
  (entree_cost : ℝ)
  (tip_percentage : ℝ)
  (h1 : appetizer_cost = 10)
  (h2 : num_entrees = 4)
  (h3 : entree_cost = 20)
  (h4 : tip_percentage = 0.20) :
  appetizer_cost + num_entrees * entree_cost + (appetizer_cost + num_entrees * entree_cost) * tip_percentage = 108 :=
by sorry

end restaurant_bill_calculation_l2298_229807


namespace system_solution_l2298_229801

theorem system_solution : ∃ (x y : ℚ), 
  (7 * x - 50 * y = 3) ∧ 
  (3 * y - x = 5) ∧ 
  (x = -259/29) ∧ 
  (y = -38/29) := by
sorry

end system_solution_l2298_229801


namespace number_relationship_l2298_229881

theorem number_relationship (x m : ℚ) : 
  x = 25 / 3 → 
  (3 * x + 15 = m * x - 10) → 
  m = 6 := by
  sorry

end number_relationship_l2298_229881


namespace max_concentration_time_l2298_229849

def drug_concentration_peak_time : ℝ := 0.65
def time_uncertainty : ℝ := 0.15

theorem max_concentration_time :
  drug_concentration_peak_time + time_uncertainty = 0.8 := by sorry

end max_concentration_time_l2298_229849


namespace solve_equation_l2298_229888

theorem solve_equation (n : ℤ) : (n + 1999) / 2 = -1 → n = -2001 := by
  sorry

end solve_equation_l2298_229888


namespace genevieve_errors_fixed_l2298_229889

/-- Represents the number of errors fixed by a programmer -/
def errors_fixed (total_lines : ℕ) (debug_interval : ℕ) (errors_per_debug : ℕ) : ℕ :=
  (total_lines / debug_interval) * errors_per_debug

/-- Theorem stating the number of errors fixed by Genevieve -/
theorem genevieve_errors_fixed :
  errors_fixed 4300 100 3 = 129 := by
  sorry

end genevieve_errors_fixed_l2298_229889


namespace pure_imaginary_fraction_l2298_229831

theorem pure_imaginary_fraction (α : ℝ) : 
  (∃ (y : ℝ), (α + 3 * Complex.I) / (1 + 2 * Complex.I) = y * Complex.I) → α = -6 := by
  sorry

end pure_imaginary_fraction_l2298_229831


namespace magazine_subscription_cost_l2298_229824

/-- If a 35% reduction in a cost results in a decrease of $611, then the original cost was $1745.71 -/
theorem magazine_subscription_cost (C : ℝ) : (0.35 * C = 611) → C = 1745.71 := by
  sorry

end magazine_subscription_cost_l2298_229824


namespace proportional_relation_l2298_229858

/-- Given that x is directly proportional to y^2 and y is inversely proportional to z,
    prove that if x = 5 when z = 20, then x = 40/81 when z = 45. -/
theorem proportional_relation (x y z : ℝ) (c d : ℝ) (h1 : x = c * y^2) (h2 : y * z = d)
  (h3 : z = 20 → x = 5) : z = 45 → x = 40 / 81 := by
  sorry

end proportional_relation_l2298_229858


namespace jim_age_in_two_years_l2298_229856

theorem jim_age_in_two_years :
  let tom_age_five_years_ago : ℕ := 32
  let years_since_tom_age : ℕ := 5
  let years_to_past_reference : ℕ := 7
  let jim_age_difference : ℕ := 5
  let years_to_future : ℕ := 2

  let tom_current_age : ℕ := tom_age_five_years_ago + years_since_tom_age
  let tom_age_at_reference : ℕ := tom_current_age - years_to_past_reference
  let jim_age_at_reference : ℕ := (tom_age_at_reference / 2) + jim_age_difference
  let jim_current_age : ℕ := jim_age_at_reference + years_to_past_reference
  let jim_future_age : ℕ := jim_current_age + years_to_future

  jim_future_age = 29 := by sorry

end jim_age_in_two_years_l2298_229856


namespace florist_roses_l2298_229861

theorem florist_roses (initial : Float) (first_pick : Float) (second_pick : Float) 
  (h1 : initial = 37.0) 
  (h2 : first_pick = 16.0) 
  (h3 : second_pick = 19.0) : 
  initial + first_pick + second_pick = 72.0 := by
  sorry

end florist_roses_l2298_229861


namespace sum_of_valid_numbers_l2298_229857

def digits : List Nat := [1, 3, 5]

def isValidNumber (n : Nat) : Bool :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) ∈ digits ∧
  ((n / 10) % 10) ∈ digits ∧
  (n % 10) ∈ digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10)

def validNumbers : List Nat :=
  (List.range 1000).filter isValidNumber

theorem sum_of_valid_numbers :
  validNumbers.sum = 1998 := by sorry

end sum_of_valid_numbers_l2298_229857


namespace T_equals_x_to_fourth_l2298_229840

theorem T_equals_x_to_fourth (x : ℝ) : 
  (x - 2)^4 + 5*(x - 2)^3 + 10*(x - 2)^2 + 10*(x - 2) + 5 = x^4 := by
  sorry

end T_equals_x_to_fourth_l2298_229840


namespace rachel_apples_remaining_l2298_229852

/-- The number of apples remaining on a tree after some are picked. -/
def applesRemaining (initial : ℕ) (picked : ℕ) : ℕ :=
  initial - picked

/-- Theorem: There are 3 apples remaining on Rachel's tree. -/
theorem rachel_apples_remaining :
  applesRemaining 7 4 = 3 := by
  sorry

end rachel_apples_remaining_l2298_229852


namespace average_favorable_draws_l2298_229806

def lottery_size : ℕ := 90
def draw_size : ℕ := 5

def favorable_draws : ℕ :=
  (86^2 * 85) / 2 + 87 * 85 + 86

def total_draws : ℕ :=
  lottery_size * (lottery_size - 1) * (lottery_size - 2) * (lottery_size - 3) * (lottery_size - 4) / 120

theorem average_favorable_draws :
  (total_draws : ℚ) / favorable_draws = 5874 / 43 := by sorry

end average_favorable_draws_l2298_229806


namespace non_shaded_perimeter_l2298_229864

/-- Given a rectangle with dimensions 12 × 10 inches, containing an inner rectangle
    of 6 × 2 inches, and a shaded area of 116 square inches, prove that the
    perimeter of the non-shaded region is 10 inches. -/
theorem non_shaded_perimeter (outer_length outer_width inner_length inner_width shaded_area : ℝ)
  (h_outer_length : outer_length = 12)
  (h_outer_width : outer_width = 10)
  (h_inner_length : inner_length = 6)
  (h_inner_width : inner_width = 2)
  (h_shaded_area : shaded_area = 116)
  (h_right_angles : ∀ angle, angle = 90) :
  let total_area := outer_length * outer_width
  let inner_area := inner_length * inner_width
  let non_shaded_area := total_area - shaded_area
  let non_shaded_length := 4
  let non_shaded_width := 1
  2 * (non_shaded_length + non_shaded_width) = 10 := by
    sorry

end non_shaded_perimeter_l2298_229864


namespace selling_price_calculation_l2298_229800

def cost_price : ℝ := 975
def profit_percentage : ℝ := 20

theorem selling_price_calculation :
  let profit := (profit_percentage / 100) * cost_price
  let selling_price := cost_price + profit
  selling_price = 1170 := by sorry

end selling_price_calculation_l2298_229800


namespace min_value_of_m_plus_2n_l2298_229890

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem min_value_of_m_plus_2n (a : ℝ) (m n : ℝ) :
  (∀ x, f a x ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3) →
  (m > 0) →
  (n > 0) →
  (1 / m + 1 / (2 * n) = a) →
  (∀ p q, p > 0 → q > 0 → 1 / p + 1 / (2 * q) = a → p + 2 * q ≥ m + 2 * n) →
  m + 2 * n = 4 * Real.sqrt 2 :=
sorry

end min_value_of_m_plus_2n_l2298_229890


namespace opposite_numbers_l2298_229859

theorem opposite_numbers : -4^2 = -((- 4)^2) := by sorry

end opposite_numbers_l2298_229859


namespace sum_of_abs_roots_l2298_229894

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 6*x^3 + 9*x^2 + 6*x - 14

-- Theorem statement
theorem sum_of_abs_roots :
  ∃ (r₁ r₂ r₃ r₄ : ℝ),
    (∀ x : ℝ, p x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    |r₁| + |r₂| + |r₃| + |r₄| = 3 + Real.sqrt 37 :=
by sorry

end sum_of_abs_roots_l2298_229894


namespace odot_composition_l2298_229865

/-- Custom operation ⊙ -/
def odot (x y : ℝ) : ℝ := x^2 + x*y - y^2

/-- Theorem stating that h ⊙ (h ⊙ h) = -4 when h = 2 -/
theorem odot_composition (h : ℝ) (h_eq : h = 2) : odot h (odot h h) = -4 := by
  sorry

end odot_composition_l2298_229865


namespace opposite_of_three_l2298_229863

theorem opposite_of_three : 
  (-(3 : ℤ) : ℤ) = -3 := by sorry

end opposite_of_three_l2298_229863


namespace derek_age_is_42_l2298_229879

-- Define the ages as natural numbers
def anne_age : ℕ := 36
def brianna_age : ℕ := (2 * anne_age) / 3
def caitlin_age : ℕ := brianna_age - 3
def derek_age : ℕ := 2 * caitlin_age

-- Theorem to prove Derek's age is 42
theorem derek_age_is_42 : derek_age = 42 := by
  sorry

end derek_age_is_42_l2298_229879


namespace quadratic_roots_problem_l2298_229813

theorem quadratic_roots_problem (x₁ x₂ b : ℝ) : 
  (∀ x, x^2 + b*x + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ - x₁*x₂ + x₂ = 2 →
  b = -6 := by
  sorry

end quadratic_roots_problem_l2298_229813


namespace sum_of_fractions_equals_one_l2298_229871

theorem sum_of_fractions_equals_one 
  (a b c : ℝ) 
  (h : a * b * c = 1) : 
  (a / (a * b + a + 1)) + (b / (b * c + b + 1)) + (c / (c * a + c + 1)) = 1 := by
  sorry

end sum_of_fractions_equals_one_l2298_229871


namespace max_third_side_of_triangle_l2298_229853

theorem max_third_side_of_triangle (a b c : ℝ) : 
  a = 7 → b = 10 → c > 0 → a + b + c ≤ 30 → 
  a + b > c → a + c > b → b + c > a → 
  ∀ n : ℕ, (n : ℝ) > c → n ≤ 13 :=
by sorry

end max_third_side_of_triangle_l2298_229853


namespace frog_escape_probability_l2298_229867

/-- Probability of frog escaping from pad N -/
noncomputable def P (N : ℕ) : ℝ :=
  sorry

/-- Total number of lily pads -/
def total_pads : ℕ := 15

/-- Starting pad for the frog -/
def start_pad : ℕ := 2

theorem frog_escape_probability :
  (∀ N, 0 < N → N < total_pads - 1 →
    P N = (N : ℝ) / total_pads * P (N - 1) + (1 - (N : ℝ) / total_pads) * P (N + 1)) →
  P 0 = 0 →
  P (total_pads - 1) = 1 →
  P start_pad = 163 / 377 :=
sorry

end frog_escape_probability_l2298_229867


namespace triangular_array_coins_l2298_229836

theorem triangular_array_coins (N : ℕ) : 
  (N * (N + 1)) / 2 = 2485 → N = 70 ∧ (N / 10 * (N % 10)) = 0 := by
  sorry

end triangular_array_coins_l2298_229836


namespace distribute_five_balls_four_boxes_l2298_229845

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with at least one box containing a ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

theorem distribute_five_balls_four_boxes :
  distribute_balls 5 4 = 52 := by
  sorry

end distribute_five_balls_four_boxes_l2298_229845


namespace mall_profit_l2298_229891

-- Define the cost prices of type A and B
def cost_A : ℝ := 120
def cost_B : ℝ := 100

-- Define the number of units of each type
def units_A : ℝ := 50
def units_B : ℝ := 30

-- Define the conditions
axiom cost_difference : cost_A = cost_B + 20
axiom cost_equality : 5 * cost_A = 6 * cost_B
axiom total_cost : cost_A * units_A + cost_B * units_B = 9000
axiom total_units : units_A + units_B = 80

-- Define the selling prices
def sell_A : ℝ := cost_A * 1.5 * 0.8
def sell_B : ℝ := cost_B + 30

-- Define the total profit
def total_profit : ℝ := (sell_A - cost_A) * units_A + (sell_B - cost_B) * units_B

-- Theorem to prove
theorem mall_profit : 
  cost_A = 120 ∧ cost_B = 100 ∧ total_profit = 2100 :=
sorry

end mall_profit_l2298_229891


namespace smallest_perfect_square_multiplier_l2298_229823

def y : ℕ := 2^5 * 3^5 * 4^5 * 5^5 * 6^5 * 7^5 * 8^5 * 9^5

theorem smallest_perfect_square_multiplier (k : ℕ) : 
  (∀ m : ℕ, m < 105 → ¬∃ n : ℕ, m * y = n^2) ∧ 
  (∃ n : ℕ, 105 * y = n^2) := by
  sorry

end smallest_perfect_square_multiplier_l2298_229823


namespace not_all_bisecting_diameters_perpendicular_l2298_229851

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A chord of a circle -/
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- A diameter of a circle -/
structure Diameter (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Predicate to check if a diameter bisects a chord -/
def bisects (d : Diameter c) (ch : Chord c) : Prop :=
  sorry

/-- Predicate to check if a diameter is perpendicular to a chord -/
def perpendicular (d : Diameter c) (ch : Chord c) : Prop :=
  sorry

/-- Theorem stating that it's not always true that a diameter bisecting a chord is perpendicular to it -/
theorem not_all_bisecting_diameters_perpendicular (c : Circle) :
  ∃ (d : Diameter c) (ch : Chord c), bisects d ch ∧ ¬perpendicular d ch :=
sorry

end not_all_bisecting_diameters_perpendicular_l2298_229851


namespace cone_lateral_surface_area_l2298_229830

/-- Given a cone with base radius 1 and lateral surface that unfolds to a 
    sector with a 90° central angle, its lateral surface area is 4π. -/
theorem cone_lateral_surface_area (r : Real) (θ : Real) : 
  r = 1 → θ = 90 → ∃ (l : Real), l * θ / 360 * (2 * Real.pi) = 2 * Real.pi ∧ 
    r * l * Real.pi = 4 * Real.pi := by
  sorry

#check cone_lateral_surface_area

end cone_lateral_surface_area_l2298_229830


namespace carbonated_water_percentage_l2298_229875

/-- Represents a solution with percentages of lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated : ℝ
  sum_to_one : lemonade + carbonated = 1

/-- Represents a mixture of two solutions -/
structure Mixture where
  solution1 : Solution
  solution2 : Solution
  proportion1 : ℝ
  proportion2 : ℝ
  sum_to_one : proportion1 + proportion2 = 1

theorem carbonated_water_percentage
  (sol1 : Solution)
  (sol2 : Solution)
  (mix : Mixture)
  (h1 : sol1.carbonated = 0.8)
  (h2 : sol2.lemonade = 0.45)
  (h3 : mix.solution1 = sol1)
  (h4 : mix.solution2 = sol2)
  (h5 : mix.proportion1 = 0.5)
  (h6 : mix.proportion2 = 0.5)
  (h7 : mix.proportion1 * sol1.carbonated + mix.proportion2 * sol2.carbonated = 0.675) :
  sol2.carbonated = 0.55 := by
  sorry


end carbonated_water_percentage_l2298_229875


namespace add_12345_seconds_to_5_45_00_l2298_229846

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem add_12345_seconds_to_5_45_00 :
  addSeconds { hours := 5, minutes := 45, seconds := 0 } 12345 =
  { hours := 9, minutes := 10, seconds := 45 } :=
sorry

end add_12345_seconds_to_5_45_00_l2298_229846


namespace simplify_polynomial_l2298_229832

theorem simplify_polynomial (y : ℝ) : 
  2*y*(4*y^3 - 3*y + 5) - 4*(y^3 - 3*y^2 + 4*y - 6) = 
  8*y^4 - 4*y^3 + 6*y^2 - 6*y + 24 := by
sorry

end simplify_polynomial_l2298_229832


namespace problem_solution_l2298_229876

theorem problem_solution :
  (∃ n : ℕ, n = 4 * 7 + 5 ∧ n = 33) ∧
  (∃ m : ℕ, m * 6 = 300 ∧ m = 50) := by
  sorry

end problem_solution_l2298_229876


namespace investment_rate_is_five_percent_l2298_229887

/-- Represents an investment account --/
structure Account where
  balance : ℝ
  rate : ℝ

/-- Calculates the interest earned on an account in one year --/
def interest (a : Account) : ℝ := a.balance * a.rate

/-- Represents the investment scenario --/
structure InvestmentScenario where
  account1 : Account
  account2 : Account
  totalInterest : ℝ

/-- The given investment scenario --/
def scenario : InvestmentScenario where
  account1 := { balance := 8000, rate := 0.05 }
  account2 := { balance := 2000, rate := 0.06 }
  totalInterest := 520

/-- Theorem stating that the given scenario satisfies all conditions --/
theorem investment_rate_is_five_percent : 
  scenario.account1.balance = 4 * scenario.account2.balance ∧
  scenario.account2.rate = 0.06 ∧
  interest scenario.account1 + interest scenario.account2 = scenario.totalInterest ∧
  scenario.account1.rate = 0.05 := by
  sorry

#check investment_rate_is_five_percent

end investment_rate_is_five_percent_l2298_229887


namespace square_area_ratio_l2298_229821

theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (y^2) / ((5*y)^2) = 1 / 25 := by sorry

end square_area_ratio_l2298_229821


namespace simplify_expression_l2298_229842

theorem simplify_expression : (27 * 10^9) / (9 * 10^2) = 3000000 := by
  sorry

end simplify_expression_l2298_229842


namespace book_purchase_total_price_l2298_229847

/-- Calculates the total price of books given the following conditions:
  * Total number of books
  * Number of math books
  * Price of a math book
  * Price of a history book
-/
def total_price (total_books : ℕ) (math_books : ℕ) (math_price : ℕ) (history_price : ℕ) : ℕ :=
  math_books * math_price + (total_books - math_books) * history_price

/-- Theorem stating that given the specific conditions in the problem,
    the total price of books is $390. -/
theorem book_purchase_total_price :
  total_price 80 10 4 5 = 390 := by
  sorry

end book_purchase_total_price_l2298_229847


namespace quadratic_one_solution_l2298_229877

/-- For a quadratic equation px^2 - 16x + 5 = 0, where p is nonzero,
    the equation has only one solution if and only if p = 64/5 -/
theorem quadratic_one_solution (p : ℝ) (hp : p ≠ 0) :
  (∃! x, p * x^2 - 16 * x + 5 = 0) ↔ p = 64/5 := by
  sorry

end quadratic_one_solution_l2298_229877


namespace ceiling_neg_sqrt_100_over_9_l2298_229854

theorem ceiling_neg_sqrt_100_over_9 : ⌈-Real.sqrt (100 / 9)⌉ = -3 := by sorry

end ceiling_neg_sqrt_100_over_9_l2298_229854


namespace fraction_multiplication_l2298_229866

theorem fraction_multiplication : (1 : ℚ) / 3 * (3 : ℚ) / 5 * (5 : ℚ) / 7 = (1 : ℚ) / 7 := by
  sorry

end fraction_multiplication_l2298_229866


namespace apple_redistribution_theorem_l2298_229820

/-- Represents the state of apples in baskets -/
structure AppleBaskets where
  total_apples : ℕ
  baskets : List ℕ
  deriving Repr

/-- Checks if all non-empty baskets have the same number of apples -/
def all_equal (ab : AppleBaskets) : Prop :=
  let non_empty := ab.baskets.filter (· > 0)
  non_empty.all (· = non_empty.head!)

/-- Checks if the total number of apples is at least 100 -/
def at_least_100 (ab : AppleBaskets) : Prop :=
  ab.total_apples ≥ 100

/-- Represents a valid redistribution of apples -/
def is_valid_redistribution (initial final : AppleBaskets) : Prop :=
  final.total_apples ≤ initial.total_apples ∧
  final.baskets.length ≤ initial.baskets.length

/-- The main theorem to prove -/
theorem apple_redistribution_theorem (initial : AppleBaskets) :
  initial.total_apples = 2000 →
  ∃ (final : AppleBaskets), 
    is_valid_redistribution initial final ∧
    all_equal final ∧
    at_least_100 final := by
  sorry

end apple_redistribution_theorem_l2298_229820


namespace molecular_weight_problem_l2298_229818

/-- Given that 3 moles of a compound have a molecular weight of 222,
    prove that the molecular weight of 1 mole of the compound is 74. -/
theorem molecular_weight_problem (moles : ℕ) (total_weight : ℝ) :
  moles = 3 →
  total_weight = 222 →
  total_weight / moles = 74 := by
  sorry

end molecular_weight_problem_l2298_229818


namespace gcd_294_84_l2298_229815

theorem gcd_294_84 : Nat.gcd 294 84 = 42 := by
  sorry

end gcd_294_84_l2298_229815


namespace table_height_l2298_229828

/-- Given three rectangular boxes with heights b, r, and g, and a table with height h,
    prove that h = 91 when the following conditions are met:
    1. h + b - g = 111
    2. h + r - b = 80
    3. h + g - r = 82 -/
theorem table_height (h b r g : ℝ) 
    (eq1 : h + b - g = 111)
    (eq2 : h + r - b = 80)
    (eq3 : h + g - r = 82) : h = 91 := by
  sorry

end table_height_l2298_229828


namespace trig_identity_l2298_229897

theorem trig_identity (θ : ℝ) (h : Real.sin (π + θ) = 1/4) :
  (Real.cos (π + θ)) / (Real.cos θ * (Real.cos (π + θ) - 1)) +
  (Real.sin (π/2 - θ)) / (Real.cos (θ + 2*π) * Real.cos (π + θ) + Real.cos (-θ)) = 32 := by
  sorry

end trig_identity_l2298_229897


namespace max_log_sum_and_min_reciprocal_sum_l2298_229827

open Real

theorem max_log_sum_and_min_reciprocal_sum (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + 5 * y = 20) :
  (∃ (u : ℝ), u = log x + log y ∧ ∀ (v : ℝ), v = log x + log y → v ≤ u) ∧
  u = 1 ∧
  (∃ (w : ℝ), w = 1/x + 1/y ∧ ∀ (z : ℝ), z = 1/x + 1/y → w ≤ z) ∧
  w = (7 + 2 * sqrt 10) / 20 := by
  sorry

end max_log_sum_and_min_reciprocal_sum_l2298_229827


namespace three_balls_selected_l2298_229885

def num_balls : ℕ := 100
def prob_odd_first : ℚ := 2/3

theorem three_balls_selected 
  (h1 : num_balls = 100)
  (h2 : prob_odd_first = 2/3)
  (h3 : ∃ (odd_count even_count : ℕ), 
    odd_count = 2 ∧ even_count = 1 ∧ 
    odd_count + even_count = num_selected) :
  num_selected = 3 := by
  sorry

end three_balls_selected_l2298_229885


namespace divisible_by_four_and_six_percentage_l2298_229805

theorem divisible_by_four_and_six_percentage (n : ℕ) : 
  (↑(Finset.filter (fun x => x % 4 = 0 ∧ x % 6 = 0) (Finset.range (n + 1))).card / n) * 100 = 8 :=
by
  sorry

end divisible_by_four_and_six_percentage_l2298_229805


namespace min_steps_even_correct_min_steps_odd_correct_l2298_229886

-- Define the stone arrangement
structure StoneArrangement where
  k : Nat
  n : Nat
  stones : List Nat

-- Define a step
def step (arrangement : StoneArrangement) : StoneArrangement := sorry

-- Define the minimum number of steps for even n
def min_steps_even (k : Nat) (n : Nat) : Nat :=
  (n^2 * k * (k-1)) / 4

-- Define the minimum number of steps for odd n and k = 3
def min_steps_odd (n : Nat) : Nat :=
  let q := (n - 1) / 2
  n^2 + 2 * q * (q + 1)

-- Theorem for even n
theorem min_steps_even_correct (k n : Nat) (h1 : k ≥ 2) (h2 : n % 2 = 0) :
  ∀ (arrangement : StoneArrangement),
    arrangement.k = k ∧ arrangement.n = n →
    ∃ (m : Nat), m ≤ min_steps_even k n ∧
      ∃ (final_arrangement : StoneArrangement),
        final_arrangement = (step^[m] arrangement) ∧
        -- The n stones of the same color are together in final_arrangement
        sorry := by sorry

-- Theorem for odd n and k = 3
theorem min_steps_odd_correct (n : Nat) (h1 : n % 2 = 1) :
  ∀ (arrangement : StoneArrangement),
    arrangement.k = 3 ∧ arrangement.n = n →
    ∃ (m : Nat), m ≤ min_steps_odd n ∧
      ∃ (final_arrangement : StoneArrangement),
        final_arrangement = (step^[m] arrangement) ∧
        -- The n stones of the same color are together in final_arrangement
        sorry := by sorry

end min_steps_even_correct_min_steps_odd_correct_l2298_229886


namespace no_perfect_square_with_conditions_l2298_229848

def is_nine_digit (n : ℕ) : Prop := 10^8 ≤ n ∧ n < 10^9

def contains_all_nonzero_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 9 → ∃ k : ℕ, n / 10^k % 10 = d

def last_digit_is_five (n : ℕ) : Prop := n % 10 = 5

theorem no_perfect_square_with_conditions :
  ¬ ∃ n : ℕ, is_nine_digit n ∧ contains_all_nonzero_digits n ∧ last_digit_is_five n ∧ ∃ m : ℕ, n = m^2 := by
  sorry

end no_perfect_square_with_conditions_l2298_229848


namespace parallelogram_area_theorem_l2298_229835

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : Real) : Real :=
  base * height

/-- The inclination of the parallelogram -/
def inclination : Real := 6

theorem parallelogram_area_theorem (base height : Real) 
  (h_base : base = 20) 
  (h_height : height = 4) :
  parallelogram_area base height = 80 := by
  sorry

end parallelogram_area_theorem_l2298_229835


namespace imaginary_part_of_one_minus_i_squared_l2298_229816

theorem imaginary_part_of_one_minus_i_squared :
  Complex.im ((1 - Complex.I) ^ 2) = -2 := by sorry

end imaginary_part_of_one_minus_i_squared_l2298_229816


namespace bell_rings_count_l2298_229896

/-- Represents a school event that causes the bell to ring at its start and end -/
structure SchoolEvent where
  name : String

/-- Represents the school schedule for a day -/
structure SchoolSchedule where
  events : List SchoolEvent

/-- Counts the number of bell rings for a given schedule up to and including a specific event -/
def countBellRings (schedule : SchoolSchedule) (currentEvent : SchoolEvent) : Nat :=
  sorry

/-- Monday's altered schedule -/
def mondaySchedule : SchoolSchedule :=
  { events := [
    { name := "Assembly" },
    { name := "Maths" },
    { name := "History" },
    { name := "Surprise Quiz" },
    { name := "Geography" },
    { name := "Science" },
    { name := "Music" }
  ] }

/-- The current event (Geography class) -/
def currentEvent : SchoolEvent :=
  { name := "Geography" }

theorem bell_rings_count :
  countBellRings mondaySchedule currentEvent = 9 := by
  sorry

end bell_rings_count_l2298_229896


namespace clothing_selection_probability_l2298_229802

/-- The probability of selecting exactly one shirt, one pair of shorts, one pair of socks, and one hat
    when randomly choosing 4 articles of clothing from a drawer containing 6 shirts, 7 pairs of shorts,
    8 pairs of socks, and 3 hats. -/
theorem clothing_selection_probability :
  let num_shirts : ℕ := 6
  let num_shorts : ℕ := 7
  let num_socks : ℕ := 8
  let num_hats : ℕ := 3
  let total_items : ℕ := num_shirts + num_shorts + num_socks + num_hats
  let favorable_outcomes : ℕ := num_shirts * num_shorts * num_socks * num_hats
  let total_outcomes : ℕ := Nat.choose total_items 4
  (favorable_outcomes : ℚ) / total_outcomes = 144 / 1815 := by
  sorry


end clothing_selection_probability_l2298_229802


namespace exists_terrorist_with_eleven_raids_l2298_229819

/-- Represents a terrorist in the band -/
structure Terrorist : Type :=
  (id : Nat)

/-- Represents a raid -/
structure Raid : Type :=
  (id : Nat)

/-- Represents the participation of a terrorist in a raid -/
def Participation : Type := Terrorist → Raid → Prop

/-- The total number of terrorists in the band -/
def num_terrorists : Nat := 101

/-- Axiom: Each pair of terrorists has met exactly once in a raid -/
axiom met_once (p : Participation) (t1 t2 : Terrorist) :
  t1 ≠ t2 → ∃! r : Raid, p t1 r ∧ p t2 r

/-- Axiom: No two terrorists have participated in more than one raid together -/
axiom no_multiple_raids (p : Participation) (t1 t2 : Terrorist) (r1 r2 : Raid) :
  t1 ≠ t2 → p t1 r1 → p t2 r1 → p t1 r2 → p t2 r2 → r1 = r2

/-- Theorem: There exists a terrorist who participated in at least 11 different raids -/
theorem exists_terrorist_with_eleven_raids (p : Participation) :
  ∃ t : Terrorist, ∃ (raids : Finset Raid), raids.card ≥ 11 ∧ ∀ r ∈ raids, p t r :=
sorry

end exists_terrorist_with_eleven_raids_l2298_229819


namespace chess_tournament_players_l2298_229809

theorem chess_tournament_players (total_games : ℕ) (h_total_games : total_games = 42) : 
  ∃ n : ℕ, n > 0 ∧ total_games = n * (n - 1) ∧ n = 7 :=
by sorry

end chess_tournament_players_l2298_229809


namespace multiply_divide_sqrt_l2298_229874

theorem multiply_divide_sqrt (x y : ℝ) (hx : x = 1.4) (hx_neq_zero : x ≠ 0) :
  Real.sqrt ((x * y) / 5) = x → y = 7 := by
  sorry

end multiply_divide_sqrt_l2298_229874


namespace sqrt_sum_equals_2sqrt10_l2298_229808

theorem sqrt_sum_equals_2sqrt10 : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 10 := by
  sorry

end sqrt_sum_equals_2sqrt10_l2298_229808


namespace system_solutions_l2298_229804

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * (x^2 - 3*y^2) = 16
def equation2 (x y : ℝ) : Prop := y * (3*x^2 - y^2) = 88

-- Define the approximate equality for real numbers
def approx_equal (a b : ℝ) (ε : ℝ) : Prop := abs (a - b) < ε

-- Theorem statement
theorem system_solutions :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    -- Exact solution
    equation1 x₁ y₁ ∧ equation2 x₁ y₁ ∧ x₁ = 4 ∧ y₁ = 2 ∧
    -- Approximate solutions
    equation1 x₂ y₂ ∧ equation2 x₂ y₂ ∧ 
    approx_equal x₂ (-3.7) 0.1 ∧ approx_equal y₂ 2.5 0.1 ∧
    equation1 x₃ y₃ ∧ equation2 x₃ y₃ ∧ 
    approx_equal x₃ (-0.3) 0.1 ∧ approx_equal y₃ (-4.5) 0.1 :=
by sorry


end system_solutions_l2298_229804


namespace fraction_equality_l2298_229860

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 3 * y) / (3 * x - y) = 16 / 13 := by
sorry

end fraction_equality_l2298_229860


namespace detergent_water_ratio_change_l2298_229834

-- Define the original ratio
def original_ratio : Fin 3 → ℚ
  | 0 => 2  -- bleach
  | 1 => 40 -- detergent
  | 2 => 100 -- water

-- Define the altered ratio
def altered_ratio : Fin 3 → ℚ
  | 0 => 6  -- bleach (tripled)
  | 1 => 40 -- detergent
  | 2 => 200 -- water

-- Theorem to prove
theorem detergent_water_ratio_change :
  (altered_ratio 1 / altered_ratio 2) / (original_ratio 1 / original_ratio 2) = 1/2 := by
  sorry

end detergent_water_ratio_change_l2298_229834


namespace soccer_balls_count_l2298_229873

/-- The cost of a football in dollars -/
def football_cost : ℝ := 35

/-- The cost of a soccer ball in dollars -/
def soccer_ball_cost : ℝ := 50

/-- The cost of 2 footballs and some soccer balls in dollars -/
def first_set_cost : ℝ := 220

/-- The cost of 3 footballs and 1 soccer ball in dollars -/
def second_set_cost : ℝ := 155

/-- The number of soccer balls in the second set -/
def soccer_balls_in_second_set : ℕ := 1

theorem soccer_balls_count : 
  2 * football_cost + soccer_balls_in_second_set * soccer_ball_cost = first_set_cost ∧
  3 * football_cost + soccer_ball_cost = second_set_cost →
  soccer_balls_in_second_set = 1 := by
  sorry

end soccer_balls_count_l2298_229873
