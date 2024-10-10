import Mathlib

namespace base_six_conversion_and_addition_l483_48310

def base_six_to_ten (n : Nat) : Nat :=
  (n % 10) + 6 * ((n / 10) % 10) + 36 * (n / 100)

theorem base_six_conversion_and_addition :
  let base_six_num := 214
  let base_ten_num := base_six_to_ten base_six_num
  base_ten_num = 82 ∧ base_ten_num + 15 = 97 := by
  sorry

end base_six_conversion_and_addition_l483_48310


namespace stock_certificate_tearing_impossible_2002_pieces_l483_48363

theorem stock_certificate_tearing (n : ℕ) : n > 0 → (∃ k : ℕ, n = 1 + 7 * k) ↔ n % 7 = 1 :=
by sorry

theorem impossible_2002_pieces : ¬(∃ k : ℕ, 2002 = 1 + 7 * k) :=
by sorry

end stock_certificate_tearing_impossible_2002_pieces_l483_48363


namespace A_empty_A_singleton_A_at_most_one_A_element_when_zero_A_element_when_nine_eighths_l483_48314

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

-- Theorem 1: A is empty iff a > 9/8
theorem A_empty (a : ℝ) : A a = ∅ ↔ a > 9/8 := by sorry

-- Theorem 2: A contains exactly one element iff a = 0 or a = 9/8
theorem A_singleton (a : ℝ) : (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 9/8 := by sorry

-- Theorem 3: A contains at most one element iff a = 0 or a ≥ 9/8
theorem A_at_most_one (a : ℝ) : (∀ x y, x ∈ A a → y ∈ A a → x = y) ↔ a = 0 ∨ a ≥ 9/8 := by sorry

-- Additional theorems for specific elements when A is a singleton
theorem A_element_when_zero : (∀ x, x ∈ A 0 ↔ x = 2/3) := by sorry

theorem A_element_when_nine_eighths : (∀ x, x ∈ A (9/8) ↔ x = 4/3) := by sorry

end A_empty_A_singleton_A_at_most_one_A_element_when_zero_A_element_when_nine_eighths_l483_48314


namespace combined_average_score_l483_48366

theorem combined_average_score (score_a score_b score_c : ℝ)
  (ratio_a ratio_b ratio_c : ℕ) :
  score_a = 65 →
  score_b = 90 →
  score_c = 77 →
  ratio_a = 4 →
  ratio_b = 6 →
  ratio_c = 5 →
  (ratio_a * score_a + ratio_b * score_b + ratio_c * score_c) / (ratio_a + ratio_b + ratio_c) = 79 := by
  sorry

end combined_average_score_l483_48366


namespace flight_departure_requirement_l483_48352

/-- The minimum number of people required for the flight to depart -/
def min_required : ℕ := 16

/-- The number of people currently on the plane -/
def current_people : ℕ := 9

/-- The number of additional people needed to board before departure -/
def additional_people : ℕ := min_required - current_people

theorem flight_departure_requirement :
  min_required > 15 ∧ current_people = 9 → additional_people = 7 := by
  sorry

end flight_departure_requirement_l483_48352


namespace composite_divisibility_l483_48398

theorem composite_divisibility (k : ℕ) (p_k : ℕ) (n : ℕ) 
  (h1 : k ≥ 14)
  (h2 : Nat.Prime p_k)
  (h3 : p_k < k)
  (h4 : ∀ p, Nat.Prime p → p < k → p ≤ p_k)
  (h5 : p_k ≥ 3 * k / 4)
  (h6 : ¬ Nat.Prime n) :
  (n = 2 * p_k → ¬ (n ∣ Nat.factorial (n - k))) ∧
  (n > 2 * p_k → n ∣ Nat.factorial (n - k)) := by
  sorry

end composite_divisibility_l483_48398


namespace third_to_fourth_l483_48364

/-- An angle is in the third quadrant if it's between 180° and 270° -/
def is_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 180 + 360 * k < α ∧ α < 270 + 360 * k

/-- An angle is in the fourth quadrant if it's between 270° and 360° -/
def is_fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 270 + 360 * k < α ∧ α < 360 + 360 * k

theorem third_to_fourth (α : ℝ) (h : is_third_quadrant α) :
  is_fourth_quadrant (180 - α) :=
by sorry

end third_to_fourth_l483_48364


namespace connie_savings_theorem_connie_savings_value_l483_48370

/-- The amount of money Connie saved up -/
def connie_savings : ℕ := sorry

/-- The cost of the watch -/
def watch_cost : ℕ := 55

/-- The additional amount Connie needs -/
def additional_needed : ℕ := 16

/-- Theorem stating that Connie's savings plus the additional amount needed equals the watch cost -/
theorem connie_savings_theorem : connie_savings + additional_needed = watch_cost := by sorry

/-- Theorem proving that Connie's savings equal $39 -/
theorem connie_savings_value : connie_savings = 39 := by sorry

end connie_savings_theorem_connie_savings_value_l483_48370


namespace flour_weight_acceptable_l483_48333

/-- A weight is acceptable if it falls within the labeled range -/
def is_acceptable (labeled_weight : ℝ) (tolerance : ℝ) (actual_weight : ℝ) : Prop :=
  actual_weight ≥ labeled_weight - tolerance ∧ actual_weight ≤ labeled_weight + tolerance

/-- Theorem stating that 99.80 kg is acceptable for a bag labeled as 100 ± 0.25 kg -/
theorem flour_weight_acceptable :
  is_acceptable 100 0.25 99.80 := by
  sorry

end flour_weight_acceptable_l483_48333


namespace trajectory_of_point_l483_48323

/-- The trajectory of a point P satisfying |PF₁| + |PF₂| = 8, where F₁ and F₂ are fixed points -/
theorem trajectory_of_point (P : ℝ × ℝ) : 
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  (dist P F₁ + dist P F₂ = 8) → 
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂ :=
by sorry


end trajectory_of_point_l483_48323


namespace bus_fraction_is_two_thirds_l483_48386

def total_distance : ℝ := 30.000000000000007

theorem bus_fraction_is_two_thirds :
  let foot_distance := (1 / 5 : ℝ) * total_distance
  let car_distance := 4
  let bus_distance := total_distance - (foot_distance + car_distance)
  bus_distance / total_distance = 2 / 3 := by sorry

end bus_fraction_is_two_thirds_l483_48386


namespace solution_sum_l483_48378

-- Define the solution set for |2x-3| ≤ 1
def solution_set (m n : ℝ) : Prop :=
  ∀ x, |2*x - 3| ≤ 1 ↔ m ≤ x ∧ x ≤ n

-- Theorem statement
theorem solution_sum (m n : ℝ) : solution_set m n → m + n = 3 := by
  sorry

end solution_sum_l483_48378


namespace square_equation_solution_l483_48348

theorem square_equation_solution : ∃! (M : ℕ), M > 0 ∧ 16^2 * 40^2 = 20^2 * M^2 ∧ M = 32 := by sorry

end square_equation_solution_l483_48348


namespace connected_vertices_probability_is_correct_l483_48321

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 20
  edge_count : edges.card = 30
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of choosing at least two connected vertices when selecting three random vertices -/
def connected_vertices_probability (d : RegularDodecahedron) : ℚ :=
  9 / 19

/-- Theorem stating the probability of choosing at least two connected vertices -/
theorem connected_vertices_probability_is_correct (d : RegularDodecahedron) :
  connected_vertices_probability d = 9 / 19 := by
  sorry


end connected_vertices_probability_is_correct_l483_48321


namespace complex_number_equality_l483_48388

theorem complex_number_equality (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end complex_number_equality_l483_48388


namespace aquarium_original_price_l483_48311

/-- Proves that the original price of an aquarium is $120 given the conditions of the problem -/
theorem aquarium_original_price (P : ℝ) : 
  (0.5 * P + 0.05 * (0.5 * P) = 63) → P = 120 := by
  sorry

end aquarium_original_price_l483_48311


namespace right_triangle_hypotenuse_l483_48390

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 75 → b = 100 → c^2 = a^2 + b^2 → c = 125 := by
  sorry

end right_triangle_hypotenuse_l483_48390


namespace infinite_series_sum_l483_48391

/-- The sum of the infinite series ∑(n=1 to ∞) (4n^2 - 2n + 1) / 3^n is equal to 5 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (4 * n^2 - 2 * n + 1) / 3^n) = 5 := by
  sorry

end infinite_series_sum_l483_48391


namespace probability_mathematics_letter_l483_48330

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

def unique_letters (s : String) : Finset Char :=
  s.toList.toFinset

theorem probability_mathematics_letter : 
  (unique_letters mathematics).card / alphabet.card = 4 / 13 := by
  sorry

end probability_mathematics_letter_l483_48330


namespace sunzi_problem_correct_l483_48339

/-- Represents the problem from "The Mathematical Classic of Sunzi" -/
structure SunziProblem where
  x : ℕ  -- Total number of people
  y : ℕ  -- Total number of carriages

/-- Checks if the given numbers satisfy the conditions of the problem -/
def is_valid_solution (p : SunziProblem) : Prop :=
  (p.x / 3 : ℚ) = p.y - 2 ∧ (p.x - 9) / 2 = p.y

/-- The system of equations correctly represents the Sunzi problem -/
theorem sunzi_problem_correct (p : SunziProblem) : 
  is_valid_solution p ↔ 
    (∃ (empty_carriages : ℕ), p.y = p.x / 3 + empty_carriages ∧ empty_carriages = 2) ∧
    (∃ (walking_people : ℕ), p.y = (p.x - walking_people) / 2 ∧ walking_people = 9) :=
sorry

end sunzi_problem_correct_l483_48339


namespace cookie_rows_per_tray_l483_48312

/-- Given the total number of cookies, cookies per row, and number of baking trays,
    calculate the number of rows of cookies on each baking tray. -/
def rows_per_tray (total_cookies : ℕ) (cookies_per_row : ℕ) (num_trays : ℕ) : ℕ :=
  (total_cookies / cookies_per_row) / num_trays

/-- Theorem stating that with 120 total cookies, 6 cookies per row, and 4 baking trays,
    there are 5 rows of cookies on each baking tray. -/
theorem cookie_rows_per_tray :
  rows_per_tray 120 6 4 = 5 := by
  sorry

end cookie_rows_per_tray_l483_48312


namespace complement_of_A_when_a_is_one_range_of_a_given_subset_l483_48307

-- Define set A
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 8*a^2 ≤ 0}

-- Part I
theorem complement_of_A_when_a_is_one :
  Set.compl (A 1) = {x | x < -2 ∨ x > 4} := by sorry

-- Part II
theorem range_of_a_given_subset (a : ℝ) (h1 : a > 0) (h2 : Set.Ioo (-1 : ℝ) 1 ⊆ A a) :
  a ≥ 1/2 := by sorry

end complement_of_A_when_a_is_one_range_of_a_given_subset_l483_48307


namespace coffee_maker_price_l483_48347

/-- The final price of a coffee maker after applying a discount -/
def final_price (original_price discount : ℕ) : ℕ :=
  original_price - discount

/-- Theorem: The customer pays $70 for a coffee maker with original price $90 and a $20 discount -/
theorem coffee_maker_price :
  final_price 90 20 = 70 := by
  sorry

end coffee_maker_price_l483_48347


namespace aira_rubber_bands_l483_48357

theorem aira_rubber_bands (samantha aira joe : ℕ) : 
  samantha = aira + 5 →
  joe = aira + 1 →
  samantha + aira + joe = 18 →
  aira = 4 := by
sorry

end aira_rubber_bands_l483_48357


namespace ellipse_parabola_intersection_hyperbola_l483_48397

/-- Given an ellipse and a parabola that intersect, prove that the radius of the circumcircle
    of the triangle formed by their intersection points and the origin, along with the parameter
    of the parabola, satisfy a hyperbolic equation. -/
theorem ellipse_parabola_intersection_hyperbola (p r : ℝ) (hp : p > 0) (hr : r > 0) :
  (∃ x y : ℝ, x^2/4 + y^2/2 = 1 ∧ y^2 = 2*p*x) →
  (∃ x₀ y₀ : ℝ, x₀^2/4 + y₀^2/2 = 1 ∧ y₀^2 = 2*p*x₀ ∧ x₀^2 + y₀^2 = r^2) →
  r^2 - p^2 = 1 :=
sorry

end ellipse_parabola_intersection_hyperbola_l483_48397


namespace polynomial_division_degree_l483_48343

theorem polynomial_division_degree (f d q r : Polynomial ℝ) : 
  Polynomial.degree f = 15 →
  Polynomial.degree q = 8 →
  Polynomial.degree r = 2 →
  f = d * q + r →
  Polynomial.degree r < Polynomial.degree d →
  Polynomial.degree d = 7 := by
sorry

end polynomial_division_degree_l483_48343


namespace milburg_children_l483_48393

/-- The number of children in Milburg -/
def children : ℕ := 8243 - 5256

/-- The total population of Milburg -/
def total_population : ℕ := 8243

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := 5256

theorem milburg_children : children = 2987 := by
  sorry

end milburg_children_l483_48393


namespace a_equals_one_m_geq_two_l483_48337

/-- The function f defined as f(x) = |x + 2a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

/-- Theorem stating that given the conditions, a must equal 1 -/
theorem a_equals_one (a : ℝ) : 
  (∀ x, f a x < 4 - 2*a ↔ -4 < x ∧ x < 0) → a = 1 := by sorry

/-- The function g defined as g(x) = |x + 2| -/
def g (x : ℝ) : ℝ := |x + 2|

/-- Theorem stating that given the conditions, m must be greater than or equal to 2 -/
theorem m_geq_two (m : ℝ) :
  (∀ x, g x - g (-2*x) ≤ x + m) → m ≥ 2 := by sorry

end a_equals_one_m_geq_two_l483_48337


namespace initial_tree_height_l483_48396

/-- Represents the growth of a tree over time -/
def TreeGrowth (initial_height growth_rate years final_height : ℝ) : Prop :=
  initial_height + growth_rate * years = final_height

theorem initial_tree_height : 
  ∃ (h : ℝ), TreeGrowth h 5 5 29 ∧ h = 4 := by sorry

end initial_tree_height_l483_48396


namespace greatest_common_factor_36_54_81_l483_48338

theorem greatest_common_factor_36_54_81 : Nat.gcd 36 (Nat.gcd 54 81) = 9 := by
  sorry

end greatest_common_factor_36_54_81_l483_48338


namespace wendy_extraction_cost_l483_48394

/-- The cost of a dental cleaning in dollars -/
def cleaning_cost : ℕ := 70

/-- The cost of a dental filling in dollars -/
def filling_cost : ℕ := 120

/-- The number of fillings Wendy had -/
def num_fillings : ℕ := 2

/-- The total cost of Wendy's dental bill in dollars -/
def total_bill : ℕ := 5 * filling_cost

/-- The cost of Wendy's tooth extraction in dollars -/
def extraction_cost : ℕ := total_bill - (cleaning_cost + num_fillings * filling_cost)

theorem wendy_extraction_cost : extraction_cost = 290 := by
  sorry

end wendy_extraction_cost_l483_48394


namespace stratified_sampling_group_c_l483_48335

/-- Represents the number of cities to be selected from a group in a stratified sampling -/
def citiesSelected (totalSelected : ℕ) (totalCities : ℕ) (groupCities : ℕ) : ℕ :=
  (totalSelected * groupCities) / totalCities

theorem stratified_sampling_group_c (totalCities : ℕ) (groupACities : ℕ) (groupBCities : ℕ) 
    (totalSelected : ℕ) (hTotal : totalCities = 48) (hA : groupACities = 8) (hB : groupBCities = 24) 
    (hSelected : totalSelected = 12) :
    citiesSelected totalSelected totalCities (totalCities - groupACities - groupBCities) = 4 := by
  sorry

end stratified_sampling_group_c_l483_48335


namespace high_confidence_possible_no_cases_l483_48340

/-- Represents the confidence level of the relationship between smoking and lung cancer -/
def confidence_level : ℝ := 0.99

/-- Represents a sample of smokers -/
def sample_size : ℕ := 100

/-- Represents the possibility of having no lung cancer cases in a sample -/
def possible_no_cases : Prop := true

/-- Theorem stating that despite high confidence in the smoking-lung cancer relationship,
    it's possible to have a sample with no lung cancer cases -/
theorem high_confidence_possible_no_cases :
  confidence_level > 0.99 → possible_no_cases := by sorry

end high_confidence_possible_no_cases_l483_48340


namespace real_solutions_quadratic_l483_48354

theorem real_solutions_quadratic (x : ℝ) :
  (∃ y : ℝ, 9 * y^2 + 6 * x * y + 2 * x + 1 = 0) ↔ (x < 2 - Real.sqrt 6 ∨ x > 2 + Real.sqrt 6) :=
sorry

end real_solutions_quadratic_l483_48354


namespace g_2010_value_l483_48376

-- Define the property of the function g
def g_property (g : ℕ → ℝ) : Prop :=
  ∀ x y m : ℕ, x > 0 → y > 0 → m > 0 → x + y = 2^m → g x + g y = ((m + 1) : ℝ)^2

-- Theorem statement
theorem g_2010_value (g : ℕ → ℝ) (h : g_property g) : g 2010 = 126 := by
  sorry

end g_2010_value_l483_48376


namespace max_value_constraint_l483_48316

theorem max_value_constraint (x y : ℝ) (h : 2 * x^2 + 3 * y^2 ≤ 12) :
  |x + 2*y| ≤ Real.sqrt 22 ∧ ∃ x y : ℝ, 2 * x^2 + 3 * y^2 = 12 ∧ |x + 2*y| = Real.sqrt 22 := by
  sorry

end max_value_constraint_l483_48316


namespace sum_is_five_digits_l483_48320

/-- Represents a digit (1-9) -/
def Digit := Fin 9

/-- The sum of 10765, AB4, and CB is always a 5-digit number -/
theorem sum_is_five_digits (A B C : Digit) (h : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  let AB4 := 100 * A.val + 10 * B.val + 4
  let CB := 10 * C.val + B.val
  let sum := 10765 + AB4 + CB
  9999 < sum ∧ sum < 100000 := by
  sorry

end sum_is_five_digits_l483_48320


namespace job_completion_time_specific_job_completion_time_l483_48318

/-- 
If one person completes a job in a days and another person completes the same job in b days,
then together they will complete the job in (a * b) / (a + b) days.
-/
theorem job_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let combined_time := (a * b) / (a + b)
  combined_time = (a⁻¹ + b⁻¹)⁻¹ :=
by sorry

/--
If one person completes a job in 8 days and another person completes the same job in 24 days,
then together they will complete the job in 6 days.
-/
theorem specific_job_completion_time :
  let a := 8
  let b := 24
  let combined_time := (a * b) / (a + b)
  combined_time = 6 :=
by sorry

end job_completion_time_specific_job_completion_time_l483_48318


namespace quadratic_discriminant_zero_l483_48362

theorem quadratic_discriminant_zero (b : ℝ) : 
  (∀ x, 3 * x^2 + 5 * b * x + 7 = 0 → (5 * b)^2 - 4 * 3 * 7 = 0) → 
  b = 2 * Real.sqrt 21 ∨ b = -2 * Real.sqrt 21 := by
  sorry

end quadratic_discriminant_zero_l483_48362


namespace sqrt_21_times_sqrt_7_minus_sqrt_3_l483_48356

theorem sqrt_21_times_sqrt_7_minus_sqrt_3 :
  Real.sqrt 21 * Real.sqrt 7 - Real.sqrt 3 = 6 * Real.sqrt 3 := by
  sorry

end sqrt_21_times_sqrt_7_minus_sqrt_3_l483_48356


namespace normal_distribution_symmetry_l483_48313

-- Define a random variable following normal distribution
def normalDistribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def probability (ξ : normalDistribution 4 σ) (event : Set ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_symmetry 
  (σ : ℝ) 
  (ξ : normalDistribution 4 σ) 
  (h : probability ξ {x | x > 8} = 0.4) : 
  probability ξ {x | x < 0} = 0.4 :=
by sorry

end normal_distribution_symmetry_l483_48313


namespace no_real_roots_for_nonzero_k_l483_48300

theorem no_real_roots_for_nonzero_k (k : ℝ) (hk : k ≠ 0) :
  ∀ x : ℝ, x^2 + k*x + 2*k^2 ≠ 0 := by
  sorry

end no_real_roots_for_nonzero_k_l483_48300


namespace calculate_expression_l483_48382

theorem calculate_expression : (1000 * 0.09999) / 10 * 999 = 998001 := by
  sorry

end calculate_expression_l483_48382


namespace not_p_sufficient_not_necessary_for_q_l483_48334

def p (x : ℝ) : Prop := |x - 4| > 2

def q (x : ℝ) : Prop := x > 1

def not_p (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 6

theorem not_p_sufficient_not_necessary_for_q :
  (∀ x, not_p x → q x) ∧ ¬(∀ x, q x → not_p x) := by sorry

end not_p_sufficient_not_necessary_for_q_l483_48334


namespace gem_stone_necklaces_sold_megan_sold_three_gem_stone_necklaces_l483_48325

/-- The number of gem stone necklaces sold at a garage sale -/
theorem gem_stone_necklaces_sold (bead_necklaces : ℕ) (price_per_necklace : ℕ) (total_earnings : ℕ) : ℕ :=
  let gem_stone_necklaces := (total_earnings - bead_necklaces * price_per_necklace) / price_per_necklace
  gem_stone_necklaces

/-- Proof that Megan sold 3 gem stone necklaces -/
theorem megan_sold_three_gem_stone_necklaces : 
  gem_stone_necklaces_sold 7 9 90 = 3 := by
  sorry

end gem_stone_necklaces_sold_megan_sold_three_gem_stone_necklaces_l483_48325


namespace imaginary_power_2016_l483_48392

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_2016 : i ^ 2016 = 1 := by
  sorry

end imaginary_power_2016_l483_48392


namespace number_calculation_l483_48374

theorem number_calculation (number : ℝ) : 
  (number / 0.3 = 0.03) → number = 0.009 := by
  sorry

end number_calculation_l483_48374


namespace bills_score_l483_48304

theorem bills_score (john sue bill : ℕ) 
  (h1 : bill = john + 20)
  (h2 : bill * 2 = sue)
  (h3 : john + bill + sue = 160) :
  bill = 45 := by
sorry

end bills_score_l483_48304


namespace sum_factorials_mod_30_l483_48349

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem sum_factorials_mod_30 :
  sum_factorials 10 % 30 = 3 := by sorry

end sum_factorials_mod_30_l483_48349


namespace toy_purchase_cost_l483_48346

theorem toy_purchase_cost (yoyo_cost whistle_cost : ℕ) 
  (h1 : yoyo_cost = 24) 
  (h2 : whistle_cost = 14) : 
  yoyo_cost + whistle_cost = 38 := by
  sorry

end toy_purchase_cost_l483_48346


namespace father_age_twice_marika_l483_48383

/-- Marika's birth year -/
def marika_birth_year : ℕ := 1996

/-- Marika's father's birth year -/
def father_birth_year : ℕ := 1956

/-- The year when the father's age is twice Marika's age -/
def target_year : ℕ := 2036

theorem father_age_twice_marika (year : ℕ) :
  year = target_year ↔ 
  (year - father_birth_year = 2 * (year - marika_birth_year)) ∧
  (year > marika_birth_year) ∧
  (year > father_birth_year) := by
sorry

end father_age_twice_marika_l483_48383


namespace wait_probability_is_two_thirds_l483_48372

/-- The duration of the red light in seconds -/
def red_light_duration : ℕ := 30

/-- The minimum waiting time in seconds -/
def min_wait_time : ℕ := 10

/-- The probability of waiting at least 'min_wait_time' seconds for the green light -/
def wait_probability : ℚ := (red_light_duration - min_wait_time) / red_light_duration

theorem wait_probability_is_two_thirds : 
  wait_probability = 2 / 3 := by sorry

end wait_probability_is_two_thirds_l483_48372


namespace polynomial_three_distinct_roots_l483_48317

theorem polynomial_three_distinct_roots : 
  ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∀ (x : ℝ), (x - 4) * (x^2 + 4*x + 3) = 0 ↔ x = a ∨ x = b ∨ x = c :=
by sorry

end polynomial_three_distinct_roots_l483_48317


namespace least_n_satisfying_inequality_five_satisfies_inequality_least_n_is_five_l483_48389

theorem least_n_satisfying_inequality :
  ∀ n : ℕ+, n < 5 → (1 : ℚ) / n.val - (1 : ℚ) / (n.val + 2) ≥ (1 : ℚ) / 15 :=
by sorry

theorem five_satisfies_inequality :
  (1 : ℚ) / 5 - (1 : ℚ) / 7 < (1 : ℚ) / 15 :=
by sorry

theorem least_n_is_five :
  ∃! (n : ℕ+), 
    ((1 : ℚ) / n.val - (1 : ℚ) / (n.val + 2) < (1 : ℚ) / 15) ∧
    (∀ m : ℕ+, m < n → (1 : ℚ) / m.val - (1 : ℚ) / (m.val + 2) ≥ (1 : ℚ) / 15) :=
by sorry

end least_n_satisfying_inequality_five_satisfies_inequality_least_n_is_five_l483_48389


namespace supplement_complement_difference_l483_48345

/-- An acute angle is between 0° and 90° -/
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

/-- The supplement of an angle θ -/
def supplement (θ : ℝ) : ℝ := 180 - θ

/-- The complement of an angle θ -/
def complement (θ : ℝ) : ℝ := 90 - θ

/-- For any acute angle, the difference between its supplement and complement is 90° -/
theorem supplement_complement_difference (θ : ℝ) (h : is_acute_angle θ) :
  supplement θ - complement θ = 90 := by
  sorry

end supplement_complement_difference_l483_48345


namespace roots_condition_implies_a_equals_neg_nine_l483_48324

/-- The polynomial p(x) = x³ - 6x² + ax + a, where a is a parameter --/
def p (a : ℝ) (x : ℝ) : ℝ := x^3 - 6*x^2 + a*x + a

/-- The condition that the sum of cubes of the roots minus 3 is zero --/
def sum_of_cubes_minus_3_is_zero (x₁ x₂ x₃ : ℝ) : Prop :=
  (x₁ - 3)^3 + (x₂ - 3)^3 + (x₃ - 3)^3 = 0

theorem roots_condition_implies_a_equals_neg_nine (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 
    (∀ x : ℝ, p a x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    sum_of_cubes_minus_3_is_zero x₁ x₂ x₃) →
  a = -9 :=
sorry

end roots_condition_implies_a_equals_neg_nine_l483_48324


namespace number_game_l483_48387

theorem number_game (x : ℤ) : 3 * (3 * (x + 3) - 3) = 3 * (3 * x) := by
  sorry

end number_game_l483_48387


namespace multiple_of_smaller_number_l483_48385

theorem multiple_of_smaller_number (S L k : ℤ) : 
  S = 14 → 
  L = k * S - 3 → 
  S + L = 39 → 
  k = 2 := by
sorry

end multiple_of_smaller_number_l483_48385


namespace max_value_constraint_l483_48301

theorem max_value_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 * a + 6 * b < 110) :
  a * b * (110 - 5 * a - 6 * b) ≤ 1331000 / 810 := by
sorry

end max_value_constraint_l483_48301


namespace hash_solution_l483_48399

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem hash_solution :
  ∃ (A : ℝ), (hash A 7 = 225) ∧ (A = 7 * Real.sqrt 3) := by
  sorry

end hash_solution_l483_48399


namespace integer_fraction_pairs_l483_48336

theorem integer_fraction_pairs : 
  {p : ℕ × ℕ | p.1 > 0 ∧ p.2 > 0 ∧ (∃ k : ℤ, (p.2^3 + 1 : ℤ) = k * (p.1 * p.2 - 1))} = 
  {(2,1), (3,1), (2,2), (5,2), (5,3), (2,5), (3,5)} := by
sorry

end integer_fraction_pairs_l483_48336


namespace corn_acreage_l483_48332

/-- Given a total of 1034 acres of land divided among beans, wheat, and corn
    in the ratio of 5:2:4, prove that the number of acres used for corn is 376. -/
theorem corn_acreage (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
    (h1 : total_land = 1034)
    (h2 : beans_ratio = 5)
    (h3 : wheat_ratio = 2)
    (h4 : corn_ratio = 4) :
    (corn_ratio * total_land) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry


end corn_acreage_l483_48332


namespace arrangement_remainder_l483_48377

/-- The number of green marbles --/
def green_marbles : ℕ := 7

/-- The maximum number of blue marbles satisfying the arrangement condition --/
def max_blue_marbles : ℕ := 19

/-- The total number of marbles --/
def total_marbles : ℕ := green_marbles + max_blue_marbles

/-- The number of ways to arrange the marbles --/
def arrangement_count : ℕ := Nat.choose total_marbles green_marbles

/-- Theorem stating the remainder when the number of arrangements is divided by 500 --/
theorem arrangement_remainder : arrangement_count % 500 = 30 := by sorry

end arrangement_remainder_l483_48377


namespace equal_vectors_have_equal_magnitudes_l483_48353

theorem equal_vectors_have_equal_magnitudes {V : Type*} [NormedAddCommGroup V] 
  {a b : V} (h : a = b) : ‖a‖ = ‖b‖ := by
  sorry

end equal_vectors_have_equal_magnitudes_l483_48353


namespace sum_of_ages_l483_48381

/-- Given that Jed is 10 years older than Matt and in 10 years, Jed will be 25 years old,
    prove that the sum of their present ages is 20. -/
theorem sum_of_ages (jed_age matt_age : ℕ) : 
  jed_age = matt_age + 10 →  -- Jed is 10 years older than Matt
  jed_age + 10 = 25 →        -- In 10 years, Jed will be 25 years old
  jed_age + matt_age = 20 :=   -- The sum of their present ages is 20
by sorry

end sum_of_ages_l483_48381


namespace intersection_of_M_and_N_l483_48367

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end intersection_of_M_and_N_l483_48367


namespace tv_price_change_l483_48380

theorem tv_price_change (x : ℝ) : 
  (1 - x / 100) * (1 + 40 / 100) = 1 + 12 / 100 → x = 20 := by
  sorry

end tv_price_change_l483_48380


namespace cone_apex_angle_l483_48326

theorem cone_apex_angle (α β : ℝ) : 
  β = Real.arcsin (1/4) →
  2 * α = Real.arcsin (2 * Real.sin β) + β →
  2 * α = π/6 + Real.arcsin (1/4) :=
by sorry

end cone_apex_angle_l483_48326


namespace mixed_doubles_groupings_l483_48305

theorem mixed_doubles_groupings (male_players : Nat) (female_players : Nat) :
  male_players = 5 → female_players = 3 →
  (Nat.choose male_players 2) * (Nat.choose female_players 2) * (Nat.factorial 2) = 60 :=
by sorry

end mixed_doubles_groupings_l483_48305


namespace intersection_M_N_l483_48351

def M : Set ℝ := {x | x ≤ 0}
def N : Set ℝ := {-2, 0, 1}

theorem intersection_M_N : M ∩ N = {-2, 0} := by sorry

end intersection_M_N_l483_48351


namespace total_snowballs_l483_48350

def lucy_snowballs : ℕ := 19
def peter_snowballs : ℕ := 47
def charlie_snowballs : ℕ := lucy_snowballs + 31

theorem total_snowballs : lucy_snowballs + charlie_snowballs + peter_snowballs = 116 := by
  sorry

end total_snowballs_l483_48350


namespace pencil_multiple_l483_48369

theorem pencil_multiple (reeta_pencils : ℕ) (total_pencils : ℕ) (anika_pencils : ℕ → ℕ) :
  reeta_pencils = 20 →
  total_pencils = 64 →
  (∀ M : ℕ, anika_pencils M = 20 * M + 4) →
  ∃ M : ℕ, M = 2 ∧ anika_pencils M + reeta_pencils = total_pencils :=
by sorry

end pencil_multiple_l483_48369


namespace christmas_play_volunteers_l483_48368

theorem christmas_play_volunteers 
  (total_needed : ℕ) 
  (num_classes : ℕ) 
  (teachers_volunteered : ℕ) 
  (more_needed : ℕ) 
  (h1 : total_needed = 50) 
  (h2 : num_classes = 6) 
  (h3 : teachers_volunteered = 13) 
  (h4 : more_needed = 7) :
  (total_needed - teachers_volunteered - more_needed) / num_classes = 5 := by
  sorry

end christmas_play_volunteers_l483_48368


namespace max_members_is_414_l483_48327

/-- The number of members in the dance group. -/
def m : ℕ := 414

/-- Represents the condition that when arranged in a square formation, there are 11 members left over. -/
def square_formation_condition (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k^2 + 11

/-- Represents the condition that when arranged in a formation with 5 more rows than columns, there are no members left over. -/
def rectangular_formation_condition (m : ℕ) : Prop :=
  ∃ n : ℕ, m = n * (n + 5)

/-- Theorem stating that 414 is the maximum number of members satisfying both conditions. -/
theorem max_members_is_414 :
  square_formation_condition m ∧
  rectangular_formation_condition m ∧
  ∀ x > m, ¬(square_formation_condition x ∧ rectangular_formation_condition x) :=
by sorry

end max_members_is_414_l483_48327


namespace total_digits_in_books_l483_48302

/-- Calculate the number of digits used to number pages in a book -/
def digitsInBook (pages : ℕ) : ℕ :=
  let singleDigitPages := min pages 9
  let doubleDigitPages := min (pages - 9) 90
  let tripleDigitPages := min (pages - 99) 900
  let quadrupleDigitPages := max (pages - 999) 0
  singleDigitPages * 1 +
  doubleDigitPages * 2 +
  tripleDigitPages * 3 +
  quadrupleDigitPages * 4

/-- The total number of digits used to number pages in the collection of books -/
def totalDigits : ℕ :=
  digitsInBook 450 + digitsInBook 675 + digitsInBook 1125 + digitsInBook 2430

theorem total_digits_in_books :
  totalDigits = 15039 := by sorry

end total_digits_in_books_l483_48302


namespace circle_C_properties_line_l_property_circle_E_fixed_points_l483_48371

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 4)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - 2*y + 4 = 0

-- Define the circle E
def circle_E (x y y1 y2 : ℝ) : Prop :=
  x^2 + y^2 - 12*x - (y1 + y2)*y - 64 = 0

theorem circle_C_properties :
  (circle_C 0 0) ∧ 
  (circle_C 6 0) ∧ 
  (∃ x : ℝ, circle_C x 1) :=
sorry

theorem line_l_property (a b : ℝ) :
  line_l a b ↔ 
  ∃ t : ℝ, 
    (t - 3)^2 + (b + 4)^2 = 25 ∧
    ((a - t)^2 + (b - 1)^2) = ((a - 2)^2 + (b + 2)^2) :=
sorry

theorem circle_E_fixed_points (y1 y2 : ℝ) :
  (y1 * y2 = -100) →
  (circle_E 16 0 y1 y2) ∧
  (circle_E (-4) 0 y1 y2) :=
sorry

end circle_C_properties_line_l_property_circle_E_fixed_points_l483_48371


namespace cricket_team_average_age_l483_48308

/-- The average age of a cricket team given specific conditions -/
theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (average_age : ℝ),
  team_size = 11 →
  captain_age = 24 →
  wicket_keeper_age_diff = 7 →
  (team_size : ℝ) * average_age = 
    (captain_age : ℝ) + (captain_age + wicket_keeper_age_diff : ℝ) + 
    ((team_size - 2 : ℝ) * (average_age - 1)) →
  average_age = 23 := by
sorry

end cricket_team_average_age_l483_48308


namespace coloring_book_shelves_l483_48365

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 87 → books_sold = 33 → books_per_shelf = 6 → 
  (initial_stock - books_sold) / books_per_shelf = 9 := by
sorry

end coloring_book_shelves_l483_48365


namespace middle_seat_is_A_l483_48331

/-- Represents the position of a person in the train -/
inductive Position
| first
| second
| third
| fourth
| fifth

/-- Represents a person -/
inductive Person
| A
| B
| C
| D
| E

/-- The seating arrangement in the train -/
def SeatingArrangement := Person → Position

theorem middle_seat_is_A (arrangement : SeatingArrangement) : 
  (arrangement Person.D = Position.fifth) →
  (arrangement Person.A = Position.fourth ∧ arrangement Person.E = Position.second) ∨
  (arrangement Person.A = Position.third ∧ arrangement Person.E = Position.second) →
  (arrangement Person.B = Position.first ∨ arrangement Person.B = Position.second) →
  (arrangement Person.B ≠ arrangement Person.C ∧ 
   arrangement Person.A ≠ arrangement Person.C ∧
   arrangement Person.E ≠ arrangement Person.C) →
  arrangement Person.A = Position.third :=
by sorry

end middle_seat_is_A_l483_48331


namespace specific_quilt_shaded_fraction_l483_48379

/-- Represents a square quilt composed of unit squares -/
structure Quilt :=
  (total_squares : ℕ)
  (whole_squares : ℕ)
  (half_shaded_squares : ℕ)
  (fully_half_shaded_squares : ℕ)

/-- Calculates the shaded fraction of a quilt -/
def shaded_fraction (q : Quilt) : ℚ :=
  let shaded_area := (q.half_shaded_squares : ℚ) / 2 + (q.fully_half_shaded_squares : ℚ) / 2
  shaded_area / q.total_squares

/-- Theorem stating that for a specific quilt configuration, the shaded fraction is 2.5/9 -/
theorem specific_quilt_shaded_fraction :
  let q := Quilt.mk 9 4 1 4
  shaded_fraction q = 5/18 := by sorry

end specific_quilt_shaded_fraction_l483_48379


namespace arithmetic_sequence_12th_term_l483_48309

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions, its 12th term equals 14. -/
theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 2) :
  a 12 = 14 := by
sorry

end arithmetic_sequence_12th_term_l483_48309


namespace triangle_existence_condition_l483_48384

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the conditions
def angle_ABC (t : Triangle) : ℝ := sorry
def side_AC (t : Triangle) : ℝ := sorry
def side_BC (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_existence_condition (t : Triangle) (k : ℝ) :
  (∃! t, angle_ABC t = π/3 ∧ side_AC t = 12 ∧ side_BC t = k) ↔
  (0 < k ∧ k ≤ 12) ∨ k = 8 * Real.sqrt 3 := by
  sorry

end triangle_existence_condition_l483_48384


namespace x_intercept_implies_m_slope_implies_m_l483_48358

/-- The equation of line l -/
def line_equation (m x y : ℝ) : Prop :=
  (m^2 - 2*m - 3)*x + (2*m^2 + m - 1)*y = 2*m - 6

/-- The x-intercept of line l is -3 -/
def x_intercept (m : ℝ) : Prop :=
  line_equation m (-3) 0

/-- The slope of line l is -1 -/
def slope_negative_one (m : ℝ) : Prop :=
  m^2 - 2*m - 3 = -(2*m^2 + m - 1) ∧ m^2 - 2*m - 3 ≠ 0

theorem x_intercept_implies_m (m : ℝ) :
  x_intercept m → m = -5/3 :=
sorry

theorem slope_implies_m (m : ℝ) :
  slope_negative_one m → m = 4/3 :=
sorry

end x_intercept_implies_m_slope_implies_m_l483_48358


namespace car_city_efficiency_l483_48361

/-- Represents the fuel efficiency of a car -/
structure CarEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city
  tank : ℝ     -- Tank size in gallons

/-- Theorem stating the car's fuel efficiency in the city given the conditions -/
theorem car_city_efficiency (car : CarEfficiency) 
  (highway_distance : car.highway * car.tank = 900)
  (city_distance : car.city * car.tank = 600)
  (efficiency_difference : car.city = car.highway - 5) :
  car.city = 10 := by sorry

end car_city_efficiency_l483_48361


namespace hotdogs_sold_l483_48341

theorem hotdogs_sold (initial : ℕ) (final : ℕ) (h1 : initial = 99) (h2 : final = 97) :
  initial - final = 2 := by
  sorry

end hotdogs_sold_l483_48341


namespace harry_initial_bid_was_500_l483_48375

/-- Represents the auction scenario with given conditions --/
structure Auction where
  startingBid : ℕ
  harryFirstBid : ℕ
  harryFinalBid : ℕ
  finalBidDifference : ℕ

/-- Calculates the second bidder's bid --/
def secondBid (a : Auction) : ℕ := a.startingBid + 2 * a.harryFirstBid

/-- Calculates the third bidder's bid --/
def thirdBid (a : Auction) : ℕ := a.startingBid + 5 * a.harryFirstBid

/-- Theorem stating that Harry's initial bid increment was $500 --/
theorem harry_initial_bid_was_500 (a : Auction) 
  (h1 : a.startingBid = 300)
  (h2 : a.harryFinalBid = 4000)
  (h3 : a.finalBidDifference = 1500)
  (h4 : a.harryFinalBid = thirdBid a + a.finalBidDifference) :
  a.harryFirstBid = 500 := by
  sorry


end harry_initial_bid_was_500_l483_48375


namespace square_sectors_semicircle_differences_l483_48360

/-- Given a square with side length 300 cm, containing two right-angle sectors and a semicircle,
    prove the difference in area between the two shaded regions and the difference in their perimeters. -/
theorem square_sectors_semicircle_differences (π : ℝ) (h_π : π = 3.14) :
  let square_side : ℝ := 300
  let quarter_circle_area : ℝ := 1/4 * π * square_side^2
  let semicircle_area : ℝ := 1/2 * π * (square_side/2)^2
  let quarter_circle_perimeter : ℝ := 1/2 * π * square_side
  let semicircle_perimeter : ℝ := π * square_side/2 + square_side
  let area_difference : ℝ := 2 * quarter_circle_area - square_side^2 - semicircle_area
  let perimeter_difference : ℝ := 2 * quarter_circle_perimeter - semicircle_perimeter
  area_difference = 15975 ∧ perimeter_difference = 485 :=
by sorry

end square_sectors_semicircle_differences_l483_48360


namespace negation_equivalence_l483_48395

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  sorry

end negation_equivalence_l483_48395


namespace horse_rider_ratio_l483_48303

theorem horse_rider_ratio (total_horses : ℕ) (total_legs_walking : ℕ) 
  (h1 : total_horses = 10)
  (h2 : total_legs_walking = 50) :
  (total_horses - (total_legs_walking / 6)) / total_horses = 1 / 5 := by
  sorry

end horse_rider_ratio_l483_48303


namespace curve_touches_x_axis_and_area_l483_48342

noncomputable def curve (a : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t + Real.exp (a * t), -t + Real.exp (a * t))

theorem curve_touches_x_axis_and_area (a : ℝ) (h : a > 0) :
  (∃ t : ℝ, (curve a t).2 = 0 ∧ 
    (∀ s : ℝ, s ≠ t → (curve a s).2 ≠ 0 ∨ (curve a s).2 < 0)) →
  a = 1 / Real.exp 1 ∧
  (∫ t in (0)..(Real.exp 1), (curve a t).2 - min (curve a t).1 (curve a t).2) = Real.exp 2 / 2 - Real.exp 1 := by
  sorry

end curve_touches_x_axis_and_area_l483_48342


namespace nested_diamond_result_l483_48355

/-- Diamond operation for real numbers -/
def diamond (a b : ℝ) : ℝ := a^2 + b^2 - a * b

/-- Theorem stating the result of the nested diamond operations -/
theorem nested_diamond_result :
  diamond (diamond 3 8) (diamond 8 (-3)) = 7057 := by
  sorry

end nested_diamond_result_l483_48355


namespace not_prime_special_expression_l483_48344

theorem not_prime_special_expression (n : ℕ) (h : n > 2) :
  ¬ Nat.Prime (n^(n^n) - 4*n^n + 3) := by
  sorry

end not_prime_special_expression_l483_48344


namespace max_ab_value_l483_48329

def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : ∃ (ε : ℝ), ε ≠ 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → 
    f a b x ≤ f a b 1 ∨ f a b x ≥ f a b 1) :
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 
    (∃ (ε : ℝ), ε ≠ 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → 
      f a' b' x ≤ f a' b' 1 ∨ f a' b' x ≥ f a' b' 1) → 
    a' * b' ≤ a * b) →
  a * b = 9 := by sorry

end max_ab_value_l483_48329


namespace inequality_proof_l483_48315

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1/a) + (1/b) + (1/c) ≤ (a^8 + b^8 + c^8) / (a^3 * b^3 * c^3) := by
sorry

end inequality_proof_l483_48315


namespace first_company_manager_percentage_l483_48319

/-- Represents a company's workforce composition -/
structure Company where
  total : ℝ
  managers : ℝ

/-- The merged company resulting from two companies -/
def MergedCompany (c1 c2 : Company) : Company where
  total := c1.total + c2.total
  managers := c1.managers + c2.managers

theorem first_company_manager_percentage (c1 c2 : Company) :
  let merged := MergedCompany c1 c2
  (c1.total = 0.25 * merged.total) →
  (merged.managers = 0.25 * merged.total) →
  (c1.managers / c1.total = 0.25) := by
  sorry

end first_company_manager_percentage_l483_48319


namespace at_least_two_equations_have_solutions_l483_48373

-- Define the functions
def f₁ (a b c x : ℝ) : ℝ := (x - b) * (x - c) - (x - a)
def f₂ (a b c x : ℝ) : ℝ := (x - c) * (x - a) - (x - b)
def f₃ (a b c x : ℝ) : ℝ := (x - a) * (x - b) - (x - c)

-- Define the theorem
theorem at_least_two_equations_have_solutions (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x : ℝ, f₁ a b c x = 0) ∧ (∃ x : ℝ, f₂ a b c x = 0) ∨
  (∃ x : ℝ, f₁ a b c x = 0) ∧ (∃ x : ℝ, f₃ a b c x = 0) ∨
  (∃ x : ℝ, f₂ a b c x = 0) ∧ (∃ x : ℝ, f₃ a b c x = 0) :=
sorry

end at_least_two_equations_have_solutions_l483_48373


namespace chord_arrangement_count_l483_48328

/-- The number of ways to choose k items from n items without replacement and without regard to order. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to connect 4 points with 3 chords such that each chord intersects the other two. -/
def fourPointConnections : ℕ := 8

/-- The number of ways to connect 5 points with 3 chords such that exactly two chords share a common endpoint and the remaining chord intersects these two. -/
def fivePointConnections : ℕ := 5

/-- The total number of ways to arrange three intersecting chords among 20 points on a circle. -/
def totalChordArrangements : ℕ := 
  choose 20 3 + choose 20 4 * fourPointConnections + 
  choose 20 5 * fivePointConnections + choose 20 6

theorem chord_arrangement_count : totalChordArrangements = 156180 := by
  sorry

end chord_arrangement_count_l483_48328


namespace class_composition_l483_48359

theorem class_composition (girls boys : ℕ) : 
  girls * 6 = boys * 5 →  -- Initial ratio of girls to boys is 5:6
  (girls - 20) * 3 = boys * 2 →  -- New ratio after 20 girls leave is 2:3
  boys = 120 := by  -- The number of boys in the class is 120
sorry

end class_composition_l483_48359


namespace strawberry_plants_l483_48306

theorem strawberry_plants (initial : ℕ) : 
  (((initial * 2) * 2) * 2) - 4 = 20 → initial = 3 := by
  sorry

end strawberry_plants_l483_48306


namespace movie_expense_ratio_l483_48322

/-- Proves the ratio of movie expenses to weekly allowance -/
theorem movie_expense_ratio (weekly_allowance : ℚ) (car_wash_earning : ℚ) (final_amount : ℚ) :
  weekly_allowance = 10 →
  car_wash_earning = 6 →
  final_amount = 11 →
  (weekly_allowance - (final_amount - car_wash_earning)) / weekly_allowance = 1 / 2 := by
sorry

end movie_expense_ratio_l483_48322
