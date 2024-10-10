import Mathlib

namespace min_value_expression_l820_82054

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x - 1)^2 / (y - 2) + (y - 1)^2 / (x - 2) ≥ 8 ∧
  (∃ x y : ℝ, x > 2 ∧ y > 2 ∧ (x - 1)^2 / (y - 2) + (y - 1)^2 / (x - 2) = 8) :=
by sorry

end min_value_expression_l820_82054


namespace marble_ratio_l820_82069

theorem marble_ratio (total : ℕ) (blue green yellow : ℕ) 
  (h_total : total = 164)
  (h_blue : blue = total / 2)
  (h_green : green = 27)
  (h_yellow : yellow = 14) :
  (total - (blue + green + yellow)) * 4 = total := by
  sorry

end marble_ratio_l820_82069


namespace portfolio_growth_portfolio_growth_example_l820_82037

theorem portfolio_growth (initial_investment : ℝ) (first_year_rate : ℝ) 
  (additional_investment : ℝ) (second_year_rate : ℝ) : ℝ :=
  let first_year_value := initial_investment * (1 + first_year_rate)
  let second_year_initial := first_year_value + additional_investment
  let final_value := second_year_initial * (1 + second_year_rate)
  final_value

theorem portfolio_growth_example : 
  portfolio_growth 80 0.15 28 0.10 = 132 := by
  sorry

end portfolio_growth_portfolio_growth_example_l820_82037


namespace equation_solution_l820_82003

theorem equation_solution :
  ∃ x : ℝ, -2 * (x - 1) = 4 ∧ x = -1 :=
by sorry

end equation_solution_l820_82003


namespace freshman_class_size_l820_82048

theorem freshman_class_size :
  ∃! n : ℕ, n < 600 ∧ n % 19 = 15 ∧ n % 17 = 11 ∧ n = 53 := by
  sorry

end freshman_class_size_l820_82048


namespace number_of_winning_scores_l820_82082

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  /-- The number of runners in each team -/
  runnersPerTeam : Nat
  /-- The total number of runners -/
  totalRunners : Nat
  /-- Assertion that there are two teams -/
  twoTeams : totalRunners = 2 * runnersPerTeam

/-- Calculates the total score of all runners -/
def totalScore (meet : CrossCountryMeet) : Nat :=
  (meet.totalRunners * (meet.totalRunners + 1)) / 2

/-- Calculates the minimum possible team score -/
def minTeamScore (meet : CrossCountryMeet) : Nat :=
  (meet.runnersPerTeam * (meet.runnersPerTeam + 1)) / 2

/-- Calculates the maximum possible winning score -/
def maxWinningScore (meet : CrossCountryMeet) : Nat :=
  (totalScore meet) / 2 - 1

/-- The main theorem stating the number of possible winning scores -/
theorem number_of_winning_scores (meet : CrossCountryMeet) 
  (h : meet.runnersPerTeam = 6) : 
  (maxWinningScore meet) - (minTeamScore meet) + 1 = 18 := by
  sorry


end number_of_winning_scores_l820_82082


namespace jamie_tax_payment_l820_82008

/-- Calculates the tax amount based on a progressive tax system --/
def calculate_tax (gross_income : ℕ) (deduction : ℕ) : ℕ :=
  let taxable_income := gross_income - deduction
  let first_bracket := min taxable_income 150
  let second_bracket := min (taxable_income - 150) 150
  let third_bracket := max (taxable_income - 300) 0
  0 * first_bracket + 
  (10 * second_bracket) / 100 + 
  (15 * third_bracket) / 100

/-- Theorem stating that Jamie's tax payment is $30 --/
theorem jamie_tax_payment : 
  calculate_tax 450 50 = 30 := by
  sorry

#eval calculate_tax 450 50  -- This should output 30

end jamie_tax_payment_l820_82008


namespace divisibility_equivalence_l820_82030

theorem divisibility_equivalence (m n : ℕ+) :
  83 ∣ (25 * m + 3 * n) ↔ 83 ∣ (3 * m + 7 * n) := by
  sorry

end divisibility_equivalence_l820_82030


namespace orange_stack_count_l820_82084

/-- Calculates the number of oranges in a triangular layer -/
def orangesInLayer (a b : ℕ) : ℕ := (a * b) / 2

/-- Calculates the total number of oranges in the stack -/
def totalOranges (baseWidth baseLength : ℕ) : ℕ :=
  let rec sumLayers (width length : ℕ) : ℕ :=
    if width = 0 ∨ length = 0 then 0
    else orangesInLayer width length + sumLayers (width - 1) (length - 1)
  sumLayers baseWidth baseLength

theorem orange_stack_count :
  totalOranges 6 9 = 78 := by
  sorry

#eval totalOranges 6 9  -- Should output 78

end orange_stack_count_l820_82084


namespace sum_of_fractions_equals_one_l820_82029

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  (a^3*b^3 / ((a^3 - b*c)*(b^3 - a*c))) + 
  (a^3*c^3 / ((a^3 - b*c)*(c^3 - a*b))) + 
  (b^3*c^3 / ((b^3 - a*c)*(c^3 - a*b))) = 1 := by
sorry

end sum_of_fractions_equals_one_l820_82029


namespace polynomial_degree_condition_l820_82094

theorem polynomial_degree_condition (k m : ℝ) : 
  (∀ x, ∃ a b, (k - 1) * x^2 + 4 * x - m = a * x + b) → k = 1 := by
  sorry

end polynomial_degree_condition_l820_82094


namespace all_statements_imply_p_and_q_implies_r_l820_82071

theorem all_statements_imply_p_and_q_implies_r (p q r : Prop) :
  ((p ∧ q ∧ r) → ((p ∧ q) → r)) ∧
  ((¬p ∧ q ∧ ¬r) → ((p ∧ q) → r)) ∧
  ((p ∧ ¬q ∧ ¬r) → ((p ∧ q) → r)) ∧
  ((¬p ∧ ¬q ∧ r) → ((p ∧ q) → r)) :=
by sorry

end all_statements_imply_p_and_q_implies_r_l820_82071


namespace odd_function_m_zero_l820_82081

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- The function f(x) = 2x^3 + m -/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ 2 * x^3 + m

theorem odd_function_m_zero :
  ∀ m : ℝ, IsOdd (f m) → m = 0 := by
  sorry

end odd_function_m_zero_l820_82081


namespace emilys_skirt_cost_l820_82050

theorem emilys_skirt_cost (art_supplies_cost total_cost : ℕ) (num_skirts : ℕ) 
  (h1 : art_supplies_cost = 20)
  (h2 : num_skirts = 2)
  (h3 : total_cost = 50) :
  ∃ (skirt_cost : ℕ), skirt_cost * num_skirts + art_supplies_cost = total_cost ∧ skirt_cost = 15 :=
by
  sorry

#check emilys_skirt_cost

end emilys_skirt_cost_l820_82050


namespace find_k_value_l820_82014

theorem find_k_value (k : ℝ) (h : 64 / k = 4) : k = 16 := by
  sorry

end find_k_value_l820_82014


namespace complex_expression_simplification_l820_82080

theorem complex_expression_simplification :
  Real.rpow 0.027 (1/3) * Real.rpow (225/64) (-1/2) / Real.sqrt (Real.rpow (-8/125) (2/3)) = 1/5 := by
  sorry

end complex_expression_simplification_l820_82080


namespace final_crayon_count_l820_82017

def crayon_count (initial : ℕ) (added1 : ℕ) (removed : ℕ) (added2 : ℕ) : ℕ :=
  initial + added1 - removed + added2

theorem final_crayon_count :
  crayon_count 25 15 8 12 = 44 := by
  sorry

end final_crayon_count_l820_82017


namespace prime_sequence_extension_l820_82015

theorem prime_sequence_extension (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) := by
sorry

end prime_sequence_extension_l820_82015


namespace nina_taller_than_lena_probability_zero_l820_82051

-- Define the set of friends
inductive Friend : Type
| Masha : Friend
| Nina : Friend
| Lena : Friend
| Olya : Friend

-- Define a height ordering relation
def TallerThan : Friend → Friend → Prop :=
  sorry

-- Define the conditions
axiom all_different_heights :
  ∀ (a b : Friend), a ≠ b → (TallerThan a b ∨ TallerThan b a)

axiom nina_shorter_than_masha :
  TallerThan Friend.Masha Friend.Nina

axiom lena_taller_than_olya :
  TallerThan Friend.Lena Friend.Olya

-- Define the probability function
def Probability (event : Prop) : ℚ :=
  sorry

-- The theorem to prove
theorem nina_taller_than_lena_probability_zero :
  Probability (TallerThan Friend.Nina Friend.Lena) = 0 :=
sorry

end nina_taller_than_lena_probability_zero_l820_82051


namespace factorization_equality_l820_82010

theorem factorization_equality (x y : ℝ) : x + x^2 - y - y^2 = (x + y + 1) * (x - y) := by
  sorry

end factorization_equality_l820_82010


namespace comparison_and_inequality_l820_82097

theorem comparison_and_inequality (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  (2 * x^2 + y^2 > x^2 + x * y) ∧ (Real.sqrt 6 - Real.sqrt 5 < 2 - Real.sqrt 3) := by
  sorry

end comparison_and_inequality_l820_82097


namespace sqrt_equality_implies_one_and_five_l820_82019

theorem sqrt_equality_implies_one_and_five (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) :
  Real.sqrt (1 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 5 := by
sorry

end sqrt_equality_implies_one_and_five_l820_82019


namespace simplify_trig_expression_l820_82079

theorem simplify_trig_expression : 
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) / 
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) = 
  Real.tan (45 * π / 180) := by
  sorry

end simplify_trig_expression_l820_82079


namespace angle_bisector_length_l820_82022

/-- Given a triangle ABC with sides b and c, and angle A between them,
    prove that the length of the angle bisector of A is (2bc cos(A/2)) / (b + c) -/
theorem angle_bisector_length (b c A : ℝ) (hb : b > 0) (hc : c > 0) (hA : 0 < A ∧ A < π) :
  let S := (1/2) * b * c * Real.sin A
  let l_a := (2 * b * c * Real.cos (A/2)) / (b + c)
  ∀ S', S' = S → l_a = (2 * b * c * Real.cos (A/2)) / (b + c) :=
by sorry

end angle_bisector_length_l820_82022


namespace total_tables_proof_l820_82012

/-- Represents the number of table styles -/
def num_styles : ℕ := 10

/-- Represents the sum of x values for all styles -/
def sum_x : ℕ := 100

/-- Calculates the total number of tables made in both months -/
def total_tables (num_styles : ℕ) (sum_x : ℕ) : ℕ :=
  num_styles * (2 * (sum_x / num_styles) - 3)

theorem total_tables_proof :
  total_tables num_styles sum_x = 170 :=
by sorry

end total_tables_proof_l820_82012


namespace point_not_in_region_l820_82021

def region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

theorem point_not_in_region :
  ¬(region 2 0) ∧ 
  (region 0 0) ∧ 
  (region 1 1) ∧ 
  (region 0 2) := by
  sorry

end point_not_in_region_l820_82021


namespace equation_solution_l820_82062

theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt (x + 16) - (8 * Real.cos (π / 6)) / Real.sqrt (x + 16) = 4) ∧
  (x = (2 + 2 * Real.sqrt (1 + Real.sqrt 3))^2 - 16) := by
  sorry

end equation_solution_l820_82062


namespace smoking_and_sickness_are_distinct_categorical_variables_l820_82039

-- Define a structure for categorical variables
structure CategoricalVariable where
  name : String
  values : List String

-- Define the "Whether smoking" categorical variable
def whetherSmoking : CategoricalVariable := {
  name := "Whether smoking"
  values := ["smoking", "not smoking"]
}

-- Define the "Whether sick" categorical variable
def whetherSick : CategoricalVariable := {
  name := "Whether sick"
  values := ["sick", "not sick"]
}

-- Theorem to prove that "Whether smoking" and "Whether sick" are two distinct categorical variables
theorem smoking_and_sickness_are_distinct_categorical_variables :
  whetherSmoking ≠ whetherSick :=
sorry

end smoking_and_sickness_are_distinct_categorical_variables_l820_82039


namespace fraction_sum_simplification_l820_82085

theorem fraction_sum_simplification :
  150 / 225 + 90 / 135 = 4 / 3 := by sorry

end fraction_sum_simplification_l820_82085


namespace original_class_size_l820_82057

theorem original_class_size (initial_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  initial_avg = 40 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ (original_size : ℕ),
    (original_size * initial_avg + new_students * new_avg) / (original_size + new_students) = initial_avg - avg_decrease ∧
    original_size = 12 := by
  sorry

end original_class_size_l820_82057


namespace problem_solution_l820_82095

theorem problem_solution : 
  (Real.sqrt 4 + abs (-3) + (2 - Real.pi) ^ 0 = 6) ∧ 
  (Real.sqrt 18 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt ((-5)^2) = 5) := by
  sorry

end problem_solution_l820_82095


namespace sequence_sum_theorem_l820_82058

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, b (n + 1) = b n * r

def increasing_sequence (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, s (n + 1) > s n

theorem sequence_sum_theorem (a b : ℕ → ℕ) (k : ℕ) :
  a 1 = 1 →
  b 1 = 1 →
  arithmetic_sequence a →
  geometric_sequence b →
  increasing_sequence a →
  increasing_sequence b →
  (∃ k : ℕ, a (k - 1) + b (k - 1) = 250 ∧ a (k + 1) + b (k + 1) = 1250) →
  a k + b k = 502 :=
sorry

end sequence_sum_theorem_l820_82058


namespace smaller_root_of_equation_l820_82067

theorem smaller_root_of_equation : 
  let f (x : ℚ) := (x - 3/5) * (x - 3/5) + 2 * (x - 3/5) * (x - 1/3)
  ∃ r : ℚ, f r = 0 ∧ r = 19/45 ∧ ∀ s : ℚ, f s = 0 → s ≠ r → r < s :=
by sorry

end smaller_root_of_equation_l820_82067


namespace helium_cost_per_ounce_l820_82047

-- Define the constants
def total_money : ℚ := 200
def sheet_cost : ℚ := 42
def rope_cost : ℚ := 18
def propane_cost : ℚ := 14
def height_per_ounce : ℚ := 113
def max_height : ℚ := 9492

-- Define the theorem
theorem helium_cost_per_ounce :
  let money_left := total_money - (sheet_cost + rope_cost + propane_cost)
  let ounces_needed := max_height / height_per_ounce
  let cost_per_ounce := money_left / ounces_needed
  cost_per_ounce = 3/2 := by sorry

end helium_cost_per_ounce_l820_82047


namespace ellipse_eccentricity_l820_82098

/-- Defines an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Defines a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a line passing through two points -/
def Line (p1 p2 : Point) :=
  {p : Point | (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)}

theorem ellipse_eccentricity (Γ : Ellipse) 
  (F : Point) 
  (A : Point) 
  (B : Point) 
  (N : Point) :
  F.x = 3 ∧ F.y = 0 →
  A.x = 0 ∧ A.y = Γ.b →
  B.x = 0 ∧ B.y = -Γ.b →
  N.x = 12 ∧ N.y = 0 →
  ∃ (M : Point), M ∈ Line A F ∧ M ∈ Line B N ∧ 
    (M.x^2 / Γ.a^2 + M.y^2 / Γ.b^2 = 1) →
  (Γ.a^2 - Γ.b^2) / Γ.a^2 = 1/4 := by
  sorry

end ellipse_eccentricity_l820_82098


namespace complement_intersection_problem_l820_82088

theorem complement_intersection_problem (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5} →
  A = {2, 3, 4} →
  B = {3, 4, 5} →
  (Aᶜ ∩ Bᶜ : Set ℕ) = {1, 2, 5} := by
  sorry

end complement_intersection_problem_l820_82088


namespace parallel_lines_corresponding_angles_l820_82066

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of corresponding angles
def corresponding_angles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the concept of equal angles
def angle_equal (a1 a2 : Angle) : Prop := sorry

-- The theorem to be proved
theorem parallel_lines_corresponding_angles 
  (l1 l2 : Line) (a1 a2 : Angle) : 
  parallel l1 l2 → corresponding_angles a1 a2 l1 l2 → angle_equal a1 a2 := by
  sorry


end parallel_lines_corresponding_angles_l820_82066


namespace incorrect_subset_l820_82036

-- Define the sets
def set1 : Set ℕ := {1, 2, 3}
def set2 : Set ℕ := {1, 2}

-- Theorem statement
theorem incorrect_subset : ¬(set1 ⊆ set2) := by
  sorry

end incorrect_subset_l820_82036


namespace tianjin_population_scientific_notation_l820_82024

/-- The population of Tianjin -/
def tianjin_population : ℕ := 13860000

/-- Scientific notation representation of Tianjin's population -/
def tianjin_scientific : ℝ := 1.386 * (10 ^ 7)

/-- Theorem stating that the population of Tianjin in scientific notation is correct -/
theorem tianjin_population_scientific_notation :
  (tianjin_population : ℝ) = tianjin_scientific :=
by sorry

end tianjin_population_scientific_notation_l820_82024


namespace quadratic_equation_roots_l820_82020

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k - 1)*x₁ + k^2 - 1 = 0 ∧ 
    x₂^2 + (2*k - 1)*x₂ + k^2 - 1 = 0 ∧ 
    x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 :=
by sorry

end quadratic_equation_roots_l820_82020


namespace function_extrema_product_l820_82068

theorem function_extrema_product (a b : Real) :
  let f := fun x => a - Real.sqrt 3 * Real.tan (2 * x)
  (∀ x ∈ Set.Icc (-Real.pi/6) b, f x ≤ 7) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) b, f x ≥ 3) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) b, f x = 7) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) b, f x = 3) →
  a * b = Real.pi / 3 :=
by sorry

end function_extrema_product_l820_82068


namespace quadratic_root_value_l820_82099

theorem quadratic_root_value (a : ℝ) : 
  a^2 - 2*a - 3 = 0 → 2*a^2 - 4*a + 1 = 7 := by
  sorry

end quadratic_root_value_l820_82099


namespace sector_properties_l820_82032

/-- Properties of a circular sector --/
theorem sector_properties :
  -- Given a sector with central angle α and radius R
  ∀ (α R : ℝ),
  -- When α = π/3 and R = 10
  α = π / 3 ∧ R = 10 →
  -- The arc length is 10π/3
  (α * R = 10 * π / 3) ∧
  -- The area is 50π/3
  (1 / 2 * α * R^2 = 50 * π / 3) ∧
  -- For a sector with perimeter 12
  ∀ (r l : ℝ),
  (l + 2 * r = 12) →
  -- The maximum area is 9
  (1 / 2 * l * r ≤ 9) ∧
  -- The maximum area occurs when α = 2
  (1 / 2 * l * r = 9 → l / r = 2) := by
sorry

end sector_properties_l820_82032


namespace east_region_difference_l820_82092

/-- The difference in square miles between the regions east of two plains -/
def region_difference (total_area plain_B_area : ℕ) : ℕ :=
  plain_B_area - (total_area - plain_B_area)

/-- Theorem stating the difference between regions east of plain B and A -/
theorem east_region_difference :
  ∀ (total_area plain_B_area : ℕ),
  total_area = 350 →
  plain_B_area = 200 →
  region_difference total_area plain_B_area = 50 :=
by
  sorry

#eval region_difference 350 200

end east_region_difference_l820_82092


namespace invalid_votes_percentage_l820_82033

theorem invalid_votes_percentage (total_votes : ℕ) (valid_votes_winner_percentage : ℚ)
  (valid_votes_loser : ℕ) (h1 : total_votes = 5500)
  (h2 : valid_votes_winner_percentage = 55/100)
  (h3 : valid_votes_loser = 1980) :
  (total_votes - (valid_votes_loser / (1 - valid_votes_winner_percentage))) / total_votes = 1/5 := by
  sorry

#check invalid_votes_percentage

end invalid_votes_percentage_l820_82033


namespace polygon_with_60_degree_exterior_angles_has_6_sides_l820_82016

-- Define a polygon type
structure Polygon where
  sides : ℕ
  exteriorAngle : ℝ

-- Theorem statement
theorem polygon_with_60_degree_exterior_angles_has_6_sides :
  ∀ p : Polygon, p.exteriorAngle = 60 → p.sides = 6 := by
  sorry

end polygon_with_60_degree_exterior_angles_has_6_sides_l820_82016


namespace arithmetic_sequence_fifth_term_l820_82006

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the sum of the first and ninth terms is 10,
    the fifth term is equal to 5. -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 := by
  sorry


end arithmetic_sequence_fifth_term_l820_82006


namespace monotone_decreasing_implies_a_geq_one_l820_82087

/-- The function f(x) = x³ - ax² - x + 6 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - x + 6

/-- f is monotonically decreasing in the interval (0,1) --/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y ∧ y < 1 → f a x > f a y

theorem monotone_decreasing_implies_a_geq_one (a : ℝ) :
  is_monotone_decreasing a → a ≥ 1 := by sorry

end monotone_decreasing_implies_a_geq_one_l820_82087


namespace range_of_a_l820_82055

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → |x^2 - a| + |x + a| = |x^2 + x|) → 
  a ∈ Set.Icc (-1) 1 := by
sorry

end range_of_a_l820_82055


namespace quadratic_two_distinct_roots_l820_82083

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 3) * x^2 - 4 * x - 1 = 0 ∧ (a - 3) * y^2 - 4 * y - 1 = 0) ↔
  (a > -1 ∧ a ≠ 3) :=
by sorry

end quadratic_two_distinct_roots_l820_82083


namespace travel_time_l820_82025

/-- Given a person's travel rate, calculate the time to travel a certain distance -/
theorem travel_time (distance_to_julia : ℝ) (time_to_julia : ℝ) (distance_to_bernard : ℝ) :
  distance_to_julia = 2 →
  time_to_julia = 8 →
  distance_to_bernard = 5 →
  (distance_to_bernard / distance_to_julia) * time_to_julia = 20 :=
by
  sorry

end travel_time_l820_82025


namespace jackie_has_six_apples_l820_82038

/-- The number of apples Adam has -/
def adam_apples : ℕ := 9

/-- The difference between Adam's and Jackie's apples -/
def difference : ℕ := 3

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := adam_apples - difference

theorem jackie_has_six_apples : jackie_apples = 6 := by sorry

end jackie_has_six_apples_l820_82038


namespace round_trip_average_speed_l820_82065

/-- Calculate the average speed of a round trip flight with wind vectors -/
theorem round_trip_average_speed 
  (speed_to_mother : ℝ) 
  (tailwind_speed : ℝ) 
  (tailwind_angle : ℝ) 
  (speed_to_home : ℝ) 
  (headwind_speed : ℝ) 
  (headwind_angle : ℝ) 
  (h1 : speed_to_mother = 96) 
  (h2 : tailwind_speed = 12) 
  (h3 : tailwind_angle = 30 * π / 180) 
  (h4 : speed_to_home = 88) 
  (h5 : headwind_speed = 15) 
  (h6 : headwind_angle = 60 * π / 180) : 
  ∃ (average_speed : ℝ), 
    abs (average_speed - 93.446) < 0.001 ∧ 
    average_speed = (
      (speed_to_mother + tailwind_speed * Real.cos tailwind_angle) + 
      (speed_to_home - headwind_speed * Real.cos headwind_angle)
    ) / 2 := by
  sorry

end round_trip_average_speed_l820_82065


namespace perimeter_of_specific_triangle_l820_82042

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of DP, where P is the tangency point on DE -/
  dp : ℝ
  /-- The length of PE, where P is the tangency point on DE -/
  pe : ℝ
  /-- The length of the tangent from vertex F to the circle -/
  ft : ℝ

/-- The perimeter of the triangle -/
def perimeter (t : TriangleWithInscribedCircle) : ℝ :=
  2 * (t.dp + t.pe + t.ft)

theorem perimeter_of_specific_triangle :
  let t : TriangleWithInscribedCircle := {
    r := 13,
    dp := 17,
    pe := 31,
    ft := 20
  }
  perimeter t = 136 := by sorry

end perimeter_of_specific_triangle_l820_82042


namespace x_squared_minus_y_squared_l820_82056

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 77) : x^2 - y^2 = 5 / 847 := by
  sorry

end x_squared_minus_y_squared_l820_82056


namespace cone_base_circumference_l820_82052

theorem cone_base_circumference (r : ℝ) (angle : ℝ) :
  r = 6 →
  angle = 300 →
  let full_circumference := 2 * Real.pi * r
  let remaining_fraction := angle / 360
  let cone_base_circumference := remaining_fraction * full_circumference
  cone_base_circumference = 10 * Real.pi :=
by sorry

end cone_base_circumference_l820_82052


namespace nested_fraction_evaluation_l820_82075

theorem nested_fraction_evaluation : 
  let f (x : ℝ) := (x + 2) / (x - 2)
  let g (x : ℝ) := (f x + 2) / (f x - 2)
  g 1 = 1/5 := by sorry

end nested_fraction_evaluation_l820_82075


namespace arithmetic_ellipse_properties_l820_82078

/-- An arithmetic ellipse with semi-major axis a, semi-minor axis b, and focal distance c -/
structure ArithmeticEllipse where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  a_gt_b : b < a
  arithmetic_progression : 2 * b = a + c
  ellipse_equation : a^2 - b^2 = c^2

/-- Main theorem about arithmetic ellipses -/
theorem arithmetic_ellipse_properties (Γ : ArithmeticEllipse) :
  -- 1. Eccentricity is 3/5
  (Γ.c / Γ.a = 3/5) ∧
  -- 2. Slope of tangent line at (0, -a) is ±3/5
  (∃ k : ℝ, k^2 = (3/5)^2 ∧
    ∀ x y : ℝ, (x^2 / Γ.a^2 + y^2 / Γ.b^2 = 1) → (y = k * x - Γ.a) →
      x^2 + (k * x - Γ.a)^2 / Γ.b^2 = 1) ∧
  -- 3. Circle with diameter MN passes through (±b, 0)
  (∀ m n : ℝ, (m^2 / Γ.a^2 + n^2 / Γ.b^2 = 1) →
    ∃ y : ℝ, ((-Γ.b)^2 + y^2 + (Γ.a * n / (m + Γ.a) + Γ.a * n / (m - Γ.a)) * y - Γ.b^2 = 0) ∧
              (Γ.b^2 + y^2 + (Γ.a * n / (m + Γ.a) + Γ.a * n / (m - Γ.a)) * y - Γ.b^2 = 0)) :=
by sorry

end arithmetic_ellipse_properties_l820_82078


namespace simplified_fourth_root_l820_82043

theorem simplified_fourth_root (c d : ℕ+) :
  (3^5 * 5^3 : ℝ)^(1/4) = c * d^(1/4) → c + d = 3378 := by
  sorry

end simplified_fourth_root_l820_82043


namespace initial_bacteria_count_l820_82064

/-- The number of bacteria after a given number of 30-second intervals -/
def bacteria_count (initial : ℕ) (intervals : ℕ) : ℕ :=
  initial * 4^intervals

/-- The theorem stating the initial number of bacteria -/
theorem initial_bacteria_count : 
  ∃ (initial : ℕ), bacteria_count initial 8 = 1048576 ∧ initial = 16 :=
by
  sorry

end initial_bacteria_count_l820_82064


namespace only_integer_solution_is_two_l820_82089

theorem only_integer_solution_is_two :
  ∀ x : ℤ, (0 < (x - 1)^2 / (x + 1) ∧ (x - 1)^2 / (x + 1) < 1) ↔ x = 2 :=
by sorry

end only_integer_solution_is_two_l820_82089


namespace sqrt_12_div_sqrt_3_equals_2_l820_82013

theorem sqrt_12_div_sqrt_3_equals_2 : Real.sqrt 12 / Real.sqrt 3 = 2 := by
  sorry

end sqrt_12_div_sqrt_3_equals_2_l820_82013


namespace power_of_32_equals_power_of_2_l820_82000

theorem power_of_32_equals_power_of_2 : ∀ q : ℕ, 32^5 = 2^q → q = 25 := by
  sorry

end power_of_32_equals_power_of_2_l820_82000


namespace roots_of_polynomial_l820_82044

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) ∧
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 3) = f x) :=
sorry

end roots_of_polynomial_l820_82044


namespace sum_product_inequality_l820_82007

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + a * c + b * c ≤ 0 := by
  sorry

end sum_product_inequality_l820_82007


namespace rectangle_area_l820_82070

/-- 
A rectangle with diagonal length x and length three times its width 
has an area of (3/10)x^2
-/
theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ w : ℝ, w > 0 ∧ 
    w^2 + (3*w)^2 = x^2 ∧ 
    3 * w^2 = (3/10) * x^2 := by
  sorry

end rectangle_area_l820_82070


namespace bookshelf_selection_l820_82004

theorem bookshelf_selection (math_books : ℕ) (chinese_books : ℕ) (english_books : ℕ) 
  (h1 : math_books = 3) (h2 : chinese_books = 5) (h3 : english_books = 8) :
  math_books + chinese_books + english_books = 16 := by
  sorry

end bookshelf_selection_l820_82004


namespace f_composition_equals_8c_implies_c_equals_1_l820_82011

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 199^x + 1 else x^2 + 2*c*x

theorem f_composition_equals_8c_implies_c_equals_1 (c : ℝ) :
  f c (f c 0) = 8*c → c = 1 := by
  sorry

end f_composition_equals_8c_implies_c_equals_1_l820_82011


namespace angle_calculation_l820_82001

-- Define a structure for angles in degrees, minutes, and seconds
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)
  (seconds : ℕ)

-- Define multiplication of an angle by a natural number
def multiply_angle (a : Angle) (n : ℕ) : Angle :=
  sorry

-- Define division of an angle by a natural number
def divide_angle (a : Angle) (n : ℕ) : Angle :=
  sorry

-- Define addition of two angles
def add_angles (a b : Angle) : Angle :=
  sorry

-- Theorem statement
theorem angle_calculation :
  let a1 := Angle.mk 50 24 0
  let a2 := Angle.mk 98 12 25
  add_angles (multiply_angle a1 3) (divide_angle a2 5) = Angle.mk 170 50 29 :=
sorry

end angle_calculation_l820_82001


namespace specific_parallelogram_area_l820_82059

/-- A parallelogram in 2D space -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := sorry

/-- The specific parallelogram in the problem -/
def specificParallelogram : Parallelogram := {
  v1 := (0, 0)
  v2 := (4, 0)
  v3 := (1, 5)
  v4 := (5, 5)
}

/-- Theorem: The area of the specific parallelogram is 20 square units -/
theorem specific_parallelogram_area :
  parallelogramArea specificParallelogram = 20 := by sorry

end specific_parallelogram_area_l820_82059


namespace exactly_two_valid_pairs_l820_82074

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_valid_pair (n m : ℕ+) : Prop := sum_factorials n.val = m.val ^ 2

theorem exactly_two_valid_pairs :
  ∃! (s : Finset (ℕ+ × ℕ+)), s.card = 2 ∧ ∀ (p : ℕ+ × ℕ+), p ∈ s ↔ is_valid_pair p.1 p.2 :=
sorry

end exactly_two_valid_pairs_l820_82074


namespace jade_lego_tower_level_width_l820_82093

/-- Calculates the width of each level in Jade's Lego tower -/
theorem jade_lego_tower_level_width 
  (initial_pieces : ℕ) 
  (levels : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : initial_pieces = 100)
  (h2 : levels = 11)
  (h3 : remaining_pieces = 23) :
  (initial_pieces - remaining_pieces) / levels = 7 := by
  sorry

end jade_lego_tower_level_width_l820_82093


namespace factorization_proof_l820_82005

theorem factorization_proof (m n : ℝ) : 12 * m^2 * n - 12 * m * n + 3 * n = 3 * n * (2 * m - 1)^2 := by
  sorry

end factorization_proof_l820_82005


namespace gcd_of_polynomial_and_multiple_l820_82090

theorem gcd_of_polynomial_and_multiple (x : ℤ) : 
  (∃ k : ℤ, x = 34567 * k) → 
  Nat.gcd ((3*x+4)*(8*x+3)*(15*x+11)*(x+15) : ℤ).natAbs x.natAbs = 1 := by
  sorry

end gcd_of_polynomial_and_multiple_l820_82090


namespace composition_central_symmetries_is_translation_composition_translation_central_symmetry_is_central_symmetry_l820_82076

-- Define the types for our transformations
def CentralSymmetry (center : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry
def Translation (vector : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

-- Define composition of transformations
def Compose (f g : (ℝ × ℝ) → (ℝ × ℝ)) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

-- Theorem 1: Composition of two central symmetries is a translation
theorem composition_central_symmetries_is_translation 
  (c1 c2 : ℝ × ℝ) : 
  ∃ (v : ℝ × ℝ), Compose (CentralSymmetry c2) (CentralSymmetry c1) = Translation v := by sorry

-- Theorem 2: Composition of translation and central symmetry (both orders) is a central symmetry
theorem composition_translation_central_symmetry_is_central_symmetry 
  (v : ℝ × ℝ) (c : ℝ × ℝ) : 
  (∃ (c1 : ℝ × ℝ), Compose (Translation v) (CentralSymmetry c) = CentralSymmetry c1) ∧
  (∃ (c2 : ℝ × ℝ), Compose (CentralSymmetry c) (Translation v) = CentralSymmetry c2) := by sorry

end composition_central_symmetries_is_translation_composition_translation_central_symmetry_is_central_symmetry_l820_82076


namespace square_difference_l820_82086

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 9/20) 
  (h2 : x - y = 1/20) : 
  x^2 - y^2 = 9/400 := by
sorry

end square_difference_l820_82086


namespace ellipse_equation_proof_l820_82027

/-- The equation of the given ellipse -/
def given_ellipse (x y : ℝ) : Prop := 3 * x^2 + 8 * y^2 = 24

/-- The equation of the ellipse we want to prove -/
def target_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

/-- The foci of an ellipse with equation ax^2 + by^2 = c -/
def foci (a b c : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 = (1/a - 1/b) * c ∧ y = 0}

theorem ellipse_equation_proof :
  (∀ x y, given_ellipse x y ↔ 3 * x^2 + 8 * y^2 = 24) →
  (target_ellipse 3 2) →
  (foci 3 8 24 = foci (1/15) (1/10) 1) →
  ∀ x y, target_ellipse x y ↔ x^2 / 15 + y^2 / 10 = 1 :=
by sorry

end ellipse_equation_proof_l820_82027


namespace intersection_of_M_and_P_l820_82053

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = 3^x}
def P : Set ℝ := {y | y ≥ 1}

-- State the theorem
theorem intersection_of_M_and_P : M ∩ P = {y | y ≥ 1} := by sorry

end intersection_of_M_and_P_l820_82053


namespace quadratic_inequality_solution_set_l820_82034

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | a * x^2 - (2 + a) * x + 2 > 0} = {x : ℝ | 2 / a < x ∧ x < 1} := by
  sorry

end quadratic_inequality_solution_set_l820_82034


namespace union_of_sets_l820_82035

theorem union_of_sets : 
  let M : Set Int := {1, 0, -1}
  let N : Set Int := {1, 2}
  M ∪ N = {1, 2, 0, -1} := by sorry

end union_of_sets_l820_82035


namespace range_of_a_l820_82026

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x - 1 ≥ a^2 ∧ x - 4 < 2*a) → 
  -1 < a ∧ a < 3 :=
by sorry

end range_of_a_l820_82026


namespace shortest_paths_count_l820_82073

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the grid and gas station locations -/
structure Grid where
  width : ℕ
  height : ℕ
  gasStations : List Point

/-- Represents the problem setup -/
structure ProblemSetup where
  grid : Grid
  start : Point
  finish : Point
  refuelDistance : ℕ

/-- Calculates the number of shortest paths between two points on a grid -/
def numberOfShortestPaths (start : Point) (finish : Point) : ℕ :=
  sorry

/-- Checks if a path is valid given the refuel constraints -/
def isValidPath (path : List Point) (gasStations : List Point) (refuelDistance : ℕ) : Bool :=
  sorry

/-- Main theorem: The number of shortest paths from A to B with refueling constraints is 24 -/
theorem shortest_paths_count (setup : ProblemSetup) : 
  (numberOfShortestPaths setup.start setup.finish) = 24 :=
sorry

end shortest_paths_count_l820_82073


namespace unique_special_polynomial_l820_82049

/-- A polynomial function that satisfies the given conditions -/
structure SpecialPolynomial where
  f : ℝ → ℝ
  is_polynomial : Polynomial ℝ
  degree_ge_one : (Polynomial.degree is_polynomial) ≥ 1
  cond_square : ∀ x, f (x^2) = (f x)^2
  cond_compose : ∀ x, f (x^2) = f (f x)

/-- Theorem stating that there exists exactly one special polynomial -/
theorem unique_special_polynomial :
  ∃! (p : SpecialPolynomial), True :=
sorry

end unique_special_polynomial_l820_82049


namespace odd_function_power_l820_82072

def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - |x - a|

theorem odd_function_power (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is an odd function
  (∃ x, f a x ≠ 0) →          -- f is not identically zero
  a^2012 = 1 := by
sorry

end odd_function_power_l820_82072


namespace quadratic_solution_l820_82009

theorem quadratic_solution (x : ℝ) : x^2 - 6*x + 8 = 0 → x = 2 ∨ x = 4 := by
  sorry

end quadratic_solution_l820_82009


namespace rectangle_formation_ways_l820_82028

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 4

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 4

/-- The number of lines needed to form a side of the rectangle -/
def lines_per_side : ℕ := 2

/-- Theorem stating that the number of ways to choose lines to form a rectangle is 36 -/
theorem rectangle_formation_ways :
  (choose num_horizontal_lines lines_per_side) * (choose num_vertical_lines lines_per_side) = 36 := by
  sorry


end rectangle_formation_ways_l820_82028


namespace triangle_side_calculation_l820_82096

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  B = π / 4 →
  (1 / 2) * a * b * Real.sin C = 2 →
  b = 4 * Real.sqrt 2 :=
by
  sorry

end triangle_side_calculation_l820_82096


namespace smaller_circle_circumference_l820_82046

theorem smaller_circle_circumference 
  (square_area : ℝ) 
  (larger_radius : ℝ) 
  (smaller_radius : ℝ) 
  (h1 : square_area = 784) 
  (h2 : square_area = (2 * larger_radius)^2) 
  (h3 : larger_radius = (7/3) * smaller_radius) : 
  2 * Real.pi * smaller_radius = 12 * Real.pi := by
sorry

end smaller_circle_circumference_l820_82046


namespace power_function_theorem_l820_82002

theorem power_function_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^a) →  -- f is a power function with exponent a
  f 2 = 1/4 →         -- f passes through the point (2, 1/4)
  f (-2) = 1/4 :=     -- prove that f(-2) = 1/4
by
  sorry

end power_function_theorem_l820_82002


namespace correct_derivatives_l820_82063

open Real

theorem correct_derivatives :
  (∀ x : ℝ, deriv (λ x => (2 * x) / (x^2 + 1)) x = (2 - 2 * x^2) / (x^2 + 1)^2) ∧
  (∀ x : ℝ, deriv (λ x => exp (3 * x + 1)) x = 3 * exp (3 * x + 1)) := by
  sorry

end correct_derivatives_l820_82063


namespace toy_price_problem_l820_82077

theorem toy_price_problem (num_toys : ℕ) (sixth_toy_price : ℝ) (new_average : ℝ) :
  num_toys = 5 →
  sixth_toy_price = 16 →
  new_average = 11 →
  (num_toys : ℝ) * (num_toys + 1 : ℝ)⁻¹ * (num_toys * new_average - sixth_toy_price) = 10 :=
by sorry

end toy_price_problem_l820_82077


namespace compare_sqrt_l820_82060

theorem compare_sqrt : 2 * Real.sqrt 3 < 3 * Real.sqrt 2 := by
  sorry

end compare_sqrt_l820_82060


namespace jerrys_age_l820_82018

/-- Given that Mickey's age is 6 years less than 200% of Jerry's age and Mickey is 20 years old, prove that Jerry is 13 years old. -/
theorem jerrys_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 2 * jerry_age - 6) 
  (h2 : mickey_age = 20) : 
  jerry_age = 13 := by
sorry

end jerrys_age_l820_82018


namespace triangle_side_length_l820_82061

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ

-- State the theorem
theorem triangle_side_length (t : Triangle) :
  (t.b - t.c = 2) →
  (1/2 * t.b * t.c * Real.sqrt (1 - (-1/4)^2) = 3 * Real.sqrt 15) →
  (Real.cos t.A = -1/4) →
  t.a = 8 := by
  sorry

end triangle_side_length_l820_82061


namespace sqrt_five_multiplication_l820_82091

theorem sqrt_five_multiplication : 2 * Real.sqrt 5 * (3 * Real.sqrt 5) = 30 := by
  sorry

end sqrt_five_multiplication_l820_82091


namespace max_correct_answers_l820_82045

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 30 →
  correct_points = 4 →
  incorrect_points = -3 →
  total_score = 54 →
  ∃ (correct incorrect blank : ℕ),
    correct + incorrect + blank = total_questions ∧
    correct * correct_points + incorrect * incorrect_points = total_score ∧
    correct ≤ 20 ∧
    ∀ (c : ℕ), c > 20 →
      ¬∃ (i b : ℕ), c + i + b = total_questions ∧
                    c * correct_points + i * incorrect_points = total_score :=
by sorry

end max_correct_answers_l820_82045


namespace cubic_cm_in_cubic_meter_proof_l820_82023

/-- The number of cubic centimeters in one cubic meter -/
def cubic_cm_in_cubic_meter : ℕ := 1000000

/-- The number of centimeters in one meter -/
def cm_in_meter : ℕ := 100

/-- Theorem stating that the number of cubic centimeters in one cubic meter is 1,000,000,
    given that one meter is equal to one hundred centimeters -/
theorem cubic_cm_in_cubic_meter_proof :
  cubic_cm_in_cubic_meter = cm_in_meter ^ 3 := by
  sorry

end cubic_cm_in_cubic_meter_proof_l820_82023


namespace smaller_tv_diagonal_l820_82031

theorem smaller_tv_diagonal (d : ℝ) : 
  d > 0 → 
  (28 / Real.sqrt 2)^2 = (d / Real.sqrt 2)^2 + 79.5 → 
  d = 25 := by
sorry

end smaller_tv_diagonal_l820_82031


namespace books_read_is_seven_l820_82041

-- Define the number of movies watched
def movies_watched : ℕ := 21

-- Define the relationship between movies watched and books read
def books_read : ℕ := movies_watched - 14

-- Theorem to prove
theorem books_read_is_seven : books_read = 7 := by
  sorry

end books_read_is_seven_l820_82041


namespace A_intersect_C_R_B_eq_interval_l820_82040

-- Define set A
def A : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x ∧ 0 ≤ x ∧ x ≤ 4}

-- Define the complement of B relative to ℝ
def C_R_B : Set ℝ := (Set.univ : Set ℝ) \ B

-- Theorem statement
theorem A_intersect_C_R_B_eq_interval :
  A ∩ C_R_B = Set.Icc (-3 : ℝ) 0 := by sorry

end A_intersect_C_R_B_eq_interval_l820_82040
