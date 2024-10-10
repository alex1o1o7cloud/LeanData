import Mathlib

namespace triangle_angle_calculation_l221_22102

theorem triangle_angle_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = b) →
  (C = π / 5) →
  (B = 4 * π / 15) :=
by sorry

end triangle_angle_calculation_l221_22102


namespace complement_of_union_MN_l221_22188

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_of_union_MN : 
  (U \ (M ∪ N)) = {2, 4, 8} := by sorry

end complement_of_union_MN_l221_22188


namespace cupcake_frosting_l221_22130

def cagney_rate : ℚ := 1 / 15
def lacey_rate : ℚ := 1 / 25
def lacey_delay : ℕ := 30
def total_time : ℕ := 600

def total_cupcakes : ℕ := 62

theorem cupcake_frosting :
  (cagney_rate * total_time).floor +
  (lacey_rate * (total_time - lacey_delay)).floor = total_cupcakes :=
sorry

end cupcake_frosting_l221_22130


namespace geometric_sequence_ratio_l221_22131

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ r < 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geom : is_positive_geometric_sequence a)
  (h_decreasing : ∀ n : ℕ, a (n + 1) < a n)
  (h_product : a 2 * a 8 = 6)
  (h_sum : a 4 + a 6 = 5) :
  a 5 / a 7 = 3/2 :=
sorry

end geometric_sequence_ratio_l221_22131


namespace problem_statement_l221_22119

theorem problem_statement (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 2))
  (h2 : (27 : ℝ) ^ y = 9 ^ (x - 8)) :
  x + y = 40 := by
  sorry

end problem_statement_l221_22119


namespace correct_average_calculation_l221_22162

theorem correct_average_calculation (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 17 ∧ incorrect_num = 26 ∧ correct_num = 56 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 20 := by
  sorry

end correct_average_calculation_l221_22162


namespace boat_speed_in_still_water_l221_22175

theorem boat_speed_in_still_water : 
  ∀ (v_b v_c v_w : ℝ),
    v_b - v_c - v_w = 4 →
    v_c ≤ 4 →
    v_w ≥ -1 →
    v_b = 7 := by
  sorry

end boat_speed_in_still_water_l221_22175


namespace triangle_operation_result_l221_22195

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := a * b / (-6)

-- State the theorem
theorem triangle_operation_result :
  triangle 4 (triangle 3 2) = 2/3 := by
  sorry

end triangle_operation_result_l221_22195


namespace negation_of_p_l221_22173

-- Define the proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

-- State the theorem
theorem negation_of_p (f : ℝ → ℝ) :
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
by sorry

end negation_of_p_l221_22173


namespace biased_die_probability_l221_22107

theorem biased_die_probability (p : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) :
  (Nat.choose 8 6 : ℝ) * p^6 * (1-p)^2 = 125/256 → p^6 * (1-p)^2 = 125/7168 := by
  sorry

end biased_die_probability_l221_22107


namespace arithmetic_mean_of_fractions_l221_22112

theorem arithmetic_mean_of_fractions : 
  let a := 3 / 8
  let b := 5 / 9
  (a + b) / 2 = 67 / 144 := by sorry

end arithmetic_mean_of_fractions_l221_22112


namespace inequality_solution_l221_22180

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) - 4 / (x + 8) > 1 / 2) ↔ (x > -4 ∧ x ≠ -2) :=
by sorry

end inequality_solution_l221_22180


namespace zero_in_interval_l221_22152

def f (x : ℝ) : ℝ := x^5 + 8*x^3 - 1

theorem zero_in_interval :
  (f 0 < 0) → (f 0.5 > 0) → ∃ x, x ∈ Set.Ioo 0 0.5 ∧ f x = 0 :=
by
  sorry

end zero_in_interval_l221_22152


namespace modified_lucas_units_digit_l221_22124

/-- Modified Lucas sequence -/
def L' : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | n + 2 => 2 * L' (n + 1) + L' n

/-- Function to get the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the units digit of L'_{L'_{20}} is d -/
theorem modified_lucas_units_digit :
  ∃ d : ℕ, d < 10 ∧ unitsDigit (L' (L' 20)) = d :=
sorry

end modified_lucas_units_digit_l221_22124


namespace total_dresses_l221_22134

theorem total_dresses (emily_dresses : ℕ) (melissa_dresses : ℕ) (debora_dresses : ℕ) : 
  emily_dresses = 16 →
  melissa_dresses = emily_dresses / 2 →
  debora_dresses = melissa_dresses + 12 →
  emily_dresses + melissa_dresses + debora_dresses = 44 := by
sorry

end total_dresses_l221_22134


namespace fourth_root_is_negative_seven_l221_22108

/-- Represents a polynomial of degree 4 with rational coefficients -/
structure QuarticPolynomial where
  d : ℚ
  e : ℚ
  f : ℚ

/-- Checks if a given number is a root of the polynomial -/
def isRoot (p : QuarticPolynomial) (x : ℝ) : Prop :=
  x^4 + p.d * x^2 + p.e * x + p.f = 0

theorem fourth_root_is_negative_seven
  (p : QuarticPolynomial)
  (h1 : isRoot p (3 - Real.sqrt 5))
  (h2 : ∃ (a b : ℤ), isRoot p a ∧ isRoot p b) :
  isRoot p (-7) :=
sorry

end fourth_root_is_negative_seven_l221_22108


namespace martin_fruit_ratio_l221_22118

/-- Given that Martin has twice as many oranges as limes now, 50 oranges, and initially had 150 fruits,
    prove that the ratio of fruits eaten to initial fruits is 1/2 -/
theorem martin_fruit_ratio :
  ∀ (oranges_now limes_now fruits_initial : ℕ),
    oranges_now = 50 →
    fruits_initial = 150 →
    oranges_now = 2 * limes_now →
    (fruits_initial - (oranges_now + limes_now)) / fruits_initial = 1 / 2 := by
  sorry

end martin_fruit_ratio_l221_22118


namespace monthly_cost_correct_l221_22196

/-- Represents the monthly cost for online access -/
def monthly_cost : ℝ := 8

/-- Represents the initial app cost -/
def app_cost : ℝ := 5

/-- Represents the number of months of online access -/
def months : ℝ := 2

/-- Represents the total cost for the app and online access -/
def total_cost : ℝ := 21

/-- Proves that the monthly cost for online access is correct -/
theorem monthly_cost_correct : app_cost + months * monthly_cost = total_cost := by
  sorry

end monthly_cost_correct_l221_22196


namespace root_product_equality_l221_22105

theorem root_product_equality (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 1 = 0) → 
  (β^2 + p*β + 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
  sorry

end root_product_equality_l221_22105


namespace robert_basic_salary_l221_22148

/-- Represents Robert's financial situation --/
structure RobertFinances where
  basic_salary : ℝ
  total_sales : ℝ
  monthly_expenses : ℝ

/-- Calculates Robert's total earnings --/
def total_earnings (r : RobertFinances) : ℝ :=
  r.basic_salary + 0.1 * r.total_sales

/-- Theorem stating Robert's basic salary --/
theorem robert_basic_salary :
  ∃ (r : RobertFinances),
    r.total_sales = 23600 ∧
    r.monthly_expenses = 2888 ∧
    0.8 * (total_earnings r) = r.monthly_expenses ∧
    r.basic_salary = 1250 := by
  sorry


end robert_basic_salary_l221_22148


namespace tangent_parabola_hyperbola_l221_22104

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 5

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m * x^2 = 1

/-- Tangency condition -/
def are_tangent (m : ℝ) : Prop := ∃ (x y : ℝ), parabola x y ∧ hyperbola m x y ∧
  ∀ (x' y' : ℝ), parabola x' y' ∧ hyperbola m x' y' → (x' = x ∧ y' = y)

theorem tangent_parabola_hyperbola (m : ℝ) :
  are_tangent m ↔ (m = 10 + 2 * Real.sqrt 6 ∨ m = 10 - 2 * Real.sqrt 6) :=
sorry

end tangent_parabola_hyperbola_l221_22104


namespace solution_to_linear_equation_l221_22117

theorem solution_to_linear_equation (x y m : ℝ) : 
  x = 1 → y = 3 → x - 2 * y = m → m = -5 := by
  sorry

end solution_to_linear_equation_l221_22117


namespace shared_edge_angle_l221_22165

-- Define the angle of a regular decagon
def decagon_angle : ℝ := 144

-- Define the angle of a square
def square_angle : ℝ := 90

-- Theorem statement
theorem shared_edge_angle (x : ℝ) : 
  x + 36 + (360 - decagon_angle) + square_angle = 360 → x = 18 := by
  sorry

end shared_edge_angle_l221_22165


namespace reeya_average_is_67_l221_22139

def reeya_scores : List ℝ := [55, 67, 76, 82, 55]

theorem reeya_average_is_67 : 
  (reeya_scores.sum / reeya_scores.length : ℝ) = 67 := by
  sorry

end reeya_average_is_67_l221_22139


namespace focaccia_price_is_four_l221_22136

/-- The price of a focaccia loaf given Sean's Sunday purchases -/
def focaccia_price : ℝ :=
  let almond_croissant : ℝ := 4.50
  let salami_cheese_croissant : ℝ := 4.50
  let plain_croissant : ℝ := 3.00
  let latte : ℝ := 2.50
  let total_spent : ℝ := 21.00
  total_spent - (almond_croissant + salami_cheese_croissant + plain_croissant + 2 * latte)

theorem focaccia_price_is_four : focaccia_price = 4.00 := by
  sorry

end focaccia_price_is_four_l221_22136


namespace systematic_sampling_theorem_l221_22182

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  students_per_group : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Calculates the number of the selected student in a given group. -/
def selected_number_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.selected_number + s.students_per_group * (group - s.selected_group)

/-- Theorem stating that if student 12 is selected from group 3, 
    then student 37 will be selected from group 8 in a systematic sampling of 50 students. -/
theorem systematic_sampling_theorem (s : SystematicSampling) :
  s.total_students = 50 ∧ 
  s.num_groups = 10 ∧ 
  s.students_per_group = 5 ∧ 
  s.selected_number = 12 ∧ 
  s.selected_group = 3 →
  selected_number_in_group s 8 = 37 := by
  sorry

end systematic_sampling_theorem_l221_22182


namespace parabola_vertex_l221_22168

/-- The equation of a parabola is x^2 - 4x + 3y + 8 = 0. -/
def parabola_equation (x y : ℝ) : Prop := x^2 - 4*x + 3*y + 8 = 0

/-- The vertex of a parabola is the point where it reaches its maximum or minimum y-value. -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x' y', eq x' y' → y ≤ y' ∨ y ≥ y'

/-- The vertex of the parabola defined by x^2 - 4x + 3y + 8 = 0 is (2, -4/3). -/
theorem parabola_vertex : is_vertex 2 (-4/3) parabola_equation := by sorry

end parabola_vertex_l221_22168


namespace probability_two_sunny_days_l221_22123

/-- The probability of exactly k sunny days in n days, given the probability of a sunny day --/
def probability_k_sunny_days (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The number of days in the holiday weekend --/
def num_days : ℕ := 5

/-- The probability of a sunny day --/
def prob_sunny : ℝ := 0.3

/-- The number of desired sunny days --/
def desired_sunny_days : ℕ := 2

theorem probability_two_sunny_days :
  probability_k_sunny_days num_days desired_sunny_days prob_sunny = 0.3087 := by
  sorry

end probability_two_sunny_days_l221_22123


namespace tangent_slope_at_zero_l221_22179

-- Define the function
def f (x : ℝ) : ℝ := (2*x - 1)^3

-- State the theorem
theorem tangent_slope_at_zero : 
  (deriv f) 0 = 6 := by sorry

end tangent_slope_at_zero_l221_22179


namespace M_intersect_N_eq_open_zero_one_l221_22109

-- Define set M
def M : Set ℝ := {x | x^2 + x - 2 < 0}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Theorem statement
theorem M_intersect_N_eq_open_zero_one : M ∩ N = Set.Ioo 0 1 := by sorry

end M_intersect_N_eq_open_zero_one_l221_22109


namespace negation_of_existence_sqrt_leq_x_minus_one_negation_l221_22146

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) := by sorry

theorem sqrt_leq_x_minus_one_negation :
  (¬ ∃ x > 0, Real.sqrt x ≤ x - 1) ↔ (∀ x > 0, Real.sqrt x > x - 1) := by sorry

end negation_of_existence_sqrt_leq_x_minus_one_negation_l221_22146


namespace matrix_equation_solution_l221_22127

theorem matrix_equation_solution (A B : Matrix (Fin 2) (Fin 2) ℝ) :
  A * B = A - B →
  A * B = ![![7, -2], ![3, -1]] →
  B * A = ![![8, -2], ![3, 0]] := by sorry

end matrix_equation_solution_l221_22127


namespace sedan_count_l221_22174

theorem sedan_count (trucks sedans motorcycles : ℕ) : 
  trucks * 7 = sedans * 3 →
  sedans * 2 = motorcycles * 7 →
  motorcycles = 2600 →
  sedans = 9100 := by
sorry

end sedan_count_l221_22174


namespace min_value_exponential_product_l221_22147

theorem min_value_exponential_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 3) :
  Real.exp (1 / a) * Real.exp (1 / b) ≥ Real.exp 3 := by
  sorry

end min_value_exponential_product_l221_22147


namespace triangle_circumcircle_radius_l221_22142

theorem triangle_circumcircle_radius 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π) -- Sum of angles in a triangle
  (h5 : Real.sin C + Real.sin B = 4 * Real.sin A) -- Given condition
  (h6 : a = 2) -- Given condition
  (h7 : a = 2 * Real.sin (A/2) * R) -- Relation between side and circumradius
  (h8 : b = 2 * Real.sin (B/2) * R) -- Relation between side and circumradius
  (h9 : c = 2 * Real.sin (C/2) * R) -- Relation between side and circumradius
  (h10 : ∀ R' > 0, R ≤ R') -- R is the minimum possible radius
  : R = 8 * Real.sqrt 15 / 15 := by
  sorry

end triangle_circumcircle_radius_l221_22142


namespace negation_of_proposition_l221_22150

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - 2*x + 1 ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 2*x + 1 < 0) :=
by sorry

end negation_of_proposition_l221_22150


namespace flu_virus_diameter_scientific_notation_l221_22159

theorem flu_virus_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000054 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 5.4 ∧ n = -6 :=
sorry

end flu_virus_diameter_scientific_notation_l221_22159


namespace lcm_9_12_15_l221_22138

theorem lcm_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end lcm_9_12_15_l221_22138


namespace digit_150_of_1_13_l221_22197

/-- The decimal representation of 1/13 as a sequence of digits -/
def decimal_rep_1_13 : ℕ → Fin 10 := fun n => 
  match n % 6 with
  | 0 => 0
  | 1 => 7
  | 2 => 6
  | 3 => 9
  | 4 => 2
  | 5 => 3
  | _ => 0 -- This case is unreachable, but needed for exhaustiveness

/-- The 150th digit after the decimal point in the decimal representation of 1/13 is 3 -/
theorem digit_150_of_1_13 : decimal_rep_1_13 150 = 3 := by
  sorry

end digit_150_of_1_13_l221_22197


namespace a_worked_days_proof_l221_22121

/-- The number of days A needs to complete the entire work alone -/
def a_complete_days : ℝ := 40

/-- The number of days B needs to complete the entire work alone -/
def b_complete_days : ℝ := 60

/-- The number of days B needs to complete the remaining work after A leaves -/
def b_remaining_days : ℝ := 45

/-- The number of days A worked before leaving -/
def a_worked_days : ℝ := 10

theorem a_worked_days_proof :
  (1 / a_complete_days * a_worked_days) + (b_remaining_days / b_complete_days) = 1 :=
sorry

end a_worked_days_proof_l221_22121


namespace ace_diamond_probability_l221_22166

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of Diamonds in a standard deck -/
def NumDiamonds : ℕ := 13

/-- Probability of drawing an Ace as the first card and a Diamond as the second card -/
def prob_ace_then_diamond (deck : ℕ) (aces : ℕ) (diamonds : ℕ) : ℚ :=
  (aces : ℚ) / (deck : ℚ) * (diamonds : ℚ) / ((deck - 1) : ℚ)

theorem ace_diamond_probability :
  prob_ace_then_diamond StandardDeck NumAces NumDiamonds = 1 / StandardDeck :=
sorry

end ace_diamond_probability_l221_22166


namespace unique_positive_integer_solution_l221_22113

theorem unique_positive_integer_solution :
  ∀ x y : ℕ+, 2 * x^2 + 5 * y^2 = 11 * (x * y - 11) → x = 14 ∧ y = 27 :=
by
  sorry

end unique_positive_integer_solution_l221_22113


namespace student_committee_candidates_l221_22170

theorem student_committee_candidates (n : ℕ) : n * (n - 1) = 72 → n = 9 := by
  sorry

end student_committee_candidates_l221_22170


namespace probability_of_opening_classroom_door_l221_22181

/-- Represents a keychain with a total number of keys and a number of keys that can open a specific door. -/
structure Keychain where
  total_keys : ℕ
  opening_keys : ℕ
  h_opening_keys_le_total : opening_keys ≤ total_keys

/-- Calculates the probability of randomly selecting a key that can open the door. -/
def probability_of_opening (k : Keychain) : ℚ :=
  k.opening_keys / k.total_keys

/-- The class monitor's keychain. -/
def class_monitor_keychain : Keychain :=
  { total_keys := 5
    opening_keys := 2
    h_opening_keys_le_total := by norm_num }

theorem probability_of_opening_classroom_door :
  probability_of_opening class_monitor_keychain = 2 / 5 := by
  sorry

end probability_of_opening_classroom_door_l221_22181


namespace magnitude_of_complex_power_l221_22190

theorem magnitude_of_complex_power (z : ℂ) (n : ℕ) (h : z = 4/5 + 3/5 * I) :
  Complex.abs (z^n) = 1 :=
by
  sorry

end magnitude_of_complex_power_l221_22190


namespace combined_salaries_l221_22193

/-- The combined salaries of A, C, D, and E given B's salary and the average salary of all five. -/
theorem combined_salaries 
  (salary_B : ℕ) 
  (average_salary : ℕ) 
  (num_individuals : ℕ) 
  (h1 : salary_B = 5000)
  (h2 : average_salary = 8400)
  (h3 : num_individuals = 5) :
  average_salary * num_individuals - salary_B = 37000 := by
  sorry

end combined_salaries_l221_22193


namespace intersection_slope_l221_22151

/-- Given two lines that intersect at a specific point, prove the slope of one line. -/
theorem intersection_slope (m : ℝ) : 
  (∀ x y, y = 5 * x + 3 → (x = 2 ∧ y = 13)) →  -- Line p passes through (2, 13)
  (∀ x y, y = m * x + 1 → (x = 2 ∧ y = 13)) →  -- Line q passes through (2, 13)
  m = 6 := by
sorry

end intersection_slope_l221_22151


namespace unique_number_exists_l221_22110

theorem unique_number_exists : ∃! x : ℕ, (∃ k : ℕ, 3 * x = 9 * k) ∧ 4 * x = 108 := by
  sorry

end unique_number_exists_l221_22110


namespace point_in_third_quadrant_l221_22135

def point : ℝ × ℝ := (-3, -2)

def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem point_in_third_quadrant : in_third_quadrant point := by
  sorry

end point_in_third_quadrant_l221_22135


namespace inequalities_hold_l221_22141

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (a^2 + b^2 ≥ 8) ∧ (1/(a*b) ≥ 1/4) := by
  sorry

end inequalities_hold_l221_22141


namespace square_side_length_l221_22140

theorem square_side_length (square_area rectangle_area : ℝ) 
  (rectangle_width rectangle_length : ℝ) (h1 : rectangle_width = 4) 
  (h2 : rectangle_length = 4) (h3 : square_area = rectangle_area) 
  (h4 : rectangle_area = rectangle_width * rectangle_length) : 
  ∃ (side_length : ℝ), side_length * side_length = square_area ∧ side_length = 4 :=
by
  sorry

end square_side_length_l221_22140


namespace largest_six_digit_number_l221_22185

/-- Represents a six-digit number where each digit, starting from the third,
    is the sum of the two preceding digits. -/
structure SixDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  h1 : c = a + b
  h2 : d = b + c
  h3 : e = c + d
  h4 : f = d + e
  h5 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10

/-- Converts a SixDigitNumber to its numerical value -/
def toNumber (n : SixDigitNumber) : Nat :=
  100000 * n.a + 10000 * n.b + 1000 * n.c + 100 * n.d + 10 * n.e + n.f

/-- The largest SixDigitNumber is 303369 -/
theorem largest_six_digit_number :
  ∀ n : SixDigitNumber, toNumber n ≤ 303369 := by
  sorry

end largest_six_digit_number_l221_22185


namespace unique_solution_x_squared_minus_x_minus_one_l221_22122

theorem unique_solution_x_squared_minus_x_minus_one (x : ℝ) :
  x^2 - x - 1 = (x + 1)^0 ↔ x = 2 := by
  sorry

end unique_solution_x_squared_minus_x_minus_one_l221_22122


namespace soda_costs_80_cents_l221_22189

/-- The cost of a burger in cents -/
def burger_cost : ℕ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

/-- The cost of fries in cents -/
def fries_cost : ℕ := sorry

/-- Alice's purchase equation -/
axiom alice_purchase : 5 * burger_cost + 3 * soda_cost + 2 * fries_cost = 520

/-- Bill's purchase equation -/
axiom bill_purchase : 3 * burger_cost + 2 * soda_cost + fries_cost = 340

/-- Theorem stating that a soda costs 80 cents -/
theorem soda_costs_80_cents : soda_cost = 80 := by sorry

end soda_costs_80_cents_l221_22189


namespace empty_set_subset_subset_transitive_l221_22114

-- Define the empty set
def emptySet : Set α := ∅

-- Define subset relation
def isSubset (A B : Set α) : Prop := ∀ x, x ∈ A → x ∈ B

-- Theorem 1: The empty set is a subset of any set
theorem empty_set_subset (S : Set α) : isSubset emptySet S := by sorry

-- Theorem 2: Transitivity of subset relation
theorem subset_transitive (A B C : Set α) 
  (h1 : isSubset A B) (h2 : isSubset B C) : isSubset A C := by sorry

end empty_set_subset_subset_transitive_l221_22114


namespace specific_triangle_area_l221_22157

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The altitude to the base
  altitude : ℝ
  -- The perimeter of the triangle
  perimeter : ℝ
  -- The ratio of equal sides to base (represented as two integers)
  ratio_equal_to_base : ℕ × ℕ
  -- Condition: altitude is positive
  altitude_pos : altitude > 0
  -- Condition: perimeter is positive
  perimeter_pos : perimeter > 0
  -- Condition: ratio components are positive
  ratio_pos : ratio_equal_to_base.1 > 0 ∧ ratio_equal_to_base.2 > 0

/-- The area of an isosceles triangle with given properties -/
def triangle_area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of the specific isosceles triangle is 75 -/
theorem specific_triangle_area :
  ∃ t : IsoscelesTriangle,
    t.altitude = 10 ∧
    t.perimeter = 40 ∧
    t.ratio_equal_to_base = (5, 3) ∧
    triangle_area t = 75 :=
  sorry

end specific_triangle_area_l221_22157


namespace determinant_of_cubic_roots_l221_22125

/-- Given a, b, c are roots of x^3 - 2px + q = 0, 
    prove that the determinant of the matrix is 5 - 6p + q -/
theorem determinant_of_cubic_roots (p q a b c : ℝ) : 
  a^3 - 2*p*a + q = 0 → 
  b^3 - 2*p*b + q = 0 → 
  c^3 - 2*p*c + q = 0 → 
  Matrix.det !![2 + a, 1, 1; 1, 2 + b, 1; 1, 1, 2 + c] = 5 - 6*p + q := by
  sorry

end determinant_of_cubic_roots_l221_22125


namespace shaded_region_perimeter_square_area_l221_22154

theorem shaded_region_perimeter_square_area (PS PQ QR RS : ℝ) : 
  PS = 4 ∧ PQ + QR + RS = PS →
  let shaded_perimeter := (π/2) * (PS + PQ + QR + RS)
  let square_side := shaded_perimeter / 4
  square_side ^ 2 = π ^ 2 := by
  sorry

end shaded_region_perimeter_square_area_l221_22154


namespace log_xy_value_l221_22163

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^4) = 2) (h2 : Real.log (x^3 * y) = 2) :
  Real.log (x * y) = 10 / 11 := by
sorry

end log_xy_value_l221_22163


namespace sector_radius_l221_22153

/-- Given a circular sector with central angle 5π/7 and perimeter 5π + 14, its radius is 7. -/
theorem sector_radius (r : ℝ) : 
  (5 / 7 : ℝ) * π * r + 2 * r = 5 * π + 14 → r = 7 := by
  sorry

end sector_radius_l221_22153


namespace range_of_m_l221_22137

-- Define the conditions
def p (x : ℝ) : Prop := -x^2 + 7*x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - 4*m^2 ≤ 0

-- Define the main theorem
theorem range_of_m :
  (∀ x m : ℝ, (¬(p x) → ¬(q x m)) ∧ (∃ x : ℝ, q x m ∧ p x)) →
  (∀ m : ℝ, -1 ≤ m ∧ m ≤ 1) ∧ (∃ m : ℝ, m = -1 ∨ m = 1) :=
sorry

end range_of_m_l221_22137


namespace max_triangles_six_lines_l221_22100

/-- A configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  on_plane : Bool

/-- Counts the number of equilateral triangles formed by line intersections -/
def count_equilateral_triangles (config : LineConfiguration) : ℕ :=
  sorry

/-- The maximum number of equilateral triangles for a given configuration -/
def max_equilateral_triangles (config : LineConfiguration) : ℕ :=
  sorry

/-- Theorem: The maximum number of equilateral triangles formed by six lines on a plane is 8 -/
theorem max_triangles_six_lines :
  ∀ (config : LineConfiguration),
    config.num_lines = 6 ∧ config.on_plane →
    max_equilateral_triangles config = 8 :=
by sorry

end max_triangles_six_lines_l221_22100


namespace domain_of_g_l221_22111

-- Define the function f with domain [-1, 2]
def f : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- Define the function g(x) = f(2x+1)
def g (x : ℝ) : Prop := (2*x + 1) ∈ f

-- Theorem stating that the domain of g is [-1, 1/2]
theorem domain_of_g : 
  {x : ℝ | g x} = {x : ℝ | -1 ≤ x ∧ x ≤ 1/2} :=
by sorry

end domain_of_g_l221_22111


namespace train_speed_l221_22149

/-- Given a train of length 360 m passing a platform of length 130 m in 39.2 seconds,
    prove that the speed of the train is 45 km/hr. -/
theorem train_speed (train_length platform_length time_to_pass : ℝ) 
  (h1 : train_length = 360)
  (h2 : platform_length = 130)
  (h3 : time_to_pass = 39.2) : 
  (train_length + platform_length) / time_to_pass * 3.6 = 45 := by
  sorry

#check train_speed

end train_speed_l221_22149


namespace solve_equation_one_solve_equation_two_l221_22161

-- Equation 1
theorem solve_equation_one (x : ℝ) : 1 - 2 * (2 * x + 3) = -3 * (2 * x + 1) ↔ x = 1 := by sorry

-- Equation 2
theorem solve_equation_two (x : ℝ) : (x - 3) / 2 - (4 * x + 1) / 5 = 1 ↔ x = -9 := by sorry

end solve_equation_one_solve_equation_two_l221_22161


namespace probability_of_specific_colors_l221_22164

def black_balls : ℕ := 5
def white_balls : ℕ := 7
def green_balls : ℕ := 2
def blue_balls : ℕ := 3
def red_balls : ℕ := 4

def total_balls : ℕ := black_balls + white_balls + green_balls + blue_balls + red_balls

def favorable_outcomes : ℕ := black_balls * green_balls * red_balls

def total_outcomes : ℕ := (total_balls.choose 3)

theorem probability_of_specific_colors : 
  (favorable_outcomes : ℚ) / total_outcomes = 4 / 133 := by sorry

end probability_of_specific_colors_l221_22164


namespace quadratic_one_solution_l221_22116

theorem quadratic_one_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → a = 0 ∨ a = 1 := by
  sorry

end quadratic_one_solution_l221_22116


namespace simplify_and_rationalize_l221_22176

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 5) * (Real.sqrt 3 / Real.sqrt 6) * (Real.sqrt 4 / Real.sqrt 8) * (Real.sqrt 5 / Real.sqrt 9) = 1 / 3 := by
  sorry

end simplify_and_rationalize_l221_22176


namespace octahedron_tetrahedron_volume_ratio_l221_22187

/-- The volume of a regular tetrahedron with edge length a -/
noncomputable def tetrahedron_volume (a : ℝ) : ℝ := sorry

/-- The volume of a regular octahedron with edge length a -/
noncomputable def octahedron_volume (a : ℝ) : ℝ := sorry

/-- Theorem stating that the volume of a regular octahedron is 4 times 
    the volume of a regular tetrahedron with the same edge length -/
theorem octahedron_tetrahedron_volume_ratio (a : ℝ) (h : a > 0) : 
  octahedron_volume a = 4 * tetrahedron_volume a := by sorry

end octahedron_tetrahedron_volume_ratio_l221_22187


namespace barry_vitamin_d3_serving_size_l221_22120

/-- Calculates the daily serving size of capsules given the total number of days,
    capsules per bottle, and number of bottles. -/
def daily_serving_size (days : ℕ) (capsules_per_bottle : ℕ) (bottles : ℕ) : ℕ :=
  (capsules_per_bottle * bottles) / days

theorem barry_vitamin_d3_serving_size :
  let days : ℕ := 180
  let capsules_per_bottle : ℕ := 60
  let bottles : ℕ := 6
  daily_serving_size days capsules_per_bottle bottles = 2 := by
  sorry

end barry_vitamin_d3_serving_size_l221_22120


namespace h_value_at_4_l221_22186

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x - 5

-- Define the properties of h
def is_valid_h (h : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ),
    (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    (∀ x, h x = 0 ↔ x = a^2 ∨ x = b^2 ∨ x = c^2) ∧
    (h 1 = 2)

-- Theorem statement
theorem h_value_at_4 (h : ℝ → ℝ) (hvalid : is_valid_h h) : h 4 = 9 := by
  sorry

end h_value_at_4_l221_22186


namespace split_2017_implies_45_l221_22194

-- Define the sum of consecutive integers from 2 to n
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2 - 1

-- Define the property that 2017 is in the split of m³
def split_contains_2017 (m : ℕ) : Prop :=
  m > 1 ∧ sum_to_n m ≥ 1008 ∧ sum_to_n (m - 1) < 1008

theorem split_2017_implies_45 :
  ∀ m : ℕ, split_contains_2017 m → m = 45 :=
by sorry

end split_2017_implies_45_l221_22194


namespace julie_lettuce_purchase_l221_22167

/-- The total pounds of lettuce Julie bought -/
def total_lettuce (green_cost red_cost price_per_pound : ℚ) : ℚ :=
  green_cost / price_per_pound + red_cost / price_per_pound

/-- Proof that Julie bought 7 pounds of lettuce -/
theorem julie_lettuce_purchase : 
  total_lettuce 8 6 2 = 7 := by
  sorry

end julie_lettuce_purchase_l221_22167


namespace perpendicular_planes_from_line_relationships_l221_22191

/-- Two lines are different if they are not equal -/
def different_lines (m n : Line) : Prop := m ≠ n

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two lines are parallel -/
def lines_parallel (m n : Line) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_to_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perpendicular (α β : Plane) : Prop := sorry

theorem perpendicular_planes_from_line_relationships 
  (m n : Line) (α β : Plane) 
  (h1 : different_lines m n)
  (h2 : different_planes α β)
  (h3 : line_perpendicular_to_plane m α)
  (h4 : lines_parallel m n)
  (h5 : line_parallel_to_plane n β) :
  planes_perpendicular α β :=
sorry

end perpendicular_planes_from_line_relationships_l221_22191


namespace books_about_science_l221_22199

theorem books_about_science 
  (total_books : ℕ) 
  (school_books : ℕ) 
  (sports_books : ℕ) 
  (h1 : total_books = 85) 
  (h2 : school_books = 19) 
  (h3 : sports_books = 35) :
  total_books - (school_books + sports_books) = 31 :=
by sorry

end books_about_science_l221_22199


namespace f_properties_l221_22132

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x / Real.log 2

theorem f_properties :
  (∀ x > 0, f (-x) ≠ -f x ∧ f (-x) ≠ f x) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
sorry

end f_properties_l221_22132


namespace no_positive_integer_solutions_l221_22133

theorem no_positive_integer_solutions 
  (p : ℕ) (hp : Nat.Prime p) (hp_mod : p % 4 = 3) (n : ℕ+) :
  ¬ ∃ (x y : ℕ+), p^(n : ℕ) = x^2 + y^2 := by
  sorry

end no_positive_integer_solutions_l221_22133


namespace imaginary_part_of_reciprocal_plus_one_l221_22106

theorem imaginary_part_of_reciprocal_plus_one (z : ℂ) (x y : ℝ) 
  (h1 : z = x + y * I) 
  (h2 : z ≠ x) -- z is nonreal
  (h3 : Complex.abs z = 1) : 
  Complex.im (1 / (1 + z)) = -y / (2 * (1 + x)) :=
by sorry

end imaginary_part_of_reciprocal_plus_one_l221_22106


namespace tangent_slope_angle_l221_22171

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 4*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 4

-- Define the point of interest
def point : ℝ × ℝ := (1, -3)

-- Theorem statement
theorem tangent_slope_angle :
  let slope := f' point.1
  let angle := Real.arctan slope
  angle = 3 * π / 4 := by sorry

end tangent_slope_angle_l221_22171


namespace women_who_bought_apples_l221_22126

/-- The number of women who bought apples -/
def num_women : ℕ := 3

/-- The number of men who bought apples -/
def num_men : ℕ := 2

/-- The number of apples each man bought -/
def apples_per_man : ℕ := 30

/-- The additional number of apples each woman bought compared to each man -/
def additional_apples_per_woman : ℕ := 20

/-- The total number of apples bought -/
def total_apples : ℕ := 210

theorem women_who_bought_apples :
  num_women * (apples_per_man + additional_apples_per_woman) +
  num_men * apples_per_man = total_apples :=
by sorry

end women_who_bought_apples_l221_22126


namespace similar_triangles_not_necessarily_equal_sides_l221_22155

-- Define a structure for triangles
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ ∧
    t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c ∧
    t1.a / t2.a = k

-- Define a property for equal corresponding sides
def equal_sides (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Theorem statement
theorem similar_triangles_not_necessarily_equal_sides :
  ¬ (∀ t1 t2 : Triangle, similar t1 t2 → equal_sides t1 t2) :=
sorry

end similar_triangles_not_necessarily_equal_sides_l221_22155


namespace sum_one_to_fortyfive_base6_l221_22156

/-- Represents a number in base 6 --/
def Base6 := ℕ

/-- Converts a natural number to its base 6 representation --/
def to_base6 (n : ℕ) : Base6 := sorry

/-- Converts a base 6 number to its natural number representation --/
def from_base6 (b : Base6) : ℕ := sorry

/-- Adds two base 6 numbers --/
def add_base6 (a b : Base6) : Base6 := sorry

/-- Multiplies two base 6 numbers --/
def mul_base6 (a b : Base6) : Base6 := sorry

/-- Divides a base 6 number by 2 --/
def div2_base6 (a : Base6) : Base6 := sorry

/-- Calculates the sum of an arithmetic sequence in base 6 --/
def sum_arithmetic_base6 (first last : Base6) : Base6 :=
  let n := add_base6 (from_base6 last) (to_base6 1)
  div2_base6 (mul_base6 n (add_base6 first last))

/-- The main theorem to be proved --/
theorem sum_one_to_fortyfive_base6 :
  sum_arithmetic_base6 (to_base6 1) (to_base6 45) = to_base6 2003 := by sorry

end sum_one_to_fortyfive_base6_l221_22156


namespace odd_function_log_property_l221_22192

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_log_property (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_pos : ∀ x > 0, f x = Real.log (x + 1)) : 
  ∀ x < 0, f x = -Real.log (1 - x) :=
by sorry

end odd_function_log_property_l221_22192


namespace message_spread_time_l221_22103

/-- The number of people who have received the message after n minutes -/
def people_reached (n : ℕ) : ℕ := 2^(n + 1) - 1

/-- The time required for the message to reach 2047 people -/
def time_to_reach_2047 : ℕ := 10

theorem message_spread_time :
  people_reached time_to_reach_2047 = 2047 :=
sorry

end message_spread_time_l221_22103


namespace cubic_identity_l221_22183

theorem cubic_identity (a b c : ℝ) : 
  a^3*(b^3 - c^3) + b^3*(c^3 - a^3) + c^3*(a^3 - b^3) = 
  (a - b)*(b - c)*(c - a) * ((a^2 + a*b + b^2)*(b^2 + b*c + c^2)*(c^2 + c*a + a^2)) :=
by sorry

end cubic_identity_l221_22183


namespace sandwich_shop_jalapeno_requirement_l221_22198

/-- Represents the number of jalapeno peppers required for a day's operation --/
def jalapeno_peppers_required (strips_per_sandwich : ℕ) (slices_per_pepper : ℕ) 
  (minutes_per_sandwich : ℕ) (hours_of_operation : ℕ) : ℕ :=
  let peppers_per_sandwich := strips_per_sandwich / slices_per_pepper
  let sandwiches_per_hour := 60 / minutes_per_sandwich
  let peppers_per_hour := peppers_per_sandwich * sandwiches_per_hour
  peppers_per_hour * hours_of_operation

/-- Theorem stating the number of jalapeno peppers required for the Sandwich Shop's 8-hour day --/
theorem sandwich_shop_jalapeno_requirement :
  jalapeno_peppers_required 4 8 5 8 = 48 := by
  sorry

end sandwich_shop_jalapeno_requirement_l221_22198


namespace curve_symmetry_l221_22158

/-- A curve f is symmetric with respect to the line x - y - 3 = 0 if and only if
    it can be expressed as f(y+3, x-3) = 0 for all x and y. -/
theorem curve_symmetry (f : ℝ → ℝ → ℝ) :
  (∀ x y, f x y = 0 ↔ f (y + 3) (x - 3) = 0) ↔
  (∀ x y, (x - y = 3) → (f x y = 0 ↔ f y x = 0)) :=
sorry

end curve_symmetry_l221_22158


namespace sum_of_multiples_of_6_and_9_is_multiple_of_3_l221_22101

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (x y : ℤ) 
  (hx : ∃ m : ℤ, x = 6 * m) 
  (hy : ∃ n : ℤ, y = 9 * n) : 
  ∃ k : ℤ, x + y = 3 * k := by
sorry

end sum_of_multiples_of_6_and_9_is_multiple_of_3_l221_22101


namespace max_weight_proof_l221_22129

def max_weight_single_trip : ℕ := 8750

theorem max_weight_proof (crate_weight_min crate_weight_max : ℕ) 
  (weight_8_crates weight_12_crates : ℕ) :
  crate_weight_min = 150 →
  crate_weight_max = 250 →
  weight_8_crates ≤ 1300 →
  weight_12_crates ≤ 2100 →
  max_weight_single_trip = 8750 := by
  sorry

end max_weight_proof_l221_22129


namespace lukes_trips_l221_22178

/-- Luke's tray-carrying problem -/
theorem lukes_trips (trays_per_trip : ℕ) (trays_table1 : ℕ) (trays_table2 : ℕ)
  (h1 : trays_per_trip = 4)
  (h2 : trays_table1 = 20)
  (h3 : trays_table2 = 16) :
  (trays_table1 + trays_table2) / trays_per_trip = 9 :=
by sorry

end lukes_trips_l221_22178


namespace exists_a_C_is_line_C_passes_through_origin_L_intersects_C_l221_22169

-- Define the curve C
def C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1^2 + a * p.2^2 - 2 * p.1 - 2 * p.2 = 0}

-- Define a straight line
def isLine (S : Set (ℝ × ℝ)) : Prop :=
  ∃ A B C : ℝ, ∀ p : ℝ × ℝ, p ∈ S ↔ A * p.1 + B * p.2 + C = 0

-- Statement 1: C is a straight line for some a
theorem exists_a_C_is_line : ∃ a : ℝ, isLine (C a) := by sorry

-- Statement 2: C passes through (0, 0) for all a
theorem C_passes_through_origin : ∀ a : ℝ, (0, 0) ∈ C a := by sorry

-- Define the line x + 2y = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + 2 * p.2 = 0}

-- Statement 3: When a = 1, L intersects C
theorem L_intersects_C : (C 1) ∩ L ≠ ∅ := by sorry

end exists_a_C_is_line_C_passes_through_origin_L_intersects_C_l221_22169


namespace circle_diameter_l221_22184

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 64 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 16 := by
  sorry

end circle_diameter_l221_22184


namespace fly_path_on_cone_l221_22177

/-- A right circular cone -/
structure Cone where
  base_radius : ℝ
  height : ℝ

/-- A point on the surface of a cone -/
structure ConePoint where
  distance_from_vertex : ℝ

/-- The shortest distance between two points on the surface of a cone -/
def shortest_distance (c : Cone) (p1 p2 : ConePoint) : ℝ := sorry

/-- The theorem statement -/
theorem fly_path_on_cone :
  let c : Cone := { base_radius := 600, height := 200 * Real.sqrt 7 }
  let p1 : ConePoint := { distance_from_vertex := 125 }
  let p2 : ConePoint := { distance_from_vertex := 375 * Real.sqrt 2 }
  shortest_distance c p1 p2 = 625 := by sorry

end fly_path_on_cone_l221_22177


namespace correct_problems_l221_22144

theorem correct_problems (total : ℕ) (h1 : total = 54) : ∃ (correct : ℕ), 
  correct + 2 * correct = total ∧ correct = 18 := by
  sorry

end correct_problems_l221_22144


namespace expand_product_l221_22143

theorem expand_product (x : ℝ) : 5 * (x + 6) * (x^2 + 2*x + 3) = 5*x^3 + 40*x^2 + 75*x + 90 := by
  sorry

end expand_product_l221_22143


namespace rate_of_increase_comparison_l221_22160

theorem rate_of_increase_comparison (x : ℝ) :
  let f (x : ℝ) := 1000 * x
  let g (x : ℝ) := x^2 / 1000
  (0 < x ∧ x < 500000) → (deriv f x > deriv g x) ∧
  (x > 500000) → (deriv f x < deriv g x) := by
  sorry

end rate_of_increase_comparison_l221_22160


namespace quadratic_condition_l221_22145

/-- The condition for a quadratic equation in x with parameter m -/
def is_quadratic_in_x (m : ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, m * x^2 - 3*x = x^2 - m*x + 2 ↔ a * x^2 + b * x + c = 0

/-- Theorem stating that for the given equation to be quadratic in x, m must not equal 1 -/
theorem quadratic_condition (m : ℝ) : is_quadratic_in_x m → m ≠ 1 :=
by sorry

end quadratic_condition_l221_22145


namespace problem_solution_l221_22128

theorem problem_solution (a b : ℚ) 
  (h1 : 2020 * a + 2024 * b = 2030) 
  (h2 : 2022 * a + 2026 * b = 2032) : 
  a - b = -4 := by
sorry

end problem_solution_l221_22128


namespace stevens_height_l221_22172

-- Define the building's height and shadow length
def building_height : ℝ := 50
def building_shadow : ℝ := 25

-- Define Steven's shadow length
def steven_shadow : ℝ := 20

-- Define the theorem
theorem stevens_height :
  ∃ (h : ℝ), h = (building_height / building_shadow) * steven_shadow ∧ h = 40 :=
by sorry

end stevens_height_l221_22172


namespace negation_of_exists_le_zero_is_forall_gt_zero_l221_22115

theorem negation_of_exists_le_zero_is_forall_gt_zero :
  (¬ ∃ x : ℝ, (2 : ℝ) ^ x ≤ 0) ↔ (∀ x : ℝ, (2 : ℝ) ^ x > 0) := by sorry

end negation_of_exists_le_zero_is_forall_gt_zero_l221_22115
