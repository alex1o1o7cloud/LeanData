import Mathlib

namespace hyperbola_eccentricity_l4081_408186

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and right focus F(c, 0),
    if point P on the hyperbola satisfies |FM| = 2|FP| where M is the intersection of the circle
    centered at F with radius 2c and the positive y-axis, then the eccentricity of the hyperbola
    is √3 + 1. -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  let F : ℝ × ℝ := (c, 0)
  let M : ℝ × ℝ := (0, Real.sqrt 3 * c)
  let P : ℝ × ℝ := (c / 2, Real.sqrt 3 / 2 * c)
  (P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1) →  -- P is on the hyperbola
  (Real.sqrt ((M.1 - F.1) ^ 2 + (M.2 - F.2) ^ 2) = 2 * Real.sqrt ((P.1 - F.1) ^ 2 + (P.2 - F.2) ^ 2)) →  -- |FM| = 2|FP|
  (c ^ 2 / a ^ 2 - b ^ 2 / a ^ 2 = 1) →  -- Relation between a, b, and c for a hyperbola
  Real.sqrt (c ^ 2 / a ^ 2) = Real.sqrt 3 + 1  -- Eccentricity is √3 + 1
:= by sorry

end hyperbola_eccentricity_l4081_408186


namespace min_value_theorem_l4081_408147

theorem min_value_theorem (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 10) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (w : ℝ), w = x^2 + y^2 + z^2 + x^2*y → w ≥ min :=
by sorry

end min_value_theorem_l4081_408147


namespace basketball_lineup_combinations_l4081_408173

def team_size : ℕ := 18
def lineup_size : ℕ := 8
def non_pg_players : ℕ := lineup_size - 1

theorem basketball_lineup_combinations :
  (team_size : ℕ) * (Nat.choose (team_size - 1) non_pg_players) = 349864 := by
  sorry

end basketball_lineup_combinations_l4081_408173


namespace complicated_expression_equality_l4081_408199

theorem complicated_expression_equality : 
  Real.sqrt (11 * 13) * (1/3) + 2 * (Real.sqrt 17 / 3) - 4 * (Real.sqrt 7 / 5) = 
  (5 * Real.sqrt 143 + 10 * Real.sqrt 17 - 12 * Real.sqrt 7) / 15 := by
sorry

end complicated_expression_equality_l4081_408199


namespace square_area_from_perimeter_l4081_408163

/-- The area of a square with perimeter 24 is 36 -/
theorem square_area_from_perimeter : 
  ∀ s : Real, 
  (s > 0) → 
  (4 * s = 24) → 
  (s * s = 36) :=
by
  sorry

end square_area_from_perimeter_l4081_408163


namespace die_roll_average_l4081_408119

def die_rolls : List Nat := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]
def next_roll : Nat := 2
def total_rolls : Nat := die_rolls.length + 1

theorem die_roll_average :
  (die_rolls.sum + next_roll) / total_rolls = 3 := by
sorry

end die_roll_average_l4081_408119


namespace region_upper_left_l4081_408184

def line (x y : ℝ) : ℝ := 3 * x - 2 * y - 6

theorem region_upper_left :
  ∀ (x y : ℝ), line x y < 0 →
  ∃ (x' y' : ℝ), x' > x ∧ y' < y ∧ line x' y' = 0 :=
by sorry

end region_upper_left_l4081_408184


namespace set_operations_and_complements_l4081_408156

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 2}

-- Define the theorem
theorem set_operations_and_complements :
  (A ∩ B = {x | -1 ≤ x ∧ x < 2}) ∧
  (A ∪ B = {x | -2 ≤ x ∧ x ≤ 3}) ∧
  ((Uᶜ ∪ (A ∩ B)) = {x | x < -1 ∨ 2 ≤ x}) ∧
  ((Uᶜ ∪ (A ∪ B)) = {x | x < -2 ∨ 3 < x}) :=
by sorry

end set_operations_and_complements_l4081_408156


namespace proportion_with_reciprocals_l4081_408110

theorem proportion_with_reciprocals (a b c d : ℝ) : 
  a / b = c / d →  -- proportion
  b * c = 1 →      -- inner terms are reciprocals
  a = 0.2 →        -- one outer term is 0.2
  d = 5 :=         -- prove the other outer term is 5
by sorry

end proportion_with_reciprocals_l4081_408110


namespace first_triangular_covering_all_remainders_triangular_22_is_253_l4081_408185

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to check if a number covers all remainders modulo 10 -/
def covers_all_remainders (n : ℕ) : Prop :=
  ∀ r : Fin 10, ∃ k : ℕ, k ≤ n ∧ triangular_number k % 10 = r

/-- Main theorem: 22 is the smallest n for which triangular_number n covers all remainders modulo 10 -/
theorem first_triangular_covering_all_remainders :
  (covers_all_remainders 22 ∧ ∀ m < 22, ¬ covers_all_remainders m) :=
sorry

/-- Corollary: The 22nd triangular number is 253 -/
theorem triangular_22_is_253 : triangular_number 22 = 253 :=
sorry

end first_triangular_covering_all_remainders_triangular_22_is_253_l4081_408185


namespace square_of_sum_leq_sum_of_squares_l4081_408168

theorem square_of_sum_leq_sum_of_squares (a b : ℝ) :
  ((a + b) / 2) ^ 2 ≤ (a^2 + b^2) / 2 := by
  sorry

end square_of_sum_leq_sum_of_squares_l4081_408168


namespace vector_magnitude_l4081_408121

def problem (m n : ℝ × ℝ) : Prop :=
  let ⟨mx, my⟩ := m
  let ⟨nx, ny⟩ := n
  (mx * nx + my * ny = 0) ∧  -- m perpendicular to n
  (m.1 - 2 * n.1 = 11) ∧     -- x-component of m - 2n = 11
  (m.2 - 2 * n.2 = -2) ∧     -- y-component of m - 2n = -2
  (mx^2 + my^2 = 25)         -- |m| = 5

theorem vector_magnitude (m n : ℝ × ℝ) :
  problem m n → n.1^2 + n.2^2 = 25 := by
  sorry

end vector_magnitude_l4081_408121


namespace complex_modulus_problem_l4081_408100

theorem complex_modulus_problem (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l4081_408100


namespace max_value_of_largest_integer_l4081_408188

theorem max_value_of_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℝ) / 5 = 45 →
  (max a (max b (max c (max d e)))) - (min a (min b (min c (min d e)))) = 10 →
  max a (max b (max c (max d e))) ≤ 215 :=
sorry

end max_value_of_largest_integer_l4081_408188


namespace percentage_calculation_l4081_408134

theorem percentage_calculation : (200 / 50) * 100 = 400 := by
  sorry

end percentage_calculation_l4081_408134


namespace third_dimension_of_smaller_box_l4081_408170

/-- The length of the third dimension of the smaller box -/
def h : ℕ := sorry

/-- The volume of the larger box -/
def large_box_volume : ℕ := 12 * 14 * 16

/-- The volume of a single smaller box -/
def small_box_volume (h : ℕ) : ℕ := 3 * 7 * h

/-- The number of smaller boxes that fit into the larger box -/
def num_boxes : ℕ := 64

theorem third_dimension_of_smaller_box :
  (num_boxes * small_box_volume h ≤ large_box_volume) → h = 2 := by
  sorry

end third_dimension_of_smaller_box_l4081_408170


namespace sqrt_x_div_sqrt_y_l4081_408154

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/2)^2 + (1/3)^2 = ((1/4)^2 + (1/5)^2) * (13*x)/(41*y)) : 
  Real.sqrt x / Real.sqrt y = 10/3 := by
  sorry

end sqrt_x_div_sqrt_y_l4081_408154


namespace billy_youtube_suggestions_l4081_408138

/-- The number of suggestion sets Billy watches before finding a video he likes -/
def num_sets : ℕ := 5

/-- The number of videos Billy watches from the final set -/
def videos_from_final_set : ℕ := 5

/-- The total number of videos Billy watches -/
def total_videos : ℕ := 65

/-- The number of suggestions generated each time -/
def suggestions_per_set : ℕ := 15

theorem billy_youtube_suggestions :
  (num_sets - 1) * suggestions_per_set + videos_from_final_set = total_videos :=
by sorry

end billy_youtube_suggestions_l4081_408138


namespace arithmetic_sequence_12th_term_l4081_408192

/-- An arithmetic sequence with a₃ = 10 and a₉ = 28 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a 3 = 10 ∧ a 9 = 28

/-- The 12th term of the arithmetic sequence is 37 -/
theorem arithmetic_sequence_12th_term (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 12 = 37 := by
  sorry

end arithmetic_sequence_12th_term_l4081_408192


namespace arithmetic_sqrt_of_nine_l4081_408105

theorem arithmetic_sqrt_of_nine : ∃ x : ℝ, x ≥ 0 ∧ x ^ 2 = 9 ∧ x = 3 := by
  sorry

end arithmetic_sqrt_of_nine_l4081_408105


namespace soccer_balls_theorem_l4081_408101

/-- The number of soccer balls originally purchased by the store -/
def original_balls : ℕ := 130

/-- The wholesale price of each soccer ball -/
def wholesale_price : ℕ := 30

/-- The retail price of each soccer ball -/
def retail_price : ℕ := 45

/-- The number of soccer balls remaining when the profit is calculated -/
def remaining_balls : ℕ := 30

/-- The profit made when there are 30 balls remaining -/
def profit : ℕ := 1500

/-- Theorem stating that the number of originally purchased soccer balls is 130 -/
theorem soccer_balls_theorem :
  (retail_price - wholesale_price) * (original_balls - remaining_balls) = profit :=
by sorry

end soccer_balls_theorem_l4081_408101


namespace cassidy_grounding_l4081_408180

/-- The number of days Cassidy is grounded for lying about her report card -/
def base_grounding : ℕ := 14

/-- The number of extra days Cassidy is grounded for each grade below a B -/
def extra_days_per_grade : ℕ := 3

/-- The number of grades Cassidy got below a B -/
def grades_below_b : ℕ := 4

/-- The total number of days Cassidy is grounded -/
def total_grounding : ℕ := base_grounding + extra_days_per_grade * grades_below_b

theorem cassidy_grounding :
  total_grounding = 26 := by
  sorry

end cassidy_grounding_l4081_408180


namespace external_tangent_intercept_l4081_408176

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculates the common external tangent line with positive slope for two circles -/
def commonExternalTangent (c1 c2 : Circle) : Line :=
  sorry

theorem external_tangent_intercept :
  let c1 : Circle := { center := (2, 4), radius := 5 }
  let c2 : Circle := { center := (14, 9), radius := 10 }
  let tangent := commonExternalTangent c1 c2
  tangent.slope > 0 → tangent.intercept = 912 / 119 :=
sorry

end external_tangent_intercept_l4081_408176


namespace t_value_on_line_l4081_408198

/-- A straight line passing through points (1, 7), (3, 13), (5, 19), and (28, t) -/
def straightLine (t : ℝ) : Prop :=
  ∃ (m c : ℝ),
    (7 = m * 1 + c) ∧
    (13 = m * 3 + c) ∧
    (19 = m * 5 + c) ∧
    (t = m * 28 + c)

/-- Theorem stating that t = 88 for the given straight line -/
theorem t_value_on_line : straightLine 88 := by
  sorry

end t_value_on_line_l4081_408198


namespace mean_equality_implies_y_value_prove_mean_equality_implies_y_value_l4081_408164

theorem mean_equality_implies_y_value : ℝ → Prop :=
  fun y =>
    (6 + 9 + 18) / 3 = (12 + y) / 2 →
    y = 10

-- The proof is omitted
theorem prove_mean_equality_implies_y_value :
  ∃ y : ℝ, mean_equality_implies_y_value y :=
sorry

end mean_equality_implies_y_value_prove_mean_equality_implies_y_value_l4081_408164


namespace root_sum_product_l4081_408179

theorem root_sum_product (p q r : ℂ) : 
  (5 * p ^ 3 - 10 * p ^ 2 + 17 * p - 7 = 0) →
  (5 * q ^ 3 - 10 * q ^ 2 + 17 * q - 7 = 0) →
  (5 * r ^ 3 - 10 * r ^ 2 + 17 * r - 7 = 0) →
  p * q + p * r + q * r = 17 / 5 := by
sorry

end root_sum_product_l4081_408179


namespace factors_of_504_l4081_408128

def number_of_positive_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem factors_of_504 : number_of_positive_factors 504 = 24 := by
  sorry

end factors_of_504_l4081_408128


namespace sum_of_first_5n_integers_l4081_408171

theorem sum_of_first_5n_integers (n : ℕ) : 
  (4*n*(4*n+1))/2 = (n*(n+1))/2 + 210 → (5*n*(5*n+1))/2 = 465 := by
  sorry

end sum_of_first_5n_integers_l4081_408171


namespace equation_solution_l4081_408178

theorem equation_solution : ∃ Z : ℤ, 80 - (5 - (Z + 2 * (7 - 8 - 5))) = 89 ∧ Z = 26 := by
  sorry

end equation_solution_l4081_408178


namespace polynomial_value_at_negative_five_l4081_408140

theorem polynomial_value_at_negative_five (a b c : ℝ) : 
  (5^5 * a + 5^3 * b + 5 * c + 2 = 8) → 
  ((-5)^5 * a + (-5)^3 * b + (-5) * c - 3 = -9) :=
by
  sorry

end polynomial_value_at_negative_five_l4081_408140


namespace intersection_sum_zero_l4081_408182

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 1)^2
def parabola2 (x y : ℝ) : Prop := x - 2 = (y + 1)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

theorem intersection_sum_zero :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧
    (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by sorry

end intersection_sum_zero_l4081_408182


namespace fred_balloons_l4081_408136

theorem fred_balloons (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 709 → given_away = 221 → remaining = initial - given_away → remaining = 488 := by
  sorry

end fred_balloons_l4081_408136


namespace hyperbola_satisfies_conditions_l4081_408149

/-- A hyperbola with the equation 4x² - 9y² = -32 -/
def hyperbola (x y : ℝ) : Prop := 4 * x^2 - 9 * y^2 = -32

/-- The asymptotes of the hyperbola -/
def asymptotes (x y : ℝ) : Prop := (2 * x + 3 * y = 0) ∨ (2 * x - 3 * y = 0)

theorem hyperbola_satisfies_conditions :
  (∀ x y : ℝ, asymptotes x y ↔ (4 * x^2 - 9 * y^2 = 0)) ∧
  hyperbola 1 2 := by sorry

end hyperbola_satisfies_conditions_l4081_408149


namespace x_squared_plus_7x_plus_12_bounds_l4081_408132

theorem x_squared_plus_7x_plus_12_bounds 
  (x : ℝ) (h : x^2 - 7*x + 12 < 0) : 
  48 < x^2 + 7*x + 12 ∧ x^2 + 7*x + 12 < 64 := by
  sorry

end x_squared_plus_7x_plus_12_bounds_l4081_408132


namespace students_interested_in_both_l4081_408158

theorem students_interested_in_both (total : ℕ) (sports : ℕ) (entertainment : ℕ) (neither : ℕ) :
  total = 1400 →
  sports = 1250 →
  entertainment = 952 →
  neither = 60 →
  ∃ x : ℕ, x = 862 ∧
    total = neither + x + (sports - x) + (entertainment - x) :=
by sorry

end students_interested_in_both_l4081_408158


namespace halloween_candy_eaten_l4081_408155

/-- The number of candy pieces eaten on Halloween night -/
def candy_eaten (debby_initial : ℕ) (sister_initial : ℕ) (remaining : ℕ) : ℕ :=
  debby_initial + sister_initial - remaining

/-- Theorem stating the number of candy pieces eaten on Halloween night -/
theorem halloween_candy_eaten :
  candy_eaten 32 42 39 = 35 := by
  sorry

end halloween_candy_eaten_l4081_408155


namespace school_students_count_l4081_408193

theorem school_students_count :
  ∀ (total_students boys girls : ℕ),
  total_students = boys + girls →
  boys = 80 →
  girls = (80 * total_students) / 100 →
  total_students = 400 := by
sorry

end school_students_count_l4081_408193


namespace third_candidate_votes_correct_l4081_408124

/-- The number of votes received by the third candidate in an election with three candidates,
    where two candidates received 7636 and 11628 votes respectively,
    and the winning candidate got 54.336448598130836% of the total votes. -/
def third_candidate_votes : ℕ :=
  let total_votes : ℕ := 7636 + 11628 + 2136
  let winning_votes : ℕ := 11628
  let winning_percentage : ℚ := 54336448598130836 / 100000000000000000
  2136

theorem third_candidate_votes_correct :
  let total_votes : ℕ := 7636 + 11628 + third_candidate_votes
  let winning_votes : ℕ := 11628
  let winning_percentage : ℚ := 54336448598130836 / 100000000000000000
  (winning_votes : ℚ) / (total_votes : ℚ) = winning_percentage :=
by sorry

#eval third_candidate_votes

end third_candidate_votes_correct_l4081_408124


namespace total_albums_l4081_408152

/-- The number of albums each person has -/
structure Albums where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ

/-- The conditions of the problem -/
def album_conditions (a : Albums) : Prop :=
  a.adele = 30 ∧
  a.bridget = a.adele - 15 ∧
  a.katrina = 6 * a.bridget ∧
  a.miriam = 5 * a.katrina

/-- The theorem to be proved -/
theorem total_albums (a : Albums) (h : album_conditions a) : 
  a.adele + a.bridget + a.katrina + a.miriam = 585 := by
  sorry


end total_albums_l4081_408152


namespace sufficient_condition_implies_true_implication_l4081_408196

theorem sufficient_condition_implies_true_implication (p q : Prop) :
  (p → q) → (p → q) := by
  sorry

end sufficient_condition_implies_true_implication_l4081_408196


namespace kishore_savings_percentage_l4081_408172

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 6100
def savings : ℕ := 2400

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous
def total_salary : ℕ := total_expenses + savings

theorem kishore_savings_percentage :
  (savings : ℚ) / (total_salary : ℚ) = 1 / 10 := by sorry

end kishore_savings_percentage_l4081_408172


namespace V_min_at_2_minus_sqrt2_l4081_408114

open Real

/-- The volume function V(a) -/
noncomputable def V (a : ℝ) : ℝ := 
  π * ((3-a) * (log (3-a))^2 + 2*a * log (3-a) - (1-a) * (log (1-a))^2 - 2*a * log (1-a))

/-- The theorem stating that V(a) has a minimum at a = 2 - √2 -/
theorem V_min_at_2_minus_sqrt2 :
  ∃ (a : ℝ), 0 < a ∧ a < 1 ∧ 
  (∀ (x : ℝ), 0 < x → x < 1 → V x ≥ V a) ∧ 
  a = 2 - sqrt 2 := by
  sorry

/-- Verify that 2 - √2 is indeed between 0 and 1 -/
lemma two_minus_sqrt_two_in_range : 
  0 < 2 - sqrt 2 ∧ 2 - sqrt 2 < 1 := by
  sorry

end V_min_at_2_minus_sqrt2_l4081_408114


namespace quadratic_root_existence_l4081_408107

theorem quadratic_root_existence (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  ∃ x : ℝ, (x^2 + a*x + b = 0) ∨ (x^2 + c*x + d = 0) := by
  sorry

end quadratic_root_existence_l4081_408107


namespace salon_customers_l4081_408151

/-- Represents the daily operations of a hair salon -/
structure Salon where
  total_cans : ℕ
  extra_cans : ℕ
  cans_per_customer : ℕ

/-- Calculates the number of customers given a salon's daily operations -/
def customers (s : Salon) : ℕ :=
  (s.total_cans - s.extra_cans) / s.cans_per_customer

/-- Theorem stating that a salon with the given parameters has 14 customers per day -/
theorem salon_customers :
  let s : Salon := {
    total_cans := 33,
    extra_cans := 5,
    cans_per_customer := 2
  }
  customers s = 14 := by
  sorry

end salon_customers_l4081_408151


namespace second_derivative_implies_m_l4081_408190

/-- Given a function f(x) = 2/x, prove that if its second derivative at m is -1/2, then m = -2 -/
theorem second_derivative_implies_m (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = 2 / x) →
  (deriv^[2] f m = -1/2) →
  m = -2 :=
by sorry

end second_derivative_implies_m_l4081_408190


namespace matrix_power_2019_l4081_408139

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2019 :
  A ^ 2019 = !![1, 0; 4038, 1] := by sorry

end matrix_power_2019_l4081_408139


namespace even_product_probability_spinners_l4081_408120

/-- Represents a spinner with sections labeled by natural numbers -/
structure Spinner :=
  (sections : List ℕ)

/-- The probability of getting an even product when spinning two spinners -/
def evenProductProbability (spinnerA spinnerB : Spinner) : ℚ :=
  sorry

/-- Spinner A with 6 equal sections: 1, 1, 2, 2, 3, 4 -/
def spinnerA : Spinner :=
  ⟨[1, 1, 2, 2, 3, 4]⟩

/-- Spinner B with 4 equal sections: 1, 3, 5, 6 -/
def spinnerB : Spinner :=
  ⟨[1, 3, 5, 6]⟩

/-- Theorem stating that the probability of getting an even product
    when spinning spinnerA and spinnerB is 5/8 -/
theorem even_product_probability_spinners :
  evenProductProbability spinnerA spinnerB = 5/8 :=
sorry

end even_product_probability_spinners_l4081_408120


namespace total_sales_correct_l4081_408144

/-- Calculates the total amount of money made from selling bracelets and necklaces. -/
def calculate_total_sales (
  bracelet_price : ℕ)
  (bracelet_discount_price : ℕ)
  (necklace_price : ℕ)
  (necklace_discount_price : ℕ)
  (regular_bracelets_sold : ℕ)
  (discounted_bracelets_sold : ℕ)
  (regular_necklaces_sold : ℕ)
  (discounted_necklace_sets_sold : ℕ) : ℕ :=
  (regular_bracelets_sold * bracelet_price) +
  (discounted_bracelets_sold / 2 * bracelet_discount_price) +
  (regular_necklaces_sold * necklace_price) +
  (discounted_necklace_sets_sold * necklace_discount_price)

theorem total_sales_correct :
  calculate_total_sales 5 8 10 25 12 12 8 2 = 238 := by
  sorry

end total_sales_correct_l4081_408144


namespace unique_digit_satisfying_conditions_l4081_408126

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

/-- Constructs a number in the form 282,1A4 given a digit A -/
def constructNumber (A : Digit) : ℕ := 282100 + 10 * A.val + 4

/-- The main theorem: there exists exactly one digit A satisfying both conditions -/
theorem unique_digit_satisfying_conditions : 
  ∃! (A : Digit), isDivisibleBy 75 A.val ∧ isDivisibleBy (constructNumber A) 4 :=
sorry

end unique_digit_satisfying_conditions_l4081_408126


namespace l_plate_four_equal_parts_l4081_408153

/-- Represents an L-shaped plate -/
structure LPlate where
  width : ℝ
  height : ℝ
  isRightAngled : Bool

/-- Represents a cut on the L-shaped plate -/
inductive Cut
  | Vertical : ℝ → Cut  -- x-coordinate of the vertical cut
  | Horizontal : ℝ → Cut  -- y-coordinate of the horizontal cut

/-- Checks if a set of cuts divides an L-shaped plate into four equal parts -/
def dividesIntoFourEqualParts (plate : LPlate) (cuts : List Cut) : Prop :=
  sorry

/-- Theorem stating that an L-shaped plate can be divided into four equal L-shaped pieces -/
theorem l_plate_four_equal_parts (plate : LPlate) :
  ∃ (cuts : List Cut), dividesIntoFourEqualParts plate cuts :=
sorry

end l_plate_four_equal_parts_l4081_408153


namespace log_equation_solution_l4081_408146

theorem log_equation_solution :
  ∀ x : ℝ, (Real.log 729 / Real.log (3 * x) = x) →
    (x = 3 ∧ ¬ ∃ n : ℕ, x = n^2 ∧ ¬ ∃ m : ℕ, x = m^3 ∧ ∃ k : ℤ, x = k) := by
  sorry

end log_equation_solution_l4081_408146


namespace triangle_area_l4081_408137

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_area (abc : Triangle) (h1 : abc.c = 3) 
  (h2 : abc.a / Real.cos abc.A = abc.b / Real.cos abc.B)
  (h3 : Real.cos abc.C = 1/4) : 
  (1/2 * abc.a * abc.b * Real.sin abc.C) = (3 * Real.sqrt 15) / 4 := by
  sorry

#check triangle_area

end triangle_area_l4081_408137


namespace solution_set_of_inequality_l4081_408165

-- Define the function f
def f (x : ℝ) : ℝ := x * abs (x - 2)

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) : 
  f (2 - Real.log (x + 1)) > f 3 ↔ -1 < x ∧ x < Real.exp (-1) - 1 := by
  sorry

end solution_set_of_inequality_l4081_408165


namespace a_earnings_l4081_408127

-- Define the work rates and total wages
def a_rate : ℚ := 1 / 10
def b_rate : ℚ := 1 / 15
def total_wages : ℚ := 3400

-- Define A's share of the work when working together
def a_share : ℚ := a_rate / (a_rate + b_rate)

-- Theorem stating A's earnings
theorem a_earnings : a_share * total_wages = 2040 := by
  sorry

end a_earnings_l4081_408127


namespace min_a_for_quadratic_inequality_l4081_408108

theorem min_a_for_quadratic_inequality :
  (∀ x : ℝ, 0 < x → x ≤ 1/2 → x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
sorry

end min_a_for_quadratic_inequality_l4081_408108


namespace intersection_M_N_l4081_408142

def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 6*x + 5 < 0}

theorem intersection_M_N : M ∩ N = Set.Icc 2 5 := by sorry

end intersection_M_N_l4081_408142


namespace min_value_x_plus_2y_l4081_408113

theorem min_value_x_plus_2y (x y : ℝ) (h : x^2 + 4*y^2 - 2*x + 8*y + 1 = 0) :
  ∃ (m : ℝ), m = -2*Real.sqrt 2 - 1 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 - 2*a + 8*b + 1 = 0 → m ≤ a + 2*b :=
sorry

end min_value_x_plus_2y_l4081_408113


namespace third_shot_probability_at_least_one_hit_probability_l4081_408187

/-- A marksman shoots four times independently with a probability of hitting the target of 0.9 each time. -/
structure Marksman where
  shots : Fin 4 → ℝ
  prob_hit : ∀ i, shots i = 0.9
  independent : ∀ i j, i ≠ j → shots i = shots j

/-- The probability of hitting the target on the third shot is 0.9. -/
theorem third_shot_probability (m : Marksman) : m.shots 2 = 0.9 := by sorry

/-- The probability of hitting the target at least once is 1 - 0.1^4. -/
theorem at_least_one_hit_probability (m : Marksman) : 
  1 - (1 - m.shots 0) * (1 - m.shots 1) * (1 - m.shots 2) * (1 - m.shots 3) = 1 - 0.1^4 := by sorry

end third_shot_probability_at_least_one_hit_probability_l4081_408187


namespace arithmetic_sequence_sum_l4081_408129

/-- An arithmetic sequence with first term 0 and non-zero common difference -/
structure ArithmeticSequence where
  d : ℝ
  hd : d ≠ 0
  a : ℕ → ℝ
  h_init : a 1 = 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  ∃ m : ℕ, seq.a m = seq.a 1 + seq.a 2 + seq.a 3 + seq.a 4 + seq.a 5 → m = 11 := by
  sorry

end arithmetic_sequence_sum_l4081_408129


namespace smallest_n_square_and_cube_l4081_408131

theorem smallest_n_square_and_cube : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 4 * n = k ^ 2) ∧ 
    (∃ (m : ℕ), 3 * n = m ^ 3)) ∧
  (∀ (n : ℕ), n > 0 ∧ n < 144 → 
    ¬(∃ (k : ℕ), 4 * n = k ^ 2) ∨ 
    ¬(∃ (m : ℕ), 3 * n = m ^ 3)) :=
by sorry

end smallest_n_square_and_cube_l4081_408131


namespace height_of_congruent_triangles_l4081_408143

/-- Triangle congruence relation -/
def CongruentTriangles (t1 t2 : Type) : Prop := sorry

/-- Area of a triangle -/
def TriangleArea (t : Type) : ℝ := sorry

/-- Height of a triangle on a given side -/
def TriangleHeight (t : Type) (side : ℝ) : ℝ := sorry

/-- Side length of a triangle -/
def TriangleSide (t : Type) (side : String) : ℝ := sorry

theorem height_of_congruent_triangles 
  (ABC DEF : Type) 
  (h_cong : CongruentTriangles ABC DEF) 
  (h_side : TriangleSide ABC "AB" = TriangleSide DEF "DE" ∧ TriangleSide ABC "AB" = 4) 
  (h_area : TriangleArea DEF = 10) :
  TriangleHeight ABC (TriangleSide ABC "AB") = 5 := by
  sorry

end height_of_congruent_triangles_l4081_408143


namespace tan_value_for_given_point_l4081_408103

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_value_for_given_point (θ : Real) (h : ∃ (r : Real), r * (Real.cos θ) = -Real.sqrt 3 / 2 ∧ r * (Real.sin θ) = 1 / 2) : 
  Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end tan_value_for_given_point_l4081_408103


namespace sequence_general_term_l4081_408109

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℝ := 4 * n^2 + 2 * n

/-- The general term of the sequence -/
def a (n : ℕ) : ℝ := 8 * n - 2

/-- Theorem stating that the given general term formula is correct -/
theorem sequence_general_term (n : ℕ) : 
  S n - S (n - 1) = a n := by sorry

end sequence_general_term_l4081_408109


namespace range_of_x_in_triangle_l4081_408145

/-- Given a triangle ABC with vectors AB and AC, prove the range of x -/
theorem range_of_x_in_triangle (x : ℝ) : 
  let AB : ℝ × ℝ := (x, 2*x)
  let AC : ℝ × ℝ := (3*x, 2)
  -- Dot product is negative for obtuse angle
  (x * (3*x) + (2*x) * 2 < 0) →
  -- x is in the open interval (-4/3, 0)
  -4/3 < x ∧ x < 0 :=
by sorry


end range_of_x_in_triangle_l4081_408145


namespace sin_90_degrees_l4081_408157

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end sin_90_degrees_l4081_408157


namespace vector_decomposition_l4081_408175

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![8, 9, 4]
def p : Fin 3 → ℝ := ![1, 0, 1]
def q : Fin 3 → ℝ := ![0, -2, 1]
def r : Fin 3 → ℝ := ![1, 3, 0]

/-- Theorem: Vector x can be decomposed as 7p - 3q + r -/
theorem vector_decomposition :
  x = fun i => 7 * p i - 3 * q i + r i :=
by sorry

end vector_decomposition_l4081_408175


namespace gabby_makeup_set_savings_l4081_408130

/-- Proves that Gabby needs $10 more to buy the makeup set -/
theorem gabby_makeup_set_savings (makeup_cost initial_savings mom_contribution : ℕ) 
  (h1 : makeup_cost = 65)
  (h2 : initial_savings = 35)
  (h3 : mom_contribution = 20) :
  makeup_cost - initial_savings - mom_contribution = 10 := by
  sorry

end gabby_makeup_set_savings_l4081_408130


namespace complex_multiplication_l4081_408125

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i := by
  sorry

end complex_multiplication_l4081_408125


namespace problem_solution_l4081_408160

theorem problem_solution : 3^(0^(2^3)) + ((3^1)^0)^2 = 2 := by
  sorry

end problem_solution_l4081_408160


namespace child_height_at_last_visit_l4081_408150

/-- Given a child's current height and growth since last visit, 
    prove the height at the last visit. -/
theorem child_height_at_last_visit 
  (current_height : ℝ) 
  (growth_since_last_visit : ℝ) 
  (h1 : current_height = 41.5) 
  (h2 : growth_since_last_visit = 3.0) : 
  current_height - growth_since_last_visit = 38.5 := by
sorry

end child_height_at_last_visit_l4081_408150


namespace salt_solution_percentage_l4081_408177

theorem salt_solution_percentage (S : ℝ) : 
  S ≥ 0 ∧ S ≤ 100 →  -- Ensure S is a valid percentage
  (3/4 * S + 1/4 * 28 = 16) →  -- Equation representing the mixing of solutions
  S = 12 := by
sorry

end salt_solution_percentage_l4081_408177


namespace no_winning_strategy_card_game_probability_l4081_408161

/-- Represents a deck of cards with red and black suits. -/
structure Deck :=
  (red : ℕ)
  (black : ℕ)

/-- Represents a strategy for playing the card game. -/
def Strategy := Deck → Bool

/-- The probability of winning given a deck and a strategy. -/
def winProbability (d : Deck) (s : Strategy) : ℚ :=
  d.red / (d.red + d.black)

/-- The theorem stating that no strategy can have a winning probability greater than 0.5. -/
theorem no_winning_strategy (d : Deck) (s : Strategy) :
  d.red = d.black → winProbability d s = 1/2 := by
  sorry

/-- The main theorem stating that for any strategy, 
    the probability of winning is always 0.5 for a standard deck. -/
theorem card_game_probability (s : Strategy) : 
  ∀ d : Deck, d.red = d.black → d.red + d.black = 52 → winProbability d s = 1/2 := by
  sorry

end no_winning_strategy_card_game_probability_l4081_408161


namespace trigonometric_identity_l4081_408197

theorem trigonometric_identity : 
  (Real.cos (10 * π / 180)) / (Real.tan (20 * π / 180)) + 
  Real.sqrt 3 * (Real.sin (10 * π / 180)) * (Real.tan (70 * π / 180)) - 
  2 * (Real.cos (40 * π / 180)) = 2 := by sorry

end trigonometric_identity_l4081_408197


namespace min_value_expression_equality_condition_l4081_408141

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^4 + 18 * b^4 + 72 * c^4 + 1 / (27 * a * b * c) ≥ 4 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^4 + 18 * b^4 + 72 * c^4 + 1 / (27 * a * b * c) = 4 ↔
  a = ((9/4)^(1/4) * (1/(18 * ((9/4)^(1/4)) * (2^(1/4))))^(1/3)) ∧
  b = (2^(1/4) * (1/(18 * ((9/4)^(1/4)) * (2^(1/4))))^(1/3)) ∧
  c = (1/(18 * ((9/4)^(1/4)) * (2^(1/4))))^(1/3) :=
by sorry

end min_value_expression_equality_condition_l4081_408141


namespace inequalities_proof_l4081_408116

theorem inequalities_proof (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : a + b > 0) :
  (a^3 * b < a * b^3) ∧ (a / b + b / a < -2) := by
  sorry

end inequalities_proof_l4081_408116


namespace ruth_math_class_hours_l4081_408189

/-- Calculates the number of hours spent in math class per week for a student with given school schedule and math class percentage. -/
def math_class_hours_per_week (hours_per_day : ℕ) (days_per_week : ℕ) (math_class_percentage : ℚ) : ℚ :=
  (hours_per_day * days_per_week : ℚ) * math_class_percentage

/-- Theorem stating that a student who attends school for 8 hours a day, 5 days a week, and spends 25% of their school time in math class, spends 10 hours per week in math class. -/
theorem ruth_math_class_hours :
  math_class_hours_per_week 8 5 (1/4) = 10 := by
  sorry

end ruth_math_class_hours_l4081_408189


namespace min_value_expression_equality_condition_l4081_408159

theorem min_value_expression (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  a^2 + c^2 + 1/a^2 + c/a + 1/c^2 ≥ Real.sqrt 15 :=
sorry

theorem equality_condition (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  ∃ a c, a^2 + c^2 + 1/a^2 + c/a + 1/c^2 = Real.sqrt 15 :=
sorry

end min_value_expression_equality_condition_l4081_408159


namespace student_count_l4081_408169

theorem student_count (avg_age : ℝ) (teacher_age : ℝ) (new_avg : ℝ) :
  avg_age = 20 →
  teacher_age = 40 →
  new_avg = avg_age + 1 →
  (∃ n : ℕ, n * avg_age + teacher_age = (n + 1) * new_avg ∧ n = 19) :=
by sorry

end student_count_l4081_408169


namespace derivative_special_function_l4081_408117

open Real

/-- The derivative of (1 + 8 cosh² x * ln(cosh x)) / (2 cosh² x) -/
theorem derivative_special_function (x : ℝ) :
  deriv (λ x => (1 + 8 * (cosh x)^2 * log (cosh x)) / (2 * (cosh x)^2)) x
  = (sinh x * (4 * (cosh x)^2 - 1)) / (cosh x)^3 :=
by sorry

end derivative_special_function_l4081_408117


namespace f_zero_is_zero_l4081_408195

-- Define the function f
variable (f : ℝ → ℝ)

-- State the condition
axiom functional_equation : ∀ x y : ℝ, f (x + y) = f x + f y

-- Theorem to prove
theorem f_zero_is_zero : f 0 = 0 := by
  sorry

end f_zero_is_zero_l4081_408195


namespace least_four_digit_9_heavy_l4081_408102

def is_9_heavy (n : ℕ) : Prop := n % 9 = 6

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem least_four_digit_9_heavy : 
  (∀ m : ℕ, is_four_digit m → is_9_heavy m → 1005 ≤ m) ∧ 
  is_four_digit 1005 ∧ 
  is_9_heavy 1005 := by sorry

end least_four_digit_9_heavy_l4081_408102


namespace markup_markdown_equivalence_l4081_408162

theorem markup_markdown_equivalence (original_price : ℝ) (markup_percentage : ℝ) (markdown_percentage : ℝ)
  (h1 : markup_percentage = 25)
  (h2 : original_price * (1 + markup_percentage / 100) * (1 - markdown_percentage / 100) = original_price) :
  markdown_percentage = 20 := by
  sorry

end markup_markdown_equivalence_l4081_408162


namespace james_sticker_payment_ratio_l4081_408118

/-- Proves that the ratio of James's payment to the total cost of stickers is 1/2 -/
theorem james_sticker_payment_ratio :
  let num_packs : ℕ := 4
  let stickers_per_pack : ℕ := 30
  let cost_per_sticker : ℚ := 1/10
  let james_payment : ℚ := 6
  let total_stickers : ℕ := num_packs * stickers_per_pack
  let total_cost : ℚ := (total_stickers : ℚ) * cost_per_sticker
  james_payment / total_cost = 1/2 := by
sorry


end james_sticker_payment_ratio_l4081_408118


namespace pure_imaginary_fraction_l4081_408135

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (a - 2 * Complex.I) / (1 + 2 * Complex.I)) → a = 4 := by
  sorry

end pure_imaginary_fraction_l4081_408135


namespace hexagon_circumradius_theorem_l4081_408115

-- Define a hexagon as a set of 6 points in 2D space
def Hexagon : Type := Fin 6 → ℝ × ℝ

-- Define the property of being convex for a hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define the property that all sides of the hexagon have length 1
def all_sides_unit_length (h : Hexagon) : Prop := sorry

-- Define the circumradius of a triangle given by three points
def circumradius (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hexagon_circumradius_theorem (h : Hexagon) 
  (convex : is_convex h) 
  (unit_sides : all_sides_unit_length h) : 
  max (circumradius (h 0) (h 2) (h 4)) (circumradius (h 1) (h 3) (h 5)) ≥ 1 := by sorry

end hexagon_circumradius_theorem_l4081_408115


namespace jed_cards_per_week_l4081_408191

/-- Represents the number of cards Jed has after a given number of weeks -/
def cards_after_weeks (initial_cards : ℕ) (cards_per_week : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + cards_per_week * weeks - 2 * (weeks / 2)

/-- Proves that Jed gets 6 cards per week given the conditions -/
theorem jed_cards_per_week :
  ∃ (cards_per_week : ℕ),
    cards_after_weeks 20 cards_per_week 4 = 40 ∧ cards_per_week = 6 := by
  sorry

#check jed_cards_per_week

end jed_cards_per_week_l4081_408191


namespace cauchy_mean_value_theorem_sine_cosine_l4081_408111

open Real

theorem cauchy_mean_value_theorem_sine_cosine :
  ∃ c : ℝ, 0 < c ∧ c < π / 2 ∧
    (cos c) / (-sin c) = (sin (π / 2) - sin 0) / (cos (π / 2) - cos 0) := by
  sorry

end cauchy_mean_value_theorem_sine_cosine_l4081_408111


namespace repeating_decimal_equals_fraction_l4081_408148

/-- The common ratio of the geometric sequence representing 0.72̄ -/
def q : ℚ := 1 / 100

/-- The first term of the geometric sequence representing 0.72̄ -/
def a₁ : ℚ := 72 / 100

/-- The sum of the infinite geometric series representing 0.72̄ -/
def S : ℚ := a₁ / (1 - q)

/-- The repeating decimal 0.72̄ as a rational number -/
def repeating_decimal : ℚ := 8 / 11

theorem repeating_decimal_equals_fraction : S = repeating_decimal := by sorry

end repeating_decimal_equals_fraction_l4081_408148


namespace smallest_integer_linear_combination_l4081_408123

theorem smallest_integer_linear_combination (m n : ℤ) : 
  ∃ (k : ℕ), k > 0 ∧ (∃ (a b : ℤ), k = 5013 * a + 111111 * b) ∧
  ∀ (l : ℕ), l > 0 → (∃ (c d : ℤ), l = 5013 * c + 111111 * d) → k ≤ l :=
by sorry

end smallest_integer_linear_combination_l4081_408123


namespace cone_prism_volume_ratio_l4081_408167

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism -/
theorem cone_prism_volume_ratio (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  (1 / 3 * π * r^2 * h) / (6 * r^2 * h) = π / 18 := by
  sorry

#check cone_prism_volume_ratio

end cone_prism_volume_ratio_l4081_408167


namespace equation_solution_l4081_408104

theorem equation_solution : ∃ (z₁ z₂ : ℂ), 
  z₁ = -1 + Complex.I ∧ 
  z₂ = -1 - Complex.I ∧ 
  (∀ x : ℂ, x ≠ -2 → -x^3 = (4*x + 2)/(x + 2) ↔ (x = z₁ ∨ x = z₂)) := by
  sorry

end equation_solution_l4081_408104


namespace sqrt_x_minus_two_defined_l4081_408194

theorem sqrt_x_minus_two_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end sqrt_x_minus_two_defined_l4081_408194


namespace opposite_of_negative_2023_l4081_408183

theorem opposite_of_negative_2023 : -((-2023 : ℚ)) = (2023 : ℚ) := by
  sorry

end opposite_of_negative_2023_l4081_408183


namespace ball_probability_l4081_408174

/-- The probability of choosing a ball that is neither red nor purple from a bag -/
theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 10)
  (h_green : green = 30)
  (h_yellow : yellow = 10)
  (h_red : red = 47)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 1/2 :=
sorry

end ball_probability_l4081_408174


namespace arithmetic_sequence_common_ratio_l4081_408122

/-- For an arithmetic sequence {a_n} with sum S_n = 2n - 1, prove the common ratio is 2 -/
theorem arithmetic_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = 2 * n - 1) 
  (h2 : ∀ n, S n = n * a 1) : 
  (a 2) / (a 1) = 2 := by
sorry

end arithmetic_sequence_common_ratio_l4081_408122


namespace max_min_difference_z_l4081_408106

theorem max_min_difference_z (x y z : ℝ) 
  (sum_condition : x + y + z = 3) 
  (sum_squares_condition : x^2 + y^2 + z^2 = 18) : 
  ∃ (z_max z_min : ℝ), 
    (∀ w, w = x ∨ w = y ∨ w = z → z_min ≤ w ∧ w ≤ z_max) ∧ 
    z_max - z_min = 6 :=
sorry

end max_min_difference_z_l4081_408106


namespace base_7_23456_equals_6068_l4081_408133

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_23456_equals_6068 :
  base_7_to_10 [6, 5, 4, 3, 2] = 6068 := by
  sorry

end base_7_23456_equals_6068_l4081_408133


namespace tetrahedron_planes_intersection_l4081_408166

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  normal : Point3D
  point : Point3D

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- The circumcenter of a triangle -/
def circumcenter (a b c : Point3D) : Point3D := sorry

/-- The center of the circumsphere of a tetrahedron -/
def circumsphere_center (t : Tetrahedron) : Point3D := sorry

/-- A plane passing through a point and perpendicular to a line -/
def perpendicular_plane (point line_start line_end : Point3D) : Plane3D := sorry

/-- Check if a point lies on a plane -/
def point_on_plane (point : Point3D) (plane : Plane3D) : Prop := sorry

/-- Check if a tetrahedron is regular -/
def is_regular (t : Tetrahedron) : Prop := sorry

/-- The main theorem -/
theorem tetrahedron_planes_intersection
  (t : Tetrahedron)
  (A' : Point3D) (B' : Point3D) (C' : Point3D) (D' : Point3D)
  (h_A' : A' = circumcenter t.B t.C t.D)
  (h_B' : B' = circumcenter t.C t.D t.A)
  (h_C' : C' = circumcenter t.D t.A t.B)
  (h_D' : D' = circumcenter t.A t.B t.C)
  (P_A : Plane3D) (P_B : Plane3D) (P_C : Plane3D) (P_D : Plane3D)
  (h_P_A : P_A = perpendicular_plane t.A C' D')
  (h_P_B : P_B = perpendicular_plane t.B D' A')
  (h_P_C : P_C = perpendicular_plane t.C A' B')
  (h_P_D : P_D = perpendicular_plane t.D B' C')
  (P : Point3D)
  (h_P : P = circumsphere_center t) :
  ∃ (I : Point3D),
    point_on_plane I P_A ∧
    point_on_plane I P_B ∧
    point_on_plane I P_C ∧
    point_on_plane I P_D ∧
    (I = P ↔ is_regular t) := by
  sorry

end tetrahedron_planes_intersection_l4081_408166


namespace unique_solution_l4081_408181

/-- Represents a 3-digit number with distinct digits -/
structure ThreeDigitNumber where
  f : Nat
  o : Nat
  g : Nat
  h_distinct : f ≠ o ∧ f ≠ g ∧ o ≠ g
  h_valid : f ≠ 0 ∧ f < 10 ∧ o < 10 ∧ g < 10

def value (n : ThreeDigitNumber) : Nat :=
  100 * n.f + 10 * n.o + n.g

theorem unique_solution (n : ThreeDigitNumber) :
  value n * (n.f + n.o + n.g) = value n →
  n.f = 1 ∧ n.o = 0 ∧ n.g = 0 ∧ n.f + n.o + n.g = 1 := by
  sorry

end unique_solution_l4081_408181


namespace probability_test_l4081_408112

def probability_at_least_3_of_4 (p : ℝ) : ℝ :=
  (4 : ℝ) * p^3 * (1 - p) + p^4

theorem probability_test (p : ℝ) (hp : p = 4/5) :
  probability_at_least_3_of_4 p = 512/625 := by
  sorry

end probability_test_l4081_408112
