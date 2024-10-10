import Mathlib

namespace sum_of_arithmetic_sequence_l3480_348026

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a6 : a 6 = -2) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
by sorry

end sum_of_arithmetic_sequence_l3480_348026


namespace yellow_marbles_count_l3480_348054

theorem yellow_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 85 → 
  red = 14 → 
  blue = 3 * red → 
  yellow = total - (red + blue) → 
  yellow = 29 := by sorry

end yellow_marbles_count_l3480_348054


namespace f_derivative_at_zero_l3480_348022

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 1 - Real.cos (x * Real.sin (1 / x))
  else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by
  sorry

end f_derivative_at_zero_l3480_348022


namespace inscribed_circle_probability_l3480_348058

/-- Given a right-angled triangle with legs of 5 and 12 steps, 
    the probability that a randomly selected point within the triangle 
    lies within its inscribed circle is 2π/15 -/
theorem inscribed_circle_probability (a b : ℝ) (h1 : a = 5) (h2 : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let r := (a + b - c) / 2
  let triangle_area := a * b / 2
  let circle_area := π * r^2
  circle_area / triangle_area = 2 * π / 15 := by
sorry

end inscribed_circle_probability_l3480_348058


namespace tangent_parallel_points_l3480_348081

def f (x : ℝ) := x^3 + x - 2

theorem tangent_parallel_points :
  {P : ℝ × ℝ | P.1 ^ 3 + P.1 - 2 = P.2 ∧ (3 * P.1 ^ 2 + 1 = 4)} =
  {(-1, -4), (1, 0)} := by
sorry

end tangent_parallel_points_l3480_348081


namespace long_sleeved_jersey_cost_l3480_348014

/-- Represents the cost of jerseys and proves the cost of long-sleeved jerseys --/
theorem long_sleeved_jersey_cost 
  (long_sleeved_count : ℕ) 
  (striped_count : ℕ) 
  (striped_cost : ℕ) 
  (total_spent : ℕ) 
  (h1 : long_sleeved_count = 4)
  (h2 : striped_count = 2)
  (h3 : striped_cost = 10)
  (h4 : total_spent = 80) :
  ∃ (long_sleeved_cost : ℕ), 
    long_sleeved_count * long_sleeved_cost + striped_count * striped_cost = total_spent ∧ 
    long_sleeved_cost = 15 :=
by sorry

end long_sleeved_jersey_cost_l3480_348014


namespace candy_distribution_l3480_348001

theorem candy_distribution (A B : ℕ) 
  (h1 : 7 * A = B + 12) 
  (h2 : 3 * A = B - 20) : 
  A + B = 52 := by
sorry

end candy_distribution_l3480_348001


namespace range_of_m_l3480_348033

/-- Given propositions p and q, where ¬q is a sufficient but not necessary condition for ¬p,
    prove that the range of values for m is m ≥ 1. -/
theorem range_of_m (x m : ℝ) : 
  (∀ x, (x^2 + x - 2 > 0 ↔ x > 1 ∨ x < -2)) →
  (∀ x, (x ≤ m → x^2 + x - 2 ≤ 0) ∧ 
        ∃ y, (y^2 + y - 2 ≤ 0 ∧ y > m)) →
  m ≥ 1 :=
by sorry

end range_of_m_l3480_348033


namespace prob_three_heads_before_two_tails_l3480_348095

/-- The probability of getting a specific outcome when flipping a fair coin -/
def fair_coin_prob : ℚ := 1/2

/-- The state space for the coin flipping process -/
inductive CoinState
| H0  -- No heads or tails flipped yet
| H1  -- 1 consecutive head flipped
| H2  -- 2 consecutive heads flipped
| T1  -- 1 tail flipped
| HHH -- 3 consecutive heads (win state)
| TT  -- 2 consecutive tails (lose state)

/-- The probability of reaching the HHH state from a given state -/
noncomputable def prob_reach_HHH : CoinState → ℚ
| CoinState.H0 => sorry
| CoinState.H1 => sorry
| CoinState.H2 => sorry
| CoinState.T1 => sorry
| CoinState.HHH => 1
| CoinState.TT => 0

/-- The main theorem: probability of reaching HHH from the initial state is 3/8 -/
theorem prob_three_heads_before_two_tails : prob_reach_HHH CoinState.H0 = 3/8 := by sorry

end prob_three_heads_before_two_tails_l3480_348095


namespace community_service_selection_schemes_l3480_348027

theorem community_service_selection_schemes :
  let total_boys : ℕ := 4
  let total_girls : ℕ := 2
  let group_size : ℕ := 4
  let min_girls : ℕ := 1

  let selection_schemes : ℕ := 
    Nat.choose total_girls 1 * Nat.choose total_boys 3 +
    Nat.choose total_girls 2 * Nat.choose total_boys 2

  selection_schemes = 14 :=
by sorry

end community_service_selection_schemes_l3480_348027


namespace gcd_2025_2070_l3480_348088

theorem gcd_2025_2070 : Nat.gcd 2025 2070 = 45 := by
  sorry

end gcd_2025_2070_l3480_348088


namespace smallest_c_for_cosine_zero_l3480_348029

theorem smallest_c_for_cosine_zero (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (∀ x : ℝ, x < 0 → a * Real.cos (b * x + c) ≠ 0) →
  a * Real.cos c = 0 →
  c ≥ π / 2 :=
by sorry

end smallest_c_for_cosine_zero_l3480_348029


namespace largest_fraction_l3480_348051

theorem largest_fraction :
  (202 : ℚ) / 403 > 5 / 11 ∧
  (202 : ℚ) / 403 > 7 / 16 ∧
  (202 : ℚ) / 403 > 23 / 50 ∧
  (202 : ℚ) / 403 > 99 / 200 :=
by sorry

end largest_fraction_l3480_348051


namespace total_onions_l3480_348019

theorem total_onions (sara sally fred jack : ℕ) 
  (h1 : sara = 4) 
  (h2 : sally = 5) 
  (h3 : fred = 9) 
  (h4 : jack = 7) : 
  sara + sally + fred + jack = 25 := by
  sorry

end total_onions_l3480_348019


namespace sector_arc_length_l3480_348060

/-- Given a sector with a central angle of 60° and a radius of 3,
    the length of the arc is equal to π. -/
theorem sector_arc_length (θ : Real) (r : Real) : 
  θ = 60 * π / 180 → r = 3 → θ * r = π := by sorry

end sector_arc_length_l3480_348060


namespace largest_four_digit_sum_20_l3480_348044

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
by sorry

end largest_four_digit_sum_20_l3480_348044


namespace division_of_fractions_l3480_348053

theorem division_of_fractions : (4 - 1/4) / (2 - 1/2) = 5/2 := by
  sorry

end division_of_fractions_l3480_348053


namespace john_number_theorem_l3480_348021

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem john_number_theorem :
  ∃! x : ℕ, is_two_digit x ∧
    84 ≤ switch_digits (5 * x - 7) ∧
    switch_digits (5 * x - 7) ≤ 90 ∧
    x = 11 := by
  sorry

end john_number_theorem_l3480_348021


namespace intersection_empty_iff_union_equals_B_iff_l3480_348049

def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem intersection_empty_iff (a : ℝ) : A a ∩ B = ∅ ↔ a ≤ -4 ∨ a ≥ 5 := by
  sorry

theorem union_equals_B_iff (a : ℝ) : A a ∪ B = B ↔ a > 2 := by
  sorry

end intersection_empty_iff_union_equals_B_iff_l3480_348049


namespace distance_between_points_l3480_348085

def P1 : ℝ × ℝ := (-1, 1)
def P2 : ℝ × ℝ := (2, 5)

theorem distance_between_points : Real.sqrt ((P2.1 - P1.1)^2 + (P2.2 - P1.2)^2) = 5 := by
  sorry

end distance_between_points_l3480_348085


namespace convex_ngon_angle_theorem_l3480_348080

theorem convex_ngon_angle_theorem (n : ℕ) : 
  (n ≥ 3) →  -- n-gon must have at least 3 sides
  (∃ (x : ℝ), x > 0 ∧ x < 150 ∧ 150 * (n - 1) + x = 180 * (n - 2)) →
  (n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11) := by
sorry

end convex_ngon_angle_theorem_l3480_348080


namespace circus_ticket_price_l3480_348091

theorem circus_ticket_price :
  let total_tickets : ℕ := 522
  let child_ticket_price : ℚ := 8
  let total_receipts : ℚ := 5086
  let adult_tickets_sold : ℕ := 130
  let child_tickets_sold : ℕ := total_tickets - adult_tickets_sold
  let adult_ticket_price : ℚ := (total_receipts - child_ticket_price * child_tickets_sold) / adult_tickets_sold
  adult_ticket_price = 15 := by
sorry

end circus_ticket_price_l3480_348091


namespace union_condition_implies_a_range_l3480_348078

def A (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B := {x : ℝ | x < -1 ∨ x > 5}

theorem union_condition_implies_a_range (a : ℝ) :
  A a ∪ B = B → a < -4 ∨ a > 5 := by
  sorry

end union_condition_implies_a_range_l3480_348078


namespace square_value_l3480_348079

theorem square_value (x y z : ℝ) 
  (eq1 : 2*x + y + z = 17)
  (eq2 : x + 2*y + z = 14)
  (eq3 : x + y + 2*z = 13) :
  x = 6 := by
sorry

end square_value_l3480_348079


namespace sum_of_y_values_l3480_348017

/-- Given 5 experiments with x and y values, prove the sum of y values -/
theorem sum_of_y_values
  (x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ : ℝ)
  (sum_x : x₁ + x₂ + x₃ + x₄ + x₅ = 150)
  (regression_eq : ∀ x y, y = 0.67 * x + 54.9) :
  y₁ + y₂ + y₃ + y₄ + y₅ = 375 := by
  sorry

end sum_of_y_values_l3480_348017


namespace intersection_with_complement_l3480_348066

-- Define the universe set U
def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Finset ℕ := {2, 4, 6}

-- Define set B
def B : Finset ℕ := {1, 3, 5, 7}

-- Theorem statement
theorem intersection_with_complement :
  A ∩ (U \ B) = {2, 4, 6} := by sorry

end intersection_with_complement_l3480_348066


namespace intersection_of_A_and_B_l3480_348005

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x < -1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by
  sorry

end intersection_of_A_and_B_l3480_348005


namespace systematic_sampling_proof_l3480_348057

/-- Represents a systematic sampling sequence -/
def SystematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => start + i * (total / sampleSize))

/-- The problem statement -/
theorem systematic_sampling_proof (total : ℕ) (sampleSize : ℕ) (start : ℕ) :
  total = 60 →
  sampleSize = 6 →
  start = 3 →
  SystematicSample total sampleSize start = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end systematic_sampling_proof_l3480_348057


namespace right_triangle_perimeter_equals_sum_of_radii_l3480_348056

/-- For a right-angled triangle, the perimeter equals the sum of radii of inscribed and excircles -/
theorem right_triangle_perimeter_equals_sum_of_radii 
  (a b c ρ ρ_a ρ_b ρ_c : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Pythagorean theorem for right-angled triangle
  (h_ρ : ρ = (a + b - c) / 2)  -- Formula for inscribed circle radius
  (h_ρ_a : ρ_a = (a + b + c) / 2 - a)  -- Formula for excircle radius opposite to side a
  (h_ρ_b : ρ_b = (a + b + c) / 2 - b)  -- Formula for excircle radius opposite to side b
  (h_ρ_c : ρ_c = (a + b + c) / 2)  -- Formula for excircle radius opposite to side c
  : a + b + c = ρ + ρ_a + ρ_b + ρ_c := by
  sorry

end right_triangle_perimeter_equals_sum_of_radii_l3480_348056


namespace expression_evaluation_l3480_348059

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -2
  (3*x + 2*y) * (3*x - 2*y) - 5*x*(x - y) - (2*x - y)^2 = -14 := by
  sorry

end expression_evaluation_l3480_348059


namespace complex_product_magnitude_l3480_348096

theorem complex_product_magnitude : Complex.abs ((20 - 15 * Complex.I) * (12 + 25 * Complex.I)) = 25 * Real.sqrt 769 := by
  sorry

end complex_product_magnitude_l3480_348096


namespace floor_area_from_partial_coverage_l3480_348004

/-- The total area of a floor given a carpet covering a known percentage -/
theorem floor_area_from_partial_coverage (carpet_area : ℝ) (coverage_percentage : ℝ) 
  (h1 : carpet_area = 36) 
  (h2 : coverage_percentage = 0.45) : 
  carpet_area / coverage_percentage = 80 := by
  sorry

end floor_area_from_partial_coverage_l3480_348004


namespace vector_linear_combination_l3480_348050

/-- Given vectors a, b, and c in ℝ², prove that c is a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (1, -1)) 
  (hc : c = (-1, 2)) : 
  ∃ (k l : ℝ), c = k • a + l • b ∧ k = (1/2 : ℝ) ∧ l = (-3/2 : ℝ) := by
  sorry

end vector_linear_combination_l3480_348050


namespace q_is_false_l3480_348071

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
by sorry

end q_is_false_l3480_348071


namespace max_odd_digits_in_sum_l3480_348070

/-- A function that counts the number of odd digits in a natural number -/
def count_odd_digits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly 10 digits -/
def has_ten_digits (n : ℕ) : Prop := sorry

theorem max_odd_digits_in_sum (a b c : ℕ) 
  (ha : has_ten_digits a) 
  (hb : has_ten_digits b) 
  (hc : has_ten_digits c) 
  (sum_eq : a + b = c) : 
  count_odd_digits a + count_odd_digits b + count_odd_digits c ≤ 29 :=
sorry

end max_odd_digits_in_sum_l3480_348070


namespace students_with_no_books_l3480_348020

/-- Represents the number of students who borrowed a specific number of books -/
structure BookBorrowers where
  zero : ℕ
  one : ℕ
  two : ℕ
  threeOrMore : ℕ

/-- The total number of students in the class -/
def totalStudents : ℕ := 40

/-- The average number of books borrowed per student -/
def averageBooks : ℚ := 2

/-- Calculates the total number of books borrowed -/
def totalBooksBorrowed (b : BookBorrowers) : ℕ :=
  0 * b.zero + 1 * b.one + 2 * b.two + 3 * b.threeOrMore

/-- Theorem stating the number of students who did not borrow books -/
theorem students_with_no_books (b : BookBorrowers) : 
  b.zero = 1 ∧ 
  b.one = 12 ∧ 
  b.two = 13 ∧ 
  b.zero + b.one + b.two + b.threeOrMore = totalStudents ∧
  (totalBooksBorrowed b : ℚ) / totalStudents = averageBooks :=
by
  sorry


end students_with_no_books_l3480_348020


namespace intersection_S_T_l3480_348041

def S : Set ℝ := {x : ℝ | x^2 - 5*x - 6 < 0}
def T : Set ℝ := {x : ℝ | x + 2 ≤ 3}

theorem intersection_S_T : S ∩ T = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end intersection_S_T_l3480_348041


namespace problem_solution_l3480_348083

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 3) (h2 : y^2 / z = 4) (h3 : z^2 / x = 5) :
  x = (36 * Real.sqrt 5) ^ (4/11) := by
sorry

end problem_solution_l3480_348083


namespace gardener_work_theorem_l3480_348065

/-- Represents the outcome of the gardener's work. -/
structure GardenerOutcome where
  diligentDays : ℕ
  shirkingDays : ℕ

/-- Calculates the pretzel balance based on the gardener's work outcome. -/
def pretzelBalance (outcome : GardenerOutcome) : ℤ :=
  (3 * outcome.diligentDays) - outcome.shirkingDays

theorem gardener_work_theorem :
  ∃ (outcome : GardenerOutcome),
    outcome.diligentDays + outcome.shirkingDays = 26 ∧
    pretzelBalance outcome = 62 ∧
    outcome.diligentDays = 22 ∧
    outcome.shirkingDays = 4 := by
  sorry

#check gardener_work_theorem

end gardener_work_theorem_l3480_348065


namespace angle_CDE_is_right_angle_l3480_348028

theorem angle_CDE_is_right_angle 
  (angle_A angle_B angle_C : Real)
  (angle_AEB angle_BED angle_BDE : Real)
  (h1 : angle_A = 90)
  (h2 : angle_B = 90)
  (h3 : angle_C = 90)
  (h4 : angle_AEB = 50)
  (h5 : angle_BED = 40)
  (h6 : angle_BDE = 50)
  : ∃ (angle_CDE : Real), angle_CDE = 90 := by
  sorry

end angle_CDE_is_right_angle_l3480_348028


namespace base_equation_solution_l3480_348038

/-- Represents a number in a given base -/
def to_base (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Consecutive even positive integers -/
def consecutive_even (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ Even x ∧ Even y ∧ y = x + 2

theorem base_equation_solution (X Y : ℕ) :
  consecutive_even X Y →
  to_base 241 X + to_base 36 Y = to_base 94 (X + Y) →
  X + Y = 22 := by sorry

end base_equation_solution_l3480_348038


namespace quadratic_two_real_roots_condition_l3480_348068

theorem quadratic_two_real_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0) ↔ m ≤ 1 := by
  sorry

end quadratic_two_real_roots_condition_l3480_348068


namespace ripe_oranges_calculation_l3480_348045

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := 52

/-- The number of days of harvest -/
def harvest_days : ℕ := 26

/-- The total number of sacks of oranges after the harvest period -/
def total_oranges : ℕ := 2080

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges_per_day : ℕ := 28

theorem ripe_oranges_calculation :
  ripe_oranges_per_day * harvest_days + unripe_oranges_per_day * harvest_days = total_oranges :=
by sorry

end ripe_oranges_calculation_l3480_348045


namespace expression_simplification_and_evaluation_l3480_348067

theorem expression_simplification_and_evaluation :
  ∀ x y : ℝ,
  x - y = 5 →
  x + 2*y = 2 →
  (x^2 - 4*x*y + 4*y^2) / (x^2 - x*y) / (x + y - 3*y^2 / (x - y)) + 1/x = 1 :=
by
  sorry

end expression_simplification_and_evaluation_l3480_348067


namespace point_on_circle_l3480_348075

theorem point_on_circle (t : ℝ) :
  let x := (2 - t^2) / (2 + t^2)
  let y := 3*t / (2 + t^2)
  x^2 + y^2 = 1 := by sorry

end point_on_circle_l3480_348075


namespace polynomial_factorization_l3480_348034

theorem polynomial_factorization (a b : ℤ) : 
  (∀ x : ℝ, x^2 + a*x + b = (x+1)*(x-3)) → (a = -2 ∧ b = -3) := by
  sorry

end polynomial_factorization_l3480_348034


namespace airplane_passengers_l3480_348012

theorem airplane_passengers (total : ℕ) (men : ℕ) : 
  total = 170 → men = 90 → 2 * (total - men - (men / 2)) = men → total - men - (men / 2) = 35 := by
  sorry

end airplane_passengers_l3480_348012


namespace circle_radius_from_polar_equation_l3480_348015

/-- Given a circle with polar equation ρ² - 2ρcosθ + 4ρsinθ + 4 = 0, its radius is 1 -/
theorem circle_radius_from_polar_equation :
  ∀ ρ θ : ℝ,
  ρ^2 - 2*ρ*(Real.cos θ) + 4*ρ*(Real.sin θ) + 4 = 0 →
  ∃ x y : ℝ,
  (x - 1)^2 + (y + 2)^2 = 1 :=
by sorry

end circle_radius_from_polar_equation_l3480_348015


namespace range_of_k_l3480_348052

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the non-collinearity of e₁ and e₂
variable (h_non_collinear : ¬ ∃ (c : ℝ), e₁ = c • e₂)

-- Define vectors a and b
def a : V := 2 • e₁ + e₂
def b (k : ℝ) : V := k • e₁ + 3 • e₂

-- Define the condition that a and b form a basis
variable (h_basis : ∀ (k : ℝ), k ≠ 6 → LinearIndependent ℝ ![a, b k])

-- Theorem statement
theorem range_of_k : 
  {k : ℝ | k ≠ 6} = {k : ℝ | LinearIndependent ℝ ![a, b k]} :=
sorry

end range_of_k_l3480_348052


namespace double_quarter_four_percent_l3480_348087

theorem double_quarter_four_percent : (2 * (1/4 * (4/100))) = 0.02 := by
  sorry

end double_quarter_four_percent_l3480_348087


namespace equilateral_triangle_area_perimeter_ratio_l3480_348093

/-- The ratio of area to perimeter for an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 6
  let area : ℝ := s^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * s
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end equilateral_triangle_area_perimeter_ratio_l3480_348093


namespace peach_difference_l3480_348031

/-- Given a basket of peaches with specified quantities of red, yellow, and green peaches,
    prove that there are 8 more green peaches than yellow peaches. -/
theorem peach_difference (red yellow green : ℕ) 
    (h_red : red = 2)
    (h_yellow : yellow = 6)
    (h_green : green = 14) :
  green - yellow = 8 := by
  sorry

end peach_difference_l3480_348031


namespace ab_equals_twelve_l3480_348069

-- Define the set A
def A (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

-- Define the complement of A
def complement_A : Set ℝ := {x | x < 3 ∨ x > 4}

-- Theorem statement
theorem ab_equals_twelve (a b : ℝ) : 
  A a b ∪ complement_A = Set.univ → a * b = 12 := by
  sorry

end ab_equals_twelve_l3480_348069


namespace tub_drain_time_l3480_348094

/-- Represents the time it takes to drain a tub -/
def drainTime (initialFraction : ℚ) (drainedFraction : ℚ) (initialTime : ℚ) : ℚ :=
  (drainedFraction * initialTime) / initialFraction

theorem tub_drain_time :
  let initialFraction : ℚ := 5 / 7
  let remainingFraction : ℚ := 1 - initialFraction
  let initialTime : ℚ := 4
  drainTime initialFraction remainingFraction initialTime = 8 / 5 := by
  sorry

end tub_drain_time_l3480_348094


namespace singer_work_hours_l3480_348008

-- Define the number of songs
def num_songs : ℕ := 3

-- Define the number of days per song
def days_per_song : ℕ := 10

-- Define the total number of hours worked
def total_hours : ℕ := 300

-- Define the function to calculate hours per day
def hours_per_day (n s d t : ℕ) : ℚ :=
  t / (n * d)

-- Theorem statement
theorem singer_work_hours :
  hours_per_day num_songs days_per_song total_hours = 10 := by
  sorry

end singer_work_hours_l3480_348008


namespace book_cost_price_l3480_348016

theorem book_cost_price (cost : ℝ) : 
  (cost * 1.18 - cost * 1.12 = 18) → cost = 300 := by
  sorry

end book_cost_price_l3480_348016


namespace white_dandelions_on_saturday_l3480_348072

/-- Represents the state of a dandelion -/
inductive DandelionState
  | Yellow
  | White
  | Dispersed

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the lifecycle of a dandelion -/
def dandelionLifecycle (openDay : Day) (currentDay : Day) : DandelionState :=
  sorry

/-- Counts the number of dandelions in a specific state on a given day -/
def countDandelions (day : Day) (state : DandelionState) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem white_dandelions_on_saturday :
  (countDandelions Day.Monday DandelionState.Yellow = 20) →
  (countDandelions Day.Monday DandelionState.White = 14) →
  (countDandelions Day.Wednesday DandelionState.Yellow = 15) →
  (countDandelions Day.Wednesday DandelionState.White = 11) →
  (countDandelions Day.Saturday DandelionState.White = 6) :=
by sorry

end white_dandelions_on_saturday_l3480_348072


namespace propositions_true_l3480_348099

-- Define the propositions
def proposition1 (x y : ℝ) : Prop := x + y = 0 → (x = -y ∨ y = -x)
def proposition3 (q : ℝ) : Prop := q ≤ 1 → ∃ x : ℝ, x^2 + 2*x + q = 0

-- Theorem statement
theorem propositions_true :
  (∀ x y : ℝ, ¬(x + y = 0) → ¬(x = -y ∨ y = -x)) ∧
  (∀ q : ℝ, (¬∃ x : ℝ, x^2 + 2*x + q = 0) → ¬(q ≤ 1)) := by sorry

end propositions_true_l3480_348099


namespace abs_neg_five_plus_three_l3480_348025

theorem abs_neg_five_plus_three : |(-5 + 3)| = 2 := by
  sorry

end abs_neg_five_plus_three_l3480_348025


namespace functional_equation_identity_l3480_348002

open Function Real

theorem functional_equation_identity (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f y + y) = f (f (x * y)) + y) →
  (∀ y : ℝ, f y = y) :=
by sorry

end functional_equation_identity_l3480_348002


namespace elevator_weight_problem_l3480_348092

/-- Given an elevator with 6 people and an average weight of 152 lbs, 
    prove that when a new person weighing 145 lbs enters, 
    the new average weight of all 7 people is 151 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℚ) 
  (new_person_weight : ℚ) (new_avg_weight : ℚ) :
  initial_people = 6 →
  initial_avg_weight = 152 →
  new_person_weight = 145 →
  new_avg_weight = (initial_people * initial_avg_weight + new_person_weight) / (initial_people + 1) →
  new_avg_weight = 151 :=
by sorry

end elevator_weight_problem_l3480_348092


namespace sample_is_reading_time_data_l3480_348040

/-- Represents a resident in the study -/
structure Resident where
  id : Nat
  readingTime : ℝ

/-- Represents the statistical study -/
structure ReadingStudy where
  population : Finset Resident
  sampleSize : Nat
  sample : Finset Resident

/-- Definition of a valid sample in the reading study -/
def validSample (study : ReadingStudy) : Prop :=
  study.sample.card = study.sampleSize ∧
  study.sample ⊆ study.population

/-- The main theorem about the sample definition -/
theorem sample_is_reading_time_data (study : ReadingStudy)
    (h_pop_size : study.population.card = 5000)
    (h_sample_size : study.sampleSize = 200)
    (h_valid_sample : validSample study) :
    ∃ (sample_data : Finset ℝ),
      sample_data = study.sample.image Resident.readingTime ∧
      sample_data.card = study.sampleSize :=
  sorry


end sample_is_reading_time_data_l3480_348040


namespace well_digging_rate_l3480_348023

/-- The hourly rate paid to workers for digging a well --/
def hourly_rate (total_payment : ℚ) (num_workers : ℕ) (hours_day1 hours_day2 hours_day3 : ℕ) : ℚ :=
  total_payment / (num_workers * (hours_day1 + hours_day2 + hours_day3))

/-- Theorem stating that under the given conditions, the hourly rate is $10 --/
theorem well_digging_rate : 
  hourly_rate 660 2 10 8 15 = 10 := by
  sorry


end well_digging_rate_l3480_348023


namespace william_riding_time_l3480_348063

theorem william_riding_time :
  let max_daily_time : ℝ := 6
  let total_days : ℕ := 6
  let max_time_days : ℕ := 2
  let min_time_days : ℕ := 2
  let half_time_days : ℕ := 2
  let min_daily_time : ℝ := 1.5

  max_time_days * max_daily_time +
  min_time_days * min_daily_time +
  half_time_days * (max_daily_time / 2) = 21 := by
  sorry

end william_riding_time_l3480_348063


namespace f_monotone_increasing_l3480_348018

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem f_monotone_increasing (x : ℝ) (h : x > 0) :
  Monotone (fun y ↦ f y) ↔ x > 1 := by
  sorry

end f_monotone_increasing_l3480_348018


namespace charlie_ate_fifteen_cookies_l3480_348064

/-- The number of cookies eaten by Charlie's family -/
def total_cookies : ℕ := 30

/-- The number of cookies eaten by Charlie's father -/
def father_cookies : ℕ := 10

/-- The number of cookies eaten by Charlie's mother -/
def mother_cookies : ℕ := 5

/-- Charlie's cookies -/
def charlie_cookies : ℕ := total_cookies - (father_cookies + mother_cookies)

theorem charlie_ate_fifteen_cookies : charlie_cookies = 15 := by
  sorry

end charlie_ate_fifteen_cookies_l3480_348064


namespace jury_duty_duration_l3480_348003

/-- Calculates the total number of days spent on jury duty -/
def total_jury_duty_days (jury_selection_days : ℕ) (trial_duration_factor : ℕ) 
  (deliberation_full_days : ℕ) (daily_deliberation_hours : ℕ) : ℕ :=
  let trial_days := jury_selection_days * trial_duration_factor
  let deliberation_hours := deliberation_full_days * 24
  let deliberation_days := deliberation_hours / daily_deliberation_hours
  jury_selection_days + trial_days + deliberation_days

/-- Theorem stating that the total number of days spent on jury duty is 19 -/
theorem jury_duty_duration : 
  total_jury_duty_days 2 4 6 16 = 19 := by
  sorry

end jury_duty_duration_l3480_348003


namespace integer_solutions_cubic_equation_l3480_348035

theorem integer_solutions_cubic_equation :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    (x = -7 ∧ y = -99) ∨ (x = -1 ∧ y = -9) ∨ (x = 1 ∧ y = 5) ∨ (x = 7 ∧ y = -97) :=
by sorry

end integer_solutions_cubic_equation_l3480_348035


namespace veridux_male_associates_l3480_348086

/-- Proves the number of male associates at Veridux Corporation --/
theorem veridux_male_associates :
  let total_employees : ℕ := 250
  let female_employees : ℕ := 90
  let total_managers : ℕ := 40
  let female_managers : ℕ := 40
  let male_employees : ℕ := total_employees - female_employees
  let male_associates : ℕ := male_employees
  male_associates = 160 := by
  sorry

end veridux_male_associates_l3480_348086


namespace uncovered_area_of_squares_l3480_348077

theorem uncovered_area_of_squares (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 10 →
  small_square_side = 4 →
  (large_square_side ^ 2) - 2 * (small_square_side ^ 2) = 68 := by
  sorry

end uncovered_area_of_squares_l3480_348077


namespace smallest_group_size_exists_smallest_group_size_l3480_348030

theorem smallest_group_size (n : ℕ) : n > 0 ∧ n % 5 = 0 ∧ n % 13 = 0 → n ≥ 65 := by
  sorry

theorem exists_smallest_group_size : ∃ n : ℕ, n > 0 ∧ n % 5 = 0 ∧ n % 13 = 0 ∧ n = 65 := by
  sorry

end smallest_group_size_exists_smallest_group_size_l3480_348030


namespace inscribed_cube_volume_in_specific_pyramid_l3480_348043

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  lateral_face_is_equilateral : Bool

/-- A cube inscribed in a pyramid -/
structure InscribedCube where
  pyramid : Pyramid
  bottom_face_on_base : Bool
  top_face_edges_on_lateral_faces : Bool

/-- The volume of an inscribed cube -/
noncomputable def inscribed_cube_volume (cube : InscribedCube) : ℝ :=
  sorry

theorem inscribed_cube_volume_in_specific_pyramid :
  ∀ (cube : InscribedCube),
    cube.pyramid.base_side = 2 ∧
    cube.pyramid.lateral_face_is_equilateral = true ∧
    cube.bottom_face_on_base = true ∧
    cube.top_face_edges_on_lateral_faces = true →
    inscribed_cube_volume cube = 2 * Real.sqrt 6 / 9 :=
by sorry

end inscribed_cube_volume_in_specific_pyramid_l3480_348043


namespace flowers_per_set_l3480_348010

theorem flowers_per_set (total_flowers : ℕ) (num_sets : ℕ) (h1 : total_flowers = 270) (h2 : num_sets = 3) :
  total_flowers / num_sets = 90 := by
  sorry

end flowers_per_set_l3480_348010


namespace mustard_at_second_table_l3480_348000

/-- The amount of mustard found at each table and the total amount --/
def MustardProblem (total first second third : ℚ) : Prop :=
  total = first + second + third

theorem mustard_at_second_table :
  ∃ (second : ℚ), MustardProblem 0.88 0.25 second 0.38 ∧ second = 0.25 := by
  sorry

end mustard_at_second_table_l3480_348000


namespace sandbox_length_l3480_348089

theorem sandbox_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 146 → area = 45552 → length * width = area → length = 312 := by
  sorry

end sandbox_length_l3480_348089


namespace fifth_term_smallest_l3480_348013

/-- The sequence term for a given n -/
def sequence_term (n : ℕ) : ℤ := 3 * n^2 - 28 * n

/-- The 5th term is the smallest in the sequence -/
theorem fifth_term_smallest : ∀ k : ℕ, sequence_term 5 ≤ sequence_term k := by
  sorry

end fifth_term_smallest_l3480_348013


namespace eliana_steps_l3480_348098

/-- The total number of steps Eliana walked during three days -/
def total_steps (first_day_initial : ℕ) (first_day_additional : ℕ) (third_day_additional : ℕ) : ℕ :=
  let first_day := first_day_initial + first_day_additional
  let second_day := 2 * first_day
  let third_day := second_day + third_day_additional
  first_day + second_day + third_day

/-- Theorem stating the total number of steps Eliana walked during three days -/
theorem eliana_steps : 
  total_steps 200 300 100 = 2600 := by
  sorry

end eliana_steps_l3480_348098


namespace plane_equation_proof_l3480_348024

/-- A plane equation is represented by a tuple of integers (A, B, C, D) corresponding to the equation Ax + By + Cz + D = 0 --/
def PlaneEquation := (ℤ × ℤ × ℤ × ℤ)

/-- The given plane equation 3x - 2y + 4z = 10 --/
def given_plane : PlaneEquation := (3, -2, 4, -10)

/-- The point through which the new plane must pass --/
def point : (ℤ × ℤ × ℤ) := (2, -3, 5)

/-- Check if a plane equation passes through a given point --/
def passes_through (plane : PlaneEquation) (p : ℤ × ℤ × ℤ) : Prop :=
  let (A, B, C, D) := plane
  let (x, y, z) := p
  A * x + B * y + C * z + D = 0

/-- Check if two plane equations are parallel --/
def is_parallel (plane1 plane2 : PlaneEquation) : Prop :=
  let (A1, B1, C1, _) := plane1
  let (A2, B2, C2, _) := plane2
  ∃ (k : ℚ), k ≠ 0 ∧ A1 = k * A2 ∧ B1 = k * B2 ∧ C1 = k * C2

/-- Check if the first coefficient of a plane equation is positive --/
def first_coeff_positive (plane : PlaneEquation) : Prop :=
  let (A, _, _, _) := plane
  A > 0

/-- Calculate the greatest common divisor of the absolute values of all coefficients --/
def gcd_of_coeffs (plane : PlaneEquation) : ℕ :=
  let (A, B, C, D) := plane
  Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D)))

theorem plane_equation_proof (solution : PlaneEquation) : 
  passes_through solution point ∧ 
  is_parallel solution given_plane ∧ 
  first_coeff_positive solution ∧ 
  gcd_of_coeffs solution = 1 ∧ 
  solution = (3, -2, 4, -32) := by
  sorry

end plane_equation_proof_l3480_348024


namespace triangle_angle_measure_l3480_348037

theorem triangle_angle_measure (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : C = 3 * B) (h3 : B = 15) : A = 120 :=
sorry

end triangle_angle_measure_l3480_348037


namespace light_2011_is_green_l3480_348032

def light_pattern : ℕ → String
  | 0 => "green"
  | 1 => "yellow"
  | 2 => "yellow"
  | 3 => "red"
  | 4 => "red"
  | 5 => "red"
  | n + 6 => light_pattern n

theorem light_2011_is_green : light_pattern 2010 = "green" := by
  sorry

end light_2011_is_green_l3480_348032


namespace compare_a_b_fraction_inequality_l3480_348039

-- Problem 1
theorem compare_a_b (m n : ℝ) :
  (m^2 + 1) * (n^2 + 4) ≥ (m * n + 2)^2 := by sorry

-- Problem 2
theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) (h5 : e > 0) :
  e / (a - c) < e / (b - d) := by sorry

end compare_a_b_fraction_inequality_l3480_348039


namespace percentage_problem_l3480_348036

theorem percentage_problem (p : ℝ) : (p / 100) * 40 = 140 → p = 350 := by
  sorry

end percentage_problem_l3480_348036


namespace sequence_congruence_l3480_348074

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ → ℤ
  | 0, _ => 0
  | 1, _ => 1
  | (n + 2), k => 2 * k * a (n + 1) k - (k^2 + 1) * a n k

/-- Main theorem -/
theorem sequence_congruence (k : ℤ) (p : ℕ) (hp : Nat.Prime p) (hp_mod : p % 4 = 3) :
  (∀ n : ℕ, a (n + p^2 - 1) k ≡ a n k [ZMOD p]) ∧
  (∀ n : ℕ, a (n + p^3 - p) k ≡ a n k [ZMOD p^2]) := by
  sorry

#check sequence_congruence

end sequence_congruence_l3480_348074


namespace y_derivative_l3480_348055

noncomputable def y (x : ℝ) : ℝ := 4 * Real.arcsin (4 / (2 * x + 3)) + Real.sqrt (4 * x^2 + 12 * x - 7)

theorem y_derivative (x : ℝ) (h : 2 * x + 3 > 0) :
  deriv y x = (2 * Real.sqrt (4 * x^2 + 12 * x - 7)) / (2 * x + 3) :=
by sorry

end y_derivative_l3480_348055


namespace hot_dog_consumption_l3480_348062

theorem hot_dog_consumption (x : ℕ) : 
  x + (x + 2) + (x + 4) = 36 → x = 10 := by
  sorry

end hot_dog_consumption_l3480_348062


namespace set_inclusion_iff_range_l3480_348047

/-- Given sets A and B, prove that (ℝ \ B) ⊆ A if and only if a ≤ -2 or 1/2 ≤ a < 1 -/
theorem set_inclusion_iff_range (a : ℝ) : 
  let A : Set ℝ := {x | x < -1 ∨ x ≥ 1}
  let B : Set ℝ := {x | x ≤ 2*a ∨ x ≥ a+1}
  (Set.univ \ B) ⊆ A ↔ a ≤ -2 ∨ (1/2 ≤ a ∧ a < 1) :=
by sorry

end set_inclusion_iff_range_l3480_348047


namespace smallest_age_difference_l3480_348073

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : 0 ≤ tens ∧ tens ≤ 9 ∧ 0 ≤ units ∧ units ≤ 9

/-- Calculates the value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- Reverses the digits of a two-digit number -/
def TwoDigitNumber.reverse (n : TwoDigitNumber) : TwoDigitNumber where
  tens := n.units
  units := n.tens
  is_valid := by
    simp [n.is_valid]

/-- The difference between two natural numbers -/
def diff (a b : Nat) : Nat :=
  if a ≥ b then a - b else b - a

theorem smallest_age_difference :
  ∀ (mrs_age : TwoDigitNumber),
    diff (TwoDigitNumber.value mrs_age) (TwoDigitNumber.value (TwoDigitNumber.reverse mrs_age)) ≥ 9 ∧
    ∃ (age : TwoDigitNumber),
      diff (TwoDigitNumber.value age) (TwoDigitNumber.value (TwoDigitNumber.reverse age)) = 9 :=
by sorry

end smallest_age_difference_l3480_348073


namespace select_five_from_eight_l3480_348042

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end select_five_from_eight_l3480_348042


namespace prime_sequence_recurrence_relation_l3480_348011

theorem prime_sequence_recurrence_relation 
  (p : ℕ → ℕ) 
  (k : ℤ) 
  (h_prime : ∀ n, Nat.Prime (p n)) 
  (h_recurrence : ∀ n, p (n + 2) = p (n + 1) + p n + k) : 
  (∃ (prime : ℕ) (h_prime : Nat.Prime prime), 
    (∀ n, p n = prime) ∧ k = -prime) := by
  sorry

end prime_sequence_recurrence_relation_l3480_348011


namespace triangle_properties_l3480_348097

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.c * Real.cos t.B = (2 * t.a - t.b) * Real.cos t.C ∧ t.c = 4

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.C = π / 3 ∧
  (∀ s : ℝ, s = 1/2 * t.a * t.b * Real.sin t.C → s ≤ 4 * Real.sqrt 3) :=
sorry

end triangle_properties_l3480_348097


namespace system_solution_ratio_l3480_348009

theorem system_solution_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (eq1 : 8 * x - 6 * y = c) (eq2 : 10 * y - 15 * x = d) : c / d = -4 / 5 := by
  sorry

end system_solution_ratio_l3480_348009


namespace largest_four_digit_divisible_by_6_l3480_348006

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem largest_four_digit_divisible_by_6 :
  ∀ n : ℕ, is_four_digit n → divisible_by_6 n → n ≤ 9996 :=
by sorry

end largest_four_digit_divisible_by_6_l3480_348006


namespace fraction_addition_l3480_348007

theorem fraction_addition : (3/4) / (5/8) + 1/2 = 17/10 := by
  sorry

end fraction_addition_l3480_348007


namespace at_least_one_geq_two_l3480_348082

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_geq_two_l3480_348082


namespace percentage_problem_l3480_348076

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 9) : x = 30 := by
  sorry

end percentage_problem_l3480_348076


namespace vectors_opposite_direction_l3480_348090

def a : ℝ × ℝ := (-2, 4)
def b : ℝ × ℝ := (1, -2)

theorem vectors_opposite_direction :
  ∃ k : ℝ, k < 0 ∧ a = (k • b) := by sorry

end vectors_opposite_direction_l3480_348090


namespace expression_equality_l3480_348048

theorem expression_equality : 49^5 - 5 * 49^4 + 10 * 49^3 - 10 * 49^2 + 5 * 49 - 1 = 254804368 := by
  sorry

end expression_equality_l3480_348048


namespace hyperbola_focus_l3480_348084

/-- Given a hyperbola with equation (x-1)^2/7^2 - (y+8)^2/3^2 = 1,
    the coordinates of the focus with the smaller x-coordinate are (1 - √58, -8) -/
theorem hyperbola_focus (x y : ℝ) :
  (x - 1)^2 / 7^2 - (y + 8)^2 / 3^2 = 1 →
  ∃ (focus_x focus_y : ℝ),
    focus_x = 1 - Real.sqrt 58 ∧
    focus_y = -8 ∧
    ∀ (other_focus_x : ℝ),
      ((other_focus_x - 1)^2 / 7^2 - (focus_y + 8)^2 / 3^2 = 1 →
       other_focus_x ≥ focus_x) :=
by sorry

end hyperbola_focus_l3480_348084


namespace expression_simplification_l3480_348046

variable (a b : ℝ)

theorem expression_simplification :
  (2*a^2 - 3*a^3 + 5*a + 2*a^3 - a^2 = a^2 - a^3 + 5*a) ∧
  (2/3*(2*a - b) + 2*(b - 2*a) - 3*(2*a - b) - 4/3*(b - 2*a) = -6*a + 3*b) := by
  sorry

end expression_simplification_l3480_348046


namespace fibonacci_problem_l3480_348061

theorem fibonacci_problem (x : ℕ) (h : x > 0) :
  (10 : ℝ) / x = 40 / (x + 6) →
  ∃ (y : ℕ), y > 0 ∧
    (10 : ℝ) / x = 10 / y ∧
    40 / (x + 6) = 40 / (y + 6) ∧
    (10 : ℝ) / y = 40 / (y + 6) :=
by sorry

end fibonacci_problem_l3480_348061
