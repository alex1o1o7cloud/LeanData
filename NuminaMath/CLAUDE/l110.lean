import Mathlib

namespace election_percentage_l110_11029

/-- Given an election with 700 total votes where the winning candidate has a majority of 476 votes,
    prove that the winning candidate received 84% of the votes. -/
theorem election_percentage (total_votes : ℕ) (winning_majority : ℕ) (winning_percentage : ℚ) :
  total_votes = 700 →
  winning_majority = 476 →
  winning_percentage = 84 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = winning_majority :=
by sorry

end election_percentage_l110_11029


namespace grace_age_l110_11002

-- Define the ages as natural numbers
def Harriet : ℕ := 18
def Ian : ℕ := Harriet + 5
def Jack : ℕ := Ian - 7
def Grace : ℕ := 2 * Jack

-- Theorem statement
theorem grace_age : Grace = 32 := by
  sorry

end grace_age_l110_11002


namespace sum_of_cubic_equations_l110_11027

theorem sum_of_cubic_equations (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1)
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) :
  x + y = 2 := by sorry

end sum_of_cubic_equations_l110_11027


namespace smallest_n_squared_l110_11068

theorem smallest_n_squared (n : ℕ+) : 
  (∃ x y z : ℕ+, n.val^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) ↔ 
  n.val ≥ 2 := by
sorry

end smallest_n_squared_l110_11068


namespace arithmetic_sequence_common_difference_range_l110_11040

-- Define the arithmetic sequence
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

-- State the theorem
theorem arithmetic_sequence_common_difference_range :
  ∀ d : ℝ,
  (∀ n : ℕ, n < 6 → arithmeticSequence (-15) d n ≤ 0) ∧
  (∀ n : ℕ, n ≥ 6 → arithmeticSequence (-15) d n > 0) →
  3 < d ∧ d ≤ 15/4 := by
  sorry


end arithmetic_sequence_common_difference_range_l110_11040


namespace inequality_solution_1_inequality_solution_2_l110_11053

-- Define the solution set for the first inequality
def solution_set_1 (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Define the solution set for the second inequality
def solution_set_2 (x : ℝ) : Prop := x - 1/2 < x ∧ x < 1/3

-- Theorem for the first part
theorem inequality_solution_1 : 
  ∀ x : ℝ, (1/x > 1) ↔ solution_set_1 x := by sorry

-- Theorem for the second part
theorem inequality_solution_2 (a b : ℝ) : 
  (∀ x : ℝ, solution_set_2 x ↔ a^2 + b + 2 > 0) → a + b = 10 := by sorry

end inequality_solution_1_inequality_solution_2_l110_11053


namespace bill_value_l110_11057

theorem bill_value (total_money : ℕ) (num_bills : ℕ) (h1 : total_money = 45) (h2 : num_bills = 9) :
  total_money / num_bills = 5 := by
  sorry

end bill_value_l110_11057


namespace smallest_value_sum_of_fractions_lower_bound_achievable_l110_11095

theorem smallest_value_sum_of_fractions (a b : ℤ) (h : a > b) :
  (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) ≥ 2 :=
sorry

theorem lower_bound_achievable :
  ∃ (a b : ℤ), a > b ∧ (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) = 2 :=
sorry

end smallest_value_sum_of_fractions_lower_bound_achievable_l110_11095


namespace circus_ticket_price_l110_11042

/-- Proves that the price of an upper seat ticket is $20 given the conditions of the circus ticket sales. -/
theorem circus_ticket_price :
  let lower_seat_price : ℕ := 30
  let total_tickets : ℕ := 80
  let total_revenue : ℕ := 2100
  let lower_seats_sold : ℕ := 50
  let upper_seats_sold : ℕ := total_tickets - lower_seats_sold
  let upper_seat_price : ℕ := (total_revenue - lower_seat_price * lower_seats_sold) / upper_seats_sold
  upper_seat_price = 20 := by sorry

end circus_ticket_price_l110_11042


namespace square_side_equals_circle_circumference_divided_by_four_l110_11094

theorem square_side_equals_circle_circumference_divided_by_four (π : ℝ) (h : π = Real.pi) :
  let r : ℝ := 3
  let c : ℝ := 2 * π * r
  let y : ℝ := c / 4
  y = 3 * π / 2 := by
sorry

end square_side_equals_circle_circumference_divided_by_four_l110_11094


namespace unique_score_170_l110_11041

/-- Represents a test score --/
structure TestScore where
  total_questions : ℕ
  correct : ℕ
  wrong : ℕ
  score : ℤ

/-- Calculates the score based on the number of correct and wrong answers --/
def calculate_score (ts : TestScore) : ℤ :=
  30 + 4 * ts.correct - ts.wrong

/-- Checks if a TestScore is valid according to the rules --/
def is_valid_score (ts : TestScore) : Prop :=
  ts.correct + ts.wrong ≤ ts.total_questions ∧
  ts.score = calculate_score ts ∧
  ts.score > 90

/-- Theorem stating that 170 is the only score above 90 that uniquely determines the number of correct answers --/
theorem unique_score_170 :
  ∀ (ts : TestScore),
    ts.total_questions = 35 →
    is_valid_score ts →
    (∀ (ts' : TestScore),
      ts'.total_questions = 35 →
      is_valid_score ts' →
      ts'.score = ts.score →
      ts'.correct = ts.correct) →
    ts.score = 170 :=
sorry

end unique_score_170_l110_11041


namespace complex_equation_solution_l110_11021

theorem complex_equation_solution (z₁ z₂ : ℂ) : 
  z₁ = 1 - I ∧ z₁ * z₂ = 1 + I → z₂ = I :=
by sorry

end complex_equation_solution_l110_11021


namespace distance_to_incenter_in_isosceles_right_triangle_l110_11025

/-- An isosceles right triangle with hypotenuse length 6 -/
structure IsoscelesRightTriangle where
  -- A is the right angle
  AB : ℝ
  BC : ℝ
  AC : ℝ
  isIsosceles : AB = BC
  isRight : AC = 6

/-- The incenter of a triangle -/
def incenter (t : IsoscelesRightTriangle) : ℝ × ℝ := sorry

/-- The distance from a vertex to the incenter -/
def distanceToIncenter (t : IsoscelesRightTriangle) : ℝ := sorry

theorem distance_to_incenter_in_isosceles_right_triangle (t : IsoscelesRightTriangle) :
  distanceToIncenter t = 6 * Real.sqrt 2 - 6 := by sorry

end distance_to_incenter_in_isosceles_right_triangle_l110_11025


namespace number_calculation_l110_11028

theorem number_calculation (N : ℝ) : 
  0.2 * (|(-0.05)|^3 * 0.35 * 0.7 * N) = 182.7 → N = 20880000 := by
  sorry

end number_calculation_l110_11028


namespace pizza_dough_milk_calculation_l110_11037

/-- Given a ratio of milk to flour for pizza dough, calculate the amount of milk needed for a specific amount of flour. -/
theorem pizza_dough_milk_calculation 
  (milk_base : ℚ)  -- Base amount of milk in mL
  (flour_base : ℚ) -- Base amount of flour in mL
  (flour_total : ℚ) -- Total amount of flour to be used in mL
  (h1 : milk_base = 50)  -- Condition 1: Base milk amount
  (h2 : flour_base = 250) -- Condition 1: Base flour amount
  (h3 : flour_total = 750) -- Condition 2: Total flour amount
  : (flour_total / flour_base) * milk_base = 150 := by
  sorry

end pizza_dough_milk_calculation_l110_11037


namespace complementary_and_supplementary_angles_l110_11035

/-- Given an angle of 46 degrees, its complementary angle is 44 degrees and its supplementary angle is 134 degrees. -/
theorem complementary_and_supplementary_angles (angle : ℝ) : 
  angle = 46 → 
  (90 - angle = 44) ∧ (180 - angle = 134) := by
  sorry

end complementary_and_supplementary_angles_l110_11035


namespace shortest_tangent_length_l110_11008

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 25
def C₂ (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 100

-- Define the tangent line segment
def is_tangent (R S : ℝ × ℝ) : Prop :=
  C₁ R.1 R.2 ∧ C₂ S.1 S.2 ∧
  ∀ T : ℝ × ℝ, (T ≠ R ∧ T ≠ S) →
    (C₁ T.1 T.2 → (T.1 - R.1)^2 + (T.2 - R.2)^2 < (S.1 - R.1)^2 + (S.2 - R.2)^2) ∧
    (C₂ T.1 T.2 → (T.1 - S.1)^2 + (T.2 - S.2)^2 < (R.1 - S.1)^2 + (R.2 - S.2)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∃ R S : ℝ × ℝ, is_tangent R S ∧
    ∀ R' S' : ℝ × ℝ, is_tangent R' S' →
      Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) ≤ Real.sqrt ((S'.1 - R'.1)^2 + (S'.2 - R'.2)^2) ∧
      Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) = 5 * Real.sqrt 15 + 10 :=
sorry

end shortest_tangent_length_l110_11008


namespace fraction_evaluation_l110_11044

theorem fraction_evaluation : (18 : ℝ) / (4.9 * 106) = 18 / 519.4 := by
  sorry

end fraction_evaluation_l110_11044


namespace range_of_a_l110_11082

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = B a → a ≤ 0 ∨ a ≥ 3 := by
  sorry

end range_of_a_l110_11082


namespace odd_integer_sequence_sum_l110_11014

theorem odd_integer_sequence_sum (n : ℕ) : n > 0 → (
  let sum := n / 2 * (5 + (6 * n - 1))
  sum = 597 ↔ n = 13
) := by sorry

end odd_integer_sequence_sum_l110_11014


namespace golden_retriever_pup_difference_l110_11065

/-- Represents the number of pups each golden retriever had more than each husky -/
def pup_difference : ℕ := 2

theorem golden_retriever_pup_difference :
  let num_huskies : ℕ := 5
  let num_pitbulls : ℕ := 2
  let num_golden_retrievers : ℕ := 4
  let pups_per_husky : ℕ := 3
  let pups_per_pitbull : ℕ := 3
  let total_adult_dogs : ℕ := num_huskies + num_pitbulls + num_golden_retrievers
  let total_pups : ℕ := total_adult_dogs + 30
  total_pups = num_huskies * pups_per_husky + num_pitbulls * pups_per_pitbull + 
               num_golden_retrievers * (pups_per_husky + pup_difference) →
  pup_difference = 2 := by
  sorry

end golden_retriever_pup_difference_l110_11065


namespace polar_to_cartesian_l110_11078

/-- Given a point M in polar coordinates (r, θ), 
    prove that its Cartesian coordinates are (x, y) --/
theorem polar_to_cartesian 
  (r : ℝ) (θ : ℝ) 
  (x : ℝ) (y : ℝ) 
  (h1 : r = 2) 
  (h2 : θ = π/6) 
  (h3 : x = r * Real.cos θ) 
  (h4 : y = r * Real.sin θ) : 
  x = Real.sqrt 3 ∧ y = 1 := by
  sorry

end polar_to_cartesian_l110_11078


namespace point_on_bisector_l110_11020

/-- If A(a, b) and B(b, a) represent the same point, then this point lies on the line y = x. -/
theorem point_on_bisector (a b : ℝ) : (a, b) = (b, a) → a = b :=
by sorry

end point_on_bisector_l110_11020


namespace total_cost_calculation_l110_11073

/-- The cost per kilogram of mangos -/
def mango_cost : ℝ := sorry

/-- The cost per kilogram of rice -/
def rice_cost : ℝ := sorry

/-- The cost per kilogram of flour -/
def flour_cost : ℝ := 22

theorem total_cost_calculation : 
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 941.6) :=
by sorry

end total_cost_calculation_l110_11073


namespace new_average_age_l110_11054

theorem new_average_age (n : ℕ) (initial_avg : ℝ) (new_person_age : ℝ) :
  n = 17 ∧ initial_avg = 14 ∧ new_person_age = 32 →
  (n * initial_avg + new_person_age) / (n + 1) = 15 := by
  sorry

end new_average_age_l110_11054


namespace peters_initial_money_l110_11092

/-- The cost of Peter's glasses purchase --/
def glasses_purchase (small_cost large_cost : ℕ) (small_count large_count : ℕ) (change : ℕ) : Prop :=
  ∃ (initial_amount : ℕ),
    initial_amount = small_cost * small_count + large_cost * large_count + change

/-- Theorem stating Peter's initial amount of money --/
theorem peters_initial_money :
  glasses_purchase 3 5 8 5 1 → ∃ (initial_amount : ℕ), initial_amount = 50 :=
by
  sorry

end peters_initial_money_l110_11092


namespace quadratic_minimum_l110_11079

/-- Given a quadratic function f(x) = ax² + bx + 1 where a ≠ 0,
    if the solution set of f(x) > 0 is {x | x ∈ ℝ, x ≠ -b/(2a)},
    then the minimum value of (b⁴ + 4)/(4a) is 4. -/
theorem quadratic_minimum (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (a * x^2 + b * x + 1 > 0 ↔ x ≠ -b / (2 * a))) →
  (∃ m : ℝ, m = 4 ∧ ∀ y : ℝ, y = (b^4 + 4) / (4 * a) → y ≥ m) :=
by sorry

end quadratic_minimum_l110_11079


namespace height_difference_l110_11038

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height_m : ℝ := 324

/-- The height of the Eiffel Tower in feet -/
def eiffel_tower_height_ft : ℝ := 1063

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height_m : ℝ := 830

/-- The height of the Burj Khalifa in feet -/
def burj_khalifa_height_ft : ℝ := 2722

/-- The difference in height between the Burj Khalifa and the Eiffel Tower in meters and feet -/
theorem height_difference :
  (burj_khalifa_height_m - eiffel_tower_height_m = 506) ∧
  (burj_khalifa_height_ft - eiffel_tower_height_ft = 1659) := by
  sorry

end height_difference_l110_11038


namespace problem_solution_l110_11061

theorem problem_solution (x y : ℝ) (hx : x = 3 + 2 * Real.sqrt 2) (hy : y = 3 - 2 * Real.sqrt 2) :
  (x + y = 6) ∧
  (x - y = 4 * Real.sqrt 2) ∧
  (x * y = 1) ∧
  (x^2 - 3*x*y + y^2 - x - y = 25) := by
sorry

end problem_solution_l110_11061


namespace negation_of_universal_proposition_l110_11084

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l110_11084


namespace three_solutions_l110_11045

/-- Represents a solution to the equation A^B = BA --/
structure Solution :=
  (A B : Nat)
  (h1 : A ≠ B)
  (h2 : A ≥ 1 ∧ A ≤ 9)
  (h3 : B ≥ 1 ∧ B ≤ 9)
  (h4 : A^B = 10*B + A)
  (h5 : 10*B + A ≠ A*B)

/-- The set of all valid solutions --/
def validSolutions : Set Solution := {s | s.A^s.B = 10*s.B + s.A ∧ s.A ≠ s.B ∧ s.A ≥ 1 ∧ s.A ≤ 9 ∧ s.B ≥ 1 ∧ s.B ≤ 9 ∧ 10*s.B + s.A ≠ s.A*s.B}

/-- Theorem stating that there are exactly three solutions --/
theorem three_solutions :
  ∃ (s1 s2 s3 : Solution),
    validSolutions = {s1, s2, s3} ∧
    ((s1.A = 2 ∧ s1.B = 5) ∨ (s1.A = 6 ∧ s1.B = 2) ∨ (s1.A = 4 ∧ s1.B = 3)) ∧
    ((s2.A = 2 ∧ s2.B = 5) ∨ (s2.A = 6 ∧ s2.B = 2) ∨ (s2.A = 4 ∧ s2.B = 3)) ∧
    ((s3.A = 2 ∧ s3.B = 5) ∨ (s3.A = 6 ∧ s3.B = 2) ∨ (s3.A = 4 ∧ s3.B = 3)) ∧
    s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 :=
sorry

end three_solutions_l110_11045


namespace function_characterization_l110_11076

def is_prime (p : ℕ) : Prop := Nat.Prime p

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ n p, is_prime p → (f n)^p ≡ n [MOD f p]

def is_identity (f : ℕ → ℕ) : Prop :=
  ∀ n, f n = n

def is_constant_one_on_primes (f : ℕ → ℕ) : Prop :=
  ∀ p, is_prime p → f p = 1

def is_special_function (f : ℕ → ℕ) : Prop :=
  f 2 = 2 ∧
  (∀ p, is_prime p → p > 2 → f p = 1) ∧
  (∀ n, f n ≡ n [MOD 2])

theorem function_characterization (f : ℕ → ℕ) :
  satisfies_condition f →
  (is_identity f ∨ is_constant_one_on_primes f ∨ is_special_function f) :=
sorry

end function_characterization_l110_11076


namespace interest_difference_implies_principal_l110_11006

/-- Proves that if the difference between compound interest and simple interest 
    on a sum at 10% per annum for 2 years is Rs. 61, then the sum (principal) is Rs. 6100. -/
theorem interest_difference_implies_principal (P : ℝ) : 
  P * (1 + 0.1)^2 - P - (P * 0.1 * 2) = 61 → P = 6100 := by
  sorry

end interest_difference_implies_principal_l110_11006


namespace sum_squared_odd_l110_11064

theorem sum_squared_odd (a b c : ℤ) (h : (a + b + c) % 2 = 1) : 
  (a^2 + b^2 - c^2 + 2*a*b) % 2 = 1 := by
  sorry

end sum_squared_odd_l110_11064


namespace football_club_arrangements_l110_11017

theorem football_club_arrangements (n : ℕ) (k : ℕ) 
  (h1 : n = 9) 
  (h2 : k = 2) : 
  (Nat.factorial n) * (Nat.choose n k) = 13063680 :=
by sorry

end football_club_arrangements_l110_11017


namespace product_digit_exclusion_l110_11088

theorem product_digit_exclusion : ∃ d : ℕ, d < 10 ∧ 
  (32 % 10 ≠ d) ∧ ((1024 / 32) % 10 ≠ d) := by
  sorry

end product_digit_exclusion_l110_11088


namespace spherical_coords_negated_y_theorem_l110_11015

/-- Given a point with rectangular coordinates (x, y, z) and spherical coordinates (ρ, θ, φ),
    this function returns the spherical coordinates of the point (x, -y, z) -/
def spherical_coords_negated_y (x y z ρ θ φ : Real) : Real × Real × Real :=
  sorry

/-- Theorem stating that if a point has rectangular coordinates (x, y, z) and 
    spherical coordinates (3, 5π/6, 5π/12), then the point with rectangular 
    coordinates (x, -y, z) has spherical coordinates (3, π/6, 5π/12) -/
theorem spherical_coords_negated_y_theorem (x y z : Real) :
  let (ρ, θ, φ) := (3, 5*π/6, 5*π/12)
  (x = ρ * Real.sin φ * Real.cos θ) →
  (y = ρ * Real.sin φ * Real.sin θ) →
  (z = ρ * Real.cos φ) →
  spherical_coords_negated_y x y z ρ θ φ = (3, π/6, 5*π/12) :=
by
  sorry

end spherical_coords_negated_y_theorem_l110_11015


namespace tens_digit_of_2013_pow_2018_minus_2019_l110_11048

theorem tens_digit_of_2013_pow_2018_minus_2019 :
  ∃ n : ℕ, 2013^2018 - 2019 = 100 * n + 50 :=
sorry

end tens_digit_of_2013_pow_2018_minus_2019_l110_11048


namespace triangle_properties_l110_11060

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  sin (2 * C) = Real.sqrt 3 * sin C →
  b = 4 →
  (1 / 2) * a * b * sin C = 2 * Real.sqrt 3 →
  -- Conclusions
  C = π / 6 ∧
  a + b + c = 6 + 2 * Real.sqrt 3 :=
by sorry

end triangle_properties_l110_11060


namespace ellipse_parabola_intersection_l110_11016

theorem ellipse_parabola_intersection (n m : ℝ) :
  (∀ x y : ℝ, x^2/n + y^2/9 = 1 ∧ y = x^2 - m → 
    (3/n < m ∧ m < (4*m^2 + 9)/(4*m) ∧ m > 3/2)) ∧
  (m = 4 ∧ n = 4 → 
    ∀ x y : ℝ, x^2/n + y^2/9 = 1 ∧ y = x^2 - m → 
      4*x^2 + 4*y^2 - 5*y - 16 = 0) := by
sorry

end ellipse_parabola_intersection_l110_11016


namespace jim_flour_on_counter_l110_11075

/-- The amount of flour Jim has on the kitchen counter -/
def flour_on_counter (flour_in_cupboard flour_in_pantry flour_per_loaf : ℕ) (loaves_can_bake : ℕ) : ℕ :=
  loaves_can_bake * flour_per_loaf - (flour_in_cupboard + flour_in_pantry)

/-- Theorem stating that Jim has 100g of flour on the kitchen counter -/
theorem jim_flour_on_counter :
  flour_on_counter 200 100 200 2 = 100 := by
  sorry

end jim_flour_on_counter_l110_11075


namespace x_squared_minus_y_squared_l110_11055

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end x_squared_minus_y_squared_l110_11055


namespace group_size_proof_l110_11039

def group_collection (n : ℕ) : ℕ := n * n

theorem group_size_proof (total_rupees : ℕ) (h : group_collection 90 = total_rupees * 100) : 
  ∃ (n : ℕ), group_collection n = total_rupees * 100 ∧ n = 90 := by
  sorry

end group_size_proof_l110_11039


namespace inscribed_rectangle_coefficient_l110_11070

-- Define the triangle
def triangle_side1 : ℝ := 15
def triangle_side2 : ℝ := 36
def triangle_side3 : ℝ := 39

-- Define the rectangle's area formula
def rectangle_area (α β ω : ℝ) : ℝ := α * ω - β * ω^2

-- State the theorem
theorem inscribed_rectangle_coefficient :
  ∃ (α β : ℝ),
    (∀ ω, rectangle_area α β ω ≥ 0) ∧
    (rectangle_area α β triangle_side2 = 0) ∧
    (rectangle_area α β (triangle_side2 / 2) = 
      (triangle_side1 + triangle_side2 + triangle_side3) * 
      (triangle_side1 + triangle_side2 - triangle_side3) * 
      (triangle_side1 - triangle_side2 + triangle_side3) * 
      (-triangle_side1 + triangle_side2 + triangle_side3) / 
      (4 * 16)) ∧
    β = 5 / 12 := by
  sorry

end inscribed_rectangle_coefficient_l110_11070


namespace equidistant_point_on_y_axis_l110_11072

theorem equidistant_point_on_y_axis : ∃ y : ℚ, 
  let A : ℚ × ℚ := (-3, 1)
  let B : ℚ × ℚ := (-2, 5)
  let P : ℚ × ℚ := (0, y)
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧ y = 19/8 := by
  sorry

end equidistant_point_on_y_axis_l110_11072


namespace necessary_but_not_sufficient_l110_11097

theorem necessary_but_not_sufficient (a : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + a < 0) → a < 11 ∧
  ∃ a : ℝ, a < 11 ∧ ∀ x : ℝ, x^2 - 2*x + a ≥ 0 :=
by sorry

end necessary_but_not_sufficient_l110_11097


namespace morse_high_school_students_l110_11091

/-- The number of students in the other three grades at Morse High School -/
def other_grades_students : ℕ := 1500

/-- The number of seniors at Morse High School -/
def seniors : ℕ := 300

/-- The percentage of seniors who have cars -/
def senior_car_percentage : ℚ := 40 / 100

/-- The percentage of students in other grades who have cars -/
def other_grades_car_percentage : ℚ := 10 / 100

/-- The percentage of all students who have cars -/
def total_car_percentage : ℚ := 15 / 100

theorem morse_high_school_students :
  (seniors * senior_car_percentage + other_grades_students * other_grades_car_percentage : ℚ) =
  (seniors + other_grades_students) * total_car_percentage :=
sorry

end morse_high_school_students_l110_11091


namespace sons_age_l110_11011

theorem sons_age (mother_age son_age : ℕ) : 
  mother_age = 4 * son_age →
  mother_age + son_age = 49 + 6 →
  son_age = 11 := by
sorry

end sons_age_l110_11011


namespace cistern_length_l110_11031

/-- Given a cistern with specified dimensions, prove its length is 12 meters. -/
theorem cistern_length (width : ℝ) (depth : ℝ) (total_area : ℝ) :
  width = 14 →
  depth = 1.25 →
  total_area = 233 →
  width * depth * 2 + width * (total_area / width / depth - width) + depth * (total_area / width / depth - width) * 2 = total_area →
  total_area / width / depth - width = 12 :=
by sorry

end cistern_length_l110_11031


namespace polynomial_remainder_theorem_l110_11051

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^51 + 51) % (x + 1) = 50 := by
  sorry

end polynomial_remainder_theorem_l110_11051


namespace max_digit_sum_diff_l110_11052

/-- Sum of digits function -/
def S (x : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem max_digit_sum_diff :
  (∀ x : ℕ, x > 0 → S (x + 2019) - S x ≤ 12) ∧
  (∃ x : ℕ, x > 0 ∧ S (x + 2019) - S x = 12) :=
sorry

end max_digit_sum_diff_l110_11052


namespace pitcher_juice_distribution_l110_11062

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) :
  let juice_in_pitcher := C / 2
  let cups := 4
  let juice_per_cup := juice_in_pitcher / cups
  juice_per_cup / C * 100 = 12.5 := by
sorry

end pitcher_juice_distribution_l110_11062


namespace simplify_sqrt_expression_l110_11024

theorem simplify_sqrt_expression (y : ℝ) (hy : y ≠ 0) :
  Real.sqrt (4 + ((y^6 - 4) / (3 * y^3))^2) = (Real.sqrt (y^12 + 28 * y^6 + 16)) / (3 * y^3) :=
by sorry

end simplify_sqrt_expression_l110_11024


namespace equation_solvable_by_factoring_l110_11003

/-- The equation to be solved -/
def equation (x : ℝ) : Prop := (5*x - 1)^2 = 3*(5*x - 1)

/-- Factoring method can be applied if the equation can be written as a product of factors equal to zero -/
def factoring_method_applicable (f : ℝ → Prop) : Prop :=
  ∃ g h : ℝ → ℝ, ∀ x, f x ↔ g x * h x = 0

/-- The theorem stating that the given equation can be solved using the factoring method -/
theorem equation_solvable_by_factoring : factoring_method_applicable equation := by
  sorry

end equation_solvable_by_factoring_l110_11003


namespace square_roots_problem_l110_11005

theorem square_roots_problem (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2*a + 1)^2 = x ∧ (a + 5)^2 = x) → a = 4 := by
  sorry

end square_roots_problem_l110_11005


namespace bunny_rate_is_three_l110_11081

/-- The number of times a single bunny comes out of its burrow per minute. -/
def bunny_rate : ℕ := sorry

/-- The number of bunnies. -/
def num_bunnies : ℕ := 20

/-- The number of hours observed. -/
def observation_hours : ℕ := 10

/-- The total number of times bunnies come out in the observation period. -/
def total_exits : ℕ := 36000

/-- Proves that the bunny_rate is 3 given the conditions of the problem. -/
theorem bunny_rate_is_three : bunny_rate = 3 := by
  sorry

end bunny_rate_is_three_l110_11081


namespace arithmetic_sequence_general_term_l110_11080

/-- An arithmetic sequence with first three terms a-1, a+1, and 2a+3 has general term 2n-3 -/
theorem arithmetic_sequence_general_term (a : ℝ) (n : ℕ) :
  let a₁ := a - 1
  let a₂ := a + 1
  let a₃ := 2 * a + 3
  let d := a₂ - a₁
  let aₙ := a₁ + (n - 1) * d
  (a₁ + a₃) / 2 = a₂ → aₙ = 2 * n - 3 :=
by sorry

end arithmetic_sequence_general_term_l110_11080


namespace arthur_winning_strategy_l110_11087

theorem arthur_winning_strategy (n : ℕ) (hn : n ≥ 2) :
  ∃ (A B : Finset ℕ), 
    A.card = n ∧ B.card = n ∧
    ∀ (k : ℕ) (hk : k ≤ 2*n - 2),
      ∃ (x : Fin k → ℕ),
        (∀ i : Fin k, ∃ (a : A) (b : B), x i = a * b) ∧
        (∀ i j l : Fin k, i ≠ j ∧ j ≠ l ∧ i ≠ l → ¬(x i ∣ x j * x l)) :=
by sorry

end arthur_winning_strategy_l110_11087


namespace car_ac_price_ratio_l110_11043

/-- Given a car that costs $500 more than an AC, and the AC costs $1500,
    prove that the ratio of the car's price to the AC's price is 4:3. -/
theorem car_ac_price_ratio :
  ∀ (car_price ac_price : ℕ),
  ac_price = 1500 →
  car_price = ac_price + 500 →
  (car_price : ℚ) / ac_price = 4 / 3 := by
sorry

end car_ac_price_ratio_l110_11043


namespace union_of_sets_l110_11032

theorem union_of_sets : 
  let A : Set ℤ := {0, 1}
  let B : Set ℤ := {0, -1}
  A ∪ B = {-1, 0, 1} := by
sorry

end union_of_sets_l110_11032


namespace falls_difference_l110_11096

/-- The number of falls for each person --/
structure Falls where
  steven : ℕ
  stephanie : ℕ
  sonya : ℕ

/-- The conditions of the problem --/
def satisfies_conditions (f : Falls) : Prop :=
  f.steven = 3 ∧
  f.stephanie > f.steven ∧
  f.sonya = 6 ∧
  f.sonya = f.stephanie / 2 - 2

/-- The theorem to prove --/
theorem falls_difference (f : Falls) (h : satisfies_conditions f) :
  f.stephanie - f.steven = 13 := by
  sorry

end falls_difference_l110_11096


namespace fundraising_goal_l110_11086

/-- Fundraising problem -/
theorem fundraising_goal (ken mary scott goal : ℕ) : 
  ken = 600 →
  mary = 5 * ken →
  mary = 3 * scott →
  ken + mary + scott = goal + 600 →
  goal = 4000 := by
  sorry

end fundraising_goal_l110_11086


namespace dans_remaining_money_l110_11013

/-- 
Given an initial amount of money and the cost of a candy bar, 
calculate the remaining amount after purchasing the candy bar.
-/
def remaining_money (initial_amount : ℝ) (candy_cost : ℝ) : ℝ :=
  initial_amount - candy_cost

/-- 
Theorem: Given an initial amount of $4 and a candy bar cost of $1, 
the remaining amount after purchasing the candy bar is $3.
-/
theorem dans_remaining_money : 
  remaining_money 4 1 = 3 := by
  sorry

end dans_remaining_money_l110_11013


namespace gcd_459_357_l110_11085

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l110_11085


namespace import_value_calculation_l110_11000

/-- Given the export value and its relationship to the import value, 
    calculate the import value. -/
theorem import_value_calculation (export_value : ℝ) (import_value : ℝ) : 
  export_value = 8.07 ∧ 
  export_value = 1.5 * import_value + 1.11 → 
  sorry

end import_value_calculation_l110_11000


namespace m_condition_necessary_not_sufficient_l110_11046

/-- The equation of a potential ellipse with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (3 - m) + y^2 / (m - 1) = 1

/-- The condition on m -/
def m_condition (m : ℝ) : Prop :=
  1 < m ∧ m < 3

/-- The equation represents an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  3 - m > 0 ∧ m - 1 > 0 ∧ m ≠ 2

/-- The m_condition is necessary but not sufficient for the equation to represent an ellipse -/
theorem m_condition_necessary_not_sufficient :
  (∀ m, is_ellipse m → m_condition m) ∧
  ¬(∀ m, m_condition m → is_ellipse m) :=
sorry

end m_condition_necessary_not_sufficient_l110_11046


namespace megan_total_songs_l110_11090

/-- Represents the number of albums bought in each genre -/
structure AlbumCounts where
  country : ℕ
  pop : ℕ
  rock : ℕ
  jazz : ℕ

/-- Represents the number of songs per album in each genre -/
structure SongsPerAlbum where
  country : ℕ
  pop : ℕ
  rock : ℕ
  jazz : ℕ

/-- Calculates the total number of songs bought -/
def totalSongs (counts : AlbumCounts) (songsPerAlbum : SongsPerAlbum) : ℕ :=
  counts.country * songsPerAlbum.country +
  counts.pop * songsPerAlbum.pop +
  counts.rock * songsPerAlbum.rock +
  counts.jazz * songsPerAlbum.jazz

/-- Theorem stating that Megan bought 160 songs in total -/
theorem megan_total_songs :
  let counts : AlbumCounts := ⟨2, 8, 5, 2⟩
  let songsPerAlbum : SongsPerAlbum := ⟨12, 7, 10, 15⟩
  totalSongs counts songsPerAlbum = 160 := by
  sorry

end megan_total_songs_l110_11090


namespace hyperbola_equation_l110_11083

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (y^2 / a^2) - (x^2 / b^2) = 1 ∧ a > 0 ∧ b > 0

-- Define the focus
def focus (x y : ℝ) : Prop :=
  x = 0 ∧ y = -2

-- Define the asymptote slope
def asymptote_slope (slope : ℝ) : Prop :=
  slope = Real.sqrt 3

-- Theorem statement
theorem hyperbola_equation 
  (a b x y : ℝ) 
  (h1 : hyperbola a b x y) 
  (h2 : focus 0 (-2)) 
  (h3 : asymptote_slope (a/b)) : 
  (y^2 / 3) - x^2 = 1 := by
sorry

end hyperbola_equation_l110_11083


namespace circle_symmetry_l110_11023

-- Define the symmetry condition
def symmetric_circles (C : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C ↔ ((-y : ℝ) - 1)^2 + (-x)^2 = 1

-- State the theorem
theorem circle_symmetry :
  ∀ C : Set (ℝ × ℝ),
  symmetric_circles C →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + (y + 1)^2 = 1) :=
by sorry

end circle_symmetry_l110_11023


namespace quadratic_function_properties_l110_11099

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The maximum value function of |f(x)| on [0,t] -/
noncomputable def φ (t : ℝ) : ℝ :=
  if t ≤ 1 then 2*t - t^2
  else if t ≤ 1 + Real.sqrt 2 then 1
  else t^2 - 2*t

theorem quadratic_function_properties :
  (∀ x, f x ≥ -1) ∧  -- minimum value is -1
  (f 0 = 0) ∧        -- f(0) = 0
  (∀ x, f (1 + x) = f (1 - x)) ∧  -- symmetry property
  (∃ a b c, ∀ x, f x = a*x^2 + b*x + c ∧ a ≠ 0) →  -- f is quadratic
  (∀ x, f x = x^2 - 2*x) ∧  -- part 1
  (∀ m, (∀ x, -3 ≤ x ∧ x ≤ 3 → f x > 2*m*x - 4) ↔ -3 < m ∧ m < 1) ∧  -- part 2
  (∀ t, t > 0 → ∀ x, 0 ≤ x ∧ x ≤ t → |f x| ≤ φ t) ∧  -- part 3
  (∀ t, t > 0 → ∃ x, 0 ≤ x ∧ x ≤ t ∧ |f x| = φ t)  -- part 3 (maximum is achieved)
:= by sorry

end quadratic_function_properties_l110_11099


namespace gcf_lcm_sum_36_56_l110_11074

theorem gcf_lcm_sum_36_56 : Nat.gcd 36 56 + Nat.lcm 36 56 = 508 := by
  sorry

end gcf_lcm_sum_36_56_l110_11074


namespace composite_property_l110_11036

theorem composite_property (n : ℕ) 
  (h1 : ∃ a : ℕ, 3 * n + 1 = a ^ 2)
  (h2 : ∃ b : ℕ, 10 * n + 1 = b ^ 2) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ 29 * n + 11 = x * y :=
sorry

end composite_property_l110_11036


namespace detergent_for_nine_pounds_l110_11063

/-- The amount of detergent needed for a given weight of clothes -/
def detergent_needed (rate : ℝ) (weight : ℝ) : ℝ := rate * weight

/-- Theorem: Given a rate of 2 ounces of detergent per pound of clothes,
    the amount of detergent needed for 9 pounds of clothes is 18 ounces -/
theorem detergent_for_nine_pounds :
  detergent_needed 2 9 = 18 := by
  sorry

end detergent_for_nine_pounds_l110_11063


namespace quadratic_properties_l110_11001

/-- A quadratic function y = x² + mx + n -/
def quadratic (m n : ℝ) (x : ℝ) : ℝ := x^2 + m*x + n

theorem quadratic_properties (m n : ℝ) :
  (∀ y₁ y₂ : ℝ, quadratic m n 1 = y₁ ∧ quadratic m n 3 = y₂ ∧ y₁ = y₂ → m = -4) ∧
  (m = -4 ∧ ∃! x, quadratic m n x = 0 → n = 4) ∧
  (∀ a b₁ b₂ : ℝ, quadratic m n a = b₁ ∧ quadratic m n 3 = b₂ ∧ b₁ > b₂ → a < 1 ∨ a > 3) :=
by sorry

end quadratic_properties_l110_11001


namespace compound_oxygen_count_l110_11093

/-- Represents the number of atoms of an element in a compound -/
@[ext] structure AtomCount where
  al : ℕ
  o : ℕ
  h : ℕ

/-- Calculates the molecular weight of a compound given its atom counts -/
def molecularWeight (atoms : AtomCount) : ℕ :=
  27 * atoms.al + 16 * atoms.o + atoms.h

/-- Theorem stating that a compound with 1 Al, 3 H, and molecular weight 78 has 3 O atoms -/
theorem compound_oxygen_count :
  ∃ (atoms : AtomCount),
    atoms.al = 1 ∧
    atoms.h = 3 ∧
    molecularWeight atoms = 78 ∧
    atoms.o = 3 := by
  sorry

end compound_oxygen_count_l110_11093


namespace simple_interest_time_calculation_l110_11058

theorem simple_interest_time_calculation 
  (principal : ℝ) 
  (simple_interest : ℝ) 
  (rate : ℝ) 
  (h1 : principal = 400) 
  (h2 : simple_interest = 160) 
  (h3 : rate = 20) : 
  (simple_interest * 100) / (principal * rate) = 2 := by
  sorry

end simple_interest_time_calculation_l110_11058


namespace sin45_plus_sqrt2_half_l110_11066

theorem sin45_plus_sqrt2_half (h : Real.sin (π / 4) = Real.sqrt 2 / 2) :
  Real.sin (π / 4) + Real.sqrt 2 / 2 = Real.sqrt 2 := by
  sorry

end sin45_plus_sqrt2_half_l110_11066


namespace base_eight_addition_l110_11030

/-- Given a base-8 addition where 5XY₈ + 32₈ = 62X₈, prove that X + Y = 12 in base 10 --/
theorem base_eight_addition (X Y : ℕ) : 
  (5 * 8^2 + X * 8 + Y) + 32 = 6 * 8^2 + 2 * 8 + X → X + Y = 12 := by
  sorry

end base_eight_addition_l110_11030


namespace partial_fraction_decomposition_l110_11026

theorem partial_fraction_decomposition :
  ∃ (P Q : ℝ), P = 5.5 ∧ Q = 1.5 ∧
  ∀ x : ℝ, x ≠ 12 → x ≠ -4 →
    (7 * x + 4) / (x^2 - 8*x - 48) = P / (x - 12) + Q / (x + 4) :=
by
  sorry

end partial_fraction_decomposition_l110_11026


namespace sum_of_roots_equals_three_l110_11069

theorem sum_of_roots_equals_three : ∃ (P Q : ℝ), 
  (∀ x : ℝ, 3 * x^2 - 9 * x + 6 = 0 ↔ (x = P ∨ x = Q)) ∧ 
  P + Q = 3 := by
  sorry

end sum_of_roots_equals_three_l110_11069


namespace beef_weight_after_processing_l110_11019

theorem beef_weight_after_processing (initial_weight : ℝ) (loss_percentage : ℝ) 
  (processed_weight : ℝ) (h1 : initial_weight = 840) (h2 : loss_percentage = 35) :
  processed_weight = initial_weight * (1 - loss_percentage / 100) → 
  processed_weight = 546 := by
  sorry

end beef_weight_after_processing_l110_11019


namespace coin_flip_probability_l110_11012

-- Define the coin values
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^5

-- Define the function to calculate successful outcomes
def successful_outcomes : ℕ := 18

-- Define the target value
def target_value : ℕ := 40

-- Theorem statement
theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = 9 / 16 := by
  sorry

end coin_flip_probability_l110_11012


namespace company_profit_achievable_l110_11089

/-- Represents the company's financial model -/
structure CompanyModel where
  investment : ℝ
  production_cost : ℝ
  advertising_fee : ℝ
  min_price : ℝ
  max_price : ℝ
  sales_function : ℝ → ℝ

/-- Calculates the profit for a given price -/
def profit (model : CompanyModel) (price : ℝ) : ℝ :=
  let sales := model.sales_function price
  sales * (price - model.production_cost) - model.investment - model.advertising_fee

/-- Theorem stating that the company can achieve a total profit of 35 million over two years -/
theorem company_profit_achievable (model : CompanyModel) 
  (h1 : model.investment = 25)
  (h2 : model.production_cost = 20)
  (h3 : model.advertising_fee = 5)
  (h4 : model.min_price = 30)
  (h5 : model.max_price = 70)
  (h6 : ∀ x, model.sales_function x = -x + 150)
  (h7 : ∀ x, model.min_price ≤ x ∧ x ≤ model.max_price → 0 ≤ model.sales_function x) :
  ∃ (price1 price2 : ℝ), 
    model.min_price ≤ price1 ∧ price1 ≤ model.max_price ∧
    model.min_price ≤ price2 ∧ price2 ≤ model.max_price ∧
    profit model price1 + profit model price2 = 35 :=
  sorry


end company_profit_achievable_l110_11089


namespace tens_digit_of_N_power_20_l110_11077

theorem tens_digit_of_N_power_20 (N : ℕ) 
  (h1 : Even N) 
  (h2 : ¬ (10 ∣ N)) : 
  (N^20 / 10) % 10 = 7 := by
sorry

end tens_digit_of_N_power_20_l110_11077


namespace f_difference_at_3_and_neg_3_l110_11067

-- Define the function f
def f (x : ℝ) : ℝ := x^6 - 2*x^4 + 7*x

-- State the theorem
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 42 := by
  sorry

end f_difference_at_3_and_neg_3_l110_11067


namespace min_value_geometric_sequence_l110_11007

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 7b₃ is -16/7 -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) :
  b₁ = 2 →
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) →
  (∀ x y : ℝ, x = b₂ ∧ y = b₃ → 3 * x + 7 * y ≥ -16/7) ∧
  (∃ x y : ℝ, x = b₂ ∧ y = b₃ ∧ 3 * x + 7 * y = -16/7) :=
by sorry

end min_value_geometric_sequence_l110_11007


namespace roots_sum_product_squares_l110_11047

theorem roots_sum_product_squares (x₁ x₂ : ℝ) : 
  ((x₁ - 2)^2 = 3*(x₁ + 5)) ∧ ((x₂ - 2)^2 = 3*(x₂ + 5)) →
  x₁*x₂ + x₁^2 + x₂^2 = 60 := by
sorry

end roots_sum_product_squares_l110_11047


namespace smallest_d_for_injective_g_l110_11098

def g (x : ℝ) : ℝ := (x - 3)^2 - 4

theorem smallest_d_for_injective_g :
  ∀ d : ℝ, (∀ x y : ℝ, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
sorry

end smallest_d_for_injective_g_l110_11098


namespace lcm_9_12_15_l110_11022

theorem lcm_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end lcm_9_12_15_l110_11022


namespace sector_central_angle_l110_11033

/-- Proves that a circular sector with arc length 4 and area 2 has a central angle of 4 radians -/
theorem sector_central_angle (l : ℝ) (A : ℝ) (θ : ℝ) (r : ℝ) :
  l = 4 →
  A = 2 →
  l = r * θ →
  A = 1/2 * r^2 * θ →
  θ = 4 := by
sorry


end sector_central_angle_l110_11033


namespace lizzy_scored_67_percent_l110_11049

/-- Represents the exam scores of four students -/
structure ExamScores where
  max_score : ℕ
  gibi_percent : ℕ
  jigi_percent : ℕ
  mike_percent : ℕ
  average_mark : ℕ

/-- Calculates Lizzy's score as a percentage -/
def lizzy_percent (scores : ExamScores) : ℕ :=
  let total_marks := scores.average_mark * 4
  let others_total := (scores.gibi_percent + scores.jigi_percent + scores.mike_percent) * scores.max_score / 100
  let lizzy_score := total_marks - others_total
  lizzy_score * 100 / scores.max_score

/-- Theorem stating that Lizzy's score is 67% given the conditions -/
theorem lizzy_scored_67_percent (scores : ExamScores)
  (h_max : scores.max_score = 700)
  (h_gibi : scores.gibi_percent = 59)
  (h_jigi : scores.jigi_percent = 55)
  (h_mike : scores.mike_percent = 99)
  (h_avg : scores.average_mark = 490) :
  lizzy_percent scores = 67 := by
  sorry

end lizzy_scored_67_percent_l110_11049


namespace f_upper_bound_l110_11071

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  f 1 = 1 ∧
  ∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂

theorem f_upper_bound (f : ℝ → ℝ) (h : f_properties f) :
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
sorry

end f_upper_bound_l110_11071


namespace trig_sum_equality_l110_11059

theorem trig_sum_equality : 
  (Real.cos (2 * π / 180)) / (Real.sin (47 * π / 180)) + 
  (Real.cos (88 * π / 180)) / (Real.sin (133 * π / 180)) = Real.sqrt 2 := by
  sorry

end trig_sum_equality_l110_11059


namespace max_value_sqrt_sum_l110_11010

/-- Given positive real numbers a, b, and c such that a + b + c = 1,
    the maximum value of √(a+1) + √(b+1) + √(c+1) is achieved
    when applying the Cauchy-Schwarz inequality. -/
theorem max_value_sqrt_sum (a b c : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (hsum : a + b + c = 1) : 
    ∃ (max : ℝ), ∀ (x y z : ℝ), 
    x > 0 → y > 0 → z > 0 → x + y + z = 1 →
    Real.sqrt (x + 1) + Real.sqrt (y + 1) + Real.sqrt (z + 1) ≤ max ∧
    Real.sqrt (a + 1) + Real.sqrt (b + 1) + Real.sqrt (c + 1) = max :=
sorry

end max_value_sqrt_sum_l110_11010


namespace degenerate_ellipse_max_y_coord_l110_11004

theorem degenerate_ellipse_max_y_coord :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ (x^2 / 36) + ((y + 5)^2 / 16)
  ∀ (x y : ℝ), f (x, y) = 0 → y ≤ -5 :=
by sorry

end degenerate_ellipse_max_y_coord_l110_11004


namespace susan_ate_six_candies_l110_11034

/-- Represents Susan's candy consumption and purchases over a week -/
structure CandyWeek where
  dailyLimit : ℕ
  tuesdayBought : ℕ
  thursdayBought : ℕ
  fridayBought : ℕ
  leftAtEndOfWeek : ℕ
  totalSpent : ℕ

/-- Calculates the number of candies Susan ate during the week -/
def candiesEaten (week : CandyWeek) : ℕ :=
  week.tuesdayBought + week.thursdayBought + week.fridayBought - week.leftAtEndOfWeek

/-- Theorem stating that Susan ate 6 candies during the week -/
theorem susan_ate_six_candies (week : CandyWeek)
  (h1 : week.dailyLimit = 3)
  (h2 : week.tuesdayBought = 3)
  (h3 : week.thursdayBought = 5)
  (h4 : week.fridayBought = 2)
  (h5 : week.leftAtEndOfWeek = 4)
  (h6 : week.totalSpent = 9) :
  candiesEaten week = 6 := by
  sorry

end susan_ate_six_candies_l110_11034


namespace min_value_is_214_l110_11009

-- Define the type for our permutations
def Permutation := Fin 9 → Fin 9

-- Define the function we want to minimize
def f (p : Permutation) : ℕ :=
  let x₁ := (p 0).val + 1
  let x₂ := (p 1).val + 1
  let x₃ := (p 2).val + 1
  let y₁ := (p 3).val + 1
  let y₂ := (p 4).val + 1
  let y₃ := (p 5).val + 1
  let z₁ := (p 6).val + 1
  let z₂ := (p 7).val + 1
  let z₃ := (p 8).val + 1
  x₁ * x₂ * x₃ + y₁ * y₂ * y₃ + z₁ * z₂ * z₃

theorem min_value_is_214 :
  (∃ (p : Permutation), f p = 214) ∧ (∀ (p : Permutation), f p ≥ 214) := by
  sorry

end min_value_is_214_l110_11009


namespace vector_magnitude_l110_11050

theorem vector_magnitude (a b : ℝ × ℝ) : 
  (norm a = 1) → 
  (norm (a - 2 • b) = Real.sqrt 21) → 
  (a.1 * b.1 + a.2 * b.2 = - (1/2) * norm a * norm b) →
  norm b = 2 := by
  sorry

end vector_magnitude_l110_11050


namespace difference_of_bounds_l110_11056

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := m * x + 2
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

-- Define the theorem
theorem difference_of_bounds (m : ℝ) :
  ∃ (a b : ℤ), (∀ x : ℝ, a ≤ f m x - g m x ∧ f m x - g m x ≤ b) ∧
  (∀ y : ℝ, a ≤ y ∧ y ≤ b → ∃ x : ℝ, f m x - g m x = y) →
  a - b = -2 :=
sorry

end difference_of_bounds_l110_11056


namespace arithmetic_mean_fractions_l110_11018

theorem arithmetic_mean_fractions : 
  let a := 8 / 11
  let b := 9 / 11
  let c := 5 / 6
  c = (a + b) / 2 := by sorry

end arithmetic_mean_fractions_l110_11018
