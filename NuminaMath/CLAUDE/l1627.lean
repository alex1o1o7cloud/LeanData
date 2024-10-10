import Mathlib

namespace solve_for_z_l1627_162797

theorem solve_for_z (m : ℕ) (z : ℝ) 
  (h1 : ((1 ^ (m + 1)) / (5 ^ (m + 1))) * ((1 ^ 18) / (z ^ 18)) = 1 / (2 * (10 ^ 35)))
  (h2 : m = 34) : 
  z = 4 := by
sorry

end solve_for_z_l1627_162797


namespace largest_lucky_number_is_499_l1627_162778

def lucky_number (a b : ℕ) : ℕ := a + b + a * b

def largest_lucky_number_after_three_operations : ℕ :=
  let n1 := lucky_number 1 4
  let n2 := lucky_number 4 n1
  let n3 := lucky_number n1 n2
  max n1 (max n2 n3)

theorem largest_lucky_number_is_499 :
  largest_lucky_number_after_three_operations = 499 := by sorry

end largest_lucky_number_is_499_l1627_162778


namespace number_of_black_balls_is_random_variable_l1627_162728

-- Define the total number of balls
def total_balls : ℕ := 8

-- Define the number of black balls
def black_balls : ℕ := 5

-- Define the number of white balls
def white_balls : ℕ := 3

-- Define the number of balls drawn
def drawn_balls : ℕ := 2

-- Define the possible outcomes for the number of black balls drawn
def possible_outcomes : Set ℕ := {0, 1, 2}

-- Define a random variable as a function from the sample space to the set of real numbers
def is_random_variable (X : Set ℕ → ℝ) : Prop :=
  ∀ n ∈ possible_outcomes, X {n} ∈ Set.range X

-- State the theorem
theorem number_of_black_balls_is_random_variable :
  ∃ X : Set ℕ → ℝ, is_random_variable X ∧ 
  (∀ n, X {n} = n) ∧
  (∀ n ∉ possible_outcomes, X {n} = 0) :=
sorry

end number_of_black_balls_is_random_variable_l1627_162728


namespace line_l_prime_equation_l1627_162794

-- Define the fixed point P
def P : ℝ × ℝ := (-1, 1)

-- Define the direction vector of line l'
def direction_vector : ℝ × ℝ := (3, 2)

-- Define the equation of line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y + m = 0

-- State the theorem
theorem line_l_prime_equation :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), line_l m x y ∧ (x, y) = P) →
  (∃ (k : ℝ), 2 * P.1 - 3 * P.2 + 5 = 0 ∧
              ∀ (t : ℝ), 2 * (P.1 + t * direction_vector.1) - 3 * (P.2 + t * direction_vector.2) + 5 = 0) :=
by sorry

end line_l_prime_equation_l1627_162794


namespace triangle_inequality_l1627_162747

theorem triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = 2) :
  a^2 + b^2 + c^2 < 2*(1 - a*b*c) := by
sorry

end triangle_inequality_l1627_162747


namespace socks_difference_l1627_162736

/-- Proves that after losing half of the white socks, the person still has 6 more white socks than black socks -/
theorem socks_difference (black_socks : ℕ) (white_socks : ℕ) : 
  black_socks = 6 →
  white_socks = 4 * black_socks →
  (white_socks / 2) - black_socks = 6 := by
sorry

end socks_difference_l1627_162736


namespace winning_game_score_is_3_0_l1627_162761

structure FootballTeam where
  games_played : ℕ
  total_goals_scored : ℕ
  total_goals_conceded : ℕ
  wins : ℕ
  draws : ℕ
  losses : ℕ

def winning_game_score (team : FootballTeam) : ℕ × ℕ := sorry

theorem winning_game_score_is_3_0 (team : FootballTeam) 
  (h1 : team.games_played = 3)
  (h2 : team.total_goals_scored = 3)
  (h3 : team.total_goals_conceded = 1)
  (h4 : team.wins = 1)
  (h5 : team.draws = 1)
  (h6 : team.losses = 1) :
  winning_game_score team = (3, 0) := by sorry

end winning_game_score_is_3_0_l1627_162761


namespace units_digit_of_m_squared_plus_two_to_m_l1627_162756

def m : ℕ := 2016^2 + 2^2016

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : 
  m = 2016^2 + 2^2016 → (m^2 + 2^m) % 10 = 7 := by
  sorry

end units_digit_of_m_squared_plus_two_to_m_l1627_162756


namespace annas_cupcake_sales_l1627_162777

/-- Anna's cupcake sales problem -/
theorem annas_cupcake_sales (num_trays : ℕ) (cupcakes_per_tray : ℕ) (price_per_cupcake : ℕ) (sold_fraction : ℚ) : 
  num_trays = 4 →
  cupcakes_per_tray = 20 →
  price_per_cupcake = 2 →
  sold_fraction = 3 / 5 →
  (num_trays * cupcakes_per_tray * sold_fraction * price_per_cupcake : ℚ) = 96 :=
by sorry

end annas_cupcake_sales_l1627_162777


namespace beanie_baby_ratio_l1627_162763

/-- The number of beanie babies Lori has -/
def lori_babies : ℕ := 300

/-- The total number of beanie babies Lori and Sydney have together -/
def total_babies : ℕ := 320

/-- The number of beanie babies Sydney has -/
def sydney_babies : ℕ := total_babies - lori_babies

/-- The ratio of Lori's beanie babies to Sydney's beanie babies -/
def beanie_ratio : ℚ := lori_babies / sydney_babies

theorem beanie_baby_ratio : beanie_ratio = 15 := by
  sorry

end beanie_baby_ratio_l1627_162763


namespace oldest_sibling_age_l1627_162765

/-- Represents the ages and relationships in Kay's family --/
structure KayFamily where
  kay_age : ℕ
  num_siblings : ℕ
  youngest_sibling_age : ℕ
  oldest_sibling_age : ℕ

/-- The conditions given in the problem --/
def kay_family_conditions (f : KayFamily) : Prop :=
  f.kay_age = 32 ∧
  f.num_siblings = 14 ∧
  f.youngest_sibling_age = f.kay_age / 2 - 5 ∧
  f.oldest_sibling_age = 4 * f.youngest_sibling_age

/-- Theorem stating that the oldest sibling's age is 44 given the conditions --/
theorem oldest_sibling_age (f : KayFamily) 
  (h : kay_family_conditions f) : f.oldest_sibling_age = 44 := by
  sorry


end oldest_sibling_age_l1627_162765


namespace urn_probability_l1627_162786

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents one operation on the urn -/
inductive Operation
  | Red
  | Blue

/-- Calculates the probability of a specific sequence of operations -/
def sequenceProbability (ops : List Operation) : ℚ :=
  sorry

/-- Calculates the number of sequences with 3 red and 2 blue operations -/
def validSequences : ℕ :=
  sorry

/-- The main theorem stating the probability of having 4 balls of each color -/
theorem urn_probability : 
  let initialState : UrnState := ⟨1, 1⟩
  let finalState : UrnState := ⟨4, 4⟩
  let totalOperations : ℕ := 5
  let probability : ℚ := (validSequences : ℚ) * sequenceProbability (List.replicate 3 Operation.Red ++ List.replicate 2 Operation.Blue)
  probability = 1 / 6 :=
sorry

end urn_probability_l1627_162786


namespace egyptian_fraction_for_odd_n_l1627_162744

theorem egyptian_fraction_for_odd_n (n : ℕ) 
  (h_odd : Odd n) 
  (h_gt3 : n > 3) 
  (h_not_div3 : ¬(3 ∣ n)) : 
  ∃ (a b c : ℕ), 
    Odd a ∧ Odd b ∧ Odd c ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (3 : ℚ) / n = 1 / a + 1 / b + 1 / c := by
  sorry

end egyptian_fraction_for_odd_n_l1627_162744


namespace trig_identity_l1627_162726

theorem trig_identity (θ : ℝ) (h : 2 * Real.sin θ + Real.cos θ = 0) :
  Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = - 6 / 5 := by
  sorry

end trig_identity_l1627_162726


namespace arithmetic_calculation_l1627_162746

theorem arithmetic_calculation : (-1) * (-3) + 3^2 / (8 - 5) = 6 := by
  sorry

end arithmetic_calculation_l1627_162746


namespace blackboard_numbers_l1627_162769

def blackboard_rule (a b : ℕ) : ℕ := a * b + a + b

def is_generable (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = 2^k * 3^m - 1

theorem blackboard_numbers :
  (is_generable 13121) ∧ (¬ is_generable 12131) := by sorry

end blackboard_numbers_l1627_162769


namespace weight_replacement_l1627_162725

theorem weight_replacement (n : ℕ) (old_weight new_weight avg_increase : ℝ) :
  n = 8 →
  new_weight = 95 →
  avg_increase = 2.5 →
  old_weight = new_weight - n * avg_increase →
  old_weight = 75 :=
by sorry

end weight_replacement_l1627_162725


namespace tea_box_duration_l1627_162718

-- Define the daily tea usage in ounces
def daily_usage : ℚ := 1 / 5

-- Define the box size in ounces
def box_size : ℚ := 28

-- Define the number of days in a week
def days_per_week : ℕ := 7

-- Theorem to prove
theorem tea_box_duration : 
  (box_size / daily_usage) / days_per_week = 20 := by
  sorry

end tea_box_duration_l1627_162718


namespace smallest_integer_negative_quadratic_l1627_162704

theorem smallest_integer_negative_quadratic :
  ∃ (x : ℤ), (∀ (y : ℤ), y^2 - 11*y + 24 < 0 → x ≤ y) ∧ x^2 - 11*x + 24 < 0 :=
by sorry

end smallest_integer_negative_quadratic_l1627_162704


namespace min_length_AB_l1627_162723

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the line y = 2
def line_y_2 (x y : ℝ) : Prop := y = 2

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- State the theorem
theorem min_length_AB :
  ∀ (x_A y_A x_B y_B : ℝ),
  line_y_2 x_A y_A →
  ellipse_C x_B y_B →
  perpendicular x_A y_A x_B y_B →
  ∀ (x y : ℝ),
  line_y_2 x y →
  ellipse_C x y →
  perpendicular x y x_B y_B →
  (x_A - x_B)^2 + (y_A - y_B)^2 ≤ (x - x_B)^2 + (y - y_B)^2 :=
sorry

end min_length_AB_l1627_162723


namespace polygon_angles_l1627_162717

theorem polygon_angles (n : ℕ) (sum_interior : ℝ) (sum_exterior : ℝ) : 
  sum_exterior = 180 → 
  sum_interior = 4 * sum_exterior → 
  sum_interior = (n - 2) * 180 → 
  n = 11 ∧ sum_interior = 1620 := by
  sorry

end polygon_angles_l1627_162717


namespace compound_composition_l1627_162707

/-- Prove that a compound with 2 I atoms and a molecular weight of 294 g/mol contains 1 Ca atom -/
theorem compound_composition (atomic_weight_Ca atomic_weight_I : ℝ) 
  (h1 : atomic_weight_Ca = 40.08)
  (h2 : atomic_weight_I = 126.90)
  (h3 : 2 * atomic_weight_I + atomic_weight_Ca = 294) : 
  ∃ (n : ℕ), n = 1 ∧ n * atomic_weight_Ca = 294 - 2 * atomic_weight_I :=
by sorry

end compound_composition_l1627_162707


namespace bus_catch_probability_l1627_162783

/-- The probability of catching a bus within 5 minutes -/
theorem bus_catch_probability 
  (p3 : ℝ) -- Probability of bus No. 3 arriving
  (p6 : ℝ) -- Probability of bus No. 6 arriving
  (h1 : p3 = 0.20) -- Given probability for bus No. 3
  (h2 : p6 = 0.60) -- Given probability for bus No. 6
  (h3 : 0 ≤ p3 ∧ p3 ≤ 1) -- p3 is a valid probability
  (h4 : 0 ≤ p6 ∧ p6 ≤ 1) -- p6 is a valid probability
  : p3 + p6 = 0.80 := by
  sorry

end bus_catch_probability_l1627_162783


namespace store_price_difference_l1627_162781

/-- Given the total price and quantity of shirts and sweaters, prove that the difference
    between the average price of a sweater and the average price of a shirt is $2. -/
theorem store_price_difference (shirt_price shirt_quantity sweater_price sweater_quantity : ℕ) 
  (h1 : shirt_price = 360)
  (h2 : shirt_quantity = 20)
  (h3 : sweater_price = 900)
  (h4 : sweater_quantity = 45) :
  (sweater_price / sweater_quantity : ℚ) - (shirt_price / shirt_quantity : ℚ) = 2 := by
  sorry

end store_price_difference_l1627_162781


namespace union_when_m_is_neg_half_subset_iff_m_geq_zero_l1627_162799

-- Define sets A and B
def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1 - m}

-- Theorem 1: When m = -1/2, A ∪ B = {x | -2 < x < 3/2}
theorem union_when_m_is_neg_half :
  A ∪ B (-1/2) = {x : ℝ | -2 < x ∧ x < 3/2} := by sorry

-- Theorem 2: B ⊆ A if and only if m ≥ 0
theorem subset_iff_m_geq_zero :
  ∀ m : ℝ, B m ⊆ A ↔ m ≥ 0 := by sorry

end union_when_m_is_neg_half_subset_iff_m_geq_zero_l1627_162799


namespace quadratic_polynomial_existence_l1627_162750

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a complex number -/
def evaluate (p : QuadraticPolynomial) (z : ℂ) : ℂ :=
  p.a * z^2 + p.b * z + p.c

theorem quadratic_polynomial_existence : ∃ (p : QuadraticPolynomial),
  (evaluate p (-3 - 4*I) = 0) ∧ 
  (p.b = -10) ∧
  (p.a = -5/3) ∧ 
  (p.c = -125/3) := by
  sorry

end quadratic_polynomial_existence_l1627_162750


namespace complex_equation_first_quadrant_l1627_162708

theorem complex_equation_first_quadrant (z : ℂ) (a : ℝ) : 
  (1 - I) * z = a * I + 1 → 
  (z.re > 0 ∧ z.im > 0) → 
  a = 0 := by sorry

end complex_equation_first_quadrant_l1627_162708


namespace green_ball_probability_l1627_162743

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- Theorem: The probability of selecting a green ball is 26/45 -/
theorem green_ball_probability :
  let containerA : Container := ⟨5, 7⟩
  let containerB : Container := ⟨4, 5⟩
  let containerC : Container := ⟨7, 3⟩
  let totalContainers : ℕ := 3
  let probA : ℚ := 1 / totalContainers * greenProbability containerA
  let probB : ℚ := 1 / totalContainers * greenProbability containerB
  let probC : ℚ := 1 / totalContainers * greenProbability containerC
  probA + probB + probC = 26 / 45 := by
  sorry

end green_ball_probability_l1627_162743


namespace max_negative_integers_l1627_162758

theorem max_negative_integers
  (a b c d e f : ℤ)
  (h : a * b + c * d * e * f < 0) :
  ∃ (neg_count : ℕ),
    neg_count ≤ 4 ∧
    ∀ (n : ℕ),
      (∃ (neg_set : Finset ℤ),
        neg_set.card = n ∧
        neg_set ⊆ {a, b, c, d, e, f} ∧
        (∀ x ∈ neg_set, x < 0) ∧
        (∀ x ∈ {a, b, c, d, e, f} \ neg_set, x ≥ 0)) →
      n ≤ neg_count :=
sorry

end max_negative_integers_l1627_162758


namespace sqrt_1600_minus_24_form_l1627_162742

theorem sqrt_1600_minus_24_form (a b : ℕ+) :
  (Real.sqrt 1600 - 24 : ℝ) = ((Real.sqrt a.val - b.val) : ℝ)^2 →
  a.val + b.val = 102 := by
  sorry

end sqrt_1600_minus_24_form_l1627_162742


namespace complex_expression_evaluation_l1627_162785

theorem complex_expression_evaluation : 
  let expr := (((32400 * 4^3) / (3 * Real.sqrt 343)) / 18 / (7^3 * 10)) / 
              ((2 * Real.sqrt ((49^2 * 11)^4)) / 25^3)
  ∃ ε > 0, abs (expr - 0.00005366) < ε := by
sorry

end complex_expression_evaluation_l1627_162785


namespace triangle_area_5_5_6_l1627_162796

/-- The area of a triangle with sides 5, 5, and 6 units is 12 square units. -/
theorem triangle_area_5_5_6 : ∃ (A : ℝ), A = 12 ∧ A = Real.sqrt (8 * (8 - 5) * (8 - 5) * (8 - 6)) := by
  sorry

end triangle_area_5_5_6_l1627_162796


namespace oplus_five_two_l1627_162702

def oplus (a b : ℝ) : ℝ := 3 * a + 4 * b

theorem oplus_five_two : oplus 5 2 = 23 := by sorry

end oplus_five_two_l1627_162702


namespace dinner_bill_problem_l1627_162720

theorem dinner_bill_problem (P : ℝ) : 
  (P * 0.9 + P * 0.08 + P * 0.15) - (P * 0.85 + P * 0.06 + P * 0.85 * 0.15) = 1 → 
  P = 400 / 37 := by
sorry

end dinner_bill_problem_l1627_162720


namespace pizzas_served_today_l1627_162722

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := 6

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := lunch_pizzas + dinner_pizzas

theorem pizzas_served_today : total_pizzas = 15 := by
  sorry

end pizzas_served_today_l1627_162722


namespace range_of_a_l1627_162749

theorem range_of_a (a : ℝ) : 
  let M : Set ℝ := {a}
  let P : Set ℝ := {x | -1 < x ∧ x < 1}
  M ⊆ P → a ∈ P := by
sorry

end range_of_a_l1627_162749


namespace expression_simplification_l1627_162732

theorem expression_simplification (a : ℕ) (h : a = 2023) :
  (a^3 - 2*a^2*(a+1) + 3*a*(a+1)^2 - (a+1)^3 + 2) / (a*(a+1)) = a + 1 / (a*(a+1)) := by
  sorry

end expression_simplification_l1627_162732


namespace quadratic_root_implies_k_l1627_162755

/-- 
If the quadratic equation x^2 + kx - 3 = 0 has 1 as a root, 
then k = 2.
-/
theorem quadratic_root_implies_k (k : ℝ) : 
  (1 : ℝ)^2 + k*(1 : ℝ) - 3 = 0 → k = 2 := by
  sorry

end quadratic_root_implies_k_l1627_162755


namespace ball_transfer_equality_l1627_162754

/-- Represents a box containing balls of different colors -/
structure Box where
  black : ℕ
  white : ℕ

/-- Transfers balls between boxes -/
def transfer (a b : Box) (n : ℕ) : Box × Box :=
  let blackToB := min n a.black
  let whiteToA := min (n - blackToB) b.white
  let blackToA := n - whiteToA
  ({ black := a.black - blackToB + blackToA,
     white := a.white + whiteToA },
   { black := b.black + blackToB - blackToA,
     white := b.white - whiteToA })

theorem ball_transfer_equality (a b : Box) (n : ℕ) :
  let (a', b') := transfer a b n
  a'.white = b'.black := by sorry

end ball_transfer_equality_l1627_162754


namespace least_subtraction_for_divisibility_l1627_162767

theorem least_subtraction_for_divisibility : ∃! n : ℕ, n ≤ 12 ∧ (427398 - n) % 13 = 0 ∧ ∀ m : ℕ, m < n → (427398 - m) % 13 ≠ 0 := by
  sorry

end least_subtraction_for_divisibility_l1627_162767


namespace phi_value_l1627_162741

theorem phi_value : ∃ (Φ : ℕ), 504 / Φ = 40 + 3 * Φ ∧ 0 ≤ Φ ∧ Φ ≤ 9 ∧ Φ = 8 := by
  sorry

end phi_value_l1627_162741


namespace triangle_special_angle_l1627_162734

/-- In a triangle ABC, if 2b*cos(A) = 2c - sqrt(3)*a, then the measure of angle B is π/6 --/
theorem triangle_special_angle (a b c : ℝ) (A B : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A) (h5 : A < π) (h6 : 0 < B) (h7 : B < π)
  (h8 : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a) :
  B = π / 6 := by
  sorry

end triangle_special_angle_l1627_162734


namespace duck_cow_problem_l1627_162770

theorem duck_cow_problem (ducks cows : ℕ) : 
  2 * ducks + 4 * cows = 2 * (ducks + cows) + 34 → cows = 17 := by
  sorry

end duck_cow_problem_l1627_162770


namespace expression_value_l1627_162738

theorem expression_value : (45 - 13)^2 - (45^2 + 13^2) = -1170 := by
  sorry

end expression_value_l1627_162738


namespace days_with_parrot_l1627_162768

-- Define the given conditions
def total_phrases : ℕ := 17
def phrases_per_week : ℕ := 2
def initial_phrases : ℕ := 3
def days_per_week : ℕ := 7

-- Define the theorem
theorem days_with_parrot : 
  (total_phrases - initial_phrases) / phrases_per_week * days_per_week = 49 := by
  sorry

end days_with_parrot_l1627_162768


namespace not_in_range_of_g_l1627_162735

/-- The function g(x) defined as x^3 + x^2 + bx + 2 -/
def g (b : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + b*x + 2

/-- Theorem stating that for all real b ≠ 6, -2 is not in the range of g(x) -/
theorem not_in_range_of_g (b : ℝ) (h : b ≠ 6) :
  ¬∃ x, g b x = -2 := by sorry

end not_in_range_of_g_l1627_162735


namespace book_profit_rate_l1627_162733

/-- Calculates the rate of profit given cost price and selling price -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at 50 rupees and sold at 80 rupees is 60% -/
theorem book_profit_rate :
  let cost_price : ℚ := 50
  let selling_price : ℚ := 80
  rate_of_profit cost_price selling_price = 60 := by
  sorry

end book_profit_rate_l1627_162733


namespace lcm_sum_bound_l1627_162798

theorem lcm_sum_bound (a b c d e : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > e) (h5 : e > 1) :
  (1 : ℚ) / Nat.lcm a b + (1 : ℚ) / Nat.lcm b c + (1 : ℚ) / Nat.lcm c d + (1 : ℚ) / Nat.lcm d e ≤ 15 / 16 := by
  sorry

end lcm_sum_bound_l1627_162798


namespace rectangle_area_l1627_162740

theorem rectangle_area (w : ℝ) (h : w > 0) : 
  w^2 + (3*w)^2 = 16^2 → w * (3*w) = 76.8 := by
  sorry

end rectangle_area_l1627_162740


namespace meryll_question_ratio_l1627_162703

theorem meryll_question_ratio : 
  ∀ (total_mc : ℕ) (total_ps : ℕ) (written_mc_fraction : ℚ) (remaining : ℕ),
    total_mc = 35 →
    total_ps = 15 →
    written_mc_fraction = 2/5 →
    remaining = 31 →
    (total_mc * written_mc_fraction).num.toNat + 
    (total_ps - (remaining - (total_mc - (total_mc * written_mc_fraction).num.toNat))) = 
    total_ps / 3 :=
by
  sorry

end meryll_question_ratio_l1627_162703


namespace jacket_price_reduction_l1627_162753

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P > 0 → 
  0 ≤ x → x ≤ 100 →
  P * (1 - x / 100) * (1 - 0.25) * (1 + 0.7778) = P →
  x = 25 := by
sorry

end jacket_price_reduction_l1627_162753


namespace julio_has_seven_grape_bottles_l1627_162712

-- Define the number of bottles and liters
def julio_orange_bottles : ℕ := 4
def mateo_orange_bottles : ℕ := 1
def mateo_grape_bottles : ℕ := 3
def liters_per_bottle : ℕ := 2
def julio_extra_liters : ℕ := 14

-- Define the function to calculate the number of grape bottles Julio has
def julio_grape_bottles : ℕ :=
  let mateo_total_liters := (mateo_orange_bottles + mateo_grape_bottles) * liters_per_bottle
  let julio_total_liters := mateo_total_liters + julio_extra_liters
  let julio_grape_liters := julio_total_liters - (julio_orange_bottles * liters_per_bottle)
  julio_grape_liters / liters_per_bottle

-- State the theorem
theorem julio_has_seven_grape_bottles :
  julio_grape_bottles = 7 := by sorry

end julio_has_seven_grape_bottles_l1627_162712


namespace predecessor_in_binary_l1627_162713

def binary_to_nat (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

theorem predecessor_in_binary :
  let Q : List Bool := [true, true, false, true, false, true, false]
  let Q_nat : Nat := binary_to_nat Q
  let pred_Q : List Bool := nat_to_binary (Q_nat - 1)
  pred_Q = [true, true, false, true, false, false, true] := by
  sorry

end predecessor_in_binary_l1627_162713


namespace sum_of_integers_l1627_162759

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 7)
  (eq2 : q - r + s = 8)
  (eq3 : r - s + p = 4)
  (eq4 : s - p + q = 1) :
  p + q + r + s = 20 := by sorry

end sum_of_integers_l1627_162759


namespace geometric_arithmetic_progression_ratio_l1627_162745

/-- Given a decreasing geometric progression a, b, c with common ratio q,
    if 19a, 124b/13, c/13 form an arithmetic progression, then q = 247. -/
theorem geometric_arithmetic_progression_ratio 
  (a b c : ℝ) (q : ℝ) (h_pos : a > 0) (h_decr : q > 1) :
  b = a * q ∧ c = a * q^2 ∧ 
  2 * (124 * b / 13) = 19 * a + c / 13 →
  q = 247 := by
sorry


end geometric_arithmetic_progression_ratio_l1627_162745


namespace compound_interest_10_years_l1627_162773

/-- Calculates the total amount of principal and interest after a given number of years
    with compound interest. -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem stating that the total amount after 10 years of compound interest
    is equal to the initial principal multiplied by (1 + rate) raised to the power of 10. -/
theorem compound_interest_10_years
  (a : ℝ) -- initial deposit
  (r : ℝ) -- annual interest rate
  (h1 : a > 0) -- assumption that initial deposit is positive
  (h2 : r > 0) -- assumption that interest rate is positive
  : compoundInterest a r 10 = a * (1 + r)^10 := by
  sorry

end compound_interest_10_years_l1627_162773


namespace two_p_plus_q_l1627_162779

theorem two_p_plus_q (p q : ℚ) (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 := by
  sorry

end two_p_plus_q_l1627_162779


namespace profit_maximization_l1627_162730

noncomputable def y (x : ℝ) : ℝ := 20 * x - 3 * x^2 + 96 * Real.log x - 90

theorem profit_maximization :
  ∃ (x_max : ℝ), 
    (4 ≤ x_max ∧ x_max ≤ 12) ∧
    (∀ x, 4 ≤ x ∧ x ≤ 12 → y x ≤ y x_max) ∧
    x_max = 6 ∧
    y x_max = 96 * Real.log 6 - 78 :=
sorry

end profit_maximization_l1627_162730


namespace right_trapezoid_diagonals_bases_squares_diff_l1627_162771

/-- A right trapezoid with given properties -/
structure RightTrapezoid where
  b₁ : ℝ  -- length of smaller base BC
  b₂ : ℝ  -- length of larger base AD
  h : ℝ   -- height (length of legs AB and CD)
  h_pos : h > 0
  b₁_pos : b₁ > 0
  b₂_pos : b₂ > 0
  b₁_lt_b₂ : b₁ < b₂

/-- The theorem stating that the difference of squares of diagonals equals
    the difference of squares of bases in a right trapezoid -/
theorem right_trapezoid_diagonals_bases_squares_diff
  (t : RightTrapezoid) :
  (t.h^2 + t.b₂^2) - (t.h^2 + t.b₁^2) = t.b₂^2 - t.b₁^2 := by
  sorry

end right_trapezoid_diagonals_bases_squares_diff_l1627_162771


namespace not_consecutive_numbers_l1627_162772

theorem not_consecutive_numbers (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  ¬∃ (k : ℕ), ({2023 + a - b, 2023 + b - c, 2023 + c - a} : Finset ℕ) = {k - 1, k, k + 1} :=
by sorry

end not_consecutive_numbers_l1627_162772


namespace fraction_between_main_theorem_l1627_162710

theorem fraction_between (a b c d m n : ℕ) (h1 : 0 < b) (h2 : 0 < d) (h3 : 0 < n) :
  a * d < c * b → c * n < m * d → a * n < m * b →
  (a : ℚ) / b < (m : ℚ) / n ∧ (m : ℚ) / n < (c : ℚ) / d :=
by sorry

theorem main_theorem :
  (5 : ℚ) / 14 < (8 : ℚ) / 21 ∧ (8 : ℚ) / 21 < (5 : ℚ) / 12 :=
by sorry

end fraction_between_main_theorem_l1627_162710


namespace time_to_school_gate_l1627_162711

/-- Proves that the time to arrive at the school gate is 15 minutes -/
theorem time_to_school_gate 
  (total_time : ℕ) 
  (gate_to_building : ℕ) 
  (building_to_room : ℕ) 
  (h1 : total_time = 30) 
  (h2 : gate_to_building = 6) 
  (h3 : building_to_room = 9) : 
  total_time - gate_to_building - building_to_room = 15 := by
sorry

end time_to_school_gate_l1627_162711


namespace calculate_monthly_income_l1627_162719

/-- Calculates the total monthly income given the specified distributions and remaining amount. -/
theorem calculate_monthly_income (children_percentage : Real) (investment_percentage : Real)
  (tax_percentage : Real) (fixed_expenses : Real) (donation_percentage : Real)
  (remaining_amount : Real) :
  let total_income := (remaining_amount + fixed_expenses) /
    (1 - 3 * children_percentage - investment_percentage - tax_percentage -
     donation_percentage * (1 - 3 * children_percentage - investment_percentage - tax_percentage))
  (total_income - 3 * (children_percentage * total_income) -
   investment_percentage * total_income - tax_percentage * total_income - fixed_expenses -
   donation_percentage * (total_income - 3 * (children_percentage * total_income) -
   investment_percentage * total_income - tax_percentage * total_income - fixed_expenses)) =
  remaining_amount :=
by sorry

end calculate_monthly_income_l1627_162719


namespace tan_theta_right_triangle_l1627_162788

theorem tan_theta_right_triangle (BC AC BA : ℝ) (h1 : BC = 25) (h2 : AC = 20) 
  (h3 : BA^2 + AC^2 = BC^2) : 
  Real.tan (Real.arcsin (BA / BC)) = 3 / 4 := by
  sorry

end tan_theta_right_triangle_l1627_162788


namespace theater_group_arrangement_l1627_162791

theorem theater_group_arrangement (n : ℕ) : n ≥ 1981 ∧ 
  n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n % 12 = 1 → n = 1981 :=
by sorry

end theater_group_arrangement_l1627_162791


namespace f_image_is_closed_interval_l1627_162714

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x + 7

-- Define the domain
def domain : Set ℝ := Set.Ioc 2 5

-- Theorem statement
theorem f_image_is_closed_interval :
  Set.image f domain = Set.Icc (-2) 2 := by sorry

end f_image_is_closed_interval_l1627_162714


namespace rotation_dilation_determinant_l1627_162705

theorem rotation_dilation_determinant :
  ∀ (E : Matrix (Fin 2) (Fin 2) ℝ),
  (∃ (R S : Matrix (Fin 2) (Fin 2) ℝ),
    R = !![0, -1; 1, 0] ∧
    S = !![5, 0; 0, 5] ∧
    E = S * R) →
  Matrix.det E = 25 := by
sorry

end rotation_dilation_determinant_l1627_162705


namespace division_problem_l1627_162784

theorem division_problem (x : ℝ) : 75 / x = 1500 → x = 0.05 := by
  sorry

end division_problem_l1627_162784


namespace difference_of_roots_absolute_value_l1627_162727

theorem difference_of_roots_absolute_value (a b c : ℝ) (h : a ≠ 0) :
  let r₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 1 ∧ b = -7 ∧ c = 10 → |r₁ - r₂| = 3 :=
by sorry

end difference_of_roots_absolute_value_l1627_162727


namespace investments_sum_to_22000_l1627_162715

/-- Represents the initial investment amounts of five individuals --/
structure Investments where
  raghu : ℝ
  trishul : ℝ
  vishal : ℝ
  alok : ℝ
  harshit : ℝ

/-- Calculates the total sum of investments --/
def total_investment (i : Investments) : ℝ :=
  i.raghu + i.trishul + i.vishal + i.alok + i.harshit

/-- Theorem stating that the investments satisfy the given conditions and sum to 22000 --/
theorem investments_sum_to_22000 :
  ∃ (i : Investments),
    i.trishul = 0.9 * i.raghu ∧
    i.vishal = 1.1 * i.trishul ∧
    i.alok = 1.15 * i.trishul ∧
    i.harshit = 0.95 * i.vishal ∧
    total_investment i = 22000 :=
  sorry

end investments_sum_to_22000_l1627_162715


namespace ngo_employee_count_l1627_162709

/-- The number of illiterate employees -/
def illiterate_employees : ℕ := 20

/-- The decrease in total wages of illiterate employees in Rupees -/
def total_wage_decrease : ℕ := 300

/-- The decrease in average salary for all employees in Rupees -/
def average_salary_decrease : ℕ := 10

/-- The number of educated employees in the NGO -/
def educated_employees : ℕ := 10

theorem ngo_employee_count :
  educated_employees = total_wage_decrease / average_salary_decrease - illiterate_employees :=
by sorry

end ngo_employee_count_l1627_162709


namespace parabola_directrix_l1627_162760

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -2

-- Theorem statement
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola x y → directrix y :=
sorry

end parabola_directrix_l1627_162760


namespace max_production_in_seven_days_l1627_162790

/-- Represents the daily production capacity of a group -/
structure ProductionCapacity where
  shirts : ℕ
  trousers : ℕ

/-- Represents the production assignment for a group -/
structure ProductionAssignment where
  shirtDays : ℕ
  trouserDays : ℕ

/-- Calculates the total production of a group given its capacity and assignment -/
def totalProduction (capacity : ProductionCapacity) (assignment : ProductionAssignment) : ℕ × ℕ :=
  (capacity.shirts * assignment.shirtDays, capacity.trousers * assignment.trouserDays)

/-- Theorem: Maximum production of matching sets in 7 days -/
theorem max_production_in_seven_days 
  (groupA groupB groupC groupD : ProductionCapacity)
  (assignmentA assignmentB assignmentC assignmentD : ProductionAssignment)
  (h1 : assignmentA.shirtDays + assignmentA.trouserDays = 7)
  (h2 : assignmentB.shirtDays + assignmentB.trouserDays = 7)
  (h3 : assignmentC.shirtDays + assignmentC.trouserDays = 7)
  (h4 : assignmentD.shirtDays + assignmentD.trouserDays = 7)
  (h5 : groupA.shirts = 8 ∧ groupA.trousers = 10)
  (h6 : groupB.shirts = 9 ∧ groupB.trousers = 12)
  (h7 : groupC.shirts = 7 ∧ groupC.trousers = 11)
  (h8 : groupD.shirts = 6 ∧ groupD.trousers = 7) :
  (∃ (assignmentA assignmentB assignmentC assignmentD : ProductionAssignment),
    let (shirtsTotalA, trousersTotalA) := totalProduction groupA assignmentA
    let (shirtsTotalB, trousersTotalB) := totalProduction groupB assignmentB
    let (shirtsTotalC, trousersTotalC) := totalProduction groupC assignmentC
    let (shirtsTotalD, trousersTotalD) := totalProduction groupD assignmentD
    let shirtsTotal := shirtsTotalA + shirtsTotalB + shirtsTotalC + shirtsTotalD
    let trousersTotal := trousersTotalA + trousersTotalB + trousersTotalC + trousersTotalD
    min shirtsTotal trousersTotal = 125) :=
by sorry

end max_production_in_seven_days_l1627_162790


namespace phase_without_chromatids_is_interkinesis_l1627_162751

-- Define the phases of meiosis
inductive MeiosisPhase
  | prophaseI
  | interkinesis
  | prophaseII
  | lateProphaseII

-- Define a property for the presence of chromatids
def hasChromatids (phase : MeiosisPhase) : Prop :=
  match phase with
  | MeiosisPhase.interkinesis => False
  | _ => True

-- Define a property for DNA replication
def hasDNAReplication (phase : MeiosisPhase) : Prop :=
  match phase with
  | MeiosisPhase.interkinesis => False
  | _ => True

-- Theorem statement
theorem phase_without_chromatids_is_interkinesis :
  ∀ phase : MeiosisPhase, ¬(hasChromatids phase) → phase = MeiosisPhase.interkinesis :=
by
  sorry

end phase_without_chromatids_is_interkinesis_l1627_162751


namespace constant_term_product_l1627_162724

-- Define polynomials p, q, and r
variable (p q r : ℝ → ℝ)

-- State the theorem
theorem constant_term_product (h1 : ∀ x, r x = p x * q x) 
                               (h2 : p 0 = 5) 
                               (h3 : r 0 = -10) : 
  q 0 = -2 := by
  sorry

end constant_term_product_l1627_162724


namespace sum_of_two_numbers_l1627_162731

theorem sum_of_two_numbers (x y : ℝ) (h1 : y = 4 * x) (h2 : x + y = 45) : x + y = 45 := by
  sorry

end sum_of_two_numbers_l1627_162731


namespace area_of_triangle_formed_by_tangents_l1627_162780

/-- Given two circles with radii R and r, where their common internal tangents
    are perpendicular to each other, the area of the triangle formed by these
    tangents and the common external tangent is equal to R * r. -/
theorem area_of_triangle_formed_by_tangents (R r : ℝ) (R_pos : R > 0) (r_pos : r > 0) :
  ∃ (S : ℝ), S = R * r ∧ S > 0 :=
by sorry

end area_of_triangle_formed_by_tangents_l1627_162780


namespace train_length_l1627_162782

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 24 → ∃ length : ℝ, abs (length - 199.92) < 0.01 := by
  sorry

#check train_length

end train_length_l1627_162782


namespace no_real_solution_log_equation_l1627_162792

theorem no_real_solution_log_equation :
  ¬∃ (x : ℝ), (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 5*x + 6)) ∧
              (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 5*x + 6 > 0) := by
  sorry

end no_real_solution_log_equation_l1627_162792


namespace shaded_area_in_divided_square_l1627_162762

/-- The area of shaded regions in a square with specific divisions -/
theorem shaded_area_in_divided_square (side_length : ℝ) (h_side : side_length = 4) :
  let square_area := side_length ^ 2
  let num_rectangles := 4
  let num_triangles_per_rectangle := 2
  let num_shaded_triangles := num_rectangles
  let rectangle_area := square_area / num_rectangles
  let triangle_area := rectangle_area / num_triangles_per_rectangle
  let total_shaded_area := num_shaded_triangles * triangle_area
  total_shaded_area = 8 := by
  sorry

end shaded_area_in_divided_square_l1627_162762


namespace sum_of_ages_is_fifty_l1627_162757

/-- The sum of ages of 5 children born at intervals of 3 years, with the youngest being 4 years old -/
def sum_of_ages : ℕ :=
  let youngest_age := 4
  let interval := 3
  let num_children := 5
  List.range num_children
    |>.map (fun i => youngest_age + i * interval)
    |>.sum

theorem sum_of_ages_is_fifty :
  sum_of_ages = 50 := by
  sorry

end sum_of_ages_is_fifty_l1627_162757


namespace perpendicular_planes_l1627_162700

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (m n : Line) (α β : Plane) 
  (h1 : perp_line_plane m α) 
  (h2 : perp_line_plane n β) 
  (h3 : perp_line_line m n) : 
  perp_plane_plane α β :=
sorry

end perpendicular_planes_l1627_162700


namespace min_max_abs_cubic_linear_l1627_162776

theorem min_max_abs_cubic_linear (y : ℝ) : 
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ |x^3 - x*y| ≥ 1) ∧
  (∃ (y₀ : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 0 1 → |x^3 - x*y₀| ≤ 1) := by
  sorry

end min_max_abs_cubic_linear_l1627_162776


namespace unique_prime_solution_l1627_162739

theorem unique_prime_solution :
  ∀ p q r : ℕ,
    Prime p ∧ Prime q ∧ Prime r →
    p + q^2 = r^4 →
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end unique_prime_solution_l1627_162739


namespace quadratic_root_property_l1627_162737

theorem quadratic_root_property (m : ℝ) : 
  m^2 + m - 1 = 0 → 2*m^2 + 2*m + 2025 = 2027 := by
  sorry

end quadratic_root_property_l1627_162737


namespace value_of_a_minus_2b_l1627_162775

theorem value_of_a_minus_2b (a b : ℝ) (h : |a + b + 2| + |b - 3| = 0) : a - 2*b = -11 := by
  sorry

end value_of_a_minus_2b_l1627_162775


namespace triangle_area_arithmetic_angles_l1627_162752

/-- Given a triangle ABC with sides a and c, and angles A, B, C forming an arithmetic sequence,
    prove that its area is 3√3 when a = 4 and c = 3. -/
theorem triangle_area_arithmetic_angles (A B C : ℝ) (a c : ℝ) :
  -- Angles form an arithmetic sequence
  ∃ d : ℝ, A = B - d ∧ C = B + d
  -- Sum of angles in a triangle is π (180°)
  → A + B + C = π
  -- Given side lengths
  → a = 4
  → c = 3
  -- Area of the triangle
  → (1/2) * a * c * Real.sin B = 3 * Real.sqrt 3 :=
by sorry

end triangle_area_arithmetic_angles_l1627_162752


namespace largest_quantity_l1627_162787

def X : ℚ := 2010 / 2009 + 2010 / 2011
def Y : ℚ := 2010 / 2011 + 2012 / 2011
def Z : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : X > Y ∧ X > Z := by
  sorry

end largest_quantity_l1627_162787


namespace sum_and_product_implications_l1627_162789

theorem sum_and_product_implications (a b : ℝ) 
  (h1 : a + b = 2) (h2 : a * b = -1) : 
  a^2 + b^2 = 6 ∧ (a - b)^2 = 8 := by
  sorry

end sum_and_product_implications_l1627_162789


namespace box_velvet_problem_l1627_162774

theorem box_velvet_problem (long_side_length long_side_width short_side_length short_side_width total_velvet : ℕ) 
  (h1 : long_side_length = 8)
  (h2 : long_side_width = 6)
  (h3 : short_side_length = 5)
  (h4 : short_side_width = 6)
  (h5 : total_velvet = 236) :
  let side_area := 2 * (long_side_length * long_side_width + short_side_length * short_side_width)
  let remaining_area := total_velvet - side_area
  (remaining_area / 2 : ℕ) = 40 := by
  sorry

end box_velvet_problem_l1627_162774


namespace loan_amount_l1627_162721

/-- Proves that given the conditions of the loan, the sum lent must be 500 Rs. -/
theorem loan_amount (interest_rate : ℚ) (time : ℕ) (interest_difference : ℚ) : 
  interest_rate = 4/100 →
  time = 8 →
  interest_difference = 340 →
  ∃ (principal : ℚ), 
    principal * interest_rate * time = principal - interest_difference ∧
    principal = 500 := by
  sorry

end loan_amount_l1627_162721


namespace tv_purchase_time_l1627_162748

/-- Calculates the number of months required to save for a television purchase. -/
def months_to_purchase_tv (monthly_income : ℕ) (food_expense : ℕ) (utilities_expense : ℕ) 
  (other_expenses : ℕ) (current_savings : ℕ) (tv_cost : ℕ) : ℕ :=
  let total_expenses := food_expense + utilities_expense + other_expenses
  let monthly_savings := monthly_income - total_expenses
  let additional_savings_needed := tv_cost - current_savings
  (additional_savings_needed + monthly_savings - 1) / monthly_savings

theorem tv_purchase_time :
  months_to_purchase_tv 30000 15000 5000 2500 10000 25000 = 2 := by
  sorry

end tv_purchase_time_l1627_162748


namespace tim_gave_six_kittens_to_sara_l1627_162766

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara (initial_kittens : ℕ) (kittens_to_jessica : ℕ) (remaining_kittens : ℕ) : ℕ :=
  initial_kittens - kittens_to_jessica - remaining_kittens

/-- Proof that Tim gave 6 kittens to Sara -/
theorem tim_gave_six_kittens_to_sara :
  kittens_to_sara 18 3 9 = 6 := by
  sorry

end tim_gave_six_kittens_to_sara_l1627_162766


namespace two_number_difference_l1627_162764

theorem two_number_difference (x y : ℝ) : 
  x + y = 40 → 3 * y - 4 * x = 10 → |y - x| = 60 / 7 := by
  sorry

end two_number_difference_l1627_162764


namespace quadratic_inequality_solution_l1627_162795

theorem quadratic_inequality_solution (b c : ℝ) :
  (∀ x, x^2 + b*x + c < 0 ↔ 1 < x ∧ x < 2) →
  (∀ x, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = 2) →
  b + c = -1 := by
sorry

end quadratic_inequality_solution_l1627_162795


namespace max_value_x3y2z_l1627_162706

theorem max_value_x3y2z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  x^3 * y^2 * z ≤ 1/432 := by
sorry

end max_value_x3y2z_l1627_162706


namespace two_digit_number_equation_l1627_162729

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≥ 0 ∧ units ≤ 9

/-- The property that the unit digit is 3 greater than the tens digit -/
def unit_is_three_greater (n : TwoDigitNumber) : Prop :=
  n.units = n.tens + 3

/-- The property that the square of the unit digit equals the two-digit number -/
def square_of_unit_is_number (n : TwoDigitNumber) : Prop :=
  n.units ^ 2 = 10 * n.tens + n.units

/-- Theorem: For a two-digit number satisfying the given conditions, 
    the tens digit x satisfies the equation x^2 - 5x + 6 = 0 -/
theorem two_digit_number_equation (n : TwoDigitNumber) 
  (h1 : unit_is_three_greater n) 
  (h2 : square_of_unit_is_number n) : 
  n.tens ^ 2 - 5 * n.tens + 6 = 0 := by
  sorry

end two_digit_number_equation_l1627_162729


namespace geometric_sequence_nth_term_l1627_162793

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_nth_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_sum : a 2 + a 5 = 18)
  (h_prod : a 3 * a 4 = 32)
  (h_nth : ∃ (n : ℕ), a n = 128) :
  ∃ (n : ℕ), a n = 128 ∧ n = 8 := by
sorry

end geometric_sequence_nth_term_l1627_162793


namespace min_value_of_objective_function_l1627_162716

-- Define the constraint region
def ConstraintRegion (x y : ℝ) : Prop :=
  2 * x - y ≥ 0 ∧ y ≥ x ∧ y ≥ -x + 2

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ := 2 * x + y

-- Theorem statement
theorem min_value_of_objective_function :
  ∃ (min_z : ℝ), min_z = 8/3 ∧
  (∀ (x y : ℝ), ConstraintRegion x y → ObjectiveFunction x y ≥ min_z) ∧
  (∃ (x y : ℝ), ConstraintRegion x y ∧ ObjectiveFunction x y = min_z) :=
sorry

end min_value_of_objective_function_l1627_162716


namespace value_of_expression_l1627_162701

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem value_of_expression (a b c d : ℝ) :
  f a b c d (-2) = -3 →
  8*a - 4*b + 2*c - d = 3 := by
  sorry

end value_of_expression_l1627_162701
