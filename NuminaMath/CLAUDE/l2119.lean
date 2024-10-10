import Mathlib

namespace adoption_fee_is_50_l2119_211998

/-- Represents the adoption fee for the cat -/
def adoption_fee : ℝ := sorry

/-- Represents the total vet visit costs for the first year -/
def vet_costs : ℝ := 500

/-- Represents the monthly food cost -/
def monthly_food_cost : ℝ := 25

/-- Represents the cost of toys Jenny bought -/
def jenny_toy_costs : ℝ := 200

/-- Represents Jenny's total spending on the cat in the first year -/
def jenny_total_spending : ℝ := 625

/-- Theorem stating that the adoption fee is $50 -/
theorem adoption_fee_is_50 : adoption_fee = 50 := by
  sorry

end adoption_fee_is_50_l2119_211998


namespace magnitude_e1_minus_sqrt3_e2_l2119_211920

-- Define the vector space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- State the theorem
theorem magnitude_e1_minus_sqrt3_e2 
  (h1 : ‖e₁‖ = 1) 
  (h2 : ‖e₂‖ = 1) 
  (h3 : inner e₁ e₂ = Real.sqrt 3 / 2) : 
  ‖e₁ - Real.sqrt 3 • e₂‖ = 1 := by sorry

end magnitude_e1_minus_sqrt3_e2_l2119_211920


namespace building_population_l2119_211922

/-- Calculates the total number of people housed in a building -/
def total_people (stories : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  stories * apartments_per_floor * people_per_apartment

/-- Theorem: A 25-story building with 4 apartments per floor and 2 people per apartment houses 200 people -/
theorem building_population : total_people 25 4 2 = 200 := by
  sorry

end building_population_l2119_211922


namespace power_sum_equality_two_variables_power_sum_equality_three_variables_l2119_211987

-- Part (a)
theorem power_sum_equality_two_variables (x y u v : ℝ) (h1 : x + y = u + v) (h2 : x^2 + y^2 = u^2 + v^2) :
  ∀ n : ℕ, x^n + y^n = u^n + v^n := by sorry

-- Part (b)
theorem power_sum_equality_three_variables (x y z u v t : ℝ) 
  (h1 : x + y + z = u + v + t) 
  (h2 : x^2 + y^2 + z^2 = u^2 + v^2 + t^2) 
  (h3 : x^3 + y^3 + z^3 = u^3 + v^3 + t^3) :
  ∀ n : ℕ, x^n + y^n + z^n = u^n + v^n + t^n := by sorry

end power_sum_equality_two_variables_power_sum_equality_three_variables_l2119_211987


namespace painted_unit_cubes_in_3x3x3_l2119_211924

/-- Represents a 3D cube -/
structure Cube :=
  (size : ℕ)

/-- Represents a painted cube -/
def PaintedCube := Cube

/-- Represents a unit cube (1x1x1) -/
def UnitCube := Cube

/-- The number of unit cubes with at least one painted surface in a painted cube -/
def num_painted_unit_cubes (c : PaintedCube) : ℕ :=
  sorry

/-- The main theorem: In a 3x3x3 painted cube, 26 unit cubes have at least one painted surface -/
theorem painted_unit_cubes_in_3x3x3 (c : PaintedCube) (h : c.size = 3) :
  num_painted_unit_cubes c = 26 :=
sorry

end painted_unit_cubes_in_3x3x3_l2119_211924


namespace ellipse_max_sum_l2119_211911

/-- Given an ellipse defined by x^2/4 + y^2/2 = 1, 
    the maximum value of |x| + |y| is 2√3. -/
theorem ellipse_max_sum (x y : ℝ) : 
  x^2/4 + y^2/2 = 1 → |x| + |y| ≤ 2 * Real.sqrt 3 := by
  sorry

end ellipse_max_sum_l2119_211911


namespace polynomial_remainder_l2119_211941

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 3*x + 5) % (x - 1) = 3 := by sorry

end polynomial_remainder_l2119_211941


namespace constant_odd_function_is_zero_l2119_211993

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem constant_odd_function_is_zero (k : ℝ) (h : IsOdd (fun x ↦ k)) : k = 0 := by
  sorry

end constant_odd_function_is_zero_l2119_211993


namespace a_ratio_l2119_211951

def a (n : ℕ) : ℚ := 3 - 2^n

theorem a_ratio : a 2 / a 3 = 1 / 5 := by
  sorry

end a_ratio_l2119_211951


namespace members_playing_neither_in_given_club_l2119_211958

/-- Represents a music club with members playing different instruments -/
structure MusicClub where
  total : ℕ
  guitar : ℕ
  piano : ℕ
  both : ℕ

/-- Calculates the number of members who don't play either instrument -/
def membersPlayingNeither (club : MusicClub) : ℕ :=
  club.total - (club.guitar + club.piano - club.both)

/-- Theorem stating the number of members not playing either instrument in the given club -/
theorem members_playing_neither_in_given_club :
  let club : MusicClub := {
    total := 80,
    guitar := 50,
    piano := 40,
    both := 25
  }
  membersPlayingNeither club = 15 := by
  sorry

end members_playing_neither_in_given_club_l2119_211958


namespace n_sided_polygon_exterior_angle_l2119_211975

theorem n_sided_polygon_exterior_angle (n : ℕ) : 
  (n ≠ 0) → (40 * n = 360) → n = 9 := by
  sorry

end n_sided_polygon_exterior_angle_l2119_211975


namespace investment_sum_l2119_211931

/-- Given a sum invested at simple interest for two years, 
    if the difference in interest between 15% p.a. and 12% p.a. is 420, 
    then the sum invested is 7000. -/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 420) → P = 7000 := by
  sorry

end investment_sum_l2119_211931


namespace right_triangle_area_l2119_211990

theorem right_triangle_area (a b c : ℝ) (h1 : a = 40) (h2 : c = 41) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 180 := by
sorry

end right_triangle_area_l2119_211990


namespace expression_simplification_l2119_211926

theorem expression_simplification 
  (b c d x y : ℝ) (h : cx + dy ≠ 0) :
  (c * x * (c^2 * x^2 + 3 * b^2 * y^2 + c^2 * y^2) + 
   d * y * (b^2 * x^2 + 3 * c^2 * x^2 + b^2 * y^2)) / 
  (c * x + d * y) = 
  c^2 * x^2 + d * b^2 * y^2 := by
sorry

end expression_simplification_l2119_211926


namespace yoojeong_rabbits_l2119_211986

/-- The number of animals Minyoung has -/
def minyoung_animals : ℕ := 9 + 3 + 5

/-- The number of animals Yoojeong has -/
def yoojeong_animals : ℕ := minyoung_animals + 2

/-- The number of dogs Yoojeong has -/
def yoojeong_dogs : ℕ := 7

theorem yoojeong_rabbits :
  ∃ (cats rabbits : ℕ),
    yoojeong_animals = yoojeong_dogs + cats + rabbits ∧
    cats = rabbits - 2 ∧
    rabbits = 7 := by sorry

end yoojeong_rabbits_l2119_211986


namespace meg_cat_weight_l2119_211904

theorem meg_cat_weight (meg_weight anne_weight : ℝ) 
  (h1 : meg_weight / anne_weight = 5 / 7)
  (h2 : anne_weight = meg_weight + 8) : 
  meg_weight = 20 := by
sorry

end meg_cat_weight_l2119_211904


namespace profit_difference_theorem_l2119_211988

/-- Represents the profit distribution for a business partnership --/
structure ProfitDistribution where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  b_profit : ℕ

/-- Calculates the difference between profit shares of A and C --/
def profit_difference (pd : ProfitDistribution) : ℕ :=
  let total_ratio := pd.a_investment + pd.b_investment + pd.c_investment
  let unit_profit := pd.b_profit * total_ratio / pd.b_investment
  let a_profit := unit_profit * pd.a_investment / total_ratio
  let c_profit := unit_profit * pd.c_investment / total_ratio
  c_profit - a_profit

/-- Theorem stating the difference in profit shares --/
theorem profit_difference_theorem (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 8000)
  (h2 : pd.b_investment = 10000)
  (h3 : pd.c_investment = 12000)
  (h4 : pd.b_profit = 3000) :
  profit_difference pd = 1200 := by
  sorry

#eval profit_difference ⟨8000, 10000, 12000, 3000⟩

end profit_difference_theorem_l2119_211988


namespace max_chord_line_l2119_211919

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The circle C: x^2 + y^2 + 4x + 3 = 0 -/
def C : Circle := { center := (-2, 0), radius := 1 }

/-- The point through which line l passes -/
def P : ℝ × ℝ := (2, 3)

/-- Function to check if a line passes through a point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Function to check if a line intersects a circle at two points -/
def intersects_circle (l : Line) (c : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧ 
    passes_through l p ∧ passes_through l q ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2

/-- Function to check if the chord formed by the intersection of a line and circle is maximized -/
def maximizes_chord (l : Line) (c : Circle) : Prop :=
  passes_through l c.center

/-- The theorem to be proved -/
theorem max_chord_line : 
  ∃ (l : Line), 
    passes_through l P ∧ 
    intersects_circle l C ∧ 
    maximizes_chord l C ∧ 
    l = { a := 3, b := -4, c := 6 } := by sorry

end max_chord_line_l2119_211919


namespace hundred_digit_number_theorem_l2119_211907

def is_valid_number (N : ℕ) : Prop :=
  ∃ (b : ℕ), b ∈ ({1, 2, 3} : Set ℕ) ∧ N = 325 * b * (10 ^ 97)

theorem hundred_digit_number_theorem (N : ℕ) :
  (∃ (k : ℕ) (a : ℕ), 
    a ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    N ≥ 10^99 ∧ N < 10^100 ∧
    ∃ (N' : ℕ), (N' = N - a * 10^k ∨ (k = 99 ∧ N' = N - a * 10^99)) ∧ N = 13 * N') →
  is_valid_number N :=
sorry

end hundred_digit_number_theorem_l2119_211907


namespace andy_ball_count_l2119_211950

theorem andy_ball_count : ∃ (a r m : ℕ), 
  (a = 2 * r) ∧ 
  (a = m + 5) ∧ 
  (a + r + m = 35) → 
  a = 16 := by
  sorry

end andy_ball_count_l2119_211950


namespace marcus_and_leah_games_l2119_211997

/-- The number of players in the league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The function to calculate the number of games where two specific players play together -/
def games_together (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n - 2) (k - 2)

/-- The theorem stating that Marcus and Leah play together in 210 games -/
theorem marcus_and_leah_games : 
  games_together total_players players_per_game = 210 := by
  sorry

#eval games_together total_players players_per_game

end marcus_and_leah_games_l2119_211997


namespace all_expressions_zero_l2119_211910

-- Define a 2D vector type
def Vector2D := ℝ × ℝ

-- Define vector addition
def add_vectors (v1 v2 : Vector2D) : Vector2D :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Define vector subtraction
def sub_vectors (v1 v2 : Vector2D) : Vector2D :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Define the zero vector
def zero_vector : Vector2D := (0, 0)

-- Define variables for each point
variable (A B C D E F O P Q : Vector2D)

-- Define the theorem
theorem all_expressions_zero : 
  (add_vectors (add_vectors (sub_vectors B A) (sub_vectors C B)) (sub_vectors A C) = zero_vector) ∧
  (add_vectors (sub_vectors (sub_vectors (sub_vectors B A) (sub_vectors C A)) (sub_vectors D B)) (sub_vectors D C) = zero_vector) ∧
  (sub_vectors (add_vectors (add_vectors (sub_vectors Q F) (sub_vectors P Q)) (sub_vectors F E)) (sub_vectors P E) = zero_vector) ∧
  (add_vectors (sub_vectors (sub_vectors A O) (sub_vectors B O)) (sub_vectors B A) = zero_vector) := by
  sorry

end all_expressions_zero_l2119_211910


namespace whatsapp_messages_theorem_l2119_211914

/-- The number of messages sent on Monday in a Whatsapp group -/
def monday_messages : ℕ := sorry

/-- The number of messages sent on Tuesday in a Whatsapp group -/
def tuesday_messages : ℕ := 200

/-- The number of messages sent on Wednesday in a Whatsapp group -/
def wednesday_messages : ℕ := tuesday_messages + 300

/-- The number of messages sent on Thursday in a Whatsapp group -/
def thursday_messages : ℕ := 2 * wednesday_messages

/-- The total number of messages sent over four days in a Whatsapp group -/
def total_messages : ℕ := 2000

theorem whatsapp_messages_theorem :
  monday_messages + tuesday_messages + wednesday_messages + thursday_messages = total_messages ∧
  monday_messages = 300 := by sorry

end whatsapp_messages_theorem_l2119_211914


namespace max_expression_value_l2119_211927

def is_valid_assignment (O L I M P A D : ℕ) : Prop :=
  O ≠ L ∧ O ≠ I ∧ O ≠ M ∧ O ≠ P ∧ O ≠ A ∧ O ≠ D ∧
  L ≠ I ∧ L ≠ M ∧ L ≠ P ∧ L ≠ A ∧ L ≠ D ∧
  I ≠ M ∧ I ≠ P ∧ I ≠ A ∧ I ≠ D ∧
  M ≠ P ∧ M ≠ A ∧ M ≠ D ∧
  P ≠ A ∧ P ≠ D ∧
  A ≠ D ∧
  O < 10 ∧ L < 10 ∧ I < 10 ∧ M < 10 ∧ P < 10 ∧ A < 10 ∧ D < 10 ∧
  O ≠ 0 ∧ I ≠ 0

def expression_value (O L I M P A D : ℕ) : ℤ :=
  (10 * O + L) + (10 * I + M) - P + (10 * I + A) - (10 * D + A)

theorem max_expression_value :
  ∀ O L I M P A D : ℕ,
    is_valid_assignment O L I M P A D →
    expression_value O L I M P A D ≤ 263 :=
sorry

end max_expression_value_l2119_211927


namespace arithmetic_sequence_sum_l2119_211900

theorem arithmetic_sequence_sum : 
  ∀ (a l d n : ℤ),
    a = -41 →
    l = 1 →
    d = 2 →
    n = 22 →
    a + (n - 1) * d = l →
    (n * (a + l)) / 2 = -440 :=
by
  sorry

end arithmetic_sequence_sum_l2119_211900


namespace quadratic_root_in_unit_interval_l2119_211960

theorem quadratic_root_in_unit_interval (a b c : ℝ) (h : 2*a + 3*b + 6*c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end quadratic_root_in_unit_interval_l2119_211960


namespace polynomial_factorization_l2119_211980

theorem polynomial_factorization (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end polynomial_factorization_l2119_211980


namespace modulus_of_complex_number_l2119_211944

theorem modulus_of_complex_number (z : ℂ) :
  z = Complex.mk (Real.sqrt 3 / 2) (-3 / 2) →
  Complex.abs z = Real.sqrt 3 := by
sorry

end modulus_of_complex_number_l2119_211944


namespace complex_fraction_equality_l2119_211902

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 3) :
  (x^4 + y^4) / (x^4 - y^4) - (x^4 - y^4) / (x^4 + y^4) = 49 / 600 := by
  sorry

end complex_fraction_equality_l2119_211902


namespace prism_volume_l2119_211973

theorem prism_volume (a b c : ℝ) (h1 : a * b = 18) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 150 * Real.sqrt 3 := by
  sorry

end prism_volume_l2119_211973


namespace comic_books_average_l2119_211940

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

theorem comic_books_average (a₁ : ℕ) (d : ℕ) (n : ℕ) 
  (h₁ : a₁ = 10) (h₂ : d = 6) (h₃ : n = 8) : 
  (arithmetic_sequence a₁ d n).sum / n = 31 := by
  sorry

end comic_books_average_l2119_211940


namespace inequality_proof_l2119_211996

theorem inequality_proof (a b c : ℝ) (m n k : ℕ+) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≥ (a^(m:ℝ) * b^(n:ℝ) * c^(k:ℝ))^(1/(m+n+k:ℝ)) + 
              (a^(n:ℝ) * b^(k:ℝ) * c^(m:ℝ))^(1/(m+n+k:ℝ)) + 
              (a^(k:ℝ) * b^(m:ℝ) * c^(n:ℝ))^(1/(m+n+k:ℝ)) :=
by sorry

end inequality_proof_l2119_211996


namespace good_number_theorem_l2119_211935

/-- A good number is a number of the form a + b√2 where a and b are integers -/
def GoodNumber (x : ℝ) : Prop :=
  ∃ (a b : ℤ), x = a + b * Real.sqrt 2

/-- Polynomial with good number coefficients -/
def GoodPolynomial (p : Polynomial ℝ) : Prop :=
  ∀ (i : ℕ), GoodNumber (p.coeff i)

theorem good_number_theorem (A B Q : Polynomial ℝ) 
  (hA : GoodPolynomial A)
  (hB : GoodPolynomial B)
  (hB0 : B.coeff 0 = 1)
  (hABQ : A = B * Q) :
  GoodPolynomial Q :=
sorry

end good_number_theorem_l2119_211935


namespace average_of_9_15_N_l2119_211932

theorem average_of_9_15_N (N : ℝ) (h : 12 < N ∧ N < 22) :
  let avg := (9 + 15 + N) / 3
  avg = 12 ∨ avg = 15 := by
sorry

end average_of_9_15_N_l2119_211932


namespace track_width_l2119_211901

/-- Given two concentric circles where the outer circle has a circumference of 40π feet
    and the difference between the outer and inner circle circumferences is 16π feet,
    prove that the difference between their radii is 8 feet. -/
theorem track_width (r₁ r₂ : ℝ) : 
  (2 * π * r₁ = 40 * π) →  -- Outer circle circumference
  (2 * π * r₁ - 2 * π * r₂ = 16 * π) →  -- Difference in circumferences
  r₁ - r₂ = 8 := by sorry

end track_width_l2119_211901


namespace p_sufficient_not_necessary_for_q_l2119_211969

-- Define the conditions
def p (x : ℝ) : Prop := (x - 2)^2 ≤ 1
def q (x : ℝ) : Prop := 2 / (x - 1) ≥ 1

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end p_sufficient_not_necessary_for_q_l2119_211969


namespace mark_buys_extra_large_bags_l2119_211954

/-- Represents the types of balloon bags available --/
inductive BagType
  | Small
  | Medium
  | ExtraLarge

/-- Represents a bag of balloons with its price and quantity --/
structure BalloonBag where
  bagType : BagType
  price : ℕ
  quantity : ℕ

def mark_budget : ℕ := 24
def small_bag : BalloonBag := ⟨BagType.Small, 4, 50⟩
def extra_large_bag : BalloonBag := ⟨BagType.ExtraLarge, 12, 200⟩
def total_balloons : ℕ := 400

/-- Calculates the number of bags that can be bought with a given budget --/
def bags_bought (bag : BalloonBag) (budget : ℕ) : ℕ :=
  budget / bag.price

/-- Calculates the total number of balloons from a given number of bags --/
def total_balloons_from_bags (bag : BalloonBag) (num_bags : ℕ) : ℕ :=
  num_bags * bag.quantity

theorem mark_buys_extra_large_bags :
  bags_bought extra_large_bag mark_budget = 2 ∧
  total_balloons_from_bags extra_large_bag (bags_bought extra_large_bag mark_budget) = total_balloons :=
by sorry

end mark_buys_extra_large_bags_l2119_211954


namespace large_monkey_doll_cost_l2119_211913

/-- The cost of a large monkey doll in dollars -/
def large_monkey_cost : ℝ := 6

/-- The total amount spent in dollars -/
def total_spent : ℝ := 300

/-- The number of additional dolls that can be bought if choosing small monkey dolls instead of large monkey dolls -/
def additional_small_dolls : ℕ := 25

/-- The number of fewer dolls that can be bought if choosing elephant dolls instead of large monkey dolls -/
def fewer_elephant_dolls : ℕ := 15

theorem large_monkey_doll_cost :
  (total_spent / (large_monkey_cost - 2) = total_spent / large_monkey_cost + additional_small_dolls) ∧
  (total_spent / (large_monkey_cost + 1) = total_spent / large_monkey_cost - fewer_elephant_dolls) := by
  sorry

end large_monkey_doll_cost_l2119_211913


namespace mercury_column_height_for_constant_center_of_gravity_l2119_211912

/-- Proves that the height of the mercury column for which the center of gravity
    remains at a constant distance from the top of the tube at any temperature
    is approximately 0.106 meters. -/
theorem mercury_column_height_for_constant_center_of_gravity
  (tube_length : ℝ)
  (cross_section_area : ℝ)
  (glass_expansion_coeff : ℝ)
  (mercury_expansion_coeff : ℝ)
  (h : tube_length = 1)
  (h₁ : cross_section_area = 1e-4)
  (h₂ : glass_expansion_coeff = 1 / 38700)
  (h₃ : mercury_expansion_coeff = 1 / 5550) :
  ∃ (height : ℝ), abs (height - 0.106) < 0.001 ∧
  ∀ (t : ℝ),
    (tube_length * (1 + glass_expansion_coeff / 3 * t) -
     height / 2 * (1 + (mercury_expansion_coeff - 2 * glass_expansion_coeff / 3) * t)) =
    (tube_length - height / 2) :=
sorry

end mercury_column_height_for_constant_center_of_gravity_l2119_211912


namespace g_increasing_intervals_l2119_211923

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

def g (x : ℝ) : ℝ := f (2 - x^2)

theorem g_increasing_intervals :
  ∃ (a b c : ℝ), a = -1 ∧ b = 0 ∧ c = 1 ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → g x ≤ g y) ∧
  (∀ x y, c ≤ x ∧ x < y → g x < g y) :=
sorry

end g_increasing_intervals_l2119_211923


namespace fixed_point_of_exponential_function_l2119_211966

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 1
  f 1 = 2 :=
by
  sorry

end fixed_point_of_exponential_function_l2119_211966


namespace centroid_locus_is_hyperbola_l2119_211965

/-- Given two complex points Z₁ and Z₂ with arguments θ and -θ respectively, 
    where 0 < θ < π/2, and the area of triangle OZ₁Z₂ is constant S, 
    prove that the locus of the centroid Z of triangle OZ₁Z₂ forms a hyperbola. -/
theorem centroid_locus_is_hyperbola 
  (θ : ℝ) 
  (h_θ_pos : 0 < θ) 
  (h_θ_lt_pi_half : θ < π/2) 
  (S : ℝ) 
  (h_S_pos : S > 0) 
  (Z₁ Z₂ : ℂ) 
  (h_Z₁_arg : Complex.arg Z₁ = θ) 
  (h_Z₂_arg : Complex.arg Z₂ = -θ) 
  (h_area : abs (Z₁.im * Z₂.re - Z₁.re * Z₂.im) / 2 = S) : 
  ∃ (a b : ℝ), ∀ (Z : ℂ), Z = (Z₁ + Z₂) / 3 → (Z.re / a)^2 - (Z.im / b)^2 = 1 :=
sorry

end centroid_locus_is_hyperbola_l2119_211965


namespace solve_strawberry_problem_l2119_211963

def strawberry_problem (christine_pounds rachel_pounds total_pies : ℕ) : Prop :=
  rachel_pounds = 2 * christine_pounds →
  christine_pounds = 10 →
  total_pies = 10 →
  (christine_pounds + rachel_pounds) / total_pies = 3

theorem solve_strawberry_problem :
  ∃ (christine_pounds rachel_pounds total_pies : ℕ),
    strawberry_problem christine_pounds rachel_pounds total_pies :=
by
  sorry

end solve_strawberry_problem_l2119_211963


namespace bowtie_problem_l2119_211971

-- Define the bowtie operation
noncomputable def bowtie (c d : ℝ) : ℝ := c + 1 + Real.sqrt (d + Real.sqrt (d + Real.sqrt d))

-- State the theorem
theorem bowtie_problem (h : ℝ) (hyp : bowtie 8 h = 12) : h = 6 := by
  sorry

end bowtie_problem_l2119_211971


namespace not_equivalent_fraction_l2119_211999

theorem not_equivalent_fraction (h : 0.000000275 = 2.75 * 10^(-7)) : 
  (11/40) * 10^(-7) ≠ 2.75 * 10^(-7) := by
  sorry

end not_equivalent_fraction_l2119_211999


namespace stock_decrease_duration_l2119_211925

/-- Represents the monthly decrease in bicycle stock -/
def monthly_decrease : ℕ := 4

/-- Represents the total decrease in bicycle stock from January 1 to October 1 -/
def total_decrease : ℕ := 36

/-- Represents the number of months from January 1 to October 1 -/
def total_months : ℕ := 9

/-- Represents the number of months the stock has been decreasing -/
def months_decreasing : ℕ := 5

theorem stock_decrease_duration :
  months_decreasing * monthly_decrease = total_decrease - (total_months - months_decreasing) * monthly_decrease :=
by sorry

end stock_decrease_duration_l2119_211925


namespace animal_video_ratio_l2119_211979

theorem animal_video_ratio :
  ∀ (total_time cat_time dog_time gorilla_time : ℝ),
    total_time = 36 →
    cat_time = 4 →
    gorilla_time = 2 * (cat_time + dog_time) →
    total_time = cat_time + dog_time + gorilla_time →
    dog_time / cat_time = 2 := by
  sorry

end animal_video_ratio_l2119_211979


namespace negation_of_existential_proposition_l2119_211982

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + x - 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x - 1 < 0) := by
  sorry

end negation_of_existential_proposition_l2119_211982


namespace max_value_expression_l2119_211947

theorem max_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y + y^2 = 9) :
  x^2 - 2*x*y + y^2 ≤ 9/4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + 2*a*b + b^2 = 9 ∧ a^2 - 2*a*b + b^2 = 9/4 :=
by sorry

end max_value_expression_l2119_211947


namespace space_division_cube_tetrahedron_l2119_211905

/-- The number of parts into which the space is divided by the facets of a polyhedron -/
def num_parts (V F E : ℕ) : ℕ := 1 + V + F + E

/-- Properties of a cube -/
def cube_vertices : ℕ := 8
def cube_edges : ℕ := 12
def cube_faces : ℕ := 6

/-- Properties of a tetrahedron -/
def tetrahedron_vertices : ℕ := 4
def tetrahedron_edges : ℕ := 6
def tetrahedron_faces : ℕ := 4

theorem space_division_cube_tetrahedron :
  (num_parts cube_vertices cube_faces cube_edges = 27) ∧
  (num_parts tetrahedron_vertices tetrahedron_faces tetrahedron_edges = 15) :=
by sorry

end space_division_cube_tetrahedron_l2119_211905


namespace min_blue_eyes_and_snack_bag_l2119_211989

theorem min_blue_eyes_and_snack_bag 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (snack_bag : ℕ) 
  (h1 : total_students = 35) 
  (h2 : blue_eyes = 14) 
  (h3 : snack_bag = 22) 
  (h4 : blue_eyes ≤ total_students) 
  (h5 : snack_bag ≤ total_students) : 
  ∃ (both : ℕ), both ≥ 1 ∧ 
    both ≤ blue_eyes ∧ 
    both ≤ snack_bag ∧ 
    (blue_eyes - both) + (snack_bag - both) ≤ total_students := by
  sorry

end min_blue_eyes_and_snack_bag_l2119_211989


namespace quadratic_function_uniqueness_l2119_211937

theorem quadratic_function_uniqueness (a b c : ℝ) :
  (∀ x : ℝ, a * (x - 4)^2 + b * (x - 4) + c = a * (2 - x)^2 + b * (2 - x) + c) →
  (∀ x : ℝ, a * x^2 + b * x + c ≥ x) →
  (∀ x : ℝ, x > 0 ∧ x < 2 → a * x^2 + b * x + c ≤ ((x + 1) / 2)^2) →
  (∃ x : ℝ, ∀ y : ℝ, a * x^2 + b * x + c ≤ a * y^2 + b * y + c) →
  (∃ x : ℝ, a * x^2 + b * x + c = 0) →
  (∀ x : ℝ, a * x^2 + b * x + c = (1/4) * (x + 1)^2) :=
sorry

end quadratic_function_uniqueness_l2119_211937


namespace earnings_difference_is_400_l2119_211981

/-- Represents the amount of jade Nancy has in grams -/
def total_jade : ℕ := 1920

/-- Represents the amount of jade needed for a giraffe statue in grams -/
def giraffe_jade : ℕ := 120

/-- Represents the price of a giraffe statue in dollars -/
def giraffe_price : ℕ := 150

/-- Represents the amount of jade needed for an elephant statue in grams -/
def elephant_jade : ℕ := 2 * giraffe_jade

/-- Represents the price of an elephant statue in dollars -/
def elephant_price : ℕ := 350

/-- Calculates the earnings difference between making all elephant statues
    and all giraffe statues from the total jade -/
def earnings_difference : ℕ :=
  (total_jade / elephant_jade) * elephant_price - (total_jade / giraffe_jade) * giraffe_price

theorem earnings_difference_is_400 : earnings_difference = 400 := by
  sorry

end earnings_difference_is_400_l2119_211981


namespace inverse_variation_problem_l2119_211977

theorem inverse_variation_problem (k : ℝ) (h : k > 0) :
  (∀ x y, x > 0 → y * Real.sqrt x = k) →
  (1/2 * Real.sqrt (1/4) = k) →
  (∃ x, x > 0 ∧ 8 * Real.sqrt x = k ∧ x = 1/1024) :=
by sorry

end inverse_variation_problem_l2119_211977


namespace orchids_cut_l2119_211917

theorem orchids_cut (initial_red : ℕ) (initial_white : ℕ) (final_red : ℕ) : 
  final_red - initial_red = final_red - initial_red :=
by
  sorry

#check orchids_cut 9 3 15

end orchids_cut_l2119_211917


namespace direct_proportion_constant_factor_l2119_211991

theorem direct_proportion_constant_factor 
  (k : ℝ) (x y : ℝ → ℝ) (t : ℝ) :
  (∀ t, y t = k * x t) → 
  (∀ t₁ t₂, t₁ ≠ t₂ → x t₁ ≠ x t₂ → y t₁ / x t₁ = y t₂ / x t₂) :=
by sorry

end direct_proportion_constant_factor_l2119_211991


namespace october_birth_percentage_l2119_211978

def total_people : ℕ := 100
def october_births : ℕ := 6

theorem october_birth_percentage :
  (october_births : ℚ) / total_people * 100 = 6 := by
  sorry

end october_birth_percentage_l2119_211978


namespace students_in_line_l2119_211945

/-- The number of students in a line, given specific positions of two students and the number of students between them. -/
theorem students_in_line
  (yoojung_position : ℕ)  -- Position of Yoojung
  (eunjung_position : ℕ)  -- Position of Eunjung from the back
  (students_between : ℕ)  -- Number of students between Yoojung and Eunjung
  (h1 : yoojung_position = 1)  -- Yoojung is at the front
  (h2 : eunjung_position = 5)  -- Eunjung is 5th from the back
  (h3 : students_between = 30)  -- 30 students between Yoojung and Eunjung
  : ℕ :=
by
  sorry

#check students_in_line

end students_in_line_l2119_211945


namespace doubling_points_properties_l2119_211938

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be a "doubling point" of another
def isDoublingPoint (p q : Point) : Prop :=
  2 * (p.x + q.x) = p.y + q.y

-- Given point P₁
def P₁ : Point := ⟨2, 0⟩

-- Theorem statement
theorem doubling_points_properties :
  -- 1. Q₁ and Q₂ are doubling points of P₁
  (isDoublingPoint P₁ ⟨2, 8⟩ ∧ isDoublingPoint P₁ ⟨-3, -2⟩) ∧
  -- 2. A(-2, 0) on y = x + 2 is a doubling point of P₁
  (∃ A : Point, A.y = A.x + 2 ∧ A.x = -2 ∧ A.y = 0 ∧ isDoublingPoint P₁ A) ∧
  -- 3. Two points on y = x² - 2x - 3 are doubling points of P₁
  (∃ B C : Point, B ≠ C ∧
    B.y = B.x^2 - 2*B.x - 3 ∧ C.y = C.x^2 - 2*C.x - 3 ∧
    isDoublingPoint P₁ B ∧ isDoublingPoint P₁ C) ∧
  -- 4. Minimum distance to any doubling point is 8√5/5
  (∃ minDist : ℝ, minDist = 8 * Real.sqrt 5 / 5 ∧
    ∀ Q : Point, isDoublingPoint P₁ Q →
      Real.sqrt ((Q.x - P₁.x)^2 + (Q.y - P₁.y)^2) ≥ minDist) :=
by sorry

end doubling_points_properties_l2119_211938


namespace sum_of_solutions_is_zero_l2119_211934

theorem sum_of_solutions_is_zero (y : ℝ) (h1 : y = 8) (h2 : ∃ x : ℝ, x^2 + y^2 = 225) :
  ∃ x₁ x₂ : ℝ, x₁^2 + y^2 = 225 ∧ x₂^2 + y^2 = 225 ∧ x₁ + x₂ = 0 :=
sorry

end sum_of_solutions_is_zero_l2119_211934


namespace complex_sum_equals_i_l2119_211962

theorem complex_sum_equals_i : Complex.I + 1 + Complex.I^2 = Complex.I := by sorry

end complex_sum_equals_i_l2119_211962


namespace complex_equidistant_points_l2119_211939

theorem complex_equidistant_points : ∃ (z : ℂ), 
  Complex.abs (z - 2) = 3 ∧ 
  Complex.abs (z + 1 + 2*I) = 3 ∧ 
  Complex.abs (z - 3*I) = 3 := by
  sorry

end complex_equidistant_points_l2119_211939


namespace container_volume_transformation_l2119_211961

/-- Represents a rectangular container with dimensions height, length, and width -/
structure Container where
  height : ℝ
  length : ℝ
  width : ℝ

/-- Calculates the volume of a container -/
def volume (c : Container) : ℝ := c.height * c.length * c.width

/-- Creates a new container by scaling the dimensions of an original container -/
def scaleContainer (c : Container) (h_scale l_scale w_scale : ℝ) : Container :=
  { height := c.height * h_scale,
    length := c.length * l_scale,
    width := c.width * w_scale }

theorem container_volume_transformation (original : Container) :
  volume original = 4 →
  volume (scaleContainer original 2 3 4) = 96 := by
  sorry

end container_volume_transformation_l2119_211961


namespace petya_final_vote_percentage_l2119_211915

theorem petya_final_vote_percentage 
  (x : ℝ) -- Total votes by noon
  (y : ℝ) -- Votes cast after noon
  (h1 : 0.45 * x = 0.27 * (x + y)) -- Vasya's final vote count
  (h2 : y = (2/3) * x) -- Relationship between x and y
  : (0.25 * x + y) / (x + y) = 0.55 := by
  sorry

end petya_final_vote_percentage_l2119_211915


namespace fathers_age_l2119_211995

theorem fathers_age (son_age father_age : ℕ) : 
  son_age = 10 →
  father_age = 4 * son_age →
  father_age + 20 = 2 * (son_age + 20) →
  father_age = 40 :=
by sorry

end fathers_age_l2119_211995


namespace parallelepiped_with_rectangular_opposite_faces_is_right_l2119_211906

/-- A parallelepiped is a three-dimensional figure with six faces, 
    where each pair of opposite faces are parallel parallelograms. -/
structure Parallelepiped

/-- A right parallelepiped is a parallelepiped where the lateral edges 
    are perpendicular to the base. -/
structure RightParallelepiped extends Parallelepiped

/-- A face of a parallelepiped -/
structure Face (P : Parallelepiped)

/-- Predicate to check if a face is rectangular -/
def is_rectangular (F : Face P) : Prop := sorry

/-- Predicate to check if two faces are opposite -/
def are_opposite (F1 F2 : Face P) : Prop := sorry

theorem parallelepiped_with_rectangular_opposite_faces_is_right 
  (P : Parallelepiped) 
  (F1 F2 : Face P) 
  (h1 : is_rectangular F1) 
  (h2 : is_rectangular F2) 
  (h3 : are_opposite F1 F2) : 
  RightParallelepiped := sorry

end parallelepiped_with_rectangular_opposite_faces_is_right_l2119_211906


namespace min_value_reciprocal_sum_l2119_211970

theorem min_value_reciprocal_sum (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by sorry

end min_value_reciprocal_sum_l2119_211970


namespace stadium_seats_pattern_l2119_211921

/-- Represents the number of seats in a row of the stadium -/
def seats (n : ℕ) : ℕ := n + 49

/-- The theorem states that the number of seats in each row follows the given pattern -/
theorem stadium_seats_pattern (n : ℕ) (h : 1 ≤ n ∧ n ≤ 40) : 
  seats n = 50 + (n - 1) := by sorry

end stadium_seats_pattern_l2119_211921


namespace unique_line_through_point_with_equal_intercepts_l2119_211974

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using the general form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def equal_intercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c / l.a = -l.c / l.b

theorem unique_line_through_point_with_equal_intercepts :
  ∃! l : Line2D, point_on_line ⟨0, 5⟩ l ∧ equal_intercepts l :=
sorry

end unique_line_through_point_with_equal_intercepts_l2119_211974


namespace mary_money_left_l2119_211984

/-- The amount of money Mary has left after her purchases -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 4 * p
  let total_cost := 4 * drink_cost + medium_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that Mary will have 50 - 10p dollars left after her purchases -/
theorem mary_money_left (p : ℝ) : money_left p = 50 - 10 * p := by
  sorry

end mary_money_left_l2119_211984


namespace jakes_weight_l2119_211992

theorem jakes_weight (jake sister brother : ℝ) 
  (h1 : jake - 40 = 3 * sister)
  (h2 : jake - (sister + 10) = brother)
  (h3 : jake + sister + brother = 300) :
  jake = 155 := by
sorry

end jakes_weight_l2119_211992


namespace first_term_is_five_halves_l2119_211949

/-- Sum of first n terms of an arithmetic sequence -/
def T (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- The ratio of T(3n) to T(n) is constant for all positive n -/
def ratio_is_constant (a : ℚ) : Prop :=
  ∃ k : ℚ, ∀ n : ℕ, n > 0 → T a (3*n) / T a n = k

theorem first_term_is_five_halves :
  ∀ a : ℚ, ratio_is_constant a → a = 5/2 := by sorry

end first_term_is_five_halves_l2119_211949


namespace construct_one_to_ten_l2119_211967

/-- A type representing the allowed operations in our constructions -/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide
  | Exponentiate

/-- A type representing a construction using threes and operations -/
inductive Construction
  | Three : Construction
  | Op : Operation → Construction → Construction → Construction

/-- Evaluate a construction to a rational number -/
def evaluate : Construction → ℚ
  | Construction.Three => 3
  | Construction.Op Operation.Add a b => evaluate a + evaluate b
  | Construction.Op Operation.Subtract a b => evaluate a - evaluate b
  | Construction.Op Operation.Multiply a b => evaluate a * evaluate b
  | Construction.Op Operation.Divide a b => evaluate a / evaluate b
  | Construction.Op Operation.Exponentiate a b => (evaluate a) ^ (evaluate b).num

/-- Count the number of threes used in a construction -/
def countThrees : Construction → ℕ
  | Construction.Three => 1
  | Construction.Op _ a b => countThrees a + countThrees b

/-- Predicate to check if a construction is valid (uses exactly five threes) -/
def isValidConstruction (c : Construction) : Prop := countThrees c = 5

/-- Theorem: We can construct all numbers from 1 to 10 using five threes and allowed operations -/
theorem construct_one_to_ten :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 →
  ∃ c : Construction, isValidConstruction c ∧ evaluate c = n := by sorry

end construct_one_to_ten_l2119_211967


namespace pet_ownership_l2119_211957

theorem pet_ownership (S D C B : Finset ℕ) (h1 : S.card = 50)
  (h2 : ∀ s ∈ S, s ∈ D ∨ s ∈ C ∨ s ∈ B)
  (h3 : D.card = 30) (h4 : C.card = 35) (h5 : B.card = 10)
  (h6 : (D ∩ C ∩ B).card = 5) :
  ((D ∩ C) \ B).card = 25 := by
  sorry

end pet_ownership_l2119_211957


namespace megan_eggs_count_l2119_211953

theorem megan_eggs_count :
  ∀ (broken cracked perfect : ℕ),
  broken = 3 →
  cracked = 2 * broken →
  perfect - cracked = 9 →
  broken + cracked + perfect = 24 :=
by
  sorry

end megan_eggs_count_l2119_211953


namespace intersection_implies_b_range_l2119_211903

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the line equation
def line_equation (x y b : ℝ) : Prop := y = -2 * x + b

-- Define the condition for line intersection with segment AB
def intersects_AB (b : ℝ) : Prop :=
  ∃ (x y : ℝ), line_equation x y b ∧ 
  ((x ≥ A.1 ∧ x ≤ B.1) ∨ (x ≤ A.1 ∧ x ≥ B.1)) ∧
  ((y ≥ A.2 ∧ y ≤ B.2) ∨ (y ≤ A.2 ∧ y ≥ B.2))

-- Theorem statement
theorem intersection_implies_b_range :
  ∀ b : ℝ, intersects_AB b → b ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

end intersection_implies_b_range_l2119_211903


namespace monomial_sum_equation_solution_l2119_211968

theorem monomial_sum_equation_solution :
  ∀ (a b : ℝ) (m n : ℕ),
  (∃ (k : ℝ), ∀ (a b : ℝ), (1/3 * a^m * b^3) + (-2 * a^2 * b^n) = k * a^m * b^n) →
  (∃ (x : ℝ), (x - 7) / n - (1 + x) / m = 1) →
  (∃ (x : ℝ), (x - 7) / n - (1 + x) / m = 1 ∧ x = -23) :=
by sorry

end monomial_sum_equation_solution_l2119_211968


namespace orange_ratio_l2119_211994

def total_oranges : ℕ := 180
def alice_oranges : ℕ := 120

theorem orange_ratio : 
  let emily_oranges := total_oranges - alice_oranges
  (alice_oranges : ℚ) / emily_oranges = 2 := by
sorry

end orange_ratio_l2119_211994


namespace lucky_number_2005_to_52000_l2119_211955

/-- A natural number is a lucky number if the sum of its digits is 7 -/
def is_lucky_number (n : ℕ) : Prop :=
  (n.digits 10).sum = 7

/-- The sequence of lucky numbers in ascending order -/
def lucky_number_sequence : ℕ → ℕ :=
  sorry

/-- The 2005th lucky number is the nth in the sequence -/
axiom a_2005_is_nth : ∃ n : ℕ, lucky_number_sequence n = 2005

theorem lucky_number_2005_to_52000 :
  ∃ n : ℕ, lucky_number_sequence n = 2005 ∧ lucky_number_sequence (5 * n) = 52000 :=
sorry

end lucky_number_2005_to_52000_l2119_211955


namespace carlson_ate_66_candies_l2119_211952

/-- Represents the number of candies eaten by Carlson given the initial conditions --/
def carlson_candies : ℕ :=
  let initial_candies : ℕ := 300
  let boy_daily_consumption : ℕ := 1
  let carlson_sunday_consumption : ℕ := 2
  let days_per_week : ℕ := 7
  let start_day : ℕ := 2  -- Tuesday (0-based index, where 0 is Sunday)

  let weekly_consumption : ℕ := boy_daily_consumption * days_per_week + carlson_sunday_consumption
  let complete_weeks : ℕ := initial_candies / weekly_consumption

  complete_weeks * carlson_sunday_consumption

/-- Theorem stating that Carlson ate 66 candies --/
theorem carlson_ate_66_candies : carlson_candies = 66 := by
  sorry

end carlson_ate_66_candies_l2119_211952


namespace seven_balls_three_boxes_l2119_211946

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 36 := by
  sorry

end seven_balls_three_boxes_l2119_211946


namespace digit_1997_of_1_22_digit_1997_of_1_27_l2119_211936

/-- The nth decimal digit of a rational number -/
def nthDecimalDigit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- The 1997th decimal digit of 1/22 is 0 -/
theorem digit_1997_of_1_22 : nthDecimalDigit (1/22) 1997 = 0 := by sorry

/-- The 1997th decimal digit of 1/27 is 3 -/
theorem digit_1997_of_1_27 : nthDecimalDigit (1/27) 1997 = 3 := by sorry

end digit_1997_of_1_22_digit_1997_of_1_27_l2119_211936


namespace hillarys_money_after_deposit_l2119_211930

/-- The amount of money Hillary is left with after selling crafts and making a deposit -/
def hillarys_remaining_money (craft_price : ℕ) (crafts_sold : ℕ) (extra_money : ℕ) (deposit : ℕ) : ℕ :=
  craft_price * crafts_sold + extra_money - deposit

/-- Theorem stating that Hillary is left with 25 dollars after selling crafts and making a deposit -/
theorem hillarys_money_after_deposit :
  hillarys_remaining_money 12 3 7 18 = 25 := by
  sorry

end hillarys_money_after_deposit_l2119_211930


namespace island_marriage_fraction_l2119_211959

theorem island_marriage_fraction (N : ℚ) :
  let M := (3/2) * N  -- Total number of men
  let W := (5/3) * N  -- Total number of women
  let P := M + W      -- Total population
  (2 * N) / P = 12/19 := by
  sorry

end island_marriage_fraction_l2119_211959


namespace set_equation_solution_l2119_211972

theorem set_equation_solution (p a b : ℝ) : 
  let A := {x : ℝ | x^2 - p*x + 15 = 0}
  let B := {x : ℝ | x^2 - a*x - b = 0}
  (A ∪ B = {2, 3, 5} ∧ A ∩ B = {3}) → (p = 8 ∧ a = 5 ∧ b = -6) := by
  sorry

end set_equation_solution_l2119_211972


namespace max_value_sin_cos_max_value_achievable_l2119_211942

theorem max_value_sin_cos (θ : ℝ) : 
  (1/2) * Real.sin (3 * θ)^2 - (1/2) * Real.cos (2 * θ) ≤ 1 :=
sorry

theorem max_value_achievable : 
  ∃ θ : ℝ, (1/2) * Real.sin (3 * θ)^2 - (1/2) * Real.cos (2 * θ) = 1 :=
sorry

end max_value_sin_cos_max_value_achievable_l2119_211942


namespace triangle_angle_measure_l2119_211909

theorem triangle_angle_measure (angle_CBD angle_other : ℝ) :
  angle_CBD = 117 →
  angle_other = 31 →
  ∃ (angle_y : ℝ), 
    angle_y + angle_other + (180 - angle_CBD) = 180 ∧
    angle_y = 86 := by
  sorry

end triangle_angle_measure_l2119_211909


namespace solution_x_equals_two_l2119_211933

theorem solution_x_equals_two : 
  let x : ℝ := 2
  3 * x - 6 = 0 :=
by sorry

end solution_x_equals_two_l2119_211933


namespace hyperbola_eccentricity_l2119_211964

/-- A hyperbola with center at the origin and axes of symmetry along the coordinate axes -/
structure CenteredHyperbola where
  /-- The angle of inclination of one of the asymptotes -/
  asymptote_angle : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : CenteredHyperbola) : ℝ :=
  sorry

/-- Theorem stating the possible eccentricities of a hyperbola with an asymptote angle of π/3 -/
theorem hyperbola_eccentricity (h : CenteredHyperbola) 
  (h_angle : h.asymptote_angle = π / 3) : 
  eccentricity h = 2 ∨ eccentricity h = 2 * Real.sqrt 3 / 3 :=
sorry

end hyperbola_eccentricity_l2119_211964


namespace sum_of_reciprocals_l2119_211916

theorem sum_of_reciprocals (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b + c = 0) :
  1 / (b^3 + c^3 - a^3) + 1 / (a^3 + c^3 - b^3) + 1 / (a^3 + b^3 - c^3) = 1 / (a * b * c) := by
  sorry

end sum_of_reciprocals_l2119_211916


namespace square_plot_area_l2119_211985

theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) : 
  price_per_foot = 58 → total_cost = 1624 → 
  ∃ (side_length : ℝ), 
    side_length > 0 ∧ 
    4 * side_length * price_per_foot = total_cost ∧ 
    side_length^2 = 49 := by
  sorry

#check square_plot_area

end square_plot_area_l2119_211985


namespace sugar_price_increase_l2119_211956

theorem sugar_price_increase (original_price : ℝ) (consumption_reduction : ℝ) :
  original_price = 3 →
  consumption_reduction = 0.4 →
  let new_consumption := 1 - consumption_reduction
  let new_price := original_price / new_consumption
  new_price = 5 := by
  sorry

end sugar_price_increase_l2119_211956


namespace cubic_equation_solution_l2119_211948

theorem cubic_equation_solution (x : ℝ) : (x + 2)^3 = 64 → x = 2 := by
  sorry

end cubic_equation_solution_l2119_211948


namespace black_white_area_ratio_l2119_211918

/-- The ratio of black to white areas in concentric circles -/
theorem black_white_area_ratio :
  let r₁ : ℝ := 2
  let r₂ : ℝ := 4
  let r₃ : ℝ := 6
  let r₄ : ℝ := 8
  let black_area := (r₂^2 - r₁^2) * Real.pi + (r₄^2 - r₃^2) * Real.pi
  let white_area := r₁^2 * Real.pi + (r₃^2 - r₂^2) * Real.pi
  black_area / white_area = 5 / 3 := by
  sorry

end black_white_area_ratio_l2119_211918


namespace better_fit_larger_R_squared_l2119_211908

/-- The correlation index in regression analysis -/
def correlation_index (model : Type*) : ℝ := sorry

/-- The fitting effect of a regression model -/
def fitting_effect (model : Type*) : ℝ := sorry

/-- Theorem stating that a larger correlation index implies a better fitting effect -/
theorem better_fit_larger_R_squared (model1 model2 : Type*) :
  correlation_index model1 > correlation_index model2 →
  fitting_effect model1 > fitting_effect model2 :=
by sorry

end better_fit_larger_R_squared_l2119_211908


namespace square_39_equals_square_40_minus_79_l2119_211976

theorem square_39_equals_square_40_minus_79 : 39^2 = 40^2 - 79 := by
  sorry

end square_39_equals_square_40_minus_79_l2119_211976


namespace binomial_coefficient_sum_l2119_211983

theorem binomial_coefficient_sum (n : ℕ) : 
  let m := (4 : ℕ) ^ n
  let k := (2 : ℕ) ^ n
  m + k = 1056 → n = 5 := by
sorry

end binomial_coefficient_sum_l2119_211983


namespace count_valid_numbers_is_1800_l2119_211928

/-- Define a 5-digit number -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- Define the quotient and remainder when n is divided by 50 -/
def quotient_remainder (n q r : ℕ) : Prop :=
  n = 50 * q + r ∧ r < 50

/-- Count of 5-digit numbers n where q + r is divisible by 9 -/
def count_valid_numbers : ℕ := sorry

/-- Theorem stating the count of valid numbers is 1800 -/
theorem count_valid_numbers_is_1800 :
  count_valid_numbers = 1800 := by sorry

end count_valid_numbers_is_1800_l2119_211928


namespace second_person_speed_l2119_211929

/-- Given two persons starting at the same point, walking in opposite directions
    for 3.5 hours, with one person walking at 6 km/hr, and ending up 45.5 km apart,
    the speed of the second person is 7 km/hr. -/
theorem second_person_speed (person1_speed : ℝ) (person2_speed : ℝ) (time : ℝ) (distance : ℝ) :
  person1_speed = 6 →
  time = 3.5 →
  distance = 45.5 →
  distance = (person1_speed + person2_speed) * time →
  person2_speed = 7 := by
  sorry

end second_person_speed_l2119_211929


namespace problem_solution_l2119_211943

theorem problem_solution (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 1/2) : m = 100 := by
  sorry

end problem_solution_l2119_211943
