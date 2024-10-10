import Mathlib

namespace x_intercepts_difference_l3936_393652

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the x-intercepts
variable (x₁ x₂ x₃ x₄ : ℝ)

-- State the conditions
axiom g_def : ∀ x, g x = 2 * f (200 - x)
axiom vertex_condition : ∃ v, f v = 0 ∧ g v = 0
axiom x_intercepts_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄
axiom x_intercepts_f : f x₁ = 0 ∧ f x₄ = 0
axiom x_intercepts_g : g x₂ = 0 ∧ g x₃ = 0
axiom x_diff : x₃ - x₂ = 300

-- Theorem to prove
theorem x_intercepts_difference : x₄ - x₁ = 600 := by sorry

end x_intercepts_difference_l3936_393652


namespace zoo_recovery_time_l3936_393698

/-- The total time spent recovering escaped animals from a zoo -/
def total_recovery_time (num_lions num_rhinos num_giraffes num_gorillas : ℕ) 
  (time_per_lion time_per_rhino time_per_giraffe time_per_gorilla : ℝ) : ℝ :=
  (num_lions : ℝ) * time_per_lion + 
  (num_rhinos : ℝ) * time_per_rhino + 
  (num_giraffes : ℝ) * time_per_giraffe + 
  (num_gorillas : ℝ) * time_per_gorilla

/-- Theorem stating that the total recovery time for the given scenario is 33 hours -/
theorem zoo_recovery_time : 
  total_recovery_time 5 3 2 4 2 3 4 1.5 = 33 := by
  sorry

end zoo_recovery_time_l3936_393698


namespace middle_school_eight_total_games_l3936_393635

/-- Represents a basketball conference -/
structure BasketballConference where
  numTeams : ℕ
  intraConferenceGamesPerPair : ℕ
  nonConferenceGamesPerTeam : ℕ

/-- Calculate the total number of games in a season for a given basketball conference -/
def totalGamesInSeason (conf : BasketballConference) : ℕ :=
  let intraConferenceGames := conf.numTeams.choose 2 * conf.intraConferenceGamesPerPair
  let nonConferenceGames := conf.numTeams * conf.nonConferenceGamesPerTeam
  intraConferenceGames + nonConferenceGames

/-- The "Middle School Eight" basketball conference -/
def middleSchoolEight : BasketballConference :=
  { numTeams := 8
  , intraConferenceGamesPerPair := 2
  , nonConferenceGamesPerTeam := 4 }

theorem middle_school_eight_total_games :
  totalGamesInSeason middleSchoolEight = 88 := by
  sorry


end middle_school_eight_total_games_l3936_393635


namespace equal_intercept_line_equation_l3936_393638

/-- A line passing through point (1, 1) with equal intercepts on both coordinate axes -/
def equal_intercept_line (x y : ℝ) : Prop :=
  (x = 1 ∧ y = 1) ∨ 
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a * x + b * y = a * b ∧ a = b)

/-- The equation of the line is x - y = 0 or x + y - 2 = 0 -/
theorem equal_intercept_line_equation (x y : ℝ) :
  equal_intercept_line x y ↔ (x - y = 0 ∨ x + y - 2 = 0) := by
  sorry

end equal_intercept_line_equation_l3936_393638


namespace dragon_rope_problem_l3936_393697

-- Define the constants
def rope_length : ℝ := 25
def castle_radius : ℝ := 5
def dragon_height : ℝ := 3
def rope_end_distance : ℝ := 3

-- Define the variables
variable (p q r : ℕ)

-- Define the conditions
axiom p_positive : p > 0
axiom q_positive : q > 0
axiom r_positive : r > 0
axiom r_prime : Nat.Prime r

-- Define the relationship between p, q, r and the rope length touching the castle
axiom rope_touching_castle : (p - Real.sqrt q) / r = (75 - Real.sqrt 450) / 3

-- Theorem to prove
theorem dragon_rope_problem : p + q + r = 528 := by sorry

end dragon_rope_problem_l3936_393697


namespace monochromatic_triangle_exists_l3936_393675

/-- A type representing the scientists -/
def Scientist : Type := Fin 17

/-- A type representing the topics -/
inductive Topic
| A
| B
| C

/-- A function representing the correspondence between scientists on topics -/
def correspondence : Scientist → Scientist → Topic := sorry

/-- The main theorem stating that there exists a monochromatic triangle -/
theorem monochromatic_triangle_exists :
  ∃ (s1 s2 s3 : Scientist) (t : Topic),
    s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3 ∧
    correspondence s1 s2 = t ∧
    correspondence s1 s3 = t ∧
    correspondence s2 s3 = t :=
  sorry

end monochromatic_triangle_exists_l3936_393675


namespace perfect_square_condition_l3936_393655

theorem perfect_square_condition (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - (m+1)*x + 1 = k^2) → (m = 1 ∨ m = -3) :=
by sorry

end perfect_square_condition_l3936_393655


namespace prob_no_red_3x3_is_170_171_l3936_393618

/-- Represents a 4x4 grid where each cell can be either red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 3x3 subgrid starting at (i, j) is all red -/
def is_red_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ x y, g (i + x) (j + y) = true

/-- The probability of a random 4x4 grid not containing a 3x3 red square -/
def prob_no_red_3x3 : ℚ :=
  170 / 171

/-- The main theorem stating the probability of a 4x4 grid not containing a 3x3 red square -/
theorem prob_no_red_3x3_is_170_171 :
  prob_no_red_3x3 = 170 / 171 := by sorry

end prob_no_red_3x3_is_170_171_l3936_393618


namespace circle_equation_proof_l3936_393604

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the circle equation
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- State the theorem
theorem circle_equation_proof :
  ∃ (c : Circle),
    (∃ (x : ℝ), line1 x 0 ∧ c.center = (x, 0)) ∧
    (∀ (x y : ℝ), line2 x y → ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2)) →
    ∀ (x y : ℝ), circleEquation c x y ↔ (x + 1)^2 + y^2 = 2 :=
by sorry

end circle_equation_proof_l3936_393604


namespace cubic_factorization_l3936_393666

theorem cubic_factorization (a : ℝ) : a^3 + 2*a^2 + a = a*(a+1)^2 := by
  sorry

end cubic_factorization_l3936_393666


namespace intersection_of_A_and_B_l3936_393619

def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end intersection_of_A_and_B_l3936_393619


namespace least_number_with_remainder_5_l3936_393671

/-- The least number that leaves a remainder of 5 when divided by 8, 12, 15, and 20 -/
def leastNumber : ℕ := 125

/-- Checks if a number leaves a remainder of 5 when divided by the given divisor -/
def hasRemainder5 (n : ℕ) (divisor : ℕ) : Prop :=
  n % divisor = 5

theorem least_number_with_remainder_5 :
  (∀ divisor ∈ [8, 12, 15, 20], hasRemainder5 leastNumber divisor) ∧
  (∀ m < leastNumber, ∃ divisor ∈ [8, 12, 15, 20], ¬hasRemainder5 m divisor) :=
sorry

end least_number_with_remainder_5_l3936_393671


namespace necklace_stand_capacity_l3936_393691

-- Define the given constants
def current_necklaces : ℕ := 5
def ring_capacity : ℕ := 30
def current_rings : ℕ := 18
def bracelet_capacity : ℕ := 15
def current_bracelets : ℕ := 8
def necklace_price : ℕ := 4
def ring_price : ℕ := 10
def bracelet_price : ℕ := 5
def total_cost : ℕ := 183

-- Theorem to prove
theorem necklace_stand_capacity : ∃ (total_necklaces : ℕ), 
  total_necklaces = current_necklaces + 
    ((total_cost - (ring_price * (ring_capacity - current_rings) + 
                    bracelet_price * (bracelet_capacity - current_bracelets))) / necklace_price) :=
by sorry

end necklace_stand_capacity_l3936_393691


namespace pentagonal_dodecahedron_properties_l3936_393686

/-- A polyhedron with pentagonal faces -/
structure PentagonalPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Euler's formula for polyhedra -/
axiom eulers_formula {p : PentagonalPolyhedron} : p.vertices - p.edges + p.faces = 2

/-- Each face is a pentagon -/
axiom pentagonal_faces {p : PentagonalPolyhedron} : p.edges * 2 = p.faces * 5

/-- Theorem: A polyhedron with 12 pentagonal faces has 30 edges and 20 vertices -/
theorem pentagonal_dodecahedron_properties :
  ∃ (p : PentagonalPolyhedron), p.faces = 12 ∧ p.edges = 30 ∧ p.vertices = 20 :=
sorry

end pentagonal_dodecahedron_properties_l3936_393686


namespace xyz_product_l3936_393695

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 6 * y = -24)
  (eq2 : y * z + 6 * z = -24)
  (eq3 : z * x + 6 * x = -24) :
  x * y * z = 120 := by
sorry

end xyz_product_l3936_393695


namespace train_crossing_time_l3936_393649

/-- Proves the time it takes for a train to cross a signal post given its length and the time it takes to cross a bridge -/
theorem train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (bridge_crossing_time : ℝ) :
  train_length = 600 →
  bridge_length = 18000 →
  bridge_crossing_time = 1200 →
  (train_length / (bridge_length / bridge_crossing_time)) = 40 := by
  sorry

end train_crossing_time_l3936_393649


namespace rob_has_five_nickels_l3936_393659

/-- Represents the number of coins of each type Rob has -/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in cents for a given CoinCount -/
def totalValueInCents (coins : CoinCount) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Proves that Rob has 5 nickels given the conditions -/
theorem rob_has_five_nickels :
  ∃ (robsCoins : CoinCount),
    robsCoins.quarters = 7 ∧
    robsCoins.dimes = 3 ∧
    robsCoins.pennies = 12 ∧
    totalValueInCents robsCoins = 242 ∧
    robsCoins.nickels = 5 := by
  sorry


end rob_has_five_nickels_l3936_393659


namespace horror_movie_tickets_l3936_393667

theorem horror_movie_tickets (romance_tickets : ℕ) (horror_tickets : ℕ) : 
  romance_tickets = 25 →
  horror_tickets = 3 * romance_tickets + 18 →
  horror_tickets = 93 := by
sorry

end horror_movie_tickets_l3936_393667


namespace prime_sum_equation_l3936_393613

theorem prime_sum_equation (p q s : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime s →
  p + q = s + 4 →
  1 < p →
  p < q →
  p = 2 := by
sorry

end prime_sum_equation_l3936_393613


namespace point_ordering_on_reciprocal_function_l3936_393672

/-- Given points on the graph of y = k/x where k > 0, prove a < c < b -/
theorem point_ordering_on_reciprocal_function (k a b c : ℝ) : 
  k > 0 → 
  a * (-2) = k → 
  b * 2 = k → 
  c * 3 = k → 
  a < c ∧ c < b := by
  sorry

end point_ordering_on_reciprocal_function_l3936_393672


namespace curve_properties_l3936_393603

/-- The curve equation -/
def curve (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0

theorem curve_properties :
  (∀ x y : ℝ, curve x y 1 ↔ x = 1 ∧ y = 1) ∧
  (∀ a : ℝ, a ≠ 1 → curve 1 1 a) :=
by sorry

end curve_properties_l3936_393603


namespace triangle_area_l3936_393674

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define the length of a side
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two sides
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def area (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area (A B C : ℝ × ℝ) :
  Triangle A B C →
  length A B = 6 →
  angle B A C = 30 * π / 180 →
  angle A B C = 120 * π / 180 →
  area A B C = 9 * Real.sqrt 3 := by
  sorry

end triangle_area_l3936_393674


namespace polygon_sides_l3936_393683

/-- Theorem: A polygon with 1080° as the sum of its interior angles has 8 sides. -/
theorem polygon_sides (n : ℕ) : (180 * (n - 2) = 1080) → n = 8 := by
  sorry

end polygon_sides_l3936_393683


namespace equal_sum_sequence_2011_sum_l3936_393642

/-- Definition of an equal sum sequence -/
def IsEqualSumSequence (a : ℕ → ℤ) (sum : ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = sum

/-- Definition of the sum of the first n terms of a sequence -/
def SequenceSum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum a

theorem equal_sum_sequence_2011_sum
  (a : ℕ → ℤ)
  (h_equal_sum : IsEqualSumSequence a 1)
  (h_first_term : a 1 = -1) :
  SequenceSum a 2011 = 1004 := by
sorry

end equal_sum_sequence_2011_sum_l3936_393642


namespace conference_handshakes_l3936_393625

/-- Represents a conference with a fixed number of attendees and handshakes per person -/
structure Conference where
  attendees : ℕ
  handshakes_per_person : ℕ

/-- Calculates the minimum number of unique handshakes in a conference -/
def min_handshakes (conf : Conference) : ℕ :=
  conf.attendees * conf.handshakes_per_person / 2

/-- Theorem stating that in a conference of 30 people where each person shakes hands
    with exactly 3 others, the minimum number of unique handshakes is 45 -/
theorem conference_handshakes :
  let conf : Conference := { attendees := 30, handshakes_per_person := 3 }
  min_handshakes conf = 45 := by
  sorry


end conference_handshakes_l3936_393625


namespace fraction_power_equality_l3936_393662

theorem fraction_power_equality : (72000 ^ 4 : ℝ) / (24000 ^ 4) = 81 := by sorry

end fraction_power_equality_l3936_393662


namespace shooting_probability_l3936_393661

theorem shooting_probability (accuracy : ℝ) (two_shots : ℝ) :
  accuracy = 9/10 →
  two_shots = 1/2 →
  (two_shots / accuracy) = 5/9 :=
by sorry

end shooting_probability_l3936_393661


namespace katies_flour_amount_l3936_393699

theorem katies_flour_amount (katie_flour : ℕ) (sheila_flour : ℕ) : 
  sheila_flour = katie_flour + 2 →
  katie_flour + sheila_flour = 8 →
  katie_flour = 3 := by
sorry

end katies_flour_amount_l3936_393699


namespace hyperbola_equation_l3936_393664

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c > 0 ∧
    c / a = Real.sqrt 6 / 2 ∧
    b * c / Real.sqrt (a^2 + b^2) = 1 ∧
    c^2 = a^2 + b^2) →
  a^2 = 2 ∧ b^2 = 1 := by
  sorry

end hyperbola_equation_l3936_393664


namespace student_calculation_difference_l3936_393608

theorem student_calculation_difference : 
  let number : ℝ := 80.00000000000003
  let correct_answer := number * (4/5 : ℝ)
  let student_answer := number / (4/5 : ℝ)
  student_answer - correct_answer = 36.0000000000000175 := by
sorry

end student_calculation_difference_l3936_393608


namespace product_125_sum_31_l3936_393629

theorem product_125_sum_31 :
  ∃ (a b c : ℕ+), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a * b * c : ℕ) = 125 →
    (a + b + c : ℕ) = 31 := by
sorry

end product_125_sum_31_l3936_393629


namespace donny_apple_purchase_cost_l3936_393658

-- Define the prices of apples
def small_apple_price : ℚ := 1.5
def medium_apple_price : ℚ := 2
def big_apple_price : ℚ := 3

-- Define the number of apples Donny bought
def small_apples_bought : ℕ := 6
def medium_apples_bought : ℕ := 6
def big_apples_bought : ℕ := 8

-- Calculate the total cost
def total_cost : ℚ := 
  small_apple_price * small_apples_bought +
  medium_apple_price * medium_apples_bought +
  big_apple_price * big_apples_bought

-- Theorem statement
theorem donny_apple_purchase_cost : total_cost = 45 := by
  sorry

end donny_apple_purchase_cost_l3936_393658


namespace unique_divisor_property_l3936_393612

theorem unique_divisor_property (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_5 : p > 5) :
  ∃! x : ℕ, x ≠ 0 ∧ ∀ n : ℕ, n > 0 → (5 * p + x) ∣ (5 * p^n + x^n) ∧ x = p := by
  sorry

end unique_divisor_property_l3936_393612


namespace missing_fraction_sum_l3936_393688

theorem missing_fraction_sum (x : ℚ) : x = -11/60 →
  1/3 + 1/2 + (-5/6) + 1/5 + 1/4 + (-2/15) + x = 0.13333333333333333 := by
  sorry

end missing_fraction_sum_l3936_393688


namespace internship_arrangements_l3936_393687

/-- The number of ways to arrange 4 distinct objects into 2 indistinguishable pairs 
    and then assign these pairs to 2 distinct locations -/
theorem internship_arrangements (n : Nat) (m : Nat) : n = 4 ∧ m = 2 → 
  (Nat.choose n 2 / 2) * m.factorial = 6 := by
  sorry

end internship_arrangements_l3936_393687


namespace unique_solution_2m_minus_1_eq_3n_l3936_393643

theorem unique_solution_2m_minus_1_eq_3n :
  ∀ m n : ℕ+, 2^(m : ℕ) - 1 = 3^(n : ℕ) ↔ m = 2 ∧ n = 1 := by
  sorry

end unique_solution_2m_minus_1_eq_3n_l3936_393643


namespace smallest_n_congruence_l3936_393663

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 827 * n ≡ 1369 * n [ZMOD 36] ∧ ∀ (m : ℕ), m > 0 → 827 * m ≡ 1369 * m [ZMOD 36] → n ≤ m :=
by sorry

end smallest_n_congruence_l3936_393663


namespace frog_game_result_l3936_393677

def frog_A_jump : ℕ := 10
def frog_B_jump : ℕ := 15
def trap_interval : ℕ := 12

def first_trap (jump_distance : ℕ) : ℕ :=
  (trap_interval / jump_distance) * jump_distance

theorem frog_game_result :
  let first_frog_trap := min (first_trap frog_A_jump) (first_trap frog_B_jump)
  let other_frog_distance := if first_frog_trap = first_trap frog_B_jump
                             then (first_frog_trap / frog_B_jump) * frog_A_jump
                             else (first_frog_trap / frog_A_jump) * frog_B_jump
  (trap_interval - (other_frog_distance % trap_interval)) % trap_interval = 8 := by
  sorry

end frog_game_result_l3936_393677


namespace last_number_proof_l3936_393607

theorem last_number_proof (A B C D : ℝ) 
  (h1 : (A + B + C) / 3 = 6)
  (h2 : (B + C + D) / 3 = 3)
  (h3 : A + D = 13) :
  D = 2 := by
sorry

end last_number_proof_l3936_393607


namespace smallest_integer_gcd_with_18_l3936_393680

theorem smallest_integer_gcd_with_18 : ∃ n : ℕ, n > 100 ∧ n.gcd 18 = 6 ∧ ∀ m : ℕ, m > 100 ∧ m.gcd 18 = 6 → n ≤ m := by
  sorry

end smallest_integer_gcd_with_18_l3936_393680


namespace triangle_shape_l3936_393632

theorem triangle_shape (a b : ℝ) (A B : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) 
  (h : a * Real.cos A = b * Real.cos B) :
  A = B ∨ A + B = π / 2 := by
sorry

end triangle_shape_l3936_393632


namespace third_side_length_equal_to_altitude_l3936_393690

/-- Given an acute-angled triangle with two sides of lengths √13 and √10 cm,
    if the third side is equal to the altitude drawn to it,
    then the length of the third side is 3 cm. -/
theorem third_side_length_equal_to_altitude
  (a b c : ℝ) -- sides of the triangle
  (h : ℝ) -- altitude to side c
  (acute : 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) -- acute-angled triangle
  (side1 : a = Real.sqrt 13)
  (side2 : b = Real.sqrt 10)
  (altitude_eq_side : h = c)
  (pythagorean1 : a^2 = (c - h)^2 + h^2)
  (pythagorean2 : b^2 = h^2 + h^2) :
  c = 3 := by
  sorry

end third_side_length_equal_to_altitude_l3936_393690


namespace dans_minimum_spending_l3936_393676

/-- Given Dan's purchases and spending information, prove he spent at least $9 -/
theorem dans_minimum_spending (chocolate_cost candy_cost difference : ℕ) 
  (h1 : chocolate_cost = 7)
  (h2 : candy_cost = 2)
  (h3 : chocolate_cost = candy_cost + difference)
  (h4 : difference = 5) : 
  chocolate_cost + candy_cost ≥ 9 := by
  sorry

#check dans_minimum_spending

end dans_minimum_spending_l3936_393676


namespace bus_occupancy_problem_l3936_393668

/-- Given an initial number of people on a bus, and the number of people who get on and off,
    calculate the final number of people on the bus. -/
def final_bus_occupancy (initial : ℕ) (got_on : ℕ) (got_off : ℕ) : ℕ :=
  initial + got_on - got_off

/-- Theorem stating that with 32 people initially on the bus, 19 getting on, and 13 getting off,
    the final number of people on the bus is 38. -/
theorem bus_occupancy_problem :
  final_bus_occupancy 32 19 13 = 38 := by
  sorry

end bus_occupancy_problem_l3936_393668


namespace geometric_sequence_problem_l3936_393601

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) * a m = a n * a (m + 1)

/-- The problem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_a5 : a 5 = 2) 
    (h_a7 : a 7 = 8) : 
    a 6 = 4 ∨ a 6 = -4 := by
  sorry


end geometric_sequence_problem_l3936_393601


namespace divisibility_of_sum_of_squares_l3936_393637

theorem divisibility_of_sum_of_squares (p a b : ℕ) : 
  Prime p → 
  (∃ n : ℕ, p = 4 * n + 3) → 
  p ∣ (a^2 + b^2) → 
  (p ∣ a ∧ p ∣ b) := by
sorry

end divisibility_of_sum_of_squares_l3936_393637


namespace geometric_sequence_a1_l3936_393689

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_a1 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  a 3 = 1 →
  a 5 = 1/2 →
  a 1 = 2 := by
sorry

end geometric_sequence_a1_l3936_393689


namespace total_pears_l3936_393692

-- Define the number of pears sold
def sold : ℕ := 20

-- Define the number of pears poached in terms of sold
def poached : ℕ := sold / 2

-- Define the number of pears canned in terms of poached
def canned : ℕ := poached + poached / 5

-- Theorem statement
theorem total_pears : sold + poached + canned = 42 := by
  sorry

end total_pears_l3936_393692


namespace collinear_points_imply_a_value_l3936_393609

/-- Given three points A, B, and C in the plane, 
    this function returns true if they are collinear. -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  (C.2 - A.2) * (B.1 - A.1) = (B.2 - A.2) * (C.1 - A.1)

theorem collinear_points_imply_a_value : 
  ∀ a : ℝ, collinear (3, 2) (-2, a) (8, 12) → a = -8 := by
  sorry

end collinear_points_imply_a_value_l3936_393609


namespace louisa_average_speed_l3936_393669

/-- Proves that given the travel conditions, Louisa's average speed was 40 miles per hour -/
theorem louisa_average_speed :
  ∀ (v : ℝ),
  v > 0 →
  (280 / v) = (160 / v) + 3 →
  v = 40 :=
by
  sorry

end louisa_average_speed_l3936_393669


namespace remainder_theorem_l3936_393650

theorem remainder_theorem (x : ℤ) : 
  (x^15 + 3) % (x + 2) = (-2)^15 + 3 :=
by sorry

end remainder_theorem_l3936_393650


namespace truck_capacity_l3936_393644

/-- The total fuel capacity of Donny's truck -/
def total_capacity : ℕ := 150

/-- The amount of fuel already in the truck -/
def initial_fuel : ℕ := 38

/-- The amount of money Donny started with -/
def initial_money : ℕ := 350

/-- The amount of change Donny received -/
def change : ℕ := 14

/-- The cost of fuel per liter -/
def cost_per_liter : ℕ := 3

/-- Theorem stating that the total capacity of Donny's truck is 150 liters -/
theorem truck_capacity : 
  total_capacity = initial_fuel + (initial_money - change) / cost_per_liter := by
  sorry

end truck_capacity_l3936_393644


namespace sequence_a_closed_form_l3936_393600

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 5
  | (n + 2) => (2 * (sequence_a (n + 1))^2 - 3 * sequence_a (n + 1) - 9) / (2 * sequence_a n)

theorem sequence_a_closed_form (n : ℕ) : sequence_a n = 2^(n + 2) - 3 := by
  sorry

end sequence_a_closed_form_l3936_393600


namespace product_of_fractions_l3936_393627

theorem product_of_fractions : (2 : ℚ) / 9 * (5 : ℚ) / 4 = (5 : ℚ) / 18 := by
  sorry

end product_of_fractions_l3936_393627


namespace smaller_number_proof_l3936_393647

theorem smaller_number_proof (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : min a b = 2 := by
  sorry

end smaller_number_proof_l3936_393647


namespace fourth_pillar_height_17_l3936_393616

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space using the general form Ax + By + Cz = D -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the height of the fourth pillar in a square arrangement -/
def calculateFourthPillarHeight (a b c : ℝ) : ℝ :=
  sorry

theorem fourth_pillar_height_17 :
  calculateFourthPillarHeight 15 10 12 = 17 := by
  sorry

end fourth_pillar_height_17_l3936_393616


namespace total_angle_extrema_l3936_393673

/-- A sequence of k positive real numbers -/
def PositiveSequence (k : ℕ) := { seq : Fin k → ℝ // ∀ i, seq i > 0 }

/-- The total angle of rotation for a given sequence of segment lengths -/
noncomputable def TotalAngle (k : ℕ) (seq : Fin k → ℝ) : ℝ := sorry

/-- A permutation of indices -/
def Permutation (k : ℕ) := { perm : Fin k → Fin k // Function.Bijective perm }

theorem total_angle_extrema (k : ℕ) (a : PositiveSequence k) :
  ∃ (max_perm min_perm : Permutation k),
    (∀ i j : Fin k, i ≤ j → (max_perm.val i).val ≤ (max_perm.val j).val) ∧
    (∀ i j : Fin k, i ≤ j → (min_perm.val i).val ≥ (min_perm.val j).val) ∧
    (∀ p : Permutation k,
      TotalAngle k (a.val ∘ p.val) ≤ TotalAngle k (a.val ∘ max_perm.val) ∧
      TotalAngle k (a.val ∘ p.val) ≥ TotalAngle k (a.val ∘ min_perm.val)) :=
sorry

end total_angle_extrema_l3936_393673


namespace triangle_abc_properties_l3936_393693

theorem triangle_abc_properties (a b c A B C : ℝ) : 
  a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = (1/2) * b →
  a > b →
  b = Real.sqrt 13 →
  a + c = 4 →
  B = π / 6 ∧ 
  (1/2) * a * c * Real.sin B = (6 - 3 * Real.sqrt 3) / 4 := by
  sorry

end triangle_abc_properties_l3936_393693


namespace sum_of_max_min_f_l3936_393639

def f (x : ℝ) := x^2 - 2*x - 1

theorem sum_of_max_min_f : 
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧ 
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max + min = 0 :=
by sorry

end sum_of_max_min_f_l3936_393639


namespace train_length_l3936_393624

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 24 → speed * time * (1000 / 3600) = 420 := by
  sorry

end train_length_l3936_393624


namespace solution_set_quadratic_inequality_l3936_393628

theorem solution_set_quadratic_inequality :
  let S : Set ℝ := {x | x^2 + x - 2 ≥ 0}
  S = {x : ℝ | x ≤ -2 ∨ x ≥ 1} := by sorry

end solution_set_quadratic_inequality_l3936_393628


namespace triangle_properties_l3936_393657

/-- Triangle represented by three points in 2D space -/
structure Triangle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- Calculate the squared distance between two points -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Check if a triangle is a right triangle -/
def isRightTriangle (t : Triangle) : Prop :=
  let a := distanceSquared t.p1 t.p2
  let b := distanceSquared t.p2 t.p3
  let c := distanceSquared t.p3 t.p1
  (a + b = c) ∨ (b + c = a) ∨ (c + a = b)

/-- Triangle A -/
def triangleA : Triangle :=
  { p1 := (0, 0), p2 := (3, 4), p3 := (0, 8) }

/-- Triangle B -/
def triangleB : Triangle :=
  { p1 := (3, 4), p2 := (10, 4), p3 := (3, 0) }

theorem triangle_properties :
  ¬(isRightTriangle triangleA) ∧
  (isRightTriangle triangleB) ∧
  (distanceSquared triangleB.p1 triangleB.p2 = 65 ∨
   distanceSquared triangleB.p2 triangleB.p3 = 65 ∨
   distanceSquared triangleB.p3 triangleB.p1 = 65) := by
  sorry


end triangle_properties_l3936_393657


namespace mason_car_nuts_l3936_393645

/-- The number of nuts in Mason's car after squirrels stockpile for a given number of days -/
def nutsInCar (busySquirrels sleepySquirrels : ℕ) (busyNutsPerDay sleepyNutsPerDay days : ℕ) : ℕ :=
  (busySquirrels * busyNutsPerDay + sleepySquirrels * sleepyNutsPerDay) * days

/-- Theorem stating the number of nuts in Mason's car given the problem conditions -/
theorem mason_car_nuts :
  nutsInCar 2 1 30 20 40 = 3200 := by
  sorry

#eval nutsInCar 2 1 30 20 40

end mason_car_nuts_l3936_393645


namespace point_on_x_axis_l3936_393606

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis -/
def xAxis : Set Point :=
  {p : Point | p.y = 0}

theorem point_on_x_axis (a : ℝ) :
  let P : Point := ⟨4, 2*a + 6⟩
  P ∈ xAxis → a = -3 := by
  sorry

end point_on_x_axis_l3936_393606


namespace cos_one_third_solutions_l3936_393670

theorem cos_one_third_solutions (α : Real) (h1 : α ∈ Set.Icc 0 (2 * Real.pi)) (h2 : Real.cos α = 1/3) :
  α = Real.arccos (1/3) ∨ α = 2 * Real.pi - Real.arccos (1/3) := by
  sorry

end cos_one_third_solutions_l3936_393670


namespace system_solutions_l3936_393636

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^4 + (7/2)*x^2*y + 2*y^3 = 0
def equation2 (x y : ℝ) : Prop := 4*x^2 + 7*x*y + 2*y^3 = 0

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(0, 0), (2, -1), (-11/2, -11/2)}

-- Theorem stating that the solution set contains all and only solutions to the system
theorem system_solutions :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set := by
  sorry

end system_solutions_l3936_393636


namespace intersection_of_A_and_B_l3936_393665

def set_A : Set ℤ := {x | |x| < 3}
def set_B : Set ℤ := {x | |x| > 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {-2, 2} := by sorry

end intersection_of_A_and_B_l3936_393665


namespace distance_AB_is_7_l3936_393605

/-- Represents the distance between two points A and B, given the conditions of the pedestrian problem. -/
def distance_AB : ℝ :=
  let v1 : ℝ := 4  -- Speed of the first pedestrian in km/hr
  let v2 : ℝ := 3  -- Speed of the second pedestrian in km/hr
  let t_meet : ℝ := 1.5  -- Time until meeting in hours
  let d1_before : ℝ := v1 * t_meet  -- Distance covered by first pedestrian before meeting
  let d2_before : ℝ := v2 * t_meet  -- Distance covered by second pedestrian before meeting
  let d1_after : ℝ := v1 * 0.75  -- Distance covered by first pedestrian after meeting
  let d2_after : ℝ := v2 * (4/3)  -- Distance covered by second pedestrian after meeting
  d1_before + d2_before  -- Total distance

/-- Theorem stating that the distance between points A and B is 7 km, given the conditions of the pedestrian problem. -/
theorem distance_AB_is_7 : distance_AB = 7 := by
  sorry  -- Proof is omitted as per instructions

#eval distance_AB  -- This will evaluate to 7

end distance_AB_is_7_l3936_393605


namespace min_students_with_blue_eyes_and_lunch_box_l3936_393651

theorem min_students_with_blue_eyes_and_lunch_box
  (total_students : ℕ)
  (blue_eyes : ℕ)
  (lunch_box : ℕ)
  (h1 : total_students = 35)
  (h2 : blue_eyes = 20)
  (h3 : lunch_box = 22)
  : ∃ (both : ℕ), both ≥ 7 ∧ both ≤ min blue_eyes lunch_box :=
by
  sorry

end min_students_with_blue_eyes_and_lunch_box_l3936_393651


namespace logarithm_expression_equals_two_l3936_393611

theorem logarithm_expression_equals_two :
  (Real.log 243 / Real.log 3) / (Real.log 3 / Real.log 81) -
  (Real.log 729 / Real.log 3) / (Real.log 3 / Real.log 27) = 2 := by
  sorry

end logarithm_expression_equals_two_l3936_393611


namespace method_of_continued_proportion_is_correct_l3936_393684

-- Define the possible methods
inductive AncientChineseMathMethod
| CircleCutting
| ContinuedProportion
| SuJiushaoAlgorithm
| SunTzuRemainder

-- Define a property for methods that can find GCD
def canFindGCD (method : AncientChineseMathMethod) : Prop := sorry

-- Define a property for methods from Song and Yuan dynasties
def fromSongYuanDynasties (method : AncientChineseMathMethod) : Prop := sorry

-- Define a property for methods comparable to Euclidean algorithm
def comparableToEuclidean (method : AncientChineseMathMethod) : Prop := sorry

-- Theorem stating that the Method of Continued Proportion is the correct answer
theorem method_of_continued_proportion_is_correct :
  ∃ (method : AncientChineseMathMethod),
    method = AncientChineseMathMethod.ContinuedProportion ∧
    canFindGCD method ∧
    fromSongYuanDynasties method ∧
    comparableToEuclidean method ∧
    (∀ (other : AncientChineseMathMethod),
      other ≠ AncientChineseMathMethod.ContinuedProportion →
      ¬(canFindGCD other ∧ fromSongYuanDynasties other ∧ comparableToEuclidean other)) :=
sorry

end method_of_continued_proportion_is_correct_l3936_393684


namespace range_of_f_l3936_393696

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2*x + 2)

theorem range_of_f :
  Set.range f = Set.Ioo 0 (1/2) ∪ {1/2} :=
sorry

end range_of_f_l3936_393696


namespace collatz_eighth_term_one_l3936_393685

def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def collatzSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => collatz (collatzSequence n k)

def validStartingNumbers : Set ℕ :=
  {n | n > 0 ∧ collatzSequence n 7 = 1}

theorem collatz_eighth_term_one :
  validStartingNumbers = {2, 3, 16, 20, 21, 128} :=
sorry

end collatz_eighth_term_one_l3936_393685


namespace halfway_between_fractions_l3936_393620

theorem halfway_between_fractions : 
  let a := (1 : ℚ) / 8
  let b := (1 : ℚ) / 3
  (a + b) / 2 = 11 / 48 := by sorry

end halfway_between_fractions_l3936_393620


namespace faster_walking_speed_l3936_393602

theorem faster_walking_speed 
  (actual_distance : ℝ) 
  (original_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 60) 
  (h2 : original_speed = 12) 
  (h3 : additional_distance = 20) : 
  let time := actual_distance / original_speed
  let total_distance := actual_distance + additional_distance
  let faster_speed := total_distance / time
  faster_speed = 16 := by sorry

end faster_walking_speed_l3936_393602


namespace glass_volume_l3936_393656

/-- Given a bottle and a glass, proves that the volume of the glass is 0.5 L 
    when water is poured from a full 1.5 L bottle into an empty glass 
    until both are 3/4 full. -/
theorem glass_volume (bottle_initial : ℝ) (glass : ℝ) : 
  bottle_initial = 1.5 →
  (3/4) * bottle_initial + (3/4) * glass = bottle_initial →
  glass = 0.5 := by
  sorry

end glass_volume_l3936_393656


namespace g_of_three_eq_seventeen_sixths_l3936_393679

/-- Given a function g satisfying the equation for all x ≠ 1/2, prove that g(3) = 17/6 -/
theorem g_of_three_eq_seventeen_sixths 
  (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 1/2 → g x + g ((x + 2) / (2 - 4*x)) = 2*x) : 
  g 3 = 17/6 := by
sorry

end g_of_three_eq_seventeen_sixths_l3936_393679


namespace line_AB_equation_l3936_393646

/-- Triangle ABC with given coordinates and line equations -/
structure Triangle where
  B : ℝ × ℝ
  C : ℝ × ℝ
  line_AC : ℝ → ℝ → ℝ
  altitude_A_AB : ℝ → ℝ → ℝ

/-- The equation of line AB in the given triangle -/
def line_AB (t : Triangle) : ℝ → ℝ → ℝ :=
  fun x y => 3 * (x - 3) - 2 * (y - 4)

/-- Theorem stating that the equation of line AB is correct -/
theorem line_AB_equation (t : Triangle) 
  (hB : t.B = (3, 4))
  (hC : t.C = (5, 2))
  (hAC : t.line_AC = fun x y => x - 4*y + 3)
  (hAlt : t.altitude_A_AB = fun x y => 2*x + 3*y - 16) :
  line_AB t = fun x y => 3 * (x - 3) - 2 * (y - 4) := by
  sorry

end line_AB_equation_l3936_393646


namespace sunset_duration_l3936_393626

/-- Proves that a sunset with 12 color changes occurring every 10 minutes lasts 2 hours. -/
theorem sunset_duration (color_change_interval : ℕ) (total_changes : ℕ) (minutes_per_hour : ℕ) :
  color_change_interval = 10 →
  total_changes = 12 →
  minutes_per_hour = 60 →
  (color_change_interval * total_changes) / minutes_per_hour = 2 :=
by
  sorry

end sunset_duration_l3936_393626


namespace sin_330_degrees_l3936_393654

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end sin_330_degrees_l3936_393654


namespace right_angle_vector_proof_l3936_393610

/-- Given two vectors OA and OB in a 2D Cartesian coordinate system, 
    where OA forms a right angle with AB, prove that the y-coordinate of OA is 5. -/
theorem right_angle_vector_proof (t : ℝ) : 
  let OA : ℝ × ℝ := (-1, t)
  let OB : ℝ × ℝ := (2, 2)
  let AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)
  (AB.1 * OB.1 + AB.2 * OB.2 = 0) → t = 5 := by
  sorry

end right_angle_vector_proof_l3936_393610


namespace power_relation_l3936_393633

theorem power_relation (x a : ℝ) (h : x^(-a) = 3) : x^(2*a) = 1/9 := by
  sorry

end power_relation_l3936_393633


namespace license_plate_increase_l3936_393617

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  new_plates / old_plates = 26^2 / 10 := by
sorry

end license_plate_increase_l3936_393617


namespace polynomial_factorization_l3936_393630

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 4) * (x^2 + 3*x + 2) + (x^2 + 4*x - 3) = 
  (x^2 + 4*x + 2) * (x^2 + 2*x + 4) := by
  sorry

end polynomial_factorization_l3936_393630


namespace intersection_parallel_line_l3936_393614

/-- The equation of a line passing through the intersection of two lines and parallel to a third line -/
theorem intersection_parallel_line (x y : ℝ) : 
  (y = 2*x + 1) → 
  (y = 3*x - 1) → 
  (∃ m : ℝ, 2*x + y - m = 0 ∧ (∀ x y : ℝ, 2*x + y - m = 0 ↔ 2*x + y - 3 = 0)) →
  (2*x + y - 9 = 0) :=
by sorry

end intersection_parallel_line_l3936_393614


namespace complex_magnitude_l3936_393615

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 + Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
sorry

end complex_magnitude_l3936_393615


namespace sqrt_nine_equals_three_l3936_393653

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by sorry

end sqrt_nine_equals_three_l3936_393653


namespace lcm_gcd_difference_nineteen_l3936_393621

theorem lcm_gcd_difference_nineteen (a b : ℕ+) :
  Nat.lcm a b - Nat.gcd a b = 19 →
  ((a = 1 ∧ b = 20) ∨ (a = 20 ∧ b = 1)) ∨
  ((a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4)) ∨
  ((a = 19 ∧ b = 38) ∨ (a = 38 ∧ b = 19)) :=
by sorry

end lcm_gcd_difference_nineteen_l3936_393621


namespace plastic_rings_weight_sum_l3936_393678

theorem plastic_rings_weight_sum :
  let orange_ring : Float := 0.08333333333333333
  let purple_ring : Float := 0.3333333333333333
  let white_ring : Float := 0.4166666666666667
  orange_ring + purple_ring + white_ring = 0.8333333333333333 := by
  sorry

end plastic_rings_weight_sum_l3936_393678


namespace selling_price_is_180_l3936_393660

/-- Calculates the selling price per machine to break even -/
def selling_price_per_machine (cost_parts : ℕ) (cost_patent : ℕ) (num_machines : ℕ) : ℕ :=
  (cost_parts + cost_patent) / num_machines

/-- Theorem: The selling price per machine is $180 -/
theorem selling_price_is_180 :
  selling_price_per_machine 3600 4500 45 = 180 := by
  sorry

end selling_price_is_180_l3936_393660


namespace quadratic_unique_solution_l3936_393623

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 18 * x + c = 0) →  -- exactly one solution
  (a + c = 26) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 13 + 2 * Real.sqrt 22 ∧ c = 13 - 2 * Real.sqrt 22) := by
sorry

end quadratic_unique_solution_l3936_393623


namespace line_separate_from_circle_l3936_393681

/-- Given a point (a, b) within the unit circle, prove that the line ax + by = 1 is separate from the circle -/
theorem line_separate_from_circle (a b : ℝ) (h : a^2 + b^2 < 1) :
  ∀ x y : ℝ, x^2 + y^2 = 1 → a*x + b*y ≠ 1 := by sorry

end line_separate_from_circle_l3936_393681


namespace physics_marks_calculation_l3936_393631

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def chemistry_marks : ℕ := 97
def biology_marks : ℕ := 95
def average_marks : ℚ := 93
def num_subjects : ℕ := 5

theorem physics_marks_calculation :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    physics_marks = 82 :=
by sorry

end physics_marks_calculation_l3936_393631


namespace button_sequence_l3936_393634

theorem button_sequence (a : ℕ → ℕ) : 
  a 1 = 1 →                 -- First term is 1
  (∀ n : ℕ, a (n + 1) = 3 * a n) →  -- Common ratio is 3
  a 6 = 243 →               -- Sixth term is 243
  a 5 = 81 :=               -- Prove fifth term is 81
by sorry

end button_sequence_l3936_393634


namespace hilt_garden_border_l3936_393648

/-- The number of rocks needed to complete the border -/
def total_rocks : ℕ := 125

/-- The number of rocks Mrs. Hilt already has -/
def current_rocks : ℕ := 64

/-- The number of additional rocks Mrs. Hilt needs -/
def additional_rocks : ℕ := total_rocks - current_rocks

theorem hilt_garden_border :
  additional_rocks = 61 := by sorry

end hilt_garden_border_l3936_393648


namespace stratified_sample_size_l3936_393682

/-- Represents a stratified sampling scenario by gender -/
structure StratifiedSample where
  total_population : ℕ
  male_population : ℕ
  male_sample : ℕ
  total_sample : ℕ

/-- Theorem stating that given the conditions, the total sample size is 36 -/
theorem stratified_sample_size
  (s : StratifiedSample)
  (h1 : s.total_population = 120)
  (h2 : s.male_population = 80)
  (h3 : s.male_sample = 24)
  (h4 : s.male_sample / s.total_sample = s.male_population / s.total_population) :
  s.total_sample = 36 := by
  sorry

#check stratified_sample_size

end stratified_sample_size_l3936_393682


namespace subset_implies_m_equals_four_l3936_393694

def A (m : ℝ) : Set ℝ := {-1, 3, m}
def B : Set ℝ := {3, 4}

theorem subset_implies_m_equals_four (m : ℝ) : B ⊆ A m → m = 4 := by
  sorry

end subset_implies_m_equals_four_l3936_393694


namespace second_friend_is_nina_l3936_393622

structure Friend where
  hasChild : Bool
  name : String
  childName : String

def isNinotchka (name : String) : Bool :=
  name = "Nina" || name = "Ninotchka"

theorem second_friend_is_nina (friend1 friend2 : Friend) :
  friend2.hasChild = true →
  friend2.childName = friend2.name →
  isNinotchka friend2.childName →
  friend2.name = "Nina" :=
by
  sorry

end second_friend_is_nina_l3936_393622


namespace product_remainder_mod_five_l3936_393640

theorem product_remainder_mod_five : (1234 * 1987 * 2013 * 2021) % 5 = 4 := by
  sorry

end product_remainder_mod_five_l3936_393640


namespace factory_uses_systematic_sampling_l3936_393641

/-- Represents a sampling method used in quality control -/
inductive SamplingMethod
| Systematic
| Random
| Stratified
| Cluster

/-- Represents a factory's product inspection process -/
structure InspectionProcess where
  productsOnConveyor : Bool
  fixedSamplingPosition : Bool
  regularInterval : Bool

/-- Determines the sampling method based on the inspection process -/
def determineSamplingMethod (process : InspectionProcess) : SamplingMethod :=
  if process.productsOnConveyor && process.fixedSamplingPosition && process.regularInterval then
    SamplingMethod.Systematic
  else
    SamplingMethod.Random  -- Default to Random for simplicity

/-- Theorem: The given inspection process uses Systematic Sampling -/
theorem factory_uses_systematic_sampling (process : InspectionProcess) 
  (h1 : process.productsOnConveyor = true)
  (h2 : process.fixedSamplingPosition = true)
  (h3 : process.regularInterval = true) :
  determineSamplingMethod process = SamplingMethod.Systematic :=
by sorry

end factory_uses_systematic_sampling_l3936_393641
