import Mathlib

namespace sector_area_l1343_134362

theorem sector_area (r : ℝ) (θ : ℝ) (h : θ = 2 * Real.pi / 3) :
  let area := (1 / 2) * r^2 * θ
  area = 3 * Real.pi :=
by
  sorry

end sector_area_l1343_134362


namespace f_strictly_increasing_l1343_134315

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x - 4

-- Theorem statement
theorem f_strictly_increasing :
  (∀ x y, x < y ∧ x < -2/3 → f x < f y) ∧
  (∀ x y, x < y ∧ 2 < x → f x < f y) :=
sorry

end f_strictly_increasing_l1343_134315


namespace rectangle_midpoint_angle_equality_l1343_134308

-- Define the rectangle ABCD
variable (A B C D : Point)

-- Define the property of being a rectangle
def is_rectangle (A B C D : Point) : Prop := sorry

-- Define the midpoint property
def is_midpoint (M A D : Point) : Prop := sorry

-- Define a point on the extension of a line segment
def on_extension (P D C : Point) : Prop := sorry

-- Define the intersection of two lines
def intersection (Q P M A C : Point) : Prop := sorry

-- Define the angle equality
def angle_eq (Q N M P : Point) : Prop := sorry

-- State the theorem
theorem rectangle_midpoint_angle_equality 
  (h_rect : is_rectangle A B C D)
  (h_midpoint_M : is_midpoint M A D)
  (h_midpoint_N : is_midpoint N B C)
  (h_extension_P : on_extension P D C)
  (h_intersection_Q : intersection Q P M A C) :
  angle_eq Q N M P :=
sorry

end rectangle_midpoint_angle_equality_l1343_134308


namespace coefficient_theorem_l1343_134347

theorem coefficient_theorem (a : ℝ) : 
  (∃ c : ℝ, c = 6 * a^2 - 15 * a + 20 ∧ c = 56) → (a = 6 ∨ a = -1) := by sorry

end coefficient_theorem_l1343_134347


namespace sufficient_not_necessary_l1343_134343

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧ (∃ a, a^2 > 1 ∧ a ≤ 1) := by
  sorry

end sufficient_not_necessary_l1343_134343


namespace chess_tournament_students_l1343_134377

/-- The number of university students in the chess tournament -/
def num_university_students : ℕ := 11

/-- The total score of the two Level 3 students -/
def level3_total_score : ℚ := 13/2

/-- Represents the chess tournament setup -/
structure ChessTournament where
  num_students : ℕ
  student_score : ℚ
  level3_score : ℚ

/-- Calculates the total number of games played in the tournament -/
def total_games (t : ChessTournament) : ℚ :=
  (t.num_students + 2) * (t.num_students + 1) / 2

/-- Calculates the total score of all games in the tournament -/
def total_score (t : ChessTournament) : ℚ :=
  t.num_students * t.student_score + t.level3_score

/-- Theorem stating that the number of university students in the tournament is 11 -/
theorem chess_tournament_students :
  ∃ (t : ChessTournament),
    t.num_students = num_university_students ∧
    t.level3_score = level3_total_score ∧
    total_score t = total_games t :=
by sorry

end chess_tournament_students_l1343_134377


namespace water_addition_proof_l1343_134335

/-- Proves that adding 3 litres of water to 11 litres of 42% alcohol solution results in 33% alcohol mixture -/
theorem water_addition_proof (initial_volume : ℝ) (initial_alcohol_percent : ℝ) 
  (final_alcohol_percent : ℝ) (water_added : ℝ) : 
  initial_volume = 11 →
  initial_alcohol_percent = 0.42 →
  final_alcohol_percent = 0.33 →
  water_added = 3 →
  initial_volume * initial_alcohol_percent = 
    (initial_volume + water_added) * final_alcohol_percent := by
  sorry

#check water_addition_proof

end water_addition_proof_l1343_134335


namespace right_triangle_area_l1343_134306

theorem right_triangle_area (a b : ℝ) (h1 : a = 30) (h2 : b = 34) : 
  (1/2) * a * b = 510 := by
  sorry

end right_triangle_area_l1343_134306


namespace ping_pong_ball_probability_l1343_134374

def is_multiple (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def count_multiples (upper_bound divisor : ℕ) : ℕ :=
  (upper_bound / divisor)

theorem ping_pong_ball_probability :
  let total_balls : ℕ := 75
  let multiples_of_6 := count_multiples total_balls 6
  let multiples_of_8 := count_multiples total_balls 8
  let multiples_of_24 := count_multiples total_balls 24
  let favorable_outcomes := multiples_of_6 + multiples_of_8 - multiples_of_24
  (favorable_outcomes : ℚ) / total_balls = 6 / 25 := by
  sorry

end ping_pong_ball_probability_l1343_134374


namespace a_gt_b_necessary_not_sufficient_for_ac2_gt_bc2_l1343_134378

theorem a_gt_b_necessary_not_sufficient_for_ac2_gt_bc2 :
  (∃ (a b c : ℝ), a > b ∧ a * c^2 ≤ b * c^2) ∧
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) :=
sorry

end a_gt_b_necessary_not_sufficient_for_ac2_gt_bc2_l1343_134378


namespace cubic_equation_root_sum_l1343_134314

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0 and roots 4 and -1, prove b+c = -13a -/
theorem cubic_equation_root_sum (a b c d : ℝ) (ha : a ≠ 0) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 4 ∨ x = -1 ∨ x = -(b + c + 13 * a) / a) →
  b + c = -13 * a :=
by sorry

end cubic_equation_root_sum_l1343_134314


namespace solution_set_f_max_value_g_l1343_134337

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - 2

-- Define the function g
def g (x : ℝ) : ℝ := |x + 3| - |2*x - 1| - 2

-- Theorem 1: Solution set of f(x) < |x-1|
theorem solution_set_f (x : ℝ) : f x < |x - 1| ↔ x < 0 := by sorry

-- Theorem 2: Maximum value of g(x)
theorem max_value_g : ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 3/2 := by sorry

end solution_set_f_max_value_g_l1343_134337


namespace geometric_sequence_fourth_term_l1343_134333

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h1 : a 1 + a 2 = -1)
  (h2 : a 1 - a 3 = -3) :
  a 4 = -8 := by
  sorry


end geometric_sequence_fourth_term_l1343_134333


namespace rectangle_area_is_six_l1343_134327

/-- The quadratic equation representing the sides of the rectangle -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 6 = 0

/-- The roots of the quadratic equation -/
def roots : Set ℝ := {x : ℝ | quadratic_equation x}

/-- The rectangle with sides equal to the roots of the quadratic equation -/
structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  side1_root : quadratic_equation side1
  side2_root : quadratic_equation side2
  different_sides : side1 ≠ side2

/-- The area of the rectangle -/
def area (rect : Rectangle) : ℝ := rect.side1 * rect.side2

/-- Theorem: The area of the rectangle is 6 -/
theorem rectangle_area_is_six (rect : Rectangle) : area rect = 6 := by
  sorry

end rectangle_area_is_six_l1343_134327


namespace team_formation_count_l1343_134319

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of female teachers -/
def num_female : ℕ := 4

/-- The number of male teachers -/
def num_male : ℕ := 5

/-- The total number of teachers to be selected -/
def team_size : ℕ := 3

theorem team_formation_count : 
  choose num_female 1 * choose num_male 2 + choose num_female 2 * choose num_male 1 = 70 := by
  sorry

end team_formation_count_l1343_134319


namespace smallest_base_for_100_l1343_134336

theorem smallest_base_for_100 :
  ∃ (b : ℕ), b = 5 ∧ b^2 ≤ 100 ∧ 100 < b^3 ∧ ∀ (x : ℕ), x < b → (x^2 ≤ 100 → 100 ≥ x^3) :=
by sorry

end smallest_base_for_100_l1343_134336


namespace max_projection_length_l1343_134386

noncomputable section

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def curve (x : ℝ) : ℝ × ℝ := (x, x^2 + 1)

def projection_length (P : ℝ × ℝ) : ℝ :=
  let OA := A - O
  let OP := P - O
  abs (OA.1 * OP.1 + OA.2 * OP.2) / Real.sqrt (OP.1^2 + OP.2^2)

theorem max_projection_length :
  ∃ (max_length : ℝ), max_length = Real.sqrt 5 / 5 ∧
    ∀ (x : ℝ), projection_length (curve x) ≤ max_length :=
sorry

end

end max_projection_length_l1343_134386


namespace fraction_of_fraction_one_sixth_of_three_fourths_l1343_134383

theorem fraction_of_fraction (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem one_sixth_of_three_fourths :
  (1 / 6) / (3 / 4) = 2 / 9 :=
by sorry

end fraction_of_fraction_one_sixth_of_three_fourths_l1343_134383


namespace lucas_sum_is_19_89_l1343_134376

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Function to get the nth digit of a Lucas number (with overlapping) -/
def lucasDigit (n : ℕ) : ℕ :=
  lucas n % 10

/-- The infinite sum of Lucas digits divided by increasing powers of 10 -/
noncomputable def r : ℚ :=
  ∑' n, (lucasDigit n : ℚ) / 10^(n + 1)

/-- Main theorem: The sum of Lucas digits is equal to 19/89 -/
theorem lucas_sum_is_19_89 : r = 19 / 89 := by
  sorry

end lucas_sum_is_19_89_l1343_134376


namespace cat_addition_l1343_134351

/-- Proves that buying more cats results in the correct total number of cats. -/
theorem cat_addition (initial_cats bought_cats : ℕ) :
  initial_cats = 11 →
  bought_cats = 43 →
  initial_cats + bought_cats = 54 := by
  sorry

end cat_addition_l1343_134351


namespace percentage_of_girls_taking_lunch_l1343_134300

theorem percentage_of_girls_taking_lunch (total : ℕ) (boys girls : ℕ) 
  (h_ratio : boys = 3 * girls / 2)
  (h_total : total = boys + girls)
  (boys_lunch : ℕ) (total_lunch : ℕ)
  (h_boys_lunch : boys_lunch = 3 * boys / 5)
  (h_total_lunch : total_lunch = 13 * total / 25) :
  (total_lunch - boys_lunch) * 5 = 2 * girls := by
  sorry

end percentage_of_girls_taking_lunch_l1343_134300


namespace max_d_l1343_134388

/-- The sequence b_n defined as (7^n - 4) / 3 -/
def b (n : ℕ) : ℤ := (7^n - 4) / 3

/-- The greatest common divisor of b_n and b_{n+1} -/
def d' (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1)))

/-- The maximum value of d'_n is 3 for all natural numbers n -/
theorem max_d'_is_3 : ∀ n : ℕ, d' n = 3 := by sorry

end max_d_l1343_134388


namespace polynomial_lower_bound_l1343_134395

theorem polynomial_lower_bound (x : ℝ) : x^4 - 4*x^3 + 8*x^2 - 8*x + 5 ≥ 2 := by
  sorry

end polynomial_lower_bound_l1343_134395


namespace swimming_club_van_capacity_l1343_134398

/-- Calculates the maximum capacity of each van given the conditions of the swimming club problem --/
theorem swimming_club_van_capacity 
  (num_cars : ℕ) 
  (num_vans : ℕ) 
  (people_per_car : ℕ) 
  (people_per_van : ℕ) 
  (max_car_capacity : ℕ) 
  (additional_capacity : ℕ) 
  (h1 : num_cars = 2)
  (h2 : num_vans = 3)
  (h3 : people_per_car = 5)
  (h4 : people_per_van = 3)
  (h5 : max_car_capacity = 6)
  (h6 : additional_capacity = 17) :
  (num_cars * max_car_capacity + num_vans * 
    ((num_cars * people_per_car + num_vans * people_per_van + additional_capacity) / num_vans - 
     num_cars * max_car_capacity / num_vans)) / num_vans = 8 := by
  sorry

#check swimming_club_van_capacity

end swimming_club_van_capacity_l1343_134398


namespace fraction_not_zero_l1343_134382

theorem fraction_not_zero (x : ℝ) (h : x ≠ 1) : 1 / (x - 1) ≠ 0 := by
  sorry

end fraction_not_zero_l1343_134382


namespace max_value_of_z_minus_i_l1343_134309

theorem max_value_of_z_minus_i (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z - Complex.I) ≤ 2 ∧ ∃ w : ℂ, Complex.abs w = 1 ∧ Complex.abs (w - Complex.I) = 2 := by
  sorry

end max_value_of_z_minus_i_l1343_134309


namespace certain_amount_calculation_l1343_134384

theorem certain_amount_calculation (A : ℝ) : 
  (0.65 * 150 = 0.20 * A) → A = 487.5 := by
  sorry

end certain_amount_calculation_l1343_134384


namespace magic_trick_possible_l1343_134387

-- Define a coin as either Heads or Tails
inductive Coin : Type
| Heads : Coin
| Tails : Coin

-- Define a row of 27 coins
def CoinRow : Type := Fin 27 → Coin

-- Define a function to group coins into triplets
def groupIntoTriplets (row : CoinRow) : Fin 9 → Fin 3 → Coin :=
  fun i j => row (3 * i + j)

-- Define a strategy for the assistant to uncover 5 coins
def assistantStrategy (row : CoinRow) : Fin 5 → Fin 27 :=
  sorry

-- Define a strategy for the magician to identify 5 more coins
def magicianStrategy (row : CoinRow) (uncovered : Fin 5 → Fin 27) : Fin 5 → Fin 27 :=
  sorry

-- The main theorem
theorem magic_trick_possible (row : CoinRow) :
  ∃ (uncovered : Fin 5 → Fin 27) (identified : Fin 5 → Fin 27),
    (∀ i : Fin 5, row (uncovered i) = row (uncovered 0)) ∧
    (∀ i : Fin 5, row (identified i) = row (uncovered 0)) ∧
    (∀ i j : Fin 5, uncovered i ≠ identified j) :=
  sorry

end magic_trick_possible_l1343_134387


namespace spinner_probability_l1343_134346

theorem spinner_probability (pA pB pC pD : ℚ) : 
  pA = 1/4 → pB = 1/3 → pD = 1/6 → pA + pB + pC + pD = 1 → pC = 1/4 := by
  sorry

end spinner_probability_l1343_134346


namespace x_minus_y_value_l1343_134302

theorem x_minus_y_value (x y : ℝ) 
  (h1 : |x| = 3)
  (h2 : y^2 = 1/4)
  (h3 : x + y < 0) :
  x - y = -7/2 ∨ x - y = -5/2 :=
sorry

end x_minus_y_value_l1343_134302


namespace unique_common_root_value_l1343_134341

theorem unique_common_root_value (m : ℝ) : 
  m > 5 →
  (∃! x : ℝ, x^2 - 5*x + 6 = 0 ∧ x^2 + 2*x - 2*m + 1 = 0) →
  m = 8 := by
sorry

end unique_common_root_value_l1343_134341


namespace mica_shopping_cost_l1343_134375

/-- The total cost of Mica's grocery shopping --/
def total_cost (pasta_price : ℝ) (pasta_quantity : ℝ) 
               (beef_price : ℝ) (beef_quantity : ℝ)
               (sauce_price : ℝ) (sauce_quantity : ℕ)
               (quesadilla_price : ℝ) : ℝ :=
  pasta_price * pasta_quantity + 
  beef_price * beef_quantity + 
  sauce_price * (sauce_quantity : ℝ) + 
  quesadilla_price

/-- Theorem stating that the total cost of Mica's shopping is $15 --/
theorem mica_shopping_cost : 
  total_cost 1.5 2 8 (1/4) 2 2 6 = 15 := by
  sorry

end mica_shopping_cost_l1343_134375


namespace logarithm_identity_l1343_134316

theorem logarithm_identity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha_ne_one : a ≠ 1) (hb_ne_one : b ≠ 1) : 
  Real.log c / Real.log (a * b) = (Real.log c / Real.log a * Real.log c / Real.log b) / 
    (Real.log c / Real.log a + Real.log c / Real.log b) := by
  sorry

end logarithm_identity_l1343_134316


namespace quadratic_inequality_solution_set_l1343_134311

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | a * x^2 - (a + 2) * x + 2 < 0}
  (a = 0 → solution_set = {x | x > 1}) ∧
  (0 < a ∧ a < 2 → solution_set = {x | 1 < x ∧ x < 2/a}) ∧
  (a = 2 → solution_set = ∅) ∧
  (a > 2 → solution_set = {x | 2/a < x ∧ x < 1}) ∧
  (a < 0 → solution_set = {x | x < 2/a ∨ x > 1}) :=
by sorry

end quadratic_inequality_solution_set_l1343_134311


namespace yadav_yearly_savings_yadav_savings_l1343_134301

/-- Mr. Yadav's monthly salary savings calculation --/
theorem yadav_yearly_savings (monthly_salary : ℝ) 
  (h1 : monthly_salary * 0.2 = 4038) : 
  monthly_salary * 0.2 * 12 = 48456 := by
  sorry

/-- Main theorem: Mr. Yadav's yearly savings --/
theorem yadav_savings : ∃ (monthly_salary : ℝ), 
  monthly_salary * 0.2 = 4038 ∧ 
  monthly_salary * 0.2 * 12 = 48456 := by
  sorry

end yadav_yearly_savings_yadav_savings_l1343_134301


namespace binomial_coefficient_20_19_l1343_134393

theorem binomial_coefficient_20_19 : (Nat.choose 20 19) = 20 := by
  sorry

end binomial_coefficient_20_19_l1343_134393


namespace ophelia_age_proof_l1343_134391

/-- Represents the current year -/
def currentYear : ℕ := 2022

/-- Represents the future year when ages are compared -/
def futureYear : ℕ := 2030

/-- Represents the current age of Lennon -/
def lennonAge : ℝ := 15 - (futureYear - currentYear)

/-- Represents the current age of Mike -/
def mikeAge : ℝ := lennonAge + 5

/-- Represents the current age of Ophelia -/
def opheliaAge : ℝ := 20.5

theorem ophelia_age_proof :
  /- In 15 years, Ophelia will be 3.5 times as old as Lennon -/
  opheliaAge + 15 = 3.5 * (lennonAge + 15) ∧
  /- In 15 years, Mike will be twice as old as the age difference between Ophelia and Lennon -/
  mikeAge + 15 = 2 * (opheliaAge - lennonAge) ∧
  /- In 15 years, JB will be 0.75 times as old as the sum of Ophelia's and Lennon's age -/
  mikeAge + 15 = 0.75 * (opheliaAge + lennonAge + 30) :=
by sorry

end ophelia_age_proof_l1343_134391


namespace tangent_slopes_sum_l1343_134355

/-- Parabola P with equation y = (x-3)^2 + 2 -/
def P : ℝ → ℝ := λ x ↦ (x - 3)^2 + 2

/-- Point Q -/
def Q : ℝ × ℝ := (15, 7)

/-- The sum of the slopes of the two tangent lines from Q to P is 48 -/
theorem tangent_slopes_sum : 
  ∃ (r s : ℝ), (∀ m : ℝ, (r < m ∧ m < s) ↔ 
    ∀ x : ℝ, P x ≠ (m * (x - Q.1) + Q.2)) ∧ r + s = 48 := by
  sorry

end tangent_slopes_sum_l1343_134355


namespace smallest_number_of_eggs_l1343_134349

theorem smallest_number_of_eggs : ∃ (n : ℕ), 
  (n > 150) ∧ 
  (∃ (c : ℕ), n = 18 * c - 7) ∧ 
  (∀ m : ℕ, (m > 150) ∧ (∃ (d : ℕ), m = 18 * d - 7) → m ≥ n) ∧ 
  n = 155 :=
by sorry

end smallest_number_of_eggs_l1343_134349


namespace money_division_l1343_134397

/-- Proves that given a sum of money divided between two people x and y in the ratio 2:8,
    where x receives $1000, the total amount of money is $5000. -/
theorem money_division (x y total : ℕ) : 
  x + y = total → 
  x = 1000 → 
  2 * total = 10 * x → 
  total = 5000 := by
sorry

end money_division_l1343_134397


namespace sqrt_solution_l1343_134350

theorem sqrt_solution (x : ℝ) (h : x > 0) : 
  let y : ℝ → ℝ := λ x => Real.sqrt x
  2 * y x * (deriv y x) = 1 := by
sorry

end sqrt_solution_l1343_134350


namespace inequality_proof_l1343_134379

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + b^2 - Real.sqrt 2 * a * b) + Real.sqrt (b^2 + c^2 - Real.sqrt 2 * b * c) ≥ Real.sqrt (a^2 + c^2) := by
  sorry

end inequality_proof_l1343_134379


namespace sum_a_b_equals_six_l1343_134339

theorem sum_a_b_equals_six (a b : ℝ) 
  (eq1 : 3 * a + 5 * b = 22) 
  (eq2 : 4 * a + 2 * b = 20) : 
  a + b = 6 := by
sorry

end sum_a_b_equals_six_l1343_134339


namespace second_stop_off_count_l1343_134353

/-- Represents the number of passengers on the bus after each stop -/
def passengers : List ℕ := [0, 7, 0, 11]

/-- Represents the number of people getting on at each stop -/
def people_on : List ℕ := [7, 5, 4]

/-- Represents the number of people getting off at each stop -/
def people_off : List ℕ := [0, 0, 2]

/-- The unknown number of people who got off at the second stop -/
def x : ℕ := sorry

theorem second_stop_off_count :
  x = 3 ∧
  passengers[3] = passengers[1] + people_on[1] - x + people_on[2] - people_off[2] :=
by sorry

end second_stop_off_count_l1343_134353


namespace ellipse_foci_coordinates_l1343_134399

theorem ellipse_foci_coordinates :
  let ellipse := fun (x y : ℝ) => x^2 / 25 + y^2 / 169 = 1
  let a := Real.sqrt 169
  let b := 5
  let c := Real.sqrt (a^2 - b^2)
  (∀ x y, ellipse x y ↔ x^2 / a^2 + y^2 / b^2 = 1) →
  (∀ x y, ellipse x y → x^2 / a^2 + y^2 / b^2 ≤ 1) →
  ({(0, c), (0, -c)} : Set (ℝ × ℝ)) = {p | ∃ x y, ellipse x y ∧ (x - p.1)^2 + (y - p.2)^2 = a^2} :=
by sorry

end ellipse_foci_coordinates_l1343_134399


namespace e_pi_third_in_first_quadrant_l1343_134305

-- Define Euler's formula
axiom euler_formula (x : ℝ) : Complex.exp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x

-- Define the first quadrant
def first_quadrant (z : ℂ) : Prop := 0 < z.re ∧ 0 < z.im

-- Theorem statement
theorem e_pi_third_in_first_quadrant :
  first_quadrant (Complex.exp (Complex.I * (π / 3))) :=
sorry

end e_pi_third_in_first_quadrant_l1343_134305


namespace exponential_function_point_l1343_134372

/-- Given a function f(x) = a^(x-m) + n - 3, where a > 0 and a ≠ 1,
    if f(3) = 2, then m + n = 7 -/
theorem exponential_function_point (a m n : ℝ) : 
  a > 0 → a ≠ 1 → (fun x => a^(x - m) + n - 3) 3 = 2 → m + n = 7 := by
  sorry

end exponential_function_point_l1343_134372


namespace sum_of_roots_cubic_equation_l1343_134323

theorem sum_of_roots_cubic_equation :
  let f : ℝ → ℝ := λ x => 6 * x^3 - 3 * x^2 - 18 * x + 9
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ + r₂ + r₃ = 0.5 :=
sorry

end sum_of_roots_cubic_equation_l1343_134323


namespace pond_length_l1343_134359

/-- Given a rectangular field and a square pond, prove the length of the pond. -/
theorem pond_length (field_length field_width pond_area_ratio : ℝ) 
  (h1 : field_length = 96)
  (h2 : field_width = 48)
  (h3 : field_length = 2 * field_width)
  (h4 : pond_area_ratio = 1 / 72) : 
  Real.sqrt (pond_area_ratio * field_length * field_width) = 8 := by
  sorry

end pond_length_l1343_134359


namespace alicia_tax_deduction_l1343_134389

/-- Represents Alicia's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℚ := 2 / 100

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tax amount in cents -/
def tax_amount_cents : ℚ := dollars_to_cents (hourly_wage * tax_rate)

theorem alicia_tax_deduction :
  tax_amount_cents = 50 := by sorry

end alicia_tax_deduction_l1343_134389


namespace problem_solution_l1343_134307

theorem problem_solution (x y : ℚ) : 
  x = 103 → x^3 * y - 4 * x^2 * y + 4 * x * y = 1106600 → y = 1085/1030 := by
  sorry

end problem_solution_l1343_134307


namespace billys_restaurant_bill_l1343_134390

/-- The total bill for a group at Billy's Restaurant -/
def total_bill (num_adults : ℕ) (num_children : ℕ) (cost_per_meal : ℕ) : ℕ :=
  (num_adults + num_children) * cost_per_meal

/-- Theorem: The bill for 2 adults and 5 children with meals costing $3 each is $21 -/
theorem billys_restaurant_bill :
  total_bill 2 5 3 = 21 := by
  sorry

end billys_restaurant_bill_l1343_134390


namespace perfect_square_condition_l1343_134322

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, 2^n.val + 12^n.val + 2011^n.val = m^2) ↔ n = 1 := by
  sorry

end perfect_square_condition_l1343_134322


namespace total_camp_attendance_l1343_134361

def lawrence_county_camp : ℕ := 34044
def lawrence_county_home : ℕ := 134867
def outside_county_camp : ℕ := 424944

theorem total_camp_attendance : 
  lawrence_county_camp + outside_county_camp = 459988 := by
  sorry

end total_camp_attendance_l1343_134361


namespace english_class_grouping_l1343_134352

/-- The maximum number of groups that can be formed with equal composition -/
def maxGroups (boys girls : ℕ) : ℕ := Nat.gcd boys girls

/-- The problem statement -/
theorem english_class_grouping (boys girls : ℕ) 
  (h_boys : boys = 10) 
  (h_girls : girls = 15) : 
  maxGroups boys girls = 5 := by
  sorry

end english_class_grouping_l1343_134352


namespace chocolate_gain_percent_l1343_134338

/-- Calculates the gain percent given the cost price and selling price ratio -/
theorem chocolate_gain_percent 
  (cost_price selling_price : ℝ) 
  (h : 81 * cost_price = 45 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 80 := by
sorry

end chocolate_gain_percent_l1343_134338


namespace second_prime_range_l1343_134317

theorem second_prime_range (p q : ℕ) (hp : Prime p) (hq : Prime q) : 
  15 < p * q ∧ p * q ≤ 36 → 2 < p ∧ p < 6 → p * q = 33 → q = 11 := by
  sorry

end second_prime_range_l1343_134317


namespace product_at_one_zeros_of_h_monotonicity_of_h_l1343_134394

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 4
def g (x : ℝ) : ℝ := -x + 4

-- Define the product function h
def h (x : ℝ) : ℝ := f x * g x

-- Theorem 1: f(1) * g(1) = -6
theorem product_at_one : h 1 = -6 := by sorry

-- Theorem 2: The zeros of h are x = 2 and x = 4
theorem zeros_of_h : ∀ x : ℝ, h x = 0 ↔ x = 2 ∨ x = 4 := by sorry

-- Theorem 3: h is increasing on (-∞, 3] and decreasing on [3, ∞)
theorem monotonicity_of_h :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 3 → h x ≤ h y) ∧
  (∀ x y : ℝ, 3 ≤ x ∧ x ≤ y → h x ≥ h y) := by sorry

end product_at_one_zeros_of_h_monotonicity_of_h_l1343_134394


namespace trig_comparison_l1343_134385

open Real

theorem trig_comparison : 
  sin (π/5) = sin (4*π/5) ∧ cos (π/5) > cos (4*π/5) := by
  have h1 : 0 < π/5 ∧ π/5 < 4*π/5 ∧ 4*π/5 < π := by sorry
  have h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π → cos x > cos y := by sorry
  sorry

end trig_comparison_l1343_134385


namespace event_popularity_order_l1343_134371

/-- Represents an event in the school carnival --/
inductive Event
  | dodgeball
  | karaoke
  | magicShow
  | quizBowl

/-- The fraction of students liking each event --/
def eventPopularity : Event → Rat
  | Event.dodgeball => 13 / 40
  | Event.karaoke => 9 / 30
  | Event.magicShow => 17 / 60
  | Event.quizBowl => 23 / 120

/-- Theorem stating the correct order of events from most to least popular --/
theorem event_popularity_order :
  eventPopularity Event.dodgeball > eventPopularity Event.karaoke ∧
  eventPopularity Event.karaoke > eventPopularity Event.magicShow ∧
  eventPopularity Event.magicShow > eventPopularity Event.quizBowl :=
by sorry

end event_popularity_order_l1343_134371


namespace car_speed_when_serviced_l1343_134381

/-- Proves that the speed of a car when serviced is 110 km/h, given the conditions of the problem -/
theorem car_speed_when_serviced 
  (speed_not_serviced : ℝ) 
  (time_serviced : ℝ) 
  (time_not_serviced : ℝ) 
  (h1 : speed_not_serviced = 55)
  (h2 : time_serviced = 3)
  (h3 : time_not_serviced = 6)
  (h4 : speed_not_serviced * time_not_serviced = speed_when_serviced * time_serviced) :
  speed_when_serviced = 110 := by
  sorry

#check car_speed_when_serviced

end car_speed_when_serviced_l1343_134381


namespace initial_markup_percentage_l1343_134321

theorem initial_markup_percentage
  (initial_price : ℝ)
  (price_increase : ℝ)
  (h1 : initial_price = 34)
  (h2 : price_increase = 6)
  (h3 : initial_price + price_increase = 2 * (initial_price - (initial_price + price_increase) / 2)) :
  (initial_price - (initial_price + price_increase) / 2) / ((initial_price + price_increase) / 2) = 0.7 :=
by sorry

end initial_markup_percentage_l1343_134321


namespace usual_time_to_catch_bus_l1343_134313

/-- The usual time to catch the bus, given that walking with 4/5 of the usual speed
    results in missing the bus by 4 minutes, is 16 minutes. -/
theorem usual_time_to_catch_bus (T : ℝ) (S : ℝ) : T > 0 → S > 0 → S * T = (4/5 * S) * (T + 4) → T = 16 := by
  sorry

end usual_time_to_catch_bus_l1343_134313


namespace triangular_pyramid_not_circular_top_view_l1343_134364

-- Define the types of solids
inductive Solid
  | Sphere
  | Cylinder
  | Cone
  | TriangularPyramid

-- Define a property for having a circular top view
def has_circular_top_view (s : Solid) : Prop :=
  match s with
  | Solid.Sphere => True
  | Solid.Cylinder => True
  | Solid.Cone => True
  | Solid.TriangularPyramid => False

-- Theorem statement
theorem triangular_pyramid_not_circular_top_view :
  ∀ s : Solid, ¬(has_circular_top_view s) ↔ s = Solid.TriangularPyramid :=
by sorry

end triangular_pyramid_not_circular_top_view_l1343_134364


namespace consecutive_sum_100_l1343_134334

theorem consecutive_sum_100 (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) → n = 18 := by
  sorry

end consecutive_sum_100_l1343_134334


namespace system_of_equations_sum_l1343_134357

theorem system_of_equations_sum (x y : ℝ) :
  3 * x + 2 * y = 10 →
  2 * x + 3 * y = 5 →
  x + y = 3 := by
sorry

end system_of_equations_sum_l1343_134357


namespace scooter_initial_cost_l1343_134354

theorem scooter_initial_cost (P : ℝ) : 
  (P + 300) * 1.1 = 1320 → P = 900 := by sorry

end scooter_initial_cost_l1343_134354


namespace student_average_greater_than_actual_average_l1343_134304

theorem student_average_greater_than_actual_average (x y z : ℝ) (h : x < y ∧ y < z) :
  (x + y + 2 * z) / 4 > (x + y + z) / 3 := by
  sorry

end student_average_greater_than_actual_average_l1343_134304


namespace airport_visit_total_l1343_134345

theorem airport_visit_total (first_graders : ℕ) (second_graders_difference : ℕ) : 
  first_graders = 358 →
  second_graders_difference = 64 →
  first_graders + (first_graders - second_graders_difference) = 652 :=
by sorry

end airport_visit_total_l1343_134345


namespace num_distinguishable_triangles_l1343_134396

/-- Represents the number of available colors for triangles -/
def numColors : ℕ := 8

/-- Represents the number of corner triangles in the large triangle -/
def numCorners : ℕ := 3

/-- Represents the number of triangles between center and corner -/
def numBetween : ℕ := 1

/-- Represents the number of center triangles -/
def numCenter : ℕ := 1

/-- Calculates the number of ways to choose corner colors -/
def cornerColorings : ℕ := 
  numColors + (numColors.choose 1 * (numColors - 1).choose 1) + numColors.choose numCorners

/-- Theorem: The number of distinguishable large equilateral triangles is 7680 -/
theorem num_distinguishable_triangles : 
  cornerColorings * numColors^(numBetween + numCenter) = 7680 := by sorry

end num_distinguishable_triangles_l1343_134396


namespace hexagon_arithmetic_progression_angle_l1343_134360

/-- A hexagon with angles in arithmetic progression has one angle of 114 degrees. -/
theorem hexagon_arithmetic_progression_angle (a d : ℝ) : 
  (6 * a + 15 * d = 720) →  -- Sum of angles in hexagon
  (∃ k : ℕ, k < 6 ∧ a + k * d = 114) :=  -- One angle is 114 degrees
by sorry

end hexagon_arithmetic_progression_angle_l1343_134360


namespace expression_equality_l1343_134342

theorem expression_equality : 484 + 2 * 22 * 5 + 25 = 729 := by
  sorry

end expression_equality_l1343_134342


namespace white_balls_count_l1343_134331

/-- Represents a box of balls with white and black colors. -/
structure BallBox where
  total : ℕ
  white : ℕ
  black : ℕ
  sum_correct : white + black = total
  white_condition : ∀ (n : ℕ), n ≥ 12 → n.choose white > 0
  black_condition : ∀ (n : ℕ), n ≥ 20 → n.choose black > 0

/-- Theorem stating that a box with 30 balls satisfying the given conditions has 19 white balls. -/
theorem white_balls_count (box : BallBox) (h_total : box.total = 30) : box.white = 19 := by
  sorry

end white_balls_count_l1343_134331


namespace y_value_l1343_134325

theorem y_value (y : ℕ) (h1 : ∃ k : ℕ, y = 9 * k) (h2 : y^2 > 200) (h3 : y < 30) : y = 18 := by
  sorry

end y_value_l1343_134325


namespace weighted_coin_probability_l1343_134392

/-- Represents the weighting of the coin -/
inductive CoinWeight
| Heads
| Tails

/-- The probability of getting heads given the coin's weight -/
def prob_heads (w : CoinWeight) : ℚ :=
  match w with
  | CoinWeight.Heads => 2/3
  | CoinWeight.Tails => 1/3

/-- The probability of getting the observed result (two heads) given the coin's weight -/
def prob_observed (w : CoinWeight) : ℚ :=
  (prob_heads w) * (prob_heads w)

/-- The prior probability of each weighting -/
def prior_prob : CoinWeight → ℚ
| _ => 1/2

theorem weighted_coin_probability :
  let posterior_prob_heads := (prob_observed CoinWeight.Heads * prior_prob CoinWeight.Heads) /
    (prob_observed CoinWeight.Heads * prior_prob CoinWeight.Heads + 
     prob_observed CoinWeight.Tails * prior_prob CoinWeight.Tails)
  let prob_next_heads := posterior_prob_heads * prob_heads CoinWeight.Heads +
    (1 - posterior_prob_heads) * prob_heads CoinWeight.Tails
  prob_next_heads = 3/5 := by
  sorry

end weighted_coin_probability_l1343_134392


namespace irrational_sqrt_sin_cos_l1343_134318

theorem irrational_sqrt_sin_cos (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  ¬(∃ (a b c d : ℤ), b ≠ 0 ∧ d ≠ 0 ∧ 
    Real.sqrt (Real.sin θ) = a / b ∧ 
    Real.sqrt (Real.cos θ) = c / d) :=
by sorry

end irrational_sqrt_sin_cos_l1343_134318


namespace marbles_taken_correct_l1343_134332

/-- The number of green marbles Mike took from Dan -/
def marbles_taken (initial_green : ℕ) (remaining_green : ℕ) : ℕ :=
  initial_green - remaining_green

/-- Theorem stating that the number of marbles Mike took is the difference between
    Dan's initial and remaining green marbles -/
theorem marbles_taken_correct (initial_green : ℕ) (remaining_green : ℕ) 
    (h : initial_green ≥ remaining_green) :
  marbles_taken initial_green remaining_green = initial_green - remaining_green :=
by
  sorry

#eval marbles_taken 32 9  -- Should output 23

end marbles_taken_correct_l1343_134332


namespace X_related_Y_probability_l1343_134366

/-- The probability of k² being greater than or equal to 10.83 under the null hypothesis -/
def p_k_squared_ge_10_83 : ℝ := 0.001

/-- The null hypothesis states that variable X is unrelated to variable Y -/
def H₀ : Prop := sorry

/-- The probability that variable X is related to variable Y -/
def p_X_related_Y : ℝ := sorry

/-- Theorem stating the relationship between p_X_related_Y and p_k_squared_ge_10_83 -/
theorem X_related_Y_probability : 
  p_X_related_Y = 1 - p_k_squared_ge_10_83 := by sorry

end X_related_Y_probability_l1343_134366


namespace root_difference_quadratic_nonnegative_difference_roots_l1343_134328

theorem root_difference_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * 1^2 + b * 1 + c = 0 → |r₁ - r₂| = Real.sqrt (b^2 - 4*a*c) / a :=
by sorry

theorem nonnegative_difference_roots :
  let f (x : ℝ) := x^2 + 34*x + 274
  ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ |r₁ - r₂| = 6 :=
by sorry

end root_difference_quadratic_nonnegative_difference_roots_l1343_134328


namespace quadratic_function_properties_l1343_134326

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- Another function defined in terms of f -/
def g (k : ℝ) (x : ℝ) : ℝ := f x - k*x

/-- Theorem stating the properties of f and g -/
theorem quadratic_function_properties :
  (∀ x, f x ≥ -1) ∧ 
  (f 2 = -1) ∧ 
  (f 1 + f 4 = 3) ∧
  (∀ k, (∀ x ∈ Set.Ioo 1 4, ∃ y ∈ Set.Ioo 1 4, g k y < g k x) ↔ 
    k ∈ Set.Iic (-2) ∪ Set.Ici 4) := by sorry


end quadratic_function_properties_l1343_134326


namespace no_functions_satisfying_condition_l1343_134368

theorem no_functions_satisfying_condition :
  ¬ (∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), x ≠ y → |f x - f y| + |g x - g y| > 1) :=
by sorry

end no_functions_satisfying_condition_l1343_134368


namespace no_prime_solution_for_equation_l1343_134320

theorem no_prime_solution_for_equation : 
  ¬∃ (x y z : ℕ), Prime x ∧ Prime y ∧ Prime z ∧ x^2 + y^3 = z^4 := by
  sorry

end no_prime_solution_for_equation_l1343_134320


namespace max_two_digit_decimals_l1343_134370

def digits : List Nat := [2, 0, 5]

def is_valid_two_digit_decimal (n : Nat) (d : Nat) : Bool :=
  n ∈ digits ∧ d ∈ digits ∧ (n ≠ 0 ∨ d ≠ 0)

def count_valid_decimals : Nat :=
  (List.filter (fun (pair : Nat × Nat) => is_valid_two_digit_decimal pair.1 pair.2)
    (List.product digits digits)).length

theorem max_two_digit_decimals : count_valid_decimals = 6 := by
  sorry

end max_two_digit_decimals_l1343_134370


namespace intersection_complement_equality_l1343_134363

open Set

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {1, 2, 3}
def B : Finset Nat := {1, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {2, 3} := by sorry

end intersection_complement_equality_l1343_134363


namespace sum_zero_in_2x2_square_l1343_134367

/-- Given a 2x2 square with numbers a, b, c, d that are pairwise distinct,
    with the sum of numbers in the first row equal to the sum of numbers in the second row,
    and the product of numbers in the first column equal to the product of numbers in the second column,
    prove that the sum of all four numbers is zero. -/
theorem sum_zero_in_2x2_square (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (row_sum : a + b = c + d)
  (col_prod : a * c = b * d) :
  a + b + c + d = 0 := by
sorry

end sum_zero_in_2x2_square_l1343_134367


namespace root_conditions_imply_a_range_l1343_134329

/-- Given a quadratic function f(x) = x² + (a² - 1)x + (a - 2) where 'a' is a real number,
    if one root of f(x) is greater than 1 and the other root is less than 1,
    then 'a' is in the open interval (-2, 1). -/
theorem root_conditions_imply_a_range (a : ℝ) :
  let f := fun x : ℝ => x^2 + (a^2 - 1)*x + (a - 2)
  (∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ > 1 ∧ r₂ < 1) →
  -2 < a ∧ a < 1 := by
  sorry

end root_conditions_imply_a_range_l1343_134329


namespace spring_sales_l1343_134310

/-- Represents the sales data for a fast food chain's hamburger sales across seasons --/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The total sales is the sum of sales from all seasons --/
def totalSales (s : SeasonalSales) : ℝ :=
  s.spring + s.summer + s.fall + s.winter

/-- Given the conditions of the problem --/
theorem spring_sales (s : SeasonalSales)
    (h1 : s.summer = 6)
    (h2 : s.fall = 4)
    (h3 : s.winter = 3)
    (h4 : s.winter = 0.2 * totalSales s) :
    s.spring = 2 := by
  sorry


end spring_sales_l1343_134310


namespace unique_base6_divisible_by_13_l1343_134356

/-- Converts a base-6 number of the form 2dd3₆ to base 10 --/
def base6_to_base10 (d : Nat) : Nat :=
  2 * 6^3 + d * 6^2 + d * 6 + 3

/-- Checks if a number is divisible by 13 --/
def divisible_by_13 (n : Nat) : Prop :=
  n % 13 = 0

/-- Theorem stating that 2553₆ is divisible by 13 and is the only number of the form 2dd3₆ with this property --/
theorem unique_base6_divisible_by_13 :
  divisible_by_13 (base6_to_base10 5) ∧
  ∀ d : Nat, d < 6 → d ≠ 5 → ¬(divisible_by_13 (base6_to_base10 d)) :=
by sorry

end unique_base6_divisible_by_13_l1343_134356


namespace parallelogram_angle_measure_l1343_134324

theorem parallelogram_angle_measure (α β : ℝ) : 
  (α + β = π) →  -- Adjacent angles in a parallelogram sum to π
  (β = α + π/9) →  -- One angle exceeds the other by π/9
  (α = 4*π/9) :=  -- The smaller angle is 4π/9
by sorry

end parallelogram_angle_measure_l1343_134324


namespace trinomial_fourth_power_l1343_134365

theorem trinomial_fourth_power (a b c : ℤ) : 
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^4) → a = 0 ∧ b = 0 := by
  sorry

end trinomial_fourth_power_l1343_134365


namespace lucy_crayons_l1343_134340

/-- Given that Willy has 1400 crayons and 1110 more crayons than Lucy, 
    prove that Lucy has 290 crayons. -/
theorem lucy_crayons (willy_crayons : ℕ) (difference : ℕ) (lucy_crayons : ℕ) 
  (h1 : willy_crayons = 1400) 
  (h2 : difference = 1110) 
  (h3 : willy_crayons = lucy_crayons + difference) : 
  lucy_crayons = 290 := by
  sorry

end lucy_crayons_l1343_134340


namespace complement_is_acute_l1343_134312

-- Define an angle as a real number between 0 and 180 degrees
def Angle := {x : ℝ // 0 ≤ x ∧ x ≤ 180}

-- Define an acute angle
def isAcute (a : Angle) : Prop := a.val < 90

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  ⟨90 - a.val, by sorry⟩  -- The proof that this is a valid angle is omitted

-- Theorem statement
theorem complement_is_acute (a : Angle) (h : a.val < 90) : isAcute (complement a) := by
  sorry


end complement_is_acute_l1343_134312


namespace parallel_line_equation_l1343_134373

/-- A line parallel to y = -4x + 2023 that intersects the y-axis at (0, -5) has the equation y = -4x - 5 -/
theorem parallel_line_equation (k b : ℝ) : 
  (∀ x y, y = k * x + b ↔ y = -4 * x + 2023) →  -- parallel condition
  (b = -5) →                                   -- y-intercept condition
  (∀ x y, y = k * x + b ↔ y = -4 * x - 5) :=
by sorry

end parallel_line_equation_l1343_134373


namespace factorization_of_cyclic_expression_l1343_134358

theorem factorization_of_cyclic_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + a*b + a*c + b*c) := by
  sorry

end factorization_of_cyclic_expression_l1343_134358


namespace paper_products_pallets_l1343_134330

theorem paper_products_pallets : ∃ P : ℚ,
  P / 2 + P / 4 + P / 5 + 1 = P ∧ P = 20 := by
  sorry

end paper_products_pallets_l1343_134330


namespace max_xy_given_constraint_l1343_134348

theorem max_xy_given_constraint (x y : ℝ) : 
  x > 0 → y > 0 → x + 4 * y = 1 → x * y ≤ 1 / 16 := by
  sorry

end max_xy_given_constraint_l1343_134348


namespace spending_solution_l1343_134303

def spending_problem (n : ℚ) : Prop :=
  let after_hardware := (3/4) * n
  let after_cleaners := after_hardware - 9
  let after_grocery := (1/2) * after_cleaners
  after_grocery = 12

theorem spending_solution : 
  ∃ (n : ℚ), spending_problem n ∧ n = 44 :=
sorry

end spending_solution_l1343_134303


namespace one_third_of_one_fourth_l1343_134380

theorem one_third_of_one_fourth (n : ℝ) : (3 / 10 : ℝ) * n = 54 → (1 / 3 : ℝ) * ((1 / 4 : ℝ) * n) = 15 := by
  sorry

end one_third_of_one_fourth_l1343_134380


namespace digit_property_l1343_134344

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

def S (n : ℕ) : ℕ :=
  (digits n).sum

def P (n : ℕ) : ℕ :=
  (digits n).prod

theorem digit_property :
  ({ n : ℕ | n > 0 ∧ n = P n } = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  ({ n : ℕ | n > 0 ∧ n = S n + P n } = {19, 29, 39, 49, 59, 69, 79, 89, 99}) :=
by sorry

end digit_property_l1343_134344


namespace polynomial_nonnegative_l1343_134369

theorem polynomial_nonnegative (x : ℝ) : x^8 + x^6 - 4*x^4 + x^2 + 1 ≥ 0 := by
  sorry

end polynomial_nonnegative_l1343_134369
