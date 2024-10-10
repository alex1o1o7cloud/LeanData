import Mathlib

namespace sum_of_min_max_cubic_expression_l3627_362705

theorem sum_of_min_max_cubic_expression (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 30) :
  let f := fun (x y z w : ℝ) => 4 * (x^3 + y^3 + z^3 + w^3) - 3 * (x^2 + y^2 + z^2 + w^2)^2
  (⨅ (p : Fin 4 → ℝ) (h : p 0 + p 1 + p 2 + p 3 = 10 ∧ p 0^2 + p 1^2 + p 2^2 + p 3^2 = 30), f (p 0) (p 1) (p 2) (p 3)) +
  (⨆ (p : Fin 4 → ℝ) (h : p 0 + p 1 + p 2 + p 3 = 10 ∧ p 0^2 + p 1^2 + p 2^2 + p 3^2 = 30), f (p 0) (p 1) (p 2) (p 3)) = 404 :=
by sorry

end sum_of_min_max_cubic_expression_l3627_362705


namespace vessel_width_calculation_l3627_362709

/-- Proves that given a cube with edge length 15 cm immersed in a rectangular vessel 
    with base length 20 cm, if the water level rises by 11.25 cm, 
    then the width of the vessel's base is 15 cm. -/
theorem vessel_width_calculation (cube_edge : ℝ) (vessel_length : ℝ) (water_rise : ℝ) :
  cube_edge = 15 →
  vessel_length = 20 →
  water_rise = 11.25 →
  (cube_edge ^ 3) = (vessel_length * (cube_edge ^ 3 / (vessel_length * water_rise))) * water_rise →
  cube_edge ^ 3 / (vessel_length * water_rise) = 15 := by
  sorry

#check vessel_width_calculation

end vessel_width_calculation_l3627_362709


namespace other_divisor_proof_l3627_362728

theorem other_divisor_proof (x : ℕ) : x = 5 ↔ 
  x ≠ 11 ∧ 
  x > 0 ∧
  (386 % x = 1 ∧ 386 % 11 = 1) ∧
  ∀ y : ℕ, y < x → y ≠ 11 → y > 0 → (386 % y = 1 ∧ 386 % 11 = 1) → False :=
by sorry

end other_divisor_proof_l3627_362728


namespace quadratic_inequality_range_l3627_362751

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 < 0) → (a < -4 ∨ a > 4) := by sorry

end quadratic_inequality_range_l3627_362751


namespace house_distances_l3627_362775

-- Define the positions of houses on a straight line
variable (A B V G : ℝ)

-- Define the distances between houses
def AB := |A - B|
def VG := |V - G|
def AG := |A - G|
def BV := |B - V|

-- State the theorem
theorem house_distances (h1 : AB = 600) (h2 : VG = 600) (h3 : AG = 3 * BV) :
  AG = 900 ∨ AG = 1800 := by
  sorry

end house_distances_l3627_362775


namespace cos_90_degrees_l3627_362754

theorem cos_90_degrees : Real.cos (π / 2) = 0 := by
  sorry

end cos_90_degrees_l3627_362754


namespace parabola_coefficient_b_l3627_362773

/-- Given a parabola y = ax^2 + bx + c with vertex at (q, -q) and y-intercept at (0, q),
    where q ≠ 0, the coefficient b is equal to -4. -/
theorem parabola_coefficient_b (a b c q : ℝ) (hq : q ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - q)^2 - q) →
  (a * 0^2 + b * 0 + c = q) →
  b = -4 := by
sorry

end parabola_coefficient_b_l3627_362773


namespace emily_buys_12_cucumbers_l3627_362752

/-- The cost of one apple -/
def apple_cost : ℝ := sorry

/-- The cost of one banana -/
def banana_cost : ℝ := sorry

/-- The cost of one cucumber -/
def cucumber_cost : ℝ := sorry

/-- Six apples cost the same as three bananas -/
axiom six_apples_eq_three_bananas : 6 * apple_cost = 3 * banana_cost

/-- Three bananas cost the same as four cucumbers -/
axiom three_bananas_eq_four_cucumbers : 3 * banana_cost = 4 * cucumber_cost

/-- The number of cucumbers Emily can buy for the price of 18 apples -/
def cucumbers_for_18_apples : ℕ := sorry

/-- Proof that Emily can buy 12 cucumbers for the price of 18 apples -/
theorem emily_buys_12_cucumbers : cucumbers_for_18_apples = 12 := by
  sorry

end emily_buys_12_cucumbers_l3627_362752


namespace monomial_sum_condition_l3627_362764

/-- If the sum of two monomials 3x^5y^m and -2x^ny^7 is still a monomial in terms of x and y, 
    then m - n = 2 -/
theorem monomial_sum_condition (m n : ℕ) : 
  (∃ (a : ℚ) (p q : ℕ), 3 * X^5 * Y^m + -2 * X^n * Y^7 = a * X^p * Y^q) → 
  m - n = 2 := by
  sorry

end monomial_sum_condition_l3627_362764


namespace meal_cost_l3627_362711

theorem meal_cost (total_bill : ℝ) (tip_percentage : ℝ) (payment : ℝ) (change : ℝ) :
  total_bill = 2.5 →
  tip_percentage = 0.2 →
  payment = 20 →
  change = 5 →
  ∃ (meal_cost : ℝ), meal_cost = 12.5 ∧ meal_cost + tip_percentage * meal_cost = payment - change :=
by sorry

end meal_cost_l3627_362711


namespace system_solution_l3627_362730

theorem system_solution :
  ∃! (x y : ℝ), (x + 3 * y = 7) ∧ (x + 4 * y = 8) :=
by
  -- Proof goes here
  sorry

end system_solution_l3627_362730


namespace midpoint_vector_sum_l3627_362784

-- Define the triangle ABC and its midpoints
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
axiom D_midpoint : D = (A + B) / 2
axiom E_midpoint : E = (B + C) / 2
axiom F_midpoint : F = (C + A) / 2

-- Define vector operations
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- State the theorem
theorem midpoint_vector_sum :
  vec D B - vec D E + vec C F = vec C D :=
sorry

end midpoint_vector_sum_l3627_362784


namespace weight_of_B_l3627_362712

def weight_problem (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧
  (A + B) / 2 = 40 ∧
  (B + C) / 2 = 41 ∧
  ∃ (x : ℝ), A = 2*x ∧ B = 3*x ∧ C = 5*x ∧
  A + B + C = 144

theorem weight_of_B (A B C : ℝ) (h : weight_problem A B C) : B = 43.2 :=
sorry

end weight_of_B_l3627_362712


namespace min_liking_both_mozart_and_bach_l3627_362715

theorem min_liking_both_mozart_and_bach
  (total : ℕ)
  (like_mozart : ℕ)
  (like_bach : ℕ)
  (h_total : total = 200)
  (h_mozart : like_mozart = 160)
  (h_bach : like_bach = 150) :
  like_mozart + like_bach - total ≥ 110 :=
by sorry

end min_liking_both_mozart_and_bach_l3627_362715


namespace inheritance_distribution_l3627_362704

structure Relative where
  name : String
  amount : ℕ

structure Couple where
  husband : Relative
  wife : Relative

def total_inheritance : ℕ := 1000
def wives_total : ℕ := 396

theorem inheritance_distribution (john henry tom : Relative) (katherine jane mary : Relative) :
  john.name = "John Smith" →
  henry.name = "Henry Snooks" →
  tom.name = "Tom Crow" →
  katherine.name = "Katherine" →
  jane.name = "Jane" →
  mary.name = "Mary" →
  jane.amount = katherine.amount + 10 →
  mary.amount = jane.amount + 10 →
  katherine.amount + jane.amount + mary.amount = wives_total →
  john.amount = katherine.amount →
  henry.amount = (3 * jane.amount) / 2 →
  tom.amount = 2 * mary.amount →
  john.amount + henry.amount + tom.amount + katherine.amount + jane.amount + mary.amount = total_inheritance →
  ∃ (c1 c2 c3 : Couple),
    c1.husband = john ∧ c1.wife = katherine ∧
    c2.husband = henry ∧ c2.wife = jane ∧
    c3.husband = tom ∧ c3.wife = mary :=
by
  sorry

end inheritance_distribution_l3627_362704


namespace four_fours_exist_l3627_362736

/-- A datatype representing arithmetic expressions using only the digit 4 --/
inductive Expr4
  | four : Expr4
  | add : Expr4 → Expr4 → Expr4
  | sub : Expr4 → Expr4 → Expr4
  | mul : Expr4 → Expr4 → Expr4
  | div : Expr4 → Expr4 → Expr4

/-- Evaluate an Expr4 to a rational number --/
def eval : Expr4 → ℚ
  | Expr4.four => 4
  | Expr4.add e1 e2 => eval e1 + eval e2
  | Expr4.sub e1 e2 => eval e1 - eval e2
  | Expr4.mul e1 e2 => eval e1 * eval e2
  | Expr4.div e1 e2 => eval e1 / eval e2

/-- Count the number of 4's used in an Expr4 --/
def count_fours : Expr4 → ℕ
  | Expr4.four => 1
  | Expr4.add e1 e2 => count_fours e1 + count_fours e2
  | Expr4.sub e1 e2 => count_fours e1 + count_fours e2
  | Expr4.mul e1 e2 => count_fours e1 + count_fours e2
  | Expr4.div e1 e2 => count_fours e1 + count_fours e2

/-- Theorem stating that expressions for 2, 3, 4, 5, and 6 exist using four 4's --/
theorem four_fours_exist : 
  ∃ (e2 e3 e4 e5 e6 : Expr4), 
    (count_fours e2 = 4 ∧ eval e2 = 2) ∧
    (count_fours e3 = 4 ∧ eval e3 = 3) ∧
    (count_fours e4 = 4 ∧ eval e4 = 4) ∧
    (count_fours e5 = 4 ∧ eval e5 = 5) ∧
    (count_fours e6 = 4 ∧ eval e6 = 6) := by
  sorry

end four_fours_exist_l3627_362736


namespace mirasol_initial_balance_l3627_362742

/-- Mirasol's initial account balance -/
def initial_balance : ℕ := sorry

/-- Amount spent on coffee beans -/
def coffee_cost : ℕ := 10

/-- Amount spent on tumbler -/
def tumbler_cost : ℕ := 30

/-- Amount left in account -/
def remaining_balance : ℕ := 10

/-- Theorem: Mirasol's initial account balance was $50 -/
theorem mirasol_initial_balance :
  initial_balance = coffee_cost + tumbler_cost + remaining_balance :=
by sorry

end mirasol_initial_balance_l3627_362742


namespace timePerPlayer_is_36_l3627_362787

/-- Represents a sports tournament with given parameters -/
structure Tournament where
  teamSize : ℕ
  playersOnField : ℕ
  matchDuration : ℕ
  hTeamSize : teamSize = 10
  hPlayersOnField : playersOnField = 8
  hMatchDuration : matchDuration = 45
  hPlayersOnFieldLessTeamSize : playersOnField < teamSize

/-- Calculates the time each player spends on the field -/
def timePerPlayer (t : Tournament) : ℕ :=
  t.playersOnField * t.matchDuration / t.teamSize

/-- Theorem stating that each player spends 36 minutes on the field -/
theorem timePerPlayer_is_36 (t : Tournament) : timePerPlayer t = 36 := by
  sorry

end timePerPlayer_is_36_l3627_362787


namespace sufficient_not_necessary_condition_l3627_362767

/-- Represents a plane in 3D space -/
structure Plane where
  -- We don't need to define the internal structure of a plane for this problem

/-- Perpendicularity relation between planes -/
def perpendicular (p q : Plane) : Prop :=
  sorry

/-- Parallelism relation between planes -/
def parallel (p q : Plane) : Prop :=
  sorry

theorem sufficient_not_necessary_condition 
  (α β γ : Plane) 
  (h_different : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h_perp : perpendicular α γ) :
  (∀ (α β γ : Plane), parallel α β → perpendicular β γ) ∧
  (∃ (α β γ : Plane), perpendicular β γ ∧ ¬parallel α β) :=
sorry

end sufficient_not_necessary_condition_l3627_362767


namespace max_value_2x_plus_y_l3627_362733

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2*y ≤ 3) (h2 : x ≥ 0) (h3 : y ≥ 0) :
  (∀ x' y' : ℝ, x' + 2*y' ≤ 3 → x' ≥ 0 → y' ≥ 0 → 2*x' + y' ≤ 2*x + y) →
  2*x + y = 6 :=
by sorry

end max_value_2x_plus_y_l3627_362733


namespace speaking_orders_count_l3627_362794

/-- The number of students in the class --/
def totalStudents : ℕ := 7

/-- The number of students to be selected for speaking --/
def selectedSpeakers : ℕ := 4

/-- Function to calculate the number of speaking orders --/
def speakingOrders (total : ℕ) (selected : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of speaking orders under given conditions --/
theorem speaking_orders_count :
  speakingOrders totalStudents selectedSpeakers = 600 :=
sorry

end speaking_orders_count_l3627_362794


namespace nonconsecutive_choose_18_5_l3627_362789

def nonconsecutive_choose (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k

theorem nonconsecutive_choose_18_5 :
  nonconsecutive_choose 18 5 = Nat.choose 14 5 :=
sorry

end nonconsecutive_choose_18_5_l3627_362789


namespace fifth_number_21st_row_l3627_362778

/-- Represents the array of odd numbers -/
def oddNumberArray (row : ℕ) (position : ℕ) : ℕ :=
  2 * (row * (row - 1) / 2 + position) - 1

/-- The theorem to prove -/
theorem fifth_number_21st_row :
  oddNumberArray 21 5 = 809 :=
sorry

end fifth_number_21st_row_l3627_362778


namespace triangle_angle_value_l3627_362743

theorem triangle_angle_value (A B C : Real) : 
  -- A, B, and C are internal angles of a triangle
  A + B + C = π → 
  0 < A → 0 < B → 0 < C →
  -- Given equation
  Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin C ^ 2 + Real.sin A * Real.sin B →
  -- Conclusion
  C = π / 3 := by
sorry

end triangle_angle_value_l3627_362743


namespace candy_remaining_l3627_362790

theorem candy_remaining (initial : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) 
  (h1 : initial = 21)
  (h2 : first_eaten = 5)
  (h3 : second_eaten = 9) : 
  initial - first_eaten - second_eaten = 7 := by
  sorry

end candy_remaining_l3627_362790


namespace delta_triple_72_l3627_362760

/-- Definition of Δ function -/
def Δ (N : ℝ) : ℝ := 0.4 * N + 2

/-- Theorem stating that Δ(Δ(Δ72)) = 7.728 -/
theorem delta_triple_72 : Δ (Δ (Δ 72)) = 7.728 := by
  sorry

end delta_triple_72_l3627_362760


namespace pies_per_row_l3627_362726

theorem pies_per_row (total_pies : ℕ) (num_rows : ℕ) (h1 : total_pies = 30) (h2 : num_rows = 6) :
  total_pies / num_rows = 5 := by
sorry

end pies_per_row_l3627_362726


namespace cubic_polynomials_constant_term_l3627_362702

/-- Given two cubic polynomials p(x) and q(x) with specific root relationships,
    prove that there are only two possible values for the constant term d of p(x). -/
theorem cubic_polynomials_constant_term (c d : ℝ) : 
  (∃ (r s : ℝ), (r^3 + c*r + d = 0 ∧ s^3 + c*s + d = 0) ∧
   ((r+5)^3 + c*(r+5) + (d+210) = 0 ∧ (s-4)^3 + c*(s-4) + (d+210) = 0)) →
  (d = 240 ∨ d = 420) := by
  sorry

end cubic_polynomials_constant_term_l3627_362702


namespace average_temp_bucyrus_l3627_362749

/-- The average temperature in Bucyrus, Ohio over three days -/
def average_temperature (temp1 temp2 temp3 : ℤ) : ℚ :=
  (temp1 + temp2 + temp3) / 3

/-- Theorem stating that the average of the given temperatures is -7 -/
theorem average_temp_bucyrus :
  average_temperature (-14) (-8) 1 = -7 := by
  sorry

end average_temp_bucyrus_l3627_362749


namespace relay_race_distance_l3627_362737

theorem relay_race_distance (siwon_fraction dawon_fraction : ℚ) 
  (combined_distance : ℝ) (total_distance : ℝ) : 
  siwon_fraction = 3 / 10 →
  dawon_fraction = 4 / 10 →
  combined_distance = 140 →
  (siwon_fraction + dawon_fraction : ℝ) * total_distance = combined_distance →
  total_distance = 200 :=
by sorry

end relay_race_distance_l3627_362737


namespace nth_prime_upper_bound_and_prime_counting_lower_bound_l3627_362770

-- Define the n-th prime number
def nth_prime (n : ℕ) : ℕ := sorry

-- Define the prime counting function
def prime_counting_function (x : ℝ) : ℝ := sorry

theorem nth_prime_upper_bound_and_prime_counting_lower_bound :
  (∀ n : ℕ, nth_prime n ≤ 2^(2^n)) ∧
  (∃ c : ℝ, c > 0 ∧ ∀ x : ℝ, x > Real.exp 1 → prime_counting_function x ≥ c * Real.log (Real.log x)) :=
sorry

end nth_prime_upper_bound_and_prime_counting_lower_bound_l3627_362770


namespace pencil_count_l3627_362735

/-- Given a shop with pencils, pens, and exercise books in a ratio of 10 : 2 : 3,
    and 36 exercise books in total, prove that there are 120 pencils. -/
theorem pencil_count (ratio_pencils : ℕ) (ratio_pens : ℕ) (ratio_books : ℕ) 
    (total_books : ℕ) (h1 : ratio_pencils = 10) (h2 : ratio_pens = 2) 
    (h3 : ratio_books = 3) (h4 : total_books = 36) : 
    ratio_pencils * (total_books / ratio_books) = 120 := by
  sorry

end pencil_count_l3627_362735


namespace kelly_initial_apples_l3627_362701

theorem kelly_initial_apples (initial : ℕ) (to_pick : ℕ) (total : ℕ) 
  (h1 : to_pick = 49)
  (h2 : total = 105)
  (h3 : initial + to_pick = total) : 
  initial = 56 := by
  sorry

end kelly_initial_apples_l3627_362701


namespace average_speed_two_hours_car_average_speed_l3627_362776

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 > 0 → speed2 > 0 → (speed1 + speed2) / 2 = (speed1 * 1 + speed2 * 1) / (1 + 1) := by
  sorry

/-- The average speed of a car traveling 90 km in the first hour and 30 km in the second hour is 60 km/h -/
theorem car_average_speed : 
  let speed1 : ℝ := 90
  let speed2 : ℝ := 30
  (speed1 + speed2) / 2 = 60 := by
  sorry

end average_speed_two_hours_car_average_speed_l3627_362776


namespace equation_equivalence_l3627_362759

theorem equation_equivalence (x : ℝ) : 
  (x + 3) / 3 - (x - 1) / 6 = (5 - x) / 2 ↔ 2*x + 6 - x + 1 = 15 - 3*x := by
  sorry

end equation_equivalence_l3627_362759


namespace russian_alphabet_sum_sequence_exists_l3627_362788

theorem russian_alphabet_sum_sequence_exists : ∃ (π : Fin 33 → Fin 33), Function.Bijective π ∧
  ∀ (i j : Fin 33), i ≠ j → (π i + i : Fin 33) ≠ (π j + j : Fin 33) := by
  sorry

end russian_alphabet_sum_sequence_exists_l3627_362788


namespace base_five_to_decimal_l3627_362761

/-- Converts a list of digits in a given base to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

/-- The base 5 number 243₅ is equal to 73 in base 10 -/
theorem base_five_to_decimal : to_decimal [2, 4, 3] 5 = 73 := by
  sorry

end base_five_to_decimal_l3627_362761


namespace pascal_triangle_formula_l3627_362766

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem statement
theorem pascal_triangle_formula (n k : ℕ) (h : k ≤ n) :
  binomial_coeff n k = factorial n / (factorial k * factorial (n - k)) :=
by sorry

end pascal_triangle_formula_l3627_362766


namespace p_sufficient_not_necessary_for_q_l3627_362717

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, |2*x - 3| < 1 → x*(x - 3) < 0) ∧
  (∃ x : ℝ, x*(x - 3) < 0 ∧ ¬(|2*x - 3| < 1)) := by
  sorry

end p_sufficient_not_necessary_for_q_l3627_362717


namespace sales_tax_difference_l3627_362732

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) : 
  price = 50 → tax_rate1 = 0.075 → tax_rate2 = 0.07 →
  price * tax_rate1 - price * tax_rate2 = 0.25 := by sorry

end sales_tax_difference_l3627_362732


namespace square_plus_double_is_perfect_square_l3627_362769

theorem square_plus_double_is_perfect_square (a : ℕ) : 
  ∃ (k : ℕ), a^2 + 2*a = k^2 ↔ a = 0 := by
  sorry

end square_plus_double_is_perfect_square_l3627_362769


namespace proposition_correctness_l3627_362750

-- Define the propositions
def prop1 (a b c : ℝ) : Prop := a > b → a * c^2 > b * c^2
def prop2 (a b : ℝ) : Prop := a > |b| → a^2 > b^2
def prop3 (a b : ℝ) : Prop := |a| > b → a^2 > b^2
def prop4 (a b : ℝ) : Prop := a > b → a^3 > b^3

-- Theorem stating the correctness of propositions
theorem proposition_correctness :
  (∃ a b c : ℝ, ¬(prop1 a b c)) ∧
  (∀ a b : ℝ, prop2 a b) ∧
  (∃ a b : ℝ, ¬(prop3 a b)) ∧
  (∀ a b : ℝ, prop4 a b) :=
sorry

end proposition_correctness_l3627_362750


namespace problem_statement_l3627_362763

theorem problem_statement : 
  (Real.sqrt (5 + Real.sqrt 6) + Real.sqrt (5 - Real.sqrt 6)) / Real.sqrt (Real.sqrt 6 - 1) - Real.sqrt (4 - 2 * Real.sqrt 3) = 1 := by
  sorry

end problem_statement_l3627_362763


namespace animal_books_count_animal_books_proof_l3627_362782

def book_price : ℕ := 16
def space_books : ℕ := 1
def train_books : ℕ := 3
def total_spent : ℕ := 224

theorem animal_books_count : ℕ :=
  (total_spent - book_price * (space_books + train_books)) / book_price

#check animal_books_count

theorem animal_books_proof :
  animal_books_count = 10 :=
by sorry

end animal_books_count_animal_books_proof_l3627_362782


namespace intersection_point_translated_line_l3627_362799

/-- The intersection point of the line y = 3x + 6 with the x-axis is (-2, 0) -/
theorem intersection_point_translated_line (x y : ℝ) :
  y = 3 * x + 6 ∧ y = 0 → x = -2 ∧ y = 0 := by sorry

end intersection_point_translated_line_l3627_362799


namespace triangle_properties_l3627_362762

open Real

theorem triangle_properties (A B C a b c : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 → b > 0 → c > 0 →
  2 * a * cos C + c = 2 * b →
  a = Real.sqrt 3 →
  (1 / 2) * b * c * sin A = Real.sqrt 3 / 2 →
  A = π / 3 ∧ a + b + c = 3 + Real.sqrt 3 := by
  sorry


end triangle_properties_l3627_362762


namespace max_sum_with_gcf_six_l3627_362721

theorem max_sum_with_gcf_six (a b : ℕ) : 
  10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 →  -- a and b are two-digit positive integers
  Nat.gcd a b = 6 →                    -- greatest common factor of a and b is 6
  a + b ≤ 186 ∧                        -- upper bound
  ∃ (a' b' : ℕ), 10 ≤ a' ∧ a' ≤ 99 ∧ 10 ≤ b' ∧ b' ≤ 99 ∧ 
    Nat.gcd a' b' = 6 ∧ a' + b' = 186  -- existence of a pair that achieves the maximum
  := by sorry

end max_sum_with_gcf_six_l3627_362721


namespace inequality_range_l3627_362768

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0) ↔ k ≥ 1 := by sorry

end inequality_range_l3627_362768


namespace sqrt_twelve_is_quadratic_radical_l3627_362758

/-- Definition of a quadratic radical -/
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), y ≥ 0 ∧ x = Real.sqrt y

/-- Theorem stating that √12 is a quadratic radical -/
theorem sqrt_twelve_is_quadratic_radical :
  is_quadratic_radical (Real.sqrt 12) :=
by
  sorry


end sqrt_twelve_is_quadratic_radical_l3627_362758


namespace louises_initial_toys_l3627_362707

/-- Proves that Louise initially had 28 toys in her cart -/
theorem louises_initial_toys (initial_toy_cost : ℕ) (teddy_bear_count : ℕ) (teddy_bear_cost : ℕ) (total_cost : ℕ) :
  initial_toy_cost = 10 →
  teddy_bear_count = 20 →
  teddy_bear_cost = 15 →
  total_cost = 580 →
  ∃ (initial_toy_count : ℕ), initial_toy_count * initial_toy_cost + teddy_bear_count * teddy_bear_cost = total_cost ∧ initial_toy_count = 28 :=
by
  sorry

end louises_initial_toys_l3627_362707


namespace problem_solution_l3627_362792

theorem problem_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b * c = 1) (h5 : a + 1 / c = 7) (h6 : b + 1 / a = 34) :
  c + 1 / b = 43 / 237 := by
sorry

end problem_solution_l3627_362792


namespace range_of_a_l3627_362719

def A : Set ℝ := {x | x^2 - 5*x + 4 > 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + (a+2) = 0}

theorem range_of_a (a : ℝ) : 
  (A ∩ B a).Nonempty → a ∈ {x | x < -1 ∨ x > 18/7} := by
  sorry

end range_of_a_l3627_362719


namespace min_value_of_expression_l3627_362716

theorem min_value_of_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2*y = 1) :
  ∃ (m : ℝ), m = 3/4 ∧ ∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*y'^2 ≥ m :=
by sorry

end min_value_of_expression_l3627_362716


namespace no_integer_solutions_l3627_362724

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), 
    x^6 + x^3 + x^3*y + y = 147^157 ∧
    x^3 + x^3*y + y^2 + y + z^9 = 157^147 := by
  sorry

end no_integer_solutions_l3627_362724


namespace singles_percentage_l3627_362791

def total_hits : ℕ := 40
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 6

def singles : ℕ := total_hits - (home_runs + triples + doubles)

def percentage_singles : ℚ := singles / total_hits * 100

theorem singles_percentage : percentage_singles = 72.5 := by
  sorry

end singles_percentage_l3627_362791


namespace units_digit_sum_of_powers_l3627_362718

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the exponentiation operation
def pow (base : ℕ) (exp : ℕ) : ℕ := base ^ exp

-- Theorem statement
theorem units_digit_sum_of_powers : 
  unitsDigit (pow 3 2014 + pow 4 2015 + pow 5 2016) = 8 := by
  sorry

end units_digit_sum_of_powers_l3627_362718


namespace equation_transformation_l3627_362745

theorem equation_transformation (x y : ℝ) (h : 2*x - 3*y + 6 = 0) : 
  6*x - 9*y + 6 = -12 := by
  sorry

end equation_transformation_l3627_362745


namespace san_francisco_super_bowl_probability_l3627_362795

theorem san_francisco_super_bowl_probability 
  (p_play : ℝ) 
  (p_not_play : ℝ) 
  (h1 : p_play = 9 * p_not_play) 
  (h2 : p_play + p_not_play = 1) : 
  p_play = 0.9 := by
sorry

end san_francisco_super_bowl_probability_l3627_362795


namespace additional_houses_built_l3627_362723

/-- Proves the number of additional houses built between the first half of the year and October -/
theorem additional_houses_built
  (total_houses : ℕ)
  (first_half_fraction : ℚ)
  (remaining_houses : ℕ)
  (h1 : total_houses = 2000)
  (h2 : first_half_fraction = 3/5)
  (h3 : remaining_houses = 500) :
  (total_houses - remaining_houses) - (first_half_fraction * total_houses) = 300 := by
  sorry

end additional_houses_built_l3627_362723


namespace quadratic_one_root_l3627_362722

theorem quadratic_one_root (m : ℝ) (h : m > 0) :
  (∃! x : ℝ, x^2 + 4*m*x + m = 0) ↔ m = 1/4 := by
sorry

end quadratic_one_root_l3627_362722


namespace saturn_diameter_times_ten_l3627_362713

/-- The diameter of Saturn in kilometers -/
def saturn_diameter : ℝ := 1.2 * 10^5

/-- Theorem stating the correct multiplication of Saturn's diameter by 10 -/
theorem saturn_diameter_times_ten :
  saturn_diameter * 10 = 1.2 * 10^6 := by
  sorry

end saturn_diameter_times_ten_l3627_362713


namespace miran_has_fewest_paper_l3627_362700

def miran_paper : ℕ := 6
def junga_paper : ℕ := 13
def minsu_paper : ℕ := 10

theorem miran_has_fewest_paper :
  miran_paper ≤ junga_paper ∧ miran_paper ≤ minsu_paper :=
sorry

end miran_has_fewest_paper_l3627_362700


namespace expression_factorization_l3627_362714

theorem expression_factorization (x : ℝ) : 
  (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10) = 5 * x * (5 * x^2 + 19) := by
sorry

end expression_factorization_l3627_362714


namespace quadratic_inequality_solution_set_l3627_362779

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | a * x^2 + 5 * x + b > 0}) : 
  {x : ℝ | b * x^2 - 5 * x + a > 0} = Set.Ioo (-1/2) (-1/3) := by
  sorry

end quadratic_inequality_solution_set_l3627_362779


namespace geometric_sequence_302nd_term_l3627_362725

/-- Given a geometric sequence with first term 8 and second term -16, 
    the 302nd term is -2^304 -/
theorem geometric_sequence_302nd_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 2) = a (n + 1) * (a (n + 1) / a n)) →  -- geometric sequence condition
    a 1 = 8 →                                           -- first term
    a 2 = -16 →                                         -- second term
    a 302 = -2^304 := by
  sorry


end geometric_sequence_302nd_term_l3627_362725


namespace max_cubes_in_box_l3627_362738

theorem max_cubes_in_box (box_length box_width box_height cube_volume : ℕ) 
  (h1 : box_length = 8)
  (h2 : box_width = 9)
  (h3 : box_height = 12)
  (h4 : cube_volume = 27) :
  (box_length * box_width * box_height) / cube_volume = 32 := by
  sorry

#check max_cubes_in_box

end max_cubes_in_box_l3627_362738


namespace product_of_complex_in_polar_form_specific_complex_product_l3627_362729

/-- 
Given two complex numbers in polar form, prove that their product 
is equal to the product of their magnitudes and the sum of their angles.
-/
theorem product_of_complex_in_polar_form 
  (z₁ : ℂ) (z₂ : ℂ) (r₁ r₂ θ₁ θ₂ : ℝ) :
  z₁ = r₁ * Complex.exp (θ₁ * Complex.I) →
  z₂ = r₂ * Complex.exp (θ₂ * Complex.I) →
  r₁ > 0 →
  r₂ > 0 →
  z₁ * z₂ = (r₁ * r₂) * Complex.exp ((θ₁ + θ₂) * Complex.I) :=
by sorry

/-- 
Prove that the product of 5cis(25°) and 4cis(48°) is equal to 20cis(73°).
-/
theorem specific_complex_product :
  let z₁ : ℂ := 5 * Complex.exp (25 * π / 180 * Complex.I)
  let z₂ : ℂ := 4 * Complex.exp (48 * π / 180 * Complex.I)
  z₁ * z₂ = 20 * Complex.exp (73 * π / 180 * Complex.I) :=
by sorry

end product_of_complex_in_polar_form_specific_complex_product_l3627_362729


namespace tim_travel_distance_l3627_362777

/-- Represents the distance traveled by Tim and Élan -/
structure TravelDistance where
  tim : ℝ
  elan : ℝ

/-- Calculates the distance traveled in one hour given initial speeds -/
def distanceInHour (timSpeed : ℝ) (elanSpeed : ℝ) : TravelDistance :=
  { tim := timSpeed, elan := elanSpeed }

/-- Theorem: Tim travels 60 miles before meeting Élan -/
theorem tim_travel_distance (initialDistance : ℝ) (timInitialSpeed : ℝ) (elanInitialSpeed : ℝ) :
  initialDistance = 90 ∧ timInitialSpeed = 10 ∧ elanInitialSpeed = 5 →
  (let d1 := distanceInHour timInitialSpeed elanInitialSpeed
   let d2 := distanceInHour (2 * timInitialSpeed) (2 * elanInitialSpeed)
   let d3 := distanceInHour (4 * timInitialSpeed) (4 * elanInitialSpeed)
   d1.tim + d2.tim + (initialDistance - d1.tim - d1.elan - d2.tim - d2.elan) * (4 * timInitialSpeed) / (4 * timInitialSpeed + 4 * elanInitialSpeed) = 60) :=
by
  sorry


end tim_travel_distance_l3627_362777


namespace remainder_2457633_div_25_l3627_362731

theorem remainder_2457633_div_25 : 2457633 % 25 = 8 := by
  sorry

end remainder_2457633_div_25_l3627_362731


namespace panel_discussion_selection_l3627_362755

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem panel_discussion_selection (num_boys num_girls : ℕ) 
  (hb : num_boys = 5) (hg : num_girls = 4) : 
  -- I. Number of ways to select 2 boys and 2 girls
  (choose num_boys 2) * (choose num_girls 2) = 60 ∧ 
  -- II. Number of ways to select 4 people including at least one of boy A or girl B
  (choose (num_boys + num_girls) 4) - (choose (num_boys + num_girls - 2) 4) = 91 ∧
  -- III. Number of ways to select 4 people containing both boys and girls
  (choose (num_boys + num_girls) 4) - (choose num_boys 4) - (choose num_girls 4) = 120 :=
by sorry

end panel_discussion_selection_l3627_362755


namespace complex_number_modulus_l3627_362757

theorem complex_number_modulus : Complex.abs ((1 - Complex.I) / Complex.I) = Real.sqrt 2 := by
  sorry

end complex_number_modulus_l3627_362757


namespace books_per_shelf_l3627_362708

theorem books_per_shelf 
  (mystery_shelves : ℕ) 
  (picture_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : mystery_shelves = 6) 
  (h2 : picture_shelves = 2) 
  (h3 : total_books = 72) : 
  total_books / (mystery_shelves + picture_shelves) = 9 := by
  sorry

end books_per_shelf_l3627_362708


namespace valid_numbers_l3627_362740

def is_valid_number (n : ℕ) : Prop :=
  100000 > n ∧ n ≥ 10000 ∧  -- five-digit number
  n % 72 = 0 ∧  -- divisible by 72
  (n.digits 10).count 1 = 3  -- exactly three digits are 1

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {41112, 14112, 11016, 11160} := by
  sorry

end valid_numbers_l3627_362740


namespace johns_money_in_euros_johns_money_in_euros_proof_l3627_362786

/-- Proves that John's money in Euros is 612 given the conditions of the problem -/
theorem johns_money_in_euros : ℝ → Prop :=
  fun conversion_rate =>
    ∀ (darwin mia laura john : ℝ),
      darwin = 45 →
      mia = 2 * darwin + 20 →
      laura = 3 * (mia + darwin) - 30 →
      john = 1.5 * (laura + darwin) →
      conversion_rate = 0.85 →
      john * conversion_rate = 612

/-- Proof of the theorem -/
theorem johns_money_in_euros_proof : johns_money_in_euros 0.85 := by
  sorry

end johns_money_in_euros_johns_money_in_euros_proof_l3627_362786


namespace quadratic_one_solution_l3627_362734

theorem quadratic_one_solution (q : ℝ) (hq : q ≠ 0) :
  (∃! x, q * x^2 - 18 * x + 8 = 0) ↔ q = 81/8 := by
sorry

end quadratic_one_solution_l3627_362734


namespace normal_distribution_probability_l3627_362748

/-- A random variable following a normal distribution with mean μ and standard deviation σ. -/
structure NormalRV (μ σ : ℝ) where
  (σ_pos : σ > 0)

/-- The probability that a normal random variable falls within a given interval. -/
def prob_interval (X : NormalRV μ σ) (a b : ℝ) : ℝ := sorry

/-- Theorem: For a normal distribution N(4, 1²), given specific probabilities for certain intervals,
    the probability P(5 < X < 6) is equal to 0.1359. -/
theorem normal_distribution_probability (X : NormalRV 4 1) :
  prob_interval X 2 6 = 0.9544 →
  prob_interval X 3 5 = 0.6826 →
  prob_interval X 5 6 = 0.1359 := by sorry

end normal_distribution_probability_l3627_362748


namespace purely_imaginary_condition_l3627_362727

theorem purely_imaginary_condition (a : ℝ) :
  a = -1 ↔ (∃ b : ℝ, Complex.mk (a^2 - 1) (a - 1) = Complex.I * b) := by
  sorry

end purely_imaginary_condition_l3627_362727


namespace geometric_sequence_min_value_l3627_362771

/-- A geometric sequence with positive terms where the 7th term is √2/2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ 
  (∀ n : ℕ, a n > 0) ∧
  (a 7 = Real.sqrt 2 / 2)

/-- The minimum value of 1/a_3 + 2/a_11 for the given geometric sequence is 4 -/
theorem geometric_sequence_min_value (a : ℕ → ℝ) (h : GeometricSequence a) :
  (1 / a 3 + 2 / a 11) ≥ 4 ∧ ∃ b : ℕ → ℝ, GeometricSequence b ∧ 1 / b 3 + 2 / b 11 = 4 :=
sorry

end geometric_sequence_min_value_l3627_362771


namespace red_item_count_l3627_362798

/-- Represents the number of items of a specific color in the box -/
structure ColorCount where
  hats : ℕ
  gloves : ℕ

/-- Represents the contents of the box -/
structure Box where
  red : ColorCount
  green : ColorCount
  orange : ColorCount

/-- The maximum number of draws needed to guarantee a pair of each color -/
def max_draws (b : Box) : ℕ :=
  max (b.red.hats + b.red.gloves) (max (b.green.hats + b.green.gloves) (b.orange.hats + b.orange.gloves)) + 2

/-- The theorem stating that if it takes 66 draws to guarantee a pair of each color,
    given 23 green items and 11 orange items, then there must be 30 red items -/
theorem red_item_count (b : Box) 
  (h_green : b.green.hats + b.green.gloves = 23)
  (h_orange : b.orange.hats + b.orange.gloves = 11)
  (h_draws : max_draws b = 66) :
  b.red.hats + b.red.gloves = 30 := by
  sorry

end red_item_count_l3627_362798


namespace total_spent_is_64_l3627_362756

/-- The total amount spent by Victor and his friend on trick decks -/
def total_spent (deck_price : ℕ) (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * (victor_decks + friend_decks)

/-- Proof that Victor and his friend spent $64 in total -/
theorem total_spent_is_64 :
  total_spent 8 6 2 = 64 := by
  sorry

end total_spent_is_64_l3627_362756


namespace valid_square_root_expression_l3627_362796

theorem valid_square_root_expression (a b : ℝ) : 
  (Real.sqrt (-a^2 * b^2) = -a * b) ↔ (a * b = 0) := by sorry

end valid_square_root_expression_l3627_362796


namespace sample_frequency_calculation_l3627_362783

theorem sample_frequency_calculation (total_volume : ℕ) (num_groups : ℕ) 
  (freq_3 freq_4 freq_5 freq_6 : ℕ) (ratio_group_1 : ℚ) :
  total_volume = 80 →
  num_groups = 6 →
  freq_3 = 10 →
  freq_4 = 12 →
  freq_5 = 14 →
  freq_6 = 20 →
  ratio_group_1 = 1/5 →
  ∃ (freq_1 freq_2 : ℕ),
    freq_1 = 16 ∧
    freq_2 = 8 ∧
    freq_1 + freq_2 + freq_3 + freq_4 + freq_5 + freq_6 = total_volume ∧
    freq_1 = (ratio_group_1 * total_volume).num := by
  sorry

end sample_frequency_calculation_l3627_362783


namespace triangle_inequality_l3627_362739

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  can_form_triangle 2 6 6 ∧
  ¬can_form_triangle 2 6 2 ∧
  ¬can_form_triangle 2 6 4 ∧
  ¬can_form_triangle 2 6 9 :=
by sorry

end triangle_inequality_l3627_362739


namespace student_survey_l3627_362706

theorem student_survey (french_and_english : ℕ) (french_not_english : ℕ) 
  (h1 : french_and_english = 20)
  (h2 : french_not_english = 60)
  (h3 : french_and_english + french_not_english = (2 : ℝ) / 5 * total_students) :
  total_students = 200 :=
by
  sorry

end student_survey_l3627_362706


namespace books_difference_l3627_362720

def summer_reading (june july august : ℕ) : Prop :=
  june = 8 ∧ july = 2 * june ∧ june + july + august = 37

theorem books_difference (june july august : ℕ) 
  (h : summer_reading june july august) : july - august = 3 := by
  sorry

end books_difference_l3627_362720


namespace joes_steakhouse_wages_l3627_362785

/-- Proves that the hourly wage of a manager is $8.5 given the conditions from Joe's Steakhouse --/
theorem joes_steakhouse_wages (manager_wage dishwasher_wage chef_wage : ℝ) : 
  chef_wage = dishwasher_wage * 1.22 →
  dishwasher_wage = manager_wage / 2 →
  chef_wage = manager_wage - 3.315 →
  manager_wage = 8.5 := by
sorry

end joes_steakhouse_wages_l3627_362785


namespace systematic_sampling_interval_l3627_362746

/-- The interval of segmentation for systematic sampling -/
def interval_of_segmentation (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The interval of segmentation for a population of 1200 and sample size of 40 is 30 -/
theorem systematic_sampling_interval :
  interval_of_segmentation 1200 40 = 30 := by
  sorry

end systematic_sampling_interval_l3627_362746


namespace min_value_squared_sum_l3627_362772

theorem min_value_squared_sum (a b c d : ℝ) (h1 : a * b = 2) (h2 : c * d = 18) :
  (a * c)^2 + (b * d)^2 ≥ 12 ∧ ∃ (a' b' c' d' : ℝ), a' * b' = 2 ∧ c' * d' = 18 ∧ (a' * c')^2 + (b' * d')^2 = 12 := by
  sorry

end min_value_squared_sum_l3627_362772


namespace factor_polynomial_l3627_362753

theorem factor_polynomial (x : ℝ) : 60 * x^5 - 135 * x^9 = 15 * x^5 * (4 - 9 * x^4) := by
  sorry

end factor_polynomial_l3627_362753


namespace cycle_price_calculation_l3627_362747

theorem cycle_price_calculation (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 1125)
  (h2 : gain_percentage = 25) : 
  ∃ original_price : ℝ, 
    original_price * (1 + gain_percentage / 100) = selling_price ∧ 
    original_price = 900 := by
sorry

end cycle_price_calculation_l3627_362747


namespace sufficient_but_not_necessary_condition_for_abs_x_leq_one_l3627_362710

theorem sufficient_but_not_necessary_condition_for_abs_x_leq_one :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x| ≤ 1) ∧
  ¬(∀ x : ℝ, |x| ≤ 1 → 0 ≤ x ∧ x ≤ 1) :=
by sorry

end sufficient_but_not_necessary_condition_for_abs_x_leq_one_l3627_362710


namespace exp_gt_one_plus_x_when_not_zero_l3627_362793

theorem exp_gt_one_plus_x_when_not_zero (x : ℝ) (h : x ≠ 0) : Real.exp x > 1 + x := by
  sorry

end exp_gt_one_plus_x_when_not_zero_l3627_362793


namespace average_difference_l3627_362774

theorem average_difference : 
  let set1 : List ℝ := [10, 20, 60]
  let set2 : List ℝ := [10, 40, 25]
  (set1.sum / set1.length) - (set2.sum / set2.length) = 5 := by
sorry

end average_difference_l3627_362774


namespace cubic_function_minimum_l3627_362797

theorem cubic_function_minimum (a b c : ℝ) : 
  let f := fun x => a * x^3 + b * x^2 + c * x - 34
  let f' := fun x => 3 * a * x^2 + 2 * b * x + c
  (∀ x, f' x ≤ 0 ↔ -2 ≤ x ∧ x ≤ 3) →
  (∃ x₀, ∀ x, f x ≥ f x₀) →
  f 3 = -115 →
  a = 2 := by
sorry

end cubic_function_minimum_l3627_362797


namespace table_tennis_expected_scores_l3627_362765

/-- Win probability for a match-up -/
structure MatchProbability where
  team_a_win : ℚ
  team_b_win : ℚ
  sum_to_one : team_a_win + team_b_win = 1

/-- Team scores -/
structure TeamScores where
  team_a : ℕ
  team_b : ℕ
  sum_to_three : team_a + team_b = 3

/-- Expected value of a discrete random variable -/
def expectedValue (probs : List ℚ) (values : List ℚ) : ℚ :=
  (probs.zip values).map (fun (p, v) => p * v) |>.sum

/-- Main theorem -/
theorem table_tennis_expected_scores 
  (match1 : MatchProbability) 
  (match2 : MatchProbability) 
  (match3 : MatchProbability) 
  (h1 : match1.team_a_win = 2/3)
  (h2 : match2.team_a_win = 2/5)
  (h3 : match3.team_a_win = 2/5) :
  let scores := TeamScores
  let ξ_probs := [8/75, 28/75, 2/5, 3/25]
  let ξ_values := [3, 2, 1, 0]
  let η_probs := [3/25, 2/5, 28/75, 8/75]
  let η_values := [3, 2, 1, 0]
  expectedValue ξ_probs ξ_values = 22/15 ∧ 
  expectedValue η_probs η_values = 23/15 := by
sorry


end table_tennis_expected_scores_l3627_362765


namespace function_form_proof_l3627_362781

theorem function_form_proof (f : ℝ → ℝ) (k : ℝ) 
  (h_continuous : Continuous f)
  (h_zero : f 0 = 0)
  (h_inequality : ∀ x y, f (x + y) ≥ f x + f y + k * x * y) :
  ∃ b : ℝ, ∀ x, f x = k / 2 * x^2 + b * x :=
sorry

end function_form_proof_l3627_362781


namespace triangle_inequality_squares_l3627_362703

theorem triangle_inequality_squares (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + a * c) := by
  sorry

end triangle_inequality_squares_l3627_362703


namespace cloth_sale_meters_l3627_362780

/-- Proves that the number of meters of cloth sold is 85, given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  total_selling_price = 8500 →
  profit_per_meter = 15 →
  cost_price_per_meter = 85 →
  (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
sorry

end cloth_sale_meters_l3627_362780


namespace lake_pleasant_activities_l3627_362741

theorem lake_pleasant_activities (total_kids : ℕ) (tubing_fraction : ℚ) (rafting_fraction : ℚ) (kayaking_fraction : ℚ)
  (h_total : total_kids = 40)
  (h_tubing : tubing_fraction = 1/4)
  (h_rafting : rafting_fraction = 1/2)
  (h_kayaking : kayaking_fraction = 1/3) :
  ⌊(total_kids : ℚ) * tubing_fraction * rafting_fraction * kayaking_fraction⌋ = 1 := by
sorry

end lake_pleasant_activities_l3627_362741


namespace population_exceeds_target_in_2075_l3627_362744

/-- The initial population of Nisos in the year 2000 -/
def initial_population : ℕ := 500

/-- The year when the population count starts -/
def start_year : ℕ := 2000

/-- The number of years it takes for the population to triple -/
def tripling_period : ℕ := 25

/-- The target population we want to exceed -/
def target_population : ℕ := 9000

/-- Calculate the population after a given number of tripling periods -/
def population_after (periods : ℕ) : ℕ :=
  initial_population * (3 ^ periods)

/-- Calculate the year after a given number of tripling periods -/
def year_after (periods : ℕ) : ℕ :=
  start_year + tripling_period * periods

/-- The theorem to be proved -/
theorem population_exceeds_target_in_2075 :
  ∃ n : ℕ, year_after n = 2075 ∧ 
    population_after n > target_population ∧
    population_after (n - 1) ≤ target_population :=
by
  sorry


end population_exceeds_target_in_2075_l3627_362744
