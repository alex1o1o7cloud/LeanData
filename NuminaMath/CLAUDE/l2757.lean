import Mathlib

namespace existence_of_unique_distance_point_l2757_275781

-- Define a lattice point as a pair of integers
def LatticePoint := ℤ × ℤ

-- Define a function to calculate the squared distance between two points
def squaredDistance (x y : ℝ × ℝ) : ℝ :=
  (x.1 - y.1)^2 + (x.2 - y.2)^2

theorem existence_of_unique_distance_point :
  ∃ (P : ℝ × ℝ), 
    (∃ (a b : ℝ), P = (a, b) ∧ Irrational a ∧ Irrational b) ∧
    (∀ (L₁ L₂ : LatticePoint), 
      L₁ ≠ L₂ → squaredDistance (P.1, P.2) (↑L₁.1, ↑L₁.2) ≠ 
                 squaredDistance (P.1, P.2) (↑L₂.1, ↑L₂.2)) :=
by sorry

end existence_of_unique_distance_point_l2757_275781


namespace quadratic_roots_nature_l2757_275744

/-- Represents a quadratic equation of the form ax^2 - 3x√3 + b = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ

/-- The discriminant of the quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ := 27 - 4 * eq.a * eq.b

/-- Predicate for real and distinct roots -/
def has_real_distinct_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq ≠ 0 ∧ discriminant eq > 0

theorem quadratic_roots_nature (eq : QuadraticEquation) 
  (h : discriminant eq ≠ 0) : 
  has_real_distinct_roots eq :=
sorry

end quadratic_roots_nature_l2757_275744


namespace unique_score_is_correct_l2757_275789

/-- Represents the score on the Mini-AHSME exam -/
structure MiniAHSMEScore where
  total : ℕ
  correct : ℕ
  wrong : ℕ
  h_total : total = 20 + 3 * correct - wrong
  h_questions : correct + wrong ≤ 20

/-- The unique score that satisfies all conditions of the problem -/
def unique_score : MiniAHSMEScore := ⟨53, 11, 0, by simp, by simp⟩

theorem unique_score_is_correct :
  ∀ s : MiniAHSMEScore,
    s.total > 50 →
    (∀ t : MiniAHSMEScore, t.total > 50 ∧ t.total < s.total → 
      ∃ u : MiniAHSMEScore, u.total = t.total ∧ u.correct ≠ t.correct) →
    s = unique_score := by sorry

end unique_score_is_correct_l2757_275789


namespace min_a_correct_l2757_275735

/-- The number of cards in the deck -/
def n : ℕ := 51

/-- The probability that Alex and Dylan are on the same team given that Alex picks one of the cards a and a+7, and Dylan picks the other -/
def p (a : ℕ) : ℚ :=
  (Nat.choose (42 - a) 2 + Nat.choose (a - 1) 2) / Nat.choose (n - 2) 2

/-- The minimum value of a for which p(a) ≥ 1/2 -/
def min_a : ℕ := 22

theorem min_a_correct :
  (∀ a : ℕ, 1 ≤ a ∧ a + 7 ≤ n → p a ≥ 1/2 → a ≥ min_a) ∧
  p min_a ≥ 1/2 :=
sorry

end min_a_correct_l2757_275735


namespace average_difference_is_negative_13_point_5_l2757_275720

/-- Represents a school with students and teachers -/
structure School where
  num_students : ℕ
  num_teachers : ℕ
  class_sizes : List ℕ

/-- Calculates the average number of students per teacher -/
def average_students_per_teacher (school : School) : ℚ :=
  (school.class_sizes.sum : ℚ) / school.num_teachers

/-- Calculates the average number of students per student -/
def average_students_per_student (school : School) : ℚ :=
  (school.class_sizes.map (λ size => size * size)).sum / school.num_students

/-- The main theorem to be proved -/
theorem average_difference_is_negative_13_point_5 (school : School) 
  (h1 : school.num_students = 100)
  (h2 : school.num_teachers = 5)
  (h3 : school.class_sizes = [50, 20, 20, 5, 5]) :
  average_students_per_teacher school - average_students_per_student school = -13.5 := by
  sorry

end average_difference_is_negative_13_point_5_l2757_275720


namespace pqr_value_l2757_275799

theorem pqr_value (a b c p q r : ℂ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h_a : a = (b + c) / (p - 3))
  (h_b : b = (a + c) / (q - 3))
  (h_c : c = (a + b) / (r - 3))
  (h_sum_prod : p * q + p * r + q * r = 10)
  (h_sum : p + q + r = 6) :
  p * q * r = 14 := by
sorry

end pqr_value_l2757_275799


namespace complex_number_quadrant_l2757_275706

theorem complex_number_quadrant (z : ℂ) : z * Complex.I = 2 - Complex.I → z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_number_quadrant_l2757_275706


namespace inequality_equivalence_l2757_275758

theorem inequality_equivalence (x : ℝ) : (x + 1) * (2 - x) > 0 ↔ x ∈ Set.Ioo (-1) 2 := by
  sorry

end inequality_equivalence_l2757_275758


namespace binary_to_quaternary_conversion_l2757_275778

/-- Convert a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Convert a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : List (Fin 4) := sorry

/-- The binary representation of 110111001₂ -/
def binary_number : List Bool := [true, true, false, true, true, true, false, false, true]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_number) = [1, 3, 2, 2, 1] := by sorry

end binary_to_quaternary_conversion_l2757_275778


namespace points_are_concyclic_l2757_275759

-- Define the points
variable (A B C D E F G H : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- State the given conditions
axiom condition1 : angle A E H = angle F E B
axiom condition2 : angle E F B = angle C F G
axiom condition3 : angle C G F = angle D G H
axiom condition4 : angle D H G = angle A H E

-- Define concyclicity
def concyclic (A B C D : Point) : Prop := sorry

-- State the theorem
theorem points_are_concyclic : concyclic A B C D := sorry

end points_are_concyclic_l2757_275759


namespace cricket_game_overs_l2757_275705

/-- The number of initial overs in a cricket game -/
def initial_overs : ℕ := 20

/-- The initial run rate in runs per over -/
def initial_run_rate : ℚ := 46/10

/-- The target score in runs -/
def target_score : ℕ := 396

/-- The number of remaining overs -/
def remaining_overs : ℕ := 30

/-- The required run rate for the remaining overs -/
def required_run_rate : ℚ := 10133333333333333/1000000000000000

theorem cricket_game_overs :
  initial_overs * initial_run_rate + 
  remaining_overs * required_run_rate = target_score :=
sorry

end cricket_game_overs_l2757_275705


namespace ceiling_floor_product_l2757_275792

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by sorry

end ceiling_floor_product_l2757_275792


namespace good_pair_exists_l2757_275750

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ 
  ∃ a b : ℕ, m * n = a ^ 2 ∧ (m + 1) * (n + 1) = b ^ 2 := by
  let n := m * (4 * m + 3) ^ 2
  have h1 : n > m := sorry
  have h2 : ∃ a : ℕ, m * n = a ^ 2 := sorry
  have h3 : ∃ b : ℕ, (m + 1) * (n + 1) = b ^ 2 := sorry
  exact ⟨n, h1, h2.choose, h3.choose, h2.choose_spec, h3.choose_spec⟩

end good_pair_exists_l2757_275750


namespace work_completion_time_l2757_275733

/-- 
Given that:
- A does 20% less work than B per unit time
- A completes the work in 7.5 hours
Prove that B completes the same work in 6 hours
-/
theorem work_completion_time (work_rate_A work_rate_B : ℝ) 
  (h1 : work_rate_A = 0.8 * work_rate_B) 
  (h2 : work_rate_A * 7.5 = work_rate_B * 6) : 
  work_rate_B * 6 = work_rate_A * 7.5 := by
  sorry

#check work_completion_time

end work_completion_time_l2757_275733


namespace shopping_trip_theorem_l2757_275787

/-- Shopping Trip Theorem -/
theorem shopping_trip_theorem (initial_amount : ℕ) (shoe_cost : ℕ) :
  initial_amount = 158 →
  shoe_cost = 45 →
  let bag_cost := shoe_cost - 17
  let lunch_cost := bag_cost / 4
  let total_spent := shoe_cost + bag_cost + lunch_cost
  initial_amount - total_spent = 78 := by
sorry

end shopping_trip_theorem_l2757_275787


namespace necessary_condition_for_greater_than_not_sufficient_condition_for_greater_than_l2757_275761

theorem necessary_condition_for_greater_than (a b : ℝ) :
  (a > b) → (a + 1 > b) :=
sorry

theorem not_sufficient_condition_for_greater_than :
  ∃ (a b : ℝ), (a + 1 > b) ∧ ¬(a > b) :=
sorry

end necessary_condition_for_greater_than_not_sufficient_condition_for_greater_than_l2757_275761


namespace power_five_mod_seven_l2757_275716

theorem power_five_mod_seven : 5^2010 % 7 = 1 := by
  sorry

end power_five_mod_seven_l2757_275716


namespace binomial_expansion_coefficient_l2757_275732

theorem binomial_expansion_coefficient (x : ℝ) :
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
    (2*x - 3)^6 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 ∧
    a₄ = 240 := by
  sorry

end binomial_expansion_coefficient_l2757_275732


namespace points_collinear_and_m_values_l2757_275767

noncomputable section

-- Define the points and vectors
def O : ℝ × ℝ := (0, 0)
def A (x : ℝ) : ℝ × ℝ := (1, Real.cos x)
def B (x : ℝ) : ℝ × ℝ := (1 + Real.sin x, Real.cos x)
def OA (x : ℝ) : ℝ × ℝ := A x
def OB (x : ℝ) : ℝ × ℝ := B x
def OC (x : ℝ) : ℝ × ℝ := (1/3 : ℝ) • (OA x) + (2/3 : ℝ) • (OB x)

-- Define the function f
def f (x m : ℝ) : ℝ :=
  (OA x).1 * (OC x).1 + (OA x).2 * (OC x).2 +
  (2*m + 1/3) * Real.sqrt ((B x).1 - (A x).1)^2 + ((B x).2 - (A x).2)^2 +
  m^2

-- Theorem statement
theorem points_collinear_and_m_values (x : ℝ) (h : x ∈ Set.Icc 0 (Real.pi / 2)) :
  (∃ t : ℝ, OC x = t • OA x + (1 - t) • OB x) ∧
  (∃ m : ℝ, (∀ y ∈ Set.Icc 0 (Real.pi / 2), f y m ≥ 5) ∧ f x m = 5 ∧ (m = -3 ∨ m = Real.sqrt 3)) :=
sorry

end points_collinear_and_m_values_l2757_275767


namespace minimum_laptops_l2757_275725

theorem minimum_laptops (n p : ℕ) (h1 : n > 3) (h2 : p > 0) : 
  (p / n + (n - 3) * (p / n + 15) - p = 105) → n ≥ 10 :=
by
  sorry

#check minimum_laptops

end minimum_laptops_l2757_275725


namespace cubic_root_sum_l2757_275717

theorem cubic_root_sum (p q r : ℝ) : 
  p + q + r = 4 →
  p * q + p * r + q * r = 1 →
  p * q * r = -6 →
  p / (q * r + 1) + q / (p * r + 1) + r / (p * q + 1) = 22 - 213 / 7 := by
  sorry

end cubic_root_sum_l2757_275717


namespace correct_calculation_l2757_275703

theorem correct_calculation (x : ℝ) (h : 2 * x = 22) : 20 * x + 3 = 223 := by
  sorry

end correct_calculation_l2757_275703


namespace min_c_plus_d_is_15_l2757_275788

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

theorem min_c_plus_d_is_15 :
  ∀ (A B C D : Digit),
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A.val + B.val : ℕ) ≠ 0 →
    (C.val + D.val : ℕ) ≠ 0 →
    (A.val + B.val : ℕ) < (C.val + D.val) →
    (C.val + D.val) % (A.val + B.val) = 0 →
    ∀ (E F G H : Digit),
      E ≠ F → E ≠ G → E ≠ H → F ≠ G → F ≠ H → G ≠ H →
      (E.val + F.val : ℕ) ≠ 0 →
      (G.val + H.val : ℕ) ≠ 0 →
      (E.val + F.val : ℕ) < (G.val + H.val) →
      (G.val + H.val) % (E.val + F.val) = 0 →
      (A.val + B.val : ℕ) / (C.val + D.val : ℕ) ≤ (E.val + F.val : ℕ) / (G.val + H.val : ℕ) →
      (C.val + D.val : ℕ) ≤ 15 :=
by sorry

#check min_c_plus_d_is_15

end min_c_plus_d_is_15_l2757_275788


namespace theater_performance_duration_l2757_275754

/-- The duration of a theater performance in hours -/
def performance_duration : ℝ := 3

/-- The number of weeks Mark visits the theater -/
def weeks : ℕ := 6

/-- The price per hour for a theater ticket in dollars -/
def price_per_hour : ℝ := 5

/-- The total amount spent on theater visits in dollars -/
def total_spent : ℝ := 90

theorem theater_performance_duration :
  performance_duration * price_per_hour * weeks = total_spent :=
by sorry

end theater_performance_duration_l2757_275754


namespace house_price_calculation_l2757_275712

theorem house_price_calculation (price_first : ℝ) (price_second : ℝ) : 
  price_second = 2 * price_first →
  price_first + price_second = 600000 →
  price_first = 200000 := by
sorry

end house_price_calculation_l2757_275712


namespace factorization_proof_l2757_275715

variable (x y b : ℝ)

theorem factorization_proof : 
  (-x^3 - 2*x^2 - x = -x*(x + 1)^2) ∧ 
  ((x - y) - 4*b^2*(x - y) = (x - y)*(1 + 2*b)*(1 - 2*b)) :=
by sorry

end factorization_proof_l2757_275715


namespace odd_even_sum_difference_problem_solution_l2757_275791

theorem odd_even_sum_difference : ℕ → Prop :=
  fun n =>
    let odd_sum := (n^2 + n) / 2
    let even_sum := n * (n + 1)
    odd_sum - even_sum = n + 1

theorem problem_solution :
  let n : ℕ := 1009
  let odd_sum := ((2*n + 1)^2 + (2*n + 1)) / 2
  let even_sum := n * (n + 1)
  odd_sum - even_sum = 1010 := by
  sorry

end odd_even_sum_difference_problem_solution_l2757_275791


namespace sequence_length_is_751_l2757_275760

/-- Given a sequence of real numbers satisfying certain conditions, prove that the length of the sequence is 751. -/
theorem sequence_length_is_751 (n : ℕ) (b : ℕ → ℝ) 
  (h_pos : n > 0)
  (h_b0 : b 0 = 40)
  (h_b1 : b 1 = 75)
  (h_bn : b n = 0)
  (h_rec : ∀ k, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 4 / b k) :
  n = 751 := by
  sorry

end sequence_length_is_751_l2757_275760


namespace total_net_amount_is_218_l2757_275734

/-- Represents a lottery ticket with its cost and number of winning numbers -/
structure LotteryTicket where
  cost : ℕ
  winningNumbers : ℕ

/-- Calculates the payout for a single ticket based on its winning numbers -/
def calculatePayout (ticket : LotteryTicket) : ℕ :=
  if ticket.winningNumbers ≤ 2 then
    ticket.winningNumbers * 15
  else
    30 + (ticket.winningNumbers - 2) * 20

/-- Calculates the net amount won for a single ticket -/
def calculateNetAmount (ticket : LotteryTicket) : ℤ :=
  (calculatePayout ticket : ℤ) - ticket.cost

/-- The set of lottery tickets Tony bought -/
def tonyTickets : List LotteryTicket := [
  ⟨5, 3⟩,
  ⟨7, 5⟩,
  ⟨4, 2⟩,
  ⟨6, 4⟩
]

/-- Theorem stating that the total net amount Tony won is $218 -/
theorem total_net_amount_is_218 :
  (tonyTickets.map calculateNetAmount).sum = 218 := by
  sorry

end total_net_amount_is_218_l2757_275734


namespace inequality_holds_iff_p_greater_than_two_point_five_l2757_275756

theorem inequality_holds_iff_p_greater_than_two_point_five (p q : ℝ) (hq : q > 0) :
  (5 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q ↔ p > 2.5 := by
  sorry

end inequality_holds_iff_p_greater_than_two_point_five_l2757_275756


namespace sqrt3_plus_minus_2_power_2023_l2757_275723

theorem sqrt3_plus_minus_2_power_2023 :
  (Real.sqrt 3 + 2) ^ 2023 * (Real.sqrt 3 - 2) ^ 2023 = -1 := by
  sorry

end sqrt3_plus_minus_2_power_2023_l2757_275723


namespace fruit_basket_problem_l2757_275730

theorem fruit_basket_problem (total_fruits : ℕ) 
  (basket_A basket_B : ℕ) 
  (apples_A pears_A apples_B pears_B : ℕ) :
  total_fruits = 82 →
  (basket_A + basket_B = total_fruits) →
  (basket_A ≥ basket_B → basket_A - basket_B < 10) →
  (basket_B > basket_A → basket_B - basket_A < 10) →
  (5 * apples_A = 2 * basket_A) →
  (7 * pears_B = 4 * basket_B) →
  (basket_A = apples_A + pears_A) →
  (basket_B = apples_B + pears_B) →
  (pears_A = 24 ∧ apples_B = 18) :=
by sorry

end fruit_basket_problem_l2757_275730


namespace base5_division_l2757_275782

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Theorem: The quotient of 2314₅ divided by 21₅ is equal to 110₅ -/
theorem base5_division :
  base10ToBase5 (base5ToBase10 [4, 1, 3, 2] / base5ToBase10 [1, 2]) = [0, 1, 1] :=
sorry

end base5_division_l2757_275782


namespace solution_value_l2757_275784

theorem solution_value (x m : ℤ) : 
  x = -2 → 3 * x + 5 = x - m → m = -1 := by
  sorry

end solution_value_l2757_275784


namespace weight_difference_l2757_275766

/-- The weight difference between two metal pieces -/
theorem weight_difference (iron_weight aluminum_weight : ℝ) 
  (h1 : iron_weight = 11.17)
  (h2 : aluminum_weight = 0.83) : 
  iron_weight - aluminum_weight = 10.34 := by
  sorry

end weight_difference_l2757_275766


namespace equation_sum_squares_l2757_275762

theorem equation_sum_squares (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := by
  sorry

end equation_sum_squares_l2757_275762


namespace exactly_one_greater_than_one_l2757_275701

theorem exactly_one_greater_than_one (a b c : ℝ) : 
  a * b * c = 1 → a + b + c > 1/a + 1/b + 1/c → 
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end exactly_one_greater_than_one_l2757_275701


namespace probability_in_B_l2757_275727

-- Define set A
def A : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 ≤ 2}

-- Define set B
def B : Set (ℝ × ℝ) := {p ∈ A | p.2 ≤ p.1^2}

-- Define the area function
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem probability_in_B : (area B) / (area A) = 17 / 24 := by sorry

end probability_in_B_l2757_275727


namespace complex_equation_sum_l2757_275752

theorem complex_equation_sum (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b - Complex.I →
  a + b = 1 := by sorry

end complex_equation_sum_l2757_275752


namespace subset_implies_m_eq_neg_two_l2757_275749

def set_A (m : ℝ) : Set ℝ := {3, 4, 4*m - 4}
def set_B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_eq_neg_two (m : ℝ) :
  set_B m ⊆ set_A m → m = -2 := by
  sorry

end subset_implies_m_eq_neg_two_l2757_275749


namespace infinitely_many_wrappers_l2757_275707

/-- A wrapper for a 1 × 1 painting is a rectangle with area 2 that can cover the painting on both sides. -/
def IsWrapper (width height : ℝ) : Prop :=
  width > 0 ∧ height > 0 ∧ width * height = 2 ∧ width ≥ 1 ∧ height ≥ 1

/-- There exist infinitely many wrappers for a 1 × 1 painting. -/
theorem infinitely_many_wrappers :
  ∃ f : ℕ → ℝ × ℝ, ∀ n : ℕ, IsWrapper (f n).1 (f n).2 ∧
    ∀ m : ℕ, m ≠ n → f m ≠ f n :=
sorry

end infinitely_many_wrappers_l2757_275707


namespace parabola_vertex_l2757_275729

/-- Given a quadratic function f(x) = -x^2 + cx + d where the solution to f(x) ≤ 0 
    is [-7,1] ∪ [3,∞), prove that the vertex of the parabola is (-3, 16) -/
theorem parabola_vertex (c d : ℝ) 
  (h : Set.Icc (-7 : ℝ) 1 ∪ Set.Ici 3 = {x : ℝ | -x^2 + c*x + d ≤ 0}) : 
  let f := fun (x : ℝ) ↦ -x^2 + c*x + d
  ∃ (v : ℝ × ℝ), v = (-3, 16) ∧ ∀ (x : ℝ), f x ≤ f v.1 :=
sorry

end parabola_vertex_l2757_275729


namespace rachel_apple_trees_l2757_275711

/-- The total number of apples remaining on Rachel's trees -/
def total_apples_remaining (X : ℕ) : ℕ :=
  let first_four_trees := 10 + 40 + 15 + 22
  let remaining_trees := 48 * X
  first_four_trees + remaining_trees

/-- Theorem stating the total number of apples remaining on Rachel's trees -/
theorem rachel_apple_trees (X : ℕ) :
  total_apples_remaining X = 87 + 48 * X := by
  sorry

end rachel_apple_trees_l2757_275711


namespace a_value_l2757_275747

def P (a : ℝ) : Set ℝ := {1, 2, a}
def Q : Set ℝ := {x | x^2 - 9 = 0}

theorem a_value (a : ℝ) : P a ∩ Q = {3} → a = 3 := by
  sorry

end a_value_l2757_275747


namespace cloth_loss_per_metre_l2757_275708

def cloth_problem (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) : Prop :=
  let total_cost_price := total_metres * cost_price_per_metre
  let total_loss := total_cost_price - total_selling_price
  let loss_per_metre := total_loss / total_metres
  total_metres = 300 ∧ 
  total_selling_price = 18000 ∧ 
  cost_price_per_metre = 65 ∧
  loss_per_metre = 5

theorem cloth_loss_per_metre :
  ∃ (total_metres total_selling_price cost_price_per_metre : ℕ),
    cloth_problem total_metres total_selling_price cost_price_per_metre :=
by
  sorry

end cloth_loss_per_metre_l2757_275708


namespace sum_of_min_max_x_l2757_275739

theorem sum_of_min_max_x (x y z : ℝ) (sum_eq : x + y + z = 4) (sum_sq_eq : x^2 + y^2 + z^2 = 6) :
  ∃ (m M : ℝ), (∀ x', (∃ y' z', x' + y' + z' = 4 ∧ x'^2 + y'^2 + z'^2 = 6) → m ≤ x' ∧ x' ≤ M) ∧
  m + M = 8/3 :=
sorry

end sum_of_min_max_x_l2757_275739


namespace runner_speed_ratio_l2757_275763

/-- The runner's problem -/
theorem runner_speed_ratio (total_distance v₁ v₂ : ℝ) : 
  total_distance > 0 ∧
  v₁ > 0 ∧
  v₂ > 0 ∧
  total_distance / 2 / v₁ + 11 = total_distance / 2 / v₂ ∧
  total_distance / 2 / v₂ = 22 →
  v₁ / v₂ = 2 := by
  sorry

#check runner_speed_ratio

end runner_speed_ratio_l2757_275763


namespace chocolate_theorem_l2757_275710

def chocolate_problem (total : ℕ) (typeA typeB typeC : ℕ) : Prop :=
  let typeD := 2 * typeA
  let typeE := 2 * typeB
  let typeF := typeA + 6
  let typeG := typeB + 6
  let typeH := typeC + 6
  let non_peanut := typeA + typeB + typeC + typeD + typeE + typeF + typeG + typeH
  let peanut := total - non_peanut
  (peanut : ℚ) / total = 3 / 10

theorem chocolate_theorem :
  chocolate_problem 100 5 6 4 := by
  sorry

end chocolate_theorem_l2757_275710


namespace number_division_proof_l2757_275770

theorem number_division_proof (x : ℝ) : 4 * x = 166.08 → x / 4 = 10.38 := by
  sorry

end number_division_proof_l2757_275770


namespace g_zero_l2757_275726

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(7/5) = 0 -/
theorem g_zero : g (7/5) = 0 := by
  sorry

end g_zero_l2757_275726


namespace derivative_f_at_neg_two_l2757_275769

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem derivative_f_at_neg_two :
  (deriv f) (-2) = 0 := by sorry

end derivative_f_at_neg_two_l2757_275769


namespace polygon_diagonals_sides_l2757_275771

theorem polygon_diagonals_sides (n : ℕ) : n > 2 →
  (n * (n - 3) / 2 = 2 * n) → n = 7 := by
  sorry

end polygon_diagonals_sides_l2757_275771


namespace sum_of_squares_l2757_275779

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 28) : x^2 + y^2 = 8 := by
  sorry

end sum_of_squares_l2757_275779


namespace henry_games_count_henry_games_count_proof_l2757_275776

theorem henry_games_count : ℕ → ℕ → ℕ → Prop :=
  fun initial_neil initial_henry games_given =>
    -- Neil's initial games count
    let initial_neil_games := 7

    -- Henry initially had 3 times more games than Neil
    (initial_henry = 3 * initial_neil_games + initial_neil_games) →
    
    -- After giving Neil 6 games, Henry has 4 times more games than Neil
    (initial_henry - games_given = 4 * (initial_neil_games + games_given)) →
    
    -- The number of games given to Neil
    (games_given = 6) →
    
    -- Conclusion: Henry's initial game count
    initial_henry = 58

-- The proof of the theorem
theorem henry_games_count_proof : henry_games_count 7 58 6 := by
  sorry

end henry_games_count_henry_games_count_proof_l2757_275776


namespace arithmetic_sequence_problem_l2757_275719

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : S seq 8 = 4 * seq.a 3)
  (h2 : seq.a 7 = -2) :
  seq.a 9 = -6 := by
  sorry

end arithmetic_sequence_problem_l2757_275719


namespace air_quality_probabilities_l2757_275794

def prob_grade_A : ℝ := 0.8
def prob_grade_B : ℝ := 0.1
def prob_grade_C : ℝ := 0.1

def prob_satisfactory (p_A p_B p_C : ℝ) : ℝ :=
  p_A * p_A + 2 * p_A * (1 - p_A)

def prob_two_out_of_three (p : ℝ) : ℝ :=
  3 * p * p * (1 - p)

theorem air_quality_probabilities :
  prob_satisfactory prob_grade_A prob_grade_B prob_grade_C = 0.96 ∧
  prob_two_out_of_three (prob_satisfactory prob_grade_A prob_grade_B prob_grade_C) = 0.110592 := by
  sorry

end air_quality_probabilities_l2757_275794


namespace casper_initial_candies_l2757_275737

def candy_problem (initial : ℕ) : Prop :=
  let day1_after_eating : ℚ := (3/4) * initial
  let day1_remaining : ℚ := day1_after_eating - 3
  let day2_after_eating : ℚ := (1/2) * day1_remaining
  let day2_remaining : ℚ := day2_after_eating - 2
  day2_remaining = 10

theorem casper_initial_candies :
  candy_problem 36 := by sorry

end casper_initial_candies_l2757_275737


namespace computer_additions_l2757_275742

/-- The number of additions a computer can perform in 12 hours with pauses -/
def computeAdditions (additionsPerSecond : ℕ) (totalHours : ℕ) (pauseMinutes : ℕ) : ℕ :=
  let workingMinutesPerHour := 60 - pauseMinutes
  let workingSecondsPerHour := workingMinutesPerHour * 60
  let additionsPerHour := additionsPerSecond * workingSecondsPerHour
  additionsPerHour * totalHours

/-- Theorem stating that a computer with given specifications performs 540,000,000 additions in 12 hours -/
theorem computer_additions :
  computeAdditions 15000 12 10 = 540000000 := by
  sorry

end computer_additions_l2757_275742


namespace new_person_weight_is_106_l2757_275783

/-- The number of persons in the initial group -/
def initial_group_size : ℕ := 12

/-- The increase in average weight when the new person joins (in kg) -/
def average_weight_increase : ℝ := 4

/-- The weight of the person being replaced (in kg) -/
def replaced_person_weight : ℝ := 58

/-- The weight of the new person (in kg) -/
def new_person_weight : ℝ := 106

/-- Theorem stating that the weight of the new person is 106 kg -/
theorem new_person_weight_is_106 :
  new_person_weight = replaced_person_weight + initial_group_size * average_weight_increase :=
by sorry

end new_person_weight_is_106_l2757_275783


namespace quadrilateral_with_equal_sine_sums_l2757_275764

/-- A convex quadrilateral with angles α, β, γ, δ -/
structure ConvexQuadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 360
  all_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < δ

/-- Definition of a parallelogram -/
def is_parallelogram (q : ConvexQuadrilateral) : Prop :=
  q.α = q.γ ∧ q.β = q.δ

/-- Definition of a trapezoid -/
def is_trapezoid (q : ConvexQuadrilateral) : Prop :=
  q.α + q.β = 180 ∨ q.β + q.γ = 180

theorem quadrilateral_with_equal_sine_sums (q : ConvexQuadrilateral)
  (h : Real.sin q.α + Real.sin q.γ = Real.sin q.β + Real.sin q.δ) :
  is_parallelogram q ∨ is_trapezoid q :=
sorry

end quadrilateral_with_equal_sine_sums_l2757_275764


namespace f_odd_iff_a_b_zero_l2757_275755

/-- The function f defined with parameters a and b -/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ x * |x + a| + b

/-- f is an odd function if and only if a^2 + b^2 = 0 -/
theorem f_odd_iff_a_b_zero (a b : ℝ) :
  (∀ x, f a b (-x) = -(f a b x)) ↔ a^2 + b^2 = 0 := by
  sorry

end f_odd_iff_a_b_zero_l2757_275755


namespace jacques_initial_gumballs_l2757_275743

theorem jacques_initial_gumballs : ℕ :=
  let joanna_initial : ℕ := 40
  let purchase_multiplier : ℕ := 4
  let final_each : ℕ := 250
  let jacques_initial : ℕ := 60

  have h1 : joanna_initial + jacques_initial + purchase_multiplier * (joanna_initial + jacques_initial) = 2 * final_each :=
    by sorry

  jacques_initial

end jacques_initial_gumballs_l2757_275743


namespace quadratic_radicals_sum_product_l2757_275751

theorem quadratic_radicals_sum_product (a b c d e : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (he : 0 ≤ e)
  (h1 : a = 3) (h2 : b = 5) (h3 : c = 7) (h4 : d = 9) (h5 : e = 11) :
  (Real.sqrt a - 1 + Real.sqrt b - Real.sqrt a + Real.sqrt c - Real.sqrt b + 
   Real.sqrt d - Real.sqrt c + Real.sqrt e - Real.sqrt d) * 
  (Real.sqrt e + 1) = 10 := by
  sorry

-- Additional lemmas to represent the given conditions
lemma quadratic_radical_diff (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  2 / (Real.sqrt a + Real.sqrt b) = Real.sqrt a - Real.sqrt b := by
  sorry

lemma quadratic_radical_sum (a : ℝ) (ha : 0 ≤ a) :
  Real.sqrt (a + 2 * Real.sqrt (a - 1)) = Real.sqrt a + 1 := by
  sorry

end quadratic_radicals_sum_product_l2757_275751


namespace all_propositions_false_l2757_275709

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relationships between lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem all_propositions_false :
  ∀ (m n : Line) (α : Plane),
    m ≠ n →
    (¬ (parallel_line_plane m α ∧ parallel_line_plane n α → parallel_lines m n)) ∧
    (¬ (parallel_lines m n ∧ line_in_plane n α → parallel_line_plane m α)) ∧
    (¬ (perpendicular_line_plane m α ∧ perpendicular_lines m n → parallel_line_plane n α)) ∧
    (¬ (parallel_line_plane m α ∧ perpendicular_lines m n → perpendicular_line_plane n α)) :=
by sorry

end all_propositions_false_l2757_275709


namespace equal_cost_at_60_messages_l2757_275718

/-- Represents the cost of a text messaging plan -/
structure PlanCost where
  perMessage : ℚ
  monthlyFee : ℚ

/-- Calculates the total cost for a given number of messages -/
def totalCost (plan : PlanCost) (messages : ℕ) : ℚ :=
  plan.perMessage * messages + plan.monthlyFee

/-- The two text messaging plans offered by the cell phone company -/
def planA : PlanCost := { perMessage := 0.25, monthlyFee := 9 }
def planB : PlanCost := { perMessage := 0.40, monthlyFee := 0 }

theorem equal_cost_at_60_messages :
  ∃ (messages : ℕ), messages = 60 ∧ totalCost planA messages = totalCost planB messages :=
by sorry

end equal_cost_at_60_messages_l2757_275718


namespace expression_values_l2757_275721

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let e := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  e = 5 ∨ e = 1 ∨ e = -3 :=
sorry

end expression_values_l2757_275721


namespace total_toys_l2757_275741

theorem total_toys (num_dolls : ℕ) (h1 : num_dolls = 18) : ℕ :=
  let total := 4 * num_dolls / 3
  have h2 : total = 24 := by sorry
  total

#check total_toys

end total_toys_l2757_275741


namespace short_trees_after_planting_l2757_275798

/-- The number of short trees in the park after planting -/
def total_short_trees (
  initial_short_oak : ℕ)
  (initial_short_pine : ℕ)
  (initial_short_maple : ℕ)
  (new_short_oak : ℕ)
  (new_short_pine : ℕ)
  (new_short_maple : ℕ) : ℕ :=
  initial_short_oak + initial_short_pine + initial_short_maple +
  new_short_oak + new_short_pine + new_short_maple

/-- Theorem stating the total number of short trees after planting -/
theorem short_trees_after_planting :
  total_short_trees 3 4 5 9 6 4 = 31 := by
  sorry

end short_trees_after_planting_l2757_275798


namespace smallest_difference_sides_l2757_275785

theorem smallest_difference_sides (DE EF FD : ℕ) : 
  (DE < EF ∧ EF ≤ FD) →
  (DE + EF + FD = 1801) →
  (DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF) →
  (∀ DE' EF' FD' : ℕ, 
    (DE' < EF' ∧ EF' ≤ FD') →
    (DE' + EF' + FD' = 1801) →
    (DE' + EF' > FD' ∧ EF' + FD' > DE' ∧ FD' + DE' > EF') →
    EF' - DE' ≥ EF - DE) →
  EF - DE = 1 := by
sorry

end smallest_difference_sides_l2757_275785


namespace difference_of_squares_and_sum_l2757_275790

theorem difference_of_squares_and_sum (m n : ℤ) 
  (h1 : m^2 - n^2 = 6) 
  (h2 : m + n = 3) : 
  n - m = -2 := by sorry

end difference_of_squares_and_sum_l2757_275790


namespace shenzhen_metro_growth_l2757_275702

/-- Represents the passenger growth of Shenzhen Metro Line 11 -/
theorem shenzhen_metro_growth (x : ℝ) : 
  (1.2 : ℝ) * (1 + x)^2 = 1.75 ↔ 
  120 * (1 + x)^2 = 175 := by sorry

#check shenzhen_metro_growth

end shenzhen_metro_growth_l2757_275702


namespace union_of_A_and_B_when_m_is_3_necessary_but_not_sufficient_condition_l2757_275738

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 2*m + 1}

-- Part 1
theorem union_of_A_and_B_when_m_is_3 :
  A ∪ B 3 = {x : ℝ | -1 ≤ x ∧ x < 7} := by sorry

-- Part 2
theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m) ↔ m ≤ -2 ∨ (0 ≤ m ∧ m ≤ 1) := by sorry

end union_of_A_and_B_when_m_is_3_necessary_but_not_sufficient_condition_l2757_275738


namespace quadratic_equation_integer_roots_l2757_275713

theorem quadratic_equation_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 - (a+8)*x + 8*a - 1 = 0 ∧ y^2 - (a+8)*y + 8*a - 1 = 0) → 
  a = 8 := by
sorry

end quadratic_equation_integer_roots_l2757_275713


namespace angle_measure_l2757_275773

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end angle_measure_l2757_275773


namespace triangle_abc_proof_l2757_275722

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  b = 4 →
  (1/2) * b * a * Real.sin C = 6 * Real.sqrt 3 →
  Real.sqrt 3 * a * Real.cos C - c * Real.sin A = 0 →
  C = π / 3 ∧ c = 2 * Real.sqrt 7 := by
  sorry

end triangle_abc_proof_l2757_275722


namespace complex_division_l2757_275777

theorem complex_division (z : ℂ) (h1 : z.re = 1) (h2 : z.im = -2) : 
  (5 * Complex.I) / z = -2 + Complex.I :=
sorry

end complex_division_l2757_275777


namespace largest_angle_in_special_triangle_l2757_275724

theorem largest_angle_in_special_triangle : 
  ∀ (y : ℝ), 
    60 + 70 + y = 180 →  -- Sum of angles in a triangle
    y = 70 + 15 →        -- y is 15° more than the second smallest angle (70°)
    max 60 (max 70 y) = 85 :=  -- The largest angle is 85°
by
  sorry

end largest_angle_in_special_triangle_l2757_275724


namespace divisibility_by_11_l2757_275768

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def seven_digit_number (m : ℕ) : ℕ :=
  856 * 10000 + m * 1000 + 248

theorem divisibility_by_11 (m : ℕ) : 
  is_divisible_by_11 (seven_digit_number m) ↔ m = 4 := by
sorry

end divisibility_by_11_l2757_275768


namespace paired_with_32_l2757_275765

def numbers : List ℕ := [36, 27, 42, 32, 28, 31, 23, 17]

theorem paired_with_32 (pair_sum : ℕ) 
  (h1 : pair_sum = (numbers.sum / 4))
  (h2 : ∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → a + b = pair_sum → a ≠ b)
  (h3 : ∀ (n : ℕ), n ∈ numbers → ∃ (m : ℕ), m ∈ numbers ∧ m ≠ n ∧ n + m = pair_sum) :
  ∃ (pairs : List (ℕ × ℕ)), 
    pairs.length = 4 ∧ 
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 = pair_sum) ∧
    (32, 27) ∈ pairs :=
sorry

end paired_with_32_l2757_275765


namespace paradise_park_ferris_wheel_seat_capacity_l2757_275704

/-- Represents a Ferris wheel with a given number of seats and total capacity. -/
structure FerrisWheel where
  numSeats : ℕ
  totalCapacity : ℕ

/-- Calculates the capacity of each seat in a Ferris wheel. -/
def seatCapacity (wheel : FerrisWheel) : ℕ :=
  wheel.totalCapacity / wheel.numSeats

theorem paradise_park_ferris_wheel_seat_capacity :
  let wheel : FerrisWheel := { numSeats := 14, totalCapacity := 84 }
  seatCapacity wheel = 6 := by
  sorry

end paradise_park_ferris_wheel_seat_capacity_l2757_275704


namespace range_f_minus_g_theorem_l2757_275786

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define the range of f + g
def range_f_plus_g (f g : ℝ → ℝ) : Set ℝ := Set.range (λ x ↦ f x + g x)

-- Define the range of f - g
def range_f_minus_g (f g : ℝ → ℝ) : Set ℝ := Set.range (λ x ↦ f x - g x)

-- State the theorem
theorem range_f_minus_g_theorem (hf : is_odd f) (hg : is_even g) 
  (h_range : range_f_plus_g f g = Set.Icc 1 3) : 
  range_f_minus_g f g = Set.Ioc (-3) (-1) := by
  sorry

end range_f_minus_g_theorem_l2757_275786


namespace cube_difference_not_divisible_l2757_275780

theorem cube_difference_not_divisible (a b : ℤ) 
  (ha : Odd a) (hb : Odd b) (hab : a ≠ b) : 
  ¬ (2 * (a - b) ∣ (a^3 - b^3)) := by
  sorry

end cube_difference_not_divisible_l2757_275780


namespace linda_coin_ratio_l2757_275775

/-- Represents the coin types in Linda's bag -/
inductive Coin
  | Dime
  | Quarter
  | Nickel

/-- Represents Linda's initial coin counts -/
structure InitialCoins where
  dimes : Nat
  quarters : Nat
  nickels : Nat

/-- Represents the additional coins given by Linda's mother -/
structure AdditionalCoins where
  dimes : Nat
  quarters : Nat

def total_coins : Nat := 35

theorem linda_coin_ratio 
  (initial : InitialCoins)
  (additional : AdditionalCoins)
  (h_initial_dimes : initial.dimes = 2)
  (h_initial_quarters : initial.quarters = 6)
  (h_initial_nickels : initial.nickels = 5)
  (h_additional_dimes : additional.dimes = 2)
  (h_additional_quarters : additional.quarters = 10)
  (h_total_coins : total_coins = 35) :
  (total_coins - (initial.dimes + additional.dimes + initial.quarters + additional.quarters) - initial.nickels) / initial.nickels = 2 := by
  sorry


end linda_coin_ratio_l2757_275775


namespace total_insects_l2757_275757

def insect_collection (R S C P B E : ℕ) : Prop :=
  R = 15 ∧
  S = 2 * R - 8 ∧
  C = R / 2 + 3 ∧
  P = 3 * S + 7 ∧
  B = 4 * C - 2 ∧
  E = 3 * (R + S + C + P + B)

theorem total_insects (R S C P B E : ℕ) :
  insect_collection R S C P B E →
  R + S + C + P + B + E = 652 :=
by sorry

end total_insects_l2757_275757


namespace point_on_y_axis_l2757_275748

/-- A point P on the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point P with coordinates (a^2 - 1, a + 1) -/
def P (a : ℝ) : Point := ⟨a^2 - 1, a + 1⟩

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (p : Point) : Prop := p.x = 0

/-- Theorem: If P(a^2 - 1, a + 1) is on the y-axis, then its coordinates are (0, 2) or (0, 0) -/
theorem point_on_y_axis (a : ℝ) : 
  on_y_axis (P a) → (P a = ⟨0, 2⟩ ∨ P a = ⟨0, 0⟩) := by
  sorry

end point_on_y_axis_l2757_275748


namespace least_five_digit_square_cube_l2757_275793

theorem least_five_digit_square_cube : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  (∃ a : ℕ, n = a^2) ∧ 
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, m < n → ¬(m ≥ 10000 ∧ m < 100000 ∧ (∃ x : ℕ, m = x^2) ∧ (∃ y : ℕ, m = y^3))) ∧
  n = 15625 := by
sorry

end least_five_digit_square_cube_l2757_275793


namespace characterize_satisfying_functions_l2757_275753

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

/-- The theorem stating the characterization of functions satisfying the equation -/
theorem characterize_satisfying_functions :
  ∀ f : ℤ → ℤ, SatisfiesEquation f →
    (∀ n : ℤ, f n = 0) ∨ (∃ K : ℤ, ∀ n : ℤ, f n = 2 * n + K) := by
  sorry

end characterize_satisfying_functions_l2757_275753


namespace sophias_book_length_l2757_275700

theorem sophias_book_length :
  ∀ (total_pages : ℕ),
  (2 : ℚ) / 3 * total_pages = (total_pages / 2 : ℚ) + 45 →
  total_pages = 4556 :=
by
  sorry

end sophias_book_length_l2757_275700


namespace cylinder_surface_area_l2757_275772

theorem cylinder_surface_area (h : ℝ) (d : ℝ) (cylinder_height : h = 2) (sphere_diameter : d = 2 * Real.sqrt 6) :
  let r := Real.sqrt (((d / 2) ^ 2 - (h / 2) ^ 2))
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = (10 + 4 * Real.sqrt 5) * Real.pi := by
sorry

end cylinder_surface_area_l2757_275772


namespace cube_sum_divided_by_quadratic_difference_l2757_275796

theorem cube_sum_divided_by_quadratic_difference (a c : ℝ) (h1 : a = 6) (h2 : c = 3) :
  (a^3 + c^3) / (a^2 - a*c + c^2) = 9 := by
  sorry

end cube_sum_divided_by_quadratic_difference_l2757_275796


namespace simplest_quadratic_radical_l2757_275714

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ y : ℝ, x = Real.sqrt y ∧ 
  (∀ z : ℝ, z > 0 → z ≠ y → ¬∃ (a b : ℝ), a^2 * b = y ∧ b > 0 ∧ b ≠ 1)

theorem simplest_quadratic_radical : 
  is_simplest_quadratic_radical (Real.sqrt 6) ∧ 
  ¬is_simplest_quadratic_radical (Real.sqrt 27) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 9) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/4)) :=
sorry

end simplest_quadratic_radical_l2757_275714


namespace quadratic_equation_nonnegative_solutions_l2757_275745

theorem quadratic_equation_nonnegative_solutions :
  ∃! x : ℝ, x ≥ 0 ∧ x^2 = -6*x :=
by
  sorry

end quadratic_equation_nonnegative_solutions_l2757_275745


namespace count_solutions_l2757_275774

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem count_solutions : 
  (Finset.filter (fun n => n + S n + S (S n) = 2023) (Finset.range 2024)).card = 4 := by sorry

end count_solutions_l2757_275774


namespace point_move_left_l2757_275746

def number_line_move (initial_position : ℝ) (move_distance : ℝ) : ℝ :=
  initial_position - move_distance

theorem point_move_left :
  let initial_position : ℝ := -4
  let move_distance : ℝ := 2
  number_line_move initial_position move_distance = -6 := by
sorry

end point_move_left_l2757_275746


namespace movie_profit_l2757_275731

def movie_production (main_actor_fee supporting_actor_fee extra_fee : ℕ)
                     (main_actor_food supporting_actor_food crew_food : ℕ)
                     (post_production_cost revenue : ℕ) : Prop :=
  let main_actors := 2
  let supporting_actors := 3
  let extras := 1
  let total_people := 50
  let actor_fees := main_actors * main_actor_fee + 
                    supporting_actors * supporting_actor_fee + 
                    extras * extra_fee
  let food_cost := main_actors * main_actor_food + 
                   (supporting_actors + extras) * supporting_actor_food + 
                   (total_people - main_actors - supporting_actors - extras) * crew_food
  let equipment_rental := 2 * (actor_fees + food_cost)
  let total_cost := actor_fees + food_cost + equipment_rental + post_production_cost
  let profit := revenue - total_cost
  profit = 4584

theorem movie_profit :
  movie_production 500 100 50 10 5 3 850 10000 :=
by sorry

end movie_profit_l2757_275731


namespace fourth_term_is_one_l2757_275797

-- Define the geometric progression
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the specific sequence
def our_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = (3 : ℝ) ^ (1/3) ∧ 
  a 2 = (3 : ℝ) ^ (1/4) ∧ 
  a 3 = (3 : ℝ) ^ (1/12)

-- State the theorem
theorem fourth_term_is_one 
  (a : ℕ → ℝ) 
  (h1 : geometric_progression a) 
  (h2 : our_sequence a) : 
  a 4 = 1 := by
  sorry


end fourth_term_is_one_l2757_275797


namespace curve_C_equation_m_equilateral_triangle_m_range_dot_product_negative_l2757_275736

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > 0 ∧ p.1^2 = 4 * p.2}

-- Define point F
def F : ℝ × ℝ := (0, 1)

-- Define the line y = -2
def line_y_neg_2 (x : ℝ) : ℝ := -2

-- Define the distance condition
def distance_condition (p : ℝ × ℝ) : Prop :=
  Real.sqrt ((p.1 - F.1)^2 + (p.2 - F.2)^2) + 1 = p.2 - line_y_neg_2 p.1

-- Define the theorem for the equation of curve C
theorem curve_C_equation :
  ∀ p : ℝ × ℝ, p ∈ C ↔ p.2 > 0 ∧ p.1^2 = 4 * p.2 :=
sorry

-- Define the theorem for the value of m when triangle AFB is equilateral
theorem m_equilateral_triangle :
  ∀ m : ℝ, (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧
    (A.2 = B.2) ∧ (A.1 = -B.1) ∧
    Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) ∧
    Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  (m = 7 + 4 * Real.sqrt 3 ∨ m = 7 - 4 * Real.sqrt 3) :=
sorry

-- Define the theorem for the range of m when FA · FB < 0
theorem m_range_dot_product_negative :
  ∀ m : ℝ, (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧
    ((A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) < 0)) →
  (3 - 2 * Real.sqrt 2 < m ∧ m < 3 + 2 * Real.sqrt 2) :=
sorry

end curve_C_equation_m_equilateral_triangle_m_range_dot_product_negative_l2757_275736


namespace expression_evaluation_l2757_275728

theorem expression_evaluation :
  let a : ℝ := 1
  let b : ℝ := -2
  a * (a - 2*b) + (a + b)^2 - (a + b)*(a - b) = 9 := by
sorry

end expression_evaluation_l2757_275728


namespace sufficient_not_necessary_condition_l2757_275795

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b + 1 → a > b) ∧
  (∃ a b, a > b ∧ ¬(a > b + 1)) := by
  sorry

end sufficient_not_necessary_condition_l2757_275795


namespace monogram_count_is_66_l2757_275740

/-- The number of letters available for the first two initials -/
def n : ℕ := 12

/-- The number of initials to choose (first and middle) -/
def k : ℕ := 2

/-- The number of ways to choose k distinct letters from n letters in alphabetical order -/
def monogram_count : ℕ := Nat.choose n k

/-- Theorem stating that the number of possible monograms is 66 -/
theorem monogram_count_is_66 : monogram_count = 66 := by
  sorry

end monogram_count_is_66_l2757_275740
