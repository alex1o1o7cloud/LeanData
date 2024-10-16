import Mathlib

namespace NUMINAMATH_CALUDE_triangle_4_5_6_l2576_257625

/-- A triangle can be formed from three line segments if the sum of any two sides is greater than the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that line segments of lengths 4, 5, and 6 can form a triangle. -/
theorem triangle_4_5_6 : can_form_triangle 4 5 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_4_5_6_l2576_257625


namespace NUMINAMATH_CALUDE_bathing_suits_for_men_l2576_257639

theorem bathing_suits_for_men (total : ℕ) (women : ℕ) (men : ℕ) : 
  total = 19766 → women = 4969 → men = total - women → men = 14797 := by
  sorry

end NUMINAMATH_CALUDE_bathing_suits_for_men_l2576_257639


namespace NUMINAMATH_CALUDE_solve_equation_l2576_257693

theorem solve_equation : ∃ x : ℚ, 5 * x - 3 * x = 405 - 9 * (x + 4) → x = 369 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2576_257693


namespace NUMINAMATH_CALUDE_remaining_payment_l2576_257666

/-- Given a 10% deposit of $55, prove that the remaining amount to be paid is $495. -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (total_cost : ℝ) : 
  deposit = 55 ∧ 
  deposit_percentage = 0.1 ∧ 
  deposit = deposit_percentage * total_cost →
  total_cost - deposit = 495 := by
sorry

end NUMINAMATH_CALUDE_remaining_payment_l2576_257666


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_area_relation_l2576_257695

/-- Given a quadrilateral with area Q divided by its diagonals into 4 triangles with areas A, B, C, and D,
    prove that A * B * C * D = ((A+B)^2 * (B+C)^2 * (C+D)^2 * (D+A)^2) / Q^4 -/
theorem quadrilateral_diagonal_area_relation (Q A B C D : ℝ) 
    (hQ : Q > 0) 
    (hA : A > 0) (hB : B > 0) (hC : C > 0) (hD : D > 0)
    (hSum : A + B + C + D = Q) : 
  A * B * C * D = ((A+B)^2 * (B+C)^2 * (C+D)^2 * (D+A)^2) / Q^4 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_area_relation_l2576_257695


namespace NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l2576_257685

theorem four_digit_number_with_specific_remainders :
  ∃! N : ℕ,
    N % 131 = 112 ∧
    N % 132 = 98 ∧
    1000 ≤ N ∧
    N < 10000 ∧
    N = 1946 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l2576_257685


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l2576_257619

/-- The quadratic function f(x) = (x-1)^2 - 2 -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

/-- The theorem stating that the vertex of f(x) = (x-1)^2 - 2 is at (1, -2) -/
theorem vertex_of_quadratic :
  (∃ (a : ℝ), ∀ x, f x = a * (x - 1)^2 - 2) →
  (∀ x, f x ≥ f 1) ∧ f 1 = -2 :=
by sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l2576_257619


namespace NUMINAMATH_CALUDE_greatest_n_value_exists_greatest_n_l2576_257692

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 8100) : n ≤ 8 :=
sorry

theorem exists_greatest_n :
  ∃ n : ℤ, n = 8 ∧ 101 * n^2 ≤ 8100 ∧ ∀ m : ℤ, 101 * m^2 ≤ 8100 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_n_value_exists_greatest_n_l2576_257692


namespace NUMINAMATH_CALUDE_badminton_players_l2576_257653

theorem badminton_players (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 30)
  (h_tennis : tennis = 21)
  (h_neither : neither = 2)
  (h_both : both = 10) :
  ∃ badminton : ℕ, badminton = 17 ∧ 
    total = tennis + badminton - both + neither :=
by sorry

end NUMINAMATH_CALUDE_badminton_players_l2576_257653


namespace NUMINAMATH_CALUDE_strongest_correlation_l2576_257605

-- Define the correlation coefficients
def r₁ : ℝ := 0
def r₂ : ℝ := -0.95
def r₃ : ℝ := 0.89  -- We use the absolute value directly as it's given
def r₄ : ℝ := 0.75

-- Theorem stating that r₂ has the largest absolute value
theorem strongest_correlation :
  abs r₂ > abs r₁ ∧ abs r₂ > abs r₃ ∧ abs r₂ > abs r₄ := by
  sorry


end NUMINAMATH_CALUDE_strongest_correlation_l2576_257605


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2576_257690

theorem polynomial_factorization (a b c : ℝ) : 
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  (a - b) * (b - c) * (c - a) * (a^6*b + a^6*c + b^6*a + b^6*c + c^6*a + c^6*b) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2576_257690


namespace NUMINAMATH_CALUDE_all_statements_false_l2576_257617

-- Define prime and composite numbers
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬(isPrime n)

-- Define the four statements
def statement1 : Prop := ∀ p q : ℕ, isPrime p → isPrime q → isComposite (p + q)
def statement2 : Prop := ∀ a b : ℕ, isComposite a → isComposite b → isComposite (a + b)
def statement3 : Prop := ∀ p c : ℕ, isPrime p → isComposite c → isComposite (p + c)
def statement4 : Prop := ∀ p c : ℕ, isPrime p → isComposite c → ¬(isComposite (p + c))

-- Theorem stating that all four statements are false
theorem all_statements_false : ¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4 :=
sorry

end NUMINAMATH_CALUDE_all_statements_false_l2576_257617


namespace NUMINAMATH_CALUDE_power_tower_mod_1000_l2576_257624

theorem power_tower_mod_1000 : 5^(5^(5^5)) ≡ 625 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_1000_l2576_257624


namespace NUMINAMATH_CALUDE_park_tree_removal_l2576_257676

/-- The number of trees removed from a park -/
def trees_removed (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem: Given 6 initial trees and 2 remaining trees, 4 trees are removed -/
theorem park_tree_removal :
  trees_removed 6 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_park_tree_removal_l2576_257676


namespace NUMINAMATH_CALUDE_dans_helmet_craters_l2576_257604

theorem dans_helmet_craters :
  ∀ (d D R r : ℕ),
  D = d + 10 →                   -- Dan's helmet has 10 more craters than Daniel's
  R = D + d + 15 →               -- Rin's helmet has 15 more craters than Dan's and Daniel's combined
  r = 2 * R - 10 →               -- Rina's helmet has double the number of craters in Rin's minus 10
  R = 75 →                       -- Rin's helmet has 75 craters
  d + D + R + r = 540 →          -- Total craters on all helmets is 540
  Even d ∧ Even D ∧ Even R ∧ Even r →  -- Number of craters in each helmet is even
  D = 168 :=
by sorry

end NUMINAMATH_CALUDE_dans_helmet_craters_l2576_257604


namespace NUMINAMATH_CALUDE_factorial_sum_div_l2576_257665

theorem factorial_sum_div (n : ℕ) : (8 * n.factorial + 9 * 8 * n.factorial + 10 * 9 * 8 * n.factorial) / n.factorial = 800 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_div_l2576_257665


namespace NUMINAMATH_CALUDE_custom_set_op_theorem_l2576_257654

-- Define the custom set operation ⊗
def customSetOp (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ (A ∪ B) ∧ x ∉ (A ∩ B)}

-- Define sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N : Set ℝ := {x | 1 < x ∧ x < 4}

-- Theorem statement
theorem custom_set_op_theorem :
  customSetOp M N = {x | (-2 < x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x < 4)} := by
  sorry

end NUMINAMATH_CALUDE_custom_set_op_theorem_l2576_257654


namespace NUMINAMATH_CALUDE_emily_furniture_time_l2576_257658

def chairs : ℕ := 4
def tables : ℕ := 2
def time_per_piece : ℕ := 8

theorem emily_furniture_time : chairs + tables * time_per_piece = 48 := by
  sorry

end NUMINAMATH_CALUDE_emily_furniture_time_l2576_257658


namespace NUMINAMATH_CALUDE_survey_mn_value_l2576_257663

/-- Proves that mn = 2.5 given the survey conditions --/
theorem survey_mn_value (total : ℕ) (table_tennis basketball soccer : ℕ) 
  (h1 : total = 100)
  (h2 : table_tennis = 40)
  (h3 : (table_tennis : ℚ) / total = 2/5)
  (h4 : (basketball : ℚ) / total = 1/4)
  (h5 : soccer = total - (table_tennis + basketball))
  (h6 : (soccer : ℚ) / total = (soccer : ℚ) / 100) :
  (basketball : ℚ) * ((soccer : ℚ) / 100) = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_survey_mn_value_l2576_257663


namespace NUMINAMATH_CALUDE_parabola_intersection_l2576_257662

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 12 * x + 15
  let g (x : ℝ) := 2 * x^2 - 8 * x + 12
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 1 ∧ y = 6) ∨ (x = 3 ∧ y = 6) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2576_257662


namespace NUMINAMATH_CALUDE_division_equals_fraction_l2576_257697

theorem division_equals_fraction : 200 / (12 + 15 * 2 - 4)^2 = 50 / 361 := by
  sorry

end NUMINAMATH_CALUDE_division_equals_fraction_l2576_257697


namespace NUMINAMATH_CALUDE_sin_three_pi_over_four_l2576_257637

theorem sin_three_pi_over_four : Real.sin (3 * π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_over_four_l2576_257637


namespace NUMINAMATH_CALUDE_gathering_handshakes_l2576_257669

/-- Represents a gathering of couples -/
structure Gathering where
  couples : Nat
  people : Nat
  men : Nat
  women : Nat

/-- Calculates the number of handshakes in a gathering -/
def handshakes (g : Gathering) : Nat :=
  g.men * (g.women - 1)

/-- Theorem: In a gathering of 7 couples with the given handshake rules, 
    the total number of handshakes is 42 -/
theorem gathering_handshakes :
  ∀ g : Gathering, 
    g.couples = 7 →
    g.people = 2 * g.couples →
    g.men = g.couples →
    g.women = g.couples →
    handshakes g = 42 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l2576_257669


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2576_257612

theorem quadratic_equation_solution (x : ℝ) (h1 : x^2 - 6*x = 0) (h2 : x ≠ 0) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2576_257612


namespace NUMINAMATH_CALUDE_valid_arrangements_l2576_257646

/-- The number of boys. -/
def num_boys : ℕ := 4

/-- The number of girls. -/
def num_girls : ℕ := 3

/-- The number of people to be selected. -/
def num_selected : ℕ := 3

/-- The number of tasks. -/
def num_tasks : ℕ := 3

/-- The function to calculate the number of permutations. -/
def permutations (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

/-- The theorem stating the number of valid arrangements. -/
theorem valid_arrangements : 
  permutations (num_boys + num_girls) num_selected - 
  permutations num_boys num_selected = 186 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_l2576_257646


namespace NUMINAMATH_CALUDE_triangle_properties_l2576_257623

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine law for triangle ABC -/
def cosineLaw (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - 2 * t.b * t.c * Real.cos t.A

/-- The given condition for the triangle -/
def givenCondition (t : Triangle) : Prop :=
  3 * t.a * Real.cos t.A = t.b * Real.cos t.C + t.c * Real.cos t.B

theorem triangle_properties (t : Triangle) 
  (h1 : givenCondition t) 
  (h2 : 0 < t.A ∧ t.A < π) : 
  Real.cos t.A = 1/3 ∧ 
  (t.a = 3 → 
    ∃ (S : ℝ), S = (9 * Real.sqrt 2) / 4 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * t.b * t.c * Real.sin t.A → S' ≤ S) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2576_257623


namespace NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l2576_257648

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_tangent_to_parallel_lines (x y : ℝ) :
  (3 * x - 4 * y = 12 ∨ 3 * x - 4 * y = -48) ∧ 
  (x - 2 * y = 0) →
  x = -18 ∧ y = -9 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l2576_257648


namespace NUMINAMATH_CALUDE_odd_function_properties_l2576_257609

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (b - 2^x) / (2^x + a)

theorem odd_function_properties (a b : ℝ) 
  (h_odd : ∀ x, f a b x = -f a b (-x)) :
  (a = 1 ∧ b = 1) ∧
  (∀ t : ℝ, f 1 1 (t^2 - 2*t) + f 1 1 (2*t^2 - 1) < 0 ↔ t > 1 ∨ t < -1/3) :=
sorry

end NUMINAMATH_CALUDE_odd_function_properties_l2576_257609


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2576_257635

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 → (∃ r : ℝ, 10 * r = b ∧ b * r = 2/3) → b = 2 * Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2576_257635


namespace NUMINAMATH_CALUDE_proposition_p_q_equivalence_l2576_257683

theorem proposition_p_q_equivalence (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) ∧
  (∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0) ↔
  2 < m ∧ m < 3 :=
sorry

end NUMINAMATH_CALUDE_proposition_p_q_equivalence_l2576_257683


namespace NUMINAMATH_CALUDE_tangent_line_ratio_l2576_257622

theorem tangent_line_ratio (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) : 
  (∃ (m b : ℝ), 
    (∀ x, m * x + b = x^2 ↔ x = x₁) ∧ 
    (∀ x, m * x + b = x^3 ↔ x = x₂)) → 
  x₁ / x₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_ratio_l2576_257622


namespace NUMINAMATH_CALUDE_zacks_marbles_l2576_257633

theorem zacks_marbles (friend1 friend2 friend3 friend4 friend5 friend6 remaining : ℕ) 
  (h1 : friend1 = 20)
  (h2 : friend2 = 30)
  (h3 : friend3 = 35)
  (h4 : friend4 = 25)
  (h5 : friend5 = 28)
  (h6 : friend6 = 40)
  (h7 : remaining = 7) :
  friend1 + friend2 + friend3 + friend4 + friend5 + friend6 + remaining = 185 := by
  sorry

end NUMINAMATH_CALUDE_zacks_marbles_l2576_257633


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2576_257671

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2576_257671


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l2576_257627

/-- The number of people at the circular table, including Cara -/
def total_people : ℕ := 7

/-- The number of Cara's friends -/
def num_friends : ℕ := 6

/-- Alex is one of Cara's friends -/
def alex_is_friend : Prop := true

/-- The number of different pairs Cara could be sitting between, where one must be Alex -/
def num_seating_arrangements : ℕ := 5

theorem cara_seating_arrangements :
  total_people = num_friends + 1 →
  alex_is_friend →
  num_seating_arrangements = num_friends - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l2576_257627


namespace NUMINAMATH_CALUDE_inequality_proof_l2576_257651

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 8) :
  (x + 1/y)^2 + (y + 1/x)^2 ≥ 289/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2576_257651


namespace NUMINAMATH_CALUDE_prime_quadratic_solution_l2576_257696

theorem prime_quadratic_solution : 
  {n : ℕ+ | Nat.Prime (n^4 - 27*n^2 + 121)} = {2, 5} := by sorry

end NUMINAMATH_CALUDE_prime_quadratic_solution_l2576_257696


namespace NUMINAMATH_CALUDE_proportion_sum_l2576_257684

theorem proportion_sum (x y : ℝ) : 
  (31.25 : ℝ) / x = 100 / (9.6 : ℝ) ∧ x / 13.75 = (9.6 : ℝ) / y → x + y = 47 := by
  sorry

end NUMINAMATH_CALUDE_proportion_sum_l2576_257684


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l2576_257687

theorem recipe_flour_amount (flour_added : ℕ) (flour_needed : ℕ) : 
  flour_added = 4 → flour_needed = 4 → flour_added + flour_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l2576_257687


namespace NUMINAMATH_CALUDE_five_student_committees_from_eight_l2576_257636

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of 5-student committees from 8 students is 56 -/
theorem five_student_committees_from_eight (n k : ℕ) (hn : n = 8) (hk : k = 5) :
  binomial n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_student_committees_from_eight_l2576_257636


namespace NUMINAMATH_CALUDE_extremum_implies_f_2_eq_18_l2576_257699

/-- A function f with an extremum at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2_eq_18 (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → f a b 2 = 18 := by
  sorry

#check extremum_implies_f_2_eq_18

end NUMINAMATH_CALUDE_extremum_implies_f_2_eq_18_l2576_257699


namespace NUMINAMATH_CALUDE_store_discount_l2576_257640

theorem store_discount (original_price : ℝ) (original_price_pos : 0 < original_price) : 
  let sale_price := 0.5 * original_price
  let coupon_discount := 0.2
  let promotion_discount := 0.1
  let final_price := (1 - promotion_discount) * ((1 - coupon_discount) * sale_price)
  (original_price - final_price) / original_price = 0.64 :=
by sorry

end NUMINAMATH_CALUDE_store_discount_l2576_257640


namespace NUMINAMATH_CALUDE_number_relation_l2576_257652

theorem number_relation (A B : ℝ) (h : A = B * (1 + 0.1)) : B = A * (10/11) := by
  sorry

end NUMINAMATH_CALUDE_number_relation_l2576_257652


namespace NUMINAMATH_CALUDE_vectors_are_collinear_l2576_257631

/-- Two 2D vectors are collinear if their cross product is zero -/
def are_collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem vectors_are_collinear : 
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (6, -4)
  are_collinear a b :=
by sorry

end NUMINAMATH_CALUDE_vectors_are_collinear_l2576_257631


namespace NUMINAMATH_CALUDE_not_p_or_not_q_is_true_l2576_257642

-- Define propositions p and q
variable (p q : Prop)

-- Define the conditions
axiom p_true : p
axiom q_false : ¬q

-- Theorem to prove
theorem not_p_or_not_q_is_true : ¬p ∨ ¬q := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_is_true_l2576_257642


namespace NUMINAMATH_CALUDE_system_real_solutions_l2576_257608

/-- The system of equations has real solutions if and only if p ≤ 0, q ≥ 0, and p^2 - 4q ≥ 0 -/
theorem system_real_solutions (p q : ℝ) :
  (∃ (x y z : ℝ), (Real.sqrt x + Real.sqrt y = z) ∧
                   (2 * x + 2 * y + p = 0) ∧
                   (z^4 + p * z^2 + q = 0)) ↔
  (p ≤ 0 ∧ q ≥ 0 ∧ p^2 - 4*q ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_system_real_solutions_l2576_257608


namespace NUMINAMATH_CALUDE_jose_share_of_profit_l2576_257691

/-- Calculates the share of profit for an investor based on their investment amount, duration, and total profit -/
def calculate_share_of_profit (investment : ℕ) (duration : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * duration * total_profit) / total_investment_months

theorem jose_share_of_profit (tom_investment : ℕ) (tom_duration : ℕ) (jose_investment : ℕ) (jose_duration : ℕ) (total_profit : ℕ) :
  tom_investment = 30000 →
  tom_duration = 12 →
  jose_investment = 45000 →
  jose_duration = 10 →
  total_profit = 27000 →
  calculate_share_of_profit jose_investment jose_duration (tom_investment * tom_duration + jose_investment * jose_duration) total_profit = 15000 := by
  sorry

#eval calculate_share_of_profit 45000 10 810000 27000

end NUMINAMATH_CALUDE_jose_share_of_profit_l2576_257691


namespace NUMINAMATH_CALUDE_fraction_difference_squared_l2576_257600

theorem fraction_difference_squared (a b : ℝ) (h : 1/a - 1/b = 1/(a + b)) :
  1/a^2 - 1/b^2 = 1/(a*b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_squared_l2576_257600


namespace NUMINAMATH_CALUDE_equation_solutions_l2576_257667

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 3 ∧ x₂ = 2 - Real.sqrt 3 ∧
    x₁^2 - 4*x₁ + 1 = 0 ∧ x₂^2 - 4*x₂ + 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 2/5 ∧ x₂ = -5/3 ∧
    5*x₁ - 2 = (2 - 5*x₁)*(3*x₁ + 4) ∧ 5*x₂ - 2 = (2 - 5*x₂)*(3*x₂ + 4)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2576_257667


namespace NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l2576_257621

/-- A Pythagorean triple is a set of three positive integers (a, b, c) where a² + b² = c² --/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (6, 8, 10) is a Pythagorean triple --/
theorem six_eight_ten_pythagorean_triple : is_pythagorean_triple 6 8 10 := by
  sorry

end NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l2576_257621


namespace NUMINAMATH_CALUDE_min_sum_of_dimensions_l2576_257660

theorem min_sum_of_dimensions (l w h : ℕ) : 
  l > 0 → w > 0 → h > 0 → l * w * h = 2310 → 
  ∀ a b c : ℕ, a > 0 → b > 0 → c > 0 → a * b * c = 2310 → 
  l + w + h ≤ a + b + c → l + w + h = 42 := by sorry

end NUMINAMATH_CALUDE_min_sum_of_dimensions_l2576_257660


namespace NUMINAMATH_CALUDE_candy_bar_profit_l2576_257650

/-- Candy Bar Problem -/
theorem candy_bar_profit :
  ∀ (total_bars : ℕ) (buy_price : ℚ) (sell_price : ℚ),
    total_bars = 1200 →
    buy_price = 1 / 3 →
    sell_price = 3 / 5 →
    (sell_price - buy_price) * total_bars = 320 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l2576_257650


namespace NUMINAMATH_CALUDE_student_fraction_l2576_257673

theorem student_fraction (total : ℕ) (third_year : ℕ) (second_year : ℕ) 
  (h1 : third_year = total * 30 / 100)
  (h2 : second_year = total * 10 / 100) :
  second_year / (total - third_year) = 1 / 7 :=
sorry

end NUMINAMATH_CALUDE_student_fraction_l2576_257673


namespace NUMINAMATH_CALUDE_xy_sum_l2576_257664

theorem xy_sum (x y : ℕ+) (h : (2 * x - 5) * (2 * y - 5) = 25) :
  x + y = 10 ∨ x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_l2576_257664


namespace NUMINAMATH_CALUDE_fourth_year_afforestation_l2576_257647

/-- The area afforested in a given year, starting from an initial area and increasing by a fixed percentage annually. -/
def afforestedArea (initialArea : ℝ) (increaseRate : ℝ) (year : ℕ) : ℝ :=
  initialArea * (1 + increaseRate) ^ (year - 1)

/-- Theorem stating that given an initial afforestation of 10,000 acres and an annual increase of 20%, 
    the area afforested in the fourth year is 17,280 acres. -/
theorem fourth_year_afforestation :
  afforestedArea 10000 0.2 4 = 17280 := by
  sorry

end NUMINAMATH_CALUDE_fourth_year_afforestation_l2576_257647


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l2576_257602

/-- Represents the value of one trillion in scientific notation -/
def trillion : ℝ := 10^12

/-- The gross domestic product in trillion yuan -/
def gdp : ℝ := 114

/-- The gross domestic product expressed in scientific notation -/
def gdp_scientific : ℝ := 1.14 * 10^14

theorem gdp_scientific_notation :
  gdp * trillion = gdp_scientific := by sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l2576_257602


namespace NUMINAMATH_CALUDE_arrangement_counts_l2576_257686

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

def girls_not_adjacent : ℕ := sorry

def boys_adjacent : ℕ := sorry

def girl_A_not_left_B_not_right : ℕ := sorry

def girls_ABC_height_order : ℕ := sorry

theorem arrangement_counts :
  girls_not_adjacent = 1440 ∧
  boys_adjacent = 576 ∧
  girl_A_not_left_B_not_right = 3720 ∧
  girls_ABC_height_order = 840 := by sorry

end NUMINAMATH_CALUDE_arrangement_counts_l2576_257686


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l2576_257656

theorem six_digit_divisibility (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  let n := 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c
  n % 7 = 0 ∧ n % 11 = 0 ∧ n % 13 = 0 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l2576_257656


namespace NUMINAMATH_CALUDE_sphere_prism_area_difference_l2576_257655

theorem sphere_prism_area_difference :
  let r : ℝ := 2  -- radius of the sphere
  let a : ℝ := 2  -- base edge length of the prism
  let sphere_surface_area : ℝ := 4 * π * r^2
  let max_prism_lateral_area : ℝ := 16 * Real.sqrt 2
  sphere_surface_area - max_prism_lateral_area = 16 * (π - Real.sqrt 2) := by
sorry


end NUMINAMATH_CALUDE_sphere_prism_area_difference_l2576_257655


namespace NUMINAMATH_CALUDE_abc_product_l2576_257674

theorem abc_product (a b c : ℤ) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 30)
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 630 / (a * b * c) = 1) :
  a * b * c = 483 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l2576_257674


namespace NUMINAMATH_CALUDE_sum_of_prime_and_odd_l2576_257638

theorem sum_of_prime_and_odd (a b : ℕ) : 
  Nat.Prime a → Odd b → a^2 + b = 2009 → a + b = 2007 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_and_odd_l2576_257638


namespace NUMINAMATH_CALUDE_move_point_left_l2576_257645

/-- Given a point A in a 2D Cartesian coordinate system, moving it
    3 units to the left results in a new point A' -/
def move_left (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 - 3, A.2)

/-- Theorem: Moving point A(1, -1) 3 units to the left results in A'(-2, -1) -/
theorem move_point_left : 
  let A : ℝ × ℝ := (1, -1)
  move_left A = (-2, -1) := by
sorry

end NUMINAMATH_CALUDE_move_point_left_l2576_257645


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2576_257618

/-- The complex number z = (1+2i)/i lies in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (1 + 2*Complex.I) / Complex.I
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2576_257618


namespace NUMINAMATH_CALUDE_exchange_rate_proof_l2576_257611

-- Define the given quantities
def jack_pounds : ℝ := 42
def jack_euros : ℝ := 11
def jack_yen : ℝ := 3000
def pounds_per_euro : ℝ := 2
def total_yen : ℝ := 9400

-- Define the exchange rate we want to prove
def yen_per_pound : ℝ := 100

-- Theorem statement
theorem exchange_rate_proof :
  (jack_pounds + jack_euros * pounds_per_euro) * yen_per_pound + jack_yen = total_yen :=
by sorry

end NUMINAMATH_CALUDE_exchange_rate_proof_l2576_257611


namespace NUMINAMATH_CALUDE_invisible_square_exists_l2576_257643

/-- A point (x, y) is invisible if gcd(x, y) > 1 -/
def invisible (x y : ℤ) : Prop := Nat.gcd x.natAbs y.natAbs > 1

/-- For any natural number L, there exists integers a and b such that
    for all integers i and j where 0 ≤ i, j ≤ L, the point (a+i, b+j) is invisible -/
theorem invisible_square_exists (L : ℕ) :
  ∃ a b : ℤ, ∀ i j : ℤ, 0 ≤ i ∧ i ≤ L ∧ 0 ≤ j ∧ j ≤ L →
    invisible (a + i) (b + j) := by
  sorry

end NUMINAMATH_CALUDE_invisible_square_exists_l2576_257643


namespace NUMINAMATH_CALUDE_complete_square_form_l2576_257614

theorem complete_square_form (x : ℝ) : 
  (∃ a b : ℝ, (-x + 1) * (1 - x) = (a + b)^2) ∧ 
  (¬∃ a b : ℝ, (1 + x) * (1 - x) = (a + b)^2) ∧ 
  (¬∃ a b : ℝ, (-x - 1) * (-1 + x) = (a + b)^2) ∧ 
  (¬∃ a b : ℝ, (x - 1) * (1 + x) = (a + b)^2) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_form_l2576_257614


namespace NUMINAMATH_CALUDE_factorization_proof_l2576_257616

theorem factorization_proof (z : ℂ) : 55 * z^17 + 121 * z^34 = 11 * z^17 * (5 + 11 * z^17) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2576_257616


namespace NUMINAMATH_CALUDE_simplify_radical_sum_l2576_257661

theorem simplify_radical_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l2576_257661


namespace NUMINAMATH_CALUDE_chess_game_probabilities_l2576_257632

/-- The probability of winning a single game -/
def prob_win : ℝ := 0.4

/-- The probability of not losing a single game -/
def prob_not_lose : ℝ := 0.9

/-- The probability of a draw in a single game -/
def prob_draw : ℝ := prob_not_lose - prob_win

/-- The probability of winning at least one game out of two independent games -/
def prob_win_at_least_one : ℝ := 1 - (1 - prob_win) ^ 2

theorem chess_game_probabilities :
  prob_draw = 0.5 ∧ prob_win_at_least_one = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probabilities_l2576_257632


namespace NUMINAMATH_CALUDE_balloon_problem_l2576_257607

-- Define the variables
def x : ℕ := 10
def y : ℕ := 46
def z : ℕ := 10
def d : ℕ := 16

-- Define the total initial number of balloons
def total_initial : ℕ := x + y

-- Define the final remaining number of balloons
def final_remaining : ℕ := total_initial - d

-- Define the total amount spent
def total_spent : ℕ := total_initial * z

-- Theorem statement
theorem balloon_problem :
  final_remaining = 40 ∧ total_spent = 560 := by
  sorry

end NUMINAMATH_CALUDE_balloon_problem_l2576_257607


namespace NUMINAMATH_CALUDE_exist_three_similar_numbers_l2576_257630

/-- A function that repeats a given 3-digit number to form a 1995-digit number -/
def repeat_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is a 1995-digit number -/
def is_1995_digit (n : ℕ) : Prop := sorry

/-- Predicate to check if two numbers use the same set of digits -/
def same_digit_set (a b : ℕ) : Prop := sorry

/-- Predicate to check if a number contains the digit 0 -/
def contains_zero (n : ℕ) : Prop := sorry

theorem exist_three_similar_numbers :
  ∃ (A B C : ℕ),
    is_1995_digit A ∧
    is_1995_digit B ∧
    is_1995_digit C ∧
    same_digit_set A B ∧
    same_digit_set B C ∧
    ¬contains_zero A ∧
    ¬contains_zero B ∧
    ¬contains_zero C ∧
    A + B = C :=
  sorry

end NUMINAMATH_CALUDE_exist_three_similar_numbers_l2576_257630


namespace NUMINAMATH_CALUDE_cos_45_degrees_l2576_257649

theorem cos_45_degrees : Real.cos (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_degrees_l2576_257649


namespace NUMINAMATH_CALUDE_f_4_1981_l2576_257668

/-- Definition of the function f satisfying the given conditions -/
def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| x + 1, 0 => f x 1
| x + 1, y + 1 => f x (f (x + 1) y)

/-- Theorem stating that f(4, 1981) equals 2^1984 - 3 -/
theorem f_4_1981 : f 4 1981 = 2^1984 - 3 := by
  sorry

end NUMINAMATH_CALUDE_f_4_1981_l2576_257668


namespace NUMINAMATH_CALUDE_light_ray_reflection_and_tangent_l2576_257688

/-- A light ray problem with reflection and tangent to a circle -/
theorem light_ray_reflection_and_tangent 
  (A : ℝ × ℝ) 
  (h_A : A = (-3, 3))
  (C : Set (ℝ × ℝ))
  (h_C : C = {(x, y) | x^2 + y^2 - 4*x - 4*y + 7 = 0}) :
  ∃ (incident_ray reflected_ray : Set (ℝ × ℝ)) (distance : ℝ),
    -- Incident ray equation
    incident_ray = {(x, y) | 4*x + 3*y + 3 = 0} ∧
    -- Reflected ray equation
    reflected_ray = {(x, y) | 3*x + 4*y - 3 = 0} ∧
    -- Reflected ray is tangent to circle C
    ∃ (p : ℝ × ℝ), p ∈ C ∧ p ∈ reflected_ray ∧
      ∀ (q : ℝ × ℝ), q ∈ C ∩ reflected_ray → q = p ∧
    -- Distance traveled
    distance = 7 ∧
    -- Distance is from A to tangent point
    ∃ (tangent_point : ℝ × ℝ), 
      tangent_point ∈ C ∧ 
      tangent_point ∈ reflected_ray ∧
      Real.sqrt ((A.1 - tangent_point.1)^2 + (A.2 - tangent_point.2)^2) +
      Real.sqrt ((0 - tangent_point.1)^2 + (0 - tangent_point.2)^2) = distance :=
by sorry

end NUMINAMATH_CALUDE_light_ray_reflection_and_tangent_l2576_257688


namespace NUMINAMATH_CALUDE_mojave_population_increase_l2576_257677

/-- The factor by which the population of Mojave has increased over a decade -/
def population_increase_factor (initial_population : ℕ) (future_population : ℕ) (future_increase_percent : ℕ) : ℚ :=
  let current_population := (100 : ℚ) / (100 + future_increase_percent) * future_population
  current_population / initial_population

/-- Theorem stating that the population increase factor is 3 -/
theorem mojave_population_increase : 
  population_increase_factor 4000 16800 40 = 3 := by sorry

end NUMINAMATH_CALUDE_mojave_population_increase_l2576_257677


namespace NUMINAMATH_CALUDE_prime_power_divisibility_l2576_257672

theorem prime_power_divisibility (p : ℕ) (x : ℕ) (h_prime : Nat.Prime p) :
  1 ≤ x ∧ x ≤ 2 * p ∧ (x ^ (p - 1) ∣ (p - 1) ^ x + 1) →
  (p = 2 ∧ (x = 1 ∨ x = 2)) ∨ (p = 3 ∧ (x = 1 ∨ x = 3)) ∨ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_prime_power_divisibility_l2576_257672


namespace NUMINAMATH_CALUDE_integer_solutions_for_mn_squared_equation_l2576_257681

theorem integer_solutions_for_mn_squared_equation : 
  ∀ (m n : ℤ), m * n^2 = 2009 * (n + 1) ↔ (m = 4018 ∧ n = 1) ∨ (m = 0 ∧ n = -1) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_for_mn_squared_equation_l2576_257681


namespace NUMINAMATH_CALUDE_proposition_q_false_l2576_257610

theorem proposition_q_false (h1 : ¬(p ∧ q)) (h2 : ¬(¬p)) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_proposition_q_false_l2576_257610


namespace NUMINAMATH_CALUDE_parallelogram_area_l2576_257698

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 and 20 is equal to 100√3. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 20) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 100 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2576_257698


namespace NUMINAMATH_CALUDE_x_squared_plus_nine_y_squared_l2576_257634

theorem x_squared_plus_nine_y_squared (x y : ℝ) 
  (h1 : x + 3 * y = 5) (h2 : x * y = -8) : 
  x^2 + 9 * y^2 = 73 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_nine_y_squared_l2576_257634


namespace NUMINAMATH_CALUDE_larger_box_capacity_l2576_257628

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.height * d.width * d.length

/-- The capacity of clay a box can carry -/
def boxCapacity (volume : ℝ) (clayPerUnit : ℝ) : ℝ :=
  volume * clayPerUnit

theorem larger_box_capacity 
  (small_box : BoxDimensions)
  (small_box_clay : ℝ)
  (h_small_height : small_box.height = 1)
  (h_small_width : small_box.width = 2)
  (h_small_length : small_box.length = 4)
  (h_small_capacity : small_box_clay = 30) :
  let large_box : BoxDimensions := {
    height := 3 * small_box.height,
    width := 2 * small_box.width,
    length := 2 * small_box.length
  }
  let small_volume := boxVolume small_box
  let large_volume := boxVolume large_box
  let clay_per_unit := small_box_clay / small_volume
  boxCapacity large_volume clay_per_unit = 360 := by
sorry

end NUMINAMATH_CALUDE_larger_box_capacity_l2576_257628


namespace NUMINAMATH_CALUDE_ellipse_standard_form_l2576_257644

/-- Given an ellipse with equation (x²/a²) + (y²/b²) = 1, where a > b > 0,
    passing through the point (√6/2, 1/2), and having an eccentricity of √2/2,
    prove that the standard form of the ellipse equation is (x²/a²) + (y²/b²) = 1 -/
theorem ellipse_standard_form 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (Real.sqrt 6 / 2)^2 / a^2 + (1/2)^2 / b^2 = 1)
  (h4 : Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2) :
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_form_l2576_257644


namespace NUMINAMATH_CALUDE_norma_cards_l2576_257613

theorem norma_cards (initial_cards lost_cards remaining_cards : ℕ) :
  lost_cards = 70 →
  remaining_cards = 18 →
  initial_cards = lost_cards + remaining_cards →
  initial_cards = 88 := by
sorry

end NUMINAMATH_CALUDE_norma_cards_l2576_257613


namespace NUMINAMATH_CALUDE_special_parallelogram_sides_prove_special_parallelogram_sides_l2576_257680

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  -- The perimeter of the parallelogram
  perimeter : ℝ
  -- The measure of the acute angle in degrees
  acuteAngle : ℝ
  -- The ratio in which the diagonal divides the obtuse angle
  diagonalRatio : ℝ × ℝ
  -- The sides of the parallelogram
  sides : ℝ × ℝ × ℝ × ℝ

/-- The theorem stating the properties of the special parallelogram -/
theorem special_parallelogram_sides (p : SpecialParallelogram) :
  p.perimeter = 90 ∧ 
  p.acuteAngle = 60 ∧ 
  p.diagonalRatio = (1, 3) →
  p.sides = (15, 15, 30, 30) := by
  sorry

/-- Proof that the sides of the special parallelogram are 15, 15, 30, and 30 -/
theorem prove_special_parallelogram_sides : 
  ∃ (p : SpecialParallelogram), 
    p.perimeter = 90 ∧ 
    p.acuteAngle = 60 ∧ 
    p.diagonalRatio = (1, 3) ∧ 
    p.sides = (15, 15, 30, 30) := by
  sorry

end NUMINAMATH_CALUDE_special_parallelogram_sides_prove_special_parallelogram_sides_l2576_257680


namespace NUMINAMATH_CALUDE_cricket_average_increase_l2576_257679

theorem cricket_average_increase 
  (score_19th_inning : ℕ) 
  (average_after_19 : ℚ) 
  (h1 : score_19th_inning = 97) 
  (h2 : average_after_19 = 25) : 
  average_after_19 - (((19 * average_after_19) - score_19th_inning) / 18) = 4 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l2576_257679


namespace NUMINAMATH_CALUDE_complex_fourth_power_magnitude_l2576_257615

theorem complex_fourth_power_magnitude : 
  Complex.abs ((5 + 2 * Complex.I * Real.sqrt 3) ^ 4) = 1369 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_magnitude_l2576_257615


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_times_two_l2576_257620

theorem sin_sixty_degrees_times_two : 2 * Real.sin (π / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_times_two_l2576_257620


namespace NUMINAMATH_CALUDE_qinJiushao_V₁_for_f_10_l2576_257626

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qinJiushao (f : ℝ → ℝ) (x : ℝ) : ℝ := sorry

/-- The polynomial f(x) = 3x⁴ + 2x² + x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

/-- V₁ in Qin Jiushao's algorithm for f(10) -/
def V₁ : ℝ := 3 * 10 + 2

theorem qinJiushao_V₁_for_f_10 : 
  V₁ = 32 := by sorry

end NUMINAMATH_CALUDE_qinJiushao_V₁_for_f_10_l2576_257626


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2576_257657

/-- The hyperbola equation -/
def hyperbola_equation (x y a : ℝ) : Prop :=
  y^2 / (2 * a^2) - x^2 / a^2 = 1

/-- The asymptote equation -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±√2x -/
theorem hyperbola_asymptotes (a : ℝ) (h : a ≠ 0) :
  ∀ x y : ℝ, hyperbola_equation x y a → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2576_257657


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_fourth_powers_equality_condition_l2576_257659

theorem min_value_of_sum_of_fourth_powers (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ((a + b) / c) ^ 4 + ((b + c) / d) ^ 4 + ((c + d) / a) ^ 4 + ((d + a) / b) ^ 4 ≥ 64 :=
by sorry

theorem equality_condition (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ((a + b) / c) ^ 4 + ((b + c) / d) ^ 4 + ((c + d) / a) ^ 4 + ((d + a) / b) ^ 4 = 64 ↔ 
  a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_fourth_powers_equality_condition_l2576_257659


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l2576_257689

theorem scooter_gain_percent : 
  let initial_cost : ℚ := 900
  let repair1 : ℚ := 150
  let repair2 : ℚ := 75
  let repair3 : ℚ := 225
  let selling_price : ℚ := 1800
  let total_cost : ℚ := initial_cost + repair1 + repair2 + repair3
  let gain : ℚ := selling_price - total_cost
  let gain_percent : ℚ := (gain / total_cost) * 100
  gain_percent = 33.33 := by sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l2576_257689


namespace NUMINAMATH_CALUDE_polynomial_correction_l2576_257603

theorem polynomial_correction (P : ℝ → ℝ) :
  (∀ x, P x + 3 * x^2 = x^2 - (1/2) * x + 1) →
  (∀ x, P x * (-3 * x^2) = -12 * x^4 + (3/2) * x^3 - 3 * x^2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_correction_l2576_257603


namespace NUMINAMATH_CALUDE_second_year_interest_l2576_257678

/-- Compound interest calculation -/
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n - P

/-- Theorem: Given compound interest for third year and interest rate, calculate second year interest -/
theorem second_year_interest (P : ℝ) (r : ℝ) (CI_3 : ℝ) :
  r = 0.06 → CI_3 = 1272 → compound_interest P r 2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_second_year_interest_l2576_257678


namespace NUMINAMATH_CALUDE_max_value_of_function_l2576_257606

theorem max_value_of_function (x : ℝ) (h : x ∈ Set.Ioo 0 (1/4)) :
  x * (1 - 4*x) ≤ (1/8) * (1 - 4*(1/8)) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2576_257606


namespace NUMINAMATH_CALUDE_nancy_books_count_l2576_257682

/-- Given that Alyssa has 36 books and Nancy has 7 times more books than Alyssa,
    prove that Nancy has 252 books. -/
theorem nancy_books_count (alyssa_books : ℕ) (nancy_books : ℕ) 
    (h1 : alyssa_books = 36)
    (h2 : nancy_books = 7 * alyssa_books) : 
  nancy_books = 252 := by
  sorry

end NUMINAMATH_CALUDE_nancy_books_count_l2576_257682


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2576_257675

theorem diophantine_equation_solutions (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  let solutions := {(a, b) : ℕ × ℕ | (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / (p * q)}
  solutions = {
    (1 + p*q, p^2*q^2 + p*q),
    (p*(q + 1), p*q*(q + 1)),
    (q*(p + 1), p*q*(p + 1)),
    (2*p*q, 2*p*q),
    (p^2*q*(p + q), q^2 + p*q),
    (q^2 + p*q, p^2 + p*q),
    (p*q*(p + 1), q*(p + 1)),
    (p*q*(q + 1), p*(q + 1)),
    (p^2*q^2 + p*q, 1 + p*q)
  } := by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2576_257675


namespace NUMINAMATH_CALUDE_percentage_60to69_is_20_percent_l2576_257670

/-- Represents the score ranges in the class --/
inductive ScoreRange
  | Below60
  | Range60to69
  | Range70to79
  | Range80to89
  | Range90to100

/-- The frequency of students for each score range --/
def frequency (range : ScoreRange) : Nat :=
  match range with
  | .Below60 => 2
  | .Range60to69 => 5
  | .Range70to79 => 6
  | .Range80to89 => 8
  | .Range90to100 => 4

/-- The total number of students in the class --/
def totalStudents : Nat :=
  frequency ScoreRange.Below60 +
  frequency ScoreRange.Range60to69 +
  frequency ScoreRange.Range70to79 +
  frequency ScoreRange.Range80to89 +
  frequency ScoreRange.Range90to100

/-- The percentage of students in the 60%-69% range --/
def percentageIn60to69Range : Rat :=
  (frequency ScoreRange.Range60to69 : Rat) / (totalStudents : Rat) * 100

theorem percentage_60to69_is_20_percent :
  percentageIn60to69Range = 20 := by
  sorry

#eval percentageIn60to69Range

end NUMINAMATH_CALUDE_percentage_60to69_is_20_percent_l2576_257670


namespace NUMINAMATH_CALUDE_final_wage_calculation_l2576_257694

/-- Calculates the final wage after a raise and a pay cut -/
theorem final_wage_calculation (initial_wage : ℝ) (raise_percentage : ℝ) (pay_cut_percentage : ℝ) :
  initial_wage = 10 →
  raise_percentage = 0.2 →
  pay_cut_percentage = 0.75 →
  initial_wage * (1 + raise_percentage) * pay_cut_percentage = 9 := by
  sorry

#check final_wage_calculation

end NUMINAMATH_CALUDE_final_wage_calculation_l2576_257694


namespace NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l2576_257601

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The original point P -/
def P : Point :=
  { x := -2, y := 3 }

theorem reflection_of_P_across_x_axis :
  reflect_x P = { x := -2, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l2576_257601


namespace NUMINAMATH_CALUDE_equation_solution_l2576_257641

theorem equation_solution : ∃! x : ℝ, (1 : ℝ) / (x + 3) = (3 : ℝ) / (x - 1) ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2576_257641


namespace NUMINAMATH_CALUDE_absent_children_count_l2576_257629

/-- Proves that the number of absent children is 32 given the conditions of the sweet distribution problem --/
theorem absent_children_count (total_children : ℕ) (original_sweets_per_child : ℕ) (extra_sweets : ℕ) : 
  total_children = 112 →
  original_sweets_per_child = 15 →
  extra_sweets = 6 →
  (total_children - (total_children - 32)) * (original_sweets_per_child + extra_sweets) = total_children * original_sweets_per_child :=
by sorry

end NUMINAMATH_CALUDE_absent_children_count_l2576_257629
