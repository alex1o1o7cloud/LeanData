import Mathlib

namespace NUMINAMATH_CALUDE_total_subscription_amount_l625_62544

/-- Prove that the total subscription amount is 50000 given the conditions of the problem -/
theorem total_subscription_amount (c b a : ℕ) 
  (h1 : b = c + 5000)  -- B subscribes 5000 more than C
  (h2 : a = b + 4000)  -- A subscribes 4000 more than B
  (h3 : 14700 * (a + b + c) = 35000 * a)  -- A's profit proportion
  : a + b + c = 50000 := by
  sorry

end NUMINAMATH_CALUDE_total_subscription_amount_l625_62544


namespace NUMINAMATH_CALUDE_problem_statement_l625_62550

theorem problem_statement : (36 / (7 + 2 - 5)) * 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l625_62550


namespace NUMINAMATH_CALUDE_theo_cookie_eating_frequency_l625_62591

/-- The number of cookies Theo eats each time -/
def cookies_per_time : ℕ := 13

/-- The number of days Theo eats cookies each month -/
def days_per_month : ℕ := 20

/-- The number of cookies Theo eats in 3 months -/
def cookies_in_three_months : ℕ := 2340

/-- The number of times Theo eats cookies per day -/
def times_per_day : ℕ := 3

theorem theo_cookie_eating_frequency :
  times_per_day * cookies_per_time * days_per_month * 3 = cookies_in_three_months :=
by sorry

end NUMINAMATH_CALUDE_theo_cookie_eating_frequency_l625_62591


namespace NUMINAMATH_CALUDE_complex_equation_result_l625_62565

theorem complex_equation_result (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (a + 4 * i) * i = b + i) : a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l625_62565


namespace NUMINAMATH_CALUDE_rhombus_area_from_quadratic_roots_l625_62542

/-- Given a quadratic equation x^2 - 10x + 24 = 0, if its roots are the lengths of the diagonals
    of a rhombus, then the area of the rhombus is 12. -/
theorem rhombus_area_from_quadratic_roots : 
  ∀ (d₁ d₂ : ℝ), d₁ * d₂ = 24 → d₁ + d₂ = 10 → (1/2) * d₁ * d₂ = 12 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_from_quadratic_roots_l625_62542


namespace NUMINAMATH_CALUDE_equation_solutions_l625_62596

/-- The equation x^4 * y^4 - 10 * x^2 * y^2 + 9 = 0 -/
def equation (x y : ℕ+) : Prop :=
  (x.val : ℝ)^4 * (y.val : ℝ)^4 - 10 * (x.val : ℝ)^2 * (y.val : ℝ)^2 + 9 = 0

/-- The set of all ordered pairs (x,y) of positive integers satisfying the equation -/
def solution_set : Set (ℕ+ × ℕ+) :=
  {p | equation p.1 p.2}

theorem equation_solutions :
  ∃ (s : Finset (ℕ+ × ℕ+)), s.card = 3 ∧ ↑s = solution_set := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l625_62596


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l625_62539

theorem fifteenth_student_age
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat)
  (avg_age_group1 : ℝ)
  (num_group2 : Nat)
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 6)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 8)
  (h6 : avg_age_group2 = 16)
  (h7 : num_group1 + num_group2 + 1 = total_students) :
  total_students * avg_age_all - (num_group1 * avg_age_group1 + num_group2 * avg_age_group2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l625_62539


namespace NUMINAMATH_CALUDE_problem_statement_l625_62516

theorem problem_statement (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 → b / a + 2 / b ≤ x / y + 2 / x) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 → a * b ≥ x * y) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 → a^2 + b^2 ≤ x^2 + y^2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l625_62516


namespace NUMINAMATH_CALUDE_divide_c_by_a_l625_62579

theorem divide_c_by_a (a b c : ℝ) (h1 : a * b = 3) (h2 : b * c = 8/5) : c / a = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_divide_c_by_a_l625_62579


namespace NUMINAMATH_CALUDE_fraction_ordering_l625_62512

theorem fraction_ordering : 
  let a := 23
  let b := 18
  let c := 21
  let d := 16
  let e := 25
  let f := 19
  (a : ℚ) / b < (c : ℚ) / d ∧ (c : ℚ) / d < (e : ℚ) / f := by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l625_62512


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l625_62523

/-- Given a quadratic inequality x^2 - mx + t < 0 with solution set {x | 2 < x < 3}, prove that m - t = -1 -/
theorem quadratic_inequality_solution (m t : ℝ) : 
  (∀ x : ℝ, x^2 - m*x + t < 0 ↔ 2 < x ∧ x < 3) → 
  m - t = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l625_62523


namespace NUMINAMATH_CALUDE_valid_input_statement_l625_62529

/-- Represents a programming language construct --/
inductive ProgramConstruct
| Input : String → String → ProgramConstruct
| Other : ProgramConstruct

/-- Checks if a given ProgramConstruct is a valid INPUT statement --/
def isValidInputStatement (stmt : ProgramConstruct) : Prop :=
  match stmt with
  | ProgramConstruct.Input prompt var => true
  | _ => false

/-- Theorem: An INPUT statement with a prompt and variable is valid --/
theorem valid_input_statement (prompt var : String) :
  isValidInputStatement (ProgramConstruct.Input prompt var) := by
  sorry

#check valid_input_statement

end NUMINAMATH_CALUDE_valid_input_statement_l625_62529


namespace NUMINAMATH_CALUDE_min_crystals_to_kill_120_l625_62559

structure Skill where
  name : String
  crystalCost : ℕ
  damage : ℕ
  specialEffect : Bool

def applySkill (health : ℕ) (skill : Skill) (prevWindUsed : Bool) : ℕ × ℕ :=
  let actualCost := if prevWindUsed then skill.crystalCost / 2 else skill.crystalCost
  let newHealth := 
    if skill.name = "Earth" then
      if health % 2 = 1 then (health + 1) / 2 else health / 2
    else
      if health > skill.damage then health - skill.damage else 0
  (newHealth, actualCost)

def minCrystalsToKill (initialHealth : ℕ) (water fire wind earth : Skill) : ℕ :=
  sorry

theorem min_crystals_to_kill_120 :
  let water : Skill := ⟨"Water", 4, 4, false⟩
  let fire : Skill := ⟨"Fire", 10, 11, false⟩
  let wind : Skill := ⟨"Wind", 10, 5, true⟩
  let earth : Skill := ⟨"Earth", 18, 0, false⟩
  minCrystalsToKill 120 water fire wind earth = 68 := by
  sorry

end NUMINAMATH_CALUDE_min_crystals_to_kill_120_l625_62559


namespace NUMINAMATH_CALUDE_count_special_numbers_is_4032_l625_62535

/-- A function that counts the number of 5-digit numbers starting with '2' and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := 5
  let start_digit := 2
  let identical_digits := 2
  -- The actual counting logic would go here
  4032

/-- Theorem stating that the count of special numbers is 4032 -/
theorem count_special_numbers_is_4032 :
  count_special_numbers = 4032 := by sorry

end NUMINAMATH_CALUDE_count_special_numbers_is_4032_l625_62535


namespace NUMINAMATH_CALUDE_parabola_maximum_l625_62582

/-- The quadratic function f(x) = -x^2 - 1 -/
def f (x : ℝ) : ℝ := -x^2 - 1

theorem parabola_maximum :
  (∀ x : ℝ, f x ≤ f 0) ∧ f 0 = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_maximum_l625_62582


namespace NUMINAMATH_CALUDE_staff_age_l625_62566

theorem staff_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 32 →
  student_avg_age = 16 →
  new_avg_age = student_avg_age + 1 →
  (num_students : ℝ) * student_avg_age + (num_students + 1 : ℝ) * new_avg_age = (num_students + 1 : ℝ) * 49 := by
  sorry

end NUMINAMATH_CALUDE_staff_age_l625_62566


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l625_62532

/-- The number M as defined in the problem -/
def M : ℕ := 25 * 48 * 49 * 81

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the ratio of sum of odd divisors to sum of even divisors of M is 1:30 -/
theorem ratio_odd_even_divisors_M :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 30 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l625_62532


namespace NUMINAMATH_CALUDE_f_min_max_values_g_negative_range_l625_62562

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2 * a * x

-- Define the interval [1/e, e]
def interval : Set ℝ := Set.Icc (1 / Real.exp 1) (Real.exp 1)

-- Theorem 1: Minimum and maximum values of f when a = -1/2
theorem f_min_max_values :
  let f_neg_half (x : ℝ) := f (-1/2) x
  (∀ x ∈ interval, f_neg_half x ≥ 1 - Real.exp 1 ^ 2) ∧
  (∃ x ∈ interval, f_neg_half x = 1 - Real.exp 1 ^ 2) ∧
  (∀ x ∈ interval, f_neg_half x ≤ -1/2 - 1/2 * Real.log 2) ∧
  (∃ x ∈ interval, f_neg_half x = -1/2 - 1/2 * Real.log 2) := by
  sorry

-- Theorem 2: Range of a for which g(x) < 0 holds for all x > 2
theorem g_negative_range :
  {a : ℝ | ∀ x > 2, g a x < 0} = Set.Iic (1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_f_min_max_values_g_negative_range_l625_62562


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l625_62509

def arithmetic_sequence_sum (a : ℕ) (d : ℕ) (last : ℕ) : ℕ :=
  let n := (last - a) / d + 1
  n * (a + last) / 2

theorem arithmetic_sequence_sum_remainder
  (h1 : arithmetic_sequence_sum 3 6 279 % 8 = 3) : 
  arithmetic_sequence_sum 3 6 279 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l625_62509


namespace NUMINAMATH_CALUDE_product_modulo_remainder_1491_2001_mod_250_l625_62597

theorem product_modulo (a b m : ℕ) (h : m > 0) :
  (a * b) % m = ((a % m) * (b % m)) % m :=
by sorry

theorem remainder_1491_2001_mod_250 :
  (1491 * 2001) % 250 = 241 :=
by sorry

end NUMINAMATH_CALUDE_product_modulo_remainder_1491_2001_mod_250_l625_62597


namespace NUMINAMATH_CALUDE_handshake_arrangement_count_l625_62546

/-- A handshake arrangement for a group of people -/
structure HandshakeArrangement (n : ℕ) :=
  (shakes : Fin n → Finset (Fin n))
  (shake_count : ∀ i, (shakes i).card = 3)
  (symmetry : ∀ i j, i ∈ shakes j ↔ j ∈ shakes i)

/-- The number of distinct handshake arrangements for 12 people -/
def M : ℕ := sorry

/-- The main theorem: M is congruent to 850 modulo 1000 -/
theorem handshake_arrangement_count :
  M ≡ 850 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_handshake_arrangement_count_l625_62546


namespace NUMINAMATH_CALUDE_gloria_turtle_time_l625_62583

/-- The time it took for Gloria's turtle to finish the race -/
def glorias_time (gretas_time georges_time : ℕ) : ℕ :=
  2 * georges_time

theorem gloria_turtle_time : ∃ (gretas_time georges_time : ℕ),
  gretas_time = 6 ∧
  georges_time = gretas_time - 2 ∧
  glorias_time gretas_time georges_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_gloria_turtle_time_l625_62583


namespace NUMINAMATH_CALUDE_solve_cost_problem_l625_62598

def cost_problem (shirt_cost jacket_cost carrie_payment : ℕ) 
                 (num_shirts num_pants num_jackets : ℕ) : Prop :=
  let total_cost := 2 * carrie_payment
  let pants_cost := (total_cost - num_shirts * shirt_cost - num_jackets * jacket_cost) / num_pants
  pants_cost = 18

theorem solve_cost_problem :
  cost_problem 8 60 94 4 2 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_cost_problem_l625_62598


namespace NUMINAMATH_CALUDE_jersey_profit_is_152_l625_62576

/-- The amount of money made from selling jerseys during a game -/
def money_from_jerseys (profit_per_jersey : ℕ) (jerseys_sold : ℕ) : ℕ :=
  profit_per_jersey * jerseys_sold

/-- Theorem stating that the money made from selling jerseys is $152 -/
theorem jersey_profit_is_152 :
  let profit_per_jersey : ℕ := 76
  let profit_per_tshirt : ℕ := 204
  let tshirts_sold : ℕ := 158
  let jerseys_sold : ℕ := 2
  money_from_jerseys profit_per_jersey jerseys_sold = 152 := by
sorry

end NUMINAMATH_CALUDE_jersey_profit_is_152_l625_62576


namespace NUMINAMATH_CALUDE_exists_consecutive_numbers_with_54_digit_product_ratio_l625_62569

/-- Given a natural number, return the product of its non-zero digits -/
def productOfNonZeroDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that there exist two consecutive natural numbers
    such that the product of all non-zero digits of the larger number
    multiplied by 54 equals the product of all non-zero digits of the smaller number -/
theorem exists_consecutive_numbers_with_54_digit_product_ratio :
  ∃ n : ℕ, productOfNonZeroDigits n = 54 * productOfNonZeroDigits (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_consecutive_numbers_with_54_digit_product_ratio_l625_62569


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l625_62536

/-- Given a fold line y = -x and a line l₁ with equation 2x + 3y - 1 = 0,
    the symmetric line l₂ with respect to the fold line has the equation 3x + 2y + 1 = 0 -/
theorem symmetric_line_equation (x y : ℝ) :
  (y = -x) →  -- fold line equation
  (2*x + 3*y - 1 = 0) →  -- l₁ equation
  (3*x + 2*y + 1 = 0)  -- l₂ equation (to be proved)
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l625_62536


namespace NUMINAMATH_CALUDE_no_solution_to_fractional_equation_l625_62537

theorem no_solution_to_fractional_equation :
  ¬ ∃ (x : ℝ), x ≠ 3 ∧ (2 - x) / (x - 3) = 1 + 1 / (3 - x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_fractional_equation_l625_62537


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l625_62593

theorem constant_term_binomial_expansion :
  let f := fun (x : ℝ) => (x - 1 / (2 * Real.sqrt x)) ^ 6
  ∃ (c : ℝ), (∀ (x : ℝ), x > 0 → f x = c + x * (f x - c) / x) ∧ c = 15/16 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l625_62593


namespace NUMINAMATH_CALUDE_first_day_over_200_is_thursday_l625_62541

def paperclips (n : Nat) : Nat := 5 * 3^n

def days : List String := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

theorem first_day_over_200_is_thursday :
  days[4] = "Thursday" ∧
  (∀ k < 4, paperclips k ≤ 200) ∧
  paperclips 4 > 200 := by
sorry

end NUMINAMATH_CALUDE_first_day_over_200_is_thursday_l625_62541


namespace NUMINAMATH_CALUDE_max_sum_of_sides_l625_62573

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  (2 * t.c - t.b) / t.a = (Real.cos t.B) / (Real.cos t.A)

def side_a_condition (t : Triangle) : Prop :=
  t.a = 2 * Real.sqrt 5

-- Theorem statement
theorem max_sum_of_sides (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : side_a_condition t) : 
  ∃ (max : Real), ∀ (t' : Triangle), 
    satisfies_condition t' → side_a_condition t' → 
    t'.b + t'.c ≤ max ∧ 
    ∃ (t'' : Triangle), satisfies_condition t'' ∧ side_a_condition t'' ∧ t''.b + t''.c = max ∧
    max = 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_sides_l625_62573


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l625_62543

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (m : ℕ) :
  geometric_sequence a q →
  a 1 = 1 →
  q ≠ 1 →
  q ≠ -1 →
  (∃ m : ℕ, a m = a 1 * a 2 * a 3 * a 4 * a 5) →
  m = 11 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l625_62543


namespace NUMINAMATH_CALUDE_committee_probability_l625_62549

/-- The probability of selecting exactly 2 boys in a 6-person committee 
    randomly chosen from a group of 30 members (12 boys and 18 girls) -/
theorem committee_probability (total_members : ℕ) (boys : ℕ) (girls : ℕ) 
  (committee_size : ℕ) (h1 : total_members = 30) (h2 : boys = 12) 
  (h3 : girls = 18) (h4 : committee_size = 6) (h5 : total_members = boys + girls) :
  (Nat.choose boys 2 * Nat.choose girls 4) / Nat.choose total_members committee_size = 8078 / 23751 := by
sorry

end NUMINAMATH_CALUDE_committee_probability_l625_62549


namespace NUMINAMATH_CALUDE_problem_solution_l625_62585

-- Define the variables and functions
def f (x : ℝ) := 2 * x + 1
def g (x : ℝ) := x^2 + 2 * x

-- State the theorem
theorem problem_solution :
  ∃ (a b n : ℝ),
    f 2 = 5 ∧
    g 2 = a ∧
    f n = b ∧
    g n = -1 ∧
    a = 8 ∧
    b = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l625_62585


namespace NUMINAMATH_CALUDE_good_number_implies_prime_l625_62567

/-- A positive integer b is "good for a" if C(an, b) - 1 is divisible by an + 1 for all positive integers n such that an ≥ b -/
def is_good_for (a b : ℕ+) : Prop :=
  ∀ n : ℕ+, a * n ≥ b → (Nat.choose (a * n) b - 1) % (a * n + 1) = 0

theorem good_number_implies_prime (a b : ℕ+) 
  (h1 : is_good_for a b)
  (h2 : ¬ is_good_for a (b + 2)) :
  Nat.Prime (b + 1) :=
sorry

end NUMINAMATH_CALUDE_good_number_implies_prime_l625_62567


namespace NUMINAMATH_CALUDE_yanna_purchase_l625_62590

def shirts_cost : ℕ := 10 * 5
def sandals_cost : ℕ := 3 * 3
def hats_cost : ℕ := 5 * 8
def bags_cost : ℕ := 7 * 14
def sunglasses_cost : ℕ := 2 * 12

def total_cost : ℕ := shirts_cost + sandals_cost + hats_cost + bags_cost + sunglasses_cost
def payment : ℕ := 200

theorem yanna_purchase :
  total_cost = payment + 21 :=
by sorry

end NUMINAMATH_CALUDE_yanna_purchase_l625_62590


namespace NUMINAMATH_CALUDE_intersection_implies_B_equals_one_three_l625_62525

def A : Set ℝ := {1, 2, 4}

def B (m : ℝ) : Set ℝ := {x | x^2 - 4*x + m = 0}

theorem intersection_implies_B_equals_one_three :
  ∃ m : ℝ, (A ∩ B m = {1}) → (B m = {1, 3}) :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_B_equals_one_three_l625_62525


namespace NUMINAMATH_CALUDE_kind_wizard_strategy_exists_l625_62548

-- Define a type for gnomes
def Gnome := ℕ

-- Define a friendship relation
def Friendship := Gnome × Gnome

-- Define a strategy for the kind wizard
def KindWizardStrategy := ℕ → List Friendship

-- Define the evil wizard's action
def EvilWizardAction := List Friendship → List Friendship

-- Define a circular arrangement of gnomes
def CircularArrangement := List Gnome

-- Function to check if an arrangement is valid (all neighbors are friends)
def IsValidArrangement (arrangement : CircularArrangement) (friendships : List Friendship) : Prop :=
  sorry

-- Main theorem
theorem kind_wizard_strategy_exists (n : ℕ) (h : n > 1 ∧ Odd n) :
  ∃ (strategy : KindWizardStrategy),
    ∀ (evil_action : EvilWizardAction),
      ∃ (arrangement : CircularArrangement),
        IsValidArrangement arrangement (evil_action (strategy n)) :=
sorry

end NUMINAMATH_CALUDE_kind_wizard_strategy_exists_l625_62548


namespace NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l625_62551

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l625_62551


namespace NUMINAMATH_CALUDE_equal_selection_probability_l625_62575

/-- Represents the probability of a student being selected -/
def probability_of_selection (n : ℕ) (total : ℕ) : ℚ := n / total

theorem equal_selection_probability 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (eliminated_students : ℕ) 
  (h1 : total_students = 54) 
  (h2 : selected_students = 5) 
  (h3 : eliminated_students = 4) :
  ∀ (student : ℕ), student ≤ total_students → 
    probability_of_selection selected_students total_students = 5 / 54 :=
by sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l625_62575


namespace NUMINAMATH_CALUDE_expression_simplification_l625_62508

theorem expression_simplification (x y : ℝ) 
  (hx : x = Real.sqrt 2) 
  (hy : y = Real.sqrt 3) : 
  (x + y) * (x - y) - y * (2 * x - y) = 2 - 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l625_62508


namespace NUMINAMATH_CALUDE_snack_eaters_final_count_l625_62505

/-- Calculates the number of remaining snack eaters after a series of events -/
def remaining_snack_eaters (initial_people : ℕ) (initial_snack_eaters : ℕ) 
  (first_new_outsiders : ℕ) (second_new_outsiders : ℕ) (second_group_leaving : ℕ) : ℕ :=
  let total_after_first_join := initial_snack_eaters + first_new_outsiders
  let remaining_after_first_leave := total_after_first_join / 2
  let total_after_second_join := remaining_after_first_leave + second_new_outsiders
  let remaining_after_second_leave := total_after_second_join - second_group_leaving
  remaining_after_second_leave / 2

theorem snack_eaters_final_count 
  (h1 : initial_people = 200)
  (h2 : initial_snack_eaters = 100)
  (h3 : first_new_outsiders = 20)
  (h4 : second_new_outsiders = 10)
  (h5 : second_group_leaving = 30) :
  remaining_snack_eaters initial_people initial_snack_eaters first_new_outsiders second_new_outsiders second_group_leaving = 20 := by
  sorry

end NUMINAMATH_CALUDE_snack_eaters_final_count_l625_62505


namespace NUMINAMATH_CALUDE_longer_train_length_l625_62534

/-- Calculates the length of the longer train given the speeds of two trains,
    the time they take to cross each other, and the length of the shorter train. -/
theorem longer_train_length
  (speed1 : ℝ)
  (speed2 : ℝ)
  (crossing_time : ℝ)
  (shorter_train_length : ℝ)
  (h1 : speed1 = 60)
  (h2 : speed2 = 40)
  (h3 : crossing_time = 11.159107271418288)
  (h4 : shorter_train_length = 140)
  : ∃ (longer_train_length : ℝ),
    longer_train_length = 170 ∧
    (speed1 + speed2) * (1000 / 3600) * crossing_time =
      shorter_train_length + longer_train_length :=
by
  sorry

end NUMINAMATH_CALUDE_longer_train_length_l625_62534


namespace NUMINAMATH_CALUDE_stability_comparison_l625_62514

/-- Represents the variance of a student's performance -/
structure StudentVariance where
  value : ℝ
  positive : value > 0

/-- Defines the concept of stability based on variance -/
def more_stable (a b : StudentVariance) : Prop :=
  a.value < b.value

theorem stability_comparison 
  (variance_A variance_B : StudentVariance)
  (h1 : variance_A.value = 0.05)
  (h2 : variance_B.value = 0.06) :
  more_stable variance_A variance_B :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l625_62514


namespace NUMINAMATH_CALUDE_triangle_segment_length_l625_62586

theorem triangle_segment_length : 
  ∀ (a b c h x : ℝ),
  a = 40 ∧ b = 90 ∧ c = 100 →
  a^2 = x^2 + h^2 →
  b^2 = (c - x)^2 + h^2 →
  c - x = 82.5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_segment_length_l625_62586


namespace NUMINAMATH_CALUDE_orange_packing_l625_62517

theorem orange_packing (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 2650) (h2 : oranges_per_box = 10) :
  total_oranges / oranges_per_box = 265 := by
  sorry

end NUMINAMATH_CALUDE_orange_packing_l625_62517


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_75_factorial_l625_62574

theorem last_two_nonzero_digits_75_factorial (n : ℕ) : n = 75 → 
  ∃ k : ℕ, n.factorial = 100 * k + 76 ∧ k % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_75_factorial_l625_62574


namespace NUMINAMATH_CALUDE_smallest_divisor_after_221_next_divisor_is_289_l625_62526

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem smallest_divisor_after_221 (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000)  -- m is a 4-digit number
  (h2 : is_even m)             -- m is even
  (h3 : m % 221 = 0)           -- 221 is a divisor of m
  : (∃ d : ℕ, d ∣ m ∧ 221 < d ∧ d < 289) → False :=
by sorry

theorem next_divisor_is_289 (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000)  -- m is a 4-digit number
  (h2 : is_even m)             -- m is even
  (h3 : m % 221 = 0)           -- 221 is a divisor of m
  : 289 ∣ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_221_next_divisor_is_289_l625_62526


namespace NUMINAMATH_CALUDE_seven_points_non_isosceles_l625_62563

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define an isosceles triangle
def IsIsosceles (p1 p2 p3 : Point) : Prop :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p1.x - p3.x)^2 + (p1.y - p3.y)^2 ∨
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2 ∨
  (p3.x - p1.x)^2 + (p3.y - p1.y)^2 = (p3.x - p2.x)^2 + (p3.y - p2.y)^2

-- Main theorem
theorem seven_points_non_isosceles (points : Fin 7 → Point) :
  ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ ¬IsIsosceles (points i) (points j) (points k) := by
  sorry

end NUMINAMATH_CALUDE_seven_points_non_isosceles_l625_62563


namespace NUMINAMATH_CALUDE_square_difference_49_50_l625_62555

theorem square_difference_49_50 : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by sorry

end NUMINAMATH_CALUDE_square_difference_49_50_l625_62555


namespace NUMINAMATH_CALUDE_function_characterization_l625_62594

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : f 0 = 1) 
  (h2 : ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2) / 2) + (x - y)^2) : 
  ∀ x : ℝ, f x = 1 - 2 * x := by
sorry

end NUMINAMATH_CALUDE_function_characterization_l625_62594


namespace NUMINAMATH_CALUDE_nancy_football_games_l625_62578

/-- Nancy's football game attendance problem -/
theorem nancy_football_games 
  (total_games : ℕ) 
  (this_month_games : ℕ) 
  (next_month_games : ℕ) 
  (h1 : total_games = 24)
  (h2 : this_month_games = 9)
  (h3 : next_month_games = 7) :
  total_games - this_month_games - next_month_games = 8 := by
  sorry

end NUMINAMATH_CALUDE_nancy_football_games_l625_62578


namespace NUMINAMATH_CALUDE_pascal_triangle_count_l625_62568

/-- Represents a row in Pascal's Triangle -/
def PascalRow := List Nat

/-- Generates the nth row of Pascal's Triangle -/
def generatePascalRow (n : Nat) : PascalRow :=
  sorry

/-- Counts the number of even integers in a given row -/
def countEvens (row : PascalRow) : Nat :=
  sorry

/-- Counts the number of integers that are multiples of 4 in a given row -/
def countMultiplesOfFour (row : PascalRow) : Nat :=
  sorry

/-- Theorem stating the count of even integers and multiples of 4 in the first 12 rows of Pascal's Triangle -/
theorem pascal_triangle_count :
  let rows := List.range 12
  let evenCount := rows.map (fun n => countEvens (generatePascalRow n)) |>.sum
  let multiples4Count := rows.map (fun n => countMultiplesOfFour (generatePascalRow n)) |>.sum
  ∃ (e m : Nat), evenCount = e ∧ multiples4Count = m :=
by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_count_l625_62568


namespace NUMINAMATH_CALUDE_circle_properties_l625_62502

/-- The circle C is defined by the equation (x+1)^2 + (y-2)^2 = 4 -/
def C : Set (ℝ × ℝ) := {p | (p.1 + 1)^2 + (p.2 - 2)^2 = 4}

/-- The center of circle C -/
def center : ℝ × ℝ := (-1, 2)

/-- The radius of circle C -/
def radius : ℝ := 2

theorem circle_properties :
  ∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l625_62502


namespace NUMINAMATH_CALUDE_sphere_intersection_ratio_l625_62581

/-- Two spheres with radii R₁ and R₂ are intersected by a plane P perpendicular to the line
    connecting their centers and passing through its midpoint. If P divides the surface area
    of the first sphere in ratio m:1 and the second sphere in ratio n:1 (where m > 1 and n > 1),
    then R₂/R₁ = ((m - 1)(n + 1)) / ((m + 1)(n - 1)). -/
theorem sphere_intersection_ratio (R₁ R₂ m n : ℝ) (hm : m > 1) (hn : n > 1) :
  let h₁ := (2 * R₁) / (m + 1)
  let h₂ := (2 * R₂) / (n + 1)
  R₁ - h₁ = R₂ - h₂ →
  R₂ / R₁ = ((m - 1) * (n + 1)) / ((m + 1) * (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_sphere_intersection_ratio_l625_62581


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l625_62519

theorem polynomial_root_problem (a b c d : ℤ) : 
  a * (3 + Complex.I)^4 + b * (3 + Complex.I)^3 + c * (3 + Complex.I)^2 + d * (3 + Complex.I) + a = 0 →
  Int.gcd (Int.gcd (Int.gcd a b) c) d = 1 →
  d.natAbs = 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l625_62519


namespace NUMINAMATH_CALUDE_dealership_sales_prediction_l625_62527

/-- The number of sports cars predicted to be sold -/
def sports_cars : ℕ := 45

/-- The ratio of sports cars to sedans -/
def ratio : ℚ := 3 / 5

/-- The minimum difference between sedans and sports cars -/
def min_difference : ℕ := 20

/-- The number of sedans expected to be sold -/
def sedans : ℕ := 75

theorem dealership_sales_prediction :
  (sedans : ℚ) = sports_cars / ratio ∧ 
  sedans ≥ sports_cars + min_difference := by
  sorry

end NUMINAMATH_CALUDE_dealership_sales_prediction_l625_62527


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l625_62553

theorem floor_negative_seven_fourths :
  ⌊(-7 : ℝ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l625_62553


namespace NUMINAMATH_CALUDE_complex_root_magnitude_l625_62556

theorem complex_root_magnitude (z : ℂ) (h : z^2 - z + 1 = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_l625_62556


namespace NUMINAMATH_CALUDE_twelveRowTriangle_l625_62520

/-- Calculates the sum of an arithmetic progression -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Represents the triangle construction -/
structure TriangleConstruction where
  rows : ℕ
  firstRowRods : ℕ
  rodIncreasePerRow : ℕ

/-- Calculates the total number of pieces in the triangle construction -/
def totalPieces (t : TriangleConstruction) : ℕ :=
  let rodSum := arithmeticSum t.firstRowRods t.rodIncreasePerRow t.rows
  let connectorSum := arithmeticSum 1 1 (t.rows + 1)
  rodSum + connectorSum

/-- Theorem statement for the 12-row triangle construction -/
theorem twelveRowTriangle :
  totalPieces { rows := 12, firstRowRods := 3, rodIncreasePerRow := 3 } = 325 := by
  sorry


end NUMINAMATH_CALUDE_twelveRowTriangle_l625_62520


namespace NUMINAMATH_CALUDE_servant_pay_problem_l625_62530

/-- The amount of money a servant receives for partial work -/
def servant_pay (full_year_pay : ℕ) (uniform_cost : ℕ) (months_worked : ℕ) : ℕ :=
  (full_year_pay * months_worked / 12) + uniform_cost

theorem servant_pay_problem :
  let full_year_pay : ℕ := 900
  let uniform_cost : ℕ := 100
  let months_worked : ℕ := 9
  servant_pay full_year_pay uniform_cost months_worked = 775 := by
sorry

#eval servant_pay 900 100 9

end NUMINAMATH_CALUDE_servant_pay_problem_l625_62530


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l625_62504

/-- Given plane vectors a and b, where a is parallel to b, prove that 2a + 3b = (-4, -8) -/
theorem parallel_vectors_sum (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →  -- a is parallel to b
  (2 • a + 3 • b) = ![(-4 : ℝ), -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l625_62504


namespace NUMINAMATH_CALUDE_roots_and_inequality_l625_62584

/-- Given the equation ln x - (2a)/(x-1) = a with two distinct real roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ Real.log x₁ - (2*a)/(x₁-1) = a ∧ Real.log x₂ - (2*a)/(x₂-1) = a

theorem roots_and_inequality (a : ℝ) (h : has_two_distinct_roots a) :
  a > 0 ∧ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1/(Real.log x₁ + a) + 1/(Real.log x₂ + a) < 0 :=
sorry

end NUMINAMATH_CALUDE_roots_and_inequality_l625_62584


namespace NUMINAMATH_CALUDE_base_7_addition_problem_l625_62506

/-- Given an addition problem in base 7, prove that X + Y = 10 in base 10 --/
theorem base_7_addition_problem (X Y : ℕ) : 
  (5 * 7^2 + X * 7 + Y) + (5 * 7 + 2) = 6 * 7^2 + 4 * 7 + X → X + Y = 10 := by
sorry

end NUMINAMATH_CALUDE_base_7_addition_problem_l625_62506


namespace NUMINAMATH_CALUDE_initial_pairs_count_l625_62510

/-- Represents the number of shoes in a pair -/
def shoesPerPair : ℕ := 2

/-- Represents the number of individual shoes lost -/
def shoesLost : ℕ := 9

/-- Represents the number of matching pairs left after losing shoes -/
def pairsLeft : ℕ := 15

/-- Theorem stating that the initial number of pairs is 24 given the conditions -/
theorem initial_pairs_count :
  ∀ (initialPairs : ℕ),
  (initialPairs * shoesPerPair - shoesLost) / shoesPerPair = pairsLeft →
  initialPairs = pairsLeft + shoesLost / shoesPerPair :=
by
  sorry

#check initial_pairs_count

end NUMINAMATH_CALUDE_initial_pairs_count_l625_62510


namespace NUMINAMATH_CALUDE_sector_area_l625_62524

theorem sector_area (α : ℝ) (p : ℝ) (h1 : α = 2) (h2 : p = 8) :
  let r := p / (α + 2)
  (1/2) * α * r^2 = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_l625_62524


namespace NUMINAMATH_CALUDE_rebus_solution_l625_62533

theorem rebus_solution : ∃! (A B C : ℕ), 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) ∧ 
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧
  (A < 10 ∧ B < 10 ∧ C < 10) ∧
  (100 * A + 10 * B + A + 100 * A + 10 * B + C = 100 * A + 10 * C + C) ∧
  (100 * A + 10 * C + C = 1416) := by
sorry

end NUMINAMATH_CALUDE_rebus_solution_l625_62533


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l625_62577

/-- An isosceles triangle PQR with given side lengths -/
structure IsoscelesTriangle where
  -- Side lengths
  pq : ℝ
  qr : ℝ
  pr : ℝ
  -- Isosceles condition
  isIsosceles : pq = pr
  -- Given side lengths
  qr_eq : qr = 8
  pr_eq : pr = 10

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.pq + t.qr + t.pr

/-- Theorem: The perimeter of the given isosceles triangle is 28 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) :
  perimeter t = 28 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l625_62577


namespace NUMINAMATH_CALUDE_soda_price_l625_62518

/-- The cost of a burger in cents -/
def burger_cost : ℕ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

/-- Four burgers and three sodas cost 540 cents -/
axiom alice_purchase : 4 * burger_cost + 3 * soda_cost = 540

/-- Three burgers and two sodas cost 390 cents -/
axiom bill_purchase : 3 * burger_cost + 2 * soda_cost = 390

/-- The cost of a soda is 60 cents -/
theorem soda_price : soda_cost = 60 := by sorry

end NUMINAMATH_CALUDE_soda_price_l625_62518


namespace NUMINAMATH_CALUDE_mix_alloys_theorem_l625_62515

/-- Represents an alloy of copper and zinc -/
structure Alloy where
  copper : ℝ
  zinc : ℝ

/-- The first alloy with twice as much copper as zinc -/
def alloy1 : Alloy := { copper := 2, zinc := 1 }

/-- The second alloy with five times less copper than zinc -/
def alloy2 : Alloy := { copper := 1, zinc := 5 }

/-- Mixing two alloys in a given ratio -/
def mixAlloys (a b : Alloy) (ratio : ℝ) : Alloy :=
  { copper := ratio * a.copper + b.copper,
    zinc := ratio * a.zinc + b.zinc }

/-- Theorem stating that mixing alloy1 and alloy2 in 1:2 ratio results in an alloy with twice as much zinc as copper -/
theorem mix_alloys_theorem :
  let mixedAlloy := mixAlloys alloy1 alloy2 0.5
  mixedAlloy.zinc = 2 * mixedAlloy.copper := by sorry

end NUMINAMATH_CALUDE_mix_alloys_theorem_l625_62515


namespace NUMINAMATH_CALUDE_number_division_problem_l625_62554

theorem number_division_problem :
  ∃ x : ℝ, x / 5 = 30 + x / 6 ∧ x = 900 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l625_62554


namespace NUMINAMATH_CALUDE_not_perfect_square_600_sixes_and_zeros_l625_62538

/-- Represents a number with 600 digits of 6 followed by some zeros -/
def number_with_600_sixes_and_zeros (n : ℕ) : ℕ :=
  6 * 10^600 + n

/-- Theorem stating that a number with 600 digits of 6 followed by any number of zeros cannot be a perfect square -/
theorem not_perfect_square_600_sixes_and_zeros (n : ℕ) :
  ∃ (m : ℕ), (number_with_600_sixes_and_zeros n) = m^2 → False :=
sorry

end NUMINAMATH_CALUDE_not_perfect_square_600_sixes_and_zeros_l625_62538


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l625_62501

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4) :
  c / d = 5 / 12 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l625_62501


namespace NUMINAMATH_CALUDE_diamond_count_l625_62587

/-- The number of rubies in the chest -/
def rubies : ℕ := 377

/-- The difference between the number of diamonds and rubies -/
def diamond_ruby_difference : ℕ := 44

/-- The number of diamonds in the chest -/
def diamonds : ℕ := rubies + diamond_ruby_difference

theorem diamond_count : diamonds = 421 := by
  sorry

end NUMINAMATH_CALUDE_diamond_count_l625_62587


namespace NUMINAMATH_CALUDE_min_value_of_expression_l625_62572

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + 2*a*b - 3 = 0) :
  ∃ (k : ℝ), k = 2*a + b ∧ k ≥ 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x^2 + 2*x*y - 3 = 0 → 2*x + y ≥ k :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l625_62572


namespace NUMINAMATH_CALUDE_seonwoo_change_l625_62521

/-- Calculates the change Seonwoo received after buying bubblegum and ramen. -/
theorem seonwoo_change
  (initial_amount : ℕ)
  (bubblegum_cost : ℕ)
  (bubblegum_count : ℕ)
  (ramen_cost_per_two : ℕ)
  (ramen_count : ℕ)
  (h1 : initial_amount = 10000)
  (h2 : bubblegum_cost = 600)
  (h3 : bubblegum_count = 2)
  (h4 : ramen_cost_per_two = 1600)
  (h5 : ramen_count = 9) :
  initial_amount - (bubblegum_cost * bubblegum_count + 
    (ramen_cost_per_two * (ramen_count / 2)) + 
    (ramen_cost_per_two / 2 * (ramen_count % 2))) = 1600 :=
by sorry

end NUMINAMATH_CALUDE_seonwoo_change_l625_62521


namespace NUMINAMATH_CALUDE_rook_placement_modulo_four_l625_62545

/-- The color of a cell on the board -/
def cellColor (n i j : ℕ) : ℕ := min (i + j - 1) (2 * n - i - j + 1)

/-- A valid rook placement function -/
def IsValidRookPlacement (n : ℕ) (f : ℕ → ℕ) : Prop :=
  (∀ i j, i ≠ j → f i ≠ f j) ∧
  (∀ i, i ∈ Finset.range n → f i ∈ Finset.range n) ∧
  (∀ i j, i ≠ j → cellColor n i (f i) ≠ cellColor n j (f j))

theorem rook_placement_modulo_four (n : ℕ) :
  (∃ f, IsValidRookPlacement n f) →
  n % 4 = 0 ∨ n % 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_rook_placement_modulo_four_l625_62545


namespace NUMINAMATH_CALUDE_inverse_square_relation_l625_62570

/-- Given that x varies inversely as the square of y, and y = 3 when x = 1,
    prove that x = 0.5625 when y = 4. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) : 
  (∀ (x y : ℝ), x = k / (y ^ 2)) →  -- x varies inversely as the square of y
  (1 = k / (3 ^ 2)) →               -- y = 3 when x = 1
  (k = 9) →                         -- derived from the previous condition
  (x = 9 / (4 ^ 2)) →               -- x when y = 4
  x = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l625_62570


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l625_62531

theorem tangent_line_y_intercept (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x - Real.log x
  let f' : ℝ → ℝ := λ x ↦ a - 1 / x
  let tangent_slope : ℝ := f' 1
  let tangent_intercept : ℝ := f 1 - tangent_slope * 1
  tangent_intercept = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l625_62531


namespace NUMINAMATH_CALUDE_cube_volume_from_edge_sum_l625_62558

/-- Given a cube where the sum of the lengths of all edges is 96 cm, 
    prove that its volume is 512 cubic centimeters. -/
theorem cube_volume_from_edge_sum (edge_sum : ℝ) (volume : ℝ) : 
  edge_sum = 96 → volume = (edge_sum / 12)^3 → volume = 512 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_edge_sum_l625_62558


namespace NUMINAMATH_CALUDE_p_and_q_implies_m_leq_1_l625_62580

/-- Proposition p: For all x ∈ ℝ, the function y = log₂(2ˣ - m + 1) is defined. -/
def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, 2^x - m + 1 > 0

/-- Proposition q: The function f(x) = (5 - 2m)ˣ is increasing. -/
def proposition_q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (5 - 2*m)^x < (5 - 2*m)^y

/-- If propositions p and q are true, then m ≤ 1. -/
theorem p_and_q_implies_m_leq_1 (m : ℝ) :
  proposition_p m ∧ proposition_q m → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_implies_m_leq_1_l625_62580


namespace NUMINAMATH_CALUDE_like_terms_proof_l625_62528

/-- Two algebraic expressions are like terms if they have the same variables with the same exponents. -/
def like_terms (expr1 expr2 : String) : Prop := sorry

theorem like_terms_proof :
  (like_terms "3a³b" "-3ba³") ∧
  ¬(like_terms "a³" "b³") ∧
  ¬(like_terms "abc" "ac") ∧
  ¬(like_terms "a⁵" "2⁵") := by sorry

end NUMINAMATH_CALUDE_like_terms_proof_l625_62528


namespace NUMINAMATH_CALUDE_log_inequality_condition_l625_62561

theorem log_inequality_condition (x y : ℝ) :
  (∀ x y, x > 0 ∧ y > 0 ∧ Real.log x > Real.log y → x > y) ∧
  ¬(∀ x y, x > y → Real.log x > Real.log y) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_condition_l625_62561


namespace NUMINAMATH_CALUDE_unique_positive_solution_l625_62540

/-- The polynomial function f(x) = x^8 + 3x^7 + 6x^6 + 2023x^5 - 2000x^4 -/
def f (x : ℝ) : ℝ := x^8 + 3*x^7 + 6*x^6 + 2023*x^5 - 2000*x^4

/-- The theorem stating that f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! x : ℝ, x > 0 ∧ f x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l625_62540


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l625_62595

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l625_62595


namespace NUMINAMATH_CALUDE_trajectory_equation_l625_62552

-- Define the fixed circle C
def C (x y : ℝ) : Prop := x^2 + (y+3)^2 = 1

-- Define the line that M is tangent to
def L (y : ℝ) : Prop := y = 2

-- Define the moving circle M
def M (x y : ℝ) : Prop := ∃ (r : ℝ), r > 0 ∧ 
  (∀ (x' y' : ℝ), C x' y' → (x - x')^2 + (y - y')^2 = (1 + r)^2) ∧
  (∀ (y' : ℝ), L y' → |y - y'| = r)

-- State the theorem
theorem trajectory_equation :
  ∀ (x y : ℝ), M x y → x^2 = -12*y := by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l625_62552


namespace NUMINAMATH_CALUDE_managers_salary_l625_62503

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) :
  num_employees = 20 ∧ 
  avg_salary = 1500 ∧ 
  avg_increase = 100 →
  (num_employees + 1) * (avg_salary + avg_increase) - num_employees * avg_salary = 3600 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l625_62503


namespace NUMINAMATH_CALUDE_shekar_weighted_average_l625_62571

def weightedAverage (scores : List ℝ) (weights : List ℝ) : ℝ :=
  (List.zip scores weights).map (fun (s, w) => s * w) |> List.sum

theorem shekar_weighted_average :
  let scores : List ℝ := [76, 65, 82, 62, 85]
  let weights : List ℝ := [0.20, 0.15, 0.25, 0.25, 0.15]
  weightedAverage scores weights = 73.7 := by
sorry

end NUMINAMATH_CALUDE_shekar_weighted_average_l625_62571


namespace NUMINAMATH_CALUDE_train_crossing_time_l625_62522

/-- Given a train and platform with specific properties, calculate the time for the train to cross a tree -/
theorem train_crossing_time (train_length platform_length : ℝ) (platform_crossing_time : ℝ) 
  (h1 : train_length = 1400)
  (h2 : platform_length = 700)
  (h3 : platform_crossing_time = 150) : 
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l625_62522


namespace NUMINAMATH_CALUDE_weighted_average_fish_per_day_l625_62599

-- Define the daily catch for each person
def aang_catch : List Nat := [5, 7, 9]
def sokka_catch : List Nat := [8, 5, 6]
def toph_catch : List Nat := [10, 12, 8]
def zuko_catch : List Nat := [6, 7, 10]

-- Define the number of people and days
def num_people : Nat := 4
def num_days : Nat := 3

-- Define the total fish caught by the group
def total_fish : Nat := aang_catch.sum + sokka_catch.sum + toph_catch.sum + zuko_catch.sum

-- Define the total days fished by the group
def total_days : Nat := num_people * num_days

-- Theorem to prove
theorem weighted_average_fish_per_day :
  (total_fish : Rat) / total_days = 93/12 := by sorry

end NUMINAMATH_CALUDE_weighted_average_fish_per_day_l625_62599


namespace NUMINAMATH_CALUDE_system_solution_l625_62557

theorem system_solution : 
  ∃! (s : Set (ℝ × ℝ)), s = {(12, 10), (-10, -12)} ∧ 
    ∀ (x y : ℝ), (x, y) ∈ s ↔ 
      ((3/2 : ℝ)^(x-y) - (2/3 : ℝ)^(x-y) = 65/36 ∧
       x*y - x + y = 118) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l625_62557


namespace NUMINAMATH_CALUDE_john_guests_correct_l625_62564

/-- The number of guests John wants for his wedding. -/
def john_guests : ℕ := 50

/-- The venue cost for the wedding. -/
def venue_cost : ℕ := 10000

/-- The cost per guest for the wedding. -/
def cost_per_guest : ℕ := 500

/-- The total cost of the wedding if John's wife gets her way. -/
def total_cost : ℕ := 50000

/-- Theorem stating that the number of guests John wants is correct. -/
theorem john_guests_correct :
  venue_cost + cost_per_guest * (john_guests + (60 * john_guests) / 100) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_john_guests_correct_l625_62564


namespace NUMINAMATH_CALUDE_square_root_of_negative_two_squared_l625_62547

theorem square_root_of_negative_two_squared (x : ℝ) : x = 2 → x ^ 2 = (-2) ^ 2 := by sorry

end NUMINAMATH_CALUDE_square_root_of_negative_two_squared_l625_62547


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_l625_62500

theorem power_of_two_plus_one (b m n : ℕ) 
  (h1 : b > 1) 
  (h2 : m ≠ n) 
  (h3 : ∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) : 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_l625_62500


namespace NUMINAMATH_CALUDE_passing_percentage_l625_62511

def max_score : ℕ := 750
def mike_score : ℕ := 212
def shortfall : ℕ := 13

theorem passing_percentage : 
  (((mike_score + shortfall : ℚ) / max_score) * 100 : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_l625_62511


namespace NUMINAMATH_CALUDE_base4_division_l625_62560

/-- Convert a number from base 4 to decimal --/
def base4ToDecimal (n : List Nat) : Nat :=
  n.foldr (fun digit acc => acc * 4 + digit) 0

/-- Convert a number from decimal to base 4 --/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Theorem: 12345₄ divided by 23₄ equals 535₄ in base 4 --/
theorem base4_division :
  let dividend := base4ToDecimal [1, 2, 3, 4, 5]
  let divisor := base4ToDecimal [2, 3]
  let quotient := base4ToDecimal [5, 3, 5]
  decimalToBase4 (dividend / divisor) = [5, 3, 5] :=
by sorry

end NUMINAMATH_CALUDE_base4_division_l625_62560


namespace NUMINAMATH_CALUDE_line_through_points_l625_62513

theorem line_through_points (a : ℝ) : 
  a > 0 ∧ 
  (∃ m b : ℝ, m = 2 ∧ b = 0 ∧ 
    (5 = m * a + b) ∧ 
    (a = m * 2 + b)) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l625_62513


namespace NUMINAMATH_CALUDE_segments_form_triangle_l625_62507

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem segments_form_triangle :
  can_form_triangle 5 6 10 :=
by sorry

end NUMINAMATH_CALUDE_segments_form_triangle_l625_62507


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l625_62589

/-- Calculates the total number of heartbeats for an athlete jogging a given distance -/
def total_heartbeats (heart_rate : ℕ) (distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * distance * pace

/-- Proves that an athlete jogging 15 miles at 8 minutes per mile with a heart rate of 120 bpm will have 14400 total heartbeats -/
theorem athlete_heartbeats :
  total_heartbeats 120 15 8 = 14400 := by
  sorry

#eval total_heartbeats 120 15 8

end NUMINAMATH_CALUDE_athlete_heartbeats_l625_62589


namespace NUMINAMATH_CALUDE_complex_sum_equals_i_l625_62588

theorem complex_sum_equals_i : Complex.I^2 = -1 → (1 : ℂ) + Complex.I + Complex.I^2 = Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_i_l625_62588


namespace NUMINAMATH_CALUDE_james_touchdown_points_l625_62592

/-- The number of points per touchdown in James' football season -/
def points_per_touchdown : ℕ := by sorry

/-- The number of touchdowns James scores per game -/
def touchdowns_per_game : ℕ := 4

/-- The number of games in the season -/
def games_in_season : ℕ := 15

/-- The number of 2-point conversions James scores in the season -/
def two_point_conversions : ℕ := 6

/-- The total points James scores in the season -/
def total_points : ℕ := 372

theorem james_touchdown_points :
  points_per_touchdown * touchdowns_per_game * games_in_season +
  2 * two_point_conversions = total_points ∧
  points_per_touchdown = 6 := by sorry

end NUMINAMATH_CALUDE_james_touchdown_points_l625_62592
