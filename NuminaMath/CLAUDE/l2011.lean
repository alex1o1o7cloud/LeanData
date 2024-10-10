import Mathlib

namespace sin_2phi_value_l2011_201184

theorem sin_2phi_value (φ : ℝ) 
  (h : ∫ x in (0)..(Real.pi / 2), Real.sin (x - φ) = Real.sqrt 7 / 4) : 
  Real.sin (2 * φ) = 9 / 16 := by
  sorry

end sin_2phi_value_l2011_201184


namespace amy_video_files_l2011_201121

/-- Represents the number of video files Amy had initially -/
def initial_video_files : ℕ := 36

theorem amy_video_files :
  let initial_music_files : ℕ := 26
  let deleted_files : ℕ := 48
  let remaining_files : ℕ := 14
  initial_video_files + initial_music_files - deleted_files = remaining_files :=
by sorry

end amy_video_files_l2011_201121


namespace cos_a_plus_beta_half_l2011_201155

theorem cos_a_plus_beta_half (a β : ℝ) 
  (h1 : 0 < a ∧ a < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (a + π / 4) = 1 / 3)
  (h4 : Real.sin (π / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (a + β / 2) = Real.sqrt 6 / 3 := by
  sorry

end cos_a_plus_beta_half_l2011_201155


namespace parabola_vertex_l2011_201142

/-- A quadratic function f(x) = -x^2 + ax + b where f(x) ≤ 0 
    has the solution (-∞,-3] ∪ [5,∞) -/
def f (a b x : ℝ) : ℝ := -x^2 + a*x + b

/-- The solution set of f(x) ≤ 0 -/
def solution_set (a b : ℝ) : Set ℝ :=
  {x | x ≤ -3 ∨ x ≥ 5}

/-- The vertex of the parabola -/
def vertex (a b : ℝ) : ℝ × ℝ := (1, 16)

theorem parabola_vertex (a b : ℝ) 
  (h : ∀ x, f a b x ≤ 0 ↔ x ∈ solution_set a b) :
  vertex a b = (1, 16) :=
sorry

end parabola_vertex_l2011_201142


namespace hibiscus_flowers_solution_l2011_201100

/-- The number of flowers on Mario's first hibiscus plant -/
def first_plant_flowers : ℕ := 2

/-- The number of flowers on Mario's second hibiscus plant -/
def second_plant_flowers : ℕ := 2 * first_plant_flowers

/-- The number of flowers on Mario's third hibiscus plant -/
def third_plant_flowers : ℕ := 4 * second_plant_flowers

/-- The total number of flowers on all three plants -/
def total_flowers : ℕ := 22

theorem hibiscus_flowers_solution :
  first_plant_flowers + second_plant_flowers + third_plant_flowers = total_flowers ∧
  first_plant_flowers = 2 := by
  sorry

end hibiscus_flowers_solution_l2011_201100


namespace age_problem_l2011_201169

/-- Represents the ages of Sandy, Molly, and Kim -/
structure Ages where
  sandy : ℝ
  molly : ℝ
  kim : ℝ

/-- The problem statement -/
theorem age_problem (current : Ages) (future : Ages) : 
  -- Current ratio condition
  (current.sandy / current.molly = 4 / 3) ∧
  (current.sandy / current.kim = 4 / 5) ∧
  -- Future age condition
  (future.sandy = current.sandy + 8) ∧
  (future.molly = current.molly + 8) ∧
  (future.kim = current.kim + 8) ∧
  -- Future Sandy's age
  (future.sandy = 74) ∧
  -- Future ratio condition
  (future.sandy / future.molly = 9 / 7) ∧
  (future.sandy / future.kim = 9 / 10) →
  -- Conclusion
  current.molly = 49.5 ∧ current.kim = 82.5 :=
by sorry

end age_problem_l2011_201169


namespace num_installments_is_40_l2011_201160

/-- Proves that the number of installments is 40 given the payment conditions --/
theorem num_installments_is_40 
  (n : ℕ) -- Total number of installments
  (h1 : n ≥ 20) -- At least 20 installments
  (first_20_payment : ℕ := 410) -- First 20 payments
  (remaining_payment : ℕ := 475) -- Remaining payments
  (average_payment : ℚ := 442.5) -- Average payment
  (h2 : (20 * first_20_payment + (n - 20) * remaining_payment : ℚ) / n = average_payment) -- Average payment equation
  : n = 40 := by
  sorry

end num_installments_is_40_l2011_201160


namespace max_surface_area_parallelepiped_l2011_201183

/-- The maximum surface area of a rectangular parallelepiped with diagonal length 3 is 18 -/
theorem max_surface_area_parallelepiped (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 + c^2 = 9 →
  2 * (a * b + b * c + c * a) ≤ 18 :=
by sorry

end max_surface_area_parallelepiped_l2011_201183


namespace larger_number_proof_l2011_201133

/-- Given two positive integers with HCF 23 and LCM factors 16 and 17, prove the larger is 391 -/
theorem larger_number_proof (a b : ℕ+) : 
  Nat.gcd a b = 23 → 
  Nat.lcm a b = 23 * 16 * 17 → 
  max a b = 391 := by sorry

end larger_number_proof_l2011_201133


namespace range_of_a_l2011_201126

/-- The set of real numbers x satisfying the condition p -/
def set_p (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

/-- The set of real numbers x satisfying the condition q -/
def set_q : Set ℝ := {x | x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0}

/-- The theorem stating the range of values for a -/
theorem range_of_a (a : ℝ) : 
  (a < 0) → 
  (set_p a)ᶜ ⊂ (set_q)ᶜ → 
  (set_p a)ᶜ ≠ (set_q)ᶜ → 
  (-4 ≤ a ∧ a < 0) ∨ (a ≤ -4) :=
sorry

end range_of_a_l2011_201126


namespace repeating_digit_equality_l2011_201106

/-- Represents a repeating digit number -/
def repeatingDigit (d : ℕ) (n : ℕ) : ℕ := d * (10^n - 1) / 9

/-- The main theorem -/
theorem repeating_digit_equality (x y z : ℕ) (h : x < 10 ∧ y < 10 ∧ z < 10) :
  (∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧
    (repeatingDigit x (2 * n₁) - repeatingDigit y n₁).sqrt = repeatingDigit z n₁ ∧
    (repeatingDigit x (2 * n₂) - repeatingDigit y n₂).sqrt = repeatingDigit z n₂) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 9 ∧ y = 8 ∧ z = 9)) ∧
  (∀ n : ℕ, (repeatingDigit x (2 * n) - repeatingDigit y n).sqrt = repeatingDigit z n) :=
sorry

end repeating_digit_equality_l2011_201106


namespace speeding_fine_calculation_l2011_201181

/-- Calculates the base fine for speeding given the total amount owed and other fees --/
theorem speeding_fine_calculation 
  (speed_limit : ℕ) 
  (actual_speed : ℕ) 
  (fine_increase_per_mph : ℕ) 
  (court_costs : ℕ) 
  (lawyer_fee_per_hour : ℕ) 
  (lawyer_hours : ℕ) 
  (total_owed : ℕ) : 
  speed_limit = 30 →
  actual_speed = 75 →
  fine_increase_per_mph = 2 →
  court_costs = 300 →
  lawyer_fee_per_hour = 80 →
  lawyer_hours = 3 →
  total_owed = 820 →
  ∃ (base_fine : ℕ),
    base_fine = 190 ∧
    total_owed = base_fine + 
      2 * (actual_speed - speed_limit) * 2 + 
      court_costs + 
      lawyer_fee_per_hour * lawyer_hours :=
by sorry

end speeding_fine_calculation_l2011_201181


namespace football_lineup_count_l2011_201179

/-- The number of ways to choose a starting lineup from a football team. -/
def starting_lineup_count (total_players : ℕ) (offensive_linemen : ℕ) (lineup_size : ℕ) (linemen_in_lineup : ℕ) : ℕ :=
  (Nat.choose offensive_linemen linemen_in_lineup) *
  (Nat.choose (total_players - linemen_in_lineup) (lineup_size - linemen_in_lineup)) *
  (Nat.factorial (lineup_size - linemen_in_lineup))

/-- Theorem stating the number of ways to choose the starting lineup. -/
theorem football_lineup_count :
  starting_lineup_count 15 5 5 2 = 17160 := by
  sorry

end football_lineup_count_l2011_201179


namespace alexa_first_day_pages_l2011_201159

/-- The number of pages Alexa read on the first day of reading a Nancy Drew mystery. -/
def pages_read_first_day (total_pages second_day_pages pages_left : ℕ) : ℕ :=
  total_pages - second_day_pages - pages_left

/-- Theorem stating that Alexa read 18 pages on the first day. -/
theorem alexa_first_day_pages :
  pages_read_first_day 95 58 19 = 18 := by
  sorry

end alexa_first_day_pages_l2011_201159


namespace inverse_functions_l2011_201189

-- Define the type for our functions
def Function : Type := ℝ → ℝ

-- Define the property of having an inverse
def has_inverse (f : Function) : Prop := ∃ g : Function, ∀ x, g (f x) = x ∧ f (g x) = x

-- Define our functions based on their graphical properties
def F : Function := sorry
def G : Function := sorry
def H : Function := sorry
def I : Function := sorry

-- State the theorem
theorem inverse_functions :
  (has_inverse F) ∧ (has_inverse H) ∧ (has_inverse I) ∧ ¬(has_inverse G) := by sorry

end inverse_functions_l2011_201189


namespace F_max_value_l2011_201154

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def f_derivative (x : ℝ) : ℝ := Real.cos x - Real.sin x

noncomputable def F (x : ℝ) : ℝ := f x * f_derivative x + f x ^ 2

theorem F_max_value :
  ∃ (M : ℝ), (∀ (x : ℝ), F x ≤ M) ∧ (∃ (x : ℝ), F x = M) ∧ M = 1 + Real.sqrt 2 := by
  sorry

end F_max_value_l2011_201154


namespace smallest_ending_number_l2011_201156

/-- A function that returns the number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- A function that checks if a number has an even number of factors -/
def has_even_factors (n : ℕ+) : Prop :=
  Even (num_factors n)

/-- A function that counts the number of even integers with an even number of factors
    in the range from 1 to n (inclusive) -/
def count_even_with_even_factors (n : ℕ) : ℕ := sorry

theorem smallest_ending_number :
  ∀ k : ℕ, k < 14 → count_even_with_even_factors k < 5 ∧
  count_even_with_even_factors 14 ≥ 5 := by sorry

end smallest_ending_number_l2011_201156


namespace parabola_shift_theorem_l2011_201118

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 1 ∧ p.b = -2 ∧ p.c = -5 →
  let p_shifted := shift_parabola p 2 3
  p_shifted.a = 1 ∧ p_shifted.b = 2 ∧ p_shifted.c = -3 := by
  sorry

#check parabola_shift_theorem

end parabola_shift_theorem_l2011_201118


namespace inequality_proof_l2011_201148

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1 / x + 1 / y ≤ 1 / x^2 + 1 / y^2 := by
  sorry

end inequality_proof_l2011_201148


namespace prob_a_before_b_is_one_third_l2011_201125

-- Define the set of people
inductive Person : Type
  | A
  | B
  | C

-- Define a duty arrangement as a list of people
def DutyArrangement := List Person

-- Define the set of all possible duty arrangements
def allArrangements : List DutyArrangement :=
  [[Person.A, Person.B, Person.C],
   [Person.A, Person.C, Person.B],
   [Person.C, Person.A, Person.B]]

-- Define a function to check if A is immediately before B in an arrangement
def isABeforeB (arrangement : DutyArrangement) : Bool :=
  match arrangement with
  | [Person.A, Person.B, _] => true
  | _ => false

-- Define the probability of A being immediately before B
def probABeforeB : ℚ :=
  (allArrangements.filter isABeforeB).length / allArrangements.length

-- Theorem statement
theorem prob_a_before_b_is_one_third :
  probABeforeB = 1 / 3 := by sorry

end prob_a_before_b_is_one_third_l2011_201125


namespace perpendicular_vectors_sum_l2011_201105

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (2, -m)

theorem perpendicular_vectors_sum (m : ℝ) 
  (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  (3 * a.1 + 2 * (b m).1, 3 * a.2 + 2 * (b m).2) = (7, 4) := by
  sorry

end perpendicular_vectors_sum_l2011_201105


namespace not_p_or_q_false_implies_p_or_q_l2011_201136

theorem not_p_or_q_false_implies_p_or_q (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
sorry

end not_p_or_q_false_implies_p_or_q_l2011_201136


namespace quadratic_inequality_solution_set_l2011_201167

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 2*x - 3 < 0 ↔ -1 < x ∧ x < 3 :=
by sorry

end quadratic_inequality_solution_set_l2011_201167


namespace max_base_eight_digit_sum_l2011_201140

def base_eight_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem max_base_eight_digit_sum (n : ℕ) (h : n < 1728) :
  (base_eight_digits n).sum ≤ 6 :=
sorry

end max_base_eight_digit_sum_l2011_201140


namespace function_value_at_2012_l2011_201176

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β) where f(2001) = 3, 
    prove that f(2012) = -3 -/
theorem function_value_at_2012 
  (a b α β : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β))
  (h2 : f 2001 = 3) :
  f 2012 = -3 := by
sorry

end function_value_at_2012_l2011_201176


namespace games_before_third_l2011_201128

theorem games_before_third (average_score : ℝ) (third_game_score : ℝ) (points_needed : ℝ) :
  average_score = 61.5 →
  third_game_score = 47 →
  points_needed = 330 →
  (∃ n : ℕ, n * average_score + third_game_score + points_needed = 500 ∧ n = 2) :=
by sorry

end games_before_third_l2011_201128


namespace evenOnesTableCountTheorem_l2011_201198

/-- The number of ways to fill an m × n table with zeros and ones,
    such that there is an even number of ones in every row and every column -/
def evenOnesTableCount (m n : ℕ) : ℕ :=
  2^((m-1)*(n-1))

/-- Theorem: The number of ways to fill an m × n table with zeros and ones,
    such that there is an even number of ones in every row and every column,
    is equal to 2^((m-1)(n-1)) -/
theorem evenOnesTableCountTheorem (m n : ℕ) :
  evenOnesTableCount m n = 2^((m-1)*(n-1)) := by
  sorry


end evenOnesTableCountTheorem_l2011_201198


namespace arithmetic_sequence_sixth_term_l2011_201158

theorem arithmetic_sequence_sixth_term 
  (a : ℕ → ℚ)  -- a is a sequence of rational numbers
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence condition
  (h_first : a 1 = 3/8)  -- first term is 3/8
  (h_eleventh : a 11 = 5/6)  -- eleventh term is 5/6
  : a 6 = 29/48 := by
  sorry

end arithmetic_sequence_sixth_term_l2011_201158


namespace circumradii_ratio_eq_side_ratio_l2011_201152

/-- Represents the properties of an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  perimeter : ℝ
  area : ℝ
  circumradius : ℝ
  perimeter_eq : perimeter = 3 * side
  area_eq : area = (side^2 * Real.sqrt 3) / 4
  circumradius_eq : circumradius = (side * Real.sqrt 3) / 3

/-- Theorem stating the relationship between circumradii of two equilateral triangles -/
theorem circumradii_ratio_eq_side_ratio 
  (n m : ℝ) 
  (fore back : EquilateralTriangle) 
  (h_perimeter_ratio : fore.perimeter / back.perimeter = n / m)
  (h_area_ratio : fore.area / back.area = n / m) :
  fore.circumradius / back.circumradius = fore.side / back.side := by
  sorry

#check circumradii_ratio_eq_side_ratio

end circumradii_ratio_eq_side_ratio_l2011_201152


namespace intersection_when_a_is_one_subset_condition_l2011_201166

-- Define the sets P and Q
def P : Set ℝ := {x | 2*x^2 - 3*x + 1 ≤ 0}
def Q (a : ℝ) : Set ℝ := {x | (x-a)*(x-a-1) ≤ 0}

-- Theorem 1: P ∩ Q = {1} when a = 1
theorem intersection_when_a_is_one : P ∩ (Q 1) = {1} := by sorry

-- Theorem 2: P ⊆ Q if and only if 0 ≤ a ≤ 1/2
theorem subset_condition (a : ℝ) : P ⊆ Q a ↔ 0 ≤ a ∧ a ≤ 1/2 := by sorry

end intersection_when_a_is_one_subset_condition_l2011_201166


namespace inequality_solution_l2011_201170

theorem inequality_solution (x : ℝ) : (x - 2) / (x - 5) ≥ 3 ↔ x ∈ Set.Ioo 5 (13/2) ∪ {13/2} := by
  sorry

end inequality_solution_l2011_201170


namespace bank_account_transfer_l2011_201151

/-- Represents a bank account transfer operation that doubles the amount in one account. -/
inductive Transfer : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → Prop
| double12 : ∀ a b c, Transfer (a, b, c) (a + b, 0, c)
| double13 : ∀ a b c, Transfer (a, b, c) (a + c, b, 0)
| double21 : ∀ a b c, Transfer (a, b, c) (0, a + b, c)
| double23 : ∀ a b c, Transfer (a, b, c) (a, b + c, 0)
| double31 : ∀ a b c, Transfer (a, b, c) (0, b, a + c)
| double32 : ∀ a b c, Transfer (a, b, c) (a, 0, b + c)

/-- Represents a sequence of transfers. -/
def TransferSeq : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → Prop :=
  Relation.ReflTransGen Transfer

theorem bank_account_transfer :
  (∀ a b c : ℕ, ∃ a' b' c', TransferSeq (a, b, c) (a', b', c') ∧ (a' = 0 ∨ b' = 0 ∨ c' = 0)) ∧
  (∃ a b c : ℕ, ∀ a' b' c', TransferSeq (a, b, c) (a', b', c') → ¬(a' = 0 ∧ b' = 0) ∧ ¬(a' = 0 ∧ c' = 0) ∧ ¬(b' = 0 ∧ c' = 0)) :=
by sorry

end bank_account_transfer_l2011_201151


namespace project_hours_total_l2011_201161

/-- Given the conditions of the project hours charged by Pat, Kate, and Mark, 
    prove that the total number of hours charged is 144. -/
theorem project_hours_total (k p m : ℕ) : 
  p = 2 * k →          -- Pat charged twice as much as Kate
  3 * p = m →          -- Pat charged 1/3 as much as Mark
  m = k + 80 →         -- Mark charged 80 more hours than Kate
  k + p + m = 144 :=   -- Total hours charged
by sorry

end project_hours_total_l2011_201161


namespace intersection_of_A_and_B_l2011_201197

def A : Set ℝ := {x | 3 * x + 2 > 0}
def B : Set ℝ := {x | (x + 1) * (x - 3) > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | x > 3} := by sorry

end intersection_of_A_and_B_l2011_201197


namespace grant_coverage_percentage_l2011_201112

def total_cost : ℝ := 30000
def savings : ℝ := 10000
def loan_amount : ℝ := 12000

theorem grant_coverage_percentage : 
  let remainder := total_cost - savings
  let grant_amount := remainder - loan_amount
  (grant_amount / remainder) * 100 = 40 := by
  sorry

end grant_coverage_percentage_l2011_201112


namespace complement_of_M_l2011_201101

-- Define the universal set U
def U : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define the set M
def M : Set ℝ := {x | x^2 - x ≤ 0}

-- Theorem statement
theorem complement_of_M (x : ℝ) : 
  x ∈ (U \ M) ↔ (1 < x ∧ x ≤ 2) :=
sorry

end complement_of_M_l2011_201101


namespace average_people_moving_per_hour_l2011_201144

/-- The number of people moving to Florida -/
def people_moving : ℕ := 3000

/-- The number of days -/
def days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculate the average number of people moving per hour -/
def average_per_hour : ℚ :=
  people_moving / (days * hours_per_day)

/-- Round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem average_people_moving_per_hour :
  round_to_nearest average_per_hour = 25 := by
  sorry

end average_people_moving_per_hour_l2011_201144


namespace isosceles_triangle_perimeter_l2011_201178

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 4 ∨ a = 5) (h2 : b = 4 ∨ b = 5) (h3 : a ≠ b) :
  ∃ (p : ℝ), (p = 13 ∨ p = 14) ∧ (p = a + 2*b ∨ p = b + 2*a) :=
sorry

end isosceles_triangle_perimeter_l2011_201178


namespace larger_number_problem_l2011_201187

theorem larger_number_problem (x y : ℝ) (h_sum : x + y = 30) (h_diff : x - y = 14) :
  max x y = 22 := by
  sorry

end larger_number_problem_l2011_201187


namespace development_inheritance_relationship_false_l2011_201199

/-- Development is a prerequisite for inheritance -/
def development_prerequisite_for_inheritance : Prop := sorry

/-- Inheritance is a requirement for development -/
def inheritance_requirement_for_development : Prop := sorry

/-- The statement that development is a prerequisite for inheritance
    and inheritance is a requirement for development is false -/
theorem development_inheritance_relationship_false :
  ¬(development_prerequisite_for_inheritance ∧ inheritance_requirement_for_development) :=
sorry

end development_inheritance_relationship_false_l2011_201199


namespace petyas_friends_l2011_201180

theorem petyas_friends (total_stickers : ℕ) : 
  (∃ (x : ℕ), 5 * x + 8 = total_stickers ∧ 6 * x = total_stickers + 11) → 
  (∃ (x : ℕ), x = 19 ∧ 5 * x + 8 = total_stickers ∧ 6 * x = total_stickers + 11) :=
by sorry

end petyas_friends_l2011_201180


namespace project_scores_mode_l2011_201162

def project_scores : List ℕ := [7, 10, 9, 8, 7, 9, 9, 8]

def mode (l : List ℕ) : ℕ := 
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem project_scores_mode :
  mode project_scores = 9 := by sorry

end project_scores_mode_l2011_201162


namespace parabola_through_point_standard_form_l2011_201116

/-- A parabola is defined by its equation and the point it passes through. -/
structure Parabola where
  /-- The point that the parabola passes through -/
  point : ℝ × ℝ
  /-- The equation of the parabola, represented as a function -/
  equation : (ℝ × ℝ) → Prop

/-- The standard form of a parabola's equation -/
inductive StandardForm
  | VerticalAxis (p : ℝ) : StandardForm  -- y² = -2px
  | HorizontalAxis (p : ℝ) : StandardForm  -- x² = 2py

/-- Theorem: If a parabola passes through the point (-2, 3), then its standard equation
    must be either y² = -9/2x or x² = 4/3y -/
theorem parabola_through_point_standard_form (P : Parabola) 
    (h : P.point = (-2, 3)) :
    (∃ (sf : StandardForm), 
      (sf = StandardForm.VerticalAxis (9/4) ∨ 
       sf = StandardForm.HorizontalAxis (2/3)) ∧
      (∀ (x y : ℝ), P.equation (x, y) ↔ 
        (sf = StandardForm.VerticalAxis (9/4) → y^2 = -9/2 * x) ∧
        (sf = StandardForm.HorizontalAxis (2/3) → x^2 = 4/3 * y))) :=
  sorry

end parabola_through_point_standard_form_l2011_201116


namespace sqrt_19_minus_1_squared_plus_2x_plus_2_l2011_201164

theorem sqrt_19_minus_1_squared_plus_2x_plus_2 :
  let x : ℝ := Real.sqrt 19 - 1
  x^2 + 2*x + 2 = 20 := by
sorry

end sqrt_19_minus_1_squared_plus_2x_plus_2_l2011_201164


namespace reciprocal_problem_l2011_201192

theorem reciprocal_problem (x : ℚ) : 8 * x = 16 → 200 * (1 / x) = 100 := by
  sorry

end reciprocal_problem_l2011_201192


namespace projectile_max_height_l2011_201153

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 161

/-- Theorem stating that the maximum height of the projectile is 161 meters -/
theorem projectile_max_height : 
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end projectile_max_height_l2011_201153


namespace probability_not_above_x_axis_is_half_l2011_201145

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Defines the specific parallelogram ABCD from the problem -/
def ABCD : Parallelogram := {
  A := { x := 3, y := 3 }
  B := { x := -3, y := -3 }
  C := { x := -9, y := -3 }
  D := { x := -3, y := 3 }
}

/-- Probability of a point being not above the x-axis in the parallelogram -/
def probability_not_above_x_axis (p : Parallelogram) : ℚ :=
  1/2

/-- Theorem stating that the probability of a randomly selected point 
    from the region determined by parallelogram ABCD being not above 
    the x-axis is 1/2 -/
theorem probability_not_above_x_axis_is_half : 
  probability_not_above_x_axis ABCD = 1/2 := by
  sorry


end probability_not_above_x_axis_is_half_l2011_201145


namespace integers_less_than_four_abs_l2011_201117

theorem integers_less_than_four_abs : 
  {n : ℤ | |n| < 4} = {-3, -2, -1, 0, 1, 2, 3} := by sorry

end integers_less_than_four_abs_l2011_201117


namespace triangle_theorem_l2011_201175

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sine_rule : a / Real.sin A = b / Real.sin B
  angle_sum : A + B + C = π

/-- The theorem to be proved -/
theorem triangle_theorem (t : AcuteTriangle) 
  (h1 : 2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b)
  (h2 : t.a = 6)
  (h3 : t.b + t.c = 8) :
  t.A = π/3 ∧ 
  1/2 * t.b * t.c * Real.sin t.A = 7 * Real.sqrt 3 / 3 := by
  sorry


end triangle_theorem_l2011_201175


namespace smallest_positive_period_sin_l2011_201135

/-- The smallest positive period of y = 5 * sin(3x + π/6) is 2π/3 --/
theorem smallest_positive_period_sin (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 5 * Real.sin (3 * x + π / 6)
  ∃ T : ℝ, T > 0 ∧ T = 2 * π / 3 ∧ (∀ t : ℝ, f (x + T) = f x) ∧
  (∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (x + S) = f x) → T ≤ S) :=
by sorry

end smallest_positive_period_sin_l2011_201135


namespace exists_m_with_all_digits_l2011_201119

/-- For any positive integer n, there exists a positive integer m such that
    the decimal representation of m * n contains all digits from 0 to 9. -/
theorem exists_m_with_all_digits (n : ℕ+) : ∃ m : ℕ+, ∀ d : Fin 10, ∃ k : ℕ,
  (m * n : ℕ) / 10^k % 10 = d.val :=
sorry

end exists_m_with_all_digits_l2011_201119


namespace find_set_M_l2011_201194

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}

def complement_M : Finset ℕ := {1, 2, 4}

theorem find_set_M : 
  ∀ M : Finset ℕ, (∀ x : ℕ, x ∈ U → (x ∈ M ↔ x ∉ complement_M)) → 
  M = {3, 5, 6} := by sorry

end find_set_M_l2011_201194


namespace sum_of_150_consecutive_integers_l2011_201191

def sum_of_consecutive_integers (n : ℕ) (count : ℕ) : ℕ :=
  count * (2 * n + count - 1) / 2

theorem sum_of_150_consecutive_integers :
  ∃ (n : ℕ), sum_of_consecutive_integers n 150 = 1725225 ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 3410775) ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 2245600) ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 1257925) ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 4146950) :=
by
  sorry

#check sum_of_150_consecutive_integers

end sum_of_150_consecutive_integers_l2011_201191


namespace square_perimeter_transformation_l2011_201108

-- Define a square type
structure Square where
  perimeter : ℝ

-- Define the transformation function
def transform (s : Square) : Square :=
  { perimeter := 12 * s.perimeter }

-- Theorem statement
theorem square_perimeter_transformation (s : Square) :
  (transform s).perimeter = 12 * s.perimeter := by
  sorry

end square_perimeter_transformation_l2011_201108


namespace trivia_contest_probability_l2011_201141

def num_questions : ℕ := 4
def num_choices : ℕ := 4
def min_correct : ℕ := 3

def probability_correct_guess : ℚ := 1 / num_choices

def probability_winning : ℚ :=
  (num_questions.choose min_correct) * (probability_correct_guess ^ min_correct) * ((1 - probability_correct_guess) ^ (num_questions - min_correct)) +
  (num_questions.choose (min_correct + 1)) * (probability_correct_guess ^ (min_correct + 1)) * ((1 - probability_correct_guess) ^ (num_questions - (min_correct + 1)))

theorem trivia_contest_probability : probability_winning = 13 / 256 := by
  sorry

end trivia_contest_probability_l2011_201141


namespace sin_greater_than_cos_l2011_201173

theorem sin_greater_than_cos (x : Real) (h : -7*Real.pi/4 < x ∧ x < -3*Real.pi/2) :
  Real.sin (x + 9*Real.pi/4) > Real.cos x := by
  sorry

end sin_greater_than_cos_l2011_201173


namespace older_sibling_age_l2011_201147

/-- Given two siblings with a two-year age gap and their combined age, 
    prove the older sibling's age -/
theorem older_sibling_age 
  (h : ℕ) -- Hyeongjun's age
  (s : ℕ) -- Older sister's age
  (age_gap : s = h + 2) -- Two-year age gap condition
  (total_age : h + s = 26) -- Sum of ages condition
  : s = 14 := by
  sorry

end older_sibling_age_l2011_201147


namespace range_of_function_l2011_201131

theorem range_of_function (f : ℝ → ℝ) (h : ∀ x, f x ∈ Set.Icc (3/8) (4/9)) :
  ∀ x, f x + Real.sqrt (1 - 2 * f x) ∈ Set.Icc (7/9) (7/8) := by
  sorry

end range_of_function_l2011_201131


namespace triangle_side_length_is_2_sqrt_3_l2011_201134

/-- Represents a semicircle with its center and radius -/
structure Semicircle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an equilateral triangle with three touching semicircles -/
structure TriangleWithSemicircles where
  triangle : List (ℝ × ℝ)
  semicircles : List Semicircle

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (triangle : List (ℝ × ℝ)) : Prop := sorry

/-- Checks if semicircles touch each other and the triangle -/
def areTouchingSemicircles (t : TriangleWithSemicircles) : Prop := sorry

/-- Checks if the diameter of each semicircle lies along a side of the triangle -/
def semicirclesAlongSides (t : TriangleWithSemicircles) : Prop := sorry

/-- Calculates the side length of the triangle -/
noncomputable def triangleSideLength (t : TriangleWithSemicircles) : ℝ := sorry

theorem triangle_side_length_is_2_sqrt_3 (t : TriangleWithSemicircles) :
  isEquilateral t.triangle ∧
  (∀ s ∈ t.semicircles, s.radius = 1) ∧
  areTouchingSemicircles t ∧
  semicirclesAlongSides t →
  triangleSideLength t = 2 * Real.sqrt 3 := by
  sorry

end triangle_side_length_is_2_sqrt_3_l2011_201134


namespace difference_of_squares_528_529_l2011_201172

theorem difference_of_squares_528_529 : (528 * 528) - (527 * 529) = 1 := by
  sorry

end difference_of_squares_528_529_l2011_201172


namespace function_composition_equality_l2011_201149

/-- Given a function f(x) = a x^2 - √2, where a is a constant,
    prove that if f(f(√2)) = -√2, then a = √2 / 2 -/
theorem function_composition_equality (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 - Real.sqrt 2
  f (f (Real.sqrt 2)) = -Real.sqrt 2 → a = Real.sqrt 2 / 2 := by
  sorry

end function_composition_equality_l2011_201149


namespace parallel_line_slope_l2011_201139

/-- The slope of any line parallel to the line containing points (3, -2) and (1, 5) is -7/2 -/
theorem parallel_line_slope : ∀ (m : ℚ), 
  (∃ (b : ℚ), ∀ (x y : ℚ), y = m * x + b → 
    (∃ (k : ℚ), y - (-2) = m * (x - 3) ∧ y - 5 = m * (x - 1))) → 
  m = -7/2 := by sorry

end parallel_line_slope_l2011_201139


namespace consecutive_squares_sum_l2011_201146

theorem consecutive_squares_sum (x : ℤ) :
  (x + 1)^2 - x^2 = 199 → x^2 + (x + 1)^2 = 19801 := by
  sorry

end consecutive_squares_sum_l2011_201146


namespace horner_method_v3_equals_55_l2011_201123

def horner_polynomial (x : ℝ) : ℝ := 3*x^5 + 8*x^4 - 3*x^3 + 5*x^2 + 12*x - 6

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 8
  let v2 := v1 * x - 3
  v2 * x + 5

theorem horner_method_v3_equals_55 :
  horner_v3 2 = 55 :=
by sorry

end horner_method_v3_equals_55_l2011_201123


namespace line_slope_l2011_201129

/-- The slope of the line x + √3y + 2 = 0 is -1/√3 -/
theorem line_slope (x y : ℝ) : x + Real.sqrt 3 * y + 2 = 0 → 
  (y - (-2/Real.sqrt 3)) / (x - 0) = -1 / Real.sqrt 3 := by
  sorry

end line_slope_l2011_201129


namespace freshman_count_proof_l2011_201115

theorem freshman_count_proof :
  ∃! n : ℕ, n < 600 ∧ n % 25 = 24 ∧ n % 19 = 10 ∧ n = 574 := by
  sorry

end freshman_count_proof_l2011_201115


namespace diagonals_25_sided_polygon_l2011_201193

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  -- Add any necessary properties for a convex polygon

/-- The number of diagonals in a convex polygon that skip exactly one vertex -/
def diagonals_skipping_one_vertex (n : ℕ) : ℕ := 2 * n

/-- Theorem: In a convex 25-sided polygon, there are 50 diagonals that skip exactly one vertex -/
theorem diagonals_25_sided_polygon :
  diagonals_skipping_one_vertex 25 = 50 := by
  sorry

#eval diagonals_skipping_one_vertex 25

end diagonals_25_sided_polygon_l2011_201193


namespace expression_evaluation_l2011_201177

theorem expression_evaluation :
  let x : ℚ := -1/2
  let expr := 2*x^2 + 6*x - 6 - (-2*x^2 + 4*x + 1)
  expr = -7 := by sorry

end expression_evaluation_l2011_201177


namespace sine_cosine_inequality_l2011_201186

theorem sine_cosine_inequality (x : ℝ) (n : ℕ+) :
  (Real.sin (2 * x))^(n : ℕ) + ((Real.sin x)^(n : ℕ) - (Real.cos x)^(n : ℕ))^2 ≤ 1 := by
  sorry

end sine_cosine_inequality_l2011_201186


namespace rectangular_field_length_l2011_201150

theorem rectangular_field_length : 
  ∀ (w : ℝ), 
    w > 0 → 
    w^2 + (w + 10)^2 = 22^2 → 
    w + 10 = 22 :=
by
  sorry

end rectangular_field_length_l2011_201150


namespace snow_volume_calculation_l2011_201137

/-- Calculates the volume of snow to be shoveled from a partially melted rectangular pathway -/
theorem snow_volume_calculation (length width : ℝ) (depth_full depth_half : ℝ) 
  (h_length : length = 30)
  (h_width : width = 4)
  (h_depth_full : depth_full = 1)
  (h_depth_half : depth_half = 1/2) :
  length * width * depth_full / 2 + length * width * depth_half / 2 = 90 :=
by sorry

end snow_volume_calculation_l2011_201137


namespace ice_cream_cost_is_734_l2011_201109

/-- The cost of Mrs. Hilt's ice cream purchase -/
def ice_cream_cost : ℚ :=
  let vanilla_price : ℚ := 99 / 100
  let chocolate_price : ℚ := 129 / 100
  let strawberry_price : ℚ := 149 / 100
  let vanilla_quantity : ℕ := 2
  let chocolate_quantity : ℕ := 3
  let strawberry_quantity : ℕ := 1
  vanilla_price * vanilla_quantity +
  chocolate_price * chocolate_quantity +
  strawberry_price * strawberry_quantity

theorem ice_cream_cost_is_734 : ice_cream_cost = 734 / 100 := by
  sorry

end ice_cream_cost_is_734_l2011_201109


namespace line_y_axis_intersection_l2011_201132

/-- The line equation is 5y + 3x = 15 -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line 5y + 3x = 15 with the y-axis is (0, 3) -/
theorem line_y_axis_intersection :
  ∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = 3 := by sorry

end line_y_axis_intersection_l2011_201132


namespace rotated_angle_measure_l2011_201157

/-- Given an initial angle of 45 degrees that is rotated 510 degrees clockwise,
    the resulting new acute angle measures 75 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 45 → 
  rotation = 510 → 
  (((rotation % 360) - initial_angle) % 180) = 75 :=
by sorry

end rotated_angle_measure_l2011_201157


namespace tangent_line_m_value_l2011_201102

/-- The curve function f(x) = x^3 + x - 1 -/
def f (x : ℝ) : ℝ := x^3 + x - 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_line_m_value :
  let p₁ : ℝ × ℝ := (1, f 1)
  let slope : ℝ := f' 1
  let p₂ : ℝ × ℝ := (2, m)
  (∀ m : ℝ, (m - p₁.2) = slope * (p₂.1 - p₁.1) → m = 5) :=
by sorry

end tangent_line_m_value_l2011_201102


namespace right_triangle_roots_l2011_201111

theorem right_triangle_roots (a b z₁ z₂ : ℂ) : 
  (z₁^2 + a*z₁ + b = 0) →
  (z₂^2 + a*z₂ + b = 0) →
  (z₂ = Complex.I * z₁) →
  a^2 / b = 2 := by
sorry

end right_triangle_roots_l2011_201111


namespace ratio_problem_l2011_201174

theorem ratio_problem (x y z w : ℚ) 
  (h1 : x / y = 24)
  (h2 : z / y = 8)
  (h3 : z / w = 1 / 12) :
  x / w = 1 / 4 := by
  sorry

end ratio_problem_l2011_201174


namespace exists_tangent_region_l2011_201168

noncomputable section

-- Define the parabolas
def parabola1 (x : ℝ) : ℝ := x - x^2
def parabola2 (a : ℝ) (x : ℝ) : ℝ := a * (x - x^2)

-- Define the tangent lines
def tangent1 (b : ℝ) (x : ℝ) : ℝ := (1 - 2*b)*x + b^2
def tangent2 (a : ℝ) (c : ℝ) (x : ℝ) : ℝ := a*(1 - 2*c)*x + a*c^2

-- Define the intersection point of tangent lines
def intersection (a b c : ℝ) : ℝ := (a*c^2 - b^2) / ((1 - 2*b) - a*(1 - 2*c))

-- Define the condition for the third point
def third_point (x b : ℝ) : ℝ := 2*x - b

-- Theorem statement
theorem exists_tangent_region (a : ℝ) (h : a ≥ 2) :
  ∃ (b c : ℝ), 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧
  let x := intersection a b c
  let d := third_point x b
  0 < d ∧ d < 1 ∧
  (tangent1 b x = tangent2 a c x ∨ tangent1 d x = tangent1 b x ∨ tangent2 a d x = tangent2 a c x) :=
sorry

end exists_tangent_region_l2011_201168


namespace equation_solution_l2011_201196

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -1 ∧ x₂ = 5 ∧ 
  (∀ x : ℝ, (x + 1)^2 = 6*x + 6 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end equation_solution_l2011_201196


namespace horners_method_for_f_l2011_201182

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 - 3*x + 2

theorem horners_method_for_f :
  f 3 = 2 := by
  sorry

end horners_method_for_f_l2011_201182


namespace triangle_angles_sum_l2011_201120

theorem triangle_angles_sum (x y : ℕ+) : 
  (5 * x + 3 * y : ℕ) + (3 * x + 20 : ℕ) + (10 * y + 30 : ℕ) = 180 → x + y = 15 := by
  sorry

end triangle_angles_sum_l2011_201120


namespace altitude_from_C_to_AB_l2011_201104

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 5)
def B : ℝ × ℝ := (1, -2)
def C : ℝ × ℝ := (-6, 4)

-- Define the equation of the altitude
def altitude_equation (x y : ℝ) : Prop := 7 * x - 6 * y + 30 = 0

-- Theorem statement
theorem altitude_from_C_to_AB :
  ∀ x y : ℝ, altitude_equation x y ↔ 
  (x - C.1) * (B.1 - A.1) + (y - C.2) * (B.2 - A.2) = 0 ∧
  (x, y) ≠ C :=
sorry

end altitude_from_C_to_AB_l2011_201104


namespace milk_production_calculation_l2011_201185

/-- Calculates the total milk production for a herd of cows over a given number of days -/
def total_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_cows * milk_per_cow_per_day * num_days

/-- Theorem stating the total milk production for 120 cows over 15 days -/
theorem milk_production_calculation :
  total_milk_production 120 1362 15 = 2451600 := by
  sorry

#eval total_milk_production 120 1362 15

end milk_production_calculation_l2011_201185


namespace compound_oxygen_atoms_l2011_201124

/-- The number of Oxygen atoms in a compound with given properties -/
def oxygenAtoms (molecularWeight : ℕ) (hydrogenAtoms carbonAtoms : ℕ) 
  (atomicWeightH atomicWeightC atomicWeightO : ℕ) : ℕ :=
  (molecularWeight - (hydrogenAtoms * atomicWeightH + carbonAtoms * atomicWeightC)) / atomicWeightO

/-- Theorem stating the number of Oxygen atoms in the compound -/
theorem compound_oxygen_atoms :
  oxygenAtoms 62 2 1 1 12 16 = 3 := by
  sorry

end compound_oxygen_atoms_l2011_201124


namespace find_number_l2011_201130

theorem find_number : ∃ x : ℝ, (0.8 * x - 20 = 60) ∧ x = 100 := by
  sorry

end find_number_l2011_201130


namespace symmetric_curve_is_correct_l2011_201165

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the symmetric curve
def symmetric_curve (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 1

-- Theorem statement
theorem symmetric_curve_is_correct : 
  ∀ (x y x' y' : ℝ), 
    given_circle x y → 
    symmetry_line ((x + x') / 2) ((y + y') / 2) → 
    symmetric_curve x' y' :=
sorry

end symmetric_curve_is_correct_l2011_201165


namespace square_diff_over_square_sum_l2011_201195

theorem square_diff_over_square_sum (a b : ℝ) (h : a * b / (a^2 + b^2) = 1/4) :
  |a^2 - b^2| / (a^2 + b^2) = Real.sqrt 3 / 2 := by
  sorry

end square_diff_over_square_sum_l2011_201195


namespace systematic_sample_theorem_l2011_201188

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  populationSize : ℕ
  sampleSize : ℕ
  firstElement : ℕ
  commonDifference : ℕ

/-- Checks if a given list is a valid systematic sample -/
def isValidSample (s : SystematicSample) (sample : List ℕ) : Prop :=
  sample.length = s.sampleSize ∧
  sample.head! = s.firstElement ∧
  ∀ i, 0 < i → i < s.sampleSize → 
    sample[i]! = sample[i-1]! + s.commonDifference ∧
    sample[i]! ≤ s.populationSize

/-- The main theorem to prove -/
theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.populationSize = 60)
  (h2 : s.sampleSize = 5)
  (h3 : 4 ∈ (List.range s.sampleSize).map (fun i => s.firstElement + i * s.commonDifference)) :
  isValidSample s [4, 16, 28, 40, 52] :=
sorry

end systematic_sample_theorem_l2011_201188


namespace cyclist_speed_ratio_l2011_201143

theorem cyclist_speed_ratio :
  ∀ (v_A v_B : ℝ),
    v_A > 0 →
    v_B > 0 →
    v_A < v_B →
    (v_B - v_A) * 4.5 = 10 →
    v_A + v_B = 10 →
    v_A / v_B = 61 / 29 := by
  sorry

end cyclist_speed_ratio_l2011_201143


namespace complex_power_sum_l2011_201114

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^2021 + 1/z^2021 = Real.sqrt 3 := by
  sorry

end complex_power_sum_l2011_201114


namespace significant_difference_l2011_201113

/-- Represents the distribution of X, where X is the number of specified two mice
    assigned to the control group out of 40 total mice --/
def distribution_X : Fin 3 → ℚ
| 0 => 19/78
| 1 => 20/39
| 2 => 19/78

/-- The expectation of X --/
def E_X : ℚ := 1

/-- The median of the increase in body weight of all 40 mice --/
def median_weight : ℝ := 23.4

/-- The contingency table of mice counts below and above median --/
def contingency_table : Fin 2 → Fin 2 → ℕ
| 0, 0 => 6  -- Control group, below median
| 0, 1 => 14 -- Control group, above or equal to median
| 1, 0 => 14 -- Experimental group, below median
| 1, 1 => 6  -- Experimental group, above or equal to median

/-- The K² statistic --/
def K_squared : ℝ := 6.400

/-- The critical value for 95% confidence level --/
def critical_value_95 : ℝ := 3.841

/-- Theorem stating that the K² value is greater than the critical value,
    indicating a significant difference between groups --/
theorem significant_difference : K_squared > critical_value_95 := by sorry

end significant_difference_l2011_201113


namespace aisha_mp3_song_count_l2011_201138

/-- The number of songs on Aisha's mp3 player after a series of additions and removals -/
def final_song_count (initial : ℕ) (first_addition : ℕ) (removed : ℕ) : ℕ :=
  let after_first_addition := initial + first_addition
  let doubled := after_first_addition * 2
  let before_removal := after_first_addition + doubled
  before_removal - removed

/-- Theorem stating that given the initial conditions, the final number of songs is 2950 -/
theorem aisha_mp3_song_count :
  final_song_count 500 500 50 = 2950 := by
  sorry

end aisha_mp3_song_count_l2011_201138


namespace sue_driving_days_l2011_201127

theorem sue_driving_days (total_cost : ℚ) (sister_days : ℚ) (sue_payment : ℚ) :
  total_cost = 2100 →
  sister_days = 4 →
  sue_payment = 900 →
  ∃ (sue_days : ℚ), sue_days + sister_days = 7 ∧ sue_days / (7 - sue_days) = sue_payment / (total_cost - sue_payment) ∧ sue_days = 3 :=
by sorry

end sue_driving_days_l2011_201127


namespace largest_prime_divisor_of_100111011_base6_l2011_201103

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Finds all divisors of a number -/
def divisors (n : ℕ) : List ℕ := sorry

theorem largest_prime_divisor_of_100111011_base6 :
  let n := base6ToBase10 100111011
  ∃ (d : ℕ), d ∈ divisors n ∧ isPrime d ∧ d = 181 ∧ ∀ (p : ℕ), p ∈ divisors n → isPrime p → p ≤ d :=
sorry

end largest_prime_divisor_of_100111011_base6_l2011_201103


namespace triangle_problem_l2011_201122

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sin (t.A + t.C) = 8 * (Real.sin (t.B / 2))^2)
  (h2 : t.a + t.c = 6)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = 2) : 
  Real.cos t.B = 15/17 ∧ t.b = 2 := by
  sorry


end triangle_problem_l2011_201122


namespace shirt_coat_ratio_l2011_201110

/-- Given a shirt costing $150 and a total cost of $600 for the shirt and coat,
    prove that the ratio of the cost of the shirt to the cost of the coat is 1:3. -/
theorem shirt_coat_ratio (shirt_cost coat_cost total_cost : ℕ) : 
  shirt_cost = 150 → 
  total_cost = 600 → 
  total_cost = shirt_cost + coat_cost →
  (shirt_cost : ℚ) / coat_cost = 1 / 3 := by
  sorry

end shirt_coat_ratio_l2011_201110


namespace total_players_l2011_201163

theorem total_players (kabaddi : ℕ) (kho_kho_only : ℕ) (both : ℕ) 
  (h1 : kabaddi = 10) 
  (h2 : kho_kho_only = 25) 
  (h3 : both = 5) : 
  kabaddi + kho_kho_only - both = 30 := by
  sorry

end total_players_l2011_201163


namespace commodity_price_difference_l2011_201107

theorem commodity_price_difference (total_cost first_price : ℕ) 
  (h1 : total_cost = 827)
  (h2 : first_price = 477)
  (h3 : first_price > total_cost - first_price) : 
  first_price - (total_cost - first_price) = 127 := by
  sorry

end commodity_price_difference_l2011_201107


namespace least_side_of_right_triangle_l2011_201190

theorem least_side_of_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a = 8 → b = 15 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c ≥ 8 :=
by sorry

end least_side_of_right_triangle_l2011_201190


namespace triangle_base_length_l2011_201171

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = (base * height) / 2 → area = 12 → height = 6 → base = 4 := by
  sorry

end triangle_base_length_l2011_201171
