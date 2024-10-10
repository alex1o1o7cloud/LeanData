import Mathlib

namespace no_special_sequence_exists_l651_65124

theorem no_special_sequence_exists : ¬ ∃ (a : ℕ → ℕ),
  (∀ n : ℕ, a n < a (n + 1)) ∧
  (∃ N : ℕ, ∀ m : ℕ, m ≥ N →
    ∃! (i j : ℕ), m = a i + a j) :=
by sorry

end no_special_sequence_exists_l651_65124


namespace james_purchase_cost_l651_65133

theorem james_purchase_cost : 
  let num_shirts : ℕ := 10
  let num_pants : ℕ := num_shirts / 2
  let shirt_cost : ℕ := 6
  let pants_cost : ℕ := 8
  let total_cost : ℕ := num_shirts * shirt_cost + num_pants * pants_cost
  total_cost = 100 := by sorry

end james_purchase_cost_l651_65133


namespace total_interest_is_860_l651_65184

def inheritance : ℝ := 12000
def investment1 : ℝ := 5000
def rate1 : ℝ := 0.06
def rate2 : ℝ := 0.08

def total_interest : ℝ :=
  investment1 * rate1 + (inheritance - investment1) * rate2

theorem total_interest_is_860 : total_interest = 860 := by sorry

end total_interest_is_860_l651_65184


namespace stack_toppled_plates_l651_65142

/-- The number of plates in Alice's stack when it toppled over --/
def total_plates (initial_plates added_plates : ℕ) : ℕ :=
  initial_plates + added_plates

/-- Theorem: The total number of plates when the stack toppled is 64 --/
theorem stack_toppled_plates :
  total_plates 27 37 = 64 := by
  sorry

end stack_toppled_plates_l651_65142


namespace triangle_property_l651_65148

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (a+c)/b = cos(C) + √3*sin(C), then B = 60° and when b = 2, the max area is √3 -/
theorem triangle_property (a b c : ℝ) (A B C : Real) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (a + c) / b = Real.cos C + Real.sqrt 3 * Real.sin C →
  B = π / 3 ∧ 
  (b = 2 → ∃ (area : ℝ), area ≤ Real.sqrt 3 ∧ 
    ∀ (other_area : ℝ), (∃ (a' c' : ℝ), a' > 0 ∧ c' > 0 ∧ 
      other_area = 1/2 * a' * 2 * Real.sin (π/3)) → other_area ≤ area) := by
  sorry

end triangle_property_l651_65148


namespace factorization_equality_l651_65138

theorem factorization_equality (x : ℝ) : 75 * x^3 - 225 * x^10 = 75 * x^3 * (1 - 3 * x^7) := by
  sorry

end factorization_equality_l651_65138


namespace summer_program_students_l651_65150

theorem summer_program_students : ∃! n : ℕ, 0 < n ∧ n < 500 ∧ n % 25 = 24 ∧ n % 21 = 14 ∧ n = 449 := by
  sorry

end summer_program_students_l651_65150


namespace equation_solution_l651_65174

theorem equation_solution : ∃! x : ℚ, (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 ∧ x = -7/6 := by
  sorry

end equation_solution_l651_65174


namespace union_complement_equality_complement_intersection_equality_l651_65109

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {1, 3, 5, 7}

-- Theorem for part (1)
theorem union_complement_equality :
  A ∪ (U \ B) = {2, 4, 5, 6} := by sorry

-- Theorem for part (2)
theorem complement_intersection_equality :
  U \ (A ∩ B) = {1, 2, 3, 4, 6, 7} := by sorry

end union_complement_equality_complement_intersection_equality_l651_65109


namespace composite_divisor_inequality_l651_65197

def d (k : ℕ) : ℕ := (Nat.divisors k).card

theorem composite_divisor_inequality (n : ℕ) (h_composite : ¬ Nat.Prime n) :
  ∃ m : ℕ, m > 0 ∧ m ∣ n ∧ m * m ≤ n ∧ d n ≤ d m * d m * d m := by
  sorry

end composite_divisor_inequality_l651_65197


namespace remainder_9876543210_mod_140_l651_65191

theorem remainder_9876543210_mod_140 : 9876543210 % 140 = 70 := by
  sorry

end remainder_9876543210_mod_140_l651_65191


namespace max_value_of_exponential_difference_l651_65116

theorem max_value_of_exponential_difference :
  ∃ (max : ℝ), max = 2/3 ∧ ∀ (x : ℝ), 2^x - 8^x ≤ max :=
by sorry

end max_value_of_exponential_difference_l651_65116


namespace A_subset_B_iff_l651_65158

/-- The set A parameterized by a -/
def A (a : ℝ) : Set ℝ := {x | 1 < a * x ∧ a * x < 2}

/-- The set B -/
def B : Set ℝ := {x | |x| < 1}

/-- Theorem stating the condition for A to be a subset of B -/
theorem A_subset_B_iff (a : ℝ) : A a ⊆ B ↔ |a| ≥ 2 ∨ a = 0 := by sorry

end A_subset_B_iff_l651_65158


namespace game_ends_after_46_rounds_game_doesnt_end_before_46_rounds_l651_65135

/-- Represents the state of the game at any point --/
structure GameState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a single round of the game --/
def play_round (state : GameState) : GameState :=
  if state.a ≥ state.b ∧ state.a ≥ state.c then
    { a := state.a - 3, b := state.b + 1, c := state.c + 1 }
  else if state.b ≥ state.a ∧ state.b ≥ state.c then
    { a := state.a + 1, b := state.b - 3, c := state.c + 1 }
  else
    { a := state.a + 1, b := state.b + 1, c := state.c - 3 }

/-- Plays the game for a given number of rounds --/
def play_game (initial_state : GameState) (rounds : ℕ) : GameState :=
  match rounds with
  | 0 => initial_state
  | n + 1 => play_round (play_game initial_state n)

/-- The main theorem stating that the game ends after 46 rounds --/
theorem game_ends_after_46_rounds :
  let initial_state := { a := 18, b := 17, c := 16 : GameState }
  let final_state := play_game initial_state 46
  final_state.a = 0 ∨ final_state.b = 0 ∨ final_state.c = 0 :=
by sorry

/-- The game doesn't end before 46 rounds --/
theorem game_doesnt_end_before_46_rounds :
  let initial_state := { a := 18, b := 17, c := 16 : GameState }
  ∀ n < 46, let state := play_game initial_state n
    state.a > 0 ∧ state.b > 0 ∧ state.c > 0 :=
by sorry

end game_ends_after_46_rounds_game_doesnt_end_before_46_rounds_l651_65135


namespace coefficient_of_x_cubed_is_39_l651_65193

def expression (x : ℝ) : ℝ :=
  2 * (x^2 - 2*x^3 + 2*x) + 4 * (x + 3*x^3 - 2*x^2 + 2*x^5 - x^3) - 7 * (2 + 2*x - 5*x^3 - x^2)

theorem coefficient_of_x_cubed_is_39 :
  (deriv (deriv (deriv expression))) 0 / 6 = 39 := by sorry

end coefficient_of_x_cubed_is_39_l651_65193


namespace isosceles_triangle_perimeter_l651_65106

/-- An isosceles triangle with sides 4 and 9 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 4 ∧ b = 9 ∧ c = 9 →  -- Two sides are 9, one side is 4
  a < b + c ∧ b < a + c ∧ c < a + b →  -- Triangle inequality
  a + b + c = 22 :=  -- Perimeter is 22
by
  sorry

end isosceles_triangle_perimeter_l651_65106


namespace no_sequence_satisfying_condition_l651_65112

theorem no_sequence_satisfying_condition :
  ¬ (∃ (a : ℝ) (a_n : ℕ → ℝ), 
    (0 < a ∧ a < 1) ∧
    (∀ n : ℕ, n > 0 → a_n n > 0) ∧
    (∀ n : ℕ, n > 0 → 1 + a_n (n + 1) ≤ a_n n + (a / n) * a_n n)) :=
by sorry

end no_sequence_satisfying_condition_l651_65112


namespace additional_money_needed_mrs_smith_shopping_l651_65186

/-- Calculates the additional money needed for Mrs. Smith's shopping --/
theorem additional_money_needed (total_budget dress_budget shoe_budget accessory_budget : ℚ)
  (dress_discount shoe_discount accessory_discount : ℚ) : ℚ :=
  let dress_needed := dress_budget * (1 + 2/5)
  let shoe_needed := shoe_budget * (1 + 2/5)
  let accessory_needed := accessory_budget * (1 + 2/5)
  let dress_discounted := dress_needed * (1 - dress_discount)
  let shoe_discounted := shoe_needed * (1 - shoe_discount)
  let accessory_discounted := accessory_needed * (1 - accessory_discount)
  let total_needed := dress_discounted + shoe_discounted + accessory_discounted
  total_needed - total_budget

/-- Proves that Mrs. Smith needs $84.50 more to complete her shopping --/
theorem mrs_smith_shopping :
  additional_money_needed 500 300 150 50 (20/100) (10/100) (15/100) = 169/2 := by
  sorry

end additional_money_needed_mrs_smith_shopping_l651_65186


namespace geometric_sequence_propositions_l651_65169

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_propositions (a : ℕ → ℝ) (h : GeometricSequence a) :
  (((a 1 < a 2) ∧ (a 2 < a 3)) → IncreasingSequence a) ∧
  (IncreasingSequence a → ((a 1 < a 2) ∧ (a 2 < a 3))) ∧
  (¬((a 1 < a 2) ∧ (a 2 < a 3)) → ¬IncreasingSequence a) ∧
  (¬IncreasingSequence a → ¬((a 1 < a 2) ∧ (a 2 < a 3))) :=
by
  sorry

end geometric_sequence_propositions_l651_65169


namespace movie_shelf_distribution_l651_65139

theorem movie_shelf_distribution (n : ℕ) : 
  (∃ k : ℕ, n + 1 = 2 * k) → Odd n := by
  sorry

end movie_shelf_distribution_l651_65139


namespace remainder_3_100_plus_4_mod_5_l651_65164

theorem remainder_3_100_plus_4_mod_5 : (3^100 + 4) % 5 = 0 := by
  sorry

end remainder_3_100_plus_4_mod_5_l651_65164


namespace inequality_proof_l651_65141

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha₁ : 0 < a₁) (ha₂ : 0 < a₂) (ha₃ : 0 < a₃) 
  (hb₁ : 0 < b₁) (hb₂ : 0 < b₂) (hb₃ : 0 < b₃) : 
  (a₁*b₂ + a₂*b₁ + a₂*b₃ + a₃*b₂ + a₃*b₁ + a₁*b₃)^2 ≥ 
  4*(a₁*a₂ + a₂*a₃ + a₃*a₁)*(b₁*b₂ + b₂*b₃ + b₃*b₁) := by
  sorry

end inequality_proof_l651_65141


namespace linear_functions_properties_l651_65151

/-- Two linear functions -/
def y₁ (x : ℝ) : ℝ := -x + 1
def y₂ (x : ℝ) : ℝ := -3*x + 2

theorem linear_functions_properties :
  (∃ a : ℝ, ∀ x > 0, y₁ x = a + y₂ x → a > -1) ∧
  (∀ x y : ℝ, y₁ x = y ∧ y₂ x = y → 12*x^2 + 12*x*y + 3*y^2 = 27/4) ∧
  (∃ A B : ℝ, ∀ x : ℝ, x ≠ 1 ∧ 3*x ≠ 2 →
    (4-2*x)/((3*x-2)*(x-1)) = A/(y₁ x) + B/(y₂ x) →
    A/B + B/A = -4.25) := by sorry

end linear_functions_properties_l651_65151


namespace two_unique_intersection_lines_l651_65145

/-- A parabola defined by the equation y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line on the plane -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Predicate to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- Predicate to check if a line intersects the parabola at only one point -/
def uniqueIntersection (l : Line) : Prop :=
  ∃! p : Point, pointOnLine p l ∧ parabola p.x p.y

/-- The point (2, 4) -/
def givenPoint : Point := ⟨2, 4⟩

/-- The theorem stating that there are exactly two lines passing through (2, 4) 
    that intersect the parabola y^2 = 8x at only one point -/
theorem two_unique_intersection_lines : 
  ∃! (l1 l2 : Line), 
    pointOnLine givenPoint l1 ∧ 
    pointOnLine givenPoint l2 ∧ 
    uniqueIntersection l1 ∧ 
    uniqueIntersection l2 ∧ 
    l1 ≠ l2 :=
sorry

end two_unique_intersection_lines_l651_65145


namespace parallel_planes_false_l651_65173

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (belongs_to : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- State the theorem
theorem parallel_planes_false :
  (¬ (α = β)) →  -- α and β are non-coincident planes
  (¬ (m = n)) →  -- m and n are non-coincident lines
  ¬ (
    (belongs_to m α ∧ belongs_to n α ∧ 
     parallel_line_plane m β ∧ parallel_line_plane n β) → 
    (parallel α β)
  ) := by sorry

end parallel_planes_false_l651_65173


namespace sum_first_six_primes_mod_seventh_prime_l651_65125

def first_six_primes : List ℕ := [2, 3, 5, 7, 11, 13]
def seventh_prime : ℕ := 17

theorem sum_first_six_primes_mod_seventh_prime :
  (first_six_primes.sum % seventh_prime) = 7 := by sorry

end sum_first_six_primes_mod_seventh_prime_l651_65125


namespace cubic_function_property_l651_65137

/-- Given a cubic function f(x) = ax³ - bx + 1 where a and b are real numbers,
    if f(-2) = -1, then f(2) = 3 -/
theorem cubic_function_property (a b : ℝ) :
  (fun x => a * x^3 - b * x + 1) (-2) = -1 →
  (fun x => a * x^3 - b * x + 1) 2 = 3 := by
  sorry

end cubic_function_property_l651_65137


namespace right_trapezoid_diagonals_l651_65188

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- Length of the smaller base -/
  small_base : ℝ
  /-- Length of the larger base -/
  large_base : ℝ
  /-- Angle at one vertex of the smaller base (in radians) -/
  angle_at_small_base : ℝ

/-- The diagonals of the trapezoid -/
def diagonals (t : RightTrapezoid) : ℝ × ℝ :=
  sorry

theorem right_trapezoid_diagonals :
  let t : RightTrapezoid := {
    small_base := 6,
    large_base := 8,
    angle_at_small_base := 2 * Real.pi / 3  -- 120° in radians
  }
  diagonals t = (4 * Real.sqrt 3, 2 * Real.sqrt 19) := by
  sorry

end right_trapezoid_diagonals_l651_65188


namespace percentage_problem_l651_65161

theorem percentage_problem (P : ℝ) : 
  (5 / 100 * 6400 = P / 100 * 650 + 190) → P = 20 := by
  sorry

end percentage_problem_l651_65161


namespace arctan_equation_solution_l651_65102

theorem arctan_equation_solution :
  ∃ x : ℝ, 2 * Real.arctan (1/3) + Real.arctan (1/5) + Real.arctan (1/x) = π/4 ∧ x = 21/2 := by
  sorry

end arctan_equation_solution_l651_65102


namespace distance_between_sets_l651_65113

/-- The distance between two sets A and B, where
    A = {y | y = 2x - 1, x ∈ ℝ} and
    B = {y | y = x² + 1, x ∈ ℝ},
    is defined as the minimum value of |a - b|, where a ∈ A and b ∈ B. -/
theorem distance_between_sets :
  ∃ (x y : ℝ), |((2 * x) - 1) - (y^2 + 1)| = 0 := by
  sorry

end distance_between_sets_l651_65113


namespace a_1995_equals_3_l651_65104

def units_digit (n : ℕ) : ℕ := n % 10

def a (n : ℕ) : ℕ := units_digit (7^n)

theorem a_1995_equals_3 : a 1995 = 3 := by sorry

end a_1995_equals_3_l651_65104


namespace bus_driver_regular_rate_l651_65163

/-- Represents the compensation structure and work hours of a bus driver --/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Theorem stating that given the conditions, the regular rate is $18 per hour --/
theorem bus_driver_regular_rate 
  (comp : BusDriverCompensation)
  (h1 : comp.regularHours = 40)
  (h2 : comp.overtimeHours = 48.12698412698413 - 40)
  (h3 : comp.overtimeRate = comp.regularRate * 1.75)
  (h4 : comp.totalCompensation = 976)
  (h5 : comp.totalCompensation = comp.regularRate * comp.regularHours + 
                                 comp.overtimeRate * comp.overtimeHours) :
  comp.regularRate = 18 := by
  sorry


end bus_driver_regular_rate_l651_65163


namespace franks_total_work_hours_l651_65187

/-- Calculates the total hours worked given the number of hours per day and number of days --/
def totalHours (hoursPerDay : ℕ) (numDays : ℕ) : ℕ :=
  hoursPerDay * numDays

/-- Theorem: Frank's total work hours --/
theorem franks_total_work_hours :
  totalHours 8 4 = 32 := by
  sorry

end franks_total_work_hours_l651_65187


namespace sales_equation_l651_65134

/-- Represents the salesman's total sales -/
def S : ℝ := sorry

/-- Old commission rate -/
def old_rate : ℝ := 0.05

/-- New fixed salary -/
def new_fixed_salary : ℝ := 1300

/-- New commission rate -/
def new_rate : ℝ := 0.025

/-- Sales threshold for new commission -/
def threshold : ℝ := 4000

/-- Difference in remuneration between new and old schemes -/
def remuneration_difference : ℝ := 600

/-- Theorem stating the equation that the salesman's total sales must satisfy -/
theorem sales_equation : 
  new_fixed_salary + new_rate * (S - threshold) = old_rate * S + remuneration_difference :=
sorry

end sales_equation_l651_65134


namespace not_prime_1000000027_l651_65131

theorem not_prime_1000000027 : ¬ Nat.Prime 1000000027 := by
  sorry

end not_prime_1000000027_l651_65131


namespace expression_simplification_l651_65152

theorem expression_simplification (a : ℚ) (h : a = -1/2) :
  (4 - 3*a)*(1 + 2*a) - 3*a*(1 - 2*a) = 3 := by
  sorry

end expression_simplification_l651_65152


namespace min_value_fraction_l651_65101

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 2*b = 2) :
  (a + b) / (a * b) ≥ (3 + 2 * Real.sqrt 2) / 2 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 2 ∧ (a₀ + b₀) / (a₀ * b₀) = (3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end min_value_fraction_l651_65101


namespace cos_two_alpha_l651_65183

theorem cos_two_alpha (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := by
  sorry

end cos_two_alpha_l651_65183


namespace f_second_derivative_at_zero_l651_65140

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x - Real.cos x

theorem f_second_derivative_at_zero :
  (deriv (deriv f)) 0 = 2 := by
  sorry

end f_second_derivative_at_zero_l651_65140


namespace inequality_system_solution_l651_65153

theorem inequality_system_solution (x : ℝ) :
  (x - 1 > 0 ∧ (2 * x + 1) / 3 ≤ 3) ↔ (1 < x ∧ x ≤ 4) :=
by sorry

end inequality_system_solution_l651_65153


namespace line_intercepts_sum_l651_65132

/-- Given a line with equation y - 5 = 3(x - 9), the sum of its x-intercept and y-intercept is -44/3 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 5 = 3 * (x - 9)) → 
  (∃ x_int y_int : ℝ, 
    (y_int - 5 = 3 * (x_int - 9)) ∧ 
    (0 - 5 = 3 * (x_int - 9)) ∧ 
    (y_int - 5 = 3 * (0 - 9)) ∧ 
    (x_int + y_int = -44/3)) := by
  sorry

end line_intercepts_sum_l651_65132


namespace factorial_fraction_equality_l651_65143

theorem factorial_fraction_equality : (5 * Nat.factorial 7 + 35 * Nat.factorial 6) / Nat.factorial 8 = 5 / 4 := by
  sorry

end factorial_fraction_equality_l651_65143


namespace digit_sum_of_squared_palindrome_l651_65189

theorem digit_sum_of_squared_palindrome (r : ℕ) (x : ℕ) (p q : ℕ) :
  r ≤ 400 →
  x = p * r^3 + p * r^2 + q * r + q →
  7 * q = 17 * p →
  ∃ (a b c : ℕ),
    x^2 = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a →
  2 * (a + b + c) = 400 :=
by sorry

end digit_sum_of_squared_palindrome_l651_65189


namespace relationship_abc_l651_65168

theorem relationship_abc : 
  let a : ℝ := Real.sqrt 2
  let b : ℝ := Real.sqrt 7 - Real.sqrt 3
  let c : ℝ := Real.sqrt 6 - Real.sqrt 2
  a > c ∧ c > b := by sorry

end relationship_abc_l651_65168


namespace coin_value_difference_l651_65110

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : Nat
  nickels : Nat
  dimes : Nat

/-- Calculates the total value in cents for a given coin count -/
def totalValue (coins : CoinCount) : Nat :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Represents the constraint that the total number of coins is 3030 -/
def totalCoins (coins : CoinCount) : Prop :=
  coins.pennies + coins.nickels + coins.dimes = 3030

/-- Represents the constraint that there are at least 10 of each coin type -/
def atLeastTenEach (coins : CoinCount) : Prop :=
  coins.pennies ≥ 10 ∧ coins.nickels ≥ 10 ∧ coins.dimes ≥ 10

/-- The main theorem stating the difference between max and min possible values -/
theorem coin_value_difference :
  ∃ (max min : CoinCount),
    totalCoins max ∧ totalCoins min ∧
    atLeastTenEach max ∧ atLeastTenEach min ∧
    (∀ c, totalCoins c → atLeastTenEach c → totalValue c ≤ totalValue max) ∧
    (∀ c, totalCoins c → atLeastTenEach c → totalValue c ≥ totalValue min) ∧
    totalValue max - totalValue min = 27000 := by
  sorry

end coin_value_difference_l651_65110


namespace intersection_of_M_and_N_l651_65192

-- Define the sets M and N
def M : Set ℝ := {x | 1 - 2/x < 0}
def N : Set ℝ := {x | -1 ≤ x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l651_65192


namespace g_2187_equals_343_l651_65182

-- Define the properties of function g
def satisfies_property (g : ℕ → ℝ) : Prop :=
  ∀ (x y m : ℕ), x > 0 → y > 0 → m > 0 → x + y = 3^m → g x + g y = m^3

-- Theorem statement
theorem g_2187_equals_343 (g : ℕ → ℝ) (h : satisfies_property g) : g 2187 = 343 := by
  sorry

end g_2187_equals_343_l651_65182


namespace parabola_midpoint_to_directrix_distance_l651_65199

/-- Given a parabola y² = 4x and a line passing through its focus intersecting the parabola
    at points A(x₁, y₁) and B(x₂, y₂) with |AB| = 7, the distance from the midpoint M of AB
    to the directrix of the parabola is 7/2. -/
theorem parabola_midpoint_to_directrix_distance
  (x₁ y₁ x₂ y₂ : ℝ) :
  y₁^2 = 4*x₁ →
  y₂^2 = 4*x₂ →
  (x₁ - 1)^2 + y₁^2 = (x₂ - 1)^2 + y₂^2 →
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 49 →
  (x₁ + x₂)/2 + 1 = 7/2 := by sorry

end parabola_midpoint_to_directrix_distance_l651_65199


namespace journey_ratio_theorem_l651_65171

/-- Represents the distance between two towns -/
structure Distance where
  miles : ℝ
  nonneg : miles ≥ 0

/-- Represents the speed of travel -/
structure Speed where
  mph : ℝ
  positive : mph > 0

/-- Represents a journey between two towns -/
structure Journey where
  distance : Distance
  speed : Speed

theorem journey_ratio_theorem 
  (speed_AB : Speed) 
  (speed_BC : Speed) 
  (avg_speed : Speed) 
  (h1 : speed_AB.mph = 60)
  (h2 : speed_BC.mph = 20)
  (h3 : avg_speed.mph = 36) :
  ∃ (dist_AB dist_BC : Distance),
    let journey_AB : Journey := ⟨dist_AB, speed_AB⟩
    let journey_BC : Journey := ⟨dist_BC, speed_BC⟩
    let total_distance : Distance := ⟨dist_AB.miles + dist_BC.miles, by sorry⟩
    let total_time : ℝ := dist_AB.miles / speed_AB.mph + dist_BC.miles / speed_BC.mph
    avg_speed.mph = total_distance.miles / total_time →
    dist_AB.miles / dist_BC.miles = 2 :=
by sorry

end journey_ratio_theorem_l651_65171


namespace square_last_two_digits_averages_l651_65180

def last_two_digits (n : ℕ) : ℕ := n % 100

def valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ 0 < a ∧ a < 50 ∧ 0 < b ∧ b < 50 ∧ last_two_digits (a^2) = last_two_digits (b^2)

def average (a b : ℕ) : ℚ := (a + b : ℚ) / 2

theorem square_last_two_digits_averages :
  {x : ℚ | ∃ a b : ℕ, valid_pair a b ∧ average a b = x} = {10, 15, 20, 25, 30, 35, 40} := by sorry

end square_last_two_digits_averages_l651_65180


namespace gcf_of_154_308_462_l651_65114

theorem gcf_of_154_308_462 : Nat.gcd 154 (Nat.gcd 308 462) = 154 := by
  sorry

end gcf_of_154_308_462_l651_65114


namespace expand_staircase_4_to_7_l651_65120

/-- Calculates the number of toothpicks needed for a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else 2 * n * n + 2 * n

/-- The number of additional toothpicks needed to expand from m steps to n steps -/
def additional_toothpicks (m n : ℕ) : ℕ :=
  toothpicks n - toothpicks m

theorem expand_staircase_4_to_7 :
  additional_toothpicks 4 7 = 48 := by
  sorry

#eval additional_toothpicks 4 7

end expand_staircase_4_to_7_l651_65120


namespace art_museum_picture_distribution_l651_65175

theorem art_museum_picture_distribution (total_pictures : ℕ) (num_exhibits : ℕ) : 
  total_pictures = 154 → num_exhibits = 9 → 
  (∃ (additional_pictures : ℕ), 
    (total_pictures + additional_pictures) % num_exhibits = 0 ∧
    additional_pictures = 8) := by
  sorry

end art_museum_picture_distribution_l651_65175


namespace valid_seating_arrangements_l651_65190

/-- The number of Democrats in the Senate committee -/
def num_democrats : ℕ := 6

/-- The number of Republicans in the Senate committee -/
def num_republicans : ℕ := 4

/-- The total number of politicians in the Senate committee -/
def total_politicians : ℕ := num_democrats + num_republicans

/-- The number of gaps between Democrats where Republicans can be placed -/
def num_gaps : ℕ := num_democrats

/-- Function to calculate the number of valid seating arrangements -/
def seating_arrangements (d r : ℕ) : ℕ :=
  (Nat.factorial (d - 1)) * (Nat.choose d r) * (Nat.factorial r)

/-- Theorem stating the number of valid seating arrangements -/
theorem valid_seating_arrangements :
  seating_arrangements num_democrats num_republicans = 43200 := by
  sorry

end valid_seating_arrangements_l651_65190


namespace complex_product_l651_65178

theorem complex_product (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2) 
  (h2 : Complex.abs z₂ = 3) 
  (h3 : 3 * z₁ - 2 * z₂ = 2 - Complex.I) : 
  z₁ * z₂ = -18/5 + 24/5 * Complex.I := by
sorry

end complex_product_l651_65178


namespace four_collinear_points_l651_65118

open Real

-- Define the curve
def curve (α : ℝ) (x : ℝ) : ℝ := x^4 + 9*x^3 + α*x^2 + 9*x + 4

-- Define the second derivative of the curve
def second_derivative (α : ℝ) (x : ℝ) : ℝ := 12*x^2 + 54*x + 2*α

-- Theorem statement
theorem four_collinear_points (α : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (∃ a b : ℝ, curve α x₁ = a*x₁ + b ∧ 
                curve α x₂ = a*x₂ + b ∧ 
                curve α x₃ = a*x₃ + b ∧ 
                curve α x₄ = a*x₄ + b)) ↔
  α < 30.375 :=
by sorry

end four_collinear_points_l651_65118


namespace sequence_relations_l651_65108

theorem sequence_relations (x y : ℕ → ℝ) 
  (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2)
  (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) :
  (∀ k, x k = 6 * x (k - 1) - x (k - 2)) ∧
  (∀ k, x k = 34 * x (k - 2) - x (k - 4)) ∧
  (∀ k, x k = 198 * x (k - 3) - x (k - 6)) ∧
  (∀ k, y k = 6 * y (k - 1) - y (k - 2)) ∧
  (∀ k, y k = 34 * y (k - 2) - y (k - 4)) ∧
  (∀ k, y k = 198 * y (k - 3) - y (k - 6)) := by
  sorry

end sequence_relations_l651_65108


namespace parallelogram_area_is_fifteen_l651_65119

/-- Represents a parallelogram EFGH with base FG and height FH -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- The theorem stating that the area of the given parallelogram EFGH is 15 -/
theorem parallelogram_area_is_fifteen : ∃ (p : Parallelogram), p.base = 5 ∧ p.height = 3 ∧ area p = 15 := by
  sorry


end parallelogram_area_is_fifteen_l651_65119


namespace opposite_of_negative_abs_two_l651_65185

theorem opposite_of_negative_abs_two : -(- |(-2)|) = 2 := by sorry

end opposite_of_negative_abs_two_l651_65185


namespace unique_non_range_value_l651_65123

def f (k : ℚ) (x : ℚ) : ℚ := (2 * x + k) / (3 * x + 4)

theorem unique_non_range_value (k : ℚ) :
  (f k 5 = 5) →
  (f k 100 = 100) →
  (∀ x ≠ (-4/3), f k (f k x) = x) →
  ∃! y, ∀ x, f k x ≠ y ∧ y = (-8/13) :=
sorry

end unique_non_range_value_l651_65123


namespace line_slope_intercept_sum_l651_65121

/-- Given a line passing through points (1, 2) and (3, 0), 
    prove that the sum of its slope and y-intercept is 2. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (2 = m * 1 + b) → (0 = m * 3 + b) → m + b = 2 := by
  sorry

end line_slope_intercept_sum_l651_65121


namespace complex_equation_solution_l651_65198

theorem complex_equation_solution (a : ℝ) : (a + Complex.I)^2 = 2 * Complex.I → a = 1 := by
  sorry

end complex_equation_solution_l651_65198


namespace probability_rain_given_wind_l651_65157

theorem probability_rain_given_wind (P_rain P_wind P_rain_and_wind : ℝ) 
  (h1 : P_rain = 4/15)
  (h2 : P_wind = 2/5)
  (h3 : P_rain_and_wind = 1/10) :
  P_rain_and_wind / P_wind = 1/4 := by
  sorry

end probability_rain_given_wind_l651_65157


namespace subset_implies_m_values_l651_65167

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 = 1}
def Q (m : ℝ) : Set ℝ := {x | m * x = 1}

-- State the theorem
theorem subset_implies_m_values (m : ℝ) : Q m ⊆ P → m = 0 ∨ m = 1 ∨ m = -1 := by
  sorry

end subset_implies_m_values_l651_65167


namespace total_campers_l651_65111

def basketball_campers : ℕ := 24
def football_campers : ℕ := 32
def soccer_campers : ℕ := 32

theorem total_campers : basketball_campers + football_campers + soccer_campers = 88 := by
  sorry

end total_campers_l651_65111


namespace square_area_from_diagonal_l651_65177

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s ^ 2 = 64 := by sorry

end square_area_from_diagonal_l651_65177


namespace orthocenter_centroid_angle_tangent_sum_l651_65105

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Tangent of an angle -/
def tan (θ : ℝ) : ℝ := sorry

/-- Predicate to check if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop := sorry

/-- Predicate to check if a triangle is isosceles -/
def isIsoscelesTriangle (t : Triangle) : Prop := sorry

theorem orthocenter_centroid_angle_tangent_sum (t : Triangle) :
  ¬(isRightTriangle t) → ¬(isIsoscelesTriangle t) →
  let M := orthocenter t
  let S := centroid t
  let θA := angle t.A M S
  let θB := angle t.B M S
  let θC := angle t.C M S
  (tan θA = tan θB + tan θC) ∨ (tan θB = tan θA + tan θC) ∨ (tan θC = tan θA + tan θB) := by
  sorry

end orthocenter_centroid_angle_tangent_sum_l651_65105


namespace function_range_l651_65122

noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp x - Real.exp x + 2 * x - (1 / 3) * x^3

theorem function_range (a : ℝ) : f (3 * a^2) + f (2 * a - 1) ≥ 0 → a ∈ Set.Icc (-1) (1/3) :=
by sorry

end function_range_l651_65122


namespace banquet_solution_l651_65146

def banquet_problem (total_attendees : ℕ) (resident_price : ℚ) (total_revenue : ℚ) (num_residents : ℕ) : ℚ :=
  let non_residents := total_attendees - num_residents
  let resident_revenue := num_residents * resident_price
  let non_resident_revenue := total_revenue - resident_revenue
  non_resident_revenue / non_residents

theorem banquet_solution :
  banquet_problem 586 12.95 9423.70 219 = 17.95 := by
  sorry

end banquet_solution_l651_65146


namespace overtime_hours_calculation_l651_65117

theorem overtime_hours_calculation (regular_rate : ℝ) (regular_hours : ℝ) (total_pay : ℝ) :
  regular_rate = 3 →
  regular_hours = 40 →
  total_pay = 180 →
  (total_pay - regular_rate * regular_hours) / (2 * regular_rate) = 10 := by
  sorry

end overtime_hours_calculation_l651_65117


namespace equation_solution_l651_65144

theorem equation_solution : 
  ∃ x : ℚ, (2*x - 30) / 3 = (5 - 3*x) / 4 + 1 ∧ x = 147 / 17 := by
  sorry

end equation_solution_l651_65144


namespace hendrix_class_size_l651_65176

theorem hendrix_class_size (initial_students : ℕ) (new_students : ℕ) (transfer_fraction : ℚ) : 
  initial_students = 160 → 
  new_students = 20 → 
  transfer_fraction = 1/3 →
  (initial_students + new_students) - ((initial_students + new_students : ℚ) * transfer_fraction).floor = 120 := by
  sorry

end hendrix_class_size_l651_65176


namespace smallest_next_divisor_after_221_l651_65159

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : Even m) (h3 : m % 221 = 0) : 
  ∃ (d : ℕ), d > 221 ∧ m % d = 0 ∧ d = 247 ∧ 
  ∀ (x : ℕ), 221 < x ∧ x < 247 → m % x ≠ 0 :=
sorry

end smallest_next_divisor_after_221_l651_65159


namespace money_distribution_l651_65154

theorem money_distribution (a b c d : ℚ) : 
  a = (1 : ℚ) / 3 * (b + c + d) →
  b = (2 : ℚ) / 7 * (a + c + d) →
  c = (3 : ℚ) / 11 * (a + b + d) →
  a = b + 20 →
  b = c + 15 →
  a + b + c + d = 720 := by
  sorry

end money_distribution_l651_65154


namespace rebecca_checkerboard_black_squares_l651_65130

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard where
  size : ℕ
  is_black : ℕ → ℕ → Prop

/-- Defines the properties of Rebecca's checkerboard -/
def rebecca_checkerboard : Checkerboard where
  size := 29
  is_black := fun i j => (i + j) % 2 = 0

/-- Counts the number of black squares in a row -/
def black_squares_in_row (c : Checkerboard) (row : ℕ) : ℕ :=
  (c.size + 1) / 2

/-- Counts the total number of black squares on the checkerboard -/
def total_black_squares (c : Checkerboard) : ℕ :=
  c.size * ((c.size + 1) / 2)

/-- Theorem stating that Rebecca's checkerboard has 435 black squares -/
theorem rebecca_checkerboard_black_squares :
  total_black_squares rebecca_checkerboard = 435 := by
  sorry


end rebecca_checkerboard_black_squares_l651_65130


namespace m_range_for_third_quadrant_l651_65129

/-- A complex number z is in the third quadrant if its real and imaginary parts are both negative -/
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

/-- The theorem stating that if z = (m+4) + (m-2)i is in the third quadrant, 
    then m is in the interval (-∞, -4) -/
theorem m_range_for_third_quadrant (m : ℝ) :
  let z : ℂ := Complex.mk (m + 4) (m - 2)
  in_third_quadrant z → m < -4 := by
  sorry

#check m_range_for_third_quadrant

end m_range_for_third_quadrant_l651_65129


namespace equal_remainders_divisor_l651_65128

theorem equal_remainders_divisor : ∃ (n : ℕ), n > 0 ∧ 
  n ∣ (2287 - 2028) ∧ 
  n ∣ (2028 - 1806) ∧ 
  n ∣ (2287 - 1806) ∧
  ∀ (m : ℕ), m > n → ¬(m ∣ (2287 - 2028) ∧ m ∣ (2028 - 1806) ∧ m ∣ (2287 - 1806)) :=
by sorry

end equal_remainders_divisor_l651_65128


namespace polynomial_root_implies_k_value_l651_65160

theorem polynomial_root_implies_k_value : ∀ k : ℚ,
  (3 : ℚ)^3 + k * 3 + 20 = 0 → k = -47/3 := by sorry

end polynomial_root_implies_k_value_l651_65160


namespace hallie_net_earnings_l651_65107

def hourly_rate : ℝ := 10

def monday_hours : ℝ := 7
def monday_tips : ℝ := 18

def tuesday_hours : ℝ := 5
def tuesday_tips : ℝ := 12

def wednesday_hours : ℝ := 7
def wednesday_tips : ℝ := 20

def thursday_hours : ℝ := 8
def thursday_tips : ℝ := 25

def friday_hours : ℝ := 6
def friday_tips : ℝ := 15

def discount_rate : ℝ := 0.05

def total_earnings : ℝ := 
  (monday_hours * hourly_rate + monday_tips) +
  (tuesday_hours * hourly_rate + tuesday_tips) +
  (wednesday_hours * hourly_rate + wednesday_tips) +
  (thursday_hours * hourly_rate + thursday_tips) +
  (friday_hours * hourly_rate + friday_tips)

def discount_amount : ℝ := total_earnings * discount_rate

def net_earnings : ℝ := total_earnings - discount_amount

theorem hallie_net_earnings : net_earnings = 399 := by
  sorry

end hallie_net_earnings_l651_65107


namespace string_average_length_l651_65115

theorem string_average_length : 
  let string1 : ℝ := 1.5
  let string2 : ℝ := 4.5
  let average := (string1 + string2) / 2
  average = 3 := by
sorry

end string_average_length_l651_65115


namespace complement_of_union_l651_65170

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end complement_of_union_l651_65170


namespace max_sides_convex_polygon_four_obtuse_l651_65103

/-- Represents a convex polygon with n sides and exactly four obtuse angles -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 0
  obtuse_angles : ℕ
  obtuse_count : obtuse_angles = 4

/-- The sum of interior angles of an n-sided polygon is (n-2) * 180 degrees -/
def interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

/-- An obtuse angle is greater than 90 degrees and less than 180 degrees -/
def is_obtuse (angle : ℝ) : Prop := 90 < angle ∧ angle < 180

/-- An acute angle is greater than 0 degrees and less than 90 degrees -/
def is_acute (angle : ℝ) : Prop := 0 < angle ∧ angle < 90

/-- The maximum number of sides for a convex polygon with exactly four obtuse angles is 7 -/
theorem max_sides_convex_polygon_four_obtuse :
  ∀ n : ℕ, ConvexPolygon n → n ≤ 7 :=
sorry

end max_sides_convex_polygon_four_obtuse_l651_65103


namespace chess_tournament_impossibility_l651_65156

theorem chess_tournament_impossibility (n : ℕ) (g : ℕ) (x : ℕ) : 
  n = 50 →  -- Total number of players
  g = 61 →  -- Total number of games played
  x ≤ n →   -- Number of players who played 3 games
  (3 * x + 2 * (n - x)) / 2 = g →  -- Total games calculation
  x * 3 > g →  -- Contradiction: games played by 3-game players exceed total games
  False :=
by
  sorry

end chess_tournament_impossibility_l651_65156


namespace number_problem_l651_65100

theorem number_problem (x : ℝ) : (0.95 * x - 12 = 178) → x = 200 := by
  sorry

end number_problem_l651_65100


namespace tangent_line_equation_l651_65155

-- Define the line l
def line_l (x y : ℝ) : Prop := y = (1/2) * x

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Define the line ST
def line_ST (x y : ℝ) : Prop := y = -2 * x + 11/2

-- Theorem statement
theorem tangent_line_equation 
  (h1 : line_l 2 1)
  (h2 : line_l 6 3)
  (h3 : ∃ (x₀ y₀ : ℝ), line_l x₀ y₀ ∧ circle_C x₀ y₀)
  (h4 : circle_C 2 0) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_ST x₁ y₁ ∧ line_ST x₂ y₂ ∧
    line_ST 6 3 :=
sorry

end tangent_line_equation_l651_65155


namespace set_equality_implies_a_value_l651_65166

theorem set_equality_implies_a_value (A B : Set ℤ) (a : ℤ) : 
  A = {2, 3} → B = {2, 2*a - 1} → A = B → a = 2 := by sorry

end set_equality_implies_a_value_l651_65166


namespace max_a_value_l651_65172

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) ∧ 
  (∃ x : ℝ, x < a ∧ x^2 - 2*x - 3 ≤ 0) →
  a ≤ 3 :=
by sorry

end max_a_value_l651_65172


namespace musical_chairs_theorem_l651_65162

/-- A function is a derangement if it has no fixed points -/
def IsDerangement {α : Type*} (f : α → α) : Prop :=
  ∀ x, f x ≠ x

/-- A positive integer is a prime power if it's of the form p^k where p is prime and k > 0 -/
def IsPrimePower (n : ℕ) : Prop :=
  ∃ (p k : ℕ), Prime p ∧ k > 0 ∧ n = p^k

theorem musical_chairs_theorem (n m : ℕ) 
    (h1 : m > 1) 
    (h2 : m ≤ n) 
    (h3 : ¬ IsPrimePower m) : 
    ∃ (f : Fin n → Fin n), 
      Function.Bijective f ∧ 
      IsDerangement f ∧ 
      (∀ x, (f^[m]) x = x) ∧ 
      (∀ (k : ℕ) (hk : k < m), ∃ x, (f^[k]) x ≠ x) := by
  sorry

end musical_chairs_theorem_l651_65162


namespace scholarship_fund_scientific_notation_l651_65126

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scholarship_fund_scientific_notation :
  toScientificNotation 445800000 = ScientificNotation.mk 4.458 8 (by norm_num) :=
sorry

end scholarship_fund_scientific_notation_l651_65126


namespace min_ratio_two_digit_integers_l651_65127

theorem min_ratio_two_digit_integers (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →  -- x is a two-digit positive integer
  (10 ≤ y ∧ y ≤ 99) →  -- y is a two-digit positive integer
  (x + y) / 2 = 75 →   -- mean of x and y is 75
  (∀ a b : ℕ, (10 ≤ a ∧ a ≤ 99) → (10 ≤ b ∧ b ≤ 99) → (a + b) / 2 = 75 → 
    x / (3 * y + 4 : ℚ) ≤ a / (3 * b + 4 : ℚ)) →
  x / (3 * y + 4 : ℚ) = 70 / 17 := by
sorry

end min_ratio_two_digit_integers_l651_65127


namespace inequality_solution_l651_65179

theorem inequality_solution (n k : ℤ) :
  let x : ℝ := (-1)^n * π/6 + 2*π*n
  let y : ℝ := π/2 + π*k
  4 * Real.sin x - Real.sqrt (Real.cos y) - Real.sqrt (Real.cos y - 16 * (Real.cos x)^2 + 12) ≥ 2 := by
  sorry

end inequality_solution_l651_65179


namespace transformation_matrix_correct_l651_65181

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def scaling_matrix (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![s, 0; 0, s]

/-- The transformation matrix M represents a 90° counter-clockwise rotation followed by a scaling of factor 3 -/
theorem transformation_matrix_correct :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -3; 3, 0]
  M = scaling_matrix 3 * rotation_matrix := by sorry

end transformation_matrix_correct_l651_65181


namespace mary_cut_ten_roses_l651_65194

/-- The number of roses Mary cut from her garden -/
def roses_cut : ℕ := 16 - 6

/-- Theorem stating that Mary cut 10 roses -/
theorem mary_cut_ten_roses : roses_cut = 10 := by
  sorry

end mary_cut_ten_roses_l651_65194


namespace fiftieth_term_is_247_l651_65136

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1 : ℤ) * d

theorem fiftieth_term_is_247 : 
  arithmetic_sequence 2 5 50 = 247 := by sorry

end fiftieth_term_is_247_l651_65136


namespace division_problem_l651_65147

theorem division_problem (x y : ℕ+) 
  (h1 : (x : ℝ) / (y : ℝ) = 96.12)
  (h2 : (x : ℝ) % (y : ℝ) = 1.44) : 
  y = 12 := by
  sorry

end division_problem_l651_65147


namespace root_implies_a_value_l651_65149

theorem root_implies_a_value (a : ℝ) : 
  ((3 : ℝ) = 3 ∧ (a - 2) / 3 - 1 / (3 - 2) = 0) → a = 5 := by
  sorry

end root_implies_a_value_l651_65149


namespace square_area_side_perimeter_l651_65196

theorem square_area_side_perimeter :
  ∀ (s p : ℝ),
  s > 0 →
  s^2 = 450 →
  p = 4 * s →
  s = 15 * Real.sqrt 2 ∧ p = 60 * Real.sqrt 2 :=
by sorry

end square_area_side_perimeter_l651_65196


namespace tan_equality_with_range_l651_65165

theorem tan_equality_with_range (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (850 * π / 180) → n = -50 := by
  sorry

end tan_equality_with_range_l651_65165


namespace triangle_angle_measure_l651_65195

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 3 * E →  -- Angle D is thrice as large as angle E
  E = 18 →     -- Angle E measures 18°
  D + E + F = 180 →  -- Sum of angles in a triangle is 180°
  F = 108 :=   -- Angle F measures 108°
by
  sorry

end triangle_angle_measure_l651_65195
