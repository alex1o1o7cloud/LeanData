import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l2951_295189

theorem simplify_expression (a : ℝ) : (-2 * a) * a - (-2 * a)^2 = -6 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2951_295189


namespace NUMINAMATH_CALUDE_coin_sum_theorem_l2951_295187

/-- Represents the possible coin values in cents -/
inductive Coin : Type
  | Penny : Coin
  | Nickel : Coin
  | Dime : Coin
  | Quarter : Coin
  | HalfDollar : Coin

/-- Returns the value of a coin in cents -/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- Checks if a given amount can be achieved using exactly six coins -/
def canAchieveWithSixCoins (amount : Nat) : Prop :=
  ∃ (c1 c2 c3 c4 c5 c6 : Coin), 
    coinValue c1 + coinValue c2 + coinValue c3 + coinValue c4 + coinValue c5 + coinValue c6 = amount

theorem coin_sum_theorem : 
  ¬ canAchieveWithSixCoins 62 ∧ 
  canAchieveWithSixCoins 80 ∧ 
  canAchieveWithSixCoins 90 ∧ 
  canAchieveWithSixCoins 96 := by
  sorry

end NUMINAMATH_CALUDE_coin_sum_theorem_l2951_295187


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2951_295156

theorem quadratic_minimum (a b : ℝ) (x₀ : ℝ) (h : a > 0) :
  (a * x₀ = b) ↔ ∀ x : ℝ, (1/2) * a * x^2 - b * x ≥ (1/2) * a * x₀^2 - b * x₀ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2951_295156


namespace NUMINAMATH_CALUDE_squash_league_max_salary_l2951_295199

/-- Represents the maximum salary a player can earn in a professional squash league --/
def max_salary (team_size : ℕ) (min_salary : ℕ) (total_payroll : ℕ) : ℕ :=
  total_payroll - (team_size - 1) * min_salary

/-- Theorem stating the maximum salary in the given conditions --/
theorem squash_league_max_salary :
  max_salary 22 16000 880000 = 544000 := by
  sorry

end NUMINAMATH_CALUDE_squash_league_max_salary_l2951_295199


namespace NUMINAMATH_CALUDE_min_students_for_three_discussing_same_l2951_295133

/-- Represents a discussion between two students about a problem -/
structure Discussion where
  student1 : ℕ
  student2 : ℕ
  problem : Fin 3

/-- Represents a valid discussion configuration for n students -/
def ValidConfiguration (n : ℕ) (discussions : List Discussion) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    ∃! d : Discussion, d ∈ discussions ∧
      ((d.student1 = i.val ∧ d.student2 = j.val) ∨
       (d.student1 = j.val ∧ d.student2 = i.val))

/-- Checks if there are at least 3 students discussing the same problem -/
def HasThreeDiscussingSame (n : ℕ) (discussions : List Discussion) : Prop :=
  ∃ p : Fin 3, ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (∃ d1 d2 d3 : Discussion,
      d1 ∈ discussions ∧ d2 ∈ discussions ∧ d3 ∈ discussions ∧
      d1.problem = p ∧ d2.problem = p ∧ d3.problem = p ∧
      ((d1.student1 = i.val ∧ d1.student2 = j.val) ∨ (d1.student1 = j.val ∧ d1.student2 = i.val)) ∧
      ((d2.student1 = j.val ∧ d2.student2 = k.val) ∨ (d2.student1 = k.val ∧ d2.student2 = j.val)) ∧
      ((d3.student1 = i.val ∧ d3.student2 = k.val) ∨ (d3.student1 = k.val ∧ d3.student2 = i.val)))

theorem min_students_for_three_discussing_same :
  (∃ n : ℕ, ∀ discussions : List Discussion,
    ValidConfiguration n discussions → HasThreeDiscussingSame n discussions) ∧
  (∀ m : ℕ, m < 17 →
    ∃ discussions : List Discussion,
      ValidConfiguration m discussions ∧ ¬HasThreeDiscussingSame m discussions) :=
by sorry

end NUMINAMATH_CALUDE_min_students_for_three_discussing_same_l2951_295133


namespace NUMINAMATH_CALUDE_five_trip_ticket_cost_l2951_295130

/-- Represents the cost of tickets in gold coins -/
structure TicketCost where
  one : ℕ
  five : ℕ
  twenty : ℕ

/-- Conditions for the ticket costs -/
def valid_ticket_cost (t : TicketCost) : Prop :=
  5 * t.one > t.five ∧ 
  4 * t.five > t.twenty ∧
  t.twenty + 3 * t.five = 33 ∧
  20 + 3 * 5 = 35

theorem five_trip_ticket_cost (t : TicketCost) (h : valid_ticket_cost t) : t.five = 5 :=
sorry

end NUMINAMATH_CALUDE_five_trip_ticket_cost_l2951_295130


namespace NUMINAMATH_CALUDE_last_digit_is_zero_l2951_295174

def number (last_digit : Nat) : Nat :=
  626840 + last_digit

theorem last_digit_is_zero :
  ∀ d : Nat, d < 10 →
  (number d % 8 = 0 ∧ number d % 5 = 0) →
  d = 0 := by
sorry

end NUMINAMATH_CALUDE_last_digit_is_zero_l2951_295174


namespace NUMINAMATH_CALUDE_unique_arith_seq_pair_a_eq_one_third_l2951_295118

/-- Two arithmetic sequences satisfying given conditions -/
structure ArithSeqPair where
  a : ℝ
  q : ℝ
  h_a_pos : a > 0
  h_b1_a1 : (a + 1) - a = 1
  h_b2_a2 : (a + q + 2) - (a + q) = 2
  h_b3_a3 : (a + 2*q + 3) - (a + 2*q) = 3
  h_unique : ∃! q, (a * q^2 - 4 * a * q + 3 * a - 1 = 0) ∧ q ≠ 0

/-- If two arithmetic sequences satisfy the given conditions and one is unique, then a = 1/3 -/
theorem unique_arith_seq_pair_a_eq_one_third (p : ArithSeqPair) : p.a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_arith_seq_pair_a_eq_one_third_l2951_295118


namespace NUMINAMATH_CALUDE_percentage_calculation_l2951_295163

theorem percentage_calculation (whole : ℝ) (part : ℝ) (h1 : whole = 200) (h2 : part = 50) :
  (part / whole) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2951_295163


namespace NUMINAMATH_CALUDE_peaches_bought_is_seven_l2951_295106

/-- Represents the cost of fruits and the quantity purchased. -/
structure FruitPurchase where
  apple_cost : ℕ
  peach_cost : ℕ
  total_fruits : ℕ
  total_cost : ℕ

/-- Calculates the number of peaches bought given a FruitPurchase. -/
def peaches_bought (purchase : FruitPurchase) : ℕ :=
  let apple_count := purchase.total_fruits - (purchase.total_cost - purchase.apple_cost * purchase.total_fruits) / (purchase.peach_cost - purchase.apple_cost)
  purchase.total_fruits - apple_count

/-- Theorem stating that given the specific conditions, 7 peaches were bought. -/
theorem peaches_bought_is_seven : 
  ∀ (purchase : FruitPurchase), 
    purchase.apple_cost = 1000 → 
    purchase.peach_cost = 2000 → 
    purchase.total_fruits = 15 → 
    purchase.total_cost = 22000 → 
    peaches_bought purchase = 7 := by
  sorry


end NUMINAMATH_CALUDE_peaches_bought_is_seven_l2951_295106


namespace NUMINAMATH_CALUDE_writers_birth_months_l2951_295102

/-- The total number of famous writers -/
def total_writers : ℕ := 200

/-- The number of writers born in October -/
def october_births : ℕ := 15

/-- The number of writers born in July -/
def july_births : ℕ := 14

/-- The percentage of writers born in October -/
def october_percentage : ℚ := (october_births : ℚ) / (total_writers : ℚ) * 100

/-- The percentage of writers born in July -/
def july_percentage : ℚ := (july_births : ℚ) / (total_writers : ℚ) * 100

theorem writers_birth_months :
  october_percentage = 15/2 ∧
  july_percentage = 7 ∧
  october_percentage > july_percentage :=
by sorry

end NUMINAMATH_CALUDE_writers_birth_months_l2951_295102


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_315_l2951_295120

/-- The sum of the digits in the binary representation of 315 is 6. -/
theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_315_l2951_295120


namespace NUMINAMATH_CALUDE_president_vice_president_election_committee_members_election_l2951_295124

-- Define the number of candidates
def num_candidates : ℕ := 4

-- Define the number of positions for the first question (president and vice president)
def num_positions_1 : ℕ := 2

-- Define the number of positions for the second question (committee members)
def num_positions_2 : ℕ := 3

-- Theorem for the first question
theorem president_vice_president_election :
  (num_candidates.choose num_positions_1) * num_positions_1.factorial = 12 := by
  sorry

-- Theorem for the second question
theorem committee_members_election :
  num_candidates.choose num_positions_2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_election_committee_members_election_l2951_295124


namespace NUMINAMATH_CALUDE_living_room_curtain_length_l2951_295154

/-- Given the dimensions of a bolt of fabric, bedroom curtain, and living room curtain width,
    as well as the remaining fabric area, prove the length of the living room curtain. -/
theorem living_room_curtain_length
  (bolt_width : ℝ)
  (bolt_length : ℝ)
  (bedroom_width : ℝ)
  (bedroom_length : ℝ)
  (living_room_width : ℝ)
  (remaining_area : ℝ)
  (h1 : bolt_width = 16)
  (h2 : bolt_length = 12)
  (h3 : bedroom_width = 2)
  (h4 : bedroom_length = 4)
  (h5 : living_room_width = 4)
  (h6 : remaining_area = 160)
  (h7 : bolt_width * bolt_length - (bedroom_width * bedroom_length + living_room_width * living_room_length) = remaining_area) :
  living_room_length = 6 :=
by sorry

#check living_room_curtain_length

end NUMINAMATH_CALUDE_living_room_curtain_length_l2951_295154


namespace NUMINAMATH_CALUDE_cos_sum_when_sin_product_one_l2951_295131

theorem cos_sum_when_sin_product_one (α β : Real) 
  (h : Real.sin α * Real.sin β = 1) : 
  Real.cos (α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_when_sin_product_one_l2951_295131


namespace NUMINAMATH_CALUDE_total_donation_equals_854_l2951_295152

/-- Represents a fundraising event with earnings and donation percentage -/
structure FundraisingEvent where
  earnings : ℝ
  donationPercentage : ℝ

/-- Calculates the donation amount for a fundraising event -/
def donationAmount (event : FundraisingEvent) : ℝ :=
  event.earnings * event.donationPercentage

/-- Theorem: The total donation from five fundraising events equals $854 -/
theorem total_donation_equals_854 
  (carWash : FundraisingEvent)
  (bakeSale : FundraisingEvent)
  (mowingLawns : FundraisingEvent)
  (handmadeCrafts : FundraisingEvent)
  (charityConcert : FundraisingEvent)
  (h1 : carWash.earnings = 200 ∧ carWash.donationPercentage = 0.9)
  (h2 : bakeSale.earnings = 160 ∧ bakeSale.donationPercentage = 0.8)
  (h3 : mowingLawns.earnings = 120 ∧ mowingLawns.donationPercentage = 1)
  (h4 : handmadeCrafts.earnings = 180 ∧ handmadeCrafts.donationPercentage = 0.7)
  (h5 : charityConcert.earnings = 500 ∧ charityConcert.donationPercentage = 0.6)
  : donationAmount carWash + donationAmount bakeSale + donationAmount mowingLawns + 
    donationAmount handmadeCrafts + donationAmount charityConcert = 854 := by
  sorry


end NUMINAMATH_CALUDE_total_donation_equals_854_l2951_295152


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2951_295158

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2951_295158


namespace NUMINAMATH_CALUDE_exists_always_last_card_l2951_295110

/-- Represents a card with a unique natural number -/
structure Card where
  number : ℕ
  unique : ℕ

/-- Represents the circular arrangement of cards -/
def CardArrangement := Vector Card 1000

/-- Simulates the card removal process -/
def removeCards (arrangement : CardArrangement) (startIndex : Fin 1000) : Card :=
  sorry

/-- Checks if a card is the last remaining for all starting positions except its own -/
def isAlwaysLast (arrangement : CardArrangement) (cardIndex : Fin 1000) : Prop :=
  ∀ i : Fin 1000, i ≠ cardIndex → removeCards arrangement i = arrangement.get cardIndex

/-- Main theorem: There exists a card arrangement where one card is always the last remaining -/
theorem exists_always_last_card : ∃ (arrangement : CardArrangement), ∃ (i : Fin 1000), isAlwaysLast arrangement i :=
  sorry

end NUMINAMATH_CALUDE_exists_always_last_card_l2951_295110


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2951_295173

theorem necessary_and_sufficient_condition : 
  (∀ x : ℝ, x = 1 ↔ x^2 - 2*x + 1 = 0) := by sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2951_295173


namespace NUMINAMATH_CALUDE_min_value_theorem_l2951_295192

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (a + 1) + 4 / (b + 1) ≥ 9 / 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / (a₀ + 1) + 4 / (b₀ + 1) = 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2951_295192


namespace NUMINAMATH_CALUDE_exam_students_count_l2951_295143

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
    N > 0 →
    T / N = 80 →
    (T - 100) / (N - 5) = 90 →
    N = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l2951_295143


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l2951_295182

/-- Represents a parabola with vertex at the origin -/
structure Parabola where
  c : ℝ
  eq : ℝ × ℝ → Prop := fun (x, y) ↦ y^2 = 4 * c * x

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 / a^2 - y^2 / b^2 = 1

/-- The main theorem -/
theorem parabola_hyperbola_equations 
  (p : Parabola) 
  (h : Hyperbola) 
  (h_a_pos : h.a > 0)
  (h_b_pos : h.b > 0)
  (directrix_passes_focus : ∃ (f : ℝ × ℝ), h.eq f ∧ p.c = 2 * h.a)
  (intersection_point : p.eq (3/2, Real.sqrt 6) ∧ h.eq (3/2, Real.sqrt 6)) :
  p.eq = fun (x, y) ↦ y^2 = 4 * x ∧ 
  h.eq = fun (x, y) ↦ 4 * x^2 - 4 * y^2 / 3 = 1 := by
  sorry


end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l2951_295182


namespace NUMINAMATH_CALUDE_general_inequality_l2951_295115

theorem general_inequality (x n : ℝ) (h1 : x > 0) (h2 : n > 0) 
  (h3 : ∃ (a : ℝ), a > 0 ∧ x + a / x^n ≥ n + 1) :
  ∃ (a : ℝ), a = n^n ∧ x + a / x^n ≥ n + 1 :=
sorry

end NUMINAMATH_CALUDE_general_inequality_l2951_295115


namespace NUMINAMATH_CALUDE_cos_negative_twentythree_fourths_pi_l2951_295171

theorem cos_negative_twentythree_fourths_pi :
  Real.cos (-23 / 4 * Real.pi) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_twentythree_fourths_pi_l2951_295171


namespace NUMINAMATH_CALUDE_max_d_value_l2951_295136

theorem max_d_value : 
  let f : ℝ → ℝ := λ d => (5 + Real.sqrt 244) / 3 - d
  ∃ d : ℝ, (4 * Real.sqrt 3) ^ 2 + (d + 5) ^ 2 = (2 * d) ^ 2 ∧ 
    (∀ x : ℝ, (4 * Real.sqrt 3) ^ 2 + (x + 5) ^ 2 = (2 * x) ^ 2 → f x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l2951_295136


namespace NUMINAMATH_CALUDE_cube_root_fourteen_problem_l2951_295193

theorem cube_root_fourteen_problem (x y z : ℝ) 
  (eq1 : (x + y) / (1 + z) = (1 - z + z^2) / (x^2 - x*y + y^2))
  (eq2 : (x - y) / (3 - z) = (9 + 3*z + z^2) / (x^2 + x*y + y^2)) :
  x = (14 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_fourteen_problem_l2951_295193


namespace NUMINAMATH_CALUDE_limit_rational_function_l2951_295113

theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 3| ∧ |x - 3| < δ → 
    |((x^6 - 54*x^3 + 729) / (x^3 - 27)) - 0| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_rational_function_l2951_295113


namespace NUMINAMATH_CALUDE_line_slope_45_degrees_l2951_295190

/-- Given a line passing through points P(-2, m) and Q(m, 4) with a slope angle of 45°, the value of m is 1. -/
theorem line_slope_45_degrees (m : ℝ) : 
  (∃ (P Q : ℝ × ℝ), 
    P = (-2, m) ∧ 
    Q = (m, 4) ∧ 
    (Q.2 - P.2) / (Q.1 - P.1) = Real.tan (π / 4)) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_45_degrees_l2951_295190


namespace NUMINAMATH_CALUDE_marys_height_marys_final_height_l2951_295132

theorem marys_height (initial_height : ℝ) (sallys_new_height : ℝ) : ℝ :=
  let sallys_growth_factor : ℝ := 1.2
  let sallys_growth : ℝ := sallys_new_height - initial_height
  let marys_growth : ℝ := sallys_growth / 2
  initial_height + marys_growth

theorem marys_final_height : 
  ∀ (initial_height : ℝ),
    initial_height > 0 →
    marys_height initial_height 180 = 165 :=
by
  sorry

end NUMINAMATH_CALUDE_marys_height_marys_final_height_l2951_295132


namespace NUMINAMATH_CALUDE_paving_cost_l2951_295169

/-- The cost of paving a rectangular floor given its dimensions and the rate per square metre. -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 8) (h2 : width = 4.75) (h3 : rate = 900) :
  length * width * rate = 34200 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l2951_295169


namespace NUMINAMATH_CALUDE_equation_solution_l2951_295141

theorem equation_solution (x y : ℕ) : 
  (x^2 + 1)^y - (x^2 - 1)^y = 2*x^y ↔ 
  (x = 1 ∧ y = 1) ∨ (x = 0 ∧ ∃ k : ℕ, y = 2*k + 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2951_295141


namespace NUMINAMATH_CALUDE_tan_75_deg_l2951_295186

/-- Tangent of angle addition formula -/
axiom tan_add (a b : ℝ) : Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)

/-- Proof that tan 75° = 2 + √3 -/
theorem tan_75_deg : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  have h1 : 75 * π / 180 = 60 * π / 180 + 15 * π / 180 := by sorry
  have h2 : Real.tan (60 * π / 180) = Real.sqrt 3 := by sorry
  have h3 : Real.tan (15 * π / 180) = 2 - Real.sqrt 3 := by sorry
  sorry


end NUMINAMATH_CALUDE_tan_75_deg_l2951_295186


namespace NUMINAMATH_CALUDE_hexagon_area_difference_l2951_295125

/-- The area between a regular hexagon with side length 8 and a smaller hexagon
    formed by joining the midpoints of its sides is 72√3. -/
theorem hexagon_area_difference : 
  let s : ℝ := 8
  let area_large := (3 * Real.sqrt 3 / 2) * s^2
  let area_small := (3 * Real.sqrt 3 / 2) * (s/2)^2
  area_large - area_small = 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_difference_l2951_295125


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_coefficient_x_squared_in_expansion_proof_l2951_295179

/-- The coefficient of x^2 in the expansion of (x + 2/x^2)^5 is 10 -/
theorem coefficient_x_squared_in_expansion : ℕ :=
  let expansion := (fun x => (x + 2 / x^2)^5)
  let coefficient_x_squared := 10
  coefficient_x_squared

/-- Proof of the theorem -/
theorem coefficient_x_squared_in_expansion_proof :
  coefficient_x_squared_in_expansion = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_coefficient_x_squared_in_expansion_proof_l2951_295179


namespace NUMINAMATH_CALUDE_nina_unanswered_questions_l2951_295139

/-- Represents the scoring details for a math test -/
structure ScoringSystem where
  initialPoints : ℕ
  correctPoints : ℕ
  wrongPoints : ℤ
  unansweredPoints : ℕ

/-- Represents the test results -/
structure TestResult where
  totalQuestions : ℕ
  score : ℕ

theorem nina_unanswered_questions
  (oldSystem : ScoringSystem)
  (newSystem : ScoringSystem)
  (oldResult : TestResult)
  (newResult : TestResult)
  (h1 : oldSystem = {
    initialPoints := 40,
    correctPoints := 5,
    wrongPoints := -2,
    unansweredPoints := 0
  })
  (h2 : newSystem = {
    initialPoints := 0,
    correctPoints := 6,
    wrongPoints := 0,
    unansweredPoints := 3
  })
  (h3 : oldResult = {totalQuestions := 35, score := 95})
  (h4 : newResult = {totalQuestions := 35, score := 120})
  (h5 : oldResult.totalQuestions = newResult.totalQuestions) :
  ∃ (correct wrong unanswered : ℕ),
    correct + wrong + unanswered = oldResult.totalQuestions ∧
    oldSystem.initialPoints + oldSystem.correctPoints * correct + oldSystem.wrongPoints * wrong = oldResult.score ∧
    newSystem.correctPoints * correct + newSystem.unansweredPoints * unanswered = newResult.score ∧
    unanswered = 10 :=
by sorry

end NUMINAMATH_CALUDE_nina_unanswered_questions_l2951_295139


namespace NUMINAMATH_CALUDE_permutation_calculation_l2951_295135

-- Define the permutation function
def A (n : ℕ) (r : ℕ) : ℚ :=
  if r ≤ n then (Nat.factorial n) / (Nat.factorial (n - r)) else 0

-- State the theorem
theorem permutation_calculation :
  (4 * A 8 4 + 2 * A 8 5) / (A 8 6 - A 9 5) * Nat.factorial 0 = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_permutation_calculation_l2951_295135


namespace NUMINAMATH_CALUDE_square_polynomial_l2951_295140

theorem square_polynomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + k*x + 16 = (a*x + b)^2) → (k = 8 ∨ k = -8) := by
  sorry

end NUMINAMATH_CALUDE_square_polynomial_l2951_295140


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l2951_295166

theorem sphere_surface_area_circumscribing_cube (edge_length : ℝ) (sphere_radius : ℝ) :
  edge_length = 2 →
  sphere_radius = edge_length * Real.sqrt 3 / 2 →
  4 * Real.pi * sphere_radius^2 = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l2951_295166


namespace NUMINAMATH_CALUDE_max_trucks_orchard_l2951_295150

def apples : ℕ := 170
def tangerines : ℕ := 268
def mangoes : ℕ := 120

def apples_leftover : ℕ := 8
def tangerines_short : ℕ := 2
def mangoes_leftover : ℕ := 12

theorem max_trucks_orchard : 
  let apples_distributed := apples - apples_leftover
  let tangerines_distributed := tangerines + tangerines_short
  let mangoes_distributed := mangoes - mangoes_leftover
  ∃ (n : ℕ), n > 0 ∧ 
    apples_distributed % n = 0 ∧ 
    tangerines_distributed % n = 0 ∧ 
    mangoes_distributed % n = 0 ∧
    ∀ (m : ℕ), m > n → 
      (apples_distributed % m = 0 ∧ 
       tangerines_distributed % m = 0 ∧ 
       mangoes_distributed % m = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_max_trucks_orchard_l2951_295150


namespace NUMINAMATH_CALUDE_function_inequality_l2951_295177

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * deriv f x > -f x) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2951_295177


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l2951_295162

def quadratic_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def is_root (f : ℝ → ℝ) (r : ℝ) : Prop := f r = 0

theorem correct_quadratic_equation :
  ∃ (a b c : ℝ),
    (∃ (b₁ c₁ : ℝ), is_root (quadratic_equation a b₁ c₁) 5 ∧ is_root (quadratic_equation a b₁ c₁) 3) ∧
    (∃ (b₂ : ℝ), is_root (quadratic_equation a b₂ c) (-6) ∧ is_root (quadratic_equation a b₂ c) (-4)) ∧
    quadratic_equation a b c = quadratic_equation 1 (-8) 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l2951_295162


namespace NUMINAMATH_CALUDE_divisors_of_60_and_90_l2951_295188

theorem divisors_of_60_and_90 : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ 60 % n = 0 ∧ 90 % n = 0) ∧ 
  (∀ n : ℕ, n > 0 → 60 % n = 0 → 90 % n = 0 → n ∈ S) ∧
  Finset.card S = 8 := by
sorry

end NUMINAMATH_CALUDE_divisors_of_60_and_90_l2951_295188


namespace NUMINAMATH_CALUDE_perfectville_run_difference_l2951_295101

theorem perfectville_run_difference (street_width : ℕ) (block_side : ℕ) : 
  street_width = 30 → block_side = 500 → 
  4 * (block_side + 2 * street_width) - 4 * block_side = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_perfectville_run_difference_l2951_295101


namespace NUMINAMATH_CALUDE_cost_of_three_batches_l2951_295159

/-- Represents the cost and quantity of ingredients for yogurt production -/
structure YogurtProduction where
  milk_price : ℝ
  fruit_price : ℝ
  milk_per_batch : ℝ
  fruit_per_batch : ℝ

/-- Calculates the cost of producing a given number of yogurt batches -/
def cost_of_batches (y : YogurtProduction) (num_batches : ℝ) : ℝ :=
  num_batches * (y.milk_price * y.milk_per_batch + y.fruit_price * y.fruit_per_batch)

/-- Theorem: The cost of producing three batches of yogurt is $63 -/
theorem cost_of_three_batches :
  ∃ (y : YogurtProduction),
    y.milk_price = 1.5 ∧
    y.fruit_price = 2 ∧
    y.milk_per_batch = 10 ∧
    y.fruit_per_batch = 3 ∧
    cost_of_batches y 3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_three_batches_l2951_295159


namespace NUMINAMATH_CALUDE_intersection_point_l2951_295184

-- Define the rectangle ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (6, 4)
def D : ℝ × ℝ := (6, 0)

-- Define the lines from A and B
def lineA1 (x : ℝ) : ℝ := x -- y = x (45° line from A)
def lineA2 (x : ℝ) : ℝ := -x -- y = -x (135° line from A)
def lineB1 (x : ℝ) : ℝ := 4 - x -- y = 4 - x (-45° line from B)
def lineB2 (x : ℝ) : ℝ := 4 + x -- y = 4 + x (-135° line from B)

-- Theorem statement
theorem intersection_point : 
  ∃! p : ℝ × ℝ, 
    (lineA1 p.1 = p.2 ∧ lineB1 p.1 = p.2) ∧ 
    (lineA2 p.1 = p.2 ∧ lineB2 p.1 = p.2) ∧
    p = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2951_295184


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2951_295142

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2951_295142


namespace NUMINAMATH_CALUDE_f_8_equals_952_l2951_295164

def f (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 27*x^2 - 24*x - 72

theorem f_8_equals_952 : f 8 = 952 := by
  sorry

end NUMINAMATH_CALUDE_f_8_equals_952_l2951_295164


namespace NUMINAMATH_CALUDE_fraction_multiplication_division_l2951_295194

theorem fraction_multiplication_division (a b c d e f g h : ℚ) 
  (h1 : c = 663 / 245)
  (h2 : f = 328 / 15) :
  a / b * c / d / f = g / h :=
by
  sorry

#check fraction_multiplication_division (145 : ℚ) (273 : ℚ) (663 / 245 : ℚ) (1 : ℚ) (1 : ℚ) (328 / 15 : ℚ) (7395 : ℚ) (112504 : ℚ)

end NUMINAMATH_CALUDE_fraction_multiplication_division_l2951_295194


namespace NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l2951_295103

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem partition_contains_perfect_square_sum (n : ℕ) : 
  (n ≥ 15) ↔ 
  (∀ (A B : Set ℕ), 
    (A ∪ B = Finset.range n.succ) → 
    (A ∩ B = ∅) → 
    ((∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ is_perfect_square (x + y)) ∨
     (∃ (x y : ℕ), x ∈ B ∧ y ∈ B ∧ x ≠ y ∧ is_perfect_square (x + y)))) :=
by sorry

end NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l2951_295103


namespace NUMINAMATH_CALUDE_sqrt_calculation_l2951_295121

theorem sqrt_calculation : Real.sqrt 6 * Real.sqrt 3 + Real.sqrt 24 / Real.sqrt 6 - |(-3) * Real.sqrt 2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l2951_295121


namespace NUMINAMATH_CALUDE_distance_covered_72min_10kmph_l2951_295107

/-- The distance covered by a man walking at a given speed for a given time. -/
def distanceCovered (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A man walking for 72 minutes at a speed of 10 km/hr covers a distance of 12 km. -/
theorem distance_covered_72min_10kmph :
  let speed : ℝ := 10  -- Speed in km/hr
  let time : ℝ := 72 / 60  -- Time in hours (72 minutes converted to hours)
  distanceCovered speed time = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_covered_72min_10kmph_l2951_295107


namespace NUMINAMATH_CALUDE_eight_equidistant_points_l2951_295116

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this problem

/-- A point in 3D space -/
structure Point3D where
  -- We don't need to define the specifics of a point for this problem

/-- Distance between a point and a plane -/
def distance (p : Point3D) (plane : Plane3D) : ℝ :=
  sorry -- Actual implementation not needed for this statement

/-- The set of points at a given distance from a plane -/
def pointsAtDistance (plane : Plane3D) (d : ℝ) : Set Point3D :=
  {p : Point3D | distance p plane = d}

/-- The theorem stating that there are exactly 8 points at given distances from three planes -/
theorem eight_equidistant_points (plane1 plane2 plane3 : Plane3D) (m n p : ℝ) :
  ∃! (points : Finset Point3D),
    points.card = 8 ∧
    ∀ point ∈ points,
      distance point plane1 = m ∧
      distance point plane2 = n ∧
      distance point plane3 = p :=
  sorry


end NUMINAMATH_CALUDE_eight_equidistant_points_l2951_295116


namespace NUMINAMATH_CALUDE_inequality_solution_l2951_295197

theorem inequality_solution (x : ℝ) : 
  (2*x - 1)/(x^2 + 2) > 5/x + 21/10 ↔ -5 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2951_295197


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l2951_295126

theorem no_real_roots_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l2951_295126


namespace NUMINAMATH_CALUDE_apples_per_basket_l2951_295180

theorem apples_per_basket (total_apples : ℕ) (num_baskets : ℕ) 
  (h1 : total_apples = 495) 
  (h2 : num_baskets = 19) 
  (h3 : total_apples % num_baskets = 0) : 
  total_apples / num_baskets = 26 := by
sorry

end NUMINAMATH_CALUDE_apples_per_basket_l2951_295180


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2951_295185

theorem complex_expression_evaluation :
  (1 : ℝ) * (0.25 ^ (1/2 : ℝ)) - 
  (-2 * ((3/7 : ℝ) ^ (0 : ℝ))) ^ 2 * 
  ((-2 : ℝ) ^ 3) ^ (4/3 : ℝ) + 
  ((2 : ℝ) ^ (1/2 : ℝ) - 1) ^ (-1 : ℝ) - 
  (2 : ℝ) ^ (1/2 : ℝ) = -125/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2951_295185


namespace NUMINAMATH_CALUDE_reinforcement_arrival_day_l2951_295165

/-- Calculates the number of days passed before reinforcement arrived -/
def days_before_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
  (reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  ((initial_garrison * initial_duration) - 
   ((initial_garrison + reinforcement) * remaining_duration)) / initial_garrison

/-- Theorem stating the number of days passed before reinforcement arrived -/
theorem reinforcement_arrival_day : 
  days_before_reinforcement 2000 54 1600 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_arrival_day_l2951_295165


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2951_295134

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.exp x}

-- Theorem statement
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2951_295134


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l2951_295100

-- Define the trapezoid and its properties
structure Trapezoid :=
  (AB CD : ℝ)
  (area_ratio : ℝ)
  (sum_parallel_sides : ℝ)
  (h_positive : AB > 0)
  (h_area_ratio : area_ratio = 5 / 3)
  (h_sum : AB + CD = sum_parallel_sides)

-- Theorem statement
theorem trapezoid_segment_length (t : Trapezoid) (h : t.sum_parallel_sides = 160) :
  t.AB = 100 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l2951_295100


namespace NUMINAMATH_CALUDE_constant_term_in_expansion_l2951_295138

theorem constant_term_in_expansion (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (x + a / x) * (2 * x - 1 / x)^5 = 2) →
  ∃ coeffs : List ℝ, 
    (∀ x : ℝ, x ≠ 0 → (x + a / x) * (2 * x - 1 / x)^5 = coeffs.sum) ∧
    coeffs.sum = 2 ∧
    (∃ const_term : ℝ, const_term = 40 ∧ 
      ∀ x : ℝ, x ≠ 0 → (x + a / x) * (2 * x - 1 / x)^5 = 
        const_term + x * (coeffs.sum - const_term - a / x) + 
        1 / x * (coeffs.sum - const_term - a * x)) :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_in_expansion_l2951_295138


namespace NUMINAMATH_CALUDE_problem_solution_l2951_295146

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x * log x + a * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := |exp x - a| + a^2 / 2

theorem problem_solution (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (exp 1), Monotone (f a)) →
  (∃ M m : ℝ, (∀ x ∈ Set.Icc 0 (log 3), m ≤ g a x ∧ g a x ≤ M) ∧ M - m = 3/2) →
  a = 5/2 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2951_295146


namespace NUMINAMATH_CALUDE_ten_tables_seating_l2951_295112

/-- Calculates the number of people that can be seated at a given number of tables arranged in a row -/
def seatsInRow (numTables : ℕ) : ℕ :=
  if numTables = 0 then 0
  else if numTables = 1 then 6
  else if numTables = 2 then 10
  else if numTables = 3 then 14
  else 4 * numTables + 2

/-- Calculates the number of people that can be seated in a rectangular arrangement of tables -/
def seatsInRectangle (rows : ℕ) (tablesPerRow : ℕ) : ℕ :=
  rows * seatsInRow tablesPerRow

theorem ten_tables_seating :
  seatsInRectangle 2 5 = 80 :=
by sorry

end NUMINAMATH_CALUDE_ten_tables_seating_l2951_295112


namespace NUMINAMATH_CALUDE_equation_a_correct_equation_b_correct_equation_c_correct_equation_d_incorrect_l2951_295168

/-- Represents the principal amount in yuan -/
def principal : ℝ := sorry

/-- The annual interest rate -/
def interest_rate : ℝ := 0.03

/-- The total amount withdrawn after one year -/
def total_amount : ℝ := 20600

/-- Theorem stating that equation A is correct -/
theorem equation_a_correct : principal + interest_rate * principal = total_amount := by sorry

/-- Theorem stating that equation B is correct -/
theorem equation_b_correct : interest_rate * principal = total_amount - principal := by sorry

/-- Theorem stating that equation C is correct -/
theorem equation_c_correct : principal - total_amount = -(interest_rate * principal) := by sorry

/-- Theorem stating that equation D is incorrect -/
theorem equation_d_incorrect : principal + interest_rate ≠ total_amount := by sorry

end NUMINAMATH_CALUDE_equation_a_correct_equation_b_correct_equation_c_correct_equation_d_incorrect_l2951_295168


namespace NUMINAMATH_CALUDE_max_a_value_l2951_295183

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- State the theorem
theorem max_a_value (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 2, f a x ≤ 6) →
  a ≤ -1 ∧ ∃ x ∈ Set.Ioo 0 2, f (-1) x = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l2951_295183


namespace NUMINAMATH_CALUDE_test_marks_theorem_l2951_295155

/-- Represents a test section with a number of questions and a success rate -/
structure TestSection where
  questions : ℕ
  successRate : ℚ
  
/-- Calculates the total marks for a given test -/
def calculateTotalMarks (sections : List TestSection) : ℚ :=
  let correctAnswers := sections.map (fun s => (s.questions : ℚ) * s.successRate)
  let totalCorrect := correctAnswers.sum
  let totalQuestions := (sections.map (fun s => s.questions)).sum
  let incorrectAnswers := totalQuestions - totalCorrect.floor
  totalCorrect.floor - 0.25 * incorrectAnswers

/-- The theorem states that given the specific test conditions, the total marks obtained is 115 -/
theorem test_marks_theorem :
  let sections := [
    { questions := 50, successRate := 85/100 },
    { questions := 60, successRate := 70/100 },
    { questions := 40, successRate := 95/100 }
  ]
  calculateTotalMarks sections = 115 := by
  sorry

end NUMINAMATH_CALUDE_test_marks_theorem_l2951_295155


namespace NUMINAMATH_CALUDE_cricket_bat_theorem_l2951_295144

def cricket_bat_problem (a_cost_price b_selling_price c_purchase_price : ℝ) 
  (a_profit_percentage : ℝ) : Prop :=
  let a_selling_price := a_cost_price * (1 + a_profit_percentage)
  let b_profit := c_purchase_price - a_selling_price
  let b_profit_percentage := b_profit / a_selling_price * 100
  a_cost_price = 156 ∧ 
  a_profit_percentage = 0.20 ∧ 
  c_purchase_price = 234 → 
  b_profit_percentage = 25

theorem cricket_bat_theorem : 
  ∃ (a_cost_price b_selling_price c_purchase_price : ℝ),
    cricket_bat_problem a_cost_price b_selling_price c_purchase_price 0.20 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_theorem_l2951_295144


namespace NUMINAMATH_CALUDE_kombucha_half_fill_time_l2951_295149

/-- Represents the area of kombucha in the jar as a fraction of the full jar -/
def kombucha_area (days : ℕ) : ℚ :=
  1 / 2^(19 - days)

theorem kombucha_half_fill_time : 
  (∀ d : ℕ, d < 19 → kombucha_area (d + 1) = 2 * kombucha_area d) →
  kombucha_area 19 = 1 →
  kombucha_area 18 = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_kombucha_half_fill_time_l2951_295149


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2951_295123

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2951_295123


namespace NUMINAMATH_CALUDE_colors_in_box_l2951_295178

/-- The number of color boxes -/
def num_boxes : ℕ := 3

/-- The total number of pencils -/
def total_pencils : ℕ := 21

/-- The number of colors in each box -/
def colors_per_box : ℕ := total_pencils / num_boxes

/-- Theorem stating that the number of colors in each box is 7 -/
theorem colors_in_box : colors_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_colors_in_box_l2951_295178


namespace NUMINAMATH_CALUDE_sum_of_leading_digits_of_roots_l2951_295111

/-- A function that returns the leading digit of a positive real number -/
def leadingDigit (x : ℝ) : ℕ :=
  sorry

/-- The number M, which is a 303-digit number consisting only of 5s -/
def M : ℕ := sorry

/-- The function g that returns the leading digit of the r-th root of M -/
def g (r : ℕ) : ℕ :=
  leadingDigit (M ^ (1 / r : ℝ))

/-- Theorem stating that the sum of g(2) to g(6) is 10 -/
theorem sum_of_leading_digits_of_roots :
  g 2 + g 3 + g 4 + g 5 + g 6 = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_leading_digits_of_roots_l2951_295111


namespace NUMINAMATH_CALUDE_area_of_region_is_4pi_l2951_295108

-- Define the region
def region (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y = -1

-- Define the area of the region
noncomputable def area_of_region : ℝ := sorry

-- Theorem statement
theorem area_of_region_is_4pi :
  area_of_region = 4 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_area_of_region_is_4pi_l2951_295108


namespace NUMINAMATH_CALUDE_twentieth_decimal_of_35_36_l2951_295109

/-- The fraction we're considering -/
def f : ℚ := 35 / 36

/-- The nth decimal digit in the decimal expansion of a rational number -/
noncomputable def nthDecimalDigit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 20th decimal digit of 35/36 is 2 -/
theorem twentieth_decimal_of_35_36 : nthDecimalDigit f 20 = 2 := by sorry

end NUMINAMATH_CALUDE_twentieth_decimal_of_35_36_l2951_295109


namespace NUMINAMATH_CALUDE_restaurant_bill_fraction_l2951_295147

theorem restaurant_bill_fraction (akshitha veena lasya total : ℚ) : 
  akshitha = (3 / 4) * veena →
  veena = (1 / 2) * lasya →
  total = akshitha + veena + lasya →
  veena / total = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_fraction_l2951_295147


namespace NUMINAMATH_CALUDE_max_circular_triples_14_players_l2951_295104

/-- Represents a round-robin tournament --/
structure Tournament :=
  (num_players : ℕ)
  (games_per_player : ℕ)
  (no_draws : Bool)

/-- Calculates the maximum number of circular triples in a tournament --/
def max_circular_triples (t : Tournament) : ℕ :=
  sorry

/-- Theorem: In a 14-player round-robin tournament where each player plays 13 games
    and there are no draws, the maximum number of circular triples is 112 --/
theorem max_circular_triples_14_players :
  let t : Tournament := ⟨14, 13, true⟩
  max_circular_triples t = 112 := by sorry

end NUMINAMATH_CALUDE_max_circular_triples_14_players_l2951_295104


namespace NUMINAMATH_CALUDE_nine_integer_chords_l2951_295195

/-- Represents a circle with a given radius and a point at a given distance from its center -/
structure CircleWithPoint where
  radius : ℝ
  pointDistance : ℝ

/-- Counts the number of different integer-length chords containing the given point -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem nine_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 20) 
  (h2 : c.pointDistance = 12) : 
  countIntegerChords c = 9 := by
    sorry

end NUMINAMATH_CALUDE_nine_integer_chords_l2951_295195


namespace NUMINAMATH_CALUDE_smallest_square_coverage_l2951_295128

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- The number of rectangles needed to cover a square -/
def rectangles_needed (r : Rectangle) (s : Square) : ℕ :=
  (s.side * s.side) / (r.length * r.width)

/-- Checks if a square can be exactly covered by rectangles -/
def is_exactly_coverable (r : Rectangle) (s : Square) : Prop :=
  (s.side * s.side) % (r.length * r.width) = 0

theorem smallest_square_coverage (r : Rectangle) (s : Square) : 
  r.length = 3 ∧ r.width = 2 ∧ 
  s.side = 12 ∧
  is_exactly_coverable r s ∧
  rectangles_needed r s = 24 ∧
  (∀ s' : Square, s'.side < s.side → ¬(is_exactly_coverable r s')) := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_coverage_l2951_295128


namespace NUMINAMATH_CALUDE_no_perfect_square_n_n_plus_one_l2951_295148

theorem no_perfect_square_n_n_plus_one : ¬∃ (n : ℕ), n > 0 ∧ ∃ (k : ℕ), n * (n + 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_n_n_plus_one_l2951_295148


namespace NUMINAMATH_CALUDE_power_product_squared_l2951_295117

theorem power_product_squared (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l2951_295117


namespace NUMINAMATH_CALUDE_green_faction_liars_exceed_truthful_l2951_295129

/-- Represents the three factions in the parliament --/
inductive Faction
  | Blue
  | Red
  | Green

/-- Represents whether a deputy tells the truth or lies --/
inductive Honesty
  | Truthful
  | Liar

/-- Represents the parliament with its properties --/
structure Parliament where
  total_deputies : ℕ
  blue_affirmative : ℕ
  red_affirmative : ℕ
  green_affirmative : ℕ
  deputies : Faction → Honesty → ℕ

/-- The theorem to be proved --/
theorem green_faction_liars_exceed_truthful (p : Parliament)
  (h1 : p.total_deputies = 2016)
  (h2 : p.blue_affirmative = 1208)
  (h3 : p.red_affirmative = 908)
  (h4 : p.green_affirmative = 608)
  (h5 : p.total_deputies = p.deputies Faction.Blue Honesty.Truthful + p.deputies Faction.Blue Honesty.Liar +
                           p.deputies Faction.Red Honesty.Truthful + p.deputies Faction.Red Honesty.Liar +
                           p.deputies Faction.Green Honesty.Truthful + p.deputies Faction.Green Honesty.Liar)
  (h6 : p.blue_affirmative = p.deputies Faction.Blue Honesty.Truthful + p.deputies Faction.Red Honesty.Liar + p.deputies Faction.Green Honesty.Liar)
  (h7 : p.red_affirmative = p.deputies Faction.Red Honesty.Truthful + p.deputies Faction.Blue Honesty.Liar + p.deputies Faction.Green Honesty.Liar)
  (h8 : p.green_affirmative = p.deputies Faction.Green Honesty.Truthful + p.deputies Faction.Blue Honesty.Liar + p.deputies Faction.Red Honesty.Liar) :
  p.deputies Faction.Green Honesty.Liar = p.deputies Faction.Green Honesty.Truthful + 100 := by
  sorry


end NUMINAMATH_CALUDE_green_faction_liars_exceed_truthful_l2951_295129


namespace NUMINAMATH_CALUDE_tangent_half_angle_identity_l2951_295153

theorem tangent_half_angle_identity (α : Real) (h : Real.tan (α / 2) = 2) :
  (1 + Real.cos α) / Real.sin α = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_half_angle_identity_l2951_295153


namespace NUMINAMATH_CALUDE_second_division_percentage_l2951_295145

theorem second_division_percentage (total_students : ℕ) 
  (first_division_percentage : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  first_division_percentage = 30 / 100 →
  just_passed = 48 →
  (total_students : ℚ) * first_division_percentage + 
    (just_passed : ℚ) + 
    (total_students : ℚ) * (54 / 100) = total_students :=
by
  sorry

end NUMINAMATH_CALUDE_second_division_percentage_l2951_295145


namespace NUMINAMATH_CALUDE_rectangle_area_l2951_295151

theorem rectangle_area (a b : ℝ) (h1 : a = 10) (h2 : 2*a + 2*b = 40) : a * b = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2951_295151


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l2951_295137

/-- Represents a cube with stripes on its faces -/
structure StripedCube where
  faces : Fin 6 → Bool
  -- True represents one stripe orientation, False represents the other

/-- The probability of a continuous stripe encircling the cube -/
def probability_continuous_stripe : ℚ :=
  3 / 16

/-- Theorem stating the probability of a continuous stripe encircling the cube -/
theorem continuous_stripe_probability :
  probability_continuous_stripe = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l2951_295137


namespace NUMINAMATH_CALUDE_oasis_water_consumption_l2951_295161

theorem oasis_water_consumption (traveler_ounces camel_multiplier ounces_per_gallon : ℕ) 
  (h1 : traveler_ounces = 32)
  (h2 : camel_multiplier = 7)
  (h3 : ounces_per_gallon = 128) :
  (traveler_ounces + camel_multiplier * traveler_ounces) / ounces_per_gallon = 2 := by
  sorry

#check oasis_water_consumption

end NUMINAMATH_CALUDE_oasis_water_consumption_l2951_295161


namespace NUMINAMATH_CALUDE_factors_and_product_l2951_295114

-- Define a multiplication equation
def multiplication_equation (a b c : ℕ) : Prop := a * b = c

-- Define factors and product
def is_factor (a b c : ℕ) : Prop := multiplication_equation a b c
def is_product (a b c : ℕ) : Prop := multiplication_equation a b c

-- Theorem statement
theorem factors_and_product (a b c : ℕ) :
  multiplication_equation a b c → (is_factor a b c ∧ is_factor b a c ∧ is_product a b c) :=
by sorry

end NUMINAMATH_CALUDE_factors_and_product_l2951_295114


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l2951_295170

theorem cube_sum_inequality (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1)
  (h4 : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 ≤ 4 ∧ 
  (a + b + c + a^2 + b^2 + c^2 = 4 ↔ 
    ((a = 1 ∧ b = 1 ∧ c = -1) ∨
     (a = 1 ∧ b = -1 ∧ c = 1) ∨
     (a = -1 ∧ b = 1 ∧ c = 1))) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l2951_295170


namespace NUMINAMATH_CALUDE_quadratic_tangent_line_l2951_295160

/-- Given a quadratic function f(x) = x^2 + ax + b, prove that if its tangent line
    at (0, b) has the equation x - y + 1 = 0, then a = 1 and b = 1. -/
theorem quadratic_tangent_line (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  let f' : ℝ → ℝ := λ x ↦ 2*x + a
  (∀ x y, y = f x → x - y + 1 = 0 → x = 0) →
  f' 0 = 1 →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_tangent_line_l2951_295160


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l2951_295175

/-- A rectangle in the 2D plane -/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- A circle in the 2D plane -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- The area of intersection between a rectangle and a circle -/
def intersectionArea (rect : Rectangle) (circ : Circle) : ℝ := sorry

/-- The theorem stating the area of intersection between the specific rectangle and circle -/
theorem intersection_area_theorem :
  let rect : Rectangle := { x1 := 3, y1 := -3, x2 := 14, y2 := 10 }
  let circ : Circle := { center_x := 3, center_y := -3, radius := 4 }
  intersectionArea rect circ = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l2951_295175


namespace NUMINAMATH_CALUDE_solution_count_implies_n_l2951_295172

/-- The number of solutions to the equation 3x + 2y + 4z = n in positive integers x, y, and z -/
def num_solutions (n : ℕ+) : ℕ :=
  (Finset.filter (fun (x, y, z) => 3 * x + 2 * y + 4 * z = n) (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card

/-- Theorem stating that if the equation 3x + 2y + 4z = n has exactly 30 solutions in positive integers,
    then n must be either 22 or 23 -/
theorem solution_count_implies_n (n : ℕ+) :
  num_solutions n = 30 → n = 22 ∨ n = 23 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_implies_n_l2951_295172


namespace NUMINAMATH_CALUDE_exponential_function_passes_through_point_l2951_295191

theorem exponential_function_passes_through_point
  (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_passes_through_point_l2951_295191


namespace NUMINAMATH_CALUDE_parallel_condition_l2951_295119

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The line ax + y = 1 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + y = 1

/-- The line x + ay = 2a -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + a * y = 2 * a

/-- The condition a = -1 is sufficient but not necessary for the lines to be parallel -/
theorem parallel_condition (a : ℝ) : 
  (a = -1 → are_parallel (-a) (1/a)) ∧ 
  ¬(are_parallel (-a) (1/a) → a = -1) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l2951_295119


namespace NUMINAMATH_CALUDE_shifted_sine_function_phi_l2951_295196

theorem shifted_sine_function_phi (θ φ : ℝ) : 
  -π/2 < θ → θ < π/2 → φ > 0 →
  (∃ f g : ℝ → ℝ, 
    (∀ x, f x = 3 * Real.sin (2 * x + θ)) ∧
    (∀ x, g x = 3 * Real.sin (2 * (x - φ) + θ)) ∧
    f 0 = 3 * Real.sqrt 2 / 2 ∧
    g 0 = 3 * Real.sqrt 2 / 2) →
  φ ≠ 5 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_shifted_sine_function_phi_l2951_295196


namespace NUMINAMATH_CALUDE_hens_in_coop_l2951_295122

/-- Represents the chicken coop scenario --/
structure ChickenCoop where
  days : ℕ
  eggs_per_hen_per_day : ℕ
  boxes_filled : ℕ
  eggs_per_box : ℕ

/-- Calculates the number of hens in the chicken coop --/
def number_of_hens (coop : ChickenCoop) : ℕ :=
  (coop.boxes_filled * coop.eggs_per_box) / (coop.days * coop.eggs_per_hen_per_day)

/-- Theorem stating the number of hens in the specific scenario --/
theorem hens_in_coop : number_of_hens {
  days := 7,
  eggs_per_hen_per_day := 1,
  boxes_filled := 315,
  eggs_per_box := 6
} = 270 := by sorry

end NUMINAMATH_CALUDE_hens_in_coop_l2951_295122


namespace NUMINAMATH_CALUDE_sales_not_notebooks_or_markers_l2951_295157

/-- The percentage of sales that are not notebooks or markers -/
def other_sales_percentage (notebook_percentage marker_percentage : ℝ) : ℝ :=
  100 - (notebook_percentage + marker_percentage)

/-- Theorem stating that the percentage of sales not consisting of notebooks or markers is 33% -/
theorem sales_not_notebooks_or_markers :
  other_sales_percentage 42 25 = 33 := by
  sorry

end NUMINAMATH_CALUDE_sales_not_notebooks_or_markers_l2951_295157


namespace NUMINAMATH_CALUDE_drink_cost_is_2_50_l2951_295176

/-- The cost of a meal and drink with tip, given the following conditions:
  * The meal costs $10
  * The tip is 20% of the total cost (meal + drink)
  * The total amount paid is $15 -/
def total_cost (drink_cost : ℝ) : ℝ :=
  10 + drink_cost + 0.2 * (10 + drink_cost)

/-- Proves that the cost of the drink is $2.50 given the conditions -/
theorem drink_cost_is_2_50 :
  ∃ (drink_cost : ℝ), total_cost drink_cost = 15 ∧ drink_cost = 2.5 := by
sorry

end NUMINAMATH_CALUDE_drink_cost_is_2_50_l2951_295176


namespace NUMINAMATH_CALUDE_expression_value_l2951_295105

theorem expression_value (x y : ℝ) (h : x - y = 1) :
  x^4 - x*y^3 - x^3*y - 3*x^2*y + 3*x*y^2 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2951_295105


namespace NUMINAMATH_CALUDE_cubic_coefficient_B_l2951_295181

/-- A cubic function with roots at -2 and 2, and value -1 at x = 0 -/
def g (A B C D : ℝ) (x : ℝ) : ℝ := A * x^3 + B * x^2 + C * x + D

/-- Theorem stating that under given conditions, B = 1 -/
theorem cubic_coefficient_B (A B C D : ℝ) :
  g A B C D (-2) = 0 →
  g A B C D 0 = -1 →
  g A B C D 2 = 0 →
  B = 1 := by
    sorry

end NUMINAMATH_CALUDE_cubic_coefficient_B_l2951_295181


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_fraction_division_simplification_l2951_295167

-- Problem 1
theorem fraction_sum_equals_one (m n : ℝ) (h : m ≠ n) :
  m / (m - n) + n / (n - m) = 1 := by sorry

-- Problem 2
theorem fraction_division_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 / (x^2 - 1)) / (1 / (x + 1)) = 2 / (x - 1) := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_fraction_division_simplification_l2951_295167


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2951_295127

theorem remainder_divisibility (N : ℤ) : 
  N % 2 = 1 → N % 35 = 1 → N % 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2951_295127


namespace NUMINAMATH_CALUDE_typists_letters_problem_l2951_295198

theorem typists_letters_problem (typists_initial : ℕ) (letters_initial : ℕ) (time_initial : ℕ) 
  (typists_final : ℕ) (time_final : ℕ) :
  typists_initial = 20 →
  letters_initial = 44 →
  time_initial = 20 →
  typists_final = 30 →
  time_final = 60 →
  (typists_final : ℚ) * (letters_initial : ℚ) * (time_final : ℚ) / 
    ((typists_initial : ℚ) * (time_initial : ℚ)) = 198 := by
  sorry

end NUMINAMATH_CALUDE_typists_letters_problem_l2951_295198
