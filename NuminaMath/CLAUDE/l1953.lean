import Mathlib

namespace wendy_walking_distance_l1953_195381

theorem wendy_walking_distance
  (ran_distance : ℝ)
  (difference : ℝ)
  (h1 : ran_distance = 19.833333333333332)
  (h2 : difference = 10.666666666666666)
  (h3 : ran_distance = walked_distance + difference) :
  walked_distance = 9.166666666666666 :=
by
  sorry

end wendy_walking_distance_l1953_195381


namespace max_card_arrangement_l1953_195345

/-- A type representing the cards with numbers from 1 to 9 -/
inductive Card : Type
| one : Card
| two : Card
| three : Card
| four : Card
| five : Card
| six : Card
| seven : Card
| eight : Card
| nine : Card

/-- Convert a Card to its corresponding natural number -/
def card_to_nat (c : Card) : Nat :=
  match c with
  | Card.one => 1
  | Card.two => 2
  | Card.three => 3
  | Card.four => 4
  | Card.five => 5
  | Card.six => 6
  | Card.seven => 7
  | Card.eight => 8
  | Card.nine => 9

/-- Check if one card is divisible by another -/
def is_divisible (a b : Card) : Prop :=
  (card_to_nat a) % (card_to_nat b) = 0 ∨ (card_to_nat b) % (card_to_nat a) = 0

/-- A valid arrangement of cards -/
def valid_arrangement (arr : List Card) : Prop :=
  ∀ i, i + 1 < arr.length → is_divisible (arr.get ⟨i, by sorry⟩) (arr.get ⟨i + 1, by sorry⟩)

/-- The main theorem -/
theorem max_card_arrangement :
  ∃ (arr : List Card), arr.length = 8 ∧ valid_arrangement arr ∧
  ∀ (arr' : List Card), valid_arrangement arr' → arr'.length ≤ 8 :=
sorry

end max_card_arrangement_l1953_195345


namespace pieces_present_l1953_195328

/-- The number of pieces in a standard chess set -/
def standard_chess_pieces : ℕ := 32

/-- The number of missing pieces -/
def missing_pieces : ℕ := 4

/-- Theorem: The number of pieces present in an incomplete chess set -/
theorem pieces_present (standard : ℕ) (missing : ℕ) 
  (h1 : standard = standard_chess_pieces) 
  (h2 : missing = missing_pieces) : 
  standard - missing = 28 := by
  sorry

end pieces_present_l1953_195328


namespace imaginaria_city_population_l1953_195364

theorem imaginaria_city_population : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + 225 = b^2 + 1 ∧
  b^2 + 76 = c^2 ∧
  5 ∣ a^2 := by
sorry

end imaginaria_city_population_l1953_195364


namespace expression_simplification_l1953_195375

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (x + 1) / (x^2 + 2*x + 1) / (1 - 2 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l1953_195375


namespace a_must_be_negative_l1953_195359

theorem a_must_be_negative (a b c d e : ℝ) 
  (h1 : a / b < -(c / d))
  (h2 : b > 0)
  (h3 : d > 0)
  (h4 : e > 0)
  (h5 : a + e > 0) :
  a < 0 := by
  sorry

end a_must_be_negative_l1953_195359


namespace units_digit_pow2_2010_l1953_195396

/-- The units digit of 2^n for n ≥ 1 -/
def units_digit_pow2 (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 0 => 6
  | _ => 0  -- This case should never occur

/-- The units digit of 2^2010 is 4 -/
theorem units_digit_pow2_2010 : units_digit_pow2 2010 = 4 := by
  sorry

end units_digit_pow2_2010_l1953_195396


namespace not_necessary_nor_sufficient_condition_l1953_195306

theorem not_necessary_nor_sufficient_condition (x : ℝ) :
  ¬((-2 < x ∧ x < 1) → (|x| > 1)) ∧ ¬((|x| > 1) → (-2 < x ∧ x < 1)) := by
  sorry

end not_necessary_nor_sufficient_condition_l1953_195306


namespace min_strikes_to_defeat_dragon_l1953_195329

/-- Represents the state of the dragon -/
structure DragonState where
  heads : Nat
  tails : Nat

/-- Represents a strike against the dragon -/
inductive Strike
  | CutOneHead
  | CutOneTail
  | CutTwoHeads
  | CutTwoTails

/-- Applies a strike to the dragon state -/
def applyStrike (state : DragonState) (strike : Strike) : DragonState :=
  match strike with
  | Strike.CutOneHead => ⟨state.heads, state.tails⟩
  | Strike.CutOneTail => ⟨state.heads, state.tails + 1⟩
  | Strike.CutTwoHeads => ⟨state.heads - 2, state.tails⟩
  | Strike.CutTwoTails => ⟨state.heads + 1, state.tails - 2⟩

/-- Checks if the dragon is defeated (no heads and tails) -/
def isDragonDefeated (state : DragonState) : Prop :=
  state.heads = 0 ∧ state.tails = 0

/-- Theorem: The minimum number of strikes to defeat the dragon is 9 -/
theorem min_strikes_to_defeat_dragon :
  ∃ (strikes : List Strike),
    strikes.length = 9 ∧
    isDragonDefeated (strikes.foldl applyStrike ⟨3, 3⟩) ∧
    ∀ (otherStrikes : List Strike),
      otherStrikes.length < 9 →
      ¬isDragonDefeated (otherStrikes.foldl applyStrike ⟨3, 3⟩) :=
by
  sorry

end min_strikes_to_defeat_dragon_l1953_195329


namespace percent_y_of_x_l1953_195371

theorem percent_y_of_x (x y : ℝ) (h : 0.2 * (x - y) = 0.15 * (x + y)) : 
  y = (100 / 7) * x / 100 := by
  sorry

end percent_y_of_x_l1953_195371


namespace birds_count_l1953_195300

/-- The number of fish-eater birds Cohen saw over three days -/
def total_birds (initial : ℕ) : ℕ :=
  let day1 := initial
  let day2 := 2 * day1
  let day3 := day2 - 200
  day1 + day2 + day3

/-- Theorem stating that the total number of birds seen over three days is 1300 -/
theorem birds_count : total_birds 300 = 1300 := by
  sorry

end birds_count_l1953_195300


namespace parabola_properties_l1953_195311

def parabola (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem parabola_properties :
  (∀ x, parabola x ≥ parabola 1) ∧
  (∀ x₁ x₂, x₁ > 1 ∧ x₂ > 1 ∧ x₂ > x₁ → parabola x₂ > parabola x₁) ∧
  (parabola 1 = -2) ∧
  (∀ x, parabola x = parabola (2 - x)) :=
by sorry

end parabola_properties_l1953_195311


namespace l₃_symmetric_to_l₁_wrt_l₂_l1953_195383

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def l₂ (x y : ℝ) : Prop := x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (f g h : ℝ → ℝ → Prop) : Prop :=
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    f x₁ y₁ → h x₂ y₂ → 
    ∃ (x₀ y₀ : ℝ), g x₀ y₀ ∧ 
      x₀ = (x₁ + x₂) / 2 ∧ 
      y₀ = (y₁ + y₂) / 2

-- Theorem statement
theorem l₃_symmetric_to_l₁_wrt_l₂ : symmetric_wrt l₁ l₂ l₃ := by
  sorry

end l₃_symmetric_to_l₁_wrt_l₂_l1953_195383


namespace equality_of_two_numbers_l1953_195367

theorem equality_of_two_numbers (x y z : ℝ) 
  (h : x * y + z = y * z + x ∧ y * z + x = z * x + y) : 
  x = y ∨ y = z ∨ z = x := by
  sorry

end equality_of_two_numbers_l1953_195367


namespace star_removal_theorem_l1953_195307

/-- Represents a 2n × 2n table with stars -/
structure StarTable (n : ℕ) where
  stars : Finset ((Fin (2*n)) × (Fin (2*n)))
  star_count : stars.card = 3*n

/-- Represents a selection of rows and columns -/
structure Selection (n : ℕ) where
  rows : Finset (Fin (2*n))
  columns : Finset (Fin (2*n))
  row_count : rows.card = n
  column_count : columns.card = n

/-- Predicate to check if a star is removed by a selection -/
def is_removed (star : (Fin (2*n)) × (Fin (2*n))) (sel : Selection n) : Prop :=
  star.1 ∈ sel.rows ∨ star.2 ∈ sel.columns

/-- Theorem: For any 2n × 2n table with 3n stars, there exists a selection
    of n rows and n columns that removes all stars -/
theorem star_removal_theorem (n : ℕ) (table : StarTable n) :
  ∃ (sel : Selection n), ∀ star ∈ table.stars, is_removed star sel :=
sorry

end star_removal_theorem_l1953_195307


namespace henrys_initial_games_henrys_initial_games_is_58_l1953_195380

/-- Proves the number of games Henry had at first -/
theorem henrys_initial_games : ℕ → Prop := fun h =>
  let neil_initial := 7
  let henry_to_neil := 6
  let neil_final := neil_initial + henry_to_neil
  let henry_final := h - henry_to_neil
  (henry_final = 4 * neil_final) → h = 58

/-- The theorem holds for 58 -/
theorem henrys_initial_games_is_58 : henrys_initial_games 58 := by
  sorry

end henrys_initial_games_henrys_initial_games_is_58_l1953_195380


namespace modular_inverse_of_seven_mod_2003_l1953_195303

theorem modular_inverse_of_seven_mod_2003 : ∃ x : ℕ, x < 2003 ∧ (7 * x) % 2003 = 1 :=
by
  use 1717
  sorry

end modular_inverse_of_seven_mod_2003_l1953_195303


namespace jhons_total_pay_l1953_195312

/-- Calculates the total pay for a worker given their work schedule and pay rates. -/
def calculate_total_pay (total_days : ℕ) (present_days : ℕ) (present_rate : ℚ) (absent_rate : ℚ) : ℚ :=
  let absent_days := total_days - present_days
  present_days * present_rate + absent_days * absent_rate

/-- Proves that Jhon's total pay is $320.00 given the specified conditions. -/
theorem jhons_total_pay :
  calculate_total_pay 60 35 7 3 = 320 := by
  sorry

end jhons_total_pay_l1953_195312


namespace count_prime_digit_even_sum_integers_l1953_195374

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is a three-digit integer
def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

-- Define a function to get the digits of a three-digit number
def getDigits (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  (hundreds, tens, ones)

-- Define the main theorem
theorem count_prime_digit_even_sum_integers :
  (∃ S : Finset ℕ, 
    (∀ n ∈ S, isThreeDigit n ∧ 
              let (d1, d2, d3) := getDigits n
              isPrime d1 ∧ isPrime d2 ∧ isPrime d3 ∧
              (d1 + d2 + d3) % 2 = 0) ∧
    S.card = 18) := by sorry

end count_prime_digit_even_sum_integers_l1953_195374


namespace functional_equation_solution_l1953_195377

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = y + (f x)^2

/-- The main theorem stating that any function satisfying the equation must be either the identity function or its negation -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end functional_equation_solution_l1953_195377


namespace not_p_and_not_p_and_q_implies_p_or_q_not_necessarily_true_l1953_195382

theorem not_p_and_not_p_and_q_implies_p_or_q_not_necessarily_true
  (h1 : ¬p)
  (h2 : ¬(p ∧ q)) :
  ¬∀ (p q : Prop), p ∨ q :=
by
  sorry

end not_p_and_not_p_and_q_implies_p_or_q_not_necessarily_true_l1953_195382


namespace frank_skee_ball_tickets_proof_l1953_195351

def frank_skee_ball_tickets (whack_a_mole_tickets : ℕ) (candy_cost : ℕ) (candies_bought : ℕ) : ℕ :=
  candies_bought * candy_cost - whack_a_mole_tickets

theorem frank_skee_ball_tickets_proof :
  frank_skee_ball_tickets 33 6 7 = 9 := by
  sorry

end frank_skee_ball_tickets_proof_l1953_195351


namespace min_value_expression_l1953_195317

theorem min_value_expression (a b c d : ℝ) (hb : b ≠ 0) (horder : b > c ∧ c > a ∧ a > d) :
  ((2*a + b)^2 + (b - 2*c)^2 + (c - a)^2 + 3*d^2) / b^2 ≥ 49/36 :=
sorry

end min_value_expression_l1953_195317


namespace least_positive_integer_divisibility_l1953_195399

theorem least_positive_integer_divisibility (n : ℕ) : 
  (n % 2 = 1) → (∃ (a : ℕ), a > 0 ∧ (55^n + a * 32^n) % 2001 = 0) → 
  (∃ (a : ℕ), a > 0 ∧ a ≤ 436 ∧ (55^n + a * 32^n) % 2001 = 0) :=
by sorry

end least_positive_integer_divisibility_l1953_195399


namespace triangle_longest_side_l1953_195313

theorem triangle_longest_side (x : ℝ) : 
  5 + (x + 3) + (3 * x - 2) = 40 → 
  max 5 (max (x + 3) (3 * x - 2)) = 23.5 := by
sorry

end triangle_longest_side_l1953_195313


namespace square_sum_identity_l1953_195315

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end square_sum_identity_l1953_195315


namespace power_of_product_equality_l1953_195362

theorem power_of_product_equality (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 := by
  sorry

end power_of_product_equality_l1953_195362


namespace real_number_inequalities_l1953_195337

theorem real_number_inequalities (a b c : ℝ) :
  (∀ (c : ℝ), c ≠ 0 → (a * c^2 > b * c^2 → a > b)) ∧
  (a < b ∧ b < 0 → a^2 > a * b) ∧
  (∃ (a b c : ℝ), c > a ∧ a > b ∧ b > 0 ∧ a / (c - a) ≥ b / (c - b)) ∧
  (a > b ∧ b > 1 → a - 1 / b > b - 1 / a) :=
by sorry

end real_number_inequalities_l1953_195337


namespace expression_simplification_and_evaluation_l1953_195355

theorem expression_simplification_and_evaluation :
  let x : ℚ := 3
  let expr := (1 / (x - 1) + 1) / ((x^2 - 1) / (x^2 - 2*x + 1))
  expr = 3/4 := by sorry

end expression_simplification_and_evaluation_l1953_195355


namespace root_transformation_l1953_195326

theorem root_transformation (a₁ a₂ a₃ b c₁ c₂ c₃ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃)
  (h_roots : ∀ x, (x - a₁) * (x - a₂) * (x - a₃) = b ↔ x = c₁ ∨ x = c₂ ∨ x = c₃)
  (h_distinct_roots : c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₂ ≠ c₃) :
  ∀ x, (x + c₁) * (x + c₂) * (x + c₃) = b ↔ x = -a₁ ∨ x = -a₂ ∨ x = -a₃ :=
sorry

end root_transformation_l1953_195326


namespace kevin_koala_leaves_kevin_koala_leaves_min_l1953_195301

theorem kevin_koala_leaves (n : ℕ) : n > 1 ∧ ∃ k : ℕ, n^2 = k^6 → n ≥ 8 :=
by sorry

theorem kevin_koala_leaves_min : ∃ k : ℕ, 8^2 = k^6 :=
by sorry

end kevin_koala_leaves_kevin_koala_leaves_min_l1953_195301


namespace right_triangle_inequality_l1953_195366

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  3 < (c^3 - a^3 - b^3) / (c * (c - a) * (c - b)) ∧ 
  (c^3 - a^3 - b^3) / (c * (c - a) * (c - b)) < Real.sqrt 2 + 2 := by
  sorry

end right_triangle_inequality_l1953_195366


namespace hostel_expenditure_l1953_195387

theorem hostel_expenditure (original_students : ℕ) (new_students : ℕ) (expense_increase : ℚ) (average_decrease : ℚ) 
  (h1 : original_students = 35)
  (h2 : new_students = 7)
  (h3 : expense_increase = 42)
  (h4 : average_decrease = 1) :
  ∃ (original_expenditure : ℚ),
    original_expenditure = original_students * 
      ((expense_increase + (original_students + new_students) * average_decrease) / new_students) := by
  sorry

end hostel_expenditure_l1953_195387


namespace quadratic_equation_solution_l1953_195368

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 3
  let c : ℝ := -1
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + 3*x₁ - 1 = 0 ∧ x₂^2 + 3*x₂ - 1 = 0 :=
by
  sorry

end quadratic_equation_solution_l1953_195368


namespace revenue_decrease_percent_l1953_195324

/-- Calculates the percentage decrease in revenue when tax is reduced and consumption is increased -/
theorem revenue_decrease_percent (tax_reduction : Real) (consumption_increase : Real)
  (h1 : tax_reduction = 0.20)
  (h2 : consumption_increase = 0.05) :
  1 - (1 - tax_reduction) * (1 + consumption_increase) = 0.16 := by
  sorry

end revenue_decrease_percent_l1953_195324


namespace all_transformations_correct_l1953_195378

-- Define the transformations
def transformation_A (a b : ℝ) : Prop := a = b → a + 5 = b + 5

def transformation_B (x y a : ℝ) : Prop := x = y → x / a = y / a

def transformation_C (m n : ℝ) : Prop := m = n → 1 - 3 * m = 1 - 3 * n

def transformation_D (x y c : ℝ) : Prop := x = y → x * c = y * c

-- Theorem stating all transformations are correct
theorem all_transformations_correct :
  (∀ a b : ℝ, transformation_A a b) ∧
  (∀ x y a : ℝ, a ≠ 0 → transformation_B x y a) ∧
  (∀ m n : ℝ, transformation_C m n) ∧
  (∀ x y c : ℝ, transformation_D x y c) :=
sorry

end all_transformations_correct_l1953_195378


namespace symmetric_line_y_axis_correct_l1953_195323

/-- Given a line with equation ax + by + c = 0, return the equation of the line symmetric to it with respect to the y-axis -/
def symmetricLineYAxis (a b c : ℝ) : ℝ × ℝ × ℝ := (-a, b, c)

theorem symmetric_line_y_axis_correct :
  let original_line := (2, 1, -4)
  let symmetric_line := symmetricLineYAxis 2 1 (-4)
  symmetric_line = (-2, 1, -4) :=
by sorry

end symmetric_line_y_axis_correct_l1953_195323


namespace smallest_norm_given_condition_l1953_195316

/-- Given a vector v in ℝ², prove that the smallest possible value of its norm,
    given that ‖v + (4, 2)‖ = 10, is 10 - 2√5. -/
theorem smallest_norm_given_condition (v : ℝ × ℝ) 
    (h : ‖v + (4, 2)‖ = 10) : 
    ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ 
    ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ := by
  sorry

end smallest_norm_given_condition_l1953_195316


namespace fifth_monday_in_leap_year_l1953_195331

/-- Represents a date in February of a leap year -/
structure FebruaryDate :=
  (day : ℕ)
  (is_leap_year : Bool)

/-- Represents a day of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the weekday of a given February date -/
def weekday_of_date (d : FebruaryDate) : Weekday :=
  sorry

/-- Returns the number of Mondays up to and including a given date in February -/
def mondays_up_to (d : FebruaryDate) : ℕ :=
  sorry

/-- Theorem: In a leap year where February 7 is a Tuesday, 
    the fifth Monday in February falls on February 27 -/
theorem fifth_monday_in_leap_year :
  let feb7 : FebruaryDate := ⟨7, true⟩
  let feb27 : FebruaryDate := ⟨27, true⟩
  weekday_of_date feb7 = Weekday.Tuesday →
  mondays_up_to feb27 = 5 :=
by
  sorry

end fifth_monday_in_leap_year_l1953_195331


namespace tetrahedron_angle_difference_l1953_195325

open Real

/-- Represents a tetrahedron -/
structure Tetrahedron where
  /-- The sum of all dihedral angles in the tetrahedron -/
  dihedral_sum : ℝ
  /-- The sum of all trihedral angles in the tetrahedron -/
  trihedral_sum : ℝ

/-- 
Theorem: For any tetrahedron, the difference between the sum of its dihedral angles 
and the sum of its trihedral angles is equal to 4π.
-/
theorem tetrahedron_angle_difference (t : Tetrahedron) : 
  t.dihedral_sum - t.trihedral_sum = 4 * π :=
sorry

end tetrahedron_angle_difference_l1953_195325


namespace fourth_root_equivalence_l1953_195310

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : (x * x^(1/3))^(1/4) = x^(1/3) := by
  sorry

end fourth_root_equivalence_l1953_195310


namespace u_value_l1953_195308

theorem u_value : 
  let u : ℝ := 1 / (2 - Real.rpow 3 (1/3))
  u = 2 + Real.rpow 3 (1/3) := by
  sorry

end u_value_l1953_195308


namespace max_value_of_f_l1953_195349

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 3

-- State the theorem
theorem max_value_of_f (b : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f b x ≥ 1) →
  (∃ x ∈ Set.Icc (-1) 2, f b x = 1) →
  (∃ x ∈ Set.Icc (-1) 2, f b x = max 13 (4 + 2*Real.sqrt 2)) :=
by sorry

end max_value_of_f_l1953_195349


namespace fraction_of_72_l1953_195314

theorem fraction_of_72 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by
  sorry

end fraction_of_72_l1953_195314


namespace factories_unchecked_l1953_195365

theorem factories_unchecked (total : ℕ) (first_group : ℕ) (second_group : ℕ)
  (h1 : total = 169)
  (h2 : first_group = 69)
  (h3 : second_group = 52) :
  total - (first_group + second_group) = 48 := by
  sorry

end factories_unchecked_l1953_195365


namespace right_triangular_prism_dimension_l1953_195385

/-- 
Given a right triangular prism with:
- base edges a = 5 and b = 12
- height c = 13
- body diagonal d = 15
Prove that the third dimension of a rectangular face (h) is equal to 2√14
-/
theorem right_triangular_prism_dimension (a b c d h : ℝ) : 
  a = 5 → b = 12 → c = 13 → d = 15 →
  a^2 + b^2 = c^2 →
  d^2 = a^2 + b^2 + h^2 →
  h = 2 * Real.sqrt 14 := by
  sorry

end right_triangular_prism_dimension_l1953_195385


namespace min_speed_to_arrive_first_l1953_195373

/-- Proves the minimum speed required for the second person to arrive first -/
theorem min_speed_to_arrive_first (distance : ℝ) (speed_A : ℝ) (head_start : ℝ) 
  (h1 : distance = 180)
  (h2 : speed_A = 40)
  (h3 : head_start = 0.5)
  (h4 : speed_A > 0) : 
  ∃ (min_speed : ℝ), min_speed > 45 ∧ 
    ∀ (speed_B : ℝ), speed_B > min_speed → 
      distance / speed_B < distance / speed_A - head_start := by
sorry

end min_speed_to_arrive_first_l1953_195373


namespace tomato_plants_per_row_is_eight_l1953_195391

/-- Represents the garden planting scenario -/
structure GardenPlanting where
  cucumber_to_tomato_ratio : ℚ
  total_rows : ℕ
  tomatoes_per_plant : ℕ
  total_tomatoes : ℕ

/-- Calculates the number of tomato plants per row -/
def tomato_plants_per_row (g : GardenPlanting) : ℚ :=
  g.total_tomatoes / (g.tomatoes_per_plant * (g.total_rows / (1 + g.cucumber_to_tomato_ratio)))

/-- Theorem stating that the number of tomato plants per row is 8 -/
theorem tomato_plants_per_row_is_eight (g : GardenPlanting) 
  (h1 : g.cucumber_to_tomato_ratio = 2)
  (h2 : g.total_rows = 15)
  (h3 : g.tomatoes_per_plant = 3)
  (h4 : g.total_tomatoes = 120) : 
  tomato_plants_per_row g = 8 := by
  sorry

end tomato_plants_per_row_is_eight_l1953_195391


namespace fraction_simplification_l1953_195353

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 4/7) 
  (hy : y = 5/8) : 
  (6*x - 4*y) / (36*x*y) = 13/180 := by
  sorry

end fraction_simplification_l1953_195353


namespace range_of_g_bounds_achievable_l1953_195321

theorem range_of_g (x : ℝ) : ∃ (y : ℝ), y ∈ Set.Icc (3/4 : ℝ) 1 ∧ y = Real.cos x ^ 4 + Real.sin x ^ 2 :=
sorry

theorem bounds_achievable :
  (∃ (x : ℝ), Real.cos x ^ 4 + Real.sin x ^ 2 = 3/4) ∧
  (∃ (x : ℝ), Real.cos x ^ 4 + Real.sin x ^ 2 = 1) :=
sorry

end range_of_g_bounds_achievable_l1953_195321


namespace jennys_bottle_cap_bounce_fraction_l1953_195395

theorem jennys_bottle_cap_bounce_fraction :
  ∀ (jenny_initial : ℝ) (mark_initial : ℝ) (jenny_fraction : ℝ),
    jenny_initial = 18 →
    mark_initial = 15 →
    (mark_initial + 2 * mark_initial) - (jenny_initial + jenny_initial * jenny_fraction) = 21 →
    jenny_fraction = 1/3 := by
  sorry

end jennys_bottle_cap_bounce_fraction_l1953_195395


namespace min_value_abs_sum_l1953_195352

theorem min_value_abs_sum (x : ℚ) : 
  |x - 1| + |x + 3| ≥ 4 ∧ ∃ y : ℚ, |y - 1| + |y + 3| = 4 := by
  sorry

end min_value_abs_sum_l1953_195352


namespace tangent_line_and_extrema_l1953_195322

/-- The function f(x) = x³ - 3ax² + 3bx -/
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*b*x

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 6*a*x + 3*b

theorem tangent_line_and_extrema :
  ∃ (a b : ℝ),
    /- f(x) is tangent to 12x + y - 1 = 0 at (1, -11) -/
    (f a b 1 = -11 ∧ f_derivative a b 1 = -12) ∧
    /- a = 1 and b = -3 -/
    (a = 1 ∧ b = -3) ∧
    /- Maximum value of f(x) in [-2, 4] is 5 -/
    (∀ x, x ∈ Set.Icc (-2) 4 → f a b x ≤ 5) ∧
    (∃ x, x ∈ Set.Icc (-2) 4 ∧ f a b x = 5) ∧
    /- Minimum value of f(x) in [-2, 4] is -27 -/
    (∀ x, x ∈ Set.Icc (-2) 4 → f a b x ≥ -27) ∧
    (∃ x, x ∈ Set.Icc (-2) 4 ∧ f a b x = -27) :=
by sorry

end tangent_line_and_extrema_l1953_195322


namespace total_cost_price_is_60_2_l1953_195343

/-- Calculates the cost price given the selling price and loss ratio -/
def costPrice (sellingPrice : ℚ) (lossRatio : ℚ) : ℚ :=
  sellingPrice / (1 - lossRatio)

/-- The total cost price of an apple, an orange, and a banana -/
def totalCostPrice : ℚ :=
  costPrice 16 (1/6) + costPrice 20 (1/5) + costPrice 12 (1/4)

theorem total_cost_price_is_60_2 :
  totalCostPrice = 60.2 := by sorry

end total_cost_price_is_60_2_l1953_195343


namespace mean_age_is_eleven_l1953_195346

/-- Represents the ages of children in the Euler family -/
def euler_ages : List ℕ := [10, 12, 8]

/-- Represents the ages of children in the Gauss family -/
def gauss_ages : List ℕ := [8, 8, 8, 16, 18]

/-- Calculates the mean age of all children from both families -/
def mean_age : ℚ := (euler_ages.sum + gauss_ages.sum) / (euler_ages.length + gauss_ages.length)

theorem mean_age_is_eleven : mean_age = 11 := by
  sorry

end mean_age_is_eleven_l1953_195346


namespace nine_solutions_mod_455_l1953_195390

theorem nine_solutions_mod_455 : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 1 ≤ n ∧ n ≤ 455 ∧ n^3 % 455 = 1) ∧ 
    (∀ n, 1 ≤ n ∧ n ≤ 455 ∧ n^3 % 455 = 1 → n ∈ s) ∧ 
    s.card = 9 :=
by sorry

end nine_solutions_mod_455_l1953_195390


namespace diagonal_intersection_probability_is_six_thirteenths_l1953_195369

/-- A regular nonagon is a 9-sided regular polygon -/
structure RegularNonagon where
  -- We don't need to define the structure explicitly for this problem

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := 27

/-- The total number of pairs of diagonals in a regular nonagon -/
def total_diagonal_pairs (n : RegularNonagon) : ℕ := 351

/-- The number of pairs of intersecting diagonals in a regular nonagon -/
def intersecting_diagonal_pairs (n : RegularNonagon) : ℕ := 126

/-- The probability that two randomly chosen diagonals in a regular nonagon intersect -/
def diagonal_intersection_probability (n : RegularNonagon) : ℚ :=
  intersecting_diagonal_pairs n / total_diagonal_pairs n

theorem diagonal_intersection_probability_is_six_thirteenths (n : RegularNonagon) :
  diagonal_intersection_probability n = 6/13 := by
  sorry

end diagonal_intersection_probability_is_six_thirteenths_l1953_195369


namespace original_line_equation_l1953_195336

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shift a line horizontally -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - l.slope * shift }

theorem original_line_equation (l : Line) :
  (shift_line l 2).slope = 2 ∧ (shift_line l 2).intercept = 3 →
  l.slope = 2 ∧ l.intercept = 7 := by
  sorry

end original_line_equation_l1953_195336


namespace emily_days_off_l1953_195397

/-- The number of holidays Emily took in a year -/
def total_holidays : ℕ := 24

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of times Emily took a day off each month -/
def days_off_per_month : ℚ := total_holidays / months_in_year

theorem emily_days_off : days_off_per_month = 2 := by
  sorry

end emily_days_off_l1953_195397


namespace decimal_17_to_binary_l1953_195335

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) (acc : List Bool) : List Bool :=
      if m = 0 then acc
      else go (m / 2) ((m % 2 = 1) :: acc)
    go n []

/-- Converts a list of bits to a string representation of a binary number -/
def binaryToString (bits : List Bool) : String :=
  bits.map (fun b => if b then '1' else '0') |> String.mk

theorem decimal_17_to_binary :
  binaryToString (toBinary 17) = "10001" := by
  sorry

end decimal_17_to_binary_l1953_195335


namespace tree_distance_l1953_195354

/-- The distance between consecutive trees in a yard with an obstacle -/
theorem tree_distance (yard_length : ℝ) (num_trees : ℕ) (obstacle_gap : ℝ) :
  yard_length = 600 →
  num_trees = 36 →
  obstacle_gap = 10 →
  (yard_length - obstacle_gap) / (num_trees - 1 : ℝ) = 590 / 35 := by
  sorry

end tree_distance_l1953_195354


namespace sin_cos_sum_equal_shifted_cos_l1953_195388

theorem sin_cos_sum_equal_shifted_cos (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.cos (3 * x - π / 4) := by
  sorry

end sin_cos_sum_equal_shifted_cos_l1953_195388


namespace x_value_theorem_l1953_195320

theorem x_value_theorem (x n : ℕ) : 
  x = 2^n - 32 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 3 ∧ q ≠ 3 ∧
    x = 3 * p * q) →
  x = 480 ∨ x = 2016 := by
sorry

end x_value_theorem_l1953_195320


namespace maximize_expression_l1953_195394

def a : Set Int := {-3, -2, -1, 0, 1, 2, 3}

theorem maximize_expression (v : ℝ) :
  ∃ (x y z : Int),
    x ∈ a ∧ y ∈ a ∧ z ∈ a ∧
    (∀ (x' y' z' : Int), x' ∈ a → y' ∈ a → z' ∈ a → v * x' - y' * z' ≤ v * x - y * z) ∧
    v * x - y * z = 15 ∧
    y = -3 ∧ z = 3 :=
sorry

end maximize_expression_l1953_195394


namespace fraction_unchanged_l1953_195318

theorem fraction_unchanged (x y : ℝ) (h : y ≠ 2*x) : 
  (3*(3*x)) / (2*(3*x) - 3*y) = (3*x) / (2*x - y) :=
by sorry

end fraction_unchanged_l1953_195318


namespace square_to_circle_ratio_l1953_195363

-- Define the sector and its properties
structure RectangularSector where
  R : ℝ  -- Radius of the sector
  a : ℝ  -- Side length of the inscribed square

-- Define the circle touching the chord, arc, and square side
def TouchingCircle (sector : RectangularSector) :=
  { r : ℝ // r > 0 }

-- State the theorem
theorem square_to_circle_ratio
  (sector : RectangularSector)
  (circle : TouchingCircle sector) :
  sector.a / circle.val =
    ((Real.sqrt 5 + Real.sqrt 2) * (3 + Real.sqrt 5)) / (6 * Real.sqrt 2) :=
sorry

end square_to_circle_ratio_l1953_195363


namespace equation_solution_l1953_195360

theorem equation_solution : ∃ x : ℝ, (144 / 0.144 = x / 0.0144) ∧ (x = 14.4) := by
  sorry

end equation_solution_l1953_195360


namespace algebraic_expression_equality_l1953_195392

theorem algebraic_expression_equality : 
  Real.sqrt (5 - 2 * Real.sqrt 6) + Real.sqrt (7 - 4 * Real.sqrt 3) = 2 - Real.sqrt 2 := by
sorry

end algebraic_expression_equality_l1953_195392


namespace election_votes_l1953_195305

theorem election_votes (V : ℕ) : 
  (60 * V / 100 - 40 * V / 100 = 1380) → V = 6900 := by sorry

end election_votes_l1953_195305


namespace arithmetic_sequence_ratio_l1953_195309

/-- Given an arithmetic sequence with common ratio q ≠ 0, where S_n is the sum of first n terms,
    if S_3, S_9, and S_6 form an arithmetic sequence, then q^3 = 3/2 -/
theorem arithmetic_sequence_ratio (q : ℝ) (a₁ : ℝ) (S : ℕ → ℝ) : 
  q ≠ 0 ∧ 
  (∀ n, S n = a₁ * (1 - q^n) / (1 - q)) ∧ 
  (2 * S 9 = S 3 + S 6) →
  q^3 = 3/2 := by
  sorry

end arithmetic_sequence_ratio_l1953_195309


namespace prob_friends_same_group_l1953_195357

/-- The total number of students -/
def total_students : ℕ := 900

/-- The number of lunch groups -/
def num_groups : ℕ := 5

/-- The number of friends we're considering -/
def num_friends : ℕ := 4

/-- Represents a random assignment of students to lunch groups -/
def random_assignment : Type := Fin total_students → Fin num_groups

/-- The probability of a specific student being assigned to a specific group -/
def prob_single_assignment : ℚ := 1 / num_groups

/-- 
The probability that all friends are assigned to the same group
given a random assignment of students to groups
-/
def prob_all_friends_same_group (assignment : random_assignment) : ℚ :=
  prob_single_assignment ^ (num_friends - 1)

theorem prob_friends_same_group :
  ∀ (assignment : random_assignment),
    prob_all_friends_same_group assignment = 1 / 125 :=
by sorry

end prob_friends_same_group_l1953_195357


namespace vector_translation_result_l1953_195361

def vector_translation (a : ℝ × ℝ) (right : ℝ) (down : ℝ) : ℝ × ℝ :=
  (a.1 + right, a.2 - down)

theorem vector_translation_result :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := vector_translation a 2 1
  b = (3, 0) := by sorry

end vector_translation_result_l1953_195361


namespace ST_SQ_ratio_is_930_2197_l1953_195370

-- Define the points
variable (P Q R S T : ℝ × ℝ)

-- Define the triangles and their properties
def triangle_PQR_right_at_R : Prop := sorry
def PR_length : ℝ := 5
def RQ_length : ℝ := 12

def triangle_PQS_right_at_P : Prop := sorry
def PS_length : ℝ := 15

-- R and S are on opposite sides of PQ
def R_S_opposite_sides : Prop := sorry

-- Line through S parallel to PR meets RQ extended at T
def S_parallel_PR_meets_RQ_at_T : Prop := sorry

-- Define the ratio ST/SQ
def ST_SQ_ratio : ℝ := sorry

-- Theorem statement
theorem ST_SQ_ratio_is_930_2197 
  (h1 : triangle_PQR_right_at_R)
  (h2 : triangle_PQS_right_at_P)
  (h3 : R_S_opposite_sides)
  (h4 : S_parallel_PR_meets_RQ_at_T) :
  ST_SQ_ratio = 930 / 2197 := by sorry

end ST_SQ_ratio_is_930_2197_l1953_195370


namespace odd_function_product_negative_l1953_195347

theorem odd_function_product_negative (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_nonzero : ∀ x, f x ≠ 0) :
  ∀ x, f x * f (-x) < 0 := by
sorry

end odd_function_product_negative_l1953_195347


namespace bryan_shelves_count_l1953_195389

/-- The number of mineral samples per shelf -/
def samples_per_shelf : ℕ := 65

/-- The total number of mineral samples -/
def total_samples : ℕ := 455

/-- The number of shelves Bryan has -/
def number_of_shelves : ℕ := total_samples / samples_per_shelf

theorem bryan_shelves_count : number_of_shelves = 7 := by
  sorry

end bryan_shelves_count_l1953_195389


namespace sum_of_absolute_values_zero_l1953_195333

theorem sum_of_absolute_values_zero (a b : ℝ) : 
  |a + 2| + |b - 7| = 0 → a + b = 5 := by
  sorry

end sum_of_absolute_values_zero_l1953_195333


namespace band_gigs_theorem_l1953_195342

/-- Represents a band with its members and earnings -/
structure Band where
  members : ℕ
  earnings_per_member_per_gig : ℕ
  total_earnings : ℕ

/-- Calculates the number of gigs played by a band -/
def gigs_played (b : Band) : ℕ :=
  b.total_earnings / (b.members * b.earnings_per_member_per_gig)

/-- Theorem stating that for a band with 4 members, $20 earnings per member per gig,
    and $400 total earnings, the number of gigs played is 5 -/
theorem band_gigs_theorem (b : Band) 
    (h1 : b.members = 4)
    (h2 : b.earnings_per_member_per_gig = 20)
    (h3 : b.total_earnings = 400) :
    gigs_played b = 5 := by
  sorry

end band_gigs_theorem_l1953_195342


namespace multiple_of_smaller_integer_l1953_195319

theorem multiple_of_smaller_integer (s l : ℤ) (k : ℚ) : 
  s + l = 30 → 
  s = 10 → 
  2 * l = k * s - 10 → 
  k = 5 := by sorry

end multiple_of_smaller_integer_l1953_195319


namespace power_of_81_five_sixths_l1953_195302

theorem power_of_81_five_sixths :
  (81 : ℝ) ^ (5/6) = 27 * (3 : ℝ) ^ (1/3) := by sorry

end power_of_81_five_sixths_l1953_195302


namespace solution_set_theorem_l1953_195393

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

-- State the theorem
theorem solution_set_theorem (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, f a b (-2*x) < 0 ↔ x < -3/2 ∨ 1/2 < x) :=
by sorry

end solution_set_theorem_l1953_195393


namespace alice_bushes_theorem_l1953_195304

/-- The number of bushes Alice needs to buy for her yard -/
def bushes_needed (sides : ℕ) (side_length : ℕ) (bush_length : ℕ) : ℕ :=
  (sides * side_length) / bush_length

theorem alice_bushes_theorem :
  bushes_needed 3 16 4 = 12 := by
  sorry

end alice_bushes_theorem_l1953_195304


namespace existence_of_solution_l1953_195376

theorem existence_of_solution : ∃ (x y : ℕ), x^99 = 2013 * y^100 := by
  sorry

end existence_of_solution_l1953_195376


namespace train_passing_time_l1953_195356

/-- The time taken for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 120 →
  train_speed = 50 * (1000 / 3600) →
  man_speed = 4 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 8 := by
  sorry

end train_passing_time_l1953_195356


namespace discounted_price_calculation_l1953_195332

def television_price : ℝ := 650
def number_of_televisions : ℕ := 2
def discount_percentage : ℝ := 0.25

theorem discounted_price_calculation :
  let total_price := television_price * number_of_televisions
  let discount_amount := total_price * discount_percentage
  let final_price := total_price - discount_amount
  final_price = 975 := by
  sorry

end discounted_price_calculation_l1953_195332


namespace construction_delay_l1953_195341

/-- Represents the construction project with given parameters -/
structure ConstructionProject where
  totalDays : ℕ
  initialWorkers : ℕ
  additionalWorkers : ℕ
  additionalWorkersStartDay : ℕ

/-- Calculates the total work units completed in the project -/
def totalWorkUnits (project : ConstructionProject) : ℕ :=
  project.initialWorkers * project.totalDays +
  project.additionalWorkers * (project.totalDays - project.additionalWorkersStartDay)

/-- Calculates the number of days needed to complete the work with only initial workers -/
def daysNeededWithoutAdditionalWorkers (project : ConstructionProject) : ℕ :=
  (totalWorkUnits project) / project.initialWorkers

/-- Theorem: The project will be 90 days behind schedule without additional workers -/
theorem construction_delay (project : ConstructionProject) 
  (h1 : project.totalDays = 100)
  (h2 : project.initialWorkers = 100)
  (h3 : project.additionalWorkers = 100)
  (h4 : project.additionalWorkersStartDay = 10) :
  daysNeededWithoutAdditionalWorkers project - project.totalDays = 90 := by
  sorry

end construction_delay_l1953_195341


namespace billy_wins_l1953_195386

/-- Represents the swimming times for Billy and Margaret -/
structure SwimmingTimes where
  billy_first_5_laps : ℕ  -- in minutes
  billy_next_3_laps : ℕ  -- in minutes
  billy_next_lap : ℕ     -- in minutes
  billy_final_lap : ℕ    -- in seconds
  margaret_total : ℕ     -- in minutes

/-- Calculates the time difference between Billy and Margaret's finish times -/
def timeDifference (times : SwimmingTimes) : ℕ :=
  let billy_total_seconds := 
    (times.billy_first_5_laps + times.billy_next_3_laps + times.billy_next_lap) * 60 + times.billy_final_lap
  let margaret_total_seconds := times.margaret_total * 60
  margaret_total_seconds - billy_total_seconds

/-- Theorem stating that Billy finishes 30 seconds before Margaret -/
theorem billy_wins (times : SwimmingTimes) 
    (h1 : times.billy_first_5_laps = 2)
    (h2 : times.billy_next_3_laps = 4)
    (h3 : times.billy_next_lap = 1)
    (h4 : times.billy_final_lap = 150)
    (h5 : times.margaret_total = 10) : 
  timeDifference times = 30 := by
  sorry


end billy_wins_l1953_195386


namespace intersection_point_and_lines_l1953_195340

/-- Given two lines that intersect at point P, this theorem proves:
    1. The equation of a line passing through P and parallel to a given line
    2. The equation of a line passing through P that maximizes the distance from the origin --/
theorem intersection_point_and_lines (x y : ℝ) :
  (2 * x + y = 8) →
  (x - 2 * y = -1) →
  (∃ P : ℝ × ℝ, P.1 = x ∧ P.2 = y) →
  (∃ l₁ : ℝ → ℝ → Prop, l₁ x y ↔ 4 * x - 3 * y = 6) ∧
  (∃ l₂ : ℝ → ℝ → Prop, l₂ x y ↔ 3 * x + 2 * y = 13) :=
by sorry

end intersection_point_and_lines_l1953_195340


namespace tv_show_duration_l1953_195350

theorem tv_show_duration (seasons_15 seasons_20 seasons_12 : ℕ)
  (episodes_15 episodes_20 episodes_12 : ℕ)
  (avg_episodes_per_year : ℕ) :
  seasons_15 = 8 →
  seasons_20 = 4 →
  seasons_12 = 2 →
  episodes_15 = 15 →
  episodes_20 = 20 →
  episodes_12 = 12 →
  avg_episodes_per_year = 16 →
  (seasons_15 * episodes_15 + seasons_20 * episodes_20 + seasons_12 * episodes_12) /
    avg_episodes_per_year = 14 :=
by
  sorry

end tv_show_duration_l1953_195350


namespace fifth_sphere_radius_l1953_195338

/-- Represents a cone with height and base radius 7 -/
structure Cone :=
  (height : ℝ := 7)
  (base_radius : ℝ := 7)

/-- Represents a sphere with a center and radius -/
structure Sphere :=
  (center : ℝ × ℝ × ℝ)
  (radius : ℝ)

/-- Represents the configuration of spheres in the cone -/
structure SphereConfiguration :=
  (cone : Cone)
  (base_spheres : Fin 4 → Sphere)
  (top_sphere : Sphere)

/-- Checks if two spheres are externally touching -/
def externally_touching (s1 s2 : Sphere) : Prop :=
  let (x1, y1, z1) := s1.center
  let (x2, y2, z2) := s2.center
  (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2 = (s1.radius + s2.radius)^2

/-- Checks if a sphere touches the lateral surface of the cone -/
def touches_lateral_surface (s : Sphere) (c : Cone) : Prop :=
  sorry -- Definition omitted for brevity

/-- Checks if a sphere touches the base of the cone -/
def touches_base (s : Sphere) (c : Cone) : Prop :=
  sorry -- Definition omitted for brevity

/-- Theorem stating the radius of the fifth sphere -/
theorem fifth_sphere_radius (config : SphereConfiguration) :
  (∀ i j : Fin 4, i ≠ j → externally_touching (config.base_spheres i) (config.base_spheres j)) →
  (∀ i : Fin 4, touches_lateral_surface (config.base_spheres i) config.cone) →
  (∀ i : Fin 4, touches_base (config.base_spheres i) config.cone) →
  (∀ i : Fin 4, externally_touching (config.base_spheres i) config.top_sphere) →
  touches_lateral_surface config.top_sphere config.cone →
  config.top_sphere.radius = 2 * Real.sqrt 2 - 1 :=
by sorry

end fifth_sphere_radius_l1953_195338


namespace triangle_properties_l1953_195334

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the area and cosine of angle ADC where D is the midpoint of BC. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  c = 4 →
  b = 3 →
  A = π / 3 →
  let S := (1 / 2) * b * c * Real.sin A
  let cos_ADC := (7 * Real.sqrt 481) / 481
  (S = 3 * Real.sqrt 3 ∧ cos_ADC = (7 * Real.sqrt 481) / 481) :=
by sorry

end triangle_properties_l1953_195334


namespace chicken_count_l1953_195330

/-- Given a farm with chickens and buffalos, prove the number of chickens. -/
theorem chicken_count (total_animals : ℕ) (total_legs : ℕ) (chickens : ℕ) (buffalos : ℕ) : 
  total_animals = 9 →
  total_legs = 26 →
  chickens + buffalos = total_animals →
  2 * chickens + 4 * buffalos = total_legs →
  chickens = 5 := by
sorry

end chicken_count_l1953_195330


namespace extra_flowers_l1953_195398

theorem extra_flowers (tulips roses used : ℕ) : 
  tulips = 4 → roses = 11 → used = 11 → tulips + roses - used = 4 := by
  sorry

end extra_flowers_l1953_195398


namespace class_average_increase_l1953_195379

/-- Proves that adding a 50-year-old student to a class of 19 students with an average age of 10 years
    increases the overall average age by 2 years. -/
theorem class_average_increase (n : ℕ) (original_avg : ℝ) (new_student_age : ℝ) :
  n = 19 →
  original_avg = 10 →
  new_student_age = 50 →
  (n * original_avg + new_student_age) / (n + 1) - original_avg = 2 := by
  sorry

#check class_average_increase

end class_average_increase_l1953_195379


namespace solve_equation_l1953_195384

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((2 / x) + 3) = 5 / 3 → x = -9 := by
  sorry

end solve_equation_l1953_195384


namespace roots_equation_l1953_195372

theorem roots_equation (A B a b c d : ℝ) : 
  (a^2 + A*a + 1 = 0) → 
  (b^2 + A*b + 1 = 0) → 
  (c^2 + B*c + 1 = 0) → 
  (d^2 + B*d + 1 = 0) → 
  (a - c)*(b - c)*(a + d)*(b + d) = B^2 - A^2 := by
  sorry

end roots_equation_l1953_195372


namespace denny_followers_after_one_year_l1953_195327

/-- Calculates the number of followers after one year --/
def followers_after_one_year (initial_followers : ℕ) (daily_new_followers : ℕ) (unfollows_per_year : ℕ) : ℕ :=
  initial_followers + daily_new_followers * 365 - unfollows_per_year

/-- Theorem stating that Denny will have 445,000 followers after one year --/
theorem denny_followers_after_one_year :
  followers_after_one_year 100000 1000 20000 = 445000 := by
  sorry

#eval followers_after_one_year 100000 1000 20000

end denny_followers_after_one_year_l1953_195327


namespace coprime_divisibility_implies_one_l1953_195348

theorem coprime_divisibility_implies_one (a b c : ℕ+) :
  Nat.Coprime a.val b.val →
  Nat.Coprime a.val c.val →
  Nat.Coprime b.val c.val →
  a.val^2 ∣ (b.val^3 + c.val^3) →
  b.val^2 ∣ (a.val^3 + c.val^3) →
  c.val^2 ∣ (a.val^3 + b.val^3) →
  a = 1 ∧ b = 1 ∧ c = 1 := by
sorry

end coprime_divisibility_implies_one_l1953_195348


namespace solve_linear_equation_l1953_195358

theorem solve_linear_equation :
  ∃! x : ℚ, 3 * x - 5 = 8 ∧ x = 13 / 3 := by
  sorry

end solve_linear_equation_l1953_195358


namespace two_numbers_difference_l1953_195339

theorem two_numbers_difference (a b : ℕ) 
  (h1 : a + b = 12390)
  (h2 : b = 2 * a + 18) : 
  b - a = 4142 := by
sorry

end two_numbers_difference_l1953_195339


namespace sword_length_difference_main_result_l1953_195344

/-- Proves that Jameson's sword is 3 inches longer than twice Christopher's sword length -/
theorem sword_length_difference : ℕ → ℕ → ℕ → Prop :=
  fun christopher_length june_christopher_diff jameson_june_diff =>
    let christopher_length : ℕ := 15
    let june_christopher_diff : ℕ := 23
    let jameson_june_diff : ℕ := 5
    let june_length : ℕ := christopher_length + june_christopher_diff
    let jameson_length : ℕ := june_length - jameson_june_diff
    let twice_christopher_length : ℕ := 2 * christopher_length
    jameson_length - twice_christopher_length = 3

/-- Main theorem stating the result -/
theorem main_result : sword_length_difference 15 23 5 := by
  sorry

end sword_length_difference_main_result_l1953_195344
