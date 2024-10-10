import Mathlib

namespace ewan_sequence_contains_113_l2870_287065

def ewanSequence (n : ℕ) : ℤ := 3 + 11 * (n - 1)

theorem ewan_sequence_contains_113 :
  ∃ n : ℕ, ewanSequence n = 113 ∧
  (∀ m : ℕ, ewanSequence m ≠ 111) ∧
  (∀ m : ℕ, ewanSequence m ≠ 112) ∧
  (∀ m : ℕ, ewanSequence m ≠ 110) ∧
  (∀ m : ℕ, ewanSequence m ≠ 114) :=
by sorry


end ewan_sequence_contains_113_l2870_287065


namespace number_above_200_is_91_l2870_287067

/-- Represents the array where the k-th row contains the first 2k natural numbers -/
def array_sum (k : ℕ) : ℕ := k * (2 * k + 1) / 2

/-- The row number in which 200 is located -/
def row_of_200 : ℕ := 14

/-- The starting number of the row containing 200 -/
def start_of_row_200 : ℕ := array_sum (row_of_200 - 1) + 1

/-- The position of 200 in its row -/
def position_of_200 : ℕ := 200 - start_of_row_200 + 1

/-- The number directly above 200 -/
def number_above_200 : ℕ := array_sum (row_of_200 - 1)

theorem number_above_200_is_91 : number_above_200 = 91 := by
  sorry

end number_above_200_is_91_l2870_287067


namespace A_equals_B_l2870_287080

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

theorem A_equals_B : A = B := by
  sorry

end A_equals_B_l2870_287080


namespace sphere_diameter_triple_volume_l2870_287032

theorem sphere_diameter_triple_volume (π : ℝ) (h_π : π > 0) : 
  let r₁ : ℝ := 6
  let v₁ : ℝ := (4 / 3) * π * r₁^3
  let v₂ : ℝ := 3 * v₁
  let r₂ : ℝ := (v₂ * 3 / (4 * π))^(1/3)
  let d₂ : ℝ := 2 * r₂
  d₂ = 18 * (12 : ℝ)^(1/3) :=
by sorry

end sphere_diameter_triple_volume_l2870_287032


namespace sum_equals_ten_l2870_287014

theorem sum_equals_ten (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 + (x + y)^3 + 30*x*y = 2000) : x + y = 10 := by
  sorry

end sum_equals_ten_l2870_287014


namespace smallest_y_for_inequality_l2870_287075

theorem smallest_y_for_inequality : ∃ y : ℕ, (∀ z : ℕ, 27^z > 3^24 → y ≤ z) ∧ 27^y > 3^24 := by
  sorry

end smallest_y_for_inequality_l2870_287075


namespace rod_cutting_l2870_287025

theorem rod_cutting (rod_length piece_length : ℝ) (h1 : rod_length = 42.5) (h2 : piece_length = 0.85) :
  ⌊rod_length / piece_length⌋ = 50 := by
sorry

end rod_cutting_l2870_287025


namespace employee_salary_problem_l2870_287087

theorem employee_salary_problem (num_employees : ℕ) (salary_increase : ℕ) (manager_salary : ℕ) :
  num_employees = 24 →
  salary_increase = 400 →
  manager_salary = 11500 →
  ∃ (avg_salary : ℕ),
    avg_salary * num_employees + manager_salary = (avg_salary + salary_increase) * (num_employees + 1) ∧
    avg_salary = 1500 := by
  sorry

end employee_salary_problem_l2870_287087


namespace junior_score_l2870_287020

theorem junior_score (n : ℝ) (junior_score : ℝ) : 
  n > 0 →
  (0.3 * n * junior_score + 0.7 * n * 75) / n = 78 →
  junior_score = 85 := by
sorry

end junior_score_l2870_287020


namespace two_real_roots_implies_m_geq_one_l2870_287047

theorem two_real_roots_implies_m_geq_one (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x + 5)^2 = m - 1 ∧ (y + 5)^2 = m - 1) →
  m ≥ 1 := by
sorry

end two_real_roots_implies_m_geq_one_l2870_287047


namespace right_triangle_ratio_minimum_l2870_287072

theorem right_triangle_ratio_minimum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  c / (a + b) ≥ Real.sqrt 2 / 2 ∧ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 = z^2 ∧ z / (x + y) = Real.sqrt 2 / 2 :=
sorry

end right_triangle_ratio_minimum_l2870_287072


namespace all_statements_valid_l2870_287056

/-- Represents a simple programming language statement --/
inductive Statement
  | Assignment (var : String) (value : Int)
  | MultiAssignment (vars : List String) (values : List Int)
  | Input (prompt : Option String) (var : String)
  | Print (prompt : Option String) (expr : Option String)

/-- Checks if a statement is valid according to our rules --/
def isValid : Statement → Bool
  | Statement.Assignment _ _ => true
  | Statement.MultiAssignment vars values => vars.length == values.length
  | Statement.Input _ _ => true
  | Statement.Print _ _ => true

/-- The set of corrected statements --/
def correctedStatements : List Statement := [
  Statement.MultiAssignment ["A", "B"] [50, 50],
  Statement.MultiAssignment ["x", "y", "z"] [1, 2, 3],
  Statement.Input (some "How old are you?") "x",
  Statement.Input none "x",
  Statement.Print (some "A+B=") (some "C"),
  Statement.Print (some "Good-bye!") none
]

theorem all_statements_valid : ∀ s ∈ correctedStatements, isValid s := by sorry

end all_statements_valid_l2870_287056


namespace inscribed_circle_radius_l2870_287051

-- Define the rhombus
structure Rhombus where
  d1 : ℝ  -- First diagonal
  d2 : ℝ  -- Second diagonal
  area : ℝ  -- Area of the rhombus

-- Define the theorem
theorem inscribed_circle_radius (r : Rhombus) (h1 : r.d1 = 8) (h2 : r.d2 = 30) (h3 : r.area = 120) :
  let side := Real.sqrt ((r.d1/2)^2 + (r.d2/2)^2)
  let radius := r.area / (2 * side)
  radius = 60 / Real.sqrt 241 := by
  sorry

end inscribed_circle_radius_l2870_287051


namespace quadratic_inequality_roots_l2870_287010

theorem quadratic_inequality_roots (a : ℝ) :
  (∀ x, x < -4 ∨ x > 5 → x^2 + a*x + 20 > 0) →
  (∀ x, -4 ≤ x ∧ x ≤ 5 → x^2 + a*x + 20 ≤ 0) →
  a = -1 := by
  sorry

end quadratic_inequality_roots_l2870_287010


namespace decimal_to_fraction_l2870_287069

theorem decimal_to_fraction : 
  ∀ (n : ℕ), (3 : ℚ) / 10 + (24 : ℚ) / (99 * 10^n) = 19 / 33 := by
  sorry

end decimal_to_fraction_l2870_287069


namespace amanda_candy_problem_l2870_287008

/-- The number of candy bars Amanda gave to her sister the first time -/
def first_given : ℕ := sorry

/-- The initial number of candy bars Amanda had -/
def initial_candy : ℕ := 7

/-- The number of candy bars Amanda bought -/
def bought_candy : ℕ := 30

/-- The number of candy bars Amanda kept for herself -/
def kept_candy : ℕ := 22

theorem amanda_candy_problem :
  first_given = 3 ∧
  initial_candy - first_given + bought_candy - 4 * first_given = kept_candy :=
sorry

end amanda_candy_problem_l2870_287008


namespace triangle_height_l2870_287000

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) :
  area = 36 →
  base = 8 →
  area = (base * height) / 2 →
  height = 9 :=
by
  sorry

end triangle_height_l2870_287000


namespace amount_after_two_years_l2870_287034

/-- Calculates the amount after n years given an initial amount and annual increase rate -/
def amount_after_years (initial_amount : ℝ) (increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + increase_rate) ^ years

/-- Theorem stating that an initial amount of 1600 increasing by 1/8 annually becomes 2025 after 2 years -/
theorem amount_after_two_years :
  let initial_amount : ℝ := 1600
  let increase_rate : ℝ := 1/8
  let years : ℕ := 2
  amount_after_years initial_amount increase_rate years = 2025 := by
  sorry


end amount_after_two_years_l2870_287034


namespace altitude_inradius_sum_implies_equilateral_l2870_287091

/-- A triangle with side lengths a, b, c, altitudes h₁, h₂, h₃, and inradius r. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  r : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_h₁ : 0 < h₁
  pos_h₂ : 0 < h₂
  pos_h₃ : 0 < h₃
  pos_r : 0 < r
  altitude_sum : h₁ + h₂ + h₃ = 9 * r

/-- A triangle is equilateral if all its sides are equal. -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- If the altitudes and the radius of the inscribed circle of a triangle satisfy
    h₁ + h₂ + h₃ = 9r, then the triangle is equilateral. -/
theorem altitude_inradius_sum_implies_equilateral (t : Triangle) :
  t.isEquilateral :=
sorry

end altitude_inradius_sum_implies_equilateral_l2870_287091


namespace rect_to_polar_8_8_l2870_287057

/-- Conversion from rectangular to polar coordinates -/
theorem rect_to_polar_8_8 :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 8 * Real.sqrt 2 ∧ θ = π / 4 ∧
  8 = r * Real.cos θ ∧ 8 = r * Real.sin θ := by
  sorry

end rect_to_polar_8_8_l2870_287057


namespace sin_160_equals_sin_20_l2870_287007

theorem sin_160_equals_sin_20 : Real.sin (160 * π / 180) = Real.sin (20 * π / 180) := by
  sorry

end sin_160_equals_sin_20_l2870_287007


namespace mrs_hilt_marbles_l2870_287048

/-- The number of marbles Mrs. Hilt lost -/
def marbles_lost : ℕ := 15

/-- The number of marbles Mrs. Hilt has left -/
def marbles_left : ℕ := 23

/-- The initial number of marbles Mrs. Hilt had -/
def initial_marbles : ℕ := marbles_lost + marbles_left

theorem mrs_hilt_marbles : initial_marbles = 38 := by
  sorry

end mrs_hilt_marbles_l2870_287048


namespace subtraction_of_decimals_l2870_287049

theorem subtraction_of_decimals : 5.75 - 1.46 = 4.29 := by
  sorry

end subtraction_of_decimals_l2870_287049


namespace prime_pair_divisibility_l2870_287088

theorem prime_pair_divisibility (n p : ℕ+) : 
  Nat.Prime p.val ∧ 
  n.val ≤ 2 * p.val ∧ 
  (n.val^(p.val - 1) ∣ (p.val - 1)^n.val + 1) → 
  ((n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
sorry

end prime_pair_divisibility_l2870_287088


namespace square_division_theorem_l2870_287006

theorem square_division_theorem (S : ℝ) (h : S > 0) :
  ∀ (squares : Finset (ℝ × ℝ)),
  (∀ (s : ℝ × ℝ), s ∈ squares → s.1 = s.2) →
  squares.card = 9 →
  (∀ (s : ℝ × ℝ), s ∈ squares → s.1 ≤ S ∧ s.2 ≤ S) →
  (∀ (s₁ s₂ : ℝ × ℝ), s₁ ∈ squares → s₂ ∈ squares → s₁ ≠ s₂ → 
    (s₁.1 ≠ s₂.1 ∨ s₁.2 ≠ s₂.2)) →
  ∃ (s₁ s₂ : ℝ × ℝ), s₁ ∈ squares ∧ s₂ ∈ squares ∧ s₁ ≠ s₂ ∧ s₁.1 = s₂.1 := by
sorry


end square_division_theorem_l2870_287006


namespace compound_molecular_weight_l2870_287079

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (n_count : ℕ) (h_count : ℕ) (br_count : ℕ) 
  (n_weight : ℝ) (h_weight : ℝ) (br_weight : ℝ) : ℝ :=
  n_count * n_weight + h_count * h_weight + br_count * br_weight

/-- The molecular weight of a compound with 1 N, 4 H, and 1 Br atom is 97.95 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 1 4 1 14.01 1.01 79.90 = 97.95 := by
  sorry

end compound_molecular_weight_l2870_287079


namespace starting_player_wins_l2870_287019

/-- Represents a game state --/
structure GameState where
  current : ℕ
  isStartingPlayerTurn : Bool

/-- Represents a valid move in the game --/
def ValidMove (state : GameState) (move : ℕ) : Prop :=
  0 < move ∧ move < state.current

/-- Represents the winning condition of the game --/
def IsWinningState (state : GameState) : Prop :=
  state.current = 1987

/-- Represents a winning strategy for the starting player --/
def WinningStrategy : Type :=
  (state : GameState) → {move : ℕ // ValidMove state move}

/-- The theorem stating that the starting player has a winning strategy --/
theorem starting_player_wins :
  ∃ (strategy : WinningStrategy),
    ∀ (game : ℕ → GameState),
      game 0 = ⟨2, true⟩ →
      (∀ n, game (n + 1) = 
        let move := (strategy (game n)).val
        ⟨(game n).current + move, ¬(game n).isStartingPlayerTurn⟩) →
      ∃ n, IsWinningState (game n) ∧ (game n).isStartingPlayerTurn :=
sorry


end starting_player_wins_l2870_287019


namespace cattle_market_problem_l2870_287041

/-- The number of animals each person brought to the market satisfies the given conditions --/
theorem cattle_market_problem (j h d : ℕ) : 
  (j + 5 = 2 * (h - 5)) →  -- Condition 1
  (h + 13 = 3 * (d - 13)) →  -- Condition 2
  (d + 3 = 6 * (j - 3)) →  -- Condition 3
  j = 7 ∧ h = 11 ∧ d = 21 := by
sorry

end cattle_market_problem_l2870_287041


namespace seal_releases_three_songs_per_month_l2870_287090

/-- Represents the earnings per song in dollars -/
def earnings_per_song : ℕ := 2000

/-- Represents the total earnings in the first 3 years in dollars -/
def total_earnings : ℕ := 216000

/-- Represents the number of months in 3 years -/
def months_in_three_years : ℕ := 3 * 12

/-- Represents the number of songs released per month -/
def songs_per_month : ℕ := total_earnings / earnings_per_song / months_in_three_years

theorem seal_releases_three_songs_per_month :
  songs_per_month = 3 :=
by sorry

end seal_releases_three_songs_per_month_l2870_287090


namespace percentage_increase_johns_raise_l2870_287078

theorem percentage_increase (original new : ℝ) (h1 : original > 0) (h2 : new > original) :
  (new - original) / original * 100 = 100 ↔ new = 2 * original :=
by sorry

theorem johns_raise :
  let original : ℝ := 40
  let new : ℝ := 80
  (new - original) / original * 100 = 100 :=
by sorry

end percentage_increase_johns_raise_l2870_287078


namespace max_sum_perfect_square_fraction_l2870_287095

def is_perfect_square (n : ℚ) : Prop := ∃ m : ℕ, n = (m : ℚ) ^ 2

def is_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

theorem max_sum_perfect_square_fraction :
  ∀ A B C D : ℕ,
    is_digit A → is_digit B → is_digit C → is_digit D →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    is_perfect_square ((A + B : ℚ) / (C + D)) →
    ∀ A' B' C' D' : ℕ,
      is_digit A' → is_digit B' → is_digit C' → is_digit D' →
      A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
      is_perfect_square ((A' + B' : ℚ) / (C' + D')) →
      (A + B : ℚ) / (C + D) ≥ (A' + B' : ℚ) / (C' + D') →
      A + B = 16 :=
by sorry

end max_sum_perfect_square_fraction_l2870_287095


namespace min_distance_curve_line_l2870_287009

theorem min_distance_curve_line (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) :
  ∃ (x y : ℝ), (a - c)^2 + (b - d)^2 ≥ (x - y)^2 ∧ (x - y)^2 = 1 :=
sorry

end min_distance_curve_line_l2870_287009


namespace perpendicular_slope_l2870_287023

theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 6 * y = 12) →
  (∃ m : ℝ, m = -3/2 ∧ m * (2/3) = -1) :=
by
  sorry

end perpendicular_slope_l2870_287023


namespace negative_roots_quadratic_l2870_287096

/-- For a quadratic polynomial x^2 + 2(p+1)x + 9p - 5, both roots are negative if and only if 5/9 < p ≤ 1 or p ≥ 6 -/
theorem negative_roots_quadratic (p : ℝ) : 
  (∀ x : ℝ, x^2 + 2*(p+1)*x + 9*p - 5 = 0 → x < 0) ↔ 
  (5/9 < p ∧ p ≤ 1) ∨ p ≥ 6 :=
by sorry

end negative_roots_quadratic_l2870_287096


namespace max_dot_product_on_ellipse_l2870_287031

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

/-- Definition of the center O -/
def O : ℝ × ℝ := (0, 0)

/-- Definition of the left focus F -/
def F : ℝ × ℝ := (-1, 0)

/-- Definition of the dot product of OP and FP -/
def dot_product (x y : ℝ) : ℝ := x^2 + x + y^2

theorem max_dot_product_on_ellipse :
  ∃ (max : ℝ), max = 6 ∧
  ∀ (x y : ℝ), is_on_ellipse x y →
  dot_product x y ≤ max :=
sorry

end max_dot_product_on_ellipse_l2870_287031


namespace intersection_of_A_and_B_l2870_287038

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sqrt (x + 1)}
def B : Set ℝ := {x | 1 / (x + 1) < 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | x > 0} := by sorry

end intersection_of_A_and_B_l2870_287038


namespace factor_implies_coefficient_value_l2870_287084

/-- Given a polynomial Q(x) = x^4 + 3x^3 + ax^2 + 17x + 27, 
    if (x-3) is a factor of Q(x), then a = -80/3 -/
theorem factor_implies_coefficient_value (a : ℚ) : 
  let Q := fun (x : ℚ) => x^4 + 3*x^3 + a*x^2 + 17*x + 27
  (∃ (P : ℚ → ℚ), Q = fun x => P x * (x - 3)) → a = -80/3 := by
sorry

end factor_implies_coefficient_value_l2870_287084


namespace laura_five_dollar_bills_l2870_287055

/-- Represents the number of bills of each denomination in Laura's piggy bank -/
structure PiggyBank where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- The conditions of Laura's piggy bank -/
def laura_piggy_bank (pb : PiggyBank) : Prop :=
  pb.ones + pb.twos + pb.fives = 40 ∧
  pb.ones + 2 * pb.twos + 5 * pb.fives = 120 ∧
  pb.twos = 2 * pb.ones

theorem laura_five_dollar_bills :
  ∃ (pb : PiggyBank), laura_piggy_bank pb ∧ pb.fives = 16 :=
sorry

end laura_five_dollar_bills_l2870_287055


namespace domain_transformation_l2870_287060

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Icc (-1) 0

-- Define the domain of f(2x)
def domain_f_double : Set ℝ := Set.Ico 0 (1/2)

-- Theorem statement
theorem domain_transformation (h : ∀ x ∈ domain_f_plus_one, f (x + 1) = f (x + 1)) :
  ∀ x ∈ domain_f_double, f (2 * x) = f (2 * x) := by
  sorry

end domain_transformation_l2870_287060


namespace total_money_earned_l2870_287036

/-- The price per kg of fish in dollars -/
def price_per_kg : ℝ := 20

/-- The amount of fish in kg caught in the past four months -/
def past_four_months_catch : ℝ := 80

/-- The amount of fish in kg caught today -/
def today_catch : ℝ := 2 * past_four_months_catch

/-- The total amount of fish in kg caught in the past four months including today -/
def total_catch : ℝ := past_four_months_catch + today_catch

/-- Theorem: The total money earned by Erica in the past four months including today is $4800 -/
theorem total_money_earned : price_per_kg * total_catch = 4800 := by
  sorry

end total_money_earned_l2870_287036


namespace x_minus_y_equals_six_l2870_287029

theorem x_minus_y_equals_six (x y : ℚ) 
  (eq1 : 3 * x + 2 * y = 16) 
  (eq2 : x + 3 * y = 26/5) : 
  x - y = 6 := by sorry

end x_minus_y_equals_six_l2870_287029


namespace replaced_person_weight_l2870_287044

theorem replaced_person_weight
  (num_persons : ℕ)
  (avg_weight_increase : ℝ)
  (new_person_weight : ℝ) :
  num_persons = 5 →
  avg_weight_increase = 1.5 →
  new_person_weight = 72.5 →
  new_person_weight - (num_persons : ℝ) * avg_weight_increase = 65 :=
by sorry

end replaced_person_weight_l2870_287044


namespace fourth_term_of_geometric_progression_l2870_287015

def geometric_progression (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∃ r : ℝ, ∀ k : ℕ, a (k + 1) = a k * r

theorem fourth_term_of_geometric_progression (a : ℕ → ℝ) :
  geometric_progression a 3 →
  a 1 = 2^6 →
  a 2 = 2^3 →
  a 3 = 2^(3/2) →
  a 4 = 2^(3/4) :=
by sorry

end fourth_term_of_geometric_progression_l2870_287015


namespace integer_set_average_l2870_287098

theorem integer_set_average : ∀ (a b c d : ℤ),
  a < b ∧ b < c ∧ c < d →  -- Ensure the integers are different and ordered
  d = 90 →                 -- The largest integer is 90
  a ≥ 5 →                  -- The smallest integer is at least 5
  (a + b + c + d) / 4 = 27 -- The average is 27
  := by sorry

end integer_set_average_l2870_287098


namespace binary_arithmetic_theorem_l2870_287001

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

theorem binary_arithmetic_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, false, true]  -- 1011₂
  let result := [false, true, false, true, true, false, true]  -- 1011010₂
  (binary_to_nat a * binary_to_nat b + binary_to_nat c) = binary_to_nat result := by
  sorry

end binary_arithmetic_theorem_l2870_287001


namespace arithmetic_sequence_sum_l2870_287063

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Theorem: Given S_3 = 2 and S_6 = 8, then S_9 = 18 -/
theorem arithmetic_sequence_sum 
  (seq : ArithmeticSequence) 
  (h1 : seq.S 3 = 2) 
  (h2 : seq.S 6 = 8) : 
  seq.S 9 = 18 := by
  sorry

end arithmetic_sequence_sum_l2870_287063


namespace ab_length_is_eleven_l2870_287045

-- Define the triangle structures
structure Triangle :=
  (a b c : ℝ)

-- Define isosceles property
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Define perimeter
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

theorem ab_length_is_eleven 
  (ABC CBD : Triangle)
  (ABC_isosceles : isIsosceles ABC)
  (CBD_isosceles : isIsosceles CBD)
  (CBD_perimeter : perimeter CBD = 24)
  (ABC_perimeter : perimeter ABC = 25)
  (BD_length : CBD.c = 10) :
  ABC.c = 11 := by
  sorry

end ab_length_is_eleven_l2870_287045


namespace constant_term_expansion_constant_term_is_21_l2870_287099

theorem constant_term_expansion (x : ℝ) : 
  (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7) = 2*x^7 + x^5 + 7*x^3 + 2*x^6 + x^4 + 7*x^2 + 6*x^4 + 3*x^2 + 21 :=
by sorry

theorem constant_term_is_21 : 
  ∃ p : Polynomial ℝ, (Polynomial.eval 0 p = 21 ∧ 
    ∀ x : ℝ, p.eval x = (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7)) :=
by sorry

end constant_term_expansion_constant_term_is_21_l2870_287099


namespace problem_solution_l2870_287097

theorem problem_solution (x y : ℚ) 
  (eq1 : 5 * x - 3 * y = 8) 
  (eq2 : 2 * x + 7 * y = 1) : 
  3 * (x + y + 4) = 12 := by
sorry

end problem_solution_l2870_287097


namespace no_cross_sum_2018_l2870_287086

theorem no_cross_sum_2018 (n : ℕ) (h : n ∈ Finset.range 4901) : 5 * n ≠ 2018 := by
  sorry

end no_cross_sum_2018_l2870_287086


namespace min_cube_sum_l2870_287089

theorem min_cube_sum (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 16) :
  Complex.abs (w^3 + z^3) ≥ 22 := by
  sorry

end min_cube_sum_l2870_287089


namespace laura_rental_cost_l2870_287083

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, number of days, and miles driven. -/
def rentalCost (dailyRate : ℝ) (mileageRate : ℝ) (days : ℕ) (miles : ℝ) : ℝ :=
  dailyRate * (days : ℝ) + mileageRate * miles

/-- Theorem stating that the total cost of Laura's car rental is $165. -/
theorem laura_rental_cost :
  let dailyRate : ℝ := 30
  let mileageRate : ℝ := 0.25
  let days : ℕ := 3
  let miles : ℝ := 300
  rentalCost dailyRate mileageRate days miles = 165 := by
  sorry

end laura_rental_cost_l2870_287083


namespace valid_n_values_l2870_287050

def is_valid_n (f : ℤ → ℤ) (n : ℤ) : Prop :=
  f 1 = -1 ∧ f 4 = 2 ∧ f 8 = 34 ∧ f n = n^2 - 4*n - 18

theorem valid_n_values (f : ℤ → ℤ) (n : ℤ) 
  (h : is_valid_n f n) : n = 3 ∨ n = 6 := by
  sorry

end valid_n_values_l2870_287050


namespace mary_balloon_count_l2870_287012

/-- Given that Nancy has 7 black balloons and Mary has 4 times more black balloons than Nancy,
    prove that Mary has 28 black balloons. -/
theorem mary_balloon_count :
  ∀ (nancy_balloons mary_balloons : ℕ),
    nancy_balloons = 7 →
    mary_balloons = 4 * nancy_balloons →
    mary_balloons = 28 :=
by sorry

end mary_balloon_count_l2870_287012


namespace museum_survey_visitors_l2870_287053

/-- Represents the survey results of visitors to a modern art museum --/
structure MuseumSurvey where
  total : ℕ
  enjoyed_and_understood : ℕ
  neither_enjoyed_nor_understood : ℕ

/-- The conditions of the survey --/
def survey_conditions (s : MuseumSurvey) : Prop :=
  s.neither_enjoyed_nor_understood = 110 ∧
  s.enjoyed_and_understood = (3 : ℚ) / 4 * s.total

/-- The theorem to be proved --/
theorem museum_survey_visitors (s : MuseumSurvey) :
  survey_conditions s → s.total = 440 := by
  sorry


end museum_survey_visitors_l2870_287053


namespace quadratic_inequality_l2870_287077

theorem quadratic_inequality (x : ℝ) : x^2 + 4*x - 21 > 0 ↔ x < -7 ∨ x > 3 := by
  sorry

end quadratic_inequality_l2870_287077


namespace special_calculator_input_l2870_287003

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Applies the special calculator operation to a number -/
def calculatorOperation (n : ℕ) : ℕ := reverseDigits (3 * n) + 2

theorem special_calculator_input (x : ℕ) :
  (1000 ≤ x ∧ x < 10000) →  -- x is a four-digit number
  calculatorOperation x = 2015 →
  x = 1034 := by sorry

end special_calculator_input_l2870_287003


namespace sum_of_radii_is_14_l2870_287046

-- Define the circle with center C
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the condition of being tangent to positive x- and y-axes
def tangentToAxes (c : Circle) : Prop :=
  c.center.1 = c.radius ∧ c.center.2 = c.radius

-- Define the condition of being externally tangent to another circle
def externallyTangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

-- Theorem statement
theorem sum_of_radii_is_14 :
  ∃ (c1 c2 : Circle),
    tangentToAxes c1 ∧
    tangentToAxes c2 ∧
    c1.center ≠ c2.center ∧
    externallyTangent c1 { center := (5, 0), radius := 2 } ∧
    externallyTangent c2 { center := (5, 0), radius := 2 } ∧
    c1.radius + c2.radius = 14 :=
by sorry

end sum_of_radii_is_14_l2870_287046


namespace totalChargeDifference_l2870_287092

/-- Represents the pricing structure for an air conditioner company -/
structure ACCompany where
  price : ℝ
  surchargeRate : ℝ
  installationCharge : ℝ
  warrantyFee : ℝ
  maintenanceFee : ℝ

/-- Calculates the total charge for a company -/
def totalCharge (c : ACCompany) : ℝ :=
  c.price + (c.surchargeRate * c.price) + c.installationCharge + c.warrantyFee + c.maintenanceFee

/-- Company X's pricing information -/
def companyX : ACCompany :=
  { price := 575
  , surchargeRate := 0.04
  , installationCharge := 82.50
  , warrantyFee := 125
  , maintenanceFee := 50 }

/-- Company Y's pricing information -/
def companyY : ACCompany :=
  { price := 530
  , surchargeRate := 0.03
  , installationCharge := 93.00
  , warrantyFee := 150
  , maintenanceFee := 40 }

/-- Theorem stating the difference in total charges between Company X and Company Y -/
theorem totalChargeDifference :
  totalCharge companyX - totalCharge companyY = 26.60 := by
  sorry

end totalChargeDifference_l2870_287092


namespace jack_christina_lindy_meeting_l2870_287061

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting
  (initial_distance : ℝ)
  (jack_speed : ℝ)
  (christina_speed : ℝ)
  (lindy_speed : ℝ)
  (h1 : initial_distance = 360)
  (h2 : jack_speed = 5)
  (h3 : christina_speed = 7)
  (h4 : lindy_speed = 12)
  (h5 : jack_speed > 0)
  (h6 : christina_speed > 0)
  (h7 : lindy_speed > jack_speed + christina_speed) :
  let meeting_time := initial_distance / (jack_speed + christina_speed)
  lindy_speed * meeting_time = initial_distance :=
sorry

end jack_christina_lindy_meeting_l2870_287061


namespace juniors_in_program_l2870_287066

theorem juniors_in_program (total_students : ℕ) (junior_club_percent : ℚ) (senior_club_percent : ℚ)
  (junior_senior_ratio : ℚ) (h_total : total_students = 40)
  (h_junior_percent : junior_club_percent = 3/10)
  (h_senior_percent : senior_club_percent = 1/5)
  (h_ratio : junior_senior_ratio = 3/2) :
  ∃ (juniors seniors : ℕ),
    juniors + seniors = total_students ∧
    (junior_club_percent * juniors : ℚ) / (senior_club_percent * seniors) = junior_senior_ratio ∧
    juniors = 20 := by
  sorry

end juniors_in_program_l2870_287066


namespace center_line_correct_l2870_287011

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  equation : PolarPoint → Prop

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : PolarPoint → Prop

/-- The given circle equation -/
def givenCircle : PolarCircle :=
  { equation := fun p => p.r = 4 * Real.cos p.θ + 6 * Real.sin p.θ }

/-- The line passing through the center of the circle and parallel to the polar axis -/
def centerLine : PolarLine :=
  { equation := fun p => p.r * Real.sin p.θ = 3 }

/-- Theorem stating that the centerLine is correct for the givenCircle -/
theorem center_line_correct (p : PolarPoint) : 
  givenCircle.equation p → centerLine.equation p := by
  sorry

end center_line_correct_l2870_287011


namespace value_of_x_l2870_287002

theorem value_of_x (x y z : ℚ) : x = y / 3 → y = z / 4 → z = 48 → x = 4 := by
  sorry

end value_of_x_l2870_287002


namespace mike_car_parts_cost_l2870_287094

-- Define the cost of speakers
def speaker_cost : ℚ := 118.54

-- Define the cost of tires
def tire_cost : ℚ := 106.33

-- Define the total cost
def total_cost : ℚ := speaker_cost + tire_cost

-- Theorem to prove
theorem mike_car_parts_cost : total_cost = 224.87 := by
  sorry

end mike_car_parts_cost_l2870_287094


namespace savings_theorem_l2870_287068

/-- Represents the prices of food items and meals --/
structure FoodPrices where
  burger : ℝ
  fries : ℝ
  drink : ℝ
  burgerMeal : ℝ
  kidsBurger : ℝ
  kidsFries : ℝ
  kidsJuice : ℝ
  kidsMeal : ℝ

/-- Calculates the savings when buying meals instead of individual items --/
def calculateSavings (prices : FoodPrices) : ℝ :=
  let individualCost := 
    2 * (prices.burger + prices.fries + prices.drink) +
    2 * (prices.kidsBurger + prices.kidsFries + prices.kidsJuice)
  let mealCost := 2 * prices.burgerMeal + 2 * prices.kidsMeal
  individualCost - mealCost

/-- The savings theorem --/
theorem savings_theorem (prices : FoodPrices) 
  (h1 : prices.burger = 5)
  (h2 : prices.fries = 3)
  (h3 : prices.drink = 3)
  (h4 : prices.burgerMeal = 9.5)
  (h5 : prices.kidsBurger = 3)
  (h6 : prices.kidsFries = 2)
  (h7 : prices.kidsJuice = 2)
  (h8 : prices.kidsMeal = 5) :
  calculateSavings prices = 7 := by
  sorry

end savings_theorem_l2870_287068


namespace cubic_root_product_l2870_287037

theorem cubic_root_product (a b c : ℂ) : 
  (3 * a^3 - 9 * a^2 + a - 7 = 0) ∧ 
  (3 * b^3 - 9 * b^2 + b - 7 = 0) ∧ 
  (3 * c^3 - 9 * c^2 + c - 7 = 0) →
  a * b * c = 7/3 := by
sorry

end cubic_root_product_l2870_287037


namespace dime_difference_l2870_287052

/-- Represents the types of coins in the piggy bank -/
inductive Coin
  | Nickel
  | Dime
  | Quarter

/-- Represents the piggy bank with its coin composition -/
structure PiggyBank where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  total_coins : nickels + dimes + quarters = 100
  total_value : 5 * nickels + 10 * dimes + 25 * quarters = 1005

/-- The value of a given coin in cents -/
def coinValue : Coin → ℕ
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Theorem stating the difference between max and min number of dimes -/
theorem dime_difference (pb : PiggyBank) : 
  ∃ (min_dimes max_dimes : ℕ), 
    (∀ pb' : PiggyBank, pb'.dimes ≥ min_dimes) ∧ 
    (∃ pb' : PiggyBank, pb'.dimes = min_dimes) ∧
    (∀ pb' : PiggyBank, pb'.dimes ≤ max_dimes) ∧ 
    (∃ pb' : PiggyBank, pb'.dimes = max_dimes) ∧
    max_dimes - min_dimes = 100 :=
  sorry


end dime_difference_l2870_287052


namespace no_natural_solutions_l2870_287035

theorem no_natural_solutions : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end no_natural_solutions_l2870_287035


namespace arithmetic_sequence_a7_l2870_287042

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 = 7 → a 5 = 13 → a 7 = 19 :=
by
  sorry

end arithmetic_sequence_a7_l2870_287042


namespace sixThousandthTerm_l2870_287062

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a₁ : ℝ
  -- Common difference
  d : ℝ
  -- Parameters p and r
  p : ℝ
  r : ℝ
  -- Conditions on the first four terms
  h₁ : a₁ = 2 * p
  h₂ : a₁ + d = 14
  h₃ : a₁ + 2 * d = 4 * p - r
  h₄ : a₁ + 3 * d = 4 * p + r

/-- The nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1) * seq.d

/-- Theorem stating that the 6000th term is 24006 -/
theorem sixThousandthTerm (seq : ArithmeticSequence) : nthTerm seq 6000 = 24006 := by
  sorry

end sixThousandthTerm_l2870_287062


namespace money_transfer_proof_l2870_287074

/-- The amount of money (in won) the older brother gave to the younger brother -/
def money_transferred : ℕ := sorry

/-- The initial amount of money (in won) the older brother had -/
def older_brother_initial : ℕ := 2800

/-- The initial amount of money (in won) the younger brother had -/
def younger_brother_initial : ℕ := 1500

/-- The difference in money (in won) between the brothers after the transfer -/
def final_difference : ℕ := 360

theorem money_transfer_proof :
  (older_brother_initial - money_transferred) - (younger_brother_initial + money_transferred) = final_difference ∧
  money_transferred = 470 := by sorry

end money_transfer_proof_l2870_287074


namespace right_triangle_area_l2870_287005

theorem right_triangle_area (a b c : ℝ) (ha : a = 15) (hb : b = 36) (hc : c = 39) :
  a^2 + b^2 = c^2 ∧ (1/2 * a * b = 270) := by
  sorry

end right_triangle_area_l2870_287005


namespace inequality_solution_l2870_287024

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) := by
  sorry

end inequality_solution_l2870_287024


namespace fourth_power_equals_sixteenth_l2870_287085

theorem fourth_power_equals_sixteenth (n : ℝ) : (1/4 : ℝ)^n = 0.0625 → n = 2 := by
  sorry

end fourth_power_equals_sixteenth_l2870_287085


namespace white_balls_fewest_l2870_287040

/-- Represents the number of balls of each color -/
structure BallCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- The conditions of the ball counting problem -/
def ballProblemConditions (counts : BallCounts) : Prop :=
  counts.red + counts.blue + counts.white = 108 ∧
  counts.blue = counts.red / 3 ∧
  counts.white = counts.blue / 2

theorem white_balls_fewest (counts : BallCounts) 
  (h : ballProblemConditions counts) : 
  counts.white = 12 ∧ 
  counts.white < counts.blue ∧ 
  counts.white < counts.red :=
sorry

end white_balls_fewest_l2870_287040


namespace cubic_sum_reciprocal_l2870_287027

theorem cubic_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^3 + 1/a^3 = 2 * Real.sqrt 5 ∨ a^3 + 1/a^3 = -2 * Real.sqrt 5 :=
sorry

end cubic_sum_reciprocal_l2870_287027


namespace polynomial_division_condition_l2870_287082

theorem polynomial_division_condition (a b : ℝ) : 
  (∀ x : ℝ, (x - 1)^2 ∣ (a * x^4 + b * x^3 + 1)) ↔ (a = 3 ∧ b = -4) := by
  sorry

end polynomial_division_condition_l2870_287082


namespace smug_twc_minimum_bouts_l2870_287013

theorem smug_twc_minimum_bouts (n : Nat) (h : n = 2008) :
  let total_edges := n * (n - 1) / 2
  let max_complement_edges := n^2 / 4
  total_edges - max_complement_edges = 999000 := by
  sorry

end smug_twc_minimum_bouts_l2870_287013


namespace value_set_of_m_l2870_287059

def A : Set ℝ := {x : ℝ | x^2 + 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m*x + 1 = 0}

theorem value_set_of_m : 
  ∀ m : ℝ, (A ∪ B m = A) ↔ m ∈ ({0, 1/2, 1/3} : Set ℝ) := by
  sorry

end value_set_of_m_l2870_287059


namespace hamburgers_left_over_l2870_287043

def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

theorem hamburgers_left_over : hamburgers_made - hamburgers_served = 6 := by
  sorry

end hamburgers_left_over_l2870_287043


namespace jeff_journey_distance_l2870_287033

-- Define the journey segments
def segment1_speed : ℝ := 80
def segment1_time : ℝ := 6
def segment2_speed : ℝ := 60
def segment2_time : ℝ := 4
def segment3_speed : ℝ := 40
def segment3_time : ℝ := 2

-- Define the total distance function
def total_distance : ℝ := 
  segment1_speed * segment1_time + 
  segment2_speed * segment2_time + 
  segment3_speed * segment3_time

-- Theorem statement
theorem jeff_journey_distance : total_distance = 800 := by
  sorry

end jeff_journey_distance_l2870_287033


namespace simplify_expression_1_simplify_expression_2_l2870_287018

-- Define variables
variable (a b x y : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : 3*a - 5*b - 2*a + b = a - 4*b := by sorry

-- Theorem for the second expression
theorem simplify_expression_2 : 4*x^2 + 5*x*y - 2*(2*x^2 - x*y) = 7*x*y := by sorry

end simplify_expression_1_simplify_expression_2_l2870_287018


namespace expansion_terms_count_expansion_terms_count_is_ten_l2870_287026

theorem expansion_terms_count : ℕ :=
  let expression := (fun (x y : ℝ) => ((x + 5*y)^3 * (x - 5*y)^3)^3)
  let simplified := (fun (x y : ℝ) => (x^2 - 25*y^2)^9)
  let distinct_terms_count := 10
  distinct_terms_count

#check expansion_terms_count

theorem expansion_terms_count_is_ten : expansion_terms_count = 10 := by
  sorry

end expansion_terms_count_expansion_terms_count_is_ten_l2870_287026


namespace cylinder_radius_theorem_l2870_287028

/-- The original radius of a cylinder with the given properties -/
def original_radius : ℝ := 8

/-- The original height of the cylinder -/
def original_height : ℝ := 3

/-- The increase in either radius or height -/
def increase : ℝ := 4

theorem cylinder_radius_theorem :
  (π * (original_radius + increase)^2 * original_height = 
   π * original_radius^2 * (original_height + increase)) ∧
  original_radius > 0 := by
  sorry

end cylinder_radius_theorem_l2870_287028


namespace functional_equation_unique_solution_l2870_287022

open Set

theorem functional_equation_unique_solution
  (f : ℝ → ℝ) (a b : ℝ) :
  (0 < a) →
  (0 < b) →
  (∀ x, 0 ≤ x → 0 ≤ f x) →
  (∀ x, 0 ≤ x → f (f x) + a * f x = b * (a + b) * x) →
  (∀ x, 0 ≤ x → f x = b * x) :=
sorry

end functional_equation_unique_solution_l2870_287022


namespace triangle_circumcircle_distance_sum_bounds_l2870_287076

theorem triangle_circumcircle_distance_sum_bounds :
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (2 * Real.sqrt 3, 0)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ → ℝ × ℝ := fun θ ↦ (Real.sqrt 3 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)
  ∀ θ : ℝ,
    let P := C θ
    let dist_squared (X Y : ℝ × ℝ) := (X.1 - Y.1)^2 + (X.2 - Y.2)^2
    let sum := dist_squared P O + dist_squared P A + dist_squared P B
    sum ≤ 32 ∧ sum ≥ 16 ∧ (∃ θ₁ θ₂, C θ₁ = 32 ∧ C θ₂ = 16) :=
by
  sorry


end triangle_circumcircle_distance_sum_bounds_l2870_287076


namespace cubic_diophantine_equation_l2870_287039

theorem cubic_diophantine_equation :
  ∀ x y z : ℤ, x^3 + 2*y^3 = 4*z^3 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end cubic_diophantine_equation_l2870_287039


namespace range_of_a_l2870_287071

theorem range_of_a : 
  (∀ x, 0 < x ∧ x < 1 → ∀ a, (x - a) * (x - (a + 2)) ≤ 0) ∧ 
  (∃ x a, ¬(0 < x ∧ x < 1) ∧ (x - a) * (x - (a + 2)) ≤ 0) →
  ∀ a, a ∈ Set.Icc (-1 : ℝ) 0 :=
by sorry

end range_of_a_l2870_287071


namespace alex_last_five_shots_l2870_287058

/-- Represents the number of shots made by Alex -/
structure ShotsMade where
  initial : ℕ
  after_60 : ℕ
  final : ℕ

/-- Represents the shooting percentages at different stages -/
structure ShootingPercentages where
  initial : ℚ
  after_60 : ℚ
  final : ℚ

/-- Theorem stating the number of shots Alex made in the last 5 attempts -/
theorem alex_last_five_shots 
  (shots : ShotsMade)
  (percentages : ShootingPercentages)
  (h1 : shots.initial = 30)
  (h2 : shots.after_60 = 37)
  (h3 : shots.final = 39)
  (h4 : percentages.initial = 3/5)
  (h5 : percentages.after_60 = 31/50)
  (h6 : percentages.final = 3/5) :
  shots.final - shots.after_60 = 2 := by
  sorry

end alex_last_five_shots_l2870_287058


namespace wire_circle_to_rectangle_area_l2870_287004

/-- Given a wire initially in the form of a circle with radius 3.5 m,
    when bent into a rectangle with length to breadth ratio of 6:5,
    the area of the resulting rectangle is (735 * π^2) / 242 square meters. -/
theorem wire_circle_to_rectangle_area :
  let r : ℝ := 3.5
  let circle_circumference := 2 * Real.pi * r
  let length_to_breadth_ratio : ℚ := 6 / 5
  let rectangle_perimeter := circle_circumference
  let length : ℝ := (21 * Real.pi) / 11
  let breadth : ℝ := (35 * Real.pi) / 22
  rectangle_perimeter = 2 * (length + breadth) →
  length / breadth = length_to_breadth_ratio →
  length * breadth = (735 * Real.pi^2) / 242 := by
  sorry

end wire_circle_to_rectangle_area_l2870_287004


namespace set_equality_implies_power_l2870_287017

theorem set_equality_implies_power (m n : ℝ) : 
  let P : Set ℝ := {1, m}
  let Q : Set ℝ := {2, -n}
  P = Q → m^n = (1/2 : ℝ) := by
  sorry

end set_equality_implies_power_l2870_287017


namespace constant_value_l2870_287064

theorem constant_value : ∀ (x : ℝ) (c : ℝ),
  (5 * x + c = 10 * x - 22) →
  (x = 5) →
  c = 3 := by
  sorry

end constant_value_l2870_287064


namespace mary_juan_income_ratio_l2870_287073

/-- Given that Mary's income is 60% more than Tim's income, and Tim's income is 60% less than Juan's income,
    prove that Mary's income is 64% of Juan's income. -/
theorem mary_juan_income_ratio (juan tim mary : ℝ) 
  (h1 : tim = 0.4 * juan)
  (h2 : mary = 1.6 * tim) : 
  mary = 0.64 * juan := by
  sorry

end mary_juan_income_ratio_l2870_287073


namespace point_outside_circle_l2870_287021

def imaginary_unit : ℂ := Complex.I

theorem point_outside_circle (a b : ℝ) (h : a + b * imaginary_unit = (2 + imaginary_unit) / (1 - imaginary_unit)) :
  a^2 + b^2 > 2 := by sorry

end point_outside_circle_l2870_287021


namespace dense_S_l2870_287081

-- Define the set S
def S : Set ℝ := {x : ℝ | ∃ (m n : ℕ+), x = Real.sqrt m - Real.sqrt n}

-- State the theorem
theorem dense_S : ∀ (a b : ℝ), a < b → Set.Infinite (S ∩ Set.Ioo a b) := by sorry

end dense_S_l2870_287081


namespace blueberry_tart_fraction_l2870_287016

theorem blueberry_tart_fraction (total : Real) (cherry : Real) (peach : Real)
  (h1 : total = 0.91)
  (h2 : cherry = 0.08)
  (h3 : peach = 0.08) :
  total - (cherry + peach) = 0.75 := by
sorry

end blueberry_tart_fraction_l2870_287016


namespace complex_equation_solution_l2870_287093

theorem complex_equation_solution :
  ∃ z : ℂ, (3 : ℂ) - 2 * Complex.I * z = -2 + 3 * Complex.I * z ∧ z = -Complex.I := by
  sorry

end complex_equation_solution_l2870_287093


namespace isosceles_trapezoid_area_l2870_287070

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- The perimeter of the trapezoid -/
  perimeter : ℝ
  /-- The diagonal bisects the obtuse angle -/
  diagonal_bisects_obtuse_angle : Bool

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific isosceles trapezoid is 96 -/
theorem isosceles_trapezoid_area :
  ∀ t : IsoscelesTrapezoid,
    t.shorter_base = 3 ∧
    t.perimeter = 42 ∧
    t.diagonal_bisects_obtuse_angle = true →
    area t = 96 :=
  sorry

end isosceles_trapezoid_area_l2870_287070


namespace largest_number_is_541_l2870_287054

def digits : List Nat := [1, 4, 5]

def is_valid_number (n : Nat) : Prop :=
  let digits_of_n := n.digits 10
  digits_of_n.length = 3 ∧ digits_of_n.toFinset = digits.toFinset

theorem largest_number_is_541 :
  ∀ n : Nat, is_valid_number n → n ≤ 541 :=
sorry

end largest_number_is_541_l2870_287054


namespace sqrt_5_minus_1_over_2_gt_half_l2870_287030

theorem sqrt_5_minus_1_over_2_gt_half : 
  (4 < 5) → (5 < 9) → (Real.sqrt 5 - 1) / 2 > 1 / 2 := by
  sorry

end sqrt_5_minus_1_over_2_gt_half_l2870_287030
