import Mathlib

namespace walking_speed_problem_l2241_224102

theorem walking_speed_problem (slower_speed : ℝ) (faster_speed : ℝ) 
  (actual_distance : ℝ) (total_distance : ℝ) :
  faster_speed = 20 →
  actual_distance = 20 →
  total_distance = actual_distance + 20 →
  actual_distance / slower_speed = total_distance / faster_speed →
  slower_speed = 10 := by
  sorry

end walking_speed_problem_l2241_224102


namespace dividend_calculation_l2241_224141

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 15) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 140 := by
  sorry

end dividend_calculation_l2241_224141


namespace inverse_proportion_function_l2241_224151

/-- 
If the inverse proportion function y = m/x passes through the point (m, m/8),
then the function can be expressed as y = 8/x.
-/
theorem inverse_proportion_function (m : ℝ) (h : m ≠ 0) : 
  (∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → f x = m / x) ∧ f m = m / 8) → 
  (∃ (g : ℝ → ℝ), ∀ x, x ≠ 0 → g x = 8 / x) :=
by sorry

end inverse_proportion_function_l2241_224151


namespace cartesian_equation_chord_length_l2241_224161

-- Define the polar equation of curve C
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ)^2 = 4 * Real.cos θ

-- Define the parametric equations of line l
def line_equation (t x y : ℝ) : Prop :=
  x = 2 + (1/2) * t ∧ y = (Real.sqrt 3 / 2) * t

-- Theorem for the Cartesian equation of curve C
theorem cartesian_equation (x y : ℝ) :
  (∃ ρ θ, polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  y^2 = 4*x :=
sorry

-- Theorem for the length of chord AB
theorem chord_length (A B : ℝ × ℝ) :
  (∃ t₁ t₂, line_equation t₁ A.1 A.2 ∧ line_equation t₂ B.1 B.2 ∧
   A.2^2 = 4*A.1 ∧ B.2^2 = 4*B.1) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 7 / 3 :=
sorry

end cartesian_equation_chord_length_l2241_224161


namespace problem_solution_l2241_224131

/-- The set of integers of the form m^k for integers m, k ≥ 2 -/
def S : Set ℕ := {n : ℕ | ∃ m k : ℕ, m ≥ 2 ∧ k ≥ 2 ∧ n = m^k}

/-- The number of ways to write n as the sum of distinct elements of S -/
def f (n : ℕ) : ℕ := sorry

/-- The set of integers for which f(n) = 3 -/
def T : Set ℕ := {n : ℕ | f n = 3}

theorem problem_solution :
  (f 30 = 0) ∧
  (∀ n : ℕ, n ≥ 31 → f n ≥ 1) ∧
  (T.Finite ∧ T.Nonempty) ∧
  (∃ m : ℕ, m ∈ T ∧ ∀ n ∈ T, n ≤ m) ∧
  (∃ m : ℕ, m ∈ T ∧ ∀ n ∈ T, n ≤ m ∧ m = 111) :=
by sorry

end problem_solution_l2241_224131


namespace both_are_dwarves_l2241_224150

-- Define the types of inhabitants
inductive Inhabitant : Type
| Elf : Inhabitant
| Dwarf : Inhabitant

-- Define the types of statements
inductive Statement : Type
| AboutGold : Statement
| AboutDwarf : Statement
| Other : Statement

-- Define the truth value of a statement given the speaker and the statement type
def isTruthful (speaker : Inhabitant) (statement : Statement) : Prop :=
  match speaker, statement with
  | Inhabitant.Dwarf, Statement.AboutGold => False
  | Inhabitant.Elf, Statement.AboutDwarf => False
  | _, _ => True

-- Define A's statement
def A_statement : Statement := Statement.AboutGold

-- Define B's statement about A
def B_statement (A_type : Inhabitant) : Statement :=
  match A_type with
  | Inhabitant.Dwarf => Statement.Other
  | Inhabitant.Elf => Statement.AboutDwarf

-- Theorem to prove
theorem both_are_dwarves :
  ∃ (A_type B_type : Inhabitant),
    A_type = Inhabitant.Dwarf ∧
    B_type = Inhabitant.Dwarf ∧
    isTruthful A_type A_statement = False ∧
    isTruthful B_type (B_statement A_type) = True :=
  sorry


end both_are_dwarves_l2241_224150


namespace vector_subtraction_l2241_224146

/-- Given two vectors OM and ON in R², prove that MN = ON - OM -/
theorem vector_subtraction (OM ON : ℝ × ℝ) (h1 : OM = (3, -2)) (h2 : ON = (-5, -1)) :
  ON - OM = (-8, 1) := by
  sorry

end vector_subtraction_l2241_224146


namespace arthurs_spending_l2241_224125

/-- The cost of Arthur's purchase on the first day -/
def arthurs_first_day_cost (hamburger_price hot_dog_price : ℝ) : ℝ :=
  3 * hamburger_price + 4 * hot_dog_price

/-- The cost of Arthur's purchase on the second day -/
def arthurs_second_day_cost (hamburger_price hot_dog_price : ℝ) : ℝ :=
  2 * hamburger_price + 3 * hot_dog_price

theorem arthurs_spending : 
  ∀ (hamburger_price : ℝ),
    arthurs_second_day_cost hamburger_price 1 = 7 →
    arthurs_first_day_cost hamburger_price 1 = 10 := by
  sorry

end arthurs_spending_l2241_224125


namespace design_area_is_16_l2241_224104

/-- A point on a 2D grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A right-angled triangle on a grid --/
structure RightTriangle where
  vertex1 : GridPoint
  vertex2 : GridPoint
  vertex3 : GridPoint

/-- The design formed by two right-angled triangles --/
structure Design where
  triangle1 : RightTriangle
  triangle2 : RightTriangle

/-- Function to calculate the area of a right-angled triangle using Pick's theorem --/
def triangleArea (t : RightTriangle) : ℕ := sorry

/-- Function to check if a design is symmetrical about the diagonal --/
def isSymmetrical (d : Design) : Prop := sorry

/-- The main theorem --/
theorem design_area_is_16 (d : Design) :
  d.triangle1.vertex1 = ⟨0, 0⟩ ∧
  d.triangle1.vertex2 = ⟨4, 0⟩ ∧
  d.triangle1.vertex3 = ⟨0, 4⟩ ∧
  d.triangle2.vertex1 = ⟨4, 0⟩ ∧
  d.triangle2.vertex2 = ⟨4, 4⟩ ∧
  d.triangle2.vertex3 = ⟨0, 4⟩ ∧
  isSymmetrical d →
  triangleArea d.triangle1 + triangleArea d.triangle2 = 16 := by
  sorry

end design_area_is_16_l2241_224104


namespace set_intersection_and_union_l2241_224157

theorem set_intersection_and_union (a : ℝ) : 
  let A : Set ℝ := {2, 3, a^2 + 4*a + 2}
  let B : Set ℝ := {0, 7, 2 - a, a^2 + 4*a - 2}
  (A ∩ B = {3, 7}) → 
  (a = 1 ∧ A ∪ B = {0, 1, 2, 3, 7}) :=
by sorry

end set_intersection_and_union_l2241_224157


namespace inequality_proof_l2241_224190

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧ ((a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2) := by
  sorry

end inequality_proof_l2241_224190


namespace least_difference_l2241_224144

theorem least_difference (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  y - x > 3 ∧
  Even x ∧ Odd y ∧ Odd z →
  ∀ w, w = z - x → w ≥ 7 := by
sorry

end least_difference_l2241_224144


namespace chess_club_mixed_groups_l2241_224105

/-- Represents the chess club structure and game information -/
structure ChessClub where
  total_children : ℕ
  total_groups : ℕ
  children_per_group : ℕ
  boy_boy_games : ℕ
  girl_girl_games : ℕ

/-- Calculates the number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ :=
  let total_games := club.total_groups * (club.children_per_group.choose 2)
  let mixed_games := total_games - club.boy_boy_games - club.girl_girl_games
  mixed_games / 2

/-- Theorem stating the number of mixed groups in the given scenario -/
theorem chess_club_mixed_groups :
  let club : ChessClub := {
    total_children := 90,
    total_groups := 30,
    children_per_group := 3,
    boy_boy_games := 30,
    girl_girl_games := 14
  }
  mixed_groups club = 23 := by sorry

end chess_club_mixed_groups_l2241_224105


namespace polynomial_factor_implies_t_value_l2241_224165

theorem polynomial_factor_implies_t_value :
  ∀ t : ℤ,
  (∃ a b : ℤ, ∀ x : ℤ, x^3 - x^2 - 7*x + t = (x + 1) * (x^2 + a*x + b)) →
  t = -5 :=
by
  sorry

end polynomial_factor_implies_t_value_l2241_224165


namespace arithmetic_sequence_12th_term_l2241_224116

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end arithmetic_sequence_12th_term_l2241_224116


namespace tan_half_angle_less_than_one_l2241_224194

theorem tan_half_angle_less_than_one (θ : Real) (h : 0 < θ ∧ θ < π / 2) : 
  Real.tan (θ / 2) < 1 := by
  sorry

end tan_half_angle_less_than_one_l2241_224194


namespace sally_seashell_money_l2241_224193

/-- The number of seashells Sally picks on Monday -/
def monday_seashells : ℕ := 30

/-- The number of seashells Sally picks on Tuesday -/
def tuesday_seashells : ℕ := monday_seashells / 2

/-- The price of each seashell in dollars -/
def seashell_price : ℚ := 6/5

/-- The total number of seashells Sally picks -/
def total_seashells : ℕ := monday_seashells + tuesday_seashells

/-- The total money Sally can make by selling all her seashells -/
def total_money : ℚ := (total_seashells : ℚ) * seashell_price

theorem sally_seashell_money : total_money = 54 := by
  sorry

end sally_seashell_money_l2241_224193


namespace angle_A_is_pi_over_six_l2241_224126

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem angle_A_is_pi_over_six (t : Triangle) :
  (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A = Real.sqrt 3 * t.a * Real.cos t.C →
  t.A = π / 6 :=
by sorry

end angle_A_is_pi_over_six_l2241_224126


namespace total_length_theorem_l2241_224118

/-- Calculates the total length of ladders climbed by two workers in centimeters -/
def total_length_climbed (keaton_ladder_height : ℕ) (keaton_climbs : ℕ) 
  (reece_ladder_diff : ℕ) (reece_climbs : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - reece_ladder_diff
  let keaton_total := keaton_ladder_height * keaton_climbs
  let reece_total := reece_ladder_height * reece_climbs
  (keaton_total + reece_total) * 100

/-- The total length of ladders climbed by both workers is 422000 centimeters -/
theorem total_length_theorem : 
  total_length_climbed 60 40 8 35 = 422000 := by
  sorry

end total_length_theorem_l2241_224118


namespace arrangement_count_l2241_224115

theorem arrangement_count :
  let total_men : ℕ := 4
  let total_women : ℕ := 5
  let group_of_four_men : ℕ := 2
  let group_of_four_women : ℕ := 2
  let remaining_men : ℕ := total_men - group_of_four_men
  let remaining_women : ℕ := total_women - group_of_four_women
  (Nat.choose total_men group_of_four_men) *
  (Nat.choose total_women group_of_four_women) *
  (Nat.choose remaining_women remaining_men) = 180 :=
by sorry

end arrangement_count_l2241_224115


namespace perpendicular_line_equation_l2241_224156

-- Define the given line
def given_line (x y : ℝ) : Prop := x + y - 5 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (2, -1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - y - 3 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  ∃ (m b : ℝ),
    (∀ x y, perpendicular_line x y ↔ y = m * x + b) ∧
    perpendicular_line point.1 point.2 ∧
    (∀ x₁ y₁ x₂ y₂, given_line x₁ y₁ → given_line x₂ y₂ → x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * m = -1) :=
sorry

end perpendicular_line_equation_l2241_224156


namespace quadratic_root_on_line_l2241_224171

/-- A root of a quadratic equation lies on a corresponding line in the p-q plane. -/
theorem quadratic_root_on_line (p q x₀ : ℝ) : 
  x₀^2 + p * x₀ + q = 0 → q = -x₀ * p - x₀^2 := by
  sorry

end quadratic_root_on_line_l2241_224171


namespace gcd_102_238_l2241_224176

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l2241_224176


namespace min_perimeter_triangle_AOB_l2241_224168

/-- The minimum perimeter of triangle AOB given the conditions -/
theorem min_perimeter_triangle_AOB :
  let P : ℝ × ℝ := (4, 2)
  let O : ℝ × ℝ := (0, 0)
  ∃ (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)),
    (P ∈ l) ∧
    (A.1 > 0 ∧ A.2 = 0) ∧
    (B.1 = 0 ∧ B.2 > 0) ∧
    (A ∈ l) ∧ (B ∈ l) ∧
    (∀ (A' B' : ℝ × ℝ) (l' : Set (ℝ × ℝ)),
      (P ∈ l') ∧
      (A'.1 > 0 ∧ A'.2 = 0) ∧
      (B'.1 = 0 ∧ B'.2 > 0) ∧
      (A' ∈ l') ∧ (B' ∈ l') →
      dist O A + dist O B + dist A B ≤ dist O A' + dist O B' + dist A' B') ∧
    (dist O A + dist O B + dist A B = 20) :=
by sorry


end min_perimeter_triangle_AOB_l2241_224168


namespace algebraic_expression_value_l2241_224184

theorem algebraic_expression_value (x : ℝ) : 
  4 * x^2 - 2 * x + 3 = 11 → 2 * x^2 - x - 7 = -3 := by
  sorry

end algebraic_expression_value_l2241_224184


namespace matrix_determinant_solution_l2241_224107

theorem matrix_determinant_solution (b : ℝ) (hb : b ≠ 0) :
  let y : ℝ := -b / 2
  ∃ (y : ℝ), Matrix.det
    ![![y + b, y, y],
      ![y, y + b, y],
      ![y, y, y + b]] = 0 ↔ y = -b / 2 := by
sorry

end matrix_determinant_solution_l2241_224107


namespace expression_perfect_square_iff_A_specific_values_l2241_224195

/-- A monomial is a term of the form cx^n where c is a constant and n is a non-negative integer. -/
def Monomial (x : ℝ) := ℝ → ℝ

/-- The expression x^6 + x^4 + xA -/
def Expression (x : ℝ) (A : Monomial x) : ℝ := x^6 + x^4 + x * A x

/-- A perfect square is a number that is the square of an integer. -/
def IsPerfectSquare (n : ℝ) : Prop := ∃ m : ℝ, n = m^2

theorem expression_perfect_square_iff_A_specific_values (x : ℝ) (A : Monomial x) :
  IsPerfectSquare (Expression x A) ↔ 
  (A = λ x => 2 * x^4) ∨ 
  (A = λ x => -2 * x^4) ∨ 
  (A = λ x => (1/4) * x^7) ∨ 
  (A = λ x => (1/4) * x) :=
sorry

end expression_perfect_square_iff_A_specific_values_l2241_224195


namespace modulus_product_complex_l2241_224140

theorem modulus_product_complex : |(7 - 4*I)*(3 + 11*I)| = Real.sqrt 8450 := by
  sorry

end modulus_product_complex_l2241_224140


namespace fraction_value_decreases_as_denominator_increases_l2241_224110

theorem fraction_value_decreases_as_denominator_increases 
  (ability : ℝ) (self_estimation : ℝ → ℝ) :
  ability > 0 → (∀ x y, x > 0 ∧ y > 0 ∧ x < y → self_estimation x > self_estimation y) →
  ∀ x y, x > 0 ∧ y > 0 ∧ x < y → ability / self_estimation x > ability / self_estimation y :=
sorry

end fraction_value_decreases_as_denominator_increases_l2241_224110


namespace y_intercept_of_line_l2241_224185

/-- The y-intercept of the line 3x - 5y = 15 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 5 * y = 15 → x = 0 → y = -3 := by
  sorry

end y_intercept_of_line_l2241_224185


namespace youtube_video_dislikes_l2241_224158

theorem youtube_video_dislikes :
  let initial_likes : ℕ := 5000
  let initial_dislikes : ℕ := (initial_likes / 3) + 50
  let likes_increase : ℕ := 2000
  let dislikes_increase : ℕ := 400
  let new_likes : ℕ := initial_likes + likes_increase
  let new_dislikes : ℕ := initial_dislikes + dislikes_increase
  let doubled_new_likes : ℕ := 2 * new_likes
  doubled_new_likes - new_dislikes = 11983 ∧ new_dislikes = 2017 :=
by sorry


end youtube_video_dislikes_l2241_224158


namespace winner_determined_by_parity_l2241_224123

/-- Represents a player in the game -/
inductive Player
  | Anthelme
  | Brunehaut

/-- Represents the game state on an m × n chessboard -/
structure GameState (m n : ℕ) where
  kingPosition : ℕ × ℕ
  visitedSquares : Set (ℕ × ℕ)

/-- Determines the winner of the game based on the board dimensions -/
def determineWinner (m n : ℕ) : Player :=
  if m * n % 2 = 0 then Player.Anthelme else Player.Brunehaut

/-- Theorem stating that the winner is determined by the parity of m × n -/
theorem winner_determined_by_parity (m n : ℕ) :
  determineWinner m n = 
    if m * n % 2 = 0 then Player.Anthelme else Player.Brunehaut :=
by sorry

end winner_determined_by_parity_l2241_224123


namespace exam_logic_l2241_224117

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (got_all_right : Student → Prop)
variable (received_A : Student → Prop)

-- State the theorem
theorem exam_logic (s : Student) 
  (h : ∀ x, got_all_right x → received_A x) :
  ¬(received_A s) → ¬(got_all_right s) := by
sorry

end exam_logic_l2241_224117


namespace average_not_two_l2241_224192

def data : List ℝ := [1, 1, 0, 2, 4]

theorem average_not_two : 
  (data.sum / data.length) ≠ 2 := by
  sorry

end average_not_two_l2241_224192


namespace cone_sphere_ratio_l2241_224148

/-- Given a right circular cone and a sphere with the same radius,
    if the volume of the cone is two-fifths that of the sphere,
    then the ratio of the cone's altitude to twice its base radius is 4/5. -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 * π * r^2 * h) = (2 / 5 * (4 / 3 * π * r^3)) → 
  h / (2 * r) = 4 / 5 := by
  sorry

end cone_sphere_ratio_l2241_224148


namespace isosceles_triangle_side_length_l2241_224164

-- Define the triangle ABC
structure Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :=
  (A B C : α)

-- Define an isosceles triangle
def IsIsosceles {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : Prop :=
  ‖t.A - t.B‖ = ‖t.A - t.C‖

-- Define the incenter
def Incenter {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : α :=
  sorry  -- Definition of incenter omitted for brevity

-- Define the distance from a point to a line segment
def DistanceToSegment {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (P : α) (A B : α) : ℝ :=
  sorry  -- Definition of distance to segment omitted for brevity

-- Theorem statement
theorem isosceles_triangle_side_length 
  {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (t : Triangle α) (I : α) :
  IsIsosceles t →
  I = Incenter t →
  ‖t.A - I‖ = 3 →
  DistanceToSegment I t.B t.C = 2 →
  ‖t.B - t.C‖ = 4 * Real.sqrt 5 :=
sorry

end isosceles_triangle_side_length_l2241_224164


namespace sum_of_four_squares_of_five_l2241_224181

theorem sum_of_four_squares_of_five : 5^2 + 5^2 + 5^2 + 5^2 = 100 := by
  sorry

end sum_of_four_squares_of_five_l2241_224181


namespace transfer_schemes_count_l2241_224186

/-- The number of torchbearers and segments in the relay --/
def n : ℕ := 6

/-- The set of possible first torchbearers --/
inductive FirstTorchbearer
| A
| B
| C

/-- The set of possible last torchbearers --/
inductive LastTorchbearer
| A
| B

/-- A function to calculate the number of transfer schemes --/
def countTransferSchemes : ℕ :=
  let firstChoices := 3  -- A, B, or C
  let lastChoices := 2   -- A or B
  let middleArrangements := Nat.factorial (n - 2)
  firstChoices * lastChoices * middleArrangements

/-- Theorem stating that the number of transfer schemes is 96 --/
theorem transfer_schemes_count :
  countTransferSchemes = 96 := by
  sorry

end transfer_schemes_count_l2241_224186


namespace order_of_powers_l2241_224113

theorem order_of_powers : 5^56 < 31^28 ∧ 31^28 < 17^35 ∧ 17^35 < 10^51 := by
  sorry

end order_of_powers_l2241_224113


namespace benny_eggs_count_l2241_224128

def dozen : ℕ := 12

def eggs_bought (num_dozens : ℕ) : ℕ := num_dozens * dozen

theorem benny_eggs_count : eggs_bought 7 = 84 := by sorry

end benny_eggs_count_l2241_224128


namespace opposite_of_negative_one_fifth_l2241_224109

theorem opposite_of_negative_one_fifth :
  -(-(1/5 : ℚ)) = 1/5 := by sorry

end opposite_of_negative_one_fifth_l2241_224109


namespace square_side_length_l2241_224101

theorem square_side_length (area : ℚ) (side : ℚ) : 
  area = 9/16 → side^2 = area → side = 3/4 := by
  sorry

end square_side_length_l2241_224101


namespace tangent_line_and_max_value_l2241_224137

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + a*x^2 + 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x - 2

theorem tangent_line_and_max_value :
  (∀ x : ℝ, x > 0 → (3*x + f (-1) x - 4 = 0 ↔ x = 1)) ∧
  (∀ a : ℝ, a > 0 →
    (∃! x : ℝ, g a x = 0) →
    (∀ x : ℝ, Real.exp (-2) < x → x < Real.exp 1 → g a x ≤ 2 * Real.exp 2 - 3 * Real.exp 1) ∧
    (∃ x : ℝ, Real.exp (-2) < x ∧ x < Real.exp 1 ∧ g a x = 2 * Real.exp 2 - 3 * Real.exp 1)) :=
by sorry

end tangent_line_and_max_value_l2241_224137


namespace top_triangle_number_l2241_224160

/-- Represents the shape of a cell in the diagram -/
inductive Shape
| Circle
| Triangle
| Hexagon

/-- The sum of numbers in each shape -/
def sum_of_shape (s : Shape) : ℕ :=
  match s with
  | Shape.Circle => 10
  | Shape.Triangle => 15
  | Shape.Hexagon => 30

/-- The total number of cells in the diagram -/
def total_cells : ℕ := 9

/-- The set of numbers used in the diagram -/
def number_set : Finset ℕ := Finset.range 9

/-- The theorem stating the possible numbers in the top triangle -/
theorem top_triangle_number :
  ∃ (n : ℕ), n ∈ number_set ∧ n ≥ 8 ∧ n ≤ 9 ∧
  (∃ (a b : ℕ), a ∈ number_set ∧ b ∈ number_set ∧ a + b + n = sum_of_shape Shape.Triangle) :=
sorry

end top_triangle_number_l2241_224160


namespace sqrt_3600_equals_60_l2241_224175

theorem sqrt_3600_equals_60 : Real.sqrt 3600 = 60 := by
  sorry

end sqrt_3600_equals_60_l2241_224175


namespace triangle_max_area_l2241_224173

theorem triangle_max_area (a b c : ℝ) (h1 : (a + b - c) * (a + b + c) = 3 * a * b) (h2 : c = 4) :
  ∃ (S : ℝ), S = (4 : ℝ) * Real.sqrt 3 ∧ ∀ (area : ℝ), area = 1/2 * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) → area ≤ S :=
sorry

end triangle_max_area_l2241_224173


namespace number_problem_l2241_224154

theorem number_problem (N : ℝ) : 
  (1/6 : ℝ) * (2/3 : ℝ) * (3/4 : ℝ) * (5/7 : ℝ) * N = 25 → 
  (60/100 : ℝ) * N = 252 := by
  sorry

end number_problem_l2241_224154


namespace farm_animals_after_addition_l2241_224153

/-- Represents the farm with its animals -/
structure Farm :=
  (cows : ℕ)
  (pigs : ℕ)
  (goats : ℕ)

/-- Calculates the total number of animals on the farm -/
def Farm.total (f : Farm) : ℕ := f.cows + f.pigs + f.goats

/-- Adds new animals to the farm -/
def Farm.add (f : Farm) (new_cows new_pigs new_goats : ℕ) : Farm :=
  { cows := f.cows + new_cows,
    pigs := f.pigs + new_pigs,
    goats := f.goats + new_goats }

/-- Theorem: The farm will have 21 animals after adding the new ones -/
theorem farm_animals_after_addition :
  let initial_farm := Farm.mk 2 3 6
  let final_farm := initial_farm.add 3 5 2
  final_farm.total = 21 := by sorry

end farm_animals_after_addition_l2241_224153


namespace youngest_child_age_l2241_224135

/-- Represents a family with its members and ages -/
structure Family where
  members : Nat
  total_age : Nat

/-- Calculates the average age of a family -/
def average_age (f : Family) : Nat :=
  f.total_age / f.members

theorem youngest_child_age :
  let initial_family : Family := { members := 4, total_age := 96 }
  let current_family : Family := { members := 6, total_age := 144 }
  let age_difference : Nat := 2
  average_age initial_family = 24 →
  average_age current_family = 24 →
  ∃ (youngest_age : Nat),
    youngest_age = 3 ∧
    youngest_age + (youngest_age + age_difference) = current_family.total_age - (initial_family.total_age + 40) :=
by sorry

end youngest_child_age_l2241_224135


namespace percentage_relation_l2241_224177

theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.05 * C) 
  (h2 : A = 0.20 * B) : 
  B = 0.25 * C := by
sorry

end percentage_relation_l2241_224177


namespace neg_i_cubed_l2241_224127

theorem neg_i_cubed (i : ℂ) (h : i^2 = -1) : (-i)^3 = -i := by
  sorry

end neg_i_cubed_l2241_224127


namespace subtract_negative_two_from_three_l2241_224111

theorem subtract_negative_two_from_three : 3 - (-2) = 5 := by
  sorry

end subtract_negative_two_from_three_l2241_224111


namespace total_candies_l2241_224174

theorem total_candies (chocolate_boxes caramel_boxes pieces_per_box : ℕ) :
  chocolate_boxes = 6 →
  caramel_boxes = 4 →
  pieces_per_box = 9 →
  chocolate_boxes * pieces_per_box + caramel_boxes * pieces_per_box = 90 :=
by
  sorry

#check total_candies

end total_candies_l2241_224174


namespace minibus_boys_count_l2241_224188

theorem minibus_boys_count : 
  ∀ (total boys girls : ℕ),
  total = 18 →
  boys + girls = total →
  boys = girls - 2 →
  boys = 8 :=
by
  sorry

end minibus_boys_count_l2241_224188


namespace decorative_window_area_ratio_l2241_224112

theorem decorative_window_area_ratio :
  let base : ℝ := 40
  let length : ℝ := (4/3) * base
  let semi_major_axis : ℝ := base / 2
  let semi_minor_axis : ℝ := base / 4
  let rectangle_area : ℝ := length * base
  let ellipse_area : ℝ := π * semi_major_axis * semi_minor_axis
  let triangle_area : ℝ := (1/2) * base * semi_minor_axis
  rectangle_area / (ellipse_area + triangle_area) = 32 / (3 * (π + 1)) :=
by sorry

end decorative_window_area_ratio_l2241_224112


namespace roots_sum_and_product_l2241_224138

theorem roots_sum_and_product (c d : ℝ) : 
  c^2 - 6*c + 8 = 0 → d^2 - 6*d + 8 = 0 → c^3 + c^4*d^2 + c^2*d^4 + d^3 = 1352 := by
  sorry

end roots_sum_and_product_l2241_224138


namespace unique_solution_l2241_224178

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y * z) = f x * f y * f z - 6 * x * y * z

/-- The main theorem stating that the only function satisfying the equation is f(x) = 2x -/
theorem unique_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) : 
  ∀ x : ℝ, f x = 2 * x := by
  sorry

end unique_solution_l2241_224178


namespace factorization_equality_l2241_224121

theorem factorization_equality (a b : ℝ) : 2 * a^2 * b - 4 * a * b + 2 * b = 2 * b * (a - 1)^2 := by
  sorry

end factorization_equality_l2241_224121


namespace shortest_rope_part_l2241_224106

theorem shortest_rope_part (total_length : ℝ) (ratio1 ratio2 ratio3 : ℝ) 
  (h1 : total_length = 196.85)
  (h2 : ratio1 = 3.6)
  (h3 : ratio2 = 8.4)
  (h4 : ratio3 = 12) :
  let total_ratio := ratio1 + ratio2 + ratio3
  let shortest_part := (total_length / total_ratio) * ratio1
  shortest_part = 29.5275 := by
sorry

end shortest_rope_part_l2241_224106


namespace salary_after_raise_l2241_224119

theorem salary_after_raise (original_salary : ℝ) (percentage_increase : ℝ) (new_salary : ℝ) :
  original_salary = 60 →
  percentage_increase = 83.33333333333334 →
  new_salary = original_salary * (1 + percentage_increase / 100) →
  new_salary = 110 := by
  sorry

end salary_after_raise_l2241_224119


namespace gunther_free_time_l2241_224189

/-- Represents the time required for cleaning tasks and available free time -/
structure CleaningTime where
  vacuum : ℕ
  dust : ℕ
  mop : ℕ
  brush_per_cat : ℕ
  num_cats : ℕ
  free_time : ℕ

/-- Calculates the remaining free time after cleaning -/
def remaining_free_time (ct : CleaningTime) : ℕ :=
  ct.free_time - (ct.vacuum + ct.dust + ct.mop + ct.brush_per_cat * ct.num_cats)

/-- Theorem: Given Gunther's cleaning tasks and available time, he will have 30 minutes left -/
theorem gunther_free_time :
  ∀ (ct : CleaningTime),
    ct.vacuum = 45 →
    ct.dust = 60 →
    ct.mop = 30 →
    ct.brush_per_cat = 5 →
    ct.num_cats = 3 →
    ct.free_time = 180 →
    remaining_free_time ct = 30 := by
  sorry

end gunther_free_time_l2241_224189


namespace complementary_sets_count_l2241_224163

/-- Represents a card with four attributes -/
structure Card where
  shape : Fin 3
  color : Fin 3
  shade : Fin 3
  pattern : Fin 3

/-- The deck of all possible cards -/
def deck : Finset Card := sorry

/-- Checks if three cards form a complementary set -/
def isComplementary (c1 c2 c3 : Card) : Prop := sorry

/-- The set of all complementary three-card sets -/
def complementarySets : Finset (Finset Card) := sorry

theorem complementary_sets_count :
  deck.card = 81 ∧ (∀ c1 c2 : Card, c1 ∈ deck → c2 ∈ deck → c1 = c2 ∨ c1.shape ≠ c2.shape ∨ c1.color ≠ c2.color ∨ c1.shade ≠ c2.shade ∨ c1.pattern ≠ c2.pattern) →
  complementarySets.card = 5400 := by
  sorry

end complementary_sets_count_l2241_224163


namespace least_possible_x_l2241_224197

theorem least_possible_x (x y z : ℤ) : 
  (∃ k : ℤ, x = 2 * k) →  -- x is even
  (∃ m : ℤ, y = 2 * m + 1) →  -- y is odd
  (∃ n : ℤ, z = 2 * n + 1) →  -- z is odd
  y - x > 5 →
  z - x ≥ 9 →
  (∀ w : ℤ, (∃ j : ℤ, w = 2 * j) → w ≥ x) →
  x = 0 :=
by sorry

end least_possible_x_l2241_224197


namespace second_quadrant_fraction_negative_l2241_224183

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating that for a point in the second quadrant, a/b < 0 -/
theorem second_quadrant_fraction_negative (p : Point) :
  is_in_second_quadrant p → p.x / p.y < 0 :=
by
  sorry


end second_quadrant_fraction_negative_l2241_224183


namespace sanya_towels_count_l2241_224145

/-- The number of bath towels Sanya can wash in one wash -/
def towels_per_wash : ℕ := 7

/-- The number of hours Sanya has per day for washing -/
def hours_per_day : ℕ := 2

/-- The number of days it takes to wash all towels -/
def days_to_wash_all : ℕ := 7

/-- The total number of bath towels Sanya has -/
def total_towels : ℕ := towels_per_wash * hours_per_day * days_to_wash_all

theorem sanya_towels_count : total_towels = 98 := by
  sorry

end sanya_towels_count_l2241_224145


namespace digit_equation_sum_l2241_224134

theorem digit_equation_sum : 
  ∀ (E M V Y : ℕ),
  (E < 10) → (M < 10) → (V < 10) → (Y < 10) →
  (V ≥ 1) →
  (Y ≠ 0) → (M ≠ 0) →
  (E ≠ M) → (E ≠ V) → (E ≠ Y) → 
  (M ≠ V) → (M ≠ Y) → 
  (V ≠ Y) →
  ((10 * Y + E) * (10 * M + E) = 111 * V) →
  (E + M + V + Y = 21) := by
sorry

end digit_equation_sum_l2241_224134


namespace rainfall_volume_calculation_l2241_224100

/-- Calculates the total rainfall volume given rainfall rates and area --/
def total_rainfall_volume (rate1 rate2 : ℝ) (area : ℝ) : ℝ :=
  (rate1 * area + rate2 * area) * 0.001

theorem rainfall_volume_calculation :
  let rate1 : ℝ := 5  -- mm/hour
  let rate2 : ℝ := 10 -- mm/hour
  let area : ℝ := 100 -- square meters
  total_rainfall_volume rate1 rate2 area = 1.5 := by
sorry

end rainfall_volume_calculation_l2241_224100


namespace sum_after_removing_terms_l2241_224142

theorem sum_after_removing_terms : 
  let sequence := [1/3, 1/6, 1/9, 1/12, 1/15, 1/18]
  let removed_terms := [1/12, 1/15]
  let remaining_terms := sequence.filter (λ x => x ∉ removed_terms)
  (remaining_terms.sum = 1) := by sorry

end sum_after_removing_terms_l2241_224142


namespace g_composed_has_two_distinct_roots_l2241_224114

/-- The function g(x) = x^2 + 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition of g with itself -/
def g_composed (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that g(g(x)) has exactly 2 distinct real roots when d = 8 -/
theorem g_composed_has_two_distinct_roots :
  ∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
  (∀ x : ℝ, g_composed 8 x = 0 ↔ x = r₁ ∨ x = r₂) :=
sorry

end g_composed_has_two_distinct_roots_l2241_224114


namespace min_value_sqrt_x_squared_plus_two_l2241_224167

theorem min_value_sqrt_x_squared_plus_two (x : ℝ) :
  Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) ≥ 3 * Real.sqrt 2 / 2 := by
  sorry

end min_value_sqrt_x_squared_plus_two_l2241_224167


namespace jogging_time_calculation_l2241_224139

theorem jogging_time_calculation (total_distance : ℝ) (total_time : ℝ) (initial_speed : ℝ) (later_speed : ℝ)
  (h1 : total_distance = 160)
  (h2 : total_time = 8)
  (h3 : initial_speed = 15)
  (h4 : later_speed = 10) :
  ∃ (initial_time : ℝ),
    initial_time * initial_speed + (total_time - initial_time) * later_speed = total_distance ∧
    initial_time = 16 / 5 := by
  sorry

end jogging_time_calculation_l2241_224139


namespace problem_solution_l2241_224122

theorem problem_solution (x y : ℝ) (h : y = Real.sqrt (x - 4) - Real.sqrt (4 - x) + 2023) :
  y - x^2 + 17 = 2024 := by
  sorry

end problem_solution_l2241_224122


namespace probability_three_in_same_group_l2241_224143

/-- The number of people to be partitioned -/
def total_people : ℕ := 15

/-- The number of groups -/
def num_groups : ℕ := 6

/-- The sizes of the groups -/
def group_sizes : List ℕ := [3, 3, 3, 2, 2, 2]

/-- The number of people we're interested in (Petruk, Gareng, and Bagong) -/
def num_interested : ℕ := 3

/-- The probability that Petruk, Gareng, and Bagong are in the same group -/
def probability_same_group : ℚ := 3 / 455

theorem probability_three_in_same_group :
  let total_ways := (total_people.factorial) / (group_sizes.map Nat.factorial).prod
  let favorable_ways := 3 * ((total_people - num_interested).factorial) / 
    ((group_sizes.tail.map Nat.factorial).prod)
  (favorable_ways : ℚ) / total_ways = probability_same_group := by
  sorry

end probability_three_in_same_group_l2241_224143


namespace max_pieces_is_112_l2241_224108

/-- Represents the dimensions of a rectangular cake cut into square pieces -/
structure CakeDimensions where
  m : ℕ
  n : ℕ

/-- Calculates the number of interior pieces in a cake -/
def interiorPieces (d : CakeDimensions) : ℕ :=
  if d.m > 2 ∧ d.n > 2 then (d.m - 2) * (d.n - 2) else 0

/-- Calculates the number of exterior pieces in a cake -/
def exteriorPieces (d : CakeDimensions) : ℕ :=
  d.m * d.n - interiorPieces d

/-- Checks if the cake satisfies the condition that exterior pieces are twice the interior pieces -/
def satisfiesCondition (d : CakeDimensions) : Prop :=
  exteriorPieces d = 2 * interiorPieces d

/-- The theorem stating that the maximum number of pieces under the given conditions is 112 -/
theorem max_pieces_is_112 :
  ∃ d : CakeDimensions, satisfiesCondition d ∧
  ∀ d' : CakeDimensions, satisfiesCondition d' → d.m * d.n ≥ d'.m * d'.n ∧ d.m * d.n = 112 :=
sorry

end max_pieces_is_112_l2241_224108


namespace percentage_increase_l2241_224179

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 400 → final = 480 → (final - initial) / initial * 100 = 20 := by
  sorry

end percentage_increase_l2241_224179


namespace hyperbola_m_range_l2241_224196

def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / m - y^2 / (m + 1) = 1 → 
    (m > 0 ∧ m + 1 > 0) ∨ (m < 0 ∧ m + 1 < 0)

theorem hyperbola_m_range :
  {m : ℝ | is_hyperbola m} = {m | m < -1 ∨ m > 0} := by sorry

end hyperbola_m_range_l2241_224196


namespace circle_area_from_circumference_l2241_224103

theorem circle_area_from_circumference :
  ∀ (C : ℝ) (r : ℝ) (A : ℝ),
  C = 36 →
  C = 2 * π * r →
  A = π * r^2 →
  A = 324 / π := by
sorry

end circle_area_from_circumference_l2241_224103


namespace largest_n_two_solutions_exceed_two_l2241_224166

/-- The cubic polynomial in question -/
def f (n : ℤ) (x : ℝ) : ℝ :=
  x^3 - (n + 9 : ℝ) * x^2 + (2 * n^2 - 3 * n - 34 : ℝ) * x + 2 * (n - 4) * (n + 3 : ℝ)

/-- The statement that 8 is the largest integer for which the equation has two solutions > 2 -/
theorem largest_n_two_solutions_exceed_two :
  ∀ n : ℤ, (∃ x y : ℝ, x > 2 ∧ y > 2 ∧ x ≠ y ∧ f n x = 0 ∧ f n y = 0) → n ≤ 8 :=
sorry

end largest_n_two_solutions_exceed_two_l2241_224166


namespace angle_equality_l2241_224120

-- Define what it means for two angles to be vertical
def are_vertical_angles (A B : ℝ) : Prop := sorry

-- State the theorem that vertical angles are equal
axiom vertical_angles_are_equal : ∀ A B : ℝ, are_vertical_angles A B → A = B

-- The statement to be proved
theorem angle_equality (A B : ℝ) (h : are_vertical_angles A B) : A = B := by
  sorry

end angle_equality_l2241_224120


namespace limit_sum_geometric_sequence_l2241_224182

def geometricSequence (n : ℕ) : ℚ := (1/2) * (1/2)^(n-1)

def sumGeometricSequence (n : ℕ) : ℚ := 
  (1/2) * (1 - (1/2)^n) / (1 - 1/2)

theorem limit_sum_geometric_sequence :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sumGeometricSequence n - 1| < ε :=
sorry

end limit_sum_geometric_sequence_l2241_224182


namespace number_division_l2241_224136

theorem number_division (x : ℝ) : x + 8 = 88 → x / 10 = 8 := by
  sorry

end number_division_l2241_224136


namespace additional_batches_is_seven_l2241_224130

/-- Represents the number of cups of flour needed for one batch of cookies -/
def flour_per_batch : ℕ := 2

/-- Represents the number of batches of cookies Gigi baked -/
def batches_baked : ℕ := 3

/-- Represents the initial amount of flour in cups -/
def initial_flour : ℕ := 20

/-- Calculates the number of additional batches that can be made with the remaining flour -/
def additional_batches : ℕ :=
  (initial_flour - flour_per_batch * batches_baked) / flour_per_batch

/-- Theorem stating that the number of additional batches is 7 -/
theorem additional_batches_is_seven :
  additional_batches = 7 := by sorry

end additional_batches_is_seven_l2241_224130


namespace intersecting_chords_probability_2023_l2241_224199

/-- Given a circle with 2023 evenly spaced points, this function calculates
    the probability that when selecting four distinct points A, B, C, and D randomly,
    chord AB intersects chord CD and chord AC intersects chord BD. -/
def intersecting_chords_probability (n : ℕ) : ℚ :=
  if n = 2023 then 1/6 else 0

/-- Theorem stating that the probability of the specific chord intersection
    scenario for 2023 points is 1/6. -/
theorem intersecting_chords_probability_2023 :
  intersecting_chords_probability 2023 = 1/6 := by sorry

end intersecting_chords_probability_2023_l2241_224199


namespace sequence_equals_index_l2241_224172

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n ≥ 1 → a n > 0) ∧
  (∀ n m : ℕ, n ≥ 1 → m ≥ 1 → n < m → a n < a m) ∧
  (∀ n : ℕ, n ≥ 1 → a (2*n) = a n + n) ∧
  (∀ n : ℕ, n ≥ 1 → is_prime (a n) → is_prime n)

theorem sequence_equals_index (a : ℕ → ℕ) (h : sequence_property a) :
  ∀ n : ℕ, n ≥ 1 → a n = n :=
sorry

end sequence_equals_index_l2241_224172


namespace prob_different_suits_78_card_deck_l2241_224147

/-- A custom deck of cards -/
structure CustomDeck where
  total_cards : ℕ
  num_suits : ℕ
  cards_per_suit : ℕ
  total_cards_eq : total_cards = num_suits * cards_per_suit

/-- The probability of drawing two cards of different suits from a custom deck -/
def prob_different_suits (deck : CustomDeck) : ℚ :=
  let remaining_cards := deck.total_cards - 1
  let cards_different_suit := (deck.num_suits - 1) * deck.cards_per_suit
  cards_different_suit / remaining_cards

/-- The main theorem stating the probability for the specific deck -/
theorem prob_different_suits_78_card_deck :
  ∃ (deck : CustomDeck),
    deck.total_cards = 78 ∧
    deck.num_suits = 6 ∧
    deck.cards_per_suit = 13 ∧
    prob_different_suits deck = 65 / 77 := by
  sorry

end prob_different_suits_78_card_deck_l2241_224147


namespace sets_equality_implies_sum_l2241_224159

-- Define the sets A and B
def A (x y : ℝ) : Set ℝ := {x, y/x, 1}
def B (x y : ℝ) : Set ℝ := {x^2, x+y, 0}

-- State the theorem
theorem sets_equality_implies_sum (x y : ℝ) (h : A x y = B x y) : x^2014 + y^2015 = 1 := by
  sorry

end sets_equality_implies_sum_l2241_224159


namespace sally_dozens_of_eggs_l2241_224162

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Sally bought -/
def total_eggs : ℕ := 48

/-- The number of dozens of eggs Sally bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem sally_dozens_of_eggs : dozens_bought = 4 := by
  sorry

end sally_dozens_of_eggs_l2241_224162


namespace tile_cutting_theorem_l2241_224132

/-- Represents a rectangular tile -/
structure Tile where
  width : ℝ
  height : ℝ

/-- Represents the arrangement of tiles -/
structure TileArrangement where
  tiles : List Tile
  width : ℝ
  height : ℝ
  tileCount : ℕ

/-- Represents a part of a cut tile -/
structure TilePart where
  width : ℝ
  height : ℝ

theorem tile_cutting_theorem (arrangement : TileArrangement) 
  (h1 : arrangement.width < arrangement.height)
  (h2 : arrangement.tileCount > 0) :
  ∃ (squareParts rectangleParts : List TilePart),
    (∀ t ∈ arrangement.tiles, ∃ p1 p2, p1 ∈ squareParts ∧ p2 ∈ rectangleParts) ∧
    (∃ s, s > 0 ∧ (∀ p ∈ squareParts, p.width * p.height = s^2 / arrangement.tileCount)) ∧
    (∃ w h, w > 0 ∧ h > 0 ∧ w ≠ h ∧ 
      (∀ p ∈ rectangleParts, p.width * p.height = w * h / arrangement.tileCount)) :=
by sorry

end tile_cutting_theorem_l2241_224132


namespace track_completion_time_l2241_224129

/-- Represents a circular running track --/
structure Track where
  circumference : ℝ
  circumference_positive : circumference > 0

/-- Represents a runner on the track --/
structure Runner where
  speed : ℝ
  speed_positive : speed > 0

/-- Represents an event where two runners meet --/
structure MeetingEvent where
  time : ℝ
  time_nonnegative : time ≥ 0

/-- The main theorem to prove --/
theorem track_completion_time
  (track : Track)
  (runner1 runner2 runner3 : Runner)
  (meeting12 : MeetingEvent)
  (meeting23 : MeetingEvent)
  (meeting31 : MeetingEvent)
  (h1 : meeting23.time - meeting12.time = 15)
  (h2 : meeting31.time - meeting23.time = 25) :
  track.circumference / runner1.speed = 80 :=
sorry

end track_completion_time_l2241_224129


namespace probability_of_flush_l2241_224155

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Represents the number of cards chosen -/
def CardsChosen : ℕ := 6

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := StandardDeck / NumSuits

/-- Calculates the number of ways to choose n items from k items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Probability of forming a flush when choosing 6 cards at random from a standard 52-card deck -/
theorem probability_of_flush : 
  (NumSuits * choose CardsPerSuit CardsChosen) / choose StandardDeck CardsChosen = 3432 / 10179260 := by
  sorry

end probability_of_flush_l2241_224155


namespace modifiedLucas_100th_term_mod_10_l2241_224198

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 5
  | n + 2 => (modifiedLucas n + modifiedLucas (n + 1)) % 10

theorem modifiedLucas_100th_term_mod_10 :
  modifiedLucas 99 % 10 = 2 := by
  sorry

end modifiedLucas_100th_term_mod_10_l2241_224198


namespace congruence_solution_count_l2241_224170

theorem congruence_solution_count :
  ∃! (x : ℕ), x > 0 ∧ x < 50 ∧ (x + 20) % 45 = 70 % 45 :=
by sorry

end congruence_solution_count_l2241_224170


namespace linear_function_solution_l2241_224149

/-- Represents a linear function y = ax + b -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- Represents a point (x, y) on the linear function -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given data points for the linear function -/
def dataPoints : List Point := [
  { x := -3, y := -4 },
  { x := -2, y := -2 },
  { x := -1, y := 0 },
  { x := 0, y := 2 },
  { x := 1, y := 4 },
  { x := 2, y := 6 }
]

/-- The linear function satisfies all given data points -/
def satisfiesDataPoints (f : LinearFunction) : Prop :=
  ∀ p ∈ dataPoints, f.a * p.x + f.b = p.y

theorem linear_function_solution (f : LinearFunction) 
  (h : satisfiesDataPoints f) : 
  f.a * 1 + f.b = 4 := by sorry

end linear_function_solution_l2241_224149


namespace simplify_expression_l2241_224169

theorem simplify_expression (m n : ℝ) : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end simplify_expression_l2241_224169


namespace lucas_football_scores_l2241_224187

def first_ten_games : List Nat := [5, 2, 6, 3, 10, 1, 3, 3, 4, 2]

def total_first_ten : Nat := first_ten_games.sum

theorem lucas_football_scores :
  ∃ (game11 game12 : Nat),
    game11 < 10 ∧
    game12 < 10 ∧
    (total_first_ten + game11) % 11 = 0 ∧
    (total_first_ten + game11 + game12) % 12 = 0 ∧
    game11 * game12 = 20 := by
  sorry

end lucas_football_scores_l2241_224187


namespace lines_one_unit_from_origin_l2241_224133

theorem lines_one_unit_from_origin (x y : ℝ) (y' : ℝ → ℝ) :
  (∀ α : ℝ, x * Real.cos α + y * Real.sin α = 1) ↔
  y = x * y' x + Real.sqrt (1 + (y' x)^2) :=
sorry

end lines_one_unit_from_origin_l2241_224133


namespace necessary_but_not_sufficient_condition_l2241_224180

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, -1 < x ∧ x < 3 → -2 < x ∧ x < 4) ∧
  (∃ x : ℝ, -2 < x ∧ x < 4 ∧ ¬(-1 < x ∧ x < 3)) :=
by sorry

end necessary_but_not_sufficient_condition_l2241_224180


namespace polygon_angle_sum_l2241_224191

theorem polygon_angle_sum (n : ℕ) (h : n = 5) :
  (n - 2) * 180 + 360 = 900 :=
sorry

end polygon_angle_sum_l2241_224191


namespace polynomial_division_remainder_l2241_224124

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^3 + 3*X^2 - 4 : Polynomial ℝ) = (X^2 + X - 2 : Polynomial ℝ) * q + 0 := by
  sorry

end polynomial_division_remainder_l2241_224124


namespace apple_distribution_result_l2241_224152

/-- Represents the apple distribution problem --/
def apple_distribution (jim jane jerry jack jill jasmine jacob : ℕ) : ℚ :=
  let jack_to_jill := jack / 4
  let jasmine_jacob_shared := jasmine + jacob
  let jim_final := jim + (jasmine_jacob_shared / 10)
  let total_apples := jim_final + jane + jerry + (jack - jack_to_jill) + 
                      (jill + jack_to_jill) + (jasmine_jacob_shared / 2) + 
                      (jasmine_jacob_shared / 2)
  let average_apples := total_apples / 7
  average_apples / jim_final

/-- Theorem stating the result of the apple distribution problem --/
theorem apple_distribution_result : 
  ∃ ε > 0, |apple_distribution 20 60 40 80 50 30 90 - 1.705| < ε :=
sorry

end apple_distribution_result_l2241_224152
