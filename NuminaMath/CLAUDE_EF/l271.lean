import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_cos_sin_equation_l271_27167

theorem no_solution_cos_sin_equation :
  ∀ x : ℝ, Real.cos (Real.cos (Real.cos (Real.cos x))) ≠ Real.sin (Real.sin (Real.sin (Real.sin x))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_cos_sin_equation_l271_27167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_17325_l271_27165

theorem prime_divisors_of_17325 : 
  let n := 17325
  let prime_divisors := Finset.filter (λ p => Nat.Prime p ∧ p ∣ n) (Finset.range (n + 1))
  (prime_divisors.card = 4) ∧ 
  (prime_divisors.sum id = 26) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_17325_l271_27165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l271_27135

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := x^2 / (x + 1) - 3 / (x - 1) - 7 / 4

-- Define the solution set
def solution_set : Set ℝ := {x | x < -1 ∨ (x > 1.75 ∧ x ≠ -1)}

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, f x > 0 ↔ x ∈ solution_set :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l271_27135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brenda_has_eight_dollars_l271_27112

/-- The amount of money Brenda has given the conditions about Emma, Daya, Jeff, and Brenda's money -/
noncomputable def brendas_money (emmas_money : ℚ) : ℚ :=
  let dayas_money := emmas_money * (5/4)
  let jeffs_money := dayas_money * (2/5)
  jeffs_money + 4

/-- Theorem stating that Brenda has $8 given the conditions -/
theorem brenda_has_eight_dollars : brendas_money 8 = 8 := by
  unfold brendas_money
  simp
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brenda_has_eight_dollars_l271_27112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_extension_l271_27128

-- Define the function f for x ≥ 0
noncomputable def f_pos (x : ℝ) : ℝ := x^3 + Real.log (x + 1)

-- State the theorem
theorem even_function_extension {f : ℝ → ℝ} (h_even : ∀ x, f x = f (-x))
  (h_pos : ∀ x, x ≥ 0 → f x = f_pos x) :
  ∀ x, x < 0 → f x = -x^3 + Real.log (1 - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_extension_l271_27128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l271_27189

noncomputable section

-- Define the diagonal length
def diagonal : ℝ := 10

-- Define the square side length
noncomputable def square_side : ℝ := (diagonal / Real.sqrt 2)

-- Define the circle radius
noncomputable def circle_radius : ℝ := diagonal / 2

-- Define the area of the square
noncomputable def square_area : ℝ := square_side ^ 2

-- Define the area of the circle
noncomputable def circle_area : ℝ := Real.pi * circle_radius ^ 2

-- Define the height of the equilateral triangle
noncomputable def triangle_height : ℝ := (Real.sqrt 3 / 2) * square_side

-- Define the area of the equilateral triangle
noncomputable def triangle_area : ℝ := (1 / 2) * square_side * triangle_height

-- Define the combined area of the square and triangle
noncomputable def combined_area : ℝ := square_area + triangle_area

-- State the theorem
theorem area_difference_approx :
  ∃ ε > 0, abs (circle_area - combined_area + 14.8) < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l271_27189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tour_group_size_l271_27134

/-- Represents the age categories in the tour group -/
inductive AgeCategory
  | Children
  | YoungAdults
  | OlderPeople
deriving Repr

/-- Represents the tour group -/
structure TourGroup where
  total_people : ℕ
  angle : AgeCategory → ℝ
  percentage : AgeCategory → ℝ
  count : AgeCategory → ℕ

/-- The conditions of the tour group -/
def tour_group_conditions (g : TourGroup) : Prop :=
  g.angle AgeCategory.OlderPeople = g.angle AgeCategory.Children + 9 ∧
  g.percentage AgeCategory.YoungAdults = g.percentage AgeCategory.OlderPeople + 5 ∧
  g.count AgeCategory.YoungAdults = g.count AgeCategory.Children + 9 ∧
  (∀ c : AgeCategory, g.percentage c = g.angle c / 3.6) ∧
  (∀ c : AgeCategory, g.count c = g.total_people * g.percentage c / 100) ∧
  g.total_people > 0

/-- The theorem stating that the total number of people in the tour group is 120 -/
theorem tour_group_size (g : TourGroup) : 
  tour_group_conditions g → g.total_people = 120 := by
  sorry

#check tour_group_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tour_group_size_l271_27134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_sign_white_area_l271_27156

/-- The area of the white portion of a sign after painting "MATH" -/
theorem math_sign_white_area : ℝ := by
  let total_area : ℝ := 6 * 18
  let m_area : ℝ := 2 * (6 * 1) + 2 * (2 * 1)
  let a_area : ℝ := 2 * (3 * 1) + 1 * 4
  let t_area : ℝ := 1 * (6 * 1) + 1 * (4 * 1)
  let h_area : ℝ := 2 * (6 * 1) + 1 * 4
  let black_area : ℝ := m_area + a_area + t_area + h_area
  have : total_area - black_area = 56 := by sorry
  exact 56


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_sign_white_area_l271_27156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_unchanged_after_removing_extremes_l271_27130

-- Define a type for our dataset
def Dataset := List ℝ

-- Function to check if a list is sorted in ascending order
def is_sorted (l : Dataset) : Prop :=
  ∀ i j, i < j → i < l.length → j < l.length → l.get! i ≤ l.get! j

-- Function to get the median of a sorted list
noncomputable def median (l : Dataset) : ℝ :=
  if l.length % 2 = 0
  then (l.get! (l.length / 2 - 1) + l.get! (l.length / 2)) / 2
  else l.get! (l.length / 2)

-- Function to remove the highest and lowest values from a list
def remove_extremes (l : Dataset) : Dataset :=
  l.drop 1 |>.dropLast

-- Theorem statement
theorem median_unchanged_after_removing_extremes 
  (d : Dataset) 
  (h1 : d.length > 3) 
  (h2 : is_sorted d) :
  median d = median (remove_extremes d) := by
  sorry

#check median_unchanged_after_removing_extremes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_unchanged_after_removing_extremes_l271_27130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l271_27119

-- Define a circle with center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a membership relation for points in a circle
def pointInCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define the three circles
variable (c1 c2 c3 : Circle)

-- Define the common point M and other intersection points A, B, C
variable (M A B C : ℝ × ℝ)

-- Axioms based on the problem conditions
axiom same_radius : c1.radius = c2.radius ∧ c2.radius = c3.radius

axiom common_point : pointInCircle M c1 ∧ pointInCircle M c2 ∧ pointInCircle M c3

axiom intersection_points :
  pointInCircle A c1 ∧ pointInCircle A c2 ∧
  pointInCircle B c2 ∧ pointInCircle B c3 ∧
  pointInCircle C c3 ∧ pointInCircle C c1

-- Define helper functions (these would need to be implemented)
def is_orthocenter (p : ℝ × ℝ) (triangle : Set (ℝ × ℝ)) : Prop := sorry

def circumradius (triangle : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the theorem to be proved
theorem circle_intersection_theorem :
  -- 1. A, B, and C form a triangle
  ∃ (triangle : Set (ℝ × ℝ)), triangle = {A, B, C} ∧
  -- 2. M is the orthocenter of triangle ABC
  is_orthocenter M triangle ∧
  -- 3. The circumradius of triangle ABC is equal to the radius of the given circles
  circumradius triangle = c1.radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l271_27119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l271_27132

theorem problem_statement (a b : ℝ) : 
  let A : Set ℝ := {a, b/a, 1}
  let B : Set ℝ := {a^2, a+b, 0}
  A ⊆ B → B ⊆ A → a^2023 + b^2023 = -1 :=
by
  intro A B h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l271_27132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sadie_speed_is_three_l271_27133

/-- Represents the relay race with given conditions -/
structure RelayRace where
  total_distance : ℝ
  total_time : ℝ
  sadie_time : ℝ
  ariana_speed : ℝ
  ariana_time : ℝ
  sarah_speed : ℝ

/-- Calculates Sadie's average speed given the relay race conditions -/
noncomputable def sadie_average_speed (race : RelayRace) : ℝ :=
  let ariana_distance := race.ariana_speed * race.ariana_time
  let sarah_time := race.total_time - race.sadie_time - race.ariana_time
  let sarah_distance := race.sarah_speed * sarah_time
  let sadie_distance := race.total_distance - ariana_distance - sarah_distance
  sadie_distance / race.sadie_time

/-- Theorem stating that Sadie's average speed is 3 miles per hour -/
theorem sadie_speed_is_three (race : RelayRace) 
    (h1 : race.total_distance = 17)
    (h2 : race.total_time = 4.5)
    (h3 : race.sadie_time = 2)
    (h4 : race.ariana_speed = 6)
    (h5 : race.ariana_time = 0.5)
    (h6 : race.sarah_speed = 4) : 
  sadie_average_speed race = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sadie_speed_is_three_l271_27133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_is_252_l271_27137

/-- The area covered by two congruent squares with side length 12, 
    where the center of one square is a vertex of the other square -/
def area_covered_by_two_squares : ℚ :=
  let square_side : ℚ := 12
  let square_area : ℚ := square_side ^ 2
  let overlap_area : ℚ := square_area / 4
  square_area * 2 - overlap_area

/-- Proof that the area covered is 252 -/
theorem area_covered_is_252 : area_covered_by_two_squares = 252 := by
  -- Unfold the definition of area_covered_by_two_squares
  unfold area_covered_by_two_squares
  -- Simplify the arithmetic
  norm_num

#eval area_covered_by_two_squares -- 252

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_is_252_l271_27137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l271_27104

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (∀ t, f (-Real.pi / 6 + t) = f (-Real.pi / 6 - t)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l271_27104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_is_L_type_min_g_implies_a_eq_e_f_is_L_type_iff_a_in_range_l271_27186

open Real

-- Define L-type function
def is_L_type (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, x * f x ≥ f x

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * log x - (x - 1) * log a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := deriv (f a) x

-- Theorem 1: ln(x) is an L-type function
theorem ln_is_L_type :
  is_L_type log (Set.Ioi 0) := by
  sorry

-- Theorem 2: If min(g(x)) = 1, then a = e
theorem min_g_implies_a_eq_e (a : ℝ) (h : a > 0) :
  (∃ x > 0, g a x = 1 ∧ ∀ y > 0, g a y ≥ 1) → a = ℯ := by
  sorry

-- Theorem 3: f(x) is an L-type function iff 0 < a ≤ e²
theorem f_is_L_type_iff_a_in_range (a : ℝ) :
  (is_L_type (f a) (Set.Ioi 0)) ↔ (0 < a ∧ a ≤ ℯ^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_is_L_type_min_g_implies_a_eq_e_f_is_L_type_iff_a_in_range_l271_27186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_partition_l271_27105

-- Define the set of irrational numbers
def MyIrrational : Set ℝ := {x : ℝ | ¬ (∃ (q : ℚ), (q : ℝ) = x)}

-- Define positive and negative irrational numbers
def PositiveIrrational : Set ℝ := {x : ℝ | x ∈ MyIrrational ∧ x > 0}
def NegativeIrrational : Set ℝ := {x : ℝ | x ∈ MyIrrational ∧ x < 0}

-- Theorem statement
theorem irrational_partition :
  (∀ x ∈ MyIrrational, x ∈ PositiveIrrational ∨ x ∈ NegativeIrrational ∨ x = 0) ∧
  (PositiveIrrational ≠ ∅) ∧
  (NegativeIrrational ≠ ∅) ∧
  (PositiveIrrational ∩ NegativeIrrational = ∅) :=
by
  sorry

#check irrational_partition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_partition_l271_27105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_distributed_iff_lambda_in_range_l271_27121

/-- The function f(x) = x^2 - λx + 2λ -/
def f (lambda : ℝ) (x : ℝ) : ℝ := x^2 - lambda*x + 2*lambda

/-- The function g(x) = ln(x + 1) -/
noncomputable def g (x : ℝ) : ℝ := Real.log (x + 1)

/-- The function u(x) = f(x) * g(x) -/
noncomputable def u (lambda : ℝ) (x : ℝ) : ℝ := f lambda x * g x

/-- Predicate that checks if a function is distributed in all quadrants -/
def distributed_in_all_quadrants (h : ℝ → ℝ) : Prop :=
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ h x₁ > 0 ∧ h (-x₂) > 0) ∧
  (∃ x₃ x₄, x₃ > 0 ∧ x₄ > 0 ∧ h x₃ < 0 ∧ h (-x₄) < 0)

/-- Theorem stating that u(x) is distributed in all quadrants iff λ ∈ (-1/3, 0) -/
theorem u_distributed_iff_lambda_in_range :
  ∀ lambda : ℝ, distributed_in_all_quadrants (u lambda) ↔ -1/3 < lambda ∧ lambda < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_distributed_iff_lambda_in_range_l271_27121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l271_27166

/-- The function f(x) = √3 * sin(2x) + cos(2x) is monotonically increasing in (0, π/6) -/
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < π/6 →
  Real.sqrt 3 * Real.sin (2*x₁) + Real.cos (2*x₁) < Real.sqrt 3 * Real.sin (2*x₂) + Real.cos (2*x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l271_27166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l271_27196

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 2, 0)

noncomputable def CurveE (P : ℝ × ℝ) : Prop :=
  abs (Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) - 
       Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)) = 2 ∧
  P.1 < 0

def LineAB (x y : ℝ) : Prop :=
  ∃ (k : ℝ), y = k * x - 1 ∧
             CurveE (x, y)

theorem line_AB_equation :
  ∃ (A B : ℝ × ℝ),
    LineAB A.1 A.2 ∧
    LineAB B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 * Real.sqrt 3 →
    ∃ (x y : ℝ), (Real.sqrt 5 / 2) * x + y + 1 = 0 ∧ LineAB x y := by
  sorry

#check line_AB_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l271_27196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_corn_purchase_l271_27115

noncomputable section

/-- The amount of corn and beans Jordan bought, in pounds -/
def total_pounds : ℝ := 24

/-- The cost of corn per pound, in cents -/
def corn_cost : ℝ := 99

/-- The cost of beans per pound, in cents -/
def bean_cost : ℝ := 55

/-- The total cost of Jordan's purchase, in cents -/
def total_cost : ℝ := 2070

/-- The amount of corn Jordan bought, in pounds -/
noncomputable def corn_pounds : ℝ := (total_cost - bean_cost * total_pounds) / (corn_cost - bean_cost)

theorem jordan_corn_purchase :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |corn_pounds - 17| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_corn_purchase_l271_27115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l271_27127

/-- A function f with specific properties -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- Theorem stating the unique number not in the range of f -/
theorem unique_number_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : f a b c d 5 = 5)
  (h2 : f a b c d 50 = 50)
  (h3 : ∀ x, x ≠ -d/c → f a b c d (f a b c d x) = x) :
  ∃! y, (∀ x, f a b c d x ≠ y) ∧ y = 27.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l271_27127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l271_27152

theorem problem_statement (a : ℤ) 
  (h1 : 0 ≤ a) (h2 : a < 13) 
  (h3 : (51^2016 + a) % 13 = 0) : 
  a = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l271_27152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l271_27191

-- Define the function f(x) = x + 3/(x-2) for x > 2
noncomputable def f (x : ℝ) : ℝ := x + 3 / (x - 2)

-- State the theorem
theorem f_minimum :
  ∃ (min_x : ℝ), min_x > 2 ∧
  (∀ x > 2, f x ≥ f min_x) ∧
  f min_x = 2 * Real.sqrt 3 + 2 ∧
  min_x = 2 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l271_27191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_narrow_black_stripes_l271_27199

/-- The number of wide black stripes -/
def w : ℕ := sorry

/-- The number of narrow black stripes -/
def n : ℕ := sorry

/-- The number of white stripes -/
def b : ℕ := sorry

/-- The number of white stripes is 7 more than the number of wide black stripes -/
axiom white_stripes : b = w + 7

/-- The total number of black stripes (wide and narrow) is one more than the number of white stripes -/
axiom black_stripes : w + n = b + 1

theorem narrow_black_stripes : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_narrow_black_stripes_l271_27199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_after_100_moves_l271_27113

/-- Represents the position of the chess piece -/
structure Position where
  x : ℤ
  y : ℤ

/-- Defines the movement rules for the chess piece -/
def move (n : ℕ) (pos : Position) : Position :=
  match n % 3 with
  | 0 => ⟨pos.x, pos.y + 1⟩  -- Move 1 unit up
  | 1 => ⟨pos.x + 1, pos.y⟩  -- Move 1 unit right
  | _ => ⟨pos.x + 2, pos.y⟩  -- Move 2 units right

/-- Applies the movement rules for n steps -/
def applyMoves : ℕ → Position
  | 0 => ⟨0, 0⟩  -- Start at origin
  | n + 1 => move (n + 1) (applyMoves n)

theorem final_position_after_100_moves :
  applyMoves 100 = ⟨100, 33⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_after_100_moves_l271_27113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l271_27150

theorem log_inequality_range (a : ℝ) :
  (∀ x : ℝ, Real.log (2 + Real.exp (x - 1)) / Real.log a ≤ -1) ↔ (1/2 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l271_27150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_match_schedule_count_l271_27164

/-- Represents a chess match between two schools --/
structure ChessMatch where
  /-- Number of players per school --/
  players_per_school : ℕ
  /-- Number of games each player plays against each opponent --/
  games_per_opponent : ℕ
  /-- Number of games played simultaneously in each round --/
  games_per_round : ℕ

/-- Calculate the total number of games in the match --/
def total_games (m : ChessMatch) : ℕ :=
  m.players_per_school * m.players_per_school * m.games_per_opponent

/-- Calculate the number of rounds in the match --/
def number_of_rounds (m : ChessMatch) : ℕ :=
  (total_games m) / m.games_per_round

/-- The number of ways to schedule the chess match --/
def schedule_count (m : ChessMatch) : ℕ :=
  (number_of_rounds m).factorial

/-- The specific chess match described in the problem --/
def specific_match : ChessMatch :=
  { players_per_school := 4
  , games_per_opponent := 2
  , games_per_round := 4 }

theorem chess_match_schedule_count :
  schedule_count specific_match = 40320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_match_schedule_count_l271_27164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_curve_max_value_achieved_l271_27160

open Real

-- Define the function y = e^2 / x
noncomputable def f (x : ℝ) : ℝ := Real.exp 2 / x

-- Define the point P(a,b) on the graph of f
def point_on_graph (a b : ℝ) : Prop := f a = b

-- Theorem statement
theorem max_value_on_curve (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h_point : point_on_graph a b) : 
  a ^ (log b) ≤ Real.exp 1 := by
  sorry

-- The maximum value is achieved when a = b = e
theorem max_value_achieved (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h_point : point_on_graph a b) : 
  a ^ (log b) = Real.exp 1 ↔ a = Real.exp 1 ∧ b = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_curve_max_value_achieved_l271_27160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_theorem_l271_27183

/-- Represents the time it takes for a valve to fill the pool alone -/
structure ValveTime where
  hours : ℝ
  hours_positive : hours > 0

/-- Represents the fill rate of a valve in pool volume per hour -/
noncomputable def fill_rate (v : ValveTime) : ℝ := 1 / v.hours

/-- Represents the problem setup -/
structure PoolProblem where
  valve_a : ValveTime
  valve_b : ValveTime
  total_time : ℝ
  total_time_positive : total_time > 0

/-- The solution to the problem -/
def simultaneous_time (p : PoolProblem) : ℝ :=
  p.total_time - 1 - 1

theorem pool_filling_theorem (p : PoolProblem) 
  (h1 : p.valve_a.hours = 10)
  (h2 : p.valve_b.hours = 15)
  (h3 : p.total_time = 7) :
  simultaneous_time p = 5 := by
  sorry

#eval simultaneous_time { 
  valve_a := { hours := 10, hours_positive := by norm_num },
  valve_b := { hours := 15, hours_positive := by norm_num },
  total_time := 7,
  total_time_positive := by norm_num
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_theorem_l271_27183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_peaceful_clients_l271_27180

/-- Represents the types of people in the bar --/
inductive PersonType
| Knight
| Liar
| Troublemaker

/-- Represents a person in the bar --/
structure Person where
  type : PersonType

/-- Represents the state of the bar --/
structure BarState where
  people : List Person
  knights_count : Nat
  liars_count : Nat
  troublemakers_count : Nat

/-- Function to simulate asking a person about another person --/
def ask (asker : Person) (subject : Person) : Bool → Bool :=
  sorry

/-- Function to simulate the bartender's strategy --/
def bartender_strategy (initial_state : BarState) : BarState :=
  sorry

/-- The main theorem to prove --/
theorem max_peaceful_clients 
  (initial_state : BarState) 
  (h1 : initial_state.people.length = 30)
  (h2 : initial_state.knights_count = 10)
  (h3 : initial_state.liars_count = 10)
  (h4 : initial_state.troublemakers_count = 10) :
  let final_state := bartender_strategy initial_state
  final_state.knights_count + final_state.liars_count ≤ 19 ∧
  final_state.troublemakers_count = 0 := by
  sorry

#check max_peaceful_clients

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_peaceful_clients_l271_27180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_correct_l271_27151

/-- The coefficient of x^4 in the expansion of (x - 1/(2x))^6 -/
def coefficient_x4 : ℤ := -3

/-- The binomial expansion of (x - 1/(2x))^6 -/
noncomputable def binomial_expansion (x : ℝ) : ℝ := (x - 1 / (2 * x)) ^ 6

theorem coefficient_x4_correct :
  ∃ (a b c d e f g : ℝ),
    binomial_expansion x = a * x^6 + b * x^5 + c * x^4 + d * x^3 + e * x^2 + f * x + g ∧
    c = ↑coefficient_x4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_correct_l271_27151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_intersection_l271_27177

-- Define the curve C in polar coordinates
noncomputable def curve_C_polar (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the curve C in rectangular coordinates
def curve_C_rect (x y : ℝ) : Prop := x^2 + y^2 = 4*x

-- Define the intersecting line
def line (t : ℝ) : ℝ × ℝ := (t + 1, t)

-- State the theorem
theorem curve_C_and_intersection :
  (∀ θ, curve_C_polar θ = 4 * Real.cos θ) →
  (∀ x y, curve_C_rect x y ↔ x^2 + y^2 = 4*x) ∧
  (∃ A B : ℝ × ℝ,
    (∃ t₁ t₂, line t₁ = A ∧ line t₂ = B) ∧
    curve_C_rect A.1 A.2 ∧
    curve_C_rect B.1 B.2 ∧
    Real.sqrt 14 = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_intersection_l271_27177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_quality_l271_27136

/-- Represents the coefficient of determination in a regression analysis -/
def R_squared : ℝ → ℝ := sorry

/-- Represents the sum of squares of residuals in a regression analysis -/
def sum_squares_residuals : ℝ → ℝ := sorry

/-- States that as R² increases, the sum of squares of residuals decreases -/
theorem regression_quality (h : ∀ x, 0 ≤ R_squared x ∧ R_squared x ≤ 1) :
  ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    R_squared x < R_squared y ∧ R_squared y ≤ 1 →
    sum_squares_residuals y < sum_squares_residuals x - ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_quality_l271_27136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_property_l271_27108

/-- Given a function f(x) = ax^2 + ln(x-1) where a > 2, if there exist distinct real numbers x₁ and x₂
    in the interval (3/2, +∞) such that f(x₁) + f(x₂) = 8a, then x₁ + x₂ < 4. -/
theorem function_sum_property (a : ℝ) (x₁ x₂ : ℝ) (h1 : a > 2) 
    (h2 : x₁ ∈ Set.Ioi (3/2 : ℝ)) (h3 : x₂ ∈ Set.Ioi (3/2 : ℝ)) 
    (h4 : x₁ ≠ x₂) (h5 : a * x₁^2 + Real.log (x₁ - 1) + a * x₂^2 + Real.log (x₂ - 1) = 8 * a) : 
  x₁ + x₂ < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_property_l271_27108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_specific_l271_27182

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

theorem geometric_series_sum_specific : 
  let a : ℝ := 1
  let r : ℝ := 3
  let last_term : ℝ := 2187
  let n : ℕ := 8
  geometric_series_sum a r n = 3280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_specific_l271_27182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequalities_l271_27185

open Real

-- Define the function f and its properties
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Axiom for the domain of f
axiom domain_f : ∀ x, 0 < x → x < π / 2 → ∃ y, f x = y

-- Axiom for the condition on f and its derivative
axiom condition_f : ∀ x, 0 < x → x < π / 2 → f' x * cos x > f x * sin x

-- Theorem to prove
theorem f_inequalities :
  (f (π / 3) > sqrt 2 * f (π / 4)) ∧
  (2 * f (π / 4) > sqrt 6 * f (π / 6)) ∧
  (f (π / 3) > 2 * cos 1 * f 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequalities_l271_27185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_salaries_l271_27116

/-- Given 5 individuals with an average salary of 8800 and one individual with a salary of 5000,
    prove that the sum of the other four individuals' salaries is 39000 -/
theorem combined_salaries (salaries : Fin 5 → ℕ) 
  (avg_salary : (Finset.sum Finset.univ salaries) / 5 = 8800)
  (b_salary : salaries 1 = 5000) :
  (Finset.sum Finset.univ salaries - salaries 1) = 39000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_salaries_l271_27116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_with_spheres_l271_27148

/-- The height of a rectangular box containing spheres --/
noncomputable def box_height (box_width : ℝ) (large_sphere_radius : ℝ) (small_sphere_radius : ℝ) : ℝ :=
  2 * (small_sphere_radius + Real.sqrt (
    (box_width / 2 - small_sphere_radius)^2 +
    (box_width / 2 - small_sphere_radius)^2
  ))

/-- Theorem stating the height of the box given specific dimensions and sphere sizes --/
theorem box_height_with_spheres :
  box_height 5 2.5 1 = 2 + 2 * Real.sqrt 7.75 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_with_spheres_l271_27148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_of_cubic_l271_27163

/-- The function f(x) = x³ - 2x² + 3 has exactly 2 extreme points -/
theorem extreme_points_of_cubic (f : ℝ → ℝ) :
  (∀ x, f x = x^3 - 2*x^2 + 3) →
  (∃! (s : Finset ℝ), s.card = 2 ∧
    ∀ x ∈ s, ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε),
      y ≠ x → (f y - f x) * (y - x) < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_of_cubic_l271_27163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nadia_walking_distance_l271_27107

/-- Represents the walking path between Nadia's house and her Grandmother's house -/
structure WalkingPath where
  flat_distance : ℝ
  uphill_distance : ℝ
  downhill_distance : ℝ

/-- Calculates the time taken to walk a given path -/
noncomputable def walk_time (path : WalkingPath) (flat_speed uphill_speed downhill_speed : ℝ) : ℝ :=
  path.flat_distance / flat_speed + path.uphill_distance / uphill_speed + path.downhill_distance / downhill_speed

theorem nadia_walking_distance :
  ∀ (path : WalkingPath),
    path.flat_distance = 2.5 →
    walk_time path 5 4 6 = 1.6 →
    walk_time ⟨path.flat_distance, path.downhill_distance, path.uphill_distance⟩ 5 4 6 = 1.65 →
    path.flat_distance + path.uphill_distance + path.downhill_distance = 7.9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nadia_walking_distance_l271_27107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_speed_calculation_l271_27143

/-- Calculates the jogging speed to school given total travel time, return speed, and distance. -/
noncomputable def jogging_speed (total_time : ℝ) (return_speed : ℝ) (distance : ℝ) : ℝ :=
  let return_time := distance / return_speed
  let jogging_time := total_time - return_time
  distance / jogging_time

/-- Theorem stating that given the specific conditions, the jogging speed is approximately 8.89 mph. -/
theorem jogging_speed_calculation :
  let total_time : ℝ := 1
  let return_speed : ℝ := 30
  let distance : ℝ := 6.857142857142858
  abs (jogging_speed total_time return_speed distance - 8.89) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_speed_calculation_l271_27143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_real_part_positive_z_imag_part_negative_z_in_fourth_quadrant_l271_27149

/-- The complex number (1+i)/√2 raised to the power of 2015 -/
noncomputable def z : ℂ := ((1 + Complex.I) / (Complex.abs (1 + Complex.I))) ^ 2015

/-- The real part of z is positive -/
theorem z_real_part_positive : 0 < z.re := by sorry

/-- The imaginary part of z is negative -/
theorem z_imag_part_negative : z.im < 0 := by sorry

/-- The point corresponding to z is in the fourth quadrant -/
theorem z_in_fourth_quadrant : 0 < z.re ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_real_part_positive_z_imag_part_negative_z_in_fourth_quadrant_l271_27149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_and_volume_l271_27153

/-- Given a cylinder with height 2a and square axial cross-section, and two identical cones
    constructed on its bases with vertices at the midpoint of the cylinder's axis,
    this theorem proves the sum of the total surface areas and volumes of the cones. -/
theorem cone_surface_area_and_volume (a : ℝ) (h : a > 0) :
  (2 * Real.pi * a^2 * (Real.sqrt 2 + 1) = 2 * Real.pi * a^2 * (Real.sqrt 2 + 1)) ∧
  ((2/3) * Real.pi * a^3 = (2/3) * Real.pi * a^3) :=
by
  -- Define local variables
  let cylinder_height : ℝ := 2 * a
  let cone_height : ℝ := a
  let cone_radius : ℝ := a
  let cone_slant_height : ℝ := a * Real.sqrt 2
  let cone_surface_area : ℝ := 2 * Real.pi * a^2 * (Real.sqrt 2 + 1)
  let cone_volume : ℝ := (2/3) * Real.pi * a^3
  
  -- Prove the equalities
  have surface_area_eq : cone_surface_area = 2 * Real.pi * a^2 * (Real.sqrt 2 + 1) := by rfl
  have volume_eq : cone_volume = (2/3) * Real.pi * a^3 := by rfl
  
  -- Combine the proofs
  exact ⟨surface_area_eq, volume_eq⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_and_volume_l271_27153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l271_27175

/-- The length of a train given its speed and time to pass a point -/
noncomputable def train_length (speed_kmph : ℝ) (time_seconds : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600) * time_seconds

/-- Theorem stating the length of the train -/
theorem train_length_approx :
  let speed := (46 : ℝ)
  let time := (5.008294988574827 : ℝ)
  ‖train_length speed time - 64‖ < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l271_27175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l271_27147

/-- The circle C with equation x^2 + (y-1)^2 = 1 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 1)^2 = 1}

/-- The ellipse E with equation x^2/4 + y^2 = 1 -/
def E : Set (ℝ × ℝ) := {p | p.1^2/4 + p.2^2 = 1}

/-- A and B are endpoints of a diameter of circle C -/
def isDiameter (A B : ℝ × ℝ) : Prop := A ∈ C ∧ B ∈ C ∧ (A.1 + B.1 = 0) ∧ (A.2 + B.2 = 2)

/-- The dot product of PA and PB -/
def dotProduct (P A B : ℝ × ℝ) : ℝ := 
  ((P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2))

theorem dot_product_range (P A B : ℝ × ℝ) (h1 : P ∈ E) (h2 : isDiameter A B) :
  -1 ≤ dotProduct P A B ∧ dotProduct P A B ≤ 13/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l271_27147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l271_27178

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  ab : Real × Real
  ac : Real × Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : 2 * (t.ab.1 * t.ac.1 + t.ab.2 * t.ac.2) = t.a^2 - (t.b + t.c)^2) : 
  (t.A = 2 * Real.pi / 3) ∧ 
  (∃ (max : Real), max = Real.sqrt 3 / 8 ∧ 
    (∀ B C, 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ B + C = Real.pi / 3 → 
      Real.sin t.A * Real.sin B * Real.sin C ≤ max)) ∧
  (Real.sin t.A * Real.sin (Real.pi / 6) * Real.sin (Real.pi / 6) = Real.sqrt 3 / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l271_27178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_cube_root_28_l271_27125

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := -x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions of the problem
def satisfies_conditions (t : Triangle) : Prop :=
  t.A = (0, 0) ∧
  t.B.2 = 0 ∧
  t.A.2 = parabola t.A.1 ∧
  t.B.2 = parabola t.B.1 ∧
  t.C.2 = parabola t.C.1

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  (abs ((t.B.1 - t.A.1) * (t.C.2 - t.A.2) - (t.C.1 - t.A.1) * (t.B.2 - t.A.2))) / 2

-- The main theorem
theorem length_AB_is_cube_root_28 (t : Triangle) 
  (h_conditions : satisfies_conditions t) 
  (h_area : triangle_area t = 28) :
  abs (t.B.1 - t.A.1) = (28 : ℝ) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_cube_root_28_l271_27125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l271_27154

/-- The distance between the foci of an ellipse -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- Theorem: The distance between the foci of the ellipse x^2/36 + y^2/16 = 8 is 8√10 -/
theorem ellipse_foci_distance :
  let a : ℝ := Real.sqrt 288
  let b : ℝ := Real.sqrt 128
  distance_between_foci a b = 8 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l271_27154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l271_27184

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6) + 2 * (Real.sin (x / 2))^2

/-- Theorem representing the problem -/
theorem triangle_side_sum (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = Real.pi ∧    -- Sum of angles in a triangle
  f A = 3/2 ∧              -- Condition on f(A)
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 ∧  -- Area condition
  a = Real.sqrt 3          -- Condition on side a
  → b + c = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l271_27184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l271_27103

noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

theorem nabla_calculation : 
  ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 →
  nabla (nabla a b) (nabla c d) = 49/56 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l271_27103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_weight_is_18_985_l271_27145

/-- Represents the weight in grams of a one-liter packet for each brand -/
def brand_weights : Fin 5 → ℚ
  | 0 => 950
  | 1 => 850
  | 2 => 900
  | 3 => 920
  | 4 => 875

/-- Represents the mixing ratio for each brand -/
def mixing_ratios : Fin 5 → ℚ
  | 0 => 7
  | 1 => 4
  | 2 => 2
  | 3 => 3
  | 4 => 5

/-- The total volume of the mixture in liters -/
def total_volume : ℚ := 21

/-- Calculates the weight of the mixture in kilograms -/
noncomputable def mixture_weight : ℚ :=
  (Finset.sum Finset.univ fun i => brand_weights i * mixing_ratios i) / 1000

theorem mixture_weight_is_18_985 :
  mixture_weight = 18985 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_weight_is_18_985_l271_27145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_circle_radius_l271_27179

/-- Represents a sequence of circles arranged in a row, tangent to each other and two parallel lines. -/
structure CircleSequence where
  num_circles : ℕ
  smallest_radius : ℝ
  largest_radius : ℝ

/-- Calculates the radius of the nth circle in the sequence. -/
noncomputable def nth_circle_radius (cs : CircleSequence) (n : ℕ) : ℝ :=
  cs.smallest_radius * (cs.largest_radius / cs.smallest_radius) ^ ((n - 1) / (cs.num_circles - 1 : ℝ))

/-- Theorem stating that the radius of the fourth circle in a specific sequence is 10√2. -/
theorem fourth_circle_radius (cs : CircleSequence) :
  cs.num_circles = 7 ∧ cs.smallest_radius = 10 ∧ cs.largest_radius = 20 →
  nth_circle_radius cs 4 = 10 * Real.sqrt 2 := by
  sorry

#check fourth_circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_circle_radius_l271_27179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_l271_27176

open Real

/-- The function f(x) parameterized by a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + x^2 + (3*a + 2)*x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.exp x + 2*x + 3*a + 2

theorem min_value_condition (a : ℝ) :
  (∃ x₀ ∈ Set.Ioo (-1 : ℝ) 0, IsLocalMin (f a) x₀) ↔ a ∈ Set.Ioo (-1 : ℝ) (-1/(3*Real.exp 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_l271_27176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_t_for_inequality_unique_common_tangent_l271_27126

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 1/2 - 1/(2*x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

-- Part 1
theorem largest_t_for_inequality (t : ℝ) :
  (∀ x ∈ Set.Ioo 0 t, f x < g (1/2) x) ↔ t ≤ 1 := by
  sorry

-- Part 2
theorem unique_common_tangent (a : ℝ) :
  (a > 0 ∧ ∃! x, (deriv f x = deriv (g a) x ∧ f x = g a x)) ↔ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_t_for_inequality_unique_common_tangent_l271_27126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_shorts_l271_27188

/-- The cost of a single pair of shorts -/
def cost_per_pair : ℝ := sorry

/-- The discount rate applied when buying 3 or more pairs -/
def discount_rate : ℝ := 0.1

/-- The amount saved when buying 3 pairs at once compared to individually -/
def amount_saved : ℝ := 3

theorem cost_of_shorts : cost_per_pair = 10 := by
  have h1 : 3 * cost_per_pair - 3 * cost_per_pair * (1 - discount_rate) = amount_saved := by sorry
  
  -- Solve the equation for cost_per_pair
  have h2 : 3 * cost_per_pair - 3 * cost_per_pair + 3 * cost_per_pair * discount_rate = amount_saved := by sorry
  have h3 : 3 * cost_per_pair * discount_rate = amount_saved := by sorry
  have h4 : cost_per_pair = amount_saved / (3 * discount_rate) := by sorry
  
  -- Substitute the known values
  have h5 : cost_per_pair = 3 / (3 * 0.1) := by sorry
  have h6 : cost_per_pair = 10 := by sorry
  
  exact h6


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_shorts_l271_27188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l271_27172

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := Real.exp x * Real.log x + Real.exp x / x

-- Theorem statement
theorem tangent_line_at_one :
  ∃ (m b : ℝ), 
    (f_derivative 1 = m) ∧ 
    (f 1 = 0) ∧
    (∀ x, f 1 + m * (x - 1) = m * x + b) ∧
    (m = Real.exp 1) ∧
    (b = -(Real.exp 1)) := by
  sorry

#check tangent_line_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l271_27172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_irrational_zero_subset_zero_one_l271_27110

-- Define the necessary sets
def Q : Set ℝ := {x : ℝ | ∃ (p q : ℤ), q ≠ 0 ∧ x = ↑p / ↑q}
def R_minus_Q : Set ℝ := {x : ℝ | x ∉ Q}

-- Statement for √3 ∈ ℝ\ℚ
theorem sqrt_3_irrational : Real.sqrt 3 ∈ R_minus_Q := by sorry

-- Statement for {0} ⊆ {0,1}
theorem zero_subset_zero_one : ({0} : Set ℕ) ⊆ {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_irrational_zero_subset_zero_one_l271_27110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_14_pow_14_mod_2_l271_27111

def a : ℕ → ℤ
  | 0 => 11^11
  | 1 => 12^12
  | 2 => 13^13
  | n+3 => Int.natAbs (a (n+2) - a (n+1)) + Int.natAbs (a (n+1) - a n)

theorem a_14_pow_14_mod_2 : a (14^14 - 1) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_14_pow_14_mod_2_l271_27111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independent_events_intersection_probability_l271_27144

theorem independent_events_intersection_probability
  (p : Set α → ℝ) (a b : Set α) 
  (h1 : p a = 1/5)
  (h2 : p b = 2/5)
  (h3 : p (a ∩ b) = p a * p b) :
  p (a ∩ b) = 2/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independent_events_intersection_probability_l271_27144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l271_27129

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- Definition of a point being on a hyperbola -/
def on_hyperbola (p : Point) (h : Hyperbola) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.m * l2.m = -1

/-- Theorem statement -/
theorem hyperbola_asymptote_slope (h : Hyperbola) 
  (A1 A2 B C : Point)
  (hA1 : A1 = ⟨-h.a, 0⟩)
  (hA2 : A2 = ⟨h.a, 0⟩)
  (hB : on_hyperbola B h)
  (hC : on_hyperbola C h)
  (hBC : B.x = C.x ∧ B.y = -C.y)
  (hPerp : perpendicular ⟨(B.y - A1.y) / (B.x - A1.x), 0⟩ 
                         ⟨(C.y - A2.y) / (C.x - A2.x), 0⟩) :
  ∃ (k : ℝ), k = 1 ∧ ∀ (l : Line), l.m = k ∨ l.m = -k → 
    ∀ (p : Point), on_hyperbola p h → 
      ∃ (t : ℝ), p.y = l.m * p.x + t ∧ 
        ∀ (q : Point), q.y = l.m * q.x + t → 
          (q.x^2 / h.a^2) - (q.y^2 / h.b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l271_27129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l271_27114

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  sine_law : a / (Real.sin A) = b / (Real.sin B)
  
-- Define vectors m and n
noncomputable def m (triangle : AcuteTriangle) : ℝ × ℝ := (Real.sqrt 3 * triangle.a, triangle.c)
noncomputable def n (triangle : AcuteTriangle) : ℝ × ℝ := (Real.sin triangle.A, Real.cos triangle.C)

-- State the theorem
theorem triangle_properties (triangle : AcuteTriangle) 
  (h : m triangle = (3 : ℝ) • (n triangle)) : 
  triangle.C = π/3 ∧ 
  (3 * Real.sqrt 3 + 3)/2 < triangle.a + triangle.b + triangle.c ∧ 
  triangle.a + triangle.b + triangle.c ≤ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l271_27114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l271_27122

/-- Given a triangle ABC with side lengths a, b, c and angle B, proves that when b = 7, c = 5, and ∠B = 2π/3, the length of side a is 3. -/
theorem triangle_side_length (a b c : ℝ) (B : ℝ) 
    (h1 : b = 7) (h2 : c = 5) (h3 : B = 2 * Real.pi / 3) :
  a^2 + c^2 - 2*a*c*Real.cos B = b^2 → a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l271_27122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_player_win_prob_l271_27157

/-- Represents the probability of a coin flip being heads -/
noncomputable def coin_prob : ℝ := 1 / 2

/-- The number of players in the game -/
def num_players : ℕ := 4

/-- Calculates the probability of the last player winning on their nth turn -/
noncomputable def prob_win_on_turn (n : ℕ) : ℝ := coin_prob^(num_players * n)

/-- Theorem: The probability of the last player winning the coin flipping game is 1/31 -/
theorem last_player_win_prob :
  (∑' n, prob_win_on_turn n) = 1 / 31 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_player_win_prob_l271_27157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l271_27158

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 8

/-- The set of x-coordinates where y = 8 -/
def P : Set ℝ := {x | f x = 8}

/-- The set of x-coordinates where y = -8 -/
def Q : Set ℝ := {x | f x = -8}

/-- The horizontal distance between two x-coordinates -/
def horizontalDistance (x1 x2 : ℝ) : ℝ := |x1 - x2|

theorem shortest_distance :
  ∃ (p q : ℝ), p ∈ P ∧ q ∈ Q ∧
    (∀ (p' q' : ℝ), p' ∈ P → q' ∈ Q → horizontalDistance p q ≤ horizontalDistance p' q') ∧
    horizontalDistance p q = Real.sqrt 17 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l271_27158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_F_minimum_l271_27170

noncomputable section

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x + k / x

-- Define the function F
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := x^2 + 4 / x^2 - 2 * a * (x - 2 / x)

-- Define the minimum value function g
noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ -1 then 5 + 2 * a
  else if a < 1 then -a^2 + 4
  else 5 - 2 * a

-- State the theorem
theorem f_properties_and_F_minimum (k : ℝ) (h : k < 0) :
  (∀ x, f k (-x) = -(f k x)) ∧  -- f is odd
  (∀ x y, 0 < x → x < y → f k x < f k y) ∧  -- f is increasing
  (∀ a, ∀ x ∈ Set.Icc 1 2, F a x ≥ g a) :=  -- g(a) is the minimum of F(x)
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_F_minimum_l271_27170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l271_27197

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the property of being in [0, 1)
def InUnitInterval (s : Sequence) : Prop :=
  ∀ n, 0 ≤ s n ∧ s n < 1

-- Define the property of infinitely many elements in an interval
def InfinitelyManyIn (s : Sequence) (a b : ℝ) : Prop :=
  ∀ N, ∃ n ≥ N, a ≤ s n ∧ s n < b

-- Main theorem
theorem sequence_properties (s : Sequence) (h : InUnitInterval s) :
  (InfinitelyManyIn s 0 (1/2) ∨ InfinitelyManyIn s (1/2) 1) ∧
  (∀ ε > 0, ε < 1/2 → ∃ α : ℚ, (α : ℝ) ∈ Set.Icc 0 1 ∧ InfinitelyManyIn s ((α : ℝ) - ε) ((α : ℝ) + ε)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l271_27197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagoras_academy_distinct_students_l271_27192

/-- Represents a class at Pythagoras Academy -/
structure AcademyClass where
  teacher : String
  students : Nat

/-- Represents the Pythagoras Academy -/
structure Academy where
  classes : List AcademyClass
  double_counted : Nat

/-- Calculates the number of distinct students in the Academy -/
def distinct_students (academy : Academy) : Nat :=
  (academy.classes.map (·.students)).sum - academy.double_counted

theorem pythagoras_academy_distinct_students :
  let academy : Academy := {
    classes := [
      { teacher := "Mr. Archimedes", students := 15 },
      { teacher := "Ms. Euler", students := 10 },
      { teacher := "Mr. Gauss", students := 12 }
    ],
    double_counted := 3
  }
  distinct_students academy = 34 := by
  sorry

#eval distinct_students {
  classes := [
    { teacher := "Mr. Archimedes", students := 15 },
    { teacher := "Ms. Euler", students := 10 },
    { teacher := "Mr. Gauss", students := 12 }
  ],
  double_counted := 3
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagoras_academy_distinct_students_l271_27192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_correlation_coefficient_perfect_positive_l271_27162

-- Define the sample correlation coefficient function
noncomputable def sample_correlation_coefficient (x y : Fin n → ℝ) : ℝ :=
  sorry -- We'll leave the actual implementation as 'sorry' for now

theorem sample_correlation_coefficient_perfect_positive (n : ℕ) 
  (x y : Fin n → ℝ) (h_n : n ≥ 2) 
  (h_not_all_equal : ∃ i j, i ≠ j ∧ x i ≠ x j) 
  (h_line : ∀ i, y i = 2 * x i + 1) :
  sample_correlation_coefficient x y = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_correlation_coefficient_perfect_positive_l271_27162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_plus_pi_4_implies_reciprocal_sin_cos_phi_l271_27109

theorem tan_phi_plus_pi_4_implies_reciprocal_sin_cos_phi (φ : ℝ) :
  Real.tan (φ + π/4) = 5 → 1 / (Real.sin φ * Real.cos φ) = 13/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_plus_pi_4_implies_reciprocal_sin_cos_phi_l271_27109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l271_27187

/-- A four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n ≤ 9999 }

/-- The digit in the thousands place -/
def thousandsDigit (n : FourDigitNumber) : ℕ := n.val / 1000

/-- The digit in the hundreds place -/
def hundredsDigit (n : FourDigitNumber) : ℕ := (n.val / 100) % 10

/-- The digit in the units place -/
def unitsDigit (n : FourDigitNumber) : ℕ := n.val % 10

/-- The set of valid four-digit numbers according to the problem conditions -/
def ValidNumbers : Set FourDigitNumber :=
  { n | thousandsDigit n ≠ 0 ∧ unitsDigit n = hundredsDigit n + 2 }

/-- Provide an instance of Fintype for ValidNumbers -/
instance : Fintype ValidNumbers := by
  sorry

theorem valid_numbers_count : Fintype.card ValidNumbers = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l271_27187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l271_27117

theorem sin_cos_sum (a x y : ℝ) (ha : a ≠ 0) 
  (h1 : Real.sin x + Real.sin y = a) (h2 : Real.cos x + Real.cos y = a) : 
  Real.sin x + Real.cos x = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l271_27117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_passes_through_fixed_point_l271_27159

/-- A parabola that intersects the coordinate axes at three distinct points -/
structure AxisIntersectingParabola where
  a : ℝ
  b : ℝ
  x₁ : ℝ
  x₂ : ℝ
  distinct_intersections : x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ b ≠ 0
  on_parabola_A : x₁^2 + a * x₁ + b = 0
  on_parabola_B : x₂^2 + a * x₂ + b = 0
  on_parabola_C : b = b

/-- The fixed point through which all circumcircles pass -/
def fixed_point : ℝ × ℝ := (0, 1)

/-- Predicate to check if a point lies on the circumcircle of three other points -/
def PointOnCircumcircle (x₁ y₁ x₂ y₂ x₃ y₃ x y : ℝ) : Prop :=
  ∃ (r : ℝ), (x - x₁)^2 + (y - y₁)^2 = r^2 ∧
             (x - x₂)^2 + (y - y₂)^2 = r^2 ∧
             (x - x₃)^2 + (y - y₃)^2 = r^2

/-- The theorem stating that the circumcircle of ABC passes through the fixed point -/
theorem circumcircle_passes_through_fixed_point (p : AxisIntersectingParabola) :
  PointOnCircumcircle p.x₁ 0 p.x₂ 0 0 p.b fixed_point.1 fixed_point.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_passes_through_fixed_point_l271_27159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isabellas_hair_length_l271_27168

/-- Calculates the final length of Isabella's hair in centimeters after growth and haircut -/
theorem isabellas_hair_length 
  (initial_length : ℝ) 
  (growth_rate : ℝ) 
  (weeks : ℕ) 
  (trim_length : ℝ) 
  (inch_to_cm : ℝ)
  (h1 : initial_length = 18)
  (h2 : growth_rate = 0.5)
  (h3 : weeks = 4)
  (h4 : trim_length = 2.25)
  (h5 : inch_to_cm = 2.54)
  : ∃ (final_length : ℝ), final_length = 45.085 := by
  let growth := growth_rate * (weeks : ℝ)
  let before_trim := initial_length + growth
  let after_trim := before_trim - trim_length
  let final_length := after_trim * inch_to_cm
  exists final_length
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isabellas_hair_length_l271_27168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_backpack_pricing_theorem_l271_27138

/-- Represents the backpack pricing and profit model --/
structure BackpackModel where
  cost_price : ℚ
  base_price : ℚ
  base_volume : ℚ
  volume_increase : ℚ
  price_decrease : ℚ
  donation : ℚ

/-- Calculates the profit function for a given price --/
def profit_function (model : BackpackModel) (x : ℚ) : ℚ :=
  let volume := model.base_volume + model.volume_increase * (model.base_price - x) / model.price_decrease
  (x - model.cost_price) * volume

/-- Main theorem about the backpack pricing model --/
theorem backpack_pricing_theorem (model : BackpackModel)
  (h_cost : model.cost_price = 60)
  (h_base_price : model.base_price = 110)
  (h_base_volume : model.base_volume = 300)
  (h_volume_increase : model.volume_increase = 10)
  (h_price_decrease : model.price_decrease = 1)
  (h_donation : model.donation = 1750) :
  -- 1. Profit function
  (∀ x, profit_function model x = -10 * x^2 + 2000 * x - 84000) ∧
  -- 2. Maximum profit
  (∃ x_max, 
    (∀ x, profit_function model x ≤ profit_function model x_max) ∧
    x_max = 100 ∧
    profit_function model x_max = 16000) ∧
  -- 3. Price range for remaining profit
  (∀ x, 85 ≤ x ∧ x ≤ 115 ↔ 
    profit_function model x - model.donation ≥ 12000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_backpack_pricing_theorem_l271_27138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_volume_approx_l271_27124

/-- Represents a cylindrical tank with a salt solution -/
structure SaltTank where
  height : ℝ
  diameter : ℝ
  fillRatio : ℝ
  saltWaterRatio : ℝ

/-- Calculates the volume of salt in the tank -/
noncomputable def saltVolume (tank : SaltTank) : ℝ :=
  let radius := tank.diameter / 2
  let solutionHeight := tank.height * tank.fillRatio
  let solutionVolume := Real.pi * radius^2 * solutionHeight
  let saltRatio := tank.saltWaterRatio / (1 + tank.saltWaterRatio)
  saltRatio * solutionVolume

/-- Theorem stating that the volume of salt in the specified tank is approximately 3.53 cubic feet -/
theorem salt_volume_approx :
  let tank : SaltTank := {
    height := 9,
    diameter := 3,
    fillRatio := 1/3,
    saltWaterRatio := 1/5
  }
  ∃ ε > 0, |saltVolume tank - 3.53| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_volume_approx_l271_27124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l271_27120

-- Define the points A, B, C
noncomputable def A : ℝ × ℝ := (0, 10)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def C : ℝ × ℝ := (10, 0)

-- Define midpoints D and E
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Theorem statement
theorem area_of_triangle_DBC : triangleArea D B C = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l271_27120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_theorem_l271_27190

def vector_problem (m n : ℝ × ℝ) : Prop :=
  let angle := Real.pi / 6
  let mag_m := Real.sqrt 3
  let mag_n := 2
  let AB := (2 * m.1 + 2 * n.1, 2 * m.2 + 2 * n.2)
  let AC := (2 * m.1 - 6 * n.1, 2 * m.2 - 6 * n.2)
  let AD := ((AB.1 + AC.1) / 2, (AB.2 + AC.2) / 2)
  (m.1 * n.1 + m.2 * n.2 = mag_m * mag_n * Real.cos angle) ∧
  (m.1^2 + m.2^2 = mag_m^2) ∧
  (n.1^2 + n.2^2 = mag_n^2) ∧
  (AD.1^2 + AD.2^2 = 16)

theorem vector_problem_theorem (m n : ℝ × ℝ) :
  vector_problem m n → 
  ∃ (AD : ℝ × ℝ), AD.1^2 + AD.2^2 = 16 :=
by
  intro h
  let ⟨_, _, _, h_AD⟩ := h
  exact ⟨_, h_AD⟩

#check vector_problem_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_theorem_l271_27190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_distance_probability_l271_27161

/-- The probability that two randomly chosen points on the perimeter of a unit square
    have a distance less than 1 between them -/
noncomputable def square_perimeter_probability : ℝ := 1/4 + Real.pi/8

/-- Theorem stating that the probability of two randomly chosen points on the perimeter
    of a unit square having a distance less than 1 is equal to 1/4 + π/8 -/
theorem square_perimeter_distance_probability :
  square_perimeter_probability = 1/4 + Real.pi/8 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_distance_probability_l271_27161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l271_27118

/-- Calculates the speed of a train in km/hr given its length and time to pass a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Proves that a train of length 225 meters passing a pole in 9 seconds has a speed of 90 km/hr -/
theorem train_speed_proof :
  train_speed 225 9 = 90 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l271_27118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_root_l271_27193

theorem quadratic_equation_root :
  (Real.sqrt 5 - 3)^2 + 6*(Real.sqrt 5 - 3) - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_root_l271_27193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_odd_function_condition_l271_27100

-- Define the function f as noncomputable
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x + φ)

-- State the theorem
theorem tan_odd_function_condition (ω : ℝ) (hω : ω > 0) :
  (∃ φ : ℝ, f ω φ 0 = 0 → (∀ x, f ω φ x = - f ω φ (-x))) ∧
  (∃ φ : ℝ, (∀ x, f ω φ x = - f ω φ (-x)) ∧ ¬ ∃ y, f ω φ 0 = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_odd_function_condition_l271_27100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equality_l271_27198

/-- Given a digit represented by d, prove that d5 in base 9 equals d2 in base 10 if and only if d equals 3 -/
theorem diamond_equality (d : ℕ) : d < 10 → (d * 9 + 5 = d * 10 + 2 ↔ d = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equality_l271_27198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_distribution_l271_27123

theorem income_distribution (total_income : ℝ) 
  (food_percentage : ℝ) (rent_percentage : ℝ) (remaining_percentage : ℝ) : 
  food_percentage = 50 →
  rent_percentage = 50 →
  remaining_percentage = 17.5 →
  (100 - food_percentage - (remaining_percentage + rent_percentage * (1 - food_percentage / 100))) = 32.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_distribution_l271_27123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_equation_linear_function_unique_l271_27174

/-- A linear function f(x) = kx + b with k > 0, defined on [0, 1] with range [-1, 1] -/
def linear_function (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

theorem linear_function_equation (k b : ℝ) (h_k : k > 0) :
  (∀ x ∈ Set.Icc 0 1, linear_function k b x ∈ Set.Icc (-1) 1) →
  (k = 2 ∧ b = -1) :=
by sorry

theorem linear_function_unique :
  ∀ k b : ℝ, k > 0 →
  (∀ x ∈ Set.Icc 0 1, linear_function k b x ∈ Set.Icc (-1) 1) →
  (∀ x, linear_function k b x = 2 * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_equation_linear_function_unique_l271_27174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ounces_in_pound_l271_27139

/-- Represents the number of ounces in one pound -/
def ounces_per_pound : ℕ → Prop := λ x => x > 0

/-- Represents the weight of a packet in pounds and ounces -/
def packet_weight (lb : ℕ) (oz : ℕ) : ℕ × ℕ := (lb, oz)

/-- Theorem stating that there are 16 ounces in one pound based on the given conditions -/
theorem ounces_in_pound :
  ∀ (x : ℕ),
  (2600 : ℕ) = (1 : ℕ) * (2600 : ℕ) →  -- One ton has 2600 pounds
  (13 : ℕ) * (2600 : ℕ) = (33800 : ℕ) →  -- 13 tons in pounds
  packet_weight 16 4 = (16, 4) →  -- Each packet weighs 16 pounds and 4 ounces
  (2080 : ℕ) * (16 + 4 / x) = (33800 : ℕ) →  -- Total weight equation
  ounces_per_pound x →
  x = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ounces_in_pound_l271_27139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_subgroup_direct_product_l271_27131

variable {G H : Type*} [Group G] [Group H]
variable (G' : Subgroup G) (H' : Subgroup H)

/-- Given two groups G and H with normal subgroups G' and H', 
    the direct product G' × H' is a normal subgroup of G × H -/
theorem normal_subgroup_direct_product 
  (hG' : G'.Normal) (hH' : H'.Normal) : 
  (Subgroup.prod G' H').Normal :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_subgroup_direct_product_l271_27131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_E_equation_and_vertices_l271_27140

-- Define the hyperbola E
def hyperbola_E (x y : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ x^2 / 4 - y^2 / 9 = k

-- Define the reference hyperbola
def reference_hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 9 = 1

-- State the theorem
theorem hyperbola_E_equation_and_vertices :
  (∀ x y : ℝ, hyperbola_E x y ↔ reference_hyperbola x y) →
  hyperbola_E 2 (3 * Real.sqrt 5) →
  (∀ x y : ℝ, hyperbola_E x y ↔ y^2 / 36 - x^2 / 16 = 1) ∧
  (hyperbola_E 0 6 ∧ hyperbola_E 0 (-6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_E_equation_and_vertices_l271_27140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_l271_27141

/-- A straight line in the xy-plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

theorem point_on_line (l : Line) (p : Point) :
  l.slope = 4 ∧ l.y_intercept = 100 ∧ p.x = 50 ∧ p.on_line l → p.y = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_l271_27141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_over_1_plus_sin_2theta_l271_27171

theorem cos_2theta_over_1_plus_sin_2theta (θ : ℝ) : 
  3 = 5 * Real.cos θ ∧ 4 = 5 * Real.sin θ → 
  (Real.cos (2 * θ)) / (1 + Real.sin (2 * θ)) = -(1 / 7) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_over_1_plus_sin_2theta_l271_27171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_integers_l271_27101

theorem count_satisfying_integers : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n ≤ 800 ∧ Even n ∧ n > 0 ∧ ∀ t : ℝ, (Complex.sin t - Complex.I * Complex.cos t) ^ n = Complex.sin (n * t) - Complex.I * Complex.cos (n * t)) ∧
  S.card = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_integers_l271_27101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_a_range_l271_27173

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 2*a else x^2 - 4*a*x + a

theorem three_zeros_a_range (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  a > 1/4 ∧ a ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_a_range_l271_27173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l271_27146

-- Define the power function as noncomputable
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x^a

-- State the theorem
theorem power_function_property (a : ℝ) :
  let f := power_function a
  (f 4 / f 2 = 3) → f (1/2) = 1/3 := by
  intro h
  -- The proof goes here
  sorry

#check power_function_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l271_27146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_4_75_l271_27194

noncomputable def floor_eq (x : ℝ) : ℤ := ⌊x⌋

theorem smallest_solution_is_4_75 :
  (∀ y : ℝ, y > 0 ∧ y < (19 : ℝ) / 4 → ⌊y^2⌋ - y * (floor_eq y) ≠ 3) ∧
  ⌊((19 : ℝ) / 4)^2⌋ - ((19 : ℝ) / 4) * (floor_eq ((19 : ℝ) / 4)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_4_75_l271_27194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_1994_2_undetermined_l271_27106

/-- The function g as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := (2 - x) / (3 + 2 * x)

/-- The nth iteration of g -/
noncomputable def g_n : ℕ → ℝ → ℝ
| 0 => id
| n + 1 => g ∘ g_n n

/-- The statement that g₁₉₉₄(2) requires further analysis -/
theorem g_1994_2_undetermined : 
  ∃ (result : ℝ), g_n 1994 2 = result ∧ 
  (∀ (m : ℕ) (x : ℝ), m < 1994 → g_n m x ≠ result) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_1994_2_undetermined_l271_27106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l271_27181

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 6 / x - x^2

-- State the theorem
theorem zero_point_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l271_27181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_inequality_l271_27169

/-- Represents a tetrahedron with given edge lengths -/
structure Tetrahedron (α : Type*) [LinearOrderedField α] :=
  (a b c d e f : α)
  (pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)

/-- Volume of a tetrahedron (placeholder function) -/
noncomputable def tetrahedronVolume {α : Type*} [LinearOrderedField α] (t : Tetrahedron α) : α :=
  sorry -- Actual implementation would go here

theorem tetrahedron_volume_inequality 
  {α : Type*} [LinearOrderedField α] [Archimedean α]
  (t : Tetrahedron α) (V : α)
  (h_volume : V > 0)
  (h_V : V = tetrahedronVolume t) :
  3 * V ≤ ((t.a^2 + t.b^2 + t.c^2 + t.d^2 + t.e^2 + t.f^2) / 12) ^ (3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_inequality_l271_27169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientist_umbrella_theorem_l271_27102

/-- The probability that the scientist takes an umbrella -/
noncomputable def umbrella_prob : ℝ := 0.2

/-- The probability of rain on any given day -/
noncomputable def rain_prob : ℝ := 1 / 9

/-- The probability of having no umbrella at the destination -/
noncomputable def no_umbrella_prob (x : ℝ) : ℝ := x / (x + 1)

theorem scientist_umbrella_theorem :
  ∀ x : ℝ, 0 < x → x < 1 →
  (x + no_umbrella_prob x - x * no_umbrella_prob x = umbrella_prob) →
  x = rain_prob := by
  sorry

#check scientist_umbrella_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientist_umbrella_theorem_l271_27102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_roots_is_two_l271_27195

/-- The function f(x) = x^3 - 2x^2 -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2

/-- The number of real roots of f(x) = 0 is 2 -/
theorem number_of_roots_is_two : ∃ (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_roots_is_two_l271_27195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_a_values_l271_27142

theorem product_of_a_values : ∃ (S : Finset ℤ),
  (∀ a ∈ S, (∃! (odd_solutions : Finset ℤ), 
    (∀ x ∈ odd_solutions, Odd x ∧ (x + 3) / 2 ≥ x - 1 ∧ 3 * x + 6 > a + 4) ∧ 
    Finset.card odd_solutions = 3) ∧
   (∃ y : ℤ, y ≥ 0 ∧ 3 * y + 6 * a = 22 - y)) ∧
  (S.prod id) = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_a_values_l271_27142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_line_passes_through_intersection_l271_27155

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point of tangency between two circles -/
structure TangencyPoint where
  point : ℝ × ℝ
  circle1 : Circle
  circle2 : Circle

/-- Defines a line segment between two points -/
def line_segment (a b : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • a + t • b}

/-- Defines the set of points in a circle -/
def circle_set (c : Circle) : Set (ℝ × ℝ) :=
  {p | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2}

theorem tangency_line_passes_through_intersection
  (C : Circle) (C1 C2 : Circle) (A B : TangencyPoint)
  (h1 : C1.radius + C2.radius = C.radius)
  (h2 : A.circle1 = C ∧ A.circle2 = C1)
  (h3 : B.circle1 = C ∧ B.circle2 = C2)
  (h4 : C1.center ≠ C2.center) :
  ∃ (P : ℝ × ℝ), P ∈ line_segment A.point B.point ∧
                 P ∈ Set.inter (circle_set C1) (circle_set C2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_line_passes_through_intersection_l271_27155
