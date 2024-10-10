import Mathlib

namespace ruffy_orlie_age_difference_l1043_104375

/-- Proves that given Ruffy's current age is 9 and Ruffy is three-fourths as old as Orlie,
    the difference between Ruffy's age and half of Orlie's age four years ago is 1 year. -/
theorem ruffy_orlie_age_difference : ∀ (ruffy_age orlie_age : ℕ),
  ruffy_age = 9 →
  ruffy_age = (3 * orlie_age) / 4 →
  (ruffy_age - 4) - ((orlie_age - 4) / 2) = 1 :=
by
  sorry

end ruffy_orlie_age_difference_l1043_104375


namespace unique_intersection_point_l1043_104341

def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

theorem unique_intersection_point :
  ∃! a : ℝ, f a = a ∧ a = -1 := by sorry

end unique_intersection_point_l1043_104341


namespace cos_225_degrees_l1043_104376

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_degrees_l1043_104376


namespace orange_theorem_l1043_104344

def orange_problem (betty_oranges bill_oranges frank_multiplier seeds_per_orange oranges_per_tree : ℕ) : ℕ :=
  let total_betty_bill := betty_oranges + bill_oranges
  let frank_oranges := frank_multiplier * total_betty_bill
  let total_seeds := frank_oranges * seeds_per_orange
  let total_oranges := total_seeds * oranges_per_tree
  total_oranges

theorem orange_theorem : 
  orange_problem 15 12 3 2 5 = 810 := by
  sorry

end orange_theorem_l1043_104344


namespace square_sum_ge_product_sum_l1043_104378

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end square_sum_ge_product_sum_l1043_104378


namespace tetrahedron_side_length_l1043_104377

/-- The side length of a regular tetrahedron given its square shadow area -/
theorem tetrahedron_side_length (shadow_area : ℝ) (h : shadow_area = 16) :
  ∃ (side_length : ℝ), side_length = 4 * Real.sqrt 2 ∧
  side_length * side_length = 2 * shadow_area :=
sorry

end tetrahedron_side_length_l1043_104377


namespace infinite_binomial_congruence_pairs_l1043_104370

theorem infinite_binomial_congruence_pairs :
  ∀ p : ℕ, Prime p → p ≠ 2 →
  ∃ a b : ℕ,
    a > b ∧
    a + b = 2 * p ∧
    (Nat.choose (2 * p) a) % (2 * p) = (Nat.choose (2 * p) b) % (2 * p) ∧
    (Nat.choose (2 * p) a) % (2 * p) ≠ 0 :=
by sorry

end infinite_binomial_congruence_pairs_l1043_104370


namespace lcm_of_8_9_5_10_l1043_104347

theorem lcm_of_8_9_5_10 : Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 5 10)) = 360 := by
  sorry

end lcm_of_8_9_5_10_l1043_104347


namespace tickets_to_be_sold_l1043_104358

theorem tickets_to_be_sold (total : ℕ) (jude andrea sandra : ℕ) : 
  total = 100 → 
  andrea = 2 * jude → 
  sandra = jude / 2 + 4 → 
  jude = 16 → 
  total - (jude + andrea + sandra) = 40 := by
sorry

end tickets_to_be_sold_l1043_104358


namespace chords_from_eight_points_l1043_104316

/-- The number of chords that can be drawn from n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords from 8 points on a circle is 28 -/
theorem chords_from_eight_points : num_chords 8 = 28 := by
  sorry

end chords_from_eight_points_l1043_104316


namespace chord_of_contact_ellipse_l1043_104379

/-- Given an ellipse and a point outside it, the chord of contact has a specific equation. -/
theorem chord_of_contact_ellipse (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_outside : (x₀^2 / a^2) + (y₀^2 / b^2) > 1) :
  ∃ (A B : ℝ × ℝ), 
    (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧ 
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
    (∀ (x y : ℝ), ((x₀ * x) / a^2 + (y₀ * y) / b^2 = 1) ↔ 
      ∃ (t : ℝ), x = A.1 + t * (B.1 - A.1) ∧ y = A.2 + t * (B.2 - A.2)) := by
  sorry

end chord_of_contact_ellipse_l1043_104379


namespace unique_three_digit_number_l1043_104389

/-- A three-digit number satisfying specific conditions -/
def three_digit_number (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  a + b + c = 10 ∧
  b = a + c ∧
  100 * c + 10 * b + a = 100 * a + 10 * b + c + 99

theorem unique_three_digit_number :
  ∃! (a b c : ℕ), three_digit_number a b c ∧ 100 * a + 10 * b + c = 253 :=
sorry

end unique_three_digit_number_l1043_104389


namespace number_of_subsets_A_l1043_104363

def U : Finset ℕ := {0, 1, 2}

theorem number_of_subsets_A (A : Finset ℕ) (h : U \ A = {2}) : Finset.card (Finset.powerset A) = 4 := by
  sorry

end number_of_subsets_A_l1043_104363


namespace fourth_degree_polynomial_theorem_l1043_104320

/-- A fourth-degree polynomial with real coefficients -/
def FourthDegreePolynomial : Type := ℝ → ℝ

/-- The condition that |g(x)| = 6 for x = 0, 1, 3, 4, 5 -/
def SatisfiesCondition (g : FourthDegreePolynomial) : Prop :=
  |g 0| = 6 ∧ |g 1| = 6 ∧ |g 3| = 6 ∧ |g 4| = 6 ∧ |g 5| = 6

theorem fourth_degree_polynomial_theorem (g : FourthDegreePolynomial) 
  (h : SatisfiesCondition g) : |g 7| = 106.8 := by
  sorry

end fourth_degree_polynomial_theorem_l1043_104320


namespace ratio_of_numbers_l1043_104327

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) :
  a / b = 2 := by
sorry

end ratio_of_numbers_l1043_104327


namespace sum_of_rationals_l1043_104372

theorem sum_of_rationals (a b : ℚ) (h : a + Real.sqrt 3 * b = Real.sqrt (4 + 2 * Real.sqrt 3)) : 
  a + b = 2 := by
  sorry

end sum_of_rationals_l1043_104372


namespace rectangle_diagonal_intersection_l1043_104383

/-- Given a rectangle with opposite vertices at (2,-3) and (14,9),
    the point where the diagonals intersect has coordinates (8, 3) -/
theorem rectangle_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (14, 9)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 3) := by
  sorry

end rectangle_diagonal_intersection_l1043_104383


namespace two_equal_real_roots_l1043_104381

def quadratic_equation (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem two_equal_real_roots (a b c : ℝ) (ha : a ≠ 0) :
  a = 4 ∧ b = -4 ∧ c = 1 →
  ∃ x : ℝ, quadratic_equation a b c x ∧
    ∀ y : ℝ, quadratic_equation a b c y → y = x :=
by
  sorry

end two_equal_real_roots_l1043_104381


namespace solution_set_part1_solution_set_part2_l1043_104346

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | |x - 2| ≥ 4 - |x - 4|} = {x : ℝ | x ≤ 1 ∨ x ≥ 5} :=
by sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 1) :
  ({x : ℝ | |f a (2*x + a) - 2*f a x| ≤ 2} = {x : ℝ | 1 ≤ x ∧ x ≤ 2}) →
  a = 3 :=
by sorry

end solution_set_part1_solution_set_part2_l1043_104346


namespace pie_eating_contest_l1043_104326

theorem pie_eating_contest (student1 student2 student3 : ℚ) 
  (h1 : student1 = 5/6)
  (h2 : student2 = 7/8)
  (h3 : student3 = 1/2) :
  student1 + student2 - student3 = 29/24 := by
  sorry

end pie_eating_contest_l1043_104326


namespace jar_flipping_problem_l1043_104371

theorem jar_flipping_problem (total_jars : Nat) (max_flip_per_move : Nat) (n_upper_bound : Nat) : 
  total_jars = 343 →
  max_flip_per_move = 27 →
  n_upper_bound = 2021 →
  (∃ (n : Nat), n ≥ (total_jars + max_flip_per_move - 1) / max_flip_per_move ∧ 
                n ≤ n_upper_bound ∧
                n % 2 = 1) →
  (Finset.filter (fun x => x % 2 = 1) (Finset.range (n_upper_bound + 1))).card = 1005 := by
sorry

end jar_flipping_problem_l1043_104371


namespace inequality_proof_l1043_104398

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a^2 - a*b + b^2) ≥ (a + b) / 2 := by
  sorry

end inequality_proof_l1043_104398


namespace abby_peeled_22_l1043_104310

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  abby_rate : ℕ
  homer_solo_time : ℕ

/-- Calculates the number of potatoes Abby peeled -/
def abby_peeled (scenario : PotatoPeeling) : ℕ :=
  let homer_solo := scenario.homer_rate * scenario.homer_solo_time
  let remaining := scenario.total_potatoes - homer_solo
  let combined_rate := scenario.homer_rate + scenario.abby_rate
  let combined_time := remaining / combined_rate
  scenario.abby_rate * combined_time

/-- The main theorem stating that Abby peeled 22 potatoes -/
theorem abby_peeled_22 (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 60)
  (h2 : scenario.homer_rate = 4)
  (h3 : scenario.abby_rate = 6)
  (h4 : scenario.homer_solo_time = 6) :
  abby_peeled scenario = 22 := by
  sorry

end abby_peeled_22_l1043_104310


namespace set_operations_l1043_104309

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x > 0}

-- Define the complement of B in ℝ
def C_R_B : Set ℝ := {x : ℝ | x ≤ 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | 0 < x ∧ x < 2}) ∧
  (C_R_B ∪ A = {x : ℝ | x < 2}) := by
  sorry

end set_operations_l1043_104309


namespace outfit_combinations_l1043_104308

/-- Represents the number of items of each type (shirts, pants, hats) -/
def num_items : ℕ := 7

/-- Represents the number of colors available for each item type -/
def num_colors : ℕ := 7

/-- Calculates the number of valid outfit combinations where no two items are the same color -/
def valid_outfits : ℕ := num_colors * (num_colors - 1) * (num_colors - 2)

/-- Proves that the number of valid outfit combinations is 210 -/
theorem outfit_combinations :
  valid_outfits = 210 :=
by sorry

end outfit_combinations_l1043_104308


namespace f_at_two_l1043_104399

/-- The polynomial function f(x) = x^6 - 2x^5 + 3x^3 + 4x^2 - 6x + 5 -/
def f (x : ℝ) : ℝ := x^6 - 2*x^5 + 3*x^3 + 4*x^2 - 6*x + 5

/-- Theorem: The value of f(2) is 29 -/
theorem f_at_two : f 2 = 29 := by
  sorry

end f_at_two_l1043_104399


namespace water_pouring_game_score_l1043_104313

/-- Represents the players in the game -/
inductive Player
| Xiaoming
| Xiaolin

/-- Defines the scoring rules for the water pouring game -/
def score (overflowPlayer : Option Player) : Nat :=
  match overflowPlayer with
  | some Player.Xiaoming => 10
  | some Player.Xiaolin => 9
  | none => 3

/-- Represents a round in the game -/
structure Round where
  xiaomingPour : Nat
  xiaolinPour : Nat
  overflowPlayer : Option Player

/-- The three rounds of the game -/
def round1 : Round := ⟨5, 5, some Player.Xiaolin⟩
def round2 : Round := ⟨2, 7, none⟩
def round3 : Round := ⟨13, 0, some Player.Xiaoming⟩

/-- Calculates the total score for the given rounds -/
def totalScore (rounds : List Round) : Nat :=
  rounds.foldl (fun acc r => acc + score r.overflowPlayer) 0

/-- The main theorem to prove -/
theorem water_pouring_game_score :
  totalScore [round1, round2, round3] = 22 := by
  sorry

end water_pouring_game_score_l1043_104313


namespace inequality_proof_l1043_104329

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a > b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
  sorry

end inequality_proof_l1043_104329


namespace intersection_characterization_l1043_104359

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 2*x < 0}

theorem intersection_characterization :
  ∀ x : ℝ, x ∈ (M ∩ N) ↔ (1 < x ∧ x < 2) :=
sorry

end intersection_characterization_l1043_104359


namespace max_correct_answers_l1043_104302

/-- Represents the result of a math exam -/
structure ExamResult where
  totalQuestions : ℕ
  correctAnswers : ℕ
  wrongAnswers : ℕ
  unansweredQuestions : ℕ
  score : ℤ

/-- Calculates the score based on the exam rules -/
def calculateScore (result : ExamResult) : ℤ :=
  result.correctAnswers - 3 * result.wrongAnswers - 2 * result.unansweredQuestions

/-- Checks if the exam result is valid according to the given conditions -/
def isValidExamResult (result : ExamResult) : Prop :=
  result.totalQuestions = 100 ∧
  result.correctAnswers + result.wrongAnswers + result.unansweredQuestions = result.totalQuestions ∧
  calculateScore result = 50

/-- Theorem stating that the maximum number of correct answers is 87 -/
theorem max_correct_answers (result : ExamResult) :
  isValidExamResult result → result.correctAnswers ≤ 87 := by
  sorry

#check max_correct_answers

end max_correct_answers_l1043_104302


namespace all_terms_irrational_l1043_104334

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define the property of √2 and √3 being in the sequence
def sqrt2_sqrt3_in_sequence (a : ℕ → ℝ) : Prop :=
  ∃ m n : ℕ, a m = Real.sqrt 2 ∧ a n = Real.sqrt 3

-- Theorem statement
theorem all_terms_irrational
  (a : ℕ → ℝ)
  (h1 : is_arithmetic_sequence a)
  (h2 : sqrt2_sqrt3_in_sequence a) :
  ∀ n : ℕ, Irrational (a n) :=
sorry

end all_terms_irrational_l1043_104334


namespace base4_calculation_l1043_104333

/-- Converts a base 4 number to base 10 --/
def base4_to_base10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 --/
def base10_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Theorem: In base 4, (1230₄ + 32₄) ÷ 13₄ = 111₄ --/
theorem base4_calculation : 
  let a := base4_to_base10 [0, 3, 2, 1]  -- 1230₄
  let b := base4_to_base10 [2, 3]        -- 32₄
  let c := base4_to_base10 [3, 1]        -- 13₄
  base10_to_base4 ((a + b) / c) = [1, 1, 1] := by
  sorry


end base4_calculation_l1043_104333


namespace farmer_ploughing_problem_l1043_104360

/-- Farmer's field ploughing problem -/
theorem farmer_ploughing_problem 
  (initial_daily_area : ℝ) 
  (productivity_increase : ℝ) 
  (days_ahead : ℕ) 
  (total_field_area : ℝ) 
  (h1 : initial_daily_area = 120)
  (h2 : productivity_increase = 0.25)
  (h3 : days_ahead = 2)
  (h4 : total_field_area = 1440) :
  ∃ (planned_days : ℕ) (actual_days : ℕ),
    planned_days = 10 ∧ 
    actual_days = planned_days - days_ahead ∧
    actual_days * initial_daily_area + 
      (planned_days - actual_days) * (initial_daily_area * (1 + productivity_increase)) = 
    total_field_area :=
by sorry

end farmer_ploughing_problem_l1043_104360


namespace equation_solution_l1043_104348

theorem equation_solution : ∃! x : ℝ, (3 / (x - 3) = 1 / (x - 1)) ∧ x = 0 := by
  sorry

end equation_solution_l1043_104348


namespace union_of_sets_l1043_104324

theorem union_of_sets (a : ℝ) : 
  let A : Set ℝ := {a^2 + 1, 2*a}
  let B : Set ℝ := {a + 1, 0}
  (A ∩ B).Nonempty → A ∪ B = {0, 1} := by
  sorry

end union_of_sets_l1043_104324


namespace correct_result_l1043_104332

def add_subtract_round (a b c : ℕ) : ℕ :=
  let result := a + b - c
  let remainder := result % 5
  if remainder < 3 then result - remainder else result + (5 - remainder)

theorem correct_result : add_subtract_round 82 56 15 = 125 := by
  sorry

end correct_result_l1043_104332


namespace sphere_surface_area_of_circumscribed_cube_l1043_104314

theorem sphere_surface_area_of_circumscribed_cube (R : ℝ) :
  let cube_edge_1 : ℝ := 2
  let cube_edge_2 : ℝ := 3
  let cube_edge_3 : ℝ := 1
  let cube_diagonal : ℝ := (cube_edge_1^2 + cube_edge_2^2 + cube_edge_3^2).sqrt
  R = cube_diagonal / 2 →
  4 * Real.pi * R^2 = 14 * Real.pi :=
by sorry

end sphere_surface_area_of_circumscribed_cube_l1043_104314


namespace union_A_B_when_a_is_4_intersection_A_B_equals_A_l1043_104307

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | (4 - x) * (x - 1) ≤ 0}

-- Statement 1
theorem union_A_B_when_a_is_4 : 
  A 4 ∪ B = {x | x ≥ 3 ∨ x ≤ 1} := by sorry

-- Statement 2
theorem intersection_A_B_equals_A (a : ℝ) : 
  A a ∩ B = A a ↔ a ≥ 5 ∨ a ≤ 0 := by sorry

end union_A_B_when_a_is_4_intersection_A_B_equals_A_l1043_104307


namespace ellipse_slope_theorem_l1043_104301

/-- Given an ellipse with specific properties, prove the slope of a line passing through a point on the ellipse --/
theorem ellipse_slope_theorem (F₁ PF : ℝ) (k₂ : ℝ) :
  F₁ = (6/5) * Real.sqrt 5 →
  PF = (4/5) * Real.sqrt 5 →
  ∃ (k : ℝ), k = (3/2) * k₂ ∧ (k = (3 * Real.sqrt 5) / 10 ∨ k = -(3 * Real.sqrt 5) / 10) := by
  sorry

end ellipse_slope_theorem_l1043_104301


namespace peanut_eating_interval_l1043_104368

/-- Proves that given a flight duration of 2 hours and 4 bags of peanuts with 30 peanuts each,
    if all peanuts are consumed at equally spaced intervals during the flight,
    the time between eating each peanut is 1 minute. -/
theorem peanut_eating_interval (flight_duration : ℕ) (bags : ℕ) (peanuts_per_bag : ℕ) :
  flight_duration = 2 →
  bags = 4 →
  peanuts_per_bag = 30 →
  (flight_duration * 60) / (bags * peanuts_per_bag) = 1 := by
  sorry

#check peanut_eating_interval

end peanut_eating_interval_l1043_104368


namespace red_sector_overlap_l1043_104323

theorem red_sector_overlap (n : ℕ) (red_sectors : ℕ) (h1 : n = 1965) (h2 : red_sectors = 200) :
  ∃ (positions : Finset ℕ), 
    (Finset.card positions ≥ 60) ∧ 
    (∀ p ∈ positions, p < n) ∧
    (∀ p ∈ positions, (red_sectors * red_sectors - n * 20) / n ≤ red_sectors - 
      (red_sectors * red_sectors - (n - p) * red_sectors) / n) :=
sorry

end red_sector_overlap_l1043_104323


namespace number_problem_l1043_104350

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 17 → (40/100 : ℝ) * N = 204 := by
  sorry

end number_problem_l1043_104350


namespace perfectSquareFactors_360_l1043_104385

/-- A function that returns the number of perfect square factors of a natural number -/
def perfectSquareFactors (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of perfect square factors of 360 is 4 -/
theorem perfectSquareFactors_360 : perfectSquareFactors 360 = 4 := by sorry

end perfectSquareFactors_360_l1043_104385


namespace rotated_square_height_l1043_104392

/-- The distance of point B from the original line when a square is rotated -/
theorem rotated_square_height (side_length : ℝ) (rotation_angle : ℝ) : 
  side_length = 4 →
  rotation_angle = 30 * π / 180 →
  let diagonal := side_length * Real.sqrt 2
  let height := (diagonal / 2) * Real.sin rotation_angle
  height = Real.sqrt 2 := by
  sorry

end rotated_square_height_l1043_104392


namespace quadratic_equation_roots_l1043_104357

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + m*y + 3 = 0 ∧ y = 3 ∧ m = -4) := by
sorry

end quadratic_equation_roots_l1043_104357


namespace scientific_notation_correct_l1043_104352

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- The number to be converted to scientific notation -/
def original_number : ℕ := 189130000000

/-- Function to convert a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem scientific_notation_correct :
  let sn := to_scientific_notation original_number
  sn.coefficient = 1.8913 ∧ sn.exponent = 11 := by sorry

end scientific_notation_correct_l1043_104352


namespace max_xyz_value_l1043_104382

theorem max_xyz_value (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum : x + y + z = 2) (h_sq_sum : x^2 + y^2 + z^2 = x*z + y*z + x*y) :
  x*y*z ≤ 8/27 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = 2 ∧ a^2 + b^2 + c^2 = a*c + b*c + a*b ∧ a*b*c = 8/27 := by
  sorry

end max_xyz_value_l1043_104382


namespace equal_expressions_l1043_104349

theorem equal_expressions : 
  (2^3 ≠ 3^2) ∧ 
  (-3^3 = (-3)^3) ∧ 
  (-2^2 ≠ (-2)^2) ∧ 
  (-|-2| ≠ -(-2)) := by
  sorry

end equal_expressions_l1043_104349


namespace nine_expressions_cover_1_to_13_l1043_104337

def nine_expressions : List (ℕ → Prop) :=
  [ (λ n => n = ((9 / 9) ^ (9 - 9))),
    (λ n => n = ((9 / 9) + (9 / 9))),
    (λ n => n = ((9 / 9) + (9 / 9) + (9 / 9))),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 2),
    (λ n => n = ((9 * 9 + 9) / 9 - 9) + 9),
    (λ n => n = ((9 / 9) + (9 / 9) + (9 / 9)) ^ 2 - 3),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 2 - (9 / 9)),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 3 - (9 / 9)),
    (λ n => n = 9),
    (λ n => n = (99 - 9) / 9),
    (λ n => n = 9 + (9 / 9) + (9 / 9)),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 3 - 4),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 2 + 9) ]

theorem nine_expressions_cover_1_to_13 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 13 → ∃ expr ∈ nine_expressions, expr n :=
sorry

end nine_expressions_cover_1_to_13_l1043_104337


namespace unknown_number_proof_l1043_104366

theorem unknown_number_proof (x : ℝ) : 3034 - (x / 200.4) = 3029 → x = 1002 := by
  sorry

end unknown_number_proof_l1043_104366


namespace dot_product_equals_one_l1043_104318

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_equals_one :
  (2 • a + b) • a = 1 := by sorry

end dot_product_equals_one_l1043_104318


namespace total_yellow_balls_is_30_l1043_104322

/-- The number of boxes containing balls -/
def num_boxes : ℕ := 6

/-- The number of yellow balls in each box -/
def yellow_balls_per_box : ℕ := 5

/-- The total number of yellow balls across all boxes -/
def total_yellow_balls : ℕ := num_boxes * yellow_balls_per_box

theorem total_yellow_balls_is_30 : total_yellow_balls = 30 := by
  sorry

end total_yellow_balls_is_30_l1043_104322


namespace cone_properties_l1043_104356

-- Define the cone
structure Cone where
  generatrix : ℝ
  base_diameter : ℝ

-- Define the theorem
theorem cone_properties (c : Cone) 
  (h1 : c.generatrix = 2 * Real.sqrt 5)
  (h2 : c.base_diameter = 4) :
  -- 1. Volume of the cone
  let volume := (1/3) * Real.pi * (c.base_diameter/2)^2 * Real.sqrt ((2*Real.sqrt 5)^2 - (c.base_diameter/2)^2)
  volume = (16/3) * Real.pi ∧
  -- 2. Minimum distance from any point on a parallel section to the vertex
  let min_distance := (4/5) * Real.sqrt 5
  (∀ r : ℝ, r > 0 → r < c.base_diameter/2 → 
    ∀ p : ℝ × ℝ, p.1^2 + p.2^2 = r^2 → 
      p.1^2 + (Real.sqrt ((2*Real.sqrt 5)^2 - (c.base_diameter/2)^2) - (c.base_diameter/2 - r))^2 ≥ min_distance^2) ∧
  -- 3. Area of the section when it's the center of the circumscribed sphere
  let section_radius := (3/5) * c.base_diameter/2
  Real.pi * section_radius^2 = (36/25) * Real.pi := by
sorry

end cone_properties_l1043_104356


namespace total_tickets_sold_l1043_104390

theorem total_tickets_sold (adult_price student_price total_amount student_count : ℕ) 
  (h1 : adult_price = 12)
  (h2 : student_price = 6)
  (h3 : total_amount = 16200)
  (h4 : student_count = 300) :
  ∃ (adult_count : ℕ), 
    adult_count * adult_price + student_count * student_price = total_amount ∧
    adult_count + student_count = 1500 := by
  sorry

end total_tickets_sold_l1043_104390


namespace existence_of_symmetry_axes_l1043_104384

/-- A bounded planar figure. -/
structure BoundedPlanarFigure where
  -- Define properties of a bounded planar figure
  is_bounded : Bool
  is_planar : Bool

/-- An axis of symmetry for a bounded planar figure. -/
structure AxisOfSymmetry (F : BoundedPlanarFigure) where
  -- Define properties of an axis of symmetry

/-- The number of axes of symmetry for a bounded planar figure. -/
def num_axes_of_symmetry (F : BoundedPlanarFigure) : Nat :=
  sorry

/-- Theorem: There exist bounded planar figures with exactly three axes of symmetry,
    and there exist bounded planar figures with more than three axes of symmetry. -/
theorem existence_of_symmetry_axes :
  (∃ F : BoundedPlanarFigure, F.is_bounded ∧ F.is_planar ∧ num_axes_of_symmetry F = 3) ∧
  (∃ G : BoundedPlanarFigure, G.is_bounded ∧ G.is_planar ∧ num_axes_of_symmetry G > 3) :=
by sorry

end existence_of_symmetry_axes_l1043_104384


namespace quadratic_root_in_interval_l1043_104328

theorem quadratic_root_in_interval
  (a b c : ℝ)
  (h_two_roots : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0)
  (h_inequality : |a * (b - c)| > |b^2 - a * c| + |c^2 - a * b|) :
  ∃ α : ℝ, 0 < α ∧ α < 2 ∧ a * α^2 + b * α + c = 0 :=
by sorry

end quadratic_root_in_interval_l1043_104328


namespace total_books_l1043_104345

/-- Given an initial number of books and a number of books bought, 
    the total number of books is equal to the sum of the initial number and the number bought. -/
theorem total_books (initial_books bought_books : ℕ) :
  initial_books + bought_books = initial_books + bought_books :=
by sorry

end total_books_l1043_104345


namespace square_probability_is_correct_l1043_104303

/-- The number of squares in a 6x6 grid -/
def gridSize : ℕ := 36

/-- The number of squares to be selected -/
def selectCount : ℕ := 4

/-- The number of ways to select 4 squares from 36 squares -/
def totalSelections : ℕ := Nat.choose gridSize selectCount

/-- The number of ways to select 4 squares that form a square -/
def favorableSelections : ℕ := 105

/-- The probability of selecting 4 squares that form a square -/
def squareProbability : ℚ := favorableSelections / totalSelections

theorem square_probability_is_correct : squareProbability = 1 / 561 := by
  sorry

#eval squareProbability

end square_probability_is_correct_l1043_104303


namespace friday_lunch_customers_l1043_104336

theorem friday_lunch_customers (breakfast : ℕ) (dinner : ℕ) (saturday_prediction : ℕ) :
  breakfast = 73 →
  dinner = 87 →
  saturday_prediction = 574 →
  ∃ (lunch : ℕ), lunch = saturday_prediction / 2 - breakfast - dinner ∧ lunch = 127 :=
by sorry

end friday_lunch_customers_l1043_104336


namespace independence_test_smoking_lung_disease_l1043_104373

-- Define the variables and constants
variable (K : ℝ)
variable (confidence_level : ℝ)
variable (error_rate : ℝ)

-- Define the relationship between smoking and lung disease
def smoking_related_to_lung_disease : Prop := sorry

-- Define the critical value for K^2
def critical_value : ℝ := 6.635

-- Define the theorem
theorem independence_test_smoking_lung_disease :
  K ≥ critical_value →
  confidence_level = 0.99 →
  error_rate = 1 - confidence_level →
  smoking_related_to_lung_disease ∧
  (smoking_related_to_lung_disease → error_rate = 0.01) :=
by sorry

end independence_test_smoking_lung_disease_l1043_104373


namespace annas_vegetable_patch_area_l1043_104362

/-- Represents a rectangular enclosure with fence posts -/
structure FencedRectangle where
  total_posts : ℕ
  post_spacing : ℝ
  long_side_post_ratio : ℕ

/-- Calculates the area of a fenced rectangle -/
def calculate_area (fence : FencedRectangle) : ℝ :=
  let short_side_posts := (fence.total_posts + 4) / (2 * (fence.long_side_post_ratio + 1))
  let long_side_posts := fence.long_side_post_ratio * short_side_posts
  let short_side_length := (short_side_posts - 1) * fence.post_spacing
  let long_side_length := (long_side_posts - 1) * fence.post_spacing
  short_side_length * long_side_length

/-- Theorem stating that the area of Anna's vegetable patch is 144 square meters -/
theorem annas_vegetable_patch_area :
  let fence := FencedRectangle.mk 24 3 3
  calculate_area fence = 144 := by sorry

end annas_vegetable_patch_area_l1043_104362


namespace complex_sum_problem_l1043_104343

theorem complex_sum_problem (p q r s t u : ℝ) : 
  q = 4 → 
  t = -p - r → 
  (p + q * I) + (r + s * I) + (t + u * I) = 3 * I → 
  s + u = -1 := by
sorry

end complex_sum_problem_l1043_104343


namespace inequality_properties_l1043_104306

theorem inequality_properties (a b c d : ℝ) :
  (∀ (a b c : ℝ), c ≠ 0 → a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b c d : ℝ), a > b → c > d → a + c > b + d) ∧
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ ¬(a * c > b * d)) ∧
  (∃ (a b : ℝ), a > b ∧ ¬(1 / a > 1 / b)) :=
by sorry

end inequality_properties_l1043_104306


namespace triangle_equilateral_iff_equation_l1043_104354

/-- A triangle ABC with side lengths a, b, and c is equilateral if and only if
    a^4 + b^4 + c^4 - a^2b^2 - b^2c^2 - a^2c^2 = 0 -/
theorem triangle_equilateral_iff_equation (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  a^4 + b^4 + c^4 - a^2*b^2 - b^2*c^2 - a^2*c^2 = 0 ↔ a = b ∧ b = c := by
  sorry

end triangle_equilateral_iff_equation_l1043_104354


namespace alpha_value_at_negative_six_l1043_104311

/-- Given that α is inversely proportional to β², prove that α = 1/3 when β = -6,
    given the condition that α = 3 when β = 2. -/
theorem alpha_value_at_negative_six (α β : ℝ) (h : ∃ k, ∀ β ≠ 0, α = k / β^2) 
    (h_condition : α = 3 ∧ β = 2) : 
    (β = -6) → (α = 1/3) := by
  sorry

end alpha_value_at_negative_six_l1043_104311


namespace union_of_A_and_B_l1043_104380

def set_A : Set ℝ := {x | x - 2 > 0}
def set_B : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem union_of_A_and_B :
  set_A ∪ set_B = Set.Ici (1 : ℝ) := by sorry

end union_of_A_and_B_l1043_104380


namespace remainder_problem_l1043_104355

theorem remainder_problem (m : ℤ) (k : ℤ) (h : m = 100 * k - 2) : 
  (m^2 + 4*m + 6) % 100 = 2 := by
sorry

end remainder_problem_l1043_104355


namespace minimum_amount_spent_on_boxes_l1043_104369

/-- The minimum amount spent on boxes for packaging a collection --/
theorem minimum_amount_spent_on_boxes
  (box_length : ℝ) (box_width : ℝ) (box_height : ℝ)
  (cost_per_box : ℝ) (total_collection_volume : ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 12)
  (h4 : cost_per_box = 0.40)
  (h5 : total_collection_volume = 2160000) :
  ⌈total_collection_volume / (box_length * box_width * box_height)⌉ * cost_per_box = 180 := by
  sorry

#check minimum_amount_spent_on_boxes

end minimum_amount_spent_on_boxes_l1043_104369


namespace find_number_l1043_104387

theorem find_number (x : ℚ) : (x + 113 / 78) * 78 = 4403 → x = 55 := by
  sorry

end find_number_l1043_104387


namespace negation_of_proposition_l1043_104317

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 + 3*x + 1 < 0) ↔ (∀ x : ℝ, x > 0 → x^2 + 3*x + 1 ≥ 0) := by
  sorry

end negation_of_proposition_l1043_104317


namespace fraction_of_employees_laid_off_l1043_104353

/-- Proves that the fraction of employees laid off is 1/3 given the initial conditions -/
theorem fraction_of_employees_laid_off 
  (initial_employees : ℕ) 
  (salary_per_employee : ℕ) 
  (total_paid_after_layoff : ℕ) 
  (h1 : initial_employees = 450)
  (h2 : salary_per_employee = 2000)
  (h3 : total_paid_after_layoff = 600000) :
  (initial_employees * salary_per_employee - total_paid_after_layoff) / (initial_employees * salary_per_employee) = 1 / 3 :=
by
  sorry

end fraction_of_employees_laid_off_l1043_104353


namespace cosine_sine_ratio_equals_sqrt_three_l1043_104395

theorem cosine_sine_ratio_equals_sqrt_three : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end cosine_sine_ratio_equals_sqrt_three_l1043_104395


namespace marbles_distribution_l1043_104325

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) : 
  total_marbles = 35 → num_boys = 5 → marbles_per_boy = total_marbles / num_boys → marbles_per_boy = 7 := by
  sorry

end marbles_distribution_l1043_104325


namespace min_distance_inverse_curves_l1043_104396

/-- The minimum distance between points on two inverse curves -/
theorem min_distance_inverse_curves :
  let f (x : ℝ) := (1/2) * Real.exp x
  let g (x : ℝ) := Real.log (2 * x)
  ∀ (x y : ℝ), x > 0 → y > 0 →
  let P := (x, f x)
  let Q := (y, g y)
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 * (1 - Real.log 2) ∧
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 →
    let P' := (x', f x')
    let Q' := (y', g y')
    Real.sqrt ((x' - y')^2 + (f x' - g y')^2) ≥ min_dist :=
by sorry

end min_distance_inverse_curves_l1043_104396


namespace half_plus_five_equals_eleven_l1043_104388

theorem half_plus_five_equals_eleven (n : ℝ) : (1/2) * n + 5 = 11 → n = 12 := by
  sorry

end half_plus_five_equals_eleven_l1043_104388


namespace alpha_range_l1043_104339

theorem alpha_range (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.cos α - Real.sin α = Real.tan α) : 
  α ∈ Set.Ioo 0 (Real.pi / 6) := by
sorry

end alpha_range_l1043_104339


namespace complex_fraction_simplification_l1043_104300

theorem complex_fraction_simplification :
  (7 + 18 * Complex.I) / (3 - 4 * Complex.I) = (-51 / 25 : ℝ) + (82 / 25 : ℝ) * Complex.I :=
by sorry

end complex_fraction_simplification_l1043_104300


namespace volume_is_one_sixth_l1043_104330

-- Define the region
def region (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1 ∧ abs x + abs y + abs (z - 1) ≤ 1

-- Define the volume of the region
noncomputable def volume_of_region : ℝ := sorry

-- Theorem statement
theorem volume_is_one_sixth : volume_of_region = 1/6 := by sorry

end volume_is_one_sixth_l1043_104330


namespace average_weight_of_children_l1043_104338

/-- The average weight of all children given the weights of boys, girls, and toddlers -/
theorem average_weight_of_children 
  (num_boys : ℕ) (num_girls : ℕ) (num_toddlers : ℕ)
  (avg_weight_boys : ℝ) (avg_weight_girls : ℝ) (avg_weight_toddlers : ℝ)
  (h_num_boys : num_boys = 8)
  (h_num_girls : num_girls = 5)
  (h_num_toddlers : num_toddlers = 3)
  (h_avg_weight_boys : avg_weight_boys = 160)
  (h_avg_weight_girls : avg_weight_girls = 130)
  (h_avg_weight_toddlers : avg_weight_toddlers = 40)
  (h_total_children : num_boys + num_girls + num_toddlers = 16) :
  let total_weight := num_boys * avg_weight_boys + num_girls * avg_weight_girls + num_toddlers * avg_weight_toddlers
  total_weight / (num_boys + num_girls + num_toddlers) = 128.125 := by
  sorry

end average_weight_of_children_l1043_104338


namespace rectangular_map_area_l1043_104319

/-- The area of a rectangular map with given length and width. -/
def map_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular map with length 5 meters and width 2 meters is 10 square meters. -/
theorem rectangular_map_area :
  map_area 5 2 = 10 := by
  sorry

end rectangular_map_area_l1043_104319


namespace shopping_tax_calculation_l1043_104342

theorem shopping_tax_calculation (total_amount : ℝ) (h_positive : total_amount > 0) :
  let clothing_percent : ℝ := 0.5
  let food_percent : ℝ := 0.25
  let other_percent : ℝ := 0.25
  let clothing_tax_rate : ℝ := 0.1
  let food_tax_rate : ℝ := 0
  let other_tax_rate : ℝ := 0.2
  
  let clothing_amount := clothing_percent * total_amount
  let food_amount := food_percent * total_amount
  let other_amount := other_percent * total_amount
  
  let clothing_tax := clothing_amount * clothing_tax_rate
  let food_tax := food_amount * food_tax_rate
  let other_tax := other_amount * other_tax_rate
  
  let total_tax := clothing_tax + food_tax + other_tax
  
  (total_tax / total_amount) = 0.1 := by sorry

end shopping_tax_calculation_l1043_104342


namespace inverted_sand_height_is_25_l1043_104351

/-- Represents the container with frustum and cylinder components -/
structure Container where
  radius : ℝ
  frustumHeight : ℝ
  cylinderHeight : ℝ
  cylinderFillHeight : ℝ

/-- Calculates the total height of sand when the container is inverted -/
def invertedSandHeight (c : Container) : ℝ :=
  c.frustumHeight + c.cylinderFillHeight

/-- Theorem stating the height of sand when the container is inverted -/
theorem inverted_sand_height_is_25 (c : Container) 
  (h_radius : c.radius = 12)
  (h_frustum_height : c.frustumHeight = 20)
  (h_cylinder_height : c.cylinderHeight = 20)
  (h_cylinder_fill : c.cylinderFillHeight = 5) :
  invertedSandHeight c = 25 := by
  sorry

#check inverted_sand_height_is_25

end inverted_sand_height_is_25_l1043_104351


namespace binomial_10_3_l1043_104394

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by sorry

end binomial_10_3_l1043_104394


namespace min_value_theorem_min_value_achievable_l1043_104315

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 3) ≥ 2 * Real.sqrt 6 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 3) = 2 * Real.sqrt 6 := by
  sorry

end min_value_theorem_min_value_achievable_l1043_104315


namespace divisibility_rule_2701_l1043_104397

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens * tens + ones * ones

theorem divisibility_rule_2701 :
  ∀ x : ℕ, is_two_digit x →
    (2701 % x = 0 ↔ sum_of_squares_of_digits x = 58) := by sorry

end divisibility_rule_2701_l1043_104397


namespace ellipse_focus_d_l1043_104335

/-- An ellipse in the first quadrant tangent to both x-axis and y-axis with foci at (4,8) and (d,8) -/
structure Ellipse where
  d : ℝ
  tangent_to_axes : Bool
  in_first_quadrant : Bool
  focus1 : ℝ × ℝ := (4, 8)
  focus2 : ℝ × ℝ := (d, 8)

/-- The value of d for the given ellipse is 15 -/
theorem ellipse_focus_d (e : Ellipse) (h1 : e.tangent_to_axes) (h2 : e.in_first_quadrant) :
  e.d = 15 := by
  sorry

end ellipse_focus_d_l1043_104335


namespace decagon_adjacent_probability_l1043_104391

/-- A decagon is a polygon with 10 sides and vertices -/
def Decagon : Type := Unit

/-- Two vertices in a polygon are adjacent if they share an edge -/
def adjacent (v1 v2 : ℕ) (p : Decagon) : Prop := sorry

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes -/
def probability (event total : ℕ) : ℚ := event / total

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem decagon_adjacent_probability :
  ∀ (d : Decagon),
  probability 
    (10 : ℕ)  -- Number of adjacent vertex pairs
    (choose_two 10)  -- Total number of vertex pairs
  = 2 / 9 := by sorry

end decagon_adjacent_probability_l1043_104391


namespace sequence_sum_l1043_104365

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_sum (a b : ℕ → ℝ) :
  is_geometric a →
  is_arithmetic b →
  a 3 * a 11 = 4 * a 7 →
  a 7 = b 7 →
  b 5 + b 9 = 8 := by
sorry

end sequence_sum_l1043_104365


namespace min_faces_for_conditions_l1043_104367

/-- Represents a pair of dice --/
structure DicePair :=
  (die1 : ℕ)
  (die2 : ℕ)

/-- Calculates the number of ways to roll a sum on a pair of dice --/
def waysToRollSum (d : DicePair) (sum : ℕ) : ℕ := sorry

/-- Checks if a pair of dice satisfies the given conditions --/
def satisfiesConditions (d : DicePair) : Prop :=
  d.die1 ≥ 6 ∧ d.die2 ≥ 6 ∧
  waysToRollSum d 8 * 12 = waysToRollSum d 11 * 5 ∧
  waysToRollSum d 14 * d.die1 * d.die2 = d.die1 * d.die2 / 15

/-- Theorem stating that the minimum number of faces on two dice satisfying the conditions is 27 --/
theorem min_faces_for_conditions :
  ∀ d : DicePair, satisfiesConditions d → d.die1 + d.die2 ≥ 27 :=
sorry

end min_faces_for_conditions_l1043_104367


namespace f_properties_l1043_104386

def f (x : ℝ) : ℝ := x^3 - 3*x^2

theorem f_properties : 
  (∀ x y, x < y ∧ ((x ≤ 0 ∧ y ≤ 0) ∨ (x ≥ 2 ∧ y ≥ 2)) → f x < f y) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x > f y) ∧
  (∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → f x < f 0) ∧
  (∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f x > f 2) :=
by sorry

end f_properties_l1043_104386


namespace suji_age_problem_l1043_104321

theorem suji_age_problem (abi_age suji_age : ℕ) : 
  (abi_age : ℚ) / suji_age = 5 / 4 →
  (abi_age + 3 : ℚ) / (suji_age + 3) = 11 / 9 →
  suji_age = 24 := by
sorry

end suji_age_problem_l1043_104321


namespace initial_population_initial_population_approx_l1043_104331

/-- Calculates the initial population of a village given the population changes over 5 years and the final population. -/
theorem initial_population (final_population : ℝ) : ℝ :=
  let year1_change := 1.05
  let year2_change := 0.93
  let year3_change := 1.03
  let year4_change := 1.10
  let year5_change := 0.95
  final_population / (year1_change * year2_change * year3_change * year4_change * year5_change)

/-- The initial population of the village is approximately 10,457. -/
theorem initial_population_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |initial_population 10450 - 10457| < ε :=
sorry

end initial_population_initial_population_approx_l1043_104331


namespace scientific_notation_equality_l1043_104304

theorem scientific_notation_equality (n : ℝ) : n = 361000000 → n = 3.61 * 10^8 := by
  sorry

end scientific_notation_equality_l1043_104304


namespace inequality_transformation_l1043_104340

theorem inequality_transformation (a b c : ℝ) (h1 : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by sorry

end inequality_transformation_l1043_104340


namespace min_knight_liar_pairs_l1043_104361

/-- Represents the type of people on the island -/
inductive Person
| Knight
| Liar

/-- Represents a friendship between two people -/
structure Friendship where
  person1 : Person
  person2 : Person

/-- The total number of people on the island -/
def total_people : Nat := 200

/-- The number of knights on the island -/
def num_knights : Nat := 100

/-- The number of liars on the island -/
def num_liars : Nat := 100

/-- The number of people who said "All my friends are knights" -/
def num_all_knight_claims : Nat := 100

/-- The number of people who said "All my friends are liars" -/
def num_all_liar_claims : Nat := 100

/-- Definition: Each person has at least one friend -/
axiom has_friend (p : Person) : ∃ (f : Friendship), f.person1 = p ∨ f.person2 = p

/-- Definition: Knights always tell the truth -/
axiom knight_truth (k : Person) (claim : Prop) : k = Person.Knight → (claim ↔ true)

/-- Definition: Liars always lie -/
axiom liar_lie (l : Person) (claim : Prop) : l = Person.Liar → (claim ↔ false)

/-- The main theorem to be proved -/
theorem min_knight_liar_pairs :
  ∃ (friendships : List Friendship),
    (∀ f ∈ friendships, (f.person1 = Person.Knight ∧ f.person2 = Person.Liar) ∨
                        (f.person1 = Person.Liar ∧ f.person2 = Person.Knight)) ∧
    friendships.length = 50 ∧
    (∀ friendships' : List Friendship,
      (∀ f' ∈ friendships', (f'.person1 = Person.Knight ∧ f'.person2 = Person.Liar) ∨
                            (f'.person1 = Person.Liar ∧ f'.person2 = Person.Knight)) →
      friendships'.length ≥ 50) := by
  sorry

end min_knight_liar_pairs_l1043_104361


namespace pens_problem_l1043_104364

theorem pens_problem (initial_pens : ℕ) (final_pens : ℕ) (sharon_pens : ℕ) 
  (h1 : initial_pens = 20)
  (h2 : final_pens = 65)
  (h3 : sharon_pens = 19) :
  ∃ (mike_pens : ℕ), 2 * (initial_pens + mike_pens) - sharon_pens = final_pens ∧ mike_pens = 22 := by
  sorry

end pens_problem_l1043_104364


namespace problem_solution_l1043_104305

theorem problem_solution (x y : ℝ) 
  (hx : x = 2 + Real.sqrt 3) 
  (hy : y = 2 - Real.sqrt 3) : 
  (x^2 + 2*x*y + y^2 = 16) ∧ (x^2 - y^2 = 8 * Real.sqrt 3) := by
  sorry

end problem_solution_l1043_104305


namespace measure_nine_kg_from_twentyfour_l1043_104374

/-- Represents a pile of nails with a given weight in kg. -/
structure NailPile :=
  (weight : ℚ)

/-- Represents the state of our nails, divided into at most four piles. -/
structure NailState :=
  (pile1 : NailPile)
  (pile2 : Option NailPile)
  (pile3 : Option NailPile)
  (pile4 : Option NailPile)

/-- Divides a pile into two equal piles. -/
def dividePile (p : NailPile) : NailPile × NailPile :=
  (⟨p.weight / 2⟩, ⟨p.weight / 2⟩)

/-- Combines two piles into one. -/
def combinePiles (p1 p2 : NailPile) : NailPile :=
  ⟨p1.weight + p2.weight⟩

/-- The theorem stating that we can measure out 9 kg from 24 kg using only division. -/
theorem measure_nine_kg_from_twentyfour :
  ∃ (final : NailState),
    (final.pile1.weight = 9 ∨ 
     (∃ p, final.pile2 = some p ∧ p.weight = 9) ∨
     (∃ p, final.pile3 = some p ∧ p.weight = 9) ∨
     (∃ p, final.pile4 = some p ∧ p.weight = 9)) ∧
    final.pile1.weight + 
    (final.pile2.map (λ p => p.weight) |>.getD 0) +
    (final.pile3.map (λ p => p.weight) |>.getD 0) +
    (final.pile4.map (λ p => p.weight) |>.getD 0) = 24 :=
sorry

end measure_nine_kg_from_twentyfour_l1043_104374


namespace remainder_problem_l1043_104312

theorem remainder_problem (M : ℕ) (h1 : M % 24 = 13) (h2 : M = 3024) : M % 1821 = 1203 := by
  sorry

end remainder_problem_l1043_104312


namespace x_y_equation_l1043_104393

theorem x_y_equation (x y : ℚ) (hx : x = 2/3) (hy : y = 9/2) : (1/3) * x^4 * y^5 = 121.5 := by
  sorry

end x_y_equation_l1043_104393
