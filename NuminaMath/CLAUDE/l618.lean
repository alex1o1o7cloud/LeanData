import Mathlib

namespace NUMINAMATH_CALUDE_complement_of_A_l618_61851

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≤ -3} ∪ {x | x ≥ 0}

-- Theorem statement
theorem complement_of_A : Set.compl A = Set.Ioo (-3) 0 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l618_61851


namespace NUMINAMATH_CALUDE_exactly_one_solves_l618_61843

/-- The probability that exactly one person solves a problem given two independent probabilities -/
theorem exactly_one_solves (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) 
  (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  p₁ * (1 - p₂) + p₂ * (1 - p₁) = p₁ + p₂ - 2 * p₁ * p₂ := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_solves_l618_61843


namespace NUMINAMATH_CALUDE_right_triangle_with_tangent_circle_l618_61860

theorem right_triangle_with_tangent_circle (a b c r : ℕ) : 
  a^2 + b^2 = c^2 → -- right triangle
  Nat.gcd a (Nat.gcd b c) = 1 → -- side lengths have no common divisor greater than 1
  r = (a + b - c) / 2 → -- radius of circle tangent to hypotenuse
  r = 420 → -- given radius
  (a = 399 ∧ b = 40 ∧ c = 401) ∨ (a = 40 ∧ b = 399 ∧ c = 401) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_with_tangent_circle_l618_61860


namespace NUMINAMATH_CALUDE_unique_solution_l618_61876

-- Define the digits as natural numbers
def A : ℕ := sorry
def B : ℕ := sorry
def d : ℕ := sorry
def I : ℕ := sorry

-- Define the conditions
axiom digit_constraint : A < 10 ∧ B < 10 ∧ d < 10 ∧ I < 10
axiom equation : 58 * (100 * A + 10 * B + A) = 1000 * I + 100 * d + 10 * B + A

-- State the theorem
theorem unique_solution : d = 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l618_61876


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l618_61883

/-- A line in 2D space defined by the equation x - y - 1 = 0 -/
def line (x y : ℝ) : Prop := x - y - 1 = 0

/-- The second quadrant of a 2D coordinate system -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem stating that the line x - y - 1 = 0 does not pass through the second quadrant -/
theorem line_not_in_second_quadrant :
  ∀ x y : ℝ, line x y → ¬(second_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l618_61883


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l618_61861

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / x + 9 / y = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / a + 9 / b = 2 → x + y ≤ a + b ∧ 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 9 / y = 2 ∧ x + y = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l618_61861


namespace NUMINAMATH_CALUDE_trajectory_equation_l618_61828

-- Define the property for a point (x, y)
def satisfiesProperty (x y : ℝ) : Prop :=
  2 * (|x| + |y|) = x^2 + y^2

-- Theorem statement
theorem trajectory_equation :
  ∀ x y : ℝ, satisfiesProperty x y ↔ x^2 + y^2 = 2 * |x| + 2 * |y| :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l618_61828


namespace NUMINAMATH_CALUDE_P_and_S_not_fourth_l618_61897

-- Define the set of runners
inductive Runner : Type
| P | Q | R | S | T | U

-- Define the relation "finishes before"
def finishes_before (a b : Runner) : Prop := sorry

-- Define the conditions
axiom P_beats_R : finishes_before Runner.P Runner.R
axiom P_beats_S : finishes_before Runner.P Runner.S
axiom Q_beats_S : finishes_before Runner.Q Runner.S
axiom Q_before_U : finishes_before Runner.Q Runner.U
axiom U_before_P : finishes_before Runner.U Runner.P
axiom T_before_U : finishes_before Runner.T Runner.U
axiom T_before_Q : finishes_before Runner.T Runner.Q

-- Define what it means to finish fourth
def finishes_fourth (r : Runner) : Prop := 
  ∃ a b c : Runner, 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a ≠ r ∧ b ≠ r ∧ c ≠ r ∧
    finishes_before a r ∧ 
    finishes_before b r ∧ 
    finishes_before c r ∧
    (∀ x : Runner, x ≠ r → x ≠ a → x ≠ b → x ≠ c → finishes_before r x)

-- Theorem to prove
theorem P_and_S_not_fourth : 
  ¬(finishes_fourth Runner.P) ∧ ¬(finishes_fourth Runner.S) :=
sorry

end NUMINAMATH_CALUDE_P_and_S_not_fourth_l618_61897


namespace NUMINAMATH_CALUDE_average_income_P_R_l618_61823

def average_income (x y : ℕ) : ℚ := (x + y) / 2

theorem average_income_P_R (P Q R : ℕ) : 
  average_income P Q = 5050 →
  average_income Q R = 6250 →
  P = 4000 →
  average_income P R = 5200 := by
sorry

end NUMINAMATH_CALUDE_average_income_P_R_l618_61823


namespace NUMINAMATH_CALUDE_parabola_midpoint_trajectory_and_intersection_l618_61815

/-- Parabola C -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Trajectory D -/
def trajectory_D (x y : ℝ) : Prop := y^2 = x

/-- Line l with slope 1 passing through (1, 0) -/
def line_l (x y : ℝ) : Prop := y = x - 1

/-- The focus of parabola C -/
def focus_C : ℝ × ℝ := (1, 0)

/-- The statement to prove -/
theorem parabola_midpoint_trajectory_and_intersection :
  (∀ x y : ℝ, parabola_C x y → ∃ x' y' : ℝ, trajectory_D x' y' ∧ y' = y / 2 ∧ x' = x) ∧
  (∃ A B : ℝ × ℝ,
    trajectory_D A.1 A.2 ∧ trajectory_D B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 10) :=
sorry

end NUMINAMATH_CALUDE_parabola_midpoint_trajectory_and_intersection_l618_61815


namespace NUMINAMATH_CALUDE_triangle_special_angle_l618_61882

open Real

/-- In a triangle ABC, given that 2b cos A = 2c - √3a, prove that angle B is π/6 --/
theorem triangle_special_angle (a b c : ℝ) (A B C : ℝ) (h : 2 * b * cos A = 2 * c - Real.sqrt 3 * a) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π →
  B = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l618_61882


namespace NUMINAMATH_CALUDE_mrs_heine_treats_l618_61839

/-- The number of treats Mrs. Heine needs to buy for her pets -/
def total_treats (num_dogs : ℕ) (num_cats : ℕ) (num_parrots : ℕ) 
                 (biscuits_per_dog : ℕ) (treats_per_cat : ℕ) (sticks_per_parrot : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog + num_cats * treats_per_cat + num_parrots * sticks_per_parrot

/-- Theorem stating that Mrs. Heine needs to buy 11 treats in total -/
theorem mrs_heine_treats : total_treats 2 1 3 3 2 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mrs_heine_treats_l618_61839


namespace NUMINAMATH_CALUDE_impossible_to_tile_rectangle_with_all_tetrominoes_l618_61867

/-- Represents the different types of tetrominoes -/
inductive Tetromino
  | I
  | Square
  | Z
  | T
  | L

/-- Represents a color in a checkerboard pattern -/
inductive Color
  | Black
  | White

/-- Represents the coverage of squares by a tetromino on a checkerboard -/
structure TetrominoCoverage where
  black : Nat
  white : Nat

/-- The number of squares covered by each tetromino -/
def tetromino_size : Nat := 4

/-- The coverage of squares by each type of tetromino on a checkerboard -/
def tetromino_coverage (t : Tetromino) : TetrominoCoverage :=
  match t with
  | Tetromino.I => ⟨2, 2⟩
  | Tetromino.Square => ⟨2, 2⟩
  | Tetromino.Z => ⟨2, 2⟩
  | Tetromino.L => ⟨2, 2⟩
  | Tetromino.T => ⟨3, 1⟩  -- or ⟨1, 3⟩, doesn't matter for the proof

/-- Theorem stating that it's impossible to tile a rectangle with one of each tetromino type -/
theorem impossible_to_tile_rectangle_with_all_tetrominoes :
  ¬ ∃ (w h : Nat), w * h = 5 * tetromino_size ∧
    (∃ (c : Color), 
      (List.sum (List.map (λ t => (tetromino_coverage t).black) [Tetromino.I, Tetromino.Square, Tetromino.Z, Tetromino.T, Tetromino.L]) = w * h / 2) ∨
      (List.sum (List.map (λ t => (tetromino_coverage t).white) [Tetromino.I, Tetromino.Square, Tetromino.Z, Tetromino.T, Tetromino.L]) = w * h / 2)) :=
by sorry


end NUMINAMATH_CALUDE_impossible_to_tile_rectangle_with_all_tetrominoes_l618_61867


namespace NUMINAMATH_CALUDE_probability_one_defective_part_l618_61812

/-- The probability of drawing exactly one defective part from a box containing 5 parts,
    of which 2 are defective, when randomly selecting 2 parts. -/
theorem probability_one_defective_part : 
  let total_parts : ℕ := 5
  let defective_parts : ℕ := 2
  let drawn_parts : ℕ := 2
  let total_ways := Nat.choose total_parts drawn_parts
  let favorable_ways := Nat.choose defective_parts 1 * Nat.choose (total_parts - defective_parts) (drawn_parts - 1)
  (favorable_ways : ℚ) / total_ways = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_defective_part_l618_61812


namespace NUMINAMATH_CALUDE_prob_at_least_one_black_is_four_fifths_l618_61838

/-- The number of balls in the bag -/
def total_balls : ℕ := 6

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one black ball when two balls are randomly drawn -/
def prob_at_least_one_black : ℚ := 4/5

theorem prob_at_least_one_black_is_four_fifths :
  prob_at_least_one_black = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_black_is_four_fifths_l618_61838


namespace NUMINAMATH_CALUDE_definite_integral_x_squared_l618_61871

theorem definite_integral_x_squared : ∫ x in (-1)..(1), x^2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_x_squared_l618_61871


namespace NUMINAMATH_CALUDE_cosine_like_properties_l618_61868

-- Define the cosine-like function
def cosine_like (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

-- State the theorem
theorem cosine_like_properties (f : ℝ → ℝ) 
  (h1 : cosine_like f) 
  (h2 : f 1 = 5/4)
  (h3 : ∀ t : ℝ, t ≠ 0 → f t > 1) :
  (f 0 = 1 ∧ f 2 = 17/8) ∧ 
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℚ, |x₁| < |x₂| → f x₁ < f x₂) := by
  sorry


end NUMINAMATH_CALUDE_cosine_like_properties_l618_61868


namespace NUMINAMATH_CALUDE_odd_function_implies_m_eq_neg_one_l618_61847

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The logarithm function with base a -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The given function f -/
noncomputable def f (a m : ℝ) (x : ℝ) : ℝ := log a ((1 - m*x) / (x - 1))

theorem odd_function_implies_m_eq_neg_one (a m : ℝ) 
    (h1 : a > 0) (h2 : a ≠ 1) (h3 : IsOddFunction (f a m)) : m = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_m_eq_neg_one_l618_61847


namespace NUMINAMATH_CALUDE_congruence_problem_l618_61832

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 27 [ZMOD 60]) (h2 : b ≡ 94 [ZMOD 60]) :
  ∃ n : ℤ, 150 ≤ n ∧ n ≤ 211 ∧ (a - b) ≡ n [ZMOD 60] ∧ n = 173 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l618_61832


namespace NUMINAMATH_CALUDE_bobby_shoe_count_bobby_shoe_count_proof_l618_61803

/-- Given the relationships between Bonny's, Becky's, and Bobby's shoe counts, 
    prove that Bobby has 27 pairs of shoes. -/
theorem bobby_shoe_count : ℕ → ℕ → Prop :=
  fun becky_shoes bobby_shoes =>
    -- Bonny has 13 pairs of shoes
    -- Bonny's shoe count is 5 less than twice Becky's
    13 = 2 * becky_shoes - 5 →
    -- Bobby has 3 times as many shoes as Becky
    bobby_shoes = 3 * becky_shoes →
    -- Prove that Bobby has 27 pairs of shoes
    bobby_shoes = 27

/-- Proof of the theorem -/
theorem bobby_shoe_count_proof : ∃ (becky_shoes : ℕ), bobby_shoe_count becky_shoes 27 := by
  sorry

end NUMINAMATH_CALUDE_bobby_shoe_count_bobby_shoe_count_proof_l618_61803


namespace NUMINAMATH_CALUDE_dodecahedron_triangles_l618_61800

/-- The number of vertices in a dodecahedron -/
def dodecahedron_vertices : ℕ := 20

/-- The number of distinct triangles that can be constructed by connecting three different vertices of a dodecahedron -/
def distinct_triangles (n : ℕ) : ℕ := n.choose 3

theorem dodecahedron_triangles :
  distinct_triangles dodecahedron_vertices = 1140 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_triangles_l618_61800


namespace NUMINAMATH_CALUDE_decreasing_interval_l618_61855

def f (x : ℝ) := x^2 - 6*x + 8

theorem decreasing_interval (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, ∀ y ∈ Set.Icc 1 a, x < y → f x > f y) ↔ 1 < a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_l618_61855


namespace NUMINAMATH_CALUDE_circumscribed_sphere_volume_regular_tetrahedron_l618_61802

/-- The volume of the circumscribed sphere of a regular tetrahedron
    is 27 times the volume of its inscribed sphere. -/
theorem circumscribed_sphere_volume_regular_tetrahedron
  (r : ℝ) -- radius of the inscribed sphere
  (R : ℝ) -- radius of the circumscribed sphere
  (h_positive : r > 0)
  (h_ratio : R = 3 * r)
  (V_inscribed : ℝ) -- volume of the inscribed sphere
  (h_volume_inscribed : V_inscribed = (4 / 3) * Real.pi * r^3)
  (V_circumscribed : ℝ) -- volume of the circumscribed sphere
  (h_volume_circumscribed : V_circumscribed = (4 / 3) * Real.pi * R^3) :
  V_circumscribed = 27 * V_inscribed :=
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_volume_regular_tetrahedron_l618_61802


namespace NUMINAMATH_CALUDE_li_ying_final_score_l618_61886

/-- Calculate the final score in a quiz where correct answers earn points and incorrect answers deduct points. -/
def calculate_final_score (correct_points : ℤ) (incorrect_points : ℤ) (num_correct : ℕ) (num_incorrect : ℕ) : ℤ :=
  correct_points * num_correct - incorrect_points * num_incorrect

/-- Theorem stating that Li Ying's final score in the safety knowledge quiz is 45 points. -/
theorem li_ying_final_score :
  let correct_points : ℤ := 5
  let incorrect_points : ℤ := 3
  let num_correct : ℕ := 12
  let num_incorrect : ℕ := 5
  calculate_final_score correct_points incorrect_points num_correct num_incorrect = 45 := by
  sorry

#eval calculate_final_score 5 3 12 5

end NUMINAMATH_CALUDE_li_ying_final_score_l618_61886


namespace NUMINAMATH_CALUDE_problem_solution_l618_61856

theorem problem_solution : (-1)^2023 + |2 * Real.sqrt 2 - 3| + (8 : ℝ)^(1/3) = 4 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l618_61856


namespace NUMINAMATH_CALUDE_night_shift_guards_l618_61845

/-- Represents the number of guards hired for a night shift -/
def num_guards (total_hours middle_guard_hours first_guard_hours last_guard_hours : ℕ) : ℕ :=
  let middle_guards := (total_hours - first_guard_hours - last_guard_hours) / middle_guard_hours
  1 + middle_guards + 1

/-- Theorem stating the number of guards hired for the night shift -/
theorem night_shift_guards : 
  num_guards 9 2 3 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_night_shift_guards_l618_61845


namespace NUMINAMATH_CALUDE_alternating_power_difference_l618_61826

theorem alternating_power_difference : (-1 : ℤ)^2010 - (-1 : ℤ)^2011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alternating_power_difference_l618_61826


namespace NUMINAMATH_CALUDE_painting_price_increase_l618_61863

theorem painting_price_increase (P : ℝ) (X : ℝ) : 
  (P * (1 + X / 100) * (1 - 0.25) = P * 0.9) → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_increase_l618_61863


namespace NUMINAMATH_CALUDE_negation_to_original_proposition_l618_61825

theorem negation_to_original_proposition :
  (¬ (∃ x : ℝ, x < 1 ∧ x^2 < 1)) ↔ (∀ x : ℝ, x < 1 → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_to_original_proposition_l618_61825


namespace NUMINAMATH_CALUDE_function_values_l618_61850

noncomputable section

def f (x : ℝ) : ℝ := -1/x

theorem function_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (f a = -1/3 → a = 3) ∧
  (f (a * b) = 1/6 → b = -2) ∧
  (f c = Real.sin c / Real.cos c → Real.tan c = -1/c) :=
by sorry

end NUMINAMATH_CALUDE_function_values_l618_61850


namespace NUMINAMATH_CALUDE_unique_card_arrangement_l618_61877

def CardPair := (Nat × Nat)

def is_valid_pair (p : CardPair) : Prop :=
  (p.1 ∣ p.2) ∨ (p.2 ∣ p.1)

def is_unique_arrangement (arr : List CardPair) : Prop :=
  arr.length = 5 ∧
  (∀ p ∈ arr, 1 ≤ p.1 ∧ p.1 ≤ 10 ∧ 1 ≤ p.2 ∧ p.2 ≤ 10) ∧
  (∀ p ∈ arr, is_valid_pair p) ∧
  (∀ n : Nat, 1 ≤ n ∧ n ≤ 10 → (arr.map Prod.fst ++ arr.map Prod.snd).count n = 1)

theorem unique_card_arrangement :
  ∃! arr : List CardPair, is_unique_arrangement arr :=
sorry

end NUMINAMATH_CALUDE_unique_card_arrangement_l618_61877


namespace NUMINAMATH_CALUDE_survey_result_l618_61846

theorem survey_result (total : ℕ) (tv_dislike_percent : ℚ) (both_dislike_percent : ℚ) 
  (h1 : total = 1500)
  (h2 : tv_dislike_percent = 25 / 100)
  (h3 : both_dislike_percent = 20 / 100) :
  ↑⌊both_dislike_percent * (tv_dislike_percent * total)⌋ = 75 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l618_61846


namespace NUMINAMATH_CALUDE_hot_dog_buns_per_student_l618_61880

theorem hot_dog_buns_per_student (
  buns_per_package : ℕ)
  (packages_bought : ℕ)
  (num_classes : ℕ)
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages_bought = 30)
  (h3 : num_classes = 4)
  (h4 : students_per_class = 30)
  : (buns_per_package * packages_bought) / (num_classes * students_per_class) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_buns_per_student_l618_61880


namespace NUMINAMATH_CALUDE_distribute_five_into_three_l618_61836

/-- The number of ways to distribute n distinct objects into k distinct non-empty groups --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 150 ways to distribute 5 distinct objects into 3 distinct non-empty groups --/
theorem distribute_five_into_three : distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_into_three_l618_61836


namespace NUMINAMATH_CALUDE_ladder_problem_l618_61811

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base_distance : ℝ, base_distance^2 + height^2 = ladder_length^2 ∧ base_distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l618_61811


namespace NUMINAMATH_CALUDE_tie_in_may_l618_61817

structure Player where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ
  may : ℕ

def johnson : Player := ⟨2, 12, 20, 15, 9⟩
def martinez : Player := ⟨5, 9, 15, 20, 9⟩

def cumulative_score (p : Player) (month : ℕ) : ℕ :=
  match month with
  | 1 => p.january
  | 2 => p.january + p.february
  | 3 => p.january + p.february + p.march
  | 4 => p.january + p.february + p.march + p.april
  | 5 => p.january + p.february + p.march + p.april + p.may
  | _ => 0

def first_tie_month : ℕ :=
  [1, 2, 3, 4, 5].find? (λ m => cumulative_score johnson m = cumulative_score martinez m)
    |>.getD 0

theorem tie_in_may :
  first_tie_month = 5 := by sorry

end NUMINAMATH_CALUDE_tie_in_may_l618_61817


namespace NUMINAMATH_CALUDE_all_statements_imply_negation_l618_61818

theorem all_statements_imply_negation (p q r : Prop) : 
  -- Statement 1
  ((p ∧ q ∧ r) → (¬p ∨ ¬q ∨ r)) ∧
  -- Statement 2
  ((p ∧ ¬q ∧ r) → (¬p ∨ ¬q ∨ r)) ∧
  -- Statement 3
  ((¬p ∧ q ∧ ¬r) → (¬p ∨ ¬q ∨ r)) ∧
  -- Statement 4
  ((¬p ∧ q ∧ ¬r) → (¬p ∨ ¬q ∨ r)) := by
  sorry

#check all_statements_imply_negation

end NUMINAMATH_CALUDE_all_statements_imply_negation_l618_61818


namespace NUMINAMATH_CALUDE_min_buses_for_field_trip_l618_61859

def min_buses (total_students : ℕ) (bus_cap_1 bus_cap_2 : ℕ) (min_bus_2 : ℕ) : ℕ :=
  let x := ((total_students - bus_cap_2 * min_bus_2 + bus_cap_1 - 1) / bus_cap_1 : ℕ)
  x + min_bus_2

theorem min_buses_for_field_trip :
  min_buses 530 45 35 3 = 13 :=
sorry

end NUMINAMATH_CALUDE_min_buses_for_field_trip_l618_61859


namespace NUMINAMATH_CALUDE_direct_proportion_through_point_decreasing_l618_61837

-- Define the direct proportion function
def direct_proportion (m : ℝ) (x : ℝ) : ℝ := m * x

-- Define the theorem
theorem direct_proportion_through_point_decreasing (m : ℝ) :
  (direct_proportion m m = 4) →
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → direct_proportion m x₁ > direct_proportion m x₂) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_through_point_decreasing_l618_61837


namespace NUMINAMATH_CALUDE_remainder_theorem_l618_61862

theorem remainder_theorem (r : ℤ) : (r^13 - r^5 + 1) % (r - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l618_61862


namespace NUMINAMATH_CALUDE_extravagant_gift_bags_carl_extravagant_bags_l618_61809

/-- The number of extravagant gift bags Carl created for his open house -/
theorem extravagant_gift_bags 
  (confirmed_attendees : ℕ) 
  (potential_attendees : ℕ) 
  (average_bags_made : ℕ) 
  (additional_bags_needed : ℕ) : ℕ :=
  let total_expected_attendees := confirmed_attendees + potential_attendees
  let total_average_bags := average_bags_made + additional_bags_needed
  total_expected_attendees - total_average_bags

/-- Proof that Carl created 10 extravagant gift bags -/
theorem carl_extravagant_bags : extravagant_gift_bags 50 40 20 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_extravagant_gift_bags_carl_extravagant_bags_l618_61809


namespace NUMINAMATH_CALUDE_perfect_square_a_value_of_a_l618_61896

theorem perfect_square_a : ∃ n : ℕ, 1995^2 + 1995^2 * 1996^2 + 1996^2 = n^2 :=
by
  use 3982021
  sorry

theorem value_of_a : 1995^2 + 1995^2 * 1996^2 + 1996^2 = 3982021^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_a_value_of_a_l618_61896


namespace NUMINAMATH_CALUDE_min_max_abs_x_squared_minus_2xy_is_zero_l618_61879

open Real

theorem min_max_abs_x_squared_minus_2xy_is_zero :
  ∃ y : ℝ, ∀ z : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x^2 - 2*x*y| ≤ z) →
    (∀ y' : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ |x^2 - 2*x*y'| ≥ z) :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_x_squared_minus_2xy_is_zero_l618_61879


namespace NUMINAMATH_CALUDE_circle_origin_range_l618_61835

theorem circle_origin_range (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*m*x + 2*m*y + 2*m^2 - 4 = 0 → x^2 + y^2 < 4) → 
  -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_circle_origin_range_l618_61835


namespace NUMINAMATH_CALUDE_solve_bowling_problem_l618_61853

def bowling_problem (gretchen_score mitzi_score average_score : ℕ) : Prop :=
  let total_score := average_score * 3
  let beth_score := total_score - gretchen_score - mitzi_score
  gretchen_score = 120 ∧ 
  mitzi_score = 113 ∧ 
  average_score = 106 →
  beth_score = 85

theorem solve_bowling_problem :
  ∃ (gretchen_score mitzi_score average_score : ℕ),
    bowling_problem gretchen_score mitzi_score average_score :=
by
  sorry

end NUMINAMATH_CALUDE_solve_bowling_problem_l618_61853


namespace NUMINAMATH_CALUDE_christine_wandering_l618_61865

/-- The distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Christine's wandering problem -/
theorem christine_wandering (christine_speed : ℝ) (christine_time : ℝ) 
  (h1 : christine_speed = 4)
  (h2 : christine_time = 5) :
  distance christine_speed christine_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_christine_wandering_l618_61865


namespace NUMINAMATH_CALUDE_polar_to_rect_transformation_l618_61869

/-- Given a point (12, 5) in rectangular coordinates and (r, θ) in polar coordinates,
    prove that the point (r³, 3θ) in polar coordinates is (5600, -325) in rectangular coordinates. -/
theorem polar_to_rect_transformation (r θ : ℝ) :
  r * Real.cos θ = 12 →
  r * Real.sin θ = 5 →
  (r^3 * Real.cos (3*θ), r^3 * Real.sin (3*θ)) = (5600, -325) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rect_transformation_l618_61869


namespace NUMINAMATH_CALUDE_sum_of_series_l618_61842

theorem sum_of_series (a₁ : ℝ) (r : ℝ) (n : ℕ) (d : ℝ) :
  let geometric_sum := a₁ / (1 - r)
  let arithmetic_sum := n * (2 * a₁ + (n - 1) * d) / 2
  geometric_sum + arithmetic_sum = 115 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_l618_61842


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l618_61821

theorem sum_of_coefficients (a b x y : ℝ) : 
  (x = 3 ∧ y = -2) → 
  (a * x + b * y = 2 ∧ b * x + a * y = -3) → 
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l618_61821


namespace NUMINAMATH_CALUDE_lisa_cleaning_time_proof_l618_61874

/-- The time it takes Lisa to clean her room alone -/
def lisa_cleaning_time : ℝ := 8

/-- The time it takes Kay to clean her room alone -/
def kay_cleaning_time : ℝ := 12

/-- The time it takes Lisa and Kay to clean a room together -/
def combined_cleaning_time : ℝ := 4.8

theorem lisa_cleaning_time_proof :
  lisa_cleaning_time = 8 ∧
  (1 / lisa_cleaning_time + 1 / kay_cleaning_time = 1 / combined_cleaning_time) :=
sorry

end NUMINAMATH_CALUDE_lisa_cleaning_time_proof_l618_61874


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_12_l618_61849

theorem binomial_coefficient_19_12 
  (h1 : Nat.choose 20 13 = 77520)
  (h2 : Nat.choose 18 11 = 31824) : 
  Nat.choose 19 12 = 77520 - 31824 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_12_l618_61849


namespace NUMINAMATH_CALUDE_johns_book_expense_l618_61816

def earnings : ℕ := 10 * 26

theorem johns_book_expense (money_left : ℕ) (book_expense : ℕ) : 
  money_left = 160 → 
  earnings = money_left + 2 * book_expense → 
  book_expense = 50 :=
by sorry

end NUMINAMATH_CALUDE_johns_book_expense_l618_61816


namespace NUMINAMATH_CALUDE_surface_area_of_solid_with_square_views_l618_61857

/-- A solid with three square views -/
structure Solid where
  /-- The side length of the square views -/
  side_length : ℝ
  /-- The three views are squares -/
  square_views : Prop

/-- The surface area of a solid -/
def surface_area (s : Solid) : ℝ := sorry

/-- Theorem: The surface area of a solid with three square views of side length 2 is 24 -/
theorem surface_area_of_solid_with_square_views (s : Solid) 
  (h1 : s.side_length = 2) 
  (h2 : s.square_views) : 
  surface_area s = 24 := by sorry

end NUMINAMATH_CALUDE_surface_area_of_solid_with_square_views_l618_61857


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l618_61807

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.6 * P) 
  (hP : P ≠ 0) : M / N = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l618_61807


namespace NUMINAMATH_CALUDE_sequence_product_l618_61895

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The main theorem -/
theorem sequence_product (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  (∀ n : ℕ, a n ≠ 0) →
  (2 * a 3 - a 7 ^ 2 + 2 * a n = 0) →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l618_61895


namespace NUMINAMATH_CALUDE_rice_distribution_l618_61873

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 33 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound) / num_containers = 15 := by
  sorry

end NUMINAMATH_CALUDE_rice_distribution_l618_61873


namespace NUMINAMATH_CALUDE_polynomial_zeros_evaluation_l618_61814

theorem polynomial_zeros_evaluation (r s : ℝ) : 
  r^2 - 3*r + 1 = 0 → 
  s^2 - 3*s + 1 = 0 → 
  (1 : ℝ)^2 - 18*(1 : ℝ) + 1 = -16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_zeros_evaluation_l618_61814


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l618_61820

/-- Given Ed and Sue's rollerblading, biking, and swimming rates and their total distances,
    prove that the sum of squares of the rates is 485. -/
theorem rates_sum_of_squares (r b s : ℕ) : 
  (2 * r + 3 * b + s = 80) →
  (4 * r + 2 * b + 3 * s = 98) →
  r^2 + b^2 + s^2 = 485 := by
  sorry

end NUMINAMATH_CALUDE_rates_sum_of_squares_l618_61820


namespace NUMINAMATH_CALUDE_spencer_sessions_per_day_l618_61872

/-- Represents the jumping routine of Spencer --/
structure JumpingRoutine where
  jumps_per_minute : ℕ
  minutes_per_session : ℕ
  total_jumps : ℕ
  total_days : ℕ

/-- Calculates the number of sessions per day for Spencer's jumping routine --/
def sessions_per_day (routine : JumpingRoutine) : ℚ :=
  (routine.total_jumps / routine.total_days) / (routine.jumps_per_minute * routine.minutes_per_session)

/-- Theorem stating that Spencer's jumping routine results in 2 sessions per day --/
theorem spencer_sessions_per_day :
  let routine := JumpingRoutine.mk 4 10 400 5
  sessions_per_day routine = 2 := by
  sorry

end NUMINAMATH_CALUDE_spencer_sessions_per_day_l618_61872


namespace NUMINAMATH_CALUDE_rectangle_square_comparison_l618_61881

/-- Proves that for a rectangle with a 3:1 length-to-width ratio and 75 cm² area,
    the difference between the side of a square with equal area and the rectangle's width
    is greater than 3 cm. -/
theorem rectangle_square_comparison : ∀ (length width : ℝ),
  length / width = 3 →
  length * width = 75 →
  ∃ (square_side : ℝ),
    square_side^2 = 75 ∧
    square_side - width > 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_square_comparison_l618_61881


namespace NUMINAMATH_CALUDE_derivative_of_f_composite_l618_61864

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem derivative_of_f_composite (a b : ℝ) :
  deriv (fun x => f (a - b*x)) = fun x => -3*b*(a - b*x)^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_composite_l618_61864


namespace NUMINAMATH_CALUDE_bus_capacity_problem_l618_61829

/-- Proves that given two buses with a capacity of 150 people each, where one bus is 70% full
    and the total number of people in both buses is 195, the percentage of capacity full
    for the other bus is 60%. -/
theorem bus_capacity_problem (bus_capacity : ℕ) (total_people : ℕ) (second_bus_percentage : ℚ) :
  bus_capacity = 150 →
  total_people = 195 →
  second_bus_percentage = 70/100 →
  ∃ (first_bus_percentage : ℚ),
    first_bus_percentage * bus_capacity + second_bus_percentage * bus_capacity = total_people ∧
    first_bus_percentage = 60/100 :=
by sorry

end NUMINAMATH_CALUDE_bus_capacity_problem_l618_61829


namespace NUMINAMATH_CALUDE_equality_equivalence_l618_61841

theorem equality_equivalence (a b c d : ℝ) : 
  (a - b)^2 + (c - d)^2 = 0 ↔ (a = b ∧ c = d) := by sorry

end NUMINAMATH_CALUDE_equality_equivalence_l618_61841


namespace NUMINAMATH_CALUDE_skew_lines_planes_perpendicularity_l618_61875

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (are_skew : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem skew_lines_planes_perpendicularity 
  (m n l : Line) (α β : Plane) :
  are_skew m n →
  parallel_plane_line α m →
  parallel_plane_line α n →
  perpendicular_line_line l m →
  perpendicular_line_line l n →
  parallel_line_plane l β →
  perpendicular_plane_plane α β ∧ perpendicular_line_plane l α :=
sorry

end NUMINAMATH_CALUDE_skew_lines_planes_perpendicularity_l618_61875


namespace NUMINAMATH_CALUDE_susan_playground_area_l618_61806

/-- Represents a rectangular playground with fence posts -/
structure Playground where
  total_posts : ℕ
  post_spacing : ℕ
  longer_side_post_ratio : ℕ

/-- Calculates the area of the playground in square yards -/
def playground_area (p : Playground) : ℕ :=
  let shorter_side := p.post_spacing * (((p.total_posts / 2) / (p.longer_side_post_ratio + 1)) - 1)
  let longer_side := p.post_spacing * (p.longer_side_post_ratio * ((p.total_posts / 2) / (p.longer_side_post_ratio + 1)) - 1)
  shorter_side * longer_side

/-- Theorem stating the area of Susan's playground -/
theorem susan_playground_area :
  ∃ (p : Playground), p.total_posts = 30 ∧ p.post_spacing = 6 ∧ p.longer_side_post_ratio = 3 ∧
  playground_area p = 1188 :=
by
  sorry


end NUMINAMATH_CALUDE_susan_playground_area_l618_61806


namespace NUMINAMATH_CALUDE_donation_problem_solution_l618_61866

/-- Represents a transportation plan with type A and B trucks -/
structure TransportPlan where
  typeA : Nat
  typeB : Nat

/-- Represents the problem setup -/
structure DonationProblem where
  totalItems : Nat
  waterExcess : Nat
  typeAWaterCapacity : Nat
  typeAVegCapacity : Nat
  typeBWaterCapacity : Nat
  typeBVegCapacity : Nat
  totalTrucks : Nat
  typeACost : Nat
  typeBCost : Nat

def isValidPlan (p : DonationProblem) (plan : TransportPlan) : Prop :=
  plan.typeA + plan.typeB = p.totalTrucks ∧
  plan.typeA * p.typeAWaterCapacity + plan.typeB * p.typeBWaterCapacity ≥ (p.totalItems + p.waterExcess) / 2 ∧
  plan.typeA * p.typeAVegCapacity + plan.typeB * p.typeBVegCapacity ≥ (p.totalItems - p.waterExcess) / 2

def planCost (p : DonationProblem) (plan : TransportPlan) : Nat :=
  plan.typeA * p.typeACost + plan.typeB * p.typeBCost

theorem donation_problem_solution (p : DonationProblem)
  (h_total : p.totalItems = 320)
  (h_excess : p.waterExcess = 80)
  (h_typeA : p.typeAWaterCapacity = 40 ∧ p.typeAVegCapacity = 10)
  (h_typeB : p.typeBWaterCapacity = 20 ∧ p.typeBVegCapacity = 20)
  (h_trucks : p.totalTrucks = 8)
  (h_costs : p.typeACost = 400 ∧ p.typeBCost = 360) :
  -- 1. Number of water and vegetable pieces
  (p.totalItems + p.waterExcess) / 2 = 200 ∧ (p.totalItems - p.waterExcess) / 2 = 120 ∧
  -- 2. Valid transportation plans
  (∀ plan, isValidPlan p plan ↔ 
    (plan = ⟨2, 6⟩ ∨ plan = ⟨3, 5⟩ ∨ plan = ⟨4, 4⟩)) ∧
  -- 3. Minimum cost plan
  (∀ plan, isValidPlan p plan → planCost p ⟨2, 6⟩ ≤ planCost p plan) ∧
  planCost p ⟨2, 6⟩ = 2960 :=
sorry

end NUMINAMATH_CALUDE_donation_problem_solution_l618_61866


namespace NUMINAMATH_CALUDE_systematic_sample_property_fourth_student_number_l618_61813

/-- Represents a systematic sample from a population --/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Calculates the nth element in a systematic sample --/
def nth_element (s : SystematicSample) (n : ℕ) : ℕ :=
  ((s.start + (n - 1) * s.interval - 1) % s.population_size) + 1

/-- Theorem stating the properties of the given systematic sample --/
theorem systematic_sample_property (s : SystematicSample) : 
  s.population_size = 54 ∧ 
  s.sample_size = 4 ∧ 
  s.start = 2 ∧ 
  nth_element s 2 = 28 ∧ 
  nth_element s 3 = 41 →
  nth_element s 4 = 1 := by
  sorry

/-- Main theorem to prove --/
theorem fourth_student_number : 
  ∃ (s : SystematicSample), 
    s.population_size = 54 ∧ 
    s.sample_size = 4 ∧ 
    s.start = 2 ∧ 
    nth_element s 2 = 28 ∧ 
    nth_element s 3 = 41 ∧
    nth_element s 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_property_fourth_student_number_l618_61813


namespace NUMINAMATH_CALUDE_student_age_problem_l618_61890

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The problem statement -/
theorem student_age_problem :
  ∃! n : ℕ, 1900 ≤ n ∧ n < 1960 ∧ (1960 - n = sum_of_digits n) := by sorry

end NUMINAMATH_CALUDE_student_age_problem_l618_61890


namespace NUMINAMATH_CALUDE_max_integer_difference_l618_61884

theorem max_integer_difference (x y : ℝ) (hx : 6 < x ∧ x < 10) (hy : 10 < y ∧ y < 17) :
  (⌊y⌋ : ℤ) - (⌈x⌉ : ℤ) ≤ 9 ∧ ∃ (x₀ y₀ : ℝ), 6 < x₀ ∧ x₀ < 10 ∧ 10 < y₀ ∧ y₀ < 17 ∧ (⌊y₀⌋ : ℤ) - (⌈x₀⌉ : ℤ) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_difference_l618_61884


namespace NUMINAMATH_CALUDE_ball_game_probabilities_l618_61858

theorem ball_game_probabilities (total : ℕ) (p_white p_red p_yellow : ℚ) 
  (h_total : total = 6)
  (h_white : p_white = 1/2)
  (h_red : p_red = 1/3)
  (h_yellow : p_yellow = 1/6)
  (h_sum : p_white + p_red + p_yellow = 1) :
  ∃ (white red yellow : ℕ),
    white + red + yellow = total ∧
    (white : ℚ) / total = p_white ∧
    (red : ℚ) / total = p_red ∧
    (yellow : ℚ) / total = p_yellow ∧
    white = 3 ∧ red = 2 ∧ yellow = 1 := by
  sorry

end NUMINAMATH_CALUDE_ball_game_probabilities_l618_61858


namespace NUMINAMATH_CALUDE_parabola_directrix_l618_61804

/-- The equation of the directrix of the parabola y = x^2 is y = -1/4 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = x^2 → (∃ (k : ℝ), y = k ∧ k = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l618_61804


namespace NUMINAMATH_CALUDE_inscribed_square_area_l618_61878

/-- An ellipse with semi-major axis 2√2 and semi-minor axis 2√2 -/
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 8 = 1

/-- A square inscribed in the ellipse with sides parallel to the axes -/
def InscribedSquare (s : ℝ) : Prop :=
  ∃ (x y : ℝ), Ellipse x y ∧ s = 2 * x ∧ s = 2 * y

/-- The area of the inscribed square is 32/3 -/
theorem inscribed_square_area :
  ∃ (s : ℝ), InscribedSquare s ∧ s^2 = 32/3 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l618_61878


namespace NUMINAMATH_CALUDE_multiply_three_neg_two_l618_61894

theorem multiply_three_neg_two : 3 * (-2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_neg_two_l618_61894


namespace NUMINAMATH_CALUDE_probability_total_more_than_seven_l618_61852

/-- The number of faces on each die -/
def numFaces : Nat := 6

/-- The total number of possible outcomes when throwing two dice -/
def totalOutcomes : Nat := numFaces * numFaces

/-- The number of favorable outcomes (total > 7) -/
def favorableOutcomes : Nat := 14

/-- The probability of getting a total more than 7 -/
def probabilityTotalMoreThan7 : Rat := favorableOutcomes / totalOutcomes

theorem probability_total_more_than_seven :
  probabilityTotalMoreThan7 = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_total_more_than_seven_l618_61852


namespace NUMINAMATH_CALUDE_square_difference_evaluation_l618_61891

theorem square_difference_evaluation : 81^2 - (45 + 9)^2 = 3645 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_evaluation_l618_61891


namespace NUMINAMATH_CALUDE_plot_size_in_acres_l618_61808

/-- Represents the scale factor for converting centimeters to miles -/
def scale : ℝ := 3

/-- Represents the conversion factor from square miles to acres -/
def milesSquareToAcres : ℝ := 640

/-- Represents the length of one side of the right triangle in the scaled drawing -/
def side1 : ℝ := 20

/-- Represents the length of the other side of the right triangle in the scaled drawing -/
def side2 : ℝ := 15

/-- Theorem stating that the actual size of the plot is 864000 acres -/
theorem plot_size_in_acres :
  let scaledArea := (side1 * side2) / 2
  let actualAreaInMilesSquare := scaledArea * scale^2
  let actualAreaInAcres := actualAreaInMilesSquare * milesSquareToAcres
  actualAreaInAcres = 864000 := by sorry

end NUMINAMATH_CALUDE_plot_size_in_acres_l618_61808


namespace NUMINAMATH_CALUDE_least_number_with_divisibility_property_l618_61887

theorem least_number_with_divisibility_property : ∃ k : ℕ, 
  k > 0 ∧ 
  (k / 23 = k % 47 + 13) ∧
  (∀ m : ℕ, m > 0 → m < k → m / 23 ≠ m % 47 + 13) ∧
  k = 576 :=
sorry

end NUMINAMATH_CALUDE_least_number_with_divisibility_property_l618_61887


namespace NUMINAMATH_CALUDE_baker_duration_l618_61854

/-- Represents the number of weeks Steve bakes pies -/
def duration : ℕ := sorry

/-- Number of days per week Steve bakes apple pies -/
def apple_days : ℕ := 3

/-- Number of days per week Steve bakes cherry pies -/
def cherry_days : ℕ := 2

/-- Number of pies Steve bakes per day -/
def pies_per_day : ℕ := 12

/-- The difference in the number of apple pies and cherry pies -/
def pie_difference : ℕ := 12

theorem baker_duration :
  apple_days * pies_per_day * duration = cherry_days * pies_per_day * duration + pie_difference ∧
  duration = 1 := by sorry

end NUMINAMATH_CALUDE_baker_duration_l618_61854


namespace NUMINAMATH_CALUDE_slope_implies_y_coordinate_l618_61830

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q
    is equal to 5/3, then the y-coordinate of Q is 56/3. -/
theorem slope_implies_y_coordinate :
  ∀ (y : ℚ),
  let P : ℚ × ℚ := (-2, 7)
  let Q : ℚ × ℚ := (5, y)
  let slope : ℚ := (Q.2 - P.2) / (Q.1 - P.1)
  slope = 5/3 → y = 56/3 :=
by
  sorry

end NUMINAMATH_CALUDE_slope_implies_y_coordinate_l618_61830


namespace NUMINAMATH_CALUDE_unique_pair_divisibility_l618_61844

theorem unique_pair_divisibility : 
  ∃! (n m : ℕ), n > 2 ∧ m > 2 ∧ 
  (∃ (S : Set ℕ), Set.Infinite S ∧ 
    ∀ k ∈ S, (k^n + k^2 - 1) ∣ (k^m + k - 1)) ∧
  n = 3 ∧ m = 5 :=
sorry

end NUMINAMATH_CALUDE_unique_pair_divisibility_l618_61844


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l618_61822

def f (x : ℝ) := x^3 + x - 1

theorem root_exists_in_interval :
  Continuous f ∧ f 0 < 0 ∧ f 1 > 0 →
  ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l618_61822


namespace NUMINAMATH_CALUDE_toy_phone_price_l618_61888

theorem toy_phone_price (bert_phones : ℕ) (tory_guns : ℕ) (gun_price : ℕ) (extra_earnings : ℕ) :
  bert_phones = 8 →
  tory_guns = 7 →
  gun_price = 20 →
  extra_earnings = 4 →
  (tory_guns * gun_price + extra_earnings) / bert_phones = 18 :=
by sorry

end NUMINAMATH_CALUDE_toy_phone_price_l618_61888


namespace NUMINAMATH_CALUDE_quadratic_function_range_l618_61833

/-- A quadratic function passing through (1,0) and (0,1) with vertex in second quadrant -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  point_zero_one : 1 = c
  point_one_zero : 0 = a + b + c
  vertex_second_quadrant : b < 0 ∧ a < 0

/-- The range of a - b + c for the given quadratic function -/
theorem quadratic_function_range (f : QuadraticFunction) : 
  0 < f.a - f.b + f.c ∧ f.a - f.b + f.c < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l618_61833


namespace NUMINAMATH_CALUDE_expression_simplification_l618_61824

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  ((a^2 / (a - 2) - 1 / (a - 2)) / ((a^2 - 2*a + 1) / (a - 2))) = (3 + 2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l618_61824


namespace NUMINAMATH_CALUDE_inverse_g_87_l618_61848

def g (x : ℝ) : ℝ := 3 * x^3 + 6

theorem inverse_g_87 : g⁻¹ 87 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_87_l618_61848


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l618_61892

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + m - 1 = 0 ∧ y^2 - 4*y + m - 1 = 0) → m < 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l618_61892


namespace NUMINAMATH_CALUDE_inequality_solution_l618_61870

theorem inequality_solution (x : ℝ) :
  (6 * x^2 + 9 * x - 48) / ((3 * x + 5) * (x - 2)) < 0 ↔ 
  -4 < x ∧ x < -5/3 ∧ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l618_61870


namespace NUMINAMATH_CALUDE_count_divisible_integers_l618_61834

theorem count_divisible_integers : 
  ∃ (S : Finset Nat), 
    (∀ n ∈ S, n > 0 ∧ (8 * n) % ((n * (n + 1)) / 2) = 0) ∧ 
    (∀ n : Nat, n > 0 → (8 * n) % ((n * (n + 1)) / 2) = 0 → n ∈ S) ∧ 
    Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_integers_l618_61834


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_negative_four_range_of_a_for_inequality_l618_61893

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x + a| + |x - 2|

-- Theorem for part (1)
theorem solution_set_when_a_is_negative_four :
  {x : ℝ | f x (-4) ≥ 6} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x, f x a ≥ 3*a^2 - |2 - x|} = {a : ℝ | -1 ≤ a ∧ a ≤ 4/3} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_negative_four_range_of_a_for_inequality_l618_61893


namespace NUMINAMATH_CALUDE_rainbow_population_proof_l618_61889

/-- The number of settlements in Solar Valley -/
def num_settlements : ℕ := 10

/-- The population of Zhovtnevo -/
def zhovtnevo_population : ℕ := 1000

/-- The amount by which Zhovtnevo's population exceeds the average -/
def excess_population : ℕ := 90

/-- The population of Rainbow settlement -/
def rainbow_population : ℕ := 900

theorem rainbow_population_proof :
  rainbow_population = 
    (num_settlements * zhovtnevo_population - num_settlements * excess_population) / (num_settlements - 1) :=
by sorry

end NUMINAMATH_CALUDE_rainbow_population_proof_l618_61889


namespace NUMINAMATH_CALUDE_hyperbola_condition_l618_61898

-- Define the equation
def equation (x y k : ℝ) : Prop :=
  x^2 / (k + 1) + y^2 / (k - 5) = 1

-- Define what it means for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  (k + 1 > 0 ∧ k - 5 < 0) ∨ (k + 1 < 0 ∧ k - 5 > 0)

-- State the theorem
theorem hyperbola_condition (k : ℝ) :
  (∀ x y, equation x y k ↔ represents_hyperbola k) ↔ k ∈ Set.Ioo (-1 : ℝ) 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l618_61898


namespace NUMINAMATH_CALUDE_exists_z_satisfying_conditions_l618_61840

-- Define the complex function g
def g (z : ℂ) : ℂ := z^2 + 2*Complex.I*z + 2

-- State the theorem
theorem exists_z_satisfying_conditions : 
  ∃ z : ℂ, Complex.im z > 0 ∧ 
    (∃ a b : ℤ, g z = ↑a + ↑b * Complex.I ∧ 
      abs a ≤ 5 ∧ abs b ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_exists_z_satisfying_conditions_l618_61840


namespace NUMINAMATH_CALUDE_base5_division_l618_61819

-- Define a function to convert from base 5 to base 10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define the dividend and divisor in base 5
def dividend : List Nat := [4, 0, 2, 3, 1]  -- 13204₅
def divisor : List Nat := [3, 2]  -- 23₅

-- Define the expected quotient and remainder in base 5
def expectedQuotient : List Nat := [1, 1, 3]  -- 311₅
def expectedRemainder : Nat := 1  -- 1₅

-- Theorem statement
theorem base5_division :
  let dividend10 := base5ToBase10 dividend
  let divisor10 := base5ToBase10 divisor
  let quotient10 := dividend10 / divisor10
  let remainder10 := dividend10 % divisor10
  base5ToBase10 expectedQuotient = quotient10 ∧
  expectedRemainder = remainder10 := by
  sorry


end NUMINAMATH_CALUDE_base5_division_l618_61819


namespace NUMINAMATH_CALUDE_students_taking_one_language_count_l618_61801

/-- The number of students taking only one language class -/
def students_taking_one_language (french spanish german french_spanish french_german spanish_german all_three : ℕ) : ℕ :=
  french + spanish + german - french_spanish - french_german - spanish_german + all_three

/-- Theorem: Given the conditions, 45 students are taking only one language class -/
theorem students_taking_one_language_count : 
  students_taking_one_language 30 25 20 10 7 5 4 = 45 := by
  sorry

#eval students_taking_one_language 30 25 20 10 7 5 4

end NUMINAMATH_CALUDE_students_taking_one_language_count_l618_61801


namespace NUMINAMATH_CALUDE_total_gum_pieces_l618_61810

theorem total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) 
  (h1 : packages = 27) (h2 : pieces_per_package = 18) : 
  packages * pieces_per_package = 486 := by
  sorry

end NUMINAMATH_CALUDE_total_gum_pieces_l618_61810


namespace NUMINAMATH_CALUDE_power_factorial_inequality_l618_61805

theorem power_factorial_inequality (n : ℕ) : 2^n * n.factorial < (n + 1)^n := by
  sorry

end NUMINAMATH_CALUDE_power_factorial_inequality_l618_61805


namespace NUMINAMATH_CALUDE_three_digit_number_puzzle_l618_61885

theorem three_digit_number_puzzle :
  ∀ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 →
    a + b + c = 10 →
    b = a + c →
    100 * c + 10 * b + a = 100 * a + 10 * b + c + 99 →
    100 * a + 10 * b + c = 253 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_puzzle_l618_61885


namespace NUMINAMATH_CALUDE_right_triangle_sets_l618_61827

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  is_pythagorean_triple 7 24 25 ∧
  is_pythagorean_triple 6 8 10 ∧
  is_pythagorean_triple 9 12 15 ∧
  ¬ is_pythagorean_triple 3 4 6 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l618_61827


namespace NUMINAMATH_CALUDE_min_value_theorem_l618_61899

/-- The line equation ax + by - 2 = 0 --/
def line_equation (a b x y : ℝ) : Prop := a * x + b * y - 2 = 0

/-- The circle equation x^2 + y^2 - 2x - 2y = 2 --/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 2

/-- The line bisects the circumference of the circle --/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y, line_equation a b x y → circle_equation x y →
    ∃ c d, c^2 + d^2 = 1 ∧ line_equation a b (1 + c) (1 + d)

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_bisect : line_bisects_circle a b) :
  (1 / (2 * a) + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l618_61899


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l618_61831

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  8 * p^4 + 18 * q^4 + 50 * r^4 + 1 / (8 * p * q * r) ≥ 6 :=
by sorry

theorem min_value_achieved (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  ∃ (p₀ q₀ r₀ : ℝ), p₀ > 0 ∧ q₀ > 0 ∧ r₀ > 0 ∧
    8 * p₀^4 + 18 * q₀^4 + 50 * r₀^4 + 1 / (8 * p₀ * q₀ * r₀) = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l618_61831
