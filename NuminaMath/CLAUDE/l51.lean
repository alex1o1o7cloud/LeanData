import Mathlib

namespace NUMINAMATH_CALUDE_more_freshmen_than_sophomores_l51_5171

theorem more_freshmen_than_sophomores 
  (total : ℕ) 
  (junior_percent : ℚ) 
  (not_sophomore_percent : ℚ) 
  (seniors : ℕ) 
  (h1 : total = 800)
  (h2 : junior_percent = 22/100)
  (h3 : not_sophomore_percent = 74/100)
  (h4 : seniors = 160)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_more_freshmen_than_sophomores_l51_5171


namespace NUMINAMATH_CALUDE_alice_winning_equivalence_l51_5191

/-- The game constant k, which is greater than 2 -/
def k : ℕ := sorry

/-- Definition of Alice-winning number -/
def is_alice_winning (n : ℕ) : Prop := sorry

/-- The radical of a number n with respect to k -/
def radical (n : ℕ) : ℕ := sorry

theorem alice_winning_equivalence (l l' : ℕ) 
  (h : ∀ p : ℕ, p.Prime → p ≤ k → (p ∣ l ↔ p ∣ l')) : 
  is_alice_winning l ↔ is_alice_winning l' := by sorry

end NUMINAMATH_CALUDE_alice_winning_equivalence_l51_5191


namespace NUMINAMATH_CALUDE_four_isosceles_triangles_l51_5175

/-- A point on the grid --/
structure Point where
  x : Int
  y : Int

/-- A triangle defined by three points --/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Check if a triangle is isosceles --/
def isIsosceles (t : Triangle) : Bool :=
  let d12 := (t.p1.x - t.p2.x)^2 + (t.p1.y - t.p2.y)^2
  let d23 := (t.p2.x - t.p3.x)^2 + (t.p2.y - t.p3.y)^2
  let d31 := (t.p3.x - t.p1.x)^2 + (t.p3.y - t.p1.y)^2
  d12 = d23 || d23 = d31 || d31 = d12

/-- The list of triangles on the grid --/
def triangles : List Triangle := [
  { p1 := { x := 1, y := 6 }, p2 := { x := 3, y := 6 }, p3 := { x := 2, y := 3 } },
  { p1 := { x := 4, y := 2 }, p2 := { x := 4, y := 4 }, p3 := { x := 6, y := 2 } },
  { p1 := { x := 0, y := 0 }, p2 := { x := 3, y := 1 }, p3 := { x := 6, y := 0 } },
  { p1 := { x := 7, y := 3 }, p2 := { x := 6, y := 5 }, p3 := { x := 9, y := 3 } },
  { p1 := { x := 8, y := 0 }, p2 := { x := 9, y := 2 }, p3 := { x := 10, y := 0 } }
]

theorem four_isosceles_triangles :
  (triangles.filter isIsosceles).length = 4 := by sorry


end NUMINAMATH_CALUDE_four_isosceles_triangles_l51_5175


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l51_5162

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (4 * (x - 1) > 3 * x - 2) ∧ (2 * x - 3 ≤ 5)}
  S = {3, 4} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l51_5162


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l51_5192

theorem sqrt_product_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (3 * x) * Real.sqrt (2 * x^2) = Real.sqrt 6 * x^(3/2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l51_5192


namespace NUMINAMATH_CALUDE_fiveDigitIntegersCount_eq_ten_l51_5128

/-- The number of permutations of n elements with repetitions, where r₁, r₂, ..., rₖ
    are the repetition counts of each repeated element. -/
def permutationsWithRepetition (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.map Nat.factorial).prod

/-- The number of different five-digit integers formed using the digits 3, 3, 3, 8, and 8. -/
def fiveDigitIntegersCount : ℕ :=
  permutationsWithRepetition 5 [3, 2]

theorem fiveDigitIntegersCount_eq_ten : fiveDigitIntegersCount = 10 := by
  sorry

end NUMINAMATH_CALUDE_fiveDigitIntegersCount_eq_ten_l51_5128


namespace NUMINAMATH_CALUDE_coefficient_of_linear_term_l51_5143

theorem coefficient_of_linear_term (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 0) → (a = 1 ∧ b = 3 ∧ c = -1) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_linear_term_l51_5143


namespace NUMINAMATH_CALUDE_cubic_expression_value_l51_5134

theorem cubic_expression_value (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2*x^2 - 12 = -11 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l51_5134


namespace NUMINAMATH_CALUDE_dilution_problem_dilution_solution_l51_5177

theorem dilution_problem (initial_volume : ℝ) (initial_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : Prop :=
  initial_volume = 12 ∧ 
  initial_concentration = 0.6 ∧
  target_concentration = 0.4 ∧
  water_added = 6 →
  initial_volume * initial_concentration = 
    (initial_volume + water_added) * target_concentration

theorem dilution_solution : 
  ∃ (water_added : ℝ), dilution_problem 12 0.6 0.4 water_added :=
by
  sorry

end NUMINAMATH_CALUDE_dilution_problem_dilution_solution_l51_5177


namespace NUMINAMATH_CALUDE_possible_two_black_one_white_l51_5110

/-- Represents the possible marble replacement operations -/
inductive Operation
  | replaceThreeBlackWithTwoBlack
  | replaceTwoBlackOneWhiteWithTwoWhite
  | replaceOneBlackTwoWhiteWithOneBlackOneWhite
  | replaceThreeWhiteWithTwoBlack

/-- Represents the state of the urn -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Applies a single operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.replaceThreeBlackWithTwoBlack =>
      UrnState.mk state.white (state.black - 1)
  | Operation.replaceTwoBlackOneWhiteWithTwoWhite =>
      UrnState.mk (state.white + 1) (state.black - 2)
  | Operation.replaceOneBlackTwoWhiteWithOneBlackOneWhite =>
      UrnState.mk (state.white - 1) state.black
  | Operation.replaceThreeWhiteWithTwoBlack =>
      UrnState.mk (state.white - 3) (state.black + 2)

/-- Theorem: It is possible to reach a state of 2 black marbles and 1 white marble -/
theorem possible_two_black_one_white :
  ∃ (operations : List Operation),
    let initial_state := UrnState.mk 150 200
    let final_state := operations.foldl applyOperation initial_state
    final_state.white = 1 ∧ final_state.black = 2 :=
  sorry


end NUMINAMATH_CALUDE_possible_two_black_one_white_l51_5110


namespace NUMINAMATH_CALUDE_jet_distance_l51_5115

/-- Given a jet that travels 580 miles in 2 hours, prove that it will travel 2900 miles in 10 hours. -/
theorem jet_distance (distance : ℝ) (time : ℝ) (new_time : ℝ) 
    (h1 : distance = 580) 
    (h2 : time = 2) 
    (h3 : new_time = 10) : 
  (distance / time) * new_time = 2900 := by
  sorry

end NUMINAMATH_CALUDE_jet_distance_l51_5115


namespace NUMINAMATH_CALUDE_total_rubber_bands_l51_5135

theorem total_rubber_bands (harper_bands : ℕ) (difference : ℕ) : 
  harper_bands = 15 → difference = 6 → harper_bands + (harper_bands - difference) = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_rubber_bands_l51_5135


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l51_5131

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem statement
theorem quadratic_function_properties :
  (f (-1) = 0 ∧ f 3 = 0 ∧ f 0 = -3) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 4 → f x ≤ 2*m) ↔ 5/2 ≤ m) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l51_5131


namespace NUMINAMATH_CALUDE_chapter_length_l51_5164

theorem chapter_length (pages_per_chapter : ℕ) 
  (h1 : 10 * pages_per_chapter + 20 + 2 * pages_per_chapter = 500) :
  pages_per_chapter = 40 := by
  sorry

end NUMINAMATH_CALUDE_chapter_length_l51_5164


namespace NUMINAMATH_CALUDE_very_spicy_peppers_l51_5100

/-- The number of peppers needed for very spicy curries -/
def V : ℕ := sorry

/-- The number of peppers needed for spicy curries -/
def spicy_peppers : ℕ := 2

/-- The number of peppers needed for mild curries -/
def mild_peppers : ℕ := 1

/-- The number of spicy curries after adjustment -/
def spicy_curries : ℕ := 15

/-- The number of mild curries after adjustment -/
def mild_curries : ℕ := 90

/-- The reduction in the number of peppers bought after adjustment -/
def pepper_reduction : ℕ := 40

theorem very_spicy_peppers : 
  V = pepper_reduction := by sorry

end NUMINAMATH_CALUDE_very_spicy_peppers_l51_5100


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l51_5181

theorem green_shirt_pairs (total_students : ℕ) (blue_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (blue_pairs : ℕ) :
  total_students = 132 →
  blue_students = 65 →
  green_students = 67 →
  total_pairs = 66 →
  blue_pairs = 29 →
  blue_students + green_students = total_students →
  ∃ (green_pairs : ℕ), green_pairs = 30 ∧ 
    blue_pairs + green_pairs + (total_students - 2 * (blue_pairs + green_pairs)) / 2 = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l51_5181


namespace NUMINAMATH_CALUDE_min_distance_between_points_l51_5196

noncomputable section

def f (x : ℝ) : ℝ := Real.sin x + (1/6) * x^3
def g (x : ℝ) : ℝ := x - 1

theorem min_distance_between_points (x₁ x₂ : ℝ) 
  (h₁ : x₁ ≥ 0) (h₂ : x₂ ≥ 0) (h₃ : f x₁ = g x₂) :
  ∃ (d : ℝ), d = |x₂ - x₁| ∧ d ≥ 1 ∧ 
  (∀ (y₁ y₂ : ℝ), y₁ ≥ 0 → y₂ ≥ 0 → f y₁ = g y₂ → |y₂ - y₁| ≥ d) :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_points_l51_5196


namespace NUMINAMATH_CALUDE_bee_count_l51_5118

theorem bee_count (initial_bees : ℕ) (incoming_bees : ℕ) : 
  initial_bees = 16 → incoming_bees = 9 → initial_bees + incoming_bees = 25 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l51_5118


namespace NUMINAMATH_CALUDE_yellow_balls_after_loss_l51_5182

theorem yellow_balls_after_loss (initial_total : ℕ) (current_total : ℕ) (blue : ℕ) (lost : ℕ) : 
  initial_total = 120 →
  current_total = 110 →
  blue = 15 →
  lost = 10 →
  let red := 3 * blue
  let green := red + blue
  let yellow := initial_total - (red + blue + green)
  yellow = 0 := by sorry

end NUMINAMATH_CALUDE_yellow_balls_after_loss_l51_5182


namespace NUMINAMATH_CALUDE_race_distance_P_300_l51_5142

/-- A race between two runners P and Q, where P is faster but Q gets a head start -/
structure Race where
  /-- The speed ratio of P to Q -/
  speed_ratio : ℝ
  /-- The head start given to Q in meters -/
  head_start : ℝ

/-- The distance run by P in the race -/
def distance_P (race : Race) : ℝ :=
  sorry

theorem race_distance_P_300 (race : Race) 
  (h_speed : race.speed_ratio = 1.25)
  (h_head_start : race.head_start = 60)
  (h_tie : distance_P race = distance_P race - race.head_start + race.head_start) :
  distance_P race = 300 :=
sorry

end NUMINAMATH_CALUDE_race_distance_P_300_l51_5142


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l51_5153

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 2}

theorem complement_of_A_in_U : 
  {x : Int | x ∈ U ∧ x ∉ A} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l51_5153


namespace NUMINAMATH_CALUDE_consecutive_nonprime_integers_l51_5141

theorem consecutive_nonprime_integers :
  ∃ n : ℕ,
    100 < n ∧
    n + 4 < 200 ∧
    (¬ Prime n) ∧
    (¬ Prime (n + 1)) ∧
    (¬ Prime (n + 2)) ∧
    (¬ Prime (n + 3)) ∧
    (¬ Prime (n + 4)) ∧
    n + 4 = 148 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_nonprime_integers_l51_5141


namespace NUMINAMATH_CALUDE_f_properties_imply_a_equals_four_l51_5160

/-- A function f(x) = x^2 - ax that is decreasing on (-∞, 2] and increasing on (2, +∞) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 - a*x

/-- The property of f being decreasing on (-∞, 2] -/
def decreasing_on_left (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 2 → f a x > f a y

/-- The property of f being increasing on (2, +∞) -/
def increasing_on_right (a : ℝ) : Prop :=
  ∀ x y, 2 < x → x < y → f a x < f a y

/-- Theorem stating that if f(x) = x^2 - ax is decreasing on (-∞, 2] and increasing on (2, +∞), then a = 4 -/
theorem f_properties_imply_a_equals_four :
  ∀ a : ℝ, decreasing_on_left a → increasing_on_right a → a = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_imply_a_equals_four_l51_5160


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_length_l51_5129

/-- Given a right triangle ABC with legs AB and AC, and points X on AB and Y on AC,
    prove that the hypotenuse BC has length 6√42 under specific conditions. -/
theorem right_triangle_hypotenuse_length 
  (A B C X Y : ℝ × ℝ) -- Points in 2D plane
  (h_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) -- Right angle at A
  (h_X_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2))
  (h_Y_on_AC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Y = (s * C.1 + (1 - s) * A.1, s * C.2 + (1 - s) * A.2))
  (h_AX_XB : dist A X = (1/4) * dist A B)
  (h_AY_YC : dist A Y = (2/3) * dist A C)
  (h_BY : dist B Y = 24)
  (h_CX : dist C X = 18) :
  dist B C = 6 * Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_length_l51_5129


namespace NUMINAMATH_CALUDE_race_time_theorem_l51_5121

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.distance = 1000 ∧
  (r.runner_a.speed * r.runner_a.time = r.distance) ∧
  (r.runner_b.speed * r.runner_b.time = r.distance - 40 ∨
   r.runner_b.speed * (r.runner_a.time + 10) = r.distance)

/-- The theorem to prove -/
theorem race_time_theorem (r : Race) :
  race_conditions r → r.runner_a.time = 240 := by
  sorry

end NUMINAMATH_CALUDE_race_time_theorem_l51_5121


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l51_5112

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = x^2 - x - 6

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - x + 5*y - 6 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-2, 5)

-- Define the line l (vertical case)
def line_l_vertical (x : ℝ) : Prop := x = -2

-- Define the line l (non-vertical case)
def line_l_nonvertical (x y : ℝ) : Prop := 4*x + 3*y - 7 = 0

-- Theorem statement
theorem circle_and_line_problem :
  ∃ (A B : ℝ × ℝ),
    -- Circle C passes through intersection points of parabola and axes
    (∀ (x y : ℝ), (x = 0 ∨ y = 0) ∧ parabola x y → circle_C x y) ∧
    -- Line l passes through P and intersects C at A and B
    (line_l_vertical A.1 ∨ line_l_nonvertical A.1 A.2) ∧
    (line_l_vertical B.1 ∨ line_l_nonvertical B.1 B.2) ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    -- Tangents at A and B are perpendicular
    (∃ (tA tB : ℝ × ℝ → ℝ × ℝ),
      (tA A = B ∨ tB B = A) →
      (tA A • tB B = 0)) →
    -- Conclusion: Equations of circle C and line l
    (∀ (x y : ℝ), circle_C x y ↔ x^2 + y^2 - x + 5*y - 6 = 0) ∧
    (∀ (x y : ℝ), (x = -2 ∨ 4*x + 3*y - 7 = 0) ↔ (line_l_vertical x ∨ line_l_nonvertical x y))
  := by sorry

end NUMINAMATH_CALUDE_circle_and_line_problem_l51_5112


namespace NUMINAMATH_CALUDE_probability_three_same_color_l51_5101

def total_marbles : ℕ := 23
def red_marbles : ℕ := 6
def white_marbles : ℕ := 8
def blue_marbles : ℕ := 9

def probability_same_color : ℚ := 160 / 1771

theorem probability_three_same_color :
  probability_same_color = (Nat.choose red_marbles 3 + Nat.choose white_marbles 3 + Nat.choose blue_marbles 3) / Nat.choose total_marbles 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_same_color_l51_5101


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l51_5146

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  c = 2 →
  C = π / 3 →
  2 * sin (2 * A) + sin (2 * B + C) = sin C →
  (∃ S : ℝ, S = (2 * Real.sqrt 3) / 3 ∧ S = (1 / 2) * a * b * sin C) ∧
  (∃ P : ℝ, P ≤ 6 ∧ P = a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l51_5146


namespace NUMINAMATH_CALUDE_two_children_gender_combinations_l51_5117

/-- Represents the gender of a child -/
inductive Gender
  | Male
  | Female

/-- Represents a pair of children's genders -/
def ChildPair := Gender × Gender

/-- The set of all possible gender combinations for two children -/
def allGenderCombinations : Set ChildPair :=
  {(Gender.Male, Gender.Male), (Gender.Male, Gender.Female),
   (Gender.Female, Gender.Male), (Gender.Female, Gender.Female)}

/-- Theorem stating that the set of all possible gender combinations
    for two children is equal to the expected set -/
theorem two_children_gender_combinations :
  {pair : ChildPair | True} = allGenderCombinations := by
  sorry

end NUMINAMATH_CALUDE_two_children_gender_combinations_l51_5117


namespace NUMINAMATH_CALUDE_jack_morning_letters_l51_5167

def morning_letters (afternoon_letters : ℕ) (difference : ℕ) : ℕ :=
  afternoon_letters + difference

theorem jack_morning_letters :
  morning_letters 7 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_morning_letters_l51_5167


namespace NUMINAMATH_CALUDE_hexadecimal_to_decimal_l51_5155

theorem hexadecimal_to_decimal (k : ℕ) : k > 0 → (1 * 6^3 + k * 6^1 + 5 = 239) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexadecimal_to_decimal_l51_5155


namespace NUMINAMATH_CALUDE_max_integers_above_18_l51_5158

/-- Given 5 integers that sum to 17, the maximum number of these integers
    that can be larger than 18 is 2. -/
theorem max_integers_above_18 (a b c d e : ℤ) : 
  a + b + c + d + e = 17 → 
  (∀ k : ℕ, k ≤ 5 → 
    (∃ (S : Finset ℤ), S.card = k ∧ S ⊆ {a, b, c, d, e} ∧ (∀ x ∈ S, x > 18)) →
    k ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_max_integers_above_18_l51_5158


namespace NUMINAMATH_CALUDE_complement_of_A_in_I_l51_5109

def I : Set ℕ := {x | 0 < x ∧ x < 6}
def A : Set ℕ := {1, 2, 3}

theorem complement_of_A_in_I :
  (I \ A) = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_I_l51_5109


namespace NUMINAMATH_CALUDE_integer_set_equivalence_l51_5148

theorem integer_set_equivalence (a : ℝ) : 
  (a ≤ 1 ∧ (Set.range (fun n : ℤ => (n : ℝ)) ∩ Set.Icc a (2 - a)).ncard = 3) ↔ 
  -1 < a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_integer_set_equivalence_l51_5148


namespace NUMINAMATH_CALUDE_parabola_tangent_intersection_l51_5193

/-- Parabola tangent intersection theorem -/
theorem parabola_tangent_intersection
  (t₁ t₂ : ℝ) (h : t₁ ≠ t₂) :
  let parabola := fun x : ℝ => x^2 / 4
  let tangent₁ := fun x : ℝ => t₁ * x - t₁^2
  let tangent₂ := fun x : ℝ => t₂ * x - t₂^2
  let intersection_x := t₁ + t₂
  let intersection_y := t₁ * t₂
  (parabola (2 * t₁) = t₁^2) ∧
  (parabola (2 * t₂) = t₂^2) ∧
  (tangent₁ intersection_x = intersection_y) ∧
  (tangent₂ intersection_x = intersection_y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_intersection_l51_5193


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l51_5133

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 20) :
  (1 / a + 1 / b) ≥ (1 / 5 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l51_5133


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_four_l51_5154

theorem sum_of_squares_divisible_by_four (n : ℤ) :
  ∃ k : ℤ, (2*n)^2 + (2*n + 2)^2 + (2*n + 4)^2 = 4 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_four_l51_5154


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l51_5140

theorem complex_square_one_plus_i (i : ℂ) : 
  i ^ 2 = -1 → (1 + i) ^ 2 = 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l51_5140


namespace NUMINAMATH_CALUDE_point_C_coordinates_l51_5183

-- Define the points A and B
def A : ℝ × ℝ := (-2, -1)
def B : ℝ × ℝ := (4, 9)

-- Define the condition for point C
def is_point_C (C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧
  C.1 = A.1 + t * (B.1 - A.1) ∧
  C.2 = A.2 + t * (B.2 - A.2) ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 16 * ((B.1 - C.1)^2 + (B.2 - C.2)^2)

-- Theorem statement
theorem point_C_coordinates :
  ∃ C : ℝ × ℝ, is_point_C C ∧ C = (-0.8, 1) :=
sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l51_5183


namespace NUMINAMATH_CALUDE_max_freshmen_is_eight_l51_5178

/-- Represents the relation of knowing each other among freshmen. -/
def Knows (n : ℕ) := Fin n → Fin n → Prop

/-- The property that any 3 people include at least 2 who know each other. -/
def AnyThreeHaveTwoKnown (n : ℕ) (knows : Knows n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    knows a b ∨ knows b c ∨ knows a c

/-- The property that any 4 people include at least 2 who do not know each other. -/
def AnyFourHaveTwoUnknown (n : ℕ) (knows : Knows n) : Prop :=
  ∀ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    ¬(knows a b) ∨ ¬(knows a c) ∨ ¬(knows a d) ∨
    ¬(knows b c) ∨ ¬(knows b d) ∨ ¬(knows c d)

/-- The theorem stating that the maximum number of freshmen satisfying the conditions is 8. -/
theorem max_freshmen_is_eight :
  ∀ n : ℕ, (∃ knows : Knows n, AnyThreeHaveTwoKnown n knows ∧ AnyFourHaveTwoUnknown n knows) →
    n ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_freshmen_is_eight_l51_5178


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l51_5165

/-- Given a circle with diameter endpoints (2, -3) and (8, 5), 
    prove that its center is at (5, 1) and its radius is 5. -/
theorem circle_center_and_radius : 
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (8, 5)
  let center : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  let radius : ℝ := Real.sqrt ((center.1 - a.1)^2 + (center.2 - a.2)^2)
  center = (5, 1) ∧ radius = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l51_5165


namespace NUMINAMATH_CALUDE_logan_gas_budget_l51_5172

/-- Calculates the amount Logan can spend on gas annually --/
def gas_budget (current_income rent groceries desired_savings income_increase : ℕ) : ℕ :=
  (current_income + income_increase) - (rent + groceries + desired_savings)

/-- Proves that Logan's gas budget is $8,000 given his financial constraints --/
theorem logan_gas_budget :
  gas_budget 65000 20000 5000 42000 10000 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_logan_gas_budget_l51_5172


namespace NUMINAMATH_CALUDE_maddie_tshirt_cost_l51_5104

/-- Calculates the total cost of T-shirts bought by Maddie -/
def total_cost (white_packs blue_packs white_per_pack blue_per_pack cost_per_shirt : ℕ) : ℕ :=
  ((white_packs * white_per_pack + blue_packs * blue_per_pack) * cost_per_shirt)

/-- Proves that Maddie spent $66 on T-shirts -/
theorem maddie_tshirt_cost :
  total_cost 2 4 5 3 3 = 66 := by
  sorry

end NUMINAMATH_CALUDE_maddie_tshirt_cost_l51_5104


namespace NUMINAMATH_CALUDE_tractor_production_proof_l51_5170

/-- The number of tractors produced in October -/
def october_production : ℕ := 1000

/-- The additional number of tractors planned to be produced in November and December -/
def additional_production : ℕ := 2310

/-- The percentage increase of the additional production compared to the original plan -/
def percentage_increase : ℚ := 21 / 100

/-- The monthly growth rate for November and December -/
def monthly_growth_rate : ℚ := 1 / 10

/-- The original annual production plan -/
def original_annual_plan : ℕ := 11000

theorem tractor_production_proof :
  (october_production * (1 + monthly_growth_rate) + october_production * (1 + monthly_growth_rate)^2 = additional_production) ∧
  (original_annual_plan + original_annual_plan * percentage_increase = original_annual_plan + additional_production) :=
by sorry

end NUMINAMATH_CALUDE_tractor_production_proof_l51_5170


namespace NUMINAMATH_CALUDE_rope_initial_length_l51_5114

/-- Given a rope cut into pieces, calculate its initial length -/
theorem rope_initial_length
  (num_pieces : ℕ)
  (tied_pieces : ℕ)
  (knot_reduction : ℕ)
  (final_length : ℕ)
  (h1 : num_pieces = 12)
  (h2 : tied_pieces = 3)
  (h3 : knot_reduction = 1)
  (h4 : final_length = 15) :
  (final_length + knot_reduction) * num_pieces = 192 :=
by sorry

end NUMINAMATH_CALUDE_rope_initial_length_l51_5114


namespace NUMINAMATH_CALUDE_lexie_crayon_count_l51_5125

/-- The number of crayons that can fit in each box -/
def crayons_per_box : ℕ := 8

/-- The number of crayon boxes Lexie needs -/
def number_of_boxes : ℕ := 10

/-- The total number of crayons Lexie has -/
def total_crayons : ℕ := crayons_per_box * number_of_boxes

theorem lexie_crayon_count : total_crayons = 80 := by
  sorry

end NUMINAMATH_CALUDE_lexie_crayon_count_l51_5125


namespace NUMINAMATH_CALUDE_asymptotes_necessary_not_sufficient_l51_5179

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form (x²/a² - y²/b² = 1) -/
  equation : ℝ → ℝ → Prop

/-- Represents the asymptotes of a hyperbola -/
structure Asymptotes where
  /-- The equation of the asymptotes in the form y = ±mx -/
  equation : ℝ → ℝ → Prop

/-- The specific hyperbola C with equation x²/9 - y²/16 = 1 -/
def hyperbola_C : Hyperbola :=
  { equation := fun x y => x^2 / 9 - y^2 / 16 = 1 }

/-- The asymptotes with equation y = ±(4/3)x -/
def asymptotes_C : Asymptotes :=
  { equation := fun x y => y = 4/3 * x ∨ y = -4/3 * x }

/-- Theorem stating that the given asymptote equation is a necessary but not sufficient condition for the hyperbola equation -/
theorem asymptotes_necessary_not_sufficient :
  (∀ x y, hyperbola_C.equation x y → asymptotes_C.equation x y) ∧
  ¬(∀ x y, asymptotes_C.equation x y → hyperbola_C.equation x y) := by
  sorry

end NUMINAMATH_CALUDE_asymptotes_necessary_not_sufficient_l51_5179


namespace NUMINAMATH_CALUDE_discount_percentage_l51_5144

def original_price : ℝ := 6
def num_bags : ℕ := 2
def total_spent : ℝ := 3

theorem discount_percentage : 
  (1 - total_spent / (original_price * num_bags)) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l51_5144


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l51_5163

/-- Given a quadratic function f(x) = 3x^2 + 2x - 5, when shifted 6 units to the right,
    the resulting function g(x) = ax^2 + bx + c has coefficients that sum to 60. -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, (3 * (x - 6)^2 + 2 * (x - 6) - 5) = (a * x^2 + b * x + c)) →
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l51_5163


namespace NUMINAMATH_CALUDE_ellipse_equation_l51_5185

/-- Represents an ellipse with its properties -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- The standard equation of an ellipse -/
def standardEquation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Conditions for the ellipse -/
def ellipseConditions (e : Ellipse) : Prop :=
  e.a = 2 * Real.sqrt 3 ∧
  e.c = Real.sqrt 3 ∧
  e.b^2 = e.a^2 - e.c^2 ∧
  e.a = 3 * e.b ∧
  standardEquation e 3 0

theorem ellipse_equation (e : Ellipse) (h : ellipseConditions e) :
  (∀ x y, standardEquation e x y ↔ x^2 / 12 + y^2 / 9 = 1) ∨
  (∀ x y, standardEquation e x y ↔ x^2 / 9 + y^2 / 12 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l51_5185


namespace NUMINAMATH_CALUDE_triangle_3_4_5_l51_5168

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Theorem stating that line segments of lengths 3, 4, and 5 can form a triangle -/
theorem triangle_3_4_5 : can_form_triangle 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_3_4_5_l51_5168


namespace NUMINAMATH_CALUDE_equation_solution_l51_5126

theorem equation_solution (x k : ℝ) : 
  (7 * x + 2 = 3 * x - 6) ∧ (x + 1 = k) → 3 * k^2 - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l51_5126


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l51_5150

/-- An isosceles triangle with side lengths 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Two sides are 9, one side is 4
    (b = c) →                  -- The triangle is isosceles
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (a + b + c = 22)           -- The perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 4 9 9 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l51_5150


namespace NUMINAMATH_CALUDE_max_temp_difference_example_l51_5113

/-- The maximum temperature difference given the highest and lowest temperatures -/
def max_temp_difference (highest lowest : ℤ) : ℤ :=
  highest - lowest

/-- Theorem: The maximum temperature difference is 20℃ given the highest temperature of 18℃ and lowest temperature of -2℃ -/
theorem max_temp_difference_example : max_temp_difference 18 (-2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_temp_difference_example_l51_5113


namespace NUMINAMATH_CALUDE_parts_probability_theorem_l51_5102

/-- Represents the outcome of drawing a part -/
inductive DrawOutcome
| Standard
| NonStandard

/-- Represents the type of part that was lost -/
inductive LostPart
| Standard
| NonStandard

/-- The probability model for the parts problem -/
structure PartsModel where
  initialStandard : ℕ
  initialNonStandard : ℕ
  lostPart : LostPart
  drawnPart : DrawOutcome

def PartsModel.totalInitial (m : PartsModel) : ℕ :=
  m.initialStandard + m.initialNonStandard

def PartsModel.remainingTotal (m : PartsModel) : ℕ :=
  m.totalInitial - 1

def PartsModel.remainingStandard (m : PartsModel) : ℕ :=
  match m.lostPart with
  | LostPart.Standard => m.initialStandard - 1
  | LostPart.NonStandard => m.initialStandard

def PartsModel.probability (m : PartsModel) (event : PartsModel → Prop) : ℚ :=
  sorry

theorem parts_probability_theorem (m : PartsModel) 
  (h1 : m.initialStandard = 21)
  (h2 : m.initialNonStandard = 10)
  (h3 : m.drawnPart = DrawOutcome.Standard) :
  (m.probability (fun model => model.lostPart = LostPart.Standard) = 2/3) ∧
  (m.probability (fun model => model.lostPart = LostPart.NonStandard) = 1/3) :=
sorry

end NUMINAMATH_CALUDE_parts_probability_theorem_l51_5102


namespace NUMINAMATH_CALUDE_peppers_total_weight_l51_5174

theorem peppers_total_weight : 
  let green_peppers : Float := 0.3333333333333333
  let red_peppers : Float := 0.3333333333333333
  let yellow_peppers : Float := 0.25
  let orange_peppers : Float := 0.5
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 1.4166666666666665 := by
  sorry

end NUMINAMATH_CALUDE_peppers_total_weight_l51_5174


namespace NUMINAMATH_CALUDE_base_nine_to_ten_l51_5198

theorem base_nine_to_ten : 
  (3 * 9^3 + 7 * 9^2 + 2 * 9^1 + 5 * 9^0) = 2777 := by
  sorry

end NUMINAMATH_CALUDE_base_nine_to_ten_l51_5198


namespace NUMINAMATH_CALUDE_ages_product_l51_5152

/-- Represents the ages of the individuals in the problem -/
structure Ages where
  thomas : ℕ
  roy : ℕ
  kelly : ℕ
  julia : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.thomas = ages.roy - 6 ∧
  ages.thomas = ages.kelly + 4 ∧
  ages.roy = ages.julia + 8 ∧
  ages.roy = ages.kelly + 4 ∧
  ages.roy + 2 = 3 * (ages.julia + 2) ∧
  ages.thomas + 2 = 2 * (ages.kelly + 2)

/-- The theorem to be proved -/
theorem ages_product (ages : Ages) :
  satisfies_conditions ages →
  (ages.roy + 2) * (ages.kelly + 2) * (ages.thomas + 2) = 576 := by
  sorry

end NUMINAMATH_CALUDE_ages_product_l51_5152


namespace NUMINAMATH_CALUDE_logarithm_and_exponent_calculation_l51_5119

theorem logarithm_and_exponent_calculation :
  (2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2) ∧
  ((0.027 ^ (-1/3 : ℝ)) - ((-1/7 : ℝ)⁻¹) + ((2 + 7/9 : ℝ) ^ (1/2 : ℝ)) - ((Real.sqrt 2 - 1) ^ (0 : ℝ)) = 11) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_and_exponent_calculation_l51_5119


namespace NUMINAMATH_CALUDE_inequality_solution_set_l51_5180

theorem inequality_solution_set (x : ℝ) : 
  (1 + 2 * (x - 1) ≤ 3) ↔ (x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l51_5180


namespace NUMINAMATH_CALUDE_exactly_two_valid_sets_l51_5151

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)  -- The first integer in the set
  (length : ℕ) -- The number of integers in the set

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set according to our problem -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 3 ∧ sum_consecutive s = 21

theorem exactly_two_valid_sets :
  ∃! (s₁ s₂ : ConsecutiveSet), is_valid_set s₁ ∧ is_valid_set s₂ ∧ s₁ ≠ s₂ ∧
    ∀ (s : ConsecutiveSet), is_valid_set s → s = s₁ ∨ s = s₂ :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_sets_l51_5151


namespace NUMINAMATH_CALUDE_domino_tiling_triomino_tiling_l_tetromino_tiling_l51_5186

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a tile --/
structure Tile :=
  (size : ℕ)

/-- Defines a 9x9 chessboard --/
def chessboard_9x9 : Chessboard :=
  ⟨9 * 9⟩

/-- Defines a 2x1 domino --/
def domino : Tile :=
  ⟨2⟩

/-- Defines a 3x1 triomino --/
def triomino : Tile :=
  ⟨3⟩

/-- Defines an L-shaped tetromino --/
def l_tetromino : Tile :=
  ⟨4⟩

/-- Determines if a chessboard can be tiled with a given tile --/
def can_tile (c : Chessboard) (t : Tile) : Prop :=
  c.size % t.size = 0

theorem domino_tiling :
  ¬ can_tile chessboard_9x9 domino :=
sorry

theorem triomino_tiling :
  can_tile chessboard_9x9 triomino :=
sorry

theorem l_tetromino_tiling :
  ¬ can_tile chessboard_9x9 l_tetromino :=
sorry

end NUMINAMATH_CALUDE_domino_tiling_triomino_tiling_l_tetromino_tiling_l51_5186


namespace NUMINAMATH_CALUDE_min_value_of_z_l51_5194

theorem min_value_of_z (x y : ℝ) (h1 : x - 1 ≥ 0) (h2 : 2 * x - y - 1 ≤ 0) (h3 : x + y - 3 ≤ 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≥ -1 ∧ ∀ (w : ℝ), w = x - y → w ≥ z :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l51_5194


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l51_5107

theorem no_real_roots_quadratic (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 - 2 * x + 1 ≠ 0) → m > 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l51_5107


namespace NUMINAMATH_CALUDE_e_2i_in_second_quadrant_l51_5189

open Complex

theorem e_2i_in_second_quadrant :
  let z : ℂ := Complex.exp (2 * I)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_e_2i_in_second_quadrant_l51_5189


namespace NUMINAMATH_CALUDE_sum_abcd_l51_5149

theorem sum_abcd (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ 
       b + 3 = c + 4 ∧ 
       c + 4 = d + 5 ∧ 
       d + 5 = a + b + c + d + 15) : 
  a + b + c + d = -46/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_l51_5149


namespace NUMINAMATH_CALUDE_initial_employees_correct_l51_5190

/-- Represents the initial number of employees in a company. -/
def initial_employees : ℕ := 450

/-- Represents the monthly salary of each employee in dollars. -/
def salary_per_employee : ℕ := 2000

/-- Represents the fraction of employees remaining after layoffs. -/
def remaining_fraction : ℚ := 2/3

/-- Represents the total amount paid to remaining employees in dollars. -/
def total_paid : ℕ := 600000

/-- Theorem stating that the initial number of employees is correct given the conditions. -/
theorem initial_employees_correct : 
  (initial_employees : ℚ) * remaining_fraction * salary_per_employee = total_paid :=
sorry

end NUMINAMATH_CALUDE_initial_employees_correct_l51_5190


namespace NUMINAMATH_CALUDE_fraction_addition_and_multiplication_l51_5157

theorem fraction_addition_and_multiplication :
  (7 / 12 + 3 / 8) * 2 / 3 = 23 / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_and_multiplication_l51_5157


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_3_l51_5184

theorem infinitely_many_primes_4k_plus_3 : 
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ Prime p ∧ ∃ k : ℕ, p = 4 * k + 3 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_3_l51_5184


namespace NUMINAMATH_CALUDE_rhombus_area_l51_5111

/-- The area of a rhombus with side length 13 and one diagonal 24 is 120 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (area : ℝ) : 
  side = 13 → diagonal1 = 24 → area = (diagonal1 * (2 * Real.sqrt (side^2 - (diagonal1/2)^2))) / 2 → area = 120 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l51_5111


namespace NUMINAMATH_CALUDE_radical_axis_intersection_squared_distance_l51_5197

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB : Real)
  (BC : Real)
  (CA : Real)

-- Define the incircle and its touchpoints
structure Incircle :=
  (I : ℝ × ℝ)
  (M N D : ℝ × ℝ)

-- Define point K
def K (t : Triangle) (inc : Incircle) : ℝ × ℝ := sorry

-- Define circumcircles of triangles MAN and KID
def CircumcircleMAN (t : Triangle) (inc : Incircle) : ℝ × ℝ := sorry
def CircumcircleKID (t : Triangle) (inc : Incircle) : ℝ × ℝ := sorry

-- Define the radical axis
def RadicalAxis (c1 c2 : ℝ × ℝ) : ℝ × ℝ → Prop := sorry

-- Define L₁ and L₂
def L₁ (t : Triangle) (ra : ℝ × ℝ → Prop) : ℝ × ℝ := sorry
def L₂ (t : Triangle) (ra : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

-- Distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem radical_axis_intersection_squared_distance
  (t : Triangle)
  (inc : Incircle)
  (h1 : t.AB = 36)
  (h2 : t.BC = 48)
  (h3 : t.CA = 60)
  (h4 : inc.M = sorry)  -- Point where incircle touches AB
  (h5 : inc.N = sorry)  -- Point where incircle touches AC
  (h6 : inc.D = sorry)  -- Point where incircle touches BC
  :
  let k := K t inc
  let c1 := CircumcircleMAN t inc
  let c2 := CircumcircleKID t inc
  let ra := RadicalAxis c1 c2
  let l1 := L₁ t ra
  let l2 := L₂ t ra
  (distance l1 l2)^2 = 720 := by sorry

end NUMINAMATH_CALUDE_radical_axis_intersection_squared_distance_l51_5197


namespace NUMINAMATH_CALUDE_traffic_sampling_is_systematic_l51_5106

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Quota

/-- Represents the characteristics of the sampling process --/
structure SamplingProcess where
  interval : ℕ  -- Time interval between samples
  continuous_stream : Bool  -- Whether there's a continuous stream of units to sample

/-- Determines if a sampling process is systematic --/
def is_systematic (process : SamplingProcess) : Prop :=
  process.interval > 0 ∧ process.continuous_stream

/-- The traffic police sampling process --/
def traffic_sampling : SamplingProcess :=
  { interval := 3,  -- 3 minutes interval
    continuous_stream := true }  -- Continuous stream of passing cars

/-- Theorem stating that the traffic sampling method is systematic --/
theorem traffic_sampling_is_systematic :
  is_systematic traffic_sampling ↔ SamplingMethod.Systematic = 
    (match traffic_sampling with
     | { interval := 3, continuous_stream := true } => SamplingMethod.Systematic
     | _ => SamplingMethod.SimpleRandom) :=
sorry

end NUMINAMATH_CALUDE_traffic_sampling_is_systematic_l51_5106


namespace NUMINAMATH_CALUDE_jo_stair_climbing_l51_5176

/-- The number of ways to climb n stairs, taking 1, 2, or 3 stairs at a time -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => climbStairs (n + 2) + climbStairs (n + 1) + climbStairs n

/-- The number of stairs Jo climbs -/
def totalStairs : ℕ := 8

theorem jo_stair_climbing :
  climbStairs totalStairs = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_jo_stair_climbing_l51_5176


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l51_5127

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}

theorem intersection_M_complement_N : M ∩ (U \ N) = {4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l51_5127


namespace NUMINAMATH_CALUDE_unique_quadrilateral_from_centers_l51_5161

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Checks if a circle can be inscribed in a quadrilateral -/
def hasInscribedCircle (q : Quadrilateral) : Prop := sorry

/-- Checks if a circle can be circumscribed around a quadrilateral -/
def hasCircumscribedCircle (q : Quadrilateral) : Prop := sorry

/-- Gets the center of the inscribed circle of a quadrilateral -/
def getInscribedCenter (q : Quadrilateral) : Point2D := sorry

/-- Gets the center of the circumscribed circle of a quadrilateral -/
def getCircumscribedCenter (q : Quadrilateral) : Point2D := sorry

/-- Gets the intersection point of lines connecting midpoints of opposite sides -/
def getMidpointIntersection (q : Quadrilateral) : Point2D := sorry

/-- Theorem: A unique quadrilateral can be determined from its inscribed circle center,
    circumscribed circle center, and the intersection of midpoint lines -/
theorem unique_quadrilateral_from_centers
  (I O M : Point2D) :
  ∃! q : Quadrilateral,
    hasInscribedCircle q ∧
    hasCircumscribedCircle q ∧
    getInscribedCenter q = I ∧
    getCircumscribedCenter q = O ∧
    getMidpointIntersection q = M :=
  sorry

end NUMINAMATH_CALUDE_unique_quadrilateral_from_centers_l51_5161


namespace NUMINAMATH_CALUDE_polynomial_identity_l51_5123

/-- 
Given a, b, and c, prove that 
a(b - c)³ + b(c - a)³ + c(a - b)³ + (a - b)²(b - c)²(c - a)² = (a - b)(b - c)(c - a)(a + b + c + abc)
-/
theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 + (a - b)^2 * (b - c)^2 * (c - a)^2 = 
  (a - b) * (b - c) * (c - a) * (a + b + c + a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l51_5123


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l51_5173

theorem quadrilateral_side_length (A B C D : ℝ × ℝ) : 
  let angle (p q r : ℝ × ℝ) := Real.arccos ((p.1 - q.1) * (r.1 - q.1) + (p.2 - q.2) * (r.2 - q.2)) / 
    (Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) * Real.sqrt ((r.1 - q.1)^2 + (r.2 - q.2)^2))
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  angle D A B = π/3 ∧ 
  angle A B C = π/2 ∧ 
  angle B C D = π/2 ∧ 
  dist B C = 2 ∧ 
  dist C D = 3 →
  dist A B = 8 / Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_quadrilateral_side_length_l51_5173


namespace NUMINAMATH_CALUDE_walking_speed_proof_l51_5124

/-- The walking speed of person A in km/h -/
def a_speed : ℝ := 10

/-- The cycling speed of person B in km/h -/
def b_speed : ℝ := 20

/-- The time difference between A's start and B's start in hours -/
def time_diff : ℝ := 4

/-- The distance at which B catches up with A in km -/
def catch_up_distance : ℝ := 80

theorem walking_speed_proof :
  (catch_up_distance / a_speed = time_diff + catch_up_distance / b_speed) →
  a_speed = 10 := by
  sorry

#check walking_speed_proof

end NUMINAMATH_CALUDE_walking_speed_proof_l51_5124


namespace NUMINAMATH_CALUDE_simplify_expression_l51_5187

theorem simplify_expression : (2^10 + 7^5) * (2^3 - (-2)^3)^8 = 76600653103936 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l51_5187


namespace NUMINAMATH_CALUDE_tethered_dog_area_tethered_dog_area_exact_l51_5139

/-- The area outside a regular hexagon that can be reached by a tethered point -/
theorem tethered_dog_area (side_length : ℝ) (rope_length : ℝ) 
  (h1 : side_length = 2)
  (h2 : rope_length = 3) : 
  ℝ :=
let hexagon_area := 3 * Real.sqrt 3 * side_length ^ 2 / 2
let circle_area := Real.pi * rope_length ^ 2
circle_area - hexagon_area

/-- The main theorem stating the exact area -/
theorem tethered_dog_area_exact : 
  tethered_dog_area 2 3 rfl rfl = 9 * Real.pi - 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_tethered_dog_area_tethered_dog_area_exact_l51_5139


namespace NUMINAMATH_CALUDE_toms_calculation_l51_5137

theorem toms_calculation (y : ℝ) (h : 4 * y + 7 = 39) : (y + 7) * 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_toms_calculation_l51_5137


namespace NUMINAMATH_CALUDE_correct_graph_representation_l51_5138

/-- Represents a segment of Mike's trip -/
inductive TripSegment
  | CityDriving
  | HighwayDriving
  | Shopping
  | Refueling

/-- Represents the slope of a graph segment -/
inductive Slope
  | Flat
  | Gradual
  | Steep

/-- Represents Mike's trip -/
structure MikeTrip where
  segments : List TripSegment
  shoppingDuration : ℝ
  refuelingDuration : ℝ

/-- Represents a graph of Mike's trip -/
structure TripGraph where
  flatSections : Nat
  slopes : List Slope

/-- The correct graph representation of Mike's trip -/
def correctGraph : TripGraph :=
  { flatSections := 2
  , slopes := [Slope.Gradual, Slope.Steep, Slope.Flat, Slope.Flat, Slope.Steep, Slope.Gradual] }

theorem correct_graph_representation (trip : MikeTrip)
  (h1 : trip.segments = [TripSegment.CityDriving, TripSegment.HighwayDriving, TripSegment.Shopping, TripSegment.Refueling, TripSegment.HighwayDriving, TripSegment.CityDriving])
  (h2 : trip.shoppingDuration = 2)
  (h3 : trip.refuelingDuration = 0.5)
  : TripGraph.flatSections correctGraph = 2 ∧ 
    TripGraph.slopes correctGraph = [Slope.Gradual, Slope.Steep, Slope.Flat, Slope.Flat, Slope.Steep, Slope.Gradual] := by
  sorry

end NUMINAMATH_CALUDE_correct_graph_representation_l51_5138


namespace NUMINAMATH_CALUDE_jennifer_blue_sweets_l51_5136

theorem jennifer_blue_sweets (green : ℕ) (yellow : ℕ) (people : ℕ) (sweets_per_person : ℕ) :
  green = 212 →
  yellow = 502 →
  people = 4 →
  sweets_per_person = 256 →
  people * sweets_per_person - (green + yellow) = 310 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_blue_sweets_l51_5136


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l51_5145

/-- Given a geometric sequence with first term a and second term b,
    this function returns the nth term of the sequence. -/
def geometric_sequence_term (a b : ℚ) (n : ℕ) : ℚ :=
  let r := b / a
  a * r ^ (n - 1)

/-- Theorem stating that the 10th term of the geometric sequence
    with first term 8 and second term -16/3 is -4096/19683. -/
theorem tenth_term_of_specific_geometric_sequence :
  geometric_sequence_term 8 (-16/3) 10 = -4096/19683 := by
  sorry

#eval geometric_sequence_term 8 (-16/3) 10

end NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l51_5145


namespace NUMINAMATH_CALUDE_twins_age_problem_l51_5156

theorem twins_age_problem :
  ∀ (x y : ℕ),
  x * x = 8 →
  (x + y) * (x + y) = x * x + 17 →
  y = 3 :=
by sorry

end NUMINAMATH_CALUDE_twins_age_problem_l51_5156


namespace NUMINAMATH_CALUDE_power_sum_of_i_l51_5159

-- Define the imaginary unit i
axiom i : ℂ
axiom i_squared : i^2 = -1

-- State the theorem
theorem power_sum_of_i : i^2023 + i^303 = -2*i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l51_5159


namespace NUMINAMATH_CALUDE_sum_abs_coeff_2x_minus_1_pow_5_l51_5116

/-- The sum of absolute values of coefficients (excluding constant term) 
    in the expansion of (2x-1)^5 is 242 -/
theorem sum_abs_coeff_2x_minus_1_pow_5 :
  let f : ℝ → ℝ := fun x ↦ (2*x - 1)^5
  ∃ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
    (∀ x, f x = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) ∧
    |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 242 :=
by sorry

end NUMINAMATH_CALUDE_sum_abs_coeff_2x_minus_1_pow_5_l51_5116


namespace NUMINAMATH_CALUDE_mountain_height_l51_5103

/-- Given a mountain where a person makes 10 round trips, reaching 3/4 of the height each time,
    and covering a total distance of 600,000 feet, the height of the mountain is 80,000 feet. -/
theorem mountain_height (trips : ℕ) (fraction_reached : ℚ) (total_distance : ℕ) 
    (h1 : trips = 10)
    (h2 : fraction_reached = 3/4)
    (h3 : total_distance = 600000) :
  (total_distance : ℚ) / (2 * trips * fraction_reached) = 80000 := by
  sorry

end NUMINAMATH_CALUDE_mountain_height_l51_5103


namespace NUMINAMATH_CALUDE_maintain_ratio_theorem_l51_5105

/-- Represents the ingredients in a cake recipe -/
structure Recipe where
  flour : Float
  sugar : Float
  oil : Float

/-- Calculates the new amounts of ingredients while maintaining the ratio -/
def calculate_new_amounts (original : Recipe) (new_flour : Float) : Recipe :=
  let scale_factor := new_flour / original.flour
  { flour := new_flour,
    sugar := original.sugar * scale_factor,
    oil := original.oil * scale_factor }

/-- Rounds a float to two decimal places -/
def round_to_two_decimals (x : Float) : Float :=
  (x * 100).round / 100

theorem maintain_ratio_theorem (original : Recipe) (extra_flour : Float) :
  let new_recipe := calculate_new_amounts original (original.flour + extra_flour)
  round_to_two_decimals new_recipe.sugar = 3.86 ∧
  round_to_two_decimals new_recipe.oil = 2.57 :=
by sorry

end NUMINAMATH_CALUDE_maintain_ratio_theorem_l51_5105


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l51_5132

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l51_5132


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l51_5199

/-- An isosceles triangle with perimeter 60 and two equal sides of length x has a base of length 60 - 2x -/
theorem isosceles_triangle_base_length (x : ℝ) (h : x > 0) : 
  let y := 60 - 2*x
  (2*x + y = 60) ∧ (y = -2*x + 60) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l51_5199


namespace NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l51_5130

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem specific_fraction_division :
  (3 : ℚ) / 7 / ((5 : ℚ) / 9) = 27 / 35 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l51_5130


namespace NUMINAMATH_CALUDE_extra_fruits_count_l51_5147

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 75

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 35

/-- The number of oranges ordered by the cafeteria -/
def oranges : ℕ := 40

/-- The number of bananas ordered by the cafeteria -/
def bananas : ℕ := 20

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 17

/-- The total number of fruits ordered by the cafeteria -/
def total_fruits : ℕ := red_apples + green_apples + oranges + bananas

/-- The number of extra fruits the cafeteria ended up with -/
def extra_fruits : ℕ := total_fruits - students_wanting_fruit

theorem extra_fruits_count : extra_fruits = 153 := by
  sorry

end NUMINAMATH_CALUDE_extra_fruits_count_l51_5147


namespace NUMINAMATH_CALUDE_imohkprelim_combinations_l51_5188

def letter_list : List Char := ['I', 'M', 'O', 'H', 'K', 'P', 'R', 'E', 'L', 'I', 'M']

def count_combinations (letters : List Char) : Nat :=
  let unique_letters := letters.eraseDups
  let combinations_distinct := Nat.choose unique_letters.length 3
  let combinations_with_repeat := 
    (letters.filter (λ c => letters.count c > 1)).eraseDups.length * (unique_letters.length - 1)
  combinations_distinct + combinations_with_repeat

theorem imohkprelim_combinations :
  count_combinations letter_list = 100 := by
  sorry

end NUMINAMATH_CALUDE_imohkprelim_combinations_l51_5188


namespace NUMINAMATH_CALUDE_remainder_13754_div_11_l51_5169

theorem remainder_13754_div_11 : 13754 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13754_div_11_l51_5169


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_product_l51_5108

theorem sum_of_powers_equals_product (n : ℕ) : 
  5^n + 5^n + 5^n + 5^n = 4 * 5^n := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_product_l51_5108


namespace NUMINAMATH_CALUDE_finite_valid_combinations_l51_5166

/-- Represents the number of banknotes of each denomination --/
structure Banknotes :=
  (hun : Nat)
  (fif : Nat)
  (twe : Nat)
  (ten : Nat)

/-- The total value of a set of banknotes in yuan --/
def totalValue (b : Banknotes) : Nat :=
  100 * b.hun + 50 * b.fif + 20 * b.twe + 10 * b.ten

/-- The available banknotes --/
def availableBanknotes : Banknotes :=
  ⟨1, 2, 5, 10⟩

/-- A valid combination of banknotes is one that sums to 200 yuan and doesn't exceed the available banknotes --/
def isValidCombination (b : Banknotes) : Prop :=
  totalValue b = 200 ∧
  b.hun ≤ availableBanknotes.hun ∧
  b.fif ≤ availableBanknotes.fif ∧
  b.twe ≤ availableBanknotes.twe ∧
  b.ten ≤ availableBanknotes.ten

theorem finite_valid_combinations :
  ∃ (n : Nat), ∃ (combinations : Finset Banknotes),
    combinations.card = n ∧
    (∀ b ∈ combinations, isValidCombination b) ∧
    (∀ b, isValidCombination b → b ∈ combinations) :=
by sorry

end NUMINAMATH_CALUDE_finite_valid_combinations_l51_5166


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l51_5122

/-- The equation of a circle passing through (0, 0), (-2, 3), and (-4, 1) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + (19/5)*x - (9/5)*y = 0

/-- Theorem stating that the circle passes through the required points -/
theorem circle_passes_through_points :
  circle_equation 0 0 ∧ circle_equation (-2) 3 ∧ circle_equation (-4) 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l51_5122


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l51_5120

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem perpendicular_tangents_ratio (a b : ℝ) :
  (a * P.1 - b * P.2 - 2 = 0) →  -- Line equation at point P
  (curve P.1 = P.2) →            -- Curve passes through P
  (curve_derivative P.1 * (a / b) = -1) →  -- Perpendicular tangents condition
  a / b = -1/4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l51_5120


namespace NUMINAMATH_CALUDE_miniature_toy_height_difference_l51_5195

/-- Heights of different poodle types -/
structure PoodleHeights where
  standard : ℕ
  miniature : ℕ
  toy : ℕ

/-- The conditions given in the problem -/
def problem_conditions (h : PoodleHeights) : Prop :=
  h.standard = 28 ∧ h.toy = 14 ∧ h.standard = h.miniature + 8

/-- The theorem to be proved -/
theorem miniature_toy_height_difference (h : PoodleHeights) 
  (hc : problem_conditions h) : h.miniature - h.toy = 6 := by
  sorry

end NUMINAMATH_CALUDE_miniature_toy_height_difference_l51_5195
