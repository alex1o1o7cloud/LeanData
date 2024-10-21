import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_one_f_greater_than_g_iff_f_squared_greater_than_g_iff_l753_75353

open Real

noncomputable def f (x : ℝ) : ℝ := sin x / x
noncomputable def g (a x : ℝ) : ℝ := a * cos x

def open_interval_0_pi_2 (x : ℝ) : Prop := 0 < x ∧ x < π/2

def open_interval_minus_pi_2_0_union_0_pi_2 (x : ℝ) : Prop :=
  (-π/2 < x ∧ x < 0) ∨ (0 < x ∧ x < π/2)

-- Statement 1
theorem f_less_than_one :
  ∀ x, open_interval_0_pi_2 x → f x < 1 := by
  sorry

-- Statement 2
theorem f_greater_than_g_iff :
  ∀ a, (∀ x, open_interval_minus_pi_2_0_union_0_pi_2 x → f x > g a x) ↔ a ≤ 1 := by
  sorry

-- Statement 3
theorem f_squared_greater_than_g_iff :
  ∀ a, (∀ x, open_interval_minus_pi_2_0_union_0_pi_2 x → (f x)^2 > g a x) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_one_f_greater_than_g_iff_f_squared_greater_than_g_iff_l753_75353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_and_maximum_l753_75336

-- Define the function f(x)
def f (x : ℝ) : ℝ := -3 * x^2 + 6 * x

-- Define the area function S(t)
noncomputable def S (t : ℝ) : ℝ := ∫ x in t..(t+1), f x

-- Theorem statement
theorem area_and_maximum (t : ℝ) (h : 0 ≤ t ∧ t ≤ 2) :
  S t = -3 * t^2 + 3 * t + 2 ∧
  ∃ (max_value : ℝ), max_value = 5/4 ∧ ∀ s, 0 ≤ s ∧ s ≤ 2 → S s ≤ max_value :=
by
  sorry

#check area_and_maximum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_and_maximum_l753_75336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_correct_l753_75399

/-- The area of the region enclosed by two squares with side lengths 3 and 6,
    and two line segments joining their vertices. -/
noncomputable def enclosedArea : ℝ := 63

/-- The side length of the smaller square. -/
noncomputable def smallSquareSide : ℝ := 3

/-- The side length of the larger square. -/
noncomputable def largeSquareSide : ℝ := 6

/-- The area of the smaller square. -/
noncomputable def smallSquareArea : ℝ := smallSquareSide ^ 2

/-- The area of the larger square. -/
noncomputable def largeSquareArea : ℝ := largeSquareSide ^ 2

/-- The area of one triangle formed by the intersecting line segments. -/
noncomputable def triangleArea : ℝ := (1 / 2) * smallSquareSide * largeSquareSide

/-- Theorem stating that the enclosed area is correct. -/
theorem enclosed_area_is_correct : 
  enclosedArea = smallSquareArea + largeSquareArea + 2 * triangleArea := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_correct_l753_75399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_beta_values_l753_75395

open Real

/-- Given function f(x) = tan(x + π/4) -/
noncomputable def f (x : ℝ) : ℝ := tan (x + π/4)

/-- The domain of f(x) -/
def domain : Set ℝ := {x | ∀ k : ℤ, x ≠ k * π + π/4}

/-- β is in the open interval (0, π) -/
def β_range (β : ℝ) : Prop := 0 < β ∧ β < π

/-- The equation f(β) = 2cos(β - π/4) -/
def equation (β : ℝ) : Prop := f β = 2 * cos (β - π/4)

theorem domain_and_beta_values :
  ∀ β : ℝ, β_range β → equation β →
  (β = π/12 ∨ β = 3*π/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_beta_values_l753_75395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_ratio_l753_75322

theorem angle_terminal_side_ratio (α : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = 4) →
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_ratio_l753_75322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_staircase_handrail_length_l753_75365

/-- The length of a spiral staircase handrail -/
noncomputable def handrail_length (radius : ℝ) (height : ℝ) (angle : ℝ) : ℝ :=
  Real.sqrt (height^2 + (angle / 360 * 2 * Real.pi * radius)^2)

/-- Theorem: The length of a spiral staircase handrail with given parameters is approximately 17.4 feet -/
theorem spiral_staircase_handrail_length :
  (⌊handrail_length 4 12 180 * 10⌋ : ℝ) / 10 = 17.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_staircase_handrail_length_l753_75365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defeated_candidate_vote_percentage_approx_l753_75341

/-- The percentage of votes received by the defeated candidate in an election -/
def defeated_candidate_vote_percentage (total_votes : ℕ) (invalid_votes : ℕ) (victory_margin : ℕ) : ℚ :=
  let valid_votes := total_votes - invalid_votes
  let defeated_votes := (valid_votes - victory_margin) / 2
  (defeated_votes : ℚ) / (valid_votes : ℚ) * 100

/-- Theorem stating the percentage of votes received by the defeated candidate -/
theorem defeated_candidate_vote_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  |defeated_candidate_vote_percentage 850 10 500 - 20.24| < ε := by
  sorry

#eval defeated_candidate_vote_percentage 850 10 500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defeated_candidate_vote_percentage_approx_l753_75341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_coverage_l753_75384

-- Define the diameters of the rings
def diameter_X : ℝ := 16
def diameter_Y : ℝ := 18

-- Define a function to calculate the area of a circle given its diameter
noncomputable def circle_area (diameter : ℝ) : ℝ := Real.pi * (diameter / 2) ^ 2

-- Define the fraction of X's surface not covered by Y
noncomputable def fraction_not_covered : ℝ := 
  (circle_area diameter_X - circle_area diameter_Y) / circle_area diameter_X

-- Theorem statement
theorem ring_coverage : fraction_not_covered = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_coverage_l753_75384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_right_triangle_13_is_182_l753_75367

/-- A Pythagorean triple representing the sides of a right triangle. -/
structure MyPythagoreanTriple where
  a : ℕ
  b : ℕ
  c : ℕ
  is_pythagorean : a * a + b * b = c * c

/-- The minimum perimeter of a right triangle with integer sides and one leg a multiple of 13. -/
def min_perimeter_right_triangle_13 : ℕ := 182

/-- Theorem stating that the minimum perimeter of a right triangle with integer sides
    and one leg a multiple of 13 is 182. -/
theorem min_perimeter_right_triangle_13_is_182 :
  ∀ (t : MyPythagoreanTriple),
    (t.a % 13 = 0 ∨ t.b % 13 = 0) →
    t.a + t.b + t.c ≥ min_perimeter_right_triangle_13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_right_triangle_13_is_182_l753_75367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_wins_l753_75339

/-- Represents the state of the game -/
structure GameState where
  current : Nat
  player_turn : Bool  -- true for Player A, false for Player B

/-- Checks if a number is a factor of another number -/
def is_factor (a b : Nat) : Bool :=
  b % a == 0

/-- Represents a valid move in the game -/
def valid_move (state : GameState) (move : Nat) : Prop :=
  move > 0 ∧ move < state.current ∧ is_factor move state.current

/-- Applies a move to the current game state -/
def apply_move (state : GameState) (move : Nat) : GameState :=
  { current := state.current - move, player_turn := ¬state.player_turn }

/-- Defines a winning strategy for a player -/
def winning_strategy (player : Bool) (depth : Nat) : Prop :=
  ∀ (state : GameState),
    state.player_turn = player →
    depth > 0 →
    ∃ (move : Nat), valid_move state move ∧
      ∀ (opponent_move : Nat),
        valid_move (apply_move state move) opponent_move →
        winning_strategy player (depth - 1)

/-- The main theorem stating that Player B has a winning strategy -/
theorem player_b_wins :
  ∃ (depth : Nat), winning_strategy false depth :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_wins_l753_75339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geese_percentage_among_non_swans_is_fifty_percent_l753_75307

/-- Represents the percentage of birds of each type in the park -/
structure BirdPercentages where
  geese : ℚ
  swans : ℚ
  herons : ℚ
  ducks : ℚ

/-- Calculates the percentage of geese among non-swan birds -/
noncomputable def geesePercentageAmongNonSwans (bp : BirdPercentages) : ℚ :=
  (bp.geese / (100 - bp.swans)) * 100

/-- Theorem stating that the percentage of geese among non-swan birds is 50% -/
theorem geese_percentage_among_non_swans_is_fifty_percent 
  (bp : BirdPercentages) 
  (h1 : bp.geese = 40)
  (h2 : bp.swans = 20)
  (h3 : bp.herons = 15)
  (h4 : bp.ducks = 25)
  (h5 : bp.geese + bp.swans + bp.herons + bp.ducks = 100) :
  geesePercentageAmongNonSwans bp = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geese_percentage_among_non_swans_is_fifty_percent_l753_75307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l753_75360

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the line
def line (x : ℝ) : ℝ := x - 2

-- Theorem statement
theorem min_distance_curve_to_line :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  ∀ (x y : ℝ), x > 0 → y = curve x →
    ∀ (x' : ℝ), (y - line x')^2 + (x - x')^2 ≥ d^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l753_75360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l753_75308

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Main theorem
theorem geometric_sequence_properties (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (is_geometric_sequence (λ n ↦ 2 * a (3 * n - 1))) ∧
  (is_geometric_sequence (λ n ↦ a n * a (n + 1))) ∧
  (∃ b : ℕ → ℝ, is_geometric_sequence b ∧ ¬ is_geometric_sequence (λ n ↦ b n + b (n + 1))) ∧
  (∃ c : ℕ → ℝ, is_geometric_sequence c ∧ ¬ is_geometric_sequence (λ n ↦ Real.log (abs (c n)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l753_75308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_division_ratio_l753_75319

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a 2D plane -/
def Point := ℝ × ℝ

/-- Represents the ratio between two parts -/
structure Ratio where
  numerator : ℝ
  denominator : ℝ

/-- Checks if a point is on a line segment between two points -/
def on_line_segment (A B P : Point) : Prop := sorry

/-- Checks if a point is on a circle -/
def on_circle (P : Point) (C : Circle) : Prop := sorry

/-- Checks if a line is tangent to a circle at a given point -/
def is_tangent (C : Circle) (A P : Point) : Prop := sorry

/-- Checks if a line divides a circle in a given ratio -/
def divides_circle (C : Circle) (R : Ratio) (A : Point) : Prop := sorry

/-- Given two circles where the first passes through the center of the second and intersects it at two points,
    and a tangent to the first circle at one of these intersection points divides the second circle in a given ratio,
    prove that the second circle divides the first in a specific ratio -/
theorem circle_division_ratio 
  (c1 c2 : Circle) 
  (A B : Point) 
  (m n : ℝ) 
  (h1 : c1.center ≠ c2.center)
  (h2 : A ≠ B)
  (h3 : m < n)
  (h4 : on_line_segment c1.center c2.center A)
  (h5 : on_line_segment c1.center c2.center B)
  (h6 : on_circle A c1)
  (h7 : on_circle A c2)
  (h8 : on_circle B c1)
  (h9 : on_circle B c2)
  (h10 : on_circle c2.center c1)
  (h11 : ∃ (P : Point), on_circle P c2 ∧ is_tangent c1 A P)
  (h12 : divides_circle c2 (Ratio.mk m n) A)
  : divides_circle c1 (Ratio.mk (n - m) (2 * m)) c2.center :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_division_ratio_l753_75319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l753_75391

-- Define the participants
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

-- Define the places
inductive Place
| First
| Second
| Third
| Fourth

-- Define a function to represent the placement of participants
def placement : Participant → Place := sorry

-- Define a predicate for a name starting with 'O'
def startsWithO (p : Participant) : Prop :=
  p = Participant.Olya ∨ p = Participant.Oleg

-- Define a predicate for odd-numbered places
def isOddPlace (pl : Place) : Prop :=
  pl = Place.First ∨ pl = Place.Third

-- Define a predicate for consecutive places
def areConsecutive (pl1 pl2 : Place) : Prop :=
  (pl1 = Place.First ∧ pl2 = Place.Second) ∨
  (pl1 = Place.Second ∧ pl2 = Place.Third) ∨
  (pl1 = Place.Third ∧ pl2 = Place.Fourth) ∨
  (pl2 = Place.First ∧ pl1 = Place.Second) ∨
  (pl2 = Place.Second ∧ pl1 = Place.Third) ∨
  (pl2 = Place.Third ∧ pl1 = Place.Fourth)

-- Define a function to get the participant in a given place
def participantInPlace (pl : Place) : Participant := sorry

-- Define the theorem
theorem competition_result :
  (∃! p : Participant, placement p = Place.First) ∧
  (∀ p : Participant, p ≠ Participant.Polya →
    (placement p = Place.First ∨
    (p = Participant.Olya → ∀ pl : Place, isOddPlace pl → startsWithO (participantInPlace pl)) ∨
    (p = Participant.Oleg → areConsecutive (placement Participant.Oleg) (placement Participant.Olya)) ∨
    (p = Participant.Pasha → ∀ pl : Place, isOddPlace pl → startsWithO (participantInPlace pl)))) →
  (placement Participant.Oleg = Place.First ∧
   placement Participant.Olya = Place.Second ∧
   placement Participant.Polya = Place.Third ∧
   placement Participant.Pasha = Place.Fourth) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l753_75391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l753_75362

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x - 1/x

-- State the theorem
theorem min_value_f_on_interval :
  ∃ (x₀ : ℝ), x₀ ∈ Set.Icc 1 2 ∧ 
  (∀ (x : ℝ), x ∈ Set.Icc 1 2 → f x₀ ≤ f x) ∧
  f x₀ = 0 := by
  -- Prove that x₀ = 1 is the minimum point
  use 1
  constructor
  · -- Show that 1 is in the interval [1, 2]
    simp [Set.Icc]
  constructor
  · -- Show that f(1) ≤ f(x) for all x in [1, 2]
    intro x hx
    sorry -- Proof omitted
  · -- Show that f(1) = 0
    simp [f]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l753_75362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_harmonic_mean_l753_75330

noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

theorem closest_integer_to_harmonic_mean :
  ∀ n : ℤ, n ≠ 8 → |harmonic_mean 4 5040 - 8| < |harmonic_mean 4 5040 - ↑n| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_harmonic_mean_l753_75330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_inequality_l753_75389

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2*x*y*z = 1) :
  8*x*y*z ≤ 1 ∧
  (8*x*y*z = 1 ↔ (x, y, z) = (1/2, 1/2, 1/2) ∨ 
                 (x, y, z) = (-1/2, -1/2, 1/2) ∨ 
                 (x, y, z) = (-1/2, 1/2, -1/2) ∨ 
                 (x, y, z) = (1/2, -1/2, -1/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_inequality_l753_75389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lake_pleasant_activities_l753_75375

theorem lake_pleasant_activities (total_kids : ℕ) (tubing_ratio : ℚ) (rafting_ratio : ℚ) (kayaking_ratio : ℚ) : 
  total_kids = 40 →
  tubing_ratio = 1/4 →
  rafting_ratio = 1/2 →
  kayaking_ratio = 1/3 →
  ⌊(total_kids : ℚ) * tubing_ratio * rafting_ratio * kayaking_ratio⌋ = 1 := by
  intro h_total h_tubing h_rafting h_kayaking
  -- Here we would normally prove the theorem
  sorry

#check lake_pleasant_activities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lake_pleasant_activities_l753_75375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_root_difference_l753_75345

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a parabola's vertex -/
noncomputable def Parabola.vertex_x (p : Parabola) : ℝ := -p.b / (2 * p.a)

/-- The y-coordinate of a parabola's vertex -/
noncomputable def Parabola.vertex_y (p : Parabola) : ℝ := p.c - p.b^2 / (4 * p.a)

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains_point (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The discriminant of the quadratic equation ax^2 + bx + c = 0 -/
noncomputable def Parabola.discriminant (p : Parabola) : ℝ := p.b^2 - 4 * p.a * p.c

/-- The larger root of the quadratic equation ax^2 + bx + c = 0 -/
noncomputable def Parabola.larger_root (p : Parabola) : ℝ := 
  (-p.b + Real.sqrt (p.discriminant)) / (2 * p.a)

/-- The smaller root of the quadratic equation ax^2 + bx + c = 0 -/
noncomputable def Parabola.smaller_root (p : Parabola) : ℝ := 
  (-p.b - Real.sqrt (p.discriminant)) / (2 * p.a)

theorem parabola_root_difference (p : Parabola) 
  (h1 : p.vertex_x = 3 ∧ p.vertex_y = -9)
  (h2 : p.contains_point 5 6)
  (h3 : p.a > 0) :
  p.larger_root - p.smaller_root = 4 * Real.sqrt 15 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_root_difference_l753_75345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_and_lambda_range_l753_75332

noncomputable section

variables (a b x₁ x₂ : ℝ) (lambda : ℝ)

def f (x : ℝ) := a * Real.log (x^2 + 1) + b * x
def g (x : ℝ) := b * x^2 + 2 * a * x + b

theorem roots_sum_and_lambda_range 
  (ha : a > 0) 
  (hb : b > 0) 
  (hx : x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0) 
  (hg : g a b x₁ = 0 ∧ g a b x₂ = 0) 
  (hlambda : f a b x₁ + f a b x₂ + 3 * a - lambda * b = 0) :
  (x₁ + x₂ < -2) ∧ (lambda > 2 * Real.log 2 + 1) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_and_lambda_range_l753_75332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_values_l753_75335

-- Define the set of possible values for α
def alpha_set : Set ℝ := {x | -4 * Real.pi < x ∧ x < -2 * Real.pi}

-- Define the symmetry condition
def is_symmetric (α : ℝ) : Prop :=
  ∃ k : ℤ, α = Real.pi / 3 + 2 * k * Real.pi

-- Theorem statement
theorem alpha_values (α : ℝ) (h1 : α ∈ alpha_set) (h2 : is_symmetric α) :
  α = -11 * Real.pi / 3 ∨ α = -5 * Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_values_l753_75335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ginger_limeade_calories_l753_75351

/-- Represents the recipe and calorie information for ginger-limeade --/
structure GingerLimeade :=
  (lime_juice : ℚ)
  (honey : ℚ)
  (water : ℚ)
  (ginger : ℚ)
  (lime_juice_calories : ℚ)
  (honey_calories : ℚ)
  (ginger_calories : ℚ)

/-- Calculates the total weight of the ginger-limeade --/
def total_weight (recipe : GingerLimeade) : ℚ :=
  recipe.lime_juice + recipe.honey + recipe.water + recipe.ginger

/-- Calculates the total calories in the ginger-limeade --/
def total_calories (recipe : GingerLimeade) : ℚ :=
  (recipe.lime_juice / 100) * recipe.lime_juice_calories +
  (recipe.honey / 100) * recipe.honey_calories +
  recipe.ginger_calories

/-- Theorem: The number of calories in 300g of Lara's ginger-limeade is approximately 159 --/
theorem ginger_limeade_calories (recipe : GingerLimeade)
  (h1 : recipe.lime_juice = 150)
  (h2 : recipe.honey = 120)
  (h3 : recipe.water = 450)
  (h4 : recipe.ginger = 30)
  (h5 : recipe.lime_juice_calories = 20)
  (h6 : recipe.honey_calories = 304)
  (h7 : recipe.ginger_calories = 2) :
  ∃ (calories : ℚ), (calories ≥ 158.5 ∧ calories ≤ 159.5) ∧ 
    calories = (300 / total_weight recipe) * total_calories recipe := by
  sorry

#eval (300 : ℚ) / 750 * ((150 / 100 * 20) + (120 / 100 * 304) + 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ginger_limeade_calories_l753_75351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_rate_l753_75348

/-- Given an investment scenario, prove the unknown interest rate -/
theorem investment_interest_rate 
  (total_investment : ℝ) 
  (investment_at_five_percent : ℝ) 
  (total_with_interest : ℝ) 
  (h1 : total_investment = 1000)
  (h2 : investment_at_five_percent = 600)
  (h3 : total_with_interest = 1054) :
  (total_with_interest - total_investment - investment_at_five_percent * 0.05) / 
  (total_investment - investment_at_five_percent) = 0.06 := by
  -- Define the intermediate calculations
  let remaining_investment := total_investment - investment_at_five_percent
  let total_interest := total_with_interest - total_investment
  let interest_from_five_percent := investment_at_five_percent * 0.05
  let remaining_interest := total_interest - interest_from_five_percent
  
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_rate_l753_75348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_weight_proof_l753_75344

/-- Leo's current weight in pounds -/
noncomputable def leo_weight : ℝ := 61.22

/-- Kendra's current weight in pounds -/
noncomputable def kendra_weight : ℝ := (leo_weight + 10) / 1.5

/-- Jeremy's current weight in pounds -/
noncomputable def jeremy_weight : ℝ := 1.3 * (leo_weight + kendra_weight)

/-- The total weight of Leo, Kendra, and Jeremy in pounds -/
noncomputable def total_weight : ℝ := leo_weight + kendra_weight + jeremy_weight

theorem leo_weight_proof :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (jeremy_weight = 1.3 * (leo_weight + kendra_weight)) ∧
  (total_weight = 250) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_weight_proof_l753_75344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l753_75343

noncomputable section

/-- Ellipse C with equation x²/4 + y²/2 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

/-- Right focus of ellipse C -/
def focus_F : ℝ × ℝ := (Real.sqrt 2, 0)

/-- Line l intersecting ellipse C -/
def line_l (k m : ℝ) (x : ℝ) : ℝ := k * x + m

/-- Midpoint of chord AB lies on x + 2y = 0 -/
def midpoint_condition (x y : ℝ) : Prop := x + 2*y = 0

/-- Area of triangle FAB -/
noncomputable def triangle_area (A B : ℝ × ℝ) : ℝ :=
  let d := abs (A.2 - B.2) / Real.sqrt (1 + ((A.2 - B.2)/(A.1 - B.1))^2)
  abs ((A.1 - focus_F.1) * (B.2 - focus_F.2) - (B.1 - focus_F.1) * (A.2 - focus_F.2)) / 2

theorem max_triangle_area :
  ∀ k m A B,
    k ≠ 0 → m ≠ 0 →
    ellipse_C A.1 A.2 →
    ellipse_C B.1 B.2 →
    A.2 = line_l k m A.1 →
    B.2 = line_l k m B.1 →
    midpoint_condition ((A.1 + B.1)/2) ((A.2 + B.2)/2) →
    triangle_area A B ≤ 8/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l753_75343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_range_l753_75301

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := lg ((1 + x) / (1 - x))

-- Theorem statement
theorem f_greater_than_one_range :
  {x : ℝ | f x > 1} = Set.Ioo (9/11 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_range_l753_75301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l753_75346

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def sum_arithmetic (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

noncomputable def b (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a (n + 1) / a n + a n / a (n + 1) - 2

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_10 : sum_arithmetic a 10 = 110)
  (h_sum_15 : sum_arithmetic a 15 = 240) :
  (∀ n : ℕ, a n = 2 * n) ∧
  (∀ n : ℕ, (Finset.range n).sum (b a) = (2 * n^2 + 3 * n) / (n + 1)) :=
by sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l753_75346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_down_l753_75393

/-- Calculates the speed going down given the speed going up and the average speed for a round trip. -/
noncomputable def speed_down (speed_up : ℝ) (average_speed : ℝ) : ℝ :=
  (2 * speed_up * average_speed) / (2 * speed_up - average_speed)

/-- Theorem stating that for a round trip with speed up 32 km/hr and average speed 38.4 km/hr, the speed down is 48 km/hr. -/
theorem round_trip_speed_down :
  speed_down 32 38.4 = 48 := by
  -- Unfold the definition of speed_down
  unfold speed_down
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_down_l753_75393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_y_axis_l753_75340

/-- A line passes through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (L : Line) : ℝ :=
  L.y₁ + (L.y₂ - L.y₁) / (L.x₂ - L.x₁) * (-L.x₁)

/-- The theorem stating that the line passing through (3, 20) and (-7, -2)
    intersects the y-axis at (0, 13.4) -/
theorem line_intersects_y_axis :
  let L : Line := { x₁ := 3, y₁ := 20, x₂ := -7, y₂ := -2 }
  y_intercept L = 13.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_y_axis_l753_75340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_local_min_and_max_l753_75381

/-- The function g(x) defined on positive real numbers -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 - Real.log x

/-- The derivative of g(x) -/
noncomputable def g_derivative (a : ℝ) (x : ℝ) : ℝ := (2 * a * x^2 - 2 * a * x - 1) / x

/-- Theorem stating the condition for g(x) to have both a local minimum and a local maximum -/
theorem g_has_local_min_and_max (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
   g_derivative a x₁ = 0 ∧ g_derivative a x₂ = 0) ↔ 
  a < -2 := by
  sorry

#check g_has_local_min_and_max

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_local_min_and_max_l753_75381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramon_enchiladas_tacos_pricing_l753_75349

/-- Given Ramon's pricing for enchiladas and tacos, prove the cost of 4 enchiladas and 5 tacos -/
theorem ramon_enchiladas_tacos_pricing (e t : ℚ) : 
  (4 * e + 5 * t = 4) →  -- First pricing condition
  (5 * e + 4 * t = 44/10) →  -- Second pricing condition
  (4 * e + 5 * t = 4) :=  -- Conclusion: cost of 4 enchiladas and 5 tacos
by
  intros h1 h2
  exact h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramon_enchiladas_tacos_pricing_l753_75349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l753_75347

theorem solve_exponential_equation (n : ℝ) : 
  (12 : ℝ)^(4*n) = (1/12 : ℝ)^(n - 30) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l753_75347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_segment_length_l753_75396

noncomputable section

theorem circle_segment_length (T : ℝ → ℝ → Prop) (X Y Z : ℝ × ℝ) :
  -- Circle T has radius 9
  (∀ p : ℝ × ℝ, T p.1 p.2 ↔ (p.1 - X.1)^2 + (p.2 - X.2)^2 = 81) →
  -- XY is a diameter
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = 324 →
  -- Angle TXZ measures 30°
  Real.arccos ((Z.1 - X.1) / 9) = π / 6 →
  -- Then the length of segment XZ is 9√(2 - √3)
  (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 81 * (2 - Real.sqrt 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_segment_length_l753_75396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_l753_75355

noncomputable def circle1_center : ℝ × ℝ := (3, -2)
noncomputable def circle1_radius : ℝ := 5

noncomputable def circle2_center : ℝ × ℝ := (3, 4)
noncomputable def circle2_radius : ℝ := Real.sqrt 17

theorem intersection_distance_squared :
  ∃ A B : ℝ × ℝ,
    (A.1 - circle1_center.1)^2 + (A.2 - circle1_center.2)^2 = circle1_radius^2 ∧
    (A.1 - circle2_center.1)^2 + (A.2 - circle2_center.2)^2 = circle2_radius^2 ∧
    (B.1 - circle1_center.1)^2 + (B.2 - circle1_center.2)^2 = circle1_radius^2 ∧
    (B.1 - circle2_center.1)^2 + (B.2 - circle2_center.2)^2 = circle2_radius^2 ∧
    A ≠ B ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 416 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_l753_75355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_implies_a_eq_two_l753_75342

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number in question -/
noncomputable def z (a : ℝ) : ℂ := (2 - a * i) / (1 + i)

/-- A complex number is pure imaginary if its real part is zero and imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Theorem: If z(a) is pure imaginary, then a = 2 -/
theorem pure_imaginary_implies_a_eq_two :
  ∀ a : ℝ, is_pure_imaginary (z a) → a = 2 := by
  sorry

#check pure_imaginary_implies_a_eq_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_implies_a_eq_two_l753_75342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_station_probability_l753_75361

-- Define the time range in minutes (2:00 PM to 4:00 PM)
noncomputable def time_range : ℝ := 120

-- Define the train's waiting time in minutes
noncomputable def train_wait_time : ℝ := 30

-- Define the probability of Maria finding the train at the station
noncomputable def probability_maria_finds_train : ℝ := 7 / 32

theorem train_station_probability :
  probability_maria_finds_train = 
    (1 / 2 * (time_range + (time_range - train_wait_time)) * train_wait_time) / (time_range * time_range) :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_station_probability_l753_75361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_circle_l753_75386

theorem max_value_on_circle (x y : ℝ) : 
  x^2 + y^2 = 25 → 
  (∀ a b : ℝ, a^2 + b^2 = 25 → 
    Real.sqrt (8*y - 6*x + 50) + Real.sqrt (8*y + 6*x + 50) ≥ 
    Real.sqrt (8*b - 6*a + 50) + Real.sqrt (8*b + 6*a + 50)) →
  Real.sqrt (8*y - 6*x + 50) + Real.sqrt (8*y + 6*x + 50) = 6 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_circle_l753_75386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_even_difference_l753_75337

/-- A circular arrangement of integers from 1 to 2010 -/
def CircularArrangement := Fin 2010 → ℕ

/-- Property that the arrangement alternates between increasing and decreasing -/
def AlternatingProperty (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 2010, 
    (arr i < arr (i + 1) ∧ arr (i + 1) > arr (i + 2)) ∨
    (arr i > arr (i + 1) ∧ arr (i + 1) < arr (i + 2))

/-- Theorem stating that in any circular arrangement with the alternating property,
    there exists a pair of adjacent numbers with even difference -/
theorem exists_even_difference (arr : CircularArrangement) 
  (h : AlternatingProperty arr) :
  ∃ i : Fin 2010, Even (arr i - arr (i + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_even_difference_l753_75337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l753_75314

open Real

theorem min_value_of_a (a : ℝ) (h_a : a > 1) :
  (∀ x : ℝ, x ≥ 1/3 → 1/(3*x) - x + log (3*x) ≤ 1/(a*(exp x)) + log a) →
  a ≥ 3/exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l753_75314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_constant_l753_75350

noncomputable def f (c : ℝ) (x : ℝ) := Real.arctan ((2 - 2*x) / (1 + 4*x)) + c

def is_odd (f : ℝ → ℝ) (I : Set ℝ) :=
  ∀ x ∈ I, f x = -f (-x)

theorem odd_function_constant (c : ℝ) :
  is_odd (f c) (Set.Ioo (-1/4) (1/4)) ↔ c = -Real.arctan 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_constant_l753_75350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l753_75306

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 1)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (x + 1) - 2

-- Theorem statement
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l753_75306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equivalence_line_equivalence_l753_75321

-- Define the parametric equations for the ellipse
noncomputable def ellipse_param (φ : Real) : Real × Real := (5 * Real.cos φ, 4 * Real.sin φ)

-- Define the standard form equation for the ellipse
def ellipse_standard (x y : Real) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the parametric equations for the line
def line_param (t : Real) : Real × Real := (1 - 3 * t, 4 * t)

-- Define the standard form equation for the line
def line_standard (x y : Real) : Prop := 4 * x + 3 * y - 4 = 0

-- Theorem for the ellipse
theorem ellipse_equivalence :
  ∀ φ x y, ellipse_param φ = (x, y) → ellipse_standard x y := by
  sorry

-- Theorem for the line
theorem line_equivalence :
  ∀ t x y, line_param t = (x, y) → line_standard x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equivalence_line_equivalence_l753_75321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_balance_payment_l753_75310

/-- The amount Quinn must pay Parker to balance their shares equally -/
noncomputable def balance_payment (P Q : ℝ) : ℝ := (P - Q) / 2

/-- Theorem stating the correct balance payment amount -/
theorem correct_balance_payment (P Q : ℝ) (h : P > Q) : 
  balance_payment P Q = (P - Q) / 2 ∧ 
  P + Q - 2 * balance_payment P Q = Q + balance_payment P Q := by
  sorry

#check correct_balance_payment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_balance_payment_l753_75310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l753_75317

noncomputable def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then x^2 - 2
  else if 0 < x ∧ x ≤ 3 then -x + 2
  else 0  -- undefined outside [-3, 3]

theorem abs_g_piecewise (x : ℝ) :
  (x ≥ -3 ∧ x ≤ 3) →
  |g x| = if -3 ≤ x ∧ x ≤ -Real.sqrt 2 then x^2 - 2
          else if -Real.sqrt 2 < x ∧ x ≤ Real.sqrt 2 then 2 - x^2
          else if Real.sqrt 2 < x ∧ x ≤ 2 then 2 - x
          else if 2 < x ∧ x ≤ 3 then x - 2
          else 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l753_75317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_for_given_point_l753_75385

noncomputable def terminal_side_point : ℝ × ℝ := (-1, 2)

noncomputable def cos_alpha : ℝ := 
  -terminal_side_point.1 / Real.sqrt (terminal_side_point.1^2 + terminal_side_point.2^2)

theorem cos_double_angle_for_given_point :
  Real.cos (2 * Real.arccos cos_alpha) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_for_given_point_l753_75385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l753_75303

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) / (x - 3)

-- Define the domain
def domain : Set ℝ := {x | x ≥ 1 ∧ x ≠ 3}

-- Theorem stating that the domain of f is correct
theorem f_domain : 
  ∀ x : ℝ, f x ∈ Set.range f ↔ x ∈ domain :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l753_75303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_ceiling_fraction_evaluation_l753_75354

theorem complex_ceiling_fraction_evaluation : 
  (⌈(19 : ℚ) / 9 - ⌈(35 : ℚ) / 19⌉⌉) / (⌈(35 : ℚ) / 9 - ⌈9 * (19 : ℚ) / 35⌉⌉) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_ceiling_fraction_evaluation_l753_75354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_increasing_interval_in_domain_l753_75373

open Real Set

-- Define the function
noncomputable def f (x : ℝ) := 2 * sin (π / 6 - 2 * x)

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x₁ x₂, x₁ ∈ Icc (π / 3) (5 * π / 6) → x₂ ∈ Icc (π / 3) (5 * π / 6) →
  x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Define the domain
def domain : Set ℝ := Icc 0 π

-- State that the increasing interval is within the domain
theorem increasing_interval_in_domain :
  Icc (π / 3) (5 * π / 6) ⊆ domain := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_increasing_interval_in_domain_l753_75373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_neg_two_implies_expression_equals_two_fifths_l753_75372

theorem tan_neg_two_implies_expression_equals_two_fifths (θ : ℝ) :
  Real.tan θ = -2 → (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_neg_two_implies_expression_equals_two_fifths_l753_75372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l753_75313

/-- A right triangular prism with equal edge lengths of 1 -/
structure RightTriangularPrism where
  edge_length : ℝ
  is_unit : edge_length = 1

/-- The surface area of a sphere -/
noncomputable def sphere_surface_area (radius : ℝ) : ℝ := 4 * Real.pi * radius^2

/-- Theorem: The surface area of the circumscribed sphere of a right triangular prism
    with equal edge lengths of 1 is 7π/3 -/
theorem circumscribed_sphere_surface_area (prism : RightTriangularPrism) :
  ∃ (radius : ℝ), sphere_surface_area radius = 7 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l753_75313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l753_75334

/-- The average speed of a train given two trips -/
noncomputable def average_speed (d1 d2 t1 t2 : ℝ) : ℝ := (d1 + d2) / (t1 + t2)

theorem train_average_speed :
  let d1 : ℝ := 210 -- distance of first trip in km
  let d2 : ℝ := 270 -- distance of second trip in km
  let t1 : ℝ := 3   -- time of first trip in hours
  let t2 : ℝ := 4   -- time of second trip in hours
  average_speed d1 d2 t1 t2 = (d1 + d2) / (t1 + t2) :=
by
  sorry

-- Use Float for evaluation
def average_speed_float (d1 d2 t1 t2 : Float) : Float := (d1 + d2) / (t1 + t2)

#eval average_speed_float 210 270 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l753_75334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_alpha_l753_75379

theorem sin_double_alpha (α : ℝ) (h : Real.cos (α - π/4) = Real.sqrt 2/4) : 
  Real.sin (2*α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_alpha_l753_75379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l753_75387

noncomputable def f (x : ℝ) := (x^2 - 6*x - 3) / (x + 1)

theorem min_value_of_f :
  ∀ x ∈ Set.Icc 0 1, f x ≥ -4 ∧ ∃ x₀ ∈ Set.Icc 0 1, f x₀ = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l753_75387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_l753_75380

noncomputable section

def F₁ : ℝ × ℝ := (1, 2)
def F₂ : ℝ × ℝ := (5, 2)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def is_on_ellipse (p : ℝ × ℝ) : Prop :=
  distance p F₁ + distance p F₂ = 10

noncomputable def center : ℝ × ℝ := ((F₁.1 + F₂.1) / 2, (F₁.2 + F₂.2) / 2)

noncomputable def h : ℝ := center.1
noncomputable def k : ℝ := center.2

noncomputable def c : ℝ := distance F₁ F₂ / 2
noncomputable def a : ℝ := 5
noncomputable def b : ℝ := Real.sqrt (a^2 - c^2)

theorem ellipse_sum :
  h + k + a + b = 10 + Real.sqrt 21 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_l753_75380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_circle_and_rectangle_l753_75331

-- Define the points R and S
def R : ℝ × ℝ := (-2, 5)
def S : ℝ × ℝ := (7, -6)

-- Define the circle
def circle_center : ℝ × ℝ := R

noncomputable def circle_radius : ℝ := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)

-- Define the rectangle
def rectangle_vertex1 : ℝ × ℝ := R
def rectangle_vertex2 : ℝ × ℝ := S

-- Theorem statement
theorem total_area_circle_and_rectangle :
  let circle_area := π * circle_radius^2
  let rectangle_area := |R.1 - S.1| * |R.2 - S.2|
  circle_area + rectangle_area = 202 * π + 99 := by
  sorry

-- Additional lemma to show the value of circle_radius
lemma circle_radius_value : circle_radius = Real.sqrt 202 := by
  unfold circle_radius
  simp [R, S]
  ring_nf

-- Lemma for rectangle area
lemma rectangle_area_value :
  |R.1 - S.1| * |R.2 - S.2| = 99 := by
  simp [R, S]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_circle_and_rectangle_l753_75331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_prime_power_l753_75333

theorem power_sum_prime_power (a b n : ℕ) (p : ℕ) (hp : Nat.Prime p) :
  a ^ 2013 + b ^ 2013 = p ^ n →
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ n = 2013 * k + 1 ∧ p = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_prime_power_l753_75333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l753_75371

theorem triangle_existence (segments : Finset ℕ) : 
  segments.card = 10 ∧ 
  (∃ a b : ℕ, a ∈ segments ∧ b ∈ segments ∧ a = 1 ∧ b = 1 ∧ (∀ x ∈ segments, x ≥ 1)) ∧
  (∃ max : ℕ, max ∈ segments ∧ max = 50 ∧ (∀ x ∈ segments, x ≤ 50)) →
  ∃ x y z : ℕ, x ∈ segments ∧ y ∈ segments ∧ z ∈ segments ∧ 
    x + y > z ∧ y + z > x ∧ z + x > y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l753_75371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_problem_l753_75338

/-- Represents the problem of determining the initial number of watermelons --/
theorem watermelon_problem (base_price min_discount max_discount profit : ℝ) (remaining : ℕ) :
  base_price = 3 →
  min_discount = 0 →
  max_discount = 0.5 →
  profit = 105 →
  remaining = 18 →
  ∃ (initial : ℕ), initial = 64 ∧
    ∃ (discounts : Fin (initial - remaining) → ℝ),
      (∀ i j, i ≠ j → discounts i ≠ discounts j) ∧
      (∀ i, min_discount ≤ discounts i ∧ discounts i ≤ max_discount) ∧
      profit = (Finset.univ.sum fun i => base_price * (1 - discounts i)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_problem_l753_75338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l753_75326

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  m : ℕ+     -- Positive natural number
  n : ℕ+     -- Positive natural number
  is_arithmetic : ∀ k, a (k + 2) - a (k + 1) = a (k + 1) - a k
  a_m_eq_n : a m = n
  a_n_eq_m : a n = m

/-- The theorem stating that a_{m+n} = 0 for the given arithmetic sequence -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  seq.a (seq.m + seq.n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l753_75326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_seven_l753_75364

theorem opposite_of_negative_seven :
  -(- 7) = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_seven_l753_75364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_lambda_range_l753_75302

/-- Parabola E: x^2 = 2py (p > 0) -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line on a 2D plane -/
structure Line where
  k : ℝ
  m : ℝ

def is_on_parabola (E : Parabola) (P : Point) : Prop :=
  P.x^2 = 2 * E.p * P.y

def is_tangent_to_parabola (E : Parabola) (L : Line) (P : Point) : Prop :=
  is_on_parabola E P ∧ ∀ Q : Point, Q ≠ P → is_on_parabola E Q → ¬(Q.y = L.k * Q.x + L.m)

def is_on_line (L : Line) (P : Point) : Prop :=
  P.y = L.k * P.x + L.m

def is_tangent_to_circle (L : Line) : Prop :=
  ∃ P : Point, (P.x^2 + (P.y - 1)^2 = 1) ∧ is_on_line L P ∧
  ∀ Q : Point, Q ≠ P → (Q.x^2 + (Q.y - 1)^2 = 1) → ¬(is_on_line L Q)

theorem parabola_and_lambda_range (E : Parabola) (M A B P Q C : Point) (l : Line) :
  is_on_parabola E A ∧ is_on_parabola E B ∧
  is_tangent_to_parabola E (Line.mk (A.x / E.p) (A.y - A.x^2 / (2 * E.p))) A ∧
  is_tangent_to_parabola E (Line.mk (B.x / E.p) (B.y - B.x^2 / (2 * E.p))) B ∧
  M.x = 1 ∧ M.y = -1 ∧
  is_on_line (Line.mk (A.x / E.p) (A.y - A.x^2 / (2 * E.p))) M ∧
  is_on_line (Line.mk (B.x / E.p) (B.y - B.x^2 / (2 * E.p))) M ∧
  (B.y - A.y) / (B.x - A.x) = 1/2 ∧
  2 < l.m ∧ l.m ≤ 4 ∧
  is_tangent_to_circle l ∧
  is_on_parabola E P ∧ is_on_parabola E Q ∧
  is_on_line l P ∧ is_on_line l Q ∧
  is_on_parabola E C ∧
  (∃ lambda : ℝ, lambda > 0 ∧ C.x = lambda * (P.x + Q.x) ∧ C.y = lambda * (P.y + Q.y)) →
  (E.p = 2 ∧ ∀ lambda : ℝ, (∃ C : Point, is_on_parabola E C ∧ C.x = lambda * (P.x + Q.x) ∧ C.y = lambda * (P.y + Q.y)) → lambda ≥ 5/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_lambda_range_l753_75302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l753_75378

-- Define the universal set U
def U : Set ℝ := {x | x ≥ -2}

-- Define set A
def A : Set ℝ := {x | 2 < x ∧ x < 10}

-- Define set B
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}

-- Theorem for the four parts of the problem
theorem set_operations :
  (U \ A = {x | -2 ≤ x ∧ x ≤ 2 ∨ x ≥ 10}) ∧
  ((U \ A) ∩ B = {2}) ∧
  (A ∩ B = {x | 2 < x ∧ x ≤ 8}) ∧
  (U \ (A ∩ B) = {x | -2 ≤ x ∧ x ≤ 2 ∨ x > 8}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l753_75378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derangement_probability_three_l753_75316

/-- The number of derangements for n items -/
def derangements' (n : ℕ) : ℕ := sorry

/-- The probability of a derangement for n items -/
def derangement_probability (n : ℕ) : ℚ :=
  (derangements' n : ℚ) / (n.factorial : ℚ)

/-- Theorem: The probability of a derangement for 3 items is 1/3 -/
theorem derangement_probability_three : derangement_probability 3 = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derangement_probability_three_l753_75316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_to_start_l753_75369

/-- Represents the number of people in the circle -/
def n : ℕ := 13

/-- Represents the number of positions skipped in each throw -/
def skip : ℕ := 5

/-- Function to calculate the next person to receive the ball -/
def next (current : ℕ) : ℕ :=
  (current + skip - 1) % n + 1

/-- Theorem stating that it takes exactly n throws for the ball to return to the start -/
theorem ball_returns_to_start :
  (∃ k : ℕ, k > 0 ∧ (Nat.iterate next 1 k) = 1) ∧
  (∀ m : ℕ, 0 < m → m < n → (Nat.iterate next 1 m) ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_to_start_l753_75369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_months_is_seven_l753_75398

/-- Represents the rent sharing problem for a pasture -/
structure PastureRent where
  total_rent : ℝ
  a_oxen : ℕ
  b_oxen : ℕ
  c_oxen : ℕ
  b_months : ℕ
  c_months : ℕ
  c_share : ℝ

/-- Calculates the number of months A put his oxen for grazing -/
noncomputable def calculate_a_months (p : PastureRent) : ℝ :=
  ((p.total_rent * (p.c_oxen * p.c_months : ℝ)) / p.c_share - 
   (p.b_oxen * p.b_months : ℝ) - (p.c_oxen * p.c_months : ℝ)) / (p.a_oxen : ℝ)

/-- Theorem stating that A put his oxen for 7 months -/
theorem a_months_is_seven (p : PastureRent) 
  (h1 : p.total_rent = 210)
  (h2 : p.a_oxen = 10)
  (h3 : p.b_oxen = 12)
  (h4 : p.c_oxen = 15)
  (h5 : p.b_months = 5)
  (h6 : p.c_months = 3)
  (h7 : p.c_share = 54) :
  calculate_a_months p = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_months_is_seven_l753_75398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l753_75397

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- The line equation with slope 1 -/
def line (x y t : ℝ) : Prop := y = x + t

/-- The intersection points of the line and ellipse -/
def intersection (A B : ℝ × ℝ) (t : ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line A.1 A.2 t ∧ line B.1 B.2 t

/-- The distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem max_chord_length :
  ∃ (max : ℝ), max = (4 * Real.sqrt 3) / 3 ∧
  (∀ (A B : ℝ × ℝ) (t : ℝ), intersection A B t → distance A B ≤ max) ∧
  (∃ (A B : ℝ × ℝ) (t : ℝ), intersection A B t ∧ distance A B = max) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l753_75397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_trajectory_is_hyperbola_moving_line_segment_trajectory_l753_75325

-- Part 1
noncomputable def quadratic_function (x α : ℝ) : ℝ := 
  x^2 - 2*x*(1/Real.cos α) + (2 + Real.sin (2*α)) / (2*(Real.cos α)^2)

theorem vertex_trajectory_is_hyperbola (α : ℝ) (h : Real.cos α ≠ 0) :
  ∃ x y : ℝ, quadratic_function x α = y ∧ x^2 - y^2 = 1 := by
  sorry

-- Part 2
theorem moving_line_segment_trajectory (a : ℝ) (h : a > 0) :
  ∃ ρ θ : ℝ, ρ = a * Real.sin (2*θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_trajectory_is_hyperbola_moving_line_segment_trajectory_l753_75325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l753_75309

/-- The function f(x) defined as sin(x/4) + sin(x/12) --/
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 12)

/-- Converts degrees to radians --/
noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

/-- Theorem stating that 120 degrees is the smallest positive value where f(x) achieves its maximum --/
theorem smallest_max_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ x = 120 ∧
  (∀ (y : ℝ), y > 0 → f (deg_to_rad y) ≤ f (deg_to_rad x)) ∧
  (∀ (z : ℝ), 0 < z ∧ z < x → f (deg_to_rad z) < f (deg_to_rad x)) :=
by
  -- The proof goes here
  sorry

#check smallest_max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l753_75309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l753_75323

noncomputable def a : ℝ := Real.sqrt 0.6
noncomputable def b : ℝ := Real.sqrt 0.7
noncomputable def c : ℝ := Real.log 0.7 / Real.log 10

theorem order_of_abc : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l753_75323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_range_in_regular_prism_l753_75394

-- Define the properties as axioms instead of definitions
axiom is_regular_prism (n : ℕ) : Prop
axiom is_dihedral_angle (θ : ℝ) : Prop

-- State the theorem
theorem dihedral_angle_range_in_regular_prism (n : ℕ) (θ : ℝ) :
  n ≥ 3 →
  is_regular_prism n →
  is_dihedral_angle θ →
  ((n - 2 : ℝ) / n * π < θ ∧ θ < π) :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_range_in_regular_prism_l753_75394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_change_calculation_l753_75374

/-- Represents the cost of items in dollars -/
def CostInDollars := ℕ

instance : Add CostInDollars := ⟨Nat.add⟩
instance : Sub CostInDollars := ⟨Nat.sub⟩
instance : Mul CostInDollars := ⟨Nat.mul⟩
instance (n : Nat) : OfNat CostInDollars n := ⟨n⟩

/-- Calculates the total cost of the initial order -/
def initial_order_cost (eggs_cost pancakes_cost cocoa_cost : CostInDollars) : CostInDollars :=
  eggs_cost + pancakes_cost + 2 * cocoa_cost

/-- Calculates the total cost including tax -/
def total_cost_with_tax (order_cost tax : CostInDollars) : CostInDollars :=
  order_cost + tax

/-- Calculates the cost of the additional order -/
def additional_order_cost (pancakes_cost cocoa_cost : CostInDollars) : CostInDollars :=
  pancakes_cost + cocoa_cost

/-- Calculates the final total cost -/
def final_total_cost (initial_cost additional_cost : CostInDollars) : CostInDollars :=
  initial_cost + additional_cost

/-- Calculates the change to be received -/
def change_received (amount_paid final_cost : CostInDollars) : CostInDollars :=
  amount_paid - final_cost

theorem correct_change_calculation 
  (eggs_cost pancakes_cost cocoa_cost tax amount_paid : CostInDollars) : 
  change_received amount_paid 
    (final_total_cost 
      (total_cost_with_tax 
        (initial_order_cost eggs_cost pancakes_cost cocoa_cost) 
        tax) 
      (additional_order_cost pancakes_cost cocoa_cost)) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_change_calculation_l753_75374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l753_75329

/-- Represents the cost of items in arrows -/
structure ItemCosts where
  drum : ℕ
  wife : ℕ
  leopard_skin : ℕ

/-- Conditions for the exchange rates -/
def satisfies_conditions (costs : ItemCosts) : Prop :=
  2 * costs.drum + 3 * costs.wife + costs.leopard_skin = 111 ∧
  3 * costs.drum + 4 * costs.wife = 2 * costs.leopard_skin + 8 ∧
  costs.leopard_skin % 2 = 0

/-- The theorem stating the unique solution -/
theorem unique_solution : 
  ∃! costs : ItemCosts, satisfies_conditions costs ∧ 
    costs.drum = 20 ∧ costs.wife = 9 ∧ costs.leopard_skin = 44 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l753_75329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_can_always_win_l753_75358

-- Define the game state
structure GameState where
  n : Nat
  player_turn : Bool  -- true for A's turn, false for B's turn

-- Define player moves
def move_A (state : GameState) (k : Nat) : GameState :=
  { n := state.n ^ k, player_turn := false }

def move_B (state : GameState) (b : Nat) : GameState :=
  { n := state.n - b * b, player_turn := true }

-- Define the game termination condition
def game_over (state : GameState) : Prop :=
  state.n = 0

-- Define the game sequence
def game_sequence (initial_n : Nat) (A_strategy B_strategy : Nat → Nat) : Nat → GameState
  | 0 => { n := initial_n, player_turn := false }
  | i + 1 =>
    let prev := game_sequence initial_n A_strategy B_strategy i
    if prev.player_turn then
      move_A prev (A_strategy i)
    else
      move_B prev (B_strategy i)

-- Theorem statement
theorem B_can_always_win :
  ∀ (initial_n : Nat), initial_n > 0 →
  ∃ (B_strategy : Nat → Nat),
    ∀ (A_strategy : Nat → Nat),
      ∃ (num_moves : Nat),
        game_over (game_sequence initial_n A_strategy B_strategy num_moves) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_can_always_win_l753_75358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_ratio_l753_75388

noncomputable def f (x : ℝ) : ℝ := (3*x - 2) / (2*x + 4)

theorem inverse_function_ratio (a b c d : ℝ) :
  (∀ x, f⁻¹ x = (a*x + b) / (c*x + d)) →
  a / c = -2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_ratio_l753_75388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_l753_75305

-- Define the matrix
def A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![2, -1, 4;
     3, x, -2;
     1, -3, 0]

-- Theorem statement
theorem det_A (x : ℝ) : Matrix.det (A x) = -46 - 4*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_l753_75305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_x_one_to_e_l753_75357

open Real MeasureTheory

theorem integral_reciprocal_x_one_to_e :
  ∫ x in Set.Icc 1 (exp 1), (1 / x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_x_one_to_e_l753_75357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_tangent_theorem_l753_75377

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point P
def P : ℝ × ℝ := (-2, 0)

-- Define a line passing through P
def line_through_P (m : ℝ) (x y : ℝ) : Prop := y = m * (x + 2)

-- Define the intersection points of the line and the circle
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | circle_eq x y ∧ line_through_P m x y}

-- State the theorem
theorem secant_tangent_theorem (m : ℝ) :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧ A ≠ B ∧
  (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)) *
  (Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_tangent_theorem_l753_75377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_abc_is_36_l753_75312

-- Define a triangle with angles in arithmetic progression and side lengths 4, 5, and x
def Triangle (x : ℝ) : Prop :=
  ∃ α d : ℝ, 
    0 < α ∧ 0 ≤ d ∧
    α - d + α + (α + d) = 180 ∧
    (4 = 5 * Real.sin (α - d) / Real.sin α ∨
     4 = x * Real.sin (α - d) / Real.sin (α + d) ∨
     5 = x * Real.sin α / Real.sin (α + d))

-- Define the sum of possible values of x
def SumOfPossibleX (a b c : ℕ) : Prop :=
  ∃ x₁ x₂ : ℝ,
    Triangle x₁ ∧ Triangle x₂ ∧
    x₁ + x₂ = a + Real.sqrt b + Real.sqrt c

-- State the theorem
theorem sum_of_abc_is_36 :
  ∃ a b c : ℕ,
    SumOfPossibleX a b c ∧
    a + b + c = 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_abc_is_36_l753_75312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l753_75390

/-- Calculates the annual interest rate given the initial principal, time period,
    compounding frequency, and total interest earned. -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (compounding_frequency : ℕ) (total_interest : ℝ) : ℝ :=
  let future_value := principal + total_interest
  let n := (compounding_frequency : ℝ) * time
  ((future_value / principal) ^ (1 / n) - 1) * (compounding_frequency : ℝ)

/-- Theorem stating that the calculated interest rate is approximately 3.96%
    given the specified conditions. -/
theorem interest_rate_approximation (ε : ℝ) (hε : ε > 0) :
  let principal := (5000 : ℝ)
  let time := (1.5 : ℝ)
  let compounding_frequency := (2 : ℕ)
  let total_interest := (302.98 : ℝ)
  let calculated_rate := calculate_interest_rate principal time compounding_frequency total_interest
  |calculated_rate - 0.0396| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l753_75390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_with_perimeter_12_l753_75356

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem max_triangle_area_with_perimeter_12 :
  (∃ (a b c : ℕ), a + b + c = 12 ∧
    ∀ (x y z : ℕ), x + y + z = 12 →
      triangle_area (a : ℝ) (b : ℝ) (c : ℝ) ≥ triangle_area (x : ℝ) (y : ℝ) (z : ℝ)) ∧
  (∀ (a b c : ℕ), a + b + c = 12 →
    triangle_area (a : ℝ) (b : ℝ) (c : ℝ) ≤ 4 * Real.sqrt 3) ∧
  (∃ (a b c : ℕ), a + b + c = 12 ∧
    triangle_area (a : ℝ) (b : ℝ) (c : ℝ) = 4 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_with_perimeter_12_l753_75356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_renovation_theorem_l753_75352

/-- Represents a river renovation project -/
structure RiverProject where
  totalLength : ℝ
  teamARate : ℝ
  teamBRate : ℝ

/-- Calculates the time taken for both teams working together -/
noncomputable def timeTogether (project : RiverProject) : ℝ :=
  project.totalLength / (project.teamARate + project.teamBRate)

/-- Calculates the length renovated by Team A when working alone first -/
noncomputable def lengthByTeamA (project : RiverProject) (totalTime : ℝ) : ℝ :=
  (2 * project.teamARate * project.teamBRate * totalTime - project.teamARate * project.totalLength) /
  (project.teamBRate - project.teamARate)

/-- Calculates the length renovated by Team B when Team A works alone first -/
noncomputable def lengthByTeamB (project : RiverProject) (totalTime : ℝ) : ℝ :=
  project.totalLength - lengthByTeamA project totalTime

theorem river_renovation_theorem (project : RiverProject) 
    (h1 : project.totalLength = 2400)
    (h2 : project.teamARate = 30)
    (h3 : project.teamBRate = 50) :
  timeTogether project = 30 ∧ 
  lengthByTeamA project 60 = 900 ∧ 
  lengthByTeamB project 60 = 1500 := by
  sorry

-- Remove #eval statements as they are not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_renovation_theorem_l753_75352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_bc_is_ten_l753_75359

/-- A triangle with vertices on a parabola -/
structure ParabolaTriangle where
  /-- The x-coordinate of point B (and negative x-coordinate of C) -/
  b : ℝ
  /-- All vertices lie on the parabola y = x^2 -/
  on_parabola : ∀ (x y : ℝ), (x = 0 ∧ y = 0) ∨ (x = b ∧ y = b^2) ∨ (x = -b ∧ y = b^2) → y = x^2
  /-- A is at the origin -/
  a_origin : (0 : ℝ) ∈ {x | ∃ y, (x = 0 ∧ y = 0) ∨ (x = b ∧ y = b^2) ∨ (x = -b ∧ y = b^2)}
  /-- BC is parallel to the x-axis -/
  bc_parallel_x : ∀ (x : ℝ), (x = b ∨ x = -b) → ∃ y, y = b^2
  /-- The triangle is symmetric about the y-axis -/
  symmetric : ∀ (x y : ℝ), (x = b ∧ y = b^2) → (-x = -b ∧ y = b^2)
  /-- The area of the triangle is 125 -/
  area : b^3 = 125

/-- The main theorem stating that the length of BC is 10 -/
theorem length_bc_is_ten (t : ParabolaTriangle) : (2 : ℝ) * t.b = 10 := by
  have h1 : t.b = 5 := by
    -- Prove that b = 5 using the area condition
    sorry
  -- Use h1 to prove the main statement
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_bc_is_ten_l753_75359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l753_75328

theorem triangle_side_value (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * Real.sin B = b * Real.sin A ∧
  b * Real.sin C = c * Real.sin B ∧
  Real.cos B / Real.cos C = -b / (2*a + c) ∧
  b = Real.sqrt 13 ∧
  a + c = 4 →
  a = 1 ∨ a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l753_75328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_special_triangle_l753_75304

/-- Given a triangle ABC where the sines of its angles are in the ratio 3:5:7,
    the largest angle of the triangle is 2π/3. -/
theorem largest_angle_in_special_triangle (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)
    (h_ratio : ∃ (k : ℝ), k > 0 ∧ Real.sin A = 3*k ∧ Real.sin B = 5*k ∧ Real.sin C = 7*k) :
    max A (max B C) = 2*Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_special_triangle_l753_75304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_l753_75315

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | |x - 1| > 1}

theorem complement_A : Set.compl A = Set.Icc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_l753_75315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_ratio_l753_75320

/-- Represents a two-segment journey with given speeds, total time, and maximum distance. -/
structure Journey where
  speed1 : ℝ
  speed2 : ℝ
  total_time : ℝ
  max_distance : ℝ

/-- Calculates the ratio of distance covered in the first segment to the total distance. -/
noncomputable def distance_ratio (j : Journey) : ℝ :=
  let d1 := (j.total_time * j.speed1 * j.speed2) / (j.speed1 + j.speed2)
  d1 / j.max_distance

/-- Theorem stating that for the given journey parameters, the distance ratio is 1/2. -/
theorem journey_distance_ratio :
  let j : Journey := {
    speed1 := 5
    speed2 := 4
    total_time := 6
    max_distance := 26.67
  }
  distance_ratio j = 1 / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_ratio_l753_75320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l753_75363

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a triangle with vertices D, E, F, and an interior point Q -/
structure TriangleDEF where
  D : Point
  E : Point
  F : Point
  Q : Point
  equilateral : distance D E = distance E F ∧ distance E F = distance F D
  interior : distance D Q = 7 ∧ distance E Q = 9 ∧ distance F Q = 11

/-- Calculates the area of an equilateral triangle given the side length -/
noncomputable def areaEquilateral (sideLength : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * sideLength^2

theorem area_of_triangle_DEF (t : TriangleDEF) :
  40 < areaEquilateral (distance t.D t.E) ∧ areaEquilateral (distance t.D t.E) < 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l753_75363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_addition_subtraction_l753_75300

-- Define complex numbers
def z1 : ℂ := 5 - 5*Complex.I
def z2 : ℂ := -2 - Complex.I
def z3 : ℂ := 3 + 4*Complex.I

-- State the theorem
theorem complex_addition_subtraction :
  z1 + z2 - z3 = -10*Complex.I :=
by
  -- Expand the definition of z1, z2, and z3
  simp [z1, z2, z3]
  -- Perform the complex arithmetic
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_addition_subtraction_l753_75300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_exponent_equality_l753_75366

theorem integer_exponent_equality (m : ℤ) : ((-2 : ℚ) ^ (2 * m) = 2 ^ (3 - m)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_exponent_equality_l753_75366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_common_elements_l753_75324

def arithmetic_progression (n : ℕ) : ℕ := 5 + 3 * n

def geometric_progression (k : ℕ) : ℕ := 10 * 2^k

def common_elements (n : ℕ) : Prop :=
  ∃ k : ℕ, arithmetic_progression n = geometric_progression k

def sum_of_first_10_common_elements : ℕ := 6990500

theorem sum_of_common_elements :
  sum_of_first_10_common_elements = 
    (Finset.range 10).sum (λ i => 20 * 4^i) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_common_elements_l753_75324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l753_75392

/-- The distance between the foci of a hyperbola given by the equation 9x^2 - 18x - 16y^2 - 32y = 144 is 10 -/
theorem hyperbola_foci_distance (x y : ℝ) : 
  (9 * x^2 - 18 * x - 16 * y^2 - 32 * y = 144) → 
  (∃ h k : ℝ × ℝ, 
    let a := Real.sqrt 16
    let b := 3
    let c := Real.sqrt (a^2 + b^2)
    h.1 - k.1 = 2 * c ∧ h.2 = k.2 ∧ h.1 - k.1 = 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l753_75392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_calculation_l753_75368

theorem cats_calculation (num_hogs : ℕ) (num_cats : ℕ) : 
  num_hogs = 75 → 
  num_hogs = 3 * num_cats → 
  (0.6 * (num_cats : ℝ) - 5 : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_calculation_l753_75368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l753_75370

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + x^2 - a*x

theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  (0 < x₁ ∧ x₁ ≤ 1) →
  (∃ (x : ℝ), x ≠ x₁ ∧ x ≠ x₂ ∧ ∀ y, f a y ≤ max (f a x₁) (f a x₂)) →
  f a x₁ - f a x₂ ≥ -3/4 + log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l753_75370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_equilateral_triangles_l753_75311

-- Define the triangle ABC and the external points A₁, B₁, C₁
variable (A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 2))

-- Define the property of equilateral triangles
def is_equilateral (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist P Q = dist Q R ∧ dist Q R = dist R P

-- State the conditions
variable (h1 : is_equilateral A₁ B C)
variable (h2 : is_equilateral A B₁ C)
variable (h3 : is_equilateral A B C₁)

-- State the theorem
theorem external_equilateral_triangles :
  dist A A₁ = dist B B₁ ∧ dist B B₁ = dist C C₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_equilateral_triangles_l753_75311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_distance_l753_75376

/-- Circle C with center (1, 2) and radius 5 -/
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

/-- Line l with slope m passing through (3, 1) -/
def L (m x y : ℝ) : Prop := m*x - y - 3*m + 1 = 0

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The minimum distance between intersection points of C and L is 4√5 -/
theorem min_intersection_distance :
  ∃ (m : ℝ), ∀ (x₁ y₁ x₂ y₂ : ℝ),
    C x₁ y₁ → C x₂ y₂ → L m x₁ y₁ → L m x₂ y₂ →
    distance x₁ y₁ x₂ y₂ ≥ 4 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_distance_l753_75376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_solution_l753_75382

/-- Base-8 representation of a two-digit number with first digit d --/
def base8 (d : ℕ) : ℕ := 8 * d + 9

/-- Base-9 representation of a two-digit number with first digit d --/
def base9 (d : ℕ) : ℕ := 9 * d + 6

/-- The unique single-digit solution to d9₈ = d6₉ is 3 --/
theorem diamond_solution : ∃! (d : ℕ), d < 10 ∧ base8 d = base9 d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_solution_l753_75382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planted_fraction_value_l753_75383

/-- Represents a right triangle with a square in the right angle corner --/
structure TriangleWithSquare where
  /-- Length of the first leg of the triangle --/
  leg1 : ℝ
  /-- Length of the second leg of the triangle --/
  leg2 : ℝ
  /-- Side length of the square in the corner --/
  square_side : ℝ
  /-- Shortest distance from the square to the hypotenuse --/
  distance_to_hypotenuse : ℝ

/-- The fraction of the triangle not covered by the square --/
noncomputable def planted_fraction (t : TriangleWithSquare) : ℝ :=
  (t.leg1 * t.leg2 / 2 - t.square_side ^ 2) / (t.leg1 * t.leg2 / 2)

/-- Theorem stating the planted fraction for the given triangle --/
theorem planted_fraction_value (t : TriangleWithSquare) 
  (h1 : t.leg1 = 5)
  (h2 : t.leg2 = 12)
  (h3 : t.distance_to_hypotenuse = 3) :
  planted_fraction t = 1740 / 8670 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planted_fraction_value_l753_75383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_radius_l753_75318

/-- The radius of the circle circumscribed about a sector with given parameters -/
noncomputable def circumscribed_radius (r : ℝ) (θ : ℝ) : ℝ :=
  r / (Real.cos (θ / 2))

/-- Theorem: The radius of the circle circumscribed about a sector cut from a circle
    with radius 10 and central angle 120° is 20 -/
theorem sector_circumscribed_radius :
  circumscribed_radius 10 (2 * Real.pi / 3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_radius_l753_75318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operation_result_l753_75327

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5}

theorem set_operation_result : (U \ M) ∩ N = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operation_result_l753_75327
