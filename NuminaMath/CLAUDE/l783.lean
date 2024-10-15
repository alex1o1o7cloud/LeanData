import Mathlib

namespace NUMINAMATH_CALUDE_inequality_theorem_l783_78322

theorem inequality_theorem (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a + b + (1/2 : ℝ) ≥ Real.sqrt a + Real.sqrt b := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l783_78322


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l783_78319

theorem arithmetic_geometric_progression (a b : ℝ) : 
  (1 - a = b - 1) ∧ (1 = a^2 * b^2) → 
  ((a = 1 ∧ b = 1) ∨ 
   (a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨ 
   (a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l783_78319


namespace NUMINAMATH_CALUDE_parabola_focus_theorem_l783_78345

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line type -/
structure Line

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle type -/
structure Circle where
  center : Point
  radius : ℝ

def intersects (l : Line) (para : Parabola) (A B : Point) : Prop :=
  sorry

def focus_on_line (para : Parabola) (l : Line) : Prop :=
  sorry

def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.x)^2 + (y - c.center.y)^2 = c.radius^2

def is_diameter (A B : Point) (c : Circle) : Prop :=
  sorry

theorem parabola_focus_theorem (para : Parabola) (l : Line) (A B : Point) (c : Circle) :
  intersects l para A B →
  focus_on_line para l →
  is_diameter A B c →
  circle_equation c 3 2 →
  c.radius = 4 →
  para.p = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_theorem_l783_78345


namespace NUMINAMATH_CALUDE_probability_all_same_suit_l783_78340

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of cards dealt to a player -/
def handSize : ℕ := 13

/-- The number of suits in a standard deck -/
def numSuits : ℕ := 4

/-- The number of cards in each suit -/
def cardsPerSuit : ℕ := deckSize / numSuits

theorem probability_all_same_suit :
  (numSuits : ℚ) / (deckSize.choose handSize : ℚ) =
  (numSuits : ℚ) / (Nat.choose deckSize handSize : ℚ) := by
  sorry

/-- The probability of all cards in a hand being from the same suit -/
def probabilitySameSuit : ℚ :=
  (numSuits : ℚ) / (Nat.choose deckSize handSize : ℚ)

end NUMINAMATH_CALUDE_probability_all_same_suit_l783_78340


namespace NUMINAMATH_CALUDE_product_equals_32_l783_78350

theorem product_equals_32 : 
  (1 / 4 : ℚ) * 8 * (1 / 16 : ℚ) * 32 * (1 / 64 : ℚ) * 128 * (1 / 256 : ℚ) * 512 * (1 / 1024 : ℚ) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_l783_78350


namespace NUMINAMATH_CALUDE_problem_statement_l783_78377

theorem problem_statement (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12)
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) :
  a + b^2 + c^3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l783_78377


namespace NUMINAMATH_CALUDE_existence_of_special_number_l783_78338

theorem existence_of_special_number (P : Finset Nat) (h_prime : ∀ p ∈ P, Prime p) :
  ∃ x : Nat,
    (∀ p ∈ P, ∃ a b : Nat, x = a^p + b^p) ∧
    (∀ p : Nat, Prime p → p ∉ P → ¬∃ a b : Nat, x = a^p + b^p) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l783_78338


namespace NUMINAMATH_CALUDE_nonreal_roots_product_l783_78317

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 6*x^3 + 15*x^2 - 20*x = 984) →
  (∃ a b : ℂ, (a ≠ b) ∧ (a.im ≠ 0) ∧ (b.im ≠ 0) ∧
   (x^4 - 6*x^3 + 15*x^2 - 20*x = 984 → (x = a ∨ x = b ∨ x.im = 0)) ∧
   (a * b = 4 - Real.sqrt 1000)) :=
sorry

end NUMINAMATH_CALUDE_nonreal_roots_product_l783_78317


namespace NUMINAMATH_CALUDE_hundredth_digit_of_seven_twentysixths_l783_78324

/-- The fraction we're working with -/
def f : ℚ := 7/26

/-- The length of the repeating sequence in the decimal representation of f -/
def repeat_length : ℕ := 9

/-- The repeating sequence in the decimal representation of f -/
def repeat_sequence : List ℕ := [2, 6, 9, 2, 3, 0, 7, 6, 9]

/-- The position we're interested in -/
def target_position : ℕ := 100

theorem hundredth_digit_of_seven_twentysixths (h1 : f = 7/26)
  (h2 : repeat_length = 9)
  (h3 : repeat_sequence = [2, 6, 9, 2, 3, 0, 7, 6, 9])
  (h4 : target_position = 100) :
  repeat_sequence[(target_position - 1) % repeat_length] = 2 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_digit_of_seven_twentysixths_l783_78324


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l783_78387

theorem rectangle_area_problem (l w : ℚ) : 
  (l + 3) * (w - 1) = l * w ∧ (l - 3) * (w + 2) = l * w → l * w = -90 / 121 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l783_78387


namespace NUMINAMATH_CALUDE_seven_pencils_of_one_color_l783_78355

/-- A box of colored pencils -/
structure ColoredPencilBox where
  pencils : Finset ℕ
  colors : ℕ → Finset ℕ
  total_pencils : pencils.card = 25
  color_property : ∀ s : Finset ℕ, s ⊆ pencils → s.card = 5 → ∃ c, (s ∩ colors c).card ≥ 2

/-- There are at least seven pencils of one color in the box -/
theorem seven_pencils_of_one_color (box : ColoredPencilBox) : 
  ∃ c, (box.pencils ∩ box.colors c).card ≥ 7 := by
  sorry


end NUMINAMATH_CALUDE_seven_pencils_of_one_color_l783_78355


namespace NUMINAMATH_CALUDE_john_video_release_l783_78347

/-- The number of videos John releases per day -/
def videos_per_day : ℕ := 3

/-- The length of a short video in minutes -/
def short_video_length : ℕ := 2

/-- The number of short videos released per day -/
def short_videos_per_day : ℕ := 2

/-- The factor by which the long video is longer than the short videos -/
def long_video_factor : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Calculates the total minutes of video John releases per week -/
def total_minutes_per_week : ℕ :=
  days_per_week * (
    short_videos_per_day * short_video_length +
    (videos_per_day - short_videos_per_day) * (long_video_factor * short_video_length)
  )

theorem john_video_release :
  total_minutes_per_week = 112 := by
  sorry

end NUMINAMATH_CALUDE_john_video_release_l783_78347


namespace NUMINAMATH_CALUDE_rearrange_incongruent_sums_l783_78362

/-- Given two lists of 2014 integers that are pairwise incongruent modulo 2014,
    there exists a permutation of the second list such that the pairwise sums
    of corresponding elements from both lists are incongruent modulo 4028. -/
theorem rearrange_incongruent_sums
  (x y : Fin 2014 → ℤ)
  (hx : ∀ i j, i ≠ j → x i % 2014 ≠ x j % 2014)
  (hy : ∀ i j, i ≠ j → y i % 2014 ≠ y j % 2014) :
  ∃ σ : Equiv.Perm (Fin 2014),
    ∀ i j, i ≠ j → (x i + y (σ i)) % 4028 ≠ (x j + y (σ j)) % 4028 :=
by sorry

end NUMINAMATH_CALUDE_rearrange_incongruent_sums_l783_78362


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l783_78395

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (x^2 - x) / (x^2 + 2*x + 1) / ((x - 1) / (x + 1)) = x / (x + 1) ∧
  (3^2 - 3) / (3^2 + 2*3 + 1) / ((3 - 1) / (3 + 1)) = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l783_78395


namespace NUMINAMATH_CALUDE_common_tangents_count_l783_78314

/-- Two circles in a plane -/
structure CirclePair :=
  (c1 c2 : Set ℝ × ℝ)
  (r1 r2 : ℝ)
  (h_unequal : r1 ≠ r2)

/-- The number of common tangents for a pair of circles -/
def num_common_tangents (cp : CirclePair) : ℕ := sorry

/-- Theorem stating that the number of common tangents for unequal circles is always 0, 1, 2, 3, or 4 -/
theorem common_tangents_count (cp : CirclePair) :
  num_common_tangents cp = 0 ∨
  num_common_tangents cp = 1 ∨
  num_common_tangents cp = 2 ∨
  num_common_tangents cp = 3 ∨
  num_common_tangents cp = 4 :=
sorry

end NUMINAMATH_CALUDE_common_tangents_count_l783_78314


namespace NUMINAMATH_CALUDE_triangle_sides_l783_78353

-- Define the rhombus side length
def rhombus_side : ℝ := 6

-- Define the triangle
structure Triangle where
  a : ℝ  -- shortest side
  b : ℝ  -- middle side
  c : ℝ  -- hypotenuse

-- Define the properties of the triangle and rhombus
def triangle_with_inscribed_rhombus (t : Triangle) : Prop :=
  -- The triangle is right-angled with a 60° angle
  t.a^2 + t.b^2 = t.c^2 ∧
  t.a / t.c = 1 / 2 ∧
  -- The rhombus is inscribed in the triangle
  -- (We don't need to explicitly state this as it's implied by the problem setup)
  -- The rhombus side length is 6
  rhombus_side = 6

-- The theorem to prove
theorem triangle_sides (t : Triangle) 
  (h : triangle_with_inscribed_rhombus t) : 
  t.a = 9 ∧ t.b = 9 * Real.sqrt 3 ∧ t.c = 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_l783_78353


namespace NUMINAMATH_CALUDE_expected_percentage_proof_l783_78312

/-- The probability of rain for a county on Monday -/
def prob_rain_monday : ℝ := 0.70

/-- The probability of rain for a county on Tuesday -/
def prob_rain_tuesday : ℝ := 0.80

/-- The probability of rain for a county on Wednesday -/
def prob_rain_wednesday : ℝ := 0.60

/-- The proportion of counties with chance of rain on Monday -/
def prop_counties_monday : ℝ := 0.60

/-- The proportion of counties with chance of rain on Tuesday -/
def prop_counties_tuesday : ℝ := 0.55

/-- The proportion of counties with chance of rain on Wednesday -/
def prop_counties_wednesday : ℝ := 0.40

/-- The proportion of counties that received rain on at least one day -/
def prop_counties_with_rain : ℝ := 0.80

/-- The expected percentage of counties that will receive rain on all three days -/
def expected_percentage : ℝ :=
  prop_counties_monday * prob_rain_monday *
  prop_counties_tuesday * prob_rain_tuesday *
  prop_counties_wednesday * prob_rain_wednesday *
  prop_counties_with_rain

theorem expected_percentage_proof :
  expected_percentage = 0.60 * 0.70 * 0.55 * 0.80 * 0.40 * 0.60 * 0.80 :=
by sorry

end NUMINAMATH_CALUDE_expected_percentage_proof_l783_78312


namespace NUMINAMATH_CALUDE_product_simplification_l783_78329

theorem product_simplification (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l783_78329


namespace NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_l783_78397

-- Define the binomial expansion function
def binomialExpand (n : ℕ) (x : ℝ) : ℝ := (1 + x) ^ n

-- Define the coefficient extraction function
def coefficientOf (term : ℕ × ℕ) (expansion : ℝ → ℝ → ℝ) : ℝ :=
  sorry -- Placeholder for the actual implementation

theorem coefficient_x2y2_in_expansion :
  coefficientOf (2, 2) (fun x y => binomialExpand 3 x * binomialExpand 4 y) = 18 := by
  sorry

#check coefficient_x2y2_in_expansion

end NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_l783_78397


namespace NUMINAMATH_CALUDE_column_compression_strength_l783_78364

theorem column_compression_strength (T H L : ℚ) : 
  T = 3 → H = 6 → L = (15 * T^5) / H^3 → L = 55 / 13 := by sorry

end NUMINAMATH_CALUDE_column_compression_strength_l783_78364


namespace NUMINAMATH_CALUDE_min_button_presses_l783_78354

/-- Represents the state of the room --/
structure RoomState where
  armedMines : ℕ
  closedDoors : ℕ

/-- Represents the actions of pressing buttons --/
inductive ButtonPress
  | Red
  | Yellow
  | Green

/-- Defines the effect of pressing a button on the room state --/
def pressButton (state : RoomState) (button : ButtonPress) : RoomState :=
  match button with
  | ButtonPress.Red => ⟨state.armedMines + 1, state.closedDoors⟩
  | ButtonPress.Yellow => 
      if state.armedMines ≥ 2 
      then ⟨state.armedMines - 2, state.closedDoors + 1⟩ 
      else ⟨3, 3⟩  -- Reset condition
  | ButtonPress.Green => 
      if state.closedDoors ≥ 2 
      then ⟨state.armedMines, state.closedDoors - 2⟩ 
      else ⟨3, 3⟩  -- Reset condition

/-- Defines the initial state of the room --/
def initialState : RoomState := ⟨3, 3⟩

/-- Defines the goal state (all mines disarmed and all doors opened) --/
def goalState : RoomState := ⟨0, 0⟩

/-- Theorem stating that the minimum number of button presses to reach the goal state is 9 --/
theorem min_button_presses : 
  ∃ (sequence : List ButtonPress), 
    sequence.length = 9 ∧ 
    (sequence.foldl pressButton initialState = goalState) ∧
    (∀ (otherSequence : List ButtonPress), 
      otherSequence.foldl pressButton initialState = goalState → 
      otherSequence.length ≥ 9) := by
  sorry


end NUMINAMATH_CALUDE_min_button_presses_l783_78354


namespace NUMINAMATH_CALUDE_first_road_workers_approx_30_man_hours_proportional_to_length_l783_78389

/-- Represents the details of a road construction project -/
structure RoadProject where
  length : ℝ  -- Road length in km
  workers : ℝ  -- Number of workers
  days : ℝ    -- Number of days worked
  hoursPerDay : ℝ  -- Hours worked per day

/-- Calculates the total man-hours for a road project -/
def manHours (project : RoadProject) : ℝ :=
  project.workers * project.days * project.hoursPerDay

/-- The first road project -/
def road1 : RoadProject := {
  length := 1,
  workers := 30,  -- This is what we're trying to prove
  days := 12,
  hoursPerDay := 8
}

/-- The second road project -/
def road2 : RoadProject := {
  length := 2,
  workers := 20,
  days := 20.571428571428573,
  hoursPerDay := 14
}

/-- Theorem stating that the number of workers on the first road is approximately 30 -/
theorem first_road_workers_approx_30 :
  ∃ ε > 0, ε < 1 ∧ |road1.workers - 30| < ε :=
by sorry

/-- Theorem showing the relationship between man-hours and road length -/
theorem man_hours_proportional_to_length :
  2 * manHours road1 = manHours road2 :=
by sorry

end NUMINAMATH_CALUDE_first_road_workers_approx_30_man_hours_proportional_to_length_l783_78389


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l783_78363

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 8*I) * (a + b*I) = y*I) : 
  a / b = -8 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l783_78363


namespace NUMINAMATH_CALUDE_min_xy_value_l783_78343

theorem min_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2*x + y + 6 = x*y) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2*a + b + 6 = a*b → x*y ≤ a*b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y + 6 = x*y ∧ x*y = 18 :=
sorry

end NUMINAMATH_CALUDE_min_xy_value_l783_78343


namespace NUMINAMATH_CALUDE_bread_slices_remaining_l783_78375

theorem bread_slices_remaining (total_slices : ℕ) (breakfast_fraction : ℚ) (lunch_slices : ℕ) : 
  total_slices = 12 →
  breakfast_fraction = 1 / 3 →
  lunch_slices = 2 →
  total_slices - (breakfast_fraction * total_slices).num - lunch_slices = 6 := by
sorry

end NUMINAMATH_CALUDE_bread_slices_remaining_l783_78375


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l783_78372

/-- Given a circle C' with equation x^2 - 4y - 15 = -y^2 + 12x + 27,
    prove that its center (p, q) and radius s satisfy p + q + s = 8 + √82 -/
theorem circle_center_radius_sum (x y p q s : ℝ) : 
  (∀ x y, x^2 - 4*y - 15 = -y^2 + 12*x + 27) →
  (∀ x y, (x - p)^2 + (y - q)^2 = s^2) →
  p + q + s = 8 + Real.sqrt 82 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l783_78372


namespace NUMINAMATH_CALUDE_volume_equivalence_l783_78342

/-- A parallelepiped with congruent rhombic faces and a special vertex -/
structure RhombicParallelepiped where
  -- Side length of the rhombic face
  a : ℝ
  -- Angle between edges at the special vertex
  α : ℝ
  -- Diagonals of the rhombic face
  e : ℝ
  f : ℝ
  -- Conditions
  a_pos : 0 < a
  α_pos : 0 < α
  α_not_right : α ≠ π / 2
  α_less_120 : α < 2 * π / 3
  e_pos : 0 < e
  f_pos : 0 < f
  diag_relation : a = (1 / 2) * Real.sqrt (e^2 + f^2)
  angle_relation : Real.tan (α / 2) = f / e

/-- The volume of a rhombic parallelepiped -/
noncomputable def volume (p : RhombicParallelepiped) : ℝ :=
  p.a^3 * Real.sin p.α * Real.sqrt (Real.sin p.α^2 - Real.cos p.α^2 * Real.tan (p.α / 2)^2)

/-- The volume of a rhombic parallelepiped in terms of diagonals -/
noncomputable def volume_diag (p : RhombicParallelepiped) : ℝ :=
  (p.f / (8 * p.e)) * (p.e^2 + p.f^2) * Real.sqrt (3 * p.e^2 - p.f^2)

/-- The main theorem: equivalence of volume formulas -/
theorem volume_equivalence (p : RhombicParallelepiped) : volume p = volume_diag p := by
  sorry

end NUMINAMATH_CALUDE_volume_equivalence_l783_78342


namespace NUMINAMATH_CALUDE_paper_clips_count_l783_78358

/-- The number of paper clips in a storage unit -/
def total_paper_clips (c b p : ℕ) : ℕ :=
  300 * c + 550 * b + 1200 * p

/-- Theorem stating that the total number of paper clips in 3 cartons, 4 boxes, and 2 bags is 5500 -/
theorem paper_clips_count : total_paper_clips 3 4 2 = 5500 := by
  sorry

end NUMINAMATH_CALUDE_paper_clips_count_l783_78358


namespace NUMINAMATH_CALUDE_lollipop_bouquets_l783_78393

theorem lollipop_bouquets (cherry orange raspberry lemon candycane chocolate : ℕ) 
  (h1 : cherry = 4)
  (h2 : orange = 6)
  (h3 : raspberry = 8)
  (h4 : lemon = 10)
  (h5 : candycane = 12)
  (h6 : chocolate = 14) :
  Nat.gcd cherry (Nat.gcd orange (Nat.gcd raspberry (Nat.gcd lemon (Nat.gcd candycane chocolate)))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_bouquets_l783_78393


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l783_78301

theorem complex_magnitude_squared (z : ℂ) (h : z + Complex.abs z = -3 + 4*I) : Complex.abs z ^ 2 = 625 / 36 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l783_78301


namespace NUMINAMATH_CALUDE_constant_term_expansion_l783_78399

theorem constant_term_expansion (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f x = (a * x^2 + 2 / Real.sqrt x)^5 ∧ 
   (∃ (c : ℝ), c = 80 ∧ (∀ x, f x = c + x * (f x - c) / x))) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l783_78399


namespace NUMINAMATH_CALUDE_homework_time_difference_l783_78330

/-- Given the conditions of the homework problem, prove that Greg has 6 hours less than Jacob. -/
theorem homework_time_difference :
  ∀ (greg_hours patrick_hours jacob_hours : ℕ),
  patrick_hours = 2 * greg_hours - 4 →
  jacob_hours = 18 →
  patrick_hours + greg_hours + jacob_hours = 50 →
  jacob_hours - greg_hours = 6 := by
sorry

end NUMINAMATH_CALUDE_homework_time_difference_l783_78330


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l783_78341

theorem sum_of_solutions_squared_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 8)^2 = 49 ∧ (b - 8)^2 = 49 ∧ a ≠ b) →
  (∃ s : ℝ, s = a + b ∧ s = 16) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l783_78341


namespace NUMINAMATH_CALUDE_tangent_line_equations_l783_78337

-- Define the function
def f (x : ℝ) : ℝ := -x^3 - 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -3 * x^2

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the tangent line equation
def tangent_line (m : ℝ) (x : ℝ) : ℝ := 
  -m^3 - 1 + (f' m) * (x - m)

-- Theorem statement
theorem tangent_line_equations :
  ∃ (m₁ m₂ : ℝ), 
    m₁ ≠ m₂ ∧
    tangent_line m₁ P.1 = P.2 ∧
    tangent_line m₂ P.1 = P.2 ∧
    (∀ (x : ℝ), tangent_line m₁ x = -3*x - 1) ∧
    (∀ (x : ℝ), tangent_line m₂ x = -(3*x + 5) / 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l783_78337


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l783_78384

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x + y > 2 → max x y > 1) ∧
  ¬(max x y > 1 → x + y > 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l783_78384


namespace NUMINAMATH_CALUDE_smallest_n_for_trig_inequality_l783_78316

theorem smallest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), Real.sin x ^ n + Real.cos x ^ n ≤ 2 / n) ∧ 
  (∀ (n : ℕ), n > 0 ∧ (∀ (x : ℝ), Real.sin x ^ n + Real.cos x ^ n ≤ 2 / n) → n ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_trig_inequality_l783_78316


namespace NUMINAMATH_CALUDE_same_last_four_digits_theorem_l783_78315

theorem same_last_four_digits_theorem (N : ℕ) (a b c d : Fin 10) :
  (a ≠ 0) →
  (N % 10000 = a * 1000 + b * 100 + c * 10 + d) →
  ((N + 2) % 10000 = a * 1000 + b * 100 + c * 10 + d) →
  (a * 100 + b * 10 + c = 999) :=
by sorry

end NUMINAMATH_CALUDE_same_last_four_digits_theorem_l783_78315


namespace NUMINAMATH_CALUDE_refrigerator_installation_cost_l783_78394

def refrigerator_problem (purchase_price : ℝ) (discount_rate : ℝ) 
  (transport_cost : ℝ) (selling_price : ℝ) : Prop :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let profit_rate := 0.1
  let total_cost := labelled_price + transport_cost + 
    (selling_price - labelled_price * (1 + profit_rate))
  total_cost - purchase_price - transport_cost = 287.5

theorem refrigerator_installation_cost :
  refrigerator_problem 13500 0.2 125 18975 :=
sorry

end NUMINAMATH_CALUDE_refrigerator_installation_cost_l783_78394


namespace NUMINAMATH_CALUDE_range_of_m_l783_78335

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3)
  (hineq : ∀ m : ℝ, (4 / (x + 1)) + (16 / y) > m^2 - 3*m + 11) :
  ∀ m : ℝ, (1 < m ∧ m < 2) ↔ (4 / (x + 1)) + (16 / y) > m^2 - 3*m + 11 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l783_78335


namespace NUMINAMATH_CALUDE_product_inequality_l783_78365

theorem product_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l783_78365


namespace NUMINAMATH_CALUDE_square_difference_401_399_l783_78367

theorem square_difference_401_399 : 401^2 - 399^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_401_399_l783_78367


namespace NUMINAMATH_CALUDE_polar_to_rect_transformation_l783_78356

/-- Given a point with rectangular coordinates (10, 3) and polar coordinates (r, θ),
    prove that the point with polar coordinates (r², 2θ) has rectangular coordinates (91, 60). -/
theorem polar_to_rect_transformation (r θ : ℝ) (h1 : r * Real.cos θ = 10) (h2 : r * Real.sin θ = 3) :
  (r^2 * Real.cos (2*θ), r^2 * Real.sin (2*θ)) = (91, 60) := by
  sorry


end NUMINAMATH_CALUDE_polar_to_rect_transformation_l783_78356


namespace NUMINAMATH_CALUDE_pascal_triangle_30th_row_28th_number_l783_78360

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : ℕ := 30

/-- The position of the number we're looking for (1-indexed) -/
def target_position : ℕ := 28

/-- The value we're proving the target position contains -/
def target_value : ℕ := 406

theorem pascal_triangle_30th_row_28th_number :
  Nat.choose (row_length - 1) (target_position - 1) = target_value := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30th_row_28th_number_l783_78360


namespace NUMINAMATH_CALUDE_inscribed_squares_product_l783_78368

theorem inscribed_squares_product (a b : ℝ) : 
  (∃ small_square large_square : ℝ → ℝ → Prop,
    (∀ x y, small_square x y → x^2 + y^2 ≤ 9) ∧
    (∀ x y, large_square x y → x^2 + y^2 ≤ 16) ∧
    (∀ x y, small_square x y → ∃ u v, large_square u v ∧ 
      ((x = u ∧ y ∈ [0, 4]) ∨ (x ∈ [0, 4] ∧ y = v) ∨ 
       (x = -u ∧ y ∈ [0, 4]) ∨ (x ∈ [0, 4] ∧ y = -v))) ∧
    (a + b = 4) ∧
    (a^2 + b^2 = 18)) →
  a * b = -1 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_product_l783_78368


namespace NUMINAMATH_CALUDE_rally_speaking_orders_l783_78396

theorem rally_speaking_orders (n : ℕ) : 
  2 * (n.factorial) * (n.factorial) = 72 → n = 3 :=
by sorry

end NUMINAMATH_CALUDE_rally_speaking_orders_l783_78396


namespace NUMINAMATH_CALUDE_shoe_cost_calculation_l783_78385

def initial_savings : ℕ := 30
def earnings_per_lawn : ℕ := 5
def lawns_per_weekend : ℕ := 3
def weekends_needed : ℕ := 6

def total_earnings : ℕ := earnings_per_lawn * lawns_per_weekend * weekends_needed

theorem shoe_cost_calculation :
  initial_savings + total_earnings = 120 := by sorry

end NUMINAMATH_CALUDE_shoe_cost_calculation_l783_78385


namespace NUMINAMATH_CALUDE_exists_steps_for_1001_free_ends_l783_78369

/-- Represents the number of free ends after k steps of construction -/
def free_ends (k : ℕ) : ℕ := 4 * k + 1

/-- Theorem stating that there exists a number of steps that results in 1001 free ends -/
theorem exists_steps_for_1001_free_ends : ∃ k : ℕ, free_ends k = 1001 := by
  sorry

end NUMINAMATH_CALUDE_exists_steps_for_1001_free_ends_l783_78369


namespace NUMINAMATH_CALUDE_divisor_property_l783_78336

theorem divisor_property (k : ℕ) : 18^k ∣ 624938 → 6^k - k^6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_property_l783_78336


namespace NUMINAMATH_CALUDE_binomial_coefficient_26_6_l783_78325

theorem binomial_coefficient_26_6 
  (h1 : Nat.choose 24 5 = 42504)
  (h2 : Nat.choose 25 5 = 53130)
  (h3 : Nat.choose 25 6 = 177100) :
  Nat.choose 26 6 = 230230 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_26_6_l783_78325


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l783_78359

theorem coefficient_x_squared_in_expansion (x : ℝ) :
  ∃ c, (x + 2)^4 = c * x^2 + (x + 2)^4 - c * x^2 ∧ c = 24 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l783_78359


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_tangent_to_circle_l783_78381

theorem hyperbola_asymptotes_tangent_to_circle (m : ℝ) : 
  m > 0 →
  (∀ x y : ℝ, y^2 - x^2 / m^2 = 1) →
  (∀ x y : ℝ, x^2 + y^2 - 4*y + 3 = 0) →
  (∃ x y : ℝ, y = m*x ∧ x^2 + y^2 - 4*y + 3 = 0) →
  m = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_tangent_to_circle_l783_78381


namespace NUMINAMATH_CALUDE_hexagon_perimeter_sum_l783_78300

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  perimeter : ℝ

/-- Represents a hexagon formed by two equilateral triangles -/
def hexagon_from_triangles (t1 t2 : EquilateralTriangle) : ℝ := t1.perimeter + t2.perimeter

theorem hexagon_perimeter_sum (t1 t2 : EquilateralTriangle) 
  (h1 : t1.perimeter = 12) (h2 : t2.perimeter = 15) : 
  hexagon_from_triangles t1 t2 = 27 := by
  sorry

#check hexagon_perimeter_sum

end NUMINAMATH_CALUDE_hexagon_perimeter_sum_l783_78300


namespace NUMINAMATH_CALUDE_intersection_condition_l783_78351

theorem intersection_condition (a : ℝ) : 
  (∃ x y : ℝ, ax + 2*y = 3 ∧ x + (a-1)*y = 1) → a ≠ 2 ∧ 
  ¬(a ≠ 2 → ∃ x y : ℝ, ax + 2*y = 3 ∧ x + (a-1)*y = 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_l783_78351


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l783_78379

/-- Given two lines a²x + y + 7 = 0 and x - 2ay + 1 = 0 that are perpendicular,
    prove that a = 0 or a = 2 -/
theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, a^2*x + y + 7 = 0 ∧ x - 2*a*y + 1 = 0 → 
    (a^2 : ℝ) * (1 / (-2*a)) = -1) →
  a = 0 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l783_78379


namespace NUMINAMATH_CALUDE_probability_of_blue_ball_l783_78344

theorem probability_of_blue_ball (p_red p_yellow p_blue : ℝ) : 
  p_red = 0.48 → p_yellow = 0.35 → p_red + p_yellow + p_blue = 1 → p_blue = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_blue_ball_l783_78344


namespace NUMINAMATH_CALUDE_special_function_is_zero_l783_78303

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≥ 0) ∧
  (∀ x, DifferentiableAt ℝ f x) ∧
  (∀ x, deriv f x ≥ 0) ∧
  (∀ n : ℤ, f n = 0)

/-- Theorem stating that any function satisfying the conditions must be identically zero -/
theorem special_function_is_zero (f : ℝ → ℝ) (hf : SpecialFunction f) : 
  ∀ x, f x = 0 := by sorry

end NUMINAMATH_CALUDE_special_function_is_zero_l783_78303


namespace NUMINAMATH_CALUDE_tangent_line_equation_l783_78307

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 + x - 8

-- Define the point of tangency
def point : ℝ × ℝ := (1, -6)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (HasDerivAt f (m * x - y + b) point.1) ∧
    (f point.1 = point.2) ∧
    (m = 4 ∧ b = -10) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l783_78307


namespace NUMINAMATH_CALUDE_otimes_one_eq_two_implies_k_eq_one_l783_78323

def otimes (a b : ℝ) : ℝ := a * b + a + b^2

theorem otimes_one_eq_two_implies_k_eq_one (k : ℝ) (h1 : k > 0) (h2 : otimes 1 k = 2) : k = 1 := by
  sorry

end NUMINAMATH_CALUDE_otimes_one_eq_two_implies_k_eq_one_l783_78323


namespace NUMINAMATH_CALUDE_fraction_always_defined_l783_78318

theorem fraction_always_defined (x : ℝ) : (x^2 + 1) ≠ 0 := by
  sorry

#check fraction_always_defined

end NUMINAMATH_CALUDE_fraction_always_defined_l783_78318


namespace NUMINAMATH_CALUDE_joeys_lawn_mowing_l783_78386

theorem joeys_lawn_mowing (
  sneaker_cost : ℕ)
  (lawn_earnings : ℕ)
  (figure_price : ℕ)
  (figure_count : ℕ)
  (job_hours : ℕ)
  (hourly_rate : ℕ)
  (h1 : sneaker_cost = 92)
  (h2 : lawn_earnings = 8)
  (h3 : figure_price = 9)
  (h4 : figure_count = 2)
  (h5 : job_hours = 10)
  (h6 : hourly_rate = 5)
  : (sneaker_cost - (figure_price * figure_count + job_hours * hourly_rate)) / lawn_earnings = 3 := by
  sorry

end NUMINAMATH_CALUDE_joeys_lawn_mowing_l783_78386


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l783_78311

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x, (1/3 < x ∧ x < 1/2) ↔ a*x^2 + 5*x + c > 0) →
  a = -6 ∧ c = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l783_78311


namespace NUMINAMATH_CALUDE_complex_multiplication_l783_78302

theorem complex_multiplication (z : ℂ) : 
  (z.re = -1 ∧ z.im = 1) → z * (1 + Complex.I) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l783_78302


namespace NUMINAMATH_CALUDE_person_a_work_time_l783_78391

theorem person_a_work_time (person_b_time : ℝ) (combined_work_rate : ℝ) (combined_work_time : ℝ) :
  person_b_time = 45 →
  combined_work_rate = 2/9 →
  combined_work_time = 4 →
  ∃ person_a_time : ℝ,
    person_a_time = 30 ∧
    combined_work_rate = combined_work_time * (1 / person_a_time + 1 / person_b_time) :=
by sorry

end NUMINAMATH_CALUDE_person_a_work_time_l783_78391


namespace NUMINAMATH_CALUDE_chicken_count_l783_78332

theorem chicken_count (coop run free_range : ℕ) : 
  coop = 14 →
  run = 2 * coop →
  free_range = 2 * run - 4 →
  free_range = 52 := by
sorry

end NUMINAMATH_CALUDE_chicken_count_l783_78332


namespace NUMINAMATH_CALUDE_sam_reading_speed_l783_78313

/-- Proves that given Dustin can read 75 pages in an hour and reads 34 more pages than Sam in 40 minutes, Sam can read 72 pages in an hour. -/
theorem sam_reading_speed (dustin_pages_per_hour : ℕ) (extra_pages : ℕ) : 
  dustin_pages_per_hour = 75 → 
  dustin_pages_per_hour * (40 : ℚ) / 60 - extra_pages = 
    (72 : ℚ) * (40 : ℚ) / 60 → 
  extra_pages = 34 →
  72 = 72 := by sorry

end NUMINAMATH_CALUDE_sam_reading_speed_l783_78313


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l783_78333

/-- An arithmetic sequence {aₙ} with a₅ = 9 and a₁ + a₇ = 14 has the general formula aₙ = 2n - 1 -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence
  a 5 = 9 →
  a 1 + a 7 = 14 →
  ∀ n : ℕ, a n = 2 * n - 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l783_78333


namespace NUMINAMATH_CALUDE_river_width_proof_l783_78382

/-- The width of a river where two men start from opposite banks, meet 340 meters from one bank
    on their forward journey, and 170 meters from the other bank on their backward journey. -/
def river_width : ℝ := 340

theorem river_width_proof (forward_meeting : ℝ) (backward_meeting : ℝ) 
  (h1 : forward_meeting = 340)
  (h2 : backward_meeting = 170)
  (h3 : forward_meeting + (river_width - forward_meeting) = river_width)
  (h4 : backward_meeting + (river_width - backward_meeting) = river_width) :
  river_width = 340 := by
  sorry

end NUMINAMATH_CALUDE_river_width_proof_l783_78382


namespace NUMINAMATH_CALUDE_cosine_sum_equality_l783_78321

theorem cosine_sum_equality : 
  Real.cos (47 * π / 180) * Real.cos (13 * π / 180) - 
  Real.cos (43 * π / 180) * Real.sin (167 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_equality_l783_78321


namespace NUMINAMATH_CALUDE_y_derivative_l783_78352

noncomputable def y (x : ℝ) : ℝ := Real.cos (Real.log 2) - (1/3) * (Real.cos (3*x))^2 / Real.sin (6*x)

theorem y_derivative (x : ℝ) (h : Real.sin (6*x) ≠ 0) :
  deriv y x = 1 / (2 * (Real.sin (3*x))^2) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l783_78352


namespace NUMINAMATH_CALUDE_discount_sales_income_increase_l783_78357

/-- Proves that a 10% discount with 30% increase in sales leads to 17% increase in gross income -/
theorem discount_sales_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1) 
  (h2 : sales_increase_rate = 0.3) : 
  (((1 + sales_increase_rate) * (1 - discount_rate) - 1) * 100 : ℝ) = 17 := by
  sorry

end NUMINAMATH_CALUDE_discount_sales_income_increase_l783_78357


namespace NUMINAMATH_CALUDE_exists_initial_order_l783_78306

/-- Represents a playing card suit -/
inductive Suit
| Diamonds
| Hearts
| Spades
| Clubs

/-- Represents a playing card rank -/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents a playing card -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- The number of letters in the name of a card's rank -/
def letterCount : Rank → Nat
| Rank.Ace => 3
| Rank.Two => 3
| Rank.Three => 5
| Rank.Four => 4
| Rank.Five => 4
| Rank.Six => 3
| Rank.Seven => 5
| Rank.Eight => 5
| Rank.Nine => 4
| Rank.Ten => 3
| Rank.Jack => 4
| Rank.Queen => 5
| Rank.King => 4

/-- Applies the card moving rule to a deck -/
def applyRule (deck : List Card) : List Card :=
  sorry

/-- The final order of cards after applying the rule -/
def finalOrder : List Card :=
  sorry

/-- Theorem: There exists an initial deck ordering that results in the specified final order -/
theorem exists_initial_order :
  ∃ (initialDeck : List Card),
    initialDeck.length = 52 ∧
    (∀ s : Suit, ∀ r : Rank, ∃ c ∈ initialDeck, c.suit = s ∧ c.rank = r) ∧
    applyRule initialDeck = finalOrder :=
  sorry

end NUMINAMATH_CALUDE_exists_initial_order_l783_78306


namespace NUMINAMATH_CALUDE_peaches_picked_l783_78373

theorem peaches_picked (initial_peaches final_peaches : ℕ) 
  (h1 : initial_peaches = 34)
  (h2 : final_peaches = 86) :
  final_peaches - initial_peaches = 52 := by
  sorry

end NUMINAMATH_CALUDE_peaches_picked_l783_78373


namespace NUMINAMATH_CALUDE_inequality_implication_l783_78310

theorem inequality_implication (a b c : ℝ) : a > b → a * c^2 ≥ b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l783_78310


namespace NUMINAMATH_CALUDE_circle_equation_proof_l783_78349

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation of a circle -/
structure CircleEquation where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Checks if a point satisfies the circle equation -/
def satisfiesCircleEquation (p : Point2D) (c : CircleEquation) : Prop :=
  p.x^2 + p.y^2 + c.D * p.x + c.E * p.y + c.F = 0

/-- Theorem: The equation x^2 + y^2 - 4x - 6y = 0 represents a circle passing through (0,0), (4,0), and (-1,1) -/
theorem circle_equation_proof :
  let c : CircleEquation := ⟨-4, -6, 0⟩
  satisfiesCircleEquation ⟨0, 0⟩ c ∧
  satisfiesCircleEquation ⟨4, 0⟩ c ∧
  satisfiesCircleEquation ⟨-1, 1⟩ c :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l783_78349


namespace NUMINAMATH_CALUDE_power_of_one_third_l783_78398

theorem power_of_one_third (a b : ℕ) : 
  (2^a : ℕ) * (5^b : ℕ) = 200 → 
  (∀ k : ℕ, 2^k ∣ 200 → k ≤ a) →
  (∀ k : ℕ, 5^k ∣ 200 → k ≤ b) →
  (1/3 : ℚ)^(b - a) = 3 := by sorry

end NUMINAMATH_CALUDE_power_of_one_third_l783_78398


namespace NUMINAMATH_CALUDE_aaron_erasers_l783_78326

/-- The number of erasers Aaron gives away -/
def erasers_given : ℕ := 34

/-- The number of erasers Aaron ends with -/
def erasers_left : ℕ := 47

/-- The initial number of erasers Aaron had -/
def initial_erasers : ℕ := erasers_given + erasers_left

theorem aaron_erasers : initial_erasers = 81 := by
  sorry

end NUMINAMATH_CALUDE_aaron_erasers_l783_78326


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_implies_a_eq_neg_one_l783_78348

/-- A geometric sequence with sum of first n terms given by Sn = 4^n + a -/
def GeometricSequence (a : ℝ) := ℕ → ℝ

/-- Sum of first n terms of the geometric sequence -/
def SumFirstNTerms (seq : GeometricSequence a) (n : ℕ) : ℝ := 4^n + a

/-- The ratio between consecutive terms in a geometric sequence is constant -/
def IsGeometric (seq : GeometricSequence a) : Prop :=
  ∀ n : ℕ, seq (n + 2) / seq (n + 1) = seq (n + 1) / seq n

theorem geometric_sequence_sum_implies_a_eq_neg_one :
  ∀ a : ℝ, ∃ seq : GeometricSequence a,
    (∀ n : ℕ, SumFirstNTerms seq n = 4^n + a) →
    IsGeometric seq →
    a = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_implies_a_eq_neg_one_l783_78348


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l783_78392

-- Problem 1
theorem problem_1 : Real.sqrt 5 ^ 2 + |(-3)| - (Real.pi + Real.sqrt 3) ^ 0 = 7 := by sorry

-- Problem 2
theorem problem_2 : 
  Set.Ioo (-1 : ℝ) 2 = {x : ℝ | 5 * x - 10 ≤ 0 ∧ x + 3 > -2 * x} := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l783_78392


namespace NUMINAMATH_CALUDE_red_peaches_count_l783_78346

/-- Given a basket of peaches, calculate the number of red peaches. -/
def red_peaches (total_peaches green_peaches : ℕ) : ℕ :=
  total_peaches - green_peaches

/-- Theorem: The number of red peaches in the basket is 4. -/
theorem red_peaches_count : red_peaches 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l783_78346


namespace NUMINAMATH_CALUDE_quadratic_minimum_l783_78383

/-- Given a quadratic function y = x^2 - px + q where the minimum value of y is 1,
    prove that q = 1 + (p^2 / 4) -/
theorem quadratic_minimum (p q : ℝ) : 
  (∀ x, x^2 - p*x + q ≥ 1) ∧ (∃ x, x^2 - p*x + q = 1) → 
  q = 1 + p^2 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l783_78383


namespace NUMINAMATH_CALUDE_x_ln_x_squared_necessary_not_sufficient_l783_78328

theorem x_ln_x_squared_necessary_not_sufficient (x : ℝ) (h1 : 1 < x) (h2 : x < Real.exp 1) :
  (∀ x, x * Real.log x < 1 → x * (Real.log x)^2 < 1) ∧
  (∃ x, x * (Real.log x)^2 < 1 ∧ x * Real.log x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_ln_x_squared_necessary_not_sufficient_l783_78328


namespace NUMINAMATH_CALUDE_bus_travel_fraction_l783_78361

/-- Given a journey with a total distance of 90 kilometers, where 1/5 of the distance is traveled by foot,
    12 kilometers are traveled by car, and the remaining distance is traveled by bus,
    prove that the fraction of the total distance traveled by bus is 2/3. -/
theorem bus_travel_fraction (total_distance : ℝ) (foot_fraction : ℝ) (car_distance : ℝ) :
  total_distance = 90 ∧ foot_fraction = 1/5 ∧ car_distance = 12 →
  (total_distance - foot_fraction * total_distance - car_distance) / total_distance = 2/3 := by
sorry

end NUMINAMATH_CALUDE_bus_travel_fraction_l783_78361


namespace NUMINAMATH_CALUDE_peruvian_coffee_cost_l783_78371

/-- Proves that the cost of Peruvian coffee beans per pound is approximately $2.29 given the specified conditions --/
theorem peruvian_coffee_cost (colombian_cost : ℝ) (total_weight : ℝ) (mix_price : ℝ) (colombian_weight : ℝ) :
  colombian_cost = 5.50 →
  total_weight = 40 →
  mix_price = 4.60 →
  colombian_weight = 28.8 →
  ∃ (peruvian_cost : ℝ), abs (peruvian_cost - 2.29) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_peruvian_coffee_cost_l783_78371


namespace NUMINAMATH_CALUDE_roses_count_l783_78327

def total_roses : ℕ := 500

def red_roses : ℕ := (total_roses * 5) / 8

def remaining_after_red : ℕ := total_roses - red_roses

def yellow_roses : ℕ := remaining_after_red / 8

def pink_roses : ℕ := (remaining_after_red * 2) / 8

def remaining_after_yellow_pink : ℕ := remaining_after_red - yellow_roses - pink_roses

def white_roses : ℕ := remaining_after_yellow_pink / 2

def purple_roses : ℕ := remaining_after_yellow_pink / 2

theorem roses_count : red_roses + white_roses + purple_roses = 430 := by
  sorry

end NUMINAMATH_CALUDE_roses_count_l783_78327


namespace NUMINAMATH_CALUDE_circle_and_line_equations_l783_78331

-- Define the circles and points
def circle_O (x y : ℝ) := x^2 + y^2 = 16
def circle_C (x y : ℝ) := (x + 1)^2 + (y + 1)^2 = 2
def point_P : ℝ × ℝ := (-4, 0)

-- Define the line l
def line_l (x y : ℝ) := (3 * x - y = 0) ∨ (3 * x - y + 4 = 0)

-- Define the conditions
def condition_P_on_O := circle_O point_P.1 point_P.2

-- Theorem statement
theorem circle_and_line_equations :
  ∃ (A B M N : ℝ × ℝ),
    -- Line l intersects circle O at A and B
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    -- Line l intersects circle C at M and N
    circle_C M.1 M.2 ∧ circle_C N.1 N.2 ∧
    line_l M.1 M.2 ∧ line_l N.1 N.2 ∧
    -- M is the midpoint of AB
    M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2 ∧
    -- |PM| = |PN|
    (M.1 - point_P.1)^2 + (M.2 - point_P.2)^2 =
    (N.1 - point_P.1)^2 + (N.2 - point_P.2)^2 ∧
    -- Point P is on circle O
    condition_P_on_O :=
  by sorry

end NUMINAMATH_CALUDE_circle_and_line_equations_l783_78331


namespace NUMINAMATH_CALUDE_certain_number_proof_l783_78334

theorem certain_number_proof (x : ℝ) : 
  (x + 40 + 60) / 3 = (10 + 80 + 15) / 3 + 5 → x = 20 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l783_78334


namespace NUMINAMATH_CALUDE_line_equation_proof_l783_78374

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 2*x - 3

-- Define the midpoint of AB
def midpoint_AB : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem line_equation_proof :
  ∀ (A B : ℝ × ℝ),
  parabola_C A.1 A.2 →
  parabola_C B.1 B.2 →
  (A.1 + B.1) / 2 = midpoint_AB.1 →
  (A.2 + B.2) / 2 = midpoint_AB.2 →
  line_l A.1 A.2 ∧ line_l B.1 B.2 :=
by sorry


end NUMINAMATH_CALUDE_line_equation_proof_l783_78374


namespace NUMINAMATH_CALUDE_exists_acute_triangle_l783_78308

/-- A set of 5 positive real numbers representing segment lengths -/
def SegmentSet : Type := Fin 5 → ℝ

/-- Predicate to check if three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate to check if a triangle is acute-angled -/
def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- Main theorem: Given 5 segments where any three can form a triangle,
    there exists at least one acute-angled triangle -/
theorem exists_acute_triangle (s : SegmentSet) 
  (h_positive : ∀ i, s i > 0)
  (h_triangle : ∀ i j k, i ≠ j → j ≠ k → k ≠ i → can_form_triangle (s i) (s j) (s k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ is_acute_triangle (s i) (s j) (s k) := by
  sorry


end NUMINAMATH_CALUDE_exists_acute_triangle_l783_78308


namespace NUMINAMATH_CALUDE_polynomial_value_l783_78339

/-- A polynomial with integer coefficients where each coefficient is between 0 and 3 (inclusive) -/
def IntPolynomial (n : ℕ) := { p : Polynomial ℤ // ∀ i, 0 ≤ p.coeff i ∧ p.coeff i < 4 }

/-- The theorem stating that if P(2) = 66, then P(3) = 111 for the given polynomial -/
theorem polynomial_value (n : ℕ) (P : IntPolynomial n) 
  (h : P.val.eval 2 = 66) : P.val.eval 3 = 111 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l783_78339


namespace NUMINAMATH_CALUDE_planes_parallel_transitivity_l783_78390

-- Define non-coincident planes
variable (α β γ : Plane)
variable (h_distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Define parallel relation
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_transitivity 
  (h1 : parallel α γ) 
  (h2 : parallel β γ) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_transitivity_l783_78390


namespace NUMINAMATH_CALUDE_class_average_l783_78320

theorem class_average (total_students : ℕ) (high_scorers : ℕ) (zero_scorers : ℕ) (high_score : ℕ) (rest_average : ℚ) : 
  total_students = 25 →
  high_scorers = 5 →
  zero_scorers = 3 →
  high_score = 95 →
  rest_average = 45 →
  let rest_students := total_students - high_scorers - zero_scorers
  let total_score := (high_scorers * high_score + rest_students * rest_average)
  (total_score / total_students : ℚ) = 49.6 := by
sorry

end NUMINAMATH_CALUDE_class_average_l783_78320


namespace NUMINAMATH_CALUDE_equation_solution_l783_78380

theorem equation_solution (a b c : ℤ) :
  (∀ x : ℝ, (x - a)*(x - 5) + 2 = (x + b)*(x + c)) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l783_78380


namespace NUMINAMATH_CALUDE_least_months_to_triple_l783_78305

def interest_factor : ℝ := 1.06

def exceeds_triple (t : ℕ) : Prop :=
  interest_factor ^ t > 3

theorem least_months_to_triple : ∃ (t : ℕ), t = 20 ∧ exceeds_triple t ∧ ∀ (k : ℕ), k < t → ¬exceeds_triple k :=
sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l783_78305


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l783_78376

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles (carol_rect jordan_rect : Rectangle) 
  (h1 : carol_rect.length = 5)
  (h2 : carol_rect.width = 24)
  (h3 : jordan_rect.length = 4)
  (h4 : area carol_rect = area jordan_rect) :
  jordan_rect.width = 30 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l783_78376


namespace NUMINAMATH_CALUDE_infinite_solutions_cube_equation_l783_78304

theorem infinite_solutions_cube_equation :
  ∃ f : ℕ → ℕ × ℕ × ℕ × ℕ, ∀ m : ℕ,
    let (n, x, y, z) := f m
    n > m ∧ (x + y + z)^3 = n^2 * x * y * z :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_cube_equation_l783_78304


namespace NUMINAMATH_CALUDE_reach_target_probability_l783_78370

-- Define the number of lily pads
def num_pads : ℕ := 16

-- Define the starting pad
def start_pad : ℕ := 2

-- Define the target pad
def target_pad : ℕ := 14

-- Define the predator pads
def predator_pads : List ℕ := [4, 9]

-- Define the movement probabilities
def move_prob : ℚ := 1/3
def skip_one_prob : ℚ := 1/3
def skip_two_prob : ℚ := 1/3

-- Function to calculate the probability of reaching the target pad
def reach_target_prob : ℚ := sorry

-- Theorem stating the probability of reaching the target pad
theorem reach_target_probability :
  reach_target_prob = 1/729 :=
sorry

end NUMINAMATH_CALUDE_reach_target_probability_l783_78370


namespace NUMINAMATH_CALUDE_expression_factorization_l783_78366

theorem expression_factorization (a b c : ℝ) : 
  2*(a+b)*(b+c)*(a+3*b+2*c) + 2*(b+c)*(c+a)*(b+3*c+2*a) + 
  2*(c+a)*(a+b)*(c+3*a+2*b) + 9*(a+b)*(b+c)*(c+a) = 
  (a + 3*b + 2*c)*(b + 3*c + 2*a)*(c + 3*a + 2*b) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l783_78366


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l783_78309

theorem imaginary_part_of_z_plus_reciprocal (z : ℂ) (h : z = 1 + I) :
  (z + z⁻¹).im = 1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l783_78309


namespace NUMINAMATH_CALUDE_x_plus_y_equals_four_l783_78378

/-- Geometric configuration with segments AB and A'B' --/
structure GeometricConfiguration where
  AB : ℝ
  APB : ℝ
  P_distance_from_D : ℝ
  total_distance : ℝ

/-- Theorem stating that x + y = 4 in the given geometric configuration --/
theorem x_plus_y_equals_four (config : GeometricConfiguration) 
  (h1 : config.AB = 6)
  (h2 : config.APB = 10)
  (h3 : config.P_distance_from_D = 2)
  (h4 : config.total_distance = 12) :
  let D := config.AB / 2
  let D' := config.APB / 2
  let x := config.P_distance_from_D
  let y := config.total_distance - (D + x + D')
  x + y = 4 := by
  sorry


end NUMINAMATH_CALUDE_x_plus_y_equals_four_l783_78378


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_range_l783_78388

/-- A cubic function with three distinct real roots -/
structure CubicFunction where
  m : ℝ
  n : ℝ
  p : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  has_three_roots : a ≠ b ∧ b ≠ c ∧ a ≠ c
  is_root_a : a^3 + m*a^2 + n*a + p = 0
  is_root_b : b^3 + m*b^2 + n*b + p = 0
  is_root_c : c^3 + m*c^2 + n*c + p = 0
  neg_one_eq_two : ((-1)^3 + m*(-1)^2 + n*(-1) + p) = (2^3 + m*2^2 + n*2 + p)
  one_eq_four : (1^3 + m*1^2 + n*1 + p) = (4^3 + m*4^2 + n*4 + p)
  neg_one_neg : ((-1)^3 + m*(-1)^2 + n*(-1) + p) < 0
  one_pos : (1^3 + m*1^2 + n*1 + p) > 0

/-- The main theorem stating the range of the sum of reciprocals of roots -/
theorem sum_of_reciprocals_range (f : CubicFunction) :
  -(3/4) < (1/f.a + 1/f.b + 1/f.c) ∧ (1/f.a + 1/f.b + 1/f.c) < -(3/14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_range_l783_78388
