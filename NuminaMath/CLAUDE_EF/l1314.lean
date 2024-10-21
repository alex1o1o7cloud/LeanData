import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_array_sum_is_32_over_21_l1314_131447

/-- Represents a 1/4-array where each row starts with 1/(2p) times the first entry of the previous row,
    and each succeeding term in a row is 1/p times the previous term, where p = 4. -/
def QuarterArray : Type := ℕ → ℕ → ℚ

/-- The general term of the 1/4-array at row r and column c. -/
def quarterArrayTerm (r c : ℕ) : ℚ := 1 / ((8 ^ r) * (4 ^ c))

/-- The sum of all terms in the 1/4-array. -/
noncomputable def quarterArraySum : ℚ := ∑' (r : ℕ), ∑' (c : ℕ), quarterArrayTerm r c

/-- Theorem stating that the sum of all terms in the 1/4-array is 32/21. -/
theorem quarter_array_sum_is_32_over_21 : quarterArraySum = 32 / 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_array_sum_is_32_over_21_l1314_131447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_german_hate_percentage_l1314_131475

/-- The percentage of students who hate a subject -/
def hate_percentage : ℝ := 0

/-- The percentage of students who hate math -/
def hate_math : ℝ := 1

/-- The percentage of students who hate English -/
def hate_english : ℝ := 2

/-- The percentage of students who hate French -/
def hate_french : ℝ := 1

/-- The percentage of students who hate all 4 subjects -/
def hate_all : ℝ := 8

/-- The percentage of students who hate German -/
def hate_german : ℝ := hate_all - (hate_math + hate_english + hate_french)

theorem german_hate_percentage : hate_german = 4 := by
  unfold hate_german
  unfold hate_all hate_math hate_english hate_french
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_german_hate_percentage_l1314_131475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_comparison_l1314_131418

/-- Represents the final salary after two percentage increases -/
noncomputable def final_salary (first_increase second_increase : ℝ) : ℝ :=
  (1 + first_increase / 100) * (1 + second_increase / 100)

theorem salary_increase_comparison (a b : ℝ) (ha : a > b) (hb : b > 0) :
  let plan_a := final_salary a b
  let plan_b := final_salary ((a + b) / 2) ((a + b) / 2)
  let plan_c := final_salary (Real.sqrt (a * b)) (Real.sqrt (a * b))
  plan_b > plan_a ∧ plan_b > plan_c := by
  sorry

#check salary_increase_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_comparison_l1314_131418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_time_calculation_l1314_131445

/-- The time (in hours) that Mary and Paul remained on the highway --/
noncomputable def highway_time (mary_speed paul_speed : ℝ) (time_diff catch_up_time : ℝ) : ℝ :=
  (time_diff + catch_up_time) / 60

theorem highway_time_calculation :
  let mary_speed : ℝ := 50
  let paul_speed : ℝ := 80
  let time_diff : ℝ := 15
  let catch_up_time : ℝ := 25
  highway_time mary_speed paul_speed time_diff catch_up_time = 2/3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval highway_time 50 80 15 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_time_calculation_l1314_131445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_plane_region_l1314_131451

/-- Defines a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the plane region 3x + 2y - 1 ≥ 0 -/
def inPlaneRegion (p : Point) : Prop :=
  3 * p.x + 2 * p.y - 1 ≥ 0

/-- Given points -/
noncomputable def P₁ : Point := ⟨0, 0⟩
noncomputable def P₂ : Point := ⟨1, 1⟩
noncomputable def P₃ : Point := ⟨1/3, 0⟩

/-- Theorem stating which points are in the plane region -/
theorem points_in_plane_region :
  ¬inPlaneRegion P₁ ∧ inPlaneRegion P₂ ∧ inPlaneRegion P₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_plane_region_l1314_131451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_green_red_last_l1314_131436

def total_marbles : ℕ := 10
def blue_marbles : ℕ := 4
def white_marbles : ℕ := 3
def red_marbles : ℕ := 2
def green_marbles : ℕ := 1

def marbles_drawn : ℕ := 8

theorem probability_green_red_last :
  (Nat.choose total_marbles marbles_drawn : ℚ)⁻¹ *
  (Nat.choose (blue_marbles + white_marbles) (marbles_drawn - red_marbles - green_marbles) : ℚ) =
  28 / 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_green_red_last_l1314_131436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marker_buyers_difference_l1314_131437

/-- Represents the price of a marker in cents -/
def marker_price : ℕ := sorry

/-- Represents the number of fifth graders who bought markers -/
def fifth_graders : ℕ := sorry

/-- Represents the number of fourth graders who bought markers -/
def fourth_graders : ℕ := sorry

/-- The total amount spent by fifth graders in cents -/
def fifth_graders_total : ℕ := 180

/-- The total amount spent by fourth graders in cents -/
def fourth_graders_total : ℕ := 200

/-- The number of fourth graders who bought markers -/
axiom fourth_graders_count : fourth_graders = 25

theorem marker_buyers_difference :
  marker_price > 0 ∧
  marker_price * fifth_graders = fifth_graders_total ∧
  marker_price * fourth_graders = fourth_graders_total →
  fourth_graders - fifth_graders = 1 := by
  sorry

#check marker_buyers_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marker_buyers_difference_l1314_131437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_speed_proof_l1314_131497

/-- Linda's walking speed in miles per hour -/
noncomputable def Linda_speed : ℝ := 4.7

/-- Tom's jogging speed in miles per hour -/
noncomputable def Tom_speed : ℝ := 9

/-- Time difference in hours -/
noncomputable def time_difference : ℝ := 216 / 60

/-- Time for Tom to cover half of Linda's distance -/
noncomputable def half_time (L : ℝ) : ℝ := L / (18 - L)

/-- Time for Tom to cover twice Linda's distance -/
noncomputable def double_time (L : ℝ) : ℝ := (2 * L) / (9 - 2 * L)

theorem linda_speed_proof :
  ∃ L : ℝ, L > 0 ∧ L < 9 ∧
  (double_time L - half_time L = time_difference) ∧
  L = Linda_speed := by
  use Linda_speed
  simp [Linda_speed, Tom_speed, time_difference, half_time, double_time]
  norm_num
  sorry

#check linda_speed_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_speed_proof_l1314_131497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ski_helmet_discount_l1314_131400

/-- Given an item with an original price of $120, after a 40% discount
    followed by a 20% discount, the final price is $57.60. -/
theorem ski_helmet_discount (original_price : ℝ) (friday_discount : ℝ) (monday_discount : ℝ)
  (h1 : original_price = 120)
  (h2 : friday_discount = 0.4)
  (h3 : monday_discount = 0.2) :
  original_price * (1 - friday_discount) * (1 - monday_discount) = 57.6 :=
by
  -- Replace the proof with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ski_helmet_discount_l1314_131400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_cakes_l1314_131487

theorem baker_cakes (cakes : ℕ → ℕ) : 
  (∀ n : ℕ, n ∈ Finset.range 5 → cakes (n + 1) = 2 * cakes n) →
  cakes 6 = 320 →
  cakes 1 = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_cakes_l1314_131487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_d_value_l1314_131416

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  tangentX : Point
  tangentY : Point

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the value of d for the given ellipse -/
theorem ellipse_focus_d_value (e : Ellipse) :
  e.focus1 = Point.mk 3 7 →
  e.focus2 = Point.mk 3 (29.8 : ℝ) →
  e.tangentX.y = 0 →
  e.tangentY.x = 0 →
  e.tangentX.x > 0 →
  e.tangentY.y > 0 →
  distance e.tangentY e.focus1 + distance e.tangentY e.focus2 =
  distance e.tangentX e.focus1 + distance e.tangentX e.focus2 :=
by sorry

#check ellipse_focus_d_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_d_value_l1314_131416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_diagonal_contains_all_numbers_l1314_131431

def is_valid_matrix {n : ℕ} (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∀ i j, A i j ∈ Finset.range n

def has_all_numbers {n : ℕ} (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∀ i, (Finset.range n).card = (Finset.image (λ j ↦ A i j) Finset.univ).card ∧
       (Finset.range n).card = (Finset.image (λ j ↦ A j i) Finset.univ).card

def is_symmetric {n : ℕ} (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∀ i j, A i j = A j i

theorem main_diagonal_contains_all_numbers 
  {n : ℕ} 
  (h_odd : Odd n) 
  (A : Matrix (Fin n) (Fin n) ℕ) 
  (h_valid : is_valid_matrix A) 
  (h_all : has_all_numbers A) 
  (h_sym : is_symmetric A) : 
  (Finset.range n).card = (Finset.image (λ i ↦ A i i) Finset.univ).card :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_diagonal_contains_all_numbers_l1314_131431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_point_l1314_131401

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ := x^2

-- Define the tangent line equation
noncomputable def tangent_line (m : ℝ) (x : ℝ) : ℝ := m^2 * (x - m) + f m

-- Theorem statement
theorem tangent_line_through_point :
  ∃ (m : ℝ), (tangent_line m 2 = 4) ∧
  ((∀ (x : ℝ), tangent_line m x = 4*x - 4) ∨
   (∀ (x : ℝ), tangent_line m x = x + 2)) := by
  sorry

#check tangent_line_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_point_l1314_131401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_example_l1314_131434

def original_number : ℚ := 40925 / 1000000

-- Function to round a rational number to the nearest hundredth
def round_to_hundredth (x : ℚ) : ℚ :=
  (x * 100).floor / 100

-- Theorem stating that rounding 0.040925 to the nearest hundredth results in 0.04
theorem round_to_hundredth_example : round_to_hundredth original_number = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_example_l1314_131434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1314_131442

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- The focal distance of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

theorem ellipse_equation (e : Ellipse) 
  (h_focus : focal_distance e = 1)
  (h_ecc : eccentricity e = 1/2) :
  e.a = 2 ∧ e.b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1314_131442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l1314_131406

theorem congruence_solutions_count :
  let solutions := {x : ℕ | x < 100 ∧ x > 0 ∧ (x + 17) % 38 = 63 % 38}
  Finset.card (Finset.filter (fun x => x ∈ solutions) (Finset.range 100)) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l1314_131406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_surface_area_l1314_131414

/-- The surface area of a regular hexagonal pyramid -/
noncomputable def surface_area_hexagonal_pyramid (base_edge : ℝ) (side_edge : ℝ) : ℝ :=
  let slant_height := Real.sqrt (side_edge^2 - (base_edge/2)^2)
  6 * (Real.sqrt 3 / 4 * base_edge^2 + 1/2 * base_edge * slant_height)

/-- Theorem: The surface area of a regular hexagonal pyramid with base edge length 2 and side edge length √5 is 6√3 + 12 -/
theorem hexagonal_pyramid_surface_area :
  surface_area_hexagonal_pyramid 2 (Real.sqrt 5) = 6 * Real.sqrt 3 + 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_surface_area_l1314_131414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_slant_height_l1314_131490

/-- The slant height of a frustum of a cone. -/
noncomputable def slant_height (r₁ r₂ : ℝ) : ℝ :=
  5

/-- The lateral surface area of a frustum of a cone. -/
noncomputable def lateral_area (r₁ r₂ l : ℝ) : ℝ :=
  Real.pi * l * (r₁ + r₂)

/-- The sum of the areas of the two bases of a frustum of a cone. -/
noncomputable def bases_area (r₁ r₂ : ℝ) : ℝ :=
  Real.pi * (r₁^2 + r₂^2)

/-- 
Theorem: The slant height of a frustum of a cone is 5, given that:
- The radius of the top face is 3.
- The radius of the bottom face is 4.
- The lateral surface area is equal to the sum of the areas of the two bases.
-/
theorem frustum_slant_height :
  let r₁ := (3 : ℝ)
  let r₂ := (4 : ℝ)
  let l := slant_height r₁ r₂
  lateral_area r₁ r₂ l = bases_area r₁ r₂ →
  l = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_slant_height_l1314_131490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intersection_range_l1314_131477

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define the point Q
def Q : ℝ × ℝ := (0, 1)

-- Define the trajectory E
def E : Set (ℝ × ℝ) := {p | p.1^2 / 4 + p.2^2 / 3 = 1}

-- Define a function to represent a line through Q
def line_through_Q (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + 1}

-- Main theorem
theorem trajectory_and_intersection_range :
  -- Part 1: Prove that E is the trajectory of P
  (∀ M : ℝ × ℝ, M ∈ C → ∃ P : ℝ × ℝ, P ∈ E ∧ ∃ N : ℝ × ℝ,
    N.2 = 0 ∧ N.1 = M.1 ∧
    (P.1 - N.1)^2 + (P.2 - N.2)^2 = 3/4 * ((M.1 - N.1)^2 + (M.2 - N.2)^2)) ∧
  
  -- Part 2: Prove the range of |AB|*|ST|
  (∀ k : ℝ, ∀ A B S T : ℝ × ℝ,
    A ∈ E → B ∈ E → S ∈ C → T ∈ C →
    A ∈ line_through_Q k → B ∈ line_through_Q k →
    S ∈ line_through_Q k → T ∈ line_through_Q k →
    A ≠ B → S ≠ T →
    8 * Real.sqrt 2 ≤ (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) *
                       Real.sqrt ((S.1 - T.1)^2 + (S.2 - T.2)^2)) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) *
     Real.sqrt ((S.1 - T.1)^2 + (S.2 - T.2)^2)) ≤ 8 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intersection_range_l1314_131477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l1314_131403

/-- Represents a birthday with day and month -/
structure Birthday where
  day : ℕ
  month : ℕ

/-- Check if two positive integers are consecutive -/
def are_consecutive (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (a = b + 1 ∨ b = a + 1)

/-- The problem statement -/
theorem birthday_problem (olena mykola : Birthday) : 
  (olena.day = mykola.day / 2) →
  (olena.month = mykola.month / 2) →
  (are_consecutive olena.day mykola.month) →
  ((olena.day + olena.month + mykola.day + mykola.month) % 17 = 0) →
  olena.day = 11 ∧ olena.month = 6 := by
  sorry

#check birthday_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l1314_131403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_area_rectangles_l1314_131457

/-- A rectangle with integer dimensions -/
structure IntRectangle where
  length : ℕ
  width : ℕ

/-- A 3x3 grid of rectangles -/
def Grid := Array (Array IntRectangle)

/-- Check if a rectangle has an odd area -/
def hasOddArea (r : IntRectangle) : Bool :=
  r.length % 2 = 1 ∧ r.width % 2 = 1

/-- Count the number of rectangles with odd area in a grid -/
def countOddAreaRectangles (g : Grid) : ℕ :=
  g.foldl (init := 0) (fun acc row =>
    acc + row.foldl (init := 0) (fun innerAcc r =>
      innerAcc + if hasOddArea r then 1 else 0))

/-- Theorem: The maximum number of rectangles with odd area in a 3x3 grid is 4 -/
theorem max_odd_area_rectangles (g : Grid) :
    countOddAreaRectangles g ≤ 4 := by
  sorry

#check max_odd_area_rectangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_area_rectangles_l1314_131457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l1314_131429

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 4*x + 6 else -x + 6

-- Define the solution set
def solution_set : Set ℝ := {x | f x < f (-1)}

-- Theorem statement
theorem solution_set_equality : 
  solution_set = (Set.Ioo (-3) (-1)) ∪ (Set.Ioi 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l1314_131429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_three_halves_pi_l1314_131441

theorem sin_alpha_minus_three_halves_pi (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (-α - π) = Real.sqrt 5 / 5) : 
  Real.sin (α - 3/2 * π) = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_three_halves_pi_l1314_131441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_represents_frequency_l1314_131476

/-- Represents a frequency distribution histogram --/
structure FrequencyHistogram where
  rectangles : List (ℝ × ℕ)

/-- The area of each rectangle in a frequency histogram represents its frequency --/
theorem area_represents_frequency (h : FrequencyHistogram) :
  ∀ r ∈ h.rectangles, r.1 = r.2 := by
  sorry

#check area_represents_frequency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_represents_frequency_l1314_131476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_250_l1314_131454

/-- The length of a platform crossed by a train -/
noncomputable def platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  train_speed_mps * crossing_time - train_length

/-- Theorem: The length of the platform is 250 meters -/
theorem platform_length_is_250 :
  platform_length 250 90 20 = 250 := by
  -- Unfold the definition of platform_length
  unfold platform_length
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_250_l1314_131454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1314_131452

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 1

-- Define point M
def point_M : ℝ × ℝ := (0, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    distance point_M A + distance point_M B = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1314_131452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_tree_decoration_l1314_131449

/-- The number of branches allocated to each son -/
def branches : ℕ := sorry

/-- The number of ornaments allocated to each son -/
def ornaments : ℕ := sorry

/-- Chuck's attempt: one ornament per branch, short one branch -/
axiom chuck_attempt : branches = ornaments - 1

/-- Huck's attempt: two ornaments per branch, one branch empty -/
axiom huck_attempt : 2 * branches = ornaments + 1

theorem christmas_tree_decoration :
  branches = 3 ∧ ornaments = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_tree_decoration_l1314_131449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_after_n_years_l1314_131478

/-- The number of years after 2001 when the price difference between commodities X and Y is 65 cents -/
noncomputable def years_until_price_difference (a b : ℝ) : ℝ :=
  2.75 / (0.25 + a - b)

/-- The price of commodity X after n years -/
def price_x (n a : ℝ) : ℝ :=
  4.20 + 0.45 * n + a * n

/-- The price of commodity Y after n years -/
def price_y (n b : ℝ) : ℝ :=
  6.30 + 0.20 * n + b * n

theorem price_difference_after_n_years (a b : ℝ) :
  let n := years_until_price_difference a b
  price_x n a = price_y n b + 0.65 := by
  sorry

#check price_difference_after_n_years

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_after_n_years_l1314_131478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_millionaire_allocation_l1314_131460

/-- Represents the number of millionaires -/
def num_millionaires : ℕ := 13

/-- Represents the number of room types -/
def num_room_types : ℕ := 3

/-- Represents the number of ways to allocate millionaires to rooms -/
def num_allocations : ℕ := 36

/-- Represents the wealth of a millionaire -/
noncomputable def wealth : Fin num_millionaires → ℝ := sorry

/-- Represents the room type assigned to a millionaire -/
noncomputable def room_type : Fin num_millionaires → Fin num_room_types := sorry

/-- Theorem stating the number of ways to allocate millionaires to rooms -/
theorem millionaire_allocation :
  (num_millionaires = 13) →
  (num_room_types = 3) →
  (∀ m₁ m₂ : Fin num_millionaires, m₁ ≠ m₂ → (wealth m₁ ≠ wealth m₂)) →
  (∀ m₁ m₂ : Fin num_millionaires, wealth m₁ > wealth m₂ → room_type m₁ ≥ room_type m₂) →
  (∀ t : Fin num_room_types, ∃ m : Fin num_millionaires, room_type m = t) →
  num_allocations = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_millionaire_allocation_l1314_131460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_iterative_average_difference_l1314_131443

noncomputable def modifiedIterativeAverage (a b c d e : ℕ) : ℝ :=
  let step1 := (a + 1 + b ^ 2 : ℝ) / 2
  let step2 := (step1 + Nat.factorial c) / 2
  let step3 := (step2 + d) / 2
  (step3 + e) / 2

def allPermutations : List (ℕ × ℕ × ℕ × ℕ × ℕ) :=
  [(1, 2, 3, 4, 6), (1, 2, 3, 6, 4), (1, 2, 4, 3, 6), (1, 2, 4, 6, 3),
   (1, 2, 6, 3, 4), (1, 2, 6, 4, 3), (1, 3, 2, 4, 6), (1, 3, 2, 6, 4),
   (1, 3, 4, 2, 6), (1, 3, 4, 6, 2), (1, 3, 6, 2, 4), (1, 3, 6, 4, 2),
   (1, 4, 2, 3, 6), (1, 4, 2, 6, 3), (1, 4, 3, 2, 6), (1, 4, 3, 6, 2),
   (1, 4, 6, 2, 3), (1, 4, 6, 3, 2), (1, 6, 2, 3, 4), (1, 6, 2, 4, 3),
   (1, 6, 3, 2, 4), (1, 6, 3, 4, 2), (1, 6, 4, 2, 3), (1, 6, 4, 3, 2),
   (2, 1, 3, 4, 6), (2, 1, 3, 6, 4), (2, 1, 4, 3, 6), (2, 1, 4, 6, 3),
   (2, 1, 6, 3, 4), (2, 1, 6, 4, 3), (2, 3, 1, 4, 6), (2, 3, 1, 6, 4),
   (2, 3, 4, 1, 6), (2, 3, 4, 6, 1), (2, 3, 6, 1, 4), (2, 3, 6, 4, 1),
   (2, 4, 1, 3, 6), (2, 4, 1, 6, 3), (2, 4, 3, 1, 6), (2, 4, 3, 6, 1),
   (2, 4, 6, 1, 3), (2, 4, 6, 3, 1), (2, 6, 1, 3, 4), (2, 6, 1, 4, 3),
   (2, 6, 3, 1, 4), (2, 6, 3, 4, 1), (2, 6, 4, 1, 3), (2, 6, 4, 3, 1),
   (3, 1, 2, 4, 6), (3, 1, 2, 6, 4), (3, 1, 4, 2, 6), (3, 1, 4, 6, 2),
   (3, 1, 6, 2, 4), (3, 1, 6, 4, 2), (3, 2, 1, 4, 6), (3, 2, 1, 6, 4),
   (3, 2, 4, 1, 6), (3, 2, 4, 6, 1), (3, 2, 6, 1, 4), (3, 2, 6, 4, 1),
   (3, 4, 1, 2, 6), (3, 4, 1, 6, 2), (3, 4, 2, 1, 6), (3, 4, 2, 6, 1),
   (3, 4, 6, 1, 2), (3, 4, 6, 2, 1), (3, 6, 1, 2, 4), (3, 6, 1, 4, 2),
   (3, 6, 2, 1, 4), (3, 6, 2, 4, 1), (3, 6, 4, 1, 2), (3, 6, 4, 2, 1),
   (4, 1, 2, 3, 6), (4, 1, 2, 6, 3), (4, 1, 3, 2, 6), (4, 1, 3, 6, 2),
   (4, 1, 6, 2, 3), (4, 1, 6, 3, 2), (4, 2, 1, 3, 6), (4, 2, 1, 6, 3),
   (4, 2, 3, 1, 6), (4, 2, 3, 6, 1), (4, 2, 6, 1, 3), (4, 2, 6, 3, 1),
   (4, 3, 1, 2, 6), (4, 3, 1, 6, 2), (4, 3, 2, 1, 6), (4, 3, 2, 6, 1),
   (4, 3, 6, 1, 2), (4, 3, 6, 2, 1), (4, 6, 1, 2, 3), (4, 6, 1, 3, 2),
   (4, 6, 2, 1, 3), (4, 6, 2, 3, 1), (4, 6, 3, 1, 2), (4, 6, 3, 2, 1),
   (6, 1, 2, 3, 4), (6, 1, 2, 4, 3), (6, 1, 3, 2, 4), (6, 1, 3, 4, 2),
   (6, 1, 4, 2, 3), (6, 1, 4, 3, 2), (6, 2, 1, 3, 4), (6, 2, 1, 4, 3),
   (6, 2, 3, 1, 4), (6, 2, 3, 4, 1), (6, 2, 4, 1, 3), (6, 2, 4, 3, 1),
   (6, 3, 1, 2, 4), (6, 3, 1, 4, 2), (6, 3, 2, 1, 4), (6, 3, 2, 4, 1),
   (6, 3, 4, 1, 2), (6, 3, 4, 2, 1), (6, 4, 1, 2, 3), (6, 4, 1, 3, 2),
   (6, 4, 2, 1, 3), (6, 4, 2, 3, 1), (6, 4, 3, 1, 2), (6, 4, 3, 2, 1)]

theorem modified_iterative_average_difference :
  let results := allPermutations.map (fun (a, b, c, d, e) => modifiedIterativeAverage a b c d e)
  ∃ max min : ℝ, max ∈ results ∧ min ∈ results ∧ max - min = 89.9375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_iterative_average_difference_l1314_131443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_sum_fifty_l1314_131463

theorem ordered_pairs_sum_fifty :
  let S := {p : ℕ × ℕ | p.1 + p.2 = 50 ∧ p.1 > 0 ∧ p.2 > 0}
  Finset.card (Finset.filter (fun p => p.1 + p.2 = 50 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.range 50 ×ˢ Finset.range 50)) = 49 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_sum_fifty_l1314_131463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1314_131499

-- Define the hyperbola and parabola
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 - y^2/m = 1
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the point of intersection P and the focus F
noncomputable def P : ℝ × ℝ := ⟨3, 2 * Real.sqrt 6⟩
noncomputable def F : ℝ × ℝ := ⟨2, 0⟩

-- Distance between P and F is 5
axiom PF_distance : dist P F = 5

-- P lies on both the hyperbola and parabola
axiom P_on_hyperbola (m : ℝ) : hyperbola m P.1 P.2
axiom P_on_parabola : parabola P.1 P.2

-- Define the asymptote equation
def is_asymptote (a b : ℝ) : Prop := ∀ x y, a*x + b*y = 0

-- Theorem statement
theorem hyperbola_asymptotes :
  ∃ m, hyperbola m P.1 P.2 ∧ is_asymptote (Real.sqrt 3) 1 ∧ is_asymptote (Real.sqrt 3) (-1) :=
by
  -- We know m = 3 from the problem solution
  let m := 3
  use m
  apply And.intro
  · exact P_on_hyperbola m
  · apply And.intro
    · -- Proof for the first asymptote
      sorry
    · -- Proof for the second asymptote
      sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1314_131499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_equality_l1314_131455

/-- α(n) is the number of compositions of n using 1's and 2's -/
def α : ℕ+ → ℕ := sorry

/-- β(n) is the number of compositions of n using integers greater than 1 -/
def β : ℕ+ → ℕ := sorry

/-- For all positive integers n, α(n) equals β(n+2) -/
theorem alpha_beta_equality (n : ℕ+) : α n = β (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_equality_l1314_131455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perp_plane_parallel_planes_perp_line_parallel_l1314_131446

-- Define a 3D space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Fact (finrank ℝ V = 3)]

-- Define lines and planes in the space
def Line (V : Type*) := Set V
def Plane (V : Type*) := Set V

-- Define perpendicularity for lines and planes
def perpendicular_line_plane {V : Type*} (l : Line V) (p : Plane V) : Prop := sorry

-- Define parallelism for lines and planes
def parallel_lines {V : Type*} (l1 l2 : Line V) : Prop := sorry
def parallel_planes {V : Type*} (p1 p2 : Plane V) : Prop := sorry

-- Theorem 1: Two lines perpendicular to the same plane are parallel
theorem lines_perp_plane_parallel {V : Type*} (l1 l2 : Line V) (p : Plane V) :
  perpendicular_line_plane l1 p → perpendicular_line_plane l2 p → parallel_lines l1 l2 := by
  sorry

-- Theorem 2: Two planes perpendicular to the same line are parallel
theorem planes_perp_line_parallel {V : Type*} (p1 p2 : Plane V) (l : Line V) :
  perpendicular_line_plane l p1 → perpendicular_line_plane l p2 → parallel_planes p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perp_plane_parallel_planes_perp_line_parallel_l1314_131446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_marked_prices_l1314_131493

/-- Represents the discount rate based on the marked price -/
noncomputable def discount_rate (marked_price : ℝ) : ℝ :=
  if marked_price ≤ 100 then 0.9 else 0.8

/-- Theorem: Given the discount rules and a payment of 90 yuan, 
    the only possible marked prices are 100 yuan or 112.5 yuan -/
theorem possible_marked_prices :
  ∀ (marked_price : ℝ), 
    marked_price > 0 →
    discount_rate marked_price * marked_price = 90 →
    (marked_price = 100 ∨ marked_price = 112.5) :=
by
  intro marked_price h_positive h_equation
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_marked_prices_l1314_131493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minDistance_l1314_131467

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define point A
def A : ℝ × ℝ := (3, 2)

-- Define the focus F of the parabola
noncomputable def F : ℝ × ℝ := (1/2, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the sum of distances from M to A and F
noncomputable def sumDistances (M : ℝ × ℝ) : ℝ :=
  distance M A + distance M F

-- State the theorem
theorem minDistance :
  ∀ M : ℝ × ℝ, parabola M.1 M.2 → sumDistances M ≥ sumDistances (2, 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minDistance_l1314_131467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_vertical_asymptote_l1314_131465

/-- The function g(x) with parameter c -/
noncomputable def g (c : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + c) / (x^2 - x - 12)

/-- Theorem stating the condition for g(x) to have exactly one vertical asymptote -/
theorem g_has_one_vertical_asymptote (c : ℝ) :
  (∃! x, ∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |g c y| > 1/ε) ↔ c = -8 ∨ c = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_vertical_asymptote_l1314_131465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_17_minus_3_problem_l1314_131474

theorem sqrt_17_minus_3_problem (h : 1 < Real.sqrt 17 - 3 ∧ Real.sqrt 17 - 3 < 2) :
  let a := Int.floor (Real.sqrt 17 - 3)
  let b := Real.sqrt 17 - 3 - a
  ((-a)^3 + (b + 4)^2).sqrt = 4 ∨ ((-a)^3 + (b + 4)^2).sqrt = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_17_minus_3_problem_l1314_131474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1314_131473

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := y = -(Real.sqrt 3)/3 * x

/-- The line passes through the origin -/
def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

/-- A point is in the fourth quadrant -/
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- A line is tangent to a circle if it intersects the circle at exactly one point -/
def is_tangent (line : ℝ → ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, circle p.1 p.2 ∧ line p.1 = p.2

theorem tangent_line_to_circle :
  ∃! k : ℝ, 
    (∀ x y, y = k * x → passes_through_origin (λ t => k * t)) ∧
    (is_tangent (λ x => k * x) circle_equation) ∧
    (∃ x y, circle_equation x y ∧ line_equation x y ∧ in_fourth_quadrant x y) ∧
    k = -(Real.sqrt 3)/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1314_131473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_count_l1314_131480

theorem ceiling_sqrt_count : 
  (Finset.filter (fun n : ℕ => ⌈Real.sqrt (n : ℝ)⌉ = 16) (Finset.range 289)).card = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_count_l1314_131480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1314_131404

-- Define the function f(x) and the base a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 - x) / Real.log a + Real.log (x + 3) / Real.log a

-- State the theorem
theorem function_properties (a : ℝ) (h_a : 0 < a ∧ a < 1) :
  -- 1. Domain of f(x) is (-3, 1)
  (∀ x, f a x ≠ 0 → -3 < x ∧ x < 1) ∧
  -- 2. Zeros of f(x) are -1 ± √3
  (∀ x, f a x = 0 ↔ x = -1 - Real.sqrt 3 ∨ x = -1 + Real.sqrt 3) ∧
  -- 3. If minimum value of f(x) is -4, then a = 1/2
  (∃ x, ∀ y, f a y ≥ f a x ∧ f a x = -4) → a = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1314_131404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_4_or_5_not_20_l1314_131433

def count_multiples (n : ℕ) (m : ℕ) : ℕ := n / m

theorem multiples_of_4_or_5_not_20 :
  let total := count_multiples 3010 4 + count_multiples 3010 5 - count_multiples 3010 20
  total = 1204 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_4_or_5_not_20_l1314_131433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_probability_l1314_131484

/-- The number of teams in the tournament -/
def num_teams : ℕ := 40

/-- The total number of games played in the tournament -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- The probability of a team winning any given game -/
def win_probability : ℚ := 1 / 2

/-- The log base 2 of the denominator in the probability fraction -/
def log2_denominator : ℕ := 742

theorem tournament_probability :
  (Nat.factorial num_teams : ℚ) / (2 ^ total_games) =
  (Nat.factorial num_teams : ℚ) / (2 ^ log2_denominator) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_probability_l1314_131484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_score_theorem_l1314_131491

def geography_score : ℕ := 50
def math_score : ℕ := 70
def english_score : ℕ := 66

def history_score : ℚ := (geography_score + math_score + english_score) / 3

theorem total_score_theorem :
  geography_score + math_score + english_score + history_score.ceil = 248 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_score_theorem_l1314_131491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_sum_l1314_131411

def is_permutation (a b c : Fin 9 → ℕ) : Prop :=
  ∀ i : Fin 9, (a i ∈ Finset.range 9) ∧ (b i ∈ Finset.range 9) ∧ (c i ∈ Finset.range 9) ∧
  (∀ j : Fin 9, j ≠ i → a i ≠ a j ∧ b i ≠ b j ∧ c i ≠ c j)

def is_arithmetic_sequence (x y z : ℕ) : Prop := y * 2 = x + z

def sequence_product (x y z : ℕ) : ℕ := x * y * z

theorem smallest_value_of_sum :
  ∀ (a b c : Fin 3 → ℕ),
  (∃ (a' b' c' : Fin 9 → ℕ), is_permutation a' b' c' ∧
    (∀ i : Fin 3, a i = a' i.val) ∧
    (∀ i : Fin 3, b i = b' i.val) ∧
    (∀ i : Fin 3, c i = c' i.val)) →
  is_arithmetic_sequence (a 0) (a 1) (a 2) →
  is_arithmetic_sequence (b 0) (b 1) (b 2) →
  is_arithmetic_sequence (c 0) (c 1) (c 2) →
  270 ≤ sequence_product (a 0) (a 1) (a 2) + 
         sequence_product (b 0) (b 1) (b 2) + 
         sequence_product (c 0) (c 1) (c 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_sum_l1314_131411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1314_131498

/-- Definition of the hyperbola C -/
def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 - y^2/3 = m ∧ m > 0

/-- Definition of the line x = m -/
def vertical_line (m : ℝ) (x : ℝ) : Prop :=
  x = m

/-- Definition of the asymptotes of the hyperbola -/
def asymptotes (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- Definition of the area of triangle OAB -/
noncomputable def triangle_area (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt 3

/-- Definition of a line with non-zero slope -/
def non_vertical_line (k p x y : ℝ) : Prop :=
  y = k * (x - p) ∧ k ≠ 0

/-- Definition of reflection across x-axis -/
def reflect_x (M M' : ℝ × ℝ) : Prop :=
  M'.1 = M.1 ∧ M'.2 = -M.2

/-- Definition of right focus of the hyperbola -/
def right_focus (F : ℝ × ℝ) : Prop :=
  F = (2, 0)

/-- Definition of collinearity -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - B.1) = (C.2 - B.2) * (B.1 - A.1)

theorem hyperbola_properties (m : ℝ) (A B M N M' F : ℝ × ℝ) (k p : ℝ) :
  hyperbola m A.1 A.2 →
  hyperbola m B.1 B.2 →
  vertical_line m A.1 →
  vertical_line m B.1 →
  asymptotes A.1 A.2 →
  asymptotes B.1 B.2 →
  triangle_area A B = Real.sqrt 3 →
  hyperbola m M.1 M.2 →
  hyperbola m N.1 N.2 →
  non_vertical_line k p M.1 M.2 →
  non_vertical_line k p N.1 N.2 →
  reflect_x M M' →
  right_focus F →
  collinear M' F N →
  (m = 1 ∧ p = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1314_131498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_visitors_is_124_l1314_131485

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) : ℕ :=
  let num_sundays := 4
  let num_other_days := 30 - num_sundays
  let total_visitors := num_sundays * sunday_visitors + num_other_days * other_day_visitors
  total_visitors / 30

/-- Proves that the average number of visitors per day is 124 -/
theorem average_visitors_is_124 (sunday_visitors other_day_visitors : ℕ) 
  (h1 : sunday_visitors = 150) (h2 : other_day_visitors = 120) : 
  library_visitors_average sunday_visitors other_day_visitors = 124 := by
  sorry

#eval library_visitors_average 150 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_visitors_is_124_l1314_131485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l1314_131435

theorem find_n : ∃ n : ℝ, 10^n = 10^(-3 : ℝ) * Real.sqrt ((10^81 : ℝ) / (1/10000 : ℝ)) ∧ n = 39.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l1314_131435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1314_131444

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x^2 < 2*x + 3 → |x - 1| ≤ 2) ∧
  (∃ x : ℝ, |x - 1| ≤ 2 ∧ x^2 ≥ 2*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1314_131444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_is_positive_integers_l1314_131421

/-- A set of positive real numbers satisfying certain closure properties -/
def SpecialSet (S : Set ℝ) : Prop :=
  (∀ x ∈ S, x > 0) ∧
  (1 ∈ S) ∧
  (∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S ∧ (x * y) ∈ S)

/-- A subset of S that generates all elements of S \ {1} uniquely -/
def GeneratingSubset (S : Set ℝ) (P : Set ℝ) : Prop :=
  P ⊆ S ∧
  ∀ s ∈ S \ {1}, ∃! (factors : Multiset ℝ),
    (∀ f ∈ factors, f ∈ P) ∧ (factors.prod = s)

/-- The main theorem: S is necessarily the set of positive real numbers that are integers -/
theorem special_set_is_positive_integers (S : Set ℝ) (P : Set ℝ)
  (hS : SpecialSet S) (hP : GeneratingSubset S P) :
  S = {x : ℝ | x > 0 ∧ ∃ n : ℕ, x = n} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_is_positive_integers_l1314_131421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_g_l1314_131492

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * Real.sqrt x

-- State the theorem
theorem evaluate_g : 2 * g 3 + g 7 = 67 - 4 * Real.sqrt 3 - 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_g_l1314_131492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1314_131409

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 - 1 / x

-- State the theorem
theorem tangent_slope_at_one :
  let f' := fun x => deriv f x
  f' 1 = 1 := by
  -- The proof goes here
  sorry

#check tangent_slope_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1314_131409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_remainder_value_l1314_131425

theorem max_remainder_value (a b c p m k : ℕ) : 
  (a > 0) → 
  (b > 0) → 
  (c > 0) → 
  (Nat.Prime p) → 
  (∀ n : ℕ, (a^n * (b + c) + b^n * (a + c) + c^n * (a + b)) % p = 8) → 
  (m = (a^p + b^p + c^p) % p) → 
  (k = m^p % p^4) → 
  (k ≤ 399) ∧ (∃ (a b c p m k : ℕ), k = 399) := by
  sorry

#check max_remainder_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_remainder_value_l1314_131425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_two_lines_in_plane_l1314_131469

-- Define the necessary structures
structure Line where

structure Plane where

-- Define perpendicularity relation between lines
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define a line being in a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Define relationships between a line and a plane
inductive LineToPlaneRelation where
| perpendicular : LineToPlaneRelation
| parallel : LineToPlaneRelation
| intersecting : LineToPlaneRelation

-- State the theorem
theorem line_perpendicular_to_two_lines_in_plane 
  (l : Line) (α : Plane) (l1 l2 : Line) 
  (h1 : line_in_plane l1 α) 
  (h2 : line_in_plane l2 α) 
  (h3 : perpendicular l l1) 
  (h4 : perpendicular l l2) :
  ∃ r : LineToPlaneRelation, true := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_two_lines_in_plane_l1314_131469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_real_zeros_product_smallest_l1314_131472

def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem non_real_zeros_product_smallest 
  (a b c d : ℝ) 
  (h1 : P a b c d (-1) = 4)
  (h2 : P a b c d 0 > 5)
  (h3 : P a b c d 1 > 1.5)
  (h4 : -a > 4)
  (h5 : ∃ (r1 r2 : ℝ), r1 * r2 < 6 ∧ (∀ x : ℝ, (x - r1) * (x - r2) = 0 → P a b c d x = 0)) :
  ∃ (z1 z2 : ℂ), Complex.abs (z1 * z2) < 6/5 ∧ 
    (∀ (x : ℂ), x^4 + a*x^3 + b*x^2 + c*x + d = 0 → x = z1 ∨ x = z2 ∨ x.im = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_real_zeros_product_smallest_l1314_131472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_125_equal_to_75_l1314_131456

/-- The percentage of 125 that is equal to 75 is 60%. -/
theorem percentage_of_125_equal_to_75 : ∃ (p : ℝ), p * 125 = 75 ∧ p = 60 / 100 := by
  -- Define the percentage as a real number
  let percentage : ℝ := 60 / 100
  
  -- Prove that this percentage satisfies the condition
  have h1 : percentage * 125 = 75 := by
    -- Calculation
    calc
      percentage * 125 = (60 / 100) * 125 := rfl
      _ = 60 * 125 / 100 := by ring
      _ = 7500 / 100 := by ring
      _ = 75 := by norm_num

  -- Prove that the percentage is equal to 60/100
  have h2 : percentage = 60 / 100 := rfl

  -- Combine the proofs to show existence
  exact ⟨percentage, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_125_equal_to_75_l1314_131456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_determination_l1314_131450

/-- Calculates simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem interest_rate_determination (principal : ℝ) (time : ℝ) (diff : ℝ) :
  principal = 1800 →
  time = 2 →
  diff = 18 →
  ∃ rate : ℝ, 
    compoundInterest principal rate time - simpleInterest principal rate time = diff ∧
    rate = 10 := by
  sorry

#check interest_rate_determination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_determination_l1314_131450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1314_131412

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the condition b² + c² = a² + bc
def triangle_condition (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = t.a^2 + t.b * t.c

-- Define the function f(x)
noncomputable def f (x A : ℝ) : ℝ :=
  Real.sin (x - A) + Real.sqrt 3 * Real.cos x

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_condition t) :
  t.A = π/3 ∧ ∃ (M : ℝ), M = 1 ∧ ∀ x, f x t.A ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1314_131412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l1314_131494

/-- A circle C tangent to two parallel lines with its center on a perpendicular line -/
structure TangentCircle where
  /-- The circle C -/
  C : Set (ℝ × ℝ)
  /-- The circle C is tangent to the line x - y = 0 -/
  tangent_line1 : ∀ (x y : ℝ), (x, y) ∈ C → x - y = 0 → False
  /-- The circle C is tangent to the line x - y - 4 = 0 -/
  tangent_line2 : ∀ (x y : ℝ), (x, y) ∈ C → x - y - 4 = 0 → False
  /-- The center of circle C lies on the line x + y = 0 -/
  center_on_line : ∃ (h k : ℝ), (h, k) ∈ C ∧ h + k = 0 ∧ ∀ (x y : ℝ), (x, y) ∈ C → (x - h)^2 + (y - k)^2 ≤ (x - h)^2 + (y - k)^2

/-- The equation of the circle C is (x - 1)^2 + (y + 1)^2 = 2 -/
theorem tangent_circle_equation (tc : TangentCircle) :
  tc.C = {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 1)^2 = 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l1314_131494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l1314_131422

theorem congruent_integers_count : 
  (Finset.filter (fun n : ℕ => n < 500 ∧ n % 7 = 4) (Finset.range 500)).card = 71 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l1314_131422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_divisible_by_four_to_fourth_l1314_131481

theorem smallest_factorial_divisible_by_four_to_fourth (n : ℕ) : 
  (∀ k : ℕ, k < 10 → ¬(4^4 ∣ Nat.factorial k)) ∧ 
  (4^4 ∣ Nat.factorial 10) ∧
  (∀ m : ℕ, m > 4 → ¬(4^m ∣ Nat.factorial 10)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_divisible_by_four_to_fourth_l1314_131481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisibility_l1314_131448

theorem smallest_n_divisibility : ∃! n : ℕ+, 
  (∀ m : ℕ+, (8 ∣ m^2) ∧ (216 ∣ m^3) → n ≤ m) ∧ 
  (8 ∣ n^2) ∧ 
  (216 ∣ n^3) ∧ 
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisibility_l1314_131448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_theorem_l1314_131432

-- Define the side length of the squares
def side_length : ℝ := 6

-- Define the rotation angles
noncomputable def rotation1 : ℝ := 0
noncomputable def rotation2 : ℝ := 30 * Real.pi / 180
noncomputable def rotation3 : ℝ := 60 * Real.pi / 180

-- Define the function to calculate the area of the resulting polygon
noncomputable def polygon_area (s : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ := sorry

-- Theorem statement
theorem polygon_area_theorem :
  polygon_area side_length rotation1 rotation2 rotation3 = 144 - 18 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_theorem_l1314_131432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_at_most_one_obtuse_angle_l1314_131439

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_of_angles : (angles 0) + (angles 1) + (angles 2) = Real.pi

-- Define what it means for an angle to be obtuse
def is_obtuse (angle : ℝ) : Prop := angle > Real.pi / 2

-- Theorem statement
theorem triangle_at_most_one_obtuse_angle (t : Triangle) : 
  ¬(∃ i j : Fin 3, i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_at_most_one_obtuse_angle_l1314_131439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_equation_l1314_131466

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflect a point about a line -/
noncomputable def reflect (p : Point) (l : Line) : Point :=
  sorry

/-- The equation of a line in the form ax + by + c = 0 -/
def lineEquation (a b c : ℝ) (p : Point) : Prop :=
  a * p.x + b * p.y + c = 0

theorem line_m_equation (P P'' : Point) (ℓ m : Line) :
  P.x = 3 ∧ P.y = -2 ∧
  P''.x = -2 ∧ P''.y = 3 ∧
  (∀ p, lineEquation 4 1 0 p ↔ 4 * p.x + p.y = 0) ∧  -- equation of ℓ: 4x + y = 0
  (∃ P' : Point, reflect P ℓ = P' ∧ reflect P' m = P'') →
  (∀ p, lineEquation 5 1 0 p ↔ 5 * p.x + p.y = 0)  -- equation of m: 5x + y = 0
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_equation_l1314_131466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_m_lcm_equal_l1314_131468

theorem least_m_lcm_equal : 
  (∀ m : ℕ, m > 0 → m < 70 → Nat.lcm 15 m ≠ Nat.lcm 42 m) ∧ 
  Nat.lcm 15 70 = Nat.lcm 42 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_m_lcm_equal_l1314_131468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_32_distances_l1314_131407

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The theorem to be proved -/
theorem at_least_32_distances (points : Finset Point) (h : points.card = 1997) :
  (Finset.image₂ distance points points).card ≥ 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_32_distances_l1314_131407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_point_sum_l1314_131438

noncomputable section

/-- The line equation y = -5/3x + 15 -/
def line_equation (x : ℝ) : ℝ := -5/3 * x + 15

/-- Point P is where the line intersects the x-axis -/
def P : ℝ × ℝ := (9, 0)

/-- Point Q is where the line intersects the y-axis -/
def Q : ℝ × ℝ := (0, 15)

/-- Point T lies on the line segment PQ -/
def T (u v : ℝ) : Prop := 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  (u, v) = ((1 - t) * P.1 + t * Q.1, (1 - t) * P.2 + t * Q.2)

/-- The area of triangle POQ is twice the area of triangle TOP -/
def area_condition (u v : ℝ) : Prop :=
  (P.1 * Q.2) / 2 = ((P.1 - u) * v) / 2 + (u * v) / 2

theorem line_intersection_point_sum :
  ∀ u v : ℝ, T u v → line_equation u = v → area_condition u v → u + v = 12 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_point_sum_l1314_131438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_divisibility_l1314_131413

theorem least_n_divisibility : ∃ (n : ℕ), 
  (n = 4) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ : ℕ, 
    (1 ≤ k₁) ∧ (k₁ ≤ m + 1) ∧ (1 ≤ k₂) ∧ (k₂ ≤ m + 1) ∧ 
    (m^2 - m) % k₁ = 0 ∧ 
    (m^2 - m) % k₂ ≠ 0)) ∧
  (∃ k₁ k₂ : ℕ, 
    (1 ≤ k₁) ∧ (k₁ ≤ n + 1) ∧ (1 ≤ k₂) ∧ (k₂ ≤ n + 1) ∧ 
    (n^2 - n) % k₁ = 0 ∧ 
    (n^2 - n) % k₂ ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_divisibility_l1314_131413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_two_l1314_131458

-- Define the polar curve and line
noncomputable def polar_curve (θ : ℝ) : ℝ := -2 * Real.sin θ
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.sin θ = -1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (ρ θ : ℝ), 
    ρ = polar_curve θ ∧ 
    polar_line ρ θ ∧ 
    p = (ρ * Real.cos θ, ρ * Real.sin θ)}

-- Theorem statement
theorem intersection_distance_is_two :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧ 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_two_l1314_131458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l1314_131482

theorem geometric_sequence_minimum :
  ∀ (a : ℕ → ℝ) (q : ℝ),
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  q > 1 →  -- positive sequence condition
  a 3 - a 1 = 2 →  -- given condition
  ∃ m : ℝ, (∀ q : ℝ, a 4 + a 3 ≥ m) ∧ (∃ q : ℝ, a 4 + a 3 = m) ∧ m = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l1314_131482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l1314_131420

-- Define the polynomial (1 + x + x^2)
noncomputable def p (x : ℝ) : ℝ := 1 + x + x^2

-- Define the function (x - 1/x)^6
noncomputable def q (x : ℝ) : ℝ := (x - 1/x)^6

-- Define the product of p and q
noncomputable def f (x : ℝ) : ℝ := p x * q x

-- Theorem statement
theorem constant_term_of_expansion : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c + x * (f x - c) ∧ c = -5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l1314_131420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1314_131464

/-- The inclination angle of a line with equation ax + by + c = 0 -/
noncomputable def inclination_angle (a b : ℝ) : ℝ := Real.arctan (-a / b)

/-- Prove that the inclination angle of the line x + √3y + 1 = 0 is 5π/6 -/
theorem line_inclination_angle :
  inclination_angle 1 (Real.sqrt 3) = 5 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1314_131464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_threes_divisible_by_19_l1314_131495

def insert_threes (n : ℕ) : ℕ :=
  12000 + 3 * ((10^n - 1) / 9) + 8

theorem threes_divisible_by_19 (n : ℕ) : 
  19 ∣ insert_threes n := by
  sorry

#eval insert_threes 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_threes_divisible_by_19_l1314_131495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1314_131459

/-- Definition of an equilateral triangle with side length 12 meters -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 12

/-- The perimeter of an equilateral triangle -/
noncomputable def perimeter (t : EquilateralTriangle) : ℝ :=
  3 * t.side_length

/-- The area of an equilateral triangle -/
noncomputable def area (t : EquilateralTriangle) : ℝ :=
  (Real.sqrt 3 / 4) * t.side_length ^ 2

/-- Theorem: The perimeter and area of the given equilateral triangle -/
theorem equilateral_triangle_properties (t : EquilateralTriangle) :
  perimeter t = 36 ∧ area t = 36 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1314_131459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l1314_131489

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0
  eq : ℝ → ℝ → Prop
  point_neg_two : ∃ t, eq (-2) t
  point_neg_one : eq (-1) 0
  point_zero : eq 0 3
  point_one : eq 1 4
  eq_def : ∀ x y, eq x y ↔ y = a * x^2 + b * x + c

/-- The vertex of a quadratic function -/
noncomputable def vertex (f : QuadraticFunction) : ℝ × ℝ :=
  (- f.b / (2 * f.a), f.c - f.b^2 / (4 * f.a))

/-- Theorem: The vertex of the quadratic function is at (1,4) -/
theorem quadratic_vertex (f : QuadraticFunction) : vertex f = (1, 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l1314_131489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1314_131440

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x, HasDerivAt (f a) (Real.exp x - a) x) →
  HasDerivAt (f a) (-1) 0 →
  (a = 2) ∧
  (∃ x_min : ℝ, x_min = Real.log 2 ∧
    ∀ x, f a x ≥ f a x_min ∧ f a x_min = 2 - Real.log 4) ∧
  (∀ M : ℝ, ∃ x : ℝ, f a x > M) ∧
  (∀ x : ℝ, x > 0 → x^2 < Real.exp x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1314_131440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_of_A_l1314_131461

/-- Given a triangle ABC with isosceles triangles constructed on its sides -/
structure TriangleWithIsosceles where
  /-- The original triangle ABC -/
  ABC : Set (Fin 3 → ℝ × ℝ)
  /-- Isosceles triangle A'BC -/
  A'BC : Set (Fin 3 → ℝ × ℝ)
  /-- Isosceles triangle AB'C -/
  AB'C : Set (Fin 3 → ℝ × ℝ)
  /-- Isosceles triangle ABC' -/
  ABC' : Set (Fin 3 → ℝ × ℝ)
  /-- Angle at vertex A' -/
  α : ℝ
  /-- Angle at vertex B' -/
  β : ℝ
  /-- Angle at vertex C' -/
  γ : ℝ
  /-- Sum of angles α, β, γ is 2π -/
  angle_sum : α + β + γ = 2 * Real.pi

/-- The theorem stating that the angles of triangle A'B'C' are α/2, β/2, γ/2 -/
theorem angles_of_A'B'C' (t : TriangleWithIsosceles) :
  ∃ A'B'C' : Set (Fin 3 → ℝ × ℝ),
    ∃ angle₁ angle₂ angle₃ : ℝ,
      angle₁ = t.α / 2 ∧
      angle₂ = t.β / 2 ∧
      angle₃ = t.γ / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_of_A_l1314_131461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_garden_plants_l1314_131417

theorem second_garden_plants : 
  ∀ (total_plants : ℕ) (second_garden_plants : ℕ),
  let first_garden_plants : ℕ := 20
  let first_garden_tomatoes : ℚ := (1 / 10) * first_garden_plants
  let second_garden_tomatoes : ℚ := (1 / 3) * second_garden_plants
  total_plants = first_garden_plants + second_garden_plants →
  (1 / 5) * total_plants = first_garden_tomatoes + second_garden_tomatoes →
  second_garden_plants = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_garden_plants_l1314_131417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_french_fries_lovers_l1314_131423

theorem french_fries_lovers (total : ℕ) (burger_lovers : ℕ) (both_lovers : ℕ) (neither_lovers : ℕ) (french_fries_lovers : ℕ)
  (h1 : total = 25)
  (h2 : burger_lovers = 10)
  (h3 : both_lovers = 6)
  (h4 : neither_lovers = 6) :
  total = (french_fries_lovers + burger_lovers - both_lovers + neither_lovers) →
  french_fries_lovers = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_french_fries_lovers_l1314_131423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_cosine_fraction_l1314_131471

theorem limit_cosine_fraction : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - π/3| ∧ |x - π/3| < δ → 
    |((1 - 2*Real.cos x) / (π - 3*x)) + Real.sqrt 3/3| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_cosine_fraction_l1314_131471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_value_l1314_131470

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0  -- Add the base case for 0
  | n + 1 => (sequence_a n - Real.sqrt 3) / (Real.sqrt 3 * sequence_a n + 1)

theorem a_20_value : sequence_a 20 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_value_l1314_131470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meditation_duration_theorem_l1314_131424

/-- Represents the duration of meditation in minutes -/
abbrev MeditationDuration := Nat

/-- Represents the number of meditation sessions per day -/
abbrev SessionsPerDay := Nat

/-- Represents the number of hours spent meditating per week -/
abbrev WeeklyHours := Nat

/-- Calculates the duration of each meditation session -/
def calculate_session_duration (sessions_per_day : SessionsPerDay) (weekly_hours : WeeklyHours) : MeditationDuration :=
  (weekly_hours * 60) / (sessions_per_day * 7)

theorem meditation_duration_theorem (sessions_per_day : SessionsPerDay) (weekly_hours : WeeklyHours) 
  (h1 : sessions_per_day = 2)
  (h2 : weekly_hours = 7) :
  calculate_session_duration sessions_per_day weekly_hours = 30 := by
  sorry

#eval calculate_session_duration 2 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meditation_duration_theorem_l1314_131424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_dog_age_difference_l1314_131419

/-- Represents the age progression rate of a dog in dog years per human year. -/
abbrev DogAgeRate := ℕ

/-- Calculates the age of a dog in dog years given the owner's age and the dog's age rate. -/
def dogAge (ownerAge : ℕ) (rate : DogAgeRate) : ℕ :=
  ownerAge * rate

/-- Calculates the difference in years between a dog's age in dog years and the owner's age in human years. -/
def ageDifference (ownerAge : ℕ) (rate : DogAgeRate) : ℕ :=
  dogAge ownerAge rate - ownerAge

/-- Theorem stating that the combined difference in dog years between a 6-year-old boy
    and his three dogs (small, medium, and large breed) is 108 dog years. -/
theorem combined_dog_age_difference
  (smallBreedRate mediumBreedRate largeBreedRate : DogAgeRate)
  (h1 : smallBreedRate = 5)
  (h2 : mediumBreedRate = 7)
  (h3 : largeBreedRate = 9)
  : ageDifference 6 smallBreedRate + ageDifference 6 mediumBreedRate + ageDifference 6 largeBreedRate = 108 := by
  sorry

#eval ageDifference 6 5 + ageDifference 6 7 + ageDifference 6 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_dog_age_difference_l1314_131419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_goose_eggs_weight_l1314_131426

/-- Represents the weight of an object in grams -/
structure Weight where
  value : ℕ

/-- Conversion factor from grams to kilograms -/
def gramsPerKilogram : ℕ := 1000

/-- Weight of one goose egg in grams -/
def gooseEggWeight : Weight := ⟨100⟩

/-- Number of goose eggs -/
def numberOfEggs : ℕ := 10

theorem ten_goose_eggs_weight :
  (numberOfEggs : ℕ) * gooseEggWeight.value = gramsPerKilogram := by
  -- The proof goes here
  sorry

#eval numberOfEggs * gooseEggWeight.value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_goose_eggs_weight_l1314_131426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_circular_arrangement_l1314_131453

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_arrangement (arr : List ℕ) : Prop :=
  arr.length = 30 ∧
  (∀ n, n ∈ arr → 1 ≤ n ∧ n ≤ 30) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 30 → n ∈ arr) ∧
  (∀ i, i < arr.length → 
    is_perfect_square (arr[i]! + arr[(i + 1) % arr.length]!))

theorem no_valid_circular_arrangement : ¬∃ arr : List ℕ, valid_arrangement arr := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_circular_arrangement_l1314_131453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_pure_imaginary_l1314_131430

-- Define the complex number z as a function of m
noncomputable def z (m : ℝ) : ℂ := Complex.mk (Real.log (m^2 - 2*m - 2)) (m^2 + 3*m + 2)

-- Theorem for real number condition
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = -2 ∨ m = -1 := by sorry

-- Theorem for pure imaginary number condition
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ↔ m = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_pure_imaginary_l1314_131430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_hilt_coin_total_l1314_131405

/-- Represents the collection of coins found by Mrs. Hilt -/
structure CoinCollection where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat
  half_dollars : Nat
  one_dollar_coins : Nat
  two_dollar_canadian_coins : Nat

/-- Calculates the total value of the coin collection in US cents -/
def total_value_in_cents (coins : CoinCollection) (exchange_rate : Float) : Float :=
  (coins.quarters.toFloat * 25) +
  (coins.dimes.toFloat * 10) +
  (coins.nickels.toFloat * 5) +
  coins.pennies.toFloat +
  (coins.half_dollars.toFloat * 50) +
  (coins.one_dollar_coins.toFloat * 100) +
  (coins.two_dollar_canadian_coins.toFloat * 200 * exchange_rate)

/-- Theorem stating that Mrs. Hilt's coin collection totals $11.82 -/
theorem mrs_hilt_coin_total :
  let coins : CoinCollection := {
    quarters := 4,
    dimes := 6,
    nickels := 8,
    pennies := 12,
    half_dollars := 3,
    one_dollar_coins := 5,
    two_dollar_canadian_coins := 2
  }
  let exchange_rate : Float := 0.80
  total_value_in_cents coins exchange_rate / 100 = 11.82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_hilt_coin_total_l1314_131405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_collection_size_is_seven_l1314_131462

/-- A collection of sets satisfying the given conditions -/
structure SetCollection where
  sets : Finset (Finset Nat)
  elem_count : ∀ s, s ∈ sets → Finset.card s = 4
  diff_in_collection : ∀ A B, A ∈ sets → B ∈ sets → (A \ B ∪ B \ A) ∈ sets

/-- The maximum number of sets in the collection -/
def max_collection_size : Nat := 7

/-- Theorem stating that the maximum size of the collection is 7 -/
theorem max_collection_size_is_seven : 
  ∀ c : SetCollection, Finset.card c.sets ≤ max_collection_size := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_collection_size_is_seven_l1314_131462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_side_c_value_l1314_131496

noncomputable section

variable (a b c A B C S : ℝ)

def triangle_abc (a b c A B C S : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  S = (1/2) * a * b * Real.sin C

theorem angle_C_measure (h : 4 * S = Real.sqrt 3 * (a^2 + b^2 - c^2)) 
  (tri : triangle_abc a b c A B C S) : C = Real.pi/3 := by
  sorry

theorem side_c_value (h1 : 1 + (Real.tan A / Real.tan B) = 2 * c / b)
  (h2 : a * c * Real.cos B = -8)
  (tri : triangle_abc a b c A B C S) : c = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_side_c_value_l1314_131496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1314_131402

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x^3 + 5*x^2 + 6*x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -3 ∧ x ≠ -2 ∧ x ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1314_131402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1314_131479

/-- The nth term of the series -/
noncomputable def a (n : ℕ) : ℝ := n^2 / ((4*n - 2)^2 * (4*n + 2)^2)

/-- The sum of the series -/
noncomputable def S : ℝ := ∑' n, a n

/-- The theorem stating the sum of the series -/
theorem series_sum : S = π^2/192 - 1/32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1314_131479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_integers_between_fractions_l1314_131488

theorem even_integers_between_fractions : 
  let lower_bound : ℚ := 9/2
  let upper_bound : ℚ := 24/1
  (Finset.filter (λ x => x % 2 = 0) 
    (Finset.Icc ⌈lower_bound⌉ ⌊upper_bound⌋)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_integers_between_fractions_l1314_131488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_velocity_and_cost_ship_optimal_velocity_and_cost_l1314_131486

noncomputable def fuel_cost_coefficient (v₀ u₀ : ℝ) : ℝ := u₀ / (v₀^3)

noncomputable def total_cost_per_km (k other_costs v : ℝ) : ℝ := k * v^2 + other_costs / v

theorem optimal_velocity_and_cost 
  (v₀ u₀ other_costs : ℝ) 
  (h_v₀_positive : v₀ > 0)
  (h_u₀_positive : u₀ > 0)
  (h_other_costs_positive : other_costs > 0) :
  ∃ (v_opt : ℝ),
    v_opt > 0 ∧
    (∀ v, v > 0 → total_cost_per_km (fuel_cost_coefficient v₀ u₀) other_costs v ≥ 
                   total_cost_per_km (fuel_cost_coefficient v₀ u₀) other_costs v_opt) ∧
    v_opt = (2 * other_costs / (fuel_cost_coefficient v₀ u₀))^(1/3) ∧
    total_cost_per_km (fuel_cost_coefficient v₀ u₀) other_costs v_opt = 
      3 * (4 * (fuel_cost_coefficient v₀ u₀) * other_costs^2)^(1/3) :=
by sorry

theorem ship_optimal_velocity_and_cost :
  ∃ (v_opt : ℝ),
    v_opt > 0 ∧
    (∀ v, v > 0 → total_cost_per_km (fuel_cost_coefficient 10 35) 560 v ≥ 
                   total_cost_per_km (fuel_cost_coefficient 10 35) 560 v_opt) ∧
    v_opt = 20 ∧
    total_cost_per_km (fuel_cost_coefficient 10 35) 560 v_opt = 42 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_velocity_and_cost_ship_optimal_velocity_and_cost_l1314_131486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircles_area_difference_l1314_131415

noncomputable def rectangle_width : ℝ := 8
noncomputable def rectangle_length : ℝ := 12

noncomputable def semicircle_area (diameter : ℝ) : ℝ := (Real.pi * diameter^2) / 8

noncomputable def larger_semicircles_area : ℝ := 2 * semicircle_area rectangle_length
noncomputable def smaller_semicircles_area : ℝ := 2 * semicircle_area rectangle_width

theorem semicircles_area_difference :
  (larger_semicircles_area - smaller_semicircles_area) / smaller_semicircles_area * 100 = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircles_area_difference_l1314_131415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_value_l1314_131427

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) / (1 + 2 * sequence_a (n + 1))

theorem tenth_term_value : sequence_a 10 = 1/19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_value_l1314_131427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_two_l1314_131410

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.cos x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := Real.exp x - Real.sin x

-- Theorem statement
theorem tangent_line_at_zero_two :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  (x y : ℝ) → (y - y₀ = m * (x - x₀)) ↔ (x - y + 2 = 0) :=
by
  -- Introduce the variables
  intro x₀ y₀ m x y
  -- Skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_two_l1314_131410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_11_is_44_l1314_131428

/-- An arithmetic progression where the sum of the third and ninth terms is 8 -/
structure ArithProgression where
  a : ℕ → ℚ  -- The sequence of terms (using rationals instead of reals)
  sum_3_9 : a 3 + a 9 = 8  -- The sum of the 3rd and 9th terms is 8
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1  -- Constant difference between consecutive terms

/-- The sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a 1 + (n - 1) * (ap.a 2 - ap.a 1))

/-- Theorem: The sum of the first 11 terms of the arithmetic progression is 44 -/
theorem sum_11_is_44 (ap : ArithProgression) : sum_n ap 11 = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_11_is_44_l1314_131428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_latus_rectum_distance_l1314_131483

/-- The distance from a point to a line --/
noncomputable def distance_point_to_line (x y : ℝ) (m b : ℝ) : ℝ :=
  |y - (m * x + b)| / Real.sqrt (1 + m^2)

/-- The latus rectum y-coordinate for a parabola y = ax² --/
noncomputable def latus_rectum_y (a : ℝ) : ℝ := -1 / (4 * a)

theorem parabola_latus_rectum_distance (a : ℝ) :
  a ≠ 0 →
  (distance_point_to_line 2 1 0 (latus_rectum_y a) = 2) →
  (a = 1/4 ∨ a = -1/12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_latus_rectum_distance_l1314_131483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_altitude_of_special_triangle_l1314_131408

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the area of a triangle given two sides and the angle between them -/
noncomputable def areaFromSides (s1 s2 : ℝ) : ℝ := (1/2) * s1 * s2

/-- Calculates the shortest altitude of a triangle -/
noncomputable def shortestAltitude (t : Triangle) : ℝ :=
  (2 * areaFromSides t.a t.b) / t.c

/-- Theorem stating that for a triangle with sides 15, 20, and 25, the shortest altitude is 12 -/
theorem shortest_altitude_of_special_triangle :
  let t : Triangle := { a := 15, b := 20, c := 25 }
  shortestAltitude t = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_altitude_of_special_triangle_l1314_131408
