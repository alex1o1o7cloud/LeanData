import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colinear_vectors_lambda_l431_43144

/-- Given vectors a, b, and c in ℝ², prove that if lambda * a + b is colinear with c, then lambda = 2 -/
theorem colinear_vectors_lambda (a b c : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 2) →
  b = (2, 3) →
  c = (-4, -7) →
  ∃ (k : ℝ), k ≠ 0 ∧ (lambda * a.1 + b.1, lambda * a.2 + b.2) = (k * c.1, k * c.2) →
  lambda = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colinear_vectors_lambda_l431_43144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_for_six_cookies_l431_43146

/-- The amount of milk in pints needed for a given number of cookies, considering measurement error --/
noncomputable def milk_needed (cookies : ℕ) (error_percentage : ℝ) : ℝ :=
  let gallons_for_24 : ℝ := 1.5
  let quarts_per_gallon : ℕ := 4
  let pints_per_quart : ℕ := 2
  let pints_for_24 : ℝ := gallons_for_24 * (quarts_per_gallon : ℝ) * (pints_per_quart : ℝ)
  let pints_without_error : ℝ := (pints_for_24 / 24) * (cookies : ℝ)
  pints_without_error * (1 + error_percentage / 100)

theorem milk_for_six_cookies :
  milk_needed 6 10 = 3.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_for_six_cookies_l431_43146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_periodicity_l431_43103

theorem function_periodicity (a : ℚ) (b c d : ℝ) 
  (f : ℝ → ℝ) (h_range : ∀ x, f x ∈ Set.Icc (-1 : ℝ) 1) 
  (h_eq : ∀ x : ℝ, f (x + a + b) - f (x + b) = 
    c * (x + 2 * a + ⌊x⌋ - 2 * ⌊x + a⌋ - ⌊b⌋) + d) :
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x : ℝ, f (x + p) = f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_periodicity_l431_43103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pastries_trick_min_pastries_is_36_l431_43137

/-- The number of different fillings -/
def num_fillings : ℕ := 10

/-- The total number of pastries -/
def total_pastries : ℕ := 45

/-- The number of pastries with each specific filling -/
def pastries_per_filling : ℕ := 9

/-- The function to calculate the minimum number of pastries needed to be examined -/
def min_pastries_examined : ℕ := total_pastries - num_fillings + 1

/-- Represents the fillings of a pastry -/
def pastry_fillings : Fin total_pastries → Finset (Fin num_fillings) :=
  sorry

theorem min_pastries_trick (n : ℕ) : 
  (n ≥ min_pastries_examined) ↔ 
  (∀ m : ℕ, m < n → ∃ filling : Fin num_fillings, 
    ∀ pastry : Fin total_pastries, pastry.val ∉ Finset.range m → 
      filling ∈ pastry_fillings pastry) :=
sorry

theorem min_pastries_is_36 : min_pastries_examined = 36 :=
by
  unfold min_pastries_examined
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pastries_trick_min_pastries_is_36_l431_43137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_f_greater_than_g_l431_43107

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * Real.exp x + 1
def g (x : ℝ) : ℝ := x^3

-- Part I: f has two zeros iff a ∈ (1, +∞)
theorem f_has_two_zeros (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ a > 1 := by
  sorry

-- Part II: When a = 1 and x ∈ (1/3, 1), f(x) > g(x)
theorem f_greater_than_g :
  ∀ x : ℝ, 1/3 < x ∧ x < 1 → f 1 x > g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_f_greater_than_g_l431_43107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1998_l431_43154

noncomputable def sequenceX (a b : ℝ) : ℕ → ℝ
  | 0 => a
  | 1 => b
  | (n + 2) => (1 + sequenceX a b (n + 1)) / (sequenceX a b n)

theorem sequence_1998 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  sequenceX a b 1998 = (1 + a + b) / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1998_l431_43154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_x_axis_l431_43128

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-coordinate of the intersection point of a line with the x-axis -/
noncomputable def xIntersection (l : Line) : ℝ :=
  l.x₁ + (l.x₂ - l.x₁) * (0 - l.y₁) / (l.y₂ - l.y₁)

theorem line_intersects_x_axis (l : Line) :
  l.x₁ = 4 ∧ l.y₁ = -2 ∧ l.x₂ = 8 ∧ l.y₂ = 2 →
  xIntersection l = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_x_axis_l431_43128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_solution_set_l431_43104

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 4|

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f x ≥ m := by
  sorry

-- Define the set of solutions for |f(x) - 6| ≤ 1
def solution_set : Set ℝ := {x : ℝ | |f x - 6| ≤ 1}

-- Theorem for the solution set of |f(x) - 6| ≤ 1
theorem f_solution_set : solution_set = Set.union (Set.Icc (-10/3) (-8/3)) (Set.Icc 0 (4/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_solution_set_l431_43104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l431_43158

noncomputable def OA : ℝ := 2
noncomputable def OB : ℝ := 3
noncomputable def OC : ℝ := 2 * Real.sqrt 5
noncomputable def tan_AOC : ℝ := 2
noncomputable def angle_BOC : ℝ := Real.pi / 3

theorem vector_decomposition (p q : ℝ) :
  OA = 2 ∧ OB = 3 ∧ OC = 2 * Real.sqrt 5 ∧
  tan_AOC = 2 ∧ angle_BOC = Real.pi / 3 →
  p = (2 * Real.sqrt 5 - 3) / 3 ∧
  q = (2 - 2 * Real.sqrt 5) / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l431_43158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_2_l431_43168

noncomputable def f (x : ℝ) : ℝ := x * Real.sin (2 * x)

theorem derivative_f_at_pi_over_2 :
  deriv f (π / 2) = -π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_2_l431_43168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cow_selling_price_l431_43197

/-- Calculates the selling price per pound for a cow given its initial weight,
    weight increase factor, and value increase. -/
noncomputable def selling_price_per_pound (initial_weight : ℝ) (weight_increase_factor : ℝ) (value_increase : ℝ) : ℝ :=
  value_increase / (initial_weight * (weight_increase_factor - 1))

/-- Theorem stating that for a cow with given parameters, the selling price is $3 per pound. -/
theorem cow_selling_price :
  let initial_weight : ℝ := 400
  let weight_increase_factor : ℝ := 1.5
  let value_increase : ℝ := 600
  selling_price_per_pound initial_weight weight_increase_factor value_increase = 3 := by
  -- Unfold the definition of selling_price_per_pound
  unfold selling_price_per_pound
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cow_selling_price_l431_43197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photocopy_order_theorem_l431_43175

/-- The number of copies needed to save $0.40 each when submitting a single order -/
def total_copies_for_discount (
  cost_per_copy : ℚ)
  (discount_rate : ℚ)
  (discount_threshold : ℕ)
  (copies_per_person : ℕ)
  (num_people : ℕ)
  (target_savings : ℚ) : ℕ :=
  let base_order := copies_per_person * num_people
  let discounted_cost := cost_per_copy * (1 - discount_rate)
  let savings_per_copy := cost_per_copy - discounted_cost
  let additional_copies := (target_savings / savings_per_copy).ceil.toNat
  base_order + additional_copies * num_people

theorem photocopy_order_theorem :
  total_copies_for_discount (2/100) (1/4) 100 80 2 (2/5) = 320 := by
  sorry

#eval total_copies_for_discount (2/100) (1/4) 100 80 2 (2/5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photocopy_order_theorem_l431_43175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l431_43163

-- Define an acute triangle
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define the cyclic sum operation
def cyclicSum (f : ℝ → ℝ → ℝ → ℝ) (t : AcuteTriangle) : ℝ :=
  f t.A t.B t.C + f t.B t.C t.A + f t.C t.A t.B

-- State the theorem
theorem triangle_inequality (t : AcuteTriangle) :
  cyclicSum (fun x y _ => 1 / (Real.cos x + Real.cos y)) t ≥
  (1/2) * (t.a + t.b + t.c) * (1/t.a + 1/t.b + 1/t.c) - 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l431_43163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_even_phase_shift_l431_43155

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The sine function with a phase shift -/
noncomputable def f (φ : ℝ) : ℝ → ℝ := λ x ↦ Real.sin (2 * x + φ)

/-- If f(x) = sin(2x + φ) is an even function on ℝ, then φ = π/2 -/
theorem sin_even_phase_shift (φ : ℝ) :
  IsEven (f φ) → φ = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_even_phase_shift_l431_43155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_A_range_l431_43151

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1

-- Define the properties of the triangle
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def is_acute_triangle (a b c : ℝ) : Prop :=
  is_valid_triangle a b c ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- Define the condition for angle B
def angle_B_condition (a b c : ℝ) : Prop :=
  Real.tan (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = Real.sqrt 3 * a * c / (a^2 + c^2 - b^2)

-- Theorem statement
theorem f_A_range (a b c A : ℝ) :
  is_acute_triangle a b c →
  angle_B_condition a b c →
  0 < A ∧ A < Real.pi / 2 →
  ∃ (y : ℝ), -1 < y ∧ y < 2 ∧ f A = y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_A_range_l431_43151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_increases_new_mean_is_correct_variance_unchanged_l431_43114

/- Define the necessary parameters -/
def daily_production : ℕ := 200
def grade_a_price : ℕ := 160
def grade_b_price : ℕ := 140
def sample_size : ℕ := 16
noncomputable def sample_mean : ℝ := 9.97
noncomputable def sample_variance : ℝ := 0.045
def improvement_cost : ℕ := 300000
noncomputable def improvement_effect : ℝ := 0.05

/- Define the grade A threshold -/
noncomputable def grade_a_threshold : ℝ := 10

/- Define the probabilities before and after improvement -/
noncomputable def prob_a_before : ℝ := 1/2
noncomputable def prob_b_before : ℝ := 1/2
noncomputable def prob_a_after : ℝ := 13/16
noncomputable def prob_b_after : ℝ := 3/16

/- Define revenue calculation functions -/
noncomputable def daily_revenue (prob_a prob_b : ℝ) : ℝ :=
  daily_production * (prob_a * grade_a_price + prob_b * grade_b_price)

noncomputable def annual_revenue_increase : ℝ :=
  (daily_revenue prob_a_after prob_b_after - daily_revenue prob_a_before prob_b_before) * 365 - improvement_cost

/- Define the theorem for revenue increase -/
theorem revenue_increases : annual_revenue_increase > 0 := by sorry

/- Define the new mean and variance after improvement -/
noncomputable def new_mean : ℝ := sample_mean + improvement_effect
noncomputable def new_variance : ℝ := sample_variance

/- Define theorems for new mean and variance -/
theorem new_mean_is_correct : new_mean = 10.02 := by sorry
theorem variance_unchanged : new_variance = sample_variance := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_increases_new_mean_is_correct_variance_unchanged_l431_43114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_possibilities_l431_43109

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def possible_side_lengths (a : ℕ) : Prop :=
  is_valid_triangle 4 7 (a : ℝ)

theorem triangle_side_possibilities : 
  {a : ℕ | possible_side_lengths a} = {4, 5, 6, 7, 8, 9, 10} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_possibilities_l431_43109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_point_partition_l431_43193

-- Define a rational point as a pair of rational numbers
def RationalPoint := ℚ × ℚ

-- Define the set of all rational points
def AllRationalPoints : Set RationalPoint := Set.univ

-- Define the property of a point being inside a circle
def InsideCircle (center : RationalPoint) (radius : ℝ) (point : RationalPoint) : Prop :=
  (point.1 - center.1) ^ 2 + (point.2 - center.2) ^ 2 < radius ^ 2

-- Define the property of three points being collinear
def AreCollinear (p1 p2 p3 : RationalPoint) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- State the theorem
theorem rational_point_partition :
  ∃ (A B C : Set RationalPoint),
    -- A, B, C are mutually disjoint
    A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
    -- A, B, C partition all rational points
    A ∪ B ∪ C = AllRationalPoints ∧
    -- Any circle contains points from all three sets
    (∀ (center : RationalPoint) (radius : ℝ),
      ∃ (a : RationalPoint) (b : RationalPoint) (c : RationalPoint),
        a ∈ A ∧ b ∈ B ∧ c ∈ C ∧
        InsideCircle center radius a ∧
        InsideCircle center radius b ∧
        InsideCircle center radius c) ∧
    -- No straight line contains points from all three sets
    (∀ (p1 p2 p3 : RationalPoint),
      p1 ∈ A → p2 ∈ B → p3 ∈ C →
        ¬AreCollinear p1 p2 p3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_point_partition_l431_43193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_start_stories_l431_43120

/-- Represents a story in the book -/
structure Story where
  pages : Nat
  start_page : Nat
deriving Inhabited

/-- Represents the book containing stories -/
def Book := List Story

/-- Check if a number is odd -/
def is_odd (n : Nat) : Bool :=
  n % 2 = 1

/-- Check if a story starts on an odd page -/
def starts_on_odd_page (s : Story) : Bool :=
  is_odd s.start_page

/-- Count the number of stories that start on odd pages -/
def count_odd_start_stories (book : Book) : Nat :=
  (book.filter starts_on_odd_page).length

/-- Generate all possible story lengths from 1 to 30 -/
def all_story_lengths : List Nat :=
  List.range 30 |>.map (·+1)

/-- Check if a book is valid according to the problem conditions -/
def is_valid_book (book : Book) : Prop :=
  book.length = 30 ∧
  (book.map (λ s => s.pages)).toFinset = all_story_lengths.toFinset ∧
  book.head?.isSome ∧ (book.head?.get!).start_page = 1 ∧
  ∀ i, 0 < i → i < book.length →
    (book.get! i).start_page = (book.get! (i-1)).start_page + (book.get! (i-1)).pages

theorem max_odd_start_stories (book : Book) (h : is_valid_book book) :
  count_odd_start_stories book ≤ 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_start_stories_l431_43120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_regular_pentagon_proof_l431_43141

/-- The sum of exterior angles of a regular pentagon is 360 degrees. -/
def sum_exterior_angles_regular_pentagon : ℝ :=
  360

/-- A regular pentagon is a polygon with 5 sides and all sides and angles equal. -/
def regular_pentagon : ℕ :=
  5

/-- The sum of exterior angles of any polygon is constant and equal to 360 degrees. -/
axiom sum_exterior_angles_constant (n : ℕ) : ℝ

/-- Proof that the sum of exterior angles of a regular pentagon is 360 degrees. -/
theorem sum_exterior_angles_regular_pentagon_proof :
  sum_exterior_angles_regular_pentagon = sum_exterior_angles_constant regular_pentagon :=
by
  sorry

#check sum_exterior_angles_regular_pentagon_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_regular_pentagon_proof_l431_43141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_flight_time_problem_l431_43198

/-- The time (in seconds) it takes for the ball to hit the ground -/
noncomputable def ball_flight_time (ball_speed dog_speed : ℝ) (dog_catch_time : ℝ) : ℝ :=
  (dog_speed * dog_catch_time) / ball_speed

theorem ball_flight_time_problem (ball_speed dog_speed : ℝ) (dog_catch_time : ℝ) 
  (h1 : ball_speed = 20)
  (h2 : dog_speed = 5)
  (h3 : dog_catch_time = 32) :
  ball_flight_time ball_speed dog_speed dog_catch_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_flight_time_problem_l431_43198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tarabar_cipher_properties_l431_43124

/-- Represents a character in the Russian alphabet -/
inductive RussianChar : Type where
  | vowel : RussianChar
  | consonant : RussianChar

/-- Represents the tarabar cipher function -/
def tarabarCipher : RussianChar → RussianChar :=
  sorry

/-- The first sentence of the problem statement -/
def originalSentence : List RussianChar :=
  sorry

/-- The ciphered text provided in the problem -/
def cipheredText : List RussianChar :=
  sorry

/-- Axiom: The ciphered text is an encoded form of the original sentence -/
axiom ciphered_is_encoded : cipheredText = originalSentence.map tarabarCipher

/-- Theorem: The tarabar cipher preserves vowels and substitutes consonants in pairs -/
theorem tarabar_cipher_properties :
  (∀ c : RussianChar, c = RussianChar.vowel → tarabarCipher c = c) ∧
  (∃ f : RussianChar → RussianChar, 
    (∀ c : RussianChar, c = RussianChar.consonant → tarabarCipher (tarabarCipher c) = c) ∧
    (∀ c : RussianChar, c = RussianChar.consonant → tarabarCipher c = f c)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tarabar_cipher_properties_l431_43124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_theorem_l431_43186

noncomputable section

-- Define the curve C and line l
def C (a : ℝ) (x : ℝ) : ℝ := x + a / x
def l (x : ℝ) : ℝ := x

-- Define the point P on curve C
structure Point (a : ℝ) where
  x : ℝ
  y : ℝ
  h : y = C a x

-- Define the perpendicular points A and B
noncomputable def B (a : ℝ) (P : Point a) : ℝ × ℝ := (0, P.y)
noncomputable def A (a : ℝ) (P : Point a) : ℝ × ℝ := (P.x + a / (2 * P.x), P.x + a / (2 * P.x))

-- Define the tangent line points M and N
noncomputable def M (a : ℝ) (P : Point a) : ℝ × ℝ := (2 * P.x, 2 * P.x)
noncomputable def N (a : ℝ) (P : Point a) : ℝ × ℝ := (0, 2 * a / P.x)

-- Define the areas of triangles ABP and OMN
noncomputable def areaABP (a : ℝ) (P : Point a) : ℝ := (1/2) * P.x * (a / (2 * P.x))
noncomputable def areaOMN (a : ℝ) (P : Point a) : ℝ := (1/2) * (2 * a / P.x) * (2 * P.x)

-- State the theorem
theorem area_theorem (a : ℝ) (P : Point a) (h : a > 0) :
  areaABP a P = (1/2) → areaOMN a P = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_theorem_l431_43186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_f_geq_one_iff_a_geq_one_l431_43184

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := a * Real.exp (x - 1) - Real.log x + Real.log a

-- Theorem for the area of the triangle
theorem triangle_area (a : ℝ) (h : a = Real.exp 1) :
  let tangent_line (x : ℝ) := (f a 1 + (Real.exp 1 - 1) * (x - 1))
  let x_intercept := -2 / (Real.exp 1 - 1)
  let y_intercept := 2
  (1/2) * x_intercept * y_intercept = 2 / (Real.exp 1 - 1) := by sorry

-- Theorem for the range of a
theorem f_geq_one_iff_a_geq_one (a : ℝ) :
  (∀ x > 0, f a x ≥ 1) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_f_geq_one_iff_a_geq_one_l431_43184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l431_43117

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  total_distance / train_speed_ms

theorem train_bridge_crossing_time :
  train_crossing_time 150 45 225 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l431_43117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_coordinates_l431_43126

-- Define the integer part function T
noncomputable def T (a : ℝ) : ℤ := Int.floor a

-- Define the recursive formulas for x and y
noncomputable def x : ℕ → ℤ
  | 0 => 1
  | k + 1 => x k + 1 - 5 * (T ((k : ℝ) / 5) - T ((k - 1 : ℝ) / 5))

noncomputable def y : ℕ → ℤ
  | 0 => 1
  | k + 1 => y k + T ((k : ℝ) / 5) - T ((k - 1 : ℝ) / 5)

-- State the theorem
theorem tree_planting_coordinates :
  (x 6 = 1 ∧ y 6 = 2) ∧ (x 2016 = 1 ∧ y 2016 = 403) := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma x_periodic (n : ℕ) : x (5 * n + 1) = 1 := by
  sorry

lemma y_formula (n k : ℕ) (h : k < 5) : y (5 * n + k) = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_coordinates_l431_43126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_sum_l431_43143

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log x + 2

def tangent_line (a b x : ℝ) : ℝ := a * x + b

theorem common_tangent_sum (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧
    (∀ x : ℝ, tangent_line a b x ≥ f x) ∧
    (tangent_line a b x₁ = f x₁) ∧
    (∀ x : ℝ, x > 0 → tangent_line a b x ≥ g x) ∧
    (tangent_line a b x₂ = g x₂)) →
  b > 0 →
  a + b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_sum_l431_43143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l431_43119

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m / x

-- Theorem statement
theorem problem_solution (m : ℝ) (a : ℝ) :
  (f m 1 = 5) →
  (∀ x > 0, x^2 + 4 ≥ a * x) →
  (m = 4 ∧ a ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l431_43119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l431_43127

theorem percentage_difference : ℝ := by
  -- Define 90% of 40
  let ninety_percent_of_forty : ℝ := 0.90 * 40
  -- Define 4/5 of 25
  let four_fifths_of_twentyfive : ℝ := (4/5) * 25
  -- The difference between these two values is 16
  have : ninety_percent_of_forty - four_fifths_of_twentyfive = 16 := by
    -- Proof steps would go here
    sorry
  -- Return the result
  exact 16


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l431_43127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_properties_l431_43113

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ := log x + (1/2) * x^2 - (a + 2) * x

theorem extreme_points_properties (a : ℝ) (m n : ℝ) 
  (h_extreme : ∀ x, x ∈ Set.Icc m n → (deriv (f a)) x = 0)
  (h_m_lt_n : m < n) :
  -- (1) When the tangent line at P(1, f(1)) is vertical to the y-axis, a = 0
  ((deriv (f a) 1 = 0 → a = 0) ∧ 
  -- (2) The range of f(m) + f(n) is (-∞, -3)
  (∀ y, y ∈ Set.Ioi (-3) → ¬(∃ x₁ x₂, x₁ ∈ Set.Icc m n ∧ x₂ ∈ Set.Icc m n ∧ f a x₁ + f a x₂ = y)) ∧
  -- (3) When n ≥ √e + 1/√e - 2, the maximum value of f(n) - f(m) is 1 - e/2 + 1/(2e)
  (n ≥ sqrt e + 1 / sqrt e - 2 → 
    f a n - f a m ≤ 1 - e / 2 + 1 / (2 * e))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_properties_l431_43113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segment_length_l431_43106

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 3 = 0

-- Define the perpendicularity condition
def perpendicular (C E F : ℝ × ℝ) : Prop := 
  (E.1 - C.1) * (F.1 - C.1) + (E.2 - C.2) * (F.2 - C.2) = 0

-- Define the midpoint condition
def is_midpoint (P E F : ℝ × ℝ) : Prop := 
  P.1 = (E.1 + F.1) / 2 ∧ P.2 = (E.2 + F.2) / 2

-- Define the angle condition
def angle_condition (A P B : ℝ × ℝ) : Prop :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) ≤ 0

-- State the theorem
theorem min_segment_length :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ 
    line_l B.1 B.2 ∧
    (∀ (E F P : ℝ × ℝ),
      circle_C E.1 E.2 ∧ 
      circle_C F.1 F.2 ∧ 
      perpendicular (1, 2) E F ∧ 
      is_midpoint P E F →
      angle_condition A P B) ∧
    (∀ (A' B' : ℝ × ℝ),
      line_l A'.1 A'.2 ∧ 
      line_l B'.1 B'.2 ∧
      (∀ (E F P : ℝ × ℝ),
        circle_C E.1 E.2 ∧ 
        circle_C F.1 F.2 ∧ 
        perpendicular (1, 2) E F ∧ 
        is_midpoint P E F →
        angle_condition A' P B') →
      (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ (A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segment_length_l431_43106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_max_l431_43178

theorem triangle_angle_sum_max (A B C : ℝ) : 
  A + B + C = Real.pi →
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 2 →
  ∃ (max : ℝ), max = Real.sqrt 5 ∧ 
    ∀ (x : ℝ), x = Real.cos A + Real.cos B + 2 * Real.cos C → x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_max_l431_43178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_graph_is_parabola_specific_quadratic_graph_is_parabola_l431_43122

/-- A quadratic function of the form y = a(x-h)^2 + k, where a ≠ 0 -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ
  a_nonzero : a ≠ 0

/-- Predicate to define if a set of points is a parabola -/
def IsParabola (P : Set (ℝ × ℝ)) : Prop := sorry

/-- The graph of a quadratic function is a parabola -/
theorem quadratic_graph_is_parabola (f : QuadraticFunction) :
  ∃ (P : Set (ℝ × ℝ)), IsParabola P ∧ P = {(x, y) | y = f.a * (x - f.h)^2 + f.k} :=
sorry

/-- The specific quadratic function y = 3(x-2)^2 + 6 -/
def specific_quadratic : QuadraticFunction :=
  { a := 3
    h := 2
    k := 6
    a_nonzero := by norm_num }

/-- The graph of y = 3(x-2)^2 + 6 is a parabola -/
theorem specific_quadratic_graph_is_parabola :
  ∃ (P : Set (ℝ × ℝ)), IsParabola P ∧ P = {(x, y) | y = 3 * (x - 2)^2 + 6} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_graph_is_parabola_specific_quadratic_graph_is_parabola_l431_43122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l431_43150

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis 5 and semi-minor axis 3 -/
def Ellipse := {p : Point | p.x^2 / 25 + p.y^2 / 9 = 1}

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The foci of the ellipse -/
def focus1 : Point := ⟨4, 0⟩
def focus2 : Point := ⟨-4, 0⟩

theorem ellipse_property (p : Point) (h : p ∈ Ellipse) :
  distance p focus1 = 5 → distance p focus2 = 5 := by
  sorry

#check ellipse_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l431_43150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_cosine_to_g_l431_43136

noncomputable def f (x : Real) : Real := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

noncomputable def g (x : Real) : Real := f (x + Real.pi/6)

theorem min_shift_cosine_to_g :
  ∃ (φ : Real), φ > 0 ∧
  (∀ (x : Real), g x = 2 * Real.cos (2 * (x - φ))) ∧
  (∀ (ψ : Real), ψ > 0 → (∀ (x : Real), g x = 2 * Real.cos (2 * (x - ψ))) → ψ ≥ φ) ∧
  φ = Real.pi/6 :=
by sorry

#check min_shift_cosine_to_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_cosine_to_g_l431_43136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_greater_than_eight_l431_43177

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : Fin n → ℝ :=
  λ i => a₁ * r ^ (i : ℕ)

theorem probability_greater_than_eight :
  let seq := geometric_sequence 1 (-3) 8
  let count_greater_than_eight := (Finset.filter (λ i => seq i > 8) Finset.univ).card
  (count_greater_than_eight : ℝ) / 8 = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_greater_than_eight_l431_43177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_l431_43182

/-- The vector v as a function of s -/
def v (s : ℝ) : Fin 3 → ℝ := fun i => match i with
  | 0 => 1 + 5*s
  | 1 => -2 + 3*s
  | 2 => 4 - 2*s

/-- The constant vector a -/
def a : Fin 3 → ℝ := fun i => match i with
  | 0 => -3
  | 1 => 6
  | 2 => 7

/-- The direction vector of v -/
def dir : Fin 3 → ℝ := fun i => match i with
  | 0 => 5
  | 1 => 3
  | 2 => -2

theorem closest_point (s : ℝ) :
  (∀ t : ℝ, ‖v t - a‖ ≥ ‖v s - a‖) ↔ s = -1/19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_l431_43182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_condition_l431_43118

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, |x| - x^2 ≠ a^2 - Real.sin (π * x)^2) ↔ 
  (a < -(Real.sqrt 5 / 2) ∨ a > Real.sqrt 5 / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_condition_l431_43118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_two_l431_43142

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^5
def curve2 (x y : ℝ) : Prop := x + y^3 = 2

-- Define the intersection points
def intersection1 : ℝ × ℝ := (1, 1)
def intersection2 : ℝ × ℝ := (1, -1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_is_two :
  (curve1 intersection1.1 intersection1.2 ∧ curve2 intersection1.1 intersection1.2) ∧
  (curve1 intersection2.1 intersection2.2 ∧ curve2 intersection2.1 intersection2.2) →
  distance intersection1 intersection2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_two_l431_43142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_half_planes_cover_l431_43125

-- Define the plane
def Plane : Type := ℝ × ℝ

-- Define an open half-plane
def OpenHalfPlane : Type := Plane → Prop

-- Define a covering of the plane
def Covers (halfPlanes : List OpenHalfPlane) : Prop :=
  ∀ p : Plane, ∃ h ∈ halfPlanes, h p

-- Theorem statement
theorem three_half_planes_cover
  (h1 h2 h3 h4 : OpenHalfPlane)
  (cover : Covers [h1, h2, h3, h4]) :
  ∃ (subset : List OpenHalfPlane),
    subset.length = 3 ∧
    subset ⊆ [h1, h2, h3, h4] ∧
    Covers subset :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_half_planes_cover_l431_43125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_l431_43147

def sequenceN (n : ℕ) : ℕ :=
  if n = 1 then 47
  else 47 * (Finset.sum (Finset.range n) (fun i => 10^(2*i)))

theorem only_first_prime :
  ∀ n > 1, ¬ Nat.Prime (sequenceN n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_l431_43147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_problem_l431_43194

/-- Proves that train A takes 6 hours to reach its destination after meeting train B -/
theorem train_meeting_problem (speed_A speed_B : ℝ) (time_B : ℝ) :
  speed_A = 110 →
  speed_B = 165 →
  time_B = 4 →
  (speed_B * time_B) / speed_A = 6 := by
  intro h_speed_A h_speed_B h_time_B
  -- Calculate the distance traveled by train B after meeting
  have distance : ℝ := speed_B * time_B
  -- Calculate the time taken by train A to cover this distance
  have time_A : ℝ := distance / speed_A
  -- Prove that time_A equals 6
  sorry

#check train_meeting_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_problem_l431_43194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l431_43149

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  O : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Calculates the surface area of a sphere given its radius -/
noncomputable def sphereSurfaceArea (radius : ℝ) : ℝ :=
  4 * Real.pi * radius^2

/-- Theorem: The surface area of the circumscribed sphere of tetrahedron OABC is 20π -/
theorem circumscribed_sphere_surface_area (t : Tetrahedron) : 
  t.O = ⟨0, 0, 0⟩ ∧ 
  t.A = ⟨2, 0, 0⟩ ∧ 
  t.B = ⟨0, 4, 0⟩ ∧ 
  t.C = ⟨0, 2, 2⟩ → 
  ∃ (center : Point3D) (radius : ℝ), 
    center = ⟨1, 2, 0⟩ ∧ 
    radius = Real.sqrt 5 ∧
    sphereSurfaceArea radius = 20 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l431_43149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l431_43183

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 2 * Real.exp x + x + Real.exp 1

-- State the theorem
theorem tangent_line_at_one :
  let f' : ℝ → ℝ := λ x ↦ (x - 1) * Real.exp x + 1
  (f' 1 = 1) ∧ 
  (f 1 = 1) ∧
  (∀ x y : ℝ, y = x ↔ y - 1 = f' 1 * (x - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l431_43183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l431_43166

-- Define the circle
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 4

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the condition for point P
def point_P_condition (x y : ℝ) : Prop :=
  ((x + 2)^2 + y^2) * ((x - 2)^2 + y^2) = (x^2 + y^2)^2

-- Theorem statement
theorem dot_product_range :
  ∀ x y : ℝ,
  circle_O x y →
  x^2 + y^2 < 4 →
  point_P_condition x y →
  -2 ≤ (x + 2) * (x - 2) + y * y ∧ (x + 2) * (x - 2) + y * y < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l431_43166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l431_43102

noncomputable def f (ω φ x : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem omega_range (ω φ : ℝ) : 
  ω > 0 ∧ 
  -π < φ ∧ φ < 0 ∧ 
  f ω φ 0 = Real.sqrt 3 / 2 ∧ 
  (∃! x, x ∈ Set.Ioo (-π/3) (π/3) ∧ f ω φ x = 0) →
  ω ∈ Set.Ioo 1 2 := by
  sorry

#check omega_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l431_43102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_l431_43140

/-- The sum of the lengths of the four sides of a square with a side length of 9 cm is 36 cm. -/
theorem square_perimeter (side_length : Real) : 
  side_length = 9 → 4 * side_length = 36 := by
  intro h
  rw [h]
  norm_num

#check square_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_l431_43140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l431_43100

/-- The radius of a sphere inscribed in a right circular cone -/
noncomputable def inscribed_sphere_radius (m r : ℝ) : ℝ :=
  r * (Real.sqrt (r^2 + m^2) - r) / m

/-- Theorem: The radius of a sphere inscribed in a right circular cone
    with height m and base radius r is equal to r(√(r^2 + m^2) - r) / m -/
theorem inscribed_sphere_radius_formula (m r : ℝ) (h₁ : m > 0) (h₂ : r > 0) :
  inscribed_sphere_radius m r = r * (Real.sqrt (r^2 + m^2) - r) / m :=
by
  -- Unfold the definition of inscribed_sphere_radius
  unfold inscribed_sphere_radius
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l431_43100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_multiplication_result_l431_43108

theorem matrix_multiplication_result (N : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : N.mulVec (![1, 3]) = ![2, 5])
  (h2 : N.mulVec (![4, -2]) = ![0, 3]) :
  N.mulVec (![9, 5]) = ![38/7, 128/7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_multiplication_result_l431_43108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l431_43101

/-- The nested radical function -/
noncomputable def nestedRadical : ℕ → ℝ
| 0     => Real.sqrt (1 + 2018 * 2020)
| (n+1) => Real.sqrt (1 + (2018 - n : ℝ) * nestedRadical n)

/-- The theorem stating that the nested radical equals 3 -/
theorem nested_radical_equals_three : nestedRadical 2018 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l431_43101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_frequency_and_number_l431_43112

/-- Given a sample size and ratios of rectangle heights in a histogram,
    calculate the frequency and frequency number of a specific group. -/
def calculate_frequency_and_number (sample_size : ℕ) (ratios : List ℕ) (group_index : ℕ) :
  Option (ℚ × ℕ) :=
  if h : group_index < ratios.length then
    let total_ratio : ℕ := ratios.sum
    let group_ratio : ℕ := ratios[group_index]'h
    let frequency_number : ℕ := (sample_size * group_ratio) / total_ratio
    let frequency : ℚ := frequency_number / sample_size
    some (frequency, frequency_number)
  else
    none

/-- Theorem stating that for a sample size of 30 and rectangle height ratios of 2:4:3:1,
    the frequency and frequency number of the second group are 0.4 and 12 respectively. -/
theorem second_group_frequency_and_number :
  let sample_size : ℕ := 30
  let ratios : List ℕ := [2, 4, 3, 1]
  let result := calculate_frequency_and_number sample_size ratios 1
  result = some (2/5, 12) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_frequency_and_number_l431_43112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l431_43131

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / (e.a^2) + p.y^2 / (e.b^2) = 1

/-- The main theorem to be proved -/
theorem ellipse_focal_distance 
  (F₁ : Point) 
  (F₂ : Point) 
  (P : Point) 
  (M : Point) 
  (e : Ellipse)
  (h1 : F₁.x = -6 ∧ F₁.y = 0)
  (h2 : F₂.x = 6 ∧ F₂.y = 0)
  (h3 : P.x = 5 ∧ P.y = 2)
  (h4 : e.c = 6)
  (h5 : isOnEllipse e P)
  (h6 : isOnEllipse e M)
  (h7 : distance M F₁ = 2 * Real.sqrt 5)
  : distance M F₂ = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l431_43131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_term_of_sequence_l431_43173

noncomputable section

/-- The nth term of a geometric sequence with first term a and common ratio r -/
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

/-- The first term of our specific sequence -/
def a₁ : ℝ := 12

/-- The second term of our specific sequence -/
def a₂ : ℝ := 4

/-- The common ratio of our specific sequence -/
def r : ℝ := a₂ / a₁

theorem twelfth_term_of_sequence : geometric_sequence a₁ r 12 = 12 / 177147 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_term_of_sequence_l431_43173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_l431_43105

def n1 : Fin 3 → ℝ := ![1, 2, 1]
def n2 : Fin 3 → ℝ := ![-3, 1, 1]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

theorem perpendicular_planes (α β : Set (Fin 3 → ℝ)) 
  (hα : ∀ x ∈ α, dot_product x n1 = 0)
  (hβ : ∀ x ∈ β, dot_product x n2 = 0) :
  dot_product n1 n2 = 0 → (∀ x ∈ α, ∀ y ∈ β, dot_product x y = 0) :=
by
  sorry

#eval dot_product n1 n2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_l431_43105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_hexagons_in_triangle_l431_43179

/-- The side length of the large equilateral triangle -/
noncomputable def large_triangle_side : ℝ := 12

/-- The side length of the small regular hexagons -/
noncomputable def hexagon_side : ℝ := 1

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The area of a regular hexagon with side length s -/
noncomputable def regular_hexagon_area (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

/-- The theorem stating the maximum number of hexagons that can fit in the triangle -/
theorem max_hexagons_in_triangle : 
  ⌊equilateral_triangle_area large_triangle_side / regular_hexagon_area hexagon_side⌋ = 24 := by
  sorry

#check max_hexagons_in_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_hexagons_in_triangle_l431_43179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_l431_43133

-- Define the function
noncomputable def f (x : ℝ) : ℝ := |Real.sin (2 * x - Real.pi / 6)|

-- State the theorem
theorem symmetry_axis (x : ℝ) :
  f (π/3 + x) = f (π/3 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_l431_43133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mariana_test_score_l431_43167

theorem mariana_test_score : 
  let test1_problems : ℕ := 15
  let test2_problems : ℕ := 20
  let test3_problems : ℕ := 40
  let test1_score : ℚ := 60 / 100
  let test2_score : ℚ := 85 / 100
  let test3_score : ℚ := 75 / 100
  let total_problems : ℕ := test1_problems + test2_problems + test3_problems
  let correct_answers : ℚ := test1_score * test1_problems + 
                             test2_score * test2_problems + 
                             test3_score * test3_problems
  let overall_score : ℚ := correct_answers / total_problems
  ⌊overall_score * 100 + 0.5⌋ = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mariana_test_score_l431_43167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_example_l431_43148

/-- Calculate simple interest given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (rate / 100) * time

/-- Proof that the simple interest is 160 given the specified conditions -/
theorem simple_interest_example : simple_interest 400 20 2 = 160 := by
  -- Unfold the definition of simple_interest
  unfold simple_interest
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_div_cancel']
  -- Check that the result is equal to 160
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_example_l431_43148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_implies_m_bound_l431_43153

noncomputable def f (x m : ℝ) : ℝ := (1/2) * Real.log (x^2 - m) + (m * (x - 1)) / (x^2 - m)

theorem f_lower_bound_implies_m_bound 
  (h1 : m > 0)
  (h2 : ∀ x > Real.sqrt m, f x m ≥ 1) :
  m ≥ (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_implies_m_bound_l431_43153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_M_l431_43185

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem stating that the sum of digits of M is 9 -/
theorem sum_of_digits_M (M : ℕ+) (h : (M : ℝ) ^ 2 = (0.04 : ℝ) ^ 32 * 256 ^ (25/2) * 3 ^ 100) :
  sum_of_digits M.val = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_M_l431_43185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_on_line_and_parallel_l431_43187

noncomputable def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 1, 2 * t + 3)

noncomputable def vector : ℝ × ℝ := (16, 32/3)

noncomputable def parallel_vector : ℝ × ℝ := (3, 2)

theorem vector_on_line_and_parallel :
  ∃ t : ℝ, line_param t = vector ∧
  ∃ k : ℝ, vector.1 = k * parallel_vector.1 ∧ vector.2 = k * parallel_vector.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_on_line_and_parallel_l431_43187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_correct_l431_43172

/-- A hyperbola with foci on the y-axis and asymptotic equations y = ±2x -/
structure Hyperbola where
  /-- The hyperbola has foci on the y-axis -/
  foci_on_y_axis : Bool
  /-- The asymptotic equations are y = ±2x -/
  asymptotic_equations : ℝ → ℝ → Prop

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 4 - x^2 = 1

theorem hyperbola_equation_correct (h : Hyperbola) :
  h.foci_on_y_axis ∧
  (∀ x y, h.asymptotic_equations x y ↔ y = 2*x ∨ y = -2*x) →
  ∀ x y, h.asymptotic_equations x y ↔ hyperbola_equation x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_correct_l431_43172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_taxicab_numbers_l431_43159

theorem infinite_taxicab_numbers :
  ∃ f : ℕ → ℕ × ℕ × ℕ × ℕ,
    Function.Injective f ∧
    (∀ n, let (a, b, c, d) := f n;
      a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
      Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 1 ∧
      ({a, b} : Set ℕ) ≠ {c, d} ∧
      a^3 + b^3 = c^3 + d^3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_taxicab_numbers_l431_43159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_ratio_l431_43196

/-- A circle with a given radius -/
structure Circle where
  radius : ℝ

/-- A regular hexagon -/
structure RegularHexagon where
  sideLength : ℝ

/-- A regular hexagon inscribed in a circle -/
noncomputable def inscribedHexagon (c : Circle) : RegularHexagon where
  sideLength := c.radius

/-- A regular hexagon circumscribed around a circle -/
noncomputable def circumscribedHexagon (c : Circle) : RegularHexagon where
  sideLength := 2 * c.radius / Real.sqrt 3

/-- The area of a regular hexagon -/
noncomputable def areaHexagon (h : RegularHexagon) : ℝ :=
  3 * Real.sqrt 3 / 2 * h.sideLength ^ 2

/-- The theorem stating the ratio of areas of inscribed and circumscribed hexagons -/
theorem hexagon_area_ratio (c : Circle) :
  areaHexagon (inscribedHexagon c) / areaHexagon (circumscribedHexagon c) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_ratio_l431_43196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l431_43174

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (1 - x)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.log (1 + x)

-- Define the domain M of f
def M : Set ℝ := {x | x < 1}

-- Define the domain N of g
def N : Set ℝ := {x | x > -1}

-- Theorem statement
theorem domain_intersection :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l431_43174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l431_43157

/-- The function f(x) defined for the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - x

/-- The theorem statement -/
theorem inequality_solution_range (a : ℝ) :
  (0 < a ∧ a < 1) →
  (∃ x : ℝ, x ≥ 1 ∧ f a x < a / (a - 1)) ↔ 
  (0 < a ∧ a < Real.sqrt 2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l431_43157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_roll_volume_difference_l431_43134

/-- The volume of a cylinder with given height and circumference -/
noncomputable def cylinderVolume (height : ℝ) (circumference : ℝ) : ℝ :=
  (circumference ^ 2 * height) / (4 * Real.pi)

/-- The problem statement -/
theorem paper_roll_volume_difference : 
  let v1 := cylinderVolume 12 10
  let v2 := cylinderVolume 10 12
  Real.pi * |v1 - v2| = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_roll_volume_difference_l431_43134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_digits_exist_l431_43199

/-- Represents a 6-digit number -/
def SixDigitNumber := Fin 1000000

/-- The first number in the addition problem -/
def num1 : Nat := 653479

/-- The second number in the addition problem -/
def num2 : Nat := 938521

/-- The incorrect sum given in the problem -/
def incorrect_sum : Nat := 1616200

/-- Function to replace all occurrences of one digit with another in a number -/
def replace_digit (n : Nat) (d e : Fin 10) : Nat :=
  sorry

/-- Theorem stating that there exist two digits d and e that correct the sum and sum to 10 -/
theorem correct_digits_exist : 
  ∃ (d e : Fin 10), 
    (replace_digit num1 d e + replace_digit num2 d e = incorrect_sum) ∧ 
    (d.val + e.val = 10) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_digits_exist_l431_43199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_inhabitants_l431_43116

/-- Represents an inhabitant of the island -/
inductive Inhabitant
| Knight
| Knave
deriving BEq, Repr

/-- The statements made by each inhabitant -/
def statement (inhabitants : List Inhabitant) (n : Nat) : Bool :=
  match n with
  | 0 => inhabitants.get? 0 == some Inhabitant.Knight
  | 1 => inhabitants.get? 0 == some Inhabitant.Knight
  | 2 => (inhabitants.take 2).count Inhabitant.Knave ≥ 1
  | 3 => (inhabitants.take 3).count Inhabitant.Knave ≥ 2
  | 4 => (inhabitants.take 4).count Inhabitant.Knight ≥ 2
  | 5 => (inhabitants.take 5).count Inhabitant.Knave ≥ 2
  | 6 => (inhabitants.take 6).count Inhabitant.Knight ≥ 4
  | _ => false

/-- The main theorem -/
theorem island_inhabitants :
  ∃ (inhabitants : List Inhabitant),
    inhabitants.length = 7 ∧
    (∀ n, n < 7 → (inhabitants.get? n == some Inhabitant.Knight) = statement inhabitants n) ∧
    inhabitants.count Inhabitant.Knight = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_inhabitants_l431_43116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_one_over_cube_root_two_l431_43132

/-- A geometric progression where the sum of the first six terms is three times 
    the sum of the first three terms -/
structure SpecialGeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sum_condition : a * (1 - r^6) / (1 - r) = 3 * (a * (1 - r^3) / (1 - r))
  r_not_one : r ≠ 1
  a_not_zero : a ≠ 0

/-- The ratio of the first term to the common ratio in the special geometric progression -/
noncomputable def ratio (gp : SpecialGeometricProgression) : ℝ := gp.a / gp.r

/-- Theorem: The ratio of the first term to the common ratio is 1/∛2 -/
theorem ratio_is_one_over_cube_root_two (gp : SpecialGeometricProgression) : 
  ratio gp = 1 / (2 ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_one_over_cube_root_two_l431_43132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l431_43160

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.cos (α + π/4) = 1/3) 
  (h2 : 0 < α ∧ α < π/2) : 
  Real.sin α = (4 - Real.sqrt 2) / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l431_43160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_elements_A_l431_43195

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else (x + 7) / 2

def A : Set ℕ :=
  {x : ℕ | f (f (f x)) = x}

theorem count_elements_A : Finset.card (Finset.filter (fun x => f (f (f x)) = x) (Finset.range 8)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_elements_A_l431_43195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_z_value_proof_l431_43169

def mean_equality_implies_z_value (z : ℝ) : Prop :=
  (5 + 10 + 20) / 3 = (15 + z) / 2 → z = 25 / 3

theorem mean_equality_implies_z_value_proof : ∀ z : ℝ, mean_equality_implies_z_value z := by
  intro z
  intro h
  -- The proof goes here
  sorry

#check mean_equality_implies_z_value_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_z_value_proof_l431_43169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_of_three_numbers_l431_43121

theorem max_of_three_numbers (a b c : ℝ) (ha : a = 3) (hb : b = 7) (hc : c = 2) :
  max a (max b c) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_of_three_numbers_l431_43121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_inscribed_ellipse_l431_43176

/-- Given an ellipse inscribed in a rectangle, with the ellipse's minor radius
    and the rectangle's length-to-width ratio, prove the area of the rectangle. -/
theorem rectangle_area_with_inscribed_ellipse 
  (minor_radius : ℝ) 
  (length_width_ratio : ℚ) : ℝ :=
by
  -- Assume the minor radius is 4
  have h1 : minor_radius = 4 := by sorry
  -- Assume the length to width ratio is 3:2
  have h2 : length_width_ratio = 3/2 := by sorry
  -- The width of the rectangle is twice the minor radius
  let width := 2 * minor_radius
  -- The length of the rectangle is 3/2 times the width
  let length := (3/2) * width
  -- The area of the rectangle is length * width
  let area := length * width
  -- Prove that the area equals 96
  have h3 : area = 96 := by sorry
  exact area

-- Remove the #eval statement as it's causing issues
-- #eval rectangle_area_with_inscribed_ellipse 4 (3/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_inscribed_ellipse_l431_43176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_S_l431_43181

def S : Set Nat := sorry

axiom S_subset : S ⊆ Finset.range 109

axiom S_nonempty : S.Nonempty

axiom condition_i : ∀ {a b : Nat}, a ∈ S → b ∈ S → a ≠ b → ∃ c ∈ S, a * c = 1 ∧ b * c = 1

axiom condition_ii : ∀ {a b : Nat}, a ∈ S → b ∈ S → ∃ c' ∈ S, c' ≠ a ∧ c' ≠ b ∧ a * c' > 1 ∧ b * c' > 1

theorem max_elements_S : ∀ (h : Fintype S), Fintype.card S ≤ 76 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_S_l431_43181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_gender_relation_l431_43162

/-- Represents the contingency table for liking sports vs gender --/
structure ContingencyTable where
  boys_not_like : ℕ
  boys_like : ℕ
  girls_not_like : ℕ
  girls_like : ℕ

/-- Calculates the chi-square statistic for a contingency table --/
noncomputable def chi_square (table : ContingencyTable) : ℝ :=
  let n := table.boys_not_like + table.boys_like + table.girls_not_like + table.girls_like
  let a := table.boys_not_like
  let b := table.boys_like
  let c := table.girls_not_like
  let d := table.girls_like
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating that liking sports is related to gender with 99% certainty --/
theorem sports_gender_relation : 
  ∃ (table : ContingencyTable), 
    table.boys_not_like + table.boys_like = 40 ∧ 
    table.girls_not_like + table.girls_like = 60 ∧
    table.boys_not_like = 8 ∧
    table.girls_not_like = 32 ∧
    chi_square table > 6.635 := by
  sorry

#check sports_gender_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_gender_relation_l431_43162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_passes_through_point_l431_43156

-- Define the original function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 3) + 1

-- State the theorem
theorem inverse_function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ g : ℝ → ℝ, Function.RightInverse g (f a) ∧ g 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_passes_through_point_l431_43156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_on_circle_l431_43139

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Defines an ellipse with foci F₁ and F₂, and constant sum 2a -/
structure Ellipse where
  F₁ : Point
  F₂ : Point
  a : ℝ

/-- Represents a point on an ellipse -/
structure PointOnEllipse (E : Ellipse) where
  P : Point
  sum_condition : distance P E.F₁ + distance P E.F₂ = 2 * E.a

/-- Theorem: Q lies on a circle with center F₁ -/
theorem q_on_circle (E : Ellipse) (P : PointOnEllipse E) 
  (Q : Point) (h : distance P.P Q = distance P.P E.F₂) :
  distance Q E.F₁ = 2 * E.a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_on_circle_l431_43139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₃_f₄_same_cluster_l431_43164

/-- Definition of "functions of the same cluster" -/
def same_cluster (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = g (x + k)

/-- The given functions -/
noncomputable def f₁ (x : ℝ) : ℝ := Real.sin x * Real.cos x
noncomputable def f₂ (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x) + 2
noncomputable def f₃ (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 4)
noncomputable def f₄ (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

/-- Theorem stating that f₃ and f₄ are functions of the same cluster -/
theorem f₃_f₄_same_cluster : same_cluster f₃ f₄ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₃_f₄_same_cluster_l431_43164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_calculation_l431_43123

noncomputable def total_investment : ℝ := 7000 + 11000 + 18000 + 13000 + 21000 + 15000 + 9000

noncomputable def a_investment : ℝ := 7000
noncomputable def b_investment : ℝ := 11000
noncomputable def b_share : ℝ := 3600

noncomputable def total_profit : ℝ := b_share * total_investment / b_investment

noncomputable def a_share : ℝ := total_profit * a_investment / total_investment

theorem a_share_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ |a_share - 2292.34| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_calculation_l431_43123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_perimeter_l431_43161

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

noncomputable def Triangle.area (t : Triangle) : ℝ := 
  (1/2) * t.b * t.c * Real.sin t.A

def Triangle.perimeter (t : Triangle) : ℝ := 
  t.a + t.b + t.c

theorem triangle_area_and_perimeter (t : Triangle) 
  (h1 : t.b * t.c = 1)
  (h2 : t.a^2 - t.b * t.c = (t.b - t.c)^2)
  (h3 : Real.cos t.B * Real.cos t.C = 1/4) :
  t.area = Real.sqrt 3 / 4 ∧ t.perimeter = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_perimeter_l431_43161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pentagon_angle_sum_l431_43138

theorem square_pentagon_angle_sum (a b : Real) : a + b = 324 := by
  -- Define the properties of the square
  let square_angle : Real := 90
  -- Define the properties of the regular pentagon
  let pentagon_angle : Real := 108
  -- Define the sum of angles in a triangle
  let triangle_angle_sum : Real := 180
  -- Define the sum of angles in a quadrilateral
  let quadrilateral_angle_sum : Real := 360
  
  -- The proof is omitted
  sorry

#check square_pentagon_angle_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pentagon_angle_sum_l431_43138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_max_value_phi_l431_43110

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.cos (2 * x + φ) + Real.sin x ^ 2

-- Theorem 1
theorem monotonic_increasing_interval (k : ℤ) :
  ∀ φ : ℝ, 0 ≤ φ ∧ φ < π →
  φ = π / 6 →
  ∀ x : ℝ, k * π - 2 * π / 3 ≤ x ∧ x ≤ k * π - π / 6 →
  Monotone (f (π / 6)) := by
  sorry

-- Theorem 2
theorem max_value_phi (φ : ℝ) :
  0 ≤ φ ∧ φ < π →
  (∀ x : ℝ, f φ x ≤ 3 / 2) ∧ (∃ x : ℝ, f φ x = 3 / 2) →
  φ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_max_value_phi_l431_43110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_ordinate_l431_43165

/-- The parabola x^2 = 4y with focus (0, 1) -/
def Parabola (P : ℝ × ℝ) : Prop :=
  P.1^2 = 4 * P.2

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (0, 1)

/-- The distance between two points in ℝ² -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem parabola_point_ordinate 
  (P : ℝ × ℝ) 
  (h1 : Parabola P) 
  (h2 : distance P Focus = 3) : 
  P.2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_ordinate_l431_43165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_of_concentric_circles_l431_43189

open Real

-- Define the radii of the two circles
variable (a b : ℝ)

-- Define the area of the ring
noncomputable def ring_area (a b : ℝ) : ℝ := π * (a^2 - b^2)

-- Define the length of the chord
noncomputable def chord_length (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

-- Theorem statement
theorem chord_length_of_concentric_circles 
  (h : ring_area a b = 25 * π) : 
  chord_length a b = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_of_concentric_circles_l431_43189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decomposition_terminates_l431_43115

/-- Represents a unit fraction 1/n -/
def UnitFraction := { n : ℕ // n > 0 }

/-- The algorithm for choosing the next unit fraction -/
def nextUnitFraction (remaining : ℚ) (prev : UnitFraction) : UnitFraction :=
  sorry

/-- The decomposition algorithm -/
def decompose (a b : ℕ) (h : b > 0) : List UnitFraction :=
  sorry

/-- Helper function to sum unit fractions -/
def sumUnitFractions (l : List UnitFraction) : ℚ :=
  l.foldl (λ acc f => acc + (1 : ℚ) / f.val) 0

theorem decomposition_terminates (a b : ℕ) (h : b > 0) :
  ∃ (l : List UnitFraction), (l = decompose a b h) ∧ 
  (sumUnitFractions l = (a : ℚ) / b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decomposition_terminates_l431_43115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_commutator_properties_l431_43180

open Matrix

theorem matrix_commutator_properties (n : ℕ) (hn : n ≥ 2) :
  ∃ (A B : Matrix (Fin n) (Fin n) ℂ), 
    A ^ 2 * B = A ∧ 
    (A * B - B * A) ^ 2 = 0 ∧
    ∀ k : ℕ, k ≤ n / 2 → ∃ (A' B' : Matrix (Fin n) (Fin n) ℂ), 
      A' ^ 2 * B' = A' ∧ 
      Matrix.rank (A' * B' - B' * A') = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_commutator_properties_l431_43180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_existence_l431_43191

theorem distinct_remainders_existence (p : ℕ) (hp : Nat.Prime p) (a : Fin p → ℤ) :
  ∃ k : ℤ, Finset.card (Finset.filter (λ i : Fin p => ∃ j : Fin p, (a j + j.val * k) % p = (a i + i.val * k) % p) Finset.univ) ≤ p / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_existence_l431_43191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l431_43135

-- Define the equation
noncomputable def f (x a : ℝ) : ℝ := (Real.cos x)^2 - Real.cos x + a

-- Define the domain
def domain : Set ℝ := Set.Icc (Real.pi/4) (4*Real.pi/3)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ∈ domain ∧ x₂ ∈ domain ∧ x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ∧
  (∀ (x₃ : ℝ), x₃ ∈ domain ∧ f x₃ a = 0 → x₃ = x₁ ∨ x₃ = x₂) →
  a ∈ Set.Ioc (-2) (-3/4) ∪ Set.Icc ((Real.sqrt 2 - 1)/2) (1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l431_43135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_dimensions_l431_43171

noncomputable section

-- Define the cylinder
def cylinder_radius : ℝ := 2
def cylinder_height : ℝ := 3

-- Define the pyramid
def pyramid_height : ℝ := 10

-- Define the volume ratio
def volume_ratio : ℝ := 1/2

-- Theorem statement
theorem pyramid_base_dimensions :
  ∀ (base_length base_width : ℝ),
  (base_length > 0) →
  (base_width > 0) →
  (base_width ≥ 2 * cylinder_radius) →
  (π * cylinder_radius^2 * cylinder_height = volume_ratio * (1/3) * base_length * base_width * pyramid_height) →
  (base_length = 18 * π / 5 ∧ base_width = 4) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_dimensions_l431_43171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_theorem_l431_43130

noncomputable def a : ℝ × ℝ := (3, 4)
noncomputable def b : ℝ × ℝ := (9, 12)
noncomputable def c : ℝ × ℝ := (4, -3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

noncomputable def angle_cos (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2))

theorem vector_angle_theorem :
  parallel a b ∧ perpendicular a c →
  angle_cos ((2 * a.1 - b.1, 2 * a.2 - b.2)) ((a.1 + c.1, a.2 + c.2)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_theorem_l431_43130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_cube_equality_l431_43190

-- Define the fractional part function as noncomputable
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- State the theorem
theorem fractional_cube_equality (x : ℝ) :
  frac ((x + 1)^3) = x^3 ↔ 0 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_cube_equality_l431_43190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_inequality_l431_43188

-- Define the even function f
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then 2 * x else 2 * (-x)

-- State the theorem
theorem even_function_inequality (x : ℝ) :
  f (1 - 2 * x) < f 3 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_inequality_l431_43188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l431_43170

def n : ℕ := 2^40 * 3^25 * 5^10

theorem divisors_count : 
  (Finset.filter (fun d => d ∣ n^2 ∧ d < n ∧ ¬(d ∣ n)) (Finset.range (n + 1))).card = 31514 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l431_43170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pass_sequence_count_l431_43192

/-- Represents a student in the handkerchief passing game -/
inductive Student
| A
| B
| C
| D

/-- A pass sequence is a list of 5 students representing the order of passes -/
def PassSequence := List Student

/-- Checks if a pass sequence is valid according to the game rules -/
def isValidPassSequence (seq : PassSequence) : Prop :=
  seq.length = 5 ∧ 
  seq.head? = some Student.A ∧ 
  seq.getLast? = some Student.A ∧
  ∀ i, i < 4 → seq.get? i ≠ seq.get? (i + 1)

/-- The total number of valid pass sequences -/
def totalValidPassSequences : ℕ := sorry

/-- Theorem: The number of valid pass sequences is 60 -/
theorem pass_sequence_count : totalValidPassSequences = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pass_sequence_count_l431_43192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_inequality_with_floor_l431_43129

open BigOperators Finset Real

theorem rearrangement_inequality_with_floor (n : ℕ+) (x y : Fin n → ℝ) (α : ℝ) 
  (hx : Monotone x) (hy : Antitone y)
  (h : ∑ i in Finset.range n, (i + 1 : ℝ) * x i ≥ ∑ i in Finset.range n, (i + 1 : ℝ) * y i) :
  ∑ i in Finset.range n, x i * ⌊(i + 1 : ℝ) * α⌋ ≥ ∑ i in Finset.range n, y i * ⌊(i + 1 : ℝ) * α⌋ := by
  sorry

#check rearrangement_inequality_with_floor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_inequality_with_floor_l431_43129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_swinging_time_l431_43145

/-- Represents the movement of a Lamplighter monkey -/
structure MonkeyMovement where
  swingingSpeed : ℚ
  runningSpeed : ℚ
  runningTime : ℚ
  totalDistance : ℚ

/-- Calculates the time spent swinging for a Lamplighter monkey -/
noncomputable def swingingTime (m : MonkeyMovement) : ℚ :=
  (m.totalDistance - m.runningSpeed * m.runningTime) / m.swingingSpeed

/-- Theorem stating that under given conditions, the monkey swings for 10 seconds -/
theorem monkey_swinging_time (m : MonkeyMovement)
  (h1 : m.swingingSpeed = 10)
  (h2 : m.runningSpeed = 15)
  (h3 : m.runningTime = 5)
  (h4 : m.totalDistance = 175) :
  swingingTime m = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_swinging_time_l431_43145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_GHI_XYZ_l431_43111

-- Define the triangle XYZ
variable (X Y Z : ℝ × ℝ)

-- Define points M, N, O on the sides of the triangle
noncomputable def M (X Y Z : ℝ × ℝ) : ℝ × ℝ := ((3/5) * Y.1 + (2/5) * Z.1, (3/5) * Y.2 + (2/5) * Z.2)
noncomputable def N (X Y Z : ℝ × ℝ) : ℝ × ℝ := ((3/5) * Z.1 + (2/5) * X.1, (3/5) * Z.2 + (2/5) * X.2)
noncomputable def O (X Y Z : ℝ × ℝ) : ℝ × ℝ := ((3/5) * X.1 + (2/5) * Y.1, (3/5) * X.2 + (2/5) * Y.2)

-- Define the intersection points G, H, I
noncomputable def G (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def H (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def I (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define a function to calculate the area of a triangle
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_ratio_GHI_XYZ (X Y Z : ℝ × ℝ) :
  triangleArea (G X Y Z) (H X Y Z) (I X Y Z) / triangleArea X Y Z = 36 / 125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_GHI_XYZ_l431_43111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_inter_B_eq_open_1_closed_2_l431_43152

def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B : Set ℝ := {x | Real.exp ((x - 1) * Real.log 2) > 1}

theorem A_inter_B_eq_open_1_closed_2 : A ∩ B = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_inter_B_eq_open_1_closed_2_l431_43152
