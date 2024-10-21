import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_two_critical_points_f_less_than_exp_plus_sin_minus_one_l1020_102062

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (deriv (f a)) x + 1 / (x + 1)

-- Theorem 1: g has two critical points iff 0 < a < 1/4
theorem g_two_critical_points (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ deriv (g a) x₁ = 0 ∧ deriv (g a) x₂ = 0) ↔ 0 < a ∧ a < 1/4 := by
  sorry

-- Theorem 2: f(x) < e^x + sin x - 1 when a = 1
theorem f_less_than_exp_plus_sin_minus_one (x : ℝ) (h : x > 0) :
  f 1 x < Real.exp x + Real.sin x - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_two_critical_points_f_less_than_exp_plus_sin_minus_one_l1020_102062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_downstream_l1020_102065

/-- Calculates the distance traveled downstream given the rowing speed in still water,
    the current speed, and the time taken. -/
noncomputable def distance_downstream (rowing_speed : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
  let effective_speed := rowing_speed + current_speed
  let speed_ms := effective_speed * (1000 / 3600)
  speed_ms * time

/-- Theorem stating that given the specified conditions, the distance covered downstream
    is approximately 45.496360291176705 meters. -/
theorem distance_covered_downstream :
  let rowing_speed := (9.5 : ℝ)
  let current_speed := (8.5 : ℝ)
  let time := (9.099272058235341 : ℝ)
  let result := distance_downstream rowing_speed current_speed time
  abs (result - 45.496360291176705) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_downstream_l1020_102065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grey_necklace_purchase_l1020_102093

/-- Theorem: Mr. Grey bought 2 necklaces -/
theorem grey_necklace_purchase : ℕ := by
  -- Define the constants
  let polo_shirt_price : ℕ := 26
  let polo_shirt_count : ℕ := 3
  let necklace_price : ℕ := 83
  let computer_game_price : ℕ := 90
  let rebate : ℕ := 12
  let total_cost_after_rebate : ℕ := 322

  -- Define the function to calculate the number of necklaces
  let calculate_necklaces : ℕ :=
    let total_before_rebate := total_cost_after_rebate + rebate
    let polo_shirts_cost := polo_shirt_price * polo_shirt_count
    let remaining_cost := total_before_rebate - polo_shirts_cost - computer_game_price
    remaining_cost / necklace_price

  -- The theorem statement
  have : calculate_necklaces = 2 := by
    -- Proof goes here
    sorry

  exact 2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grey_necklace_purchase_l1020_102093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1020_102012

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  if x < t then x + 6 else x^2 + 2*x

theorem range_of_t (t : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f t x = y) → t ∈ Set.Icc (-7 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1020_102012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_l1020_102077

-- Define the sets
def R : Set ℝ := Set.univ
def N_star : Set ℕ := {n : ℕ | n > 0}
def Z : Set ℤ := Set.univ

-- Define the intersections
def R_cap_N_star : Set ℝ := R ∩ {x : ℝ | ∃ n : ℕ, n > 0 ∧ x = n}
def R_cap_Z : Set ℝ := R ∩ {x : ℝ | ∃ z : ℤ, x = z}

-- State the theorem
theorem necessary_condition : 
  (∃ x, x ∈ R_cap_Z ∧ x ∉ R_cap_N_star) ∧ 
  (∀ x, x ∈ R_cap_N_star → x ∈ R_cap_Z) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_l1020_102077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_symmetry_l1020_102027

/-- The original line -/
def original_line (x y : ℝ) : Prop := 2 * x + 3 * y - 6 = 0

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := 2 * x + 3 * y + 8 = 0

/-- The point of symmetry -/
def symmetry_point : ℝ × ℝ := (1, -1)

/-- The distance from a point to a line -/
noncomputable def distance_to_line (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

theorem line_symmetry :
  ∀ (x y : ℝ),
    distance_to_line x y 2 3 (-6) = distance_to_line x y 2 3 8 ∧
    (x - symmetry_point.1) = (symmetry_point.1 - x) ∧
    (y - symmetry_point.2) = (symmetry_point.2 - y) := by
  sorry

#check line_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_symmetry_l1020_102027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_digit_sum_218_l1020_102011

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The smallest natural number with digit sum 218 has 25 digits -/
theorem smallest_number_with_digit_sum_218 :
  ∃ (n : ℕ), 
    (∀ m : ℕ, (digit_sum m = 218) → n ≤ m) ∧ 
    (digit_sum n = 218) ∧ 
    (num_digits n = 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_digit_sum_218_l1020_102011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karen_speed_is_60_l1020_102052

/-- Karen's average driving speed in mph -/
noncomputable def karen_speed : ℝ := 60

/-- Tom's average driving speed in mph -/
noncomputable def tom_speed : ℝ := 45

/-- Karen's late start time in hours -/
noncomputable def late_start : ℝ := 4 / 60

/-- Distance Karen beats Tom by in miles -/
noncomputable def winning_margin : ℝ := 4

/-- Distance Tom drives before Karen wins in miles -/
noncomputable def tom_distance : ℝ := 24

theorem karen_speed_is_60 :
  karen_speed = 60 ∧
  tom_speed = 45 ∧
  late_start = 4 / 60 ∧
  winning_margin = 4 ∧
  tom_distance = 24 →
  karen_speed = 60 := by
  sorry

#check karen_speed_is_60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karen_speed_is_60_l1020_102052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_shaded_square_l1020_102096

/-- Represents a rectangle in the grid --/
structure Rectangle where
  row : Fin 3
  left : Fin 1005
  right : Fin 1005
  h_left_lt_right : left < right

/-- The total number of rectangles in the grid --/
def total_rectangles : ℕ := 3 * Nat.choose 1005 2

/-- Checks if a rectangle contains a shaded square --/
def contains_shaded (r : Rectangle) : Prop :=
  r.left < 502 ∧ r.right > 502

/-- The number of rectangles containing a shaded square --/
def shaded_rectangles : ℕ := 3 * 502 * 502

/-- The main theorem --/
theorem probability_no_shaded_square :
  1 - (shaded_rectangles : ℚ) / total_rectangles = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_shaded_square_l1020_102096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_correct_statement_A_correct_statement_C_correct_statement_D_l1020_102081

-- Define the basic geometric objects
variable (Point Plane Line : Type)

-- Define the relations
variable (belongs_to_plane : Point → Plane → Prop)
variable (belongs_to_line : Point → Line → Prop)
variable (subset_of : Line → Plane → Prop)
variable (plane_intersection : Plane → Plane → Set Point)
variable (line_intersection : Line → Line → Set Point)

-- Define the axioms and statements
axiom axiom1 {l : Line} {α : Plane} {A B : Point} :
  belongs_to_line A l → belongs_to_line B l → belongs_to_plane A α → belongs_to_plane B α → subset_of l α

theorem incorrect_statement : ¬ ∀ (A B C : Point), ∃ (α : Plane), belongs_to_plane A α ∧ belongs_to_plane B α ∧ belongs_to_plane C α :=
sorry

theorem correct_statement_A {α β : Plane} {l : Line} {P : Point} :
  P ∈ (plane_intersection α β) → (plane_intersection α β : Set Point) = {P | belongs_to_line P l} → belongs_to_line P l :=
sorry

theorem correct_statement_C {a b : Line} {A : Point} {α : Plane} :
  (line_intersection a b : Set Point) = {A} → ∃ α, subset_of a α ∧ subset_of b α :=
sorry

theorem correct_statement_D {l : Line} {α : Plane} {A B : Point} :
  belongs_to_line A l → belongs_to_line B l → belongs_to_plane A α → belongs_to_plane B α → subset_of l α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_correct_statement_A_correct_statement_C_correct_statement_D_l1020_102081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobby_bought_four_more_erasers_l1020_102004

/-- The price of a single eraser in dollars -/
def eraser_price : ℚ := sorry

/-- The number of erasers Alex bought -/
def alex_erasers : ℕ := sorry

/-- The number of erasers Bobby bought -/
def bobby_erasers : ℕ := sorry

theorem bobby_bought_four_more_erasers :
  eraser_price > 1/100 →
  eraser_price * alex_erasers = 216/100 →
  eraser_price * bobby_erasers = 288/100 →
  bobby_erasers = alex_erasers + 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobby_bought_four_more_erasers_l1020_102004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_property_and_sum_l1020_102006

def binomial_coeff (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

def expansion_coefficients (n : ℕ) : List ℝ :=
  (List.range (n + 1)).map (λ k => (binomial_coeff n k : ℝ) * 2^(n - k))

theorem expansion_property_and_sum (n : ℕ) :
  n = 6 →
  (let coeffs := expansion_coefficients n
   coeffs.get? 1 = some ((1/5 : ℝ) * coeffs.get! 2)) ∧
  (expansion_coefficients n).sum = 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_property_and_sum_l1020_102006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_intersection_l1020_102067

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define a point on the ellipse
def point_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  ellipse a b x y

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + 1)

-- Define the reflection of a point across the x-axis
def reflect_x (x y : ℝ) : ℝ × ℝ :=
  (x, -y)

-- Theorem statement
theorem ellipse_fixed_point_intersection
  (a b : ℝ) (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  a > b ∧ b > 0 ∧
  point_on_ellipse a b 1 (Real.sqrt 2 / 2) ∧
  eccentricity a b = Real.sqrt 2 / 2 ∧
  point_on_ellipse a b x₁ y₁ ∧
  point_on_ellipse a b x₂ y₂ ∧
  line_l k x₁ y₁ ∧
  line_l k x₂ y₂ ∧
  x₁ ≠ x₂ →
  ∃ (t : ℝ), t * x₂ + (1 - t) * x₁ = -2 ∧ 
             t * y₂ + (1 - t) * (-y₁) = 0 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_intersection_l1020_102067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_letter_initials_count_l1020_102054

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}
def vowels : Finset Char := {'A', 'E', 'I'}
def consonants : Finset Char := letters \ vowels

theorem three_letter_initials_count :
  (Finset.filter (fun s => s.card = 3 ∧ s ⊆ letters ∧ (s ∩ vowels).card = 1) (Finset.powerset letters)).card = 441 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_letter_initials_count_l1020_102054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1020_102097

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : 4 * Real.cos t.C * (Real.sin (t.C / 2))^2 + Real.cos (2 * t.C) = 0)
  (h2 : Real.tan t.A = 2 * Real.tan t.B)
  (h3 : 3 * t.a * t.b = 25 - t.c^2) :
  (Real.sin (t.A - t.B) = Real.sqrt 3 / 6) ∧ 
  (∃ (max_area : ℝ), max_area = 25 * Real.sqrt 3 / 16 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1020_102097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1020_102021

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 
  (Real.arctan (x/3))^2 - Real.pi * Real.arctan (3/x) + (Real.arctan (3/x))^2 - 
  (Real.pi^2/18) * (3*x^2 - x + 2)

-- State the theorem about the range of g
theorem range_of_g : Set.range g = Set.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1020_102021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1020_102005

-- Define the functions f and g
def f (x : ℝ) := x^3 - 3*x - 1
noncomputable def g (a x : ℝ) := Real.exp (x * Real.log 2) - a

-- Define the closed interval [0, 2]
def I : Set ℝ := Set.Icc 0 2

-- State the theorem
theorem a_range (a : ℝ) : 
  (∀ x₁ ∈ I, ∃ x₂ ∈ I, |f x₁ - g a x₂| ≤ 2) → a ∈ Set.Icc 2 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1020_102005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1020_102090

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8 = 0

-- Define the line
def line_equation (x y k : ℝ) : Prop := y = k * x

-- Define the tangency condition
def is_tangent (k : ℝ) : Prop := ∃ (x y : ℝ), circle_equation x y ∧ line_equation x y k

-- Define the fourth quadrant condition
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- The main theorem
theorem tangent_line_slope :
  ∀ k : ℝ, (is_tangent k ∧ ∃ (x y : ℝ), circle_equation x y ∧ line_equation x y k ∧ in_fourth_quadrant x y) →
  k = -Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1020_102090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_l1020_102076

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- The water tank -/
def tank : Cone := { radius := 16, height := 120 }

/-- The proportion of the tank filled with water -/
def waterProportion : ℝ := 0.3

/-- Theorem stating the height of water in the tank -/
theorem water_height : 
  ∃ (h : ℝ), h = 60 * (6/5)^(1/3) ∧ 
  coneVolume { radius := 16 * (h/120), height := h } = waterProportion * coneVolume tank := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_l1020_102076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_419_l1020_102045

-- Define the sequence aₙ
def a : ℕ → ℤ
  | 0 => 5  -- We define a₀ to be 5 to match a₁ in the original problem
  | 1 => 19
  | 2 => 41
  | n + 3 => 3 * (a (n + 2) - a (n + 1)) + a n

-- State the theorem
theorem a_10_equals_419 : a 9 = 419 := by
  -- The proof goes here
  sorry

#eval a 9  -- This will evaluate a₁₀ (which is a 9 in 0-based indexing)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_419_l1020_102045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1020_102071

noncomputable section

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (5, 1)
def C : ℝ × ℝ := (-1, -1)

-- Define the midpoint D of side BC
noncomputable def D : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the slope of AC
noncomputable def slope_AC : ℝ := (A.2 - C.2) / (A.1 - C.1)

-- Define the equations of lines AD and BH
def line_AD (x y : ℝ) : Prop := 3 * x + y - 6 = 0
def line_BH (x y : ℝ) : Prop := x + 2 * y - 7 = 0

theorem triangle_ABC_properties :
  (∀ x y, line_AD x y ↔ (y - A.2) / (x - A.1) = (D.2 - A.2) / (D.1 - A.1)) ∧
  (∀ x y, line_BH x y ↔ (y - B.2) / (x - B.1) = -1 / slope_AC) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1020_102071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1020_102034

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  3 * x^2 - 18 * x - 2 * y^2 - 4 * y = 54

-- Define the distance between foci
noncomputable def distance_between_foci : ℝ := Real.sqrt (1580 / 6)

-- Theorem statement
theorem hyperbola_foci_distance :
  ∀ x y : ℝ, hyperbola_equation x y → 
  ∃ f1 f2 : ℝ × ℝ, 
    (f1.1 - f2.1)^2 + (f1.2 - f2.2)^2 = distance_between_foci^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1020_102034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_porter_work_days_l1020_102028

/-- Represents the number of days Porter works per week (excluding overtime) -/
def regular_days : ℕ → ℕ := fun d => d

/-- Represents Porter's daily rate in dollars -/
def daily_rate : ℕ := 8

/-- Represents Porter's overtime rate as a percentage of daily rate -/
def overtime_rate_percent : ℕ := 150

/-- Represents the number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Represents Porter's monthly earnings with overtime every week in dollars -/
def monthly_earnings : ℕ := 208

/-- Theorem stating that Porter works 5 days per week (excluding overtime) -/
theorem porter_work_days (d : ℕ) : 
  regular_days d = 5 ↔ 
    weeks_per_month * (daily_rate * regular_days d + 
      daily_rate * overtime_rate_percent / 100) = monthly_earnings :=
by sorry

#check porter_work_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_porter_work_days_l1020_102028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_reciprocals_lower_bound_achievable_l1020_102030

open BigOperators

theorem min_sum_reciprocals (b : Fin 15 → ℝ) 
  (positive : ∀ i, b i > 0) 
  (sum_one : ∑ i, b i = 1) : 
  ∑ i, (1 / b i) ≥ 225 := by
  sorry

theorem lower_bound_achievable : 
  ∃ b : Fin 15 → ℝ, (∀ i, b i > 0) ∧ (∑ i, b i = 1) ∧ (∑ i, (1 / b i) = 225) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_reciprocals_lower_bound_achievable_l1020_102030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_25th_match_l1020_102002

/-- Represents a batsman's performance -/
structure Batsman where
  totalRunsBefore : ℝ  -- Total runs before the 25th match
  averageBefore : ℝ    -- Average before the 25th match
  runsIn25th : ℝ       -- Runs scored in the 25th match
  averageIncrease : ℝ  -- Increase in average after the 25th match

/-- Calculates the average after the 25th match -/
noncomputable def averageAfter25th (b : Batsman) : ℝ :=
  (b.totalRunsBefore + b.runsIn25th) / 25

/-- Theorem stating the average after the 25th match -/
theorem average_after_25th_match (b : Batsman) 
  (h1 : b.runsIn25th = 137)
  (h2 : b.averageIncrease = 2.5)
  (h3 : b.totalRunsBefore = 24 * b.averageBefore)
  (h4 : averageAfter25th b = b.averageBefore + b.averageIncrease) :
  averageAfter25th b = 77 := by
  sorry

#check average_after_25th_match

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_25th_match_l1020_102002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1020_102091

/-- Represents a six-digit integer -/
def SixDigitInt := Fin 1000000

/-- Checks if a six-digit integer contains exactly two 3s -/
def hasTwoThrees (n : SixDigitInt) : Prop := sorry

/-- Removes two 3s from a six-digit integer -/
def removeThrees (n : SixDigitInt) : Nat := sorry

/-- The set of all six-digit integers that result in 2022 when two 3s are removed -/
def validNumbers : Set SixDigitInt :=
  {n : SixDigitInt | hasTwoThrees n ∧ removeThrees n = 2022}

/-- Assumption that validNumbers is finite -/
instance : Fintype validNumbers := sorry

theorem count_valid_numbers : Fintype.card validNumbers = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1020_102091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1020_102086

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1)

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- Part 1: If f(2) = 1/2, then a = 1/2
  (f a 2 = 1/2 → a = 1/2) ∧
  -- Part 2: Range of f(x) for x ≥ 0
  (∀ x : ℝ, x ≥ 0 → 
    ((0 < a ∧ a < 1 → f a x ∈ Set.Ioc 0 (a^(-1 : ℝ))) ∧
     (a > 1 → f a x ∈ Set.Ici (a^(-1 : ℝ))))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1020_102086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l1020_102064

theorem angle_difference (α β : ℝ) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.cos α = 2 * Real.sqrt 5 / 5 →
  Real.cos β = Real.sqrt 10 / 10 →
  α - β = -π/4 := by
  sorry

#check angle_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l1020_102064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_fewest_relatively_prime_dates_june_relatively_prime_dates_count_l1020_102072

/-- Represents a month in a year --/
inductive Month
| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec

/-- Returns the number of days in a given month during a leap year --/
def daysInMonth (m : Month) : Nat :=
  match m with
  | .Feb => 29
  | .Apr | .Jun | .Sep | .Nov => 30
  | _ => 31

/-- Checks if two natural numbers are relatively prime --/
def isRelativelyPrime (a b : Nat) : Bool :=
  Nat.gcd a b = 1

/-- Converts a Month to its corresponding number (1-12) --/
def monthToNat (m : Month) : Nat :=
  match m with
  | .Jan => 1 | .Feb => 2 | .Mar => 3 | .Apr => 4
  | .May => 5 | .Jun => 6 | .Jul => 7 | .Aug => 8
  | .Sep => 9 | .Oct => 10 | .Nov => 11 | .Dec => 12

/-- Counts the number of relatively prime dates in a given month --/
def countRelativelyPrimeDates (m : Month) : Nat :=
  let days := daysInMonth m
  let monthNum := monthToNat m
  (List.range days).filter (fun d => isRelativelyPrime monthNum (d + 1)) |>.length

/-- Theorem stating that June has the fewest relatively prime dates in a leap year --/
theorem june_fewest_relatively_prime_dates :
  ∀ m : Month, countRelativelyPrimeDates .Jun ≤ countRelativelyPrimeDates m :=
by sorry

/-- Theorem stating that June has exactly 10 relatively prime dates --/
theorem june_relatively_prime_dates_count :
  countRelativelyPrimeDates .Jun = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_fewest_relatively_prime_dates_june_relatively_prime_dates_count_l1020_102072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_semicircles_l1020_102046

/-- The number of small semicircles -/
def N : ℕ := 7

/-- The radius of each small semicircle -/
noncomputable def r : ℝ := 1

/-- The combined area of all small semicircles -/
noncomputable def A : ℝ := N * (Real.pi * r^2 / 2)

/-- The area of the region inside the large semicircle but outside the small semicircles -/
noncomputable def B : ℝ := (Real.pi * (3 * ↑N * r)^2 / 2) - A

/-- The theorem stating that N = 7 given the conditions of the problem -/
theorem number_of_semicircles :
  (3 * ↑N * (2 * r) = 6 * ↑N * r) ∧  -- Diameter of large semicircle is three times total diameter of small semicircles
  (A / B = 1 / 6) →                  -- Ratio of areas is 1:6
  N = 7 :=
by
  intro h
  -- The proof goes here
  sorry

#eval N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_semicircles_l1020_102046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_l1020_102053

/-- The combined surface area of a solid hemisphere with base area 3 and thickness t -/
noncomputable def combined_surface_area (t : ℝ) : ℝ :=
  12 - 4 * t * Real.sqrt (3 / Real.pi) + 2 * Real.pi * t^2

/-- Theorem: The combined surface area of a solid hemisphere with base area 3 and thickness t -/
theorem hemisphere_surface_area (t : ℝ) :
  let r := Real.sqrt (3 / Real.pi)
  let exterior_curved_area := 2 * Real.pi * r^2
  let interior_curved_area := 2 * Real.pi * (r - t)^2
  let base_area := 3
  exterior_curved_area + interior_curved_area + base_area = combined_surface_area t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_l1020_102053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_required_hours_approx_27_l1020_102001

/-- Calculates the required weekly work hours to reach a financial goal, given initial conditions and constraints. -/
noncomputable def calculate_required_hours (
  initial_hours_per_week : ℝ)
  (total_weeks : ℕ)
  (sick_weeks : ℕ)
  (wage_increase_weeks : ℕ)
  (wage_increase_percentage : ℝ)
  (financial_goal : ℝ) : ℝ :=
  let original_wage := financial_goal / (initial_hours_per_week * total_weeks)
  let increased_wage := original_wage * (1 + wage_increase_percentage)
  let weeks_at_original_wage := total_weeks - sick_weeks - wage_increase_weeks
  let amount_earned_at_increased_wage := increased_wage * initial_hours_per_week * wage_increase_weeks
  let amount_needed_before_increase := financial_goal - amount_earned_at_increased_wage
  amount_needed_before_increase / (original_wage * weeks_at_original_wage)

/-- Theorem stating that the calculated required hours per week is approximately 27. -/
theorem required_hours_approx_27 :
  ∃ ε > 0, |calculate_required_hours 25 15 3 5 0.5 4500 - 27| < ε :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_required_hours_approx_27_l1020_102001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_iff_fourth_quadrant_iff_l1020_102019

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 - 5*m)

/-- z is purely imaginary if and only if m = 3 -/
theorem purely_imaginary_iff (m : ℝ) : z m = Complex.I * Complex.im (z m) ↔ m = 3 := by
  sorry

/-- z lies in the fourth quadrant if and only if 0 < m < 3 -/
theorem fourth_quadrant_iff (m : ℝ) : 
  (Complex.re (z m) > 0 ∧ Complex.im (z m) < 0) ↔ (0 < m ∧ m < 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_iff_fourth_quadrant_iff_l1020_102019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1020_102066

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- First line: 3x + 4y - 3 = 0 -/
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 3 = 0

/-- Second line: 6x + 8y + 7 = 0 -/
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 7 = 0

theorem distance_between_given_lines :
  distance_between_parallel_lines 3 4 (-3) (7/2) = 13/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1020_102066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_are_coplanar_l1020_102083

/-- Given vectors in ℝ³ are coplanar if their scalar triple product is zero -/
def are_coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  let det := a.1 * (b.2.1 * c.2.2 - b.2.2 * c.2.1) -
             a.2.1 * (b.1 * c.2.2 - b.2.2 * c.1) +
             a.2.2 * (b.1 * c.2.1 - b.2.1 * c.1)
  det = 0

/-- The vectors a, b, and c are coplanar -/
theorem vectors_are_coplanar :
  let a : ℝ × ℝ × ℝ := (3, (1, -1))
  let b : ℝ × ℝ × ℝ := (-2, (-1, 0))
  let c : ℝ × ℝ × ℝ := (5, (2, -1))
  are_coplanar a b c :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_are_coplanar_l1020_102083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l1020_102032

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.sqrt (3 - x)

noncomputable def g (x : ℝ) : ℝ := f (x + 1) / (x - 1)

def domain_g : Set ℝ := Set.union (Set.Icc (-2) 1) (Set.Ioo 1 2)

theorem g_domain : 
  {x : ℝ | ∃ y, g x = y} = domain_g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l1020_102032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_tangent_l1020_102029

/-- The function f(x) defined as a^(x+1) + 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x+1) + 1

/-- The theorem stating that for any a > 0 and a ≠ 1, if the graph of f(x) = a^(x+1) + 1
    passes through a fixed point P on the terminal side of angle θ, then tan θ = -2 -/
theorem fixed_point_tangent (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ P : ℝ × ℝ, ∀ x : ℝ, f a x = P.2 → x = P.1) →
  (∃ θ : ℝ, ∃ r : ℝ, r > 0 ∧ P = (r * Real.cos θ, r * Real.sin θ)) →
  Real.tan θ = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_tangent_l1020_102029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l1020_102050

/-- Represents the tank and pipe system -/
structure TankSystem where
  tank_volume : ℚ
  inlet_rate : ℚ
  outlet_a_rate : ℚ
  outlet_b_rate : ℚ
  outlet_c_rate : ℚ
  outlet_d_rate : ℚ

/-- Conversion factor from cubic feet to cubic inches -/
def cubic_feet_to_inches : ℚ := 12 * 12 * 12

/-- The time it takes to empty the tank -/
noncomputable def time_to_empty (system : TankSystem) : ℚ :=
  let tank_volume_inches := system.tank_volume * cubic_feet_to_inches
  let cycle_flow_rate := (system.inlet_rate - system.outlet_a_rate - system.outlet_b_rate) +
                         (system.inlet_rate - system.outlet_c_rate - system.outlet_d_rate)
  tank_volume_inches / (-cycle_flow_rate / 2)

/-- Theorem stating that the given system takes 3456 minutes to empty -/
theorem tank_emptying_time :
  let system : TankSystem := {
    tank_volume := 20
    inlet_rate := 5
    outlet_a_rate := 9
    outlet_b_rate := 8
    outlet_c_rate := 7
    outlet_d_rate := 6
  }
  time_to_empty system = 3456 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l1020_102050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonality_condition_l1020_102016

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![3, 3]
def b : Fin 2 → ℝ := ![1, -1]

-- Define the dot product of two 2D vectors
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

-- Define the orthogonality condition
def orthogonal (u v : Fin 2 → ℝ) : Prop :=
  dot_product u v = 0

-- Main theorem
theorem orthogonality_condition (l : ℝ) :
  orthogonal (fun i => a i + l * b i) (fun i => a i - l * b i) ↔ l = 3 ∨ l = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonality_condition_l1020_102016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1020_102031

noncomputable def f (a : ℝ) (θ : ℝ) (x : ℝ) : ℝ := (a + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

noncomputable def g (a : ℝ) (θ : ℝ) (x : ℝ) : ℝ := f a θ x + f a θ (x + Real.pi/3)

theorem function_properties
  (h_odd : ∀ x, f a θ (-x) = -f a θ x)
  (h_zero : f a θ (Real.pi/4) = 0)
  (h_θ_range : θ ∈ Set.Ioo 0 Real.pi) :
  a = -1 ∧
  θ = Real.pi/2 ∧
  (∀ x, g a θ x = -Real.sqrt 3 / 2 * Real.cos (4 * x)) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/4),
    g a θ x ≥ -Real.sqrt 3 / 2 ∧
    g a θ x ≤ Real.sqrt 3 / 2) ∧
  g a θ (Real.pi/8) = -Real.sqrt 3 / 2 ∧
  g a θ (Real.pi/4) = Real.sqrt 3 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1020_102031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1020_102025

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
noncomputable def GeometricSequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  fun n => a₁ * q ^ (n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def GeometricSum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_problem (a₁ q : ℝ) (n : ℕ) :
  let a := GeometricSequence a₁ q
  a 1 + a n = 66 →
  a 3 * a (n - 2) = 128 →
  GeometricSum a₁ q n = 126 →
  (n = 6 ∧ (q = 2 ∨ q = 1/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1020_102025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1020_102099

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 * (x + 1)) / ((x - 4)^2)

-- Define the existence of r
axiom exists_r : ∃ r : ℝ, 4 < r ∧ r < 5

-- State the theorem
theorem inequality_holds (r : ℝ) (hr : 4 < r ∧ r < 5) :
  ∀ x ≥ r, f x ≥ 20 ∧ ∀ y < r, f y < 20 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1020_102099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_correction_effect_l1020_102014

-- Define the data set
def DataSet := List ℝ

-- Define the average function
noncomputable def average (data : DataSet) : ℝ := (data.sum) / data.length

-- Define the variance function
noncomputable def variance (data : DataSet) : ℝ :=
  let avg := average data
  (data.map (λ x => (x - avg)^2)).sum / data.length

-- Define the initial data set
def initial_data : DataSet := []  -- We'll leave this empty for now

-- State the theorem
theorem data_correction_effect (n : ℕ) (h : n > 2) :
  let initial_avg := average initial_data
  let initial_var := variance initial_data
  let corrected_data := initial_data.set (n-2) 11 |>.set (n-1) 29
  initial_avg = 20 ∧ 
  initial_var = 28 ∧ 
  initial_data.length = n ∧
  initial_data.get? (n-2) = some 21 ∧
  initial_data.get? (n-1) = some 19 →
  average corrected_data = 20 ∧ 
  variance corrected_data > 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_correction_effect_l1020_102014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l1020_102068

/-- Given that the terminal side of angle α passes through the point (a, -2),
    and tan(π + α) = 1/3, prove that a = -6. -/
theorem angle_terminal_side (α : ℝ) (a : ℝ) 
  (h1 : ∃ (x y : ℝ), x = a ∧ y = -2 ∧ x * Real.sin α = y * Real.cos α)
  (h2 : Real.tan (π + α) = 1/3) : 
  a = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l1020_102068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_side_a_value_l1020_102078

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (cos x)^2 + cos (2*x + π/3)

-- Part 1
theorem sin_2alpha_value (α : ℝ) (h1 : f α = sqrt 3 / 3 + 1) (h2 : 0 < α) (h3 : α < π/6) :
  sin (2*α) = (2 * sqrt 6 - 1) / 6 := by sorry

-- Part 2
theorem side_a_value (A B C : ℝ) (a b c : ℝ) (S : ℝ)
  (h1 : 0 < A ∧ A < π/2)  -- A is acute
  (h2 : f A = -1/2)
  (h3 : c = 3)
  (h4 : S = 3 * sqrt 3)
  (h5 : S = 1/2 * b * c * sin A)  -- Area formula
  (h6 : a^2 = b^2 + c^2 - 2*b*c*cos A)  -- Cosine rule
  : a = sqrt 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_side_a_value_l1020_102078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_specific_l1020_102000

/-- The time taken for two trains to cross each other -/
noncomputable def train_crossing_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let total_length := length1 + length2
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  total_length / relative_speed

/-- Theorem stating the time taken for two specific trains to cross each other -/
theorem train_crossing_time_specific :
  let length1 : ℝ := 250
  let length2 : ℝ := 350
  let speed1 : ℝ := 80
  let speed2 : ℝ := 60
  abs (train_crossing_time length1 length2 speed1 speed2 - 15.432) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_specific_l1020_102000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_intersection_length_theorem_l1020_102075

noncomputable def cube_stack_height (edge_lengths : List ℝ) : ℝ :=
  edge_lengths.sum

noncomputable def line_segment_length (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

noncomputable def cube_intersection_length (x1 y1 z1 x2 y2 z2 cube_edge : ℝ) (cube_bottom : ℝ) : ℝ :=
  line_segment_length x1 y1 cube_bottom (x1 + cube_edge) (y1 + cube_edge) (cube_bottom + cube_edge)

theorem cube_intersection_length_theorem (edge_lengths : List ℝ) 
    (h : edge_lengths = [2, 3, 4, 5]) :
  let total_height := cube_stack_height edge_lengths
  let target_cube_bottom := cube_stack_height (edge_lengths.take 2)
  let target_cube_edge := edge_lengths[2]!
  cube_intersection_length 0 0 0 5 5 total_height target_cube_edge target_cube_bottom = 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_intersection_length_theorem_l1020_102075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_for_integer_sequence_l1020_102033

def sequence_a (k : ℕ+) : ℕ → ℚ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | n + 1 => (1 + k / n) * sequence_a k n + 1

theorem unique_k_for_integer_sequence :
  ∃! k : ℕ+, ∀ n : ℕ, n ≥ 1 → ∃ m : ℤ, sequence_a k n = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_for_integer_sequence_l1020_102033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l1020_102056

theorem complex_fraction_equality : 
  (1 + 2 * Complex.I) / ((1 - Complex.I)^2) = -1 + (1/2) * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l1020_102056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gp_sum_20_l1020_102023

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem gp_sum_20 (a r : ℝ) (h1 : geometric_sum a r 5 = 31) 
  (h2 : geometric_sum a r 15 = 2047) : geometric_sum a r 20 = 1048575 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gp_sum_20_l1020_102023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_class_size_l1020_102037

/-- Proves that the original number of students in a class is 15, given the initial average age,
    the number and average age of new students, and the resulting new average age. -/
theorem original_class_size (initial_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (final_avg : ℝ) :
  initial_avg = 40 →
  new_students = 15 →
  new_avg = 32 →
  final_avg = 36 →
  ∃ (original_size : ℕ), original_size = 15 ∧
    (initial_avg * original_size + new_avg * new_students) / (original_size + new_students : ℝ) = final_avg :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_class_size_l1020_102037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_average_balance_l1020_102055

noncomputable def monthly_balances : List ℚ := [120, 240, 180, 180, 300, 150]

noncomputable def average_balance : ℚ := (monthly_balances.sum) / monthly_balances.length

theorem emily_average_balance : average_balance = 195 := by
  -- Unfold definitions
  unfold average_balance monthly_balances
  -- Simplify the sum and length calculations
  simp
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_average_balance_l1020_102055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1020_102058

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  m : ℝ
  b : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point lies on a parabola -/
def onParabola (point : Point) (para : Parabola) : Prop :=
  point.x^2 = 2 * para.p * point.y

/-- Check if a line is tangent to a parabola -/
def isTangent (l : Line) (para : Parabola) : Prop :=
  ∃ (p : Point), onParabola p para ∧ p.y = l.m * p.x + l.b ∧
    ∀ (q : Point), q ≠ p → onParabola q para → q.y ≠ l.m * q.x + l.b

/-- Main theorem -/
theorem parabola_properties (para : Parabola) (A B : Point)
    (h1 : A.x = 1 ∧ A.y = 1)
    (h2 : onParabola A para)
    (h3 : B.x = 0 ∧ B.y = -1) :
  let O : Point := ⟨0, 0⟩
  let AB : Line := ⟨(A.y - B.y) / (A.x - B.x), A.y - (A.y - B.y) / (A.x - B.x) * A.x⟩
  ∃ (P Q : Point),
    (∀ (l : Line), l.m ≠ AB.m → onParabola P para ∧ onParabola Q para ∧
      P.y = l.m * P.x + l.b ∧ Q.y = l.m * Q.x + l.b) →
    isTangent AB para ∧
    distance O P * distance O Q ≥ distance O A * distance O A ∧
    distance B P * distance B Q > distance B A * distance B A :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1020_102058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_specific_line_slope_l1020_102018

/-- The slope of a line given by the equation ax + by = c, where b ≠ 0, is -a/b -/
theorem line_slope (a b c : ℚ) (hb : b ≠ 0) :
  let slope := -a / b
  slope = 7/2 ↔ a = 7 ∧ b = -2 ∧ c = 14 := by
  sorry

/-- The line is defined by the equation 7x - 2y = 14 -/
theorem specific_line_slope :
  let slope := -(7 : ℚ) / (-2)
  slope = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_specific_line_slope_l1020_102018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1020_102038

theorem problem_solution (a b k : ℝ) 
  (h1 : (2 : ℝ)^a = (3 : ℝ)^b)
  (h2 : (2 : ℝ)^a = k)
  (h3 : k ≠ 1)
  (h4 : 2*a + b = 2*a*b) : 
  k = 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1020_102038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1020_102074

theorem trigonometric_identity (A B C : ℝ) 
  (h1 : Real.sin A + Real.sin B + Real.sin C = 0)
  (h2 : Real.cos A + Real.cos B + Real.cos C = 0) :
  (Real.cos (3 * A) + Real.cos (3 * B) + Real.cos (3 * C) = 3 * Real.cos (A + B + C)) ∧
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 3 * Real.sin (A + B + C)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1020_102074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_reduction_percentage_l1020_102095

/-- Calculates the percentage reduction between two numbers -/
noncomputable def percentageReduction (original : ℝ) (new : ℝ) : ℝ :=
  ((original - new) / original) * 100

theorem employee_reduction_percentage : 
  let original : ℝ := 208.04597701149424
  let new : ℝ := 181
  abs (percentageReduction original new - 13) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_reduction_percentage_l1020_102095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_abs_x_squared_minus_two_complex_magnitude_when_quotient_imaginary_l1020_102069

-- Part 1
theorem definite_integral_abs_x_squared_minus_two :
  ∫ x in (-2 : ℝ)..1, |x^2 - 2| = 1 := by sorry

-- Part 2
theorem complex_magnitude_when_quotient_imaginary (a : ℝ) :
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (∃ k : ℝ, z₁ / z₂ = k*I) → Complex.abs z₁ = 10/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_abs_x_squared_minus_two_complex_magnitude_when_quotient_imaginary_l1020_102069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_geometric_mean_l1020_102008

/-- Represents an ellipse with its key parameters -/
structure Ellipse where
  a : ℝ  -- Half the length of the major axis
  c : ℝ  -- Half the distance between foci
  d : ℝ  -- Half the distance between directrices
  h_positive_a : 0 < a
  h_positive_c : 0 < c
  h_positive_d : 0 < d
  h_c_lt_a : c < a  -- Ensures it's an ellipse, not a hyperbola

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- The ratio of distances for any point on the ellipse -/
axiom ellipse_distance_ratio (e : Ellipse) (P : ℝ × ℝ) :
  ∃ (F D : ℝ × ℝ), F.1 = -e.c ∧ D.1 = -e.d ∧ 
    dist P F / dist P D = e.eccentricity

/-- Theorem: The major axis is the geometric mean of the distance between foci and directrices -/
theorem ellipse_major_axis_geometric_mean (e : Ellipse) :
  (2 * e.a)^2 = (2 * e.c) * (2 * e.d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_geometric_mean_l1020_102008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_b_value_l1020_102048

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_b_lt_a : b < a

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The theorem to be proved -/
theorem ellipse_b_value (e : Ellipse) (h_a : e.a = 3) (h_b_lt_3 : e.b < 3)
  (F₁ F₂ : ℝ × ℝ) (h_foci : F₁.1 = -Real.sqrt (e.a^2 - e.b^2) ∧ F₂.1 = Real.sqrt (e.a^2 - e.b^2))
  (h_max_sum : ∀ (A B : PointOnEllipse e),
    distance A.x A.y F₂.1 F₂.2 + distance B.x B.y F₂.1 F₂.2 ≤ 10) :
  e.b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_b_value_l1020_102048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_l1020_102022

/-- Definition of the sequence a_n -/
def a : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => a (n + 1) + a n

/-- Theorem stating the sum of the series and the value of b -/
theorem fibonacci_series_sum :
  (∃ S : ℚ, (∑' n, a n * (2/5)^n) = S ∧ S = 10/11) ∧
  (∃! b : ℝ, b > 0 ∧ (∑' n, a n * b^n) = 1 ∧ b = -1 + Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_l1020_102022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_door_to_door_sales_percentage_l1020_102020

theorem door_to_door_sales_percentage 
  (houses_per_day : ℕ) 
  (work_days : ℕ) 
  (cheap_knife_price : ℕ) 
  (expensive_knife_price : ℕ) 
  (weekly_sales : ℕ) 
  (h1 : houses_per_day = 50)
  (h2 : work_days = 5)
  (h3 : cheap_knife_price = 50)
  (h4 : expensive_knife_price = 150)
  (h5 : weekly_sales = 5000)
  : (weekly_sales : ℝ) / ((houses_per_day * work_days : ℕ) : ℝ) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_door_to_door_sales_percentage_l1020_102020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1020_102060

theorem equation_solution (m : Nat) (h_even : Even m) (h_pos : 0 < m) :
  ∀ n x y : Nat,
    0 < n ∧ 0 < x ∧ 0 < y →
    Nat.Coprime m n →
    (x^2 + y^2)^m = (x * y)^n →
    ∃ a : Nat, n = m + 1 ∧ x = 2^a ∧ y = 2^a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1020_102060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_30_l1020_102070

open Real BigOperators

theorem sin_squared_sum_30 : 
  (∑ k in Finset.range 30, (sin (π * (2 * k + 1) / 60))^2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_30_l1020_102070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steady_speed_is_25_l1020_102044

/-- Represents the acceleration of a body at a given time -/
noncomputable def acceleration (t : ℝ) : ℝ :=
  if t ≤ 2 then 50 - 20 * t else 10

/-- The acceleration due to gravity -/
def g : ℝ := 10

/-- Theorem: The steady speed of the body is 25 m/s -/
theorem steady_speed_is_25 : ∃ (v : ℝ), v = 25 ∧ 
  (∀ (t : ℝ), t ≥ 0 → acceleration t ≤ g) ∧
  (∃ (t : ℝ), t > 2 ∧ acceleration t = g) := by
  sorry

#check steady_speed_is_25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steady_speed_is_25_l1020_102044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_speed_is_6_l1020_102007

-- Define the original distance and time
noncomputable def original_distance : ℝ := 16
noncomputable def original_time : ℝ := 4

-- Define the speed increase percentage
noncomputable def speed_increase_percentage : ℝ := 50

-- Define the function to calculate speed
noncomputable def calculate_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Define the function to calculate new speed after increase
noncomputable def calculate_new_speed (original_speed : ℝ) (increase_percentage : ℝ) : ℝ :=
  original_speed * (1 + increase_percentage / 100)

-- Theorem statement
theorem new_speed_is_6 :
  calculate_new_speed (calculate_speed original_distance original_time) speed_increase_percentage = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_speed_is_6_l1020_102007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_l1020_102026

/-- The projection of a vector onto another vector -/
noncomputable def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let norm_squared := w.1 * w.1 + w.2 * w.2
  (dot_product / norm_squared * w.1, dot_product / norm_squared * w.2)

/-- The set of vectors satisfying the projection condition -/
def S : Set (ℝ × ℝ) :=
  {u | proj (3, -4) u = (9/5, -12/5)}

/-- The line y = (3/4)x - 15/4 -/
def L : Set (ℝ × ℝ) :=
  {u | u.2 = 3/4 * u.1 - 15/4}

/-- Theorem stating that the set S is equal to the line L -/
theorem projection_line : S = L := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_l1020_102026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_sqrt2n_perfect_square_l1020_102094

/-- The integer part of a real number -/
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

/-- A function that generates the sequence of n_i -/
def generateSequence : ℕ → ℕ
| 0 => 1
| n + 1 => generateSequence n + 1  -- Placeholder implementation

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2

theorem infinitely_many_sqrt2n_perfect_square :
  ∀ k : ℕ, ∃ n > k, isPerfectSquare (Int.toNat (integerPart (Real.sqrt 2 * n))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_sqrt2n_perfect_square_l1020_102094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kamals_biology_marks_l1020_102041

theorem kamals_biology_marks (english math physics chemistry : ℕ)
  (average : ℚ) (total_subjects : ℕ) (h1 : english = 76)
  (h2 : math = 60) (h3 : physics = 72) (h4 : chemistry = 65)
  (h5 : average = 71) (h6 : total_subjects = 5) :
  (average * total_subjects : ℚ) - (english + math + physics + chemistry : ℕ) = 82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kamals_biology_marks_l1020_102041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1020_102024

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x - Real.sin x * Real.cos x

-- State the theorem
theorem f_max_value : 
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (M = 1/2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1020_102024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l1020_102049

theorem quadratic_equation_solution (a b k x : ℝ) :
  x^2 + 2*b^2 = (a - k*x)^2 ↔
  x = (2*a*k + Real.sqrt (4*a^2*k^2 - 4*(k^2-1)*(a^2 - 2*b^2))) / (2*(k^2 - 1)) ∨
  x = (2*a*k - Real.sqrt (4*a^2*k^2 - 4*(k^2-1)*(a^2 - 2*b^2))) / (2*(k^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l1020_102049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_theorem_l1020_102073

-- Define the expression to be simplified
noncomputable def simplify_expression (x : ℝ) : ℝ :=
  Real.sqrt (-x^3) - x * Real.sqrt (-1/x)

-- State the theorem
theorem simplification_theorem (x : ℝ) (h : x < 0) :
  simplify_expression x = (1 - x) * Real.sqrt (-x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_theorem_l1020_102073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_discount_percentage_l1020_102042

/-- Given a dress with original price d, prove that the staff discount percentage is 40% when:
    1. There's an initial discount of 45%
    2. A staff member pays 0.33 times the original price
-/
theorem staff_discount_percentage (d : ℝ) (h : d > 0) : 
  let initial_discount_rate : ℝ := 0.45
  let staff_payment_rate : ℝ := 0.33
  let price_after_initial_discount : ℝ := d * (1 - initial_discount_rate)
  let staff_price : ℝ := d * staff_payment_rate
  let staff_discount_amount : ℝ := price_after_initial_discount - staff_price
  let staff_discount_percentage : ℝ := (staff_discount_amount / price_after_initial_discount) * 100
  staff_discount_percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_discount_percentage_l1020_102042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_new_plan_cost_l1020_102040

/-- The cost of Mark's new phone plan given the cost of his old plan and the percentage increase -/
noncomputable def new_plan_cost (old_plan_cost : ℝ) (percent_increase : ℝ) : ℝ :=
  old_plan_cost * (1 + percent_increase / 100)

/-- Theorem stating that Mark's new phone plan costs $195 per month -/
theorem marks_new_plan_cost :
  new_plan_cost 150 30 = 195 := by
  -- Unfold the definition of new_plan_cost
  unfold new_plan_cost
  -- Simplify the arithmetic
  simp [mul_add, mul_div_right_comm]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_new_plan_cost_l1020_102040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1020_102017

open Set Real

/-- The range of inclination angles for the line y = x cos θ - 1 --/
theorem inclination_angle_range :
  ∀ θ : ℝ, ∃ α : ℝ, α = arctan (cos θ) ∧ α ∈ Icc 0 (π/4) ∪ Ico (3*π/4) π :=
by
  intro θ
  use arctan (cos θ)
  constructor
  · rfl
  · sorry  -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1020_102017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_third_quadrant_l1020_102089

theorem tan_value_third_quadrant (α : ℝ) (h1 : Real.sin α = -5/13) (h2 : α ∈ Set.Icc π (3*π/2)) : Real.tan α = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_third_quadrant_l1020_102089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_from_incircle_circumcircle_relation_l1020_102036

open Real

theorem isosceles_triangle_from_incircle_circumcircle_relation 
  (r R α : ℝ) (h : r = 4 * R * cos α * (sin (α/2))^2) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a = b ∨ a = c) := by
  -- We'll prove that there exist side lengths a, b, c forming an isosceles triangle
  -- given the relation between incircle radius r, circumcircle radius R, and angle α
  
  -- Start of the proof
  sorry -- This is a placeholder for the actual proof steps

  -- The proof would involve several steps:
  -- 1. Express r and R in terms of triangle side lengths and angles
  -- 2. Substitute these expressions into the given equation
  -- 3. Simplify and manipulate the resulting equation
  -- 4. Show that this leads to the conclusion that two sides are equal
  
  -- Conclusion: The triangle is isosceles (two sides are equal)

#check isosceles_triangle_from_incircle_circumcircle_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_from_incircle_circumcircle_relation_l1020_102036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1020_102047

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the problem conditions
def problem_conditions (t : Triangle) : Prop :=
  Real.sqrt 3 * Real.sin t.C * Real.cos t.C - (Real.cos t.C)^2 = 1/2 ∧
  t.c = 3 ∧
  ∃ (k : Real), k ≠ 0 ∧ (1, Real.sin t.A) = (2 * k, k * Real.sin t.B)

-- State the theorem
theorem triangle_problem (t : Triangle) (h : problem_conditions t) :
  t.C = Real.pi / 3 ∧ 
  t.A = Real.pi / 6 ∧
  t.B = Real.pi / 2 ∧
  t.a = Real.sqrt 3 ∧
  t.b = 2 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1020_102047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_routes_to_square_7_l1020_102080

/-- The number of routes from square 1 to square n, moving only to higher numbered adjacent squares -/
def num_routes : ℕ → ℕ
  | 0 => 1  -- Base case for 0 (although not used in the problem)
  | 1 => 1
  | 2 => 1
  | n + 3 => num_routes (n + 1) + num_routes (n + 2)

/-- The problem statement -/
theorem routes_to_square_7 : num_routes 7 = 13 := by
  -- Compute the values for squares 3 to 7
  have h3 : num_routes 3 = 2 := rfl
  have h4 : num_routes 4 = 3 := rfl
  have h5 : num_routes 5 = 5 := rfl
  have h6 : num_routes 6 = 8 := rfl
  -- The final result
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_routes_to_square_7_l1020_102080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_symmetry_l1020_102085

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -(x - a) / (x - a - 1)

theorem inverse_symmetry (a : ℝ) :
  (∀ x y : ℝ, f a x = y ↔ f a y = x) →  -- f is invertible
  (∀ x y : ℝ, f a x = y → f a (5 - y) = -1 - x) →  -- symmetry condition
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_symmetry_l1020_102085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equal_if_floor_equal_l1020_102098

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_equal_if_floor_equal
  (f g : ℝ → ℝ)
  (hf : is_quadratic f)
  (hg : is_quadratic g)
  (h : ∀ x, floor (f x) = floor (g x)) :
  ∀ x, f x = g x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equal_if_floor_equal_l1020_102098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_l1020_102061

/-- Curve C in the Cartesian coordinate system -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 3 * Real.sin θ)

/-- Point M -/
def point_M : ℝ × ℝ := (0, 1)

/-- Line l passing through point M -/
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ := (t * Real.cos α, 1 + t * Real.sin α)

/-- Condition that a point is on curve C -/
def on_curve_C (p : ℝ × ℝ) : Prop :=
  (p.1^2 / 16) + (p.2^2 / 9) = 1

/-- Theorem stating that the line satisfying all conditions has equation x = 0 -/
theorem line_l_equation :
  ∃ α : ℝ, ∀ t : ℝ,
    (∃ θ₁ θ₂ : ℝ, 
      on_curve_C (line_l α t) ∧
      curve_C θ₁ = line_l α t ∧
      curve_C θ₂ = line_l α (-2*t) ∧
      (curve_C θ₁).2 > (curve_C θ₂).2) →
    (line_l α t).1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_l1020_102061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_sets_A_B_l1020_102082

def C : Set ℕ := {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}
def S : Set ℕ := {4,5,9,14,23,37}

theorem exist_sets_A_B : ∃ (A B : Set ℕ),
  (A ∩ B = ∅) ∧
  (A ∪ B = C) ∧
  (∀ x y, x ∈ A → y ∈ A → x ≠ y → (x + y) ∉ S) ∧
  (∀ x y, x ∈ B → y ∈ B → x ≠ y → (x + y) ∉ S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_sets_A_B_l1020_102082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1020_102092

noncomputable section

-- Define the center of the circle
def center : ℝ × ℝ := (2, -1)

-- Define the line equation
def line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the chord length
def chord_length : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem circle_equation :
  ∃ (r : ℝ), ∀ (x y : ℝ),
    ((x - center.1)^2 + (y - center.2)^2 = r^2) ↔
    ((∃ (x' y' : ℝ), line x' y' ∧ 
      (x' - x)^2 + (y' - y)^2 = (chord_length / 2)^2) →
     (x - center.1)^2 + (y - center.2)^2 ≤ r^2) ∧
    r = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1020_102092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l1020_102013

/-- The volume of a pyramid with a rectangular base and four equal edges to the apex -/
noncomputable def pyramidVolume (baseLength : ℝ) (baseWidth : ℝ) (edgeLength : ℝ) : ℝ :=
  let baseArea := baseLength * baseWidth
  let halfDiagonal := Real.sqrt (baseLength ^ 2 + baseWidth ^ 2) / 2
  let height := Real.sqrt (edgeLength ^ 2 - halfDiagonal ^ 2)
  (1 / 3) * baseArea * height

/-- Theorem: The volume of a pyramid with a 9 × 12 rectangular base and four edges of length 15 
    joining the apex to the corners of the base is equal to 468 -/
theorem pyramid_volume_specific : pyramidVolume 9 12 15 = 468 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l1020_102013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l1020_102039

open Set

theorem intersection_A_complement_B : 
  let A : Set ℝ := {x | x^2 < 1}
  let B : Set ℝ := {x | x^2 - 2*x > 0}
  A ∩ (Bᶜ) = Icc (0 : ℝ) 1 ∩ Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l1020_102039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_contest_ratios_l1020_102051

def hazel_species_a : ℕ := 48
def hazel_species_b : ℕ := 32
def father_species_a : ℕ := 46
def father_species_b : ℕ := 24

def simplify_ratio (a b : ℕ) : ℕ × ℕ :=
  let d := Nat.gcd a b
  (a / d, b / d)

theorem fishing_contest_ratios :
  simplify_ratio hazel_species_a hazel_species_b = (3, 2) ∧
  simplify_ratio father_species_a father_species_b = (23, 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_contest_ratios_l1020_102051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachelle_gpa_probability_l1020_102057

theorem rachelle_gpa_probability :
  let total_classes : ℕ := 5
  let a_points : ℕ := 4
  let b_points : ℕ := 3
  let c_points : ℕ := 2
  let d_points : ℕ := 1
  let gpa_threshold : ℚ := 7/2
  let guaranteed_a_classes : ℕ := 3
  let eng_a_prob : ℚ := 1/3
  let eng_b_prob : ℚ := 1/5
  let hist_a_prob : ℚ := 1/5
  let hist_b_prob : ℚ := 1/4

  let total_points_needed : ℕ := (gpa_threshold * total_classes).ceil.toNat
  let remaining_points_needed : ℕ := total_points_needed - (guaranteed_a_classes * a_points)

  let prob_two_a : ℚ := eng_a_prob * hist_a_prob
  let prob_eng_a_hist_b : ℚ := eng_a_prob * hist_b_prob
  let prob_eng_b_hist_a : ℚ := eng_b_prob * hist_a_prob
  let prob_two_b : ℚ := eng_b_prob * hist_b_prob

  let total_prob : ℚ := prob_two_a + prob_eng_a_hist_b + prob_eng_b_hist_a + prob_two_b

  total_prob = 6/25 :=
by
  sorry

#eval (7/2 : ℚ) * 5  -- To verify the calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachelle_gpa_probability_l1020_102057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_special_eccentricity_l1020_102009

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Check if a triangle is isosceles right with right angle at a given vertex -/
def isIsoscelesRight (t : Triangle) (rightAngleVertex : Point) : Prop :=
  sorry -- Definition of isosceles right triangle property

/-- Theorem: Eccentricity of a special hyperbola configuration -/
theorem hyperbola_special_eccentricity (h : Hyperbola) 
  (F₁ F₂ A B : Point) (l : Line) (t : Triangle) :
  (∀ x y, x^2 / h.a^2 - y^2 / h.b^2 = 1 → 
    (x = A.x ∧ y = A.y) ∨ (x = B.x ∧ y = B.y)) →  -- A and B on hyperbola
  (l.p1 = F₂ ∧ (l.p2 = A ∨ l.p2 = B)) →  -- Line passes through F₂ and A or B
  (t.A = A ∧ t.B = B ∧ t.C = F₁) →  -- Triangle ABF₁
  (isIsoscelesRight t A) →  -- ABF₁ is isosceles right triangle with right angle at A
  (eccentricity h)^2 = 5 - 2 * Real.sqrt 2 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_special_eccentricity_l1020_102009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_am_value_l1020_102035

-- Define the function f
noncomputable def f (m a : ℝ) (x : ℝ) : ℝ := x^m - a*x

-- State the theorem
theorem am_value (m a : ℝ) : 
  (∀ x, deriv (f m a) x = 2*x + 1) → a * m = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_am_value_l1020_102035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1020_102003

noncomputable def f (x : ℝ) := Real.sqrt (Real.sin x) + Real.sqrt ((1/2) - Real.cos x)

theorem domain_of_f (x : ℝ) : 
  (∃ y, f x = y) ↔ 
  (∃ k : ℤ, π/3 + 2*k*π ≤ x ∧ x ≤ π + 2*k*π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1020_102003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l1020_102084

noncomputable section

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℝ) : ℝ := minutes / 60

theorem sam_distance (marguerite_distance marguerite_time sam_time : ℝ) :
  marguerite_distance = 120 →
  marguerite_time = 3 →
  sam_time = hours_to_minutes 4 →
  distance (marguerite_distance / marguerite_time) (minutes_to_hours sam_time) = 160 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l1020_102084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l1020_102063

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the property |AB| + |AC| = 3|BC|
variable (h1 : ‖A - B‖ + ‖A - C‖ = 3 * ‖B - C‖)

-- Define point T on AC
variable (T : EuclideanSpace ℝ (Fin 2))
variable (h2 : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ T = A + t • (C - A))
variable (h3 : ‖A - C‖ = 4 * ‖A - T‖)

-- Define points K and L
variable (K L : EuclideanSpace ℝ (Fin 2))
variable (h4 : ∃ k : ℝ, 0 < k ∧ k < 1 ∧ K = A + k • (B - A))
variable (h5 : ∃ l : ℝ, 0 < l ∧ l < 1 ∧ L = A + l • (C - A))

-- Define KL parallel to BC
variable (h6 : ∃ m : ℝ, K - L = m • (B - C))

-- Define KL tangent to the inscribed circle
variable (h7 : ∃ r : ℝ, r > 0 ∧ IsInscribed r A B C ∧ IsTangentLine (K - L) K r)

-- Define S as intersection of BT and KL
variable (S : EuclideanSpace ℝ (Fin 2))
variable (h8 : ∃ s t : ℝ, S = B + s • (T - B) ∧ S = K + t • (L - K))

-- Theorem statement
theorem triangle_ratio :
  ‖S - L‖ / ‖K - L‖ = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l1020_102063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l1020_102059

theorem count_integers_satisfying_inequality : 
  ∃! (S : Finset ℤ), (∀ n ∈ S, (Real.sqrt (n + 2 : ℝ) ≤ Real.sqrt (3*n + 1 : ℝ) ∧ 
                              Real.sqrt (3*n + 1 : ℝ) < Real.sqrt (5*n - 8 : ℝ))) ∧ 
                  Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l1020_102059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_other_sides_approx_l1020_102010

/-- A triangle with specific angle measures and one known side length -/
structure SpecificTriangle where
  /-- The measure of angle A in radians -/
  angleA : ℝ
  /-- The measure of angle B in radians -/
  angleB : ℝ
  /-- The measure of angle C in radians -/
  angleC : ℝ
  /-- The length of the side opposite to angle A -/
  sideA : ℝ
  /-- Ensures that the angles sum to π radians (180 degrees) -/
  angle_sum : angleA + angleB + angleC = Real.pi
  /-- Ensures that angle A is 75 degrees (π/2.4 radians) -/
  angle_a_value : angleA = Real.pi / 2.4
  /-- Ensures that angle B is 60 degrees (π/3 radians) -/
  angle_b_value : angleB = Real.pi / 3
  /-- Ensures that the side opposite to angle A is 12 units long -/
  side_a_value : sideA = 12

/-- The sum of the lengths of the sides opposite to angles B and C -/
noncomputable def sumOfOtherSides (t : SpecificTriangle) : ℝ :=
  (t.sideA * Real.sin t.angleB / Real.sin t.angleA) +
  (t.sideA * Real.sin t.angleC / Real.sin t.angleA)

/-- Theorem stating that the sum of the other two sides is approximately 19.5 -/
theorem sum_of_other_sides_approx (t : SpecificTriangle) :
  ‖sumOfOtherSides t - 19.5‖ < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_other_sides_approx_l1020_102010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_room_l1020_102087

/-- The number of tiles required to cover a rectangular room with given dimensions. -/
def tiles_required (room_length room_width tile_length tile_width : ℚ) : ℕ :=
  (room_length * room_width / (tile_length * tile_width)).ceil.toNat

/-- Theorem stating that 1440 3-inch-by-9-inch tiles are required to cover a 15-foot-by-18-foot room. -/
theorem tiles_for_room : tiles_required 15 18 (1/4) (3/4) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_room_l1020_102087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_value_decrease_l1020_102088

/-- The annual percentage decrease in market value of a machine -/
noncomputable def annual_decrease (initial_value : ℝ) (value_after_two_years : ℝ) : ℝ :=
  (1 - Real.sqrt (value_after_two_years / initial_value)) * 100

/-- Theorem stating that the annual percentage decrease for the given machine is approximately 10.56% -/
theorem machine_value_decrease : 
  let initial_value := 8000
  let value_after_two_years := 6400
  abs (annual_decrease initial_value value_after_two_years - 10.56) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_value_decrease_l1020_102088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_m_values_l1020_102079

-- Define the polynomial
def p (x m : ℝ) : ℝ := x^2 - (m - 3) * x + 25

-- Define what it means for a quadratic polynomial to be a perfect square
def is_perfect_square (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (k * x + (b / (2 * k)))^2

theorem perfect_square_m_values (m : ℝ) :
  (is_perfect_square 1 (-(m - 3)) 25) → (m = -7 ∨ m = 13) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_m_values_l1020_102079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_box_one_ball_probability_l1020_102043

/-- The probability of exactly one ball landing in the last box
    when 100 balls are randomly distributed into 100 boxes. -/
noncomputable def probability_last_box_one_ball : ℝ :=
  (99 / 100) ^ 99

/-- Theorem stating that the probability of exactly one ball
    in the last box is equal to (99/100)^99 -/
theorem last_box_one_ball_probability :
  probability_last_box_one_ball = (99 / 100) ^ 99 := by
  rfl

/-- Approximate the probability as a floating-point number -/
def probability_approx : Float :=
  (99 / 100 : Float) ^ 99

#eval probability_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_box_one_ball_probability_l1020_102043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_container_l1020_102015

/-- Represents the dimensions and volume of a rectangular container. -/
structure Container where
  width : ℝ
  length : ℝ
  height : ℝ
  volume : ℝ

/-- The total length of the steel bar used to make the container frame. -/
def totalLength : ℝ := 14.8

/-- Creates a Container given the width, ensuring the length is 0.5m more than the width
    and the height is calculated based on the remaining steel bar length. -/
noncomputable def makeContainer (w : ℝ) : Container :=
  let l := w + 0.5
  let h := (totalLength - 2 * (w + l)) / 2
  { width := w
  , length := l
  , height := h
  , volume := w * l * h }

/-- Theorem stating that the maximum volume is achieved with specific dimensions. -/
theorem max_volume_container :
  ∃ (c : Container), c = makeContainer 1 ∧ 
    c.volume = 1.8 ∧
    ∀ (w : ℝ), w > 0 → w < (totalLength / 4 - 0.25) → 
      (makeContainer w).volume ≤ c.volume := by
  sorry

#check max_volume_container

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_container_l1020_102015
