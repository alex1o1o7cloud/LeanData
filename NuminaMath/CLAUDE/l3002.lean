import Mathlib

namespace NUMINAMATH_CALUDE_water_filling_solution_l3002_300207

/-- Represents the water filling problem -/
def WaterFillingProblem (canCapacity : ℝ) (initialCans : ℕ) (initialFillRatio : ℝ) (initialTime : ℝ) (targetCans : ℕ) : Prop :=
  let initialWaterFilled := canCapacity * initialFillRatio * initialCans
  let fillRate := initialWaterFilled / initialTime
  let targetWaterToFill := canCapacity * targetCans
  targetWaterToFill / fillRate = 5

/-- Theorem stating the solution to the water filling problem -/
theorem water_filling_solution :
  WaterFillingProblem 8 20 (3/4) 3 25 := by
  sorry

end NUMINAMATH_CALUDE_water_filling_solution_l3002_300207


namespace NUMINAMATH_CALUDE_equivalent_operation_l3002_300299

theorem equivalent_operation (x : ℚ) : 
  (x * (5 / 6)) / (2 / 3) = x * (5 / 4) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l3002_300299


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l3002_300200

/-- Given a triangle with side lengths in the ratio 2:3:4 inscribed in a circle of radius 5,
    the area of the triangle is 18.75. -/
theorem triangle_area_in_circle (a b c : ℝ) (r : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  r = 5 →  -- Circle radius is 5
  b = (3/2) * a →  -- Side length ratio 2:3
  c = 2 * a →  -- Side length ratio 2:4
  c = 2 * r →  -- Diameter of the circle
  (1/2) * a * b = 18.75 :=  -- Area of the triangle
by sorry


end NUMINAMATH_CALUDE_triangle_area_in_circle_l3002_300200


namespace NUMINAMATH_CALUDE_bobs_corn_field_efficiency_l3002_300291

/-- Given a corn field with a certain number of rows and stalks per row,
    and a total harvest in bushels, calculate the number of stalks needed per bushel. -/
def stalks_per_bushel (rows : ℕ) (stalks_per_row : ℕ) (total_bushels : ℕ) : ℕ :=
  (rows * stalks_per_row) / total_bushels

/-- Theorem stating that for Bob's corn field, 8 stalks are needed per bushel. -/
theorem bobs_corn_field_efficiency :
  stalks_per_bushel 5 80 50 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bobs_corn_field_efficiency_l3002_300291


namespace NUMINAMATH_CALUDE_min_value_sum_l3002_300249

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / x + 1 / y = 1) :
  x + y ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l3002_300249


namespace NUMINAMATH_CALUDE_cos_angle_minus_pi_half_l3002_300235

/-- 
Given an angle α in a plane rectangular coordinate system whose terminal side 
passes through the point (4, -3), prove that cos(α - π/2) = -3/5.
-/
theorem cos_angle_minus_pi_half (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) →
  Real.cos (α - π/2) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_minus_pi_half_l3002_300235


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l3002_300281

theorem cubic_equation_integer_solutions :
  ∀ x y : ℤ, x^3 + 2*x*y - 7 = 0 ↔ 
    (x = -7 ∧ y = -25) ∨ 
    (x = -1 ∧ y = -4) ∨ 
    (x = 1 ∧ y = 3) ∨ 
    (x = 7 ∧ y = -24) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l3002_300281


namespace NUMINAMATH_CALUDE_connie_marbles_l3002_300298

/-- The number of marbles Connie started with -/
def initial_marbles : ℕ := 776

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℚ := 183.0

/-- The number of marbles Connie has left -/
def remaining_marbles : ℕ := 593

theorem connie_marbles :
  (initial_marbles : ℚ) - marbles_given = remaining_marbles :=
by sorry

end NUMINAMATH_CALUDE_connie_marbles_l3002_300298


namespace NUMINAMATH_CALUDE_min_plates_matching_pair_l3002_300264

/-- Represents the colors of plates -/
inductive PlateColor
  | White
  | Green
  | Red
  | Pink
  | Purple

/-- The minimum number of plates needed to guarantee a matching pair -/
def min_plates_for_match : ℕ := 6

/-- Theorem stating that given at least one plate of each of 5 colors,
    the minimum number of plates needed to guarantee a matching pair is 6 -/
theorem min_plates_matching_pair
  (white_count : ℕ) (green_count : ℕ) (red_count : ℕ) (pink_count : ℕ) (purple_count : ℕ)
  (h_white : white_count ≥ 1)
  (h_green : green_count ≥ 1)
  (h_red : red_count ≥ 1)
  (h_pink : pink_count ≥ 1)
  (h_purple : purple_count ≥ 1) :
  min_plates_for_match = 6 := by
  sorry

#check min_plates_matching_pair

end NUMINAMATH_CALUDE_min_plates_matching_pair_l3002_300264


namespace NUMINAMATH_CALUDE_prob_max_with_replacement_prob_max_without_replacement_l3002_300293

variable (M n k : ℕ)

-- Probability of drawing maximum k with replacement
def prob_with_replacement (M n k : ℕ) : ℚ :=
  (k^n - (k-1)^n) / M^n

-- Probability of drawing maximum k without replacement
def prob_without_replacement (M n k : ℕ) : ℚ :=
  (Nat.choose (k-1) (n-1)) / (Nat.choose M n)

-- Theorem for drawing with replacement
theorem prob_max_with_replacement (h1 : M > 0) (h2 : k > 0) (h3 : k ≤ M) :
  prob_with_replacement M n k = (k^n - (k-1)^n) / M^n :=
by sorry

-- Theorem for drawing without replacement
theorem prob_max_without_replacement (h1 : M > 0) (h2 : n > 0) (h3 : n ≤ k) (h4 : k ≤ M) :
  prob_without_replacement M n k = (Nat.choose (k-1) (n-1)) / (Nat.choose M n) :=
by sorry

end NUMINAMATH_CALUDE_prob_max_with_replacement_prob_max_without_replacement_l3002_300293


namespace NUMINAMATH_CALUDE_total_groom_time_is_210_l3002_300245

/-- Time to groom a poodle in minutes -/
def poodle_groom_time : ℕ := 30

/-- Time to groom a terrier in minutes -/
def terrier_groom_time : ℕ := poodle_groom_time / 2

/-- Number of poodles to groom -/
def num_poodles : ℕ := 3

/-- Number of terriers to groom -/
def num_terriers : ℕ := 8

/-- Total grooming time for all dogs -/
def total_groom_time : ℕ := num_poodles * poodle_groom_time + num_terriers * terrier_groom_time

theorem total_groom_time_is_210 : total_groom_time = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_groom_time_is_210_l3002_300245


namespace NUMINAMATH_CALUDE_oplus_neg_two_three_oplus_inequality_l3002_300234

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 2 * a - 3 * b

-- Theorem 1: (-2) ⊕ 3 = -13
theorem oplus_neg_two_three : oplus (-2) 3 = -13 := by sorry

-- Theorem 2: For all real x, ((-3/2x+1) ⊕ (-1-2x)) > ((3x-2) ⊕ (x+1))
theorem oplus_inequality (x : ℝ) : oplus (-3/2*x+1) (-1-2*x) > oplus (3*x-2) (x+1) := by sorry

end NUMINAMATH_CALUDE_oplus_neg_two_three_oplus_inequality_l3002_300234


namespace NUMINAMATH_CALUDE_a_142_equals_1995_and_unique_l3002_300231

def p (n : ℕ) : ℕ := sorry

def q (n : ℕ) : ℕ := sorry

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => a n * p (a n) / q (a n)

theorem a_142_equals_1995_and_unique :
  a 142 = 1995 ∧ ∀ n : ℕ, n ≠ 142 → a n ≠ 1995 := by sorry

end NUMINAMATH_CALUDE_a_142_equals_1995_and_unique_l3002_300231


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_l3002_300270

/-- A function f(x) = -x³ + ax² - x - 1 is monotonic on ℝ iff a ∈ [-√3, √3] -/
theorem monotonic_cubic_function (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ 
  a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_l3002_300270


namespace NUMINAMATH_CALUDE_tangent_to_circumcircle_l3002_300261

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary relations and functions
variable (midpoint : Circle → Point)
variable (intersect : Circle → Circle → Set Point)
variable (line_intersect : Point → Point → Circle → Set Point)
variable (on_line : Point → Point → Point → Prop)
variable (circumcircle : Point → Point → Point → Circle)
variable (tangent_to : Point → Point → Circle → Prop)

-- State the theorem
theorem tangent_to_circumcircle
  (Γ₁ Γ₂ Γ₃ : Circle)
  (O₁ O₂ A C D S E F : Point) :
  (midpoint Γ₁ = O₁) →
  (midpoint Γ₂ = O₂) →
  (A ∈ intersect Γ₂ (circumcircle O₁ O₂ A)) →
  ({C, D} ⊆ intersect Γ₁ Γ₂) →
  (S ∈ line_intersect A D Γ₁) →
  (on_line C S F) →
  (on_line O₁ O₂ F) →
  (Γ₃ = circumcircle A D E) →
  (E ∈ intersect Γ₁ Γ₃) →
  (E ≠ D) →
  tangent_to O₁ E Γ₃ :=
sorry

end NUMINAMATH_CALUDE_tangent_to_circumcircle_l3002_300261


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l3002_300269

theorem alpha_beta_sum (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α = 1) 
  (hβ : β^3 - 3*β^2 + 5*β = 5) : 
  α + β = 2 := by sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l3002_300269


namespace NUMINAMATH_CALUDE_value_of_a_l3002_300229

-- Define the conversion rate between paise and rupees
def paise_per_rupee : ℚ := 100

-- Define the given percentage as a rational number
def given_percentage : ℚ := 1 / 200

-- Define the given amount in paise
def given_paise : ℚ := 95

-- Theorem statement
theorem value_of_a (a : ℚ) 
  (h : given_percentage * a = given_paise) : 
  a = 190 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l3002_300229


namespace NUMINAMATH_CALUDE_unique_solution_l3002_300210

def A : Nat := 89252525 -- ... (200 digits total)

def B (x y : Nat) : Nat := 444 * x * 100000 + 18 * 1000 + y * 10 + 27

def digit_at (n : Nat) (pos : Nat) : Nat :=
  (n / (10 ^ (pos - 1))) % 10

theorem unique_solution :
  ∃! (x y : Nat),
    x < 10 ∧ y < 10 ∧
    digit_at (A * B x y) 53 = 1 ∧
    digit_at (A * B x y) 54 = 0 ∧
    x = 4 ∧ y = 6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3002_300210


namespace NUMINAMATH_CALUDE_square_sum_equality_l3002_300287

theorem square_sum_equality (a b : ℕ) (h1 : a^2 = 225) (h2 : b^2 = 25) :
  a^2 + 2*a*b + b^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3002_300287


namespace NUMINAMATH_CALUDE_total_cakes_eaten_l3002_300213

def monday_cakes : ℕ := 6
def friday_cakes : ℕ := 9
def saturday_cakes : ℕ := 3 * monday_cakes

theorem total_cakes_eaten : monday_cakes + friday_cakes + saturday_cakes = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_cakes_eaten_l3002_300213


namespace NUMINAMATH_CALUDE_man_double_son_age_l3002_300208

/-- Calculates the number of years until a man's age is twice his son's age. -/
def yearsUntilDoubleAge (manAge sonAge : ℕ) : ℕ :=
  sorry

/-- Proves that the number of years until the man's age is twice his son's age is 2. -/
theorem man_double_son_age :
  let sonAge : ℕ := 14
  let manAge : ℕ := sonAge + 16
  yearsUntilDoubleAge manAge sonAge = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_double_son_age_l3002_300208


namespace NUMINAMATH_CALUDE_franks_savings_l3002_300289

/-- The amount of money Frank had saved initially -/
def initial_savings : ℕ := sorry

/-- The cost of one toy -/
def toy_cost : ℕ := 8

/-- The number of toys Frank could buy -/
def num_toys : ℕ := 5

/-- The additional allowance Frank received -/
def additional_allowance : ℕ := 37

/-- Theorem stating that Frank's initial savings is $3 -/
theorem franks_savings : 
  (initial_savings + additional_allowance = num_toys * toy_cost) → 
  initial_savings = 3 := by sorry

end NUMINAMATH_CALUDE_franks_savings_l3002_300289


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3002_300227

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x > 0 ∧ 
  (∀ y : ℕ, y > 0 → (1056 + y) % 29 = 0 ∧ (1056 + y) % 37 = 0 ∧ (1056 + y) % 43 = 0 → x ≤ y) ∧
  (1056 + x) % 29 = 0 ∧ (1056 + x) % 37 = 0 ∧ (1056 + x) % 43 = 0 ∧
  x = 44597 := by
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3002_300227


namespace NUMINAMATH_CALUDE_probability_after_removing_pairs_l3002_300273

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (numbers : ℕ)
  (cards_per_number : ℕ)

/-- Represents the state after removing pairs -/
structure RemovedPairs :=
  (pairs_removed : ℕ)

/-- Calculates the probability of selecting a pair from the remaining deck -/
def probability_of_pair (d : Deck) (r : RemovedPairs) : ℚ :=
  sorry

/-- The main theorem -/
theorem probability_after_removing_pairs :
  let d : Deck := ⟨80, 20, 4⟩
  let r : RemovedPairs := ⟨3⟩
  probability_of_pair d r = 105 / 2701 :=
sorry

end NUMINAMATH_CALUDE_probability_after_removing_pairs_l3002_300273


namespace NUMINAMATH_CALUDE_M_properties_l3002_300214

-- Define the set M
def M : Set (ℝ × ℝ) := {p | Real.sqrt 2 * p.1 - 1 < p.2 ∧ p.2 < Real.sqrt 2 * p.1}

-- Define what it means for a point to have integer coordinates
def hasIntegerCoordinates (p : ℝ × ℝ) : Prop := ∃ (i j : ℤ), p = (↑i, ↑j)

-- Statement of the theorem
theorem M_properties :
  Convex ℝ M ∧
  (∃ (S : Set (ℝ × ℝ)), S ⊆ M ∧ Set.Infinite S ∧ ∀ p ∈ S, hasIntegerCoordinates p) ∧
  ∀ (a b : ℝ), let L := {p : ℝ × ℝ | p.2 = a * p.1 + b}
    (∃ (S : Set (ℝ × ℝ)), S ⊆ (M ∩ L) ∧ Set.Finite S ∧
      ∀ p ∈ (M ∩ L), hasIntegerCoordinates p → p ∈ S) :=
by
  sorry

end NUMINAMATH_CALUDE_M_properties_l3002_300214


namespace NUMINAMATH_CALUDE_divisibility_rule_l3002_300283

theorem divisibility_rule (x y : ℕ+) (h : (1000 * y + x : ℕ) > 0) :
  (((x : ℤ) - (y : ℤ)) % 7 = 0 ∨ ((x : ℤ) - (y : ℤ)) % 11 = 0) →
  ((1000 * y + x : ℕ) % 7 = 0 ∨ (1000 * y + x : ℕ) % 11 = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_rule_l3002_300283


namespace NUMINAMATH_CALUDE_triangle_formation_l3002_300294

/-- Function to check if three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating which set of numbers can form a triangle -/
theorem triangle_formation :
  can_form_triangle 13 12 20 ∧
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 8 7 15 ∧
  ¬ can_form_triangle 5 5 11 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3002_300294


namespace NUMINAMATH_CALUDE_min_angle_BFE_l3002_300226

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the incenter of a triangle
def incenter (t : Triangle) : Point := sorry

-- Define the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Main theorem
theorem min_angle_BFE (ABC : Triangle) :
  let D := incenter ABC
  let ABD := Triangle.mk ABC.A ABC.B D
  let E := incenter ABD
  let BDE := Triangle.mk ABC.B D E
  let F := incenter BDE
  ∀ θ : ℕ, (θ : ℝ) = angle B F E → θ ≥ 113 :=
sorry

end NUMINAMATH_CALUDE_min_angle_BFE_l3002_300226


namespace NUMINAMATH_CALUDE_parabola_focus_l3002_300232

/-- A parabola is defined by the equation x^2 = 4y -/
def is_parabola (f : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, f (x, y) ↔ x^2 = 4*y

/-- The focus of a parabola is a point (h, k) -/
def is_focus (f : ℝ × ℝ → Prop) (h k : ℝ) : Prop :=
  is_parabola f ∧ (h, k) = (0, 1)

/-- Theorem: The focus of the parabola x^2 = 4y is (0, 1) -/
theorem parabola_focus :
  ∀ f : ℝ × ℝ → Prop, is_parabola f → is_focus f 0 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l3002_300232


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3002_300243

theorem algebraic_expression_value (a b : ℝ) (h : a - b - 2 = 0) :
  a^2 - b^2 - 4*a = -4 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3002_300243


namespace NUMINAMATH_CALUDE_shaded_area_is_925_l3002_300260

-- Define the vertices of the square
def square_vertices : List (ℝ × ℝ) := [(0, 0), (40, 0), (40, 40), (0, 40)]

-- Define the vertices of the shaded polygon
def shaded_vertices : List (ℝ × ℝ) := [(0, 0), (15, 0), (40, 30), (30, 40), (0, 20)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the area of the shaded region is 925 square units
theorem shaded_area_is_925 :
  polygon_area shaded_vertices = 925 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_is_925_l3002_300260


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3002_300225

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > -1/2 * p.1 + 6 ∧ p.2 > 3 * p.1 - 4}

-- Define the first quadrant
def Q1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

-- Define the second quadrant
def Q2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

-- Theorem statement
theorem points_in_quadrants_I_and_II : S ⊆ Q1 ∪ Q2 := by
  sorry


end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3002_300225


namespace NUMINAMATH_CALUDE_permutations_not_divisible_by_three_l3002_300284

/-- The number of permutations of 1 to n where 1 is fixed and each number differs from its neighbors by at most 2 -/
def p (n : ℕ) : ℕ :=
  if n ≤ 2 then 1
  else if n = 3 then 2
  else p (n - 1) + p (n - 3) + 1

/-- The theorem stating that the number of permutations for 1996 is not divisible by 3 -/
theorem permutations_not_divisible_by_three :
  ¬ (3 ∣ p 1996) :=
sorry

end NUMINAMATH_CALUDE_permutations_not_divisible_by_three_l3002_300284


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3002_300295

theorem trigonometric_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  (Real.tan (2*x) = -24/7) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3002_300295


namespace NUMINAMATH_CALUDE_square_sum_greater_than_product_l3002_300266

theorem square_sum_greater_than_product {a b : ℝ} (h : a > b) : a^2 + b^2 > a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_greater_than_product_l3002_300266


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3002_300286

/-- The quadratic equation x^2 - 2mx + m^2 + m - 3 = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 2*m*x + m^2 + m - 3 = 0

/-- The range of m for which the equation has real roots -/
def m_range : Set ℝ :=
  {m : ℝ | m ≤ 3}

/-- The product of the roots of the equation -/
def root_product (m : ℝ) : ℝ := m^2 + m - 3

theorem quadratic_equation_properties :
  (∀ m : ℝ, has_real_roots m ↔ m ∈ m_range) ∧
  (∃ m : ℝ, root_product m = 17 ∧ m = -5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3002_300286


namespace NUMINAMATH_CALUDE_n_range_l3002_300280

/-- The function f(x) with parameters m and n -/
def f (m n x : ℝ) : ℝ := m * x^2 - (5 * m + n) * x + n

/-- Theorem stating the range of n given the conditions -/
theorem n_range :
  ∀ n : ℝ,
  (∃ m : ℝ, -2 < m ∧ m < -1 ∧
    ∃ x : ℝ, 3 < x ∧ x < 5 ∧ f m n x = 0) →
  0 < n ∧ n ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_n_range_l3002_300280


namespace NUMINAMATH_CALUDE_f_f_zero_l3002_300272

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2 * Real.exp x else Real.log x

theorem f_f_zero (x : ℝ) : f (f x) = 0 ↔ x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_f_f_zero_l3002_300272


namespace NUMINAMATH_CALUDE_percentage_chain_ten_percent_of_thirty_percent_of_fifty_percent_of_7000_l3002_300237

theorem percentage_chain (n : ℝ) : n * 0.5 * 0.3 * 0.1 = n * 0.015 := by sorry

theorem ten_percent_of_thirty_percent_of_fifty_percent_of_7000 :
  7000 * 0.5 * 0.3 * 0.1 = 105 := by sorry

end NUMINAMATH_CALUDE_percentage_chain_ten_percent_of_thirty_percent_of_fifty_percent_of_7000_l3002_300237


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3002_300262

theorem right_triangle_third_side : ∀ (a b c : ℝ),
  (a = 3 ∧ b = 2) ∨ (a = 2 ∧ b = 3) →
  (c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2) →
  c = Real.sqrt 13 ∨ c = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3002_300262


namespace NUMINAMATH_CALUDE_symmetric_circles_and_common_chord_l3002_300238

-- Define the symmetry relation with respect to line l
def symmetric_line (x y : ℝ) : Prop := ∃ (x' y' : ℝ), x' = y + 1 ∧ y' = x - 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 2*y = 0

-- Define circle C'
def circle_C' (x y : ℝ) : Prop := (x-2)^2 + (y-2)^2 = 10

-- Theorem statement
theorem symmetric_circles_and_common_chord :
  (∀ x y : ℝ, symmetric_line x y → (circle_C x y ↔ circle_C' y x)) ∧
  (∃ a b c d : ℝ, 
    circle_C a b ∧ circle_C c d ∧ 
    circle_C' a b ∧ circle_C' c d ∧
    (a - c)^2 + (b - d)^2 = 38) :=
sorry

end NUMINAMATH_CALUDE_symmetric_circles_and_common_chord_l3002_300238


namespace NUMINAMATH_CALUDE_fraction_zero_implies_negative_one_l3002_300233

theorem fraction_zero_implies_negative_one (x : ℝ) :
  (x^2 - 1) / (x - 1) = 0 ∧ x - 1 ≠ 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_negative_one_l3002_300233


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3002_300251

-- Define the property that f must satisfy
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y u v : ℝ, (f x + f y) * (f u + f v) = f (x*u - y*v) + f (x*v + y*u)

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, satisfies_property f →
    (∀ x : ℝ, f x = x^2) ∨ (∀ x : ℝ, f x = (1/2 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3002_300251


namespace NUMINAMATH_CALUDE_arccos_one_equals_zero_l3002_300211

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_equals_zero_l3002_300211


namespace NUMINAMATH_CALUDE_x_sixth_minus_six_x_when_three_l3002_300271

theorem x_sixth_minus_six_x_when_three :
  let x : ℝ := 3
  x^6 - 6*x = 711 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_minus_six_x_when_three_l3002_300271


namespace NUMINAMATH_CALUDE_steves_pool_filling_time_l3002_300240

/-- The time required to fill Steve's pool -/
theorem steves_pool_filling_time :
  let pool_capacity : ℝ := 30000  -- gallons
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℝ := 3  -- gallons per minute
  let minutes_per_hour : ℕ := 60
  
  let total_flow_rate : ℝ := num_hoses * flow_rate_per_hose  -- gallons per minute
  let hourly_flow_rate : ℝ := total_flow_rate * minutes_per_hour  -- gallons per hour
  let filling_time : ℝ := pool_capacity / hourly_flow_rate  -- hours
  
  ⌈filling_time⌉ = 34 := by
  sorry

end NUMINAMATH_CALUDE_steves_pool_filling_time_l3002_300240


namespace NUMINAMATH_CALUDE_exists_lcm_sum_for_non_power_of_two_l3002_300265

/-- Represents the least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ := (a * b) / Nat.gcd a b

/-- Theorem: For any natural number n that is not a power of 2,
    there exist positive integers a, b, and c such that
    n = lcm a b + lcm b c + lcm c a -/
theorem exists_lcm_sum_for_non_power_of_two (n : ℕ) 
    (h : ∀ k : ℕ, n ≠ 2^k) :
    ∃ (a b c : ℕ+), n = lcm a b + lcm b c + lcm c a := by
  sorry

end NUMINAMATH_CALUDE_exists_lcm_sum_for_non_power_of_two_l3002_300265


namespace NUMINAMATH_CALUDE_unique_hyperdeficient_number_l3002_300220

/-- Sum of divisors function -/
def f (n : ℕ) : ℕ := sorry

/-- A number is hyperdeficient if f(f(n)) = n + 3 -/
def is_hyperdeficient (n : ℕ) : Prop := f (f n) = n + 3

theorem unique_hyperdeficient_number : 
  ∃! n : ℕ, n > 0 ∧ is_hyperdeficient n :=
sorry

end NUMINAMATH_CALUDE_unique_hyperdeficient_number_l3002_300220


namespace NUMINAMATH_CALUDE_combined_weight_of_new_men_weight_problem_l3002_300256

/-- The combined weight of two new men replacing one man in a group, given certain conditions -/
theorem combined_weight_of_new_men (initial_count : ℕ) (weight_increase : ℝ) 
  (replaced_weight : ℝ) (new_count : ℕ) : ℝ :=
  let total_weight_increase := weight_increase * new_count
  let combined_weight := total_weight_increase + replaced_weight
  combined_weight

/-- The theorem statement matching the original problem -/
theorem weight_problem : 
  combined_weight_of_new_men 10 2.5 68 11 = 95.5 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_of_new_men_weight_problem_l3002_300256


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l3002_300206

/-- Function to transform a digit according to the problem rules -/
def transformDigit (d : Nat) : Nat :=
  match d with
  | 2 => 5
  | 5 => 2
  | _ => d

/-- Function to transform a five-digit number according to the problem rules -/
def transformNumber (n : Nat) : Nat :=
  let d1 := n / 10000
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  10000 * (transformDigit d1) + 1000 * (transformDigit d2) + 100 * (transformDigit d3) + 10 * (transformDigit d4) + (transformDigit d5)

/-- The main theorem statement -/
theorem unique_five_digit_number :
  ∃! x : Nat, 
    10000 ≤ x ∧ x < 100000 ∧  -- x is a five-digit number
    x % 2 = 1 ∧               -- x is odd
    transformNumber x = 2 * (x + 1) ∧ -- y = 2(x+1)
    x = 29995 := by
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l3002_300206


namespace NUMINAMATH_CALUDE_transformed_equation_solutions_l3002_300219

theorem transformed_equation_solutions
  (h : ∀ x : ℝ, x^2 + 2*x - 3 = 0 ↔ x = 1 ∨ x = -3) :
  ∀ x : ℝ, (x + 3)^2 + 2*(x + 3) - 3 = 0 ↔ x = -2 ∨ x = -6 := by
sorry

end NUMINAMATH_CALUDE_transformed_equation_solutions_l3002_300219


namespace NUMINAMATH_CALUDE_bottle_cap_distance_difference_l3002_300242

/-- Calculates the total distance traveled by Jenny's bottle cap -/
def jennys_distance : ℝ := 18 + 6 + 7.2 + 3.6 + 3.96

/-- Calculates the total distance traveled by Mark's bottle cap -/
def marks_distance : ℝ := 15 + 30 + 34.5 + 25.875 + 24.58125 + 7.374375 + 9.21796875

/-- The difference in distance between Mark's and Jenny's bottle caps -/
def distance_difference : ℝ := marks_distance - jennys_distance

theorem bottle_cap_distance_difference :
  distance_difference = 107.78959375 := by sorry

end NUMINAMATH_CALUDE_bottle_cap_distance_difference_l3002_300242


namespace NUMINAMATH_CALUDE_fraction_value_l3002_300248

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 3 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 4 * d) : 
  a * c / (b * d) = 12 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l3002_300248


namespace NUMINAMATH_CALUDE_wedge_volume_l3002_300290

/-- The volume of a wedge formed by two planar cuts in a cylindrical log. -/
theorem wedge_volume (d : ℝ) (angle : ℝ) (h : ℝ) (m : ℕ) : 
  d = 16 →
  angle = 60 →
  h = d →
  (1 / 6) * π * (d / 2)^2 * h = m * π →
  m = 171 := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l3002_300290


namespace NUMINAMATH_CALUDE_problem_statement_l3002_300276

theorem problem_statement (a b : ℝ) 
  (ha : |a| = 3)
  (hb : |b| = 5)
  (hab_sum : a + b > 0)
  (hab_prod : a * b < 0) :
  a^3 + 2*b = -17 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3002_300276


namespace NUMINAMATH_CALUDE_min_max_sum_bound_l3002_300236

theorem min_max_sum_bound (a b c d e f g : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ g ≥ 0) 
  (sum_one : a + b + c + d + e + f + g = 1) : 
  ∃ (x : ℝ), x ≥ 1/3 ∧ 
    (∀ y, y = max (a+b+c) (max (b+c+d) (max (c+d+e) (max (d+e+f) (e+f+g)))) → y ≤ x) ∧
    (∃ (a' b' c' d' e' f' g' : ℝ),
      a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ d' ≥ 0 ∧ e' ≥ 0 ∧ f' ≥ 0 ∧ g' ≥ 0 ∧
      a' + b' + c' + d' + e' + f' + g' = 1 ∧
      max (a'+b'+c') (max (b'+c'+d') (max (c'+d'+e') (max (d'+e'+f') (e'+f'+g')))) = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_max_sum_bound_l3002_300236


namespace NUMINAMATH_CALUDE_perimeter_ABCDE_l3002_300201

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom AB_eq_BC : dist A B = dist B C
axiom AB_eq_4 : dist A B = 4
axiom AE_eq_8 : dist A E = 8
axiom ED_eq_9 : dist E D = 9
axiom right_angle_EAB : (A.1 - E.1) * (B.1 - A.1) + (A.2 - E.2) * (B.2 - A.2) = 0
axiom right_angle_ABC : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
axiom right_angle_AED : (E.1 - A.1) * (D.1 - E.1) + (E.2 - A.2) * (D.2 - E.2) = 0

-- Define the perimeter function
def perimeter (A B C D E : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C D + dist D E + dist E A

-- State the theorem
theorem perimeter_ABCDE :
  perimeter A B C D E = 25 + Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ABCDE_l3002_300201


namespace NUMINAMATH_CALUDE_constant_t_value_l3002_300247

theorem constant_t_value : ∃ t : ℝ, 
  (∀ x : ℝ, (3*x^2 - 4*x + 5) * (5*x^2 + t*x + 15) = 15*x^4 - 47*x^3 + 115*x^2 - 110*x + 75) ∧ 
  t = -10 := by
  sorry

end NUMINAMATH_CALUDE_constant_t_value_l3002_300247


namespace NUMINAMATH_CALUDE_combined_weight_is_63_l3002_300263

/-- The combined weight of candles made by Ethan -/
def combined_weight : ℕ :=
  let beeswax_per_candle : ℕ := 8
  let coconut_oil_per_candle : ℕ := 1
  let total_candles : ℕ := 10 - 3
  let weight_per_candle : ℕ := beeswax_per_candle + coconut_oil_per_candle
  total_candles * weight_per_candle

/-- Theorem stating that the combined weight of candles is 63 ounces -/
theorem combined_weight_is_63 : combined_weight = 63 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_is_63_l3002_300263


namespace NUMINAMATH_CALUDE_sequence_length_l3002_300244

/-- The sequence defined by a(n) = 2 + 5(n-1) for n ≥ 1 -/
def a : ℕ → ℕ := λ n => 2 + 5 * (n - 1)

/-- The last term of the sequence -/
def last_term : ℕ := 57

theorem sequence_length :
  ∃ n : ℕ, n > 0 ∧ a n = last_term ∧ n = 12 := by sorry

end NUMINAMATH_CALUDE_sequence_length_l3002_300244


namespace NUMINAMATH_CALUDE_basketball_club_boys_l3002_300259

theorem basketball_club_boys (total : ℕ) (attendance : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 30 →
  attendance = 18 →
  total = boys + girls →
  attendance = boys + (girls / 3) →
  boys = 12 := by
sorry

end NUMINAMATH_CALUDE_basketball_club_boys_l3002_300259


namespace NUMINAMATH_CALUDE_grocery_weight_difference_l3002_300292

theorem grocery_weight_difference (rice sugar green_beans remaining_stock : ℝ) : 
  rice = green_beans - 30 →
  green_beans = 60 →
  remaining_stock = (2/3 * rice) + (4/5 * sugar) + green_beans →
  remaining_stock = 120 →
  green_beans - sugar = 10 := by
sorry

end NUMINAMATH_CALUDE_grocery_weight_difference_l3002_300292


namespace NUMINAMATH_CALUDE_smallest_integer_m_l3002_300218

theorem smallest_integer_m (x y m : ℝ) : 
  (2 * x + y = 4) →
  (x + 2 * y = -3 * m + 2) →
  (x - y > -3/2) →
  (∀ k : ℤ, k < m → ¬(∃ x y : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * (k : ℝ) + 2 ∧ x - y > -3/2)) →
  m = -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_m_l3002_300218


namespace NUMINAMATH_CALUDE_kaleb_ferris_wheel_spend_l3002_300297

/-- Calculates the money spent on a ferris wheel ride given initial tickets, remaining tickets, and cost per ticket. -/
def money_spent_on_ride (initial_tickets remaining_tickets cost_per_ticket : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * cost_per_ticket

/-- Proves that Kaleb spent 27 dollars on the ferris wheel ride. -/
theorem kaleb_ferris_wheel_spend :
  let initial_tickets : ℕ := 6
  let remaining_tickets : ℕ := 3
  let cost_per_ticket : ℕ := 9
  money_spent_on_ride initial_tickets remaining_tickets cost_per_ticket = 27 := by
sorry

end NUMINAMATH_CALUDE_kaleb_ferris_wheel_spend_l3002_300297


namespace NUMINAMATH_CALUDE_initial_ball_count_l3002_300277

theorem initial_ball_count (initial_blue : ℕ) (removed_blue : ℕ) (final_probability : ℚ) : 
  initial_blue = 7 → 
  removed_blue = 3 → 
  final_probability = 1/3 → 
  ∃ (total : ℕ), total = 15 ∧ 
    (initial_blue - removed_blue : ℚ) / (total - removed_blue : ℚ) = final_probability :=
by sorry

end NUMINAMATH_CALUDE_initial_ball_count_l3002_300277


namespace NUMINAMATH_CALUDE_vector_subtraction_l3002_300205

/-- Given plane vectors a and b, prove that a - 2b equals (7, 3) -/
theorem vector_subtraction (a b : ℝ × ℝ) (ha : a = (3, 5)) (hb : b = (-2, 1)) :
  a - 2 • b = (7, 3) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3002_300205


namespace NUMINAMATH_CALUDE_leftover_tarts_l3002_300275

theorem leftover_tarts (cherry_tarts blueberry_tarts peach_tarts : ℝ) 
  (h1 : cherry_tarts = 0.08)
  (h2 : blueberry_tarts = 0.75)
  (h3 : peach_tarts = 0.08) :
  cherry_tarts + blueberry_tarts + peach_tarts = 0.91 := by
  sorry

end NUMINAMATH_CALUDE_leftover_tarts_l3002_300275


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l3002_300252

theorem absolute_value_equation_product (x : ℝ) : 
  (|20 / x + 4| = 3) → (∃ y : ℝ, (|20 / y + 4| = 3) ∧ (x * y = 400 / 7)) :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l3002_300252


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3002_300279

theorem inequality_system_solution (x : ℝ) :
  (x - 3 * (x - 2) ≥ 4 ∧ (1 + 2 * x) / 3 > x - 1) ↔ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3002_300279


namespace NUMINAMATH_CALUDE_solution_is_rhombus_l3002_300250

def is_solution (x y : ℝ) : Prop :=
  max (|x + y|) (|x - y|) = 1

def rhombus_vertices : Set (ℝ × ℝ) :=
  {(-1, 0), (1, 0), (0, -1), (0, 1)}

theorem solution_is_rhombus :
  {p : ℝ × ℝ | is_solution p.1 p.2} = rhombus_vertices := by sorry

end NUMINAMATH_CALUDE_solution_is_rhombus_l3002_300250


namespace NUMINAMATH_CALUDE_tenth_term_is_144_l3002_300221

def fibonacci_like_sequence : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => fibonacci_like_sequence n + fibonacci_like_sequence (n + 1)

theorem tenth_term_is_144 : fibonacci_like_sequence 9 = 144 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_144_l3002_300221


namespace NUMINAMATH_CALUDE_terrier_to_poodle_groom_ratio_l3002_300285

def poodle_groom_time : ℕ := 30
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8
def total_groom_time : ℕ := 210

theorem terrier_to_poodle_groom_ratio :
  ∃ (terrier_groom_time : ℕ),
    terrier_groom_time * num_terriers + poodle_groom_time * num_poodles = total_groom_time ∧
    2 * terrier_groom_time = poodle_groom_time :=
by sorry

end NUMINAMATH_CALUDE_terrier_to_poodle_groom_ratio_l3002_300285


namespace NUMINAMATH_CALUDE_min_length_shared_side_l3002_300204

/-- Given two triangles ABC and DBC sharing side BC, with known side lengths,
    prove that the length of BC must be greater than 14. -/
theorem min_length_shared_side (AB AC DC BD BC : ℝ) : 
  AB = 7 → AC = 15 → DC = 9 → BD = 23 → BC > 14 := by
  sorry

end NUMINAMATH_CALUDE_min_length_shared_side_l3002_300204


namespace NUMINAMATH_CALUDE_path_length_along_squares_l3002_300274

theorem path_length_along_squares (PQ : ℝ) (h : PQ = 73) : 
  3 * PQ = 219 := by
  sorry

end NUMINAMATH_CALUDE_path_length_along_squares_l3002_300274


namespace NUMINAMATH_CALUDE_simplify_fraction_multiplication_l3002_300202

theorem simplify_fraction_multiplication :
  (175 : ℚ) / 1225 * 25 = 25 / 7 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_multiplication_l3002_300202


namespace NUMINAMATH_CALUDE_y_derivative_y_derivative_at_zero_l3002_300278

-- Define y as a function of x
variable (y : ℝ → ℝ)

-- Define the condition e^y + xy = e
variable (h : ∀ x, Real.exp (y x) + x * (y x) = Real.exp 1)

-- Theorem for y'
theorem y_derivative (x : ℝ) : 
  deriv y x = -(y x) / (Real.exp (y x) + x) := by sorry

-- Theorem for y'(0)
theorem y_derivative_at_zero : 
  deriv y 0 = -(1 / Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_y_derivative_y_derivative_at_zero_l3002_300278


namespace NUMINAMATH_CALUDE_solve_for_b_l3002_300209

theorem solve_for_b (a b : ℝ) 
  (eq1 : a * (a - 4) = 21)
  (eq2 : b * (b - 4) = 21)
  (neq : a ≠ b)
  (sum : a + b = 4) :
  b = -3 := by
sorry

end NUMINAMATH_CALUDE_solve_for_b_l3002_300209


namespace NUMINAMATH_CALUDE_min_odd_integers_l3002_300246

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_ab : a + b = 32)
  (sum_abcd : a + b + c + d = 47)
  (sum_abcdef : a + b + c + d + e + f = 66) :
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    odds.card = 2 ∧ 
    (∀ x ∈ odds, Odd x) ∧
    (∀ y ∈ {a, b, c, d, e, f} \ odds, Even y) :=
sorry

end NUMINAMATH_CALUDE_min_odd_integers_l3002_300246


namespace NUMINAMATH_CALUDE_points_opposite_sides_iff_a_in_range_l3002_300215

/-- The coordinates of point A satisfy the given equation. -/
def point_A_equation (a x y : ℝ) : Prop :=
  5 * a^2 - 6 * a * x - 4 * a * y + 2 * x^2 + 2 * x * y + y^2 = 0

/-- The equation of the circle centered at point B. -/
def circle_B_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + a^2 * y^2 - 6 * a^2 * x - 2 * a^3 * y + 4 * a * y + a^4 + 4 = 0

/-- Points A and B lie on opposite sides of the line y = 1. -/
def opposite_sides (ya yb : ℝ) : Prop :=
  (ya - 1) * (yb - 1) < 0

/-- The main theorem statement. -/
theorem points_opposite_sides_iff_a_in_range (a : ℝ) :
  (∃ (xa ya xb yb : ℝ),
    point_A_equation a xa ya ∧
    circle_B_equation a xb yb ∧
    opposite_sides ya yb) ↔
  (a > -1 ∧ a < 0) ∨ (a > 1 ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_points_opposite_sides_iff_a_in_range_l3002_300215


namespace NUMINAMATH_CALUDE_fuse_length_safety_l3002_300223

theorem fuse_length_safety (safe_distance : ℝ) (fuse_speed : ℝ) (operator_speed : ℝ) 
  (h1 : safe_distance = 400)
  (h2 : fuse_speed = 1.2)
  (h3 : operator_speed = 5) :
  ∃ (min_length : ℝ), min_length > 96 ∧ 
  ∀ (fuse_length : ℝ), fuse_length > min_length → 
  (fuse_length / fuse_speed) > (safe_distance / operator_speed) := by
  sorry

end NUMINAMATH_CALUDE_fuse_length_safety_l3002_300223


namespace NUMINAMATH_CALUDE_running_program_weekly_increase_l3002_300224

theorem running_program_weekly_increase 
  (initial_distance : ℝ) 
  (final_distance : ℝ) 
  (program_duration : ℕ) 
  (increase_duration : ℕ) 
  (h1 : initial_distance = 3)
  (h2 : final_distance = 7)
  (h3 : program_duration = 5)
  (h4 : increase_duration = 4)
  : (final_distance - initial_distance) / increase_duration = 1 := by
  sorry

end NUMINAMATH_CALUDE_running_program_weekly_increase_l3002_300224


namespace NUMINAMATH_CALUDE_shielas_drawings_l3002_300254

/-- The number of neighbors Shiela has -/
def num_neighbors : ℕ := 6

/-- The number of drawings each neighbor would receive -/
def drawings_per_neighbor : ℕ := 9

/-- The total number of animal drawings Shiela drew -/
def total_drawings : ℕ := num_neighbors * drawings_per_neighbor

theorem shielas_drawings : total_drawings = 54 := by
  sorry

end NUMINAMATH_CALUDE_shielas_drawings_l3002_300254


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3002_300228

theorem absolute_value_inequality (a b : ℝ) (h : a ≠ 0) :
  ∃ (m : ℝ), m = 2 ∧ (∀ (x y : ℝ) (h' : x ≠ 0), |x + y| + |x - y| ≥ m * |x|) ∧
  (∀ (m' : ℝ), (∀ (x y : ℝ) (h' : x ≠ 0), |x + y| + |x - y| ≥ m' * |x|) → m' ≤ m) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3002_300228


namespace NUMINAMATH_CALUDE_triangle_ABC_is_right_angled_l3002_300257

/-- Triangle ABC is defined by points A(5, -2), B(1, 5), and C(-1, 2) in a 2D Euclidean space -/
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (1, 5)
def C : ℝ × ℝ := (-1, 2)

/-- Distance squared between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Triangle ABC is right-angled -/
theorem triangle_ABC_is_right_angled : 
  dist_squared A B = dist_squared B C + dist_squared C A :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_is_right_angled_l3002_300257


namespace NUMINAMATH_CALUDE_company_fund_proof_l3002_300255

theorem company_fund_proof (n : ℕ) (initial_fund : ℕ) : 
  (80 * n - 20 = initial_fund) →  -- Planned $80 bonus, $20 short
  (70 * n + 75 = initial_fund) →  -- Actual $70 bonus, $75 left
  initial_fund = 700 := by
sorry

end NUMINAMATH_CALUDE_company_fund_proof_l3002_300255


namespace NUMINAMATH_CALUDE_area_of_right_triangle_PQR_l3002_300258

/-- A right triangle PQR in the xy-plane with specific properties -/
structure RightTrianglePQR where
  -- P, Q, R are points in ℝ²
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  -- PQR is a right triangle with right angle at R
  is_right_triangle : (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0
  -- Length of hypotenuse PQ is 50
  hypotenuse_length : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2
  -- Median through P lies along y = x + 2
  median_P : ∃ t : ℝ, (P.1 + R.1) / 2 = t ∧ (P.2 + R.2) / 2 = t + 2
  -- Median through Q lies along y = 2x + 3
  median_Q : ∃ t : ℝ, (Q.1 + R.1) / 2 = t ∧ (Q.2 + R.2) / 2 = 2*t + 3

/-- The area of the right triangle PQR is 500/3 -/
theorem area_of_right_triangle_PQR (t : RightTrianglePQR) : 
  abs ((t.P.1 - t.R.1) * (t.Q.2 - t.R.2) - (t.Q.1 - t.R.1) * (t.P.2 - t.R.2)) / 2 = 500 / 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_right_triangle_PQR_l3002_300258


namespace NUMINAMATH_CALUDE_binary_to_decimal_l3002_300239

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4 : ℕ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_l3002_300239


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l3002_300267

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The probability that a normal random variable is less than a given value -/
noncomputable def prob_less (ξ : NormalRV) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is greater than a given value -/
noncomputable def prob_greater (ξ : NormalRV) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is between two given values -/
noncomputable def prob_between (ξ : NormalRV) (a b : ℝ) : ℝ := sorry

/-- Theorem: For a normally distributed random variable ξ with 
    P(ξ < -2) = P(ξ > 2) = 0.3, P(-2 < ξ < 0) = 0.2 -/
theorem normal_distribution_probability (ξ : NormalRV) 
    (h1 : prob_less ξ (-2) = 0.3)
    (h2 : prob_greater ξ 2 = 0.3) :
    prob_between ξ (-2) 0 = 0.2 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l3002_300267


namespace NUMINAMATH_CALUDE_equation_solutions_l3002_300253

/-- Definition of matrix expression -/
def matrix_expr (a b c d : ℝ) : ℝ := a * b - c * d

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  matrix_expr (3 * x) (2 * x + 1) 1 (2 * x) = 5

/-- Theorem stating the solutions of the equation -/
theorem equation_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = -1 ∧ x₂ = 5/6 ∧ equation x₁ ∧ equation x₂ ∧
  ∀ (x : ℝ), equation x → (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3002_300253


namespace NUMINAMATH_CALUDE_mass_is_not_vector_l3002_300217

-- Define the properties of a physical quantity
structure PhysicalQuantity where
  has_magnitude : Bool
  has_direction : Bool

-- Define what makes a quantity a vector
def is_vector (q : PhysicalQuantity) : Prop :=
  q.has_magnitude ∧ q.has_direction

-- Define the physical quantities
def mass : PhysicalQuantity :=
  { has_magnitude := true, has_direction := false }

def velocity : PhysicalQuantity :=
  { has_magnitude := true, has_direction := true }

def displacement : PhysicalQuantity :=
  { has_magnitude := true, has_direction := true }

def force : PhysicalQuantity :=
  { has_magnitude := true, has_direction := true }

-- Theorem to prove
theorem mass_is_not_vector : ¬(is_vector mass) := by
  sorry

end NUMINAMATH_CALUDE_mass_is_not_vector_l3002_300217


namespace NUMINAMATH_CALUDE_correct_cookies_in_partial_bag_edgars_cookies_l3002_300212

/-- Represents the number of cookies in a paper bag that is not full. -/
def cookiesInPartialBag (totalCookies bagCapacity : ℕ) : ℕ :=
  totalCookies % bagCapacity

/-- Proves that the number of cookies in a partial bag is correct. -/
theorem correct_cookies_in_partial_bag (totalCookies bagCapacity : ℕ) 
    (h1 : bagCapacity > 0) (h2 : totalCookies ≥ bagCapacity) :
  cookiesInPartialBag totalCookies bagCapacity = 
    totalCookies - bagCapacity * (totalCookies / bagCapacity) :=
by sorry

/-- The specific problem instance. -/
theorem edgars_cookies :
  cookiesInPartialBag 292 16 = 4 :=
by sorry

end NUMINAMATH_CALUDE_correct_cookies_in_partial_bag_edgars_cookies_l3002_300212


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3002_300222

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3002_300222


namespace NUMINAMATH_CALUDE_common_divisors_90_105_l3002_300203

theorem common_divisors_90_105 : Finset.card (Finset.filter (· ∣ 105) (Nat.divisors 90)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_90_105_l3002_300203


namespace NUMINAMATH_CALUDE_rectangle_cover_theorem_l3002_300268

/-- An increasing function from [0, 1] to [0, 1] -/
def IncreasingFunction := {f : ℝ → ℝ | Monotone f ∧ Set.range f ⊆ Set.Icc 0 1}

/-- A rectangle with sides parallel to the coordinate axes -/
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- A set of rectangles covers the graph of a function -/
def covers (rs : Set Rectangle) (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, ∃ r ∈ rs, x ∈ Set.Icc r.x (r.x + r.width) ∧ f x ∈ Set.Icc r.y (r.y + r.height)

/-- Main theorem -/
theorem rectangle_cover_theorem (f : IncreasingFunction) (n : ℕ) :
  ∃ (rs : Set Rectangle), (∀ r ∈ rs, r.area = 1 / (2 * n)) ∧ covers rs f := by sorry

end NUMINAMATH_CALUDE_rectangle_cover_theorem_l3002_300268


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l3002_300216

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits -/
def Binary := List Bool

/-- The binary number 1011₂ -/
def b1 : Binary := [true, false, true, true]

/-- The binary number 101₂ -/
def b2 : Binary := [true, false, true]

/-- The binary number 11001₂ -/
def b3 : Binary := [true, true, false, false, true]

/-- The binary number 1110₂ -/
def b4 : Binary := [true, true, true, false]

/-- The binary number 100101₂ -/
def b5 : Binary := [true, false, false, true, false, true]

/-- The expected sum 1111010₂ -/
def expectedSum : Binary := [true, true, true, true, false, true, false]

/-- Theorem stating that the sum of the given binary numbers equals the expected sum -/
theorem binary_sum_theorem :
  binaryToDecimal b1 + binaryToDecimal b2 + binaryToDecimal b3 + 
  binaryToDecimal b4 + binaryToDecimal b5 = binaryToDecimal expectedSum := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_theorem_l3002_300216


namespace NUMINAMATH_CALUDE_puppy_weight_l3002_300282

/-- Given the weights of animals satisfying certain conditions, prove the puppy's weight is √2 -/
theorem puppy_weight (p s l r : ℝ) 
  (h1 : p + s + l + r = 40)
  (h2 : p^2 + l^2 = 4*s)
  (h3 : p^2 + s^2 = l^2) :
  p = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l3002_300282


namespace NUMINAMATH_CALUDE_first_half_speed_l3002_300241

theorem first_half_speed
  (total_distance : ℝ)
  (total_time : ℝ)
  (second_half_speed : ℝ)
  (h1 : total_distance = 300)
  (h2 : total_time = 11)
  (h3 : second_half_speed = 25)
  : ∃ (first_half_speed : ℝ),
    first_half_speed = 30 ∧
    total_distance / 2 / first_half_speed +
    total_distance / 2 / second_half_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_first_half_speed_l3002_300241


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3002_300230

/-- A 45°-45°-90° triangle inscribed in the first quadrant -/
structure RightIsoscelesTriangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  first_quadrant : X.1 ≥ 0 ∧ X.2 ≥ 0 ∧ Y.1 ≥ 0 ∧ Y.2 ≥ 0 ∧ Z.1 ≥ 0 ∧ Z.2 ≥ 0
  right_angle : (Z.1 - X.1) * (Y.1 - X.1) + (Z.2 - X.2) * (Y.2 - X.2) = 0
  isosceles : (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2
  hypotenuse_length : (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 32  -- 4√2 squared

/-- A circle tangent to x-axis, y-axis, and hypotenuse of the triangle -/
structure TangentCircle (t : RightIsoscelesTriangle) where
  O : ℝ × ℝ
  r : ℝ
  tangent_x : O.2 = r
  tangent_y : O.1 = r
  tangent_hypotenuse : ((t.Z.1 - t.X.1) * (O.1 - t.X.1) + (t.Z.2 - t.X.2) * (O.2 - t.X.2))^2 = 
                       r^2 * ((t.Z.1 - t.X.1)^2 + (t.Z.2 - t.X.2)^2)

theorem tangent_circle_radius 
  (t : RightIsoscelesTriangle) 
  (c : TangentCircle t) 
  (h : (t.Y.1 - t.X.1)^2 + (t.Y.2 - t.X.2)^2 = 16) : 
  c.r = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3002_300230


namespace NUMINAMATH_CALUDE_four_squares_sum_l3002_300296

theorem four_squares_sum (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a^2 + b^2 + c^2 + d^2 = 90 →
  a + b + c + d = 16 →
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 := by
sorry

end NUMINAMATH_CALUDE_four_squares_sum_l3002_300296


namespace NUMINAMATH_CALUDE_prob_spade_or_king_l3002_300288

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of spades in a standard deck -/
def num_spades : ℕ := 13

/-- The number of kings in a standard deck -/
def num_kings : ℕ := 4

/-- The number of cards that are both spades and kings -/
def overlap : ℕ := 1

/-- The probability of drawing a spade or a king from a standard 52-card deck -/
theorem prob_spade_or_king : 
  (num_spades + num_kings - overlap : ℚ) / deck_size = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_spade_or_king_l3002_300288
