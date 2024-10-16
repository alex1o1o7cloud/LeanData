import Mathlib

namespace NUMINAMATH_CALUDE_pages_left_after_tuesday_l2282_228215

def pages_read_monday : ℕ := 15
def extra_pages_tuesday : ℕ := 16
def total_pages : ℕ := 64

def pages_left : ℕ := total_pages - (pages_read_monday + (pages_read_monday + extra_pages_tuesday))

theorem pages_left_after_tuesday : pages_left = 18 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_after_tuesday_l2282_228215


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2282_228289

theorem expand_and_simplify (x y : ℝ) :
  x * (x - 3 * y) + (2 * x - y)^2 = 5 * x^2 - 7 * x * y + y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2282_228289


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l2282_228251

/-- The equation of a line passing through a given point with a given angle -/
theorem line_equation_through_point_with_angle 
  (x₀ y₀ : ℝ) (θ : ℝ) 
  (h_x₀ : x₀ = Real.sqrt 3) 
  (h_y₀ : y₀ = -2 * Real.sqrt 3) 
  (h_θ : θ = 135 * π / 180) :
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧ 
                 ∀ (x y : ℝ), a * x + b * y + c = 0 ↔ 
                 y - y₀ = Real.tan θ * (x - x₀) :=
sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l2282_228251


namespace NUMINAMATH_CALUDE_pie_pricing_theorem_l2282_228231

/-- Represents the cost of a pie in pounds -/
structure PieCost where
  cost : ℕ

/-- Represents the costs of different types of pies -/
structure PiePrices where
  apple : PieCost
  blueberry : PieCost
  cherry : PieCost
  damson : PieCost

/-- Conditions for the pie prices -/
def validPiePrices (prices : PiePrices) : Prop :=
  prices.cherry.cost = 2 * prices.apple.cost ∧
  prices.blueberry.cost = 2 * prices.damson.cost ∧
  prices.cherry.cost + 2 * prices.damson.cost = prices.apple.cost + 2 * prices.blueberry.cost

/-- The total cost of buying one of each type of pie -/
def totalCost (prices : PiePrices) : ℕ :=
  prices.apple.cost + prices.blueberry.cost + prices.cherry.cost + prices.damson.cost

/-- Theorem stating that the total cost is 18 pounds -/
theorem pie_pricing_theorem (prices : PiePrices) (h : validPiePrices prices) : totalCost prices = 18 := by
  sorry

end NUMINAMATH_CALUDE_pie_pricing_theorem_l2282_228231


namespace NUMINAMATH_CALUDE_order_of_even_monotone_increasing_l2282_228279

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def monotone_increasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

-- State the theorem
theorem order_of_even_monotone_increasing (heven : is_even f)
  (hmono : monotone_increasing_on f (Set.Ici 0)) :
  f (-Real.pi) > f 3 ∧ f 3 > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_order_of_even_monotone_increasing_l2282_228279


namespace NUMINAMATH_CALUDE_mean_of_three_numbers_l2282_228217

theorem mean_of_three_numbers (p q r : ℝ) : 
  (p + q) / 2 = 13 →
  (q + r) / 2 = 16 →
  (r + p) / 2 = 7 →
  (p + q + r) / 3 = 12 := by
sorry

end NUMINAMATH_CALUDE_mean_of_three_numbers_l2282_228217


namespace NUMINAMATH_CALUDE_triangle_side_length_l2282_228235

theorem triangle_side_length
  (A B C : ℝ)  -- Angles of the triangle
  (AB BC AC : ℝ)  -- Sides of the triangle
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)  -- Angle sum theorem
  (h5 : Real.cos (A + 2*C - B) + Real.sin (B + C - A) = 2)
  (h6 : AB = 2)
  : BC = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2282_228235


namespace NUMINAMATH_CALUDE_valid_triplets_are_solution_set_l2282_228247

def is_valid_triplet (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧
  (b + c + 1) % a = 0 ∧
  (c + a + 1) % b = 0 ∧
  (a + b + 1) % c = 0

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(1, 1, 1), (1, 2, 2), (1, 1, 3), (2, 2, 5), (3, 3, 7), (1, 4, 6),
   (2, 6, 9), (3, 8, 12), (4, 10, 15), (5, 12, 18), (6, 14, 21)}

theorem valid_triplets_are_solution_set :
  ∀ a b c : ℕ, is_valid_triplet a b c ↔ (a, b, c) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_valid_triplets_are_solution_set_l2282_228247


namespace NUMINAMATH_CALUDE_specific_normal_distribution_two_std_devs_less_l2282_228219

/-- Represents a normal distribution --/
structure NormalDistribution where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation

/-- The value that is exactly 2 standard deviations less than the mean --/
def twoStdDevsLessThanMean (nd : NormalDistribution) : ℝ :=
  nd.μ - 2 * nd.σ

/-- Theorem statement for the given problem --/
theorem specific_normal_distribution_two_std_devs_less (nd : NormalDistribution) 
  (h1 : nd.μ = 16.5) (h2 : nd.σ = 1.5) : 
  twoStdDevsLessThanMean nd = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_specific_normal_distribution_two_std_devs_less_l2282_228219


namespace NUMINAMATH_CALUDE_zora_shorter_than_brixton_l2282_228261

/-- Proves that Zora is 8 inches shorter than Brixton given the conditions of the problem -/
theorem zora_shorter_than_brixton :
  ∀ (zora itzayana zara brixton : ℕ),
    itzayana = zora + 4 →
    zara = 64 →
    brixton = zara →
    (zora + itzayana + zara + brixton) / 4 = 61 →
    brixton - zora = 8 := by
  sorry

end NUMINAMATH_CALUDE_zora_shorter_than_brixton_l2282_228261


namespace NUMINAMATH_CALUDE_percentile_rank_between_90_and_91_l2282_228229

/-- Represents a student's rank in a class -/
structure StudentRank where
  total_students : ℕ
  rank : ℕ
  h_rank_valid : rank ≤ total_students

/-- Calculates the percentile rank of a student -/
def percentile_rank (sr : StudentRank) : ℚ :=
  (sr.total_students - sr.rank : ℚ) / sr.total_students * 100

/-- Theorem stating that a student ranking 5th in a class of 48 has a percentile rank between 90 and 91 -/
theorem percentile_rank_between_90_and_91 (sr : StudentRank) 
  (h_total : sr.total_students = 48) 
  (h_rank : sr.rank = 5) : 
  90 < percentile_rank sr ∧ percentile_rank sr < 91 := by
  sorry

#eval percentile_rank ⟨48, 5, by norm_num⟩

end NUMINAMATH_CALUDE_percentile_rank_between_90_and_91_l2282_228229


namespace NUMINAMATH_CALUDE_valid_sequences_of_length_21_l2282_228293

/-- Counts valid binary sequences of given length -/
def count_valid_sequences (n : ℕ) : ℕ :=
  if n < 3 then 0
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 1
  else if n = 6 then 2
  else count_valid_sequences (n - 4) + 2 * count_valid_sequences (n - 5) + 2 * count_valid_sequences (n - 6)

/-- The main theorem stating the number of valid sequences of length 21 -/
theorem valid_sequences_of_length_21 :
  count_valid_sequences 21 = 135 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_of_length_21_l2282_228293


namespace NUMINAMATH_CALUDE_gcf_60_72_l2282_228216

theorem gcf_60_72 : Nat.gcd 60 72 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_60_72_l2282_228216


namespace NUMINAMATH_CALUDE_f_properties_l2282_228271

open Real

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - 1) * (x - a)

-- Define the derivative f'(x)
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * (1 + a) * x + a

theorem f_properties (a : ℝ) (h : a > 1) :
  -- 1. The derivative of f(x) is f'(x)
  (∀ x, deriv (f a) x = f_prime a x) ∧
  -- 2. f(x) has two different critical points
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f_prime a x₁ = 0 ∧ f_prime a x₂ = 0) ∧
  -- 3. f(x₁) + f(x₂) ≤ 0 holds if and only if a ≥ 2
  (∀ x₁ x₂, f_prime a x₁ = 0 → f_prime a x₂ = 0 → 
    (f a x₁ + f a x₂ ≤ 0 ↔ a ≥ 2)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2282_228271


namespace NUMINAMATH_CALUDE_whole_number_between_bounds_l2282_228200

theorem whole_number_between_bounds (M : ℤ) :
  (9.5 < (M : ℚ) / 5 ∧ (M : ℚ) / 5 < 10.5) ↔ (M = 49 ∨ M = 50 ∨ M = 51) := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_bounds_l2282_228200


namespace NUMINAMATH_CALUDE_min_crossing_time_l2282_228277

/-- Represents a person with their crossing time -/
structure Person where
  crossingTime : ℕ

/-- Represents the state of the bridge crossing problem -/
structure BridgeState where
  peopleOnIsland : List Person
  peopleOnMainland : List Person
  lampOnIsland : Bool
  totalTime : ℕ

/-- Defines the initial state of the problem -/
def initialState : BridgeState where
  peopleOnIsland := [
    { crossingTime := 2 },
    { crossingTime := 4 },
    { crossingTime := 8 },
    { crossingTime := 16 }
  ]
  peopleOnMainland := []
  lampOnIsland := true
  totalTime := 0

/-- Represents a valid move across the bridge -/
inductive Move
  | cross (p1 : Person) (p2 : Option Person)
  | returnLamp (p : Person)

/-- Applies a move to the current state -/
def applyMove (state : BridgeState) (move : Move) : BridgeState :=
  sorry

/-- Checks if all people have crossed to the mainland -/
def isComplete (state : BridgeState) : Bool :=
  sorry

/-- Theorem: The minimum time required to cross the bridge is 30 minutes -/
theorem min_crossing_time (initialState : BridgeState) :
  ∃ (moves : List Move), 
    (moves.foldl applyMove initialState).totalTime = 30 ∧ 
    isComplete (moves.foldl applyMove initialState) ∧
    ∀ (otherMoves : List Move), 
      isComplete (otherMoves.foldl applyMove initialState) → 
      (otherMoves.foldl applyMove initialState).totalTime ≥ 30 :=
  sorry

end NUMINAMATH_CALUDE_min_crossing_time_l2282_228277


namespace NUMINAMATH_CALUDE_total_cost_is_two_l2282_228241

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 1/10

/-- The cost of 3 pencils and 4 pens in dollars -/
def cost_3p4p : ℚ := 79/50

/-- The cost of a pen in dollars -/
def pen_cost : ℚ := (cost_3p4p - 3 * pencil_cost) / 4

/-- The total cost of 4 pencils and 5 pens in dollars -/
def total_cost : ℚ := 4 * pencil_cost + 5 * pen_cost

theorem total_cost_is_two : total_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_two_l2282_228241


namespace NUMINAMATH_CALUDE_density_function_properties_l2282_228260

/-- A density function that satisfies specific integral properties --/
noncomputable def f (g f_ζ : ℝ → ℝ) (x : ℝ) : ℝ := (g (-x) + f_ζ x) / 2

/-- The theorem stating the properties of the density function --/
theorem density_function_properties
  (g f_ζ : ℝ → ℝ)
  (hg : ∀ x, g (-x) = -g x)  -- g is odd
  (hf_ζ : ∀ x, f_ζ (-x) = f_ζ x)  -- f_ζ is even
  (hf_density : ∀ x, f g f_ζ x ≥ 0 ∧ ∫ x, f g f_ζ x = 1)  -- f is a density function
  : (∃ x, f g f_ζ x ≠ f g f_ζ (-x))  -- f is not even
  ∧ (∀ n : ℕ, n ≥ 1 → ∫ x in Set.Ici 0, |x|^n * f g f_ζ x = ∫ x in Set.Iic 0, |x|^n * f g f_ζ x) :=
sorry

end NUMINAMATH_CALUDE_density_function_properties_l2282_228260


namespace NUMINAMATH_CALUDE_radii_geometric_progression_implies_right_angle_l2282_228203

/-- A triangle with sides a, b, c, and semi-perimeter s. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  s : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_semi_perimeter : s = (a + b + c) / 2

/-- The radii of the four circles tangent to the lines of a triangle. -/
structure TriangleRadii (T : Triangle) where
  r : ℝ  -- inradius
  ra : ℝ  -- exradius opposite to side a
  rb : ℝ  -- exradius opposite to side b
  rc : ℝ  -- exradius opposite to side c
  h_positive : 0 < r ∧ 0 < ra ∧ 0 < rb ∧ 0 < rc
  h_area_relations : ∃ t : ℝ, t = r * T.s ∧ t = ra * (T.s - T.a) ∧ t = rb * (T.s - T.b) ∧ t = rc * (T.s - T.c)

/-- The radii form a geometric progression. -/
def IsGeometricProgression (R : TriangleRadii T) : Prop :=
  ∃ q : ℝ, 1 < q ∧ R.ra = q * R.r ∧ R.rb = q^2 * R.r ∧ R.rc = q^3 * R.r

/-- The largest angle of a triangle is 90 degrees. -/
def HasRightAngle (T : Triangle) : Prop :=
  T.a^2 + T.b^2 = T.c^2 ∨ T.b^2 + T.c^2 = T.a^2 ∨ T.c^2 + T.a^2 = T.b^2

/-- Main theorem: If the radii form a geometric progression, then the triangle has a right angle. -/
theorem radii_geometric_progression_implies_right_angle (T : Triangle) (R : TriangleRadii T) 
  (h_gp : IsGeometricProgression R) : HasRightAngle T :=
sorry

end NUMINAMATH_CALUDE_radii_geometric_progression_implies_right_angle_l2282_228203


namespace NUMINAMATH_CALUDE_fifteenth_term_of_arithmetic_sequence_l2282_228236

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifteenth_term_of_arithmetic_sequence 
  (a : ℕ → ℕ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 15)
  (h_third : a 3 = 27) :
  a 15 = 171 :=
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_arithmetic_sequence_l2282_228236


namespace NUMINAMATH_CALUDE_f_g_3_equals_28_l2282_228207

-- Define the functions f and g
def g (x : ℝ) : ℝ := x^2 + 1
def f (x : ℝ) : ℝ := 3*x - 2

-- State the theorem
theorem f_g_3_equals_28 : f (g 3) = 28 := by
  sorry

end NUMINAMATH_CALUDE_f_g_3_equals_28_l2282_228207


namespace NUMINAMATH_CALUDE_equation_solution_l2282_228297

theorem equation_solution : ∀ x : ℝ, 4 * x^2 - (x - 1)^2 = 0 ↔ x = -1 ∨ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2282_228297


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2282_228206

theorem gcd_lcm_product (a b : ℕ) (ha : a = 150) (hb : b = 90) :
  (Nat.gcd a b) * (Nat.lcm a b) = 13500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2282_228206


namespace NUMINAMATH_CALUDE_range_of_a_l2282_228283

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0) →
  a ∈ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2282_228283


namespace NUMINAMATH_CALUDE_franks_reading_rate_l2282_228253

/-- Represents a book with its properties --/
structure Book where
  pages : ℕ
  chapters : ℕ
  days_to_read : ℕ

/-- Calculates the number of chapters read per day --/
def chapters_per_day (b : Book) : ℚ :=
  (b.chapters : ℚ) / (b.days_to_read : ℚ)

/-- Theorem stating the number of chapters read per day for Frank's book --/
theorem franks_reading_rate (b : Book) 
    (h1 : b.pages = 193)
    (h2 : b.chapters = 15)
    (h3 : b.days_to_read = 660) :
    chapters_per_day b = 15 / 660 := by
  sorry

end NUMINAMATH_CALUDE_franks_reading_rate_l2282_228253


namespace NUMINAMATH_CALUDE_parabola_vertex_l2282_228254

/-- The vertex of the parabola y = -3x^2 + 6x + 1 is (1, 4) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -3 * x^2 + 6 * x + 1 → 
  ∃ (vertex_x vertex_y : ℝ), 
    vertex_x = 1 ∧ 
    vertex_y = 4 ∧ 
    ∀ (x' : ℝ), -3 * x'^2 + 6 * x' + 1 ≤ vertex_y :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2282_228254


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2282_228220

/-- Theorem: For a rhombus with area 150 cm² and one diagonal of 30 cm, the other diagonal is 10 cm -/
theorem rhombus_diagonal (area : ℝ) (d2 : ℝ) (d1 : ℝ) :
  area = 150 ∧ d2 = 30 ∧ area = (d1 * d2) / 2 → d1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2282_228220


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2282_228285

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 12 → b = 5 → c^2 = a^2 + b^2 → c = 13 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2282_228285


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2282_228225

def p (x : ℝ) : ℝ := 8*x^4 + 26*x^3 - 66*x^2 + 24*x

theorem roots_of_polynomial :
  (p 0 = 0) ∧ (p (1/2) = 0) ∧ (p (3/2) = 0) ∧ (p (-4) = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2282_228225


namespace NUMINAMATH_CALUDE_triangle_third_side_l2282_228244

theorem triangle_third_side (a b c : ℝ) : 
  a = 3 → b = 10 → c > 0 → 
  a + b + c = 6 * (⌊(a + b + c) / 6⌋ : ℝ) →
  c = 11 := by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l2282_228244


namespace NUMINAMATH_CALUDE_unique_fixed_point_of_odd_symmetries_l2282_228214

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Central symmetry transformation about a point -/
def centralSymmetry (center : Point) (p : Point) : Point :=
  { x := 2 * center.x - p.x, y := 2 * center.y - p.y }

/-- Composition of central symmetries -/
def compositeSymmetry (centers : List Point) : Point → Point :=
  centers.foldl (λ f center p => f (centralSymmetry center p)) id

theorem unique_fixed_point_of_odd_symmetries (n : ℕ) :
  let m := 2 * n + 1
  ∀ (midpoints : List Point),
    midpoints.length = m →
    ∃! (fixedPoint : Point), compositeSymmetry midpoints fixedPoint = fixedPoint :=
by
  sorry

#check unique_fixed_point_of_odd_symmetries

end NUMINAMATH_CALUDE_unique_fixed_point_of_odd_symmetries_l2282_228214


namespace NUMINAMATH_CALUDE_work_completion_time_l2282_228232

/-- The time it takes to complete a work given two workers with different rates and a specific work pattern. -/
theorem work_completion_time 
  (total_work : ℝ) 
  (p_rate : ℝ) 
  (q_rate : ℝ) 
  (p_alone_time : ℝ) :
  p_rate = total_work / 10 →
  q_rate = total_work / 6 →
  p_alone_time = 2 →
  let remaining_work := total_work - p_rate * p_alone_time
  let combined_rate := p_rate + q_rate
  total_work > 0 →
  p_rate > 0 →
  q_rate > 0 →
  p_alone_time + remaining_work / combined_rate = 5 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2282_228232


namespace NUMINAMATH_CALUDE_garden_length_l2282_228281

/-- Proves that a rectangular garden with length twice its width and perimeter 180 yards has a length of 60 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- Length is twice the width
  2 * width + 2 * length = 180 →  -- Perimeter is 180 yards
  length = 60 := by
sorry


end NUMINAMATH_CALUDE_garden_length_l2282_228281


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_C_nonempty_implies_a_gt_one_l2282_228259

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for part (1)
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 9} := by sorry

-- Theorem for part (2)
theorem intersection_A_C_nonempty_implies_a_gt_one (a : ℝ) :
  (A ∩ C a).Nonempty → a > 1 := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_C_nonempty_implies_a_gt_one_l2282_228259


namespace NUMINAMATH_CALUDE_model_b_sample_size_l2282_228298

/-- Calculates the number of items to be sampled from a stratum in stratified sampling -/
def stratifiedSampleSize (totalPopulation : ℕ) (stratumSize : ℕ) (totalSampleSize : ℕ) : ℕ :=
  (stratumSize * totalSampleSize) / totalPopulation

theorem model_b_sample_size :
  let totalProduction : ℕ := 9200
  let modelBProduction : ℕ := 6000
  let totalSampleSize : ℕ := 46
  stratifiedSampleSize totalProduction modelBProduction totalSampleSize = 30 := by
  sorry

end NUMINAMATH_CALUDE_model_b_sample_size_l2282_228298


namespace NUMINAMATH_CALUDE_ratio_problem_l2282_228275

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 2 / 5)
  (h3 : c / d = 9) : 
  d / a = 5 / 54 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2282_228275


namespace NUMINAMATH_CALUDE_correct_guess_probability_l2282_228272

/-- Represents a six-digit password with an unknown last digit -/
structure Password :=
  (first_five : Nat)
  (last_digit : Nat)

/-- The set of possible last digits -/
def possible_last_digits : Finset Nat := Finset.range 10

/-- The probability of guessing the correct password on the first try -/
def guess_probability (p : Password) : ℚ :=
  1 / (Finset.card possible_last_digits : ℚ)

theorem correct_guess_probability (p : Password) :
  p.last_digit ∈ possible_last_digits →
  guess_probability p = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_guess_probability_l2282_228272


namespace NUMINAMATH_CALUDE_two_squares_five_points_arrangement_l2282_228249

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a square in 2D space
structure Square where
  center : Point
  side_length : ℝ

-- Define a function to check if a point is inside a square
def is_inside (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side_length / 2 ∧
  abs (p.y - s.center.y) ≤ s.side_length / 2

-- Define the theorem
theorem two_squares_five_points_arrangement :
  ∃ (s1 s2 : Square) (p1 p2 p3 p4 p5 : Point),
    (is_inside p1 s1 ∧ is_inside p2 s1 ∧ is_inside p3 s1) ∧
    (is_inside p1 s2 ∧ is_inside p2 s2 ∧ is_inside p3 s2 ∧ is_inside p4 s2) :=
  sorry

end NUMINAMATH_CALUDE_two_squares_five_points_arrangement_l2282_228249


namespace NUMINAMATH_CALUDE_integer_roots_quadratic_l2282_228209

theorem integer_roots_quadratic (n : ℕ+) : 
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n.val = 3 ∨ n.val = 4) := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_quadratic_l2282_228209


namespace NUMINAMATH_CALUDE_reciprocal_sum_fractions_l2282_228274

theorem reciprocal_sum_fractions : (((3 : ℚ) / 4 + (1 : ℚ) / 6)⁻¹) = (12 : ℚ) / 11 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fractions_l2282_228274


namespace NUMINAMATH_CALUDE_translation_preserves_segment_find_translated_point_l2282_228256

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def apply_translation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_preserves_segment (A B A' : Point) (t : Translation) :
  apply_translation t A = A' →
  apply_translation t B = 
    { x := B.x + (A'.x - A.x), 
      y := B.y + (A'.y - A.y) } := by sorry

/-- The main theorem -/
theorem find_translated_point :
  let A : Point := { x := -1, y := 2 }
  let A' : Point := { x := 3, y := -4 }
  let B : Point := { x := 2, y := 4 }
  let t : Translation := { dx := A'.x - A.x, dy := A'.y - A.y }
  apply_translation t B = { x := 6, y := -2 } := by sorry

end NUMINAMATH_CALUDE_translation_preserves_segment_find_translated_point_l2282_228256


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_sixth_l2282_228266

theorem sqrt_fifth_power_sixth : (((5 : ℝ) ^ (1/2)) ^ 5) ^ (1/2) ^ 6 = 125 * (125 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_sixth_l2282_228266


namespace NUMINAMATH_CALUDE_inequality_proof_l2282_228218

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x) : 
  1 / (x - y)^2 + 1 / (y - z)^2 + 1 / (z - x)^2 ≥ 4 / (x*y + y*z + z*x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2282_228218


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l2282_228284

theorem square_of_binomial_constant (b : ℝ) : 
  (∃ (c : ℝ), ∀ (x : ℝ), 9*x^2 + 27*x + b = (3*x + c)^2) → b = 81/4 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l2282_228284


namespace NUMINAMATH_CALUDE_ant_farm_problem_l2282_228270

/-- Represents the number of ants of a specific species on a given day -/
def ant_count (initial : ℕ) (growth_rate : ℕ) (days : ℕ) : ℕ :=
  initial * (growth_rate ^ days)

theorem ant_farm_problem :
  ∀ a b c : ℕ,
  a + b + c = 50 →
  ant_count a 2 4 + ant_count b 3 4 + ant_count c 5 4 = 6230 →
  ant_count a 2 4 = 736 :=
by
  sorry

#check ant_farm_problem

end NUMINAMATH_CALUDE_ant_farm_problem_l2282_228270


namespace NUMINAMATH_CALUDE_find_novel_cost_l2282_228269

def novel_cost (initial_amount lunch_cost remaining_amount : ℚ) : Prop :=
  ∃ (novel_cost : ℚ),
    novel_cost > 0 ∧
    lunch_cost = 2 * novel_cost ∧
    initial_amount - (novel_cost + lunch_cost) = remaining_amount

theorem find_novel_cost :
  novel_cost 50 (2 * 7) 29 :=
sorry

end NUMINAMATH_CALUDE_find_novel_cost_l2282_228269


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2282_228210

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 5 = -3/4 →
  a 2 + a 3 + a 4 + a 5 = 5/4 →
  1/a 2 + 1/a 3 + 1/a 4 + 1/a 5 = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2282_228210


namespace NUMINAMATH_CALUDE_solution_set_eq_open_interval_l2282_228282

-- Define the logarithm function with base 10
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the solution set
def solution_set : Set ℝ := {x | log10 (x - 1) < 2}

-- State the theorem
theorem solution_set_eq_open_interval :
  solution_set = Set.Ioo 1 101 := by sorry

end NUMINAMATH_CALUDE_solution_set_eq_open_interval_l2282_228282


namespace NUMINAMATH_CALUDE_restaurant_glasses_count_l2282_228273

theorem restaurant_glasses_count :
  ∀ (x y : ℕ),
  -- x is the number of small boxes (12 glasses each)
  -- y is the number of large boxes (16 glasses each)
  y = x + 16 →  -- There are 16 more large boxes
  (12 * x + 16 * y) / (x + y) = 15 →  -- Average number of glasses per box is 15
  12 * x + 16 * y = 480  -- Total number of glasses
  := by sorry

end NUMINAMATH_CALUDE_restaurant_glasses_count_l2282_228273


namespace NUMINAMATH_CALUDE_m_range_for_g_l2282_228291

/-- Definition of an (a, b) type function -/
def is_ab_type_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) * f (a - x) = b

/-- Definition of the function g -/
def g (m : ℝ) (x : ℝ) : ℝ :=
  x^2 - m * (x - 1) + 1

/-- Main theorem -/
theorem m_range_for_g :
  ∀ m : ℝ,
  (m > 0) →
  (is_ab_type_function (g m) 1 4) →
  (∀ x ∈ Set.Icc 0 2, 1 ≤ g m x ∧ g m x ≤ 3) →
  (2 - 2 * Real.sqrt 6 / 3 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_m_range_for_g_l2282_228291


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l2282_228239

theorem consecutive_non_prime_powers (n : ℕ) : 
  ∃ x : ℤ, ∀ k ∈ Finset.range n, ¬ ∃ (p : ℕ) (m : ℕ), Prime p ∧ x + k.succ = p ^ m := by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l2282_228239


namespace NUMINAMATH_CALUDE_infinitely_many_primes_dividing_2_pow_k_minus_3_l2282_228230

theorem infinitely_many_primes_dividing_2_pow_k_minus_3 :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ k : ℕ, k > 0 ∧ p ∣ (2^k - 3)} :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_dividing_2_pow_k_minus_3_l2282_228230


namespace NUMINAMATH_CALUDE_tan_45_deg_eq_one_l2282_228201

/-- Tangent of 45 degrees is 1 -/
theorem tan_45_deg_eq_one :
  Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_deg_eq_one_l2282_228201


namespace NUMINAMATH_CALUDE_bobby_candy_left_l2282_228227

theorem bobby_candy_left (initial_candy : ℕ) (eaten_candy : ℕ) (h1 : initial_candy = 30) (h2 : eaten_candy = 23) :
  initial_candy - eaten_candy = 7 := by
sorry

end NUMINAMATH_CALUDE_bobby_candy_left_l2282_228227


namespace NUMINAMATH_CALUDE_sin_fourth_sum_eighths_pi_l2282_228233

theorem sin_fourth_sum_eighths_pi : 
  Real.sin (π / 8) ^ 4 + Real.sin (3 * π / 8) ^ 4 + 
  Real.sin (5 * π / 8) ^ 4 + Real.sin (7 * π / 8) ^ 4 = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_fourth_sum_eighths_pi_l2282_228233


namespace NUMINAMATH_CALUDE_power_function_through_point_l2282_228248

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x ^ α) →
  f 2 = Real.sqrt 2 →
  α = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2282_228248


namespace NUMINAMATH_CALUDE_shyam_weight_increase_l2282_228228

-- Define the original weight ratio
def weight_ratio : ℚ := 7 / 9

-- Define Ram's weight increase percentage
def ram_increase : ℚ := 12 / 100

-- Define the total new weight
def total_new_weight : ℚ := 165.6

-- Define the total weight increase percentage
def total_increase : ℚ := 20 / 100

-- Theorem to prove
theorem shyam_weight_increase : ∃ (original_ram : ℚ) (original_shyam : ℚ),
  original_shyam = original_ram / weight_ratio ∧
  (original_ram * (1 + ram_increase) + original_shyam * (1 + x)) = total_new_weight ∧
  (original_ram + original_shyam) * (1 + total_increase) = total_new_weight ∧
  abs (x - 26.29 / 100) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_shyam_weight_increase_l2282_228228


namespace NUMINAMATH_CALUDE_paco_initial_salty_cookies_l2282_228262

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := sorry

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := 40

/-- The number of salty cookies Paco ate -/
def eaten_salty_cookies : ℕ := 28

/-- The number of sweet cookies Paco ate -/
def eaten_sweet_cookies : ℕ := 15

/-- The difference between salty and sweet cookies eaten -/
def salty_sweet_difference : ℕ := 13

theorem paco_initial_salty_cookies :
  initial_salty_cookies = 56 :=
by sorry

end NUMINAMATH_CALUDE_paco_initial_salty_cookies_l2282_228262


namespace NUMINAMATH_CALUDE_polynomial_inequality_range_l2282_228212

theorem polynomial_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → a * x^3 - x^2 + 4*x + 3 ≥ 0) →
  a ∈ Set.Icc (-6) (-2) := by
sorry

end NUMINAMATH_CALUDE_polynomial_inequality_range_l2282_228212


namespace NUMINAMATH_CALUDE_circle_inequality_theta_range_l2282_228204

theorem circle_inequality_theta_range :
  ∀ θ : ℝ,
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (∀ x y : ℝ, (x - 2 * Real.cos θ)^2 + (y - 2 * Real.sin θ)^2 = 1 → x ≤ y) →
  (5 * Real.pi / 12 ≤ θ ∧ θ ≤ 13 * Real.pi / 12) :=
by sorry

end NUMINAMATH_CALUDE_circle_inequality_theta_range_l2282_228204


namespace NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l2282_228276

theorem rectangle_cylinder_volume_ratio :
  let rectangle_width : ℝ := 7
  let rectangle_height : ℝ := 9
  let cylinder1_height : ℝ := rectangle_height
  let cylinder1_circumference : ℝ := rectangle_width
  let cylinder2_height : ℝ := rectangle_width
  let cylinder2_circumference : ℝ := rectangle_height
  let cylinder1_volume : ℝ := (cylinder1_circumference ^ 2 * cylinder1_height) / (4 * Real.pi)
  let cylinder2_volume : ℝ := (cylinder2_circumference ^ 2 * cylinder2_height) / (4 * Real.pi)
  let larger_volume : ℝ := max cylinder1_volume cylinder2_volume
  let smaller_volume : ℝ := min cylinder1_volume cylinder2_volume
  larger_volume / smaller_volume = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l2282_228276


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2282_228257

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y => y = -2 * x + 1
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y → (y = 1/2 * x - 3/2)) →  -- Slope of L1 is 1/2
  (L2 P.1 P.2) →  -- L2 passes through P
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L2 x₂ y₂ → (x₂ - x₁) * (y₂ - y₁) = -1) →  -- L1 and L2 are perpendicular
  ∃ m b, ∀ x y, L2 x y ↔ y = m * x + b ∧ m = -2 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2282_228257


namespace NUMINAMATH_CALUDE_two_numbers_with_ratio_and_square_difference_l2282_228290

theorem two_numbers_with_ratio_and_square_difference (p q : ℝ) (hp : p > 0) (hpn : p ≠ 1) (hq : q > 0) :
  let x : ℝ := q / (p - 1)
  let y : ℝ := p * q / (p - 1)
  y / x = p ∧ (y^2 - x^2) / (y + x) = q := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_ratio_and_square_difference_l2282_228290


namespace NUMINAMATH_CALUDE_marias_age_l2282_228292

/-- 
Given that Jose is 12 years older than Maria and the sum of their ages is 40,
prove that Maria is 14 years old.
-/
theorem marias_age (maria jose : ℕ) 
  (h1 : jose = maria + 12) 
  (h2 : maria + jose = 40) : 
  maria = 14 := by
  sorry

end NUMINAMATH_CALUDE_marias_age_l2282_228292


namespace NUMINAMATH_CALUDE_soda_price_after_increase_l2282_228246

theorem soda_price_after_increase (candy_price : ℝ) (soda_price : ℝ) : 
  candy_price = 10 →
  candy_price + soda_price = 16 →
  9 = soda_price * 1.5 :=
by
  sorry

end NUMINAMATH_CALUDE_soda_price_after_increase_l2282_228246


namespace NUMINAMATH_CALUDE_south_american_stamps_cost_l2282_228258

def brazil_stamp_price : ℚ := 7 / 100
def peru_stamp_price : ℚ := 5 / 100
def brazil_50s_stamps : ℕ := 5
def brazil_60s_stamps : ℕ := 9
def peru_50s_stamps : ℕ := 12
def peru_60s_stamps : ℕ := 8

def total_south_american_stamps_cost : ℚ :=
  (brazil_stamp_price * (brazil_50s_stamps + brazil_60s_stamps)) +
  (peru_stamp_price * (peru_50s_stamps + peru_60s_stamps))

theorem south_american_stamps_cost :
  total_south_american_stamps_cost = 198 / 100 := by
  sorry

end NUMINAMATH_CALUDE_south_american_stamps_cost_l2282_228258


namespace NUMINAMATH_CALUDE_polynomial_divisibility_theorem_l2282_228224

def is_prime (p : ℕ) : Prop := Nat.Prime p

def divides (p n : ℕ) : Prop := n % p = 0

def polynomial_with_int_coeffs (P : ℕ → ℤ) : Prop :=
  ∃ (coeffs : List ℤ), ∀ x, P x = (coeffs.enum.map (λ (i, a) => a * (x ^ i))).sum

def constant_polynomial (P : ℕ → ℤ) : Prop :=
  ∃ c : ℤ, c ≠ 0 ∧ ∀ x, P x = c

def S (P : ℕ → ℤ) : Set ℕ :=
  {p | is_prime p ∧ ∃ n, divides p (P n).natAbs}

theorem polynomial_divisibility_theorem (P : ℕ → ℤ) 
  (h_poly : polynomial_with_int_coeffs P) :
  (Set.Finite (S P)) ↔ (constant_polynomial P) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_theorem_l2282_228224


namespace NUMINAMATH_CALUDE_regular_polygon_144_degrees_has_10_sides_l2282_228211

/-- A regular polygon with interior angles of 144 degrees has 10 sides -/
theorem regular_polygon_144_degrees_has_10_sides :
  ∀ (n : ℕ), n > 2 →
  (180 * (n - 2) : ℚ) / n = 144 →
  n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_144_degrees_has_10_sides_l2282_228211


namespace NUMINAMATH_CALUDE_max_sum_prism_with_pyramid_l2282_228252

/-- Represents a triangular prism --/
structure TriangularPrism :=
  (faces : Nat)
  (edges : Nat)
  (vertices : Nat)

/-- Represents the result of adding a pyramid to a face of a prism --/
structure PrismWithPyramid :=
  (faces : Nat)
  (edges : Nat)
  (vertices : Nat)

/-- Calculates the sum of faces, edges, and vertices --/
def sumElements (shape : PrismWithPyramid) : Nat :=
  shape.faces + shape.edges + shape.vertices

/-- Adds a pyramid to a triangular face of the prism --/
def addPyramidToTriangularFace (prism : TriangularPrism) : PrismWithPyramid :=
  { faces := prism.faces - 1 + 3,
    edges := prism.edges + 3,
    vertices := prism.vertices + 1 }

/-- Adds a pyramid to a quadrilateral face of the prism --/
def addPyramidToQuadrilateralFace (prism : TriangularPrism) : PrismWithPyramid :=
  { faces := prism.faces - 1 + 4,
    edges := prism.edges + 4,
    vertices := prism.vertices + 1 }

/-- The main theorem to be proved --/
theorem max_sum_prism_with_pyramid :
  let prism := TriangularPrism.mk 5 9 6
  let triangularResult := addPyramidToTriangularFace prism
  let quadrilateralResult := addPyramidToQuadrilateralFace prism
  max (sumElements triangularResult) (sumElements quadrilateralResult) = 28 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_prism_with_pyramid_l2282_228252


namespace NUMINAMATH_CALUDE_pigeons_eating_breadcrumbs_l2282_228288

theorem pigeons_eating_breadcrumbs (initial_pigeons : ℕ) (new_pigeons : ℕ) : 
  initial_pigeons = 1 → new_pigeons = 1 → initial_pigeons + new_pigeons = 2 := by
  sorry

end NUMINAMATH_CALUDE_pigeons_eating_breadcrumbs_l2282_228288


namespace NUMINAMATH_CALUDE_rosa_phone_calls_l2282_228278

theorem rosa_phone_calls (last_week : ℝ) (this_week : ℝ) (total : ℝ) 
  (h1 : last_week = 10.2)
  (h2 : this_week = 8.6)
  (h3 : total = last_week + this_week) :
  total = 18.8 := by
sorry

end NUMINAMATH_CALUDE_rosa_phone_calls_l2282_228278


namespace NUMINAMATH_CALUDE_factorization_equality_l2282_228243

theorem factorization_equality (x : ℝ) :
  (x^2 + 1) * (x^3 - x^2 + x - 1) = (x^2 + 1)^2 * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2282_228243


namespace NUMINAMATH_CALUDE_complex_square_equality_l2282_228286

theorem complex_square_equality (c d : ℕ+) :
  (↑c - Complex.I * ↑d) ^ 2 = 18 - 8 * Complex.I →
  ↑c - Complex.I * ↑d = 5 - Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_square_equality_l2282_228286


namespace NUMINAMATH_CALUDE_circle_point_range_l2282_228267

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A (m : ℝ) : ℝ × ℝ := (-m, 0)
def B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Main theorem
theorem circle_point_range (m : ℝ) :
  m > 0 →
  (∃ a b : ℝ, C a b ∧
    dot_product (a + m, b) (a - m, b) = 0) →
  4 ≤ m ∧ m ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_point_range_l2282_228267


namespace NUMINAMATH_CALUDE_nails_to_buy_l2282_228242

theorem nails_to_buy (initial_nails : ℕ) (found_nails : ℕ) (total_needed : ℕ) : 
  initial_nails = 247 → found_nails = 144 → total_needed = 500 →
  total_needed - (initial_nails + found_nails) = 109 :=
by sorry

end NUMINAMATH_CALUDE_nails_to_buy_l2282_228242


namespace NUMINAMATH_CALUDE_right_angle_on_circle_l2282_228296

/-- The circle C with equation (x - √3)² + (y - 1)² = 1 -/
def circle_C (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + (y - 1)^2 = 1

/-- The point A with coordinates (-t, 0) -/
def point_A (t : ℝ) : ℝ × ℝ := (-t, 0)

/-- The point B with coordinates (t, 0) -/
def point_B (t : ℝ) : ℝ × ℝ := (t, 0)

/-- Predicate to check if a point P forms a right angle with A and B -/
def forms_right_angle (P : ℝ × ℝ) (t : ℝ) : Prop :=
  let A := point_A t
  let B := point_B t
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

theorem right_angle_on_circle (t : ℝ) :
  t > 0 →
  (∃ P : ℝ × ℝ, circle_C P.1 P.2 ∧ forms_right_angle P t) →
  t ∈ Set.Icc 1 3 :=
sorry

end NUMINAMATH_CALUDE_right_angle_on_circle_l2282_228296


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l2282_228221

theorem binomial_expansion_sum (n : ℕ) : 
  (∃ P S : ℕ, (P = (3 + 1)^n) ∧ (S = 2^n) ∧ (P + S = 272)) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l2282_228221


namespace NUMINAMATH_CALUDE_park_trees_l2282_228222

/-- The number of trees in a rectangular park -/
def num_trees (length width tree_density : ℕ) : ℕ :=
  (length * width) / tree_density

/-- Proof that a park with given dimensions and tree density has 100,000 trees -/
theorem park_trees : num_trees 1000 2000 20 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_park_trees_l2282_228222


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2282_228250

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposites -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_sum (a b : ℝ) :
  symmetric_about_x_axis (a, 4) (-2, b) → a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2282_228250


namespace NUMINAMATH_CALUDE_min_value_sum_of_distances_min_value_achievable_l2282_228264

theorem min_value_sum_of_distances (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) + Real.sqrt ((2 + x)^2 + x^2) ≥ 6 * Real.sqrt 2 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) + Real.sqrt ((2 + x)^2 + x^2) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_distances_min_value_achievable_l2282_228264


namespace NUMINAMATH_CALUDE_simplify_expression_l2282_228265

theorem simplify_expression (a b : ℝ) : a - 4*(2*a - b) - 2*(a + 2*b) = -9*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2282_228265


namespace NUMINAMATH_CALUDE_power_sum_tenth_l2282_228263

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^10 + b^10 = 123 -/
theorem power_sum_tenth (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_tenth_l2282_228263


namespace NUMINAMATH_CALUDE_job_completion_time_l2282_228223

/-- Given that Sylvia can complete a job in 45 minutes and Carla can complete
    the same job in 30 minutes, prove that together they can complete the job
    in 18 minutes. -/
theorem job_completion_time (sylvia_time carla_time : ℝ) 
    (h_sylvia : sylvia_time = 45)
    (h_carla : carla_time = 30) :
    1 / (1 / sylvia_time + 1 / carla_time) = 18 := by
  sorry


end NUMINAMATH_CALUDE_job_completion_time_l2282_228223


namespace NUMINAMATH_CALUDE_least_k_value_l2282_228226

theorem least_k_value (k : ℤ) : ∀ n : ℤ, n ≥ 7 ↔ (0.00010101 * (10 : ℝ)^n > 1000) :=
by sorry

end NUMINAMATH_CALUDE_least_k_value_l2282_228226


namespace NUMINAMATH_CALUDE_two_digit_number_difference_divisibility_l2282_228294

theorem two_digit_number_difference_divisibility (A B : Nat) 
  (h1 : A ≠ B) (h2 : A > B) (h3 : A < 10) (h4 : B < 10) : 
  ∃ k : Int, (10 * A + B) - ((10 * B + A) - 5) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_divisibility_l2282_228294


namespace NUMINAMATH_CALUDE_min_abs_diff_solution_product_l2282_228287

theorem min_abs_diff_solution_product (x y : ℤ) : 
  (20 * x + 19 * y = 2019) →
  (∀ a b : ℤ, 20 * a + 19 * b = 2019 → |x - y| ≤ |a - b|) →
  x * y = 2623 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_diff_solution_product_l2282_228287


namespace NUMINAMATH_CALUDE_sand_price_per_ton_l2282_228205

theorem sand_price_per_ton 
  (total_cost : ℕ) 
  (cement_bags : ℕ) 
  (cement_price_per_bag : ℕ) 
  (sand_lorries : ℕ) 
  (sand_tons_per_lorry : ℕ) 
  (h1 : total_cost = 13000)
  (h2 : cement_bags = 500)
  (h3 : cement_price_per_bag = 10)
  (h4 : sand_lorries = 20)
  (h5 : sand_tons_per_lorry = 10) : 
  (total_cost - cement_bags * cement_price_per_bag) / (sand_lorries * sand_tons_per_lorry) = 40 := by
sorry

end NUMINAMATH_CALUDE_sand_price_per_ton_l2282_228205


namespace NUMINAMATH_CALUDE_jerrys_average_increase_l2282_228299

theorem jerrys_average_increase (initial_average : ℝ) (fourth_test_score : ℝ) : 
  initial_average = 78 →
  fourth_test_score = 86 →
  (3 * initial_average + fourth_test_score) / 4 - initial_average = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_average_increase_l2282_228299


namespace NUMINAMATH_CALUDE_investment_return_calculation_l2282_228240

/-- Calculates the monthly return given the current value, duration, and growth factor of an investment. -/
def calculateMonthlyReturn (currentValue : ℚ) (months : ℕ) (growthFactor : ℚ) : ℚ :=
  (currentValue * (growthFactor - 1)) / months

/-- Theorem stating that an investment tripling over 5 months with a current value of $90 has a monthly return of $12. -/
theorem investment_return_calculation :
  let currentValue : ℚ := 90
  let months : ℕ := 5
  let growthFactor : ℚ := 3
  calculateMonthlyReturn currentValue months growthFactor = 12 := by
  sorry

end NUMINAMATH_CALUDE_investment_return_calculation_l2282_228240


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l2282_228295

theorem complex_imaginary_part (z : ℂ) : 
  z = -2 + I → Complex.im (z + z⁻¹) = 4/5 := by sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l2282_228295


namespace NUMINAMATH_CALUDE_heart_king_probability_l2282_228245

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of kings in a standard deck -/
def NumKings : ℕ := 4

/-- Probability of drawing a heart as the first card and a king as the second card -/
def prob_heart_then_king (deck : ℕ) (hearts : ℕ) (kings : ℕ) : ℚ :=
  (hearts / deck) * (kings / (deck - 1))

theorem heart_king_probability :
  prob_heart_then_king StandardDeck NumHearts NumKings = 1 / StandardDeck := by
  sorry

end NUMINAMATH_CALUDE_heart_king_probability_l2282_228245


namespace NUMINAMATH_CALUDE_article_cost_l2282_228280

/-- 
Given an article with two selling prices and a relationship between the gains,
prove that the cost of the article is 60.
-/
theorem article_cost (selling_price_1 selling_price_2 : ℝ) 
  (h1 : selling_price_1 = 360)
  (h2 : selling_price_2 = 340)
  (h3 : selling_price_1 - selling_price_2 = 0.05 * (selling_price_2 - cost)) :
  cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l2282_228280


namespace NUMINAMATH_CALUDE_sum_of_inserted_numbers_l2282_228237

/-- A sequence of five real numbers -/
structure Sequence :=
  (a b c : ℝ)

/-- Check if the first four terms form a harmonic progression -/
def isHarmonicProgression (s : Sequence) : Prop :=
  ∃ (h : ℝ), 1/4 - 1/s.a = 1/s.a - 1/s.b ∧ 1/s.a - 1/s.b = 1/s.b - 1/s.c

/-- Check if the last four terms form a quadratic sequence -/
def isQuadraticSequence (s : Sequence) : Prop :=
  ∃ (p q : ℝ), 
    s.a = 1^2 + p + q ∧
    s.b = 2^2 + 2*p + q ∧
    s.c = 3^2 + 3*p + q ∧
    16 = 4^2 + 4*p + q

/-- The main theorem -/
theorem sum_of_inserted_numbers (s : Sequence) :
  s.a > 0 ∧ s.b > 0 ∧ s.c > 0 →
  isHarmonicProgression s →
  isQuadraticSequence s →
  s.a + s.b + s.c = 33 :=
sorry

end NUMINAMATH_CALUDE_sum_of_inserted_numbers_l2282_228237


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2282_228213

theorem triangle_angle_measure (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 →
  a^2 + b^2 = 6 * a * b * Real.cos C →
  Real.sin C^2 = 2 * Real.sin A * Real.sin B →
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2282_228213


namespace NUMINAMATH_CALUDE_integral_x_squared_minus_x_l2282_228255

theorem integral_x_squared_minus_x : ∫ (x : ℝ) in (0)..(1), (x^2 - x) = -1/6 := by sorry

end NUMINAMATH_CALUDE_integral_x_squared_minus_x_l2282_228255


namespace NUMINAMATH_CALUDE_friends_assignment_count_l2282_228268

theorem friends_assignment_count : 
  (∀ n : ℕ, n > 0 → n ^ 8 = (n * n ^ 7)) →
  4 ^ 8 = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friends_assignment_count_l2282_228268


namespace NUMINAMATH_CALUDE_smallest_difference_of_powers_eleven_is_representable_eleven_is_smallest_l2282_228238

theorem smallest_difference_of_powers : 
  ∀ k l : ℕ, 36^k - 5^l > 0 → 36^k - 5^l ≥ 11 :=
by sorry

theorem eleven_is_representable : 
  ∃ k l : ℕ, 36^k - 5^l = 11 :=
by sorry

theorem eleven_is_smallest :
  (∃ k l : ℕ, 36^k - 5^l = 11) ∧
  (∀ m n : ℕ, 36^m - 5^n > 0 → 36^m - 5^n ≥ 11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_of_powers_eleven_is_representable_eleven_is_smallest_l2282_228238


namespace NUMINAMATH_CALUDE_prime_sum_product_l2282_228234

theorem prime_sum_product (p q : ℕ) : 
  Prime p → Prime q → p + q = 85 → p * q = 166 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l2282_228234


namespace NUMINAMATH_CALUDE_robie_initial_cards_robie_initial_cards_proof_l2282_228202

theorem robie_initial_cards (cards_per_box : ℕ) (loose_cards : ℕ) (boxes_given : ℕ) 
  (boxes_returned : ℕ) (current_boxes : ℕ) (cards_bought : ℕ) (cards_traded : ℕ) : ℕ :=
  let initial_boxes := current_boxes - boxes_returned + boxes_given
  let boxed_cards := initial_boxes * cards_per_box
  let total_cards := boxed_cards + loose_cards
  let initial_cards := total_cards - cards_bought
  initial_cards

theorem robie_initial_cards_proof :
  robie_initial_cards 30 18 8 2 15 21 12 = 627 := by
  sorry

end NUMINAMATH_CALUDE_robie_initial_cards_robie_initial_cards_proof_l2282_228202


namespace NUMINAMATH_CALUDE_sqrt_expressions_equality_l2282_228208

theorem sqrt_expressions_equality : 
  (2 * Real.sqrt 3 - 3 * Real.sqrt 12 + 5 * Real.sqrt 27 = 11 * Real.sqrt 3) ∧
  ((1 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 6) - (2 * Real.sqrt 3 - 1)^2 = 
   -2 * Real.sqrt 2 + 4 * Real.sqrt 3 - 13) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_equality_l2282_228208
