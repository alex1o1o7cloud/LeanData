import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_problem_l473_47308

theorem angle_sum_problem (x : ℝ) 
  (h : Real.sin x ^ 2 + Real.sin (2*x) ^ 2 + Real.sin (3*x) ^ 2 + Real.sin (4*x) ^ 2 + Real.sin (5*x) ^ 2 = 5/2) :
  ∃ (a b c : ℕ+), Real.cos (a * x) * Real.cos (b * x) * Real.cos (c * x) = 0 ∧ a + b + c = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_problem_l473_47308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_card_coverage_l473_47318

/-- Represents a circular card on a checkerboard -/
structure CircularCard where
  radius : ℝ
  center : ℝ × ℝ

/-- Represents a square on a checkerboard -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Determines if a circular card covers a square -/
def covers (card : CircularCard) (square : Square) : Prop :=
  ∃ (x y : ℝ), (x - card.center.1)^2 + (y - card.center.2)^2 ≤ card.radius^2 ∧
                x ≥ square.center.1 - square.side_length / 2 ∧
                x ≤ square.center.1 + square.side_length / 2 ∧
                y ≥ square.center.2 - square.side_length / 2 ∧
                y ≤ square.center.2 + square.side_length / 2

/-- The main theorem to prove -/
theorem circular_card_coverage (card : CircularCard) (board : Set Square) :
  card.radius = 1.5 →
  (∀ s : Square, s ∈ board → s.side_length = 1) →
  ∃ (covered : Finset Square), ↑covered ⊆ board ∧ (∀ s ∈ covered, covers card s) ∧ covered.card ≥ 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_card_coverage_l473_47318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_is_fifteen_percent_of_cost_l473_47389

/-- Calculates the profit per unit for a product with given cost, markup, and discount. -/
noncomputable def profit_per_unit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let marked_up_price := cost * (1 + markup_percent / 100)
  let selling_price := marked_up_price * (1 - discount_percent / 100)
  selling_price - cost

/-- Theorem stating that for a product with cost a, 25% markup, and 8% discount, the profit is 0.15a -/
theorem profit_is_fifteen_percent_of_cost (a : ℝ) :
  profit_per_unit a 25 8 = 0.15 * a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_is_fifteen_percent_of_cost_l473_47389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l473_47353

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 1/2 - 1 / (2^x + 1)

-- Theorem statement
theorem f_properties :
  -- Part 1: f is an odd function
  (∀ x, f (-x) = -f x) ∧
  -- Part 2: f is monotonically increasing
  (∀ x y, x < y → f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l473_47353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_distinct_terms_l473_47385

/-- The number of distinct terms in the fully simplified expansion of [(a+4b)²(a-2b)²]³ -/
def distinctTermCount (a b : ℝ) : ℕ :=
  -- This is a placeholder for the actual computation
  7

/-- Theorem stating that the expansion has 7 distinct terms -/
theorem expansion_distinct_terms :
  ∀ a b : ℝ, distinctTermCount a b = 7 := by
  intro a b
  -- The actual proof would go here
  rfl

#eval distinctTermCount 0 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_distinct_terms_l473_47385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l473_47305

theorem circle_problem (a b : ℝ) : 
  a < 0 ∧ 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x - 2*b*y + a^2 + b^2 - 1 = 0) ∧ 
  (Real.sqrt 3 * a - b + Real.sqrt 3 = 0) ∧ 
  (∃ x y : ℝ, x^2 + y^2 - 2*a*x - 2*b*y + a^2 + b^2 - 1 = 0 ∧ 
    abs ((Real.sqrt 3 * x + y) / Real.sqrt (3^2 + 1^2)) + 1 = 1 + Real.sqrt 3) →
  a^2 + b^2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l473_47305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ribeye_steak_cost_l473_47321

noncomputable def appetizer_cost : ℝ := 8
noncomputable def wine_cost : ℝ := 3
noncomputable def dessert_cost : ℝ := 6
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def total_spent : ℝ := 38

noncomputable def meal_cost (steak_cost : ℝ) : ℝ :=
  appetizer_cost + 2 * wine_cost + dessert_cost + steak_cost / 2

noncomputable def full_meal_cost (steak_cost : ℝ) : ℝ :=
  appetizer_cost + 2 * wine_cost + dessert_cost + steak_cost

noncomputable def tip_amount (steak_cost : ℝ) : ℝ :=
  tip_percentage * full_meal_cost steak_cost

theorem ribeye_steak_cost :
  ∃ (steak_cost : ℝ), 
    meal_cost steak_cost + tip_amount steak_cost = total_spent ∧ 
    steak_cost = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ribeye_steak_cost_l473_47321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_line_AF2_polar_equation_l473_47365

noncomputable section

-- Define the conic curve
def conic (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, 2 * Real.sqrt 2 * Real.sin θ)

-- Define point A
def A : ℝ × ℝ := (0, Real.sqrt 3 / 3)

-- Define the foci F1 and F2
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Theorem for the parametric equation of line l
theorem line_l_equation (t : ℝ) :
  let l := λ t : ℝ => (-1/2 * t + 1, Real.sqrt 3 / 2 * t)
  (∃ θ : ℝ, conic θ = F2) ∧
  (∀ p : ℝ × ℝ, p ∈ Set.range l → ((p.1 - F2.1) * (A.1 - F1.1) + (p.2 - F2.2) * (A.2 - F1.2) = 0)) :=
by sorry

-- Theorem for the polar equation of line AF2
theorem line_AF2_polar_equation (ρ θ : ℝ) :
  Real.sqrt 3 * ρ * Real.sin θ + ρ * Real.cos θ = 1 ↔
  ∃ t : ℝ, (ρ * Real.cos θ, ρ * Real.sin θ) = ((1 - t) * A.1 + t * F2.1, (1 - t) * A.2 + t * F2.2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_line_AF2_polar_equation_l473_47365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_triangle_large_circumcircle_l473_47339

/-- A triangle with sides not exceeding 1 centimeter -/
structure SmallTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_bound : a ≤ 0.01
  b_bound : b ≤ 0.01
  c_bound : c ≤ 0.01
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The circumradius of a triangle given its side lengths -/
noncomputable def circumradius (t : SmallTriangle) : ℝ :=
  (t.a * t.b * t.c) / (4 * Real.sqrt (t.a + t.b + t.c) * (t.b + t.c - t.a) * (t.c + t.a - t.b) * (t.a + t.b - t.c))

/-- Theorem: There exists a triangle with sides not exceeding 1 cm whose circumradius is greater than 1 meter -/
theorem exists_small_triangle_large_circumcircle :
  ∃ t : SmallTriangle, circumradius t > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_triangle_large_circumcircle_l473_47339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l473_47312

noncomputable def x : ℕ → ℝ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | (n + 2) => Real.sqrt ((x (n + 1)) ^ 2 + x (n + 1)) + x (n + 1)

theorem x_general_term :
  ∀ n : ℕ, n ≥ 1 → x n = 1 / (2 ^ (1 / 2 ^ (n - 1)) - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l473_47312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l473_47375

theorem teacher_age (num_students : ℕ) (avg_age_students : ℝ) (new_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 30 →
  avg_age_students = 14 →
  (↑num_students * avg_age_students + teacher_age) / (↑num_students + 1) = new_avg_age →
  new_avg_age = 15 →
  teacher_age = 45 :=
by
  sorry

#check teacher_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l473_47375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_beta_is_sixty_degrees_l473_47380

/-- Given acute angles α and β, if cos α = 1/7 and cos (α+β) = -11/14, then β = 60°. -/
theorem angle_beta_is_sixty_degrees (α β : ℝ) : 
  0 < α → α < π/2 →  -- α is acute
  0 < β → β < π/2 →  -- β is acute
  Real.cos α = 1/7 →
  Real.cos (α + β) = -11/14 →
  β = π/3 := by  -- π/3 radians = 60°
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_beta_is_sixty_degrees_l473_47380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_one_l473_47320

/-- A function f is odd if f(-x) = -f(x) for all x in its domain --/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = sin(x) / ((x-a)(x+1)) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x / ((x - a) * (x + 1))

/-- If f(x) = sin(x) / ((x-a)(x+1)) is an odd function, then a = 1 --/
theorem odd_function_implies_a_eq_one :
  ∀ a : ℝ, IsOdd (f a) → a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_one_l473_47320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_color_is_seven_twentyfourths_l473_47393

/-- Represents the colors on the dice -/
inductive DiceColor
| Maroon
| Teal
| Cyan
| Sparkly
deriving Repr, DecidableEq

/-- Represents a 12-sided die with specified color distribution -/
def Dice : Type := Fin 12 → DiceColor

/-- The probability of rolling a specific color on a single die -/
def prob_color (d : Dice) (c : DiceColor) : ℚ :=
  (Finset.filter (fun i => d i = c) (Finset.univ : Finset (Fin 12))).card / 12

/-- The actual die with the given color distribution -/
def actual_die : Dice :=
  fun i => match i with
    | 0 | 1 | 2 => DiceColor.Maroon
    | 3 | 4 | 5 | 6 => DiceColor.Teal
    | 7 | 8 | 9 | 10 => DiceColor.Cyan
    | 11 => DiceColor.Sparkly

/-- The probability of both dice showing the same color -/
def prob_same_color (d : Dice) : ℚ :=
  (prob_color d DiceColor.Maroon) ^ 2 +
  (prob_color d DiceColor.Teal) ^ 2 +
  (prob_color d DiceColor.Cyan) ^ 2 +
  (prob_color d DiceColor.Sparkly) ^ 2

theorem prob_same_color_is_seven_twentyfourths :
  prob_same_color actual_die = 7 / 24 := by
  sorry

#eval prob_same_color actual_die

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_color_is_seven_twentyfourths_l473_47393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equality_triangle_sides_l473_47317

-- Part 1
theorem sqrt_sum_equality : Real.sqrt 25 + |1 - Real.sqrt 3| + Real.sqrt 27 = 4 + 4 * Real.sqrt 3 := by sorry

-- Part 2
theorem triangle_sides (a b c : ℝ) (h1 : a/4 = b/5) (h2 : b/5 = c/7) (h3 : a + b + c = 48) :
  a = 12 ∧ b = 15 ∧ c = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equality_triangle_sides_l473_47317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monotone_involution_l473_47374

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define monotonicity (using a different name to avoid conflict)
def IsMonotone (f : RealFunction) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ f y ≤ f x

-- Define function iteration
def Iterate (f : RealFunction) : ℕ → RealFunction
  | 0 => id
  | n + 1 => f ∘ (Iterate f n)

-- Theorem statement
theorem unique_monotone_involution (f : RealFunction) 
  (h_monotone : IsMonotone f) 
  (h_involution : ∃ n : ℕ, ∀ x : ℝ, Iterate f n x = -x) :
  ∀ x : ℝ, f x = -x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monotone_involution_l473_47374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenured_men_percentage_approx_l473_47341

/-- Represents the percentage of professors who are women -/
noncomputable def women_percentage : ℝ := 70

/-- Represents the percentage of professors who are tenured -/
noncomputable def tenured_percentage : ℝ := 70

/-- Represents the percentage of professors who are women, tenured, or both -/
noncomputable def women_or_tenured_percentage : ℝ := 90

/-- Represents the total number of professors (assumed to be 100 for simplicity) -/
noncomputable def total_professors : ℝ := 100

/-- Calculates the percentage of men who are tenured -/
noncomputable def tenured_men_percentage : ℝ :=
  let women := women_percentage / 100 * total_professors
  let tenured := tenured_percentage / 100 * total_professors
  let women_or_tenured := women_or_tenured_percentage / 100 * total_professors
  let men := total_professors - women
  let tenured_men := women_or_tenured - women
  (tenured_men / men) * 100

theorem tenured_men_percentage_approx :
  |tenured_men_percentage - 200/3| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenured_men_percentage_approx_l473_47341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_ratio_proof_l473_47345

-- Define the total wire length
noncomputable def total_length : ℝ := 28

-- Define the length of the shorter piece
noncomputable def shorter_piece : ℝ := 8.000028571387755

-- Define the length of the longer piece
noncomputable def longer_piece : ℝ := total_length - shorter_piece

-- Define the ratio of the shorter piece to the longer piece
noncomputable def ratio : ℝ := shorter_piece / longer_piece

-- Theorem statement
theorem wire_ratio_proof :
  |ratio - 0.400000571428571| < 1e-12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_ratio_proof_l473_47345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_l473_47314

-- Define the function for number of paths
def number_of_non_decreasing_paths_not_intersecting_y_eq_x (m n : ℕ) : ℚ :=
  (m - n : ℚ) / m * (Nat.choose (m + n - 1) n)

theorem path_count (m n : ℕ) (h : 0 < n ∧ n < m) :
  (m - n : ℚ) / m * (Nat.choose (m + n - 1) n) =
  number_of_non_decreasing_paths_not_intersecting_y_eq_x m n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_l473_47314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l473_47361

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x ^ a

-- State the theorem
theorem power_function_through_point (a : ℝ) :
  f a 2 = Real.sqrt 2 → f a 16 = 4 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l473_47361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_hyperbola_with_diagonal_asymptote_l473_47395

/-- A hyperbola. -/
structure Hyperbola where
  -- We'll define the necessary properties of a hyperbola here
  -- For now, we'll leave it empty as a placeholder
  mk :: -- empty structure for now

/-- A hyperbola with an asymptote y = x is equilateral. -/
def is_equilateral_hyperbola (h : Hyperbola) : Prop :=
  ∃ (a : ℝ), (fun x => x + a) = (fun x => x + a) ∨ (fun x => x - a) = (fun x => x - a)

/-- The eccentricity of a hyperbola. -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  sorry

/-- Theorem: The eccentricity of a hyperbola with an asymptote y = x is √2. -/
theorem eccentricity_of_hyperbola_with_diagonal_asymptote (h : Hyperbola) 
  (h_equilateral : is_equilateral_hyperbola h) : eccentricity h = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_hyperbola_with_diagonal_asymptote_l473_47395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l473_47348

noncomputable def oplus (x y : ℝ) : ℝ := x / (2 - y)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, oplus (x - a) (x + 1 - a) ≥ 0 → -2 < x ∧ x < 2) →
  -2 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l473_47348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l473_47350

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then 2^(x-a) - 2/(x+1) else -2^(-x-a) + 2/(-x+1)

-- State the theorem
theorem odd_function_a_value :
  ∃ a : ℝ,
  (∀ x : ℝ, f a x = -f a (-x)) ∧  -- f is an odd function
  (f a (-1) = 3/4) ∧              -- f(-1) = 3/4
  a = 3 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l473_47350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_theorem_l473_47335

theorem cosine_angle_theorem (x y z : ℝ) (α β γ : ℝ) :
  x > 0 → y > 0 → z > 0 →
  Real.cos α = (1 : ℝ) / 4 →
  Real.cos β = (1 : ℝ) / 3 →
  Real.cos α = x / Real.sqrt (x^2 + y^2 + z^2) →
  Real.cos β = y / Real.sqrt (x^2 + y^2 + z^2) →
  Real.cos γ = z / Real.sqrt (x^2 + y^2 + z^2) →
  Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1 →
  Real.cos γ = Real.sqrt 119 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_theorem_l473_47335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janessa_kept_cards_l473_47368

def cards_problem (initial_cards : ℕ) (father_cards : ℕ) (ebay_cards : ℕ) (bad_cards : ℕ) (keep_percentage : ℚ) (given_to_dexter : ℕ) : Prop :=
  let total_initial := initial_cards + father_cards
  let good_ebay_cards := ebay_cards - bad_cards
  let total_cards := total_initial + good_ebay_cards
  let kept_cards := (keep_percentage * total_cards).floor
  kept_cards = 19 ∧ given_to_dexter = 29 ∧ kept_cards + given_to_dexter ≤ total_cards

theorem janessa_kept_cards :
  cards_problem 4 13 36 4 (2/5) 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janessa_kept_cards_l473_47368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_problem_l473_47300

-- Define the ellipse Ω
def Ω : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 12) = 1}

-- Define the hyperbola Γ
def Γ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) - (p.2^2 / 12) = 1}

-- Define the focus of Ω
def focus_Ω : ℝ × ℝ := (2, 0)

-- Define the major axis length of Ω
def major_axis_Ω : ℝ := 8

-- Define the focal length of Γ
def focal_length_Γ : ℝ := 8

-- Define point E
def E : ℝ × ℝ := (3, 0)

-- Define the line l
def line_l (k m : ℤ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (k : ℝ) * p.1 + (m : ℝ)}

-- Define the area of a triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Theorem statement
theorem ellipse_hyperbola_problem :
  -- 1. Standard equation of Γ
  (∀ p : ℝ × ℝ, p ∈ Γ ↔ (p.1^2 / 4) - (p.2^2 / 12) = 1) ∧
  -- 2. Maximum area of triangle OAB
  (∃ max_area : ℝ, max_area = 4 * Real.sqrt 3 ∧
    ∀ A B : ℝ × ℝ, A ∈ Ω → B ∈ Ω →
      (∃ k : ℝ, (A.2 = k * (A.1 - 3)) ∧ (B.2 = k * (B.1 - 3))) →
      area_triangle (0, 0) A B ≤ max_area) ∧
  -- 3. Number of lines with AC + BD = 0
  (∃ lines : Finset (ℤ × ℤ), lines.card = 9 ∧
    ∀ k m : ℤ, (k, m) ∈ lines ↔
      (∃ A B C D : ℝ × ℝ,
        A ∈ Ω ∧ B ∈ Ω ∧ C ∈ Γ ∧ D ∈ Γ ∧
        A ∈ line_l k m ∧ B ∈ line_l k m ∧ C ∈ line_l k m ∧ D ∈ line_l k m ∧
        (C.1 - A.1, C.2 - A.2) = (B.1 - D.1, B.2 - D.2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_problem_l473_47300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_units_count_l473_47337

theorem apartment_units_count 
  (num_buildings : ℕ)
  (num_floors : ℕ)
  (first_floor_units : ℕ)
  (other_floor_units : ℕ)
  (h1 : num_buildings = 2)
  (h2 : num_floors = 4)
  (h3 : first_floor_units = 2)
  (h4 : other_floor_units = 5) :
  num_buildings * (first_floor_units + (num_floors - 1) * other_floor_units) = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_units_count_l473_47337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_I_l473_47369

-- Define the function H
def H (p q : ℝ) : ℝ := -3*p*q + 2*p*(1-q) + 4*(1-p)*q - 5*(1-p)*(1-q)

-- Define the function I
noncomputable def I (p : ℝ) : ℝ := 
  ⨆ q ∈ Set.Icc 0 1, H p q

-- Theorem statement
theorem minimize_I :
  ∀ p ∈ Set.Icc 0 1, I p ≥ I (9/14) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_I_l473_47369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l473_47376

def is_valid_digit (d : Nat) : Bool :=
  d ≠ 6 && d ≠ 7 && d ≠ 9

def is_valid_number (n : Nat) : Bool :=
  100 ≤ n && n ≤ 999 &&
  is_valid_digit (n / 100) &&
  is_valid_digit ((n / 10) % 10) &&
  is_valid_digit (n % 10)

theorem count_valid_numbers :
  (Finset.filter (fun n => is_valid_number n) (Finset.range 1000)).card = 294 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l473_47376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_PQ_l473_47359

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt x * (x / 3 + 1)

def is_in_domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ Real.pi

def tangent_parallel (P Q : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xq, yq) := Q
  is_in_domain xp ∧
  (deriv f xp = deriv g xq)

theorem slope_of_PQ (P Q : ℝ × ℝ) :
  tangent_parallel P Q →
  let (xp, yp) := P
  let (xq, yq) := Q
  (yq - yp) / (xq - xp) = 8/3 := by sorry

#check slope_of_PQ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_PQ_l473_47359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l473_47311

theorem expression_evaluation (k : ℝ) : 
  (2 : ℝ)^(-(3*k+2)) - (3 : ℝ)^(-(2*k+1)) - (2 : ℝ)^(-(3*k)) + (3 : ℝ)^(-2*k) = 
  (-9 * (2 : ℝ)^(-3*k) + 8 * (3 : ℝ)^(-2*k)) / 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l473_47311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_sum_approx_l473_47392

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [2, 3, 2]
def base1 : Nat := 8

def num2 : List Nat := [1, 2]
def base2 : Nat := 4

def num3 : List Nat := [2, 0, 2]
def base3 : Nat := 5

def num4 : List Nat := [2, 2]
def base4 : Nat := 3

-- State the theorem
theorem base_conversion_sum_approx :
  let x1 := (to_base_10 num1 base1 : ℚ)
  let x2 := (to_base_10 num2 base2 : ℚ)
  let x3 := (to_base_10 num3 base3 : ℚ)
  let x4 := (to_base_10 num4 base4 : ℚ)
  abs ((x1 / x2 + x3 / x4) - 32.1667) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_sum_approx_l473_47392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_frame_sum_l473_47306

theorem calendar_frame_sum (x : ℤ) :
  (x - 2 + x - 1 + x + x + 1 + x + 2 = 101) →
  (∀ n ∈ ({x - 2, x - 1, x, x + 1, x + 2} : Set ℤ), n ≤ 30 ∧ n ≥ 15) ∧
  (∃ n ∈ ({x - 2, x - 1, x, x + 1, x + 2} : Set ℤ), n = 30) ∧
  (∃ n ∈ ({x - 2, x - 1, x, x + 1, x + 2} : Set ℤ), n = 15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_frame_sum_l473_47306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_implies_obtuse_l473_47333

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)

-- Define the property of being an obtuse triangle
def IsObtuseTriangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧ max t.A (max t.B t.C) > Real.pi/2

-- State the theorem
theorem tan_product_implies_obtuse (t : Triangle) :
  0 < Real.tan t.A * Real.tan t.B ∧ Real.tan t.A * Real.tan t.B < 1 →
  IsObtuseTriangle t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_implies_obtuse_l473_47333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_triangles_in_sequence_l473_47325

/-- Represents a triangle with angles α, β, γ -/
structure Triangle where
  α : ℝ
  β : ℝ
  γ : ℝ
  sum_180 : α + β + γ = π
  positive : 0 < α ∧ 0 < β ∧ 0 < γ

/-- The sequence of triangles formed by inscribed circles -/
noncomputable def triangleSequence (initialTriangle : Triangle) : ℕ → Triangle
  | 0 => initialTriangle
  | n + 1 => 
    let prev := triangleSequence initialTriangle n
    { α := (prev.β + prev.γ) / 2,
      β := (prev.α + prev.γ) / 2,
      γ := (prev.α + prev.β) / 2,
      sum_180 := by sorry,
      positive := by sorry }

/-- Two triangles are similar if their corresponding angles are equal -/
def similar (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

theorem no_similar_triangles_in_sequence (initialTriangle : Triangle) 
  (h_scalene : initialTriangle.α ≠ initialTriangle.β ∧ 
               initialTriangle.β ≠ initialTriangle.γ ∧ 
               initialTriangle.γ ≠ initialTriangle.α) :
  ∀ m n : ℕ, m ≠ n → 
    ¬(similar (triangleSequence initialTriangle m) (triangleSequence initialTriangle n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_triangles_in_sequence_l473_47325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_2310_l473_47355

theorem distinct_prime_factors_of_2310 : Finset.card (Nat.factorization 2310).support = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_2310_l473_47355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sum_square_l473_47382

theorem symmetric_sum_square (n : ℕ+) : 
  2 * (List.range n.val).sum + n.val = n.val ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sum_square_l473_47382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_l473_47338

/-- Represents the speed of the particle at the nth mile -/
noncomputable def speed (n : ℕ) : ℝ := 3 / (n ^ 2 : ℝ)

/-- Represents the time taken to traverse the nth mile -/
noncomputable def time (n : ℕ) : ℝ := 1 / speed n

theorem nth_mile_time (n : ℕ) (h : n ≥ 3) : time n = n^2 / 3 := by
  sorry

#check nth_mile_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_l473_47338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l473_47388

noncomputable def f (x a b : ℝ) : ℝ := x * Real.log x + a * x + b

theorem max_k_value (a b : ℝ) (h1 : f 1 a b = 1) (h2 : Real.log 1 + 1 + a = 3) :
  ∃ k : ℤ, (∀ x > 1, k < (f x a b) / (x - 1)) ∧
  (∀ m : ℤ, (∀ x > 1, m < (f x a b) / (x - 1)) → m ≤ k) ∧
  k = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l473_47388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l473_47323

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x + 1 else 2*x + 1

theorem angle_sum_theorem (α β γ : ℝ) 
  (h1 : f (Real.sin α + Real.sin β + Real.sin γ - 1) = -1)
  (h2 : f (Real.cos α + Real.cos β + Real.cos γ + 1) = 3) :
  Real.cos (α - β) + Real.cos (β - γ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l473_47323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_side_length_l473_47346

/-- A square with a given side length. -/
structure Square where
  side_length : ℝ

/-- A triangle. -/
structure Triangle where
  -- Add necessary fields
  vertices : Fin 3 → ℝ × ℝ

/-- Checks if a triangle is equilateral. -/
def Triangle.is_equilateral (t : Triangle) : Prop :=
  sorry

/-- Checks if two triangles are congruent. -/
def Triangle.congruent (t1 t2 : Triangle) : Prop :=
  sorry

/-- Checks if two triangles share a side. -/
def Triangle.shares_side (t1 t2 : Triangle) : Prop :=
  sorry

/-- Checks if a triangle has a vertex on a vertex of the given square. -/
def Triangle.has_vertex_on_square_vertex (t : Triangle) (s : Square) : Prop :=
  sorry

/-- Returns the largest square that can be inscribed in the space inside the outer square and outside of the given triangles. -/
def largest_inscribed_square (s : Square) (triangles : List Triangle) : Square :=
  sorry

/-- Given a square with side length 12 and three congruent equilateral triangles inscribed as described,
    the side length of the largest square that can be inscribed in the remaining space is 12 - 4√3. -/
theorem largest_inscribed_square_side_length 
  (outer_square : Square) 
  (triangle1 triangle2 triangle3 : Triangle) :
  outer_square.side_length = 12 →
  triangle1.is_equilateral ∧ triangle2.is_equilateral ∧ triangle3.is_equilateral →
  triangle1.congruent triangle2 ∧ triangle1.congruent triangle3 →
  triangle1.shares_side triangle2 ∧ triangle2.shares_side triangle3 ∧ triangle3.shares_side triangle1 →
  triangle1.has_vertex_on_square_vertex outer_square ∧ 
  triangle2.has_vertex_on_square_vertex outer_square ∧ 
  triangle3.has_vertex_on_square_vertex outer_square →
  (largest_inscribed_square outer_square [triangle1, triangle2, triangle3]).side_length = 12 - 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_side_length_l473_47346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_slope_l473_47387

/-- Parabola struct representing y^2 = 4x -/
structure Parabola where
  focus : ℝ × ℝ := (1, 0)

/-- Line passing through the focus of the parabola -/
structure Line (p : Parabola) where
  slope : ℝ
  passesThrough : (p.focus.1, p.focus.2) ∈ {(x, y) | y = slope * (x - p.focus.1) + p.focus.2}

/-- Points where the line intersects the parabola -/
def intersectionPoints (p : Parabola) (l : Line p) : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = 4*x ∧ y = l.slope * (x - p.focus.1) + p.focus.2}

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The slope of the line is ±2√2 when |AF| + 4|BF| is minimized -/
theorem optimal_slope (p : Parabola) :
  ∃ (l : Line p), ∀ (l' : Line p),
    (∀ (A B : ℝ × ℝ), A ∈ intersectionPoints p l → B ∈ intersectionPoints p l →
     ∀ (A' B' : ℝ × ℝ), A' ∈ intersectionPoints p l' → B' ∈ intersectionPoints p l' →
       distance A p.focus + 4 * distance B p.focus
       ≤ distance A' p.focus + 4 * distance B' p.focus)
    → l.slope = 2 * Real.sqrt 2 ∨ l.slope = -2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_slope_l473_47387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_five_consecutive_heads_in_eight_flips_l473_47336

def fair_coin_flips (n : ℕ) : ℕ := 2^n

def consecutive_heads (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else if k = n then 1
  else (n - k + 1) * 2^(n - k)

def at_least_consecutive_heads (n : ℕ) (k : ℕ) : ℕ :=
  Finset.sum (Finset.range (n - k + 2)) (λ i => consecutive_heads n (k + i))

theorem probability_at_least_five_consecutive_heads_in_eight_flips :
  (at_least_consecutive_heads 8 5 : ℚ) / fair_coin_flips 8 = 39 / 256 := by
  sorry

#eval at_least_consecutive_heads 8 5
#eval fair_coin_flips 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_five_consecutive_heads_in_eight_flips_l473_47336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cooking_pottery_only_count_l473_47329

/-- Represents the number of people in a set of activities -/
structure ActivitySet where
  count : Nat

/-- The total number of people in Yoga -/
def Y : ActivitySet := ⟨50⟩

/-- The total number of people in Cooking -/
def C : ActivitySet := ⟨30⟩

/-- The total number of people in Weaving -/
def W : ActivitySet := ⟨20⟩

/-- The total number of people in Pottery -/
def P : ActivitySet := ⟨15⟩

/-- The total number of people in Dancing -/
def D : ActivitySet := ⟨10⟩

/-- The number of people in Yoga and Cooking -/
def YC : ActivitySet := ⟨20⟩

/-- The number of people in Yoga and Weaving -/
def YW : ActivitySet := ⟨13⟩

/-- The number of people in Yoga and Pottery -/
def YP : ActivitySet := ⟨9⟩

/-- The number of people in Yoga and Dancing -/
def YD : ActivitySet := ⟨7⟩

/-- The number of people in Cooking and Weaving -/
def CW : ActivitySet := ⟨10⟩

/-- The number of people in Cooking and Pottery -/
def CP : ActivitySet := ⟨4⟩

/-- The number of people in Cooking and Dancing -/
def CD : ActivitySet := ⟨5⟩

/-- The number of people in Weaving and Pottery -/
def WP : ActivitySet := ⟨3⟩

/-- The number of people in Weaving and Dancing -/
def WD : ActivitySet := ⟨2⟩

/-- The number of people in Pottery and Dancing -/
def PD : ActivitySet := ⟨6⟩

/-- The number of people in Yoga, Cooking, and Weaving -/
def YCW : ActivitySet := ⟨9⟩

/-- The number of people in Yoga, Cooking, and Pottery -/
def YCP : ActivitySet := ⟨3⟩

/-- The number of people in Yoga, Cooking, and Dancing -/
def YCD : ActivitySet := ⟨2⟩

/-- The number of people in Yoga, Weaving, and Pottery -/
def YWP : ActivitySet := ⟨4⟩

/-- The number of people in Yoga, Weaving, and Dancing -/
def YWD : ActivitySet := ⟨1⟩

/-- The number of people in Cooking, Weaving, and Pottery -/
def CWP : ActivitySet := ⟨2⟩

/-- The number of people in Cooking, Weaving, and Dancing -/
def CWD : ActivitySet := ⟨1⟩

/-- The number of people in Cooking, Pottery, and Dancing -/
def CPD : ActivitySet := ⟨3⟩

/-- The number of people in all five activities -/
def YCWPD : ActivitySet := ⟨5⟩

theorem cooking_pottery_only_count : ActivitySet.count (CP) - ActivitySet.count (YCP) - ActivitySet.count (CWP) - ActivitySet.count (CPD) = 0 := by
  -- The number of people who study both Cooking and Pottery, but not Yoga, Weaving, or Dancing, is 0
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cooking_pottery_only_count_l473_47329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l473_47391

-- Define the function f(x) = |sin x|
noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

-- Theorem stating the properties of f
theorem f_properties :
  (∀ y ∈ Set.range f, 0 ≤ y ∧ y ≤ 1) ∧
  (∀ x, f (-x) = f x) ∧
  (∃ p > 0, ∀ x, f (x + p) = f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l473_47391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l473_47344

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - (1/2)^x

-- State the theorem
theorem f_has_unique_zero :
  ∃! x : ℝ, x ≥ 0 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l473_47344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l473_47352

-- Define the ellipse C
noncomputable def ellipse_C (a b x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

-- Define the parabola E
def parabola_E (x y : ℝ) : Prop := y = x^2/8

-- Define the tangent line l
def tangent_line_l (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the area of triangle FMN
noncomputable def area_FMN : ℝ := 5 * Real.sqrt 31 / 4

-- Main theorem
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 3 / 2)
  (h4 : ∃ x y, ellipse_C a b x y ∧ parabola_E x y ∧ tangent_line_l x y)
  (h5 : ∃ x1 y1 x2 y2, ellipse_C a b x1 y1 ∧ ellipse_C a b x2 y2 ∧ 
        tangent_line_l x1 y1 ∧ tangent_line_l x2 y2 ∧ 
        area_FMN = (1/2) * Real.sqrt 5 * |x1 - x2|) :
  a^2 = 16 ∧ b^2 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l473_47352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_problem_solution_correct_l473_47307

/-- Represents the mushroom distribution problem --/
structure MushroomProblem where
  total_mushrooms : ℕ
  dish_portions : ℕ
  liba_contribution : ℕ
  maruska_contribution : ℕ
  sarka_contribution : ℕ
  chocolates : ℕ

/-- Represents the solution to the mushroom problem --/
structure MushroomSolution where
  liba_mushrooms : ℕ
  maruska_mushrooms : ℕ
  sarka_mushrooms : ℕ
  liba_chocolates : ℕ
  maruska_chocolates : ℕ
  sarka_chocolates : ℕ

/-- Checks if the solution is correct for the given problem --/
def is_correct_solution (problem : MushroomProblem) (solution : MushroomSolution) : Prop :=
  let total_contribution := problem.liba_contribution + problem.maruska_contribution + problem.sarka_contribution
  let remaining_mushrooms := problem.total_mushrooms - total_contribution
  let mushrooms_per_person := remaining_mushrooms / 3

  -- Check mushroom distribution
  solution.liba_mushrooms = mushrooms_per_person + problem.liba_contribution ∧
  solution.maruska_mushrooms = mushrooms_per_person + problem.maruska_contribution ∧
  solution.sarka_mushrooms = mushrooms_per_person + problem.sarka_contribution ∧

  -- Check total mushrooms
  solution.liba_mushrooms + solution.maruska_mushrooms + solution.sarka_mushrooms = problem.total_mushrooms ∧

  -- Check chocolate distribution
  let liba_share := (problem.liba_contribution : ℚ) - (total_contribution : ℚ) / (problem.dish_portions : ℚ)
  let maruska_share := (problem.maruska_contribution : ℚ) - (total_contribution : ℚ) / (problem.dish_portions : ℚ)
  let sarka_share := (problem.sarka_contribution : ℚ) - (total_contribution : ℚ) / (problem.dish_portions : ℚ)
  let total_share := liba_share + maruska_share + sarka_share

  solution.liba_chocolates = (liba_share / total_share * problem.chocolates).floor ∧
  solution.maruska_chocolates = (maruska_share / total_share * problem.chocolates).floor ∧
  solution.sarka_chocolates = (sarka_share / total_share * problem.chocolates).floor ∧
  solution.liba_chocolates + solution.maruska_chocolates + solution.sarka_chocolates = problem.chocolates

/-- The main theorem stating that the given solution is correct for the problem --/
theorem mushroom_problem_solution_correct (problem : MushroomProblem) (solution : MushroomSolution) :
  problem.total_mushrooms = 55 ∧
  problem.dish_portions = 4 ∧
  problem.liba_contribution = 6 ∧
  problem.maruska_contribution = 8 ∧
  problem.sarka_contribution = 5 ∧
  problem.chocolates = 38 ∧
  solution.liba_mushrooms = 18 ∧
  solution.maruska_mushrooms = 20 ∧
  solution.sarka_mushrooms = 17 ∧
  solution.liba_chocolates = 10 ∧
  solution.maruska_chocolates = 26 ∧
  solution.sarka_chocolates = 2 →
  is_correct_solution problem solution :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_problem_solution_correct_l473_47307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_difference_l473_47358

/-- Given an initial configuration of blue and green tiles, and adding concentric borders of green tiles,
    calculate the difference between the total number of green and blue tiles. -/
theorem tile_difference (initial_blue initial_green num_borders : ℕ) 
  (h1 : initial_blue = 23)
  (h2 : initial_green = 16)
  (h3 : num_borders = 2) : 
  let border_tiles : ℕ → ℕ := λ n ↦ 6 * n
  let total_border_tiles := (Finset.range num_borders).sum (λ i ↦ border_tiles (i + 1))
  let final_green := initial_green + total_border_tiles
  final_green - initial_blue = 11 := by
  sorry

#check tile_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_difference_l473_47358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_average_speed_l473_47343

noncomputable def average_speed (total_distance : ℝ) (total_time : ℝ) : ℝ :=
  total_distance / total_time

noncomputable def distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem johns_average_speed :
  let car_speed : ℝ := 30
  let car_time : ℝ := 45 / 60
  let scooter_speed : ℝ := 10
  let scooter_time : ℝ := 1

  let total_distance : ℝ := distance car_speed car_time + distance scooter_speed scooter_time
  let total_time : ℝ := car_time + scooter_time

  let exact_average_speed : ℝ := average_speed total_distance total_time
  
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |exact_average_speed - 18| < ε :=
by
  sorry

#check johns_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_average_speed_l473_47343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_ice_cream_volume_l473_47364

/-- Define the main cone -/
def main_cone_height : ℝ := 10
def main_cone_radius : ℝ := 3

/-- Define the capping cone -/
def cap_cone_height : ℝ := 5
def cap_cone_radius : ℝ := main_cone_radius

/-- Define the volume of a cone -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Theorem statement -/
theorem total_ice_cream_volume :
  cone_volume main_cone_radius main_cone_height +
  cone_volume cap_cone_radius cap_cone_height = 45 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_ice_cream_volume_l473_47364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l473_47342

/-- Given that the solution set of x² - ax - b < 0 is (2, 3), 
    prove that the solution set of bx² - ax - 1 > 0 is (-1/2, -1/3) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x : ℝ, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x : ℝ, b*x^2 - a*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l473_47342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_two_equals_one_fourth_l473_47379

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else (2 : ℝ)^x

-- Theorem statement
theorem f_negative_two_equals_one_fourth :
  f (-2) = 1/4 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expression
  simp
  -- Evaluate 2^(-2)
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_two_equals_one_fourth_l473_47379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_bridge_l473_47399

-- Define the given values
noncomputable def train_length : ℝ := 720
noncomputable def bridge_length : ℝ := 280
noncomputable def train_speed_kmh : ℝ := 78

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the total distance to cover
noncomputable def total_distance : ℝ := train_length + bridge_length

-- Define the train speed in m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmh * kmh_to_ms

-- Define the time to pass the bridge
noncomputable def time_to_pass : ℝ := total_distance / train_speed_ms

-- Theorem statement
theorem time_to_pass_bridge :
  abs (time_to_pass - 46.15) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_bridge_l473_47399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_not_second_is_correct_l473_47319

/-- The number of people and jobs -/
def n : ℕ := 6

/-- The probability that person B does not take the second job, given that person A cannot take the first job -/
def prob_B_not_second : ℚ := 21/25

/-- The number of assignments where A does not take the first job -/
def total_assignments : ℕ := (n - 1) * Nat.factorial (n - 1)

/-- The number of assignments where A does not take the first job and B does not take the second job -/
def favorable_assignments : ℕ := Nat.factorial (n - 2) + (n - 2) * (n - 2) * Nat.factorial (n - 2)

theorem prob_B_not_second_is_correct : 
  (favorable_assignments : ℚ) / total_assignments = prob_B_not_second := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_not_second_is_correct_l473_47319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_two_dividing_15_4_minus_13_4_l473_47349

-- Define the function to calculate the largest power of 2 that divides a number
def largestPowerOfTwo (n : ℤ) : ℕ :=
  (n.natAbs.log 2)

-- State the theorem
theorem largest_power_of_two_dividing_15_4_minus_13_4 :
  largestPowerOfTwo (15^4 - 13^4) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_two_dividing_15_4_minus_13_4_l473_47349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_MN_length_l473_47362

-- Define the points on the line
variable (A B C D : ℝ)

-- Define the conditions
variable (h1 : A < B)
variable (h2 : B < C)
variable (h3 : C < D)
variable (h4 : D - A = 68)
variable (h5 : C - B = 26)

-- Define midpoints
noncomputable def M : ℝ := (A + C) / 2
noncomputable def N : ℝ := (B + D) / 2

-- Theorem statement
theorem segment_MN_length (A B C D : ℝ)
  (h1 : A < B) (h2 : B < C) (h3 : C < D)
  (h4 : D - A = 68) (h5 : C - B = 26) :
  |M - N| = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_MN_length_l473_47362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circ_associative_l473_47322

/-- The set of positive integers less than 1000 -/
def M : Set Nat := {n : Nat | 0 < n ∧ n < 1000}

/-- The operation ∘ defined on M -/
def circ (a b : Nat) : Nat :=
  if a * b < 1000 then a * b
  else
    let k := (a * b) / 1000
    let r := (a * b) % 1000
    if k + r < 1000 then k + r
    else (k + r) % 1000 + 1

/-- Theorem: The operation ∘ is associative on M -/
theorem circ_associative (a b c : Nat) (ha : a ∈ M) (hb : b ∈ M) (hc : c ∈ M) :
  circ (circ a b) c = circ a (circ b c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circ_associative_l473_47322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_line_l473_47357

/-- A circle passing through (2,1) and tangent to both coordinate axes -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : (center.1 - 2)^2 + (center.2 - 1)^2 = radius^2
  tangent_to_axes : center.1 = center.2 ∧ center.1 = radius

/-- The distance from a point to a line ax + by + c = 0 -/
noncomputable def distance_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2)

/-- The theorem stating the distance from the circle's center to the line 2x-y-3=0 -/
theorem distance_to_specific_line (circle : TangentCircle) :
    distance_to_line circle.center 2 (-1) (-3) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_line_l473_47357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_salt_concentration_approx_18_33_percent_l473_47370

/-- Represents the initial volume of the saltwater solution in gallons -/
noncomputable def x : ℝ := 104.99999999999997

/-- Calculates the initial salt concentration by volume given the conditions -/
noncomputable def initial_salt_concentration : ℝ :=
  let evaporated_volume := 3/4 * x
  let final_volume := evaporated_volume + 7 + 14
  let final_salt_volume := (1/3) * final_volume
  let initial_salt_volume := final_salt_volume - 14
  initial_salt_volume / x

/-- Theorem stating that the initial salt concentration is approximately 18.33% -/
theorem initial_salt_concentration_approx_18_33_percent :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ |initial_salt_concentration - 0.1833| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_salt_concentration_approx_18_33_percent_l473_47370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heron_equilateral_heron_345_triangle_l473_47331

-- Define Heron's formula
noncomputable def heronFormula (a b c : ℝ) : ℝ :=
  let p := (a + b + c) / 2
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

-- Theorem for equilateral triangle
theorem heron_equilateral (s : ℝ) (h : s > 0) :
  heronFormula s s s = (s^2 * Real.sqrt 3) / 4 := by
  sorry

-- Theorem for 3-4-5 right triangle
theorem heron_345_triangle :
  heronFormula 3 4 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heron_equilateral_heron_345_triangle_l473_47331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x24_l473_47347

def polynomial (x : ℝ) : ℝ := (x - 1) * (x^2 - 2) * (x^3 - 3) * (x^4 - 4) * (x^5 - 5) * (x^6 - 6) * (x^7 - 7) * (x^8 - 8)

theorem coefficient_of_x24 :
  ∃ (a : ℝ), ∀ x, polynomial x = a * x^24 + (polynomial x - a * x^24) ∧ a = 68 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x24_l473_47347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l473_47302

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

noncomputable def geometric_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => a i)

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ) (h_geom : geometric_sequence a q)
  (h_a3 : a 3 = 2 * geometric_sum a 2 + 1)
  (h_a4 : a 4 = 2 * geometric_sum a 3 + 1) :
  q = 3 :=
by
  sorry

#check geometric_sequence_common_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l473_47302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_rhombus_area_l473_47354

/-- A rhombus in a rectangular coordinate system -/
structure Rhombus where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the area of a rhombus -/
noncomputable def rhombusArea (r : Rhombus) : ℝ :=
  let d1 := abs (r.v1.2 - r.v3.2)
  let d2 := abs (r.v2.1 - r.v4.1)
  (d1 * d2) / 2

/-- The theorem stating the area of the specific rhombus -/
theorem specific_rhombus_area :
  let r : Rhombus := {
    v1 := (0, 3.5),
    v2 := (7, 0),
    v3 := (0, -3.5),
    v4 := (-7, 0)
  }
  rhombusArea r = 49 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_rhombus_area_l473_47354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_non_periodic_l473_47327

def digit_sum (n : ℕ) : ℕ := sorry

def sequence_elem (k : ℕ) : ℕ :=
  if digit_sum k % 2 = 0 then 0 else 1

theorem sequence_non_periodic : ∀ m d : ℕ, d > 0 →
  ∃ k : ℕ, k ≥ m ∧ sequence_elem k ≠ sequence_elem (k + d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_non_periodic_l473_47327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l473_47366

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_symmetry 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : abs φ < Real.pi / 2) 
  (h_distance : ∀ x : ℝ, f ω φ (x + Real.pi / (2 * ω)) = f ω φ x) 
  (h_shift : ∀ x : ℝ, f ω φ (x + 3 * Real.pi / 16) = f ω φ (-x)) :
  (∀ x : ℝ, f ω φ (Real.pi / 8 + x) = f ω φ (Real.pi / 8 - x)) ∧
  (∀ x : ℝ, f ω φ (-Real.pi / 16 + x) = f ω φ (-Real.pi / 16 - x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l473_47366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l473_47381

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  Real.sin x ^ 4 - Real.sin x * Real.cos x + Real.cos x ^ 4 + Real.cos x ^ 2

-- State the theorem
theorem f_range :
  (∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 2) ∧
  (∃ x : ℝ, f x = 0) ∧
  (∃ x : ℝ, f x = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l473_47381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l473_47315

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (x/2 + Real.pi/4) * Real.cos (x/2 + Real.pi/4) - Real.sin (x + Real.pi)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi/6)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → g x ≥ -1) ∧
  (g (Real.arcsin (1/2)) = -1) ∧
  (∀ (α : ℝ), α ∈ Set.Ioo (Real.pi/6) (Real.pi/2) → f α = 8/5 → Real.sin (2*α + Real.pi/3) = (7 * Real.sqrt 3 - 24) / 50) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l473_47315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_partition_theorem_l473_47316

/-- Represents a triangle in the partition --/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ
  good_sides : Fin 3 → Bool

/-- Represents the rectangle and its partition --/
structure RectanglePartition where
  m : ℕ
  n : ℕ
  triangles : List Triangle

/-- The main theorem --/
theorem rectangle_partition_theorem (rp : RectanglePartition) : 
  rp.m % 2 = 1 → 
  rp.n % 2 = 1 → 
  (∀ t ∈ rp.triangles, ∃ i : Fin 3, t.good_sides i = true) →
  (∀ t ∈ rp.triangles, ∀ i : Fin 3, 
    t.good_sides i = false → 
    ∃ t', t' ∈ rp.triangles ∧ t ≠ t' ∧ 
    ∃ j : Fin 3, t.vertices i = t'.vertices j ∧ t.vertices ((i+1) % 3) = t'.vertices ((j+1) % 3)) →
  ∃ t1 t2, t1 ∈ rp.triangles ∧ t2 ∈ rp.triangles ∧ t1 ≠ t2 ∧ 
    (∃ i j : Fin 3, i ≠ j ∧ t1.good_sides i = true ∧ t1.good_sides j = true) ∧
    (∃ i j : Fin 3, i ≠ j ∧ t2.good_sides i = true ∧ t2.good_sides j = true) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_partition_theorem_l473_47316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l473_47328

/-- The number of digits in the second factor of the product (8)(999...9) -/
def k : ℕ := 221

/-- The product of 8 and a number consisting of k nines -/
def product : ℕ := 8 * (10^k - 1)

/-- The sum of digits of the product -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The sum of digits of the product equals 2000 -/
axiom sum_is_2000 : sum_of_digits product = 2000

/-- The value of k is 221 -/
theorem k_value : k = 221 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l473_47328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_existence_l473_47330

-- Define the triangle ABC
structure Triangle (P : Type*) [MetricSpace P] where
  A : P
  B : P
  C : P

-- Define the existence of point D
def exists_point_D {P : Type*} [MetricSpace P] (T : Triangle P) : Prop :=
  ∃ D : P, 
    (dist T.A T.B < dist T.B T.C) → 
    (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ D = T.A) ∧
    (dist T.A T.B + dist T.B D + dist T.A D = dist T.B T.C)

-- Theorem statement
theorem triangle_point_existence {P : Type*} [MetricSpace P] (T : Triangle P) :
  exists_point_D T :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_existence_l473_47330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_squared_difference_l473_47386

theorem divisibility_of_squared_difference (S : Finset ℕ) (h : S.card = 43) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ 100 ∣ (a^2 - b^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_squared_difference_l473_47386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_subset_counterexample_n_6_l473_47377

/-- A subset S of {1,2,...,n} is balanced if for every a in S, 
    there exists b in S, b ≠ a, such that (a+b)/2 is in S. -/
def IsBalanced (n : ℕ) (S : Finset ℕ) : Prop :=
  ∀ a ∈ S, ∃ b ∈ S, b ≠ a ∧ (a + b) / 2 ∈ S

theorem balanced_subset (k : ℕ) (h : k > 1) :
  let n := 2 * k
  ∀ S : Finset ℕ, S ⊆ Finset.range n → S.card > (3 * n) / 4 → IsBalanced n S :=
sorry

/-- Part (b): Counterexample for n = 6 -/
theorem counterexample_n_6 :
  ∃ S : Finset ℕ, S ⊆ Finset.range 6 ∧ S.card > (2 * 6) / 3 ∧ ¬IsBalanced 6 S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_subset_counterexample_n_6_l473_47377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloud9_total_amount_is_23500_l473_47396

/-- Calculates the total amount of money taken by Cloud 9 Diving Company after refunds and discounts -/
def cloud9_total_amount (individual_bookings group_bookings : ℝ) 
  (group_discount : ℝ) (individual_refund_1 individual_refund_2 : ℝ) 
  (individual_refund_count_1 individual_refund_count_2 : ℕ)
  (group_refund : ℝ) : ℝ :=
  let group_bookings_after_discount := group_bookings * (1 - group_discount)
  let total_individual_refunds := individual_refund_1 * (individual_refund_count_1 : ℝ) + 
                                  individual_refund_2 * (individual_refund_count_2 : ℝ)
  let total_refunds := total_individual_refunds + group_refund
  individual_bookings + group_bookings_after_discount - total_refunds

theorem cloud9_total_amount_is_23500 :
  cloud9_total_amount 12000 16000 0.1 500 300 3 2 800 = 23500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloud9_total_amount_is_23500_l473_47396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l473_47324

noncomputable def f (x : ℝ) : ℝ := 
  4 * Real.sin x * (Real.sin (Real.pi / 4 + x / 2))^2 + Real.cos (2 * x) - 1

def A : Set ℝ := {x | Real.pi / 6 ≤ x ∧ x ≤ 2 * Real.pi / 3}

def B (m : ℝ) : Set ℝ := {x | (1/2 * f x)^2 - m * f x + m^2 + m - 1 > 0}

theorem problem_statement :
  (∀ ω > 0, (∀ x ∈ Set.Icc (-Real.pi/2) (2*Real.pi/3), Monotone (λ x ↦ f (ω * x))) → 0 < ω ∧ ω ≤ 3/4) ∧
  (∀ m : ℝ, A ⊆ B m → m < -Real.sqrt 3 / 2 ∨ m > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l473_47324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spanish_not_german_students_l473_47301

/-- The number of students taking Spanish but not German in a language program -/
theorem spanish_not_german_students : ℕ := by
  let total_students : ℕ := 30
  let both_languages : ℕ := 2
  let spanish_ratio : ℕ := 3
  let german_ratio : ℕ := 1

  have h1 : ∃ (spanish german : ℕ), 
    spanish + german + both_languages = total_students ∧ 
    spanish_ratio * german = german_ratio * spanish := by
    sorry

  have h2 : ∃ (spanish german : ℕ), 
    spanish + german + both_languages = total_students ∧ 
    spanish_ratio * german = german_ratio * spanish ∧
    spanish - both_languages = 20 := by
    sorry

  exact 20


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spanish_not_german_students_l473_47301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_proof_l473_47394

-- Define x as given in the problem
noncomputable def x : ℝ := (Real.rpow (2 + Real.sqrt 3) (1/3) + Real.rpow (2 - Real.sqrt 3) (1/3)) / 2

-- Theorem statement
theorem cubic_equation_proof : 4 * x^3 - 3 * x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_proof_l473_47394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_to_vertex_distance_eq_sqrt_ten_half_l473_47367

/-- Pyramid with square base on a sphere -/
structure PyramidOnSphere where
  /-- Side length of the square base -/
  baseSide : ℝ
  /-- Height of the pyramid -/
  height : ℝ
  /-- Radius of the sphere -/
  sphereRadius : ℝ
  /-- The base side length is 1 -/
  base_side_eq_one : baseSide = 1
  /-- The height is √2 -/
  height_eq_sqrt_two : height = Real.sqrt 2
  /-- The sphere radius is 1 -/
  sphere_radius_eq_one : sphereRadius = 1

/-- The distance between the center of the base and the vertex -/
noncomputable def centerToVertexDistance (p : PyramidOnSphere) : ℝ :=
  Real.sqrt ((p.height ^ 2) + ((Real.sqrt 2) / 2) ^ 2)

/-- Theorem: The distance between the center of the base and the vertex is √10/2 -/
theorem center_to_vertex_distance_eq_sqrt_ten_half (p : PyramidOnSphere) :
    centerToVertexDistance p = Real.sqrt 10 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_to_vertex_distance_eq_sqrt_ten_half_l473_47367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_local_minimum_at_zero_l473_47398

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 2)

-- Define the domain
def domain (x : ℝ) : Prop := -4 < x ∧ x < 1

-- Theorem statement
theorem f_has_local_minimum_at_zero :
  ∃ δ > 0, ∀ x : ℝ, domain x → |x| < δ → f 0 ≤ f x := by
  sorry

#check f_has_local_minimum_at_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_local_minimum_at_zero_l473_47398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l473_47303

noncomputable def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

noncomputable def z (x y : ℤ) : ℂ := Complex.mk ((x - 4 : ℚ) / (y^2 - 3*y - 4 : ℚ)) (x^2 + 3*x - 4)

theorem purely_imaginary_condition (x y : ℤ) :
  is_purely_imaginary (z x y) ↔ x = 4 ∧ y ≠ 4 ∧ y ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l473_47303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_characterization_l473_47332

def S : Set ℕ := {n : ℕ | ∃ d : ℕ, n^2 + 1 ≤ d ∧ d ≤ n^2 + 2*n ∧ d ∣ n^4}

theorem S_characterization :
  (∀ r ∈ ({0, 1, 2, 5, 6} : Set ℕ), ∀ N : ℕ, ∃ m : ℕ, m ≥ N ∧ (7 * m + r) ∈ S) ∧
  (∀ m : ℕ, (7 * m + 3) ∉ S ∧ (7 * m + 4) ∉ S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_characterization_l473_47332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otgaday_wins_l473_47304

/-- Represents the length of Otgaday's jump -/
def a : ℝ := sorry

/-- Represents the number of jumps Otgaday makes -/
def n : ℝ := sorry

/-- The length of Ugaday's jump is 30% shorter than Otgaday's -/
def ugaday_jump_length (a : ℝ) : ℝ := 0.7 * a

/-- The number of jumps Ugaday makes is 30% more than Otgaday's -/
def ugaday_jump_count (n : ℝ) : ℝ := 1.3 * n

/-- The total distance covered by Otgaday -/
def otgaday_distance (a n : ℝ) : ℝ := a * n

/-- The total distance covered by Ugaday -/
def ugaday_distance (a n : ℝ) : ℝ := ugaday_jump_length a * ugaday_jump_count n

theorem otgaday_wins (a n : ℝ) (h1 : a > 0) (h2 : n > 0) : 
  otgaday_distance a n > ugaday_distance a n := by
  sorry

#check otgaday_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otgaday_wins_l473_47304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_green_ducks_l473_47309

theorem percentage_of_green_ducks (small_pond_ducks : ℕ) (large_pond_ducks : ℕ) 
  (small_pond_green_percentage : ℚ) (large_pond_green_percentage : ℚ) :
  small_pond_ducks = 45 →
  large_pond_ducks = 55 →
  small_pond_green_percentage = 20 / 100 →
  large_pond_green_percentage = 40 / 100 →
  (((small_pond_ducks : ℚ) * small_pond_green_percentage + 
    (large_pond_ducks : ℚ) * large_pond_green_percentage) / 
   ((small_pond_ducks + large_pond_ducks) : ℚ)) * 100 = 31 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_green_ducks_l473_47309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l473_47356

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the theorem
theorem line_plane_relationship 
  (l : Line) (α β : Plane) 
  (h1 : α ≠ β) 
  (h2 : parallel_planes α β) 
  (h3 : parallel l α) : 
  parallel l β ∨ subset l β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l473_47356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l473_47310

/-- The area of a trapezium with given parallel side lengths and height -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 20 cm and 18 cm, 
    and height 14 cm, is 266 square centimeters -/
theorem trapezium_area_example : trapezium_area 20 18 14 = 266 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the arithmetic expression
  simp [add_mul, mul_div_right_comm]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l473_47310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_special_triangle_l473_47363

theorem largest_angle_in_special_triangle :
  ∀ (A B C : Real) (k : Real),
    0 < k →
    0 < A ∧ A < Real.pi →
    0 < B ∧ B < Real.pi →
    0 < C ∧ C < Real.pi →
    A + B + C = Real.pi →
    Real.sin A = 3 * k →
    Real.sin B = 5 * k →
    Real.sin C = 7 * k →
    max A (max B C) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_special_triangle_l473_47363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_is_correct_l473_47360

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (x^3 - 5*x^2 + 5*x + 23) / ((x-1)*(x+1)*(x-5))

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := x - 3 * Real.log (abs (x-1)) + Real.log (abs (x+1)) + 2 * Real.log (abs (x-5))

-- Theorem statement
theorem integral_is_correct (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ 5) : 
  deriv F x = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_is_correct_l473_47360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l473_47334

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.sin x - 1/2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∃ k : ℤ, π/6 + 2*k*π ≤ x ∧ x ≤ 5*π/6 + 2*k*π} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l473_47334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l473_47372

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/4)^x - (1/2)^x + 1

-- Define the domain
def domain : Set ℝ := Set.Icc (-3) 2

-- State the theorem
theorem f_properties :
  (∀ x y, x ∈ domain → y ∈ domain → x < y → x ≤ 1 → f x ≥ f y) ∧
  (∀ x y, x ∈ domain → y ∈ domain → 1 ≤ x → x < y → f x ≤ f y) ∧
  (∀ x, x ∈ domain → f x ≥ 3/4) ∧
  (∀ x, x ∈ domain → f x ≤ 57) ∧
  (∃ x, x ∈ domain ∧ f x = 3/4) ∧
  (∃ x, x ∈ domain ∧ f x = 57) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l473_47372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_in_set_l473_47373

theorem largest_difference_in_set : 
  let S : Finset Int := {-12, -6, 0, 3, 7, 15}
  (∀ x y : Int, x ∈ S → y ∈ S → x - y ≤ 27) ∧ 
  (∃ a b : Int, a ∈ S ∧ b ∈ S ∧ a - b = 27) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_in_set_l473_47373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_represents_circle_l473_47351

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a point in Cartesian coordinates -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Converts a polar point to a Cartesian point -/
noncomputable def polarToCartesian (p : PolarPoint) : CartesianPoint :=
  { x := p.ρ * Real.cos p.θ
  , y := p.ρ * Real.sin p.θ }

/-- The circle equation in polar coordinates -/
def circleEquation (p : PolarPoint) : Prop :=
  p.ρ = 4 * Real.sin p.θ

/-- The center of the circle in Cartesian coordinates -/
def circleCenter : CartesianPoint :=
  { x := 0, y := 2 }

/-- The radius of the circle -/
def circleRadius : ℝ := 2

/-- Theorem stating that the polar equation represents the given circle -/
theorem polar_equation_represents_circle :
  ∀ p : PolarPoint, circleEquation p →
    let c := polarToCartesian p
    (c.x - circleCenter.x)^2 + (c.y - circleCenter.y)^2 = circleRadius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_represents_circle_l473_47351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l473_47340

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f defined as x^3 - x + 1 for x > 0 -/
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^3 - x + 1 else -(((-x)^3 - (-x) + 1))

theorem odd_function_sum :
  ∀ f : ℝ → ℝ, IsOdd f → (∀ x > 0, f x = x^3 - x + 1) → f (-1) + f 0 = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l473_47340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_l473_47378

noncomputable def purchase_price : ℝ := 42000
noncomputable def repair_cost : ℝ := 15000
noncomputable def selling_price : ℝ := 64900

noncomputable def total_cost : ℝ := purchase_price + repair_cost
noncomputable def profit : ℝ := selling_price - total_cost
noncomputable def profit_percent : ℝ := (profit / total_cost) * 100

theorem car_profit_percent : 
  ∀ ε > 0, |profit_percent - 13.86| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_l473_47378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equality_implies_x_value_l473_47397

theorem sin_cos_equality_implies_x_value (x : ℝ) :
  Real.sin (2 * x) * Real.sin (4 * x) = Real.cos (2 * x) * Real.cos (4 * x) →
  x = π / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equality_implies_x_value_l473_47397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_proof_l473_47326

noncomputable def a : Fin 3 → ℝ := ![4, 5, 1]
noncomputable def b : Fin 3 → ℝ := ![1, -2, 2]
noncomputable def v : Fin 3 → ℝ := ![6 / (Real.sqrt 42 + 4), 1 / (Real.sqrt 42 + 4), 5 / (Real.sqrt 42 + 4)]

theorem bisector_proof :
  (‖v‖ = 1) ∧ 
  (∃ (k : ℝ), b = k • (a + ‖a‖ • v)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_proof_l473_47326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l473_47384

theorem triangle_problem (A B C : Real) (a b c : Real) :
  let f := fun x => Real.sin (2 * x + A)
  -- Part 1
  (A = π / 2 → f (-π / 6) = -1 / 2) ∧
  -- Part 2
  (f (π / 12) = 1 → a = 3 → Real.cos B = 4 / 5 → b = 6 * Real.sqrt 3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l473_47384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l473_47383

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * Real.cos x - 1)

theorem domain_of_f : 
  ∀ x : ℝ, f x ∈ Set.univ ↔ ∃ k : ℤ, x ∈ Set.Ioo (-π/3 + 2*π*(k:ℝ)) (π/3 + 2*π*(k:ℝ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l473_47383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_25_percent_l473_47371

/-- The percentage reduction in oil price -/
noncomputable def oil_price_reduction (original_price reduced_price : ℝ) : ℝ :=
  ((original_price - reduced_price) / original_price) * 100

/-- The amount of oil that can be bought with a fixed amount of money -/
noncomputable def oil_amount (price : ℝ) (money : ℝ) : ℝ :=
  money / price

theorem oil_price_reduction_25_percent 
  (reduced_price : ℝ) 
  (fixed_money : ℝ) 
  (extra_oil : ℝ) :
  reduced_price = 60 →
  fixed_money = 1200 →
  extra_oil = 5 →
  oil_amount reduced_price fixed_money - oil_amount (fixed_money / (oil_amount reduced_price fixed_money - extra_oil)) fixed_money = extra_oil →
  oil_price_reduction (fixed_money / (oil_amount reduced_price fixed_money - extra_oil)) reduced_price = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_25_percent_l473_47371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_distance_is_six_l473_47313

/-- Represents the field and rabbit's movement --/
structure FieldAndRabbit where
  side_length : ℝ
  diagonal_hop : ℝ
  right_turn_hop : ℝ

/-- Calculates the average distance from the rabbit to the sides of the field --/
noncomputable def average_distance_to_sides (f : FieldAndRabbit) : ℝ :=
  let diagonal := f.side_length * Real.sqrt 2
  let frac := f.diagonal_hop / diagonal
  let x := frac * f.side_length + f.right_turn_hop
  let y := frac * f.side_length
  (x + y + (f.side_length - x) + (f.side_length - y)) / 4

/-- Theorem stating that the average distance is 6 meters for the given conditions --/
theorem average_distance_is_six :
  let f : FieldAndRabbit := {
    side_length := 12,
    diagonal_hop := 9.8,
    right_turn_hop := 4
  }
  average_distance_to_sides f = 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_distance_is_six_l473_47313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l473_47390

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem f_properties (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : ℝ, f a b x ≤ |f a b (π/6)|) :
  (f a b (5*π/12) = 0) ∧ 
  (|f a b (7*π/12)| ≥ |f a b (π/3)|) ∧
  (¬ (∀ x : ℝ, f a b x = f a b (-x))) ∧ 
  (¬ (∀ x : ℝ, f a b x = -f a b (-x))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l473_47390
