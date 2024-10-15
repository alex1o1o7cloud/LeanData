import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2991_299159

theorem equation_solution (m n : ℝ) : 21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2991_299159


namespace NUMINAMATH_CALUDE_folded_paper_area_ratio_l2991_299103

/-- Represents a square piece of paper -/
structure Paper where
  side : ℝ
  area : ℝ
  area_eq : area = side ^ 2

/-- Represents the folded paper -/
structure FoldedPaper where
  original : Paper
  new_area : ℝ

/-- Theorem stating the ratio of areas after folding -/
theorem folded_paper_area_ratio (p : Paper) (fp : FoldedPaper) 
  (h_fp : fp.original = p) 
  (h_fold : fp.new_area = (7 / 8) * p.area) : 
  fp.new_area / p.area = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_area_ratio_l2991_299103


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l2991_299194

/-- Proves that the ratio of sheep to horses is 4:7 given the farm conditions --/
theorem stewart_farm_ratio (sheep : ℕ) (horse_food_per_day : ℕ) (total_horse_food : ℕ) :
  sheep = 32 →
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  (sheep : ℚ) / (total_horse_food / horse_food_per_day : ℚ) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l2991_299194


namespace NUMINAMATH_CALUDE_exactly_two_correct_propositions_l2991_299183

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the relations
def parallel (a b : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def intersect (p1 p2 : Plane) (l : Line) : Prop := sorry

-- State the theorem
theorem exactly_two_correct_propositions 
  (l m n : Line) (α β γ : Plane) : 
  (∃! (correct : List Prop), 
    correct.length = 2 ∧ 
    correct ⊆ [
      (parallel m l ∧ perpendicular m α → perpendicular l α),
      (parallel m l ∧ parallel m α → parallel l α),
      (intersect α β l ∧ intersect β γ m ∧ intersect γ α n → 
        parallel l m ∧ parallel m n ∧ parallel l n),
      (intersect α β m ∧ intersect β γ l ∧ intersect α γ n ∧ 
        parallel n β → parallel m l)
    ] ∧
    (∀ p ∈ correct, p)) := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_correct_propositions_l2991_299183


namespace NUMINAMATH_CALUDE_difference_of_squares_divisible_by_18_l2991_299178

theorem difference_of_squares_divisible_by_18 (a b : ℤ) 
  (ha : Odd a) (hb : Odd b) : 
  ∃ k : ℤ, (3*a + 2)^2 - (3*b + 2)^2 = 18 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_divisible_by_18_l2991_299178


namespace NUMINAMATH_CALUDE_alpha_computation_l2991_299171

theorem alpha_computation (α β : ℂ) :
  (α + β).re > 0 →
  (Complex.I * (α - 3 * β)).re > 0 →
  β = 4 + 3 * Complex.I →
  α = 3 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_alpha_computation_l2991_299171


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l2991_299104

theorem product_sum_of_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 149) 
  (h2 : a + b + c = 17) : 
  a * b + b * c + c * a = 70 := by
sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l2991_299104


namespace NUMINAMATH_CALUDE_invisible_dots_count_l2991_299155

/-- The number of dots on a standard six-sided die -/
def standardDieDots : ℕ := 21

/-- The total number of dots on four standard six-sided dice -/
def totalDots : ℕ := 4 * standardDieDots

/-- The list of visible face values on the stacked dice -/
def visibleFaces : List ℕ := [1, 1, 2, 3, 4, 4, 5, 6]

/-- The sum of the visible face values -/
def visibleDotsSum : ℕ := visibleFaces.sum

/-- Theorem: The number of dots not visible on four stacked standard six-sided dice -/
theorem invisible_dots_count : totalDots - visibleDotsSum = 58 := by
  sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l2991_299155


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2991_299129

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 3)^2) - Real.sqrt ((x - 7)^2 + (y + 3)^2) = 4

/-- The positive slope of an asymptote of the hyperbola -/
def positive_asymptote_slope : ℝ := 0.75

/-- Theorem stating that the positive slope of an asymptote of the given hyperbola is 0.75 -/
theorem hyperbola_asymptote_slope :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ positive_asymptote_slope = 0.75 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2991_299129


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l2991_299147

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b/a = (1-cos B)/cos A, then A = C, implying it's an isosceles triangle. -/
theorem isosceles_triangle_condition
  (A B C : ℝ) (a b c : ℝ)
  (triangle_sum : A + B + C = π)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (sine_law : b / a = Real.sin B / Real.sin A)
  (hypothesis : b / a = (1 - Real.cos B) / Real.cos A) :
  A = C :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l2991_299147


namespace NUMINAMATH_CALUDE_triangle_area_l2991_299193

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define a median
def isMedian (t : Triangle) (M : ℝ × ℝ) (X Y Z : ℝ × ℝ) : Prop :=
  M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2) ∨ 
  M = ((Y.1 + Z.1) / 2, (Y.2 + Z.2) / 2) ∨ 
  M = ((Z.1 + X.1) / 2, (Z.2 + X.2) / 2)

-- Define the intersection point O
def intersectionPoint (XM YN : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the right angle intersection
def isRightAngle (XM YN : ℝ × ℝ) (O : ℝ × ℝ) : Prop := sorry

-- Define the length of a line segment
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_area (t : Triangle) (M N O : ℝ × ℝ) :
  isMedian t M t.X t.Y t.Z →
  isMedian t N t.X t.Y t.Z →
  O = intersectionPoint M N →
  isRightAngle M N O →
  length t.X M = 18 →
  length t.Y N = 24 →
  area t = 288 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2991_299193


namespace NUMINAMATH_CALUDE_words_per_page_l2991_299112

theorem words_per_page (total_pages : ℕ) (max_words_per_page : ℕ) (total_words_mod : ℕ) :
  total_pages = 150 →
  max_words_per_page = 120 →
  total_words_mod = 270 →
  ∃ (words_per_page : ℕ),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 221 = total_words_mod % 221 ∧
    words_per_page = 107 :=
by sorry

end NUMINAMATH_CALUDE_words_per_page_l2991_299112


namespace NUMINAMATH_CALUDE_min_value_expression_l2991_299141

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / (2 * a) + 1 / b) ≥ Real.sqrt 2 + 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2991_299141


namespace NUMINAMATH_CALUDE_turnip_bag_options_l2991_299179

def bag_weights : List Nat := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (turnip_weight : Nat) : Prop :=
  turnip_weight ∈ bag_weights ∧
  ∃ (onion_weights carrots_weights : List Nat),
    onion_weights ++ carrots_weights ++ [turnip_weight] = bag_weights ∧
    onion_weights.sum * 2 = carrots_weights.sum

theorem turnip_bag_options :
  ∀ w ∈ bag_weights, is_valid_turnip_weight w ↔ w = 13 ∨ w = 16 := by sorry

end NUMINAMATH_CALUDE_turnip_bag_options_l2991_299179


namespace NUMINAMATH_CALUDE_inequality_proof_l2991_299120

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  1 / (8 * a^2 - 18 * a + 11) + 1 / (8 * b^2 - 18 * b + 11) + 1 / (8 * c^2 - 18 * c + 11) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2991_299120


namespace NUMINAMATH_CALUDE_women_married_fraction_l2991_299105

theorem women_married_fraction (total : ℕ) (h_total_pos : total > 0) :
  let women := (76 : ℚ) / 100 * total
  let married := (60 : ℚ) / 100 * total
  let men := total - women
  let single_men := (2 : ℚ) / 3 * men
  let married_men := men - single_men
  let married_women := married - married_men
  married_women / women = 13 / 19 := by
sorry

end NUMINAMATH_CALUDE_women_married_fraction_l2991_299105


namespace NUMINAMATH_CALUDE_square_1225_identity_l2991_299123

theorem square_1225_identity (x : ℤ) (h : x^2 = 1225) : (x + 2) * (x - 2) = 1221 := by
  sorry

end NUMINAMATH_CALUDE_square_1225_identity_l2991_299123


namespace NUMINAMATH_CALUDE_sum_of_squared_sums_of_roots_l2991_299182

theorem sum_of_squared_sums_of_roots (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_sums_of_roots_l2991_299182


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_prime_square_difference_l2991_299158

theorem largest_prime_divisor_of_prime_square_difference (m n : ℕ) 
  (hm : Prime m) (hn : Prime n) (hmn : m ≠ n) :
  (∃ (p : ℕ) (hp : Prime p), p ∣ (m^2 - n^2)) ∧
  (∀ (q : ℕ) (hq : Prime q), q ∣ (m^2 - n^2) → q ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_prime_square_difference_l2991_299158


namespace NUMINAMATH_CALUDE_g_100_value_l2991_299139

/-- A function satisfying the given property for all positive real numbers -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → x * g y - y * g x = g (x / y) + x - y

/-- The main theorem stating the value of g(100) -/
theorem g_100_value (g : ℝ → ℝ) (h : SatisfiesProperty g) : g 100 = -99 / 2 := by
  sorry


end NUMINAMATH_CALUDE_g_100_value_l2991_299139


namespace NUMINAMATH_CALUDE_fraction_increase_l2991_299156

theorem fraction_increase (a b : ℝ) (h : 3 * a - 4 * b ≠ 0) :
  (2 * (3 * a) * (3 * b)) / (3 * (3 * a) - 4 * (3 * b)) = 3 * ((2 * a * b) / (3 * a - 4 * b)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_increase_l2991_299156


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l2991_299161

theorem dvd_rental_cost (num_dvds : ℕ) (cost_per_dvd : ℚ) : 
  num_dvds = 4 → cost_per_dvd = 6/5 → num_dvds * cost_per_dvd = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l2991_299161


namespace NUMINAMATH_CALUDE_blue_twice_prob_octahedron_l2991_299121

/-- A regular octahedron with colored faces -/
structure ColoredOctahedron where
  blue_faces : ℕ
  red_faces : ℕ
  total_faces : ℕ
  is_regular : Prop
  face_sum : blue_faces + red_faces = total_faces

/-- The probability of an event occurring twice in independent trials -/
def independent_event_twice_prob (single_prob : ℚ) : ℚ :=
  single_prob * single_prob

/-- The probability of rolling a blue face twice in succession on a colored octahedron -/
def blue_twice_prob (o : ColoredOctahedron) : ℚ :=
  independent_event_twice_prob ((o.blue_faces : ℚ) / (o.total_faces : ℚ))

theorem blue_twice_prob_octahedron :
  ∃ (o : ColoredOctahedron),
    o.blue_faces = 5 ∧
    o.red_faces = 3 ∧
    o.total_faces = 8 ∧
    o.is_regular ∧
    blue_twice_prob o = 25 / 64 := by
  sorry

end NUMINAMATH_CALUDE_blue_twice_prob_octahedron_l2991_299121


namespace NUMINAMATH_CALUDE_power_of_square_l2991_299134

theorem power_of_square (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l2991_299134


namespace NUMINAMATH_CALUDE_problem_statement_l2991_299166

theorem problem_statement (n : ℝ) (h : n + 1/n = 5) : n^2 + 1/n^2 + 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2991_299166


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2991_299188

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : InverselyProportional x₁ y₁)
  (h2 : InverselyProportional x₂ y₂)
  (h3 : x₁ = 40)
  (h4 : y₁ = 8)
  (h5 : y₂ = 10) :
  x₂ = 32 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2991_299188


namespace NUMINAMATH_CALUDE_slope_of_solutions_l2991_299132

/-- The equation that defines the relationship between x and y -/
def equation (x y : ℝ) : Prop := (4 / x) + (6 / y) = 0

/-- Theorem stating that the slope between any two distinct solutions of the equation is -3/2 -/
theorem slope_of_solutions (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : equation x₁ y₁) (h₂ : equation x₂ y₂) (h_dist : (x₁, y₁) ≠ (x₂, y₂)) :
  (y₂ - y₁) / (x₂ - x₁) = -3/2 := by
sorry

end NUMINAMATH_CALUDE_slope_of_solutions_l2991_299132


namespace NUMINAMATH_CALUDE_height_on_hypotenuse_l2991_299190

theorem height_on_hypotenuse (a b h : ℝ) (hyp : ℝ) : 
  a = 2 → b = 3 → a^2 + b^2 = hyp^2 → (a * b) / 2 = (hyp * h) / 2 → h = (6 * Real.sqrt 13) / 13 := by
  sorry

end NUMINAMATH_CALUDE_height_on_hypotenuse_l2991_299190


namespace NUMINAMATH_CALUDE_circle_sum_radii_geq_rectangle_sides_l2991_299115

/-- Given a rectangle ABCD with sides a and b, and two circles k₁ and k₂ where:
    - k₁ passes through A and B and is tangent to CD
    - k₂ passes through A and D and is tangent to BC
    - r₁ and r₂ are the radii of k₁ and k₂ respectively
    Prove that r₁ + r₂ ≥ 5/8 * (a + b) -/
theorem circle_sum_radii_geq_rectangle_sides 
  (a b r₁ r₂ : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (hr₁ : r₁ = (a^2 + 4*b^2) / (8*b))
  (hr₂ : r₂ = (b^2 + 4*a^2) / (8*a)) :
  r₁ + r₂ ≥ 5/8 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_radii_geq_rectangle_sides_l2991_299115


namespace NUMINAMATH_CALUDE_f_pi_third_eq_half_l2991_299143

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sin x
  else if 1 ≤ x ∧ x ≤ Real.sqrt 2 then Real.cos x
  else Real.tan x

-- Theorem statement
theorem f_pi_third_eq_half : f (Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_pi_third_eq_half_l2991_299143


namespace NUMINAMATH_CALUDE_hundredth_power_mod_125_l2991_299165

theorem hundredth_power_mod_125 (n : ℤ) : (n^100 : ℤ) % 125 = 0 ∨ (n^100 : ℤ) % 125 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_power_mod_125_l2991_299165


namespace NUMINAMATH_CALUDE_square_area_ratio_l2991_299128

theorem square_area_ratio (x : ℝ) (hx : x > 0) : (x^2) / ((3*x)^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2991_299128


namespace NUMINAMATH_CALUDE_card_game_guarantee_l2991_299149

/-- Represents a cell on the 4x9 board --/
structure Cell :=
  (row : Fin 4)
  (col : Fin 9)

/-- Represents a pair of cells --/
structure CellPair :=
  (cell1 : Cell)
  (cell2 : Cell)

/-- Represents the state of the board --/
def Board := Fin 4 → Fin 9 → Bool

/-- A valid pairing of cells --/
def ValidPairing (board : Board) (pairs : List CellPair) : Prop :=
  ∀ p ∈ pairs,
    (board p.cell1.row p.cell1.col ≠ board p.cell2.row p.cell2.col) ∧
    ((p.cell1.row = p.cell2.row) ∨ (p.cell1.col = p.cell2.col))

/-- The main theorem --/
theorem card_game_guarantee (board : Board) :
  (∃ black_count : ℕ, black_count = 18 ∧ 
    (∀ r : Fin 4, ∀ c : Fin 9, (board r c = true) → black_count = black_count - 1)) →
  ∃ pairs : List CellPair, ValidPairing board pairs ∧ pairs.length ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_card_game_guarantee_l2991_299149


namespace NUMINAMATH_CALUDE_correct_calculation_l2991_299124

theorem correct_calculation (x y : ℝ) : 3 * x^4 * y / (x^2 * y) = 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2991_299124


namespace NUMINAMATH_CALUDE_path_area_theorem_l2991_299152

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Theorem: The area of a 2.5m wide path around a 60m x 55m field is 600 sq m -/
theorem path_area_theorem :
  path_area 60 55 2.5 = 600 := by sorry

end NUMINAMATH_CALUDE_path_area_theorem_l2991_299152


namespace NUMINAMATH_CALUDE_seedling_probability_value_l2991_299199

/-- The germination rate of seeds in a batch -/
def germination_rate : ℝ := 0.9

/-- The survival rate of seedlings after germination -/
def survival_rate : ℝ := 0.8

/-- The probability that a randomly selected seed will grow into a seedling -/
def seedling_probability : ℝ := germination_rate * survival_rate

/-- Theorem stating that the probability of a randomly selected seed growing into a seedling is 0.72 -/
theorem seedling_probability_value : seedling_probability = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_seedling_probability_value_l2991_299199


namespace NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l2991_299107

theorem exists_integer_sqrt_8m : ∃ m : ℕ+, ∃ k : ℕ, (8 * m.val : ℕ) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l2991_299107


namespace NUMINAMATH_CALUDE_ellipse_sum_l2991_299173

theorem ellipse_sum (h k a b : ℝ) : 
  (h = 3) → 
  (k = -5) → 
  (a = 7) → 
  (b = 4) → 
  h + k + a + b = 9 := by
sorry

end NUMINAMATH_CALUDE_ellipse_sum_l2991_299173


namespace NUMINAMATH_CALUDE_estimate_eight_minus_two_sqrt_seven_l2991_299140

theorem estimate_eight_minus_two_sqrt_seven :
  2 < 8 - 2 * Real.sqrt 7 ∧ 8 - 2 * Real.sqrt 7 < 3 := by
  sorry

end NUMINAMATH_CALUDE_estimate_eight_minus_two_sqrt_seven_l2991_299140


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2991_299133

theorem complex_fraction_simplification : ∀ (x : ℝ),
  x = (3 * (Real.sqrt 3 + Real.sqrt 7)) / (4 * Real.sqrt (3 + Real.sqrt 2)) →
  x ≠ 3 * Real.sqrt 7 / 4 ∧
  x ≠ 9 * Real.sqrt 2 / 16 ∧
  x ≠ 3 * Real.sqrt 3 / 4 ∧
  x ≠ 15 / 8 ∧
  x ≠ 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2991_299133


namespace NUMINAMATH_CALUDE_contrapositive_squared_sum_l2991_299172

theorem contrapositive_squared_sum (x y : ℝ) : x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_squared_sum_l2991_299172


namespace NUMINAMATH_CALUDE_coal_consumption_factory_coal_consumption_l2991_299157

/-- Given a factory that burns coal at a constant daily rate, calculate the total coal burned over a longer period. -/
theorem coal_consumption (initial_coal : ℝ) (initial_days : ℝ) (total_days : ℝ) :
  initial_coal > 0 → initial_days > 0 → total_days > initial_days →
  (initial_coal / initial_days) * total_days = 
    initial_coal * (total_days / initial_days) := by
  sorry

/-- Specific instance of coal consumption calculation -/
theorem factory_coal_consumption :
  let initial_coal : ℝ := 37.5
  let initial_days : ℝ := 5
  let total_days : ℝ := 13
  (initial_coal / initial_days) * total_days = 97.5 := by
  sorry

end NUMINAMATH_CALUDE_coal_consumption_factory_coal_consumption_l2991_299157


namespace NUMINAMATH_CALUDE_possible_pen_counts_l2991_299175

def total_money : ℕ := 11
def pen_cost : ℕ := 3
def notebook_cost : ℕ := 1

def valid_pen_count (x : ℕ) : Prop :=
  ∃ y : ℕ, x * pen_cost + y * notebook_cost = total_money

theorem possible_pen_counts : 
  (valid_pen_count 1 ∧ valid_pen_count 2 ∧ valid_pen_count 3) ∧
  (∀ x : ℕ, valid_pen_count x → x = 1 ∨ x = 2 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_possible_pen_counts_l2991_299175


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2991_299108

theorem floor_ceil_sum : ⌊(0.99 : ℝ)⌋ + ⌈(2.99 : ℝ)⌉ + 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2991_299108


namespace NUMINAMATH_CALUDE_parabola_point_order_l2991_299154

/-- Parabola function -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem parabola_point_order (a b c d : ℝ) : 
  f a = 2 → f b = 6 → f c = d → d < 1 → a < 0 → b > 0 → a < c ∧ c < b :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_order_l2991_299154


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2991_299185

theorem trigonometric_equation_solution (k : ℤ) : 
  let x : ℝ := -Real.arccos (-4/5) + (2 * k + 1 : ℝ) * Real.pi
  let y : ℝ := -1/2
  3 * Real.sin x - 4 * Real.cos x = 4 * y^2 + 4 * y + 6 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2991_299185


namespace NUMINAMATH_CALUDE_min_sum_of_dimensions_l2991_299109

def is_valid_box (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1729

theorem min_sum_of_dimensions :
  ∃ (a b c : ℕ), is_valid_box a b c ∧
  ∀ (x y z : ℕ), is_valid_box x y z → a + b + c ≤ x + y + z ∧
  a + b + c = 39 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_dimensions_l2991_299109


namespace NUMINAMATH_CALUDE_sum_equals_four_l2991_299135

theorem sum_equals_four (x y : ℝ) (h : |x - 3| + |y + 2| = 0) : x + y + 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_four_l2991_299135


namespace NUMINAMATH_CALUDE_nonnegative_sum_one_inequality_l2991_299170

theorem nonnegative_sum_one_inequality (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum_one : x + y + z = 1) : 
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_sum_one_inequality_l2991_299170


namespace NUMINAMATH_CALUDE_bounded_by_one_l2991_299125

/-- A function from integers to reals satisfying certain properties -/
def IntToRealFunction (f : ℤ → ℝ) : Prop :=
  (∀ n, f n ≥ 0) ∧ 
  (∀ m n, f (m * n) = f m * f n) ∧ 
  (∀ m n, f (m + n) ≤ max (f m) (f n))

/-- Theorem stating that any function satisfying IntToRealFunction is bounded above by 1 -/
theorem bounded_by_one (f : ℤ → ℝ) (hf : IntToRealFunction f) : 
  ∀ n, f n ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_bounded_by_one_l2991_299125


namespace NUMINAMATH_CALUDE_new_average_age_after_move_l2991_299176

theorem new_average_age_after_move (room_a_initial_count : ℕ)
                                   (room_a_initial_avg : ℚ)
                                   (room_b_initial_count : ℕ)
                                   (room_b_initial_avg : ℚ)
                                   (moving_person_age : ℕ) :
  room_a_initial_count = 8 →
  room_a_initial_avg = 35 →
  room_b_initial_count = 5 →
  room_b_initial_avg = 30 →
  moving_person_age = 40 →
  let total_initial_a := room_a_initial_count * room_a_initial_avg
  let total_initial_b := room_b_initial_count * room_b_initial_avg
  let new_total_a := total_initial_a - moving_person_age
  let new_total_b := total_initial_b + moving_person_age
  let new_count_a := room_a_initial_count - 1
  let new_count_b := room_b_initial_count + 1
  let total_new_age := new_total_a + new_total_b
  let total_new_count := new_count_a + new_count_b
  (total_new_age / total_new_count : ℚ) = 33.08 := by
sorry

end NUMINAMATH_CALUDE_new_average_age_after_move_l2991_299176


namespace NUMINAMATH_CALUDE_iron_wire_length_l2991_299138

/-- The length of each cut-off part of the wire in centimeters. -/
def cut_length : ℝ := 10

/-- The original length of the iron wire in centimeters. -/
def original_length : ℝ := 110

/-- The length of the remaining part of the wire after cutting both ends. -/
def remaining_length : ℝ := original_length - 2 * cut_length

/-- Theorem stating that the original length of the iron wire is 110 cm. -/
theorem iron_wire_length :
  (remaining_length = 4 * (2 * cut_length) + 10) →
  original_length = 110 :=
by sorry

end NUMINAMATH_CALUDE_iron_wire_length_l2991_299138


namespace NUMINAMATH_CALUDE_peter_book_percentage_l2991_299180

theorem peter_book_percentage (total_books : ℕ) (brother_percentage : ℚ) (difference : ℕ) : 
  total_books = 20 →
  brother_percentage = 1/10 →
  difference = 6 →
  (↑(brother_percentage * ↑total_books + ↑difference) / ↑total_books : ℚ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_peter_book_percentage_l2991_299180


namespace NUMINAMATH_CALUDE_fence_perimeter_is_112_l2991_299126

-- Define the parameters of the fence
def total_posts : ℕ := 28
def posts_on_long_side : ℕ := 6
def gap_between_posts : ℕ := 4

-- Define the function to calculate the perimeter
def fence_perimeter : ℕ := 
  let posts_on_short_side := (total_posts - 2 * posts_on_long_side + 2) / 2 + 1
  let long_side_length := (posts_on_long_side - 1) * gap_between_posts
  let short_side_length := (posts_on_short_side - 1) * gap_between_posts
  2 * (long_side_length + short_side_length)

-- Theorem statement
theorem fence_perimeter_is_112 : fence_perimeter = 112 := by
  sorry

end NUMINAMATH_CALUDE_fence_perimeter_is_112_l2991_299126


namespace NUMINAMATH_CALUDE_sphere_surface_area_increase_l2991_299189

theorem sphere_surface_area_increase (r : ℝ) (h : r > 0) :
  let original_area := 4 * Real.pi * r^2
  let new_radius := 1.5 * r
  let new_area := 4 * Real.pi * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_increase_l2991_299189


namespace NUMINAMATH_CALUDE_least_b_value_l2991_299118

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The smallest prime factor of a positive integer -/
def smallest_prime_factor (n : ℕ+) : ℕ := sorry

theorem least_b_value (a b : ℕ+) 
  (ha_factors : num_factors a = 3)
  (hb_factors : num_factors b = a)
  (hb_div_a : a ∣ b)
  (ha_smallest_prime : smallest_prime_factor a = 3) :
  36 ≤ b :=
sorry

end NUMINAMATH_CALUDE_least_b_value_l2991_299118


namespace NUMINAMATH_CALUDE_similar_rectangles_l2991_299162

theorem similar_rectangles (w1 l1 w2 : ℝ) (hw1 : w1 = 25) (hl1 : l1 = 40) (hw2 : w2 = 15) :
  let l2 := w2 * l1 / w1
  let perimeter := 2 * (w2 + l2)
  let area := w2 * l2
  (l2 = 24 ∧ perimeter = 78 ∧ area = 360) := by sorry

end NUMINAMATH_CALUDE_similar_rectangles_l2991_299162


namespace NUMINAMATH_CALUDE_field_length_is_28_l2991_299168

/-- Proves that the length of a rectangular field is 28 meters given specific conditions --/
theorem field_length_is_28 (l w : ℝ) (h1 : l = 2 * w) (h2 : (7 : ℝ)^2 = (1/8) * (l * w)) : l = 28 :=
by sorry

end NUMINAMATH_CALUDE_field_length_is_28_l2991_299168


namespace NUMINAMATH_CALUDE_max_area_CDFE_l2991_299144

/-- The area of quadrilateral CDFE in a square ABCD with side length 2,
    where E and F are on sides AB and AD respectively, and AE = AF = 2k. -/
def area_CDFE (k : ℝ) : ℝ := 2 * (1 - k)^2

/-- The theorem stating that the area of CDFE is maximized when k = 1/2,
    and the maximum area is 1/2. -/
theorem max_area_CDFE :
  ∀ k : ℝ, 0 < k → k < 1 →
  area_CDFE k ≤ area_CDFE (1/2) ∧ area_CDFE (1/2) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_area_CDFE_l2991_299144


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l2991_299150

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  asymptote1 : ℝ → ℝ
  asymptote2 : ℝ → ℝ
  point : ℝ × ℝ

/-- The distance between the foci of a hyperbola -/
def focalDistance (h : Hyperbola) : ℝ := sorry

/-- Theorem: For a hyperbola with asymptotes y = x + 3 and y = -x + 5, 
    passing through the point (4,6), the distance between its foci is 2√10 -/
theorem hyperbola_focal_distance :
  let h : Hyperbola := {
    asymptote1 := fun x ↦ x + 3,
    asymptote2 := fun x ↦ -x + 5,
    point := (4, 6)
  }
  focalDistance h = 2 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l2991_299150


namespace NUMINAMATH_CALUDE_joan_balloon_count_l2991_299127

theorem joan_balloon_count (total : ℕ) (melanie : ℕ) (joan : ℕ) : 
  total = 81 → melanie = 41 → joan + melanie = total → joan = 40 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloon_count_l2991_299127


namespace NUMINAMATH_CALUDE_root_sum_of_quadratic_l2991_299163

theorem root_sum_of_quadratic : ∃ (C D : ℝ), 
  (∀ x : ℝ, 3 * x^2 - 9 * x + 6 = 0 ↔ (x = C ∨ x = D)) ∧ 
  C + D = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_of_quadratic_l2991_299163


namespace NUMINAMATH_CALUDE_equation_exists_l2991_299198

theorem equation_exists : ∃ (a b c d e f g h i : ℕ),
  (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧ 
  (e < 10) ∧ (f < 10) ∧ (g < 10) ∧ (h < 10) ∧ (i < 10) ∧
  (a + 100 * b + 10 * c + d = 10 * e + f + 100 * g + 10 * h + i) ∧
  (b = d) ∧ (g = h) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (a ≠ i) ∧
  (b ≠ c) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (b ≠ i) ∧
  (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ (c ≠ i) ∧
  (e ≠ f) ∧ (e ≠ g) ∧ (e ≠ i) ∧
  (f ≠ g) ∧ (f ≠ i) ∧
  (g ≠ i) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_exists_l2991_299198


namespace NUMINAMATH_CALUDE_triangle_inequality_l2991_299164

-- Define a structure for a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  nondeg : min a b < c ∧ c < a + b -- Nondegenerate condition
  unit_perimeter : a + b + c = 1 -- Unit perimeter condition

-- Define the theorem
theorem triangle_inequality (t : Triangle) :
  |((t.a - t.b)/(t.c + t.a*t.b))| + |((t.b - t.c)/(t.a + t.b*t.c))| + |((t.c - t.a)/(t.b + t.a*t.c))| < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2991_299164


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l2991_299110

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b^16)^(1/8))^(1/4))^2 * (((b^16)^(1/4))^(1/8))^2 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l2991_299110


namespace NUMINAMATH_CALUDE_wage_recovery_percentage_l2991_299181

theorem wage_recovery_percentage (original_wage : ℝ) (h : original_wage > 0) :
  let decreased_wage := 0.7 * original_wage
  let required_increase := (original_wage / decreased_wage - 1) * 100
  ∃ ε > 0, abs (required_increase - 42.86) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_wage_recovery_percentage_l2991_299181


namespace NUMINAMATH_CALUDE_rope_length_proof_l2991_299145

theorem rope_length_proof : 
  ∀ (L : ℝ), 
    (L / 4 - L / 6 = 2) →  -- Difference between parts is 2 meters
    (2 * L = 48)           -- Total length of two ropes is 48 meters
  := by sorry

end NUMINAMATH_CALUDE_rope_length_proof_l2991_299145


namespace NUMINAMATH_CALUDE_centroid_circumcenter_distance_squared_l2991_299119

/-- Given a triangle with medians m_a, m_b, m_c and circumradius R,
    the squared distance between the centroid and circumcenter (SM^2)
    is equal to R^2 - (4/27)(m_a^2 + m_b^2 + m_c^2) -/
theorem centroid_circumcenter_distance_squared
  (m_a m_b m_c R : ℝ) :
  ∃ (SM : ℝ),
    SM^2 = R^2 - (4/27) * (m_a^2 + m_b^2 + m_c^2) :=
by sorry

end NUMINAMATH_CALUDE_centroid_circumcenter_distance_squared_l2991_299119


namespace NUMINAMATH_CALUDE_infinite_prime_pairs_l2991_299136

theorem infinite_prime_pairs : 
  ∃ (S : Set (ℕ × ℕ)), 
    (∀ (p q : ℕ), (p, q) ∈ S → Nat.Prime p ∧ Nat.Prime q) ∧ 
    (∀ (p q : ℕ), (p, q) ∈ S → p ∣ (2^(q-1) - 1) ∧ q ∣ (2^(p-1) - 1)) ∧ 
    Set.Infinite S :=
by sorry

end NUMINAMATH_CALUDE_infinite_prime_pairs_l2991_299136


namespace NUMINAMATH_CALUDE_cassidy_grounded_days_l2991_299122

/-- The number of days Cassidy is grounded for lying about her report card -/
def days_grounded_for_lying (total_days : ℕ) (grades_below_b : ℕ) (extra_days_per_grade : ℕ) : ℕ :=
  total_days - (grades_below_b * extra_days_per_grade)

/-- Theorem stating that Cassidy was grounded for 14 days for lying about her report card -/
theorem cassidy_grounded_days : 
  days_grounded_for_lying 26 4 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_cassidy_grounded_days_l2991_299122


namespace NUMINAMATH_CALUDE_investment_interest_rate_l2991_299151

/-- Given an investment scenario, prove the interest rate for the second investment --/
theorem investment_interest_rate 
  (total_investment : ℝ) 
  (desired_interest : ℝ) 
  (first_investment : ℝ) 
  (first_rate : ℝ) 
  (h1 : total_investment = 10000)
  (h2 : desired_interest = 980)
  (h3 : first_investment = 6000)
  (h4 : first_rate = 0.09)
  : 
  let second_investment := total_investment - first_investment
  let first_interest := first_investment * first_rate
  let second_interest := desired_interest - first_interest
  let second_rate := second_interest / second_investment
  second_rate = 0.11 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l2991_299151


namespace NUMINAMATH_CALUDE_foreign_trade_income_l2991_299106

/-- Foreign trade income problem -/
theorem foreign_trade_income 
  (m : ℝ) -- Foreign trade income in 2001 (billion yuan)
  (x : ℝ) -- Percentage increase in 2002
  (n : ℝ) -- Foreign trade income in 2003 (billion yuan)
  (h1 : x > 0) -- Ensure x is positive
  (h2 : m > 0) -- Ensure initial income is positive
  : n = m * (1 + x / 100) * (1 + 2 * x / 100) :=
by sorry

end NUMINAMATH_CALUDE_foreign_trade_income_l2991_299106


namespace NUMINAMATH_CALUDE_equality_proof_l2991_299169

theorem equality_proof : 2222 - 222 + 22 - 2 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_equality_proof_l2991_299169


namespace NUMINAMATH_CALUDE_root_relation_iff_p_values_l2991_299111

-- Define the quadratic equation
def quadratic_equation (p : ℝ) (x : ℝ) : ℝ := x^2 + p*x + 2*p

-- Define the condition for one root being three times the other
def root_condition (p : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), 
    quadratic_equation p x₁ = 0 ∧ 
    quadratic_equation p x₂ = 0 ∧ 
    x₂ = 3 * x₁

-- Theorem statement
theorem root_relation_iff_p_values :
  ∀ p : ℝ, root_condition p ↔ (p = 0 ∨ p = 32/3) :=
by sorry

end NUMINAMATH_CALUDE_root_relation_iff_p_values_l2991_299111


namespace NUMINAMATH_CALUDE_subtract_inequality_l2991_299130

theorem subtract_inequality {a b c : ℝ} (h : a > b) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtract_inequality_l2991_299130


namespace NUMINAMATH_CALUDE_T_increasing_T_not_perfect_square_non_perfect_square_in_T_T_2012th_term_l2991_299100

/-- The sequence of positive integers that are not perfect squares -/
def T : ℕ → ℕ := sorry

/-- T is increasing -/
theorem T_increasing : ∀ n : ℕ, T n < T (n + 1) := sorry

/-- T consists of non-perfect squares -/
theorem T_not_perfect_square : ∀ n : ℕ, ¬ ∃ m : ℕ, T n = m^2 := sorry

/-- Every non-perfect square is in T -/
theorem non_perfect_square_in_T : ∀ k : ℕ, (¬ ∃ m : ℕ, k = m^2) → ∃ n : ℕ, T n = k := sorry

/-- The 2012th term of T is 2057 -/
theorem T_2012th_term : T 2011 = 2057 := sorry

end NUMINAMATH_CALUDE_T_increasing_T_not_perfect_square_non_perfect_square_in_T_T_2012th_term_l2991_299100


namespace NUMINAMATH_CALUDE_simplify_expression_l2991_299195

theorem simplify_expression (a : ℝ) (h : a ≠ 0) : -2 * a^3 / a = -2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2991_299195


namespace NUMINAMATH_CALUDE_sum_of_two_before_last_l2991_299113

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → j < n → a (j + 1) - a j = a (i + 1) - a i

theorem sum_of_two_before_last (a : ℕ → ℕ) :
  arithmetic_sequence a 7 →
  a 0 = 3 →
  a 1 = 8 →
  a 2 = 13 →
  a 6 = 33 →
  a 4 + a 5 = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_before_last_l2991_299113


namespace NUMINAMATH_CALUDE_max_min_product_l2991_299186

theorem max_min_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq : a + b + c = 12) (sum_prod : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 6 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l2991_299186


namespace NUMINAMATH_CALUDE_inequality_proof_l2991_299114

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  Real.exp a - 1 > a ∧ a > a ^ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2991_299114


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l2991_299148

/-- A parallelogram with three known vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ := (1, 1)
  v2 : ℝ × ℝ := (2, 2)
  v3 : ℝ × ℝ := (3, -1)

/-- The fourth vertex of the parallelogram -/
def fourth_vertex (p : Parallelogram) : Set (ℝ × ℝ) :=
  {(2, -2), (4, 0)}

/-- Theorem stating that the fourth vertex of the parallelogram is either (2, -2) or (4, 0) -/
theorem parallelogram_fourth_vertex (p : Parallelogram) :
  ∃ v4 : ℝ × ℝ, v4 ∈ fourth_vertex p :=
sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l2991_299148


namespace NUMINAMATH_CALUDE_midpoint_trajectory_equation_l2991_299167

/-- The equation of the trajectory of the midpoint of the line connecting a fixed point to any point on a circle -/
theorem midpoint_trajectory_equation (P : ℝ × ℝ) (r : ℝ) :
  P = (4, -2) →
  r = 2 →
  ∀ (x y : ℝ), (∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 = r^2 ∧ x = (x₁ + P.1) / 2 ∧ y = (y₁ + P.2) / 2) →
  (x - 2)^2 + (y + 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_equation_l2991_299167


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l2991_299116

-- Define the parabola
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

-- Define a as the x-intercept
def a : ℝ := parabola 0

-- Define b and c as y-intercepts
noncomputable def b : ℝ := (9 - Real.sqrt 33) / 6
noncomputable def c : ℝ := (9 + Real.sqrt 33) / 6

-- Theorem statement
theorem parabola_intercepts_sum : a + b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l2991_299116


namespace NUMINAMATH_CALUDE_tangents_intersect_on_AB_l2991_299196

-- Define the basic geometric objects
structure Point : Type :=
  (x y : ℝ)

structure Line : Type :=
  (a b c : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

-- Define the triangle
def Triangle (A B C : Point) : Prop := sorry

-- Define a point on a line segment
def PointOnSegment (D A B : Point) : Prop := sorry

-- Define incircle and excircle
def Incircle (ω : Circle) (A C D : Point) : Prop := sorry
def Excircle (Ω : Circle) (A C D : Point) : Prop := sorry

-- Define tangent line to a circle
def TangentLine (l : Line) (c : Circle) : Prop := sorry

-- Define intersection of lines
def Intersect (l₁ l₂ : Line) (P : Point) : Prop := sorry

-- Define a point on a line
def PointOnLine (P : Point) (l : Line) : Prop := sorry

-- Main theorem
theorem tangents_intersect_on_AB 
  (A B C D : Point) 
  (ω₁ ω₂ Ω₁ Ω₂ : Circle) 
  (AB : Line) :
  Triangle A B C →
  PointOnSegment D A B →
  Incircle ω₁ A C D →
  Incircle ω₂ B C D →
  Excircle Ω₁ A C D →
  Excircle Ω₂ B C D →
  PointOnLine A AB →
  PointOnLine B AB →
  ∃ (P Q : Point) (l₁ l₂ l₃ l₄ : Line),
    TangentLine l₁ ω₁ ∧ TangentLine l₁ ω₂ ∧
    TangentLine l₂ ω₁ ∧ TangentLine l₂ ω₂ ∧
    TangentLine l₃ Ω₁ ∧ TangentLine l₃ Ω₂ ∧
    TangentLine l₄ Ω₁ ∧ TangentLine l₄ Ω₂ ∧
    Intersect l₁ l₂ P ∧ Intersect l₃ l₄ Q ∧
    PointOnLine P AB ∧ PointOnLine Q AB :=
sorry

end NUMINAMATH_CALUDE_tangents_intersect_on_AB_l2991_299196


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_triangle_height_equation_l2991_299102

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem 1
theorem perpendicular_line_equation (A : Point) (h : A.x = -2 ∧ A.y = 3) :
  ∃ l : Line, perpendicular l ⟨A.y, -A.x, 0⟩ ∧ on_line A l ∧ l.a = 2 ∧ l.b = -3 ∧ l.c = 13 :=
sorry

-- Theorem 2
theorem triangle_height_equation (A B C : Point) 
  (hA : A.x = 4 ∧ A.y = 0) (hB : B.x = 6 ∧ B.y = 7) (hC : C.x = 0 ∧ C.y = 3) :
  ∃ l : Line, perpendicular l ⟨B.y - A.y, A.x - B.x, B.x * A.y - A.x * B.y⟩ ∧ 
    on_line C l ∧ l.a = 2 ∧ l.b = 7 ∧ l.c = -21 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_triangle_height_equation_l2991_299102


namespace NUMINAMATH_CALUDE_product_zero_given_sum_and_seventh_power_sum_zero_l2991_299117

theorem product_zero_given_sum_and_seventh_power_sum_zero 
  (w x y z : ℝ) 
  (sum_zero : w + x + y + z = 0) 
  (seventh_power_sum_zero : w^7 + x^7 + y^7 + z^7 = 0) : 
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_given_sum_and_seventh_power_sum_zero_l2991_299117


namespace NUMINAMATH_CALUDE_rectangle_area_with_hole_l2991_299131

theorem rectangle_area_with_hole (x : ℝ) 
  (h : (3*x ≤ 2*x + 10) ∧ (x ≤ x + 3)) : 
  (2*x + 10) * (x + 3) - (3*x * x) = -x^2 + 16*x + 30 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_with_hole_l2991_299131


namespace NUMINAMATH_CALUDE_cake_payment_dimes_l2991_299191

/-- Represents the number of each type of coin used in the payment -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The value of the payment in cents -/
def payment_value (c : CoinCount) : ℕ :=
  c.pennies + 5 * c.nickels + 10 * c.dimes

theorem cake_payment_dimes :
  ∃ (c : CoinCount),
    c.pennies + c.nickels + c.dimes = 50 ∧
    payment_value c = 200 ∧
    c.dimes = 14 := by
  sorry

end NUMINAMATH_CALUDE_cake_payment_dimes_l2991_299191


namespace NUMINAMATH_CALUDE_prob_select_AB_correct_l2991_299197

/-- The number of students in the class -/
def num_students : ℕ := 5

/-- The number of students to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly A and B -/
def prob_select_AB : ℚ := 1 / 10

theorem prob_select_AB_correct :
  prob_select_AB = (1 : ℚ) / (num_students.choose num_selected) :=
by sorry

end NUMINAMATH_CALUDE_prob_select_AB_correct_l2991_299197


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l2991_299187

/-- Represents a digit in a given base --/
def Digit (d : ℕ) := { n : ℕ // n < d }

/-- Represents a two-digit number in a given base --/
def TwoDigitNumber (d : ℕ) (A B : Digit d) : ℕ := A.val * d + B.val

theorem digit_difference_in_base_d 
  (d : ℕ) 
  (h_d : d > 7) 
  (A B : Digit d) 
  (h_sum : TwoDigitNumber d A B + TwoDigitNumber d A A = 175) : 
  A.val - B.val = 2 := by
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l2991_299187


namespace NUMINAMATH_CALUDE_sequence_sum_proof_l2991_299137

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℚ := (n + 1) / 2

-- Define the geometric sequence {b_n}
def b (n : ℕ) : ℚ := 2^(n-1)

-- Define the sum of the first n terms of {b_n}
def T (n : ℕ) : ℚ := 2^n - 1

theorem sequence_sum_proof :
  -- Given conditions
  (a 3 = 2) ∧
  ((a 1 + a 2 + a 3) = 9/2) ∧
  (b 1 = a 1) ∧
  (b 4 = a 15) →
  -- Conclusion
  ∀ n : ℕ, T n = 2^n - 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_proof_l2991_299137


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l2991_299160

theorem binomial_square_coefficient (a : ℝ) : 
  (∃ r s : ℝ, (r * x + s)^2 = a * x^2 + 18 * x + 9) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l2991_299160


namespace NUMINAMATH_CALUDE_largest_blue_balls_l2991_299174

theorem largest_blue_balls (total : ℕ) (is_prime : ℕ → Prop) : 
  total = 72 →
  (∃ (red blue prime : ℕ), 
    red + blue = total ∧ 
    is_prime prime ∧ 
    red = blue + prime) →
  (∃ (max_blue : ℕ), 
    max_blue ≤ total ∧
    (∀ (blue : ℕ), 
      blue ≤ total →
      (∃ (red prime : ℕ), 
        red + blue = total ∧ 
        is_prime prime ∧ 
        red = blue + prime) →
      blue ≤ max_blue) ∧
    max_blue = 35) :=
by sorry

end NUMINAMATH_CALUDE_largest_blue_balls_l2991_299174


namespace NUMINAMATH_CALUDE_square_properties_l2991_299101

theorem square_properties (a b : ℤ) (h : 2*a^2 + a = 3*b^2 + b) :
  ∃ (x y : ℤ), (a - b = x^2) ∧ (2*a + 2*b + 1 = y^2) := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l2991_299101


namespace NUMINAMATH_CALUDE_equation_one_solutions_l2991_299153

theorem equation_one_solutions (x : ℝ) :
  (x - 2)^2 = 4 → x = 4 ∨ x = 0 := by
  sorry

#check equation_one_solutions

end NUMINAMATH_CALUDE_equation_one_solutions_l2991_299153


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2991_299142

def M : Set ℕ := {1, 2, 4, 8, 16}
def N : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_M_and_N : M ∩ N = {2, 4, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2991_299142


namespace NUMINAMATH_CALUDE_intersection_distance_l2991_299177

/-- Given a linear function f(x) = ax + b, if the distance between the intersection points
    of y=x^2+2 and y=f(x) is √10, and the distance between the intersection points of
    y=x^2-1 and y=f(x)+1 is √42, then the distance between the intersection points of
    y=x^2 and y=f(x)+1 is √34. -/
theorem intersection_distance (a b : ℝ) : 
  let f := (fun x : ℝ => a * x + b)
  let d1 := Real.sqrt ((a^2 + 1) * (a^2 + 4*b - 8))
  let d2 := Real.sqrt ((a^2 + 1) * (a^2 + 4*b + 8))
  d1 = Real.sqrt 10 ∧ d2 = Real.sqrt 42 →
  Real.sqrt ((a^2 + 1) * (a^2 + 4*b + 4)) = Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l2991_299177


namespace NUMINAMATH_CALUDE_function_characterization_l2991_299184

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- Theorem stating that any function satisfying the equation must be of the form f(x) = cx -/
theorem function_characterization (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l2991_299184


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_9_mod_18_l2991_299192

theorem least_five_digit_congruent_to_9_mod_18 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 18 = 9 → n ≥ 10008 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_9_mod_18_l2991_299192


namespace NUMINAMATH_CALUDE_lcm_of_6_8_10_l2991_299146

theorem lcm_of_6_8_10 : Nat.lcm (Nat.lcm 6 8) 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_6_8_10_l2991_299146
