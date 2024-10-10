import Mathlib

namespace unique_divisible_by_13_l4101_410186

def base_7_to_10 (d : Nat) : Nat :=
  3 * 7^3 + d * 7^2 + d * 7 + 6

theorem unique_divisible_by_13 : 
  ∃! d : Nat, d < 7 ∧ (base_7_to_10 d) % 13 = 0 ∧ base_7_to_10 d = 1035 + 56 * d :=
by sorry

end unique_divisible_by_13_l4101_410186


namespace complement_A_union_B_equals_interval_intersection_A_B_empty_t_range_l4101_410178

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * 3 * x + 10

-- Define set A
def A : Set ℝ := {x | f x > 0}

-- Define set B
def B (t : ℝ) : Set ℝ := {x | |x - t| ≤ 1}

-- Theorem for part (I)
theorem complement_A_union_B_equals_interval :
  (Aᶜ ∪ B 1) = {x | -3 ≤ x ∧ x ≤ 2} :=
sorry

-- Theorem for part (II)
theorem intersection_A_B_empty_t_range (t : ℝ) :
  (A ∩ B t = ∅) ↔ (-2 ≤ t ∧ t ≤ 0) :=
sorry

end complement_A_union_B_equals_interval_intersection_A_B_empty_t_range_l4101_410178


namespace sum_of_cubes_l4101_410117

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 + b^3 = 1008 := by
  sorry

end sum_of_cubes_l4101_410117


namespace rhombohedron_volume_l4101_410116

/-- The volume of a rhombohedron formed by extruding a rhombus -/
theorem rhombohedron_volume
  (d1 : ℝ) (d2 : ℝ) (h : ℝ)
  (hd1 : d1 = 25)
  (hd2 : d2 = 50)
  (hh : h = 20) :
  (d1 * d2 / 2) * h = 12500 := by
  sorry

end rhombohedron_volume_l4101_410116


namespace rock_splash_width_l4101_410144

theorem rock_splash_width 
  (num_pebbles num_rocks num_boulders : ℕ)
  (total_width pebble_splash_width boulder_splash_width : ℝ)
  (h1 : num_pebbles = 6)
  (h2 : num_rocks = 3)
  (h3 : num_boulders = 2)
  (h4 : total_width = 7)
  (h5 : pebble_splash_width = 1/4)
  (h6 : boulder_splash_width = 2)
  : (total_width - num_pebbles * pebble_splash_width - num_boulders * boulder_splash_width) / num_rocks = 1/2 := by
  sorry

end rock_splash_width_l4101_410144


namespace triangle_problem_l4101_410173

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : abc.c = 13)
  (h2 : Real.cos abc.A = 5/13) :
  (abc.a = 36 → Real.sin abc.C = 1/3) ∧ 
  (abc.a * abc.b * Real.sin abc.C / 2 = 6 → abc.a = 4 * Real.sqrt 10 ∧ abc.b = 1) :=
by sorry

end triangle_problem_l4101_410173


namespace tylers_age_l4101_410129

theorem tylers_age :
  ∀ (tyler_age brother_age : ℕ),
  tyler_age = brother_age - 3 →
  tyler_age + brother_age = 11 →
  tyler_age = 4 := by
sorry

end tylers_age_l4101_410129


namespace parallel_vectors_a_value_l4101_410107

def m (a : ℝ) : Fin 2 → ℝ := ![a, -2]
def n (a : ℝ) : Fin 2 → ℝ := ![1, 2-a]

theorem parallel_vectors_a_value :
  ∀ a : ℝ, (∃ k : ℝ, k ≠ 0 ∧ m a = k • n a) → (a = 1 + Real.sqrt 3 ∨ a = 1 - Real.sqrt 3) :=
by sorry

end parallel_vectors_a_value_l4101_410107


namespace profit_calculation_correct_l4101_410132

/-- Represents the profit distribution in a partnership business --/
structure ProfitDistribution where
  a_investment : ℚ
  b_investment : ℚ
  c_investment : ℚ
  a_period : ℚ
  b_period : ℚ
  c_period : ℚ
  c_profit : ℚ

/-- Calculates the profit shares and total profit for a given profit distribution --/
def calculate_profits (pd : ProfitDistribution) : 
  (ℚ × ℚ × ℚ × ℚ) :=
  sorry

/-- Theorem stating the correctness of profit calculation --/
theorem profit_calculation_correct (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 2 * pd.b_investment)
  (h2 : pd.b_investment = 3 * pd.c_investment)
  (h3 : pd.a_period = 2 * pd.b_period)
  (h4 : pd.b_period = 3 * pd.c_period)
  (h5 : pd.c_profit = 3000) :
  calculate_profits pd = (108000, 27000, 3000, 138000) :=
  sorry

end profit_calculation_correct_l4101_410132


namespace complex_equation_solution_l4101_410179

theorem complex_equation_solution (x : ℝ) : 
  45 - (28 - (37 - (15 - x))) = 56 → x = 17 := by
  sorry

end complex_equation_solution_l4101_410179


namespace smallest_class_size_class_size_satisfies_conditions_l4101_410192

theorem smallest_class_size (n : ℕ) : (n ≡ 1 [ZMOD 6] ∧ n ≡ 2 [ZMOD 8] ∧ n ≡ 4 [ZMOD 10]) → n ≥ 274 :=
by sorry

theorem class_size_satisfies_conditions : 274 ≡ 1 [ZMOD 6] ∧ 274 ≡ 2 [ZMOD 8] ∧ 274 ≡ 4 [ZMOD 10] :=
by sorry

end smallest_class_size_class_size_satisfies_conditions_l4101_410192


namespace stock_price_calculation_l4101_410162

def initial_price : ℝ := 50
def first_year_increase : ℝ := 2  -- 200% increase
def second_year_decrease : ℝ := 0.5  -- 50% decrease

def final_price : ℝ :=
  initial_price * (1 + first_year_increase) * second_year_decrease

theorem stock_price_calculation :
  final_price = 75 := by sorry

end stock_price_calculation_l4101_410162


namespace father_son_age_ratio_l4101_410122

/-- Represents the age ratio problem between a father and his son Ronit -/
theorem father_son_age_ratio :
  ∀ (ronit_age : ℕ) (father_age : ℕ),
  father_age = 4 * ronit_age →
  father_age + 8 = (5/2) * (ronit_age + 8) →
  (father_age + 16) = 2 * (ronit_age + 16) :=
by
  sorry

end father_son_age_ratio_l4101_410122


namespace consecutive_integers_sum_l4101_410103

/-- If three consecutive integers have a product of 504, their sum is 24. -/
theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 504) → (a + b + c = 24) := by
sorry

end consecutive_integers_sum_l4101_410103


namespace sample_capacity_l4101_410147

/-- Given a sample divided into groups, prove that the total sample capacity is 144
    when one group has a frequency of 36 and a frequency rate of 0.25. -/
theorem sample_capacity (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ) : 
  frequency = 36 → frequency_rate = 1/4 → n = frequency / frequency_rate → n = 144 := by
  sorry

end sample_capacity_l4101_410147


namespace factors_of_2310_l4101_410108

theorem factors_of_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end factors_of_2310_l4101_410108


namespace systematic_sampling_seventh_group_l4101_410158

/-- Represents the systematic sampling method for a population --/
structure SystematicSampling where
  populationSize : Nat
  numGroups : Nat
  sampleSize : Nat
  firstDrawn : Nat

/-- Calculates the number drawn for a given group --/
def SystematicSampling.numberDrawn (s : SystematicSampling) (group : Nat) : Nat :=
  let offset := (s.firstDrawn + 33 * group) % 100
  let baseNumber := (group - 1) * (s.populationSize / s.numGroups)
  baseNumber + offset

/-- The main theorem to prove --/
theorem systematic_sampling_seventh_group 
  (s : SystematicSampling)
  (h1 : s.populationSize = 1000)
  (h2 : s.numGroups = 10)
  (h3 : s.sampleSize = 10)
  (h4 : s.firstDrawn = 57) :
  s.numberDrawn 7 = 688 := by
  sorry

end systematic_sampling_seventh_group_l4101_410158


namespace ten_zeros_in_expansion_l4101_410100

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The main theorem: The number of trailing zeros in (10^11 - 2)^2 is 10 -/
theorem ten_zeros_in_expansion : trailingZeros ((10^11 - 2)^2) = 10 := by
  sorry

end ten_zeros_in_expansion_l4101_410100


namespace mystery_book_shelves_l4101_410181

theorem mystery_book_shelves :
  let books_per_shelf : ℕ := 4
  let picture_book_shelves : ℕ := 3
  let total_books : ℕ := 32
  let mystery_book_shelves : ℕ := (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf
  mystery_book_shelves = 5 :=
by sorry

end mystery_book_shelves_l4101_410181


namespace sweet_salty_difference_l4101_410118

/-- Represents the number of cookies of each type --/
structure CookieCount where
  sweet : ℕ
  salty : ℕ
  chocolate : ℕ

/-- The initial number of cookies Paco had --/
def initialCookies : CookieCount :=
  { sweet := 39, salty := 18, chocolate := 12 }

/-- The number of cookies Paco ate --/
def eatenCookies : CookieCount :=
  { sweet := 27, salty := 6, chocolate := 8 }

/-- Theorem stating the difference between sweet and salty cookies eaten --/
theorem sweet_salty_difference :
  eatenCookies.sweet - eatenCookies.salty = 21 := by
  sorry


end sweet_salty_difference_l4101_410118


namespace sum_of_reciprocals_of_roots_l4101_410180

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 17*x + 8 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 17*x + 8 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ = 17 / 8) := by
sorry

end sum_of_reciprocals_of_roots_l4101_410180


namespace cookie_bags_count_l4101_410183

theorem cookie_bags_count (cookies_per_bag : ℕ) (total_cookies : ℕ) (h1 : cookies_per_bag = 19) (h2 : total_cookies = 703) :
  total_cookies / cookies_per_bag = 37 := by
  sorry

end cookie_bags_count_l4101_410183


namespace first_digit_891_base8_l4101_410172

/-- Represents a positive integer in a given base --/
def BaseRepresentation (n : ℕ+) (base : ℕ) : List ℕ :=
  sorry

/-- Returns the first (leftmost) digit of a number's representation in a given base --/
def firstDigit (n : ℕ+) (base : ℕ) : ℕ :=
  match BaseRepresentation n base with
  | [] => 0  -- This case should never occur for positive integers
  | d::_ => d

theorem first_digit_891_base8 :
  firstDigit 891 8 = 1 := by
  sorry

end first_digit_891_base8_l4101_410172


namespace tangent_line_at_1_0_l4101_410145

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_1_0 :
  let p : ℝ × ℝ := (1, 0)
  let m : ℝ := f' p.1
  let tangent_line (x : ℝ) : ℝ := m * (x - p.1) + p.2
  (∀ x, tangent_line x = x - 1) ∧ f p.1 = p.2 := by
  sorry


end tangent_line_at_1_0_l4101_410145


namespace ordered_pairs_count_l4101_410146

theorem ordered_pairs_count : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 36) (Finset.range 37 ×ˢ Finset.range 37)).card ∧ n = 9 := by
  sorry

end ordered_pairs_count_l4101_410146


namespace line_and_segment_properties_l4101_410161

-- Define the points and lines
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (1, -1)
def C : ℝ × ℝ := (0, 2)

def line_l (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line_m (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Define the theorem
theorem line_and_segment_properties :
  -- Given conditions
  (∃ a : ℝ, A = (2, a) ∧ B = (a, -1)) →
  (∀ x y : ℝ, line_l x y ↔ ∃ t : ℝ, x = 2 * (1 - t) + t ∧ y = 1 * (1 - t) + (-1) * t) →
  (∀ x y : ℝ, line_l x y → line_m (x + 1) (y + 1)) →
  -- Conclusions
  (∀ x y : ℝ, line_l x y ↔ 2 * x - y - 3 = 0) ∧
  Real.sqrt 10 = Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) :=
by sorry

end line_and_segment_properties_l4101_410161


namespace fixed_point_on_line_l4101_410130

theorem fixed_point_on_line (m : ℝ) : 
  (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by sorry

end fixed_point_on_line_l4101_410130


namespace sum_of_quadratic_roots_l4101_410127

theorem sum_of_quadratic_roots (a b c d e : ℝ) (h : ∀ x, a * x^2 + b * x + c = d * x + e) :
  let x₁ := (10 + 4 * Real.sqrt 5) / 2
  let x₂ := (10 - 4 * Real.sqrt 5) / 2
  x₁ + x₂ = 10 := by
sorry

end sum_of_quadratic_roots_l4101_410127


namespace power_sum_integer_l4101_410157

theorem power_sum_integer (x : ℝ) (h : ∃ (a : ℤ), x + 1/x = a) :
  ∀ (n : ℕ), ∃ (b : ℤ), x^n + 1/(x^n) = b :=
by sorry

end power_sum_integer_l4101_410157


namespace binomial_20_4_l4101_410191

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end binomial_20_4_l4101_410191


namespace difference_of_numbers_l4101_410133

theorem difference_of_numbers (x y : ℝ) (h_sum : x + y = 36) (h_product : x * y = 105) :
  |x - y| = 6 * Real.sqrt 24.333 := by
  sorry

end difference_of_numbers_l4101_410133


namespace factorial_calculation_l4101_410140

theorem factorial_calculation : (4 * Nat.factorial 6 + 36 * Nat.factorial 5) / Nat.factorial 7 = 10 / 7 := by
  sorry

end factorial_calculation_l4101_410140


namespace max_four_digit_sum_l4101_410170

def A (s n k : ℕ) : ℕ :=
  if n = 1 then
    if 1 ≤ s ∧ s ≤ k then 1 else 0
  else if s < n then 0
  else if k = 0 then 0
  else A (s - k) (n - 1) (k - 1) + A s n (k - 1)

theorem max_four_digit_sum :
  (∀ s, s ≠ 20 → A s 4 9 ≤ A 20 4 9) ∧
  A 20 4 9 = 12 := by sorry

end max_four_digit_sum_l4101_410170


namespace inequality_proof_l4101_410169

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 1/2) : 
  (1 - a + c) / (Real.sqrt c * (Real.sqrt a + 2 * Real.sqrt b)) ≥ 2 := by
sorry

end inequality_proof_l4101_410169


namespace earliest_retirement_year_l4101_410160

/-- Rule of 70 retirement eligibility function -/
def eligible_to_retire (current_year : ℕ) (hire_year : ℕ) (hire_age : ℕ) : Prop :=
  (current_year - hire_year) + (hire_age + (current_year - hire_year)) ≥ 70

/-- Theorem: The earliest retirement year for an employee hired in 1989 at age 32 is 2008 -/
theorem earliest_retirement_year :
  ∀ year : ℕ, year ≥ 1989 →
  (eligible_to_retire year 1989 32 ↔ year ≥ 2008) :=
by sorry

end earliest_retirement_year_l4101_410160


namespace all_red_raise_hands_eventually_l4101_410138

/-- Represents the color of a stamp -/
inductive StampColor
| Red
| Green

/-- Represents a faculty member -/
structure FacultyMember where
  stamp : StampColor

/-- Represents the state of the game on a given day -/
structure GameState where
  day : ℕ
  faculty : List FacultyMember
  handsRaised : List FacultyMember

/-- Predicate to check if a faculty member raises their hand -/
def raisesHand (member : FacultyMember) (state : GameState) : Prop :=
  member ∈ state.handsRaised

/-- The main theorem to be proved -/
theorem all_red_raise_hands_eventually 
  (n : ℕ) 
  (faculty : List FacultyMember) 
  (h1 : faculty.length = n) 
  (h2 : ∃ m, m ∈ faculty ∧ m.stamp = StampColor.Red) :
  ∃ (finalState : GameState), 
    finalState.day = n ∧ 
    ∀ m, m ∈ faculty → m.stamp = StampColor.Red → raisesHand m finalState :=
  sorry


end all_red_raise_hands_eventually_l4101_410138


namespace square_trinomial_equality_l4101_410184

theorem square_trinomial_equality : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end square_trinomial_equality_l4101_410184


namespace colored_plane_triangles_l4101_410182

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point
  sideLength : ℝ
  isEquilateral : 
    (a.x - b.x)^2 + (a.y - b.y)^2 = sideLength^2 ∧
    (b.x - c.x)^2 + (b.y - c.y)^2 = sideLength^2 ∧
    (c.x - a.x)^2 + (c.y - a.y)^2 = sideLength^2

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  a : Point
  b : Point
  c : Point
  legLength : ℝ
  isIsoscelesRight :
    (a.x - b.x)^2 + (a.y - b.y)^2 = 2 * legLength^2 ∧
    (b.x - c.x)^2 + (b.y - c.y)^2 = legLength^2 ∧
    (c.x - a.x)^2 + (c.y - a.y)^2 = legLength^2

-- State the theorem
theorem colored_plane_triangles (coloring : Coloring) :
  (∃ t : EquilateralTriangle, 
    (t.sideLength = 673 * Real.sqrt 3 ∨ t.sideLength = 2019) ∧
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c) ∧
  (∃ t : IsoscelesRightTriangle,
    (t.legLength = 1010 * Real.sqrt 2 ∨ t.legLength = 2020) ∧
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c) :=
by sorry

end colored_plane_triangles_l4101_410182


namespace simplify_radical_product_l4101_410163

theorem simplify_radical_product (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (40 * x) * Real.sqrt (45 * x) * Real.sqrt (56 * x) = 120 * Real.sqrt (7 * x) :=
by sorry

end simplify_radical_product_l4101_410163


namespace hexagon_triangle_area_ratio_l4101_410131

structure RegularHexagon where
  vertices : Finset (Fin 6 → ℝ × ℝ)
  is_regular : sorry
  is_divided : sorry

def center (h : RegularHexagon) : ℝ × ℝ := sorry

def small_triangle (h : RegularHexagon) (i j : Fin 6) (g : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

def large_triangle (h : RegularHexagon) (i j k : Fin 6) : Set (ℝ × ℝ) := sorry

def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem hexagon_triangle_area_ratio (h : RegularHexagon) :
  let g := center h
  let small_tri := small_triangle h 0 1 g
  let large_tri := large_triangle h 0 3 5
  area small_tri / area large_tri = 1 / 6 := by sorry

end hexagon_triangle_area_ratio_l4101_410131


namespace age_difference_l4101_410112

-- Define the ages of the siblings
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := 16

-- Theorem to prove
theorem age_difference : greg_age - marcia_age = 2 := by
  sorry

end age_difference_l4101_410112


namespace jeans_cost_proof_l4101_410106

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.50
def quarters_left : ℕ := 97

def jeans_cost : ℚ := initial_amount - pizza_cost - soda_cost - (quarters_left : ℚ) * (1 / 4)

theorem jeans_cost_proof : jeans_cost = 11.50 := by
  sorry

end jeans_cost_proof_l4101_410106


namespace polygon_internal_diagonals_l4101_410128

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  -- Add necessary fields and constraints
  sides : ℕ
  sides_eq : sides = n
  sides_ge_3 : n ≥ 3

/-- A diagonal of a polygon -/
structure Diagonal (p : Polygon n) where
  -- Add necessary fields and constraints

/-- Predicate to check if a diagonal is completely inside the polygon -/
def is_inside (d : Diagonal p) : Prop :=
  -- Define the condition for a diagonal to be inside the polygon
  sorry

/-- The number of complete internal diagonals in a polygon -/
def num_internal_diagonals (p : Polygon n) : ℕ :=
  -- Define the number of internal diagonals
  sorry

/-- Theorem: Any polygon with more than 3 sides has at least one internal diagonal,
    and the minimum number of internal diagonals is n-3 -/
theorem polygon_internal_diagonals (n : ℕ) (h : n > 3) (p : Polygon n) :
  (∃ d : Diagonal p, is_inside d) ∧ num_internal_diagonals p = n - 3 :=
sorry

end polygon_internal_diagonals_l4101_410128


namespace money_distribution_l4101_410150

theorem money_distribution (a b c d : ℕ) : 
  a + b + c + d = 2000 →
  a + c = 900 →
  b + c = 1100 →
  a + d = 700 →
  c = 200 := by
sorry

end money_distribution_l4101_410150


namespace m_range_l4101_410148

theorem m_range (m : ℝ) : 
  (2 * 3 - m > 4) ∧ (2 * 2 - m ≤ 4) → 0 ≤ m ∧ m < 2 := by
  sorry

end m_range_l4101_410148


namespace inequality_proof_l4101_410187

theorem inequality_proof (h : Real.log (1/2) < 0) : (1/2)^3 < (1/2)^2 := by
  sorry

end inequality_proof_l4101_410187


namespace max_d_value_l4101_410177

def a (n : ℕ+) : ℕ := 99 + 2 * n ^ 2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ (n : ℕ+), d n = 11) ∧ (∀ (n : ℕ+), d n ≤ 11) :=
by sorry

end max_d_value_l4101_410177


namespace gate_buyers_count_l4101_410139

/-- The number of people who pre-bought tickets -/
def preBuyers : ℕ := 20

/-- The price of a pre-bought ticket -/
def prePrice : ℕ := 155

/-- The price of a ticket bought at the gate -/
def gatePrice : ℕ := 200

/-- The additional amount paid by gate buyers compared to pre-buyers -/
def additionalAmount : ℕ := 2900

/-- The number of people who bought tickets at the gate -/
def gateBuyers : ℕ := 30

theorem gate_buyers_count :
  gateBuyers * gatePrice = preBuyers * prePrice + additionalAmount := by
  sorry

end gate_buyers_count_l4101_410139


namespace line_plane_intersection_l4101_410165

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (intersects : Line → Plane → Prop)

-- State the theorem
theorem line_plane_intersection
  (a b : Line) (α : Plane)
  (h1 : parallel a α)
  (h2 : perpendicular b a) :
  intersects b α :=
sorry

end line_plane_intersection_l4101_410165


namespace circle_centers_locus_l4101_410166

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 16

-- Define the property of being externally tangent to C₁
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

-- Define the property of being internally tangent to C₂
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 2)^2 + b^2 = (4 - r)^2

-- Define the locus equation
def locus_equation (a b : ℝ) : Prop := 84 * a^2 + 100 * b^2 - 168 * a - 441 = 0

-- State the theorem
theorem circle_centers_locus (a b : ℝ) :
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) ↔ locus_equation a b :=
sorry

end circle_centers_locus_l4101_410166


namespace reciprocal_problem_l4101_410174

theorem reciprocal_problem (x : ℝ) (h : 6 * x = 12) : 150 * (1 / x) = 75 := by
  sorry

end reciprocal_problem_l4101_410174


namespace bedbug_growth_proof_l4101_410109

def bedbug_population (initial_population : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_population * growth_factor ^ days

theorem bedbug_growth_proof :
  bedbug_population 30 3 4 = 2430 := by
  sorry

end bedbug_growth_proof_l4101_410109


namespace largest_consecutive_non_prime_under_50_l4101_410171

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_consecutive_non_prime_under_50 (a b c d e f : ℕ) :
  a < 100 ∧ b < 100 ∧ c < 100 ∧ d < 100 ∧ e < 100 ∧ f < 100 →  -- two-digit integers
  a < 50 ∧ b < 50 ∧ c < 50 ∧ d < 50 ∧ e < 50 ∧ f < 50 →  -- less than 50
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧ f = e + 1 →  -- consecutive
  ¬(is_prime a) ∧ ¬(is_prime b) ∧ ¬(is_prime c) ∧ 
  ¬(is_prime d) ∧ ¬(is_prime e) ∧ ¬(is_prime f) →  -- not prime
  f = 37 :=
by sorry

end largest_consecutive_non_prime_under_50_l4101_410171


namespace brooke_jumping_jacks_l4101_410164

def sidney_monday : ℕ := 20
def sidney_tuesday : ℕ := 36
def sidney_wednesday : ℕ := 40
def sidney_thursday : ℕ := 50

def sidney_total : ℕ := sidney_monday + sidney_tuesday + sidney_wednesday + sidney_thursday

def brooke_multiplier : ℕ := 3

theorem brooke_jumping_jacks : sidney_total * brooke_multiplier = 438 := by
  sorry

end brooke_jumping_jacks_l4101_410164


namespace intersection_coordinate_sum_l4101_410135

/-- Given points A, B, C, D, E, and F in a coordinate plane, where:
    A is at (0,8), B at (0,0), C at (10,0)
    D is the midpoint of AB
    E is the midpoint of BC
    F is the intersection of lines AE and CD
    Prove that the sum of F's coordinates is 6 -/
theorem intersection_coordinate_sum :
  let A : ℝ × ℝ := (0, 8)
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (10, 0)
  let D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let m_AE : ℝ := (E.2 - A.2) / (E.1 - A.1)
  let b_AE : ℝ := A.2 - m_AE * A.1
  let m_CD : ℝ := (D.2 - C.2) / (D.1 - C.1)
  let b_CD : ℝ := C.2 - m_CD * C.1
  let F : ℝ × ℝ := ((b_CD - b_AE) / (m_AE - m_CD), m_AE * ((b_CD - b_AE) / (m_AE - m_CD)) + b_AE)
  F.1 + F.2 = 6 :=
by sorry

end intersection_coordinate_sum_l4101_410135


namespace cosine_set_product_l4101_410168

open Real Set

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

def S (a₁ : ℝ) : Set ℝ := {x | ∃ n : ℕ+, x = cos (arithmeticSequence a₁ (2 * π / 3) n)}

theorem cosine_set_product (a₁ : ℝ) :
  (∃ a b : ℝ, S a₁ = {a, b} ∧ a ≠ b) → 
  ∀ a b : ℝ, S a₁ = {a, b} → a * b = -1/2 := by
sorry

end cosine_set_product_l4101_410168


namespace both_balls_prob_at_least_one_ball_prob_l4101_410176

/-- The probability space for the ball experiment -/
structure BallProbSpace where
  /-- The probability of ball A falling into the box -/
  prob_A : ℝ
  /-- The probability of ball B falling into the box -/
  prob_B : ℝ
  /-- The probability of ball A falling into the box is 1/2 -/
  hA : prob_A = 1/2
  /-- The probability of ball B falling into the box is 1/3 -/
  hB : prob_B = 1/3
  /-- The events A and B are independent -/
  indep : ∀ {p : ℝ} {q : ℝ}, p = prob_A → q = prob_B → p * q = prob_A * prob_B

/-- The probability that both ball A and ball B fall into the box is 1/6 -/
theorem both_balls_prob (space : BallProbSpace) : space.prob_A * space.prob_B = 1/6 := by
  sorry

/-- The probability that at least one of ball A and ball B falls into the box is 2/3 -/
theorem at_least_one_ball_prob (space : BallProbSpace) :
  1 - (1 - space.prob_A) * (1 - space.prob_B) = 2/3 := by
  sorry

end both_balls_prob_at_least_one_ball_prob_l4101_410176


namespace night_shift_arrangements_count_l4101_410123

/-- The number of days in the shift schedule -/
def num_days : ℕ := 6

/-- The number of people available for shifts -/
def num_people : ℕ := 4

/-- The number of scenarios for arranging consecutive shifts -/
def num_scenarios : ℕ := 6

/-- Calculates the number of different night shift arrangements -/
def night_shift_arrangements : ℕ := 
  num_scenarios * (num_people.factorial / (num_people - 2).factorial) * 
  ((num_people - 2).factorial / (num_people - 4).factorial)

/-- Theorem stating the number of different night shift arrangements -/
theorem night_shift_arrangements_count : night_shift_arrangements = 144 := by
  sorry

end night_shift_arrangements_count_l4101_410123


namespace number_above_265_l4101_410185

/-- Represents the pyramid-like array of numbers -/
def pyramid_array (n : ℕ) : List ℕ :=
  List.range (n * n + 1) -- This generates a list of numbers from 0 to n^2

/-- The number of elements in the nth row of the pyramid -/
def row_length (n : ℕ) : ℕ := 2 * n - 1

/-- The starting number of the nth row -/
def row_start (n : ℕ) : ℕ := (n - 1) ^ 2 + 1

/-- The position of a number in its row -/
def position_in_row (x : ℕ) : ℕ :=
  x - row_start (Nat.sqrt x) + 1

/-- The number directly above a given number in the pyramid -/
def number_above (x : ℕ) : ℕ :=
  row_start (Nat.sqrt x - 1) + position_in_row x - 1

theorem number_above_265 :
  number_above 265 = 234 := by sorry

end number_above_265_l4101_410185


namespace rectangle_area_l4101_410156

/-- Given a rectangle where the length is 3 times the width and the width is 6 inches,
    prove that the area is 108 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 6 →
  length = 3 * width →
  area = length * width →
  area = 108 := by
  sorry

end rectangle_area_l4101_410156


namespace min_sum_squares_l4101_410114

theorem min_sum_squares (a b t : ℝ) (h : a + b = t) :
  ∃ (min : ℝ), min = t^2 / 2 ∧ ∀ (x y : ℝ), x + y = t → x^2 + y^2 ≥ min :=
sorry

end min_sum_squares_l4101_410114


namespace hemisphere_surface_area_l4101_410189

theorem hemisphere_surface_area (C : ℝ) (h : C = 36) :
  let r := C / (2 * Real.pi)
  3 * Real.pi * r^2 = 972 / Real.pi :=
by sorry

end hemisphere_surface_area_l4101_410189


namespace constant_value_l4101_410101

-- Define the function [[x]]
def bracket (x : ℝ) (c : ℝ) : ℝ := x^2 + 2*x + c

-- State the theorem
theorem constant_value :
  ∃ c : ℝ, (∀ x : ℝ, bracket x c = x^2 + 2*x + c) ∧ bracket 2 c = 12 → c = 4 := by
sorry

end constant_value_l4101_410101


namespace gcd_six_digit_repeated_is_1001_l4101_410195

/-- A function that generates a six-digit number by repeating a three-digit number -/
def repeat_three_digit (n : ℕ) : ℕ :=
  1001 * n

/-- The set of all six-digit numbers formed by repeating a three-digit number -/
def six_digit_repeated_set : Set ℕ :=
  {m | ∃ n, 100 ≤ n ∧ n < 1000 ∧ m = repeat_three_digit n}

/-- Theorem stating that the greatest common divisor of all numbers in the set is 1001 -/
theorem gcd_six_digit_repeated_is_1001 :
  ∃ d, d > 0 ∧ (∀ m ∈ six_digit_repeated_set, d ∣ m) ∧
  (∀ k, k > 0 → (∀ m ∈ six_digit_repeated_set, k ∣ m) → k ≤ d) ∧ d = 1001 := by
  sorry

end gcd_six_digit_repeated_is_1001_l4101_410195


namespace inequality_solution_l4101_410113

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the inequality function
def inequality (a : ℝ) (x : ℝ) : Prop :=
  log a (x^2 - x - 2) > log a (x - 2/a) + 1

-- Theorem statement
theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → ∀ x, inequality a x ↔ x > 1 + a) ∧
  (0 < a ∧ a < 1 → ¬∃ x, inequality a x) :=
sorry

end inequality_solution_l4101_410113


namespace factorization_difference_l4101_410198

theorem factorization_difference (c d : ℤ) : 
  (∀ x : ℝ, 4 * x^2 - 17 * x - 15 = (4 * x + c) * (x + d)) → c - d = 8 := by
  sorry

end factorization_difference_l4101_410198


namespace necessary_not_sufficient_l4101_410124

theorem necessary_not_sufficient :
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ x^2 - 3*x + 2 ≥ 0) :=
by sorry

end necessary_not_sufficient_l4101_410124


namespace mike_ride_distance_l4101_410105

/-- Represents the taxi fare structure -/
structure TaxiFare where
  base_fare : ℚ
  per_mile_rate : ℚ
  toll : ℚ

/-- Calculates the total fare for a given distance -/
def calculate_fare (fare_structure : TaxiFare) (distance : ℚ) : ℚ :=
  fare_structure.base_fare + fare_structure.toll + fare_structure.per_mile_rate * distance

theorem mike_ride_distance (mike_fare annie_fare : TaxiFare) 
  (h1 : mike_fare.base_fare = 2.5)
  (h2 : mike_fare.per_mile_rate = 0.25)
  (h3 : mike_fare.toll = 0)
  (h4 : annie_fare.base_fare = 2.5)
  (h5 : annie_fare.per_mile_rate = 0.25)
  (h6 : annie_fare.toll = 5)
  (h7 : calculate_fare annie_fare 26 = calculate_fare mike_fare (46 : ℚ)) :
  ∃ x : ℚ, calculate_fare mike_fare x = calculate_fare annie_fare 26 ∧ x = 46 := by
  sorry

#eval (46 : ℚ)

end mike_ride_distance_l4101_410105


namespace geometric_sequence_value_l4101_410136

theorem geometric_sequence_value (a : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*a + 2) = r * a ∧ (3*a + 3) = r * (2*a + 2)) → a = -4 := by
  sorry

end geometric_sequence_value_l4101_410136


namespace quadratic_equation_coefficients_l4101_410194

/-- Given a quadratic equation 2x^2 - 1 = 6x, prove its coefficients -/
theorem quadratic_equation_coefficients :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 - 1 - 6 * x
  ∃ (a b c : ℝ), (∀ x, f x = a * x^2 + b * x + c) ∧ a = 2 ∧ b = -6 ∧ c = -1 := by
  sorry

end quadratic_equation_coefficients_l4101_410194


namespace system_solution_l4101_410115

theorem system_solution : ∃ (x y : ℝ), 2 * x + y = 4 ∧ 3 * x - 2 * y = 13 := by
  use 3, -2
  sorry

#check system_solution

end system_solution_l4101_410115


namespace ninth_term_is_18_l4101_410141

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  first_term : a 1 = 1 / 2
  condition : a 2 * a 8 = 2 * a 5 + 3

/-- The 9th term of the geometric sequence is 18 -/
theorem ninth_term_is_18 (seq : GeometricSequence) : seq.a 9 = 18 := by
  sorry

end ninth_term_is_18_l4101_410141


namespace unique_six_digit_number_l4101_410149

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ (n / 100000 = 2)

def move_first_to_last (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

theorem unique_six_digit_number : 
  ∀ n : ℕ, is_valid_number n → (move_first_to_last n = 3 * n) → n = 285714 :=
by sorry

end unique_six_digit_number_l4101_410149


namespace additional_charging_time_l4101_410190

/-- Represents the charging characteristics of a mobile battery -/
structure BatteryCharging where
  initial_charge_time : ℕ  -- Time to reach 20% charge in minutes
  initial_charge_percent : ℕ  -- Initial charge percentage
  total_charge_time : ℕ  -- Total time to reach P% charge in minutes

/-- Theorem stating the additional charging time -/
theorem additional_charging_time (b : BatteryCharging) 
  (h1 : b.initial_charge_time = 60)  -- 1 hour = 60 minutes
  (h2 : b.initial_charge_percent = 20)
  (h3 : b.total_charge_time = b.initial_charge_time + 150) :
  b.total_charge_time - b.initial_charge_time = 150 := by
  sorry

#check additional_charging_time

end additional_charging_time_l4101_410190


namespace wind_velocity_calculation_l4101_410152

/-- The relationship between pressure, area, and velocity -/
def pressure_relationship (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^3

/-- The theorem to prove -/
theorem wind_velocity_calculation (k : ℝ) :
  pressure_relationship k 2 8 = 4 →
  pressure_relationship k 4 12.8 = 32 := by
  sorry

end wind_velocity_calculation_l4101_410152


namespace roots_sum_of_powers_l4101_410111

theorem roots_sum_of_powers (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 3 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 3 + 1 = 0 →
  r^12 + s^12 = 940802 := by
  sorry

end roots_sum_of_powers_l4101_410111


namespace prime_divides_sum_of_powers_l4101_410153

theorem prime_divides_sum_of_powers (p : ℕ) (hp : Prime p) :
  ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) := by
  sorry

end prime_divides_sum_of_powers_l4101_410153


namespace sequence_matches_given_terms_l4101_410167

/-- Definition of the sequence -/
def a (n : ℕ) : ℚ := (n + 2 : ℚ) / (2 * n + 3 : ℚ)

/-- The theorem stating that the first four terms of the sequence match the given values -/
theorem sequence_matches_given_terms :
  (a 1 = 3 / 5) ∧ (a 2 = 4 / 7) ∧ (a 3 = 5 / 9) ∧ (a 4 = 6 / 11) :=
by sorry

end sequence_matches_given_terms_l4101_410167


namespace triangle_area_l4101_410110

def a : Fin 2 → ℝ := ![4, -3]
def b : Fin 2 → ℝ := ![6, 1]

theorem triangle_area : 
  (1/2 : ℝ) * |a 0 * b 1 - a 1 * b 0| = 11 := by sorry

end triangle_area_l4101_410110


namespace distance_to_focus_l4101_410120

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the point P
structure Point (x y : ℝ) where
  on_parabola : parabola x y
  distance_to_y_axis : x = 4

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem distance_to_focus (P : Point x y) : 
  Real.sqrt ((x - 2)^2 + y^2) = 6 := by sorry

end distance_to_focus_l4101_410120


namespace min_value_quadratic_l4101_410193

theorem min_value_quadratic :
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 + 6 * x + 1487
  ∃ (m : ℝ), m = 1484 ∧ ∀ x, f x ≥ m := by
  sorry

end min_value_quadratic_l4101_410193


namespace salary_increase_proof_l4101_410188

/-- Proves that given an employee's new annual salary of $90,000 after a 38.46153846153846% increase, the amount of the salary increase is $25,000. -/
theorem salary_increase_proof (new_salary : ℝ) (percent_increase : ℝ) 
  (h1 : new_salary = 90000)
  (h2 : percent_increase = 38.46153846153846) : 
  new_salary - (new_salary / (1 + percent_increase / 100)) = 25000 := by
  sorry

end salary_increase_proof_l4101_410188


namespace polynomial_remainder_l4101_410197

/-- The polynomial f(x) = x^5 - 5x^4 + 8x^3 + 25x^2 - 14x - 40 -/
def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 8*x^3 + 25*x^2 - 14*x - 40

/-- The remainder when f(x) is divided by (x-2) -/
def remainder : ℝ := f 2

theorem polynomial_remainder : remainder = 48 := by sorry

end polynomial_remainder_l4101_410197


namespace cos_symmetry_l4101_410175

theorem cos_symmetry (x : ℝ) :
  let f (x : ℝ) := Real.cos (2 * x + π / 3)
  let symmetry_point := π / 3
  f (symmetry_point + x) = f (symmetry_point - x) := by
  sorry

#check cos_symmetry

end cos_symmetry_l4101_410175


namespace bret_in_seat_three_l4101_410104

-- Define the type for seats
inductive Seat
| one
| two
| three
| four

-- Define the type for people
inductive Person
| Abby
| Bret
| Carl
| Dana

-- Define the seating arrangement as a function from Seat to Person
def SeatingArrangement := Seat → Person

-- Define what it means for two people to be adjacent
def adjacent (s : SeatingArrangement) (p1 p2 : Person) : Prop :=
  ∃ (seat1 seat2 : Seat), 
    (s seat1 = p1 ∧ s seat2 = p2) ∧ 
    (seat1 = Seat.one ∧ seat2 = Seat.two ∨
     seat1 = Seat.two ∧ seat2 = Seat.three ∨
     seat1 = Seat.three ∧ seat2 = Seat.four ∨
     seat2 = Seat.one ∧ seat1 = Seat.two ∨
     seat2 = Seat.two ∧ seat1 = Seat.three ∨
     seat2 = Seat.three ∧ seat1 = Seat.four)

-- Define what it means for one person to be between two others
def between (s : SeatingArrangement) (p1 p2 p3 : Person) : Prop :=
  (s Seat.one = p1 ∧ s Seat.two = p2 ∧ s Seat.three = p3) ∨
  (s Seat.two = p1 ∧ s Seat.three = p2 ∧ s Seat.four = p3) ∨
  (s Seat.four = p1 ∧ s Seat.three = p2 ∧ s Seat.two = p3) ∨
  (s Seat.three = p1 ∧ s Seat.two = p2 ∧ s Seat.one = p3)

theorem bret_in_seat_three :
  ∀ (s : SeatingArrangement),
    (s Seat.two = Person.Abby) →
    (¬ adjacent s Person.Bret Person.Dana) →
    (¬ between s Person.Carl Person.Abby Person.Dana) →
    (s Seat.three = Person.Bret) :=
by sorry

end bret_in_seat_three_l4101_410104


namespace integer_fraction_characterization_l4101_410126

theorem integer_fraction_characterization (a b : ℕ) :
  (∃ k : ℤ, (a^3 + 1 : ℤ) = k * (2*a*b^2 + 1)) ↔
  (∃ n : ℕ, a = 2*n^2 + 1 ∧ b = n) :=
sorry

end integer_fraction_characterization_l4101_410126


namespace geometric_sequence_property_l4101_410151

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of the first n terms of a sequence -/
def SequenceProduct (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (·*·) 1

theorem geometric_sequence_property (a : ℕ → ℝ) (m : ℕ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  (∀ m : ℕ, m > 0 → a m * a (m + 2) = 2 * a (m + 1)) →
  SequenceProduct a (2 * m + 1) = 128 →
  m = 3 := by
  sorry


end geometric_sequence_property_l4101_410151


namespace shorter_pipe_length_l4101_410155

/-- Given a pipe of 177 inches cut into two pieces, where one piece is twice the length of the other,
    prove that the length of the shorter piece is 59 inches. -/
theorem shorter_pipe_length (total_length : ℝ) (short_length : ℝ) :
  total_length = 177 →
  total_length = short_length + 2 * short_length →
  short_length = 59 := by
  sorry

end shorter_pipe_length_l4101_410155


namespace dragon_boat_purchase_equations_l4101_410121

/-- Represents the purchase of items during the Dragon Boat Festival -/
structure DragonBoatPurchase where
  lotus_pouches : ℕ
  color_ropes : ℕ
  total_items : ℕ
  total_cost : ℕ
  lotus_price : ℕ
  rope_price : ℕ

/-- Theorem stating that the given system of equations correctly represents the purchase -/
theorem dragon_boat_purchase_equations (p : DragonBoatPurchase)
  (h1 : p.total_items = 20)
  (h2 : p.total_cost = 72)
  (h3 : p.lotus_price = 4)
  (h4 : p.rope_price = 3) :
  p.lotus_pouches + p.color_ropes = p.total_items ∧
  p.lotus_price * p.lotus_pouches + p.rope_price * p.color_ropes = p.total_cost :=
by sorry

end dragon_boat_purchase_equations_l4101_410121


namespace stating_election_cases_l4101_410196

/-- Represents the number of candidates for the election -/
def num_candidates : ℕ := 3

/-- Represents the number of positions to be filled -/
def num_positions : ℕ := 2

/-- 
Theorem stating that the number of ways to select a president and vice president 
from a group of three people, where one person cannot hold both positions, is equal to 6.
-/
theorem election_cases : 
  num_candidates * (num_candidates - 1) = 6 :=
sorry

end stating_election_cases_l4101_410196


namespace total_equipment_cost_l4101_410102

/-- The number of players on the team -/
def num_players : ℕ := 25

/-- The cost of a jersey in dollars -/
def jersey_cost : ℚ := 25

/-- The cost of shorts in dollars -/
def shorts_cost : ℚ := 15.20

/-- The cost of socks in dollars -/
def socks_cost : ℚ := 6.80

/-- The cost of cleats in dollars -/
def cleats_cost : ℚ := 40

/-- The cost of a water bottle in dollars -/
def water_bottle_cost : ℚ := 12

/-- The total cost of equipment for all players on the team -/
theorem total_equipment_cost : 
  num_players * (jersey_cost + shorts_cost + socks_cost + cleats_cost + water_bottle_cost) = 2475 := by
  sorry

end total_equipment_cost_l4101_410102


namespace expression_evaluation_l4101_410137

theorem expression_evaluation : 2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9 := by
  sorry

end expression_evaluation_l4101_410137


namespace odd_prime_divisor_condition_l4101_410159

theorem odd_prime_divisor_condition (n : ℕ) :
  (n > 0 ∧ ∀ d : ℕ, d > 0 → d ∣ n → (d + 1) ∣ (n + 1)) ↔ (Nat.Prime n ∧ n % 2 = 1) :=
by sorry

end odd_prime_divisor_condition_l4101_410159


namespace prove_arrangements_l4101_410154

def num_students : ℕ := 7

def adjacent_pair : ℕ := 1
def non_adjacent_pair : ℕ := 1
def remaining_students : ℕ := num_students - 4

def arrangements_theorem : Prop :=
  (num_students = 7) →
  (adjacent_pair = 1) →
  (non_adjacent_pair = 1) →
  (remaining_students = num_students - 4) →
  (Nat.factorial 2 * Nat.factorial 4 * (Nat.factorial 5 / Nat.factorial 3) =
   Nat.factorial 2 * Nat.factorial 4 * Nat.factorial 5 / Nat.factorial 3)

theorem prove_arrangements : arrangements_theorem := by sorry

end prove_arrangements_l4101_410154


namespace abc_inequality_l4101_410119

theorem abc_inequality : ∀ (a b c : ℕ),
  a = 20^22 → b = 21^21 → c = 22^20 → a > b ∧ b > c := by
  sorry

end abc_inequality_l4101_410119


namespace set_equality_implies_x_zero_l4101_410125

theorem set_equality_implies_x_zero (x : ℝ) : 
  ({1, x^2} : Set ℝ) = ({1, x} : Set ℝ) → x = 0 :=
by
  sorry

end set_equality_implies_x_zero_l4101_410125


namespace floor_abs_negative_34_1_l4101_410199

theorem floor_abs_negative_34_1 : ⌊|(-34.1 : ℝ)|⌋ = 34 := by sorry

end floor_abs_negative_34_1_l4101_410199


namespace expected_value_8_sided_die_l4101_410142

def standard_8_sided_die : Finset ℕ := Finset.range 8

theorem expected_value_8_sided_die :
  let outcomes := standard_8_sided_die
  let prob (n : ℕ) := if n ∈ outcomes then (1 : ℚ) / 8 else 0
  let value (n : ℕ) := n + 1
  Finset.sum outcomes (λ n ↦ prob n * value n) = (9 : ℚ) / 2 := by
  sorry

end expected_value_8_sided_die_l4101_410142


namespace reciprocal_equal_self_l4101_410143

theorem reciprocal_equal_self (x : ℝ) : x ≠ 0 ∧ x = 1 / x ↔ x = 1 ∨ x = -1 := by sorry

end reciprocal_equal_self_l4101_410143


namespace pet_shop_inventory_l4101_410134

/-- Given a pet shop with dogs, cats, and bunnies in stock, prove the total number of dogs and bunnies. -/
theorem pet_shop_inventory (dogs cats bunnies : ℕ) : 
  dogs = 112 →
  dogs / bunnies = 4 / 9 →
  dogs / cats = 4 / 7 →
  dogs + bunnies = 364 := by
  sorry

end pet_shop_inventory_l4101_410134
