import Mathlib

namespace nine_numbers_system_solution_l1862_186205

theorem nine_numbers_system_solution (n : ℕ) (S : Finset ℕ) 
  (h₁ : n ≥ 3)
  (h₂ : S ⊆ Finset.range (n^3 + 1))
  (h₃ : S.card = 3 * n^2) :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ) (x y z : ℤ),
    a₁ ∈ S ∧ a₂ ∈ S ∧ a₃ ∈ S ∧ a₄ ∈ S ∧ a₅ ∈ S ∧ a₆ ∈ S ∧ a₇ ∈ S ∧ a₈ ∈ S ∧ a₉ ∈ S ∧
    a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧ a₁ ≠ a₈ ∧ a₁ ≠ a₉ ∧
    a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧ a₂ ≠ a₈ ∧ a₂ ≠ a₉ ∧
    a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧ a₃ ≠ a₈ ∧ a₃ ≠ a₉ ∧
    a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧ a₄ ≠ a₈ ∧ a₄ ≠ a₉ ∧
    a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧ a₅ ≠ a₈ ∧ a₅ ≠ a₉ ∧
    a₆ ≠ a₇ ∧ a₆ ≠ a₈ ∧ a₆ ≠ a₉ ∧
    a₇ ≠ a₈ ∧ a₇ ≠ a₉ ∧
    a₈ ≠ a₉ ∧
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    (a₁ : ℤ) * x + (a₂ : ℤ) * y + (a₃ : ℤ) * z = 0 ∧
    (a₄ : ℤ) * x + (a₅ : ℤ) * y + (a₆ : ℤ) * z = 0 ∧
    (a₇ : ℤ) * x + (a₈ : ℤ) * y + (a₉ : ℤ) * z = 0 := by
  sorry

end nine_numbers_system_solution_l1862_186205


namespace circumradius_arithmetic_angles_max_inradius_arithmetic_sides_max_inradius_achieved_l1862_186246

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- The inradius of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- Theorem: Circumradius of triangle with arithmetic sequence angles -/
theorem circumradius_arithmetic_angles (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ k : ℝ, t.B = t.A + k ∧ t.C = t.B + k) : 
  circumradius t = 2 * Real.sqrt 3 / 3 := by sorry

/-- Theorem: Maximum inradius of triangle with arithmetic sequence sides -/
theorem max_inradius_arithmetic_sides (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ k : ℝ, t.b = t.a + k ∧ t.c = t.b + k) :
  inradius t ≤ Real.sqrt 3 / 3 := by sorry

/-- Corollary: The maximum inradius is achieved -/
theorem max_inradius_achieved (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ k : ℝ, t.b = t.a + k ∧ t.c = t.b + k) :
  ∃ t' : Triangle, t'.b = 2 ∧ (∃ k : ℝ, t'.b = t'.a + k ∧ t'.c = t'.b + k) ∧ 
  inradius t' = Real.sqrt 3 / 3 := by sorry

end circumradius_arithmetic_angles_max_inradius_arithmetic_sides_max_inradius_achieved_l1862_186246


namespace range_of_k_l1862_186279

theorem range_of_k (x k : ℝ) : 
  (x - 1) / (x - 2) = k / (x - 2) + 2 ∧ x ≥ 0 → k ≤ 3 ∧ k ≠ 1 := by
  sorry

end range_of_k_l1862_186279


namespace complex_magnitude_fourth_power_l1862_186297

theorem complex_magnitude_fourth_power : 
  Complex.abs ((1 + Complex.I * Real.sqrt 3) ^ 4) = 16 := by sorry

end complex_magnitude_fourth_power_l1862_186297


namespace cameron_wins_probability_l1862_186265

-- Define the faces of each cube
def cameron_cube : Finset Nat := {6}
def dean_cube : Finset Nat := {1, 2, 3}
def olivia_cube : Finset Nat := {3, 6}

-- Define the number of faces for each number on each cube
def cameron_faces (n : Nat) : Nat := if n = 6 then 6 else 0
def dean_faces (n : Nat) : Nat := if n ∈ dean_cube then 2 else 0
def olivia_faces (n : Nat) : Nat := if n = 3 then 4 else if n = 6 then 2 else 0

-- Define the probability of rolling less than 6 for each player
def dean_prob_less_than_6 : ℚ :=
  (dean_faces 1 + dean_faces 2 + dean_faces 3) / 6

def olivia_prob_less_than_6 : ℚ :=
  olivia_faces 3 / 6

-- Theorem statement
theorem cameron_wins_probability :
  dean_prob_less_than_6 * olivia_prob_less_than_6 = 2 / 3 := by
  sorry

end cameron_wins_probability_l1862_186265


namespace polygon_with_540_degree_sum_is_pentagon_l1862_186216

/-- A polygon with interior angles summing to 540° has 5 sides -/
theorem polygon_with_540_degree_sum_is_pentagon (n : ℕ) : 
  (n - 2) * 180 = 540 → n = 5 := by sorry

end polygon_with_540_degree_sum_is_pentagon_l1862_186216


namespace sine_function_properties_l1862_186245

/-- The function f we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := sorry

/-- The angular frequency ω -/
noncomputable def ω : ℝ := sorry

/-- The phase shift φ -/
noncomputable def φ : ℝ := sorry

/-- The constant M -/
noncomputable def M : ℝ := sorry

/-- Theorem stating the properties of f and the conclusion -/
theorem sine_function_properties :
  (∃ A : ℝ, ∀ x : ℝ, f x ≤ f A) ∧  -- A is a highest point
  (ω > 0) ∧
  (0 < φ ∧ φ < 2 * Real.pi) ∧
  (∃ B C : ℝ, B < C ∧  -- B and C are adjacent centers of symmetry
    (∀ x : ℝ, f (B + x) = f (B - x)) ∧
    (∀ x : ℝ, f (C + x) = f (C - x)) ∧
    (C - B = Real.pi / ω)) ∧
  ((C - B) * (f A) / 2 = 1 / 2) ∧  -- Area of triangle ABC is 1/2
  (M > 0 ∧ ∀ x : ℝ, f (x + M) = M * f (-x)) →  -- Functional equation
  (∀ x : ℝ, f x = -Real.sin (Real.pi * x)) := by
sorry

end sine_function_properties_l1862_186245


namespace existence_of_three_similar_numbers_l1862_186249

def is_1995_digit (n : ℕ) : Prop := n ≥ 10^1994 ∧ n < 10^1995

def composed_of_4_5_9 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 5 ∨ d = 9

def similar (a b : ℕ) : Prop :=
  ∀ d : ℕ, (d ∈ a.digits 10 ↔ d ∈ b.digits 10)

theorem existence_of_three_similar_numbers :
  ∃ (A B C : ℕ),
    is_1995_digit A ∧
    is_1995_digit B ∧
    is_1995_digit C ∧
    composed_of_4_5_9 A ∧
    composed_of_4_5_9 B ∧
    composed_of_4_5_9 C ∧
    similar A B ∧
    similar B C ∧
    similar A C ∧
    A + B = C :=
  sorry

end existence_of_three_similar_numbers_l1862_186249


namespace quadratic_no_real_roots_l1862_186248

theorem quadratic_no_real_roots (m : ℝ) : 
  (∀ x : ℝ, (m + 2) * x^2 - x + m ≠ 0) ↔ 
  (m < -1 - Real.sqrt 5 / 2 ∨ m > -1 + Real.sqrt 5 / 2) :=
by sorry

end quadratic_no_real_roots_l1862_186248


namespace z_equals_negative_four_l1862_186223

theorem z_equals_negative_four (x y z : ℤ) : x = 2 → y = x^2 - 5 → z = y^2 - 5 → z = -4 := by
  sorry

end z_equals_negative_four_l1862_186223


namespace lumberjack_problem_l1862_186251

theorem lumberjack_problem (logs_per_tree : ℕ) (firewood_per_log : ℕ) (total_firewood : ℕ) :
  logs_per_tree = 4 →
  firewood_per_log = 5 →
  total_firewood = 500 →
  (total_firewood / firewood_per_log) / logs_per_tree = 25 :=
by
  sorry

end lumberjack_problem_l1862_186251


namespace total_air_conditioner_sales_l1862_186293

theorem total_air_conditioner_sales (june_sales : ℕ) (july_increase : ℚ) : 
  june_sales = 96 →
  july_increase = 1/3 →
  june_sales + (june_sales * (1 + july_increase)).floor = 224 := by
  sorry

end total_air_conditioner_sales_l1862_186293


namespace cassandra_apple_pie_l1862_186228

/-- The number of apples in each slice of pie -/
def apples_per_slice (total_apples : ℕ) (num_pies : ℕ) (slices_per_pie : ℕ) : ℚ :=
  (total_apples : ℚ) / (num_pies * slices_per_pie)

/-- Cassandra's apple pie problem -/
theorem cassandra_apple_pie :
  let total_apples : ℕ := 4 * 12  -- 4 dozen
  let num_pies : ℕ := 4
  let slices_per_pie : ℕ := 6
  apples_per_slice total_apples num_pies slices_per_pie = 2 := by
sorry

end cassandra_apple_pie_l1862_186228


namespace infinitely_many_nondivisible_l1862_186208

theorem infinitely_many_nondivisible (a b : ℕ) : 
  Set.Infinite {n : ℕ | ¬(n^b + 1 ∣ a^n + 1)} := by
sorry

end infinitely_many_nondivisible_l1862_186208


namespace power_24_in_terms_of_P_l1862_186206

theorem power_24_in_terms_of_P (a b : ℕ) (P : ℝ) (h_P : P = 2^a) : 24^(a*b) = P^(3*b) * 3^(a*b) := by
  sorry

end power_24_in_terms_of_P_l1862_186206


namespace passengers_at_terminal_l1862_186253

/-- Represents the number of stations on the bus route. -/
def num_stations : ℕ := 8

/-- Represents the number of people who boarded the bus at the first 6 stations. -/
def passengers_boarded : ℕ := 100

/-- Represents the number of people who got off at all stations except the terminal station. -/
def passengers_got_off : ℕ := 80

/-- Theorem stating that the number of passengers who boarded at the first 6 stations
    and got off at the terminal station is 20. -/
theorem passengers_at_terminal : ℕ := by
  sorry

#check passengers_at_terminal

end passengers_at_terminal_l1862_186253


namespace smallest_group_size_fifty_nine_satisfies_conditions_fewest_students_l1862_186204

theorem smallest_group_size (N : ℕ) : 
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 8 = 4) → N ≥ 59 :=
by sorry

theorem fifty_nine_satisfies_conditions : 
  (59 % 5 = 2) ∧ (59 % 6 = 3) ∧ (59 % 8 = 4) :=
by sorry

theorem fewest_students : 
  ∃ (N : ℕ), (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 8 = 4) ∧ 
  (∀ (M : ℕ), (M % 5 = 2) ∧ (M % 6 = 3) ∧ (M % 8 = 4) → M ≥ N) ∧
  N = 59 :=
by sorry

end smallest_group_size_fifty_nine_satisfies_conditions_fewest_students_l1862_186204


namespace bag_properties_l1862_186292

/-- A bag containing colored balls -/
structure Bag where
  red : ℕ
  black : ℕ
  white : ℕ

/-- The scoring system for the balls -/
def score (color : String) : ℕ :=
  match color with
  | "white" => 2
  | "black" => 1
  | "red" => 0
  | _ => 0

/-- The theorem stating the properties of the bag and the probabilities -/
theorem bag_properties (b : Bag) : 
  b.red = 1 ∧ b.black = 1 ∧ b.white = 2 →
  (b.white : ℚ) / (b.red + b.black + b.white : ℚ) = 1/2 ∧
  (2 : ℚ) / ((b.red + b.black + b.white) * (b.red + b.black + b.white - 1) : ℚ) = 1/3 :=
by sorry

#check bag_properties

end bag_properties_l1862_186292


namespace triangle_inequality_proof_l1862_186286

/-- A structure representing a set of three line segments. -/
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The triangle inequality theorem for a set of line segments. -/
def satisfies_triangle_inequality (s : LineSegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

/-- The set of line segments that can form a triangle. -/
def triangle_set : LineSegmentSet :=
  { a := 3, b := 4, c := 5 }

/-- The sets of line segments that cannot form triangles. -/
def non_triangle_sets : List LineSegmentSet :=
  [{ a := 1, b := 2, c := 3 },
   { a := 4, b := 5, c := 10 },
   { a := 6, b := 9, c := 2 }]

theorem triangle_inequality_proof :
  satisfies_triangle_inequality triangle_set ∧
  ∀ s ∈ non_triangle_sets, ¬satisfies_triangle_inequality s :=
sorry

end triangle_inequality_proof_l1862_186286


namespace divisibility_by_24_l1862_186233

theorem divisibility_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  24 ∣ p^2 - 1 := by
  sorry

end divisibility_by_24_l1862_186233


namespace sin_330_degrees_l1862_186220

theorem sin_330_degrees : 
  Real.sin (330 * π / 180) = -(1/2) := by
  sorry

end sin_330_degrees_l1862_186220


namespace find_a_l1862_186225

theorem find_a : ∃ a : ℚ, (a + 3) / 4 = (2 * a - 3) / 7 + 1 → a = 5 := by
  sorry

end find_a_l1862_186225


namespace continuous_midpoint_property_implies_affine_l1862_186284

open Real

/-- A function satisfying the midpoint property -/
def HasMidpointProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y, f ((x + y) / 2) = (f x + f y) / 2

/-- The main theorem stating that a continuous function with the midpoint property is affine -/
theorem continuous_midpoint_property_implies_affine
  (f : ℝ → ℝ) (hf : Continuous f) (hm : HasMidpointProperty f) :
  ∃ c b : ℝ, ∀ x, f x = c * x + b := by
  sorry

end continuous_midpoint_property_implies_affine_l1862_186284


namespace smallest_x_value_l1862_186280

theorem smallest_x_value (x y : ℕ+) (h : (3 : ℚ) / 5 = y / (468 + x)) : 2 ≤ x := by
  sorry

end smallest_x_value_l1862_186280


namespace handbag_price_l1862_186281

/-- Calculates the total selling price of a product given its original price, discount rate, and tax rate. -/
def totalSellingPrice (originalPrice : ℝ) (discountRate : ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  discountedPrice * (1 + taxRate)

/-- Theorem stating that the total selling price of a $100 product with 30% discount and 8% tax is $75.6 -/
theorem handbag_price : 
  totalSellingPrice 100 0.3 0.08 = 75.6 := by
  sorry

end handbag_price_l1862_186281


namespace f_properties_l1862_186236

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (1/3) * x^3 + ((1-a)/2) * x^2 - a^2 * Real.log x + a^2 * Real.log a

theorem f_properties (a : ℝ) (h : a > 0) :
  (∀ x > 0, f 1 x ≥ 1/3 ∧ f 1 1 = 1/3) ∧
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ a > 3 :=
by sorry

end f_properties_l1862_186236


namespace function_values_l1862_186219

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else x^2

theorem function_values (a : ℝ) : f (-1) = 2 * f a → a = Real.sqrt 3 ∨ a = -Real.sqrt 2 / 2 := by
  sorry

end function_values_l1862_186219


namespace expected_pairs_for_given_deck_l1862_186240

/-- Represents a deck of cards with numbered pairs and Joker pairs -/
structure Deck :=
  (num_pairs : ℕ)
  (joker_pairs : ℕ)

/-- Calculates the expected number of complete pairs when drawing until a Joker pair is found -/
def expected_complete_pairs (d : Deck) : ℚ :=
  (d.num_pairs : ℚ) / 3 + 1

theorem expected_pairs_for_given_deck :
  let d : Deck := ⟨7, 2⟩
  expected_complete_pairs d = 10 / 3 := by sorry

end expected_pairs_for_given_deck_l1862_186240


namespace ab_geq_2_sufficient_not_necessary_l1862_186212

theorem ab_geq_2_sufficient_not_necessary :
  (∀ a b : ℝ, a * b ≥ 2 → a^2 + b^2 ≥ 4) ∧
  (∃ a b : ℝ, a^2 + b^2 ≥ 4 ∧ a * b < 2) := by
  sorry

end ab_geq_2_sufficient_not_necessary_l1862_186212


namespace crossword_puzzle_subset_l1862_186215

def is_three_identical_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ ∃ d, n = d * 100 + d * 10 + d

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def has_three_middle_threes (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ ∃ a b, n = a * 10000 + 3 * 1000 + 3 * 100 + 3 * 10 + b

theorem crossword_puzzle_subset :
  ∀ x y z : ℕ,
  is_three_identical_digits x →
  y = x^2 →
  digit_sum z = 18 →
  has_three_middle_threes z →
  x = 111 ∧ y = 12321 ∧ z = 33333 :=
by sorry

end crossword_puzzle_subset_l1862_186215


namespace expand_expression_l1862_186289

theorem expand_expression (x : ℝ) : -3*x*(x^2 - x - 2) = -3*x^3 + 3*x^2 + 6*x := by
  sorry

end expand_expression_l1862_186289


namespace prob_two_sunny_days_value_l1862_186226

/-- The probability of exactly 2 sunny days out of 5 days, where each day has a 75% chance of rain -/
def prob_two_sunny_days : ℚ :=
  (Nat.choose 5 2 : ℚ) * (1/4)^2 * (3/4)^3

/-- The main theorem stating that the probability is equal to 135/512 -/
theorem prob_two_sunny_days_value : prob_two_sunny_days = 135/512 := by
  sorry

end prob_two_sunny_days_value_l1862_186226


namespace ellipse_tangent_to_lines_l1862_186296

/-- The first line tangent to the ellipse -/
def line1 (x y : ℝ) : Prop := x + 2*y = 27

/-- The second line tangent to the ellipse -/
def line2 (x y : ℝ) : Prop := 7*x + 4*y = 81

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := 162*x^2 + 81*y^2 = 13122

/-- Theorem stating that the given ellipse equation is tangent to both lines -/
theorem ellipse_tangent_to_lines :
  ∀ x y : ℝ, line1 x y ∨ line2 x y → ellipse_equation x y := by sorry

end ellipse_tangent_to_lines_l1862_186296


namespace book_distribution_l1862_186217

theorem book_distribution (n : ℕ) (b : ℕ) : 
  (3 * n + 6 = b) →                     -- Condition 1
  (5 * n - 5 ≤ b) →                     -- Condition 2 (lower bound)
  (b < 5 * n - 2) →                     -- Condition 2 (upper bound)
  (n = 5 ∧ b = 21) :=                   -- Conclusion
by sorry

end book_distribution_l1862_186217


namespace sector_inscribed_circle_area_ratio_l1862_186247

/-- 
Given a sector with a central angle of 120°, 
the ratio of the area of the sector to the area of its inscribed circle is (7 + 4√3) / 9.
-/
theorem sector_inscribed_circle_area_ratio :
  ∀ R r : ℝ,
  R > 0 → r > 0 →
  r / (R - r) = Real.sqrt 3 / 2 →
  (1/3 * π * R^2) / (π * r^2) = (7 + 4 * Real.sqrt 3) / 9 :=
by sorry

end sector_inscribed_circle_area_ratio_l1862_186247


namespace three_fifths_of_difference_l1862_186237

theorem three_fifths_of_difference : (3 : ℚ) / 5 * ((7 * 9) - (4 * 3)) = 153 / 5 := by
  sorry

end three_fifths_of_difference_l1862_186237


namespace sine_inequality_l1862_186227

theorem sine_inequality (x : ℝ) : 
  (∃ k : ℤ, (π / 6 + k * π < x ∧ x < π / 2 + k * π) ∨ 
            (5 * π / 6 + k * π < x ∧ x < 3 * π / 2 + k * π)) ↔ 
  (Real.sin x)^2 + (Real.sin (2 * x))^2 > (Real.sin (3 * x))^2 := by
  sorry

end sine_inequality_l1862_186227


namespace alex_final_silver_tokens_l1862_186258

/-- Represents the number of tokens Alex has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents a token exchange booth -/
structure Booth where
  red_in : ℕ
  blue_in : ℕ
  red_out : ℕ
  blue_out : ℕ
  silver_out : ℕ

/-- Applies a single exchange at a booth -/
def apply_exchange (tokens : TokenCount) (booth : Booth) : TokenCount :=
  { red := tokens.red - booth.red_in + booth.red_out,
    blue := tokens.blue - booth.blue_in + booth.blue_out,
    silver := tokens.silver + booth.silver_out }

/-- Checks if an exchange is possible -/
def can_exchange (tokens : TokenCount) (booth : Booth) : Prop :=
  tokens.red ≥ booth.red_in ∧ tokens.blue ≥ booth.blue_in

/-- The final state after all possible exchanges -/
def final_state (initial : TokenCount) (booth1 booth2 : Booth) : TokenCount :=
  sorry  -- The implementation would go here

/-- Theorem stating that Alex will end up with 58 silver tokens -/
theorem alex_final_silver_tokens :
  let initial := TokenCount.mk 100 50 0
  let booth1 := Booth.mk 3 0 0 1 2
  let booth2 := Booth.mk 0 4 1 0 1
  (final_state initial booth1 booth2).silver = 58 := by
  sorry


end alex_final_silver_tokens_l1862_186258


namespace total_selling_price_calculation_craig_appliance_sales_l1862_186254

/-- Calculates the total selling price of appliances given commission details --/
theorem total_selling_price_calculation 
  (fixed_commission : ℝ) 
  (variable_commission_rate : ℝ) 
  (num_appliances : ℕ) 
  (total_commission : ℝ) : ℝ :=
  let total_fixed_commission := fixed_commission * num_appliances
  let variable_commission := total_commission - total_fixed_commission
  variable_commission / variable_commission_rate

/-- Proves that the total selling price is $3620 given the problem conditions --/
theorem craig_appliance_sales : 
  total_selling_price_calculation 50 0.1 6 662 = 3620 := by
  sorry

end total_selling_price_calculation_craig_appliance_sales_l1862_186254


namespace min_value_zero_l1862_186229

open Real

/-- The quadratic expression in x and y with parameter c -/
def f (c x y : ℝ) : ℝ :=
  3 * x^2 - 4 * c * x * y + (2 * c^2 + 1) * y^2 - 6 * x - 3 * y + 5

/-- The theorem stating the condition for minimum value of f to be 0 -/
theorem min_value_zero (c : ℝ) :
  (∀ x y : ℝ, f c x y ≥ 0) ∧ (∃ x y : ℝ, f c x y = 0) ↔ c = 2/3 := by
  sorry

end min_value_zero_l1862_186229


namespace unique_fraction_decomposition_l1862_186277

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_not_two : p ≠ 2) :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ (2 : ℚ) / p = 1 / x + 1 / y ∧ 
  x = (p^2 + p) / 2 ∧ y = (p + 1) / 2 := by
  sorry

end unique_fraction_decomposition_l1862_186277


namespace initial_state_is_winning_starting_player_wins_starting_player_always_wins_l1862_186231

/-- Represents a pile of matches -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Checks if a game state is a winning position for the current player -/
def isWinningPosition (state : GameState) : Prop :=
  ∃ (n m : Nat), n < m ∧
    ∃ (a b c : Nat), 
      state.piles = [Pile.mk (2^n * a), Pile.mk (2^n * b), Pile.mk (2^m * c)] ∧
      Odd a ∧ Odd b ∧ Odd c

/-- The initial game state -/
def initialState : GameState :=
  { piles := [Pile.mk 100, Pile.mk 200, Pile.mk 300] }

/-- Theorem stating that the initial state is a winning position -/
theorem initial_state_is_winning : isWinningPosition initialState := by
  sorry

/-- Theorem stating that the starting player has a winning strategy -/
theorem starting_player_wins (state : GameState) :
  isWinningPosition state → ∃ (nextState : GameState), 
    (∃ (move : GameState → GameState), nextState = move state) ∧
    ¬isWinningPosition nextState := by
  sorry

/-- Main theorem: The starting player wins with correct play -/
theorem starting_player_always_wins : 
  ∃ (strategy : GameState → GameState), 
    ∀ (state : GameState), 
      isWinningPosition state → 
      ¬isWinningPosition (strategy state) := by
  sorry

end initial_state_is_winning_starting_player_wins_starting_player_always_wins_l1862_186231


namespace luxury_car_price_l1862_186266

def initial_price : ℝ := 80000

def discounts : List ℝ := [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price : ℝ := discounts.foldl apply_discount initial_price

theorem luxury_car_price : final_price = 24418.80 := by
  sorry

end luxury_car_price_l1862_186266


namespace min_square_side_for_9x21_l1862_186224

/-- The minimum side length of a square that can contain 9x21 rectangles without rotation and overlap -/
def min_square_side (width : ℕ) (length : ℕ) : ℕ :=
  Nat.lcm width length

/-- Theorem stating that the minimum side length for 9x21 rectangles is 63 -/
theorem min_square_side_for_9x21 :
  min_square_side 9 21 = 63 := by sorry

end min_square_side_for_9x21_l1862_186224


namespace even_decreasing_function_ordering_l1862_186298

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def monotone_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x

theorem even_decreasing_function_ordering (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : monotone_decreasing_on_pos f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
sorry

end even_decreasing_function_ordering_l1862_186298


namespace triangle_count_l1862_186250

/-- The number of points on the circumference of the circle -/
def n : ℕ := 7

/-- The number of points needed to form a triangle -/
def k : ℕ := 3

/-- The number of different triangles that can be formed -/
def num_triangles : ℕ := Nat.choose n k

theorem triangle_count : num_triangles = 35 := by sorry

end triangle_count_l1862_186250


namespace f_decreasing_range_l1862_186269

/-- A piecewise function f(x) defined on the real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

/-- The theorem stating the range of 'a' for which f is decreasing on ℝ. -/
theorem f_decreasing_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) ↔ 1/7 ≤ a ∧ a < 1/3 :=
sorry

end f_decreasing_range_l1862_186269


namespace calculator_result_l1862_186274

def special_key (x : ℚ) : ℚ := 1 / (1 - x)

def apply_n_times (f : ℚ → ℚ) (x : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (apply_n_times f x n)

theorem calculator_result : apply_n_times special_key 3 50 = 2/3 := by
  sorry

end calculator_result_l1862_186274


namespace pair_one_six_least_restricted_l1862_186259

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a license plate ending pair -/
structure LicensePlatePair :=
  (first : Nat)
  (second : Nat)

/-- The restriction schedule for each license plate ending pair -/
def restrictionSchedule : LicensePlatePair → List DayOfWeek
  | ⟨1, 6⟩ => [DayOfWeek.Monday, DayOfWeek.Tuesday]
  | ⟨2, 7⟩ => [DayOfWeek.Tuesday, DayOfWeek.Wednesday]
  | ⟨3, 8⟩ => [DayOfWeek.Wednesday, DayOfWeek.Thursday]
  | ⟨4, 9⟩ => [DayOfWeek.Thursday, DayOfWeek.Friday]
  | ⟨5, 0⟩ => [DayOfWeek.Friday, DayOfWeek.Monday]
  | _ => []

/-- Calculate the number of restricted days for a given license plate pair in January 2014 -/
def restrictedDays (pair : LicensePlatePair) : Nat :=
  sorry

/-- All possible license plate ending pairs -/
def allPairs : List LicensePlatePair :=
  [⟨1, 6⟩, ⟨2, 7⟩, ⟨3, 8⟩, ⟨4, 9⟩, ⟨5, 0⟩]

/-- Theorem: The license plate pair (1,6) has the fewest restricted days in January 2014 -/
theorem pair_one_six_least_restricted :
  ∀ pair ∈ allPairs, restrictedDays ⟨1, 6⟩ ≤ restrictedDays pair := by
  sorry

end pair_one_six_least_restricted_l1862_186259


namespace triangle_is_acute_l1862_186232

theorem triangle_is_acute (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C)
  (h_ratio : (Real.sin A + Real.sin B) / (Real.sin B + Real.sin C) = 9 / 11 ∧
             (Real.sin B + Real.sin C) / (Real.sin C + Real.sin A) = 11 / 10) :
  A < π/2 ∧ B < π/2 ∧ C < π/2 := by
  sorry

end triangle_is_acute_l1862_186232


namespace books_lost_during_move_phil_books_lost_l1862_186268

theorem books_lost_during_move (initial_books : ℕ) (pages_per_book : ℕ) (pages_left : ℕ) : ℕ :=
  let total_pages := initial_books * pages_per_book
  let pages_lost := total_pages - pages_left
  pages_lost / pages_per_book

theorem phil_books_lost :
  books_lost_during_move 10 100 800 = 2 := by
  sorry

end books_lost_during_move_phil_books_lost_l1862_186268


namespace solve_for_z_l1862_186261

theorem solve_for_z (x y z : ℚ) 
  (h1 : x = 11)
  (h2 : y = 8)
  (h3 : 2 * x + 3 * z = 5 * y) :
  z = 6 := by
sorry

end solve_for_z_l1862_186261


namespace equation_solution_exists_l1862_186278

theorem equation_solution_exists (m n : ℤ) :
  ∃ (w x y z : ℤ), w + x + 2*y + 2*z = m ∧ 2*w - 2*x + y - z = n := by
  sorry

end equation_solution_exists_l1862_186278


namespace fibonacci_sum_identity_l1862_186218

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_sum_identity (n m : ℕ) (h1 : n ≥ 1) (h2 : m ≥ 0) :
  fib (n + m) = fib (n - 1) * fib m + fib n * fib (m + 1) := by
  sorry

end fibonacci_sum_identity_l1862_186218


namespace negation_of_existential_proposition_l1862_186291

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by
  sorry

end negation_of_existential_proposition_l1862_186291


namespace james_singing_lessons_l1862_186295

/-- Calculates the number of singing lessons James gets given the conditions --/
def number_of_lessons (lesson_cost : ℕ) (james_payment : ℕ) : ℕ :=
  let total_cost := james_payment * 2
  let initial_paid_lessons := 10
  let remaining_cost := total_cost - (initial_paid_lessons * lesson_cost)
  let additional_paid_lessons := remaining_cost / (lesson_cost * 2)
  1 + initial_paid_lessons + additional_paid_lessons

/-- Theorem stating that James gets 13 singing lessons --/
theorem james_singing_lessons :
  number_of_lessons 5 35 = 13 := by
  sorry


end james_singing_lessons_l1862_186295


namespace polynomial_mapping_l1862_186202

def polynomial_equation (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) : Prop :=
  ∀ x, x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄ = (x+1)^4 + b₁*(x+1)^3 + b₂*(x+1)^2 + b₃*(x+1) + b₄

def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let b₁ := 0
  let b₂ := -3
  let b₃ := 4
  let b₄ := -1
  (b₁, b₂, b₃, b₄)

theorem polynomial_mapping :
  polynomial_equation 4 3 2 1 0 (-3) 4 (-1) → f 4 3 2 1 = (0, -3, 4, -1) :=
by sorry

end polynomial_mapping_l1862_186202


namespace max_square_side_is_40_l1862_186210

def distances_L : List ℕ := [2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6, 2]
def distances_P : List ℕ := [3, 1, 2, 6, 3, 1, 2, 6, 3, 1, 2, 6, 3, 1]

def max_square_side_length (L : List ℕ) (P : List ℕ) : ℕ := sorry

theorem max_square_side_is_40 :
  max_square_side_length distances_L distances_P = 40 := by sorry

end max_square_side_is_40_l1862_186210


namespace cart_distance_l1862_186201

/-- The distance traveled by a cart with three wheels of different circumferences -/
theorem cart_distance (front_circ rear_circ third_circ : ℕ)
  (h1 : front_circ = 30)
  (h2 : rear_circ = 32)
  (h3 : third_circ = 34)
  (rev_rear : ℕ)
  (h4 : front_circ * (rev_rear + 5) = rear_circ * rev_rear)
  (h5 : third_circ * (rev_rear - 8) = rear_circ * rev_rear) :
  rear_circ * rev_rear = 2400 :=
sorry

end cart_distance_l1862_186201


namespace same_rate_different_time_l1862_186213

/-- Given that a person drives 150 miles in 3 hours, 
    prove that another person driving at the same rate for 4 hours will cover 200 miles. -/
theorem same_rate_different_time (distance₁ : ℝ) (time₁ : ℝ) (time₂ : ℝ) 
  (h₁ : distance₁ = 150) 
  (h₂ : time₁ = 3) 
  (h₃ : time₂ = 4) : 
  (distance₁ / time₁) * time₂ = 200 := by
  sorry

end same_rate_different_time_l1862_186213


namespace simplify_expression_exponent_calculation_l1862_186221

-- Part 1
theorem simplify_expression (x : ℝ) : 
  (-2*x)^3 * x^2 + (3*x^4)^2 / x^3 = x^5 := by sorry

-- Part 2
theorem exponent_calculation (a m n : ℝ) 
  (hm : a^m = 2) (hn : a^n = 3) : a^(m+2*n) = 18 := by sorry

end simplify_expression_exponent_calculation_l1862_186221


namespace sqrt_x_plus_y_equals_three_l1862_186276

theorem sqrt_x_plus_y_equals_three (x y : ℝ) (h : y = 4 + Real.sqrt (5 - x) + Real.sqrt (x - 5)) : 
  Real.sqrt (x + y) = 3 := by
  sorry

end sqrt_x_plus_y_equals_three_l1862_186276


namespace points_three_units_from_negative_two_l1862_186243

theorem points_three_units_from_negative_two :
  ∃! (S : Set ℝ), (∀ x ∈ S, |x - (-2)| = 3) ∧ S = {-5, 1} := by
  sorry

end points_three_units_from_negative_two_l1862_186243


namespace phi_function_form_l1862_186299

/-- A direct proportion function -/
def DirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, m ≠ 0 ∧ ∀ x, f x = m * x

/-- An inverse proportion function -/
def InverseProportion (g : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, n ≠ 0 ∧ ∀ x, x ≠ 0 → g x = n / x

/-- The main theorem -/
theorem phi_function_form (f g : ℝ → ℝ) (φ : ℝ → ℝ) :
  DirectProportion f →
  InverseProportion g →
  (∀ x, φ x = f x + g x) →
  φ 1 = 8 →
  (∃ x, φ x = 16) →
  ∀ x, x ≠ 0 → φ x = 3 * x + 5 / x := by
  sorry

end phi_function_form_l1862_186299


namespace inequality_solution_set_l1862_186262

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 2) > (5/x) + (21/10)) ↔ (-2 < x ∧ x < 0) :=
by sorry

end inequality_solution_set_l1862_186262


namespace money_distribution_l1862_186270

theorem money_distribution (a b c : ℕ) 
  (h1 : a + b + c = 1000)
  (h2 : a + c = 700)
  (h3 : b + c = 600) :
  c = 300 := by
  sorry

end money_distribution_l1862_186270


namespace yoongi_position_l1862_186239

/-- Calculates the number of students behind a runner after passing others. -/
def students_behind (total : ℕ) (initial_position : ℕ) (passed : ℕ) : ℕ :=
  total - (initial_position - passed)

/-- Theorem stating the number of students behind Yoongi after passing others. -/
theorem yoongi_position (total : ℕ) (initial_position : ℕ) (passed : ℕ) 
  (h_total : total = 9)
  (h_initial : initial_position = 7)
  (h_passed : passed = 4) :
  students_behind total initial_position passed = 6 := by
sorry

end yoongi_position_l1862_186239


namespace log_sum_equals_one_implies_product_equals_ten_l1862_186244

theorem log_sum_equals_one_implies_product_equals_ten (a b : ℝ) (h : Real.log a + Real.log b = 1) : a * b = 10 := by
  sorry

end log_sum_equals_one_implies_product_equals_ten_l1862_186244


namespace divisor_inequality_l1862_186209

theorem divisor_inequality (n : ℕ) (a b c d : ℕ) : 
  (1 < a) → (a < b) → (b < c) → (c < d) → (d < n) →
  (∀ k : ℕ, k ∣ n → (k = 1 ∨ k = a ∨ k = b ∨ k = c ∨ k = d ∨ k = n)) →
  (a ∣ n) → (b ∣ n) → (c ∣ n) → (d ∣ n) →
  b - a ≤ d - c := by
  sorry

#check divisor_inequality

end divisor_inequality_l1862_186209


namespace closest_fraction_l1862_186288

def medals_won : ℚ := 23 / 150

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction (closest : ℚ) :
  closest ∈ options ∧
  ∀ x ∈ options, |medals_won - closest| ≤ |medals_won - x| :=
by sorry

end closest_fraction_l1862_186288


namespace tan_five_pi_over_four_l1862_186241

theorem tan_five_pi_over_four : Real.tan (5 * π / 4) = 1 := by
  sorry

end tan_five_pi_over_four_l1862_186241


namespace parallel_vectors_t_value_l1862_186200

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_t_value :
  ∀ t : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (4, t)
  parallel a b → t = 2 := by
sorry

end parallel_vectors_t_value_l1862_186200


namespace pear_seed_average_l1862_186234

theorem pear_seed_average (total_seeds : ℕ) (apple_seeds : ℕ) (grape_seeds : ℕ)
  (num_apples : ℕ) (num_pears : ℕ) (num_grapes : ℕ) (seeds_needed : ℕ) :
  total_seeds = 60 →
  apple_seeds = 6 →
  grape_seeds = 3 →
  num_apples = 4 →
  num_pears = 3 →
  num_grapes = 9 →
  seeds_needed = 3 →
  ∃ (pear_seeds : ℕ), pear_seeds = 2 ∧
    num_apples * apple_seeds + num_pears * pear_seeds + num_grapes * grape_seeds = total_seeds - seeds_needed :=
by sorry

end pear_seed_average_l1862_186234


namespace tangent_slope_three_points_l1862_186230

theorem tangent_slope_three_points (x y : ℝ) : 
  y = x^3 ∧ (3 * x^2 = 3) → (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
sorry

end tangent_slope_three_points_l1862_186230


namespace f_value_at_3_l1862_186252

theorem f_value_at_3 (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x^2 + b * x + 2) →
  f 1 = 3 →
  f 2 = 12 →
  f 3 = 29 := by
sorry

end f_value_at_3_l1862_186252


namespace complex_number_opposites_l1862_186275

theorem complex_number_opposites (b : ℝ) : 
  (Complex.re ((2 - b * Complex.I) * Complex.I) = 
   -Complex.im ((2 - b * Complex.I) * Complex.I)) → b = -2 := by
sorry

end complex_number_opposites_l1862_186275


namespace quadratic_with_inequality_has_negative_root_l1862_186260

/-- A quadratic polynomial with two distinct roots satisfying a specific inequality has at least one negative root. -/
theorem quadratic_with_inequality_has_negative_root 
  (f : ℝ → ℝ) 
  (h_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (h_distinct_roots : ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ f r₁ = 0 ∧ f r₂ = 0)
  (h_inequality : ∀ a b : ℝ, f (a^2 + b^2) ≥ f (2*a*b)) :
  ∃ r : ℝ, f r = 0 ∧ r < 0 :=
sorry

end quadratic_with_inequality_has_negative_root_l1862_186260


namespace exists_painted_subpolygon_l1862_186222

/-- Represents a convex polygon --/
structure ConvexPolygon where
  -- Add necessary fields

/-- Represents a diagonal of a polygon --/
structure Diagonal where
  -- Add necessary fields

/-- Represents a subpolygon formed by diagonals --/
structure Subpolygon where
  -- Add necessary fields

/-- A function to check if a subpolygon is entirely painted on the outside --/
def is_entirely_painted_outside (sp : Subpolygon) : Prop :=
  sorry

/-- The main theorem --/
theorem exists_painted_subpolygon 
  (P : ConvexPolygon) 
  (sides_painted_outside : Prop) 
  (diagonals : List Diagonal)
  (no_three_intersect : Prop)
  (diagonals_painted_one_side : Prop) :
  ∃ (sp : Subpolygon), is_entirely_painted_outside sp :=
sorry

end exists_painted_subpolygon_l1862_186222


namespace intersection_equals_N_l1862_186263

def M : Set ℝ := {x | x ≤ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem intersection_equals_N : M ∩ N = N := by sorry

end intersection_equals_N_l1862_186263


namespace books_combination_l1862_186242

def choose (n : ℕ) (r : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

theorem books_combination : choose 15 3 = 455 := by
  sorry

end books_combination_l1862_186242


namespace fuel_food_ratio_l1862_186235

theorem fuel_food_ratio 
  (fuel_cost : ℝ) 
  (distance_per_tank : ℝ) 
  (total_distance : ℝ) 
  (total_spent : ℝ) 
  (h1 : fuel_cost = 45)
  (h2 : distance_per_tank = 500)
  (h3 : total_distance = 2000)
  (h4 : total_spent = 288) :
  (total_spent - (total_distance / distance_per_tank * fuel_cost)) / 
  (total_distance / distance_per_tank * fuel_cost) = 3 / 5 := by
sorry


end fuel_food_ratio_l1862_186235


namespace center_is_five_l1862_186211

/-- Represents a 3x3 grid with numbers from 1 to 9 --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two positions in the grid are adjacent --/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ q.2.val + 1 = p.2.val)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ q.1.val + 1 = p.1.val))

/-- Checks if two numbers are consecutive --/
def consecutive (m n : Fin 9) : Prop :=
  m.val + 1 = n.val ∨ n.val + 1 = m.val

/-- Main theorem --/
theorem center_is_five (g : Grid) : 
  (∀ i j, g i j ≠ g i j → False) → -- Each number is used once
  (∀ i j k l, consecutive (g i j) (g k l) → adjacent (i, j) (k, l)) → -- Consecutive numbers are adjacent
  g 0 0 = 1 → g 0 2 = 3 → g 2 0 = 5 → g 2 2 = 7 → -- Corner numbers are 2, 4, 6, 8
  g 1 1 = 4 -- Center is 5
  := by sorry

end center_is_five_l1862_186211


namespace problem_solution_l1862_186271

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.cos y = 3005)
  (h2 : x + 3005 * Real.sin y = 3004)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 3004 := by
sorry

end problem_solution_l1862_186271


namespace quadratic_equation_roots_l1862_186282

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2 * x₁ + 3 = 0 ∧ a * x₂^2 + 2 * x₂ + 3 = 0) →
  a = -2 :=
by sorry

end quadratic_equation_roots_l1862_186282


namespace factorization_equality_l1862_186290

theorem factorization_equality (y : ℝ) : 3 * y * (y - 5) + 4 * (y - 5) = (3 * y + 4) * (y - 5) := by
  sorry

end factorization_equality_l1862_186290


namespace figure_with_perimeter_91_has_11_tiles_l1862_186255

/-- Represents a figure in the sequence --/
structure Figure where
  tiles : ℕ
  perimeter : ℕ

/-- The side length of each equilateral triangle tile in cm --/
def tileSideLength : ℕ := 7

/-- The first figure in the sequence --/
def firstFigure : Figure :=
  { tiles := 1
  , perimeter := 3 * tileSideLength }

/-- Generates the next figure in the sequence --/
def nextFigure (f : Figure) : Figure :=
  { tiles := f.tiles + 1
  , perimeter := f.perimeter + tileSideLength }

/-- Theorem: The figure with perimeter 91 cm consists of 11 tiles --/
theorem figure_with_perimeter_91_has_11_tiles :
  ∃ (n : ℕ), (n.iterate nextFigure firstFigure).perimeter = 91 ∧
             (n.iterate nextFigure firstFigure).tiles = 11 := by
  sorry

end figure_with_perimeter_91_has_11_tiles_l1862_186255


namespace toddler_count_problem_l1862_186294

/-- The actual number of toddlers given Bill's count and errors -/
def actual_toddler_count (counted : ℕ) (double_counted : ℕ) (hidden : ℕ) : ℕ :=
  counted - double_counted + hidden

/-- Theorem stating the actual number of toddlers in the given scenario -/
theorem toddler_count_problem : 
  actual_toddler_count 34 10 4 = 28 := by
  sorry

end toddler_count_problem_l1862_186294


namespace closest_approximation_l1862_186207

def x_values : List ℝ := [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

def y (x : ℝ) : ℝ := x^2 - x

def distance_to_target (x : ℝ) : ℝ := |y x - 1.4|

theorem closest_approximation :
  ∀ x ∈ x_values, distance_to_target 1.8 ≤ distance_to_target x := by
  sorry

end closest_approximation_l1862_186207


namespace can_measure_four_liters_l1862_186238

/-- Represents the state of water in the buckets -/
structure BucketState :=
  (small : ℕ)  -- Amount of water in the 3-liter bucket
  (large : ℕ)  -- Amount of water in the 5-liter bucket

/-- Represents the possible operations on the buckets -/
inductive BucketOperation
  | FillSmall
  | FillLarge
  | EmptySmall
  | EmptyLarge
  | PourSmallToLarge
  | PourLargeToSmall

/-- Applies a single operation to a bucket state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  match op with
  | BucketOperation.FillSmall => { small := 3, large := state.large }
  | BucketOperation.FillLarge => { small := state.small, large := 5 }
  | BucketOperation.EmptySmall => { small := 0, large := state.large }
  | BucketOperation.EmptyLarge => { small := state.small, large := 0 }
  | BucketOperation.PourSmallToLarge =>
      let amount := min state.small (5 - state.large)
      { small := state.small - amount, large := state.large + amount }
  | BucketOperation.PourLargeToSmall =>
      let amount := min state.large (3 - state.small)
      { small := state.small + amount, large := state.large - amount }

/-- Theorem: It is possible to measure exactly 4 liters using buckets of 3 and 5 liters -/
theorem can_measure_four_liters : ∃ (ops : List BucketOperation), 
  let final_state := ops.foldl applyOperation { small := 0, large := 0 }
  final_state.small + final_state.large = 4 := by
  sorry

end can_measure_four_liters_l1862_186238


namespace major_axis_length_l1862_186285

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- State the theorem
theorem major_axis_length :
  ∃ (a b : ℝ), a > b ∧ a > 0 ∧ b > 0 ∧
  (∀ x y, ellipse_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  2 * a = 6 :=
sorry

end major_axis_length_l1862_186285


namespace exists_increasing_perfect_squares_sequence_l1862_186267

theorem exists_increasing_perfect_squares_sequence : 
  ∃ (a : ℕ → ℕ), 
    (∀ k : ℕ, k > 0 → ∃ n : ℕ, a k = n ^ 2) ∧ 
    (∀ k : ℕ, k > 0 → a k < a (k + 1)) ∧
    (∀ k : ℕ, k > 0 → (13 ^ k) ∣ (a k + 1)) := by
  sorry

end exists_increasing_perfect_squares_sequence_l1862_186267


namespace triangle_area_proof_l1862_186203

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  b * Real.cos A = (Real.sqrt 2 * c - a) * Real.cos B →
  B = π / 4 →
  C > π / 2 →
  a = 4 →
  b = 3 →
  (1 / 2) * a * b * Real.sin C = 4 - Real.sqrt 2 :=
by sorry

end triangle_area_proof_l1862_186203


namespace max_length_sum_l1862_186256

def length (k : ℕ) : ℕ := sorry

def has_even_power_prime_factor (n : ℕ) : Prop := sorry

def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem max_length_sum (x y : ℕ) 
  (hx : x > 1) 
  (hy : y > 1) 
  (hsum : x + 3 * y < 1000) 
  (hx_even : has_even_power_prime_factor x) 
  (hy_even : has_even_power_prime_factor y) 
  (hp : smallest_prime_factor x + smallest_prime_factor y ≡ 0 [MOD 3]) :
  ∀ (a b : ℕ), a > 1 → b > 1 → a + 3 * b < 1000 → 
    has_even_power_prime_factor a → has_even_power_prime_factor b → 
    smallest_prime_factor a + smallest_prime_factor b ≡ 0 [MOD 3] →
    length x + length y ≥ length a + length b :=
sorry

end max_length_sum_l1862_186256


namespace cos_3x_minus_pi_3_equals_sin_3x_plus_pi_18_l1862_186214

theorem cos_3x_minus_pi_3_equals_sin_3x_plus_pi_18 (x : ℝ) :
  Real.cos (3 * x - π / 3) = Real.sin (3 * (x + π / 18)) := by
  sorry

end cos_3x_minus_pi_3_equals_sin_3x_plus_pi_18_l1862_186214


namespace problem_statement_l1862_186264

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : 
  (a + b)^2002 + a^2001 = 2 := by
  sorry

end problem_statement_l1862_186264


namespace cut_square_problem_l1862_186287

/-- Given a square with integer side length and four isosceles right triangles
    cut from its corners, if the total area of the cut triangles is 40 square centimeters,
    then the area of the remaining rectangle is 24 square centimeters. -/
theorem cut_square_problem (s a b : ℕ) : 
  s = a + b →  -- The side length of the square is the sum of the leg lengths
  a^2 + b^2 = 40 →  -- The total area of cut triangles is 40
  s^2 - (a^2 + b^2) = 24 :=  -- The area of the remaining rectangle is 24
by sorry

end cut_square_problem_l1862_186287


namespace car_repair_cost_l1862_186272

/-- Calculates the total cost for a car repair given the hourly rate, hours worked per day,
    number of days worked, and cost of parts. -/
def total_cost (hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (parts_cost : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_worked + parts_cost

/-- Proves that the total cost for the car repair is $9220 given the specified conditions. -/
theorem car_repair_cost :
  total_cost 60 8 14 2500 = 9220 := by
  sorry

end car_repair_cost_l1862_186272


namespace bags_difference_l1862_186283

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 8

/-- The number of bags Tiffany found the next day -/
def next_day_bags : ℕ := 7

/-- Theorem stating the difference in bags between Monday and the next day -/
theorem bags_difference : monday_bags - next_day_bags = 1 := by
  sorry

end bags_difference_l1862_186283


namespace tan_theta_values_l1862_186273

theorem tan_theta_values (θ : Real) (h : 2 * Real.sin θ = 1 + Real.cos θ) : 
  Real.tan θ = 4/3 ∨ Real.tan θ = 0 := by
  sorry

end tan_theta_values_l1862_186273


namespace pages_per_booklet_l1862_186257

theorem pages_per_booklet (total_booklets : ℕ) (total_pages : ℕ) 
  (h1 : total_booklets = 49) 
  (h2 : total_pages = 441) : 
  total_pages / total_booklets = 9 := by
  sorry

end pages_per_booklet_l1862_186257
