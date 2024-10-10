import Mathlib

namespace odd_function_property_l61_6144

-- Define an odd function f from ℝ to ℝ
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property f(x+2) = -1/f(x)
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -1 / f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_prop : has_property f) : 
  f 8 = 0 := by
sorry

end odd_function_property_l61_6144


namespace max_m_value_l61_6153

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  (∀ m : ℝ, a + b/2 + Real.sqrt (a^2/2 + 2*b^2) - m*a*b ≥ 0) →
  (∃ m : ℝ, m = 3/2 ∧ a + b/2 + Real.sqrt (a^2/2 + 2*b^2) - m*a*b = 0 ∧
    ∀ m' : ℝ, m' > m → a + b/2 + Real.sqrt (a^2/2 + 2*b^2) - m'*a*b < 0) :=
by sorry

end max_m_value_l61_6153


namespace volleyball_managers_l61_6135

theorem volleyball_managers (num_teams : ℕ) (people_per_team : ℕ) (num_employees : ℕ) : 
  num_teams = 6 → people_per_team = 5 → num_employees = 7 →
  num_teams * people_per_team - num_employees = 23 := by
  sorry

end volleyball_managers_l61_6135


namespace complement_A_intersect_B_l61_6126

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}

-- Define the set B
def B : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end complement_A_intersect_B_l61_6126


namespace divisible_by_225_l61_6169

theorem divisible_by_225 (n : ℕ) : ∃ k : ℤ, 16^n - 15*n - 1 = 225*k := by
  sorry

end divisible_by_225_l61_6169


namespace factorization_problem_1_factorization_problem_2_l61_6181

-- Problem 1
theorem factorization_problem_1 (a : ℝ) : a^3 - 16*a = a*(a+4)*(a-4) := by sorry

-- Problem 2
theorem factorization_problem_2 (x : ℝ) : (x-2)*(x-4)+1 = (x-3)^2 := by sorry

end factorization_problem_1_factorization_problem_2_l61_6181


namespace function_not_monotonic_iff_m_gt_four_l61_6140

/-- A function f(x) = mln(x+1) + x^2 - mx is not monotonic on (1, +∞) iff m > 4 -/
theorem function_not_monotonic_iff_m_gt_four (m : ℝ) :
  (∃ (x y : ℝ), 1 < x ∧ x < y ∧
    (m * Real.log (x + 1) + x^2 - m * x ≤ m * Real.log (y + 1) + y^2 - m * y ∧
     m * Real.log (y + 1) + y^2 - m * y ≤ m * Real.log (x + 1) + x^2 - m * x)) ↔
  m > 4 :=
by sorry


end function_not_monotonic_iff_m_gt_four_l61_6140


namespace fraction_of_sales_for_ingredients_l61_6165

/-- Proves that the fraction of sales used to buy ingredients is 3/5 -/
theorem fraction_of_sales_for_ingredients
  (num_pies : ℕ)
  (price_per_pie : ℚ)
  (amount_remaining : ℚ)
  (h1 : num_pies = 200)
  (h2 : price_per_pie = 20)
  (h3 : amount_remaining = 1600) :
  (num_pies * price_per_pie - amount_remaining) / (num_pies * price_per_pie) = 3 / 5 := by
  sorry

end fraction_of_sales_for_ingredients_l61_6165


namespace remainder_divisibility_l61_6166

theorem remainder_divisibility (N : ℤ) : N % 13 = 5 → N % 39 = 5 := by
  sorry

end remainder_divisibility_l61_6166


namespace rectangle_area_change_l61_6138

/-- Given a rectangle with area 540 square centimeters, if its length is decreased by 15% and
    its width is increased by 20%, the new area will be 550.8 square centimeters. -/
theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 540) : 
  (L * 0.85) * (W * 1.2) = 550.8 := by
  sorry

end rectangle_area_change_l61_6138


namespace quadratic_domain_range_existence_l61_6100

/-- 
Given a quadratic function f(x) = -1/2 * x^2 + x + a, where a is a constant,
there exist real numbers m and n (with m < n) such that the domain of f is [m, n]
and the range is [3m, 3n] if and only if -2 < a ≤ 5/2.
-/
theorem quadratic_domain_range_existence (a : ℝ) :
  (∃ (m n : ℝ), m < n ∧
    (∀ x, x ∈ Set.Icc m n ↔ -1/2 * x^2 + x + a ∈ Set.Icc (3*m) (3*n)) ∧
    (∀ y, y ∈ Set.Icc (3*m) (3*n) → ∃ x ∈ Set.Icc m n, y = -1/2 * x^2 + x + a)) ↔
  -2 < a ∧ a ≤ 5/2 :=
by sorry

end quadratic_domain_range_existence_l61_6100


namespace cyclist_speed_is_25_l61_6172

/-- The speed of the motorcyclist in km/h -/
def V_M : ℝ := 50

/-- The system of equations for the cyclist and motorcyclist problem -/
def equations (x y : ℝ) : Prop :=
  (20 / x - 20 / V_M = y) ∧ (70 - 8 / 3 * x = V_M * (7 / 15 - y))

/-- Theorem stating that x = 25 km/h satisfies the system of equations for some y -/
theorem cyclist_speed_is_25 : ∃ y : ℝ, equations 25 y := by
  sorry

end cyclist_speed_is_25_l61_6172


namespace valid_pairs_count_l61_6174

def count_valid_pairs : ℕ := by sorry

theorem valid_pairs_count :
  count_valid_pairs = 8 :=
by
  have h1 : ∀ a b : ℕ+, (a : ℝ) + 2 / (b : ℝ) = 17 * ((1 : ℝ) / a + 2 * b) →
            (a : ℕ) + b ≤ 150 →
            (a : ℕ) = 17 * b := by sorry
  
  have h2 : ∀ b : ℕ+, b ≤ 8 → (17 * b : ℕ) + b ≤ 150 := by sorry
  
  have h3 : ∀ b : ℕ+, b > 8 → (17 * b : ℕ) + b > 150 := by sorry
  
  sorry

end valid_pairs_count_l61_6174


namespace binomial_20_18_l61_6127

theorem binomial_20_18 : Nat.choose 20 18 = 190 := by
  sorry

end binomial_20_18_l61_6127


namespace animus_tower_spiders_l61_6134

/-- The number of spiders hired for the Animus Tower project -/
def spiders_hired (total_workers beavers_hired : ℕ) : ℕ :=
  total_workers - beavers_hired

/-- Theorem stating the number of spiders hired for the Animus Tower project -/
theorem animus_tower_spiders :
  spiders_hired 862 318 = 544 := by
  sorry

end animus_tower_spiders_l61_6134


namespace largest_band_size_l61_6118

/-- Represents a band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid --/
def isValidFormation (f : BandFormation) (totalMembers : ℕ) : Prop :=
  f.rows * f.membersPerRow + 3 = totalMembers

/-- Checks if the new formation after rearrangement is valid --/
def isValidNewFormation (f : BandFormation) (totalMembers : ℕ) : Prop :=
  (f.rows - 3) * (f.membersPerRow + 2) = totalMembers

/-- Main theorem: The largest possible number of band members is 147 --/
theorem largest_band_size :
  ∃ (f : BandFormation) (m : ℕ),
    m < 150 ∧
    isValidFormation f m ∧
    isValidNewFormation f m ∧
    ∀ (f' : BandFormation) (m' : ℕ),
      m' < 150 →
      isValidFormation f' m' →
      isValidNewFormation f' m' →
      m' ≤ m ∧
    m = 147 := by
  sorry

end largest_band_size_l61_6118


namespace coordinates_sum_of_X_l61_6105

-- Define the points as pairs of real numbers
def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := (3, 5)
def Z : ℝ × ℝ := (1, -3)

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem coordinates_sum_of_X :
  (distance X Z) / (distance X Y) = 1/2 ∧
  (distance Z Y) / (distance X Y) = 1/2 →
  X.1 + X.2 = -12 := by
  sorry

end coordinates_sum_of_X_l61_6105


namespace no_valid_n_l61_6162

theorem no_valid_n : ¬ ∃ (n : ℕ), 
  n > 0 ∧ 
  (1000 ≤ n / 5) ∧ (n / 5 ≤ 9999) ∧ 
  (1000 ≤ 5 * n) ∧ (5 * n ≤ 9999) := by
  sorry

end no_valid_n_l61_6162


namespace call_center_fraction_l61_6129

/-- Represents the fraction of calls processed by team B given the conditions of the problem -/
theorem call_center_fraction (team_a team_b : ℕ) (calls_a calls_b : ℝ) : 
  team_a = (5 : ℝ) / 8 * team_b →
  calls_a = (1 : ℝ) / 5 * calls_b →
  (team_b * calls_b) / (team_a * calls_a + team_b * calls_b) = 8 / 9 := by
  sorry

end call_center_fraction_l61_6129


namespace simplify_expression_l61_6148

theorem simplify_expression (a : ℝ) (h1 : a^2 - 1 ≠ 0) (h2 : a ≠ 0) :
  (1 / (a + 1) + 1 / (a^2 - 1)) / (a / (a^2 - 2*a + 1)) = (a - 1) / (a + 1) := by
  sorry

end simplify_expression_l61_6148


namespace smaller_number_proof_l61_6164

theorem smaller_number_proof (S L : ℕ) 
  (h1 : L - S = 2468) 
  (h2 : L = 8 * S + 27) : 
  S = 349 := by
  sorry

end smaller_number_proof_l61_6164


namespace triangle_angle_c_l61_6187

theorem triangle_angle_c (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = Real.pi) (h5 : B = Real.pi / 4)
  (h6 : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a = c * Real.sqrt 2 ∧
    a / (Real.sin C) = b / (Real.sin A) ∧ b / (Real.sin B) = c / (Real.sin A)) :
  C = 7 * Real.pi / 12 := by
sorry

end triangle_angle_c_l61_6187


namespace sports_club_purchase_l61_6109

/-- The price difference between a basketball and a soccer ball -/
def price_difference : ℕ := 30

/-- The budget for soccer balls -/
def soccer_budget : ℕ := 1500

/-- The budget for basketballs -/
def basketball_budget : ℕ := 2400

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 100

/-- The minimum discount on basketballs -/
def min_discount : ℕ := 25

/-- The maximum discount on basketballs -/
def max_discount : ℕ := 35

/-- The price of a soccer ball -/
def soccer_price : ℕ := 50

/-- The price of a basketball -/
def basketball_price : ℕ := 80

theorem sports_club_purchase :
  ∀ (m : ℕ), min_discount ≤ m → m ≤ max_discount →
  (∃ (y : ℕ), y ≤ total_balls ∧ 3 * (total_balls - y) ≤ y ∧
    (∀ (z : ℕ), z ≤ total_balls → 3 * (total_balls - z) ≤ z →
      (if 30 < m then
        (basketball_price - m) * y + soccer_price * (total_balls - y) ≤ (basketball_price - m) * z + soccer_price * (total_balls - z)
      else if m < 30 then
        (basketball_price - m) * y + soccer_price * (total_balls - y) ≤ (basketball_price - m) * z + soccer_price * (total_balls - z)
      else
        (basketball_price - m) * y + soccer_price * (total_balls - y) = (basketball_price - m) * z + soccer_price * (total_balls - z)))) ∧
  basketball_price = soccer_price + price_difference ∧
  soccer_budget / soccer_price = basketball_budget / basketball_price := by
  sorry

end sports_club_purchase_l61_6109


namespace right_rectangular_prism_volume_l61_6196

theorem right_rectangular_prism_volume
  (face_area1 face_area2 face_area3 : ℝ)
  (h1 : face_area1 = 6.5)
  (h2 : face_area2 = 8)
  (h3 : face_area3 = 13)
  : ∃ (l w h : ℝ),
    l * w = face_area1 ∧
    w * h = face_area2 ∧
    l * h = face_area3 ∧
    l * w * h = 26 :=
by sorry

end right_rectangular_prism_volume_l61_6196


namespace original_true_implies_contrapositive_true_l61_6113

-- Define a proposition type
variable (P Q : Prop)

-- Define the contrapositive of an implication
def contrapositive (P Q : Prop) : Prop := ¬Q → ¬P

-- Theorem: If the original proposition is true, then its contrapositive is also true
theorem original_true_implies_contrapositive_true (h : P → Q) : contrapositive P Q :=
  sorry

end original_true_implies_contrapositive_true_l61_6113


namespace alternating_ones_zeros_composite_l61_6114

/-- The number formed by k+1 ones with k zeros interspersed between them -/
def alternating_ones_zeros (k : ℕ) : ℕ :=
  (10^(k+1) - 1) / 9

/-- Theorem stating that the alternating_ones_zeros number is composite for k ≥ 2 -/
theorem alternating_ones_zeros_composite (k : ℕ) (h : k ≥ 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ alternating_ones_zeros k = a * b :=
sorry


end alternating_ones_zeros_composite_l61_6114


namespace g_of_one_eq_neg_two_l61_6115

theorem g_of_one_eq_neg_two :
  let g : ℝ → ℝ := fun x ↦ x^3 - x^2 - 2*x
  g 1 = -2 := by sorry

end g_of_one_eq_neg_two_l61_6115


namespace sphere_surface_area_containing_cuboid_l61_6107

theorem sphere_surface_area_containing_cuboid (a b c : ℝ) (S : ℝ) :
  a = 3 → b = 4 → c = 5 →
  S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
  S = 50 * Real.pi :=
by sorry

end sphere_surface_area_containing_cuboid_l61_6107


namespace geometric_sequence_sum_l61_6147

theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                            -- a_1 = 1
  4 * a 2 - 2 * a 3 = 2 * a 3 - a 4 →  -- arithmetic sequence condition
  a 2 + a 3 + a 4 = 14 := by            
sorry

end geometric_sequence_sum_l61_6147


namespace max_value_of_product_l61_6170

theorem max_value_of_product (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2*x + 5*y < 105) :
  ∃ (max : ℝ), max = 4287.5 ∧ xy*(105 - 2*x - 5*y) ≤ max ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 5*y₀ < 105 ∧ x₀*y₀*(105 - 2*x₀ - 5*y₀) = max :=
by
  sorry

end max_value_of_product_l61_6170


namespace abc_is_right_triangle_l61_6141

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Parabola y^2 = 4x -/
def on_parabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- Line passes through a point -/
def line_passes_through (l : Line) (p : Point) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

/-- Triangle formed by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  let ab_slope := (t.b.y - t.a.y) / (t.b.x - t.a.x)
  let ac_slope := (t.c.y - t.a.y) / (t.c.x - t.a.x)
  ab_slope * ac_slope = -1

theorem abc_is_right_triangle (a b c : Point) (h1 : a.x = 1 ∧ a.y = 2)
    (h2 : on_parabola b) (h3 : on_parabola c)
    (h4 : line_passes_through (Line.mk b c) (Point.mk 5 (-2))) :
    is_right_triangle (Triangle.mk a b c) := by
  sorry

end abc_is_right_triangle_l61_6141


namespace unique_solution_condition_l61_6116

theorem unique_solution_condition (a : ℝ) : 
  (∃! x, 0 ≤ x^2 + a*x + 6 ∧ x^2 + a*x + 6 ≤ 4) ↔ (a = 2*Real.sqrt 2 ∨ a = -2*Real.sqrt 2) :=
sorry

end unique_solution_condition_l61_6116


namespace baseball_card_value_decrease_l61_6122

theorem baseball_card_value_decrease : ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - 0.6)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := (initial_value - value_after_second_year) / initial_value
  total_decrease = 0.64 := by
  sorry

end baseball_card_value_decrease_l61_6122


namespace least_power_congruence_l61_6101

theorem least_power_congruence (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 195 → 3^m % 143^2 ≠ 1) ∧ 
  3^195 % 143^2 = 1 := by
  sorry

end least_power_congruence_l61_6101


namespace quadratic_roots_l61_6139

theorem quadratic_roots : ∃ x₁ x₂ : ℝ, 
  (x₁ = 2 ∧ x₂ = -1) ∧ 
  (∀ x : ℝ, x * (x - 2) = 2 - x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_roots_l61_6139


namespace calcium_chloride_formation_l61_6130

/-- Represents a chemical reaction --/
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String
  ratio1 : ℚ
  ratio2 : ℚ

/-- Calculates the moles of product formed in a chemical reaction --/
def calculate_product_moles (r : Reaction) (moles_reactant1 : ℚ) (moles_reactant2 : ℚ) : ℚ :=
  min (moles_reactant1 / r.ratio1) (moles_reactant2 / r.ratio2)

/-- Theorem: Given 4 moles of HCl and 2 moles of CaCO3, 2 moles of CaCl2 are formed --/
theorem calcium_chloride_formation :
  let r : Reaction := {
    reactant1 := "CaCO3",
    reactant2 := "HCl",
    product1 := "CaCl2",
    product2 := "CO2",
    product3 := "H2O",
    ratio1 := 1,
    ratio2 := 2
  }
  calculate_product_moles r 2 4 = 2 := by
  sorry

end calcium_chloride_formation_l61_6130


namespace solve_class_problem_l61_6194

def class_problem (N : ℕ) : Prop :=
  ∃ (taqeesha_score : ℕ),
    N > 1 ∧
    (77 * (N - 1) + taqeesha_score) / N = 78 ∧
    N - 1 = 16

theorem solve_class_problem :
  ∃ (N : ℕ), class_problem N ∧ N = 17 := by
  sorry

end solve_class_problem_l61_6194


namespace polynomial_expansion_l61_6173

theorem polynomial_expansion (x : ℝ) : 
  (5*x^2 + 3*x - 4) * (2*x^3 + x^2 - x + 1) = 
  10*x^5 + 11*x^4 - 10*x^3 - 2*x^2 + 7*x - 4 := by
  sorry

end polynomial_expansion_l61_6173


namespace problem_statement_l61_6190

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 2) :
  (∀ x : ℝ, (1/a^2 + 4/b^2 ≥ |2*x - 1| - |x - 1|) → (-9/2 ≤ x ∧ x ≤ 9/2)) ∧
  ((1/a + 1/b) * (a^5 + b^5) ≥ 4) :=
by sorry

end problem_statement_l61_6190


namespace cube_sum_and_reciprocal_l61_6185

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^3 + 1/x^3 = 18 := by
  sorry

end cube_sum_and_reciprocal_l61_6185


namespace vectors_are_coplanar_l61_6121

/-- Prove that vectors a, b, and c are coplanar -/
theorem vectors_are_coplanar :
  let a : ℝ × ℝ × ℝ := (1, 2, -3)
  let b : ℝ × ℝ × ℝ := (-2, -4, 6)
  let c : ℝ × ℝ × ℝ := (1, 0, 5)
  ∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :=
by sorry


end vectors_are_coplanar_l61_6121


namespace a_greater_equal_four_l61_6152

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x - a < 0}

-- State the theorem
theorem a_greater_equal_four (a : ℝ) : A ⊆ B a → a ≥ 4 := by
  sorry

end a_greater_equal_four_l61_6152


namespace solution_set_implies_a_value_l61_6131

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 1 < x ∧ x < 3) →
  a = 2 := by
sorry

end solution_set_implies_a_value_l61_6131


namespace negation_of_existence_is_universal_negation_l61_6151

theorem negation_of_existence_is_universal_negation :
  (¬ ∃ (x : ℝ), x^2 = 1) ↔ (∀ (x : ℝ), x^2 ≠ 1) := by sorry

end negation_of_existence_is_universal_negation_l61_6151


namespace paint_usage_for_large_canvas_l61_6128

/-- Given an artist who uses L ounces of paint for every large canvas and 2 ounces for every small canvas,
    prove that L = 3 when the artist has completed 3 large paintings and 4 small paintings,
    using a total of 17 ounces of paint. -/
theorem paint_usage_for_large_canvas (L : ℝ) : 
  (3 * L + 4 * 2 = 17) → L = 3 := by
  sorry

end paint_usage_for_large_canvas_l61_6128


namespace max_cars_ac_no_stripes_l61_6179

theorem max_cars_ac_no_stripes (total_cars : Nat) (cars_no_ac : Nat) (cars_with_stripes : Nat)
  (red_cars : Nat) (red_cars_ac_stripes : Nat) (cars_2000s : Nat) (cars_2010s : Nat)
  (min_new_cars_stripes : Nat) (h1 : total_cars = 150) (h2 : cars_no_ac = 47)
  (h3 : cars_with_stripes = 65) (h4 : red_cars = 25) (h5 : red_cars_ac_stripes = 10)
  (h6 : cars_2000s = 30) (h7 : cars_2010s = 43) (h8 : min_new_cars_stripes = 39)
  (h9 : min_new_cars_stripes ≤ cars_2000s + cars_2010s) :
  (cars_2000s + cars_2010s) - min_new_cars_stripes - red_cars_ac_stripes = 24 :=
by sorry

end max_cars_ac_no_stripes_l61_6179


namespace driving_time_calculation_l61_6157

/-- 
Given a trip with the following conditions:
1. The total trip duration is 15 hours
2. The time stuck in traffic is twice the driving time
This theorem proves that the driving time is 5 hours
-/
theorem driving_time_calculation (total_time : ℝ) (driving_time : ℝ) (traffic_time : ℝ) 
  (h1 : total_time = 15)
  (h2 : traffic_time = 2 * driving_time)
  (h3 : total_time = driving_time + traffic_time) :
  driving_time = 5 := by
sorry

end driving_time_calculation_l61_6157


namespace product_not_always_greater_l61_6182

theorem product_not_always_greater : ∃ (a b : ℝ), a * b ≤ a ∨ a * b ≤ b := by
  sorry

end product_not_always_greater_l61_6182


namespace exponent_multiplication_l61_6154

theorem exponent_multiplication (x : ℝ) (a b : ℕ) :
  x^a * x^b = x^(a + b) := by sorry

end exponent_multiplication_l61_6154


namespace m_is_fengli_fengli_condition_l61_6150

/-- Definition of a Fengli number -/
def is_fengli (n : ℤ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

/-- M is a Fengli number -/
theorem m_is_fengli (x y : ℕ) :
  is_fengli (x^2 + 2*x*y + 2*y^2) :=
sorry

/-- Theorem about the value of m for p to be a Fengli number -/
theorem fengli_condition (x y : ℕ) (m : ℤ) (h : x > y) (h' : y > 0) :
  is_fengli (4*x^2 + m*x*y + 2*y^2 - 10*y + 25) ↔ m = 4 ∨ m = -4 :=
sorry

end m_is_fengli_fengli_condition_l61_6150


namespace third_shift_members_l61_6156

theorem third_shift_members (first_shift : ℕ) (second_shift : ℕ) (first_participation : ℚ)
  (second_participation : ℚ) (third_participation : ℚ) (total_participation : ℚ)
  (h1 : first_shift = 60)
  (h2 : second_shift = 50)
  (h3 : first_participation = 20 / 100)
  (h4 : second_participation = 40 / 100)
  (h5 : third_participation = 10 / 100)
  (h6 : total_participation = 24 / 100) :
  ∃ (third_shift : ℕ),
    (first_shift * first_participation + second_shift * second_participation + third_shift * third_participation) /
    (first_shift + second_shift + third_shift) = total_participation ∧
    third_shift = 40 := by
  sorry

end third_shift_members_l61_6156


namespace inequality_properties_l61_6120

theorem inequality_properties (a b : ℝ) (h : a > b ∧ b > 0) :
  (∀ c, a + c > b + c) ∧
  (a^2 > b^2) ∧
  (Real.sqrt a > Real.sqrt b) ∧
  (∃ c, a * c ≤ b * c) := by
sorry

end inequality_properties_l61_6120


namespace angle_with_touching_circles_theorem_l61_6104

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an angle in 2D space --/
structure Angle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop

/-- Predicate to check if a circle touches a line internally --/
def touches_internally (c : Circle) (l : ℝ × ℝ → Prop) : Prop := sorry

/-- Predicate to check if two circles are non-intersecting --/
def non_intersecting (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a point is on an angle --/
def on_angle (p : ℝ × ℝ) (a : Angle) : Prop := sorry

/-- Predicate to check if a point describes the arc of a circle --/
def describes_circle_arc (p : ℝ × ℝ) : Prop := sorry

theorem angle_with_touching_circles_theorem (a : Angle) (c1 c2 : Circle) 
  (h1 : touches_internally c1 a.side1)
  (h2 : touches_internally c2 a.side2)
  (h3 : non_intersecting c1 c2) :
  ∃ p : ℝ × ℝ, on_angle p a ∧ describes_circle_arc p := by
  sorry

end angle_with_touching_circles_theorem_l61_6104


namespace hiring_probability_l61_6133

theorem hiring_probability (n m k : ℕ) (hn : n = 5) (hm : m = 3) (hk : k = 2) :
  let total_combinations := Nat.choose n m
  let favorable_combinations := total_combinations - Nat.choose (n - k) m
  (favorable_combinations : ℚ) / total_combinations = 9 / 10 :=
by sorry

end hiring_probability_l61_6133


namespace first_term_value_l61_6161

/-- A geometric sequence with five terms -/
def GeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ 36 = b * r ∧ c = 36 * r ∧ 144 = c * r

/-- The first term of the geometric sequence is 9/4 -/
theorem first_term_value (a b c : ℝ) (h : GeometricSequence a b c) : a = 9/4 := by
  sorry

end first_term_value_l61_6161


namespace halfway_between_one_fourth_and_one_seventh_l61_6195

/-- The fraction halfway between two fractions is their average -/
def halfway (a b : ℚ) : ℚ := (a + b) / 2

/-- The fraction halfway between 1/4 and 1/7 is 11/56 -/
theorem halfway_between_one_fourth_and_one_seventh :
  halfway (1/4) (1/7) = 11/56 := by
  sorry

end halfway_between_one_fourth_and_one_seventh_l61_6195


namespace prism_21_edges_9_faces_l61_6125

/-- A prism is a polyhedron with two congruent parallel faces (bases) and lateral faces that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism. -/
def num_faces (p : Prism) : ℕ :=
  2 + (p.edges / 3)

/-- Theorem: A prism with 21 edges has 9 faces. -/
theorem prism_21_edges_9_faces :
  ∀ p : Prism, p.edges = 21 → num_faces p = 9 := by
  sorry


end prism_21_edges_9_faces_l61_6125


namespace cynthia_gallons_proof_l61_6159

def pool_capacity : ℕ := 105
def num_trips : ℕ := 7
def caleb_gallons : ℕ := 7

theorem cynthia_gallons_proof :
  ∃ (cynthia_gallons : ℕ),
    cynthia_gallons * num_trips + caleb_gallons * num_trips = pool_capacity ∧
    cynthia_gallons = 8 := by
  sorry

end cynthia_gallons_proof_l61_6159


namespace perpendicular_vectors_x_value_l61_6192

/-- Given two vectors a and b in ℝ², prove that if a = (1, -2) and b = (3, x) are perpendicular, then x = 3/2. -/
theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (1, -2) →
  b = (3, x) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = 3/2 := by
  sorry

end perpendicular_vectors_x_value_l61_6192


namespace ceiling_floor_sum_l61_6199

theorem ceiling_floor_sum : ⌈(5/4 : ℝ)⌉ + ⌊-(5/4 : ℝ)⌋ = 0 :=
by sorry

end ceiling_floor_sum_l61_6199


namespace f_properties_l61_6180

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^3 - 6*x^2 - 9*x + 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-4) 2

-- Theorem statement
theorem f_properties :
  -- 1. f is strictly decreasing on (-∞, -3) and (-1, +∞)
  (∀ x y, x < y → x < -3 → f y < f x) ∧
  (∀ x y, x < y → -1 < x → f y < f x) ∧
  -- 2. The minimum value of f on [-4, 2] is -47
  (∀ x ∈ interval, f x ≥ -47) ∧
  (∃ x ∈ interval, f x = -47) ∧
  -- 3. The maximum value of f on [-4, 2] is 7
  (∀ x ∈ interval, f x ≤ 7) ∧
  (∃ x ∈ interval, f x = 7) :=
sorry

end f_properties_l61_6180


namespace probability_at_least_six_heads_in_eight_flips_probability_at_least_six_heads_in_eight_flips_proof_l61_6171

/-- The probability of getting at least 6 heads in 8 flips of a fair coin -/
theorem probability_at_least_six_heads_in_eight_flips : ℚ :=
  37 / 256

/-- Proof that the probability of getting at least 6 heads in 8 flips of a fair coin is 37/256 -/
theorem probability_at_least_six_heads_in_eight_flips_proof :
  probability_at_least_six_heads_in_eight_flips = 37 / 256 := by
  sorry

end probability_at_least_six_heads_in_eight_flips_probability_at_least_six_heads_in_eight_flips_proof_l61_6171


namespace basketball_win_rate_l61_6158

theorem basketball_win_rate (total_games : ℕ) (first_segment : ℕ) (won_first : ℕ) (target_rate : ℚ) : 
  total_games = 130 →
  first_segment = 70 →
  won_first = 60 →
  target_rate = 3/4 →
  ∃ (x : ℕ), x = 38 ∧ 
    (won_first + x : ℚ) / total_games = target_rate ∧
    x ≤ total_games - first_segment :=
by sorry

end basketball_win_rate_l61_6158


namespace count_valid_pairs_l61_6137

/-- The number of ordered pairs (m,n) of positive integers satisfying the given conditions -/
def valid_pairs : ℕ := 4

/-- Definition of a valid pair (m,n) -/
def is_valid_pair (m n : ℕ) : Prop :=
  m ≥ n ∧ n % 2 = 1 ∧ m^2 - n^2 = 120

/-- Theorem stating that there are exactly 4 valid pairs -/
theorem count_valid_pairs :
  (∃! (s : Finset (ℕ × ℕ)), s.card = valid_pairs ∧
    ∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_pair p.1 p.2) :=
sorry

end count_valid_pairs_l61_6137


namespace fraction_problem_l61_6132

theorem fraction_problem (f : ℚ) : f * 20 + 7 = 17 → f = 1/2 := by
  sorry

end fraction_problem_l61_6132


namespace intersection_point_l61_6110

-- Define the two linear functions
def f (x : ℝ) (m : ℝ) : ℝ := x + m
def g (x : ℝ) : ℝ := 2 * x - 2

-- Theorem statement
theorem intersection_point (m : ℝ) : 
  (∃ y : ℝ, f 0 m = y ∧ g 0 = y) → m = -2 := by
  sorry

end intersection_point_l61_6110


namespace camera_imaging_formula_l61_6111

/-- Given the camera imaging formula, prove the relationship between focal length,
    object distance, and image distance. -/
theorem camera_imaging_formula (f u v : ℝ) (hf : f ≠ 0) (hu : u ≠ 0) (hv : v ≠ 0) (hv_neq_f : v ≠ f) :
  1 / f = 1 / u + 1 / v → v = f * u / (u - f) := by
  sorry

end camera_imaging_formula_l61_6111


namespace hyperbola_vertex_distance_l61_6143

/-- The distance between the vertices of the hyperbola x²/64 - y²/81 = 1 is 16 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/64 - y^2/81 = 1
  ∃ x₁ x₂ : ℝ, h x₁ 0 ∧ h x₂ 0 ∧ |x₁ - x₂| = 16 :=
by sorry

end hyperbola_vertex_distance_l61_6143


namespace positive_integer_solutions_of_inequality_l61_6183

theorem positive_integer_solutions_of_inequality :
  ∀ x : ℕ+, 9 - 3 * (x : ℝ) > 0 ↔ x = 1 ∨ x = 2 := by
sorry

end positive_integer_solutions_of_inequality_l61_6183


namespace exist_sequence_l61_6193

/-- Sum of digits of a positive integer -/
def S (m : ℕ+) : ℕ := sorry

/-- Product of digits of a positive integer -/
def P (m : ℕ+) : ℕ := sorry

/-- For any positive integer n, there exist positive integers a₁, a₂, ..., aₙ
    satisfying the required conditions -/
theorem exist_sequence (n : ℕ+) : 
  ∃ (a : Fin n → ℕ+), 
    (∀ i j : Fin n, i < j → S (a i) < S (a j)) ∧ 
    (∀ i : Fin n, S (a i) = P (a ((i + 1) % n))) := by
  sorry

end exist_sequence_l61_6193


namespace triangle_midpoint_sum_l61_6188

theorem triangle_midpoint_sum (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) + (a + c) + (b + c) = 30 := by
  sorry

end triangle_midpoint_sum_l61_6188


namespace delta_value_l61_6142

theorem delta_value : ∃ Δ : ℂ, (4 * (-3) = Δ^2 + 3) ∧ (Δ = Complex.I * Real.sqrt 15 ∨ Δ = -Complex.I * Real.sqrt 15) := by
  sorry

end delta_value_l61_6142


namespace probability_of_winning_reward_l61_6198

/-- The number of different types of blessing cards -/
def num_card_types : ℕ := 3

/-- The number of red envelopes Xiao Ming has -/
def num_envelopes : ℕ := 4

/-- The probability of winning the reward -/
def win_probability : ℚ := 4/9

/-- Theorem stating the probability of winning the reward -/
theorem probability_of_winning_reward :
  (num_card_types = 3) →
  (num_envelopes = 4) →
  (win_probability = 4/9) := by
  sorry

end probability_of_winning_reward_l61_6198


namespace at_least_one_hit_probability_l61_6168

theorem at_least_one_hit_probability (p1 p2 : ℝ) (h1 : p1 = 0.7) (h2 : p2 = 0.8) :
  p1 + p2 - p1 * p2 = 0.94 := by
  sorry

end at_least_one_hit_probability_l61_6168


namespace inequality_preservation_l61_6175

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 1 > b - 1 := by
  sorry

end inequality_preservation_l61_6175


namespace parallelogram_other_vertices_y_sum_l61_6146

/-- A parallelogram with two opposite vertices at (2,15) and (8,-2) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (2, 15)
  v2 : ℝ × ℝ := (8, -2)
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  is_parallelogram : True  -- We assume this is a valid parallelogram

/-- The sum of y-coordinates of the other two vertices is 13 -/
theorem parallelogram_other_vertices_y_sum (p : Parallelogram) : 
  (p.v3).2 + (p.v4).2 = 13 := by
  sorry


end parallelogram_other_vertices_y_sum_l61_6146


namespace rectangular_plot_area_l61_6167

theorem rectangular_plot_area (perimeter : ℝ) (length width : ℝ) : 
  perimeter = 24 →
  length = 2 * width →
  2 * (length + width) = perimeter →
  length * width = 32 := by
sorry

end rectangular_plot_area_l61_6167


namespace focus_to_asymptote_distance_l61_6191

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define one asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Statement: The distance from the focus to the asymptote is √3
theorem focus_to_asymptote_distance :
  ∃ (x y : ℝ), parabola x y ∧ hyperbola x y ∧ asymptote x y ∧
  (Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2) = Real.sqrt 3) :=
sorry

end focus_to_asymptote_distance_l61_6191


namespace solution_set_inequality_l61_6117

/-- Given that the solution set of ax^2 + bx + 4 > 0 is (-1, 2),
    prove that the solution set of ax + b + 4 > 0 is (-∞, 3) -/
theorem solution_set_inequality (a b : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + 4 > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x : ℝ, a*x + b + 4 > 0 ↔ x < 3) :=
by sorry

end solution_set_inequality_l61_6117


namespace greatest_common_divisor_of_three_l61_6124

theorem greatest_common_divisor_of_three (n : ℕ) : 
  (∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
    {d : ℕ | d ∣ 180 ∧ d ∣ n} = {d1, d2, d3}) →
  (Nat.gcd 180 n = 9) :=
sorry

end greatest_common_divisor_of_three_l61_6124


namespace integer_cube_between_zero_and_nine_l61_6102

theorem integer_cube_between_zero_and_nine (a : ℤ) : 0 < a^3 ∧ a^3 < 9 → a = 1 ∨ a = 2 := by
  sorry

end integer_cube_between_zero_and_nine_l61_6102


namespace tangent_segment_length_is_3cm_l61_6136

/-- An isosceles triangle with a base of 12 cm and a height of 8 cm -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  isIsosceles : base = 12 ∧ height = 8

/-- A circle inscribed in the isosceles triangle -/
structure InscribedCircle (t : IsoscelesTriangle) where
  center : ℝ × ℝ
  radius : ℝ
  isInscribed : True  -- This is a placeholder for the inscribed circle condition

/-- A tangent line parallel to the base of the triangle -/
structure ParallelTangent (t : IsoscelesTriangle) (c : InscribedCircle t) where
  point : ℝ × ℝ
  isParallel : True  -- This is a placeholder for the parallel condition
  isTangent : True   -- This is a placeholder for the tangent condition

/-- The length of the segment of the tangent line between the sides of the triangle -/
def tangentSegmentLength (t : IsoscelesTriangle) (c : InscribedCircle t) (l : ParallelTangent t c) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The main theorem stating that the length of the tangent segment is 3 cm -/
theorem tangent_segment_length_is_3cm (t : IsoscelesTriangle) (c : InscribedCircle t) (l : ParallelTangent t c) :
  tangentSegmentLength t c l = 3 := by
  sorry

end tangent_segment_length_is_3cm_l61_6136


namespace min_a_is_neg_two_l61_6112

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

/-- The theorem stating that -2 is the minimum value of a for which f(x) is always non-negative -/
theorem min_a_is_neg_two :
  (∀ a : ℝ, (∀ x : ℝ, f a x ≥ 0) → a ≥ -2) ∧
  (∀ x : ℝ, f (-2) x ≥ 0) :=
sorry

end min_a_is_neg_two_l61_6112


namespace leading_coefficient_of_g_l61_6177

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := ((x/2)^α) / (x-1)

noncomputable def g (α : ℝ) : ℝ := (deriv^[4] (f α)) 2

theorem leading_coefficient_of_g (α : ℝ) : 
  ∃ (p : Polynomial ℝ), (∀ x, g x = p.eval x) ∧ p.leadingCoeff = 1/16 :=
sorry

end leading_coefficient_of_g_l61_6177


namespace pure_imaginary_fraction_l61_6108

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end pure_imaginary_fraction_l61_6108


namespace circles_intersect_l61_6123

theorem circles_intersect (r₁ r₂ d : ℝ) 
  (h₁ : r₁ = 5) 
  (h₂ : r₂ = 3) 
  (h₃ : d = 7) : 
  (r₁ - r₂ < d) ∧ (d < r₁ + r₂) := by
  sorry

#check circles_intersect

end circles_intersect_l61_6123


namespace cube_equality_solution_l61_6184

theorem cube_equality_solution : ∃! (N : ℕ), N > 0 ∧ 12^3 * 30^3 = 20^3 * N^3 :=
by
  use 18
  sorry

end cube_equality_solution_l61_6184


namespace exam_students_count_l61_6149

/-- Proves that given the conditions of the exam results, the total number of students is 14 -/
theorem exam_students_count (total_average : ℝ) (excluded_count : ℕ) (excluded_average : ℝ) (remaining_average : ℝ)
  (h1 : total_average = 65)
  (h2 : excluded_count = 5)
  (h3 : excluded_average = 20)
  (h4 : remaining_average = 90) :
  ∃ (n : ℕ), n = 14 ∧ 
    (n : ℝ) * total_average = 
      ((n - excluded_count) : ℝ) * remaining_average + (excluded_count : ℝ) * excluded_average :=
by sorry

end exam_students_count_l61_6149


namespace first_number_in_ratio_l61_6163

theorem first_number_in_ratio (A B : ℕ) (h1 : A > 0) (h2 : B > 0) : 
  A * 4 = B * 5 → lcm A B = 80 → A = 10 := by
  sorry

end first_number_in_ratio_l61_6163


namespace highlighter_spent_theorem_l61_6103

def total_money : ℝ := 150
def sharpener_price : ℝ := 3
def notebook_price : ℝ := 7
def eraser_price : ℝ := 2
def sharpener_count : ℕ := 5
def notebook_count : ℕ := 6
def eraser_count : ℕ := 15

def heaven_spent : ℝ := sharpener_price * sharpener_count + notebook_price * notebook_count

def money_left_after_heaven : ℝ := total_money - heaven_spent

def brother_eraser_spent : ℝ := eraser_price * eraser_count

theorem highlighter_spent_theorem :
  money_left_after_heaven - brother_eraser_spent = 63 := by sorry

end highlighter_spent_theorem_l61_6103


namespace mode_of_interest_groups_l61_6119

def interest_groups : List Nat := [4, 7, 5, 4, 6, 4, 5]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_interest_groups :
  mode interest_groups = 4 := by
  sorry

end mode_of_interest_groups_l61_6119


namespace tigers_wins_l61_6189

theorem tigers_wins (total_games : ℕ) (games_lost_more : ℕ) 
  (h1 : total_games = 120)
  (h2 : games_lost_more = 38) :
  let games_won := (total_games - games_lost_more) / 2
  games_won = 41 := by
sorry

end tigers_wins_l61_6189


namespace red_joker_probability_l61_6178

/-- A modified deck of cards -/
structure ModifiedDeck :=
  (total_cards : ℕ)
  (standard_cards : ℕ)
  (red_jokers : ℕ)
  (black_jokers : ℕ)

/-- Definition of our specific modified deck -/
def our_deck : ModifiedDeck :=
  { total_cards := 54,
    standard_cards := 52,
    red_jokers := 1,
    black_jokers := 1 }

/-- The probability of drawing a specific card from a deck -/
def probability_of_draw (deck : ModifiedDeck) (specific_cards : ℕ) : ℚ :=
  specific_cards / deck.total_cards

theorem red_joker_probability :
  probability_of_draw our_deck our_deck.red_jokers = 1 / 54 := by
  sorry


end red_joker_probability_l61_6178


namespace min_detectors_for_cross_l61_6197

/-- The size of the board --/
def boardSize : Nat := 5

/-- The number of cells in the cross pattern --/
def crossSize : Nat := 5

/-- The number of possible positions for the cross on the board --/
def possiblePositions : Nat := 3 * 3

/-- Function to calculate the number of possible detector states --/
def detectorStates (n : Nat) : Nat := 2^n

/-- Theorem stating the minimum number of detectors needed --/
theorem min_detectors_for_cross :
  ∃ (n : Nat), (n = 4) ∧ 
  (∀ (k : Nat), detectorStates k ≥ possiblePositions → k ≥ n) ∧
  (detectorStates n ≥ possiblePositions) := by
  sorry

end min_detectors_for_cross_l61_6197


namespace fib_150_mod_5_l61_6145

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_150_mod_5 : fib 150 % 5 = 0 := by
  sorry

end fib_150_mod_5_l61_6145


namespace ways_to_sum_3060_l61_6176

/-- Represents the number of ways to write a given number as the sum of twos and threes -/
def waysToSum (n : ℕ) : ℕ := sorry

/-- The target number we want to represent -/
def targetNumber : ℕ := 3060

/-- Theorem stating that there are 511 ways to write 3060 as the sum of twos and threes -/
theorem ways_to_sum_3060 : waysToSum targetNumber = 511 := by sorry

end ways_to_sum_3060_l61_6176


namespace prob_heart_diamond_standard_deck_l61_6106

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (num_red_suits : ℕ)
  (num_black_suits : ℕ)

/-- Standard deck properties -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    num_red_suits := 2,
    num_black_suits := 2 }

/-- Probability of drawing a heart first and a diamond second -/
def prob_heart_then_diamond (d : Deck) : ℚ :=
  (d.cards_per_suit : ℚ) / (d.total_cards : ℚ) *
  (d.cards_per_suit : ℚ) / ((d.total_cards - 1) : ℚ)

/-- Theorem stating the probability of drawing a heart then a diamond -/
theorem prob_heart_diamond_standard_deck :
  prob_heart_then_diamond standard_deck = 169 / 2652 :=
sorry

end prob_heart_diamond_standard_deck_l61_6106


namespace sum_product_difference_l61_6160

theorem sum_product_difference (x y : ℝ) : 
  x + y = 500 → x * y = 22000 → y - x = -402.5 := by
sorry

end sum_product_difference_l61_6160


namespace pipe_c_fill_time_l61_6186

/-- The time (in minutes) it takes for pipe a to fill the tank -/
def time_a : ℝ := 20

/-- The time (in minutes) it takes for pipe b to fill the tank -/
def time_b : ℝ := 20

/-- The time (in minutes) it takes for pipe c to fill the tank -/
def time_c : ℝ := 30

/-- The proportion of solution r in the tank after 3 minutes -/
def proportion_r : ℝ := 0.25

/-- The time (in minutes) after which we measure the proportion of solution r -/
def measure_time : ℝ := 3

theorem pipe_c_fill_time :
  (time_c = 30) ∧
  (measure_time * (1 / time_a + 1 / time_b + 1 / time_c) * (1 / time_c) /
   (measure_time * (1 / time_a + 1 / time_b + 1 / time_c)) = proportion_r) :=
sorry

end pipe_c_fill_time_l61_6186


namespace m_equals_two_sufficient_not_necessary_l61_6155

def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

theorem m_equals_two_sufficient_not_necessary :
  (∀ m : ℝ, m = 2 → A m ∩ B = {4}) ∧
  (∃ m : ℝ, m ≠ 2 ∧ A m ∩ B = {4}) :=
sorry

end m_equals_two_sufficient_not_necessary_l61_6155
