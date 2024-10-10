import Mathlib

namespace max_cookie_price_l25_2568

theorem max_cookie_price (k p : ℕ) 
  (h1 : 8 * k + 3 * p < 200)
  (h2 : 4 * k + 5 * p > 150) :
  k ≤ 19 ∧ ∃ (k' p' : ℕ), k' = 19 ∧ 8 * k' + 3 * p' < 200 ∧ 4 * k' + 5 * p' > 150 :=
sorry

end max_cookie_price_l25_2568


namespace part_one_part_two_l25_2538

/-- The function f(x) = mx^2 - mx - 1 --/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

/-- Theorem for part 1 of the problem --/
theorem part_one :
  ∀ x : ℝ, f (1/2) x < 0 ↔ -1 < x ∧ x < 2 := by sorry

/-- Theorem for part 2 of the problem --/
theorem part_two (m : ℝ) (x : ℝ) :
  f m x < (m - 1) * x^2 + 2 * x - 2 * m - 1 ↔
    (m < 2 ∧ m < x ∧ x < 2) ∨ (m > 2 ∧ 2 < x ∧ x < m) := by sorry

end part_one_part_two_l25_2538


namespace marias_flower_bed_area_l25_2581

/-- Represents a rectangular flower bed with fence posts --/
structure FlowerBed where
  total_posts : ℕ
  post_spacing : ℕ
  longer_side_posts : ℕ
  shorter_side_posts : ℕ

/-- Calculates the area of the flower bed --/
def flower_bed_area (fb : FlowerBed) : ℕ :=
  (fb.shorter_side_posts - 1) * fb.post_spacing * ((fb.longer_side_posts - 1) * fb.post_spacing)

/-- Theorem stating that Maria's flower bed has an area of 350 square yards --/
theorem marias_flower_bed_area :
  ∃ fb : FlowerBed,
    fb.total_posts = 24 ∧
    fb.post_spacing = 5 ∧
    fb.longer_side_posts = 3 * fb.shorter_side_posts - 1 ∧
    fb.total_posts = fb.longer_side_posts + fb.shorter_side_posts + 2 ∧
    flower_bed_area fb = 350 :=
by sorry

end marias_flower_bed_area_l25_2581


namespace squared_lengths_sum_l25_2588

/-- Two circles O and O₁, where O has equation x² + y² = 25 and O₁ has center (m, 0) -/
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 25}
def circle_O₁ (m : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - m)^2 + p.2^2 = (m - 3)^2 + 4^2}

/-- Point P where the circles intersect -/
def P : ℝ × ℝ := (3, 4)

/-- Line l with slope k passing through P -/
def line_l (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 - 4 = k * (p.1 - 3)}

/-- Line l₁ perpendicular to l passing through P -/
def line_l₁ (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 - 4 = (-1/k) * (p.1 - 3)}

/-- Points A and B where line l intersects circles O and O₁ -/
def A (k m : ℝ) : ℝ × ℝ := sorry
def B (k m : ℝ) : ℝ × ℝ := sorry

/-- Points C and D where line l₁ intersects circles O and O₁ -/
def C (k m : ℝ) : ℝ × ℝ := sorry
def D (k m : ℝ) : ℝ × ℝ := sorry

/-- The main theorem -/
theorem squared_lengths_sum (m : ℝ) (k : ℝ) (h : k ≠ 0) :
  ∀ (A B : ℝ × ℝ) (C D : ℝ × ℝ),
  A ∈ circle_O ∧ A ∈ line_l k ∧
  B ∈ circle_O₁ m ∧ B ∈ line_l k ∧
  C ∈ circle_O ∧ C ∈ line_l₁ k ∧
  D ∈ circle_O₁ m ∧ D ∈ line_l₁ k →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (C.1 - D.1)^2 + (C.2 - D.2)^2 = 4 * m^2 := by
  sorry

end squared_lengths_sum_l25_2588


namespace remaining_investment_rate_l25_2559

def total_investment : ℝ := 12000
def investment_at_7_percent : ℝ := 5500
def total_interest : ℝ := 970

def remaining_investment : ℝ := total_investment - investment_at_7_percent
def interest_from_7_percent : ℝ := investment_at_7_percent * 0.07
def interest_from_remaining : ℝ := total_interest - interest_from_7_percent

theorem remaining_investment_rate : 
  (interest_from_remaining / remaining_investment) * 100 = 9 := by
  sorry

end remaining_investment_rate_l25_2559


namespace abs_neg_five_plus_three_l25_2575

theorem abs_neg_five_plus_three : |(-5 : ℤ) + 3| = 2 := by
  sorry

end abs_neg_five_plus_three_l25_2575


namespace insecticide_potency_range_specific_insecticide_potency_range_l25_2526

/-- Given two insecticide powders, find the range of potency for the second powder
    to achieve a specific mixture potency. -/
theorem insecticide_potency_range 
  (weight1 : ℝ) (potency1 : ℝ) (weight2 : ℝ) 
  (lower_bound : ℝ) (upper_bound : ℝ) :
  weight1 > 0 ∧ weight2 > 0 ∧
  0 < potency1 ∧ potency1 < 1 ∧
  0 < lower_bound ∧ lower_bound < upper_bound ∧ upper_bound < 1 →
  ∃ (lower_x upper_x : ℝ),
    lower_x > potency1 ∧
    ∀ x, lower_x < x ∧ x < upper_x →
      lower_bound < (weight1 * potency1 + weight2 * x) / (weight1 + weight2) ∧
      (weight1 * potency1 + weight2 * x) / (weight1 + weight2) < upper_bound :=
by sorry

/-- The specific insecticide potency range problem. -/
theorem specific_insecticide_potency_range :
  ∃ (lower_x upper_x : ℝ),
    lower_x = 0.33 ∧ upper_x = 0.42 ∧
    ∀ x, 0.33 < x ∧ x < 0.42 →
      0.25 < (40 * 0.15 + 50 * x) / (40 + 50) ∧
      (40 * 0.15 + 50 * x) / (40 + 50) < 0.30 :=
by sorry

end insecticide_potency_range_specific_insecticide_potency_range_l25_2526


namespace circle_satisfies_conditions_l25_2525

-- Define the given circle
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 5 = 0

-- Define the sought circle
def sought_circle (x y : ℝ) : Prop :=
  (x - 5)^2 + (y + 1)^2 = 5

-- Define the tangent condition
def is_tangent (c1 c2 : (ℝ → ℝ → Prop)) (x y : ℝ) : Prop :=
  c1 x y ∧ c2 x y ∧ ∃ (m : ℝ), ∀ (dx dy : ℝ),
    (c1 (x + dx) (y + dy) → m * dx = dy) ∧
    (c2 (x + dx) (y + dy) → m * dx = dy)

theorem circle_satisfies_conditions :
  sought_circle 3 (-2) ∧
  is_tangent given_circle sought_circle 0 1 :=
sorry

end circle_satisfies_conditions_l25_2525


namespace age_problem_l25_2596

theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 52) : 
  b = 20 := by
sorry

end age_problem_l25_2596


namespace min_value_sum_reciprocals_l25_2539

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : 3 * a + 4 * b + 2 * c = 3) :
  (1 / (2 * a + b) + 1 / (a + 3 * c) + 1 / (4 * b + c)) ≥ (3 / 2) :=
by sorry

end min_value_sum_reciprocals_l25_2539


namespace quadratic_inequality_range_l25_2590

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) → -8 ≤ a ∧ a ≤ 0 := by
  sorry

end quadratic_inequality_range_l25_2590


namespace cargo_loaded_in_bahamas_l25_2577

/-- The amount of cargo loaded in the Bahamas -/
def cargo_loaded (initial_cargo final_cargo : ℕ) : ℕ :=
  final_cargo - initial_cargo

/-- Theorem: The amount of cargo loaded in the Bahamas is 8723 tons -/
theorem cargo_loaded_in_bahamas :
  cargo_loaded 5973 14696 = 8723 := by
  sorry

end cargo_loaded_in_bahamas_l25_2577


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l25_2555

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The first line from the problem -/
def line1 (a : ℝ) : Line :=
  { a := a, b := -1, c := 3 }

/-- The second line from the problem -/
def line2 (a : ℝ) : Line :=
  { a := 2, b := -(a+1), c := 4 }

/-- The condition a=-2 is sufficient for the lines to be parallel -/
theorem sufficient_condition :
  parallel (line1 (-2)) (line2 (-2)) := by sorry

/-- The condition a=-2 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ a : ℝ, a ≠ -2 ∧ parallel (line1 a) (line2 a) := by sorry

/-- The main theorem stating that a=-2 is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary :
  (parallel (line1 (-2)) (line2 (-2))) ∧
  (∃ a : ℝ, a ≠ -2 ∧ parallel (line1 a) (line2 a)) := by sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l25_2555


namespace intersection_M_N_l25_2515

def M : Set ℝ := {-1, 0, 1, 2, 3}
def N : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} := by sorry

end intersection_M_N_l25_2515


namespace limit_of_a_is_three_fourths_l25_2548

def a (n : ℕ) : ℚ := (3 * n^2 + 2) / (4 * n^2 - 1)

theorem limit_of_a_is_three_fourths :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/4| < ε := by
  sorry

end limit_of_a_is_three_fourths_l25_2548


namespace photo_lineup_arrangements_l25_2517

def number_of_male_actors : ℕ := 4
def number_of_female_actors : ℕ := 5

def arrangement_count (m n : ℕ) : ℕ := sorry

theorem photo_lineup_arrangements :
  arrangement_count number_of_male_actors number_of_female_actors =
    (arrangement_count number_of_female_actors number_of_female_actors) *
    (arrangement_count (number_of_female_actors + 1) number_of_male_actors) -
    2 * (arrangement_count (number_of_female_actors - 1) (number_of_female_actors - 1)) *
    (arrangement_count number_of_female_actors number_of_male_actors) :=
  sorry

end photo_lineup_arrangements_l25_2517


namespace lcm_gcd_product_24_60_l25_2552

theorem lcm_gcd_product_24_60 : Nat.lcm 24 60 * Nat.gcd 24 60 = 1440 := by
  sorry

end lcm_gcd_product_24_60_l25_2552


namespace problem_statement_l25_2543

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x * (x - 4) * (x + 1) < 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

theorem problem_statement :
  (∀ x : ℝ, x ∈ (A ∪ B 4) ↔ -1 < x ∧ x < 5) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ (U \ A) ↔ x ∈ (U \ B a)) ↔ 0 ≤ a ∧ a ≤ 3) :=
sorry

end problem_statement_l25_2543


namespace polygon_sides_count_polygon_has_2023_sides_l25_2505

/-- A polygon with the property that at most 2021 triangles can be formed
    when a diagonal is drawn from a vertex has 2023 sides. -/
theorem polygon_sides_count : ℕ :=
  2023

/-- The maximum number of triangles formed when drawing a diagonal from a vertex
    of a polygon with n sides is n - 2. -/
def max_triangles (n : ℕ) : ℕ := n - 2

/-- The condition that at most 2021 triangles can be formed. -/
axiom triangle_condition : max_triangles polygon_sides_count ≤ 2021

/-- Theorem stating that the polygon has 2023 sides. -/
theorem polygon_has_2023_sides : polygon_sides_count = 2023 := by
  sorry

#check polygon_has_2023_sides

end polygon_sides_count_polygon_has_2023_sides_l25_2505


namespace x_eq_3_sufficient_not_necessary_for_x_sq_eq_9_l25_2527

theorem x_eq_3_sufficient_not_necessary_for_x_sq_eq_9 :
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) ∧
  (∀ x : ℝ, x = 3 → x^2 = 9) :=
by sorry

end x_eq_3_sufficient_not_necessary_for_x_sq_eq_9_l25_2527


namespace product_of_squares_and_products_l25_2561

theorem product_of_squares_and_products (a b c : ℝ) 
  (sum_eq : a + b + c = 5)
  (sum_squares_eq : a^2 + b^2 + c^2 = 15)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 47) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) = 625 := by
  sorry

end product_of_squares_and_products_l25_2561


namespace inequality_proof_l25_2502

theorem inequality_proof (a b c A B C k : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_A : 0 < A) (pos_B : 0 < B) (pos_C : 0 < C)
  (sum_a : a + A = k) (sum_b : b + B = k) (sum_c : c + C = k) :
  a * B + b * C + c * A ≤ k^2 := by
sorry

end inequality_proof_l25_2502


namespace simplify_expressions_l25_2569

theorem simplify_expressions :
  (∀ x y : ℝ, x^2 - 5*y - 4*x^2 + y - 1 = -3*x^2 - 4*y - 1) ∧
  (∀ a b : ℝ, 7*a + 3*(a - 3*b) - 2*(b - 3*a) = 16*a - 11*b) :=
by
  sorry

end simplify_expressions_l25_2569


namespace min_value_sum_l25_2553

theorem min_value_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 19 / x + 98 / y = 1) :
  x + y ≥ 117 + 14 * Real.sqrt 38 := by
  sorry

end min_value_sum_l25_2553


namespace extreme_values_imply_b_zero_l25_2560

/-- A cubic function with extreme values at 1 and -1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x * (a * x^2 + b * x + c)

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem extreme_values_imply_b_zero (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : f' a b c 1 = 0) (h3 : f' a b c (-1) = 0) : b = 0 := by
  sorry

end extreme_values_imply_b_zero_l25_2560


namespace mina_pi_digits_l25_2536

/-- The number of digits of pi memorized by each person -/
structure PiDigits where
  sam : ℕ
  carlos : ℕ
  mina : ℕ

/-- The conditions of the problem -/
def problem_conditions (d : PiDigits) : Prop :=
  d.sam = d.carlos + 6 ∧
  d.mina = 6 * d.carlos ∧
  d.sam = 10

/-- The theorem to prove -/
theorem mina_pi_digits (d : PiDigits) : 
  problem_conditions d → d.mina = 24 := by
  sorry

end mina_pi_digits_l25_2536


namespace number_of_orders_is_1536_l25_2508

/-- Represents the number of letters --/
def n : ℕ := 10

/-- Represents the number of letters that can be in the stack (excluding 9 and 10) --/
def m : ℕ := 8

/-- Calculates the number of different orders for typing the remaining letters --/
def number_of_orders : ℕ :=
  Finset.sum (Finset.range (m + 1)) (λ k => (Nat.choose m k) * (k + 2))

/-- Theorem stating that the number of different orders is 1536 --/
theorem number_of_orders_is_1536 : number_of_orders = 1536 := by
  sorry

end number_of_orders_is_1536_l25_2508


namespace complex_equation_solution_l25_2545

theorem complex_equation_solution (z : ℂ) : (1 + 3*I)*z = I - 3 → z = I := by
  sorry

end complex_equation_solution_l25_2545


namespace icosagon_diagonals_from_vertex_l25_2562

/-- The number of sides in an icosagon -/
def icosagon_sides : ℕ := 20

/-- The number of diagonals from a single vertex in an icosagon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

theorem icosagon_diagonals_from_vertex :
  diagonals_from_vertex icosagon_sides = 17 := by sorry

end icosagon_diagonals_from_vertex_l25_2562


namespace coronavirus_case_ratio_l25_2547

/-- Given the number of coronavirus cases in a country during two waves, 
    prove the ratio of average daily cases between the waves. -/
theorem coronavirus_case_ratio 
  (first_wave_daily : ℕ) 
  (second_wave_total : ℕ) 
  (second_wave_days : ℕ) 
  (h1 : first_wave_daily = 300)
  (h2 : second_wave_total = 21000)
  (h3 : second_wave_days = 14) :
  (second_wave_total / second_wave_days) / first_wave_daily = 5 := by
  sorry


end coronavirus_case_ratio_l25_2547


namespace cafeteria_pies_l25_2519

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : 
  initial_apples = 62 → 
  handed_out = 8 → 
  apples_per_pie = 9 → 
  (initial_apples - handed_out) / apples_per_pie = 6 := by
sorry

end cafeteria_pies_l25_2519


namespace circle_radii_sum_l25_2580

theorem circle_radii_sum : 
  ∀ r : ℝ, 
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 14) :=
by sorry

end circle_radii_sum_l25_2580


namespace concert_revenue_l25_2542

def adult_price : ℕ := 26
def child_price : ℕ := adult_price / 2
def adult_attendees : ℕ := 183
def child_attendees : ℕ := 28

theorem concert_revenue :
  adult_price * adult_attendees + child_price * child_attendees = 5122 :=
by sorry

end concert_revenue_l25_2542


namespace smallest_eulerian_polyhedron_sum_l25_2563

/-- A polyhedron is Eulerian if it has an Eulerian path -/
def IsEulerianPolyhedron (V E F : ℕ) : Prop :=
  ∃ (oddDegreeVertices : ℕ), oddDegreeVertices = 2 ∧ 
  V ≥ 4 ∧ E ≥ 6 ∧ F ≥ 4 ∧ V - E + F = 2

/-- The sum of vertices, edges, and faces for a polyhedron -/
def PolyhedronSum (V E F : ℕ) : ℕ := V + E + F

theorem smallest_eulerian_polyhedron_sum :
  ∀ V E F : ℕ, IsEulerianPolyhedron V E F →
  PolyhedronSum V E F ≥ 20 :=
by sorry

end smallest_eulerian_polyhedron_sum_l25_2563


namespace largest_k_exists_l25_2507

def X : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => X (n + 1) + 2 * X n

def Y : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 3 * Y (n + 1) + 4 * Y n

theorem largest_k_exists : ∃! k : ℕ, k < 10^2007 ∧
  (∃ i : ℕ+, |X i - k| ≤ 2007) ∧
  (∃ j : ℕ+, |Y j - k| ≤ 2007) ∧
  ∀ m : ℕ, m > k → ¬(
    (∃ i : ℕ+, |X i - m| ≤ 2007) ∧
    (∃ j : ℕ+, |Y j - m| ≤ 2007) ∧
    m < 10^2007
  ) :=
by sorry

end largest_k_exists_l25_2507


namespace quadratic_roots_negative_l25_2597

theorem quadratic_roots_negative (p : ℝ) : 
  (∀ x : ℝ, x^2 + 2*(p+1)*x + 9*p - 5 = 0 → x < 0) ↔ 
  (p > 5/9 ∧ p ≤ 1) ∨ p ≥ 6 := by
  sorry

end quadratic_roots_negative_l25_2597


namespace sphere_surface_area_from_rectangular_solid_l25_2573

/-- The surface area of a sphere that circumscribes a rectangular solid -/
theorem sphere_surface_area_from_rectangular_solid 
  (length width height : ℝ) 
  (h_length : length = 3) 
  (h_width : width = 2) 
  (h_height : height = 1) : 
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 14 * Real.pi := by
  sorry

end sphere_surface_area_from_rectangular_solid_l25_2573


namespace real_part_of_complex_fraction_l25_2524

theorem real_part_of_complex_fraction : 
  (5 * Complex.I / (1 + 2 * Complex.I)).re = 2 := by
  sorry

end real_part_of_complex_fraction_l25_2524


namespace lcm_gcf_problem_l25_2509

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 16 = 48 → Nat.gcd n 16 = 18 → n = 54 := by
  sorry

end lcm_gcf_problem_l25_2509


namespace prime_square_plus_one_triples_l25_2557

theorem prime_square_plus_one_triples :
  ∀ a b c : ℕ,
    Prime (a^2 + 1) →
    Prime (b^2 + 1) →
    (a^2 + 1) * (b^2 + 1) = c^2 + 1 →
    ((a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3)) :=
by sorry

end prime_square_plus_one_triples_l25_2557


namespace gcd_g_x_equals_120_l25_2534

def g (x : ℤ) : ℤ := (5*x + 7)*(11*x + 3)*(17*x + 8)*(4*x + 5)

theorem gcd_g_x_equals_120 (x : ℤ) (h : ∃ k : ℤ, x = 17280 * k) :
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 120 := by
  sorry

end gcd_g_x_equals_120_l25_2534


namespace a_value_equation_solution_l25_2522

-- Define the positive number whose square root is both a+6 and 2a-9
def positive_number (a : ℝ) : Prop := ∃ n : ℝ, n > 0 ∧ (a + 6 = Real.sqrt n) ∧ (2*a - 9 = Real.sqrt n)

-- Theorem 1: Prove that a = 15
theorem a_value (a : ℝ) (h : positive_number a) : a = 15 := by sorry

-- Theorem 2: Prove that the solution to ax³-64=0 is x = 4 when a = 15
theorem equation_solution (x : ℝ) : 15 * x^3 - 64 = 0 ↔ x = 4 := by sorry

end a_value_equation_solution_l25_2522


namespace tips_fraction_l25_2586

/-- Represents the income structure of a waiter -/
structure WaiterIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the fraction of income from tips -/
def fractionFromTips (income : WaiterIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: Given the conditions, the fraction of income from tips is 9/13 -/
theorem tips_fraction (income : WaiterIncome) 
  (h : income.tips = (9 / 4) * income.salary) : 
  fractionFromTips income = 9 / 13 := by
  sorry

#check tips_fraction

end tips_fraction_l25_2586


namespace financial_equation_proof_l25_2578

-- Define variables
variable (q v j p : ℝ)

-- Define the theorem
theorem financial_equation_proof :
  (3 * q - v = 8000) →
  (q = 4) →
  (v = 4 + 50 * j) →
  (p = 2669 + (50/3) * j) := by
sorry

end financial_equation_proof_l25_2578


namespace lottery_expected_profit_l25_2541

-- Define the lottery ticket parameters
def ticket_cost : ℝ := 2
def winning_probability : ℝ := 0.01
def prize : ℝ := 50

-- Define the expected profit function
def expected_profit (cost winning_prob prize : ℝ) : ℝ :=
  winning_prob * prize - cost

-- Theorem statement
theorem lottery_expected_profit :
  expected_profit ticket_cost winning_probability prize = -1.5 := by
  sorry

end lottery_expected_profit_l25_2541


namespace raffle_ticket_sales_l25_2513

theorem raffle_ticket_sales (total_members : ℕ) (male_members : ℕ) (female_members : ℕ) 
  (total_tickets : ℕ) (female_tickets : ℕ) :
  total_members > 0 →
  male_members > 0 →
  female_members = 2 * male_members →
  total_members = male_members + female_members →
  (total_tickets : ℚ) / total_members = 66 →
  (female_tickets : ℚ) / female_members = 70 →
  (total_tickets - female_tickets : ℚ) / male_members = 66 :=
by sorry

end raffle_ticket_sales_l25_2513


namespace triangle_inequality_l25_2585

/-- For any triangle ABC with side lengths a, b, c, circumradius R, and inradius r,
    the inequality (b² + c²) / (2bc) ≤ R / (2r) holds. -/
theorem triangle_inequality (a b c R r : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (hR : 0 < R) (hr : 0 < r) (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) :
    (b^2 + c^2) / (2 * b * c) ≤ R / (2 * r) := by
  sorry

end triangle_inequality_l25_2585


namespace only_negative_three_l25_2530

theorem only_negative_three (a b c d : ℝ) : 
  a = |-3| ∧ b = -3 ∧ c = -(-3) ∧ d = 1/3 → 
  (b < 0 ∧ a ≥ 0 ∧ c ≥ 0 ∧ d > 0) := by sorry

end only_negative_three_l25_2530


namespace chess_draw_probability_l25_2566

theorem chess_draw_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.6)
  (h2 : prob_A_not_lose = 0.8) :
  prob_A_not_lose - prob_A_win = 0.2 := by
  sorry

end chess_draw_probability_l25_2566


namespace no_positive_integer_solutions_l25_2572

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ+), x^2 + y^2 + x = 2 * x^3 := by sorry

end no_positive_integer_solutions_l25_2572


namespace distance_center_M_to_line_L_is_zero_l25_2591

/-- The circle M -/
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 1 = 0

/-- The line L -/
def line_L (t x y : ℝ) : Prop :=
  x = 4*t + 3 ∧ y = 3*t + 1

/-- The center of circle M -/
def center_M : ℝ × ℝ :=
  (1, 2)

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2)

theorem distance_center_M_to_line_L_is_zero :
  distance_point_to_line center_M 3 (-4) 5 = 0 := by sorry

end distance_center_M_to_line_L_is_zero_l25_2591


namespace two_reciprocal_sets_l25_2511

-- Define a reciprocal set
def ReciprocalSet (A : Set ℝ) : Prop :=
  A.Nonempty ∧ (0 ∉ A) ∧ ∀ x ∈ A, (1 / x) ∈ A

-- Define the three sets
def Set1 (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + 1 = 0}

def Set2 : Set ℝ := {x : ℝ | x^2 - 4*x + 1 < 0}

def Set3 : Set ℝ := {y : ℝ | ∃ x : ℝ, 
  (0 ≤ x ∧ x < 1 ∧ y = 2*x + 2/5) ∨ 
  (1 ≤ x ∧ x ≤ 2 ∧ y = x + 1/x)}

-- Theorem to prove
theorem two_reciprocal_sets : 
  ∃ (a : ℝ), (ReciprocalSet (Set2) ∧ ReciprocalSet (Set3) ∧ ¬ReciprocalSet (Set1 a)) ∨
             (ReciprocalSet (Set1 a) ∧ ReciprocalSet (Set2) ∧ ¬ReciprocalSet (Set3)) ∨
             (ReciprocalSet (Set1 a) ∧ ReciprocalSet (Set3) ∧ ¬ReciprocalSet (Set2)) :=
sorry

end two_reciprocal_sets_l25_2511


namespace min_area_line_equation_l25_2501

/-- The equation of the line passing through (3, 1) that minimizes the area of the triangle formed by its x and y intercepts and the origin --/
theorem min_area_line_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ (x y : ℝ), x / a + y / b = 1 → (3 / a + 1 / b = 1)) ∧
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 
    (∀ (x y : ℝ), x / a' + y / b' = 1 → (3 / a' + 1 / b' = 1)) →
    a * b ≤ a' * b') ∧
  a = 6 ∧ b = 2 :=
by sorry

end min_area_line_equation_l25_2501


namespace divisor_problem_l25_2504

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 149 →
  quotient = 9 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 16 := by
sorry

end divisor_problem_l25_2504


namespace stream_current_is_six_l25_2500

/-- Represents the man's rowing scenario -/
structure RowingScenario where
  r : ℝ  -- man's usual rowing speed in still water (miles per hour)
  w : ℝ  -- speed of the stream's current (miles per hour)

/-- The conditions of the rowing problem -/
def rowing_conditions (s : RowingScenario) : Prop :=
  -- Downstream time is 6 hours less than upstream time
  18 / (s.r + s.w) + 6 = 18 / (s.r - s.w) ∧
  -- When rowing speed is tripled, downstream time is 2 hours less than upstream time
  18 / (3 * s.r + s.w) + 2 = 18 / (3 * s.r - s.w)

/-- The theorem stating that the stream's current is 6 miles per hour -/
theorem stream_current_is_six (s : RowingScenario) :
  rowing_conditions s → s.w = 6 := by
  sorry

end stream_current_is_six_l25_2500


namespace matrix_power_four_l25_2594

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_power_four :
  A^4 = !![(-4), 6; (-6), 5] := by sorry

end matrix_power_four_l25_2594


namespace least_sum_of_bases_l25_2510

/-- Represent a number in a given base --/
def baseRepresentation (n : ℕ) (base : ℕ) : ℕ := 
  (n / base) * base + (n % base)

/-- The problem statement --/
theorem least_sum_of_bases : 
  ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ 
  baseRepresentation 58 c = baseRepresentation 85 d ∧
  (∀ (c' d' : ℕ), c' > 0 → d' > 0 → 
    baseRepresentation 58 c' = baseRepresentation 85 d' → 
    c + d ≤ c' + d') ∧
  c + d = 15 :=
sorry

end least_sum_of_bases_l25_2510


namespace sum_of_two_equals_third_l25_2592

theorem sum_of_two_equals_third (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = a + c ∨ c = a + b := by
  sorry

end sum_of_two_equals_third_l25_2592


namespace simplify_fraction_l25_2520

theorem simplify_fraction : 4 * (14 / 5) * (20 / -42) = -(4 / 15) := by
  sorry

end simplify_fraction_l25_2520


namespace simplify_fraction_solve_inequality_system_l25_2544

-- Problem 1
theorem simplify_fraction (m n : ℝ) (hm : m ≠ 0) (hmn : 9*m^2 ≠ 4*n^2) :
  (1/(3*m-2*n) - 1/(3*m+2*n)) / (m*n/((9*m^2)-(4*n^2))) = 4/m := by sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) :
  (3*x + 10 > 5*x - 2*(5-x) ∧ (x+3)/5 > 1-x) ↔ (1/3 < x ∧ x < 5) := by sorry

end simplify_fraction_solve_inequality_system_l25_2544


namespace unique_g_2_value_l25_2595

theorem unique_g_2_value (g : ℤ → ℤ) 
  (h : ∀ m n : ℤ, g (m + n) + g (m * n + 1) = g m * g n + 1) : 
  ∃! x : ℤ, g 2 = x ∧ x = 1 := by sorry

end unique_g_2_value_l25_2595


namespace equation_D_is_quadratic_l25_2564

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (a b c : ℝ) : Prop := a ≠ 0

/-- The equation 3x² + 1 = 0 -/
def equation_D : ℝ → Prop := fun x ↦ 3 * x^2 + 1 = 0

theorem equation_D_is_quadratic :
  is_quadratic_equation 3 0 1 := by sorry

end equation_D_is_quadratic_l25_2564


namespace k_range_for_negative_sum_l25_2583

/-- A power function that passes through the point (3, 27) -/
def f (x : ℝ) : ℝ := x^3

/-- The theorem stating the range of k for which f(k^2 + 3) + f(9 - 8k) < 0 holds -/
theorem k_range_for_negative_sum (k : ℝ) :
  f (k^2 + 3) + f (9 - 8*k) < 0 ↔ 2 < k ∧ k < 6 := by
  sorry


end k_range_for_negative_sum_l25_2583


namespace intersected_cubes_count_l25_2587

/-- Represents a 3x3x3 cube composed of unit cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  diagonal : Real

/-- Represents a plane that bisects the diagonal of the large cube -/
structure BisectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

/-- Represents the number of unit cubes intersected by the bisecting plane -/
def intersected_cubes (cube : LargeCube) (plane : BisectingPlane) : Nat :=
  sorry

/-- Main theorem: A plane perpendicular to and bisecting a space diagonal of a 3x3x3 cube
    intersects exactly 19 of the unit cubes -/
theorem intersected_cubes_count 
  (cube : LargeCube) 
  (plane : BisectingPlane) 
  (h1 : cube.size = 3)
  (h2 : cube.total_cubes = 27)
  (h3 : plane.perpendicular_to_diagonal)
  (h4 : plane.bisects_diagonal) :
  intersected_cubes cube plane = 19 := by
  sorry

end intersected_cubes_count_l25_2587


namespace binomial_product_l25_2535

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binomial_product_l25_2535


namespace orange_juice_fraction_l25_2551

theorem orange_juice_fraction (pitcher_capacity : ℚ) 
  (orange_juice_fraction : ℚ) (apple_juice_fraction : ℚ) : 
  pitcher_capacity > 0 →
  orange_juice_fraction = 1/4 →
  apple_juice_fraction = 3/8 →
  (orange_juice_fraction * pitcher_capacity) / (2 * pitcher_capacity) = 1/8 := by
  sorry

end orange_juice_fraction_l25_2551


namespace tarts_distribution_l25_2512

/-- Represents the number of tarts eaten by a child in a 10-minute interval -/
structure EatingRate :=
  (tarts : ℕ)

/-- Represents the total eating time in minutes -/
def total_time : ℕ := 90

/-- Represents the total number of tarts eaten -/
def total_tarts : ℕ := 35

/-- Zhenya's eating rate -/
def zhenya_rate : EatingRate := ⟨5⟩

/-- Sasha's eating rate -/
def sasha_rate : EatingRate := ⟨3⟩

/-- Calculates the number of tarts eaten by a child given their eating rate and number of 10-minute intervals -/
def tarts_eaten (rate : EatingRate) (intervals : ℕ) : ℕ := rate.tarts * intervals

/-- The main theorem to prove -/
theorem tarts_distribution :
  ∃ (zhenya_intervals sasha_intervals : ℕ),
    zhenya_intervals + sasha_intervals = total_time / 10 ∧
    tarts_eaten zhenya_rate zhenya_intervals + tarts_eaten sasha_rate sasha_intervals = total_tarts ∧
    tarts_eaten zhenya_rate zhenya_intervals = 20 ∧
    tarts_eaten sasha_rate sasha_intervals = 15 :=
sorry

end tarts_distribution_l25_2512


namespace fathers_age_l25_2532

theorem fathers_age (man_age father_age : ℕ) : 
  man_age = (2 * father_age) / 5 →
  man_age + 5 = (father_age + 5) / 2 →
  father_age = 25 := by
  sorry

end fathers_age_l25_2532


namespace cos_sin_75_product_l25_2567

theorem cos_sin_75_product (θ : Real) (h : θ = 75 * π / 180) : 
  (Real.cos θ + Real.sin θ) * (Real.cos θ - Real.sin θ) = -Real.sqrt 3 / 2 := by
  sorry

end cos_sin_75_product_l25_2567


namespace sum_of_coefficients_l25_2599

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 36 := by
sorry

end sum_of_coefficients_l25_2599


namespace second_vessel_capacity_l25_2589

/-- Proves that the capacity of the second vessel is 6 liters given the problem conditions -/
theorem second_vessel_capacity : 
  ∀ (vessel2_capacity : ℝ),
    -- Given conditions
    let vessel1_capacity : ℝ := 2
    let vessel1_concentration : ℝ := 0.25
    let vessel2_concentration : ℝ := 0.40
    let total_liquid : ℝ := 8
    let final_vessel_capacity : ℝ := 10
    let final_concentration : ℝ := 0.29000000000000004

    -- Total liquid equation
    vessel1_capacity + vessel2_capacity = total_liquid →
    
    -- Alcohol balance equation
    (vessel1_capacity * vessel1_concentration + 
     vessel2_capacity * vessel2_concentration) / final_vessel_capacity = final_concentration →
    
    -- Conclusion
    vessel2_capacity = 6 := by
  sorry

end second_vessel_capacity_l25_2589


namespace new_person_weight_l25_2518

theorem new_person_weight (W : ℝ) :
  let initial_avg := W / 20
  let intermediate_avg := (W - 95) / 19
  let final_avg := initial_avg + 4.2
  let new_person_weight := (final_avg * 20) - (W - 95)
  new_person_weight = 179 := by
sorry

end new_person_weight_l25_2518


namespace tree_planting_speeds_l25_2516

-- Define the given constants
def distance : ℝ := 10
def time_difference : ℝ := 1.5
def speed_ratio : ℝ := 2.5

-- Define the walking speed and cycling speed
def walking_speed : ℝ := 4
def cycling_speed : ℝ := 10

-- Define the increased cycling speed
def increased_cycling_speed : ℝ := 12

-- Theorem statement
theorem tree_planting_speeds :
  (distance / walking_speed - distance / cycling_speed = time_difference) ∧
  (cycling_speed = speed_ratio * walking_speed) ∧
  (distance / increased_cycling_speed = distance / cycling_speed - 1/6) :=
sorry

end tree_planting_speeds_l25_2516


namespace mans_age_twice_sons_age_l25_2533

/-- Proves that the number of years until a man's age is twice his son's age is 2,
    given the man is 22 years older than his son and the son's present age is 20 years. -/
theorem mans_age_twice_sons_age (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 20 → age_difference = 22 → 
  ∃ (years : ℕ), years = 2 ∧ 
    (son_age + years) * 2 = (son_age + age_difference + years) := by
  sorry

end mans_age_twice_sons_age_l25_2533


namespace cubic_difference_l25_2584

theorem cubic_difference (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
sorry

end cubic_difference_l25_2584


namespace john_cookies_left_l25_2550

/-- The number of cookies John has left after sharing with his friend -/
def cookies_left : ℕ :=
  let initial_cookies : ℕ := 2 * 12
  let after_first_day : ℕ := initial_cookies - (initial_cookies / 4)
  let after_second_day : ℕ := after_first_day - 5
  let shared_cookies : ℕ := after_second_day / 3
  after_second_day - shared_cookies

theorem john_cookies_left : cookies_left = 9 := by
  sorry

end john_cookies_left_l25_2550


namespace smallest_positive_solution_l25_2571

theorem smallest_positive_solution (x : ℕ) : x = 21 ↔ 
  (x > 0 ∧ 
   (45 * x + 7) % 25 = 3 ∧ 
   ∀ y : ℕ, y > 0 → y < x → (45 * y + 7) % 25 ≠ 3) :=
by sorry

end smallest_positive_solution_l25_2571


namespace parabola_directrix_l25_2579

/-- Given a parabola x² = ay with directrix y = -1/4, prove that a = 1 -/
theorem parabola_directrix (x y a : ℝ) : 
  (x^2 = a * y) →  -- Parabola equation
  (y = -1/4 → a = 1) :=  -- Directrix equation implies a = 1
by sorry

end parabola_directrix_l25_2579


namespace weight_of_A_l25_2582

theorem weight_of_A (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 7 →
  (b + c + d + e) / 4 = 79 →
  a = 79 :=
by sorry

end weight_of_A_l25_2582


namespace fixed_point_on_line_l25_2529

theorem fixed_point_on_line (a b : ℝ) (h : a + b = 1) :
  2 * a * (1/2) - b * (-1) = 1 := by sorry

end fixed_point_on_line_l25_2529


namespace a_positive_necessary_not_sufficient_l25_2593

theorem a_positive_necessary_not_sufficient :
  (∀ a : ℝ, a^2 < a → a > 0) ∧
  (∃ a : ℝ, a > 0 ∧ a^2 ≥ a) :=
by sorry

end a_positive_necessary_not_sufficient_l25_2593


namespace cubic_inequality_with_equality_l25_2549

theorem cubic_inequality_with_equality (a b : ℝ) :
  a < b → a^3 - 3*a ≤ b^3 - 3*b + 4 ∧
  (a = -1 ∧ b = 1 → a^3 - 3*a = b^3 - 3*b + 4) :=
by sorry

end cubic_inequality_with_equality_l25_2549


namespace monotonic_function_upper_bound_l25_2570

open Real

/-- A monotonic function on (0, +∞) satisfying certain conditions -/
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ 
  (∀ x > 0, f (f x - exp x + x) = exp 1) ∧
  (∀ x > 0, DifferentiableAt ℝ f x)

/-- The theorem stating the upper bound of a -/
theorem monotonic_function_upper_bound 
  (f : ℝ → ℝ) 
  (hf : MonotonicFunction f) 
  (h : ∀ x > 0, f x + deriv f x ≥ (a : ℝ) * x) :
  a ≤ 2 * exp 1 - 1 := by
  sorry

end monotonic_function_upper_bound_l25_2570


namespace sum_of_four_cubes_1998_l25_2546

theorem sum_of_four_cubes_1998 : ∃ (a b c d : ℤ), 1998 = a^3 + b^3 + c^3 + d^3 := by
  sorry

end sum_of_four_cubes_1998_l25_2546


namespace line_l_equation_l25_2537

-- Define points A and B
def A : ℝ × ℝ := (3, 3)
def B : ℝ × ℝ := (5, 2)

-- Define lines l1 and l2
def l1 (x y : ℝ) : Prop := 3 * x - y - 1 = 0
def l2 (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the intersection point of l1 and l2
def intersection : ℝ × ℝ := (1, 2)

-- Define the property that l passes through the intersection
def passes_through_intersection (l : ℝ → ℝ → Prop) : Prop :=
  l (intersection.1) (intersection.2)

-- Define the property of equal distance from A and B to l
def equal_distance (l : ℝ → ℝ → Prop) : Prop :=
  ∃ d : ℝ, d > 0 ∧
    (∃ x y : ℝ, l x y ∧ (x - A.1)^2 + (y - A.2)^2 = d^2) ∧
    (∃ x y : ℝ, l x y ∧ (x - B.1)^2 + (y - B.2)^2 = d^2)

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := x - 6 * y + 11 = 0 ∨ x + 2 * y - 5 = 0

-- Theorem statement
theorem line_l_equation :
  ∀ l : ℝ → ℝ → Prop,
    passes_through_intersection l →
    equal_distance l →
    (∀ x y : ℝ, l x y ↔ line_l x y) :=
sorry

end line_l_equation_l25_2537


namespace honey_harvest_increase_l25_2506

/-- Proves that the increase in honey harvest is 6085 pounds -/
theorem honey_harvest_increase 
  (last_year_harvest : ℕ) 
  (this_year_harvest : ℕ) 
  (h1 : last_year_harvest = 2479)
  (h2 : this_year_harvest = 8564) : 
  this_year_harvest - last_year_harvest = 6085 := by
  sorry

end honey_harvest_increase_l25_2506


namespace total_bananas_is_110_l25_2576

/-- The total number of bananas Willie, Charles, and Lucy had originally -/
def total_bananas (willie_bananas charles_bananas lucy_bananas : ℕ) : ℕ :=
  willie_bananas + charles_bananas + lucy_bananas

/-- Theorem stating that the total number of bananas is 110 -/
theorem total_bananas_is_110 :
  total_bananas 48 35 27 = 110 := by
  sorry

end total_bananas_is_110_l25_2576


namespace inequality_proof_l25_2523

theorem inequality_proof (a b c : ℝ) (h : a * b < 0) :
  a^2 + b^2 + c^2 > 2*a*b + 2*b*c + 2*c*a := by
  sorry

end inequality_proof_l25_2523


namespace product_remainder_l25_2528

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

theorem product_remainder (a₁ : ℕ) (d : ℕ) (n : ℕ) (m : ℕ) :
  a₁ = 2 → d = 10 → n = 21 → m = 7 →
  (arithmetic_sequence a₁ d n).prod % m = 1 := by
  sorry

end product_remainder_l25_2528


namespace num_al_sandwiches_l25_2574

/-- Represents the number of different types of bread available. -/
def num_breads : Nat := 5

/-- Represents the number of different types of meat available. -/
def num_meats : Nat := 6

/-- Represents the number of different types of cheese available. -/
def num_cheeses : Nat := 6

/-- Represents the number of forbidden combinations. -/
def num_forbidden : Nat := 3

/-- Represents the number of overcounted combinations. -/
def num_overcounted : Nat := 1

/-- Calculates the total number of possible sandwich combinations. -/
def total_combinations : Nat := num_breads * num_meats * num_cheeses

/-- Calculates the number of forbidden sandwich combinations. -/
def forbidden_combinations : Nat :=
  num_breads + num_cheeses + num_cheeses - num_overcounted

/-- Theorem stating the number of different sandwiches Al can order. -/
theorem num_al_sandwiches : 
  total_combinations - forbidden_combinations = 164 := by
  sorry

end num_al_sandwiches_l25_2574


namespace train_length_l25_2554

/-- The length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length (v : ℝ) (t : ℝ) (bridge_length : ℝ) (h1 : v = 70) (h2 : t = 13.884603517432893) (h3 : bridge_length = 150) :
  ∃ (train_length : ℝ), abs (train_length - 120) < 1 := by
  sorry

end train_length_l25_2554


namespace union_and_intersection_when_m_is_3_intersection_empty_iff_l25_2556

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m}

-- Theorem for part 1
theorem union_and_intersection_when_m_is_3 :
  (A ∪ B 3 = {x | -2 ≤ x ∧ x < 6}) ∧ (A ∩ B 3 = ∅) := by sorry

-- Theorem for part 2
theorem intersection_empty_iff :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ≤ 1 ∨ m ≥ 3 := by sorry

end union_and_intersection_when_m_is_3_intersection_empty_iff_l25_2556


namespace complex_in_first_quadrant_l25_2531

theorem complex_in_first_quadrant (m : ℝ) (h : m < 1) :
  let z : ℂ := (1 - m) + I
  z.re > 0 ∧ z.im > 0 :=
by sorry

end complex_in_first_quadrant_l25_2531


namespace solution_set_x_abs_x_minus_one_l25_2514

theorem solution_set_x_abs_x_minus_one (x : ℝ) :
  {x : ℝ | x * |x - 1| > 0} = {x : ℝ | 0 < x ∧ x ≠ 1} := by
  sorry

end solution_set_x_abs_x_minus_one_l25_2514


namespace farm_animals_after_transaction_l25_2565

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  horses : ℕ
  cows : ℕ

/-- Represents the ratio of horses to cows -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

def initial_ratio : Ratio := { numerator := 3, denominator := 1 }
def final_ratio : Ratio := { numerator := 5, denominator := 3 }

def transaction (farm : FarmAnimals) : FarmAnimals :=
  { horses := farm.horses - 15, cows := farm.cows + 15 }

theorem farm_animals_after_transaction (farm : FarmAnimals) :
  farm.horses / farm.cows = initial_ratio.numerator / initial_ratio.denominator →
  (transaction farm).horses / (transaction farm).cows = final_ratio.numerator / final_ratio.denominator →
  (transaction farm).horses - (transaction farm).cows = 30 :=
by sorry

end farm_animals_after_transaction_l25_2565


namespace initial_average_production_l25_2558

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) 
  (h1 : n = 1)
  (h2 : today_production = 60)
  (h3 : new_average = 55) :
  ∃ initial_average : ℕ, initial_average = 50 ∧ 
    (initial_average * n + today_production) / (n + 1) = new_average := by
  sorry

end initial_average_production_l25_2558


namespace not_hearing_favorite_song_probability_l25_2503

/-- Represents the duration of a song in seconds -/
def SongDuration := ℕ

/-- Represents a playlist of songs -/
def Playlist := List SongDuration

/-- Calculates the duration of the nth song in the sequence -/
def nthSongDuration (n : ℕ) : SongDuration :=
  45 + 15 * n

/-- Generates a playlist of 12 songs with increasing durations -/
def generatePlaylist : Playlist :=
  List.range 12 |>.map nthSongDuration

/-- The duration of the favorite song in seconds -/
def favoriteSongDuration : SongDuration := 4 * 60

/-- The total duration we're interested in (5 minutes in seconds) -/
def totalDuration : SongDuration := 5 * 60

/-- Calculates the probability of not hearing the entire favorite song 
    within the first 5 minutes of a random playlist -/
def probabilityNotHearingFavoriteSong (playlist : Playlist) (favoriteDuration : SongDuration) (totalDuration : SongDuration) : ℚ :=
  sorry

theorem not_hearing_favorite_song_probability :
  probabilityNotHearingFavoriteSong generatePlaylist favoriteSongDuration totalDuration = 65 / 66 := by
  sorry

end not_hearing_favorite_song_probability_l25_2503


namespace tim_stacked_bales_l25_2598

theorem tim_stacked_bales (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 28)
  (h2 : final_bales = 82) :
  final_bales - initial_bales = 54 := by
  sorry

end tim_stacked_bales_l25_2598


namespace second_number_proof_l25_2540

theorem second_number_proof (a b c : ℝ) 
  (sum_eq : a + b + c = 98)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 5 / 8)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c) :
  b = 30 := by
sorry

end second_number_proof_l25_2540


namespace ground_beef_cost_l25_2521

/-- The cost of ground beef in dollars per kilogram -/
def price_per_kg : ℝ := 5

/-- The quantity of ground beef in kilograms -/
def quantity : ℝ := 12

/-- The total cost of ground beef -/
def total_cost : ℝ := price_per_kg * quantity

theorem ground_beef_cost : total_cost = 60 := by
  sorry

end ground_beef_cost_l25_2521
