import Mathlib

namespace product_less_than_60_probability_l266_26691

def paco_range : Finset ℕ := Finset.range 5
def manu_range : Finset ℕ := Finset.range 20

def total_outcomes : ℕ := paco_range.card * manu_range.card

def favorable_outcomes : ℕ :=
  (paco_range.filter (fun p => p + 1 ≤ 2)).sum (fun p =>
    (manu_range.filter (fun m => (p + 1) * (m + 1) < 60)).card)
  +
  (paco_range.filter (fun p => p + 1 > 2)).sum (fun p =>
    (manu_range.filter (fun m => (p + 1) * (m + 1) < 60)).card)

theorem product_less_than_60_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 21 / 25 := by sorry

end product_less_than_60_probability_l266_26691


namespace chord_equation_l266_26613

/-- Given positive real numbers m, n, s, t satisfying certain conditions,
    prove that the equation of a line containing a chord of an ellipse is 2x + y - 4 = 0 -/
theorem chord_equation (m n s t : ℝ) (hm : m > 0) (hn : n > 0) (hs : s > 0) (ht : t > 0)
  (h_sum : m + n = 3)
  (h_frac : m / s + n / t = 1)
  (h_order : m < n)
  (h_min : ∀ (s' t' : ℝ), s' > 0 → t' > 0 → m / s' + n / t' = 1 → s' + t' ≥ 3 + 2 * Real.sqrt 2)
  (h_midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2 / 4 + y₁^2 / 16 = 1 ∧
    x₂^2 / 4 + y₂^2 / 16 = 1 ∧
    (x₁ + x₂) / 2 = m ∧
    (y₁ + y₂) / 2 = n) :
  ∃ (a b c : ℝ), a * m + b * n + c = 0 ∧ 2 * a + b = 0 ∧ a ≠ 0 := by
  sorry


end chord_equation_l266_26613


namespace greatest_divisor_four_consecutive_integers_l266_26687

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m = 12 ∧ 
  (∀ k : ℕ, k > m → ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3)))) ∧
  (12 ∣ (n * (n + 1) * (n + 2) * (n + 3))) := by
sorry

end greatest_divisor_four_consecutive_integers_l266_26687


namespace sphere_volume_l266_26690

theorem sphere_volume (R : ℝ) (x y : ℝ) : 
  R > 0 ∧ 
  x ≠ y ∧
  R^2 = x^2 + 5 ∧ 
  R^2 = y^2 + 8 ∧ 
  |x - y| = 1 →
  (4/3) * Real.pi * R^3 = 36 * Real.pi :=
by sorry

end sphere_volume_l266_26690


namespace max_overlap_area_isosceles_triangles_l266_26631

/-- The maximal area of overlap between two congruent right-angled isosceles triangles -/
theorem max_overlap_area_isosceles_triangles :
  ∃ (overlap_area : ℝ),
    overlap_area = 2/9 ∧
    ∀ (x : ℝ),
      0 ≤ x ∧ x ≤ 1 →
      let triangle_area := 1/4 * (1 - x)^2
      let pentagon_area := 1/4 * (1 - x) * (3*x + 1)
      overlap_area ≥ max triangle_area pentagon_area :=
by sorry

end max_overlap_area_isosceles_triangles_l266_26631


namespace maxwells_walking_speed_l266_26684

/-- Proves that Maxwell's walking speed is 3 km/h given the problem conditions --/
theorem maxwells_walking_speed 
  (total_distance : ℝ) 
  (maxwell_distance : ℝ) 
  (brad_speed : ℝ) 
  (h1 : total_distance = 36) 
  (h2 : maxwell_distance = 12) 
  (h3 : brad_speed = 6) : 
  maxwell_distance / (total_distance - maxwell_distance) * brad_speed = 3 := by
  sorry

end maxwells_walking_speed_l266_26684


namespace sqrt_sum_equals_ten_l266_26647

theorem sqrt_sum_equals_ten : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end sqrt_sum_equals_ten_l266_26647


namespace h1n1_diameter_scientific_notation_l266_26604

theorem h1n1_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000081 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -8 :=
sorry

end h1n1_diameter_scientific_notation_l266_26604


namespace max_area_rectangle_with_fixed_perimeter_l266_26617

/-- The maximum area of a rectangle with integer side lengths and perimeter 160 feet -/
theorem max_area_rectangle_with_fixed_perimeter :
  ∃ (w h : ℕ), 
    (2 * w + 2 * h = 160) ∧ 
    (∀ (x y : ℕ), (2 * x + 2 * y = 160) → (x * y ≤ w * h)) ∧
    (w * h = 1600) := by
  sorry

end max_area_rectangle_with_fixed_perimeter_l266_26617


namespace calculator_squaring_min_presses_1000_eq_3_l266_26692

def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | m + 1 => (repeated_square x m) ^ 2

theorem calculator_squaring (target : ℕ) : 
  ∃ (n : ℕ), repeated_square 3 n > target ∧ 
  ∀ (m : ℕ), m < n → repeated_square 3 m ≤ target := by
  sorry

def min_presses (target : ℕ) : ℕ :=
  Nat.find (calculator_squaring target)

theorem min_presses_1000_eq_3 : min_presses 1000 = 3 := by
  sorry

end calculator_squaring_min_presses_1000_eq_3_l266_26692


namespace factor_expression_l266_26620

theorem factor_expression (b : ℝ) : 45 * b^2 + 135 * b^3 = 45 * b^2 * (1 + 3 * b) := by
  sorry

end factor_expression_l266_26620


namespace b_95_mod_121_l266_26648

/-- Calculate b₉₅ modulo 121 where bₙ = 5ⁿ + 11ⁿ -/
theorem b_95_mod_121 : (5^95 + 11^95) % 121 = 16 := by
  sorry

end b_95_mod_121_l266_26648


namespace cylinder_cone_surface_area_l266_26666

/-- The total surface area of a cylinder topped with a cone -/
theorem cylinder_cone_surface_area (h_cyl h_cone r : ℝ) (h_cyl_pos : h_cyl > 0) (h_cone_pos : h_cone > 0) (r_pos : r > 0) :
  let cylinder_base_area := π * r^2
  let cylinder_lateral_area := 2 * π * r * h_cyl
  let cone_slant_height := Real.sqrt (r^2 + h_cone^2)
  let cone_lateral_area := π * r * cone_slant_height
  cylinder_base_area + cylinder_lateral_area + cone_lateral_area = 175 * π + 5 * π * Real.sqrt 89 :=
by sorry

end cylinder_cone_surface_area_l266_26666


namespace largest_n_divisible_by_seven_l266_26618

theorem largest_n_divisible_by_seven : 
  ∀ n : ℕ, n < 50000 → 
  (6 * (n - 3)^3 - n^2 + 10*n - 15) % 7 = 0 → 
  n ≤ 49999 ∧ 
  (6 * (49999 - 3)^3 - 49999^2 + 10*49999 - 15) % 7 = 0 :=
by sorry

end largest_n_divisible_by_seven_l266_26618


namespace supplement_of_complement_of_75_degrees_l266_26646

/-- Given a 75-degree angle, prove that the degree measure of the supplement of its complement is 165°. -/
theorem supplement_of_complement_of_75_degrees :
  let angle : ℝ := 75
  let complement : ℝ := 90 - angle
  let supplement : ℝ := 180 - complement
  supplement = 165 := by
  sorry

end supplement_of_complement_of_75_degrees_l266_26646


namespace square_exterior_points_distance_l266_26606

/-- Given a square ABCD with side length 10 and exterior points E and F,
    prove that EF^2 = 850 + 250√125 when BE = DF = 7 and AE = CF = 15 -/
theorem square_exterior_points_distance (A B C D E F : ℝ × ℝ) : 
  let side_length : ℝ := 10
  let be_df_length : ℝ := 7
  let ae_cf_length : ℝ := 15
  -- Square ABCD definition
  (A = (0, side_length) ∧ 
   B = (side_length, side_length) ∧ 
   C = (side_length, 0) ∧ 
   D = (0, 0)) →
  -- E and F are exterior points
  (E.1 > side_length ∧ E.2 = side_length) →
  (F.1 = 0 ∧ F.2 < 0) →
  -- BE and DF lengths
  (dist B E = be_df_length ∧ dist D F = be_df_length) →
  -- AE and CF lengths
  (dist A E = ae_cf_length ∧ dist C F = ae_cf_length) →
  -- Conclusion: EF^2 = 850 + 250√125
  dist E F ^ 2 = 850 + 250 * Real.sqrt 125 :=
by sorry


end square_exterior_points_distance_l266_26606


namespace partial_fraction_decomposition_l266_26694

theorem partial_fraction_decomposition (x P Q R : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  5 * x / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2 ↔ P = 5 ∧ Q = -5 ∧ R = -5 := by
  sorry

end partial_fraction_decomposition_l266_26694


namespace arithmetic_mean_of_fractions_l266_26614

theorem arithmetic_mean_of_fractions :
  (5/6 : ℚ) = (7/9 + 8/9) / 2 := by sorry

end arithmetic_mean_of_fractions_l266_26614


namespace triangulation_labeling_exists_l266_26622

/-- A convex polygon with n+1 vertices -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin (n+1) → ℝ × ℝ

/-- A triangulation of a convex polygon -/
structure Triangulation (n : ℕ) where
  polygon : ConvexPolygon n
  triangles : Fin (n-1) → Fin 3 → Fin (n+1)

/-- A labeling of triangles in a triangulation -/
def Labeling (n : ℕ) := Fin (n-1) → Fin (n-1)

/-- Predicate to check if a vertex is part of a triangle -/
def isVertexOfTriangle (n : ℕ) (t : Triangulation n) (v : Fin (n+1)) (tri : Fin (n-1)) : Prop :=
  ∃ i : Fin 3, t.triangles tri i = v

/-- Main theorem statement -/
theorem triangulation_labeling_exists (n : ℕ) (t : Triangulation n) :
  ∃ l : Labeling n, ∀ i : Fin (n-1), isVertexOfTriangle n t i (l i) :=
sorry

end triangulation_labeling_exists_l266_26622


namespace sum_of_digits_of_greatest_prime_divisor_l266_26670

def n : ℕ := 16385

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor n) = 13 := by sorry

end sum_of_digits_of_greatest_prime_divisor_l266_26670


namespace concyclicity_equivalence_l266_26655

-- Define the points
variable (A B C D P E F G H O₁ O₂ O₃ O₄ : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D P : EuclideanPlane) : Prop := sorry

-- Define midpoints
def is_midpoint (M A B : EuclideanPlane) : Prop := sorry

-- Define circumcenter
def is_circumcenter (O P Q R : EuclideanPlane) : Prop := sorry

-- Define concyclicity
def are_concyclic (P Q R S : EuclideanPlane) : Prop := sorry

-- Main theorem
theorem concyclicity_equivalence 
  (h_quad : is_convex_quadrilateral A B C D)
  (h_diag : diagonals_intersect_at A B C D P)
  (h_mid_E : is_midpoint E A B)
  (h_mid_F : is_midpoint F B C)
  (h_mid_G : is_midpoint G C D)
  (h_mid_H : is_midpoint H D A)
  (h_circ_O₁ : is_circumcenter O₁ P H E)
  (h_circ_O₂ : is_circumcenter O₂ P E F)
  (h_circ_O₃ : is_circumcenter O₃ P F G)
  (h_circ_O₄ : is_circumcenter O₄ P G H) :
  are_concyclic O₁ O₂ O₃ O₄ ↔ are_concyclic A B C D := by sorry

end concyclicity_equivalence_l266_26655


namespace find_a_l266_26608

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

theorem find_a : ∃ (a : ℝ), A ∩ B a = {3} → a = 1 := by
  sorry

end find_a_l266_26608


namespace inflation_cost_increase_l266_26632

def original_lumber_cost : ℝ := 450
def original_nails_cost : ℝ := 30
def original_fabric_cost : ℝ := 80
def lumber_inflation_rate : ℝ := 0.20
def nails_inflation_rate : ℝ := 0.10
def fabric_inflation_rate : ℝ := 0.05

def total_increased_cost : ℝ :=
  (original_lumber_cost * lumber_inflation_rate) +
  (original_nails_cost * nails_inflation_rate) +
  (original_fabric_cost * fabric_inflation_rate)

theorem inflation_cost_increase :
  total_increased_cost = 97 := by sorry

end inflation_cost_increase_l266_26632


namespace intersection_x_product_l266_26650

/-- Given a line y = mx + k and a parabola y = ax² + bx + c that intersect at two points,
    the product of the x-coordinates of these intersection points is equal to (c - k) / a. -/
theorem intersection_x_product (a m b c k : ℝ) (ha : a ≠ 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := m * x + k
  let h (x : ℝ) := f x - g x
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ h x₁ = 0 ∧ h x₂ = 0 →
  x₁ * x₂ = (c - k) / a :=
sorry

end intersection_x_product_l266_26650


namespace inequality_proof_l266_26624

theorem inequality_proof (A B C a b c r : ℝ) 
  (hA : A > 0) (hB : B > 0) (hC : C > 0) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) : 
  (A + a + B + b) / (A + a + B + b + c + r) + 
  (B + b + C + c) / (B + b + C + c + a + r) > 
  (c + c + A + a) / (C + c + A + a + b + r) := by
  sorry

end inequality_proof_l266_26624


namespace equation_has_real_roots_l266_26693

theorem equation_has_real_roots (a b : ℝ) : 
  ∃ x : ℝ, (x^2 / (x^2 - a^2) + x^2 / (x^2 - b^2) = 4) := by sorry

end equation_has_real_roots_l266_26693


namespace frank_reading_speed_l266_26603

/-- The number of days Frank took to finish all books -/
def total_days : ℕ := 492

/-- The total number of books Frank read -/
def total_books : ℕ := 41

/-- The number of days it took Frank to finish each book -/
def days_per_book : ℚ := total_days / total_books

/-- Theorem stating that Frank took 12 days to finish each book -/
theorem frank_reading_speed : days_per_book = 12 := by
  sorry

end frank_reading_speed_l266_26603


namespace complex_modulus_product_l266_26611

theorem complex_modulus_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end complex_modulus_product_l266_26611


namespace complement_A_intersect_B_is_correct_l266_26630

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | abs x ≤ 2}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

-- Define the complement of A ∩ B in ℝ
def complement_A_intersect_B : Set ℝ := {x : ℝ | x < -2 ∨ x > 0}

-- State the theorem
theorem complement_A_intersect_B_is_correct :
  (Set.univ : Set ℝ) \ (A ∩ B) = complement_A_intersect_B := by sorry

end complement_A_intersect_B_is_correct_l266_26630


namespace athletes_arrangement_count_l266_26679

/-- Represents the number of athletes in each team --/
def team_sizes : List Nat := [3, 3, 2, 4]

/-- The total number of athletes --/
def total_athletes : Nat := team_sizes.sum

/-- Calculates the number of ways to arrange the athletes --/
def arrangement_count : Nat :=
  (Nat.factorial team_sizes.length) * (team_sizes.map Nat.factorial).prod

theorem athletes_arrangement_count :
  total_athletes = 12 →
  team_sizes = [3, 3, 2, 4] →
  arrangement_count = 41472 := by
  sorry

end athletes_arrangement_count_l266_26679


namespace polygon_sides_l266_26696

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  sorry

end polygon_sides_l266_26696


namespace floor_sum_equals_n_l266_26671

theorem floor_sum_equals_n (N : ℕ+) :
  N = ∑' n : ℕ, ⌊(N : ℝ) / (2 ^ n : ℝ)⌋ := by sorry

end floor_sum_equals_n_l266_26671


namespace twice_gcf_equals_180_l266_26605

def a : ℕ := 180
def b : ℕ := 270
def c : ℕ := 450

theorem twice_gcf_equals_180 : 2 * Nat.gcd a (Nat.gcd b c) = 180 := by
  sorry

end twice_gcf_equals_180_l266_26605


namespace runner_speeds_l266_26668

/-- The speed of runner A in meters per second -/
def speed_A : ℝ := 9

/-- The speed of runner B in meters per second -/
def speed_B : ℝ := 7

/-- The length of the circular track in meters -/
def track_length : ℝ := 400

/-- The time in seconds it takes for A and B to meet when running in opposite directions -/
def opposite_meeting_time : ℝ := 25

/-- The time in seconds it takes for A to catch up with B when running in the same direction -/
def same_direction_catchup_time : ℝ := 200

theorem runner_speeds :
  speed_A * opposite_meeting_time + speed_B * opposite_meeting_time = track_length ∧
  speed_A * same_direction_catchup_time - speed_B * same_direction_catchup_time = track_length :=
by sorry

end runner_speeds_l266_26668


namespace cube_volume_from_total_edge_length_l266_26652

/-- Given a cube where the sum of the lengths of all edges is 48 cm, 
    prove that its volume is 64 cubic centimeters. -/
theorem cube_volume_from_total_edge_length : 
  ∀ (edge_length : ℝ), 
    (12 * edge_length = 48) →
    (edge_length^3 = 64) := by
  sorry

end cube_volume_from_total_edge_length_l266_26652


namespace phi_tau_ge_n_l266_26638

/-- The number of divisors of a positive integer n -/
def tau (n : ℕ+) : ℕ := sorry

/-- Euler's totient function for a positive integer n -/
def phi (n : ℕ+) : ℕ := sorry

/-- For any positive integer n, the product of φ(n) and τ(n) is greater than or equal to n -/
theorem phi_tau_ge_n (n : ℕ+) : phi n * tau n ≥ n := by sorry

end phi_tau_ge_n_l266_26638


namespace total_drawings_l266_26662

/-- The number of neighbors Shiela has -/
def num_neighbors : ℕ := 6

/-- The number of drawings each neighbor receives -/
def drawings_per_neighbor : ℕ := 9

/-- Theorem: The total number of drawings Shiela made is 54 -/
theorem total_drawings : num_neighbors * drawings_per_neighbor = 54 := by
  sorry

end total_drawings_l266_26662


namespace benny_stored_bales_l266_26615

/-- The number of bales Benny stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Proof that Benny stored 35 bales in the barn -/
theorem benny_stored_bales : 
  let initial_bales : ℕ := 47
  let final_bales : ℕ := 82
  bales_stored initial_bales final_bales = 35 := by
  sorry

end benny_stored_bales_l266_26615


namespace megan_initial_albums_l266_26607

/-- The number of albums Megan initially put in her shopping cart -/
def initial_albums : ℕ := sorry

/-- The number of albums Megan removed from her cart -/
def removed_albums : ℕ := 2

/-- The number of songs in each album -/
def songs_per_album : ℕ := 7

/-- The total number of songs Megan bought -/
def total_songs : ℕ := 42

/-- Theorem stating that Megan initially put 8 albums in her shopping cart -/
theorem megan_initial_albums :
  initial_albums = 8 :=
by sorry

end megan_initial_albums_l266_26607


namespace translation_theorem_l266_26602

/-- Represents a point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def apply_translation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem :
  let A : Point := { x := -1, y := 0 }
  let B : Point := { x := 1, y := 2 }
  let A1 : Point := { x := 2, y := -1 }
  let t : Translation := { dx := A1.x - A.x, dy := A1.y - A.y }
  let B1 : Point := apply_translation B t
  B1 = { x := 4, y := 1 } := by
  sorry

end translation_theorem_l266_26602


namespace john_share_l266_26645

def total_amount : ℕ := 6000
def john_ratio : ℕ := 2
def jose_ratio : ℕ := 4
def binoy_ratio : ℕ := 6

theorem john_share :
  let total_ratio := john_ratio + jose_ratio + binoy_ratio
  (john_ratio : ℚ) / total_ratio * total_amount = 1000 := by sorry

end john_share_l266_26645


namespace two_translations_result_l266_26675

def complex_translation (z w : ℂ) : ℂ → ℂ := fun x ↦ x + w - z

theorem two_translations_result (t₁ t₂ : ℂ → ℂ) :
  t₁ (-3 + 2*I) = -7 - I →
  t₂ (-7 - I) = -10 →
  t₁ = complex_translation (-3 + 2*I) (-7 - I) →
  t₂ = complex_translation (-7 - I) (-10) →
  (t₂ ∘ t₁) (-4 + 5*I) = -11 + 3*I := by
  sorry

end two_translations_result_l266_26675


namespace arithmetic_geometric_sequence_l266_26641

theorem arithmetic_geometric_sequence (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (2 * b = a + c) →  -- arithmetic sequence condition
  (b ^ 2 = c * (a + 1)) →  -- geometric sequence condition when a is increased by 1
  (b ^ 2 = a * (c + 2)) →  -- geometric sequence condition when c is increased by 2
  b = 12 := by
sorry

end arithmetic_geometric_sequence_l266_26641


namespace ratio_x_y_is_two_l266_26612

theorem ratio_x_y_is_two (x y a : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (eq1 : x^3 + Real.log x + 2*a^2 = 0) 
  (eq2 : 4*y^3 + Real.log (Real.sqrt y) + Real.log (Real.sqrt 2) + a^2 = 0) : 
  x / y = 2 := by
sorry

end ratio_x_y_is_two_l266_26612


namespace arithmetic_sequence_third_term_l266_26639

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 + a 5 = 16) :
  a 3 = 8 := by
sorry

end arithmetic_sequence_third_term_l266_26639


namespace solution_satisfies_equations_l266_26625

theorem solution_satisfies_equations :
  let x : ℚ := -5/7
  let y : ℚ := -18/7
  (6 * x + 3 * y = -12) ∧ (4 * x = 5 * y + 10) := by
  sorry

end solution_satisfies_equations_l266_26625


namespace marias_reading_capacity_l266_26621

/-- Given Maria's reading speed and available time, prove how many complete books she can read --/
theorem marias_reading_capacity (pages_per_hour : ℕ) (book_pages : ℕ) (available_hours : ℕ) : 
  pages_per_hour = 120 → book_pages = 360 → available_hours = 8 → 
  (available_hours * pages_per_hour) / book_pages = 2 := by
  sorry

#check marias_reading_capacity

end marias_reading_capacity_l266_26621


namespace unique_prime_with_prime_sums_l266_26695

theorem unique_prime_with_prime_sums : ∃! p : ℕ, 
  Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) :=
sorry

end unique_prime_with_prime_sums_l266_26695


namespace parabola_properties_l266_26667

def is_valid_parabola (a b c : ℝ) : Prop :=
  a ≠ 0 ∧
  a * (-1)^2 + b * (-1) + c = -1 ∧
  c = 1 ∧
  a * (-2)^2 + b * (-2) + c > 1

theorem parabola_properties (a b c : ℝ) 
  (h : is_valid_parabola a b c) : 
  a * b * c > 0 ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c - 3 = 0 ∧ a * x₂^2 + b * x₂ + c - 3 = 0) ∧
  a + b + c > 7 := by
  sorry

end parabola_properties_l266_26667


namespace distance_origin_to_line_l266_26653

/-- The distance from the origin to the line x + √3y - 2 = 0 is 1 -/
theorem distance_origin_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + Real.sqrt 3 * y - 2 = 0}
  ∃ d : ℝ, d = 1 ∧ ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≥ d :=
by sorry

end distance_origin_to_line_l266_26653


namespace not_perfect_square_123_ones_l266_26609

def number_with_ones (n : ℕ) : ℕ :=
  (10^n - 1) * 10^n + 123

theorem not_perfect_square_123_ones :
  ∀ n : ℕ, ∃ k : ℕ, (number_with_ones n) ≠ k^2 := by
  sorry

end not_perfect_square_123_ones_l266_26609


namespace factor_expression_l266_26656

theorem factor_expression (a b c : ℝ) :
  ((a^2 + b^2)^3 + (b^2 + c^2)^3 + (c^2 + a^2)^3) / ((a + b)^3 + (b + c)^3 + (c + a)^3)
  = (a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2) / ((a + b) * (b + c) * (c + a)) :=
by sorry

end factor_expression_l266_26656


namespace boat_downstream_distance_l266_26619

/-- Proves that given a boat with a speed of 20 km/hr in still water and a stream with
    speed of 6 km/hr, if the boat travels the same time downstream as it does to
    travel 14 km upstream, then the distance traveled downstream is 26 km. -/
theorem boat_downstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (upstream_distance : ℝ)
  (h1 : boat_speed = 20)
  (h2 : stream_speed = 6)
  (h3 : upstream_distance = 14)
  (h4 : (upstream_distance / (boat_speed - stream_speed)) =
        (downstream_distance / (boat_speed + stream_speed))) :
  downstream_distance = 26 :=
by
  sorry


end boat_downstream_distance_l266_26619


namespace unique_solution_quadratic_l266_26660

theorem unique_solution_quadratic (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) ↔ (a = 0 ∨ a = 9/8) := by
  sorry

end unique_solution_quadratic_l266_26660


namespace series_sum_equals_three_fourths_l266_26674

theorem series_sum_equals_three_fourths : 
  ∑' k, (k : ℝ) / (3 : ℝ) ^ k = 3 / 4 := by sorry

end series_sum_equals_three_fourths_l266_26674


namespace great_pyramid_dimensions_l266_26699

/-- The Great Pyramid of Giza's dimensions and sum of height and width -/
theorem great_pyramid_dimensions :
  let height := 500 + 20
  let width := height + 234
  height + width = 1274 := by sorry

end great_pyramid_dimensions_l266_26699


namespace e_value_is_negative_72_l266_26636

/-- A cubic polynomial with specific properties -/
structure SpecialCubicPolynomial where
  d : ℝ
  e : ℝ
  mean_zeros : ℝ
  product_zeros : ℝ
  sum_coefficients : ℝ
  h1 : mean_zeros = 2 * product_zeros
  h2 : mean_zeros = sum_coefficients
  h3 : sum_coefficients = 3 + d + e + 9

/-- The value of e in the special cubic polynomial -/
def find_e (p : SpecialCubicPolynomial) : ℝ := -72

/-- Theorem stating that the value of e is -72 for the given polynomial -/
theorem e_value_is_negative_72 (p : SpecialCubicPolynomial) : find_e p = -72 := by
  sorry

end e_value_is_negative_72_l266_26636


namespace rational_inequality_l266_26628

theorem rational_inequality (a b : ℚ) (h1 : a + b > 0) (h2 : a * b < 0) :
  a > 0 ∧ b < 0 ∧ |a| > |b| := by
  sorry

end rational_inequality_l266_26628


namespace beth_wins_743_l266_26682

/-- Represents a configuration of brick walls -/
def Configuration := List Nat

/-- Calculates the nim-value of a single wall -/
noncomputable def nimValue (n : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum of a list of nim-values -/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a configuration is a winning position for the current player -/
def isWinningPosition (config : Configuration) : Prop :=
  nimSum (config.map nimValue) ≠ 0

/-- The game of brick removal -/
theorem beth_wins_743 (config : Configuration) :
  config = [7, 4, 4] → ¬isWinningPosition config :=
  sorry

end beth_wins_743_l266_26682


namespace count_satisfying_numbers_is_45_l266_26698

/-- A function that checks if a three-digit number satisfies the condition -/
def satisfiesCondition (n : Nat) : Bool :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  b = a + c ∧ 100 ≤ n ∧ n ≤ 999

/-- The count of three-digit numbers satisfying the condition -/
def countSatisfyingNumbers : Nat :=
  (List.range 900).map (· + 100)
    |>.filter satisfiesCondition
    |>.length

/-- Theorem stating that the count of satisfying numbers is 45 -/
theorem count_satisfying_numbers_is_45 : countSatisfyingNumbers = 45 := by
  sorry

end count_satisfying_numbers_is_45_l266_26698


namespace max_reciprocal_sum_l266_26610

theorem max_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (∀ m : ℝ, (1 / a + 1 / b ≥ m) → m ≤ 4) ∧ 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 1 ∧ 1 / a + 1 / b = 4) :=
sorry

end max_reciprocal_sum_l266_26610


namespace divisibility_property_l266_26683

theorem divisibility_property (n : ℕ) : 
  ∃ (a b : ℕ), (a * n + 1)^6 + b ≡ 0 [MOD (n^2 + n + 1)] :=
by
  use 2, 27
  sorry

end divisibility_property_l266_26683


namespace longest_segment_in_quarter_circle_l266_26633

theorem longest_segment_in_quarter_circle (r : ℝ) (h : r = 9) :
  let sector_chord_length_squared := 2 * r^2
  sector_chord_length_squared = 162 := by sorry

end longest_segment_in_quarter_circle_l266_26633


namespace pencil_packing_problem_l266_26649

theorem pencil_packing_problem :
  ∃ a : ℕ, 200 ≤ a ∧ a ≤ 300 ∧ 
    a % 10 = 7 ∧ 
    a % 12 = 9 ∧
    (a = 237 ∨ a = 297) := by
  sorry

end pencil_packing_problem_l266_26649


namespace equation_solution_l266_26658

theorem equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 6*x * Real.sqrt (x + 5) = 24 ∧ 
  x = (-17 + Real.sqrt 277) / 2 := by
sorry

end equation_solution_l266_26658


namespace theater_ticket_price_l266_26665

/-- The price of tickets at a theater with discounts for children and seniors -/
theorem theater_ticket_price :
  ∀ (adult_price : ℝ),
  (6 * adult_price + 5 * (adult_price / 2) + 3 * (adult_price * 0.75) = 42) →
  (10 * adult_price + 8 * (adult_price / 2) + 4 * (adult_price * 0.75) = 58.65) :=
by
  sorry

end theater_ticket_price_l266_26665


namespace light_reflection_l266_26635

/-- Given a ray of light reflecting off a line, this theorem proves the equations of the incident and reflected rays. -/
theorem light_reflection (A B : ℝ × ℝ) (reflecting_line : ℝ → ℝ → Prop) :
  A = (2, 3) →
  B = (1, 1) →
  (∀ x y, reflecting_line x y ↔ x + y + 1 = 0) →
  ∃ (incident_ray reflected_ray : ℝ → ℝ → Prop),
    (∀ x y, incident_ray x y ↔ 5*x - 4*y + 2 = 0) ∧
    (∀ x y, reflected_ray x y ↔ 4*x - 5*y + 1 = 0) ∧
    (∃ C : ℝ × ℝ, incident_ray C.1 C.2 ∧ reflecting_line C.1 C.2 ∧ reflected_ray C.1 C.2) :=
by sorry

end light_reflection_l266_26635


namespace arithmetic_sequence_property_l266_26689

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S seq n + seq.a (n + 1)

theorem arithmetic_sequence_property (seq : ArithmeticSequence) (m : ℕ) 
  (h1 : S seq (m - 1) = -2)
  (h2 : S seq m = 0)
  (h3 : S seq (m + 1) = 3) :
  seq.d = 1 ∧ m = 5 := by
  sorry

end arithmetic_sequence_property_l266_26689


namespace complex_equality_implies_ratio_l266_26643

theorem complex_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4 → b / a = 1 := by
  sorry

end complex_equality_implies_ratio_l266_26643


namespace even_number_in_rows_l266_26640

/-- Definition of the triangle table -/
def triangle_table : ℕ → ℤ → ℕ
| 1, 0 => 1
| n, k => if n > 1 ∧ abs k < n then
            triangle_table (n-1) (k-1) + triangle_table (n-1) k + triangle_table (n-1) (k+1)
          else 0

/-- Theorem: From the third row onward, each row contains at least one even number -/
theorem even_number_in_rows (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℤ, Even (triangle_table n k) := by sorry

end even_number_in_rows_l266_26640


namespace addition_problems_l266_26634

theorem addition_problems :
  (15 + (-22) = -7) ∧
  ((-13) + (-8) = -21) ∧
  ((-0.9) + 1.5 = 0.6) ∧
  (1/2 + (-2/3) = -1/6) := by
  sorry

end addition_problems_l266_26634


namespace jelly_ratio_l266_26644

def jelly_problem (grape strawberry raspberry plum : ℕ) : Prop :=
  grape = 2 * strawberry ∧
  raspberry = 2 * plum ∧
  plum = 6 ∧
  strawberry = 18 ∧
  raspberry * 3 = grape

theorem jelly_ratio :
  ∀ grape strawberry raspberry plum : ℕ,
  jelly_problem grape strawberry raspberry plum →
  raspberry * 3 = grape :=
by
  sorry

end jelly_ratio_l266_26644


namespace det_B_is_one_l266_26661

theorem det_B_is_one (b e : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![b, 2; -3, e]
  B + B⁻¹ = 1 → Matrix.det B = 1 := by
sorry

end det_B_is_one_l266_26661


namespace system_solutions_l266_26601

-- Define the system of equations
def system (t y a : ℝ) : Prop :=
  (|t| - y = 1 - a^4 - a^4 * t^4) ∧ (t^2 + y^2 = 1)

-- Define the property of having multiple solutions
def has_multiple_solutions (a : ℝ) : Prop :=
  ∃ (t₁ y₁ t₂ y₂ : ℝ), t₁ ≠ t₂ ∧ system t₁ y₁ a ∧ system t₂ y₂ a

-- Define the property of having a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∀ (t y : ℝ), system t y a → t = 0 ∧ y = 1

-- Theorem statement
theorem system_solutions :
  (has_multiple_solutions 0) ∧
  (has_unique_solution (Real.sqrt (Real.sqrt 2))) :=
sorry

end system_solutions_l266_26601


namespace smallest_among_three_l266_26676

theorem smallest_among_three : ∀ (a b c : ℕ), a = 5 ∧ b = 8 ∧ c = 4 → c ≤ a ∧ c ≤ b := by
  sorry

end smallest_among_three_l266_26676


namespace gcf_lcm_sum_8_12_l266_26654

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcf_lcm_sum_8_12_l266_26654


namespace austin_friday_hours_l266_26637

/-- Represents the problem of Austin saving for a bicycle --/
def bicycle_savings (hourly_rate : ℚ) (monday_hours : ℚ) (wednesday_hours : ℚ) (total_weeks : ℕ) (bicycle_cost : ℚ) : Prop :=
  let monday_earnings := hourly_rate * monday_hours
  let wednesday_earnings := hourly_rate * wednesday_hours
  let weekly_earnings := monday_earnings + wednesday_earnings
  let total_earnings_without_friday := weekly_earnings * total_weeks
  let remaining_earnings_needed := bicycle_cost - total_earnings_without_friday
  let friday_hours := remaining_earnings_needed / (hourly_rate * total_weeks)
  friday_hours = 3

/-- Theorem stating that Austin needs to work 3 hours on Fridays --/
theorem austin_friday_hours : 
  bicycle_savings 5 2 1 6 180 := by sorry

end austin_friday_hours_l266_26637


namespace fermat_prime_equation_solutions_l266_26669

/-- A Fermat's Prime is a prime number of the form 2^α + 1, for α a positive integer -/
def IsFermatPrime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ α : ℕ, α > 0 ∧ p = 2^α + 1

/-- The main theorem statement -/
theorem fermat_prime_equation_solutions :
  ∀ p n k : ℕ,
  p > 0 ∧ n > 0 ∧ k > 0 →
  IsFermatPrime p →
  p^n + n = (n+1)^k →
  (p = 3 ∧ n = 1 ∧ k = 2) ∨ (p = 5 ∧ n = 2 ∧ k = 3) :=
by sorry

end fermat_prime_equation_solutions_l266_26669


namespace room_width_calculation_l266_26627

/-- Given a rectangular room with known length, paving cost per square meter, and total paving cost,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : cost_per_sqm = 700)
    (h3 : total_cost = 14437.5) :
    total_cost / cost_per_sqm / length = 3.75 := by
  sorry

end room_width_calculation_l266_26627


namespace range_of_a_l266_26678

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a) ↔ -1 ≤ a ∧ a ≤ 4 := by
sorry

end range_of_a_l266_26678


namespace tennis_tournament_matches_l266_26626

theorem tennis_tournament_matches (total_players : ℕ) (bye_players : ℕ) (first_round_players : ℕ) :
  total_players = 128 →
  bye_players = 40 →
  first_round_players = 88 →
  (total_players = bye_players + first_round_players) →
  (∃ (total_matches : ℕ), total_matches = 127 ∧
    total_matches = (first_round_players / 2) + (total_players - 1)) := by
  sorry

end tennis_tournament_matches_l266_26626


namespace expression_simplification_l266_26681

theorem expression_simplification (m : ℝ) (h : m = 10) :
  (1 - m / (m + 2)) / ((m^2 - 4*m + 4) / (m^2 - 4)) = 1/4 := by
  sorry

end expression_simplification_l266_26681


namespace triangle_centroid_product_l266_26688

theorem triangle_centroid_product (AP PD BP PE CP PF : ℝ) 
  (h : AP / PD + BP / PE + CP / PF = 90) : 
  AP / PD * BP / PE * CP / PF = 94 := by
  sorry

end triangle_centroid_product_l266_26688


namespace intersection_of_P_and_Q_l266_26672

def P : Set ℝ := {-2, 0, 2, 4}
def Q : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_P_and_Q : P ∩ Q = {2} := by
  sorry

end intersection_of_P_and_Q_l266_26672


namespace minsu_running_time_l266_26651

theorem minsu_running_time 
  (total_distance : Real) 
  (speed : Real) 
  (distance_remaining : Real) : Real :=
  let distance_run := total_distance - distance_remaining
  let time_elapsed := distance_run / speed
  have h1 : total_distance = 120 := by sorry
  have h2 : speed = 4 := by sorry
  have h3 : distance_remaining = 20 := by sorry
  have h4 : time_elapsed = 25 := by sorry
  time_elapsed

#check minsu_running_time

end minsu_running_time_l266_26651


namespace earrings_sold_count_l266_26600

def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earrings_price : ℕ := 10
def ensemble_price : ℕ := 45

def necklaces_sold : ℕ := 5
def bracelets_sold : ℕ := 10
def ensembles_sold : ℕ := 2

def total_sales : ℕ := 565

theorem earrings_sold_count :
  ∃ (x : ℕ), 
    necklace_price * necklaces_sold + 
    bracelet_price * bracelets_sold + 
    earrings_price * x + 
    ensemble_price * ensembles_sold = total_sales ∧
    x = 20 := by sorry

end earrings_sold_count_l266_26600


namespace calvin_insect_collection_l266_26623

/-- Calculates the total number of insects in Calvin's collection --/
def total_insects (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := 2 * scorpions
  let beetles := 4 * crickets
  let other_insects := roaches + scorpions + crickets + caterpillars + beetles
  let exotic_insects := 3 * other_insects
  other_insects + exotic_insects

/-- Theorem stating that Calvin has 204 insects in his collection --/
theorem calvin_insect_collection : total_insects 12 3 = 204 := by
  sorry

end calvin_insect_collection_l266_26623


namespace cube_edge_length_specific_l266_26616

/-- The edge length of a cube with the same volume as a rectangular block -/
def cube_edge_length (l w h : ℝ) : ℝ :=
  (l * w * h) ^ (1/3)

/-- Theorem: The edge length of a cube with the same volume as a 50cm × 8cm × 20cm rectangular block is 20cm -/
theorem cube_edge_length_specific : cube_edge_length 50 8 20 = 20 := by
  sorry

end cube_edge_length_specific_l266_26616


namespace parabola_shift_l266_26673

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * h + p.b, c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- The main theorem stating that shifting y = 3x^2 right by 1 and down by 2 results in y = 3(x-1)^2 - 2 -/
theorem parabola_shift :
  let p := Parabola.mk 3 0 0
  let p_shifted := shift_vertical (shift_horizontal p 1) (-2)
  p_shifted = Parabola.mk 3 (-6) 1 := by sorry

end parabola_shift_l266_26673


namespace salt_mixture_problem_l266_26659

/-- Proves that the amount of initial 20% salt solution is 30 ounces when mixed with 30 ounces of 60% salt solution to create a 40% salt solution. -/
theorem salt_mixture_problem (x : ℝ) :
  (0.20 * x + 0.60 * 30 = 0.40 * (x + 30)) → x = 30 := by
  sorry

end salt_mixture_problem_l266_26659


namespace largest_non_representable_number_l266_26642

theorem largest_non_representable_number : ∃ (n : ℕ), n > 0 ∧
  (∀ x y : ℕ, x > 0 → y > 0 → 9 * x + 11 * y ≠ n) ∧
  (∀ m : ℕ, m > n → ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 9 * x + 11 * y = m) ∧
  (∀ k : ℕ, k > n → ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 9 * x + 11 * y = k) →
  n = 99 := by
sorry

end largest_non_representable_number_l266_26642


namespace artist_paintings_l266_26663

/-- Calculates the number of paintings an artist can complete in a given number of weeks. -/
def paintings_completed (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) : ℕ :=
  (hours_per_week / hours_per_painting) * num_weeks

/-- Proves that an artist spending 30 hours per week painting, taking 3 hours per painting,
    can complete 40 paintings in 4 weeks. -/
theorem artist_paintings : paintings_completed 30 3 4 = 40 := by
  sorry

end artist_paintings_l266_26663


namespace chloe_profit_l266_26657

/-- Calculates Chloe's profit from selling chocolate-dipped strawberries during a 3-day Mother's Day celebration. -/
theorem chloe_profit (buy_price : ℝ) (sell_price : ℝ) (bulk_discount : ℝ) 
  (min_production_cost : ℝ) (max_production_cost : ℝ) 
  (day1_price_factor : ℝ) (day2_price_factor : ℝ) (day3_price_factor : ℝ)
  (total_dozens : ℕ) (day1_dozens : ℕ) (day2_dozens : ℕ) (day3_dozens : ℕ) :
  buy_price = 50 →
  sell_price = 60 →
  bulk_discount = 0.1 →
  min_production_cost = 40 →
  max_production_cost = 45 →
  day1_price_factor = 1 →
  day2_price_factor = 1.2 →
  day3_price_factor = 0.85 →
  total_dozens = 50 →
  day1_dozens = 12 →
  day2_dozens = 18 →
  day3_dozens = 20 →
  total_dozens ≥ 10 →
  day1_dozens + day2_dozens + day3_dozens = total_dozens →
  ∃ profit : ℝ, profit = 152 ∧ 
    profit = (day1_dozens * sell_price * day1_price_factor +
              day2_dozens * sell_price * day2_price_factor +
              day3_dozens * sell_price * day3_price_factor) * (1 - bulk_discount) -
             total_dozens * (min_production_cost + max_production_cost) / 2 :=
by sorry

end chloe_profit_l266_26657


namespace solution_set_x_squared_leq_four_l266_26664

theorem solution_set_x_squared_leq_four :
  {x : ℝ | x^2 ≤ 4} = {x : ℝ | -2 ≤ x ∧ x ≤ 2} := by
  sorry

end solution_set_x_squared_leq_four_l266_26664


namespace complex_fraction_division_l266_26697

theorem complex_fraction_division : 
  (5 / (8 / 13)) / (10 / 7) = 91 / 16 := by sorry

end complex_fraction_division_l266_26697


namespace initial_marbles_count_initial_marbles_proof_l266_26677

def marbles_to_juan : ℕ := 1835
def marbles_to_lisa : ℕ := 985
def marbles_left : ℕ := 5930

theorem initial_marbles_count : ℕ :=
  marbles_to_juan + marbles_to_lisa + marbles_left

#check initial_marbles_count

theorem initial_marbles_proof : initial_marbles_count = 8750 := by
  sorry

end initial_marbles_count_initial_marbles_proof_l266_26677


namespace round_trip_average_speed_l266_26686

/-- Calculates the average speed for a round trip given uphill speed, uphill time, and downhill time -/
theorem round_trip_average_speed 
  (uphill_speed : ℝ) 
  (uphill_time : ℝ) 
  (downhill_time : ℝ) 
  (h1 : uphill_speed = 2.5) 
  (h2 : uphill_time = 3) 
  (h3 : downhill_time = 2) : 
  (2 * uphill_speed * uphill_time) / (uphill_time + downhill_time) = 3 := by
  sorry

#check round_trip_average_speed

end round_trip_average_speed_l266_26686


namespace sin_minus_cos_for_specific_tan_l266_26680

theorem sin_minus_cos_for_specific_tan (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end sin_minus_cos_for_specific_tan_l266_26680


namespace dormitory_arrangements_l266_26629

def num_students : ℕ := 7
def min_per_dorm : ℕ := 2

-- Function to calculate the number of arrangements
def calculate_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  sorry

theorem dormitory_arrangements :
  calculate_arrangements num_students min_per_dorm 2 = 60 :=
sorry

end dormitory_arrangements_l266_26629


namespace no_constant_term_implies_n_not_eight_l266_26685

theorem no_constant_term_implies_n_not_eight (n : ℕ) :
  (∀ r : ℕ, r ≤ n → n ≠ 4 / 3 * r) →
  n ≠ 8 := by
  sorry

end no_constant_term_implies_n_not_eight_l266_26685
