import Mathlib

namespace product_of_solutions_abs_eq_three_abs_minus_two_l1755_175507

theorem product_of_solutions_abs_eq_three_abs_minus_two (x : ℝ) :
  (∃ x₁ x₂ : ℝ, (|x₁| = 3 * (|x₁| - 2) ∧ |x₂| = 3 * (|x₂| - 2) ∧ x₁ ≠ x₂) →
  x₁ * x₂ = -9) :=
sorry

end product_of_solutions_abs_eq_three_abs_minus_two_l1755_175507


namespace triangle_properties_l1755_175558

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin (2 * t.B) = Real.sqrt 3 * t.b * Real.sin t.A)
  (h2 : Real.cos t.A = 1/3) :
  t.B = π/6 ∧ Real.sin t.C = (2 * Real.sqrt 6 + 1) / 6 := by
  sorry

end

end triangle_properties_l1755_175558


namespace larger_triangle_perimeter_l1755_175599

/-- Two similar triangles where one has side lengths 12, 12, and 15, and the other has longest side 30 -/
structure SimilarTriangles where
  /-- Side lengths of the smaller triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- Longest side of the larger triangle -/
  longest_side : ℝ
  /-- The smaller triangle is isosceles -/
  h_isosceles : a = b
  /-- The side lengths of the smaller triangle -/
  h_sides : a = 12 ∧ c = 15
  /-- The longest side of the larger triangle -/
  h_longest : longest_side = 30

/-- The perimeter of the larger triangle is 78 -/
theorem larger_triangle_perimeter (t : SimilarTriangles) : 
  (t.longest_side / t.c) * (t.a + t.b + t.c) = 78 := by
  sorry

end larger_triangle_perimeter_l1755_175599


namespace ochos_friends_l1755_175575

theorem ochos_friends (total : ℕ) (boys girls : ℕ) (h1 : boys = girls) (h2 : boys + girls = total) (h3 : boys = 4) : total = 8 := by
  sorry

end ochos_friends_l1755_175575


namespace equation_solutions_l1755_175550

theorem equation_solutions :
  (∀ x, (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4) ∧
  (∀ x, x * (x - 6) = 6 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15) := by
  sorry

end equation_solutions_l1755_175550


namespace lcm_of_36_and_220_l1755_175534

theorem lcm_of_36_and_220 : Nat.lcm 36 220 = 1980 := by
  sorry

end lcm_of_36_and_220_l1755_175534


namespace range_of_f_l1755_175598

/-- The diamond operation -/
def diamond (x y : ℝ) : ℝ := (x + y)^2 - x * y

/-- The function f -/
def f (a x : ℝ) : ℝ := diamond a x

theorem range_of_f (a : ℝ) (h : diamond 1 a = 3) :
  ∀ y : ℝ, y > 1 → ∃ x : ℝ, x > 0 ∧ f a x = y ∧
  ∀ z : ℝ, z > 0 → f a z ≥ 1 :=
sorry

end range_of_f_l1755_175598


namespace least_sum_exponents_for_896_l1755_175501

theorem least_sum_exponents_for_896 :
  ∃ (a b c : ℕ), 
    (a < b ∧ b < c) ∧ 
    (2^a + 2^b + 2^c = 896) ∧
    (∀ (x y z : ℕ), x < y ∧ y < z ∧ 2^x + 2^y + 2^z = 896 → a + b + c ≤ x + y + z) ∧
    (a + b + c = 24) := by
  sorry

end least_sum_exponents_for_896_l1755_175501


namespace housing_price_growth_l1755_175554

/-- Proves that the equation relating initial housing price, final housing price, 
    and annual growth rate over two years is correct. -/
theorem housing_price_growth (initial_price final_price : ℝ) (x : ℝ) 
  (h_initial : initial_price = 5500)
  (h_final : final_price = 7000) :
  initial_price * (1 + x)^2 = final_price := by
  sorry

end housing_price_growth_l1755_175554


namespace justin_pencils_l1755_175592

theorem justin_pencils (total_pencils sabrina_pencils : ℕ) : 
  total_pencils = 50 →
  sabrina_pencils = 14 →
  total_pencils - sabrina_pencils > 2 * sabrina_pencils →
  (total_pencils - sabrina_pencils) - 2 * sabrina_pencils = 8 := by
  sorry

end justin_pencils_l1755_175592


namespace regular_octagon_angles_l1755_175511

theorem regular_octagon_angles :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
    n = 8 →
    interior_angle = (180 * (n - 2 : ℝ)) / n →
    exterior_angle = 180 - interior_angle →
    interior_angle = 135 ∧ exterior_angle = 45 := by
  sorry

end regular_octagon_angles_l1755_175511


namespace root_sum_reciprocal_products_l1755_175586

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  (p^4 + 6*p^3 - 4*p^2 + 7*p + 3 = 0) →
  (q^4 + 6*q^3 - 4*q^2 + 7*q + 3 = 0) →
  (r^4 + 6*r^3 - 4*r^2 + 7*r + 3 = 0) →
  (s^4 + 6*s^3 - 4*s^2 + 7*s + 3 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = -4/3 :=
by sorry

end root_sum_reciprocal_products_l1755_175586


namespace commission_proof_l1755_175566

/-- Calculates the commission earned from selling a coupe and an SUV --/
def calculate_commission (coupe_price : ℝ) (suv_multiplier : ℝ) (commission_rate : ℝ) : ℝ :=
  let suv_price := coupe_price * suv_multiplier
  let total_sales := coupe_price + suv_price
  total_sales * commission_rate

/-- Proves that the commission earned is $1,800 given the specified conditions --/
theorem commission_proof :
  calculate_commission 30000 2 0.02 = 1800 := by
  sorry

end commission_proof_l1755_175566


namespace sum_squares_equals_two_l1755_175574

theorem sum_squares_equals_two (x y z : ℝ) 
  (sum_eq : x + y = 2) 
  (product_eq : x * y = z^2 + 1) : 
  x^2 + y^2 + z^2 = 2 := by
sorry

end sum_squares_equals_two_l1755_175574


namespace representative_count_l1755_175594

/-- The number of ways to choose a math class representative from a class with a given number of boys and girls. -/
def choose_representative (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  num_boys + num_girls

/-- Theorem: The number of ways to choose a math class representative from a class with 26 boys and 24 girls is 50. -/
theorem representative_count : choose_representative 26 24 = 50 := by
  sorry

end representative_count_l1755_175594


namespace log_power_sum_l1755_175530

theorem log_power_sum (a b : ℝ) (h1 : a = Real.log 8) (h2 : b = Real.log 27) :
  (5 : ℝ) ^ (a / b) + 2 ^ (b / a) = 8 := by
  sorry

end log_power_sum_l1755_175530


namespace det_dilation_matrix_det_dilation_matrix_7_l1755_175533

def dilation_matrix (scale_factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![scale_factor, 0],
    ![0, scale_factor]]

theorem det_dilation_matrix (scale_factor : ℝ) :
  Matrix.det (dilation_matrix scale_factor) = scale_factor ^ 2 := by
  sorry

theorem det_dilation_matrix_7 :
  Matrix.det (dilation_matrix 7) = 49 := by
  sorry

end det_dilation_matrix_det_dilation_matrix_7_l1755_175533


namespace polygon_area_is_two_l1755_175543

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Calculates the area of a polygon given its vertices -/
noncomputable def polygonArea (vertices : List Point) : ℚ :=
  sorry

/-- The list of vertices of the polygon -/
def polygonVertices : List Point := [
  ⟨0, 0⟩, ⟨1, 0⟩, ⟨2, 1⟩, ⟨2, 0⟩, ⟨3, 0⟩, ⟨3, 1⟩,
  ⟨3, 2⟩, ⟨2, 2⟩, ⟨2, 3⟩, ⟨1, 2⟩, ⟨0, 2⟩, ⟨0, 1⟩
]

/-- The theorem stating that the area of the polygon is 2 square units -/
theorem polygon_area_is_two :
  polygonArea polygonVertices = 2 := by
  sorry

end polygon_area_is_two_l1755_175543


namespace unique_tangent_line_l1755_175505

/-- The function whose graph we are considering -/
def f (x : ℝ) : ℝ := x^4 - 4*x^3 - 26*x^2

/-- The line we are trying to prove is unique -/
def L (x : ℝ) : ℝ := -60*x - 225

/-- Predicate to check if a point (x, y) is on or above the line L -/
def onOrAboveLine (x y : ℝ) : Prop := y ≥ L x

/-- Predicate to check if a point (x, y) is on the graph of f -/
def onGraph (x y : ℝ) : Prop := y = f x

/-- The main theorem stating the uniqueness of the line L -/
theorem unique_tangent_line :
  (∀ x y, onGraph x y → onOrAboveLine x y) ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ onGraph x₁ (L x₁) ∧ onGraph x₂ (L x₂)) ∧
  (∀ a b, (∀ x y, onGraph x y → y ≥ a*x + b) ∧
          (∃ x₁ x₂, x₁ ≠ x₂ ∧ onGraph x₁ (a*x₁ + b) ∧ onGraph x₂ (a*x₂ + b))
          → a = -60 ∧ b = -225) :=
by sorry

end unique_tangent_line_l1755_175505


namespace edge_probability_in_cube_l1755_175526

/-- A regular cube -/
structure RegularCube where
  vertices : Nat
  edges_per_vertex : Nat

/-- The probability of selecting two vertices that form an edge in a regular cube -/
def edge_probability (cube : RegularCube) : ℚ :=
  (cube.vertices * cube.edges_per_vertex / 2) / (cube.vertices.choose 2)

/-- Theorem stating the probability of selecting two vertices that form an edge in a regular cube -/
theorem edge_probability_in_cube :
  ∃ (cube : RegularCube), cube.vertices = 8 ∧ cube.edges_per_vertex = 3 ∧ edge_probability cube = 3/7 :=
sorry

end edge_probability_in_cube_l1755_175526


namespace men_who_left_hostel_l1755_175581

/-- Proves that 50 men left the hostel given the initial and final conditions -/
theorem men_who_left_hostel 
  (initial_men : ℕ) 
  (initial_days : ℕ) 
  (final_days : ℕ) 
  (h1 : initial_men = 250)
  (h2 : initial_days = 28)
  (h3 : final_days = 35)
  (h4 : initial_men * initial_days = (initial_men - men_who_left) * final_days) :
  men_who_left = 50 := by
  sorry

#check men_who_left_hostel

end men_who_left_hostel_l1755_175581


namespace tan_seventeen_pi_fourths_l1755_175588

theorem tan_seventeen_pi_fourths : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end tan_seventeen_pi_fourths_l1755_175588


namespace sum_of_first_three_coefficients_l1755_175515

theorem sum_of_first_three_coefficients (b : ℝ) : 
  let expansion := (1 + 2/b)^7
  let first_term_coeff := 1
  let second_term_coeff := 7 * 2 / b
  let third_term_coeff := (7 * 6 / 2) * (2 / b)^2
  first_term_coeff + second_term_coeff + third_term_coeff = 211 := by
sorry

end sum_of_first_three_coefficients_l1755_175515


namespace books_taken_out_on_tuesday_l1755_175512

/-- Prove that the number of books taken out on Tuesday is 120, given the initial and final number of books in the library and the changes on Wednesday and Thursday. -/
theorem books_taken_out_on_tuesday (initial_books : ℕ) (final_books : ℕ) (returned_wednesday : ℕ) (withdrawn_thursday : ℕ) 
  (h_initial : initial_books = 250)
  (h_final : final_books = 150)
  (h_wednesday : returned_wednesday = 35)
  (h_thursday : withdrawn_thursday = 15) :
  initial_books - final_books + returned_wednesday - withdrawn_thursday = 120 := by
  sorry

end books_taken_out_on_tuesday_l1755_175512


namespace max_interior_angles_less_than_120_is_5_l1755_175514

/-- A convex polygon with 532 sides -/
structure ConvexPolygon532 where
  sides : ℕ
  convex : Bool
  sidesEq532 : sides = 532

/-- The maximum number of interior angles less than 120° in a ConvexPolygon532 -/
def maxInteriorAnglesLessThan120 (p : ConvexPolygon532) : ℕ :=
  5

/-- Theorem stating that the maximum number of interior angles less than 120° in a ConvexPolygon532 is 5 -/
theorem max_interior_angles_less_than_120_is_5 (p : ConvexPolygon532) :
  maxInteriorAnglesLessThan120 p = 5 := by
  sorry

end max_interior_angles_less_than_120_is_5_l1755_175514


namespace largest_share_proof_l1755_175544

def profit_distribution (ratio : List Nat) (total_profit : Nat) : List Nat :=
  let total_parts := ratio.sum
  let part_value := total_profit / total_parts
  ratio.map (· * part_value)

theorem largest_share_proof (ratio : List Nat) (total_profit : Nat) :
  ratio = [3, 3, 4, 5, 6] → total_profit = 42000 →
  (profit_distribution ratio total_profit).maximum? = some 12000 := by
  sorry

end largest_share_proof_l1755_175544


namespace probability_of_drawing_k_l1755_175506

/-- The probability of drawing a "K" from a standard deck of 54 playing cards -/
theorem probability_of_drawing_k (total_cards : ℕ) (k_cards : ℕ) : 
  total_cards = 54 → k_cards = 4 → (k_cards : ℚ) / total_cards = 2 / 27 := by
  sorry

end probability_of_drawing_k_l1755_175506


namespace book_words_per_page_l1755_175521

theorem book_words_per_page 
  (total_pages : ℕ)
  (words_per_page : ℕ)
  (max_words_per_page : ℕ)
  (total_words_mod : ℕ)
  (h1 : total_pages = 224)
  (h2 : words_per_page ≤ max_words_per_page)
  (h3 : max_words_per_page = 150)
  (h4 : (total_pages * words_per_page) % 253 = total_words_mod)
  (h5 : total_words_mod = 156) :
  words_per_page = 106 := by
sorry

end book_words_per_page_l1755_175521


namespace quadratic_inequality_solution_set_l1755_175508

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 + a*x + b < 0}) :
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/3) ∪ Set.Ioi (1/2) :=
sorry

end quadratic_inequality_solution_set_l1755_175508


namespace toy_store_shelves_l1755_175503

theorem toy_store_shelves (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : 
  initial_stock = 6 →
  new_shipment = 18 →
  bears_per_shelf = 6 →
  (initial_stock + new_shipment) / bears_per_shelf = 4 :=
by
  sorry

end toy_store_shelves_l1755_175503


namespace west_8m_is_negative_8m_l1755_175557

/-- Represents the direction of movement --/
inductive Direction
  | East
  | West

/-- Represents a movement with magnitude and direction --/
structure Movement where
  magnitude : ℝ
  direction : Direction

/-- Convention for representing movement as a signed real number --/
def movementValue (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.magnitude
  | Direction.West => -m.magnitude

/-- Theorem stating that moving west 8m is equivalent to -8m --/
theorem west_8m_is_negative_8m :
  let west8m : Movement := { magnitude := 8, direction := Direction.West }
  movementValue west8m = -8 := by
  sorry

end west_8m_is_negative_8m_l1755_175557


namespace area_of_T_is_34_l1755_175542

/-- The area of a "T" shape formed within a rectangle -/
def area_of_T (rectangle_width rectangle_height removed_width removed_height : ℕ) : ℕ :=
  rectangle_width * rectangle_height - removed_width * removed_height

/-- Theorem stating that the area of the "T" shape is 34 square units -/
theorem area_of_T_is_34 :
  area_of_T 10 4 6 1 = 34 := by
  sorry

end area_of_T_is_34_l1755_175542


namespace equation_represents_three_lines_l1755_175562

-- Define the equation
def equation (x y : ℝ) : Prop := x^2 * (x + 2*y - 3) = y^2 * (x + 2*y - 3)

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = (3 - x) / 2
def line2 (x y : ℝ) : Prop := y = x
def line3 (x y : ℝ) : Prop := y = -x

-- Theorem stating that the equation represents three distinct lines
-- that do not all intersect at a single point
theorem equation_represents_three_lines :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    (∀ x y, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y)) ∧
    (line1 p1.1 p1.2 ∧ line2 p2.1 p2.2 ∧ line3 p3.1 p3.2) ∧
    (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) :=
  sorry


end equation_represents_three_lines_l1755_175562


namespace solution_set_of_system_l1755_175518

theorem solution_set_of_system (x y : ℝ) :
  x - 2 * y = 1 →
  x^3 - 6 * x * y - 8 * y^3 = 1 →
  y = (x - 1) / 2 := by
sorry

end solution_set_of_system_l1755_175518


namespace symmetric_cubic_homogeneous_decomposition_non_negative_equivalence_l1755_175596

-- Define the symmetric polynomials g₁, g₂, g₃
def g₁ (x y z : ℝ) : ℝ := x * (x - y) * (x - z) + y * (y - x) * (y - z) + z * (z - x) * (z - y)
def g₂ (x y z : ℝ) : ℝ := (y + z) * (x - y) * (x - z) + (x + z) * (y - x) * (y - z) + (x + y) * (z - x) * (z - y)
def g₃ (x y z : ℝ) : ℝ := x * y * z

-- Define a ternary symmetric cubic homogeneous polynomial
def SymmetricCubicHomogeneous (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f x y z = f y z x ∧ f x y z = f y x z ∧ ∀ t : ℝ, f (t*x) (t*y) (t*z) = t^3 * f x y z

theorem symmetric_cubic_homogeneous_decomposition
  (f : ℝ → ℝ → ℝ → ℝ) (h : SymmetricCubicHomogeneous f) :
  ∃! (a b c : ℝ), ∀ x y z : ℝ, f x y z = a * g₁ x y z + b * g₂ x y z + c * g₃ x y z :=
sorry

theorem non_negative_equivalence
  (f : ℝ → ℝ → ℝ → ℝ) (h : SymmetricCubicHomogeneous f)
  (a b c : ℝ) (h_decomp : ∀ x y z : ℝ, f x y z = a * g₁ x y z + b * g₂ x y z + c * g₃ x y z) :
  (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → f x y z ≥ 0) ↔ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :=
sorry

end symmetric_cubic_homogeneous_decomposition_non_negative_equivalence_l1755_175596


namespace go_match_results_l1755_175567

structure GoMatch where
  redWinProb : ℝ
  mk_prob_valid : 0 ≤ redWinProb ∧ redWinProb ≤ 1

def RedTeam := Fin 3 → GoMatch

def atLeastTwoWins (team : RedTeam) : ℝ :=
  sorry

def expectedWins (team : RedTeam) : ℝ :=
  sorry

theorem go_match_results (team : RedTeam) 
  (h1 : team 0 = ⟨0.6, sorry⟩) 
  (h2 : team 1 = ⟨0.5, sorry⟩)
  (h3 : team 2 = ⟨0.5, sorry⟩) :
  atLeastTwoWins team = 0.55 ∧ expectedWins team = 1.6 := by
  sorry

end go_match_results_l1755_175567


namespace xe_pow_x_strictly_increasing_l1755_175517

/-- The function f(x) = xe^x is strictly increasing on the interval (-1, +∞) -/
theorem xe_pow_x_strictly_increasing :
  ∀ x₁ x₂, -1 < x₁ → x₁ < x₂ → x₁ * Real.exp x₁ < x₂ * Real.exp x₂ := by
  sorry

end xe_pow_x_strictly_increasing_l1755_175517


namespace angle_measure_in_special_quadrilateral_l1755_175595

theorem angle_measure_in_special_quadrilateral :
  ∀ (P Q R S : ℝ),
  P = 3 * Q →
  P = 4 * R →
  P = 6 * S →
  P + Q + R + S = 360 →
  P = 206 := by
sorry

end angle_measure_in_special_quadrilateral_l1755_175595


namespace gcd_of_90_and_405_l1755_175564

theorem gcd_of_90_and_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_of_90_and_405_l1755_175564


namespace shortest_path_length_l1755_175523

/-- Regular tetrahedron with edge length 2 -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_regular : edge_length = 2)

/-- Point on the surface of a regular tetrahedron -/
structure SurfacePoint (t : RegularTetrahedron) :=
  (coordinates : ℝ × ℝ × ℝ)

/-- Midpoint of an edge on a regular tetrahedron -/
def edge_midpoint (t : RegularTetrahedron) : SurfacePoint t :=
  sorry

/-- Distance between two points on the surface of a regular tetrahedron -/
def surface_distance (t : RegularTetrahedron) (p q : SurfacePoint t) : ℝ :=
  sorry

/-- Sequentially next edge midpoint -/
def next_edge_midpoint (t : RegularTetrahedron) (p : SurfacePoint t) : SurfacePoint t :=
  sorry

/-- Theorem: Shortest path between midpoints of sequentially next edges is √6 -/
theorem shortest_path_length (t : RegularTetrahedron) (p : SurfacePoint t) :
  surface_distance t p (next_edge_midpoint t p) = Real.sqrt 6 :=
sorry

end shortest_path_length_l1755_175523


namespace tile_draw_probability_l1755_175525

/-- The number of tiles in box A -/
def box_a_size : ℕ := 25

/-- The number of tiles in box B -/
def box_b_size : ℕ := 30

/-- The lowest number on a tile in box A -/
def box_a_min : ℕ := 1

/-- The highest number on a tile in box A -/
def box_a_max : ℕ := 25

/-- The lowest number on a tile in box B -/
def box_b_min : ℕ := 10

/-- The highest number on a tile in box B -/
def box_b_max : ℕ := 39

/-- The threshold for "less than" condition in box A -/
def box_a_threshold : ℕ := 18

/-- The threshold for "greater than" condition in box B -/
def box_b_threshold : ℕ := 30

theorem tile_draw_probability : 
  (((box_a_threshold - box_a_min : ℚ) / box_a_size) * 
   ((box_b_size - (box_b_threshold - box_b_min + 1) / 2 + (box_b_max - box_b_threshold)) / box_b_size)) = 323 / 750 := by
  sorry


end tile_draw_probability_l1755_175525


namespace dihedral_angle_range_l1755_175597

/-- The dihedral angle between adjacent faces in a regular n-prism -/
def dihedral_angle (n : ℕ) (θ : ℝ) : Prop :=
  n > 2 ∧ ((n - 2 : ℝ) / n) * Real.pi < θ ∧ θ < Real.pi

/-- Theorem stating the range of dihedral angles in a regular n-prism -/
theorem dihedral_angle_range (n : ℕ) :
  ∃ θ : ℝ, dihedral_angle n θ :=
sorry

end dihedral_angle_range_l1755_175597


namespace right_triangle_with_median_condition_l1755_175537

theorem right_triangle_with_median_condition (c : ℝ) (h : c > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    a^2 + b^2 = c^2 ∧
    (c / 2)^2 = a * b ∧
    a = (c * (Real.sqrt 6 + Real.sqrt 2)) / 4 ∧
    b = (c * (Real.sqrt 6 - Real.sqrt 2)) / 4 :=
by sorry

end right_triangle_with_median_condition_l1755_175537


namespace school_trip_theorem_l1755_175551

/-- The number of school buses -/
def num_buses : ℕ := 95

/-- The number of seats on each bus -/
def seats_per_bus : ℕ := 118

/-- All buses are fully filled -/
axiom buses_full : True

/-- The total number of students in the school -/
def total_students : ℕ := num_buses * seats_per_bus

theorem school_trip_theorem : total_students = 11210 := by
  sorry

end school_trip_theorem_l1755_175551


namespace sum_of_star_equation_l1755_175585

/-- Custom operation ★ -/
def star (a b : ℕ) : ℕ := a^b + a + b

theorem sum_of_star_equation {a b : ℕ} (ha : a ≥ 2) (hb : b ≥ 2) (heq : star a b = 20) :
  a + b = 6 := by
  sorry

end sum_of_star_equation_l1755_175585


namespace hearts_clubs_equal_prob_l1755_175561

/-- Represents the suit of a playing card -/
inductive Suit
| Hearts
| Clubs
| Diamonds
| Spades

/-- Represents the rank of a playing card -/
inductive Rank
| Ace
| Two
| Three
| Four
| Five
| Six
| Seven
| Eight
| Nine
| Ten
| Jack
| Queen
| King

/-- Represents a playing card -/
structure Card where
  suit : Suit
  rank : Rank

/-- Represents a standard deck of playing cards -/
def standardDeck : Finset Card := sorry

/-- The number of cards in a standard deck -/
def deckSize : Nat := 52

/-- The number of cards of each suit in a standard deck -/
def suitCount : Nat := 13

/-- The probability of drawing a card of a specific suit from a standard deck -/
def probSuit (s : Suit) : ℚ :=
  suitCount / deckSize

theorem hearts_clubs_equal_prob :
  probSuit Suit.Hearts = probSuit Suit.Clubs := by
  sorry

end hearts_clubs_equal_prob_l1755_175561


namespace consecutive_squares_sum_l1755_175524

theorem consecutive_squares_sum (x : ℕ) : 
  (x - 1)^2 + x^2 + (x + 1)^2 = 8 * ((x - 1) + x + (x + 1)) + 2 →
  ∃ n : ℕ, (n - 1)^2 + n^2 + (n + 1)^2 = 194 := by
  sorry

end consecutive_squares_sum_l1755_175524


namespace frank_riding_time_l1755_175559

-- Define the riding times for each person
def dave_time : ℝ := 10

-- Chuck's time is 5 times Dave's time
def chuck_time : ℝ := 5 * dave_time

-- Erica's time is 30% longer than Chuck's time
def erica_time : ℝ := chuck_time * (1 + 0.3)

-- Frank's time is 20% longer than Erica's time
def frank_time : ℝ := erica_time * (1 + 0.2)

-- Theorem to prove
theorem frank_riding_time : frank_time = 78 := by
  sorry

end frank_riding_time_l1755_175559


namespace money_division_l1755_175589

theorem money_division (total : ℕ) (p q r : ℕ) (h1 : p + q + r = total) (h2 : p = 3 * (total / 22)) (h3 : q = 7 * (total / 22)) (h4 : r = 12 * (total / 22)) (h5 : r - q = 5500) : q - p = 4400 := by
  sorry

end money_division_l1755_175589


namespace intersection_of_sets_l1755_175546

theorem intersection_of_sets (P Q : Set ℝ) : 
  (P = {y : ℝ | ∃ x : ℝ, y = x + 1}) → 
  (Q = {y : ℝ | ∃ x : ℝ, y = 1 - x}) → 
  P ∩ Q = Set.univ := by
  sorry

end intersection_of_sets_l1755_175546


namespace solve_system_of_equations_l1755_175568

theorem solve_system_of_equations (x y : ℤ) 
  (h1 : x + y = 14) 
  (h2 : x - y = 60) : 
  x = 37 := by sorry

end solve_system_of_equations_l1755_175568


namespace f_even_and_increasing_l1755_175519

def f (x : ℝ) : ℝ := |x| + 1

theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end f_even_and_increasing_l1755_175519


namespace larger_solution_quadratic_l1755_175522

theorem larger_solution_quadratic (x : ℝ) : 
  x^2 - 9*x - 22 = 0 → x ≤ 11 :=
by
  sorry

end larger_solution_quadratic_l1755_175522


namespace cake_distribution_l1755_175587

theorem cake_distribution (total_cake : ℕ) (friends : ℕ) (pieces_per_friend : ℕ) 
  (h1 : total_cake = 150)
  (h2 : friends = 50)
  (h3 : pieces_per_friend * friends = total_cake) :
  pieces_per_friend = 3 := by
  sorry

end cake_distribution_l1755_175587


namespace kiera_had_one_fruit_cup_l1755_175578

/-- Represents the breakfast items and their costs -/
structure Breakfast where
  muffin_cost : ℕ
  fruit_cup_cost : ℕ
  francis_muffins : ℕ
  francis_fruit_cups : ℕ
  kiera_muffins : ℕ
  total_cost : ℕ

/-- Calculates the number of fruit cups Kiera had -/
def kieras_fruit_cups (b : Breakfast) : ℕ :=
  (b.total_cost - (b.francis_muffins * b.muffin_cost + b.francis_fruit_cups * b.fruit_cup_cost + b.kiera_muffins * b.muffin_cost)) / b.fruit_cup_cost

/-- Theorem stating that Kiera had 1 fruit cup given the problem conditions -/
theorem kiera_had_one_fruit_cup (b : Breakfast) 
  (h1 : b.muffin_cost = 2)
  (h2 : b.fruit_cup_cost = 3)
  (h3 : b.francis_muffins = 2)
  (h4 : b.francis_fruit_cups = 2)
  (h5 : b.kiera_muffins = 2)
  (h6 : b.total_cost = 17) :
  kieras_fruit_cups b = 1 := by
  sorry

#eval kieras_fruit_cups { muffin_cost := 2, fruit_cup_cost := 3, francis_muffins := 2, francis_fruit_cups := 2, kiera_muffins := 2, total_cost := 17 }

end kiera_had_one_fruit_cup_l1755_175578


namespace root_transformation_l1755_175555

theorem root_transformation (a b c d : ℂ) : 
  (a^4 - 2*a - 6 = 0) ∧ 
  (b^4 - 2*b - 6 = 0) ∧ 
  (c^4 - 2*c - 6 = 0) ∧ 
  (d^4 - 2*d - 6 = 0) →
  ∃ (y₁ y₂ y₃ y₄ : ℂ), 
    y₁ = 2*(a + b + c)/d^3 ∧
    y₂ = 2*(a + b + d)/c^3 ∧
    y₃ = 2*(a + c + d)/b^3 ∧
    y₄ = 2*(b + c + d)/a^3 ∧
    (2*y₁^4 - 2*y₁ + 48 = 0) ∧
    (2*y₂^4 - 2*y₂ + 48 = 0) ∧
    (2*y₃^4 - 2*y₃ + 48 = 0) ∧
    (2*y₄^4 - 2*y₄ + 48 = 0) :=
by sorry

end root_transformation_l1755_175555


namespace max_k_value_l1755_175590

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * (x / y + y / x)) :
  k ≤ 3/7 := by
  sorry

end max_k_value_l1755_175590


namespace perpendicular_vectors_k_value_l1755_175527

/-- Given vectors a, b, and c in ℝ², where a = (1,2), b = (0,1), and c = (-2,k),
    if (a + 2b) is perpendicular to c, then k = 1/2. -/
theorem perpendicular_vectors_k_value :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![0, 1]
  let c : Fin 2 → ℝ := ![-2, k]
  (∀ i : Fin 2, (a i + 2 * b i) * c i = 0) →
  k = 1/2 := by
  sorry

end perpendicular_vectors_k_value_l1755_175527


namespace rain_probability_l1755_175502

/-- The probability of rain on Friday -/
def prob_rain_friday : ℝ := 0.40

/-- The probability of rain on Monday -/
def prob_rain_monday : ℝ := 0.35

/-- The probability of rain on both Friday and Monday -/
def prob_rain_both : ℝ := prob_rain_friday * prob_rain_monday

theorem rain_probability : prob_rain_both = 0.14 := by
  sorry

end rain_probability_l1755_175502


namespace transformation_maps_correctly_l1755_175565

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Scales a point by a given factor -/
def scale (p : Point) (factor : ℝ) : Point :=
  ⟨p.x * factor, p.y * factor⟩

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Applies scaling followed by reflection across x-axis -/
def scaleAndReflect (p : Point) (factor : ℝ) : Point :=
  reflectX (scale p factor)

theorem transformation_maps_correctly :
  let A : Point := ⟨1, 2⟩
  let B : Point := ⟨2, 3⟩
  let A' : Point := ⟨3, -6⟩
  let B' : Point := ⟨6, -9⟩
  (scaleAndReflect A 3 = A') ∧ (scaleAndReflect B 3 = B') := by
  sorry

end transformation_maps_correctly_l1755_175565


namespace sum_distances_foci_to_line_l1755_175531

/-- The ellipse C in the xy-plane -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

/-- The left focus of ellipse C -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- The right focus of ellipse C -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Distance from a point to a line -/
def dist_point_to_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- Theorem: The sum of distances from the foci of ellipse C to line l is 2√2 -/
theorem sum_distances_foci_to_line :
  dist_point_to_line F₁ line_l + dist_point_to_line F₂ line_l = 2 * Real.sqrt 2 :=
sorry

end sum_distances_foci_to_line_l1755_175531


namespace periodic_exponential_function_l1755_175582

theorem periodic_exponential_function (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f (x + 2) = f x) →
  (∀ x ∈ Set.Icc (-1) 1, f x = 2^(x + a)) →
  f 2017 = 8 →
  a = 2 := by
sorry

end periodic_exponential_function_l1755_175582


namespace white_bread_count_l1755_175576

/-- The number of loaves of white bread bought each week -/
def white_bread_loaves : ℕ := sorry

/-- The cost of a loaf of white bread -/
def white_bread_cost : ℚ := 7/2

/-- The cost of a baguette -/
def baguette_cost : ℚ := 3/2

/-- The number of sourdough loaves bought each week -/
def sourdough_loaves : ℕ := 2

/-- The cost of a loaf of sourdough bread -/
def sourdough_cost : ℚ := 9/2

/-- The cost of an almond croissant -/
def croissant_cost : ℚ := 2

/-- The number of weeks -/
def weeks : ℕ := 4

/-- The total amount spent over all weeks -/
def total_spent : ℚ := 78

theorem white_bread_count :
  white_bread_loaves = 2 ∧
  (white_bread_loaves : ℚ) * white_bread_cost +
  baguette_cost +
  (sourdough_loaves : ℚ) * sourdough_cost +
  croissant_cost =
  total_spent / (weeks : ℚ) :=
sorry

end white_bread_count_l1755_175576


namespace sector_area_theorem_l1755_175569

/-- Given a circular sector with central angle θ and arc length l,
    prove that if θ = 2 and l = 2, then the area of the sector is 1. -/
theorem sector_area_theorem (θ l : Real) (h1 : θ = 2) (h2 : l = 2) :
  let r := l / θ
  (1 / 2) * r^2 * θ = 1 := by
  sorry

end sector_area_theorem_l1755_175569


namespace alcohol_solution_proof_l1755_175593

/-- Proves that adding 2.4 liters of pure alcohol to a 6-liter solution that is 30% alcohol
    will result in a solution that is 50% alcohol. -/
theorem alcohol_solution_proof (initial_volume : ℝ) (initial_concentration : ℝ)
  (added_alcohol : ℝ) (target_concentration : ℝ)
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.3)
  (h3 : added_alcohol = 2.4)
  (h4 : target_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = target_concentration :=
by sorry

end alcohol_solution_proof_l1755_175593


namespace plan_d_cheaper_at_291_l1755_175513

def plan_c_cost (minutes : ℕ) : ℚ := 15 * minutes

def plan_d_cost (minutes : ℕ) : ℚ :=
  if minutes ≤ 100 then
    2500 + 4 * minutes
  else
    2900 + 5 * (minutes - 100)

theorem plan_d_cheaper_at_291 :
  ∀ m : ℕ, m < 291 → plan_c_cost m ≤ plan_d_cost m ∧
  plan_c_cost 291 > plan_d_cost 291 :=
by sorry

end plan_d_cheaper_at_291_l1755_175513


namespace unique_solution_exists_l1755_175573

theorem unique_solution_exists : ∃! (x y : ℝ), 
  0.75 * x - 0.40 * y = 0.20 * 422.50 ∧ 
  0.30 * x + 0.50 * y = 0.35 * 530 := by
  sorry

end unique_solution_exists_l1755_175573


namespace clara_cookies_sold_l1755_175560

/-- Calculates the total number of cookies sold by Clara -/
def total_cookies_sold (cookies_per_box : Fin 3 → ℕ) (boxes_sold : Fin 3 → ℕ) : ℕ :=
  (cookies_per_box 0) * (boxes_sold 0) + 
  (cookies_per_box 1) * (boxes_sold 1) + 
  (cookies_per_box 2) * (boxes_sold 2)

/-- Proves that Clara sells 3320 cookies in total -/
theorem clara_cookies_sold :
  let cookies_per_box : Fin 3 → ℕ := ![12, 20, 16]
  let boxes_sold : Fin 3 → ℕ := ![50, 80, 70]
  total_cookies_sold cookies_per_box boxes_sold = 3320 := by
  sorry

end clara_cookies_sold_l1755_175560


namespace kitchen_tile_size_l1755_175570

/-- Given a rectangular kitchen floor and the number of tiles needed, 
    calculate the size of each tile. -/
theorem kitchen_tile_size 
  (length : ℕ) 
  (width : ℕ) 
  (num_tiles : ℕ) 
  (h1 : length = 48) 
  (h2 : width = 72) 
  (h3 : num_tiles = 96) : 
  (length * width) / num_tiles = 36 := by
  sorry

#check kitchen_tile_size

end kitchen_tile_size_l1755_175570


namespace inequality_solution_quadratic_solution_l1755_175539

-- Part 1: Integer solutions of the inequality
def integer_solutions : Set ℤ :=
  {x : ℤ | -2 ≤ (1 + 2*x) / 3 ∧ (1 + 2*x) / 3 ≤ 2}

theorem inequality_solution :
  integer_solutions = {-3, -2, -1, 0, 1, 2} := by sorry

-- Part 2: Quadratic equation
def quadratic_equation (a b : ℚ) (x : ℚ) : ℚ :=
  a * x^2 + b * x

theorem quadratic_solution (a b : ℚ) :
  (quadratic_equation a b 1 = 0 ∧ quadratic_equation a b 2 = 3) →
  quadratic_equation a b (-2) = 9 := by sorry

end inequality_solution_quadratic_solution_l1755_175539


namespace solution_set_of_inequality_l1755_175500

theorem solution_set_of_inequality (x : ℝ) :
  (x + 3) / (x - 1) > 0 ↔ x < -3 ∨ x > 1 :=
by sorry

end solution_set_of_inequality_l1755_175500


namespace ribbon_per_gift_l1755_175577

theorem ribbon_per_gift (total_gifts : ℕ) (total_ribbon : ℝ) (remaining_ribbon : ℝ) 
  (h1 : total_gifts = 8)
  (h2 : total_ribbon = 15)
  (h3 : remaining_ribbon = 3) :
  (total_ribbon - remaining_ribbon) / total_gifts = 1.5 := by
  sorry

end ribbon_per_gift_l1755_175577


namespace correct_subtraction_result_l1755_175571

theorem correct_subtraction_result 
  (mistaken_result : ℕ)
  (tens_digit_increase : ℕ)
  (units_digit_increase : ℕ)
  (h1 : mistaken_result = 217)
  (h2 : tens_digit_increase = 3)
  (h3 : units_digit_increase = 4) :
  mistaken_result - (tens_digit_increase * 10 - units_digit_increase) = 191 :=
by sorry

end correct_subtraction_result_l1755_175571


namespace platform_length_l1755_175545

/-- The length of a platform given a train's speed and passing times -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) :
  train_speed = 54 →
  platform_time = 30 →
  man_time = 20 →
  (train_speed * 1000 / 3600) * platform_time - (train_speed * 1000 / 3600) * man_time = 150 := by
  sorry

end platform_length_l1755_175545


namespace complex_power_2019_l1755_175584

-- Define the imaginary unit i
variable (i : ℂ)

-- Define the property of i being the imaginary unit
axiom i_squared : i^2 = -1

-- State the theorem
theorem complex_power_2019 : (((1 + i) / (1 - i)) ^ 2019 : ℂ) = -i := by sorry

end complex_power_2019_l1755_175584


namespace not_p_and_not_q_true_l1755_175549

theorem not_p_and_not_q_true (p q : Prop)
  (h1 : ¬(p ∧ q))
  (h2 : ¬(p ∨ q)) :
  (¬p ∧ ¬q) :=
by sorry

end not_p_and_not_q_true_l1755_175549


namespace robot_position_l1755_175516

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the Cartesian plane defined by y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The robot's path is defined as the set of points equidistant from two given points -/
def RobotPath (A B : Point) : Set Point :=
  {P : Point | (P.x - A.x)^2 + (P.y - A.y)^2 = (P.x - B.x)^2 + (P.y - B.y)^2}

/-- Check if a point is on a line -/
def isOnLine (P : Point) (L : Line) : Prop :=
  P.y = L.m * P.x + L.b

theorem robot_position (a : ℝ) : 
  let A : Point := ⟨a, 0⟩
  let B : Point := ⟨0, 1⟩
  let L : Line := ⟨1, 1⟩  -- y = x + 1
  (∀ P ∈ RobotPath A B, ¬isOnLine P L) → a = 1 := by
  sorry

end robot_position_l1755_175516


namespace parallel_vectors_m_values_l1755_175510

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The theorem statement -/
theorem parallel_vectors_m_values (m : ℝ) :
  parallel (2, m) (m, 2) → m = -2 ∨ m = 2 := by
  sorry


end parallel_vectors_m_values_l1755_175510


namespace power_multiplication_l1755_175509

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l1755_175509


namespace circle_center_coordinates_l1755_175572

-- Define the circle's center
structure CircleCenter where
  x : ℝ
  y : ℝ

-- Define the conditions for the circle's center
def satisfiesConditions (c : CircleCenter) : Prop :=
  c.x - 2 * c.y = 0 ∧
  3 * c.x - 4 * c.y = 10

-- Theorem statement
theorem circle_center_coordinates :
  ∃ (c : CircleCenter), satisfiesConditions c ∧ c.x = 10 ∧ c.y = 5 :=
by sorry

end circle_center_coordinates_l1755_175572


namespace cosine_is_periodic_l1755_175541

-- Define a type for functions from ℝ to ℝ
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be periodic
def IsPeriodic (f : RealFunction) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

-- Define what it means for a function to be trigonometric
def IsTrigonometric (f : RealFunction) : Prop :=
  -- This is a placeholder definition
  True

-- State the theorem
theorem cosine_is_periodic :
  (∀ f : RealFunction, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric (λ x : ℝ => Real.cos x) →
  IsPeriodic (λ x : ℝ => Real.cos x) :=
by
  sorry

end cosine_is_periodic_l1755_175541


namespace line_m_equation_l1755_175556

/-- Two distinct lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point about a line -/
def reflect (p : Point) (l : Line) : Point := sorry

/-- The theorem statement -/
theorem line_m_equation (l m : Line) (Q Q'' : Point) :
  l.a = 3 ∧ l.b = -4 ∧ l.c = 0 →  -- Line ℓ: 3x - 4y = 0
  Q.x = 3 ∧ Q.y = -2 →  -- Point Q(3, -2)
  Q''.x = 2 ∧ Q''.y = 5 →  -- Point Q''(2, 5)
  (∃ Q' : Point, reflect Q l = Q' ∧ reflect Q' m = Q'') →  -- Reflection conditions
  m.a = 1 ∧ m.b = 7 ∧ m.c = 0  -- Line m: x + 7y = 0
  := by sorry

end line_m_equation_l1755_175556


namespace range_of_f_l1755_175579

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end range_of_f_l1755_175579


namespace parabola_intersection_length_l1755_175504

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop :=
  ∃ (t : ℝ), x = 2 + t ∧ y = t

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_through_focus x₁ y₁ ∧ line_through_focus x₂ y₂

-- Theorem statement
theorem parabola_intersection_length
  (x₁ y₁ x₂ y₂ : ℝ)
  (h_intersection : intersection_points x₁ y₁ x₂ y₂)
  (h_sum : x₁ + x₂ = 6) :
  ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2 : ℝ) = 10 :=
sorry

end parabola_intersection_length_l1755_175504


namespace extremum_point_and_monotonicity_l1755_175591

noncomputable section

variables (x : ℝ) (m : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + m)

def f_derivative (x : ℝ) : ℝ := Real.exp x - 1 / (x + m)

theorem extremum_point_and_monotonicity :
  (f_derivative 0 = 0) →
  (m = 1) ∧
  (∀ x > 0, f_derivative x > 0) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f_derivative x < 0) :=
by sorry

end

end extremum_point_and_monotonicity_l1755_175591


namespace equation_proof_l1755_175547

-- Define the variables and the given equation
theorem equation_proof (x : ℝ) (Q : ℝ) (h : 4 * (5 * x + 7 * Real.pi) = Q) :
  8 * (10 * x + 14 * Real.pi) = 4 * Q := by
  sorry

end equation_proof_l1755_175547


namespace minimum_bottles_needed_l1755_175520

def small_bottle_capacity : ℚ := 45
def large_bottle_1_capacity : ℚ := 630
def large_bottle_2_capacity : ℚ := 850

theorem minimum_bottles_needed : 
  ∃ (n : ℕ), n * small_bottle_capacity ≥ large_bottle_1_capacity + large_bottle_2_capacity ∧
  ∀ (m : ℕ), m * small_bottle_capacity ≥ large_bottle_1_capacity + large_bottle_2_capacity → m ≥ n ∧
  n = 33 := by
  sorry

end minimum_bottles_needed_l1755_175520


namespace max_height_foldable_triangle_l1755_175583

/-- The maximum height of a foldable table constructed from a triangle --/
theorem max_height_foldable_triangle (PQ QR PR : ℝ) (h_PQ : PQ = 24) (h_QR : QR = 32) (h_PR : PR = 40) :
  let s := (PQ + QR + PR) / 2
  let A := Real.sqrt (s * (s - PQ) * (s - QR) * (s - PR))
  let h_p := 2 * A / QR
  let h_q := 2 * A / PR
  let h_r := 2 * A / PQ
  let h' := min (h_p * h_q / (h_p + h_q)) (min (h_q * h_r / (h_q + h_r)) (h_r * h_p / (h_r + h_p)))
  h' = 48 * Real.sqrt 2 / 7 :=
by sorry

end max_height_foldable_triangle_l1755_175583


namespace factorizable_polynomial_count_l1755_175563

theorem factorizable_polynomial_count : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 2000) ∧
    (∀ n ∈ S, ∃ a b : ℤ, ∀ x, x^2 - 3*x - n = (x - a) * (x - b)) ∧
    (∀ n : ℕ, 1 ≤ n → n ≤ 2000 → 
      (∃ a b : ℤ, ∀ x, x^2 - 3*x - n = (x - a) * (x - b)) → n ∈ S) ∧
    S.card = 44 :=
by
  sorry

end factorizable_polynomial_count_l1755_175563


namespace basketball_tryouts_l1755_175529

theorem basketball_tryouts (girls boys called_back : ℕ) 
  (h1 : girls = 15)
  (h2 : boys = 25)
  (h3 : called_back = 7) :
  girls + boys - called_back = 33 := by
sorry

end basketball_tryouts_l1755_175529


namespace problem_solution_l1755_175548

noncomputable section

def U : Set ℝ := Set.univ

def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}

def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

theorem problem_solution (a : ℝ) :
  (∃ x, x ∈ A a) ∧ (∃ x, x ∈ B a) →
  (a = 1/2 → (U \ B a) ∩ A a = {x | 9/4 ≤ x ∧ x < 5/2}) ∧
  (A a ⊆ B a ↔ a ∈ Set.Icc (-1/2) (1/3) ∪ Set.Ioc (1/3) ((3 - Real.sqrt 5) / 2)) :=
sorry

end

end problem_solution_l1755_175548


namespace rhonda_marbles_l1755_175528

theorem rhonda_marbles (total : ℕ) (diff : ℕ) (rhonda : ℕ) : 
  total = 215 → diff = 55 → total = rhonda + (rhonda + diff) → rhonda = 80 := by
  sorry

end rhonda_marbles_l1755_175528


namespace johns_bookshop_l1755_175536

/-- The total number of books sold over 5 days -/
def total_sold : ℕ := 280

/-- The percentage of books that were not sold -/
def percent_not_sold : ℚ := 54.83870967741935

/-- The initial number of books in John's bookshop -/
def initial_books : ℕ := 620

theorem johns_bookshop :
  initial_books = total_sold / ((100 - percent_not_sold) / 100) := by sorry

end johns_bookshop_l1755_175536


namespace three_Z_seven_l1755_175540

-- Define the operation Z
def Z (a b : ℝ) : ℝ := b + 5 * a - 2 * a^2

-- Theorem to prove
theorem three_Z_seven : Z 3 7 = 4 := by
  sorry

end three_Z_seven_l1755_175540


namespace circle_diameter_ratio_l1755_175538

theorem circle_diameter_ratio (D C : Real) (h1 : D = 20) 
  (h2 : C > 0) (h3 : C < D) 
  (h4 : (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 4) : 
  C = 4 * Real.sqrt 5 := by
sorry

end circle_diameter_ratio_l1755_175538


namespace square_sum_given_cube_sum_and_product_l1755_175553

theorem square_sum_given_cube_sum_and_product (x y : ℝ) : 
  (x + y)^3 = 8 → x * y = 5 → x^2 + y^2 = -6 :=
by
  sorry

end square_sum_given_cube_sum_and_product_l1755_175553


namespace baker_april_earnings_l1755_175532

def baker_earnings (cake_price cake_sold pie_price pie_sold bread_price bread_sold cookie_price cookie_sold pie_discount tax_rate : ℚ) : ℚ :=
  let cake_revenue := cake_price * cake_sold
  let pie_revenue := pie_price * pie_sold * (1 - pie_discount)
  let bread_revenue := bread_price * bread_sold
  let cookie_revenue := cookie_price * cookie_sold
  let total_revenue := cake_revenue + pie_revenue + bread_revenue + cookie_revenue
  total_revenue * (1 + tax_rate)

theorem baker_april_earnings :
  baker_earnings 12 453 7 126 3.5 95 1.5 320 0.1 0.05 = 7394.42 := by
  sorry

end baker_april_earnings_l1755_175532


namespace choose_three_from_eight_l1755_175552

theorem choose_three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end choose_three_from_eight_l1755_175552


namespace chess_tournament_solutions_l1755_175580

def chess_tournament (x : ℕ) : Prop :=
  ∃ y : ℕ,
    -- Two 7th graders scored 8 points in total
    8 + x * y = (x + 2) * (x + 1) / 2 ∧
    -- y is the number of points each 8th grader scored
    y > 0

theorem chess_tournament_solutions :
  ∀ x : ℕ, chess_tournament x ↔ (x = 7 ∨ x = 14) :=
by sorry

end chess_tournament_solutions_l1755_175580


namespace fraction_decomposition_l1755_175535

theorem fraction_decomposition :
  ∃ (C D : ℝ),
    (C = -0.1 ∧ D = 7.3) ∧
    ∀ (x : ℝ), x ≠ 2 ∧ 3*x ≠ -4 →
      (7*x - 15) / (3*x^2 + 2*x - 8) = C / (x - 2) + D / (3*x + 4) := by
  sorry

end fraction_decomposition_l1755_175535
