import Mathlib

namespace ohara_triple_49_16_l3668_366834

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: The value of x in the O'Hara triple (49, 16, x) is 11 -/
theorem ohara_triple_49_16 :
  ∃ x : ℕ, is_ohara_triple 49 16 x ∧ x = 11 := by
  sorry

end ohara_triple_49_16_l3668_366834


namespace smallest_difference_l3668_366898

theorem smallest_difference (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) :
  ∃ (m : ℤ), m = a - b ∧ (∀ (c d : ℤ), c + d < 11 → c > 6 → c - d ≥ m) := by
  sorry

end smallest_difference_l3668_366898


namespace modulus_of_z_l3668_366882

-- Define the complex number z
def z : ℂ := sorry

-- State the given equation
axiom z_equation : z^2 + z = 1 - 3*Complex.I

-- Define the theorem
theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by sorry

end modulus_of_z_l3668_366882


namespace sqrt_sum_problem_l3668_366837

theorem sqrt_sum_problem (x : ℝ) (h_pos : x > 0) (h_eq : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_sum_problem_l3668_366837


namespace smallest_integer_ending_in_9_divisible_by_13_l3668_366813

theorem smallest_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 9 ∧ n % 13 = 0 ∧
  ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 13 = 0 → m ≥ n :=
by
  use 39
  sorry

end smallest_integer_ending_in_9_divisible_by_13_l3668_366813


namespace trapezoid_longer_side_length_l3668_366808

/-- Given a square with side length 1, divided into one triangle and three trapezoids
    by joining the center to points on each side, where these points divide each side
    into segments of length 1/4 and 3/4, and each section has equal area,
    prove that the length of the longer parallel side of the trapezoids is 3/4. -/
theorem trapezoid_longer_side_length (square_side : ℝ) (segment_short : ℝ) (segment_long : ℝ)
  (h_square_side : square_side = 1)
  (h_segment_short : segment_short = 1/4)
  (h_segment_long : segment_long = 3/4)
  (h_segments_sum : segment_short + segment_long = square_side)
  (h_equal_areas : ∀ section_area : ℝ, section_area = (square_side^2) / 4) :
  ∃ x : ℝ, x = 3/4 ∧ x = segment_long :=
by sorry

end trapezoid_longer_side_length_l3668_366808


namespace line_parameterization_l3668_366811

/-- Given a line y = 2x - 40 parameterized by (x, y) = (g(t), 20t - 14),
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) :
  (∀ x y, y = 2 * x - 40 ↔ ∃ t, x = g t ∧ y = 20 * t - 14) →
  ∀ t, g t = 10 * t + 13 := by
sorry

end line_parameterization_l3668_366811


namespace x_intercept_of_line_l3668_366815

/-- The x-intercept of the line 2x - 4y = 12 is 6 -/
theorem x_intercept_of_line (x y : ℝ) : 2 * x - 4 * y = 12 → y = 0 → x = 6 := by
  sorry

end x_intercept_of_line_l3668_366815


namespace lilia_earnings_l3668_366885

/-- Represents Lilia's peach selling scenario -/
structure PeachSale where
  total : Nat
  sold_to_friends : Nat
  price_friends : Real
  sold_to_relatives : Nat
  price_relatives : Real
  kept : Nat

/-- Calculates the total earnings from selling peaches -/
def total_earnings (sale : PeachSale) : Real :=
  sale.sold_to_friends * sale.price_friends + sale.sold_to_relatives * sale.price_relatives

/-- Theorem stating that Lilia's earnings from selling 14 peaches is $25 -/
theorem lilia_earnings (sale : PeachSale) 
  (h1 : sale.total = 15)
  (h2 : sale.sold_to_friends = 10)
  (h3 : sale.price_friends = 2)
  (h4 : sale.sold_to_relatives = 4)
  (h5 : sale.price_relatives = 1.25)
  (h6 : sale.kept = 1)
  (h7 : sale.sold_to_friends + sale.sold_to_relatives + sale.kept = sale.total) :
  total_earnings sale = 25 := by
  sorry

end lilia_earnings_l3668_366885


namespace orange_calories_l3668_366849

/-- Proves that the number of calories per orange is 80 given the problem conditions -/
theorem orange_calories (orange_cost : ℚ) (initial_amount : ℚ) (required_calories : ℕ) (remaining_amount : ℚ) :
  orange_cost = 6/5 ∧ 
  initial_amount = 10 ∧ 
  required_calories = 400 ∧ 
  remaining_amount = 4 →
  (initial_amount - remaining_amount) / orange_cost * required_calories / ((initial_amount - remaining_amount) / orange_cost) = 80 := by
sorry

end orange_calories_l3668_366849


namespace locus_characterization_l3668_366888

/-- The locus of points equidistant from A(4, 1) and the y-axis -/
def locus_equation (x y : ℝ) : Prop :=
  (y - 1)^2 = 16 * (x - 2)

/-- A point P(x, y) is equidistant from A(4, 1) and the y-axis -/
def is_equidistant (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 1)^2 = x^2

theorem locus_characterization (x y : ℝ) :
  is_equidistant x y ↔ locus_equation x y := by sorry

end locus_characterization_l3668_366888


namespace infinite_power_tower_equals_four_l3668_366816

-- Define the infinite power tower function
noncomputable def powerTower (x : ℝ) : ℝ := Real.sqrt (4 : ℝ)

-- State the theorem
theorem infinite_power_tower_equals_four (x : ℝ) (h₁ : x > 0) :
  powerTower x = 4 → x = Real.sqrt 2 := by
  sorry

end infinite_power_tower_equals_four_l3668_366816


namespace abc_inequality_l3668_366810

theorem abc_inequality (a b c : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) (hc : -1 < c ∧ c < 1) :
  a * b * c + 2 > a + b + c := by
  sorry

end abc_inequality_l3668_366810


namespace books_per_bookshelf_l3668_366897

/-- Given that Bryan has 34 books distributed equally in 2 bookshelves,
    prove that there are 17 books in each bookshelf. -/
theorem books_per_bookshelf :
  ∀ (total_books : ℕ) (num_bookshelves : ℕ) (books_per_shelf : ℕ),
    total_books = 34 →
    num_bookshelves = 2 →
    total_books = num_bookshelves * books_per_shelf →
    books_per_shelf = 17 := by
  sorry

end books_per_bookshelf_l3668_366897


namespace lollipop_reimbursement_l3668_366838

/-- Given that Sarah bought 12 lollipops for 3 dollars and shared one-quarter with Julie,
    prove that Julie reimbursed Sarah 75 cents. -/
theorem lollipop_reimbursement (total_lollipops : ℕ) (total_cost : ℚ) (share_fraction : ℚ) :
  total_lollipops = 12 →
  total_cost = 3 →
  share_fraction = 1/4 →
  (share_fraction * total_lollipops : ℚ) * (total_cost / total_lollipops) * 100 = 75 := by
  sorry

#check lollipop_reimbursement

end lollipop_reimbursement_l3668_366838


namespace smallest_period_scaled_l3668_366859

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x - p) = f x

theorem smallest_period_scaled (f : ℝ → ℝ) (h : is_periodic f 15) :
  ∃ a : ℝ, a > 0 ∧ (∀ x, f ((x - a) / 3) = f (x / 3)) ∧
    ∀ b, b > 0 → (∀ x, f ((x - b) / 3) = f (x / 3)) → a ≤ b :=
  sorry

end smallest_period_scaled_l3668_366859


namespace empty_pencil_cases_l3668_366821

theorem empty_pencil_cases (total : ℕ) (pencils : ℕ) (pens : ℕ) (both : ℕ) (empty : ℕ) : 
  total = 10 ∧ pencils = 5 ∧ pens = 4 ∧ both = 2 → empty = 3 :=
by
  sorry

end empty_pencil_cases_l3668_366821


namespace union_A_complement_B_range_of_a_l3668_366889

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}
def C (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 3}

-- Theorem for part (1)
theorem union_A_complement_B : A ∪ (Set.univ \ B) = {x : ℝ | x < 4} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) (h : A ∩ C a = A) : 1 ≤ a ∧ a ≤ 3 := by sorry

end union_A_complement_B_range_of_a_l3668_366889


namespace curve_S_properties_l3668_366846

-- Define the curve S
def S (x : ℝ) : ℝ := x^3 - 6*x^2 - x + 6

-- Define the derivative of S
def S' (x : ℝ) : ℝ := 3*x^2 - 12*x - 1

-- Define the point P
def P : ℝ × ℝ := (2, -12)

theorem curve_S_properties :
  -- 1. P is the point where the tangent line has the smallest slope
  (∀ x : ℝ, S' P.1 ≤ S' x) ∧
  -- 2. S is symmetric about P
  (∀ x : ℝ, S (P.1 + x) - P.2 = -(S (P.1 - x) - P.2)) :=
sorry

end curve_S_properties_l3668_366846


namespace average_equation_solution_l3668_366806

theorem average_equation_solution (x : ℝ) : 
  (1/4 : ℝ) * ((x + 8) + (7*x - 3) + (3*x + 10) + (-x + 6)) = 5*x - 4 → x = 3.7 := by
  sorry

end average_equation_solution_l3668_366806


namespace distinct_power_differences_exist_l3668_366833

theorem distinct_power_differences_exist : ∃ (N : ℕ) (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ),
  (∃ (x₁ x₂ : ℕ), a₁ = x₁^2 ∧ a₂ = x₂^2) ∧
  (∃ (y₁ y₂ : ℕ), b₁ = y₁^3 ∧ b₂ = y₂^3) ∧
  (∃ (z₁ z₂ : ℕ), c₁ = z₁^5 ∧ c₂ = z₂^5) ∧
  (∃ (w₁ w₂ : ℕ), d₁ = w₁^7 ∧ d₂ = w₂^7) ∧
  N = a₁ - a₂ ∧
  N = b₁ - b₂ ∧
  N = c₁ - c₂ ∧
  N = d₁ - d₂ ∧
  a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ c₁ ≠ d₁ :=
by sorry

end distinct_power_differences_exist_l3668_366833


namespace age_ratio_l3668_366878

/-- Given that Billy is 4 years old and you were 12 years older than Billy when he was born,
    prove that the ratio of your current age to Billy's current age is 4:1. -/
theorem age_ratio (billy_age : ℕ) (age_difference : ℕ) : 
  billy_age = 4 → age_difference = 12 → (age_difference + billy_age) / billy_age = 4 := by
  sorry

end age_ratio_l3668_366878


namespace cubic_inequality_solution_l3668_366801

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 9*x^2 > -27*x ↔ (0 < x ∧ x < 3) ∨ (6 < x) :=
sorry

end cubic_inequality_solution_l3668_366801


namespace race_speed_ratio_l3668_366852

/-- 
Given two runners a and b, where:
- a's speed is some multiple of b's speed
- a gives b a head start of 1/16 of the race length
- They finish at the same time (dead heat)
Then the ratio of a's speed to b's speed is 15/16
-/
theorem race_speed_ratio (v_a v_b : ℝ) (h : v_a > 0 ∧ v_b > 0) :
  (∃ k : ℝ, v_a = k * v_b) →
  (v_a * 1 = v_b * (15/16)) →
  v_a / v_b = 15/16 := by
sorry

end race_speed_ratio_l3668_366852


namespace school_boys_count_l3668_366884

/-- The percentage of boys who are Muslims -/
def muslim_percentage : ℚ := 44 / 100

/-- The percentage of boys who are Hindus -/
def hindu_percentage : ℚ := 28 / 100

/-- The percentage of boys who are Sikhs -/
def sikh_percentage : ℚ := 10 / 100

/-- The number of boys belonging to other communities -/
def other_communities : ℕ := 153

/-- The total number of boys in the school -/
def total_boys : ℕ := 850

theorem school_boys_count :
  (1 - (muslim_percentage + hindu_percentage + sikh_percentage)) * (total_boys : ℚ) = other_communities := by
  sorry

end school_boys_count_l3668_366884


namespace polyhedron_edge_vertex_relation_l3668_366855

/-- Represents a polyhedron with its vertex and edge properties -/
structure Polyhedron where
  /-- p k is the number of vertices where k edges meet -/
  p : ℕ → ℕ
  /-- a is the total number of edges -/
  a : ℕ

/-- The sum of k * p k for all k ≥ 3 equals twice the total number of edges -/
theorem polyhedron_edge_vertex_relation (P : Polyhedron) :
  2 * P.a = ∑' k, k * P.p k := by sorry

end polyhedron_edge_vertex_relation_l3668_366855


namespace computer_profit_profit_function_max_profit_l3668_366856

/-- Profit from selling computers -/
theorem computer_profit (profit_A profit_B : ℚ) : 
  (10 * profit_A + 20 * profit_B = 4000) →
  (20 * profit_A + 10 * profit_B = 3500) →
  (profit_A = 100 ∧ profit_B = 150) :=
sorry

/-- Functional relationship between total profit and number of type A computers -/
theorem profit_function (x y : ℚ) :
  (x ≥ 0 ∧ x ≤ 100) →
  (y = 100 * x + 150 * (100 - x)) →
  (y = -50 * x + 15000) :=
sorry

/-- Maximum profit when purchasing at least 20 units of type A -/
theorem max_profit (x y : ℚ) :
  (x ≥ 20 ∧ x ≤ 100) →
  (y = -50 * x + 15000) →
  (∀ z, z ≥ 20 ∧ z ≤ 100 → -50 * z + 15000 ≤ 14000) :=
sorry

end computer_profit_profit_function_max_profit_l3668_366856


namespace product_remainder_ten_l3668_366814

theorem product_remainder_ten (a b c : ℕ) (ha : a = 2153) (hb : b = 3491) (hc : c = 925) :
  (a * b * c) % 10 = 5 := by
  sorry

end product_remainder_ten_l3668_366814


namespace triangle_area_l3668_366857

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 5, b = 4, and cos(A - B) = 31/32, then the area of the triangle is (15 * √7) / 4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  a = 5 →
  b = 4 →
  Real.cos (A - B) = 31/32 →
  (1/2) * a * b * Real.sin C = (15 * Real.sqrt 7) / 4 := by
sorry

end triangle_area_l3668_366857


namespace regular_polyhedra_coloring_l3668_366845

structure RegularPolyhedron where
  edges : ℕ
  vertexDegree : ℕ
  faceEdges : ℕ

def isGoodColoring (p : RegularPolyhedron) (redEdges : ℕ) : Prop :=
  redEdges ≤ p.edges * (p.vertexDegree - 1) / p.vertexDegree

def isCompletelyGoodColoring (p : RegularPolyhedron) (redEdges : ℕ) : Prop :=
  isGoodColoring p redEdges ∧ redEdges < p.edges

def maxGoodColoring (p : RegularPolyhedron) : ℕ :=
  p.edges * (p.vertexDegree - 1) / p.vertexDegree

def maxCompletelyGoodColoring (p : RegularPolyhedron) : ℕ :=
  min (maxGoodColoring p) (p.edges - 1)

def tetrahedron : RegularPolyhedron := ⟨6, 3, 3⟩
def cube : RegularPolyhedron := ⟨12, 3, 4⟩
def octahedron : RegularPolyhedron := ⟨12, 4, 3⟩
def dodecahedron : RegularPolyhedron := ⟨30, 3, 5⟩
def icosahedron : RegularPolyhedron := ⟨30, 5, 3⟩

theorem regular_polyhedra_coloring :
  (maxGoodColoring tetrahedron = maxCompletelyGoodColoring tetrahedron) ∧
  (maxGoodColoring cube = maxCompletelyGoodColoring cube) ∧
  (maxGoodColoring dodecahedron = maxCompletelyGoodColoring dodecahedron) ∧
  (maxGoodColoring octahedron ≠ maxCompletelyGoodColoring octahedron) ∧
  (maxGoodColoring icosahedron ≠ maxCompletelyGoodColoring icosahedron) := by
  sorry

end regular_polyhedra_coloring_l3668_366845


namespace average_of_numbers_l3668_366843

def numbers : List ℝ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1252140, 2345]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : ℝ) = 125831.9 := by sorry

end average_of_numbers_l3668_366843


namespace lcm_of_5_8_12_20_l3668_366822

theorem lcm_of_5_8_12_20 : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 12 20)) = 120 := by
  sorry

end lcm_of_5_8_12_20_l3668_366822


namespace digit_sum_proof_l3668_366893

/-- Represents the number of '1's in the original number -/
def num_ones : ℕ := 2018

/-- Represents the number of '5's in the original number -/
def num_fives : ℕ := 2017

/-- Represents the original number under the square root -/
def original_number : ℕ :=
  (10^num_ones - 1) / 9 * 10^(num_fives + 1) + 
  5 * (10^num_fives - 1) / 9 * 10^num_ones + 
  6

/-- The sum of digits in the decimal representation of the integer part 
    of the square root of the original number -/
def digit_sum : ℕ := num_ones * 3 + 4

theorem digit_sum_proof : digit_sum = 6055 := by
  sorry

end digit_sum_proof_l3668_366893


namespace train_passing_tree_l3668_366809

/-- Proves that a train 280 meters long, traveling at 72 km/hr, will take 14 seconds to pass a tree. -/
theorem train_passing_tree (train_length : ℝ) (train_speed_kmh : ℝ) (time : ℝ) :
  train_length = 280 ∧ 
  train_speed_kmh = 72 →
  time = train_length / (train_speed_kmh * (5/18)) ∧ 
  time = 14 := by
  sorry

end train_passing_tree_l3668_366809


namespace expand_product_l3668_366847

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 := by
  sorry

end expand_product_l3668_366847


namespace xyz_value_l3668_366819

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 10)
  (eq5 : x + y + z = 6) :
  x * y * z = 14 := by
sorry

end xyz_value_l3668_366819


namespace right_triangle_sets_l3668_366829

theorem right_triangle_sets : 
  (¬ (4^2 + 6^2 = 8^2)) ∧ 
  (5^2 + 12^2 = 13^2) ∧ 
  (6^2 + 8^2 = 10^2) ∧ 
  (7^2 + 24^2 = 25^2) := by
  sorry

end right_triangle_sets_l3668_366829


namespace particle_max_height_l3668_366891

/-- The height function of the particle -/
def h (t : ℝ) : ℝ := 180 * t - 18 * t^2

/-- The maximum height reached by the particle -/
def max_height : ℝ := 450

/-- Theorem stating that the maximum height reached by the particle is 450 meters -/
theorem particle_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ h t :=
sorry

end particle_max_height_l3668_366891


namespace value_of_a_l3668_366854

theorem value_of_a (U : Set ℝ) (A : Set ℝ) (a : ℝ) : 
  U = {2, 3, a^2 - a - 1} →
  A = {2, 3} →
  U \ A = {1} →
  a = -1 ∨ a = 2 :=
by sorry

end value_of_a_l3668_366854


namespace room_tiles_count_l3668_366828

/-- Calculates the least number of square tiles required to cover a rectangular floor. -/
def leastSquareTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  (length / tileSize) * (width / tileSize)

/-- Theorem stating that for a room with given dimensions, 153 square tiles are required. -/
theorem room_tiles_count :
  leastSquareTiles 816 432 = 153 := by
  sorry

#eval leastSquareTiles 816 432

end room_tiles_count_l3668_366828


namespace sqrt_product_simplification_l3668_366877

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by sorry

end sqrt_product_simplification_l3668_366877


namespace proposition_2_proposition_3_l3668_366830

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)

-- Define the given conditions
variable (m n a b : Line) (α β : Plane)
variable (h_mn_distinct : m ≠ n)
variable (h_αβ_distinct : α ≠ β)
variable (h_a_perp_α : perpendicularLP a α)
variable (h_b_perp_β : perpendicularLP b β)

-- State the theorems to be proved
theorem proposition_2 
  (h_m_parallel_a : parallel m a)
  (h_n_parallel_b : parallel n b)
  (h_α_perp_β : perpendicularPP α β) :
  perpendicular m n :=
sorry

theorem proposition_3
  (h_m_parallel_α : parallelLP m α)
  (h_n_parallel_b : parallel n b)
  (h_α_parallel_β : parallelPP α β) :
  perpendicular m n :=
sorry

end proposition_2_proposition_3_l3668_366830


namespace josephs_total_cards_l3668_366826

/-- The number of cards in a standard deck -/
def cards_per_deck : ℕ := 52

/-- The number of decks Joseph has -/
def josephs_decks : ℕ := 4

/-- Theorem: Joseph has 208 cards in total -/
theorem josephs_total_cards : 
  josephs_decks * cards_per_deck = 208 := by
  sorry

end josephs_total_cards_l3668_366826


namespace nested_fraction_evaluation_l3668_366865

theorem nested_fraction_evaluation :
  1 / (3 - 1 / (2 - 1 / (3 - 1 / (2 - 1 / 2)))) = 11 / 26 := by
  sorry

end nested_fraction_evaluation_l3668_366865


namespace craig_apples_l3668_366879

/-- Theorem: If Craig shares 7 apples and has 13 apples left after sharing,
    then Craig initially had 20 apples. -/
theorem craig_apples (initial : ℕ) (shared : ℕ) (remaining : ℕ)
    (h1 : shared = 7)
    (h2 : remaining = 13)
    (h3 : initial = shared + remaining) :
  initial = 20 := by
  sorry

end craig_apples_l3668_366879


namespace rectangle_area_comparison_l3668_366890

theorem rectangle_area_comparison 
  (A B C D A' B' C' D' : ℝ) 
  (hA : 0 ≤ A) (hB : 0 ≤ B) (hC : 0 ≤ C) (hD : 0 ≤ D)
  (hA' : 0 ≤ A') (hB' : 0 ≤ B') (hC' : 0 ≤ C') (hD' : 0 ≤ D')
  (hAA' : A ≤ A') (hBB' : B ≤ B') (hCC' : C ≤ C') (hDB' : D ≤ B') :
  A + B + C + D ≤ A' + B' + C' + D' := by
  sorry

end rectangle_area_comparison_l3668_366890


namespace investment_value_proof_l3668_366861

theorem investment_value_proof (x : ℝ) : 
  (0.07 * x + 0.11 * 1500 = 0.10 * (x + 1500)) → x = 500 := by
  sorry

end investment_value_proof_l3668_366861


namespace cody_marbles_l3668_366876

def initial_marbles : ℕ := 12
def marbles_given : ℕ := 5

theorem cody_marbles : initial_marbles - marbles_given = 7 := by
  sorry

end cody_marbles_l3668_366876


namespace range_of_m_l3668_366802

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -Real.sqrt (9 - p.1^2)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2}

-- Define the point A
def A (m : ℝ) : ℝ × ℝ := (0, m)

-- Define the vector from A to P
def AP (m : ℝ) (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - 0, P.2 - m)

-- Define the vector from A to Q
def AQ (m : ℝ) (Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - 0, Q.2 - m)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∃ P ∈ C, ∃ Q ∈ l, AP m P + AQ m Q = (0, 0)) → m ∈ Set.Icc (-1/2) 1 :=
sorry

end range_of_m_l3668_366802


namespace project_assignment_count_l3668_366824

/-- The number of ways to assign projects to teams --/
def assign_projects (total_projects : ℕ) (num_teams : ℕ) (max_for_one_team : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of assignments for the given conditions --/
theorem project_assignment_count :
  assign_projects 5 3 2 = 130 :=
sorry

end project_assignment_count_l3668_366824


namespace only_paintable_number_l3668_366858

/-- Represents a painting configuration for the railings. -/
structure PaintConfig where
  h : ℕ+  -- Harold's interval
  t : ℕ+  -- Tanya's interval
  u : ℕ+  -- Ulysses' interval

/-- Checks if a given railing number is painted by Harold. -/
def paintedByHarold (config : PaintConfig) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 1 + k * config.h

/-- Checks if a given railing number is painted by Tanya. -/
def paintedByTanya (config : PaintConfig) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 + k * config.t

/-- Checks if a given railing number is painted by Ulysses. -/
def paintedByUlysses (config : PaintConfig) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 + k * config.u

/-- Checks if every railing is painted exactly once. -/
def validPainting (config : PaintConfig) : Prop :=
  ∀ n : ℕ+, (paintedByHarold config n ∨ paintedByTanya config n ∨ paintedByUlysses config n) ∧
            ¬(paintedByHarold config n ∧ paintedByTanya config n) ∧
            ¬(paintedByHarold config n ∧ paintedByUlysses config n) ∧
            ¬(paintedByTanya config n ∧ paintedByUlysses config n)

/-- Computes the paintable number for a given configuration. -/
def paintableNumber (config : PaintConfig) : ℕ :=
  100 * config.h + 10 * config.t + config.u

/-- Theorem stating that 453 is the only paintable number. -/
theorem only_paintable_number :
  ∃! n : ℕ, n = 453 ∧ ∃ config : PaintConfig, validPainting config ∧ paintableNumber config = n :=
sorry

end only_paintable_number_l3668_366858


namespace product_543_7_base9_l3668_366841

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (n : ℕ) : ℕ :=
  sorry

/-- Converts a base-10 number to base-9 --/
def base10ToBase9 (n : ℕ) : ℕ :=
  sorry

/-- Multiplies two base-9 numbers and returns the result in base-9 --/
def multiplyBase9 (a b : ℕ) : ℕ :=
  base10ToBase9 (base9ToBase10 a * base9ToBase10 b)

theorem product_543_7_base9 :
  multiplyBase9 543 7 = 42333 :=
sorry

end product_543_7_base9_l3668_366841


namespace erased_numbers_theorem_l3668_366827

def sumBetween (a b : ℕ) : ℕ := (b - a - 1) * (a + b) / 2

def sumOutside (a b : ℕ) : ℕ := (2018 * 2019) / 2 - sumBetween a b - a - b

theorem erased_numbers_theorem (a b : ℕ) (ha : a = 673) (hb : b = 1346) :
  2 * sumBetween a b = sumOutside a b := by
  sorry

end erased_numbers_theorem_l3668_366827


namespace farmer_seeds_sowed_l3668_366848

/-- The number of buckets of seeds sowed by a farmer -/
def seeds_sowed (initial final : ℝ) : ℝ := initial - final

/-- Theorem stating that the farmer sowed 2.75 buckets of seeds -/
theorem farmer_seeds_sowed :
  seeds_sowed 8.75 6 = 2.75 := by
  sorry

end farmer_seeds_sowed_l3668_366848


namespace clothing_discount_l3668_366817

theorem clothing_discount (original_price : ℝ) (first_sale_price second_sale_price : ℝ) :
  first_sale_price = (4 / 5) * original_price →
  second_sale_price = (1 - 0.4) * first_sale_price →
  second_sale_price = (12 / 25) * original_price :=
by sorry

end clothing_discount_l3668_366817


namespace unique_polynomial_composition_l3668_366832

/-- The polynomial P(x) = x^2 - x satisfies P(P(x)) = (x^2 - x + 1) P(x) and is the only nonconstant polynomial solution. -/
theorem unique_polynomial_composition (x : ℝ) : ∃! P : ℝ → ℝ, 
  (∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) ∧ 
  (a ≠ 0 ∨ b ≠ 0) ∧
  (∀ x, P (P x) = (x^2 - x + 1) * P x) ∧
  P = fun x ↦ x^2 - x := by
  sorry

end unique_polynomial_composition_l3668_366832


namespace fraction_division_evaluate_fraction_l3668_366836

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  a / (c / d) = (a * d) / c :=
by sorry

theorem evaluate_fraction :
  (4 : ℚ) / ((8 : ℚ) / 13) = 13 / 2 :=
by sorry

end fraction_division_evaluate_fraction_l3668_366836


namespace line_passes_through_parabola_vertex_l3668_366872

/-- The number of values of b for which the line y = 2x + b passes through
    the vertex of the parabola y = x^2 - 2bx + b^2 is exactly 1. -/
theorem line_passes_through_parabola_vertex :
  ∃! b : ℝ, ∀ x y : ℝ,
    (y = 2 * x + b) ∧ (y = x^2 - 2 * b * x + b^2) →
    (x = b ∧ y = 0) :=
by sorry

end line_passes_through_parabola_vertex_l3668_366872


namespace intersection_A_C_R_B_range_of_a_l3668_366892

-- Define sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- Define the relative complement of B with respect to ℝ
def C_R_B : Set ℝ := {x | x ∉ B}

-- Theorem 1: A ∩ (C_R B) = {x | -3 < x ≤ 2}
theorem intersection_A_C_R_B : A ∩ C_R_B = {x : ℝ | -3 < x ∧ x ≤ 2} := by
  sorry

-- Theorem 2: If C ⊇ (A ∩ B), then 4/3 ≤ a ≤ 2
theorem range_of_a (a : ℝ) (h : a ≠ 0) :
  (C a ⊇ (A ∩ B)) → (4/3 ≤ a ∧ a ≤ 2) := by
  sorry

end intersection_A_C_R_B_range_of_a_l3668_366892


namespace vectors_not_coplanar_l3668_366871

def a : Fin 3 → ℝ := ![1, 5, 2]
def b : Fin 3 → ℝ := ![-1, 1, -1]
def c : Fin 3 → ℝ := ![1, 1, 1]

theorem vectors_not_coplanar : ¬(∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) := by
  sorry

end vectors_not_coplanar_l3668_366871


namespace typing_service_cost_l3668_366867

/-- Typing service cost calculation -/
theorem typing_service_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) 
  (revision_cost : ℚ) (total_cost : ℚ) :
  total_pages = 100 →
  revised_once = 20 →
  revised_twice = 30 →
  revision_cost = 5 →
  total_cost = 1400 →
  ∃ (first_time_cost : ℚ),
    first_time_cost * total_pages + 
    revision_cost * (revised_once + 2 * revised_twice) = total_cost ∧
    first_time_cost = 10 := by
  sorry


end typing_service_cost_l3668_366867


namespace locus_of_intersection_points_l3668_366863

/-- The locus of intersection points of perpendiculars drawn from a circle's 
    intersections with two perpendicular lines. -/
theorem locus_of_intersection_points (u v x y : ℝ) :
  (u ≠ v ∨ u ≠ -v) →
  (∃ (r : ℝ), r > 0 ∧ r > |u| ∧ r > |v| ∧
    (x - u)^2 / (u^2 - v^2) - (y - v)^2 / (u^2 - v^2) = 1) ∨
  (u = v ∨ u = -v) →
    (x - y) * (x + y) = 0 :=
sorry

end locus_of_intersection_points_l3668_366863


namespace sum_of_number_and_its_square_l3668_366825

theorem sum_of_number_and_its_square : 
  let n : ℕ := 8
  (n + n^2) = 72 := by sorry

end sum_of_number_and_its_square_l3668_366825


namespace sqrt_4_minus_x_real_range_l3668_366869

theorem sqrt_4_minus_x_real_range : 
  {x : ℝ | ∃ y : ℝ, y ^ 2 = 4 - x} = {x : ℝ | x ≤ 4} := by
sorry

end sqrt_4_minus_x_real_range_l3668_366869


namespace product_even_sum_undetermined_l3668_366803

theorem product_even_sum_undetermined (a b : ℤ) : 
  Even (a * b) → (Even (a + b) ∨ Odd (a + b)) :=
by sorry

end product_even_sum_undetermined_l3668_366803


namespace mountain_loop_trail_length_l3668_366899

/-- Represents the Mountain Loop Trail hike --/
structure MountainLoopTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hike --/
def validHike (hike : MountainLoopTrail) : Prop :=
  hike.day1 + hike.day2 = 28 ∧
  (hike.day2 + hike.day3) / 2 = 14 ∧
  hike.day4 + hike.day5 = 36 ∧
  hike.day1 + hike.day3 = 30

/-- The theorem stating the total length of the trail --/
theorem mountain_loop_trail_length (hike : MountainLoopTrail) 
  (h : validHike hike) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 94 := by
  sorry


end mountain_loop_trail_length_l3668_366899


namespace fraction_evaluation_l3668_366886

theorem fraction_evaluation : (4 - 3/5) / (3 - 2/7) = 119/95 := by
  sorry

end fraction_evaluation_l3668_366886


namespace equation_solution_l3668_366840

theorem equation_solution : ∃ x : ℝ, 24 - 6 = 3 + x ∧ x = 15 := by sorry

end equation_solution_l3668_366840


namespace compound_proposition_negation_l3668_366862

theorem compound_proposition_negation (p q : Prop) : 
  ¬((p ∧ q → false) → (¬p → false) ∧ (¬q → false)) := by
  sorry

end compound_proposition_negation_l3668_366862


namespace intersection_range_l3668_366875

-- Define the line equation
def line_equation (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the condition for intersection
def intersects_hyperbola (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  hyperbola_equation x₁ (line_equation k x₁) ∧
  hyperbola_equation x₂ (line_equation k x₂) ∧
  x₁ * x₂ < 0  -- Ensures points are on different branches

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, intersects_hyperbola k ↔ -1 < k ∧ k < 1 :=
sorry

end intersection_range_l3668_366875


namespace consecutive_integers_product_plus_one_is_square_l3668_366874

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  ∃ m : ℤ, n * (n + 1) * (n + 2) * (n + 3) + 1 = m^2 := by
sorry

end consecutive_integers_product_plus_one_is_square_l3668_366874


namespace triangle_side_count_is_35_l3668_366812

/-- The number of integer values for the third side of a triangle with two sides of length 18 and 45 -/
def triangle_side_count : ℕ :=
  let possible_x := Finset.filter (fun x : ℕ =>
    x > 27 ∧ x < 63 ∧ x + 18 > 45 ∧ x + 45 > 18 ∧ 18 + 45 > x) (Finset.range 100)
  possible_x.card

theorem triangle_side_count_is_35 : triangle_side_count = 35 := by
  sorry

end triangle_side_count_is_35_l3668_366812


namespace student_bicycle_speed_l3668_366853

theorem student_bicycle_speed
  (distance : ℝ)
  (speed_ratio : ℝ)
  (time_difference : ℝ)
  (h_distance : distance = 12)
  (h_speed_ratio : speed_ratio = 1.2)
  (h_time_difference : time_difference = 1/6) :
  ∃ (speed_B : ℝ), speed_B = 12 ∧
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference :=
by sorry

end student_bicycle_speed_l3668_366853


namespace integer_sum_problem_l3668_366820

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 12 → x * y = 45 → x + y = 18 := by sorry

end integer_sum_problem_l3668_366820


namespace quadratic_polynomial_with_complex_root_l3668_366881

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3 ∧ 
     b = -12 ∧ 
     c = 24) ∧
    (Complex.I : ℂ)^2 = -1 ∧
    (a * (2 + 2 * Complex.I)^2 + b * (2 + 2 * Complex.I) + c = 0) :=
by sorry

end quadratic_polynomial_with_complex_root_l3668_366881


namespace path_count_on_grid_l3668_366831

/-- The number of distinct paths on a 6x5 grid from upper left to lower right corner -/
def number_of_paths : ℕ := 126

/-- The number of right moves required to reach the right edge of a 6x5 grid -/
def right_moves : ℕ := 5

/-- The number of down moves required to reach the bottom edge of a 6x5 grid -/
def down_moves : ℕ := 4

/-- The total number of moves (right + down) required to reach the bottom right corner -/
def total_moves : ℕ := right_moves + down_moves

theorem path_count_on_grid : 
  number_of_paths = Nat.choose total_moves right_moves :=
by sorry

end path_count_on_grid_l3668_366831


namespace investment_problem_l3668_366860

theorem investment_problem (x y : ℝ) (h1 : x + y = 3000) 
  (h2 : 0.08 * x + 0.05 * y = 490 ∨ 0.08 * y + 0.05 * x = 490) : 
  x + y = 8000 := by
sorry

end investment_problem_l3668_366860


namespace counterfeit_coins_l3668_366835

def bags : List Nat := [18, 19, 21, 23, 25, 34]

structure Distribution where
  xiaocong : List Nat
  xiaomin : List Nat
  counterfeit : Nat

def isValidDistribution (d : Distribution) : Prop :=
  d.xiaocong.length = 3 ∧
  d.xiaomin.length = 2 ∧
  d.xiaocong.sum = 2 * d.xiaomin.sum ∧
  d.counterfeit ∈ bags ∧
  d.xiaocong.sum + d.xiaomin.sum + d.counterfeit = bags.sum

theorem counterfeit_coins (d : Distribution) :
  isValidDistribution d → d.counterfeit = 23 := by
  sorry

end counterfeit_coins_l3668_366835


namespace championship_outcomes_l3668_366844

def number_of_competitors : ℕ := 5
def number_of_events : ℕ := 3

theorem championship_outcomes :
  (number_of_competitors ^ number_of_events : ℕ) = 125 := by
  sorry

end championship_outcomes_l3668_366844


namespace apple_distribution_l3668_366866

/-- The number of ways to distribute n items among k people with a minimum of m items each -/
def distribution_ways (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- Theorem: There are 253 ways to distribute 30 apples among 3 people with at least 3 apples each -/
theorem apple_distribution :
  distribution_ways 30 3 3 = 253 := by
  sorry

end apple_distribution_l3668_366866


namespace solution_problem_l3668_366895

theorem solution_problem (x y : ℕ) 
  (h1 : 0 < x ∧ x < 30) 
  (h2 : 0 < y ∧ y < 30) 
  (h3 : x + y + x * y = 104) : 
  x + y = 20 := by
sorry

end solution_problem_l3668_366895


namespace fruit_punch_water_amount_l3668_366804

/-- Represents the ratio of ingredients in the fruit punch -/
structure PunchRatio :=
  (water : ℚ)
  (orange_juice : ℚ)
  (cranberry_juice : ℚ)

/-- Calculates the amount of water needed for a given amount of punch -/
def water_needed (ratio : PunchRatio) (total_gallons : ℚ) (quarts_per_gallon : ℚ) : ℚ :=
  let total_parts := ratio.water + ratio.orange_juice + ratio.cranberry_juice
  let water_fraction := ratio.water / total_parts
  water_fraction * total_gallons * quarts_per_gallon

/-- Theorem stating the amount of water needed for the fruit punch -/
theorem fruit_punch_water_amount :
  let ratio : PunchRatio := ⟨5, 2, 1⟩
  let total_gallons : ℚ := 3
  let quarts_per_gallon : ℚ := 4
  water_needed ratio total_gallons quarts_per_gallon = 15 / 2 := by
  sorry

end fruit_punch_water_amount_l3668_366804


namespace quadratic_equation_solution_l3668_366887

theorem quadratic_equation_solution (x m k : ℝ) :
  (x + m) * (x - 5) = x^2 - 3*x + k →
  k = -10 ∧ m = 2 := by
  sorry

end quadratic_equation_solution_l3668_366887


namespace line_equation_intersection_condition_max_value_condition_l3668_366807

-- Define the parabola and line
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 1
def line (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Theorem 1: Line equation
theorem line_equation :
  ∃ k b : ℝ, (line k b (-2) = -5/2) ∧ (line k b 3 = 0) →
  (k = 1/2 ∧ b = -3/2) :=
sorry

-- Theorem 2: Intersection condition
theorem intersection_condition (a : ℝ) :
  a ≠ 0 →
  (∃ x : ℝ, parabola a x = line (1/2) (-3/2) x) ↔
  (a ≤ 9/8) :=
sorry

-- Theorem 3: Maximum value condition
theorem max_value_condition :
  ∃ m : ℝ, (∀ x : ℝ, m ≤ x ∧ x ≤ m + 2 →
    parabola (-1) x ≤ -4) ∧
    (∃ x : ℝ, m ≤ x ∧ x ≤ m + 2 ∧ parabola (-1) x = -4) →
  (m = -3 ∨ m = 3) :=
sorry

end line_equation_intersection_condition_max_value_condition_l3668_366807


namespace negation_of_proposition_l3668_366805

theorem negation_of_proposition :
  (¬ ∀ a : ℕ+, 2^(a : ℕ) ≥ (a : ℕ)^2) ↔ (∃ a : ℕ+, 2^(a : ℕ) < (a : ℕ)^2) :=
by sorry

end negation_of_proposition_l3668_366805


namespace jack_apples_l3668_366823

/-- Calculates the remaining apples after a series of sales and a gift --/
def remaining_apples (initial : ℕ) (sale1_percent : ℕ) (sale2_percent : ℕ) (sale3_percent : ℕ) (gift : ℕ) : ℕ :=
  let after_sale1 := initial - initial * sale1_percent / 100
  let after_sale2 := after_sale1 - after_sale1 * sale2_percent / 100
  let after_sale3 := after_sale2 - (after_sale2 * sale3_percent / 100)
  after_sale3 - gift

/-- Theorem stating that given the specific conditions, Jack ends up with 75 apples --/
theorem jack_apples : remaining_apples 150 30 20 10 1 = 75 := by
  sorry

end jack_apples_l3668_366823


namespace three_intersections_iff_a_in_open_interval_l3668_366851

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the condition for three distinct intersection points
def has_three_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
  f x₁ = a ∧ f x₂ = a ∧ f x₃ = a

-- Theorem statement
theorem three_intersections_iff_a_in_open_interval :
  ∀ a : ℝ, has_three_intersections a ↔ -2 < a ∧ a < 2 :=
sorry

end three_intersections_iff_a_in_open_interval_l3668_366851


namespace soccer_field_kids_l3668_366868

/-- The number of kids initially on the soccer field -/
def initial_kids : ℕ := 14

/-- The number of kids who decided to join -/
def joining_kids : ℕ := 22

/-- The total number of kids on the soccer field after new kids join -/
def total_kids : ℕ := initial_kids + joining_kids

theorem soccer_field_kids : total_kids = 36 := by
  sorry

end soccer_field_kids_l3668_366868


namespace smallest_number_with_property_l3668_366896

theorem smallest_number_with_property : ∃ n : ℕ, 
  n > 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧ m < n →
    m % 2 ≠ 1 ∨
    m % 3 ≠ 2 ∨
    m % 4 ≠ 3 ∨
    m % 5 ≠ 4 ∨
    m % 6 ≠ 5 ∨
    m % 7 ≠ 6 ∨
    m % 8 ≠ 7 ∨
    m % 9 ≠ 8 ∨
    m % 10 ≠ 9) ∧
  n = 2519 :=
by sorry

end smallest_number_with_property_l3668_366896


namespace triangle_angle_sum_l3668_366873

theorem triangle_angle_sum (A B C : ℝ) (h : A + B = 80) : C = 100 :=
  by
  sorry

end triangle_angle_sum_l3668_366873


namespace remainder_theorem_l3668_366870

theorem remainder_theorem (N : ℤ) (h : N % 100 = 11) : N % 11 = 0 := by
  sorry

end remainder_theorem_l3668_366870


namespace problem_solution_l3668_366818

theorem problem_solution : 
  let a := (5 / 6) * 180
  let b := 0.70 * 250
  let diff := a - b
  let c := 0.35 * 480
  diff / c = -0.14880952381 := by
sorry

end problem_solution_l3668_366818


namespace debate_team_arrangements_l3668_366894

/-- Represents the debate team composition -/
structure DebateTeam :=
  (male_count : Nat)
  (female_count : Nat)

/-- The number of arrangements where no two male members are adjacent -/
def non_adjacent_male_arrangements (team : DebateTeam) : Nat :=
  sorry

/-- The number of ways to divide the team into four groups of two and assign them to four classes -/
def seminar_groupings (team : DebateTeam) : Nat :=
  sorry

/-- The number of ways to select 4 members (with at least one male) and assign them to four speaker roles -/
def speaker_selections (team : DebateTeam) : Nat :=
  sorry

theorem debate_team_arrangements (team : DebateTeam) 
  (h1 : team.male_count = 3) 
  (h2 : team.female_count = 5) : 
  non_adjacent_male_arrangements team = 14400 ∧ 
  seminar_groupings team = 2520 ∧ 
  speaker_selections team = 1560 :=
sorry

end debate_team_arrangements_l3668_366894


namespace bucket_capacity_l3668_366800

/-- The capacity of a bucket in litres, given that when it is 2/3 full, it contains 9 litres of maple syrup. -/
theorem bucket_capacity : ℝ := by
  -- Define the capacity of the bucket
  let C : ℝ := 13.5

  -- Define the fraction of the bucket that is full
  let fraction_full : ℝ := 2/3

  -- Define the current volume of maple syrup
  let current_volume : ℝ := 9

  -- State that the current volume is equal to the fraction of the capacity
  have h1 : fraction_full * C = current_volume := by sorry

  -- Prove that the capacity is indeed 13.5 litres
  have h2 : C = 13.5 := by sorry

  -- Return the capacity
  exact C

end bucket_capacity_l3668_366800


namespace crackers_box_sleeves_l3668_366880

/-- The number of crackers Chad uses per sandwich -/
def crackers_per_sandwich : ℕ := 2

/-- The number of sandwiches Chad eats per night -/
def sandwiches_per_night : ℕ := 5

/-- The number of crackers in each sleeve -/
def crackers_per_sleeve : ℕ := 28

/-- The number of boxes of crackers -/
def num_boxes : ℕ := 5

/-- The number of nights the crackers last -/
def num_nights : ℕ := 56

/-- The number of sleeves in a box of crackers -/
def sleeves_per_box : ℕ := 4

theorem crackers_box_sleeves :
  sleeves_per_box = 4 :=
sorry

end crackers_box_sleeves_l3668_366880


namespace store_max_profit_l3668_366839

/-- A store selling clothing with the following conditions:
    - Cost price is 60 yuan per item
    - Selling price must not be lower than the cost price
    - Profit must not exceed 40%
    - Sales volume (y) and selling price (x) follow a linear function y = kx + b
    - When x = 80, y = 40
    - When x = 70, y = 50
    - 60 ≤ x ≤ 84
-/
theorem store_max_profit (x : ℝ) (y : ℝ) (k : ℝ) (b : ℝ) (W : ℝ → ℝ) :
  (∀ x, 60 ≤ x ∧ x ≤ 84) →
  (∀ x, y = k * x + b) →
  (80 * k + b = 40) →
  (70 * k + b = 50) →
  (∀ x, W x = (x - 60) * (k * x + b)) →
  (∃ x₀, ∀ x, 60 ≤ x ∧ x ≤ 84 → W x ≤ W x₀) →
  (∃ x₀, W x₀ = 864 ∧ x₀ = 84) := by
  sorry


end store_max_profit_l3668_366839


namespace modulus_z_l3668_366850

/-- Given complex numbers w and z such that wz = 20 - 15i and |w| = √34, prove that |z| = (25√34) / 34 -/
theorem modulus_z (w z : ℂ) (h1 : w * z = 20 - 15 * I) (h2 : Complex.abs w = Real.sqrt 34) :
  Complex.abs z = (25 * Real.sqrt 34) / 34 := by
  sorry

end modulus_z_l3668_366850


namespace collinear_points_a_equals_9_l3668_366842

/-- Three points (x₁, y₁), (x₂, y₂), and (x₃, y₃) are collinear if and only if
    (y₂ - y₁)*(x₃ - x₂) = (y₃ - y₂)*(x₂ - x₁) -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁)*(x₃ - x₂) = (y₃ - y₂)*(x₂ - x₁)

/-- If the points (1, 3), (2, 5), and (4, a) are collinear, then a = 9 -/
theorem collinear_points_a_equals_9 :
  collinear 1 3 2 5 4 a → a = 9 :=
by
  sorry


end collinear_points_a_equals_9_l3668_366842


namespace binomial_12_11_l3668_366883

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end binomial_12_11_l3668_366883


namespace average_trees_planted_l3668_366864

theorem average_trees_planted (trees_A trees_B trees_C : ℕ) : 
  trees_A = 225 →
  trees_B = trees_A + 48 →
  trees_C = trees_A - 24 →
  (trees_A + trees_B + trees_C) / 3 = 233 := by
  sorry

end average_trees_planted_l3668_366864
