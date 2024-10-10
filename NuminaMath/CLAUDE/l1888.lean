import Mathlib

namespace monotone_increasing_condition_l1888_188876

open Real

/-- A function f(x) = kx - ln(x) is monotonically increasing on (1, +∞) if and only if k ≥ 1 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > 1, StrictMono (fun x => k * x - log x)) ↔ k ≥ 1 := by
sorry

end monotone_increasing_condition_l1888_188876


namespace product_of_powers_l1888_188838

theorem product_of_powers : 2^4 * 3^2 * 5^2 * 7 * 11 = 277200 := by
  sorry

end product_of_powers_l1888_188838


namespace decompose_power_l1888_188830

theorem decompose_power (a : ℝ) (h : a > 0) : 
  ∃ (x y z w : ℝ), 
    a^(3/4) = a^x * a^y * a^z * a^w ∧ 
    y = x + 1/6 ∧ 
    z = y + 1/6 ∧ 
    w = z + 1/6 ∧
    x = -1/16 ∧ 
    y = 5/48 ∧ 
    z = 13/48 ∧ 
    w = 7/16 := by
  sorry

end decompose_power_l1888_188830


namespace no_prime_factor_3j_plus_2_l1888_188822

/-- A number is a cube if it's the cube of some integer -/
def IsCube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

/-- The number of divisors of a natural number -/
def NumDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- A number is the smallest with k divisors if it has k divisors and no smaller number has k divisors -/
def IsSmallestWithKDivisors (n k : ℕ) : Prop :=
  NumDivisors n = k ∧ ∀ m < n, NumDivisors m ≠ k

theorem no_prime_factor_3j_plus_2 (n k : ℕ) (h1 : IsSmallestWithKDivisors n k) (h2 : IsCube n) :
  ¬∃ (p : ℕ), Nat.Prime p ∧ (∃ j : ℕ, p = 3*j + 2) ∧ p ∣ k := by
  sorry

end no_prime_factor_3j_plus_2_l1888_188822


namespace relation_implications_l1888_188836

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between propositions
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

def necessary (P Q : Prop) : Prop :=
  Q → P

-- State the theorem
theorem relation_implications :
  sufficient_not_necessary A B →
  necessary B C →
  sufficient_not_necessary C D →
  (sufficient_not_necessary A C ∧ 
   ¬(sufficient_not_necessary D A ∨ necessary D A)) :=
by sorry

end relation_implications_l1888_188836


namespace system_solution_l1888_188865

theorem system_solution : 
  ∀ x y : ℝ, 
    (3 * x + Real.sqrt (3 * x - y) + y = 6 ∧ 
     9 * x^2 + 3 * x - y - y^2 = 36) ↔ 
    ((x = 2 ∧ y = -3) ∨ (x = 6 ∧ y = -18)) := by
  sorry

end system_solution_l1888_188865


namespace duck_ratio_is_two_to_one_l1888_188875

/-- The ratio of ducks at North Pond to Lake Michigan -/
def duck_ratio (north_pond : ℕ) (lake_michigan : ℕ) : ℚ :=
  (north_pond : ℚ) / (lake_michigan : ℚ)

theorem duck_ratio_is_two_to_one :
  let lake_michigan : ℕ := 100
  let north_pond : ℕ := 206
  ∀ R : ℚ, north_pond = lake_michigan * R + 6 →
    duck_ratio north_pond lake_michigan = 2 := by
  sorry

end duck_ratio_is_two_to_one_l1888_188875


namespace fifty_cent_items_count_l1888_188878

/-- Represents the number of items at each price point -/
structure ItemCounts where
  fiftyc : ℕ
  twofifty : ℕ
  four : ℕ

/-- Calculates the total number of items -/
def total_items (c : ItemCounts) : ℕ :=
  c.fiftyc + c.twofifty + c.four

/-- Calculates the total cost in cents -/
def total_cost (c : ItemCounts) : ℕ :=
  50 * c.fiftyc + 250 * c.twofifty + 400 * c.four

/-- The main theorem to prove -/
theorem fifty_cent_items_count (c : ItemCounts) :
  total_items c = 50 ∧ total_cost c = 5000 → c.fiftyc = 40 := by
  sorry

#check fifty_cent_items_count

end fifty_cent_items_count_l1888_188878


namespace polyhedron_vertices_l1888_188890

/-- A polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices --/
structure Polyhedron where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- Euler's formula for polyhedra states that for a polyhedron with F faces, E edges, and V vertices, F - E + V = 2 --/
axiom euler_formula (p : Polyhedron) : p.faces - p.edges + p.vertices = 2

/-- Theorem: A polyhedron with 6 faces and 12 edges has 8 vertices --/
theorem polyhedron_vertices (p : Polyhedron) (h1 : p.faces = 6) (h2 : p.edges = 12) : p.vertices = 8 := by
  sorry

end polyhedron_vertices_l1888_188890


namespace tangent_line_to_polar_curve_l1888_188853

/-- Given a line in polar coordinates ρcos(θ + π/3) = 1 tangent to a curve ρ = r (r > 0),
    prove that r = 1 -/
theorem tangent_line_to_polar_curve (r : ℝ) (h1 : r > 0) : 
  (∃ θ : ℝ, r * Real.cos (θ + π/3) = 1) → r = 1 := by
  sorry

end tangent_line_to_polar_curve_l1888_188853


namespace vector_relation_l1888_188862

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (P A B C : V)

theorem vector_relation (h : (A - P) + 2 • (B - P) + 3 • (C - P) = 0) :
  P - A = (1/3) • (B - A) + (1/2) • (C - A) := by sorry

end vector_relation_l1888_188862


namespace class_size_l1888_188804

theorem class_size (S : ℚ) 
  (basketball : S / 2 = S * (1 / 2))
  (volleyball : S * (2 / 5) = S * (2 / 5))
  (both : S / 10 = S * (1 / 10))
  (neither : S * (1 / 5) = 4) : S = 20 := by
  sorry

end class_size_l1888_188804


namespace f_difference_bound_l1888_188841

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem f_difference_bound (a x : ℝ) (h : |x - a| < 1) : 
  |f x - f a| < 2 * |a| + 3 := by
  sorry

end f_difference_bound_l1888_188841


namespace problem_1_l1888_188872

theorem problem_1 : (1) - 3 + 8 - 15 - 6 = -16 := by
  sorry

end problem_1_l1888_188872


namespace equality_of_M_and_N_l1888_188826

theorem equality_of_M_and_N (a b : ℝ) (hab : a * b = 1) : 
  (1 / (1 + a) + 1 / (1 + b)) = (a / (1 + a) + b / (1 + b)) := by
  sorry

#check equality_of_M_and_N

end equality_of_M_and_N_l1888_188826


namespace infinite_chessboard_rightlines_l1888_188848

-- Define a rightline as a sequence of natural numbers
def Rightline := ℕ → ℕ

-- A rightline without multiples of 3
def NoMultiplesOfThree (r : Rightline) : Prop :=
  ∀ n : ℕ, r n % 3 ≠ 0

-- Pairwise disjoint rightlines
def PairwiseDisjoint (rs : ℕ → Rightline) : Prop :=
  ∀ i j : ℕ, i ≠ j → (∀ n : ℕ, rs i n ≠ rs j n)

theorem infinite_chessboard_rightlines :
  (∃ r : Rightline, NoMultiplesOfThree r) ∧
  (∃ rs : ℕ → Rightline, PairwiseDisjoint rs ∧ (∀ i : ℕ, NoMultiplesOfThree (rs i))) :=
sorry

end infinite_chessboard_rightlines_l1888_188848


namespace f_is_odd_iff_l1888_188856

-- Define the function f(x) = x|x + a| + b
def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

-- State the theorem
theorem f_is_odd_iff (a b : ℝ) :
  (∀ x, f a b (-x) = -f a b x) ↔ 
  (∀ x, x * abs (-x + a) + b = -(x * abs (x + a) + b)) :=
by sorry

end f_is_odd_iff_l1888_188856


namespace expand_product_l1888_188861

theorem expand_product (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end expand_product_l1888_188861


namespace direct_variation_problem_l1888_188894

/-- z varies directly as w -/
def direct_variation (z w : ℝ) := ∃ k : ℝ, z = k * w

theorem direct_variation_problem (z w : ℝ → ℝ) :
  (∀ x, direct_variation (z x) (w x)) →  -- z varies directly as w
  z 5 = 10 →                             -- z = 10 when w = 5
  w 5 = 5 →                              -- w = 5 when z = 10
  w (-15) = -15 →                        -- w = -15
  z (-15) = -30                          -- z = -30 when w = -15
  := by sorry

end direct_variation_problem_l1888_188894


namespace inequality_solution_l1888_188827

def solution_set : Set ℝ := {x : ℝ | x^2 - 3*x - 4 < 0}

theorem inequality_solution : solution_set = Set.Ioo (-1) 4 := by sorry

end inequality_solution_l1888_188827


namespace optimal_strategy_minimizes_cost_l1888_188802

/-- Represents the bookstore's ordering strategy --/
structure OrderStrategy where
  numOrders : ℕ
  copiesPerOrder : ℕ

/-- Calculates the total cost for a given order strategy --/
def totalCost (s : OrderStrategy) : ℝ :=
  let handlingCost := 30 * s.numOrders
  let storageCost := 40 * (s.copiesPerOrder / 1000) * s.numOrders / 2
  handlingCost + storageCost

/-- The optimal order strategy --/
def optimalStrategy : OrderStrategy :=
  { numOrders := 10, copiesPerOrder := 15000 }

/-- Theorem stating that the optimal strategy minimizes total cost --/
theorem optimal_strategy_minimizes_cost :
  ∀ s : OrderStrategy,
    s.numOrders * s.copiesPerOrder = 150000 →
    totalCost optimalStrategy ≤ totalCost s :=
by sorry

#check optimal_strategy_minimizes_cost

end optimal_strategy_minimizes_cost_l1888_188802


namespace checkerboard_exists_l1888_188805

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100x100 board -/
def Board := Fin 100 → Fin 100 → Color

/-- Checks if a cell is adjacent to the boundary -/
def isAdjacentToBoundary (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square is monochromatic -/
def isMonochromatic2x2 (board : Board) (i j : Fin 100) : Prop :=
  ∃ c : Color, 
    board i j = c ∧ 
    board (i+1) j = c ∧ 
    board i (j+1) = c ∧ 
    board (i+1) (j+1) = c

/-- Checks if a 2x2 square has a checkerboard pattern -/
def isCheckerboard2x2 (board : Board) (i j : Fin 100) : Prop :=
  (board i j = Color.Black ∧ board (i+1) (j+1) = Color.Black ∧
   board (i+1) j = Color.White ∧ board i (j+1) = Color.White) ∨
  (board i j = Color.White ∧ board (i+1) (j+1) = Color.White ∧
   board (i+1) j = Color.Black ∧ board i (j+1) = Color.Black)

theorem checkerboard_exists (board : Board) 
  (boundary_black : ∀ i j : Fin 100, isAdjacentToBoundary i j → board i j = Color.Black)
  (no_monochromatic : ∀ i j : Fin 100, ¬isMonochromatic2x2 board i j) :
  ∃ i j : Fin 100, isCheckerboard2x2 board i j :=
sorry

end checkerboard_exists_l1888_188805


namespace log_inequality_l1888_188864

theorem log_inequality : ∀ x : ℝ, x > 0 → x - 1 ≥ Real.log x ∧ (x - 1 = Real.log x ↔ x = 1) := by
  sorry

end log_inequality_l1888_188864


namespace algebraic_expression_value_l1888_188891

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 5 = 0) :
  (x - 1)^2 - x*(x - 3) + (x + 2)*(x - 2) = 2 := by
  sorry

end algebraic_expression_value_l1888_188891


namespace largest_five_digit_with_product_l1888_188833

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem largest_five_digit_with_product : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ 
    digit_product n = 9 * 8 * 7 * 6 * 5 → 
    n ≤ 98765 :=
by
  sorry

end largest_five_digit_with_product_l1888_188833


namespace fruit_display_total_l1888_188831

/-- Proves that the total number of fruits on a display is 35, given specific conditions --/
theorem fruit_display_total (bananas oranges apples : ℕ) : 
  bananas = 5 →
  oranges = 2 * bananas →
  apples = 2 * oranges →
  bananas + oranges + apples = 35 := by
sorry

end fruit_display_total_l1888_188831


namespace geometric_series_sum_l1888_188885

/-- The sum of a geometric series with given parameters -/
theorem geometric_series_sum (a₁ : ℝ) (q : ℝ) (aₙ : ℝ) (h₁ : a₁ = 100) (h₂ : q = 1/10) (h₃ : aₙ = 0.01) :
  (a₁ - aₙ * q) / (1 - q) = (10^5 - 1) / 900 := by
  sorry

end geometric_series_sum_l1888_188885


namespace smallest_term_at_four_l1888_188883

def a (n : ℕ+) : ℚ := (1 / 3) * n^3 - 13 * n

theorem smallest_term_at_four :
  ∀ k : ℕ+, a 4 ≤ a k := by sorry

end smallest_term_at_four_l1888_188883


namespace farmer_euclid_field_l1888_188871

theorem farmer_euclid_field (a b c x : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c^2 = a^2 + b^2)
  (h4 : (b / c) * x + (a / c) * x = 3) :
  (a * b / 2 - x^2) / (a * b / 2) = 2393 / 2890 := by sorry

end farmer_euclid_field_l1888_188871


namespace negative_division_equals_positive_division_negative_three_hundred_by_negative_twenty_five_l1888_188868

theorem negative_division_equals_positive (x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (-x) / (-y) = x / y :=
sorry

theorem division_negative_three_hundred_by_negative_twenty_five :
  (-300) / (-25) = 12 :=
sorry

end negative_division_equals_positive_division_negative_three_hundred_by_negative_twenty_five_l1888_188868


namespace area_ratio_theorem_l1888_188811

/-- Given a triangle ABC with sides a, b, c, this structure represents the triangle and related points --/
structure TriangleWithIntersections where
  -- The lengths of the sides of triangle ABC
  a : ℝ
  b : ℝ
  c : ℝ
  -- The area of triangle ABC
  S_ABC : ℝ
  -- The area of hexagon PQRSTF
  S_PQRSTF : ℝ
  -- Assumption that a, b, c form a valid triangle
  triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem stating the relationship between the areas --/
theorem area_ratio_theorem (t : TriangleWithIntersections) :
  t.S_PQRSTF / t.S_ABC = 1 - (t.a * t.b + t.b * t.c + t.c * t.a) / (t.a + t.b + t.c)^2 := by
  sorry

end area_ratio_theorem_l1888_188811


namespace reciprocal_power_2006_l1888_188847

theorem reciprocal_power_2006 (a : ℚ) : 
  (a ≠ 0 ∧ a = 1 / a) → a^2006 = 1 := by
  sorry

end reciprocal_power_2006_l1888_188847


namespace jake_brought_one_balloon_l1888_188866

/-- The number of balloons Allan and Jake brought to the park in total -/
def total_balloons : ℕ := 3

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := total_balloons - allan_balloons

/-- Theorem stating that Jake brought 1 balloon to the park -/
theorem jake_brought_one_balloon : jake_balloons = 1 := by
  sorry

end jake_brought_one_balloon_l1888_188866


namespace xyz_sum_square_l1888_188859

theorem xyz_sum_square (x y z : ℕ+) 
  (h_gcd : Nat.gcd x.val (Nat.gcd y.val z.val) = 1)
  (h_x_div : x.val ∣ y.val * z.val * (x.val + y.val + z.val))
  (h_y_div : y.val ∣ x.val * z.val * (x.val + y.val + z.val))
  (h_z_div : z.val ∣ x.val * y.val * (x.val + y.val + z.val))
  (h_sum_div : (x.val + y.val + z.val) ∣ (x.val * y.val * z.val)) :
  ∃ (k : ℕ), x.val * y.val * z.val * (x.val + y.val + z.val) = k * k := by
  sorry

end xyz_sum_square_l1888_188859


namespace nickel_count_l1888_188820

def total_cents : ℕ := 400
def num_quarters : ℕ := 10
def num_dimes : ℕ := 12
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5

theorem nickel_count : 
  (total_cents - (num_quarters * quarter_value + num_dimes * dime_value)) / nickel_value = 6 := by
  sorry

end nickel_count_l1888_188820


namespace tan_half_sum_l1888_188892

theorem tan_half_sum (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5)
  (h2 : Real.sin a + Real.sin b = 1/5) :
  Real.tan ((a + b) / 2) = 1/3 := by
sorry

end tan_half_sum_l1888_188892


namespace wedge_volume_l1888_188846

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d h r : ℝ) (θ : ℝ) : 
  d = 18 →                           -- diameter of the log
  h = d →                            -- height of the cylindrical section
  r = d / 2 →                        -- radius of the log
  θ = 60 →                           -- angle between cuts in degrees
  (π * r^2 * h) / 2 = 729 * π := by
  sorry

#check wedge_volume

end wedge_volume_l1888_188846


namespace quadratic_equality_existence_l1888_188882

theorem quadratic_equality_existence (P : ℝ → ℝ) (h : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c ∧ a ≠ 0) :
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    P (b + c) = P a ∧ P (c + a) = P b ∧ P (a + b) = P c :=
by sorry

end quadratic_equality_existence_l1888_188882


namespace min_triangle_area_l1888_188895

/-- A point in the 2D plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- The rectangle OABC -/
structure Rectangle where
  O : IntPoint
  B : IntPoint

/-- Checks if a point is inside the rectangle -/
def isInside (r : Rectangle) (p : IntPoint) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.B.x ∧ 0 ≤ p.y ∧ p.y ≤ r.B.y

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : IntPoint) : ℚ :=
  (1 / 2 : ℚ) * |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)|

/-- The main theorem -/
theorem min_triangle_area (r : Rectangle) :
  r.O = ⟨0, 0⟩ → r.B = ⟨11, 8⟩ →
  ∃ (X : IntPoint), isInside r X ∧
    ∀ (Y : IntPoint), isInside r Y →
      triangleArea r.O r.B X ≤ triangleArea r.O r.B Y ∧
      triangleArea r.O r.B X = (1 / 2 : ℚ) := by
  sorry

end min_triangle_area_l1888_188895


namespace real_roots_condition_l1888_188898

theorem real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 2 * k * x + (k - 3) = 0) ↔ k ≥ 3/4 := by
sorry

end real_roots_condition_l1888_188898


namespace green_peaches_count_l1888_188807

/-- The number of baskets of peaches -/
def num_baskets : ℕ := 1

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 4

/-- The total number of peaches in all baskets -/
def total_peaches : ℕ := 7

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := total_peaches - (num_baskets * red_peaches_per_basket)

theorem green_peaches_count : green_peaches_per_basket = 3 := by
  sorry

end green_peaches_count_l1888_188807


namespace common_chord_length_O1_O2_l1888_188879

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The length of the common chord between two circles -/
def common_chord_length (c1 c2 : Circle) : ℝ := sorry

/-- Circle O₁ with equation (x+1)²+(y-3)²=9 -/
def O1 : Circle :=
  { equation := λ x y ↦ (x + 1)^2 + (y - 3)^2 = 9 }

/-- Circle O₂ with equation x²+y²-4x+2y-11=0 -/
def O2 : Circle :=
  { equation := λ x y ↦ x^2 + y^2 - 4*x + 2*y - 11 = 0 }

/-- Theorem stating that the length of the common chord between O₁ and O₂ is 24/5 -/
theorem common_chord_length_O1_O2 :
  common_chord_length O1 O2 = 24/5 := by sorry

end common_chord_length_O1_O2_l1888_188879


namespace minimum_handshakes_l1888_188855

theorem minimum_handshakes (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  (n * k) / 2 = 45 := by
  sorry

end minimum_handshakes_l1888_188855


namespace quadratic_form_bounds_l1888_188843

theorem quadratic_form_bounds (x y : ℝ) (h : x^2 + x*y + y^2 = 3) :
  1 ≤ x^2 - x*y + y^2 ∧ x^2 - x*y + y^2 ≤ 9 := by
  sorry

end quadratic_form_bounds_l1888_188843


namespace quadratic_roots_l1888_188812

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) →
  (∃ u v : ℝ, u ≠ v ∧ a * u^2 + b * u + c = 0 ∧ a * v^2 + b * v + c = 0) :=
by sorry

end quadratic_roots_l1888_188812


namespace milk_calculation_l1888_188801

/-- The amount of milk Yuna's family drank as a fraction of the total -/
def milk_drunk : ℝ := 0.4

/-- The amount of leftover milk in liters -/
def leftover_milk : ℝ := 0.69

/-- The initial amount of milk in liters -/
def initial_milk : ℝ := 1.15

theorem milk_calculation (milk_drunk : ℝ) (leftover_milk : ℝ) (initial_milk : ℝ) :
  milk_drunk = 0.4 →
  leftover_milk = 0.69 →
  initial_milk = 1.15 →
  initial_milk * (1 - milk_drunk) = leftover_milk :=
by sorry

end milk_calculation_l1888_188801


namespace must_divide_p_l1888_188844

theorem must_divide_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 40)
  (h3 : Nat.gcd r s = 60)
  (h4 : 120 < Nat.gcd s p)
  (h5 : Nat.gcd s p < 180) : 
  7 ∣ p.val := by
  sorry

end must_divide_p_l1888_188844


namespace maximal_k_inequality_l1888_188870

theorem maximal_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ k : ℝ, (k + a/b) * (k + b/c) * (k + c/a) ≤ (a/b + b/c + c/a) * (b/a + c/b + a/c) ↔ k ≤ Real.rpow 9 (1/3) - 1 :=
by sorry

end maximal_k_inequality_l1888_188870


namespace intersection_line_slope_l1888_188897

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 2*y + 40 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧
  circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧
  C ≠ D

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) (h : intersection C D) :
  (D.2 - C.2) / (D.1 - C.1) = -2/3 :=
sorry

end intersection_line_slope_l1888_188897


namespace system_solution_l1888_188884

theorem system_solution (x y z u v : ℝ) : 
  (x + y + z + u = 5) ∧
  (y + z + u + v = 1) ∧
  (z + u + v + x = 2) ∧
  (u + v + x + y = 0) ∧
  (v + x + y + z = 4) →
  (v = -2 ∧ x = 2 ∧ y = 1 ∧ z = 3 ∧ u = -1) := by
sorry

end system_solution_l1888_188884


namespace min_sum_of_squares_l1888_188863

theorem min_sum_of_squares (a b c t : ℝ) (h : a + b + c = t) :
  ∃ (m : ℝ), m = t^2 / 3 ∧ ∀ (x y z : ℝ), x + y + z = t → x^2 + y^2 + z^2 ≥ m :=
by sorry

end min_sum_of_squares_l1888_188863


namespace residue_mod_37_l1888_188817

theorem residue_mod_37 : ∃ k : ℤ, -927 = 37 * k + 35 ∧ (35 : ℤ) ∈ Set.range (fun i => i : Fin 37 → ℤ) := by
  sorry

end residue_mod_37_l1888_188817


namespace sequence_length_6_to_202_l1888_188877

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Proof that the arithmetic sequence from 6 to 202 with step 2 has 99 terms -/
theorem sequence_length_6_to_202 : 
  arithmeticSequenceLength 6 202 2 = 99 := by
  sorry

end sequence_length_6_to_202_l1888_188877


namespace mario_blossoms_l1888_188814

/-- The number of hibiscus plants Mario has -/
def num_plants : ℕ := 3

/-- The number of flowers on the first hibiscus plant -/
def flowers_first : ℕ := 2

/-- The number of flowers on the second hibiscus plant -/
def flowers_second : ℕ := 2 * flowers_first

/-- The number of flowers on the third hibiscus plant -/
def flowers_third : ℕ := 4 * flowers_second

/-- The total number of blossoms Mario has -/
def total_blossoms : ℕ := flowers_first + flowers_second + flowers_third

theorem mario_blossoms : total_blossoms = 22 := by
  sorry

end mario_blossoms_l1888_188814


namespace like_terms_mn_value_l1888_188809

theorem like_terms_mn_value (n m : ℕ) :
  (∃ (a b : ℝ) (x y : ℝ), a * x^n * y^3 = b * x^3 * y^m) →
  m^n = 27 := by
  sorry

end like_terms_mn_value_l1888_188809


namespace inequality_solution_set_l1888_188880

theorem inequality_solution_set (x : ℝ) : -x^2 + 2*x + 3 ≥ 0 ↔ x ∈ Set.Icc (-1) 3 := by
  sorry

end inequality_solution_set_l1888_188880


namespace diamond_example_l1888_188893

/-- Definition of the diamond operation for real numbers -/
def diamond (x y : ℝ) : ℝ := (x + y)^2 * (x - y)^2

/-- Theorem stating that 2 ◇ (3 ◇ 4) = 5745329 -/
theorem diamond_example : diamond 2 (diamond 3 4) = 5745329 := by
  sorry

end diamond_example_l1888_188893


namespace cafeteria_extra_apples_l1888_188874

/-- The number of extra apples in the cafeteria -/
def extra_apples (red_apples green_apples students_wanting_fruit : ℕ) : ℕ :=
  red_apples + green_apples - students_wanting_fruit

/-- Theorem: The cafeteria ends up with 40 extra apples -/
theorem cafeteria_extra_apples :
  extra_apples 42 7 9 = 40 := by
  sorry

end cafeteria_extra_apples_l1888_188874


namespace root_equation_implication_l1888_188881

theorem root_equation_implication (m : ℝ) : 
  m^2 - m - 3 = 0 → m^2 - m - 2 = 1 := by
  sorry

end root_equation_implication_l1888_188881


namespace even_odd_sum_difference_prove_even_odd_sum_difference_l1888_188834

theorem even_odd_sum_difference : ℕ → Prop :=
  fun n =>
    let even_sum := (n + 1) * (2 + 2 * n)
    let odd_sum := n * (1 + 2 * n - 1)
    even_sum - odd_sum = 6017

theorem prove_even_odd_sum_difference :
  even_odd_sum_difference 2003 := by sorry

end even_odd_sum_difference_prove_even_odd_sum_difference_l1888_188834


namespace largest_value_is_E_l1888_188823

theorem largest_value_is_E :
  let a := 24680 + 2 / 1357
  let b := 24680 - 2 / 1357
  let c := 24680 * 2 / 1357
  let d := 24680 / (2 / 1357)
  let e := 24680 ^ 1.357
  (e > a) ∧ (e > b) ∧ (e > c) ∧ (e > d) := by
  sorry

end largest_value_is_E_l1888_188823


namespace virus_length_scientific_notation_l1888_188842

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem virus_length_scientific_notation :
  toScientificNotation 0.00000032 = ScientificNotation.mk 3.2 (-7) (by norm_num) :=
sorry

end virus_length_scientific_notation_l1888_188842


namespace some_number_value_l1888_188810

theorem some_number_value : ∀ some_number : ℝ, 
  (54 / some_number) * (54 / 162) = 1 → some_number = 18 :=
by
  sorry

end some_number_value_l1888_188810


namespace mrs_martin_coffee_cups_l1888_188819

-- Define the cost of a bagel
def bagel_cost : ℝ := 1.5

-- Define Mrs. Martin's purchase
def mrs_martin_total : ℝ := 12.75
def mrs_martin_bagels : ℕ := 2

-- Define Mr. Martin's purchase
def mr_martin_total : ℝ := 14.00
def mr_martin_coffee : ℕ := 2
def mr_martin_bagels : ℕ := 5

-- Theorem to prove
theorem mrs_martin_coffee_cups : ℕ := by
  -- The proof goes here
  sorry

end mrs_martin_coffee_cups_l1888_188819


namespace solve_for_a_l1888_188806

/-- Given that x + 2a - 6 = 0 and x = -2, prove that a = 4 -/
theorem solve_for_a (x a : ℝ) (h1 : x + 2*a - 6 = 0) (h2 : x = -2) : a = 4 := by
  sorry

end solve_for_a_l1888_188806


namespace garrison_problem_l1888_188837

/-- Represents the initial number of men in the garrison -/
def initial_men : ℕ := 2000

/-- Represents the number of reinforcement men -/
def reinforcement : ℕ := 1600

/-- Represents the initial number of days the provisions would last -/
def initial_days : ℕ := 54

/-- Represents the number of days passed before reinforcement -/
def days_before_reinforcement : ℕ := 18

/-- Represents the number of days the provisions last after reinforcement -/
def remaining_days : ℕ := 20

theorem garrison_problem :
  initial_men * initial_days = 
  (initial_men + reinforcement) * remaining_days + 
  initial_men * days_before_reinforcement :=
by sorry

end garrison_problem_l1888_188837


namespace binary_1101_equals_13_l1888_188899

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- The binary representation of 1101₂ -/
def binary_1101 : List Bool := [true, true, false, true]

theorem binary_1101_equals_13 : binary_to_decimal binary_1101 = 13 := by
  sorry

end binary_1101_equals_13_l1888_188899


namespace monomial_properties_l1888_188800

/-- The coefficient of a monomial -/
def coefficient (m : ℚ) (x y : ℚ) : ℚ := m

/-- The degree of a monomial -/
def degree (x y : ℚ) : ℕ := 2 + 1

theorem monomial_properties :
  let m : ℚ := -π / 7
  let x : ℚ := 0  -- Placeholder value, not used in computation
  let y : ℚ := 0  -- Placeholder value, not used in computation
  (coefficient m x y = -π / 7) ∧ (degree x y = 3) := by
  sorry

end monomial_properties_l1888_188800


namespace expand_and_simplify_l1888_188832

theorem expand_and_simplify (x : ℝ) : (x - 1) * (x + 3) - x * (x - 2) = 4 * x - 3 := by
  sorry

end expand_and_simplify_l1888_188832


namespace sum_reciprocals_equals_negative_five_l1888_188858

theorem sum_reciprocals_equals_negative_five (x y : ℝ) 
  (eq1 : x^2 + Real.sqrt 3 * y = 4)
  (eq2 : y^2 + Real.sqrt 3 * x = 4)
  (neq : x ≠ y) :
  y / x + x / y = -5 := by
  sorry

end sum_reciprocals_equals_negative_five_l1888_188858


namespace james_local_taxes_l1888_188886

/-- Calculates the amount of local taxes paid in cents per hour -/
def local_taxes_cents (hourly_wage : ℚ) (tax_rate : ℚ) : ℚ :=
  hourly_wage * 100 * tax_rate

theorem james_local_taxes :
  local_taxes_cents 25 (24/1000) = 60 := by
  sorry

end james_local_taxes_l1888_188886


namespace series_sum_l1888_188815

/-- The sum of the infinite series Σ(n=1 to ∞) of n/(3^n) equals 9/4 -/
theorem series_sum : ∑' n : ℕ, (n : ℝ) / (3 : ℝ) ^ n = 9 / 4 := by
  sorry

end series_sum_l1888_188815


namespace m_range_l1888_188850

def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

theorem m_range (m : ℝ) : A ⊆ B m ∧ A ≠ B m → m > 2 := by
  sorry

end m_range_l1888_188850


namespace fraction_to_percentage_l1888_188829

/-- Represents a mixed repeating decimal number -/
structure MixedRepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : ℚ
  repeatingPart : ℚ

/-- Converts a rational number to a MixedRepeatingDecimal -/
def toMixedRepeatingDecimal (q : ℚ) : MixedRepeatingDecimal :=
  sorry

/-- Converts a MixedRepeatingDecimal to a percentage string -/
def toPercentageString (m : MixedRepeatingDecimal) : String :=
  sorry

theorem fraction_to_percentage (n d : ℕ) (h : d ≠ 0) :
  toPercentageString (toMixedRepeatingDecimal (n / d)) = "8.(923076)%" :=
sorry

end fraction_to_percentage_l1888_188829


namespace next_free_haircut_in_ten_l1888_188857

-- Define the constants from the problem
def haircuts_per_free : ℕ := 14
def free_haircuts_received : ℕ := 5
def total_haircuts : ℕ := 79

-- Define a function to calculate the number of haircuts until the next free one
def haircuts_until_next_free (total : ℕ) (free : ℕ) (per_free : ℕ) : ℕ :=
  per_free - (total - free) % per_free

-- Theorem statement
theorem next_free_haircut_in_ten :
  haircuts_until_next_free total_haircuts free_haircuts_received haircuts_per_free = 10 := by
  sorry


end next_free_haircut_in_ten_l1888_188857


namespace inequality_solution_range_l1888_188888

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) ↔ 
  (a < -2 ∨ a ≥ 6/5) :=
sorry

end inequality_solution_range_l1888_188888


namespace square_field_area_l1888_188840

theorem square_field_area (wire_length : ℝ) (wire_rounds : ℕ) (field_area : ℝ) : 
  wire_length = 7348 →
  wire_rounds = 11 →
  wire_length = 4 * wire_rounds * Real.sqrt field_area →
  field_area = 27889 := by
  sorry

end square_field_area_l1888_188840


namespace exam_day_percentage_l1888_188813

/-- Represents the percentage of students who took the exam on the assigned day -/
def assigned_day_percentage : ℝ := 70

/-- Represents the total number of students in the class -/
def total_students : ℕ := 100

/-- Represents the average score of students who took the exam on the assigned day -/
def assigned_day_score : ℝ := 60

/-- Represents the average score of students who took the exam on the make-up date -/
def makeup_day_score : ℝ := 80

/-- Represents the average score for the entire class -/
def class_average_score : ℝ := 66

theorem exam_day_percentage :
  assigned_day_percentage * assigned_day_score / 100 +
  (100 - assigned_day_percentage) * makeup_day_score / 100 =
  class_average_score :=
sorry

end exam_day_percentage_l1888_188813


namespace rhombus_area_l1888_188869

-- Define the vertices of the rhombus
def v1 : ℝ × ℝ := (1.2, 4.1)
def v2 : ℝ × ℝ := (7.3, 2.5)
def v3 : ℝ × ℝ := (1.2, -2.8)
def v4 : ℝ × ℝ := (-4.9, 2.5)

-- Define the vectors representing two adjacent sides of the rhombus
def vector1 : ℝ × ℝ := (v2.1 - v1.1, v2.2 - v1.2)
def vector2 : ℝ × ℝ := (v4.1 - v1.1, v4.2 - v1.2)

-- Theorem stating that the area of the rhombus is 19.52 square units
theorem rhombus_area : 
  abs ((vector1.1 * vector2.2) - (vector1.2 * vector2.1)) = 19.52 := by
  sorry

end rhombus_area_l1888_188869


namespace park_trees_after_planting_l1888_188849

theorem park_trees_after_planting (current_trees new_trees : ℕ) 
  (h1 : current_trees = 25)
  (h2 : new_trees = 73) :
  current_trees + new_trees = 98 :=
by sorry

end park_trees_after_planting_l1888_188849


namespace common_point_theorem_l1888_188887

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Constructs a line based on the given conditions -/
def construct_line (a d r : ℝ) : Line :=
  { a := a
  , b := a + d
  , c := a * r + 2 * d }

theorem common_point_theorem (a d r : ℝ) :
  (construct_line a d r).contains (-1) 2 := by
  sorry

end common_point_theorem_l1888_188887


namespace completing_square_l1888_188818

theorem completing_square (x : ℝ) : x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
  sorry

end completing_square_l1888_188818


namespace square_perimeter_is_96_l1888_188851

/-- A square ABCD with side lengths expressed in terms of x -/
structure Square (x : ℝ) where
  AB : ℝ := x + 16
  BC : ℝ := 3 * x
  is_square : AB = BC

/-- The perimeter of the square ABCD is 96 -/
theorem square_perimeter_is_96 (x : ℝ) (ABCD : Square x) : 
  4 * ABCD.AB = 96 := by
  sorry

#check square_perimeter_is_96

end square_perimeter_is_96_l1888_188851


namespace fraction_dislike_but_interested_l1888_188873

/-- Represents the student population at Novo Middle School -/
structure SchoolPopulation where
  total : ℕ
  artInterested : ℕ
  artUninterested : ℕ
  interestedLike : ℕ
  interestedDislike : ℕ
  uninterestedLike : ℕ
  uninterestedDislike : ℕ

/-- Theorem about the fraction of students who dislike art but are interested -/
theorem fraction_dislike_but_interested (pop : SchoolPopulation) : 
  pop.total = 200 ∧ 
  pop.artInterested = 150 ∧ 
  pop.artUninterested = 50 ∧
  pop.interestedLike = 105 ∧
  pop.interestedDislike = 45 ∧
  pop.uninterestedLike = 10 ∧
  pop.uninterestedDislike = 40 →
  (pop.interestedDislike : ℚ) / (pop.interestedDislike + pop.uninterestedDislike) = 9/17 := by
  sorry

#check fraction_dislike_but_interested

end fraction_dislike_but_interested_l1888_188873


namespace replaced_person_weight_l1888_188825

/-- Proves that the weight of the replaced person is 55 kg given the conditions -/
theorem replaced_person_weight (initial_count : ℕ) (weight_increase : ℝ) (new_person_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 4 →
  new_person_weight = 87 →
  (initial_count : ℝ) * weight_increase + new_person_weight = 
    (initial_count : ℝ) * weight_increase + 55 := by
  sorry

end replaced_person_weight_l1888_188825


namespace stating_shop_owner_cheat_percentage_l1888_188821

/-- Represents the percentage by which the shop owner cheats -/
def cheat_percentage : ℝ := 22.22222222222222

/-- Represents the profit percentage of the shop owner -/
def profit_percentage : ℝ := 22.22222222222222

/-- 
Theorem stating that if a shop owner cheats by the same percentage while buying and selling,
and their profit percentage is 22.22222222222222%, then the cheat percentage is also 22.22222222222222%.
-/
theorem shop_owner_cheat_percentage :
  cheat_percentage = profit_percentage :=
sorry

end stating_shop_owner_cheat_percentage_l1888_188821


namespace least_five_digit_congruent_to_5_mod_15_l1888_188824

theorem least_five_digit_congruent_to_5_mod_15 : ∃ (n : ℕ), 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  n % 15 = 5 ∧
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ m % 15 = 5 → n ≤ m) ∧
  n = 10010 :=
sorry

end least_five_digit_congruent_to_5_mod_15_l1888_188824


namespace inequality_proof_l1888_188835

theorem inequality_proof (a b c : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0)
  (sum_one : a + b + c = 1) : 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := by
sorry

end inequality_proof_l1888_188835


namespace triangle_properties_l1888_188889

-- Define the triangle ABC
def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  a = 2 * Real.sqrt 2 ∧ b = 5 ∧ c = Real.sqrt 13

-- Theorem to prove the three parts of the problem
theorem triangle_properties {A B C a b c : Real} 
  (h : triangle_ABC A B C a b c) : 
  C = π / 4 ∧ 
  Real.sin A = 2 * Real.sqrt 13 / 13 ∧ 
  Real.sin (2 * A + π / 4) = 17 * Real.sqrt 2 / 26 := by
  sorry


end triangle_properties_l1888_188889


namespace sufficient_not_necessary_l1888_188867

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 2 → 1/x < 1/2) ∧
  (∃ x, 1/x < 1/2 ∧ x ≤ 2) := by
  sorry

end sufficient_not_necessary_l1888_188867


namespace negation_of_forall_positive_l1888_188828

theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) :=
by sorry

end negation_of_forall_positive_l1888_188828


namespace solve_equation_l1888_188803

theorem solve_equation (x : ℝ) : (x - 5) ^ 4 = 16 ↔ x = 7 := by sorry

end solve_equation_l1888_188803


namespace function_equality_condition_l1888_188816

theorem function_equality_condition (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x
  ({x : ℝ | f x = 0} = {x : ℝ | f (f x) = 0} ∧ {x : ℝ | f x = 0}.Nonempty) ↔ 0 ≤ a ∧ a < 4 := by
  sorry

end function_equality_condition_l1888_188816


namespace geometric_sequence_property_l1888_188808

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∃ r, ∀ n, a (n + 1) = r * a n)

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_roots : a 3 + a 15 = 6 ∧ a 3 * a 15 = 8) :
  a 1 * a 17 / a 9 = 2 * Real.sqrt 2 :=
sorry

end geometric_sequence_property_l1888_188808


namespace monopoly_wins_ratio_l1888_188839

/-- 
Proves that given the conditions of the Monopoly game wins, 
the ratio of Susan's wins to Betsy's wins is 3:1
-/
theorem monopoly_wins_ratio :
  ∀ (betsy helen susan : ℕ),
  betsy = 5 →
  helen = 2 * betsy →
  betsy + helen + susan = 30 →
  susan / betsy = 3 := by
sorry

end monopoly_wins_ratio_l1888_188839


namespace inequality_equivalence_l1888_188845

theorem inequality_equivalence (x y : ℝ) : 
  y - x < Real.sqrt (4 * x^2) ↔ (x ≥ 0 ∧ y < 3 * x) ∨ (x < 0 ∧ y < -x) := by
  sorry

end inequality_equivalence_l1888_188845


namespace divisibility_of_A_l1888_188860

def A : ℕ := 2013 * (10^(4*165) - 1) / (10^4 - 1)

theorem divisibility_of_A : 2013^2 ∣ A := by sorry

end divisibility_of_A_l1888_188860


namespace smallest_x_value_l1888_188852

theorem smallest_x_value (x : ℝ) : 
  (3 * x^2 + 36 * x - 90 = 2 * x * (x + 16)) → x ≥ -10 :=
by sorry

end smallest_x_value_l1888_188852


namespace factorial_ratio_l1888_188854

theorem factorial_ratio : (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_ratio_l1888_188854


namespace triangle_inequality_l1888_188896

theorem triangle_inequality (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h1 : a * y + b * x = c)
  (h2 : c * x + a * z = b)
  (h3 : b * z + c * y = a) :
  (x / (1 - y * z)) + (y / (1 - z * x)) + (z / (1 - x * y)) ≤ 2 := by
sorry

end triangle_inequality_l1888_188896
