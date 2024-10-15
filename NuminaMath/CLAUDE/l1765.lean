import Mathlib

namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l1765_176544

theorem indefinite_integral_proof (x : Real) :
  let f := fun x => -(1 / (x - Real.sin x))
  let g := fun x => (1 - Real.cos x) / (x - Real.sin x)^2
  deriv f x = g x :=
by sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l1765_176544


namespace NUMINAMATH_CALUDE_least_cube_divisible_by_168_l1765_176577

theorem least_cube_divisible_by_168 :
  ∀ k : ℕ, k > 0 → k^3 % 168 = 0 → k ≥ 42 :=
by
  sorry

end NUMINAMATH_CALUDE_least_cube_divisible_by_168_l1765_176577


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l1765_176525

theorem ratio_a_to_b (a b : ℝ) (h : (3 * a + 2 * b) / (3 * a - 2 * b) = 3) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l1765_176525


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l1765_176561

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection_plane_plane : Plane → Plane → Line)

theorem line_parallel_to_intersection_of_parallel_planes 
  (a : Line) (α β : Plane) (b : Line) :
  parallel_line_plane a α →
  parallel_line_plane a β →
  intersection_plane_plane α β = b →
  parallel_line_line a b := by
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l1765_176561


namespace NUMINAMATH_CALUDE_sams_age_l1765_176571

theorem sams_age (billy joe sam : ℕ) 
  (h1 : billy = 2 * joe) 
  (h2 : billy + joe = 60) 
  (h3 : sam = (billy + joe) / 2) : 
  sam = 30 := by
sorry

end NUMINAMATH_CALUDE_sams_age_l1765_176571


namespace NUMINAMATH_CALUDE_min_cost_rectangular_container_l1765_176572

/-- Represents the cost function for a rectangular container -/
def cost_function (a b : ℝ) : ℝ := 20 * a * b + 10 * 2 * (a + b)

/-- Theorem stating the minimum cost for the rectangular container -/
theorem min_cost_rectangular_container :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = 4 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y = 4 → cost_function a b ≤ cost_function x y) ∧
  cost_function a b = 160 :=
sorry

end NUMINAMATH_CALUDE_min_cost_rectangular_container_l1765_176572


namespace NUMINAMATH_CALUDE_trishSellPriceIs150Cents_l1765_176535

/-- The price at which Trish sells each stuffed animal -/
def trishSellPrice (barbaraStuffedAnimals : ℕ) (barbaraSellPrice : ℚ) (totalDonation : ℚ) : ℚ :=
  let trishStuffedAnimals := 2 * barbaraStuffedAnimals
  let barbaraContribution := barbaraStuffedAnimals * barbaraSellPrice
  let trishContribution := totalDonation - barbaraContribution
  trishContribution / trishStuffedAnimals

theorem trishSellPriceIs150Cents 
  (barbaraStuffedAnimals : ℕ) 
  (barbaraSellPrice : ℚ) 
  (totalDonation : ℚ) 
  (h1 : barbaraStuffedAnimals = 9)
  (h2 : barbaraSellPrice = 2)
  (h3 : totalDonation = 45) :
  trishSellPrice barbaraStuffedAnimals barbaraSellPrice totalDonation = 3/2 := by
  sorry

#eval trishSellPrice 9 2 45

end NUMINAMATH_CALUDE_trishSellPriceIs150Cents_l1765_176535


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l1765_176595

theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    field_length = 2 * field_width →
    field_length = 16 →
    pond_side = 4 →
    (pond_side^2) / (field_length * field_width) = 1/8 :=
by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l1765_176595


namespace NUMINAMATH_CALUDE_solution_value_l1765_176558

theorem solution_value (x y a : ℝ) : 
  x = 1 ∧ y = 1 ∧ 2*x - a*y = 3 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1765_176558


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l1765_176532

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_2015th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a5 : a 5 = 6) :
  a 2015 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l1765_176532


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1765_176508

open Set
open Real

-- Define a type for points in the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point has integer coordinates
def isIntegerPoint (p : Point) : Prop :=
  ∃ (m n : ℤ), p.x = m ∧ p.y = n

-- Define a circle with center and radius
def Circle (center : Point) (radius : ℝ) : Set Point :=
  {p : Point | (p.x - center.x)^2 + (p.y - center.y)^2 ≤ radius^2}

-- Define the intersection of two circles
def circlesIntersect (c1 c2 : Set Point) : Prop :=
  ∃ (p : Point), p ∈ c1 ∧ p ∈ c2

-- State the theorem
theorem circle_intersection_theorem :
  ∀ (O : Point),
    ∃ (I : Point),
      isIntegerPoint I ∧
      circlesIntersect (Circle O 100) (Circle I (1/14)) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1765_176508


namespace NUMINAMATH_CALUDE_bagel_store_spending_l1765_176580

theorem bagel_store_spending :
  ∀ (B D : ℝ),
  D = (7/10) * B →
  B = D + 15 →
  B + D = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_bagel_store_spending_l1765_176580


namespace NUMINAMATH_CALUDE_beth_coin_sale_l1765_176563

/-- Given Beth's initial gold coins and Carl's gift, prove the number of coins Beth sold when she sold half her total. -/
theorem beth_coin_sale (initial_coins : ℕ) (gift_coins : ℕ) : 
  initial_coins = 125 → gift_coins = 35 → (initial_coins + gift_coins) / 2 = 80 := by
sorry

end NUMINAMATH_CALUDE_beth_coin_sale_l1765_176563


namespace NUMINAMATH_CALUDE_train_problem_l1765_176515

/-- Calculates the number of people who got on a train given the initial count, 
    the number who got off, and the final count. -/
def peopleGotOn (initial : ℕ) (gotOff : ℕ) (final : ℕ) : ℕ :=
  final - (initial - gotOff)

theorem train_problem : peopleGotOn 78 27 63 = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_problem_l1765_176515


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l1765_176584

-- Define sets M and N
def M (a : ℝ) : Set ℝ := {3, 2*a}
def N (a : ℝ) : Set ℝ := {a+1, 3}

-- State the theorem
theorem subset_implies_a_equals_one :
  ∀ a : ℝ, M a ⊆ N a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l1765_176584


namespace NUMINAMATH_CALUDE_abs_3_minus_4i_l1765_176551

theorem abs_3_minus_4i : Complex.abs (3 - 4 * Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_3_minus_4i_l1765_176551


namespace NUMINAMATH_CALUDE_grunters_win_all_games_l1765_176506

/-- The number of games played between the Grunters and the Screamers -/
def num_games : ℕ := 6

/-- The probability of the Grunters winning a game that doesn't go to overtime -/
def p_win_no_overtime : ℝ := 0.6

/-- The probability of the Grunters winning a game that goes to overtime -/
def p_win_overtime : ℝ := 0.5

/-- The probability of a game going to overtime -/
def p_overtime : ℝ := 0.1

/-- The theorem stating the probability of the Grunters winning all games -/
theorem grunters_win_all_games : 
  (((1 - p_overtime) * p_win_no_overtime + p_overtime * p_win_overtime) ^ num_games : ℝ) = 
  (823543 : ℝ) / 10000000 := by sorry

end NUMINAMATH_CALUDE_grunters_win_all_games_l1765_176506


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1765_176537

theorem linear_equation_solution :
  ∀ x : ℝ, x - 2 = 0 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1765_176537


namespace NUMINAMATH_CALUDE_tan_beta_value_l1765_176554

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l1765_176554


namespace NUMINAMATH_CALUDE_square_with_semicircular_arcs_perimeter_l1765_176536

/-- The perimeter of a region bounded by semicircular arcs constructed on the sides of a square -/
theorem square_with_semicircular_arcs_perimeter (side_length : Real) : 
  side_length = 4 / Real.pi → 
  (4 : Real) * Real.pi * (side_length / 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_with_semicircular_arcs_perimeter_l1765_176536


namespace NUMINAMATH_CALUDE_xiao_he_purchase_cost_l1765_176559

/-- The total cost of Xiao He's purchase -/
def total_cost (notebook_price pen_price : ℝ) : ℝ :=
  4 * notebook_price + 10 * pen_price

/-- Theorem: The total cost of Xiao He's purchase is 4a + 10b -/
theorem xiao_he_purchase_cost (a b : ℝ) :
  total_cost a b = 4 * a + 10 * b := by
  sorry

end NUMINAMATH_CALUDE_xiao_he_purchase_cost_l1765_176559


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l1765_176578

theorem product_from_lcm_gcd (x y : ℕ+) 
  (h_lcm : Nat.lcm x y = 48) 
  (h_gcd : Nat.gcd x y = 8) : 
  x * y = 384 := by
sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l1765_176578


namespace NUMINAMATH_CALUDE_oxide_other_element_weight_l1765_176522

/-- The atomic weight of the other element in a calcium oxide -/
def atomic_weight_other_element (molecular_weight : ℝ) (calcium_weight : ℝ) : ℝ :=
  molecular_weight - calcium_weight

/-- Theorem stating that the atomic weight of the other element in the oxide is 16 -/
theorem oxide_other_element_weight :
  let molecular_weight : ℝ := 56
  let calcium_weight : ℝ := 40
  atomic_weight_other_element molecular_weight calcium_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_oxide_other_element_weight_l1765_176522


namespace NUMINAMATH_CALUDE_sandwich_jam_cost_l1765_176539

theorem sandwich_jam_cost 
  (N B J : ℕ) 
  (h1 : N > 1) 
  (h2 : B > 0) 
  (h3 : J > 0) 
  (h4 : N * (4 * B + 5 * J + 20) = 414) : 
  N * 5 * J = 225 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_jam_cost_l1765_176539


namespace NUMINAMATH_CALUDE_victory_guarantee_l1765_176514

/-- Represents the state of the archery tournament -/
structure ArcheryTournament where
  totalShots : ℕ
  halfwayPoint : ℕ
  jessicaLead : ℕ
  bullseyeScore : ℕ
  minJessicaScore : ℕ

/-- Calculates the minimum number of bullseyes Jessica needs to guarantee victory -/
def minBullseyesForVictory (tournament : ArcheryTournament) : ℕ :=
  let remainingShots := tournament.totalShots - tournament.halfwayPoint
  let maxOpponentScore := tournament.bullseyeScore * remainingShots
  let jessicaNeededScore := maxOpponentScore - tournament.jessicaLead + 1
  (jessicaNeededScore + remainingShots * tournament.minJessicaScore - 1) / 
    (tournament.bullseyeScore - tournament.minJessicaScore) + 1

theorem victory_guarantee (tournament : ArcheryTournament) 
  (h1 : tournament.totalShots = 80)
  (h2 : tournament.halfwayPoint = 40)
  (h3 : tournament.jessicaLead = 30)
  (h4 : tournament.bullseyeScore = 10)
  (h5 : tournament.minJessicaScore = 2) :
  minBullseyesForVictory tournament = 37 := by
  sorry

#eval minBullseyesForVictory { 
  totalShots := 80, 
  halfwayPoint := 40, 
  jessicaLead := 30, 
  bullseyeScore := 10, 
  minJessicaScore := 2 
}

end NUMINAMATH_CALUDE_victory_guarantee_l1765_176514


namespace NUMINAMATH_CALUDE_quadratic_inequality_set_l1765_176555

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 6 * m * x + 5 * m + 1

-- State the theorem
theorem quadratic_inequality_set :
  {m : ℝ | ∀ x : ℝ, f m x > 0} = {m : ℝ | 0 ≤ m ∧ m < 1/4} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_set_l1765_176555


namespace NUMINAMATH_CALUDE_half_area_closest_to_longest_side_l1765_176530

/-- Represents a trapezoid field with specific measurements -/
structure TrapezoidField where
  short_base : ℝ
  long_base : ℝ
  slant_side : ℝ
  slant_angle : ℝ

/-- The fraction of the area closer to the longest side of the trapezoid field -/
def fraction_closest_to_longest_side (field : TrapezoidField) : ℝ :=
  sorry

/-- Theorem stating that for a specific trapezoid field, the fraction of area closest to the longest side is 1/2 -/
theorem half_area_closest_to_longest_side :
  let field : TrapezoidField := {
    short_base := 80,
    long_base := 160,
    slant_side := 120,
    slant_angle := π / 4
  }
  fraction_closest_to_longest_side field = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_area_closest_to_longest_side_l1765_176530


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l1765_176564

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles :
  let circle1 := fun (x y : ℝ) => x^2 - 6*x + y^2 - 8*y - 15 = 0
  let circle2 := fun (x y : ℝ) => x^2 + 10*x + y^2 + 12*y + 21 = 0
  ∃ d : ℝ, d = 2 * Real.sqrt 41 - Real.sqrt 97 ∧
    ∀ p q : ℝ × ℝ, circle1 p.1 p.2 → circle2 q.1 q.2 →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l1765_176564


namespace NUMINAMATH_CALUDE_quarters_for_mowing_lawns_l1765_176565

def penny_value : ℚ := 1 / 100
def quarter_value : ℚ := 25 / 100

def pennies : ℕ := 9
def total_amount : ℚ := 184 / 100

theorem quarters_for_mowing_lawns :
  (total_amount - pennies * penny_value) / quarter_value = 7 := by sorry

end NUMINAMATH_CALUDE_quarters_for_mowing_lawns_l1765_176565


namespace NUMINAMATH_CALUDE_tiger_tree_trunk_time_l1765_176557

/-- The time taken for a tiger to run above a fallen tree trunk -/
theorem tiger_tree_trunk_time (tiger_length : ℝ) (tree_trunk_length : ℝ) (time_to_pass_point : ℝ) : 
  tiger_length = 5 →
  tree_trunk_length = 20 →
  time_to_pass_point = 1 →
  (tiger_length + tree_trunk_length) / (tiger_length / time_to_pass_point) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tiger_tree_trunk_time_l1765_176557


namespace NUMINAMATH_CALUDE_exists_top_choice_l1765_176547

/- Define the type for houses and people -/
variable {α : Type*} [Finite α]

/- Define the preference relation -/
def Prefers (p : α → α → Prop) : Prop :=
  ∀ x y z, p x y ∧ p y z → p x z

/- Define the assignment function -/
def Assignment (f : α → α) : Prop :=
  Function.Bijective f

/- Define the stability condition -/
def Stable (f : α → α) (p : α → α → Prop) : Prop :=
  ∀ g : α → α, Assignment g →
    ∃ x, p x (f x) ∧ ¬p x (g x)

/- State the theorem -/
theorem exists_top_choice
  (f : α → α)
  (p : α → α → Prop)
  (h_assign : Assignment f)
  (h_prefers : Prefers p)
  (h_stable : Stable f p) :
  ∃ x, ∀ y, p x (f x) ∧ (p x y → y = f x) :=
sorry

end NUMINAMATH_CALUDE_exists_top_choice_l1765_176547


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l1765_176568

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l1765_176568


namespace NUMINAMATH_CALUDE_min_fence_posts_for_field_l1765_176594

/-- Calculates the number of fence posts needed for a rectangular field -/
def fence_posts (length width post_spacing_long post_spacing_short : ℕ) : ℕ :=
  let long_side_posts := length / post_spacing_long + 1
  let short_side_posts := width / post_spacing_short + 1
  long_side_posts + 2 * (short_side_posts - 1)

/-- Theorem stating the minimum number of fence posts required for the given field -/
theorem min_fence_posts_for_field : 
  fence_posts 150 50 15 10 = 21 :=
by sorry

end NUMINAMATH_CALUDE_min_fence_posts_for_field_l1765_176594


namespace NUMINAMATH_CALUDE_max_median_amount_l1765_176505

/-- Represents the initial amounts of money for each person -/
def initial_amounts : List ℕ := [28, 72, 98]

/-- The total amount of money after pooling -/
def total_amount : ℕ := initial_amounts.sum

/-- The number of people -/
def num_people : ℕ := initial_amounts.length

theorem max_median_amount :
  ∃ (distribution : List ℕ),
    distribution.length = num_people ∧
    distribution.sum = total_amount ∧
    (∃ (median : ℕ), median ∈ distribution ∧ 
      (distribution.filter (λ x => x ≤ median)).length ≥ num_people / 2 ∧
      (distribution.filter (λ x => x ≥ median)).length ≥ num_people / 2) ∧
    (∀ (other_distribution : List ℕ),
      other_distribution.length = num_people →
      other_distribution.sum = total_amount →
      (∃ (other_median : ℕ), other_median ∈ other_distribution ∧ 
        (other_distribution.filter (λ x => x ≤ other_median)).length ≥ num_people / 2 ∧
        (other_distribution.filter (λ x => x ≥ other_median)).length ≥ num_people / 2) →
      ∃ (median : ℕ), median ∈ distribution ∧ 
        (distribution.filter (λ x => x ≤ median)).length ≥ num_people / 2 ∧
        (distribution.filter (λ x => x ≥ median)).length ≥ num_people / 2 ∧
        median ≥ other_median) ∧
    (∃ (median : ℕ), median ∈ distribution ∧ 
      (distribution.filter (λ x => x ≤ median)).length ≥ num_people / 2 ∧
      (distribution.filter (λ x => x ≥ median)).length ≥ num_people / 2 ∧
      median = 196) := by
  sorry


end NUMINAMATH_CALUDE_max_median_amount_l1765_176505


namespace NUMINAMATH_CALUDE_contrapositive_odd_product_l1765_176581

theorem contrapositive_odd_product (a b : ℤ) :
  (¬(Odd (a * b)) → ¬(Odd a ∧ Odd b)) ↔
  ((Odd a ∧ Odd b) → Odd (a * b)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_odd_product_l1765_176581


namespace NUMINAMATH_CALUDE_cube_packing_surface_area_l1765_176589

/-- A rectangular box that can fit cubic products. -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The surface area of a box in square centimeters. -/
def surfaceArea (b : Box) : ℕ :=
  2 * (b.length * b.width + b.length * b.height + b.width * b.height)

/-- The volume of a box in cubic centimeters. -/
def volume (b : Box) : ℕ :=
  b.length * b.width * b.height

theorem cube_packing_surface_area :
  ∃ (b : Box), volume b = 12 ∧ (surfaceArea b = 40 ∨ surfaceArea b = 38 ∨ surfaceArea b = 32) := by
  sorry


end NUMINAMATH_CALUDE_cube_packing_surface_area_l1765_176589


namespace NUMINAMATH_CALUDE_blackboard_numbers_theorem_l1765_176502

theorem blackboard_numbers_theorem (n : ℕ) (h_n : n > 3) 
  (numbers : Fin n → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → numbers i ≠ numbers j) 
  (h_bound : ∀ i, numbers i < Nat.factorial (n - 1)) :
  ∃ (i j k l : Fin n), i ≠ k ∧ j ≠ l ∧ numbers i > numbers j ∧ numbers k > numbers l ∧
    (numbers i / numbers j : ℕ) = (numbers k / numbers l : ℕ) :=
sorry

end NUMINAMATH_CALUDE_blackboard_numbers_theorem_l1765_176502


namespace NUMINAMATH_CALUDE_complex_power_problem_l1765_176591

theorem complex_power_problem : ((1 - Complex.I) / (1 + Complex.I)) ^ 10 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l1765_176591


namespace NUMINAMATH_CALUDE_unique_line_through_points_l1765_176516

-- Define a type for points in Euclidean geometry
variable (Point : Type)

-- Define a type for lines in Euclidean geometry
variable (Line : Type)

-- Define a relation for a point being on a line
variable (on_line : Point → Line → Prop)

-- Axiom: For any two distinct points, there exists a line passing through both points
axiom line_through_points (P Q : Point) (h : P ≠ Q) : ∃ (l : Line), on_line P l ∧ on_line Q l

-- Axiom: Any line passing through two distinct points is unique
axiom line_uniqueness (P Q : Point) (h : P ≠ Q) (l1 l2 : Line) :
  on_line P l1 ∧ on_line Q l1 → on_line P l2 ∧ on_line Q l2 → l1 = l2

-- Theorem: There exists exactly one line passing through any two distinct points
theorem unique_line_through_points (P Q : Point) (h : P ≠ Q) :
  ∃! (l : Line), on_line P l ∧ on_line Q l :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_points_l1765_176516


namespace NUMINAMATH_CALUDE_equation_solutions_l1765_176597

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧
    x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -4 ∧ x₂ = 1 ∧
    (x₁ + 4)^2 = 5*(x₁ + 4) ∧ (x₂ + 4)^2 = 5*(x₂ + 4)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1765_176597


namespace NUMINAMATH_CALUDE_first_month_sale_is_3435_l1765_176579

/-- Calculates the sale in the first month given the sales for the next 5 months and the average sale -/
def first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale2 + sale3 + sale4 + sale5 + sale6)

/-- The sale in the first month is 3435 given the conditions of the problem -/
theorem first_month_sale_is_3435 :
  first_month_sale 3927 3855 4230 3562 1991 3500 = 3435 := by
sorry

#eval first_month_sale 3927 3855 4230 3562 1991 3500

end NUMINAMATH_CALUDE_first_month_sale_is_3435_l1765_176579


namespace NUMINAMATH_CALUDE_opposite_signs_and_larger_absolute_value_l1765_176501

theorem opposite_signs_and_larger_absolute_value (a b : ℚ) 
  (h1 : a * b < 0) (h2 : a + b > 0) : 
  (a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0) ∧ 
  ((a < 0 ∧ b > 0 → abs b > abs a) ∧ (a > 0 ∧ b < 0 → abs a > abs b)) :=
sorry

end NUMINAMATH_CALUDE_opposite_signs_and_larger_absolute_value_l1765_176501


namespace NUMINAMATH_CALUDE_sin_sum_of_angles_l1765_176504

theorem sin_sum_of_angles (θ φ : ℝ) 
  (h1 : Complex.exp (θ * Complex.I) = (4/5 : ℂ) + (3/5 : ℂ) * Complex.I)
  (h2 : Complex.exp (φ * Complex.I) = -(5/13 : ℂ) + (12/13 : ℂ) * Complex.I) : 
  Real.sin (θ + φ) = 33/65 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_of_angles_l1765_176504


namespace NUMINAMATH_CALUDE_max_m_value_l1765_176582

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x + x^2 - m*x + Real.exp (2 - x)

theorem max_m_value :
  ∃ (m_max : ℝ), m_max = 3 ∧ 
  (∀ (m : ℝ), (∀ (x : ℝ), x > 0 → f m x ≥ 0) → m ≤ m_max) ∧
  (∀ (x : ℝ), x > 0 → f m_max x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l1765_176582


namespace NUMINAMATH_CALUDE_max_value_of_a_l1765_176548

/-- Given that "x^2 + 2x - 3 > 0" is a necessary but not sufficient condition for "x < a",
    prove that the maximum value of a is -3. -/
theorem max_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x < a → x^2 + 2*x - 3 > 0) ∧ 
  (∃ x : ℝ, x^2 + 2*x - 3 > 0 ∧ x ≥ a) →
  a ≤ -3 ∧ ∀ b : ℝ, b > -3 → ¬((∀ x : ℝ, x < b → x^2 + 2*x - 3 > 0) ∧ 
                               (∃ x : ℝ, x^2 + 2*x - 3 > 0 ∧ x ≥ b)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1765_176548


namespace NUMINAMATH_CALUDE_rectangle_to_square_length_l1765_176533

theorem rectangle_to_square_length (width : ℝ) (height : ℝ) (y : ℝ) :
  width = 10 →
  height = 20 →
  (width * height = y * y * 16) →
  y = 5 * Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_length_l1765_176533


namespace NUMINAMATH_CALUDE_quadratic_always_two_roots_l1765_176513

theorem quadratic_always_two_roots (k : ℝ) : 
  let a := (1 : ℝ)
  let b := 2 * k
  let c := k - 1
  let discriminant := b^2 - 4*a*c
  0 < discriminant := by sorry

end NUMINAMATH_CALUDE_quadratic_always_two_roots_l1765_176513


namespace NUMINAMATH_CALUDE_recurrence_sequence_is_natural_l1765_176575

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ+ → ℚ) : Prop :=
  a 2 = 2 ∧ ∀ n : ℕ+, (n - 1) * a (n + 1) - n * a n + 1 = 0

/-- The theorem stating that the sequence is equal to the natural numbers -/
theorem recurrence_sequence_is_natural (a : ℕ+ → ℚ) (h : RecurrenceSequence a) :
    ∀ n : ℕ+, a n = n := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_is_natural_l1765_176575


namespace NUMINAMATH_CALUDE_smaller_angle_is_70_l1765_176512

/-- A parallelogram with one angle exceeding the other by 40 degrees -/
structure Parallelogram40 where
  -- The measure of the smaller angle
  small_angle : ℝ
  -- The measure of the larger angle
  large_angle : ℝ
  -- The larger angle exceeds the smaller by 40 degrees
  angle_difference : large_angle = small_angle + 40
  -- Adjacent angles are supplementary (sum to 180 degrees)
  supplementary : small_angle + large_angle = 180

/-- The smaller angle in a Parallelogram40 measures 70 degrees -/
theorem smaller_angle_is_70 (p : Parallelogram40) : p.small_angle = 70 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_is_70_l1765_176512


namespace NUMINAMATH_CALUDE_equal_roots_condition_l1765_176534

theorem equal_roots_condition (m : ℝ) : 
  (∃ (x : ℝ), (x * (x - 3) - (m + 2)) / ((x - 2) * (m - 2)) = x / m) ∧ 
  (∀ (x y : ℝ), (x * (x - 3) - (m + 2)) / ((x - 2) * (m - 2)) = x / m ∧ 
                 (y * (y - 3) - (m + 2)) / ((y - 2) * (m - 2)) = y / m 
                 → x = y) ↔ 
  m = (-7 + Real.sqrt 2) / 2 ∨ m = (-7 - Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l1765_176534


namespace NUMINAMATH_CALUDE_coin_flip_sequences_l1765_176549

theorem coin_flip_sequences (n : ℕ) : n = 10 → (2 : ℕ) ^ n = 1024 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_sequences_l1765_176549


namespace NUMINAMATH_CALUDE_cos_plus_one_is_pseudo_even_l1765_176543

-- Define the concept of a pseudo-even function
def isPseudoEven (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = f (2 * a - x)

-- State the theorem
theorem cos_plus_one_is_pseudo_even :
  isPseudoEven (λ x => Real.cos (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_cos_plus_one_is_pseudo_even_l1765_176543


namespace NUMINAMATH_CALUDE_min_value_and_existence_l1765_176556

/-- The circle C defined by x^2 + y^2 = x + y where x, y > 0 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = p.1 + p.2 ∧ p.1 > 0 ∧ p.2 > 0}

theorem min_value_and_existence : 
  (∀ p ∈ C, 1 / p.1 + 1 / p.2 ≥ 2) ∧ 
  (∃ p ∈ C, (p.1 + 1) * (p.2 + 1) = 4) := by
sorry

end NUMINAMATH_CALUDE_min_value_and_existence_l1765_176556


namespace NUMINAMATH_CALUDE_maria_remaining_towels_l1765_176542

def green_towels : ℕ := 35
def white_towels : ℕ := 21
def towels_given_to_mother : ℕ := 34

theorem maria_remaining_towels :
  green_towels + white_towels - towels_given_to_mother = 22 :=
by sorry

end NUMINAMATH_CALUDE_maria_remaining_towels_l1765_176542


namespace NUMINAMATH_CALUDE_no_solution_for_digit_difference_l1765_176596

theorem no_solution_for_digit_difference : 
  ¬ ∃ (x : ℕ), x < 10 ∧ 
    (max (max (max x 3) 1) 4 * 1000 + 
     max (max (min x 3) 1) 4 * 100 + 
     min (min (max x 3) 1) 4 * 10 + 
     min (min (min x 3) 1) 4) - 
    (min (min (min x 3) 1) 4 * 1000 + 
     min (min (max x 3) 1) 4 * 100 + 
     max (max (min x 3) 1) 4 * 10 + 
     max (max (max x 3) 1) 4) = 4086 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_digit_difference_l1765_176596


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1765_176569

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℤ)
  (h_arith : ArithmeticSequence a)
  (h_diff : a 3 - a 2 = -2)
  (h_seventh : a 7 = -2) :
  a 9 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1765_176569


namespace NUMINAMATH_CALUDE_inverse_implies_negation_l1765_176518

theorem inverse_implies_negation (P : Prop) :
  (¬P → False) → (¬P) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_implies_negation_l1765_176518


namespace NUMINAMATH_CALUDE_erica_earnings_l1765_176586

def fish_price : ℕ := 20
def past_four_months_catch : ℕ := 80
def monthly_maintenance : ℕ := 50
def fuel_cost_per_kg : ℕ := 2
def num_months : ℕ := 5

def total_catch : ℕ := past_four_months_catch * 3

def total_income : ℕ := total_catch * fish_price

def total_maintenance_cost : ℕ := monthly_maintenance * num_months

def total_fuel_cost : ℕ := fuel_cost_per_kg * total_catch

def total_cost : ℕ := total_maintenance_cost + total_fuel_cost

def net_income : ℤ := total_income - total_cost

theorem erica_earnings : net_income = 4070 := by
  sorry

end NUMINAMATH_CALUDE_erica_earnings_l1765_176586


namespace NUMINAMATH_CALUDE_dynaco_shares_sold_is_150_l1765_176524

/-- Represents the stock portfolio problem --/
structure StockPortfolio where
  microtron_price : ℝ
  dynaco_price : ℝ
  total_shares : ℕ
  average_price : ℝ

/-- Calculates the number of Dynaco shares sold --/
def dynaco_shares_sold (portfolio : StockPortfolio) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, 150 Dynaco shares were sold --/
theorem dynaco_shares_sold_is_150 : 
  let portfolio := StockPortfolio.mk 36 44 300 40
  dynaco_shares_sold portfolio = 150 := by
  sorry

end NUMINAMATH_CALUDE_dynaco_shares_sold_is_150_l1765_176524


namespace NUMINAMATH_CALUDE_fraction_multiplication_addition_l1765_176503

theorem fraction_multiplication_addition : (1/3 : ℚ) * (2/5 : ℚ) + (1/4 : ℚ) = 23/60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_addition_l1765_176503


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l1765_176510

theorem rectangle_side_ratio (a b c d : ℝ) (h1 : a * b / (c * d) = 0.16) (h2 : b / d = 2 / 5) :
  a / c = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l1765_176510


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_80_l1765_176526

theorem last_three_digits_of_7_to_80 : 7^80 ≡ 961 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_80_l1765_176526


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1765_176511

/-- Given a quadratic function f(x) = ax² + bx + c with a > 0, and roots α and β of f(x) = x 
    where 0 < α < β, prove that x < f(x) for all x such that 0 < x < α -/
theorem quadratic_inequality (a b c α β : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : a > 0)
  (h3 : f α = α)
  (h4 : f β = β)
  (h5 : 0 < α)
  (h6 : α < β) :
  ∀ x, 0 < x → x < α → x < f x :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1765_176511


namespace NUMINAMATH_CALUDE_sum_of_3rd_4th_5th_terms_l1765_176587

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem sum_of_3rd_4th_5th_terms
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_common_ratio : q = 2)
  (h_sum_first_3 : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
sorry

end NUMINAMATH_CALUDE_sum_of_3rd_4th_5th_terms_l1765_176587


namespace NUMINAMATH_CALUDE_inequality_solution_l1765_176500

theorem inequality_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) (h3 : x ≠ 4) (h4 : x ≠ 5) :
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 24) ↔
  (x < 1 ∨ (4 < x ∧ x < 5) ∨ 6 < x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1765_176500


namespace NUMINAMATH_CALUDE_ron_book_picks_l1765_176507

/-- Represents the number of times a person gets to pick a book in a year -/
def picks_per_year (total_members : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weeks_per_year / total_members

/-- The book club scenario -/
theorem ron_book_picks :
  let couples := 3
  let singles := 5
  let ron_and_wife := 2
  let total_members := couples * 2 + singles + ron_and_wife
  let weeks_per_year := 52
  picks_per_year total_members weeks_per_year = 4 := by
sorry

end NUMINAMATH_CALUDE_ron_book_picks_l1765_176507


namespace NUMINAMATH_CALUDE_expression_simplification_l1765_176529

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - 3 / (x + 2)) / ((x^2 - 1) / (x + 2)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1765_176529


namespace NUMINAMATH_CALUDE_tom_final_coin_value_l1765_176598

/-- Represents the types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- Calculates the total value of a collection of coins --/
def totalValue (coins : List (Coin × ℕ)) : ℕ :=
  coins.foldl (fun acc (c, n) => acc + n * coinValue c) 0

/-- Tom's initial coins --/
def initialCoins : List (Coin × ℕ) :=
  [(Coin.Penny, 27), (Coin.Dime, 15), (Coin.Quarter, 9), (Coin.HalfDollar, 2)]

/-- Coins given by dad --/
def coinsFromDad : List (Coin × ℕ) :=
  [(Coin.Dime, 33), (Coin.Nickel, 49), (Coin.Quarter, 7), (Coin.HalfDollar, 4)]

/-- Coins spent by Tom --/
def spentCoins : List (Coin × ℕ) :=
  [(Coin.Dime, 11), (Coin.Quarter, 5)]

/-- Number of half dollars exchanged for quarters --/
def exchangedHalfDollars : ℕ := 5

/-- Theorem stating the final value of Tom's coins --/
theorem tom_final_coin_value :
  totalValue initialCoins +
  totalValue coinsFromDad -
  totalValue spentCoins +
  exchangedHalfDollars * 2 * coinValue Coin.Quarter =
  1702 := by sorry

end NUMINAMATH_CALUDE_tom_final_coin_value_l1765_176598


namespace NUMINAMATH_CALUDE_angle_with_complement_one_third_of_supplement_l1765_176521

theorem angle_with_complement_one_third_of_supplement (x : Real) : 
  (90 - x = (1 / 3) * (180 - x)) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_one_third_of_supplement_l1765_176521


namespace NUMINAMATH_CALUDE_count_students_without_A_l1765_176523

/-- The number of students who did not receive an A in any subject -/
def students_without_A (total_students : ℕ) (history_A : ℕ) (math_A : ℕ) (science_A : ℕ) 
  (math_history_A : ℕ) (history_science_A : ℕ) (science_math_A : ℕ) (all_subjects_A : ℕ) : ℕ :=
  total_students - (history_A + math_A + science_A - math_history_A - history_science_A - science_math_A + all_subjects_A)

theorem count_students_without_A :
  students_without_A 50 9 15 12 5 3 4 1 = 28 := by
  sorry

end NUMINAMATH_CALUDE_count_students_without_A_l1765_176523


namespace NUMINAMATH_CALUDE_complement_of_sqrt_range_l1765_176528

-- Define the universal set U as ℝ
def U := ℝ

-- Define the set A as the range of y = x^(1/2)
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt x}

-- State the theorem
theorem complement_of_sqrt_range :
  Set.compl A = Set.Iio (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_complement_of_sqrt_range_l1765_176528


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1765_176562

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k - 2) * x₁^2 - 2 * x₁ + (1/2) = 0 ∧ 
    (k - 2) * x₂^2 - 2 * x₂ + (1/2) = 0) ↔ 
  (k < 4 ∧ k ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1765_176562


namespace NUMINAMATH_CALUDE_second_price_reduction_l1765_176519

theorem second_price_reduction (P : ℝ) (x : ℝ) (h1 : P > 0) :
  (P - 0.25 * P) * (1 - x / 100) = P * (1 - 0.7) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_second_price_reduction_l1765_176519


namespace NUMINAMATH_CALUDE_tower_height_ratio_l1765_176566

theorem tower_height_ratio :
  ∀ (grace_height clyde_height : ℕ),
    grace_height = 40 →
    grace_height = clyde_height + 35 →
    (grace_height : ℚ) / (clyde_height : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_tower_height_ratio_l1765_176566


namespace NUMINAMATH_CALUDE_ruble_combinations_l1765_176583

theorem ruble_combinations : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 5 * p.1 + 3 * p.2 = 78) 
    (Finset.product (Finset.range 79) (Finset.range 79))).card ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ruble_combinations_l1765_176583


namespace NUMINAMATH_CALUDE_increasing_geometric_sequence_exists_l1765_176567

theorem increasing_geometric_sequence_exists : ∃ (a : ℕ → ℝ), 
  (∀ n : ℕ, a (n + 1) > a n) ∧  -- increasing
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) ∧  -- geometric
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 4 ∧  -- first three terms
  a 2 + a 3 = 6 * a 1  -- given condition
:= by sorry

end NUMINAMATH_CALUDE_increasing_geometric_sequence_exists_l1765_176567


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1765_176592

open Real

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ > a, log x₁ + 1/x₁ ≥ x₂ + 1/(x₂ - a)) → 
  a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1765_176592


namespace NUMINAMATH_CALUDE_expression_bounds_l1765_176570

theorem expression_bounds (x y z w : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ 
    Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ∧
  Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l1765_176570


namespace NUMINAMATH_CALUDE_triathlon_bike_speed_l1765_176588

def triathlon_speed (swim_distance : ℚ) (bike_distance : ℚ) (run_distance : ℚ)
                    (swim_speed : ℚ) (run_speed : ℚ) (total_time : ℚ) : ℚ :=
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let bike_time := total_time - swim_time - run_time
  bike_distance / bike_time

theorem triathlon_bike_speed :
  triathlon_speed (1/2) 10 2 (3/2) 5 (3/2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_bike_speed_l1765_176588


namespace NUMINAMATH_CALUDE_first_grade_sample_size_l1765_176520

/-- Given a total sample size and ratios for three groups, 
    calculate the number of samples for the first group -/
def stratifiedSampleSize (totalSample : ℕ) (ratio1 ratio2 ratio3 : ℕ) : ℕ :=
  (ratio1 * totalSample) / (ratio1 + ratio2 + ratio3)

/-- Theorem: For a total sample of 80 and ratios 4:3:3, 
    the first group's sample size is 32 -/
theorem first_grade_sample_size :
  stratifiedSampleSize 80 4 3 3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_first_grade_sample_size_l1765_176520


namespace NUMINAMATH_CALUDE_sum_of_two_squares_l1765_176574

theorem sum_of_two_squares (x y : ℝ) : 2 * x^2 + 2 * y^2 = (x + y)^2 + (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_l1765_176574


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_less_than_one_l1765_176553

theorem quadratic_roots_imply_a_less_than_one (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_less_than_one_l1765_176553


namespace NUMINAMATH_CALUDE_siblings_ages_l1765_176576

/-- Represents the ages of the siblings -/
structure SiblingAges where
  richard : ℕ
  david : ℕ
  scott : ℕ
  emily : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : SiblingAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.david = ages.scott + 8 ∧
  ages.emily = ages.richard - 5 ∧
  ages.richard + 8 = 2 * (ages.scott + 8)

/-- The theorem to be proved -/
theorem siblings_ages : 
  ∃ (ages : SiblingAges), satisfies_conditions ages ∧ 
    ages.richard = 20 ∧ ages.david = 14 ∧ ages.scott = 6 ∧ ages.emily = 15 := by
  sorry

end NUMINAMATH_CALUDE_siblings_ages_l1765_176576


namespace NUMINAMATH_CALUDE_percentage_difference_l1765_176560

theorem percentage_difference (z y x : ℝ) (total : ℝ) : 
  y = 1.2 * z →
  z = 300 →
  total = 1110 →
  x = total - y - z →
  (x - y) / y * 100 = 25 :=
by sorry

end NUMINAMATH_CALUDE_percentage_difference_l1765_176560


namespace NUMINAMATH_CALUDE_regular_octagon_side_length_l1765_176538

/-- A regular octagon with a perimeter of 23.6 cm has sides of length 2.95 cm. -/
theorem regular_octagon_side_length : 
  ∀ (perimeter side_length : ℝ),
  perimeter = 23.6 →
  perimeter = 8 * side_length →
  side_length = 2.95 := by
sorry

end NUMINAMATH_CALUDE_regular_octagon_side_length_l1765_176538


namespace NUMINAMATH_CALUDE_benny_spent_85_dollars_l1765_176545

def baseball_gear_total (glove_price baseball_price bat_price helmet_price gloves_price : ℕ) : ℕ :=
  glove_price + baseball_price + bat_price + helmet_price + gloves_price

theorem benny_spent_85_dollars : 
  baseball_gear_total 25 5 30 15 10 = 85 := by
  sorry

end NUMINAMATH_CALUDE_benny_spent_85_dollars_l1765_176545


namespace NUMINAMATH_CALUDE_square_grid_15_toothpicks_l1765_176517

/-- Calculates the total number of toothpicks needed for a square grid -/
def toothpicks_in_square_grid (side_length : ℕ) : ℕ :=
  2 * (side_length + 1) * side_length

/-- Theorem: A square grid with 15 toothpicks on each side requires 480 toothpicks -/
theorem square_grid_15_toothpicks :
  toothpicks_in_square_grid 15 = 480 := by
  sorry

#eval toothpicks_in_square_grid 15

end NUMINAMATH_CALUDE_square_grid_15_toothpicks_l1765_176517


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_seven_halves_l1765_176552

theorem sqrt_expression_equals_seven_halves :
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 / Real.sqrt 24 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_seven_halves_l1765_176552


namespace NUMINAMATH_CALUDE_seven_ways_to_make_eight_cents_l1765_176599

/-- Represents the number of ways to make a certain amount with given coins -/
def num_ways_to_make_amount (one_cent : ℕ) (two_cent : ℕ) (five_cent : ℕ) (target : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 7 ways to make 8 cents with the given coins -/
theorem seven_ways_to_make_eight_cents :
  num_ways_to_make_amount 8 4 1 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_ways_to_make_eight_cents_l1765_176599


namespace NUMINAMATH_CALUDE_point_not_on_line_l1765_176509

theorem point_not_on_line (p q : ℝ) (h : p * q > 0) :
  ¬(∃ (x y : ℝ), x = 2023 ∧ y = 0 ∧ y = p * x + q) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l1765_176509


namespace NUMINAMATH_CALUDE_infinite_lcm_greater_than_ck_l1765_176541

theorem infinite_lcm_greater_than_ck 
  (a : ℕ → ℕ) 
  (c : ℝ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_positive : ∀ n, a n > 0) 
  (h_c : 0 < c ∧ c < 1.5) : 
  ∀ N, ∃ k > N, Nat.lcm (a k) (a (k + 1)) > ⌊c * k⌋ := by
  sorry

end NUMINAMATH_CALUDE_infinite_lcm_greater_than_ck_l1765_176541


namespace NUMINAMATH_CALUDE_unique_number_with_conditions_l1765_176540

theorem unique_number_with_conditions : ∃! n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ 
  2 ∣ n ∧
  3 ∣ (n + 1) ∧
  4 ∣ (n + 2) ∧
  5 ∣ (n + 3) ∧
  n = 62 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_conditions_l1765_176540


namespace NUMINAMATH_CALUDE_least_cans_proof_l1765_176590

/-- The number of liters of Maaza -/
def maaza_liters : ℕ := 50

/-- The number of liters of Pepsi -/
def pepsi_liters : ℕ := 144

/-- The number of liters of Sprite -/
def sprite_liters : ℕ := 368

/-- The least number of cans required to pack all drinks -/
def least_cans : ℕ := 281

/-- Theorem stating that the least number of cans required is 281 -/
theorem least_cans_proof :
  ∃ (can_size : ℕ), can_size > 0 ∧
  maaza_liters % can_size = 0 ∧
  pepsi_liters % can_size = 0 ∧
  sprite_liters % can_size = 0 ∧
  least_cans = maaza_liters / can_size + pepsi_liters / can_size + sprite_liters / can_size ∧
  ∀ (other_size : ℕ), other_size > 0 →
    maaza_liters % other_size = 0 →
    pepsi_liters % other_size = 0 →
    sprite_liters % other_size = 0 →
    least_cans ≤ maaza_liters / other_size + pepsi_liters / other_size + sprite_liters / other_size :=
by
  sorry

end NUMINAMATH_CALUDE_least_cans_proof_l1765_176590


namespace NUMINAMATH_CALUDE_stating_average_enter_exit_time_l1765_176527

/-- Represents the speed of the car in miles per minute -/
def car_speed : ℚ := 5/4

/-- Represents the speed of the storm in miles per minute -/
def storm_speed : ℚ := 1/2

/-- Represents the radius of the storm in miles -/
def storm_radius : ℚ := 51

/-- Represents the initial y-coordinate of the storm center in miles -/
def initial_storm_y : ℚ := 110

/-- 
Theorem stating that the average time at which the car enters and exits the storm is 880/29 minutes
-/
theorem average_enter_exit_time : 
  let car_pos (t : ℚ) := (car_speed * t, 0)
  let storm_center (t : ℚ) := (0, initial_storm_y - storm_speed * t)
  let distance (t : ℚ) := 
    ((car_pos t).1 - (storm_center t).1)^2 + ((car_pos t).2 - (storm_center t).2)^2
  ∃ t₁ t₂,
    distance t₁ = storm_radius^2 ∧ 
    distance t₂ = storm_radius^2 ∧ 
    t₁ < t₂ ∧
    (t₁ + t₂) / 2 = 880 / 29 :=
sorry

end NUMINAMATH_CALUDE_stating_average_enter_exit_time_l1765_176527


namespace NUMINAMATH_CALUDE_triangle_interior_angle_ratio_l1765_176546

theorem triangle_interior_angle_ratio 
  (α β γ : ℝ) 
  (h1 : 2 * α + 3 * β = 4 * γ) 
  (h2 : α = 4 * β - γ) :
  ∃ (k : ℝ), k > 0 ∧ 
    2 * k = 180 - α ∧
    9 * k = 180 - β ∧
    4 * k = 180 - γ := by
sorry

end NUMINAMATH_CALUDE_triangle_interior_angle_ratio_l1765_176546


namespace NUMINAMATH_CALUDE_no_fraternity_member_is_club_member_l1765_176573

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (student : U → Prop)
variable (club_member : U → Prop)
variable (fraternity_member : U → Prop)
variable (honest : U → Prop)

-- State the theorem
theorem no_fraternity_member_is_club_member
  (h1 : ∀ x, club_member x → student x)
  (h2 : ∀ x, club_member x → ¬honest x)
  (h3 : ∀ x, fraternity_member x → honest x) :
  ∀ x, fraternity_member x → ¬club_member x :=
by
  sorry


end NUMINAMATH_CALUDE_no_fraternity_member_is_club_member_l1765_176573


namespace NUMINAMATH_CALUDE_problem_solution_l1765_176531

theorem problem_solution (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^2 + 1/a^2 + a^3 + 1/a^3 = 3 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1765_176531


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1765_176550

theorem fraction_sum_equality (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2*x - 5) / (x^2 - 1) + 3 / (1 - x) = -(x + 8) / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1765_176550


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1765_176585

theorem complex_fraction_simplification :
  (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1765_176585


namespace NUMINAMATH_CALUDE_triangle_problem_l1765_176593

theorem triangle_problem (A B C : Real) (a b c : Real) :
  0 < A → A < π / 2 →
  (1 / 2) * b * c * Real.sin A = (Real.sqrt 3 / 4) * b * c →
  c / b = 1 / 2 + Real.sqrt 3 →
  A = π / 3 ∧ Real.tan B = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1765_176593
