import Mathlib

namespace NUMINAMATH_CALUDE_triangle_theorem_l1247_124759

noncomputable def triangle_proof (a b c : ℝ) (A B C : ℝ) : Prop :=
  let S := (33 : ℝ) / 2
  3 * a = 5 * c * Real.sin A ∧
  Real.cos B = -(5 : ℝ) / 13 ∧
  S = (1 / 2) * a * c * Real.sin B →
  Real.sin A = (33 : ℝ) / 65 ∧
  b = 10

theorem triangle_theorem :
  ∀ (a b c : ℝ) (A B C : ℝ),
  triangle_proof a b c A B C :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1247_124759


namespace NUMINAMATH_CALUDE_fourth_month_sale_l1247_124724

def sales_problem (sale1 sale2 sale3 sale5 sale6_target average_target : ℕ) : Prop :=
  let total_sales := 6 * average_target
  let known_sales := sale1 + sale2 + sale3 + sale5 + sale6_target
  let sale4 := total_sales - known_sales
  sale4 = 6350

theorem fourth_month_sale :
  sales_problem 5420 5660 6200 6500 7070 6200 :=
by sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l1247_124724


namespace NUMINAMATH_CALUDE_expand_product_l1247_124733

theorem expand_product (x : ℝ) : 
  (2 * x^3 - 3 * x + 1) * (x^2 + 4 * x + 3) = 
  2 * x^5 + 8 * x^4 + 3 * x^3 - 11 * x^2 - 5 * x + 3 := by
sorry

end NUMINAMATH_CALUDE_expand_product_l1247_124733


namespace NUMINAMATH_CALUDE_sprint_tournament_races_l1247_124789

/-- Calculates the number of races needed to determine a winner in a sprint tournament. -/
def races_needed (total_athletes : ℕ) (runners_per_race : ℕ) (advancing_per_race : ℕ) : ℕ :=
  sorry

/-- The sprint tournament problem -/
theorem sprint_tournament_races (total_athletes : ℕ) (runners_per_race : ℕ) (advancing_per_race : ℕ) 
  (h1 : total_athletes = 300)
  (h2 : runners_per_race = 8)
  (h3 : advancing_per_race = 2) :
  races_needed total_athletes runners_per_race advancing_per_race = 53 :=
by sorry

end NUMINAMATH_CALUDE_sprint_tournament_races_l1247_124789


namespace NUMINAMATH_CALUDE_distance_calculation_l1247_124762

theorem distance_calculation (speed : ℝ) (time : ℝ) (h1 : speed = 100) (h2 : time = 5) :
  speed * time = 500 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l1247_124762


namespace NUMINAMATH_CALUDE_limit_fraction_binomial_sums_l1247_124723

def a (n : ℕ+) : ℝ := (3 : ℝ) ^ n.val
def b (n : ℕ+) : ℝ := (2 : ℝ) ^ n.val

theorem limit_fraction_binomial_sums :
  ∀ ε > 0, ∃ N : ℕ+, ∀ n ≥ N,
    |((b (n + 1) - a n) / (a (n + 1) + b n)) + (1 / 3)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_fraction_binomial_sums_l1247_124723


namespace NUMINAMATH_CALUDE_equal_wealth_after_transfer_l1247_124712

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.foma - 70 = w.ierema + 70) ∧ 
  (w.foma - 40 = w.yuliy)

/-- The theorem to be proved -/
theorem equal_wealth_after_transfer (w : MerchantWealth) 
  (h : problem_conditions w) : 
  w.foma - 55 = w.ierema + 55 := by
  sorry

end NUMINAMATH_CALUDE_equal_wealth_after_transfer_l1247_124712


namespace NUMINAMATH_CALUDE_jacket_price_calculation_l1247_124781

/-- Calculates the final price of a jacket after discount and tax --/
def finalPrice (originalPrice : ℝ) (discountRate : ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  discountedPrice * (1 + taxRate)

/-- Theorem stating that the final price of the jacket is 92.4 --/
theorem jacket_price_calculation :
  finalPrice 120 0.3 0.1 = 92.4 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_calculation_l1247_124781


namespace NUMINAMATH_CALUDE_log_stack_total_l1247_124764

/-- The sum of an arithmetic sequence with 15 terms, starting at 15 and ending at 1 -/
def log_stack_sum : ℕ := 
  let first_term := 15
  let last_term := 1
  let num_terms := 15
  (num_terms * (first_term + last_term)) / 2

/-- The total number of logs in the stack is 120 -/
theorem log_stack_total : log_stack_sum = 120 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_total_l1247_124764


namespace NUMINAMATH_CALUDE_mode_of_data_set_l1247_124710

def data_set : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_data_set :
  mode data_set = 3 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_data_set_l1247_124710


namespace NUMINAMATH_CALUDE_rectangle_segment_product_l1247_124738

theorem rectangle_segment_product (AB BC CD DE x : ℝ) : 
  AB = 5 →
  BC = 11 →
  CD = 3 →
  DE = 9 →
  0 < x →
  x < DE →
  AB * (AB + BC + CD + x) = x * (DE - x) →
  x = 11.95 := by
sorry

end NUMINAMATH_CALUDE_rectangle_segment_product_l1247_124738


namespace NUMINAMATH_CALUDE_bombardier_solution_l1247_124704

/-- Represents the number of bombs thrown by each bombardier -/
structure BombardierShots where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Defines the conditions of the bombardier problem -/
def satisfiesConditions (shots : BombardierShots) : Prop :=
  (shots.first + shots.second = shots.third + 26) ∧
  (shots.second + shots.third = shots.first + shots.second + 38) ∧
  (shots.first + shots.third = shots.second + 24)

/-- Theorem stating the solution to the bombardier problem -/
theorem bombardier_solution :
  ∃ (shots : BombardierShots), satisfiesConditions shots ∧
    shots.first = 25 ∧ shots.second = 64 ∧ shots.third = 63 := by
  sorry

end NUMINAMATH_CALUDE_bombardier_solution_l1247_124704


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1247_124769

theorem quadratic_equation_roots (a b c : ℝ) (h : a = 2 ∧ b = -3 ∧ c = 1) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1247_124769


namespace NUMINAMATH_CALUDE_larger_cuboid_length_l1247_124707

/-- Proves that the length of a larger cuboid is 12 meters, given its width, height, and the number and dimensions of smaller cuboids it can be divided into. -/
theorem larger_cuboid_length (width height : ℝ) (num_small_cuboids : ℕ) 
  (small_length small_width small_height : ℝ) : 
  width = 14 →
  height = 10 →
  num_small_cuboids = 56 →
  small_length = 5 →
  small_width = 3 →
  small_height = 2 →
  (width * height * (num_small_cuboids * small_length * small_width * small_height) / (width * height)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_larger_cuboid_length_l1247_124707


namespace NUMINAMATH_CALUDE_rectangle_to_square_transformation_l1247_124796

theorem rectangle_to_square_transformation (a b : ℝ) : 
  a > 0 → b > 0 → a * b = 54 → (3 * a) * (b / 2) = 9^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_transformation_l1247_124796


namespace NUMINAMATH_CALUDE_find_number_given_hcf_lcm_l1247_124798

/-- Given two positive integers with specific HCF and LCM, prove that one is 24 if the other is 169 -/
theorem find_number_given_hcf_lcm (A B : ℕ+) : 
  (Nat.gcd A B = 13) →
  (Nat.lcm A B = 312) →
  (B = 169) →
  A = 24 := by
sorry

end NUMINAMATH_CALUDE_find_number_given_hcf_lcm_l1247_124798


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1247_124715

theorem arithmetic_mean_problem (p q r : ℝ) : 
  (p + q) / 2 = 10 → 
  (q + r) / 2 = 22 → 
  r - p = 24 → 
  (q + r) / 2 = 22 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1247_124715


namespace NUMINAMATH_CALUDE_line_and_circle_equations_l1247_124761

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Theorem statement
theorem line_and_circle_equations :
  ∀ (x y : ℝ),
  (∃ (t : ℝ), x = 2 + 4*t ∧ y = 1 + 2*t) →  -- Line l passes through (2, 1) and (6, 3)
  (∃ (a : ℝ), line_l (2*a) a ∧ circle_C (2*a) a) →  -- Circle C's center lies on line l
  circle_C 2 0 →  -- Circle C is tangent to x-axis at (2, 0)
  (line_l x y ↔ x - 2*y = 0) ∧  -- Equation of line l
  (circle_C x y ↔ (x - 2)^2 + (y - 1)^2 = 1)  -- Equation of circle C
  := by sorry

end NUMINAMATH_CALUDE_line_and_circle_equations_l1247_124761


namespace NUMINAMATH_CALUDE_number_of_proper_subsets_l1247_124793

def U : Finset Nat := {0, 1, 2, 3}

def A : Finset Nat := {0, 1, 3}

def complement_A : Finset Nat := {2}

theorem number_of_proper_subsets :
  (U = {0, 1, 2, 3}) →
  (complement_A = {2}) →
  (A = U \ complement_A) →
  (Finset.powerset A).card - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_proper_subsets_l1247_124793


namespace NUMINAMATH_CALUDE_simplify_expression_l1247_124746

theorem simplify_expression : 20 + (-14) - (-18) + 13 = 37 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1247_124746


namespace NUMINAMATH_CALUDE_fish_problem_l1247_124716

theorem fish_problem (total : ℕ) (carla_fish : ℕ) (kyle_fish : ℕ) (tasha_fish : ℕ) :
  total = 36 →
  carla_fish = 8 →
  kyle_fish = tasha_fish →
  total = carla_fish + kyle_fish + tasha_fish →
  kyle_fish = 14 := by
  sorry

end NUMINAMATH_CALUDE_fish_problem_l1247_124716


namespace NUMINAMATH_CALUDE_smallest_yellow_candy_count_l1247_124742

/-- The cost of a piece of yellow candy in cents -/
def yellow_candy_cost : ℕ := 15

/-- The number of red candies Joe can buy -/
def red_candy_count : ℕ := 10

/-- The number of green candies Joe can buy -/
def green_candy_count : ℕ := 16

/-- The number of blue candies Joe can buy -/
def blue_candy_count : ℕ := 18

theorem smallest_yellow_candy_count :
  ∃ n : ℕ, n > 0 ∧
  (yellow_candy_cost * n) % red_candy_count = 0 ∧
  (yellow_candy_cost * n) % green_candy_count = 0 ∧
  (yellow_candy_cost * n) % blue_candy_count = 0 ∧
  (∀ m : ℕ, m > 0 →
    (yellow_candy_cost * m) % red_candy_count = 0 →
    (yellow_candy_cost * m) % green_candy_count = 0 →
    (yellow_candy_cost * m) % blue_candy_count = 0 →
    m ≥ n) ∧
  n = 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_yellow_candy_count_l1247_124742


namespace NUMINAMATH_CALUDE_second_discount_percentage_l1247_124747

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 200 →
  first_discount = 20 →
  final_price = 152 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 5 :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l1247_124747


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1247_124701

theorem complex_equation_solution (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + i) * i = b + i →
  a = 1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1247_124701


namespace NUMINAMATH_CALUDE_anna_initial_stamps_l1247_124777

theorem anna_initial_stamps (x : ℕ) (alison_stamps : ℕ) : 
  alison_stamps = 28 → 
  x + alison_stamps / 2 = 50 → 
  x = 36 := by
sorry

end NUMINAMATH_CALUDE_anna_initial_stamps_l1247_124777


namespace NUMINAMATH_CALUDE_toy_price_after_discounts_l1247_124757

theorem toy_price_after_discounts (initial_price : ℝ) (discount : ℝ) : 
  initial_price = 200 → discount = 0.1 → 
  initial_price * (1 - discount)^2 = 162 := by
  sorry

#eval (200 : ℝ) * (1 - 0.1)^2

end NUMINAMATH_CALUDE_toy_price_after_discounts_l1247_124757


namespace NUMINAMATH_CALUDE_function_and_range_proof_l1247_124749

-- Define the function f
def f (x : ℝ) (b c : ℝ) : ℝ := 2 * x^2 + b * x + c

-- State the theorem
theorem function_and_range_proof :
  ∀ b c : ℝ,
  (∀ x : ℝ, f x b c < 0 ↔ 1 < x ∧ x < 5) →
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → ∃ t : ℝ, f x b c ≤ 2 + t) →
  (∀ x : ℝ, f x b c = 2 * x^2 - 12 * x + 10) ∧
  (∀ t : ℝ, t ≥ -10 ↔ ∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ f x b c ≤ 2 + t) :=
by sorry

end NUMINAMATH_CALUDE_function_and_range_proof_l1247_124749


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1247_124756

theorem quadratic_root_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1247_124756


namespace NUMINAMATH_CALUDE_color_cartridge_cost_l1247_124721

/-- The cost of each color cartridge given the total cost, number of cartridges, and cost of black-and-white cartridge. -/
theorem color_cartridge_cost (total_cost : ℕ) (bw_cost : ℕ) (num_color : ℕ) : 
  total_cost = bw_cost + num_color * 32 → 32 = (total_cost - bw_cost) / num_color :=
by sorry

end NUMINAMATH_CALUDE_color_cartridge_cost_l1247_124721


namespace NUMINAMATH_CALUDE_quick_response_solution_l1247_124766

def quick_response_problem (x y z : ℕ) : Prop :=
  5 * x + 4 * y + 3 * z = 15 ∧ 
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1)

theorem quick_response_solution :
  ∀ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 → quick_response_problem x y z :=
by
  sorry

#check quick_response_solution

end NUMINAMATH_CALUDE_quick_response_solution_l1247_124766


namespace NUMINAMATH_CALUDE_ryan_sandwiches_l1247_124736

def slices_per_sandwich : ℕ := 3
def total_slices : ℕ := 15

theorem ryan_sandwiches :
  total_slices / slices_per_sandwich = 5 := by sorry

end NUMINAMATH_CALUDE_ryan_sandwiches_l1247_124736


namespace NUMINAMATH_CALUDE_central_angle_twice_inscribed_l1247_124740

/-- A circle with a diameter and a point on its circumference -/
structure CircleWithDiameterAndPoint where
  /-- The center of the circle -/
  O : ℝ × ℝ
  /-- One end of the diameter -/
  A : ℝ × ℝ
  /-- The other end of the diameter -/
  B : ℝ × ℝ
  /-- An arbitrary point on the circle -/
  C : ℝ × ℝ
  /-- AB is a diameter -/
  diameter : dist O A = dist O B
  /-- C is on the circle -/
  on_circle : dist O C = dist O A

/-- The angle between two vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- The theorem: Central angle COB is twice the inscribed angle CAB -/
theorem central_angle_twice_inscribed 
  (circle : CircleWithDiameterAndPoint) : 
  angle (circle.C - circle.O) (circle.B - circle.O) = 
  2 * angle (circle.C - circle.A) (circle.B - circle.A) := by
  sorry

end NUMINAMATH_CALUDE_central_angle_twice_inscribed_l1247_124740


namespace NUMINAMATH_CALUDE_unfolded_paper_has_symmetric_holes_l1247_124785

/-- Represents a rectangular piece of paper -/
structure Paper :=
  (width : ℝ)
  (height : ℝ)
  (is_rectangular : width > 0 ∧ height > 0)

/-- Represents a hole on the paper -/
structure Hole :=
  (x : ℝ)
  (y : ℝ)

/-- Represents the state of the paper after folding and punching -/
structure FoldedPaper :=
  (original : Paper)
  (hole : Hole)
  (is_folded_left_right : Bool)
  (is_folded_diagonally : Bool)
  (is_hole_near_center : Bool)

/-- Represents the state of the paper after unfolding -/
structure UnfoldedPaper :=
  (original : Paper)
  (holes : List Hole)

/-- Function to unfold the paper -/
def unfold (fp : FoldedPaper) : UnfoldedPaper :=
  sorry

/-- Predicate to check if holes are symmetrically placed -/
def are_holes_symmetric (up : UnfoldedPaper) : Prop :=
  sorry

/-- Main theorem: Unfolding a properly folded and punched paper results in four symmetrically placed holes -/
theorem unfolded_paper_has_symmetric_holes (fp : FoldedPaper) 
  (h1 : fp.is_folded_left_right = true)
  (h2 : fp.is_folded_diagonally = true)
  (h3 : fp.is_hole_near_center = true) :
  let up := unfold fp
  (up.holes.length = 4) ∧ (are_holes_symmetric up) :=
  sorry

end NUMINAMATH_CALUDE_unfolded_paper_has_symmetric_holes_l1247_124785


namespace NUMINAMATH_CALUDE_kath_group_cost_l1247_124728

/-- Calculates the total cost of movie admission for a group, given a regular price, 
    discount amount, and number of people in the group. -/
def total_cost (regular_price discount : ℕ) (group_size : ℕ) : ℕ :=
  (regular_price - discount) * group_size

/-- Proves that the total cost for Kath's group is $30 -/
theorem kath_group_cost : 
  let regular_price : ℕ := 8
  let discount : ℕ := 3
  let kath_siblings : ℕ := 2
  let kath_friends : ℕ := 3
  let group_size : ℕ := 1 + kath_siblings + kath_friends
  total_cost regular_price discount group_size = 30 := by
  sorry

#eval total_cost 8 3 6

end NUMINAMATH_CALUDE_kath_group_cost_l1247_124728


namespace NUMINAMATH_CALUDE_equilibrium_force_l1247_124729

/-- A 2D vector representation --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Addition of two 2D vectors --/
def Vector2D.add (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Negation of a 2D vector --/
def Vector2D.neg (v : Vector2D) : Vector2D :=
  ⟨-v.x, -v.y⟩

/-- Zero 2D vector --/
def Vector2D.zero : Vector2D :=
  ⟨0, 0⟩

theorem equilibrium_force (f₁ f₂ f₃ f₄ : Vector2D) 
    (h₁ : f₁ = ⟨-2, -1⟩) 
    (h₂ : f₂ = ⟨-3, 2⟩)
    (h₃ : f₃ = ⟨4, -3⟩)
    (h₄ : f₄ = ⟨1, 2⟩) :
    Vector2D.add (Vector2D.add (Vector2D.add f₁ f₂) f₃) f₄ = Vector2D.zero := by
  sorry

#check equilibrium_force

end NUMINAMATH_CALUDE_equilibrium_force_l1247_124729


namespace NUMINAMATH_CALUDE_internal_diagonal_cubes_l1247_124771

def cuboid_diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

theorem internal_diagonal_cubes :
  cuboid_diagonal_cubes 168 350 390 = 880 := by
  sorry

end NUMINAMATH_CALUDE_internal_diagonal_cubes_l1247_124771


namespace NUMINAMATH_CALUDE_units_digit_sum_powers_l1247_124779

theorem units_digit_sum_powers : (19^89 + 89^19) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_powers_l1247_124779


namespace NUMINAMATH_CALUDE_max_value_of_a_l1247_124778

theorem max_value_of_a : 
  (∀ x : ℝ, x ≠ 0 → |a - 2| ≤ |x + 1/x|) → 
  ∃ a_max : ℝ, a_max = 4 ∧ ∀ a : ℝ, (∀ x : ℝ, x ≠ 0 → |a - 2| ≤ |x + 1/x|) → a ≤ a_max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1247_124778


namespace NUMINAMATH_CALUDE_cylinder_volume_from_unit_square_l1247_124786

/-- The volume of a cylinder formed by rolling a unit square -/
theorem cylinder_volume_from_unit_square : 
  ∃ (V : ℝ), V = (1 : ℝ) / (4 * Real.pi) ∧ 
  (∃ (r h : ℝ), r = (1 : ℝ) / (2 * Real.pi) ∧ h = 1 ∧ V = Real.pi * r^2 * h) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_unit_square_l1247_124786


namespace NUMINAMATH_CALUDE_lecture_hall_tables_l1247_124751

theorem lecture_hall_tables (total_legs : ℕ) (stools_per_table : ℕ) (stool_legs : ℕ) (table_legs : ℕ) :
  total_legs = 680 →
  stools_per_table = 8 →
  stool_legs = 4 →
  table_legs = 4 →
  (total_legs : ℚ) / ((stools_per_table * stool_legs + table_legs) : ℚ) = 680 / 36 :=
by sorry

end NUMINAMATH_CALUDE_lecture_hall_tables_l1247_124751


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l1247_124775

/-- The solution set of the inequality |2x-a|+a≤4 -/
def SolutionSet (a : ℝ) : Set ℝ := {x : ℝ | |2*x - a| + a ≤ 4}

/-- The theorem stating that if the solution set of |2x-a|+a≤4 is {x|-1≤x≤2}, then a = 1 -/
theorem solution_set_implies_a_equals_one :
  SolutionSet 1 = {x : ℝ | -1 ≤ x ∧ x ≤ 2} → 1 = 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l1247_124775


namespace NUMINAMATH_CALUDE_simplify_expression_l1247_124794

theorem simplify_expression (x y : ℚ) (hx : x = 5) (hy : y = 2) :
  (10 * x * y^3) / (15 * x^2 * y^2) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1247_124794


namespace NUMINAMATH_CALUDE_equation_solution_l1247_124741

theorem equation_solution : ∃ x : ℝ, (27 - 5 = 4 + x) ∧ (x = 18) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1247_124741


namespace NUMINAMATH_CALUDE_orangeade_price_day1_l1247_124731

/-- Represents the price and volume data for orangeade sales over two days -/
structure OrangeadeSales where
  orange_juice : ℝ
  water_day1 : ℝ
  water_day2 : ℝ
  price_day2 : ℝ
  revenue : ℝ

/-- Calculates the price per glass on the first day given orangeade sales data -/
def price_day1 (sales : OrangeadeSales) : ℝ :=
  1.5 * sales.price_day2

/-- Theorem stating that under the given conditions, the price on the first day is $0.30 -/
theorem orangeade_price_day1 (sales : OrangeadeSales) 
  (h1 : sales.water_day1 = sales.orange_juice)
  (h2 : sales.water_day2 = 2 * sales.water_day1)
  (h3 : sales.price_day2 = 0.2)
  (h4 : sales.revenue = (sales.orange_juice + sales.water_day1) * (price_day1 sales))
  (h5 : sales.revenue = (sales.orange_juice + sales.water_day2) * sales.price_day2) :
  price_day1 sales = 0.3 := by
  sorry

#eval price_day1 { orange_juice := 1, water_day1 := 1, water_day2 := 2, price_day2 := 0.2, revenue := 0.6 }

end NUMINAMATH_CALUDE_orangeade_price_day1_l1247_124731


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2016_l1247_124719

/-- The last two digits of 7^n, for n ≥ 1 -/
def lastTwoDigits (n : ℕ) : ℕ :=
  (7^n) % 100

/-- The period of the last two digits of powers of 7 -/
def period : ℕ := 4

theorem last_two_digits_of_7_pow_2016 :
  lastTwoDigits 2016 = 01 :=
by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2016_l1247_124719


namespace NUMINAMATH_CALUDE_parallel_and_perpendicular_relations_l1247_124734

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State that m and n are different lines
variable (m_ne_n : m ≠ n)

-- State that α, β, and γ are different planes
variable (α_ne_β : α ≠ β)
variable (α_ne_γ : α ≠ γ)
variable (β_ne_γ : β ≠ γ)

-- Define the theorem
theorem parallel_and_perpendicular_relations :
  (∀ (a b c : Plane), parallel_planes a c → parallel_planes b c → parallel_planes a b) ∧
  (∀ (l1 l2 : Line) (p1 p2 : Plane), 
    line_perpendicular_to_plane l1 p1 → 
    line_perpendicular_to_plane l2 p2 → 
    parallel_planes p1 p2 → 
    parallel_lines l1 l2) :=
sorry

end NUMINAMATH_CALUDE_parallel_and_perpendicular_relations_l1247_124734


namespace NUMINAMATH_CALUDE_minimal_distance_point_l1247_124773

/-- Given points A and B in ℝ², prove that P(0, 3) on the y-axis minimizes |PA| + |PB| -/
theorem minimal_distance_point (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, -1)) :
  let P : ℝ × ℝ := (0, 3)
  (∀ Q : ℝ × ℝ, Q.1 = 0 → dist A P + dist B P ≤ dist A Q + dist B Q) :=
by sorry


end NUMINAMATH_CALUDE_minimal_distance_point_l1247_124773


namespace NUMINAMATH_CALUDE_dark_tile_fraction_is_16_81_l1247_124795

/-- Represents a tiling system with darker tiles along diagonals -/
structure TilingSystem where
  size : Nat
  corner_size : Nat
  dark_tiles_per_corner : Nat

/-- The fraction of darker tiles in the entire floor -/
def dark_tile_fraction (ts : TilingSystem) : Rat :=
  (4 * ts.dark_tiles_per_corner : Rat) / (ts.size^2 : Rat)

/-- The specific tiling system described in the problem -/
def floor_tiling : TilingSystem :=
  { size := 9
  , corner_size := 4
  , dark_tiles_per_corner := 4 }

/-- Theorem: The fraction of darker tiles in the floor is 16/81 -/
theorem dark_tile_fraction_is_16_81 :
  dark_tile_fraction floor_tiling = 16 / 81 := by
  sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_is_16_81_l1247_124795


namespace NUMINAMATH_CALUDE_cl2_moles_required_l1247_124709

/-- Represents the stoichiometric ratio of Cl2 to CH4 in the reaction -/
def cl2_ch4_ratio : ℚ := 4

/-- Represents the number of moles of CH4 given -/
def ch4_moles : ℚ := 3

/-- Represents the number of moles of CCl4 produced -/
def ccl4_moles : ℚ := 3

/-- Theorem stating that the number of moles of Cl2 required is 12 -/
theorem cl2_moles_required : cl2_ch4_ratio * ch4_moles = 12 := by
  sorry

end NUMINAMATH_CALUDE_cl2_moles_required_l1247_124709


namespace NUMINAMATH_CALUDE_tire_usage_calculation_tire_usage_proof_l1247_124768

/-- Calculates the miles each tire was used given the total distance and tire usage pattern. -/
theorem tire_usage_calculation (total_distance : ℕ) (first_part_distance : ℕ) (second_part_distance : ℕ) 
  (total_tires : ℕ) (tires_used_first_part : ℕ) (tires_used_second_part : ℕ) : ℕ :=
  let total_tire_miles := first_part_distance * tires_used_first_part + second_part_distance * tires_used_second_part
  total_tire_miles / total_tires

/-- Proves that each tire was used for 38,571 miles given the specific conditions of the problem. -/
theorem tire_usage_proof : 
  tire_usage_calculation 50000 40000 10000 7 5 7 = 38571 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_calculation_tire_usage_proof_l1247_124768


namespace NUMINAMATH_CALUDE_AlF3_MgCl2_cell_potential_l1247_124745

/-- Standard reduction potential for Al^3+/Al in volts -/
def E_Al : ℝ := -1.66

/-- Standard reduction potential for Mg^2+/Mg in volts -/
def E_Mg : ℝ := -2.37

/-- Calculate the cell potential of an electrochemical cell -/
def cell_potential (E_reduction E_oxidation : ℝ) : ℝ :=
  E_reduction - E_oxidation

/-- Theorem: The cell potential of an electrochemical cell involving 
    Aluminum Fluoride and Magnesium Chloride is 0.71 V -/
theorem AlF3_MgCl2_cell_potential : 
  cell_potential E_Al (-E_Mg) = 0.71 := by
  sorry

end NUMINAMATH_CALUDE_AlF3_MgCl2_cell_potential_l1247_124745


namespace NUMINAMATH_CALUDE_f_equals_g_l1247_124790

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l1247_124790


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l1247_124774

/-- Given a quadratic function with vertex (4, 10) and one x-intercept at (-1, 0),
    the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 10 + a * (x - 4)^2) →  -- vertex form of quadratic
  a * (-1)^2 + b * (-1) + c = 0 →                    -- x-intercept at (-1, 0)
  ∃ x, x ≠ -1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9      -- other x-intercept at 9
  := by sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l1247_124774


namespace NUMINAMATH_CALUDE_union_and_intersection_conditions_l1247_124752

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m + 3}

theorem union_and_intersection_conditions (m : ℝ) :
  (A ∪ B m = A ↔ m ∈ Set.Ioi (-2) ∪ Set.Iio (-1/2)) ∧
  (A ∩ B m ≠ ∅ ↔ m ∈ Set.Ioo (-2) 1) := by
  sorry

end NUMINAMATH_CALUDE_union_and_intersection_conditions_l1247_124752


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1247_124783

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 294 → volume = (((surface_area / 6) ^ (1/2 : ℝ)) ^ 3) → volume = 343 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1247_124783


namespace NUMINAMATH_CALUDE_shelves_per_case_l1247_124765

theorem shelves_per_case (num_cases : ℕ) (records_per_shelf : ℕ) (ridges_per_record : ℕ) 
  (shelf_fullness : ℚ) (total_ridges : ℕ) : ℕ :=
  let shelves_per_case := (total_ridges / (shelf_fullness * records_per_shelf * ridges_per_record)) / num_cases
  3

#check shelves_per_case 4 20 60 (3/5) 8640

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_shelves_per_case_l1247_124765


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1247_124782

-- Define the set of real numbers greater than 1
def greater_than_one : Set ℝ := {x | x > 1}

-- Define the solution set of ax - 1 > 0
def solution_set_linear (a : ℝ) : Set ℝ := {x | a * x - 1 > 0}

-- Define the solution set of (ax - 1)(x + 2) ≥ 0
def solution_set_quadratic (a : ℝ) : Set ℝ := {x | (a * x - 1) * (x + 2) ≥ 0}

-- Define the set (-∞, -2] ∪ [1, +∞)
def target_set : Set ℝ := {x | x ≤ -2 ∨ x ≥ 1}

theorem solution_set_equivalence (a : ℝ) : 
  solution_set_linear a = greater_than_one → solution_set_quadratic a = target_set := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1247_124782


namespace NUMINAMATH_CALUDE_advertising_customers_l1247_124797

/-- Proves that the number of customers brought to a site by advertising is 100,
    given the cost of advertising, purchase rate, item cost, and profit. -/
theorem advertising_customers (ad_cost profit item_cost : ℝ) (purchase_rate : ℝ) :
  ad_cost = 1000 →
  profit = 1000 →
  item_cost = 25 →
  purchase_rate = 0.8 →
  ∃ (num_customers : ℕ), 
    (↑num_customers : ℝ) * purchase_rate * item_cost = ad_cost + profit ∧
    num_customers = 100 :=
by sorry

end NUMINAMATH_CALUDE_advertising_customers_l1247_124797


namespace NUMINAMATH_CALUDE_cortland_apples_l1247_124799

theorem cortland_apples (total : ℝ) (golden : ℝ) (macintosh : ℝ) 
  (h1 : total = 0.67)
  (h2 : golden = 0.17)
  (h3 : macintosh = 0.17) :
  total - (golden + macintosh) = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_cortland_apples_l1247_124799


namespace NUMINAMATH_CALUDE_all_statements_incorrect_l1247_124730

-- Define the types for functions and properties
def Function := ℝ → ℝ
def Periodic (f : Function) : Prop := ∃ T > 0, ∀ x, f (x + T) = f x
def Monotonic (f : Function) : Prop := ∀ x y, x < y → f x < f y

-- Define the original proposition
def OriginalProposition : Prop := ∀ f : Function, Periodic f → ¬(Monotonic f)

-- Define the given statements
def GivenConverse : Prop := ∀ f : Function, Monotonic f → ¬(Periodic f)
def GivenNegation : Prop := ∀ f : Function, Periodic f → Monotonic f
def GivenContrapositive : Prop := ∀ f : Function, Monotonic f → Periodic f

-- Theorem stating that none of the given statements are correct
theorem all_statements_incorrect : 
  (GivenConverse ≠ (¬OriginalProposition → OriginalProposition)) ∧
  (GivenNegation ≠ ¬OriginalProposition) ∧
  (GivenContrapositive ≠ (¬¬OriginalProposition → ¬OriginalProposition)) :=
sorry

end NUMINAMATH_CALUDE_all_statements_incorrect_l1247_124730


namespace NUMINAMATH_CALUDE_salary_increase_l1247_124713

theorem salary_increase (num_employees : ℕ) (initial_avg : ℚ) (manager_salary : ℚ) :
  num_employees = 20 →
  initial_avg = 1600 →
  manager_salary = 3700 →
  let total_salary := num_employees * initial_avg
  let new_total := total_salary + manager_salary
  let new_avg := new_total / (num_employees + 1)
  new_avg - initial_avg = 100 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l1247_124713


namespace NUMINAMATH_CALUDE_diagonal_intersection_fixed_point_l1247_124720

/-- An ellipse with equation x^2/4 + y^2/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- A point is on the ellipse C -/
def on_ellipse_C (p : ℝ × ℝ) : Prop := ellipse_C p.1 p.2

/-- Quadrilateral MNPQ with vertices on ellipse C -/
structure Quadrilateral_MNPQ where
  M : ℝ × ℝ
  N : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  hM : on_ellipse_C M
  hN : on_ellipse_C N
  hP : on_ellipse_C P
  hQ : on_ellipse_C Q
  hMQ_NP : M.2 + Q.2 = 0 ∧ N.2 + P.2 = 0  -- MQ || NP and MQ ⊥ x-axis
  hS : ∃ t : ℝ, (M.2 - N.2) * (4 - P.1) = (M.1 - 4) * (P.2 - N.2) ∧
                (Q.2 - P.2) * (4 - N.1) = (Q.1 - 4) * (N.2 - P.2)  -- MN and QP intersect at S(4,0)

/-- The theorem to be proved -/
theorem diagonal_intersection_fixed_point (q : Quadrilateral_MNPQ) :
  ∃ (I : ℝ × ℝ), I = (1, 0) ∧
  (q.M.2 - q.P.2) * (I.1 - q.N.1) = (q.M.1 - I.1) * (I.2 - q.N.2) ∧
  (q.N.2 - q.Q.2) * (I.1 - q.M.1) = (q.N.1 - I.1) * (I.2 - q.M.2) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersection_fixed_point_l1247_124720


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1247_124735

theorem complex_equation_solution (a b : ℝ) :
  (Complex.mk 1 2) / (Complex.mk a b) = Complex.mk 1 1 →
  a = (3 : ℝ) / 2 ∧ b = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1247_124735


namespace NUMINAMATH_CALUDE_smallest_k_for_sum_of_squares_multiple_of_400_l1247_124755

theorem smallest_k_for_sum_of_squares_multiple_of_400 : 
  ∀ k : ℕ+, k < 800 → ¬(∃ m : ℕ, k * (k + 1) * (2 * k + 1) = 6 * 400 * m) ∧ 
  ∃ m : ℕ, 800 * (800 + 1) * (2 * 800 + 1) = 6 * 400 * m :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_sum_of_squares_multiple_of_400_l1247_124755


namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l1247_124770

theorem shopping_tax_calculation (total : ℝ) (clothing_percent : ℝ) (food_percent : ℝ) 
  (other_percent : ℝ) (clothing_tax : ℝ) (food_tax : ℝ) (total_tax_percent : ℝ) 
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.1)
  (h3 : other_percent = 0.4)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax = 0.04)
  (h6 : food_tax = 0)
  (h7 : total_tax_percent = 0.052)
  : ∃ other_tax : ℝ, 
    clothing_tax * clothing_percent * total + 
    food_tax * food_percent * total + 
    other_tax * other_percent * total = 
    total_tax_percent * total ∧ 
    other_tax = 0.08 := by
sorry

end NUMINAMATH_CALUDE_shopping_tax_calculation_l1247_124770


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1247_124708

open Complex

theorem complex_modulus_problem (m : ℝ) : 
  (↑(1 + m * I) * (3 + I) * I).im ≠ 0 →
  (↑(1 + m * I) * (3 + I) * I).re = 0 →
  abs ((m + 3 * I) / (1 - I)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1247_124708


namespace NUMINAMATH_CALUDE_wanda_blocks_theorem_l1247_124737

/-- The total number of blocks Wanda has after receiving more from Theresa -/
def total_blocks (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that Wanda's total blocks is the sum of her initial blocks and additional blocks -/
theorem wanda_blocks_theorem (initial : ℕ) (additional : ℕ) :
  total_blocks initial additional = initial + additional := by
  sorry

end NUMINAMATH_CALUDE_wanda_blocks_theorem_l1247_124737


namespace NUMINAMATH_CALUDE_smallest_N_is_255_l1247_124739

/-- Represents a team in the basketball championship --/
structure Team where
  id : ℕ
  isCalifornian : Bool
  wins : ℕ

/-- Represents the basketball championship --/
structure Championship where
  N : ℕ
  teams : Finset Team
  games : Finset (Team × Team)

/-- The conditions of the championship --/
def ChampionshipConditions (c : Championship) : Prop :=
  -- Total number of teams is 5N
  c.teams.card = 5 * c.N
  -- Every two teams played exactly one game
  ∧ c.games.card = (c.teams.card * (c.teams.card - 1)) / 2
  -- 251 teams are from California
  ∧ (c.teams.filter (λ t => t.isCalifornian)).card = 251
  -- Alcatraz is a Californian team
  ∧ ∃ alcatraz ∈ c.teams, alcatraz.isCalifornian
    -- Alcatraz is the unique Californian champion
    ∧ ∀ t ∈ c.teams, t.isCalifornian → t.wins ≤ alcatraz.wins
    -- Alcatraz is the unique loser of the tournament
    ∧ ∀ t ∈ c.teams, t.id ≠ alcatraz.id → alcatraz.wins < t.wins

/-- The theorem stating that the smallest possible value of N is 255 --/
theorem smallest_N_is_255 :
  ∀ c : Championship, ChampionshipConditions c → c.N ≥ 255 :=
sorry

end NUMINAMATH_CALUDE_smallest_N_is_255_l1247_124739


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1247_124702

/-- Given two vectors a and b in ℝ², where a = (2, 3) and b = (x, -9),
    if a is parallel to b, then x = -6. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![x, -9]
  (∃ (k : ℝ), b = k • a) →
  x = -6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1247_124702


namespace NUMINAMATH_CALUDE_solve_farmer_problem_l1247_124754

def farmer_problem (total_cattle : ℕ) (male_percentage : ℚ) (male_count : ℕ) (total_milk : ℚ) : Prop :=
  let female_percentage : ℚ := 1 - male_percentage
  let female_count : ℕ := total_cattle - male_count
  let milk_per_female : ℚ := total_milk / female_count
  (male_percentage * total_cattle = male_count) ∧
  (female_percentage * total_cattle = female_count) ∧
  (milk_per_female = 2)

theorem solve_farmer_problem :
  ∃ (total_cattle : ℕ),
    farmer_problem total_cattle (2/5) 50 150 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_farmer_problem_l1247_124754


namespace NUMINAMATH_CALUDE_magazine_cost_l1247_124743

theorem magazine_cost (b m : ℝ) 
  (h1 : 2 * b + 2 * m = 26) 
  (h2 : b + 3 * m = 27) : 
  m = 7 := by
sorry

end NUMINAMATH_CALUDE_magazine_cost_l1247_124743


namespace NUMINAMATH_CALUDE_count_squares_with_six_or_more_black_l1247_124714

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  position : Nat × Nat

/-- The checkerboard -/
def checkerboard : Nat := 8

/-- Function to count black squares in a given square -/
def countBlackSquares (s : Square) : Nat :=
  sorry

/-- Function to check if a square is valid (fits on the board) -/
def isValidSquare (s : Square) : Bool :=
  s.size > 0 && s.size ≤ checkerboard &&
  s.position.1 + s.size ≤ checkerboard &&
  s.position.2 + s.size ≤ checkerboard

/-- Function to generate all valid squares on the board -/
def allValidSquares : List Square :=
  sorry

/-- Main theorem -/
theorem count_squares_with_six_or_more_black : 
  (allValidSquares.filter (fun s => isValidSquare s && countBlackSquares s ≥ 6)).length = 55 :=
  sorry

end NUMINAMATH_CALUDE_count_squares_with_six_or_more_black_l1247_124714


namespace NUMINAMATH_CALUDE_smallest_m_is_20_l1247_124725

/-- The set of complex numbers with real part between 1/2 and 2/3 -/
def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ 2/3}

/-- The property that for all n ≥ m, there exists a complex number z in T such that z^n = 1 -/
def has_nth_root_of_unity (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z ∈ T, z^n = 1

/-- 20 is the smallest positive integer satisfying the property -/
theorem smallest_m_is_20 :
  has_nth_root_of_unity 20 ∧ ∀ m : ℕ, 0 < m → m < 20 → ¬(has_nth_root_of_unity m) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_20_l1247_124725


namespace NUMINAMATH_CALUDE_decryption_result_l1247_124772

/-- Represents an encrypted text -/
def EncryptedText := String

/-- Represents a decrypted text -/
def DecryptedText := String

/-- The encryption method used for the original message -/
def encryptionMethod (original : String) (encrypted : EncryptedText) : Prop :=
  encrypted.toList.filter (· ∈ original.toList) = original.toList

/-- The decryption function -/
noncomputable def decrypt (text : EncryptedText) : DecryptedText :=
  sorry

/-- Theorem stating the decryption results -/
theorem decryption_result 
  (text1 text2 text3 : EncryptedText)
  (h1 : encryptionMethod "МОСКВА" "ЙМЫВОТСБЛКЪГВЦАЯЯ")
  (h2 : encryptionMethod "МОСКВА" "УКМАПОЧСРКЩВЗАХ")
  (h3 : encryptionMethod "МОСКВА" "ШМФЭОГЧСЙЪКФЬВЫЕАКК")
  (h4 : text1 = "ТПЕОИРВНТМОЛАРГЕИАНВИЛЕДНМТААГТДЬТКУБЧКГЕИШНЕИАЯРЯ")
  (h5 : text2 = "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП")
  (h6 : text3 = "РТПАИОМВСВТИЕОБПРОЕННИГЬКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК") :
  (decrypt text1 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧
   decrypt text2 = "С ЧИСТОЙ СОВЕСТЬЮ" ∧
   decrypt text3 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ") :=
by sorry

end NUMINAMATH_CALUDE_decryption_result_l1247_124772


namespace NUMINAMATH_CALUDE_product_xy_is_264_l1247_124776

theorem product_xy_is_264 (x y : ℝ) 
  (eq1 : -3 * x + 4 * y = 28) 
  (eq2 : 3 * x - 2 * y = 8) : 
  x * y = 264 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_is_264_l1247_124776


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l1247_124787

open Set

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 3}

theorem set_intersection_theorem : A ∩ B = Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l1247_124787


namespace NUMINAMATH_CALUDE_desired_interest_percentage_l1247_124744

/-- Calculates the desired interest percentage for a share investment. -/
theorem desired_interest_percentage
  (face_value : ℝ)
  (dividend_rate : ℝ)
  (market_value : ℝ)
  (h1 : face_value = 20)
  (h2 : dividend_rate = 0.09)
  (h3 : market_value = 15) :
  (dividend_rate * face_value) / market_value = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_desired_interest_percentage_l1247_124744


namespace NUMINAMATH_CALUDE_probability_power_of_two_four_digit_l1247_124791

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A number is a power of 2 if its base-2 logarithm is an integer. -/
def IsPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- The count of four-digit numbers that are powers of 2. -/
def CountPowersOfTwoFourDigit : ℕ := 4

/-- The total count of four-digit numbers. -/
def TotalFourDigitNumbers : ℕ := 9000

/-- The probability of a randomly chosen four-digit number being a power of 2. -/
def ProbabilityPowerOfTwo : ℚ := CountPowersOfTwoFourDigit / TotalFourDigitNumbers

theorem probability_power_of_two_four_digit :
  ProbabilityPowerOfTwo = 1 / 2250 := by sorry

end NUMINAMATH_CALUDE_probability_power_of_two_four_digit_l1247_124791


namespace NUMINAMATH_CALUDE_some_number_value_l1247_124732

theorem some_number_value : ∃ (some_number : ℝ), 
  |5 - 8 * (3 - some_number)| - |5 - 11| = 71 ∧ some_number = 12 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1247_124732


namespace NUMINAMATH_CALUDE_calculate_expression_l1247_124703

theorem calculate_expression : (1/3)⁻¹ + Real.sqrt 12 - |Real.sqrt 3 - 2| - (π - 2023)^0 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1247_124703


namespace NUMINAMATH_CALUDE_sunglasses_cap_probability_l1247_124706

theorem sunglasses_cap_probability (total_sunglasses : ℕ) (total_caps : ℕ) 
  (prob_cap_given_sunglasses : ℚ) :
  total_sunglasses = 80 →
  total_caps = 60 →
  prob_cap_given_sunglasses = 3/8 →
  (prob_cap_given_sunglasses * total_sunglasses : ℚ) / total_caps = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_cap_probability_l1247_124706


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1247_124727

/-- 
Given a boat traveling downstream with the following conditions:
1. The rate of the stream is 5 km/hr
2. The boat takes 3 hours to cover a distance of 63 km downstream

This theorem proves that the speed of the boat in still water is 16 km/hr.
-/
theorem boat_speed_in_still_water : 
  ∀ (stream_rate : ℝ) (downstream_time : ℝ) (downstream_distance : ℝ),
  stream_rate = 5 →
  downstream_time = 3 →
  downstream_distance = 63 →
  ∃ (still_water_speed : ℝ),
    still_water_speed = 16 ∧
    downstream_distance = (still_water_speed + stream_rate) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1247_124727


namespace NUMINAMATH_CALUDE_polygon_interior_less_than_exterior_has_three_sides_l1247_124784

theorem polygon_interior_less_than_exterior_has_three_sides
  (n : ℕ) -- number of sides of the polygon
  (h_polygon : n ≥ 3) -- n is at least 3 for a polygon
  (interior_sum : ℝ) -- sum of interior angles
  (exterior_sum : ℝ) -- sum of exterior angles
  (h_interior : interior_sum = (n - 2) * 180) -- formula for interior angle sum
  (h_exterior : exterior_sum = 360) -- exterior angle sum is always 360°
  (h_less : interior_sum < exterior_sum) -- given condition
  : n = 3 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_less_than_exterior_has_three_sides_l1247_124784


namespace NUMINAMATH_CALUDE_fuel_used_fraction_l1247_124718

def car_speed : ℝ := 50
def fuel_efficiency : ℝ := 30
def tank_capacity : ℝ := 15
def travel_time : ℝ := 5

theorem fuel_used_fraction (speed : ℝ) (efficiency : ℝ) (capacity : ℝ) (time : ℝ)
  (h1 : speed = car_speed)
  (h2 : efficiency = fuel_efficiency)
  (h3 : capacity = tank_capacity)
  (h4 : time = travel_time) :
  (speed * time / efficiency) / capacity = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fuel_used_fraction_l1247_124718


namespace NUMINAMATH_CALUDE_range_of_a_l1247_124717

theorem range_of_a (a : ℝ) : 
  (∀ x, -2 < x ∧ x < 3 → -2 < x ∧ x < a) ∧ 
  (∃ x, -2 < x ∧ x < a ∧ ¬(-2 < x ∧ x < 3)) 
  ↔ a > 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1247_124717


namespace NUMINAMATH_CALUDE_common_chord_equation_l1247_124748

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the common chord line
def common_chord (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem common_chord_equation :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l1247_124748


namespace NUMINAMATH_CALUDE_part_one_part_two_l1247_124763

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 4| + |x - a|

-- Part I
theorem part_one : 
  {x : ℝ | f 2 x > 10} = {x : ℝ | x > 8 ∨ x < -2} :=
sorry

-- Part II
theorem part_two : 
  (∀ x : ℝ, f a x ≥ 1) → (a ≥ 5 ∨ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1247_124763


namespace NUMINAMATH_CALUDE_range_of_a_l1247_124705

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 < 0}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- Define the theorem
theorem range_of_a (a : ℝ) : M ⊆ N a ↔ a ∈ Set.Ici 2 := by
  sorry

-- Note: Set.Ici 2 represents the set [2, +∞) in Lean

end NUMINAMATH_CALUDE_range_of_a_l1247_124705


namespace NUMINAMATH_CALUDE_point_ratio_on_line_l1247_124700

/-- Given four points P, Q, R, and S on a line in that order, with specific distances between them,
    prove that the ratio of PR to QS is 7/17. -/
theorem point_ratio_on_line (P Q R S : ℝ) : 
  Q - P = 3 →
  R - Q = 4 →
  S - P = 20 →
  P < Q ∧ Q < R ∧ R < S →
  (R - P) / (S - Q) = 7 / 17 := by
sorry

end NUMINAMATH_CALUDE_point_ratio_on_line_l1247_124700


namespace NUMINAMATH_CALUDE_fraction_inequality_l1247_124750

theorem fraction_inequality (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hab : a < b) :
  a / b < (a + m) / (b + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1247_124750


namespace NUMINAMATH_CALUDE_cubic_sum_identity_l1247_124780

theorem cubic_sum_identity (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h_sum : a + b + c = d) :
  (a^3 + b^3 + c^3 - 3*a*b*c) / (a*b*c) = d * (a^2 + b^2 + c^2 - a*b - a*c - b*c) / (a*b*c) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_identity_l1247_124780


namespace NUMINAMATH_CALUDE_max_a_for_three_solutions_l1247_124767

/-- The equation function that we're analyzing -/
def f (x a : ℝ) : ℝ := (|x - 2| + 2*a)^2 - 3*(|x - 2| + 2*a) + 4*a*(3 - 4*a)

/-- Predicate to check if the equation has three solutions for a given 'a' -/
def has_three_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0

/-- The theorem stating that 0.5 is the maximum value of 'a' for which the equation has three solutions -/
theorem max_a_for_three_solutions :
  ∀ a : ℝ, has_three_solutions a → a ≤ 0.5 ∧
  has_three_solutions 0.5 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_three_solutions_l1247_124767


namespace NUMINAMATH_CALUDE_tenth_day_is_monday_l1247_124788

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a schedule for a month -/
structure MonthSchedule where
  numDays : Nat
  firstDay : DayOfWeek
  runningDays : List DayOfWeek
  runningTimePerDay : Nat
  totalRunningTime : Nat

/-- Returns the day of the week for a given day of the month -/
def dayOfMonth (schedule : MonthSchedule) (day : Nat) : DayOfWeek :=
  sorry

theorem tenth_day_is_monday (schedule : MonthSchedule) :
  schedule.numDays = 31 ∧
  schedule.runningDays = [DayOfWeek.Monday, DayOfWeek.Saturday, DayOfWeek.Sunday] ∧
  schedule.runningTimePerDay = 20 ∧
  schedule.totalRunningTime = 5 * 60 →
  dayOfMonth schedule 10 = DayOfWeek.Monday :=
by sorry

end NUMINAMATH_CALUDE_tenth_day_is_monday_l1247_124788


namespace NUMINAMATH_CALUDE_muffin_spending_l1247_124722

theorem muffin_spending (x : ℝ) : 
  (x = 0.9 * x + 15) → (x + 0.9 * x = 285) :=
by sorry

end NUMINAMATH_CALUDE_muffin_spending_l1247_124722


namespace NUMINAMATH_CALUDE_stair_climbing_time_l1247_124726

theorem stair_climbing_time : 
  let n : ℕ := 4  -- number of flights
  let a : ℕ := 30 -- time for first flight
  let d : ℕ := 10 -- time increase for each subsequent flight
  let S := n * (2 * a + (n - 1) * d) / 2  -- sum formula for arithmetic sequence
  S = 180 := by sorry

end NUMINAMATH_CALUDE_stair_climbing_time_l1247_124726


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1247_124711

/-- Given an ellipse with equation 2x^2 + 3y^2 = 1, its major axis length is √2 -/
theorem ellipse_major_axis_length :
  let ellipse_eq : ℝ → ℝ → Prop := λ x y => 2 * x^2 + 3 * y^2 = 1
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ x y, ellipse_eq x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    2 * a = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1247_124711


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l1247_124758

/-- The polynomial z^5 - z^3 + z -/
def f (z : ℂ) : ℂ := z^5 - z^3 + z

/-- n-th root of unity -/
def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

/-- All roots of f are n-th roots of unity -/
def all_roots_are_nth_roots_of_unity (n : ℕ) : Prop :=
  ∀ z : ℂ, f z = 0 → is_nth_root_of_unity z n

/-- 12 is the smallest positive integer n such that all roots of f are n-th roots of unity -/
theorem smallest_n_for_roots_of_unity :
  (all_roots_are_nth_roots_of_unity 12) ∧
  (∀ m : ℕ, 0 < m → m < 12 → ¬(all_roots_are_nth_roots_of_unity m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l1247_124758


namespace NUMINAMATH_CALUDE_percentage_calculation_l1247_124760

theorem percentage_calculation (y : ℝ) : 
  0.11 * y = 0.3 * (0.7 * y) - 0.1 * y := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1247_124760


namespace NUMINAMATH_CALUDE_or_false_sufficient_not_necessary_for_and_false_l1247_124792

theorem or_false_sufficient_not_necessary_for_and_false (p q : Prop) :
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬(p ∨ q)) :=
sorry

end NUMINAMATH_CALUDE_or_false_sufficient_not_necessary_for_and_false_l1247_124792


namespace NUMINAMATH_CALUDE_convincing_statement_l1247_124753

-- Define the types of people
inductive Person
| Knight
| Knave

-- Define the wealth status of knights
inductive KnightWealth
| Poor
| Rich

-- Define a function to determine if a person tells the truth
def tellsTruth (p : Person) : Prop :=
  match p with
  | Person.Knight => True
  | Person.Knave => False

-- Define the statement "I am not a poor knight"
def statement (p : Person) (w : KnightWealth) : Prop :=
  p = Person.Knight ∧ w ≠ KnightWealth.Poor

-- Theorem to prove
theorem convincing_statement 
  (p : Person) (w : KnightWealth) : 
  tellsTruth p → statement p w → (p = Person.Knight ∧ w = KnightWealth.Rich) :=
by
  sorry


end NUMINAMATH_CALUDE_convincing_statement_l1247_124753
