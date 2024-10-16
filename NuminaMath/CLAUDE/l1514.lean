import Mathlib

namespace NUMINAMATH_CALUDE_apple_picking_l1514_151491

theorem apple_picking (minjae_apples father_apples : ℝ) 
  (h1 : minjae_apples = 2.6)
  (h2 : father_apples = 5.98) :
  minjae_apples + father_apples = 8.58 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_l1514_151491


namespace NUMINAMATH_CALUDE_dragons_games_count_l1514_151436

theorem dragons_games_count :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (60 * initial_games) / 100 →
    ∃ (total_games : ℕ),
      total_games = initial_games + 11 ∧
      (initial_wins + 8 : ℚ) / total_games = 55 / 100 ∧
      total_games = 50 :=
by sorry

end NUMINAMATH_CALUDE_dragons_games_count_l1514_151436


namespace NUMINAMATH_CALUDE_abs_square_not_always_equal_to_value_l1514_151418

theorem abs_square_not_always_equal_to_value : ¬ ∀ a : ℝ, |a^2| = a := by
  sorry

end NUMINAMATH_CALUDE_abs_square_not_always_equal_to_value_l1514_151418


namespace NUMINAMATH_CALUDE_exponential_sum_rule_l1514_151462

theorem exponential_sum_rule (a : ℝ) (x₁ x₂ : ℝ) (ha : 0 < a) :
  a^(x₁ + x₂) = a^x₁ * a^x₂ := by
  sorry

end NUMINAMATH_CALUDE_exponential_sum_rule_l1514_151462


namespace NUMINAMATH_CALUDE_short_trees_calculation_l1514_151468

/-- The number of short trees initially in the park -/
def initial_short_trees : ℕ := 41

/-- The number of short trees to be planted -/
def planted_short_trees : ℕ := 57

/-- The total number of short trees after planting -/
def total_short_trees : ℕ := 98

/-- Theorem stating that the initial number of short trees plus the planted short trees equals the total short trees -/
theorem short_trees_calculation : 
  initial_short_trees + planted_short_trees = total_short_trees :=
by sorry

end NUMINAMATH_CALUDE_short_trees_calculation_l1514_151468


namespace NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l1514_151472

theorem smallest_solution_absolute_value_equation :
  ∃ (x : ℝ), x * |x| = 2 * x + 1 ∧ 
  ∀ (y : ℝ), y * |y| = 2 * y + 1 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l1514_151472


namespace NUMINAMATH_CALUDE_lcm_factor_theorem_l1514_151470

theorem lcm_factor_theorem (A B : ℕ) (hcf lcm X : ℕ) : 
  A > 0 → B > 0 → 
  A = 368 → 
  hcf = Nat.gcd A B → 
  hcf = 23 → 
  lcm = Nat.lcm A B → 
  lcm = hcf * X * 16 → 
  X = 1 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_theorem_l1514_151470


namespace NUMINAMATH_CALUDE_triangle_problem_l1514_151420

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a * sin (2 * B) = Real.sqrt 3 * b * sin A →
  cos A = 1 / 3 →
  B = π / 6 ∧ sin C = (2 * Real.sqrt 6 + 1) / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1514_151420


namespace NUMINAMATH_CALUDE_min_workers_for_job_l1514_151410

/-- Represents a construction job with workers -/
structure ConstructionJob where
  totalDays : ℕ
  elapsedDays : ℕ
  initialWorkers : ℕ
  completedPortion : ℚ
  
/-- Calculates the minimum number of workers needed to complete the job on time -/
def minWorkersNeeded (job : ConstructionJob) : ℕ :=
  job.initialWorkers

/-- Theorem stating that for the given job specifications, 
    the minimum number of workers needed is 10 -/
theorem min_workers_for_job :
  let job := ConstructionJob.mk 40 10 10 (1/4)
  minWorkersNeeded job = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_workers_for_job_l1514_151410


namespace NUMINAMATH_CALUDE_molecular_weight_N2O5_l1514_151467

/-- Molecular weight calculation for Dinitrogen pentoxide (N2O5) -/
theorem molecular_weight_N2O5 (atomic_weight_N atomic_weight_O : ℝ)
  (h1 : atomic_weight_N = 14.01)
  (h2 : atomic_weight_O = 16.00) :
  2 * atomic_weight_N + 5 * atomic_weight_O = 108.02 := by
  sorry

#check molecular_weight_N2O5

end NUMINAMATH_CALUDE_molecular_weight_N2O5_l1514_151467


namespace NUMINAMATH_CALUDE_share_ratio_l1514_151494

/-- 
Given:
- The total amount of money is $400
- A's share is $160
- A gets a certain fraction (x) as much as B and C together
- B gets 6/9 as much as A and C together

Prove that the ratio of A's share to the combined share of B and C is 2:3
-/
theorem share_ratio (total : ℕ) (a b c : ℕ) (x : ℚ) :
  total = 400 →
  a = 160 →
  a = x * (b + c) →
  b = (6/9 : ℚ) * (a + c) →
  a + b + c = total →
  (a : ℚ) / ((b + c) : ℚ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l1514_151494


namespace NUMINAMATH_CALUDE_sequence_to_zero_l1514_151428

/-- A transformation that applies |x - α| to each element of a sequence -/
def transform (s : List ℝ) (α : ℝ) : List ℝ :=
  s.map (fun x => |x - α|)

/-- Predicate to check if all elements in a list are zero -/
def all_zero (s : List ℝ) : Prop :=
  s.all (fun x => x = 0)

theorem sequence_to_zero (n : ℕ) :
  ∀ (s : List ℝ), s.length = n →
  (∃ (transformations : List ℝ),
    transformations.length = n ∧
    all_zero (transformations.foldl transform s)) ∧
  (∀ (transformations : List ℝ),
    transformations.length < n →
    ¬ all_zero (transformations.foldl transform s)) :=
sorry

end NUMINAMATH_CALUDE_sequence_to_zero_l1514_151428


namespace NUMINAMATH_CALUDE_min_value_p_l1514_151437

theorem min_value_p (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20)
  (prod_sq_eq : p^2 * q^2 * r^2 * s^2 = 16) :
  ∃ (min_p : ℝ), min_p = 2 ∧ p ≥ min_p := by
  sorry

end NUMINAMATH_CALUDE_min_value_p_l1514_151437


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1514_151402

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem solution_set_part1 (x : ℝ) :
  f (-12) (-2) x < 0 ↔ -1/2 < x ∧ x < 1/3 := by sorry

-- Part 2
theorem range_of_a_part2 (a : ℝ) :
  (∀ x, f a (-1) x ≥ 0) → a ≥ 1/8 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1514_151402


namespace NUMINAMATH_CALUDE_read_book_in_six_days_book_structure_l1514_151421

/-- The number of days required to read a book -/
def days_to_read (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- Theorem: It takes 6 days to read a 612-page book at 102 pages per day -/
theorem read_book_in_six_days :
  days_to_read 612 102 = 6 := by
  sorry

/-- The book has 24 chapters with pages equally distributed -/
def pages_per_chapter (total_pages : ℕ) (num_chapters : ℕ) : ℕ :=
  total_pages / num_chapters

/-- The book has 612 pages and 24 chapters -/
theorem book_structure :
  pages_per_chapter 612 24 = 612 / 24 := by
  sorry

end NUMINAMATH_CALUDE_read_book_in_six_days_book_structure_l1514_151421


namespace NUMINAMATH_CALUDE_rectangle_side_difference_l1514_151495

theorem rectangle_side_difference (A d x y : ℝ) (h1 : A > 0) (h2 : d > 0) (h3 : x > y) (h4 : x * y = A) (h5 : x^2 + y^2 = d^2) : x - y = 2 * Real.sqrt A := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_difference_l1514_151495


namespace NUMINAMATH_CALUDE_mans_upstream_speed_l1514_151493

/-- Given a man's rowing speeds, calculate his upstream speed -/
theorem mans_upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 45)
  (h2 : speed_downstream = 60) :
  speed_still - (speed_downstream - speed_still) = 30 := by
  sorry

#check mans_upstream_speed

end NUMINAMATH_CALUDE_mans_upstream_speed_l1514_151493


namespace NUMINAMATH_CALUDE_distribute_six_interns_three_schools_l1514_151409

/-- The number of ways to distribute n interns among k schools, where each intern is assigned to exactly one school and each school receives at least one intern. -/
def distribute_interns (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 540 ways to distribute 6 interns among 3 schools under the given conditions. -/
theorem distribute_six_interns_three_schools : distribute_interns 6 3 = 540 := by sorry

end NUMINAMATH_CALUDE_distribute_six_interns_three_schools_l1514_151409


namespace NUMINAMATH_CALUDE_product_of_three_consecutive_even_numbers_divisible_by_48_l1514_151458

theorem product_of_three_consecutive_even_numbers_divisible_by_48 (k : ℤ) :
  ∃ (n : ℤ), (2*k) * (2*k + 2) * (2*k + 4) = 48 * n :=
sorry

end NUMINAMATH_CALUDE_product_of_three_consecutive_even_numbers_divisible_by_48_l1514_151458


namespace NUMINAMATH_CALUDE_combination_sum_l1514_151459

theorem combination_sum : Nat.choose 99 2 + Nat.choose 99 3 = 161700 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_l1514_151459


namespace NUMINAMATH_CALUDE_inscribed_polygon_division_l1514_151440

-- Define a polygon inscribed around a circle
structure InscribedPolygon where
  vertices : List (ℝ × ℝ)
  center : ℝ × ℝ
  radius : ℝ
  is_inscribed : ∀ v ∈ vertices, dist center v = radius

-- Define a line passing through a point
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Define the area of a polygon
def area (p : InscribedPolygon) : ℝ := sorry

-- Define the perimeter of a polygon
def perimeter (p : InscribedPolygon) : ℝ := sorry

-- Define the two parts of a polygon divided by a line
def divided_parts (p : InscribedPolygon) (l : Line) : (InscribedPolygon × InscribedPolygon) := sorry

theorem inscribed_polygon_division (p : InscribedPolygon) (l : Line) 
  (h : l.point = p.center) : 
  let (p1, p2) := divided_parts p l
  (area p1 = area p2) ∧ (perimeter p1 = perimeter p2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_polygon_division_l1514_151440


namespace NUMINAMATH_CALUDE_pet_store_dogs_l1514_151400

theorem pet_store_dogs (cat_count : ℕ) (cat_ratio dog_ratio : ℕ) : 
  cat_count = 18 → cat_ratio = 3 → dog_ratio = 4 → 
  (cat_count * dog_ratio) / cat_ratio = 24 := by
sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l1514_151400


namespace NUMINAMATH_CALUDE_edge_bound_l1514_151484

/-- A simple graph with no 4-cycles -/
structure NoCycleFourGraph where
  -- The vertex set
  V : Type
  -- The edge relation
  E : V → V → Prop
  -- Symmetry of edges
  symm : ∀ u v, E u v → E v u
  -- No self-loops
  irrefl : ∀ v, ¬E v v
  -- No 4-cycles
  no_four_cycle : ∀ a b c d, E a b → E b c → E c d → E d a → (a = c ∨ b = d)

/-- The number of vertices in a graph -/
def num_vertices (G : NoCycleFourGraph) : ℕ := sorry

/-- The number of edges in a graph -/
def num_edges (G : NoCycleFourGraph) : ℕ := sorry

/-- The main theorem -/
theorem edge_bound (G : NoCycleFourGraph) :
  let n := num_vertices G
  let m := num_edges G
  m ≤ (n / 4) * (1 + Real.sqrt (4 * n - 3)) := by sorry

end NUMINAMATH_CALUDE_edge_bound_l1514_151484


namespace NUMINAMATH_CALUDE_sphere_pyramid_height_l1514_151469

/-- The height of a square pyramid of spheres -/
def pyramid_height (n : ℕ) : ℝ :=
  2 * (n - 1)

/-- Theorem: The height of a square pyramid of spheres with radius 1,
    where the base layer has n^2 spheres and each subsequent layer has
    (n-1)^2 spheres until the top layer with 1 sphere, is 2(n-1). -/
theorem sphere_pyramid_height (n : ℕ) (h : n > 0) :
  let base_layer := n^2
  let top_layer := 1
  let sphere_radius := 1
  pyramid_height n = 2 * (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sphere_pyramid_height_l1514_151469


namespace NUMINAMATH_CALUDE_class_average_problem_l1514_151403

theorem class_average_problem (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) :
  total_students = 20 →
  excluded_students = 2 →
  excluded_avg = 45 →
  remaining_avg = 95 →
  (total_students * (total_students - excluded_students) * remaining_avg + 
   excluded_students * excluded_avg) / total_students = 90 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l1514_151403


namespace NUMINAMATH_CALUDE_x_value_proof_l1514_151446

theorem x_value_proof (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((5 * x) / 3) = x) : x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1514_151446


namespace NUMINAMATH_CALUDE_visual_range_increase_l1514_151488

theorem visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 100)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_visual_range_increase_l1514_151488


namespace NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l1514_151415

theorem min_value_of_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b * (a + b) = 4) :
  (∀ x y : ℝ, x > 0 → y > 0 → x * y * (x + y) = 4 → 2 * a + b ≤ 2 * x + y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y * (x + y) = 4 ∧ 2 * a + b = 2 * x + y) ∧
  2 * a + b = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l1514_151415


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l1514_151465

theorem max_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h1 : x + y = 16) (h2 : x = 2 * y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 16 → 1/a + 1/b ≤ 1/x + 1/y) ∧ 1/x + 1/y = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l1514_151465


namespace NUMINAMATH_CALUDE_matt_card_trade_profit_l1514_151413

def matt_card_value : ℕ := 6
def jane_card1_value : ℕ := 2
def jane_card2_value : ℕ := 9
def matt_cards_traded : ℕ := 2
def jane_cards1_received : ℕ := 3
def jane_cards2_received : ℕ := 1
def profit : ℕ := 3

theorem matt_card_trade_profit :
  (jane_cards1_received * jane_card1_value + jane_cards2_received * jane_card2_value) -
  (matt_cards_traded * matt_card_value) = profit := by
  sorry

end NUMINAMATH_CALUDE_matt_card_trade_profit_l1514_151413


namespace NUMINAMATH_CALUDE_expression_simplification_l1514_151450

theorem expression_simplification :
  Real.sqrt 3 + Real.sqrt (3 + 5) + Real.sqrt (3 + 5 + 7) + Real.sqrt (3 + 5 + 7 + 9) =
  Real.sqrt 3 + 2 * Real.sqrt 2 + Real.sqrt 15 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1514_151450


namespace NUMINAMATH_CALUDE_angle_C_measure_side_c_length_l1514_151497

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition1 (t : Triangle) : Prop :=
  t.b * Real.cos t.A + t.a * Real.cos t.B = -2 * t.c * Real.cos t.C

def satisfiesCondition2 (t : Triangle) : Prop :=
  t.b = 2 * t.a ∧ 
  (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3

-- Theorem 1
theorem angle_C_measure (t : Triangle) (h : satisfiesCondition1 t) : t.C = 2 * π / 3 := by
  sorry

-- Theorem 2
theorem side_c_length (t : Triangle) (h1 : satisfiesCondition1 t) (h2 : satisfiesCondition2 t) : 
  t.c = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_side_c_length_l1514_151497


namespace NUMINAMATH_CALUDE_equality_of_cyclic_powers_l1514_151444

theorem equality_of_cyclic_powers (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_eq1 : x^y = y^z) (h_eq2 : y^z = z^x) : x = y ∧ y = z :=
sorry

end NUMINAMATH_CALUDE_equality_of_cyclic_powers_l1514_151444


namespace NUMINAMATH_CALUDE_forty_percent_of_jacquelines_candy_bars_l1514_151426

def fred_candy_bars : ℕ := 12
def uncle_bob_candy_bars : ℕ := fred_candy_bars + 6
def total_fred_and_bob : ℕ := fred_candy_bars + uncle_bob_candy_bars
def jacqueline_candy_bars : ℕ := 10 * total_fred_and_bob

theorem forty_percent_of_jacquelines_candy_bars :
  (40 : ℚ) / 100 * jacqueline_candy_bars = 120 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_jacquelines_candy_bars_l1514_151426


namespace NUMINAMATH_CALUDE_product_of_ab_is_one_l1514_151476

theorem product_of_ab_is_one (a b : ℝ) 
  (h1 : a + 1/b = 4) 
  (h2 : 1/a + b = 16/15) : 
  ∃ x y : ℝ, x * y = 1 ∧ (a = x ∧ b = y ∨ a = y ∧ b = x) :=
by sorry

end NUMINAMATH_CALUDE_product_of_ab_is_one_l1514_151476


namespace NUMINAMATH_CALUDE_total_share_l1514_151492

theorem total_share (z y x : ℝ) : 
  z = 250 →
  y = 1.2 * z →
  x = 1.25 * y →
  x + y + z = 925 := by sorry

end NUMINAMATH_CALUDE_total_share_l1514_151492


namespace NUMINAMATH_CALUDE_min_dot_product_l1514_151401

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the center O and left focus F
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-3, 0)

-- Define a point P on the right branch of the hyperbola
def P : ℝ × ℝ → Prop := λ p => hyperbola p.1 p.2 ∧ p.1 ≥ 2

-- Define the dot product of OP and FP
def dot_product (p : ℝ × ℝ) : ℝ := p.1 * (p.1 + 3) + p.2 * p.2

-- Theorem statement
theorem min_dot_product :
  ∀ p : ℝ × ℝ, P p → ∀ q : ℝ × ℝ, P q → dot_product p ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_l1514_151401


namespace NUMINAMATH_CALUDE_typist_salary_problem_l1514_151485

/-- Proves that if a salary is increased by 10% and then decreased by 5%, 
    resulting in 1045, the original salary was 1000. -/
theorem typist_salary_problem (original_salary : ℝ) : 
  (original_salary * 1.1 * 0.95 = 1045) → original_salary = 1000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l1514_151485


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1514_151456

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 28) :
  (face_perimeter / 4) ^ 3 = 343 := by
  sorry

#check cube_volume_from_face_perimeter

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1514_151456


namespace NUMINAMATH_CALUDE_reyansh_farm_cows_l1514_151473

/-- Represents the number of cows on Mr. Reyansh's farm -/
def num_cows : ℕ := sorry

/-- Represents the daily water consumption of one cow in liters -/
def cow_water_daily : ℕ := 80

/-- Represents the number of sheep on Mr. Reyansh's farm -/
def num_sheep : ℕ := 10 * num_cows

/-- Represents the daily water consumption of one sheep in liters -/
def sheep_water_daily : ℕ := cow_water_daily / 4

/-- Represents the total water consumption for all animals in a week in liters -/
def total_water_weekly : ℕ := 78400

/-- Theorem stating that the number of cows on Mr. Reyansh's farm is 40 -/
theorem reyansh_farm_cows :
  num_cows = 40 :=
by sorry

end NUMINAMATH_CALUDE_reyansh_farm_cows_l1514_151473


namespace NUMINAMATH_CALUDE_rational_operations_closure_l1514_151479

theorem rational_operations_closure (a b : ℚ) (h : b ≠ 0) :
  (∃ (x : ℚ), x = a + b) ∧
  (∃ (y : ℚ), y = a - b) ∧
  (∃ (z : ℚ), z = a * b) ∧
  (∃ (w : ℚ), w = a / b) :=
by sorry

end NUMINAMATH_CALUDE_rational_operations_closure_l1514_151479


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_range_f_below_g_implies_a_range_l1514_151460

-- Define the function f
def f (a x : ℝ) : ℝ := x * |x - a| + 3 * x

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 1

-- Theorem 1
theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → -3 ≤ a ∧ a ≤ 3 :=
sorry

-- Theorem 2
theorem f_below_g_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a x < g x) → 3/2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_range_f_below_g_implies_a_range_l1514_151460


namespace NUMINAMATH_CALUDE_badminton_racket_cost_proof_l1514_151424

/-- The cost price of a badminton racket satisfying the given conditions -/
def badminton_racket_cost : ℝ := 125

/-- The markup percentage applied to the cost price -/
def markup_percentage : ℝ := 0.4

/-- The discount percentage applied to the marked price -/
def discount_percentage : ℝ := 0.2

/-- The profit made on the sale -/
def profit : ℝ := 15

theorem badminton_racket_cost_proof :
  (badminton_racket_cost * (1 + markup_percentage) * (1 - discount_percentage) =
   badminton_racket_cost + profit) := by
  sorry

end NUMINAMATH_CALUDE_badminton_racket_cost_proof_l1514_151424


namespace NUMINAMATH_CALUDE_relationship_abc_l1514_151408

theorem relationship_abc (a b c : ℝ) (ha : a = Real.sqrt 6 + Real.sqrt 7) 
  (hb : b = Real.sqrt 5 + Real.sqrt 8) (hc : c = 5) : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1514_151408


namespace NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1514_151471

/-- The number of tickets sold by the Richmond Tigers in the second half of the season -/
def second_half_tickets (total : ℕ) (first_half : ℕ) : ℕ :=
  total - first_half

/-- Theorem stating that the number of tickets sold in the second half of the season is 5703 -/
theorem richmond_tigers_ticket_sales :
  second_half_tickets 9570 3867 = 5703 := by
  sorry

end NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1514_151471


namespace NUMINAMATH_CALUDE_mortgage_payment_l1514_151480

theorem mortgage_payment (P : ℝ) : 
  (P * (1 - 3^10) / (1 - 3) = 2952400) → P = 100 := by
  sorry

end NUMINAMATH_CALUDE_mortgage_payment_l1514_151480


namespace NUMINAMATH_CALUDE_cube_difference_as_sum_of_squares_l1514_151452

theorem cube_difference_as_sum_of_squares (n : ℤ) :
  (n + 2)^3 - n^3 = n^2 + (n + 2)^2 + (2*n + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_as_sum_of_squares_l1514_151452


namespace NUMINAMATH_CALUDE_sequence_problem_l1514_151423

def arithmetic_sequence (a b : ℕ) (n : ℕ) : ℕ := a + b * (n - 1)

def geometric_sequence (b a : ℕ) (n : ℕ) : ℕ := b * a^(n - 1)

theorem sequence_problem (a b : ℕ) 
  (h1 : a > 1) 
  (h2 : b > 1) 
  (h3 : a < b) 
  (h4 : b * a < arithmetic_sequence a b 3) 
  (h5 : ∀ n : ℕ, n > 0 → ∃ m : ℕ, m > 0 ∧ geometric_sequence b a n = arithmetic_sequence a b m + 3) :
  a = 2 ∧ ∀ n : ℕ, arithmetic_sequence a b n = 5 * n - 3 :=
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1514_151423


namespace NUMINAMATH_CALUDE_multiplication_sum_l1514_151481

theorem multiplication_sum (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → (30 * a + a) * (10 * b + 4) = 126 → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_sum_l1514_151481


namespace NUMINAMATH_CALUDE_tree_spacing_l1514_151404

/-- Given a yard of length 400 meters with 26 trees planted at equal distances,
    including one at each end, the distance between two consecutive trees is 16 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (tree_spacing : ℝ) :
  yard_length = 400 →
  num_trees = 26 →
  tree_spacing * (num_trees - 1) = yard_length →
  tree_spacing = 16 := by
sorry

end NUMINAMATH_CALUDE_tree_spacing_l1514_151404


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1514_151412

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  n = Nat.gcd (1557 - 7) (Nat.gcd (2037 - 5) (2765 - 9)) ∧
  1557 % n = 7 ∧
  2037 % n = 5 ∧
  2765 % n = 9 ∧
  ∀ (m : ℕ), m > n → 
    (1557 % m = 7 ∧ 2037 % m = 5 ∧ 2765 % m = 9) → False :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1514_151412


namespace NUMINAMATH_CALUDE_bisection_arbitrary_precision_l1514_151475

/-- Represents a continuous function on a closed interval -/
def ContinuousFunction (a b : ℝ) := ℝ → ℝ

/-- Represents the bisection method applied to a function -/
def BisectionMethod (f : ContinuousFunction a b) (ε : ℝ) : ℝ := sorry

/-- Theorem stating that the bisection method can achieve arbitrary precision -/
theorem bisection_arbitrary_precision 
  (f : ContinuousFunction a b) 
  (h₁ : a < b) 
  (h₂ : f a * f b ≤ 0) 
  (ε : ℝ) 
  (h₃ : ε > 0) :
  ∃ x : ℝ, |f x| < ε ∧ x ∈ Set.Icc a b :=
sorry

end NUMINAMATH_CALUDE_bisection_arbitrary_precision_l1514_151475


namespace NUMINAMATH_CALUDE_max_operation_result_l1514_151464

def operation (n : ℕ) : ℚ :=
  2 * (2/3 * (300 - n))

theorem max_operation_result :
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → operation n ≤ 1160/3) ∧
  (∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ operation n = 1160/3) :=
sorry

end NUMINAMATH_CALUDE_max_operation_result_l1514_151464


namespace NUMINAMATH_CALUDE_max_earnings_is_zero_l1514_151496

/-- Represents the state of the boxes and Sisyphus's earnings -/
structure BoxState where
  a : ℕ  -- number of stones in box A
  b : ℕ  -- number of stones in box B
  c : ℕ  -- number of stones in box C
  earnings : ℤ  -- Sisyphus's current earnings (can be negative)

/-- Represents a move of a stone from one box to another -/
inductive Move
  | AtoB | AtoC | BtoA | BtoC | CtoA | CtoB

/-- Applies a move to the current state and returns the new state -/
def applyMove (state : BoxState) (move : Move) : BoxState :=
  match move with
  | Move.AtoB => { state with 
      a := state.a - 1, 
      b := state.b + 1, 
      earnings := state.earnings + state.b - (state.a - 1) }
  | Move.AtoC => { state with 
      a := state.a - 1, 
      c := state.c + 1, 
      earnings := state.earnings + state.c - (state.a - 1) }
  | Move.BtoA => { state with 
      b := state.b - 1, 
      a := state.a + 1, 
      earnings := state.earnings + state.a - (state.b - 1) }
  | Move.BtoC => { state with 
      b := state.b - 1, 
      c := state.c + 1, 
      earnings := state.earnings + state.c - (state.b - 1) }
  | Move.CtoA => { state with 
      c := state.c - 1, 
      a := state.a + 1, 
      earnings := state.earnings + state.a - (state.c - 1) }
  | Move.CtoB => { state with 
      c := state.c - 1, 
      b := state.b + 1, 
      earnings := state.earnings + state.b - (state.c - 1) }

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to an initial state -/
def applyMoves (initialState : BoxState) (moves : MoveSequence) : BoxState :=
  moves.foldl applyMove initialState

/-- Theorem: The maximum earnings of Sisyphus is 0 -/
theorem max_earnings_is_zero (initialState : BoxState) (moves : MoveSequence) :
  let finalState := applyMoves initialState moves
  (finalState.a = initialState.a ∧ 
   finalState.b = initialState.b ∧ 
   finalState.c = initialState.c) →
  finalState.earnings ≤ 0 := by
  sorry

#check max_earnings_is_zero

end NUMINAMATH_CALUDE_max_earnings_is_zero_l1514_151496


namespace NUMINAMATH_CALUDE_inverse_variation_sqrt_l1514_151489

/-- Given that z varies inversely as √w, prove that w = 64 when z = 2, 
    given that z = 8 when w = 4. -/
theorem inverse_variation_sqrt (z w : ℝ) (h : ∃ k : ℝ, ∀ w z, z * Real.sqrt w = k) :
  (∃ z₀ w₀, z₀ = 8 ∧ w₀ = 4 ∧ z₀ * Real.sqrt w₀ = 8 * Real.sqrt 4) →
  (∃ w₁, 2 * Real.sqrt w₁ = 8 * Real.sqrt 4 ∧ w₁ = 64) :=
by sorry


end NUMINAMATH_CALUDE_inverse_variation_sqrt_l1514_151489


namespace NUMINAMATH_CALUDE_concert_revenue_is_955000_l1514_151432

/-- Calculates the total revenue of a concert given the following parameters:
  * total_seats: Total number of seats in the arena
  * main_seat_cost: Cost of a main seat ticket
  * back_seat_cost: Cost of a back seat ticket
  * back_seats_sold: Number of back seat tickets sold
-/
def concert_revenue (total_seats : ℕ) (main_seat_cost back_seat_cost : ℕ) (back_seats_sold : ℕ) : ℕ :=
  let main_seats_sold := total_seats - back_seats_sold
  let main_seat_revenue := main_seats_sold * main_seat_cost
  let back_seat_revenue := back_seats_sold * back_seat_cost
  main_seat_revenue + back_seat_revenue

/-- Theorem stating that the concert revenue is $955,000 given the specific conditions -/
theorem concert_revenue_is_955000 :
  concert_revenue 20000 55 45 14500 = 955000 := by
  sorry

#eval concert_revenue 20000 55 45 14500

end NUMINAMATH_CALUDE_concert_revenue_is_955000_l1514_151432


namespace NUMINAMATH_CALUDE_smallest_square_multiplier_l1514_151414

/-- Given that y is the smallest positive integer such that a number n multiplied by y
    is the square of an integer, and y = 10, prove that n = 10. -/
theorem smallest_square_multiplier (y : ℕ) (n : ℕ) : 
  y = 10 →
  (∀ k : ℕ, k < y → ¬∃ m : ℕ, n * k = m^2) →
  (∃ m : ℕ, n * y = m^2) →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_multiplier_l1514_151414


namespace NUMINAMATH_CALUDE_monotonicity_intervals_max_value_on_interval_min_value_on_interval_l1514_151454

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval of interest
def interval : Set ℝ := Set.Icc (-3) 2

-- Statement for intervals of monotonicity
theorem monotonicity_intervals (x : ℝ) :
  (∀ y z, y < x → x < z → y < -1 → z < -1 → f y < f z) ∧
  (∀ y z, y < x → x < z → -1 < y → z < 1 → f y > f z) ∧
  (∀ y z, y < x → x < z → 1 < y → f y < f z) :=
sorry

-- Statement for maximum value on the interval
theorem max_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 2 :=
sorry

-- Statement for minimum value on the interval
theorem min_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≥ f x ∧ f x = -18 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_max_value_on_interval_min_value_on_interval_l1514_151454


namespace NUMINAMATH_CALUDE_min_value_problem_l1514_151447

theorem min_value_problem (x : ℝ) (h : x ≥ 3/2) :
  (∀ y, y ≥ 3/2 → (2*x^2 - 2*x + 1)/(x - 1) ≤ (2*y^2 - 2*y + 1)/(y - 1)) →
  (2*x^2 - 2*x + 1)/(x - 1) = 2*Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1514_151447


namespace NUMINAMATH_CALUDE_jasmine_needs_seven_cans_l1514_151487

/-- Represents the paint coverage problem for Jasmine --/
def paint_coverage_problem (initial_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) (rooms_per_new_can : ℕ) (total_rooms : ℕ) : Prop :=
  ∃ (additional_cans : ℕ),
    remaining_rooms + additional_cans * rooms_per_new_can = total_rooms

/-- Theorem stating that 7 additional cans are needed to cover all rooms --/
theorem jasmine_needs_seven_cans :
  paint_coverage_problem 50 4 36 2 50 →
  ∃ (additional_cans : ℕ), additional_cans = 7 ∧ 36 + additional_cans * 2 = 50 := by
  sorry

#check jasmine_needs_seven_cans

end NUMINAMATH_CALUDE_jasmine_needs_seven_cans_l1514_151487


namespace NUMINAMATH_CALUDE_smallest_number_l1514_151478

/-- Converts a number from base 6 to decimal -/
def base6ToDecimal (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 4 to decimal -/
def base4ToDecimal (n : Nat) : Nat :=
  (n / 1000) * 64 + ((n / 100) % 10) * 16 + ((n / 10) % 10) * 4 + (n % 10)

/-- Converts a number from base 2 to decimal -/
def base2ToDecimal (n : Nat) : Nat :=
  (n / 100000) * 32 + ((n / 10000) % 10) * 16 + ((n / 1000) % 10) * 8 +
  ((n / 100) % 10) * 4 + ((n / 10) % 10) * 2 + (n % 10)

theorem smallest_number (n1 n2 n3 : Nat) 
  (h1 : n1 = 210)
  (h2 : n2 = 1000)
  (h3 : n3 = 111111) :
  base2ToDecimal n3 < base6ToDecimal n1 ∧ base2ToDecimal n3 < base4ToDecimal n2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1514_151478


namespace NUMINAMATH_CALUDE_largest_common_term_correct_l1514_151433

/-- First arithmetic progression with initial term 4 and common difference 5 -/
def ap1 (n : ℕ) : ℕ := 4 + 5 * n

/-- Second arithmetic progression with initial term 5 and common difference 10 -/
def ap2 (n : ℕ) : ℕ := 5 + 10 * n

/-- Predicate to check if a number is in both arithmetic progressions -/
def isCommonTerm (x : ℕ) : Prop :=
  ∃ n m : ℕ, ap1 n = x ∧ ap2 m = x

/-- The largest common term less than 300 -/
def largestCommonTerm : ℕ := 299

theorem largest_common_term_correct :
  isCommonTerm largestCommonTerm ∧
  largestCommonTerm < 300 ∧
  ∀ x : ℕ, isCommonTerm x → x < 300 → x ≤ largestCommonTerm :=
by sorry

#check largest_common_term_correct

end NUMINAMATH_CALUDE_largest_common_term_correct_l1514_151433


namespace NUMINAMATH_CALUDE_credit_card_more_beneficial_l1514_151434

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 8000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.0075

/-- Represents the monthly interest rate on the debit card balance -/
def debit_interest_rate : ℝ := 0.005

/-- Calculates the total income from using the credit card -/
def credit_card_income (amount : ℝ) : ℝ :=
  amount * credit_cashback_rate + amount * debit_interest_rate

/-- Calculates the total income from using the debit card -/
def debit_card_income (amount : ℝ) : ℝ :=
  amount * debit_cashback_rate

/-- Theorem stating that the credit card is more beneficial -/
theorem credit_card_more_beneficial :
  credit_card_income purchase_amount > debit_card_income purchase_amount :=
by sorry


end NUMINAMATH_CALUDE_credit_card_more_beneficial_l1514_151434


namespace NUMINAMATH_CALUDE_consecutive_points_distance_l1514_151422

/-- Given five consecutive points on a straight line, if certain distance conditions are met,
    then the distance between the last two points is 4. -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (b - a) + (c - b) + (d - c) + (e - d) = (e - a)  -- Points are consecutive on a line
  → (c - b) = 2 * (d - c)  -- bc = 2 cd
  → (b - a) = 5  -- ab = 5
  → (c - a) = 11  -- ac = 11
  → (e - a) = 18  -- ae = 18
  → (e - d) = 4  -- de = 4
:= by sorry

end NUMINAMATH_CALUDE_consecutive_points_distance_l1514_151422


namespace NUMINAMATH_CALUDE_Q_equals_G_l1514_151411

-- Define the sets Q and G
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def G : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_equals_G : Q = G := by sorry

end NUMINAMATH_CALUDE_Q_equals_G_l1514_151411


namespace NUMINAMATH_CALUDE_smallest_debt_theorem_l1514_151453

/-- The value of a pig in dollars -/
def pig_value : ℕ := 250

/-- The value of a goat in dollars -/
def goat_value : ℕ := 175

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 125

/-- The smallest positive debt that can be resolved -/
def smallest_resolvable_debt : ℕ := 25

theorem smallest_debt_theorem :
  (∃ (p g s : ℤ), smallest_resolvable_debt = pig_value * p + goat_value * g + sheep_value * s) ∧
  (∀ (d : ℕ), d > 0 ∧ d < smallest_resolvable_debt →
    ¬∃ (p g s : ℤ), d = pig_value * p + goat_value * g + sheep_value * s) :=
by sorry

end NUMINAMATH_CALUDE_smallest_debt_theorem_l1514_151453


namespace NUMINAMATH_CALUDE_decision_box_distinguishes_l1514_151490

/-- Represents a flowchart element --/
inductive FlowchartElement
  | ProcessingBox
  | DecisionBox
  | InputOutputBox
  | StartEndBox

/-- Represents a flowchart structure --/
structure FlowchartStructure :=
  (elements : Set FlowchartElement)

/-- Definition of a conditional structure --/
def is_conditional (s : FlowchartStructure) : Prop :=
  FlowchartElement.DecisionBox ∈ s.elements ∧ 
  (∃ (b1 b2 : Set FlowchartElement), b1 ⊆ s.elements ∧ b2 ⊆ s.elements ∧ b1 ≠ b2)

/-- Definition of a sequential structure --/
def is_sequential (s : FlowchartStructure) : Prop :=
  FlowchartElement.DecisionBox ∉ s.elements

/-- Theorem: The inclusion of a decision box distinguishes conditional from sequential structures --/
theorem decision_box_distinguishes :
  ∀ (s : FlowchartStructure), 
    (is_conditional s ↔ FlowchartElement.DecisionBox ∈ s.elements) ∧
    (is_sequential s ↔ FlowchartElement.DecisionBox ∉ s.elements) :=
by sorry

end NUMINAMATH_CALUDE_decision_box_distinguishes_l1514_151490


namespace NUMINAMATH_CALUDE_trees_represents_41225_l1514_151499

-- Define the type for our digit mapping
def DigitMapping := Char → Option Nat

-- Define our specific mapping
def greatSuccessMapping : DigitMapping := fun c =>
  match c with
  | 'G' => some 0
  | 'R' => some 1
  | 'E' => some 2
  | 'A' => some 3
  | 'T' => some 4
  | 'S' => some 5
  | 'U' => some 6
  | 'C' => some 7
  | _ => none

-- Function to convert a string to a number using the mapping
def stringToNumber (s : String) (m : DigitMapping) : Option Nat :=
  s.foldr (fun c acc =>
    match acc, m c with
    | some n, some d => some (n * 10 + d)
    | _, _ => none
  ) (some 0)

-- Theorem statement
theorem trees_represents_41225 :
  stringToNumber "TREES" greatSuccessMapping = some 41225 := by
  sorry

end NUMINAMATH_CALUDE_trees_represents_41225_l1514_151499


namespace NUMINAMATH_CALUDE_jokes_increase_factor_l1514_151425

/-- The factor by which Jessy and Alan increased their jokes -/
def increase_factor (first_saturday_jokes : ℕ) (total_jokes : ℕ) : ℚ :=
  (total_jokes - first_saturday_jokes : ℚ) / first_saturday_jokes

/-- Theorem stating that the increase factor is 2 -/
theorem jokes_increase_factor : increase_factor 18 54 = 2 := by
  sorry

#eval increase_factor 18 54

end NUMINAMATH_CALUDE_jokes_increase_factor_l1514_151425


namespace NUMINAMATH_CALUDE_angle_complement_when_supplement_is_110_l1514_151498

/-- If the supplement of an angle is 110°, then its complement is 20°. -/
theorem angle_complement_when_supplement_is_110 (x : ℝ) : 
  x + 110 = 180 → 90 - (180 - 110) = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_when_supplement_is_110_l1514_151498


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1514_151466

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r1 r2 d : ℝ) : Prop := d = r1 + r2

/-- Given two circles with radii 2 and 3, and the distance between their centers is 5,
    prove that they are externally tangent -/
theorem circles_externally_tangent :
  let r1 : ℝ := 2
  let r2 : ℝ := 3
  let d : ℝ := 5
  externally_tangent r1 r2 d := by
sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1514_151466


namespace NUMINAMATH_CALUDE_range_of_a_l1514_151439

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > a → x > 2) ∧ (∃ x : ℝ, x > 2 ∧ x ≤ a) ↔ a > 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1514_151439


namespace NUMINAMATH_CALUDE_positive_A_value_l1514_151486

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 290) : A = Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l1514_151486


namespace NUMINAMATH_CALUDE_not_prime_n_l1514_151448

theorem not_prime_n (p a b c n : ℕ) : 
  Prime p → 
  0 < a → 0 < b → 0 < c → 0 < n →
  a < p → b < p → c < p →
  p^2 ∣ a + (n-1) * b →
  p^2 ∣ b + (n-1) * c →
  p^2 ∣ c + (n-1) * a →
  ¬ Prime n :=
by sorry


end NUMINAMATH_CALUDE_not_prime_n_l1514_151448


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1514_151441

/-- Given a geometric sequence {aₙ} where a₁ = 1/3 and 2a₂ = a₄, prove that a₅ = 4/3 -/
theorem geometric_sequence_fifth_term (a : ℕ → ℚ) (h1 : a 1 = 1/3) (h2 : 2 * a 2 = a 4) :
  a 5 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1514_151441


namespace NUMINAMATH_CALUDE_p_one_eq_p_two_p_decreasing_l1514_151449

/-- The number of items in the collection -/
def n : ℕ := 10

/-- The probability of finding any specific item in a randomly chosen container -/
def prob_item : ℝ := 0.1

/-- The probability that exactly k items are missing from the second collection when the first collection is completed -/
noncomputable def p (k : ℕ) : ℝ := sorry

/-- Theorem stating that p_1 equals p_2 -/
theorem p_one_eq_p_two : p 1 = p 2 := by sorry

/-- Theorem stating the strict decreasing order of probabilities -/
theorem p_decreasing {i j : ℕ} (h1 : 2 ≤ i) (h2 : i < j) (h3 : j ≤ n) : p i > p j := by sorry

end NUMINAMATH_CALUDE_p_one_eq_p_two_p_decreasing_l1514_151449


namespace NUMINAMATH_CALUDE_only_event3_mutually_exclusive_l1514_151427

-- Define the set of numbers
def numbers : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the concept of odd and even numbers
def isOdd (n : Nat) : Prop := n % 2 = 1
def isEven (n : Nat) : Prop := n % 2 = 0

-- Define the events
def event1 (a b : Nat) : Prop := (isOdd a ∧ isEven b) ∨ (isEven a ∧ isOdd b)
def event2 (a b : Nat) : Prop := isOdd a ∨ isOdd b
def event3 (a b : Nat) : Prop := (isOdd a ∨ isOdd b) ∧ (isEven a ∧ isEven b)
def event4 (a b : Nat) : Prop := (isOdd a ∨ isOdd b) ∧ (isEven a ∨ isEven b)

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : (Nat → Nat → Prop)) : Prop :=
  ∀ a b, a ∈ numbers → b ∈ numbers → ¬(e1 a b ∧ e2 a b)

-- Theorem statement
theorem only_event3_mutually_exclusive :
  (mutuallyExclusive event1 event3) ∧
  (¬mutuallyExclusive event1 event1) ∧
  (¬mutuallyExclusive event2 event4) ∧
  (¬mutuallyExclusive event4 event4) :=
sorry


end NUMINAMATH_CALUDE_only_event3_mutually_exclusive_l1514_151427


namespace NUMINAMATH_CALUDE_num_dogs_in_pool_l1514_151445

-- Define the total number of legs/paws in the pool
def total_legs : ℕ := 24

-- Define the number of humans in the pool
def num_humans : ℕ := 2

-- Define the number of legs per human
def legs_per_human : ℕ := 2

-- Define the number of legs per dog
def legs_per_dog : ℕ := 4

-- Theorem to prove
theorem num_dogs_in_pool : 
  (total_legs - num_humans * legs_per_human) / legs_per_dog = 5 := by
  sorry


end NUMINAMATH_CALUDE_num_dogs_in_pool_l1514_151445


namespace NUMINAMATH_CALUDE_jace_road_trip_distance_l1514_151431

/-- Represents a driving segment with speed in miles per hour and duration in hours -/
structure DrivingSegment where
  speed : ℝ
  duration : ℝ

/-- Calculates the total distance covered given a list of driving segments -/
def totalDistance (segments : List DrivingSegment) : ℝ :=
  segments.foldl (fun acc segment => acc + segment.speed * segment.duration) 0

/-- Jace's road trip theorem -/
theorem jace_road_trip_distance :
  let segments : List DrivingSegment := [
    { speed := 50, duration := 3 },
    { speed := 65, duration := 4.5 },
    { speed := 60, duration := 2.75 },
    { speed := 75, duration := 1.8333 },
    { speed := 55, duration := 2.6667 }
  ]
  ∃ ε > 0, |totalDistance segments - 891.67| < ε :=
by sorry

end NUMINAMATH_CALUDE_jace_road_trip_distance_l1514_151431


namespace NUMINAMATH_CALUDE_integer_pair_theorem_l1514_151477

/-- Given positive integers a and b where a > b, prove that a²b - ab² = 30 
    if and only if (a, b) is one of (5, 2), (5, 3), (6, 1), or (6, 5) -/
theorem integer_pair_theorem (a b : ℕ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) :
  a^2 * b - a * b^2 = 30 ↔ 
  ((a = 5 ∧ b = 2) ∨ (a = 5 ∧ b = 3) ∨ (a = 6 ∧ b = 1) ∨ (a = 6 ∧ b = 5)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_theorem_l1514_151477


namespace NUMINAMATH_CALUDE_men_work_hours_per_day_l1514_151463

-- Define the number of men, women, and days
def num_men : ℕ := 15
def num_women : ℕ := 21
def days_men : ℕ := 21
def days_women : ℕ := 20
def hours_women : ℕ := 9

-- Define the ratio of work done by women to men
def women_to_men_ratio : ℚ := 2 / 3

-- Define the function to calculate total work hours
def total_work_hours (num_workers : ℕ) (num_days : ℕ) (hours_per_day : ℕ) : ℕ :=
  num_workers * num_days * hours_per_day

-- Theorem statement
theorem men_work_hours_per_day :
  ∃ (hours_men : ℕ),
    (total_work_hours num_men days_men hours_men : ℚ) * women_to_men_ratio =
    (total_work_hours num_women days_women hours_women : ℚ) ∧
    hours_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_men_work_hours_per_day_l1514_151463


namespace NUMINAMATH_CALUDE_max_value_on_curves_l1514_151417

theorem max_value_on_curves (m n x y : ℝ) (α β : ℝ) : 
  m = Real.sqrt 6 * Real.cos α →
  n = Real.sqrt 6 * Real.sin α →
  x = Real.sqrt 24 * Real.cos β →
  y = Real.sqrt 24 * Real.sin β →
  (∀ m' n' x' y' α' β', 
    m' = Real.sqrt 6 * Real.cos α' →
    n' = Real.sqrt 6 * Real.sin α' →
    x' = Real.sqrt 24 * Real.cos β' →
    y' = Real.sqrt 24 * Real.sin β' →
    m' * x' + n' * y' ≤ 12) ∧
  (∃ m' n' x' y' α' β', 
    m' = Real.sqrt 6 * Real.cos α' ∧
    n' = Real.sqrt 6 * Real.sin α' ∧
    x' = Real.sqrt 24 * Real.cos β' ∧
    y' = Real.sqrt 24 * Real.sin β' ∧
    m' * x' + n' * y' = 12) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_curves_l1514_151417


namespace NUMINAMATH_CALUDE_binary_product_theorem_l1514_151405

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- The main theorem stating that the product of the given binary numbers equals the expected result -/
theorem binary_product_theorem :
  let a := [false, false, true, true, false, true]  -- 101100₂
  let b := [true, true, true]                       -- 111₂
  let c := [false, true]                            -- 10₂
  let result := [false, false, true, false, true, true, false, false, true]  -- 100110100₂
  binary_to_decimal a * binary_to_decimal b * binary_to_decimal c = binary_to_decimal result := by
  sorry


end NUMINAMATH_CALUDE_binary_product_theorem_l1514_151405


namespace NUMINAMATH_CALUDE_dot_product_range_l1514_151416

/-- A circle centered at the origin and tangent to the line x-√3y=4 -/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- The line x-√3y=4 -/
def TangentLine := {p : ℝ × ℝ | p.1 - Real.sqrt 3 * p.2 = 4}

/-- Point A where the circle intersects the negative x-axis -/
def A : ℝ × ℝ := (-2, 0)

/-- Point B where the circle intersects the positive x-axis -/
def B : ℝ × ℝ := (2, 0)

/-- A point P inside the circle satisfying the geometric sequence condition -/
def P := {p : ℝ × ℝ | p ∈ Circle ∧ p.1^2 = p.2^2 + 2}

/-- The dot product of PA and PB -/
def dotProduct (p : ℝ × ℝ) : ℝ := 
  (A.1 - p.1) * (B.1 - p.1) + (A.2 - p.2) * (B.2 - p.2)

theorem dot_product_range :
  ∀ p ∈ P, -2 ≤ dotProduct p ∧ dotProduct p < 0 := by sorry

end NUMINAMATH_CALUDE_dot_product_range_l1514_151416


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l1514_151474

-- Problem 1
theorem factorization_1 (a : ℝ) : 3*a^3 - 6*a^2 + 3*a = 3*a*(a - 1)^2 := by sorry

-- Problem 2
theorem factorization_2 (a b x y : ℝ) : a^2*(x - y) + b^2*(y - x) = (x - y)*(a - b)*(a + b) := by sorry

-- Problem 3
theorem factorization_3 (a b : ℝ) : 16*(a + b)^2 - 9*(a - b)^2 = (a + 7*b)*(7*a + b) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l1514_151474


namespace NUMINAMATH_CALUDE_bakery_sugar_amount_l1514_151429

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount 
  (sugar flour baking_soda : ℝ)
  (h1 : sugar / flour = 5 / 6)
  (h2 : flour / baking_soda = 10)
  (h3 : flour / (baking_soda + 60) = 8) :
  sugar = 2000 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sugar_amount_l1514_151429


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1514_151438

theorem other_root_of_quadratic (b : ℝ) : 
  ((-1 : ℝ)^2 + b * (-1) - 5 = 0) → 
  (∃ x : ℝ, x ≠ -1 ∧ x^2 + b*x - 5 = 0 ∧ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1514_151438


namespace NUMINAMATH_CALUDE_circle_equation_with_hyperbola_asymptotes_as_tangents_l1514_151482

/-- The standard equation of a circle with center (0,5) and tangents that are the asymptotes of the hyperbola x^2 - y^2 = 1 -/
theorem circle_equation_with_hyperbola_asymptotes_as_tangents :
  ∃ (r : ℝ),
    (∀ (x y : ℝ), x^2 + (y - 5)^2 = r^2 ↔
      (∃ (t : ℝ), (x = t ∧ y = t + 5) ∨ (x = -t ∧ y = -t + 5))) ∧
    r^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_hyperbola_asymptotes_as_tangents_l1514_151482


namespace NUMINAMATH_CALUDE_min_value_implies_t_l1514_151407

/-- Given a real number t, f(x) is defined as the sum of the absolute values of (x-t) and (5-x) -/
def f (t : ℝ) (x : ℝ) : ℝ := |x - t| + |5 - x|

/-- The theorem states that if the minimum value of f(x) is 3, then t must be either 2 or 8 -/
theorem min_value_implies_t (t : ℝ) (h : ∀ x, f t x ≥ 3) (h_min : ∃ x, f t x = 3) : t = 2 ∨ t = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_t_l1514_151407


namespace NUMINAMATH_CALUDE_sum_x_y_equals_one_l1514_151442

theorem sum_x_y_equals_one (x y : ℝ) 
  (eq1 : x + 2*y = 1) 
  (eq2 : 2*x + y = 2) : 
  x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_one_l1514_151442


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_of_factorials_15_l1514_151483

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_sum_of_factorials_15 :
  sum_of_factorials 15 % 100 = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_of_factorials_15_l1514_151483


namespace NUMINAMATH_CALUDE_cylinder_cut_face_area_l1514_151451

/-- Represents a cylinder with given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a cut through the cylinder -/
structure CylinderCut (c : Cylinder) where
  arcAngle : ℝ  -- Angle between the two points on the circular face

/-- The area of the rectangular face resulting from the cut -/
def cutFaceArea (c : Cylinder) (cut : CylinderCut c) : ℝ :=
  c.height * (2 * c.radius)

theorem cylinder_cut_face_area 
  (c : Cylinder) 
  (cut : CylinderCut c) 
  (h_radius : c.radius = 4) 
  (h_height : c.height = 10) 
  (h_angle : cut.arcAngle = π) : 
  cutFaceArea c cut = 80 := by
  sorry

#eval (80 : ℤ) + (0 : ℤ) + (1 : ℤ)  -- Should evaluate to 81

end NUMINAMATH_CALUDE_cylinder_cut_face_area_l1514_151451


namespace NUMINAMATH_CALUDE_count_primes_between_50_and_70_l1514_151435

theorem count_primes_between_50_and_70 : 
  (Finset.filter Nat.Prime (Finset.range 19)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_primes_between_50_and_70_l1514_151435


namespace NUMINAMATH_CALUDE_split_investment_average_rate_l1514_151457

/-- The average interest rate for a split investment --/
theorem split_investment_average_rate (total_investment : ℝ) 
  (rate1 rate2 : ℝ) (fee : ℝ) : 
  total_investment > 0 →
  rate1 > 0 →
  rate2 > 0 →
  rate1 < rate2 →
  fee > 0 →
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < total_investment ∧
    rate1 * (total_investment - x) - fee = rate2 * x →
  (rate2 * x + (rate1 * (total_investment - x) - fee)) / total_investment = 0.05133 :=
by sorry

end NUMINAMATH_CALUDE_split_investment_average_rate_l1514_151457


namespace NUMINAMATH_CALUDE_base_8_addition_problem_l1514_151406

/-- Converts a base 8 digit to base 10 -/
def to_base_10 (d : Nat) : Nat :=
  d

/-- Converts a base 10 number to base 8 -/
def to_base_8 (n : Nat) : Nat :=
  n

theorem base_8_addition_problem (X Y : Nat) 
  (h1 : X < 8 ∧ Y < 8)  -- X and Y are single digits in base 8
  (h2 : to_base_8 (4 * 8 * 8 + X * 8 + Y) + to_base_8 (5 * 8 + 3) = to_base_8 (6 * 8 * 8 + 1 * 8 + X)) :
  to_base_10 X + to_base_10 Y = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_8_addition_problem_l1514_151406


namespace NUMINAMATH_CALUDE_three_white_marbles_possible_l1514_151430

/-- Represents the state of the urn -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Represents the possible operations on the urn -/
inductive Operation
  | op1 | op2 | op3 | op4 | op5

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.op1 => UrnState.mk state.white (state.black - 2)
  | Operation.op2 => UrnState.mk (state.white - 1) (state.black - 2)
  | Operation.op3 => UrnState.mk state.white (state.black - 1)
  | Operation.op4 => UrnState.mk state.white (state.black - 1)
  | Operation.op5 => UrnState.mk (state.white - 3) (state.black + 2)

/-- Applies a sequence of operations to the urn state -/
def applyOperations (initial : UrnState) (ops : List Operation) : UrnState :=
  ops.foldl applyOperation initial

/-- The theorem to be proved -/
theorem three_white_marbles_possible :
  ∃ (ops : List Operation),
    let final := applyOperations (UrnState.mk 150 50) ops
    final.white = 3 ∧ final.black ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_three_white_marbles_possible_l1514_151430


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1514_151443

theorem fraction_multiplication : (1 / 2) * (3 / 5) * (7 / 11) * (4 / 13) = 84 / 1430 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1514_151443


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l1514_151455

theorem larger_number_of_pair (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 50) : max x y = 29 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l1514_151455


namespace NUMINAMATH_CALUDE_sector_radius_l1514_151461

/-- Given a sector with arc length and area, calculate its radius -/
theorem sector_radius (arc_length : ℝ) (area : ℝ) (radius : ℝ) : 
  arc_length = 2 → area = 2 → (1/2) * arc_length * radius = area → radius = 2 := by
  sorry

#check sector_radius

end NUMINAMATH_CALUDE_sector_radius_l1514_151461


namespace NUMINAMATH_CALUDE_choose_four_from_nine_l1514_151419

theorem choose_four_from_nine (n : ℕ) (k : ℕ) : n = 9 ∧ k = 4 → Nat.choose n k = 126 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_nine_l1514_151419
