import Mathlib

namespace NUMINAMATH_CALUDE_cubic_symmetry_l1155_115500

/-- A cubic function of the form ax^3 + bx + 6 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 6

/-- Theorem: For a cubic function f(x) = ax^3 + bx + 6, if f(5) = 7, then f(-5) = 5 -/
theorem cubic_symmetry (a b : ℝ) : f a b 5 = 7 → f a b (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_symmetry_l1155_115500


namespace NUMINAMATH_CALUDE_morgan_experiment_correct_l1155_115549

/-- Statements about biological experiments and research -/
inductive BiologicalStatement
| A : BiologicalStatement -- Ovary of locusts for observing animal cell meiosis
| B : BiologicalStatement -- Morgan's fruit fly experiment
| C : BiologicalStatement -- Hydrogen peroxide as substrate in enzyme activity experiment
| D : BiologicalStatement -- Investigating red-green color blindness incidence

/-- Predicate to determine if a biological statement is correct -/
def is_correct : BiologicalStatement → Prop
| BiologicalStatement.A => False
| BiologicalStatement.B => True
| BiologicalStatement.C => False
| BiologicalStatement.D => False

/-- Theorem stating that Morgan's fruit fly experiment statement is correct -/
theorem morgan_experiment_correct :
  is_correct BiologicalStatement.B :=
by sorry

end NUMINAMATH_CALUDE_morgan_experiment_correct_l1155_115549


namespace NUMINAMATH_CALUDE_bike_clamps_theorem_l1155_115537

/-- The number of bike clamps given away with each bicycle sale -/
def clamps_per_bike : ℕ := 2

/-- The number of bikes sold in the morning -/
def morning_sales : ℕ := 19

/-- The number of bikes sold in the afternoon -/
def afternoon_sales : ℕ := 27

/-- The total number of bike clamps given away -/
def total_clamps : ℕ := clamps_per_bike * (morning_sales + afternoon_sales)

theorem bike_clamps_theorem : total_clamps = 92 := by
  sorry

end NUMINAMATH_CALUDE_bike_clamps_theorem_l1155_115537


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l1155_115524

theorem isosceles_triangle_leg_length 
  (base : ℝ) 
  (leg : ℝ) 
  (h1 : base = 8) 
  (h2 : leg^2 - 9*leg + 20 = 0) 
  (h3 : leg > base/2) : 
  leg = 5 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l1155_115524


namespace NUMINAMATH_CALUDE_pea_patch_problem_l1155_115561

theorem pea_patch_problem (radish_patch : ℝ) (pea_patch : ℝ) :
  radish_patch = 15 →
  pea_patch = 2 * radish_patch →
  pea_patch / 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_pea_patch_problem_l1155_115561


namespace NUMINAMATH_CALUDE_total_potatoes_l1155_115574

theorem total_potatoes (nancy_potatoes sandy_potatoes : ℕ) 
  (h1 : nancy_potatoes = 6) 
  (h2 : sandy_potatoes = 7) : 
  nancy_potatoes + sandy_potatoes = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_potatoes_l1155_115574


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1155_115543

theorem gcd_of_specific_numbers : Nat.gcd 333333 9999999 = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1155_115543


namespace NUMINAMATH_CALUDE_limit_of_polynomial_at_two_l1155_115578

theorem limit_of_polynomial_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |4*x^2 - 6*x + 3 - 7| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_polynomial_at_two_l1155_115578


namespace NUMINAMATH_CALUDE_sticker_distribution_l1155_115583

theorem sticker_distribution (n m : ℕ) (hn : n = 5) (hm : m = 5) :
  (Nat.choose (n + m - 1) (m - 1) : ℕ) = 126 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1155_115583


namespace NUMINAMATH_CALUDE_lisa_spoons_count_l1155_115599

/-- The number of spoons Lisa has after combining all sets -/
def total_spoons (num_children : ℕ) (baby_spoons_per_child : ℕ) (decorative_spoons : ℕ)
  (large_spoons : ℕ) (dessert_spoons : ℕ) (soup_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * baby_spoons_per_child + decorative_spoons +
  large_spoons + dessert_spoons + soup_spoons + teaspoons

/-- Theorem stating that Lisa has 98 spoons in total -/
theorem lisa_spoons_count :
  total_spoons 6 4 4 20 10 15 25 = 98 := by
  sorry

end NUMINAMATH_CALUDE_lisa_spoons_count_l1155_115599


namespace NUMINAMATH_CALUDE_race_finish_order_l1155_115555

-- Define the athletes
inductive Athlete : Type
| Grisha : Athlete
| Sasha : Athlete
| Lena : Athlete

-- Define the race
structure Race where
  start_order : List Athlete
  finish_order : List Athlete
  overtakes : Athlete → Nat
  no_triple_overtake : Bool

-- Define the specific race conditions
def race_conditions (r : Race) : Prop :=
  r.start_order = [Athlete.Grisha, Athlete.Sasha, Athlete.Lena] ∧
  r.overtakes Athlete.Grisha = 10 ∧
  r.overtakes Athlete.Lena = 6 ∧
  r.overtakes Athlete.Sasha = 4 ∧
  r.no_triple_overtake = true ∧
  r.finish_order.length = 3 ∧
  r.finish_order.Nodup

-- Theorem statement
theorem race_finish_order (r : Race) :
  race_conditions r →
  r.finish_order = [Athlete.Grisha, Athlete.Sasha, Athlete.Lena] :=
by sorry

end NUMINAMATH_CALUDE_race_finish_order_l1155_115555


namespace NUMINAMATH_CALUDE_exist_positive_integers_with_nonzero_integer_roots_l1155_115571

theorem exist_positive_integers_with_nonzero_integer_roots :
  ∃ (a b c : ℕ+), 
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 + (b:ℤ) * x + (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 + (b:ℤ) * y + (c:ℤ) = 0) ∧
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 + (b:ℤ) * x - (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 + (b:ℤ) * y - (c:ℤ) = 0) ∧
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 - (b:ℤ) * x + (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 - (b:ℤ) * y + (c:ℤ) = 0) ∧
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 - (b:ℤ) * x - (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 - (b:ℤ) * y - (c:ℤ) = 0) :=
by sorry

end NUMINAMATH_CALUDE_exist_positive_integers_with_nonzero_integer_roots_l1155_115571


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l1155_115590

/-- Represents the number of basil plants -/
def basil_count : ℕ := 5

/-- Represents the number of tomato plants -/
def tomato_count : ℕ := 5

/-- Represents the total number of plant positions (basil + tomato block) -/
def total_positions : ℕ := basil_count + 1

/-- Calculates the number of ways to arrange the plants with given constraints -/
def plant_arrangements : ℕ :=
  (Nat.factorial total_positions) * (Nat.factorial tomato_count)

theorem plant_arrangement_count :
  plant_arrangements = 86400 := by sorry

end NUMINAMATH_CALUDE_plant_arrangement_count_l1155_115590


namespace NUMINAMATH_CALUDE_parabola_f_value_l1155_115507

/-- A parabola with equation x = dy² + ey + f, vertex at (5, 3), and passing through (2, 6) -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_condition : 5 = d * 3^2 + e * 3 + f
  point_condition : 2 = d * 6^2 + e * 6 + f

/-- The value of f for the given parabola is 2 -/
theorem parabola_f_value (p : Parabola) : p.f = 2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_f_value_l1155_115507


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1155_115508

theorem sum_of_squares_of_roots (u v w : ℝ) : 
  (3 * u^3 - 7 * u^2 + 6 * u + 15 = 0) →
  (3 * v^3 - 7 * v^2 + 6 * v + 15 = 0) →
  (3 * w^3 - 7 * w^2 + 6 * w + 15 = 0) →
  u^2 + v^2 + w^2 = 13/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1155_115508


namespace NUMINAMATH_CALUDE_power_of_seven_mod_hundred_l1155_115575

theorem power_of_seven_mod_hundred : 7^700 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_hundred_l1155_115575


namespace NUMINAMATH_CALUDE_system_solution_l1155_115504

theorem system_solution (x y z w : ℝ) : 
  (x - y + z - w = 2) ∧
  (x^2 - y^2 + z^2 - w^2 = 6) ∧
  (x^3 - y^3 + z^3 - w^3 = 20) ∧
  (x^4 - y^4 + z^4 - w^4 = 60) →
  ((x = 1 ∧ y = 2 ∧ z = 3 ∧ w = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 3 ∧ w = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ w = 0) ∨
   (x = 3 ∧ y = 0 ∧ z = 1 ∧ w = 2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1155_115504


namespace NUMINAMATH_CALUDE_no_integer_list_with_mean_6_35_l1155_115577

theorem no_integer_list_with_mean_6_35 :
  ¬ ∃ (lst : List ℤ), lst.length = 35 ∧ (lst.sum : ℚ) / 35 = 35317 / 5560 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_list_with_mean_6_35_l1155_115577


namespace NUMINAMATH_CALUDE_solve_equation_l1155_115530

theorem solve_equation (x : ℝ) (h : x - 2*x + 3*x = 100) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1155_115530


namespace NUMINAMATH_CALUDE_total_crosswalk_lines_l1155_115570

/-- Given 5 intersections, 4 crosswalks per intersection, and 20 lines per crosswalk,
    the total number of lines in all crosswalks is 400. -/
theorem total_crosswalk_lines
  (num_intersections : ℕ)
  (crosswalks_per_intersection : ℕ)
  (lines_per_crosswalk : ℕ)
  (h1 : num_intersections = 5)
  (h2 : crosswalks_per_intersection = 4)
  (h3 : lines_per_crosswalk = 20) :
  num_intersections * crosswalks_per_intersection * lines_per_crosswalk = 400 :=
by sorry

end NUMINAMATH_CALUDE_total_crosswalk_lines_l1155_115570


namespace NUMINAMATH_CALUDE_families_with_items_l1155_115505

theorem families_with_items (total_telephone : ℕ) (total_tricycle : ℕ) (both : ℕ)
  (h1 : total_telephone = 35)
  (h2 : total_tricycle = 65)
  (h3 : both = 20) :
  total_telephone + total_tricycle - both = 80 := by
  sorry

end NUMINAMATH_CALUDE_families_with_items_l1155_115505


namespace NUMINAMATH_CALUDE_book_price_proof_l1155_115587

theorem book_price_proof (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 75)
  (h2 : profit_percentage = 25) :
  ∃ original_price : ℝ, 
    original_price * (1 + profit_percentage / 100) = selling_price ∧ 
    original_price = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_book_price_proof_l1155_115587


namespace NUMINAMATH_CALUDE_bumper_car_queue_count_l1155_115560

theorem bumper_car_queue_count : ∀ (initial leaving joining : ℕ),
  initial = 9 →
  leaving = 6 →
  joining = 3 →
  initial + joining = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_bumper_car_queue_count_l1155_115560


namespace NUMINAMATH_CALUDE_monroe_collection_legs_l1155_115567

/-- Represents the number of legs for each type of creature -/
structure CreatureLegs where
  ant : Nat
  spider : Nat
  beetle : Nat
  centipede : Nat

/-- Represents the count of each type of creature in the collection -/
structure CreatureCount where
  ants : Nat
  spiders : Nat
  beetles : Nat
  centipedes : Nat

/-- Calculates the total number of legs in the collection -/
def totalLegs (legs : CreatureLegs) (count : CreatureCount) : Nat :=
  legs.ant * count.ants + 
  legs.spider * count.spiders + 
  legs.beetle * count.beetles + 
  legs.centipede * count.centipedes

/-- Theorem: The total number of legs in Monroe's collection is 726 -/
theorem monroe_collection_legs : 
  let legs : CreatureLegs := { ant := 6, spider := 8, beetle := 6, centipede := 100 }
  let count : CreatureCount := { ants := 12, spiders := 8, beetles := 15, centipedes := 5 }
  totalLegs legs count = 726 := by
  sorry

end NUMINAMATH_CALUDE_monroe_collection_legs_l1155_115567


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1155_115520

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1155_115520


namespace NUMINAMATH_CALUDE_pool_capacity_l1155_115556

theorem pool_capacity (C : ℝ) 
  (h1 : 0.8 * C - 0.5 * C = 300) : C = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l1155_115556


namespace NUMINAMATH_CALUDE_finite_difference_polynomial_l1155_115569

/-- The finite difference operator -/
def finite_difference (f : ℕ → ℚ) : ℕ → ℚ := λ x => f (x + 1) - f x

/-- The n-th finite difference -/
def nth_finite_difference (n : ℕ) (f : ℕ → ℚ) : ℕ → ℚ :=
  match n with
  | 0 => f
  | n + 1 => finite_difference (nth_finite_difference n f)

/-- Polynomial of degree m -/
def polynomial_degree_m (m : ℕ) (coeffs : Fin (m + 1) → ℚ) : ℕ → ℚ :=
  λ x => (Finset.range (m + 1)).sum (λ i => coeffs i * x^i)

theorem finite_difference_polynomial (m n : ℕ) (coeffs : Fin (m + 1) → ℚ) :
  (m < n → ∀ x, nth_finite_difference n (polynomial_degree_m m coeffs) x = 0) ∧
  (∀ x, nth_finite_difference m (polynomial_degree_m m coeffs) x = m.factorial * coeffs m) :=
sorry

end NUMINAMATH_CALUDE_finite_difference_polynomial_l1155_115569


namespace NUMINAMATH_CALUDE_janine_reading_ratio_l1155_115550

/-- The number of books Janine read last month -/
def books_last_month : ℕ := 5

/-- The number of pages in each book -/
def pages_per_book : ℕ := 10

/-- The total number of pages Janine read in two months -/
def total_pages : ℕ := 150

/-- The number of books Janine read this month -/
def books_this_month : ℕ := (total_pages - books_last_month * pages_per_book) / pages_per_book

/-- The ratio of books read this month to last month -/
def book_ratio : ℚ := books_this_month / books_last_month

theorem janine_reading_ratio :
  book_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_janine_reading_ratio_l1155_115550


namespace NUMINAMATH_CALUDE_negation_of_universal_quantification_l1155_115511

theorem negation_of_universal_quantification :
  (¬ ∀ x : ℝ, x + Real.log x > 0) ↔ (∃ x : ℝ, x + Real.log x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantification_l1155_115511


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1155_115521

theorem algebraic_simplification (m n : ℝ) : 3 * m^2 * n - 3 * m^2 * n = 0 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1155_115521


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l1155_115593

structure Survey where
  population_size : ℕ
  sample_size : ℕ
  has_distinct_subgroups : Bool

def is_large_population (s : Survey) : Bool :=
  s.population_size ≥ 1000

def is_small_population (s : Survey) : Bool :=
  s.population_size < 100

def stratified_sampling_appropriate (s : Survey) : Bool :=
  is_large_population s ∧ s.has_distinct_subgroups

def simple_random_sampling_appropriate (s : Survey) : Bool :=
  is_small_population s ∧ ¬s.has_distinct_subgroups

theorem appropriate_sampling_methods 
  (survey_A survey_B : Survey)
  (h_A : survey_A.population_size = 20000 ∧ survey_A.sample_size = 200 ∧ survey_A.has_distinct_subgroups = true)
  (h_B : survey_B.population_size = 15 ∧ survey_B.sample_size = 3 ∧ survey_B.has_distinct_subgroups = false) :
  stratified_sampling_appropriate survey_A ∧ simple_random_sampling_appropriate survey_B :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l1155_115593


namespace NUMINAMATH_CALUDE_common_ratio_equation_l1155_115592

/-- A geometric progression with positive terms where the first term is equal to the sum of the next three terms -/
structure GeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  sum_condition : a = a * r + a * r^2 + a * r^3

/-- The common ratio of the geometric progression satisfies the equation r^3 + r^2 + r - 1 = 0 -/
theorem common_ratio_equation (gp : GeometricProgression) : gp.r^3 + gp.r^2 + gp.r - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_equation_l1155_115592


namespace NUMINAMATH_CALUDE_brand_y_pen_price_l1155_115501

theorem brand_y_pen_price 
  (price_x : ℝ) 
  (total_pens : ℕ) 
  (total_cost : ℝ) 
  (num_x_pens : ℕ) 
  (h1 : price_x = 4)
  (h2 : total_pens = 12)
  (h3 : total_cost = 42)
  (h4 : num_x_pens = 6) :
  (total_cost - price_x * num_x_pens) / (total_pens - num_x_pens) = 3 := by
  sorry

end NUMINAMATH_CALUDE_brand_y_pen_price_l1155_115501


namespace NUMINAMATH_CALUDE_shortest_time_to_camp_l1155_115545

/-- The shortest time to reach the camp across a river -/
theorem shortest_time_to_camp (river_width : ℝ) (camp_distance : ℝ) 
  (swim_speed : ℝ) (walk_speed : ℝ) (h1 : river_width = 1) 
  (h2 : camp_distance = 1) (h3 : swim_speed = 2) (h4 : walk_speed = 3) :
  ∃ (t : ℝ), t = (1 + Real.sqrt 13) / (3 * Real.sqrt 13) ∧ 
  (∀ (x : ℝ), x ≥ 0 ∧ x ≤ 1 → 
    t ≤ x / swim_speed + (camp_distance - Real.sqrt (river_width^2 - x^2)) / walk_speed) :=
by sorry

end NUMINAMATH_CALUDE_shortest_time_to_camp_l1155_115545


namespace NUMINAMATH_CALUDE_min_value_theorem_l1155_115526

theorem min_value_theorem (a b : ℝ) (h1 : a + b = 2) (h2 : b > 0) :
  (∃ (x : ℝ), x = 1 / (2 * abs a) + abs a / b ∧
    (∀ (y : ℝ), y = 1 / (2 * abs a) + abs a / b → x ≤ y)) →
  (∃ (min_val : ℝ), min_val = 3/4 ∧
    (∃ (x : ℝ), x = 1 / (2 * abs a) + abs a / b ∧ x = min_val) ∧
    a = -2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1155_115526


namespace NUMINAMATH_CALUDE_men_in_business_class_l1155_115597

def total_passengers : ℕ := 160
def men_percentage : ℚ := 3/4
def business_class_percentage : ℚ := 1/4

theorem men_in_business_class : 
  ⌊(total_passengers : ℚ) * men_percentage * business_class_percentage⌋ = 30 := by
  sorry

end NUMINAMATH_CALUDE_men_in_business_class_l1155_115597


namespace NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l1155_115512

/-- Given two planes α and β with normal vectors n1 and n2 respectively,
    prove that if the planes are parallel, then k = 4. -/
theorem parallel_planes_normal_vectors (n1 n2 : ℝ × ℝ × ℝ) (k : ℝ) : 
  n1 = (1, 2, -2) → n2 = (-2, -4, k) → (∃ (c : ℝ), n1 = c • n2) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l1155_115512


namespace NUMINAMATH_CALUDE_largest_number_with_given_hcf_lcm_factors_l1155_115525

theorem largest_number_with_given_hcf_lcm_factors :
  ∀ a b c : ℕ+,
  (∃ (hcf : ℕ+) (lcm : ℕ+), 
    (Nat.gcd a b = hcf) ∧ 
    (Nat.gcd (Nat.gcd a b) c = hcf) ∧
    (Nat.lcm (Nat.lcm a b) c = lcm) ∧
    (hcf = 59) ∧
    (∃ (k : ℕ+), lcm = hcf * 13 * (2^4) * 23 * k)) →
  max a (max b c) = 282256 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_given_hcf_lcm_factors_l1155_115525


namespace NUMINAMATH_CALUDE_square_minus_product_l1155_115582

theorem square_minus_product (a : ℝ) : (a - 1)^2 - a*(a - 1) = -a + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_l1155_115582


namespace NUMINAMATH_CALUDE_patch_net_profit_l1155_115532

/-- Calculates the net profit from selling patches --/
theorem patch_net_profit (order_quantity : ℕ) (cost_per_patch : ℚ) (sell_price : ℚ) : 
  order_quantity = 100 ∧ cost_per_patch = 125/100 ∧ sell_price = 12 →
  (sell_price * order_quantity) - (cost_per_patch * order_quantity) = 1075 := by
  sorry

end NUMINAMATH_CALUDE_patch_net_profit_l1155_115532


namespace NUMINAMATH_CALUDE_factorization_proof_l1155_115598

theorem factorization_proof (x y : ℝ) : 
  (x^2 - 9*y^2 = (x+3*y)*(x-3*y)) ∧ 
  (x^2*y - 6*x*y + 9*y = y*(x-3)^2) ∧ 
  (9*(x+2*y)^2 - 4*(x-y)^2 = (5*x+4*y)*(x+8*y)) ∧ 
  ((x-1)*(x-3) + 1 = (x-2)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1155_115598


namespace NUMINAMATH_CALUDE_special_isosceles_sine_l1155_115589

/-- An isosceles triangle with a special property on inscribed rectangles -/
structure SpecialIsoscelesTriangle where
  -- The vertex angle of the isosceles triangle
  vertex_angle : ℝ
  -- The base and height of the isosceles triangle
  base : ℝ
  height : ℝ
  -- The isosceles property
  isosceles : base = height
  -- The property that all inscribed rectangles with two vertices on the base have the same perimeter
  constant_perimeter : ∀ (x : ℝ), 0 ≤ x → x ≤ base → 
    2 * (x + (base * (height - x)) / height) = base + height

/-- The main theorem stating that the sine of the vertex angle is 4/5 -/
theorem special_isosceles_sine (t : SpecialIsoscelesTriangle) : 
  Real.sin t.vertex_angle = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_special_isosceles_sine_l1155_115589


namespace NUMINAMATH_CALUDE_students_in_c_class_l1155_115591

theorem students_in_c_class (a b c : ℕ) : 
  a = 44 ∧ a + 2 = b ∧ b = c + 1 → c = 45 := by
  sorry

end NUMINAMATH_CALUDE_students_in_c_class_l1155_115591


namespace NUMINAMATH_CALUDE_max_stores_visited_is_four_l1155_115513

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  num_stores : ℕ
  num_shoppers : ℕ
  two_store_visitors : ℕ
  total_visits : ℕ

/-- The maximum number of stores visited by any individual -/
def max_stores_visited (s : ShoppingScenario) : ℕ :=
  let remaining_visits := s.total_visits - 2 * s.two_store_visitors
  let remaining_shoppers := s.num_shoppers - s.two_store_visitors
  let extra_visits := remaining_visits - remaining_shoppers
  1 + extra_visits

/-- Theorem stating the maximum number of stores visited by any individual in the given scenario -/
theorem max_stores_visited_is_four (s : ShoppingScenario) :
  s.num_stores = 8 ∧ 
  s.num_shoppers = 12 ∧ 
  s.two_store_visitors = 8 ∧ 
  s.total_visits = 23 →
  max_stores_visited s = 4 :=
by
  sorry

#eval max_stores_visited {num_stores := 8, num_shoppers := 12, two_store_visitors := 8, total_visits := 23}

end NUMINAMATH_CALUDE_max_stores_visited_is_four_l1155_115513


namespace NUMINAMATH_CALUDE_sam_mystery_books_l1155_115559

/-- Represents the number of books in each category --/
structure BookCount where
  adventure : ℕ
  mystery : ℕ
  used : ℕ
  new : ℕ

/-- The total number of books is the sum of used and new books --/
def total_books (b : BookCount) : ℕ := b.used + b.new

/-- Theorem stating the number of mystery books Sam bought --/
theorem sam_mystery_books :
  ∃ (b : BookCount),
    b.adventure = 13 ∧
    b.used = 15 ∧
    b.new = 15 ∧
    total_books b = b.adventure + b.mystery ∧
    b.mystery = 17 := by
  sorry

end NUMINAMATH_CALUDE_sam_mystery_books_l1155_115559


namespace NUMINAMATH_CALUDE_rest_worker_salary_l1155_115541

def workshop (total_workers : ℕ) (avg_salary : ℚ) (technicians : ℕ) (avg_technician_salary : ℚ) : Prop :=
  total_workers = 12 ∧
  avg_salary = 9000 ∧
  technicians = 6 ∧
  avg_technician_salary = 12000

theorem rest_worker_salary (total_workers : ℕ) (avg_salary : ℚ) (technicians : ℕ) (avg_technician_salary : ℚ) :
  workshop total_workers avg_salary technicians avg_technician_salary →
  (total_workers * avg_salary - technicians * avg_technician_salary) / (total_workers - technicians) = 6000 :=
by
  sorry

#check rest_worker_salary

end NUMINAMATH_CALUDE_rest_worker_salary_l1155_115541


namespace NUMINAMATH_CALUDE_john_zoo_animals_l1155_115502

def zoo_animals (snakes : ℕ) : ℕ :=
  let monkeys := 2 * snakes
  let lions := monkeys - 5
  let pandas := lions + 8
  let dogs := pandas / 3
  snakes + monkeys + lions + pandas + dogs

theorem john_zoo_animals :
  zoo_animals 15 = 114 := by sorry

end NUMINAMATH_CALUDE_john_zoo_animals_l1155_115502


namespace NUMINAMATH_CALUDE_min_positive_period_sin_l1155_115523

/-- The minimum positive period of the function y = 3 * sin(2x + π/4) is π -/
theorem min_positive_period_sin (x : ℝ) : 
  let f := fun x => 3 * Real.sin (2 * x + π / 4)
  ∃ p : ℝ, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
    ∀ q : ℝ, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_min_positive_period_sin_l1155_115523


namespace NUMINAMATH_CALUDE_congruence_problem_l1155_115534

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % (2^3) = 3^2 % (2^3))
  (h2 : (6 + x) % (3^3) = 2^2 % (3^3))
  (h3 : (8 + x) % (5^3) = 7^2 % (5^3)) :
  x % 30 = 1 := by sorry

end NUMINAMATH_CALUDE_congruence_problem_l1155_115534


namespace NUMINAMATH_CALUDE_power_product_rule_l1155_115503

theorem power_product_rule (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l1155_115503


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l1155_115546

theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width → 
  area = length * width → 
  area = 432 → 
  width = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l1155_115546


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_product_l1155_115522

theorem distinct_prime_factors_of_product : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → (p ∣ (86 * 88 * 90 * 92) ↔ p ∈ s)) ∧ 
  Finset.card s = 6 := by
sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_product_l1155_115522


namespace NUMINAMATH_CALUDE_inequality_contradiction_l1155_115519

theorem inequality_contradiction (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l1155_115519


namespace NUMINAMATH_CALUDE_correct_proposition_l1155_115558

theorem correct_proposition :
  ∀ (p q : Prop),
    (p ∨ q) →
    ¬(p ∧ q) →
    ¬p →
    (p ↔ (5 + 2 = 6)) →
    (q ↔ (6 > 2)) →
    (¬p ∧ q) :=
by sorry

end NUMINAMATH_CALUDE_correct_proposition_l1155_115558


namespace NUMINAMATH_CALUDE_percent_decrease_proof_l1155_115509

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 60) : 
  (original_price - sale_price) / original_price * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_proof_l1155_115509


namespace NUMINAMATH_CALUDE_line_plane_relations_l1155_115572

/-- Represents a plane in 3D space -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- Represents a line in 3D space -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  z₁ : ℝ
  m : ℝ
  n : ℝ
  p : ℝ

/-- Determines if a line is parallel to a plane -/
def isParallel (plane : Plane) (line : Line) : Prop :=
  plane.A * line.m + plane.B * line.n + plane.C * line.p = 0

/-- Determines if a line is perpendicular to a plane -/
def isPerpendicular (plane : Plane) (line : Line) : Prop :=
  plane.A / line.m = plane.B / line.n ∧ plane.B / line.n = plane.C / line.p

theorem line_plane_relations (plane : Plane) (line : Line) :
  (isParallel plane line ↔ plane.A * line.m + plane.B * line.n + plane.C * line.p = 0) ∧
  (isPerpendicular plane line ↔ plane.A / line.m = plane.B / line.n ∧ plane.B / line.n = plane.C / line.p) :=
sorry

end NUMINAMATH_CALUDE_line_plane_relations_l1155_115572


namespace NUMINAMATH_CALUDE_max_stamps_for_50_dollars_l1155_115595

/-- The maximum number of stamps that can be purchased with a given budget and stamp price -/
def maxStamps (budget : ℕ) (stampPrice : ℕ) : ℕ :=
  (budget / stampPrice : ℕ)

/-- Theorem stating the maximum number of stamps that can be purchased with $50 when stamps cost 45 cents each -/
theorem max_stamps_for_50_dollars : maxStamps 5000 45 = 111 := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_for_50_dollars_l1155_115595


namespace NUMINAMATH_CALUDE_machine_N_output_fraction_l1155_115539

/-- Represents the production time of a machine relative to machine N -/
structure MachineTime where
  relative_to_N : ℚ

/-- Represents the production rate of a machine -/
def production_rate (m : MachineTime) : ℚ := 1 / m.relative_to_N

/-- The production time of machine T -/
def machine_T : MachineTime := ⟨3/4⟩

/-- The production time of machine N -/
def machine_N : MachineTime := ⟨1⟩

/-- The production time of machine O -/
def machine_O : MachineTime := ⟨3/2⟩

/-- The total production rate of all machines -/
def total_rate : ℚ :=
  production_rate machine_T + production_rate machine_N + production_rate machine_O

/-- The fraction of total output produced by machine N -/
def fraction_by_N : ℚ := production_rate machine_N / total_rate

theorem machine_N_output_fraction :
  fraction_by_N = 1/3 := by sorry

end NUMINAMATH_CALUDE_machine_N_output_fraction_l1155_115539


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1155_115531

theorem quadratic_inequality_solution_set (x : ℝ) :
  (∃ y ∈ Set.Icc (24 - 2 * Real.sqrt 19) (24 + 2 * Real.sqrt 19), x = y) ↔ 
  x^2 - 48*x + 500 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1155_115531


namespace NUMINAMATH_CALUDE_optimal_chair_removal_l1155_115596

theorem optimal_chair_removal :
  let chairs_per_row : ℕ := 15
  let initial_chairs : ℕ := 150
  let expected_attendees : ℕ := 125
  let removed_chairs : ℕ := 45

  -- All rows are complete
  (initial_chairs - removed_chairs) % chairs_per_row = 0 ∧
  -- At least one row is empty
  (initial_chairs - removed_chairs) / chairs_per_row < initial_chairs / chairs_per_row ∧
  -- Remaining chairs are sufficient for attendees
  initial_chairs - removed_chairs ≥ expected_attendees ∧
  -- Minimizes empty seats
  ∀ (x : ℕ), x < removed_chairs →
    (initial_chairs - x) % chairs_per_row ≠ 0 ∨
    (initial_chairs - x) / chairs_per_row ≥ initial_chairs / chairs_per_row ∨
    initial_chairs - x < expected_attendees :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_chair_removal_l1155_115596


namespace NUMINAMATH_CALUDE_ellipse_distance_theorem_l1155_115588

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Theorem about the distance AF₂ in the given ellipse problem -/
theorem ellipse_distance_theorem (E : Ellipse) (F₁ F₂ A B : Point) : 
  -- F₁ and F₂ are the foci of E
  (∀ P : Point, distance P F₁ + distance P F₂ = 2 * E.a) →
  -- Line through F₁ intersects E at A and B
  (∃ t : ℝ, A = ⟨t * F₁.x, t * F₁.y⟩ ∧ B = ⟨(1 - t) * F₁.x, (1 - t) * F₁.y⟩) →
  -- |AF₁| = 3|F₁B|
  distance A F₁ = 3 * distance F₁ B →
  -- |AB| = 4
  distance A B = 4 →
  -- Perimeter of triangle ABF₂ is 16
  distance A B + distance B F₂ + distance F₂ A = 16 →
  -- Then |AF₂| = 5
  distance A F₂ = 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_distance_theorem_l1155_115588


namespace NUMINAMATH_CALUDE_fraction_equality_l1155_115553

theorem fraction_equality (w x y : ℝ) 
  (h1 : w / x = 1 / 6)
  (h2 : (x + y) / y = 2.2) :
  w / y = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1155_115553


namespace NUMINAMATH_CALUDE_salary_after_changes_l1155_115551

/-- Given an original salary, calculate the final salary after a raise and a reduction -/
def finalSalary (originalSalary : ℚ) (raisePercentage : ℚ) (reductionPercentage : ℚ) : ℚ :=
  let salaryAfterRaise := originalSalary * (1 + raisePercentage / 100)
  salaryAfterRaise * (1 - reductionPercentage / 100)

theorem salary_after_changes : 
  finalSalary 5000 10 5 = 5225 := by sorry

end NUMINAMATH_CALUDE_salary_after_changes_l1155_115551


namespace NUMINAMATH_CALUDE_school_students_count_l1155_115536

theorem school_students_count : ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧ 
  n % 6 = 1 ∧ n % 8 = 2 ∧ n % 9 = 3 ∧ n = 265 := by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l1155_115536


namespace NUMINAMATH_CALUDE_distance_product_l1155_115584

noncomputable def f (x : ℝ) : ℝ := 2 * x + 5 / x

theorem distance_product (x : ℝ) (hx : x ≠ 0) :
  let P : ℝ × ℝ := (x, f x)
  let d₁ : ℝ := |f x - 2 * x| / Real.sqrt 5
  let d₂ : ℝ := |x|
  d₁ * d₂ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_product_l1155_115584


namespace NUMINAMATH_CALUDE_inverse_equals_original_at_three_l1155_115562

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - 9

-- Define the property of being an inverse function
def is_inverse (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

-- Theorem statement
theorem inverse_equals_original_at_three :
  ∃ g_inv : ℝ → ℝ, is_inverse g g_inv ∧
  ∀ x : ℝ, g x = g_inv x ↔ x = 3 :=
sorry

end NUMINAMATH_CALUDE_inverse_equals_original_at_three_l1155_115562


namespace NUMINAMATH_CALUDE_Q_has_negative_root_l1155_115527

/-- The polynomial Q(x) = x^7 - 4x^6 + 2x^5 - 9x^3 + 2x + 16 -/
def Q (x : ℝ) : ℝ := x^7 - 4*x^6 + 2*x^5 - 9*x^3 + 2*x + 16

/-- The polynomial Q(x) has at least one negative root -/
theorem Q_has_negative_root : ∃ x : ℝ, x < 0 ∧ Q x = 0 := by sorry

end NUMINAMATH_CALUDE_Q_has_negative_root_l1155_115527


namespace NUMINAMATH_CALUDE_mikes_cards_l1155_115517

theorem mikes_cards (x : ℕ) : x + 18 = 82 → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_mikes_cards_l1155_115517


namespace NUMINAMATH_CALUDE_initial_kola_percentage_l1155_115540

/-- Proves that the initial percentage of concentrated kola in a solution is 6% -/
theorem initial_kola_percentage (
  initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (final_sugar_percentage : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percentage = 80)
  (h3 : added_sugar = 3.2)
  (h4 : added_water = 10)
  (h5 : added_kola = 6.8)
  (h6 : final_sugar_percentage = 14.111111111111112)
  : ∃ (initial_kola_percentage : ℝ),
    initial_kola_percentage = 6 ∧
    (initial_volume - initial_water_percentage / 100 * initial_volume - initial_kola_percentage / 100 * initial_volume + added_sugar) /
    (initial_volume + added_sugar + added_water + added_kola) =
    final_sugar_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_kola_percentage_l1155_115540


namespace NUMINAMATH_CALUDE_annual_interest_calculation_l1155_115586

/-- Calculates the simple interest for a given principal, rate, and time. -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The loan amount in dollars -/
def loan_amount : ℝ := 9000

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.09

/-- The time period in years -/
def time_period : ℝ := 1

theorem annual_interest_calculation :
  simple_interest loan_amount interest_rate time_period = 810 := by
  sorry

end NUMINAMATH_CALUDE_annual_interest_calculation_l1155_115586


namespace NUMINAMATH_CALUDE_expression_equality_l1155_115533

theorem expression_equality : -1^4 + (-2)^3 / 4 * (5 - (-3)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1155_115533


namespace NUMINAMATH_CALUDE_expected_value_of_12_sided_die_l1155_115576

/-- A fair 12-sided die -/
def fair_12_sided_die : Finset ℕ := Finset.range 12

/-- The probability of each outcome for a fair 12-sided die -/
def prob (n : ℕ) : ℚ := if n ∈ fair_12_sided_die then 1 / 12 else 0

/-- The expected value of a roll of a fair 12-sided die -/
def expected_value : ℚ := (fair_12_sided_die.sum (λ x => x * prob x)) / 1

theorem expected_value_of_12_sided_die : expected_value = 13/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_12_sided_die_l1155_115576


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1155_115544

theorem rectangle_perimeter (x y : ℝ) 
  (h1 : 6 * x + 2 * y = 56)  -- perimeter of figure A
  (h2 : 4 * x + 6 * y = 56)  -- perimeter of figure B
  : 2 * x + 6 * y = 40 :=    -- perimeter of figure C
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1155_115544


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l1155_115542

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflection_across_x_axis :
  let P : ℝ × ℝ := (-2, 1)
  reflect_x P = (-2, -1) := by sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l1155_115542


namespace NUMINAMATH_CALUDE_walking_distance_l1155_115516

/-- Proves that given a walking speed where 1 mile is covered in 20 minutes, 
    the distance covered in 40 minutes is 2 miles. -/
theorem walking_distance (speed : ℝ) (time : ℝ) : 
  speed = 1 / 20 → time = 40 → speed * time = 2 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_l1155_115516


namespace NUMINAMATH_CALUDE_equal_probabilities_l1155_115563

/-- Represents a box containing colored balls -/
structure Box where
  red : ℕ
  green : ℕ

/-- The initial state of the boxes -/
def initial_state : Box × Box :=
  (⟨100, 0⟩, ⟨0, 100⟩)

/-- The state after transferring 8 red balls to the green box -/
def after_first_transfer (state : Box × Box) : Box × Box :=
  let (red_box, green_box) := state
  (⟨red_box.red - 8, red_box.green⟩, ⟨green_box.red + 8, green_box.green⟩)

/-- The final state after transferring 8 balls back to the red box -/
def final_state (state : Box × Box) : Box × Box :=
  let (red_box, green_box) := after_first_transfer state
  (⟨red_box.red + 8, red_box.green + 8⟩, ⟨green_box.red - 8, green_box.green - 8⟩)

/-- The probability of drawing a specific color from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  if color = "red" then
    box.red / (box.red + box.green)
  else
    box.green / (box.red + box.green)

theorem equal_probabilities :
  let (final_red_box, final_green_box) := final_state initial_state
  prob_draw final_red_box "green" = prob_draw final_green_box "red" := by
  sorry

end NUMINAMATH_CALUDE_equal_probabilities_l1155_115563


namespace NUMINAMATH_CALUDE_prob_not_same_group_three_groups_l1155_115554

/-- The probability that two students are not in the same interest group -/
def prob_not_same_group (num_groups : ℕ) : ℚ :=
  if num_groups = 0 then 0
  else (num_groups - 1 : ℚ) / num_groups

theorem prob_not_same_group_three_groups :
  prob_not_same_group 3 = 2/3 := by sorry

end NUMINAMATH_CALUDE_prob_not_same_group_three_groups_l1155_115554


namespace NUMINAMATH_CALUDE_olivers_earnings_theorem_l1155_115514

/-- Calculates the earnings of Oliver's laundry shop over three days -/
def olivers_earnings (price_per_kilo : ℝ) (day1_kilos : ℝ) (day2_increase : ℝ) : ℝ :=
  let day2_kilos := day1_kilos + day2_increase
  let day3_kilos := 2 * day2_kilos
  let total_kilos := day1_kilos + day2_kilos + day3_kilos
  price_per_kilo * total_kilos

/-- Theorem stating that Oliver's earnings for three days equal $70 -/
theorem olivers_earnings_theorem :
  olivers_earnings 2 5 5 = 70 := by
  sorry

#eval olivers_earnings 2 5 5

end NUMINAMATH_CALUDE_olivers_earnings_theorem_l1155_115514


namespace NUMINAMATH_CALUDE_intersection_A_B_quadratic_inequality_solution_l1155_115565

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 > 0}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem quadratic_inequality_solution (a b : ℝ) :
  ({x : ℝ | x^2 + a*x + b < 0} = {x | 2 < x ∧ x < 3}) ↔ (a = -5 ∧ b = 6) := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_quadratic_inequality_solution_l1155_115565


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1155_115573

theorem decimal_sum_to_fraction : 
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00001 = 24681 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1155_115573


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1155_115506

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
    (6 * x - 3) / (x^2 - 8*x + 15) = C / (x - 3) + D / (x - 5)) →
  C = -15/2 ∧ D = 27/2 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1155_115506


namespace NUMINAMATH_CALUDE_frank_breakfast_shopping_l1155_115557

/-- The cost of one bun in dollars -/
def bun_cost : ℚ := 1/10

/-- The cost of one bottle of milk in dollars -/
def milk_cost : ℚ := 2

/-- The number of bottles of milk Frank bought -/
def milk_bottles : ℕ := 2

/-- The cost of a carton of eggs in dollars -/
def egg_cost : ℚ := 3 * milk_cost

/-- The total amount Frank paid in dollars -/
def total_paid : ℚ := 11

/-- The number of buns Frank bought -/
def buns_bought : ℕ := 10

theorem frank_breakfast_shopping :
  buns_bought * bun_cost + milk_bottles * milk_cost + egg_cost = total_paid :=
by sorry

end NUMINAMATH_CALUDE_frank_breakfast_shopping_l1155_115557


namespace NUMINAMATH_CALUDE_production_cost_correct_l1155_115552

/-- The production cost per performance for Steve's circus investment -/
def production_cost_per_performance : ℝ := 7000

/-- The overhead cost for the circus production -/
def overhead_cost : ℝ := 81000

/-- The income from a single sold-out performance -/
def sold_out_income : ℝ := 16000

/-- The number of sold-out performances needed to break even -/
def break_even_performances : ℕ := 9

/-- Theorem stating that the production cost per performance is correct -/
theorem production_cost_correct :
  production_cost_per_performance * break_even_performances + overhead_cost =
  sold_out_income * break_even_performances :=
by sorry

end NUMINAMATH_CALUDE_production_cost_correct_l1155_115552


namespace NUMINAMATH_CALUDE_log_equation_solution_l1155_115510

noncomputable def LogEquation (a b x : ℝ) : Prop :=
  5 * (Real.log x / Real.log b) ^ 2 + 2 * (Real.log x / Real.log a) ^ 2 = 10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)

theorem log_equation_solution (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  LogEquation a b x → (b = a ^ (1 + Real.sqrt 15 / 5) ∨ b = a ^ (1 - Real.sqrt 15 / 5)) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1155_115510


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1155_115528

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ (1/a + 1/b = 2 ↔ a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1155_115528


namespace NUMINAMATH_CALUDE_triangles_containing_center_201_l1155_115581

/-- Given a regular 201-sided polygon inscribed in a circle with center C,
    this function computes the number of triangles formed by connecting
    any three vertices of the polygon such that C lies inside the triangle. -/
def triangles_containing_center (n : ℕ) : ℕ :=
  if n = 201 then
    let vertex_count := n
    let half_vertex_count := (vertex_count - 1) / 2
    let triangles_per_vertex := half_vertex_count * (half_vertex_count + 1) / 2
    vertex_count * triangles_per_vertex / 3
  else
    0

/-- Theorem stating that the number of triangles containing the center
    for a regular 201-sided polygon is 338350. -/
theorem triangles_containing_center_201 :
  triangles_containing_center 201 = 338350 := by
  sorry

end NUMINAMATH_CALUDE_triangles_containing_center_201_l1155_115581


namespace NUMINAMATH_CALUDE_charity_dinner_cost_l1155_115548

/-- The total cost of dinners given the number of plates and the cost of rice and chicken per plate -/
def total_cost (num_plates : ℕ) (rice_cost chicken_cost : ℚ) : ℚ :=
  num_plates * (rice_cost + chicken_cost)

/-- Theorem stating that the total cost for 100 plates with rice costing $0.10 and chicken costing $0.40 per plate is $50.00 -/
theorem charity_dinner_cost :
  total_cost 100 (10 / 100) (40 / 100) = 50 := by
  sorry

end NUMINAMATH_CALUDE_charity_dinner_cost_l1155_115548


namespace NUMINAMATH_CALUDE_special_rectangle_dimensions_l1155_115566

/-- A rectangle with the given properties -/
structure SpecialRectangle where
  width : ℝ
  length : ℝ
  perimeter_area_relation : 2 * (width + length) = 3 * (width * length)
  length_width_relation : length = 2 * width

/-- The dimensions of the special rectangle are 1 inch width and 2 inches length -/
theorem special_rectangle_dimensions (rect : SpecialRectangle) : rect.width = 1 ∧ rect.length = 2 := by
  sorry

#check special_rectangle_dimensions

end NUMINAMATH_CALUDE_special_rectangle_dimensions_l1155_115566


namespace NUMINAMATH_CALUDE_racket_price_l1155_115564

theorem racket_price (total_spent : ℚ) (h1 : total_spent = 90) : ∃ (original_price : ℚ),
  original_price + original_price / 2 = total_spent ∧ original_price = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_racket_price_l1155_115564


namespace NUMINAMATH_CALUDE_not_all_datasets_have_regression_equation_l1155_115568

-- Define a type for datasets
def Dataset : Type := Set (ℝ × ℝ)

-- Define a predicate for whether a dataset has a regression equation
def has_regression_equation (d : Dataset) : Prop := sorry

-- Theorem stating that not every dataset has a regression equation
theorem not_all_datasets_have_regression_equation : 
  ¬ (∀ d : Dataset, has_regression_equation d) := by sorry

end NUMINAMATH_CALUDE_not_all_datasets_have_regression_equation_l1155_115568


namespace NUMINAMATH_CALUDE_no_triangle_solution_l1155_115580

theorem no_triangle_solution (A B C : Real) (a b c : Real) : 
  A = Real.pi / 3 →  -- 60 degrees in radians
  b = 4 → 
  a = 2 → 
  ¬ (∃ (B C : Real), 
      0 < B ∧ 0 < C ∧ 
      A + B + C = Real.pi ∧ 
      a / Real.sin A = b / Real.sin B ∧ 
      b / Real.sin B = c / Real.sin C) :=
by
  sorry


end NUMINAMATH_CALUDE_no_triangle_solution_l1155_115580


namespace NUMINAMATH_CALUDE_jade_transactions_l1155_115529

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + (mabel * 10 / 100) →
  cal = anthony * 2 / 3 →
  jade = cal + 15 →
  jade = 81 := by
sorry

end NUMINAMATH_CALUDE_jade_transactions_l1155_115529


namespace NUMINAMATH_CALUDE_intersection_A_B_min_value_fraction_l1155_115585

-- Define the parameters b and c based on the given inequality
def b : ℝ := 3
def c : ℝ := 6

-- Define the solution set of the original inequality
def original_solution_set : Set ℝ := {x | 2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -2 ≤ x ∧ x < 2}

-- Define the solution set of bx^2 - (c+1)x - c > 0
def A : Set ℝ := {x | b * x^2 - (c + 1) * x - c > 0}

-- Theorem 1: Intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -2 ≤ x ∧ x < -2/3} := by sorry

-- Theorem 2: Minimum value of the fraction
theorem min_value_fraction :
  ∀ x > 1, (x^2 - b*x + c) / (x - 1) ≥ 3 ∧
  ∃ x > 1, (x^2 - b*x + c) / (x - 1) = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_min_value_fraction_l1155_115585


namespace NUMINAMATH_CALUDE_max_regions_theorem_l1155_115515

/-- Represents a convex polygon with a given number of sides -/
structure ConvexPolygon where
  sides : ℕ

/-- Represents two convex polygons on a plane -/
structure TwoPolygonsOnPlane where
  polygon1 : ConvexPolygon
  polygon2 : ConvexPolygon
  sides_condition : polygon1.sides > polygon2.sides

/-- The maximum number of regions into which two convex polygons can divide a plane -/
def max_regions (polygons : TwoPolygonsOnPlane) : ℕ :=
  2 * polygons.polygon2.sides + 2

/-- Theorem stating the maximum number of regions formed by two convex polygons on a plane -/
theorem max_regions_theorem (polygons : TwoPolygonsOnPlane) :
  max_regions polygons = 2 * polygons.polygon2.sides + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_theorem_l1155_115515


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1155_115518

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x^2 - 2| < 2} = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1155_115518


namespace NUMINAMATH_CALUDE_equation_implies_difference_l1155_115547

theorem equation_implies_difference (m n : ℝ) :
  (∀ x : ℝ, (x - m) * (x + 2) = x^2 + n*x - 8) →
  m - n = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_implies_difference_l1155_115547


namespace NUMINAMATH_CALUDE_triangle_vector_parallel_l1155_115594

/-- Given a triangle ABC with sides a, b, c, if the vector (sin B - sin A, √3a + c) 
    is parallel to the vector (sin C, a + b), then angle B = 5π/6 -/
theorem triangle_vector_parallel (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C)
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ 
    k * (Real.sin B - Real.sin A) = Real.sin C ∧
    k * (Real.sqrt 3 * a + c) = a + b) :
  B = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_parallel_l1155_115594


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l1155_115579

/-- Given a geometric sequence {a_n} with a_1 = 1, prove that a_2 = 4 is sufficient but not necessary for a_3 = 16 -/
theorem geometric_sequence_condition (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 1 →                            -- First term is 1
  (a 2 = 4 → a 3 = 16) ∧               -- Sufficient condition
  ¬(a 3 = 16 → a 2 = 4)                -- Not necessary condition
  := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l1155_115579


namespace NUMINAMATH_CALUDE_box_length_calculation_l1155_115538

/-- The length of a cubic box given total volume, cost per box, and total cost -/
theorem box_length_calculation (total_volume : ℝ) (cost_per_box : ℝ) (total_cost : ℝ) :
  total_volume = 1080000 ∧ cost_per_box = 0.8 ∧ total_cost = 480 →
  ∃ (length : ℝ), abs (length - (total_volume / (total_cost / cost_per_box))^(1/3)) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_box_length_calculation_l1155_115538


namespace NUMINAMATH_CALUDE_supplier_A_better_performance_l1155_115535

def supplier_A : List ℕ := [10, 9, 10, 10, 11, 11, 9, 11, 10, 10]
def supplier_B : List ℕ := [8, 10, 14, 7, 10, 11, 10, 8, 15, 12]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (fun x => ((x : ℚ) - m) ^ 2)).sum / l.length

theorem supplier_A_better_performance (A : List ℕ) (B : List ℕ)
  (hA : A = supplier_A) (hB : B = supplier_B) :
  mean A < mean B ∧ variance A < variance B := by
  sorry

end NUMINAMATH_CALUDE_supplier_A_better_performance_l1155_115535
