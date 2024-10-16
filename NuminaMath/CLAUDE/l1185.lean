import Mathlib

namespace NUMINAMATH_CALUDE_prob_different_ranks_l1185_118571

def total_cards : ℕ := 5
def ace_count : ℕ := 1
def king_count : ℕ := 2
def queen_count : ℕ := 2

def different_ranks_probability : ℚ := 4/5

theorem prob_different_ranks :
  let total_combinations := total_cards.choose 2
  let same_rank_combinations := king_count.choose 2 + queen_count.choose 2
  (total_combinations - same_rank_combinations : ℚ) / total_combinations = different_ranks_probability :=
sorry

end NUMINAMATH_CALUDE_prob_different_ranks_l1185_118571


namespace NUMINAMATH_CALUDE_triangular_pyramid_least_faces_triangular_pyramid_faces_l1185_118585

-- Define the shapes
inductive Shape
  | TriangularPrism
  | QuadrangularPrism
  | TriangularPyramid
  | QuadrangularPyramid
  | TruncatedQuadrangularPyramid

-- Function to count the number of faces for each shape
def numFaces (s : Shape) : ℕ :=
  match s with
  | Shape.TriangularPrism => 5
  | Shape.QuadrangularPrism => 6
  | Shape.TriangularPyramid => 4
  | Shape.QuadrangularPyramid => 5
  | Shape.TruncatedQuadrangularPyramid => 6  -- Assuming the truncated pyramid has a top face

-- Theorem stating that the triangular pyramid has the least number of faces
theorem triangular_pyramid_least_faces :
  ∀ s : Shape, numFaces Shape.TriangularPyramid ≤ numFaces s :=
by
  sorry

-- Theorem stating that the number of faces of a triangular pyramid is 4
theorem triangular_pyramid_faces :
  numFaces Shape.TriangularPyramid = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_least_faces_triangular_pyramid_faces_l1185_118585


namespace NUMINAMATH_CALUDE_min_clerks_needed_is_84_l1185_118539

/-- The number of forms a clerk can process per hour -/
def forms_per_hour : ℕ := 25

/-- The time in minutes to process a type A form -/
def time_per_type_a : ℕ := 3

/-- The time in minutes to process a type B form -/
def time_per_type_b : ℕ := 4

/-- The number of type A forms to process -/
def num_type_a : ℕ := 3000

/-- The number of type B forms to process -/
def num_type_b : ℕ := 4000

/-- The number of hours in a workday -/
def hours_per_day : ℕ := 5

/-- The function to calculate the minimum number of clerks needed -/
def min_clerks_needed : ℕ :=
  let total_minutes := num_type_a * time_per_type_a + num_type_b * time_per_type_b
  let total_hours := (total_minutes + 59) / 60  -- Ceiling division
  (total_hours + hours_per_day - 1) / hours_per_day  -- Ceiling division

theorem min_clerks_needed_is_84 : min_clerks_needed = 84 := by
  sorry

end NUMINAMATH_CALUDE_min_clerks_needed_is_84_l1185_118539


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1185_118549

theorem complex_modulus_problem (z : ℂ) (i : ℂ) : 
  i * i = -1 → z = (3 - i) / (1 + i) → Complex.abs (z + i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1185_118549


namespace NUMINAMATH_CALUDE_stone_piles_impossible_l1185_118529

/-- Represents a configuration of stone piles -/
def StonePiles := List Nat

/-- The initial configuration of stone piles -/
def initial_piles : StonePiles := [51, 49, 5]

/-- Merges two piles in the configuration -/
def merge_piles (piles : StonePiles) (i j : Nat) : StonePiles :=
  sorry

/-- Splits an even-numbered pile into two equal piles -/
def split_pile (piles : StonePiles) (i : Nat) : StonePiles :=
  sorry

/-- Checks if a configuration consists of 105 piles of 1 stone each -/
def is_final_state (piles : StonePiles) : Prop :=
  piles.length = 105 ∧ piles.all (· = 1)

/-- Represents a sequence of operations on the stone piles -/
inductive Operation
  | Merge (i j : Nat)
  | Split (i : Nat)

/-- Applies a sequence of operations to the initial configuration -/
def apply_operations (ops : List Operation) : StonePiles :=
  sorry

theorem stone_piles_impossible :
  ∀ (ops : List Operation), ¬(is_final_state (apply_operations ops)) :=
sorry

end NUMINAMATH_CALUDE_stone_piles_impossible_l1185_118529


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1185_118572

theorem solution_set_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  {x : ℝ | -b < 1/x ∧ 1/x < a} = {x : ℝ | x < -1/b ∨ x > 1/a} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1185_118572


namespace NUMINAMATH_CALUDE_roots_differ_by_one_l1185_118570

theorem roots_differ_by_one (a : ℝ) : 
  (∃ x y : ℝ, x^2 - a*x + 1 = 0 ∧ y^2 - a*y + 1 = 0 ∧ y - x = 1) → a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_differ_by_one_l1185_118570


namespace NUMINAMATH_CALUDE_log_inequality_l1185_118597

theorem log_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |Real.log a| > |Real.log b|) : a * b < 1 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1185_118597


namespace NUMINAMATH_CALUDE_complement_of_union_l1185_118559

def U : Finset Nat := {1,2,3,4,5,6}
def M : Finset Nat := {1,3,6}
def N : Finset Nat := {2,3,4}

theorem complement_of_union : (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1185_118559


namespace NUMINAMATH_CALUDE_f_increasing_when_a_zero_f_odd_when_a_one_f_domain_iff_a_lt_two_l1185_118599

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4^x - 1) / (4^x + 1 - a * 2^x)

-- Theorem 1: When a = 0, f is increasing
theorem f_increasing_when_a_zero : 
  ∀ x y : ℝ, x < y → f 0 x < f 0 y :=
sorry

-- Theorem 2: When a = 1, f is odd
theorem f_odd_when_a_one :
  ∀ x : ℝ, f 1 (-x) = -(f 1 x) :=
sorry

-- Theorem 3: Domain of f is R iff a < 2
theorem f_domain_iff_a_lt_two :
  ∀ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a < 2 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_when_a_zero_f_odd_when_a_one_f_domain_iff_a_lt_two_l1185_118599


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l1185_118504

theorem initial_number_of_persons (avg_weight_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_weight_increase = 1.5 →
  old_weight = 65 →
  new_weight = 78.5 →
  (new_weight - old_weight) / avg_weight_increase = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l1185_118504


namespace NUMINAMATH_CALUDE_counterexample_exists_l1185_118558

theorem counterexample_exists : ∃ a : ℝ, a^2 > 0 ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1185_118558


namespace NUMINAMATH_CALUDE_inequality_problem_l1185_118551

theorem inequality_problem (a b c m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : m > 0)
  (h4 : ∀ a b c, a > b → b > c → (1 / (a - b) + m / (b - c) ≥ 9 / (a - c))) :
  m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l1185_118551


namespace NUMINAMATH_CALUDE_square_sum_and_product_l1185_118598

theorem square_sum_and_product (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 3) : 
  a^2 + b^2 = 5 ∧ a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l1185_118598


namespace NUMINAMATH_CALUDE_zero_overtime_accidents_l1185_118506

/-- Represents the linear relationship between overtime hours and accidents -/
structure AccidentModel where
  slope : ℝ
  intercept : ℝ

/-- Calculates the expected number of accidents for a given number of overtime hours -/
def expected_accidents (model : AccidentModel) (hours : ℝ) : ℝ :=
  model.slope * hours + model.intercept

/-- Theorem stating the expected number of accidents when no overtime is logged -/
theorem zero_overtime_accidents 
  (model : AccidentModel)
  (h1 : expected_accidents model 1000 = 8)
  (h2 : expected_accidents model 400 = 5) :
  expected_accidents model 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_zero_overtime_accidents_l1185_118506


namespace NUMINAMATH_CALUDE_first_worker_time_l1185_118540

/-- Given three workers who make parts with the following conditions:
    1. They need to make 80 identical parts in total.
    2. Together, they produce 20 parts per hour.
    3. The first worker makes 20 parts, taking more than 3 hours.
    4. The remaining work is completed by the second and third workers together.
    5. The total time taken to complete the work is 8 hours.
    
    This theorem proves that it would take the first worker 16 hours to make all 80 parts by himself. -/
theorem first_worker_time (x y z : ℝ) (h1 : x + y + z = 20) 
  (h2 : 20 / x > 3) (h3 : 20 / x + 60 / (y + z) = 8) : 80 / x = 16 := by
  sorry

end NUMINAMATH_CALUDE_first_worker_time_l1185_118540


namespace NUMINAMATH_CALUDE_beka_miles_l1185_118511

/-- The number of miles Jackson flew -/
def jackson_miles : ℕ := 563

/-- The difference in miles between Beka's and Jackson's flights -/
def difference_miles : ℕ := 310

/-- Theorem: Beka flew 873 miles -/
theorem beka_miles : jackson_miles + difference_miles = 873 := by
  sorry

end NUMINAMATH_CALUDE_beka_miles_l1185_118511


namespace NUMINAMATH_CALUDE_walter_age_theorem_l1185_118517

/-- Walter's age at the end of 1998 -/
def walter_age_1998 : ℕ := 34

/-- Walter's grandmother's age at the end of 1998 -/
def grandmother_age_1998 : ℕ := 3 * walter_age_1998

/-- The sum of Walter's and his grandmother's birth years -/
def birth_years_sum : ℕ := 3860

/-- Walter's age at the end of 2003 -/
def walter_age_2003 : ℕ := walter_age_1998 + 5

theorem walter_age_theorem :
  (1998 - walter_age_1998) + (1998 - grandmother_age_1998) = birth_years_sum ∧
  walter_age_2003 = 39 := by
  sorry

end NUMINAMATH_CALUDE_walter_age_theorem_l1185_118517


namespace NUMINAMATH_CALUDE_mason_car_nuts_l1185_118538

/-- The number of nuts in Mason's car after squirrels stockpile for a given number of days -/
def nutsInCar (busySquirrels sleepySquirrels : ℕ) (busyNutsPerDay sleepyNutsPerDay days : ℕ) : ℕ :=
  (busySquirrels * busyNutsPerDay + sleepySquirrels * sleepyNutsPerDay) * days

/-- Theorem stating the number of nuts in Mason's car given the problem conditions -/
theorem mason_car_nuts :
  nutsInCar 2 1 30 20 40 = 3200 := by
  sorry

#eval nutsInCar 2 1 30 20 40

end NUMINAMATH_CALUDE_mason_car_nuts_l1185_118538


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l1185_118535

theorem chess_game_draw_probability
  (p_a_not_lose : ℝ)
  (p_b_not_lose : ℝ)
  (h_a : p_a_not_lose = 0.8)
  (h_b : p_b_not_lose = 0.7)
  (h_game : ∀ (p_a_win p_draw : ℝ),
    p_a_win + p_draw = p_a_not_lose ∧
    (1 - p_a_win) = p_b_not_lose) :
  ∃ (p_draw : ℝ), p_draw = 0.5 := by
sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l1185_118535


namespace NUMINAMATH_CALUDE_smallest_k_for_monochromatic_rectangle_l1185_118503

/-- A chessboard coloring is a function that assigns a color to each square of the board. -/
def Coloring (n k : ℕ) := Fin (2 * n) → Fin k → Fin n

/-- Predicate that checks if there exist 2 columns and 2 rows with 4 squares of the same color at their intersections. -/
def HasMonochromaticRectangle (n k : ℕ) (c : Coloring n k) : Prop :=
  ∃ (i j : Fin (2 * n)) (x y : Fin k),
    i ≠ j ∧ x ≠ y ∧ 
    c i x = c i y ∧ c j x = c j y ∧ c i x = c j x

/-- The main theorem stating the smallest k that guarantees a monochromatic rectangle for any n-coloring. -/
theorem smallest_k_for_monochromatic_rectangle (n : ℕ+) :
  ∃ (k : ℕ), k = 2 * n^2 - n + 1 ∧
  (∀ (m : ℕ), m ≥ k → ∀ (c : Coloring n m), HasMonochromaticRectangle n m c) ∧
  (∀ (m : ℕ), m < k → ∃ (c : Coloring n m), ¬HasMonochromaticRectangle n m c) :=
sorry

#check smallest_k_for_monochromatic_rectangle

end NUMINAMATH_CALUDE_smallest_k_for_monochromatic_rectangle_l1185_118503


namespace NUMINAMATH_CALUDE_sequence_eventually_periodic_l1185_118589

def sequence_next (a : ℕ) (un : ℕ) : ℕ :=
  if un % 2 = 0 then un / 2 else a + un

def is_periodic (s : ℕ → ℕ) (k p : ℕ) : Prop :=
  ∀ n, n ≥ k → s (n + p) = s n

theorem sequence_eventually_periodic (a : ℕ) (h_a : Odd a) (u : ℕ → ℕ) 
  (h_u : ∀ n, u (n + 1) = sequence_next a (u n)) :
  ∃ k p, p > 0 ∧ is_periodic u k p :=
sorry

end NUMINAMATH_CALUDE_sequence_eventually_periodic_l1185_118589


namespace NUMINAMATH_CALUDE_turtles_jumped_off_l1185_118534

/-- The fraction of turtles that jumped off the log --/
def fraction_jumped_off (initial : ℕ) (remaining : ℕ) : ℚ :=
  let additional := 3 * initial - 2
  let total := initial + additional
  (total - remaining) / total

/-- Theorem stating that the fraction of turtles that jumped off is 1/2 --/
theorem turtles_jumped_off :
  fraction_jumped_off 9 17 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_turtles_jumped_off_l1185_118534


namespace NUMINAMATH_CALUDE_box_matching_problem_l1185_118556

/-- Represents the problem of matching box bodies and bottoms --/
theorem box_matching_problem (total_tinplates : ℕ) 
  (bodies_per_tinplate : ℕ) (bottoms_per_tinplate : ℕ) 
  (bottoms_per_body : ℕ) (bodies_tinplates : ℕ) (bottoms_tinplates : ℕ) :
  total_tinplates = 36 →
  bodies_per_tinplate = 25 →
  bottoms_per_tinplate = 40 →
  bottoms_per_body = 2 →
  bodies_tinplates = 16 →
  bottoms_tinplates = 20 →
  bodies_tinplates + bottoms_tinplates = total_tinplates ∧
  bodies_per_tinplate * bodies_tinplates * bottoms_per_body = 
    bottoms_per_tinplate * bottoms_tinplates :=
by sorry

end NUMINAMATH_CALUDE_box_matching_problem_l1185_118556


namespace NUMINAMATH_CALUDE_truck_capacity_l1185_118537

/-- The total fuel capacity of Donny's truck -/
def total_capacity : ℕ := 150

/-- The amount of fuel already in the truck -/
def initial_fuel : ℕ := 38

/-- The amount of money Donny started with -/
def initial_money : ℕ := 350

/-- The amount of change Donny received -/
def change : ℕ := 14

/-- The cost of fuel per liter -/
def cost_per_liter : ℕ := 3

/-- Theorem stating that the total capacity of Donny's truck is 150 liters -/
theorem truck_capacity : 
  total_capacity = initial_fuel + (initial_money - change) / cost_per_liter := by
  sorry

end NUMINAMATH_CALUDE_truck_capacity_l1185_118537


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l1185_118500

/-- The area of a triangle with base 10 and height 5 is 25 -/
theorem triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 10 ∧ height = 5 → area = (base * height) / 2 → area = 25

#check triangle_area

theorem triangle_area_proof : triangle_area 10 5 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l1185_118500


namespace NUMINAMATH_CALUDE_smallest_positive_integer_2016m_40404n_l1185_118592

theorem smallest_positive_integer_2016m_40404n :
  ∃ (k : ℕ), k > 0 ∧
  (∀ (j : ℕ), j > 0 → (∃ (m n : ℤ), j = 2016 * m + 40404 * n) → k ≤ j) ∧
  (∃ (m n : ℤ), k = 2016 * m + 40404 * n) →
  k = 12 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_2016m_40404n_l1185_118592


namespace NUMINAMATH_CALUDE_parabola_focus_l1185_118544

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = -4 * x^2 + 4 * x - 1

/-- The focus of a parabola -/
def is_focus (f_x f_y : ℝ) : Prop :=
  f_x = 1/2 ∧ f_y = -1/8

/-- Theorem: The focus of the parabola y = -4x^2 + 4x - 1 is (1/2, -1/8) -/
theorem parabola_focus :
  ∃ (f_x f_y : ℝ), (∀ x y, parabola_equation x y → is_focus f_x f_y) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1185_118544


namespace NUMINAMATH_CALUDE_inequality_range_m_l1185_118564

/-- The function f(x) defined in the problem -/
def f (x : ℝ) : ℝ := x^2 + |x - 1|

/-- The theorem stating the range of m for which the inequality always holds -/
theorem inequality_range_m :
  (∀ x : ℝ, f x ≥ (m + 2) * x - 1) ↔ m ∈ Set.Icc (-3 - 2 * Real.sqrt 2) 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_m_l1185_118564


namespace NUMINAMATH_CALUDE_kite_area_is_40_l1185_118577

/-- A point in 2D space represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- A kite defined by its four vertices -/
structure Kite where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculate the area of a kite given its vertices -/
def kiteArea (k : Kite) : ℝ := sorry

/-- The specific kite from the problem -/
def problemKite : Kite := {
  v1 := { x := 0, y := 6 }
  v2 := { x := 4, y := 10 }
  v3 := { x := 8, y := 6 }
  v4 := { x := 4, y := 0 }
}

theorem kite_area_is_40 : kiteArea problemKite = 40 := by sorry

end NUMINAMATH_CALUDE_kite_area_is_40_l1185_118577


namespace NUMINAMATH_CALUDE_elliot_book_pages_left_l1185_118507

def pages_left_after_week (total_pages : ℕ) (pages_read : ℕ) (pages_per_day : ℕ) (days : ℕ) : ℕ :=
  total_pages - pages_read - (pages_per_day * days)

theorem elliot_book_pages_left : pages_left_after_week 381 149 20 7 = 92 := by
  sorry

end NUMINAMATH_CALUDE_elliot_book_pages_left_l1185_118507


namespace NUMINAMATH_CALUDE_line_not_parallel_in_plane_l1185_118567

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (not_parallel : Line → Plane → Prop)
variable (coplanar : Line → Line → Plane → Prop)
variable (not_parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_not_parallel_in_plane 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : contained_in m α) 
  (h4 : not_parallel n α) 
  (h5 : coplanar m n β) : 
  not_parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_not_parallel_in_plane_l1185_118567


namespace NUMINAMATH_CALUDE_indeterminate_disjunction_l1185_118550

theorem indeterminate_disjunction (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (r : Prop), r ↔ (p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_disjunction_l1185_118550


namespace NUMINAMATH_CALUDE_cheaper_fluid_cost_l1185_118579

/-- Represents the cost of cleaning fluids and drum quantities -/
structure CleaningSupplies where
  total_drums : ℕ
  expensive_drums : ℕ
  cheap_drums : ℕ
  expensive_cost : ℚ
  total_cost : ℚ

/-- Theorem stating that given the conditions, the cheaper fluid costs $20 per drum -/
theorem cheaper_fluid_cost (supplies : CleaningSupplies)
  (h1 : supplies.total_drums = 7)
  (h2 : supplies.expensive_drums + supplies.cheap_drums = supplies.total_drums)
  (h3 : supplies.expensive_cost = 30)
  (h4 : supplies.total_cost = 160)
  (h5 : supplies.cheap_drums = 5) :
  (supplies.total_cost - supplies.expensive_cost * supplies.expensive_drums) / supplies.cheap_drums = 20 :=
by sorry

end NUMINAMATH_CALUDE_cheaper_fluid_cost_l1185_118579


namespace NUMINAMATH_CALUDE_number_equation_solution_l1185_118565

theorem number_equation_solution :
  ∃ (x : ℝ), 7 * x = 3 * x + 12 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1185_118565


namespace NUMINAMATH_CALUDE_total_molecular_weight_l1185_118594

-- Define molecular weights of elements
def mw_C : ℝ := 12.01
def mw_H : ℝ := 1.008
def mw_O : ℝ := 16.00
def mw_Na : ℝ := 22.99

-- Define composition of compounds
def acetic_acid_C : ℕ := 2
def acetic_acid_H : ℕ := 4
def acetic_acid_O : ℕ := 2

def sodium_hydroxide_Na : ℕ := 1
def sodium_hydroxide_O : ℕ := 1
def sodium_hydroxide_H : ℕ := 1

-- Define number of moles
def moles_acetic_acid : ℝ := 7
def moles_sodium_hydroxide : ℝ := 10

-- Theorem statement
theorem total_molecular_weight :
  let mw_acetic_acid := acetic_acid_C * mw_C + acetic_acid_H * mw_H + acetic_acid_O * mw_O
  let mw_sodium_hydroxide := sodium_hydroxide_Na * mw_Na + sodium_hydroxide_O * mw_O + sodium_hydroxide_H * mw_H
  let total_weight := moles_acetic_acid * mw_acetic_acid + moles_sodium_hydroxide * mw_sodium_hydroxide
  total_weight = 820.344 := by
  sorry

end NUMINAMATH_CALUDE_total_molecular_weight_l1185_118594


namespace NUMINAMATH_CALUDE_factor_grid_theorem_l1185_118530

/-- The factors of 100 -/
def factors_of_100 : Finset Nat := {1, 2, 4, 5, 10, 20, 25, 50, 100}

/-- The product of all factors of 100 -/
def product_of_factors : Nat := Finset.prod factors_of_100 id

/-- The common product for each row, column, and diagonal -/
def common_product : Nat := 1000

/-- The 3x3 grid representation -/
structure Grid :=
  (a b c d e f g h i : Nat)

/-- Predicate to check if a grid is valid -/
def is_valid_grid (grid : Grid) : Prop :=
  grid.a ∈ factors_of_100 ∧ grid.b ∈ factors_of_100 ∧ grid.c ∈ factors_of_100 ∧
  grid.d ∈ factors_of_100 ∧ grid.e ∈ factors_of_100 ∧ grid.f ∈ factors_of_100 ∧
  grid.g ∈ factors_of_100 ∧ grid.h ∈ factors_of_100 ∧ grid.i ∈ factors_of_100

/-- Predicate to check if a grid satisfies the product condition -/
def satisfies_product_condition (grid : Grid) : Prop :=
  grid.a * grid.b * grid.c = common_product ∧
  grid.d * grid.e * grid.f = common_product ∧
  grid.g * grid.h * grid.i = common_product ∧
  grid.a * grid.d * grid.g = common_product ∧
  grid.b * grid.e * grid.h = common_product ∧
  grid.c * grid.f * grid.i = common_product ∧
  grid.a * grid.e * grid.i = common_product ∧
  grid.c * grid.e * grid.g = common_product

/-- The main theorem -/
theorem factor_grid_theorem (x : Nat) :
  is_valid_grid { a := x, b := 1, c := 50, d := 2, e := 25, f := 20, g := 10, h := 4, i := 5 } ∧
  satisfies_product_condition { a := x, b := 1, c := 50, d := 2, e := 25, f := 20, g := 10, h := 4, i := 5 } →
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_factor_grid_theorem_l1185_118530


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l1185_118591

-- Define the function f(x) = 4x³ - 5x + 6
def f (x : ℝ) : ℝ := 4 * x^3 - 5 * x + 6

-- State the theorem
theorem root_exists_in_interval :
  ∃ x ∈ Set.Ioo (-2 : ℝ) (-1 : ℝ), f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l1185_118591


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1185_118533

theorem sum_of_squares_of_roots : ∃ (r₁ r₂ r₃ r₄ : ℝ),
  (∀ x : ℝ, (x^2 + 4*x)^2 - 2016*(x^2 + 4*x) + 2017 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
  r₁^2 + r₂^2 + r₃^2 + r₄^2 = 4048 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1185_118533


namespace NUMINAMATH_CALUDE_cubic_sum_equals_36_l1185_118582

theorem cubic_sum_equals_36 (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  a^3 + b^3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_36_l1185_118582


namespace NUMINAMATH_CALUDE_colored_copies_count_l1185_118514

/-- Represents the number of copies and their costs --/
structure CopyData where
  totalCopies : ℕ
  regularHoursCopies : ℕ
  coloredRegularCost : ℚ
  coloredAfterHoursCost : ℚ
  whiteCopyCost : ℚ
  totalBill : ℚ

/-- Theorem stating that given the conditions, the number of colored copies is 300 --/
theorem colored_copies_count (data : CopyData)
  (h1 : data.totalCopies = 400)
  (h2 : data.regularHoursCopies = 180)
  (h3 : data.coloredRegularCost = 10/100)
  (h4 : data.coloredAfterHoursCost = 8/100)
  (h5 : data.whiteCopyCost = 5/100)
  (h6 : data.totalBill = 45/2)
  : ∃ (coloredCopies : ℕ), coloredCopies = 300 ∧ 
    (coloredCopies : ℚ) * data.coloredRegularCost * (data.regularHoursCopies : ℚ) / data.totalCopies +
    (coloredCopies : ℚ) * data.coloredAfterHoursCost * (data.totalCopies - data.regularHoursCopies : ℚ) / data.totalCopies +
    (data.totalCopies - coloredCopies : ℚ) * data.whiteCopyCost = data.totalBill :=
sorry

end NUMINAMATH_CALUDE_colored_copies_count_l1185_118514


namespace NUMINAMATH_CALUDE_common_first_digit_is_two_l1185_118508

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def first_digit (n : ℕ) : ℕ :=
  if n < 10 then n
  else first_digit (n / 10)

def three_digit_powers_of_2 : Set ℕ :=
  {n | ∃ m : ℕ, n = 2^m ∧ is_three_digit n}

def three_digit_powers_of_3 : Set ℕ :=
  {n | ∃ m : ℕ, n = 3^m ∧ is_three_digit n}

theorem common_first_digit_is_two :
  ∃! d : ℕ, (∃ n ∈ three_digit_powers_of_2, first_digit n = d) ∧
            (∃ m ∈ three_digit_powers_of_3, first_digit m = d) ∧
            d = 2 :=
sorry

end NUMINAMATH_CALUDE_common_first_digit_is_two_l1185_118508


namespace NUMINAMATH_CALUDE_roots_product_theorem_l1185_118555

theorem roots_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 16/3 := by
sorry

end NUMINAMATH_CALUDE_roots_product_theorem_l1185_118555


namespace NUMINAMATH_CALUDE_min_distance_complex_l1185_118527

/-- Given a complex number z satisfying |z + 3i| = 1, 
    the minimum value of |z - 1 + 2i| is √2 - 1. -/
theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 3 * Complex.I) = 1) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 2 - 1 ∧
  ∀ (w : ℂ), Complex.abs (w + 3 * Complex.I) = 1 →
  Complex.abs (w - 1 + 2 * Complex.I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l1185_118527


namespace NUMINAMATH_CALUDE_root_polynomial_relation_l1185_118524

theorem root_polynomial_relation : ∃ (b c : ℤ), 
  (∀ r : ℝ, r^2 - 2*r - 1 = 0 → r^5 - b*r - c = 0) ∧ b*c = 348 := by
  sorry

end NUMINAMATH_CALUDE_root_polynomial_relation_l1185_118524


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1185_118531

/-- A sufficient but not necessary condition for a quadratic function to have no real roots -/
theorem quadratic_no_real_roots (a b c : ℝ) (ha : a ≠ 0) :
  b^2 - 4*a*c < -1 → ∀ x, a*x^2 + b*x + c ≠ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1185_118531


namespace NUMINAMATH_CALUDE_simplify_expression_l1185_118552

/-- Proves that the simplified expression is equal to the original expression for all real x. -/
theorem simplify_expression (x : ℝ) : 3*x + 9*x^2 + 16 - (5 - 3*x - 9*x^2 + x^3) = -x^3 + 18*x^2 + 6*x + 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1185_118552


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l1185_118561

-- Define the siblings
inductive Sibling : Type
| Alex : Sibling
| Beth : Sibling
| Cyril : Sibling
| Daria : Sibling
| Ed : Sibling

-- Define a function to represent the fraction of pizza eaten by each sibling
def pizza_fraction (s : Sibling) : ℚ :=
  match s with
  | Sibling.Alex => 1/6
  | Sibling.Beth => 1/4
  | Sibling.Cyril => 1/3
  | Sibling.Daria => 1/8
  | Sibling.Ed => 1 - (1/6 + 1/4 + 1/3 + 1/8)

-- Define the theorem
theorem pizza_consumption_order :
  ∃ (l : List Sibling),
    l = [Sibling.Cyril, Sibling.Beth, Sibling.Alex, Sibling.Daria, Sibling.Ed] ∧
    ∀ (i j : Nat), i < j → j < l.length →
      pizza_fraction (l.get ⟨i, by sorry⟩) ≥ pizza_fraction (l.get ⟨j, by sorry⟩) :=
by sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l1185_118561


namespace NUMINAMATH_CALUDE_triangle_angle_solution_l1185_118553

/-- Given a triangle with angles measuring 60°, (5x)°, and (3x)°, prove that x = 15 -/
theorem triangle_angle_solution (x : ℝ) : 
  (60 : ℝ) + 5*x + 3*x = 180 → x = 15 := by
  sorry

#check triangle_angle_solution

end NUMINAMATH_CALUDE_triangle_angle_solution_l1185_118553


namespace NUMINAMATH_CALUDE_joe_school_travel_time_l1185_118575

-- Define the variables
def walking_speed : ℝ := 1 -- Arbitrary unit speed
def running_speed : ℝ := 3 * walking_speed
def walking_time : ℝ := 9 -- minutes
def break_time : ℝ := 1 -- minute

-- Define the theorem
theorem joe_school_travel_time :
  let running_time := walking_time / 3
  walking_time + break_time + running_time = 13 := by
  sorry

end NUMINAMATH_CALUDE_joe_school_travel_time_l1185_118575


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l1185_118580

def is_multiple_of_29 (n : ℕ) : Prop := ∃ k : ℕ, n = 29 * k

def last_two_digits_are_29 (n : ℕ) : Prop := n % 100 = 29

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem smallest_number_with_conditions : 
  (is_multiple_of_29 51729) ∧ 
  (last_two_digits_are_29 51729) ∧ 
  (sum_of_digits 51729 = 29) ∧
  (∀ m : ℕ, m < 51729 → 
    ¬(is_multiple_of_29 m ∧ last_two_digits_are_29 m ∧ sum_of_digits m = 29)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l1185_118580


namespace NUMINAMATH_CALUDE_cubic_expression_value_l1185_118568

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 5 * p - 12 = 0 →
  3 * q^2 - 5 * q - 12 = 0 →
  p ≠ q →
  (9 * p^3 - 9 * q^3) / (p - q) = 61 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l1185_118568


namespace NUMINAMATH_CALUDE_johns_price_per_sheet_l1185_118566

/-- The price per sheet charged by John's Photo World -/
def J : ℝ := sorry

/-- The number of sheets being compared -/
def sheets : ℕ := 12

/-- The sitting fee charged by John's Photo World -/
def john_sitting_fee : ℝ := 125

/-- The price per sheet charged by Sam's Picture Emporium -/
def sam_price_per_sheet : ℝ := 1.50

/-- The sitting fee charged by Sam's Picture Emporium -/
def sam_sitting_fee : ℝ := 140

theorem johns_price_per_sheet :
  J * sheets + john_sitting_fee = sam_price_per_sheet * sheets + sam_sitting_fee ∧ J = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_johns_price_per_sheet_l1185_118566


namespace NUMINAMATH_CALUDE_constant_k_value_l1185_118573

theorem constant_k_value (k : ℝ) : 
  (∀ x : ℝ, -x^2 - (k + 9)*x - 8 = -(x - 2)*(x - 4)) → k = -15 := by
sorry

end NUMINAMATH_CALUDE_constant_k_value_l1185_118573


namespace NUMINAMATH_CALUDE_willowton_vampires_l1185_118519

/-- The number of vampires after a given number of nights -/
def vampires_after_nights (initial_population : ℕ) (initial_vampires : ℕ) (turned_per_night : ℕ) (nights : ℕ) : ℕ :=
  initial_vampires + nights * (initial_vampires * turned_per_night)

/-- Theorem stating the number of vampires after two nights in Willowton -/
theorem willowton_vampires :
  vampires_after_nights 300 2 5 2 = 72 := by
  sorry

#eval vampires_after_nights 300 2 5 2

end NUMINAMATH_CALUDE_willowton_vampires_l1185_118519


namespace NUMINAMATH_CALUDE_trig_functions_and_expression_l1185_118528

theorem trig_functions_and_expression (α : Real) (h : Real.tan α = -Real.sqrt 3) :
  (((Real.sin α = Real.sqrt 3 / 2) ∧ (Real.cos α = -1 / 2)) ∨
   ((Real.sin α = -Real.sqrt 3 / 2) ∧ (Real.cos α = 1 / 2))) ∧
  ((Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_trig_functions_and_expression_l1185_118528


namespace NUMINAMATH_CALUDE_complex_average_equals_three_halves_l1185_118562

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- The main theorem to prove -/
theorem complex_average_equals_three_halves :
  avg3 (avg3 (avg2 2 2) 3 1) (avg2 1 2) 1 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_average_equals_three_halves_l1185_118562


namespace NUMINAMATH_CALUDE_no_solution_for_sqrt_equation_l1185_118523

theorem no_solution_for_sqrt_equation :
  ¬∃ x : ℝ, Real.sqrt (3*x - 2) + Real.sqrt (2*x - 2) + Real.sqrt (x - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_sqrt_equation_l1185_118523


namespace NUMINAMATH_CALUDE_alberts_earnings_increase_l1185_118526

/-- Proves that given Albert's earnings of $495 after a 36% increase and $454.96 after an unknown percentage increase, the unknown percentage increase is 25%. -/
theorem alberts_earnings_increase (original : ℝ) (increased : ℝ) (percentage : ℝ) : 
  (original * 1.36 = 495) → 
  (original * (1 + percentage) = 454.96) → 
  percentage = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_alberts_earnings_increase_l1185_118526


namespace NUMINAMATH_CALUDE_max_linear_term_bound_l1185_118501

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem max_linear_term_bound {a b c : ℝ} :
  (∀ x : ℝ, |x| ≤ 1 → |quadratic_function a b c x| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |a * x + b| ≤ 2) ∧
  (∃ a b : ℝ, ∃ x : ℝ, |x| ≤ 1 ∧ |a * x + b| = 2) :=
sorry

end NUMINAMATH_CALUDE_max_linear_term_bound_l1185_118501


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l1185_118548

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmeticSum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem arithmetic_sequence_max_sum :
  let a₁ : ℚ := 4
  let d : ℚ := -5/7
  ∀ n : ℕ, n ≠ 0 → arithmeticSum a₁ d 6 ≥ arithmeticSum a₁ d n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l1185_118548


namespace NUMINAMATH_CALUDE_supply_duration_with_three_leaks_l1185_118583

/-- Represents a water tank with its supply duration and leak information -/
structure WaterTank where
  normalDuration : ℕ  -- Duration in days without leaks
  singleLeakDuration : ℕ  -- Duration in days with a single leak
  singleLeakRate : ℕ  -- Rate of the single leak in liters per day
  leakRates : List ℕ  -- List of leak rates for multiple leaks

/-- Calculates the duration of water supply given multiple leaks -/
def supplyDurationWithLeaks (tank : WaterTank) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the correct supply duration for the given scenario -/
theorem supply_duration_with_three_leaks 
  (tank : WaterTank) 
  (h1 : tank.normalDuration = 60)
  (h2 : tank.singleLeakDuration = 45)
  (h3 : tank.singleLeakRate = 10)
  (h4 : tank.leakRates = [10, 15, 20]) :
  supplyDurationWithLeaks tank = 24 := by
  sorry

end NUMINAMATH_CALUDE_supply_duration_with_three_leaks_l1185_118583


namespace NUMINAMATH_CALUDE_inequality_proof_l1185_118578

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b^3 / (a^2 + 8*b*c)) + (c^3 / (b^2 + 8*c*a)) + (a^3 / (c^2 + 8*a*b)) ≥ (1/9) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1185_118578


namespace NUMINAMATH_CALUDE_pass_through_walls_l1185_118542

theorem pass_through_walls (k : ℕ) (n : ℕ) : 
  k = 8 → 
  (k * Real.sqrt (k / ((k - 1) * k + (k - 1))) = Real.sqrt (k * (k / n))) ↔ 
  n = 63 := by
sorry

end NUMINAMATH_CALUDE_pass_through_walls_l1185_118542


namespace NUMINAMATH_CALUDE_average_age_of_friends_l1185_118588

theorem average_age_of_friends (age1 age2 age3 : ℕ) : 
  age1 = 40 →
  age2 = 30 →
  age3 = age1 + 10 →
  (age1 + age2 + age3) / 3 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_age_of_friends_l1185_118588


namespace NUMINAMATH_CALUDE_exactly_one_hit_probability_l1185_118569

theorem exactly_one_hit_probability (p : ℝ) (h : p = 0.6) :
  p * (1 - p) + (1 - p) * p = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_hit_probability_l1185_118569


namespace NUMINAMATH_CALUDE_can_display_rows_l1185_118513

/-- Represents a display of cans arranged in rows -/
structure CanDisplay where
  topRowCans : ℕ
  rowIncrement : ℕ
  totalCans : ℕ

/-- Calculates the number of rows in a can display -/
def numberOfRows (display : CanDisplay) : ℕ :=
  sorry

/-- Theorem: A display with 3 cans in the top row, 2 more cans in each subsequent row, 
    and 225 total cans has 15 rows -/
theorem can_display_rows (display : CanDisplay) 
  (h1 : display.topRowCans = 3)
  (h2 : display.rowIncrement = 2)
  (h3 : display.totalCans = 225) : 
  numberOfRows display = 15 := by
  sorry

end NUMINAMATH_CALUDE_can_display_rows_l1185_118513


namespace NUMINAMATH_CALUDE_probability_of_selection_l1185_118557

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 7

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 8

/-- The total number of articles of clothing -/
def total_items : ℕ := num_shirts + num_shorts + num_socks

/-- The number of items to be selected -/
def items_selected : ℕ := 5

/-- The probability of selecting two shirts, two pairs of shorts, and one pair of socks -/
theorem probability_of_selection : 
  (Nat.choose num_shirts 2 * Nat.choose num_shorts 2 * Nat.choose num_socks 1) / 
  Nat.choose total_items items_selected = 280 / 2261 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_l1185_118557


namespace NUMINAMATH_CALUDE_area_perimeter_relation_l1185_118532

/-- A stepped shape with n rows -/
structure SteppedShape (n : ℕ) where
  (n_pos : n > 0)
  (bottom_row : ℕ)
  (bottom_odd : Odd bottom_row)
  (bottom_eq : bottom_row = 2 * n - 1)
  (top_row : ℕ)
  (top_eq : top_row = 1)

/-- The area of a stepped shape -/
def area (shape : SteppedShape n) : ℕ := n ^ 2

/-- The perimeter of a stepped shape -/
def perimeter (shape : SteppedShape n) : ℕ := 6 * n - 2

/-- The main theorem relating area and perimeter of a stepped shape -/
theorem area_perimeter_relation (shape : SteppedShape n) :
  36 * (area shape) = (perimeter shape + 2) ^ 2 := by sorry

end NUMINAMATH_CALUDE_area_perimeter_relation_l1185_118532


namespace NUMINAMATH_CALUDE_least_common_remainder_least_common_remainder_achieved_least_common_remainder_is_126_l1185_118502

theorem least_common_remainder (n : ℕ) : n > 1 ∧ n % 25 = 1 ∧ n % 7 = 1 → n ≥ 126 := by
  sorry

theorem least_common_remainder_achieved : 126 % 25 = 1 ∧ 126 % 7 = 1 := by
  sorry

theorem least_common_remainder_is_126 : ∃ (n : ℕ), n = 126 ∧ n > 1 ∧ n % 25 = 1 ∧ n % 7 = 1 ∧ 
  ∀ (m : ℕ), m > 1 ∧ m % 25 = 1 ∧ m % 7 = 1 → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_least_common_remainder_least_common_remainder_achieved_least_common_remainder_is_126_l1185_118502


namespace NUMINAMATH_CALUDE_cups_per_girl_l1185_118521

/-- Given a class with students, boys, and girls, prove the number of cups each girl brought. -/
theorem cups_per_girl (total_students : ℕ) (num_boys : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ)
  (h1 : total_students = 30)
  (h2 : num_boys = 10)
  (h3 : cups_per_boy = 5)
  (h4 : total_cups = 90)
  (h5 : total_students = num_boys + 2 * num_boys) :
  (total_cups - num_boys * cups_per_boy) / (total_students - num_boys) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cups_per_girl_l1185_118521


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1185_118545

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 48 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 3 + a 10 + a 11 = 48

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : sum_condition a) : 
  a 6 + a 7 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1185_118545


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1185_118536

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2*Complex.I → z = -1 + (3/2)*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1185_118536


namespace NUMINAMATH_CALUDE_complex_magnitude_l1185_118554

theorem complex_magnitude (z : ℂ) : Complex.abs (z - (1 + 2*I)) = 0 → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1185_118554


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_proof_l1185_118586

theorem inequality_and_minimum_value_proof 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ (min_val : ℝ) (min_x : ℝ),
    (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
      a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y)) ∧ 
    (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
      a^2 / x + b^2 / y = (a + b)^2 / (x + y) ↔ x / y = a / b) ∧
    (∀ (x : ℝ) (hx : 0 < x ∧ x < 1/2), 
      2 / x + 9 / (1 - 2*x) ≥ min_val) ∧
    (0 < min_x ∧ min_x < 1/2) ∧
    (2 / min_x + 9 / (1 - 2*min_x) = min_val) ∧
    min_val = 25 ∧
    min_x = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_proof_l1185_118586


namespace NUMINAMATH_CALUDE_smallest_year_after_2010_with_digit_sum_16_l1185_118590

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Predicate to check if a year is after 2010 -/
def is_after_2010 (year : ℕ) : Prop :=
  year > 2010

/-- Theorem stating that 2059 is the smallest year after 2010 with digit sum 16 -/
theorem smallest_year_after_2010_with_digit_sum_16 :
  (∀ year : ℕ, is_after_2010 year → sum_of_digits year = 16 → year ≥ 2059) ∧
  (is_after_2010 2059 ∧ sum_of_digits 2059 = 16) :=
sorry

end NUMINAMATH_CALUDE_smallest_year_after_2010_with_digit_sum_16_l1185_118590


namespace NUMINAMATH_CALUDE_pond_depth_l1185_118546

theorem pond_depth (d : ℝ) 
  (h1 : ¬(d ≥ 10))  -- Adam's statement is false
  (h2 : ¬(d ≤ 8))   -- Ben's statement is false
  (h3 : d ≠ 7)      -- Carla's statement is false
  : 8 < d ∧ d < 10 := by
  sorry

end NUMINAMATH_CALUDE_pond_depth_l1185_118546


namespace NUMINAMATH_CALUDE_workers_savings_l1185_118593

theorem workers_savings (P : ℝ) (f : ℝ) (h1 : P > 0) (h2 : 0 ≤ f ∧ f ≤ 1) 
  (h3 : 12 * f * P = 6 * (1 - f) * P) : f = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_workers_savings_l1185_118593


namespace NUMINAMATH_CALUDE_street_trees_l1185_118563

theorem street_trees (road_length : ℝ) (tree_spacing : ℝ) (h1 : road_length = 268.8) (h2 : tree_spacing = 6.4) : 
  ⌊road_length / tree_spacing⌋ + 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_street_trees_l1185_118563


namespace NUMINAMATH_CALUDE_stratified_sampling_ratio_l1185_118509

theorem stratified_sampling_ratio (total : ℕ) (first_year : ℕ) (second_year : ℕ) (selected_first : ℕ) :
  total = first_year + second_year →
  first_year = 30 →
  second_year = 40 →
  selected_first = 6 →
  (selected_first * second_year) / first_year = 8 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_ratio_l1185_118509


namespace NUMINAMATH_CALUDE_nine_valid_sets_l1185_118560

def count_valid_sets : ℕ → Prop :=
  λ n => ∃ S : Finset (ℕ × ℕ × ℕ),
    (∀ (a b c : ℕ), (a, b, c) ∈ S ↔ 
      (Nat.gcd a b = 4 ∧ 
       Nat.lcm a c = 100 ∧ 
       Nat.lcm b c = 100 ∧ 
       a ≤ b)) ∧
    S.card = n

theorem nine_valid_sets : count_valid_sets 9 := by sorry

end NUMINAMATH_CALUDE_nine_valid_sets_l1185_118560


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l1185_118515

theorem rationalize_and_simplify :
  32 / Real.sqrt 8 + 8 / Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l1185_118515


namespace NUMINAMATH_CALUDE_adam_book_purchase_l1185_118512

/-- The number of books Adam bought on his shopping trip -/
def books_bought : ℕ := sorry

/-- The number of books Adam had before shopping -/
def initial_books : ℕ := 56

/-- The number of shelves in Adam's bookcase -/
def num_shelves : ℕ := 4

/-- The average number of books per shelf in Adam's bookcase -/
def avg_books_per_shelf : ℕ := 20

/-- The number of books left over after filling the bookcase -/
def leftover_books : ℕ := 2

/-- The theorem stating how many books Adam bought -/
theorem adam_book_purchase :
  books_bought = 
    num_shelves * avg_books_per_shelf + leftover_books - initial_books :=
by sorry

end NUMINAMATH_CALUDE_adam_book_purchase_l1185_118512


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l1185_118587

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricPointXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

theorem symmetric_point_x_axis :
  let P : Point3D := { x := 1, y := 3, z := 6 }
  symmetricPointXAxis P = { x := 1, y := -3, z := -6 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l1185_118587


namespace NUMINAMATH_CALUDE_f_negative_nine_halves_l1185_118595

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def periodic_2 (f : ℝ → ℝ) := ∀ x, f (x + 2) = f x

def f_on_unit_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

theorem f_negative_nine_halves 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : periodic_2 f) 
  (h_unit_interval : f_on_unit_interval f) : 
  f (-9/2) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_f_negative_nine_halves_l1185_118595


namespace NUMINAMATH_CALUDE_blue_hat_cost_l1185_118518

/-- Proves that the cost of each blue hat is $6 given the conditions of the hat purchase problem -/
theorem blue_hat_cost (total_hats : ℕ) (green_hats : ℕ) (green_cost : ℕ) (total_cost : ℕ) : 
  total_hats = 85 →
  green_hats = 40 →
  green_cost = 7 →
  total_cost = 550 →
  (total_cost - green_hats * green_cost) / (total_hats - green_hats) = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_hat_cost_l1185_118518


namespace NUMINAMATH_CALUDE_paper_length_wrapped_around_cylinder_l1185_118581

/-- Calculates the length of paper wrapped around a cylindrical tube. -/
theorem paper_length_wrapped_around_cylinder
  (initial_diameter : ℝ)
  (paper_width : ℝ)
  (num_wraps : ℕ)
  (final_diameter : ℝ)
  (h1 : initial_diameter = 3)
  (h2 : paper_width = 4)
  (h3 : num_wraps = 400)
  (h4 : final_diameter = 11) :
  ∃ (paper_length : ℝ), paper_length = 28 * π ∧ paper_length * 100 = π * (num_wraps * (initial_diameter + final_diameter)) :=
by sorry

end NUMINAMATH_CALUDE_paper_length_wrapped_around_cylinder_l1185_118581


namespace NUMINAMATH_CALUDE_parabola_vertex_not_in_second_quadrant_l1185_118584

/-- The vertex of the parabola y = 4x^2 - 4(a+1)x + a cannot lie in the second quadrant for any real value of a. -/
theorem parabola_vertex_not_in_second_quadrant (a : ℝ) : 
  let f (x : ℝ) := 4 * x^2 - 4 * (a + 1) * x + a
  let vertex_x := (a + 1) / 2
  let vertex_y := f vertex_x
  ¬(vertex_x < 0 ∧ vertex_y > 0) := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_not_in_second_quadrant_l1185_118584


namespace NUMINAMATH_CALUDE_hexagon_division_l1185_118547

/-- A regular hexagon with all sides and diagonals drawn -/
structure RegularHexagonWithDiagonals where
  /-- The number of vertices in a regular hexagon -/
  num_vertices : Nat
  /-- The number of sides in a regular hexagon -/
  num_sides : Nat
  /-- The number of diagonals in a regular hexagon -/
  num_diagonals : Nat
  /-- Assertion that the number of vertices is 6 -/
  vertex_count : num_vertices = 6
  /-- Assertion that the number of sides is equal to the number of vertices -/
  side_count : num_sides = num_vertices
  /-- Formula for the number of diagonals in a hexagon -/
  diagonal_count : num_diagonals = (num_vertices * (num_vertices - 3)) / 2

/-- The number of regions into which a regular hexagon is divided when all its sides and diagonals are drawn -/
def num_regions (h : RegularHexagonWithDiagonals) : Nat := 24

/-- Theorem stating that drawing all sides and diagonals of a regular hexagon divides it into 24 regions -/
theorem hexagon_division (h : RegularHexagonWithDiagonals) : num_regions h = 24 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_division_l1185_118547


namespace NUMINAMATH_CALUDE_mean_of_three_numbers_l1185_118541

theorem mean_of_three_numbers (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 92 →
  d = 120 →
  b = 60 →
  (a + b + c) / 3 = 82 + 2/3 :=
by sorry

end NUMINAMATH_CALUDE_mean_of_three_numbers_l1185_118541


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1185_118576

-- Problem 1
theorem equation_one_solution (x : ℝ) : 
  x / (2 * x - 5) + 5 / (5 - 2 * x) = 1 ↔ x = 0 :=
sorry

-- Problem 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℝ), (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1185_118576


namespace NUMINAMATH_CALUDE_edwards_initial_money_l1185_118520

theorem edwards_initial_money (spent_first spent_second remaining : ℕ) : 
  spent_first = 9 → 
  spent_second = 8 → 
  remaining = 17 → 
  spent_first + spent_second + remaining = 34 :=
by sorry

end NUMINAMATH_CALUDE_edwards_initial_money_l1185_118520


namespace NUMINAMATH_CALUDE_unique_prime_sum_10003_l1185_118522

/-- A function that returns the number of ways to write a given natural number as the sum of two primes -/
def countPrimeSumWays (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there is exactly one way to write 10003 as the sum of two primes -/
theorem unique_prime_sum_10003 : countPrimeSumWays 10003 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_10003_l1185_118522


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_l1185_118510

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_l1185_118510


namespace NUMINAMATH_CALUDE_max_sum_square_roots_l1185_118525

/-- Given a positive real number k, the function f(x) = x + √(k - x^2) 
    reaches its maximum value of √(2k) when x = √(k/2) -/
theorem max_sum_square_roots (k : ℝ) (h : k > 0) :
  ∃ (x : ℝ), x ≥ 0 ∧ x^2 ≤ k ∧
    (∀ (y : ℝ), y ≥ 0 → y^2 ≤ k → x + Real.sqrt (k - x^2) ≥ y + Real.sqrt (k - y^2)) ∧
    x + Real.sqrt (k - x^2) = Real.sqrt (2 * k) ∧
    x = Real.sqrt (k / 2) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_square_roots_l1185_118525


namespace NUMINAMATH_CALUDE_leo_weight_l1185_118505

/-- Given the weights of Leo (L), Kendra (K), and Ethan (E) satisfying the following conditions:
    1. L + K + E = 210
    2. L + 10 = 1.5K
    3. L + 10 = 0.75E
    We prove that Leo's weight (L) is approximately 63.33 pounds. -/
theorem leo_weight (L K E : ℝ) 
    (h1 : L + K + E = 210)
    (h2 : L + 10 = 1.5 * K)
    (h3 : L + 10 = 0.75 * E) : 
    ∃ ε > 0, |L - 63.33| < ε := by
  sorry

end NUMINAMATH_CALUDE_leo_weight_l1185_118505


namespace NUMINAMATH_CALUDE_correct_statements_l1185_118574

-- Define the statements
def statement1 : Prop := False
def statement2 : Prop := False
def statement3 : Prop := True
def statement4 : Prop := True

-- Define the regression line
def regression_line (x : ℝ) : ℝ := 0.1 * x + 10

-- Theorem to prove
theorem correct_statements :
  (statement3 ∧ statement4) ∧ 
  (¬statement1 ∧ ¬statement2) ∧
  (∀ x y : ℝ, y = regression_line x → regression_line (x + 1) = y + 0.1) :=
sorry

end NUMINAMATH_CALUDE_correct_statements_l1185_118574


namespace NUMINAMATH_CALUDE_rectangle_to_square_l1185_118516

theorem rectangle_to_square (k : ℕ) (h1 : k > 5) : 
  (∃ (n : ℕ), k * (k - 5) = n^2) → k * (k - 5) = 6^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l1185_118516


namespace NUMINAMATH_CALUDE_parabola_through_AC_not_ABC_l1185_118596

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a parabola of the form y = ax^2 + bx + 1 -/
def Parabola (a b : ℝ) (p : Point) : Prop :=
  p.y = a * p.x^2 + b * p.x + 1

/-- The given points A, B, and C -/
def A : Point := ⟨1, 2⟩
def B : Point := ⟨2, 3⟩
def C : Point := ⟨2, 1⟩

theorem parabola_through_AC_not_ABC :
  (∃ a b : ℝ, Parabola a b A ∧ Parabola a b C) ∧
  (¬ ∃ a b : ℝ, Parabola a b A ∧ Parabola a b B ∧ Parabola a b C) := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_AC_not_ABC_l1185_118596


namespace NUMINAMATH_CALUDE_personalized_pencil_cost_l1185_118543

/-- The cost of personalized pencils with a discount for large orders -/
theorem personalized_pencil_cost 
  (base_cost : ℝ)  -- Cost for 100 pencils
  (base_quantity : ℕ)  -- Base quantity (100 pencils)
  (discount_threshold : ℕ)  -- Threshold for discount (1000 pencils)
  (discount_rate : ℝ)  -- Discount rate (5%)
  (order_quantity : ℕ)  -- Quantity ordered (2500 pencils)
  (h1 : base_cost = 30)
  (h2 : base_quantity = 100)
  (h3 : discount_threshold = 1000)
  (h4 : discount_rate = 0.05)
  (h5 : order_quantity = 2500) :
  let cost_per_pencil := base_cost / base_quantity
  let full_cost := cost_per_pencil * order_quantity
  let discounted_cost := full_cost * (1 - discount_rate)
  (if order_quantity > discount_threshold then discounted_cost else full_cost) = 712.5 := by
  sorry


end NUMINAMATH_CALUDE_personalized_pencil_cost_l1185_118543
