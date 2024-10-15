import Mathlib

namespace NUMINAMATH_CALUDE_unique_m_solution_l4095_409538

theorem unique_m_solution : 
  ∀ m : ℕ+, 
  (∃ a b c : ℕ+, (a.val * b.val * c.val * m.val : ℕ) = 1 + a.val^2 + b.val^2 + c.val^2) ↔ 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_m_solution_l4095_409538


namespace NUMINAMATH_CALUDE_distribution_ways_eq_1080_l4095_409539

/-- The number of ways to distribute 6 distinct items among 4 groups,
    where two groups receive 2 items each and two groups receive 1 item each -/
def distribution_ways : ℕ :=
  (Nat.choose 6 2 * Nat.choose 4 2) / 2 * 24

/-- Theorem stating that the number of distribution ways is 1080 -/
theorem distribution_ways_eq_1080 : distribution_ways = 1080 := by
  sorry

end NUMINAMATH_CALUDE_distribution_ways_eq_1080_l4095_409539


namespace NUMINAMATH_CALUDE_library_book_distribution_l4095_409512

/-- Represents the number of books in a library -/
def total_books : ℕ := 6

/-- Calculates the number of ways to distribute books between library and checked-out status -/
def distribution_ways (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

/-- Theorem stating that the number of ways to distribute the books is 5 -/
theorem library_book_distribution :
  distribution_ways total_books = 5 := by sorry

end NUMINAMATH_CALUDE_library_book_distribution_l4095_409512


namespace NUMINAMATH_CALUDE_inverse_composition_f_inv_of_f_inv_of_f_inv_4_l4095_409516

def f : ℕ → ℕ
| 1 => 4
| 2 => 6
| 3 => 2
| 4 => 5
| 5 => 3
| 6 => 1
| _ => 0  -- Default case for completeness

-- Assumption that f is invertible
axiom f_invertible : Function.Injective f

-- Define f_inv as the inverse of f
noncomputable def f_inv : ℕ → ℕ := Function.invFun f

theorem inverse_composition (n : ℕ) : f_inv (f n) = n :=
  sorry

theorem f_inv_of_f_inv_of_f_inv_4 : f_inv (f_inv (f_inv 4)) = 2 :=
  sorry

end NUMINAMATH_CALUDE_inverse_composition_f_inv_of_f_inv_of_f_inv_4_l4095_409516


namespace NUMINAMATH_CALUDE_weed_spread_incomplete_weeds_cannot_fill_grid_l4095_409593

/-- Represents a grid with weeds -/
structure WeedGrid :=
  (size : Nat)
  (initial_weeds : Nat)

/-- Calculates the maximum possible boundary length of a grid -/
def max_boundary (g : WeedGrid) : Nat :=
  4 * g.size

/-- Calculates the maximum initial boundary length of weed-filled cells -/
def initial_boundary (g : WeedGrid) : Nat :=
  4 * g.initial_weeds

/-- The weed spread theorem -/
theorem weed_spread_incomplete (g : WeedGrid) 
  (h_size : g.size = 10) 
  (h_initial : g.initial_weeds = 9) :
  initial_boundary g < max_boundary g := by
  sorry

/-- The main theorem: weeds cannot spread to all cells -/
theorem weeds_cannot_fill_grid (g : WeedGrid) 
  (h_size : g.size = 10) 
  (h_initial : g.initial_weeds = 9) :
  ¬ (∃ (final_weeds : Nat), final_weeds = g.size * g.size) := by
  sorry

end NUMINAMATH_CALUDE_weed_spread_incomplete_weeds_cannot_fill_grid_l4095_409593


namespace NUMINAMATH_CALUDE_john_marathon_remainder_l4095_409528

/-- The length of a marathon in miles -/
def marathon_miles : ℕ := 26

/-- The additional length of a marathon in yards -/
def marathon_extra_yards : ℕ := 385

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The number of marathons John has run -/
def john_marathons : ℕ := 15

/-- Theorem stating that the remainder of yards after converting the total distance of John's marathons to miles is 495 -/
theorem john_marathon_remainder :
  (john_marathons * (marathon_miles * yards_per_mile + marathon_extra_yards)) % yards_per_mile = 495 := by
  sorry

end NUMINAMATH_CALUDE_john_marathon_remainder_l4095_409528


namespace NUMINAMATH_CALUDE_sisters_birth_year_l4095_409599

/-- Represents the birth years of family members --/
structure FamilyBirthYears where
  brother : Nat
  sister : Nat
  grandmother : Nat

/-- Checks if the birth years satisfy the given conditions --/
def validBirthYears (years : FamilyBirthYears) : Prop :=
  years.brother = 1932 ∧
  years.grandmother = 1944 ∧
  (years.grandmother - years.sister) = 2 * (years.sister - years.brother)

/-- Theorem stating that the grandmother's older sister was born in 1936 --/
theorem sisters_birth_year (years : FamilyBirthYears) 
  (h : validBirthYears years) : years.sister = 1936 := by
  sorry

#check sisters_birth_year

end NUMINAMATH_CALUDE_sisters_birth_year_l4095_409599


namespace NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l4095_409548

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 3| - 2 * |x + a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f x 3 > 2} = {x : ℝ | -7 < x ∧ x < -5/3} := by sorry

-- Part II
theorem solution_set_part_ii (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) (-1), f x a + x + 1 ≤ 0) →
  a ≥ 4 ∨ a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l4095_409548


namespace NUMINAMATH_CALUDE_worst_player_is_father_l4095_409574

-- Define the family members
inductive FamilyMember
  | Father
  | Sister
  | Daughter
  | Son

-- Define the sex of a family member
def sex : FamilyMember → Bool
  | FamilyMember.Father => true   -- true represents male
  | FamilyMember.Sister => false  -- false represents female
  | FamilyMember.Daughter => false
  | FamilyMember.Son => true

-- Define the twin relationship
def isTwin : FamilyMember → FamilyMember → Bool
  | FamilyMember.Father, FamilyMember.Sister => true
  | FamilyMember.Sister, FamilyMember.Father => true
  | FamilyMember.Daughter, FamilyMember.Son => true
  | FamilyMember.Son, FamilyMember.Daughter => true
  | _, _ => false

-- Define the theorem
theorem worst_player_is_father :
  ∀ (worst best : FamilyMember),
    (∃ twin : FamilyMember, isTwin worst twin ∧ sex twin ≠ sex best) →
    isTwin worst best →
    worst = FamilyMember.Father :=
by sorry

end NUMINAMATH_CALUDE_worst_player_is_father_l4095_409574


namespace NUMINAMATH_CALUDE_arrangement_existence_l4095_409580

/-- Represents a group of kindergarten children -/
structure ChildrenGroup where
  total : ℕ  -- Total number of children

/-- Represents an arrangement of children in pairs -/
structure Arrangement where
  boy_pairs : ℕ  -- Number of pairs of two boys
  girl_pairs : ℕ  -- Number of pairs of two girls
  mixed_pairs : ℕ  -- Number of pairs with one boy and one girl

/-- Checks if an arrangement is valid for a given group -/
def is_valid_arrangement (group : ChildrenGroup) (arr : Arrangement) : Prop :=
  2 * (arr.boy_pairs + arr.girl_pairs) + arr.mixed_pairs = group.total

/-- Theorem stating the existence of a specific arrangement -/
theorem arrangement_existence (group : ChildrenGroup) 
  (arr1 arr2 : Arrangement) 
  (h1 : is_valid_arrangement group arr1)
  (h2 : is_valid_arrangement group arr2)
  (h3 : arr1.boy_pairs = 3 * arr1.girl_pairs)
  (h4 : arr2.boy_pairs = 4 * arr2.girl_pairs) :
  ∃ (arr3 : Arrangement), 
    is_valid_arrangement group arr3 ∧ 
    arr3.boy_pairs = 7 * arr3.girl_pairs := by
  sorry

end NUMINAMATH_CALUDE_arrangement_existence_l4095_409580


namespace NUMINAMATH_CALUDE_equation_solution_l4095_409530

theorem equation_solution : ∃ x : ℝ, (6 + 1.5 * x = 2.5 * x - 30 + Real.sqrt 100) ∧ x = 26 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4095_409530


namespace NUMINAMATH_CALUDE_evaluate_dagger_l4095_409568

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem evaluate_dagger : dagger (5/16) (12/5) = 75/4 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_dagger_l4095_409568


namespace NUMINAMATH_CALUDE_impossible_to_blacken_board_l4095_409500

/-- Represents the state of a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- A move is represented by its top-left corner and orientation -/
structure Move where
  row : Fin 8
  col : Fin 8
  horizontal : Bool

/-- Apply a move to a chessboard -/
def applyMove (board : Chessboard) (move : Move) : Chessboard :=
  sorry

/-- Count the number of black squares on the board -/
def countBlackSquares (board : Chessboard) : Nat :=
  sorry

/-- The initial all-white chessboard -/
def initialBoard : Chessboard :=
  fun _ _ => false

/-- The final all-black chessboard -/
def finalBoard : Chessboard :=
  fun _ _ => true

/-- Theorem: It's impossible to transform the initial board to the final board using only valid moves -/
theorem impossible_to_blacken_board :
  ¬∃ (moves : List Move), (moves.foldl applyMove initialBoard) = finalBoard :=
sorry

end NUMINAMATH_CALUDE_impossible_to_blacken_board_l4095_409500


namespace NUMINAMATH_CALUDE_zoo_animal_difference_l4095_409547

theorem zoo_animal_difference (parrots : ℕ) (snakes : ℕ) (monkeys : ℕ) (elephants : ℕ) (zebras : ℕ) : 
  parrots = 8 → 
  snakes = 3 * parrots → 
  monkeys = 2 * snakes → 
  elephants = (parrots + snakes) / 2 → 
  zebras = elephants - 3 → 
  monkeys - zebras = 35 := by
sorry

end NUMINAMATH_CALUDE_zoo_animal_difference_l4095_409547


namespace NUMINAMATH_CALUDE_joe_spending_l4095_409508

def entrance_fee_under_18 : ℝ := 5
def entrance_fee_over_18 : ℝ := entrance_fee_under_18 * 1.2
def group_discount_rate : ℝ := 0.15
def ride_cost : ℝ := 0.5
def joe_age : ℕ := 30
def twin_age : ℕ := 6
def joe_rides : ℕ := 4
def twin_a_rides : ℕ := 3
def twin_b_rides : ℕ := 5

def group_size : ℕ := 3

def total_entrance_fee : ℝ := 
  entrance_fee_over_18 + 2 * entrance_fee_under_18

def discounted_entrance_fee : ℝ := 
  total_entrance_fee * (1 - group_discount_rate)

def total_ride_cost : ℝ := 
  ride_cost * (joe_rides + twin_a_rides + twin_b_rides)

theorem joe_spending (joe_spending : ℝ) : 
  joe_spending = discounted_entrance_fee + total_ride_cost ∧ 
  joe_spending = 19.60 := by sorry

end NUMINAMATH_CALUDE_joe_spending_l4095_409508


namespace NUMINAMATH_CALUDE_bridgette_dogs_l4095_409544

/-- Represents the number of baths given to an animal in a year. -/
def baths_per_year (frequency : ℕ) : ℕ := 12 / frequency

/-- Represents the total number of baths given to a group of animals in a year. -/
def total_baths (num_animals : ℕ) (frequency : ℕ) : ℕ :=
  num_animals * baths_per_year frequency

theorem bridgette_dogs :
  ∃ (num_dogs : ℕ),
    total_baths num_dogs 2 + -- Dogs bathed twice a month
    total_baths 3 1 + -- 3 cats bathed once a month
    total_baths 4 4 = 96 ∧ -- 4 birds bathed once every 4 months
    num_dogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_bridgette_dogs_l4095_409544


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4095_409531

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = Set.Ioo a (1/a) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4095_409531


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l4095_409537

theorem sqrt_fraction_simplification : 
  (Real.sqrt 3) / ((Real.sqrt 3) + (Real.sqrt 12)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l4095_409537


namespace NUMINAMATH_CALUDE_equation_solution_l4095_409591

theorem equation_solution (a : ℝ) : (3 * 5 + 2 * a = 3) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4095_409591


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_pow_5_minus_5_pow_4_l4095_409554

theorem least_prime_factor_of_5_pow_5_minus_5_pow_4 :
  Nat.minFac (5^5 - 5^4) = 2 := by
sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_pow_5_minus_5_pow_4_l4095_409554


namespace NUMINAMATH_CALUDE_initial_cards_l4095_409582

theorem initial_cards (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 3 → total = 7 → initial + added = total → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_cards_l4095_409582


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l4095_409560

/-- The probability of finding treasure and no traps on a single island -/
def p_treasure : ℚ := 1/5

/-- The probability of finding traps and no treasure on a single island -/
def p_traps : ℚ := 1/10

/-- The probability of finding neither treasure nor traps on a single island -/
def p_neither : ℚ := 7/10

/-- The total number of islands -/
def total_islands : ℕ := 8

/-- The number of islands with treasure we want to find -/
def treasure_islands : ℕ := 4

theorem pirate_treasure_probability :
  (Nat.choose total_islands treasure_islands : ℚ) *
  p_treasure ^ treasure_islands *
  p_neither ^ (total_islands - treasure_islands) =
  33614 / 1250000 := by
  sorry


end NUMINAMATH_CALUDE_pirate_treasure_probability_l4095_409560


namespace NUMINAMATH_CALUDE_polyhedron_property_l4095_409555

/-- A convex polyhedron with the given properties -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  t : ℕ  -- number of triangular faces
  h : ℕ  -- number of hexagonal faces
  T : ℕ  -- number of triangular faces meeting at each vertex
  H : ℕ  -- number of hexagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 40
  face_composition : F = t + h
  vertex_property : 2 * T + H = 7
  edge_count : E = (3 * t + 6 * h) / 2

theorem polyhedron_property (P : ConvexPolyhedron) : 100 * P.H + 10 * P.T + P.V = 367 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_property_l4095_409555


namespace NUMINAMATH_CALUDE_complex_roots_of_quadratic_l4095_409596

theorem complex_roots_of_quadratic (a b : ℝ) : 
  (Complex.I + 1) ^ 2 + a * (Complex.I + 1) + b = 0 → 
  (a = -2 ∧ b = 2) ∧ (Complex.I - 1) ^ 2 + a * (Complex.I - 1) + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_of_quadratic_l4095_409596


namespace NUMINAMATH_CALUDE_solution_existence_condition_l4095_409587

theorem solution_existence_condition (m : ℝ) : 
  (∃ x ∈ Set.Icc 0 2, x^3 - 3*x + m = 0) → m ≤ 2 ∧ 
  ¬(∀ m ≤ 2, ∃ x ∈ Set.Icc 0 2, x^3 - 3*x + m = 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_existence_condition_l4095_409587


namespace NUMINAMATH_CALUDE_object_length_increase_l4095_409564

/-- The number of days required for an object to reach 50 times its original length -/
def n : ℕ := 147

/-- The factor by which the object's length increases on day k -/
def increase_factor (k : ℕ) : ℚ := (k + 3 : ℚ) / (k + 2 : ℚ)

/-- The total increase factor after n days -/
def total_increase_factor (n : ℕ) : ℚ := (n + 3 : ℚ) / 3

theorem object_length_increase :
  total_increase_factor n = 50 := by sorry

end NUMINAMATH_CALUDE_object_length_increase_l4095_409564


namespace NUMINAMATH_CALUDE_tangent_slope_sin_pi_sixth_l4095_409597

theorem tangent_slope_sin_pi_sixth :
  let f : ℝ → ℝ := λ x ↦ Real.sin x
  (deriv f) (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_sin_pi_sixth_l4095_409597


namespace NUMINAMATH_CALUDE_jaime_sum_with_square_l4095_409577

theorem jaime_sum_with_square (n : ℕ) (k : ℕ) : 
  (∃ (i : ℕ), i < 100 ∧ n + i = k) →
  (50 * (2 * n + 99) - k + k^2 = 7500) →
  k = 26 := by
sorry

end NUMINAMATH_CALUDE_jaime_sum_with_square_l4095_409577


namespace NUMINAMATH_CALUDE_ticket_distribution_ways_l4095_409536

/-- The number of ways to distribute tickets among programs -/
def distribute_tickets (total_tickets : ℕ) (num_programs : ℕ) (min_tickets_a : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 6 tickets among 4 programs with program A receiving at least 3 and the most -/
theorem ticket_distribution_ways : distribute_tickets 6 4 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_ways_l4095_409536


namespace NUMINAMATH_CALUDE_insect_legs_count_l4095_409525

theorem insect_legs_count (num_insects : ℕ) (legs_per_insect : ℕ) : 
  num_insects = 5 → legs_per_insect = 6 → num_insects * legs_per_insect = 30 := by
  sorry

end NUMINAMATH_CALUDE_insect_legs_count_l4095_409525


namespace NUMINAMATH_CALUDE_vertical_shift_graph_l4095_409543

-- Define a type for functions from real numbers to real numbers
def RealFunction := ℝ → ℝ

-- Define a vertical shift operation on functions
def verticalShift (f : RealFunction) (k : ℝ) : RealFunction :=
  λ x => f x + k

-- State the theorem
theorem vertical_shift_graph (f : RealFunction) (k : ℝ) :
  ∀ x y, y = f x + k ↔ y - k = f x :=
sorry

end NUMINAMATH_CALUDE_vertical_shift_graph_l4095_409543


namespace NUMINAMATH_CALUDE_sqrt_054_in_terms_of_sqrt_2_and_sqrt_3_l4095_409504

theorem sqrt_054_in_terms_of_sqrt_2_and_sqrt_3 (a b : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) : 
  Real.sqrt 0.54 = 0.3 * a * b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_054_in_terms_of_sqrt_2_and_sqrt_3_l4095_409504


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l4095_409505

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 + a 5 = Real.pi →
  a 4 * (a 2 + 2 * a 4 + a 6) = Real.pi^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l4095_409505


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l4095_409520

def total_knights : ℕ := 30
def chosen_knights : ℕ := 4

def prob_adjacent_knights : ℚ :=
  1 - (26 * 24 * 22 * 20 : ℚ) / (26 * 27 * 28 * 29 : ℚ)

theorem adjacent_knights_probability :
  prob_adjacent_knights = 553 / 1079 := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l4095_409520


namespace NUMINAMATH_CALUDE_inequality_proof_l4095_409561

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 1/2) :
  1/(1-a) + 1/(1-b) ≥ 4 ∧ (1/(1-a) + 1/(1-b) = 4 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l4095_409561


namespace NUMINAMATH_CALUDE_one_fifth_of_ten_x_plus_five_l4095_409552

theorem one_fifth_of_ten_x_plus_five (x : ℝ) : (1 / 5) * (10 * x + 5) = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_ten_x_plus_five_l4095_409552


namespace NUMINAMATH_CALUDE_jakes_weight_ratio_l4095_409595

/-- Proves that the ratio of Jake's weight after losing 8 pounds to his sister's weight is 2:1 -/
theorem jakes_weight_ratio :
  let jake_current_weight : ℕ := 188
  let total_weight : ℕ := 278
  let weight_loss : ℕ := 8
  let jake_new_weight : ℕ := jake_current_weight - weight_loss
  let sister_weight : ℕ := total_weight - jake_current_weight
  (jake_new_weight : ℚ) / (sister_weight : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_jakes_weight_ratio_l4095_409595


namespace NUMINAMATH_CALUDE_lcm_18_35_l4095_409503

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_35_l4095_409503


namespace NUMINAMATH_CALUDE_initial_average_mark_l4095_409578

/-- Proves that the initial average mark of a class is 60, given the specified conditions. -/
theorem initial_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_avg : ℚ) (remaining_avg : ℚ) :
  total_students = 9 →
  excluded_students = 5 →
  excluded_avg = 44 →
  remaining_avg = 80 →
  (total_students * (total_students * excluded_avg + (total_students - excluded_students) * remaining_avg)) / 
  (excluded_students * total_students + (total_students - excluded_students) * total_students) = 60 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_mark_l4095_409578


namespace NUMINAMATH_CALUDE_even_function_implies_m_eq_two_l4095_409514

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The given function f(x) -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem even_function_implies_m_eq_two :
  ∀ m : ℝ, IsEven (f m) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_eq_two_l4095_409514


namespace NUMINAMATH_CALUDE_unknown_number_proof_l4095_409521

theorem unknown_number_proof (x : ℝ) : 
  (14 + 32 + 53) / 3 = (21 + 47 + x) / 3 + 3 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l4095_409521


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_and_eccentricity_l4095_409559

/-- Given a hyperbola with equation x² - y²/3 = 1, prove its focal length is 4 and eccentricity is 2 -/
theorem hyperbola_focal_length_and_eccentricity :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2/3 = 1
  ∃ (a b c : ℝ),
    (a = 1 ∧ b^2 = 3) ∧
    (c^2 = a^2 + b^2) ∧
    (2 * c = 4) ∧
    (c / a = 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_and_eccentricity_l4095_409559


namespace NUMINAMATH_CALUDE_discounted_cost_l4095_409563

/-- The cost of a pencil without discount -/
def pencil_cost : ℚ := sorry

/-- The cost of a notebook -/
def notebook_cost : ℚ := sorry

/-- The discount per pencil when buying more than 10 pencils -/
def discount : ℚ := 0.05

/-- Condition: Cost of 8 pencils and 10 notebooks without discount -/
axiom condition1 : 8 * pencil_cost + 10 * notebook_cost = 5.36

/-- Condition: Cost of 12 pencils and 5 notebooks with discount -/
axiom condition2 : 12 * (pencil_cost - discount) + 5 * notebook_cost = 4.05

/-- The cost of 15 pencils and 12 notebooks with discount -/
def total_cost : ℚ := 15 * (pencil_cost - discount) + 12 * notebook_cost

theorem discounted_cost : total_cost = 7.01 := by sorry

end NUMINAMATH_CALUDE_discounted_cost_l4095_409563


namespace NUMINAMATH_CALUDE_complex_number_properties_l4095_409569

theorem complex_number_properties (z : ℂ) (h : Complex.I * (z + 1) = -2 + 2 * Complex.I) :
  (Complex.im z = 2) ∧ (let ω := z / (1 - 2 * Complex.I); Complex.abs ω ^ 2015 = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l4095_409569


namespace NUMINAMATH_CALUDE_composite_numbers_l4095_409558

theorem composite_numbers (n : ℕ) (h : n > 0) : 
  (∃ k : ℕ, k > 1 ∧ k < 2 * 2^(2^(2*n)) + 1 ∧ (2 * 2^(2^(2*n)) + 1) % k = 0) ∧ 
  (∃ m : ℕ, m > 1 ∧ m < 3 * 2^(2*n) + 1 ∧ (3 * 2^(2*n) + 1) % m = 0) := by
sorry

end NUMINAMATH_CALUDE_composite_numbers_l4095_409558


namespace NUMINAMATH_CALUDE_child_tickets_sold_l4095_409592

/-- Proves the number of child tickets sold given the ticket prices and total sales information -/
theorem child_tickets_sold 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_sales : ℕ) 
  (total_tickets : ℕ) 
  (h1 : adult_price = 5)
  (h2 : child_price = 3)
  (h3 : total_sales = 178)
  (h4 : total_tickets = 42) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_sales ∧
    child_tickets = 16 := by
  sorry


end NUMINAMATH_CALUDE_child_tickets_sold_l4095_409592


namespace NUMINAMATH_CALUDE_greatest_integer_x_l4095_409572

theorem greatest_integer_x (x : ℕ) : x^4 / x^2 < 18 → x ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_x_l4095_409572


namespace NUMINAMATH_CALUDE_function_ordering_l4095_409556

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_ordering (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_ordering_l4095_409556


namespace NUMINAMATH_CALUDE_middle_digit_guess_probability_l4095_409524

/-- Represents a three-digit lock --/
structure DigitLock :=
  (first : Nat)
  (second : Nat)
  (third : Nat)

/-- Condition: Each digit is between 0 and 9 --/
def isValidDigit (d : Nat) : Prop := d ≤ 9

/-- A lock is valid if all its digits are valid --/
def isValidLock (lock : DigitLock) : Prop :=
  isValidDigit lock.first ∧ isValidDigit lock.second ∧ isValidDigit lock.third

/-- The probability of guessing the middle digit correctly --/
def middleDigitGuessProbability (lock : DigitLock) : ℚ :=
  1 / 10

/-- Theorem: The probability of guessing the middle digit of a valid lock is 1/10 --/
theorem middle_digit_guess_probability 
  (lock : DigitLock) 
  (h : isValidLock lock) : 
  middleDigitGuessProbability lock = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_middle_digit_guess_probability_l4095_409524


namespace NUMINAMATH_CALUDE_vector_operations_l4095_409575

theorem vector_operations (a b : ℝ × ℝ) :
  a = (1, 2) → b = (3, 1) →
  (a + b = (4, 3)) ∧ (a.1 * b.1 + a.2 * b.2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l4095_409575


namespace NUMINAMATH_CALUDE_square_field_side_length_l4095_409511

theorem square_field_side_length (area : Real) (side_length : Real) :
  area = 196 ∧ area = side_length ^ 2 → side_length = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_field_side_length_l4095_409511


namespace NUMINAMATH_CALUDE_car_profit_percent_l4095_409546

/-- Calculate the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent (car_cost repair_cost taxes insurance selling_price : ℝ) :
  car_cost = 36400 →
  repair_cost = 8000 →
  taxes = 4500 →
  insurance = 2500 →
  selling_price = 68400 →
  let total_cost := car_cost + repair_cost + taxes + insurance
  let profit := selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  abs (profit_percent - 33.07) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_car_profit_percent_l4095_409546


namespace NUMINAMATH_CALUDE_inverse_inequality_l4095_409519

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l4095_409519


namespace NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l4095_409522

theorem range_of_a_minus_abs_b (a b : ℝ) :
  1 < a ∧ a < 8 ∧ -4 < b ∧ b < 2 →
  ∃ x, -3 < x ∧ x < 8 ∧ x = a - |b| :=
sorry

end NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l4095_409522


namespace NUMINAMATH_CALUDE_correct_division_incorrect_others_l4095_409533

theorem correct_division_incorrect_others :
  ((-8) / (-4) = 8 / 4) ∧
  ¬((-5) + 9 = -(9 - 5)) ∧
  ¬(7 - (-10) = 7 - 10) ∧
  ¬((-5) * 0 = -5) := by
  sorry

end NUMINAMATH_CALUDE_correct_division_incorrect_others_l4095_409533


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l4095_409541

def point : ℝ × ℝ := (-2, 3)

def is_in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_in_second_quadrant : is_in_second_quadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l4095_409541


namespace NUMINAMATH_CALUDE_max_value_squared_l4095_409506

theorem max_value_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y)
  (h : x^3 + 2013*y = y^3 + 2013*x) :
  ∃ (M : ℝ), M = (Real.sqrt 3 + 1) * x + 2 * y ∧
    ∀ (N : ℝ), N = (Real.sqrt 3 + 1) * x + 2 * y → N^2 ≤ 16104 :=
by sorry

end NUMINAMATH_CALUDE_max_value_squared_l4095_409506


namespace NUMINAMATH_CALUDE_find_a_find_m_range_l4095_409517

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

-- Part 1
theorem find_a : 
  (∀ x, f 1 x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧ 
  (∀ a, (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1) :=
sorry

-- Part 2
theorem find_m_range : 
  ∀ m : ℝ, (∃ n : ℝ, f 1 n ≤ m - f 1 (-n)) ↔ m ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_find_a_find_m_range_l4095_409517


namespace NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l4095_409542

/-- The sum of the infinite series Σ(n=1 to ∞) (n^5 + 5n^3 + 15n + 15) / (2^n * (n^5 + 5)) is equal to 1 -/
theorem infinite_series_sum_equals_one :
  let f : ℕ → ℝ := λ n => (n^5 + 5*n^3 + 15*n + 15) / (2^n * (n^5 + 5))
  ∑' n, f n = 1 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l4095_409542


namespace NUMINAMATH_CALUDE_tan_150_degrees_l4095_409526

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l4095_409526


namespace NUMINAMATH_CALUDE_sum_1_to_12_mod_9_l4095_409576

theorem sum_1_to_12_mod_9 : (List.sum (List.range 12)).mod 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_1_to_12_mod_9_l4095_409576


namespace NUMINAMATH_CALUDE_seventh_group_sample_l4095_409535

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (m : ℕ) (k : ℕ) : ℕ :=
  10 * (k - 1) + (m + k) % 10

/-- The problem statement translated to a theorem -/
theorem seventh_group_sample :
  ∀ m : ℕ,
  m = 6 →
  systematicSample m 7 = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_seventh_group_sample_l4095_409535


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l4095_409573

theorem cubic_equation_roots :
  let f : ℝ → ℝ := λ x ↦ x^3 - 2*x
  f 0 = 0 ∧ f (Real.sqrt 2) = 0 ∧ f (-Real.sqrt 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l4095_409573


namespace NUMINAMATH_CALUDE_kaleb_toys_l4095_409585

def number_of_toys (initial_savings allowance toy_cost : ℕ) : ℕ :=
  (initial_savings + allowance) / toy_cost

theorem kaleb_toys : number_of_toys 21 15 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_toys_l4095_409585


namespace NUMINAMATH_CALUDE_initial_stock_proof_l4095_409579

/-- The number of coloring books sold during the sale -/
def books_sold : ℕ := 6

/-- The number of shelves used for remaining books -/
def shelves_used : ℕ := 3

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 7

/-- The initial number of coloring books in stock -/
def initial_stock : ℕ := shelves_used * books_per_shelf + books_sold

theorem initial_stock_proof : initial_stock = 27 := by
  sorry

end NUMINAMATH_CALUDE_initial_stock_proof_l4095_409579


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l4095_409501

/-- Proves that 66 gallons of fuel A were added to a 204-gallon tank -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 204 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (fuel_a : ℝ), 
    fuel_a = 66 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l4095_409501


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l4095_409589

/-- The area of a circle with diameter endpoints at (1, 3) and (8, 6) is 58π/4 square units. -/
theorem circle_area_from_diameter_endpoints :
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (8, 6)
  let diameter_squared := (B.1 - A.1)^2 + (B.2 - A.2)^2
  let radius_squared := diameter_squared / 4
  let circle_area := π * radius_squared
  circle_area = 58 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l4095_409589


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4095_409551

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x - 5 ≥ 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 5/2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4095_409551


namespace NUMINAMATH_CALUDE_chord_length_theorem_l4095_409509

/-- In a right triangle ABC with inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of leg AB -/
  a : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- a and r are positive -/
  a_pos : 0 < a
  r_pos : 0 < r

/-- The chord length theorem -/
theorem chord_length_theorem (t : RightTriangleWithInscribedCircle) :
  ∃ (chord_length : ℝ),
    chord_length = (2 * t.a * t.r) / Real.sqrt (t.a^2 + t.r^2) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l4095_409509


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l4095_409571

theorem equilateral_triangle_side_length 
  (circular_radius : ℝ) 
  (circular_speed : ℝ) 
  (triangular_speed : ℝ) 
  (h1 : circular_radius = 60) 
  (h2 : circular_speed = 6) 
  (h3 : triangular_speed = 5) :
  ∃ x : ℝ, 
    (3 * x = triangular_speed * ((2 * Real.pi * circular_radius) / circular_speed)) ∧ 
    x = 100 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l4095_409571


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4095_409590

theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (2 - i) / (1 + 4*i) = -2/17 - (9/17)*i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4095_409590


namespace NUMINAMATH_CALUDE_y_values_l4095_409527

theorem y_values (x : ℝ) (h : x^2 + 6 * (x / (x - 3))^2 = 72) :
  let y := ((x - 3)^3 * (x + 4)) / (3 * x - 4)
  y = 135 / 7 ∨ y = 216 / 13 := by sorry

end NUMINAMATH_CALUDE_y_values_l4095_409527


namespace NUMINAMATH_CALUDE_bisection_method_structures_l4095_409581

/-- Bisection method for finding the approximate root of x^2 - 5 = 0 -/
def bisection_method (f : ℝ → ℝ) (a b : ℝ) (ε : ℝ) : ℝ := sorry

/-- The equation to solve -/
def equation (x : ℝ) : ℝ := x^2 - 5

theorem bisection_method_structures :
  ∃ (sequential conditional loop : Bool),
    sequential ∧ conditional ∧ loop ∧
    (∀ (a b ε : ℝ), ε > 0 → 
      ∃ (result : ℝ), 
        bisection_method equation a b ε = result ∧ 
        |equation result| < ε) :=
sorry

end NUMINAMATH_CALUDE_bisection_method_structures_l4095_409581


namespace NUMINAMATH_CALUDE_six_couples_handshakes_l4095_409565

/-- The number of handshakes in a gathering of couples where each person
    shakes hands with everyone except their spouse -/
def handshakes (n : ℕ) : ℕ :=
  let total_people := 2 * n
  let total_potential_handshakes := total_people * (total_people - 1) / 2
  total_potential_handshakes - n

theorem six_couples_handshakes :
  handshakes 6 = 60 := by sorry

end NUMINAMATH_CALUDE_six_couples_handshakes_l4095_409565


namespace NUMINAMATH_CALUDE_line_slope_proportionality_l4095_409545

/-- Given a line where an increase of 3 units in x corresponds to an increase of 7 units in y,
    prove that an increase of 9 units in x results in an increase of 21 units in y. -/
theorem line_slope_proportionality (f : ℝ → ℝ) (x : ℝ) :
  (f (x + 3) - f x = 7) → (f (x + 9) - f x = 21) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_proportionality_l4095_409545


namespace NUMINAMATH_CALUDE_train_speed_l4095_409502

/-- Calculates the speed of a train given its length and time to cross an electric pole. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 800) (h2 : time = 20) :
  length / time = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l4095_409502


namespace NUMINAMATH_CALUDE_units_digit_of_6541_pow_826_l4095_409562

theorem units_digit_of_6541_pow_826 : (6541^826) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_6541_pow_826_l4095_409562


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l4095_409532

/-- The inradius of a right triangle with side lengths 7, 24, and 25 is 3 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 7 ∧ b = 24 ∧ c = 25 →
  a^2 + b^2 = c^2 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 3 := by sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l4095_409532


namespace NUMINAMATH_CALUDE_pizza_piece_volume_l4095_409513

/-- The volume of a piece of pizza -/
theorem pizza_piece_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) :
  thickness = 1/4 →
  diameter = 16 →
  num_pieces = 16 →
  (π * (diameter/2)^2 * thickness) / num_pieces = π := by
  sorry

end NUMINAMATH_CALUDE_pizza_piece_volume_l4095_409513


namespace NUMINAMATH_CALUDE_vector_equation_solution_l4095_409523

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) : 
  a = (2, 1) → b = (1, -2) → m • a + n • b = (9, -8) → m - n = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l4095_409523


namespace NUMINAMATH_CALUDE_max_value_inequality_l4095_409557

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 2*y)^2 / (x^2 + y^2) ≤ 9/2 := by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l4095_409557


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4620_l4095_409598

theorem largest_prime_factor_of_4620 : 
  (Nat.factors 4620).maximum? = some 11 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4620_l4095_409598


namespace NUMINAMATH_CALUDE_problem_solution_l4095_409510

theorem problem_solution (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4095_409510


namespace NUMINAMATH_CALUDE_complex_division_result_l4095_409570

theorem complex_division_result : (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l4095_409570


namespace NUMINAMATH_CALUDE_books_sold_l4095_409588

theorem books_sold (total_books : ℕ) (fraction_left : ℚ) (books_sold : ℕ) : 
  total_books = 15750 →
  fraction_left = 7 / 23 →
  books_sold = total_books - (total_books * fraction_left).floor →
  books_sold = 10957 := by
sorry

end NUMINAMATH_CALUDE_books_sold_l4095_409588


namespace NUMINAMATH_CALUDE_total_episodes_watched_l4095_409549

def episode_length : ℕ := 44
def monday_minutes : ℕ := 138
def thursday_minutes : ℕ := 21
def friday_episodes : ℕ := 2
def weekend_minutes : ℕ := 105

theorem total_episodes_watched :
  (monday_minutes + thursday_minutes + friday_episodes * episode_length + weekend_minutes) / episode_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_episodes_watched_l4095_409549


namespace NUMINAMATH_CALUDE_circles_intersect_l4095_409584

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are intersecting -/
def are_intersecting (c1 c2 : Circle) : Prop :=
  let d := Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2)
  abs (c1.radius - c2.radius) < d ∧ d < c1.radius + c2.radius

theorem circles_intersect : 
  let circle1 : Circle := { center := (0, 0), radius := 2 }
  let circle2 : Circle := { center := (2, 0), radius := 3 }
  are_intersecting circle1 circle2 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l4095_409584


namespace NUMINAMATH_CALUDE_lizzy_money_calculation_l4095_409507

/-- Calculates Lizzy's final amount after lending money and receiving it back with interest -/
def lizzys_final_amount (initial_amount loan_amount interest_rate : ℚ) : ℚ :=
  initial_amount - loan_amount + loan_amount * (1 + interest_rate)

/-- Theorem stating that Lizzy will have $33 after lending $15 from her initial $30 and receiving it back with 20% interest -/
theorem lizzy_money_calculation :
  lizzys_final_amount 30 15 (1/5) = 33 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_money_calculation_l4095_409507


namespace NUMINAMATH_CALUDE_planes_formed_by_three_lines_through_point_l4095_409594

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in three-dimensional space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents the number of planes formed by three lines -/
inductive NumPlanes
  | one
  | three

/-- Given a point and three lines through it, determines the number of planes formed -/
def planesFormedByThreeLines (p : Point3D) (l1 l2 l3 : Line3D) : NumPlanes :=
  sorry

theorem planes_formed_by_three_lines_through_point 
  (p : Point3D) (l1 l2 l3 : Line3D) 
  (h1 : l1.point = p) (h2 : l2.point = p) (h3 : l3.point = p) :
  planesFormedByThreeLines p l1 l2 l3 = NumPlanes.one ∨ 
  planesFormedByThreeLines p l1 l2 l3 = NumPlanes.three :=
sorry

end NUMINAMATH_CALUDE_planes_formed_by_three_lines_through_point_l4095_409594


namespace NUMINAMATH_CALUDE_tv_price_increase_l4095_409583

theorem tv_price_increase (P : ℝ) (x : ℝ) : 
  (1.30 * P) * (1 + x / 100) = 1.82 * P ↔ x = 40 :=
sorry

end NUMINAMATH_CALUDE_tv_price_increase_l4095_409583


namespace NUMINAMATH_CALUDE_quadratic_even_function_coeff_l4095_409518

/-- A quadratic function f(x) = ax^2 + (2a^2 - a)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 * a^2 - a) * x + 1

/-- Definition of an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem quadratic_even_function_coeff (a : ℝ) :
  is_even_function (f a) → a = 1/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_even_function_coeff_l4095_409518


namespace NUMINAMATH_CALUDE_prob_ice_given_ski_l4095_409566

/-- The probability that a high school student likes ice skating -/
def P_ice_skating : ℝ := 0.6

/-- The probability that a high school student likes skiing -/
def P_skiing : ℝ := 0.5

/-- The probability that a high school student likes either ice skating or skiing -/
def P_ice_or_ski : ℝ := 0.7

/-- The probability that a high school student likes both ice skating and skiing -/
def P_ice_and_ski : ℝ := P_ice_skating + P_skiing - P_ice_or_ski

theorem prob_ice_given_ski :
  P_ice_and_ski / P_skiing = 0.8 := by sorry

end NUMINAMATH_CALUDE_prob_ice_given_ski_l4095_409566


namespace NUMINAMATH_CALUDE_tapanga_corey_candy_difference_l4095_409553

theorem tapanga_corey_candy_difference (total : ℕ) (corey : ℕ) (h1 : total = 66) (h2 : corey = 29) (h3 : corey < total - corey) :
  total - corey - corey = 8 := by
  sorry

end NUMINAMATH_CALUDE_tapanga_corey_candy_difference_l4095_409553


namespace NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l4095_409534

theorem scientific_notation_of_56_99_million :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    56990000 = a * (10 : ℝ) ^ n ∧
    a = 5.699 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l4095_409534


namespace NUMINAMATH_CALUDE_time_to_walk_five_miles_l4095_409529

/-- Given that Tom walks 2 miles in 6 minutes, prove that it takes 15 minutes to walk 5 miles at the same rate. -/
theorem time_to_walk_five_miles (distance_to_jerry : ℝ) (time_to_jerry : ℝ) (distance_to_sam : ℝ) :
  distance_to_jerry = 2 →
  time_to_jerry = 6 →
  distance_to_sam = 5 →
  (distance_to_sam / (distance_to_jerry / time_to_jerry)) = 15 := by
sorry

end NUMINAMATH_CALUDE_time_to_walk_five_miles_l4095_409529


namespace NUMINAMATH_CALUDE_circle_condition_l4095_409567

theorem circle_condition (m : ℝ) :
  (∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔
  (m < 1/4 ∨ m > 1) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l4095_409567


namespace NUMINAMATH_CALUDE_negative_200_means_send_out_l4095_409515

/-- Represents a WeChat payment transaction -/
structure WeChatTransaction where
  amount : ℝ
  balance_before : ℝ
  balance_after : ℝ

/-- Axiom: Receiving money increases the balance -/
axiom receive_increases_balance {t : WeChatTransaction} (h : t.amount > 0) : 
  t.balance_after = t.balance_before + t.amount

/-- Axiom: Sending money decreases the balance -/
axiom send_decreases_balance {t : WeChatTransaction} (h : t.amount < 0) :
  t.balance_after = t.balance_before + t.amount

/-- The meaning of a -200 transaction in WeChat payments -/
theorem negative_200_means_send_out (t : WeChatTransaction) 
  (h1 : t.amount = -200)
  (h2 : t.balance_before = 867.35)
  (h3 : t.balance_after = 667.35) :
  "Sending out 200 yuan" = "The meaning of -200 in WeChat payments" := by
  sorry

end NUMINAMATH_CALUDE_negative_200_means_send_out_l4095_409515


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l4095_409550

/-- Proves that a train with given speed and crossing time will take 10 seconds to pass a pole -/
theorem train_passing_pole_time 
  (train_speed_kmh : ℝ) 
  (stationary_train_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_speed_kmh = 72) 
  (h2 : stationary_train_length = 500) 
  (h3 : crossing_time = 35) :
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let train_length := train_speed_ms * crossing_time - stationary_train_length
  train_length / train_speed_ms = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l4095_409550


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l4095_409586

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 10*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (3/2) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l4095_409586


namespace NUMINAMATH_CALUDE_lineup_constraint_ways_l4095_409540

/-- The number of ways to arrange 5 people in a line with constraints -/
def lineupWays : ℕ :=
  let totalPeople : ℕ := 5
  let firstPositionOptions : ℕ := totalPeople - 1
  let lastPositionOptions : ℕ := totalPeople - 2
  let middlePositionsOptions : ℕ := 3 * 2 * 1
  firstPositionOptions * lastPositionOptions * middlePositionsOptions

theorem lineup_constraint_ways :
  lineupWays = 216 := by
  sorry

end NUMINAMATH_CALUDE_lineup_constraint_ways_l4095_409540
