import Mathlib

namespace NUMINAMATH_CALUDE_sine_shift_left_l2314_231459

/-- Shifting a sine function to the left --/
theorem sine_shift_left (x : ℝ) :
  let f (t : ℝ) := Real.sin t
  let g (t : ℝ) := Real.sin (t + π / 6)
  ∀ y : ℝ, f (x + π / 6) = g x :=
by sorry

end NUMINAMATH_CALUDE_sine_shift_left_l2314_231459


namespace NUMINAMATH_CALUDE_trucks_distance_l2314_231496

-- Define the speeds of trucks A and B in km/h
def speed_A : ℝ := 54
def speed_B : ℝ := 72

-- Define the time elapsed in seconds
def time_elapsed : ℝ := 30

-- Define the conversion factor from km to meters
def km_to_meters : ℝ := 1000

-- Define the conversion factor from hours to seconds
def hours_to_seconds : ℝ := 3600

-- Theorem statement
theorem trucks_distance :
  let speed_A_mps := speed_A * km_to_meters / hours_to_seconds
  let speed_B_mps := speed_B * km_to_meters / hours_to_seconds
  let distance_A := speed_A_mps * time_elapsed
  let distance_B := speed_B_mps * time_elapsed
  distance_A + distance_B = 1050 :=
by sorry

end NUMINAMATH_CALUDE_trucks_distance_l2314_231496


namespace NUMINAMATH_CALUDE_solution_range_l2314_231446

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (x + m) / (x - 2) + (2 * m) / (2 - x) = 3) → 
  m < 6 ∧ m ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l2314_231446


namespace NUMINAMATH_CALUDE_birds_berries_consumption_l2314_231464

theorem birds_berries_consumption (num_birds : ℕ) (total_berries : ℕ) (num_days : ℕ) 
  (h1 : num_birds = 5)
  (h2 : total_berries = 140)
  (h3 : num_days = 4) :
  total_berries / num_days / num_birds = 7 := by
  sorry

end NUMINAMATH_CALUDE_birds_berries_consumption_l2314_231464


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l2314_231420

/-- The interval between segments in systematic sampling --/
def systematic_sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: Given a population of 2000 and a sample size of 40, 
    the interval between segments in systematic sampling is 50 --/
theorem systematic_sampling_interval_example :
  systematic_sampling_interval 2000 40 = 50 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l2314_231420


namespace NUMINAMATH_CALUDE_probability_of_cooking_l2314_231436

/-- The set of courses Xiao Ming is interested in -/
inductive Course
| Planting
| Cooking
| Pottery
| Woodworking

/-- The probability of selecting a specific course from the set of courses -/
def probability_of_course (c : Course) : ℚ :=
  1 / 4

/-- Theorem stating that the probability of selecting "Cooking" is 1/4 -/
theorem probability_of_cooking :
  probability_of_course Course.Cooking = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_cooking_l2314_231436


namespace NUMINAMATH_CALUDE_q_polynomial_form_l2314_231472

def q (x : ℝ) : ℝ := sorry

theorem q_polynomial_form :
  ∀ x, q x + (2*x^6 + 5*x^4 + 10*x^2) = (9*x^4 + 30*x^3 + 40*x^2 + 5*x + 3) →
  q x = -2*x^6 + 4*x^4 + 30*x^3 + 30*x^2 + 5*x + 3 :=
by sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l2314_231472


namespace NUMINAMATH_CALUDE_equivalent_inequalities_l2314_231456

theorem equivalent_inequalities :
  ∀ x : ℝ, (0 < x ∧ x < 1) ↔ ((1 / x > 1) ∧ (Real.log x < 0)) :=
sorry

end NUMINAMATH_CALUDE_equivalent_inequalities_l2314_231456


namespace NUMINAMATH_CALUDE_multiplication_problem_l2314_231409

theorem multiplication_problem (x : ℝ) (n : ℝ) (h1 : x = 13) (h2 : x * n = (36 - x) + 16) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l2314_231409


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l2314_231495

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

theorem intersection_implies_a_equals_one :
  ∀ a : ℝ, (A ∩ B a = {3}) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l2314_231495


namespace NUMINAMATH_CALUDE_least_four_digit_square_fourth_power_l2314_231484

theorem least_four_digit_square_fourth_power : ∃ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∃ a : ℕ, n = a ^ 2) ∧ 
  (∃ b : ℕ, n = b ^ 4) ∧
  (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) → (∃ c : ℕ, m = c ^ 2) → (∃ d : ℕ, m = d ^ 4) → n ≤ m) ∧
  n = 6561 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_square_fourth_power_l2314_231484


namespace NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l2314_231412

-- Define the arithmetic progression
def arithmetic_progression (a d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmetic_progression a d n + d

-- Define the geometric progression
def geometric_progression (g_1 g_2 g_3 : ℝ) : Prop :=
  g_2 ^ 2 = g_1 * g_3

-- Theorem statement
theorem smallest_third_term_of_geometric_progression :
  ∀ d : ℝ,
  let a := arithmetic_progression 9 d
  let g_1 := 9
  let g_2 := a 1 + 5
  let g_3 := a 2 + 30
  geometric_progression g_1 g_2 g_3 →
  ∃ min_g_3 : ℝ, min_g_3 = 29 - 20 * Real.sqrt 2 ∧
  ∀ other_g_3 : ℝ, geometric_progression g_1 g_2 other_g_3 → min_g_3 ≤ other_g_3 :=
sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l2314_231412


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l2314_231432

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 19 ∧ (1156 + x) % 25 = 0 ∧ ∀ (y : ℕ), y < x → (1156 + y) % 25 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l2314_231432


namespace NUMINAMATH_CALUDE_function_difference_l2314_231448

theorem function_difference (f : ℝ → ℝ) (h : ∀ x, f x = 8^x) :
  ∀ x, f (x + 1) - f x = 7 * f x := by
sorry

end NUMINAMATH_CALUDE_function_difference_l2314_231448


namespace NUMINAMATH_CALUDE_berry_fraction_proof_l2314_231401

theorem berry_fraction_proof (steve skylar stacy : ℕ) : 
  skylar = 20 →
  stacy = 32 →
  stacy = 3 * steve + 2 →
  steve * 2 = skylar :=
by
  sorry

end NUMINAMATH_CALUDE_berry_fraction_proof_l2314_231401


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2314_231457

/-- The area of a square with a diagonal of 28 meters is 392 square meters. -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 28) : 
  (d ^ 2 / 2) = 392 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2314_231457


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2314_231402

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * c = b^2 - a^2 →
  A = π / 6 →
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2314_231402


namespace NUMINAMATH_CALUDE_paper_cranes_count_l2314_231447

theorem paper_cranes_count (T : ℕ) : 
  (T / 2 : ℚ) - (T / 2 : ℚ) / 5 = 400 → T = 1000 := by
  sorry

end NUMINAMATH_CALUDE_paper_cranes_count_l2314_231447


namespace NUMINAMATH_CALUDE_f_4_1981_l2314_231424

def f : ℕ → ℕ → ℕ 
  | 0, y => y + 1
  | x + 1, 0 => f x 1
  | x + 1, y + 1 => f x (f (x + 1) y)

theorem f_4_1981 : f 4 1981 = 2^(2^(2^(1981 + 1) + 1)) - 3 := by
  sorry

end NUMINAMATH_CALUDE_f_4_1981_l2314_231424


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l2314_231468

theorem solution_satisfies_system :
  let x : ℝ := 1
  let y : ℝ := 1/2
  let w : ℝ := -1/2
  let z : ℝ := 1/3
  (Real.sqrt x - 1/y - 2*w + 3*z = 1) ∧
  (x + 1/(y^2) - 4*(w^2) - 9*(z^2) = 3) ∧
  (x * Real.sqrt x - 1/(y^3) - 8*(w^3) + 27*(z^3) = -5) ∧
  (x^2 + 1/(y^4) - 16*(w^4) - 81*(z^4) = 15) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l2314_231468


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2314_231445

/-- The surface area of a cylinder with height 2 and base circumference 2π is 6π -/
theorem cylinder_surface_area :
  ∀ (h : ℝ) (c : ℝ),
  h = 2 →
  c = 2 * Real.pi →
  2 * Real.pi * (c / (2 * Real.pi)) * (c / (2 * Real.pi)) + c * h = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2314_231445


namespace NUMINAMATH_CALUDE_dumbbell_distribution_impossible_l2314_231450

def dumbbell_weights : List ℕ := [4, 5, 6, 9, 10, 11, 14, 19, 23, 24]

theorem dumbbell_distribution_impossible :
  ¬ ∃ (rack1 rack2 rack3 : List ℕ),
    (rack1 ++ rack2 ++ rack3).toFinset = dumbbell_weights.toFinset ∧
    (rack1.sum : ℚ) * 2 = rack2.sum ∧
    (rack2.sum : ℚ) * 2 = rack3.sum :=
by sorry

end NUMINAMATH_CALUDE_dumbbell_distribution_impossible_l2314_231450


namespace NUMINAMATH_CALUDE_johnson_work_completion_l2314_231449

/-- Johnson and Vincent's Work Completion Problem -/
theorem johnson_work_completion (vincent_days : ℕ) (together_days : ℕ) (johnson_days : ℕ) : 
  vincent_days = 40 → together_days = 8 → johnson_days = 10 →
  (1 : ℚ) / johnson_days + (1 : ℚ) / vincent_days = (1 : ℚ) / together_days := by
  sorry

#check johnson_work_completion

end NUMINAMATH_CALUDE_johnson_work_completion_l2314_231449


namespace NUMINAMATH_CALUDE_banana_theorem_l2314_231414

def banana_problem (initial_bananas final_bananas : ℕ) : Prop :=
  final_bananas - initial_bananas = 7

theorem banana_theorem : banana_problem 2 9 := by
  sorry

end NUMINAMATH_CALUDE_banana_theorem_l2314_231414


namespace NUMINAMATH_CALUDE_therapy_hours_is_five_l2314_231442

/-- Represents the cost structure and billing for a psychologist's therapy sessions. -/
structure TherapyCost where
  firstHourCost : ℕ
  additionalHourCost : ℕ
  firstHourPremium : ℕ
  twoHourTotal : ℕ
  someHoursTotal : ℕ

/-- Calculates the number of therapy hours given the cost structure and total charge. -/
def calculateTherapyHours (cost : TherapyCost) : ℕ :=
  sorry

/-- Theorem stating that given the specific cost structure, the calculated therapy hours is 5. -/
theorem therapy_hours_is_five (cost : TherapyCost)
  (h1 : cost.firstHourCost = cost.additionalHourCost + cost.firstHourPremium)
  (h2 : cost.firstHourPremium = 25)
  (h3 : cost.twoHourTotal = 115)
  (h4 : cost.someHoursTotal = 250) :
  calculateTherapyHours cost = 5 :=
sorry

end NUMINAMATH_CALUDE_therapy_hours_is_five_l2314_231442


namespace NUMINAMATH_CALUDE_factor_expression_l2314_231444

theorem factor_expression (b : ℝ) : 53 * b^2 + 159 * b = 53 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2314_231444


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2314_231489

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 - 7 * x - 6 = 0 :=
by
  -- The unique positive solution is x = 3
  use 3
  constructor
  · -- Prove that x = 3 satisfies the conditions
    constructor
    · -- Prove 3 > 0
      sorry
    · -- Prove 3 * 3^2 - 7 * 3 - 6 = 0
      sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2314_231489


namespace NUMINAMATH_CALUDE_fourth_root_difference_l2314_231481

theorem fourth_root_difference : (81 : ℝ) ^ (1/4) - (1296 : ℝ) ^ (1/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_difference_l2314_231481


namespace NUMINAMATH_CALUDE_employee_count_l2314_231458

theorem employee_count (total_profit : ℝ) (owner_percentage : ℝ) (employee_share : ℝ) : 
  total_profit = 50 →
  owner_percentage = 0.1 →
  employee_share = 5 →
  (1 - owner_percentage) * total_profit / employee_share = 9 := by
  sorry

end NUMINAMATH_CALUDE_employee_count_l2314_231458


namespace NUMINAMATH_CALUDE_complex_cube_root_l2314_231460

theorem complex_cube_root (a b : ℕ+) :
  (↑a + ↑b * Complex.I) ^ 3 = 2 + 11 * Complex.I →
  ↑a + ↑b * Complex.I = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l2314_231460


namespace NUMINAMATH_CALUDE_smallest_sticker_collection_l2314_231466

theorem smallest_sticker_collection (S : ℕ) : 
  S > 1 ∧ 
  S % 5 = 2 ∧ 
  S % 9 = 2 ∧ 
  S % 11 = 2 → 
  S ≥ 497 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sticker_collection_l2314_231466


namespace NUMINAMATH_CALUDE_fraction_value_l2314_231498

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 10) :
  (x + y) / (x - y) = -Real.sqrt (3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2314_231498


namespace NUMINAMATH_CALUDE_remainder_proof_l2314_231485

theorem remainder_proof (x y : ℕ+) (r : ℕ) 
  (h1 : x = 11 * y + 4)
  (h2 : 2 * x = 8 * (3 * y) + r)
  (h3 : 13 * y - x = 3) :
  r = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_proof_l2314_231485


namespace NUMINAMATH_CALUDE_complex_quadrant_range_l2314_231474

theorem complex_quadrant_range (z : ℂ) (a : ℝ) :
  z * (a + Complex.I) = 2 + 3 * Complex.I →
  (z.re * z.im < 0 ↔ -3/2 < a ∧ a < 2/3) :=
by sorry

end NUMINAMATH_CALUDE_complex_quadrant_range_l2314_231474


namespace NUMINAMATH_CALUDE_permutations_of_five_l2314_231497

theorem permutations_of_five (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_five_l2314_231497


namespace NUMINAMATH_CALUDE_position_relationships_complete_l2314_231434

-- Define the type for position relationships
inductive PositionRelationship
  | Intersection
  | Parallel
  | Skew

-- Define a type for straight lines in 3D space
structure Line3D where
  -- We don't need to specify the internal structure of Line3D for this statement

-- Define the function that determines the position relationship between two lines
noncomputable def positionRelationship (l1 l2 : Line3D) : PositionRelationship :=
  sorry

-- Theorem statement
theorem position_relationships_complete (l1 l2 : Line3D) :
  ∃ (r : PositionRelationship), positionRelationship l1 l2 = r :=
sorry

end NUMINAMATH_CALUDE_position_relationships_complete_l2314_231434


namespace NUMINAMATH_CALUDE_square_area_with_two_side_expressions_l2314_231499

theorem square_area_with_two_side_expressions (x : ℝ) :
  (5 * x + 10 = 35 - 2 * x) →
  ((5 * x + 10) ^ 2 : ℝ) = 38025 / 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_two_side_expressions_l2314_231499


namespace NUMINAMATH_CALUDE_permutation_sum_l2314_231491

theorem permutation_sum (n : ℕ+) (h1 : n + 3 ≤ 2*n) (h2 : n + 1 ≤ 4) : 
  (Nat.descFactorial (2*n) (n+3)) + (Nat.descFactorial 4 (n+1)) = 744 :=
by sorry

end NUMINAMATH_CALUDE_permutation_sum_l2314_231491


namespace NUMINAMATH_CALUDE_seating_theorem_l2314_231406

/-- The number of ways to arrange 5 boys and 4 girls in a row of 9 chairs such that at least 2 boys are next to each other -/
def seating_arrangements (num_boys num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating that the number of seating arrangements for 5 boys and 4 girls with at least 2 boys next to each other is 359000 -/
theorem seating_theorem :
  seating_arrangements 5 4 = 359000 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l2314_231406


namespace NUMINAMATH_CALUDE_equation_solution_l2314_231410

theorem equation_solution (y : ℚ) : (4 * y - 2) / (5 * y - 5) = 3 / 4 → y = -7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2314_231410


namespace NUMINAMATH_CALUDE_election_votes_difference_l2314_231473

theorem election_votes_difference (total_votes : ℕ) (winner_votes : ℕ) (second_votes : ℕ) (third_votes : ℕ) (fourth_votes : ℕ) 
  (h_total : total_votes = 979)
  (h_candidates : winner_votes + second_votes + third_votes + fourth_votes = total_votes)
  (h_winner_second : winner_votes = second_votes + 53)
  (h_winner_fourth : winner_votes = fourth_votes + 105)
  (h_fourth : fourth_votes = 199) :
  winner_votes - third_votes = 79 := by
sorry

end NUMINAMATH_CALUDE_election_votes_difference_l2314_231473


namespace NUMINAMATH_CALUDE_at_least_two_equations_have_solution_l2314_231475

theorem at_least_two_equations_have_solution (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  let f₁ : ℝ → ℝ := λ x ↦ (x - b) * (x - c) - (x - a)
  let f₂ : ℝ → ℝ := λ x ↦ (x - c) * (x - a) - (x - b)
  let f₃ : ℝ → ℝ := λ x ↦ (x - a) * (x - b) - (x - c)
  ∃ (i j : Fin 3), i ≠ j ∧ (∃ x : ℝ, [f₁, f₂, f₃][i] x = 0) ∧ (∃ y : ℝ, [f₁, f₂, f₃][j] y = 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_equations_have_solution_l2314_231475


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2314_231427

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (3 - 2 * Complex.I^3) / (1 + Complex.I)
  Complex.im z = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2314_231427


namespace NUMINAMATH_CALUDE_books_shop1_is_65_l2314_231416

-- Define the problem parameters
def total_spent_shop1 : ℕ := 6500
def books_shop2 : ℕ := 35
def total_spent_shop2 : ℕ := 2000
def avg_price : ℕ := 85

-- Define the function to calculate the number of books from the first shop
def books_shop1 : ℕ := 
  (total_spent_shop1 + total_spent_shop2) / avg_price - books_shop2

-- Theorem to prove
theorem books_shop1_is_65 : books_shop1 = 65 := by
  sorry

end NUMINAMATH_CALUDE_books_shop1_is_65_l2314_231416


namespace NUMINAMATH_CALUDE_point_five_units_from_origin_l2314_231441

theorem point_five_units_from_origin (a : ℝ) : |a| = 5 ↔ a = 5 ∨ a = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_five_units_from_origin_l2314_231441


namespace NUMINAMATH_CALUDE_total_beads_used_l2314_231477

def necklaces_monday : ℕ := 10
def necklaces_tuesday : ℕ := 2
def bracelets : ℕ := 5
def earrings : ℕ := 7
def beads_per_necklace : ℕ := 20
def beads_per_bracelet : ℕ := 10
def beads_per_earring : ℕ := 5

theorem total_beads_used :
  (necklaces_monday + necklaces_tuesday) * beads_per_necklace +
  bracelets * beads_per_bracelet +
  earrings * beads_per_earring = 325 := by
sorry

end NUMINAMATH_CALUDE_total_beads_used_l2314_231477


namespace NUMINAMATH_CALUDE_percent_commutation_l2314_231463

theorem percent_commutation (x : ℝ) (h : (25 / 100) * ((10 / 100) * x) = 15) :
  (10 / 100) * ((25 / 100) * x) = 15 := by
sorry

end NUMINAMATH_CALUDE_percent_commutation_l2314_231463


namespace NUMINAMATH_CALUDE_mower_blades_cost_l2314_231453

def total_earned : ℕ := 104
def num_games : ℕ := 7
def game_price : ℕ := 9

theorem mower_blades_cost (remaining : ℕ) 
  (h1 : remaining = num_games * game_price) 
  (h2 : remaining + (total_earned - remaining) = total_earned) : 
  total_earned - remaining = 41 := by
  sorry

end NUMINAMATH_CALUDE_mower_blades_cost_l2314_231453


namespace NUMINAMATH_CALUDE_james_tin_collection_l2314_231423

/-- The number of tins James collected on the first day -/
def first_day_tins : ℕ := sorry

/-- The total number of tins James collected in a week -/
def total_tins : ℕ := 500

/-- The number of tins James collected on each of the last four days -/
def last_four_days_tins : ℕ := 50

theorem james_tin_collection :
  first_day_tins = 50 ∧
  first_day_tins +
  (3 * first_day_tins) +
  (3 * first_day_tins - 50) +
  (4 * last_four_days_tins) = total_tins :=
sorry

end NUMINAMATH_CALUDE_james_tin_collection_l2314_231423


namespace NUMINAMATH_CALUDE_lcm_of_48_and_14_l2314_231451

theorem lcm_of_48_and_14 (n : ℕ) (h1 : n = 48) (h2 : Nat.gcd n 14 = 12) :
  Nat.lcm n 14 = 56 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_48_and_14_l2314_231451


namespace NUMINAMATH_CALUDE_blending_markers_count_l2314_231408

/-- Proof that the number of drawings made with blending markers is 7 -/
theorem blending_markers_count (total : ℕ) (colored_pencils : ℕ) (charcoal : ℕ) 
  (h1 : total = 25)
  (h2 : colored_pencils = 14)
  (h3 : charcoal = 4) :
  total - (colored_pencils + charcoal) = 7 := by
  sorry

end NUMINAMATH_CALUDE_blending_markers_count_l2314_231408


namespace NUMINAMATH_CALUDE_right_triangle_circle_theorem_l2314_231483

/-- A right triangle with a circle inscribed on one side --/
structure RightTriangleWithCircle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Point where the circle meets AC
  D : ℝ × ℝ
  -- B is a right angle
  right_angle_at_B : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
  -- BC is the diameter of the circle
  BC_is_diameter : ∃ (center : ℝ × ℝ), 
    (center.1 - B.1)^2 + (center.2 - B.2)^2 = (center.1 - C.1)^2 + (center.2 - C.2)^2
  -- D lies on AC
  D_on_AC : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t*(C.1 - A.1), A.2 + t*(C.2 - A.2))
  -- D lies on the circle
  D_on_circle : ∃ (center : ℝ × ℝ), 
    (center.1 - D.1)^2 + (center.2 - D.2)^2 = (center.1 - B.1)^2 + (center.2 - B.2)^2

/-- The theorem to be proved --/
theorem right_triangle_circle_theorem (t : RightTriangleWithCircle) 
  (h1 : Real.sqrt ((t.A.1 - t.D.1)^2 + (t.A.2 - t.D.2)^2) = 3)
  (h2 : Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2) = 6) :
  Real.sqrt ((t.C.1 - t.D.1)^2 + (t.C.2 - t.D.2)^2) = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_circle_theorem_l2314_231483


namespace NUMINAMATH_CALUDE_office_age_problem_l2314_231438

theorem office_age_problem (total_persons : ℕ) (avg_age_all : ℝ) 
  (group1_size : ℕ) (group2_size : ℕ) (avg_age_group2 : ℝ) 
  (age_person15 : ℕ) :
  total_persons = 18 →
  avg_age_all = 15 →
  group1_size = 5 →
  group2_size = 9 →
  avg_age_group2 = 16 →
  age_person15 = 56 →
  (total_persons * avg_age_all - group2_size * avg_age_group2 - age_person15) / group1_size = 14 := by
sorry

end NUMINAMATH_CALUDE_office_age_problem_l2314_231438


namespace NUMINAMATH_CALUDE_max_acute_angles_2000_sided_polygon_l2314_231443

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : ℕ
  convex : Bool
  sides_eq : sides = n

/-- The maximum number of acute angles in a convex polygon -/
def max_acute_angles (p : ConvexPolygon n) : ℕ :=
  sorry

/-- Theorem: The maximum number of acute angles in a convex 2000-sided polygon is 3 -/
theorem max_acute_angles_2000_sided_polygon :
  ∀ (p : ConvexPolygon 2000), max_acute_angles p = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_acute_angles_2000_sided_polygon_l2314_231443


namespace NUMINAMATH_CALUDE_range_of_a_l2314_231435

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2314_231435


namespace NUMINAMATH_CALUDE_inequality_theorem_l2314_231419

theorem inequality_theorem (a b : ℝ) (h : a < b) : -a - 1 > -b - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2314_231419


namespace NUMINAMATH_CALUDE_leastSquaresSolution_l2314_231465

-- Define the data points
def x : List ℝ := [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
def y : List ℝ := [6.01, 5.07, 4.30, 3.56, 3.07, 2.87, 2.18, 2.00, 2.14]

-- Define the quadratic model
def model (a₁ a₂ a₃ x : ℝ) : ℝ := a₁ * x^2 + a₂ * x + a₃

-- Define the sum of squared residuals
def sumSquaredResiduals (a₁ a₂ a₃ : ℝ) : ℝ :=
  List.sum (List.zipWith (λ xᵢ yᵢ => (yᵢ - model a₁ a₂ a₃ xᵢ)^2) x y)

-- State the theorem
theorem leastSquaresSolution :
  let a₁ : ℝ := 0.95586
  let a₂ : ℝ := -1.9733
  let a₃ : ℝ := 3.0684
  ∀ b₁ b₂ b₃ : ℝ, sumSquaredResiduals a₁ a₂ a₃ ≤ sumSquaredResiduals b₁ b₂ b₃ := by
  sorry

end NUMINAMATH_CALUDE_leastSquaresSolution_l2314_231465


namespace NUMINAMATH_CALUDE_tessa_apples_for_pie_l2314_231431

def apples_needed_for_pie (initial_apples : ℕ) (received_apples : ℕ) (required_apples : ℕ) : ℕ :=
  max (required_apples - (initial_apples + received_apples)) 0

theorem tessa_apples_for_pie :
  apples_needed_for_pie 4 5 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tessa_apples_for_pie_l2314_231431


namespace NUMINAMATH_CALUDE_garrison_provision_days_l2314_231413

/-- Calculates the initial number of days provisions were supposed to last for a garrison -/
def initialProvisionDays (initialGarrison : ℕ) (reinforcement : ℕ) (daysBeforeReinforcement : ℕ) (daysAfterReinforcement : ℕ) : ℕ :=
  ((initialGarrison + reinforcement) * daysAfterReinforcement + initialGarrison * daysBeforeReinforcement) / initialGarrison

theorem garrison_provision_days :
  initialProvisionDays 1000 1250 15 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_garrison_provision_days_l2314_231413


namespace NUMINAMATH_CALUDE_additional_grazing_area_l2314_231437

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 23^2 - π * 9^2 = 448 * π := by
  sorry

end NUMINAMATH_CALUDE_additional_grazing_area_l2314_231437


namespace NUMINAMATH_CALUDE_swimming_laps_per_day_l2314_231462

/-- Proves that swimming 300 laps in 5 weeks, 5 days per week, results in 12 laps per day -/
theorem swimming_laps_per_day 
  (total_laps : ℕ) 
  (weeks : ℕ) 
  (days_per_week : ℕ) 
  (h1 : total_laps = 300) 
  (h2 : weeks = 5) 
  (h3 : days_per_week = 5) : 
  total_laps / (weeks * days_per_week) = 12 := by
  sorry

end NUMINAMATH_CALUDE_swimming_laps_per_day_l2314_231462


namespace NUMINAMATH_CALUDE_cookies_milk_ratio_l2314_231407

/-- Proof that 5 cookies require 20/3 pints of milk given the established ratio and conversion rates -/
theorem cookies_milk_ratio (cookies_base : ℕ) (milk_base : ℕ) (cookies_target : ℕ) :
  cookies_base = 18 →
  milk_base = 3 →
  cookies_target = 5 →
  (milk_base * 4 * 2 : ℚ) / cookies_base * cookies_target = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_milk_ratio_l2314_231407


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l2314_231440

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def num1 : List Bool := [true, false, true, true]  -- 1101₂
def num2 : List Bool := [true, false, true]        -- 101₂
def num3 : List Bool := [false, true, true, true]  -- 1110₂
def num4 : List Bool := [true, true, true]         -- 111₂
def num5 : List Bool := [false, true, false, true] -- 1010₂
def result : List Bool := [true, false, true, false, true] -- 10101₂

theorem binary_sum_theorem :
  binary_to_nat num1 + binary_to_nat num2 + binary_to_nat num3 +
  binary_to_nat num4 + binary_to_nat num5 = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_theorem_l2314_231440


namespace NUMINAMATH_CALUDE_sharas_age_l2314_231455

theorem sharas_age (jaymee_age shara_age : ℕ) : 
  jaymee_age = 22 →
  jaymee_age = 2 * shara_age + 2 →
  shara_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_sharas_age_l2314_231455


namespace NUMINAMATH_CALUDE_initial_men_fraction_l2314_231478

theorem initial_men_fraction (initial_total : ℕ) (new_hires : ℕ) (final_women_percentage : ℚ) : 
  initial_total = 90 → 
  new_hires = 10 → 
  final_women_percentage = 2/5 → 
  (initial_total - (final_women_percentage * (initial_total + new_hires)).num) / initial_total = 2/3 := by
sorry

end NUMINAMATH_CALUDE_initial_men_fraction_l2314_231478


namespace NUMINAMATH_CALUDE_divisibility_of_20_pow_15_minus_1_l2314_231493

theorem divisibility_of_20_pow_15_minus_1 :
  (11 : ℕ) * 31 * 61 ∣ 20^15 - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_20_pow_15_minus_1_l2314_231493


namespace NUMINAMATH_CALUDE_perfect_square_identification_l2314_231454

theorem perfect_square_identification :
  let a := 3^6 * 7^7 * 8^8
  let b := 3^8 * 7^6 * 8^7
  let c := 3^7 * 7^8 * 8^6
  let d := 3^7 * 7^7 * 8^8
  let e := 3^8 * 7^8 * 8^8
  ∃ n : ℕ, e = n^2 ∧ 
  (∀ m : ℕ, a ≠ m^2) ∧ 
  (∀ m : ℕ, b ≠ m^2) ∧ 
  (∀ m : ℕ, c ≠ m^2) ∧ 
  (∀ m : ℕ, d ≠ m^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_identification_l2314_231454


namespace NUMINAMATH_CALUDE_second_next_perfect_square_l2314_231404

theorem second_next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k ^ 2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m ^ 2) ∧
  (∀ y : ℕ, y > x ∧ (∃ l : ℕ, y = l ^ 2) → y ≥ n) ∧
  n = x + 4 * Int.sqrt x + 4 :=
sorry

end NUMINAMATH_CALUDE_second_next_perfect_square_l2314_231404


namespace NUMINAMATH_CALUDE_elijah_card_count_l2314_231426

/-- The number of cards in a standard deck of playing cards -/
def cards_per_deck : ℕ := 52

/-- The number of decks Elijah has -/
def number_of_decks : ℕ := 6

/-- The total number of cards Elijah has -/
def total_cards : ℕ := number_of_decks * cards_per_deck

theorem elijah_card_count : total_cards = 312 := by
  sorry

end NUMINAMATH_CALUDE_elijah_card_count_l2314_231426


namespace NUMINAMATH_CALUDE_arcsin_sin_2x_solutions_l2314_231469

theorem arcsin_sin_2x_solutions (x : Real) :
  x ∈ Set.Icc (-π/2) (π/2) ∧ Real.arcsin (Real.sin (2*x)) = x ↔ x = 0 ∨ x = -π/3 ∨ x = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sin_2x_solutions_l2314_231469


namespace NUMINAMATH_CALUDE_arithmetic_proof_l2314_231425

theorem arithmetic_proof : 4 * (8 - 3) - 2 * 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l2314_231425


namespace NUMINAMATH_CALUDE_final_price_percentage_l2314_231403

-- Define the discounts and tax rate
def discount1 : ℝ := 0.5
def discount2 : ℝ := 0.1
def discount3 : ℝ := 0.2
def taxRate : ℝ := 0.08

-- Define the function to calculate the final price
def finalPrice (originalPrice : ℝ) : ℝ :=
  let price1 := originalPrice * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let price3 := price2 * (1 - discount3)
  price3 * (1 + taxRate)

-- Theorem statement
theorem final_price_percentage (originalPrice : ℝ) (originalPrice_pos : originalPrice > 0) :
  finalPrice originalPrice / originalPrice = 0.3888 := by
  sorry

end NUMINAMATH_CALUDE_final_price_percentage_l2314_231403


namespace NUMINAMATH_CALUDE_farm_work_hourly_rate_l2314_231479

theorem farm_work_hourly_rate 
  (total_amount : ℕ) 
  (tips : ℕ) 
  (hours_worked : ℕ) 
  (h1 : total_amount = 240)
  (h2 : tips = 50)
  (h3 : hours_worked = 19) :
  (total_amount - tips) / hours_worked = 10 := by
sorry

end NUMINAMATH_CALUDE_farm_work_hourly_rate_l2314_231479


namespace NUMINAMATH_CALUDE_unique_number_property_l2314_231488

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l2314_231488


namespace NUMINAMATH_CALUDE_investment_average_rate_l2314_231494

def total_investment : ℝ := 5000
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

theorem investment_average_rate :
  ∃ (x y : ℝ),
    x + y = total_investment ∧
    x * rate1 = y * rate2 / 2 ∧
    (x * rate1 + y * rate2) / total_investment = 0.041 :=
by sorry

end NUMINAMATH_CALUDE_investment_average_rate_l2314_231494


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l2314_231428

def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 4*x - 6*y

theorem circle_passes_through_points :
  (circle_equation 0 0 = 0) ∧
  (circle_equation 4 0 = 0) ∧
  (circle_equation (-1) 1 = 0) :=
by sorry

#check circle_passes_through_points

end NUMINAMATH_CALUDE_circle_passes_through_points_l2314_231428


namespace NUMINAMATH_CALUDE_sin_cos_product_trig_expression_value_l2314_231415

-- Part I
theorem sin_cos_product (α : ℝ) 
  (h : (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7) :
  Real.sin α * Real.cos α = 3 / 10 := by
  sorry

-- Part II
theorem trig_expression_value :
  (Real.sqrt (1 - 2 * Real.sin (10 * π / 180) * Real.cos (10 * π / 180))) / 
  (Real.cos (10 * π / 180) - Real.sqrt (1 - Real.cos (170 * π / 180)^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_trig_expression_value_l2314_231415


namespace NUMINAMATH_CALUDE_audio_cassette_count_audio_cassette_count_proof_l2314_231430

/-- Proves that the number of audio cassettes in the first set is 7 given the problem conditions --/
theorem audio_cassette_count : ℕ :=
  let video_cost : ℕ := 300
  let some_audio_and_3_video_cost : ℕ := 1110
  let five_audio_and_4_video_cost : ℕ := 1350
  7

theorem audio_cassette_count_proof :
  let video_cost : ℕ := 300
  let some_audio_and_3_video_cost : ℕ := 1110
  let five_audio_and_4_video_cost : ℕ := 1350
  ∃ (audio_cost : ℕ) (first_set_count : ℕ),
    first_set_count * audio_cost + 3 * video_cost = some_audio_and_3_video_cost ∧
    5 * audio_cost + 4 * video_cost = five_audio_and_4_video_cost ∧
    first_set_count = audio_cassette_count :=
by
  sorry

end NUMINAMATH_CALUDE_audio_cassette_count_audio_cassette_count_proof_l2314_231430


namespace NUMINAMATH_CALUDE_quadratic_roots_l2314_231400

theorem quadratic_roots (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2314_231400


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l2314_231480

-- Define the triangle ABC
theorem triangle_abc_theorem (a b c A B C : ℝ) 
  (h1 : a / Real.tan A = b / (2 * Real.sin B))
  (h2 : a = 6)
  (h3 : b = 2 * c)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π) : 
  A = π / 3 ∧ 
  (1/2 * b * c * Real.sin A : ℝ) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l2314_231480


namespace NUMINAMATH_CALUDE_angle_between_lines_l2314_231405

def line1 (x : ℝ) : ℝ := -2 * x
def line2 (x : ℝ) : ℝ := 3 * x + 5

theorem angle_between_lines :
  let k1 := -2
  let k2 := 3
  let tan_phi := abs ((k2 - k1) / (1 + k1 * k2))
  Real.arctan tan_phi * (180 / Real.pi) = 45 :=
sorry

end NUMINAMATH_CALUDE_angle_between_lines_l2314_231405


namespace NUMINAMATH_CALUDE_number_relationships_l2314_231492

theorem number_relationships : 
  (100000000 = 10 * 10000000) ∧ (1000000 = 100 * 10000) := by
  sorry

#check number_relationships

end NUMINAMATH_CALUDE_number_relationships_l2314_231492


namespace NUMINAMATH_CALUDE_age_problem_l2314_231417

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 27 → 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2314_231417


namespace NUMINAMATH_CALUDE_bruce_mangoes_purchase_l2314_231471

/-- The amount of grapes purchased in kg -/
def grapes_kg : ℕ := 8

/-- The price of grapes per kg -/
def grapes_price : ℕ := 70

/-- The price of mangoes per kg -/
def mangoes_price : ℕ := 55

/-- The total amount paid -/
def total_paid : ℕ := 1110

/-- The amount of mangoes purchased in kg -/
def mangoes_kg : ℕ := (total_paid - grapes_kg * grapes_price) / mangoes_price

theorem bruce_mangoes_purchase :
  mangoes_kg = 10 := by sorry

end NUMINAMATH_CALUDE_bruce_mangoes_purchase_l2314_231471


namespace NUMINAMATH_CALUDE_correct_statements_reflect_relationship_l2314_231486

-- Define the statements
inductive Statement
| WaitingForRabbit
| GoodThingsThroughHardship
| PreventMinorIssues
| Insignificant

-- Define the philosophical principles
structure PhilosophicalPrinciple where
  name : String
  description : String

-- Define the relationship between quantitative and qualitative change
def reflectsQuantQualRelationship (s : Statement) (p : PhilosophicalPrinciple) : Prop :=
  match s with
  | Statement.GoodThingsThroughHardship => p.name = "Accumulation"
  | Statement.PreventMinorIssues => p.name = "Moderation"
  | _ => False

-- Theorem statement
theorem correct_statements_reflect_relationship :
  ∃ (p1 p2 : PhilosophicalPrinciple),
    reflectsQuantQualRelationship Statement.GoodThingsThroughHardship p1 ∧
    reflectsQuantQualRelationship Statement.PreventMinorIssues p2 :=
  sorry

end NUMINAMATH_CALUDE_correct_statements_reflect_relationship_l2314_231486


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2314_231490

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℝ, 
    5 * X^6 + 3 * X^4 - 2 * X^3 + 7 * X^2 + 4 = 
    (X^2 + 2 * X + 1) * q + (-38 * X - 29) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2314_231490


namespace NUMINAMATH_CALUDE_anil_tomato_production_l2314_231452

/-- Represents the number of tomatoes in a square backyard -/
def TomatoCount (side : ℕ) : ℕ := side * side

/-- Proves that given the conditions of Anil's tomato garden, he produced 4356 tomatoes this year -/
theorem anil_tomato_production : 
  ∃ (last_year current_year : ℕ),
    TomatoCount current_year = TomatoCount last_year + 131 ∧
    current_year > last_year ∧
    TomatoCount current_year = 4356 := by
  sorry


end NUMINAMATH_CALUDE_anil_tomato_production_l2314_231452


namespace NUMINAMATH_CALUDE_expression_value_l2314_231470

theorem expression_value : (4 * 4 + 4) / (2 * 2 - 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2314_231470


namespace NUMINAMATH_CALUDE_cost_per_square_foot_calculation_l2314_231421

/-- Calculates the cost per square foot of a rented house. -/
theorem cost_per_square_foot_calculation 
  (master_bedroom_bath_area : ℝ)
  (guest_bedroom_area : ℝ)
  (num_guest_bedrooms : ℕ)
  (kitchen_bath_living_area : ℝ)
  (monthly_rent : ℝ)
  (h1 : master_bedroom_bath_area = 500)
  (h2 : guest_bedroom_area = 200)
  (h3 : num_guest_bedrooms = 2)
  (h4 : kitchen_bath_living_area = 600)
  (h5 : monthly_rent = 3000) :
  monthly_rent / (master_bedroom_bath_area + num_guest_bedrooms * guest_bedroom_area + kitchen_bath_living_area) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_per_square_foot_calculation_l2314_231421


namespace NUMINAMATH_CALUDE_solve_inequality_find_range_of_a_l2314_231411

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solve_inequality (x : ℝ) :
  f x (-1) ≥ 3 ↔ x ≤ -3/2 ∨ x ≥ 3/2 :=
sorry

-- Part 2
theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 2) ↔ a ≤ -1 ∨ a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_solve_inequality_find_range_of_a_l2314_231411


namespace NUMINAMATH_CALUDE_smallest_x_for_cube_l2314_231418

theorem smallest_x_for_cube (x M : ℕ+) : 
  (∀ y : ℕ+, y < x → ¬∃ N : ℕ+, 720 * y = N^3) → 
  (∃ N : ℕ+, 720 * x = N^3) → 
  x = 300 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_cube_l2314_231418


namespace NUMINAMATH_CALUDE_square_difference_equality_l2314_231461

theorem square_difference_equality : 1012^2 - 1008^2 - 1006^2 + 1002^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2314_231461


namespace NUMINAMATH_CALUDE_max_cylinder_lateral_area_l2314_231487

/-- Given a rectangle with perimeter 36, prove that when rotated around one of its edges
    to form a cylinder, the maximum lateral surface area of the cylinder is 81. -/
theorem max_cylinder_lateral_area (l w : ℝ) : 
  (l + w = 18) →  -- Perimeter condition: 2(l + w) = 36, simplified to l + w = 18
  (∃ (h r : ℝ), h = w ∧ 2 * π * r = l ∧ 2 * π * r * h ≤ 81) ∧ 
  (∃ (h r : ℝ), h = w ∧ 2 * π * r = l ∧ 2 * π * r * h = 81) :=
sorry

end NUMINAMATH_CALUDE_max_cylinder_lateral_area_l2314_231487


namespace NUMINAMATH_CALUDE_last_boat_occupancy_l2314_231467

/-- The number of tourists in the travel group -/
def total_tourists (x : ℕ) : ℕ := 8 * x + 6

/-- The number of people that can be seated in (x-2) fully occupied 12-seat boats -/
def seated_tourists (x : ℕ) : ℕ := 12 * (x - 2)

theorem last_boat_occupancy (x : ℕ) (h : x > 2) :
  total_tourists x - seated_tourists x = 30 - 4 * x :=
by sorry

end NUMINAMATH_CALUDE_last_boat_occupancy_l2314_231467


namespace NUMINAMATH_CALUDE_no_integer_solution_l2314_231482

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2314_231482


namespace NUMINAMATH_CALUDE_soy_sauce_bottles_l2314_231439

/-- Represents the amount of soy sauce in ounces -/
def OuncesPerBottle : ℕ := 16

/-- Represents the number of ounces in a cup -/
def OuncesPerCup : ℕ := 8

/-- Represents the amount of soy sauce needed for each recipe in cups -/
def RecipeCups : List ℕ := [2, 1, 3]

/-- Calculates the total number of cups needed for all recipes -/
def TotalCups : ℕ := RecipeCups.sum

/-- Calculates the total number of ounces needed for all recipes -/
def TotalOunces : ℕ := TotalCups * OuncesPerCup

/-- Calculates the number of bottles needed, rounding up to the nearest whole number -/
def BottlesNeeded : ℕ := (TotalOunces + OuncesPerBottle - 1) / OuncesPerBottle

theorem soy_sauce_bottles : BottlesNeeded = 3 := by sorry

end NUMINAMATH_CALUDE_soy_sauce_bottles_l2314_231439


namespace NUMINAMATH_CALUDE_equation_solutions_l2314_231422

theorem equation_solutions :
  (∀ x, x * (x - 6) = 2 * (x - 8) ↔ x = 4) ∧
  (∀ x, (2 * x - 1)^2 + 3 * (2 * x - 1) + 2 = 0 ↔ x = 0 ∨ x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2314_231422


namespace NUMINAMATH_CALUDE_range_of_m_l2314_231429

-- Define the set A as the solution set of |x+m| ≤ 4
def A (m : ℝ) : Set ℝ := {x : ℝ | |x + m| ≤ 4}

-- Define the theorem
theorem range_of_m :
  (∀ m : ℝ, A m ⊆ {x : ℝ | -2 ≤ x ∧ x ≤ 8}) →
  {m : ℝ | ∃ x : ℝ, x ∈ A m} = {m : ℝ | -4 ≤ m ∧ m ≤ -2} :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2314_231429


namespace NUMINAMATH_CALUDE_distinct_tower_heights_94_bricks_l2314_231433

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of distinct tower heights possible -/
def distinctTowerHeights (brickCount : ℕ) (dimensions : BrickDimensions) : ℕ :=
  let maxY := 4
  List.range (maxY + 1)
    |> List.map (fun y => brickCount - y + 1)
    |> List.sum

/-- Theorem stating the number of distinct tower heights -/
theorem distinct_tower_heights_94_bricks :
  let brickDimensions : BrickDimensions := ⟨4, 10, 19⟩
  distinctTowerHeights 94 brickDimensions = 465 := by
  sorry

#eval distinctTowerHeights 94 ⟨4, 10, 19⟩

end NUMINAMATH_CALUDE_distinct_tower_heights_94_bricks_l2314_231433


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l2314_231476

theorem quadratic_roots_difference (R : ℝ) : 
  (∃ α β : ℝ, α^2 - 2*α - R = 0 ∧ β^2 - 2*β - R = 0 ∧ α - β = 12) → R = 35 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l2314_231476
