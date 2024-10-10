import Mathlib

namespace range_of_function_l3624_362447

theorem range_of_function : 
  ∀ (x : ℝ), 12 ≤ |x + 5| - |x - 3| + 4 ∧ 
  (∃ (x₁ x₂ : ℝ), |x₁ + 5| - |x₁ - 3| + 4 = 12 ∧ |x₂ + 5| - |x₂ - 3| + 4 = 18) ∧
  (∀ (y : ℝ), (∃ (x : ℝ), y = |x + 5| - |x - 3| + 4) → 12 ≤ y ∧ y ≤ 18) :=
by sorry

end range_of_function_l3624_362447


namespace select_four_boots_from_five_pairs_l3624_362440

/-- The number of ways to select 4 boots from 5 pairs, including exactly one pair -/
def select_boots (n : ℕ) : ℕ :=
  let total_pairs := 5
  let pairs_to_choose := 1
  let remaining_pairs := total_pairs - pairs_to_choose
  let boots_to_choose := n - 2 * pairs_to_choose
  (total_pairs.choose pairs_to_choose) * 
  (remaining_pairs.choose (boots_to_choose / 2)) * 
  2^(boots_to_choose)

/-- Theorem stating that there are 120 ways to select 4 boots from 5 pairs, including exactly one pair -/
theorem select_four_boots_from_five_pairs : select_boots 4 = 120 := by
  sorry

end select_four_boots_from_five_pairs_l3624_362440


namespace function_upper_bound_l3624_362459

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, x ≥ 0 → ∃ (fx : ℝ), f x = fx) ∧  -- f is defined for x ≥ 0
  (∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)) ∧
  (∃ M : ℝ, M > 0 ∧ ∀ x, 0 ≤ x → x ≤ 1 → |f x| ≤ M)

/-- The main theorem -/
theorem function_upper_bound (f : ℝ → ℝ) (h : satisfies_conditions f) :
  ∀ x, x ≥ 0 → f x ≤ x^2 := by
  sorry

end function_upper_bound_l3624_362459


namespace abigail_saving_period_l3624_362412

def saving_period (monthly_saving : ℕ) (total_saved : ℕ) : ℕ :=
  total_saved / monthly_saving

theorem abigail_saving_period :
  let monthly_saving : ℕ := 4000
  let total_saved : ℕ := 48000
  saving_period monthly_saving total_saved = 12 := by
  sorry

end abigail_saving_period_l3624_362412


namespace parabola_range_l3624_362487

/-- The parabola y = x^2 + 2x + 4 -/
def parabola (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem parabola_range :
  ∀ a b : ℝ, -2 ≤ a → a < 3 → b = parabola a → 3 ≤ b ∧ b < 19 :=
by sorry

end parabola_range_l3624_362487


namespace max_value_sum_l3624_362475

theorem max_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt (3 * b^2) = Real.sqrt ((1 - a) * (1 + a))) : 
  ∃ (x : ℝ), x = a + Real.sqrt (3 * b^2) ∧ x ≤ Real.sqrt 2 ∧ 
  ∀ (y : ℝ), y = a + Real.sqrt (3 * b^2) → y ≤ x := by
  sorry

end max_value_sum_l3624_362475


namespace kays_total_exercise_time_l3624_362477

/-- Kay's weekly exercise routine -/
structure ExerciseRoutine where
  aerobics : ℕ
  weightTraining : ℕ

/-- The total exercise time is the sum of aerobics and weight training times -/
def totalExerciseTime (routine : ExerciseRoutine) : ℕ :=
  routine.aerobics + routine.weightTraining

/-- Kay's actual exercise routine -/
def kaysRoutine : ExerciseRoutine :=
  { aerobics := 150, weightTraining := 100 }

/-- Theorem: Kay's total exercise time is 250 minutes per week -/
theorem kays_total_exercise_time :
  totalExerciseTime kaysRoutine = 250 := by
  sorry

end kays_total_exercise_time_l3624_362477


namespace polygon_sides_count_l3624_362451

theorem polygon_sides_count : ∃ n : ℕ, 
  n > 2 ∧ 
  (n * (n - 3)) / 2 = 2 * n ∧ 
  ∀ m : ℕ, m > 2 → (m * (m - 3)) / 2 = 2 * m → m = n :=
by sorry

end polygon_sides_count_l3624_362451


namespace tom_age_ratio_l3624_362484

/-- Tom's current age -/
def T : ℕ := sorry

/-- Number of years ago mentioned in the second condition -/
def N : ℕ := 5

/-- Sum of the current ages of Tom's three children -/
def children_sum : ℕ := T / 2

/-- Tom's age N years ago -/
def tom_age_N_years_ago : ℕ := T - N

/-- Sum of the ages of Tom's children N years ago -/
def children_sum_N_years_ago : ℕ := children_sum - 3 * N

/-- The theorem stating the ratio of T to N -/
theorem tom_age_ratio : T / N = 16 := by
  sorry

end tom_age_ratio_l3624_362484


namespace hexagonal_pyramid_not_regular_l3624_362427

/-- A pyramid with a regular polygon base and all edges of equal length -/
structure RegularPyramid (n : ℕ) where
  /-- The number of sides of the base polygon -/
  base_sides : n > 2
  /-- The length of each edge of the pyramid -/
  edge_length : ℝ
  /-- The edge length is positive -/
  edge_positive : edge_length > 0

/-- Theorem stating that a hexagonal pyramid cannot have all edges of equal length -/
theorem hexagonal_pyramid_not_regular : ¬∃ (p : RegularPyramid 6), True :=
sorry

end hexagonal_pyramid_not_regular_l3624_362427


namespace find_divisor_l3624_362448

theorem find_divisor (d : ℕ) (h1 : d > 0) (h2 : 1050 % d = 0) (h3 : 1049 % d ≠ 0) : d = 1050 := by
  sorry

end find_divisor_l3624_362448


namespace mean_not_imply_concentration_stability_range_imply_concentration_stability_std_dev_imply_concentration_stability_variance_imply_concentration_stability_l3624_362491

/- Define a dataset as a list of real numbers -/
def Dataset := List ℝ

/- Define statistical measures -/
def range (data : Dataset) : ℝ := sorry
def mean (data : Dataset) : ℝ := sorry
def standardDeviation (data : Dataset) : ℝ := sorry
def variance (data : Dataset) : ℝ := sorry

/- Define a measure of concentration and stability -/
def isConcentratedAndStable (data : Dataset) : Prop := sorry

/- Theorem stating that mean does not imply concentration and stability -/
theorem mean_not_imply_concentration_stability :
  ∃ (data1 data2 : Dataset), mean data1 < mean data2 ∧
    (isConcentratedAndStable data1 ↔ ¬isConcentratedAndStable data2) := by sorry

/- Theorems stating that other measures imply concentration and stability -/
theorem range_imply_concentration_stability :
  ∀ (data1 data2 : Dataset), range data1 < range data2 →
    (isConcentratedAndStable data1 → isConcentratedAndStable data2) := by sorry

theorem std_dev_imply_concentration_stability :
  ∀ (data1 data2 : Dataset), standardDeviation data1 < standardDeviation data2 →
    (isConcentratedAndStable data1 → isConcentratedAndStable data2) := by sorry

theorem variance_imply_concentration_stability :
  ∀ (data1 data2 : Dataset), variance data1 < variance data2 →
    (isConcentratedAndStable data1 → isConcentratedAndStable data2) := by sorry

end mean_not_imply_concentration_stability_range_imply_concentration_stability_std_dev_imply_concentration_stability_variance_imply_concentration_stability_l3624_362491


namespace log_ratio_equality_l3624_362426

theorem log_ratio_equality : (Real.log 2 / Real.log 3) / (Real.log 8 / Real.log 9) = 2 / 3 := by
  sorry

end log_ratio_equality_l3624_362426


namespace div_2880_by_smallest_is_square_smaller_divisors_not_square_l3624_362450

/-- The smallest positive integer that divides 2880 and results in a perfect square -/
def smallest_divisor_to_square : ℕ := 10

/-- 2880 divided by the smallest divisor is a perfect square -/
theorem div_2880_by_smallest_is_square :
  ∃ m : ℕ, 2880 / smallest_divisor_to_square = m ^ 2 :=
sorry

/-- For any positive integer smaller than the smallest divisor, 
    dividing 2880 by it does not result in a perfect square -/
theorem smaller_divisors_not_square :
  ∀ k : ℕ, 0 < k → k < smallest_divisor_to_square →
  ¬∃ m : ℕ, 2880 / k = m ^ 2 :=
sorry

end div_2880_by_smallest_is_square_smaller_divisors_not_square_l3624_362450


namespace angle_negative_1445_quadrant_l3624_362435

theorem angle_negative_1445_quadrant : 
  ∃ (k : ℤ) (θ : ℝ), -1445 = 360 * k + θ ∧ 270 < θ ∧ θ ≤ 360 :=
sorry

end angle_negative_1445_quadrant_l3624_362435


namespace dream_number_k_value_l3624_362444

def is_dream_number (p : ℕ) : Prop :=
  p ≥ 100 ∧ p < 1000 ∧
  let h := p / 100
  let t := (p / 10) % 10
  let u := p % 10
  h ≠ 0 ∧ t ≠ 0 ∧ u ≠ 0 ∧
  (h - t : ℤ) = (t - u : ℤ)

def m (p : ℕ) : ℕ :=
  let h := p / 100
  let t := (p / 10) % 10
  let u := p % 10
  (10 * h + t) + (10 * t + u)

def n (p : ℕ) : ℕ :=
  let h := p / 100
  let u := p % 10
  (10 * h + u) + (10 * u + h)

def F (p : ℕ) : ℚ :=
  (m p - n p : ℚ) / 9

def s (x y : ℕ) : ℕ := 10 * x + y + 502

def t (a b : ℕ) : ℕ := 10 * a + b + 200

theorem dream_number_k_value
  (x y a b : ℕ)
  (hx : 1 ≤ x ∧ x ≤ 9)
  (hy : 1 ≤ y ∧ y ≤ 7)
  (ha : 1 ≤ a ∧ a ≤ 9)
  (hb : 1 ≤ b ∧ b ≤ 9)
  (hs : is_dream_number (s x y))
  (ht : is_dream_number (t a b))
  (h_eq : 2 * F (s x y) + F (t a b) = -1)
  : F (s x y) / F (s x y) = -3 := by
  sorry

end dream_number_k_value_l3624_362444


namespace vertex_of_quadratic_l3624_362464

/-- The x-coordinate of the vertex of a quadratic function f(x) = x^2 + 2px + 3q -/
def vertex_x_coord (p q : ℝ) : ℝ := -p

/-- The quadratic function f(x) = x^2 + 2px + 3q -/
def f (p q x : ℝ) : ℝ := x^2 + 2*p*x + 3*q

theorem vertex_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∀ x : ℝ, f p q x ≥ f p q (vertex_x_coord p q) :=
sorry

end vertex_of_quadratic_l3624_362464


namespace largest_prime_factor_of_1337_l3624_362497

theorem largest_prime_factor_of_1337 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 1337 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 1337 → q ≤ p := by
  sorry

end largest_prime_factor_of_1337_l3624_362497


namespace topological_subgraph_is_subgraph_max_degree_3_topological_subgraph_iff_subgraph_l3624_362423

/-- A graph. -/
structure Graph (V : Type) where
  edge : V → V → Prop

/-- The maximum degree of a graph. -/
def maxDegree {V : Type} (G : Graph V) : ℕ := sorry

/-- A subgraph relation between graphs. -/
def isSubgraph {V : Type} (H G : Graph V) : Prop := sorry

/-- A topological subgraph relation between graphs. -/
def isTopologicalSubgraph {V : Type} (H G : Graph V) : Prop := sorry

theorem topological_subgraph_is_subgraph {V : Type} (G H : Graph V) :
  isTopologicalSubgraph H G → isSubgraph H G := by sorry

theorem max_degree_3_topological_subgraph_iff_subgraph {V : Type} (G H : Graph V) :
  maxDegree G ≤ 3 →
  (isTopologicalSubgraph H G ↔ isSubgraph H G) := by sorry

end topological_subgraph_is_subgraph_max_degree_3_topological_subgraph_iff_subgraph_l3624_362423


namespace books_not_shared_l3624_362472

/-- The number of books that are in either Emily's or Olivia's collection, but not both -/
def books_in_either_not_both (shared_books : ℕ) (emily_total : ℕ) (olivia_unique : ℕ) : ℕ :=
  (emily_total - shared_books) + olivia_unique

/-- Theorem stating the number of books in either Emily's or Olivia's collection, but not both -/
theorem books_not_shared (shared_books : ℕ) (emily_total : ℕ) (olivia_unique : ℕ) 
  (h1 : shared_books = 15)
  (h2 : emily_total = 23)
  (h3 : olivia_unique = 8) :
  books_in_either_not_both shared_books emily_total olivia_unique = 16 := by
  sorry

end books_not_shared_l3624_362472


namespace boat_journey_distance_l3624_362442

def boat_journey (total_time : ℝ) (stream_velocity : ℝ) (boat_speed : ℝ) : Prop :=
  let downstream_speed : ℝ := boat_speed + stream_velocity
  let upstream_speed : ℝ := boat_speed - stream_velocity
  let distance : ℝ := 180
  (distance / downstream_speed + (distance / 2) / upstream_speed = total_time) ∧
  (downstream_speed > 0) ∧
  (upstream_speed > 0)

theorem boat_journey_distance :
  boat_journey 19 4 14 := by sorry

end boat_journey_distance_l3624_362442


namespace baron_munchausen_theorem_l3624_362421

theorem baron_munchausen_theorem :
  ∀ (a b : ℕ+), ∃ (n : ℕ+), ∃ (k m : ℕ+), (a * n = k ^ 2) ∧ (b * n = m ^ 3) := by
  sorry

end baron_munchausen_theorem_l3624_362421


namespace painting_area_is_5400_l3624_362499

/-- The area of a painting inside a uniform frame -/
def painting_area (outer_width : ℝ) (outer_height : ℝ) (frame_width : ℝ) : ℝ :=
  (outer_width - 2 * frame_width) * (outer_height - 2 * frame_width)

/-- Theorem: The area of the painting inside the frame is 5400 cm² -/
theorem painting_area_is_5400 :
  painting_area 90 120 15 = 5400 := by
  sorry

end painting_area_is_5400_l3624_362499


namespace fraction_value_l3624_362455

theorem fraction_value (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (1 / x + 1 / y) / (1 / x - 1 / y) = 7 := by
  sorry

end fraction_value_l3624_362455


namespace rectangular_garden_area_l3624_362457

theorem rectangular_garden_area (perimeter width length : ℝ) : 
  perimeter = 72 →
  length = 3 * width →
  2 * length + 2 * width = perimeter →
  length * width = 243 := by
  sorry

end rectangular_garden_area_l3624_362457


namespace fifth_term_is_eight_l3624_362443

/-- Represents a geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  ratio : ∀ n, a (n + 1) = 2 * a n

/-- Theorem: In a geometric sequence with common ratio 2 and a₂a₆ = 16, a₅ = 8 -/
theorem fifth_term_is_eight (seq : GeometricSequence) 
    (h : seq.a 2 * seq.a 6 = 16) : seq.a 5 = 8 := by
  sorry

#check fifth_term_is_eight

end fifth_term_is_eight_l3624_362443


namespace cylinder_volume_equality_l3624_362465

theorem cylinder_volume_equality (r h x : ℝ) : 
  r = 5 ∧ h = 7 ∧ x > 0 ∧ 
  π * (2 * r + x)^2 * h = π * r^2 * (3 * h + x) → 
  x = (5 + Real.sqrt 9125) / 14 := by sorry

end cylinder_volume_equality_l3624_362465


namespace equation_solution_exists_l3624_362434

theorem equation_solution_exists : ∃ x : ℝ, 
  x * 3967 + 36990 - 204790 / 19852 = 322299 ∧ 
  abs (x - 71.924) < 0.001 := by
  sorry

end equation_solution_exists_l3624_362434


namespace cafeteria_pies_problem_l3624_362493

def cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies_problem :
  cafeteria_pies 75 19 8 = 7 := by
  sorry

end cafeteria_pies_problem_l3624_362493


namespace algebraic_identity_l3624_362433

theorem algebraic_identity (a b : ℝ) : 2 * a * b - a^2 - b^2 = -((a - b)^2) := by
  sorry

end algebraic_identity_l3624_362433


namespace refrigerator_installment_l3624_362428

/-- Calculates the monthly installment amount for a purchase --/
def monthly_installment (cash_price deposit num_installments cash_savings : ℕ) : ℕ :=
  ((cash_price + cash_savings - deposit) / num_installments)

theorem refrigerator_installment :
  monthly_installment 8000 3000 30 4000 = 300 := by
  sorry

end refrigerator_installment_l3624_362428


namespace inequality_proof_l3624_362495

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d)) + (b^3 / (c + d + a)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
  sorry

end inequality_proof_l3624_362495


namespace triangle_side_length_l3624_362424

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a = 2, c = 2√3, and C = π/3, then b = 4 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  C = π / 3 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  b = 4 := by
  sorry

end triangle_side_length_l3624_362424


namespace product_remainder_mod_five_l3624_362458

theorem product_remainder_mod_five :
  (14452 * 15652 * 16781) % 5 = 4 := by
  sorry

end product_remainder_mod_five_l3624_362458


namespace path_bounds_l3624_362453

/-- Represents a tile with two segments -/
structure Tile :=
  (segments : Fin 2 → Unit)

/-- Represents a 2N × 2N board assembled with tiles -/
structure Board (N : ℕ) :=
  (tiles : Fin (4 * N^2) → Tile)

/-- The number of paths on a board -/
def num_paths (N : ℕ) (board : Board N) : ℕ := sorry

theorem path_bounds (N : ℕ) (board : Board N) :
  4 * N ≤ num_paths N board ∧ num_paths N board ≤ 2 * N^2 + 2 * N :=
sorry

end path_bounds_l3624_362453


namespace tan_sum_eq_neg_one_l3624_362449

theorem tan_sum_eq_neg_one (α β : ℝ) 
  (h : 2 * Real.sin β * Real.sin (α - π/4) = Real.sin (α - β + π/4)) : 
  Real.tan (α + β) = -1 := by
  sorry

end tan_sum_eq_neg_one_l3624_362449


namespace intersection_angle_proof_l3624_362478

/-- Given two curves in polar coordinates and a ray that intersects both curves, 
    prove that the angle of the ray is π/4 when the product of the distances 
    from the origin to the intersection points is 12. -/
theorem intersection_angle_proof (θ₀ : Real) 
  (h1 : 0 < θ₀) (h2 : θ₀ < Real.pi / 2) : 
  let curve_m := fun (θ : Real) => 4 * Real.cos θ
  let curve_n := fun (ρ θ : Real) => ρ^2 * Real.sin (2 * θ) = 18
  let ray := fun (ρ : Real) => (ρ * Real.cos θ₀, ρ * Real.sin θ₀)
  let point_a := (curve_m θ₀ * Real.cos θ₀, curve_m θ₀ * Real.sin θ₀)
  let point_b := 
    (Real.sqrt (18 / Real.sin (2 * θ₀)) * Real.cos θ₀, 
     Real.sqrt (18 / Real.sin (2 * θ₀)) * Real.sin θ₀)
  (curve_m θ₀ * Real.sqrt (18 / Real.sin (2 * θ₀)) = 12) → 
  θ₀ = Real.pi / 4 := by
sorry


end intersection_angle_proof_l3624_362478


namespace probability_of_red_ball_l3624_362439

/-- The probability of drawing a red ball from a bag with red and white balls -/
theorem probability_of_red_ball (red_balls white_balls : ℕ) :
  red_balls = 7 → white_balls = 3 →
  (red_balls : ℚ) / (red_balls + white_balls : ℚ) = 7 / 10 := by
  sorry

end probability_of_red_ball_l3624_362439


namespace scientific_notation_of_1300000_l3624_362438

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_1300000 :
  to_scientific_notation 1300000 = ScientificNotation.mk 1.3 6 sorry :=
sorry

end scientific_notation_of_1300000_l3624_362438


namespace misha_notebooks_l3624_362460

theorem misha_notebooks (a b c : ℕ) 
  (h1 : a + 6 = b + c)  -- Vera bought 6 notebooks less than Misha and Vasya together
  (h2 : b + 10 = a + c) -- Vasya bought 10 notebooks less than Vera and Misha together
  : c = 8 := by  -- Misha bought 8 notebooks
  sorry

end misha_notebooks_l3624_362460


namespace correct_sums_l3624_362425

theorem correct_sums (total : ℕ) (wrong : ℕ → ℕ) (h1 : total = 48) (h2 : wrong = λ x => 2 * x) : 
  ∃ x : ℕ, x + wrong x = total ∧ x = 16 := by
sorry

end correct_sums_l3624_362425


namespace range_of_a_l3624_362408

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 6) * (x - (2 * a + 5)) > 0}
def B (a : ℝ) : Set ℝ := {x | ((a^2 + 2) - x) * (2 * a - x) < 0}

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
    a > 1/2 → 
    (∀ x : ℝ, x ∈ B a → x ∈ A a) →
    (∃ x : ℝ, x ∈ A a ∧ x ∉ B a) →
    a > 1/2 ∧ a ≤ 2 :=
by sorry

end range_of_a_l3624_362408


namespace game_probability_l3624_362417

theorem game_probability (lose_prob : ℚ) (h1 : lose_prob = 5/8) (h2 : lose_prob + win_prob = 1) : win_prob = 3/8 := by
  sorry

end game_probability_l3624_362417


namespace quadratic_inequality_solution_l3624_362454

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, mx^2 + 8*m*x + 60 < 0 ↔ -5 < x ∧ x < -3) → m = 4 := by
  sorry

end quadratic_inequality_solution_l3624_362454


namespace product_of_special_set_l3624_362418

theorem product_of_special_set (n : ℕ) (M : Finset ℝ) : 
  Odd n → 
  n > 1 → 
  Finset.card M = n →
  (∀ x ∈ M, (M.sum id - x) + x = M.sum id) →
  M.prod id = 0 := by
sorry

end product_of_special_set_l3624_362418


namespace floor_equation_solutions_l3624_362437

theorem floor_equation_solutions : 
  (∃ (s : Finset ℕ), s.card = 110 ∧ 
    (∀ x : ℕ, x ∈ s ↔ ⌊(x : ℚ) / 10⌋ = ⌊(x : ℚ) / 11⌋ + 1)) := by
  sorry

end floor_equation_solutions_l3624_362437


namespace unique_solution_l3624_362420

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_range : hundreds ∈ Finset.range 10
  t_range : tens ∈ Finset.range 10
  o_range : ones ∈ Finset.range 10
  h_nonzero : hundreds ≠ 0

/-- Calculates the value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Calculates the product of digits of a number -/
def digitProduct (n : Nat) : Nat :=
  if n < 10 then n
  else if n < 100 then (n / 10) * (n % 10)
  else (n / 100) * ((n / 10) % 10) * (n % 10)

/-- Checks if a three-digit number satisfies the given conditions -/
def satisfiesConditions (n : ThreeDigitNumber) : Prop :=
  let firstProduct := digitProduct n.value
  let secondProduct := digitProduct firstProduct
  (10 ≤ firstProduct ∧ firstProduct < 100) ∧
  (0 < secondProduct ∧ secondProduct < 10) ∧
  n.hundreds = 1 ∧
  n.tens = firstProduct / 10 ∧
  n.ones = firstProduct % 10 ∧
  secondProduct = firstProduct % 10

theorem unique_solution :
  ∃! n : ThreeDigitNumber, satisfiesConditions n ∧ n.value = 144 :=
sorry

end unique_solution_l3624_362420


namespace exists_x_y_sequences_l3624_362463

/-- The sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * a (n + 1) - a n

/-- Theorem stating the existence of sequences x_n and y_n satisfying the given property -/
theorem exists_x_y_sequences :
  ∃ (x y : ℕ → ℕ), ∀ n, a n = (y n ^ 2 + 7 : ℚ) / ((x n : ℚ) - y n) :=
sorry

end exists_x_y_sequences_l3624_362463


namespace eight_digit_divisible_by_11_l3624_362462

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def digit (d : ℕ) : Prop :=
  d ≥ 0 ∧ d < 10

theorem eight_digit_divisible_by_11 (m : ℕ) :
  digit m →
  is_divisible_by_11 (73400000 + m * 100000 + 8527) →
  m = 6 := by
sorry

end eight_digit_divisible_by_11_l3624_362462


namespace six_balls_three_boxes_l3624_362461

/-- Represents a distribution of balls into boxes -/
def Distribution := List Nat

/-- Calculates the number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeWays (n : Nat) (k : Nat) : Nat :=
  sorry

/-- The number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distributeWays 6 3 = 122 := by
  sorry

end six_balls_three_boxes_l3624_362461


namespace apartment_expenditure_difference_l3624_362456

def akeno_expenditure : ℕ := 2985
def lev_expenditure : ℕ := akeno_expenditure / 3
def extra_akeno : ℕ := 1172

theorem apartment_expenditure_difference :
  ∃ (ambrocio_expenditure : ℕ),
    ambrocio_expenditure < lev_expenditure ∧
    akeno_expenditure = lev_expenditure + ambrocio_expenditure + extra_akeno ∧
    lev_expenditure - ambrocio_expenditure = 177 :=
by sorry

end apartment_expenditure_difference_l3624_362456


namespace f_min_at_three_l3624_362471

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem: The function f(x) = 3x^2 - 18x + 7 has a minimum value when x = 3 -/
theorem f_min_at_three : 
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 := by
  sorry

end f_min_at_three_l3624_362471


namespace tiger_speed_l3624_362466

/-- Proves that the tiger's speed is 30 kmph given the problem conditions -/
theorem tiger_speed (tiger_head_start : ℝ) (zebra_chase_time : ℝ) (zebra_speed : ℝ)
  (h1 : tiger_head_start = 5)
  (h2 : zebra_chase_time = 6)
  (h3 : zebra_speed = 55) :
  tiger_head_start * (zebra_speed * zebra_chase_time / (tiger_head_start + zebra_chase_time)) = 30 * tiger_head_start :=
by sorry

end tiger_speed_l3624_362466


namespace ratio_to_percent_l3624_362419

theorem ratio_to_percent (a b : ℕ) (h : a = 2 ∧ b = 3) : (a : ℚ) / (a + b : ℚ) * 100 = 40 := by
  sorry

end ratio_to_percent_l3624_362419


namespace figure_214_is_triangle_l3624_362485

/-- Represents the figures in the sequence -/
inductive Figure
| triangle : Figure
| square : Figure
| circle : Figure

/-- The pattern of the sequence -/
def pattern : List Figure := 
  [Figure.triangle, Figure.square, Figure.triangle, Figure.circle]

/-- The length of the pattern -/
def pattern_length : Nat := pattern.length

/-- The figure at a given position in the sequence -/
def figure_at_position (n : Nat) : Figure :=
  pattern[n % pattern_length]'
  (by 
    have h : n % pattern_length < pattern_length := 
      Nat.mod_lt n (Nat.zero_lt_succ _)
    exact h
  )

/-- Theorem: The 214th figure in the sequence is a triangle -/
theorem figure_214_is_triangle : 
  figure_at_position 213 = Figure.triangle :=
sorry

end figure_214_is_triangle_l3624_362485


namespace fraction_simplification_l3624_362488

theorem fraction_simplification (x y : ℝ) (h : x^2 ≠ 4*y^2) :
  (-x + 2*y) / (x^2 - 4*y^2) = -1 / (x + 2*y) := by
  sorry

end fraction_simplification_l3624_362488


namespace shirts_to_wash_l3624_362403

theorem shirts_to_wash (total_shirts : ℕ) (rewash_shirts : ℕ) (correctly_washed : ℕ) : 
  total_shirts = 63 → rewash_shirts = 12 → correctly_washed = 29 →
  total_shirts - correctly_washed + rewash_shirts = 46 := by
  sorry

end shirts_to_wash_l3624_362403


namespace regular_ngon_smallest_area_and_perimeter_l3624_362409

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- An n-gon circumscribed about a circle --/
structure CircumscribedNGon where
  n : ℕ
  circle : Circle
  vertices : Fin n → ℝ × ℝ

/-- Checks if an n-gon is regular --/
def is_regular (ngon : CircumscribedNGon) : Prop :=
  sorry

/-- Calculates the area of an n-gon --/
def area (ngon : CircumscribedNGon) : ℝ :=
  sorry

/-- Calculates the perimeter of an n-gon --/
def perimeter (ngon : CircumscribedNGon) : ℝ :=
  sorry

/-- Theorem: The regular n-gon has the smallest area and perimeter among all n-gons circumscribed about a given circle --/
theorem regular_ngon_smallest_area_and_perimeter (n : ℕ) (c : Circle) :
  ∀ (ngon : CircumscribedNGon), ngon.n = n ∧ ngon.circle = c →
    ∃ (reg_ngon : CircumscribedNGon), 
      reg_ngon.n = n ∧ reg_ngon.circle = c ∧ is_regular reg_ngon ∧
      area reg_ngon ≤ area ngon ∧ perimeter reg_ngon ≤ perimeter ngon :=
  sorry

end regular_ngon_smallest_area_and_perimeter_l3624_362409


namespace equation_solution_l3624_362429

theorem equation_solution (x : ℝ) : 
  (((1 - (Real.cos (3 * x))^15 * (Real.cos (5 * x))^2)^(1/4) = Real.sin (5 * x)) ∧ 
   (Real.sin (5 * x) ≥ 0)) ↔ 
  ((∃ n : ℤ, x = π / 10 + 2 * π * n / 5) ∨ 
   (∃ s : ℤ, x = 2 * π * s)) := by sorry

end equation_solution_l3624_362429


namespace carmichael_family_children_l3624_362446

/-- The Carmichael family problem -/
theorem carmichael_family_children (f : ℝ) (x : ℝ) (y : ℝ) : 
  (45 + f + x * y) / (2 + x) = 25 →   -- average age of the family
  (f + x * y) / (1 + x) = 20 →        -- average age of father and children
  x = 3 := by
sorry

end carmichael_family_children_l3624_362446


namespace reflected_ray_equation_l3624_362481

-- Define the points and lines
def P : ℝ × ℝ := (2, 3)
def A : ℝ × ℝ := (1, 1)
def incident_line (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the reflected ray
def reflected_ray (x y : ℝ) : Prop := 4*x - 5*y + 1 = 0

-- Theorem statement
theorem reflected_ray_equation :
  ∃ (x₀ y₀ : ℝ), 
    incident_line x₀ y₀ ∧  -- The incident ray strikes the line x + y + 1 = 0
    (∃ (t : ℝ), (1 - t) • P.1 + t • x₀ = P.1 ∧ (1 - t) • P.2 + t • y₀ = P.2) ∧  -- The incident ray passes through P
    reflected_ray A.1 A.2  -- The reflected ray passes through A
  → ∀ (x y : ℝ), reflected_ray x y ↔ 4*x - 5*y + 1 = 0 :=
by sorry

end reflected_ray_equation_l3624_362481


namespace nilpotent_matrix_powers_l3624_362415

theorem nilpotent_matrix_powers (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : A ^ 2 = 0 ∧ A ^ 3 = 0 := by
  sorry

end nilpotent_matrix_powers_l3624_362415


namespace smallest_AAB_value_l3624_362467

/-- Represents a digit (1 to 9) -/
def Digit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Represents a two-digit number AB -/
def TwoDigitNumber (A B : Digit) : ℕ := 10 * A.val + B.val

/-- Represents a three-digit number AAB -/
def ThreeDigitNumber (A B : Digit) : ℕ := 100 * A.val + 10 * A.val + B.val

/-- The main theorem -/
theorem smallest_AAB_value :
  ∀ (A B : Digit),
    A ≠ B →
    TwoDigitNumber A B = (ThreeDigitNumber A B) / 7 →
    ∀ (A' B' : Digit),
      A' ≠ B' →
      TwoDigitNumber A' B' = (ThreeDigitNumber A' B') / 7 →
      ThreeDigitNumber A B ≤ ThreeDigitNumber A' B' →
      ThreeDigitNumber A B = 664 :=
sorry

end smallest_AAB_value_l3624_362467


namespace range_of_a_l3624_362470

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x ≥ a
def q (x : ℝ) : Prop := |x - 1| < 1

-- Define the property that p is necessary but not sufficient for q
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  necessary_not_sufficient a → a ≤ 0 :=
by
  sorry


end range_of_a_l3624_362470


namespace goose_egg_count_l3624_362410

theorem goose_egg_count (
  hatch_rate : ℚ)
  (first_month_survival : ℚ)
  (next_three_months_survival : ℚ)
  (following_six_months_survival : ℚ)
  (first_half_second_year_survival : ℚ)
  (second_year_survival : ℚ)
  (final_survivors : ℕ)
  (h1 : hatch_rate = 4 / 7)
  (h2 : first_month_survival = 3 / 5)
  (h3 : next_three_months_survival = 7 / 10)
  (h4 : following_six_months_survival = 5 / 8)
  (h5 : first_half_second_year_survival = 2 / 3)
  (h6 : second_year_survival = 4 / 5)
  (h7 : final_survivors = 200) :
  ∃ (original_eggs : ℕ), original_eggs = 2503 ∧
  (↑final_survivors : ℚ) = ↑original_eggs * hatch_rate * first_month_survival *
    next_three_months_survival * following_six_months_survival *
    first_half_second_year_survival * second_year_survival :=
by sorry

end goose_egg_count_l3624_362410


namespace min_value_of_expression_l3624_362413

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let a : Fin 2 → ℝ := ![3, -2]
  let b : Fin 2 → ℝ := ![x, y - 1]
  (∃ (k : ℝ), a = k • b) →
  (3 / x + 2 / y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8) :=
by sorry

end min_value_of_expression_l3624_362413


namespace reading_competition_result_l3624_362400

/-- Represents the number of pages read by each girl --/
structure Pages where
  sasa : ℕ
  zuzka : ℕ
  ivana : ℕ
  majka : ℕ
  lucka : ℕ

/-- The conditions of the reading competition --/
def reading_conditions (p : Pages) : Prop :=
  p.lucka = 32 ∧
  p.lucka = (p.sasa + p.zuzka) / 2 ∧
  p.ivana = p.zuzka + 5 ∧
  p.majka = p.sasa - 8 ∧
  p.ivana = (p.majka + p.zuzka) / 2

/-- The theorem stating the correct number of pages read by each girl --/
theorem reading_competition_result :
  ∃ (p : Pages), reading_conditions p ∧
    p.sasa = 41 ∧ p.zuzka = 23 ∧ p.ivana = 28 ∧ p.majka = 33 ∧ p.lucka = 32 := by
  sorry

end reading_competition_result_l3624_362400


namespace trig_calculation_l3624_362496

theorem trig_calculation :
  Real.sin (π / 3) + Real.tan (π / 4) - Real.cos (π / 6) * Real.tan (π / 3) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end trig_calculation_l3624_362496


namespace money_division_l3624_362469

/-- Given an amount of money divided between A and B in the ratio 1:2, where A receives $200,
    prove that the total amount to be divided is $600. -/
theorem money_division (a b total : ℕ) : 
  (a : ℚ) / b = 1 / 2 →  -- The ratio of A's share to B's share is 1:2
  a = 200 →              -- A gets $200
  total = a + b →        -- Total is the sum of A's and B's shares
  total = 600 :=
by sorry

end money_division_l3624_362469


namespace donna_weekly_episodes_l3624_362407

/-- The number of episodes Donna can watch on a weekday -/
def weekday_episodes : ℕ := 8

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The factor by which weekend watching increases compared to weekdays -/
def weekend_factor : ℕ := 3

/-- The total number of episodes Donna can watch in a week -/
def total_episodes : ℕ := weekday_episodes * weekdays + weekend_factor * weekday_episodes * weekend_days

theorem donna_weekly_episodes : total_episodes = 88 := by
  sorry

end donna_weekly_episodes_l3624_362407


namespace imaginary_part_of_z_l3624_362404

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im ((i / (1 + i)) - (1 / (2 * i))) = 1 := by sorry

end imaginary_part_of_z_l3624_362404


namespace shortest_path_is_3_sqrt_2_l3624_362489

/-- A polyhedron with right dihedral angles that unfolds into three adjacent unit squares -/
structure RightAnglePolyhedron where
  -- We don't need to define the full structure, just the properties we need
  unfoldsToThreeUnitSquares : Bool

/-- Two vertices on the polyhedron -/
structure Vertex where
  -- We don't need to define the full structure, just declare it exists

/-- The shortest path between two vertices on the surface of the polyhedron -/
def shortestPath (p : RightAnglePolyhedron) (v1 v2 : Vertex) : ℝ :=
  sorry

/-- Theorem: The shortest path between opposite corners of the unfolded net is 3√2 -/
theorem shortest_path_is_3_sqrt_2 (p : RightAnglePolyhedron) (x y : Vertex) :
  p.unfoldsToThreeUnitSquares → shortestPath p x y = 3 * Real.sqrt 2 := by
  sorry

end shortest_path_is_3_sqrt_2_l3624_362489


namespace tangyuan_purchase_solution_l3624_362452

/-- Represents the number and price of tangyuan bags for two brands -/
structure TangyuanPurchase where
  brandA_quantity : ℕ
  brandB_quantity : ℕ
  brandA_price : ℕ
  brandB_price : ℕ

/-- Checks if a TangyuanPurchase satisfies all conditions -/
def is_valid_purchase (p : TangyuanPurchase) : Prop :=
  p.brandA_quantity + p.brandB_quantity = 1000 ∧
  p.brandA_quantity = 2 * p.brandB_quantity + 20 ∧
  p.brandB_price = p.brandA_price + 6 ∧
  5 * p.brandA_price = 3 * p.brandB_price

/-- The theorem to be proved -/
theorem tangyuan_purchase_solution :
  ∃ (p : TangyuanPurchase),
    is_valid_purchase p ∧
    p.brandA_quantity = 670 ∧
    p.brandB_quantity = 330 ∧
    p.brandA_price = 9 ∧
    p.brandB_price = 15 :=
  sorry

end tangyuan_purchase_solution_l3624_362452


namespace probability_is_one_third_l3624_362405

-- Define the number of hot dishes
def num_dishes : ℕ := 3

-- Define the number of dishes a student can choose
def num_choices : ℕ := 2

-- Define the function to calculate combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of two students choosing the same two dishes
def probability_same_choices : ℚ :=
  (combinations num_dishes num_choices) / (combinations num_dishes num_choices * combinations num_dishes num_choices)

-- Theorem to prove
theorem probability_is_one_third :
  probability_same_choices = 1 / 3 := by sorry

end probability_is_one_third_l3624_362405


namespace alternating_arrangement_white_first_arrangement_group_formation_l3624_362436

/- Define the total number of balls -/
def total_balls : ℕ := 12

/- Define the number of white balls -/
def white_balls : ℕ := 6

/- Define the number of black balls -/
def black_balls : ℕ := 6

/- Define the function to calculate factorial -/
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

/- Define the function to calculate combinations -/
def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

/- Theorem for part (a) -/
theorem alternating_arrangement : 
  factorial white_balls * factorial black_balls = 518400 := by sorry

/- Theorem for part (b) -/
theorem white_first_arrangement : 
  factorial white_balls * factorial black_balls = 518400 := by sorry

/- Theorem for part (c) -/
theorem group_formation : 
  choose white_balls 4 * factorial 4 * choose black_balls 3 * factorial 3 = 43200 := by sorry

end alternating_arrangement_white_first_arrangement_group_formation_l3624_362436


namespace two_noncongruent_triangles_l3624_362402

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  triangle_inequality : a ≤ b + c ∧ b ≤ a + c ∧ c ≤ a + b

/-- Two triangles are congruent if they have the same side lengths (up to permutation) -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.a ∧ t1.b = t2.c ∧ t1.c = t2.b) ∨
  (t1.a = t2.b ∧ t1.b = t2.a ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b) ∨
  (t1.a = t2.c ∧ t1.b = t2.b ∧ t1.c = t2.a)

/-- The set of all triangles with integer side lengths and perimeter 9 -/
def triangles_with_perimeter_9 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 9}

/-- There are exactly 2 non-congruent triangles with integer side lengths and perimeter 9 -/
theorem two_noncongruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ triangles_with_perimeter_9 ∧
    t2 ∈ triangles_with_perimeter_9 ∧
    ¬congruent t1 t2 ∧
    ∀ (t : IntTriangle),
      t ∈ triangles_with_perimeter_9 →
      (congruent t t1 ∨ congruent t t2) :=
sorry

end two_noncongruent_triangles_l3624_362402


namespace rachels_age_problem_l3624_362422

/-- Rachel's age problem -/
theorem rachels_age_problem 
  (rachel_age : ℕ)
  (grandfather_age : ℕ)
  (mother_age : ℕ)
  (father_age : ℕ)
  (h1 : rachel_age = 12)
  (h2 : grandfather_age = 7 * rachel_age)
  (h3 : mother_age = grandfather_age / 2)
  (h4 : father_age + (25 - rachel_age) = 60) :
  father_age - mother_age = 5 := by
sorry

end rachels_age_problem_l3624_362422


namespace system_solutions_l3624_362432

-- Define the system of equations
def equation1 (y z : ℚ) : Prop := y * z = 3 * y + 2 * z - 8
def equation2 (z x : ℚ) : Prop := z * x = 4 * z + 3 * x - 8
def equation3 (x y : ℚ) : Prop := x * y = 2 * x + y - 1

-- Define the solutions
def solution1 : (ℚ × ℚ × ℚ) := (2, 3, 1)
def solution2 : (ℚ × ℚ × ℚ) := (3, 5/2, -1)

-- Theorem statement
theorem system_solutions :
  (equation1 solution1.2.1 solution1.2.2 ∧ 
   equation2 solution1.2.2 solution1.1 ∧ 
   equation3 solution1.1 solution1.2.1) ∧
  (equation1 solution2.2.1 solution2.2.2 ∧ 
   equation2 solution2.2.2 solution2.1 ∧ 
   equation3 solution2.1 solution2.2.1) := by
  sorry

end system_solutions_l3624_362432


namespace function_maximum_implies_inequality_l3624_362431

/-- Given a function f(x) = ln x - mx² + 2nx where m is real and n is positive,
    if f(x) ≤ f(1) for all positive x, then ln n < 8m -/
theorem function_maximum_implies_inequality (m : ℝ) (n : ℝ) (h_n_pos : n > 0) :
  (∀ x > 0, Real.log x - m * x^2 + 2 * n * x ≤ Real.log 1 - m * 1^2 + 2 * n * 1) →
  Real.log n < 8 * m :=
by sorry

end function_maximum_implies_inequality_l3624_362431


namespace inscribed_cube_properties_l3624_362486

/-- Given a cube with a sphere inscribed in it, and another cube inscribed in that sphere,
    this theorem proves the surface area and volume of the inner cube. -/
theorem inscribed_cube_properties (outer_cube_surface_area : ℝ) 
  (h : outer_cube_surface_area = 96) :
  ∃ (inner_cube_surface_area inner_cube_volume : ℝ),
    inner_cube_surface_area = 32 ∧
    inner_cube_volume = 64 * Real.sqrt 3 / 9 := by
  sorry

end inscribed_cube_properties_l3624_362486


namespace inscribed_squares_ratio_l3624_362498

/-- Given a right triangle with sides 3, 4, and 5, x is the side length of a square
    inscribed with one vertex at the right angle, and y is the side length of a square
    inscribed with one side on the hypotenuse. -/
def triangle_with_squares (x y : ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a^2 + b^2 = c^2 ∧
    a = 3 ∧ b = 4 ∧ c = 5 ∧
    x / 4 = (3 - x) / 3 ∧
    4/3 * y + y + 3/4 * y = 5

theorem inscribed_squares_ratio :
  ∀ x y : ℝ, triangle_with_squares x y → x / y = 37 / 35 := by
  sorry

end inscribed_squares_ratio_l3624_362498


namespace camp_athlete_difference_l3624_362416

/-- The difference in the total number of athletes in the camp over two nights -/
def athlete_difference (initial : ℕ) (leaving_rate : ℕ) (leaving_hours : ℕ) (arriving_rate : ℕ) (arriving_hours : ℕ) : ℕ :=
  initial - (initial - leaving_rate * leaving_hours + arriving_rate * arriving_hours)

/-- Theorem stating the difference in the total number of athletes in the camp over two nights -/
theorem camp_athlete_difference : athlete_difference 600 35 6 20 9 = 30 := by
  sorry

end camp_athlete_difference_l3624_362416


namespace tan_theta_eq_two_implies_expression_eq_neg_two_l3624_362474

theorem tan_theta_eq_two_implies_expression_eq_neg_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / 
  (Real.sin (π / 2 - θ) - Real.sin (π - θ)) = -2 := by sorry

end tan_theta_eq_two_implies_expression_eq_neg_two_l3624_362474


namespace cos_18_degrees_l3624_362476

open Real

theorem cos_18_degrees : cos (18 * π / 180) = (Real.sqrt (10 + 2 * Real.sqrt 5)) / 4 := by sorry

end cos_18_degrees_l3624_362476


namespace absolute_value_equation_solutions_l3624_362441

theorem absolute_value_equation_solutions :
  {x : ℝ | |x - 2| = |x - 3| + |x - 6| + 2} = {-9, 9} := by sorry

end absolute_value_equation_solutions_l3624_362441


namespace mary_picked_14_oranges_l3624_362480

/-- The number of oranges picked by Jason -/
def jason_oranges : ℕ := 41

/-- The total number of oranges picked -/
def total_oranges : ℕ := 55

/-- The number of oranges picked by Mary -/
def mary_oranges : ℕ := total_oranges - jason_oranges

theorem mary_picked_14_oranges : mary_oranges = 14 := by
  sorry

end mary_picked_14_oranges_l3624_362480


namespace max_value_product_max_value_achieved_l3624_362479

theorem max_value_product (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  x^2 * y^3 * z ≤ 9/16 := by
  sorry

theorem max_value_achieved (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  ∃ x y z, x^2 * y^3 * z = 9/16 ∧ x + y + z = 3 := by
  sorry

end max_value_product_max_value_achieved_l3624_362479


namespace f_1994_4_l3624_362473

def f (x : ℚ) : ℚ := (2 + x) / (2 - 2*x)

def f_n : ℕ → (ℚ → ℚ)
| 0 => id
| (n+1) => f ∘ (f_n n)

theorem f_1994_4 : f_n 1994 4 = 1/4 := by sorry

end f_1994_4_l3624_362473


namespace perpendicular_planes_from_line_l3624_362490

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_line 
  (α β γ : Plane) (l : Line) 
  (distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  perpendicular_line_plane l α → 
  parallel_line_plane l β → 
  perpendicular α β :=
sorry

end perpendicular_planes_from_line_l3624_362490


namespace a_5_equals_18_l3624_362414

/-- For a sequence defined by a_n = n^2 - 2n + 3, a_5 = 18 -/
theorem a_5_equals_18 :
  let a : ℕ → ℤ := λ n => n^2 - 2*n + 3
  a 5 = 18 := by sorry

end a_5_equals_18_l3624_362414


namespace unit_digit_of_3_to_58_l3624_362401

theorem unit_digit_of_3_to_58 : 3^58 % 10 = 9 := by
  sorry

end unit_digit_of_3_to_58_l3624_362401


namespace adams_collection_worth_80_dollars_l3624_362430

/-- The value of Adam's coin collection -/
def adams_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℕ) : ℕ :=
  total_coins * (sample_value / sample_coins)

/-- Theorem: Adam's coin collection is worth 80 dollars -/
theorem adams_collection_worth_80_dollars :
  adams_collection_value 20 4 16 = 80 := by
  sorry

end adams_collection_worth_80_dollars_l3624_362430


namespace typist_salary_problem_l3624_362445

theorem typist_salary_problem (original_salary : ℝ) : 
  (original_salary * 1.1 * 0.95 = 5225) → original_salary = 5000 := by
  sorry

end typist_salary_problem_l3624_362445


namespace perfect_square_trinomial_l3624_362482

theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - a*x + 9 = (x + b)^2) → a = 6 ∨ a = -6 := by
  sorry

end perfect_square_trinomial_l3624_362482


namespace veterinary_clinic_payment_l3624_362483

/-- Veterinary clinic problem -/
theorem veterinary_clinic_payment
  (dog_charge : ℕ)
  (cat_charge : ℕ)
  (parrot_charge : ℕ)
  (rabbit_charge : ℕ)
  (dogs : ℕ)
  (cats : ℕ)
  (parrots : ℕ)
  (rabbits : ℕ)
  (h1 : dog_charge = 60)
  (h2 : cat_charge = 40)
  (h3 : parrot_charge = 70)
  (h4 : rabbit_charge = 50)
  (h5 : dogs = 25)
  (h6 : cats = 45)
  (h7 : parrots = 15)
  (h8 : rabbits = 10) :
  dog_charge * dogs + cat_charge * cats + parrot_charge * parrots + rabbit_charge * rabbits = 4850 := by
  sorry

end veterinary_clinic_payment_l3624_362483


namespace unfactorable_expression_difference_of_squares_factorization_common_factor_factorization_perfect_square_trinomial_factorization_l3624_362411

theorem unfactorable_expression (x : ℝ) : ¬∃ (a b : ℝ), x^2 + 9 = a * b ∧ (a ≠ 1 ∨ b ≠ x^2 + 9) ∧ (a ≠ x^2 + 9 ∨ b ≠ 1) := by
  sorry

-- Helper theorems to show that other expressions can be factored
theorem difference_of_squares_factorization (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

theorem common_factor_factorization (x : ℝ) : 9*x - 9 = 9 * (x - 1) := by
  sorry

theorem perfect_square_trinomial_factorization (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 := by
  sorry

end unfactorable_expression_difference_of_squares_factorization_common_factor_factorization_perfect_square_trinomial_factorization_l3624_362411


namespace p_or_q_is_true_l3624_362468

-- Define proposition p
def p : Prop := ∀ x : ℝ, (Real.exp x > 1) → (x > 0)

-- Define proposition q
def q : Prop := ∀ x : ℝ, (|x - 3| > 1) → (x > 4)

-- Theorem to prove
theorem p_or_q_is_true : p ∨ q := by sorry

end p_or_q_is_true_l3624_362468


namespace area_perimeter_product_l3624_362492

/-- A square on a grid with vertices at (1,5), (5,5), (5,1), and (1,1) -/
structure GridSquare where
  v1 : (ℕ × ℕ) := (1, 5)
  v2 : (ℕ × ℕ) := (5, 5)
  v3 : (ℕ × ℕ) := (5, 1)
  v4 : (ℕ × ℕ) := (1, 1)

/-- Calculate the side length of the GridSquare -/
def sideLength (s : GridSquare) : ℕ :=
  (s.v2.1 - s.v1.1)

/-- Calculate the area of the GridSquare -/
def area (s : GridSquare) : ℕ :=
  (sideLength s) ^ 2

/-- Calculate the perimeter of the GridSquare -/
def perimeter (s : GridSquare) : ℕ :=
  4 * (sideLength s)

/-- Theorem: The product of the area and perimeter of the GridSquare is 256 -/
theorem area_perimeter_product (s : GridSquare) : 
  area s * perimeter s = 256 := by
  sorry


end area_perimeter_product_l3624_362492


namespace parabola_points_difference_l3624_362406

/-- Given a parabola x^2 = 4y with focus F, and two points A and B on it satisfying |AF| - |BF| = 2,
    prove that y₁ + x₁² - y₂ - x₂² = 10 -/
theorem parabola_points_difference (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 = 4*y₁) →  -- A is on the parabola
  (x₂^2 = 4*y₂) →  -- B is on the parabola
  (y₁ + 1 - (y₂ + 1) = 2) →  -- |AF| - |BF| = 2, where F is (0, 1)
  y₁ + x₁^2 - y₂ - x₂^2 = 10 := by
sorry

end parabola_points_difference_l3624_362406


namespace one_third_1206_percent_of_400_l3624_362494

theorem one_third_1206_percent_of_400 : 
  (1206 / 3) / 400 * 100 = 100.5 := by sorry

end one_third_1206_percent_of_400_l3624_362494
