import Mathlib

namespace NUMINAMATH_CALUDE_collinear_points_theorem_l3064_306475

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if points A(a, 1), B(9, 0), and C(-3, 4) are collinear, then a = 6. -/
theorem collinear_points_theorem (a : ℝ) :
  collinear a 1 9 0 (-3) 4 → a = 6 := by
  sorry

#check collinear_points_theorem

end NUMINAMATH_CALUDE_collinear_points_theorem_l3064_306475


namespace NUMINAMATH_CALUDE_first_quiz_score_l3064_306455

def quiz_scores (score1 score2 score3 : ℕ) : Prop :=
  score2 = 90 ∧ score3 = 92

def average_score (score1 score2 score3 : ℕ) : ℚ :=
  (score1 + score2 + score3) / 3

theorem first_quiz_score (score1 : ℕ) :
  quiz_scores score1 90 92 →
  average_score score1 90 92 = 91 →
  score1 = 91 := by
sorry

end NUMINAMATH_CALUDE_first_quiz_score_l3064_306455


namespace NUMINAMATH_CALUDE_negative_seven_x_is_product_l3064_306497

theorem negative_seven_x_is_product : 
  ∀ x : ℝ, -7 * x = (-7) * x :=
by
  sorry

end NUMINAMATH_CALUDE_negative_seven_x_is_product_l3064_306497


namespace NUMINAMATH_CALUDE_mirror_side_length_l3064_306478

/-- Given a rectangular wall and a square mirror, proves that the mirror's side length is 34 inches -/
theorem mirror_side_length (wall_width wall_length mirror_area : ℝ) : 
  wall_width = 54 →
  wall_length = 42.81481481481482 →
  mirror_area = (wall_width * wall_length) / 2 →
  Real.sqrt mirror_area = 34 := by
  sorry

end NUMINAMATH_CALUDE_mirror_side_length_l3064_306478


namespace NUMINAMATH_CALUDE_product_of_geometric_terms_l3064_306464

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

def geometric_sequence (n : ℕ) : ℕ := 2^(n - 1)

theorem product_of_geometric_terms : 
  geometric_sequence (arithmetic_sequence 1) * 
  geometric_sequence (arithmetic_sequence 3) * 
  geometric_sequence (arithmetic_sequence 5) = 4096 := by
sorry

end NUMINAMATH_CALUDE_product_of_geometric_terms_l3064_306464


namespace NUMINAMATH_CALUDE_symmetric_partitions_generating_function_main_theorem_l3064_306427

/-- A partition is a non-increasing sequence of positive integers. -/
def Partition := List Nat

/-- A partition is symmetric if its Ferrers diagram is symmetric with respect to the diagonal. -/
def IsSymmetric (p : Partition) : Prop := sorry

/-- A partition consists of distinct odd parts if all its parts are odd and unique. -/
def HasDistinctOddParts (p : Partition) : Prop := sorry

/-- The generating function for partitions with a given property. -/
noncomputable def GeneratingFunction (P : Partition → Prop) : ℕ → ℚ := sorry

/-- The infinite product ∏_{k=1}^{∞} (1 + x^(2k+1)) -/
noncomputable def InfiniteProduct : ℕ → ℚ := sorry

theorem symmetric_partitions_generating_function :
  GeneratingFunction IsSymmetric = GeneratingFunction HasDistinctOddParts :=
by sorry

theorem main_theorem :
  GeneratingFunction IsSymmetric = InfiniteProduct :=
by sorry

end NUMINAMATH_CALUDE_symmetric_partitions_generating_function_main_theorem_l3064_306427


namespace NUMINAMATH_CALUDE_smallest_digit_sum_of_successor_l3064_306437

def digit_sum (n : ℕ) : ℕ := sorry

theorem smallest_digit_sum_of_successor (n : ℕ) (h : digit_sum n = 2017) :
  ∃ (m : ℕ), digit_sum (n + 1) = m ∧ ∀ (k : ℕ), digit_sum (n + 1) ≤ k := by sorry

end NUMINAMATH_CALUDE_smallest_digit_sum_of_successor_l3064_306437


namespace NUMINAMATH_CALUDE_sequence_fourth_term_l3064_306431

theorem sequence_fourth_term 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h : ∀ n, S n = (n + 1 : ℚ) / (n + 2 : ℚ)) : 
  a 4 = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_sequence_fourth_term_l3064_306431


namespace NUMINAMATH_CALUDE_electric_guitars_sold_l3064_306487

theorem electric_guitars_sold (total_guitars : ℕ) (total_revenue : ℕ) 
  (electric_price : ℕ) (acoustic_price : ℕ) :
  total_guitars = 9 →
  total_revenue = 3611 →
  electric_price = 479 →
  acoustic_price = 339 →
  ∃ (electric_sold : ℕ) (acoustic_sold : ℕ),
    electric_sold + acoustic_sold = total_guitars ∧
    electric_sold * electric_price + acoustic_sold * acoustic_price = total_revenue ∧
    electric_sold = 4 :=
by sorry

end NUMINAMATH_CALUDE_electric_guitars_sold_l3064_306487


namespace NUMINAMATH_CALUDE_minimum_point_l3064_306412

-- Define the function
def f (x : ℝ) : ℝ := 2 * |x - 4| - 2

-- State the theorem
theorem minimum_point :
  ∃! p : ℝ × ℝ, p.1 = 4 ∧ p.2 = -2 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
by sorry

end NUMINAMATH_CALUDE_minimum_point_l3064_306412


namespace NUMINAMATH_CALUDE_inequality_solution_l3064_306482

theorem inequality_solution (x : ℝ) :
  x ≠ 1 ∧ x ≠ 2 →
  ((x^3 - 3*x^2 + 2*x) / (x^2 - 3*x + 2) ≤ 1 ↔ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3064_306482


namespace NUMINAMATH_CALUDE_daniel_added_four_eggs_l3064_306409

/-- The number of eggs Daniel put in the box -/
def eggs_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Daniel put 4 eggs in the box -/
theorem daniel_added_four_eggs (initial final : ℕ) 
  (h1 : initial = 7) 
  (h2 : final = 11) : 
  eggs_added initial final = 4 := by
  sorry

end NUMINAMATH_CALUDE_daniel_added_four_eggs_l3064_306409


namespace NUMINAMATH_CALUDE_series_sum_l3064_306411

open Real

/-- The sum of the series ∑_{n=1}^{∞} (sin^n x) / n for x ≠ π/2 + 2πk, where k is an integer -/
theorem series_sum (x : ℝ) (h : ∀ k : ℤ, x ≠ π / 2 + 2 * π * k) :
  ∑' n, (sin x) ^ n / n = -log (1 - sin x) :=
by sorry


end NUMINAMATH_CALUDE_series_sum_l3064_306411


namespace NUMINAMATH_CALUDE_boys_in_line_l3064_306438

theorem boys_in_line (n : ℕ) (h : n > 0) :
  (∃ k : ℕ, k > 0 ∧ k = 19 ∧ k = n + 1 - 19) → n = 37 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_line_l3064_306438


namespace NUMINAMATH_CALUDE_school_paintable_area_l3064_306407

/-- Represents the dimensions of a classroom -/
structure ClassroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total wall area to be painted in all classrooms -/
def totalPaintableArea (dimensions : ClassroomDimensions) (numClassrooms : ℕ) (unpaintableArea : ℝ) : ℝ :=
  let wallArea := 2 * (dimensions.length * dimensions.height + dimensions.width * dimensions.height)
  let paintableArea := wallArea - unpaintableArea
  numClassrooms * paintableArea

/-- Theorem stating the total paintable area for the given school -/
theorem school_paintable_area :
  let dimensions : ClassroomDimensions := ⟨15, 12, 10⟩
  let numClassrooms : ℕ := 4
  let unpaintableArea : ℝ := 80
  totalPaintableArea dimensions numClassrooms unpaintableArea = 1840 := by
  sorry

#check school_paintable_area

end NUMINAMATH_CALUDE_school_paintable_area_l3064_306407


namespace NUMINAMATH_CALUDE_return_speed_calculation_l3064_306480

/-- Given a round trip with the following conditions:
    - Total distance is 20 km (10 km each way)
    - Return speed is twice the outbound speed
    - Total travel time is 6 hours
    Prove that the return speed is 5 km/h -/
theorem return_speed_calculation (distance : ℝ) (total_time : ℝ) : 
  distance = 10 →
  total_time = 6 →
  ∃ (outbound_speed : ℝ),
    outbound_speed > 0 ∧
    distance / outbound_speed + distance / (2 * outbound_speed) = total_time →
    2 * outbound_speed = 5 := by
  sorry

#check return_speed_calculation

end NUMINAMATH_CALUDE_return_speed_calculation_l3064_306480


namespace NUMINAMATH_CALUDE_derivative_of_linear_function_l3064_306463

theorem derivative_of_linear_function (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x + 5
  HasDerivAt f 3 x := by sorry

end NUMINAMATH_CALUDE_derivative_of_linear_function_l3064_306463


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l3064_306477

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- Theorem stating that (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean_triple :
  is_pythagorean_triple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l3064_306477


namespace NUMINAMATH_CALUDE_real_roots_sum_product_l3064_306406

theorem real_roots_sum_product (a b : ℝ) : 
  (a^4 - 6*a^2 - a + 2 = 0) → 
  (b^4 - 6*b^2 - b + 2 = 0) → 
  (∀ x : ℝ, x^4 - 6*x^2 - x + 2 = 0 → x = a ∨ x = b) →
  a * b + a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_sum_product_l3064_306406


namespace NUMINAMATH_CALUDE_ceiling_minus_x_zero_l3064_306498

theorem ceiling_minus_x_zero (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_zero_l3064_306498


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3064_306492

theorem imaginary_part_of_complex_fraction :
  Complex.im ((3 * Complex.I + 4) / (1 + 2 * Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3064_306492


namespace NUMINAMATH_CALUDE_two_digit_number_proof_l3064_306459

theorem two_digit_number_proof :
  ∀ n : ℕ,
  (10 ≤ n ∧ n < 100) →  -- two-digit number
  (n % 2 = 0) →  -- even number
  (n / 10 * (n % 10) = 20) →  -- product of digits is 20
  n = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_proof_l3064_306459


namespace NUMINAMATH_CALUDE_balloon_difference_l3064_306415

-- Define the initial conditions
def allan_initial : ℕ := 6
def jake_initial : ℕ := 2
def jake_bought : ℕ := 3
def allan_bought : ℕ := 4
def claire_from_jake : ℕ := 2
def claire_from_allan : ℕ := 3

-- Theorem statement
theorem balloon_difference :
  (allan_initial + allan_bought - claire_from_allan) -
  (jake_initial + jake_bought - claire_from_jake) = 4 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l3064_306415


namespace NUMINAMATH_CALUDE_computer_price_increase_l3064_306491

theorem computer_price_increase (c : ℝ) (h : 2 * c = 540) : 
  c * (1 + 0.3) = 351 := by
sorry

end NUMINAMATH_CALUDE_computer_price_increase_l3064_306491


namespace NUMINAMATH_CALUDE_cheese_slices_lcm_l3064_306439

theorem cheese_slices_lcm : 
  let cheddar_slices : ℕ := 12
  let swiss_slices : ℕ := 28
  let gouda_slices : ℕ := 18
  Nat.lcm (Nat.lcm cheddar_slices swiss_slices) gouda_slices = 252 := by
  sorry

end NUMINAMATH_CALUDE_cheese_slices_lcm_l3064_306439


namespace NUMINAMATH_CALUDE_birds_on_fence_l3064_306416

theorem birds_on_fence (initial_birds : ℕ) (new_birds : ℕ) : 
  initial_birds = 12 → new_birds = 8 → initial_birds + new_birds = 20 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3064_306416


namespace NUMINAMATH_CALUDE_sum_of_powers_l3064_306499

theorem sum_of_powers : (-2)^4 + (-2)^(3/2) + (-2)^1 + 2^1 + 2^(3/2) + 2^4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l3064_306499


namespace NUMINAMATH_CALUDE_min_packs_for_event_l3064_306448

/-- Represents a pack of utensils -/
structure UtensilPack where
  total : Nat
  knife : Nat
  fork : Nat
  spoon : Nat
  equal_distribution : knife = fork ∧ fork = spoon
  sum_equals_total : knife + fork + spoon = total

/-- Represents the required ratio of utensils -/
structure UtensilRatio where
  knife : Nat
  fork : Nat
  spoon : Nat

/-- Calculates the minimum number of packs needed to satisfy the ratio and spoon requirement -/
def min_packs_needed (pack : UtensilPack) (ratio : UtensilRatio) (min_spoons : Nat) : Nat :=
  sorry

/-- Main theorem: The minimum number of 30-utensil packs needed to satisfy a 2:3:5 ratio 
    of knives:forks:spoons and have at least 50 spoons is 5 packs -/
theorem min_packs_for_event : 
  let pack : UtensilPack := ⟨30, 10, 10, 10, ⟨rfl, rfl⟩, by simp⟩
  let ratio : UtensilRatio := ⟨2, 3, 5⟩
  let min_spoons : Nat := 50
  min_packs_needed pack ratio min_spoons = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_packs_for_event_l3064_306448


namespace NUMINAMATH_CALUDE_problem_statement_l3064_306436

theorem problem_statement (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 5) :
  x^2 * y + x * y^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3064_306436


namespace NUMINAMATH_CALUDE_square_floor_tiles_l3064_306495

/-- A square floor covered with congruent square tiles -/
structure SquareFloor :=
  (side_length : ℕ)

/-- The number of tiles along the diagonals of a square floor -/
def diagonal_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- The total number of tiles covering a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length ^ 2

/-- Theorem: If the total number of tiles along the two diagonals is 49,
    then the number of tiles covering the entire floor is 625 -/
theorem square_floor_tiles (floor : SquareFloor) :
  diagonal_tiles floor = 49 → total_tiles floor = 625 :=
by
  sorry


end NUMINAMATH_CALUDE_square_floor_tiles_l3064_306495


namespace NUMINAMATH_CALUDE_absolute_value_subtraction_l3064_306442

theorem absolute_value_subtraction : 4 - |(-3)| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_subtraction_l3064_306442


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_implies_perpendicular_l3064_306488

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_parallel_implies_perpendicular
  (α β : Plane) (m n : Line)
  (h1 : α ≠ β)
  (h2 : m ≠ n)
  (h3 : perpendicular m β)
  (h4 : parallel n β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_implies_perpendicular_l3064_306488


namespace NUMINAMATH_CALUDE_polynomial_identities_l3064_306468

theorem polynomial_identities (x y : ℝ) : 
  ((x + y)^3 - x^3 - y^3 = 3*x*y*(x + y)) ∧ 
  ((x + y)^5 - x^5 - y^5 = 5*x*y*(x + y)*(x^2 + x*y + y^2)) ∧ 
  ((x + y)^7 - x^7 - y^7 = 7*x*y*(x + y)*(x^2 + x*y + y^2)^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identities_l3064_306468


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l3064_306450

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l3064_306450


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_sqrt_ten_l3064_306458

theorem sqrt_difference_equals_negative_two_sqrt_ten :
  Real.sqrt (25 - 10 * Real.sqrt 6) - Real.sqrt (25 + 10 * Real.sqrt 6) = -2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_sqrt_ten_l3064_306458


namespace NUMINAMATH_CALUDE_perpendicular_point_sets_l3064_306402

-- Definition of a perpendicular point set
def is_perpendicular_point_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ M, ∃ p₂ ∈ M, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the sets M₃ and M₄
def M₃ : Set (ℝ × ℝ) := {p | p.2 = Real.exp p.1 - 2}
def M₄ : Set (ℝ × ℝ) := {p | p.2 = Real.sin p.1 + 1}

-- Theorem statement
theorem perpendicular_point_sets :
  is_perpendicular_point_set M₃ ∧ is_perpendicular_point_set M₄ := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_point_sets_l3064_306402


namespace NUMINAMATH_CALUDE_vector_sum_in_triangle_l3064_306465

-- Define the triangle ABC and points E and F
variable (A B C E F : ℝ × ℝ)

-- Define vectors
def AB : ℝ × ℝ := B - A
def AC : ℝ × ℝ := C - A
def AE : ℝ × ℝ := E - A
def CF : ℝ × ℝ := F - C
def FA : ℝ × ℝ := A - F
def EF : ℝ × ℝ := F - E

-- Define conditions
variable (h1 : AE = (1/2 : ℝ) • AB)
variable (h2 : CF = (2 : ℝ) • FA)
variable (x y : ℝ)
variable (h3 : EF = x • AB + y • AC)

-- Theorem statement
theorem vector_sum_in_triangle : x + y = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_in_triangle_l3064_306465


namespace NUMINAMATH_CALUDE_investment_plans_count_l3064_306433

/-- The number of candidate cities -/
def num_cities : ℕ := 4

/-- The number of projects to invest -/
def num_projects : ℕ := 3

/-- The maximum number of projects allowed in a single city -/
def max_projects_per_city : ℕ := 2

/-- A function that calculates the number of investment plans -/
def investment_plans : ℕ := sorry

/-- Theorem stating that the number of investment plans is 60 -/
theorem investment_plans_count : investment_plans = 60 := by sorry

end NUMINAMATH_CALUDE_investment_plans_count_l3064_306433


namespace NUMINAMATH_CALUDE_application_schemes_five_graduates_three_universities_l3064_306441

/-- The number of possible application schemes for high school graduates choosing universities. -/
def application_schemes (num_graduates : ℕ) (num_universities : ℕ) : ℕ :=
  num_universities ^ num_graduates

/-- Theorem: The number of possible application schemes for 5 high school graduates
    choosing from 3 universities, where each graduate can only fill in one preference,
    is equal to 3^5. -/
theorem application_schemes_five_graduates_three_universities :
  application_schemes 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_application_schemes_five_graduates_three_universities_l3064_306441


namespace NUMINAMATH_CALUDE_quadratic_function_bounds_l3064_306467

theorem quadratic_function_bounds (a : ℝ) (m : ℝ) : 
  a ≠ 0 → a < 0 → 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ m → 
    -2 ≤ a * x^2 + 2 * x + 1 ∧ a * x^2 + 2 * x + 1 ≤ 2) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_bounds_l3064_306467


namespace NUMINAMATH_CALUDE_square_of_zero_is_not_positive_l3064_306414

theorem square_of_zero_is_not_positive : ¬ (∀ x : ℕ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_of_zero_is_not_positive_l3064_306414


namespace NUMINAMATH_CALUDE_zero_point_implies_m_range_l3064_306429

theorem zero_point_implies_m_range (m : ℝ) :
  (∃ x₀ ∈ Set.Icc (-2 : ℝ) 0, 3 * m * x₀ - 4 = 0) →
  m ∈ Set.Iic (-2/3 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_zero_point_implies_m_range_l3064_306429


namespace NUMINAMATH_CALUDE_number_pairs_theorem_l3064_306483

theorem number_pairs_theorem (a b : ℝ) :
  a^2 + b^2 = 15 * (a + b) ∧ (a^2 - b^2 = 3 * (a - b) ∨ a^2 - b^2 = -3 * (a - b)) →
  (a = 6 ∧ b = -3) ∨ (a = -3 ∧ b = 6) ∨ (a = 0 ∧ b = 0) ∨ (a = 15 ∧ b = 15) :=
by sorry

end NUMINAMATH_CALUDE_number_pairs_theorem_l3064_306483


namespace NUMINAMATH_CALUDE_train_length_l3064_306471

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length (train_speed : Real) (crossing_time : Real) (bridge_length : Real) :
  train_speed = 45 * 1000 / 3600 →
  crossing_time = 30 →
  bridge_length = 295 →
  (train_speed * crossing_time) - bridge_length = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3064_306471


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3064_306445

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1) / (2 - x)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3064_306445


namespace NUMINAMATH_CALUDE_bug_movement_l3064_306444

/-- Probability of the bug being at the starting vertex after n moves -/
def Q (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1 / 3) * (1 - Q (n - 1))

/-- The bug's movement on a square -/
theorem bug_movement :
  Q 8 = 547 / 2187 :=
sorry

end NUMINAMATH_CALUDE_bug_movement_l3064_306444


namespace NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l3064_306410

def number_of_divisors (n : ℕ) : ℕ := sorry

theorem smallest_number_with_55_divisors :
  ∀ n : ℕ, number_of_divisors n = 55 → n ≥ 3^4 * 2^10 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l3064_306410


namespace NUMINAMATH_CALUDE_walk_legs_and_wheels_l3064_306432

/-- Calculates the total number of legs and wheels for a group of organisms and a wheelchair -/
def total_legs_and_wheels (humans : ℕ) (dogs : ℕ) (cats : ℕ) (horses : ℕ) (monkeys : ℕ) (wheelchair_wheels : ℕ) : ℕ :=
  humans * 2 + dogs * 4 + cats * 4 + horses * 4 + monkeys * 4 + wheelchair_wheels

/-- Proves that the total number of legs and wheels for the given group is 46 -/
theorem walk_legs_and_wheels :
  total_legs_and_wheels 9 3 1 1 1 4 = 46 := by
  sorry

end NUMINAMATH_CALUDE_walk_legs_and_wheels_l3064_306432


namespace NUMINAMATH_CALUDE_triangle_count_on_circle_l3064_306496

theorem triangle_count_on_circle (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) : 
  Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_on_circle_l3064_306496


namespace NUMINAMATH_CALUDE_emily_egg_collection_l3064_306456

theorem emily_egg_collection (num_baskets : ℕ) (eggs_per_basket : ℕ) 
  (h1 : num_baskets = 303) 
  (h2 : eggs_per_basket = 28) : 
  num_baskets * eggs_per_basket = 8484 := by
  sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l3064_306456


namespace NUMINAMATH_CALUDE_tan_periodicity_l3064_306404

theorem tan_periodicity (n : ℤ) : 
  -180 < n ∧ n < 180 → 
  Real.tan (n * π / 180) = Real.tan (1230 * π / 180) → 
  n = -30 := by
  sorry

end NUMINAMATH_CALUDE_tan_periodicity_l3064_306404


namespace NUMINAMATH_CALUDE_no_four_digit_sum_9_div_11_l3064_306461

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Theorem: There are no four-digit numbers whose digits add up to 9 and are divisible by 11 -/
theorem no_four_digit_sum_9_div_11 :
  ¬∃ (n : FourDigitNumber), sumOfDigits n.value = 9 ∧ n.value % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_four_digit_sum_9_div_11_l3064_306461


namespace NUMINAMATH_CALUDE_largest_value_of_P_10_l3064_306428

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial := ℝ → ℝ

/-- The largest possible value of P(10) for a quadratic polynomial P satisfying given conditions -/
theorem largest_value_of_P_10 (P : QuadraticPolynomial) 
  (h1 : P 1 = 20)
  (h2 : P (-1) = 22)
  (h3 : P (P 0) = 400) :
  ∃ (max : ℝ), P 10 ≤ max ∧ max = 2486 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_of_P_10_l3064_306428


namespace NUMINAMATH_CALUDE_arithmetic_seq_2015th_term_l3064_306405

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_1_eq_1 : a 1 = 1
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  d_neq_0 : a 2 - a 1 ≠ 0
  geometric_subseq : (a 2)^2 = a 1 * a 5

/-- The 2015th term of the arithmetic sequence is 4029 -/
theorem arithmetic_seq_2015th_term (seq : ArithmeticSequence) : seq.a 2015 = 4029 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_2015th_term_l3064_306405


namespace NUMINAMATH_CALUDE_current_speed_l3064_306489

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 25)
  (h2 : speed_against_current = 20) : 
  ∃ (mans_speed current_speed : ℝ),
    speed_with_current = mans_speed + current_speed ∧
    speed_against_current = mans_speed - current_speed ∧
    current_speed = 2.5 := by
sorry

end NUMINAMATH_CALUDE_current_speed_l3064_306489


namespace NUMINAMATH_CALUDE_jessica_roses_cut_l3064_306413

/-- The number of roses Jessica cut from her garden -/
def roses_cut : ℕ := 99

theorem jessica_roses_cut :
  let initial_roses : ℕ := 17
  let roses_thrown : ℕ := 8
  let roses_now : ℕ := 42
  let roses_given : ℕ := 6
  (initial_roses - roses_thrown + roses_cut / 3 = roses_now) ∧
  (roses_cut / 3 + roses_given = roses_now - initial_roses + roses_thrown + roses_given) →
  roses_cut = 99 := by
sorry

end NUMINAMATH_CALUDE_jessica_roses_cut_l3064_306413


namespace NUMINAMATH_CALUDE_food_drive_problem_l3064_306457

/-- Food drive problem -/
theorem food_drive_problem (rachel_cans jaydon_cans mark_cans : ℕ) : 
  jaydon_cans = 2 * rachel_cans + 5 →
  mark_cans = 4 * jaydon_cans →
  rachel_cans + jaydon_cans + mark_cans = 135 →
  mark_cans = 100 := by
  sorry

#check food_drive_problem

end NUMINAMATH_CALUDE_food_drive_problem_l3064_306457


namespace NUMINAMATH_CALUDE_points_on_line_equidistant_from_axes_l3064_306425

theorem points_on_line_equidistant_from_axes :
  ∃ (x y : ℝ), 4 * x - 3 * y = 24 ∧ |x| = |y| ∧
  ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by sorry

end NUMINAMATH_CALUDE_points_on_line_equidistant_from_axes_l3064_306425


namespace NUMINAMATH_CALUDE_book_distribution_count_correct_l3064_306423

/-- The number of ways to distribute 5 distinct books among 3 people,
    where one person receives 1 book and two people receive 2 books each. -/
def book_distribution_count : ℕ := 90

/-- Theorem stating that the number of book distribution methods is correct. -/
theorem book_distribution_count_correct :
  let n_books : ℕ := 5
  let n_people : ℕ := 3
  let books_per_person : List ℕ := [2, 2, 1]
  true → book_distribution_count = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_book_distribution_count_correct_l3064_306423


namespace NUMINAMATH_CALUDE_vishal_investment_percentage_l3064_306452

/-- Represents the investment amounts in rupees -/
structure Investment where
  vishal : ℝ
  trishul : ℝ
  raghu : ℝ

/-- The given conditions of the investment problem -/
def investment_conditions (i : Investment) : Prop :=
  i.raghu = 2300 ∧
  i.trishul = 0.9 * i.raghu ∧
  i.vishal + i.trishul + i.raghu = 6647

/-- The theorem stating that Vishal invested 10% more than Trishul -/
theorem vishal_investment_percentage (i : Investment) 
  (h : investment_conditions i) : 
  (i.vishal - i.trishul) / i.trishul = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_vishal_investment_percentage_l3064_306452


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l3064_306447

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_roots : a 2 * a 10 = 8 ∧ a 2 + a 10 = -12) :
  a 6 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l3064_306447


namespace NUMINAMATH_CALUDE_floor_difference_equals_ten_l3064_306476

theorem floor_difference_equals_ten : 
  ⌊(2010^4 / (2008 * 2009 : ℝ)) - (2008^4 / (2009 * 2010 : ℝ))⌋ = 10 := by sorry

end NUMINAMATH_CALUDE_floor_difference_equals_ten_l3064_306476


namespace NUMINAMATH_CALUDE_complex_equation_imaginary_part_l3064_306486

theorem complex_equation_imaginary_part :
  ∀ (z : ℂ), (3 + 4*I) * z = 5 → z.im = -4/5 := by sorry

end NUMINAMATH_CALUDE_complex_equation_imaginary_part_l3064_306486


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_inequality_l3064_306460

theorem sqrt_equality_implies_inequality (b : ℝ) : 
  Real.sqrt ((3 - b)^2) = 3 - b → b ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_inequality_l3064_306460


namespace NUMINAMATH_CALUDE_laundry_dry_cycle_time_l3064_306420

theorem laundry_dry_cycle_time 
  (total_loads : ℕ) 
  (wash_time_per_load : ℚ) 
  (total_time : ℚ) 
  (h1 : total_loads = 8) 
  (h2 : wash_time_per_load = 45 / 60) 
  (h3 : total_time = 14) : 
  (total_time - (total_loads : ℚ) * wash_time_per_load) / total_loads = 1 := by
  sorry

end NUMINAMATH_CALUDE_laundry_dry_cycle_time_l3064_306420


namespace NUMINAMATH_CALUDE_problem_solution_l3064_306403

theorem problem_solution (x y z : ℝ) (h1 : x + y = 5) (h2 : z^2 = x*y + y - 9) :
  x + 2*y + 3*z = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3064_306403


namespace NUMINAMATH_CALUDE_tractor_count_tractor_count_proof_l3064_306481

theorem tractor_count : ℝ → Prop :=
  fun T : ℝ =>
    let field_work : ℝ := T * 12
    let second_scenario_work : ℝ := 15 * 6.4
    (field_work = second_scenario_work) → T = 8

-- Proof
theorem tractor_count_proof : tractor_count 8 := by
  sorry

end NUMINAMATH_CALUDE_tractor_count_tractor_count_proof_l3064_306481


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3064_306490

/-- Given a school with a boy-to-girl ratio of 5:13 and 50 boys, prove that there are 80 more girls than boys. -/
theorem more_girls_than_boys (num_boys : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) : 
  num_boys = 50 →
  ratio_boys = 5 →
  ratio_girls = 13 →
  (ratio_girls * num_boys / ratio_boys) - num_boys = 80 :=
by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3064_306490


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_one_thirty_eight_satisfies_conditions_one_thirty_eight_is_greatest_main_result_l3064_306421

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
by sorry

theorem one_thirty_eight_satisfies_conditions : 138 < 150 ∧ Nat.gcd 138 18 = 6 :=
by sorry

theorem one_thirty_eight_is_greatest : 
  ∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ 138 :=
by sorry

theorem main_result : 
  (∃ n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ 
    ∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ n) ∧
  (∀ n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ 
    (∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ n) → n = 138) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_one_thirty_eight_satisfies_conditions_one_thirty_eight_is_greatest_main_result_l3064_306421


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l3064_306466

theorem geometric_progression_problem (b₂ b₆ : ℚ) 
  (h₂ : b₂ = 37 + 1/3) 
  (h₆ : b₆ = 2 + 1/3) : 
  ∃ (a q : ℚ), a * q = b₂ ∧ a * q^5 = b₆ ∧ a = 224/3 ∧ q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l3064_306466


namespace NUMINAMATH_CALUDE_circles_are_tangent_l3064_306485

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

-- Define what it means for two circles to be tangent
def are_tangent (c1 c2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), c1 x y ∧ c2 x y ∧ 
  ∀ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) → ¬(c1 x' y' ∧ c2 x' y')

-- Theorem statement
theorem circles_are_tangent : are_tangent circle1 circle2 := by
  sorry

end NUMINAMATH_CALUDE_circles_are_tangent_l3064_306485


namespace NUMINAMATH_CALUDE_closed_under_subtraction_l3064_306446

/-- A set of integers with special properties -/
structure SpecialIntegerSet where
  M : Set Int
  has_pos : ∃ x ∈ M, x > 0
  has_neg : ∃ x ∈ M, x < 0
  closed_double : ∀ a ∈ M, (2 * a) ∈ M
  closed_sum : ∀ a b, a ∈ M → b ∈ M → (a + b) ∈ M

/-- The main theorem: M is closed under subtraction -/
theorem closed_under_subtraction (S : SpecialIntegerSet) :
  ∀ a b, a ∈ S.M → b ∈ S.M → (a - b) ∈ S.M := by
  sorry

end NUMINAMATH_CALUDE_closed_under_subtraction_l3064_306446


namespace NUMINAMATH_CALUDE_smallest_unbounded_population_l3064_306484

theorem smallest_unbounded_population : ∃ N : ℕ, N = 61 ∧ 
  (∀ m : ℕ, m < N → 2 * (m - 30) ≤ m) ∧ 
  (2 * (N - 30) > N) := by
  sorry

end NUMINAMATH_CALUDE_smallest_unbounded_population_l3064_306484


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3064_306424

/-- The value of j for which the line 4x - 7y + j = 0 is tangent to the ellipse x^2 + 4y^2 = 16 -/
theorem line_tangent_to_ellipse (x y j : ℝ) : 
  (∀ x y, 4*x - 7*y + j = 0 → x^2 + 4*y^2 = 16) ↔ j^2 = 450.5 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3064_306424


namespace NUMINAMATH_CALUDE_smallest_square_area_l3064_306419

/-- The smallest square area containing two non-overlapping rectangles -/
theorem smallest_square_area (r1_width r1_height r2_width r2_height : ℕ) 
  (h1 : r1_width = 2 ∧ r1_height = 4)
  (h2 : r2_width = 3 ∧ r2_height = 5) : 
  (max (r1_width + r2_width) (max r1_height r2_height))^2 = 36 := by
  sorry

#check smallest_square_area

end NUMINAMATH_CALUDE_smallest_square_area_l3064_306419


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3064_306462

theorem quadratic_inequality (x : ℝ) : x^2 - 50*x + 625 ≤ 25 ↔ 20 ≤ x ∧ x ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3064_306462


namespace NUMINAMATH_CALUDE_compute_expression_l3064_306494

theorem compute_expression : 8 * (2 / 3)^4 = 128 / 81 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3064_306494


namespace NUMINAMATH_CALUDE_range_of_m_l3064_306454

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 10 then (1/10)^x else -Real.log (x + 2)

theorem range_of_m (m : ℝ) : f (8 - m^2) < f (2*m) → m ∈ Set.Ioo (-4) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3064_306454


namespace NUMINAMATH_CALUDE_old_manufacturing_cost_calculation_l3064_306451

def selling_price : ℝ := 100
def new_profit_percentage : ℝ := 0.50
def old_profit_percentage : ℝ := 0.20
def new_manufacturing_cost : ℝ := 50

theorem old_manufacturing_cost_calculation :
  let old_manufacturing_cost := selling_price * (1 - old_profit_percentage)
  old_manufacturing_cost = 80 :=
by sorry

end NUMINAMATH_CALUDE_old_manufacturing_cost_calculation_l3064_306451


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3064_306426

theorem simplify_trigonometric_expression (α : Real) 
  (h : π < α ∧ α < 3*π/2) : 
  Real.sqrt ((1 + Real.cos (9*π/2 - α)) / (1 + Real.sin (α - 5*π))) - 
  Real.sqrt ((1 - Real.cos (-3*π/2 - α)) / (1 - Real.sin (α - 9*π))) = 
  -2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3064_306426


namespace NUMINAMATH_CALUDE_integer_polynomial_roots_l3064_306479

/-- A polynomial with integer coefficients of the form x^4 + a₃x³ + a₂x² + a₁x - 27 = 0 -/
def IntegerPolynomial (a₃ a₂ a₁ : ℤ) (x : ℤ) : ℤ :=
  x^4 + a₃*x^3 + a₂*x^2 + a₁*x - 27

/-- The set of possible integer roots of the polynomial -/
def PossibleRoots : Set ℤ :=
  {-27, -9, -3, -1, 1, 3, 9, 27}

theorem integer_polynomial_roots (a₃ a₂ a₁ : ℤ) :
  ∀ x : ℤ, (IntegerPolynomial a₃ a₂ a₁ x = 0) ↔ x ∈ PossibleRoots :=
sorry

end NUMINAMATH_CALUDE_integer_polynomial_roots_l3064_306479


namespace NUMINAMATH_CALUDE_solution_is_121_l3064_306440

/-- Sum of first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- Sum of first n even numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The equation from the problem -/
def equation (n : ℕ) : Prop :=
  (sumOddNumbers n : ℚ) / (sumEvenNumbers n : ℚ) = 121 / 122

theorem solution_is_121 : ∃ (n : ℕ), n > 0 ∧ equation n ∧ n = 121 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_121_l3064_306440


namespace NUMINAMATH_CALUDE_spinner_probability_D_l3064_306443

-- Define the spinner with four regions
structure Spinner :=
  (A B C D : ℝ)

-- Define the properties of the spinner
def valid_spinner (s : Spinner) : Prop :=
  s.A = 1/4 ∧ s.B = 1/3 ∧ s.A + s.B + s.C + s.D = 1

-- Theorem statement
theorem spinner_probability_D (s : Spinner) 
  (h : valid_spinner s) : s.D = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_D_l3064_306443


namespace NUMINAMATH_CALUDE_segments_in_proportion_l3064_306418

/-- A set of four line segments is in proportion if the product of the extremes
    equals the product of the means. -/
def is_in_proportion (a b c d : ℝ) : Prop :=
  a * d = b * c

/-- The set of line segments (2, 3, 4, 6) is in proportion. -/
theorem segments_in_proportion :
  is_in_proportion 2 3 4 6 := by
  sorry

end NUMINAMATH_CALUDE_segments_in_proportion_l3064_306418


namespace NUMINAMATH_CALUDE_unique_function_solution_l3064_306408

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x ≥ 1, f x ≥ 1) → 
  (∀ x ≥ 1, f x ≤ 2 * (x + 1)) → 
  (∀ x ≥ 1, f (x + 1) = (1 / x) * ((f x)^2 - 1)) → 
  (∀ x ≥ 1, f x = x + 1) := by
sorry

end NUMINAMATH_CALUDE_unique_function_solution_l3064_306408


namespace NUMINAMATH_CALUDE_cupcakes_remaining_l3064_306435

theorem cupcakes_remaining (total : ℕ) (given_away_fraction : ℚ) (eaten : ℕ) : 
  total = 60 → given_away_fraction = 4/5 → eaten = 3 →
  total * (1 - given_away_fraction) - eaten = 9 := by
sorry

end NUMINAMATH_CALUDE_cupcakes_remaining_l3064_306435


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l3064_306453

theorem no_simultaneous_squares : ∀ n : ℤ, ¬(∃ a b c : ℤ, (10 * n - 1 = a^2) ∧ (13 * n - 1 = b^2) ∧ (85 * n - 1 = c^2)) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l3064_306453


namespace NUMINAMATH_CALUDE_no_solution_for_k_2_and_3_solution_exists_for_k_ge_4_l3064_306474

theorem no_solution_for_k_2_and_3 :
  (¬ ∃ (m n : ℕ+), m * (m + 2) = n * (n + 1)) ∧
  (¬ ∃ (m n : ℕ+), m * (m + 3) = n * (n + 1)) :=
sorry

theorem solution_exists_for_k_ge_4 :
  ∀ (k : ℕ), k ≥ 4 → ∃ (m n : ℕ+), m * (m + k) = n * (n + 1) :=
sorry

end NUMINAMATH_CALUDE_no_solution_for_k_2_and_3_solution_exists_for_k_ge_4_l3064_306474


namespace NUMINAMATH_CALUDE_M_mod_51_l3064_306400

def M : ℕ := sorry

theorem M_mod_51 : M % 51 = 15 := by sorry

end NUMINAMATH_CALUDE_M_mod_51_l3064_306400


namespace NUMINAMATH_CALUDE_jason_total_games_l3064_306401

/-- The number of football games Jason attended this month -/
def games_this_month : ℕ := 11

/-- The number of football games Jason attended last month -/
def games_last_month : ℕ := 17

/-- The number of football games Jason plans to attend next month -/
def games_next_month : ℕ := 16

/-- The total number of games Jason will attend -/
def total_games : ℕ := games_this_month + games_last_month + games_next_month

theorem jason_total_games : total_games = 44 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_games_l3064_306401


namespace NUMINAMATH_CALUDE_director_dividends_director_dividends_calculation_l3064_306493

/-- Calculates the dividends for the General Director given the financial data of the company. -/
theorem director_dividends (revenue : ℝ) (expenses : ℝ) (tax_rate : ℝ)
                           (monthly_loan_payment : ℝ) (annual_interest : ℝ)
                           (total_shares : ℕ) (director_shares : ℕ) : ℝ :=
  let net_profit := (revenue - expenses) - (revenue - expenses) * tax_rate
  let total_loan_payments := monthly_loan_payment * 12 - annual_interest
  let profits_for_dividends := net_profit - total_loan_payments
  let dividend_per_share := profits_for_dividends / total_shares
  dividend_per_share * director_shares

/-- The General Director's dividends are 246,400.0 rubles given the specified financial conditions. -/
theorem director_dividends_calculation :
  director_dividends 1500000 674992 0.2 23914 74992 1000 550 = 246400 := by
  sorry

end NUMINAMATH_CALUDE_director_dividends_director_dividends_calculation_l3064_306493


namespace NUMINAMATH_CALUDE_smallest_number_proof_l3064_306430

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  (a ≤ b ∧ b ≤ c) ∨ (a ≤ c ∧ c ≤ b) ∨ (b ≤ a ∧ a ≤ c) ∨ 
  (b ≤ c ∧ c ≤ a) ∨ (c ≤ a ∧ a ≤ b) ∨ (c ≤ b ∧ b ≤ a) →  -- Median condition
  b = 28 →                 -- Median is 28
  c = 34 →                 -- Largest number is 6 more than median
  a = 28                   -- Smallest number is 28
:= by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l3064_306430


namespace NUMINAMATH_CALUDE_not_fourth_power_prime_minus_four_l3064_306417

theorem not_fourth_power_prime_minus_four (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  ¬ ∃ (a : ℕ), p - 4 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_not_fourth_power_prime_minus_four_l3064_306417


namespace NUMINAMATH_CALUDE_time_rosa_sees_leo_l3064_306473

/-- Calculates the time Rosa can see Leo given their speeds and distances -/
theorem time_rosa_sees_leo (rosa_speed leo_speed initial_distance final_distance : ℚ) :
  rosa_speed = 15 →
  leo_speed = 5 →
  initial_distance = 3/4 →
  final_distance = 3/4 →
  (initial_distance + final_distance) / (rosa_speed - leo_speed) * 60 = 9 := by
  sorry

#check time_rosa_sees_leo

end NUMINAMATH_CALUDE_time_rosa_sees_leo_l3064_306473


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3064_306472

theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (2*a + 2*b = 2*Real.sqrt 2*c) →
  (a = 2) →
  (c^2 = a^2 + b^2) →
  (a^2 = 4 ∧ b^2 = 4) := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3064_306472


namespace NUMINAMATH_CALUDE_incorrect_inequality_l3064_306470

theorem incorrect_inequality (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) : 
  ¬(a * b > b^2) := by
sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l3064_306470


namespace NUMINAMATH_CALUDE_grassland_area_ratio_l3064_306422

/-- Represents a grassland with two parts -/
structure Grassland where
  areaA : ℝ
  areaB : ℝ
  growthRate : ℝ
  cowEatingRate : ℝ

/-- The conditions of the problem -/
def problem_conditions (g : Grassland) : Prop :=
  g.areaA > 0 ∧ g.areaB > 0 ∧ g.areaA ≠ g.areaB ∧
  g.growthRate > 0 ∧ g.cowEatingRate > 0 ∧
  g.areaA * g.growthRate = 7 * g.cowEatingRate ∧
  g.areaB * g.growthRate = 4 * g.cowEatingRate ∧
  7 * g.growthRate = g.areaA * g.growthRate

/-- The theorem stating the ratio of areas -/
theorem grassland_area_ratio (g : Grassland) :
  problem_conditions g → g.areaA / g.areaB = 105 / 44 :=
by sorry

end NUMINAMATH_CALUDE_grassland_area_ratio_l3064_306422


namespace NUMINAMATH_CALUDE_money_division_theorem_l3064_306469

theorem money_division_theorem (p q r : ℕ) (total : ℚ) :
  p + q + r = 22 →  -- Ratio sum: 3 + 7 + 12 = 22
  (7 * total / 22 - 3 * total / 22 = 2800) →  -- Difference between q and p's shares
  (12 * total / 22 - 7 * total / 22 = 3500) :=  -- Difference between r and q's shares
by sorry

end NUMINAMATH_CALUDE_money_division_theorem_l3064_306469


namespace NUMINAMATH_CALUDE_triangle_area_l3064_306434

/-- Given a triangle ABC with circumcircle diameter 4√3/3, angle C = 60°, and a + b = ab, 
    the area of the triangle is √3. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  (∃ (R : ℝ), 2 * R = 4 * Real.sqrt 3 / 3) →  -- Circumcircle diameter condition
  C = π / 3 →                                 -- Angle C = 60°
  a + b = a * b →                             -- Given condition
  (∃ (S : ℝ), S = a * b * Real.sin C / 2 ∧ S = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3064_306434


namespace NUMINAMATH_CALUDE_vector_q_solution_l3064_306449

/-- Custom vector operation ⊗ -/
def vector_op (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 * b.1, a.2 * b.2)

theorem vector_q_solution :
  let p : ℝ × ℝ := (1, 2)
  let q : ℝ × ℝ := (-3, -2)
  vector_op p q = (-3, -4) :=
by sorry

end NUMINAMATH_CALUDE_vector_q_solution_l3064_306449
