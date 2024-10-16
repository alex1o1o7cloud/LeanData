import Mathlib

namespace NUMINAMATH_CALUDE_stating_third_shirt_discount_is_sixty_percent_l2495_249532

/-- Represents the discount on a shirt as a fraction between 0 and 1 -/
def Discount : Type := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- The regular price of a shirt -/
def regularPrice : ℝ := 10

/-- The discount on the second shirt -/
def secondShirtDiscount : Discount := ⟨0.5, by norm_num⟩

/-- The total savings when buying three shirts -/
def totalSavings : ℝ := 11

/-- The discount on the third shirt -/
def thirdShirtDiscount : Discount := ⟨0.6, by norm_num⟩

/-- 
Theorem stating that given the regular price, second shirt discount, and total savings,
the discount on the third shirt is 60%.
-/
theorem third_shirt_discount_is_sixty_percent :
  (1 - thirdShirtDiscount.val) * regularPrice = 
    3 * regularPrice - totalSavings - regularPrice - (1 - secondShirtDiscount.val) * regularPrice :=
by sorry

end NUMINAMATH_CALUDE_stating_third_shirt_discount_is_sixty_percent_l2495_249532


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2495_249557

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2495_249557


namespace NUMINAMATH_CALUDE_orange_harvest_theorem_l2495_249592

/-- The number of sacks of oranges harvested per day -/
def sacks_per_day : ℕ := 38

/-- The number of days of harvest -/
def harvest_days : ℕ := 49

/-- The total number of sacks of oranges harvested after the harvest period -/
def total_sacks : ℕ := sacks_per_day * harvest_days

theorem orange_harvest_theorem : total_sacks = 1862 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_theorem_l2495_249592


namespace NUMINAMATH_CALUDE_three_additional_workers_needed_l2495_249565

/-- Given that 4 workers can produce 108 parts in 3 hours, this function calculates the number of additional workers needed to produce 504 parts in 8 hours. -/
def additional_workers_needed (current_workers : ℕ) (current_parts : ℕ) (current_hours : ℕ) (target_parts : ℕ) (target_hours : ℕ) : ℕ :=
  let production_rate := current_parts / (current_workers * current_hours)
  let required_workers := (target_parts / target_hours) / production_rate
  required_workers - current_workers

/-- Proves that 3 additional workers are needed under the given conditions. -/
theorem three_additional_workers_needed :
  additional_workers_needed 4 108 3 504 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_additional_workers_needed_l2495_249565


namespace NUMINAMATH_CALUDE_number_of_operations_indicates_quality_l2495_249529

-- Define a type for algorithms
structure Algorithm : Type where
  name : String

-- Define a measure for the number of operations
def numberOfOperations (a : Algorithm) : ℕ := sorry

-- Define a measure for algorithm quality
def algorithmQuality (a : Algorithm) : ℝ := sorry

-- Define a measure for computer speed
def computerSpeed : ℝ := sorry

-- State the theorem
theorem number_of_operations_indicates_quality (a : Algorithm) :
  computerSpeed > 0 →
  algorithmQuality a = (1 / numberOfOperations a) * computerSpeed :=
sorry

end NUMINAMATH_CALUDE_number_of_operations_indicates_quality_l2495_249529


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_trajectory_is_ellipse_proof_l2495_249535

/-- The set of points P satisfying the condition that |F₁F₂| is the arithmetic mean of |PF₁| and |PF₂|, 
    where F₁(-1,0) and F₂(1,0) are fixed points, forms an ellipse. -/
theorem trajectory_is_ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (-1, 0)
  let F₂ : ℝ × ℝ := (1, 0)
  let d₁ := dist P F₁
  let d₂ := dist P F₂
  (dist F₁ F₂ = (d₁ + d₂) / 2) → is_in_ellipse P F₁ F₂
  where
    dist : ℝ × ℝ → ℝ × ℝ → ℝ := λ (x₁, y₁) (x₂, y₂) => Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)
    is_in_ellipse : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop := sorry

theorem trajectory_is_ellipse_proof : ∀ P, trajectory_is_ellipse P := by sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_trajectory_is_ellipse_proof_l2495_249535


namespace NUMINAMATH_CALUDE_shopping_cart_fruit_ratio_l2495_249510

theorem shopping_cart_fruit_ratio (apples oranges pears : ℕ) : 
  oranges = 3 * apples →
  pears = 4 * oranges →
  (apples : ℚ) / pears = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_shopping_cart_fruit_ratio_l2495_249510


namespace NUMINAMATH_CALUDE_job_completion_proof_l2495_249520

/-- The number of days initially planned for 6 workers to complete a job -/
def initial_days : ℕ := sorry

/-- The number of workers who started the job -/
def initial_workers : ℕ := 6

/-- The number of days worked before additional workers joined -/
def days_before_joining : ℕ := 3

/-- The number of additional workers who joined -/
def additional_workers : ℕ := 4

/-- The number of days worked after additional workers joined -/
def days_after_joining : ℕ := 3

/-- The total number of worker-days required to complete the job -/
def total_worker_days : ℕ := initial_workers * initial_days

theorem job_completion_proof :
  total_worker_days = 
    initial_workers * days_before_joining + 
    (initial_workers + additional_workers) * days_after_joining ∧
  initial_days = 8 := by sorry

end NUMINAMATH_CALUDE_job_completion_proof_l2495_249520


namespace NUMINAMATH_CALUDE_factorization_equality_l2495_249551

theorem factorization_equality (a b : ℝ) : a * b^2 - a = a * (b + 1) * (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2495_249551


namespace NUMINAMATH_CALUDE_modulo_17_residue_l2495_249555

theorem modulo_17_residue : (3^4 + 6 * 49 + 8 * 137 + 7 * 34) % 17 = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulo_17_residue_l2495_249555


namespace NUMINAMATH_CALUDE_round_balloons_count_l2495_249504

/-- The number of balloons in each bag of round balloons -/
def round_balloons_per_bag : ℕ := sorry

/-- The number of bags of round balloons -/
def round_balloon_bags : ℕ := 5

/-- The number of bags of long balloons -/
def long_balloon_bags : ℕ := 4

/-- The number of long balloons in each bag -/
def long_balloons_per_bag : ℕ := 30

/-- The number of round balloons that burst -/
def burst_balloons : ℕ := 5

/-- The total number of balloons left -/
def total_balloons_left : ℕ := 215

theorem round_balloons_count : round_balloons_per_bag = 20 := by
  sorry

end NUMINAMATH_CALUDE_round_balloons_count_l2495_249504


namespace NUMINAMATH_CALUDE_gate_code_combinations_l2495_249521

theorem gate_code_combinations : Nat.factorial 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gate_code_combinations_l2495_249521


namespace NUMINAMATH_CALUDE_marquita_garden_count_l2495_249538

/-- The number of gardens Mancino is tending -/
def mancino_gardens : ℕ := 3

/-- The length of each of Mancino's gardens in feet -/
def mancino_garden_length : ℕ := 16

/-- The width of each of Mancino's gardens in feet -/
def mancino_garden_width : ℕ := 5

/-- The length of each of Marquita's gardens in feet -/
def marquita_garden_length : ℕ := 8

/-- The width of each of Marquita's gardens in feet -/
def marquita_garden_width : ℕ := 4

/-- The total area of all gardens combined in square feet -/
def total_garden_area : ℕ := 304

/-- Theorem stating the number of gardens Marquita is tilling -/
theorem marquita_garden_count : 
  ∃ n : ℕ, n * (marquita_garden_length * marquita_garden_width) = 
    total_garden_area - mancino_gardens * (mancino_garden_length * mancino_garden_width) ∧ 
    n = 2 := by
  sorry

end NUMINAMATH_CALUDE_marquita_garden_count_l2495_249538


namespace NUMINAMATH_CALUDE_percentage_difference_l2495_249561

theorem percentage_difference : (0.6 * 50) - (0.42 * 30) = 17.4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2495_249561


namespace NUMINAMATH_CALUDE_town_population_theorem_l2495_249596

theorem town_population_theorem (total_population : ℕ) (num_groups : ℕ) (male_groups : ℕ) :
  total_population = 450 →
  num_groups = 4 →
  male_groups = 2 →
  (male_groups * (total_population / num_groups) : ℕ) = 225 :=
by sorry

end NUMINAMATH_CALUDE_town_population_theorem_l2495_249596


namespace NUMINAMATH_CALUDE_sequence_property_l2495_249531

theorem sequence_property (a b c : ℝ) 
  (h1 : (4 * b) ^ 2 = 3 * a * 5 * c)  -- geometric sequence condition
  (h2 : 2 / b = 1 / a + 1 / c)        -- arithmetic sequence condition
  : a / c + c / a = 34 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l2495_249531


namespace NUMINAMATH_CALUDE_remainder_theorem_application_l2495_249583

/-- Given a polynomial q(x) = Mx^4 + Nx^2 + Dx - 5, 
    if the remainder when divided by (x - 2) is 15,
    then the remainder when divided by (x + 2) is also 15. -/
theorem remainder_theorem_application (M N D : ℝ) : 
  let q : ℝ → ℝ := λ x => M * x^4 + N * x^2 + D * x - 5
  (q 2 = 15) → (q (-2) = 15) := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_application_l2495_249583


namespace NUMINAMATH_CALUDE_shortened_card_area_l2495_249509

/-- Represents a rectangular card with given dimensions -/
structure Card where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular card -/
def area (c : Card) : ℝ := c.length * c.width

/-- Represents the amount by which each side is shortened -/
structure Shortening where
  length_reduction : ℝ
  width_reduction : ℝ

/-- Applies a shortening to a card -/
def apply_shortening (c : Card) (s : Shortening) : Card :=
  { length := c.length - s.length_reduction,
    width := c.width - s.width_reduction }

theorem shortened_card_area 
  (original : Card)
  (shortening : Shortening)
  (h1 : original.length = 5)
  (h2 : original.width = 7)
  (h3 : shortening.length_reduction = 2)
  (h4 : shortening.width_reduction = 1) :
  area (apply_shortening original shortening) = 18 := by
  sorry

end NUMINAMATH_CALUDE_shortened_card_area_l2495_249509


namespace NUMINAMATH_CALUDE_diagonal_sum_is_384_l2495_249528

/-- A cyclic hexagon with five sides of length 81 and one side of length 31 -/
structure CyclicHexagon where
  -- Five sides have length 81
  side_length : ℝ
  side_length_eq : side_length = 81
  -- One side (AB) has length 31
  AB_length : ℝ
  AB_length_eq : AB_length = 31

/-- The sum of the lengths of the three diagonals drawn from one vertex in the hexagon -/
def diagonal_sum (h : CyclicHexagon) : ℝ := sorry

/-- Theorem: The sum of the lengths of the three diagonals drawn from one vertex is 384 -/
theorem diagonal_sum_is_384 (h : CyclicHexagon) : diagonal_sum h = 384 := by sorry

end NUMINAMATH_CALUDE_diagonal_sum_is_384_l2495_249528


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l2495_249594

theorem probability_of_black_ball (prob_red prob_white : ℝ) 
  (h1 : prob_red = 0.42)
  (h2 : prob_white = 0.28)
  (h3 : prob_red + prob_white + prob_black = 1) :
  prob_black = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l2495_249594


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l2495_249500

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^3 * (x + 1) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l2495_249500


namespace NUMINAMATH_CALUDE_grade_assignments_12_students_4_grades_l2495_249548

/-- The number of possible grade assignments for a class -/
def gradeAssignments (numStudents : ℕ) (numGrades : ℕ) : ℕ :=
  numGrades ^ numStudents

/-- Theorem stating the number of ways to assign 4 grades to 12 students -/
theorem grade_assignments_12_students_4_grades :
  gradeAssignments 12 4 = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignments_12_students_4_grades_l2495_249548


namespace NUMINAMATH_CALUDE_product_mod_eight_l2495_249581

theorem product_mod_eight : (55 * 57) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_eight_l2495_249581


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l2495_249577

/-- Represents a cube that has been painted on all sides and then cut into smaller cubes -/
structure PaintedCube where
  edge_length : ℕ
  small_cube_edge : ℕ

/-- Counts the number of small cubes with a given number of painted faces -/
def count_painted_faces (c : PaintedCube) (num_faces : ℕ) : ℕ := sorry

theorem painted_cube_theorem (c : PaintedCube) 
  (h1 : c.edge_length = 5) 
  (h2 : c.small_cube_edge = 1) : 
  (count_painted_faces c 3 = 8) ∧ 
  (count_painted_faces c 2 = 36) ∧ 
  (count_painted_faces c 1 = 54) := by sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l2495_249577


namespace NUMINAMATH_CALUDE_constant_term_value_l2495_249507

theorem constant_term_value (x y C : ℝ) 
  (eq1 : 7 * x + y = C)
  (eq2 : x + 3 * y = 1)
  (eq3 : 2 * x + y = 5) : 
  C = 19 := by
sorry

end NUMINAMATH_CALUDE_constant_term_value_l2495_249507


namespace NUMINAMATH_CALUDE_vector_a_solution_l2495_249580

theorem vector_a_solution (a b : ℝ × ℝ) : 
  b = (1, 2) → 
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  (a.1^2 + a.2^2 = 20) → 
  (a = (4, -2) ∨ a = (-4, 2)) := by
sorry

end NUMINAMATH_CALUDE_vector_a_solution_l2495_249580


namespace NUMINAMATH_CALUDE_pyramid_lateral_surface_area_l2495_249570

/-- Given a cylinder with height H and base radius R, and a pyramid inside the cylinder
    with its height coinciding with the cylinder's slant height, and its base being an
    isosceles triangle ABC inscribed in the cylinder's base with ∠A = 120°,
    the lateral surface area of the pyramid is (R/4) * (4H + √(3R² + 12H²)). -/
theorem pyramid_lateral_surface_area 
  (H R : ℝ) 
  (H_pos : H > 0) 
  (R_pos : R > 0) : 
  ∃ (pyramid_area : ℝ), 
    pyramid_area = (R / 4) * (4 * H + Real.sqrt (3 * R^2 + 12 * H^2)) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_lateral_surface_area_l2495_249570


namespace NUMINAMATH_CALUDE_rugby_banquet_min_guests_l2495_249545

/-- The minimum number of guests at a banquet given the total food consumed and maximum individual consumption --/
def min_guests (total_food : ℕ) (max_individual_consumption : ℕ) : ℕ :=
  (total_food + max_individual_consumption - 1) / max_individual_consumption

/-- Theorem stating the minimum number of guests at the rugby banquet --/
theorem rugby_banquet_min_guests :
  min_guests 4875 3 = 1625 := by
  sorry

end NUMINAMATH_CALUDE_rugby_banquet_min_guests_l2495_249545


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_l2495_249536

theorem cubic_sum_of_roots (r s : ℝ) : 
  r^2 - 5*r + 3 = 0 → 
  s^2 - 5*s + 3 = 0 → 
  r^3 + s^3 = 80 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_l2495_249536


namespace NUMINAMATH_CALUDE_prob_no_adjacent_birch_is_2_55_l2495_249562

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of trees -/
def total_trees : ℕ := 15

/-- The number of birch trees -/
def birch_trees : ℕ := 6

/-- The number of non-birch trees -/
def non_birch_trees : ℕ := total_trees - birch_trees

/-- The probability of no two birch trees being adjacent when arranged randomly -/
def prob_no_adjacent_birch : ℚ := 
  choose (non_birch_trees + 1) birch_trees / choose total_trees birch_trees

theorem prob_no_adjacent_birch_is_2_55 : 
  prob_no_adjacent_birch = 2 / 55 := by sorry

end NUMINAMATH_CALUDE_prob_no_adjacent_birch_is_2_55_l2495_249562


namespace NUMINAMATH_CALUDE_intersection_equality_l2495_249586

def M : Set ℝ := {x : ℝ | x < 2012}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_equality : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_equality_l2495_249586


namespace NUMINAMATH_CALUDE_remainder_b_107_mod_64_l2495_249593

theorem remainder_b_107_mod_64 : (5^107 + 9^107) % 64 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_b_107_mod_64_l2495_249593


namespace NUMINAMATH_CALUDE_radical_product_equals_64_l2495_249585

theorem radical_product_equals_64 : 
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_equals_64_l2495_249585


namespace NUMINAMATH_CALUDE_initial_tadpoles_correct_l2495_249582

/-- The initial number of tadpoles Trent caught -/
def initial_tadpoles : ℕ := 180

/-- The percentage of tadpoles Trent kept -/
def kept_percentage : ℚ := 1 - 3/4

/-- The number of tadpoles Trent kept -/
def kept_tadpoles : ℕ := 45

/-- Theorem stating that the initial number of tadpoles is correct -/
theorem initial_tadpoles_correct : 
  initial_tadpoles = kept_tadpoles / kept_percentage :=
by sorry

end NUMINAMATH_CALUDE_initial_tadpoles_correct_l2495_249582


namespace NUMINAMATH_CALUDE_hide_and_seek_l2495_249533

-- Define the players
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
variable (h1 : Andrew → (Boris ∧ ¬Vasya))
variable (h2 : Boris → (Gena ∨ Denis))
variable (h3 : ¬Vasya → (¬Boris ∧ ¬Denis))
variable (h4 : ¬Andrew → (Boris ∧ ¬Gena))

-- Theorem statement
theorem hide_and_seek : 
  Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena := by sorry

end NUMINAMATH_CALUDE_hide_and_seek_l2495_249533


namespace NUMINAMATH_CALUDE_jellybean_box_theorem_l2495_249571

/-- The number of jellybeans in a box that is three times larger in each dimension
    compared to a box that holds 200 jellybeans -/
theorem jellybean_box_theorem (ella_jellybeans : ℕ) (scale_factor : ℕ) :
  ella_jellybeans = 200 →
  scale_factor = 3 →
  scale_factor ^ 3 * ella_jellybeans = 5400 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_box_theorem_l2495_249571


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2495_249552

/-- Given a rectangle with initial dimensions 3 × 7 inches, if shortening one side by 2 inches 
    results in an area of 15 square inches, then shortening the other side by 2 inches 
    will result in an area of 7 square inches. -/
theorem rectangle_area_change (initial_width initial_length : ℝ) 
  (h1 : initial_width = 3)
  (h2 : initial_length = 7)
  (h3 : initial_width * (initial_length - 2) = 15 ∨ (initial_width - 2) * initial_length = 15) :
  (initial_width - 2) * initial_length = 7 ∨ initial_width * (initial_length - 2) = 7 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2495_249552


namespace NUMINAMATH_CALUDE_intersection_of_P_and_complement_of_M_l2495_249527

-- Define the sets
def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x ≥ 3}
def M : Set ℝ := {x | x < 4}

-- State the theorem
theorem intersection_of_P_and_complement_of_M :
  P ∩ (Set.univ \ M) = {x : ℝ | x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_complement_of_M_l2495_249527


namespace NUMINAMATH_CALUDE_comparison_theorem_l2495_249516

theorem comparison_theorem :
  let a : ℝ := (5/3)^(1/5)
  let b : ℝ := (2/3)^10
  let c : ℝ := Real.log 6 / Real.log 0.3
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_comparison_theorem_l2495_249516


namespace NUMINAMATH_CALUDE_sprocket_production_rate_l2495_249563

/-- The number of sprockets both machines produce -/
def total_sprockets : ℕ := 330

/-- The additional time (in hours) machine A takes compared to machine B -/
def time_difference : ℕ := 10

/-- The production rate increase of machine B compared to machine A -/
def rate_increase : ℚ := 1/10

/-- The production rate of machine A in sprockets per hour -/
def machine_a_rate : ℚ := 3

/-- The production rate of machine B in sprockets per hour -/
def machine_b_rate : ℚ := machine_a_rate * (1 + rate_increase)

/-- The time taken by machine A to produce the total sprockets -/
def machine_a_time : ℚ := total_sprockets / machine_a_rate

/-- The time taken by machine B to produce the total sprockets -/
def machine_b_time : ℚ := total_sprockets / machine_b_rate

theorem sprocket_production_rate :
  (machine_a_time = machine_b_time + time_difference) ∧
  (machine_b_rate = machine_a_rate * (1 + rate_increase)) ∧
  (total_sprockets = machine_a_rate * machine_a_time) ∧
  (total_sprockets = machine_b_rate * machine_b_time) :=
sorry

end NUMINAMATH_CALUDE_sprocket_production_rate_l2495_249563


namespace NUMINAMATH_CALUDE_evaluate_expression_l2495_249575

theorem evaluate_expression : (2023 - 1984)^2 / 144 = 10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2495_249575


namespace NUMINAMATH_CALUDE_subset_pair_existence_l2495_249511

theorem subset_pair_existence (n : ℕ) (A : Fin n → Set ℕ) :
  ∃ (X Y : ℕ), ∀ i : Fin n, (X ∈ A i ∧ Y ∈ A i) ∨ (X ∉ A i ∧ Y ∉ A i) := by
  sorry

end NUMINAMATH_CALUDE_subset_pair_existence_l2495_249511


namespace NUMINAMATH_CALUDE_greatest_integer_gcd_4_with_12_l2495_249506

theorem greatest_integer_gcd_4_with_12 : 
  ∃ n : ℕ, n < 100 ∧ Nat.gcd n 12 = 4 ∧ ∀ m : ℕ, m < 100 → Nat.gcd m 12 = 4 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_gcd_4_with_12_l2495_249506


namespace NUMINAMATH_CALUDE_proposition_truth_values_l2495_249556

theorem proposition_truth_values :
  let p := ∀ x y : ℝ, x > y → -x < -y
  let q := ∀ x y : ℝ, x > y → x^2 > y^2
  (p ∨ q) ∧ (p ∧ (¬q)) ∧ ¬(p ∧ q) ∧ ¬((¬p) ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l2495_249556


namespace NUMINAMATH_CALUDE_calculate_expression_l2495_249508

theorem calculate_expression : -1^2 + 8 / (-2)^2 - (-4) * (-3) = -11 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2495_249508


namespace NUMINAMATH_CALUDE_square_property_l2495_249544

theorem square_property (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_property_l2495_249544


namespace NUMINAMATH_CALUDE_circle_equation_a_range_l2495_249584

/-- A circle in the xy-plane can be represented by the equation x^2 + y^2 - 2x + 2y + a = 0,
    where a is a real number. This theorem states that the range of a for which this equation
    represents a circle is (-∞, 2). -/
theorem circle_equation_a_range :
  ∀ a : ℝ, (∃ x y : ℝ, x^2 + y^2 - 2*x + 2*y + a = 0 ∧ 
    ∀ x' y' : ℝ, x'^2 + y'^2 - 2*x' + 2*y' + a = 0 → (x' - x)^2 + (y' - y)^2 = Constant)
  ↔ a < 2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_a_range_l2495_249584


namespace NUMINAMATH_CALUDE_function_bound_l2495_249541

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 13

-- State the theorem
theorem function_bound (x m : ℝ) (h : |x - m| < 1) : |f x - f m| < 2 * (|m| + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l2495_249541


namespace NUMINAMATH_CALUDE_leftover_value_is_five_fifty_l2495_249540

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents the number of coins a person has --/
structure CoinCount where
  quarters : Nat
  dimes : Nat

/-- Calculates the value of leftover coins in dollars --/
def leftoverValue (rollSize : RollSize) (james : CoinCount) (lindsay : CoinCount) : ℚ :=
  let totalQuarters := james.quarters + lindsay.quarters
  let totalDimes := james.dimes + lindsay.dimes
  let leftoverQuarters := totalQuarters % rollSize.quarters
  let leftoverDimes := totalDimes % rollSize.dimes
  (leftoverQuarters : ℚ) * (1 / 4) + (leftoverDimes : ℚ) * (1 / 10)

theorem leftover_value_is_five_fifty :
  let rollSize : RollSize := { quarters := 40, dimes := 50 }
  let james : CoinCount := { quarters := 83, dimes := 159 }
  let lindsay : CoinCount := { quarters := 129, dimes := 266 }
  leftoverValue rollSize james lindsay = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_leftover_value_is_five_fifty_l2495_249540


namespace NUMINAMATH_CALUDE_first_floor_bedrooms_count_l2495_249546

/-- Represents a two-story house with bedrooms -/
structure House where
  total_bedrooms : ℕ
  second_floor_bedrooms : ℕ

/-- Calculates the number of bedrooms on the first floor -/
def first_floor_bedrooms (h : House) : ℕ :=
  h.total_bedrooms - h.second_floor_bedrooms

/-- Theorem: For a house with 10 total bedrooms and 2 bedrooms on the second floor,
    the first floor has 8 bedrooms -/
theorem first_floor_bedrooms_count (h : House) 
    (h_total : h.total_bedrooms = 10)
    (h_second : h.second_floor_bedrooms = 2) : 
    first_floor_bedrooms h = 8 := by
  sorry

end NUMINAMATH_CALUDE_first_floor_bedrooms_count_l2495_249546


namespace NUMINAMATH_CALUDE_larger_number_proof_l2495_249501

/-- Given two positive integers with specific h.c.f. and l.c.m., prove the larger number is 391 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf : Nat.gcd a b = 23)
  (lcm : Nat.lcm a b = 23 * 13 * 17) :
  max a b = 391 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2495_249501


namespace NUMINAMATH_CALUDE_additional_plates_l2495_249554

/-- The number of choices for each letter position in the original license plate system -/
def original_choices : Fin 3 → Nat
  | 0 => 5  -- First position
  | 1 => 3  -- Second position
  | 2 => 4  -- Third position

/-- The total number of possible license plates in the original system -/
def original_total : Nat := (original_choices 0) * (original_choices 1) * (original_choices 2)

/-- The number of choices for each letter position after adding one letter to each set -/
def new_choices : Fin 3 → Nat
  | i => (original_choices i) + 1

/-- The total number of possible license plates in the new system -/
def new_total : Nat := (new_choices 0) * (new_choices 1) * (new_choices 2)

/-- The theorem stating the number of additional license plates -/
theorem additional_plates : new_total - original_total = 60 := by
  sorry

end NUMINAMATH_CALUDE_additional_plates_l2495_249554


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2495_249595

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4 * x + 3 = 0 ∧ 
   ∀ y : ℝ, a * y^2 - 4 * y + 3 = 0 → y = x) → 
  a = 4/3 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2495_249595


namespace NUMINAMATH_CALUDE_cubic_difference_l2495_249530

theorem cubic_difference (a b : ℝ) 
  (h1 : a - b = 7)
  (h2 : a^2 + b^2 = 65)
  (h3 : a + b = 6) :
  a^3 - b^3 = 432.25 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l2495_249530


namespace NUMINAMATH_CALUDE_rotation_exists_l2495_249589

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line
  mk :: -- Constructor

/-- Represents a point in 3D space -/
structure Point3D where
  -- Add necessary fields for a 3D point
  mk :: -- Constructor

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane
  mk :: -- Constructor

/-- Represents a rotation in 3D space -/
structure Rotation3D where
  -- Add necessary fields for a 3D rotation
  mk :: -- Constructor
  apply : Point3D → Point3D  -- Applies the rotation to a point

def are_skew (l1 l2 : Line3D) : Prop := sorry

def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

def rotation_maps (r : Rotation3D) (l1 l2 : Line3D) (p1 p2 : Point3D) : Prop := sorry

def plane_of_symmetry (p1 p2 : Point3D) : Plane3D := sorry

def plane_intersection (p1 p2 : Plane3D) : Line3D := sorry

theorem rotation_exists (a a' : Line3D) (A : Point3D) (A' : Point3D) 
  (h1 : are_skew a a')
  (h2 : point_on_line A a)
  (h3 : point_on_line A' a') :
  ∃ (l : Line3D), ∃ (r : Rotation3D), rotation_maps r a a' A A' := by
  sorry

end NUMINAMATH_CALUDE_rotation_exists_l2495_249589


namespace NUMINAMATH_CALUDE_hexagon_diagonal_intersection_theorem_l2495_249588

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A hexagon defined by its six vertices -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- The intersection point of the diagonals -/
def G (h : Hexagon) : Point := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a hexagon is convex -/
def isConvex (h : Hexagon) : Prop := sorry

/-- Checks if a hexagon is inscribed in a circle -/
def isInscribed (h : Hexagon) : Prop := sorry

/-- Checks if three lines intersect at a point forming 60° angles -/
def intersectAt60Degrees (p1 p2 p3 p4 p5 p6 : Point) : Prop := sorry

/-- The main theorem -/
theorem hexagon_diagonal_intersection_theorem (h : Hexagon) 
  (convex : isConvex h)
  (inscribed : isInscribed h)
  (intersect : intersectAt60Degrees h.A h.D h.B h.E h.C h.F) :
  distance (G h) h.A + distance (G h) h.C + distance (G h) h.E =
  distance (G h) h.B + distance (G h) h.D + distance (G h) h.F := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_intersection_theorem_l2495_249588


namespace NUMINAMATH_CALUDE_cistern_water_depth_l2495_249598

/-- Proves that for a cistern with given dimensions and wet surface area, the water depth is 1.25 m -/
theorem cistern_water_depth
  (length : ℝ)
  (width : ℝ)
  (total_wet_area : ℝ)
  (h_length : length = 12)
  (h_width : width = 14)
  (h_total_wet_area : total_wet_area = 233) :
  let bottom_area := length * width
  let perimeter := 2 * (length + width)
  let water_depth := (total_wet_area - bottom_area) / perimeter
  water_depth = 1.25 := by
sorry

end NUMINAMATH_CALUDE_cistern_water_depth_l2495_249598


namespace NUMINAMATH_CALUDE_crayon_selection_l2495_249566

theorem crayon_selection (n k : ℕ) (h1 : n = 15) (h2 : k = 5) :
  Nat.choose n k = 3003 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_l2495_249566


namespace NUMINAMATH_CALUDE_chandler_saves_26_weeks_l2495_249590

/-- The number of weeks it takes Chandler to save for a mountain bike. -/
def weeks_to_save : ℕ :=
  let bike_cost : ℕ := 650
  let birthday_money : ℕ := 80 + 35 + 15
  let weekly_earnings : ℕ := 20
  (bike_cost - birthday_money) / weekly_earnings

/-- Theorem stating that it takes 26 weeks for Chandler to save for the mountain bike. -/
theorem chandler_saves_26_weeks : weeks_to_save = 26 := by
  sorry

end NUMINAMATH_CALUDE_chandler_saves_26_weeks_l2495_249590


namespace NUMINAMATH_CALUDE_shaded_area_equals_circle_area_l2495_249587

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- Configuration of the diagram -/
structure DiagramConfig where
  x : ℝ
  A : Point
  B : Point
  C : Point
  D : Point
  semicircleAB : Semicircle
  semicircleAC : Semicircle
  semicircleCB : Semicircle

/-- The main theorem -/
theorem shaded_area_equals_circle_area (config : DiagramConfig) : 
  (config.A.x - config.B.x = 8 * config.x) →
  (config.A.x - config.C.x = 6 * config.x) →
  (config.C.x - config.B.x = 2 * config.x) →
  (config.D.y - config.C.y = Real.sqrt 3 * config.x) →
  (config.semicircleAB.radius = 4 * config.x) →
  (config.semicircleAC.radius = 3 * config.x) →
  (config.semicircleCB.radius = config.x) →
  (config.semicircleAB.center.x = (config.A.x + config.B.x) / 2) →
  (config.semicircleAC.center.x = (config.A.x + config.C.x) / 2) →
  (config.semicircleCB.center.x = (config.C.x + config.B.x) / 2) →
  (config.A.y = config.B.y) →
  (config.C.y = config.A.y) →
  (config.D.x = config.C.x) →
  (π * (4 * config.x)^2 / 2 - π * (3 * config.x)^2 / 2 - π * config.x^2 / 2 = π * (Real.sqrt 3 * config.x)^2) :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_equals_circle_area_l2495_249587


namespace NUMINAMATH_CALUDE_pump_rates_determination_l2495_249539

/-- Represents the pumping rates and durations of three water pumps -/
structure PumpSystem where
  rate1 : ℝ  -- Pumping rate of the first pump
  rate2 : ℝ  -- Pumping rate of the second pump
  rate3 : ℝ  -- Pumping rate of the third pump
  time1 : ℝ  -- Working time of the first pump
  time2 : ℝ  -- Working time of the second pump
  time3 : ℝ  -- Working time of the third pump

/-- Checks if the given pump system satisfies all the conditions -/
def satisfiesConditions (p : PumpSystem) : Prop :=
  p.time1 = p.time3 ∧  -- First and third pumps finish simultaneously
  p.time2 = 2 ∧  -- Second pump works for 2 hours
  p.rate1 * p.time1 = 9 ∧  -- First pump pumps 9 m³
  p.rate2 * p.time2 + p.rate3 * p.time3 = 28 ∧  -- Second and third pumps pump 28 m³ together
  p.rate3 = p.rate1 + 3 ∧  -- Third pump pumps 3 m³ more per hour than the first
  p.rate1 + p.rate2 + p.rate3 = 14  -- Three pumps together pump 14 m³ per hour

/-- Theorem stating that the given conditions imply specific pumping rates -/
theorem pump_rates_determination (p : PumpSystem) 
  (h : satisfiesConditions p) : p.rate1 = 3 ∧ p.rate2 = 5 ∧ p.rate3 = 6 := by
  sorry


end NUMINAMATH_CALUDE_pump_rates_determination_l2495_249539


namespace NUMINAMATH_CALUDE_average_problem_l2495_249534

theorem average_problem (x : ℝ) : (15 + 25 + x) / 3 = 23 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l2495_249534


namespace NUMINAMATH_CALUDE_no_real_solutions_l2495_249599

theorem no_real_solutions :
  ¬∃ (x : ℝ), (6 * x) / (x^2 + 2 * x + 5) + (7 * x) / (x^2 - 7 * x + 5) = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2495_249599


namespace NUMINAMATH_CALUDE_min_value_theorem_l2495_249537

-- Define the condition for a, b, c
def satisfies_condition (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3

-- State the theorem
theorem min_value_theorem :
  ∃ (a b c : ℝ), satisfies_condition a b c ∧
  (∀ (a' b' c' : ℝ), satisfies_condition a' b' c' → a + 2*b - 3*c ≤ a' + 2*b' - 3*c') ∧
  a + 2*b - 3*c = -2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2495_249537


namespace NUMINAMATH_CALUDE_total_cats_sum_l2495_249574

/-- The number of cats owned by Mr. Thompson -/
def thompson_cats : ℝ := 15.5

/-- The number of cats owned by Mrs. Sheridan -/
def sheridan_cats : ℝ := 11.6

/-- The number of cats owned by Mrs. Garrett -/
def garrett_cats : ℝ := 24.2

/-- The number of cats owned by Mr. Ravi -/
def ravi_cats : ℝ := 18.3

/-- The total number of cats owned by all four people -/
def total_cats : ℝ := thompson_cats + sheridan_cats + garrett_cats + ravi_cats

theorem total_cats_sum :
  total_cats = 69.6 := by sorry

end NUMINAMATH_CALUDE_total_cats_sum_l2495_249574


namespace NUMINAMATH_CALUDE_total_notes_count_l2495_249526

/-- Proves that given a total amount of Rs. 10350 in Rs. 50 and Rs. 500 notes,
    with 77 notes of Rs. 50 denomination, the total number of notes is 90. -/
theorem total_notes_count (total_amount : ℕ) (notes_50_count : ℕ) (notes_50_value : ℕ) (notes_500_value : ℕ) :
  total_amount = 10350 →
  notes_50_count = 77 →
  notes_50_value = 50 →
  notes_500_value = 500 →
  ∃ (notes_500_count : ℕ),
    total_amount = notes_50_count * notes_50_value + notes_500_count * notes_500_value ∧
    notes_50_count + notes_500_count = 90 :=
by sorry

end NUMINAMATH_CALUDE_total_notes_count_l2495_249526


namespace NUMINAMATH_CALUDE_floor_sqrt_eight_count_l2495_249524

theorem floor_sqrt_eight_count : 
  (Finset.range 81 \ Finset.range 64).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_eight_count_l2495_249524


namespace NUMINAMATH_CALUDE_arithmetic_expression_proof_l2495_249543

theorem arithmetic_expression_proof : (6 + 6 * 3 - 3) / 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_proof_l2495_249543


namespace NUMINAMATH_CALUDE_sequence_property_l2495_249568

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ p q : ℕ, a (p + q) = a p * a q) →
  a 8 = 16 →
  a 10 = 32 := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l2495_249568


namespace NUMINAMATH_CALUDE_racers_meeting_time_l2495_249550

/-- The time in seconds for the Racing Magic to complete one lap -/
def racing_magic_lap_time : ℕ := 60

/-- The number of laps the Charging Bull completes in one hour -/
def charging_bull_laps_per_hour : ℕ := 40

/-- The time in seconds for the Charging Bull to complete one lap -/
def charging_bull_lap_time : ℕ := 3600 / charging_bull_laps_per_hour

/-- The least common multiple of the two lap times -/
def lcm_lap_times : ℕ := Nat.lcm racing_magic_lap_time charging_bull_lap_time

/-- The time in minutes for the racers to meet at the starting point for the second time -/
def meeting_time_minutes : ℕ := lcm_lap_times / 60

theorem racers_meeting_time :
  meeting_time_minutes = 3 := by sorry

end NUMINAMATH_CALUDE_racers_meeting_time_l2495_249550


namespace NUMINAMATH_CALUDE_stickers_remaining_proof_l2495_249505

/-- Calculates the number of stickers remaining after losing a page. -/
def stickers_remaining (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) : ℕ :=
  (initial_pages - pages_lost) * stickers_per_page

/-- Proves that the number of stickers remaining is 220. -/
theorem stickers_remaining_proof :
  stickers_remaining 20 12 1 = 220 := by
  sorry

end NUMINAMATH_CALUDE_stickers_remaining_proof_l2495_249505


namespace NUMINAMATH_CALUDE_line_segment_division_l2495_249502

/-- Given a line segment with endpoints A(3, 2) and B(12, 8) divided into three equal parts,
    prove that the coordinates of the division points are C(6, 4) and D(9, 6),
    and the length of the segment AB is √117. -/
theorem line_segment_division (A B C D : ℝ × ℝ) : 
  A = (3, 2) → 
  B = (12, 8) → 
  C = ((3 + 0.5 * 12) / 1.5, (2 + 0.5 * 8) / 1.5) → 
  D = ((3 + 2 * 12) / 3, (2 + 2 * 8) / 3) → 
  C = (6, 4) ∧ 
  D = (9, 6) ∧ 
  Real.sqrt ((12 - 3)^2 + (8 - 2)^2) = Real.sqrt 117 :=
sorry

end NUMINAMATH_CALUDE_line_segment_division_l2495_249502


namespace NUMINAMATH_CALUDE_light_nanosecond_distance_l2495_249572

/-- The speed of light in meters per second -/
def speed_of_light : ℝ := 3e8

/-- One billionth of a second in seconds -/
def one_billionth : ℝ := 1e-9

/-- The distance traveled by light in one billionth of a second in meters -/
def light_nanosecond : ℝ := speed_of_light * one_billionth

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

theorem light_nanosecond_distance :
  light_nanosecond * meters_to_cm = 30 := by sorry

end NUMINAMATH_CALUDE_light_nanosecond_distance_l2495_249572


namespace NUMINAMATH_CALUDE_golden_ratio_equivalences_l2495_249549

open Real

theorem golden_ratio_equivalences :
  let φ : ℝ := 2 * sin (18 * π / 180)
  (sin (102 * π / 180) + Real.sqrt 3 * cos (102 * π / 180) = φ) ∧
  (sin (36 * π / 180) / sin (108 * π / 180) = φ) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_equivalences_l2495_249549


namespace NUMINAMATH_CALUDE_president_secretary_selection_l2495_249518

/-- The number of ways to select one president and one secretary from five different people. -/
def select_president_and_secretary (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: The number of ways to select one president and one secretary from five different people is 20. -/
theorem president_secretary_selection :
  select_president_and_secretary 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_president_secretary_selection_l2495_249518


namespace NUMINAMATH_CALUDE_carols_allowance_l2495_249503

/-- Carol's allowance problem -/
theorem carols_allowance
  (fixed_allowance : ℚ)
  (extra_chore_pay : ℚ)
  (weeks : ℕ)
  (total_amount : ℚ)
  (avg_extra_chores : ℚ)
  (h1 : extra_chore_pay = 1.5)
  (h2 : weeks = 10)
  (h3 : total_amount = 425)
  (h4 : avg_extra_chores = 15) :
  fixed_allowance = 20 := by
  sorry

end NUMINAMATH_CALUDE_carols_allowance_l2495_249503


namespace NUMINAMATH_CALUDE_product_of_sines_equals_one_sixteenth_l2495_249523

theorem product_of_sines_equals_one_sixteenth :
  (1 + Real.sin (π / 12)) * (1 + Real.sin (5 * π / 12)) *
  (1 + Real.sin (7 * π / 12)) * (1 + Real.sin (11 * π / 12)) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_equals_one_sixteenth_l2495_249523


namespace NUMINAMATH_CALUDE_factoring_polynomial_l2495_249512

theorem factoring_polynomial (x y : ℝ) : 
  -6 * x^2 * y + 12 * x * y^2 - 3 * x * y = -3 * x * y * (2 * x - 4 * y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factoring_polynomial_l2495_249512


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l2495_249513

theorem danny_bottle_caps (thrown_away old_caps new_caps_initial new_caps_additional : ℕ) 
  (h1 : thrown_away = 6)
  (h2 : new_caps_initial = 50)
  (h3 : new_caps_additional = thrown_away + 44) :
  new_caps_initial + new_caps_additional - thrown_away = 94 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l2495_249513


namespace NUMINAMATH_CALUDE_carpet_cost_l2495_249591

/-- The cost of a carpet with increased dimensions -/
theorem carpet_cost (b₁ : ℝ) (l₁_factor : ℝ) (l₂_increase : ℝ) (b₂_increase : ℝ) (rate : ℝ) :
  b₁ = 6 →
  l₁_factor = 1.44 →
  l₂_increase = 0.4 →
  b₂_increase = 0.25 →
  rate = 45 →
  (b₁ * (1 + b₂_increase)) * (b₁ * l₁_factor * (1 + l₂_increase)) * rate = 4082.4 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_l2495_249591


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2495_249519

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 3) ^ 2 - 3 * (a 3) + 2 = 0 →
  (a 7) ^ 2 - 3 * (a 7) + 2 = 0 →
  a 5 = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2495_249519


namespace NUMINAMATH_CALUDE_white_copy_cost_is_five_cents_l2495_249576

/- Define the problem parameters -/
def total_copies : ℕ := 400
def colored_copies : ℕ := 50
def colored_cost : ℚ := 10 / 100  -- 10 cents in dollars
def total_bill : ℚ := 225 / 10    -- $22.50

/- Define the cost of a white copy -/
def white_copy_cost : ℚ := (total_bill - colored_copies * colored_cost) / (total_copies - colored_copies)

/- Theorem statement -/
theorem white_copy_cost_is_five_cents : white_copy_cost = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_white_copy_cost_is_five_cents_l2495_249576


namespace NUMINAMATH_CALUDE_shoe_matching_probability_l2495_249522

/-- Represents the number of pairs of shoes for each color -/
structure ShoeInventory :=
  (black : ℕ)
  (brown : ℕ)
  (gray : ℕ)
  (red : ℕ)

/-- Calculates the probability of picking a matching pair of different feet -/
def matchingProbability (inventory : ShoeInventory) : ℚ :=
  let totalShoes := 2 * (inventory.black + inventory.brown + inventory.gray + inventory.red)
  let matchingPairs := 
    inventory.black * (inventory.black - 1) +
    inventory.brown * (inventory.brown - 1) +
    inventory.gray * (inventory.gray - 1) +
    inventory.red * (inventory.red - 1)
  ↑matchingPairs / (totalShoes * (totalShoes - 1))

theorem shoe_matching_probability (inventory : ShoeInventory) :
  inventory.black = 8 ∧ 
  inventory.brown = 4 ∧ 
  inventory.gray = 3 ∧ 
  inventory.red = 2 →
  matchingProbability inventory = 93 / 551 :=
by sorry

end NUMINAMATH_CALUDE_shoe_matching_probability_l2495_249522


namespace NUMINAMATH_CALUDE_largest_fraction_add_to_one_seventh_l2495_249525

theorem largest_fraction_add_to_one_seventh :
  ∀ (a b : ℕ) (hb : 0 < b) (hb_lt_5 : b < 5),
    (1 : ℚ) / 7 + (a : ℚ) / b < 1 →
    (a : ℚ) / b ≤ 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_add_to_one_seventh_l2495_249525


namespace NUMINAMATH_CALUDE_negation_of_proposition_P_l2495_249553

theorem negation_of_proposition_P :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_P_l2495_249553


namespace NUMINAMATH_CALUDE_contradiction_assumption_l2495_249558

theorem contradiction_assumption (x y z : ℝ) :
  (¬ (x > 0 ∨ y > 0 ∨ z > 0)) ↔ (x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l2495_249558


namespace NUMINAMATH_CALUDE_sallys_initial_cards_l2495_249559

/-- Proves that Sally initially had 27 Pokemon cards given the problem conditions --/
theorem sallys_initial_cards (x : ℕ) : x + 20 = 41 + 6 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sallys_initial_cards_l2495_249559


namespace NUMINAMATH_CALUDE_polynomial_gp_roots_condition_l2495_249567

/-- A polynomial with coefficients a, b, and c -/
def polynomial (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Condition for three distinct real roots in geometric progression -/
def has_three_distinct_real_roots_in_gp (a b c : ℝ) : Prop :=
  ∃ x y z : ℝ, 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    polynomial a b c x = 0 ∧
    polynomial a b c y = 0 ∧
    polynomial a b c z = 0 ∧
    ∃ r : ℝ, r ≠ 0 ∧ y = x * r ∧ z = y * r

/-- Theorem stating the conditions on coefficients a, b, and c -/
theorem polynomial_gp_roots_condition (a b c : ℝ) :
  has_three_distinct_real_roots_in_gp a b c ↔ 
    a^3 * c = b^3 ∧ -a^2 < b ∧ b < a^2 / 3 :=
sorry

end NUMINAMATH_CALUDE_polynomial_gp_roots_condition_l2495_249567


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l2495_249514

theorem quadratic_equation_from_means (η ζ : ℝ) 
  (h_arithmetic_mean : (η + ζ) / 2 = 7)
  (h_geometric_mean : Real.sqrt (η * ζ) = 8) :
  ∀ x : ℝ, x^2 - 14*x + 64 = 0 ↔ (x = η ∨ x = ζ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l2495_249514


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2495_249597

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (5 * x + 15) = 15) ∧ (x = 42) := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2495_249597


namespace NUMINAMATH_CALUDE_max_intersection_points_l2495_249515

-- Define the geometric objects
def Circle : Type := Unit
def Ellipse : Type := Unit
def Line : Type := Unit

-- Define the intersection function
def intersection_points (c : Circle) (e : Ellipse) (l : Line) : ℕ := sorry

-- Theorem statement
theorem max_intersection_points :
  ∃ (c : Circle) (e : Ellipse) (l : Line),
    ∀ (c' : Circle) (e' : Ellipse) (l' : Line),
      intersection_points c e l ≥ intersection_points c' e' l' ∧
      intersection_points c e l = 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_l2495_249515


namespace NUMINAMATH_CALUDE_handshakes_in_gathering_l2495_249578

/-- The number of handshakes in a gathering with specific conditions -/
def number_of_handshakes (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: In a gathering of 8 married couples with specific handshake rules, there are 104 handshakes -/
theorem handshakes_in_gathering : number_of_handshakes 16 = 104 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_in_gathering_l2495_249578


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2495_249573

theorem arithmetic_expression_equality : 8 + 12 / 3 - 2^3 + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2495_249573


namespace NUMINAMATH_CALUDE_sqrt_gt_sufficient_not_necessary_for_exp_gt_l2495_249564

theorem sqrt_gt_sufficient_not_necessary_for_exp_gt (a b : ℝ) :
  (∀ a b : ℝ, Real.sqrt a > Real.sqrt b → Real.exp a > Real.exp b) ∧
  ¬(∀ a b : ℝ, Real.exp a > Real.exp b → Real.sqrt a > Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_gt_sufficient_not_necessary_for_exp_gt_l2495_249564


namespace NUMINAMATH_CALUDE_solution_set_solves_inequality_l2495_249542

/-- The solution set of the inequality 12x^2 - ax > a^2 for a given real number a -/
def solution_set (a : ℝ) : Set ℝ :=
  if a > 0 then {x | x < -a/4 ∨ x > a/3}
  else if a = 0 then {x | x ≠ 0}
  else {x | x < a/3 ∨ x > -a/4}

/-- Theorem stating that the solution_set function correctly solves the inequality -/
theorem solution_set_solves_inequality (a : ℝ) :
  ∀ x, x ∈ solution_set a ↔ 12 * x^2 - a * x > a^2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_solves_inequality_l2495_249542


namespace NUMINAMATH_CALUDE_tourist_distance_l2495_249517

theorem tourist_distance (x : ℝ) : 
  1 + (1/2) * (x - 1) + (1/3) * x + 1 = x ↔ x = 9 :=
by sorry

end NUMINAMATH_CALUDE_tourist_distance_l2495_249517


namespace NUMINAMATH_CALUDE_sqrt_expressions_simplification_l2495_249579

theorem sqrt_expressions_simplification :
  (∀ (x y : ℝ), x > 0 → y > 0 → (Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y)) →
  (Real.sqrt 45 + Real.sqrt 50) - (Real.sqrt 18 - Real.sqrt 20) = 5 * Real.sqrt 5 + 2 * Real.sqrt 2 ∧
  Real.sqrt 24 / (6 * Real.sqrt (1/6)) - Real.sqrt 12 * (Real.sqrt 3 / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_simplification_l2495_249579


namespace NUMINAMATH_CALUDE_seats_taken_l2495_249560

theorem seats_taken (rows : ℕ) (chairs_per_row : ℕ) (unoccupied : ℕ) : 
  rows = 40 → chairs_per_row = 20 → unoccupied = 10 → 
  rows * chairs_per_row - unoccupied = 790 := by
  sorry

end NUMINAMATH_CALUDE_seats_taken_l2495_249560


namespace NUMINAMATH_CALUDE_students_on_pullout_couch_l2495_249547

theorem students_on_pullout_couch (total_students : ℕ) (num_rooms : ℕ) (students_per_bed : ℕ) (beds_per_room : ℕ) :
  total_students = 30 →
  num_rooms = 6 →
  students_per_bed = 2 →
  beds_per_room = 2 →
  (total_students / num_rooms - students_per_bed * beds_per_room : ℕ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_students_on_pullout_couch_l2495_249547


namespace NUMINAMATH_CALUDE_b_initial_investment_l2495_249569

/-- Represents the business scenario with two partners A and B -/
structure BusinessScenario where
  a_initial : ℕ  -- A's initial investment
  b_initial : ℕ  -- B's initial investment (unknown)
  a_withdraw : ℕ  -- Amount A withdraws after 8 months
  b_add : ℕ  -- Amount B adds after 8 months
  total_profit : ℕ  -- Total profit at the end of the year
  a_profit : ℕ  -- A's share of the profit

/-- Calculates the investment value for a partner -/
def investment_value (initial : ℕ) (change : ℕ) (is_withdraw : Bool) : ℕ :=
  if is_withdraw then
    8 * initial + 4 * (initial - change)
  else
    8 * initial + 4 * (initial + change)

/-- Theorem stating that B's initial investment was 4000 -/
theorem b_initial_investment
  (scenario : BusinessScenario)
  (h1 : scenario.a_initial = 6000)
  (h2 : scenario.a_withdraw = 1000)
  (h3 : scenario.b_add = 1000)
  (h4 : scenario.total_profit = 630)
  (h5 : scenario.a_profit = 357)
  : scenario.b_initial = 4000 := by
  sorry

end NUMINAMATH_CALUDE_b_initial_investment_l2495_249569
