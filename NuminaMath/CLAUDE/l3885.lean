import Mathlib

namespace NUMINAMATH_CALUDE_christmas_tree_lights_l3885_388555

theorem christmas_tree_lights (total : ℕ) (red : ℕ) (yellow : ℕ) (green : ℕ) 
  (h1 : total = 350) (h2 : red = 85) (h3 : yellow = 112) (h4 : green = 65) :
  total - (red + yellow + green) = 88 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l3885_388555


namespace NUMINAMATH_CALUDE_vanilla_jelly_beans_count_l3885_388587

theorem vanilla_jelly_beans_count :
  ∀ (vanilla grape : ℕ),
    grape = 5 * vanilla + 50 →
    vanilla + grape = 770 →
    vanilla = 120 := by
  sorry

end NUMINAMATH_CALUDE_vanilla_jelly_beans_count_l3885_388587


namespace NUMINAMATH_CALUDE_cold_virus_diameter_scientific_notation_l3885_388583

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem cold_virus_diameter_scientific_notation :
  to_scientific_notation 0.00000036 = ScientificNotation.mk 3.6 (-7) sorry := by
  sorry

end NUMINAMATH_CALUDE_cold_virus_diameter_scientific_notation_l3885_388583


namespace NUMINAMATH_CALUDE_granola_initial_price_l3885_388530

/-- Proves that the initial selling price per bag was $6.00 --/
theorem granola_initial_price (ingredient_cost : ℝ) (total_bags : ℕ) 
  (full_price_sold : ℕ) (discount_price : ℝ) (net_profit : ℝ) :
  ingredient_cost = 3 →
  total_bags = 20 →
  full_price_sold = 15 →
  discount_price = 4 →
  net_profit = 50 →
  ∃ (initial_price : ℝ), 
    initial_price * full_price_sold + discount_price * (total_bags - full_price_sold) - 
    ingredient_cost * total_bags = net_profit ∧
    initial_price = 6 := by
  sorry

#check granola_initial_price

end NUMINAMATH_CALUDE_granola_initial_price_l3885_388530


namespace NUMINAMATH_CALUDE_cafeteria_vertical_stripes_l3885_388540

def cafeteria_problem (total : ℕ) (checkered : ℕ) (horizontal_multiplier : ℕ) : Prop :=
  let stripes : ℕ := total - checkered
  let horizontal : ℕ := horizontal_multiplier * checkered
  let vertical : ℕ := stripes - horizontal
  vertical = 5

theorem cafeteria_vertical_stripes :
  cafeteria_problem 40 7 4 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_vertical_stripes_l3885_388540


namespace NUMINAMATH_CALUDE_workout_ratio_theorem_l3885_388594

/-- Represents the workout schedule for Rayman, Junior, and Wolverine -/
structure WorkoutSchedule where
  junior_hours : ℝ
  rayman_hours : ℝ
  wolverine_hours : ℝ
  ratio : ℝ

/-- Theorem stating the relationship between workout hours -/
theorem workout_ratio_theorem (w : WorkoutSchedule) 
  (h1 : w.rayman_hours = w.junior_hours / 2)
  (h2 : w.wolverine_hours = 60)
  (h3 : w.wolverine_hours = w.ratio * (w.rayman_hours + w.junior_hours)) :
  w.ratio = 40 / w.junior_hours :=
sorry

end NUMINAMATH_CALUDE_workout_ratio_theorem_l3885_388594


namespace NUMINAMATH_CALUDE_ages_sum_after_three_years_l3885_388578

theorem ages_sum_after_three_years 
  (ava_age bob_age carlo_age : ℕ) 
  (h : ava_age + bob_age + carlo_age = 31) : 
  (ava_age + 3) + (bob_age + 3) + (carlo_age + 3) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_after_three_years_l3885_388578


namespace NUMINAMATH_CALUDE_sum_coordinates_point_D_l3885_388552

/-- Given a point N which is the midpoint of segment CD, and point C,
    prove that the sum of coordinates of point D is 5. -/
theorem sum_coordinates_point_D (N C D : ℝ × ℝ) : 
  N = (3, 5) →
  C = (1, 10) →
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_point_D_l3885_388552


namespace NUMINAMATH_CALUDE_total_liquid_proof_l3885_388504

/-- The amount of oil used in cups -/
def oil_amount : ℝ := 0.17

/-- The amount of water used in cups -/
def water_amount : ℝ := 1.17

/-- The total amount of liquid used in cups -/
def total_liquid : ℝ := oil_amount + water_amount

theorem total_liquid_proof : total_liquid = 1.34 := by
  sorry

end NUMINAMATH_CALUDE_total_liquid_proof_l3885_388504


namespace NUMINAMATH_CALUDE_age_difference_l3885_388503

/-- Proves the age difference between a man and his son --/
theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 22 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 24 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3885_388503


namespace NUMINAMATH_CALUDE_teacher_remaining_budget_l3885_388548

/-- Calculates the remaining budget for a teacher after purchasing school supplies. -/
theorem teacher_remaining_budget 
  (last_year_remaining : ℕ) 
  (this_year_budget : ℕ) 
  (first_purchase : ℕ) 
  (second_purchase : ℕ) 
  (h1 : last_year_remaining = 6)
  (h2 : this_year_budget = 50)
  (h3 : first_purchase = 13)
  (h4 : second_purchase = 24) :
  last_year_remaining + this_year_budget - (first_purchase + second_purchase) = 19 :=
by sorry

end NUMINAMATH_CALUDE_teacher_remaining_budget_l3885_388548


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3885_388582

/-- A function that returns the digits of a three-digit number -/
def digits (n : ℕ) : Fin 3 → ℕ :=
  fun i => (n / (100 / 10^i.val)) % 10

/-- Check if three numbers form a geometric progression -/
def isGeometricProgression (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Check if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℕ) : Prop :=
  2 * b = a + c

/-- The main theorem -/
theorem unique_three_digit_number : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  (let d := digits n
   isGeometricProgression (d 0) (d 1) (d 2) ∧
   d 0 ≠ d 1 ∧ d 1 ≠ d 2 ∧ d 0 ≠ d 2) ∧
  (let m := n - 200
   100 ≤ m ∧ m < 1000 ∧
   let d := digits m
   isArithmeticProgression (d 0) (d 1) (d 2)) ∧
  n = 842 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3885_388582


namespace NUMINAMATH_CALUDE_eighth_hexagonal_number_l3885_388562

/-- Definition of hexagonal numbers -/
def hexagonal (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The 8th hexagonal number is 120 -/
theorem eighth_hexagonal_number : hexagonal 8 = 120 := by
  sorry

end NUMINAMATH_CALUDE_eighth_hexagonal_number_l3885_388562


namespace NUMINAMATH_CALUDE_final_answer_calculation_l3885_388526

theorem final_answer_calculation (chosen_number : ℤ) (h : chosen_number = 848) : 
  (chosen_number / 8 : ℚ) - 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_final_answer_calculation_l3885_388526


namespace NUMINAMATH_CALUDE_negation_equivalence_l3885_388590

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 2 * x^2 < Real.cos x) ↔ (∀ x : ℝ, 2 * x^2 ≥ Real.cos x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3885_388590


namespace NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l3885_388557

theorem arithmetic_progression_common_difference 
  (a₁ : ℝ) (a₂₁ : ℝ) (d : ℝ) :
  a₁ = 3 → a₂₁ = 103 → a₂₁ = a₁ + 20 * d → d = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l3885_388557


namespace NUMINAMATH_CALUDE_xy_equals_two_l3885_388550

theorem xy_equals_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h : x^2 + 2/x = y + 2/y) : x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_two_l3885_388550


namespace NUMINAMATH_CALUDE_n_is_even_l3885_388569

-- Define a type for points in space
def Point : Type := ℝ × ℝ × ℝ

-- Define a function to check if four points are coplanar
def are_coplanar (p q r s : Point) : Prop := sorry

-- Define a function to check if a point is inside a tetrahedron
def is_interior_point (p q r s t : Point) : Prop := sorry

-- Define the main theorem
theorem n_is_even (n : ℕ) (P : Fin n → Point) (Q : Point) :
  (∀ (i j k l : Fin n), i ≠ j → j ≠ k → k ≠ l → i ≠ k → j ≠ l → i ≠ l → 
    ¬ are_coplanar (P i) (P j) (P k) (P l)) →
  (∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    ∃ (l : Fin n), l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ 
      is_interior_point Q (P i) (P j) (P k) (P l)) →
  Even n := by
  sorry

end NUMINAMATH_CALUDE_n_is_even_l3885_388569


namespace NUMINAMATH_CALUDE_set_intersection_equality_l3885_388512

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2^x - 1)}

-- Define set B
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

-- Define the intersection set
def intersection_set : Set ℝ := {x | 0 ≤ x ∧ x < 2}

-- Theorem statement
theorem set_intersection_equality : A ∩ B = intersection_set := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l3885_388512


namespace NUMINAMATH_CALUDE_parabola_transformation_l3885_388508

-- Define the original function
def f (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the transformation
def transform (g : ℝ → ℝ) : ℝ → ℝ := λ x => g (x - 1) + 2

-- Define the expected result function
def expected_result (x : ℝ) : ℝ := (x - 4)^2 - 2

-- Theorem statement
theorem parabola_transformation :
  ∀ x, transform f x = expected_result x :=
sorry

end NUMINAMATH_CALUDE_parabola_transformation_l3885_388508


namespace NUMINAMATH_CALUDE_all_triangles_in_S_are_similar_l3885_388545

-- Define a structure for triangles in set S
structure TriangleS where
  A : Real
  B : Real
  C : Real
  tan_A_pos_int : ℕ+
  tan_B_pos_int : ℕ+
  tan_C_pos_int : ℕ+
  angle_sum : A + B + C = Real.pi
  tan_sum_identity : Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C

-- Define similarity for triangles in S
def similar (t1 t2 : TriangleS) : Prop :=
  ∃ (k : Real), k > 0 ∧
    t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

-- State the theorem
theorem all_triangles_in_S_are_similar (t1 t2 : TriangleS) :
  similar t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_all_triangles_in_S_are_similar_l3885_388545


namespace NUMINAMATH_CALUDE_ellipse_equation_l3885_388500

/-- Given a circle and an ellipse with specific properties, prove the equation of the ellipse -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → 
    (x^2 / a^2 + y^2 / b^2 = 1 ∧ 
     ((x = 0 ∧ y = b) ∨ (x = 0 ∧ y = -b) ∨ 
      (y = 0 ∧ x^2 = a^2 - b^2) ∨ (y = 0 ∧ x^2 = a^2 - b^2)))) →
  a^2 = 8 ∧ b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3885_388500


namespace NUMINAMATH_CALUDE_poor_people_distribution_l3885_388589

theorem poor_people_distribution (x : ℕ) : 
  (120 / (x - 10) - 120 / x = 120 / x - 120 / (x + 20)) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_poor_people_distribution_l3885_388589


namespace NUMINAMATH_CALUDE_line_equation_transformation_l3885_388505

/-- Given a line L with equation y = (2/3)x + 4, prove that a line M with twice the slope
    and half the y-intercept of L has the equation y = (4/3)x + 2 -/
theorem line_equation_transformation (x y : ℝ) :
  let L : ℝ → ℝ := λ x => (2/3) * x + 4
  let M : ℝ → ℝ := λ x => (4/3) * x + 2
  (∀ x, M x = 2 * ((2/3) * x) + (1/2) * 4) → (∀ x, M x = (4/3) * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_transformation_l3885_388505


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3885_388517

theorem inequality_solution_set (x : ℝ) : 
  (x - 20) / (x + 16) ≤ 0 ↔ -16 < x ∧ x ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3885_388517


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l3885_388511

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 5 / Real.log 30 + 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l3885_388511


namespace NUMINAMATH_CALUDE_correct_selection_methods_l3885_388574

/-- The number of members in the class committee -/
def total_members : ℕ := 5

/-- The number of roles to be filled -/
def roles_to_fill : ℕ := 3

/-- The number of members who cannot serve as the entertainment officer -/
def restricted_members : ℕ := 2

/-- The number of different selection methods -/
def selection_methods : ℕ := 36

/-- Theorem stating that the number of selection methods is correct -/
theorem correct_selection_methods :
  (total_members - restricted_members) * (total_members - 1) * (total_members - 2) = selection_methods :=
by sorry

end NUMINAMATH_CALUDE_correct_selection_methods_l3885_388574


namespace NUMINAMATH_CALUDE_margarets_mean_score_l3885_388588

def scores : List ℝ := [82, 85, 88, 90, 95, 97, 98, 100]

theorem margarets_mean_score 
  (h1 : scores.length = 8)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        margaret_scores.length = 4 ∧ 
        cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        cyprian_scores.sum / cyprian_scores.length = 91) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 4 ∧ 
    margaret_scores.sum / margaret_scores.length = 92.75 := by
  sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l3885_388588


namespace NUMINAMATH_CALUDE_only_subtraction_correct_l3885_388580

theorem only_subtraction_correct : 
  (¬(Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2)) ∧ 
  (¬(3 - 27/64 = 3/4)) ∧ 
  (3 - 8 = -5) ∧ 
  (¬(|Real.sqrt 2 - 1| = 1 - Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_only_subtraction_correct_l3885_388580


namespace NUMINAMATH_CALUDE_count_even_one_matrices_l3885_388553

/-- The number of m × n matrices with entries 0 or 1, where the number of 1's in each row and column is even -/
def evenOneMatrices (m n : ℕ) : ℕ :=
  2^((m-1)*(n-1))

/-- Theorem stating that the number of m × n matrices with entries 0 or 1, 
    where the number of 1's in each row and column is even, is 2^((m-1)(n-1)) -/
theorem count_even_one_matrices (m n : ℕ) :
  evenOneMatrices m n = 2^((m-1)*(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_count_even_one_matrices_l3885_388553


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l3885_388586

theorem sum_of_fifth_powers (n : ℕ) : 
  (∃ (A B C D E : ℤ), n = A^5 + B^5 + C^5 + D^5 + E^5) ∧ 
  (¬ ∃ (A B C D : ℤ), n = A^5 + B^5 + C^5 + D^5) := by
  sorry

#check sum_of_fifth_powers 2018

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l3885_388586


namespace NUMINAMATH_CALUDE_product_sum_6936_l3885_388531

theorem product_sum_6936 : ∃ a b : ℕ, 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 6936 ∧ 
  a + b = 168 := by
sorry

end NUMINAMATH_CALUDE_product_sum_6936_l3885_388531


namespace NUMINAMATH_CALUDE_greatest_x_satisfying_equation_l3885_388596

theorem greatest_x_satisfying_equation : 
  ∃ (x : ℝ), x = -3 ∧ 
  (∀ y : ℝ, y ≠ 6 → y ≠ -4 → (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≤ x) ∧
  (x^2 - x - 30) / (x - 6) = 2 / (x + 4) ∧
  x ≠ 6 ∧ x ≠ -4 := by
sorry

end NUMINAMATH_CALUDE_greatest_x_satisfying_equation_l3885_388596


namespace NUMINAMATH_CALUDE_telescope_visual_range_increase_l3885_388538

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 80)
  (h2 : new_range = 150) :
  ((new_range - original_range) / original_range) * 100 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_increase_l3885_388538


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l3885_388581

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex polygon with 9 sides has 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l3885_388581


namespace NUMINAMATH_CALUDE_problem_solution_l3885_388563

theorem problem_solution (a : ℝ) (h : a^2 - 2*a - 1 = 0) : -3*a^2 + 6*a + 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3885_388563


namespace NUMINAMATH_CALUDE_parent_current_age_l3885_388575

-- Define the son's age next year
def sons_age_next_year : ℕ := 8

-- Define the relation between parent's and son's age
def parent_age_relation (parent_age son_age : ℕ) : Prop :=
  parent_age = 5 * son_age

-- Theorem to prove
theorem parent_current_age : 
  ∃ (parent_age : ℕ), parent_age_relation parent_age (sons_age_next_year - 1) ∧ parent_age = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_parent_current_age_l3885_388575


namespace NUMINAMATH_CALUDE_trent_tears_for_soup_l3885_388564

/-- Calculates the number of tears Trent cries when chopping onions for soup -/
theorem trent_tears_for_soup (tears_per_set : ℕ) (onions_per_set : ℕ) (onions_per_pot : ℕ) (num_pots : ℕ) : 
  tears_per_set * (onions_per_pot * num_pots / onions_per_set) = 16 :=
by
  sorry

#check trent_tears_for_soup 2 3 4 6

end NUMINAMATH_CALUDE_trent_tears_for_soup_l3885_388564


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3885_388521

def M : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (-1, 1) + x • (1, 2)}
def N : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (1, -2) + x • (2, 3)}

theorem intersection_of_M_and_N :
  M ∩ N = {(-13, -23)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3885_388521


namespace NUMINAMATH_CALUDE_valentines_theorem_l3885_388573

/-- The number of Valentines Mrs. Franklin initially had -/
def initial_valentines : ℕ := 58

/-- The number of additional Valentines Mrs. Franklin needs -/
def additional_valentines : ℕ := 16

/-- The number of students Mrs. Franklin has -/
def num_students : ℕ := 74

/-- Theorem stating that the initial number of Valentines plus the additional Valentines
    equals the total number of students -/
theorem valentines_theorem :
  initial_valentines + additional_valentines = num_students :=
by sorry

end NUMINAMATH_CALUDE_valentines_theorem_l3885_388573


namespace NUMINAMATH_CALUDE_least_subtrahend_l3885_388532

theorem least_subtrahend (n : Nat) : 
  (∀ (d : Nat), d ∈ [17, 19, 23] → (997 - n) % d = 3) →
  (∀ (m : Nat), m < n → ∃ (d : Nat), d ∈ [17, 19, 23] ∧ (997 - m) % d ≠ 3) →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_least_subtrahend_l3885_388532


namespace NUMINAMATH_CALUDE_sin_2x_eq_sin_x_solution_l3885_388554

open Set
open Real

def solution_set : Set ℝ := {0, π, -π/3, π/3, 5*π/3}

theorem sin_2x_eq_sin_x_solution :
  {x : ℝ | x ∈ Ioo (-π) (2*π) ∧ sin (2*x) = sin x} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_eq_sin_x_solution_l3885_388554


namespace NUMINAMATH_CALUDE_nonagon_dissection_l3885_388593

/-- Represents a rhombus with unit side length and a specific angle -/
structure Rhombus :=
  (angle : ℝ)

/-- Represents an isosceles triangle with unit side length and a specific vertex angle -/
structure IsoscelesTriangle :=
  (vertex_angle : ℝ)

/-- Represents a regular polygon with a specific number of sides -/
structure RegularPolygon :=
  (sides : ℕ)

/-- The original 9-gon composed of specific shapes -/
def original_nonagon : RegularPolygon :=
  { sides := 9 }

/-- The set of rhombuses with 40° angles -/
def rhombuses_40 : Finset Rhombus :=
  sorry

/-- The set of rhombuses with 80° angles -/
def rhombuses_80 : Finset Rhombus :=
  sorry

/-- The set of isosceles triangles with 120° vertex angles -/
def triangles_120 : Finset IsoscelesTriangle :=
  sorry

/-- Represents the dissection of the original nonagon into three congruent regular nonagons -/
def dissection (original : RegularPolygon) (parts : Finset RegularPolygon) : Prop :=
  sorry

/-- The theorem stating that the original nonagon can be dissected into three congruent regular nonagons -/
theorem nonagon_dissection :
  ∃ (parts : Finset RegularPolygon),
    (parts.card = 3) ∧
    (∀ p ∈ parts, p.sides = 9) ∧
    (dissection original_nonagon parts) :=
sorry

end NUMINAMATH_CALUDE_nonagon_dissection_l3885_388593


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_product_l3885_388565

/-- A geometric sequence with common ratio r -/
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum_product (a : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  a 4 + a 8 = -2 →
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_product_l3885_388565


namespace NUMINAMATH_CALUDE_cricket_player_innings_l3885_388510

theorem cricket_player_innings 
  (average : ℝ) 
  (next_innings_runs : ℝ) 
  (average_increase : ℝ) 
  (h1 : average = 33) 
  (h2 : next_innings_runs = 77) 
  (h3 : average_increase = 4) :
  ∃ n : ℕ, 
    (n : ℝ) * average + next_innings_runs = (n + 1) * (average + average_increase) ∧ 
    n = 10 :=
by sorry

end NUMINAMATH_CALUDE_cricket_player_innings_l3885_388510


namespace NUMINAMATH_CALUDE_solve_system_l3885_388516

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 8) (eq2 : x + 3 * y = 7) : x = 38 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3885_388516


namespace NUMINAMATH_CALUDE_cafe_tables_l3885_388529

-- Define the seating capacity in base 8
def seating_capacity_base8 : ℕ := 312

-- Define the number of people per table
def people_per_table : ℕ := 3

-- Define the function to convert from base 8 to base 10
def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Theorem statement
theorem cafe_tables :
  (base8_to_base10 seating_capacity_base8) / people_per_table = 67 := by
  sorry

end NUMINAMATH_CALUDE_cafe_tables_l3885_388529


namespace NUMINAMATH_CALUDE_alex_trips_l3885_388597

def savings : ℝ := 14500
def car_cost : ℝ := 14600
def trip_charge : ℝ := 1.5
def grocery_percentage : ℝ := 0.05
def grocery_value : ℝ := 800

def earnings_per_trip : ℝ := trip_charge + grocery_percentage * grocery_value

theorem alex_trips : 
  ∃ n : ℕ, (n : ℝ) * earnings_per_trip ≥ car_cost - savings ∧ 
  ∀ m : ℕ, (m : ℝ) * earnings_per_trip ≥ car_cost - savings → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_alex_trips_l3885_388597


namespace NUMINAMATH_CALUDE_syllogism_form_is_correct_l3885_388592

-- Define deductive reasoning
structure DeductiveReasoning where
  general_to_specific : Bool
  syllogism_form : Bool
  conclusion_correctness : Bool
  conclusion_depends_on_premises : Bool

-- Define the correct properties of deductive reasoning
def correct_deductive_reasoning : DeductiveReasoning :=
  { general_to_specific := true,
    syllogism_form := true,
    conclusion_correctness := false,
    conclusion_depends_on_premises := true }

-- Theorem to prove
theorem syllogism_form_is_correct (dr : DeductiveReasoning) :
  dr = correct_deductive_reasoning → dr.syllogism_form = true :=
by sorry

end NUMINAMATH_CALUDE_syllogism_form_is_correct_l3885_388592


namespace NUMINAMATH_CALUDE_negative_correlation_implies_negative_slope_l3885_388546

/-- A linear regression model with slope b and intercept a -/
structure LinearRegression where
  b : ℝ
  a : ℝ

/-- Defines a negative correlation between two variables -/
def NegativeCorrelation (model : LinearRegression) : Prop :=
  model.b < 0

theorem negative_correlation_implies_negative_slope
  (model : LinearRegression)
  (h1 : ∃ (x y : ℝ → ℝ), ∀ t, y t = model.b * (x t) + model.a)
  (h2 : NegativeCorrelation model) :
  model.b < 0 :=
sorry

end NUMINAMATH_CALUDE_negative_correlation_implies_negative_slope_l3885_388546


namespace NUMINAMATH_CALUDE_complex_rational_sum_l3885_388501

def separate_and_sum (a b c d : ℚ) : ℚ :=
  let int_part := a.floor + b.floor + c.floor + d.floor
  let frac_part := (a - a.floor) + (b - b.floor) + (c - c.floor) + (d - d.floor)
  int_part + frac_part

theorem complex_rational_sum :
  separate_and_sum (-206) (401 + 3/4) (-204 - 2/3) (-1 - 1/2) = -10 - 5/12 :=
by sorry

end NUMINAMATH_CALUDE_complex_rational_sum_l3885_388501


namespace NUMINAMATH_CALUDE_table_tennis_choices_l3885_388506

theorem table_tennis_choices (rackets balls nets : ℕ) 
  (h_rackets : rackets = 7)
  (h_balls : balls = 7)
  (h_nets : nets = 3) :
  rackets * balls * nets = 147 := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_choices_l3885_388506


namespace NUMINAMATH_CALUDE_min_value_theorem_l3885_388507

/-- Given positive real numbers x and y satisfying x + y = 1,
    if the minimum value of 1/x + a/y is 9, then a = 4 -/
theorem min_value_theorem (x y a : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1)
    (hmin : ∀ (u v : ℝ), u > 0 → v > 0 → u + v = 1 → 1/u + a/v ≥ 9) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3885_388507


namespace NUMINAMATH_CALUDE_true_discount_calculation_l3885_388591

/-- Calculates the true discount given the banker's discount and present value -/
def true_discount (bankers_discount : ℚ) (present_value : ℚ) : ℚ :=
  bankers_discount / (1 + bankers_discount / present_value)

/-- Theorem stating that given a banker's discount of 36 and a present value of 180, 
    the true discount is 30 -/
theorem true_discount_calculation :
  true_discount 36 180 = 30 := by
  sorry

#eval true_discount 36 180

end NUMINAMATH_CALUDE_true_discount_calculation_l3885_388591


namespace NUMINAMATH_CALUDE_percentage_of_m1_products_l3885_388515

theorem percentage_of_m1_products (m1_defective : Real) (m2_defective : Real)
  (m3_non_defective : Real) (m2_percentage : Real) (total_defective : Real) :
  m1_defective = 0.03 →
  m2_defective = 0.01 →
  m3_non_defective = 0.93 →
  m2_percentage = 0.3 →
  total_defective = 0.036 →
  ∃ (m1_percentage : Real),
    m1_percentage = 0.4 ∧
    m1_percentage + m2_percentage + (1 - m1_percentage - m2_percentage) = 1 ∧
    m1_percentage * m1_defective +
    m2_percentage * m2_defective +
    (1 - m1_percentage - m2_percentage) * (1 - m3_non_defective) = total_defective :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_m1_products_l3885_388515


namespace NUMINAMATH_CALUDE_common_root_theorem_l3885_388559

theorem common_root_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0) ∧
  (∃ y : ℝ, b * y^11 + c * y^4 + a = 0 ∧ c * y^11 + a * y^4 + b = 0) ∧
  (∃ z : ℝ, c * z^11 + a * z^4 + b = 0 ∧ a * z^11 + b * z^4 + c = 0) →
  ∃ w : ℝ, a * w^11 + b * w^4 + c = 0 ∧
           b * w^11 + c * w^4 + a = 0 ∧
           c * w^11 + a * w^4 + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_common_root_theorem_l3885_388559


namespace NUMINAMATH_CALUDE_function_sum_positive_l3885_388541

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x < y → x < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = -f (-x + 2))

-- Define the theorem
theorem function_sum_positive (x₁ x₂ : ℝ) 
  (hx₁ : x₁ < 2) (hx₂ : x₂ > 2) (h : |x₁ - 2| < |x₂ - 2|) :
  f x₁ + f x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_sum_positive_l3885_388541


namespace NUMINAMATH_CALUDE_subset_relation_l3885_388524

theorem subset_relation (A B C : Set α) (h : A ∪ B = B ∩ C) : A ⊆ C := by
  sorry

end NUMINAMATH_CALUDE_subset_relation_l3885_388524


namespace NUMINAMATH_CALUDE_nested_average_equals_25_18_l3885_388584

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- The main theorem to prove -/
theorem nested_average_equals_25_18 :
  avg3 (avg3 2 2 1) (avg2 1 2) 1 = 25 / 18 := by sorry

end NUMINAMATH_CALUDE_nested_average_equals_25_18_l3885_388584


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3885_388551

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set P
def P : Set Nat := {1, 2, 3, 4}

-- Define set Q
def Q : Set Nat := {3, 4, 5}

-- Theorem statement
theorem intersection_with_complement : P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3885_388551


namespace NUMINAMATH_CALUDE_complement_intersection_equality_l3885_388519

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {2, 3, 5}

-- Define set N
def N : Set Nat := {4, 6}

-- Theorem statement
theorem complement_intersection_equality :
  (U \ M) ∩ N = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_equality_l3885_388519


namespace NUMINAMATH_CALUDE_function_inequality_l3885_388509

noncomputable def f (x : ℝ) := x^2 - Real.pi * x

theorem function_inequality (α β γ : ℝ) 
  (h_α : α ∈ Set.Ioo 0 Real.pi) 
  (h_β : β ∈ Set.Ioo 0 Real.pi) 
  (h_γ : γ ∈ Set.Ioo 0 Real.pi)
  (h_sin_α : Real.sin α = 1/3)
  (h_tan_β : Real.tan β = 5/4)
  (h_cos_γ : Real.cos γ = -1/3) :
  f α > f β ∧ f β > f γ := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3885_388509


namespace NUMINAMATH_CALUDE_diagonal_bd_length_l3885_388561

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- The length of base AD -/
  ad : ℝ
  /-- The length of base BC -/
  bc : ℝ
  /-- The length of diagonal AC -/
  ac : ℝ
  /-- The circles on AB, BC, and CD as diameters intersect at one point -/
  circles_intersect : Prop

/-- The theorem about the length of diagonal BD in a special trapezoid -/
theorem diagonal_bd_length (t : SpecialTrapezoid)
    (h_ad : t.ad = 20)
    (h_bc : t.bc = 14)
    (h_ac : t.ac = 16) :
  ∃ (bd : ℝ), bd = 30 ∧ bd * bd = t.ac * t.ac + (t.ad - t.bc) * (t.ad - t.bc) / 4 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_bd_length_l3885_388561


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3885_388514

theorem gcd_of_three_numbers : Nat.gcd 13456 (Nat.gcd 25345 15840) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3885_388514


namespace NUMINAMATH_CALUDE_inequality_proof_l3885_388544

theorem inequality_proof (m : ℕ+) (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (x^(m : ℕ) / ((1 + y) * (1 + z))) + 
  (y^(m : ℕ) / ((1 + x) * (1 + z))) + 
  (z^(m : ℕ) / ((1 + x) * (1 + y))) ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3885_388544


namespace NUMINAMATH_CALUDE_arithmetic_progression_same_digit_sum_l3885_388502

/-- Sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Arithmetic progression with first term a and common difference d -/
def arithmeticProgression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

theorem arithmetic_progression_same_digit_sum (a d : ℕ) :
  ∃ m n : ℕ, m ≠ n ∧ 
    digitSum (arithmeticProgression a d m) = digitSum (arithmeticProgression a d n) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_same_digit_sum_l3885_388502


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l3885_388571

/-- The number of participants who started the national quiz competition -/
def initial_participants : ℕ := 300

/-- The fraction of participants remaining after the first round -/
def first_round_remaining : ℚ := 2/5

/-- The fraction of participants remaining after the second round, relative to those who remained after the first round -/
def second_round_remaining : ℚ := 1/4

/-- The number of participants remaining after the second round -/
def final_participants : ℕ := 30

theorem quiz_competition_participants :
  (↑initial_participants * first_round_remaining * second_round_remaining : ℚ) = ↑final_participants :=
sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l3885_388571


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l3885_388520

/-- Given a function f(x) = (1/2)x^4 - 2x^3 + 3m where x ∈ ℝ, 
    if f(x) + 12 ≥ 0 for all x, then m ≥ 1/2 -/
theorem function_inequality_implies_m_bound (m : ℝ) : 
  (∀ x : ℝ, (1/2) * x^4 - 2 * x^3 + 3 * m + 12 ≥ 0) → m ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l3885_388520


namespace NUMINAMATH_CALUDE_susan_gave_eight_apples_l3885_388534

/-- The number of apples Susan gave to Sean -/
def apples_from_susan (initial_apples final_apples : ℕ) : ℕ :=
  final_apples - initial_apples

/-- Theorem stating that Susan gave Sean 8 apples -/
theorem susan_gave_eight_apples (initial_apples final_apples : ℕ) 
  (h1 : initial_apples = 9)
  (h2 : final_apples = 17) :
  apples_from_susan initial_apples final_apples = 8 := by
  sorry

end NUMINAMATH_CALUDE_susan_gave_eight_apples_l3885_388534


namespace NUMINAMATH_CALUDE_sum_of_f_values_l3885_388585

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2/x) + 1

theorem sum_of_f_values : 
  f (-7) + f (-5) + f (-3) + f (-1) + f 3 + f 5 + f 7 + f 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l3885_388585


namespace NUMINAMATH_CALUDE_log_equality_implies_product_one_l3885_388598

theorem log_equality_implies_product_one (M N : ℝ) 
  (h1 : (Real.log N / Real.log M)^2 = (Real.log M / Real.log N)^2)
  (h2 : M ≠ N)
  (h3 : M * N > 0)
  (h4 : M ≠ 1)
  (h5 : N ≠ 1) :
  M * N = 1 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_product_one_l3885_388598


namespace NUMINAMATH_CALUDE_jerry_action_figures_l3885_388537

theorem jerry_action_figures (initial_books initial_figures added_figures : ℕ) :
  initial_books = 3 →
  initial_figures = 4 →
  initial_books + 3 = initial_figures + added_figures →
  added_figures = 2 := by
sorry

end NUMINAMATH_CALUDE_jerry_action_figures_l3885_388537


namespace NUMINAMATH_CALUDE_plane_equation_correct_l3885_388528

def plane_equation (x y z : ℝ) : ℝ := 3 * x - y + 2 * z - 11

theorem plane_equation_correct :
  ∃ (A B C D : ℤ),
    (∀ (s t : ℝ),
      plane_equation (2 + 2*s - 2*t) (3 - 2*s) (4 - s + 3*t) = 0) ∧
    A > 0 ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    ∀ (x y z : ℝ), A * x + B * y + C * z + D = plane_equation x y z := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l3885_388528


namespace NUMINAMATH_CALUDE_smallest_value_of_expression_l3885_388522

theorem smallest_value_of_expression (p q t : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime t →
  p < q → q < t → p ≠ q → q ≠ t → p ≠ t →
  (∀ p' q' t' : ℕ, Nat.Prime p' → Nat.Prime q' → Nat.Prime t' →
    p' < q' → q' < t' → p' ≠ q' → q' ≠ t' → p' ≠ t' →
    p' * q' * t' + p' * t' + q' * t' + q' * t' ≥ p * q * t + p * t + q * t + q * t) →
  p * q * t + p * t + q * t + q * t = 70 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_expression_l3885_388522


namespace NUMINAMATH_CALUDE_triangle_area_rational_l3885_388549

/-- Represents a point with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- Condition that the absolute difference between coordinates is at most 2 -/
def coordDiffAtMostTwo (p q : IntPoint) : Prop :=
  (abs (p.x - q.x) ≤ 2) ∧ (abs (p.y - q.y) ≤ 2)

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : IntPoint) : ℚ :=
  (1 / 2 : ℚ) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Theorem stating that the area of a triangle with integer coordinates and limited coordinate differences is rational -/
theorem triangle_area_rational (p1 p2 p3 : IntPoint) 
  (h12 : coordDiffAtMostTwo p1 p2)
  (h23 : coordDiffAtMostTwo p2 p3)
  (h31 : coordDiffAtMostTwo p3 p1) :
  ∃ (q : ℚ), triangleArea p1 p2 p3 = q := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_rational_l3885_388549


namespace NUMINAMATH_CALUDE_trig_identity_l3885_388572

theorem trig_identity : 
  Real.sin (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (138 * π / 180) * Real.cos (72 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l3885_388572


namespace NUMINAMATH_CALUDE_custom_mul_four_three_l3885_388525

/-- Custom multiplication operation -/
def custom_mul (a b : ℤ) : ℤ := a^2 + a*b + a - b^2

/-- Theorem stating that 4 * 3 = 23 under the custom multiplication -/
theorem custom_mul_four_three : custom_mul 4 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_four_three_l3885_388525


namespace NUMINAMATH_CALUDE_share_distribution_l3885_388533

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 578 →
  a = (2/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  a = 68 := by sorry

end NUMINAMATH_CALUDE_share_distribution_l3885_388533


namespace NUMINAMATH_CALUDE_consecutive_digit_difference_constant_l3885_388547

/-- Represents a four-digit number -/
def FourDigitNumber (a b c d : ℕ) : ℤ := 1000 * a + 100 * b + 10 * c + d

/-- The difference between abcd and dcba for consecutive digits -/
def ConsecutiveDigitDifference (a : ℕ) : ℤ :=
  let b := a + 1
  let c := a + 2
  let d := a + 3
  FourDigitNumber a b c d - FourDigitNumber d c b a

theorem consecutive_digit_difference_constant :
  ∀ a : ℕ, 0 ≤ a ∧ a ≤ 6 → ConsecutiveDigitDifference a = -3096 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_digit_difference_constant_l3885_388547


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3885_388523

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 3 * x - 5) ↔ x ≥ 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3885_388523


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l3885_388576

/-- A hyperbola with equation mx² + y² = 1 where the length of its imaginary axis 
    is twice the length of its real axis -/
structure Hyperbola where
  m : ℝ
  eq : ∀ x y : ℝ, m * x^2 + y^2 = 1
  axis_ratio : (imaginary_axis_length : ℝ) = 2 * (real_axis_length : ℝ)

/-- The value of m for a hyperbola with the given properties is -1/4 -/
theorem hyperbola_m_value (h : Hyperbola) : h.m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l3885_388576


namespace NUMINAMATH_CALUDE_zero_discriminant_implies_geometric_progression_l3885_388566

-- Define the quadratic equation ax^2 - 6bx + 9c = 0
def quadratic_equation (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 - 6 * b * x + 9 * c = 0

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℝ) : ℝ :=
  36 * b^2 - 36 * a * c

-- Define a geometric progression
def is_geometric_progression (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Theorem statement
theorem zero_discriminant_implies_geometric_progression
  (a b c : ℝ) (h : discriminant a b c = 0) :
  is_geometric_progression a b c := by
sorry

end NUMINAMATH_CALUDE_zero_discriminant_implies_geometric_progression_l3885_388566


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3885_388556

/-- The surface area of a cylinder with height 2 and base circumference 2π is 6π. -/
theorem cylinder_surface_area : 
  ∀ (h c : ℝ), h = 2 → c = 2 * Real.pi → 
  2 * Real.pi * (c / (2 * Real.pi)) * (c / (2 * Real.pi)) + c * h = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3885_388556


namespace NUMINAMATH_CALUDE_min_cubes_required_l3885_388560

-- Define the dimensions of the box
def box_length : ℕ := 9
def box_width : ℕ := 12
def box_height : ℕ := 3

-- Define the volume of a single cube
def cube_volume : ℕ := 3

-- Theorem: The minimum number of cubes required is 108
theorem min_cubes_required : 
  (box_length * box_width * box_height) / cube_volume = 108 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_required_l3885_388560


namespace NUMINAMATH_CALUDE_square_diff_cubed_l3885_388568

theorem square_diff_cubed : (5^2 - 4^2)^3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_cubed_l3885_388568


namespace NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l3885_388570

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add any necessary fields here

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def smallestRotationalSymmetryAngle (p : RegularPolygon n) : ℝ := sorry

theorem regular_18gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
    (linesOfSymmetry p : ℝ) + smallestRotationalSymmetryAngle p = 38 := by sorry

end NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l3885_388570


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3885_388558

theorem ceiling_floor_difference : ⌈(15 : ℚ) / 8 * (-34 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-34 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3885_388558


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l3885_388542

theorem complex_number_coordinates : 
  let i : ℂ := Complex.I
  let z : ℂ := i * (1 + i)
  z = -1 + i := by sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l3885_388542


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3885_388599

/-- Given that Rahul's age after 6 years will be 26 and Deepak's current age is 15,
    prove that the ratio of their current ages is 4:3. -/
theorem age_ratio_problem (rahul_future_age : ℕ) (deepak_age : ℕ) : 
  rahul_future_age = 26 → 
  deepak_age = 15 → 
  (rahul_future_age - 6) / deepak_age = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3885_388599


namespace NUMINAMATH_CALUDE_even_function_sum_l3885_388595

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_sum (f : ℝ → ℝ) (h_even : is_even_function f) (h_f4 : f 4 = 5) :
  f 4 + f (-4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l3885_388595


namespace NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l3885_388567

/-- Given an arithmetic progression with first term 7, if adding 3 to the second term and 25 to the third term
    results in a geometric progression, then the smallest possible value for the third term of the geometric
    progression is -0.62. -/
theorem smallest_third_term_of_geometric_progression (d : ℝ) :
  let a₁ := 7
  let a₂ := a₁ + d
  let a₃ := a₁ + 2*d
  let g₁ := a₁
  let g₂ := a₂ + 3
  let g₃ := a₃ + 25
  (g₂^2 = g₁ * g₃) →
  ∃ (d' : ℝ), g₃ ≥ -0.62 ∧ (∀ (d'' : ℝ),
    let g₁' := 7
    let g₂' := (7 + d'') + 3
    let g₃' := (7 + 2*d'') + 25
    (g₂'^2 = g₁' * g₃') → g₃' ≥ g₃) :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l3885_388567


namespace NUMINAMATH_CALUDE_decimal_521_equals_octal_1011_l3885_388536

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid octal number -/
def is_valid_octal (l : List ℕ) : Prop :=
  l.all (λ d => d < 8)

theorem decimal_521_equals_octal_1011 :
  decimal_to_octal 521 = [1, 0, 1, 1] ∧ is_valid_octal [1, 0, 1, 1] :=
by sorry

end NUMINAMATH_CALUDE_decimal_521_equals_octal_1011_l3885_388536


namespace NUMINAMATH_CALUDE_corn_growth_first_week_l3885_388579

/-- Represents the growth of corn over three weeks -/
structure CornGrowth where
  week1 : ℝ
  week2 : ℝ
  week3 : ℝ

/-- The conditions of corn growth as described in the problem -/
def valid_growth (g : CornGrowth) : Prop :=
  g.week2 = 2 * g.week1 ∧
  g.week3 = 4 * g.week2 ∧
  g.week1 + g.week2 + g.week3 = 22

/-- The theorem stating that the corn grew 2 inches in the first week -/
theorem corn_growth_first_week :
  ∀ g : CornGrowth, valid_growth g → g.week1 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_corn_growth_first_week_l3885_388579


namespace NUMINAMATH_CALUDE_exists_counterfeit_finding_algorithm_l3885_388543

/-- Represents a coin, which can be either genuine or counterfeit -/
inductive Coin
| genuine : Coin
| counterfeit : Coin

/-- Represents the result of a weighing operation -/
inductive WeighResult
| balanced : WeighResult
| leftLighter : WeighResult
| rightLighter : WeighResult

/-- A function that simulates weighing two sets of coins -/
def weigh (left : List Coin) (right : List Coin) : WeighResult :=
  sorry

/-- The type of an algorithm to find the counterfeit coin -/
def FindCounterfeitAlgorithm := List Coin → Coin

/-- Theorem stating that there exists an algorithm to find the counterfeit coin -/
theorem exists_counterfeit_finding_algorithm :
  ∃ (algo : FindCounterfeitAlgorithm),
    ∀ (coins : List Coin),
      coins.length = 9 →
      (∃! (c : Coin), c ∈ coins ∧ c = Coin.counterfeit) →
      algo coins = Coin.counterfeit :=
sorry

end NUMINAMATH_CALUDE_exists_counterfeit_finding_algorithm_l3885_388543


namespace NUMINAMATH_CALUDE_solve_pickle_problem_l3885_388513

def pickle_problem (total_jars total_cucumbers initial_vinegar pickles_per_jar vinegar_per_jar remaining_vinegar : ℕ) : Prop :=
  let used_vinegar := initial_vinegar - remaining_vinegar
  let filled_jars := used_vinegar / vinegar_per_jar
  let total_pickles := filled_jars * pickles_per_jar
  let pickles_per_cucumber := total_pickles / total_cucumbers
  pickles_per_cucumber = 4 ∧
  total_jars = 4 ∧
  total_cucumbers = 10 ∧
  initial_vinegar = 100 ∧
  pickles_per_jar = 12 ∧
  vinegar_per_jar = 10 ∧
  remaining_vinegar = 60

theorem solve_pickle_problem :
  ∃ (total_jars total_cucumbers initial_vinegar pickles_per_jar vinegar_per_jar remaining_vinegar : ℕ),
  pickle_problem total_jars total_cucumbers initial_vinegar pickles_per_jar vinegar_per_jar remaining_vinegar :=
by
  sorry

end NUMINAMATH_CALUDE_solve_pickle_problem_l3885_388513


namespace NUMINAMATH_CALUDE_inequality_proof_l3885_388535

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3885_388535


namespace NUMINAMATH_CALUDE_min_value_of_function_max_sum_with_constraint_l3885_388518

-- Part 1
theorem min_value_of_function (x : ℝ) (h : x > -1) :
  ∃ (min_y : ℝ), min_y = 9 ∧ ∀ y, y = (x^2 + 7*x + 10) / (x + 1) → y ≥ min_y :=
sorry

-- Part 2
theorem max_sum_with_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  x + y ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_max_sum_with_constraint_l3885_388518


namespace NUMINAMATH_CALUDE_exactly_two_correct_l3885_388539

-- Define the propositions
def prop1 : Prop := ∃ n : ℤ, ∀ m : ℤ, m < 0 → m ≤ n
def prop2 : Prop := ∃ n : ℤ, ∀ m : ℤ, n ≤ m
def prop3 : Prop := ∀ n : ℤ, n < 0 → n ≤ -1
def prop4 : Prop := ∀ n : ℤ, n > 0 → 1 ≤ n

-- Theorem stating that exactly two propositions are correct
theorem exactly_two_correct : 
  ¬prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_correct_l3885_388539


namespace NUMINAMATH_CALUDE_multiply_with_negative_l3885_388527

theorem multiply_with_negative (a : ℝ) : 3 * a * (-2 * a) = -6 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_with_negative_l3885_388527


namespace NUMINAMATH_CALUDE_diophantine_equation_solvable_l3885_388577

theorem diophantine_equation_solvable (a : ℕ+) :
  ∃ (x y : ℤ), x^2 - y^2 = (a : ℤ)^3 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solvable_l3885_388577
