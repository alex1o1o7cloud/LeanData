import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l2549_254950

theorem expression_evaluation : (20 * 3 + 10) / (5 + 3) = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2549_254950


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l2549_254994

def geometric_sequence (a : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a * r^(n - 1)

theorem first_term_of_geometric_sequence 
  (a : ℝ) (r : ℝ) (h1 : r ≠ 0) (h2 : r ≠ 1) :
  (geometric_sequence a r 1 + geometric_sequence a r 2 + 
   geometric_sequence a r 3 + geometric_sequence a r 4 = 240) →
  (geometric_sequence a r 2 + geometric_sequence a r 4 = 180) →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l2549_254994


namespace NUMINAMATH_CALUDE_division_remainder_and_divisibility_l2549_254942

theorem division_remainder_and_divisibility : 
  let n : ℕ := 1234567
  let d : ℕ := 256
  let r : ℕ := n % d
  (r = 2) ∧ (r % 7 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_division_remainder_and_divisibility_l2549_254942


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l2549_254959

theorem modular_arithmetic_problem (m : ℕ) : 
  m < 41 ∧ (5 * m) % 41 = 1 → (3^m % 41)^2 % 41 - 3 % 41 = 6 % 41 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l2549_254959


namespace NUMINAMATH_CALUDE_star_calculation_l2549_254995

-- Define the ☆ operation for rational numbers
def star (a b : ℚ) : ℚ := 2 * a - b + 1

-- Theorem statement
theorem star_calculation : star 1 (star 2 (-3)) = -5 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l2549_254995


namespace NUMINAMATH_CALUDE_student_pairs_l2549_254957

theorem student_pairs (n : ℕ) (same_letter_pairs : ℕ) (total_pairs : ℕ) :
  n = 12 →
  same_letter_pairs = 3 →
  total_pairs = n.choose 2 →
  total_pairs - same_letter_pairs = 63 := by
  sorry

end NUMINAMATH_CALUDE_student_pairs_l2549_254957


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_six_l2549_254917

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_six (n : ℕ) : Prop := n % 6 = 0

theorem smallest_four_digit_divisible_by_six :
  ∀ n : ℕ, is_four_digit n → divisible_by_six n → n ≥ 1002 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_six_l2549_254917


namespace NUMINAMATH_CALUDE_local_road_speed_l2549_254984

theorem local_road_speed (local_distance : ℝ) (highway_distance : ℝ) 
  (highway_speed : ℝ) (average_speed : ℝ) (local_speed : ℝ) : 
  local_distance = 60 ∧ 
  highway_distance = 65 ∧ 
  highway_speed = 65 ∧ 
  average_speed = 41.67 ∧
  (local_distance + highway_distance) / ((local_distance / local_speed) + (highway_distance / highway_speed)) = average_speed →
  local_speed = 30 := by
sorry

end NUMINAMATH_CALUDE_local_road_speed_l2549_254984


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_7_l2549_254987

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem parallel_lines_imply_a_equals_7 :
  let l1 : Line := { a := 2, b := 1, c := -1 }
  let l2 : Line := { a := a - 1, b := 3, c := -2 }
  parallel l1 l2 → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_7_l2549_254987


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2549_254956

theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  x^2 + (y - 2)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2549_254956


namespace NUMINAMATH_CALUDE_A_inter_B_a_upper_bound_a_sufficient_l2549_254905

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 < x ∧ x ≤ 5}
def B : Set ℝ := {x | (2*x - 1)/(x - 3) > 0}
def C (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 4*a - 3}

-- Theorem for A ∩ B
theorem A_inter_B : A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 5} := by sorry

-- Theorem for the upper bound of a
theorem a_upper_bound (a : ℝ) (h : C a ∪ A = A) : a ≤ 2 := by sorry

-- Theorem for the sufficiency of a ≤ 2
theorem a_sufficient (a : ℝ) (h : a ≤ 2) : C a ∪ A = A := by sorry

end NUMINAMATH_CALUDE_A_inter_B_a_upper_bound_a_sufficient_l2549_254905


namespace NUMINAMATH_CALUDE_train_crossing_time_l2549_254952

-- Define constants
def train_length : Real := 120
def train_speed_kmh : Real := 70
def bridge_length : Real := 150

-- Define the theorem
theorem train_crossing_time :
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let crossing_time := total_distance / train_speed_ms
  ∃ ε > 0, abs (crossing_time - 13.89) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2549_254952


namespace NUMINAMATH_CALUDE_factorization_proof_l2549_254932

variable (x y b : ℝ)

theorem factorization_proof : 
  (-x^3 - 2*x^2 - x = -x*(x + 1)^2) ∧ 
  ((x - y) - 4*b^2*(x - y) = (x - y)*(1 + 2*b)*(1 - 2*b)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_proof_l2549_254932


namespace NUMINAMATH_CALUDE_investment_partnership_problem_l2549_254904

/-- Investment partnership problem -/
theorem investment_partnership_problem 
  (a b c d : ℝ) -- Investments of partners A, B, C, and D
  (total_profit : ℝ) -- Total profit
  (ha : a = 3 * b) -- A invests 3 times as much as B
  (hb : b = (2/3) * c) -- B invests two-thirds of what C invests
  (hd : d = (1/2) * a) -- D invests half as much as A
  (hp : total_profit = 19900) -- Total profit is Rs.19900
  : b * total_profit / (a + b + c + d) = 2842.86 := by
  sorry

end NUMINAMATH_CALUDE_investment_partnership_problem_l2549_254904


namespace NUMINAMATH_CALUDE_briefcase_pen_price_ratio_l2549_254913

/-- Given a pen price of 4 and a total cost of 24 for the pen and a briefcase,
    where the briefcase's price is some multiple of the pen's price,
    prove that the ratio of the briefcase's price to the pen's price is 5. -/
theorem briefcase_pen_price_ratio :
  ∀ (briefcase_price : ℝ),
  briefcase_price > 0 →
  ∃ (multiple : ℝ), multiple > 0 ∧ briefcase_price = 4 * multiple →
  4 + briefcase_price = 24 →
  briefcase_price / 4 = 5 := by
sorry

end NUMINAMATH_CALUDE_briefcase_pen_price_ratio_l2549_254913


namespace NUMINAMATH_CALUDE_milk_level_lowered_l2549_254931

/-- Proves that removing 5250 gallons of milk from a 56ft by 25ft rectangular box
    lowers the milk level by 6 inches. -/
theorem milk_level_lowered (box_length box_width : ℝ)
                            (milk_volume_gallons : ℝ)
                            (cubic_feet_to_gallons : ℝ)
                            (inches_per_foot : ℝ) :
  box_length = 56 →
  box_width = 25 →
  milk_volume_gallons = 5250 →
  cubic_feet_to_gallons = 7.5 →
  inches_per_foot = 12 →
  (milk_volume_gallons / cubic_feet_to_gallons) /
  (box_length * box_width) * inches_per_foot = 6 :=
by sorry

end NUMINAMATH_CALUDE_milk_level_lowered_l2549_254931


namespace NUMINAMATH_CALUDE_dot_product_bound_l2549_254902

theorem dot_product_bound (a b c x y z : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 4) 
  (h2 : x^2 + y^2 + z^2 = 9) : 
  -6 ≤ a * x + b * y + c * z ∧ a * x + b * y + c * z ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_bound_l2549_254902


namespace NUMINAMATH_CALUDE_optimal_AD_length_l2549_254960

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)

/-- Point D on AB -/
def D (t : Triangle) := ℝ

/-- Expected value of EF -/
noncomputable def expectedEF (t : Triangle) (d : D t) : ℝ := sorry

/-- Theorem statement -/
theorem optimal_AD_length (t : Triangle) 
  (h1 : t.AB = 14) 
  (h2 : t.AC = 13) 
  (h3 : t.BC = 15) : 
  ∃ (d : D t), 
    (∀ (d' : D t), expectedEF t d ≥ expectedEF t d') ∧ 
    d = Real.sqrt 70 :=
sorry

end NUMINAMATH_CALUDE_optimal_AD_length_l2549_254960


namespace NUMINAMATH_CALUDE_sqrt_8_div_7_same_type_as_sqrt_2_l2549_254977

-- Define what it means for two quadratic radicals to be of the same type
def same_type (a b : ℝ) : Prop :=
  ∃ (q : ℚ), a = q * b

-- State the theorem
theorem sqrt_8_div_7_same_type_as_sqrt_2 :
  same_type (Real.sqrt 8 / 7) (Real.sqrt 2) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 2) ∧
  ¬ same_type (Real.sqrt (1/3)) (Real.sqrt 2) ∧
  ¬ same_type (Real.sqrt 12) (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_8_div_7_same_type_as_sqrt_2_l2549_254977


namespace NUMINAMATH_CALUDE_correct_transformation_l2549_254927

theorem correct_transformation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a = 2 * b) :
  a / 2 = b / 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l2549_254927


namespace NUMINAMATH_CALUDE_siblings_average_age_l2549_254914

theorem siblings_average_age (youngest_age : ℕ) (age_differences : List ℕ) : 
  youngest_age = 20 → 
  age_differences = [2, 7, 11] →
  (youngest_age + youngest_age + 2 + youngest_age + 7 + youngest_age + 11) / 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_siblings_average_age_l2549_254914


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l2549_254929

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically --/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift_theorem (x : ℝ) :
  let original := Parabola.mk 2 0 0  -- y = 2x²
  let shifted := shift_parabola original 3 1  -- Shift 3 right, 1 down
  shifted.a * x^2 + shifted.b * x + shifted.c = 2 * (x - 3)^2 - 1 := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l2549_254929


namespace NUMINAMATH_CALUDE_combined_boys_avg_is_correct_l2549_254907

/-- Represents a high school with exam scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two schools -/
structure CombinedSchools where
  school1 : School
  school2 : School
  combined_girls_avg : ℝ

/-- Calculates the combined average score for boys given two schools' data -/
def combined_boys_avg (schools : CombinedSchools) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating that the combined boys' average is approximately 48.57 -/
theorem combined_boys_avg_is_correct (schools : CombinedSchools) 
  (h1 : schools.school1 = ⟨68, 72, 70⟩)
  (h2 : schools.school2 = ⟨74, 88, 82⟩)
  (h3 : schools.combined_girls_avg = 83) :
  abs (combined_boys_avg schools - 48.57) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_combined_boys_avg_is_correct_l2549_254907


namespace NUMINAMATH_CALUDE_empty_seats_l2549_254974

theorem empty_seats (children : ℕ) (adults : ℕ) (total_seats : ℕ) : 
  children = 52 → adults = 29 → total_seats = 95 → 
  total_seats - (children + adults) = 14 := by
  sorry

end NUMINAMATH_CALUDE_empty_seats_l2549_254974


namespace NUMINAMATH_CALUDE_train_length_is_600_l2549_254968

/-- The length of the train in meters -/
def train_length : ℝ := 600

/-- The time it takes for the train to cross a tree, in seconds -/
def time_to_cross_tree : ℝ := 60

/-- The time it takes for the train to pass a platform, in seconds -/
def time_to_pass_platform : ℝ := 105

/-- The length of the platform, in meters -/
def platform_length : ℝ := 450

/-- Theorem stating that the train length is 600 meters -/
theorem train_length_is_600 :
  train_length = (time_to_pass_platform * platform_length) / (time_to_pass_platform - time_to_cross_tree) :=
by sorry

end NUMINAMATH_CALUDE_train_length_is_600_l2549_254968


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2549_254969

theorem complex_magnitude_product : 
  Complex.abs ((5 * Real.sqrt 2 - 5 * Complex.I) * (2 * Real.sqrt 3 + 6 * Complex.I)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2549_254969


namespace NUMINAMATH_CALUDE_max_value_cos_sin_sum_l2549_254911

theorem max_value_cos_sin_sum :
  ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 ∧ 
  ∃ y : ℝ, 3 * Real.cos y + 4 * Real.sin y = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_sum_l2549_254911


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l2549_254985

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l2549_254985


namespace NUMINAMATH_CALUDE_additional_cards_proof_l2549_254963

/-- The number of cards in the original deck -/
def original_deck : ℕ := 52

/-- The number of players -/
def num_players : ℕ := 3

/-- The number of cards each player has after splitting the deck -/
def cards_per_player : ℕ := 18

/-- The number of additional cards added to the deck -/
def additional_cards : ℕ := (num_players * cards_per_player) - original_deck

theorem additional_cards_proof :
  additional_cards = 2 := by sorry

end NUMINAMATH_CALUDE_additional_cards_proof_l2549_254963


namespace NUMINAMATH_CALUDE_chefs_flour_calculation_l2549_254997

theorem chefs_flour_calculation (recipe_ratio : ℚ) (eggs_needed : ℕ) (flour_used : ℚ) : 
  recipe_ratio = 7 / 2 →
  eggs_needed = 28 →
  flour_used = eggs_needed / recipe_ratio →
  flour_used = 8 := by
sorry

end NUMINAMATH_CALUDE_chefs_flour_calculation_l2549_254997


namespace NUMINAMATH_CALUDE_fish_ratio_proof_l2549_254967

theorem fish_ratio_proof (ken_caught : ℕ) (ken_released : ℕ) (kendra_caught : ℕ) (total_brought_home : ℕ)
  (h1 : ken_released = 3)
  (h2 : kendra_caught = 30)
  (h3 : (ken_caught - ken_released) + kendra_caught = total_brought_home)
  (h4 : total_brought_home = 87) :
  (ken_caught : ℚ) / kendra_caught = 19 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fish_ratio_proof_l2549_254967


namespace NUMINAMATH_CALUDE_central_sum_theorem_l2549_254976

/-- Represents a 4x4 matrix of integers -/
def Matrix4x4 := Fin 4 → Fin 4 → ℕ

/-- Checks if two positions in the matrix are adjacent -/
def isAdjacent (a b : Fin 4 × Fin 4) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ b.2 = a.2 + 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ b.1 = a.1 + 1))

/-- Checks if the matrix contains all numbers from 1 to 16 -/
def containsAllNumbers (m : Matrix4x4) : Prop :=
  ∀ n : Fin 16, ∃ i j : Fin 4, m i j = n.val + 1

/-- Checks if consecutive numbers are adjacent in the matrix -/
def consecutiveAdjacent (m : Matrix4x4) : Prop :=
  ∀ n : Fin 15, ∃ i₁ j₁ i₂ j₂ : Fin 4,
    m i₁ j₁ = n.val + 1 ∧ m i₂ j₂ = n.val + 2 ∧ isAdjacent (i₁, j₁) (i₂, j₂)

/-- Calculates the sum of corner numbers in the matrix -/
def cornerSum (m : Matrix4x4) : ℕ :=
  m 0 0 + m 0 3 + m 3 0 + m 3 3

/-- Calculates the sum of central numbers in the matrix -/
def centerSum (m : Matrix4x4) : ℕ :=
  m 1 1 + m 1 2 + m 2 1 + m 2 2

theorem central_sum_theorem (m : Matrix4x4)
  (h1 : containsAllNumbers m)
  (h2 : consecutiveAdjacent m)
  (h3 : cornerSum m = 34) :
  centerSum m = 34 := by
  sorry

end NUMINAMATH_CALUDE_central_sum_theorem_l2549_254976


namespace NUMINAMATH_CALUDE_compare_sqrt_l2549_254972

theorem compare_sqrt : 2 * Real.sqrt 11 < 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_compare_sqrt_l2549_254972


namespace NUMINAMATH_CALUDE_complex_unit_circle_representation_l2549_254990

theorem complex_unit_circle_representation (z : ℂ) (h1 : Complex.abs z = 1) (h2 : z ≠ -1) :
  ∃ t : ℝ, z = (1 + Complex.I * t) / (1 - Complex.I * t) := by
  sorry

end NUMINAMATH_CALUDE_complex_unit_circle_representation_l2549_254990


namespace NUMINAMATH_CALUDE_exists_g_compose_eq_f_l2549_254945

noncomputable def f (k ℓ : ℝ) (x : ℝ) : ℝ := k * x + ℓ

theorem exists_g_compose_eq_f (k ℓ : ℝ) (h : k > 0) :
  ∃ (a b : ℝ), ∀ x, f k ℓ x = f a b (f a b x) ∧ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_g_compose_eq_f_l2549_254945


namespace NUMINAMATH_CALUDE_greatest_product_sum_2024_l2549_254943

theorem greatest_product_sum_2024 :
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144 ∧
    ∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_product_sum_2024_l2549_254943


namespace NUMINAMATH_CALUDE_exam_success_probability_l2549_254900

theorem exam_success_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/2) (h2 : p2 = 1/4) (h3 : p3 = 1/5) :
  let at_least_two_success := 
    p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3 + p1 * p2 * p3
  at_least_two_success = 9/40 := by
  sorry

end NUMINAMATH_CALUDE_exam_success_probability_l2549_254900


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2549_254970

theorem unique_solution_condition (p q : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + p = q * x + 2) ↔ q ≠ 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2549_254970


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2549_254916

theorem absolute_value_inequality_solution (x : ℝ) :
  (|x + 2| + |x - 2| < x + 7) ↔ (-7/3 < x ∧ x < 7) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2549_254916


namespace NUMINAMATH_CALUDE_determinant_problem_l2549_254926

theorem determinant_problem (x y z w : ℝ) 
  (h : x * w - y * z = 7) : 
  x * (8 * z + 4 * w) - z * (8 * x + 4 * y) = 28 := by
  sorry

end NUMINAMATH_CALUDE_determinant_problem_l2549_254926


namespace NUMINAMATH_CALUDE_g_tan_squared_l2549_254965

open Real

noncomputable def g (x : ℝ) : ℝ := 1 / ((x - 1) / x)

theorem g_tan_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/2) :
  g (tan t ^ 2) = tan t ^ 2 - tan t ^ 4 :=
by sorry

end NUMINAMATH_CALUDE_g_tan_squared_l2549_254965


namespace NUMINAMATH_CALUDE_three_color_circle_existence_l2549_254954

-- Define a color type
inductive Color
| Red
| Green
| Blue

-- Define a point on a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- State the theorem
theorem three_color_circle_existence 
  (coloring : Coloring) 
  (all_colors_used : ∀ c : Color, ∃ p : Point, coloring p = c) :
  ∃ circ : Circle, ∀ c : Color, ∃ p : Point, 
    coloring p = c ∧ (p.x - circ.center.x)^2 + (p.y - circ.center.y)^2 ≤ circ.radius^2 :=
sorry

end NUMINAMATH_CALUDE_three_color_circle_existence_l2549_254954


namespace NUMINAMATH_CALUDE_min_intersection_size_l2549_254933

theorem min_intersection_size (total students_with_brown_eyes students_with_lunch_box : ℕ) 
  (h1 : total = 25)
  (h2 : students_with_brown_eyes = 15)
  (h3 : students_with_lunch_box = 18) :
  ∃ (intersection : ℕ), 
    intersection ≤ students_with_brown_eyes ∧ 
    intersection ≤ students_with_lunch_box ∧
    intersection ≥ students_with_brown_eyes + students_with_lunch_box - total ∧
    intersection = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_intersection_size_l2549_254933


namespace NUMINAMATH_CALUDE_sin_210_degrees_l2549_254936

theorem sin_210_degrees : Real.sin (210 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l2549_254936


namespace NUMINAMATH_CALUDE_hamilton_marching_band_max_members_l2549_254920

theorem hamilton_marching_band_max_members :
  ∀ n : ℕ,
  (∃ k : ℕ, 30 * n = 34 * k + 2) →
  30 * n < 1500 →
  (∀ m : ℕ, (∃ j : ℕ, 30 * m = 34 * j + 2) → 30 * m < 1500 → 30 * m ≤ 30 * n) →
  30 * n = 1260 :=
by sorry

end NUMINAMATH_CALUDE_hamilton_marching_band_max_members_l2549_254920


namespace NUMINAMATH_CALUDE_polynomial_coefficient_ratio_l2549_254949

/-- Given a polynomial representation of x^5, prove that the ratio of even-indexed
    coefficients to odd-indexed coefficients (excluding a₅) is -61/60. -/
theorem polynomial_coefficient_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(2-x) + a₂*(2-x)^2 + a₃*(2-x)^3 + a₄*(2-x)^4 + a₅*(2-x)^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61/60 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_ratio_l2549_254949


namespace NUMINAMATH_CALUDE_find_x_l2549_254993

theorem find_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2549_254993


namespace NUMINAMATH_CALUDE_certain_number_equation_l2549_254988

theorem certain_number_equation (x : ℝ) : 300 + 5 * x = 340 ↔ x = 8 := by sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2549_254988


namespace NUMINAMATH_CALUDE_existence_of_special_polynomial_l2549_254924

theorem existence_of_special_polynomial :
  ∃ (P : Polynomial ℝ), 
    (∃ (i : ℕ), (P.coeff i < 0)) ∧ 
    (∀ (n : ℕ), n > 1 → ∀ (j : ℕ), ((P^n).coeff j > 0)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_polynomial_l2549_254924


namespace NUMINAMATH_CALUDE_bug_path_theorem_l2549_254986

/-- Represents a rectangular floor with a broken tile -/
structure Floor :=
  (width : ℕ)
  (length : ℕ)
  (broken_tile : ℕ × ℕ)

/-- Calculates the number of tiles a bug visits when walking diagonally across the floor -/
def tiles_visited (f : Floor) : ℕ :=
  f.width + f.length - Nat.gcd f.width f.length

/-- Theorem: A bug walking diagonally across a 12x25 floor with a broken tile visits 36 tiles -/
theorem bug_path_theorem (f : Floor) 
    (h_width : f.width = 12)
    (h_length : f.length = 25)
    (h_broken : f.broken_tile = (12, 18)) : 
  tiles_visited f = 36 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_theorem_l2549_254986


namespace NUMINAMATH_CALUDE_arc_length_300_degrees_l2549_254946

/-- The length of an arc in a circle with radius 2 and central angle 300° is 10π/3 -/
theorem arc_length_300_degrees (r : ℝ) (θ : ℝ) : 
  r = 2 → θ = 300 * π / 180 → r * θ = 10 * π / 3 := by sorry

end NUMINAMATH_CALUDE_arc_length_300_degrees_l2549_254946


namespace NUMINAMATH_CALUDE_stock_annual_return_l2549_254999

/-- Calculates the annual return percentage given initial price and price increase -/
def annual_return_percentage (initial_price price_increase : ℚ) : ℚ :=
  (price_increase / initial_price) * 100

/-- Theorem: The annual return percentage for a stock with initial price 8000 and price increase 400 is 5% -/
theorem stock_annual_return :
  let initial_price : ℚ := 8000
  let price_increase : ℚ := 400
  annual_return_percentage initial_price price_increase = 5 := by
  sorry

#eval annual_return_percentage 8000 400

end NUMINAMATH_CALUDE_stock_annual_return_l2549_254999


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angle_sum_l2549_254971

theorem polygon_interior_exterior_angle_sum (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 = 2 * 360) → 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angle_sum_l2549_254971


namespace NUMINAMATH_CALUDE_brian_tape_problem_l2549_254975

/-- The amount of tape needed for a rectangular box -/
def tape_needed (length width : ℕ) : ℕ := length + 2 * width

/-- The total amount of tape needed for multiple boxes of the same size -/
def total_tape_for_boxes (length width count : ℕ) : ℕ :=
  count * tape_needed length width

/-- The problem statement -/
theorem brian_tape_problem :
  let tape_for_small_boxes := total_tape_for_boxes 30 15 5
  let tape_for_large_boxes := total_tape_for_boxes 40 40 2
  tape_for_small_boxes + tape_for_large_boxes = 540 := by
sorry


end NUMINAMATH_CALUDE_brian_tape_problem_l2549_254975


namespace NUMINAMATH_CALUDE_optimal_point_distribution_l2549_254998

/-- A configuration of points in a space -/
structure PointConfiguration where
  total_points : ℕ
  num_groups : ℕ
  group_sizes : List ℕ
  no_collinear_triple : Prop
  distinct_group_sizes : Prop
  sum_of_sizes_equals_total : group_sizes.sum = total_points

/-- The number of triangles formed by choosing one point from each of any three different groups -/
def num_triangles (config : PointConfiguration) : ℕ :=
  sorry

/-- The optimal configuration maximizes the number of triangles -/
def is_optimal (config : PointConfiguration) : Prop :=
  ∀ other : PointConfiguration, num_triangles config ≥ num_triangles other

/-- The theorem stating the optimal configuration -/
theorem optimal_point_distribution :
  ∃ (optimal_config : PointConfiguration),
    optimal_config.total_points = 1989 ∧
    optimal_config.num_groups = 30 ∧
    optimal_config.group_sizes = [51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81] ∧
    is_optimal optimal_config :=
  sorry

end NUMINAMATH_CALUDE_optimal_point_distribution_l2549_254998


namespace NUMINAMATH_CALUDE_roberto_outfits_l2549_254981

/-- The number of different outfits Roberto can put together -/
def number_of_outfits (trousers shirts jackets belts : ℕ) 
  (restricted_jacket_trousers : ℕ) : ℕ :=
  let unrestricted_jackets := jackets - 1
  let unrestricted_combinations := trousers * shirts * unrestricted_jackets * belts
  let restricted_combinations := restricted_jacket_trousers * shirts * belts
  let overlapping_combinations := (trousers - restricted_jacket_trousers) * shirts * belts
  unrestricted_combinations + restricted_combinations - overlapping_combinations

/-- Theorem stating the number of outfits Roberto can put together -/
theorem roberto_outfits : 
  number_of_outfits 5 7 4 2 3 = 168 := by
  sorry

#eval number_of_outfits 5 7 4 2 3

end NUMINAMATH_CALUDE_roberto_outfits_l2549_254981


namespace NUMINAMATH_CALUDE_two_digit_number_special_property_l2549_254941

theorem two_digit_number_special_property : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (∃ x y : ℕ, n = 10 * x + y ∧ x < 10 ∧ y < 10 ∧ n = x^3 + y^2) ∧
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_special_property_l2549_254941


namespace NUMINAMATH_CALUDE_intersection_point_on_fixed_line_l2549_254964

/-- Hyperbola C with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  eccentricity : ℝ
  left_vertex : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- Line passing through a point and intersecting the hyperbola -/
structure IntersectingLine where
  passing_point : ℝ × ℝ
  intersection_point1 : ℝ × ℝ
  intersection_point2 : ℝ × ℝ

/-- Theorem stating that the intersection point P lies on a fixed line -/
theorem intersection_point_on_fixed_line (C : Hyperbola) (L : IntersectingLine) : 
  C.center = (0, 0) →
  C.left_focus = (-2 * Real.sqrt 5, 0) →
  C.eccentricity = Real.sqrt 5 →
  C.left_vertex = (-2, 0) →
  C.right_vertex = (2, 0) →
  L.passing_point = (-4, 0) →
  L.intersection_point1.1 < 0 ∧ L.intersection_point1.2 > 0 → -- M in second quadrant
  ∃ (P : ℝ × ℝ), P.1 = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_on_fixed_line_l2549_254964


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_one_fourth_cubed_l2549_254937

theorem decimal_equivalent_of_one_fourth_cubed : (1 / 4 : ℚ) ^ 3 = 0.015625 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_one_fourth_cubed_l2549_254937


namespace NUMINAMATH_CALUDE_average_tree_height_height_pattern_known_heights_l2549_254906

def tree_heights : List ℝ := [8, 4, 16, 8, 32, 16]

theorem average_tree_height : 
  (tree_heights.sum / tree_heights.length : ℝ) = 14 :=
by
  sorry

theorem height_pattern (i : Fin 5) : 
  tree_heights[i] = 2 * tree_heights[i.succ] ∨ 
  tree_heights[i] = tree_heights[i.succ] / 2 :=
by
  sorry

theorem known_heights : 
  tree_heights[0] = 8 ∧ tree_heights[2] = 16 ∧ tree_heights[4] = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_average_tree_height_height_pattern_known_heights_l2549_254906


namespace NUMINAMATH_CALUDE_cathys_wallet_theorem_l2549_254966

/-- Calculates the remaining money in Cathy's wallet after receiving money from parents, buying a book, and saving some money. -/
def cathys_remaining_money (initial_amount dad_contribution book_cost savings_rate : ℚ) : ℚ :=
  let mom_contribution := 2 * dad_contribution
  let total_received := initial_amount + dad_contribution + mom_contribution
  let after_book_purchase := total_received - book_cost
  let savings_amount := savings_rate * after_book_purchase
  after_book_purchase - savings_amount

/-- Theorem stating that Cathy's remaining money is $57.60 given the initial conditions. -/
theorem cathys_wallet_theorem :
  cathys_remaining_money 12 25 15 (1/5) = 288/5 := by sorry

end NUMINAMATH_CALUDE_cathys_wallet_theorem_l2549_254966


namespace NUMINAMATH_CALUDE_smallest_ending_in_9_divisible_by_13_l2549_254928

theorem smallest_ending_in_9_divisible_by_13 : 
  ∃ (n : ℕ), n > 0 ∧ n % 10 = 9 ∧ n % 13 = 0 ∧ n = 69 ∧ 
  ∀ (m : ℕ), m > 0 → m % 10 = 9 → m % 13 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_ending_in_9_divisible_by_13_l2549_254928


namespace NUMINAMATH_CALUDE_not_first_year_percentage_l2549_254901

/-- Represents the percentage of associates in each category at a law firm -/
structure LawFirmAssociates where
  secondYear : ℝ
  moreThanTwoYears : ℝ

/-- Theorem stating the percentage of associates who are not first-year associates -/
theorem not_first_year_percentage (firm : LawFirmAssociates) 
  (h1 : firm.secondYear = 25)
  (h2 : firm.moreThanTwoYears = 50) :
  100 - (100 - firm.moreThanTwoYears - firm.secondYear) = 75 := by
  sorry

#check not_first_year_percentage

end NUMINAMATH_CALUDE_not_first_year_percentage_l2549_254901


namespace NUMINAMATH_CALUDE_plane_equation_correct_l2549_254991

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by parametric equations -/
structure Line3D where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- A plane in 3D space defined by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def lineInPlane (plane : Plane) (line : Line3D) : Prop :=
  ∀ t, pointOnPlane plane ⟨line.x t, line.y t, line.z t⟩

/-- The given point that the plane passes through -/
def givenPoint : Point3D :=
  ⟨1, 4, -5⟩

/-- The given line that the plane contains -/
def givenLine : Line3D :=
  ⟨λ t => 4 * t + 2, λ t => -t + 1, λ t => 5 * t - 3⟩

/-- The plane we want to prove -/
def solutionPlane : Plane :=
  ⟨2, 7, 6, -66⟩

theorem plane_equation_correct :
  pointOnPlane solutionPlane givenPoint ∧
  lineInPlane solutionPlane givenLine ∧
  solutionPlane.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs solutionPlane.A) (Int.natAbs solutionPlane.B))
          (Nat.gcd (Int.natAbs solutionPlane.C) (Int.natAbs solutionPlane.D)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l2549_254991


namespace NUMINAMATH_CALUDE_largest_divisor_of_2n3_minus_2n_l2549_254930

theorem largest_divisor_of_2n3_minus_2n (n : ℤ) : 
  (∃ (k : ℤ), 2 * n^3 - 2 * n = 12 * k) ∧ 
  (∀ (m : ℤ), m > 12 → ∃ (l : ℤ), 2 * l^3 - 2 * l ≠ m * (2 * l^3 - 2 * l) / m) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_2n3_minus_2n_l2549_254930


namespace NUMINAMATH_CALUDE_problem_solution_l2549_254940

def I : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Finset ℕ := {3, 4, 5}
def N : Finset ℕ := {1, 3, 6}

theorem problem_solution :
  (M ∩ (I \ N) = {4, 5}) ∧
  (I \ (M ∪ N) = {2, 7, 8}) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2549_254940


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2549_254923

theorem child_ticket_cost (num_adults num_children : ℕ) (adult_ticket_price total_bill : ℚ) :
  num_adults = 10 →
  num_children = 11 →
  adult_ticket_price = 8 →
  total_bill = 124 →
  (total_bill - num_adults * adult_ticket_price) / num_children = 4 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2549_254923


namespace NUMINAMATH_CALUDE_y_derivative_l2549_254948

noncomputable def y (x : ℝ) : ℝ :=
  Real.cos (Real.log 2) - (1/3) * (Real.cos (3*x))^2 / Real.sin (6*x)

theorem y_derivative (x : ℝ) :
  deriv y x = 1 / (2 * (Real.sin (3*x))^2) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l2549_254948


namespace NUMINAMATH_CALUDE_traditionalist_fraction_l2549_254978

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) (num_progressives : ℚ) :
  num_provinces = 15 →
  num_traditionalists_per_province = num_progressives / 20 →
  (num_provinces : ℚ) * num_traditionalists_per_province / ((num_provinces : ℚ) * num_traditionalists_per_province + num_progressives) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_traditionalist_fraction_l2549_254978


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l2549_254951

/-- The set M of real numbers less than 3 -/
def M : Set ℝ := {x : ℝ | x < 3}

/-- The set N of real numbers less than 1 -/
def N : Set ℝ := {x : ℝ | x < 1}

/-- Theorem stating that the intersection of M and the complement of N in ℝ
    is equal to the set of real numbers x where 1 ≤ x < 3 -/
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l2549_254951


namespace NUMINAMATH_CALUDE_triangle_existence_l2549_254973

/-- A set of points in space -/
structure PointSet where
  n : ℕ
  points : Finset (Fin (2 * n))
  segments : Finset (Fin (2 * n) × Fin (2 * n))
  n_gt_one : n > 1
  segment_count : segments.card ≥ n^2 + 1

/-- A triangle in a point set -/
def Triangle (ps : PointSet) : Prop :=
  ∃ a b c, a ∈ ps.points ∧ b ∈ ps.points ∧ c ∈ ps.points ∧
    (a, b) ∈ ps.segments ∧ (b, c) ∈ ps.segments ∧ (c, a) ∈ ps.segments

/-- Theorem: If a point set satisfies the conditions, then it contains a triangle -/
theorem triangle_existence (ps : PointSet) : Triangle ps := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_l2549_254973


namespace NUMINAMATH_CALUDE_second_number_proof_l2549_254918

theorem second_number_proof (h1 : 268 * x = 19832) (h2 : 2.68 * 0.74 = 1.9832) : x = 74 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l2549_254918


namespace NUMINAMATH_CALUDE_sum_of_real_roots_l2549_254908

theorem sum_of_real_roots (x : ℝ) : 
  let f : ℝ → ℝ := fun x => x^4 - 8*x + 4
  ∃ (r₁ r₂ : ℝ), (f r₁ = 0 ∧ f r₂ = 0 ∧ (∀ r : ℝ, f r = 0 → r = r₁ ∨ r = r₂)) ∧ 
  r₁ + r₂ = -2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_real_roots_l2549_254908


namespace NUMINAMATH_CALUDE_line_equation_l2549_254996

/-- Proves that the line represented by the given parametric equations has the equation y = 2x - 4 -/
theorem line_equation (t : ℝ) :
  let x := 3 * t + 1
  let y := 6 * t - 2
  y = 2 * x - 4 := by sorry

end NUMINAMATH_CALUDE_line_equation_l2549_254996


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2549_254939

theorem cubic_equation_solution :
  let x : ℝ := -1 / (1 + Real.rpow 2 (1/3))
  x^3 + x^2 + x + 1/3 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2549_254939


namespace NUMINAMATH_CALUDE_isosceles_triangles_count_l2549_254992

/-- The number of ways to choose three vertices of a regular nonagon to form an isosceles triangle -/
def isosceles_triangles_in_nonagon : ℕ := 33

/-- A regular nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- The number of ways to choose 2 vertices from a nonagon -/
def choose_two_vertices : ℕ := (nonagon_sides * (nonagon_sides - 1)) / 2

/-- The number of equilateral triangles in a nonagon -/
def equilateral_triangles : ℕ := 3

theorem isosceles_triangles_count :
  isosceles_triangles_in_nonagon = choose_two_vertices - equilateral_triangles :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_count_l2549_254992


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2549_254912

/-- Given a cubic equation x³ + px + q = 0 where p and q are real numbers,
    if 2 + i is a root, then p + q = 9 -/
theorem cubic_root_sum (p q : ℝ) : 
  (Complex.I : ℂ) ^ 3 + p * (Complex.I : ℂ) + q = 0 → p + q = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2549_254912


namespace NUMINAMATH_CALUDE_hiking_rate_ratio_l2549_254934

/-- Proves that the ratio of the hiking rate down to the rate up is 1.5 -/
theorem hiking_rate_ratio : 
  let rate_up : ℝ := 7 -- miles per day
  let days_up : ℝ := 2
  let distance_down : ℝ := 21 -- miles
  let days_down : ℝ := days_up -- same time for both routes
  let rate_down : ℝ := distance_down / days_down
  rate_down / rate_up = 1.5 := by
sorry


end NUMINAMATH_CALUDE_hiking_rate_ratio_l2549_254934


namespace NUMINAMATH_CALUDE_weeding_rate_calculation_l2549_254953

/-- The hourly rate for mowing lawns -/
def mowing_rate : ℝ := 4

/-- The number of hours spent mowing lawns in September -/
def mowing_hours : ℝ := 25

/-- The number of hours spent pulling weeds in September -/
def weeding_hours : ℝ := 3

/-- The total earnings for September and October -/
def total_earnings : ℝ := 248

/-- The hourly rate for pulling weeds -/
def weeding_rate : ℝ := 8

theorem weeding_rate_calculation :
  2 * (mowing_rate * mowing_hours + weeding_rate * weeding_hours) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_weeding_rate_calculation_l2549_254953


namespace NUMINAMATH_CALUDE_m_plus_n_values_l2549_254961

theorem m_plus_n_values (m n : ℤ) (hm : |m| = 4) (hn : |n| = 5) (hn_neg : n < 0) :
  m + n = -1 ∨ m + n = -9 := by
  sorry

end NUMINAMATH_CALUDE_m_plus_n_values_l2549_254961


namespace NUMINAMATH_CALUDE_reading_time_difference_l2549_254947

/-- The reading problem setup -/
structure ReadingProblem where
  xanthia_rate : ℕ  -- pages per hour
  molly_rate : ℕ    -- pages per hour
  book_pages : ℕ
  
/-- Calculate the time difference in minutes -/
def time_difference (p : ReadingProblem) : ℕ :=
  ((p.book_pages / p.molly_rate - p.book_pages / p.xanthia_rate) * 60 : ℕ)

/-- The main theorem -/
theorem reading_time_difference (p : ReadingProblem) 
  (h1 : p.xanthia_rate = 120)
  (h2 : p.molly_rate = 60)
  (h3 : p.book_pages = 360) : 
  time_difference p = 180 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l2549_254947


namespace NUMINAMATH_CALUDE_asha_win_probability_l2549_254979

theorem asha_win_probability (lose_prob tie_prob : ℚ) 
  (lose_eq : lose_prob = 3 / 7)
  (tie_eq : tie_prob = 1 / 7)
  (total_prob : lose_prob + tie_prob + (1 - lose_prob - tie_prob) = 1) :
  1 - lose_prob - tie_prob = 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_asha_win_probability_l2549_254979


namespace NUMINAMATH_CALUDE_correct_calculation_l2549_254983

-- Define the variables
variable (AB : ℝ) (C : ℝ) (D : ℝ) (E : ℝ)

-- Define the conditions
def xiao_hu_error := AB * C + D * E * 10 = 39.6
def da_hu_error := AB * C * D * E = 36.9

-- State the theorem
theorem correct_calculation (h1 : xiao_hu_error AB C D E) (h2 : da_hu_error AB C D E) :
  AB * C + D * E = 26.1 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2549_254983


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2549_254921

theorem complex_fraction_calculation : 
  ((5 / 8 : ℚ) * (3 / 7) - (2 / 3) * (1 / 4)) * ((7 / 9 : ℚ) * (2 / 5) * (1 / 2) * 5040) = 79 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2549_254921


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l2549_254938

/-- Pizza sharing problem -/
theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let plain_pizza_cost : ℚ := 12
  let bacon_cost : ℚ := 3
  let bacon_slices : ℕ := 9
  let dave_plain_slices : ℕ := 1
  let dave_total_slices : ℕ := bacon_slices + dave_plain_slices
  let doug_slices : ℕ := total_slices - dave_total_slices
  let total_cost : ℚ := plain_pizza_cost + bacon_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let dave_payment : ℚ := cost_per_slice * dave_total_slices
  let doug_payment : ℚ := cost_per_slice * doug_slices
  dave_payment - doug_payment = 10 :=
by sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l2549_254938


namespace NUMINAMATH_CALUDE_alice_has_winning_strategy_l2549_254958

/-- Represents a game on a complete graph -/
structure GraphGame where
  n : ℕ  -- number of vertices
  m : ℕ  -- maximum number of edges Bob can direct per turn

/-- Represents a strategy for Alice -/
def Strategy := GraphGame → Bool

/-- Checks if a strategy is winning for Alice -/
def is_winning_strategy (s : Strategy) (g : GraphGame) : Prop :=
  ∀ (bob_moves : ℕ → Fin g.m), ∃ (cycle : List (Fin g.n)), 
    cycle.length > 0 ∧ 
    cycle.Nodup ∧
    (∀ (i : Fin cycle.length), 
      ∃ (edge_directed_by_alice : Bool), 
        edge_directed_by_alice = true)

/-- The main theorem stating that Alice has a winning strategy -/
theorem alice_has_winning_strategy : 
  ∃ (s : Strategy), is_winning_strategy s ⟨2014, 1000⟩ := by
  sorry


end NUMINAMATH_CALUDE_alice_has_winning_strategy_l2549_254958


namespace NUMINAMATH_CALUDE_monkey_climb_proof_l2549_254903

/-- The height of the tree in feet -/
def tree_height : ℝ := 19

/-- The number of hours the monkey climbs -/
def climbing_hours : ℕ := 17

/-- The distance the monkey slips back each hour in feet -/
def slip_distance : ℝ := 2

/-- The distance the monkey hops each hour in feet -/
def hop_distance : ℝ := 3

theorem monkey_climb_proof :
  tree_height = (climbing_hours - 1) * (hop_distance - slip_distance) + hop_distance :=
by sorry

end NUMINAMATH_CALUDE_monkey_climb_proof_l2549_254903


namespace NUMINAMATH_CALUDE_appliance_purchase_total_cost_l2549_254915

theorem appliance_purchase_total_cost : 
  let vacuum_original : ℝ := 250
  let vacuum_discount : ℝ := 0.20
  let dishwasher_cost : ℝ := 450
  let bundle_discount : ℝ := 75
  let sales_tax : ℝ := 0.07

  let vacuum_discounted : ℝ := vacuum_original * (1 - vacuum_discount)
  let subtotal : ℝ := vacuum_discounted + dishwasher_cost - bundle_discount
  let total_with_tax : ℝ := subtotal * (1 + sales_tax)

  total_with_tax = 615.25 := by sorry

end NUMINAMATH_CALUDE_appliance_purchase_total_cost_l2549_254915


namespace NUMINAMATH_CALUDE_max_guests_correct_l2549_254982

/-- The maximum number of guests that can dine at a restaurant with n choices
    for each of starters, main dishes, desserts, and wines, such that:
    1) No two guests have the same order
    2) There is no collection of n guests whose orders coincide in three aspects
       but differ in the fourth -/
def max_guests (n : ℕ+) : ℕ :=
  if n = 1 then 1 else n^4 - n^3

theorem max_guests_correct (n : ℕ+) :
  (max_guests n = 1 ∧ n = 1) ∨
  (max_guests n = n^4 - n^3 ∧ n ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_max_guests_correct_l2549_254982


namespace NUMINAMATH_CALUDE_twenty_dollars_combinations_l2549_254910

/-- The number of ways to make 20 dollars with nickels, dimes, and quarters -/
def ways_to_make_20_dollars : ℕ :=
  (Finset.filter (fun (n, d, q) => 
    5 * n + 10 * d + 25 * q = 2000 ∧ 
    n ≥ 2 ∧ 
    q ≥ 1) 
  (Finset.product (Finset.range 401) (Finset.product (Finset.range 201) (Finset.range 81)))).card

/-- Theorem stating that there are exactly 130 ways to make 20 dollars 
    with nickels, dimes, and quarters, using at least two nickels and one quarter -/
theorem twenty_dollars_combinations : ways_to_make_20_dollars = 130 := by
  sorry

end NUMINAMATH_CALUDE_twenty_dollars_combinations_l2549_254910


namespace NUMINAMATH_CALUDE_monotonic_range_a_l2549_254909

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x)

theorem monotonic_range_a :
  ∀ a : ℝ, is_monotonic (f a) (-2) 3 ↔ a ≤ -27 ∨ 0 ≤ a :=
by sorry

end NUMINAMATH_CALUDE_monotonic_range_a_l2549_254909


namespace NUMINAMATH_CALUDE_votes_for_both_policies_l2549_254922

-- Define the total number of students
def total_students : ℕ := 185

-- Define the number of students voting for the first policy
def first_policy_votes : ℕ := 140

-- Define the number of students voting for the second policy
def second_policy_votes : ℕ := 110

-- Define the number of students voting against both policies
def against_both : ℕ := 22

-- Define the number of students abstaining from both policies
def abstained : ℕ := 15

-- Theorem stating that the number of students voting for both policies is 102
theorem votes_for_both_policies : 
  first_policy_votes + second_policy_votes - total_students + against_both + abstained = 102 :=
by sorry

end NUMINAMATH_CALUDE_votes_for_both_policies_l2549_254922


namespace NUMINAMATH_CALUDE_solve_system_l2549_254919

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 7 * q = 20) 
  (eq2 : 7 * p + 5 * q = 26) : 
  q = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2549_254919


namespace NUMINAMATH_CALUDE_even_iff_period_two_l2549_254955

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define the condition f(1+x) = f(1-x)
def symmetry_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + x) = f (1 - x)

-- Define an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define a function with period 2
def has_period_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

-- Theorem statement
theorem even_iff_period_two (f : ℝ → ℝ) (h : symmetry_condition f) :
  is_even f ↔ has_period_two f :=
sorry

end NUMINAMATH_CALUDE_even_iff_period_two_l2549_254955


namespace NUMINAMATH_CALUDE_intersection_M_N_l2549_254925

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x / (x - 1) ≤ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2549_254925


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_squared_l2549_254962

theorem greatest_divisor_four_consecutive_integers_squared (n : ℕ) :
  ∃ (k : ℕ), k = 144 ∧ (∀ m : ℕ, m > k → ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3))^2)) ∧
  (k ∣ (n * (n + 1) * (n + 2) * (n + 3))^2) := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_squared_l2549_254962


namespace NUMINAMATH_CALUDE_congruence_solution_l2549_254944

theorem congruence_solution (a m : ℕ) (h1 : a < m) (h2 : m ≥ 2) :
  (∃ x : ℕ, (10 * x + 3) % 18 = 7 % 18 ∧ x % m = a) →
  (∃ x : ℕ, x % 9 = 4 ∧ a = 4 ∧ m = 9 ∧ a + m = 13) :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_l2549_254944


namespace NUMINAMATH_CALUDE_sequence_formula_main_theorem_l2549_254989

def a (n : ℕ+) : ℚ := 1 / ((2 * n.val - 1) * (2 * n.val + 1))

def S (n : ℕ+) : ℚ := sorry

theorem sequence_formula (n : ℕ+) :
  S n / (n.val * (2 * n.val - 1)) = a n ∧ 
  S 1 / (1 * (2 * 1 - 1)) = 1 / 3 :=
by sorry

theorem main_theorem (n : ℕ+) : 
  S n / (n.val * (2 * n.val - 1)) = 1 / ((2 * n.val - 1) * (2 * n.val + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_main_theorem_l2549_254989


namespace NUMINAMATH_CALUDE_delightful_numbers_l2549_254980

def is_delightful (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  n % 25 = 0 ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) % 25 = 0 ∧
  ((n / 1000) * (n / 100 % 10) * (n / 10 % 10) * (n % 10)) % 25 = 0

theorem delightful_numbers :
  ∀ n : ℕ, is_delightful n ↔ n = 5875 ∨ n = 8575 := by sorry

end NUMINAMATH_CALUDE_delightful_numbers_l2549_254980


namespace NUMINAMATH_CALUDE_house_price_calculation_l2549_254935

theorem house_price_calculation (price_first : ℝ) (price_second : ℝ) : 
  price_second = 2 * price_first →
  price_first + price_second = 600000 →
  price_first = 200000 := by
sorry

end NUMINAMATH_CALUDE_house_price_calculation_l2549_254935
