import Mathlib

namespace NUMINAMATH_CALUDE_union_P_complement_Q_l3152_315209

def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

theorem union_P_complement_Q : P ∪ (Set.univ \ Q) = {x : ℝ | -2 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_union_P_complement_Q_l3152_315209


namespace NUMINAMATH_CALUDE_arrangement_count_l3152_315245

-- Define the total number of people
def total_people : ℕ := 7

-- Define the number of people to be selected
def selected_people : ℕ := 5

-- Define a function to calculate the number of arrangements
def arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

-- Theorem statement
theorem arrangement_count :
  arrangements total_people selected_people = 600 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l3152_315245


namespace NUMINAMATH_CALUDE_monic_quadratic_polynomial_l3152_315244

theorem monic_quadratic_polynomial (f : ℝ → ℝ) : 
  (∀ x, f x = x^2 + 5*x + 6) → 
  f 0 = 6 ∧ f 1 = 12 := by
sorry

end NUMINAMATH_CALUDE_monic_quadratic_polynomial_l3152_315244


namespace NUMINAMATH_CALUDE_video_game_theorem_l3152_315238

def video_game_problem (x : ℝ) (n : ℕ) (y : ℝ) : Prop :=
  x > 0 ∧ n > 0 ∧ y > 0 ∧
  (1/4 : ℝ) * x = (1/2 : ℝ) * n * y ∧
  (1/3 : ℝ) * x = x - ((1/2 : ℝ) * x + (1/6 : ℝ) * x)

theorem video_game_theorem (x : ℝ) (n : ℕ) (y : ℝ) 
  (h : video_game_problem x n y) : True :=
by
  sorry

end NUMINAMATH_CALUDE_video_game_theorem_l3152_315238


namespace NUMINAMATH_CALUDE_min_value_xyz_l3152_315274

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : (x + y) / (x * y * z) ≥ 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xyz_l3152_315274


namespace NUMINAMATH_CALUDE_negation_of_tangent_positive_l3152_315265

open Real

theorem negation_of_tangent_positive :
  (¬ ∀ x : ℝ, x ∈ Set.Ioo (-π/2) (π/2) → tan x > 0) ↔
  (∃ x : ℝ, x ∈ Set.Ioo (-π/2) (π/2) ∧ tan x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_tangent_positive_l3152_315265


namespace NUMINAMATH_CALUDE_sqrt_ratio_equality_l3152_315275

theorem sqrt_ratio_equality : 
  (Real.sqrt (3^2 + 4^2)) / (Real.sqrt (25 + 16)) = (5 * Real.sqrt 41) / 41 := by
sorry

end NUMINAMATH_CALUDE_sqrt_ratio_equality_l3152_315275


namespace NUMINAMATH_CALUDE_triangle_properties_l3152_315216

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 6)
  (h2 : Real.cos t.A = 1/8)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = 15 * Real.sqrt 7 / 4) :
  Real.sin t.C = Real.sqrt 7 / 4 ∧ t.b + t.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3152_315216


namespace NUMINAMATH_CALUDE_roses_mother_age_l3152_315283

theorem roses_mother_age (rose_age mother_age : ℕ) : 
  rose_age = mother_age / 3 →
  rose_age + mother_age = 100 →
  mother_age = 75 := by
sorry

end NUMINAMATH_CALUDE_roses_mother_age_l3152_315283


namespace NUMINAMATH_CALUDE_work_time_for_c_l3152_315235

/-- The time it takes for worker c to complete the work alone, given the combined work rates of pairs of workers. -/
theorem work_time_for_c (a b c : ℝ) 
  (h1 : a + b = 1/4)   -- a and b can do the work in 4 days
  (h2 : b + c = 1/6)   -- b and c can do the work in 6 days
  (h3 : c + a = 1/3) : -- c and a can do the work in 3 days
  1/c = 8 := by sorry

end NUMINAMATH_CALUDE_work_time_for_c_l3152_315235


namespace NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l3152_315247

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l3152_315247


namespace NUMINAMATH_CALUDE_angle_relation_l3152_315200

theorem angle_relation (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.tan (α - β) = 1/3) (h4 : Real.tan β = 1/7) :
  2 * α - β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_l3152_315200


namespace NUMINAMATH_CALUDE_aziz_is_36_l3152_315221

/-- Calculates Aziz's age in 2021 given the year his parents moved to America and the number of years they lived there before he was born -/
def aziz_age (parents_move_year : ℕ) (years_before_birth : ℕ) : ℕ :=
  2021 - (parents_move_year + years_before_birth)

/-- Theorem stating that Aziz's age in 2021 is 36 years -/
theorem aziz_is_36 : aziz_age 1982 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_aziz_is_36_l3152_315221


namespace NUMINAMATH_CALUDE_jeff_shelter_cats_l3152_315264

/-- Calculates the number of cats in Jeff's shelter after a series of events -/
def cats_in_shelter (initial : ℕ) (monday_found : ℕ) (tuesday_found : ℕ) (wednesday_adopted : ℕ) : ℕ :=
  initial + monday_found + tuesday_found - wednesday_adopted

/-- Theorem stating the number of cats in Jeff's shelter after the given events -/
theorem jeff_shelter_cats : cats_in_shelter 20 2 1 6 = 17 := by
  sorry

#eval cats_in_shelter 20 2 1 6

end NUMINAMATH_CALUDE_jeff_shelter_cats_l3152_315264


namespace NUMINAMATH_CALUDE_no_triple_squares_l3152_315256

theorem no_triple_squares : ¬∃ (m n k : ℕ), 
  (∃ a : ℕ, m^2 + n + k = a^2) ∧ 
  (∃ b : ℕ, n^2 + k + m = b^2) ∧ 
  (∃ c : ℕ, k^2 + m + n = c^2) := by
sorry

end NUMINAMATH_CALUDE_no_triple_squares_l3152_315256


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3152_315299

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first five terms of the sequence is 20. -/
def SumOfFirstFiveIs20 (a : ℕ → ℚ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 = 20

theorem arithmetic_sequence_third_term
    (a : ℕ → ℚ)
    (h_arithmetic : IsArithmeticSequence a)
    (h_sum : SumOfFirstFiveIs20 a) :
    a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3152_315299


namespace NUMINAMATH_CALUDE_work_completion_time_l3152_315249

theorem work_completion_time (original_men : ℕ) (added_men : ℕ) (time_reduction : ℕ) : 
  original_men = 40 →
  added_men = 8 →
  time_reduction = 10 →
  ∃ (original_time : ℕ), 
    original_time * original_men = (original_time - time_reduction) * (original_men + added_men) ∧
    original_time = 60 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3152_315249


namespace NUMINAMATH_CALUDE_comic_book_collections_l3152_315228

/-- Kymbrea's initial comic book collection -/
def kymbrea_initial : ℕ := 50

/-- Kymbrea's monthly comic book collection rate -/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection -/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book collection rate -/
def lashawn_rate : ℕ := 7

/-- The number of months after which LaShawn's collection is twice Kymbrea's -/
def months : ℕ := 80

theorem comic_book_collections : 
  lashawn_initial + lashawn_rate * months = 2 * (kymbrea_initial + kymbrea_rate * months) := by
  sorry

end NUMINAMATH_CALUDE_comic_book_collections_l3152_315228


namespace NUMINAMATH_CALUDE_rook_placements_count_l3152_315257

/-- The number of ways to place 3 rooks on a 6 × 2006 chessboard such that they don't attack each other -/
def rook_placements : ℕ :=
  (Nat.choose 6 3) * 2006 * 2005 * 2004

/-- Theorem stating the correct number of rook placements -/
theorem rook_placements_count : rook_placements = 20 * 2006 * 2005 * 2004 := by
  sorry

end NUMINAMATH_CALUDE_rook_placements_count_l3152_315257


namespace NUMINAMATH_CALUDE_s2_side_length_l3152_315286

/-- A composite rectangle structure -/
structure CompositeRectangle where
  width : ℕ
  height : ℕ
  s2_side : ℕ

/-- The composite rectangle satisfies the given conditions -/
def satisfies_conditions (cr : CompositeRectangle) : Prop :=
  cr.width = 3782 ∧ cr.height = 2260 ∧
  ∃ (r : ℕ), 2 * r + cr.s2_side = cr.height ∧ 2 * r + 3 * cr.s2_side = cr.width

/-- Theorem: The side length of S2 in the composite rectangle is 761 units -/
theorem s2_side_length :
  ∀ (cr : CompositeRectangle), satisfies_conditions cr → cr.s2_side = 761 :=
by
  sorry

end NUMINAMATH_CALUDE_s2_side_length_l3152_315286


namespace NUMINAMATH_CALUDE_negative_one_to_zero_power_l3152_315224

theorem negative_one_to_zero_power : (-1 : ℤ) ^ (0 : ℕ) = 1 := by sorry

end NUMINAMATH_CALUDE_negative_one_to_zero_power_l3152_315224


namespace NUMINAMATH_CALUDE_dining_group_size_l3152_315252

theorem dining_group_size (total_bill : ℝ) (tip_percentage : ℝ) (individual_payment : ℝ) : 
  total_bill = 139 ∧ tip_percentage = 0.1 ∧ individual_payment = 25.48 →
  Int.floor ((total_bill * (1 + tip_percentage)) / individual_payment) = 6 := by
sorry

end NUMINAMATH_CALUDE_dining_group_size_l3152_315252


namespace NUMINAMATH_CALUDE_distinct_distances_lower_bound_l3152_315251

/-- Given n points on a plane, where n ≥ 2, the number of distinct distances k
    between these points satisfies k ≥ √(n - 3/4) - 1/2. -/
theorem distinct_distances_lower_bound (n : ℕ) (k : ℕ) (h : n ≥ 2) :
  k ≥ Real.sqrt (n - 3/4) - 1/2 :=
by sorry

end NUMINAMATH_CALUDE_distinct_distances_lower_bound_l3152_315251


namespace NUMINAMATH_CALUDE_perfect_power_relation_l3152_315208

theorem perfect_power_relation (x y : ℕ+) (k : ℕ+) :
  (x * y^433 = k^2016) → ∃ m : ℕ+, x^433 * y = m^2016 := by
  sorry

end NUMINAMATH_CALUDE_perfect_power_relation_l3152_315208


namespace NUMINAMATH_CALUDE_house_square_footage_l3152_315219

def house_problem (smaller_house_original : ℝ) : Prop :=
  let larger_house : ℝ := 7300
  let expansion : ℝ := 3500
  let total_after_expansion : ℝ := 16000
  (smaller_house_original + expansion + larger_house = total_after_expansion) ∧
  (smaller_house_original = 5200)

theorem house_square_footage : ∃ (x : ℝ), house_problem x :=
  sorry

end NUMINAMATH_CALUDE_house_square_footage_l3152_315219


namespace NUMINAMATH_CALUDE_latte_cost_is_2_50_l3152_315231

/-- The cost of Sean's Sunday purchases -/
def seans_purchase (latte_cost : ℚ) : Prop :=
  let almond_croissant := (4.5 : ℚ)
  let salami_cheese_croissant := (4.5 : ℚ)
  let plain_croissant := (3 : ℚ)
  let focaccia := (4 : ℚ)
  let num_lattes := (2 : ℚ)
  let total_spent := (21 : ℚ)
  almond_croissant + salami_cheese_croissant + plain_croissant + focaccia + num_lattes * latte_cost = total_spent

/-- Theorem stating that each latte costs $2.50 -/
theorem latte_cost_is_2_50 : ∃ (latte_cost : ℚ), seans_purchase latte_cost ∧ latte_cost = (2.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_latte_cost_is_2_50_l3152_315231


namespace NUMINAMATH_CALUDE_smallest_x_y_sum_l3152_315280

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_x_y_sum (x y : ℕ) : 
  (x > 0 ∧ y > 0) →
  (is_square (450 * x)) →
  (is_cube (450 * y)) →
  (∀ x' : ℕ, x' > 0 → x' < x → ¬(is_square (450 * x'))) →
  (∀ y' : ℕ, y' > 0 → y' < y → ¬(is_cube (450 * y'))) →
  x = 2 ∧ y = 60 ∧ x + y = 62 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_y_sum_l3152_315280


namespace NUMINAMATH_CALUDE_sphere_radius_increase_l3152_315297

theorem sphere_radius_increase (r : ℝ) (h : r > 0) : 
  let A := 4 * Real.pi * r^2
  let r' := Real.sqrt (2.25 * r^2)
  let A' := 4 * Real.pi * r'^2
  A' = 2.25 * A → r' = 1.5 * r :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_increase_l3152_315297


namespace NUMINAMATH_CALUDE_gcd_lcm_pairs_l3152_315210

theorem gcd_lcm_pairs : 
  ∀ a b : ℕ, 
    a > 0 ∧ b > 0 →
    Nat.gcd a b = 24 ∧ Nat.lcm a b = 360 → 
    ((a = 24 ∧ b = 360) ∨ (a = 360 ∧ b = 24) ∨ (a = 72 ∧ b = 120) ∨ (a = 120 ∧ b = 72)) :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_pairs_l3152_315210


namespace NUMINAMATH_CALUDE_correct_propositions_l3152_315215

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations and operations
def subset (l : Line) (p : Plane) : Prop := sorry
def parallel (l₁ l₂ : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p₁ p₂ : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p₁ p₂ : Plane) : Prop := sorry
def intersection (p₁ p₂ : Plane) : Line := sorry

-- State the theorem
theorem correct_propositions 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (∀ (m n : Line) (α β : Plane) (l : Line),
    subset m α → subset n β → perpendicular_planes α β → 
    intersection α β = l → perpendicular m l → perpendicular m n) ∧
  (∀ (m : Line) (α β : Plane),
    perpendicular m α → perpendicular m β → parallel_planes α β) := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_l3152_315215


namespace NUMINAMATH_CALUDE_brandon_textbook_weight_l3152_315282

def jon_textbook_weights : List ℝ := [2, 8, 5, 9]

theorem brandon_textbook_weight (brandon_weight : ℝ) : 
  (List.sum jon_textbook_weights = 3 * brandon_weight) → brandon_weight = 8 := by
  sorry

end NUMINAMATH_CALUDE_brandon_textbook_weight_l3152_315282


namespace NUMINAMATH_CALUDE_management_subcommittee_count_l3152_315214

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid subcommittees -/
def validSubcommittees (totalMembers managers subcommitteeSize minManagers : ℕ) : ℕ :=
  choose totalMembers subcommitteeSize -
  (choose (totalMembers - managers) subcommitteeSize +
   choose managers 1 * choose (totalMembers - managers) (subcommitteeSize - 1))

theorem management_subcommittee_count :
  validSubcommittees 12 5 5 2 = 596 := by sorry

end NUMINAMATH_CALUDE_management_subcommittee_count_l3152_315214


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3152_315267

-- Define an arithmetic sequence of 8 terms
def arithmetic_sequence (a d : ℤ) : Fin 8 → ℤ :=
  fun i => a + i.val * d

theorem arithmetic_sequence_proof :
  ∀ a d : ℤ,
  (arithmetic_sequence a d 3 + arithmetic_sequence a d 4 = 41) →
  (arithmetic_sequence a d 0 * arithmetic_sequence a d 7 = 114) →
  ((∀ i : Fin 8, arithmetic_sequence a d i = arithmetic_sequence 3 5 i) ∨
   (∀ i : Fin 8, arithmetic_sequence a d i = arithmetic_sequence 38 (-5) i)) :=
by
  sorry

#check arithmetic_sequence_proof

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3152_315267


namespace NUMINAMATH_CALUDE_tan_product_eighths_pi_l3152_315292

theorem tan_product_eighths_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_eighths_pi_l3152_315292


namespace NUMINAMATH_CALUDE_project_exceeds_budget_l3152_315232

def field_area : ℝ := 3136
def wire_cost_per_meter : ℝ := 1.10
def gate_width : ℝ := 1
def gate_height : ℝ := 2
def iron_cost_per_kg : ℝ := 350
def gate_weight : ℝ := 25
def labor_cost_per_day : ℝ := 1500
def work_days : ℝ := 2
def budget : ℝ := 10000

theorem project_exceeds_budget :
  let field_side := Real.sqrt field_area
  let perimeter := 4 * field_side
  let wire_length := perimeter - 2 * gate_width
  let wire_cost := wire_length * wire_cost_per_meter
  let gates_cost := 2 * gate_weight * iron_cost_per_kg
  let labor_cost := work_days * labor_cost_per_day
  let total_cost := wire_cost + gates_cost + labor_cost
  total_cost > budget := by sorry

end NUMINAMATH_CALUDE_project_exceeds_budget_l3152_315232


namespace NUMINAMATH_CALUDE_berry_package_cost_l3152_315217

/-- The cost of one package of berries given Martin's consumption habits and spending --/
theorem berry_package_cost (daily_consumption : ℚ) (package_size : ℚ) (days : ℕ) (total_spent : ℚ) : 
  daily_consumption = 1/2 →
  package_size = 1 →
  days = 30 →
  total_spent = 30 →
  (total_spent / (days * daily_consumption / package_size) = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_berry_package_cost_l3152_315217


namespace NUMINAMATH_CALUDE_range_of_k_l3152_315253

open Set

def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}

def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2*k + 1}

theorem range_of_k : ∀ k : ℝ, (Aᶜ ∩ B k = ∅) ↔ (k ≤ 0 ∨ k ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l3152_315253


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3152_315223

theorem system_solution_ratio (x y z a b : ℝ) : 
  (4 * x - 2 * y + z = a) →
  (6 * y - 12 * x - 3 * z = b) →
  (b ≠ 0) →
  (a / b = -1 / 3) := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3152_315223


namespace NUMINAMATH_CALUDE_largest_cubic_root_bound_l3152_315207

theorem largest_cubic_root_bound (a₂ a₁ a₀ : ℝ) 
  (h₂ : |a₂| ≤ 2) (h₁ : |a₁| ≤ 2) (h₀ : |a₀| ≤ 2) :
  ∃ r : ℝ, (r > 0) ∧ (5/2 ≤ r) ∧ (r < 3) ∧
  (∀ x : ℝ, x^3 + a₂*x^2 + a₁*x + a₀ = 0 → x ≤ r) :=
sorry

end NUMINAMATH_CALUDE_largest_cubic_root_bound_l3152_315207


namespace NUMINAMATH_CALUDE_systematic_sample_smallest_number_l3152_315258

/-- Systematic sampling function -/
def systematicSample (n : ℕ) (k : ℕ) (i : ℕ) : ℕ := i * k

/-- Proposition: In a systematic sample of size 5 from 80 products, if 42 is in the sample, 
    then the smallest number in the sample is 10 -/
theorem systematic_sample_smallest_number :
  ∀ (i : ℕ), i < 5 →
  systematicSample 80 5 i = 42 →
  (∀ (j : ℕ), j < 5 → systematicSample 80 5 j ≥ 10) ∧
  (∃ (j : ℕ), j < 5 ∧ systematicSample 80 5 j = 10) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sample_smallest_number_l3152_315258


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3152_315227

-- Define the line equation
def line_equation (x y a b : ℝ) : Prop := x / (a^2) - y / (b^2) = 1

-- Define y-intercept
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

-- Theorem statement
theorem y_intercept_of_line (a b : ℝ) (h : b ≠ 0) :
  ∃ f : ℝ → ℝ, (∀ x, line_equation x (f x) a b) ∧ y_intercept f = -b^2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3152_315227


namespace NUMINAMATH_CALUDE_square_plus_linear_equals_square_l3152_315270

theorem square_plus_linear_equals_square (x y : ℕ+) 
  (h : x^2 + 84*x + 2016 = y^2) : 
  x^3 + y^2 = 12096 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_linear_equals_square_l3152_315270


namespace NUMINAMATH_CALUDE_max_four_digit_quotient_l3152_315271

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_nonzero_digit (n : ℕ) : Prop := n > 0 ∧ n ≤ 9

def four_digit_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

def digit_sum (a b c d : ℕ) : ℕ := a + b + c + d

theorem max_four_digit_quotient :
  ∀ (a b c d : ℕ),
    is_nonzero_digit a →
    is_digit b →
    is_nonzero_digit c →
    is_nonzero_digit d →
    (four_digit_number a b c d) / (digit_sum a b c d) ≤ 337 :=
by sorry

end NUMINAMATH_CALUDE_max_four_digit_quotient_l3152_315271


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l3152_315241

/-- The average speed of a car over two hours, given its speeds in each hour -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 90 → speed2 = 75 → (speed1 + speed2) / 2 = 82.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l3152_315241


namespace NUMINAMATH_CALUDE_min_value_sin_function_l3152_315278

theorem min_value_sin_function : 
  ∀ x : ℝ, -Real.sin x ^ 3 - 2 * Real.sin x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sin_function_l3152_315278


namespace NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_l3152_315213

/-- Theorem: One cubic meter is equal to 1,000,000 cubic centimeters -/
theorem cubic_meter_to_cubic_centimeters :
  ∀ (m cm : ℝ), m = 100 * cm → m^3 = 1000000 * cm^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_l3152_315213


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3152_315259

theorem fraction_equation_solution (n : ℚ) : 
  2 / (n + 2) + 3 / (n + 2) + 2 * n / (n + 2) = 4 → n = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3152_315259


namespace NUMINAMATH_CALUDE_new_students_average_age_l3152_315250

/-- Proves that the average age of new students is 32 years given the conditions of the problem -/
theorem new_students_average_age
  (original_average : ℝ)
  (new_students : ℕ)
  (new_average : ℝ)
  (original_strength : ℕ)
  (h1 : original_average = 40)
  (h2 : new_students = 12)
  (h3 : new_average = 36)
  (h4 : original_strength = 12) :
  (original_strength : ℝ) * original_average + (new_students : ℝ) * 32 =
    ((original_strength + new_students) : ℝ) * new_average :=
by sorry

end NUMINAMATH_CALUDE_new_students_average_age_l3152_315250


namespace NUMINAMATH_CALUDE_solutions_of_fourth_power_equation_l3152_315234

theorem solutions_of_fourth_power_equation :
  let S : Set ℂ := {x | x^4 - 16 = 0}
  S = {2, -2, Complex.I * 2, -Complex.I * 2} := by
  sorry

end NUMINAMATH_CALUDE_solutions_of_fourth_power_equation_l3152_315234


namespace NUMINAMATH_CALUDE_sum_of_first_49_odd_numbers_l3152_315220

theorem sum_of_first_49_odd_numbers : 
  (Finset.range 49).sum (fun i => 2 * i + 1) = 2401 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_49_odd_numbers_l3152_315220


namespace NUMINAMATH_CALUDE_scored_at_least_once_and_not_scored_both_times_mutually_exclusive_l3152_315233

-- Define the sample space for two shots
inductive ShotOutcome
  | Score
  | Miss

-- Define the event of scoring at least once
def scoredAtLeastOnce (outcome : ShotOutcome × ShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Score ∨ outcome.2 = ShotOutcome.Score

-- Define the event of not scoring both times
def notScoredBothTimes (outcome : ShotOutcome × ShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Miss ∨ outcome.2 = ShotOutcome.Miss

-- Theorem stating that the events are mutually exclusive
theorem scored_at_least_once_and_not_scored_both_times_mutually_exclusive :
  ∀ (outcome : ShotOutcome × ShotOutcome),
    ¬(scoredAtLeastOnce outcome ∧ notScoredBothTimes outcome) := by
  sorry


end NUMINAMATH_CALUDE_scored_at_least_once_and_not_scored_both_times_mutually_exclusive_l3152_315233


namespace NUMINAMATH_CALUDE_circle_equation_l3152_315218

/-- The equation of a circle with center (2, 1) that shares a common chord with another circle,
    where the chord lies on a line passing through a specific point. -/
theorem circle_equation (x y : ℝ) : 
  ∃ (r : ℝ), 
    -- The first circle has center (2, 1) and radius r
    ((x - 2)^2 + (y - 1)^2 = r^2) ∧
    -- The second circle is described by x^2 + y^2 - 3x = 0
    (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 3*x₀ = 0) ∧
    -- The common chord lies on a line passing through (5, -2)
    (∃ (a b c : ℝ), a*5 + b*(-2) + c = 0 ∧ a*x + b*y + c = 0) →
    -- The equation of the first circle is (x-2)^2 + (y-1)^2 = 4
    (x - 2)^2 + (y - 1)^2 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l3152_315218


namespace NUMINAMATH_CALUDE_class_test_result_l3152_315291

theorem class_test_result (boys : ℕ) (grade5 : ℕ) : ∃ (low_grade : ℕ), low_grade ≤ 2 ∧ low_grade > 0 := by
  -- Define the number of girls
  let girls : ℕ := boys + 3
  
  -- Define the number of grade 4s
  let grade4 : ℕ := grade5 + 6
  
  -- Define the number of grade 3s
  let grade3 : ℕ := 2 * grade4
  
  -- Define the total number of students
  let total_students : ℕ := boys + girls
  
  -- Define the total number of positive grades (3, 4, 5)
  let total_positive_grades : ℕ := grade3 + grade4 + grade5
  
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_class_test_result_l3152_315291


namespace NUMINAMATH_CALUDE_fraction_equality_l3152_315225

theorem fraction_equality : (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
                            (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3152_315225


namespace NUMINAMATH_CALUDE_sum_of_squares_l3152_315212

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3152_315212


namespace NUMINAMATH_CALUDE_phone_rep_hourly_wage_l3152_315261

/-- Calculates the hourly wage for phone reps given the number of reps, hours worked per day, days worked, and total payment -/
def hourly_wage (num_reps : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (total_payment : ℕ) : ℚ :=
  total_payment / (num_reps * hours_per_day * days_worked)

/-- Proves that the hourly wage for phone reps is $14 given the specified conditions -/
theorem phone_rep_hourly_wage :
  hourly_wage 50 8 5 28000 = 14 := by
  sorry

#eval hourly_wage 50 8 5 28000

end NUMINAMATH_CALUDE_phone_rep_hourly_wage_l3152_315261


namespace NUMINAMATH_CALUDE_shortest_player_height_l3152_315294

theorem shortest_player_height 
  (tallest_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : tallest_height = 77.75)
  (h2 : height_difference = 9.5) : 
  tallest_height - height_difference = 68.25 := by
sorry

end NUMINAMATH_CALUDE_shortest_player_height_l3152_315294


namespace NUMINAMATH_CALUDE_milk_per_milkshake_l3152_315260

/-- The amount of milk needed for each milkshake, given:
  * Blake has 72 ounces of milk initially
  * Blake has 192 ounces of ice cream
  * Each milkshake needs 12 ounces of ice cream
  * After making milkshakes, Blake has 8 ounces of milk left
-/
theorem milk_per_milkshake (initial_milk : ℕ) (ice_cream : ℕ) (ice_cream_per_shake : ℕ) (milk_left : ℕ)
  (h1 : initial_milk = 72)
  (h2 : ice_cream = 192)
  (h3 : ice_cream_per_shake = 12)
  (h4 : milk_left = 8) :
  (initial_milk - milk_left) / (ice_cream / ice_cream_per_shake) = 4 := by
  sorry

end NUMINAMATH_CALUDE_milk_per_milkshake_l3152_315260


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3152_315290

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → Prop) :
  (∀ n : ℕ, p n 2 → (∃ m : ℕ, n = 2 * m)) →
  (∀ n : ℕ, p 2 n → n = 2 ∨ n > 2) →
  p (3^20 + 11^14) 2 ∧ ∀ q : ℕ, p (3^20 + 11^14) q → q ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3152_315290


namespace NUMINAMATH_CALUDE_correct_article_usage_l3152_315272

/-- Represents the possible articles that can be used. -/
inductive Article
  | The
  | A
  | Blank

/-- Represents a pair of articles used in the sentence. -/
structure ArticlePair where
  first : Article
  second : Article

/-- Defines the correct article usage for the given sentence. -/
def correct_usage : ArticlePair :=
  { first := Article.The, second := Article.The }

/-- Determines if a noun is specific and known. -/
def is_specific_known (noun : String) : Bool :=
  match noun with
  | "bed" => true
  | _ => false

/-- Determines if a noun is made specific by additional information. -/
def is_specific_by_info (noun : String) (info : String) : Bool :=
  match noun, info with
  | "book", "I lost last week" => true
  | _, _ => false

/-- Theorem stating that the correct article usage is "the; the" given the conditions. -/
theorem correct_article_usage
  (bed : String)
  (book : String)
  (info : String)
  (h1 : is_specific_known bed = true)
  (h2 : is_specific_by_info book info = true) :
  correct_usage = { first := Article.The, second := Article.The } :=
by sorry

end NUMINAMATH_CALUDE_correct_article_usage_l3152_315272


namespace NUMINAMATH_CALUDE_worker_productivity_ratio_l3152_315239

theorem worker_productivity_ratio :
  ∀ (x y : ℝ),
  (x > 0) →
  (y > 0) →
  (2 * (x + y) = 1) →
  (x / 3 + 3 * y = 1) →
  (y / x = 5 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_worker_productivity_ratio_l3152_315239


namespace NUMINAMATH_CALUDE_no_valid_A_exists_l3152_315298

theorem no_valid_A_exists : ¬∃ (A : ℕ), 1 ≤ A ∧ A ≤ 9 ∧
  ∃ (x : ℕ), x^2 - (2*A)*x + (A+1)*0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_A_exists_l3152_315298


namespace NUMINAMATH_CALUDE_robs_money_total_l3152_315255

/-- Represents the value of coins in dollars -/
def coin_value (coin : String) : ℚ :=
  match coin with
  | "quarter" => 25 / 100
  | "dime" => 10 / 100
  | "nickel" => 5 / 100
  | "penny" => 1 / 100
  | _ => 0

/-- Calculates the total value of a given number of coins -/
def coin_total (coin : String) (count : ℕ) : ℚ :=
  (coin_value coin) * count

/-- Theorem: Rob's total money is $2.42 -/
theorem robs_money_total :
  let quarters := coin_total "quarter" 7
  let dimes := coin_total "dime" 3
  let nickels := coin_total "nickel" 5
  let pennies := coin_total "penny" 12
  quarters + dimes + nickels + pennies = 242 / 100 := by
  sorry

end NUMINAMATH_CALUDE_robs_money_total_l3152_315255


namespace NUMINAMATH_CALUDE_solution_difference_l3152_315287

theorem solution_difference (s : ℝ) : 
  (s + 3 ≠ 0) → 
  ((s^2 - 5*s - 24) / (s + 3) = 3*s + 10) → 
  ∃ (s1 s2 : ℝ), 
    ((s1^2 - 5*s1 - 24) / (s1 + 3) = 3*s1 + 10) ∧ 
    ((s2^2 - 5*s2 - 24) / (s2 + 3) = 3*s2 + 10) ∧ 
    s1 ≠ s2 ∧ 
    |s1 - s2| = 26 :=
by sorry

end NUMINAMATH_CALUDE_solution_difference_l3152_315287


namespace NUMINAMATH_CALUDE_solve_for_y_l3152_315211

theorem solve_for_y (x y : ℤ) (h1 : x^2 + 5 = y - 8) (h2 : x = -7) : y = 62 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3152_315211


namespace NUMINAMATH_CALUDE_farm_tree_count_l3152_315229

/-- Represents the state of trees on the farm --/
structure FarmTrees where
  mahogany : ℕ
  narra : ℕ

/-- Calculates the total number of trees --/
def total_trees (ft : FarmTrees) : ℕ := ft.mahogany + ft.narra

/-- Represents the number of fallen trees --/
structure FallenTrees where
  mahogany : ℕ
  narra : ℕ

/-- Represents the farm's tree management process --/
def farm_process (initial : FarmTrees) (fallen : FallenTrees) : ℕ :=
  let remaining := total_trees initial - (fallen.mahogany + fallen.narra)
  let new_mahogany := 3 * fallen.mahogany
  let new_narra := 2 * fallen.narra
  remaining + new_mahogany + new_narra

/-- Theorem stating the final number of trees on the farm --/
theorem farm_tree_count : 
  ∀ (initial : FarmTrees) (fallen : FallenTrees),
  initial.mahogany = 50 → 
  initial.narra = 30 → 
  fallen.mahogany + fallen.narra = 5 →
  fallen.mahogany = fallen.narra + 1 →
  farm_process initial fallen = 88 := by
  sorry


end NUMINAMATH_CALUDE_farm_tree_count_l3152_315229


namespace NUMINAMATH_CALUDE_min_value_of_f_l3152_315230

theorem min_value_of_f (x : ℝ) : 1 / Real.sqrt (x^2 + 2) + Real.sqrt (x^2 + 2) ≥ 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3152_315230


namespace NUMINAMATH_CALUDE_exceeded_goal_l3152_315268

/-- Represents the school band's car wash fundraiser --/
def car_wash_fundraiser (goal : ℕ) (basic_price deluxe_price premium_price cookie_price : ℕ)
  (basic_count deluxe_count premium_count cookie_count : ℕ) : Prop :=
  let total_earnings := basic_price * basic_count + deluxe_price * deluxe_count +
                        premium_price * premium_count + cookie_price * cookie_count
  total_earnings - goal = 32

/-- Theorem stating that the school band has exceeded their fundraising goal by $32 --/
theorem exceeded_goal : car_wash_fundraiser 150 5 8 12 2 10 6 2 30 := by
  sorry

end NUMINAMATH_CALUDE_exceeded_goal_l3152_315268


namespace NUMINAMATH_CALUDE_f_2016_value_l3152_315276

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem f_2016_value (f : ℝ → ℝ) 
  (h1 : f 1 = 1/4)
  (h2 : functional_equation f) : 
  f 2016 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_f_2016_value_l3152_315276


namespace NUMINAMATH_CALUDE_complex_power_sum_l3152_315248

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1000 + 1/(z^1000) = -2 * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3152_315248


namespace NUMINAMATH_CALUDE_range_of_a_l3152_315285

theorem range_of_a (a : ℝ) : 
  (∀ t ∈ Set.Ioo 0 2, t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2) → 
  a ∈ Set.Icc (2/13) 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3152_315285


namespace NUMINAMATH_CALUDE_prob_same_color_specific_l3152_315293

/-- The probability of selecting two plates of the same color -/
def prob_same_color (red blue green : ℕ) : ℚ :=
  let total := red + blue + green
  let same_color := (red.choose 2) + (blue.choose 2) + (green.choose 2)
  same_color / total.choose 2

/-- Theorem: The probability of selecting two plates of the same color
    given 6 red, 5 blue, and 3 green plates is 28/91 -/
theorem prob_same_color_specific : prob_same_color 6 5 3 = 28 / 91 := by
  sorry

#eval prob_same_color 6 5 3

end NUMINAMATH_CALUDE_prob_same_color_specific_l3152_315293


namespace NUMINAMATH_CALUDE_no_tetrahedron_with_heights_1_2_3_6_l3152_315262

/-- Represents a tetrahedron with face heights -/
structure Tetrahedron where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ

/-- The theorem stating that a tetrahedron with heights 1, 2, 3, and 6 cannot exist -/
theorem no_tetrahedron_with_heights_1_2_3_6 :
  ¬ ∃ (t : Tetrahedron), t.h₁ = 1 ∧ t.h₂ = 2 ∧ t.h₃ = 3 ∧ t.h₄ = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_tetrahedron_with_heights_1_2_3_6_l3152_315262


namespace NUMINAMATH_CALUDE_representatives_count_l3152_315281

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of boys -/
def numBoys : ℕ := 4

/-- The number of girls -/
def numGirls : ℕ := 4

/-- The total number of representatives to be selected -/
def totalReps : ℕ := 3

/-- The minimum number of girls to be selected -/
def minGirls : ℕ := 2

theorem representatives_count :
  (choose numGirls 2 * choose numBoys 1) + (choose numGirls 3) = 28 := by sorry

end NUMINAMATH_CALUDE_representatives_count_l3152_315281


namespace NUMINAMATH_CALUDE_soldiers_divisible_by_six_l3152_315254

theorem soldiers_divisible_by_six (b : ℕ+) : 
  ∃ k : ℕ, b + 3 * b ^ 2 + 2 * b ^ 3 = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_soldiers_divisible_by_six_l3152_315254


namespace NUMINAMATH_CALUDE_f_of_2_equals_2_l3152_315266

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Theorem statement
theorem f_of_2_equals_2 : f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_2_l3152_315266


namespace NUMINAMATH_CALUDE_wages_payment_duration_l3152_315222

/-- Given a sum of money that can pay two workers' wages separately for different periods,
    this theorem proves how long it can pay both workers together. -/
theorem wages_payment_duration (S : ℝ) (p q : ℝ) (hp : S = 24 * p) (hq : S = 40 * q) :
  ∃ D : ℝ, D = 15 ∧ S = D * (p + q) := by
  sorry

end NUMINAMATH_CALUDE_wages_payment_duration_l3152_315222


namespace NUMINAMATH_CALUDE_second_train_speed_l3152_315205

/-- Given two trains starting from the same station, traveling in the same direction
    on parallel tracks for 8 hours, with one train moving at 11 mph and ending up
    160 miles behind the other train, prove that the speed of the second train is 31 mph. -/
theorem second_train_speed (v : ℝ) : 
  v > 0 → -- The speed of the second train is positive
  (v * 8 - 11 * 8 = 160) → -- Distance difference after 8 hours
  v = 31 :=
by sorry

end NUMINAMATH_CALUDE_second_train_speed_l3152_315205


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3152_315246

theorem unique_solution_condition (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 + 1 ∧ p.2 = 4*p.1 + k) ↔ k = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3152_315246


namespace NUMINAMATH_CALUDE_B_power_99_l3152_315202

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, 0, 0;
     0, 0, 1;
     0, -1, 0]

theorem B_power_99 : B^99 = B := by sorry

end NUMINAMATH_CALUDE_B_power_99_l3152_315202


namespace NUMINAMATH_CALUDE_planes_with_parallel_lines_are_parallel_or_intersecting_l3152_315240

/-- Two planes in 3D space -/
structure Plane3D where
  -- Add necessary fields here
  
/-- A straight line in 3D space -/
structure Line3D where
  -- Add necessary fields here

/-- Predicate to check if a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate to check if two lines are parallel -/
def lines_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate to check if two planes are parallel -/
def planes_parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Predicate to check if two planes are intersecting -/
def planes_intersecting (p1 p2 : Plane3D) : Prop :=
  sorry

theorem planes_with_parallel_lines_are_parallel_or_intersecting 
  (p1 p2 : Plane3D) (l1 l2 : Line3D) 
  (h1 : line_in_plane l1 p1) 
  (h2 : line_in_plane l2 p2) 
  (h3 : lines_parallel l1 l2) : 
  planes_parallel p1 p2 ∨ planes_intersecting p1 p2 :=
sorry

end NUMINAMATH_CALUDE_planes_with_parallel_lines_are_parallel_or_intersecting_l3152_315240


namespace NUMINAMATH_CALUDE_oliver_ate_seventeen_fruits_l3152_315296

/-- The number of fruits Oliver ate -/
def fruits_eaten (initial_cherries initial_strawberries initial_blueberries
                  final_cherries final_strawberries final_blueberries : ℕ) : ℕ :=
  (initial_cherries - final_cherries) +
  (initial_strawberries - final_strawberries) +
  (initial_blueberries - final_blueberries)

/-- Theorem stating that Oliver ate 17 fruits in total -/
theorem oliver_ate_seventeen_fruits :
  fruits_eaten 16 10 20 6 8 15 = 17 := by
  sorry

end NUMINAMATH_CALUDE_oliver_ate_seventeen_fruits_l3152_315296


namespace NUMINAMATH_CALUDE_objective_function_range_l3152_315204

-- Define the constraint set
def ConstraintSet (x y : ℝ) : Prop :=
  x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ -1

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ := 3*x - y

-- State the theorem
theorem objective_function_range :
  ∀ x y : ℝ, ConstraintSet x y →
  ∃ z_min z_max : ℝ, z_min = -3/2 ∧ z_max = 6 ∧
  z_min ≤ ObjectiveFunction x y ∧ ObjectiveFunction x y ≤ z_max :=
sorry

end NUMINAMATH_CALUDE_objective_function_range_l3152_315204


namespace NUMINAMATH_CALUDE_sum_of_cubes_values_l3152_315203

open Complex Matrix

/-- A 3x3 circulant matrix with complex entries a, b, c -/
def M (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  !![a, b, c; b, c, a; c, a, b]

/-- The theorem statement -/
theorem sum_of_cubes_values (a b c : ℂ) : 
  M a b c ^ 2 = 1 → a * b * c = 1 → a^3 + b^3 + c^3 = 2 ∨ a^3 + b^3 + c^3 = 4 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_cubes_values_l3152_315203


namespace NUMINAMATH_CALUDE_gold_coins_in_urn_l3152_315288

-- Define the total percentage
def total_percentage : ℝ := 100

-- Define the percentage of beads
def bead_percentage : ℝ := 30

-- Define the percentage of silver coins among all coins
def silver_coin_percentage : ℝ := 50

-- Define the percentage of coins
def coin_percentage : ℝ := total_percentage - bead_percentage

-- Define the percentage of gold coins among all coins
def gold_coin_percentage : ℝ := total_percentage - silver_coin_percentage

-- Theorem to prove
theorem gold_coins_in_urn : 
  (coin_percentage * gold_coin_percentage) / total_percentage = 35 := by
  sorry

end NUMINAMATH_CALUDE_gold_coins_in_urn_l3152_315288


namespace NUMINAMATH_CALUDE_max_value_of_f_l3152_315269

/-- The function f(x) = -2x^2 + 4x + 10 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 10

/-- The maximum value of f(x) for x ≥ 0 is 12 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 12 ∧ ∀ (x : ℝ), x ≥ 0 → f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3152_315269


namespace NUMINAMATH_CALUDE_equation_solutions_l3152_315284

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 9 ↔ x = 5 ∨ x = -1) ∧
  (∀ x : ℝ, 2*x^2 - 3*x - 1 = 0 ↔ x = (3 + Real.sqrt 17) / 4 ∨ x = (3 - Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3152_315284


namespace NUMINAMATH_CALUDE_equation_solutions_l3152_315206

theorem equation_solutions : 
  {x : ℝ | x^4 + (3 - x)^4 = 130} = {0, 3} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3152_315206


namespace NUMINAMATH_CALUDE_prob_at_least_one_in_three_games_l3152_315277

/-- The probability of revealing a golden flower when smashing a single egg -/
def p : ℚ := 1/2

/-- The number of eggs smashed in each game -/
def n : ℕ := 3

/-- The number of games played -/
def games : ℕ := 3

/-- The probability of revealing at least one golden flower in a single game -/
def prob_at_least_one_in_game : ℚ := 1 - (1 - p)^n

/-- Theorem: The probability of revealing at least one golden flower in three games -/
theorem prob_at_least_one_in_three_games :
  (1 : ℚ) - (1 - prob_at_least_one_in_game)^games = 511/512 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_in_three_games_l3152_315277


namespace NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_2010_l3152_315295

theorem smallest_k_for_divisibility_by_2010 :
  ∃ (k : ℕ), k > 1 ∧
  (∀ (n : ℕ), n > 0 → (n^k - n) % 2010 = 0) ∧
  (∀ (m : ℕ), m > 1 ∧ m < k → ∃ (n : ℕ), n > 0 ∧ (n^m - n) % 2010 ≠ 0) ∧
  k = 133 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_2010_l3152_315295


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l3152_315236

theorem fixed_point_of_line (m : ℝ) : m * (-2) - 1 + 2 * m + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l3152_315236


namespace NUMINAMATH_CALUDE_greatest_y_value_l3152_315226

theorem greatest_y_value (y : ℕ) (h1 : y > 0) (h2 : ∃ k : ℕ, y = 4 * k) (h3 : y^3 < 8000) :
  y ≤ 16 ∧ ∃ (y' : ℕ), y' = 16 ∧ ∃ (k : ℕ), y' = 4 * k ∧ y'^3 < 8000 :=
sorry

end NUMINAMATH_CALUDE_greatest_y_value_l3152_315226


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3152_315289

theorem absolute_value_inequality (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3152_315289


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_l3152_315242

/-- The curve function f(x) = ax^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2

/-- The derivative of f(x) = ax^2 -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * a * x

/-- The slope of the line 4x - y + 4 = 0 -/
def line_slope : ℝ := 4

theorem tangent_perpendicular_implies_a (a : ℝ) :
  f a 2 = 4 * a ∧
  f_derivative a 2 * line_slope = -1 →
  a = -1/16 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_l3152_315242


namespace NUMINAMATH_CALUDE_negation_of_all_squared_nonnegative_l3152_315263

theorem negation_of_all_squared_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_squared_nonnegative_l3152_315263


namespace NUMINAMATH_CALUDE_child_tickets_sold_l3152_315237

/-- Proves the number of child tickets sold given ticket prices and total sales information -/
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
    child_tickets = 16 :=
by sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l3152_315237


namespace NUMINAMATH_CALUDE_min_value_theorem_solution_set_theorem_l3152_315201

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 1|

-- Theorem for part (1)
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = f (-1)) :
  (2/a + 1/b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = f (-1) ∧ 2/a₀ + 1/b₀ = 8 := by
  sorry

-- Theorem for part (2)
theorem solution_set_theorem (x : ℝ) :
  f x > 1/2 ↔ x < 5/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_solution_set_theorem_l3152_315201


namespace NUMINAMATH_CALUDE_soccer_team_lineups_l3152_315243

/-- The number of ways to select a starting lineup from a soccer team -/
def numStartingLineups (totalPlayers : ℕ) (regularPlayers : ℕ) : ℕ :=
  totalPlayers * Nat.choose (totalPlayers - 1) regularPlayers

/-- Theorem stating that the number of starting lineups for a team of 16 players,
    with 1 goalie and 10 regular players, is 48,048 -/
theorem soccer_team_lineups :
  numStartingLineups 16 10 = 48048 := by
  sorry

#eval numStartingLineups 16 10

end NUMINAMATH_CALUDE_soccer_team_lineups_l3152_315243


namespace NUMINAMATH_CALUDE_negPowersOfTwo_is_geometric_l3152_315279

/-- A sequence is geometric if it has a constant ratio between consecutive terms. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence of the form a_n = cq^n (where cq ≠ 0) is geometric. -/
axiom geometric_sequence_criterion (c q : ℝ) (hcq : c * q ≠ 0) :
  IsGeometricSequence (fun n => c * q ^ n)

/-- The sequence {-2^n} -/
def negPowersOfTwo (n : ℕ) : ℝ := -2 ^ n

/-- Theorem: The sequence {-2^n} is a geometric sequence -/
theorem negPowersOfTwo_is_geometric : IsGeometricSequence negPowersOfTwo := by
  sorry

end NUMINAMATH_CALUDE_negPowersOfTwo_is_geometric_l3152_315279


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l3152_315273

theorem salary_reduction_percentage
  (initial_increase : Real)
  (net_change : Real)
  (reduction_percentage : Real)
  (h1 : initial_increase = 0.25)
  (h2 : net_change = 0.0625)
  (h3 : (1 + initial_increase) * (1 - reduction_percentage) = 1 + net_change) :
  reduction_percentage = 0.15 := by
sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l3152_315273
