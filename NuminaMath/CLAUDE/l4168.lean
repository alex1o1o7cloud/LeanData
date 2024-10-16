import Mathlib

namespace NUMINAMATH_CALUDE_cycling_speed_l4168_416804

/-- The speed of Alice and Bob when cycling under specific conditions -/
theorem cycling_speed : ∃ (x : ℝ),
  (x^2 - 5*x - 14 = (x^2 + x - 20) / (x - 4)) ∧
  (x^2 - 5*x - 14 = 8 + 2*Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_cycling_speed_l4168_416804


namespace NUMINAMATH_CALUDE_blown_out_sand_dunes_with_treasure_l4168_416838

theorem blown_out_sand_dunes_with_treasure :
  let sand_dunes_remain_prob : ℚ := 1 / 3
  let sand_dunes_with_coupon_prob : ℚ := 2 / 3
  let total_blown_out_dunes : ℕ := 5
  let both_treasure_and_coupon_prob : ℚ := 8 / 90
  ∃ (treasure_dunes : ℕ),
    (treasure_dunes : ℚ) / total_blown_out_dunes * sand_dunes_with_coupon_prob = both_treasure_and_coupon_prob ∧
    treasure_dunes = 20 :=
by sorry

end NUMINAMATH_CALUDE_blown_out_sand_dunes_with_treasure_l4168_416838


namespace NUMINAMATH_CALUDE_divisors_of_2013_power_13_l4168_416869

theorem divisors_of_2013_power_13 : 
  let n : ℕ := 2013^13
  ∀ (p : ℕ → Prop), 
    (∀ k, p k ↔ k ∣ n ∧ k > 0) →
    (2013 = 3 * 11 * 61) →
    (∃! (s : Finset ℕ), ∀ k, k ∈ s ↔ p k) →
    Finset.card s = 2744 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_2013_power_13_l4168_416869


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l4168_416853

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ -4) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = -4) →
  a = 8 ∨ a = -8 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l4168_416853


namespace NUMINAMATH_CALUDE_mrs_hilt_markers_l4168_416823

theorem mrs_hilt_markers (num_packages : ℕ) (markers_per_package : ℕ) 
  (h1 : num_packages = 7) 
  (h2 : markers_per_package = 5) : 
  num_packages * markers_per_package = 35 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_markers_l4168_416823


namespace NUMINAMATH_CALUDE_local_minimum_at_one_l4168_416882

/-- The function f(x) = ax³ - 2x² + a²x has a local minimum at x=1 if and only if a = 1 -/
theorem local_minimum_at_one (a : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), 
    a*x^3 - 2*x^2 + a^2*x ≥ a*1^3 - 2*1^2 + a^2*1) ↔ a = 1 := by
  sorry


end NUMINAMATH_CALUDE_local_minimum_at_one_l4168_416882


namespace NUMINAMATH_CALUDE_ten_spheres_melted_l4168_416847

/-- The radius of a sphere formed by melting multiple smaller spheres -/
def large_sphere_radius (n : ℕ) (r : ℝ) : ℝ :=
  (n * r ^ 3) ^ (1/3)

/-- Theorem: The radius of a sphere formed by melting 10 smaller spheres 
    with radius 3 inches is equal to the cube root of 270 inches -/
theorem ten_spheres_melted (n : ℕ) (r : ℝ) : 
  n = 10 ∧ r = 3 → large_sphere_radius n r = 270 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ten_spheres_melted_l4168_416847


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l4168_416841

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.2 : ℝ)⌉ = 27 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l4168_416841


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4168_416837

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x^2 + y^2 = 0 → x * y = 0) ∧
  (∃ x y, x * y = 0 ∧ x^2 + y^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4168_416837


namespace NUMINAMATH_CALUDE_roll_distribution_probability_l4168_416842

/-- The number of guests -/
def num_guests : ℕ := 4

/-- The number of roll types -/
def num_roll_types : ℕ := 4

/-- The total number of rolls -/
def total_rolls : ℕ := 16

/-- The number of rolls per type -/
def rolls_per_type : ℕ := 4

/-- The number of rolls given to each guest -/
def rolls_per_guest : ℕ := 4

/-- The probability of each guest receiving one of each type of roll -/
def probability_each_guest_gets_one_of_each : ℚ := 1 / 6028032000

theorem roll_distribution_probability :
  probability_each_guest_gets_one_of_each = 
    (rolls_per_type / total_rolls) *
    ((rolls_per_type - 1) / (total_rolls - 1)) *
    ((rolls_per_type - 2) / (total_rolls - 2)) *
    ((rolls_per_type - 3) / (total_rolls - 3)) *
    ((rolls_per_type - 1) / (total_rolls - 4)) *
    ((rolls_per_type - 2) / (total_rolls - 5)) *
    ((rolls_per_type - 3) / (total_rolls - 6)) *
    ((rolls_per_type - 2) / (total_rolls - 8)) *
    ((rolls_per_type - 3) / (total_rolls - 9)) *
    ((rolls_per_type - 3) / (total_rolls - 12)) := by
  sorry

#eval probability_each_guest_gets_one_of_each

end NUMINAMATH_CALUDE_roll_distribution_probability_l4168_416842


namespace NUMINAMATH_CALUDE_tree_planting_percentage_l4168_416856

theorem tree_planting_percentage (boys : ℕ) (girls : ℕ) : 
  boys = 600 → 
  girls = boys + 400 → 
  (960 : ℝ) / (boys + girls : ℝ) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_percentage_l4168_416856


namespace NUMINAMATH_CALUDE_solve_trading_card_problem_l4168_416816

def trading_card_problem (initial_cards : ℕ) (brother_sets : ℕ) (friend_sets : ℕ) 
  (total_given : ℕ) (cards_per_set : ℕ) : ℕ :=
  let cards_to_brother := brother_sets * cards_per_set
  let cards_to_friend := friend_sets * cards_per_set
  let remaining_cards := total_given - (cards_to_brother + cards_to_friend)
  remaining_cards / cards_per_set

theorem solve_trading_card_problem :
  trading_card_problem 365 8 2 195 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_trading_card_problem_l4168_416816


namespace NUMINAMATH_CALUDE_area_at_stage_5_is_24_l4168_416843

/-- Calculates the length of the rectangle at a given stage -/
def length_at_stage (stage : ℕ) : ℕ := 4 + 2 * (stage - 1)

/-- Calculates the area of the rectangle at a given stage -/
def area_at_stage (stage : ℕ) : ℕ := length_at_stage stage * 2

/-- Theorem stating that the area at Stage 5 is 24 square inches -/
theorem area_at_stage_5_is_24 : area_at_stage 5 = 24 := by
  sorry

#eval area_at_stage 5  -- This should output 24

end NUMINAMATH_CALUDE_area_at_stage_5_is_24_l4168_416843


namespace NUMINAMATH_CALUDE_five_letter_words_with_consonant_l4168_416848

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def word_length : Nat := 5

theorem five_letter_words_with_consonant :
  (alphabet.card ^ word_length) - (vowels.card ^ word_length) = 7744 :=
by sorry

end NUMINAMATH_CALUDE_five_letter_words_with_consonant_l4168_416848


namespace NUMINAMATH_CALUDE_skate_cost_theorem_l4168_416825

/-- The cost of a new pair of skates is equal to 26 times the rental fee. -/
theorem skate_cost_theorem (admission_fee : ℝ) (rental_fee : ℝ) (visits : ℕ) 
  (h1 : admission_fee = 5)
  (h2 : rental_fee = 2.5)
  (h3 : visits = 26) :
  visits * rental_fee = 65 := by
  sorry

#check skate_cost_theorem

end NUMINAMATH_CALUDE_skate_cost_theorem_l4168_416825


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l4168_416857

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 + (n / 10) % 10 + n % 10 = 14) ∧
  ((n / 10) % 10 = (n / 100) + (n % 10)) ∧
  (n - (n % 10 * 100 + (n / 10) % 10 * 10 + n / 100) = 99)

theorem three_digit_number_proof :
  is_valid_number 473 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l4168_416857


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l4168_416828

/-- The area of a square with adjacent vertices at (1, -2) and (4, 1) on a Cartesian coordinate plane is 18. -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, -2)
  let p2 : ℝ × ℝ := (4, 1)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l4168_416828


namespace NUMINAMATH_CALUDE_possible_m_values_l4168_416836

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem possible_m_values (m : ℝ) : A ∪ B m = A → m = 0 ∨ m = -1/3 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_possible_m_values_l4168_416836


namespace NUMINAMATH_CALUDE_isabellas_house_paintable_area_l4168_416813

/-- Calculates the total paintable wall area in a house with specified room dimensions and non-paintable areas. -/
def total_paintable_area (num_bedrooms num_bathrooms : ℕ)
                         (bedroom_length bedroom_width bedroom_height : ℝ)
                         (bathroom_length bathroom_width bathroom_height : ℝ)
                         (bedroom_nonpaintable_area bathroom_nonpaintable_area : ℝ) : ℝ :=
  let bedroom_wall_area := 2 * (bedroom_length * bedroom_height + bedroom_width * bedroom_height)
  let bathroom_wall_area := 2 * (bathroom_length * bathroom_height + bathroom_width * bathroom_height)
  let paintable_bedroom_area := bedroom_wall_area - bedroom_nonpaintable_area
  let paintable_bathroom_area := bathroom_wall_area - bathroom_nonpaintable_area
  num_bedrooms * paintable_bedroom_area + num_bathrooms * paintable_bathroom_area

/-- The total paintable wall area in Isabella's house is 1637 square feet. -/
theorem isabellas_house_paintable_area :
  total_paintable_area 3 2 14 11 9 9 7 8 55 30 = 1637 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_house_paintable_area_l4168_416813


namespace NUMINAMATH_CALUDE_assignment_schemes_eq_240_l4168_416800

/-- The number of ways to assign 4 out of 6 students to tasks A, B, C, and D,
    given that two specific students can perform task A. -/
def assignment_schemes : ℕ :=
  Nat.descFactorial 6 4 - 2 * Nat.descFactorial 5 3

/-- Theorem stating that the number of assignment schemes is 240. -/
theorem assignment_schemes_eq_240 : assignment_schemes = 240 := by
  sorry

end NUMINAMATH_CALUDE_assignment_schemes_eq_240_l4168_416800


namespace NUMINAMATH_CALUDE_only_seven_has_integer_solution_solutions_for_seven_l4168_416835

/-- The product of terms (1 + 1/(x+k)) from k = 0 to n -/
def productTerm (x : ℤ) (n : ℕ) : ℚ :=
  (List.range (n + 1)).foldl (fun acc k => acc * (1 + 1 / (x + k))) 1

/-- The main theorem stating that 7 is the only positive integer solution -/
theorem only_seven_has_integer_solution :
  ∀ a : ℕ+, (∃ x : ℤ, productTerm x a = a - x) ↔ a = 7 := by
  sorry

/-- Verification of the two integer solutions for a = 7 -/
theorem solutions_for_seven :
  productTerm 2 7 = 5 ∧ productTerm 4 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_only_seven_has_integer_solution_solutions_for_seven_l4168_416835


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l4168_416884

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- Define the domain
def domain (x : ℝ) : Prop := -2 < x ∧ x < 4

-- Theorem statement
theorem f_has_max_and_min :
  ∃ (x_max x_min : ℝ),
    domain x_max ∧ domain x_min ∧
    (∀ x, domain x → f x ≤ f x_max) ∧
    (∀ x, domain x → f x_min ≤ f x) ∧
    f x_max = 5 ∧ f x_min = -27 ∧
    x_max = -1 ∧ x_min = 3 :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l4168_416884


namespace NUMINAMATH_CALUDE_carolyns_silverware_after_trade_l4168_416858

/-- Represents Carolyn's silverware set -/
structure SilverwareSet where
  knives : ℕ
  forks : ℕ
  spoons : ℕ

/-- Calculates the total number of pieces in a silverware set -/
def totalPieces (set : SilverwareSet) : ℕ :=
  set.knives + set.forks + set.spoons

/-- Represents the trade operation -/
def trade (set : SilverwareSet) (knivesGained : ℕ) (spoonsLost : ℕ) : SilverwareSet :=
  { knives := set.knives + knivesGained,
    forks := set.forks,
    spoons := set.spoons - spoonsLost }

/-- Calculates the percentage of knives in a silverware set -/
def knifePercentage (set : SilverwareSet) : ℚ :=
  (set.knives : ℚ) / (totalPieces set : ℚ) * 100

theorem carolyns_silverware_after_trade :
  let initialSet : SilverwareSet := { knives := 6, forks := 12, spoons := 6 * 3 }
  let finalSet := trade initialSet 10 6
  knifePercentage finalSet = 40 := by sorry

end NUMINAMATH_CALUDE_carolyns_silverware_after_trade_l4168_416858


namespace NUMINAMATH_CALUDE_vector_proof_l4168_416873

/-- Given two planar vectors a and b, with a parallel to b, and a linear combination
    of these vectors with a third vector c equal to the zero vector,
    prove that c is equal to (-7, 14). -/
theorem vector_proof (a b c : ℝ × ℝ) (m : ℝ) : 
  a = (1, -2) →
  b = (2, m) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  3 • a + 2 • b + c = (0, 0) →
  c = (-7, 14) := by
  sorry

end NUMINAMATH_CALUDE_vector_proof_l4168_416873


namespace NUMINAMATH_CALUDE_arithmetic_progression_difference_divisibility_l4168_416815

theorem arithmetic_progression_difference_divisibility
  (p : ℕ) (a : ℕ → ℕ) (d : ℕ) 
  (h_p_prime : Nat.Prime p)
  (h_a_prime : ∀ i, i ∈ Finset.range p → Nat.Prime (a i))
  (h_arithmetic_progression : ∀ i, i ∈ Finset.range (p - 1) → a (i + 1) = a i + d)
  (h_increasing : ∀ i j, i < j → j < p → a i < a j)
  (h_a1_gt_p : a 0 > p) :
  p ∣ d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_difference_divisibility_l4168_416815


namespace NUMINAMATH_CALUDE_quadruple_solutions_l4168_416895

theorem quadruple_solutions : 
  ∀ a b c d : ℝ, 
    (a * b + c * d = 6) ∧ 
    (a * c + b * d = 3) ∧ 
    (a * d + b * c = 2) ∧ 
    (a + b + c + d = 6) → 
    ((a = 0 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
     (a = 2 ∧ b = 3 ∧ c = 0 ∧ d = 1) ∨
     (a = 1 ∧ b = 0 ∧ c = 3 ∧ d = 2) ∨
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_solutions_l4168_416895


namespace NUMINAMATH_CALUDE_max_sum_xyz_l4168_416879

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorial (x : ℕ) : ℕ := (List.range x).map factorial |>.sum

theorem max_sum_xyz (x y z : ℕ) (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_eq : sum_factorial x = y^2) :
  x + y + z ≤ 8 := by sorry

end NUMINAMATH_CALUDE_max_sum_xyz_l4168_416879


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4168_416809

theorem expression_simplification_and_evaluation :
  let x : ℝ := 6 * Real.sin (30 * π / 180) - Real.sqrt 2 * Real.cos (45 * π / 180)
  ((x / (x - 2) - x / (x + 2)) / (4 * x / (x - 2))) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4168_416809


namespace NUMINAMATH_CALUDE_perception_permutations_l4168_416801

/-- The number of letters in the word "PERCEPTION" -/
def word_length : ℕ := 10

/-- The number of times 'P' appears in "PERCEPTION" -/
def p_count : ℕ := 2

/-- The number of times 'E' appears in "PERCEPTION" -/
def e_count : ℕ := 2

/-- The number of unique arrangements of the letters in "PERCEPTION" -/
def perception_arrangements : ℕ := 907200

theorem perception_permutations :
  perception_arrangements = (Nat.factorial word_length) / ((Nat.factorial p_count) * (Nat.factorial e_count)) :=
by sorry

end NUMINAMATH_CALUDE_perception_permutations_l4168_416801


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4168_416871

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + a₃ + a₅ = 123 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4168_416871


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l4168_416896

theorem fraction_sum_theorem (x y : ℝ) (h : x ≠ y) :
  (x + y) / (x - y) + (x - y) / (x + y) = 3 →
  (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = 13/6 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l4168_416896


namespace NUMINAMATH_CALUDE_cardinality_of_P_l4168_416863

def M : Finset ℕ := {0, 1, 2, 3, 4}
def N : Finset ℕ := {1, 3, 5}
def P : Finset ℕ := M ∩ N

theorem cardinality_of_P : Finset.card P = 2 := by sorry

end NUMINAMATH_CALUDE_cardinality_of_P_l4168_416863


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l4168_416878

theorem smallest_constant_inequality (x y z : ℝ) (h : x + y + z = -1) :
  ∃ C : ℝ, (∀ x y z : ℝ, x + y + z = -1 → |x^3 + y^3 + z^3 + 1| ≤ C * |x^5 + y^5 + z^5 + 1|) ∧
  C = 9/10 ∧
  ∀ C' : ℝ, (∀ x y z : ℝ, x + y + z = -1 → |x^3 + y^3 + z^3 + 1| ≤ C' * |x^5 + y^5 + z^5 + 1|) → C' ≥ 9/10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l4168_416878


namespace NUMINAMATH_CALUDE_uneven_picture_distribution_l4168_416874

theorem uneven_picture_distribution (total_pictures : Nat) (num_albums : Nat) : 
  total_pictures = 101 → num_albums = 7 → ¬∃ (pics_per_album : Nat), total_pictures = num_albums * pics_per_album :=
by
  sorry

end NUMINAMATH_CALUDE_uneven_picture_distribution_l4168_416874


namespace NUMINAMATH_CALUDE_geraldine_jazmin_doll_difference_l4168_416844

theorem geraldine_jazmin_doll_difference : 
  let geraldine_dolls : ℝ := 2186.0
  let jazmin_dolls : ℝ := 1209.0
  geraldine_dolls - jazmin_dolls = 977.0 := by
  sorry

end NUMINAMATH_CALUDE_geraldine_jazmin_doll_difference_l4168_416844


namespace NUMINAMATH_CALUDE_cube_plus_linear_inequality_l4168_416851

theorem cube_plus_linear_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4 * a * b := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_linear_inequality_l4168_416851


namespace NUMINAMATH_CALUDE_number_puzzle_l4168_416897

theorem number_puzzle : 
  ∀ x : ℚ, (x / 7 - x / 11 = 100) → x = 1925 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l4168_416897


namespace NUMINAMATH_CALUDE_new_student_weight_l4168_416893

/-- Theorem: Weight of the new student when average weight decreases --/
theorem new_student_weight
  (n : ℕ) -- number of students
  (w : ℕ) -- weight of the replaced student
  (d : ℕ) -- decrease in average weight
  (h1 : n = 8)
  (h2 : w = 86)
  (h3 : d = 5)
  : ∃ (new_weight : ℕ), 
    (n : ℝ) * d = w - new_weight ∧ new_weight = 46 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l4168_416893


namespace NUMINAMATH_CALUDE_seventeen_meter_rod_pieces_l4168_416824

/-- The number of pieces of a given length that can be cut from a rod --/
def num_pieces (rod_length : ℕ) (piece_length : ℕ) : ℕ :=
  rod_length / piece_length

/-- Theorem: The number of 85 cm pieces that can be cut from a 17-meter rod is 20 --/
theorem seventeen_meter_rod_pieces : num_pieces (17 * 100) 85 = 20 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_meter_rod_pieces_l4168_416824


namespace NUMINAMATH_CALUDE_inequality_solution_implies_k_value_l4168_416814

theorem inequality_solution_implies_k_value (k : ℚ) :
  (∀ x : ℚ, 3 * x - (2 * k - 3) < 4 * x + 3 * k + 6 ↔ x > 1) →
  k = -4/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_k_value_l4168_416814


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l4168_416832

theorem problems_left_to_grade 
  (problems_per_worksheet : ℕ) 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (h1 : problems_per_worksheet = 4)
  (h2 : total_worksheets = 9)
  (h3 : graded_worksheets = 5) :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 16 :=
by sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l4168_416832


namespace NUMINAMATH_CALUDE_proper_subsets_of_B_l4168_416834

-- Define the sets A and B
def A (b : ℝ) : Set ℝ := {x | x^2 + (b+2)*x + b + 1 = 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

-- State the theorem
theorem proper_subsets_of_B (b : ℝ) (a : ℝ) (h : A b = {a}) :
  {s : Set ℝ | s ⊂ B a b ∧ s ≠ B a b} = {∅, {1}, {0}} := by
  sorry

end NUMINAMATH_CALUDE_proper_subsets_of_B_l4168_416834


namespace NUMINAMATH_CALUDE_gravel_pile_volume_l4168_416892

/-- The volume of a hemispherical pile of gravel -/
theorem gravel_pile_volume (d : ℝ) (h : ℝ) (v : ℝ) : 
  d = 10 → -- diameter is 10 feet
  h = d / 2 → -- height is half the diameter
  v = (250 * Real.pi) / 3 → -- volume is (250π)/3 cubic feet
  v = (2 / 3) * Real.pi * (d / 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_gravel_pile_volume_l4168_416892


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l4168_416803

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_arithmetic a 15 = 0) :
  a 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l4168_416803


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l4168_416810

theorem rectangular_plot_length_difference (length breadth perimeter : ℝ) : 
  length = 63 ∧ 
  perimeter = 200 ∧ 
  perimeter = 2 * (length + breadth) → 
  length - breadth = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l4168_416810


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l4168_416845

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 231 → n = 22 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l4168_416845


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l4168_416862

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (Real.sqrt a + Real.sqrt b)^8 ≥ 64 * a * b * (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l4168_416862


namespace NUMINAMATH_CALUDE_fruit_cost_prices_l4168_416888

/-- Represents the cost and selling prices of fruits -/
structure FruitPrices where
  apple_sell : ℚ
  orange_sell : ℚ
  banana_sell : ℚ
  apple_loss : ℚ
  orange_loss : ℚ
  banana_gain : ℚ

/-- Calculates the cost price given selling price and loss/gain percentage -/
def cost_price (sell : ℚ) (loss_gain : ℚ) (is_gain : Bool) : ℚ :=
  if is_gain then
    sell / (1 + loss_gain)
  else
    sell / (1 - loss_gain)

/-- Theorem stating the correct cost prices for the fruits -/
theorem fruit_cost_prices (prices : FruitPrices)
  (h_apple_sell : prices.apple_sell = 20)
  (h_orange_sell : prices.orange_sell = 15)
  (h_banana_sell : prices.banana_sell = 6)
  (h_apple_loss : prices.apple_loss = 1/6)
  (h_orange_loss : prices.orange_loss = 1/4)
  (h_banana_gain : prices.banana_gain = 1/8) :
  cost_price prices.apple_sell prices.apple_loss false = 24 ∧
  cost_price prices.orange_sell prices.orange_loss false = 20 ∧
  cost_price prices.banana_sell prices.banana_gain true = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_prices_l4168_416888


namespace NUMINAMATH_CALUDE_birds_in_tree_l4168_416830

theorem birds_in_tree (initial_birds final_birds : ℕ) : 
  initial_birds = 14 → final_birds = 35 → final_birds - initial_birds = 21 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l4168_416830


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_ratio_l4168_416861

/-- Theorem: For a hyperbola with equation x^2/a^2 - y^2/b^2 = 1, where a > 0, b > 0,
    and eccentricity = 2, the ratio b/a equals √3. -/
theorem hyperbola_eccentricity_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c / a = 2) →
  b / a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_ratio_l4168_416861


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l4168_416864

/-- Represents the configuration of rectangles around a square -/
structure SquareWithRectangles where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The conditions of the problem -/
def problem_conditions (config : SquareWithRectangles) : Prop :=
  -- The area of the outer square is 9 times that of the inner square
  (config.inner_square_side + 2 * config.rectangle_short_side) ^ 2 = 9 * config.inner_square_side ^ 2 ∧
  -- The outer square's side length is composed of the inner square and two short sides of rectangles
  config.inner_square_side + 2 * config.rectangle_short_side = 
    config.rectangle_long_side + config.rectangle_short_side

/-- The theorem to prove -/
theorem rectangle_ratio_is_two (config : SquareWithRectangles) 
  (h : problem_conditions config) : 
  config.rectangle_long_side / config.rectangle_short_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l4168_416864


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l4168_416819

theorem consecutive_pages_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 285 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l4168_416819


namespace NUMINAMATH_CALUDE_missing_village_population_l4168_416811

def village_count : Nat := 7
def known_populations : List Nat := [803, 900, 1100, 1023, 980, 1249]
def average_population : Nat := 1000

theorem missing_village_population :
  village_count * average_population - known_populations.sum = 945 := by
  sorry

end NUMINAMATH_CALUDE_missing_village_population_l4168_416811


namespace NUMINAMATH_CALUDE_parabola_equation_l4168_416852

/-- A vertical line passing through a point (x₀, y₀) -/
structure VerticalLine where
  x₀ : ℝ
  y₀ : ℝ

/-- A parabola with vertical axis and vertex at origin -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := λ x y => y^2 = -2 * p * x

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation (l : VerticalLine) (para : Parabola) :
  l.x₀ = 3/2 ∧ l.y₀ = 2 ∧ para.eq = λ x y => y^2 = -6*x := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l4168_416852


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4168_416866

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (3 * x₁^2 - 3 * x₁ - 4 = 0) ∧ (3 * x₂^2 - 3 * x₂ - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4168_416866


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l4168_416886

theorem max_product_under_constraint (a b : ℝ) :
  a > 0 → b > 0 → 5 * a + 8 * b = 80 → ab ≤ 40 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ 5 * a + 8 * b = 80 ∧ a * b = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l4168_416886


namespace NUMINAMATH_CALUDE_problem_solution_l4168_416850

theorem problem_solution (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k)*(x - k) = x^3 - k*(x^2 + x + 3)) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4168_416850


namespace NUMINAMATH_CALUDE_least_marbles_divisible_l4168_416854

theorem least_marbles_divisible (n : ℕ) : n > 0 ∧ 
  (∀ k ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ), n % k = 0) →
  n ≥ 420 :=
by sorry

end NUMINAMATH_CALUDE_least_marbles_divisible_l4168_416854


namespace NUMINAMATH_CALUDE_decimal_sum_l4168_416898

theorem decimal_sum : 5.467 + 2.349 + 3.785 = 11.751 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_l4168_416898


namespace NUMINAMATH_CALUDE_no_special_eight_digit_number_l4168_416822

theorem no_special_eight_digit_number : ¬∃ N : ℕ,
  (10000000 ≤ N ∧ N < 100000000) ∧
  (∀ i : Fin 8, 
    let digit := (N / (10 ^ (7 - i.val))) % 10
    digit ≠ 0 ∧
    N % digit = i.val + 1) :=
sorry

end NUMINAMATH_CALUDE_no_special_eight_digit_number_l4168_416822


namespace NUMINAMATH_CALUDE_ratio_problem_l4168_416817

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) :
  x / y = 11 / 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l4168_416817


namespace NUMINAMATH_CALUDE_sequence_general_term_l4168_416867

theorem sequence_general_term (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ k, S k = k^2 + k) →
  (a 1 = S 1) →
  (∀ k ≥ 2, a k = S k - S (k - 1)) →
  ∀ k, a k = 2 * k :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l4168_416867


namespace NUMINAMATH_CALUDE_money_division_l4168_416880

theorem money_division (a b c : ℝ) 
  (h1 : a = (1/3) * (b + c))
  (h2 : b = (2/7) * (a + c))
  (h3 : a = b + 15) :
  a + b + c = 540 := by
sorry

end NUMINAMATH_CALUDE_money_division_l4168_416880


namespace NUMINAMATH_CALUDE_jills_earnings_ratio_l4168_416818

/-- Jill's earnings over three months --/
def total_earnings : ℝ := 1200

/-- Jill's daily earnings in the first month --/
def first_month_daily : ℝ := 10

/-- Number of days in each month --/
def days_per_month : ℕ := 30

/-- Ratio of second month's daily earnings to first month's daily earnings --/
def earnings_ratio : ℝ := 2

theorem jills_earnings_ratio : 
  ∃ (second_month_daily : ℝ),
    first_month_daily * days_per_month +
    second_month_daily * days_per_month +
    second_month_daily * (days_per_month / 2) = total_earnings ∧
    second_month_daily / first_month_daily = earnings_ratio :=
by sorry

end NUMINAMATH_CALUDE_jills_earnings_ratio_l4168_416818


namespace NUMINAMATH_CALUDE_angle_420_equals_60_l4168_416868

/-- The angle (in degrees) that represents a full rotation in a standard coordinate system -/
def full_rotation : ℝ := 360

/-- Two angles have the same terminal side if their difference is a multiple of a full rotation -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α - β = k * full_rotation

/-- Theorem: The angle 420° has the same terminal side as 60° -/
theorem angle_420_equals_60 : same_terminal_side 420 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_420_equals_60_l4168_416868


namespace NUMINAMATH_CALUDE_circle_area_ratio_l4168_416829

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 : ℝ) * (2 * Real.pi * r₁) = (30 / 360 : ℝ) * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l4168_416829


namespace NUMINAMATH_CALUDE_initial_amount_problem_l4168_416820

theorem initial_amount_problem (initial_amount : ℝ) : 
  (initial_amount * (1 + 1/8) * (1 + 1/8) = 97200) → initial_amount = 76800 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_problem_l4168_416820


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainder_l4168_416802

theorem least_positive_integer_with_remainder (n : ℕ) : n = 662 ↔ 
  (n > 1) ∧ 
  (∀ d ∈ ({3, 4, 5, 11} : Set ℕ), n % d = 2) ∧ 
  (∀ m : ℕ, m > 1 → (∀ d ∈ ({3, 4, 5, 11} : Set ℕ), m % d = 2) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainder_l4168_416802


namespace NUMINAMATH_CALUDE_sum_equals_1998_l4168_416827

theorem sum_equals_1998 (a b c d : ℕ) (h : a * c + b * d + a * d + b * c = 1997) :
  a + b + c + d = 1998 := by sorry

end NUMINAMATH_CALUDE_sum_equals_1998_l4168_416827


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4168_416885

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  -3 + 23*x - x^2 + 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4168_416885


namespace NUMINAMATH_CALUDE_rectangular_to_polar_equivalence_l4168_416876

/-- Given a curve C in the xy-plane, prove that its rectangular coordinate equation
    x^2 + y^2 - 2x = 0 is equivalent to the polar coordinate equation ρ = 2cosθ. -/
theorem rectangular_to_polar_equivalence :
  ∀ (x y ρ θ : ℝ),
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x^2 + y^2 - 2*x = 0) ↔ (ρ = 2 * Real.cos θ) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_equivalence_l4168_416876


namespace NUMINAMATH_CALUDE_f_max_and_roots_l4168_416875

/-- The function f(x) defined as x(x-m)^2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x * (x - m)^2

theorem f_max_and_roots (m : ℝ) :
  (∃ (x_max : ℝ), x_max = 2 ∧ ∀ x, f m x ≤ f m x_max) →
  (m = 6 ∧
   ∀ a : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
                            f m x₁ = a ∧ f m x₂ = a ∧ f m x₃ = a) →
             0 < a ∧ a < 32) :=
by sorry

end NUMINAMATH_CALUDE_f_max_and_roots_l4168_416875


namespace NUMINAMATH_CALUDE_rationalize_denominator_l4168_416872

theorem rationalize_denominator : 
  ∃ (a b : ℝ) (h : b ≠ 0), (7 / (2 * Real.sqrt 98)) = a / b ∧ b * Real.sqrt b = b := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l4168_416872


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l4168_416889

theorem mean_of_added_numbers (original_count : ℕ) (original_mean : ℚ) 
  (new_count : ℕ) (new_mean : ℚ) (x y z : ℚ) : 
  original_count = 7 →
  original_mean = 40 →
  new_count = original_count + 3 →
  new_mean = 50 →
  (original_count * original_mean + x + y + z) / new_count = new_mean →
  (x + y + z) / 3 = 220 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l4168_416889


namespace NUMINAMATH_CALUDE_simple_interest_time_calculation_l4168_416821

/-- Simple interest calculation theorem -/
theorem simple_interest_time_calculation
  (P : ℝ) (R : ℝ) (SI : ℝ)
  (h_P : P = 800)
  (h_R : R = 6.25)
  (h_SI : SI = 200)
  (h_formula : SI = P * R * (SI * 100 / (P * R)) / 100) :
  SI * 100 / (P * R) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_calculation_l4168_416821


namespace NUMINAMATH_CALUDE_sallys_cards_l4168_416865

/-- The number of Pokemon cards Sally had initially -/
def initial_cards : ℕ := 27

/-- The number of cards Dan gave to Sally -/
def dans_cards : ℕ := 41

/-- The number of cards Sally bought -/
def bought_cards : ℕ := 20

/-- The total number of cards Sally has now -/
def total_cards : ℕ := 88

/-- Theorem stating that the initial number of cards plus the acquired cards equals the total cards -/
theorem sallys_cards : initial_cards + dans_cards + bought_cards = total_cards := by
  sorry

end NUMINAMATH_CALUDE_sallys_cards_l4168_416865


namespace NUMINAMATH_CALUDE_johns_age_l4168_416807

theorem johns_age (john grandmother : ℕ) 
  (age_difference : john = grandmother - 48)
  (sum_of_ages : john + grandmother = 100) :
  john = 26 := by
sorry

end NUMINAMATH_CALUDE_johns_age_l4168_416807


namespace NUMINAMATH_CALUDE_cube_volume_l4168_416860

theorem cube_volume (edge : ℝ) (h : edge = 7) : edge^3 = 343 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l4168_416860


namespace NUMINAMATH_CALUDE_digit_difference_zero_l4168_416846

/-- Given two digits A and B in base d > 8, if AB̅_d + AA̅_d = 234_d, then A_d - B_d = 0_d -/
theorem digit_difference_zero (d A B : ℕ) (h1 : d > 8) 
  (h2 : A < d) (h3 : B < d) 
  (h4 : A * d + B + A * d + A = 2 * d + 3 * d + 4) : 
  A - B = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_zero_l4168_416846


namespace NUMINAMATH_CALUDE_surface_area_ratio_l4168_416849

/-- The ratio of the surface area of a cube to the surface area of a rectangular solid
    with dimensions 2L, 3W, and 4H, where L, W, H are the cube's dimensions. -/
theorem surface_area_ratio (s : ℝ) (h : s > 0) : 
  (6 * s^2) / (2 * (2*s) * (3*s) + 2 * (2*s) * (4*s) + 2 * (3*s) * (4*s)) = 3 / 26 := by
  sorry

#check surface_area_ratio

end NUMINAMATH_CALUDE_surface_area_ratio_l4168_416849


namespace NUMINAMATH_CALUDE_least_cans_required_l4168_416831

theorem least_cans_required (a b c d e f g h : ℕ+) : 
  a = 139 → b = 223 → c = 179 → d = 199 → e = 173 → f = 211 → g = 131 → h = 257 →
  (∃ (x : ℕ+), x = a + b + c + d + e + f + g + h ∧ 
   x = Nat.gcd a (Nat.gcd b (Nat.gcd c (Nat.gcd d (Nat.gcd e (Nat.gcd f (Nat.gcd g h))))))) :=
by sorry

end NUMINAMATH_CALUDE_least_cans_required_l4168_416831


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l4168_416890

theorem fraction_zero_implies_x_negative_two (x : ℝ) : 
  (x^2 - 4) / (x^2 - 4*x + 4) = 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l4168_416890


namespace NUMINAMATH_CALUDE_square_area_ratio_l4168_416899

theorem square_area_ratio : 
  ∀ (a b : ℝ), 
  (4 * a = 16 * b) →  -- Perimeter relation
  (a = 2 * b + 5) →   -- Side length relation
  (a^2 / b^2 = 16) := by  -- Area ratio
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l4168_416899


namespace NUMINAMATH_CALUDE_sqrt_sum_quotient_l4168_416859

theorem sqrt_sum_quotient : (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 75 = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_quotient_l4168_416859


namespace NUMINAMATH_CALUDE_inequality_proof_l4168_416806

theorem inequality_proof (a b c : ℝ) (k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 2) (habc : a * b * c = 1) :
  (a^k / (a + b)) + (b^k / (b + c)) + (c^k / (c + a)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4168_416806


namespace NUMINAMATH_CALUDE_adult_meal_cost_l4168_416883

/-- Proves that the cost of each adult meal is $3 given the specified conditions -/
theorem adult_meal_cost (total_people : Nat) (kids : Nat) (total_cost : Nat) :
  total_people = 12 →
  kids = 7 →
  total_cost = 15 →
  (total_cost / (total_people - kids) : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_adult_meal_cost_l4168_416883


namespace NUMINAMATH_CALUDE_difference_of_roots_quadratic_l4168_416881

theorem difference_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → |r₁ - r₂| = 3 :=
by
  sorry

#check difference_of_roots_quadratic 1 (-9) 18

end NUMINAMATH_CALUDE_difference_of_roots_quadratic_l4168_416881


namespace NUMINAMATH_CALUDE_square_99_is_white_l4168_416839

def Grid := Fin 9 → Fin 9 → Bool

def is_adjacent (x1 y1 x2 y2 : Fin 9) : Prop :=
  (x1 = x2 ∧ y1.val + 1 = y2.val) ∨
  (x1 = x2 ∧ y2.val + 1 = y1.val) ∨
  (y1 = y2 ∧ x1.val + 1 = x2.val) ∨
  (y1 = y2 ∧ x2.val + 1 = x1.val)

def valid_grid (g : Grid) : Prop :=
  (g 4 4 = true) ∧
  (g 4 9 = true) ∧
  (∀ x y, g x y → (∃! x' y', is_adjacent x y x' y' ∧ g x' y')) ∧
  (∀ x y, ¬g x y → (∃! x' y', is_adjacent x y x' y' ∧ ¬g x' y'))

theorem square_99_is_white (g : Grid) (h : valid_grid g) : g 9 9 = false := by
  sorry

#check square_99_is_white

end NUMINAMATH_CALUDE_square_99_is_white_l4168_416839


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l4168_416833

/-- Proves that given a train of length 360 meters traveling at 36 km/hour,
    if it takes 50 seconds to pass a bridge, then the length of the bridge is 140 meters. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  train_length = 360 →
  train_speed_kmh = 36 →
  time_to_pass = 50 →
  (train_speed_kmh * 1000 / 3600) * time_to_pass - train_length = 140 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l4168_416833


namespace NUMINAMATH_CALUDE_part_one_part_two_l4168_416826

-- Part 1
theorem part_one (a b : ℤ) (h1 : a = 4) (h2 : b = 5) : a - b = -1 := by
  sorry

-- Part 2
theorem part_two (a b m n s : ℝ) 
  (h1 : a + b = 0) 
  (h2 : m * n = 1) 
  (h3 : |s| = 3) : 
  a + b + m * n + s = 4 ∨ a + b + m * n + s = -2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4168_416826


namespace NUMINAMATH_CALUDE_concrete_slab_height_l4168_416870

/-- Proves that the height of each concrete slab is 0.5 feet given the specified conditions --/
theorem concrete_slab_height :
  let num_homes : ℕ := 3
  let slab_length : ℝ := 100
  let slab_width : ℝ := 100
  let concrete_density : ℝ := 150
  let concrete_cost_per_pound : ℝ := 0.02
  let total_foundation_cost : ℝ := 45000

  let total_weight : ℝ := total_foundation_cost / concrete_cost_per_pound
  let total_volume : ℝ := total_weight / concrete_density
  let volume_per_home : ℝ := total_volume / num_homes
  let slab_area : ℝ := slab_length * slab_width
  let slab_height : ℝ := volume_per_home / slab_area

  slab_height = 0.5 := by sorry

end NUMINAMATH_CALUDE_concrete_slab_height_l4168_416870


namespace NUMINAMATH_CALUDE_jefferson_carriage_cost_l4168_416840

/-- Represents the carriage rental cost calculation --/
def carriageRentalCost (
  totalDistance : ℝ)
  (stopDistances : List ℝ)
  (speeds : List ℝ)
  (baseRate : ℝ)
  (flatFee : ℝ)
  (additionalChargeThreshold : ℝ)
  (additionalChargeRate : ℝ)
  (discountRate : ℝ) : ℝ :=
  sorry

/-- Theorem stating the correct total cost for Jefferson's carriage rental --/
theorem jefferson_carriage_cost :
  carriageRentalCost
    20                     -- total distance to church
    [4, 6, 3]              -- distances to each stop
    [8, 12, 10, 15]        -- speeds for each leg
    35                     -- base rate per hour
    20                     -- flat fee
    10                     -- additional charge speed threshold
    5                      -- additional charge rate per mile
    0.1                    -- discount rate
  = 132.15 := by sorry

end NUMINAMATH_CALUDE_jefferson_carriage_cost_l4168_416840


namespace NUMINAMATH_CALUDE_books_redistribution_l4168_416887

theorem books_redistribution (mark_initial : ℕ) (alice_initial : ℕ) (books_given : ℕ) : 
  mark_initial = 105 →
  alice_initial = 15 →
  books_given = 15 →
  mark_initial - books_given = 3 * (alice_initial + books_given) :=
by
  sorry

end NUMINAMATH_CALUDE_books_redistribution_l4168_416887


namespace NUMINAMATH_CALUDE_cookie_and_game_cost_l4168_416891

/-- Represents the cost and profit information for an item --/
structure ItemInfo where
  cost : ℚ
  price : ℚ
  profit : ℚ
  makeTime : ℚ

/-- Represents the sales quota for each item --/
structure SalesQuota where
  bracelets : ℕ
  necklaces : ℕ
  rings : ℕ

def bracelet : ItemInfo := ⟨1, 1.5, 0.5, 10/60⟩
def necklace : ItemInfo := ⟨2, 3, 1, 15/60⟩
def ring : ItemInfo := ⟨0.5, 1, 0.5, 5/60⟩

def salesQuota : SalesQuota := ⟨5, 3, 10⟩

def profitMargin : ℚ := 0.5
def workingHoursPerDay : ℚ := 2
def daysInWeek : ℕ := 7
def remainingMoney : ℚ := 5

theorem cookie_and_game_cost (totalSales totalCost : ℚ) :
  totalSales = (bracelet.price * salesQuota.bracelets + 
                necklace.price * salesQuota.necklaces + 
                ring.price * salesQuota.rings) →
  totalCost = (bracelet.cost * salesQuota.bracelets + 
               necklace.cost * salesQuota.necklaces + 
               ring.cost * salesQuota.rings) →
  totalSales = totalCost * (1 + profitMargin) →
  (bracelet.makeTime * salesQuota.bracelets + 
   necklace.makeTime * salesQuota.necklaces + 
   ring.makeTime * salesQuota.rings) ≤ workingHoursPerDay * daysInWeek →
  totalSales - remainingMoney = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookie_and_game_cost_l4168_416891


namespace NUMINAMATH_CALUDE_prob_through_C_value_l4168_416894

/-- Represents a grid of city blocks -/
structure CityGrid where
  width : ℕ
  height : ℕ

/-- Represents a position on the grid -/
structure Position where
  x : ℕ
  y : ℕ

/-- Probability of moving east at an intersection -/
def prob_east : ℚ := 2/3

/-- Probability of moving south at an intersection -/
def prob_south : ℚ := 1/3

/-- The starting position A -/
def start_pos : Position := ⟨0, 0⟩

/-- The ending position D -/
def end_pos : Position := ⟨5, 5⟩

/-- The intermediate position C -/
def mid_pos : Position := ⟨3, 2⟩

/-- Calculate the probability of reaching position C when moving from A to D -/
def prob_through_C (grid : CityGrid) (A B C : Position) : ℚ := sorry

/-- Theorem stating that the probability of passing through C is 25/63 -/
theorem prob_through_C_value :
  prob_through_C ⟨5, 5⟩ start_pos end_pos mid_pos = 25/63 := by sorry

end NUMINAMATH_CALUDE_prob_through_C_value_l4168_416894


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_3_l4168_416808

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The first line equation: ax + 2y + 3a = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0

/-- The second line equation: 3x + (a - 1)y = a - 7 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := 3 * x + (a - 1) * y = a - 7

/-- The theorem stating that if the two lines are parallel, then a = 3 -/
theorem parallel_lines_imply_a_eq_3 :
  ∀ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_3_l4168_416808


namespace NUMINAMATH_CALUDE_sphere_equal_volume_surface_area_l4168_416855

theorem sphere_equal_volume_surface_area (r k S : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = S ∧ 
  4 * Real.pi * r^2 = S ∧ 
  k * r = S → 
  r = 3 ∧ k = 12 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sphere_equal_volume_surface_area_l4168_416855


namespace NUMINAMATH_CALUDE_no_finite_vector_set_with_equal_sums_property_l4168_416805

theorem no_finite_vector_set_with_equal_sums_property (n : ℕ) :
  ¬ ∃ (S : Finset (ℝ × ℝ)),
    (S.card = n) ∧
    (∀ (a b : ℝ × ℝ), a ∈ S → b ∈ S → a ≠ b →
      ∃ (c d : ℝ × ℝ), c ∈ S ∧ d ∈ S ∧ c ≠ d ∧ c ≠ a ∧ c ≠ b ∧ d ≠ a ∧ d ≠ b ∧
        a.1 + b.1 = c.1 + d.1 ∧ a.2 + b.2 = c.2 + d.2) :=
by sorry

end NUMINAMATH_CALUDE_no_finite_vector_set_with_equal_sums_property_l4168_416805


namespace NUMINAMATH_CALUDE_rainfall_ratio_is_two_l4168_416877

-- Define the parameters
def total_rainfall : ℝ := 180
def first_half_daily_rainfall : ℝ := 4
def days_in_november : ℕ := 30
def first_half_days : ℕ := 15

-- Define the theorem
theorem rainfall_ratio_is_two :
  let first_half_total := first_half_daily_rainfall * first_half_days
  let second_half_total := total_rainfall - first_half_total
  let second_half_days := days_in_november - first_half_days
  let second_half_daily_rainfall := second_half_total / second_half_days
  (second_half_daily_rainfall / first_half_daily_rainfall) = 2 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_ratio_is_two_l4168_416877


namespace NUMINAMATH_CALUDE_ratio_u_to_x_l4168_416812

theorem ratio_u_to_x (u v x y : ℚ) 
  (h1 : u / v = 5 / 2)
  (h2 : x / y = 4 / 1)
  (h3 : v / y = 3 / 4) :
  u / x = 15 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ratio_u_to_x_l4168_416812
