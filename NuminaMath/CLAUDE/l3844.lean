import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequalities_l3844_384480

theorem arithmetic_geometric_mean_inequalities
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequalities_l3844_384480


namespace NUMINAMATH_CALUDE_average_stamps_is_25_l3844_384466

/-- Calculates the average number of stamps collected per day -/
def average_stamps_collected (days : ℕ) (initial_stamps : ℕ) (daily_increase : ℕ) : ℚ :=
  let total_stamps := (days : ℚ) / 2 * (2 * initial_stamps + (days - 1) * daily_increase)
  total_stamps / days

/-- Proves that the average number of stamps collected per day is 25 -/
theorem average_stamps_is_25 :
  average_stamps_collected 6 10 6 = 25 := by
sorry

end NUMINAMATH_CALUDE_average_stamps_is_25_l3844_384466


namespace NUMINAMATH_CALUDE_object_crosses_x_axis_l3844_384475

/-- The position vector of an object moving in two dimensions -/
def position_vector (t : ℝ) : ℝ × ℝ :=
  (4 * t^2 - 9, 2 * t - 5)

/-- The time when the object crosses the x-axis -/
def crossing_time : ℝ := 2.5

/-- Theorem: The object crosses the x-axis at t = 2.5 seconds -/
theorem object_crosses_x_axis :
  (position_vector crossing_time).2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_object_crosses_x_axis_l3844_384475


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3844_384417

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_1 + 3a_8 + a_15 = 120, 
    prove that 2a_6 - a_4 = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
    2 * a 6 - a 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3844_384417


namespace NUMINAMATH_CALUDE_candy_distribution_proof_l3844_384490

def distribute_candies (n : ℕ) (k : ℕ) (min_counts : List ℕ) : ℕ :=
  sorry

theorem candy_distribution_proof :
  distribute_candies 10 4 [1, 1, 1, 0] = 3176 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_proof_l3844_384490


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3844_384441

theorem diophantine_equation_solution :
  ∀ (a b c : ℤ), 5 * a^2 + 9 * b^2 = 13 * c^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3844_384441


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l3844_384427

theorem quadratic_polynomial_property (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  let p := fun x => (a^2 + a*b + b^2 + a*c + b*c + c^2) * x^2 - 
                    (a + b) * (b + c) * (a + c) * x + 
                    a * b * c * (a + b + c)
  p a = a^4 ∧ p b = b^4 ∧ p c = c^4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l3844_384427


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l3844_384449

/-- The diameter of the inscribed circle in a triangle with sides 11, 6, and 7 is √10 -/
theorem inscribed_circle_diameter (a b c : ℝ) (h1 : a = 11) (h2 : b = 6) (h3 : c = 7) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  2 * area / s = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l3844_384449


namespace NUMINAMATH_CALUDE_stone_length_calculation_l3844_384447

/-- Calculates the length of stones used to pave a hall -/
theorem stone_length_calculation (hall_length hall_width : ℕ) 
  (stone_width num_stones : ℕ) (stone_length : ℚ) : 
  hall_length = 36 ∧ 
  hall_width = 15 ∧ 
  stone_width = 5 ∧ 
  num_stones = 5400 ∧
  (hall_length * 10 * hall_width * 10 : ℚ) = stone_length * stone_width * num_stones →
  stone_length = 2 := by
sorry

end NUMINAMATH_CALUDE_stone_length_calculation_l3844_384447


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3844_384437

theorem smallest_x_absolute_value_equation : 
  ∃ x : ℝ, (∀ y : ℝ, |4*y + 9| = 37 → x ≤ y) ∧ |4*x + 9| = 37 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3844_384437


namespace NUMINAMATH_CALUDE_vector_expression_equality_l3844_384411

theorem vector_expression_equality : 
  let v1 : Fin 2 → ℝ := ![3, -4]
  let v2 : Fin 2 → ℝ := ![2, -3]
  let v3 : Fin 2 → ℝ := ![1, 6]
  v1 + 5 • v2 - v3 = ![12, -25] := by sorry

end NUMINAMATH_CALUDE_vector_expression_equality_l3844_384411


namespace NUMINAMATH_CALUDE_brunchCombinationsCount_l3844_384450

/-- The number of ways to choose one item from a set of 3, two different items from a set of 4, 
    and one item from another set of 3, where the order of selection doesn't matter. -/
def brunchCombinations : ℕ :=
  3 * (Nat.choose 4 2) * 3

/-- Theorem stating that the number of brunch combinations is 54. -/
theorem brunchCombinationsCount : brunchCombinations = 54 := by
  sorry

end NUMINAMATH_CALUDE_brunchCombinationsCount_l3844_384450


namespace NUMINAMATH_CALUDE_vector_inequality_l3844_384470

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors
variable (A B C D : V)

-- Define the theorem
theorem vector_inequality (h : C - B = -(B - C)) :
  C - B + (A - D) - (B - C) ≠ A - D :=
sorry

end NUMINAMATH_CALUDE_vector_inequality_l3844_384470


namespace NUMINAMATH_CALUDE_f_integer_iff_l3844_384464

def f (x : ℝ) : ℝ := (1 + x) ^ (1/3) + (3 - x) ^ (1/3)

theorem f_integer_iff (x : ℝ) : 
  ∃ (n : ℤ), f x = n ↔ 
  (x = 1 + Real.sqrt 5 ∨ 
   x = 1 - Real.sqrt 5 ∨ 
   x = 1 + (10/9) * Real.sqrt 3 ∨ 
   x = 1 - (10/9) * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_f_integer_iff_l3844_384464


namespace NUMINAMATH_CALUDE_cleaning_times_l3844_384444

/-- Proves the cleaning times for Bob and Carol given Alice's cleaning time -/
theorem cleaning_times (alice_time : ℕ) (bob_time carol_time : ℕ) : 
  alice_time = 40 →
  bob_time = alice_time / 4 →
  carol_time = 2 * bob_time →
  (bob_time = 10 ∧ carol_time = 20) := by
  sorry

end NUMINAMATH_CALUDE_cleaning_times_l3844_384444


namespace NUMINAMATH_CALUDE_root_difference_product_l3844_384401

theorem root_difference_product (p q a b c : ℝ) : 
  (a^2 + p*a + 1 = 0) → 
  (b^2 + p*b + 1 = 0) → 
  (b^2 + q*b + 2 = 0) → 
  (c^2 + q*c + 2 = 0) → 
  (b - a) * (b - c) = p*q - 6 := by
sorry

end NUMINAMATH_CALUDE_root_difference_product_l3844_384401


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_n_l3844_384419

theorem binomial_coefficient_n_n (n : ℕ) : Nat.choose n n = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_n_l3844_384419


namespace NUMINAMATH_CALUDE_base_conversion_property_l3844_384440

def convert_base (n : ℕ) (from_base to_base : ℕ) : ℕ :=
  sorry

def digits_to_nat (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

def nat_to_digits (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem base_conversion_property :
  ∀ b : ℕ, b ∈ [13, 12, 11] →
    let n := digits_to_nat [1, 2, 2, 1] b
    nat_to_digits (convert_base n b (b - 1)) (b - 1) = [1, 2, 2, 1] ∧
  let n₁₀ := digits_to_nat [1, 2, 2, 1] 10
  nat_to_digits (convert_base n₁₀ 10 9) 9 ≠ [1, 2, 2, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_base_conversion_property_l3844_384440


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l3844_384459

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 32)
  (h2 : failed_english = 56)
  (h3 : failed_both = 12) :
  100 - (failed_hindi + failed_english - failed_both) = 24 := by
sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l3844_384459


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l3844_384458

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l3844_384458


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3844_384404

/-- 
Given an angle α with vertex at the origin, initial side on the positive x-axis,
and terminal side on the ray 3x + 4y = 0 with x > 0, prove that sin α = -3/5.
-/
theorem sin_alpha_value (α : Real) : 
  (∃ (x y : Real), x > 0 ∧ 3 * x + 4 * y = 0 ∧ 
   x = Real.cos α ∧ y = Real.sin α) → 
  Real.sin α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3844_384404


namespace NUMINAMATH_CALUDE_john_chore_time_l3844_384493

/-- Given a ratio of cartoon watching time to chore time and the total cartoon watching time,
    calculate the required chore time. -/
def chore_time (cartoon_ratio : ℕ) (chore_ratio : ℕ) (total_cartoon_time : ℕ) : ℕ :=
  (chore_ratio * total_cartoon_time) / cartoon_ratio

theorem john_chore_time :
  let cartoon_ratio : ℕ := 10
  let chore_ratio : ℕ := 8
  let total_cartoon_time : ℕ := 120
  chore_time cartoon_ratio chore_ratio total_cartoon_time = 96 := by
  sorry

#eval chore_time 10 8 120

end NUMINAMATH_CALUDE_john_chore_time_l3844_384493


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3844_384445

open Complex

theorem min_distance_to_line (z : ℂ) (h : abs (z - 1) = abs (z + 2*I)) :
  ∃ (min_val : ℝ), min_val = (9 * Real.sqrt 5) / 10 ∧
  ∀ (w : ℂ), abs (w - 1) = abs (w + 2*I) → abs (w - 1 - I) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3844_384445


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l3844_384424

theorem sin_cos_fourth_power_sum (θ : Real) (h : Real.sin (2 * θ) = 1 / 2) :
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l3844_384424


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3844_384423

theorem polynomial_factorization (x y : ℝ) : 
  x^3 - 4*x^2*y + 4*x*y^2 = x*(x - 2*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3844_384423


namespace NUMINAMATH_CALUDE_biology_score_calculation_l3844_384460

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 67
def average_score : ℕ := 75
def total_subjects : ℕ := 5

theorem biology_score_calculation :
  let known_scores_sum := math_score + science_score + social_studies_score + english_score
  let total_score := average_score * total_subjects
  total_score - known_scores_sum = 85 := by
  sorry

#check biology_score_calculation

end NUMINAMATH_CALUDE_biology_score_calculation_l3844_384460


namespace NUMINAMATH_CALUDE_jesse_room_area_l3844_384431

/-- The length of Jesse's room in feet -/
def room_length : ℝ := 12

/-- The width of Jesse's room in feet -/
def room_width : ℝ := 8

/-- The area of Jesse's room floor in square feet -/
def room_area : ℝ := room_length * room_width

theorem jesse_room_area : room_area = 96 := by
  sorry

end NUMINAMATH_CALUDE_jesse_room_area_l3844_384431


namespace NUMINAMATH_CALUDE_expected_cost_is_3500_l3844_384408

/-- The number of machines -/
def total_machines : ℕ := 5

/-- The number of faulty machines -/
def faulty_machines : ℕ := 2

/-- The cost of testing one machine in yuan -/
def cost_per_test : ℕ := 1000

/-- The possible outcomes of the number of tests needed -/
def possible_tests : List ℕ := [2, 3, 4]

/-- The probabilities corresponding to each outcome -/
def probabilities : List ℚ := [1/10, 3/10, 3/5]

/-- The expected cost of testing in yuan -/
def expected_cost : ℚ := 3500

/-- Theorem stating that the expected cost of testing is 3500 yuan -/
theorem expected_cost_is_3500 :
  (List.sum (List.zipWith (· * ·) (List.map (λ n => n * cost_per_test) possible_tests) probabilities) : ℚ) = expected_cost :=
sorry

end NUMINAMATH_CALUDE_expected_cost_is_3500_l3844_384408


namespace NUMINAMATH_CALUDE_relationship_correctness_l3844_384456

theorem relationship_correctness :
  (∃ a b c : ℝ, (a > b ↔ a * c^2 > b * c^2) → False) ∧
  (∃ a b : ℝ, (a > b → 1/a < 1/b) → False) ∧
  (∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c > d → a/d > b/c) ∧
  (∃ a b c : ℝ, (a > b ∧ b > 0 → a^c < b^c) → False) :=
by sorry

end NUMINAMATH_CALUDE_relationship_correctness_l3844_384456


namespace NUMINAMATH_CALUDE_frank_can_collection_l3844_384454

/-- Represents the number of cans in each bag for a given day -/
def BagContents := List Nat

/-- Calculates the total number of cans from a list of bag contents -/
def totalCans (bags : BagContents) : Nat :=
  bags.sum

theorem frank_can_collection :
  let saturday : BagContents := [4, 6, 5, 7, 8]
  let sunday : BagContents := [6, 5, 9]
  let monday : BagContents := [8, 8]
  totalCans saturday + totalCans sunday + totalCans monday = 66 := by
  sorry

end NUMINAMATH_CALUDE_frank_can_collection_l3844_384454


namespace NUMINAMATH_CALUDE_problem_solution_l3844_384415

theorem problem_solution : 
  ((-3 : ℝ)^0 + (1/3)^2 + (-2)^3 = -62/9) ∧ 
  (∀ x : ℝ, (x + 1)^2 - (1 - 2*x)*(1 + 2*x) = 5*x^2 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3844_384415


namespace NUMINAMATH_CALUDE_dice_events_properties_l3844_384479

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A, B, C, and D
def A : Set Ω := {ω | ω.1 = 2}
def B : Set Ω := {ω | ω.2 < 5}
def C : Set Ω := {ω | (ω.1.val + ω.2.val) % 2 = 1}
def D : Set Ω := {ω | ω.1.val + ω.2.val = 9}

-- State the theorem
theorem dice_events_properties :
  (¬(A ∩ B = ∅) ∧ P (A ∩ B) = P A * P B) ∧
  (A ∩ D = ∅ ∧ P (A ∩ D) ≠ P A * P D) ∧
  (¬(A ∩ C = ∅) ∧ P (A ∩ C) = P A * P C) :=
sorry

end NUMINAMATH_CALUDE_dice_events_properties_l3844_384479


namespace NUMINAMATH_CALUDE_larger_number_problem_l3844_384436

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 1375)
  (h2 : L = 6 * S + 15) : 
  L = 1647 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3844_384436


namespace NUMINAMATH_CALUDE_complex_sum_real_l3844_384452

theorem complex_sum_real (a : ℝ) : 
  (a / (1 + 2*I) + (1 + 2*I) / 5 : ℂ).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_real_l3844_384452


namespace NUMINAMATH_CALUDE_base_side_from_sphere_volume_l3844_384400

/-- Regular triangular prism with inscribed sphere -/
structure RegularTriangularPrism :=
  (base_side : ℝ)
  (height : ℝ)
  (sphere_volume : ℝ)

/-- The theorem stating the relationship between the inscribed sphere volume
    and the base side length of a regular triangular prism -/
theorem base_side_from_sphere_volume
  (prism : RegularTriangularPrism)
  (h_positive : prism.base_side > 0)
  (h_sphere_volume : prism.sphere_volume = 36 * Real.pi)
  (h_height_eq_diameter : prism.height = 2 * (prism.base_side * Real.sqrt 3 / 6)) :
  prism.base_side = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_base_side_from_sphere_volume_l3844_384400


namespace NUMINAMATH_CALUDE_coffee_cups_total_l3844_384418

theorem coffee_cups_total (sandra_cups marcie_cups : ℕ) 
  (h1 : sandra_cups = 6) 
  (h2 : marcie_cups = 2) : 
  sandra_cups + marcie_cups = 8 := by
sorry

end NUMINAMATH_CALUDE_coffee_cups_total_l3844_384418


namespace NUMINAMATH_CALUDE_product_divisibility_l3844_384402

def die_numbers : Finset ℕ := Finset.range 8

theorem product_divisibility (visible : Finset ℕ) 
  (h1 : visible ⊆ die_numbers) 
  (h2 : visible.card = 6) : 
  96 ∣ visible.prod id :=
sorry

end NUMINAMATH_CALUDE_product_divisibility_l3844_384402


namespace NUMINAMATH_CALUDE_percentage_of_rejected_meters_l3844_384438

def total_meters : ℕ := 100
def rejected_meters : ℕ := 10

theorem percentage_of_rejected_meters :
  (rejected_meters : ℚ) / (total_meters : ℚ) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_rejected_meters_l3844_384438


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3844_384492

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- A point on the right branch of a hyperbola -/
structure RightBranchPoint (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2/a^2 - y^2/b^2 = 1
  right_branch : x ≥ a

/-- Distance between two points -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := sorry

/-- Left focus of the hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Right focus of the hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The ratio |PF₁|²/|PF₂| for a point P on the hyperbola -/
def focal_ratio (h : Hyperbola a b) (p : RightBranchPoint h) : ℝ := sorry

/-- The minimum value of focal_ratio over all points on the right branch -/
def min_focal_ratio (h : Hyperbola a b) : ℝ := sorry

theorem hyperbola_eccentricity_range (a b : ℝ) (h : Hyperbola a b) :
  min_focal_ratio h = 8 * a → 1 < eccentricity h ∧ eccentricity h ≤ 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3844_384492


namespace NUMINAMATH_CALUDE_choose_three_roles_from_eight_people_l3844_384432

def number_of_people : ℕ := 8
def number_of_roles : ℕ := 3

theorem choose_three_roles_from_eight_people : 
  (number_of_people * (number_of_people - 1) * (number_of_people - 2) = 336) := by
  sorry

end NUMINAMATH_CALUDE_choose_three_roles_from_eight_people_l3844_384432


namespace NUMINAMATH_CALUDE_sum_integers_from_neg50_to_75_l3844_384433

def sum_integers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_from_neg50_to_75 :
  sum_integers (-50) 75 = 1575 := by
  sorry

end NUMINAMATH_CALUDE_sum_integers_from_neg50_to_75_l3844_384433


namespace NUMINAMATH_CALUDE_units_digit_of_50_factorial_l3844_384407

theorem units_digit_of_50_factorial (n : ℕ) : n = 50 → (n.factorial % 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_50_factorial_l3844_384407


namespace NUMINAMATH_CALUDE_regular_tetrahedron_has_four_faces_l3844_384443

/-- A regular tetrahedron is a three-dimensional shape with four congruent equilateral triangular faces. -/
structure RegularTetrahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of faces in a regular tetrahedron -/
def num_faces (t : RegularTetrahedron) : ℕ := 4

/-- Theorem: A regular tetrahedron has 4 faces -/
theorem regular_tetrahedron_has_four_faces (t : RegularTetrahedron) : num_faces t = 4 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_has_four_faces_l3844_384443


namespace NUMINAMATH_CALUDE_remainder_91_92_mod_100_l3844_384403

theorem remainder_91_92_mod_100 : 91^92 % 100 = 81 := by
  sorry

end NUMINAMATH_CALUDE_remainder_91_92_mod_100_l3844_384403


namespace NUMINAMATH_CALUDE_library_visitors_proof_l3844_384461

/-- The total number of visitors to a library in a week -/
def total_visitors (monday : ℕ) (tuesday_multiplier : ℕ) (remaining_days : ℕ) (avg_remaining : ℕ) : ℕ :=
  monday + (tuesday_multiplier * monday) + (remaining_days * avg_remaining)

/-- Theorem stating that the total number of visitors to the library in a week is 250 -/
theorem library_visitors_proof : 
  total_visitors 50 2 5 20 = 250 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_proof_l3844_384461


namespace NUMINAMATH_CALUDE_max_value_theorem_l3844_384494

/-- Given a quadratic function y = ax² + x - b where a > 0 and b > 1,
    if the solution set P of y > 0 intersects with Q = {x | -2-t < x < -2+t}
    for all positive t, then the maximum value of 1/a - 1/b is 1/2. -/
theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ t > 0, ∃ x, (a * x^2 + x - b > 0 ∧ -2 - t < x ∧ x < -2 + t)) →
  (∃ m, m = 1/a - 1/b ∧ ∀ a' b', a' > 0 → b' > 1 →
    (∀ t > 0, ∃ x, (a' * x^2 + x - b' > 0 ∧ -2 - t < x ∧ x < -2 + t)) →
    1/a' - 1/b' ≤ m) ∧
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3844_384494


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l3844_384446

theorem largest_prime_factor_of_12321 : ∃ p : ℕ, p.Prime ∧ p ∣ 12321 ∧ ∀ q : ℕ, q.Prime → q ∣ 12321 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l3844_384446


namespace NUMINAMATH_CALUDE_smaller_number_proof_l3844_384420

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 15) (h2 : 3 * x = 5 * y - 11) : 
  min x y = 8 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l3844_384420


namespace NUMINAMATH_CALUDE_exponent_sum_l3844_384467

theorem exponent_sum (x m n : ℝ) (hm : x^m = 6) (hn : x^n = 2) : x^(m+n) = 12 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_l3844_384467


namespace NUMINAMATH_CALUDE_snyder_income_proof_l3844_384439

/-- Mrs. Snyder's previous monthly income -/
def previous_income : ℝ := 1700

/-- Mrs. Snyder's salary increase -/
def salary_increase : ℝ := 850

/-- Percentage of income spent on rent and utilities before salary increase -/
def previous_percentage : ℝ := 0.45

/-- Percentage of income spent on rent and utilities after salary increase -/
def new_percentage : ℝ := 0.30

theorem snyder_income_proof :
  (previous_percentage * previous_income = new_percentage * (previous_income + salary_increase)) ∧
  previous_income = 1700 := by
  sorry

end NUMINAMATH_CALUDE_snyder_income_proof_l3844_384439


namespace NUMINAMATH_CALUDE_unique_fraction_l3844_384414

theorem unique_fraction : ∃! (m n : ℕ), 
  m < 10 ∧ n < 10 ∧ 
  n = m^2 - 1 ∧
  (m + 2 : ℚ) / (n + 2) > 1/3 ∧
  (m - 3 : ℚ) / (n - 3) < 1/10 ∧
  m = 3 ∧ n = 8 := by
sorry

end NUMINAMATH_CALUDE_unique_fraction_l3844_384414


namespace NUMINAMATH_CALUDE_solution_difference_l3844_384469

theorem solution_difference (p q : ℝ) : 
  (p - 2) * (p + 4) = 26 * p - 100 →
  (q - 2) * (q + 4) = 26 * q - 100 →
  p ≠ q →
  p > q →
  p - q = 4 * Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l3844_384469


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3844_384434

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_prod : a 3 * a 11 = 16) :
  a 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3844_384434


namespace NUMINAMATH_CALUDE_ben_egg_count_l3844_384448

/-- Given that Ben has 7 trays of eggs and each tray contains 10 eggs,
    prove that the total number of eggs Ben examined is 70. -/
theorem ben_egg_count (num_trays : ℕ) (eggs_per_tray : ℕ) :
  num_trays = 7 → eggs_per_tray = 10 → num_trays * eggs_per_tray = 70 := by
  sorry

end NUMINAMATH_CALUDE_ben_egg_count_l3844_384448


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3844_384478

theorem solution_set_inequality (x : ℝ) :
  (x + 5) * (1 - x) ≥ 8 ↔ -3 ≤ x ∧ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3844_384478


namespace NUMINAMATH_CALUDE_age_difference_l3844_384412

/-- Given the ages of Taehyung, his father, and his mother, prove that the age difference
between the father and mother is equal to Taehyung's age. -/
theorem age_difference (taehyung_age : ℕ) (father_age : ℕ) (mother_age : ℕ)
  (h1 : taehyung_age = 9)
  (h2 : father_age = 5 * taehyung_age)
  (h3 : mother_age = 4 * taehyung_age) :
  father_age - mother_age = taehyung_age :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l3844_384412


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3844_384487

/-- The hyperbola and parabola intersection problem -/
theorem hyperbola_parabola_intersection
  (a : ℝ) (P F₁ F₂ : ℝ × ℝ) 
  (h_a_pos : a > 0)
  (h_hyperbola : 3 * P.1^2 - P.2^2 = 3 * a^2)
  (h_parabola : P.2^2 = 8 * a * P.1)
  (h_F₁ : F₁ = (-2*a, 0))
  (h_F₂ : F₂ = (2*a, 0))
  (h_distance : Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
                Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 12) :
  ∃ (x : ℝ), x = -2 ∧ x = -a/2 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3844_384487


namespace NUMINAMATH_CALUDE_positive_X_value_l3844_384481

-- Define the * operation
def star (X Y : ℝ) : ℝ := X^3 + Y^2

-- Theorem statement
theorem positive_X_value :
  ∃ X : ℝ, X > 0 ∧ star X 4 = 280 ∧ X = 6 :=
sorry

end NUMINAMATH_CALUDE_positive_X_value_l3844_384481


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3844_384430

theorem age_ratio_proof (my_current_age brother_current_age : ℕ) 
  (h1 : my_current_age = 20)
  (h2 : my_current_age + 10 + (brother_current_age + 10) = 45) :
  (my_current_age + 10) / (brother_current_age + 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3844_384430


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2011_l3844_384485

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_2011 :
  ∃ n : ℕ, arithmeticSequence 1 3 n = 2011 ∧ n = 671 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2011_l3844_384485


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l3844_384474

/-- The interval for systematic sampling -/
def systematic_sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The systematic sampling interval for a population of 1200 and sample size of 30 is 40 -/
theorem systematic_sampling_interval_example :
  systematic_sampling_interval 1200 30 = 40 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l3844_384474


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l3844_384497

theorem no_solution_to_inequalities :
  ¬∃ x : ℝ, (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 5) := by
sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l3844_384497


namespace NUMINAMATH_CALUDE_recipe_fat_calculation_l3844_384468

/-- Calculates the grams of fat per serving in a recipe -/
def fat_per_serving (servings : ℕ) (cream_cups : ℚ) (fat_per_cup : ℕ) : ℚ :=
  (cream_cups * fat_per_cup) / servings

theorem recipe_fat_calculation :
  fat_per_serving 4 (1/2) 88 = 11 := by
  sorry

end NUMINAMATH_CALUDE_recipe_fat_calculation_l3844_384468


namespace NUMINAMATH_CALUDE_angle_trig_values_l3844_384496

/-- Given an angle α whose terminal side passes through the point (3,4),
    prove the values of sin α, cos α, and tan α. -/
theorem angle_trig_values (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = 4) →
  Real.sin α = 4/5 ∧ Real.cos α = 3/5 ∧ Real.tan α = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_trig_values_l3844_384496


namespace NUMINAMATH_CALUDE_translation_of_line_segment_l3844_384488

/-- Given a line segment AB with endpoints A(1,0) and B(3,2), if it is translated to a new position
    where the new endpoints are A₁(a,1) and B₁(4,b), then a = 2 and b = 3. -/
theorem translation_of_line_segment (a b : ℝ) : 
  (∃ (dx dy : ℝ), (1 + dx = a ∧ 0 + dy = 1) ∧ (3 + dx = 4 ∧ 2 + dy = b)) → 
  (a = 2 ∧ b = 3) :=
by sorry

end NUMINAMATH_CALUDE_translation_of_line_segment_l3844_384488


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3844_384416

theorem quadratic_equation_solution :
  ∃ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3844_384416


namespace NUMINAMATH_CALUDE_range_of_a_l3844_384471

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  (((x + 2) / 3 - x / 2) > 1) ∧ (2 * (x - a) ≤ 0)

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < -2

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x, inequality_system x a ↔ solution_set x) →
  a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3844_384471


namespace NUMINAMATH_CALUDE_parallelogram_height_calculation_l3844_384457

/-- Given a parallelogram-shaped field with specified dimensions and costs, 
    calculate the perpendicular distance from the other side. -/
theorem parallelogram_height_calculation 
  (base : ℝ)
  (cost_per_10sqm : ℝ)
  (total_cost : ℝ)
  (h : base = 54)
  (i : cost_per_10sqm = 50)
  (j : total_cost = 6480) :
  (total_cost / cost_per_10sqm * 10) / base = 24 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_height_calculation_l3844_384457


namespace NUMINAMATH_CALUDE_square_sum_of_squares_l3844_384499

theorem square_sum_of_squares (a b : ℝ) : ∃ c d : ℝ, (a^2 + b^2)^2 = c^2 + d^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_squares_l3844_384499


namespace NUMINAMATH_CALUDE_sequence_is_increasing_l3844_384498

def IsIncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem sequence_is_increasing (a : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) - a n - 3 = 0) : 
  IsIncreasingSequence a :=
sorry

end NUMINAMATH_CALUDE_sequence_is_increasing_l3844_384498


namespace NUMINAMATH_CALUDE_arithmetic_sign_change_geometric_sign_alternation_l3844_384482

-- Define an arithmetic progression
def arithmetic_progression (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Define a geometric progression
def geometric_progression (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

-- Theorem for arithmetic progression sign change
theorem arithmetic_sign_change (a₁ : ℝ) (d : ℝ) :
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → (arithmetic_progression a₁ d n > 0) ∧
                     ∀ m : ℕ, m > k → (arithmetic_progression a₁ d m < 0) :=
sorry

-- Theorem for geometric progression sign alternation
theorem geometric_sign_alternation (a₁ : ℝ) (r : ℝ) (h : r < 0) :
  ∀ n : ℕ, (geometric_progression a₁ r (2*n) > 0) ∧ 
           (geometric_progression a₁ r (2*n + 1) < 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sign_change_geometric_sign_alternation_l3844_384482


namespace NUMINAMATH_CALUDE_tank_water_calculation_l3844_384455

theorem tank_water_calculation : 
  let tank1_capacity : ℚ := 7000
  let tank2_capacity : ℚ := 5000
  let tank3_capacity : ℚ := 3000
  let tank1_fill_ratio : ℚ := 3/4
  let tank2_fill_ratio : ℚ := 4/5
  let tank3_fill_ratio : ℚ := 1/2
  let total_water : ℚ := tank1_capacity * tank1_fill_ratio + 
                         tank2_capacity * tank2_fill_ratio + 
                         tank3_capacity * tank3_fill_ratio
  total_water = 10750 := by
sorry

end NUMINAMATH_CALUDE_tank_water_calculation_l3844_384455


namespace NUMINAMATH_CALUDE_eric_green_marbles_l3844_384428

/-- Represents the number of marbles of each color and the total number of marbles. -/
structure MarbleCount where
  total : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- Theorem stating that Eric has 2 green marbles given the conditions. -/
theorem eric_green_marbles (m : MarbleCount)
  (h1 : m.total = 20)
  (h2 : m.white = 12)
  (h3 : m.blue = 6)
  (h4 : m.green = m.total - (m.white + m.blue)) :
  m.green = 2 := by
  sorry

end NUMINAMATH_CALUDE_eric_green_marbles_l3844_384428


namespace NUMINAMATH_CALUDE_min_days_to_triple_debt_l3844_384426

/-- The borrowed amount in dollars -/
def borrowed_amount : ℝ := 15

/-- The daily interest rate as a decimal -/
def daily_interest_rate : ℝ := 0.1

/-- Calculate the amount owed after a given number of days -/
def amount_owed (days : ℝ) : ℝ :=
  borrowed_amount * (1 + daily_interest_rate * days)

/-- The minimum number of days needed to owe at least triple the borrowed amount -/
def min_days : ℕ := 20

theorem min_days_to_triple_debt :
  (∀ d : ℕ, d < min_days → amount_owed d < 3 * borrowed_amount) ∧
  amount_owed min_days ≥ 3 * borrowed_amount :=
sorry

end NUMINAMATH_CALUDE_min_days_to_triple_debt_l3844_384426


namespace NUMINAMATH_CALUDE_kendra_remaining_words_l3844_384451

/-- Theorem: Given Kendra's goal of learning 60 new words and having already learned 36 words,
    she needs to learn 24 more words to reach her goal. -/
theorem kendra_remaining_words (total_goal : ℕ) (learned : ℕ) (remaining : ℕ) :
  total_goal = 60 →
  learned = 36 →
  remaining = total_goal - learned →
  remaining = 24 :=
by sorry

end NUMINAMATH_CALUDE_kendra_remaining_words_l3844_384451


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l3844_384406

/-- Given a sphere and a right circular cone, where:
    - The radius of the cone's base is twice the radius of the sphere
    - The volume of the cone is one-third the volume of the sphere
    Prove that the ratio of the cone's altitude to its base radius is 1/6 -/
theorem cone_sphere_ratio (r : ℝ) (h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) : 
  (4 / 3 * Real.pi * r^2 * h = 1 / 3 * (4 / 3 * Real.pi * r^3)) → 
  (h / (2 * r) = 1 / 6) := by
sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l3844_384406


namespace NUMINAMATH_CALUDE_cake_sugar_calculation_l3844_384477

/-- The amount of sugar stored in the house (in pounds) -/
def sugar_stored : ℕ := 287

/-- The amount of additional sugar needed (in pounds) -/
def sugar_additional : ℕ := 163

/-- The total amount of sugar needed for the cake (in pounds) -/
def total_sugar_needed : ℕ := sugar_stored + sugar_additional

theorem cake_sugar_calculation :
  total_sugar_needed = 450 :=
by sorry

end NUMINAMATH_CALUDE_cake_sugar_calculation_l3844_384477


namespace NUMINAMATH_CALUDE_cube_root_ratio_l3844_384453

theorem cube_root_ratio (r_old r_new : ℝ) (a_old a_new : ℝ) : 
  a_old = (2 * r_old)^3 → 
  a_new = (2 * r_new)^3 → 
  a_new = 0.125 * a_old → 
  r_new / r_old = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cube_root_ratio_l3844_384453


namespace NUMINAMATH_CALUDE_circle_ratio_l3844_384435

theorem circle_ratio (a b : ℝ) (h : a > 0) (k : b > 0) 
  (h1 : π * b^2 - π * a^2 = 4 * (π * a^2)) : a / b = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l3844_384435


namespace NUMINAMATH_CALUDE_cost_per_metre_l3844_384465

theorem cost_per_metre (total_length : ℝ) (total_cost : ℝ) (h1 : total_length = 9.25) (h2 : total_cost = 434.75) :
  total_cost / total_length = 47 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_metre_l3844_384465


namespace NUMINAMATH_CALUDE_fraction_identity_l3844_384442

theorem fraction_identity (a b c : ℝ) 
  (h1 : a + b + c ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0)
  (h5 : (a + b + c)⁻¹ = a⁻¹ + b⁻¹ + c⁻¹) :
  (a^5 + b^5 + c^5)⁻¹ = a⁻¹^5 + b⁻¹^5 + c⁻¹^5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_identity_l3844_384442


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l3844_384422

theorem sum_of_three_squares (square triangle : ℚ) : 
  (square + triangle + 2 * square + triangle = 34) →
  (triangle + square + triangle + 3 * square = 40) →
  (3 * square = 66 / 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l3844_384422


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l3844_384489

/-- A rhombus with given area and diagonal ratio has a specific longer diagonal length -/
theorem rhombus_diagonal_length 
  (area : ℝ) 
  (diagonal_ratio : ℚ) 
  (h_area : area = 135) 
  (h_ratio : diagonal_ratio = 5 / 3) : 
  ∃ (d1 d2 : ℝ), d1 > d2 ∧ d1 / d2 = diagonal_ratio ∧ d1 * d2 / 2 = area ∧ d1 = 15 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l3844_384489


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3844_384491

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) (h : ∀ x, f (1 + x) + 2 * f (1 - x) = 6 - 1 / x) :
  f (Real.sqrt 2) = 3 + Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 (g : ℝ → ℝ) (m : ℝ)
  (h1 : ∀ x > 0, g x = (m^2 - 2*m - 2) * x^(m^2 + 3*m + 2))
  (h2 : StrictMono g)
  (h3 : ∀ x, g (2*x - 1) ≥ 1) :
  ∀ x, x ≤ 0 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3844_384491


namespace NUMINAMATH_CALUDE_tree_space_calculation_l3844_384413

/-- The space taken up by each tree in square feet -/
def tree_space : ℝ := 1

/-- The total length of the road in feet -/
def road_length : ℝ := 148

/-- The number of trees to be planted -/
def num_trees : ℕ := 8

/-- The space between each tree in feet -/
def space_between : ℝ := 20

theorem tree_space_calculation :
  tree_space * num_trees + space_between * (num_trees - 1) = road_length := by
  sorry

end NUMINAMATH_CALUDE_tree_space_calculation_l3844_384413


namespace NUMINAMATH_CALUDE_number_of_non_officers_l3844_384483

/-- Proves that the number of non-officers is 525 given the salary conditions --/
theorem number_of_non_officers (avg_salary : ℝ) (officer_salary : ℝ) (non_officer_salary : ℝ) 
  (num_officers : ℕ) (h1 : avg_salary = 120) (h2 : officer_salary = 470) 
  (h3 : non_officer_salary = 110) (h4 : num_officers = 15) : 
  ∃ (num_non_officers : ℕ), 
    (↑num_officers * officer_salary + ↑num_non_officers * non_officer_salary) / 
    (↑num_officers + ↑num_non_officers) = avg_salary ∧ num_non_officers = 525 := by
  sorry

#check number_of_non_officers

end NUMINAMATH_CALUDE_number_of_non_officers_l3844_384483


namespace NUMINAMATH_CALUDE_consecutive_even_squares_l3844_384486

theorem consecutive_even_squares (x : ℕ) : 
  (x % 2 = 0) → (x^2 - (x-2)^2 = 2012) → x = 504 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_squares_l3844_384486


namespace NUMINAMATH_CALUDE_inequality_proof_l3844_384410

theorem inequality_proof (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3844_384410


namespace NUMINAMATH_CALUDE_symmetry_of_point_l3844_384484

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to y-axis -/
def symmetricToYAxis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

theorem symmetry_of_point :
  let A : Point := ⟨6, 4⟩
  let B : Point := symmetricToYAxis A
  B = ⟨-6, 4⟩ := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l3844_384484


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3844_384463

/-- The number of coins being flipped -/
def num_coins : ℕ := 6

/-- The number of specific coins we're interested in -/
def num_specific_coins : ℕ := 3

/-- The number of possible outcomes for each coin (heads or tails) -/
def outcomes_per_coin : ℕ := 2

/-- The probability of three specific coins out of six showing the same face -/
def probability_same_face : ℚ := 1 / 4

theorem coin_flip_probability :
  (outcomes_per_coin ^ num_specific_coins * outcomes_per_coin ^ (num_coins - num_specific_coins)) /
  (outcomes_per_coin ^ num_coins) = probability_same_face :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3844_384463


namespace NUMINAMATH_CALUDE_red_light_probability_l3844_384495

theorem red_light_probability (n : ℕ) (p : ℝ) (h1 : n = 4) (h2 : p = 1/3) :
  let q := 1 - p
  (q * q * p : ℝ) = 4/27 :=
by sorry

end NUMINAMATH_CALUDE_red_light_probability_l3844_384495


namespace NUMINAMATH_CALUDE_find_c_l3844_384462

theorem find_c (a b c : ℝ) 
  (eq1 : a + b = 3) 
  (eq2 : a * c + b = 18) 
  (eq3 : b * c + a = 6) : 
  c = 7 := by sorry

end NUMINAMATH_CALUDE_find_c_l3844_384462


namespace NUMINAMATH_CALUDE_rabbit_run_time_l3844_384421

/-- The time taken for a rabbit to run from the end to the front of a moving line and back -/
theorem rabbit_run_time (line_length : ℝ) (line_speed : ℝ) (rabbit_speed : ℝ) : 
  line_length = 40 →
  line_speed = 3 →
  rabbit_speed = 5 →
  (line_length / (rabbit_speed - line_speed)) + (line_length / (rabbit_speed + line_speed)) = 25 :=
by sorry

end NUMINAMATH_CALUDE_rabbit_run_time_l3844_384421


namespace NUMINAMATH_CALUDE_percentage_of_men_in_company_l3844_384476

/-- The percentage of men in a company, given attendance rates at a company picnic -/
theorem percentage_of_men_in_company : 
  ∀ (M : ℝ), 
  (M ≥ 0) →  -- M is non-negative
  (M ≤ 1) →  -- M is at most 1
  (0.20 * M + 0.40 * (1 - M) = 0.33) →  -- Picnic attendance equation
  (M = 0.35) :=  -- Conclusion: 35% of employees are men
by sorry

end NUMINAMATH_CALUDE_percentage_of_men_in_company_l3844_384476


namespace NUMINAMATH_CALUDE_subset_0_2_is_5th_subset_211_is_01467_l3844_384409

/-- The set E with 10 elements -/
def E : Finset ℕ := Finset.range 10

/-- Function to calculate the k value for a given subset -/
def kValue (subset : Finset ℕ) : ℕ :=
  subset.sum (fun i => 2^i)

/-- The first theorem: {0, 2} (representing {a₁, a₃}) corresponds to k = 5 -/
theorem subset_0_2_is_5th : kValue {0, 2} = 5 := by sorry

/-- The second theorem: k = 211 corresponds to the subset {0, 1, 4, 6, 7} 
    (representing {a₁, a₂, a₅, a₇, a₈}) -/
theorem subset_211_is_01467 : 
  (Finset.filter (fun i => (211 / 2^i) % 2 = 1) E) = {0, 1, 4, 6, 7} := by sorry

end NUMINAMATH_CALUDE_subset_0_2_is_5th_subset_211_is_01467_l3844_384409


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_three_lines_l3844_384472

/-- A line in a plane --/
structure Line where
  -- Add necessary fields to represent a line

/-- An equilateral triangle --/
structure EquilateralTriangle where
  -- Add necessary fields to represent an equilateral triangle

/-- A point in a plane --/
structure Point where
  -- Add necessary fields to represent a point

/-- Checks if a point lies on a given line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  sorry

/-- Checks if a triangle is equilateral --/
def isEquilateralTriangle (t : EquilateralTriangle) : Prop :=
  sorry

/-- The main theorem --/
theorem equilateral_triangle_on_three_lines 
  (d₁ d₂ d₃ : Line) : 
  ∃ (t : EquilateralTriangle), 
    isEquilateralTriangle t ∧ 
    (∃ (p₁ p₂ p₃ : Point), 
      pointOnLine p₁ d₁ ∧ 
      pointOnLine p₂ d₂ ∧ 
      pointOnLine p₃ d₃ ∧ 
      -- Add conditions to relate p₁, p₂, p₃ to the vertices of t
      sorry) :=
by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_on_three_lines_l3844_384472


namespace NUMINAMATH_CALUDE_triangle_pqr_properties_l3844_384425

/-- Triangle PQR with vertices P(-2,3), Q(4,5), and R(1,-4), and point S(p,q) inside the triangle such that triangles PQS, QRS, and RPS have equal areas -/
structure TrianglePQR where
  P : ℝ × ℝ := (-2, 3)
  Q : ℝ × ℝ := (4, 5)
  R : ℝ × ℝ := (1, -4)
  S : ℝ × ℝ
  equal_areas : True  -- Placeholder for the equal areas condition

/-- The coordinates of point S -/
def point_S (t : TrianglePQR) : ℝ × ℝ := t.S

/-- The perimeter of triangle PQR -/
noncomputable def perimeter (t : TrianglePQR) : ℝ :=
  Real.sqrt 40 + Real.sqrt 90 + Real.sqrt 58

/-- Main theorem about the triangle PQR and point S -/
theorem triangle_pqr_properties (t : TrianglePQR) :
  point_S t = (1, 4/3) ∧
  10 * (point_S t).1 + (point_S t).2 = 34/3 ∧
  perimeter t = Real.sqrt 40 + Real.sqrt 90 + Real.sqrt 58 ∧
  34/3 < perimeter t :=
by sorry

end NUMINAMATH_CALUDE_triangle_pqr_properties_l3844_384425


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l3844_384473

theorem smallest_two_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 17 ∧ 
  n % 17 = 0 ∧ 
  10 ≤ n ∧ 
  n < 100 ∧ 
  ∀ m : ℕ, (m % 17 = 0 ∧ 10 ≤ m ∧ m < 100) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l3844_384473


namespace NUMINAMATH_CALUDE_ripe_oranges_harvested_per_day_l3844_384405

/-- The number of days of harvest -/
def harvest_days : ℕ := 25

/-- The total number of sacks of ripe oranges after the harvest period -/
def total_ripe_oranges : ℕ := 2050

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges_per_day : ℕ := total_ripe_oranges / harvest_days

theorem ripe_oranges_harvested_per_day :
  ripe_oranges_per_day = 82 := by
  sorry

end NUMINAMATH_CALUDE_ripe_oranges_harvested_per_day_l3844_384405


namespace NUMINAMATH_CALUDE_valid_rectangle_exists_l3844_384429

/-- Represents a point in a triangular grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a rectangle in the triangular grid -/
structure Rectangle where
  bottomLeft : GridPoint
  width : ℕ
  height : ℕ

/-- Counts the number of grid points on the boundary of a rectangle -/
def boundaryPoints (rect : Rectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- Counts the number of grid points in the interior of a rectangle -/
def interiorPoints (rect : Rectangle) : ℕ :=
  rect.width * rect.height + (rect.width - 1) * (rect.height - 1)

/-- Checks if a rectangle satisfies the required conditions -/
def isValidRectangle (rect : Rectangle) : Prop :=
  boundaryPoints rect = interiorPoints rect

/-- Main theorem: There exists a valid rectangle in the triangular grid -/
theorem valid_rectangle_exists : ∃ (rect : Rectangle), isValidRectangle rect :=
  sorry

end NUMINAMATH_CALUDE_valid_rectangle_exists_l3844_384429
