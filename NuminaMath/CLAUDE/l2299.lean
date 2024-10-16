import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_equality_l2299_229916

theorem polynomial_equality (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x + 1) ^ 4 = a + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) →
  a - a₁ + a₂ - a₃ + a₄ = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2299_229916


namespace NUMINAMATH_CALUDE_dot_product_special_vectors_l2299_229982

theorem dot_product_special_vectors :
  let a : ℝ × ℝ := (Real.sin (15 * π / 180), Real.sin (75 * π / 180))
  let b : ℝ × ℝ := (Real.cos (30 * π / 180), Real.sin (30 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_special_vectors_l2299_229982


namespace NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l2299_229950

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_two_planes_implies_parallel 
  (l : Line) (α β : Plane) (h1 : α ≠ β) (h2 : perpendicular l α) (h3 : perpendicular l β) :
  parallel α β := by sorry

end NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l2299_229950


namespace NUMINAMATH_CALUDE_g_range_l2299_229976

/-- The function representing the curve C -/
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 3)

/-- The function g(t) representing the product of magnitudes of OP and OQ -/
def g (t : ℝ) : ℝ := |(3 - t) * (1 + t^2)|

/-- Theorem stating that the range of g(t) is [0, ∞) -/
theorem g_range : Set.range g = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_g_range_l2299_229976


namespace NUMINAMATH_CALUDE_probability_four_even_dice_l2299_229997

theorem probability_four_even_dice (n : ℕ) (p : ℚ) : 
  n = 8 →
  p = 1/2 →
  (n.choose (n/2)) * p^n = 35/128 :=
by sorry

end NUMINAMATH_CALUDE_probability_four_even_dice_l2299_229997


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2299_229917

/-- The line l is tangent to the circle C -/
theorem line_tangent_to_circle :
  ∀ (x y : ℝ),
  (x - y + 4 = 0) →
  ((x - 2)^2 + (y - 2)^2 = 8) →
  ∃! (p : ℝ × ℝ), p.1 - p.2 + 4 = 0 ∧ (p.1 - 2)^2 + (p.2 - 2)^2 = 8 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2299_229917


namespace NUMINAMATH_CALUDE_aluminum_carbonate_molecular_weight_l2299_229989

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of the given moles of Aluminum carbonate in grams -/
def given_molecular_weight : ℝ := 1170

/-- The molecular formula of Aluminum carbonate: Al₂(CO₃)₃ -/
structure AluminumCarbonate where
  Al : Nat
  C : Nat
  O : Nat

/-- The correct molecular formula of Aluminum carbonate -/
def Al2CO3_3 : AluminumCarbonate := ⟨2, 3, 9⟩

/-- Calculate the molecular weight of Aluminum carbonate -/
def molecular_weight (formula : AluminumCarbonate) : ℝ :=
  formula.Al * atomic_weight_Al + formula.C * atomic_weight_C + formula.O * atomic_weight_O

/-- Theorem: The molecular weight of Aluminum carbonate is approximately 234.99 g/mol -/
theorem aluminum_carbonate_molecular_weight :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |molecular_weight Al2CO3_3 - 234.99| < ε :=
sorry

end NUMINAMATH_CALUDE_aluminum_carbonate_molecular_weight_l2299_229989


namespace NUMINAMATH_CALUDE_complex_expression_equals_zero_l2299_229983

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The theorem stating that the complex expression equals 0 -/
theorem complex_expression_equals_zero : (1 + i) / (1 - i) + i ^ 3 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_zero_l2299_229983


namespace NUMINAMATH_CALUDE_deck_size_l2299_229928

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1 / 5 →
  (r : ℚ) / (r + b + 6) = 1 / 7 →
  r + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_deck_size_l2299_229928


namespace NUMINAMATH_CALUDE_square_plot_area_l2299_229939

theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) : 
  price_per_foot = 58 → total_cost = 1624 → 
  ∃ (side_length : ℝ), 
    side_length > 0 ∧ 
    4 * side_length * price_per_foot = total_cost ∧ 
    side_length^2 = 49 := by
  sorry

#check square_plot_area

end NUMINAMATH_CALUDE_square_plot_area_l2299_229939


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l2299_229900

theorem imaginary_part_of_2_minus_i :
  Complex.im (2 - Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l2299_229900


namespace NUMINAMATH_CALUDE_two_yellow_marbles_prob_l2299_229945

/-- Represents the number of marbles of each color in the box -/
structure MarbleBox where
  blue : ℕ
  yellow : ℕ
  orange : ℕ

/-- Calculates the total number of marbles in the box -/
def totalMarbles (box : MarbleBox) : ℕ :=
  box.blue + box.yellow + box.orange

/-- Calculates the probability of drawing a yellow marble -/
def probYellow (box : MarbleBox) : ℚ :=
  box.yellow / (totalMarbles box)

/-- The probability of drawing two yellow marbles in succession with replacement -/
def probTwoYellow (box : MarbleBox) : ℚ :=
  (probYellow box) * (probYellow box)

theorem two_yellow_marbles_prob :
  let box : MarbleBox := { blue := 4, yellow := 5, orange := 6 }
  probTwoYellow box = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_two_yellow_marbles_prob_l2299_229945


namespace NUMINAMATH_CALUDE_vector_magnitude_l2299_229906

/-- Given two vectors a and b in ℝ², prove that the magnitude of b is 5 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (3, -2) → a + b = (0, 2) → ‖b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2299_229906


namespace NUMINAMATH_CALUDE_defective_draws_count_l2299_229934

/-- The number of ways to draw at least 3 defective products out of 5 from a batch of 50 products containing 4 defective ones -/
def defective_draws : ℕ := sorry

/-- Total number of products in the batch -/
def total_products : ℕ := 50

/-- Number of defective products in the batch -/
def defective_products : ℕ := 4

/-- Number of products drawn -/
def drawn_products : ℕ := 5

theorem defective_draws_count : defective_draws = 4186 := by sorry

end NUMINAMATH_CALUDE_defective_draws_count_l2299_229934


namespace NUMINAMATH_CALUDE_remainder_theorem_l2299_229902

def polynomial (x : ℝ) : ℝ := 4*x^6 - x^5 - 8*x^4 + 3*x^2 + 5*x - 15

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (x - 3) * q x + 2079 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2299_229902


namespace NUMINAMATH_CALUDE_probability_second_class_first_given_first_class_second_is_two_fifths_l2299_229975

/-- Represents the class of an item -/
inductive ItemClass
| FirstClass
| SecondClass

/-- Represents the box with items -/
structure Box where
  firstClassCount : ℕ
  secondClassCount : ℕ

/-- Represents the outcome of drawing two items -/
structure DrawOutcome where
  first : ItemClass
  second : ItemClass

def Box.totalCount (b : Box) : ℕ := b.firstClassCount + b.secondClassCount

/-- The probability of drawing a second-class item first, given that the second item is first-class -/
def probabilitySecondClassFirstGivenFirstClassSecond (b : Box) : ℚ :=
  let totalOutcomes := b.firstClassCount * (b.firstClassCount - 1) + b.secondClassCount * b.firstClassCount
  let favorableOutcomes := b.secondClassCount * b.firstClassCount
  favorableOutcomes / totalOutcomes

theorem probability_second_class_first_given_first_class_second_is_two_fifths :
  let b : Box := { firstClassCount := 4, secondClassCount := 2 }
  probabilitySecondClassFirstGivenFirstClassSecond b = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_class_first_given_first_class_second_is_two_fifths_l2299_229975


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l2299_229943

theorem triangle_is_equilateral (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π / 2 →  -- Angle A is acute
  3 * b = 2 * Real.sqrt 3 * a * Real.sin B →  -- Given equation
  Real.cos B = Real.cos C →  -- Given condition
  0 < B ∧ B < π →  -- B is a valid angle
  0 < C ∧ C < π →  -- C is a valid angle
  A + B + C = π →  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3  -- Equilateral triangle
  := by sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l2299_229943


namespace NUMINAMATH_CALUDE_largest_divisible_by_all_less_than_cube_root_l2299_229922

theorem largest_divisible_by_all_less_than_cube_root : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < n^(1/3) → n % k = 0) ∧
  (∀ (m : ℕ), m > n → ∃ (j : ℕ), j > 0 ∧ j < m^(1/3) ∧ m % j ≠ 0) ∧
  n = 420 := by
sorry

end NUMINAMATH_CALUDE_largest_divisible_by_all_less_than_cube_root_l2299_229922


namespace NUMINAMATH_CALUDE_john_uber_profit_l2299_229935

/-- John's profit from driving Uber --/
def uber_profit (earnings depreciation : ℕ) : ℕ :=
  earnings - depreciation

/-- Depreciation of John's car --/
def car_depreciation (purchase_price trade_in_value : ℕ) : ℕ :=
  purchase_price - trade_in_value

theorem john_uber_profit :
  let earnings : ℕ := 30000
  let purchase_price : ℕ := 18000
  let trade_in_value : ℕ := 6000
  uber_profit earnings (car_depreciation purchase_price trade_in_value) = 18000 := by
sorry

end NUMINAMATH_CALUDE_john_uber_profit_l2299_229935


namespace NUMINAMATH_CALUDE_correct_answers_for_zero_score_correct_answers_for_zero_score_proof_l2299_229978

theorem correct_answers_for_zero_score (total_questions : Nat) 
  (points_for_correct : Int) (points_for_wrong : Int) 
  (correct_answers : Nat) : Prop :=
  total_questions = 26 →
  points_for_correct = 8 →
  points_for_wrong = -5 →
  correct_answers ≤ total_questions →
  (correct_answers : Int) * points_for_correct + 
    (total_questions - correct_answers : Int) * points_for_wrong = 0 →
  correct_answers = 10

-- Proof
theorem correct_answers_for_zero_score_proof :
  correct_answers_for_zero_score 26 8 (-5) 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_answers_for_zero_score_correct_answers_for_zero_score_proof_l2299_229978


namespace NUMINAMATH_CALUDE_least_frood_number_l2299_229932

def drop_score (n : ℕ) : ℕ := n * (n + 1) / 2

def eat_score (n : ℕ) : ℕ := 10 * n

theorem least_frood_number : ∀ k : ℕ, k < 20 → drop_score k ≤ eat_score k ∧ drop_score 20 > eat_score 20 := by
  sorry

end NUMINAMATH_CALUDE_least_frood_number_l2299_229932


namespace NUMINAMATH_CALUDE_salary_change_l2299_229925

theorem salary_change (original : ℝ) (h : original > 0) :
  ∃ (increase : ℝ),
    (original * 0.7 * (1 + increase) = original * 0.91) ∧
    increase = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_l2299_229925


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l2299_229998

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l2299_229998


namespace NUMINAMATH_CALUDE_distance_is_27_l2299_229936

/-- The distance between two locations A and B, where two people walk towards each other, 
    meet, continue to their destinations, turn back, and meet again. -/
def distance_between_locations : ℝ :=
  let first_meeting_distance_from_A : ℝ := 10
  let second_meeting_distance_from_B : ℝ := 3
  first_meeting_distance_from_A + (2 * first_meeting_distance_from_A - second_meeting_distance_from_B)

/-- Theorem stating that the distance between locations A and B is 27 kilometers. -/
theorem distance_is_27 : distance_between_locations = 27 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_27_l2299_229936


namespace NUMINAMATH_CALUDE_sum_of_squares_first_15_sum_of_squares_16_to_30_main_theorem_l2299_229992

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_first_15 :
  sum_of_squares 15 = 1240 :=
by sorry

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8205 :=
by sorry

theorem main_theorem :
  sum_of_squares 15 = 1240 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_first_15_sum_of_squares_16_to_30_main_theorem_l2299_229992


namespace NUMINAMATH_CALUDE_tims_prank_combinations_l2299_229923

/-- Represents the number of choices for each day of the prank --/
structure PrankChoices where
  day1 : Nat
  day2 : Nat
  day3 : Nat
  day4 : Nat → Nat
  day5 : Nat

/-- Calculates the total number of combinations for the prank --/
def totalCombinations (choices : PrankChoices) : Nat :=
  choices.day1 * choices.day2 * choices.day3 * 
  (choices.day3 * choices.day4 1 + choices.day3 * choices.day4 2 + choices.day3 * choices.day4 3) *
  choices.day5

/-- The specific choices for Tim's prank --/
def timsPrankChoices : PrankChoices where
  day1 := 1
  day2 := 2
  day3 := 3
  day4 := fun n => match n with
    | 1 => 2
    | 2 => 3
    | _ => 1
  day5 := 1

theorem tims_prank_combinations :
  totalCombinations timsPrankChoices = 36 := by
  sorry

end NUMINAMATH_CALUDE_tims_prank_combinations_l2299_229923


namespace NUMINAMATH_CALUDE_line_intersection_point_l2299_229991

theorem line_intersection_point :
  ∃! p : ℝ × ℝ, 
    5 * p.1 - 3 * p.2 = 15 ∧ 
    4 * p.1 + 2 * p.2 = 14 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_line_intersection_point_l2299_229991


namespace NUMINAMATH_CALUDE_dilation_of_negative_i_l2299_229942

def dilation (c k z : ℂ) : ℂ := c + k * (z - c)

theorem dilation_of_negative_i :
  let c : ℂ := 2 - 3*I
  let k : ℝ := 3
  let z : ℂ := -I
  dilation c k z = -4 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_dilation_of_negative_i_l2299_229942


namespace NUMINAMATH_CALUDE_opposite_pairs_l2299_229919

theorem opposite_pairs : 
  ¬((-2 : ℝ) = -(1/2)) ∧ 
  ¬(|(-1)| = -1) ∧ 
  ¬(((-3)^2 : ℝ) = -(3^2)) ∧ 
  (-5 : ℝ) = -(-(-5)) := by sorry

end NUMINAMATH_CALUDE_opposite_pairs_l2299_229919


namespace NUMINAMATH_CALUDE_no_equal_distribution_l2299_229911

/-- Represents the number of glasses --/
def num_glasses : ℕ := 2018

/-- Represents the total amount of champagne --/
def total_champagne : ℕ := 2019

/-- Represents a distribution of champagne among glasses --/
def Distribution := Fin num_glasses → ℚ

/-- Checks if a distribution is valid (sums to total_champagne) --/
def is_valid_distribution (d : Distribution) : Prop :=
  (Finset.sum Finset.univ (λ i => d i)) = total_champagne

/-- Represents the equalization operation on two glasses --/
def equalize (d : Distribution) (i j : Fin num_glasses) : Distribution :=
  λ k => if k = i ∨ k = j then (d i + d j) / 2 else d k

/-- Represents the property of all glasses having equal integer amount --/
def all_equal_integer (d : Distribution) : Prop :=
  ∃ n : ℕ, ∀ i : Fin num_glasses, d i = n

/-- The main theorem stating that no initial distribution can result in
    all glasses having equal integer amount after repeated equalization --/
theorem no_equal_distribution :
  ¬∃ (d : Distribution), is_valid_distribution d ∧
    ∃ (seq : ℕ → Fin num_glasses × Fin num_glasses),
      ∃ (n : ℕ), all_equal_integer (Nat.iterate (λ d' => equalize d' (seq n).1 (seq n).2) n d) := by
  sorry

end NUMINAMATH_CALUDE_no_equal_distribution_l2299_229911


namespace NUMINAMATH_CALUDE_conic_section_foci_l2299_229938

-- Define the polar equation of the conic section
def polar_equation (ρ θ : ℝ) : Prop := ρ = 16 / (5 - 3 * Real.cos θ)

-- Define the focus coordinates
def focus1 : ℝ × ℝ := (0, 0)
def focus2 : ℝ × ℝ := (6, 0)

-- Theorem statement
theorem conic_section_foci (ρ θ : ℝ) :
  polar_equation ρ θ → (focus1 = (0, 0) ∧ focus2 = (6, 0)) :=
sorry

end NUMINAMATH_CALUDE_conic_section_foci_l2299_229938


namespace NUMINAMATH_CALUDE_diamond_neg_one_six_l2299_229955

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a * b^2 - b + 1

-- Theorem statement
theorem diamond_neg_one_six : diamond (-1) 6 = -41 := by
  sorry

end NUMINAMATH_CALUDE_diamond_neg_one_six_l2299_229955


namespace NUMINAMATH_CALUDE_range_of_square_root_set_l2299_229973

theorem range_of_square_root_set (A : Set ℝ) (a : ℝ) :
  (A.Nonempty) →
  (A = {x : ℝ | x^2 = a}) →
  (∃ (y : ℝ), ∀ (x : ℝ), x ∈ A ↔ y ≤ x ∧ x^2 = a) :=
by sorry

end NUMINAMATH_CALUDE_range_of_square_root_set_l2299_229973


namespace NUMINAMATH_CALUDE_total_cost_l2299_229988

/-- The cost of an enchilada -/
def e : ℝ := sorry

/-- The cost of a taco -/
def t : ℝ := sorry

/-- The cost of a burrito -/
def b : ℝ := sorry

/-- The first condition: 4 enchiladas, 5 tacos, and 2 burritos cost $8.20 -/
axiom condition1 : 4 * e + 5 * t + 2 * b = 8.20

/-- The second condition: 6 enchiladas, 3 tacos, and 4 burritos cost $9.40 -/
axiom condition2 : 6 * e + 3 * t + 4 * b = 9.40

/-- Theorem stating that the total cost of 5 enchiladas, 6 tacos, and 3 burritos is $12.20 -/
theorem total_cost : 5 * e + 6 * t + 3 * b = 12.20 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_l2299_229988


namespace NUMINAMATH_CALUDE_archery_probabilities_l2299_229903

/-- Represents the probabilities of hitting different rings in archery --/
structure ArcheryProbabilities where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ
  ring7 : ℝ
  below7 : ℝ

/-- The given probabilities for archer Zhang Qiang --/
def zhangQiang : ArcheryProbabilities :=
  { ring10 := 0.24
  , ring9 := 0.28
  , ring8 := 0.19
  , ring7 := 0.16
  , below7 := 0.13 }

/-- The sum of all probabilities should be 1 --/
axiom probSum (p : ArcheryProbabilities) : p.ring10 + p.ring9 + p.ring8 + p.ring7 + p.below7 = 1

theorem archery_probabilities (p : ArcheryProbabilities) 
  (h : p = zhangQiang) : 
  (p.ring10 + p.ring9 = 0.52) ∧ 
  (p.ring10 + p.ring9 + p.ring8 + p.ring7 = 0.87) ∧ 
  (p.ring7 + p.below7 = 0.29) := by
  sorry


end NUMINAMATH_CALUDE_archery_probabilities_l2299_229903


namespace NUMINAMATH_CALUDE_yoojeong_rabbits_l2299_229940

/-- The number of animals Minyoung has -/
def minyoung_animals : ℕ := 9 + 3 + 5

/-- The number of animals Yoojeong has -/
def yoojeong_animals : ℕ := minyoung_animals + 2

/-- The number of dogs Yoojeong has -/
def yoojeong_dogs : ℕ := 7

theorem yoojeong_rabbits :
  ∃ (cats rabbits : ℕ),
    yoojeong_animals = yoojeong_dogs + cats + rabbits ∧
    cats = rabbits - 2 ∧
    rabbits = 7 := by sorry

end NUMINAMATH_CALUDE_yoojeong_rabbits_l2299_229940


namespace NUMINAMATH_CALUDE_land_conversion_equation_l2299_229933

/-- Represents the land conversion scenario in a village --/
theorem land_conversion_equation (x : ℝ) : True :=
  let original_forest : ℝ := 108
  let original_arable : ℝ := 54
  let conversion_percentage : ℝ := 0.2
  let new_forest : ℝ := original_forest + x
  let new_arable : ℝ := original_arable - x
  let equation := (new_arable = conversion_percentage * new_forest)
by
  sorry

end NUMINAMATH_CALUDE_land_conversion_equation_l2299_229933


namespace NUMINAMATH_CALUDE_file_download_rate_l2299_229926

/-- Given a file download scenario, prove the download rate for the latter part. -/
theorem file_download_rate 
  (file_size : ℝ) 
  (initial_rate : ℝ) 
  (initial_size : ℝ) 
  (total_time : ℝ) 
  (h1 : file_size = 90) 
  (h2 : initial_rate = 5) 
  (h3 : initial_size = 60) 
  (h4 : total_time = 15) : 
  (file_size - initial_size) / (total_time - initial_size / initial_rate) = 10 := by
  sorry

end NUMINAMATH_CALUDE_file_download_rate_l2299_229926


namespace NUMINAMATH_CALUDE_parallel_lines_intersection_l2299_229944

/-- Two lines are parallel if they have the same slope -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- A point (x, y) lies on a line ax + by = c if the equation is satisfied -/
def point_on_line (a b c x y : ℝ) : Prop := a * x + b * y = c

theorem parallel_lines_intersection (c d : ℝ) : 
  parallel (3 / 4) (-6 / d) ∧ 
  point_on_line 3 (-4) c 2 (-3) ∧
  point_on_line 6 d (2 * c) 2 (-3) →
  c = 18 ∧ d = -8 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_intersection_l2299_229944


namespace NUMINAMATH_CALUDE_square_equation_solution_l2299_229962

theorem square_equation_solution : ∃ x : ℤ, (2012 + x)^2 = x^2 ∧ x = -1006 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2299_229962


namespace NUMINAMATH_CALUDE_abc_inequality_l2299_229968

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = a * b * c) : 
  max a (max b c) > 17/10 := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l2299_229968


namespace NUMINAMATH_CALUDE_select_students_l2299_229952

theorem select_students (n m : ℕ) (h1 : n = 10) (h2 : m = 3) : 
  Nat.choose n m = 120 := by
  sorry

end NUMINAMATH_CALUDE_select_students_l2299_229952


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solution_exists_l2299_229913

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for part I
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = Set.Ioo (-4 : ℝ) (2/3) :=
sorry

-- Theorem for part II
theorem range_of_a_for_solution_exists :
  {a : ℝ | ∃ x, f x ≤ a - a^2/2} = Set.Icc (-1 : ℝ) 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solution_exists_l2299_229913


namespace NUMINAMATH_CALUDE_sqrt_63_minus_7_sqrt_one_seventh_l2299_229960

theorem sqrt_63_minus_7_sqrt_one_seventh (x : ℝ) : 
  Real.sqrt 63 - 7 * Real.sqrt (1 / 7) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_63_minus_7_sqrt_one_seventh_l2299_229960


namespace NUMINAMATH_CALUDE_multiple_solutions_and_no_solution_for_2891_l2299_229995

def equation (x y n : ℤ) : Prop := x^3 - 3*x*y^2 + y^3 = n

theorem multiple_solutions_and_no_solution_for_2891 :
  (∀ n : ℤ, (∃ x y : ℤ, equation x y n) → 
    (∃ a b c d : ℤ, equation a b n ∧ equation c d n ∧ 
      (a, b) ≠ (x, y) ∧ (c, d) ≠ (x, y) ∧ (a, b) ≠ (c, d))) ∧
  (¬ ∃ x y : ℤ, equation x y 2891) :=
by sorry

end NUMINAMATH_CALUDE_multiple_solutions_and_no_solution_for_2891_l2299_229995


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2299_229964

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, (m - 1) * x^2 + x + m^2 - 1 = 0) ∧ 
  ((m - 1) * 0^2 + 0 + m^2 - 1 = 0) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2299_229964


namespace NUMINAMATH_CALUDE_five_T_three_equals_38_l2299_229949

-- Define the new operation ⊤
def T (a b : ℤ) : ℤ := 4*a + 6*b

-- Theorem statement
theorem five_T_three_equals_38 : T 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_five_T_three_equals_38_l2299_229949


namespace NUMINAMATH_CALUDE_solution_equivalence_l2299_229924

theorem solution_equivalence :
  (∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_equivalence_l2299_229924


namespace NUMINAMATH_CALUDE_least_multiple_24_greater_500_l2299_229905

theorem least_multiple_24_greater_500 : ∃ n : ℕ, 
  (24 * n > 500) ∧ (∀ m : ℕ, 24 * m > 500 → 24 * n ≤ 24 * m) ∧ (24 * n = 504) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_24_greater_500_l2299_229905


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l2299_229999

-- Define the points and quadrilateral
variable (A B C D P Q R S X Y : Point₂)

-- Define the cyclic quadrilateral property
def is_cyclic_quadrilateral (A B C D : Point₂) : Prop := sorry

-- Define the property of opposite sides not being parallel
def opposite_sides_not_parallel (A B C D : Point₂) : Prop := sorry

-- Define the interior point property
def is_interior_point (P : Point₂) (A B : Point₂) : Prop := sorry

-- Define angle equality
def angle_eq (A B C D E F : Point₂) : Prop := sorry

-- Define line intersection
def intersects_at (A B C D X : Point₂) : Prop := sorry

-- Define parallel or coincident lines
def parallel_or_coincide (A B C D : Point₂) : Prop := sorry

-- Theorem statement
theorem cyclic_quadrilateral_theorem 
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_not_parallel : opposite_sides_not_parallel A B C D)
  (h_P_interior : is_interior_point P A B)
  (h_Q_interior : is_interior_point Q B C)
  (h_R_interior : is_interior_point R C D)
  (h_S_interior : is_interior_point S D A)
  (h_angle1 : angle_eq P D A P C B)
  (h_angle2 : angle_eq Q A B Q D C)
  (h_angle3 : angle_eq R B C R A D)
  (h_angle4 : angle_eq S C D S B A)
  (h_intersect1 : intersects_at A Q B S X)
  (h_intersect2 : intersects_at D Q C S Y) :
  parallel_or_coincide P R X Y :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l2299_229999


namespace NUMINAMATH_CALUDE_arnel_pencil_sharing_l2299_229972

/-- Calculates the number of pencils each friend receives when Arnel shares his pencils --/
def pencils_per_friend (num_boxes : ℕ) (pencils_per_box : ℕ) (kept_pencils : ℕ) (num_friends : ℕ) : ℕ :=
  ((num_boxes * pencils_per_box) - kept_pencils) / num_friends

/-- Proves that each friend receives 8 pencils under the given conditions --/
theorem arnel_pencil_sharing :
  pencils_per_friend 10 5 10 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arnel_pencil_sharing_l2299_229972


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l2299_229971

-- Define the number of white and black balls
def white_balls : ℕ := 7
def black_balls : ℕ := 8
def total_balls : ℕ := white_balls + black_balls

-- Define the number of balls drawn
def balls_drawn : ℕ := 3

-- Define the probability function
def probability_all_white : ℚ := (white_balls.choose balls_drawn) / (total_balls.choose balls_drawn)

-- Theorem statement
theorem probability_three_white_balls :
  probability_all_white = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l2299_229971


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l2299_229918

theorem fruit_basket_problem (total_fruits : ℕ) (mango_count : ℕ) (pear_count : ℕ) (pawpaw_count : ℕ) 
  (h1 : total_fruits = 58)
  (h2 : mango_count = 18)
  (h3 : pear_count = 10)
  (h4 : pawpaw_count = 12) :
  ∃ (lemon_count : ℕ), 
    lemon_count = (total_fruits - (mango_count + pear_count + pawpaw_count)) / 2 ∧ 
    lemon_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l2299_229918


namespace NUMINAMATH_CALUDE_mike_bird_feeding_l2299_229951

/-- The number of seeds Mike throws to the birds on the left -/
def seeds_left : ℕ := 20

/-- The total number of seeds Mike starts with -/
def total_seeds : ℕ := 120

/-- The number of additional seeds thrown -/
def additional_seeds : ℕ := 30

/-- The number of seeds left at the end -/
def remaining_seeds : ℕ := 30

theorem mike_bird_feeding :
  seeds_left + 2 * seeds_left + additional_seeds + remaining_seeds = total_seeds :=
by sorry

end NUMINAMATH_CALUDE_mike_bird_feeding_l2299_229951


namespace NUMINAMATH_CALUDE_strictly_increasing_function_property_l2299_229910

/-- A function f: ℝ → ℝ that satisfies the given conditions -/
def StrictlyIncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

theorem strictly_increasing_function_property
  (f : ℝ → ℝ)
  (h : StrictlyIncreasingFunction f)
  (h1 : f 5 = -1)
  (h2 : f 7 = 0) :
  f (-3) < -1 := by
  sorry

end NUMINAMATH_CALUDE_strictly_increasing_function_property_l2299_229910


namespace NUMINAMATH_CALUDE_better_fit_larger_r_squared_l2299_229956

/-- Represents a regression model -/
structure RegressionModel where
  /-- The correlation index (R-squared) of the model -/
  r_squared : ℝ
  /-- Assumes R-squared is between 0 and 1 -/
  r_squared_bounds : 0 ≤ r_squared ∧ r_squared ≤ 1

/-- Represents the fitting effect of a regression model -/
def fitting_effect (model : RegressionModel) : ℝ := sorry

/-- Theorem stating that a larger R-squared indicates a better fitting effect -/
theorem better_fit_larger_r_squared 
  (model1 model2 : RegressionModel) 
  (h : model1.r_squared < model2.r_squared) : 
  fitting_effect model1 < fitting_effect model2 := by sorry

end NUMINAMATH_CALUDE_better_fit_larger_r_squared_l2299_229956


namespace NUMINAMATH_CALUDE_x_prime_condition_x_divisibility_l2299_229929

def x : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => x (n + 1) + x n

theorem x_prime_condition (n : ℕ) (h : n ≥ 1) :
  Nat.Prime (x n) → Nat.Prime n ∨ (∀ p, Nat.Prime p → p > 2 → ¬p ∣ n) :=
sorry

theorem x_divisibility (m n : ℕ) :
  x m ∣ x n ↔ (∃ k, (m = 0 ∧ n = 3 * k) ∨ (m = 1 ∧ n = k) ∨ (∃ t, m = n ∧ n = (2 * t + 1) * n)) :=
sorry

end NUMINAMATH_CALUDE_x_prime_condition_x_divisibility_l2299_229929


namespace NUMINAMATH_CALUDE_division_in_base5_l2299_229985

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Performs division in base 5 -/
def divBase5 (a b : ℕ) : ℕ := 
  base10ToBase5 (base5ToBase10 a / base5ToBase10 b)

theorem division_in_base5 : divBase5 1302 23 = 30 := by sorry

end NUMINAMATH_CALUDE_division_in_base5_l2299_229985


namespace NUMINAMATH_CALUDE_subtract_preserves_inequality_l2299_229909

theorem subtract_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtract_preserves_inequality_l2299_229909


namespace NUMINAMATH_CALUDE_reflect_triangle_xy_l2299_229937

/-- A triangle in a 2D coordinate plane -/
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- Reflection of a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflection of a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Composition of reflections over x-axis and y-axis -/
def reflect_xy (p : ℝ × ℝ) : ℝ × ℝ := reflect_y (reflect_x p)

/-- Theorem: Reflecting a triangle over x-axis then y-axis negates both coordinates -/
theorem reflect_triangle_xy (t : Triangle) :
  let t' := Triangle.mk (reflect_xy t.v1) (reflect_xy t.v2) (reflect_xy t.v3)
  t'.v1 = (-t.v1.1, -t.v1.2) ∧
  t'.v2 = (-t.v2.1, -t.v2.2) ∧
  t'.v3 = (-t.v3.1, -t.v3.2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_triangle_xy_l2299_229937


namespace NUMINAMATH_CALUDE_pool_width_proof_l2299_229963

theorem pool_width_proof (drain_rate : ℝ) (drain_time : ℝ) (length : ℝ) (depth : ℝ) 
  (h1 : drain_rate = 60)
  (h2 : drain_time = 2000)
  (h3 : length = 150)
  (h4 : depth = 10) :
  drain_rate * drain_time / (length * depth) = 80 :=
by sorry

end NUMINAMATH_CALUDE_pool_width_proof_l2299_229963


namespace NUMINAMATH_CALUDE_bus_travel_cost_l2299_229979

/-- The cost of bus travel in dollars per kilometer -/
def bus_cost_per_km : ℚ := 0.20

/-- The distance from A to B in kilometers -/
def distance_AB : ℕ := 4500

/-- The total cost of bus travel from A to B in dollars -/
def total_cost : ℚ := bus_cost_per_km * distance_AB

theorem bus_travel_cost :
  total_cost = 900 :=
sorry

end NUMINAMATH_CALUDE_bus_travel_cost_l2299_229979


namespace NUMINAMATH_CALUDE_correct_reasoning_combination_l2299_229946

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| PartToWhole
| GeneralToSpecific
| SpecificToSpecific

-- Define a function that maps reasoning types to their directions
def reasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating that the correct combination is Inductive, Deductive, and Analogical
theorem correct_reasoning_combination :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
sorry

end NUMINAMATH_CALUDE_correct_reasoning_combination_l2299_229946


namespace NUMINAMATH_CALUDE_hyperbola_range_l2299_229953

/-- A function that represents the equation of a hyperbola -/
def hyperbola_equation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / (m + 2) - y^2 / (2*m - 1) = 1

/-- The theorem stating the range of m for which the equation represents a hyperbola -/
theorem hyperbola_range (m : ℝ) :
  (∀ x y, ∃ (h : hyperbola_equation m x y), True) ↔ m < -2 ∨ m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_range_l2299_229953


namespace NUMINAMATH_CALUDE_one_subject_count_l2299_229980

/-- The number of students taking both chemistry and physics -/
def both_subjects : ℕ := 15

/-- The total number of students in the chemistry class -/
def chemistry_total : ℕ := 30

/-- The number of students taking only physics -/
def physics_only : ℕ := 18

/-- The number of students taking chemistry or physics but not both -/
def one_subject_only : ℕ := (chemistry_total - both_subjects) + physics_only

theorem one_subject_count : one_subject_only = 33 := by
  sorry

end NUMINAMATH_CALUDE_one_subject_count_l2299_229980


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2299_229907

-- Define the discriminant function for a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- State the theorem
theorem quadratic_discriminant :
  discriminant 1 3 1 = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2299_229907


namespace NUMINAMATH_CALUDE_percent_decrease_proof_l2299_229957

def initial_cost : ℝ := 120
def final_cost : ℝ := 45

theorem percent_decrease_proof :
  (initial_cost - final_cost) / initial_cost * 100 = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_proof_l2299_229957


namespace NUMINAMATH_CALUDE_additional_cats_needed_l2299_229931

def current_cats : ℕ := 11
def target_cats : ℕ := 43

theorem additional_cats_needed : target_cats - current_cats = 32 := by
  sorry

end NUMINAMATH_CALUDE_additional_cats_needed_l2299_229931


namespace NUMINAMATH_CALUDE_second_number_problem_l2299_229908

theorem second_number_problem (x y : ℤ) : 
  y = 2 * x - 3 → 
  x + y = 57 → 
  y = 37 := by
sorry

end NUMINAMATH_CALUDE_second_number_problem_l2299_229908


namespace NUMINAMATH_CALUDE_equation_solution_l2299_229954

theorem equation_solution (n : ℝ) : 
  let m := 5 * n + 5
  2 / (n + 2) + 3 / (n + 2) + m / (n + 2) = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2299_229954


namespace NUMINAMATH_CALUDE_correct_total_distance_l2299_229996

/-- Converts kilometers to meters -/
def km_to_m (km : ℝ) : ℝ := km * 1000

/-- Calculates the total distance in meters -/
def total_distance (initial_km : ℝ) (additional_m : ℝ) : ℝ :=
  km_to_m initial_km + additional_m

/-- Theorem: The correct total distance is 3700 meters -/
theorem correct_total_distance :
  total_distance 3.5 200 = 3700 := by sorry

end NUMINAMATH_CALUDE_correct_total_distance_l2299_229996


namespace NUMINAMATH_CALUDE_total_shot_cost_l2299_229904

/-- Represents the types of dogs Chuck breeds -/
inductive DogBreed
  | GoldenRetriever
  | GermanShepherd
  | Bulldog

/-- Represents the information for each dog breed -/
structure BreedInfo where
  pregnantDogs : Nat
  puppiesPerDog : Nat
  shotsPerPuppy : Nat
  costPerShot : Nat

/-- Calculates the total cost of shots for a specific breed -/
def breedShotCost (info : BreedInfo) : Nat :=
  info.pregnantDogs * info.puppiesPerDog * info.shotsPerPuppy * info.costPerShot

/-- Represents Chuck's dog breeding operation -/
def ChucksDogs : DogBreed → BreedInfo
  | DogBreed.GoldenRetriever => ⟨3, 4, 2, 5⟩
  | DogBreed.GermanShepherd => ⟨2, 5, 3, 8⟩
  | DogBreed.Bulldog => ⟨4, 3, 4, 10⟩

/-- Theorem stating the total cost of shots for all puppies -/
theorem total_shot_cost :
  (breedShotCost (ChucksDogs DogBreed.GoldenRetriever) +
   breedShotCost (ChucksDogs DogBreed.GermanShepherd) +
   breedShotCost (ChucksDogs DogBreed.Bulldog)) = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_shot_cost_l2299_229904


namespace NUMINAMATH_CALUDE_f_increasing_and_not_in_second_quadrant_l2299_229969

-- Define the function
def f (x : ℝ) : ℝ := 2 * x - 5

-- State the theorem
theorem f_increasing_and_not_in_second_quadrant :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x y : ℝ, x < 0 ∧ y > 0 → ¬(f x = y)) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_and_not_in_second_quadrant_l2299_229969


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2299_229984

/-- A quadratic function is a function of the form f(x) = ax² + bx + c, where a ≠ 0 -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 2x² - 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

/-- Theorem: f(x) = 2x² - 2x + 1 is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2299_229984


namespace NUMINAMATH_CALUDE_proposition_conditions_l2299_229912

theorem proposition_conditions (p q : Prop) : 
  (p ∨ q) → ¬(p ∧ q) → ¬p → (¬p ∧ q) := by sorry

end NUMINAMATH_CALUDE_proposition_conditions_l2299_229912


namespace NUMINAMATH_CALUDE_reams_needed_l2299_229966

-- Define the constants
def stories_per_week : ℕ := 3
def pages_per_story : ℕ := 50
def novel_pages_per_year : ℕ := 1200
def pages_per_sheet : ℕ := 2
def sheets_per_ream : ℕ := 500
def weeks_in_year : ℕ := 52
def weeks_to_calculate : ℕ := 12

-- Theorem to prove
theorem reams_needed : 
  (stories_per_week * pages_per_story * weeks_in_year + novel_pages_per_year) / pages_per_sheet / sheets_per_ream = 9 := by
  sorry


end NUMINAMATH_CALUDE_reams_needed_l2299_229966


namespace NUMINAMATH_CALUDE_henrys_brothers_ages_sum_l2299_229993

theorem henrys_brothers_ages_sum :
  ∀ (a b c : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    a < 10 ∧ b < 10 ∧ c < 10 →
    a > 0 ∧ b > 0 ∧ c > 0 →
    a = 2 * b →
    c * c = b →
    a + b + c = 14 :=
by sorry

end NUMINAMATH_CALUDE_henrys_brothers_ages_sum_l2299_229993


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2299_229994

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2299_229994


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2299_229927

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2299_229927


namespace NUMINAMATH_CALUDE_sophia_saves_two_dimes_l2299_229961

/-- Represents the number of pennies in a dime -/
def pennies_per_dime : ℕ := 10

/-- Represents the number of days Sophia saves -/
def saving_days : ℕ := 20

/-- Represents the number of pennies Sophia saves per day -/
def pennies_per_day : ℕ := 1

/-- Calculates the total number of pennies saved -/
def total_pennies : ℕ := saving_days * pennies_per_day

/-- Theorem: Sophia saves 2 dimes in total -/
theorem sophia_saves_two_dimes : 
  total_pennies / pennies_per_dime = 2 := by sorry

end NUMINAMATH_CALUDE_sophia_saves_two_dimes_l2299_229961


namespace NUMINAMATH_CALUDE_cit_beaver_difference_l2299_229958

/-- A Beaver-number is a positive 5-digit integer whose digit sum is divisible by 17. -/
def is_beaver_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ (n.digits 10).sum % 17 = 0

/-- A Beaver-pair is a pair of consecutive Beaver-numbers. -/
def is_beaver_pair (m n : ℕ) : Prop :=
  is_beaver_number m ∧ is_beaver_number n ∧ n = m + 1

/-- An MIT Beaver is the smaller number in a Beaver-pair. -/
def is_mit_beaver (m : ℕ) : Prop :=
  ∃ n, is_beaver_pair m n

/-- A CIT Beaver is the larger number in a Beaver-pair. -/
def is_cit_beaver (n : ℕ) : Prop :=
  ∃ m, is_beaver_pair m n

/-- The theorem stating the difference between the maximum and minimum CIT Beaver numbers. -/
theorem cit_beaver_difference : 
  ∃ max min : ℕ, 
    is_cit_beaver max ∧ 
    is_cit_beaver min ∧ 
    (∀ n, is_cit_beaver n → n ≤ max) ∧ 
    (∀ n, is_cit_beaver n → min ≤ n) ∧ 
    max - min = 79200 :=
sorry

end NUMINAMATH_CALUDE_cit_beaver_difference_l2299_229958


namespace NUMINAMATH_CALUDE_sqrt_37_range_l2299_229948

theorem sqrt_37_range : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_37_range_l2299_229948


namespace NUMINAMATH_CALUDE_inverse_as_polynomial_of_N_l2299_229921

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 0; 2, -4]

theorem inverse_as_polynomial_of_N :
  let c : ℚ := 1 / 36
  let d : ℚ := -1 / 12
  N⁻¹ = c • (N ^ 2) + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by sorry

end NUMINAMATH_CALUDE_inverse_as_polynomial_of_N_l2299_229921


namespace NUMINAMATH_CALUDE_remainder_theorem_l2299_229986

theorem remainder_theorem : (1225^3 * 1227^4 * 1229^5) % 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2299_229986


namespace NUMINAMATH_CALUDE_tan_alpha_and_expression_l2299_229967

theorem tan_alpha_and_expression (α : ℝ) 
  (h : Real.tan (π / 4 + α) = 2) : 
  Real.tan α = 1 / 3 ∧ 
  (2 * Real.sin α ^ 2 + Real.sin (2 * α)) / (1 + Real.tan α) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_and_expression_l2299_229967


namespace NUMINAMATH_CALUDE_prob_red_then_blue_is_one_thirteenth_l2299_229914

def total_marbles : ℕ := 4 + 3 + 6

def red_marbles : ℕ := 4
def blue_marbles : ℕ := 3
def yellow_marbles : ℕ := 6

def prob_red_then_blue : ℚ := (red_marbles : ℚ) / total_marbles * blue_marbles / (total_marbles - 1)

theorem prob_red_then_blue_is_one_thirteenth :
  prob_red_then_blue = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_blue_is_one_thirteenth_l2299_229914


namespace NUMINAMATH_CALUDE_product_of_roots_l2299_229920

theorem product_of_roots (a b c : ℂ) : 
  (3 * a^3 - 9 * a^2 + a - 7 = 0) ∧ 
  (3 * b^3 - 9 * b^2 + b - 7 = 0) ∧ 
  (3 * c^3 - 9 * c^2 + c - 7 = 0) →
  a * b * c = 7/3 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l2299_229920


namespace NUMINAMATH_CALUDE_cake_cutting_l2299_229981

/-- Represents a square cake -/
structure Cake where
  side : ℕ
  pieces : ℕ

/-- The maximum number of pieces obtainable with a single straight cut -/
def max_pieces_single_cut (c : Cake) : ℕ := sorry

/-- The minimum number of straight cuts required to intersect all original pieces -/
def min_cuts_all_pieces (c : Cake) : ℕ := sorry

/-- The theorem statement -/
theorem cake_cutting (c : Cake) 
  (h1 : c.side = 4) 
  (h2 : c.pieces = 16) : 
  max_pieces_single_cut c = 23 ∧ min_cuts_all_pieces c = 3 := by sorry

end NUMINAMATH_CALUDE_cake_cutting_l2299_229981


namespace NUMINAMATH_CALUDE_rectangle_side_relation_l2299_229970

/-- For a rectangle with adjacent sides x and y, and area 30, y = 30/x -/
theorem rectangle_side_relation (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  x * y = 30 → y = 30 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_relation_l2299_229970


namespace NUMINAMATH_CALUDE_quadratic_roots_l2299_229915

theorem quadratic_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : 2 * a^2 + a * a + b = 0 ∧ 2 * b^2 + a * b + b = 0) : 
  a = 1/2 ∧ b = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2299_229915


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2299_229959

theorem right_triangle_hypotenuse : 
  ∀ (short_leg long_leg hypotenuse : ℝ),
  short_leg > 0 →
  long_leg = 3 * short_leg - 1 →
  (1 / 2) * short_leg * long_leg = 90 →
  hypotenuse^2 = short_leg^2 + long_leg^2 →
  hypotenuse = Real.sqrt 593 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2299_229959


namespace NUMINAMATH_CALUDE_min_sum_squared_eccentricities_l2299_229930

/-- Given an ellipse and a hyperbola sharing the same foci, with one of their
    intersection points P forming an angle ∠F₁PF₂ = 60°, and their respective
    eccentricities e₁ and e₂, the minimum value of e₁² + e₂² is 1 + √3/2. -/
theorem min_sum_squared_eccentricities (e₁ e₂ : ℝ) 
  (h_ellipse : e₁ ∈ Set.Ioo 0 1)
  (h_hyperbola : e₂ > 1)
  (h_shared_foci : True)  -- Represents the condition that the ellipse and hyperbola share foci
  (h_intersection : True)  -- Represents the condition that P is an intersection point
  (h_angle : True)  -- Represents the condition that ∠F₁PF₂ = 60°
  : (∀ ε₁ ε₂, ε₁ ∈ Set.Ioo 0 1 → ε₂ > 1 → ε₁^2 + ε₂^2 ≥ 1 + Real.sqrt 3 / 2) ∧ 
    (∃ ε₁ ε₂, ε₁ ∈ Set.Ioo 0 1 ∧ ε₂ > 1 ∧ ε₁^2 + ε₂^2 = 1 + Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squared_eccentricities_l2299_229930


namespace NUMINAMATH_CALUDE_bugs_and_flowers_l2299_229990

theorem bugs_and_flowers (total_bugs : ℝ) (total_flowers : ℝ) 
  (h1 : total_bugs = 2.0) 
  (h2 : total_flowers = 3.0) : 
  total_flowers / total_bugs = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_bugs_and_flowers_l2299_229990


namespace NUMINAMATH_CALUDE_distributive_property_negative_three_l2299_229965

theorem distributive_property_negative_three (a b : ℝ) : -3 * (-a - b) = 3 * a + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_negative_three_l2299_229965


namespace NUMINAMATH_CALUDE_solve_for_y_l2299_229987

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2299_229987


namespace NUMINAMATH_CALUDE_student_combinations_l2299_229941

/-- The number of possible combinations when n people each have 2 choices -/
def combinations (n : ℕ) : ℕ := 2^n

/-- There are 5 students -/
def num_students : ℕ := 5

/-- Theorem: The number of combinations for 5 students with 2 choices each is 32 -/
theorem student_combinations : combinations num_students = 32 := by
  sorry

end NUMINAMATH_CALUDE_student_combinations_l2299_229941


namespace NUMINAMATH_CALUDE_last_four_digits_5_2011_l2299_229977

-- Define a function to get the last four digits of a number
def lastFourDigits (n : ℕ) : ℕ := n % 10000

-- Define the cycle length of the last four digits of powers of 5
def cycleLengthPowersOf5 : ℕ := 4

-- Theorem statement
theorem last_four_digits_5_2011 :
  lastFourDigits (5^2011) = lastFourDigits (5^7) :=
by
  sorry

#eval lastFourDigits (5^7)  -- This should output 8125

end NUMINAMATH_CALUDE_last_four_digits_5_2011_l2299_229977


namespace NUMINAMATH_CALUDE_mary_seashells_count_l2299_229947

/-- The number of seashells Sam found -/
def sam_seashells : ℕ := 18

/-- The total number of seashells Sam and Mary found together -/
def total_seashells : ℕ := 65

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := total_seashells - sam_seashells

theorem mary_seashells_count : mary_seashells = 47 := by sorry

end NUMINAMATH_CALUDE_mary_seashells_count_l2299_229947


namespace NUMINAMATH_CALUDE_root_transformation_l2299_229901

theorem root_transformation {a₁ a₂ a₃ b c₁ c₂ c₃ : ℝ} 
  (h_distinct : c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃)
  (h_roots : ∀ x : ℝ, (x - a₁) * (x - a₂) * (x - a₃) = b ↔ x = c₁ ∨ x = c₂ ∨ x = c₃) :
  ∀ x : ℝ, (x + c₁) * (x + c₂) * (x + c₃) = b ↔ x = -a₁ ∨ x = -a₂ ∨ x = -a₃ := by
sorry

end NUMINAMATH_CALUDE_root_transformation_l2299_229901


namespace NUMINAMATH_CALUDE_unbounded_function_identity_l2299_229974

/-- A function f: ℤ → ℤ is unbounded if for any integer N, there exists an x such that |f(x)| > N -/
def Unbounded (f : ℤ → ℤ) : Prop :=
  ∀ N : ℤ, ∃ x : ℤ, |f x| > N

/-- The main theorem: if f is unbounded and satisfies the given condition, then f(x) = x for all x -/
theorem unbounded_function_identity
  (f : ℤ → ℤ)
  (h_unbounded : Unbounded f)
  (h_condition : ∀ x y : ℤ, (f (f x - y)) ∣ (x - f y)) :
  ∀ x : ℤ, f x = x :=
sorry

end NUMINAMATH_CALUDE_unbounded_function_identity_l2299_229974
