import Mathlib

namespace NUMINAMATH_CALUDE_triangle_side_length_range_l2684_268470

theorem triangle_side_length_range (a : ℝ) : 
  (∃ (s₁ s₂ s₃ : ℝ), s₁ = 3*a - 1 ∧ s₂ = 4*a + 1 ∧ s₃ = 12 - a ∧ 
    s₁ + s₂ > s₃ ∧ s₁ + s₃ > s₂ ∧ s₂ + s₃ > s₁) ↔ 
  (3/2 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_range_l2684_268470


namespace NUMINAMATH_CALUDE_max_container_volume_l2684_268456

/-- The volume of an open-top container made from a rectangular sheet metal --/
def container_volume (l w h : ℝ) : ℝ := h * (l - 2*h) * (w - 2*h)

/-- The theorem stating the maximum volume of the container --/
theorem max_container_volume :
  let l : ℝ := 90
  let w : ℝ := 48
  ∃ (h : ℝ), 
    (h > 0) ∧ 
    (h < w/2) ∧ 
    (h < l/2) ∧
    (∀ (x : ℝ), x > 0 → x < w/2 → x < l/2 → container_volume l w h ≥ container_volume l w x) ∧
    (container_volume l w h = 16848) ∧
    (h = 6) :=
sorry

end NUMINAMATH_CALUDE_max_container_volume_l2684_268456


namespace NUMINAMATH_CALUDE_four_digit_count_l2684_268476

/-- The count of four-digit numbers -/
def count_four_digit_numbers : ℕ := 9000

/-- The smallest four-digit number -/
def min_four_digit : ℕ := 1000

/-- The largest four-digit number -/
def max_four_digit : ℕ := 9999

/-- Theorem: The count of integers from the smallest to the largest four-digit number
    (inclusive) is equal to the count of four-digit numbers. -/
theorem four_digit_count :
  (Finset.range (max_four_digit - min_four_digit + 1)).card = count_four_digit_numbers := by
  sorry

end NUMINAMATH_CALUDE_four_digit_count_l2684_268476


namespace NUMINAMATH_CALUDE_calculator_probability_l2684_268481

/-- Represents a 7-segment calculator display --/
def SegmentDisplay := Fin 7 → Bool

/-- The probability of a segment being illuminated --/
def segmentProbability : ℚ := 1/2

/-- The total number of possible displays --/
def totalDisplays : ℕ := 2^7

/-- The number of valid digit displays (0-9) --/
def validDigitDisplays : ℕ := 10

/-- The probability of displaying a valid digit --/
def validDigitProbability : ℚ := validDigitDisplays / totalDisplays

theorem calculator_probability (a b : ℕ) (h : validDigitProbability = a / b) :
  9 * a + 2 * b = 173 := by
  sorry

end NUMINAMATH_CALUDE_calculator_probability_l2684_268481


namespace NUMINAMATH_CALUDE_existence_of_special_set_l2684_268469

theorem existence_of_special_set :
  ∃ (S : Finset ℕ), 
    Finset.card S = 1998 ∧ 
    ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → (a * b) % ((a - b) ^ 2) = 0 :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l2684_268469


namespace NUMINAMATH_CALUDE_correct_num_cages_l2684_268415

/-- The number of bird cages in a pet store -/
def num_cages : ℕ := 4

/-- The number of birds in each cage -/
def birds_per_cage : ℕ := 10

/-- The total number of birds in the pet store -/
def total_birds : ℕ := 40

/-- Theorem: The number of bird cages is correct given the conditions -/
theorem correct_num_cages : num_cages * birds_per_cage = total_birds := by
  sorry

end NUMINAMATH_CALUDE_correct_num_cages_l2684_268415


namespace NUMINAMATH_CALUDE_sum_of_999_and_999_l2684_268454

theorem sum_of_999_and_999 : 999 + 999 = 1998 := by sorry

end NUMINAMATH_CALUDE_sum_of_999_and_999_l2684_268454


namespace NUMINAMATH_CALUDE_die_visible_combinations_l2684_268480

/-- A die is represented as a cube with 6 faces, 12 edges, and 8 vertices -/
structure Die :=
  (faces : Fin 6)
  (edges : Fin 12)
  (vertices : Fin 8)

/-- The number of visible faces from a point in space can be 1, 2, or 3 -/
inductive VisibleFaces
  | one
  | two
  | three

/-- The number of combinations for each type of view -/
def combinationsForView (v : VisibleFaces) : ℕ :=
  match v with
  | VisibleFaces.one => 6    -- One face visible: 6 possibilities
  | VisibleFaces.two => 12   -- Two faces visible: 12 possibilities
  | VisibleFaces.three => 8  -- Three faces visible: 8 possibilities

/-- The total number of different visible face combinations -/
def totalCombinations (d : Die) : ℕ :=
  (combinationsForView VisibleFaces.one) +
  (combinationsForView VisibleFaces.two) +
  (combinationsForView VisibleFaces.three)

theorem die_visible_combinations (d : Die) :
  totalCombinations d = 26 := by
  sorry

end NUMINAMATH_CALUDE_die_visible_combinations_l2684_268480


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l2684_268422

theorem sqrt_difference_equality (p q : ℝ) 
  (h1 : p > 0) (h2 : 0 ≤ q) (h3 : q ≤ 5 * p) : 
  Real.sqrt (10 * p + 2 * Real.sqrt (25 * p^2 - q^2)) - 
  Real.sqrt (10 * p - 2 * Real.sqrt (25 * p^2 - q^2)) = 
  2 * Real.sqrt (5 * p - q) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l2684_268422


namespace NUMINAMATH_CALUDE_largest_n_for_rational_sum_of_roots_l2684_268499

theorem largest_n_for_rational_sum_of_roots : 
  ∀ n : ℕ, n > 2501 → ¬(∃ (q : ℚ), q = Real.sqrt (n - 100) + Real.sqrt (n + 100)) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_rational_sum_of_roots_l2684_268499


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l2684_268445

theorem smallest_prime_dividing_sum : ∃ p : ℕ, 
  Nat.Prime p ∧ p > 5 ∧ p ∣ (2^14 + 3^15) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (2^14 + 3^15) → q ≥ p := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l2684_268445


namespace NUMINAMATH_CALUDE_petrol_price_reduction_l2684_268478

def original_price : ℝ := 4.444444444444445

theorem petrol_price_reduction (budget : ℝ) (additional_gallons : ℝ) 
  (h1 : budget = 200) 
  (h2 : additional_gallons = 5) :
  let reduced_price := budget / (budget / original_price + additional_gallons)
  (original_price - reduced_price) / original_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_petrol_price_reduction_l2684_268478


namespace NUMINAMATH_CALUDE_radio_cost_price_l2684_268431

theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 1430 →
  loss_percentage = 20.555555555555554 →
  selling_price = cost_price * (1 - loss_percentage / 100) →
  cost_price = 1800 := by
sorry

end NUMINAMATH_CALUDE_radio_cost_price_l2684_268431


namespace NUMINAMATH_CALUDE_second_rectangle_perimeter_l2684_268436

theorem second_rectangle_perimeter (a b : ℝ) : 
  (a + 3) * (b + 3) - a * b = 48 →
  2 * ((a + 3) + (b + 3)) = 38 := by
sorry

end NUMINAMATH_CALUDE_second_rectangle_perimeter_l2684_268436


namespace NUMINAMATH_CALUDE_election_vote_count_l2684_268408

/-- Represents the number of votes in an election round -/
structure ElectionRound where
  totalVotes : ℕ
  firstCandidateVotes : ℕ
  secondCandidateVotes : ℕ

/-- Represents a two-round election -/
structure TwoRoundElection where
  firstRound : ElectionRound
  secondRound : ElectionRound

theorem election_vote_count (election : TwoRoundElection) : election.firstRound.totalVotes = 48000 :=
  by
  have h1 : election.firstRound.firstCandidateVotes = election.firstRound.secondCandidateVotes :=
    sorry
  have h2 : election.secondRound.totalVotes = election.firstRound.totalVotes := sorry
  have h3 : election.secondRound.firstCandidateVotes =
    election.firstRound.firstCandidateVotes - 16000 := sorry
  have h4 : election.secondRound.secondCandidateVotes =
    election.firstRound.secondCandidateVotes + 16000 := sorry
  have h5 : election.secondRound.secondCandidateVotes =
    5 * election.secondRound.firstCandidateVotes := sorry
  sorry

end NUMINAMATH_CALUDE_election_vote_count_l2684_268408


namespace NUMINAMATH_CALUDE_square_field_area_l2684_268472

theorem square_field_area (side_length : ℝ) (h : side_length = 25) : 
  side_length * side_length = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l2684_268472


namespace NUMINAMATH_CALUDE_lcm_of_135_and_195_l2684_268433

theorem lcm_of_135_and_195 : Nat.lcm 135 195 = 1755 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_135_and_195_l2684_268433


namespace NUMINAMATH_CALUDE_sum_first_8_even_numbers_l2684_268471

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

theorem sum_first_8_even_numbers :
  (first_n_even_numbers 8).sum = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_8_even_numbers_l2684_268471


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_equals_five_halves_l2684_268423

theorem sqrt_x_div_sqrt_y_equals_five_halves (x y : ℝ) 
  (h : (1/3)^2 + (1/4)^2 / ((1/5)^2 + (1/6)^2) = 25*x / (73*y)) : 
  Real.sqrt x / Real.sqrt y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_equals_five_halves_l2684_268423


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2684_268421

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≤ 4}

-- Define set A
def A : Set ℝ := {x | |x - 1| ≤ 1}

-- Theorem statement
theorem complement_of_A_in_U :
  (U \ A) = {x : ℝ | -2 ≤ x ∧ x < 0} :=
sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2684_268421


namespace NUMINAMATH_CALUDE_max_value_of_f_l2684_268410

-- Define the function f
def f (x a : ℝ) : ℝ := -x^2 + 4*x + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 1) := by
sorry


end NUMINAMATH_CALUDE_max_value_of_f_l2684_268410


namespace NUMINAMATH_CALUDE_unique_number_l2684_268414

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  Even n ∧ 
  n % 11 = 0 ∧ 
  is_perfect_cube (digit_product n) ∧
  n = 88 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_l2684_268414


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2684_268401

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem perpendicular_line_equation (given_line : Line) (point : Point) : 
  given_line.slope = 2/3 ∧ given_line.intercept = -2 ∧ point.x = 4 ∧ point.y = 2 →
  ∃ (result_line : Line), 
    result_line.slope = -3/2 ∧ 
    result_line.intercept = 8 ∧
    pointOnLine point result_line ∧
    perpendicular given_line result_line :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2684_268401


namespace NUMINAMATH_CALUDE_queen_middle_school_teachers_l2684_268449

structure School where
  students : ℕ
  classes_per_student : ℕ
  classes_per_teacher : ℕ
  students_per_class : ℕ

def number_of_teachers (school : School) : ℕ :=
  (school.students * school.classes_per_student) / (school.students_per_class * school.classes_per_teacher)

theorem queen_middle_school_teachers :
  let queen_middle : School := {
    students := 1500,
    classes_per_student := 5,
    classes_per_teacher := 5,
    students_per_class := 25
  }
  number_of_teachers queen_middle = 60 := by
  sorry

end NUMINAMATH_CALUDE_queen_middle_school_teachers_l2684_268449


namespace NUMINAMATH_CALUDE_fourth_quadrant_m_range_l2684_268482

theorem fourth_quadrant_m_range (m : ℝ) :
  let z : ℂ := (1 + m * Complex.I) / (1 + Complex.I)
  (z.re > 0 ∧ z.im < 0) → -1 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_quadrant_m_range_l2684_268482


namespace NUMINAMATH_CALUDE_candle_burn_theorem_l2684_268475

theorem candle_burn_theorem (t : ℝ) (h : t > 0) :
  let rate_second : ℝ := (3 / 5) / t
  let rate_third : ℝ := (4 / 7) / t
  let time_second_remaining : ℝ := (2 / 5) / rate_second
  let third_burned_while_second_finishes : ℝ := time_second_remaining * rate_third
  (3 / 7) - third_burned_while_second_finishes = 1 / 21 := by
sorry

end NUMINAMATH_CALUDE_candle_burn_theorem_l2684_268475


namespace NUMINAMATH_CALUDE_nested_expression_equals_one_l2684_268465

def nested_expression : ℤ :=
  (3 * (3 * (3 * (3 * (3 - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1)

theorem nested_expression_equals_one : nested_expression = 1 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_equals_one_l2684_268465


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2684_268468

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → 1007 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2684_268468


namespace NUMINAMATH_CALUDE_xy_max_value_l2684_268405

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 2) :
  ∃ (m : ℝ), m = 1/2 ∧ ∀ z, z = x*y → z ≤ m :=
sorry

end NUMINAMATH_CALUDE_xy_max_value_l2684_268405


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2684_268495

theorem expression_simplification_and_evaluation (a : ℝ) 
  (h1 : a ≠ -1) (h2 : a ≠ 2) :
  let original_expr := (a - 3*a/(a+1)) / ((a^2 - 4*a + 4)/(a+1))
  let simplified_expr := a / (a-2)
  original_expr = simplified_expr ∧ 
  (a = -2 → original_expr = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2684_268495


namespace NUMINAMATH_CALUDE_gold_alloy_composition_l2684_268426

/-- Proves that adding 62.5 ounces of pure gold to a 100-ounce alloy that is 35% gold
    will result in a new alloy that is 60% gold. -/
theorem gold_alloy_composition (original_weight : ℝ) (original_gold_percentage : ℝ) 
    (added_gold : ℝ) (new_gold_percentage : ℝ) : 
  original_weight = 100 →
  original_gold_percentage = 0.35 →
  added_gold = 62.5 →
  new_gold_percentage = 0.60 →
  (original_weight * original_gold_percentage + added_gold) / (original_weight + added_gold) = new_gold_percentage :=
by
  sorry

#eval (100 * 0.35 + 62.5) / (100 + 62.5)

end NUMINAMATH_CALUDE_gold_alloy_composition_l2684_268426


namespace NUMINAMATH_CALUDE_factor_expression_l2684_268486

theorem factor_expression (x : ℝ) : 3*x*(x+3) + 2*(x+3) = (x+3)*(3*x+2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2684_268486


namespace NUMINAMATH_CALUDE_field_trip_vans_l2684_268440

/-- The number of vans needed for a field trip --/
def vans_needed (students : ℕ) (adults : ℕ) (van_capacity : ℕ) : ℕ :=
  ((students + adults + van_capacity - 1) / van_capacity : ℕ)

/-- Theorem: For 33 students, 9 adults, and vans with capacity 7, 6 vans are needed --/
theorem field_trip_vans : vans_needed 33 9 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_vans_l2684_268440


namespace NUMINAMATH_CALUDE_prop_A_prop_B_prop_C_false_prop_D_main_theorem_l2684_268428

-- Define the basic concepts
def Line : Type := sorry
def Plane : Type := sorry
def Point : Type := sorry

-- Define the relationships
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def intersect (l : Line) (p : Plane) : Prop := sorry
def onPlane (p : Point) (pl : Plane) : Prop := sorry
def angle (l1 l2 : Line) : ℝ := sorry
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Proposition A
theorem prop_A (l1 l2 l3 : Line) :
  parallel l1 l2 → perpendicular l3 l1 → perpendicular l3 l2 := sorry

-- Proposition B
theorem prop_B (a b c : Line) (θ : ℝ) :
  parallel a b → ¬(intersect c a) → ¬(intersect c b) → angle c a = θ → angle c b = θ := sorry

-- Proposition C (false statement)
theorem prop_C_false :
  ∃ (p1 p2 p3 p4 : Point) (pl : Plane), 
    ¬(onPlane p1 pl ∧ onPlane p2 pl ∧ onPlane p3 pl ∧ onPlane p4 pl) →
    ¬(collinear p1 p2 p3 ∨ collinear p1 p2 p4 ∨ collinear p1 p3 p4 ∨ collinear p2 p3 p4) := sorry

-- Proposition D
theorem prop_D (a : Line) (α : Plane) (P : Point) :
  parallel a α → onPlane P α → ∃ (l : Line), parallel l a ∧ onPlane P l ∧ (∀ (Q : Point), onPlane Q l → onPlane Q α) := sorry

-- Main theorem stating that A, B, and D are true while C is false
theorem main_theorem : 
  (∀ l1 l2 l3, parallel l1 l2 → perpendicular l3 l1 → perpendicular l3 l2) ∧
  (∀ a b c θ, parallel a b → ¬(intersect c a) → ¬(intersect c b) → angle c a = θ → angle c b = θ) ∧
  (∃ p1 p2 p3 p4 pl, ¬(onPlane p1 pl ∧ onPlane p2 pl ∧ onPlane p3 pl ∧ onPlane p4 pl) ∧
    (collinear p1 p2 p3 ∨ collinear p1 p2 p4 ∨ collinear p1 p3 p4 ∨ collinear p2 p3 p4)) ∧
  (∀ a α P, parallel a α → onPlane P α → 
    ∃ l, parallel l a ∧ onPlane P l ∧ (∀ Q, onPlane Q l → onPlane Q α)) := sorry

end NUMINAMATH_CALUDE_prop_A_prop_B_prop_C_false_prop_D_main_theorem_l2684_268428


namespace NUMINAMATH_CALUDE_cylindrical_block_volume_l2684_268403

/-- Represents a cylindrical iron block -/
structure CylindricalBlock where
  height : ℝ
  volume : ℝ

/-- Represents a frustum-shaped iron block -/
structure FrustumBlock where
  height : ℝ
  base_radius : ℝ

/-- Represents a container with a cylindrical and a frustum-shaped block -/
structure Container where
  cylindrical_block : CylindricalBlock
  frustum_block : FrustumBlock

/-- Theorem stating the volume of the cylindrical block in the container -/
theorem cylindrical_block_volume (container : Container) 
  (h1 : container.cylindrical_block.height = 3)
  (h2 : container.frustum_block.height = 3)
  (h3 : container.frustum_block.base_radius = container.frustum_block.base_radius) :
  container.cylindrical_block.volume = 15.42 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_block_volume_l2684_268403


namespace NUMINAMATH_CALUDE_fraction_problem_l2684_268425

theorem fraction_problem (f : ℚ) : f * 76 = 76 - 19 → f = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2684_268425


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l2684_268447

theorem simplify_complex_fraction (b : ℝ) 
  (h1 : b ≠ 1/2) (h2 : b ≠ 1) : 
  1 - 2 / (1 + b / (1 - 2*b)) = (3*b - 1) / (1 - b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l2684_268447


namespace NUMINAMATH_CALUDE_negation_of_existence_l2684_268459

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2684_268459


namespace NUMINAMATH_CALUDE_apartment_complexes_count_l2684_268424

/-- The maximum number of apartment complexes that can be built on a rectangular piece of land -/
def max_apartment_complexes (land_width land_length complex_side : ℕ) : ℕ :=
  (land_width / complex_side) * (land_length / complex_side)

/-- Theorem: Given the specified land dimensions and apartment complex size, 
    the maximum number of apartment complexes that can be built is 140 -/
theorem apartment_complexes_count :
  max_apartment_complexes 262 185 18 = 140 := by
  sorry

end NUMINAMATH_CALUDE_apartment_complexes_count_l2684_268424


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_for_given_problem_l2684_268462

/-- Calculates the systematic sampling interval for a given population and sample size -/
def systematicSamplingInterval (population : ℕ) (sampleSize : ℕ) : ℕ :=
  (population - (population % sampleSize)) / sampleSize

theorem systematic_sampling_interval_for_given_problem :
  systematicSamplingInterval 1203 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_for_given_problem_l2684_268462


namespace NUMINAMATH_CALUDE_equation_solution_l2684_268488

theorem equation_solution (x : ℝ) (h : 5 / (4 + 1/x) = 1) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2684_268488


namespace NUMINAMATH_CALUDE_marks_lawyer_hourly_rate_l2684_268434

/-- Calculates the lawyer's hourly rate for Mark's speeding ticket case -/
theorem marks_lawyer_hourly_rate 
  (base_fine : ℕ) 
  (speed_fine_rate : ℕ) 
  (marks_speed : ℕ) 
  (speed_limit : ℕ) 
  (court_costs : ℕ) 
  (lawyer_hours : ℕ) 
  (total_owed : ℕ) 
  (h1 : base_fine = 50)
  (h2 : speed_fine_rate = 2)
  (h3 : marks_speed = 75)
  (h4 : speed_limit = 30)
  (h5 : court_costs = 300)
  (h6 : lawyer_hours = 3)
  (h7 : total_owed = 820) :
  (total_owed - (2 * (base_fine + speed_fine_rate * (marks_speed - speed_limit)) + court_costs)) / lawyer_hours = 80 := by
  sorry

end NUMINAMATH_CALUDE_marks_lawyer_hourly_rate_l2684_268434


namespace NUMINAMATH_CALUDE_triangle_problem_l2684_268464

theorem triangle_problem (DC CB : ℝ) (h1 : DC = 12) (h2 : CB = 9)
  (AD : ℝ) (h3 : AD > 0)
  (AB : ℝ) (h4 : AB = (1/3) * AD)
  (ED : ℝ) (h5 : ED = (3/4) * AD) :
  ∃ FC : ℝ, FC = 14.625 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2684_268464


namespace NUMINAMATH_CALUDE_ten_player_modified_round_robin_l2684_268485

/-- The number of matches in a modified round-robin tournament --/
def modifiedRoundRobinMatches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 - 1

/-- Theorem: In a round-robin tournament with 10 players, where each player
    plays every other player once, but the match between the first and
    second players is not held, the total number of matches is 44. --/
theorem ten_player_modified_round_robin :
  modifiedRoundRobinMatches 10 = 44 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_modified_round_robin_l2684_268485


namespace NUMINAMATH_CALUDE_clock_cost_price_l2684_268467

theorem clock_cost_price (total_clocks : ℕ) (sold_at_10_percent : ℕ) (sold_at_20_percent : ℕ) 
  (uniform_profit_difference : ℝ) :
  total_clocks = 90 →
  sold_at_10_percent = 40 →
  sold_at_20_percent = 50 →
  uniform_profit_difference = 40 →
  ∃ (cost_price : ℝ),
    cost_price = 80 ∧
    (sold_at_10_percent : ℝ) * cost_price * 1.1 + 
    (sold_at_20_percent : ℝ) * cost_price * 1.2 - 
    (total_clocks : ℝ) * cost_price * 1.15 = uniform_profit_difference :=
by sorry

end NUMINAMATH_CALUDE_clock_cost_price_l2684_268467


namespace NUMINAMATH_CALUDE_greatest_possible_N_l2684_268427

theorem greatest_possible_N : ∃ (N : ℕ), 
  (N = 5) ∧ 
  (∀ k : ℕ, k > 5 → ¬∃ (S : Finset ℕ), 
    (Finset.card S = 2^k - 1) ∧ 
    (∀ x y : ℕ, x ∈ S → y ∈ S → x ≠ y → x ≠ y) ∧
    (Finset.sum S id = 2014)) ∧
  (∃ (S : Finset ℕ), 
    (Finset.card S = 2^5 - 1) ∧ 
    (∀ x y : ℕ, x ∈ S → y ∈ S → x ≠ y → x ≠ y) ∧
    (Finset.sum S id = 2014)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_N_l2684_268427


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l2684_268474

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem intersection_complement_equals_set : N ∩ (U \ M) = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l2684_268474


namespace NUMINAMATH_CALUDE_minimum_bottles_needed_l2684_268479

def medium_bottle_capacity : ℕ := 120
def jumbo_bottle_capacity : ℕ := 2000

theorem minimum_bottles_needed : 
  (Nat.ceil (jumbo_bottle_capacity / medium_bottle_capacity : ℚ) : ℕ) = 17 := by
  sorry

end NUMINAMATH_CALUDE_minimum_bottles_needed_l2684_268479


namespace NUMINAMATH_CALUDE_residue_mod_16_l2684_268489

theorem residue_mod_16 : 260 * 18 - 21 * 8 + 4 ≡ 4 [ZMOD 16] := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_16_l2684_268489


namespace NUMINAMATH_CALUDE_integer_solutions_system_l2684_268498

theorem integer_solutions_system :
  ∀ x y z : ℤ,
  (x^2 - y^2 = z ∧ 3*x*y + (x-y)*z = z^2) →
  ((x = 2 ∧ y = 1 ∧ z = 3) ∨
   (x = 1 ∧ y = 2 ∧ z = -3) ∨
   (x = 1 ∧ y = 0 ∧ z = 1) ∨
   (x = 0 ∧ y = 1 ∧ z = -1) ∨
   (x = 0 ∧ y = 0 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_system_l2684_268498


namespace NUMINAMATH_CALUDE_orange_distribution_difference_l2684_268497

/-- Calculates the difference in oranges per student before and after removing bad oranges -/
theorem orange_distribution_difference (total_oranges : ℕ) (num_students : ℕ) (bad_oranges : ℕ) :
  total_oranges > bad_oranges →
  num_students > 0 →
  (total_oranges : ℝ) / num_students - (total_oranges - bad_oranges : ℝ) / num_students = 2.6 :=
by sorry

end NUMINAMATH_CALUDE_orange_distribution_difference_l2684_268497


namespace NUMINAMATH_CALUDE_F_is_even_l2684_268446

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function F
def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  |f x| + f (|x|)

-- Theorem statement
theorem F_is_even (f : ℝ → ℝ) (h : is_odd_function f) :
  ∀ x, F f (-x) = F f x :=
by sorry

end NUMINAMATH_CALUDE_F_is_even_l2684_268446


namespace NUMINAMATH_CALUDE_apples_in_basket_proof_l2684_268491

/-- Given a total number of apples and the capacity of each box,
    calculate the number of apples left for the basket. -/
def applesInBasket (totalApples : ℕ) (applesPerBox : ℕ) : ℕ :=
  totalApples - (totalApples / applesPerBox) * applesPerBox

/-- Prove that with 138 total apples and boxes of 18 apples each,
    there will be 12 apples left for the basket. -/
theorem apples_in_basket_proof :
  applesInBasket 138 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_proof_l2684_268491


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_sum_l2684_268483

theorem cubic_polynomial_root_sum (f : ℝ → ℝ) (r₁ r₂ r₃ : ℝ) :
  (∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  ((f (1/2) + f (-1/2)) / f 0 = 1003) →
  (1 / (r₁ * r₂) + 1 / (r₂ * r₃) + 1 / (r₃ * r₁) = 2002) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_sum_l2684_268483


namespace NUMINAMATH_CALUDE_average_headcount_rounded_l2684_268430

def fall_headcount_03_04 : ℕ := 11500
def fall_headcount_04_05 : ℕ := 11300
def fall_headcount_05_06 : ℕ := 11400

def average_headcount : ℚ := (fall_headcount_03_04 + fall_headcount_04_05 + fall_headcount_05_06) / 3

theorem average_headcount_rounded : 
  round average_headcount = 11400 := by sorry

end NUMINAMATH_CALUDE_average_headcount_rounded_l2684_268430


namespace NUMINAMATH_CALUDE_geom_seq_306th_term_l2684_268402

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem geom_seq_306th_term (a₁ a₂ : ℝ) (h1 : a₁ = 7) (h2 : a₂ = -7) :
  geometric_sequence a₁ (a₂ / a₁) 306 = -7 :=
by sorry

end NUMINAMATH_CALUDE_geom_seq_306th_term_l2684_268402


namespace NUMINAMATH_CALUDE_johns_age_satisfies_condition_l2684_268452

/-- Represents John's current age in years -/
def johnsCurrentAge : ℕ := 18

/-- Represents the condition that five years ago, John's age was half of what it will be in 8 years -/
def ageCondition (age : ℕ) : Prop :=
  age - 5 = (age + 8) / 2

/-- Theorem stating that John's current age satisfies the given condition -/
theorem johns_age_satisfies_condition : ageCondition johnsCurrentAge := by
  sorry

#check johns_age_satisfies_condition

end NUMINAMATH_CALUDE_johns_age_satisfies_condition_l2684_268452


namespace NUMINAMATH_CALUDE_least_number_divisibility_l2684_268466

theorem least_number_divisibility (x : ℕ) : x = 171011 ↔ 
  (∀ y : ℕ, y < x → ¬(41 ∣ (1076 + y) ∧ 59 ∣ (1076 + y) ∧ 67 ∣ (1076 + y))) ∧
  (41 ∣ (1076 + x) ∧ 59 ∣ (1076 + x) ∧ 67 ∣ (1076 + x)) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l2684_268466


namespace NUMINAMATH_CALUDE_rectangle_area_is_48_l2684_268441

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle PQRS with points U and V on its diagonal -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point
  U : Point
  V : Point

/-- Given conditions for the rectangle problem -/
def rectangle_conditions (rect : Rectangle) : Prop :=
  -- PQRS is a rectangle (implied by other conditions)
  -- PQ is parallel to RS (implied by rectangle property)
  (rect.P.x - rect.Q.x = rect.R.x - rect.S.x) ∧ 
  (rect.P.y - rect.Q.y = rect.R.y - rect.S.y) ∧ 
  -- PQ = RS
  ((rect.P.x - rect.Q.x)^2 + (rect.P.y - rect.Q.y)^2 = 
   (rect.R.x - rect.S.x)^2 + (rect.R.y - rect.S.y)^2) ∧
  -- U and V lie on diagonal PS
  ((rect.U.x - rect.P.x) * (rect.S.y - rect.P.y) = 
   (rect.U.y - rect.P.y) * (rect.S.x - rect.P.x)) ∧
  ((rect.V.x - rect.P.x) * (rect.S.y - rect.P.y) = 
   (rect.V.y - rect.P.y) * (rect.S.x - rect.P.x)) ∧
  -- U is between P and V
  ((rect.U.x - rect.P.x) * (rect.V.x - rect.U.x) ≥ 0) ∧
  ((rect.U.y - rect.P.y) * (rect.V.y - rect.U.y) ≥ 0) ∧
  -- Angle PUV = 90°
  ((rect.P.x - rect.U.x) * (rect.V.x - rect.U.x) + 
   (rect.P.y - rect.U.y) * (rect.V.y - rect.U.y) = 0) ∧
  -- Angle QVR = 90°
  ((rect.Q.x - rect.V.x) * (rect.R.x - rect.V.x) + 
   (rect.Q.y - rect.V.y) * (rect.R.y - rect.V.y) = 0) ∧
  -- PU = 4
  ((rect.P.x - rect.U.x)^2 + (rect.P.y - rect.U.y)^2 = 16) ∧
  -- UV = 2
  ((rect.U.x - rect.V.x)^2 + (rect.U.y - rect.V.y)^2 = 4) ∧
  -- VS = 6
  ((rect.V.x - rect.S.x)^2 + (rect.V.y - rect.S.y)^2 = 36)

/-- The area of a rectangle -/
def rectangle_area (rect : Rectangle) : ℝ :=
  abs ((rect.P.x - rect.Q.x) * (rect.Q.y - rect.R.y) - 
       (rect.P.y - rect.Q.y) * (rect.Q.x - rect.R.x))

/-- Theorem stating that the area of the rectangle is 48 -/
theorem rectangle_area_is_48 (rect : Rectangle) : 
  rectangle_conditions rect → rectangle_area rect = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_48_l2684_268441


namespace NUMINAMATH_CALUDE_triangle_area_reduction_l2684_268437

theorem triangle_area_reduction (b h m : ℝ) (hb : b > 0) (hh : h > 0) (hm : m ≥ 0) :
  ∃ x : ℝ, 
    (1/2 : ℝ) * (b - x) * (h + m) = (1/2 : ℝ) * ((1/2 : ℝ) * b * h) ∧
    x = b * (2 * m + h) / (2 * (h + m)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_reduction_l2684_268437


namespace NUMINAMATH_CALUDE_lcm_210_297_l2684_268420

theorem lcm_210_297 : Nat.lcm 210 297 = 20790 := by
  sorry

end NUMINAMATH_CALUDE_lcm_210_297_l2684_268420


namespace NUMINAMATH_CALUDE_latest_time_82_degrees_l2684_268407

-- Define the temperature function
def T (t : ℝ) : ℝ := -t^2 + 12*t + 55

-- Define the derivative of the temperature function
def T' (t : ℝ) : ℝ := -2*t + 12

-- Theorem statement
theorem latest_time_82_degrees (t : ℝ) :
  (T t = 82) ∧ (T' t < 0) →
  t = 6 + (3 * Real.sqrt 28) / 2 :=
by sorry

end NUMINAMATH_CALUDE_latest_time_82_degrees_l2684_268407


namespace NUMINAMATH_CALUDE_extremum_condition_l2684_268494

/-- The function f(x) = ax³ + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- A function has an extremum if it has either a local maximum or a local minimum -/
def has_extremum (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, (∀ x : ℝ, f x ≤ f x₀) ∨ (∀ x : ℝ, f x ≥ f x₀)

/-- The necessary and sufficient condition for f(x) = ax³ + x + 1 to have an extremum -/
theorem extremum_condition (a : ℝ) :
  has_extremum (f a) ↔ a < 0 := by sorry

end NUMINAMATH_CALUDE_extremum_condition_l2684_268494


namespace NUMINAMATH_CALUDE_total_length_eleven_segments_l2684_268413

/-- The total length of 11 congruent segments -/
def total_length (segment_length : ℝ) (num_segments : ℕ) : ℝ :=
  segment_length * (num_segments : ℝ)

/-- Theorem: The total length of 11 congruent segments of 7 cm each is 77 cm -/
theorem total_length_eleven_segments :
  total_length 7 11 = 77 := by sorry

end NUMINAMATH_CALUDE_total_length_eleven_segments_l2684_268413


namespace NUMINAMATH_CALUDE_test_questions_l2684_268484

theorem test_questions (sections : ℕ) (correct_answers : ℕ) (lower_bound : ℚ) (upper_bound : ℚ) :
  sections = 5 →
  correct_answers = 32 →
  lower_bound = 70/100 →
  upper_bound = 77/100 →
  ∃ (total_questions : ℕ),
    (total_questions % sections = 0) ∧
    (lower_bound < (correct_answers : ℚ) / total_questions) ∧
    ((correct_answers : ℚ) / total_questions < upper_bound) ∧
    (total_questions = 45) := by
  sorry

#check test_questions

end NUMINAMATH_CALUDE_test_questions_l2684_268484


namespace NUMINAMATH_CALUDE_f_properties_l2684_268448

noncomputable section

variable (a : ℝ)
variable (h₁ : a > 0)
variable (h₂ : a ≠ 1)

def f (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

theorem f_properties :
  (∀ x, f a x = -f a (-x)) ∧ 
  (StrictMono (f a)) ∧
  (∀ m, 1 < m → m < Real.sqrt 2 → f a (1 - m) + f a (1 - m^2) < 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2684_268448


namespace NUMINAMATH_CALUDE_average_coins_per_day_l2684_268450

def coins_collected (day : ℕ) : ℕ :=
  if day = 0 then 0
  else if day < 7 then 10 * day
  else 10 * 7 + 20

def total_coins : ℕ := (List.range 7).map (λ i => coins_collected (i + 1)) |>.sum

theorem average_coins_per_day :
  (total_coins : ℚ) / 7 = 300 / 7 := by sorry

end NUMINAMATH_CALUDE_average_coins_per_day_l2684_268450


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2684_268443

theorem rationalize_denominator : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2684_268443


namespace NUMINAMATH_CALUDE_parabola_translation_l2684_268438

/-- Represents a vertical translation of a parabola -/
def verticalTranslation (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x ↦ f x + k

/-- The original parabola function -/
def originalParabola : ℝ → ℝ := λ x ↦ x^2

theorem parabola_translation :
  verticalTranslation originalParabola 4 = λ x ↦ x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l2684_268438


namespace NUMINAMATH_CALUDE_lcm_gcd_product_24_60_l2684_268417

theorem lcm_gcd_product_24_60 : Nat.lcm 24 60 * Nat.gcd 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_24_60_l2684_268417


namespace NUMINAMATH_CALUDE_circle_a_range_l2684_268429

-- Define the equation of the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + a = 0

-- Define what it means for an equation to represent a circle
def is_circle (a : ℝ) : Prop :=
  ∃ h k r, ∀ x y, circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

-- Theorem statement
theorem circle_a_range (a : ℝ) :
  is_circle a → a < 5 :=
sorry

end NUMINAMATH_CALUDE_circle_a_range_l2684_268429


namespace NUMINAMATH_CALUDE_earnings_difference_l2684_268435

/-- Represents the delivery areas --/
inductive DeliveryArea
  | A
  | B
  | C

/-- Represents a delivery worker --/
structure DeliveryWorker where
  name : String
  deliveries : DeliveryArea → Nat

/-- Get the fee for a specific delivery area --/
def areaFee (area : DeliveryArea) : Nat :=
  match area with
  | DeliveryArea.A => 100
  | DeliveryArea.B => 125
  | DeliveryArea.C => 150

/-- Calculate the total earnings for a worker --/
def totalEarnings (worker : DeliveryWorker) : Nat :=
  (worker.deliveries DeliveryArea.A * areaFee DeliveryArea.A) +
  (worker.deliveries DeliveryArea.B * areaFee DeliveryArea.B) +
  (worker.deliveries DeliveryArea.C * areaFee DeliveryArea.C)

/-- Oula's delivery data --/
def oula : DeliveryWorker :=
  { name := "Oula"
    deliveries := fun
      | DeliveryArea.A => 48
      | DeliveryArea.B => 32
      | DeliveryArea.C => 16 }

/-- Tona's delivery data --/
def tona : DeliveryWorker :=
  { name := "Tona"
    deliveries := fun
      | DeliveryArea.A => 27
      | DeliveryArea.B => 18
      | DeliveryArea.C => 9 }

/-- The main theorem to prove --/
theorem earnings_difference : totalEarnings oula - totalEarnings tona = 4900 := by
  sorry


end NUMINAMATH_CALUDE_earnings_difference_l2684_268435


namespace NUMINAMATH_CALUDE_square_area_ratio_l2684_268458

theorem square_area_ratio (y : ℝ) (y_pos : y > 0) : 
  (3 * y)^2 / (12 * y)^2 = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2684_268458


namespace NUMINAMATH_CALUDE_unique_paths_equal_binomial_coefficient_l2684_268416

/-- The number of rows in the grid -/
def n : ℕ := 6

/-- The number of columns in the grid -/
def m : ℕ := 6

/-- The total number of steps required to reach the destination -/
def total_steps : ℕ := n + m

/-- The number of ways to choose n right moves out of total_steps moves -/
def num_paths : ℕ := Nat.choose total_steps n

/-- Theorem stating that the number of unique paths from A to B is equal to C(12,6) -/
theorem unique_paths_equal_binomial_coefficient : 
  num_paths = 924 := by sorry

end NUMINAMATH_CALUDE_unique_paths_equal_binomial_coefficient_l2684_268416


namespace NUMINAMATH_CALUDE_opposite_of_one_over_23_l2684_268492

theorem opposite_of_one_over_23 : 
  -(1 / 23) = -1 / 23 := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_over_23_l2684_268492


namespace NUMINAMATH_CALUDE_circle_C_is_correct_l2684_268490

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 6)^2 = 4

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - 2*y + 2 = 0

-- Define points A and B
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem circle_C_is_correct :
  (∀ x y : ℝ, circle_C x y → tangent_line x y → False) ∧ 
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 := by
  sorry

end NUMINAMATH_CALUDE_circle_C_is_correct_l2684_268490


namespace NUMINAMATH_CALUDE_no_line_exists_l2684_268496

-- Define the points A and Q
def A : ℝ × ℝ := (8, 0)
def Q : ℝ × ℝ := (-1, 0)

-- Define the curve y² = -4x
def curve (x y : ℝ) : Prop := y^2 = -4*x

-- Define a line passing through A
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 8)

-- Define the dot product of vectors QM and QN
def dot_product_QM_QN (M N : ℝ × ℝ) : ℝ :=
  (M.1 - Q.1) * (N.1 - Q.1) + (M.2 - Q.2) * (N.2 - Q.2)

-- The main theorem
theorem no_line_exists : ¬∃ (k : ℝ) (M N : ℝ × ℝ),
  M ≠ N ∧
  curve M.1 M.2 ∧
  curve N.1 N.2 ∧
  line_through_A k M.1 M.2 ∧
  line_through_A k N.1 N.2 ∧
  dot_product_QM_QN M N = 97 :=
sorry

end NUMINAMATH_CALUDE_no_line_exists_l2684_268496


namespace NUMINAMATH_CALUDE_even_function_property_l2684_268460

/-- A function f is even on an interval [-a, a] -/
def IsEvenOn (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-a) a, f x = f (-x)

theorem even_function_property
  (f : ℝ → ℝ) (h_even : IsEvenOn f 6) (h_gt : f 3 > f 1) :
  f (-1) < f 3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l2684_268460


namespace NUMINAMATH_CALUDE_distinct_triangles_in_3x2_grid_l2684_268406

/-- Represents a grid of dots -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Calculates the total number of dots in the grid -/
def Grid.totalDots (g : Grid) : Nat :=
  g.rows * g.cols

/-- Calculates the number of collinear groups in the grid -/
def Grid.collinearGroups (g : Grid) : Nat :=
  g.rows + g.cols

/-- Theorem: In a 3x2 grid, the number of distinct triangles is 15 -/
theorem distinct_triangles_in_3x2_grid :
  let g : Grid := { rows := 3, cols := 2 }
  let totalCombinations := Nat.choose (g.totalDots) 3
  let validTriangles := totalCombinations - g.collinearGroups
  validTriangles = 15 := by sorry


end NUMINAMATH_CALUDE_distinct_triangles_in_3x2_grid_l2684_268406


namespace NUMINAMATH_CALUDE_mets_fans_count_l2684_268455

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  redsox : ℕ

/-- The total number of baseball fans in the town -/
def total_fans : ℕ := 360

/-- Theorem stating that given the conditions, there are 96 NY Mets fans -/
theorem mets_fans_count (fc : FanCounts) : 
  fc.yankees + fc.mets + fc.redsox = total_fans →
  3 * fc.mets = 2 * fc.yankees →
  4 * fc.redsox = 5 * fc.mets →
  fc.mets = 96 := by
  sorry


end NUMINAMATH_CALUDE_mets_fans_count_l2684_268455


namespace NUMINAMATH_CALUDE_eighth_of_two_to_forty_l2684_268442

theorem eighth_of_two_to_forty (x : ℤ) : (1 / 8 : ℚ) * (2 ^ 40 : ℚ) = (2 : ℚ) ^ x → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_eighth_of_two_to_forty_l2684_268442


namespace NUMINAMATH_CALUDE_min_value_theorem_l2684_268451

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 4) :
  (1 / x + 4 / y) ≥ 9 / 4 ∧ 
  (1 / x + 4 / y = 9 / 4 ↔ y = 8 / 3 ∧ x = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2684_268451


namespace NUMINAMATH_CALUDE_function_domain_range_implies_b_equals_two_l2684_268412

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 4

-- Define the theorem
theorem function_domain_range_implies_b_equals_two :
  ∀ b : ℝ,
  (∀ x ∈ Set.Icc 2 (2*b), f x ∈ Set.Icc 2 (2*b)) ∧
  (∀ y ∈ Set.Icc 2 (2*b), ∃ x ∈ Set.Icc 2 (2*b), f x = y) →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_range_implies_b_equals_two_l2684_268412


namespace NUMINAMATH_CALUDE_kim_earrings_proof_l2684_268409

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_earring : ℕ := 9

/-- The number of pairs of earrings Kim brings on the first day -/
def first_day_earrings : ℕ := 3

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The number of days the gumballs last -/
def days_gumballs_last : ℕ := 42

theorem kim_earrings_proof :
  (first_day_earrings * gumballs_per_earring + 
   2 * first_day_earrings * gumballs_per_earring + 
   (2 * first_day_earrings - 1) * gumballs_per_earring) = 
  (gumballs_eaten_per_day * days_gumballs_last) := by
  sorry

end NUMINAMATH_CALUDE_kim_earrings_proof_l2684_268409


namespace NUMINAMATH_CALUDE_only_B_on_x_axis_l2684_268461

def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (3, 0)
def point_C : ℝ × ℝ := (0, -1)
def point_D : ℝ × ℝ := (-5, 6)

def is_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

theorem only_B_on_x_axis :
  ¬(is_on_x_axis point_A) ∧
  is_on_x_axis point_B ∧
  ¬(is_on_x_axis point_C) ∧
  ¬(is_on_x_axis point_D) :=
by sorry

end NUMINAMATH_CALUDE_only_B_on_x_axis_l2684_268461


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_999973_l2684_268432

theorem sum_of_prime_factors_999973 :
  ∃ (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    999973 = p * q * r ∧
    p + q + r = 171 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_999973_l2684_268432


namespace NUMINAMATH_CALUDE_tree_planting_event_girls_count_l2684_268419

theorem tree_planting_event_girls_count (boys : ℕ) (difference : ℕ) (total_percentage : ℚ) (partial_count : ℕ) 
  (h1 : boys = 600)
  (h2 : difference = 400)
  (h3 : total_percentage = 60 / 100)
  (h4 : partial_count = 960) : 
  ∃ (girls : ℕ), girls = 1000 ∧ girls > boys := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_event_girls_count_l2684_268419


namespace NUMINAMATH_CALUDE_max_stamps_proof_l2684_268487

/-- The price of a single stamp in cents -/
def stamp_price : ℕ := 50

/-- The discount rate applied when buying more than 100 stamps -/
def discount_rate : ℚ := 1/10

/-- The threshold number of stamps for applying the discount -/
def discount_threshold : ℕ := 100

/-- The total amount available in cents -/
def total_amount : ℕ := 10000

/-- The maximum number of stamps that can be purchased -/
def max_stamps : ℕ := 200

theorem max_stamps_proof :
  (∀ n : ℕ, n ≤ max_stamps → n * stamp_price ≤ total_amount) ∧
  (∀ n : ℕ, n > max_stamps → 
    (if n > discount_threshold 
     then n * stamp_price * (1 - discount_rate)
     else n * stamp_price) > total_amount) :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_proof_l2684_268487


namespace NUMINAMATH_CALUDE_simplify_expression_l2684_268457

theorem simplify_expression (x : ℝ) : 2 * x + 1 - (x + 1) = x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2684_268457


namespace NUMINAMATH_CALUDE_max_m_value_l2684_268439

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the inequality condition
def inequality_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x ≤ 1 → det (x + 1) x m (x - 1) ≥ -2

-- Theorem statement
theorem max_m_value :
  ∃ m : ℝ, inequality_condition m ∧ ∀ m' : ℝ, inequality_condition m' → m' ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l2684_268439


namespace NUMINAMATH_CALUDE_square_difference_plus_constant_problem_solution_l2684_268411

theorem square_difference_plus_constant (a b c : ℤ) :
  a ^ 2 - b ^ 2 + c = (a + b) * (a - b) + c := by sorry

theorem problem_solution :
  632 ^ 2 - 568 ^ 2 + 100 = 76900 := by sorry

end NUMINAMATH_CALUDE_square_difference_plus_constant_problem_solution_l2684_268411


namespace NUMINAMATH_CALUDE_mary_picked_12kg_l2684_268453

/-- Given three people picking chestnuts, prove that one person picked 12 kg. -/
theorem mary_picked_12kg (peter lucy mary : ℕ) : 
  mary = 2 * peter →  -- Mary picked twice as much as Peter
  lucy = peter + 2 →  -- Lucy picked 2 kg more than Peter
  peter + mary + lucy = 26 →  -- Total amount picked is 26 kg
  mary = 12 := by
sorry

end NUMINAMATH_CALUDE_mary_picked_12kg_l2684_268453


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2684_268477

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + 2 * x^2 + 15) - (x^6 + 4 * x^5 - 2 * x^4 + x^3 - 3 * x^2 + 18) =
  x^6 - x^5 + 3 * x^4 - x^3 + 5 * x^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2684_268477


namespace NUMINAMATH_CALUDE_malcolm_lights_theorem_l2684_268404

/-- The number of white lights Malcolm had initially --/
def initial_white_lights : ℕ := 59

/-- The number of red lights Malcolm bought --/
def red_lights : ℕ := 12

/-- The number of blue lights Malcolm bought --/
def blue_lights : ℕ := red_lights * 3

/-- The number of green lights Malcolm bought --/
def green_lights : ℕ := 6

/-- The number of colored lights Malcolm still needs to buy --/
def remaining_lights : ℕ := 5

/-- Theorem stating that the initial number of white lights is equal to 
    the sum of all colored lights bought and still to be bought --/
theorem malcolm_lights_theorem : 
  initial_white_lights = red_lights + blue_lights + green_lights + remaining_lights :=
by sorry

end NUMINAMATH_CALUDE_malcolm_lights_theorem_l2684_268404


namespace NUMINAMATH_CALUDE_square_root_of_x_plus_y_l2684_268400

theorem square_root_of_x_plus_y (x y : ℝ) :
  (x - 1)^(1/3) = 1 →
  ((2 * y + 2)^(1/2) : ℝ) = 4 →
  (x + y)^(1/2) = 3 ∨ (x + y)^(1/2) = -3 :=
by sorry

end NUMINAMATH_CALUDE_square_root_of_x_plus_y_l2684_268400


namespace NUMINAMATH_CALUDE_unique_prime_sum_diff_l2684_268493

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem unique_prime_sum_diff :
  ∃! p : ℕ, is_prime p ∧ 
    (∃ a b : ℕ, is_prime a ∧ is_prime b ∧ p = a + b) ∧
    (∃ c d : ℕ, is_prime c ∧ is_prime d ∧ p = c - d) :=
by
  use 5
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_diff_l2684_268493


namespace NUMINAMATH_CALUDE_difference_of_squares_l2684_268444

theorem difference_of_squares (m : ℝ) : m^2 - 9 = (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2684_268444


namespace NUMINAMATH_CALUDE_pebbles_distribution_l2684_268473

/-- The number of pebbles in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of pebbles Janice had -/
def janice_dozens : ℕ := 3

/-- The total number of pebbles Janice had -/
def total_pebbles : ℕ := janice_dozens * dozen

/-- The number of friends who received pebbles -/
def num_friends : ℕ := 9

/-- The number of pebbles each friend received -/
def pebbles_per_friend : ℕ := total_pebbles / num_friends

theorem pebbles_distribution :
  pebbles_per_friend = 4 :=
sorry

end NUMINAMATH_CALUDE_pebbles_distribution_l2684_268473


namespace NUMINAMATH_CALUDE_sum_x_y_equals_twenty_l2684_268463

theorem sum_x_y_equals_twenty (x y : ℝ) (h : (x + 1 + (y - 1)) / 2 = 10) : x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_twenty_l2684_268463


namespace NUMINAMATH_CALUDE_notebook_pen_cost_ratio_l2684_268418

theorem notebook_pen_cost_ratio : 
  let pen_cost : ℚ := 3/2  -- $1.50 as a rational number
  let notebooks_cost : ℚ := 18  -- Total cost of 4 notebooks
  let notebooks_count : ℕ := 4  -- Number of notebooks
  let notebook_cost : ℚ := notebooks_cost / notebooks_count  -- Cost of one notebook
  (notebook_cost / pen_cost) = 3 := by sorry

end NUMINAMATH_CALUDE_notebook_pen_cost_ratio_l2684_268418
