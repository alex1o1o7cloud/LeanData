import Mathlib

namespace NUMINAMATH_CALUDE_min_value_problem_l1579_157958

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2023) + (y + 1/x) * (y + 1/x - 2023) ≥ -2048113 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l1579_157958


namespace NUMINAMATH_CALUDE_networking_event_handshakes_l1579_157987

/-- Represents a group of people at a networking event -/
structure NetworkingEvent where
  total_people : Nat
  partner_pairs : Nat
  handshakes_per_person : Nat

/-- Calculates the total number of handshakes at a networking event -/
def total_handshakes (event : NetworkingEvent) : Nat :=
  event.total_people * event.handshakes_per_person / 2

/-- Theorem: The number of handshakes at the specific networking event is 60 -/
theorem networking_event_handshakes :
  ∃ (event : NetworkingEvent),
    event.total_people = 12 ∧
    event.partner_pairs = 6 ∧
    event.handshakes_per_person = 10 ∧
    total_handshakes event = 60 := by
  sorry

end NUMINAMATH_CALUDE_networking_event_handshakes_l1579_157987


namespace NUMINAMATH_CALUDE_point_coordinates_l1579_157900

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

/-- Distance from a point to the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- Predicate for a point being in the first quadrant -/
def inFirstQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y > 0

/-- Theorem: If a point M is in the first quadrant, its distance to the x-axis is 3,
    and its distance to the y-axis is 2, then its coordinates are (2, 3) -/
theorem point_coordinates (M : Point) 
  (h1 : inFirstQuadrant M) 
  (h2 : distToXAxis M = 3) 
  (h3 : distToYAxis M = 2) : 
  M.x = 2 ∧ M.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1579_157900


namespace NUMINAMATH_CALUDE_min_value_F_range_of_m_l1579_157956

noncomputable section

def f (x : ℝ) := x * Real.exp x
def g (x : ℝ) := (1/2) * x^2 + x
def F (x : ℝ) := f x + g x

-- Part 1
theorem min_value_F :
  ∃ (x_min : ℝ), ∀ (x : ℝ), F x_min ≤ F x ∧ F x_min = -1 - 1/Real.exp 1 :=
sorry

-- Part 2
theorem range_of_m (m : ℝ) :
  (∀ (x₁ x₂ : ℝ), -1 ≤ x₂ ∧ x₂ < x₁ →
    m * (f x₁ - f x₂) > g x₁ - g x₂) ↔ m ≥ Real.exp 1 :=
sorry

end

end NUMINAMATH_CALUDE_min_value_F_range_of_m_l1579_157956


namespace NUMINAMATH_CALUDE_windows_per_floor_l1579_157907

theorem windows_per_floor (floors : ℕ) (payment_per_window : ℚ) 
  (deduction_per_3days : ℚ) (days_taken : ℕ) (final_payment : ℚ) :
  floors = 3 →
  payment_per_window = 2 →
  deduction_per_3days = 1 →
  days_taken = 6 →
  final_payment = 16 →
  ∃ (windows_per_floor : ℕ), 
    windows_per_floor = 3 ∧
    (floors * windows_per_floor * payment_per_window - 
      (days_taken / 3 : ℚ) * deduction_per_3days = final_payment) :=
by
  sorry

end NUMINAMATH_CALUDE_windows_per_floor_l1579_157907


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l1579_157968

theorem unique_positive_integer_solution : ∃! (x : ℕ), x > 0 ∧ (3 * x)^2 - x = 2010 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l1579_157968


namespace NUMINAMATH_CALUDE_jimmy_drinks_eight_times_per_day_l1579_157919

/-- The number of times Jimmy drinks water per day -/
def times_per_day : ℕ :=
  let ounces_per_drink : ℚ := 8
  let gallons_for_five_days : ℚ := 5/2
  let ounces_per_gallon : ℚ := 1 / 0.0078125
  let days : ℕ := 5
  let total_ounces : ℚ := gallons_for_five_days * ounces_per_gallon
  let ounces_per_day : ℚ := total_ounces / days
  (ounces_per_day / ounces_per_drink).num.toNat

theorem jimmy_drinks_eight_times_per_day : times_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_drinks_eight_times_per_day_l1579_157919


namespace NUMINAMATH_CALUDE_inequality_proof_l1579_157949

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c = 1) : 
  (2*a*b + b*c + c*a + c^2/2 ≤ 1/2) ∧ 
  ((a^2 + c^2)/b + (b^2 + a^2)/c + (c^2 + b^2)/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1579_157949


namespace NUMINAMATH_CALUDE_existence_of_prime_and_integers_l1579_157986

theorem existence_of_prime_and_integers (k : ℕ+) : 
  ∃ (p : ℕ) (a : Fin (k+3) → ℕ), 
    Prime p ∧ 
    (∀ i : Fin (k+3), 1 ≤ a i ∧ a i < p) ∧
    (∀ i j : Fin (k+3), i ≠ j → a i ≠ a j) ∧
    (∀ i : Fin k, p ∣ (a i * a (i+1) * a (i+2) * a (i+3) - i)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_and_integers_l1579_157986


namespace NUMINAMATH_CALUDE_glove_selection_ways_l1579_157954

/-- The number of different pairs of gloves -/
def num_pairs : ℕ := 6

/-- The number of gloves to be selected -/
def num_selected : ℕ := 4

/-- The number of matching pairs in the selection -/
def num_matching_pairs : ℕ := 1

/-- The total number of ways to select the gloves -/
def total_ways : ℕ := 240

theorem glove_selection_ways :
  (num_pairs : ℕ) = 6 →
  (num_selected : ℕ) = 4 →
  (num_matching_pairs : ℕ) = 1 →
  (total_ways : ℕ) = 240 := by
  sorry

end NUMINAMATH_CALUDE_glove_selection_ways_l1579_157954


namespace NUMINAMATH_CALUDE_molecular_weight_AlI3_correct_l1579_157993

/-- The molecular weight of AlI3 in grams per mole -/
def molecular_weight_AlI3 : ℝ := 408

/-- The number of moles given in the problem -/
def num_moles : ℝ := 8

/-- The total weight of the given number of moles in grams -/
def total_weight : ℝ := 3264

/-- Theorem stating that the molecular weight of AlI3 is correct -/
theorem molecular_weight_AlI3_correct : 
  molecular_weight_AlI3 = total_weight / num_moles :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_AlI3_correct_l1579_157993


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1579_157931

/-- The area of the union of a square with side length 10 and a circle with radius 10 
    centered at one of the square's vertices is equal to 100 + 75π. -/
theorem square_circle_union_area : 
  let square_side : ℝ := 10
  let circle_radius : ℝ := 10
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  let overlap_area := (1 / 4 : ℝ) * circle_area
  square_area + circle_area - overlap_area = 100 + 75 * π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1579_157931


namespace NUMINAMATH_CALUDE_count_special_four_digit_numbers_l1579_157996

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (ℕ × ℕ × ℕ × ℕ)

/-- Checks if a FourDigitNumber is valid (between 1000 and 9999) -/
def is_valid (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

/-- Converts a pair of digits to a two-digit number -/
def to_two_digit (a b : ℕ) : ℕ := 10 * a + b

/-- Checks if three two-digit numbers form an increasing arithmetic sequence -/
def is_increasing_arithmetic_seq (ab bc cd : ℕ) : Prop :=
  ab < bc ∧ bc < cd ∧ bc - ab = cd - bc

/-- The main theorem to be proved -/
theorem count_special_four_digit_numbers :
  (∃ (S : Finset FourDigitNumber),
    (∀ n ∈ S, is_valid n ∧ 
      let (a, b, c, d) := n
      is_increasing_arithmetic_seq (to_two_digit a b) (to_two_digit b c) (to_two_digit c d)) ∧
    S.card = 17 ∧
    (∀ n : FourDigitNumber, 
      is_valid n ∧ 
      let (a, b, c, d) := n
      is_increasing_arithmetic_seq (to_two_digit a b) (to_two_digit b c) (to_two_digit c d) 
      → n ∈ S)) := by
  sorry

end NUMINAMATH_CALUDE_count_special_four_digit_numbers_l1579_157996


namespace NUMINAMATH_CALUDE_plane_division_by_lines_l1579_157924

/-- The number of regions created by n non-parallel lines in a plane --/
def num_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of infinite regions created by n non-parallel lines in a plane --/
def num_infinite_regions (n : ℕ) : ℕ := 2 * n

theorem plane_division_by_lines (n : ℕ) (h : n = 20) :
  num_regions n = 211 ∧ num_regions n - num_infinite_regions n = 171 :=
sorry

end NUMINAMATH_CALUDE_plane_division_by_lines_l1579_157924


namespace NUMINAMATH_CALUDE_opposite_reciprocal_abs_neg_seven_l1579_157921

theorem opposite_reciprocal_abs_neg_seven :
  -(1 / |(-7 : ℤ)|) = -((1 : ℚ) / 7) := by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_abs_neg_seven_l1579_157921


namespace NUMINAMATH_CALUDE_projection_implies_y_coordinate_l1579_157977

/-- Given vectors a and b, if the projection of b in the direction of a is -√2, then the y-coordinate of b is 4. -/
theorem projection_implies_y_coordinate (a b : ℝ × ℝ) :
  a = (1, -1) →
  b.1 = 2 →
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt ((a.1 ^ 2 + a.2 ^ 2) : ℝ) = -Real.sqrt 2 →
  b.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_projection_implies_y_coordinate_l1579_157977


namespace NUMINAMATH_CALUDE_larger_number_in_ratio_l1579_157904

theorem larger_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 2 / 3 →
  Nat.lcm a b = 120 →
  b = 72 := by
sorry

end NUMINAMATH_CALUDE_larger_number_in_ratio_l1579_157904


namespace NUMINAMATH_CALUDE_sum_of_squares_power_l1579_157928

theorem sum_of_squares_power (a p q : ℤ) (h : a = p^2 + q^2) :
  ∀ k : ℕ+, ∃ x y : ℤ, a^(k : ℕ) = x^2 + y^2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_power_l1579_157928


namespace NUMINAMATH_CALUDE_notebook_distribution_l1579_157941

theorem notebook_distribution (x : ℕ) : 
  (x > 0) → 
  ((x - 1) % 3 = 0) → 
  ((x + 2) % 4 = 0) → 
  ((x - 1) / 3 : ℚ) = ((x + 2) / 4 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_notebook_distribution_l1579_157941


namespace NUMINAMATH_CALUDE_number_equation_solution_l1579_157963

theorem number_equation_solution : ∃ x : ℝ, (3 * x - 1 = 2 * x) ∧ (x = 1) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1579_157963


namespace NUMINAMATH_CALUDE_max_value_of_a_l1579_157975

theorem max_value_of_a (a b c d : ℤ) 
  (h1 : a < 2*b) 
  (h2 : b < 3*c) 
  (h3 : c < 4*d) 
  (h4 : d < 100) : 
  a ≤ 2367 ∧ ∃ (a₀ b₀ c₀ d₀ : ℤ), 
    a₀ < 2*b₀ ∧ 
    b₀ < 3*c₀ ∧ 
    c₀ < 4*d₀ ∧ 
    d₀ < 100 ∧ 
    a₀ = 2367 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1579_157975


namespace NUMINAMATH_CALUDE_polyhedron_surface_area_l1579_157929

/-- Represents a polyhedron with three orthographic views --/
structure Polyhedron where
  front_view : Set (ℝ × ℝ)
  side_view : Set (ℝ × ℝ)
  top_view : Set (ℝ × ℝ)

/-- Calculates the surface area of a polyhedron --/
noncomputable def surface_area (p : Polyhedron) : ℝ := sorry

/-- Theorem stating that the surface area of the given polyhedron is 8 --/
theorem polyhedron_surface_area (p : Polyhedron) : surface_area p = 8 := by sorry

end NUMINAMATH_CALUDE_polyhedron_surface_area_l1579_157929


namespace NUMINAMATH_CALUDE_stratified_sample_small_supermarkets_l1579_157959

/-- Calculates the number of small supermarkets in a stratified sample -/
def smallSupermarketsInSample (totalSupermarkets : ℕ) (smallSupermarkets : ℕ) (sampleSize : ℕ) : ℕ :=
  (smallSupermarkets * sampleSize) / totalSupermarkets

theorem stratified_sample_small_supermarkets :
  smallSupermarketsInSample 3000 2100 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_small_supermarkets_l1579_157959


namespace NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_when_disjoint_l1579_157989

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_when_a_half :
  A (1/2) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

theorem range_of_a_when_disjoint (h1 : A a ≠ ∅) (h2 : A a ∩ B = ∅) :
  (-2 < a ∧ a ≤ 1/2) ∨ (a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_when_disjoint_l1579_157989


namespace NUMINAMATH_CALUDE_notepad_lasts_four_days_l1579_157922

/-- Represents the number of pieces of letter-size paper used --/
def letter_size_papers : ℕ := 5

/-- Represents the number of times each paper is folded --/
def folds : ℕ := 3

/-- Represents the number of notes written per day --/
def notes_per_day : ℕ := 10

/-- Calculates the number of note-size papers produced from one letter-size paper --/
def note_papers_per_letter_paper : ℕ := 2^folds

/-- Calculates the total number of note-size papers in a notepad --/
def total_note_papers : ℕ := letter_size_papers * note_papers_per_letter_paper

/-- Represents how long a notepad lasts in days --/
def notepad_duration : ℕ := total_note_papers / notes_per_day

theorem notepad_lasts_four_days : notepad_duration = 4 := by
  sorry

end NUMINAMATH_CALUDE_notepad_lasts_four_days_l1579_157922


namespace NUMINAMATH_CALUDE_yellow_bows_count_l1579_157916

theorem yellow_bows_count (total : ℕ) 
  (h_red : (total : ℚ) / 4 = total / 4)
  (h_blue : (total : ℚ) / 3 = total / 3)
  (h_green : (total : ℚ) / 6 = total / 6)
  (h_yellow : (total : ℚ) / 12 = total / 12)
  (h_white : (total : ℚ) - (total / 4 + total / 3 + total / 6 + total / 12) = 40) :
  (total : ℚ) / 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_yellow_bows_count_l1579_157916


namespace NUMINAMATH_CALUDE_equation_solutions_l1579_157972

theorem equation_solutions : 
  let f (x : ℝ) := 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 
                   1 / ((x - 4) * (x - 5)) + 1 / ((x - 5) * (x - 6))
  ∀ x : ℝ, f x = 1 / 12 ↔ x = 12 ∨ x = -4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1579_157972


namespace NUMINAMATH_CALUDE_product_sale_loss_l1579_157962

/-- Represents the pricing and sale of a product -/
def ProductSale (cost_price : ℝ) : Prop :=
  let initial_markup := 1.20
  let price_reduction := 0.80
  let sale_price := 96
  initial_markup * cost_price * price_reduction = sale_price ∧
  cost_price > sale_price ∧
  cost_price - sale_price = 4

/-- Theorem stating the loss in the product sale -/
theorem product_sale_loss :
  ∃ (cost_price : ℝ), ProductSale cost_price :=
sorry

end NUMINAMATH_CALUDE_product_sale_loss_l1579_157962


namespace NUMINAMATH_CALUDE_parallelogram_roots_l1579_157955

/-- The polynomial in question -/
def polynomial (b : ℝ) (z : ℂ) : ℂ :=
  z^4 - 8*z^3 + 13*b*z^2 - 5*(2*b^2 + 4*b - 4)*z + 4

/-- Predicate to check if four complex numbers form a parallelogram -/
def form_parallelogram (z₁ z₂ z₃ z₄ : ℂ) : Prop :=
  (z₁ + z₃ = z₂ + z₄) ∧ (z₁ - z₂ = z₄ - z₃)

/-- The main theorem -/
theorem parallelogram_roots (b : ℝ) : 
  (∃ z₁ z₂ z₃ z₄ : ℂ, 
    (polynomial b z₁ = 0) ∧ 
    (polynomial b z₂ = 0) ∧ 
    (polynomial b z₃ = 0) ∧ 
    (polynomial b z₄ = 0) ∧ 
    form_parallelogram z₁ z₂ z₃ z₄) ↔ 
  b = 1.5 := by sorry

end NUMINAMATH_CALUDE_parallelogram_roots_l1579_157955


namespace NUMINAMATH_CALUDE_two_thousand_nineteen_in_group_63_l1579_157988

/-- The last number in the nth group -/
def last_in_group (n : ℕ) : ℕ := n * (n + 1) / 2 + n

/-- The first number in the nth group -/
def first_in_group (n : ℕ) : ℕ := last_in_group (n - 1) + 1

/-- Predicate to check if a number is in the nth group -/
def in_group (x n : ℕ) : Prop :=
  first_in_group n ≤ x ∧ x ≤ last_in_group n

theorem two_thousand_nineteen_in_group_63 :
  in_group 2019 63 := by sorry

end NUMINAMATH_CALUDE_two_thousand_nineteen_in_group_63_l1579_157988


namespace NUMINAMATH_CALUDE_coin_toss_probability_l1579_157978

theorem coin_toss_probability : 
  let p_head : ℝ := 1/2  -- Probability of getting heads on a single toss
  let n : ℕ := 3  -- Number of tosses
  let p_all_tails : ℝ := (1 - p_head)^n  -- Probability of getting all tails
  1 - p_all_tails = 7/8 := by sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l1579_157978


namespace NUMINAMATH_CALUDE_two_digit_number_existence_l1579_157995

/-- Two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- First digit of a two-digit number -/
def firstDigit (n : ℕ) : ℕ := n / 10

/-- Second digit of a two-digit number -/
def secondDigit (n : ℕ) : ℕ := n % 10

/-- Sum of digits of a two-digit number -/
def digitSum (n : ℕ) : ℕ := firstDigit n + secondDigit n

/-- Absolute difference of digits of a two-digit number -/
def digitDiff (n : ℕ) : ℕ := Int.natAbs (firstDigit n - secondDigit n)

theorem two_digit_number_existence :
  ∃ (X Y : ℕ), 
    TwoDigitNumber X ∧ 
    TwoDigitNumber Y ∧ 
    X = 2 * Y ∧
    (firstDigit Y = digitSum X ∨ secondDigit Y = digitSum X) ∧
    (firstDigit Y = digitDiff X ∨ secondDigit Y = digitDiff X) ∧
    X = 34 ∧ 
    Y = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_existence_l1579_157995


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l1579_157935

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total : Nat
  sample_size : Nat
  interval : Nat
  elements : Finset Nat

/-- Checks if a number is in the systematic sample -/
def in_sample (n : Nat) (s : SystematicSample) : Prop :=
  n ∈ s.elements

theorem systematic_sample_theorem (s : SystematicSample) 
  (h_total : s.total = 52)
  (h_sample_size : s.sample_size = 4)
  (h_interval : s.interval = 13)
  (h_6 : in_sample 6 s)
  (h_32 : in_sample 32 s)
  (h_45 : in_sample 45 s) :
  in_sample 19 s := by
  sorry

#check systematic_sample_theorem

end NUMINAMATH_CALUDE_systematic_sample_theorem_l1579_157935


namespace NUMINAMATH_CALUDE_sum_of_powers_l1579_157947

theorem sum_of_powers (a b : ℝ) : 
  a + b = 1 →
  a^2 + b^2 = 3 →
  a^3 + b^3 = 4 →
  a^4 + b^4 = 7 →
  a^5 + b^5 = 11 →
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1579_157947


namespace NUMINAMATH_CALUDE_base_3_10201_equals_100_l1579_157936

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_3_10201_equals_100 :
  base_3_to_10 [1, 0, 2, 0, 1] = 100 := by
  sorry

end NUMINAMATH_CALUDE_base_3_10201_equals_100_l1579_157936


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l1579_157937

theorem rectangle_shorter_side (area perimeter : ℝ) (h_area : area = 104) (h_perimeter : perimeter = 42) :
  ∃ (length width : ℝ), 
    length * width = area ∧ 
    2 * (length + width) = perimeter ∧ 
    min length width = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l1579_157937


namespace NUMINAMATH_CALUDE_complex_multiplication_l1579_157991

theorem complex_multiplication : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 → 
  (2 + Complex.I) * (1 - 3 * Complex.I) = 5 - 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1579_157991


namespace NUMINAMATH_CALUDE_shopkeeper_theft_loss_l1579_157974

theorem shopkeeper_theft_loss (profit_percent : ℝ) (loss_percent : ℝ) : 
  profit_percent = 10 → loss_percent = 45 → 
  (loss_percent / 100) * (1 + profit_percent / 100) * 100 = 49.5 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_theft_loss_l1579_157974


namespace NUMINAMATH_CALUDE_yellow_ball_percentage_l1579_157950

/-- Given the number of yellow and brown balls, calculate the percentage of yellow balls -/
theorem yellow_ball_percentage (yellow_balls brown_balls : ℕ) : 
  yellow_balls = 27 → brown_balls = 33 → 
  (yellow_balls : ℚ) / (yellow_balls + brown_balls : ℚ) * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_percentage_l1579_157950


namespace NUMINAMATH_CALUDE_no_real_solutions_l1579_157911

theorem no_real_solutions : ∀ x : ℝ, ¬∃ y : ℝ, 
  (y = 3 * x - 1) ∧ (4 * y^2 + y + 3 = 3 * (8 * x^2 + 3 * y + 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1579_157911


namespace NUMINAMATH_CALUDE_base_9_to_10_3562_l1579_157976

def base_9_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

theorem base_9_to_10_3562 :
  base_9_to_10 [2, 6, 5, 3] = 2648 := by
  sorry

end NUMINAMATH_CALUDE_base_9_to_10_3562_l1579_157976


namespace NUMINAMATH_CALUDE_g_five_equals_248_l1579_157942

theorem g_five_equals_248 (g : ℤ → ℤ) 
  (h1 : g 1 > 1)
  (h2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y)
  (h3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1) :
  g 5 = 248 := by sorry

end NUMINAMATH_CALUDE_g_five_equals_248_l1579_157942


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l1579_157930

/-- A rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ :=
  4 * 3

theorem rectangular_prism_parallel_edges :
  ∀ (prism : RectangularPrism),
    prism.length = 4 ∧ prism.width = 3 ∧ prism.height = 2 →
    parallel_edge_pairs prism = 12 := by
  sorry

#check rectangular_prism_parallel_edges

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l1579_157930


namespace NUMINAMATH_CALUDE_task_completion_rate_l1579_157985

/-- Given two people A and B who can complete a task in x and y days respectively,
    this theorem proves that together they can complete a fraction of 1/x + 1/y of the task in one day. -/
theorem task_completion_rate (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 : ℝ) / x + (1 : ℝ) / y = (x + y) / (x * y) := by
  sorry


end NUMINAMATH_CALUDE_task_completion_rate_l1579_157985


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1579_157910

theorem polynomial_identity_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 1728 * x^4 + 64 = (a * x^3 + b * x^2 + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 416 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1579_157910


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1579_157920

theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 10
  let altitude : ℝ := s * Real.sqrt 3 / 2
  let area : ℝ := s * altitude / 2
  let perimeter : ℝ := 3 * s
  area / perimeter = 5 * Real.sqrt 3 / 6 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1579_157920


namespace NUMINAMATH_CALUDE_root_sum_square_l1579_157901

theorem root_sum_square (α β : ℝ) : 
  (α^2 + α - 2023 = 0) → 
  (β^2 + β - 2023 = 0) → 
  α^2 + 2*α + β = 2022 := by
sorry

end NUMINAMATH_CALUDE_root_sum_square_l1579_157901


namespace NUMINAMATH_CALUDE_equation_solutions_l1579_157998

theorem equation_solutions :
  (∀ x : ℝ, 4 * x * (2 * x - 1) = 3 * (2 * x - 1) → x = 1/2 ∨ x = 3/4) ∧
  (∀ x : ℝ, x^2 + 2*x - 2 = 0 → x = -1 + Real.sqrt 3 ∨ x = -1 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1579_157998


namespace NUMINAMATH_CALUDE_first_machine_rate_is_35_l1579_157973

/-- The number of copies the first machine makes per minute -/
def first_machine_rate : ℝ := sorry

/-- The number of copies the second machine makes per minute -/
def second_machine_rate : ℝ := 75

/-- The total number of copies both machines make in 30 minutes -/
def total_copies : ℝ := 3300

/-- The time period in minutes -/
def time_period : ℝ := 30

theorem first_machine_rate_is_35 :
  first_machine_rate = 35 :=
by
  have h1 : first_machine_rate * time_period + second_machine_rate * time_period = total_copies :=
    sorry
  sorry

#check first_machine_rate_is_35

end NUMINAMATH_CALUDE_first_machine_rate_is_35_l1579_157973


namespace NUMINAMATH_CALUDE_exists_term_between_zero_and_one_l1579_157965

/-- An infinite sequence satisfying a_{n+2} = |a_{n+1} - a_n| -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1) - a n|

/-- Theorem: For any special sequence, there exists a term between 0 and 1 -/
theorem exists_term_between_zero_and_one (a : ℕ → ℝ) (h : SpecialSequence a) :
    ∃ k : ℕ, 0 ≤ a k ∧ a k < 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_term_between_zero_and_one_l1579_157965


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1579_157915

open Real

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  (∀ φ, 0 < φ ∧ φ < π / 2 →
    3 * cos θ + 2 / sin θ + 2 * sqrt 2 * tan θ ≤ 3 * cos φ + 2 / sin φ + 2 * sqrt 2 * tan φ) ∧
  3 * cos θ + 2 / sin θ + 2 * sqrt 2 * tan θ = 7 * sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1579_157915


namespace NUMINAMATH_CALUDE_hotel_bill_problem_l1579_157966

theorem hotel_bill_problem (total_bill : ℕ) (equal_share : ℕ) (extra_payment : ℕ) (num_paying_80 : ℕ) :
  (num_paying_80 = 7) →
  (80 * num_paying_80 + 160 = total_bill) →
  (equal_share + 70 = 160) →
  (total_bill / equal_share = 8) :=
by
  sorry

#check hotel_bill_problem

end NUMINAMATH_CALUDE_hotel_bill_problem_l1579_157966


namespace NUMINAMATH_CALUDE_hexagon_area_is_32_l1579_157902

/-- A hexagon surrounded by four right triangles forming a rectangle -/
structure HexagonWithTriangles where
  -- Side length of the hexagon
  side_length : ℝ
  -- Height of each triangle
  triangle_height : ℝ
  -- The shape forms a rectangle
  is_rectangle : Bool
  -- There are four identical right triangles
  triangle_count : Nat
  -- The triangles are identical and right-angled
  triangles_identical_right : Bool

/-- The area of the hexagon given its structure -/
def hexagon_area (h : HexagonWithTriangles) : ℝ :=
  sorry

/-- Theorem stating the area of the hexagon is 32 square units -/
theorem hexagon_area_is_32 (h : HexagonWithTriangles) 
  (h_side : h.side_length = 2)
  (h_height : h.triangle_height = 4)
  (h_rect : h.is_rectangle = true)
  (h_count : h.triangle_count = 4)
  (h_tri : h.triangles_identical_right = true) :
  hexagon_area h = 32 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_is_32_l1579_157902


namespace NUMINAMATH_CALUDE_complex_modulus_l1579_157969

theorem complex_modulus (z : ℂ) (h : (3 - 4*I)*z = 1) : Complex.abs z = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1579_157969


namespace NUMINAMATH_CALUDE_polynomial_identity_l1579_157932

theorem polynomial_identity (a b c d : ℝ) :
  (∀ x y : ℝ, (10*x + 6*y)^3 = a*x^3 + b*x^2*y + c*x*y^2 + d*y^3) →
  -a + 2*b - 4*c + 8*d = 8 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1579_157932


namespace NUMINAMATH_CALUDE_percentage_students_taking_music_l1579_157912

theorem percentage_students_taking_music
  (total_students : ℕ)
  (dance_students : ℕ)
  (art_students : ℕ)
  (h1 : total_students = 400)
  (h2 : dance_students = 120)
  (h3 : art_students = 200) :
  (total_students - (dance_students + art_students)) / total_students * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_students_taking_music_l1579_157912


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l1579_157982

/-- Parabola function -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- Theorem stating the relationship between y-coordinates of three points on the parabola -/
theorem parabola_point_relationship : 
  let y₁ := f (-5)
  let y₂ := f 1
  let y₃ := f 12
  y₂ < y₁ ∧ y₁ < y₃ := by sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l1579_157982


namespace NUMINAMATH_CALUDE_some_number_solution_l1579_157967

theorem some_number_solution : 
  ∃ x : ℚ, (1 / 2 : ℚ) + ((2 / 3 : ℚ) * (3 / 8 : ℚ) + 4) - x = (17 / 4 : ℚ) ∧ x = (1 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_some_number_solution_l1579_157967


namespace NUMINAMATH_CALUDE_max_value_of_f_l1579_157906

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The maximum value of f(x) is 2 -/
theorem max_value_of_f : ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M :=
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1579_157906


namespace NUMINAMATH_CALUDE_max_value_of_f_l1579_157905

noncomputable def f (x : ℝ) : ℝ := (-x^2 + x - 4) / x

theorem max_value_of_f :
  ∃ (x_max : ℝ), x_max > 0 ∧
  (∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
  f x_max = -3 ∧
  x_max = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1579_157905


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_8_l1579_157943

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) (A : ℝ) : ℝ := 2 - A * 2^(n - 1)

/-- The geometric sequence {a_n} -/
def a (n : ℕ) (A : ℝ) : ℝ := S n A - S (n-1) A

/-- Theorem stating that S_8 equals -510 for the given geometric sequence -/
theorem geometric_sequence_sum_8 (A : ℝ) (h1 : ∀ n : ℕ, n ≥ 1 → S n A = 2 - A * 2^(n - 1))
  (h2 : ∀ k : ℕ, k ≥ 1 → a (k+1) A / a k A = a (k+2) A / a (k+1) A) :
  S 8 A = -510 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_8_l1579_157943


namespace NUMINAMATH_CALUDE_line_slope_perpendicular_lines_b_value_l1579_157934

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (x₁ y₁ x₂ y₂ : ℝ), y₁ = m₁ * x₁ ∧ y₂ = m₂ * x₂ ∧ (y₂ - y₁) * (x₂ - x₁) = 0)

/-- The slope of a line ax + by + c = 0 where b ≠ 0 is -a/b -/
theorem line_slope (a b c : ℝ) (hb : b ≠ 0) :
  ∃ m : ℝ, m = -a / b ∧ ∀ x y : ℝ, a * x + b * y + c = 0 → y = m * x - c / b :=
sorry

theorem perpendicular_lines_b_value : 
  ∀ b : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ, 
    2 * x₁ + 3 * y₁ - 4 = 0 ∧ 
    b * x₂ + 3 * y₂ - 4 = 0 ∧ 
    (y₂ - y₁) * (x₂ - x₁) = 0) → 
  b = -9/2 :=
sorry

end NUMINAMATH_CALUDE_line_slope_perpendicular_lines_b_value_l1579_157934


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1579_157939

theorem algebraic_simplification (m n : ℝ) :
  (3 * m^2 - m * n + 5) - 2 * (5 * m * n - 4 * m^2 + 2) = 11 * m^2 - 11 * m * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1579_157939


namespace NUMINAMATH_CALUDE_elf_nuts_problem_l1579_157992

theorem elf_nuts_problem (nuts : Fin 10 → ℕ) 
  (h1 : (nuts 0) + (nuts 2) = 110)
  (h2 : (nuts 1) + (nuts 3) = 120)
  (h3 : (nuts 2) + (nuts 4) = 130)
  (h4 : (nuts 3) + (nuts 5) = 140)
  (h5 : (nuts 4) + (nuts 6) = 150)
  (h6 : (nuts 5) + (nuts 7) = 160)
  (h7 : (nuts 6) + (nuts 8) = 170)
  (h8 : (nuts 7) + (nuts 9) = 180)
  (h9 : (nuts 8) + (nuts 0) = 190)
  (h10 : (nuts 9) + (nuts 1) = 200) :
  nuts 5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_elf_nuts_problem_l1579_157992


namespace NUMINAMATH_CALUDE_section_A_average_weight_l1579_157979

/-- Proves that the average weight of section A is 40 kg given the conditions of the problem -/
theorem section_A_average_weight
  (students_A : ℕ)
  (students_B : ℕ)
  (avg_weight_B : ℝ)
  (avg_weight_total : ℝ)
  (h1 : students_A = 30)
  (h2 : students_B = 20)
  (h3 : avg_weight_B = 35)
  (h4 : avg_weight_total = 38) :
  let total_students := students_A + students_B
  let avg_weight_A := (avg_weight_total * total_students - avg_weight_B * students_B) / students_A
  avg_weight_A = 40 := by
sorry


end NUMINAMATH_CALUDE_section_A_average_weight_l1579_157979


namespace NUMINAMATH_CALUDE_rectangle_area_unchanged_l1579_157961

theorem rectangle_area_unchanged (A l w : ℝ) (h1 : A = l * w) (h2 : A > 0) :
  let l' := 0.8 * l
  let w' := 1.25 * w
  l' * w' = A := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_unchanged_l1579_157961


namespace NUMINAMATH_CALUDE_equilateral_triangle_product_l1579_157960

/-- Given that (0, 0), (a, 8), and (b, 20) form an equilateral triangle,
    prove that ab = 320/3 -/
theorem equilateral_triangle_product (a b : ℝ) : 
  (∃ (θ : ℝ), θ = π/3 ∨ θ = -π/3) →
  (Complex.abs (Complex.I * 8 - 0) = Complex.abs (b + Complex.I * 20 - 0)) →
  (Complex.abs (b + Complex.I * 20 - (a + Complex.I * 8)) = Complex.abs (Complex.I * 8 - 0)) →
  (b + Complex.I * 20 = (a + Complex.I * 8) * Complex.exp (Complex.I * θ)) →
  a * b = 320 / 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_product_l1579_157960


namespace NUMINAMATH_CALUDE_composite_expression_l1579_157903

/-- A positive integer is composite if it can be expressed as a product of two integers,
    each greater than or equal to 2. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≥ 2 ∧ b ≥ 2 ∧ n = a * b

/-- Every composite number can be expressed as xy + xz + yz + 1,
    where x, y, and z are positive integers. -/
theorem composite_expression (c : ℕ) (h : IsComposite c) :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ c = x * y + x * z + y * z + 1 :=
sorry

end NUMINAMATH_CALUDE_composite_expression_l1579_157903


namespace NUMINAMATH_CALUDE_one_switch_determines_light_l1579_157913

/-- Represents the state of a switch -/
inductive SwitchState
| Position1
| Position2
| Position3

/-- Represents a light bulb -/
inductive Light
| Bulb1
| Bulb2
| Bulb3

/-- Configuration of all switches -/
def SwitchConfig (n : ℕ) := Fin n → SwitchState

/-- Function that determines which light is on given a switch configuration -/
def lightOn (n : ℕ) (config : SwitchConfig n) : Light := sorry

theorem one_switch_determines_light (n : ℕ) :
  (∀ (config : SwitchConfig n), ∃! (l : Light), lightOn n config = l) →
  (∀ (config1 config2 : SwitchConfig n), 
    (∀ i, config1 i ≠ config2 i) → lightOn n config1 ≠ lightOn n config2) →
  ∃ (k : Fin n), ∀ (config1 config2 : SwitchConfig n),
    (∀ (i : Fin n), i ≠ k → config1 i = config2 i) →
    (config1 k = config2 k → lightOn n config1 = lightOn n config2) :=
sorry

end NUMINAMATH_CALUDE_one_switch_determines_light_l1579_157913


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1579_157980

theorem units_digit_of_expression : ∃ n : ℕ, (12 + Real.sqrt 36)^17 + (12 - Real.sqrt 36)^17 = 10 * n + 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1579_157980


namespace NUMINAMATH_CALUDE_max_ate_second_most_l1579_157945

-- Define the children as a finite type
inductive Child : Type
  | Chris : Child
  | Max : Child
  | Brandon : Child
  | Kayla : Child
  | Tanya : Child

-- Define the eating relation
def ate_more_than (a b : Child) : Prop := sorry

-- Define the conditions
axiom chris_ate_more_than_max : ate_more_than Child.Chris Child.Max
axiom brandon_ate_less_than_kayla : ate_more_than Child.Kayla Child.Brandon
axiom kayla_ate_less_than_max : ate_more_than Child.Max Child.Kayla
axiom kayla_ate_more_than_tanya : ate_more_than Child.Kayla Child.Tanya

-- Define what it means to be the second most
def is_second_most (c : Child) : Prop :=
  ∃ (first : Child), (first ≠ c) ∧
    (∀ (other : Child), other ≠ first → other ≠ c → ate_more_than c other)

-- The theorem to prove
theorem max_ate_second_most : is_second_most Child.Max := by sorry

end NUMINAMATH_CALUDE_max_ate_second_most_l1579_157945


namespace NUMINAMATH_CALUDE_sin_135_degrees_l1579_157964

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l1579_157964


namespace NUMINAMATH_CALUDE_distribution_equivalence_l1579_157925

/-- The number of ways to distribute n indistinguishable objects among k recipients,
    with each recipient receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribution_equivalence :
  distribute 10 7 = choose 9 6 := by sorry

end NUMINAMATH_CALUDE_distribution_equivalence_l1579_157925


namespace NUMINAMATH_CALUDE_nina_running_distance_l1579_157933

theorem nina_running_distance : 0.08 + 0.08 + 0.67 = 0.83 := by sorry

end NUMINAMATH_CALUDE_nina_running_distance_l1579_157933


namespace NUMINAMATH_CALUDE_vector_subtraction_l1579_157909

/-- Given vectors BA and CA in ℝ², prove that BC = BA - CA -/
theorem vector_subtraction (BA CA : ℝ × ℝ) (h1 : BA = (2, 3)) (h2 : CA = (4, 7)) :
  (BA.1 - CA.1, BA.2 - CA.2) = (-2, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1579_157909


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_1080_l1579_157946

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items --/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of ways to divide 6 families into 4 groups and allocate to 4 villages --/
def allocation_schemes : ℕ :=
  let group_formations := choose 6 2 * choose 4 2 * choose 2 1 * choose 1 1 / (arrange 2 2 * arrange 2 2)
  let village_allocations := arrange 4 4
  group_formations * village_allocations

theorem allocation_schemes_eq_1080 : allocation_schemes = 1080 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_1080_l1579_157946


namespace NUMINAMATH_CALUDE_harvest_calculation_l1579_157917

/-- Represents the harvest schedule and quantities for oranges and apples -/
structure HarvestData where
  total_days : ℕ
  orange_sacks : ℕ
  apple_sacks : ℕ
  orange_interval : ℕ
  apple_interval : ℕ

/-- Calculates the number of sacks harvested per day when both fruits are harvested together -/
def sacks_per_joint_harvest_day (data : HarvestData) : ℚ :=
  let orange_days := data.total_days / data.orange_interval
  let apple_days := data.total_days / data.apple_interval
  let orange_per_day := data.orange_sacks / orange_days
  let apple_per_day := data.apple_sacks / apple_days
  orange_per_day + apple_per_day

/-- The main theorem stating the result of the harvest calculation -/
theorem harvest_calculation (data : HarvestData) 
  (h1 : data.total_days = 20)
  (h2 : data.orange_sacks = 56)
  (h3 : data.apple_sacks = 35)
  (h4 : data.orange_interval = 2)
  (h5 : data.apple_interval = 3) :
  sacks_per_joint_harvest_day data = 11.4333 := by
  sorry

end NUMINAMATH_CALUDE_harvest_calculation_l1579_157917


namespace NUMINAMATH_CALUDE_log_c_27_is_0_75_implies_c_is_81_l1579_157957

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_c_27_is_0_75_implies_c_is_81 :
  ∀ c : ℝ, c > 0 → log c 27 = 0.75 → c = 81 := by
  sorry

end NUMINAMATH_CALUDE_log_c_27_is_0_75_implies_c_is_81_l1579_157957


namespace NUMINAMATH_CALUDE_max_value_constraint_l1579_157981

theorem max_value_constraint (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) :
  a*b + b*c + c*d + d*a + a*c + 4*b*d ≤ 5/2 := by
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1579_157981


namespace NUMINAMATH_CALUDE_circle_circumference_l1579_157999

-- Define the circles x and y
def circle_x : Real → Prop := sorry
def circle_y : Real → Prop := sorry

-- Define the area of a circle
def area (circle : Real → Prop) : Real := sorry

-- Define the radius of a circle
def radius (circle : Real → Prop) : Real := sorry

-- Define the circumference of a circle
def circumference (circle : Real → Prop) : Real := sorry

-- State the theorem
theorem circle_circumference :
  (area circle_x = area circle_y) →  -- Circles x and y have the same area
  (radius circle_y / 2 = 3.5) →      -- Half of the radius of circle y is 3.5
  (circumference circle_x = 14 * Real.pi) := -- The circumference of circle x is 14π
by sorry

end NUMINAMATH_CALUDE_circle_circumference_l1579_157999


namespace NUMINAMATH_CALUDE_two_fifths_300_minus_three_fifths_125_l1579_157953

theorem two_fifths_300_minus_three_fifths_125 : 
  (2 : ℚ) / 5 * 300 - (3 : ℚ) / 5 * 125 = 45 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_300_minus_three_fifths_125_l1579_157953


namespace NUMINAMATH_CALUDE_train_length_l1579_157994

/-- Given a train traveling at 72 km/hr crossing a 270 m long platform in 26 seconds,
    the length of the train is 250 meters. -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  speed = 72 * 1000 / 3600 →
  platform_length = 270 →
  crossing_time = 26 →
  speed * crossing_time - platform_length = 250 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1579_157994


namespace NUMINAMATH_CALUDE_prob_sum_five_is_one_ninth_l1579_157970

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := faces * faces

/-- The number of favorable outcomes (sum of 5) when rolling two dice -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two dice -/
def prob_sum_five : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_five_is_one_ninth :
  prob_sum_five = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_sum_five_is_one_ninth_l1579_157970


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l1579_157914

theorem quadratic_equation_problem (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + 2*m*x₁ + m^2 - m + 2 = 0 ∧
    x₂^2 + 2*m*x₂ + m^2 - m + 2 = 0 ∧
    x₁ + x₂ + x₁ * x₂ = 2) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l1579_157914


namespace NUMINAMATH_CALUDE_square_area_problem_l1579_157952

theorem square_area_problem (a b : ℝ) (h : a > b) :
  let diagonal_I := a - b
  let area_I := (diagonal_I^2) / 2
  let area_II := 3 * area_I
  area_II = (3 * (a - b)^2) / 2 := by
sorry

end NUMINAMATH_CALUDE_square_area_problem_l1579_157952


namespace NUMINAMATH_CALUDE_f_of_4_equals_9_l1579_157940

/-- The function f is defined as f(x) = x^2 - 2x + 1 for all x. -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem: The value of f(4) is 9. -/
theorem f_of_4_equals_9 : f 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_4_equals_9_l1579_157940


namespace NUMINAMATH_CALUDE_average_book_cost_l1579_157984

/-- Given that Fred had $236 initially, bought 6 books, and had $14 left after the purchase,
    prove that the average cost of each book is $37. -/
theorem average_book_cost (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) :
  initial_amount = 236 →
  num_books = 6 →
  remaining_amount = 14 →
  (initial_amount - remaining_amount) / num_books = 37 :=
by sorry

end NUMINAMATH_CALUDE_average_book_cost_l1579_157984


namespace NUMINAMATH_CALUDE_optionC_most_suitable_l1579_157918

/-- Represents a sampling option with population size and sample size -/
structure SamplingOption where
  populationSize : ℕ
  sampleSize : ℕ

/-- Determines if a sampling option is suitable for simple random sampling -/
def isSuitableForSimpleRandomSampling (option : SamplingOption) : Prop :=
  option.populationSize ≤ 30 ∧ option.sampleSize ≤ 5

/-- The given sampling options -/
def optionA : SamplingOption := ⟨1320, 300⟩
def optionB : SamplingOption := ⟨1135, 50⟩
def optionC : SamplingOption := ⟨30, 5⟩
def optionD : SamplingOption := ⟨5000, 200⟩

/-- Theorem stating that Option C is most suitable for simple random sampling -/
theorem optionC_most_suitable :
  isSuitableForSimpleRandomSampling optionC ∧
  ¬isSuitableForSimpleRandomSampling optionA ∧
  ¬isSuitableForSimpleRandomSampling optionB ∧
  ¬isSuitableForSimpleRandomSampling optionD :=
by sorry

end NUMINAMATH_CALUDE_optionC_most_suitable_l1579_157918


namespace NUMINAMATH_CALUDE_solve_for_d_l1579_157938

theorem solve_for_d (a c d n : ℝ) (h : n = (c * d * a) / (a - d)) :
  d = (n * a) / (c * d + n) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_d_l1579_157938


namespace NUMINAMATH_CALUDE_doubled_cost_percentage_new_cost_percentage_l1579_157983

-- Define the cost function
def cost (t b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_cost_percentage (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  cost t (2 * b) = 16 * cost t b :=
by sorry

-- Main theorem
theorem new_cost_percentage (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  (cost t (2 * b) / cost t b) * 100 = 1600 :=
by sorry

end NUMINAMATH_CALUDE_doubled_cost_percentage_new_cost_percentage_l1579_157983


namespace NUMINAMATH_CALUDE_function_odd_iff_sum_squares_zero_l1579_157997

/-- The function f(x) = x|x-a| + b is odd if and only if a^2 + b^2 = 0 -/
theorem function_odd_iff_sum_squares_zero (a b : ℝ) :
  (∀ x : ℝ, x * |x - a| + b = -((-x) * |(-x) - a| + b)) ↔ a^2 + b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_odd_iff_sum_squares_zero_l1579_157997


namespace NUMINAMATH_CALUDE_price_decrease_l1579_157927

theorem price_decrease (original_price reduced_price : ℝ) 
  (h1 : reduced_price = original_price * (1 - 0.24))
  (h2 : reduced_price = 532) : original_price = 700 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_l1579_157927


namespace NUMINAMATH_CALUDE_soccer_ball_inflation_time_l1579_157951

/-- The time in minutes it takes to inflate one soccer ball -/
def inflationTime : ℕ := 20

/-- The number of balls Alexia inflates -/
def alexiaBalls : ℕ := 20

/-- The number of additional balls Ermias inflates compared to Alexia -/
def ermiasDifference : ℕ := 5

/-- The total number of balls Ermias inflates -/
def ermiasBalls : ℕ := alexiaBalls + ermiasDifference

/-- The total number of balls inflated by both Alexia and Ermias -/
def totalBalls : ℕ := alexiaBalls + ermiasBalls

/-- The total time in minutes taken to inflate all soccer balls -/
def totalTime : ℕ := totalBalls * inflationTime

theorem soccer_ball_inflation_time : totalTime = 900 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_inflation_time_l1579_157951


namespace NUMINAMATH_CALUDE_unique_solution_abs_equation_l1579_157926

theorem unique_solution_abs_equation :
  ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_abs_equation_l1579_157926


namespace NUMINAMATH_CALUDE_simplify_fourth_roots_l1579_157948

theorem simplify_fourth_roots : 64^(1/4) - 144^(1/4) = 2 * Real.sqrt 2 - 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fourth_roots_l1579_157948


namespace NUMINAMATH_CALUDE_bike_tractor_speed_ratio_l1579_157990

/-- Given the speeds and distances of vehicles, prove the ratio of bike speed to tractor speed --/
theorem bike_tractor_speed_ratio :
  ∀ (car_speed bike_speed tractor_speed : ℝ),
  car_speed = (9/5) * bike_speed →
  tractor_speed = 575 / 25 →
  car_speed = 331.2 / 4 →
  bike_speed / tractor_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_bike_tractor_speed_ratio_l1579_157990


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l1579_157944

theorem unique_solution_floor_equation :
  ∃! (x : ℝ), x > 0 ∧ x * ↑(⌊x⌋) = 72 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l1579_157944


namespace NUMINAMATH_CALUDE_hyperbola_min_focal_length_l1579_157908

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    focal length 2c, and a + b - c = 2, the minimum value of 2c is 4 + 4√2. -/
theorem hyperbola_min_focal_length (a b c : ℝ) : 
  a > 0 → b > 0 → a + b - c = 2 → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  2 * c ≥ 4 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_min_focal_length_l1579_157908


namespace NUMINAMATH_CALUDE_linear_function_proof_l1579_157923

/-- A linear function passing through points (1, 3) and (-2, 12) -/
def f (x : ℝ) : ℝ := -3 * x + 6

theorem linear_function_proof :
  (f 1 = 3 ∧ f (-2) = 12) ∧
  (∀ a : ℝ, f (2 * a) ≠ -6 * a + 8) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_proof_l1579_157923


namespace NUMINAMATH_CALUDE_complex_number_problem_l1579_157971

theorem complex_number_problem (z ω : ℂ) :
  (((1 : ℂ) + 3*Complex.I) * z).re = 0 →
  ω = z / ((2 : ℂ) + Complex.I) →
  Complex.abs ω = 5 * Real.sqrt 2 →
  ω = 7 - Complex.I ∨ ω = -7 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1579_157971
