import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l3276_327648

theorem problem_solution (x : ℝ) : 
  x + Real.sqrt (x^2 - 1) + 1 / (x - Real.sqrt (x^2 - 1)) = 24 →
  x^2 + Real.sqrt (x^4 - 1) + 1 / (x^2 + Real.sqrt (x^4 - 1)) = 10525 / 144 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3276_327648


namespace NUMINAMATH_CALUDE_P_no_negative_roots_l3276_327629

/-- The polynomial P(x) = x^4 - 5x^3 + 3x^2 - 7x + 1 -/
def P (x : ℝ) : ℝ := x^4 - 5*x^3 + 3*x^2 - 7*x + 1

/-- Theorem: The polynomial P(x) has no negative roots -/
theorem P_no_negative_roots : ∀ x : ℝ, x < 0 → P x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_P_no_negative_roots_l3276_327629


namespace NUMINAMATH_CALUDE_complex_product_theorem_l3276_327622

theorem complex_product_theorem (z₁ z₂ : ℂ) (a b : ℝ) : 
  z₁ = (1 - Complex.I) * (3 + Complex.I) →
  a = z₁.im →
  z₂ = (1 + Complex.I) / (2 - Complex.I) →
  b = z₂.re →
  a * b = -2/5 := by
    sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l3276_327622


namespace NUMINAMATH_CALUDE_discount_ratio_proof_l3276_327630

/-- Proves that the ratio of discounts at different times is 1.833 -/
theorem discount_ratio_proof (original_bill : ℝ) (original_discount : ℝ) (longer_discount : ℝ) :
  original_bill = 110 →
  original_discount = 10 →
  longer_discount = 18.33 →
  longer_discount / original_discount = 1.833 := by
  sorry

end NUMINAMATH_CALUDE_discount_ratio_proof_l3276_327630


namespace NUMINAMATH_CALUDE_no_equal_factorial_and_even_factorial_l3276_327600

theorem no_equal_factorial_and_even_factorial :
  ¬ ∃ (n m : ℕ), n.factorial = 2^m * m.factorial ∧ m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_equal_factorial_and_even_factorial_l3276_327600


namespace NUMINAMATH_CALUDE_power_sum_equation_l3276_327673

/-- Given two real numbers a and b satisfying certain conditions, prove that a^10 + b^10 = 123 -/
theorem power_sum_equation (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equation_l3276_327673


namespace NUMINAMATH_CALUDE_log_less_than_one_range_l3276_327680

theorem log_less_than_one_range (a : ℝ) :
  (∃ (x : ℝ), Real.log x / Real.log a < 1) → a ∈ Set.union (Set.Ioo 0 1) (Set.Ioi 1) := by
  sorry

end NUMINAMATH_CALUDE_log_less_than_one_range_l3276_327680


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3276_327659

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x + 2 < 1 ↔ 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3276_327659


namespace NUMINAMATH_CALUDE_window_side_length_main_theorem_l3276_327676

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio_height_to_width : height = 3 * width

/-- Represents the dimensions of a square window -/
structure SquareWindow where
  pane : Pane
  border_width : ℝ
  side_length : ℝ
  pane_arrangement : side_length = 3 * pane.width + 4 * border_width

/-- Theorem: The side length of the square window is 24 inches -/
theorem window_side_length (w : SquareWindow) 
  (h1 : w.border_width = 3) : w.side_length = 24 := by
  sorry

/-- Main theorem combining all conditions -/
theorem main_theorem : ∃ (w : SquareWindow), 
  w.border_width = 3 ∧ w.side_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_window_side_length_main_theorem_l3276_327676


namespace NUMINAMATH_CALUDE_chocolate_price_after_discount_l3276_327691

/-- The final price of a chocolate after discount -/
def final_price (original_cost discount : ℚ) : ℚ :=
  original_cost - discount

/-- Theorem: The final price of a chocolate with original cost $2 and discount $0.57 is $1.43 -/
theorem chocolate_price_after_discount :
  final_price 2 0.57 = 1.43 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_price_after_discount_l3276_327691


namespace NUMINAMATH_CALUDE_circle_condition_l3276_327653

/-- The equation x^2 + y^2 - 2x + m = 0 represents a circle if and only if m < 1 -/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + m = 0 ∧ ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - 2*x + m = 0) ↔ 
  m < 1 := by sorry

end NUMINAMATH_CALUDE_circle_condition_l3276_327653


namespace NUMINAMATH_CALUDE_min_black_cells_l3276_327677

/-- Represents a board configuration -/
def Board := Fin 2007 → Fin 2007 → Bool

/-- Checks if three cells form an L-trinome -/
def is_L_trinome (b : Board) (i j k : Fin 2007 × Fin 2007) : Prop :=
  sorry

/-- Checks if a board configuration is valid -/
def is_valid_configuration (b : Board) : Prop :=
  ∀ i j k, is_L_trinome b i j k → ¬(b i.1 i.2 ∧ b j.1 j.2 ∧ b k.1 k.2)

/-- Counts the number of black cells in a board configuration -/
def count_black_cells (b : Board) : Nat :=
  sorry

/-- The main theorem -/
theorem min_black_cells :
  ∃ (b : Board),
    is_valid_configuration b ∧
    count_black_cells b = (2007^2 / 3 : Nat) ∧
    ∀ (b' : Board),
      (∀ i j, b i j → b' i j) →
      count_black_cells b' > count_black_cells b →
      ¬is_valid_configuration b' :=
sorry

end NUMINAMATH_CALUDE_min_black_cells_l3276_327677


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_one_third_l3276_327685

theorem reciprocal_of_repeating_decimal_one_third (x : ℚ) : 
  (∀ n : ℕ, (10 * x - x) * 10^n = 3 * 10^n - 3) → 
  (1 / x = 3) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_one_third_l3276_327685


namespace NUMINAMATH_CALUDE_omelets_per_person_l3276_327612

/-- Given 3 dozen eggs, 4 eggs per omelet, and 3 people, prove that each person gets 3 omelets when all eggs are used. -/
theorem omelets_per_person (total_eggs : ℕ) (eggs_per_omelet : ℕ) (num_people : ℕ) :
  total_eggs = 3 * 12 →
  eggs_per_omelet = 4 →
  num_people = 3 →
  (total_eggs / eggs_per_omelet) / num_people = 3 :=
by sorry

end NUMINAMATH_CALUDE_omelets_per_person_l3276_327612


namespace NUMINAMATH_CALUDE_circular_track_circumference_l3276_327621

/-- Represents a circular track with two moving points. -/
structure CircularTrack where
  /-- The circumference of the track in yards. -/
  circumference : ℝ
  /-- The constant speed of both points (assumed to be the same). -/
  speed : ℝ
  /-- The distance B travels before the first meeting. -/
  first_meeting_distance : ℝ
  /-- The remaining distance A needs to travel after the second meeting to complete a lap. -/
  second_meeting_remaining : ℝ

/-- The theorem stating the conditions and the result to be proved. -/
theorem circular_track_circumference (track : CircularTrack) 
  (h1 : track.first_meeting_distance = 100)
  (h2 : track.second_meeting_remaining = 60)
  (h3 : track.speed > 0) :
  track.circumference = 480 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_circumference_l3276_327621


namespace NUMINAMATH_CALUDE_altered_solution_detergent_volume_l3276_327618

/-- Given a cleaning solution with initial ratio of bleach:detergent:disinfectant:water as 2:40:10:100,
    and after altering the solution such that:
    1) The ratio of bleach to detergent is tripled
    2) The ratio of detergent to water is halved
    3) The ratio of disinfectant to bleach is doubled
    If the altered solution contains 300 liters of water, prove that it contains 60 liters of detergent. -/
theorem altered_solution_detergent_volume (b d f w : ℚ) : 
  b / d = 2 / 40 →
  d / w = 40 / 100 →
  f / b = 10 / 2 →
  (3 * b) / d = 3 * (2 / 40) →
  d / w = (1 / 2) * (40 / 100) →
  f / (3 * b) = 2 * (10 / 2) →
  w = 300 →
  d = 60 := by
sorry

end NUMINAMATH_CALUDE_altered_solution_detergent_volume_l3276_327618


namespace NUMINAMATH_CALUDE_simplify_part1_simplify_part2_l3276_327678

-- Part 1
theorem simplify_part1 (x : ℝ) (h1 : 1 ≤ x) (h2 : x < 4) :
  Real.sqrt (1 - 2*x + x^2) + Real.sqrt (x^2 - 8*x + 16) = 3 := by sorry

-- Part 2
theorem simplify_part2 (x : ℝ) (h : 2 - x ≥ 0) :
  (Real.sqrt (2 - x))^2 - Real.sqrt (x^2 - 6*x + 9) = -1 := by sorry

end NUMINAMATH_CALUDE_simplify_part1_simplify_part2_l3276_327678


namespace NUMINAMATH_CALUDE_min_a_value_l3276_327696

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a * x^3
def g (x : ℝ) : ℝ := 9 * x^2 + 3 * x - 1

-- State the theorem
theorem min_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, f a x ≥ g x) → a ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_min_a_value_l3276_327696


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3276_327675

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented -/
def original_number : ℕ := 384000

/-- The scientific notation representation -/
def scientific_rep : ScientificNotation :=
  { coefficient := 3.84
    exponent := 5
    coeff_range := by sorry }

theorem scientific_notation_correct :
  (scientific_rep.coefficient * (10 : ℝ) ^ scientific_rep.exponent) = original_number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l3276_327675


namespace NUMINAMATH_CALUDE_square_difference_identity_l3276_327697

theorem square_difference_identity :
  287 * 287 + 269 * 269 - 2 * 287 * 269 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l3276_327697


namespace NUMINAMATH_CALUDE_area_covered_by_overlapping_strips_l3276_327628

/-- Represents a rectangular strip with a given length and width of 1 unit -/
structure Strip where
  length : ℝ
  width : ℝ := 1

/-- Calculates the total area of overlaps between strips -/
def totalOverlapArea (strips : List Strip) : ℝ := sorry

/-- Theorem: Area covered by overlapping strips -/
theorem area_covered_by_overlapping_strips
  (strips : List Strip)
  (h_strips : strips = [
    { length := 8 },
    { length := 10 },
    { length := 12 },
    { length := 7 },
    { length := 9 }
  ])
  (h_overlap : totalOverlapArea strips = 16) :
  (strips.map (λ s => s.length * s.width)).sum - totalOverlapArea strips = 30 := by
  sorry

end NUMINAMATH_CALUDE_area_covered_by_overlapping_strips_l3276_327628


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3276_327672

def A (m : ℝ) : Set ℝ := {2, m^2}
def B : Set ℝ := {0, 1, 3}

theorem sufficient_not_necessary :
  (∀ m : ℝ, m = 1 → A m ∩ B = {1}) ∧
  (∃ m : ℝ, m ≠ 1 ∧ A m ∩ B = {1}) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3276_327672


namespace NUMINAMATH_CALUDE_rectangle_image_is_curved_region_l3276_327649

-- Define the rectangle OAPB
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (0, 3)

-- Define the transformation
def u (x y : ℝ) : ℝ := x^2 - y^2
def v (x y : ℝ) : ℝ := x * y

-- Define the image of a point under the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ := (u p.1 p.2, v p.1 p.2)

-- Theorem statement
theorem rectangle_image_is_curved_region :
  ∃ (R : Set (ℝ × ℝ)), 
    (∀ p ∈ R, ∃ q ∈ Set.Icc O A ∪ Set.Icc A P ∪ Set.Icc P B ∪ Set.Icc B O, p = transform q) ∧
    (∀ q ∈ Set.Icc O A ∪ Set.Icc A P ∪ Set.Icc P B ∪ Set.Icc B O, transform q ∈ R) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform O ∧ (f 1, g 1) = transform A) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform A ∧ (f 1, g 1) = transform P) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform P ∧ (f 1, g 1) = transform B) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform B ∧ (f 1, g 1) = transform O) :=
sorry

end NUMINAMATH_CALUDE_rectangle_image_is_curved_region_l3276_327649


namespace NUMINAMATH_CALUDE_correct_average_l3276_327682

theorem correct_average (numbers : Finset ℕ) (incorrect_sum : ℕ) (incorrect_number correct_number : ℕ) :
  numbers.card = 10 →
  incorrect_sum = 17 * 10 →
  incorrect_number = 26 →
  correct_number = 56 →
  (incorrect_sum - incorrect_number + correct_number) / numbers.card = 20 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l3276_327682


namespace NUMINAMATH_CALUDE_min_c_value_l3276_327639

theorem min_c_value (a b c k : ℕ+) (h1 : b = a + k) (h2 : c = a + 2*k) 
  (h3 : a < b ∧ b < c) 
  (h4 : ∃! (x y : ℝ), 3*x + y = 3005 ∧ y = |x - a| + |x - (a + k)| + |x - (a + 2*k)|) :
  c ≥ 6005 ∧ ∃ (a₀ b₀ c₀ k₀ : ℕ+), 
    b₀ = a₀ + k₀ ∧ c₀ = a₀ + 2*k₀ ∧ a₀ < b₀ ∧ b₀ < c₀ ∧ c₀ = 6005 ∧
    ∃! (x y : ℝ), 3*x + y = 3005 ∧ y = |x - a₀| + |x - (a₀ + k₀)| + |x - (a₀ + 2*k₀)| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l3276_327639


namespace NUMINAMATH_CALUDE_probability_of_even_sum_l3276_327633

def number_of_balls : ℕ := 20

def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_is_even (a b : ℕ) : Prop := is_even (a + b)

theorem probability_of_even_sum :
  let total_outcomes := number_of_balls * (number_of_balls - 1)
  let favorable_outcomes := (number_of_balls / 2) * ((number_of_balls / 2) - 1) * 2
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 19 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_even_sum_l3276_327633


namespace NUMINAMATH_CALUDE_M_equals_P_l3276_327620

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = x^2 - 1}
def P : Set ℝ := {a | ∃ b, a = b^2 - 1}

-- Theorem statement
theorem M_equals_P : M = P := by
  sorry

end NUMINAMATH_CALUDE_M_equals_P_l3276_327620


namespace NUMINAMATH_CALUDE_only_winning_lottery_is_random_l3276_327605

-- Define the type for events
inductive Event
  | WaterBoiling
  | WinningLottery
  | AthleteRunning
  | DrawingRedBall

-- Define the property of being a random event
def isRandomEvent (e : Event) : Prop :=
  match e with
  | Event.WaterBoiling => false
  | Event.WinningLottery => true
  | Event.AthleteRunning => false
  | Event.DrawingRedBall => false

-- Theorem statement
theorem only_winning_lottery_is_random :
  ∀ e : Event, isRandomEvent e ↔ e = Event.WinningLottery :=
sorry

end NUMINAMATH_CALUDE_only_winning_lottery_is_random_l3276_327605


namespace NUMINAMATH_CALUDE_expression_value_l3276_327636

theorem expression_value (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 27665/27 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3276_327636


namespace NUMINAMATH_CALUDE_g_derivative_at_one_l3276_327656

/-- The sequence of functions gₖ(x) -/
noncomputable def g : ℕ → (ℝ → ℝ)
| 0 => λ x => x^2 / (2 - x)
| (k+1) => λ x => x * g k x / (2 - g k x)

/-- The statement to be proved -/
theorem g_derivative_at_one (k : ℕ) :
  HasDerivAt (g k) (2^(k+1) - 1) 1 :=
sorry

end NUMINAMATH_CALUDE_g_derivative_at_one_l3276_327656


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3276_327651

/-- Given a cylinder with base area S whose lateral surface unfolds into a square,
    prove that its lateral surface area is 4πS. -/
theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let h := 2 * Real.pi * r
  (h = 2 * r)  → 2 * Real.pi * r * h = 4 * Real.pi * S :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3276_327651


namespace NUMINAMATH_CALUDE_c_value_for_four_roots_l3276_327615

/-- A complex number is a root of the polynomial Q(x) if Q(x) = 0 -/
def is_root (Q : ℂ → ℂ) (z : ℂ) : Prop := Q z = 0

/-- The polynomial Q(x) -/
def Q (c : ℂ) (x : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - c*x + 2) * (x^2 - 5*x + 5)

/-- The theorem stating the value of |c| for Q(x) with exactly 4 distinct roots -/
theorem c_value_for_four_roots :
  ∃ (c : ℂ), (∃ (z₁ z₂ z₃ z₄ : ℂ), z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    is_root (Q c) z₁ ∧ is_root (Q c) z₂ ∧ is_root (Q c) z₃ ∧ is_root (Q c) z₄ ∧
    (∀ (z : ℂ), is_root (Q c) z → z = z₁ ∨ z = z₂ ∨ z = z₃ ∨ z = z₄)) →
  Complex.abs c = Real.sqrt (18 - Real.sqrt 15 / 2) :=
sorry

end NUMINAMATH_CALUDE_c_value_for_four_roots_l3276_327615


namespace NUMINAMATH_CALUDE_right_triangle_area_l3276_327601

/-- The area of a right-angled triangle can be expressed in terms of its hypotenuse and one of its acute angles. -/
theorem right_triangle_area (c α : ℝ) (h_c : c > 0) (h_α : 0 < α ∧ α < π / 2) :
  let t := (1 / 4) * c^2 * Real.sin (2 * α)
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧ (1 / 2) * a * b = t :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3276_327601


namespace NUMINAMATH_CALUDE_roots_sum_powers_l3276_327662

theorem roots_sum_powers (α β : ℝ) : 
  α^2 - 4*α + 1 = 0 → β^2 - 4*β + 1 = 0 → 7*α^3 + 3*β^4 = 1019 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l3276_327662


namespace NUMINAMATH_CALUDE_seating_arrangements_l3276_327655

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people sit next to each other -/
def adjacentArrangements (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * 2

/-- The number of ways to arrange n people in a row where two specific people cannot sit next to each other -/
def nonAdjacentArrangements (n : ℕ) : ℕ := totalArrangements n - adjacentArrangements n

theorem seating_arrangements :
  nonAdjacentArrangements 7 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3276_327655


namespace NUMINAMATH_CALUDE_positive_derivative_implies_increasing_exists_increasing_with_nonpositive_derivative_l3276_327686

open Set
open Function

-- Define a differentiable function on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Part 1: If f'(x) > 0 for all x, then f is monotonically increasing
theorem positive_derivative_implies_increasing :
  (∀ x, deriv f x > 0) → MonotonicallyIncreasing f :=
sorry

-- Part 2: There exists a monotonically increasing function with f'(x) ≤ 0 for some x
theorem exists_increasing_with_nonpositive_derivative :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ MonotonicallyIncreasing f ∧ ∃ x, deriv f x ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_positive_derivative_implies_increasing_exists_increasing_with_nonpositive_derivative_l3276_327686


namespace NUMINAMATH_CALUDE_star_equality_implies_x_equals_six_l3276_327663

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a + c, b - d)

theorem star_equality_implies_x_equals_six :
  ∀ x y : ℤ, star 5 5 2 2 = star x y 1 3 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_implies_x_equals_six_l3276_327663


namespace NUMINAMATH_CALUDE_part_one_part_two_l3276_327661

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Part I
theorem part_one (x : ℝ) :
  (∃ a : ℝ, a = 1 ∧ a > 0 ∧ p x a ∧ q x) → 2 < x ∧ x < 3 :=
sorry

-- Part II
theorem part_two :
  (∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x)) →
  (∃ a : ℝ, 1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3276_327661


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l3276_327669

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  A * B = !![23, -7; 24, -16] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l3276_327669


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3276_327638

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number only uses specified digits -/
def uses_specified_digits (n : ℕ) : Prop := sorry

theorem digit_sum_problem (M : ℕ) 
  (h_even : Even M)
  (h_digits : uses_specified_digits M)
  (h_double : sum_of_digits (2 * M) = 35)
  (h_half : sum_of_digits (M / 2) = 29) :
  sum_of_digits M = 31 := by sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3276_327638


namespace NUMINAMATH_CALUDE_polygonal_chain_existence_l3276_327650

/-- A type representing a line in a plane -/
structure Line where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- A type representing a point in a plane -/
structure Point where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- A type representing a polygonal chain -/
structure PolygonalChain (n : ℕ) where
  vertices : Fin (n + 1) → Point
  segments : Fin n → Line

/-- Predicate to check if a polygonal chain is non-self-intersecting -/
def is_non_self_intersecting (chain : PolygonalChain n) : Prop :=
  sorry

/-- Predicate to check if each segment of a polygonal chain lies on a unique line -/
def segments_on_unique_lines (chain : PolygonalChain n) (lines : Fin n → Line) : Prop :=
  sorry

/-- Predicate to check if no two lines are parallel -/
def no_parallel_lines (lines : Fin n → Line) : Prop :=
  sorry

/-- Predicate to check if no three lines intersect at the same point -/
def no_three_lines_intersect (lines : Fin n → Line) : Prop :=
  sorry

/-- Main theorem statement -/
theorem polygonal_chain_existence (n : ℕ) (lines : Fin n → Line) 
  (h1 : no_parallel_lines lines) 
  (h2 : no_three_lines_intersect lines) : 
  ∃ (chain : PolygonalChain n), 
    is_non_self_intersecting chain ∧ 
    segments_on_unique_lines chain lines :=
  sorry

end NUMINAMATH_CALUDE_polygonal_chain_existence_l3276_327650


namespace NUMINAMATH_CALUDE_max_lateral_surface_area_l3276_327610

/-- A right prism ABCD-A₁B₁C₁D₁ inscribed in a sphere O -/
structure InscribedPrism where
  /-- The base edge length of the prism -/
  a : ℝ
  /-- The height of the prism -/
  h : ℝ
  /-- The radius of the sphere -/
  r : ℝ
  /-- The surface area of the sphere is 12π -/
  sphere_area : 4 * π * r^2 = 12 * π
  /-- The prism is inscribed in the sphere -/
  inscribed : 2 * a^2 + h^2 = 4 * r^2

/-- The lateral surface area of the prism -/
def lateralSurfaceArea (p : InscribedPrism) : ℝ := 4 * p.a * p.h

/-- The theorem stating the maximum lateral surface area of the inscribed prism -/
theorem max_lateral_surface_area (p : InscribedPrism) : 
  lateralSurfaceArea p ≤ 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_lateral_surface_area_l3276_327610


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3276_327647

theorem polynomial_expansion (x : ℝ) :
  (7 * x^2 + 3) * (5 * x^3 + 4 * x + 1) = 35 * x^5 + 43 * x^3 + 7 * x^2 + 12 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3276_327647


namespace NUMINAMATH_CALUDE_subtracted_number_l3276_327635

theorem subtracted_number (a b x : ℕ) : 
  (a : ℚ) / b = 6 / 5 →
  (a - x : ℚ) / (b - x) = 5 / 4 →
  a - b = 5 →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_subtracted_number_l3276_327635


namespace NUMINAMATH_CALUDE_ratio_problem_l3276_327683

theorem ratio_problem (x : ℚ) : x / 8 = 6 / (4 * 60) ↔ x = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3276_327683


namespace NUMINAMATH_CALUDE_road_repair_length_l3276_327641

theorem road_repair_length : 
  ∀ (total_length : ℝ),
  (200 : ℝ) + 0.4 * total_length + 700 = total_length →
  total_length = 1500 := by
sorry

end NUMINAMATH_CALUDE_road_repair_length_l3276_327641


namespace NUMINAMATH_CALUDE_angle_q_approx_77_14_l3276_327631

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  /-- Angle P in degrees -/
  angleP : ℝ
  /-- Angle Q in degrees -/
  angleQ : ℝ
  /-- Angle R in degrees -/
  angleR : ℝ
  /-- The sum of all angles is 180° -/
  angle_sum : angleP + angleQ + angleR = 180
  /-- Angles Q and R are congruent -/
  qr_congruent : angleQ = angleR
  /-- Angle R is three times angle P -/
  r_triple_p : angleR = 3 * angleP

/-- The measure of angle Q in the isosceles triangle -/
def angle_q_measure (t : IsoscelesTriangle) : ℝ := t.angleQ

/-- Theorem: The measure of angle Q is approximately 77.14° -/
theorem angle_q_approx_77_14 (t : IsoscelesTriangle) :
  abs (angle_q_measure t - 540 / 7) < 0.01 := by
  sorry

#eval (540 : ℚ) / 7

end NUMINAMATH_CALUDE_angle_q_approx_77_14_l3276_327631


namespace NUMINAMATH_CALUDE_min_socks_for_pairs_l3276_327667

/-- Represents a sock with a color -/
inductive Sock
| Blue
| Red

/-- Represents a drawer containing socks -/
structure Drawer where
  socks : List Sock
  blue_count : Nat
  red_count : Nat
  balanced : blue_count = red_count

/-- Checks if a list of socks contains a pair of the same color -/
def hasSameColorPair (socks : List Sock) : Bool :=
  sorry

/-- Checks if a list of socks contains a pair of different colors -/
def hasDifferentColorPair (socks : List Sock) : Bool :=
  sorry

/-- Theorem stating the minimum number of socks required -/
theorem min_socks_for_pairs (d : Drawer) :
  (∀ n : Nat, n < 4 → ¬(∀ subset : List Sock, subset.length = n →
    (hasSameColorPair subset ∧ hasDifferentColorPair subset))) ∧
  (∃ subset : List Sock, subset.length = 4 ∧
    (hasSameColorPair subset ∧ hasDifferentColorPair subset)) :=
  sorry

end NUMINAMATH_CALUDE_min_socks_for_pairs_l3276_327667


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l3276_327625

theorem shaded_fraction_of_rectangle (length width : ℕ) (h1 : length = 15) (h2 : width = 24) :
  let total_area := length * width
  let third_area := total_area / 3
  let shaded_area := third_area / 3
  (shaded_area : ℚ) / total_area = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l3276_327625


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3276_327645

theorem sufficient_not_necessary (x : ℝ) :
  (x > 2 → abs (x - 1) > 1) ∧ ¬(abs (x - 1) > 1 → x > 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3276_327645


namespace NUMINAMATH_CALUDE_sum_seven_is_thirtyfive_l3276_327699

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_property : a 2 + a 10 = 16
  eighth_term : a 8 = 11

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n * (seq.a 1 + seq.a n)) / 2

/-- The main theorem to prove -/
theorem sum_seven_is_thirtyfive (seq : ArithmeticSequence) : 
  sum_n seq 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_is_thirtyfive_l3276_327699


namespace NUMINAMATH_CALUDE_simplify_sqrt_450_l3276_327670

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_450_l3276_327670


namespace NUMINAMATH_CALUDE_number_added_after_doubling_l3276_327681

theorem number_added_after_doubling (x : ℕ) (y : ℕ) (h : x = 13) :
  3 * (2 * x + y) = 99 → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_added_after_doubling_l3276_327681


namespace NUMINAMATH_CALUDE_complex_power_series_sum_l3276_327665

def complex_power_sequence (n : ℕ) : ℂ := (2 + Complex.I) ^ n

def real_part_sequence (n : ℕ) : ℝ := (complex_power_sequence n).re
def imag_part_sequence (n : ℕ) : ℝ := (complex_power_sequence n).im

theorem complex_power_series_sum :
  (∑' n, (real_part_sequence n * imag_part_sequence n) / 7 ^ n) = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_series_sum_l3276_327665


namespace NUMINAMATH_CALUDE_midpoint_sum_and_product_l3276_327692

/-- Given a line segment with endpoints (8, 15) and (-2, -3), 
    prove that the sum of the coordinates of the midpoint is 9 
    and the product of the coordinates of the midpoint is 18. -/
theorem midpoint_sum_and_product : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := 15
  let x₂ : ℝ := -2
  let y₂ : ℝ := -3
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  (midpoint_x + midpoint_y = 9) ∧ (midpoint_x * midpoint_y = 18) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_and_product_l3276_327692


namespace NUMINAMATH_CALUDE_not_perfect_square_with_digit_sum_2006_l3276_327608

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem not_perfect_square_with_digit_sum_2006 (n : ℕ) 
  (h : sum_of_digits n = 2006) : 
  ¬ ∃ (m : ℕ), n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_with_digit_sum_2006_l3276_327608


namespace NUMINAMATH_CALUDE_rectangle_and_triangle_l3276_327611

/-- Given a rectangle ABCD and an isosceles right triangle DCE, prove that DE = 4√3 -/
theorem rectangle_and_triangle (AB AD DC DE : ℝ) : 
  AB = 6 →
  AD = 8 →
  DC = DE →
  AB * AD = 2 * (1/2 * DC * DE) →
  DE = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_and_triangle_l3276_327611


namespace NUMINAMATH_CALUDE_max_self_intersections_l3276_327627

/-- A closed six-segment broken line with vertices on a circle -/
structure BrokenLine where
  vertices : Fin 6 → ℝ × ℝ
  on_circle : ∀ i, (vertices i).1^2 + (vertices i).2^2 = 1

/-- The number of self-intersections in a broken line -/
def num_self_intersections (bl : BrokenLine) : ℕ := sorry

/-- Theorem: The maximum number of self-intersections is 7 -/
theorem max_self_intersections (bl : BrokenLine) :
  num_self_intersections bl ≤ 7 := by sorry

end NUMINAMATH_CALUDE_max_self_intersections_l3276_327627


namespace NUMINAMATH_CALUDE_ratio_difference_bound_l3276_327679

theorem ratio_difference_bound (a : Fin 5 → ℝ) (h : ∀ i, 0 < a i) :
  ∃ i j k l : Fin 5, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    |a i / a j - a k / a l| < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_bound_l3276_327679


namespace NUMINAMATH_CALUDE_cleo_final_marbles_l3276_327634

def initial_marbles : ℕ := 240

def day2_fraction : ℚ := 2/3
def day2_people : ℕ := 3

def day3_fraction : ℚ := 3/5
def day3_people : ℕ := 2

def day4_cleo_fraction : ℚ := 7/8
def day4_estela_fraction : ℚ := 1/4

theorem cleo_final_marbles :
  let day2_marbles := (initial_marbles : ℚ) * day2_fraction
  let day2_per_person := ⌊day2_marbles / day2_people⌋
  let day3_remaining := initial_marbles - (day2_per_person * day2_people)
  let day3_marbles := (day3_remaining : ℚ) * day3_fraction
  let day3_cleo := ⌊day3_marbles / day3_people⌋
  let day4_cleo := ⌊(day3_cleo : ℚ) * day4_cleo_fraction⌋
  let day4_estela := ⌊(day4_cleo : ℚ) * day4_estela_fraction⌋
  day4_cleo - day4_estela = 16 := by sorry

end NUMINAMATH_CALUDE_cleo_final_marbles_l3276_327634


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3276_327619

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence condition
  (2 * a 2^2 - 7 * a 2 + 6 = 0) →  -- a_2 is a root of 2x^2 - 7x + 6 = 0
  (2 * a 8^2 - 7 * a 8 + 6 = 0) →  -- a_8 is a root of 2x^2 - 7x + 6 = 0
  (a 1 * a 3 * a 5 * a 7 * a 9 = 9 * Real.sqrt 3 ∨ 
   a 1 * a 3 * a 5 * a 7 * a 9 = -9 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3276_327619


namespace NUMINAMATH_CALUDE_pet_store_cages_l3276_327603

theorem pet_store_cages (initial_puppies : Nat) (sold_puppies : Nat) (puppies_per_cage : Nat) : 
  initial_puppies = 18 → sold_puppies = 3 → puppies_per_cage = 5 → 
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3276_327603


namespace NUMINAMATH_CALUDE_binomial_expansion_ratio_l3276_327695

theorem binomial_expansion_ratio : 
  let n : ℕ := 10
  let k : ℕ := 5
  let a : ℕ := Nat.choose n k
  let b : ℤ := -Nat.choose n 3 * (-2)^3
  (b : ℚ) / a = -80 / 21 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_ratio_l3276_327695


namespace NUMINAMATH_CALUDE_income_comparison_l3276_327640

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mary = tim * (1 + 0.4)) :
  mary = juan * 0.84 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l3276_327640


namespace NUMINAMATH_CALUDE_david_twice_rosy_age_l3276_327664

/-- Represents the current age of Rosy -/
def rosy_age : ℕ := 8

/-- Represents the current age difference between David and Rosy -/
def age_difference : ℕ := 12

/-- Calculates the number of years until David is twice Rosy's age -/
def years_until_double : ℕ :=
  let david_age := rosy_age + age_difference
  (david_age - 2 * rosy_age)

theorem david_twice_rosy_age : years_until_double = 4 := by
  sorry

end NUMINAMATH_CALUDE_david_twice_rosy_age_l3276_327664


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3276_327624

/-- Given a complex number z satisfying (1+2i)z=4+3i, prove that z is located in the fourth quadrant of the complex plane -/
theorem z_in_fourth_quadrant (z : ℂ) (h : (1 + 2*Complex.I)*z = 4 + 3*Complex.I) : 
  Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3276_327624


namespace NUMINAMATH_CALUDE_sphere_deflation_radius_l3276_327654

theorem sphere_deflation_radius (r : ℝ) (h : r = 4) :
  let hemisphere_volume := (2/3) * Real.pi * r^3
  let original_sphere_volume := (4/3) * Real.pi * (((4 * Real.rpow 2 (1/3)) / Real.rpow 3 (1/3))^3)
  hemisphere_volume = (3/4) * original_sphere_volume :=
by sorry

end NUMINAMATH_CALUDE_sphere_deflation_radius_l3276_327654


namespace NUMINAMATH_CALUDE_cassidy_poster_addition_l3276_327609

/-- The number of posters Cassidy had two years ago -/
def posters_two_years_ago : ℕ := 14

/-- The number of posters Cassidy has currently -/
def current_posters : ℕ := 22

/-- The number of posters Cassidy will have after this summer -/
def future_posters : ℕ := 2 * posters_two_years_ago

/-- The number of posters Cassidy will add this summer -/
def posters_to_add : ℕ := future_posters - current_posters

theorem cassidy_poster_addition : posters_to_add = 6 := by
  sorry

end NUMINAMATH_CALUDE_cassidy_poster_addition_l3276_327609


namespace NUMINAMATH_CALUDE_joans_cake_eggs_l3276_327602

/-- The number of eggs needed for baking cakes -/
def total_eggs (vanilla_count chocolate_count carrot_count : ℕ) 
               (vanilla_eggs chocolate_eggs carrot_eggs : ℕ) : ℕ :=
  vanilla_count * vanilla_eggs + chocolate_count * chocolate_eggs + carrot_count * carrot_eggs

/-- Theorem stating the total number of eggs needed for Joan's cakes -/
theorem joans_cake_eggs : 
  total_eggs 5 4 3 8 6 10 = 94 := by
  sorry

end NUMINAMATH_CALUDE_joans_cake_eggs_l3276_327602


namespace NUMINAMATH_CALUDE_triangle_division_exists_l3276_327666

/-- Represents a part of the triangle -/
structure TrianglePart where
  numbers : List Nat
  sum : Nat

/-- Represents the entire triangle -/
structure Triangle where
  total_sum : Nat
  parts : List TrianglePart

/-- Checks if a triangle is valid according to the problem conditions -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.total_sum = 63 ∧
  t.parts.length = 3 ∧
  (∀ p ∈ t.parts, p.sum = p.numbers.sum) ∧
  (∀ p ∈ t.parts, p.sum = t.total_sum / 3) ∧
  (t.parts.map (·.numbers)).join.sum = t.total_sum

theorem triangle_division_exists :
  ∃ t : Triangle, is_valid_triangle t :=
sorry

end NUMINAMATH_CALUDE_triangle_division_exists_l3276_327666


namespace NUMINAMATH_CALUDE_stating_min_time_to_find_faulty_bulb_l3276_327690

/-- Represents the time in seconds for a single bulb operation (screwing or unscrewing) -/
def bulb_operation_time : ℕ := 10

/-- Represents the total number of bulbs in the series -/
def total_bulbs : ℕ := 4

/-- Represents the number of spare bulbs available -/
def spare_bulbs : ℕ := 1

/-- Represents the number of faulty bulbs in the series -/
def faulty_bulbs : ℕ := 1

/-- 
Theorem stating that the minimum time to identify a faulty bulb 
in a series of 4 bulbs is 60 seconds, given the conditions of the problem.
-/
theorem min_time_to_find_faulty_bulb : 
  (bulb_operation_time * 2 * (total_bulbs - 1) : ℕ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_stating_min_time_to_find_faulty_bulb_l3276_327690


namespace NUMINAMATH_CALUDE_range_of_a_l3276_327637

-- Define the proposition
def proposition (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 + 2*x - a ≥ 0

-- State the theorem
theorem range_of_a (h : ∀ a : ℝ, proposition a ↔ a ∈ Set.Iic 15) :
  {a : ℝ | proposition a} = Set.Iic 15 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3276_327637


namespace NUMINAMATH_CALUDE_unique_number_l3276_327660

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a natural number has exactly two prime factors -/
def hasTwoPrimeFactors (n : ℕ) : Prop :=
  ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p ≠ q ∧ n = p * q

/-- A function that checks if a number doesn't contain the digit 7 -/
def noSeven (n : ℕ) : Prop :=
  ∀ d : ℕ, d < n → (n / (10^d)) % 10 ≠ 7

theorem unique_number : 
  ∃! n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧  -- four-digit number
    noSeven n ∧             -- doesn't contain 7
    hasTwoPrimeFactors n ∧  -- product of exactly two primes
    ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ n = p * q ∧ q = p + 4 ∧  -- prime factors differ by 4
    n = 2021                -- the number is 2021
  := by sorry

end NUMINAMATH_CALUDE_unique_number_l3276_327660


namespace NUMINAMATH_CALUDE_sandro_children_l3276_327689

/-- Calculates the total number of children for a person with a given number of sons
    and a ratio of daughters to sons. -/
def totalChildren (numSons : ℕ) (daughterToSonRatio : ℕ) : ℕ :=
  numSons + numSons * daughterToSonRatio

/-- Theorem stating that for a person with 3 sons and 6 times as many daughters as sons,
    the total number of children is 21. -/
theorem sandro_children :
  totalChildren 3 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sandro_children_l3276_327689


namespace NUMINAMATH_CALUDE_student_selection_l3276_327674

theorem student_selection (male_count : Nat) (female_count : Nat) :
  male_count = 5 →
  female_count = 4 →
  (Nat.choose (male_count + female_count) 3 -
   Nat.choose male_count 3 -
   Nat.choose female_count 3) = 70 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_l3276_327674


namespace NUMINAMATH_CALUDE_aunt_angela_nephews_l3276_327626

theorem aunt_angela_nephews (total_jellybeans : ℕ) (jellybeans_per_child : ℕ) (num_nieces : ℕ) :
  total_jellybeans = 70 →
  jellybeans_per_child = 14 →
  num_nieces = 2 →
  total_jellybeans = (num_nieces + 3) * jellybeans_per_child :=
by sorry

end NUMINAMATH_CALUDE_aunt_angela_nephews_l3276_327626


namespace NUMINAMATH_CALUDE_cheesecake_eggs_proof_l3276_327607

/-- The number of eggs needed for each chocolate cake -/
def chocolate_cake_eggs : ℕ := 3

/-- The number of eggs needed for each cheesecake -/
def cheesecake_eggs : ℕ := 8

/-- Proof that the number of eggs for each cheesecake is 8 -/
theorem cheesecake_eggs_proof : 
  9 * cheesecake_eggs = 5 * chocolate_cake_eggs + 57 :=
by sorry

end NUMINAMATH_CALUDE_cheesecake_eggs_proof_l3276_327607


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3276_327693

/-- Given a quadratic function y = x^2 + px + q + r where the minimum value is -r, 
    prove that q = p^2 / 4 -/
theorem quadratic_minimum (p q r : ℝ) : 
  (∀ x, x^2 + p*x + q + r ≥ -r) → 
  (∃ x, x^2 + p*x + q + r = -r) → 
  q = p^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3276_327693


namespace NUMINAMATH_CALUDE_childrens_tickets_l3276_327671

theorem childrens_tickets (adult_price child_price total_tickets total_cost : ℚ)
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_tickets child_tickets : ℚ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_cost ∧
    child_tickets = 16 := by
  sorry

end NUMINAMATH_CALUDE_childrens_tickets_l3276_327671


namespace NUMINAMATH_CALUDE_bryden_quarters_value_l3276_327668

/-- The face value of a regular quarter in dollars -/
def regular_quarter_value : ℚ := 1/4

/-- The number of regular quarters Bryden has -/
def regular_quarters : ℕ := 4

/-- The number of special quarters Bryden has -/
def special_quarters : ℕ := 1

/-- The value multiplier for a special quarter compared to a regular quarter -/
def special_quarter_multiplier : ℚ := 2

/-- The percentage of face value the collector offers -/
def collector_offer_percentage : ℚ := 1500

theorem bryden_quarters_value :
  let total_face_value := regular_quarter_value * regular_quarters +
                          regular_quarter_value * special_quarter_multiplier * special_quarters
  let collector_offer_multiplier := collector_offer_percentage / 100
  collector_offer_multiplier * total_face_value = 45/2 :=
sorry

end NUMINAMATH_CALUDE_bryden_quarters_value_l3276_327668


namespace NUMINAMATH_CALUDE_absolute_value_properties_l3276_327614

theorem absolute_value_properties :
  (∀ a : ℚ, a = 5 → |a| / a = 1) ∧
  (∀ a : ℚ, a = -2 → a / |a| = -1) ∧
  (∀ a b : ℚ, a * b > 0 → a / |a| + |b| / b = 2 ∨ a / |a| + |b| / b = -2) ∧
  (∀ a b c : ℚ, a * b * c < 0 → 
    a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = 0 ∨
    a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = -4) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_properties_l3276_327614


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3276_327604

theorem product_of_three_numbers (a b c m : ℚ) : 
  a + b + c = 200 ∧ 
  8 * a = m ∧ 
  b = m + 10 ∧ 
  c = m - 10 →
  a * b * c = 505860000 / 4913 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3276_327604


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l3276_327657

theorem complex_equation_solutions :
  let f : ℂ → ℂ := λ z => (z^4 - 1) / (z^3 + z^2 - 2*z)
  ∃! (S : Finset ℂ), S.card = 3 ∧ ∀ z ∈ S, f z = 0 ∧ ∀ z ∉ S, f z ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l3276_327657


namespace NUMINAMATH_CALUDE_pears_equivalent_to_24_bananas_is_12_l3276_327646

/-- The number of pears equivalent in cost to 24 bananas -/
def pears_equivalent_to_24_bananas (banana_apple_ratio : ℚ) (apple_pear_ratio : ℚ) : ℚ :=
  24 * banana_apple_ratio * apple_pear_ratio

theorem pears_equivalent_to_24_bananas_is_12 :
  pears_equivalent_to_24_bananas (3/4) (6/9) = 12 := by
  sorry

end NUMINAMATH_CALUDE_pears_equivalent_to_24_bananas_is_12_l3276_327646


namespace NUMINAMATH_CALUDE_juice_theorem_l3276_327687

def juice_problem (sam_initial ben_initial sam_consumed ben_consumed sam_received : ℚ) : Prop :=
  let sam_final := sam_consumed + sam_received
  let ben_final := ben_consumed - sam_received
  sam_initial = 12 ∧
  ben_initial = sam_initial + 8 ∧
  sam_consumed = 2 / 3 * sam_initial ∧
  ben_consumed = 2 / 3 * ben_initial ∧
  sam_received = (1 / 2 * (ben_initial - ben_consumed)) + 1 ∧
  sam_final = ben_final ∧
  sam_initial + ben_initial = 32

theorem juice_theorem :
  ∃ (sam_initial ben_initial sam_consumed ben_consumed sam_received : ℚ),
    juice_problem sam_initial ben_initial sam_consumed ben_consumed sam_received :=
by
  sorry

#check juice_theorem

end NUMINAMATH_CALUDE_juice_theorem_l3276_327687


namespace NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l3276_327613

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the focus
def line_through_focus (m b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + b ∧ focus.2 = m * focus.1 + b

-- Define the intersection points of the line and the parabola
def intersection_points (m b : ℝ) : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line_through_focus m b p}

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (m b : ℝ) 
  (M N : ℝ × ℝ) 
  (h_M : M ∈ intersection_points m b) 
  (h_N : N ∈ intersection_points m b) 
  (h_distinct : M ≠ N) :
  let midpoint := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  midpoint.1 = 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l3276_327613


namespace NUMINAMATH_CALUDE_teaching_arrangements_l3276_327644

-- Define the number of classes
def num_classes : ℕ := 4

-- Define the number of Chinese teachers
def num_chinese_teachers : ℕ := 2

-- Define the number of math teachers
def num_math_teachers : ℕ := 2

-- Define the number of classes each teacher teaches
def classes_per_teacher : ℕ := 2

-- Theorem statement
theorem teaching_arrangements :
  (Nat.choose num_classes classes_per_teacher) * (Nat.choose num_classes classes_per_teacher) = 36 := by
  sorry

end NUMINAMATH_CALUDE_teaching_arrangements_l3276_327644


namespace NUMINAMATH_CALUDE_cuboid_s_value_l3276_327698

/-- Represents a cuboid with adjacent face areas a, b, and s, 
    whose vertices lie on a sphere with surface area sa -/
structure Cuboid where
  a : ℝ
  b : ℝ
  s : ℝ
  sa : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < s ∧ 0 < sa
  h_sphere : sa = 152 * Real.pi
  h_face1 : a * b = 6
  h_face2 : b * (s / b) = 10
  h_vertices_on_sphere : ∃ (r : ℝ), 
    a^2 + b^2 + (s / b)^2 = 4 * r^2 ∧ sa = 4 * Real.pi * r^2

/-- The theorem stating that for a cuboid satisfying the given conditions, s must equal 15 -/
theorem cuboid_s_value (c : Cuboid) : c.s = 15 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_s_value_l3276_327698


namespace NUMINAMATH_CALUDE_sum_of_non_solutions_l3276_327623

/-- Given an equation with infinitely many solutions, prove the sum of non-solutions -/
theorem sum_of_non_solutions (A B C : ℝ) : 
  (∀ x : ℝ, (x + B) * (A * x + 28) = 2 * (x + C) * (x + 7)) →
  (∃ S : Finset ℝ, (∀ x ∉ S, (x + B) * (A * x + 28) = 2 * (x + C) * (x + 7)) ∧ 
    (∀ x ∈ S, (x + B) * (A * x + 28) ≠ 2 * (x + C) * (x + 7)) ∧
    (Finset.sum S id = -21)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_non_solutions_l3276_327623


namespace NUMINAMATH_CALUDE_problem_paths_l3276_327632

/-- Represents the number of ways to reach a specific arrow type -/
structure ArrowPaths where
  count : Nat
  arrows : Nat

/-- The modified hexagonal lattice structure -/
structure HexLattice where
  redPaths : ArrowPaths
  bluePaths : ArrowPaths
  greenPaths : ArrowPaths
  endPaths : Nat

/-- The specific hexagonal lattice in the problem -/
def problemLattice : HexLattice :=
  { redPaths := { count := 1, arrows := 1 }
    bluePaths := { count := 3, arrows := 2 }
    greenPaths := { count := 6, arrows := 2 }
    endPaths := 4 }

/-- Calculates the total number of paths in the lattice -/
def totalPaths (lattice : HexLattice) : Nat :=
  lattice.redPaths.count *
  lattice.bluePaths.count * lattice.bluePaths.arrows *
  lattice.greenPaths.count * lattice.greenPaths.arrows *
  lattice.endPaths

theorem problem_paths :
  totalPaths problemLattice = 288 := by sorry

end NUMINAMATH_CALUDE_problem_paths_l3276_327632


namespace NUMINAMATH_CALUDE_circle_ratio_l3276_327616

/-- For a circle with diameter 100 cm and circumference 314 cm, 
    the ratio of circumference to diameter is 3.14 -/
theorem circle_ratio : 
  ∀ (diameter circumference : ℝ), 
    diameter = 100 → 
    circumference = 314 → 
    circumference / diameter = 3.14 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l3276_327616


namespace NUMINAMATH_CALUDE_photographer_theorem_l3276_327684

/-- Represents the number of birds of each species -/
structure BirdCount where
  starlings : Nat
  wagtails : Nat
  woodpeckers : Nat

/-- The initial bird count -/
def initial_birds : BirdCount :=
  { starlings := 8, wagtails := 7, woodpeckers := 5 }

/-- The total number of birds -/
def total_birds : Nat := 20

/-- The number of photos to be taken -/
def photos_taken : Nat := 7

/-- Predicate to check if the remaining birds meet the condition -/
def meets_condition (b : BirdCount) : Prop :=
  (b.starlings ≥ 4 ∧ (b.wagtails ≥ 3 ∨ b.woodpeckers ≥ 3)) ∨
  (b.wagtails ≥ 4 ∧ (b.starlings ≥ 3 ∨ b.woodpeckers ≥ 3)) ∨
  (b.woodpeckers ≥ 4 ∧ (b.starlings ≥ 3 ∨ b.wagtails ≥ 3))

theorem photographer_theorem :
  ∀ (remaining : BirdCount),
    remaining.starlings + remaining.wagtails + remaining.woodpeckers = total_birds - photos_taken →
    remaining.starlings ≤ initial_birds.starlings →
    remaining.wagtails ≤ initial_birds.wagtails →
    remaining.woodpeckers ≤ initial_birds.woodpeckers →
    meets_condition remaining :=
by
  sorry

end NUMINAMATH_CALUDE_photographer_theorem_l3276_327684


namespace NUMINAMATH_CALUDE_ring_cost_l3276_327688

theorem ring_cost (total_cost : ℝ) (num_rings : ℕ) (h1 : total_cost = 24) (h2 : num_rings = 2) :
  total_cost / num_rings = 12 :=
by sorry

end NUMINAMATH_CALUDE_ring_cost_l3276_327688


namespace NUMINAMATH_CALUDE_impossible_transformation_l3276_327617

/-- Represents a binary sequence -/
inductive BinarySeq
| empty : BinarySeq
| cons : Bool → BinarySeq → BinarySeq

/-- Represents the color of a digit in the sequence -/
inductive Color
| Red
| Green
| Blue

/-- Assigns colors to a binary sequence -/
def colorSequence : BinarySeq → List Color
| BinarySeq.empty => []
| BinarySeq.cons _ rest => [Color.Red, Color.Green, Color.Blue] ++ colorSequence rest

/-- Counts the number of red 1s in a colored binary sequence -/
def countRed1s : BinarySeq → Nat
| BinarySeq.empty => 0
| BinarySeq.cons true (BinarySeq.cons _ (BinarySeq.cons _ rest)) => 1 + countRed1s rest
| BinarySeq.cons false (BinarySeq.cons _ (BinarySeq.cons _ rest)) => countRed1s rest
| _ => 0

/-- Represents an operation on the binary sequence -/
inductive Operation
| Insert : BinarySeq → Operation
| Delete : BinarySeq → Operation

/-- Applies an operation to a binary sequence -/
def applyOperation : BinarySeq → Operation → BinarySeq := sorry

/-- Theorem: It's impossible to transform "10" into "01" using the allowed operations -/
theorem impossible_transformation :
  ∀ (ops : List Operation),
    let initial := BinarySeq.cons true (BinarySeq.cons false BinarySeq.empty)
    let final := BinarySeq.cons false (BinarySeq.cons true BinarySeq.empty)
    let result := ops.foldl applyOperation initial
    result ≠ final :=
sorry

end NUMINAMATH_CALUDE_impossible_transformation_l3276_327617


namespace NUMINAMATH_CALUDE_sum_of_roots_inequality_l3276_327642

theorem sum_of_roots_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq_one : a + b + c = 1) :
  Real.sqrt ((1 / a) - 1) * Real.sqrt ((1 / b) - 1) +
  Real.sqrt ((1 / b) - 1) * Real.sqrt ((1 / c) - 1) +
  Real.sqrt ((1 / c) - 1) * Real.sqrt ((1 / a) - 1) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_inequality_l3276_327642


namespace NUMINAMATH_CALUDE_stone_slab_length_l3276_327606

theorem stone_slab_length (num_slabs : ℕ) (total_area : ℝ) (slab_length : ℝ) :
  num_slabs = 50 →
  total_area = 98 →
  num_slabs * (slab_length ^ 2) = total_area →
  slab_length = 1.4 :=
by
  sorry

#check stone_slab_length

end NUMINAMATH_CALUDE_stone_slab_length_l3276_327606


namespace NUMINAMATH_CALUDE_rational_equation_solutions_l3276_327643

theorem rational_equation_solutions (a b : ℚ) :
  (∃ x y : ℚ, a * x^2 + b * y^2 = 1) →
  (∀ n : ℕ, ∃ (x₁ y₁ : ℚ) (x₂ y₂ : ℚ), 
    (a * x₁^2 + b * y₁^2 = 1) ∧ 
    (a * x₂^2 + b * y₂^2 = 1) ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solutions_l3276_327643


namespace NUMINAMATH_CALUDE_vertical_line_properties_l3276_327694

/-- A line passing through two points with the same x-coordinate but different y-coordinates has an undefined slope and its x-intercept is equal to the common x-coordinate. -/
theorem vertical_line_properties (x y₁ y₂ : ℝ) (h : y₁ ≠ y₂) :
  let C : ℝ × ℝ := (x, y₁)
  let D : ℝ × ℝ := (x, y₂)
  let line := {P : ℝ × ℝ | ∃ t : ℝ, P = (1 - t) • C + t • D}
  (∀ P Q : ℝ × ℝ, P ∈ line → Q ∈ line → P.1 ≠ Q.1 → (Q.2 - P.2) / (Q.1 - P.1) = (0 : ℝ)/0) ∧
  (∃ y : ℝ, (x, y) ∈ line) :=
by sorry

end NUMINAMATH_CALUDE_vertical_line_properties_l3276_327694


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3276_327658

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define the foci of the ellipse
def foci (F₁ F₂ : ℝ × ℝ) : Prop := ∃ c : ℝ, c^2 = 7 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the hyperbola
def on_hyperbola (P : ℝ × ℝ) : Prop := ∃ a b : ℝ, a^2 - b^2 = 1 ∧ (P.1^2/a^2) - (P.2^2/b^2) = 1

-- Define perpendicularity of PF₁ and PF₂
def perpendicular (P F₁ F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

-- Define the product condition
def product_condition (P F₁ F₂ : ℝ × ℝ) : Prop :=
  ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 4

-- Theorem statement
theorem hyperbola_equation (P F₁ F₂ : ℝ × ℝ) :
  foci F₁ F₂ →
  on_hyperbola P →
  perpendicular P F₁ F₂ →
  product_condition P F₁ F₂ →
  ∃ x y : ℝ, P = (x, y) ∧ x^2/6 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3276_327658


namespace NUMINAMATH_CALUDE_yellow_score_mixture_l3276_327652

theorem yellow_score_mixture (white black : ℕ) : 
  white * 6 = black * 7 →
  2 * (white - black) = 3 * 4 →
  white + black = 78 := by
sorry

end NUMINAMATH_CALUDE_yellow_score_mixture_l3276_327652
