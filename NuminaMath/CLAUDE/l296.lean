import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l296_29640

/-- A quadratic function of the form y = kx^2 - 7x - 7 -/
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 7 * x - 7

/-- The discriminant of the quadratic function -/
def discriminant (k : ℝ) : ℝ := 49 + 28 * k

/-- Theorem stating the conditions for the quadratic function to intersect the x-axis -/
theorem quadratic_intersects_x_axis (k : ℝ) :
  (∃ x, quadratic_function k x = 0) ↔ (k ≥ -7/4 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l296_29640


namespace NUMINAMATH_CALUDE_passes_through_first_and_third_quadrants_l296_29633

def proportional_function (x : ℝ) : ℝ := x

theorem passes_through_first_and_third_quadrants :
  (∀ x : ℝ, x > 0 → proportional_function x > 0) ∧
  (∀ x : ℝ, x < 0 → proportional_function x < 0) :=
by sorry

end NUMINAMATH_CALUDE_passes_through_first_and_third_quadrants_l296_29633


namespace NUMINAMATH_CALUDE_calculation_proof_l296_29683

theorem calculation_proof : 
  (Real.sqrt 18) / 3 + |Real.sqrt 2 - 2| + 2023^0 - (-1)^1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l296_29683


namespace NUMINAMATH_CALUDE_seven_digit_multiple_l296_29661

theorem seven_digit_multiple : ∀ (A B C : ℕ),
  (A < 10 ∧ B < 10 ∧ C < 10) →
  (∃ (k₁ k₂ k₃ : ℕ), 
    25000000 + A * 100000 + B * 10000 + 3300 + C = 8 * k₁ ∧
    25000000 + A * 100000 + B * 10000 + 3300 + C = 9 * k₂ ∧
    25000000 + A * 100000 + B * 10000 + 3300 + C = 11 * k₃) →
  A + B + C = 14 := by
sorry

end NUMINAMATH_CALUDE_seven_digit_multiple_l296_29661


namespace NUMINAMATH_CALUDE_costume_cost_is_660_l296_29627

/-- Represents the cost of materials for Jenna's costume --/
def costume_cost : ℝ :=
  let velvet_price := 3
  let silk_price := 6
  let lace_price := 10
  let satin_price := 4
  let leather_price := 5
  let wool_price := 8
  let ribbon_price := 2

  let skirt_area := 12 * 4
  let skirts_count := 3
  let bodice_silk_area := 2
  let bodice_lace_area := 5 * 2
  let bonnet_area := 2.5 * 1.5
  let shoe_cover_area := 1 * 1.5 * 2
  let cape_area := 5 * 2
  let ribbon_length := 3

  let velvet_cost := velvet_price * skirt_area * skirts_count
  let bodice_cost := silk_price * bodice_silk_area + lace_price * bodice_lace_area
  let bonnet_cost := satin_price * bonnet_area
  let shoe_covers_cost := leather_price * shoe_cover_area
  let cape_cost := wool_price * cape_area
  let ribbon_cost := ribbon_price * ribbon_length

  velvet_cost + bodice_cost + bonnet_cost + shoe_covers_cost + cape_cost + ribbon_cost

/-- Theorem stating that the total cost of Jenna's costume materials is $660 --/
theorem costume_cost_is_660 : costume_cost = 660 := by
  sorry

end NUMINAMATH_CALUDE_costume_cost_is_660_l296_29627


namespace NUMINAMATH_CALUDE_sugar_weight_loss_fraction_l296_29634

theorem sugar_weight_loss_fraction (green_beans_weight sugar_weight rice_weight remaining_weight : ℝ) :
  green_beans_weight = 60 →
  rice_weight = green_beans_weight - 30 →
  sugar_weight = green_beans_weight - 10 →
  remaining_weight = 120 →
  (green_beans_weight + (2/3 * rice_weight) + sugar_weight - remaining_weight) / sugar_weight = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sugar_weight_loss_fraction_l296_29634


namespace NUMINAMATH_CALUDE_cube_split_31_l296_29639

/-- 
Given a natural number m > 1, returns the sequence of consecutive odd numbers 
that sum to m^3, starting from 2m - 1
-/
def cubeOddSequence (m : ℕ) : List ℕ := sorry

/-- 
Theorem: If 31 is in the sequence of odd numbers that sum to m^3 for m > 1, 
then m = 6
-/
theorem cube_split_31 (m : ℕ) (h1 : m > 1) : 
  31 ∈ cubeOddSequence m → m = 6 := by sorry

end NUMINAMATH_CALUDE_cube_split_31_l296_29639


namespace NUMINAMATH_CALUDE_sum_difference_inequality_l296_29671

theorem sum_difference_inequality 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0) 
  (hb : b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0) 
  (ha_sum : a₁*a₁ + a₁*a₂ + a₁*a₃ + a₂*a₂ + a₂*a₃ + a₃*a₃ ≤ 1) 
  (hb_sum : b₁*b₁ + b₁*b₂ + b₁*b₃ + b₂*b₂ + b₂*b₃ + b₃*b₃ ≤ 1) : 
  (a₁-b₁)*(a₁-b₁) + (a₁-b₁)*(a₂-b₂) + (a₁-b₁)*(a₃-b₃) + 
  (a₂-b₂)*(a₂-b₂) + (a₂-b₂)*(a₃-b₃) + (a₃-b₃)*(a₃-b₃) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_inequality_l296_29671


namespace NUMINAMATH_CALUDE_distribute_six_among_four_l296_29680

/-- The number of ways to distribute n indistinguishable objects among k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 84 ways to distribute 6 objects among 4 containers -/
theorem distribute_six_among_four : distribute 6 4 = 84 := by sorry

end NUMINAMATH_CALUDE_distribute_six_among_four_l296_29680


namespace NUMINAMATH_CALUDE_integer_solution_abc_l296_29604

theorem integer_solution_abc : ∀ a b c : ℕ,
  1 < a ∧ a < b ∧ b < c ∧ (abc - 1) % ((a - 1) * (b - 1) * (c - 1)) = 0 →
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_abc_l296_29604


namespace NUMINAMATH_CALUDE_complete_square_equation_l296_29684

theorem complete_square_equation (x : ℝ) : 
  (x^2 - 8*x + 15 = 0) ↔ ((x - 4)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_complete_square_equation_l296_29684


namespace NUMINAMATH_CALUDE_junk_food_ratio_l296_29642

theorem junk_food_ratio (weekly_allowance sweets_cost savings : ℚ)
  (h1 : weekly_allowance = 30)
  (h2 : sweets_cost = 8)
  (h3 : savings = 12)
  (h4 : weekly_allowance = sweets_cost + savings + (weekly_allowance - sweets_cost - savings)) :
  (weekly_allowance - sweets_cost - savings) / weekly_allowance = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_junk_food_ratio_l296_29642


namespace NUMINAMATH_CALUDE_benny_piggy_bank_l296_29608

theorem benny_piggy_bank (january_savings : ℕ) (february_savings : ℕ) (total_savings : ℕ) : 
  january_savings = 19 →
  february_savings = 19 →
  total_savings = 46 →
  total_savings - (january_savings + february_savings) = 8 := by
sorry

end NUMINAMATH_CALUDE_benny_piggy_bank_l296_29608


namespace NUMINAMATH_CALUDE_square_side_length_l296_29662

-- Define the right triangle PQR
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the square on the hypotenuse
structure SquareOnHypotenuse where
  triangle : RightTriangle
  side_length : ℝ
  on_hypotenuse : side_length ≤ triangle.hypotenuse
  vertex_on_legs : ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ triangle.leg1 ∧ 0 ≤ y ∧ y ≤ triangle.leg2 ∧
    x^2 + y^2 = side_length^2

-- Theorem statement
theorem square_side_length (t : RightTriangle) (s : SquareOnHypotenuse) 
  (h1 : t.leg1 = 5) (h2 : t.leg2 = 12) (h3 : s.triangle = t) :
  s.side_length = 480.525 / 101.925 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l296_29662


namespace NUMINAMATH_CALUDE_equation_solution_l296_29646

theorem equation_solution : ∃! x : ℝ, 5 * 5^x + Real.sqrt (25 * 25^x) = 50 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l296_29646


namespace NUMINAMATH_CALUDE_expanded_identity_properties_l296_29696

theorem expanded_identity_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -243) ∧
  (a₀ + a₂ + a₄ = -121) := by
  sorry

end NUMINAMATH_CALUDE_expanded_identity_properties_l296_29696


namespace NUMINAMATH_CALUDE_negation_of_proposition_l296_29692

theorem negation_of_proposition (a : ℝ) :
  (¬ ∀ x : ℝ, x ≥ 0 → x^2 - a*x + 3 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 - a*x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l296_29692


namespace NUMINAMATH_CALUDE_stone_counting_l296_29687

/-- Represents the number of stones in the sequence -/
def num_stones : ℕ := 11

/-- Represents the length of a complete counting cycle -/
def cycle_length : ℕ := 2 * num_stones - 2

/-- The number we want to find the corresponding stone for -/
def target_number : ℕ := 123

/-- The initial number of the stone that corresponds to the target number -/
def corresponding_stone : ℕ := 3

theorem stone_counting (n : ℕ) :
  n % cycle_length = corresponding_stone - 1 →
  ∃ (k : ℕ), n = k * cycle_length + corresponding_stone :=
sorry

end NUMINAMATH_CALUDE_stone_counting_l296_29687


namespace NUMINAMATH_CALUDE_f_neg_ten_eq_two_l296_29644

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else x + 12

-- State the theorem
theorem f_neg_ten_eq_two : f (-10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_ten_eq_two_l296_29644


namespace NUMINAMATH_CALUDE_shopkeeper_profit_loss_l296_29611

theorem shopkeeper_profit_loss (cost : ℝ) : 
  cost > 0 →
  let profit_percent := 10
  let loss_percent := 10
  let selling_price1 := cost * (1 + profit_percent / 100)
  let selling_price2 := cost * (1 - loss_percent / 100)
  let total_cost := 2 * cost
  let total_selling_price := selling_price1 + selling_price2
  (total_selling_price - total_cost) / total_cost * 100 = 0 :=
by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_loss_l296_29611


namespace NUMINAMATH_CALUDE_last_card_identifiable_determine_last_card_back_l296_29610

/-- Represents a card with two sides -/
structure Card where
  front : ℕ
  back : ℕ

/-- Creates a deck of n cards -/
def create_deck (n : ℕ) : List Card :=
  List.range n |>.map (λ i => ⟨i, i + 1⟩)

/-- Checks if a number appears in a list -/
def appears_in (k : ℕ) (list : List ℕ) : Prop :=
  k ∈ list

/-- Theorem: Determine if the back of the last card can be identified -/
theorem last_card_identifiable (n : ℕ) (shown : List ℕ) (last : ℕ) : Prop :=
  let deck := create_deck n
  last = 0 ∨ last = n ∨
  (1 ≤ last ∧ last ≤ n - 1 ∧ (appears_in (last - 1) shown ∨ appears_in (last + 1) shown))

/-- Main theorem: Characterization of when the back of the last card can be determined -/
theorem determine_last_card_back (n : ℕ) (shown : List ℕ) (last : ℕ) :
  last_card_identifiable n shown last ↔
  ∃ (card : Card), card ∈ create_deck n ∧
    ((card.front = last ∧ ∃ k, k ∈ shown ∧ k = card.back) ∨
     (card.back = last ∧ ∃ k, k ∈ shown ∧ k = card.front)) :=
  sorry


end NUMINAMATH_CALUDE_last_card_identifiable_determine_last_card_back_l296_29610


namespace NUMINAMATH_CALUDE_largest_value_l296_29636

theorem largest_value (a b c d e : ℝ) 
  (h : a - 2 = b + 3 ∧ a - 2 = c - 4 ∧ a - 2 = d + 5 ∧ a - 2 = e - 6) :
  e = max a (max b (max c d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l296_29636


namespace NUMINAMATH_CALUDE_dinner_time_l296_29675

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_def : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

/-- The starting time (4:00 pm) -/
def startTime : Time := ⟨16, 0, sorry⟩

/-- The total duration of tasks in minutes -/
def totalTaskDuration : ℕ := 30 + 30 + 10 + 20 + 90

/-- Theorem: Adding the total task duration to the start time results in 7:00 pm -/
theorem dinner_time : addMinutes startTime totalTaskDuration = ⟨19, 0, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_dinner_time_l296_29675


namespace NUMINAMATH_CALUDE_max_items_purchasable_l296_29650

theorem max_items_purchasable (available : ℚ) (cost_per_item : ℚ) (h1 : available = 9.2) (h2 : cost_per_item = 1.05) :
  ⌊available / cost_per_item⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_items_purchasable_l296_29650


namespace NUMINAMATH_CALUDE_adjacent_diff_one_l296_29613

/-- Represents a 9x9 table filled with integers from 1 to 81 --/
def Table := Fin 9 → Fin 9 → Fin 81

/-- Two cells are adjacent if they are horizontally or vertically neighboring --/
def adjacent (i j i' j' : Fin 9) : Prop :=
  (i = i' ∧ (j.val + 1 = j'.val ∨ j'.val + 1 = j.val)) ∨
  (j = j' ∧ (i.val + 1 = i'.val ∨ i'.val + 1 = i.val))

/-- The table contains all integers from 1 to 81 exactly once --/
def validTable (t : Table) : Prop :=
  ∀ n : Fin 81, ∃! (i j : Fin 9), t i j = n

/-- Main theorem: In a 9x9 table filled with integers from 1 to 81,
    there exist two adjacent cells whose values differ by exactly 1 --/
theorem adjacent_diff_one (t : Table) (h : validTable t) :
  ∃ (i j i' j' : Fin 9), adjacent i j i' j' ∧ 
    (t i j).val = (t i' j').val + 1 ∨ (t i j).val + 1 = (t i' j').val :=
sorry

end NUMINAMATH_CALUDE_adjacent_diff_one_l296_29613


namespace NUMINAMATH_CALUDE_three_folds_halved_cut_segments_l296_29657

/-- A rope folded into equal parts, then folded in half, and cut in the middle -/
structure FoldedRope where
  initial_folds : ℕ  -- number of initial equal folds
  halved : Bool      -- whether the rope is folded in half after initial folding
  cut : Bool         -- whether the rope is cut in the middle

/-- Calculate the number of segments after folding and cutting -/
def num_segments (rope : FoldedRope) : ℕ :=
  if rope.halved ∧ rope.cut then
    rope.initial_folds * 2 + 1
  else
    rope.initial_folds

/-- Theorem: A rope folded into 3 equal parts, then folded in half, and cut in the middle results in 7 segments -/
theorem three_folds_halved_cut_segments :
  ∀ (rope : FoldedRope), rope.initial_folds = 3 → rope.halved → rope.cut →
  num_segments rope = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_three_folds_halved_cut_segments_l296_29657


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l296_29698

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 * I - 1) / I
  z.re > 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l296_29698


namespace NUMINAMATH_CALUDE_blue_tissue_length_l296_29688

theorem blue_tissue_length (red blue : ℝ) : 
  red = blue + 12 →
  2 * red = 3 * blue →
  blue = 24 := by
sorry

end NUMINAMATH_CALUDE_blue_tissue_length_l296_29688


namespace NUMINAMATH_CALUDE_jelly_bean_count_jelly_bean_theorem_l296_29649

theorem jelly_bean_count : ℕ → Prop :=
  fun total_jelly_beans =>
    let red_jelly_beans := (3 * total_jelly_beans) / 4
    let coconut_red_jelly_beans := red_jelly_beans / 4
    coconut_red_jelly_beans = 750 →
    total_jelly_beans = 4000

-- Proof
theorem jelly_bean_theorem : jelly_bean_count 4000 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_count_jelly_bean_theorem_l296_29649


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l296_29620

open Real

theorem angle_sum_is_pi_over_two (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : sin α ^ 2 + sin β ^ 2 - (Real.sqrt 6 / 2) * sin α - (Real.sqrt 10 / 2) * sin β + 1 = 0) : 
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l296_29620


namespace NUMINAMATH_CALUDE_linear_function_properties_l296_29694

def f (x : ℝ) := -2 * x + 4

theorem linear_function_properties :
  (∀ x y, x < y → f x > f y) ∧
  (∀ x y, x < 0 ∧ y < 0 → ¬(f x < 0 ∧ f y < 0)) ∧
  (f 0 ≠ 0 ∨ 4 ≠ 0) ∧
  (∀ x, f x - 4 = -2 * x) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l296_29694


namespace NUMINAMATH_CALUDE_half_hexagon_perimeter_l296_29641

/-- A polygon that forms one half of a regular hexagon by symmetrically splitting it -/
structure HalfHexagonPolygon where
  side_length : ℝ
  is_positive : side_length > 0

/-- The perimeter of a HalfHexagonPolygon -/
def perimeter (p : HalfHexagonPolygon) : ℝ :=
  3 * p.side_length

/-- Theorem: The perimeter of a HalfHexagonPolygon is equal to 3 times its side length -/
theorem half_hexagon_perimeter (p : HalfHexagonPolygon) :
  perimeter p = 3 * p.side_length := by
  sorry

end NUMINAMATH_CALUDE_half_hexagon_perimeter_l296_29641


namespace NUMINAMATH_CALUDE_additional_girls_needed_prove_additional_girls_l296_29632

theorem additional_girls_needed (initial_girls initial_boys : ℕ) 
  (target_ratio : ℚ) (additional_girls : ℕ) : Prop :=
  initial_girls = 2 →
  initial_boys = 6 →
  target_ratio = 5/8 →
  (initial_girls + additional_girls : ℚ) / 
    (initial_girls + initial_boys + additional_girls) = target_ratio →
  additional_girls = 8

theorem prove_additional_girls : 
  ∃ (additional_girls : ℕ), 
    additional_girls_needed 2 6 (5/8) additional_girls :=
sorry

end NUMINAMATH_CALUDE_additional_girls_needed_prove_additional_girls_l296_29632


namespace NUMINAMATH_CALUDE_fundraiser_problem_l296_29682

/-- The fundraiser problem -/
theorem fundraiser_problem
  (total_promised : ℕ)
  (sally_owes : ℕ)
  (carl_owes : ℕ)
  (amy_owes : ℕ)
  (derek_owes : ℕ)
  (h1 : total_promised = 400)
  (h2 : sally_owes = 35)
  (h3 : carl_owes = 35)
  (h4 : amy_owes = 30)
  (h5 : derek_owes = amy_owes / 2)
  : total_promised - (sally_owes + carl_owes + amy_owes + derek_owes) = 285 := by
  sorry

#check fundraiser_problem

end NUMINAMATH_CALUDE_fundraiser_problem_l296_29682


namespace NUMINAMATH_CALUDE_problem_solution_l296_29638

theorem problem_solution : (3^5 + 9720) * (Real.sqrt 289 - (845 / 169.1)) = 119556 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l296_29638


namespace NUMINAMATH_CALUDE_multiplication_problem_l296_29685

theorem multiplication_problem (A B C D : Nat) : 
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  A < 10 → B < 10 → C < 10 → D < 10 →
  C * 10 + D = 25 →
  (A * 100 + B * 10 + A) * (C * 10 + D) = C * 1000 + D * 100 + C * 10 + 0 →
  A + B = 2 := by
sorry

end NUMINAMATH_CALUDE_multiplication_problem_l296_29685


namespace NUMINAMATH_CALUDE_function_must_be_constant_l296_29629

-- Define the function type
def FunctionType := ℤ × ℤ → ℝ

-- Define the property of the function
def SatisfiesEquation (f : FunctionType) : Prop :=
  ∀ x y : ℤ, f (x, y) = (f (x - 1, y) + f (x, y - 1)) / 2

-- Define the range constraint
def InRange (f : FunctionType) : Prop :=
  ∀ x y : ℤ, 0 ≤ f (x, y) ∧ f (x, y) ≤ 1

-- Main theorem statement
theorem function_must_be_constant (f : FunctionType) 
  (h_eq : SatisfiesEquation f) (h_range : InRange f) : 
  ∃ c : ℝ, ∀ x y : ℤ, f (x, y) = c ∧ 0 ≤ c ∧ c ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_function_must_be_constant_l296_29629


namespace NUMINAMATH_CALUDE_multiple_of_a_l296_29679

theorem multiple_of_a (a : ℤ) : 
  (∃ k : ℤ, 97 * a^2 + 84 * a - 55 = k * a) ↔ 
  (a = 1 ∨ a = 5 ∨ a = 11 ∨ a = 55 ∨ a = -1 ∨ a = -5 ∨ a = -11 ∨ a = -55) :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_a_l296_29679


namespace NUMINAMATH_CALUDE_tan_2theta_value_l296_29690

theorem tan_2theta_value (θ : Real) (h1 : θ ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin θ - Real.cos θ = Real.sqrt 5 / 5) : 
  Real.tan (2 * θ) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_2theta_value_l296_29690


namespace NUMINAMATH_CALUDE_gretchens_earnings_l296_29606

/-- The amount Gretchen charges per drawing -/
def price_per_drawing : ℕ := 20

/-- The number of drawings sold on Saturday -/
def saturday_sales : ℕ := 24

/-- The number of drawings sold on Sunday -/
def sunday_sales : ℕ := 16

/-- Gretchen's total earnings over the weekend -/
def total_earnings : ℕ := price_per_drawing * (saturday_sales + sunday_sales)

/-- Theorem stating that Gretchen's total earnings are $800 -/
theorem gretchens_earnings : total_earnings = 800 := by
  sorry

end NUMINAMATH_CALUDE_gretchens_earnings_l296_29606


namespace NUMINAMATH_CALUDE_suhwan_milk_consumption_l296_29677

/-- Amount of milk Suhwan drinks per time in liters -/
def milk_per_time : ℝ := 0.2

/-- Number of times Suhwan drinks milk per day -/
def times_per_day : ℕ := 3

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Suhwan's weekly milk consumption in liters -/
def weekly_milk_consumption : ℝ :=
  milk_per_time * (times_per_day : ℝ) * (days_in_week : ℝ)

theorem suhwan_milk_consumption :
  weekly_milk_consumption = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_suhwan_milk_consumption_l296_29677


namespace NUMINAMATH_CALUDE_number_equation_solution_l296_29656

theorem number_equation_solution : ∃! x : ℝ, x + 2 + 8 = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l296_29656


namespace NUMINAMATH_CALUDE_unique_polynomial_mapping_l296_29603

/-- A second-degree polynomial in two variables -/
def p (x y : ℕ) : ℕ := ((x + y)^2 + 3*x + y) / 2

/-- Theorem stating the existence of a unique mapping for non-negative integers -/
theorem unique_polynomial_mapping :
  ∀ n : ℕ, ∃! (k m : ℕ), p k m = n :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_mapping_l296_29603


namespace NUMINAMATH_CALUDE_recipe_measurements_l296_29637

/- Define the required amounts and cup capacities -/
def required_flour : ℚ := 15/4  -- 3¾ cups
def required_milk : ℚ := 3/2    -- 1½ cups
def flour_cup_capacity : ℚ := 1/3
def milk_cup_capacity : ℚ := 1/4

/- Define the number of fills for each ingredient -/
def flour_fills : ℕ := 12
def milk_fills : ℕ := 6

/- Theorem statement -/
theorem recipe_measurements :
  (↑flour_fills * flour_cup_capacity ≥ required_flour) ∧
  ((↑flour_fills - 1) * flour_cup_capacity < required_flour) ∧
  (↑milk_fills * milk_cup_capacity = required_milk) :=
by sorry

end NUMINAMATH_CALUDE_recipe_measurements_l296_29637


namespace NUMINAMATH_CALUDE_f_symmetry_about_y_axis_l296_29666

def f (x : ℝ) : ℝ := |x|

theorem f_symmetry_about_y_axis : ∀ x : ℝ, f x = f (-x) := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_about_y_axis_l296_29666


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_simplification_l296_29664

theorem sqrt_sum_fractions_simplification :
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_simplification_l296_29664


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_252_630_l296_29651

theorem lcm_gcf_ratio_252_630 : Nat.lcm 252 630 / Nat.gcd 252 630 = 10 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_252_630_l296_29651


namespace NUMINAMATH_CALUDE_jerry_remaining_money_l296_29669

/-- Calculates the remaining money after spending -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem: Jerry's remaining money is $12 -/
theorem jerry_remaining_money :
  remaining_money 18 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_jerry_remaining_money_l296_29669


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l296_29691

-- Define the system of equations
def equation1 (x y z : ℚ) : Prop :=
  6 / (3 * x + 4 * y) + 4 / (5 * x - 4 * z) = 7 / 12

def equation2 (x y z : ℚ) : Prop :=
  9 / (4 * y + 3 * z) - 4 / (3 * x + 4 * y) = 1 / 3

def equation3 (x y z : ℚ) : Prop :=
  2 / (5 * x - 4 * z) + 6 / (4 * y + 3 * z) = 1 / 2

-- Theorem statement
theorem solution_satisfies_system :
  equation1 4 3 2 ∧ equation2 4 3 2 ∧ equation3 4 3 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l296_29691


namespace NUMINAMATH_CALUDE_problem_solution_l296_29689

theorem problem_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 20) : x - y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l296_29689


namespace NUMINAMATH_CALUDE_sum_four_pentagons_l296_29654

/-- The value of a square -/
def square : ℚ := sorry

/-- The value of a pentagon -/
def pentagon : ℚ := sorry

/-- First equation: square + 3*pentagon + square + pentagon = 25 -/
axiom eq1 : square + 3*pentagon + square + pentagon = 25

/-- Second equation: pentagon + 2*square + pentagon + square + pentagon = 22 -/
axiom eq2 : pentagon + 2*square + pentagon + square + pentagon = 22

/-- The sum of four pentagons is equal to 62/3 -/
theorem sum_four_pentagons : 4 * pentagon = 62/3 := by sorry

end NUMINAMATH_CALUDE_sum_four_pentagons_l296_29654


namespace NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_factors_l296_29600

theorem larger_number_given_hcf_lcm_factors (a b : ℕ) : 
  a > 0 → b > 0 → 
  Nat.gcd a b = 10 → 
  Nat.lcm a b = 10 * 11 * 15 → 
  max a b = 150 := by
sorry

end NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_factors_l296_29600


namespace NUMINAMATH_CALUDE_gcd_problem_l296_29665

theorem gcd_problem (b : ℤ) (h : 570 ∣ b) : Int.gcd (4*b^3 + b^2 + 5*b + 95) b = 95 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l296_29665


namespace NUMINAMATH_CALUDE_parabola_reflection_translation_l296_29643

theorem parabola_reflection_translation (a b c : ℝ) :
  let f := fun x => a * (x - 3)^2 + b * (x - 3) + c
  let g := fun x => -a * (x + 3)^2 - b * (x + 3) - c
  ∀ x, (f + g) x = -12 * a * x - 6 * b := by sorry

end NUMINAMATH_CALUDE_parabola_reflection_translation_l296_29643


namespace NUMINAMATH_CALUDE_expression_simplification_l296_29635

theorem expression_simplification (x y : ℝ) :
  3*x + 4*y + 5*x^2 + 2 - (8 - 5*x - 3*y - 2*x^2) = 7*x^2 + 8*x + 7*y - 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l296_29635


namespace NUMINAMATH_CALUDE_power_function_through_point_l296_29653

/-- A power function passing through (4, 1/2) has f(1/16) = 4 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x^a) →  -- f is a power function
  f 4 = 1/2 →             -- f passes through (4, 1/2)
  f (1/16) = 4 :=         -- prove f(1/16) = 4
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l296_29653


namespace NUMINAMATH_CALUDE_prove_first_ingot_weight_l296_29686

theorem prove_first_ingot_weight (weights : Fin 11 → ℕ) 
  (h_distinct : Function.Injective weights)
  (h_range : ∀ i, weights i ∈ Finset.range 12 \ {0}) : 
  ∃ (a b c d e f : Fin 11), 
    weights a + weights b + weights c + weights d ≤ 11 ∧
    weights a + weights e + weights f ≤ 11 ∧
    weights a = 1 :=
sorry

end NUMINAMATH_CALUDE_prove_first_ingot_weight_l296_29686


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l296_29658

theorem sugar_solution_percentage (x : ℝ) :
  (3/4 * x + 1/4 * 40 = 16) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l296_29658


namespace NUMINAMATH_CALUDE_half_radius_of_equal_area_circle_l296_29697

/-- Given two circles with the same area, where one has a circumference of 12π,
    half of the radius of the other circle is 3. -/
theorem half_radius_of_equal_area_circle (x y : ℝ) :
  (π * x^2 = π * y^2) →  -- Circles x and y have the same area
  (2 * π * x = 12 * π) →  -- Circle x has a circumference of 12π
  y / 2 = 3 := by  -- Half of the radius of circle y is 3
  sorry

end NUMINAMATH_CALUDE_half_radius_of_equal_area_circle_l296_29697


namespace NUMINAMATH_CALUDE_x_one_value_l296_29663

theorem x_one_value (x₁ x₂ x₃ : ℝ) 
  (h_order : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_sum : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/4) : 
  x₁ = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_x_one_value_l296_29663


namespace NUMINAMATH_CALUDE_inequality_proof_l296_29647

theorem inequality_proof (a b c : ℝ) : 
  a = Real.log 5 - Real.log 3 →
  b = (2 / 5) * Real.exp (2 / 3) →
  c = 2 / 3 →
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l296_29647


namespace NUMINAMATH_CALUDE_circle_c_value_l296_29622

/-- The circle equation with parameter c -/
def circle_equation (x y c : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 6*y + c = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-4, 3)

/-- The radius of the circle -/
def circle_radius : ℝ := 5

/-- Theorem: The value of c in the circle equation is 0 -/
theorem circle_c_value : ∃ (c : ℝ), 
  (∀ (x y : ℝ), circle_equation x y c ↔ 
    ((x + 4)^2 + (y - 3)^2 = circle_radius^2)) ∧ 
  c = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_c_value_l296_29622


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l296_29673

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l296_29673


namespace NUMINAMATH_CALUDE_village_population_l296_29672

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) : 
  partial_population = 36000 → percentage = 9/10 → 
  (percentage * total_population : ℚ) = partial_population → 
  total_population = 40000 := by
sorry

end NUMINAMATH_CALUDE_village_population_l296_29672


namespace NUMINAMATH_CALUDE_congruence_mod_210_l296_29614

theorem congruence_mod_210 (x : ℤ) : x^5 ≡ x [ZMOD 210] ↔ x ≡ 0 [ZMOD 7] ∨ x ≡ 1 [ZMOD 7] ∨ x ≡ -1 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_congruence_mod_210_l296_29614


namespace NUMINAMATH_CALUDE_point_symmetry_l296_29619

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two points are symmetric with respect to a line -/
def isSymmetric (p1 p2 : Point) (l : Line) : Prop :=
  -- The midpoint of the two points lies on the line
  l.a * ((p1.x + p2.x) / 2) + l.b * ((p1.y + p2.y) / 2) + l.c = 0 ∧
  -- The line connecting the two points is perpendicular to the given line
  (p2.x - p1.x) * l.a + (p2.y - p1.y) * l.b = 0

theorem point_symmetry :
  let a : Point := ⟨-1, 2⟩
  let b : Point := ⟨1, 4⟩
  let l : Line := ⟨1, 1, -3⟩
  isSymmetric a b l := by sorry

end NUMINAMATH_CALUDE_point_symmetry_l296_29619


namespace NUMINAMATH_CALUDE_trip_cost_difference_l296_29616

def trip_cost_sharing (alice_paid bob_paid charlie_paid dex_paid : ℚ) : ℚ :=
  let total_paid := alice_paid + bob_paid + charlie_paid + dex_paid
  let fair_share := total_paid / 4
  let alice_owes := max (fair_share - alice_paid) 0
  let charlie_owes := max (fair_share - charlie_paid) 0
  let bob_receives := max (bob_paid - fair_share) 0
  min alice_owes bob_receives - min charlie_owes (bob_receives - min alice_owes bob_receives)

theorem trip_cost_difference :
  trip_cost_sharing 160 220 190 95 = -35/2 :=
by sorry

end NUMINAMATH_CALUDE_trip_cost_difference_l296_29616


namespace NUMINAMATH_CALUDE_goldfish_equality_l296_29621

theorem goldfish_equality (n : ℕ) : (∃ m : ℕ, m < n ∧ 3^(m + 1) = 81 * 3^m) ↔ n > 3 :=
by sorry

end NUMINAMATH_CALUDE_goldfish_equality_l296_29621


namespace NUMINAMATH_CALUDE_hannah_trip_cost_l296_29652

/-- Calculates the cost of gas for a trip given odometer readings, fuel efficiency, and gas price -/
def trip_gas_cost (initial_reading : ℕ) (final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  ((final_reading - initial_reading : ℚ) / fuel_efficiency) * gas_price

theorem hannah_trip_cost :
  let initial_reading : ℕ := 32150
  let final_reading : ℕ := 32178
  let fuel_efficiency : ℚ := 25
  let gas_price : ℚ := 375/100
  trip_gas_cost initial_reading final_reading fuel_efficiency gas_price = 420/100 := by
  sorry

end NUMINAMATH_CALUDE_hannah_trip_cost_l296_29652


namespace NUMINAMATH_CALUDE_prob_spade_first_ace_last_value_l296_29628

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of aces in a standard deck -/
def NumAces : ℕ := 4

/-- Probability of drawing three cards from a standard 52-card deck,
    where the first card is a spade and the last card is an ace -/
def prob_spade_first_ace_last : ℚ :=
  (NumSpades * NumAces + NumAces - 1) / (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2))

theorem prob_spade_first_ace_last_value :
  prob_spade_first_ace_last = 51 / 2600 := by
  sorry

end NUMINAMATH_CALUDE_prob_spade_first_ace_last_value_l296_29628


namespace NUMINAMATH_CALUDE_strictly_decreasing_quadratic_function_l296_29693

/-- A function f(x) = kx² - 4x - 8 is strictly decreasing on [4, 16] iff k ∈ (-∞, 1/8] -/
theorem strictly_decreasing_quadratic_function (k : ℝ) :
  (∀ x₁ x₂ : ℝ, 4 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 16 →
    k * x₂^2 - 4*x₂ - 8 < k * x₁^2 - 4*x₁ - 8) ↔
  k ≤ 1/8 :=
sorry

end NUMINAMATH_CALUDE_strictly_decreasing_quadratic_function_l296_29693


namespace NUMINAMATH_CALUDE_vector_perpendicular_to_difference_l296_29624

/-- Given vectors a = (-1, 2) and b = (1, 3), prove that a is perpendicular to (a - b) -/
theorem vector_perpendicular_to_difference (a b : ℝ × ℝ) :
  a = (-1, 2) →
  b = (1, 3) →
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_vector_perpendicular_to_difference_l296_29624


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l296_29676

theorem quadratic_roots_sum_and_product :
  let a : ℝ := 9
  let b : ℝ := -45
  let c : ℝ := 50
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  sum_of_roots = 5 ∧ product_of_roots = 50 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l296_29676


namespace NUMINAMATH_CALUDE_amp_2_neg1_4_l296_29618

-- Define the operation &
def amp (a b c : ℝ) : ℝ := b^3 - 3*a*b*c - 4*a*c^2

-- Theorem statement
theorem amp_2_neg1_4 : amp 2 (-1) 4 = -105 := by
  sorry

end NUMINAMATH_CALUDE_amp_2_neg1_4_l296_29618


namespace NUMINAMATH_CALUDE_farmers_harvest_l296_29678

/-- Farmer's harvest problem -/
theorem farmers_harvest
  (total_potatoes : ℕ)
  (potatoes_per_bundle : ℕ)
  (potato_bundle_price : ℚ)
  (total_carrots : ℕ)
  (carrot_bundle_price : ℚ)
  (total_revenue : ℚ)
  (h1 : total_potatoes = 250)
  (h2 : potatoes_per_bundle = 25)
  (h3 : potato_bundle_price = 190/100)
  (h4 : total_carrots = 320)
  (h5 : carrot_bundle_price = 2)
  (h6 : total_revenue = 51)
  : (total_carrots / ((total_revenue - (total_potatoes / potatoes_per_bundle * potato_bundle_price)) / carrot_bundle_price) : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_farmers_harvest_l296_29678


namespace NUMINAMATH_CALUDE_f_upper_bound_implies_a_bound_l296_29605

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - x) * Real.exp x + a * (x - 1)^2

theorem f_upper_bound_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, f a x ≤ 2 * Real.exp x) →
  a ≤ ((1 - Real.sqrt 2) * Real.exp (1 - Real.sqrt 2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_f_upper_bound_implies_a_bound_l296_29605


namespace NUMINAMATH_CALUDE_margo_travel_distance_l296_29648

/-- The total distance traveled by Margo -/
def total_distance (bicycle_time walk_time average_rate : ℚ) : ℚ :=
  average_rate * (bicycle_time + walk_time) / 60

/-- Theorem: Given the conditions, Margo traveled 4 miles -/
theorem margo_travel_distance :
  let bicycle_time : ℚ := 15
  let walk_time : ℚ := 25
  let average_rate : ℚ := 6
  total_distance bicycle_time walk_time average_rate = 4 := by
  sorry

end NUMINAMATH_CALUDE_margo_travel_distance_l296_29648


namespace NUMINAMATH_CALUDE_no_such_function_l296_29615

theorem no_such_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1987 := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l296_29615


namespace NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l296_29601

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem min_zeros_odd_periodic_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : has_period f (2 * Real.pi)) 
  (h_zero_3 : f 3 = 0) 
  (h_zero_4 : f 4 = 0) : 
  ∃ (zeros : Finset ℝ), 
    (∀ x ∈ zeros, x ∈ Set.Icc 0 10 ∧ f x = 0) ∧ 
    Finset.card zeros ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l296_29601


namespace NUMINAMATH_CALUDE_largest_root_of_cubic_l296_29674

theorem largest_root_of_cubic (p q r : ℝ) : 
  p + q + r = 3 → 
  p * q + p * r + q * r = -8 → 
  p * q * r = -18 → 
  ∃ (x : ℝ), x = Real.sqrt 6 ∧ 
    x = max p (max q r) ∧
    x^3 - 3*x^2 - 8*x + 18 = 0 := by
sorry

end NUMINAMATH_CALUDE_largest_root_of_cubic_l296_29674


namespace NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l296_29681

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 12) (h2 : b = 30) : 
  (a + b + x = 58 ∨ a + b + x = 85) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l296_29681


namespace NUMINAMATH_CALUDE_total_sibling_age_l296_29623

/-- Represents the ages of four siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ

/-- Theorem stating the total age of the siblings -/
theorem total_sibling_age (ages : SiblingAges) : 
  ages.susan = 15 → 
  ages.bob = 11 → 
  ages.arthur = ages.susan + 2 → 
  ages.tom = ages.bob - 3 → 
  ages.susan + ages.arthur + ages.tom + ages.bob = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_sibling_age_l296_29623


namespace NUMINAMATH_CALUDE_dichromate_molecular_weight_l296_29609

/-- The molecular weight of 9 moles of dichromate (Cr2O7^2-) -/
theorem dichromate_molecular_weight (Cr_weight O_weight : ℝ) 
  (h1 : Cr_weight = 52.00)
  (h2 : O_weight = 16.00) :
  9 * (2 * Cr_weight + 7 * O_weight) = 1944.00 := by
  sorry

end NUMINAMATH_CALUDE_dichromate_molecular_weight_l296_29609


namespace NUMINAMATH_CALUDE_set_formation_criterion_l296_29699

-- Define a type for objects
variable {α : Type}

-- Define a predicate for definiteness and distinctness
def is_definite_and_distinct (S : Set α) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x = y ∨ x ≠ y)

-- Define a predicate for forming a set
def can_form_set (S : Set α) : Prop :=
  is_definite_and_distinct S

-- Theorem statement
theorem set_formation_criterion (S : Set α) :
  can_form_set S ↔ is_definite_and_distinct S :=
by
  sorry


end NUMINAMATH_CALUDE_set_formation_criterion_l296_29699


namespace NUMINAMATH_CALUDE_no_consecutive_fourth_powers_l296_29617

theorem no_consecutive_fourth_powers (n : ℤ) : 
  n^4 + (n+1)^4 + (n+2)^4 + (n+3)^4 ≠ (n+4)^4 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_fourth_powers_l296_29617


namespace NUMINAMATH_CALUDE_pattern_equality_l296_29645

theorem pattern_equality (n : ℕ+) : (2*n + 2)^2 - 4*n^2 = 8*n + 4 := by
  sorry

end NUMINAMATH_CALUDE_pattern_equality_l296_29645


namespace NUMINAMATH_CALUDE_orange_distribution_l296_29631

theorem orange_distribution (total_oranges : ℕ) (pieces_per_orange : ℕ) (pieces_per_friend : ℕ) :
  total_oranges = 80 →
  pieces_per_orange = 10 →
  pieces_per_friend = 4 →
  (total_oranges * pieces_per_orange) / pieces_per_friend = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l296_29631


namespace NUMINAMATH_CALUDE_type_b_machine_time_l296_29626

def job_completion_time (machine_q : ℝ) (machine_b : ℝ) (combined_time : ℝ) : Prop :=
  2 / machine_q + 3 / machine_b = 1 / combined_time

theorem type_b_machine_time : 
  ∀ (machine_b : ℝ),
    job_completion_time 5 machine_b 1.2 →
    machine_b = 90 / 13 := by
  sorry

end NUMINAMATH_CALUDE_type_b_machine_time_l296_29626


namespace NUMINAMATH_CALUDE_average_minutes_run_is_16_l296_29612

/-- Represents the average number of minutes run per day for each grade --/
structure GradeRunningAverage where
  sixth : ℝ
  seventh : ℝ
  eighth : ℝ

/-- Represents the ratio of students in each grade --/
structure GradeRatio where
  sixth_to_eighth : ℝ
  sixth_to_seventh : ℝ

/-- Calculates the average number of minutes run per day by all students --/
def average_minutes_run (avg : GradeRunningAverage) (ratio : GradeRatio) : ℝ :=
  sorry

/-- Theorem stating that the average number of minutes run per day is 16 --/
theorem average_minutes_run_is_16 (avg : GradeRunningAverage) (ratio : GradeRatio) 
  (h1 : avg.sixth = 16)
  (h2 : avg.seventh = 18)
  (h3 : avg.eighth = 12)
  (h4 : ratio.sixth_to_eighth = 3)
  (h5 : ratio.sixth_to_seventh = 1.5) :
  average_minutes_run avg ratio = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_minutes_run_is_16_l296_29612


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l296_29667

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Define set N
def N : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (U \ N) = {x : ℝ | x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l296_29667


namespace NUMINAMATH_CALUDE_frustum_volume_ratio_l296_29602

/-- Given a right prism with square base of side length L₁ and height H, 
    and a frustum of a pyramid extracted from it with square bases of 
    side lengths L₁ (lower) and L₂ (upper) and height H, 
    if the volume of the frustum is 2/3 of the total volume of the prism, 
    then L₁/L₂ = (1 + √5) / 2 -/
theorem frustum_volume_ratio (L₁ L₂ H : ℝ) (h_positive : L₁ > 0 ∧ L₂ > 0 ∧ H > 0) :
  (H / 3 * (L₁^2 + L₁*L₂ + L₂^2)) = (2 / 3 * L₁^2 * H) → 
  L₁ / L₂ = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_frustum_volume_ratio_l296_29602


namespace NUMINAMATH_CALUDE_work_completion_proof_l296_29695

/-- The number of days it takes W women to complete the work -/
def women_days : ℕ := 8

/-- The number of days it takes W children to complete the work -/
def children_days : ℕ := 12

/-- The number of days it takes 6 women and 3 children to complete the work -/
def combined_days : ℕ := 10

/-- The number of women in the combined group -/
def combined_women : ℕ := 6

/-- The number of children in the combined group -/
def combined_children : ℕ := 3

/-- The initial number of women working on the task -/
def initial_women : ℕ := 10

theorem work_completion_proof :
  (combined_women : ℚ) / (women_days * initial_women) +
  (combined_children : ℚ) / (children_days * initial_women) =
  1 / combined_days :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l296_29695


namespace NUMINAMATH_CALUDE_infinite_solutions_for_primes_l296_29668

theorem infinite_solutions_for_primes (p : ℕ) (hp : Prime p) :
  Set.Infinite {n : ℕ | n > 0 ∧ p ∣ 2^n - n} :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_for_primes_l296_29668


namespace NUMINAMATH_CALUDE_problem_statement_l296_29630

theorem problem_statement : 
  (∀ x : ℝ, ∀ a : ℝ, x^2 + a*x + a^2 ≥ 0) ∨ (∃ x₀ : ℕ+, 2 * (x₀.val)^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l296_29630


namespace NUMINAMATH_CALUDE_triangle_centroid_theorem_l296_29670

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Point O inside triangle ABC -/
structure PointInTriangle (t : Triangle) where
  O : Vector2D
  A : Vector2D
  B : Vector2D
  C : Vector2D

theorem triangle_centroid_theorem (t : Triangle) (p : PointInTriangle t) 
  (h1 : t.b = 6)
  (h2 : t.a * t.c * Real.cos t.B = t.a^2 - t.b^2 + (Real.sqrt 7 / 4) * t.b * t.c)
  (h3 : p.O.x + p.A.x + p.B.x + p.C.x = 0 ∧ p.O.y + p.A.y + p.B.y + p.C.y = 0)
  (h4 : Real.cos (t.A - π/6) = Real.cos t.A * Real.cos (π/6) + Real.sin t.A * Real.sin (π/6)) :
  (p.O.x - p.A.x)^2 + (p.O.y - p.A.y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_centroid_theorem_l296_29670


namespace NUMINAMATH_CALUDE_watch_cost_in_dollars_l296_29655

/-- The cost of a watch in dollars when paid with dimes -/
def watch_cost (num_dimes : ℕ) (dime_value : ℚ) : ℚ :=
  (num_dimes : ℚ) * dime_value

/-- Theorem: If Greyson paid for a watch with 50 dimes, and each dime is worth $0.10, then the cost of the watch is $5.00 -/
theorem watch_cost_in_dollars :
  watch_cost 50 (1/10) = 5 :=
sorry

end NUMINAMATH_CALUDE_watch_cost_in_dollars_l296_29655


namespace NUMINAMATH_CALUDE_backpack_solution_l296_29660

/-- Represents the prices and quantities of backpacks -/
structure BackpackData where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- Conditions for the backpack problem -/
def backpack_conditions (d : BackpackData) : Prop :=
  d.price_a = 2 * d.price_b - 30 ∧
  2 * d.price_a + 3 * d.price_b = 255 ∧
  d.quantity_a + d.quantity_b = 200 ∧
  50 * d.quantity_a + 40 * d.quantity_b ≤ 8900 ∧
  d.quantity_a > 87

/-- The theorem stating the correct prices and possible purchasing plans -/
theorem backpack_solution :
  ∃ (d : BackpackData),
    backpack_conditions d ∧
    d.price_a = 60 ∧
    d.price_b = 45 ∧
    ((d.quantity_a = 88 ∧ d.quantity_b = 112) ∨
     (d.quantity_a = 89 ∧ d.quantity_b = 111) ∨
     (d.quantity_a = 90 ∧ d.quantity_b = 110)) :=
  sorry

end NUMINAMATH_CALUDE_backpack_solution_l296_29660


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l296_29625

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (non_overlapping : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : perpendicular m β)
  (h3 : non_overlapping α β) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l296_29625


namespace NUMINAMATH_CALUDE_difference_of_squares_262_258_l296_29659

theorem difference_of_squares_262_258 : 262^2 - 258^2 = 2080 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_262_258_l296_29659


namespace NUMINAMATH_CALUDE_parallelogram_area_l296_29607

def v : Fin 2 → ℝ := ![3, -7]
def w : Fin 2 → ℝ := ![6, 4]

theorem parallelogram_area : 
  abs (Matrix.det !![v 0, v 1; w 0, w 1]) = 54 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l296_29607
