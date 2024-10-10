import Mathlib

namespace john_smith_payment_l1186_118610

def number_of_cakes : ℕ := 3
def cost_per_cake : ℕ := 12
def number_of_people_splitting_cost : ℕ := 2

theorem john_smith_payment (total_cost : ℕ) (johns_share : ℕ) : 
  total_cost = number_of_cakes * cost_per_cake →
  johns_share = total_cost / number_of_people_splitting_cost →
  johns_share = 18 := by
sorry

end john_smith_payment_l1186_118610


namespace rectangle_area_with_inscribed_circle_l1186_118606

theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 8 → ratio = 3 → 
  let width := 2 * r
  let length := ratio * width
  width * length = 768 := by
sorry

end rectangle_area_with_inscribed_circle_l1186_118606


namespace george_has_twelve_blocks_l1186_118628

/-- The number of blocks George has -/
def georgesBlocks (numBoxes : ℕ) (blocksPerBox : ℕ) : ℕ :=
  numBoxes * blocksPerBox

/-- Theorem: George has 12 blocks given 2 boxes with 6 blocks each -/
theorem george_has_twelve_blocks :
  georgesBlocks 2 6 = 12 := by
  sorry

end george_has_twelve_blocks_l1186_118628


namespace nina_running_distance_l1186_118653

theorem nina_running_distance (x : ℝ) : 
  2 * x + 0.6666666666666666 = 0.8333333333333334 → 
  x = 0.08333333333333337 := by
  sorry

end nina_running_distance_l1186_118653


namespace max_m_plus_2n_max_fraction_min_fraction_l1186_118671

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define a point M on the circle C
def M (m n : ℝ) : Prop := C m n

-- Theorem for the maximum value of m + 2n
theorem max_m_plus_2n :
  ∃ (max : ℝ), (∀ m n, M m n → m + 2*n ≤ max) ∧ (∃ m n, M m n ∧ m + 2*n = max) ∧ max = 16 + 2*Real.sqrt 10 :=
sorry

-- Theorem for the maximum value of (n-3)/(m+2)
theorem max_fraction :
  ∃ (max : ℝ), (∀ m n, M m n → (n - 3) / (m + 2) ≤ max) ∧ (∃ m n, M m n ∧ (n - 3) / (m + 2) = max) ∧ max = 2 + Real.sqrt 3 :=
sorry

-- Theorem for the minimum value of (n-3)/(m+2)
theorem min_fraction :
  ∃ (min : ℝ), (∀ m n, M m n → min ≤ (n - 3) / (m + 2)) ∧ (∃ m n, M m n ∧ (n - 3) / (m + 2) = min) ∧ min = 2 - Real.sqrt 3 :=
sorry

end max_m_plus_2n_max_fraction_min_fraction_l1186_118671


namespace hotel_room_charges_l1186_118605

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R - 0.5 * R) 
  (h2 : P = G - 0.1 * G) : 
  R = 1.8 * G := by
sorry

end hotel_room_charges_l1186_118605


namespace difference_of_numbers_l1186_118675

theorem difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (diff_squares : x^2 - y^2 = 32) : 
  |x - y| = 4 := by
  sorry

end difference_of_numbers_l1186_118675


namespace cubic_fraction_equality_l1186_118698

theorem cubic_fraction_equality : 
  let a : ℝ := 5
  let b : ℝ := 4
  (a^3 + b^3) / (a^2 - a*b + b^2) = 9 := by
  sorry

end cubic_fraction_equality_l1186_118698


namespace point_in_fourth_quadrant_range_l1186_118649

/-- A point in the fourth quadrant has positive x-coordinate and negative y-coordinate -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The theorem states that if a point P(a, a-2) is in the fourth quadrant, then 0 < a < 2 -/
theorem point_in_fourth_quadrant_range (a : ℝ) :
  fourth_quadrant a (a - 2) → 0 < a ∧ a < 2 := by
  sorry

end point_in_fourth_quadrant_range_l1186_118649


namespace square_perimeter_l1186_118686

/-- Given a square with side length 15 cm, prove that its perimeter is 60 cm. -/
theorem square_perimeter (side_length : ℝ) (h : side_length = 15) : 
  4 * side_length = 60 := by
  sorry

end square_perimeter_l1186_118686


namespace y_equal_y_greater_l1186_118615

-- Define the functions y₁ and y₂
def y₁ (x : ℝ) : ℝ := -x + 3
def y₂ (x : ℝ) : ℝ := 2 + x

-- Theorem 1: y₁ = y₂ when x = 1/2
theorem y_equal (x : ℝ) : y₁ x = y₂ x ↔ x = 1/2 := by sorry

-- Theorem 2: y₁ = 2y₂ + 5 when x = -2
theorem y_greater (x : ℝ) : y₁ x = 2 * y₂ x + 5 ↔ x = -2 := by sorry

end y_equal_y_greater_l1186_118615


namespace consecutive_integers_product_812_sum_57_l1186_118683

theorem consecutive_integers_product_812_sum_57 :
  ∀ x y : ℕ+, 
    x.val + 1 = y.val →
    x * y = 812 →
    x + y = 57 := by
  sorry

end consecutive_integers_product_812_sum_57_l1186_118683


namespace fourth_root_over_seventh_root_of_seven_l1186_118679

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) :
  (x ^ (1/4)) / (x ^ (1/7)) = x ^ (3/28) :=
by sorry

end fourth_root_over_seventh_root_of_seven_l1186_118679


namespace union_complement_problem_l1186_118662

universe u

def I : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem union_complement_problem : A ∪ (I \ B) = {0, 1, 2} := by sorry

end union_complement_problem_l1186_118662


namespace EF_equals_5_sqrt_35_div_3_l1186_118693

/-- A rectangle ABCD with a point E inside -/
structure Rectangle :=
  (A B C D E : ℝ × ℝ)
  (is_rectangle : A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ C.2 = D.2)
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 30)
  (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 40)
  (E_inside : E.1 > A.1 ∧ E.1 < C.1 ∧ E.2 > A.2 ∧ E.2 < C.2)
  (EA_length : Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 10)
  (EB_length : Real.sqrt ((E.1 - B.1)^2 + (E.2 - B.2)^2) = 30)

/-- The length of EF, where F is the foot of the perpendicular from E to AD -/
def EF_length (r : Rectangle) : ℝ := r.E.2 - r.A.2

theorem EF_equals_5_sqrt_35_div_3 (r : Rectangle) : 
  EF_length r = 5 * Real.sqrt 35 / 3 :=
sorry

end EF_equals_5_sqrt_35_div_3_l1186_118693


namespace f_of_4_eq_17_g_of_2_eq_29_l1186_118614

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x - 3

-- Define the function g
def g (t : ℝ) : ℝ := 4 * t^3 + 2 * t - 7

-- Theorem for f(4) = 17
theorem f_of_4_eq_17 : f 4 = 17 := by sorry

-- Theorem for g(2) = 29
theorem g_of_2_eq_29 : g 2 = 29 := by sorry

end f_of_4_eq_17_g_of_2_eq_29_l1186_118614


namespace regular_polygon_with_four_to_one_angle_ratio_l1186_118674

/-- A regular polygon where the interior angle is exactly 4 times the exterior angle has 10 sides -/
theorem regular_polygon_with_four_to_one_angle_ratio (n : ℕ) : 
  n > 2 → 
  (360 / n : ℚ) * 4 = (180 - 360 / n : ℚ) → 
  n = 10 := by
  sorry

end regular_polygon_with_four_to_one_angle_ratio_l1186_118674


namespace container_capacity_l1186_118669

theorem container_capacity (initial_fill : Real) (added_water : Real) (final_fill : Real) :
  initial_fill = 0.3 →
  added_water = 45 →
  final_fill = 0.75 →
  ∃ (capacity : Real), capacity = 100 ∧
    final_fill * capacity = initial_fill * capacity + added_water :=
by sorry

end container_capacity_l1186_118669


namespace f_upper_bound_l1186_118657

-- Define the set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Define the properties of function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧
  (∀ x ∈ A, f x ≥ 0) ∧
  (∀ x y, x ∈ A → y ∈ A → x + y ∈ A → f (x + y) ≥ f x + f y)

-- Theorem statement
theorem f_upper_bound 
  (f : ℝ → ℝ) 
  (hf : is_valid_f f) :
  ∀ x ∈ A, f x ≤ 2 * x :=
sorry

end f_upper_bound_l1186_118657


namespace binary_10011_equals_19_l1186_118629

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_10011_equals_19 : 
  binary_to_decimal [true, false, false, true, true] = 19 := by
  sorry

end binary_10011_equals_19_l1186_118629


namespace geometric_sequence_minimum_l1186_118654

/-- A positive geometric sequence satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 1 / a 0
  condition : a 8 = a 6 + 2 * a 4

/-- The theorem statement -/
theorem geometric_sequence_minimum (seq : GeometricSequence) :
  (∃ m n : ℕ, Real.sqrt (seq.a m * seq.a n) = Real.sqrt 2 * seq.a 1) →
  (∀ m n : ℕ, 1 / m + 9 / n ≥ 4) ∧
  (∃ m n : ℕ, 1 / m + 9 / n = 4) :=
by sorry

end geometric_sequence_minimum_l1186_118654


namespace m_equals_one_sufficient_not_necessary_for_abs_m_equals_one_l1186_118634

theorem m_equals_one_sufficient_not_necessary_for_abs_m_equals_one :
  (∀ m : ℝ, m = 1 → |m| = 1) ∧
  (∃ m : ℝ, |m| = 1 ∧ m ≠ 1) :=
by sorry

end m_equals_one_sufficient_not_necessary_for_abs_m_equals_one_l1186_118634


namespace range_of_expression_l1186_118646

theorem range_of_expression (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 ≤ β ∧ β ≤ π/2) :
  -π/6 < 2*α - β/3 ∧ 2*α - β/3 < π := by
  sorry

end range_of_expression_l1186_118646


namespace club_president_vicepresident_selection_l1186_118604

theorem club_president_vicepresident_selection (total_members : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_members = 30)
  (h2 : boys = 18)
  (h3 : girls = 12)
  (h4 : total_members = boys + girls) :
  (boys * (total_members - 1)) = 522 := by
  sorry

end club_president_vicepresident_selection_l1186_118604


namespace frog_jump_probability_l1186_118670

/-- Represents the probability of ending on a horizontal side -/
def probability_horizontal_end (x y : ℝ) : ℝ := sorry

/-- The rectangle's dimensions -/
def rectangle_width : ℝ := 5
def rectangle_height : ℝ := 5

/-- The frog's starting position -/
def start_x : ℝ := 2
def start_y : ℝ := 3

/-- Theorem stating the probability of ending on a horizontal side -/
theorem frog_jump_probability :
  probability_horizontal_end start_x start_y = 13 / 14 := by sorry

end frog_jump_probability_l1186_118670


namespace polynomial_factorization_l1186_118625

theorem polynomial_factorization :
  ∀ x : ℝ, (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) := by
  sorry

end polynomial_factorization_l1186_118625


namespace smallest_n_congruence_four_satisfies_congruence_four_is_smallest_smallest_positive_integer_congruence_l1186_118668

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 13 * n ≡ 567 [MOD 5] → n ≥ 4 :=
by sorry

theorem four_satisfies_congruence : 13 * 4 ≡ 567 [MOD 5] :=
by sorry

theorem four_is_smallest (m : ℕ) : m > 0 ∧ m < 4 → ¬(13 * m ≡ 567 [MOD 5]) :=
by sorry

theorem smallest_positive_integer_congruence : 
  ∃ (n : ℕ), n > 0 ∧ 13 * n ≡ 567 [MOD 5] ∧ 
  ∀ (m : ℕ), m > 0 ∧ 13 * m ≡ 567 [MOD 5] → n ≤ m :=
by sorry

end smallest_n_congruence_four_satisfies_congruence_four_is_smallest_smallest_positive_integer_congruence_l1186_118668


namespace balance_disruption_possible_l1186_118678

/-- Represents a coin with a weight of either 7 or 8 grams -/
inductive Coin
  | Light : Coin  -- 7 grams
  | Heavy : Coin  -- 8 grams

/-- Represents the state of the balance scale -/
structure BalanceState :=
  (left : List Coin)
  (right : List Coin)

/-- Checks if the balance scale is in equilibrium -/
def isBalanced (state : BalanceState) : Bool :=
  (state.left.length = state.right.length) &&
  (state.left.foldl (fun acc c => acc + match c with
    | Coin.Light => 7
    | Coin.Heavy => 8) 0 =
   state.right.foldl (fun acc c => acc + match c with
    | Coin.Light => 7
    | Coin.Heavy => 8) 0)

/-- Performs a swap operation on the balance scale -/
def swapCoins (state : BalanceState) (n : Nat) : BalanceState :=
  { left := state.right.take n ++ state.left.drop n,
    right := state.left.take n ++ state.right.drop n }

/-- The main theorem to be proved -/
theorem balance_disruption_possible :
  ∀ (initialState : BalanceState),
    initialState.left.length = 144 →
    initialState.right.length = 144 →
    isBalanced initialState →
    ∃ (finalState : BalanceState),
      ∃ (numOperations : Nat),
        numOperations ≤ 11 ∧
        ¬isBalanced finalState ∧
        (∃ (swaps : List Nat),
          swaps.length = numOperations ∧
          finalState = swaps.foldl swapCoins initialState) :=
sorry

end balance_disruption_possible_l1186_118678


namespace dave_guitar_strings_l1186_118664

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 2

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 12

/-- The total number of guitar strings Dave needs to replace -/
def total_strings_needed : ℕ := 144

/-- Theorem stating that the total number of guitar strings Dave needs to replace is 144 -/
theorem dave_guitar_strings :
  strings_per_night * shows_per_week * total_weeks = total_strings_needed :=
by sorry

end dave_guitar_strings_l1186_118664


namespace f_72_value_l1186_118617

/-- A function satisfying f(ab) = f(a) + f(b) for all a and b -/
def MultiplicativeToAdditive (f : ℕ → ℝ) : Prop :=
  ∀ a b : ℕ, f (a * b) = f a + f b

/-- The main theorem -/
theorem f_72_value (f : ℕ → ℝ) (p q : ℝ) 
    (h1 : MultiplicativeToAdditive f) 
    (h2 : f 2 = p) 
    (h3 : f 3 = q) : 
  f 72 = 3 * p + 2 * q := by
  sorry

end f_72_value_l1186_118617


namespace eve_envelope_count_l1186_118665

def envelope_numbers : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128]

theorem eve_envelope_count :
  ∀ (eve_numbers alie_numbers : List ℕ),
    eve_numbers ++ alie_numbers = envelope_numbers →
    eve_numbers.sum = alie_numbers.sum + 31 →
    eve_numbers.length = 5 := by
  sorry

end eve_envelope_count_l1186_118665


namespace mean_home_runs_l1186_118642

def number_of_players : ℕ := 9

def home_run_distribution : List (ℕ × ℕ) :=
  [(5, 2), (6, 3), (8, 2), (10, 1), (12, 1)]

def total_home_runs : ℕ :=
  (home_run_distribution.map (λ (hr, count) => hr * count)).sum

theorem mean_home_runs :
  (total_home_runs : ℚ) / number_of_players = 66 / 9 := by sorry

end mean_home_runs_l1186_118642


namespace parallel_resistance_calculation_l1186_118622

/-- 
Represents the combined resistance of two resistors connected in parallel.
x: resistance of the first resistor in ohms
y: resistance of the second resistor in ohms
r: combined resistance in ohms
-/
def parallel_resistance (x y : ℝ) (r : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ r > 0 ∧ (1 / r = 1 / x + 1 / y)

theorem parallel_resistance_calculation :
  ∃ (r : ℝ), parallel_resistance 4 6 r ∧ r = 2.4 := by sorry

end parallel_resistance_calculation_l1186_118622


namespace line_circle_separation_l1186_118626

theorem line_circle_separation (a b : ℝ) (h_inside : a^2 + b^2 < 1) (h_not_origin : (a, b) ≠ (0, 0)) :
  ∀ x y : ℝ, (x^2 + y^2 = 1) → (a*x + b*y ≠ 1) := by
  sorry

end line_circle_separation_l1186_118626


namespace arithmetic_sequence_common_difference_l1186_118689

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence. -/
def CommonDifference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (p q : ℕ) (h_arith : ArithmeticSequence a)
  (h_p : a p = 4) (h_q : a q = 2) (h_pq : p = 4 + q) :
  ∃ d : ℝ, CommonDifference a d ∧ d = 1/2 := by
  sorry

end arithmetic_sequence_common_difference_l1186_118689


namespace abc_value_l1186_118650

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a * (b + c) = 156)
  (eq2 : b * (c + a) = 168)
  (eq3 : c * (a + b) = 180) :
  a * b * c = 288 * Real.sqrt 7 := by
sorry

end abc_value_l1186_118650


namespace shoes_cost_is_74_l1186_118676

-- Define the discount rate
def discount_rate : ℚ := 0.1

-- Define the cost of socks and bag
def socks_cost : ℚ := 2 * 2
def bag_cost : ℚ := 42

-- Define the discount threshold
def discount_threshold : ℚ := 100

-- Define the final payment amount
def final_payment : ℚ := 118

-- Theorem to prove
theorem shoes_cost_is_74 :
  ∃ (shoes_cost : ℚ),
    let total_cost := shoes_cost + socks_cost + bag_cost
    let discount := max (discount_rate * (total_cost - discount_threshold)) 0
    total_cost - discount = final_payment ∧ shoes_cost = 74 := by
  sorry

end shoes_cost_is_74_l1186_118676


namespace pentagonal_grid_toothpicks_l1186_118666

/-- The number of toothpicks in the base of the pentagonal grid -/
def base_toothpicks : ℕ := 10

/-- The number of toothpicks in each of the four non-base sides -/
def side_toothpicks : ℕ := 8

/-- The number of sides excluding the base -/
def num_sides : ℕ := 4

/-- The number of vertices in a pentagon -/
def num_vertices : ℕ := 5

/-- The total number of toothpicks needed for the framed pentagonal grid -/
def total_toothpicks : ℕ := base_toothpicks + num_sides * side_toothpicks + num_vertices

theorem pentagonal_grid_toothpicks : total_toothpicks = 47 := by
  sorry

end pentagonal_grid_toothpicks_l1186_118666


namespace sin_translation_l1186_118630

/-- Given a function f(x) = sin(2x), when translated π/3 units to the right,
    the resulting function g(x) is equal to sin(2x - 2π/3). -/
theorem sin_translation (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.sin (2 * x)
  let g : ℝ → ℝ := fun x => f (x - π / 3)
  g x = Real.sin (2 * x - 2 * π / 3) :=
by sorry

end sin_translation_l1186_118630


namespace square_perimeter_l1186_118612

/-- The sum of the lengths of all sides of a square with side length 5 cm is 20 cm. -/
theorem square_perimeter (side_length : ℝ) (h : side_length = 5) : 
  4 * side_length = 20 := by
  sorry

end square_perimeter_l1186_118612


namespace exists_fifteen_classmates_l1186_118681

/-- A type representing students. -/
def Student : Type := ℕ

/-- The total number of students. -/
def total_students : ℕ := 60

/-- A function that returns true if the given students are classmates. -/
def are_classmates : List Student → Prop := sorry

/-- The property that among any 10 students, there are always 3 classmates. -/
axiom three_classmates_in_ten : 
  ∀ (s : Finset Student), s.card = 10 → ∃ (t : Finset Student), t ⊆ s ∧ t.card = 3 ∧ are_classmates t.toList

/-- The theorem to be proved. -/
theorem exists_fifteen_classmates :
  ∃ (s : Finset Student), s.card ≥ 15 ∧ are_classmates s.toList :=
sorry

end exists_fifteen_classmates_l1186_118681


namespace A_characterization_and_inequality_l1186_118631

def f (x : ℝ) : ℝ := |2*x + 1| + |x - 2|

def A : Set ℝ := {x | f x < 3}

theorem A_characterization_and_inequality :
  (A = {x : ℝ | -2/3 < x ∧ x < 0}) ∧
  (∀ s t : ℝ, s ∈ A → t ∈ A → |1 - t/s| < |t - 1/s|) := by sorry

end A_characterization_and_inequality_l1186_118631


namespace small_cuboids_needed_for_large_l1186_118696

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℕ :=
  d.width * d.length * d.height

/-- The dimensions of the large cuboid -/
def largeCuboid : CuboidDimensions :=
  { width := 24, length := 15, height := 28 }

/-- The dimensions of the small cuboid -/
def smallCuboid : CuboidDimensions :=
  { width := 4, length := 5, height := 7 }

/-- Theorem stating that 72 small cuboids are needed to create the large cuboid -/
theorem small_cuboids_needed_for_large : 
  (cuboidVolume largeCuboid) / (cuboidVolume smallCuboid) = 72 := by
  sorry

end small_cuboids_needed_for_large_l1186_118696


namespace shopping_trip_cost_theorem_l1186_118639

/-- Calculates the total cost of James' shopping trip -/
def shopping_trip_cost : ℝ :=
  let milk_price : ℝ := 4.50
  let milk_tax_rate : ℝ := 0.20
  let bananas_price : ℝ := 3.00
  let bananas_tax_rate : ℝ := 0.15
  let baguette_price : ℝ := 2.50
  let cereal_price : ℝ := 6.00
  let cereal_discount : ℝ := 0.20
  let cereal_tax_rate : ℝ := 0.12
  let eggs_price : ℝ := 3.50
  let eggs_coupon : ℝ := 1.00
  let eggs_tax_rate : ℝ := 0.18

  let milk_total := milk_price * (1 + milk_tax_rate)
  let bananas_total := bananas_price * (1 + bananas_tax_rate)
  let baguette_total := baguette_price
  let cereal_discounted := cereal_price * (1 - cereal_discount)
  let cereal_total := cereal_discounted * (1 + cereal_tax_rate)
  let eggs_discounted := eggs_price - eggs_coupon
  let eggs_total := eggs_discounted * (1 + eggs_tax_rate)

  milk_total + bananas_total + baguette_total + cereal_total + eggs_total

theorem shopping_trip_cost_theorem : shopping_trip_cost = 19.68 := by
  sorry

end shopping_trip_cost_theorem_l1186_118639


namespace initial_number_of_persons_l1186_118685

theorem initial_number_of_persons (n : ℕ) 
  (h1 : 4 * n = 48) : n = 12 := by
  sorry

#check initial_number_of_persons

end initial_number_of_persons_l1186_118685


namespace reciprocal_of_negative_fraction_l1186_118602

theorem reciprocal_of_negative_fraction :
  ((-5 : ℚ) / 3)⁻¹ = -3 / 5 := by sorry

end reciprocal_of_negative_fraction_l1186_118602


namespace room_width_to_perimeter_ratio_l1186_118688

theorem room_width_to_perimeter_ratio :
  let length : ℝ := 22
  let width : ℝ := 15
  let perimeter : ℝ := 2 * (length + width)
  (width / perimeter) = (15 / 74) := by
sorry

end room_width_to_perimeter_ratio_l1186_118688


namespace fraction_simplification_l1186_118660

theorem fraction_simplification (a b : ℝ) : (9 * b) / (6 * a + 3) = (3 * b) / (2 * a + 1) := by
  sorry

end fraction_simplification_l1186_118660


namespace sum_of_two_numbers_l1186_118692

theorem sum_of_two_numbers (A B : ℝ) (h1 : A + B = 147) (h2 : A = 0.375 * B + 4) : A + B = 147 := by
  sorry

end sum_of_two_numbers_l1186_118692


namespace power_difference_zero_l1186_118691

theorem power_difference_zero : (2^3)^2 - 4^3 = 0 := by
  sorry

end power_difference_zero_l1186_118691


namespace angle_bisection_quadrant_l1186_118643

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, k * 2 * Real.pi + Real.pi < α ∧ α < k * 2 * Real.pi + 3 * Real.pi / 2

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, (k * Real.pi + Real.pi / 2 < α ∧ α < k * Real.pi + Real.pi) ∨
             (k * Real.pi + 3 * Real.pi / 2 < α ∧ α < (k + 1) * Real.pi)

theorem angle_bisection_quadrant (α : Real) :
  is_in_third_quadrant α → is_in_second_or_fourth_quadrant (α / 2) :=
by
  sorry

end angle_bisection_quadrant_l1186_118643


namespace arithmetic_progression_product_not_square_l1186_118601

/-- Four distinct positive integers in arithmetic progression -/
structure ArithmeticProgression :=
  (a : ℕ+) -- First term
  (r : ℕ+) -- Common difference
  (distinct : a < a + r ∧ a + r < a + 2*r ∧ a + 2*r < a + 3*r)

/-- The product of four terms in arithmetic progression is not a perfect square -/
theorem arithmetic_progression_product_not_square (ap : ArithmeticProgression) :
  ¬ ∃ (m : ℕ), (ap.a * (ap.a + ap.r) * (ap.a + 2*ap.r) * (ap.a + 3*ap.r) : ℕ) = m^2 := by
  sorry


end arithmetic_progression_product_not_square_l1186_118601


namespace triangle_with_consecutive_sides_and_obtuse_angle_l1186_118613

-- Define a triangle with sides of consecutive natural numbers
def ConsecutiveSidedTriangle (a b c : ℕ) : Prop :=
  (b = a + 1) ∧ (c = b + 1) ∧ (a > 0)

-- Define the condition for the largest angle to be obtuse
def HasObtuseAngle (a b c : ℕ) : Prop :=
  let cosLargestAngle := (a^2 + b^2 - c^2) / (2 * a * b)
  cosLargestAngle < 0

-- Theorem statement
theorem triangle_with_consecutive_sides_and_obtuse_angle
  (a b c : ℕ) (h1 : ConsecutiveSidedTriangle a b c) (h2 : HasObtuseAngle a b c) :
  (a = 2 ∧ b = 3 ∧ c = 4) :=
sorry

end triangle_with_consecutive_sides_and_obtuse_angle_l1186_118613


namespace simplify_and_rationalize_l1186_118640

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 9) = Real.sqrt 15 / 9 := by
  sorry

end simplify_and_rationalize_l1186_118640


namespace inequality_solution_l1186_118635

theorem inequality_solution (x : ℝ) : 
  1 / (x + 2) + 7 / (x + 6) ≥ 1 ↔ 
  x ≤ -6 ∨ (-2 < x ∧ x ≤ -Real.sqrt 15) ∨ x ≥ Real.sqrt 15 :=
by sorry

end inequality_solution_l1186_118635


namespace specific_square_figure_perimeter_l1186_118621

/-- A figure composed of squares arranged in a specific pattern -/
structure SquareFigure where
  squareSideLength : ℝ
  horizontalSegments : ℕ
  verticalSegments : ℕ

/-- The perimeter of a SquareFigure -/
def perimeter (f : SquareFigure) : ℝ :=
  (f.horizontalSegments + f.verticalSegments) * f.squareSideLength * 2

/-- Theorem stating that the perimeter of the specific square figure is 52 -/
theorem specific_square_figure_perimeter :
  ∃ (f : SquareFigure),
    f.squareSideLength = 2 ∧
    f.horizontalSegments = 16 ∧
    f.verticalSegments = 10 ∧
    perimeter f = 52 := by
  sorry

end specific_square_figure_perimeter_l1186_118621


namespace shampoo_bottles_l1186_118651

theorem shampoo_bottles (small_capacity large_capacity current_amount : ℕ) 
  (h1 : small_capacity = 40)
  (h2 : large_capacity = 800)
  (h3 : current_amount = 120) : 
  (large_capacity - current_amount) / small_capacity = 17 := by
  sorry

end shampoo_bottles_l1186_118651


namespace pentagon_y_coordinate_l1186_118684

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- Check if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop :=
  p.A.x = 0 ∧ p.B.x = 0 ∧ p.D.x = p.E.x ∧ p.C.x = (p.D.x / 2)

/-- Calculate the area of a pentagon -/
noncomputable def pentagonArea (p : Pentagon) : ℝ :=
  sorry -- Actual implementation would go here

theorem pentagon_y_coordinate (p : Pentagon) :
  p.A = ⟨0, 0⟩ →
  p.B = ⟨0, 5⟩ →
  p.D = ⟨6, 5⟩ →
  p.E = ⟨6, 0⟩ →
  p.C.x = 3 →
  hasVerticalSymmetry p →
  pentagonArea p = 50 →
  p.C.y = 35/3 := by
  sorry

end pentagon_y_coordinate_l1186_118684


namespace equal_solutions_iff_n_eq_neg_one_third_l1186_118695

theorem equal_solutions_iff_n_eq_neg_one_third 
  (x y n : ℝ) : 
  (2 * x - 5 * y = 3 * n + 7 ∧ x - 3 * y = 4) → 
  (∃! (x y : ℝ), 2 * x - 5 * y = 3 * n + 7 ∧ x - 3 * y = 4) ↔ 
  n = -1/3 := by
sorry

end equal_solutions_iff_n_eq_neg_one_third_l1186_118695


namespace xiaobing_jumps_189_ropes_per_minute_l1186_118682

/-- The number of ropes Xiaohan jumps per minute -/
def xiaohan_ropes_per_minute : ℕ := 168

/-- The number of ropes Xiaobing jumps per minute -/
def xiaobing_ropes_per_minute : ℕ := xiaohan_ropes_per_minute + 21

/-- The number of ropes Xiaobing jumps in the given time -/
def xiaobing_ropes : ℕ := 135

/-- The number of ropes Xiaohan jumps in the given time -/
def xiaohan_ropes : ℕ := 120

theorem xiaobing_jumps_189_ropes_per_minute :
  (xiaobing_ropes : ℚ) / xiaobing_ropes_per_minute = (xiaohan_ropes : ℚ) / xiaohan_ropes_per_minute →
  xiaobing_ropes_per_minute = 189 := by
  sorry

end xiaobing_jumps_189_ropes_per_minute_l1186_118682


namespace power_equation_solution_l1186_118655

theorem power_equation_solution (m : ℕ) : 8^36 * 6^21 = 3 * 24^m → m = 43 := by
  sorry

end power_equation_solution_l1186_118655


namespace al_karhi_square_root_approximation_l1186_118633

theorem al_karhi_square_root_approximation 
  (N a r : ℝ) 
  (h1 : N > 0) 
  (h2 : a > 0) 
  (h3 : a^2 ≤ N) 
  (h4 : (a+1)^2 > N) 
  (h5 : r = N - a^2) 
  (h6 : r < 2*a + 1) : 
  ∃ (ε : ℝ), ε > 0 ∧ |Real.sqrt N - (a + r / (2*a + 1))| < ε :=
sorry

end al_karhi_square_root_approximation_l1186_118633


namespace randy_walks_dog_twice_daily_l1186_118611

/-- The number of times Randy walks his dog per day -/
def walks_per_day (wipes_per_pack : ℕ) (packs_for_360_days : ℕ) : ℕ :=
  (wipes_per_pack * packs_for_360_days) / 360

theorem randy_walks_dog_twice_daily :
  walks_per_day 120 6 = 2 := by
  sorry

end randy_walks_dog_twice_daily_l1186_118611


namespace root_value_theorem_l1186_118603

theorem root_value_theorem (a : ℝ) (h : a^2 + 2*a - 1 = 0) : -a^2 - 2*a + 8 = 7 := by
  sorry

end root_value_theorem_l1186_118603


namespace calculation_proof_l1186_118659

theorem calculation_proof : 
  Real.sqrt 27 - |2 * Real.sqrt 3 - 9 * Real.tan (30 * π / 180)| + (1/2)⁻¹ - (1 - π)^0 = 2 * Real.sqrt 3 + 1 := by
  sorry

end calculation_proof_l1186_118659


namespace problem_solution_l1186_118687

def star (a b : ℚ) : ℚ := 2 * a - b

theorem problem_solution (x : ℚ) (h : star x (star 1 3) = 2) : x = 1/2 := by
  sorry

end problem_solution_l1186_118687


namespace incorrect_height_calculation_l1186_118648

theorem incorrect_height_calculation (n : ℕ) (initial_avg real_avg actual_height : ℝ) 
  (h1 : n = 20)
  (h2 : initial_avg = 175)
  (h3 : real_avg = 174.25)
  (h4 : actual_height = 136) :
  ∃ (incorrect_height : ℝ),
    incorrect_height = n * initial_avg - (n * real_avg - actual_height) ∧
    incorrect_height = 151 := by
  sorry

end incorrect_height_calculation_l1186_118648


namespace train_speed_conversion_l1186_118641

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- Train's speed in meters per second -/
def train_speed_mps : ℝ := 45.0036

/-- Train's speed in kilometers per hour -/
def train_speed_kmph : ℝ := train_speed_mps * mps_to_kmph

theorem train_speed_conversion :
  train_speed_kmph = 162.013 := by sorry

end train_speed_conversion_l1186_118641


namespace derivative_at_one_l1186_118699

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = x * Real.exp x) : 
  deriv f 1 = 2 * Real.exp 1 := by
sorry

end derivative_at_one_l1186_118699


namespace smallest_number_divisibility_l1186_118620

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 551245 → ¬(∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    m + 5 = 9 * k₁ ∧ 
    m + 5 = 70 * k₂ ∧ 
    m + 5 = 25 * k₃ ∧ 
    m + 5 = 21 * k₄ ∧ 
    m + 5 = 49 * k₅)) ∧ 
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    551245 + 5 = 9 * k₁ ∧ 
    551245 + 5 = 70 * k₂ ∧ 
    551245 + 5 = 25 * k₃ ∧ 
    551245 + 5 = 21 * k₄ ∧ 
    551245 + 5 = 49 * k₅) :=
by sorry

end smallest_number_divisibility_l1186_118620


namespace strictly_increasing_function_inequality_l1186_118619

theorem strictly_increasing_function_inequality (k : ℕ) (f : ℕ → ℕ)
  (h_increasing : ∀ m n, m < n → f m < f n)
  (h_composite : ∀ n, f (f n) = k * n) :
  ∀ n : ℕ, n ≠ 0 → (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ (k + 1) * n / 2 :=
by sorry

end strictly_increasing_function_inequality_l1186_118619


namespace exists_expression_equal_100_l1186_118645

/-- Represents a sequence of digits with operators between them -/
inductive DigitExpression
  | single : Nat → DigitExpression
  | add : DigitExpression → DigitExpression → DigitExpression
  | sub : DigitExpression → DigitExpression → DigitExpression

/-- Evaluates a DigitExpression to its integer value -/
def evaluate : DigitExpression → Int
  | DigitExpression.single n => n
  | DigitExpression.add a b => evaluate a + evaluate b
  | DigitExpression.sub a b => evaluate a - evaluate b

/-- Checks if a DigitExpression uses the digits 1 to 9 in order -/
def usesDigitsInOrder : DigitExpression → Bool := sorry

/-- The main theorem stating that there exists a valid expression equaling 100 -/
theorem exists_expression_equal_100 : 
  ∃ (expr : DigitExpression), usesDigitsInOrder expr ∧ evaluate expr = 100 := by
  sorry

end exists_expression_equal_100_l1186_118645


namespace percentage_problem_l1186_118667

theorem percentage_problem (y : ℝ) (h1 : y > 0) (h2 : (y / 100) * y = 9) : y = 30 := by
  sorry

end percentage_problem_l1186_118667


namespace random_selection_probability_l1186_118632

theorem random_selection_probability (m : ℝ) : 
  m > -1 →
  (1 - (-1)) / (m - (-1)) = 2/5 →
  m = 4 := by
sorry

end random_selection_probability_l1186_118632


namespace smallest_k_for_congruence_l1186_118623

theorem smallest_k_for_congruence : 
  (∃ k : ℕ, k > 0 ∧ (201 + k) % (24 + k) = (9 + k) % (24 + k) ∧
    ∀ m : ℕ, m > 0 ∧ m < k → (201 + m) % (24 + m) ≠ (9 + m) % (24 + m)) ∧
  201 % 24 = 9 % 24 →
  (∃ k : ℕ, k = 8 ∧ k > 0 ∧ (201 + k) % (24 + k) = (9 + k) % (24 + k) ∧
    ∀ m : ℕ, m > 0 ∧ m < k → (201 + m) % (24 + m) ≠ (9 + m) % (24 + m)) :=
by sorry

end smallest_k_for_congruence_l1186_118623


namespace construct_quadrilateral_l1186_118680

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Checks if three sides of a quadrilateral are equal -/
def hasThreeEqualSides (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (M : Point) (A B : Point) : Prop := sorry

/-- Theorem: Given three points that are midpoints of three equal sides of a convex quadrilateral,
    a unique quadrilateral can be constructed -/
theorem construct_quadrilateral 
  (P Q R : Point) 
  (h_exists : ∃ (q : Quadrilateral), 
    isConvex q ∧ 
    hasThreeEqualSides q ∧
    isMidpoint P q.A q.B ∧
    isMidpoint Q q.B q.C ∧
    isMidpoint R q.C q.D) :
  ∃! (q : Quadrilateral), 
    isConvex q ∧ 
    hasThreeEqualSides q ∧
    isMidpoint P q.A q.B ∧
    isMidpoint Q q.B q.C ∧
    isMidpoint R q.C q.D :=
sorry

end construct_quadrilateral_l1186_118680


namespace sum_of_solutions_quadratic_l1186_118618

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let equation := -16 * x^2 + 72 * x - 108
  let sum_of_roots := -72 / (-16)
  equation = 0 → sum_of_roots = 9/2 := by
sorry

end sum_of_solutions_quadratic_l1186_118618


namespace largest_divisor_of_m_l1186_118694

theorem largest_divisor_of_m (m : ℕ+) (h : 54 ∣ m ^ 2) :
  ∃ (d : ℕ), d ∣ m ∧ d = 18 ∧ ∀ (k : ℕ), k ∣ m → k ≤ d :=
by sorry

end largest_divisor_of_m_l1186_118694


namespace symmetry_of_f_l1186_118624

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
axiom functional_equation : ∀ x : ℝ, f (x + 5) = f (9 - x)

-- State the theorem to be proved
theorem symmetry_of_f : 
  (∀ x : ℝ, f (7 + x) = f (7 - x)) := by sorry

end symmetry_of_f_l1186_118624


namespace inner_triangle_perimeter_is_270_l1186_118600

/-- Triangle ABC with given side lengths and parallel lines -/
structure TriangleWithParallels where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Lengths of segments formed by parallel lines
  ℓA_segment : ℝ
  ℓB_segment : ℝ
  ℓC_segment : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  AC_positive : AC > 0
  ℓA_segment_positive : ℓA_segment > 0
  ℓB_segment_positive : ℓB_segment > 0
  ℓC_segment_positive : ℓC_segment > 0
  AB_eq : AB = 150
  BC_eq : BC = 270
  AC_eq : AC = 210
  ℓA_segment_eq : ℓA_segment = 60
  ℓB_segment_eq : ℓB_segment = 50
  ℓC_segment_eq : ℓC_segment = 20

/-- The perimeter of the inner triangle formed by parallel lines -/
def innerTrianglePerimeter (t : TriangleWithParallels) : ℝ := sorry

/-- Theorem: The perimeter of the inner triangle is 270 -/
theorem inner_triangle_perimeter_is_270 (t : TriangleWithParallels) :
  innerTrianglePerimeter t = 270 := by
  sorry

end inner_triangle_perimeter_is_270_l1186_118600


namespace complement_union_theorem_l1186_118690

def U : Set ℕ := {x : ℕ | x > 0 ∧ x < 9}
def A : Set ℕ := {1, 2, 3, 4, 5, 6}
def B : Set ℕ := {4, 5, 6}

theorem complement_union_theorem :
  (U \ A) ∪ B = {4, 5, 6, 7, 8} := by sorry

end complement_union_theorem_l1186_118690


namespace man_son_age_difference_l1186_118656

/-- Given a man and his son, where the son's present age is 16, and in two years
    the man's age will be twice the age of his son, prove that the man is 18 years
    older than his son. -/
theorem man_son_age_difference (man_age son_age : ℕ) : 
  son_age = 16 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 18 := by
sorry

end man_son_age_difference_l1186_118656


namespace arithmetic_sequence_property_l1186_118644

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end arithmetic_sequence_property_l1186_118644


namespace book_selection_combination_l1186_118677

theorem book_selection_combination : ∃ n : ℕ, n * 10^9 + 306249080 = Nat.choose 20 8 := by sorry

end book_selection_combination_l1186_118677


namespace gcd_of_B_is_two_l1186_118636

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end gcd_of_B_is_two_l1186_118636


namespace vip_ticket_price_l1186_118647

/-- Represents the price of concert tickets and savings --/
structure ConcertTickets where
  savings : ℕ
  vipTickets : ℕ
  regularTickets : ℕ
  regularPrice : ℕ
  remainingMoney : ℕ

/-- Theorem: The price of each VIP ticket is $100 --/
theorem vip_ticket_price (ct : ConcertTickets)
  (h1 : ct.savings = 500)
  (h2 : ct.vipTickets = 2)
  (h3 : ct.regularTickets = 3)
  (h4 : ct.regularPrice = 50)
  (h5 : ct.remainingMoney = 150) :
  (ct.savings - ct.remainingMoney - ct.regularTickets * ct.regularPrice) / ct.vipTickets = 100 := by
  sorry


end vip_ticket_price_l1186_118647


namespace line_contains_point_l1186_118661

/-- The value of k for which the line 1 - 3kx + y = 7y contains the point (-1/3, -2) -/
theorem line_contains_point (k : ℝ) : 
  (1 - 3 * k * (-1/3) + (-2) = 7 * (-2)) ↔ k = -13 := by sorry

end line_contains_point_l1186_118661


namespace new_shoes_cost_l1186_118616

theorem new_shoes_cost (repair_cost : ℝ) (repair_duration : ℝ) (new_duration : ℝ) (percentage_increase : ℝ) :
  repair_cost = 14.50 →
  repair_duration = 1 →
  new_duration = 2 →
  percentage_increase = 0.10344827586206897 →
  ∃ (new_cost : ℝ), new_cost / new_duration = repair_cost / repair_duration * (1 + percentage_increase) ∧ new_cost = 32 :=
by sorry

end new_shoes_cost_l1186_118616


namespace f_properties_l1186_118697

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.tan x

def is_in_domain (x : ℝ) : Prop :=
  ∀ k : ℤ, x ≠ k * Real.pi + Real.pi / 2

def is_period (p : ℝ) : Prop :=
  ∀ x : ℝ, is_in_domain x → f (x + p) = f x

theorem f_properties :
  (∀ x : ℝ, is_in_domain x ↔ ∃ y : ℝ, f y = f x) ∧
  ¬ is_period Real.pi ∧
  is_period (2 * Real.pi) := by sorry

end f_properties_l1186_118697


namespace base7_multiplication_l1186_118673

/-- Converts a number from base 7 to base 10 --/
def toBase10 (n : ℕ) : ℕ :=
  sorry

/-- Converts a number from base 10 to base 7 --/
def toBase7 (n : ℕ) : ℕ :=
  sorry

/-- Multiplies two numbers in base 7 --/
def multiplyBase7 (a b : ℕ) : ℕ :=
  toBase7 (toBase10 a * toBase10 b)

theorem base7_multiplication :
  multiplyBase7 325 6 = 2624 :=
sorry

end base7_multiplication_l1186_118673


namespace equation_solution_l1186_118663

theorem equation_solution (x : ℝ) : 
  (8 / (Real.sqrt (x - 10) - 10) + 2 / (Real.sqrt (x - 10) - 5) + 
   10 / (Real.sqrt (x - 10) + 5) + 16 / (Real.sqrt (x - 10) + 10) = 0) ↔ 
  x = 60 :=
by sorry

end equation_solution_l1186_118663


namespace inequality_solution_l1186_118672

theorem inequality_solution (x : ℝ) :
  (3 * (x + 2) - 1 ≥ 5 - 2 * (x - 2)) ↔ (x ≥ 4 / 5) := by
sorry

end inequality_solution_l1186_118672


namespace shelf_filling_theorem_l1186_118658

theorem shelf_filling_theorem (A H S M E : ℕ) 
  (distinct : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ E ∧ 
              H ≠ S ∧ H ≠ M ∧ H ≠ E ∧ 
              S ≠ M ∧ S ≠ E ∧ 
              M ≠ E)
  (positive : A > 0 ∧ H > 0 ∧ S > 0 ∧ M > 0 ∧ E > 0)
  (thicker : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y > x ∧ 
             A * x + H * y = S * x + M * y ∧ 
             A * x + H * y = E * x) : 
  E = (A * M - S * H) / (M - H) :=
by sorry

end shelf_filling_theorem_l1186_118658


namespace maximum_marks_proof_l1186_118607

/-- Given that a student needs 33% of total marks to pass, got 59 marks, and failed by 40 marks,
    prove that the maximum marks are 300. -/
theorem maximum_marks_proof (pass_percentage : Real) (obtained_marks : ℕ) (failing_margin : ℕ) :
  pass_percentage = 0.33 →
  obtained_marks = 59 →
  failing_margin = 40 →
  ∃ (max_marks : ℕ), max_marks = 300 ∧ pass_percentage * max_marks = obtained_marks + failing_margin :=
by sorry

end maximum_marks_proof_l1186_118607


namespace min_lines_is_seven_l1186_118609

/-- A line in a Cartesian coordinate system --/
structure Line where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0

/-- The quadrants a line passes through --/
def quadrants (l : Line) : Set (Fin 4) :=
  sorry

/-- The minimum number of lines needed to ensure two lines pass through the same quadrants --/
def min_lines_same_quadrants : ℕ :=
  sorry

/-- Theorem stating that the minimum number of lines is 7 --/
theorem min_lines_is_seven : min_lines_same_quadrants = 7 := by
  sorry

end min_lines_is_seven_l1186_118609


namespace expression_evaluation_l1186_118652

theorem expression_evaluation : (50 - (2050 - 150)) + (2050 - (150 - 50)) = 100 := by
  sorry

end expression_evaluation_l1186_118652


namespace marbles_given_correct_l1186_118608

/-- The number of marbles Jack gave to Josh -/
def marbles_given (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that the number of marbles given is the difference between final and initial counts -/
theorem marbles_given_correct (initial final : ℕ) (h : final ≥ initial) :
  marbles_given initial final = final - initial :=
by sorry

end marbles_given_correct_l1186_118608


namespace females_with_advanced_degrees_l1186_118637

theorem females_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (employees_with_advanced_degrees : ℕ) 
  (males_with_college_degree_only : ℕ) 
  (h1 : total_employees = 180) 
  (h2 : total_females = 110) 
  (h3 : employees_with_advanced_degrees = 90) 
  (h4 : males_with_college_degree_only = 35) :
  total_females - (total_employees - employees_with_advanced_degrees) + 
  (employees_with_advanced_degrees - (total_employees - total_females - males_with_college_degree_only)) = 55 :=
by sorry

end females_with_advanced_degrees_l1186_118637


namespace rick_ironed_45_pieces_l1186_118638

/-- Represents Rick's ironing rates and time spent ironing --/
structure IroningData where
  weekday_shirt_rate : ℕ
  weekday_pants_rate : ℕ
  weekday_jacket_rate : ℕ
  weekend_shirt_rate : ℕ
  weekend_pants_rate : ℕ
  weekend_jacket_rate : ℕ
  weekday_shirt_time : ℕ
  weekday_pants_time : ℕ
  weekday_jacket_time : ℕ
  weekend_shirt_time : ℕ
  weekend_pants_time : ℕ
  weekend_jacket_time : ℕ

/-- Calculates the total number of pieces of clothing ironed --/
def total_ironed (data : IroningData) : ℕ :=
  (data.weekday_shirt_rate * data.weekday_shirt_time + data.weekend_shirt_rate * data.weekend_shirt_time) +
  (data.weekday_pants_rate * data.weekday_pants_time + data.weekend_pants_rate * data.weekend_pants_time) +
  (data.weekday_jacket_rate * data.weekday_jacket_time + data.weekend_jacket_rate * data.weekend_jacket_time)

/-- Theorem stating that Rick irons 45 pieces of clothing given the specified rates and times --/
theorem rick_ironed_45_pieces : 
  ∀ (data : IroningData), 
    data.weekday_shirt_rate = 4 ∧ 
    data.weekday_pants_rate = 3 ∧ 
    data.weekday_jacket_rate = 2 ∧
    data.weekend_shirt_rate = 5 ∧ 
    data.weekend_pants_rate = 4 ∧ 
    data.weekend_jacket_rate = 3 ∧
    data.weekday_shirt_time = 2 ∧ 
    data.weekday_pants_time = 3 ∧ 
    data.weekday_jacket_time = 1 ∧
    data.weekend_shirt_time = 3 ∧ 
    data.weekend_pants_time = 2 ∧ 
    data.weekend_jacket_time = 1 
    → total_ironed data = 45 := by
  sorry

end rick_ironed_45_pieces_l1186_118638


namespace absolute_value_properties_problem_solutions_l1186_118627

theorem absolute_value_properties :
  (∀ x y : ℝ, |x - y| = |y - x|) ∧
  (∀ x : ℝ, |x| ≥ 0) ∧
  (∀ x : ℝ, |x| = 0 ↔ x = 0) ∧
  (∀ x y : ℝ, |x + y| ≤ |x| + |y|) :=
sorry

theorem problem_solutions :
  (|3 - (-2)| = 5) ∧
  (∀ x : ℝ, |x + 2| = 3 → (x = 1 ∨ x = -5)) ∧
  (∃ m : ℝ, (∀ x : ℝ, |x - 1| + |x + 3| ≥ m) ∧ (∃ x : ℝ, |x - 1| + |x + 3| = m) ∧ m = 4) ∧
  (∃ m : ℝ, (∀ x : ℝ, |x + 1| + |x - 2| + |x - 4| ≥ m) ∧ (|2 + 1| + |2 - 2| + |2 - 4| = m) ∧ m = 5) ∧
  (∀ x y z : ℝ, (|x + 1| + |x - 2|) * (|y - 2| + |y + 1|) * (|z - 3| + |z + 1|) = 36 →
    (-3 ≤ x + y + z ∧ x + y + z ≤ 7)) :=
sorry

end absolute_value_properties_problem_solutions_l1186_118627
