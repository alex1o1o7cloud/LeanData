import Mathlib

namespace NUMINAMATH_CALUDE_equal_area_segment_property_l1965_196549

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid
  midline_ratio : (b + 75) / (b + 150) = 3 / 4  -- Area ratio condition for midline
  h_pos : h > 0
  b_pos : b > 0

/-- The length of the segment that divides the trapezoid into two equal areas -/
def equal_area_segment (t : Trapezoid) : ℝ :=
  225  -- This is the value of x we found in the solution

/-- The theorem to be proved -/
theorem equal_area_segment_property (t : Trapezoid) :
  ⌊(equal_area_segment t)^2 / 100⌋ = 506 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_segment_property_l1965_196549


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l1965_196590

theorem correct_quadratic_equation :
  ∀ (a b c : ℝ),
  (∃ (a' : ℝ), a' ≠ a ∧ (a' * 4 * 4 + b * 4 + c = 0) ∧ (a' * (-3) * (-3) + b * (-3) + c = 0)) →
  (∃ (c' : ℝ), c' ≠ c ∧ (a * 7 * 7 + b * 7 + c' = 0) ∧ (a * 3 * 3 + b * 3 + c' = 0)) →
  (a = 1 ∧ b = 10 ∧ c = 21) :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l1965_196590


namespace NUMINAMATH_CALUDE_total_chocolate_bars_l1965_196554

/-- The number of chocolate bars in a large box -/
def chocolate_bars_in_large_box (small_boxes : ℕ) (bars_per_small_box : ℕ) : ℕ :=
  small_boxes * bars_per_small_box

/-- Theorem: There are 640 chocolate bars in the large box -/
theorem total_chocolate_bars :
  chocolate_bars_in_large_box 20 32 = 640 := by
sorry

end NUMINAMATH_CALUDE_total_chocolate_bars_l1965_196554


namespace NUMINAMATH_CALUDE_f_monotone_increasing_local_max_condition_l1965_196523

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (x - 1) + Real.log x - (a + 1) * x

theorem f_monotone_increasing (x : ℝ) (hx : x > 0) :
  let f₁ := f 1
  (deriv f₁) x ≥ 0 := by sorry

theorem local_max_condition (a : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f a x ≤ f a 1) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_local_max_condition_l1965_196523


namespace NUMINAMATH_CALUDE_sunglasses_hat_probability_l1965_196543

theorem sunglasses_hat_probability 
  (total_sunglasses : ℕ) 
  (total_hats : ℕ) 
  (hat_also_sunglasses_prob : ℚ) :
  total_sunglasses = 80 →
  total_hats = 45 →
  hat_also_sunglasses_prob = 1/3 →
  (total_hats * hat_also_sunglasses_prob : ℚ) / total_sunglasses = 3/16 := by
sorry

end NUMINAMATH_CALUDE_sunglasses_hat_probability_l1965_196543


namespace NUMINAMATH_CALUDE_rectangle_in_circle_distances_l1965_196514

theorem rectangle_in_circle_distances (a b : ℝ) (ha : a = 24) (hb : b = 7) :
  let r := (a^2 + b^2).sqrt / 2
  let of := ((r^2 - (a/2)^2).sqrt : ℝ)
  let mf := r - of
  let mk := r + of
  ((mf^2 + (a/2)^2).sqrt, (mk^2 + (a/2)^2).sqrt) = (15, 20) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_in_circle_distances_l1965_196514


namespace NUMINAMATH_CALUDE_sum_squares_s_r_l1965_196505

def r : Finset Int := {-2, -1, 0, 1, 3}
def r_range : Finset Int := {-1, 0, 3, 4, 6}

def s_domain : Finset Int := {0, 1, 2, 3, 4, 5}
def s (x : Int) : Int := x^2 + x + 1

theorem sum_squares_s_r : 
  (r_range ∩ s_domain).sum (fun x => (s x)^2) = 611 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_s_r_l1965_196505


namespace NUMINAMATH_CALUDE_workshop_payment_digit_l1965_196542

-- Define the total payment as 2B0 where B is a single digit
def total_payment (B : Nat) : Nat := 200 + 10 * B

-- Define the condition that B is a single digit
def is_single_digit (B : Nat) : Prop := B ≥ 0 ∧ B ≤ 9

-- Define the condition that the payment is equally divisible among 15 people
def is_equally_divisible (payment : Nat) : Prop := 
  ∃ (individual_payment : Nat), payment = 15 * individual_payment

-- Theorem statement
theorem workshop_payment_digit :
  ∀ B : Nat, is_single_digit B → 
  (is_equally_divisible (total_payment B) ↔ (B = 1 ∨ B = 4)) :=
sorry

end NUMINAMATH_CALUDE_workshop_payment_digit_l1965_196542


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1965_196527

theorem solution_set_inequality (x : ℝ) : 
  Set.Icc (-2 : ℝ) 1 = {x | (1 - x) / (2 + x) ≥ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1965_196527


namespace NUMINAMATH_CALUDE_bus_stop_theorem_l1965_196556

/-- Represents the number of passengers boarding at stop i and alighting at stop j -/
def passenger_count (i j : Fin 6) : ℕ := sorry

/-- The total number of passengers on the bus between stops i and j -/
def bus_load (i j : Fin 6) : ℕ := sorry

theorem bus_stop_theorem :
  ∀ (passenger_count : Fin 6 → Fin 6 → ℕ),
  (∀ (i j : Fin 6), i < j → bus_load i j ≤ 5) →
  ∃ (A₁ B₁ A₂ B₂ : Fin 6),
    A₁ < B₁ ∧ A₂ < B₂ ∧ A₁ ≠ A₂ ∧ B₁ ≠ B₂ ∧
    passenger_count A₁ B₁ = 0 ∧ passenger_count A₂ B₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_theorem_l1965_196556


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1965_196567

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_ratio (a₁ d : ℝ) :
  a₁ ≠ 0 →
  d ≠ 0 →
  (arithmetic_sequence a₁ d 2) * (arithmetic_sequence a₁ d 8) = (arithmetic_sequence a₁ d 4)^2 →
  (arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 6 + arithmetic_sequence a₁ d 9) /
  (arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 5) = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1965_196567


namespace NUMINAMATH_CALUDE_factoring_theorem_l1965_196515

theorem factoring_theorem (x : ℝ) : x^2 * (x + 3) + 2 * (x + 3) + (x + 3) = (x^2 + 3) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_theorem_l1965_196515


namespace NUMINAMATH_CALUDE_circle_area_sum_l1965_196530

/-- The sum of the areas of all circles in an infinite sequence, where the radii form a geometric
    sequence with first term 10/3 and common ratio 4/9, is equal to 180π/13. -/
theorem circle_area_sum : 
  let r₁ : ℝ := 10 / 3  -- First term of the radii sequence
  let r : ℝ := 4 / 9    -- Common ratio of the radii sequence
  let area_sum := ∑' n, π * (r₁ * r ^ n) ^ 2  -- Sum of areas of all circles
  area_sum = 180 * π / 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_sum_l1965_196530


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l1965_196559

/-- Given a real number a, a function f, and its derivative f', 
    prove that the tangent line at the origin has slope -3 
    when f'(x) is an even function. -/
theorem tangent_line_at_origin (a : ℝ) 
  (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a*x^2 + (a-3)*x) 
  (h2 : ∀ x, (deriv f) x = f' x) 
  (h3 : ∀ x, f' x = f' (-x)) : 
  (deriv f) 0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l1965_196559


namespace NUMINAMATH_CALUDE_binary_11011_equals_27_l1965_196573

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_equals_27_l1965_196573


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l1965_196595

/-- The number of sides of a polygon given the sum of its interior angles -/
theorem polygon_sides_from_angle_sum (sum_of_angles : ℝ) : sum_of_angles = 720 → ∃ n : ℕ, n = 6 ∧ (n - 2) * 180 = sum_of_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l1965_196595


namespace NUMINAMATH_CALUDE_percentage_of_120_to_50_l1965_196571

theorem percentage_of_120_to_50 : (120 : ℝ) / 50 * 100 = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_50_l1965_196571


namespace NUMINAMATH_CALUDE_a_1_greater_than_500_l1965_196544

theorem a_1_greater_than_500 (a : Fin 10000 → ℕ)
  (h1 : ∀ i j, i < j → a i < a j)
  (h2 : a 0 > 0)
  (h3 : a 9999 < 20000)
  (h4 : ∀ i j, i < j → Nat.gcd (a i) (a j) < a i) :
  500 < a 0 := by
  sorry

end NUMINAMATH_CALUDE_a_1_greater_than_500_l1965_196544


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1965_196531

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x : ℝ, (1/3 < x ∧ x < 1/2) ↔ (a*x^2 + 5*x + c > 0)) → 
  (a = -6 ∧ c = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1965_196531


namespace NUMINAMATH_CALUDE_not_much_different_from_2023_l1965_196532

theorem not_much_different_from_2023 (x : ℝ) : 
  (x - 2023 ≤ 0) ↔ (x ≤ 2023) :=
by sorry

end NUMINAMATH_CALUDE_not_much_different_from_2023_l1965_196532


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l1965_196585

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l1965_196585


namespace NUMINAMATH_CALUDE_rotate_W_180_is_M_l1965_196578

/-- Represents an uppercase English letter -/
inductive UppercaseLetter
| W
| M

/-- Represents a geometric figure -/
class GeometricFigure where
  /-- Indicates if the figure is axisymmetric -/
  is_axisymmetric : Bool

/-- Represents the result of rotating a letter -/
def rotate_180_degrees (letter : UppercaseLetter) (is_axisymmetric : Bool) : UppercaseLetter :=
  sorry

/-- Theorem: Rotating W 180° results in M -/
theorem rotate_W_180_is_M :
  ∀ (w : UppercaseLetter) (fig : GeometricFigure),
    w = UppercaseLetter.W →
    fig.is_axisymmetric = true →
    rotate_180_degrees w fig.is_axisymmetric = UppercaseLetter.M :=
  sorry

end NUMINAMATH_CALUDE_rotate_W_180_is_M_l1965_196578


namespace NUMINAMATH_CALUDE_olga_aquarium_fish_count_l1965_196506

/-- The number of fish in Olga's aquarium -/
def fish_count : ℕ := 76

/-- The colors of fish in the aquarium -/
inductive FishColor
| Yellow | Blue | Green | Orange | Purple | Pink | Grey | Other

/-- The count of fish for each color -/
def fish_by_color (color : FishColor) : ℕ :=
  match color with
  | FishColor.Yellow => 12
  | FishColor.Blue => 6
  | FishColor.Green => 24
  | FishColor.Purple => 3
  | FishColor.Pink => 8
  | _ => 0  -- We don't have exact numbers for Orange, Grey, and Other

theorem olga_aquarium_fish_count :
  fish_count = 76 ∧
  fish_by_color FishColor.Yellow = 12 ∧
  fish_by_color FishColor.Blue = fish_by_color FishColor.Yellow / 2 ∧
  fish_by_color FishColor.Green = 2 * fish_by_color FishColor.Yellow ∧
  fish_by_color FishColor.Purple = fish_by_color FishColor.Blue / 2 ∧
  fish_by_color FishColor.Pink = fish_by_color FishColor.Green / 3 ∧
  (fish_count : ℚ) = (fish_by_color FishColor.Yellow +
                      fish_by_color FishColor.Blue +
                      fish_by_color FishColor.Green +
                      fish_by_color FishColor.Purple +
                      fish_by_color FishColor.Pink) / 0.7 :=
by sorry

#check olga_aquarium_fish_count

end NUMINAMATH_CALUDE_olga_aquarium_fish_count_l1965_196506


namespace NUMINAMATH_CALUDE_distribute_5_3_l1965_196593

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: Distributing 5 distinct objects into 3 distinct groups,
    where each group must contain at least one object, can be done in 150 ways -/
theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l1965_196593


namespace NUMINAMATH_CALUDE_least_apples_total_l1965_196518

/-- Represents the number of apples a monkey initially takes --/
structure MonkeyTake where
  apples : ℕ

/-- Represents the final distribution of apples for each monkey --/
structure MonkeyFinal where
  apples : ℕ

/-- Calculates the final number of apples for each monkey based on initial takes --/
def calculateFinal (m1 m2 m3 : MonkeyTake) : (MonkeyFinal × MonkeyFinal × MonkeyFinal) :=
  let f1 := MonkeyFinal.mk ((m1.apples / 2) + (m2.apples / 3) + (5 * m3.apples / 12))
  let f2 := MonkeyFinal.mk ((m1.apples / 4) + (m2.apples / 3) + (5 * m3.apples / 12))
  let f3 := MonkeyFinal.mk ((m1.apples / 4) + (m2.apples / 3) + (m3.apples / 6))
  (f1, f2, f3)

/-- Checks if the final distribution satisfies the 4:3:2 ratio --/
def satisfiesRatio (f1 f2 f3 : MonkeyFinal) : Prop :=
  4 * f2.apples = 3 * f1.apples ∧ 3 * f3.apples = 2 * f2.apples

/-- The main theorem stating the least possible total number of apples --/
theorem least_apples_total : 
  ∃ (m1 m2 m3 : MonkeyTake), 
    let (f1, f2, f3) := calculateFinal m1 m2 m3
    satisfiesRatio f1 f2 f3 ∧ 
    m1.apples + m2.apples + m3.apples = 336 ∧
    (∀ (n1 n2 n3 : MonkeyTake),
      let (g1, g2, g3) := calculateFinal n1 n2 n3
      satisfiesRatio g1 g2 g3 → 
      n1.apples + n2.apples + n3.apples ≥ 336) :=
sorry

end NUMINAMATH_CALUDE_least_apples_total_l1965_196518


namespace NUMINAMATH_CALUDE_expression_simplification_l1965_196525

theorem expression_simplification (q : ℝ) : 
  ((7 * q - 4) - 3 * q * 2) * 4 + (5 - 2 / 2) * (8 * q - 12) = 36 * q - 64 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1965_196525


namespace NUMINAMATH_CALUDE_factory_non_defective_percentage_l1965_196548

/-- Represents a machine in the factory -/
structure Machine where
  production_percentage : Real
  defective_rate : Real

/-- Calculates the percentage of non-defective products given a list of machines -/
def non_defective_percentage (machines : List Machine) : Real :=
  100 - (machines.map (λ m => m.production_percentage * m.defective_rate)).sum

/-- The theorem stating that the percentage of non-defective products is 95.25% -/
theorem factory_non_defective_percentage : 
  let machines : List Machine := [
    ⟨20, 2⟩,
    ⟨25, 4⟩,
    ⟨30, 5⟩,
    ⟨15, 7⟩,
    ⟨10, 8⟩
  ]
  non_defective_percentage machines = 95.25 := by
  sorry

end NUMINAMATH_CALUDE_factory_non_defective_percentage_l1965_196548


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l1965_196588

/-- The number of tickets Amanda needs to sell in total -/
def total_tickets : ℕ := 150

/-- The number of friends Amanda sells tickets to on the first day -/
def friends : ℕ := 8

/-- The number of tickets each friend buys on the first day -/
def tickets_per_friend : ℕ := 4

/-- The number of tickets Amanda sells on the second day -/
def second_day_tickets : ℕ := 45

/-- The number of tickets Amanda sells on the third day -/
def third_day_tickets : ℕ := 25

/-- The number of tickets Amanda needs to sell on the fourth and fifth day combined -/
def remaining_tickets : ℕ := total_tickets - (friends * tickets_per_friend + second_day_tickets + third_day_tickets)

theorem amanda_ticket_sales : remaining_tickets = 48 := by
  sorry

end NUMINAMATH_CALUDE_amanda_ticket_sales_l1965_196588


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1965_196504

/-- Represents a linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a quadrant in the Cartesian plane -/
inductive Quadrant
  | I   -- x > 0, y > 0
  | II  -- x < 0, y > 0
  | III -- x < 0, y < 0
  | IV  -- x > 0, y < 0

/-- Determines if a linear function passes through a given quadrant -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I   => ∃ x > 0, f.m * x + f.b > 0
  | Quadrant.II  => ∃ x < 0, f.m * x + f.b > 0
  | Quadrant.III => ∃ x < 0, f.m * x + f.b < 0
  | Quadrant.IV  => ∃ x > 0, f.m * x + f.b < 0

/-- The main theorem to prove -/
theorem linear_function_quadrants (f : LinearFunction) 
  (h1 : f.m = 4) 
  (h2 : f.b = 2) : 
  passesThrough f Quadrant.I ∧ 
  passesThrough f Quadrant.II ∧ 
  passesThrough f Quadrant.III :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l1965_196504


namespace NUMINAMATH_CALUDE_prime_square_plus_twelve_mod_twelve_l1965_196591

theorem prime_square_plus_twelve_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) :
  (p^2 + 12) % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_plus_twelve_mod_twelve_l1965_196591


namespace NUMINAMATH_CALUDE_four_of_a_kind_count_l1965_196533

/-- Represents a standard deck of 52 playing cards --/
def Deck : Type := Fin 52

/-- Represents a 5-card hand --/
def Hand : Type := Finset Deck

/-- Returns true if a hand contains exactly four cards of the same value --/
def hasFourOfAKind (h : Hand) : Prop := sorry

/-- The number of 5-card hands containing exactly four cards of the same value --/
def numHandsWithFourOfAKind : ℕ := sorry

theorem four_of_a_kind_count : numHandsWithFourOfAKind = 624 := by sorry

end NUMINAMATH_CALUDE_four_of_a_kind_count_l1965_196533


namespace NUMINAMATH_CALUDE_square_and_cube_difference_l1965_196509

theorem square_and_cube_difference (a b : ℝ) 
  (sum_eq : a + b = 8) 
  (diff_eq : a - b = 4) : 
  a^2 - b^2 = 32 ∧ a^3 - b^3 = 208 := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_difference_l1965_196509


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_equals_256_l1965_196502

/-- Given a cubic polynomial with roots p, q, r, prove that the sum of reciprocals of 
    partial fraction decomposition coefficients equals 256. -/
theorem sum_of_reciprocals_equals_256 
  (p q r : ℝ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_roots : x^3 - 27*x^2 + 98*x - 72 = (x - p) * (x - q) * (x - r)) 
  (A B C : ℝ) 
  (h_partial_fraction : ∀ (s : ℝ), s ≠ p → s ≠ q → s ≠ r → 
    1 / (s^3 - 27*s^2 + 98*s - 72) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1/A + 1/B + 1/C = 256 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_equals_256_l1965_196502


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1965_196577

-- Define the quadratic function f(x) = ax^2 + bx + c
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) :
  (∀ x, ax^2 + bx + c > 0 ↔ -2 < x ∧ x < 4) →
  f a b c 5 < f a b c (-1) ∧ f a b c (-1) < f a b c 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1965_196577


namespace NUMINAMATH_CALUDE_surface_area_after_vertex_removal_l1965_196579

/-- The surface area of a cube after removing unit cubes from its vertices -/
theorem surface_area_after_vertex_removal (side_length : ℝ) (h : side_length = 4) :
  6 * side_length^2 = 6 * side_length^2 := by sorry

end NUMINAMATH_CALUDE_surface_area_after_vertex_removal_l1965_196579


namespace NUMINAMATH_CALUDE_marks_trip_length_l1965_196540

theorem marks_trip_length (total : ℚ) 
  (h1 : total / 4 + 30 + total / 6 = total) : 
  total = 360 / 7 := by
sorry

end NUMINAMATH_CALUDE_marks_trip_length_l1965_196540


namespace NUMINAMATH_CALUDE_product_of_numbers_l1965_196575

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : x * y = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1965_196575


namespace NUMINAMATH_CALUDE_straight_line_shortest_l1965_196520

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line segment between two points
def LineSegment (p1 p2 : Point2D) : Set Point2D :=
  {p : Point2D | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = p1.x + t * (p2.x - p1.x) ∧ p.y = p1.y + t * (p2.y - p1.y)}

-- Define the length of a path between two points
def PathLength (path : Set Point2D) : ℝ := sorry

-- Theorem: The straight line segment between two points has the shortest length among all paths between those points
theorem straight_line_shortest (p1 p2 : Point2D) :
  ∀ path : Set Point2D, p1 ∈ path ∧ p2 ∈ path →
    PathLength (LineSegment p1 p2) ≤ PathLength path :=
sorry

end NUMINAMATH_CALUDE_straight_line_shortest_l1965_196520


namespace NUMINAMATH_CALUDE_square_greater_than_self_for_x_greater_than_one_l1965_196508

theorem square_greater_than_self_for_x_greater_than_one (x : ℝ) : x > 1 → x^2 > x := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_self_for_x_greater_than_one_l1965_196508


namespace NUMINAMATH_CALUDE_sequence_inequality_l1965_196584

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_pos : ∀ k, 0 ≤ k ∧ k ≤ n → 0 < a k)
  (h_eq : ∀ k, 1 ≤ k ∧ k < n → (a (k-1) + a k) * (a k + a (k+1)) = a (k-1) - a (k+1)) :
  a n < 1 / (n - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1965_196584


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1965_196551

/-- The surface area of a sphere, given properties of a hemisphere. -/
theorem sphere_surface_area (r : ℝ) (h_base_area : π * r^2 = 3) (h_hemisphere_area : 3 * π * r^2 = 9) :
  ∃ A : ℝ → ℝ, A r = 4 * π * r^2 := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1965_196551


namespace NUMINAMATH_CALUDE_eighth_term_value_l1965_196526

/-- An arithmetic sequence with the given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, 
    a 1 = 1 ∧ 
    (∀ n, a (n + 1) = a n + d) ∧
    a 3 + a 4 + a 5 + a 6 = 20

theorem eighth_term_value (a : ℕ → ℚ) (h : arithmetic_sequence a) : 
  a 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l1965_196526


namespace NUMINAMATH_CALUDE_principal_amount_proof_l1965_196538

/-- 
Given a principal amount P put at simple interest for 3 years,
if increasing the interest rate by 3% results in 81 more interest,
then P must equal 900.
-/
theorem principal_amount_proof (P : ℝ) (R : ℝ) : 
  (P * (R + 3) * 3) / 100 = (P * R * 3) / 100 + 81 → P = 900 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l1965_196538


namespace NUMINAMATH_CALUDE_evaluate_expression_l1965_196561

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/3) (hz : z = -12) :
  x^2 * y^3 * z = -1/36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1965_196561


namespace NUMINAMATH_CALUDE_solve_custom_equation_l1965_196558

-- Define the custom operation
def custom_op (m n : ℤ) : ℤ := n^2 - m

-- Theorem statement
theorem solve_custom_equation :
  ∀ x : ℤ, custom_op x 3 = 5 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_custom_equation_l1965_196558


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l1965_196589

/-- The mass of a man who causes a boat to sink by a certain depth -/
def mass_of_man (length breadth depth_sunk : ℝ) (water_density : ℝ) : ℝ :=
  length * breadth * depth_sunk * water_density

/-- Theorem stating that the mass of the man is 60 kg -/
theorem mass_of_man_on_boat :
  mass_of_man 3 2 0.01 1000 = 60 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_man_on_boat_l1965_196589


namespace NUMINAMATH_CALUDE_a_plus_b_value_l1965_196541

theorem a_plus_b_value (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 4) (h3 : a * b < 0) :
  a + b = 2 ∨ a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l1965_196541


namespace NUMINAMATH_CALUDE_library_repacking_l1965_196534

theorem library_repacking (initial_packages : ℕ) (pamphlets_per_initial_package : ℕ) (pamphlets_per_new_package : ℕ) : 
  initial_packages = 1450 →
  pamphlets_per_initial_package = 42 →
  pamphlets_per_new_package = 45 →
  (initial_packages * pamphlets_per_initial_package) % pamphlets_per_new_package = 15 :=
by
  sorry

#check library_repacking

end NUMINAMATH_CALUDE_library_repacking_l1965_196534


namespace NUMINAMATH_CALUDE_system_solution_correct_l1965_196537

theorem system_solution_correct (x y : ℝ) : 
  (x = 2 ∧ y = -2) → (x + 2*y = -2 ∧ 2*x + y = 2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_correct_l1965_196537


namespace NUMINAMATH_CALUDE_g_15_equals_274_l1965_196581

/-- The function g defined for all natural numbers -/
def g (n : ℕ) : ℕ := n^2 + 2*n + 19

/-- Theorem stating that g(15) equals 274 -/
theorem g_15_equals_274 : g 15 = 274 := by
  sorry

end NUMINAMATH_CALUDE_g_15_equals_274_l1965_196581


namespace NUMINAMATH_CALUDE_angle_c_is_right_angle_l1965_196536

theorem angle_c_is_right_angle 
  (A B C : ℝ) 
  (triangle_condition : A + B + C = Real.pi)
  (condition1 : Real.sin A + Real.cos B = Real.sqrt 2)
  (condition2 : Real.cos A + Real.sin B = Real.sqrt 2) : 
  C = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_is_right_angle_l1965_196536


namespace NUMINAMATH_CALUDE_no_factors_of_p_l1965_196594

def p (x : ℝ) : ℝ := x^4 - 3*x^2 + 5

theorem no_factors_of_p :
  (∀ x, p x ≠ (x^2 + 1) * (x^2 - 3*x + 5)) ∧
  (∀ x, p x ≠ (x - 1) * (x^3 + x^2 - 2*x - 5)) ∧
  (∀ x, p x ≠ (x^2 + 5) * (x^2 - 5)) ∧
  (∀ x, p x ≠ (x^2 + 2*x + 1) * (x^2 - 2*x + 4)) :=
by
  sorry

#check no_factors_of_p

end NUMINAMATH_CALUDE_no_factors_of_p_l1965_196594


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1965_196572

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  2 = Nat.minFac (5^23 + 7^17) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1965_196572


namespace NUMINAMATH_CALUDE_area_ratio_eab_abcd_l1965_196596

/-- Represents a trapezoid ABCD with an extended triangle EAB -/
structure ExtendedTrapezoid where
  /-- Length of base AB -/
  ab : ℝ
  /-- Length of base CD -/
  cd : ℝ
  /-- Height of the trapezoid -/
  h : ℝ
  /-- Assertion that AB = 7 -/
  ab_eq : ab = 7
  /-- Assertion that CD = 15 -/
  cd_eq : cd = 15
  /-- Assertion that the height of triangle EAB is thrice the height of the trapezoid -/
  eab_height : ℝ
  eab_height_eq : eab_height = 3 * h

/-- The ratio of the area of triangle EAB to the area of trapezoid ABCD is 21/22 -/
theorem area_ratio_eab_abcd (t : ExtendedTrapezoid) : 
  (t.ab * t.eab_height) / ((t.ab + t.cd) * t.h) = 21 / 22 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_eab_abcd_l1965_196596


namespace NUMINAMATH_CALUDE_katie_new_games_l1965_196582

/-- Given that Katie has some new games and 39 old games,
    her friends have 34 new games, and Katie has 62 more games than her friends,
    prove that Katie has 57 new games. -/
theorem katie_new_games :
  ∀ (new_games : ℕ),
  new_games + 39 = 34 + 62 →
  new_games = 57 :=
by
  sorry

end NUMINAMATH_CALUDE_katie_new_games_l1965_196582


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_five_primes_l1965_196597

theorem smallest_number_divisible_by_five_primes : ∃ n : ℕ, 
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ q₁ q₂ q₃ q₄ q₅ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0)) ∧
  n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_five_primes_l1965_196597


namespace NUMINAMATH_CALUDE_total_weight_is_120_pounds_l1965_196553

/-- The weight of a single dumbbell in pounds -/
def dumbbell_weight : ℕ := 20

/-- The number of dumbbells initially set up -/
def initial_dumbbells : ℕ := 4

/-- The number of additional dumbbells Parker adds -/
def added_dumbbells : ℕ := 2

/-- The total number of dumbbells Parker uses -/
def total_dumbbells : ℕ := initial_dumbbells + added_dumbbells

/-- Theorem: The total weight of dumbbells Parker is using is 120 pounds -/
theorem total_weight_is_120_pounds :
  total_dumbbells * dumbbell_weight = 120 := by
  sorry


end NUMINAMATH_CALUDE_total_weight_is_120_pounds_l1965_196553


namespace NUMINAMATH_CALUDE_stamp_cost_l1965_196557

theorem stamp_cost (total_cost : ℕ) (num_stamps : ℕ) (h1 : total_cost = 136) (h2 : num_stamps = 4) :
  total_cost / num_stamps = 34 := by
  sorry

end NUMINAMATH_CALUDE_stamp_cost_l1965_196557


namespace NUMINAMATH_CALUDE_simple_random_sampling_probability_l1965_196547

theorem simple_random_sampling_probability 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 100) 
  (h2 : sample_size = 30) :
  (sample_size : ℚ) / (population_size : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_random_sampling_probability_l1965_196547


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1965_196583

/-- The probability of selecting 2 non-defective pens from a box of 9 pens with 3 defective pens -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h_total : total_pens = 9) 
  (h_defective : defective_pens = 3) : 
  (Nat.choose (total_pens - defective_pens) 2 : ℚ) / (Nat.choose total_pens 2) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1965_196583


namespace NUMINAMATH_CALUDE_unique_number_pair_l1965_196599

theorem unique_number_pair : ∃! (x y : ℕ), 
  100 ≤ x ∧ x < 1000 ∧ 
  1000 ≤ y ∧ y < 10000 ∧ 
  10000 * x + y = 12 * x * y ∧
  x + y = 1083 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_pair_l1965_196599


namespace NUMINAMATH_CALUDE_max_value_expression_l1965_196507

theorem max_value_expression (x : ℝ) :
  (Real.exp (2 * x) + Real.exp (-2 * x) + 1) / (Real.exp x + Real.exp (-x) + 2) ≤ 2 * (1 - Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l1965_196507


namespace NUMINAMATH_CALUDE_c_investment_time_l1965_196517

/-- Represents the investment details of a partnership --/
structure Partnership where
  x : ℝ  -- A's investment amount
  m : ℝ  -- Number of months after which C invests
  annual_gain : ℝ 
  a_share : ℝ 

/-- Calculates the investment share of partner A --/
def a_investment_share (p : Partnership) : ℝ := p.x * 12

/-- Calculates the investment share of partner B --/
def b_investment_share (p : Partnership) : ℝ := 2 * p.x * 6

/-- Calculates the investment share of partner C --/
def c_investment_share (p : Partnership) : ℝ := 3 * p.x * (12 - p.m)

/-- Calculates the total investment share --/
def total_investment_share (p : Partnership) : ℝ :=
  a_investment_share p + b_investment_share p + c_investment_share p

/-- The main theorem stating that C invests after 3 months --/
theorem c_investment_time (p : Partnership) 
  (h1 : p.annual_gain = 18300)
  (h2 : p.a_share = 6100)
  (h3 : a_investment_share p / total_investment_share p = p.a_share / p.annual_gain) :
  p.m = 3 := by
  sorry

end NUMINAMATH_CALUDE_c_investment_time_l1965_196517


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l1965_196566

/-- The quadratic inequality -2 + 3x - 2x^2 > 0 has an empty solution set -/
theorem quadratic_inequality_empty_solution : 
  ∀ x : ℝ, ¬(-2 + 3*x - 2*x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l1965_196566


namespace NUMINAMATH_CALUDE_rectangle_width_l1965_196539

/-- Given a square and a rectangle, if the area of the square is five times the area of the rectangle,
    the perimeter of the square is 800 cm, and the length of the rectangle is 125 cm,
    then the width of the rectangle is 64 cm. -/
theorem rectangle_width (square_perimeter : ℝ) (rectangle_length : ℝ) :
  square_perimeter = 800 ∧
  rectangle_length = 125 ∧
  (square_perimeter / 4) ^ 2 = 5 * (rectangle_length * (64 : ℝ)) →
  64 = (square_perimeter / 4) ^ 2 / (5 * rectangle_length) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_l1965_196539


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l1965_196574

theorem cube_sum_theorem (a b c : ℕ) : 
  a^3 = 1 + 7 ∧ 
  3^3 = 1 + 7 + b ∧ 
  4^3 = 1 + 7 + c → 
  a + b + c = 77 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l1965_196574


namespace NUMINAMATH_CALUDE_correct_balloons_given_to_fred_l1965_196545

/-- The number of balloons Sam gave to Fred -/
def balloons_given_to_fred (sam_initial : ℕ) (mary : ℕ) (total : ℕ) : ℕ :=
  sam_initial - (total - mary)

theorem correct_balloons_given_to_fred :
  balloons_given_to_fred 6 7 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_balloons_given_to_fred_l1965_196545


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1965_196513

/-- Proves that an arithmetic sequence with first term 2, last term 3007,
    and common difference 5 has 602 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a l d n : ℕ), 
    a = 2 → 
    l = 3007 → 
    d = 5 → 
    l = a + (n - 1) * d → 
    n = 602 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1965_196513


namespace NUMINAMATH_CALUDE_lizzie_has_27_crayons_l1965_196586

def billie_crayons : ℕ := 18

def bobbie_crayons (billie : ℕ) : ℕ := 3 * billie

def lizzie_crayons (bobbie : ℕ) : ℕ := bobbie / 2

theorem lizzie_has_27_crayons :
  lizzie_crayons (bobbie_crayons billie_crayons) = 27 :=
by sorry

end NUMINAMATH_CALUDE_lizzie_has_27_crayons_l1965_196586


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l1965_196522

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_sum_theorem (p q : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (h1 : is_prime (7*p + q)) 
  (h2 : is_prime (p*q + 11)) : 
  p^q + q^p = 17 := by sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l1965_196522


namespace NUMINAMATH_CALUDE_third_grade_boys_count_l1965_196521

/-- The number of third-grade boys in an elementary school -/
def third_grade_boys (total : ℕ) (fourth_grade_excess : ℕ) (third_grade_girl_deficit : ℕ) : ℕ :=
  let third_graders := (total - fourth_grade_excess) / 2
  let third_grade_boys := (third_graders + third_grade_girl_deficit) / 2
  third_grade_boys

/-- Theorem stating the number of third-grade boys given the conditions -/
theorem third_grade_boys_count :
  third_grade_boys 531 31 22 = 136 :=
by sorry

end NUMINAMATH_CALUDE_third_grade_boys_count_l1965_196521


namespace NUMINAMATH_CALUDE_equilateral_triangle_exists_l1965_196598

-- Define a point in a plane
def Point := ℝ × ℝ

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define a predicate for an equilateral triangle
def is_equilateral_triangle (p1 p2 p3 : Point) : Prop :=
  distance p1 p2 = 1 ∧ distance p2 p3 = 1 ∧ distance p3 p1 = 1

-- Main theorem
theorem equilateral_triangle_exists 
  (points : Finset Point) 
  (h1 : points.card = 6) 
  (h2 : ∃ (pairs : Finset (Point × Point)), 
    pairs.card = 8 ∧ 
    ∀ (pair : Point × Point), pair ∈ pairs → distance pair.1 pair.2 = 1) :
  ∃ (p1 p2 p3 : Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
    is_equilateral_triangle p1 p2 p3 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_exists_l1965_196598


namespace NUMINAMATH_CALUDE_sweater_cost_l1965_196529

theorem sweater_cost (initial_amount : ℕ) (tshirt_cost : ℕ) (shoes_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 91 → 
  tshirt_cost = 6 → 
  shoes_cost = 11 → 
  remaining_amount = 50 → 
  initial_amount - remaining_amount - tshirt_cost - shoes_cost = 24 := by
sorry

end NUMINAMATH_CALUDE_sweater_cost_l1965_196529


namespace NUMINAMATH_CALUDE_janelle_gave_six_green_marbles_l1965_196516

/-- Represents the number of marbles Janelle has and gives away. -/
structure MarbleCount where
  initialGreen : Nat
  blueBags : Nat
  marblesPerBag : Nat
  giftBlue : Nat
  finalTotal : Nat

/-- Calculates the number of green marbles Janelle gave to her friend. -/
def greenMarblesGiven (m : MarbleCount) : Nat :=
  m.initialGreen - (m.finalTotal - (m.blueBags * m.marblesPerBag - m.giftBlue))

/-- Theorem stating that Janelle gave 6 green marbles to her friend. -/
theorem janelle_gave_six_green_marbles (m : MarbleCount) 
    (h1 : m.initialGreen = 26)
    (h2 : m.blueBags = 6)
    (h3 : m.marblesPerBag = 10)
    (h4 : m.giftBlue = 8)
    (h5 : m.finalTotal = 72) :
  greenMarblesGiven m = 6 := by
  sorry

#eval greenMarblesGiven { initialGreen := 26, blueBags := 6, marblesPerBag := 10, giftBlue := 8, finalTotal := 72 }

end NUMINAMATH_CALUDE_janelle_gave_six_green_marbles_l1965_196516


namespace NUMINAMATH_CALUDE_ice_pop_price_l1965_196512

theorem ice_pop_price :
  ∀ (price : ℝ),
  (∃ (xiaoming_money xiaodong_money : ℝ),
    xiaoming_money = price - 0.5 ∧
    xiaodong_money = price - 1 ∧
    xiaoming_money + xiaodong_money < price) →
  price = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_pop_price_l1965_196512


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l1965_196519

theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h1 : area = 80)
  (h2 : d1 = 16)
  (h3 : area = (d1 * d2) / 2) :
  d2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l1965_196519


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1965_196511

theorem rectangular_box_volume 
  (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 20) 
  (h3 : x * z = 12) : 
  x * y * z = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1965_196511


namespace NUMINAMATH_CALUDE_initial_population_approximation_l1965_196500

/-- The initial population of a town given its final population after a decade of growth. -/
def initial_population (final_population : ℕ) (growth_rate : ℚ) (years : ℕ) : ℚ :=
  final_population / (1 + growth_rate) ^ years

theorem initial_population_approximation :
  let final_population : ℕ := 297500
  let growth_rate : ℚ := 7 / 100
  let years : ℕ := 10
  ⌊initial_population final_population growth_rate years⌋ = 151195 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_approximation_l1965_196500


namespace NUMINAMATH_CALUDE_mrs_brown_utility_bill_l1965_196562

def utility_bill_total (fifty_bills : ℕ) (ten_bills : ℕ) : ℕ :=
  fifty_bills * 50 + ten_bills * 10

theorem mrs_brown_utility_bill : utility_bill_total 3 2 = 170 := by
  sorry

end NUMINAMATH_CALUDE_mrs_brown_utility_bill_l1965_196562


namespace NUMINAMATH_CALUDE_bridge_length_l1965_196568

/-- The length of a bridge given specific train crossing conditions -/
theorem bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 100 →
  crossing_time = 36 →
  train_speed = 40 →
  train_speed * crossing_time - train_length = 1340 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_l1965_196568


namespace NUMINAMATH_CALUDE_height_difference_l1965_196576

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Converts a Height to total inches -/
def heightToInches (h : Height) : ℕ := h.feet * 12 + h.inches

/-- Mark's height -/
def markHeight : Height := ⟨5, 3⟩

/-- Mike's height -/
def mikeHeight : Height := ⟨6, 1⟩

theorem height_difference : heightToInches mikeHeight - heightToInches markHeight = 10 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l1965_196576


namespace NUMINAMATH_CALUDE_lcm_150_540_l1965_196560

theorem lcm_150_540 : Nat.lcm 150 540 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_lcm_150_540_l1965_196560


namespace NUMINAMATH_CALUDE_point_in_region_range_l1965_196510

theorem point_in_region_range (a : ℝ) : 
  (2 * a + 3 < 3) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_range_l1965_196510


namespace NUMINAMATH_CALUDE_rice_containers_l1965_196587

theorem rice_containers (total_weight : ℚ) (container_weight : ℕ) (pound_to_ounce : ℕ) :
  total_weight = 35 / 2 →
  container_weight = 70 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce : ℚ) / container_weight = 4 :=
by sorry

end NUMINAMATH_CALUDE_rice_containers_l1965_196587


namespace NUMINAMATH_CALUDE_fraction_problem_l1965_196528

theorem fraction_problem : ∃ (a b : ℤ), 
  (a - 1 : ℚ) / b = 2/3 ∧ 
  (a - 2 : ℚ) / b = 1/2 ∧ 
  (a : ℚ) / b = 5/6 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1965_196528


namespace NUMINAMATH_CALUDE_complex_sum_l1965_196501

theorem complex_sum (a b : ℝ) (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := a + b * i
  z = (1 - i)^2 / (1 + i) →
  a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_l1965_196501


namespace NUMINAMATH_CALUDE_parabola_intersection_angles_l1965_196535

/-- Parabola C: y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Point on the parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point on the directrix -/
def directrix_point : ℝ × ℝ := (-1, 0)

/-- Line passing through P(m,0) -/
def line (m k : ℝ) (x y : ℝ) : Prop := x = k*y + m

/-- Intersection points of line and parabola -/
def intersection_points (m k : ℝ) : Prop :=
  ∃ (A B : PointOnParabola), A ≠ B ∧ line m k A.x A.y ∧ line m k B.x B.y

/-- Angle between two vectors -/
def angle (v₁ v₂ : ℝ × ℝ) : ℝ := sorry

theorem parabola_intersection_angles (m : ℝ) 
  (h_intersect : ∀ k, intersection_points m k) : 
  (m = 3 → ∀ A B : PointOnParabola, 
    line m (sorry) A.x A.y → line m (sorry) B.x B.y → 
    angle (A.x - directrix_point.1, A.y - directrix_point.2) 
          (B.x - directrix_point.1, B.y - directrix_point.2) < π/2) ∧
  (m = 3 → ∀ A B : PointOnParabola, 
    line m (sorry) A.x A.y → line m (sorry) B.x B.y → 
    angle (A.x - focus.1, A.y - focus.2) 
          (B.x - focus.1, B.y - focus.2) > π/2) ∧
  (m = 4 → ∀ A B : PointOnParabola, 
    line m (sorry) A.x A.y → line m (sorry) B.x B.y → 
    angle (A.x, A.y) (B.x, B.y) = π/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_angles_l1965_196535


namespace NUMINAMATH_CALUDE_point_not_on_line_l1965_196570

theorem point_not_on_line (a c : ℝ) (h : a * c > 0) : 
  ¬ (∃ (x y : ℝ), x = 2023 ∧ y = 0 ∧ y = a * x + c) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l1965_196570


namespace NUMINAMATH_CALUDE_count_numbers_with_three_between_100_and_499_l1965_196552

def count_numbers_with_three (lower_bound upper_bound : ℕ) : ℕ :=
  let first_digit_three := 100
  let second_digit_three := 40
  let third_digit_three := 40
  let all_digits_three := 1
  first_digit_three + second_digit_three + third_digit_three - all_digits_three

theorem count_numbers_with_three_between_100_and_499 :
  count_numbers_with_three 100 499 = 181 := by
  sorry

#eval count_numbers_with_three 100 499

end NUMINAMATH_CALUDE_count_numbers_with_three_between_100_and_499_l1965_196552


namespace NUMINAMATH_CALUDE_solution_range_l1965_196592

theorem solution_range (a : ℝ) : 
  (∃ x ∈ Set.Icc (-1) 1, x^2 + 2*x - a = 0) ↔ a ∈ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l1965_196592


namespace NUMINAMATH_CALUDE_cubic_function_property_l1965_196503

/-- Given a cubic function f(x) = ax³ + bx + 1 where ab ≠ 0, 
    if f(2016) = k, then f(-2016) = 2-k -/
theorem cubic_function_property (a b k : ℝ) (h1 : a * b ≠ 0) :
  let f := λ x : ℝ => a * x^3 + b * x + 1
  f 2016 = k → f (-2016) = 2 - k := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1965_196503


namespace NUMINAMATH_CALUDE_rectangle_area_l1965_196565

/-- Given a rectangle with length four times its width and perimeter 200 cm, its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 →
  l * w = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1965_196565


namespace NUMINAMATH_CALUDE_expression_value_l1965_196564

theorem expression_value : 
  let a : ℕ := 2017
  let b : ℕ := 2016
  let c : ℕ := 2015
  ((a^2 + b^2)^2 - c^2 - 4*a^2*b^2) / (a^2 + c - b^2) = 2018 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1965_196564


namespace NUMINAMATH_CALUDE_least_sum_problem_l1965_196580

theorem least_sum_problem (x y z : ℕ+) 
  (h1 : 4 * x.val = 6 * z.val)
  (h2 : ∀ (a b c : ℕ+), 4 * a.val = 6 * c.val → a.val + b.val + c.val ≥ 37)
  (h3 : x.val + y.val + z.val = 37) :
  y = 32 := by
sorry

end NUMINAMATH_CALUDE_least_sum_problem_l1965_196580


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1965_196563

theorem polynomial_factorization (y : ℝ) : 
  y^4 - 4*y^2 + 4 + 49*y^2 = (y^2 + 1) * (y^2 + 13) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1965_196563


namespace NUMINAMATH_CALUDE_degree_to_radian_15_l1965_196546

theorem degree_to_radian_15 : 
  (15 : ℝ) * (π / 180) = π / 12 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_15_l1965_196546


namespace NUMINAMATH_CALUDE_complex_number_properties_l1965_196569

theorem complex_number_properties : 
  (∃ (s₁ s₂ : Prop) (s₃ s₄ : Prop), 
    s₁ ∧ s₂ ∧ ¬s₃ ∧ ¬s₄ ∧
    s₁ = (∀ z₁ z₂ : ℂ, z₁ * z₂ = z₂ * z₁) ∧
    s₂ = (∀ z₁ z₂ : ℂ, Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂) ∧
    s₃ = (∀ z : ℂ, Complex.abs z = 1 → z = 1 ∨ z = -1) ∧
    s₄ = (∀ z : ℂ, (Complex.abs z)^2 = z^2)) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1965_196569


namespace NUMINAMATH_CALUDE_blue_pens_removed_l1965_196550

/-- Represents the number of pens in a jar -/
structure JarContents where
  blue : ℕ
  black : ℕ
  red : ℕ

/-- The initial contents of the jar -/
def initial_jar : JarContents := ⟨9, 21, 6⟩

/-- The number of black pens removed -/
def black_pens_removed : ℕ := 7

/-- The final number of pens in the jar after removals -/
def final_pens : ℕ := 25

/-- Theorem stating that 4 blue pens were removed -/
theorem blue_pens_removed :
  ∃ (x : ℕ),
    x = 4 ∧
    initial_jar.blue - x +
    (initial_jar.black - black_pens_removed) +
    initial_jar.red = final_pens :=
  sorry

end NUMINAMATH_CALUDE_blue_pens_removed_l1965_196550


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_of_3_5_7_l1965_196555

theorem least_four_digit_multiple_of_3_5_7 :
  (∀ n : ℕ, n ≥ 1000 ∧ n < 1050 → ¬(3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n)) ∧
  (1050 ≥ 1000 ∧ 3 ∣ 1050 ∧ 5 ∣ 1050 ∧ 7 ∣ 1050) :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_of_3_5_7_l1965_196555


namespace NUMINAMATH_CALUDE_rectangle_area_l1965_196524

theorem rectangle_area (w : ℝ) (h : w > 0) : 
  (4 * w = 4 * w) ∧ (2 * (4 * w) + 2 * w = 200) → 4 * w * w = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1965_196524
