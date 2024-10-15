import Mathlib

namespace NUMINAMATH_CALUDE_gcd_values_count_l3814_381435

theorem gcd_values_count (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (s : Finset ℕ), s.card = 6 ∧ ∀ (x : ℕ), x ∈ s ↔ ∃ (c d : ℕ+), Nat.gcd c d = x ∧ Nat.gcd c d * Nat.lcm c d = 360) :=
by sorry

end NUMINAMATH_CALUDE_gcd_values_count_l3814_381435


namespace NUMINAMATH_CALUDE_binary_to_decimal_1100101_l3814_381476

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1100101₂ -/
def binary_number : List Bool := [true, false, true, false, false, true, true]

/-- Theorem: The decimal equivalent of 1100101₂ is 101 -/
theorem binary_to_decimal_1100101 :
  binary_to_decimal binary_number = 101 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_1100101_l3814_381476


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3814_381441

theorem first_discount_percentage (initial_price final_price : ℝ) 
  (second_discount : ℝ) (h1 : initial_price = 560) 
  (h2 : final_price = 313.6) (h3 : second_discount = 0.3) : 
  ∃ (first_discount : ℝ), 
    first_discount = 0.2 ∧ 
    final_price = initial_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3814_381441


namespace NUMINAMATH_CALUDE_inequality_proof_l3814_381452

theorem inequality_proof (x y z : ℝ) (h : x + y + z = 1) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 3) ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3814_381452


namespace NUMINAMATH_CALUDE_k_value_at_4_l3814_381440

-- Define the polynomial h
def h (x : ℝ) : ℝ := x^3 - x + 1

-- Define k as a function of h's roots
def k (α β γ : ℝ) (x : ℝ) : ℝ := -(x - α^3) * (x - β^3) * (x - γ^3)

theorem k_value_at_4 (α β γ : ℝ) :
  h α = 0 → h β = 0 → h γ = 0 →  -- α, β, γ are roots of h
  k α β γ 0 = 1 →                -- k(0) = 1
  k α β γ 4 = -61 :=             -- k(4) = -61
by sorry

end NUMINAMATH_CALUDE_k_value_at_4_l3814_381440


namespace NUMINAMATH_CALUDE_inequality_proof_l3814_381471

theorem inequality_proof (a b : ℝ) (h : a > b) : 2 - a < 2 - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3814_381471


namespace NUMINAMATH_CALUDE_pictures_deleted_vacation_pictures_deleted_l3814_381439

theorem pictures_deleted (zoo_pics museum_pics remaining_pics : ℕ) :
  zoo_pics + museum_pics - remaining_pics =
  (zoo_pics + museum_pics) - remaining_pics :=
by sorry

theorem vacation_pictures_deleted (zoo_pics museum_pics remaining_pics : ℕ) :
  zoo_pics = 15 →
  museum_pics = 18 →
  remaining_pics = 2 →
  zoo_pics + museum_pics - remaining_pics = 31 :=
by sorry

end NUMINAMATH_CALUDE_pictures_deleted_vacation_pictures_deleted_l3814_381439


namespace NUMINAMATH_CALUDE_factorization_equality_l3814_381436

theorem factorization_equality (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3814_381436


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3814_381489

theorem polynomial_simplification (x : ℝ) :
  (3 * x^4 + 2 * x^3 - 9 * x^2 + 4 * x - 5) + (-5 * x^4 - 3 * x^3 + x^2 - 4 * x + 7) =
  -2 * x^4 - x^3 - 8 * x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3814_381489


namespace NUMINAMATH_CALUDE_empty_set_condition_l3814_381403

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | Real.sqrt (x - 3) = a * x + 1}

-- State the theorem
theorem empty_set_condition (a : ℝ) :
  IsEmpty (A a) ↔ a < -1/2 ∨ a > 1/6 := by sorry

end NUMINAMATH_CALUDE_empty_set_condition_l3814_381403


namespace NUMINAMATH_CALUDE_region_is_lower_left_l3814_381430

-- Define the line
def line (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x + y - 6 < 0

-- Define a point on the lower left side of the line
def lower_left_point (x y : ℝ) : Prop := x + y < 6

-- Theorem stating that the region is on the lower left side of the line
theorem region_is_lower_left :
  ∀ (x y : ℝ), region x y ↔ lower_left_point x y :=
sorry

end NUMINAMATH_CALUDE_region_is_lower_left_l3814_381430


namespace NUMINAMATH_CALUDE_symmetry_point_yOz_l3814_381469

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the yOz plane
def yOz_plane (p : Point3D) : Prop := p.x = 0

-- Define symmetry with respect to the yOz plane
def symmetric_to_yOz (a b : Point3D) : Prop :=
  b.x = -a.x ∧ b.y = a.y ∧ b.z = a.z

theorem symmetry_point_yOz :
  let a := Point3D.mk (-2) 4 3
  let b := Point3D.mk 2 4 3
  symmetric_to_yOz a b := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_yOz_l3814_381469


namespace NUMINAMATH_CALUDE_line_equation_from_circle_and_symmetry_l3814_381464

/-- The equation of a line given a circle and a point of symmetry -/
theorem line_equation_from_circle_and_symmetry (x y : ℝ) :
  let circle := {(x, y) | x^2 + (y - 4)^2 = 4}
  let center := (0, 4)
  let P := (2, 0)
  ∃ l : Set (ℝ × ℝ), 
    (∀ (p : ℝ × ℝ), p ∈ l ↔ x - 2*y + 3 = 0) ∧
    (∀ (q : ℝ × ℝ), q ∈ circle → ∃ (r : ℝ × ℝ), r ∈ l ∧ 
      center.1 + r.1 = q.1 + P.1 ∧ 
      center.2 + r.2 = q.2 + P.2) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_from_circle_and_symmetry_l3814_381464


namespace NUMINAMATH_CALUDE_circle_properties_l3814_381494

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Define the bisecting line
def bisecting_line (x y : ℝ) : Prop := x - y = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y = 0

-- Theorem statement
theorem circle_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    (∀ x y : ℝ, circle_equation x y → bisecting_line x y) ∧
    (∃ x y : ℝ, circle_equation x y ∧ tangent_line x y) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3814_381494


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l3814_381492

theorem sqrt_three_squared : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l3814_381492


namespace NUMINAMATH_CALUDE_marble_probability_l3814_381404

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) : 
  total = 120 → 
  p_white = 1/4 → 
  p_green = 1/3 → 
  ∃ (p_red_blue : ℚ), p_red_blue = 5/12 ∧ p_white + p_green + p_red_blue = 1 :=
by sorry

end NUMINAMATH_CALUDE_marble_probability_l3814_381404


namespace NUMINAMATH_CALUDE_roots_greater_than_two_range_l3814_381478

theorem roots_greater_than_two_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m-4)*x + (6-m) = 0 → x > 2) →
  -2 < m ∧ m ≤ 2 - 2*Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_roots_greater_than_two_range_l3814_381478


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3814_381497

/-- Given a line with equation 4x + 6y - 2z = 24 and z = 3, prove that the y-intercept is (0, 5) -/
theorem y_intercept_of_line (x y z : ℝ) :
  4 * x + 6 * y - 2 * z = 24 →
  z = 3 →
  x = 0 →
  y = 5 := by
sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3814_381497


namespace NUMINAMATH_CALUDE_consecutive_points_segment_length_l3814_381411

/-- Given 5 consecutive points on a line, prove the length of the last segment -/
theorem consecutive_points_segment_length 
  (a b c d e : ℝ) -- Points represented as real numbers
  (h_consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Ensures points are consecutive
  (h_bc_cd : c - b = 2 * (d - c)) -- bc = 2 cd
  (h_ab : b - a = 5) -- ab = 5
  (h_ac : c - a = 11) -- ac = 11
  (h_ae : e - a = 22) -- ae = 22
  : e - d = 8 := by sorry

end NUMINAMATH_CALUDE_consecutive_points_segment_length_l3814_381411


namespace NUMINAMATH_CALUDE_swap_meet_backpack_price_l3814_381474

/-- Proves that the price of each backpack sold at the swap meet was $18 --/
theorem swap_meet_backpack_price :
  ∀ (swap_meet_price : ℕ),
    (48 : ℕ) = 17 + 10 + (48 - 17 - 10) →
    (576 : ℕ) = 48 * 12 →
    (442 : ℕ) = (17 * swap_meet_price + 10 * 25 + (48 - 17 - 10) * 22) - 576 →
    swap_meet_price = 18 := by
  sorry


end NUMINAMATH_CALUDE_swap_meet_backpack_price_l3814_381474


namespace NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l3814_381412

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by sorry

end NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l3814_381412


namespace NUMINAMATH_CALUDE_coefficient_of_x4_in_expansion_l3814_381423

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def expansion_coefficient (n : ℕ) : ℕ :=
  binomial_coefficient n 4

theorem coefficient_of_x4_in_expansion : 
  expansion_coefficient 5 + expansion_coefficient 6 + expansion_coefficient 7 = 55 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x4_in_expansion_l3814_381423


namespace NUMINAMATH_CALUDE_volume_of_one_gram_l3814_381444

/-- Given a substance with a density of 200 kg per cubic meter, 
    the volume of 1 gram of this substance is 5 cubic centimeters. -/
theorem volume_of_one_gram (density : ℝ) (h : density = 200) : 
  (1 / density) * (100 ^ 3) = 5 :=
sorry

end NUMINAMATH_CALUDE_volume_of_one_gram_l3814_381444


namespace NUMINAMATH_CALUDE_stone_placement_possible_l3814_381454

/-- Represents the state of the stone placement game -/
structure GameState where
  cellStones : Nat → Bool
  bagStones : Nat

/-- Defines the allowed moves in the game -/
inductive Move
  | PlaceInFirst : Move
  | RemoveFromFirst : Move
  | PlaceInNext : Nat → Move
  | RemoveFromNext : Nat → Move

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.PlaceInFirst => sorry
  | Move.RemoveFromFirst => sorry
  | Move.PlaceInNext n => sorry
  | Move.RemoveFromNext n => sorry

/-- Checks if a cell contains a stone -/
def hasStone (state : GameState) (cell : Nat) : Bool :=
  state.cellStones cell

/-- The main theorem stating that with 10 stones, 
    we can place a stone in any cell from 1 to 1023 -/
theorem stone_placement_possible :
  ∀ n : Nat, n ≤ 1023 → 
  ∃ (moves : List Move), 
    let finalState := (moves.foldl applyMove 
      { cellStones := fun _ => false, bagStones := 10 })
    hasStone finalState n := by sorry

end NUMINAMATH_CALUDE_stone_placement_possible_l3814_381454


namespace NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l3814_381457

/-- Given that 1 yard equals 3 feet, prove that 5 cubic yards is equal to 135 cubic feet. -/
theorem cubic_yards_to_cubic_feet :
  (1 : ℝ) * (1 : ℝ) * (1 : ℝ) = 27 * (1 / 3 : ℝ) * (1 / 3 : ℝ) * (1 / 3 : ℝ) →
  5 * (1 : ℝ) * (1 : ℝ) * (1 : ℝ) = 135 * (1 / 3 : ℝ) * (1 / 3 : ℝ) * (1 / 3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l3814_381457


namespace NUMINAMATH_CALUDE_circle_intersection_perpendicular_l3814_381472

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the intersect relation between circles
variable (intersect : Circle → Circle → Prop)

-- Define the on_circle relation between points and circles
variable (on_circle : Point → Circle → Prop)

-- Define the distance function between points
variable (dist : Point → Point → ℝ)

-- Define the intersect_line_circle relation
variable (intersect_line_circle : Point → Point → Circle → Point → Prop)

-- Define the center_of_arc relation
variable (center_of_arc : Point → Point → Circle → Point → Prop)

-- Define the intersection_of_lines relation
variable (intersection_of_lines : Point → Point → Point → Point → Point → Prop)

-- Define the perpendicular relation
variable (perpendicular : Point → Point → Point → Point → Prop)

-- State the theorem
theorem circle_intersection_perpendicular 
  (C₁ C₂ : Circle) 
  (A B P Q M N C D E : Point) :
  intersect C₁ C₂ →
  on_circle P C₁ →
  on_circle Q C₂ →
  dist A P = dist A Q →
  intersect_line_circle P Q C₁ M →
  intersect_line_circle P Q C₂ N →
  center_of_arc B P C₁ C →
  center_of_arc B Q C₂ D →
  intersection_of_lines C M D N E →
  perpendicular A E C D :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_perpendicular_l3814_381472


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3814_381400

def solution_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3}

def satisfies_inequalities (x : ℤ) : Prop :=
  2 * x ≥ 3 * (x - 1) ∧ 2 - x / 2 < 5

theorem inequality_system_solution :
  ∀ x : ℤ, x ∈ solution_set ↔ satisfies_inequalities x :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3814_381400


namespace NUMINAMATH_CALUDE_complex_equation_result_l3814_381480

theorem complex_equation_result (x y : ℝ) (h : Complex.I * Real.exp (-1) + 2 = y + x * Complex.I) : x^3 + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l3814_381480


namespace NUMINAMATH_CALUDE_cookie_bags_l3814_381413

theorem cookie_bags (cookies_per_bag : ℕ) (total_cookies : ℕ) (h1 : cookies_per_bag = 41) (h2 : total_cookies = 2173) :
  total_cookies / cookies_per_bag = 53 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_l3814_381413


namespace NUMINAMATH_CALUDE_laundry_loads_count_l3814_381428

theorem laundry_loads_count :
  let wash_time : ℚ := 45 / 60  -- wash time in hours
  let dry_time : ℚ := 1  -- dry time in hours
  let total_time : ℚ := 14  -- total time in hours
  let load_time : ℚ := wash_time + dry_time  -- time per load in hours
  ∃ (loads : ℕ), (loads : ℚ) * load_time = total_time ∧ loads = 8
  := by sorry

end NUMINAMATH_CALUDE_laundry_loads_count_l3814_381428


namespace NUMINAMATH_CALUDE_green_equals_purple_l3814_381407

/-- Proves that the number of green shoe pairs is equal to the number of purple shoe pairs -/
theorem green_equals_purple (total : ℕ) (blue : ℕ) (purple : ℕ)
  (h_total : total = 1250)
  (h_blue : blue = 540)
  (h_purple : purple = 355)
  (h_sum : total = blue + purple + (total - blue - purple)) :
  total - blue - purple = purple := by
  sorry

end NUMINAMATH_CALUDE_green_equals_purple_l3814_381407


namespace NUMINAMATH_CALUDE_correct_2star_reviews_l3814_381459

/-- The number of 2-star reviews for Indigo Restaurant --/
def num_2star_reviews : ℕ := 
  let total_reviews : ℕ := 18
  let num_5star : ℕ := 6
  let num_4star : ℕ := 7
  let num_3star : ℕ := 4
  let avg_rating : ℚ := 4
  1

/-- Theorem stating that the number of 2-star reviews is correct --/
theorem correct_2star_reviews : 
  let total_reviews : ℕ := 18
  let num_5star : ℕ := 6
  let num_4star : ℕ := 7
  let num_3star : ℕ := 4
  let avg_rating : ℚ := 4
  num_2star_reviews = 1 ∧ 
  (5 * num_5star + 4 * num_4star + 3 * num_3star + 2 * num_2star_reviews : ℚ) / total_reviews = avg_rating :=
by sorry

end NUMINAMATH_CALUDE_correct_2star_reviews_l3814_381459


namespace NUMINAMATH_CALUDE_greatest_divisor_of_p_plus_one_l3814_381487

theorem greatest_divisor_of_p_plus_one (n : ℕ+) : 
  ∃ (d : ℕ), d = 6 ∧ 
  (∀ (p : ℕ), Prime p → p % 3 = 2 → ¬(p ∣ n) → d ∣ (p + 1)) ∧
  (∀ (k : ℕ), k > d → ∃ (p : ℕ), Prime p ∧ p % 3 = 2 ∧ ¬(p ∣ n) ∧ ¬(k ∣ (p + 1))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_p_plus_one_l3814_381487


namespace NUMINAMATH_CALUDE_repeating_decimal_seven_three_five_equals_fraction_l3814_381495

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def repeatingDecimalToRational (x : RepeatingDecimal) : ℚ :=
  sorry

theorem repeating_decimal_seven_three_five_equals_fraction : 
  repeatingDecimalToRational ⟨7, 35⟩ = 728 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_seven_three_five_equals_fraction_l3814_381495


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3814_381415

theorem simplify_polynomial (x : ℝ) : 
  (x - 2)^4 + 4*(x - 2)^3 + 6*(x - 2)^2 + 4*(x - 2) + 1 = (x - 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3814_381415


namespace NUMINAMATH_CALUDE_school_population_l3814_381465

theorem school_population (b g t : ℕ) : 
  b = 6 * g → g = 5 * t → b + g + t = 36 * t :=
by sorry

end NUMINAMATH_CALUDE_school_population_l3814_381465


namespace NUMINAMATH_CALUDE_difference_zero_for_sqrt_three_l3814_381490

-- Define the custom operation
def custom_op (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem difference_zero_for_sqrt_three :
  let x : ℝ := Real.sqrt 3
  let y : ℝ := Real.sqrt 3
  x - y = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_difference_zero_for_sqrt_three_l3814_381490


namespace NUMINAMATH_CALUDE_small_jars_count_l3814_381484

/-- Proves that the number of small jars is 62 given the conditions of the problem -/
theorem small_jars_count :
  ∀ (small_jars large_jars : ℕ),
    small_jars + large_jars = 100 →
    3 * small_jars + 5 * large_jars = 376 →
    small_jars = 62 := by
  sorry

end NUMINAMATH_CALUDE_small_jars_count_l3814_381484


namespace NUMINAMATH_CALUDE_complementary_angles_adjustment_l3814_381431

theorem complementary_angles_adjustment (x y : ℝ) (h1 : x + y = 90) (h2 : x / y = 3 / 7) :
  let new_x := x * 1.2
  let new_y := 90 - new_x
  (y - new_y) / y * 100 = 8.57143 := by sorry

end NUMINAMATH_CALUDE_complementary_angles_adjustment_l3814_381431


namespace NUMINAMATH_CALUDE_function_characterization_l3814_381417

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  f 0 ≠ 0 ∧
  ∀ x y : ℝ, f (x + y)^2 = 2 * f x * f y + max (f (x^2) + f (y^2)) (f (x^2 + y^2))

/-- The theorem stating that any function satisfying the equation must be either constant -1 or x - 1 -/
theorem function_characterization (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  (∀ x, f x = -1) ∨ (∀ x, f x = x - 1) := by sorry

end NUMINAMATH_CALUDE_function_characterization_l3814_381417


namespace NUMINAMATH_CALUDE_total_items_sold_l3814_381477

/-- The total revenue from all items sold -/
def total_revenue : ℝ := 2550

/-- The average price of a pair of ping pong rackets -/
def ping_pong_price : ℝ := 9.8

/-- The average price of a tennis racquet -/
def tennis_price : ℝ := 35

/-- The average price of a badminton racket -/
def badminton_price : ℝ := 15

/-- The number of each type of equipment sold -/
def items_per_type : ℕ := 42

theorem total_items_sold :
  3 * items_per_type = 126 ∧
  (ping_pong_price + tennis_price + badminton_price) * items_per_type = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_total_items_sold_l3814_381477


namespace NUMINAMATH_CALUDE_napkin_division_l3814_381470

structure Napkin :=
  (is_square : Bool)
  (folds : Nat)
  (cut_type : String)

def can_divide (n : Napkin) (parts : Nat) : Prop :=
  n.is_square ∧ n.folds = 2 ∧ n.cut_type = "straight" ∧ 
  ((parts = 2 ∨ parts = 3 ∨ parts = 4) ∨ parts ≠ 5)

theorem napkin_division (n : Napkin) (parts : Nat) :
  can_divide n parts ↔ (parts = 2 ∨ parts = 3 ∨ parts = 4) :=
sorry

end NUMINAMATH_CALUDE_napkin_division_l3814_381470


namespace NUMINAMATH_CALUDE_water_addition_proof_l3814_381447

/-- Proves that adding 23 litres of water to a 45-litre mixture with initial milk to water ratio of 4:1 results in a new mixture with milk to water ratio of 1.125 -/
theorem water_addition_proof (initial_volume : ℝ) (initial_ratio : ℚ) (water_added : ℝ) (final_ratio : ℚ) : 
  initial_volume = 45 ∧ 
  initial_ratio = 4/1 ∧ 
  water_added = 23 ∧ 
  final_ratio = 1125/1000 →
  let initial_milk := (initial_ratio / (initial_ratio + 1)) * initial_volume
  let initial_water := (1 / (initial_ratio + 1)) * initial_volume
  let final_water := initial_water + water_added
  initial_milk / final_water = final_ratio :=
by sorry

end NUMINAMATH_CALUDE_water_addition_proof_l3814_381447


namespace NUMINAMATH_CALUDE_kelly_games_given_away_l3814_381401

/-- Given that Kelly initially had 50 Nintendo games and now has 35 games left,
    prove that she gave away 15 games. -/
theorem kelly_games_given_away :
  let initial_games : ℕ := 50
  let remaining_games : ℕ := 35
  let games_given_away := initial_games - remaining_games
  games_given_away = 15 :=
by sorry

end NUMINAMATH_CALUDE_kelly_games_given_away_l3814_381401


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3814_381437

theorem sum_of_three_numbers : 4.75 + 0.303 + 0.432 = 5.485 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3814_381437


namespace NUMINAMATH_CALUDE_equal_distribution_of_eggs_l3814_381449

-- Define the number of eggs
def total_eggs : ℕ := 2 * 12

-- Define the number of people
def num_people : ℕ := 4

-- Define the function to calculate eggs per person
def eggs_per_person (total : ℕ) (people : ℕ) : ℕ := total / people

-- Theorem to prove
theorem equal_distribution_of_eggs :
  eggs_per_person total_eggs num_people = 6 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_eggs_l3814_381449


namespace NUMINAMATH_CALUDE_consecutive_non_divisors_l3814_381475

theorem consecutive_non_divisors (n : ℕ) (k : ℕ) : 
  (∀ i ∈ Finset.range 250, i ≠ k ∧ i ≠ k + 1 → n % i = 0) →
  (n % k ≠ 0 ∧ n % (k + 1) ≠ 0) →
  1 ≤ k →
  k ≤ 249 →
  k = 127 := by
sorry

end NUMINAMATH_CALUDE_consecutive_non_divisors_l3814_381475


namespace NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_four_l3814_381427

theorem fraction_equality_implies_x_equals_four (x : ℝ) :
  (x ≠ 0) → (x ≠ -2) → (6 / (x + 2) = 4 / x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_four_l3814_381427


namespace NUMINAMATH_CALUDE_valid_word_count_l3814_381442

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 3

/-- The length of the words -/
def word_length : ℕ := 20

/-- Function to calculate the number of valid words -/
def count_valid_words (n : ℕ) : ℕ :=
  alphabet_size * 2^(n - 1)

/-- Theorem stating the number of valid 20-letter words -/
theorem valid_word_count :
  count_valid_words word_length = 786432 :=
sorry

end NUMINAMATH_CALUDE_valid_word_count_l3814_381442


namespace NUMINAMATH_CALUDE_simplify_expression_l3814_381488

theorem simplify_expression (w : ℝ) : w + 2 - 3*w - 4 + 5*w + 6 - 7*w - 8 = -4*w - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3814_381488


namespace NUMINAMATH_CALUDE_divisor_problem_l3814_381410

theorem divisor_problem : ∃ (d : ℕ), d > 0 ∧ (10154 - 14) % d = 0 ∧ d = 10140 :=
by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3814_381410


namespace NUMINAMATH_CALUDE_max_disjoint_paths_iff_equal_outgoing_roads_l3814_381446

/-- Represents a city in the network -/
structure City where
  id : Nat

/-- Represents the road network -/
structure RoadNetwork where
  n : Nat
  cities : Finset City
  roads : City → City → Prop

/-- The maximum number of disjoint paths between two cities -/
def maxDisjointPaths (net : RoadNetwork) (start finish : City) : Nat :=
  sorry

/-- The number of outgoing roads from a city -/
def outgoingRoads (net : RoadNetwork) (city : City) : Nat :=
  sorry

theorem max_disjoint_paths_iff_equal_outgoing_roads
  (net : RoadNetwork) (A V : City) :
  maxDisjointPaths net A V = maxDisjointPaths net V A ↔
  outgoingRoads net A = outgoingRoads net V :=
by sorry

end NUMINAMATH_CALUDE_max_disjoint_paths_iff_equal_outgoing_roads_l3814_381446


namespace NUMINAMATH_CALUDE_smallest_triangle_leg_l3814_381414

-- Define the properties of a 30-60-90 triangle
def thirty_sixty_ninety_triangle (short_leg long_leg hypotenuse : ℝ) : Prop :=
  short_leg = hypotenuse / 2 ∧ long_leg = short_leg * Real.sqrt 3

-- Define the sequence of four connected triangles
def connected_triangles (h1 h2 h3 h4 : ℝ) : Prop :=
  ∃ (s1 l1 s2 l2 s3 l3 s4 l4 : ℝ),
    thirty_sixty_ninety_triangle s1 l1 h1 ∧
    thirty_sixty_ninety_triangle s2 l2 h2 ∧
    thirty_sixty_ninety_triangle s3 l3 h3 ∧
    thirty_sixty_ninety_triangle s4 l4 h4 ∧
    l1 = h2 ∧ l2 = h3 ∧ l3 = h4

theorem smallest_triangle_leg (h1 h2 h3 h4 : ℝ) :
  h1 = 10 → connected_triangles h1 h2 h3 h4 → l4 = 45 / 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_triangle_leg_l3814_381414


namespace NUMINAMATH_CALUDE_solve_scarf_knitting_problem_l3814_381445

/-- Represents the time (in hours) to knit various items --/
structure KnittingTime where
  hat : ℝ
  mitten : ℝ
  sock : ℝ
  sweater : ℝ

/-- The problem of finding the time to knit a scarf --/
def scarf_knitting_problem (kt : KnittingTime) (num_children : ℕ) (total_time : ℝ) : Prop :=
  let scarf_time := (total_time - num_children * (kt.hat + 2 * kt.mitten + 2 * kt.sock + kt.sweater)) / num_children
  scarf_time = 3

/-- The theorem stating the solution to the scarf knitting problem --/
theorem solve_scarf_knitting_problem :
  ∀ (kt : KnittingTime) (num_children : ℕ),
  kt.hat = 2 ∧ kt.mitten = 1 ∧ kt.sock = 1.5 ∧ kt.sweater = 6 ∧ num_children = 3 →
  scarf_knitting_problem kt num_children 48 :=
by
  sorry


end NUMINAMATH_CALUDE_solve_scarf_knitting_problem_l3814_381445


namespace NUMINAMATH_CALUDE_train_travel_time_l3814_381455

/-- Proves that a train traveling at 150 km/h for 1200 km takes 8 hours -/
theorem train_travel_time :
  ∀ (speed distance time : ℝ),
    speed = 150 ∧ distance = 1200 ∧ time = distance / speed →
    time = 8 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_l3814_381455


namespace NUMINAMATH_CALUDE_pet_store_puppies_l3814_381406

theorem pet_store_puppies (sold : ℕ) (num_cages : ℕ) (puppies_per_cage : ℕ) : 
  sold = 21 → num_cages = 9 → puppies_per_cage = 9 → 
  sold + (num_cages * puppies_per_cage) = 102 := by
sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l3814_381406


namespace NUMINAMATH_CALUDE_football_team_throwers_l3814_381450

/-- Represents the number of throwers on a football team -/
def num_throwers (total_players right_handed_players : ℕ) : ℕ :=
  total_players - (3 * right_handed_players - 2 * total_players)

theorem football_team_throwers :
  let total_players : ℕ := 70
  let right_handed_players : ℕ := 57
  num_throwers total_players right_handed_players = 31 :=
by sorry

end NUMINAMATH_CALUDE_football_team_throwers_l3814_381450


namespace NUMINAMATH_CALUDE_point_on_line_l3814_381473

/-- Given a line equation and two points on the line, prove the value of some_value -/
theorem point_on_line (m n some_value : ℝ) : 
  (m = n / 6 - 2 / 5) →  -- First point (m, n) satisfies the line equation
  (m + 3 = (n + some_value) / 6 - 2 / 5) →  -- Second point (m + 3, n + some_value) satisfies the line equation
  some_value = -12 / 5 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l3814_381473


namespace NUMINAMATH_CALUDE_solve_simultaneous_equations_l3814_381418

theorem solve_simultaneous_equations (a u : ℝ) 
  (eq1 : 3 / a + 1 / u = 7 / 2)
  (eq2 : 2 / a - 3 / u = 6) :
  a = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_solve_simultaneous_equations_l3814_381418


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3814_381453

/-- Sum of arithmetic sequence -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

/-- The sum of the first 3k terms of an arithmetic sequence with first term k^2 + k and common difference 1 -/
theorem arithmetic_sequence_sum (k : ℕ) :
  arithmetic_sum (k^2 + k) 1 (3 * k) = 3 * k^3 + (15 / 2) * k^2 - (3 / 2) * k :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3814_381453


namespace NUMINAMATH_CALUDE_triangle_problem_l3814_381481

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively

-- Define the problem statement
theorem triangle_problem (t : Triangle) 
  (h1 : Real.tan t.A = Real.sin t.B)  -- tan A = sin B
  (h2 : ∃ (D : ℝ), 2 * D = t.a ∧ t.b = t.c)  -- BD = DC (implying 2D = a and b = c)
  (h3 : t.c = t.b) :  -- AD = AB (implying c = b)
  (2 * t.a * t.c = t.b^2 + t.c^2 - t.a^2) ∧ 
  (Real.sin t.A / Real.sin t.C = 2 * Real.sqrt 2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3814_381481


namespace NUMINAMATH_CALUDE_floor_ceil_inequality_l3814_381486

theorem floor_ceil_inequality (a b c : ℝ) 
  (h : ⌈a⌉ + ⌈b⌉ + ⌈c⌉ + ⌊a + b⌋ + ⌊b + c⌋ + ⌊c + a⌋ = 2020) :
  ⌊a⌋ + ⌊b⌋ + ⌊c⌋ + ⌈a + b + c⌉ ≥ 1346 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_inequality_l3814_381486


namespace NUMINAMATH_CALUDE_quadratic_common_roots_l3814_381460

theorem quadratic_common_roots : 
  ∀ (p : ℚ) (x : ℚ),
  (9 * x^2 - 3 * (p + 6) * x + 6 * p + 5 = 0 ∧
   6 * x^2 - 3 * (p + 4) * x + 6 * p + 14 = 0) ↔
  ((p = -32/9 ∧ x = -1) ∨ (p = 32/3 ∧ x = 3)) := by sorry

end NUMINAMATH_CALUDE_quadratic_common_roots_l3814_381460


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3814_381499

/-- Time for a train to cross a bridge with another train coming from the opposite direction -/
theorem train_bridge_crossing_time
  (train1_length : ℝ)
  (train1_speed : ℝ)
  (bridge_length : ℝ)
  (train2_length : ℝ)
  (train2_speed : ℝ)
  (h1 : train1_length = 110)
  (h2 : train1_speed = 60)
  (h3 : bridge_length = 170)
  (h4 : train2_length = 90)
  (h5 : train2_speed = 45)
  : ∃ (time : ℝ), abs (time - 280 / (60 * 1000 / 3600 + 45 * 1000 / 3600)) < 0.1 :=
by
  sorry


end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3814_381499


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_thirds_l3814_381468

theorem no_solution_implies_a_leq_two_thirds (a : ℝ) :
  (∀ x : ℝ, ¬(|x + 1| < 4*x - 1 ∧ x < a)) → a ≤ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_thirds_l3814_381468


namespace NUMINAMATH_CALUDE_container_capacity_l3814_381420

theorem container_capacity (C : ℝ) 
  (h1 : 0.3 * C + 36 = 0.75 * C) : C = 80 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l3814_381420


namespace NUMINAMATH_CALUDE_equation_transformation_l3814_381405

theorem equation_transformation (x y : ℝ) :
  (2 * x - 3 * y = 6) ↔ (y = (2 * x - 6) / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l3814_381405


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l3814_381451

/-- Calculates the number of female students in a stratified sample -/
def stratified_sample_females (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ) : ℕ :=
  (female_students * sample_size) / total_students

/-- Theorem: In a class of 54 students with 18 females, a stratified sample of 9 students contains 3 females -/
theorem stratified_sample_theorem :
  stratified_sample_females 54 18 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l3814_381451


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_is_eight_l3814_381448

/-- Given two lines with slopes 1/4 and 5/4 intersecting at (1,1), and a vertical line x=5,
    the area of the triangle formed by these three lines is 8. -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    ∀ (line1 line2 : ℝ → ℝ) (x : ℝ),
      (∀ x, line1 x = 1/4 * x + 3/4) →  -- Equation of line with slope 1/4 passing through (1,1)
      (∀ x, line2 x = 5/4 * x - 1/4) →  -- Equation of line with slope 5/4 passing through (1,1)
      line1 1 = 1 →                     -- Both lines pass through (1,1)
      line2 1 = 1 →
      x = 5 →                           -- The vertical line is x=5
      area = 8                          -- The area of the formed triangle is 8

-- The proof of this theorem
theorem triangle_area_is_eight : triangle_area 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_is_eight_l3814_381448


namespace NUMINAMATH_CALUDE_product_of_numbers_l3814_381408

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 106) :
  x * y = 32 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3814_381408


namespace NUMINAMATH_CALUDE_exists_set_with_divisibility_property_l3814_381424

theorem exists_set_with_divisibility_property (n : ℕ) :
  ∃ (S : Finset ℕ), S.card = n ∧
    ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b →
      (max a b - min a b) ∣ max a b :=
sorry

end NUMINAMATH_CALUDE_exists_set_with_divisibility_property_l3814_381424


namespace NUMINAMATH_CALUDE_linear_function_composition_l3814_381429

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

-- State the theorem
theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x + 1) →
  (∀ x, f x = 2 * x + 1/3) ∨ (∀ x, f x = -2 * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_composition_l3814_381429


namespace NUMINAMATH_CALUDE_system_unique_solution_l3814_381466

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  4 * Real.sqrt y = x - a ∧ y^2 - x^2 + 2*y - 4*x - 3 = 0

-- Define the set of a values for which the system has a unique solution
def unique_solution_set : Set ℝ := {a | a < -5 ∨ a > -1}

-- Theorem statement
theorem system_unique_solution :
  ∀ a : ℝ, (∃! (x y : ℝ), system x y a) ↔ a ∈ unique_solution_set :=
sorry

end NUMINAMATH_CALUDE_system_unique_solution_l3814_381466


namespace NUMINAMATH_CALUDE_egg_tray_problem_l3814_381483

theorem egg_tray_problem (eggs_per_tray : ℕ) (total_eggs : ℕ) : 
  eggs_per_tray = 10 → total_eggs = 70 → total_eggs / eggs_per_tray = 7 := by
  sorry

end NUMINAMATH_CALUDE_egg_tray_problem_l3814_381483


namespace NUMINAMATH_CALUDE_prime_triplet_theorem_l3814_381425

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_geometric_progression (a b c : ℕ) : Prop := (b + 1)^2 = (a + 1) * (c + 1)

def valid_prime_triplet (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  a < b ∧ b < c ∧ c < 100 ∧
  is_geometric_progression a b c

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 5, 11), (2, 11, 47), (5, 11, 23), (5, 17, 53), (7, 11, 17), (7, 23, 71),
   (11, 23, 47), (17, 23, 31), (17, 41, 97), (31, 47, 71), (71, 83, 97)}

theorem prime_triplet_theorem :
  {x : ℕ × ℕ × ℕ | valid_prime_triplet x.1 x.2.1 x.2.2} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_prime_triplet_theorem_l3814_381425


namespace NUMINAMATH_CALUDE_brooks_theorem_l3814_381496

/-- A graph represented by its vertex set and an adjacency relation -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- The maximum degree of a graph -/
def maxDegree {V : Type*} (G : Graph V) : ℕ :=
  sorry

/-- The chromatic number of a graph -/
def chromaticNumber {V : Type*} (G : Graph V) : ℕ :=
  sorry

/-- Brooks' theorem: The chromatic number of a graph is at most one more than its maximum degree -/
theorem brooks_theorem {V : Type*} (G : Graph V) :
  chromaticNumber G ≤ maxDegree G + 1 :=
sorry

end NUMINAMATH_CALUDE_brooks_theorem_l3814_381496


namespace NUMINAMATH_CALUDE_amy_flash_drive_files_l3814_381422

/-- Calculates the number of remaining files on Amy's flash drive -/
def remainingFiles (musicFiles videoFiles deletedFiles : ℕ) : ℕ :=
  musicFiles + videoFiles - deletedFiles

/-- Theorem stating the number of remaining files on Amy's flash drive -/
theorem amy_flash_drive_files : remainingFiles 26 36 48 = 14 := by
  sorry

end NUMINAMATH_CALUDE_amy_flash_drive_files_l3814_381422


namespace NUMINAMATH_CALUDE_initial_blue_marbles_l3814_381493

theorem initial_blue_marbles (blue red : ℕ) : 
  (blue : ℚ) / red = 5 / 3 →
  ((blue - 10 : ℚ) / (red + 25) = 1 / 4) →
  blue = 19 := by
sorry

end NUMINAMATH_CALUDE_initial_blue_marbles_l3814_381493


namespace NUMINAMATH_CALUDE_complex_number_properties_l3814_381491

theorem complex_number_properties (i : ℂ) (h : i^2 = -1) :
  let z₁ : ℂ := 2 / (-1 + i)
  z₁^4 = -4 ∧ Complex.abs z₁ = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3814_381491


namespace NUMINAMATH_CALUDE_root_relationship_l3814_381409

theorem root_relationship (a b c x y : ℝ) (ha : a ≠ 0) :
  a * x^2 + b * x + c = 0 ∧ y^2 + b * y + a * c = 0 → x = y / a := by
  sorry

end NUMINAMATH_CALUDE_root_relationship_l3814_381409


namespace NUMINAMATH_CALUDE_time_to_install_one_window_l3814_381433

/-- Proves that the time to install one window is 5 hours -/
theorem time_to_install_one_window
  (total_windows : ℕ)
  (installed_windows : ℕ)
  (time_for_remaining : ℕ)
  (h1 : total_windows = 10)
  (h2 : installed_windows = 6)
  (h3 : time_for_remaining = 20)
  : (time_for_remaining : ℚ) / (total_windows - installed_windows : ℚ) = 5 := by
  sorry


end NUMINAMATH_CALUDE_time_to_install_one_window_l3814_381433


namespace NUMINAMATH_CALUDE_train_length_l3814_381438

/-- Given a train traveling at 270 kmph and crossing a pole in 5 seconds, its length is 375 meters. -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) (h1 : speed_kmph = 270) (h2 : crossing_time = 5) :
  let speed_ms := speed_kmph * 1000 / 3600
  speed_ms * crossing_time = 375 := by sorry

end NUMINAMATH_CALUDE_train_length_l3814_381438


namespace NUMINAMATH_CALUDE_push_mower_rate_l3814_381416

/-- Proves that the push mower's cutting rate is 1 acre per hour given the conditions of Jerry's lawn mowing scenario. -/
theorem push_mower_rate (total_acres : ℝ) (riding_mower_fraction : ℝ) (riding_mower_rate : ℝ) (total_mowing_time : ℝ) : 
  total_acres = 8 ∧ 
  riding_mower_fraction = 3/4 ∧ 
  riding_mower_rate = 2 ∧ 
  total_mowing_time = 5 → 
  (total_acres * (1 - riding_mower_fraction)) / (total_mowing_time - (total_acres * riding_mower_fraction) / riding_mower_rate) = 1 := by
  sorry

end NUMINAMATH_CALUDE_push_mower_rate_l3814_381416


namespace NUMINAMATH_CALUDE_exists_quadratic_through_point_l3814_381419

-- Define a quadratic function
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

-- State the theorem
theorem exists_quadratic_through_point :
  ∃ (a b c : ℝ), a > 0 ∧ quadratic_function a b c 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_quadratic_through_point_l3814_381419


namespace NUMINAMATH_CALUDE_second_person_work_days_l3814_381467

/-- Represents the number of days two people take to complete a task together -/
def two_people_time : ℝ := 10

/-- Represents the number of days one person takes to complete the task alone -/
def one_person_time : ℝ := 70

/-- Represents the number of days the first person took to complete the remaining work after the second person left -/
def remaining_work_time : ℝ := 42

/-- Represents the number of days the second person worked before leaving -/
def second_person_work_time : ℝ := 4

/-- Theorem stating that given the conditions, the second person worked for 4 days before leaving -/
theorem second_person_work_days :
  two_people_time = 10 ∧
  one_person_time = 70 ∧
  remaining_work_time = 42 →
  second_person_work_time = 4 :=
by sorry

end NUMINAMATH_CALUDE_second_person_work_days_l3814_381467


namespace NUMINAMATH_CALUDE_bisection_method_for_f_l3814_381462

def f (x : ℝ) := 3 * x^2 - 1

theorem bisection_method_for_f :
  ∃ (x₀ : ℝ), x₀ ∈ Set.Ioo 0 1 ∧ f x₀ = 0 ∧ ∀ x ∈ Set.Ioo 0 1, f x = 0 → x = x₀ →
  let ε : ℝ := 0.05
  let n : ℕ := 5
  let approx : ℝ := 37/64
  (∀ m : ℕ, m < n → 1 / 2^m > ε) ∧
  1 / 2^n ≤ ε ∧
  |approx - x₀| < ε :=
sorry

end NUMINAMATH_CALUDE_bisection_method_for_f_l3814_381462


namespace NUMINAMATH_CALUDE_parallelogram_point_B_trajectory_l3814_381402

-- Define the parallelogram ABCD
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

-- Define the coordinates of points A and C
def A : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (2, -3)

-- Define the line on which D moves
def D_line (x y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the trajectory of point B
def B_trajectory (x y : ℝ) : Prop := 3 * x - y - 20 = 0 ∧ x ≠ 3

-- Theorem statement
theorem parallelogram_point_B_trajectory 
  (ABCD : Parallelogram) 
  (h1 : ABCD.A = A) 
  (h2 : ABCD.C = C) 
  (h3 : ∀ x y, ABCD.D = (x, y) → D_line x y) :
  ∀ x y, ABCD.B = (x, y) → B_trajectory x y :=
sorry

end NUMINAMATH_CALUDE_parallelogram_point_B_trajectory_l3814_381402


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l3814_381421

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the total wet surface area of the given cistern -/
theorem cistern_wet_surface_area :
  let length : ℝ := 6
  let width : ℝ := 4
  let depth : ℝ := 1.25
  wetSurfaceArea length width depth = 49 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l3814_381421


namespace NUMINAMATH_CALUDE_olympic_volunteer_allocation_l3814_381479

/-- The number of ways to allocate n distinct objects into k distinct groups,
    where each group must contain at least one object. -/
def allocationSchemes (n k : ℕ) : ℕ :=
  if n < k then 0
  else (n - 1).choose (k - 1) * k.factorial

/-- The number of allocation schemes for 5 volunteers to 4 projects -/
theorem olympic_volunteer_allocation :
  allocationSchemes 5 4 = 240 :=
by sorry

end NUMINAMATH_CALUDE_olympic_volunteer_allocation_l3814_381479


namespace NUMINAMATH_CALUDE_common_tangent_implies_a_equals_one_l3814_381456

/-- Given two curves y = (1/2e)x^2 and y = a ln x with a common tangent at their common point P(s, t), prove that a = 1 -/
theorem common_tangent_implies_a_equals_one (s t a : ℝ) : 
  t = (1 / (2 * Real.exp 1)) * s^2 →  -- Point P(s, t) lies on the first curve
  t = a * Real.log s →                -- Point P(s, t) lies on the second curve
  (s / Real.exp 1 = a / s) →          -- Common tangent condition
  a = 1 := by
sorry


end NUMINAMATH_CALUDE_common_tangent_implies_a_equals_one_l3814_381456


namespace NUMINAMATH_CALUDE_number_of_workers_l3814_381434

/-- Given the wages for two groups of workers, prove the number of workers in the first group -/
theorem number_of_workers (W : ℕ) : 
  (6 * W * (9975 / (5 * 19)) = 9450) →
  (W = 15) := by
sorry

end NUMINAMATH_CALUDE_number_of_workers_l3814_381434


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l3814_381458

/-- Represents a die in the cube -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents the 4x4x4 cube made of dice -/
def Cube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Calculates the sum of visible faces on the large cube -/
def visible_sum (c : Cube) : ℕ :=
  sorry

theorem smallest_visible_sum (c : Cube) : 
  visible_sum c ≥ 144 :=
sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l3814_381458


namespace NUMINAMATH_CALUDE_triangle_exists_l3814_381443

/-- Theorem: A triangle exists given an angle, sum of two sides, and a median -/
theorem triangle_exists (α : Real) (sum_sides : Real) (median : Real) :
  ∃ (a b c : Real),
    0 < a ∧ 0 < b ∧ 0 < c ∧
    0 < α ∧ α < π ∧
    a + b = sum_sides ∧
    ((a + b) / 2)^2 + (c / 2)^2 = median^2 + ((a - b) / 2)^2 ∧
    c^2 = a^2 + b^2 - 2 * a * b * Real.cos α :=
by sorry


end NUMINAMATH_CALUDE_triangle_exists_l3814_381443


namespace NUMINAMATH_CALUDE_largest_solution_reciprocal_sixth_power_l3814_381485

/-- Given that x is the largest solution to the equation log_{2x^3} 2 + log_{4x^4} 2 = -1,
    prove that 1/x^6 = 4 -/
theorem largest_solution_reciprocal_sixth_power (x : ℝ) 
  (h : x > 0)
  (eq : Real.log 2 / Real.log (2 * x^3) + Real.log 2 / Real.log (4 * x^4) = -1)
  (largest : ∀ y > 0, Real.log 2 / Real.log (2 * y^3) + Real.log 2 / Real.log (4 * y^4) = -1 → y ≤ x) :
  1 / x^6 = 4 := by
sorry

end NUMINAMATH_CALUDE_largest_solution_reciprocal_sixth_power_l3814_381485


namespace NUMINAMATH_CALUDE_age_difference_l3814_381461

/-- Given the ages of Frank, Ty, Carla, and Karen, prove that Ty's current age is 4 years more than twice Carla's age. -/
theorem age_difference (frank_future ty_now carla_now karen_now : ℕ) : 
  karen_now = 2 →
  carla_now = karen_now + 2 →
  frank_future = 36 →
  frank_future = ty_now * 3 + 5 →
  ty_now > 2 * carla_now →
  ty_now - 2 * carla_now = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3814_381461


namespace NUMINAMATH_CALUDE_binomial_coefficient_formula_l3814_381426

theorem binomial_coefficient_formula (n k : ℕ) (h1 : k < n) (h2 : 0 < k) :
  Nat.choose n k = (Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_formula_l3814_381426


namespace NUMINAMATH_CALUDE_cat_puppy_weight_difference_l3814_381432

theorem cat_puppy_weight_difference : 
  let num_puppies : ℕ := 4
  let puppy_weight : ℝ := 7.5
  let num_cats : ℕ := 14
  let cat_weight : ℝ := 2.5
  let total_puppy_weight := (num_puppies : ℝ) * puppy_weight
  let total_cat_weight := (num_cats : ℝ) * cat_weight
  total_cat_weight - total_puppy_weight = 5
  := by sorry

end NUMINAMATH_CALUDE_cat_puppy_weight_difference_l3814_381432


namespace NUMINAMATH_CALUDE_tape_length_problem_l3814_381482

theorem tape_length_problem (original_length : ℝ) : 
  (original_length > 0) →
  (original_length * (1 - 1/5) * (1 - 3/4) = 1.5) →
  (original_length = 7.5) := by
sorry

end NUMINAMATH_CALUDE_tape_length_problem_l3814_381482


namespace NUMINAMATH_CALUDE_equation_solution_l3814_381463

theorem equation_solution : 
  ∀ x : ℝ, x > 0 → (x^(Real.log x / Real.log 5) = x^4 / 250 ↔ x = 5 ∨ x = 125) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3814_381463


namespace NUMINAMATH_CALUDE_sum_reciprocals_geq_nine_l3814_381498

theorem sum_reciprocals_geq_nine (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  1/x + 1/y + 1/z ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_geq_nine_l3814_381498
