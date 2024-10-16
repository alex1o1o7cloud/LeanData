import Mathlib

namespace NUMINAMATH_CALUDE_exists_non_integer_root_l1904_190432

theorem exists_non_integer_root (a b c : ℤ) : ∃ n : ℕ+, ¬ ∃ m : ℤ, (m : ℝ)^2 = (n : ℝ)^3 + (a : ℝ) * (n : ℝ)^2 + (b : ℝ) * (n : ℝ) + (c : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_integer_root_l1904_190432


namespace NUMINAMATH_CALUDE_max_value_inequality_l1904_190489

theorem max_value_inequality (x y z : ℝ) :
  ∃ (A : ℝ), A > 0 ∧ 
  (∀ (B : ℝ), B > A → 
    ∃ (a b c : ℝ), a^4 + b^4 + c^4 + a^2*b*c + a*b^2*c + a*b*c^2 - B*(a*b + b*c + c*a)^2 < 0) ∧
  (x^4 + y^4 + z^4 + x^2*y*z + x*y^2*z + x*y*z^2 - A*(x*y + y*z + z*x)^2 ≥ 0) ∧
  A = 2/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1904_190489


namespace NUMINAMATH_CALUDE_complex_number_problem_l1904_190484

theorem complex_number_problem (a : ℝ) : 
  let z : ℂ := Complex.I * (2 + a * Complex.I)
  (Complex.re z = -Complex.im z) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1904_190484


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l1904_190409

theorem cubic_sum_over_product (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x^3 + 1/(y+2016) = y^3 + 1/(z+2016) ∧ 
  y^3 + 1/(z+2016) = z^3 + 1/(x+2016) → 
  (x^3 + y^3 + z^3) / (x*y*z) = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l1904_190409


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1904_190425

theorem absolute_value_inequality (x y z : ℝ) :
  (|x + y + z| + |x*y + y*z + z*x| + |x*y*z| ≤ 1) →
  (max (|x|) (max (|y|) (|z|)) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1904_190425


namespace NUMINAMATH_CALUDE_hexagon_angle_sum_l1904_190441

theorem hexagon_angle_sum (x y : ℝ) : 
  x ≥ 0 → y ≥ 0 → 
  34 + 80 + 30 + 90 + x + y = 720 → 
  x + y = 36 := by
sorry

end NUMINAMATH_CALUDE_hexagon_angle_sum_l1904_190441


namespace NUMINAMATH_CALUDE_prize_stickers_l1904_190428

/-- The number of stickers Christine already has -/
def current_stickers : ℕ := 11

/-- The number of additional stickers Christine needs -/
def additional_stickers : ℕ := 19

/-- The total number of stickers needed for a prize -/
def total_stickers : ℕ := current_stickers + additional_stickers

theorem prize_stickers : total_stickers = 30 := by sorry

end NUMINAMATH_CALUDE_prize_stickers_l1904_190428


namespace NUMINAMATH_CALUDE_bathroom_tiling_savings_janet_bathroom_savings_l1904_190411

/-- Calculates the savings when choosing the least expensive tiles over the most expensive ones for a bathroom tiling project. -/
theorem bathroom_tiling_savings (wall1_length wall1_width wall2_length wall2_width wall3_length wall3_width : ℕ)
  (tiles_per_sqft : ℕ) (cheap_tile_cost expensive_tile_cost : ℚ) : ℚ :=
  let total_area := wall1_length * wall1_width + wall2_length * wall2_width + wall3_length * wall3_width
  let total_tiles := total_area * tiles_per_sqft
  let expensive_total := total_tiles * expensive_tile_cost
  let cheap_total := total_tiles * cheap_tile_cost
  expensive_total - cheap_total

/-- The savings for Janet's specific bathroom tiling project is $2,400. -/
theorem janet_bathroom_savings : 
  bathroom_tiling_savings 5 8 7 8 6 9 4 11 15 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_tiling_savings_janet_bathroom_savings_l1904_190411


namespace NUMINAMATH_CALUDE_unique_divisibility_condition_l1904_190426

theorem unique_divisibility_condition : 
  ∃! A : ℕ, A < 10 ∧ 45 % A = 0 ∧ (273100 + A * 10 + 6) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisibility_condition_l1904_190426


namespace NUMINAMATH_CALUDE_fraction_sum_equivalence_l1904_190494

theorem fraction_sum_equivalence (a b c : ℝ) 
  (h : a / (35 - a) + b / (75 - b) + c / (85 - c) = 5) :
  7 / (35 - a) + 15 / (75 - b) + 17 / (85 - c) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equivalence_l1904_190494


namespace NUMINAMATH_CALUDE_tangent_line_and_range_l1904_190401

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 18 * x + 12

-- State the theorem
theorem tangent_line_and_range :
  (∃ (m : ℝ), ∀ (x : ℝ), f' 0 * x = m * x ∧ m = 12) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 3 → f x ∈ Set.Icc 0 9) ∧
  (∃ (y : ℝ), y ∈ Set.Icc 0 9 ∧ ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_range_l1904_190401


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l1904_190469

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l1904_190469


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1023_l1904_190449

theorem largest_gcd_of_sum_1023 :
  ∃ (c d : ℕ+), c + d = 1023 ∧
  ∀ (a b : ℕ+), a + b = 1023 → Nat.gcd a b ≤ Nat.gcd c d ∧
  Nat.gcd c d = 341 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1023_l1904_190449


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1904_190427

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1904_190427


namespace NUMINAMATH_CALUDE_jerry_walk_time_approx_l1904_190408

/-- Calculates the time it takes Jerry to walk each way to the sink and recycling bin -/
def jerryWalkTime (totalCans : ℕ) (cansPerTrip : ℕ) (drainTime : ℕ) (totalTime : ℕ) : ℚ :=
  let trips := totalCans / cansPerTrip
  let totalDrainTime := drainTime * trips
  let totalWalkTime := totalTime - totalDrainTime
  let walkTimePerTrip := totalWalkTime / trips
  walkTimePerTrip / 3

/-- Theorem stating that Jerry's walk time is approximately 6.67 seconds -/
theorem jerry_walk_time_approx :
  let walkTime := jerryWalkTime 28 4 30 350
  (walkTime > 6.66) ∧ (walkTime < 6.68) := by
  sorry

end NUMINAMATH_CALUDE_jerry_walk_time_approx_l1904_190408


namespace NUMINAMATH_CALUDE_bug_path_tiles_l1904_190499

/-- The number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tiles_visited (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

theorem bug_path_tiles : tiles_visited 12 18 = 24 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l1904_190499


namespace NUMINAMATH_CALUDE_unique_k_for_perfect_square_and_cube_l1904_190479

theorem unique_k_for_perfect_square_and_cube (Z K : ℤ) 
  (h1 : 700 < Z) (h2 : Z < 1500) (h3 : K > 1) (h4 : Z = K^4) :
  (∃ a b : ℤ, Z = a^2 ∧ Z = b^3) ↔ K = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_perfect_square_and_cube_l1904_190479


namespace NUMINAMATH_CALUDE_jasper_chip_sales_l1904_190438

/-- Given the conditions of Jasper's sales, prove that he sold 27 bags of chips. -/
theorem jasper_chip_sales :
  ∀ (chips hotdogs drinks : ℕ),
    hotdogs = chips - 8 →
    drinks = hotdogs + 12 →
    drinks = 31 →
    chips = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_jasper_chip_sales_l1904_190438


namespace NUMINAMATH_CALUDE_eight_integer_lengths_l1904_190446

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths of line segments 
    from vertex E to points on the hypotenuse DF -/
def count_integer_lengths (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem stating that for the specific triangle, 
    there are exactly 8 distinct integer lengths -/
theorem eight_integer_lengths :
  ∃ (t : RightTriangle), t.de = 24 ∧ t.ef = 25 ∧ count_integer_lengths t = 8 :=
by sorry

end NUMINAMATH_CALUDE_eight_integer_lengths_l1904_190446


namespace NUMINAMATH_CALUDE_reconstruction_possible_iff_odd_l1904_190471

/-- A regular n-gon with numbers assigned to its vertices and center -/
structure NumberedPolygon (n : ℕ) where
  vertex_numbers : Fin n → ℕ
  center_number : ℕ

/-- The set of triples formed by connecting the center to all vertices -/
def triples (p : NumberedPolygon n) : Finset (Finset ℕ) :=
  sorry

/-- A function that attempts to reconstruct the original numbers -/
def reconstruct (t : Finset (Finset ℕ)) : Option (NumberedPolygon n) :=
  sorry

theorem reconstruction_possible_iff_odd (n : ℕ) :
  (∀ p : NumberedPolygon n, ∃! q : NumberedPolygon n, triples p = triples q) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_reconstruction_possible_iff_odd_l1904_190471


namespace NUMINAMATH_CALUDE_phone_plan_charge_equality_l1904_190487

/-- Represents the per-minute charge for plan B -/
def plan_b_charge : ℝ := 0.20

/-- Represents the fixed charge for the first 9 minutes in plan A -/
def plan_a_fixed_charge : ℝ := 0.60

/-- Represents the duration (in minutes) where both plans charge the same amount -/
def equal_charge_duration : ℝ := 3

theorem phone_plan_charge_equality :
  plan_a_fixed_charge = plan_b_charge * equal_charge_duration := by
  sorry

#check phone_plan_charge_equality

end NUMINAMATH_CALUDE_phone_plan_charge_equality_l1904_190487


namespace NUMINAMATH_CALUDE_sandwich_cost_is_90_cents_l1904_190435

/-- The cost of making a sandwich with two slices of bread, one slice of ham, and one slice of cheese -/
def sandwich_cost (bread_cost cheese_cost ham_cost : ℚ) : ℚ :=
  2 * bread_cost + cheese_cost + ham_cost

/-- Theorem stating that the cost of making a sandwich is 90 cents -/
theorem sandwich_cost_is_90_cents :
  sandwich_cost 0.15 0.35 0.25 * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_is_90_cents_l1904_190435


namespace NUMINAMATH_CALUDE_max_distance_between_circles_l1904_190478

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C₂ with equation x² + y² - 4x - 5 = 0 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

/-- The maximum distance between any point on C₁ and any point on C₂ is 13 -/
theorem max_distance_between_circles :
  ∃ (m₁ m₂ n₁ n₂ : ℝ), C₁ m₁ m₂ ∧ C₂ n₁ n₂ ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ Real.sqrt ((m₁ - n₁)^2 + (m₂ - n₂)^2) ∧
  Real.sqrt ((m₁ - n₁)^2 + (m₂ - n₂)^2) = 13 :=
sorry

end NUMINAMATH_CALUDE_max_distance_between_circles_l1904_190478


namespace NUMINAMATH_CALUDE_payment_calculation_l1904_190477

theorem payment_calculation (payment_per_room : ℚ) (rooms_cleaned : ℚ) : 
  payment_per_room = 15 / 4 →
  rooms_cleaned = 9 / 5 →
  payment_per_room * rooms_cleaned = 27 / 4 := by
sorry

end NUMINAMATH_CALUDE_payment_calculation_l1904_190477


namespace NUMINAMATH_CALUDE_xy_less_than_one_necessary_not_sufficient_l1904_190410

theorem xy_less_than_one_necessary_not_sufficient (x y : ℝ) :
  (0 < x ∧ x < 1/y) → (x*y < 1) ∧
  ¬(∀ x y : ℝ, x*y < 1 → (0 < x ∧ x < 1/y)) :=
by sorry

end NUMINAMATH_CALUDE_xy_less_than_one_necessary_not_sufficient_l1904_190410


namespace NUMINAMATH_CALUDE_shells_found_fourth_day_l1904_190415

/-- The number of shells Shara found on the fourth day of her vacation. -/
def shells_fourth_day (initial_shells : ℕ) (shells_per_day : ℕ) (vacation_days : ℕ) (total_shells : ℕ) : ℕ :=
  total_shells - (initial_shells + shells_per_day * vacation_days)

/-- Theorem stating that Shara found 6 shells on the fourth day of her vacation. -/
theorem shells_found_fourth_day :
  shells_fourth_day 20 5 3 41 = 6 := by
  sorry

end NUMINAMATH_CALUDE_shells_found_fourth_day_l1904_190415


namespace NUMINAMATH_CALUDE_pascal_triangle_46th_row_45th_number_l1904_190486

theorem pascal_triangle_46th_row_45th_number : 
  let n : ℕ := 46  -- The row number (0-indexed)
  let k : ℕ := 44  -- The position in the row (0-indexed)
  Nat.choose n k = 1035 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_46th_row_45th_number_l1904_190486


namespace NUMINAMATH_CALUDE_pencils_per_child_l1904_190460

theorem pencils_per_child (total_children : ℕ) (total_pencils : ℕ) 
  (h1 : total_children = 8) 
  (h2 : total_pencils = 16) : 
  total_pencils / total_children = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_child_l1904_190460


namespace NUMINAMATH_CALUDE_january_oil_bill_l1904_190419

theorem january_oil_bill (february_bill january_bill : ℚ) : 
  (february_bill / january_bill = 5 / 4) →
  ((february_bill + 45) / january_bill = 3 / 2) →
  january_bill = 180 := by
sorry

end NUMINAMATH_CALUDE_january_oil_bill_l1904_190419


namespace NUMINAMATH_CALUDE_brad_speed_is_6_l1904_190443

-- Define the given conditions
def maxwell_speed : ℝ := 4
def brad_delay : ℝ := 1
def total_distance : ℝ := 34
def meeting_time : ℝ := 4

-- Define Brad's speed as a variable
def brad_speed : ℝ := sorry

-- Theorem to prove
theorem brad_speed_is_6 : brad_speed = 6 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_brad_speed_is_6_l1904_190443


namespace NUMINAMATH_CALUDE_travel_time_calculation_l1904_190472

theorem travel_time_calculation (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 540)
  (h2 : speed1 = 45)
  (h3 : speed2 = 30) :
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l1904_190472


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l1904_190423

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  common_difference : Nat
  start : Nat

/-- Checks if a number is in the systematic sample -/
def in_sample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, k < s.sample_size ∧ n = (s.start + k * s.common_difference) % s.total_students

/-- The main theorem to prove -/
theorem systematic_sample_theorem :
  ∃ (s : SystematicSample),
    s.total_students = 52 ∧
    s.sample_size = 4 ∧
    in_sample s 6 ∧
    in_sample s 32 ∧
    in_sample s 45 ∧
    in_sample s 19 :=
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l1904_190423


namespace NUMINAMATH_CALUDE_abc_def_ratio_l1904_190464

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6) :
  a * b * c / (d * e * f) = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l1904_190464


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l1904_190452

theorem existence_of_special_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j : Fin 100, i < j → a i < a j) ∧ 
    (∀ k : Fin 100, 2 ≤ k.val → k.val ≤ 100 → 
      Nat.lcm (a ⟨k.val - 1, sorry⟩) (a k) > Nat.lcm (a k) (a ⟨k.val + 1, sorry⟩)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l1904_190452


namespace NUMINAMATH_CALUDE_correct_calculation_l1904_190481

theorem correct_calculation (x : ℝ) : x - 20 = 52 → x / 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1904_190481


namespace NUMINAMATH_CALUDE_smallest_number_l1904_190492

def base_6_to_decimal (x : ℕ) : ℕ := x

def base_4_to_decimal (x : ℕ) : ℕ := x

def base_2_to_decimal (x : ℕ) : ℕ := x

theorem smallest_number 
  (h1 : base_6_to_decimal 210 = 78)
  (h2 : base_4_to_decimal 100 = 16)
  (h3 : base_2_to_decimal 111111 = 63) :
  base_4_to_decimal 100 < base_6_to_decimal 210 ∧ 
  base_4_to_decimal 100 < base_2_to_decimal 111111 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l1904_190492


namespace NUMINAMATH_CALUDE_remainder_theorem_l1904_190424

theorem remainder_theorem (P D Q R D' Q'' R'' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q^2 = D' * Q'' + R'') :
  P % (D * D') = R := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1904_190424


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1904_190434

theorem quadratic_equation_solution (b : ℚ) : 
  ((-4 : ℚ)^2 + b * (-4 : ℚ) - 45 = 0) → b = -29/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1904_190434


namespace NUMINAMATH_CALUDE_red_markers_count_l1904_190468

/-- Given a total number of markers and a number of blue markers, 
    calculate the number of red markers. -/
def red_markers (total : ℝ) (blue : ℕ) : ℝ :=
  total - blue

/-- Prove that given 64.0 total markers and 23 blue markers, 
    the number of red markers is 41. -/
theorem red_markers_count : red_markers 64.0 23 = 41 := by
  sorry

end NUMINAMATH_CALUDE_red_markers_count_l1904_190468


namespace NUMINAMATH_CALUDE_domino_pile_sum_theorem_l1904_190421

/-- Definition of a domino set -/
def DominoSet := { n : ℕ | n ≤ 28 }

/-- The total sum of points on all domino pieces -/
def totalSum : ℕ := 168

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- A function that checks if four numbers are consecutive -/
def areConsecutive (a b c d : ℕ) : Prop := b = a + 1 ∧ c = b + 1 ∧ d = c + 1

/-- The main theorem to be proved -/
theorem domino_pile_sum_theorem :
  ∃ (a b c d : ℕ), 
    isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧
    areConsecutive a b c d ∧
    a + b + c + d = totalSum :=
sorry

end NUMINAMATH_CALUDE_domino_pile_sum_theorem_l1904_190421


namespace NUMINAMATH_CALUDE_current_speed_l1904_190496

/-- Proves that given a woman swimming downstream 81 km in 9 hours and upstream 36 km in 9 hours, the speed of the current is 2.5 km/h. -/
theorem current_speed (v : ℝ) (c : ℝ) : 
  (v + c) * 9 = 81 → 
  (v - c) * 9 = 36 → 
  c = 2.5 := by
sorry

end NUMINAMATH_CALUDE_current_speed_l1904_190496


namespace NUMINAMATH_CALUDE_homework_problem_distribution_l1904_190493

theorem homework_problem_distribution (total : ℕ) (finished : ℕ) (pages : ℕ) 
  (h1 : total = 60) 
  (h2 : finished = 20) 
  (h3 : pages = 5) 
  (h4 : pages > 0) :
  (total - finished) / pages = 8 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_distribution_l1904_190493


namespace NUMINAMATH_CALUDE_mobius_decomposition_l1904_190416

theorem mobius_decomposition 
  (a b c d : ℂ) 
  (h : a * d - b * c ≠ 0) : 
  ∃ (p q R : ℂ), ∀ (z : ℂ), 
    (a * z + b) / (c * z + d) = p + R / (z + q) := by
  sorry

end NUMINAMATH_CALUDE_mobius_decomposition_l1904_190416


namespace NUMINAMATH_CALUDE_abs_reciprocal_neg_six_l1904_190480

theorem abs_reciprocal_neg_six : |1 / (-6)| = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_reciprocal_neg_six_l1904_190480


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_nineteen_thirds_l1904_190406

theorem greatest_integer_less_than_negative_nineteen_thirds :
  ⌊-19/3⌋ = -7 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_nineteen_thirds_l1904_190406


namespace NUMINAMATH_CALUDE_cheryl_walking_distance_l1904_190439

/-- Calculates the total distance Cheryl walked based on her journey segments -/
def total_distance_walked (
  speed1 : ℝ) (time1 : ℝ)
  (speed2 : ℝ) (time2 : ℝ)
  (speed3 : ℝ) (time3 : ℝ)
  (speed4 : ℝ) (time4 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3 + speed4 * time4

/-- Theorem stating that Cheryl's total walking distance is 32 miles -/
theorem cheryl_walking_distance :
  total_distance_walked 2 3 4 2 1 3 3 5 = 32 := by
  sorry

#eval total_distance_walked 2 3 4 2 1 3 3 5

end NUMINAMATH_CALUDE_cheryl_walking_distance_l1904_190439


namespace NUMINAMATH_CALUDE_square_difference_l1904_190466

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x - 2) * (x + 2) = 9797 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1904_190466


namespace NUMINAMATH_CALUDE_flooring_problem_l1904_190475

theorem flooring_problem (room_length room_width box_area boxes_needed : ℕ) 
  (h1 : room_length = 16)
  (h2 : room_width = 20)
  (h3 : box_area = 10)
  (h4 : boxes_needed = 7) :
  room_length * room_width - boxes_needed * box_area = 250 :=
by sorry

end NUMINAMATH_CALUDE_flooring_problem_l1904_190475


namespace NUMINAMATH_CALUDE_seventh_twenty_ninth_725th_digit_l1904_190429

def decimal_representation (n d : ℕ) : List ℕ := sorry

theorem seventh_twenty_ninth_725th_digit :
  let rep := decimal_representation 7 29
  -- The decimal representation has a period of 28 digits
  ∀ i, rep.get? i = rep.get? (i + 28)
  -- The 725th digit is 6
  → rep.get? 724 = some 6 := by
  sorry

end NUMINAMATH_CALUDE_seventh_twenty_ninth_725th_digit_l1904_190429


namespace NUMINAMATH_CALUDE_max_n_value_l1904_190457

theorem max_n_value (a b c d : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (h4 : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ n / (a - d)) :
  n ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_n_value_l1904_190457


namespace NUMINAMATH_CALUDE_power_multiplication_addition_l1904_190495

theorem power_multiplication_addition : 2^4 * 3^2 * 5^2 + 7^3 = 3943 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_addition_l1904_190495


namespace NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l1904_190403

theorem product_of_sums_equal_difference_of_powers : 
  (5 + 3) * (5^2 + 3^2) * (5^4 + 3^4) * (5^8 + 3^8) * 
  (5^16 + 3^16) * (5^32 + 3^32) * (5^64 + 3^64) = 5^128 - 3^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l1904_190403


namespace NUMINAMATH_CALUDE_middle_is_four_l1904_190413

/-- Represents a trio of integers -/
structure Trio :=
  (left : ℕ)
  (middle : ℕ)
  (right : ℕ)

/-- Checks if a trio satisfies the given conditions -/
def validTrio (t : Trio) : Prop :=
  t.left < t.middle ∧ t.middle < t.right ∧ t.left + t.middle + t.right = 15

/-- Casey cannot determine the other two numbers -/
def caseyUncertain (t : Trio) : Prop :=
  ∃ t' : Trio, t' ≠ t ∧ validTrio t' ∧ t'.left = t.left

/-- Tracy cannot determine the other two numbers -/
def tracyUncertain (t : Trio) : Prop :=
  ∃ t' : Trio, t' ≠ t ∧ validTrio t' ∧ t'.right = t.right

/-- Stacy cannot determine the other two numbers -/
def stacyUncertain (t : Trio) : Prop :=
  ∃ t' : Trio, t' ≠ t ∧ validTrio t' ∧ t'.middle = t.middle

/-- The main theorem stating that the middle number must be 4 -/
theorem middle_is_four :
  ∀ t : Trio, validTrio t →
    caseyUncertain t → tracyUncertain t → stacyUncertain t →
    t.middle = 4 :=
by sorry

end NUMINAMATH_CALUDE_middle_is_four_l1904_190413


namespace NUMINAMATH_CALUDE_inequality_proof_l1904_190402

theorem inequality_proof (x a b : ℝ) (h1 : x < b) (h2 : b < a) (h3 : a < 0) :
  x^2 > a*b ∧ a*b > a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1904_190402


namespace NUMINAMATH_CALUDE_inequality_proof_l1904_190431

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1904_190431


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l1904_190445

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) :
  (n > 2) →
  (exterior_angle > 0) →
  (exterior_angle < 180) →
  (n * exterior_angle = 360) →
  (exterior_angle = 60) →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l1904_190445


namespace NUMINAMATH_CALUDE_amanda_pay_calculation_l1904_190455

/-- Calculates the amount Amanda receives if she doesn't finish her sales report --/
theorem amanda_pay_calculation (hourly_rate : ℝ) (hours_worked : ℝ) (withholding_percentage : ℝ) : 
  hourly_rate = 50 →
  hours_worked = 10 →
  withholding_percentage = 0.2 →
  hourly_rate * hours_worked * (1 - withholding_percentage) = 400 := by
sorry

end NUMINAMATH_CALUDE_amanda_pay_calculation_l1904_190455


namespace NUMINAMATH_CALUDE_shirt_discount_percentage_l1904_190444

theorem shirt_discount_percentage (original_price discounted_price : ℝ) 
  (h1 : original_price = 80)
  (h2 : discounted_price = 68) :
  (original_price - discounted_price) / original_price * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_shirt_discount_percentage_l1904_190444


namespace NUMINAMATH_CALUDE_symmetry_about_xOy_plane_l1904_190488

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry about the xOy plane -/
def symmetricAboutXOY (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

theorem symmetry_about_xOy_plane (p : Point3D) :
  symmetricAboutXOY p = ⟨p.x, p.y, -p.z⟩ := by
  sorry

#check symmetry_about_xOy_plane

end NUMINAMATH_CALUDE_symmetry_about_xOy_plane_l1904_190488


namespace NUMINAMATH_CALUDE_election_winner_votes_l1904_190420

/-- In an election with two candidates, where the winner received 62% of votes
    and won by 408 votes, the number of votes cast for the winning candidate is 1054. -/
theorem election_winner_votes (total_votes : ℕ) : 
  (total_votes : ℝ) * 0.62 - (total_votes : ℝ) * 0.38 = 408 →
  (total_votes : ℝ) * 0.62 = 1054 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l1904_190420


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l1904_190400

/-- Two lines intersecting the y-axis at the same non-zero point -/
structure IntersectingLines where
  b : ℝ
  s : ℝ
  t : ℝ
  hb : b ≠ 0
  h1 : 0 = (5/2) * s + b
  h2 : 0 = (7/3) * t + b

/-- The ratio of x-intercepts is 14/15 -/
theorem x_intercept_ratio (l : IntersectingLines) : l.s / l.t = 14 / 15 := by
  sorry

#check x_intercept_ratio

end NUMINAMATH_CALUDE_x_intercept_ratio_l1904_190400


namespace NUMINAMATH_CALUDE_area_at_stage_8_l1904_190465

/-- The area of a rectangle formed by adding squares in an arithmetic sequence -/
def rectangleArea (squareSize : ℕ) (stages : ℕ) : ℕ :=
  stages * (squareSize * squareSize)

/-- Theorem: The area of a rectangle formed by adding 4" by 4" squares
    in an arithmetic sequence for 8 stages is 128 square inches -/
theorem area_at_stage_8 : rectangleArea 4 8 = 128 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l1904_190465


namespace NUMINAMATH_CALUDE_zeros_of_f_l1904_190418

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 16*x

-- State the theorem
theorem zeros_of_f :
  {x : ℝ | f x = 0} = {-4, 0, 4} := by sorry

end NUMINAMATH_CALUDE_zeros_of_f_l1904_190418


namespace NUMINAMATH_CALUDE_total_sugar_amount_l1904_190467

/-- The total amount of sugar the owner started with, given the number of packs,
    weight per pack, and leftover sugar. -/
theorem total_sugar_amount
  (num_packs : ℕ)
  (weight_per_pack : ℕ)
  (leftover_sugar : ℕ)
  (h1 : num_packs = 12)
  (h2 : weight_per_pack = 250)
  (h3 : leftover_sugar = 20) :
  num_packs * weight_per_pack + leftover_sugar = 3020 :=
by sorry

end NUMINAMATH_CALUDE_total_sugar_amount_l1904_190467


namespace NUMINAMATH_CALUDE_negation_equivalence_l1904_190450

theorem negation_equivalence :
  ¬(∀ (x : ℝ), ∃ (n : ℕ+), (n : ℝ) ≥ x) ↔ 
  ∃ (x : ℝ), ∀ (n : ℕ+), (n : ℝ) < x^2 :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1904_190450


namespace NUMINAMATH_CALUDE_four_common_tangents_min_area_PAOB_l1904_190454

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the moving circle C
def circle_C (x y k : ℝ) : Prop := (x - k)^2 + (y - Real.sqrt 3 * k)^2 = 4

-- Define the line
def line (x y : ℝ) : Prop := x + y = 4

-- Theorem 1: Four common tangents condition
theorem four_common_tangents (k : ℝ) :
  (∀ x y, circle_O x y → ∀ x' y', circle_C x' y' k → 
    ∃! t1 t2 t3 t4 : ℝ × ℝ, 
      (circle_O t1.1 t1.2 ∧ circle_C t1.1 t1.2 k) ∧
      (circle_O t2.1 t2.2 ∧ circle_C t2.1 t2.2 k) ∧
      (circle_O t3.1 t3.2 ∧ circle_C t3.1 t3.2 k) ∧
      (circle_O t4.1 t4.2 ∧ circle_C t4.1 t4.2 k)) ↔
  abs k > 2 := by sorry

-- Theorem 2: Minimum area of quadrilateral PAOB
theorem min_area_PAOB :
  ∃ min_area : ℝ, 
    min_area = 4 ∧
    ∀ P A B O : ℝ × ℝ,
      line P.1 P.2 →
      circle_O A.1 A.2 →
      circle_O B.1 B.2 →
      O = (0, 0) →
      (∀ x y, (x - P.1) * (A.1 - P.1) + (y - P.2) * (A.2 - P.2) = 0 → ¬ circle_O x y) →
      (∀ x y, (x - P.1) * (B.1 - P.1) + (y - P.2) * (B.2 - P.2) = 0 → ¬ circle_O x y) →
      let area := abs ((A.1 - O.1) * (B.2 - O.2) - (B.1 - O.1) * (A.2 - O.2))
      area ≥ min_area := by sorry

end NUMINAMATH_CALUDE_four_common_tangents_min_area_PAOB_l1904_190454


namespace NUMINAMATH_CALUDE_difference_of_squares_l1904_190442

theorem difference_of_squares (x : ℝ) : x^2 - 25 = (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1904_190442


namespace NUMINAMATH_CALUDE_smallest_multiple_l1904_190474

theorem smallest_multiple (n : ℕ) : n = 1767 ↔ 
  n > 0 ∧ 
  31 ∣ n ∧ 
  n % 97 = 3 ∧ 
  ∀ m : ℕ, m > 0 → 31 ∣ m → m % 97 = 3 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1904_190474


namespace NUMINAMATH_CALUDE_hyperbola_condition_roots_or_hyperbola_condition_l1904_190412

-- Define the conditions
def has_two_distinct_positive_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
  x₁^2 + 2*m*x₁ + (m+2) = 0 ∧ x₂^2 + 2*m*x₂ + (m+2) = 0

def is_hyperbola_with_foci_on_y_axis (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2/(m+3) - y^2/(2*m-1) = 1 → 
  (m+3 < 0 ∧ 2*m-1 > 0)

-- Theorem statements
theorem hyperbola_condition (m : ℝ) : 
  is_hyperbola_with_foci_on_y_axis m → m < -3 :=
sorry

theorem roots_or_hyperbola_condition (m : ℝ) :
  (has_two_distinct_positive_roots m ∨ is_hyperbola_with_foci_on_y_axis m) ∧
  ¬(has_two_distinct_positive_roots m ∧ is_hyperbola_with_foci_on_y_axis m) →
  (-2 < m ∧ m < -1) ∨ m < -3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_roots_or_hyperbola_condition_l1904_190412


namespace NUMINAMATH_CALUDE_minimal_discs_to_separate_l1904_190430

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a disc in a plane -/
structure Disc where
  center : Point
  radius : ℝ

/-- A function that checks if a disc separates two points -/
def separates (d : Disc) (p1 p2 : Point) : Prop :=
  (((p1.x - d.center.x)^2 + (p1.y - d.center.y)^2 < d.radius^2) ∧
   ((p2.x - d.center.x)^2 + (p2.y - d.center.y)^2 > d.radius^2)) ∨
  (((p1.x - d.center.x)^2 + (p1.y - d.center.y)^2 > d.radius^2) ∧
   ((p2.x - d.center.x)^2 + (p2.y - d.center.y)^2 < d.radius^2))

/-- The main theorem stating the minimal number of discs needed -/
theorem minimal_discs_to_separate (points : Finset Point) 
  (h : points.card = 2019) :
  ∃ (discs : Finset Disc), discs.card = 1010 ∧
    ∀ p1 p2 : Point, p1 ∈ points → p2 ∈ points → p1 ≠ p2 →
      ∃ d ∈ discs, separates d p1 p2 :=
sorry

end NUMINAMATH_CALUDE_minimal_discs_to_separate_l1904_190430


namespace NUMINAMATH_CALUDE_flower_shop_utilities_percentage_l1904_190458

/-- Calculates the percentage of rent paid for utilities in James' flower shop --/
theorem flower_shop_utilities_percentage
  (weekly_rent : ℝ)
  (store_hours_per_day : ℝ)
  (store_days_per_week : ℝ)
  (employees_per_shift : ℝ)
  (employee_hourly_wage : ℝ)
  (total_weekly_expenses : ℝ)
  (h1 : weekly_rent = 1200)
  (h2 : store_hours_per_day = 16)
  (h3 : store_days_per_week = 5)
  (h4 : employees_per_shift = 2)
  (h5 : employee_hourly_wage = 12.5)
  (h6 : total_weekly_expenses = 3440)
  : (((total_weekly_expenses - (store_hours_per_day * store_days_per_week * employees_per_shift * employee_hourly_wage)) - weekly_rent) / weekly_rent) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_utilities_percentage_l1904_190458


namespace NUMINAMATH_CALUDE_function_values_l1904_190433

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 2 * x + 1
def g (x : ℝ) : ℝ := 3 * x^2 + 1

-- State the theorem
theorem function_values :
  f 2 = 9 ∧ f (-2) = 25 ∧ g (-1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_values_l1904_190433


namespace NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l1904_190451

theorem sin_sum_inverse_sin_tan (x y : ℝ) 
  (hx : x = 4 / 5) (hy : y = 1 / 2) : 
  Real.sin (Real.arcsin x + Real.arctan y) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l1904_190451


namespace NUMINAMATH_CALUDE_two_elements_condition_at_most_one_element_condition_l1904_190497

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x - 4 = 0}

-- Theorem for the first part of the problem
theorem two_elements_condition (a : ℝ) :
  (∃ x y : ℝ, x ∈ A a ∧ y ∈ A a ∧ x ≠ y) ↔ (a > -9/16 ∧ a ≠ 0) :=
sorry

-- Theorem for the second part of the problem
theorem at_most_one_element_condition (a : ℝ) :
  (∀ x y : ℝ, x ∈ A a → y ∈ A a → x = y) ↔ (a ≤ -9/16 ∨ a = 0) :=
sorry

end NUMINAMATH_CALUDE_two_elements_condition_at_most_one_element_condition_l1904_190497


namespace NUMINAMATH_CALUDE_events_A_B_complementary_l1904_190448

-- Define the sample space for a die throw
def DieOutcome := Fin 6

-- Define event A
def eventA (outcome : DieOutcome) : Prop :=
  outcome.val ≤ 2

-- Define event B
def eventB (outcome : DieOutcome) : Prop :=
  outcome.val ≥ 3

-- Define event C (not used in the theorem, but included for completeness)
def eventC (outcome : DieOutcome) : Prop :=
  outcome.val % 2 = 1

-- Theorem stating that events A and B are complementary
theorem events_A_B_complementary :
  ∀ (outcome : DieOutcome), eventA outcome ↔ ¬ eventB outcome :=
by
  sorry


end NUMINAMATH_CALUDE_events_A_B_complementary_l1904_190448


namespace NUMINAMATH_CALUDE_alice_speed_l1904_190414

theorem alice_speed (total_distance : ℝ) (abel_speed : ℝ) (time_difference : ℝ) (alice_delay : ℝ) :
  total_distance = 1000 →
  abel_speed = 50 →
  time_difference = 6 →
  alice_delay = 1 →
  (total_distance / abel_speed + alice_delay) - (total_distance / abel_speed) = time_difference →
  total_distance / ((total_distance / abel_speed) + time_difference) = 200 / 3 := by
sorry

end NUMINAMATH_CALUDE_alice_speed_l1904_190414


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l1904_190407

theorem perpendicular_lines_slope (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧ 
  (a * (a + 2) = -1) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l1904_190407


namespace NUMINAMATH_CALUDE_parallel_lines_angle_theorem_l1904_190422

/-- Given a configuration of two parallel lines intersected by two other lines,
    if one angle is 70°, its adjacent angle is 40°, and the corresponding angle
    on the other parallel line is 110°, then the remaining angle is 40°. -/
theorem parallel_lines_angle_theorem (a b c d : Real) :
  a = 70 →
  b = 40 →
  c = 110 →
  a + b + c + d = 360 →
  d = 40 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_theorem_l1904_190422


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1904_190463

-- Define the arithmetic sequence
def a (n : ℕ) : ℚ := n + 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := a (n + 1) / a n + a n / a (n + 1) - 2

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := n / (n + 1)

theorem arithmetic_geometric_sequence :
  (a 4 - a 2 = 2) ∧ 
  (a 3 * a 3 = a 1 * a 7) →
  (∀ n : ℕ, a n = n + 1) ∧
  (∀ n : ℕ, S n = n / (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1904_190463


namespace NUMINAMATH_CALUDE_number_puzzle_l1904_190490

theorem number_puzzle : ∃ x : ℝ, 13 * x = x + 180 ∧ x = 15 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l1904_190490


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1904_190462

theorem inequality_solution_set :
  let S := {x : ℝ | (x + 1/2) * (3/2 - x) ≥ 0}
  S = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1904_190462


namespace NUMINAMATH_CALUDE_kaleb_book_count_l1904_190482

theorem kaleb_book_count (initial_books sold_books new_books : ℕ) :
  initial_books = 34 →
  sold_books = 17 →
  new_books = 7 →
  initial_books - sold_books + new_books = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_kaleb_book_count_l1904_190482


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1904_190417

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence)
    (h1 : seq.a 4 + seq.S 5 = 2)
    (h2 : seq.S 7 = 14) :
  seq.a 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1904_190417


namespace NUMINAMATH_CALUDE_point_on_bisector_value_l1904_190453

/-- A point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The bisector of the angle between the two coordinate axes in the first and third quadrants -/
def isOnBisector (p : Point) : Prop :=
  p.x = p.y

/-- The theorem statement -/
theorem point_on_bisector_value (a : ℝ) :
  let A : Point := ⟨a, 2*a + 3⟩
  isOnBisector A → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_bisector_value_l1904_190453


namespace NUMINAMATH_CALUDE_paving_stone_width_l1904_190436

/-- Given a rectangular courtyard and paving stones with specified dimensions,
    prove that the width of each paving stone is 2 meters. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (num_stones : ℕ)
  (h1 : courtyard_length = 70)
  (h2 : courtyard_width = 16.5)
  (h3 : stone_length = 2.5)
  (h4 : num_stones = 231)
  : ∃ (stone_width : ℝ),
    stone_width = 2 ∧
    courtyard_length * courtyard_width = (stone_length * stone_width) * num_stones :=
by sorry

end NUMINAMATH_CALUDE_paving_stone_width_l1904_190436


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_no_real_roots_equation_four_solutions_l1904_190473

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  (x + 1)^2 - 5 = 0 ↔ x = Real.sqrt 5 - 1 ∨ x = -Real.sqrt 5 - 1 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  2 * x^2 - 4 * x + 1 = 0 ↔ x = (4 + Real.sqrt 8) / 4 ∨ x = (4 - Real.sqrt 8) / 4 := by sorry

-- Equation 3
theorem equation_three_no_real_roots :
  ¬∃ (x : ℝ), (2 * x + 1) * (x - 3) = -7 := by sorry

-- Equation 4
theorem equation_four_solutions (x : ℝ) :
  3 * (x - 2)^2 = x * (x - 2) ↔ x = 2 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_no_real_roots_equation_four_solutions_l1904_190473


namespace NUMINAMATH_CALUDE_total_price_of_hats_l1904_190470

theorem total_price_of_hats (total_hats : ℕ) (green_hats : ℕ) (blue_cost : ℕ) (green_cost : ℕ) :
  total_hats = 85 →
  green_hats = 40 →
  blue_cost = 6 →
  green_cost = 7 →
  (total_hats - green_hats) * blue_cost + green_hats * green_cost = 550 :=
by
  sorry

end NUMINAMATH_CALUDE_total_price_of_hats_l1904_190470


namespace NUMINAMATH_CALUDE_new_athlete_rate_is_15_l1904_190483

/-- The rate at which new athletes arrived at the Ultimate Fitness Camp --/
def new_athlete_rate (initial_athletes : ℕ) (leaving_rate : ℕ) (leaving_hours : ℕ) 
  (arrival_hours : ℕ) (total_difference : ℕ) : ℕ :=
  let athletes_left := leaving_rate * leaving_hours
  let remaining_athletes := initial_athletes - athletes_left
  let final_athletes := initial_athletes - total_difference
  let new_athletes := final_athletes - remaining_athletes
  new_athletes / arrival_hours

/-- Theorem stating the rate at which new athletes arrived --/
theorem new_athlete_rate_is_15 : 
  new_athlete_rate 300 28 4 7 7 = 15 := by sorry

end NUMINAMATH_CALUDE_new_athlete_rate_is_15_l1904_190483


namespace NUMINAMATH_CALUDE_circle_op_twelve_seven_l1904_190491

def circle_op (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem circle_op_twelve_seven :
  circle_op 12 7 = 95 := by sorry

end NUMINAMATH_CALUDE_circle_op_twelve_seven_l1904_190491


namespace NUMINAMATH_CALUDE_unique_solution_modular_equation_l1904_190485

theorem unique_solution_modular_equation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 103 ∧ (99 * n) % 103 = 72 % 103 ∧ n = 52 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_modular_equation_l1904_190485


namespace NUMINAMATH_CALUDE_earth_moon_distance_in_scientific_notation_l1904_190498

/-- The distance from Earth to the Moon's surface in kilometers -/
def earth_moon_distance : ℝ := 383900

/-- The scientific notation representation of the Earth-Moon distance -/
def earth_moon_distance_scientific : ℝ := 3.839 * (10 ^ 5)

theorem earth_moon_distance_in_scientific_notation :
  earth_moon_distance = earth_moon_distance_scientific :=
sorry

end NUMINAMATH_CALUDE_earth_moon_distance_in_scientific_notation_l1904_190498


namespace NUMINAMATH_CALUDE_regular_tetrahedron_inscribed_sphere_ratio_l1904_190447

/-- For a regular tetrahedron with height H and inscribed sphere radius R, 
    the ratio R:H is 1:4 -/
theorem regular_tetrahedron_inscribed_sphere_ratio 
  (H : ℝ) (R : ℝ) (h : H > 0) (r : R > 0) : R / H = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_inscribed_sphere_ratio_l1904_190447


namespace NUMINAMATH_CALUDE_total_rulers_problem_solution_l1904_190440

/-- Given an initial number of rulers and a number of rulers added, 
    the total number of rulers is equal to the sum of these two numbers. -/
theorem total_rulers (initial_rulers added_rulers : ℕ) :
  initial_rulers + added_rulers = 
    initial_rulers + added_rulers := by sorry

/-- The specific case for the problem -/
theorem problem_solution : 
  11 + 14 = 25 := by sorry

end NUMINAMATH_CALUDE_total_rulers_problem_solution_l1904_190440


namespace NUMINAMATH_CALUDE_max_square_side_length_is_40_l1904_190461

def distances_L : List ℕ := [2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6, 2]
def distances_P : List ℕ := [3, 1, 2, 6, 3, 1, 2, 6, 3, 1, 2, 6, 3, 1]

def max_square_side_length : ℕ := 40

theorem max_square_side_length_is_40 :
  ∃ (start_L end_L start_P end_P : ℕ),
    start_L < end_L ∧
    start_P < end_P ∧
    end_L - start_L = max_square_side_length ∧
    end_P - start_P = max_square_side_length ∧
    (∀ (s e : ℕ), s < e →
      (e - s > max_square_side_length →
        ¬(∃ (i j : ℕ), i + 1 = j ∧ 
          (List.sum (List.take j distances_L) - List.sum (List.take i distances_L) = e - s) ∨
          (List.sum (List.take j distances_P) - List.sum (List.take i distances_P) = e - s)))) :=
by sorry

end NUMINAMATH_CALUDE_max_square_side_length_is_40_l1904_190461


namespace NUMINAMATH_CALUDE_power_function_through_point_l1904_190405

theorem power_function_through_point (a : ℝ) : 
  (∀ x : ℝ, (fun x => x^a) x = x^a) → 
  (2 : ℝ)^a = 16 → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1904_190405


namespace NUMINAMATH_CALUDE_trajectory_line_passes_fixed_point_l1904_190476

/-- The trajectory C is defined by the equation y^2 = 4x -/
def trajectory (x y : ℝ) : Prop := y^2 = 4*x

/-- A point P is on the trajectory if it satisfies the equation -/
def on_trajectory (P : ℝ × ℝ) : Prop :=
  trajectory P.1 P.2

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- A line passing through two points -/
def line_through (A B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

theorem trajectory_line_passes_fixed_point :
  ∀ A B : ℝ × ℝ,
  A ≠ (0, 0) → B ≠ (0, 0) → A ≠ B →
  on_trajectory A → on_trajectory B →
  dot_product A B = 0 →
  line_through A B (4, 0) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_line_passes_fixed_point_l1904_190476


namespace NUMINAMATH_CALUDE_bridge_renovation_problem_l1904_190404

theorem bridge_renovation_problem (bridge_length : ℝ) (efficiency_increase : ℝ) (days_ahead : ℝ) 
  (h1 : bridge_length = 36)
  (h2 : efficiency_increase = 0.5)
  (h3 : days_ahead = 2) :
  ∃ x : ℝ, x = 6 ∧ 
    bridge_length / x = bridge_length / ((1 + efficiency_increase) * x) + days_ahead :=
by sorry

end NUMINAMATH_CALUDE_bridge_renovation_problem_l1904_190404


namespace NUMINAMATH_CALUDE_prob_at_least_three_same_value_l1904_190459

def num_dice : ℕ := 5
def num_sides : ℕ := 8

def prob_at_least_three_same : ℚ :=
  (num_dice.choose 3) * (1 / num_sides^2) * ((num_sides - 1) / num_sides)^2 +
  (num_dice.choose 4) * (1 / num_sides^3) * ((num_sides - 1) / num_sides) +
  (1 / num_sides^4)

theorem prob_at_least_three_same_value :
  prob_at_least_three_same = 526 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_same_value_l1904_190459


namespace NUMINAMATH_CALUDE_quartet_performances_theorem_l1904_190456

/-- Represents the number of performances for each friend -/
structure Performances where
  sarah : ℕ
  lily : ℕ
  emma : ℕ
  nora : ℕ
  kate : ℕ

/-- The total number of quartet performances -/
def total_performances (p : Performances) : ℕ :=
  (p.sarah + p.lily + p.emma + p.nora + p.kate) / 4

theorem quartet_performances_theorem (p : Performances) :
  p.nora = 10 →
  p.sarah = 6 →
  p.lily > 6 →
  p.emma > 6 →
  p.kate > 6 →
  p.lily < 10 →
  p.emma < 10 →
  p.kate < 10 →
  (p.sarah + p.lily + p.emma + p.nora + p.kate) % 4 = 0 →
  total_performances p = 10 := by
  sorry

#check quartet_performances_theorem

end NUMINAMATH_CALUDE_quartet_performances_theorem_l1904_190456


namespace NUMINAMATH_CALUDE_min_color_changes_l1904_190437

/-- Represents a 10x10 board with colored chips -/
def Board := Fin 10 → Fin 10 → Fin 100

/-- Checks if a chip is unique in its row or column -/
def is_unique (b : Board) (i j : Fin 10) : Prop :=
  (∀ k : Fin 10, k ≠ j → b i k ≠ b i j) ∨
  (∀ k : Fin 10, k ≠ i → b k j ≠ b i j)

/-- Represents a valid color change operation -/
def valid_change (b1 b2 : Board) : Prop :=
  ∃ i j : Fin 10, 
    (∀ x y : Fin 10, (x ≠ i ∨ y ≠ j) → b1 x y = b2 x y) ∧
    is_unique b1 i j ∧
    b1 i j ≠ b2 i j

/-- Represents a sequence of valid color changes -/
def valid_sequence (n : ℕ) : Prop :=
  ∃ (seq : Fin (n + 1) → Board),
    (∀ i : Fin 10, ∀ j : Fin 10, seq 0 i j = i.val * 10 + j.val) ∧
    (∀ k : Fin n, valid_change (seq k) (seq (k + 1))) ∧
    (∀ i j : Fin 10, ¬is_unique (seq n) i j)

/-- The main theorem stating the minimum number of color changes -/
theorem min_color_changes : 
  (∃ n : ℕ, valid_sequence n) ∧ 
  (∀ m : ℕ, m < 75 → ¬valid_sequence m) :=
sorry

end NUMINAMATH_CALUDE_min_color_changes_l1904_190437
