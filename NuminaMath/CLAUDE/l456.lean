import Mathlib

namespace NUMINAMATH_CALUDE_parabola_equation_l456_45658

/-- Prove that for a parabola y^2 = 2px with p > 0, if there exists a point M(3, y) on the parabola
    such that the distance from M to the focus F(p/2, 0) is 5, then p = 4. -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) : 
  (∃ y : ℝ, y^2 = 2*p*3 ∧ (3 - p/2)^2 + y^2 = 5^2) → p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l456_45658


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l456_45620

theorem geometric_sequence_common_ratio (q : ℝ) : 
  (1 + q + q^2 = 13) ↔ (q = 3 ∨ q = -4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l456_45620


namespace NUMINAMATH_CALUDE_dice_probability_l456_45626

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 7

/-- The total number of possible outcomes when rolling seven dice -/
def total_outcomes : ℕ := num_sides ^ num_dice

/-- The number of ways to get exactly one pair with the other five dice all different -/
def one_pair_outcomes : ℕ := num_sides * (num_dice.choose 2) * (num_sides - 1) * (num_sides - 2) * (num_sides - 3) * (num_sides - 4) * (num_sides - 5)

/-- The number of ways to get exactly two pairs with the other three dice all different -/
def two_pairs_outcomes : ℕ := (num_sides.choose 2) * (num_dice.choose 2) * ((num_dice - 2).choose 2) * (num_sides - 2) * (num_sides - 3) * (num_sides - 4)

/-- The total number of favorable outcomes -/
def favorable_outcomes : ℕ := one_pair_outcomes + two_pairs_outcomes

/-- The probability of getting at least one pair but no three of a kind when rolling seven standard six-sided dice -/
theorem dice_probability : (favorable_outcomes : ℚ) / total_outcomes = 315 / 972 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l456_45626


namespace NUMINAMATH_CALUDE_decimal_insertion_sum_l456_45653

-- Define a function to represent the possible ways to insert a decimal point in 2016
def insert_decimal (n : ℕ) : List ℝ :=
  [2.016, 20.16, 201.6]

-- Define the problem statement
theorem decimal_insertion_sum :
  ∃ (a b c d e f : ℝ),
    (a ∈ insert_decimal 2016) ∧
    (b ∈ insert_decimal 2016) ∧
    (c ∈ insert_decimal 2016) ∧
    (d ∈ insert_decimal 2016) ∧
    (e ∈ insert_decimal 2016) ∧
    (f ∈ insert_decimal 2016) ∧
    (a + b + c + d + e + f = 46368 / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_decimal_insertion_sum_l456_45653


namespace NUMINAMATH_CALUDE_smallest_multiple_of_2019_l456_45667

/-- A number of the form abcabcabc... where a, b, and c are digits -/
def RepeatingDigitNumber (a b c : ℕ) : ℕ := 
  a * 100000000 + b * 10000000 + c * 1000000 +
  a * 100000 + b * 10000 + c * 1000 +
  a * 100 + b * 10 + c

/-- The smallest multiple of 2019 of the form abcabcabc... -/
def SmallestMultiple : ℕ := 673673673

theorem smallest_multiple_of_2019 :
  (∀ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 →
    RepeatingDigitNumber a b c % 2019 = 0 →
    RepeatingDigitNumber a b c ≥ SmallestMultiple) ∧
  SmallestMultiple % 2019 = 0 ∧
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧
    RepeatingDigitNumber a b c = SmallestMultiple :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_2019_l456_45667


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l456_45621

theorem square_sum_zero_implies_both_zero (a b : ℝ) : 
  a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l456_45621


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l456_45688

/-- Given a parabola y^2 = 2px (p > 0) and a point A(t, 0) (t > 0), 
    a line through A intersects the parabola at B and C. 
    Lines OB and OC intersect the line x = -t at M and N respectively. 
    This theorem states that the circle with diameter MN intersects 
    the x-axis at two fixed points. -/
theorem parabola_circle_intersection 
  (p t : ℝ) 
  (hp : p > 0) 
  (ht : t > 0) : 
  ∃ (x₁ x₂ : ℝ), 
    x₁ = -t + Real.sqrt (2 * p * t) ∧ 
    x₂ = -t - Real.sqrt (2 * p * t) ∧ 
    (∀ (x y : ℝ), 
      (x + t)^2 + y^2 = (x₁ + t)^2 → 
      y = 0 → x = x₁ ∨ x = x₂) :=
by sorry


end NUMINAMATH_CALUDE_parabola_circle_intersection_l456_45688


namespace NUMINAMATH_CALUDE_kevin_collected_18_frisbees_l456_45611

/-- The number of frisbees Kevin collected for prizes at the fair. -/
def num_frisbees (total_prizes stuffed_animals yo_yos : ℕ) : ℕ :=
  total_prizes - (stuffed_animals + yo_yos)

/-- Theorem stating that Kevin collected 18 frisbees. -/
theorem kevin_collected_18_frisbees : 
  num_frisbees 50 14 18 = 18 := by
  sorry

end NUMINAMATH_CALUDE_kevin_collected_18_frisbees_l456_45611


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_l456_45676

/-- The total amount Mike spent on car parts -/
def total_spent : ℝ := 224.87

/-- The amount Mike spent on speakers -/
def speakers_cost : ℝ := 118.54

/-- The amount Mike spent on new tires -/
def tires_cost : ℝ := 106.33

/-- Theorem stating that the total amount spent is the sum of speakers and tires costs -/
theorem total_spent_equals_sum : total_spent = speakers_cost + tires_cost := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_l456_45676


namespace NUMINAMATH_CALUDE_fourth_level_open_spots_l456_45622

-- Define the structure of the parking garage
structure ParkingGarage where
  total_levels : Nat
  spots_per_level : Nat
  open_spots_first : Nat
  open_spots_second : Nat
  open_spots_third : Nat
  full_spots_total : Nat

-- Define the problem instance
def parking_problem : ParkingGarage :=
  { total_levels := 4
  , spots_per_level := 100
  , open_spots_first := 58
  , open_spots_second := 60  -- 58 + 2
  , open_spots_third := 65   -- 60 + 5
  , full_spots_total := 186 }

-- Theorem statement
theorem fourth_level_open_spots :
  let p := parking_problem
  let total_spots := p.total_levels * p.spots_per_level
  let open_spots_first_three := p.open_spots_first + p.open_spots_second + p.open_spots_third
  let total_open_spots := total_spots - p.full_spots_total
  total_open_spots - open_spots_first_three = 31 := by
  sorry

end NUMINAMATH_CALUDE_fourth_level_open_spots_l456_45622


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l456_45637

theorem largest_n_satisfying_inequality : 
  (∀ n : ℕ, n^6033 < 2011^2011 → n ≤ 12) ∧ 12^6033 < 2011^2011 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l456_45637


namespace NUMINAMATH_CALUDE_ice_cream_melt_l456_45687

theorem ice_cream_melt (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 3) (h_cylinder : r_cylinder = 10) :
  (4 / 3 * Real.pi * r_sphere ^ 3) / (Real.pi * r_cylinder ^ 2) = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_melt_l456_45687


namespace NUMINAMATH_CALUDE_octal_addition_and_conversion_l456_45661

/-- Represents a number in base 8 --/
def OctalNum (n : ℕ) : Prop := n < 8

/-- Converts a base 8 number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Adds two base 8 numbers --/
def octal_add (a b : ℕ) : ℕ := sorry

theorem octal_addition_and_conversion :
  OctalNum 5 → OctalNum 1 → OctalNum 7 →
  octal_add 5 (8 + 7) = 24 ∧ octal_to_decimal 24 = 20 := by sorry

end NUMINAMATH_CALUDE_octal_addition_and_conversion_l456_45661


namespace NUMINAMATH_CALUDE_inequality_solution_l456_45604

theorem inequality_solution (x : ℝ) : 4 ≤ (2*x)/(3*x-7) ∧ (2*x)/(3*x-7) < 9 ↔ 63/25 < x ∧ x ≤ 2.8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l456_45604


namespace NUMINAMATH_CALUDE_inequality_proof_l456_45643

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1/2) :
  (1 - a^2 + c^2) / (c * (a + 2*b)) + 
  (1 - b^2 + a^2) / (a * (b + 2*c)) + 
  (1 - c^2 + b^2) / (b * (c + 2*a)) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l456_45643


namespace NUMINAMATH_CALUDE_park_area_theorem_l456_45698

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure RectangularPark where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- Calculates the area of a rectangular park -/
def area (park : RectangularPark) : ℝ :=
  park.length * park.width

/-- Calculates the perimeter of a rectangular park -/
def perimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width)

/-- Calculates the fencing cost for a rectangular park -/
def fencingCost (park : RectangularPark) (costPerMeter : ℝ) : ℝ :=
  perimeter park * costPerMeter

theorem park_area_theorem (park : RectangularPark) :
  fencingCost park 0.5 = 155 → area park = 5766 := by
  sorry

end NUMINAMATH_CALUDE_park_area_theorem_l456_45698


namespace NUMINAMATH_CALUDE_colored_segment_existence_l456_45610

/-- A color type with exactly 4 colors -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A point on a line with a color -/
structure ColoredPoint where
  position : ℝ
  color : Color

/-- The theorem statement -/
theorem colored_segment_existence 
  (n : ℕ) 
  (points : Fin n → ColoredPoint) 
  (h_n : n ≥ 4) 
  (h_distinct : ∀ i j, i ≠ j → (points i).position ≠ (points j).position) 
  (h_all_colors : ∀ c : Color, ∃ i, (points i).color = c) :
  ∃ (i j : Fin n), i < j ∧
    (∃ (c₁ c₂ c₃ c₄ : Color), 
      c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₁ ≠ c₄ ∧ c₂ ≠ c₃ ∧ c₂ ≠ c₄ ∧ c₃ ≠ c₄ ∧
      (∃! k, i ≤ k ∧ k ≤ j ∧ (points k).color = c₁) ∧
      (∃! k, i ≤ k ∧ k ≤ j ∧ (points k).color = c₂) ∧
      (∃ k, i < k ∧ k < j ∧ (points k).color = c₃) ∧
      (∃ k, i < k ∧ k < j ∧ (points k).color = c₄)) :=
by
  sorry

end NUMINAMATH_CALUDE_colored_segment_existence_l456_45610


namespace NUMINAMATH_CALUDE_infinite_sets_with_special_divisibility_l456_45672

theorem infinite_sets_with_special_divisibility :
  ∃ f : ℕ → Fin 1983 → ℕ,
    (∀ k : ℕ, ∀ i j : Fin 1983, i < j → f k i < f k j) ∧
    (∀ k : ℕ, ∀ i : Fin 1983, ∃ a : ℕ, a > 1 ∧ (a ^ 1983 ∣ f k i)) ∧
    (∀ k : ℕ, ∀ i : Fin 1983, i.val < 1982 → f k i.succ = f k i + 1) :=
by sorry

end NUMINAMATH_CALUDE_infinite_sets_with_special_divisibility_l456_45672


namespace NUMINAMATH_CALUDE_white_balls_count_l456_45694

theorem white_balls_count (red_balls : ℕ) (white_balls : ℕ) :
  red_balls = 4 →
  (white_balls : ℚ) / (red_balls + white_balls : ℚ) = 2 / 3 →
  white_balls = 8 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l456_45694


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_parallel_planes_imply_line_parallel_l456_45623

-- Define the basic types
variable (P : Type) -- Type for planes
variable (L : Type) -- Type for lines

-- Define the relations
variable (perpendicular : P → L → Prop) -- Plane is perpendicular to a line
variable (parallel_planes : P → P → Prop) -- Two planes are parallel
variable (parallel_line_plane : L → P → Prop) -- A line is parallel to a plane
variable (line_in_plane : L → P → Prop) -- A line is in a plane

-- Theorem 1: If two planes are both perpendicular to the same line, then these two planes are parallel
theorem planes_perpendicular_to_line_are_parallel 
  (p1 p2 : P) (l : L) 
  (h1 : perpendicular p1 l) 
  (h2 : perpendicular p2 l) : 
  parallel_planes p1 p2 :=
sorry

-- Theorem 2: If two planes are parallel to each other, then a line in one of the planes is parallel to the other plane
theorem parallel_planes_imply_line_parallel 
  (p1 p2 : P) (l : L) 
  (h1 : parallel_planes p1 p2) 
  (h2 : line_in_plane l p1) : 
  parallel_line_plane l p2 :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_parallel_planes_imply_line_parallel_l456_45623


namespace NUMINAMATH_CALUDE_fraction_equality_l456_45603

theorem fraction_equality (x y : ℝ) (h : x / y = 1 / 2) :
  (x - y) / (x + y) = -1 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l456_45603


namespace NUMINAMATH_CALUDE_x_y_z_relation_l456_45614

theorem x_y_z_relation (x y z : ℝ) : 
  x = 100.48 → 
  y = 100.48 → 
  x * z = y^2 → 
  z = 1 :=
by sorry

end NUMINAMATH_CALUDE_x_y_z_relation_l456_45614


namespace NUMINAMATH_CALUDE_geometric_sum_problem_l456_45633

theorem geometric_sum_problem : 
  let a : ℚ := 1/2
  let r : ℚ := -1/3
  let n : ℕ := 6
  let S := (a * (1 - r^n)) / (1 - r)
  S = 91/243 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_problem_l456_45633


namespace NUMINAMATH_CALUDE_point_q_midpoint_l456_45645

/-- Given five points on a line, prove that Q is the midpoint of A and B -/
theorem point_q_midpoint (O A B C D Q : ℝ) (l m n p : ℝ) : 
  O < A ∧ A < B ∧ B < C ∧ C < D →  -- Points are in order
  A - O = l →  -- OA = l
  B - O = m →  -- OB = m
  C - O = n →  -- OC = n
  D - O = p →  -- OD = p
  A ≤ Q ∧ Q ≤ B →  -- Q is between A and B
  (C - Q) / (Q - D) = (B - Q) / (Q - A) →  -- CQ : QD = BQ : QA
  Q - O = (l + m) / 2 :=  -- OQ = (l + m) / 2
by sorry

end NUMINAMATH_CALUDE_point_q_midpoint_l456_45645


namespace NUMINAMATH_CALUDE_balloons_in_park_l456_45624

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 1

/-- The total number of balloons brought to the park -/
def total_balloons : ℕ := allan_balloons + jake_balloons

theorem balloons_in_park : total_balloons = 3 := by
  sorry

end NUMINAMATH_CALUDE_balloons_in_park_l456_45624


namespace NUMINAMATH_CALUDE_no_rational_solution_l456_45639

theorem no_rational_solution : ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  1 / (x - y)^2 + 1 / (y - z)^2 + 1 / (z - x)^2 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l456_45639


namespace NUMINAMATH_CALUDE_road_trip_distance_l456_45668

/-- Calculates the total distance of a road trip given specific conditions --/
theorem road_trip_distance 
  (speed : ℝ) 
  (break_duration : ℝ) 
  (time_between_breaks : ℝ) 
  (hotel_search_time : ℝ) 
  (total_trip_time : ℝ) 
  (h1 : speed = 62) 
  (h2 : break_duration = 0.5) 
  (h3 : time_between_breaks = 5) 
  (h4 : hotel_search_time = 0.5) 
  (h5 : total_trip_time = 50) :
  ∃ (distance : ℝ), distance = 2790 := by
  sorry

#check road_trip_distance

end NUMINAMATH_CALUDE_road_trip_distance_l456_45668


namespace NUMINAMATH_CALUDE_equation_with_parentheses_is_true_l456_45678

theorem equation_with_parentheses_is_true : 7 * 9 + 12 / (3 - 2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_equation_with_parentheses_is_true_l456_45678


namespace NUMINAMATH_CALUDE_integer_solution_of_quadratic_equation_l456_45656

theorem integer_solution_of_quadratic_equation (x y : ℤ) :
  x^2 + y^2 = 3*x*y → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_of_quadratic_equation_l456_45656


namespace NUMINAMATH_CALUDE_yearly_dumpling_production_l456_45618

/-- The monthly production of dumplings in kilograms -/
def monthly_production : ℝ := 182.88

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The yearly production of dumplings in kilograms -/
def yearly_production : ℝ := monthly_production * months_in_year

/-- Theorem stating that the yearly production of dumplings is 2194.56 kg -/
theorem yearly_dumpling_production :
  yearly_production = 2194.56 := by
  sorry

end NUMINAMATH_CALUDE_yearly_dumpling_production_l456_45618


namespace NUMINAMATH_CALUDE_integer_solution_abc_l456_45644

theorem integer_solution_abc : 
  ∀ a b c : ℤ, 1 < a ∧ a < b ∧ b < c ∧ ((a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1)) → 
  ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_abc_l456_45644


namespace NUMINAMATH_CALUDE_powder_division_theorem_l456_45691

/-- Represents the measurements and properties of the magical powder division problem. -/
structure PowderDivision where
  total_measured : ℝ
  remaining_measured : ℝ
  removed_measured : ℝ
  error : ℝ

/-- The actual weights of the two portions of the magical powder. -/
def actual_weights (pd : PowderDivision) : ℝ × ℝ :=
  (pd.remaining_measured - pd.error, pd.removed_measured - pd.error)

/-- Theorem stating that given the measurements and assuming a consistent error,
    the actual weights of the two portions are 4 and 3 zolotniks. -/
theorem powder_division_theorem (pd : PowderDivision) 
  (h1 : pd.total_measured = 6)
  (h2 : pd.remaining_measured = 3)
  (h3 : pd.removed_measured = 2)
  (h4 : pd.total_measured = pd.remaining_measured + pd.removed_measured - pd.error) :
  actual_weights pd = (4, 3) := by
  sorry

#eval actual_weights { total_measured := 6, remaining_measured := 3, removed_measured := 2, error := -1 }

end NUMINAMATH_CALUDE_powder_division_theorem_l456_45691


namespace NUMINAMATH_CALUDE_intersection_range_l456_45660

/-- Line l with parameter a -/
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - p.2 - 2 * a = 0}

/-- Circle C -/
def circle_c : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 16}

/-- Theorem stating the range of a when line_l and circle_c intersect -/
theorem intersection_range (a : ℝ) :
  (∃ p, p ∈ line_l a ∩ circle_c) ↔ a ∈ Set.Icc (-2 * Real.sqrt 5) (2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l456_45660


namespace NUMINAMATH_CALUDE_even_function_negative_domain_l456_45640

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_negative_domain
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_pos : ∀ x ≥ 0, f x = x^3 + x) :
  ∀ x < 0, f x = -x^3 - x :=
sorry

end NUMINAMATH_CALUDE_even_function_negative_domain_l456_45640


namespace NUMINAMATH_CALUDE_existence_of_x_and_y_l456_45619

theorem existence_of_x_and_y (f : ℝ → ℝ) : ∃ x y : ℝ, f (x - f y) > y * f x + x := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_and_y_l456_45619


namespace NUMINAMATH_CALUDE_binomial_square_constant_l456_45685

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l456_45685


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l456_45659

/-- The number of ways to arrange the letters of BANANA with indistinguishable A's and N's -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in BANANA -/
def total_letters : ℕ := 6

/-- The number of A's in BANANA -/
def num_a : ℕ := 3

/-- The number of N's in BANANA -/
def num_n : ℕ := 2

/-- The number of B's in BANANA -/
def num_b : ℕ := 1

theorem banana_arrangement_count : 
  banana_arrangements = (Nat.factorial total_letters) / 
    ((Nat.factorial num_a) * (Nat.factorial num_n) * (Nat.factorial num_b)) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l456_45659


namespace NUMINAMATH_CALUDE_product_closest_to_127_l456_45605

def product : ℝ := 2.5 * (50.5 + 0.25)

def options : List ℝ := [120, 125, 127, 130, 140]

theorem product_closest_to_127 :
  ∀ x ∈ options, x ≠ 127 → |product - 127| < |product - x| :=
by sorry

end NUMINAMATH_CALUDE_product_closest_to_127_l456_45605


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_315_l456_45642

/-- The sum of the digits in the binary representation of 315 is 6 -/
theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_315_l456_45642


namespace NUMINAMATH_CALUDE_jackson_deduction_l456_45669

/-- Calculates the deduction in cents given an hourly wage in dollars and a deduction rate. -/
def calculate_deduction (hourly_wage : ℚ) (deduction_rate : ℚ) : ℚ :=
  hourly_wage * 100 * deduction_rate

theorem jackson_deduction :
  let hourly_wage : ℚ := 25
  let deduction_rate : ℚ := 25 / 1000  -- 2.5% expressed as a rational number
  calculate_deduction hourly_wage deduction_rate = 62.5 := by
  sorry

#eval calculate_deduction 25 (25/1000)

end NUMINAMATH_CALUDE_jackson_deduction_l456_45669


namespace NUMINAMATH_CALUDE_triangle_problem_l456_45649

-- Define the triangle
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sin (t.A + t.C) = 8 * (Real.sin (t.B / 2))^2)
  (h2 : t.a + t.c = 6)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = 2) :
  Real.cos t.B = 15/17 ∧ t.b = 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l456_45649


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l456_45699

/-- The probability of drawing two red shoes from a set of 6 red shoes and 4 green shoes is 1/3. -/
theorem probability_two_red_shoes (total_shoes : ℕ) (red_shoes : ℕ) (green_shoes : ℕ) :
  total_shoes = red_shoes + green_shoes →
  red_shoes = 6 →
  green_shoes = 4 →
  (red_shoes : ℚ) / total_shoes * ((red_shoes - 1) : ℚ) / (total_shoes - 1) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l456_45699


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l456_45647

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = 67 / 144 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l456_45647


namespace NUMINAMATH_CALUDE_hen_price_l456_45615

theorem hen_price (total_cost : ℕ) (pig_price : ℕ) (num_pigs : ℕ) (num_hens : ℕ) :
  total_cost = 1200 →
  pig_price = 300 →
  num_pigs = 3 →
  num_hens = 10 →
  (total_cost - num_pigs * pig_price) / num_hens = 30 :=
by sorry

end NUMINAMATH_CALUDE_hen_price_l456_45615


namespace NUMINAMATH_CALUDE_solve_for_b_l456_45606

theorem solve_for_b (a b : ℝ) (eq1 : 2 * a + 1 = 1) (eq2 : b + a = 3) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l456_45606


namespace NUMINAMATH_CALUDE_ashas_initial_savings_l456_45613

def borrowed_money : ℕ := 20 + 40 + 30
def gift_money : ℕ := 70
def remaining_money : ℕ := 65
def spending_fraction : ℚ := 3 / 4

theorem ashas_initial_savings :
  ∃ (initial_savings : ℕ),
    let total_money := initial_savings + borrowed_money + gift_money
    (total_money : ℚ) * (1 - spending_fraction) = remaining_money ∧
    initial_savings = 100 := by
  sorry

end NUMINAMATH_CALUDE_ashas_initial_savings_l456_45613


namespace NUMINAMATH_CALUDE_solve_equation_l456_45627

theorem solve_equation : ∃ x : ℚ, 5 * (x - 6) = 3 * (3 - 3 * x) + 9 ∧ x = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l456_45627


namespace NUMINAMATH_CALUDE_phone_plan_ratio_l456_45675

/-- Given Mandy's phone data plan details, prove the ratio of promotional rate to normal rate -/
theorem phone_plan_ratio : 
  ∀ (normal_rate promotional_rate : ℚ),
  normal_rate = 30 →
  promotional_rate + 2 * normal_rate + (normal_rate + 15) + 2 * normal_rate = 175 →
  promotional_rate / normal_rate = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_phone_plan_ratio_l456_45675


namespace NUMINAMATH_CALUDE_program_arrangements_l456_45673

theorem program_arrangements (n : ℕ) (k : ℕ) : 
  n = 4 → k = 2 → (n + 1) * (n + 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_program_arrangements_l456_45673


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l456_45695

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(abs (x + 2) + abs (x - 1) < a)) → a ∈ Set.Iic 3 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l456_45695


namespace NUMINAMATH_CALUDE_line_passes_through_P_and_intersects_l_l456_45609

-- Define the point P
def P : ℝ × ℝ := (0, 2)

-- Define the line l
def l (x y : ℝ) : Prop := x + 2 * y - 1 = 0

-- Define the line we found
def found_line (x y : ℝ) : Prop := y = x + 2

-- Theorem statement
theorem line_passes_through_P_and_intersects_l :
  -- The line passes through P
  found_line P.1 P.2 ∧
  -- The line is not parallel to l (they intersect)
  ∃ x y, found_line x y ∧ l x y ∧ (x, y) ≠ P :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_P_and_intersects_l_l456_45609


namespace NUMINAMATH_CALUDE_museum_paintings_l456_45651

theorem museum_paintings (removed : ℕ) (remaining : ℕ) (initial : ℕ) : 
  removed = 3 → remaining = 95 → initial = remaining + removed → initial = 98 := by
  sorry

end NUMINAMATH_CALUDE_museum_paintings_l456_45651


namespace NUMINAMATH_CALUDE_unique_base_solution_l456_45663

def base_to_decimal (n : ℕ) (b : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * b^2 + tens * b + ones

theorem unique_base_solution :
  ∃! (c : ℕ), c > 0 ∧ base_to_decimal 243 c + base_to_decimal 156 c = base_to_decimal 421 c :=
by sorry

end NUMINAMATH_CALUDE_unique_base_solution_l456_45663


namespace NUMINAMATH_CALUDE_quadratic_solutions_second_eq_solutions_l456_45686

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 + x - 3 = 0

-- Define the second equation
def second_eq (x : ℝ) : Prop := (2*x + 1)^2 = 3*(2*x + 1)

-- Theorem for the first equation
theorem quadratic_solutions :
  ∃ x1 x2 : ℝ, 
    quadratic_eq x1 ∧ 
    quadratic_eq x2 ∧ 
    x1 = (-1 + Real.sqrt 13) / 2 ∧ 
    x2 = (-1 - Real.sqrt 13) / 2 :=
sorry

-- Theorem for the second equation
theorem second_eq_solutions :
  ∃ x1 x2 : ℝ, 
    second_eq x1 ∧ 
    second_eq x2 ∧ 
    x1 = -1/2 ∧ 
    x2 = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solutions_second_eq_solutions_l456_45686


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l456_45632

-- Define the sets M and P
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def P : Set ℝ := {x : ℝ | x ≤ -1}

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x ∈ M ∩ P → x ∈ M ∪ P) ∧
  ¬(∀ x : ℝ, x ∈ M ∪ P → x ∈ M ∩ P) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l456_45632


namespace NUMINAMATH_CALUDE_product_of_special_ratio_numbers_l456_45648

theorem product_of_special_ratio_numbers :
  ∀ a b : ℝ,
  (a + b) / (a - b) = 5 ∧
  (a * b) / (a - b) = 18 →
  a * b = 54 := by
sorry

end NUMINAMATH_CALUDE_product_of_special_ratio_numbers_l456_45648


namespace NUMINAMATH_CALUDE_stating_school_travel_time_l456_45638

/-- Represents the time in minutes to get from home to school -/
def time_to_school : ℕ := 12

/-- Represents the fraction of the way Kolya walks before realizing he forgot his book -/
def initial_fraction : ℚ := 1/4

/-- Represents the time in minutes Kolya arrives early if he doesn't go back -/
def early_time : ℕ := 5

/-- Represents the time in minutes Kolya arrives late if he goes back -/
def late_time : ℕ := 1

/-- 
Theorem stating that the time to get to school is 12 minutes, given the conditions of the problem.
-/
theorem school_travel_time :
  time_to_school = 12 ∧
  initial_fraction = 1/4 ∧
  early_time = 5 ∧
  late_time = 1 →
  time_to_school = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_stating_school_travel_time_l456_45638


namespace NUMINAMATH_CALUDE_final_state_is_green_l456_45665

/-- Represents the colors of chameleons -/
inductive Color
  | Yellow
  | Red
  | Green

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons -/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 34

/-- Simulates the color change when two different colored chameleons meet -/
def colorChange (state : ChameleonState) : ChameleonState :=
  sorry

/-- Checks if all chameleons are the same color -/
def allSameColor (state : ChameleonState) : Bool :=
  sorry

/-- Theorem: The only possible final state where all chameleons are the same color is green -/
theorem final_state_is_green (state : ChameleonState) :
  (state.yellow + state.red + state.green = totalChameleons) →
  (allSameColor state = true) →
  (state.green = totalChameleons ∧ state.yellow = 0 ∧ state.red = 0) :=
by sorry

end NUMINAMATH_CALUDE_final_state_is_green_l456_45665


namespace NUMINAMATH_CALUDE_sum_reciprocals_bounds_l456_45664

theorem sum_reciprocals_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  (1 / a + 1 / b ≥ 4 / 3) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 3 ∧ 1 / x + 1 / y = 4 / 3) ∧
  (∀ M : ℝ, ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 3 ∧ 1 / x + 1 / y > M) :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bounds_l456_45664


namespace NUMINAMATH_CALUDE_each_angle_less_than_sum_implies_acute_l456_45607

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the property that each angle is less than the sum of the other two
def each_angle_less_than_sum (t : Triangle) : Prop :=
  t.A < t.B + t.C ∧ t.B < t.A + t.C ∧ t.C < t.A + t.B

-- Define an acute triangle
def is_acute_triangle (t : Triangle) : Prop :=
  t.A < 90 ∧ t.B < 90 ∧ t.C < 90

-- Theorem statement
theorem each_angle_less_than_sum_implies_acute (t : Triangle) :
  each_angle_less_than_sum t → is_acute_triangle t :=
by sorry

end NUMINAMATH_CALUDE_each_angle_less_than_sum_implies_acute_l456_45607


namespace NUMINAMATH_CALUDE_concatenated_numbers_divisibility_l456_45617

def concatenate_numbers (n : ℕ) : ℕ :=
  sorry

theorem concatenated_numbers_divisibility (n : ℕ) :
  ¬(3 ∣ concatenate_numbers n) ↔ n % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_concatenated_numbers_divisibility_l456_45617


namespace NUMINAMATH_CALUDE_fraction_inequality_l456_45629

theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) 
  (h5 : e < 0) : 
  e / (a - c)^2 > e / (b - d)^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l456_45629


namespace NUMINAMATH_CALUDE_wire_attachment_point_existence_l456_45692

theorem wire_attachment_point_existence :
  ∃! x : ℝ, 0 < x ∧ x < 5 ∧ Real.sqrt (x^2 + 3.6^2) + Real.sqrt ((x + 5)^2 + 3.6^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_wire_attachment_point_existence_l456_45692


namespace NUMINAMATH_CALUDE_divisors_of_16n4_l456_45690

theorem divisors_of_16n4 (n : ℕ) (h_odd : Odd n) (h_divisors : (Nat.divisors n).card = 13) :
  (Nat.divisors (16 * n^4)).card = 245 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_16n4_l456_45690


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l456_45682

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → 
  a^2004 + b^2005 = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l456_45682


namespace NUMINAMATH_CALUDE_johns_donation_is_100_l456_45697

/-- The size of John's donation to a charity fund --/
def johns_donation (initial_average : ℝ) (num_initial_contributions : ℕ) (new_average : ℝ) : ℝ :=
  (num_initial_contributions + 1) * new_average - num_initial_contributions * initial_average

/-- Theorem stating that John's donation is $100 given the problem conditions --/
theorem johns_donation_is_100 :
  johns_donation 50 1 75 = 100 := by
  sorry

end NUMINAMATH_CALUDE_johns_donation_is_100_l456_45697


namespace NUMINAMATH_CALUDE_volleyball_lineup_combinations_l456_45634

theorem volleyball_lineup_combinations (n : ℕ) (k : ℕ) (h : n = 10 ∧ k = 5) : 
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) = 30240 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_combinations_l456_45634


namespace NUMINAMATH_CALUDE_monotonicity_condition_l456_45657

/-- A function f is monotonically increasing on an interval [a, +∞) if for all x, y in the interval with x < y, f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → f x < f y

/-- The function f(x) = kx^2 + (3k-2)x - 5 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (3*k - 2) * x - 5

theorem monotonicity_condition (k : ℝ) :
  (MonotonicallyIncreasing (f k) 1) ↔ k ≥ 2/5 := by sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l456_45657


namespace NUMINAMATH_CALUDE_midpoint_dot_product_sum_of_squares_l456_45670

/-- Given vectors a and b in ℝ², if m is their midpoint [3, 7] and their dot product is 6,
    then the sum of their squared norms is 220. -/
theorem midpoint_dot_product_sum_of_squares (a b : Fin 2 → ℝ) :
  let m : Fin 2 → ℝ := ![3, 7]
  (∀ i, m i = (a i + b i) / 2) →
  a • b = 6 →
  ‖a‖^2 + ‖b‖^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_dot_product_sum_of_squares_l456_45670


namespace NUMINAMATH_CALUDE_polar_line_through_point_parallel_to_axis_l456_45602

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- The polar equation of a line parallel to the polar axis -/
def isPolarLineParallelToAxis (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ θ, f θ * Real.sin θ = k

theorem polar_line_through_point_parallel_to_axis 
  (P : PolarPoint) 
  (h_P : P.ρ = 2 ∧ P.θ = π/3) :
  isPolarLineParallelToAxis (fun θ ↦ Real.sqrt 3 / Real.sin θ) ∧ 
  (Real.sqrt 3 / Real.sin P.θ) * Real.sin P.θ = P.ρ * Real.sin P.θ :=
sorry

end NUMINAMATH_CALUDE_polar_line_through_point_parallel_to_axis_l456_45602


namespace NUMINAMATH_CALUDE_sine_of_intersection_angle_l456_45635

/-- The sine of the angle formed by a point on y = 3x and x^2 + y^2 = 1 in the first quadrant -/
theorem sine_of_intersection_angle (x y : ℝ) (h1 : y = 3 * x) (h2 : x^2 + y^2 = 1) 
  (h3 : x > 0) (h4 : y > 0) : 
  Real.sin (Real.arctan (y / x)) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_intersection_angle_l456_45635


namespace NUMINAMATH_CALUDE_trigonometric_sum_zero_l456_45681

theorem trigonometric_sum_zero : 
  Real.sin (29/6 * Real.pi) + Real.cos (-29/3 * Real.pi) + Real.tan (-25/4 * Real.pi) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_zero_l456_45681


namespace NUMINAMATH_CALUDE_intersection_M_N_l456_45654

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x < 0}

-- Define set N
def N : Set ℝ := {x | |x| < 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l456_45654


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l456_45631

theorem absolute_value_equals_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l456_45631


namespace NUMINAMATH_CALUDE_joan_music_store_spending_l456_45650

/-- The amount Joan spent at the music store -/
def total_spent (trumpet_cost music_tool_cost song_book_cost : ℚ) : ℚ :=
  trumpet_cost + music_tool_cost + song_book_cost

/-- Proof that Joan spent $163.28 at the music store -/
theorem joan_music_store_spending :
  total_spent 149.16 9.98 4.14 = 163.28 := by
  sorry

end NUMINAMATH_CALUDE_joan_music_store_spending_l456_45650


namespace NUMINAMATH_CALUDE_tangent_line_equation_l456_45616

/-- Given a real number a and a function f(x) = x³ + ax² + (a-3)x with derivative f'(x),
    where f'(x) is an even function, prove that the equation of the tangent line to
    the curve y = f(x) at the point (2, f(2)) is 9x - y - 16 = 0. -/
theorem tangent_line_equation (a : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + (a-3)*x
  let f' : ℝ → ℝ := λ x => 3*x^2 + 2*a*x + (a-3)
  (∀ x, f' x = f' (-x)) → 
  (λ x y => 9*x - y - 16 = 0) = (λ x y => y - f 2 = f' 2 * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l456_45616


namespace NUMINAMATH_CALUDE_binary_101101_equals_base5_140_l456_45693

/-- Converts a binary number to decimal --/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to base 5 --/
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_101101_equals_base5_140 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true, false, true]) = [1, 4, 0] :=
sorry

end NUMINAMATH_CALUDE_binary_101101_equals_base5_140_l456_45693


namespace NUMINAMATH_CALUDE_parabola_ellipse_focus_coincidence_l456_45677

/-- Given a parabola and an ellipse, prove that the parameter m of the parabola
    has a specific value when the focus of the parabola coincides with the left
    focus of the ellipse. -/
theorem parabola_ellipse_focus_coincidence (m : ℝ) : 
  (∀ x y : ℝ, y^2 = (4/m)*x → x^2/7 + y^2/3 = 1) →
  (∃ x y : ℝ, y^2 = (4/m)*x ∧ x^2/7 + y^2/3 = 1 ∧ x = -2) →
  m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_ellipse_focus_coincidence_l456_45677


namespace NUMINAMATH_CALUDE_symmetric_point_polar_axis_l456_45680

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Reflects a polar point about the polar axis -/
def reflectAboutPolarAxis (p : PolarPoint) : PolarPoint :=
  { r := p.r, θ := -p.θ }

theorem symmetric_point_polar_axis (A : PolarPoint) (h : A = { r := 1, θ := π/3 }) :
  reflectAboutPolarAxis A = { r := 1, θ := -π/3 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_polar_axis_l456_45680


namespace NUMINAMATH_CALUDE_function_inequality_l456_45666

open Real

theorem function_inequality (f : ℝ → ℝ) (a : ℝ) (h_cont : Continuous f) (h_pos : a > 0) 
  (h_fa : f a = 1) (h_ineq : ∀ x y, x > 0 → y > 0 → f x * f y + f (a / x) * f (a / y) ≤ 2 * f (x * y)) :
  ∀ x y, x > 0 → y > 0 → f x * f y ≤ f (x * y) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l456_45666


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l456_45684

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧ 
                 a^2 + b^2 = r^2 ∧
                 (a - 7)^2 + (b - 7)^2 = r^2 ∧
                 b = 4/3 * a

-- Define the tangent line l
def tangent_line_l (x y : ℝ) : Prop :=
  (y = -3/4 * x) ∨ (x + y + 5 * Real.sqrt 2 - 7 = 0) ∨ (x + y - 5 * Real.sqrt 2 - 7 = 0)

theorem circle_and_tangent_line :
  (∀ x y, circle_C x y ↔ (x - 3)^2 + (y - 4)^2 = 25) ∧
  (∀ x y, tangent_line_l x y ↔ 
    ((x + y = 0 ∨ x = y) ∧ 
     ∃ (t : ℝ), (x - 3 + t)^2 + (y - 4 + 3/4 * t)^2 = 25 ∧
                ((x + t)^2 + (y + 3/4 * t)^2 > 25 ∨ (x - t)^2 + (y - 3/4 * t)^2 > 25))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l456_45684


namespace NUMINAMATH_CALUDE_kishore_saved_ten_percent_l456_45652

/-- Represents Mr. Kishore's financial situation --/
structure KishoreFinances where
  rent : ℕ
  milk : ℕ
  groceries : ℕ
  education : ℕ
  petrol : ℕ
  miscellaneous : ℕ
  savings : ℕ

/-- Calculates the total expenses --/
def totalExpenses (k : KishoreFinances) : ℕ :=
  k.rent + k.milk + k.groceries + k.education + k.petrol + k.miscellaneous

/-- Calculates the total monthly salary --/
def totalSalary (k : KishoreFinances) : ℕ :=
  totalExpenses k + k.savings

/-- Calculates the percentage saved --/
def percentageSaved (k : KishoreFinances) : ℚ :=
  (k.savings : ℚ) / (totalSalary k : ℚ) * 100

/-- Theorem: Mr. Kishore saved 10% of his monthly salary --/
theorem kishore_saved_ten_percent (k : KishoreFinances)
    (h1 : k.rent = 5000)
    (h2 : k.milk = 1500)
    (h3 : k.groceries = 4500)
    (h4 : k.education = 2500)
    (h5 : k.petrol = 2000)
    (h6 : k.miscellaneous = 3940)
    (h7 : k.savings = 2160) :
    percentageSaved k = 10 := by
  sorry

end NUMINAMATH_CALUDE_kishore_saved_ten_percent_l456_45652


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_l456_45674

-- Equation 1: x^2 - 2x = 0
theorem equation_one_solutions (x : ℝ) : 
  (x = 0 ∨ x = 2) ↔ x^2 - 2*x = 0 := by sorry

-- Equation 2: (2x-1)^2 = (3-x)^2
theorem equation_two_solutions (x : ℝ) : 
  (x = -2 ∨ x = 4/3) ↔ (2*x - 1)^2 = (3 - x)^2 := by sorry

-- Equation 3: 3x(x-2) = x-2
theorem equation_three_solutions (x : ℝ) : 
  (x = 2 ∨ x = 1/3) ↔ 3*x*(x - 2) = x - 2 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_l456_45674


namespace NUMINAMATH_CALUDE_largest_common_term_l456_45600

/-- The largest common term of two arithmetic sequences -/
theorem largest_common_term : ∃ (n m : ℕ), 
  138 = 2 + 4 * n ∧ 
  138 = 5 + 5 * m ∧ 
  138 ≤ 150 ∧ 
  ∀ (k l : ℕ), (2 + 4 * k = 5 + 5 * l) → (2 + 4 * k ≤ 150) → (2 + 4 * k ≤ 138) :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l456_45600


namespace NUMINAMATH_CALUDE_no_rearrangement_sum_999999999_l456_45608

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define a predicate to check if a number is a digit rearrangement of another
def isDigitRearrangement (n m : ℕ) : Prop :=
  sumOfDigits n = sumOfDigits m

theorem no_rearrangement_sum_999999999 (n : ℕ) :
  ¬∃ m : ℕ, isDigitRearrangement n m ∧ m + n = 999999999 :=
sorry

end NUMINAMATH_CALUDE_no_rearrangement_sum_999999999_l456_45608


namespace NUMINAMATH_CALUDE_find_X_l456_45628

theorem find_X : ∃ X : ℚ, (X + 43 / 151) * 151 = 2912 ∧ X = 19 := by
  sorry

end NUMINAMATH_CALUDE_find_X_l456_45628


namespace NUMINAMATH_CALUDE_fraction_sum_values_l456_45662

theorem fraction_sum_values (a b c d : ℝ) 
  (h1 : a / b + b / c + c / d + d / a = 6)
  (h2 : a / c + b / d + c / a + d / b = 8) :
  (a / b + c / d = 2) ∨ (a / b + c / d = 4) := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_values_l456_45662


namespace NUMINAMATH_CALUDE_adiabatic_compression_work_l456_45641

theorem adiabatic_compression_work
  (k : ℝ) (p₁ V₁ V₂ : ℝ) (h_k : k > 1) (h_V : V₂ > 0) :
  let W := (p₁ * V₁) / (k - 1) * (1 - (V₁ / V₂) ^ (k - 1))
  let c := p₁ * V₁^k
  ∀ (p v : ℝ), p * v^k = c →
  W = -(∫ (x : ℝ) in V₁..V₂, c / x^k) :=
sorry

end NUMINAMATH_CALUDE_adiabatic_compression_work_l456_45641


namespace NUMINAMATH_CALUDE_max_ratio_ab_l456_45636

theorem max_ratio_ab (a b : ℕ+) (h : (a : ℚ) / ((a : ℚ) - 2) = ((b : ℚ) + 2021) / ((b : ℚ) + 2008)) :
  (a : ℚ) / (b : ℚ) ≤ 312 / 7 := by
sorry

end NUMINAMATH_CALUDE_max_ratio_ab_l456_45636


namespace NUMINAMATH_CALUDE_trapezoid_total_area_l456_45679

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the total possible area of the trapezoid with different configurations -/
def totalPossibleArea (t : Trapezoid) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem trapezoid_total_area :
  let t := Trapezoid.mk 4 6 8 10
  totalPossibleArea t = 48 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_total_area_l456_45679


namespace NUMINAMATH_CALUDE_difference_of_squares_l456_45630

theorem difference_of_squares : 550^2 - 450^2 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l456_45630


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l456_45625

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0 ∧ (∀ y : ℝ, y^2 + y + m = 0 → y = x)) → m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l456_45625


namespace NUMINAMATH_CALUDE_distance_on_line_l456_45696

/-- The distance between two points on a line --/
theorem distance_on_line (m k a b c d : ℝ) :
  b = m * a + k →
  d = m * c + k →
  Real.sqrt ((a - c)^2 + (b - d)^2) = |a - c| * Real.sqrt (1 + m^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l456_45696


namespace NUMINAMATH_CALUDE_max_games_24_l456_45601

/-- Represents a chess tournament with 8 players -/
structure ChessTournament where
  players : Finset (Fin 8)
  games : Finset (Fin 8 × Fin 8)
  hplayers : players.card = 8
  hgames : ∀ (i j : Fin 8), (i, j) ∈ games → i ≠ j
  hunique : ∀ (i j : Fin 8), (i, j) ∈ games → (j, i) ∉ games

/-- No five players all play each other -/
def noFiveAllPlay (t : ChessTournament) : Prop :=
  ∀ (s : Finset (Fin 8)), s.card = 5 →
    ∃ (i j : Fin 8), i ∈ s ∧ j ∈ s ∧ (i, j) ∉ t.games ∧ (j, i) ∉ t.games

/-- The main theorem: maximum number of games is 24 -/
theorem max_games_24 (t : ChessTournament) (h : noFiveAllPlay t) :
  t.games.card ≤ 24 :=
sorry

end NUMINAMATH_CALUDE_max_games_24_l456_45601


namespace NUMINAMATH_CALUDE_inequality_solution_set_l456_45655

theorem inequality_solution_set (a b c : ℝ) (h1 : a > c) (h2 : b + c > 0) :
  {x : ℝ | (x - c) * (x + b) / (x - a) > 0} = {x : ℝ | -b < x ∧ x < c ∨ x > a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l456_45655


namespace NUMINAMATH_CALUDE_worker_travel_time_l456_45689

theorem worker_travel_time (normal_speed : ℝ) (normal_time : ℝ) 
  (h1 : normal_speed > 0) (h2 : normal_time > 0) : 
  (3/4 * normal_speed) * (normal_time + 8) = normal_speed * normal_time → 
  normal_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_worker_travel_time_l456_45689


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l456_45683

theorem quadratic_root_difference (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + 12 = 0 ∧ 
                x₂^2 + p*x₂ + 12 = 0 ∧ 
                x₁ - x₂ = 1) → 
  p = 7 ∨ p = -7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l456_45683


namespace NUMINAMATH_CALUDE_manager_team_selection_l456_45646

theorem manager_team_selection : Nat.choose 10 6 = 210 := by
  sorry

end NUMINAMATH_CALUDE_manager_team_selection_l456_45646


namespace NUMINAMATH_CALUDE_sqrt_nested_square_l456_45612

theorem sqrt_nested_square : (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nested_square_l456_45612


namespace NUMINAMATH_CALUDE_divisibility_implication_l456_45671

theorem divisibility_implication (x y : ℤ) :
  ∃ k : ℤ, 14 * x + 13 * y = 11 * k → ∃ m : ℤ, 19 * x + 9 * y = 11 * m :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implication_l456_45671
