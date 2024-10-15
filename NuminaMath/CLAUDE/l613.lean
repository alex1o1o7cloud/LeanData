import Mathlib

namespace NUMINAMATH_CALUDE_weightlifting_ratio_l613_61398

theorem weightlifting_ratio (total weight_first weight_second : ℕ) 
  (h1 : total = weight_first + weight_second)
  (h2 : weight_first = 700)
  (h3 : 2 * weight_first = weight_second + 300)
  (h4 : total = 1800) : 
  weight_first * 11 = weight_second * 7 := by
  sorry

end NUMINAMATH_CALUDE_weightlifting_ratio_l613_61398


namespace NUMINAMATH_CALUDE_area_relation_l613_61373

/-- A square with vertices O, P, Q, R where O is the origin and Q is at (3,3) -/
structure Square :=
  (O : ℝ × ℝ)
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)
  (is_origin : O = (0, 0))
  (is_square : Q = (3, 3))

/-- The area of a square -/
def area_square (s : Square) : ℝ := sorry

/-- The area of a triangle given three points -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating that T(3, -12) makes the area of triangle PQT twice the area of square OPQR -/
theorem area_relation (s : Square) : 
  let T : ℝ × ℝ := (3, -12)
  area_triangle s.P s.Q T = 2 * area_square s := by sorry

end NUMINAMATH_CALUDE_area_relation_l613_61373


namespace NUMINAMATH_CALUDE_dartboard_angle_measure_l613_61396

/-- The measure of the central angle of a region on a circular dartboard, given its probability -/
theorem dartboard_angle_measure (p : ℝ) (h : p = 1 / 8) : 
  p * 360 = 45 := by sorry

end NUMINAMATH_CALUDE_dartboard_angle_measure_l613_61396


namespace NUMINAMATH_CALUDE_no_positive_triples_sum_l613_61338

theorem no_positive_triples_sum : 
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a = b + c ∧ b = c + a ∧ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_no_positive_triples_sum_l613_61338


namespace NUMINAMATH_CALUDE_fraction_sum_l613_61307

theorem fraction_sum (x y : ℚ) (h : x / y = 5 / 2) : (x + y) / y = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l613_61307


namespace NUMINAMATH_CALUDE_problem_solution_l613_61342

noncomputable def f (a k x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- Part 1: k = 2
  (∃ k : ℝ, ∀ x : ℝ, f a k x = -f a k (-x)) ∧
  -- Part 2: f is monotonically decreasing
  (f a 2 1 < 0 → ∀ x y : ℝ, x < y → f a 2 x > f a 2 y) ∧
  -- Part 3: range of t
  (∃ t1 t2 : ℝ, t1 = -3 ∧ t2 = 5 ∧
    ∀ t : ℝ, (∀ x : ℝ, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0) ↔ t1 < t ∧ t < t2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l613_61342


namespace NUMINAMATH_CALUDE_downstream_distance_l613_61343

-- Define the given conditions
def boat_speed : ℝ := 16
def stream_speed : ℝ := 5
def time_downstream : ℝ := 7

-- Define the theorem
theorem downstream_distance :
  let effective_speed := boat_speed + stream_speed
  effective_speed * time_downstream = 147 :=
by sorry

end NUMINAMATH_CALUDE_downstream_distance_l613_61343


namespace NUMINAMATH_CALUDE_imaginary_part_of_2i_plus_1_l613_61349

theorem imaginary_part_of_2i_plus_1 :
  let z : ℂ := 2 * Complex.I * (1 + Complex.I)
  (z.im : ℝ) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2i_plus_1_l613_61349


namespace NUMINAMATH_CALUDE_partition_fifth_power_l613_61332

/-- Number of partitions of a 1 × n rectangle into 1 × 1 squares and broken dominoes -/
def F (n : ℕ) : ℕ :=
  sorry

/-- A broken domino consists of two 1 × 1 squares separated by four squares -/
def is_broken_domino (tile : List (ℕ × ℕ)) : Prop :=
  tile.length = 2 ∧ ∃ i : ℕ, tile = [(i, 1), (i + 5, 1)]

/-- A valid tiling of a 1 × n rectangle -/
def valid_tiling (n : ℕ) (tiling : List (List (ℕ × ℕ))) : Prop :=
  (tiling.join.map Prod.fst).toFinset = Finset.range n ∧
  ∀ tile ∈ tiling, tile.length = 1 ∨ is_broken_domino tile

theorem partition_fifth_power (n : ℕ) :
  (F (5 * n) : ℕ) = (F n) ^ 5 :=
sorry

end NUMINAMATH_CALUDE_partition_fifth_power_l613_61332


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_d_value_l613_61372

theorem quadratic_roots_imply_d_value (d : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 9 * x + d = 0 ↔ x = (-9 + Real.sqrt 17) / 4 ∨ x = (-9 - Real.sqrt 17) / 4) →
  d = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_d_value_l613_61372


namespace NUMINAMATH_CALUDE_simplify_expression_l613_61315

theorem simplify_expression (x : ℝ) :
  Real.sqrt (x^6 + 3*x^4 + 2*x^2) = |x| * Real.sqrt ((x^2 + 1) * (x^2 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l613_61315


namespace NUMINAMATH_CALUDE_joint_order_savings_l613_61391

/-- Represents the cost and discount structure for photocopies -/
structure PhotocopyOrder where
  cost_per_copy : ℚ
  discount_rate : ℚ
  discount_threshold : ℕ

/-- Calculates the total cost of an order with potential discount -/
def total_cost (order : PhotocopyOrder) (num_copies : ℕ) : ℚ :=
  let base_cost := order.cost_per_copy * num_copies
  if num_copies > order.discount_threshold then
    base_cost * (1 - order.discount_rate)
  else
    base_cost

/-- Theorem: Steve and David each save $0.40 by submitting a joint order -/
theorem joint_order_savings (steve_copies david_copies : ℕ) :
  let order := PhotocopyOrder.mk 0.02 0.25 100
  let individual_cost := total_cost order steve_copies
  let joint_copies := steve_copies + david_copies
  let joint_cost := total_cost order joint_copies
  steve_copies = 80 ∧ david_copies = 80 →
  individual_cost - (joint_cost / 2) = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_joint_order_savings_l613_61391


namespace NUMINAMATH_CALUDE_smallest_x_properties_l613_61368

/-- The smallest integer with 18 positive factors that is divisible by both 18 and 24 -/
def smallest_x : ℕ := 288

/-- The number of positive factors of smallest_x -/
def factor_count : ℕ := 18

theorem smallest_x_properties :
  (∃ (factors : Finset ℕ), factors.card = factor_count ∧ 
    ∀ d ∈ factors, d ∣ smallest_x) ∧
  18 ∣ smallest_x ∧
  24 ∣ smallest_x ∧
  ∀ y : ℕ, y < smallest_x →
    ¬(∃ (factors : Finset ℕ), factors.card = factor_count ∧
      ∀ d ∈ factors, d ∣ y ∧ 18 ∣ y ∧ 24 ∣ y) :=
by
  sorry

#eval smallest_x

end NUMINAMATH_CALUDE_smallest_x_properties_l613_61368


namespace NUMINAMATH_CALUDE_inequality_solution_set_l613_61327

theorem inequality_solution_set (x : ℝ) : 5 * x + 1 ≥ 3 * x - 5 ↔ x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l613_61327


namespace NUMINAMATH_CALUDE_volume_of_rotated_specific_cone_l613_61361

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  O : Point3D
  A : Point3D
  B : Point3D

/-- Represents a cone in 3D space -/
structure Cone3D where
  base_center : Point3D
  apex : Point3D
  base_radius : ℝ

/-- Function to create a cone by rotating a triangle around the x-axis -/
def createConeFromTriangle (t : Triangle3D) : Cone3D :=
  { base_center := ⟨t.A.x, 0, 0⟩,
    apex := t.O,
    base_radius := t.B.y - t.A.y }

/-- Function to calculate the volume of a solid obtained by rotating a cone around the y-axis -/
noncomputable def volumeOfRotatedCone (c : Cone3D) : ℝ := sorry

/-- The main theorem to prove -/
theorem volume_of_rotated_specific_cone :
  let t : Triangle3D := { O := ⟨0, 0, 0⟩, A := ⟨1, 0, 0⟩, B := ⟨1, 1, 0⟩ }
  let c : Cone3D := createConeFromTriangle t
  volumeOfRotatedCone c = (8 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_volume_of_rotated_specific_cone_l613_61361


namespace NUMINAMATH_CALUDE_ad_time_theorem_l613_61397

/-- Calculates the total advertisement time in a week given the ad duration and cycle time -/
def total_ad_time_per_week (ad_duration : ℚ) (cycle_time : ℚ) : ℚ :=
  let ads_per_hour : ℚ := 60 / cycle_time
  let ad_time_per_hour : ℚ := ads_per_hour * ad_duration
  let hours_per_week : ℚ := 24 * 7
  ad_time_per_hour * hours_per_week

/-- Converts minutes to hours and minutes -/
def minutes_to_hours_and_minutes (total_minutes : ℚ) : ℚ × ℚ :=
  let hours : ℚ := total_minutes / 60
  let minutes : ℚ := total_minutes % 60
  (hours.floor, minutes)

theorem ad_time_theorem :
  let ad_duration : ℚ := 3/2  -- 1.5 minutes
  let cycle_time : ℚ := 20    -- 20 minutes (including ad duration)
  let total_minutes : ℚ := total_ad_time_per_week ad_duration cycle_time
  let (hours, minutes) := minutes_to_hours_and_minutes total_minutes
  hours = 12 ∧ minutes = 36 := by
  sorry


end NUMINAMATH_CALUDE_ad_time_theorem_l613_61397


namespace NUMINAMATH_CALUDE_sum_13_eq_26_l613_61399

/-- An arithmetic sequence. -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The sum of the first n terms of an arithmetic sequence. -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (List.range n).map seq.a |>.sum

theorem sum_13_eq_26 (seq : ArithmeticSequence) 
    (h : seq.a 3 + seq.a 7 + seq.a 11 = 6) : 
  sum_n seq 13 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_13_eq_26_l613_61399


namespace NUMINAMATH_CALUDE_actual_toddler_count_l613_61306

theorem actual_toddler_count (bill_count : ℕ) (double_counted : ℕ) (missed : ℕ) 
  (h1 : bill_count = 26) 
  (h2 : double_counted = 8) 
  (h3 : missed = 3) : 
  bill_count - double_counted + missed = 21 := by
  sorry

end NUMINAMATH_CALUDE_actual_toddler_count_l613_61306


namespace NUMINAMATH_CALUDE_smallest_number_l613_61357

theorem smallest_number (a b c d : Int) (h1 : a = 2023) (h2 : b = 2022) (h3 : c = -2023) (h4 : d = -2022) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l613_61357


namespace NUMINAMATH_CALUDE_expression_evaluation_l613_61375

theorem expression_evaluation : 8 / 4 - 3^2 + 4 * 2 + Nat.factorial 5 = 121 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l613_61375


namespace NUMINAMATH_CALUDE_license_plate_count_l613_61346

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in the license plate -/
def num_plate_digits : ℕ := 5

/-- The number of letters in the license plate -/
def num_plate_letters : ℕ := 2

/-- The number of possible positions for the letter block (start or end) -/
def num_letter_positions : ℕ := 2

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  num_digits ^ num_plate_digits * 
  num_letters ^ num_plate_letters * 
  num_letter_positions

theorem license_plate_count : total_license_plates = 2704000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l613_61346


namespace NUMINAMATH_CALUDE_exam_score_problem_l613_61324

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 50 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 130 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 36 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l613_61324


namespace NUMINAMATH_CALUDE_probability_diamond_or_ace_l613_61369

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (target_cards : ℕ)
  (h_total : total_cards = 52)
  (h_target : target_cards = 16)

/-- The probability of drawing at least one target card in two draws with replacement -/
def probability_at_least_one (d : Deck) : ℚ :=
  1 - (((d.total_cards - d.target_cards : ℚ) / d.total_cards) ^ 2)

theorem probability_diamond_or_ace (d : Deck) :
  probability_at_least_one d = 88 / 169 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_or_ace_l613_61369


namespace NUMINAMATH_CALUDE_remaining_average_of_prime_numbers_l613_61330

theorem remaining_average_of_prime_numbers 
  (total_count : Nat) 
  (subset_count : Nat) 
  (total_average : ℚ) 
  (subset_average : ℚ) 
  (h1 : total_count = 20) 
  (h2 : subset_count = 10) 
  (h3 : total_average = 95) 
  (h4 : subset_average = 85) : 
  (total_count * total_average - subset_count * subset_average) / (total_count - subset_count) = 105 := by
sorry

end NUMINAMATH_CALUDE_remaining_average_of_prime_numbers_l613_61330


namespace NUMINAMATH_CALUDE_alternating_color_probability_l613_61339

def total_balls : ℕ := 10
def white_balls : ℕ := 5
def black_balls : ℕ := 5

def alternating_sequences : ℕ := 2

def total_arrangements : ℕ := Nat.choose total_balls white_balls

theorem alternating_color_probability :
  (alternating_sequences : ℚ) / total_arrangements = 1 / 126 :=
sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l613_61339


namespace NUMINAMATH_CALUDE_garden_to_land_ratio_l613_61378

/-- A rectangle with width 3/5 of its length -/
structure Rectangle where
  length : ℝ
  width : ℝ
  width_prop : width = (3/5) * length

theorem garden_to_land_ratio (land garden : Rectangle) : 
  (garden.length * garden.width) / (land.length * land.width) = 9/25 := by
  sorry

end NUMINAMATH_CALUDE_garden_to_land_ratio_l613_61378


namespace NUMINAMATH_CALUDE_tank_capacity_l613_61302

/-- Represents a cylindrical tank with a given capacity and current fill level. -/
structure CylindricalTank where
  capacity : ℝ
  fill_percentage : ℝ
  current_volume : ℝ

/-- 
Theorem: Given a cylindrical tank that contains 60 liters of water when it is 40% full, 
the total capacity of the tank when it is completely full is 150 liters.
-/
theorem tank_capacity (tank : CylindricalTank) 
  (h1 : tank.fill_percentage = 0.4)
  (h2 : tank.current_volume = 60) : 
  tank.capacity = 150 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l613_61302


namespace NUMINAMATH_CALUDE_sin_neg_pi_half_l613_61350

theorem sin_neg_pi_half : Real.sin (-π / 2) = -1 := by sorry

end NUMINAMATH_CALUDE_sin_neg_pi_half_l613_61350


namespace NUMINAMATH_CALUDE_square_difference_pattern_l613_61344

theorem square_difference_pattern (n : ℕ) : (2*n + 1)^2 - (2*n - 1)^2 = 8*n := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l613_61344


namespace NUMINAMATH_CALUDE_max_diff_correct_l613_61356

/-- A convex N-gon divided into triangles by non-intersecting diagonals -/
structure ConvexNGon (N : ℕ) where
  triangles : ℕ
  diagonals : ℕ
  triangles_eq : triangles = N - 2
  diagonals_eq : diagonals = N - 3

/-- Coloring of triangles in black and white -/
structure Coloring (N : ℕ) where
  ngon : ConvexNGon N
  white : ℕ
  black : ℕ
  sum_eq : white + black = ngon.triangles
  adjacent_diff : white ≠ black → white > black

/-- Maximum difference between white and black triangles -/
def max_diff (N : ℕ) : ℕ :=
  if N % 3 = 1 then N / 3 - 1 else N / 3

theorem max_diff_correct (N : ℕ) (c : Coloring N) :
  c.white - c.black ≤ max_diff N :=
sorry

end NUMINAMATH_CALUDE_max_diff_correct_l613_61356


namespace NUMINAMATH_CALUDE_exists_special_sequence_l613_61383

/-- A sequence of natural numbers with the property that all natural numbers
    appear exactly once as differences between its members. -/
def special_sequence : Set ℕ → Prop :=
  λ S => (∀ n : ℕ, ∃! (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a > b ∧ a - b = n) ∧
         (∀ a : ℕ, a ∈ S → ∃ b : ℕ, b > a ∧ b ∈ S)

/-- Theorem stating the existence of a special sequence of natural numbers. -/
theorem exists_special_sequence : ∃ S : Set ℕ, special_sequence S :=
  sorry


end NUMINAMATH_CALUDE_exists_special_sequence_l613_61383


namespace NUMINAMATH_CALUDE_tom_reading_pages_l613_61323

theorem tom_reading_pages (total_hours : ℕ) (total_days : ℕ) (pages_per_hour : ℕ) (target_days : ℕ) :
  total_hours = 10 →
  total_days = 5 →
  pages_per_hour = 50 →
  target_days = 7 →
  (total_hours / total_days) * pages_per_hour * target_days = 700 :=
by
  sorry

end NUMINAMATH_CALUDE_tom_reading_pages_l613_61323


namespace NUMINAMATH_CALUDE_population_ratio_theorem_l613_61353

/-- Represents the population ratio in a town --/
structure PopulationRatio where
  men : ℝ
  women : ℝ
  children : ℝ
  elderly : ℝ

/-- The population ratio satisfies the given conditions --/
def satisfiesConditions (p : PopulationRatio) : Prop :=
  p.women = 0.9 * p.men ∧
  p.children = 0.6 * (p.men + p.women) ∧
  p.elderly = 0.25 * (p.women + p.children)

/-- The theorem stating the ratio of men to the combined population of others --/
theorem population_ratio_theorem (p : PopulationRatio) 
  (h : satisfiesConditions p) : 
  p.men / (p.women + p.children + p.elderly) = 1 / 2.55 := by
  sorry

#check population_ratio_theorem

end NUMINAMATH_CALUDE_population_ratio_theorem_l613_61353


namespace NUMINAMATH_CALUDE_student_council_committees_l613_61395

theorem student_council_committees (n : ℕ) : 
  (n.choose 2 = 15) → (n.choose 3 = 20) :=
by sorry

end NUMINAMATH_CALUDE_student_council_committees_l613_61395


namespace NUMINAMATH_CALUDE_average_weight_equation_indeterminate_section_b_size_l613_61311

theorem average_weight_equation (x : ℕ) : (36 * 30) + (x * 30) = (36 + x) * 30 := by
  sorry

theorem indeterminate_section_b_size : 
  ∀ (x : ℕ), (36 * 30) + (x * 30) = (36 + x) * 30 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_equation_indeterminate_section_b_size_l613_61311


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l613_61314

theorem fraction_to_decimal : (59 : ℚ) / (2^2 * 5^7) = (1888 : ℚ) / 10^7 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l613_61314


namespace NUMINAMATH_CALUDE_expression_equals_6500_l613_61354

theorem expression_equals_6500 : (2015 / 1 + 2015 / 0.31) / (1 + 0.31) = 6500 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_6500_l613_61354


namespace NUMINAMATH_CALUDE_function_identification_l613_61371

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define the property of being symmetric about the y-axis
def symmetric_about_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- Define the property of being a translation to the right by 1 unit
def translated_right_by_one (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (x - 1)

-- State the theorem
theorem function_identification
  (f g : ℝ → ℝ)
  (h1 : translated_right_by_one f g)
  (h2 : symmetric_about_y_axis g exp) :
  ∀ x, f x = exp (-x - 1) :=
by sorry

end NUMINAMATH_CALUDE_function_identification_l613_61371


namespace NUMINAMATH_CALUDE_fifteenth_term_of_specific_sequence_l613_61320

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 15th term of the specific arithmetic sequence -/
theorem fifteenth_term_of_specific_sequence (a : ℕ → ℝ) 
    (h_arithmetic : ArithmeticSequence a)
    (h_first : a 1 = 3)
    (h_second : a 2 = 15)
    (h_third : a 3 = 27) :
  a 15 = 171 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_specific_sequence_l613_61320


namespace NUMINAMATH_CALUDE_map_distance_to_actual_distance_l613_61366

/-- Given a map scale and a distance on the map, calculate the actual distance -/
theorem map_distance_to_actual_distance 
  (scale : ℚ) 
  (map_distance : ℚ) 
  (h_scale : scale = 1 / 10000) 
  (h_map_distance : map_distance = 16) : 
  let actual_distance := map_distance / scale
  actual_distance = 1600 := by sorry

end NUMINAMATH_CALUDE_map_distance_to_actual_distance_l613_61366


namespace NUMINAMATH_CALUDE_solution_set_part_i_value_of_a_l613_61385

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Part I
theorem solution_set_part_i (a : ℝ) (h : a = 2) :
  {x : ℝ | f a x ≥ 3 * x + 2} = {x : ℝ | x ≥ 4 ∨ x ≤ 0} :=
sorry

-- Part II
theorem value_of_a (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -1}) → a = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_value_of_a_l613_61385


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l613_61367

theorem product_of_three_numbers (x y z : ℚ) : 
  x + y + z = 190 ∧ 
  8 * x = y - 7 ∧ 
  8 * x = z + 11 ∧
  x ≤ y ∧ 
  x ≤ z →
  x * y * z = (97 * 215 * 161) / 108 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l613_61367


namespace NUMINAMATH_CALUDE_floor_sum_example_l613_61309

theorem floor_sum_example : ⌊(12.7 : ℝ)⌋ + ⌊(-12.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l613_61309


namespace NUMINAMATH_CALUDE_final_selling_price_theorem_l613_61303

/-- The final selling price of a batch of computers -/
def final_selling_price (a : ℝ) : ℝ :=
  a * (1 + 0.2) * (1 - 0.09)

/-- Theorem stating the final selling price calculation -/
theorem final_selling_price_theorem (a : ℝ) :
  final_selling_price a = a * (1 + 0.2) * (1 - 0.09) :=
by sorry

end NUMINAMATH_CALUDE_final_selling_price_theorem_l613_61303


namespace NUMINAMATH_CALUDE_distance_product_l613_61328

theorem distance_product (b₁ b₂ : ℝ) : 
  (∀ b : ℝ, (3*b - 5)^2 + (b - 3)^2 = 39 → b = b₁ ∨ b = b₂) →
  (3*b₁ - 5)^2 + (b₁ - 3)^2 = 39 →
  (3*b₂ - 5)^2 + (b₂ - 3)^2 = 39 →
  b₁ * b₂ = -(9/16) := by
sorry

end NUMINAMATH_CALUDE_distance_product_l613_61328


namespace NUMINAMATH_CALUDE_range_of_f_l613_61358

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -8 ≤ y ∧ y ≤ 8 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l613_61358


namespace NUMINAMATH_CALUDE_intersection_point_theorem_l613_61384

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- A point in 3D space -/
structure Point

/-- The intersection of two planes is a line -/
def plane_intersection (p1 p2 : Plane) : Line :=
  sorry

/-- A point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  sorry

/-- A point lies on a plane -/
def point_on_plane (p : Point) (pl : Plane) : Prop :=
  sorry

theorem intersection_point_theorem 
  (α β γ : Plane) 
  (M : Point) :
  let a := plane_intersection α β
  let b := plane_intersection α γ
  let c := plane_intersection β γ
  (point_on_line M a ∧ point_on_line M b) → 
  point_on_line M c :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_theorem_l613_61384


namespace NUMINAMATH_CALUDE_problem_statement_l613_61364

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the problem statement
theorem problem_statement (A B : Nat) (h1 : A = 2 * B) 
  (h2 : baseToDecimal [2, 2, 4] B + baseToDecimal [5, 5] A = baseToDecimal [1, 3, 4] (A + B)) :
  A + B = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l613_61364


namespace NUMINAMATH_CALUDE_scout_cookies_unpacked_l613_61377

/-- The number of boxes that cannot be fully packed into cases -/
def unpacked_boxes (total_boxes : ℕ) (boxes_per_case : ℕ) : ℕ :=
  total_boxes % boxes_per_case

/-- Proof that 7 boxes cannot be fully packed into cases -/
theorem scout_cookies_unpacked :
  unpacked_boxes 31 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_scout_cookies_unpacked_l613_61377


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l613_61362

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 - 2*x + b > 0 ↔ -3 < x ∧ x < 1) →
  (a = -1 ∧ b = 3 ∧ 
   ∀ x, 3*x^2 - x - 2 ≤ 0 ↔ -2/3 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l613_61362


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l613_61359

theorem absolute_value_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l613_61359


namespace NUMINAMATH_CALUDE_cd_length_in_isosceles_triangles_l613_61345

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.leg + t.base

theorem cd_length_in_isosceles_triangles 
  (abc : IsoscelesTriangle) 
  (cbd : IsoscelesTriangle) 
  (h1 : perimeter cbd = 25)
  (h2 : perimeter abc = 20)
  (h3 : cbd.base = 9) :
  cbd.leg = 8 := by
  sorry

end NUMINAMATH_CALUDE_cd_length_in_isosceles_triangles_l613_61345


namespace NUMINAMATH_CALUDE_dirt_pile_volume_decomposition_l613_61393

/-- Represents the dimensions of a rectangular storage bin -/
structure BinDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the parameters of a dirt pile around the storage bin -/
structure DirtPileParams where
  slantDistance : ℝ

/-- Calculates the volume of the dirt pile around a storage bin -/
def dirtPileVolume (bin : BinDimensions) (pile : DirtPileParams) : ℝ :=
  sorry

theorem dirt_pile_volume_decomposition (bin : BinDimensions) (pile : DirtPileParams) :
  bin.length = 10 ∧ bin.width = 12 ∧ bin.height = 3 ∧ pile.slantDistance = 4 →
  ∃ (m n : ℕ), dirtPileVolume bin pile = m + n * Real.pi ∧ m + n = 280 :=
sorry

end NUMINAMATH_CALUDE_dirt_pile_volume_decomposition_l613_61393


namespace NUMINAMATH_CALUDE_frank_fence_length_l613_61310

/-- Given a rectangular yard with one side of 40 feet and an area of 320 square feet,
    the perimeter minus one side is 56 feet. -/
theorem frank_fence_length :
  ∀ (length width : ℝ),
    length = 40 →
    length * width = 320 →
    2 * width + length = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_frank_fence_length_l613_61310


namespace NUMINAMATH_CALUDE_simple_interest_problem_l613_61304

/-- Represents a date with year, month, and day components. -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Calculates the ending date given the start date and time period. -/
def calculateEndDate (startDate : Date) (timePeriod : Rat) : Date :=
  sorry

/-- Calculates the time period in years given principal, rate, and interest. -/
def calculateTimePeriod (principal : Rat) (rate : Rat) (interest : Rat) : Rat :=
  sorry

theorem simple_interest_problem (principal : Rat) (rate : Rat) (startDate : Date) (interest : Rat) :
  principal = 2000 →
  rate = 25 / (4 * 100) →
  startDate = ⟨2005, 2, 4⟩ →
  interest = 25 →
  let timePeriod := calculateTimePeriod principal rate interest
  let endDate := calculateEndDate startDate timePeriod
  endDate = ⟨2005, 4, 16⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l613_61304


namespace NUMINAMATH_CALUDE_billy_crayons_l613_61351

theorem billy_crayons (jane_crayons : ℝ) (total_crayons : ℕ) 
  (h1 : jane_crayons = 52.0) 
  (h2 : total_crayons = 114) : 
  ↑total_crayons - jane_crayons = 62 := by
  sorry

end NUMINAMATH_CALUDE_billy_crayons_l613_61351


namespace NUMINAMATH_CALUDE_mrs_sheridan_initial_fish_l613_61360

/-- The number of fish Mrs. Sheridan received from her sister -/
def fish_from_sister : ℕ := 47

/-- The total number of fish Mrs. Sheridan has after receiving fish from her sister -/
def total_fish : ℕ := 69

/-- The initial number of fish Mrs. Sheridan had -/
def initial_fish : ℕ := total_fish - fish_from_sister

theorem mrs_sheridan_initial_fish :
  initial_fish = 22 :=
sorry

end NUMINAMATH_CALUDE_mrs_sheridan_initial_fish_l613_61360


namespace NUMINAMATH_CALUDE_sophomore_selection_l613_61321

/-- Represents the number of students selected from a grade -/
structure GradeSelection where
  total : Nat
  selected : Nat

/-- Represents the stratified sampling of students across grades -/
structure StratifiedSampling where
  freshmen : GradeSelection
  sophomores : GradeSelection
  seniors : GradeSelection

/-- 
Given a stratified sampling where:
- There are 210 freshmen, 270 sophomores, and 300 seniors
- 7 freshmen were selected
- The same selection rate is applied across all grades

Prove that 9 sophomores were selected
-/
theorem sophomore_selection (s : StratifiedSampling) 
  (h1 : s.freshmen.total = 210)
  (h2 : s.sophomores.total = 270)
  (h3 : s.seniors.total = 300)
  (h4 : s.freshmen.selected = 7)
  (h5 : s.freshmen.selected * s.sophomores.total = s.sophomores.selected * s.freshmen.total) :
  s.sophomores.selected = 9 := by
  sorry


end NUMINAMATH_CALUDE_sophomore_selection_l613_61321


namespace NUMINAMATH_CALUDE_survey_is_simple_random_sampling_l613_61355

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | ComplexRandom

/-- Represents a population of students --/
structure Population where
  size : Nat
  year : Nat

/-- Represents a sample from a population --/
structure Sample where
  size : Nat
  method : SamplingMethod

/-- Defines the conditions of the survey --/
def survey_conditions (pop : Population) (samp : Sample) : Prop :=
  pop.size = 200 ∧ pop.year = 1 ∧ samp.size = 20

/-- Theorem stating that the sampling method used is Simple Random Sampling --/
theorem survey_is_simple_random_sampling 
  (pop : Population) (samp : Sample) 
  (h : survey_conditions pop samp) : 
  samp.method = SamplingMethod.SimpleRandom := by
  sorry


end NUMINAMATH_CALUDE_survey_is_simple_random_sampling_l613_61355


namespace NUMINAMATH_CALUDE_secretary_project_hours_l613_61388

/-- Proves that given three secretaries whose work times are in the ratio of 2:3:5 and who worked a combined total of 80 hours, the secretary who worked the longest spent 40 hours on the project. -/
theorem secretary_project_hours (t1 t2 t3 : ℝ) : 
  t1 + t2 + t3 = 80 ∧ 
  t2 = (3/2) * t1 ∧ 
  t3 = (5/2) * t1 → 
  t3 = 40 := by
sorry

end NUMINAMATH_CALUDE_secretary_project_hours_l613_61388


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l613_61347

-- Define the cubic polynomial
def cubic_poly (x : ℝ) : ℝ := x^3 - 7*x^2 + 3*x + 4

-- Define the roots of the polynomial
axiom a : ℝ
axiom b : ℝ
axiom c : ℝ

-- State that a, b, c are roots of the polynomial
axiom a_root : cubic_poly a = 0
axiom b_root : cubic_poly b = 0
axiom c_root : cubic_poly c = 0

-- State that a, b, c are distinct
axiom roots_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Theorem to prove
theorem sum_of_reciprocal_squares : 
  1/a^2 + 1/b^2 + 1/c^2 = 65/16 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l613_61347


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_one_l613_61352

def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (a : ℝ) : Set ℝ := {x | x ≤ a}

theorem intersection_nonempty_implies_a_geq_neg_one (a : ℝ) :
  (M ∩ N a).Nonempty → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_one_l613_61352


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l613_61325

theorem solution_set_inequalities :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l613_61325


namespace NUMINAMATH_CALUDE_cuboid_edge_sum_l613_61340

/-- The sum of the lengths of the edges of a cuboid -/
def sumOfEdges (width length height : ℝ) : ℝ :=
  4 * (width + length + height)

/-- Theorem: The sum of the lengths of the edges of a cuboid with
    width 10 cm, length 8 cm, and height 5 cm is equal to 92 cm -/
theorem cuboid_edge_sum :
  sumOfEdges 10 8 5 = 92 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_sum_l613_61340


namespace NUMINAMATH_CALUDE_delta_theta_solution_l613_61317

theorem delta_theta_solution :
  ∃ (Δ Θ : ℤ), 4 * 3 = Δ - 5 + Θ ∧ Θ = 14 ∧ Δ = 3 := by
  sorry

end NUMINAMATH_CALUDE_delta_theta_solution_l613_61317


namespace NUMINAMATH_CALUDE_team_point_difference_l613_61300

/-- The difference in points between two teams -/
def point_difference (beth_score jan_score judy_score angel_score : ℕ) : ℕ :=
  (beth_score + jan_score) - (judy_score + angel_score)

/-- Theorem stating the point difference between the two teams -/
theorem team_point_difference :
  point_difference 12 10 8 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_team_point_difference_l613_61300


namespace NUMINAMATH_CALUDE_speedboat_drift_time_l613_61319

/-- The time taken for a speedboat to drift along a river --/
theorem speedboat_drift_time 
  (L : ℝ) -- Total length of the river
  (v : ℝ) -- Speed of the speedboat in still water
  (u : ℝ) -- Speed of the water flow when reservoir is discharging
  (h1 : v = L / 150) -- Speed of boat in still water
  (h2 : v + u = L / 60) -- Speed of boat with water flow
  (h3 : u > 0) -- Water flow is positive
  : (L / 3) / u = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_speedboat_drift_time_l613_61319


namespace NUMINAMATH_CALUDE_negation_equivalence_l613_61308

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - x ≤ 0) ↔ (∀ x : ℝ, x > 0 → x^2 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l613_61308


namespace NUMINAMATH_CALUDE_solution_set_for_a_3_min_value_and_range_l613_61334

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part 1
theorem solution_set_for_a_3 :
  {x : ℝ | f 3 x ≤ 6} = {x : ℝ | 0 ≤ x ∧ x ≤ 3} :=
sorry

-- Part 2
theorem min_value_and_range :
  (∀ x : ℝ, f a x + g x ≥ 3) ↔ a ∈ Set.Ici 2 :=
sorry

#check solution_set_for_a_3
#check min_value_and_range

end NUMINAMATH_CALUDE_solution_set_for_a_3_min_value_and_range_l613_61334


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l613_61313

def U : Finset Nat := {0, 1, 2, 3, 4}
def A : Finset Nat := {0, 1, 3}
def B : Finset Nat := {2, 3}

theorem intersection_complement_equality : A ∩ (U \ B) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l613_61313


namespace NUMINAMATH_CALUDE_geometric_sequence_arithmetic_means_l613_61394

theorem geometric_sequence_arithmetic_means (a b c m n : ℝ) 
  (h1 : b^2 = a*c)  -- geometric sequence condition
  (h2 : m = (a + b) / 2)  -- arithmetic mean of a and b
  (h3 : n = (b + c) / 2)  -- arithmetic mean of b and c
  : a / m + c / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_arithmetic_means_l613_61394


namespace NUMINAMATH_CALUDE_angle_BXY_is_30_degrees_l613_61326

-- Define the points and angles
variable (A B C D X Y E : Point)
variable (angle_AXE angle_CYX angle_BXY : ℝ)

-- Define the parallel lines condition
variable (h1 : Parallel (Line.mk A B) (Line.mk C D))

-- Define the angle relationship
variable (h2 : angle_AXE = 4 * angle_CYX - 90)

-- Define the equality of alternate interior angles
variable (h3 : angle_AXE = angle_CYX)

-- Define the relationship between BXY and AXE due to parallel lines
variable (h4 : angle_BXY = angle_AXE)

-- State the theorem
theorem angle_BXY_is_30_degrees :
  angle_BXY = 30 := by sorry

end NUMINAMATH_CALUDE_angle_BXY_is_30_degrees_l613_61326


namespace NUMINAMATH_CALUDE_nested_expression_evaluation_l613_61363

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_evaluation_l613_61363


namespace NUMINAMATH_CALUDE_marble_selection_ways_l613_61386

def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def remaining_marbles : ℕ := total_marbles - 1
def remaining_to_choose : ℕ := marbles_to_choose - 1

theorem marble_selection_ways :
  (remaining_marbles.choose remaining_to_choose) = 56 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l613_61386


namespace NUMINAMATH_CALUDE_client_ladder_cost_l613_61370

/-- The total cost for a set of ladders given the number of ladders, rungs per ladder, and cost per rung -/
def total_cost (num_ladders : ℕ) (rungs_per_ladder : ℕ) (cost_per_rung : ℕ) : ℕ :=
  num_ladders * rungs_per_ladder * cost_per_rung

/-- The theorem stating the total cost for the client's ladder order -/
theorem client_ladder_cost :
  let cost_per_rung := 2
  let cost_first_set := total_cost 10 50 cost_per_rung
  let cost_second_set := total_cost 20 60 cost_per_rung
  cost_first_set + cost_second_set = 3400 := by sorry

end NUMINAMATH_CALUDE_client_ladder_cost_l613_61370


namespace NUMINAMATH_CALUDE_triangle_inequality_l613_61390

/-- Given that a, b, and c are the side lengths of a triangle, 
    prove that a^2(b+c-a) + b^2(c+a-b) + c^2(a+b-c) ≤ 3abc -/
theorem triangle_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l613_61390


namespace NUMINAMATH_CALUDE_no_solution_factorial_power_l613_61379

theorem no_solution_factorial_power (n k : ℕ) (hn : n > 5) (hk : k > 0) :
  (Nat.factorial (n - 1) + 1 ≠ n ^ k) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_factorial_power_l613_61379


namespace NUMINAMATH_CALUDE_python_eating_theorem_l613_61322

/-- Represents the eating rate of a python in terms of days per alligator --/
structure PythonEatingRate where
  days_per_alligator : ℕ

/-- Calculates the number of alligators a python can eat in a given number of days --/
def alligators_eaten (rate : PythonEatingRate) (days : ℕ) : ℕ :=
  days / rate.days_per_alligator

/-- The total number of alligators eaten by all pythons --/
def total_alligators_eaten (p1 p2 p3 : PythonEatingRate) (days : ℕ) : ℕ :=
  alligators_eaten p1 days + alligators_eaten p2 days + alligators_eaten p3 days

theorem python_eating_theorem (p1 p2 p3 : PythonEatingRate) 
  (h1 : p1.days_per_alligator = 7)  -- P1 eats one alligator per week
  (h2 : p2.days_per_alligator = 5)  -- P2 eats one alligator every 5 days
  (h3 : p3.days_per_alligator = 10) -- P3 eats one alligator every 10 days
  : total_alligators_eaten p1 p2 p3 21 = 9 := by
  sorry

#check python_eating_theorem

end NUMINAMATH_CALUDE_python_eating_theorem_l613_61322


namespace NUMINAMATH_CALUDE_cube_side_area_l613_61312

/-- Given a cube with volume 125 cubic decimeters, 
    prove that the surface area of one side is 2500 square centimeters. -/
theorem cube_side_area (volume : ℝ) (side_length : ℝ) : 
  volume = 125 →
  side_length^3 = volume →
  (side_length * 10)^2 = 2500 := by
sorry

end NUMINAMATH_CALUDE_cube_side_area_l613_61312


namespace NUMINAMATH_CALUDE_externally_tangent_circles_m_value_l613_61301

/-- A circle in the 2D plane defined by its equation coefficients -/
structure Circle where
  a : ℝ -- coefficient of x^2
  b : ℝ -- coefficient of y^2
  c : ℝ -- coefficient of x
  d : ℝ -- coefficient of y
  e : ℝ -- constant term

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let center1 := (- c1.c / (2 * c1.a), - c1.d / (2 * c1.b))
  let center2 := (- c2.c / (2 * c2.a), - c2.d / (2 * c2.b))
  let radius1 := Real.sqrt ((c1.c^2 / (4 * c1.a^2) + c1.d^2 / (4 * c1.b^2) - c1.e / c1.a))
  let radius2 := Real.sqrt ((c2.c^2 / (4 * c2.a^2) + c2.d^2 / (4 * c2.b^2) - c2.e / c2.a))
  let distance := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  distance = radius1 + radius2

/-- The main theorem -/
theorem externally_tangent_circles_m_value :
  ∀ m : ℝ,
  let c1 : Circle := { a := 1, b := 1, c := -2, d := -4, e := m }
  let c2 : Circle := { a := 1, b := 1, c := -8, d := -12, e := 36 }
  are_externally_tangent c1 c2 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_externally_tangent_circles_m_value_l613_61301


namespace NUMINAMATH_CALUDE_cylinder_to_sphere_l613_61382

/-- Given a cylinder with base radius 4 and lateral area 16π/3,
    prove its volume and the radius of an equivalent sphere -/
theorem cylinder_to_sphere (r : ℝ) (L : ℝ) (h : ℝ) (V : ℝ) (R : ℝ) :
  r = 4 →
  L = 16 / 3 * Real.pi →
  L = 2 * Real.pi * r * h →
  V = Real.pi * r^2 * h →
  V = 4 / 3 * Real.pi * R^3 →
  V = 32 / 3 * Real.pi ∧ R = 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_to_sphere_l613_61382


namespace NUMINAMATH_CALUDE_range_of_even_quadratic_function_l613_61387

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the domain of the function
def domain (a : ℝ) : Set ℝ := Set.Icc (1 + a) 2

-- State the theorem
theorem range_of_even_quadratic_function (a b : ℝ) :
  is_even (f a b) ∧ (∀ x ∈ domain a, f a b x ∈ Set.Icc (-10) 2) →
  Set.range (f a b) = Set.Icc (-10) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_even_quadratic_function_l613_61387


namespace NUMINAMATH_CALUDE_prime_representation_l613_61341

theorem prime_representation (N : ℕ) (hN : Nat.Prime N) :
  ∃ (n : ℤ) (p : ℕ), Nat.Prime p ∧ p < 30 ∧ N = 30 * n.natAbs + p :=
sorry

end NUMINAMATH_CALUDE_prime_representation_l613_61341


namespace NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l613_61331

/-- Given two 2D vectors a and b, prove that a + 3b equals the specified result. -/
theorem vector_addition_scalar_multiplication 
  (a b : ℝ × ℝ) 
  (ha : a = (2, 3)) 
  (hb : b = (-1, 5)) : 
  a + 3 • b = (-1, 18) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l613_61331


namespace NUMINAMATH_CALUDE_product_equals_square_l613_61337

theorem product_equals_square : 500 * 3986 * 0.3986 * 20 = (3986 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l613_61337


namespace NUMINAMATH_CALUDE_circle_tangent_to_lines_l613_61336

/-- A circle with center (0, k) is tangent to lines y = x, y = -x, y = 10, and y = -4x. -/
theorem circle_tangent_to_lines (k : ℝ) (h : k > 10) :
  let r := 10 * Real.sqrt 34 * (Real.sqrt 2 / (Real.sqrt 2 - 1 / Real.sqrt 17)) - 10 * Real.sqrt 2
  ∃ (circle : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ circle ↔ (x^2 + (y - k)^2 = r^2)) ∧
    (∃ (x₁ y₁ : ℝ), (x₁, y₁) ∈ circle ∧ y₁ = x₁) ∧
    (∃ (x₂ y₂ : ℝ), (x₂, y₂) ∈ circle ∧ y₂ = -x₂) ∧
    (∃ (x₃ y₃ : ℝ), (x₃, y₃) ∈ circle ∧ y₃ = 10) ∧
    (∃ (x₄ y₄ : ℝ), (x₄, y₄) ∈ circle ∧ y₄ = -4*x₄) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_lines_l613_61336


namespace NUMINAMATH_CALUDE_special_triangle_properties_l613_61333

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a * Real.cos t.C + t.c * Real.cos t.A = 4 * t.b * Real.cos t.B ∧
  t.b = 2 * Real.sqrt 19 ∧
  (1 / 2) * t.a * t.b * Real.sin t.C = 6 * Real.sqrt 15

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  Real.cos t.B = 1 / 4 ∧ t.a + t.b + t.c = 14 + 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l613_61333


namespace NUMINAMATH_CALUDE_part1_part2_l613_61365

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2*|x + 1|

-- Part 1
theorem part1 : 
  {x : ℝ | f 2 x > 4} = {x : ℝ | x < -4/3 ∨ x > 0} := by sorry

-- Part 2
theorem part2 : 
  ({x : ℝ | f a x < 3*x + 4} = {x : ℝ | x > 2}) → a = 6 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l613_61365


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l613_61374

/-- The complex number z = i(-2-i) is located in the third quadrant of the complex plane. -/
theorem complex_number_in_third_quadrant : 
  let z : ℂ := Complex.I * (-2 - Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l613_61374


namespace NUMINAMATH_CALUDE_operation_equality_l613_61381

-- Define a custom type for the allowed operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the operation
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_equality (star mul : Operation) :
  (applyOp star 20 5) / (applyOp mul 15 5) = 1 →
  (applyOp star 8 4) / (applyOp mul 10 2) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_operation_equality_l613_61381


namespace NUMINAMATH_CALUDE_inscribed_circle_inequality_l613_61318

variable (a b c u v w : ℝ)

-- a, b, c are positive real numbers representing side lengths of a triangle
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- u, v, w are positive real numbers representing distances from incenter to opposite vertices
variable (hu : u > 0) (hv : v > 0) (hw : w > 0)

-- Triangle inequality
variable (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b)

theorem inscribed_circle_inequality :
  (a + b + c) * (1/u + 1/v + 1/w) ≤ 3 * (a/u + b/v + c/w) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_inequality_l613_61318


namespace NUMINAMATH_CALUDE_first_day_exceeding_2000_l613_61316

def algae_population (n : ℕ) : ℕ := 5 * 3^n

theorem first_day_exceeding_2000 :
  ∃ n : ℕ, n > 0 ∧ algae_population n > 2000 ∧ ∀ m : ℕ, m < n → algae_population m ≤ 2000 :=
by
  use 7
  sorry

end NUMINAMATH_CALUDE_first_day_exceeding_2000_l613_61316


namespace NUMINAMATH_CALUDE_cornelia_area_is_17_over_6_l613_61335

/-- Represents an equiangular octagon with alternating side lengths -/
structure EquiangularOctagon where
  side1 : ℝ
  side2 : ℝ

/-- Represents a self-intersecting octagon formed by connecting alternate vertices of an equiangular octagon -/
structure SelfIntersectingOctagon where
  base : EquiangularOctagon

/-- The area enclosed by a self-intersecting octagon -/
def enclosed_area (octagon : SelfIntersectingOctagon) : ℝ := sorry

/-- The theorem stating that the area enclosed by CORNELIA is 17/6 -/
theorem cornelia_area_is_17_over_6 (caroline : EquiangularOctagon) 
  (cornelia : SelfIntersectingOctagon) (h1 : caroline.side1 = Real.sqrt 2) 
  (h2 : caroline.side2 = 1) (h3 : cornelia.base = caroline) : 
  enclosed_area cornelia = 17 / 6 := by sorry

end NUMINAMATH_CALUDE_cornelia_area_is_17_over_6_l613_61335


namespace NUMINAMATH_CALUDE_fair_hair_percentage_l613_61329

/-- Given a company where 10% of all employees are women with fair hair,
    and 40% of fair-haired employees are women,
    prove that 25% of all employees have fair hair. -/
theorem fair_hair_percentage
  (total_employees : ℝ)
  (women_fair_hair_percentage : ℝ)
  (women_among_fair_hair_percentage : ℝ)
  (h1 : women_fair_hair_percentage = 0.1)
  (h2 : women_among_fair_hair_percentage = 0.4)
  : (women_fair_hair_percentage / women_among_fair_hair_percentage) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_fair_hair_percentage_l613_61329


namespace NUMINAMATH_CALUDE_password_decryption_probability_l613_61392

theorem password_decryption_probability :
  let p_a : ℝ := 1/5  -- Probability of A's success
  let p_b : ℝ := 1/3  -- Probability of B's success
  let p_c : ℝ := 1/4  -- Probability of C's success
  let p_success : ℝ := 1 - (1 - p_a) * (1 - p_b) * (1 - p_c)  -- Probability of successful decryption
  p_success = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_password_decryption_probability_l613_61392


namespace NUMINAMATH_CALUDE_least_subtraction_l613_61305

theorem least_subtraction (n : ℕ) : ∃! x : ℕ, 
  (∀ d ∈ ({9, 11, 17} : Set ℕ), (3381 - x) % d = 8) ∧ 
  (∀ y : ℕ, y < x → ∃ d ∈ ({9, 11, 17} : Set ℕ), (3381 - y) % d ≠ 8) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_l613_61305


namespace NUMINAMATH_CALUDE_prob_two_boys_from_three_boys_one_girl_l613_61380

/-- The probability of selecting 2 boys from a group of 3 boys and 1 girl is 1/2 -/
theorem prob_two_boys_from_three_boys_one_girl :
  let total_students : ℕ := 4
  let num_boys : ℕ := 3
  let num_girls : ℕ := 1
  let students_to_select : ℕ := 2
  (Nat.choose num_boys students_to_select : ℚ) / (Nat.choose total_students students_to_select) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_prob_two_boys_from_three_boys_one_girl_l613_61380


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l613_61376

/-- A complex number is purely imaginary if its real part is zero. -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

/-- For a, b ∈ ℝ, "a = 0" is a necessary but not sufficient condition 
    for the complex number a + bi to be purely imaginary. -/
theorem a_zero_necessary_not_sufficient (a b : ℝ) :
  (is_purely_imaginary (Complex.mk a b) → a = 0) ∧
  ¬(a = 0 → is_purely_imaginary (Complex.mk a b)) :=
sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l613_61376


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l613_61348

theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h1 : rectangle_width = 20)
  (h2 : rectangle_height = 15)
  (h3 : circle_diameter = 8)
  (h4 : circle_diameter ≤ rectangle_width)
  (h5 : circle_diameter ≤ rectangle_height) :
  let max_horizontal_distance := rectangle_width - circle_diameter
  let max_vertical_distance := rectangle_height - circle_diameter
  Real.sqrt (max_horizontal_distance ^ 2 + max_vertical_distance ^ 2) = Real.sqrt 193 :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l613_61348


namespace NUMINAMATH_CALUDE_initial_strawberries_l613_61389

/-- The number of strawberries Paul picked -/
def picked : ℕ := 78

/-- The total number of strawberries Paul had after picking more -/
def total : ℕ := 120

/-- The initial number of strawberries in Paul's basket -/
def initial : ℕ := total - picked

theorem initial_strawberries : initial + picked = total := by
  sorry

end NUMINAMATH_CALUDE_initial_strawberries_l613_61389
