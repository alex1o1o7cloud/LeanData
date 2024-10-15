import Mathlib

namespace NUMINAMATH_CALUDE_souvenir_october_price_l164_16479

/-- Represents the selling price and sales data of a souvenir --/
structure SouvenirSales where
  september_price : ℝ
  september_revenue : ℝ
  october_discount : ℝ
  october_volume_increase : ℕ
  october_revenue_increase : ℝ

/-- Calculates the October price of a souvenir given its sales data --/
def october_price (s : SouvenirSales) : ℝ :=
  s.september_price * (1 - s.october_discount)

/-- Theorem stating the October price of the souvenir --/
theorem souvenir_october_price (s : SouvenirSales) 
  (h1 : s.september_revenue = 2000)
  (h2 : s.october_discount = 0.1)
  (h3 : s.october_volume_increase = 20)
  (h4 : s.october_revenue_increase = 700) :
  october_price s = 45 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_october_price_l164_16479


namespace NUMINAMATH_CALUDE_average_weight_problem_l164_16413

theorem average_weight_problem (a b c : ℝ) : 
  (a + b) / 2 = 41 →
  (b + c) / 2 = 43 →
  b = 33 →
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l164_16413


namespace NUMINAMATH_CALUDE_south_american_stamps_cost_l164_16483

def brazil_stamp_price : ℚ := 7 / 100
def peru_stamp_price : ℚ := 5 / 100
def brazil_50s_stamps : ℕ := 5
def brazil_60s_stamps : ℕ := 9
def peru_50s_stamps : ℕ := 12
def peru_60s_stamps : ℕ := 8

def total_south_american_stamps_cost : ℚ :=
  (brazil_stamp_price * (brazil_50s_stamps + brazil_60s_stamps)) +
  (peru_stamp_price * (peru_50s_stamps + peru_60s_stamps))

theorem south_american_stamps_cost :
  total_south_american_stamps_cost = 198 / 100 := by
  sorry

end NUMINAMATH_CALUDE_south_american_stamps_cost_l164_16483


namespace NUMINAMATH_CALUDE_range_of_7a_minus_5b_l164_16421

theorem range_of_7a_minus_5b (a b : ℝ) 
  (h1 : 5 ≤ a - b ∧ a - b ≤ 27) 
  (h2 : 6 ≤ a + b ∧ a + b ≤ 30) : 
  ∃ (x : ℝ), 36 ≤ 7*a - 5*b ∧ 7*a - 5*b ≤ 192 ∧
  (∀ (y : ℝ), 36 ≤ y ∧ y ≤ 192 → ∃ (a' b' : ℝ), 
    (5 ≤ a' - b' ∧ a' - b' ≤ 27) ∧ 
    (6 ≤ a' + b' ∧ a' + b' ≤ 30) ∧ 
    y = 7*a' - 5*b') :=
by sorry

end NUMINAMATH_CALUDE_range_of_7a_minus_5b_l164_16421


namespace NUMINAMATH_CALUDE_negation_zero_collinear_with_any_l164_16473

open Set

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def IsCollinear (v w : V) : Prop := ∃ (c : ℝ), v = c • w ∨ w = c • v

theorem negation_zero_collinear_with_any :
  (¬ ∀ (v : V), IsCollinear (0 : V) v) ↔ ∃ (v : V), ¬ IsCollinear (0 : V) v :=
sorry

end NUMINAMATH_CALUDE_negation_zero_collinear_with_any_l164_16473


namespace NUMINAMATH_CALUDE_hexagon_largest_angle_l164_16461

/-- Given a hexagon with internal angles in the ratio 2:3:3:4:5:6, 
    the measure of the largest angle is 4320°/23. -/
theorem hexagon_largest_angle (a b c d e f : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  b / a = 3 / 2 →
  c / a = 3 / 2 →
  d / a = 2 →
  e / a = 5 / 2 →
  f / a = 3 →
  a + b + c + d + e + f = 720 →
  f = 4320 / 23 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_largest_angle_l164_16461


namespace NUMINAMATH_CALUDE_min_m_for_24m_equals_n4_l164_16454

theorem min_m_for_24m_equals_n4 (m n : ℕ+) (h : 24 * m = n^4) :
  ∀ k : ℕ+, 24 * k = (some_nat : ℕ+)^4 → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_min_m_for_24m_equals_n4_l164_16454


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l164_16414

/-- The polynomial function we're working with -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

/-- Theorem stating that 1, -1, and 3 are the only roots of the polynomial -/
theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := by
  sorry

#check roots_of_polynomial

end NUMINAMATH_CALUDE_roots_of_polynomial_l164_16414


namespace NUMINAMATH_CALUDE_banana_purchase_l164_16408

theorem banana_purchase (banana_price apple_price total_weight total_cost : ℚ)
  (h1 : banana_price = 76 / 100)
  (h2 : apple_price = 59 / 100)
  (h3 : total_weight = 30)
  (h4 : total_cost = 1940 / 100) :
  ∃ (banana_weight : ℚ),
    banana_weight + (total_weight - banana_weight) = total_weight ∧
    banana_price * banana_weight + apple_price * (total_weight - banana_weight) = total_cost ∧
    banana_weight = 10 := by
sorry

end NUMINAMATH_CALUDE_banana_purchase_l164_16408


namespace NUMINAMATH_CALUDE_rally_attendance_l164_16409

/-- Represents the rally attendance problem --/
def RallyAttendance (total_receipts : ℚ) (before_rally_tickets : ℕ) 
  (before_rally_price : ℚ) (at_door_price : ℚ) : Prop :=
  ∃ (at_door_tickets : ℕ),
    total_receipts = before_rally_price * before_rally_tickets + at_door_price * at_door_tickets ∧
    before_rally_tickets + at_door_tickets = 750

/-- Theorem stating the total attendance at the rally --/
theorem rally_attendance :
  RallyAttendance (1706.25 : ℚ) 475 2 (2.75 : ℚ) :=
by
  sorry


end NUMINAMATH_CALUDE_rally_attendance_l164_16409


namespace NUMINAMATH_CALUDE_least_n_with_1987_zeros_l164_16402

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The least natural number n such that n! ends in exactly 1987 zeros -/
theorem least_n_with_1987_zeros : ∃ (n : ℕ), trailingZeros n = 1987 ∧ ∀ m < n, trailingZeros m < 1987 :=
  sorry

end NUMINAMATH_CALUDE_least_n_with_1987_zeros_l164_16402


namespace NUMINAMATH_CALUDE_square_transformation_l164_16447

-- Define the square in the xy-plane
def square_vertices : List (ℝ × ℝ) := [(0, 0), (1, 0), (1, 1), (0, 1)]

-- Define the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x^2 - y^2, 2*x*y)

-- Define the transformed square
def transformed_square : List (ℝ × ℝ) := square_vertices.map transform

-- Define the expected shape in the uv-plane
def expected_shape (u v : ℝ) : Prop :=
  (u = 0 ∧ 0 ≤ v ∧ v ≤ 1) ∨  -- Line segment from (0,0) to (1,0)
  (u = 1 - v^2/4 ∧ 0 ≤ v ∧ v ≤ 2) ∨  -- Parabola segment
  (u = v^2/4 - 1 ∧ 0 ≤ v ∧ v ≤ 2) ∨  -- Parabola segment
  (v = 0 ∧ -1 ≤ u ∧ u ≤ 0)  -- Line segment from (-1,0) to (0,0)

theorem square_transformation :
  ∀ (u v : ℝ), (∃ (x y : ℝ), (x, y) ∈ square_vertices ∧ transform (x, y) = (u, v)) ↔ expected_shape u v := by
  sorry

end NUMINAMATH_CALUDE_square_transformation_l164_16447


namespace NUMINAMATH_CALUDE_clothes_spending_fraction_l164_16404

theorem clothes_spending_fraction (initial_amount : ℝ) (fraction_clothes : ℝ) : 
  initial_amount = 249.99999999999994 →
  (3/4 : ℝ) * (4/5 : ℝ) * (1 - fraction_clothes) * initial_amount = 100 →
  fraction_clothes = 11/15 := by
  sorry

end NUMINAMATH_CALUDE_clothes_spending_fraction_l164_16404


namespace NUMINAMATH_CALUDE_fraction_equality_l164_16437

theorem fraction_equality (a b c : ℝ) (hb : b ≠ 0) (hc : c^2 + 1 ≠ 0) :
  (a * (c^2 + 1)) / (b * (c^2 + 1)) = a / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l164_16437


namespace NUMINAMATH_CALUDE_university_subjects_overlap_l164_16465

/-- The problem of students studying Physics and Chemistry at a university --/
theorem university_subjects_overlap (total : ℕ) (physics_min physics_max chem_min chem_max : ℕ) :
  total = 2500 →
  physics_min = 1750 →
  physics_max = 1875 →
  chem_min = 1000 →
  chem_max = 1125 →
  let m := physics_min + chem_min - total
  let M := physics_max + chem_max - total
  M - m = 250 := by
  sorry

end NUMINAMATH_CALUDE_university_subjects_overlap_l164_16465


namespace NUMINAMATH_CALUDE_divisors_of_360_l164_16495

theorem divisors_of_360 : ∃ (d : Finset Nat), 
  (∀ x ∈ d, x ∣ 360) ∧ 
  (∀ x : Nat, x ∣ 360 → x ∈ d) ∧
  d.card = 24 ∧
  d.sum id = 1170 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_360_l164_16495


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_eighteenths_l164_16492

theorem rationalize_sqrt_five_eighteenths : 
  Real.sqrt (5 / 18) = Real.sqrt 10 / 6 := by sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_eighteenths_l164_16492


namespace NUMINAMATH_CALUDE_find_n_l164_16453

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) = 15 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l164_16453


namespace NUMINAMATH_CALUDE_intersection_value_l164_16494

theorem intersection_value (m n : ℝ) (h1 : n = 3 / m) (h2 : n = m + 1) :
  (m - n)^2 * (1 / n - 1 / m) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l164_16494


namespace NUMINAMATH_CALUDE_application_methods_eq_sixteen_l164_16438

/-- The number of universities -/
def total_universities : ℕ := 6

/-- The number of universities to be chosen -/
def universities_to_choose : ℕ := 3

/-- The number of universities with overlapping schedules -/
def overlapping_universities : ℕ := 2

/-- The function to calculate the number of different application methods -/
def application_methods : ℕ := sorry

/-- Theorem stating that the number of different application methods is 16 -/
theorem application_methods_eq_sixteen :
  application_methods = 16 := by sorry

end NUMINAMATH_CALUDE_application_methods_eq_sixteen_l164_16438


namespace NUMINAMATH_CALUDE_zora_shorter_than_brixton_l164_16452

/-- Proves that Zora is 8 inches shorter than Brixton given the conditions of the problem -/
theorem zora_shorter_than_brixton :
  ∀ (zora itzayana zara brixton : ℕ),
    itzayana = zora + 4 →
    zara = 64 →
    brixton = zara →
    (zora + itzayana + zara + brixton) / 4 = 61 →
    brixton - zora = 8 := by
  sorry

end NUMINAMATH_CALUDE_zora_shorter_than_brixton_l164_16452


namespace NUMINAMATH_CALUDE_max_sum_of_squared_distances_l164_16429

variables {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem max_sum_of_squared_distances (a b c d : E) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2 ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squared_distances_l164_16429


namespace NUMINAMATH_CALUDE_hash_toy_difference_l164_16439

theorem hash_toy_difference (bill_toys : ℕ) (total_toys : ℕ) : 
  bill_toys = 60 →
  total_toys = 99 →
  total_toys - bill_toys - bill_toys / 2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_hash_toy_difference_l164_16439


namespace NUMINAMATH_CALUDE_sum_of_combinations_specific_combination_l164_16446

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Statement for the first part
theorem sum_of_combinations : C 5 0 + C 6 5 + C 7 5 + C 8 5 + C 9 5 + C 10 5 = 462 := by sorry

-- Statement for the second part
theorem specific_combination (m : ℕ) :
  (1 / C 5 m : ℚ) - (1 / C 6 m : ℚ) = (7 : ℚ) / (10 * C 7 m) → C 8 m = 28 := by sorry

end NUMINAMATH_CALUDE_sum_of_combinations_specific_combination_l164_16446


namespace NUMINAMATH_CALUDE_prob_third_batch_given_two_standard_l164_16445

/-- Represents the number of parts in each batch -/
def batch_size : ℕ := 20

/-- Represents the number of standard parts in the first batch -/
def standard_parts_batch1 : ℕ := 20

/-- Represents the number of standard parts in the second batch -/
def standard_parts_batch2 : ℕ := 15

/-- Represents the number of standard parts in the third batch -/
def standard_parts_batch3 : ℕ := 10

/-- Represents the probability of selecting a batch -/
def prob_select_batch : ℚ := 1 / 3

/-- Theorem stating the probability of selecting two standard parts from the third batch,
    given that two standard parts were selected consecutively from a randomly chosen batch -/
theorem prob_third_batch_given_two_standard : 
  (prob_select_batch * (standard_parts_batch3 / batch_size)^2) /
  (prob_select_batch * (standard_parts_batch1 / batch_size)^2 +
   prob_select_batch * (standard_parts_batch2 / batch_size)^2 +
   prob_select_batch * (standard_parts_batch3 / batch_size)^2) = 4 / 29 := by
  sorry

end NUMINAMATH_CALUDE_prob_third_batch_given_two_standard_l164_16445


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l164_16471

-- Define the box contents
def white_balls : ℕ := 1
def red_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := white_balls + red_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball : prob_white = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l164_16471


namespace NUMINAMATH_CALUDE_special_function_properties_l164_16430

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x > 0 → f x < 0)

/-- The main theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l164_16430


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l164_16415

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_prod : a 7 * a 9 = 4) 
  (h_a4 : a 4 = 1) : 
  a 12 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l164_16415


namespace NUMINAMATH_CALUDE_symmetry_of_point_l164_16489

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to X-axis -/
def symmetryXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

theorem symmetry_of_point :
  let P : Point3D := { x := -1, y := 8, z := 4 }
  symmetryXAxis P = { x := -1, y := -8, z := 4 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l164_16489


namespace NUMINAMATH_CALUDE_subset_condition_empty_intersection_l164_16464

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

theorem empty_intersection (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_empty_intersection_l164_16464


namespace NUMINAMATH_CALUDE_rope_for_third_post_l164_16498

theorem rope_for_third_post 
  (total_rope : ℕ) 
  (first_post second_post fourth_post : ℕ) 
  (h1 : total_rope = 70)
  (h2 : first_post = 24)
  (h3 : second_post = 20)
  (h4 : fourth_post = 12) :
  total_rope - (first_post + second_post + fourth_post) = 14 := by
  sorry

end NUMINAMATH_CALUDE_rope_for_third_post_l164_16498


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l164_16441

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_of_power : 
  tens_digit ((3 + 4)^25) + ones_digit ((3 + 4)^25) = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l164_16441


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_57_degree_angle_l164_16434

-- Define the original angle
def original_angle : ℝ := 57

-- Define complement
def complement (angle : ℝ) : ℝ := 90 - angle

-- Define supplement
def supplement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem supplement_of_complement_of_57_degree_angle :
  supplement (complement original_angle) = 147 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_57_degree_angle_l164_16434


namespace NUMINAMATH_CALUDE_class_size_with_sports_participation_l164_16490

/-- The number of students in a class with given sports participation. -/
theorem class_size_with_sports_participation
  (football : ℕ)
  (long_tennis : ℕ)
  (both : ℕ)
  (neither : ℕ)
  (h1 : football = 26)
  (h2 : long_tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 6) :
  football + long_tennis - both + neither = 35 := by
  sorry

end NUMINAMATH_CALUDE_class_size_with_sports_participation_l164_16490


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l164_16443

/-- Quadrilateral EFGH with given properties -/
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)
  (EF_perp_FG : (F.1 - E.1) * (G.2 - F.2) + (F.2 - E.2) * (G.1 - F.1) = 0)
  (HG_perp_FG : (G.1 - H.1) * (G.2 - F.2) + (G.2 - H.2) * (G.1 - F.1) = 0)
  (EF_length : Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) = 12)
  (HG_length : Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2) = 3)
  (FG_length : Real.sqrt ((G.1 - F.1)^2 + (G.2 - F.2)^2) = 16)

/-- The perimeter of quadrilateral EFGH is 31 + √337 -/
theorem quadrilateral_perimeter (q : Quadrilateral) :
  Real.sqrt ((q.E.1 - q.F.1)^2 + (q.E.2 - q.F.2)^2) +
  Real.sqrt ((q.F.1 - q.G.1)^2 + (q.F.2 - q.G.2)^2) +
  Real.sqrt ((q.G.1 - q.H.1)^2 + (q.G.2 - q.H.2)^2) +
  Real.sqrt ((q.H.1 - q.E.1)^2 + (q.H.2 - q.E.2)^2) =
  31 + Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l164_16443


namespace NUMINAMATH_CALUDE_possible_values_of_a_l164_16497

theorem possible_values_of_a :
  ∀ (a b c : ℤ), (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) → (a = 3 ∨ a = 7) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l164_16497


namespace NUMINAMATH_CALUDE_two_squares_five_points_arrangement_l164_16451

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a square in 2D space
structure Square where
  center : Point
  side_length : ℝ

-- Define a function to check if a point is inside a square
def is_inside (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side_length / 2 ∧
  abs (p.y - s.center.y) ≤ s.side_length / 2

-- Define the theorem
theorem two_squares_five_points_arrangement :
  ∃ (s1 s2 : Square) (p1 p2 p3 p4 p5 : Point),
    (is_inside p1 s1 ∧ is_inside p2 s1 ∧ is_inside p3 s1) ∧
    (is_inside p1 s2 ∧ is_inside p2 s2 ∧ is_inside p3 s2 ∧ is_inside p4 s2) :=
  sorry

end NUMINAMATH_CALUDE_two_squares_five_points_arrangement_l164_16451


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_sixth_l164_16455

theorem sqrt_fifth_power_sixth : (((5 : ℝ) ^ (1/2)) ^ 5) ^ (1/2) ^ 6 = 125 * (125 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_sixth_l164_16455


namespace NUMINAMATH_CALUDE_probability_open_path_correct_l164_16477

/-- The probability of being able to go from the first floor to the last floor through only open doors in a building with n floors and randomly locked doors. -/
def probability_open_path (n : ℕ) : ℚ :=
  if n ≤ 1 then 1
  else (2 ^ (n - 1) : ℚ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℚ)

/-- Theorem stating the probability of an open path in the building. -/
theorem probability_open_path_correct (n : ℕ) (h : n > 1) :
  probability_open_path n =
    (2 ^ (n - 1) : ℚ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℚ) := by
  sorry

#eval probability_open_path 5

end NUMINAMATH_CALUDE_probability_open_path_correct_l164_16477


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l164_16482

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y => y = -2 * x + 1
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y → (y = 1/2 * x - 3/2)) →  -- Slope of L1 is 1/2
  (L2 P.1 P.2) →  -- L2 passes through P
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L2 x₂ y₂ → (x₂ - x₁) * (y₂ - y₁) = -1) →  -- L1 and L2 are perpendicular
  ∃ m b, ∀ x y, L2 x y ↔ y = m * x + b ∧ m = -2 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l164_16482


namespace NUMINAMATH_CALUDE_set_operations_l164_16470

def A : Set ℝ := {x | x^2 + 3*x - 4 > 0}
def B : Set ℝ := {x | x^2 - x - 6 < 0}

theorem set_operations :
  (A ∩ B = {x | 1 < x ∧ x < 3}) ∧
  (Set.compl (A ∩ B) = {x | x ≤ 1 ∨ x ≥ 3}) ∧
  (A ∪ Set.compl B = {x | x ≤ -2 ∨ x > 1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l164_16470


namespace NUMINAMATH_CALUDE_orange_shells_count_l164_16405

theorem orange_shells_count (total shells purple pink yellow blue : ℕ) 
  (h1 : total = 65)
  (h2 : purple = 13)
  (h3 : pink = 8)
  (h4 : yellow = 18)
  (h5 : blue = 12) :
  total - (purple + pink + yellow + blue) = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_shells_count_l164_16405


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l164_16448

/-- A geometric sequence with a_1 * a_3 = a_4 = 4 has a_6 = 8 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_condition : a 1 * a 3 = a 4 ∧ a 4 = 4) : a 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l164_16448


namespace NUMINAMATH_CALUDE_x_value_when_y_is_4_l164_16400

-- Define the inverse square relationship between x and y
def inverse_square_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, x = k / (y ^ 2)

-- State the theorem
theorem x_value_when_y_is_4 :
  ∀ x₀ y₀ x₁ y₁ : ℝ,
  inverse_square_relation x₀ y₀ →
  inverse_square_relation x₁ y₁ →
  x₀ = 1 →
  y₀ = 3 →
  y₁ = 4 →
  x₁ = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_4_l164_16400


namespace NUMINAMATH_CALUDE_candy_game_solution_l164_16420

theorem candy_game_solution (total_questions : ℕ) (correct_reward : ℕ) (incorrect_penalty : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_reward = 7)
  (h3 : incorrect_penalty = 3) :
  ∃ correct_answers : ℕ, 
    correct_answers * correct_reward = (total_questions - correct_answers) * incorrect_penalty ∧ 
    correct_answers = 15 := by
  sorry

end NUMINAMATH_CALUDE_candy_game_solution_l164_16420


namespace NUMINAMATH_CALUDE_alpha_plus_three_beta_range_l164_16456

theorem alpha_plus_three_beta_range (α β : ℝ) 
  (h1 : -1 ≤ α + β ∧ α + β ≤ 1) 
  (h2 : 1 ≤ α + 2*β ∧ α + 2*β ≤ 3) : 
  1 ≤ α + 3*β ∧ α + 3*β ≤ 7 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_three_beta_range_l164_16456


namespace NUMINAMATH_CALUDE_and_or_relationship_l164_16427

theorem and_or_relationship (p q : Prop) : 
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by
  sorry

end NUMINAMATH_CALUDE_and_or_relationship_l164_16427


namespace NUMINAMATH_CALUDE_jack_walking_distance_l164_16474

/-- Calculates the distance walked given the time in hours and minutes and the walking rate in miles per hour -/
def distance_walked (hours : ℕ) (minutes : ℕ) (rate : ℚ) : ℚ :=
  rate * (hours + minutes / 60)

/-- Proves that walking for 1 hour and 15 minutes at a rate of 7.2 miles per hour results in a distance of 9 miles -/
theorem jack_walking_distance :
  distance_walked 1 15 (7.2 : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_jack_walking_distance_l164_16474


namespace NUMINAMATH_CALUDE_integer_root_values_l164_16424

theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, x^3 + a*x^2 + 3*x + 7 = 0) ↔ (a = -11 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_values_l164_16424


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l164_16496

theorem simplify_and_evaluate_expression (x : ℚ) (h : x = -4) :
  (x^2 / (x - 1) - x + 1) / ((4 * x^2 - 4 * x + 1) / (1 - x)) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l164_16496


namespace NUMINAMATH_CALUDE_inequality_proof_l164_16486

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (a - b + c) * (1/a - 1/b + 1/c) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l164_16486


namespace NUMINAMATH_CALUDE_peggy_needs_825_stamps_l164_16406

/-- The number of stamps Peggy needs to add to have as many as Bert -/
def stamps_to_add (peggy_stamps ernie_stamps bert_stamps : ℕ) : ℕ :=
  bert_stamps - peggy_stamps

/-- Proof that Peggy needs to add 825 stamps to have as many as Bert -/
theorem peggy_needs_825_stamps : 
  ∀ (peggy_stamps ernie_stamps bert_stamps : ℕ),
    peggy_stamps = 75 →
    ernie_stamps = 3 * peggy_stamps →
    bert_stamps = 4 * ernie_stamps →
    stamps_to_add peggy_stamps ernie_stamps bert_stamps = 825 := by
  sorry

end NUMINAMATH_CALUDE_peggy_needs_825_stamps_l164_16406


namespace NUMINAMATH_CALUDE_fraction_simplification_l164_16481

theorem fraction_simplification (a b : ℚ) (ha : a = 5) (hb : b = 4) :
  (1 / b) / (1 / a) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l164_16481


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l164_16460

theorem diet_soda_bottles (total : ℕ) (regular : ℕ) (h1 : total = 30) (h2 : regular = 28) :
  total - regular = 2 := by
  sorry

end NUMINAMATH_CALUDE_diet_soda_bottles_l164_16460


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_remainder_l164_16467

theorem polynomial_division_quotient_remainder (z : ℚ) :
  3 * z^4 - 4 * z^3 + 5 * z^2 - 11 * z + 2 =
  (2 + 3 * z) * (z^3 - 2 * z^2 + 3 * z - 17/3) + 40/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_remainder_l164_16467


namespace NUMINAMATH_CALUDE_cylinder_height_ratio_l164_16499

/-- 
Given two right circular cylinders V and B, where:
- The radius of V is twice the radius of B
- It costs $4 to fill half of B
- It costs $16 to fill V completely
This theorem proves that the ratio of the height of V to the height of B is 1:2
-/
theorem cylinder_height_ratio 
  (r_B : ℝ) -- radius of cylinder B
  (h_B : ℝ) -- height of cylinder B
  (h_V : ℝ) -- height of cylinder V
  (cost_half_B : ℝ) -- cost to fill half of cylinder B
  (cost_full_V : ℝ) -- cost to fill cylinder V completely
  (h_radius : r_B > 0) -- radius of B is positive
  (h_cost_half_B : cost_half_B = 4) -- cost to fill half of B is $4
  (h_cost_full_V : cost_full_V = 16) -- cost to fill V completely is $16
  : h_V / h_B = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_ratio_l164_16499


namespace NUMINAMATH_CALUDE_c_symmetric_l164_16485

def c : ℕ → ℕ → ℤ
  | m, 0 => 1
  | 0, n => 1
  | m+1, n+1 => c m (n+1) - (n+1) * c m n

theorem c_symmetric (m n : ℕ) (hm : m > 0) (hn : n > 0) : c m n = c n m := by
  sorry

end NUMINAMATH_CALUDE_c_symmetric_l164_16485


namespace NUMINAMATH_CALUDE_batsman_average_increase_17_innings_l164_16442

def batsman_average_increase (total_innings : ℕ) (last_innings_score : ℕ) (final_average : ℚ) : ℚ :=
  let previous_total := (total_innings - 1) * (total_innings * final_average - last_innings_score) / total_innings
  let previous_average := previous_total / (total_innings - 1)
  final_average - previous_average

theorem batsman_average_increase_17_innings 
  (h1 : total_innings = 17)
  (h2 : last_innings_score = 85)
  (h3 : final_average = 37) :
  batsman_average_increase total_innings last_innings_score final_average = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_17_innings_l164_16442


namespace NUMINAMATH_CALUDE_power_function_through_point_l164_16450

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x ^ α) →
  f 2 = Real.sqrt 2 →
  α = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l164_16450


namespace NUMINAMATH_CALUDE_quadratic_property_l164_16401

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_property (a b c : ℝ) (h1 : a ≠ 0) :
  (quadratic a b c 2 = 0.35) →
  (quadratic a b c 4 = 0.35) →
  (quadratic a b c 5 = 3) →
  (a + b + c) * (-b / a) = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_property_l164_16401


namespace NUMINAMATH_CALUDE_exists_m_satisfying_conditions_l164_16488

-- Define the sequence of concatenated natural numbers
def concatNaturals : ℕ → ℕ
| 0 => 0
| (n + 1) => concatNaturals n * 10^(Nat.digits 10 (n + 1)).length + (n + 1)

-- Define A_k as the first k digits of the concatenated sequence
def A (k : ℕ) : ℕ := concatNaturals k % (10^k)

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

-- State the theorem
theorem exists_m_satisfying_conditions (n : ℕ+) :
  ∃ m : ℕ+, (A m.val ∣ n) ∧ (m ∣ n) ∧ (sumOfDigits (A m.val) ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_satisfying_conditions_l164_16488


namespace NUMINAMATH_CALUDE_probability_no_growth_pies_l164_16426

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def given_pies : ℕ := 3

theorem probability_no_growth_pies :
  let shrink_pies := total_pies - growth_pies
  let prob_mary_no_growth := (shrink_pies.choose given_pies : ℚ) / (total_pies.choose given_pies : ℚ)
  let prob_alice_no_growth := 1 - (1 - prob_mary_no_growth)
  prob_mary_no_growth + prob_alice_no_growth = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_growth_pies_l164_16426


namespace NUMINAMATH_CALUDE_solution_subset_nonpositive_l164_16458

/-- The solution set of |x| > ax + 1 is a subset of {x | x ≤ 0} if and only if a ≥ 1 -/
theorem solution_subset_nonpositive (a : ℝ) :
  (∀ x : ℝ, |x| > a * x + 1 → x ≤ 0) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_subset_nonpositive_l164_16458


namespace NUMINAMATH_CALUDE_integral_f_minus_pi_to_zero_l164_16469

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem integral_f_minus_pi_to_zero :
  ∫ x in Set.Icc (-Real.pi) 0, f x = -2 - (1/2) * Real.pi^2 := by
  sorry

end NUMINAMATH_CALUDE_integral_f_minus_pi_to_zero_l164_16469


namespace NUMINAMATH_CALUDE_distance_between_points_l164_16468

def point1 : ℝ × ℝ := (-3, 1)
def point2 : ℝ × ℝ := (4, -9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 149 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l164_16468


namespace NUMINAMATH_CALUDE_unique_cube_fourth_power_l164_16425

theorem unique_cube_fourth_power : 
  ∃! (K : ℤ), ∃ (Z : ℤ),
    600 < Z ∧ Z < 2000 ∧ 
    Z = K^4 ∧ 
    ∃ (n : ℤ), Z = n^3 ∧
    K = 8 :=
sorry

end NUMINAMATH_CALUDE_unique_cube_fourth_power_l164_16425


namespace NUMINAMATH_CALUDE_expression_simplification_l164_16407

theorem expression_simplification (x : ℝ) 
  (hx : x ≠ 0 ∧ x ≠ 3 ∧ x ≠ 2) : 
  (x - 5) / (x - 3) - ((x^2 + 2*x + 1) / (x^2 + x)) / ((x + 1) / (x - 2)) = 
  -6 / (x^2 - 3*x) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l164_16407


namespace NUMINAMATH_CALUDE_paint_left_is_four_liters_l164_16433

/-- The amount of paint Dexter used in gallons -/
def dexter_paint : ℚ := 3/8

/-- The amount of paint Jay used in gallons -/
def jay_paint : ℚ := 5/8

/-- The conversion factor from gallons to liters -/
def gallon_to_liter : ℚ := 4

/-- The total amount of paint in gallons -/
def total_paint : ℚ := 2

theorem paint_left_is_four_liters : 
  (total_paint * gallon_to_liter) - ((dexter_paint + jay_paint) * gallon_to_liter) = 4 := by
  sorry

end NUMINAMATH_CALUDE_paint_left_is_four_liters_l164_16433


namespace NUMINAMATH_CALUDE_symmetry_about_x_equals_one_symmetry_about_x_equals_three_halves_odd_function_shift_l164_16478

open Function

-- Define a function f from reals to reals
variable (f : ℝ → ℝ)

-- Statement ①
theorem symmetry_about_x_equals_one :
  (∀ x, f (x - 1) = f (1 - x)) ↔ 
  (∀ x, f (2 - x) = f x) :=
sorry

-- Statement ②
theorem symmetry_about_x_equals_three_halves 
  (h1 : ∀ x, f (-3/2 - x) = f x) 
  (h2 : ∀ x, f (x + 3/2) = -f x) :
  ∀ x, f (3 - x) = f x :=
sorry

-- Statement ③
theorem odd_function_shift
  (h : ∀ x, f (x + 2) = -f (-x + 4)) :
  ∀ x, f ((x + 3) + 3) = -f (-(x + 3) + 3) :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_x_equals_one_symmetry_about_x_equals_three_halves_odd_function_shift_l164_16478


namespace NUMINAMATH_CALUDE_stamps_in_first_book_l164_16462

theorem stamps_in_first_book (x : ℕ) : 
  (4 * x + 6 * 15 = 130) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_stamps_in_first_book_l164_16462


namespace NUMINAMATH_CALUDE_intersection_complement_equals_seven_l164_16459

def U : Finset Nat := {4,5,6,7,8}
def M : Finset Nat := {5,8}
def N : Finset Nat := {1,3,5,7,9}

theorem intersection_complement_equals_seven :
  (N ∩ (U \ M)) = {7} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_seven_l164_16459


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_l164_16440

/-- The perpendicular bisector of a line segment connecting two points (x₁, y₁) and (x₂, y₂) 
    is defined by the equation x + 2y = c. -/
def is_perpendicular_bisector (x₁ y₁ x₂ y₂ c : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + 2 * midpoint_y = c

/-- The value of c for the perpendicular bisector of the line segment 
    connecting (2,4) and (8,16) is 25. -/
theorem perpendicular_bisector_c : 
  is_perpendicular_bisector 2 4 8 16 25 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_l164_16440


namespace NUMINAMATH_CALUDE_polynomial_factorization_l164_16436

theorem polynomial_factorization (x y : ℝ) : 
  x^2 - y^2 - 2*x - 4*y - 3 = (x+y+1)*(x-y-3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l164_16436


namespace NUMINAMATH_CALUDE_area_between_line_and_curve_l164_16428

/-- The area enclosed between the line y = 2x and the curve y = x^2 from x = 0 to x = 2 is 4/3 -/
theorem area_between_line_and_curve : 
  ∫ x in (0 : ℝ)..2, (2 * x - x^2) = 4/3 := by sorry

end NUMINAMATH_CALUDE_area_between_line_and_curve_l164_16428


namespace NUMINAMATH_CALUDE_fraction_problem_l164_16444

theorem fraction_problem : ∃ x : ℚ, x * 1206 = 3 * 134 ∧ x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l164_16444


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_C_nonempty_implies_a_gt_one_l164_16484

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for part (1)
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 9} := by sorry

-- Theorem for part (2)
theorem intersection_A_C_nonempty_implies_a_gt_one (a : ℝ) :
  (A ∩ C a).Nonempty → a > 1 := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_C_nonempty_implies_a_gt_one_l164_16484


namespace NUMINAMATH_CALUDE_variation_problem_l164_16466

theorem variation_problem (R S T : ℚ) (c : ℚ) : 
  (∀ R S T, R = c * S / T) →  -- R varies directly as S and inversely as T
  (2 = c * 4 / (1/2)) →       -- When R = 2, T = 1/2, S = 4
  (8 = c * S / (1/3)) →       -- When R = 8 and T = 1/3
  S = 32/3 := by
sorry

end NUMINAMATH_CALUDE_variation_problem_l164_16466


namespace NUMINAMATH_CALUDE_distance_inequality_l164_16416

theorem distance_inequality (a : ℝ) : 
  (|a - 1| < 3) → (-2 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_distance_inequality_l164_16416


namespace NUMINAMATH_CALUDE_sum_lent_is_400_l164_16475

/-- Calculates the simple interest for a given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem sum_lent_is_400 :
  let rate : ℚ := 4
  let time : ℚ := 8
  let principal : ℚ := 400
  simpleInterest principal rate time = principal - 272 :=
by
  sorry

#check sum_lent_is_400

end NUMINAMATH_CALUDE_sum_lent_is_400_l164_16475


namespace NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l164_16435

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem ninth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_sixth : a 6 = 16)
  (h_twelfth : a 12 = 4) :
  a 9 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l164_16435


namespace NUMINAMATH_CALUDE_trapezoid_area_in_hexagon_triangle_l164_16423

/-- Given a regular hexagon with an inscribed equilateral triangle, this theorem calculates the area of one of the six congruent trapezoids formed between the hexagon and the triangle. -/
theorem trapezoid_area_in_hexagon_triangle (hexagon_area : ℝ) (triangle_area : ℝ) :
  hexagon_area = 24 →
  triangle_area = 4 →
  (hexagon_area - triangle_area) / 6 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_in_hexagon_triangle_l164_16423


namespace NUMINAMATH_CALUDE_tan_570_degrees_l164_16493

theorem tan_570_degrees : Real.tan (570 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_570_degrees_l164_16493


namespace NUMINAMATH_CALUDE_integer_roots_quadratic_l164_16431

theorem integer_roots_quadratic (m n : ℤ) : 
  (∃ x y : ℤ, (2*m - 3)*(n - 1)*x^2 + (2*m - 3)*(n - 1)*(m - n - 4)*x - 2*(2*m - 3)*(n - 1)*(m - n - 2) - 1 = 0 ∧
               (2*m - 3)*(n - 1)*y^2 + (2*m - 3)*(n - 1)*(m - n - 4)*y - 2*(2*m - 3)*(n - 1)*(m - n - 2) - 1 = 0 ∧
               x ≠ y) ↔
  ((m = 2 ∧ n = 2) ∨ (m = 2 ∧ n = 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_quadratic_l164_16431


namespace NUMINAMATH_CALUDE_exp_addition_property_l164_16422

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_addition_property (x y : ℝ) : f (x + y) = f x * f y := by
  sorry

end NUMINAMATH_CALUDE_exp_addition_property_l164_16422


namespace NUMINAMATH_CALUDE_smallest_x_value_l164_16432

theorem smallest_x_value (x : ℝ) : 
  (2 * x^2 + 24 * x - 60 = x * (x + 13)) → x ≥ -15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l164_16432


namespace NUMINAMATH_CALUDE_complex_root_implies_positive_and_triangle_l164_16487

theorem complex_root_implies_positive_and_triangle (a b c α β : ℝ) : 
  (α > 0) →
  (β ≠ 0) →
  (Complex.I : ℂ)^2 = -1 →
  (α + β * Complex.I : ℂ)^2 - (a + b + c) * (α + β * Complex.I : ℂ) + (a * b + b * c + c * a : ℂ) = 0 →
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ (Real.sqrt a < Real.sqrt b + Real.sqrt c) :=
by sorry

end NUMINAMATH_CALUDE_complex_root_implies_positive_and_triangle_l164_16487


namespace NUMINAMATH_CALUDE_pages_left_after_tuesday_l164_16418

def pages_read_monday : ℕ := 15
def extra_pages_tuesday : ℕ := 16
def total_pages : ℕ := 64

def pages_left : ℕ := total_pages - (pages_read_monday + (pages_read_monday + extra_pages_tuesday))

theorem pages_left_after_tuesday : pages_left = 18 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_after_tuesday_l164_16418


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l164_16476

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (Complex.I - 1) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l164_16476


namespace NUMINAMATH_CALUDE_unique_fixed_point_of_odd_symmetries_l164_16417

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Central symmetry transformation about a point -/
def centralSymmetry (center : Point) (p : Point) : Point :=
  { x := 2 * center.x - p.x, y := 2 * center.y - p.y }

/-- Composition of central symmetries -/
def compositeSymmetry (centers : List Point) : Point → Point :=
  centers.foldl (λ f center p => f (centralSymmetry center p)) id

theorem unique_fixed_point_of_odd_symmetries (n : ℕ) :
  let m := 2 * n + 1
  ∀ (midpoints : List Point),
    midpoints.length = m →
    ∃! (fixedPoint : Point), compositeSymmetry midpoints fixedPoint = fixedPoint :=
by
  sorry

#check unique_fixed_point_of_odd_symmetries

end NUMINAMATH_CALUDE_unique_fixed_point_of_odd_symmetries_l164_16417


namespace NUMINAMATH_CALUDE_library_visitors_on_sunday_l164_16449

/-- Proves that the average number of visitors on Sundays is 140 given the specified conditions --/
theorem library_visitors_on_sunday (
  total_days : Nat) 
  (sunday_count : Nat)
  (avg_visitors_per_day : ℝ)
  (avg_visitors_non_sunday : ℝ)
  (h1 : total_days = 30)
  (h2 : sunday_count = 5)
  (h3 : avg_visitors_per_day = 90)
  (h4 : avg_visitors_non_sunday = 80)
  : ℝ :=
by
  -- Proof goes here
  sorry

#check library_visitors_on_sunday

end NUMINAMATH_CALUDE_library_visitors_on_sunday_l164_16449


namespace NUMINAMATH_CALUDE_gcf_60_72_l164_16419

theorem gcf_60_72 : Nat.gcd 60 72 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_60_72_l164_16419


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l164_16457

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l164_16457


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l164_16480

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 2 * x^2 - m * x + n = 0 ∧ 2 * y^2 - m * y + n = 0 ∧ x + y = 10 ∧ x * y = 24) →
  m + n = 68 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l164_16480


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l164_16411

theorem min_value_sum_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 3) :
  (1 / (1 + a) + 4 / (4 + b)) ≥ 9/8 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 3 ∧ 1 / (1 + a₀) + 4 / (4 + b₀) = 9/8 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l164_16411


namespace NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l164_16412

/-- A function f is decreasing on ℝ -/
def DecreasingOnReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- Given a decreasing function f on ℝ, if f(m-1) > f(2m-1), then m > 0 -/
theorem range_of_m_for_decreasing_function (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingOnReals f) (h_inequality : f (m - 1) > f (2 * m - 1)) : 
  m > 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l164_16412


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l164_16491

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 100)
  (h2 : selling_price = 120) : 
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l164_16491


namespace NUMINAMATH_CALUDE_power_simplification_l164_16472

theorem power_simplification (x : ℝ) : (5 * x^4)^3 = 125 * x^12 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l164_16472


namespace NUMINAMATH_CALUDE_no_numbers_equal_seven_times_sum_of_digits_l164_16410

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem no_numbers_equal_seven_times_sum_of_digits : 
  ∀ n : ℕ, n ≤ 1000 → n ≠ 7 * sum_of_digits n :=
sorry

end NUMINAMATH_CALUDE_no_numbers_equal_seven_times_sum_of_digits_l164_16410


namespace NUMINAMATH_CALUDE_supplementary_angles_difference_l164_16403

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 3 →  -- ratio of angles is 5:3
  abs (a - b) = 45 :=  -- positive difference is 45°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_difference_l164_16403


namespace NUMINAMATH_CALUDE_parabola_c_value_l164_16463

/-- A parabola passing through three given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ := λ x => x^2 + b*x + c
  point1 : eq 2 = 12
  point2 : eq (-2) = 0
  point3 : eq 4 = 40

/-- The value of c for the parabola passing through (2, 12), (-2, 0), and (4, 40) is 2 -/
theorem parabola_c_value (p : Parabola) : p.c = 2 := by
  sorry

#check parabola_c_value

end NUMINAMATH_CALUDE_parabola_c_value_l164_16463
