import Mathlib

namespace NUMINAMATH_CALUDE_inverse_proportion_through_neg_one_three_l1118_111870

/-- An inverse proportion function passing through (-1, 3) has k = -3 --/
theorem inverse_proportion_through_neg_one_three (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (k / x = 3 ↔ x = -1)) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_through_neg_one_three_l1118_111870


namespace NUMINAMATH_CALUDE_rectangle_area_l1118_111883

/-- The area of a rectangle with perimeter 176 inches and length 8 inches more than its width is 1920 square inches. -/
theorem rectangle_area (w l : ℝ) (h1 : l = w + 8) (h2 : 2*l + 2*w = 176) : w * l = 1920 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1118_111883


namespace NUMINAMATH_CALUDE_roots_cubic_expression_l1118_111805

theorem roots_cubic_expression (γ δ : ℝ) : 
  (γ^2 - 3*γ + 2 = 0) → 
  (δ^2 - 3*δ + 2 = 0) → 
  8*γ^3 - 6*δ^2 = 48 := by
sorry

end NUMINAMATH_CALUDE_roots_cubic_expression_l1118_111805


namespace NUMINAMATH_CALUDE_total_crates_sold_l1118_111894

/-- Calculates the total number of crates sold over four days given specific sales conditions --/
theorem total_crates_sold (monday : ℕ) : monday = 5 → 28 = monday + (2 * monday) + (2 * monday - 2) + (monday) := by
  sorry

end NUMINAMATH_CALUDE_total_crates_sold_l1118_111894


namespace NUMINAMATH_CALUDE_angle_triple_supplement_measure_l1118_111879

theorem angle_triple_supplement_measure : 
  ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ x = 3 * (180 - x) ∧ x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_measure_l1118_111879


namespace NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_alpha_l1118_111884

theorem cos_two_pi_thirds_minus_alpha (α : ℝ) 
  (h : Real.sin (α - π/6) = 1/3) : 
  Real.cos ((2*π)/3 - α) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_alpha_l1118_111884


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1118_111810

-- Define the repeating decimal 0.6̄
def repeating_decimal : ℚ := 2/3

-- State the theorem
theorem product_of_repeating_decimal_and_eight :
  repeating_decimal * 8 = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1118_111810


namespace NUMINAMATH_CALUDE_sum_product_bound_l1118_111835

theorem sum_product_bound (α β γ : ℝ) (h : α^2 + β^2 + γ^2 = 1) :
  -1/2 ≤ α*β + β*γ + γ*α ∧ α*β + β*γ + γ*α ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_bound_l1118_111835


namespace NUMINAMATH_CALUDE_min_chicken_hits_l1118_111819

def ring_toss (chicken monkey dog : ℕ) : Prop :=
  chicken * 9 + monkey * 5 + dog * 2 = 61 ∧
  chicken + monkey + dog = 10 ∧
  chicken ≥ 1 ∧ monkey ≥ 1 ∧ dog ≥ 1

theorem min_chicken_hits :
  ∀ chicken monkey dog : ℕ,
    ring_toss chicken monkey dog →
    chicken ≥ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_min_chicken_hits_l1118_111819


namespace NUMINAMATH_CALUDE_xy_less_18_implies_x_less_2_or_y_less_9_l1118_111820

theorem xy_less_18_implies_x_less_2_or_y_less_9 :
  ∀ x y : ℝ, x * y < 18 → x < 2 ∨ y < 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_less_18_implies_x_less_2_or_y_less_9_l1118_111820


namespace NUMINAMATH_CALUDE_min_face_sum_l1118_111872

-- Define a cube as a set of 8 integers
def Cube := Fin 8 → ℕ

-- Define a face as a set of 4 vertices
def Face := Fin 4 → Fin 8

-- Condition: numbers are from 1 to 8
def valid_cube (c : Cube) : Prop :=
  (∀ i, c i ≥ 1 ∧ c i ≤ 8) ∧ (∀ i j, i ≠ j → c i ≠ c j)

-- Condition: sum of any three vertices on a face is at least 10
def valid_face_sums (c : Cube) (f : Face) : Prop :=
  ∀ i j k, i < j → j < k → c (f i) + c (f j) + c (f k) ≥ 10

-- The sum of numbers on a face
def face_sum (c : Cube) (f : Face) : ℕ :=
  (c (f 0)) + (c (f 1)) + (c (f 2)) + (c (f 3))

-- The theorem to prove
theorem min_face_sum (c : Cube) :
  valid_cube c → (∀ f : Face, valid_face_sums c f) →
  ∃ f : Face, face_sum c f = 16 ∧ ∀ g : Face, face_sum c g ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_face_sum_l1118_111872


namespace NUMINAMATH_CALUDE_circle_numbers_solution_l1118_111874

def CircleNumbers (a b c d e f : ℚ) : Prop :=
  a + b + c + d + e + f = 1 ∧
  a = |b - c| ∧
  b = |c - d| ∧
  c = |d - e| ∧
  d = |e - f| ∧
  e = |f - a| ∧
  f = |a - b|

theorem circle_numbers_solution :
  ∀ a b c d e f : ℚ, CircleNumbers a b c d e f →
  ((a = 1/4 ∧ b = 1/4 ∧ c = 0 ∧ d = 1/4 ∧ e = 1/4 ∧ f = 0) ∨
   (a = 1/4 ∧ b = 0 ∧ c = 1/4 ∧ d = 1/4 ∧ e = 0 ∧ f = 1/4) ∨
   (a = 0 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 0 ∧ e = 1/4 ∧ f = 1/4)) :=
by sorry

end NUMINAMATH_CALUDE_circle_numbers_solution_l1118_111874


namespace NUMINAMATH_CALUDE_saramago_readers_ratio_l1118_111818

theorem saramago_readers_ratio (W : ℕ) (S K B : ℕ) : 
  W = 150 →
  K = W / 6 →
  W - S - K + B = S - B - 1 →
  B = 12 →
  S * 2 = W :=
by sorry

end NUMINAMATH_CALUDE_saramago_readers_ratio_l1118_111818


namespace NUMINAMATH_CALUDE_rainfall_increase_l1118_111866

/-- Given the rainfall data for Rainville in 2010 and 2011, prove the increase in average monthly rainfall. -/
theorem rainfall_increase (average_2010 total_2011 : ℝ) (h1 : average_2010 = 35) 
  (h2 : total_2011 = 504) : ∃ x : ℝ, 
  12 * (average_2010 + x) = total_2011 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_increase_l1118_111866


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1118_111844

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬(253 * m ≡ 989 * m [ZMOD 15])) →
  (253 * n ≡ 989 * n [ZMOD 15]) →
  n = 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1118_111844


namespace NUMINAMATH_CALUDE_seats_filled_percentage_l1118_111815

/-- Given a hall with 700 seats where 175 are vacant, prove that 75% of the seats are filled. -/
theorem seats_filled_percentage (total_seats : ℕ) (vacant_seats : ℕ) 
  (h1 : total_seats = 700) 
  (h2 : vacant_seats = 175) : 
  (((total_seats - vacant_seats : ℚ) / total_seats) * 100 : ℚ) = 75 := by
  sorry

#check seats_filled_percentage

end NUMINAMATH_CALUDE_seats_filled_percentage_l1118_111815


namespace NUMINAMATH_CALUDE_percentage_calculation_l1118_111891

theorem percentage_calculation (total : ℝ) (difference : ℝ) : 
  total = 6000 ∧ difference = 693 → 
  ∃ P : ℝ, (1/10 * total) - (P/100 * total) = difference ∧ P = 1.55 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1118_111891


namespace NUMINAMATH_CALUDE_distribute_seven_among_three_l1118_111876

/-- The number of ways to distribute n indistinguishable items among k distinct groups,
    with each group receiving at least one item. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 15 ways to distribute 7 recommended places among 3 schools,
    with each school receiving at least one place. -/
theorem distribute_seven_among_three :
  distribute 7 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_among_three_l1118_111876


namespace NUMINAMATH_CALUDE_james_berets_l1118_111863

/-- The number of spools required to make one beret -/
def spools_per_beret : ℕ := 3

/-- The number of red yarn spools James has -/
def red_spools : ℕ := 12

/-- The number of black yarn spools James has -/
def black_spools : ℕ := 15

/-- The number of blue yarn spools James has -/
def blue_spools : ℕ := 6

/-- The total number of spools James has -/
def total_spools : ℕ := red_spools + black_spools + blue_spools

/-- The number of berets James can make -/
def berets_made : ℕ := total_spools / spools_per_beret

theorem james_berets :
  berets_made = 11 := by sorry

end NUMINAMATH_CALUDE_james_berets_l1118_111863


namespace NUMINAMATH_CALUDE_jacobs_february_bill_l1118_111834

/-- Calculates the total cell phone bill given the plan details and usage --/
def calculate_bill (base_cost : ℚ) (included_hours : ℚ) (cost_per_text : ℚ) 
  (cost_per_extra_minute : ℚ) (texts_sent : ℚ) (hours_talked : ℚ) : ℚ :=
  let text_cost := texts_sent * cost_per_text
  let extra_hours := max (hours_talked - included_hours) 0
  let extra_minutes := extra_hours * 60
  let extra_cost := extra_minutes * cost_per_extra_minute
  base_cost + text_cost + extra_cost

/-- Theorem stating that Jacob's cell phone bill for February is $83.80 --/
theorem jacobs_february_bill :
  calculate_bill 25 25 0.08 0.13 150 31 = 83.80 := by
  sorry

end NUMINAMATH_CALUDE_jacobs_february_bill_l1118_111834


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1118_111896

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 2 = 0 ∧ x₂^2 + m*x₂ - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1118_111896


namespace NUMINAMATH_CALUDE_projection_length_l1118_111808

def vector_a : ℝ × ℝ := (3, 4)
def vector_b : ℝ × ℝ := (0, 1)

theorem projection_length :
  let a := vector_a
  let b := vector_b
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = 4 := by sorry

end NUMINAMATH_CALUDE_projection_length_l1118_111808


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1118_111850

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = -2) (h2 : x * y = -3) : 
  x^4 + y^4 = 82 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1118_111850


namespace NUMINAMATH_CALUDE_means_inequality_l1118_111880

theorem means_inequality (a b w v : ℝ) 
  (ha : a > 0) (hb : b > 0) (hw : w > 0) (hv : v > 0) 
  (hab : a ≠ b) (hwv : w + v = 1) : 
  (w * a + v * b) / (w + v) > Real.sqrt (a * b) ∧ 
  Real.sqrt (a * b) > 2 * a * b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_means_inequality_l1118_111880


namespace NUMINAMATH_CALUDE_ten_square_shape_perimeter_l1118_111833

/-- A shape made from unit squares joined edge to edge -/
structure UnitSquareShape where
  /-- The number of unit squares in the shape -/
  num_squares : ℕ
  /-- The perimeter of the shape in cm -/
  perimeter : ℕ

/-- Theorem: A shape made from 10 unit squares has a perimeter of 18 cm -/
theorem ten_square_shape_perimeter :
  ∀ (shape : UnitSquareShape),
    shape.num_squares = 10 →
    shape.perimeter = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ten_square_shape_perimeter_l1118_111833


namespace NUMINAMATH_CALUDE_integral_exp_sin_l1118_111889

open Real

theorem integral_exp_sin (α β : ℝ) :
  deriv (fun x => (exp (α * x) * (α * sin (β * x) - β * cos (β * x))) / (α^2 + β^2)) =
  fun x => exp (α * x) * sin (β * x) := by
sorry

end NUMINAMATH_CALUDE_integral_exp_sin_l1118_111889


namespace NUMINAMATH_CALUDE_point_coordinates_l1118_111877

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian coordinate system -/
def fourth_quadrant (p : Point) : Prop := p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ := |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ := |p.x|

theorem point_coordinates :
  ∀ (A : Point),
    fourth_quadrant A →
    distance_to_x_axis A = 3 →
    distance_to_y_axis A = 6 →
    A.x = 6 ∧ A.y = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1118_111877


namespace NUMINAMATH_CALUDE_line_circle_no_intersection_l1118_111888

/-- The range of m for which a line and circle have no intersection -/
theorem line_circle_no_intersection (m : ℝ) : 
  (∀ x y : ℝ, 3*x + 4*y + m ≠ 0 ∨ (x+1)^2 + (y-2)^2 ≠ 1) →
  m < -10 ∨ m > 0 :=
sorry

end NUMINAMATH_CALUDE_line_circle_no_intersection_l1118_111888


namespace NUMINAMATH_CALUDE_water_height_in_cylinder_l1118_111848

/-- The height of water in a cylinder when poured from a cone -/
theorem water_height_in_cylinder (cone_radius cone_height cyl_radius : ℝ) 
  (h_cone_radius : cone_radius = 12)
  (h_cone_height : cone_height = 18)
  (h_cyl_radius : cyl_radius = 24) : 
  (1 / 3 * π * cone_radius^2 * cone_height) / (π * cyl_radius^2) = 1.5 := by
  sorry

#check water_height_in_cylinder

end NUMINAMATH_CALUDE_water_height_in_cylinder_l1118_111848


namespace NUMINAMATH_CALUDE_no_solution_equation_l1118_111851

theorem no_solution_equation : 
  ¬∃ (x : ℝ), (2 / (x + 1) + 3 / (x - 1) = 6 / (x^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l1118_111851


namespace NUMINAMATH_CALUDE_sum_of_squares_not_prime_l1118_111801

theorem sum_of_squares_not_prime (a b c d : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : a * b = c * d) : 
  ¬ Nat.Prime (Int.natAbs (a^2 + b^2 + c^2 + d^2)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_not_prime_l1118_111801


namespace NUMINAMATH_CALUDE_gcd_problem_l1118_111899

theorem gcd_problem (b : ℤ) (h : 1039 ∣ b) : Int.gcd (b^2 + 7*b + 18) (b + 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1118_111899


namespace NUMINAMATH_CALUDE_equation_has_root_minus_one_l1118_111875

theorem equation_has_root_minus_one : ∃ x : ℝ, x = -1 ∧ x^2 - x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_root_minus_one_l1118_111875


namespace NUMINAMATH_CALUDE_bottle_cost_l1118_111859

theorem bottle_cost (total : ℕ) (wine_extra : ℕ) (h1 : total = 30) (h2 : wine_extra = 26) : 
  ∃ (bottle : ℕ), bottle + (bottle + wine_extra) = total ∧ bottle = 2 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cost_l1118_111859


namespace NUMINAMATH_CALUDE_sum_reciprocal_squared_bound_l1118_111887

theorem sum_reciprocal_squared_bound (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) 
  (h_sum : x₁ + x₂ + x₃ = 1) : 
  (1 / (1 + x₁^2)) + (1 / (1 + x₂^2)) + (1 / (1 + x₃^2)) ≤ 27/10 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squared_bound_l1118_111887


namespace NUMINAMATH_CALUDE_jamies_class_girls_l1118_111807

theorem jamies_class_girls (total : ℕ) (girls boys : ℕ) : 
  total = 35 →
  4 * girls = 3 * boys →
  girls + boys = total →
  girls = 15 := by
sorry

end NUMINAMATH_CALUDE_jamies_class_girls_l1118_111807


namespace NUMINAMATH_CALUDE_room_tiles_l1118_111821

/-- Calculates the number of tiles needed for a rectangular room with a border --/
def total_tiles (length width : ℕ) (border_width : ℕ) : ℕ :=
  let border_tiles := 2 * (length + width - 2 * border_width) * border_width
  let inner_length := length - 2 * border_width
  let inner_width := width - 2 * border_width
  let inner_tiles := (inner_length * inner_width) / 4
  border_tiles + inner_tiles

/-- Theorem stating that a 15x20 room with a 2-foot border requires 168 tiles --/
theorem room_tiles : total_tiles 20 15 2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_room_tiles_l1118_111821


namespace NUMINAMATH_CALUDE_amy_hair_length_l1118_111804

/-- Amy's hair length before the haircut -/
def hair_length_before : ℕ := 11

/-- Amy's hair length after the haircut -/
def hair_length_after : ℕ := 7

/-- Length of hair cut off -/
def hair_cut_off : ℕ := 4

/-- Theorem: Amy's hair length before the haircut was 11 inches -/
theorem amy_hair_length : hair_length_before = hair_length_after + hair_cut_off := by
  sorry

end NUMINAMATH_CALUDE_amy_hair_length_l1118_111804


namespace NUMINAMATH_CALUDE_climb_out_of_well_l1118_111826

/-- The number of days required for a man to climb out of a well -/
def days_to_climb_out (well_depth : ℕ) (climb_up : ℕ) (slip_down : ℕ) : ℕ :=
  let net_progress := climb_up - slip_down
  let days_before_last := (well_depth - climb_up) / net_progress
  days_before_last + 1

/-- Theorem stating that it takes 65 days to climb out of a 70-meter well 
    under specific climbing conditions -/
theorem climb_out_of_well : 
  days_to_climb_out 70 6 5 = 65 := by
  sorry

end NUMINAMATH_CALUDE_climb_out_of_well_l1118_111826


namespace NUMINAMATH_CALUDE_lesogoria_inhabitants_l1118_111854

-- Define the types of inhabitants
inductive Inhabitant
| Elf
| Dwarf

-- Define the types of statements
inductive Statement
| AboutGold
| AboutDwarf
| Other

-- Define a function to determine if a statement is true based on the speaker and the type of statement
def isTruthful (speaker : Inhabitant) (statement : Statement) : Prop :=
  match speaker, statement with
  | Inhabitant.Dwarf, Statement.AboutGold => false
  | Inhabitant.Elf, Statement.AboutDwarf => false
  | _, _ => true

-- Define the statements made by A and B
def statementA : Statement := Statement.AboutGold
def statementB : Statement := Statement.Other

-- Define the theorem
theorem lesogoria_inhabitants :
  ∃ (a b : Inhabitant),
    (isTruthful a statementA = false) ∧
    (isTruthful b statementB = true) ∧
    (a = Inhabitant.Dwarf) ∧
    (b = Inhabitant.Dwarf) :=
  sorry


end NUMINAMATH_CALUDE_lesogoria_inhabitants_l1118_111854


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l1118_111839

theorem min_product_of_three_numbers (x y z : ℝ) :
  x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  x ≤ 2*y ∧ x ≤ 2*z ∧ y ≤ 2*x ∧ y ≤ 2*z ∧ z ≤ 2*x ∧ z ≤ 2*y →
  x * y * z ≥ 1/32 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l1118_111839


namespace NUMINAMATH_CALUDE_time_before_second_rewind_is_45_l1118_111869

/-- Represents the movie watching scenario with rewinds -/
structure MovieWatching where
  totalTime : ℕ
  initialWatchTime : ℕ
  firstRewindTime : ℕ
  secondRewindTime : ℕ
  finalWatchTime : ℕ

/-- Calculates the time watched before the second rewind -/
def timeBeforeSecondRewind (m : MovieWatching) : ℕ :=
  m.totalTime - (m.initialWatchTime + m.firstRewindTime + m.secondRewindTime + m.finalWatchTime)

/-- Theorem stating the time watched before the second rewind is 45 minutes -/
theorem time_before_second_rewind_is_45 (m : MovieWatching)
    (h1 : m.totalTime = 120)
    (h2 : m.initialWatchTime = 35)
    (h3 : m.firstRewindTime = 5)
    (h4 : m.secondRewindTime = 15)
    (h5 : m.finalWatchTime = 20) :
    timeBeforeSecondRewind m = 45 := by
  sorry

end NUMINAMATH_CALUDE_time_before_second_rewind_is_45_l1118_111869


namespace NUMINAMATH_CALUDE_student_count_l1118_111855

theorem student_count : ∃ n : ℕ, n > 0 ∧ 
  (n / 2 : ℚ) + (n / 4 : ℚ) + (n / 8 : ℚ) + 3 = n ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1118_111855


namespace NUMINAMATH_CALUDE_vector_dot_product_symmetry_and_value_l1118_111802

/-- Given vectors a and b, and function f as defined, prove the axis of symmetry and a specific function value. -/
theorem vector_dot_product_symmetry_and_value 
  (x θ : ℝ) 
  (a : ℝ → ℝ × ℝ)
  (b : ℝ → ℝ × ℝ)
  (f : ℝ → ℝ)
  (h1 : a = λ x => (Real.sin x, 1))
  (h2 : b = λ x => (1, Real.cos x))
  (h3 : f = λ x => (a x).1 * (b x).1 + (a x).2 * (b x).2)
  (h4 : f (θ + π/4) = Real.sqrt 2 / 3)
  (h5 : 0 < θ)
  (h6 : θ < π/2) :
  (∃ k : ℤ, ∀ x, f x = f (2 * (k * π + π/4) - x)) ∧ 
  f (θ - π/4) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_symmetry_and_value_l1118_111802


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1118_111892

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -1

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop :=
  y - (focus.2) = m * (x - (focus.1))

-- Define the theorem
theorem parabola_intersection_theorem 
  (A B C : ℝ × ℝ) 
  (m : ℝ) 
  (h1 : parabola A.1 A.2)
  (h2 : parabola B.1 B.2)
  (h3 : directrix C.1)
  (h4 : line_through_focus m A.1 A.2)
  (h5 : line_through_focus m B.1 B.2)
  (h6 : line_through_focus m C.1 C.2)
  (h7 : A.2 * C.2 ≥ 0)  -- A and C on the same side of x-axis
  (h8 : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 
        2 * Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2)) :
  Real.sqrt ((B.1 - focus.1)^2 + (B.2 - focus.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1118_111892


namespace NUMINAMATH_CALUDE_total_out_of_pocket_is_190_50_l1118_111803

def consultation_cost : ℝ := 300
def consultation_coverage : ℝ := 0.8
def xray_cost : ℝ := 150
def xray_coverage : ℝ := 0.7
def medication_cost : ℝ := 75
def medication_coverage : ℝ := 0.5
def therapy_cost : ℝ := 120
def therapy_coverage : ℝ := 0.6

def total_out_of_pocket_cost : ℝ :=
  (1 - consultation_coverage) * consultation_cost +
  (1 - xray_coverage) * xray_cost +
  (1 - medication_coverage) * medication_cost +
  (1 - therapy_coverage) * therapy_cost

theorem total_out_of_pocket_is_190_50 :
  total_out_of_pocket_cost = 190.50 := by
  sorry

end NUMINAMATH_CALUDE_total_out_of_pocket_is_190_50_l1118_111803


namespace NUMINAMATH_CALUDE_ticket_revenue_calculation_l1118_111800

/-- Calculates the total revenue from ticket sales given the following conditions:
  * Child ticket cost: $6
  * Adult ticket cost: $9
  * Total tickets sold: 225
  * Number of adult tickets: 175
-/
theorem ticket_revenue_calculation (child_cost adult_cost total_tickets adult_tickets : ℕ) 
  (h1 : child_cost = 6)
  (h2 : adult_cost = 9)
  (h3 : total_tickets = 225)
  (h4 : adult_tickets = 175) :
  child_cost * (total_tickets - adult_tickets) + adult_cost * adult_tickets = 1875 :=
by sorry

end NUMINAMATH_CALUDE_ticket_revenue_calculation_l1118_111800


namespace NUMINAMATH_CALUDE_similar_right_triangles_leg_length_l1118_111843

/-- Given two similar right triangles, where one triangle has legs of length 12 and 9,
    and the other triangle has one leg of length 6, prove that the length of the other
    leg in the second triangle is 4.5. -/
theorem similar_right_triangles_leg_length
  (a b c d : ℝ)
  (h1 : a = 12)
  (h2 : b = 9)
  (h3 : c = 6)
  (h4 : a / b = c / d)
  : d = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_similar_right_triangles_leg_length_l1118_111843


namespace NUMINAMATH_CALUDE_sqrt_sum_squared_l1118_111886

theorem sqrt_sum_squared (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) →
  ((10 + x) * (30 - x) = 144) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_squared_l1118_111886


namespace NUMINAMATH_CALUDE_product_of_square_roots_l1118_111849

theorem product_of_square_roots (q : ℝ) (h : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (8 * q^3) * Real.sqrt (12 * q^5) = 12 * q^4 * Real.sqrt (10 * q) :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l1118_111849


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l1118_111890

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 24) 
  (h2 : Nat.gcd a b = 8) : 
  a * b = 192 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l1118_111890


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1118_111816

/-- Given a quadratic inequality x^2 + ax + b < 0 with solution set (-1, 4), prove that ab = 12 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ -1 < x ∧ x < 4) → a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1118_111816


namespace NUMINAMATH_CALUDE_steve_total_cost_l1118_111881

/-- The price Mike paid for the DVD at the store -/
def store_price : ℝ := 5

/-- The price Steve paid for the DVD online -/
def online_price : ℝ := 2 * store_price

/-- The shipping cost as a percentage of the online price -/
def shipping_rate : ℝ := 0.8

/-- The total amount Steve paid for the DVD -/
def total_cost : ℝ := online_price + shipping_rate * online_price

/-- Theorem stating that the total cost for Steve is $18 -/
theorem steve_total_cost : total_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_steve_total_cost_l1118_111881


namespace NUMINAMATH_CALUDE_simplify_fraction_l1118_111864

theorem simplify_fraction : 3 * (11 / 4) * (16 / -55) = -12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1118_111864


namespace NUMINAMATH_CALUDE_equation_solution_l1118_111811

theorem equation_solution (a : ℝ) (ha : a < 0) :
  ∃! x : ℝ, x * |x| + |x| - x - a = 0 ∧ x = -1 - Real.sqrt (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1118_111811


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1118_111873

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {2, 3, 5}

-- Define set B
def B : Set ℕ := {1, 2}

-- Theorem statement
theorem complement_intersection_theorem :
  (I \ B) ∩ A = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1118_111873


namespace NUMINAMATH_CALUDE_inverse_36_mod_47_l1118_111853

theorem inverse_36_mod_47 (h : (11⁻¹ : ZMod 47) = 43) : (36⁻¹ : ZMod 47) = 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_36_mod_47_l1118_111853


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l1118_111842

def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 3

theorem min_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 2 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) 2 → f y ≥ f x) ∧
  f x = -37 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l1118_111842


namespace NUMINAMATH_CALUDE_number_of_different_products_l1118_111857

def set_a : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}
def set_b : Finset ℕ := {2, 4, 6, 19, 21, 24, 27, 31, 35}

theorem number_of_different_products : 
  (Finset.card (set_a.powersetCard 2) * Finset.card set_b) = 405 := by
  sorry

end NUMINAMATH_CALUDE_number_of_different_products_l1118_111857


namespace NUMINAMATH_CALUDE_poles_not_moved_l1118_111828

theorem poles_not_moved (total_distance : ℕ) (original_spacing : ℕ) (new_spacing : ℕ) : 
  total_distance = 2340 ∧ 
  original_spacing = 45 ∧ 
  new_spacing = 60 → 
  (total_distance / (Nat.lcm original_spacing new_spacing)) - 1 = 12 := by
sorry

end NUMINAMATH_CALUDE_poles_not_moved_l1118_111828


namespace NUMINAMATH_CALUDE_maze_paths_count_l1118_111860

/-- Represents a junction in the maze --/
structure Junction where
  choices : Nat  -- Number of possible directions at this junction

/-- Represents the maze structure --/
structure Maze where
  entrance_choices : Nat  -- Number of choices at the entrance
  x_junctions : Nat       -- Number of x junctions
  dot_junctions : Nat     -- Number of dot junctions per x junction

/-- Calculates the number of paths through the maze --/
def count_paths (m : Maze) : Nat :=
  m.entrance_choices * m.x_junctions * (2 ^ m.dot_junctions)

/-- Theorem stating that the number of paths in the given maze is 16 --/
theorem maze_paths_count :
  ∃ (m : Maze), count_paths m = 16 :=
sorry

end NUMINAMATH_CALUDE_maze_paths_count_l1118_111860


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1118_111897

/-- Given two digits X and Y in base d > 8, if XY_d + XX_d = 234_d, then X_d - Y_d = -2_d. -/
theorem digit_difference_in_base_d (d : ℕ) (X Y : ℕ) (h_d : d > 8) 
  (h_digits : X < d ∧ Y < d) 
  (h_sum : X * d + Y + X * d + X = 2 * d * d + 3 * d + 4) :
  X - Y = d - 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1118_111897


namespace NUMINAMATH_CALUDE_min_ratio_partition_l1118_111823

def S : Finset ℕ := Finset.range 10

theorem min_ratio_partition (p₁ p₂ : ℕ) 
  (h_partition : ∃ (A B : Finset ℕ), A ∪ B = S ∧ A ∩ B = ∅ ∧ 
    p₁ = A.prod id ∧ p₂ = B.prod id)
  (h_divisible : p₁ % p₂ = 0) :
  p₁ / p₂ ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_min_ratio_partition_l1118_111823


namespace NUMINAMATH_CALUDE_multiples_of_six_or_nine_l1118_111882

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_of_six_or_nine (n : ℕ) (h : n = 201) : 
  (count_multiples n 6 + count_multiples n 9) - count_multiples n (lcm 6 9) = 33 :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_six_or_nine_l1118_111882


namespace NUMINAMATH_CALUDE_no_common_real_root_l1118_111831

theorem no_common_real_root (a b : ℚ) : ¬∃ (r : ℝ), r^5 - r - 1 = 0 ∧ r^2 + a*r + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_real_root_l1118_111831


namespace NUMINAMATH_CALUDE_complex_cube_theorem_l1118_111862

theorem complex_cube_theorem (z : ℂ) (h1 : Complex.abs (z - 2) = 2) (h2 : Complex.abs z = 2) : 
  z^3 = -8 := by sorry

end NUMINAMATH_CALUDE_complex_cube_theorem_l1118_111862


namespace NUMINAMATH_CALUDE_max_identical_papers_l1118_111858

def heart_stickers : ℕ := 240
def star_stickers : ℕ := 162
def smiley_stickers : ℕ := 90
def sun_stickers : ℕ := 54

def ratio_heart_to_smiley (n : ℕ) : Prop :=
  2 * (n * smiley_stickers) = n * heart_stickers

def ratio_star_to_sun (n : ℕ) : Prop :=
  3 * (n * sun_stickers) = n * star_stickers

def all_stickers_used (n : ℕ) : Prop :=
  n * (heart_stickers / n + star_stickers / n + smiley_stickers / n + sun_stickers / n) =
    heart_stickers + star_stickers + smiley_stickers + sun_stickers

theorem max_identical_papers : 
  ∃ (n : ℕ), n = 18 ∧ 
    ratio_heart_to_smiley n ∧ 
    ratio_star_to_sun n ∧ 
    all_stickers_used n ∧ 
    ∀ (m : ℕ), m > n → 
      ¬(ratio_heart_to_smiley m ∧ ratio_star_to_sun m ∧ all_stickers_used m) :=
by sorry

end NUMINAMATH_CALUDE_max_identical_papers_l1118_111858


namespace NUMINAMATH_CALUDE_melissa_total_score_l1118_111852

/-- Given a player who scores the same number of points in each game,
    calculate their total score across multiple games. -/
def totalPoints (pointsPerGame : ℕ) (numGames : ℕ) : ℕ :=
  pointsPerGame * numGames

/-- Theorem: A player scoring 120 points per game for 10 games
    will have a total score of 1200 points. -/
theorem melissa_total_score :
  totalPoints 120 10 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_melissa_total_score_l1118_111852


namespace NUMINAMATH_CALUDE_dragon_jewels_l1118_111837

theorem dragon_jewels (x : ℕ) (h1 : x / 3 = 6) : x + 6 = 24 := by
  sorry

#check dragon_jewels

end NUMINAMATH_CALUDE_dragon_jewels_l1118_111837


namespace NUMINAMATH_CALUDE_doubled_side_cube_weight_l1118_111829

/-- Represents the weight of a cube given its side length -/
def cube_weight (side_length : ℝ) : ℝ := sorry

theorem doubled_side_cube_weight (original_side : ℝ) :
  cube_weight original_side = 6 →
  cube_weight (2 * original_side) = 48 := by
  sorry

end NUMINAMATH_CALUDE_doubled_side_cube_weight_l1118_111829


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l1118_111868

/-- Proves that the average speed of a round trip is 34 mph, given that:
    1. The speed from A to B is 51 mph
    2. The return trip from B to A takes twice as long -/
theorem round_trip_average_speed : ∀ (distance : ℝ) (time : ℝ),
  distance > 0 → time > 0 →
  distance = 51 * time →
  (2 * distance) / (3 * time) = 34 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l1118_111868


namespace NUMINAMATH_CALUDE_original_number_proof_l1118_111898

theorem original_number_proof (x : ℝ) : ((x - 3) / 6) * 12 = 8 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1118_111898


namespace NUMINAMATH_CALUDE_factorization_problem_l1118_111817

theorem factorization_problem (C D : ℤ) :
  (∀ y : ℝ, 15 * y^2 - 76 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_l1118_111817


namespace NUMINAMATH_CALUDE_point_inside_circle_range_l1118_111814

/-- A point (x, y) is inside a circle if the left side of the circle's equation is less than the right side -/
def is_inside_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*y - 4 < 0

/-- The theorem stating the range of a for which the point (a+1, a-1) is inside the given circle -/
theorem point_inside_circle_range (a : ℝ) :
  is_inside_circle (a+1) (a-1) a ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_range_l1118_111814


namespace NUMINAMATH_CALUDE_counterexample_exists_l1118_111845

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Statement of the theorem
theorem counterexample_exists : ∃ n : ℕ, 
  (sumOfDigits n % 9 = 0) ∧ (n % 9 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1118_111845


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l1118_111871

theorem divisibility_by_seven (k : ℕ) : 
  (∀ n : ℕ, n > 0 → ∃ q : ℤ, (3^(6*n - 1) - k * 2^(3*n - 2) + 1 : ℤ) = 7 * q) ↔ 
  (∃ m : ℤ, k = 7 * m + 3) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l1118_111871


namespace NUMINAMATH_CALUDE_semicircle_area_shaded_area_proof_l1118_111830

/-- The area of semicircles lined up along a line -/
theorem semicircle_area (diameter : Real) (length : Real) : 
  diameter > 0 → length > 0 → 
  (length / diameter) * (π * diameter^2 / 8) = 3 * π * length / 2 := by
  sorry

/-- The specific case for the given problem -/
theorem shaded_area_proof :
  let diameter : Real := 4
  let length : Real := 24  -- 2 feet in inches
  (length / diameter) * (π * diameter^2 / 8) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_shaded_area_proof_l1118_111830


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_fiftyfive_l1118_111822

theorem largest_multiple_of_seven_below_negative_fiftyfive :
  ∀ n : ℤ, n % 7 = 0 ∧ n < -55 → n ≤ -56 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_fiftyfive_l1118_111822


namespace NUMINAMATH_CALUDE_ellipse_problem_l1118_111856

-- Define the circles and curve C
def F₁ (r : ℝ) (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = r^2
def F₂ (r : ℝ) (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = (4 - r)^2
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point M
def M : ℝ × ℝ := (0, 1)

-- Define the orthogonality condition for points A and B
def orthogonal (A B : ℝ × ℝ) : Prop :=
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0

-- Theorem statement
theorem ellipse_problem (r : ℝ) (h : 0 < r ∧ r < 4) :
  -- 1. Equation of curve C
  (∀ x y : ℝ, (∃ r', F₁ r' x y ∧ F₂ r' x y) ↔ C x y) ∧
  -- 2. Line AB passes through fixed point
  (∀ A B : ℝ × ℝ, C A.1 A.2 → C B.1 B.2 → A ≠ B → orthogonal A B →
    ∃ t : ℝ, A.1 + t * (B.1 - A.1) = 0 ∧ A.2 + t * (B.2 - A.2) = -3/5) ∧
  -- 3. Maximum area of triangle ABM
  (∀ A B : ℝ × ℝ, C A.1 A.2 → C B.1 B.2 → A ≠ B → orthogonal A B →
    abs ((A.1 - M.1) * (B.2 - M.2) - (A.2 - M.2) * (B.1 - M.1)) / 2 ≤ 64/25) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_problem_l1118_111856


namespace NUMINAMATH_CALUDE_carter_has_152_cards_l1118_111809

/-- The number of baseball cards Marcus has -/
def marcus_cards : ℕ := 210

/-- The difference between Marcus's and Carter's cards -/
def marcus_carter_diff : ℕ := 58

/-- The difference between Carter's and Jenny's cards -/
def carter_jenny_diff : ℕ := 35

/-- Carter's number of baseball cards -/
def carter_cards : ℕ := marcus_cards - marcus_carter_diff

theorem carter_has_152_cards : carter_cards = 152 := by
  sorry

end NUMINAMATH_CALUDE_carter_has_152_cards_l1118_111809


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l1118_111841

-- Define the original number
def original_number : ℝ := 0.0000084

-- Define the scientific notation components
def significand : ℝ := 8.4
def exponent : ℤ := -6

-- Theorem statement
theorem scientific_notation_proof :
  original_number = significand * (10 : ℝ) ^ exponent :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l1118_111841


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1118_111847

theorem two_numbers_difference (x y : ℕ) : 
  x ∈ Finset.range 38 →
  y ∈ Finset.range 38 →
  x ≠ y →
  (Finset.sum (Finset.range 38) id) - x - y = x * y + 1 →
  y - x = 20 :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1118_111847


namespace NUMINAMATH_CALUDE_helen_total_cookies_l1118_111838

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 435

/-- The number of cookies Helen baked this morning -/
def cookies_this_morning : ℕ := 139

/-- The total number of cookies Helen baked -/
def total_cookies : ℕ := cookies_yesterday + cookies_this_morning

/-- Theorem stating that the total number of cookies Helen baked is 574 -/
theorem helen_total_cookies : total_cookies = 574 := by
  sorry

end NUMINAMATH_CALUDE_helen_total_cookies_l1118_111838


namespace NUMINAMATH_CALUDE_even_function_tangent_slope_l1118_111865

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x > 0 then a * x^2 / (x + 1) else a * x^2 / (-x + 1)

theorem even_function_tangent_slope (a : ℝ) :
  (∀ x, f a x = f a (-x)) →
  (∀ x > 0, f a x = a * x^2 / (x + 1)) →
  (deriv (f a)) (-1) = -1 →
  a = 4/3 := by sorry

end NUMINAMATH_CALUDE_even_function_tangent_slope_l1118_111865


namespace NUMINAMATH_CALUDE_g_of_three_equals_fourteen_l1118_111827

-- Define the function g
def g : ℝ → ℝ := fun x ↦ 2 * x + 4

-- State the theorem
theorem g_of_three_equals_fourteen : g 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_equals_fourteen_l1118_111827


namespace NUMINAMATH_CALUDE_two_black_cards_selection_l1118_111885

/-- The number of cards in each suit of a standard deck -/
def cards_per_suit : ℕ := 13

/-- The number of black suits in a standard deck -/
def black_suits : ℕ := 2

/-- The total number of black cards in a standard deck -/
def total_black_cards : ℕ := black_suits * cards_per_suit

/-- The number of ways to select two different black cards from a standard deck, where order matters -/
def ways_to_select_two_black_cards : ℕ := total_black_cards * (total_black_cards - 1)

theorem two_black_cards_selection :
  ways_to_select_two_black_cards = 650 := by
  sorry

end NUMINAMATH_CALUDE_two_black_cards_selection_l1118_111885


namespace NUMINAMATH_CALUDE_coeff_comparison_l1118_111824

open Polynomial

/-- The coefficient of x^20 in (1 + x^2 - x^3)^1000 is greater than
    the coefficient of x^20 in (1 - x^2 + x^3)^1000 --/
theorem coeff_comparison (x : ℝ) : 
  (coeff ((1 + X^2 - X^3 : ℝ[X])^1000) 20) > 
  (coeff ((1 - X^2 + X^3 : ℝ[X])^1000) 20) := by
  sorry

end NUMINAMATH_CALUDE_coeff_comparison_l1118_111824


namespace NUMINAMATH_CALUDE_guitar_price_theorem_l1118_111806

theorem guitar_price_theorem (hendricks_price : ℝ) (discount_percentage : ℝ) (gerald_price : ℝ) : 
  hendricks_price = 200 →
  discount_percentage = 20 →
  hendricks_price = gerald_price * (1 - discount_percentage / 100) →
  gerald_price = 250 :=
by sorry

end NUMINAMATH_CALUDE_guitar_price_theorem_l1118_111806


namespace NUMINAMATH_CALUDE_election_theorem_l1118_111878

def total_candidates : ℕ := 20
def past_officers : ℕ := 8
def positions_available : ℕ := 4

def elections_with_at_least_two_past_officers : ℕ :=
  Nat.choose past_officers 2 * Nat.choose (total_candidates - past_officers) 2 +
  Nat.choose past_officers 3 * Nat.choose (total_candidates - past_officers) 1 +
  Nat.choose past_officers 4 * Nat.choose (total_candidates - past_officers) 0

theorem election_theorem :
  elections_with_at_least_two_past_officers = 2590 :=
by sorry

end NUMINAMATH_CALUDE_election_theorem_l1118_111878


namespace NUMINAMATH_CALUDE_earth_surface_area_scientific_notation_l1118_111832

/-- The surface area of the Earth in square kilometers. -/
def earth_surface_area : ℝ := 510000000

/-- The scientific notation representation of the Earth's surface area. -/
def earth_surface_area_scientific : ℝ := 5.1 * (10 ^ 8)

/-- Theorem stating that the Earth's surface area is correctly represented in scientific notation. -/
theorem earth_surface_area_scientific_notation : 
  earth_surface_area = earth_surface_area_scientific := by sorry

end NUMINAMATH_CALUDE_earth_surface_area_scientific_notation_l1118_111832


namespace NUMINAMATH_CALUDE_eighteen_times_thirtysix_minus_twentyseven_times_eighteen_l1118_111813

theorem eighteen_times_thirtysix_minus_twentyseven_times_eighteen : 
  18 * 36 - 27 * 18 = 162 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_times_thirtysix_minus_twentyseven_times_eighteen_l1118_111813


namespace NUMINAMATH_CALUDE_quadratic_origin_condition_l1118_111867

/-- A quadratic function passing through the origin -/
def passes_through_origin (m : ℝ) : Prop :=
  ∃ x y : ℝ, y = m * x^2 + x + m * (m - 2) ∧ x = 0 ∧ y = 0

/-- The theorem stating the conditions for the quadratic function to pass through the origin -/
theorem quadratic_origin_condition :
  ∀ m : ℝ, passes_through_origin m ↔ m = 2 ∨ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_origin_condition_l1118_111867


namespace NUMINAMATH_CALUDE_fourth_arrangement_follows_pattern_l1118_111812

/-- Represents the four possible positions in a 2x2 grid --/
inductive Position
| topLeft
| topRight
| bottomLeft
| bottomRight

/-- Represents the orientation of the line segment --/
inductive LineOrientation
| horizontal
| vertical

/-- Represents a geometric shape --/
inductive Shape
| circle
| triangle
| square
| line

/-- Represents the arrangement of shapes in a square --/
structure Arrangement where
  circlePos : Position
  trianglePos : Position
  squarePos : Position
  lineOrientation : LineOrientation

/-- The sequence of arrangements in the first three squares --/
def firstThreeArrangements : List Arrangement := [
  { circlePos := Position.topLeft, trianglePos := Position.bottomLeft, 
    squarePos := Position.topRight, lineOrientation := LineOrientation.horizontal },
  { circlePos := Position.bottomLeft, trianglePos := Position.bottomRight, 
    squarePos := Position.topRight, lineOrientation := LineOrientation.vertical },
  { circlePos := Position.bottomRight, trianglePos := Position.topRight, 
    squarePos := Position.bottomLeft, lineOrientation := LineOrientation.horizontal }
]

/-- The predicted arrangement for the fourth square --/
def predictedFourthArrangement : Arrangement :=
  { circlePos := Position.topRight, trianglePos := Position.topLeft, 
    squarePos := Position.bottomLeft, lineOrientation := LineOrientation.vertical }

/-- Theorem stating that the predicted fourth arrangement follows the pattern --/
theorem fourth_arrangement_follows_pattern :
  predictedFourthArrangement = 
    { circlePos := Position.topRight, trianglePos := Position.topLeft, 
      squarePos := Position.bottomLeft, lineOrientation := LineOrientation.vertical } :=
by sorry

end NUMINAMATH_CALUDE_fourth_arrangement_follows_pattern_l1118_111812


namespace NUMINAMATH_CALUDE_negative_three_less_than_negative_sqrt_eight_l1118_111895

theorem negative_three_less_than_negative_sqrt_eight : -3 < -Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_less_than_negative_sqrt_eight_l1118_111895


namespace NUMINAMATH_CALUDE_stating_max_triangulation_is_19_l1118_111846

/-- Represents a triangulation of a triangle -/
structure Triangulation where
  num_triangles : ℕ
  num_segments_per_vertex : ℕ
  vertices_dont_split_sides : Bool

/-- The maximum number of triangles in a valid triangulation -/
def max_triangles : ℕ := 19

/-- Checks if a triangulation is valid according to the problem conditions -/
def is_valid_triangulation (t : Triangulation) : Prop :=
  t.num_segments_per_vertex > 1 ∧ t.vertices_dont_split_sides

/-- 
Theorem stating that the maximum number of triangles in a valid triangulation is 19
-/
theorem max_triangulation_is_19 :
  ∀ t : Triangulation, is_valid_triangulation t → t.num_triangles ≤ max_triangles :=
by sorry

end NUMINAMATH_CALUDE_stating_max_triangulation_is_19_l1118_111846


namespace NUMINAMATH_CALUDE_pipe_filling_time_l1118_111861

theorem pipe_filling_time (fill_rate_A fill_rate_B fill_rate_C : ℝ) 
  (h1 : fill_rate_A + fill_rate_B + fill_rate_C = 1 / 5)
  (h2 : fill_rate_B = 2 * fill_rate_A)
  (h3 : fill_rate_C = 2 * fill_rate_B) :
  1 / fill_rate_A = 35 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l1118_111861


namespace NUMINAMATH_CALUDE_add_base_seven_example_l1118_111836

/-- Represents a number in base 7 --/
def BaseSevenNum (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Addition in base 7 --/
def addBaseSeven (a b : List Nat) : List Nat :=
  sorry

theorem add_base_seven_example :
  addBaseSeven [2, 1] [2, 5, 4] = [5, 0, 5] :=
by sorry

end NUMINAMATH_CALUDE_add_base_seven_example_l1118_111836


namespace NUMINAMATH_CALUDE_problem_1_l1118_111893

theorem problem_1 : |-2| - 8 / (-2) / (-1/2) = -6 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1118_111893


namespace NUMINAMATH_CALUDE_platform_length_calculation_l1118_111825

/-- Calculates the length of a platform given train specifications and crossing time -/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmph = 72 →
  crossing_time = 25 →
  (train_speed_kmph * 1000 / 3600 * crossing_time) - train_length = 380 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l1118_111825


namespace NUMINAMATH_CALUDE_x_eleven_percent_greater_than_80_l1118_111840

/-- If x is 11 percent greater than 80, then x equals 88.8 -/
theorem x_eleven_percent_greater_than_80 (x : ℝ) :
  x = 80 * (1 + 11 / 100) → x = 88.8 := by
  sorry

end NUMINAMATH_CALUDE_x_eleven_percent_greater_than_80_l1118_111840
