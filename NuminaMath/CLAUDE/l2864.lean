import Mathlib

namespace NUMINAMATH_CALUDE_largest_initial_number_l2864_286482

theorem largest_initial_number :
  ∃ (a b c d e : ℕ), 
    189 + a + b + c + d + e = 200 ∧
    a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
    ¬(189 ∣ a) ∧ ¬(189 ∣ b) ∧ ¬(189 ∣ c) ∧ ¬(189 ∣ d) ∧ ¬(189 ∣ e) ∧
    ∀ (n : ℕ), n > 189 → 
      ¬∃ (x y z w v : ℕ),
        n + x + y + z + w + v = 200 ∧
        x ≥ 2 ∧ y ≥ 2 ∧ z ≥ 2 ∧ w ≥ 2 ∧ v ≥ 2 ∧
        ¬(n ∣ x) ∧ ¬(n ∣ y) ∧ ¬(n ∣ z) ∧ ¬(n ∣ w) ∧ ¬(n ∣ v) :=
by sorry

end NUMINAMATH_CALUDE_largest_initial_number_l2864_286482


namespace NUMINAMATH_CALUDE_daily_medicine_dose_l2864_286472

theorem daily_medicine_dose (total_medicine : ℝ) (daily_fraction : ℝ) :
  total_medicine = 426 →
  daily_fraction = 0.06 →
  total_medicine * daily_fraction = 25.56 := by
  sorry

end NUMINAMATH_CALUDE_daily_medicine_dose_l2864_286472


namespace NUMINAMATH_CALUDE_adams_shopping_cost_l2864_286478

/-- Calculates the total cost of Adam's shopping given the specified conditions --/
def calculate_total_cost (sandwich_price : ℚ) (sandwich_count : ℕ) 
                         (chip_price : ℚ) (chip_count : ℕ) 
                         (water_price : ℚ) (water_count : ℕ) : ℚ :=
  let sandwich_cost := (sandwich_count - 1) * sandwich_price
  let chip_cost := chip_count * chip_price * (1 - 0.2)
  let water_cost := water_count * water_price * 1.05
  sandwich_cost + chip_cost + water_cost

/-- Theorem stating that Adam's total shopping cost is $31.75 --/
theorem adams_shopping_cost : 
  calculate_total_cost 4 5 3.5 3 1.75 4 = 31.75 := by
  sorry

end NUMINAMATH_CALUDE_adams_shopping_cost_l2864_286478


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_equivalence_l2864_286479

/-- The quadratic function in standard form -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The quadratic function in vertex form -/
def g (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating that f and g are equivalent -/
theorem quadratic_vertex_form_equivalence :
  ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_equivalence_l2864_286479


namespace NUMINAMATH_CALUDE_beaver_change_l2864_286499

theorem beaver_change (initial_beavers initial_chipmunks chipmunk_decrease total_animals : ℕ) :
  initial_beavers = 20 →
  initial_chipmunks = 40 →
  chipmunk_decrease = 10 →
  total_animals = 130 →
  (total_animals - (initial_beavers + initial_chipmunks)) - (initial_chipmunks - chipmunk_decrease) - initial_beavers = 20 := by
  sorry

end NUMINAMATH_CALUDE_beaver_change_l2864_286499


namespace NUMINAMATH_CALUDE_equation_solutions_l2864_286442

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5))
  ∀ x : ℝ, f x = 1 / 12 ↔ x = 5 + Real.sqrt 19 ∨ x = 5 - Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2864_286442


namespace NUMINAMATH_CALUDE_cubic_root_sum_log_l2864_286419

theorem cubic_root_sum_log (a b : ℝ) : 
  (∃ r s t : ℝ, r > 0 ∧ s > 0 ∧ t > 0 ∧ 
   r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
   16 * r^3 + 7 * a * r^2 + 6 * b * r + 2 * a = 0 ∧
   16 * s^3 + 7 * a * s^2 + 6 * b * s + 2 * a = 0 ∧
   16 * t^3 + 7 * a * t^2 + 6 * b * t + 2 * a = 0 ∧
   Real.log r / Real.log 4 + Real.log s / Real.log 4 + Real.log t / Real.log 4 = 3) →
  a = -512 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_log_l2864_286419


namespace NUMINAMATH_CALUDE_bill_total_l2864_286439

/-- Proves that if three people divide a bill evenly and each pays $33, then the total bill is $99. -/
theorem bill_total (people : Fin 3 → ℕ) (h : ∀ i, people i = 33) : 
  (Finset.univ.sum people) = 99 := by
  sorry

end NUMINAMATH_CALUDE_bill_total_l2864_286439


namespace NUMINAMATH_CALUDE_cosine_range_in_triangle_l2864_286447

theorem cosine_range_in_triangle (A B C : Real) (h : 1 / Real.tan B + 1 / Real.tan C = 1 / Real.tan A) :
  2/3 ≤ Real.cos A ∧ Real.cos A < 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_range_in_triangle_l2864_286447


namespace NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l2864_286474

/-- The total cost of Amanda's kitchen upgrade --/
def kitchen_upgrade_cost (cabinet_knobs : ℕ) (knob_price : ℚ) (drawer_pulls : ℕ) (pull_price : ℚ) : ℚ :=
  (cabinet_knobs : ℚ) * knob_price + (drawer_pulls : ℚ) * pull_price

/-- Proof that Amanda's kitchen upgrade costs $77.00 --/
theorem amanda_kitchen_upgrade_cost :
  kitchen_upgrade_cost 18 (5/2) 8 4 = 77 := by
  sorry

end NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l2864_286474


namespace NUMINAMATH_CALUDE_no_seven_digit_number_divisible_by_another_l2864_286466

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 : ℕ),
    d1 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d2 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d3 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d4 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d5 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d6 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d7 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧
    d6 ≠ d7 ∧
    n = d1 * 1000000 + d2 * 100000 + d3 * 10000 + d4 * 1000 + d5 * 100 + d6 * 10 + d7

theorem no_seven_digit_number_divisible_by_another :
  ∀ a b : ℕ, is_valid_number a → is_valid_number b → a ≠ b → ¬(a % b = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_seven_digit_number_divisible_by_another_l2864_286466


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2864_286471

/-- The polar equation ρ = 5 sin θ represents a circle in Cartesian coordinates. -/
theorem polar_to_cartesian_circle :
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), ρ = 5 * Real.sin θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2864_286471


namespace NUMINAMATH_CALUDE_joe_market_spend_l2864_286409

/-- Calculates the total cost of Joe's market purchases -/
def market_total_cost (orange_price : ℚ) (juice_price : ℚ) (honey_price : ℚ) (plant_pair_price : ℚ)
  (orange_count : ℕ) (juice_count : ℕ) (honey_count : ℕ) (plant_count : ℕ) : ℚ :=
  orange_price * orange_count +
  juice_price * juice_count +
  honey_price * honey_count +
  plant_pair_price * (plant_count / 2)

/-- Theorem stating that Joe's total market spend is $68 -/
theorem joe_market_spend :
  market_total_cost 4.5 0.5 5 18 3 7 3 4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_joe_market_spend_l2864_286409


namespace NUMINAMATH_CALUDE_sqrt_25_times_sqrt_25_l2864_286498

theorem sqrt_25_times_sqrt_25 : Real.sqrt (25 * Real.sqrt 25) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_25_times_sqrt_25_l2864_286498


namespace NUMINAMATH_CALUDE_sum_of_parts_l2864_286480

theorem sum_of_parts (x y : ℝ) : 
  x + y = 52 → 
  y = 30.333333333333332 → 
  10 * x + 22 * y = 884 := by
sorry

end NUMINAMATH_CALUDE_sum_of_parts_l2864_286480


namespace NUMINAMATH_CALUDE_min_value_sum_l2864_286443

theorem min_value_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 = Real.sqrt (a * b)) :
  ∀ x y, x > 0 → y > 0 → 2 = Real.sqrt (x * y) → a + 4 * b ≤ x + 4 * y :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l2864_286443


namespace NUMINAMATH_CALUDE_sum_of_squares_ratio_l2864_286453

theorem sum_of_squares_ratio (a b c : ℚ) : 
  a + b + c = 14 → 
  b = 2 * a → 
  c = 3 * a → 
  a^2 + b^2 + c^2 = 686/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_ratio_l2864_286453


namespace NUMINAMATH_CALUDE_sin_squared_sum_less_than_one_l2864_286421

theorem sin_squared_sum_less_than_one (x y z : ℝ) 
  (h1 : Real.tan x + Real.tan y + Real.tan z = 2)
  (h2 : 0 < x ∧ x < Real.pi / 2)
  (h3 : 0 < y ∧ y < Real.pi / 2)
  (h4 : 0 < z ∧ z < Real.pi / 2) :
  Real.sin x ^ 2 + Real.sin y ^ 2 + Real.sin z ^ 2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_less_than_one_l2864_286421


namespace NUMINAMATH_CALUDE_range_of_m_l2864_286436

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (m + 2)*x - 1 < (m + 2)*y - 1

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (m ≤ -2 ∨ m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2864_286436


namespace NUMINAMATH_CALUDE_intersection_implies_k_value_l2864_286444

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The x-coordinate of the intersection point -/
def x_intersect : ℝ := 2

/-- The y-coordinate of the intersection point -/
def y_intersect : ℝ := 13

/-- Line p with equation y = 5x + 3 -/
def p : Line := { slope := 5, intercept := 3 }

/-- Line q with equation y = kx + 7, where k is to be determined -/
def q (k : ℝ) : Line := { slope := k, intercept := 7 }

/-- Theorem stating that if lines p and q intersect at (2, 13), then k = 3 -/
theorem intersection_implies_k_value :
  y_intersect = p.slope * x_intersect + p.intercept ∧
  y_intersect = (q k).slope * x_intersect + (q k).intercept →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_k_value_l2864_286444


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identity_l2864_286403

theorem triangle_trigonometric_identity (A B C : Real) 
  (h : A + B + C = Real.pi) : 
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 - 
  2 * Real.cos A * Real.cos B * Real.cos C = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identity_l2864_286403


namespace NUMINAMATH_CALUDE_fraction_power_equality_l2864_286437

theorem fraction_power_equality : (72000 ^ 4) / (24000 ^ 4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l2864_286437


namespace NUMINAMATH_CALUDE_triangle_height_proof_l2864_286411

/-- Triangle ABC with vertices A(-2,10), B(2,0), and C(10,0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A vertical line intersecting AC at R and BC at S -/
structure IntersectingLine :=
  (R : ℝ × ℝ)
  (S : ℝ × ℝ)

/-- The problem statement -/
theorem triangle_height_proof 
  (ABC : Triangle)
  (RS : IntersectingLine)
  (h1 : ABC.A = (-2, 10))
  (h2 : ABC.B = (2, 0))
  (h3 : ABC.C = (10, 0))
  (h4 : RS.R.1 = RS.S.1)  -- R and S have the same x-coordinate (vertical line)
  (h5 : RS.S.2 = 0)  -- S lies on BC (y-coordinate is 0)
  (h6 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ RS.R = (1 - t) • ABC.A + t • ABC.C)  -- R lies on AC
  (h7 : (1/2) * |RS.R.2| * 8 = 24)  -- Area of RSC is 24
  : RS.R.2 = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_height_proof_l2864_286411


namespace NUMINAMATH_CALUDE_next_term_is_2500x4_l2864_286492

def geometric_sequence (x : ℝ) : ℕ → ℝ
  | 0 => 4
  | 1 => 20 * x
  | 2 => 100 * x^2
  | 3 => 500 * x^3
  | (n + 4) => geometric_sequence x n * 5 * x

theorem next_term_is_2500x4 (x : ℝ) : geometric_sequence x 4 = 2500 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_next_term_is_2500x4_l2864_286492


namespace NUMINAMATH_CALUDE_base_6_representation_of_1729_base_6_to_decimal_1729_l2864_286454

/-- Converts a natural number to its base-6 representation as a list of digits -/
def toBase6 (n : ℕ) : List ℕ :=
  if n < 6 then [n]
  else (n % 6) :: toBase6 (n / 6)

/-- Converts a list of base-6 digits to a natural number -/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 6 * acc) 0

theorem base_6_representation_of_1729 :
  toBase6 1729 = [1, 0, 0, 0, 2, 1] :=
sorry

theorem base_6_to_decimal_1729 :
  fromBase6 [1, 0, 0, 0, 2, 1] = 1729 :=
sorry

end NUMINAMATH_CALUDE_base_6_representation_of_1729_base_6_to_decimal_1729_l2864_286454


namespace NUMINAMATH_CALUDE_math_class_size_l2864_286489

theorem math_class_size :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 8 = 5 ∧ n % 6 = 1 ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_math_class_size_l2864_286489


namespace NUMINAMATH_CALUDE_ship_passengers_l2864_286494

theorem ship_passengers : ∀ (P : ℕ),
  (P / 12 : ℚ) + (P / 8 : ℚ) + (P / 3 : ℚ) + (P / 6 : ℚ) + 35 = P →
  P = 120 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l2864_286494


namespace NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l2864_286457

theorem no_simultaneous_integer_fractions :
  ¬ ∃ (n : ℤ), (∃ (A B : ℤ), (n - 6) / 15 = A ∧ (n - 5) / 24 = B) := by
sorry

end NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l2864_286457


namespace NUMINAMATH_CALUDE_rabbit_groupings_count_l2864_286483

/-- The number of ways to divide 12 rabbits into specific groups -/
def rabbit_groupings : ℕ :=
  let total_rabbits : ℕ := 12
  let group1_size : ℕ := 4
  let group2_size : ℕ := 6
  let group3_size : ℕ := 2
  let remaining_rabbits : ℕ := total_rabbits - 2  -- BunBun and Thumper are already placed
  Nat.choose remaining_rabbits (group1_size - 1) * Nat.choose (remaining_rabbits - (group1_size - 1)) (group2_size - 1)

/-- Theorem stating the number of ways to divide the rabbits -/
theorem rabbit_groupings_count : rabbit_groupings = 2520 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_groupings_count_l2864_286483


namespace NUMINAMATH_CALUDE_value_of_x_l2864_286432

theorem value_of_x (x y z : ℝ) 
  (h1 : x = (1/2) * y) 
  (h2 : y = (1/4) * z) 
  (h3 : z = 100) : 
  x = 12.5 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l2864_286432


namespace NUMINAMATH_CALUDE_max_distance_complex_l2864_286420

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (max_dist : ℝ), max_dist = 729 + 162 * Real.sqrt 13 ∧
  ∀ (w : ℂ), Complex.abs w = 3 →
    Complex.abs ((2 + 3*Complex.I)*(w^4) - w^6) ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_l2864_286420


namespace NUMINAMATH_CALUDE_merry_go_round_diameter_l2864_286402

/-- The diameter of a circular platform with area 3.14 square yards is 2 yards. -/
theorem merry_go_round_diameter : 
  ∀ (r : ℝ), r > 0 → π * r^2 = 3.14 → 2 * r = 2 := by sorry

end NUMINAMATH_CALUDE_merry_go_round_diameter_l2864_286402


namespace NUMINAMATH_CALUDE_science_marks_calculation_l2864_286415

/-- Calculates the marks scored in science given the total marks and marks in other subjects. -/
def science_marks (total : ℕ) (music : ℕ) (social_studies : ℕ) : ℕ :=
  total - (music + social_studies + music / 2)

/-- Theorem stating that given the specific marks, the science marks must be 70. -/
theorem science_marks_calculation :
  science_marks 275 80 85 = 70 := by
  sorry

end NUMINAMATH_CALUDE_science_marks_calculation_l2864_286415


namespace NUMINAMATH_CALUDE_problem_1_l2864_286433

theorem problem_1 (a b c : ℝ) (h : a * c + b * c + c^2 < 0) : b^2 > 4 * a * c := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2864_286433


namespace NUMINAMATH_CALUDE_circle_M_equation_l2864_286490

/-- A circle M with the following properties:
    1. Tangent to the y-axis
    2. Its center lies on the line y = 1/2x
    3. The chord it cuts on the x-axis is 2√3 long -/
structure CircleM where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_y_axis : abs (center.1) = radius
  center_on_line : center.2 = 1/2 * center.1
  x_axis_chord : 2 * radius = 2 * Real.sqrt 3

/-- The standard equation of circle M is either (x-2)² + (y-1)² = 4 or (x+2)² + (y+1)² = 4 -/
theorem circle_M_equation (M : CircleM) :
  (∀ x y, (x - 2)^2 + (y - 1)^2 = 4) ∨ (∀ x y, (x + 2)^2 + (y + 1)^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_circle_M_equation_l2864_286490


namespace NUMINAMATH_CALUDE_paper_clip_cost_l2864_286458

/-- The cost of one box of paper clips and one package of index cards satisfying given conditions -/
def paper_clip_and_index_card_cost (p i : ℝ) : Prop :=
  15 * p + 7 * i = 55.40 ∧ 12 * p + 10 * i = 61.70

/-- The theorem stating that the cost of one box of paper clips is 1.835 -/
theorem paper_clip_cost : ∃ (p i : ℝ), paper_clip_and_index_card_cost p i ∧ p = 1.835 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_cost_l2864_286458


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2864_286495

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 560) 
  (h2 : a*b + b*c + c*a = 8) : 
  a + b + c = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2864_286495


namespace NUMINAMATH_CALUDE_bathroom_length_proof_l2864_286440

/-- Proves the length of a rectangular bathroom given its width, tile size, and number of tiles needed --/
theorem bathroom_length_proof (width : ℝ) (tile_side : ℝ) (num_tiles : ℕ) (length : ℝ) : 
  width = 6 →
  tile_side = 0.5 →
  num_tiles = 240 →
  width * length = (tile_side * tile_side) * num_tiles →
  length = 10 := by
sorry

end NUMINAMATH_CALUDE_bathroom_length_proof_l2864_286440


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2864_286408

/-- A quadratic function f(x) = x^2 + bx + 3 where b is a real number -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- The theorem stating that if the range of f is [0, +∞) and the solution set of f(x) < c
    is an open interval of length 8, then c = 16 -/
theorem quadratic_function_theorem (b : ℝ) (c : ℝ) :
  (∀ x, f b x ≥ 0) →
  (∃ m, ∀ x, f b x < c ↔ m - 8 < x ∧ x < m) →
  c = 16 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2864_286408


namespace NUMINAMATH_CALUDE_nells_baseball_cards_l2864_286448

/-- Nell's baseball card collection problem -/
theorem nells_baseball_cards 
  (cards_given_to_jeff : ℕ) 
  (cards_left : ℕ) 
  (h1 : cards_given_to_jeff = 28) 
  (h2 : cards_left = 276) : 
  cards_given_to_jeff + cards_left = 304 := by
sorry

end NUMINAMATH_CALUDE_nells_baseball_cards_l2864_286448


namespace NUMINAMATH_CALUDE_square_root_of_square_l2864_286488

theorem square_root_of_square (x : ℝ) (h : x = 36) : Real.sqrt (x^2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_square_l2864_286488


namespace NUMINAMATH_CALUDE_divide_ten_items_between_two_people_l2864_286486

theorem divide_ten_items_between_two_people : 
  Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_divide_ten_items_between_two_people_l2864_286486


namespace NUMINAMATH_CALUDE_two_places_distribution_three_places_distribution_ambulance_distribution_l2864_286465

/-- The number of volunteers --/
def num_volunteers : ℕ := 4

/-- The number of places --/
def num_places : ℕ := 3

/-- The number of ambulances --/
def num_ambulances : ℕ := 20

/-- The number of ways to distribute 4 volunteers to 2 places with 2 volunteers in each place --/
theorem two_places_distribution (n : ℕ) (h : n = num_volunteers) : 
  Nat.choose n 2 = 6 := by sorry

/-- The number of ways to distribute 4 volunteers to 3 places with at least one volunteer in each place --/
theorem three_places_distribution (n m : ℕ) (h1 : n = num_volunteers) (h2 : m = num_places) : 
  6 * Nat.factorial (m - 1) = 36 := by sorry

/-- The number of ways to distribute 20 identical ambulances to 3 places with at least one ambulance in each place --/
theorem ambulance_distribution (a m : ℕ) (h1 : a = num_ambulances) (h2 : m = num_places) : 
  Nat.choose (a - 1) (m - 1) = 171 := by sorry

end NUMINAMATH_CALUDE_two_places_distribution_three_places_distribution_ambulance_distribution_l2864_286465


namespace NUMINAMATH_CALUDE_cost_calculation_l2864_286427

/-- The cost of mangos per kg -/
def mango_cost : ℝ := sorry

/-- The cost of rice per kg -/
def rice_cost : ℝ := sorry

/-- The cost of flour per kg -/
def flour_cost : ℝ := 21

/-- The total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour -/
def total_cost : ℝ := 4 * mango_cost + 3 * rice_cost + 5 * flour_cost

theorem cost_calculation :
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  total_cost = 898.8 := by sorry

end NUMINAMATH_CALUDE_cost_calculation_l2864_286427


namespace NUMINAMATH_CALUDE_two_numbers_sum_difference_product_l2864_286473

theorem two_numbers_sum_difference_product 
  (x y : ℝ) 
  (sum_eq : x + y = 40) 
  (diff_eq : x - y = 16) : 
  x = 28 ∧ y = 12 ∧ x * y = 336 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_difference_product_l2864_286473


namespace NUMINAMATH_CALUDE_problem_solution_l2864_286412

theorem problem_solution : 
  let X := (354 * 28)^2
  let Y := (48 * 14)^2
  (X * 9) / (Y * 2) = 2255688 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2864_286412


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_values_l2864_286467

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, a 1 = 1 ∧ ∀ n ≥ 3, a n = 100 ∧ ∀ k : ℕ, a (k + 1) = a k + d

/-- The set of possible n values -/
def PossibleN : Set ℕ := {4, 10, 12, 34, 100}

/-- The main theorem -/
theorem arithmetic_sequence_n_values (a : ℕ → ℕ) :
  ArithmeticSequence a →
  (∀ n : ℕ, n ∈ PossibleN ↔ (n ≥ 3 ∧ a n = 100)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_values_l2864_286467


namespace NUMINAMATH_CALUDE_max_consecutive_good_proof_l2864_286475

/-- Sum of all positive divisors of n -/
def α (n : ℕ) : ℕ := sorry

/-- A number n is "good" if gcd(n, α(n)) = 1 -/
def is_good (n : ℕ) : Prop := Nat.gcd n (α n) = 1

/-- The maximum number of consecutive good numbers -/
def max_consecutive_good : ℕ := 5

theorem max_consecutive_good_proof :
  ∀ k : ℕ, k > max_consecutive_good →
    ∃ n : ℕ, n ≥ 2 ∧ ∃ i : Fin k, ¬is_good (n + i) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_good_proof_l2864_286475


namespace NUMINAMATH_CALUDE_sqrt_real_implies_x_leq_two_l2864_286496

theorem sqrt_real_implies_x_leq_two (x : ℝ) : (∃ y : ℝ, y * y = 2 - x) → x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_real_implies_x_leq_two_l2864_286496


namespace NUMINAMATH_CALUDE_mothers_day_discount_l2864_286435

theorem mothers_day_discount (original_price : ℝ) : 
  (original_price * 0.9 * 0.96 = 108) → original_price = 125 := by
  sorry

end NUMINAMATH_CALUDE_mothers_day_discount_l2864_286435


namespace NUMINAMATH_CALUDE_marble_selection_combinations_l2864_286431

def total_marbles : ℕ := 15
def special_marbles : ℕ := 6
def marbles_to_choose : ℕ := 5
def special_marbles_to_choose : ℕ := 2

theorem marble_selection_combinations :
  (Nat.choose special_marbles special_marbles_to_choose) *
  (Nat.choose (total_marbles - special_marbles) (marbles_to_choose - special_marbles_to_choose)) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_combinations_l2864_286431


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2864_286429

-- Problem 1
theorem simplify_expression_1 (a : ℝ) : 2*a*(a-3) - a^2 = a^2 - 6*a := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) : (x-1)*(x+2) - x*(x+1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2864_286429


namespace NUMINAMATH_CALUDE_marie_messages_l2864_286451

/-- The number of new messages Marie gets per day -/
def new_messages_per_day : ℕ := 6

/-- The initial number of unread messages -/
def initial_messages : ℕ := 98

/-- The number of messages Marie reads per day -/
def messages_read_per_day : ℕ := 20

/-- The number of days it takes Marie to read all messages -/
def days_to_read_all : ℕ := 7

theorem marie_messages :
  initial_messages + days_to_read_all * new_messages_per_day = 
  days_to_read_all * messages_read_per_day :=
by sorry

end NUMINAMATH_CALUDE_marie_messages_l2864_286451


namespace NUMINAMATH_CALUDE_g_critical_points_l2864_286400

noncomputable def g (x : ℝ) : ℝ :=
  if -3 < x ∧ x ≤ 0 then -x - 3
  else if 0 < x ∧ x ≤ 2 then x - 3
  else if 2 < x ∧ x ≤ 3 then x^2 - 4*x + 6
  else 0  -- Default value for x outside the defined range

def is_critical_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → f y ≤ f x

def is_local_minimum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ δ > 0, ∀ y, |y - x| < δ → f x ≤ f y

theorem g_critical_points :
  is_critical_point g 0 ∧ 
  is_critical_point g 2 ∧
  is_local_minimum g 2 :=
sorry

end NUMINAMATH_CALUDE_g_critical_points_l2864_286400


namespace NUMINAMATH_CALUDE_turtle_initial_coins_l2864_286425

def bridge_crossing (initial_coins : ℕ) : Prop :=
  let after_first_crossing := 3 * initial_coins - 30
  let after_second_crossing := 3 * after_first_crossing - 30
  after_second_crossing = 0

theorem turtle_initial_coins : 
  ∃ (x : ℕ), bridge_crossing x ∧ x = 15 :=
sorry

end NUMINAMATH_CALUDE_turtle_initial_coins_l2864_286425


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2864_286446

theorem min_value_of_expression (x : ℝ) (h : x > 2) : 
  4 / (x - 2) + 4 * x ≥ 16 ∧ ∃ y > 2, 4 / (y - 2) + 4 * y = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2864_286446


namespace NUMINAMATH_CALUDE_company_fund_problem_l2864_286460

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  initial_fund = 60 * n - 10 →
  initial_fund = 50 * n + 120 →
  initial_fund = 770 :=
by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l2864_286460


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2864_286461

/-- The equation of the tangent line to the circle x^2 + y^2 = 5 at the point (2, 1) is 2x + y - 5 = 0 -/
theorem tangent_line_to_circle (x y : ℝ) : 
  (2 : ℝ)^2 + 1^2 = 5 →  -- Point (2, 1) lies on the circle
  (∀ (a b : ℝ), a^2 + b^2 = 5 → (2*a + b = 5 → a = 2 ∧ b = 1)) →  -- (2, 1) is the only point of intersection
  2*x + y - 5 = 0 ↔ (x - 2)*(2) + (y - 1)*(1) = 0  -- Equation of tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2864_286461


namespace NUMINAMATH_CALUDE_tank_filling_ratio_l2864_286434

theorem tank_filling_ratio (tank_capacity : ℝ) (inflow_rate : ℝ) (outflow_rate1 : ℝ) (outflow_rate2 : ℝ) (filling_time : ℝ) :
  tank_capacity = 1 →
  inflow_rate = 0.5 →
  outflow_rate1 = 0.25 →
  outflow_rate2 = 1/6 →
  filling_time = 6 →
  (tank_capacity - (inflow_rate - outflow_rate1 - outflow_rate2) * filling_time) / tank_capacity = 0.5 := by
  sorry

#check tank_filling_ratio

end NUMINAMATH_CALUDE_tank_filling_ratio_l2864_286434


namespace NUMINAMATH_CALUDE_all_terms_are_squares_l2864_286493

/-- Definition of the n-th term of the sequence -/
def sequence_term (n : ℕ) : ℕ :=
  10^(2*n + 1) + 5 * (10^n - 1) * 10^n + 6

/-- Theorem stating that all terms in the sequence are perfect squares -/
theorem all_terms_are_squares :
  ∀ n : ℕ, ∃ k : ℕ, sequence_term n = k^2 :=
by sorry

end NUMINAMATH_CALUDE_all_terms_are_squares_l2864_286493


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2864_286414

theorem deal_or_no_deal_probability (total_boxes : ℕ) (desired_boxes : ℕ) (eliminated_boxes : ℕ) : 
  total_boxes = 30 →
  desired_boxes = 6 →
  eliminated_boxes = 18 →
  (desired_boxes : ℚ) / (total_boxes - eliminated_boxes : ℚ) ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2864_286414


namespace NUMINAMATH_CALUDE_thor_fraction_is_two_ninths_l2864_286481

-- Define the friends
inductive Friend
| Moe
| Loki
| Nick
| Thor
| Ott

-- Define the function that returns the fraction of money given by each friend
def fractionGiven (f : Friend) : ℚ :=
  match f with
  | Friend.Moe => 1/6
  | Friend.Loki => 1/5
  | Friend.Nick => 1/4
  | Friend.Ott => 1/3
  | Friend.Thor => 0

-- Define the amount of money given by each friend
def amountGiven : ℚ := 2

-- Define the total money of the group
def totalMoney : ℚ := (amountGiven / fractionGiven Friend.Moe) +
                      (amountGiven / fractionGiven Friend.Loki) +
                      (amountGiven / fractionGiven Friend.Nick) +
                      (amountGiven / fractionGiven Friend.Ott)

-- Define Thor's share
def thorShare : ℚ := 4 * amountGiven

-- Theorem to prove
theorem thor_fraction_is_two_ninths :
  thorShare / totalMoney = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_thor_fraction_is_two_ninths_l2864_286481


namespace NUMINAMATH_CALUDE_range_of_increasing_function_l2864_286410

/-- Given an increasing function f: ℝ → ℝ with f(0) = -1 and f(3) = 1,
    the set of x ∈ ℝ such that |f(x+1)| < 1 is equal to [-1, 2] -/
theorem range_of_increasing_function (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_f_0 : f 0 = -1) 
  (h_f_3 : f 3 = 1) : 
  {x : ℝ | |f (x + 1)| < 1} = Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_increasing_function_l2864_286410


namespace NUMINAMATH_CALUDE_liam_keeps_three_balloons_l2864_286452

/-- The number of balloons Liam keeps for himself when distributing
    balloons evenly among his friends. -/
def balloons_kept (total : ℕ) (friends : ℕ) : ℕ :=
  total % friends

theorem liam_keeps_three_balloons :
  balloons_kept 243 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_liam_keeps_three_balloons_l2864_286452


namespace NUMINAMATH_CALUDE_largest_number_in_sampling_l2864_286441

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  smallest_number : ℕ
  second_smallest : ℕ
  selected_count : ℕ
  common_difference : ℕ

/-- The largest number in a systematic sampling. -/
def largest_number (s : SystematicSampling) : ℕ :=
  s.smallest_number + (s.selected_count - 1) * s.common_difference

/-- Theorem stating the largest number in the given systematic sampling. -/
theorem largest_number_in_sampling :
  let s : SystematicSampling := {
    total_students := 80,
    smallest_number := 6,
    second_smallest := 14,
    selected_count := 10,
    common_difference := 8
  }
  largest_number s = 78 := by sorry

end NUMINAMATH_CALUDE_largest_number_in_sampling_l2864_286441


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l2864_286449

theorem quadratic_roots_difference (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*a*x₁ + 5*a^2 - 6*a = 0 ∧ 
    x₂^2 - 4*a*x₂ + 5*a^2 - 6*a = 0 ∧
    |x₁ - x₂| = 6) → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l2864_286449


namespace NUMINAMATH_CALUDE_tan_x_2_implies_expression_half_l2864_286438

theorem tan_x_2_implies_expression_half (x : ℝ) (h : Real.tan x = 2) :
  (2 * Real.sin (Real.pi + x) * Real.cos (Real.pi - x) - Real.cos (Real.pi + x)) /
  (1 + Real.sin x ^ 2 + Real.sin (Real.pi - x) - Real.cos (Real.pi - x) ^ 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_2_implies_expression_half_l2864_286438


namespace NUMINAMATH_CALUDE_expression_simplification_l2864_286405

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 1 - (2 * x - 2) / (x + 1)) / ((x^2 - x) / (2 * x + 2)) = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2864_286405


namespace NUMINAMATH_CALUDE_eulers_formula_l2864_286450

/-- A convex polyhedron is a structure with faces, vertices, and edges. -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- Euler's formula for convex polyhedra states that V + F - E = 2 -/
theorem eulers_formula (P : ConvexPolyhedron) : 
  P.vertices + P.faces - P.edges = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l2864_286450


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2864_286416

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ x * ↑(⌊x⌋) = 162 ∧ x = 13.5 := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2864_286416


namespace NUMINAMATH_CALUDE_arithmetic_seq_bicolored_l2864_286407

/-- A coloring function for natural numbers -/
def coloring (n : ℕ) : Bool :=
  let segment := (Nat.sqrt (8 * n + 1) - 1) / 2
  segment % 2 = 0

/-- Definition of an arithmetic sequence -/
def isArithmeticSeq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + r

/-- Theorem stating that every infinite arithmetic sequence is bi-colored -/
theorem arithmetic_seq_bicolored :
  ∀ (a : ℕ → ℕ) (r : ℕ), isArithmeticSeq a r →
  (∃ k, coloring (a k) = true) ∧ (∃ m, coloring (a m) = false) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_seq_bicolored_l2864_286407


namespace NUMINAMATH_CALUDE_find_unknown_areas_l2864_286477

/-- Represents the areas of rectangles in a divided larger rectangle -/
structure RectangleAreas where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  area5 : ℝ
  a : ℝ
  b : ℝ

/-- Theorem stating the values of unknown areas a and b given other known areas -/
theorem find_unknown_areas (areas : RectangleAreas) 
  (h1 : areas.area1 = 25)
  (h2 : areas.area2 = 104)
  (h3 : areas.area3 = 40)
  (h4 : areas.area4 = 143)
  (h5 : areas.area5 = 66)
  (h6 : areas.area2 / areas.area3 = areas.area4 / areas.b)
  (h7 : areas.area1 / areas.a = areas.b / areas.area5) :
  areas.a = 30 ∧ areas.b = 55 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_areas_l2864_286477


namespace NUMINAMATH_CALUDE_common_tangent_line_sum_of_coefficients_l2864_286424

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = x^2 + 121/100

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = y^2 + 49/4

/-- The common tangent line L -/
def L (x y : ℝ) : Prop := x + 25*y = 12

/-- The theorem stating that L is a common tangent to P₁ and P₂, 
    and 1, 25, 12 are the smallest positive integers satisfying the equation -/
theorem common_tangent_line : 
  (∀ x y : ℝ, P₁ x y → L x y → (∃ u : ℝ, ∀ v : ℝ, P₁ u v → L u v → (u, v) = (x, y))) ∧ 
  (∀ x y : ℝ, P₂ x y → L x y → (∃ u : ℝ, ∀ v : ℝ, P₂ u v → L u v → (u, v) = (x, y))) ∧
  (∀ a b c : ℕ+, (∀ x y : ℝ, a*x + b*y = c ↔ L x y) → a ≥ 1 ∧ b ≥ 25 ∧ c ≥ 12) :=
sorry

/-- The sum of the coefficients -/
def coefficient_sum : ℕ := 38

/-- Theorem stating that the sum of coefficients is 38 -/
theorem sum_of_coefficients : 
  ∀ a b c : ℕ+, (∀ x y : ℝ, a*x + b*y = c ↔ L x y) → (a : ℕ) + (b : ℕ) + (c : ℕ) = coefficient_sum :=
sorry

end NUMINAMATH_CALUDE_common_tangent_line_sum_of_coefficients_l2864_286424


namespace NUMINAMATH_CALUDE_cone_prism_volume_ratio_l2864_286423

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism
    to the volume of the prism is π/16. -/
theorem cone_prism_volume_ratio :
  ∀ (cone_volume prism_volume : ℝ) (prism_base_length prism_base_width prism_height : ℝ),
  prism_base_length = 3 →
  prism_base_width = 4 →
  prism_height = 5 →
  prism_volume = prism_base_length * prism_base_width * prism_height →
  cone_volume = (1/3) * π * (prism_base_length/2)^2 * prism_height →
  cone_volume / prism_volume = π/16 := by
sorry

end NUMINAMATH_CALUDE_cone_prism_volume_ratio_l2864_286423


namespace NUMINAMATH_CALUDE_correct_calculation_l2864_286445

theorem correct_calculation (x : ℝ) (h : x / 15 = 6) : 15 * x = 1350 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2864_286445


namespace NUMINAMATH_CALUDE_rocket_height_problem_l2864_286468

theorem rocket_height_problem (h : ℝ) : 
  h + 2 * h = 1500 → h = 500 := by
  sorry

end NUMINAMATH_CALUDE_rocket_height_problem_l2864_286468


namespace NUMINAMATH_CALUDE_min_positive_period_sin_cos_squared_l2864_286470

theorem min_positive_period_sin_cos_squared (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (Real.sin x + Real.cos x)^2 + 1
  ∃ T : ℝ, T > 0 ∧ (∀ t : ℝ, f (t + T) = f t) ∧
    (∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (t + S) = f t) → T ≤ S) ∧
    T = π :=
by sorry

end NUMINAMATH_CALUDE_min_positive_period_sin_cos_squared_l2864_286470


namespace NUMINAMATH_CALUDE_rectangle_area_with_hole_l2864_286404

theorem rectangle_area_with_hole (x : ℝ) : 
  (x + 7) * (x + 5) - (x + 1) * (x + 4) = 7 * x + 31 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_hole_l2864_286404


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l2864_286459

def current_salary : ℝ := 300

theorem salary_increase_percentage : 
  (∃ (increase_percent : ℝ), 
    current_salary * (1 + 0.16) = 348 ∧ 
    current_salary * (1 + increase_percent / 100) = 330 ∧ 
    increase_percent = 10) :=
by sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l2864_286459


namespace NUMINAMATH_CALUDE_correct_water_ratio_l2864_286426

/-- Represents the time in minutes to fill the bathtub with hot water -/
def hot_water_fill_time : ℝ := 23

/-- Represents the time in minutes to fill the bathtub with cold water -/
def cold_water_fill_time : ℝ := 17

/-- Represents the ratio of hot water to cold water when the bathtub is full -/
def hot_to_cold_ratio : ℝ := 1.5

/-- Represents the delay in minutes before opening the cold water tap -/
def cold_water_delay : ℝ := 7

/-- Proves that opening the cold water tap after the specified delay results in the correct ratio of hot to cold water -/
theorem correct_water_ratio : 
  let hot_water_volume := (hot_water_fill_time - cold_water_delay) / hot_water_fill_time
  let cold_water_volume := cold_water_delay / cold_water_fill_time
  hot_water_volume = hot_to_cold_ratio * cold_water_volume := by
  sorry

end NUMINAMATH_CALUDE_correct_water_ratio_l2864_286426


namespace NUMINAMATH_CALUDE_comparison_theorem_l2864_286485

theorem comparison_theorem :
  (3 * 10^5 < 2 * 10^6) ∧ (-2 - 1/3 > -3 - 1/2) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l2864_286485


namespace NUMINAMATH_CALUDE_girls_in_classroom_l2864_286417

theorem girls_in_classroom (total_students : ℕ) (girls_ratio boys_ratio : ℕ) : 
  total_students = 28 → 
  girls_ratio = 3 → 
  boys_ratio = 4 → 
  (girls_ratio + boys_ratio) * (total_students / (girls_ratio + boys_ratio)) = girls_ratio * (total_students / (girls_ratio + boys_ratio)) + boys_ratio * (total_students / (girls_ratio + boys_ratio)) →
  girls_ratio * (total_students / (girls_ratio + boys_ratio)) = 12 := by
sorry

end NUMINAMATH_CALUDE_girls_in_classroom_l2864_286417


namespace NUMINAMATH_CALUDE_total_writing_instruments_l2864_286418

theorem total_writing_instruments (pens pencils markers : ℕ) : 
  (5 * pens = 6 * pencils - 54) →  -- Ratio of pens to pencils is 5:6, and 9 more pencils
  (4 * pencils = 3 * markers) →    -- Ratio of markers to pencils is 4:3
  pens + pencils + markers = 171   -- Total number of writing instruments
  := by sorry

end NUMINAMATH_CALUDE_total_writing_instruments_l2864_286418


namespace NUMINAMATH_CALUDE_maggie_grandfather_subscriptions_l2864_286476

/-- Represents the number of magazine subscriptions Maggie sold to her grandfather. -/
def grandfather_subscriptions : ℕ := sorry

/-- The amount Maggie earns per subscription in dollars. -/
def earnings_per_subscription : ℕ := 5

/-- The number of subscriptions Maggie sold to her parents. -/
def parent_subscriptions : ℕ := 4

/-- The number of subscriptions Maggie sold to the next-door neighbor. -/
def neighbor_subscriptions : ℕ := 2

/-- The number of subscriptions Maggie sold to another neighbor. -/
def other_neighbor_subscriptions : ℕ := 2 * neighbor_subscriptions

/-- The total amount Maggie earned in dollars. -/
def total_earnings : ℕ := 55

/-- Theorem stating that Maggie sold 1 subscription to her grandfather. -/
theorem maggie_grandfather_subscriptions : grandfather_subscriptions = 1 := by
  sorry

end NUMINAMATH_CALUDE_maggie_grandfather_subscriptions_l2864_286476


namespace NUMINAMATH_CALUDE_number_puzzle_l2864_286430

theorem number_puzzle (x : ℝ) : 9 * (((x + 1.4) / 3) - 0.7) = 5.4 → x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2864_286430


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l2864_286464

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (2, 3)

/-- First line equation -/
def line1 (x y : ℝ) : Prop := 9 * x - 4 * y = 6

/-- Second line equation -/
def line2 (x y : ℝ) : Prop := 7 * x + y = 17

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique :
  ∀ (x y : ℝ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l2864_286464


namespace NUMINAMATH_CALUDE_polygon_diagonals_l2864_286462

theorem polygon_diagonals (n : ℕ) (h : n ≥ 3) :
  (n * (n - 1)) / 2 - n = 20 → n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l2864_286462


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2864_286491

theorem fixed_point_on_line (k : ℝ) : k * 1 - k = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2864_286491


namespace NUMINAMATH_CALUDE_river_distance_l2864_286456

/-- The distance between two points on a river, given boat speeds and time difference -/
theorem river_distance (v_down v_up : ℝ) (time_diff : ℝ) (h1 : v_down = 20)
  (h2 : v_up = 15) (h3 : time_diff = 5) :
  ∃ d : ℝ, d = 300 ∧ d / v_up - d / v_down = time_diff :=
by sorry

end NUMINAMATH_CALUDE_river_distance_l2864_286456


namespace NUMINAMATH_CALUDE_complement_of_P_union_Q_is_M_l2864_286469

def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}
def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

theorem complement_of_P_union_Q_is_M : (P ∪ Q)ᶜ = M := by sorry

end NUMINAMATH_CALUDE_complement_of_P_union_Q_is_M_l2864_286469


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2864_286497

theorem trigonometric_identity (α : ℝ) : 
  (Real.cos (2 * α))^4 - 6 * (Real.cos (2 * α))^2 * (Real.sin (2 * α))^2 + (Real.sin (2 * α))^4 = Real.cos (8 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2864_286497


namespace NUMINAMATH_CALUDE_f_2018_is_zero_l2864_286455

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def has_period_property (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) - f x = 2 * f 2

-- State the theorem
theorem f_2018_is_zero 
  (h_even : is_even f) 
  (h_period : has_period_property f) : 
  f 2018 = 0 := by sorry

end NUMINAMATH_CALUDE_f_2018_is_zero_l2864_286455


namespace NUMINAMATH_CALUDE_nancy_sweaters_count_l2864_286422

/-- Represents the washing machine capacity -/
def machine_capacity : ℕ := 9

/-- Represents the number of shirts Nancy had to wash -/
def number_of_shirts : ℕ := 19

/-- Represents the total number of loads Nancy did -/
def total_loads : ℕ := 3

/-- Calculates the number of sweaters Nancy had to wash -/
def number_of_sweaters : ℕ := machine_capacity

theorem nancy_sweaters_count :
  number_of_sweaters = machine_capacity := by sorry

end NUMINAMATH_CALUDE_nancy_sweaters_count_l2864_286422


namespace NUMINAMATH_CALUDE_point_line_plane_relation_l2864_286406

-- Define the types for point, line, and plane
variable (Point Line Plane : Type)

-- Define the relations
variable (lies_on : Point → Line → Prop)
variable (is_in : Line → Plane → Prop)

-- Define the set membership and subset relations
variable (mem : Point → Line → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem point_line_plane_relation 
  (P : Point) (m : Line) (α : Plane) 
  (h1 : lies_on P m) 
  (h2 : is_in m α) : 
  mem P m ∧ subset m α :=
sorry

end NUMINAMATH_CALUDE_point_line_plane_relation_l2864_286406


namespace NUMINAMATH_CALUDE_inverse_equivalent_is_contrapositive_l2864_286413

theorem inverse_equivalent_is_contrapositive (p q : Prop) :
  (q → p) ↔ (¬p → ¬q) :=
sorry

end NUMINAMATH_CALUDE_inverse_equivalent_is_contrapositive_l2864_286413


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l2864_286428

theorem square_plus_inverse_square (x : ℝ) (h : x^4 + 1/x^4 = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l2864_286428


namespace NUMINAMATH_CALUDE_probability_theorem_l2864_286484

def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def green_marbles : ℕ := 5
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles
def marbles_selected : ℕ := 4

def probability_one_red_two_blue_one_green : ℚ :=
  (red_marbles.choose 1 * blue_marbles.choose 2 * green_marbles.choose 1) /
  (total_marbles.choose marbles_selected)

theorem probability_theorem :
  probability_one_red_two_blue_one_green = 411 / 4200 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2864_286484


namespace NUMINAMATH_CALUDE_margin_relation_l2864_286487

theorem margin_relation (n : ℝ) (C S M : ℝ) 
  (h1 : M = (1/n) * C) 
  (h2 : S = C + M) : 
  M = (1/(n+1)) * S := by
sorry

end NUMINAMATH_CALUDE_margin_relation_l2864_286487


namespace NUMINAMATH_CALUDE_complex_product_modulus_l2864_286401

theorem complex_product_modulus : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_modulus_l2864_286401


namespace NUMINAMATH_CALUDE_f_of_five_equals_sixtytwo_l2864_286463

/-- Given a function f where f(x) = 2x² + y and f(2) = 20, prove that f(5) = 62 -/
theorem f_of_five_equals_sixtytwo (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 20) : 
  f 5 = 62 := by
  sorry

end NUMINAMATH_CALUDE_f_of_five_equals_sixtytwo_l2864_286463
