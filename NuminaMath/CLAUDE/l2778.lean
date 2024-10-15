import Mathlib

namespace NUMINAMATH_CALUDE_percentage_sum_l2778_277833

theorem percentage_sum (P Q R x y : ℝ) 
  (h_pos_P : P > 0) (h_pos_Q : Q > 0) (h_pos_R : R > 0)
  (h_PQ : P = (1 + x / 100) * Q)
  (h_QR : Q = (1 + y / 100) * R)
  (h_PR : P = 2.4 * R) : 
  x + y = 140 := by sorry

end NUMINAMATH_CALUDE_percentage_sum_l2778_277833


namespace NUMINAMATH_CALUDE_positive_numbers_properties_l2778_277837

theorem positive_numbers_properties (a b : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_lt : a < b) (h_sum : a + b = 2) : 
  (1 < b ∧ b < 2) ∧ (Real.sqrt a + Real.sqrt b < 2) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_properties_l2778_277837


namespace NUMINAMATH_CALUDE_cube_surface_area_from_diagonal_l2778_277868

/-- Given a cube with diagonal length a/2, prove that its total surface area is a^2/2 -/
theorem cube_surface_area_from_diagonal (a : ℝ) (h : a > 0) :
  let diagonal := a / 2
  let side := diagonal / Real.sqrt 3
  let surface_area := 6 * side ^ 2
  surface_area = a ^ 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_diagonal_l2778_277868


namespace NUMINAMATH_CALUDE_hollow_cube_side_length_l2778_277809

/-- The number of cubes required to construct a hollow cube -/
def hollow_cube_cubes (n : ℕ) : ℕ := n^3 - (n-2)^3

/-- Theorem: A hollow cube made of 98 unit cubes has a side length of 5 -/
theorem hollow_cube_side_length :
  ∃ (n : ℕ), n > 0 ∧ hollow_cube_cubes n = 98 → n = 5 := by
sorry

end NUMINAMATH_CALUDE_hollow_cube_side_length_l2778_277809


namespace NUMINAMATH_CALUDE_sabertooth_tails_count_l2778_277891

/-- Represents the number of legs for Triassic Discoglossus tadpoles -/
def triassic_legs : ℕ := 5

/-- Represents the number of tails for Triassic Discoglossus tadpoles -/
def triassic_tails : ℕ := 1

/-- Represents the number of legs for Sabertooth Frog tadpoles -/
def sabertooth_legs : ℕ := 4

/-- Represents the total number of legs of all captured tadpoles -/
def total_legs : ℕ := 100

/-- Represents the total number of tails of all captured tadpoles -/
def total_tails : ℕ := 64

/-- Proves that the number of tails per Sabertooth Frog tadpole is 3 -/
theorem sabertooth_tails_count :
  ∃ (n k : ℕ),
    n * triassic_legs + k * sabertooth_legs = total_legs ∧
    n * triassic_tails + k * 3 = total_tails :=
by sorry

end NUMINAMATH_CALUDE_sabertooth_tails_count_l2778_277891


namespace NUMINAMATH_CALUDE_exists_divisible_by_n_l2778_277802

theorem exists_divisible_by_n (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ k : ℕ, k < n ∧ (n ∣ 2^k - 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_n_l2778_277802


namespace NUMINAMATH_CALUDE_sum_of_base9_series_l2778_277850

/-- Converts a base 9 number to base 10 -/
def base9ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 9 -/
def base10ToBase9 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of an arithmetic series in base 10 -/
def arithmeticSeriesSum (n : ℕ) (a1 : ℕ) (an : ℕ) : ℕ := sorry

theorem sum_of_base9_series :
  let n : ℕ := 36
  let a1 : ℕ := base9ToBase10 1
  let an : ℕ := base9ToBase10 36
  let sum : ℕ := arithmeticSeriesSum n a1 an
  base10ToBase9 sum = 750 := by sorry

end NUMINAMATH_CALUDE_sum_of_base9_series_l2778_277850


namespace NUMINAMATH_CALUDE_range_of_m_l2778_277848

theorem range_of_m (x m : ℝ) : 
  (∀ x, 1/3 < x ∧ x < 1/2 → |x - m| < 1) ∧ 
  (∃ x, |x - m| < 1 ∧ ¬(1/3 < x ∧ x < 1/2)) →
  -1/2 ≤ m ∧ m ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2778_277848


namespace NUMINAMATH_CALUDE_min_value_squared_difference_l2778_277892

theorem min_value_squared_difference (f : ℝ → ℝ) :
  (∀ x, f x = (x - 1)^2) →
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_squared_difference_l2778_277892


namespace NUMINAMATH_CALUDE_sin_2A_eq_sin_2B_neither_sufficient_nor_necessary_l2778_277801

/-- A triangle ABC -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π

/-- Definition of an isosceles triangle -/
def is_isosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

/-- The condition sin 2A = sin 2B -/
def condition (t : Triangle) : Prop :=
  Real.sin (2 * t.A) = Real.sin (2 * t.B)

/-- The main theorem to prove -/
theorem sin_2A_eq_sin_2B_neither_sufficient_nor_necessary :
  ¬(∀ t : Triangle, condition t → is_isosceles t) ∧
  ¬(∀ t : Triangle, is_isosceles t → condition t) := by
  sorry

end NUMINAMATH_CALUDE_sin_2A_eq_sin_2B_neither_sufficient_nor_necessary_l2778_277801


namespace NUMINAMATH_CALUDE_annalise_tissue_purchase_cost_l2778_277872

/-- Calculates the total cost of tissues given the number of boxes, packs per box, tissues per pack, and cost per tissue -/
def totalCost (boxes : ℕ) (packsPerBox : ℕ) (tissuesPerPack : ℕ) (costPerTissue : ℚ) : ℚ :=
  boxes * packsPerBox * tissuesPerPack * costPerTissue

/-- Proves that the total cost for Annalise's purchase is $1,000 -/
theorem annalise_tissue_purchase_cost :
  totalCost 10 20 100 (5 / 100) = 1000 := by
  sorry

#eval totalCost 10 20 100 (5 / 100)

end NUMINAMATH_CALUDE_annalise_tissue_purchase_cost_l2778_277872


namespace NUMINAMATH_CALUDE_remainder_two_pow_33_mod_9_l2778_277803

theorem remainder_two_pow_33_mod_9 : 2^33 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_two_pow_33_mod_9_l2778_277803


namespace NUMINAMATH_CALUDE_class_size_l2778_277816

theorem class_size (female_students : ℕ) (male_students : ℕ) : 
  female_students = 13 → 
  male_students = 3 * female_students → 
  female_students + male_students = 52 := by
sorry

end NUMINAMATH_CALUDE_class_size_l2778_277816


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2778_277817

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) : 
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = (Real.sqrt (x^12 + 7*x^6 + 1)) / (3 * x^3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2778_277817


namespace NUMINAMATH_CALUDE_grandmother_age_multiple_l2778_277838

def milena_age : ℕ := 7

def grandfather_age_difference (grandmother_age : ℕ) : ℕ := grandmother_age + 2

theorem grandmother_age_multiple : ∃ (grandmother_age : ℕ), 
  grandfather_age_difference grandmother_age - milena_age = 58 ∧ 
  grandmother_age = 9 * milena_age := by
  sorry

end NUMINAMATH_CALUDE_grandmother_age_multiple_l2778_277838


namespace NUMINAMATH_CALUDE_expression_value_at_four_l2778_277828

theorem expression_value_at_four :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 5)
  f 4 = 6 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_four_l2778_277828


namespace NUMINAMATH_CALUDE_system_solution_l2778_277814

theorem system_solution (a b c : ℝ) :
  ∃! (x y z : ℝ),
    (x + a * y + a^2 * z + a^3 = 0) ∧
    (x + b * y + b^2 * z + b^3 = 0) ∧
    (x + c * y + c^2 * z + c^3 = 0) ∧
    (x = -a * b * c) ∧
    (y = a * b + b * c + c * a) ∧
    (z = -(a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2778_277814


namespace NUMINAMATH_CALUDE_patrol_results_l2778_277888

def travel_records : List Int := [10, -8, 6, -13, 7, -12, 3, -1]

def fuel_consumption_rate : ℝ := 0.05

def gas_station_distance : Int := 6

def final_position (records : List Int) : Int :=
  records.sum

def total_distance (records : List Int) : Int :=
  records.map (Int.natAbs) |>.sum

def times_passed_gas_station (records : List Int) (station_dist : Int) : Nat :=
  sorry

theorem patrol_results :
  (final_position travel_records = -8) ∧
  (total_distance travel_records = 60) ∧
  (times_passed_gas_station travel_records gas_station_distance = 4) := by
  sorry

end NUMINAMATH_CALUDE_patrol_results_l2778_277888


namespace NUMINAMATH_CALUDE_existence_of_abc_l2778_277813

theorem existence_of_abc (n : ℕ) : ∃ (a b c : ℤ),
  n = Int.gcd a b * (c^2 - a*b) + Int.gcd b c * (a^2 - b*c) + Int.gcd c a * (b^2 - c*a) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_abc_l2778_277813


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2778_277889

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define S_n as the sum of first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1 + a n) / 2

-- Define b_n
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Define T_n as the sum of first n terms of b_n
def T (n : ℕ) : ℚ := n / (2 * n + 1)

theorem arithmetic_sequence_properties (n : ℕ) :
  (S 3 = a 4 + 2) ∧ 
  (a 3 ^ 2 = a 1 * a 13) ∧ 
  (∀ k : ℕ, a (k + 1) - a k = a 2 - a 1) ∧
  (∀ k : ℕ, b k = 1 / (a k * a (k + 1))) ∧
  (∀ k : ℕ, T k = k / (2 * k + 1)) := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2778_277889


namespace NUMINAMATH_CALUDE_journey_distance_l2778_277852

theorem journey_distance (train_fraction bus_fraction : ℚ) (walk_distance : ℝ) 
  (h1 : train_fraction = 3/5)
  (h2 : bus_fraction = 7/20)
  (h3 : walk_distance = 6.5)
  (h4 : train_fraction + bus_fraction + (walk_distance / total_distance) = 1) :
  total_distance = 130 :=
by
  sorry

#check journey_distance

end NUMINAMATH_CALUDE_journey_distance_l2778_277852


namespace NUMINAMATH_CALUDE_C_sufficient_for_A_l2778_277804

-- Define propositions A, B, and C
variable (A B C : Prop)

-- Define the conditions
variable (h1 : A ↔ B)
variable (h2 : C → B)
variable (h3 : ¬(B → C))

-- Theorem statement
theorem C_sufficient_for_A : C → A := by
  sorry

end NUMINAMATH_CALUDE_C_sufficient_for_A_l2778_277804


namespace NUMINAMATH_CALUDE_trebled_result_proof_l2778_277883

theorem trebled_result_proof (initial_number : ℕ) : 
  initial_number = 18 → 
  3 * (2 * initial_number + 5) = 123 := by
sorry

end NUMINAMATH_CALUDE_trebled_result_proof_l2778_277883


namespace NUMINAMATH_CALUDE_externally_tangent_circles_distance_l2778_277829

/-- The distance between the centers of two externally tangent circles
    is equal to the sum of their radii -/
theorem externally_tangent_circles_distance
  (r₁ r₂ d : ℝ) 
  (h₁ : r₁ = 2)
  (h₂ : r₂ = 3)
  (h_tangent : d = r₁ + r₂) :
  d = 5 := by sorry

end NUMINAMATH_CALUDE_externally_tangent_circles_distance_l2778_277829


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2778_277877

theorem yellow_balls_count (total : ℕ) (yellow : ℕ) (h1 : total = 15) 
  (h2 : yellow ≤ total) 
  (h3 : (yellow : ℚ) / total * (yellow - 1) / (total - 1) = 1 / 21) : 
  yellow = 5 := by sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2778_277877


namespace NUMINAMATH_CALUDE_equal_area_divide_sum_of_squares_l2778_277870

-- Define the region S as a set of points in the plane
def S : Set (ℝ × ℝ) := sorry

-- Define the line m with slope 4
def m : Set (ℝ × ℝ) := {(x, y) | 4 * x = y + c} where c : ℝ := sorry

-- Define the property of m dividing S into two equal areas
def divides_equally (l : Set (ℝ × ℝ)) (r : Set (ℝ × ℝ)) : Prop := sorry

-- Define the equation of line m in the form ax = by + c
def line_equation (a b c : ℕ) : Set (ℝ × ℝ) := {(x, y) | a * x = b * y + c}

-- Main theorem
theorem equal_area_divide_sum_of_squares :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.gcd a (Nat.gcd b c) = 1 ∧
    divides_equally (line_equation a b c) S ∧
    m = line_equation a b c ∧
    a^2 + b^2 + c^2 = 65 := by sorry

end NUMINAMATH_CALUDE_equal_area_divide_sum_of_squares_l2778_277870


namespace NUMINAMATH_CALUDE_unique_solution_mod_37_l2778_277886

theorem unique_solution_mod_37 :
  ∃! (a b c d : ℤ),
    (a^2 + b*c) % 37 = a % 37 ∧
    (b*(a + d)) % 37 = b % 37 ∧
    (c*(a + d)) % 37 = c % 37 ∧
    (b*c + d^2) % 37 = d % 37 ∧
    (a*d - b*c) % 37 = 1 % 37 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_mod_37_l2778_277886


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2778_277893

-- Define the diamond operation
noncomputable def diamond (b c : ℝ) : ℝ := b + Real.sqrt (c + Real.sqrt (c + Real.sqrt c))

-- State the theorem
theorem diamond_equation_solution (k : ℝ) :
  diamond 10 k = 13 → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2778_277893


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l2778_277865

theorem smallest_n_for_sqrt_difference (n : ℕ) : 
  (n > 0) → 
  (∀ m : ℕ, m > 0 → m < 626 → Real.sqrt m - Real.sqrt (m - 1) ≥ 0.02) → 
  (Real.sqrt 626 - Real.sqrt 625 < 0.02) → 
  (626 = n) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l2778_277865


namespace NUMINAMATH_CALUDE_cube_root_of_product_l2778_277834

theorem cube_root_of_product (a b c : ℕ) : 
  (2^9 * 3^6 * 7^3 : ℝ)^(1/3) = 504 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l2778_277834


namespace NUMINAMATH_CALUDE_mountain_bike_helmet_cost_l2778_277897

/-- Calculates the cost of a mountain bike helmet based on Alfonso's savings and earnings --/
theorem mountain_bike_helmet_cost
  (daily_earnings : ℕ)
  (current_savings : ℕ)
  (days_per_week : ℕ)
  (weeks_to_work : ℕ)
  (h1 : daily_earnings = 6)
  (h2 : current_savings = 40)
  (h3 : days_per_week = 5)
  (h4 : weeks_to_work = 10) :
  daily_earnings * days_per_week * weeks_to_work + current_savings = 340 :=
by
  sorry

end NUMINAMATH_CALUDE_mountain_bike_helmet_cost_l2778_277897


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2778_277873

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2778_277873


namespace NUMINAMATH_CALUDE_min_students_with_blue_eyes_and_backpack_l2778_277864

theorem min_students_with_blue_eyes_and_backpack
  (total_students : ℕ)
  (blue_eyes : ℕ)
  (backpack : ℕ)
  (h1 : total_students = 25)
  (h2 : blue_eyes = 15)
  (h3 : backpack = 18)
  : ∃ (both : ℕ), both ≥ 7 ∧ both ≤ min blue_eyes backpack :=
by
  sorry

end NUMINAMATH_CALUDE_min_students_with_blue_eyes_and_backpack_l2778_277864


namespace NUMINAMATH_CALUDE_max_volume_at_one_sixth_l2778_277808

/-- The volume of an open-topped box made from a square sheet of cardboard --/
def boxVolume (a x : ℝ) : ℝ := (a - 2*x)^2 * x

/-- The theorem stating that the volume is maximized when the cutout side length is a/6 --/
theorem max_volume_at_one_sixth (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x > 0 ∧ x < a/2 ∧
  ∀ (y : ℝ), y > 0 → y < a/2 → boxVolume a x ≥ boxVolume a y ∧
  x = a/6 :=
sorry

end NUMINAMATH_CALUDE_max_volume_at_one_sixth_l2778_277808


namespace NUMINAMATH_CALUDE_miyeon_gets_48_sheets_l2778_277827

/-- The number of sheets Miyeon gets given the conditions of the paper sharing problem -/
def miyeon_sheets (total_sheets : ℕ) (pink_sheets : ℕ) : ℕ :=
  (total_sheets - pink_sheets) / 2 + pink_sheets

/-- Theorem stating that Miyeon gets 48 sheets under the given conditions -/
theorem miyeon_gets_48_sheets :
  miyeon_sheets 85 11 = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_miyeon_gets_48_sheets_l2778_277827


namespace NUMINAMATH_CALUDE_card_sum_proof_l2778_277899

theorem card_sum_proof (H S D C : ℕ) : 
  (∃ (h₁ h₂ : ℕ), h₁ + h₂ = H ∧ h₁ ≥ 1 ∧ h₂ ≥ 1 ∧ h₁ ≤ 13 ∧ h₂ ≤ 13) →
  (∃ (s₁ s₂ s₃ : ℕ), s₁ + s₂ + s₃ = S ∧ s₁ ≥ 1 ∧ s₂ ≥ 1 ∧ s₃ ≥ 1 ∧ s₁ ≤ 13 ∧ s₂ ≤ 13 ∧ s₃ ≤ 13) →
  (∃ (d₁ d₂ d₃ d₄ : ℕ), d₁ + d₂ + d₃ + d₄ = D ∧ d₁ ≥ 1 ∧ d₂ ≥ 1 ∧ d₃ ≥ 1 ∧ d₄ ≥ 1 ∧ d₁ ≤ 13 ∧ d₂ ≤ 13 ∧ d₃ ≤ 13 ∧ d₄ ≤ 13) →
  (∃ (c₁ c₂ c₃ c₄ c₅ : ℕ), c₁ + c₂ + c₃ + c₄ + c₅ = C ∧ c₁ ≥ 1 ∧ c₂ ≥ 1 ∧ c₃ ≥ 1 ∧ c₄ ≥ 1 ∧ c₅ ≥ 1 ∧ c₁ ≤ 13 ∧ c₂ ≤ 13 ∧ c₃ ≤ 13 ∧ c₄ ≤ 13 ∧ c₅ ≤ 13) →
  S = 11 * H →
  C = D + 45 →
  H + S + D + C = 101 :=
by
  sorry

end NUMINAMATH_CALUDE_card_sum_proof_l2778_277899


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2778_277821

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {-1, a^2}
def B : Set ℝ := {2, 4}

-- Define the property we want to prove
def property (a : ℝ) : Prop := A a ∩ B = {4}

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = -2 → property a) ∧
  ¬(∀ a : ℝ, property a → a = -2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2778_277821


namespace NUMINAMATH_CALUDE_absolute_value_expression_l2778_277853

theorem absolute_value_expression : |(|-|-2 + 3| - 2| + 2)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l2778_277853


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2778_277879

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2778_277879


namespace NUMINAMATH_CALUDE_final_diaries_count_l2778_277822

def calculate_final_diaries (initial : ℕ) : ℕ :=
  let after_buying := initial + 3 * initial
  let lost := (3 * after_buying) / 5
  after_buying - lost

theorem final_diaries_count : calculate_final_diaries 15 = 24 := by
  sorry

end NUMINAMATH_CALUDE_final_diaries_count_l2778_277822


namespace NUMINAMATH_CALUDE_S_when_m_is_one_l_range_when_m_is_neg_half_m_range_when_l_is_half_l2778_277849

-- Define the set S
def S (m l : ℝ) : Set ℝ := {x : ℝ | m ≤ x ∧ x ≤ l}

-- State the condition that if x ∈ S, then x^2 ∈ S
axiom S_closed_square (m l : ℝ) : ∀ x ∈ S m l, x^2 ∈ S m l

-- Theorem 1
theorem S_when_m_is_one (l : ℝ) : 
  S 1 l = {1} := by sorry

-- Theorem 2
theorem l_range_when_m_is_neg_half : 
  ∀ l, S (-1/2) l ≠ ∅ ↔ 1/4 ≤ l ∧ l ≤ 1 := by sorry

-- Theorem 3
theorem m_range_when_l_is_half : 
  ∀ m, S m (1/2) ≠ ∅ ↔ -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_S_when_m_is_one_l_range_when_m_is_neg_half_m_range_when_l_is_half_l2778_277849


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2778_277887

theorem quadratic_roots_properties (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  (r₁^2 + p*r₁ + 12 = 0) → 
  (r₂^2 + p*r₂ + 12 = 0) → 
  (|r₁ + r₂| > 5 ∧ |r₁ * r₂| > 4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2778_277887


namespace NUMINAMATH_CALUDE_jeffrey_mailbox_steps_l2778_277841

/-- Represents Jeffrey's walking pattern -/
structure WalkingPattern where
  forward : ℕ
  backward : ℕ

/-- Calculates the total steps taken given a walking pattern and distance -/
def totalSteps (pattern : WalkingPattern) (distance : ℕ) : ℕ :=
  distance * (pattern.forward + pattern.backward) / (pattern.forward - pattern.backward)

/-- Theorem: Jeffrey takes 330 steps to reach the mailbox -/
theorem jeffrey_mailbox_steps :
  let pattern : WalkingPattern := { forward := 3, backward := 2 }
  let distance : ℕ := 66
  totalSteps pattern distance = 330 := by
  sorry

#eval totalSteps { forward := 3, backward := 2 } 66

end NUMINAMATH_CALUDE_jeffrey_mailbox_steps_l2778_277841


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l2778_277881

/-- The function y(x) satisfies the given differential equation. -/
theorem function_satisfies_equation (x : ℝ) (hx : x > 0) :
  let y : ℝ → ℝ := λ x => Real.tan (Real.log (3 * x))
  (1 + y x ^ 2) = x * (deriv y x) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l2778_277881


namespace NUMINAMATH_CALUDE_expected_sixes_is_half_l2778_277840

-- Define the number of dice
def num_dice : ℕ := 3

-- Define the probability of rolling a 6 on one die
def prob_six : ℚ := 1 / 6

-- Define the expected number of 6's
def expected_sixes : ℚ := num_dice * prob_six

-- Theorem statement
theorem expected_sixes_is_half : expected_sixes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_sixes_is_half_l2778_277840


namespace NUMINAMATH_CALUDE_factor_polynomial_l2778_277846

theorem factor_polynomial (x : ℝ) : 
  x^2 + 6*x + 9 - 16*x^4 = (-4*x^2 + 2*x + 3)*(4*x^2 + 2*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2778_277846


namespace NUMINAMATH_CALUDE_final_fish_count_l2778_277842

def fish_count (initial : ℕ) (days : ℕ) : ℕ :=
  let day1 := initial
  let day2 := day1 * 2
  let day3 := day2 * 2 - (day2 * 2) / 3
  let day4 := day3 * 2
  let day5 := day4 * 2 - (day4 * 2) / 4
  let day6 := day5 * 2
  let day7 := day6 * 2 + 15
  day7

theorem final_fish_count :
  fish_count 6 7 = 207 :=
by sorry

end NUMINAMATH_CALUDE_final_fish_count_l2778_277842


namespace NUMINAMATH_CALUDE_alex_walking_distance_l2778_277859

/-- Represents the bike trip with given conditions -/
structure BikeTrip where
  total_distance : ℝ
  flat_time : ℝ
  flat_speed : ℝ
  uphill_time : ℝ
  uphill_speed : ℝ
  downhill_time : ℝ
  downhill_speed : ℝ

/-- Calculates the distance walked given a BikeTrip -/
def distance_walked (trip : BikeTrip) : ℝ :=
  trip.total_distance - (
    trip.flat_time * trip.flat_speed +
    trip.uphill_time * trip.uphill_speed +
    trip.downhill_time * trip.downhill_speed
  )

/-- Proves that Alex walked 8 miles given the conditions of the problem -/
theorem alex_walking_distance :
  let trip : BikeTrip := {
    total_distance := 164,
    flat_time := 4.5,
    flat_speed := 20,
    uphill_time := 2.5,
    uphill_speed := 12,
    downhill_time := 1.5,
    downhill_speed := 24
  }
  distance_walked trip = 8 := by
  sorry

end NUMINAMATH_CALUDE_alex_walking_distance_l2778_277859


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l2778_277800

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 2}

-- Theorem for the first part
theorem union_condition (m : ℝ) : A ∪ B m = A → m = 1 := by sorry

-- Theorem for the second part
theorem intersection_condition (m : ℝ) : A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3} → m = 2 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l2778_277800


namespace NUMINAMATH_CALUDE_capital_payment_theorem_l2778_277880

def remaining_capital (m : ℕ) (d : ℚ) : ℚ :=
  (3/2)^(m-1) * (3000 - 3*d) + 2*d

theorem capital_payment_theorem (m : ℕ) (h : m ≥ 3) :
  ∃ d : ℚ, remaining_capital m d = 4000 ∧ 
    d = (1000 * (3^m - 2^(m+1))) / (3^m - 2^m) := by
  sorry

end NUMINAMATH_CALUDE_capital_payment_theorem_l2778_277880


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2778_277874

theorem perpendicular_slope (x y : ℝ) :
  (3 * x - 4 * y = 8) →
  (∃ m : ℝ, m = -4/3 ∧ m * (3/4) = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2778_277874


namespace NUMINAMATH_CALUDE_probability_independent_of_shape_l2778_277845

/-- A geometric model related to area -/
structure GeometricModel where
  area : ℝ
  shape : Type

/-- The probability of a geometric model -/
def probability (model : GeometricModel) : ℝ := sorry

theorem probability_independent_of_shape (model1 model2 : GeometricModel) 
  (h : model1.area = model2.area) : 
  probability model1 = probability model2 := by sorry

end NUMINAMATH_CALUDE_probability_independent_of_shape_l2778_277845


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_fourths_l2778_277862

theorem tan_thirteen_pi_fourths : Real.tan (13 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_fourths_l2778_277862


namespace NUMINAMATH_CALUDE_exists_g_compose_eq_f_l2778_277811

noncomputable def f (k ℓ : ℝ) (x : ℝ) : ℝ := k * x + ℓ

theorem exists_g_compose_eq_f (k ℓ : ℝ) (h : k > 0) :
  ∃ (a b : ℝ), ∀ x, f k ℓ x = f a b (f a b x) ∧ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_g_compose_eq_f_l2778_277811


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l2778_277898

theorem negation_of_existence_proposition :
  (¬ ∃ (c : ℝ), c > 0 ∧ ∃ (x : ℝ), x^2 - x + c = 0) ↔
  (∀ (c : ℝ), c > 0 → ¬ ∃ (x : ℝ), x^2 - x + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l2778_277898


namespace NUMINAMATH_CALUDE_circumscribed_sphere_area_folded_equilateral_triangle_l2778_277844

/-- The surface area of the circumscribed sphere of a tetrahedron formed by folding an equilateral triangle --/
theorem circumscribed_sphere_area_folded_equilateral_triangle :
  let side_length : ℝ := 2
  let height : ℝ := Real.sqrt 3
  let tetrahedron_edge1 : ℝ := 1
  let tetrahedron_edge2 : ℝ := 1
  let tetrahedron_edge3 : ℝ := height
  let sphere_radius : ℝ := Real.sqrt 5 / 2
  let sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 5 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_circumscribed_sphere_area_folded_equilateral_triangle_l2778_277844


namespace NUMINAMATH_CALUDE_male_democrat_ratio_l2778_277806

theorem male_democrat_ratio (total_participants : ℕ) 
  (female_democrats : ℕ) (h1 : total_participants = 840) 
  (h2 : female_democrats = 140) 
  (h3 : female_democrats * 2 ≤ total_participants) : 
  (total_participants / 3 - female_democrats) * 4 = 
  (total_participants - female_democrats * 2) := by
  sorry

#check male_democrat_ratio

end NUMINAMATH_CALUDE_male_democrat_ratio_l2778_277806


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l2778_277858

/-- Pizza sharing problem -/
theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let plain_pizza_cost : ℚ := 12
  let bacon_cost : ℚ := 3
  let bacon_slices : ℕ := 9
  let dave_plain_slices : ℕ := 1
  let dave_total_slices : ℕ := bacon_slices + dave_plain_slices
  let doug_slices : ℕ := total_slices - dave_total_slices
  let total_cost : ℚ := plain_pizza_cost + bacon_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let dave_payment : ℚ := cost_per_slice * dave_total_slices
  let doug_payment : ℚ := cost_per_slice * doug_slices
  dave_payment - doug_payment = 10 :=
by sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l2778_277858


namespace NUMINAMATH_CALUDE_investment_condition_l2778_277819

/-- Represents the investment scenario with three banks -/
structure InvestmentScenario where
  national_investment : ℝ
  national_rate : ℝ
  a_rate : ℝ
  b_rate : ℝ
  total_rate : ℝ

/-- The given investment scenario -/
def given_scenario : InvestmentScenario :=
  { national_investment := 7500
  , national_rate := 0.09
  , a_rate := 0.12
  , b_rate := 0.14
  , total_rate := 0.11 }

/-- The total annual income from all three banks -/
def total_income (s : InvestmentScenario) (a b : ℝ) : ℝ :=
  s.national_rate * s.national_investment + s.a_rate * a + s.b_rate * b

/-- The total investment across all three banks -/
def total_investment (s : InvestmentScenario) (a b : ℝ) : ℝ :=
  s.national_investment + a + b

/-- The theorem stating the condition for the desired total annual income -/
theorem investment_condition (s : InvestmentScenario) (a b : ℝ) :
  total_income s a b = s.total_rate * total_investment s a b ↔ 0.01 * a + 0.03 * b = 150 :=
by sorry

end NUMINAMATH_CALUDE_investment_condition_l2778_277819


namespace NUMINAMATH_CALUDE_trick_deck_cost_l2778_277847

theorem trick_deck_cost (frank_decks : ℕ) (friend_decks : ℕ) (total_spent : ℕ) :
  frank_decks = 3 →
  friend_decks = 2 →
  total_spent = 35 →
  ∃ (cost : ℕ), frank_decks * cost + friend_decks * cost = total_spent ∧ cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_trick_deck_cost_l2778_277847


namespace NUMINAMATH_CALUDE_sqrt_sum_representation_l2778_277835

theorem sqrt_sum_representation : ∃ (a b c : ℕ+),
  (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) = 
   (a.val * Real.sqrt 3 + b.val * Real.sqrt 11) / c.val) ∧
  (∀ (a' b' c' : ℕ+),
    (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) = 
     (a'.val * Real.sqrt 3 + b'.val * Real.sqrt 11) / c'.val) →
    c'.val ≥ c.val) ∧
  a.val = 44 ∧ b.val = 36 ∧ c.val = 33 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_representation_l2778_277835


namespace NUMINAMATH_CALUDE_congruence_solution_l2778_277810

theorem congruence_solution (a m : ℕ) (h1 : a < m) (h2 : m ≥ 2) :
  (∃ x : ℕ, (10 * x + 3) % 18 = 7 % 18 ∧ x % m = a) →
  (∃ x : ℕ, x % 9 = 4 ∧ a = 4 ∧ m = 9 ∧ a + m = 13) :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_l2778_277810


namespace NUMINAMATH_CALUDE_condition_analysis_l2778_277878

theorem condition_analysis (a b c : ℝ) : 
  (∀ a b c : ℝ, a * c^2 < b * c^2 → a < b) ∧ 
  (∃ a b c : ℝ, a < b ∧ a * c^2 ≥ b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l2778_277878


namespace NUMINAMATH_CALUDE_problem_statement_l2778_277882

theorem problem_statement (m n p q : ℕ) 
  (h : ∀ x : ℝ, x > 0 → (x + 1)^m / x^n - 1 = (x + 1)^p / x^q) :
  (m^2 + 2*n + p)^(2*q) = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2778_277882


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l2778_277818

theorem missing_fraction_sum (x : ℚ) : 
  x = 7/15 → 
  (1/3 : ℚ) + (1/2 : ℚ) + (-5/6 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-2/15 : ℚ) + x = 
  (13333333333333333 : ℚ) / (100000000000000000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l2778_277818


namespace NUMINAMATH_CALUDE_kimberly_skittles_l2778_277815

/-- The number of Skittles Kimberly bought -/
def skittles_bought (initial : ℕ) (final : ℕ) : ℕ := final - initial

/-- Proof that Kimberly bought 7 Skittles -/
theorem kimberly_skittles : skittles_bought 5 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_skittles_l2778_277815


namespace NUMINAMATH_CALUDE_number_of_employees_l2778_277823

def gift_cost : ℕ := 100
def boss_contribution : ℕ := 15
def employee_contribution : ℕ := 11

theorem number_of_employees : 
  ∃ (n : ℕ), 
    gift_cost = boss_contribution + 2 * boss_contribution + n * employee_contribution ∧ 
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_employees_l2778_277823


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2778_277820

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 1) * (x^2 + 6*x + 37) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2778_277820


namespace NUMINAMATH_CALUDE_a_plus_b_equals_seven_thirds_l2778_277832

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -3 * x + 2

-- State the theorem
theorem a_plus_b_equals_seven_thirds 
  (a b : ℝ) 
  (h : ∀ x, g (f a b x) = -2 * x - 3) : 
  a + b = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_seven_thirds_l2778_277832


namespace NUMINAMATH_CALUDE_work_completion_theorem_l2778_277851

theorem work_completion_theorem (original_days : ℕ) (reduced_days : ℕ) (additional_men : ℕ) : ∃ (original_men : ℕ), 
  original_days = 10 ∧ 
  reduced_days = 7 ∧ 
  additional_men = 10 ∧
  original_men * original_days = (original_men + additional_men) * reduced_days ∧
  original_men = 24 := by
sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l2778_277851


namespace NUMINAMATH_CALUDE_acute_angle_equation_l2778_277860

theorem acute_angle_equation (x : Real) : 
  x = π/3 → (Real.sin (2*x) + Real.cos x) * (Real.sin x - Real.cos x) = Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_equation_l2778_277860


namespace NUMINAMATH_CALUDE_mikes_seashells_l2778_277876

/-- Given that Joan initially found 79 seashells and has 142 seashells in total after Mike gave her some,
    prove that Mike gave Joan 63 seashells. -/
theorem mikes_seashells (joans_initial : ℕ) (joans_total : ℕ) (mikes_gift : ℕ)
    (h1 : joans_initial = 79)
    (h2 : joans_total = 142)
    (h3 : joans_total = joans_initial + mikes_gift) :
  mikes_gift = 63 := by
  sorry

end NUMINAMATH_CALUDE_mikes_seashells_l2778_277876


namespace NUMINAMATH_CALUDE_binomial_8_5_l2778_277825

theorem binomial_8_5 : Nat.choose 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_binomial_8_5_l2778_277825


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2778_277854

theorem quadratic_inequality_empty_solution_set (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 ≥ 0) → k ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2778_277854


namespace NUMINAMATH_CALUDE_sum_A_B_equals_one_l2778_277843

theorem sum_A_B_equals_one (a : ℝ) (ha : a ≠ 1 ∧ a ≠ -1) :
  let x : ℝ := (1 - a) / (1 - 1/a)
  let y : ℝ := 1 - 1/x
  let A : ℝ := 1 / (1 - (1-x)/y)
  let B : ℝ := 1 / (1 - y/(1-x))
  A + B = 1 := by
sorry


end NUMINAMATH_CALUDE_sum_A_B_equals_one_l2778_277843


namespace NUMINAMATH_CALUDE_polynomial_real_root_l2778_277831

-- Define the polynomial
def P (b x : ℝ) : ℝ := x^4 + b*x^3 - 3*x^2 + b*x + 1

-- State the theorem
theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, P b x = 0) ↔ b ≤ -1/2 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l2778_277831


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_one_fourth_cubed_l2778_277857

theorem decimal_equivalent_of_one_fourth_cubed : (1 / 4 : ℚ) ^ 3 = 0.015625 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_one_fourth_cubed_l2778_277857


namespace NUMINAMATH_CALUDE_stool_height_is_34cm_l2778_277867

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height (ceiling_height floor_height : ℝ) 
                 (light_bulb_distance_from_ceiling : ℝ)
                 (alice_height alice_reach : ℝ) : ℝ :=
  ceiling_height - floor_height - light_bulb_distance_from_ceiling - 
  (alice_height + alice_reach)

/-- Theorem stating the height of the stool Alice needs -/
theorem stool_height_is_34cm :
  let ceiling_height : ℝ := 2.4 * 100  -- Convert to cm
  let floor_height : ℝ := 0
  let light_bulb_distance_from_ceiling : ℝ := 10
  let alice_height : ℝ := 1.5 * 100  -- Convert to cm
  let alice_reach : ℝ := 46
  stool_height ceiling_height floor_height light_bulb_distance_from_ceiling
                alice_height alice_reach = 34 := by
  sorry

#eval stool_height (2.4 * 100) 0 10 (1.5 * 100) 46

end NUMINAMATH_CALUDE_stool_height_is_34cm_l2778_277867


namespace NUMINAMATH_CALUDE_rectangle_area_l2778_277885

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width * rectangle_width = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2778_277885


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_factorization_l2778_277826

theorem perfect_square_trinomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 = (x + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_factorization_l2778_277826


namespace NUMINAMATH_CALUDE_min_value_product_quotient_min_value_achieved_l2778_277895

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 4*x + 2) * (y^2 + 5*y + 3) * (z^2 + 6*z + 4) / (x*y*z) ≥ 336 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 4*a + 2) * (b^2 + 5*b + 3) * (c^2 + 6*c + 4) / (a*b*c) = 336 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_min_value_achieved_l2778_277895


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sufficient_not_necessary_l2778_277839

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

/-- Given sequences a and b with the relation b_n = a_n + a_{n+1} -/
def sequence_relation (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = a n + a (n + 1)

/-- Theorem stating that {a_n} being arithmetic is sufficient but not necessary for {b_n} to be arithmetic -/
theorem arithmetic_sequence_sufficient_not_necessary
  (a b : ℕ → ℝ) (h : sequence_relation a b) :
  (is_arithmetic_sequence a → is_arithmetic_sequence b) ∧
  ¬(is_arithmetic_sequence b → is_arithmetic_sequence a) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sufficient_not_necessary_l2778_277839


namespace NUMINAMATH_CALUDE_max_min_ratio_l2778_277824

/-- The curve on which point P moves --/
def curve (x y : ℝ) : Prop := y = 3 * Real.sqrt (1 - x^2 / 4)

/-- The expression we're maximizing and minimizing --/
def expr (x y : ℝ) : ℝ := 2 * x - y

/-- Theorem stating the ratio of max to min values of the expression --/
theorem max_min_ratio :
  ∃ (max min : ℝ),
    (∀ x y : ℝ, curve x y → expr x y ≤ max) ∧
    (∃ x y : ℝ, curve x y ∧ expr x y = max) ∧
    (∀ x y : ℝ, curve x y → expr x y ≥ min) ∧
    (∃ x y : ℝ, curve x y ∧ expr x y = min) ∧
    max / min = -4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_min_ratio_l2778_277824


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l2778_277869

theorem complete_square_quadratic (x : ℝ) : 
  x^2 + 10*x - 3 = 0 ↔ (x + 5)^2 = 28 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l2778_277869


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2778_277856

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |a * x - 1|

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Theorem statement
theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, is_even (f a) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2778_277856


namespace NUMINAMATH_CALUDE_sampling_appropriate_l2778_277875

/-- Represents methods of investigation -/
inductive InvestigationMethod
  | Sampling
  | Comprehensive
  | Other

/-- Represents the characteristics of an investigation -/
structure InvestigationCharacteristics where
  isElectronicProduct : Bool
  largeVolume : Bool
  needComprehensive : Bool

/-- Determines the appropriate investigation method based on given characteristics -/
def appropriateMethod (chars : InvestigationCharacteristics) : InvestigationMethod :=
  sorry

/-- Theorem stating that sampling investigation is appropriate for the given conditions -/
theorem sampling_appropriate (chars : InvestigationCharacteristics)
  (h1 : chars.isElectronicProduct = true)
  (h2 : chars.largeVolume = true)
  (h3 : chars.needComprehensive = false) :
  appropriateMethod chars = InvestigationMethod.Sampling :=
sorry

end NUMINAMATH_CALUDE_sampling_appropriate_l2778_277875


namespace NUMINAMATH_CALUDE_parabola_equation_l2778_277807

/-- Represents a parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  a : ℝ
  eq : ∀ x y : ℝ, y^2 = a * x

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ
  eq : ∀ x y : ℝ, y = m * x + b

/-- The chord length of a parabola intercepted by a line -/
def chordLength (p : Parabola) (l : Line) : ℝ := sorry

theorem parabola_equation (p : Parabola) (l : Line) :
  l.m = 2 ∧ l.b = -4 ∧ chordLength p l = 3 * Real.sqrt 5 →
  p.a = 4 ∨ p.a = -36 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2778_277807


namespace NUMINAMATH_CALUDE_f_symmetry_l2778_277884

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- State the theorem
theorem f_symmetry (a b : ℝ) : 
  f a b (-5) = 3 → f a b 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l2778_277884


namespace NUMINAMATH_CALUDE_max_brand_A_is_15_l2778_277836

/-- The price difference between brand A and brand B soccer balls -/
def price_difference : ℕ := 10

/-- The number of brand A soccer balls in the initial purchase -/
def initial_brand_A : ℕ := 20

/-- The number of brand B soccer balls in the initial purchase -/
def initial_brand_B : ℕ := 15

/-- The total cost of the initial purchase -/
def initial_total_cost : ℕ := 3350

/-- The total number of soccer balls to be purchased -/
def total_balls : ℕ := 50

/-- The maximum total cost for the new purchase -/
def max_total_cost : ℕ := 4650

/-- The price of a brand A soccer ball -/
def price_A : ℕ := initial_total_cost / (initial_brand_A + initial_brand_B)

/-- The price of a brand B soccer ball -/
def price_B : ℕ := price_A - price_difference

/-- The maximum number of brand A soccer balls that can be purchased -/
def max_brand_A : ℕ := (max_total_cost - price_B * total_balls) / (price_A - price_B)

theorem max_brand_A_is_15 : max_brand_A = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_brand_A_is_15_l2778_277836


namespace NUMINAMATH_CALUDE_triangle_cosine_l2778_277812

theorem triangle_cosine (A B C : Real) :
  -- Triangle conditions
  A + B + C = Real.pi →
  -- Given conditions
  Real.sin A = 3 / 5 →
  Real.cos B = 5 / 13 →
  -- Conclusion
  Real.cos C = 16 / 65 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_l2778_277812


namespace NUMINAMATH_CALUDE_total_age_proof_l2778_277805

/-- Given three people a, b, and c, where a is two years older than b, 
    b is twice as old as c, and b is 20 years old, 
    prove that the total of their ages is 52 years. -/
theorem total_age_proof (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  b = 20 → 
  a + b + c = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_age_proof_l2778_277805


namespace NUMINAMATH_CALUDE_triangle_inequality_l2778_277855

/-- Given a triangle ABC with circumradius R, inradius r, and semiperimeter p,
    prove that 16 R r - 5 r^2 ≤ p^2 ≤ 4 R^2 + 4 R r + 3 r^2 --/
theorem triangle_inequality (R r p : ℝ) (hR : R > 0) (hr : r > 0) (hp : p > 0) :
  16 * R * r - 5 * r^2 ≤ p^2 ∧ p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2778_277855


namespace NUMINAMATH_CALUDE_rubble_purchase_l2778_277896

/-- Calculates the remaining money after a purchase. -/
def remaining_money (initial_amount notebook_cost pen_cost notebook_count pen_count : ℚ) : ℚ :=
  initial_amount - (notebook_cost * notebook_count + pen_cost * pen_count)

/-- Proves that Rubble will have $4.00 left after his purchase. -/
theorem rubble_purchase : 
  let initial_amount : ℚ := 15
  let notebook_cost : ℚ := 4
  let pen_cost : ℚ := 1.5
  let notebook_count : ℚ := 2
  let pen_count : ℚ := 2
  remaining_money initial_amount notebook_cost pen_cost notebook_count pen_count = 4 := by
  sorry

#eval remaining_money 15 4 1.5 2 2

end NUMINAMATH_CALUDE_rubble_purchase_l2778_277896


namespace NUMINAMATH_CALUDE_inverse_function_property_l2778_277861

theorem inverse_function_property (f : ℝ → ℝ) (hf : Function.Bijective f) 
  (h : ∀ x : ℝ, f x + f (1 - x) = 2) :
  ∀ x : ℝ, Function.invFun f (x - 2) + Function.invFun f (4 - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_property_l2778_277861


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l2778_277894

/-- An isosceles triangle with one angle of 94 degrees has a base angle of 43 degrees. -/
theorem isosceles_triangle_base_angle : ∀ (a b c : ℝ),
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Two angles are equal (isosceles property)
  c = 94 →           -- One angle is 94°
  a = 43 :=          -- One of the base angles is 43°
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l2778_277894


namespace NUMINAMATH_CALUDE_student_take_home_pay_l2778_277871

/-- Calculates the take-home pay for a well-performing student at a fast-food chain --/
def takeHomePay (baseSalary bonus taxRate : ℚ) : ℚ :=
  let totalEarnings := baseSalary + bonus
  let taxAmount := totalEarnings * taxRate
  totalEarnings - taxAmount

/-- Theorem stating that the take-home pay for a well-performing student is 26100 rubles --/
theorem student_take_home_pay :
  takeHomePay 25000 5000 (13/100) = 26100 := by
  sorry

#eval takeHomePay 25000 5000 (13/100)

end NUMINAMATH_CALUDE_student_take_home_pay_l2778_277871


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_on_interval_l2778_277866

-- Define the function f(x) = 3x^5 - 5x^3
def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

-- State the theorem
theorem f_monotone_decreasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_on_interval_l2778_277866


namespace NUMINAMATH_CALUDE_right_triangle_existence_l2778_277890

theorem right_triangle_existence (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  c + a = p ∧ c + b = q ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l2778_277890


namespace NUMINAMATH_CALUDE_sqrt_x_minus_9_real_l2778_277863

theorem sqrt_x_minus_9_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 9) ↔ x ≥ 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_9_real_l2778_277863


namespace NUMINAMATH_CALUDE_jennifer_spending_l2778_277830

theorem jennifer_spending (initial_amount : ℚ) : 
  initial_amount - (initial_amount / 5 + initial_amount / 6 + initial_amount / 2) = 24 →
  initial_amount = 180 := by
sorry

end NUMINAMATH_CALUDE_jennifer_spending_l2778_277830
