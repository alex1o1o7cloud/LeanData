import Mathlib

namespace NUMINAMATH_CALUDE_share_distribution_l1367_136734

theorem share_distribution (total : ℚ) (a b c d : ℚ) 
  (h1 : total = 1000)
  (h2 : a = b + 100)
  (h3 : a = c - 100)
  (h4 : d = b - 50)
  (h5 : d = a + 150)
  (h6 : a + b + c + d = total) :
  a = 212.5 ∧ b = 112.5 ∧ c = 312.5 ∧ d = 362.5 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l1367_136734


namespace NUMINAMATH_CALUDE_calculation_proof_l1367_136721

theorem calculation_proof : (1/2)⁻¹ + (π + 2023)^0 - 2 * Real.cos (π/3) + Real.sqrt 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1367_136721


namespace NUMINAMATH_CALUDE_binary_110101_equals_53_l1367_136701

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101₂ -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

theorem binary_110101_equals_53 : binary_to_decimal binary_110101 = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_equals_53_l1367_136701


namespace NUMINAMATH_CALUDE_largest_package_size_l1367_136715

theorem largest_package_size (anna beatrice carlos : ℕ) 
  (h1 : anna = 60) (h2 : beatrice = 45) (h3 : carlos = 75) :
  Nat.gcd anna (Nat.gcd beatrice carlos) = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l1367_136715


namespace NUMINAMATH_CALUDE_scouts_hike_car_occupancy_l1367_136737

theorem scouts_hike_car_occupancy (cars : ℕ) (taxis : ℕ) (vans : ℕ) 
  (people_per_taxi : ℕ) (people_per_van : ℕ) (total_people : ℕ) :
  cars = 3 →
  taxis = 6 →
  vans = 2 →
  people_per_taxi = 6 →
  people_per_van = 5 →
  total_people = 58 →
  ∃ (people_per_car : ℕ), 
    people_per_car * cars + people_per_taxi * taxis + people_per_van * vans = total_people ∧
    people_per_car = 4 :=
by sorry

end NUMINAMATH_CALUDE_scouts_hike_car_occupancy_l1367_136737


namespace NUMINAMATH_CALUDE_grocery_store_soda_l1367_136728

theorem grocery_store_soda (diet_soda : ℕ) (regular_soda : ℕ) : 
  diet_soda = 19 → 
  regular_soda = diet_soda + 41 → 
  regular_soda = 60 := by
sorry

end NUMINAMATH_CALUDE_grocery_store_soda_l1367_136728


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1367_136759

/-- 
Given a quadratic equation x^2 - 4x - a = 0, prove that it has two distinct real roots
if and only if a > -4.
-/
theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x - a = 0 ∧ y^2 - 4*y - a = 0) ↔ a > -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1367_136759


namespace NUMINAMATH_CALUDE_milk_packets_problem_l1367_136722

theorem milk_packets_problem (n : ℕ) 
  (h1 : n > 2)
  (h2 : n * 20 = (n - 2) * 12 + 2 * 32) : 
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_milk_packets_problem_l1367_136722


namespace NUMINAMATH_CALUDE_min_sum_geometric_sequence_l1367_136763

/-- Given a positive geometric sequence {a_n} where a₅ * a₄ * a₂ * a₁ = 16,
    the minimum value of a₁ + a₅ is 4. -/
theorem min_sum_geometric_sequence (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)
  (h_prod : a 5 * a 4 * a 2 * a 1 = 16) :
  ∀ x y, x > 0 ∧ y > 0 ∧ x * y = a 1 * a 5 → x + y ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_geometric_sequence_l1367_136763


namespace NUMINAMATH_CALUDE_square_twelve_y_minus_five_l1367_136753

theorem square_twelve_y_minus_five (y : ℝ) (h : 6 * y^2 + 7 = 4 * y + 13) : 
  (12 * y - 5)^2 = 161 := by
  sorry

end NUMINAMATH_CALUDE_square_twelve_y_minus_five_l1367_136753


namespace NUMINAMATH_CALUDE_track_length_proof_l1367_136795

/-- The length of the circular track -/
def track_length : ℝ := 600

/-- The distance Brenda runs before the first meeting -/
def brenda_first_distance : ℝ := 120

/-- The additional distance Sally runs between the first and second meeting -/
def sally_additional_distance : ℝ := 180

/-- Theorem stating the length of the track given the meeting conditions -/
theorem track_length_proof :
  ∃ (brenda_speed sally_speed : ℝ),
    brenda_speed > 0 ∧ sally_speed > 0 ∧
    brenda_first_distance / (track_length / 2 - brenda_first_distance) = brenda_speed / sally_speed ∧
    (track_length / 2 - brenda_first_distance + sally_additional_distance) / (brenda_first_distance + track_length / 2 - (track_length / 2 - brenda_first_distance + sally_additional_distance)) = sally_speed / brenda_speed :=
by
  sorry

end NUMINAMATH_CALUDE_track_length_proof_l1367_136795


namespace NUMINAMATH_CALUDE_square_sum_divided_l1367_136711

theorem square_sum_divided : (2005^2 + 2 * 2005 * 1995 + 1995^2) / 800 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_divided_l1367_136711


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1367_136717

-- Define set A
def A : Set ℝ := {x | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1367_136717


namespace NUMINAMATH_CALUDE_intersection_point_existence_l1367_136769

theorem intersection_point_existence : ∃! x₀ : ℝ, x₀ ∈ Set.Ioo 1 2 ∧ x₀^3 = (1/2)^(x₀ - 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_existence_l1367_136769


namespace NUMINAMATH_CALUDE_non_seniors_playing_instrument_l1367_136783

theorem non_seniors_playing_instrument (total_students : ℕ) 
  (senior_play_percent : ℚ) (non_senior_not_play_percent : ℚ) 
  (total_not_play_percent : ℚ) :
  total_students = 500 →
  senior_play_percent = 2/5 →
  non_senior_not_play_percent = 3/10 →
  total_not_play_percent = 234/500 →
  ∃ (seniors non_seniors : ℕ),
    seniors + non_seniors = total_students ∧
    (seniors : ℚ) * (1 - senior_play_percent) + 
    (non_seniors : ℚ) * non_senior_not_play_percent = 
    (total_students : ℚ) * total_not_play_percent ∧
    (non_seniors : ℚ) * (1 - non_senior_not_play_percent) = 154 :=
by sorry

end NUMINAMATH_CALUDE_non_seniors_playing_instrument_l1367_136783


namespace NUMINAMATH_CALUDE_check_mistake_problem_l1367_136750

theorem check_mistake_problem :
  ∃ (x y : ℕ), 
    10 ≤ x ∧ x < 100 ∧
    10 ≤ y ∧ y < 100 ∧
    100 * y + x - (100 * x + y) = 2556 ∧
    (x + y) % 11 = 0 ∧
    x = 9 := by
  sorry

end NUMINAMATH_CALUDE_check_mistake_problem_l1367_136750


namespace NUMINAMATH_CALUDE_routes_on_grid_l1367_136745

/-- The number of routes from A to B on a 3x3 grid -/
def num_routes : ℕ := 20

/-- The size of the grid -/
def grid_size : ℕ := 3

/-- The number of right moves required -/
def right_moves : ℕ := 3

/-- The number of down moves required -/
def down_moves : ℕ := 3

/-- The total number of moves required -/
def total_moves : ℕ := right_moves + down_moves

theorem routes_on_grid : 
  num_routes = (Nat.choose total_moves right_moves) := by
  sorry

end NUMINAMATH_CALUDE_routes_on_grid_l1367_136745


namespace NUMINAMATH_CALUDE_parabola_symmetry_line_l1367_136733

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- The line of symmetry -/
def symmetry_line (x m : ℝ) : ℝ := x + m

/-- Theorem: For a parabola y = 2x² with two points symmetric about y = x + m, 
    and their x-coordinates multiply to -1/2, m equals 3/2 -/
theorem parabola_symmetry_line (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = parabola x₁ →
  y₂ = parabola x₂ →
  (y₁ + y₂) / 2 = symmetry_line ((x₁ + x₂) / 2) m →
  x₁ * x₂ = -1/2 →
  m = 3/2 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_line_l1367_136733


namespace NUMINAMATH_CALUDE_election_vote_majority_l1367_136787

/-- In an election with two candidates, prove the vote majority for the winner. -/
theorem election_vote_majority
  (total_votes : ℕ)
  (winning_percentage : ℚ)
  (h_total : total_votes = 700)
  (h_percentage : winning_percentage = 70 / 100) :
  (winning_percentage * total_votes : ℚ).floor -
  ((1 - winning_percentage) * total_votes : ℚ).floor = 280 := by
  sorry

end NUMINAMATH_CALUDE_election_vote_majority_l1367_136787


namespace NUMINAMATH_CALUDE_min_length_of_rectangle_l1367_136718

theorem min_length_of_rectangle (a : ℝ) (h : a > 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = a^2 → min x y ≥ a :=
by sorry

end NUMINAMATH_CALUDE_min_length_of_rectangle_l1367_136718


namespace NUMINAMATH_CALUDE_quadratic_maximum_l1367_136731

theorem quadratic_maximum (s : ℝ) : -3 * s^2 + 24 * s - 8 ≤ 40 ∧ ∃ s, -3 * s^2 + 24 * s - 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l1367_136731


namespace NUMINAMATH_CALUDE_eleven_million_scientific_notation_l1367_136762

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem eleven_million_scientific_notation :
  toScientificNotation 11000000 = ScientificNotation.mk 1.1 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_eleven_million_scientific_notation_l1367_136762


namespace NUMINAMATH_CALUDE_battery_life_is_19_5_hours_l1367_136785

/-- Represents the tablet's battery and usage characteristics -/
structure TabletBattery where
  passive_life : ℝ  -- Battery life in hours when not actively used
  active_life : ℝ   -- Battery life in hours when actively used
  used_time : ℝ     -- Total time the tablet has been on since last charge
  gaming_time : ℝ   -- Time spent gaming since last charge
  charge_rate_passive : ℝ  -- Additional passive battery life gained per hour of charging
  charge_rate_active : ℝ   -- Additional active battery life gained per hour of charging
  charge_time : ℝ   -- Time spent charging the tablet

/-- Calculates the remaining battery life after usage and charging -/
def remaining_battery_life (tb : TabletBattery) : ℝ :=
  sorry

/-- Theorem stating that the remaining battery life is 19.5 hours -/
theorem battery_life_is_19_5_hours (tb : TabletBattery) 
  (h1 : tb.passive_life = 36)
  (h2 : tb.active_life = 6)
  (h3 : tb.used_time = 15)
  (h4 : tb.gaming_time = 1.5)
  (h5 : tb.charge_rate_passive = 2)
  (h6 : tb.charge_rate_active = 0.5)
  (h7 : tb.charge_time = 3) :
  remaining_battery_life tb = 19.5 :=
sorry

end NUMINAMATH_CALUDE_battery_life_is_19_5_hours_l1367_136785


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l1367_136790

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ a + c > b

theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    a < b ∧ b < c →  -- scalene condition
    is_prime a ∧ is_prime b ∧ is_prime c →  -- prime side lengths
    a = 5 →  -- smallest side is 5
    triangle_inequality a b c →  -- valid triangle
    is_prime (a + b + c) →  -- prime perimeter
    a + b + c ≥ 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l1367_136790


namespace NUMINAMATH_CALUDE_street_lights_per_side_l1367_136754

/-- The number of neighborhoods in the town -/
def num_neighborhoods : ℕ := 10

/-- The number of roads in each neighborhood -/
def roads_per_neighborhood : ℕ := 4

/-- The total number of street lights in the town -/
def total_street_lights : ℕ := 20000

/-- The number of street lights on each opposite side of a road -/
def lights_per_side : ℚ := total_street_lights / (2 * num_neighborhoods * roads_per_neighborhood)

theorem street_lights_per_side :
  lights_per_side = 250 :=
sorry

end NUMINAMATH_CALUDE_street_lights_per_side_l1367_136754


namespace NUMINAMATH_CALUDE_gondor_earnings_l1367_136729

/-- Calculates the total earnings of a technician named Gondor based on his repair work --/
theorem gondor_earnings :
  let phone_repair_fee : ℕ := 10
  let laptop_repair_fee : ℕ := 20
  let phones_monday : ℕ := 3
  let phones_tuesday : ℕ := 5
  let laptops_wednesday : ℕ := 2
  let laptops_thursday : ℕ := 4
  
  let total_phones : ℕ := phones_monday + phones_tuesday
  let total_laptops : ℕ := laptops_wednesday + laptops_thursday
  
  let phone_earnings : ℕ := phone_repair_fee * total_phones
  let laptop_earnings : ℕ := laptop_repair_fee * total_laptops
  
  let total_earnings : ℕ := phone_earnings + laptop_earnings
  
  total_earnings = 200 :=
by
  sorry


end NUMINAMATH_CALUDE_gondor_earnings_l1367_136729


namespace NUMINAMATH_CALUDE_berry_ratio_l1367_136703

theorem berry_ratio (total : ℕ) (blueberries : ℕ) : 
  total = 42 →
  blueberries = 7 →
  (total / 2 : ℚ) = (total - blueberries - (total / 2) : ℚ) →
  (total - blueberries - (total / 2) : ℚ) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_berry_ratio_l1367_136703


namespace NUMINAMATH_CALUDE_remaining_toenail_capacity_l1367_136739

/- Jar capacity in terms of regular toenails -/
def jar_capacity : ℕ := 100

/- Size ratio of big toenails to regular toenails -/
def big_toenail_ratio : ℕ := 2

/- Number of big toenails already in the jar -/
def big_toenails_in_jar : ℕ := 20

/- Number of regular toenails already in the jar -/
def regular_toenails_in_jar : ℕ := 40

/- Theorem: The number of additional regular toenails that can fit in the jar is 20 -/
theorem remaining_toenail_capacity :
  jar_capacity - (big_toenails_in_jar * big_toenail_ratio + regular_toenails_in_jar) = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_toenail_capacity_l1367_136739


namespace NUMINAMATH_CALUDE_base7_subtraction_l1367_136766

/-- Converts a base-7 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 7 * acc) 0

/-- Converts a decimal number to its base-7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The first number in base 7 -/
def num1 : List Nat := [2, 4, 5, 6]

/-- The second number in base 7 -/
def num2 : List Nat := [1, 2, 3, 4]

/-- The expected difference in base 7 -/
def expected_diff : List Nat := [1, 2, 2, 2]

theorem base7_subtraction :
  toBase7 (toDecimal num1 - toDecimal num2) = expected_diff := by
  sorry

end NUMINAMATH_CALUDE_base7_subtraction_l1367_136766


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1367_136786

/-- The parabola equation -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2/3 - x^2 = 1

/-- The directrix of the parabola -/
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

/-- Point F is the focus of the parabola -/
def focus (p : ℝ) (F : ℝ × ℝ) : Prop := F.1 = p/2 ∧ F.2 = 0

/-- Points M and N are the intersections of the directrix and hyperbola -/
def intersection_points (p : ℝ) (M N : ℝ × ℝ) : Prop :=
  directrix p M.1 ∧ hyperbola M.1 M.2 ∧
  directrix p N.1 ∧ hyperbola N.1 N.2

/-- Triangle MNF is a right-angled triangle with F as the right angle vertex -/
def right_triangle (F M N : ℝ × ℝ) : Prop :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2 + (N.1 - F.1)^2 + (N.2 - F.2)^2 =
  (M.1 - N.1)^2 + (M.2 - N.2)^2

theorem parabola_hyperbola_intersection (p : ℝ) (F M N : ℝ × ℝ) :
  parabola p F.1 F.2 →
  focus p F →
  intersection_points p M N →
  right_triangle F M N →
  p = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1367_136786


namespace NUMINAMATH_CALUDE_map_width_l1367_136776

/-- Given a rectangular map with area 10 square meters and length 5 meters, prove its width is 2 meters. -/
theorem map_width (area : ℝ) (length : ℝ) (width : ℝ) 
    (h_area : area = 10) 
    (h_length : length = 5) 
    (h_rectangle : area = length * width) : width = 2 := by
  sorry

end NUMINAMATH_CALUDE_map_width_l1367_136776


namespace NUMINAMATH_CALUDE_percentage_calculation_l1367_136784

theorem percentage_calculation (x : ℝ) (h : 0.255 * x = 153) : 0.678 * x = 406.8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1367_136784


namespace NUMINAMATH_CALUDE_square_room_carpet_area_l1367_136706

theorem square_room_carpet_area (room_side : ℝ) (sq_yard_to_sq_feet : ℝ) : 
  room_side = 9 → sq_yard_to_sq_feet = 9 → (room_side * room_side) / sq_yard_to_sq_feet = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_room_carpet_area_l1367_136706


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l1367_136781

/-- Given two parallel vectors p and q, prove that their sum has a magnitude of √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) :
  p = (2, -3) →
  q.1 = x ∧ q.2 = 6 →
  (∃ (k : ℝ), q = k • p) →
  ‖p + q‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l1367_136781


namespace NUMINAMATH_CALUDE_increasing_interval_ln_minus_x_l1367_136730

/-- The function f(x) = ln x - x is increasing on the interval (0,1] -/
theorem increasing_interval_ln_minus_x : 
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ ≤ 1 → 
  (Real.log x₁ - x₁) < (Real.log x₂ - x₂) := by
  sorry

end NUMINAMATH_CALUDE_increasing_interval_ln_minus_x_l1367_136730


namespace NUMINAMATH_CALUDE_inverse_of_P_l1367_136751

-- Define the original proposition P
def P : Prop → Prop → Prop := λ odd prime => odd → prime

-- Define the inverse proposition
def inverse_prop (p : Prop → Prop → Prop) : Prop → Prop → Prop :=
  λ a b => p b a

-- Theorem stating that the inverse of P is as described
theorem inverse_of_P :
  inverse_prop P = (λ prime odd => prime → odd) :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_P_l1367_136751


namespace NUMINAMATH_CALUDE_trig_identity_simplification_l1367_136735

theorem trig_identity_simplification (x y : ℝ) : 
  Real.sin (x + y) * Real.sin (x - y) - Real.cos (x + y) * Real.cos (x - y) = -Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_simplification_l1367_136735


namespace NUMINAMATH_CALUDE_square_difference_formula_l1367_136702

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 1 / 45) : 
  x^2 - y^2 = 8 / 675 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l1367_136702


namespace NUMINAMATH_CALUDE_simplify_expression_l1367_136797

theorem simplify_expression : 
  (625 : ℝ) ^ (1/4 : ℝ) * (256 : ℝ) ^ (1/2 : ℝ) = 80 :=
by
  have h1 : (625 : ℝ) = 5^4 := by norm_num
  have h2 : (256 : ℝ) = 2^8 := by norm_num
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1367_136797


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1367_136743

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating that the intersection point of lines AB and CD is (4/3, -7/3, 14/3) --/
theorem intersection_of_lines :
  let A : ℝ × ℝ × ℝ := (6, -7, 7)
  let B : ℝ × ℝ × ℝ := (16, -17, 12)
  let C : ℝ × ℝ × ℝ := (0, 3, -6)
  let D : ℝ × ℝ × ℝ := (2, -5, 10)
  intersection_point A B C D = (4/3, -7/3, 14/3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1367_136743


namespace NUMINAMATH_CALUDE_log_2_5_gt_log_2_3_l1367_136738

-- Define log_2 as a strictly increasing function
def log_2 : ℝ → ℝ := sorry

-- Axiom: log_2 is strictly increasing
axiom log_2_strictly_increasing : 
  ∀ x y : ℝ, x > y → log_2 x > log_2 y

-- Theorem to prove
theorem log_2_5_gt_log_2_3 : log_2 5 > log_2 3 := by
  sorry

end NUMINAMATH_CALUDE_log_2_5_gt_log_2_3_l1367_136738


namespace NUMINAMATH_CALUDE_least_five_digit_square_and_cube_l1367_136799

theorem least_five_digit_square_and_cube : 
  (∀ n : ℕ, n < 15625 → ¬(∃ a b : ℕ, n = a^2 ∧ n = b^3 ∧ n ≥ 10000)) ∧ 
  (∃ a b : ℕ, 15625 = a^2 ∧ 15625 = b^3) := by
  sorry

end NUMINAMATH_CALUDE_least_five_digit_square_and_cube_l1367_136799


namespace NUMINAMATH_CALUDE_finite_decimal_fraction_l1367_136712

def is_finite_decimal (n : ℚ) : Prop :=
  ∃ (a b : ℤ), n = a / (2^b * 5^b)

theorem finite_decimal_fraction :
  (is_finite_decimal (9/12)) ∧
  (¬ is_finite_decimal (11/27)) ∧
  (¬ is_finite_decimal (4/7)) ∧
  (¬ is_finite_decimal (8/15)) :=
by sorry

end NUMINAMATH_CALUDE_finite_decimal_fraction_l1367_136712


namespace NUMINAMATH_CALUDE_broker_commission_rate_l1367_136770

theorem broker_commission_rate 
  (initial_rate : ℝ) 
  (slump_percentage : ℝ) 
  (new_rate : ℝ) :
  initial_rate = 0.04 →
  slump_percentage = 0.20000000000000007 →
  new_rate = initial_rate / (1 - slump_percentage) →
  new_rate = 0.05 := by
sorry

end NUMINAMATH_CALUDE_broker_commission_rate_l1367_136770


namespace NUMINAMATH_CALUDE_property_square_footage_l1367_136704

/-- Given a property worth $333,200 and a price of $98 per square foot,
    prove that the total square footage is 3400 square feet. -/
theorem property_square_footage :
  let property_value : ℕ := 333200
  let price_per_sqft : ℕ := 98
  let total_sqft : ℕ := property_value / price_per_sqft
  total_sqft = 3400 := by
  sorry

end NUMINAMATH_CALUDE_property_square_footage_l1367_136704


namespace NUMINAMATH_CALUDE_quadratic_maximum_l1367_136748

theorem quadratic_maximum (s : ℝ) : -7 * s^2 + 56 * s + 20 ≤ 132 ∧ ∃ t : ℝ, -7 * t^2 + 56 * t + 20 = 132 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l1367_136748


namespace NUMINAMATH_CALUDE_b_used_car_for_10_hours_l1367_136723

/-- Represents the car hire scenario -/
structure CarHire where
  totalCost : ℕ
  aHours : ℕ
  cHours : ℕ
  bPaid : ℕ

/-- Calculates the number of hours b used the car -/
def calculateBHours (ch : CarHire) : ℕ :=
  let totalHours := ch.aHours + ch.cHours + (ch.bPaid * (ch.aHours + ch.cHours) / (ch.totalCost - ch.bPaid))
  ch.bPaid * totalHours / ch.totalCost

/-- Theorem stating that given the conditions, b used the car for 10 hours -/
theorem b_used_car_for_10_hours (ch : CarHire)
  (h1 : ch.totalCost = 720)
  (h2 : ch.aHours = 9)
  (h3 : ch.cHours = 13)
  (h4 : ch.bPaid = 225) :
  calculateBHours ch = 10 := by
  sorry

#eval calculateBHours ⟨720, 9, 13, 225⟩

end NUMINAMATH_CALUDE_b_used_car_for_10_hours_l1367_136723


namespace NUMINAMATH_CALUDE_cards_in_boxes_l1367_136713

/-- The number of ways to distribute n distinct objects into k distinct boxes with no box left empty -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 4 cards and 3 boxes -/
def num_cards : ℕ := 4
def num_boxes : ℕ := 3

/-- The theorem to prove -/
theorem cards_in_boxes : distribute num_cards num_boxes = 36 := by sorry

end NUMINAMATH_CALUDE_cards_in_boxes_l1367_136713


namespace NUMINAMATH_CALUDE_star_commutative_star_not_distributive_no_star_identity_star_not_associative_l1367_136760

-- Define the binary operation ⋆
def star (x y : ℝ) : ℝ := x^2 * y^2 + x + y

-- Commutativity
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

-- Non-distributivity
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

-- Non-existence of identity element
theorem no_star_identity : ¬(∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x) := by sorry

-- Non-associativity
theorem star_not_associative : ¬(∀ x y z : ℝ, star (star x y) z = star x (star y z)) := by sorry

end NUMINAMATH_CALUDE_star_commutative_star_not_distributive_no_star_identity_star_not_associative_l1367_136760


namespace NUMINAMATH_CALUDE_non_decreasing_sequence_count_l1367_136796

theorem non_decreasing_sequence_count :
  let max_value : ℕ := 1003
  let seq_length : ℕ := 7
  let sequence_count := Nat.choose 504 seq_length
  ∀ (b : Fin seq_length → ℕ),
    (∀ i j : Fin seq_length, i ≤ j → b i ≤ b j) →
    (∀ i : Fin seq_length, b i ≤ max_value) →
    (∀ i : Fin seq_length, Odd (b i - i.val.succ)) →
    (∃! c : ℕ, c = sequence_count) :=
by sorry

end NUMINAMATH_CALUDE_non_decreasing_sequence_count_l1367_136796


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1367_136780

theorem remainder_divisibility (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1367_136780


namespace NUMINAMATH_CALUDE_C_symmetric_C_area_inequality_C_perimeter_inequality_l1367_136714

-- Define the curve C
def C (a : ℝ) (P : ℝ × ℝ) : Prop :=
  a > 1 ∧ (Real.sqrt ((P.1 + 1)^2 + P.2^2) * Real.sqrt ((P.1 - 1)^2 + P.2^2) = a^2)

-- Define the fixed points
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Theorem for symmetry
theorem C_symmetric (a : ℝ) :
  ∀ P : ℝ × ℝ, C a P ↔ C a (-P.1, -P.2) := by sorry

-- Theorem for area inequality
theorem C_area_inequality (a : ℝ) :
  ∀ P : ℝ × ℝ, C a P → 
    (1/2 * Real.sqrt ((P.1 + 1)^2 + P.2^2) * Real.sqrt ((P.1 - 1)^2 + P.2^2) * 
      Real.sin (Real.arccos ((P.1 + 1) * (P.1 - 1) + P.2^2) / 
        (Real.sqrt ((P.1 + 1)^2 + P.2^2) * Real.sqrt ((P.1 - 1)^2 + P.2^2))))
    ≤ (1/2) * a^2 := by sorry

-- Theorem for perimeter inequality
theorem C_perimeter_inequality (a : ℝ) :
  ∀ P : ℝ × ℝ, C a P → 
    Real.sqrt ((P.1 + 1)^2 + P.2^2) + Real.sqrt ((P.1 - 1)^2 + P.2^2) + 2 ≥ 2*a + 2 := by sorry

end NUMINAMATH_CALUDE_C_symmetric_C_area_inequality_C_perimeter_inequality_l1367_136714


namespace NUMINAMATH_CALUDE_initial_chicken_wings_l1367_136779

theorem initial_chicken_wings 
  (num_friends : ℕ) 
  (additional_wings : ℕ) 
  (wings_per_friend : ℕ) 
  (h1 : num_friends = 4) 
  (h2 : additional_wings = 7) 
  (h3 : wings_per_friend = 4) : 
  num_friends * wings_per_friend - additional_wings = 9 := by
sorry

end NUMINAMATH_CALUDE_initial_chicken_wings_l1367_136779


namespace NUMINAMATH_CALUDE_library_books_count_l1367_136741

/-- The number of shelves in the library -/
def num_shelves : ℕ := 1780

/-- The number of books each shelf can hold -/
def books_per_shelf : ℕ := 8

/-- The total number of books in the library -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem library_books_count : total_books = 14240 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l1367_136741


namespace NUMINAMATH_CALUDE_exists_valid_sequence_l1367_136788

/-- A sequence of natural numbers satisfying the given conditions -/
def ValidSequence (s : List Nat) : Prop :=
  s.length > 10 ∧
  s.sum = 20 ∧
  3 ∉ s ∧
  ∀ i j, i ≤ j → j < s.length → (s.take (j + 1)).drop i ≠ [3]

/-- Theorem stating the existence of a valid sequence -/
theorem exists_valid_sequence : ∃ s : List Nat, ValidSequence s := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_sequence_l1367_136788


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1367_136727

theorem inequality_solution_set :
  ∀ x : ℝ, (1/2: ℝ)^(x - x^2) < Real.log 81 / Real.log 3 ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1367_136727


namespace NUMINAMATH_CALUDE_positive_solutions_x_minus_y_nonnegative_l1367_136707

-- Define the system of linear equations
def system (x y m : ℝ) : Prop :=
  x + y = 3 * m ∧ 2 * x - 3 * y = m + 5

-- Part 1: Positive solutions
theorem positive_solutions (m : ℝ) :
  (∃ x y : ℝ, system x y m ∧ x > 0 ∧ y > 0) → m > 1 := by
  sorry

-- Part 2: x - y ≥ 0
theorem x_minus_y_nonnegative (m : ℝ) :
  (∃ x y : ℝ, system x y m ∧ x - y ≥ 0) → m ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_positive_solutions_x_minus_y_nonnegative_l1367_136707


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1367_136746

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → c = 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1367_136746


namespace NUMINAMATH_CALUDE_relay_race_total_time_l1367_136758

/-- Represents the data for each athlete in the relay race -/
structure AthleteData where
  distance : ℕ
  time : ℕ

/-- Calculates the total time of the relay race given the data for each athlete -/
def relay_race_time (athletes : Vector AthleteData 8) : ℕ :=
  athletes.toList.map (·.time) |>.sum

theorem relay_race_total_time : ∃ (athletes : Vector AthleteData 8),
  (athletes.get 0).distance = 200 ∧ (athletes.get 0).time = 55 ∧
  (athletes.get 1).distance = 300 ∧ (athletes.get 1).time = (athletes.get 0).time + 10 ∧
  (athletes.get 2).distance = 250 ∧ (athletes.get 2).time = (athletes.get 1).time - 15 ∧
  (athletes.get 3).distance = 150 ∧ (athletes.get 3).time = (athletes.get 0).time - 25 ∧
  (athletes.get 4).distance = 400 ∧ (athletes.get 4).time = 80 ∧
  (athletes.get 5).distance = 350 ∧ (athletes.get 5).time = (athletes.get 4).time - 20 ∧
  (athletes.get 6).distance = 275 ∧ (athletes.get 6).time = 70 ∧
  (athletes.get 7).distance = 225 ∧ (athletes.get 7).time = (athletes.get 6).time - 5 ∧
  relay_race_time athletes = 475 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_total_time_l1367_136758


namespace NUMINAMATH_CALUDE_pet_store_cats_l1367_136716

theorem pet_store_cats (siamese_cats : ℕ) (sold_cats : ℕ) (remaining_cats : ℕ) (house_cats : ℕ) : 
  siamese_cats = 13 → 
  sold_cats = 10 → 
  remaining_cats = 8 → 
  siamese_cats + house_cats - sold_cats = remaining_cats → 
  house_cats = 5 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cats_l1367_136716


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_one_implies_fractional_part_l1367_136752

theorem ceiling_floor_difference_one_implies_fractional_part (x : ℝ) :
  ⌈x⌉ - ⌊x⌋ = 1 → 0 < x - ⌊x⌋ ∧ x - ⌊x⌋ < 1 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_one_implies_fractional_part_l1367_136752


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l1367_136708

theorem no_real_roots_for_nonzero_k :
  ∀ k : ℝ, k ≠ 0 → ¬∃ x : ℝ, x^2 + k*x + k^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l1367_136708


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1367_136700

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  a 6 = 1 →
  a 7 = 0.25 →
  a 3 + a 4 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1367_136700


namespace NUMINAMATH_CALUDE_sum_15_terms_eq_56_25_l1367_136756

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  -- The 11th term is 5.25
  a11 : ℝ
  a11_eq : a11 = 5.25
  -- The 7th term is 3.25
  a7 : ℝ
  a7_eq : a7 = 3.25

/-- The sum of the first 15 terms of the arithmetic progression -/
def sum_15_terms (ap : ArithmeticProgression) : ℝ :=
  -- Definition of the sum (to be proved)
  56.25

/-- Theorem stating that the sum of the first 15 terms is 56.25 -/
theorem sum_15_terms_eq_56_25 (ap : ArithmeticProgression) :
  sum_15_terms ap = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_sum_15_terms_eq_56_25_l1367_136756


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1367_136720

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) :
  2 * x + 2 * y = 60 → x * y ≤ 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1367_136720


namespace NUMINAMATH_CALUDE_charlies_garden_min_cost_l1367_136755

/-- Represents a rectangular region in the garden -/
structure Region where
  length : ℝ
  width : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Represents the cost of fertilizer per square meter for each vegetable type -/
structure FertilizerCost where
  lettuce : ℝ
  spinach : ℝ
  carrots : ℝ
  beans : ℝ
  tomatoes : ℝ

/-- The given garden layout -/
def garden_layout : List Region := [
  ⟨3, 1⟩,  -- Upper left
  ⟨4, 2⟩,  -- Lower right
  ⟨6, 2⟩,  -- Upper right
  ⟨2, 3⟩,  -- Middle center
  ⟨5, 4⟩   -- Bottom left
]

/-- The given fertilizer costs -/
def fertilizer_costs : FertilizerCost :=
  { lettuce := 2
  , spinach := 2.5
  , carrots := 3
  , beans := 3.5
  , tomatoes := 4
  }

/-- Calculates the minimum cost of fertilizers for the garden -/
def min_fertilizer_cost (layout : List Region) (costs : FertilizerCost) : ℝ :=
  sorry  -- Proof implementation goes here

/-- Theorem stating that the minimum fertilizer cost for Charlie's garden is $127 -/
theorem charlies_garden_min_cost :
  min_fertilizer_cost garden_layout fertilizer_costs = 127 := by
  sorry  -- Proof goes here

end NUMINAMATH_CALUDE_charlies_garden_min_cost_l1367_136755


namespace NUMINAMATH_CALUDE_time_addition_and_digit_sum_l1367_136778

/-- Represents time in a 12-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  isPM : Bool

/-- Represents a duration of time -/
structure Duration where
  hours : Nat
  minutes : Nat
  seconds : Nat

def addTime (t : Time) (d : Duration) : Time :=
  sorry

def sumDigits (t : Time) : Nat :=
  sorry

theorem time_addition_and_digit_sum :
  let initialTime : Time := ⟨3, 25, 15, true⟩
  let duration : Duration := ⟨137, 59, 59⟩
  let newTime := addTime initialTime duration
  newTime = ⟨9, 25, 14, true⟩ ∧ sumDigits newTime = 21 := by
  sorry

end NUMINAMATH_CALUDE_time_addition_and_digit_sum_l1367_136778


namespace NUMINAMATH_CALUDE_fourDigitNumbers_eq_14_l1367_136749

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of 4-digit numbers formed using digits 2 and 3, where each number must include at least one occurrence of both digits -/
def fourDigitNumbers : ℕ :=
  choose 4 1 + choose 4 2 + choose 4 3

theorem fourDigitNumbers_eq_14 : fourDigitNumbers = 14 := by sorry

end NUMINAMATH_CALUDE_fourDigitNumbers_eq_14_l1367_136749


namespace NUMINAMATH_CALUDE_number_problem_l1367_136742

theorem number_problem (x : ℝ) : 5 * x + 4 = 19 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1367_136742


namespace NUMINAMATH_CALUDE_f_properties_l1367_136793

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := x^3 - (3*(t+1)/2)*x^2 + 3*t*x + 1

theorem f_properties (t : ℝ) (h : t > 0) :
  (∃ (max : ℝ), t = 2 → ∀ x, f t x ≤ max ∧ ∃ y, f t y = max) ∧
  (∃ (a b : ℝ), a < b ∧ a > 0 ∧ b ≤ 1/3 ∧
    (∀ t', a < t' ∧ t' ≤ b →
      ∃ x₀, 0 < x₀ ∧ x₀ < 2 ∧ ∀ x, 0 ≤ x ∧ x ≤ 2 → f t' x₀ ≤ f t' x)) ∧
  (∃ (a b : ℝ), a < b ∧ a > 0 ∧ b ≤ 1/3 ∧
    (∀ t', a < t' ∧ t' ≤ b →
      ∀ x, x ≥ 0 → f t' x ≤ x * Real.exp x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1367_136793


namespace NUMINAMATH_CALUDE_candy_count_l1367_136744

/-- The number of candy pieces Jake had initially -/
def initial_candy : ℕ := 80

/-- The number of candy pieces Jake sold on Monday -/
def monday_sales : ℕ := 15

/-- The number of candy pieces Jake sold on Tuesday -/
def tuesday_sales : ℕ := 58

/-- The number of candy pieces Jake had left on Wednesday -/
def wednesday_left : ℕ := 7

/-- Theorem stating that the initial number of candy pieces equals the sum of pieces sold on Monday and Tuesday plus the pieces left on Wednesday -/
theorem candy_count : initial_candy = monday_sales + tuesday_sales + wednesday_left := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l1367_136744


namespace NUMINAMATH_CALUDE_ruby_height_l1367_136757

/-- Given the heights of various people, prove Ruby's height --/
theorem ruby_height
  (janet_height : ℕ)
  (charlene_height : ℕ)
  (pablo_height : ℕ)
  (ruby_height : ℕ)
  (h1 : janet_height = 62)
  (h2 : charlene_height = 2 * janet_height)
  (h3 : pablo_height = charlene_height + 70)
  (h4 : ruby_height = pablo_height - 2)
  : ruby_height = 192 := by
  sorry

#check ruby_height

end NUMINAMATH_CALUDE_ruby_height_l1367_136757


namespace NUMINAMATH_CALUDE_frog_jumps_equivalence_l1367_136798

/-- Represents a frog's position on an integer line -/
def FrogPosition := ℤ

/-- Represents a configuration of frogs on the line -/
def FrogConfiguration := List FrogPosition

/-- Represents a direction of movement -/
inductive Direction
| Left : Direction
| Right : Direction

/-- Represents a sequence of n moves -/
def MoveSequence (n : ℕ) := Vector Direction n

/-- Predicate to check if a configuration has distinct positions -/
def HasDistinctPositions (config : FrogConfiguration) : Prop :=
  config.Nodup

/-- Function to count valid move sequences -/
def CountValidMoveSequences (n : ℕ) (initialConfig : FrogConfiguration) (dir : Direction) : ℕ :=
  sorry  -- Implementation details omitted

theorem frog_jumps_equivalence 
  (n : ℕ) 
  (initialConfig : FrogConfiguration) 
  (h : HasDistinctPositions initialConfig) :
  CountValidMoveSequences n initialConfig Direction.Right = 
  CountValidMoveSequences n initialConfig Direction.Left :=
sorry

end NUMINAMATH_CALUDE_frog_jumps_equivalence_l1367_136798


namespace NUMINAMATH_CALUDE_equation_solutions_l1367_136768

theorem equation_solutions :
  (∃ x : ℚ, (17/2 : ℚ) * x = (17/2 : ℚ) + x ∧ x = (17/15 : ℚ)) ∧
  (∃ y : ℚ, y / (2/3 : ℚ) = y + (2/3 : ℚ) ∧ y = (4/3 : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1367_136768


namespace NUMINAMATH_CALUDE_oxford_high_school_population_is_1247_l1367_136789

/-- Represents the number of people in Oxford High School -/
def oxford_high_school_population : ℕ :=
  let full_time_teachers : ℕ := 80
  let part_time_teachers : ℕ := 5
  let principal : ℕ := 1
  let vice_principals : ℕ := 3
  let librarians : ℕ := 2
  let guidance_counselors : ℕ := 6
  let other_staff : ℕ := 25
  let classes : ℕ := 40
  let avg_students_per_class : ℕ := 25
  let part_time_students : ℕ := 250

  let full_time_students : ℕ := classes * avg_students_per_class
  let total_staff : ℕ := full_time_teachers + part_time_teachers + principal + 
                         vice_principals + librarians + guidance_counselors + other_staff
  let total_students : ℕ := full_time_students + (part_time_students / 2)

  total_staff + total_students

/-- Theorem stating that the total number of people in Oxford High School is 1247 -/
theorem oxford_high_school_population_is_1247 : 
  oxford_high_school_population = 1247 := by
  sorry

end NUMINAMATH_CALUDE_oxford_high_school_population_is_1247_l1367_136789


namespace NUMINAMATH_CALUDE_four_r_applications_l1367_136764

def r (θ : ℚ) : ℚ := 1 / (1 - θ)

theorem four_r_applications : r (r (r (r 15))) = -1/14 := by
  sorry

end NUMINAMATH_CALUDE_four_r_applications_l1367_136764


namespace NUMINAMATH_CALUDE_calculation_proof_l1367_136705

theorem calculation_proof :
  ((-4)^2 * ((-3/4) + (-5/8)) = -22) ∧
  (-2^2 - (1 - 0.5) * (1/3) * (2 - (-4)^2) = -5/3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1367_136705


namespace NUMINAMATH_CALUDE_pepperoni_coverage_l1367_136740

/-- Represents a circular pizza with pepperoni toppings -/
structure PepperoniPizza where
  pizza_diameter : ℝ
  pepperoni_count : ℕ
  pepperoni_across_diameter : ℕ

/-- Calculates the fraction of pizza covered by pepperoni -/
def fraction_covered (p : PepperoniPizza) : ℚ :=
  sorry

/-- Theorem stating the fraction of pizza covered by pepperoni -/
theorem pepperoni_coverage (p : PepperoniPizza) 
  (h1 : p.pizza_diameter = 18)
  (h2 : p.pepperoni_across_diameter = 9)
  (h3 : p.pepperoni_count = 40) : 
  fraction_covered p = 40 / 81 := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_l1367_136740


namespace NUMINAMATH_CALUDE_range_of_fraction_l1367_136761

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  (∃ (x₁ y₁ : ℝ), 1 ≤ x₁ ∧ x₁ ≤ 4 ∧ 3 ≤ y₁ ∧ y₁ ≤ 6 ∧ x₁ / y₁ = 1/6) ∧
  (∃ (x₂ y₂ : ℝ), 1 ≤ x₂ ∧ x₂ ≤ 4 ∧ 3 ≤ y₂ ∧ y₂ ≤ 6 ∧ x₂ / y₂ = 4/3) ∧
  (∀ (x' y' : ℝ), 1 ≤ x' ∧ x' ≤ 4 → 3 ≤ y' ∧ y' ≤ 6 → 1/6 ≤ x' / y' ∧ x' / y' ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1367_136761


namespace NUMINAMATH_CALUDE_quadratic_vertex_coordinates_l1367_136791

/-- The vertex coordinates of the quadratic function y = 2x^2 - 4x + 5 are (1, 3) -/
theorem quadratic_vertex_coordinates :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4 * x + 5
  ∃ (h k : ℝ), h = 1 ∧ k = 3 ∧ 
    (∀ x : ℝ, f x = 2 * (x - h)^2 + k) ∧
    (∀ x : ℝ, f x ≥ k) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_coordinates_l1367_136791


namespace NUMINAMATH_CALUDE_volcano_theorem_l1367_136773

def volcano_problem (initial_volcanoes : ℕ) (first_explosion_rate : ℚ) 
  (mid_year_explosion_rate : ℚ) (end_year_explosion_rate : ℚ) (intact_volcanoes : ℕ) : Prop :=
  let remaining_after_first := initial_volcanoes - (initial_volcanoes * first_explosion_rate).floor
  let remaining_after_mid := remaining_after_first - (remaining_after_first * mid_year_explosion_rate).floor
  let final_exploded := (remaining_after_mid * end_year_explosion_rate).floor
  initial_volcanoes - intact_volcanoes = 
    (initial_volcanoes * first_explosion_rate).floor + 
    (remaining_after_first * mid_year_explosion_rate).floor + 
    final_exploded

theorem volcano_theorem : 
  volcano_problem 200 (20/100) (40/100) (50/100) 48 := by
  sorry

end NUMINAMATH_CALUDE_volcano_theorem_l1367_136773


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1367_136771

theorem mod_equivalence_unique_solution :
  ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1367_136771


namespace NUMINAMATH_CALUDE_circle_center_on_line_l1367_136772

/-- Given a circle with equation x² + y² - 2ax + 4y - 6 = 0,
    if its center (h, k) satisfies h + 2k + 1 = 0, then a = 3 -/
theorem circle_center_on_line (a : ℝ) :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 - 2*a*x + 4*y - 6 = 0
  let center := fun (h k : ℝ) => ∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + (k+2)^2 + 10)
  let on_line := fun (h k : ℝ) => h + 2*k + 1 = 0
  (∃ h k, center h k ∧ on_line h k) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_on_line_l1367_136772


namespace NUMINAMATH_CALUDE_bills_remaining_money_bills_remaining_money_proof_l1367_136782

/-- Calculates the amount of money Bill is left with after selling fool's gold and paying a fine -/
theorem bills_remaining_money (ounces_sold : ℕ) (price_per_ounce : ℕ) (fine : ℕ) : ℕ :=
  let total_earned := ounces_sold * price_per_ounce
  total_earned - fine

/-- Proves that Bill is left with $22 given the specific conditions -/
theorem bills_remaining_money_proof :
  bills_remaining_money 8 9 50 = 22 := by
  sorry

end NUMINAMATH_CALUDE_bills_remaining_money_bills_remaining_money_proof_l1367_136782


namespace NUMINAMATH_CALUDE_larger_fraction_l1367_136726

theorem larger_fraction (x y : ℚ) (sum_eq : x + y = 7/8) (prod_eq : x * y = 1/4) :
  max x y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_larger_fraction_l1367_136726


namespace NUMINAMATH_CALUDE_final_price_calculation_l1367_136725

/-- Calculates the final price of a set containing coffee, cheesecake, and sandwich -/
theorem final_price_calculation (coffee_price cheesecake_price sandwich_price : ℝ)
  (coffee_discount : ℝ) (additional_discount : ℝ) :
  coffee_price = 6 →
  cheesecake_price = 10 →
  sandwich_price = 8 →
  coffee_discount = 0.25 * coffee_price →
  additional_discount = 3 →
  (coffee_price - coffee_discount + cheesecake_price + sandwich_price) - additional_discount = 19.5 :=
by sorry

end NUMINAMATH_CALUDE_final_price_calculation_l1367_136725


namespace NUMINAMATH_CALUDE_solve_system_for_q_l1367_136724

theorem solve_system_for_q :
  ∀ p q : ℚ,
  5 * p + 3 * q = 7 →
  3 * p + 5 * q = 8 →
  q = 19 / 16 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_system_for_q_l1367_136724


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1367_136777

def A : Set ℝ := {x : ℝ | (2*x - 5)*(x + 3) > 0}
def B : Set ℝ := {1, 2, 3, 4, 5}

theorem complement_A_intersect_B : 
  (Set.compl A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1367_136777


namespace NUMINAMATH_CALUDE_nikita_mistaken_l1367_136732

theorem nikita_mistaken (b s : ℕ) : 
  (9 * b + 4 * s) - (4 * b + 9 * s) ≠ 49 := by
  sorry

end NUMINAMATH_CALUDE_nikita_mistaken_l1367_136732


namespace NUMINAMATH_CALUDE_formula_describes_relationship_l1367_136794

/-- The formula y = 80 - 10x describes the relationship between x and y for a given set of points -/
theorem formula_describes_relationship : ∀ (x y : ℝ), 
  ((x = 0 ∧ y = 80) ∨ 
   (x = 1 ∧ y = 70) ∨ 
   (x = 2 ∧ y = 60) ∨ 
   (x = 3 ∧ y = 50) ∨ 
   (x = 4 ∧ y = 40)) → 
  y = 80 - 10 * x := by
sorry

end NUMINAMATH_CALUDE_formula_describes_relationship_l1367_136794


namespace NUMINAMATH_CALUDE_probability_two_common_books_is_36_105_l1367_136767

def total_books : ℕ := 12
def books_to_choose : ℕ := 4

def probability_two_common_books : ℚ :=
  (Nat.choose total_books 2 * Nat.choose (total_books - 2) 2 * Nat.choose (total_books - 4) 2) /
  (Nat.choose total_books books_to_choose * Nat.choose total_books books_to_choose)

theorem probability_two_common_books_is_36_105 :
  probability_two_common_books = 36 / 105 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_common_books_is_36_105_l1367_136767


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l1367_136709

theorem pentagon_angle_sum (P Q R a b : ℝ) : 
  P = 34 → Q = 82 → R = 30 → 
  (P + Q + (360 - a) + 90 + (120 - b) = 540) → 
  a + b = 146 := by sorry

end NUMINAMATH_CALUDE_pentagon_angle_sum_l1367_136709


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1367_136792

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ (7^n ≡ n^5 [ZMOD 3]) ∧ 
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬(7^m ≡ m^5 [ZMOD 3]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1367_136792


namespace NUMINAMATH_CALUDE_product_sum_fractions_l1367_136747

theorem product_sum_fractions : (3 * 4 * 5) * (1 / 3 + 1 / 4 - 1 / 5) = 23 := by sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l1367_136747


namespace NUMINAMATH_CALUDE_conic_section_classification_l1367_136710

/-- The equation y^4 - 9x^4 = 3y^2 - 4 represents the union of two hyperbolas -/
theorem conic_section_classification (x y : ℝ) :
  (y^4 - 9*x^4 = 3*y^2 - 4) ↔
  ((y^2 - 3*x^2 = 5/2) ∨ (y^2 - 3*x^2 = 1)) :=
sorry

end NUMINAMATH_CALUDE_conic_section_classification_l1367_136710


namespace NUMINAMATH_CALUDE_desired_interest_rate_l1367_136775

/-- Calculate the desired interest rate (dividend yield) for a share -/
theorem desired_interest_rate (face_value : ℝ) (dividend_rate : ℝ) (market_value : ℝ) :
  face_value = 48 →
  dividend_rate = 0.09 →
  market_value = 36.00000000000001 →
  (face_value * dividend_rate) / market_value * 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_desired_interest_rate_l1367_136775


namespace NUMINAMATH_CALUDE_fifteenth_term_of_modified_arithmetic_sequence_l1367_136774

/-- Given an arithmetic sequence with first term 3, second term 15, and third term 27,
    prove that the 15th term is 339 when the common difference is doubled. -/
theorem fifteenth_term_of_modified_arithmetic_sequence :
  ∀ (a : ℕ → ℝ),
    a 1 = 3 →
    a 2 = 15 →
    a 3 = 27 →
    (∀ n : ℕ, a (n + 1) - a n = 2 * (a 2 - a 1)) →
    a 15 = 339 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_modified_arithmetic_sequence_l1367_136774


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1367_136736

theorem square_plus_reciprocal_square (x : ℝ) (h : x^4 + 1/x^4 = 23) : 
  x^2 + 1/x^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1367_136736


namespace NUMINAMATH_CALUDE_partner_A_profit_share_l1367_136765

/-- Calculates the share of profit for partner A in a business venture --/
theorem partner_A_profit_share 
  (initial_investment : ℕ) 
  (a_withdrawal b_withdrawal c_investment : ℕ)
  (total_profit : ℕ) :
  let a_investment_months := initial_investment * 5 + (initial_investment - a_withdrawal) * 7
  let b_investment_months := initial_investment * 5 + (initial_investment - b_withdrawal) * 7
  let c_investment_months := initial_investment * 5 + (initial_investment + c_investment) * 7
  let total_investment_months := a_investment_months + b_investment_months + c_investment_months
  (a_investment_months : ℚ) / total_investment_months * total_profit = 20500 :=
by
  sorry

#check partner_A_profit_share 20000 5000 4000 6000 69900

end NUMINAMATH_CALUDE_partner_A_profit_share_l1367_136765


namespace NUMINAMATH_CALUDE_target_probabilities_l1367_136719

def prob_hit : ℝ := 0.8
def total_shots : ℕ := 4

theorem target_probabilities :
  let prob_miss := 1 - prob_hit
  (1 - prob_miss ^ total_shots = 0.9984) ∧
  (prob_hit ^ 3 * prob_miss * total_shots + prob_hit ^ total_shots = 0.8192) ∧
  (prob_miss ^ total_shots + total_shots * prob_hit * prob_miss ^ 3 = 0.2576) := by
  sorry

#check target_probabilities

end NUMINAMATH_CALUDE_target_probabilities_l1367_136719
