import Mathlib

namespace NUMINAMATH_CALUDE_forty_percent_of_three_fifths_of_150_forty_percent_of_three_fifths_of_150_equals_36_l1926_192676

theorem forty_percent_of_three_fifths_of_150 : ℚ :=
  let number : ℚ := 150
  let three_fifths : ℚ := 3 / 5
  let forty_percent : ℚ := 40 / 100
  forty_percent * (three_fifths * number)
  
-- Prove that the above expression equals 36
theorem forty_percent_of_three_fifths_of_150_equals_36 :
  forty_percent_of_three_fifths_of_150 = 36 := by sorry

end NUMINAMATH_CALUDE_forty_percent_of_three_fifths_of_150_forty_percent_of_three_fifths_of_150_equals_36_l1926_192676


namespace NUMINAMATH_CALUDE_dice_sum_product_l1926_192603

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 120 →
  a + b + c + d ≠ 14 :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_product_l1926_192603


namespace NUMINAMATH_CALUDE_binomial_square_condition_l1926_192641

/-- If 9x^2 - 24x + a is the square of a binomial, then a = 16 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ p q : ℝ, ∀ x, 9*x^2 - 24*x + a = (p*x + q)^2) → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l1926_192641


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_parallelism_l1926_192639

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity_parallelism
  (m n : Line) (α β : Plane)
  (h_different_lines : m ≠ n)
  (h_different_planes : α ≠ β)
  (h_m_perp_α : perpendicular m α)
  (h_n_parallel_β : parallel_line_plane n β)
  (h_α_parallel_β : parallel_plane α β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_parallelism_l1926_192639


namespace NUMINAMATH_CALUDE_derivative_ln_inverse_sqrt_plus_one_squared_l1926_192657

open Real

theorem derivative_ln_inverse_sqrt_plus_one_squared (x : ℝ) :
  deriv (λ x => Real.log (1 / Real.sqrt (1 + x^2))) x = -x / (1 + x^2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_ln_inverse_sqrt_plus_one_squared_l1926_192657


namespace NUMINAMATH_CALUDE_hexagon_sixth_angle_measure_l1926_192646

/-- The measure of the sixth angle in a hexagon, given the other five angles -/
theorem hexagon_sixth_angle_measure (a b c d e : ℝ) 
  (ha : a = 130)
  (hb : b = 95)
  (hc : c = 122)
  (hd : d = 108)
  (he : e = 114) :
  720 - (a + b + c + d + e) = 151 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_sixth_angle_measure_l1926_192646


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1926_192688

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → m ≤ n) → n + (n + 1) = 43 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1926_192688


namespace NUMINAMATH_CALUDE_sixteen_even_numbers_l1926_192621

/-- Represents a card with two numbers -/
structure Card where
  front : Nat
  back : Nat

/-- Counts the number of three-digit even numbers that can be formed from the given cards -/
def countEvenNumbers (cards : List Card) : Nat :=
  cards.foldl (fun acc card => 
    acc + (if card.front % 2 == 0 then 1 else 0) + 
          (if card.back % 2 == 0 then 1 else 0)
  ) 0

/-- The main theorem stating that 16 different three-digit even numbers can be formed -/
theorem sixteen_even_numbers : 
  let cards := [Card.mk 0 1, Card.mk 2 3, Card.mk 4 5]
  countEvenNumbers cards = 16 := by
  sorry


end NUMINAMATH_CALUDE_sixteen_even_numbers_l1926_192621


namespace NUMINAMATH_CALUDE_duck_flying_days_l1926_192684

/-- The number of days a duck spends flying during winter, summer, and spring -/
def total_flying_days (south_days : ℕ) (east_days : ℕ) : ℕ :=
  south_days + 2 * south_days + east_days

/-- Theorem: The duck spends 180 days flying during winter, summer, and spring -/
theorem duck_flying_days : total_flying_days 40 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_duck_flying_days_l1926_192684


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1926_192627

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1926_192627


namespace NUMINAMATH_CALUDE_basketball_tryouts_l1926_192698

theorem basketball_tryouts (girls : ℕ) (boys : ℕ) (called_back : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : called_back = 26) :
  girls + boys - called_back = 17 := by
sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l1926_192698


namespace NUMINAMATH_CALUDE_klinker_double_age_l1926_192634

/-- Represents the current age of Mr. Klinker -/
def klinker_age : ℕ := 35

/-- Represents the current age of Mr. Klinker's daughter -/
def daughter_age : ℕ := 10

/-- Represents the number of years until Mr. Klinker is twice as old as his daughter -/
def years_until_double : ℕ := 15

/-- Proves that in 15 years, Mr. Klinker will be twice as old as his daughter -/
theorem klinker_double_age :
  klinker_age + years_until_double = 2 * (daughter_age + years_until_double) :=
by sorry

end NUMINAMATH_CALUDE_klinker_double_age_l1926_192634


namespace NUMINAMATH_CALUDE_no_intersection_absolute_value_graphs_l1926_192666

theorem no_intersection_absolute_value_graphs : 
  ∀ x : ℝ, ¬(|3 * x + 6| = -|4 * x - 1|) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_absolute_value_graphs_l1926_192666


namespace NUMINAMATH_CALUDE_average_age_of_five_students_l1926_192628

theorem average_age_of_five_students
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat)
  (avg_age_group1 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 20)
  (h2 : avg_age_all = 20)
  (h3 : num_group1 = 9)
  (h4 : avg_age_group1 = 16)
  (h5 : age_last_student = 186)
  : ∃ (avg_age_group2 : ℝ),
    avg_age_group2 = 14 ∧
    avg_age_group2 * (total_students - num_group1 - 1) =
      total_students * avg_age_all - num_group1 * avg_age_group1 - age_last_student :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_five_students_l1926_192628


namespace NUMINAMATH_CALUDE_no_decagon_partition_l1926_192605

/-- A partition of a polygon into triangles -/
structure TrianglePartition (n : ℕ) where
  black_sides : ℕ
  white_sides : ℕ
  adjacent_diff_color : Prop
  decagon_sides_black : Prop

/-- The theorem stating that a decagon cannot be partitioned in the specified manner -/
theorem no_decagon_partition : ¬ ∃ (p : TrianglePartition 10),
  p.black_sides % 3 = 0 ∧ 
  p.white_sides % 3 = 0 ∧
  p.black_sides - p.white_sides = 10 :=
sorry

end NUMINAMATH_CALUDE_no_decagon_partition_l1926_192605


namespace NUMINAMATH_CALUDE_initial_price_increase_l1926_192642

theorem initial_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.25 = 1.4375 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_price_increase_l1926_192642


namespace NUMINAMATH_CALUDE_lorry_weight_is_1800_l1926_192694

/-- The total weight of a fully loaded lorry -/
def lorry_weight (empty_weight : ℕ) (apple_bags : ℕ) (apple_weight : ℕ) 
  (orange_bags : ℕ) (orange_weight : ℕ) (watermelon_crates : ℕ) (watermelon_weight : ℕ)
  (firewood_bundles : ℕ) (firewood_weight : ℕ) : ℕ :=
  empty_weight + 
  apple_bags * apple_weight + 
  orange_bags * orange_weight + 
  watermelon_crates * watermelon_weight + 
  firewood_bundles * firewood_weight

/-- Theorem stating the total weight of the fully loaded lorry is 1800 pounds -/
theorem lorry_weight_is_1800 : 
  lorry_weight 500 10 55 5 45 3 125 2 75 = 1800 := by
  sorry

#eval lorry_weight 500 10 55 5 45 3 125 2 75

end NUMINAMATH_CALUDE_lorry_weight_is_1800_l1926_192694


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l1926_192689

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a|

-- State the theorem
theorem monotonic_increasing_interval (a : ℝ) :
  (∀ x ≥ 3, Monotone (fun x => f a x)) ↔ a = -6 :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l1926_192689


namespace NUMINAMATH_CALUDE_delivery_driver_boxes_l1926_192647

/-- Theorem: A delivery driver with 3 stops and 9 boxes per stop has 27 boxes in total. -/
theorem delivery_driver_boxes (stops : ℕ) (boxes_per_stop : ℕ) (h1 : stops = 3) (h2 : boxes_per_stop = 9) :
  stops * boxes_per_stop = 27 := by
  sorry

end NUMINAMATH_CALUDE_delivery_driver_boxes_l1926_192647


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1926_192640

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 4 * (Real.tan α)^2 + Real.tan α - 3 = 0) : Real.sin α = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1926_192640


namespace NUMINAMATH_CALUDE_midpoint_complex_numbers_l1926_192643

theorem midpoint_complex_numbers : 
  let A : ℂ := 1 / (1 + Complex.I)
  let B : ℂ := 1 / (1 - Complex.I)
  let C : ℂ := (A + B) / 2
  C = (1 : ℂ) / 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_complex_numbers_l1926_192643


namespace NUMINAMATH_CALUDE_sunflower_seed_distribution_l1926_192617

theorem sunflower_seed_distribution (total_seeds : ℕ) (num_cans : ℕ) (seeds_per_can : ℕ) 
  (h1 : total_seeds = 54)
  (h2 : num_cans = 9)
  (h3 : total_seeds = num_cans * seeds_per_can) :
  seeds_per_can = 6 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_seed_distribution_l1926_192617


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1926_192604

theorem quadratic_form_sum (x : ℝ) : ∃ (a h k : ℝ), 
  (6 * x^2 + 12 * x + 8 = a * (x - h)^2 + k) ∧ (a + h + k = 9) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1926_192604


namespace NUMINAMATH_CALUDE_largest_five_digit_integer_l1926_192650

def digit_product (n : ℕ) : ℕ := 
  (n.digits 10).prod

def digit_sum (n : ℕ) : ℕ := 
  (n.digits 10).sum

theorem largest_five_digit_integer : 
  ∀ n : ℕ, 
    n ≤ 99999 ∧ 
    n ≥ 10000 ∧ 
    digit_product n = 40320 ∧ 
    digit_sum n < 35 → 
    n ≤ 98764 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_integer_l1926_192650


namespace NUMINAMATH_CALUDE_swimmer_speed_l1926_192618

/-- The speed of a swimmer in still water, given downstream and upstream distances and times. -/
theorem swimmer_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 36)
  (h2 : upstream_distance = 26) (h3 : downstream_time = 2) (h4 : upstream_time = 2) :
  ∃ (speed_still : ℝ), speed_still = 15.5 ∧ 
  downstream_distance / downstream_time = speed_still + (downstream_distance - upstream_distance) / (downstream_time + upstream_time) ∧
  upstream_distance / upstream_time = speed_still - (downstream_distance - upstream_distance) / (downstream_time + upstream_time) :=
by sorry

end NUMINAMATH_CALUDE_swimmer_speed_l1926_192618


namespace NUMINAMATH_CALUDE_right_triangle_pq_length_l1926_192662

/-- Represents a right triangle PQR -/
structure RightTriangle where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  tanQ : ℝ

/-- Theorem: In a right triangle PQR where ∠R = 90°, tan Q = 3/4, and PR = 12, PQ = 9 -/
theorem right_triangle_pq_length 
  (t : RightTriangle) 
  (h1 : t.tanQ = 3 / 4) 
  (h2 : t.PR = 12) : 
  t.PQ = 9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_pq_length_l1926_192662


namespace NUMINAMATH_CALUDE_ice_distribution_proof_l1926_192683

/-- Calculates the number of ice cubes per ice chest after melting --/
def ice_cubes_per_chest (initial_cubes : ℕ) (num_chests : ℕ) (melt_rate : ℕ) (hours : ℕ) : ℕ :=
  let remaining_cubes := initial_cubes - melt_rate * hours
  (remaining_cubes / num_chests : ℕ)

/-- Theorem: Given the initial conditions, each ice chest will contain 39 ice cubes --/
theorem ice_distribution_proof :
  ice_cubes_per_chest 294 7 5 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_ice_distribution_proof_l1926_192683


namespace NUMINAMATH_CALUDE_base_10_to_base_12_153_l1926_192630

def base_12_digit (n : ℕ) : Char :=
  if n < 10 then Char.ofNat (n + 48)
  else if n = 10 then 'A'
  else 'B'

def to_base_12 (n : ℕ) : String :=
  let d₁ := n / 12
  let d₀ := n % 12
  String.mk [base_12_digit d₁, base_12_digit d₀]

theorem base_10_to_base_12_153 :
  to_base_12 153 = "B9" := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_12_153_l1926_192630


namespace NUMINAMATH_CALUDE_room_tiles_count_l1926_192678

/-- Represents the dimensions of a room in centimeters -/
structure RoomDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a room given its dimensions -/
def roomVolume (d : RoomDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Finds the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Calculates the number of cubic tiles needed to fill a room -/
def numTiles (d : RoomDimensions) : ℕ :=
  let tileSize := gcd3 d.length d.width d.height
  roomVolume d / (tileSize * tileSize * tileSize)

/-- The main theorem stating the number of tiles needed for the given room -/
theorem room_tiles_count (room : RoomDimensions) 
    (h1 : room.length = 624)
    (h2 : room.width = 432)
    (h3 : room.height = 356) : 
  numTiles room = 1493952 := by
  sorry

end NUMINAMATH_CALUDE_room_tiles_count_l1926_192678


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l1926_192674

theorem fahrenheit_to_celsius (C F : ℝ) : C = (5/9) * (F - 32) → C = 40 → F = 104 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l1926_192674


namespace NUMINAMATH_CALUDE_jug_emptying_l1926_192673

theorem jug_emptying (Cx Cy Cz : ℝ) (hx : Cx > 0) (hy : Cy > 0) (hz : Cz > 0) :
  let initial_x := (1/4 : ℝ) * Cx
  let initial_y := (2/3 : ℝ) * Cy
  let initial_z := (3/5 : ℝ) * Cz
  let water_to_fill_y := Cy - initial_y
  let remaining_x := initial_x - water_to_fill_y
  remaining_x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_jug_emptying_l1926_192673


namespace NUMINAMATH_CALUDE_insufficient_pharmacies_l1926_192612

/-- Represents a grid of streets -/
structure StreetGrid where
  north_south : Nat
  west_east : Nat

/-- Represents a pharmacy's coverage area -/
structure PharmacyCoverage where
  width : Nat
  height : Nat

/-- Calculates the number of street segments in a grid -/
def streetSegments (grid : StreetGrid) : Nat :=
  2 * (grid.north_south - 1) * grid.west_east

/-- Calculates the number of intersections covered by a single pharmacy -/
def intersectionsCovered (coverage : PharmacyCoverage) : Nat :=
  (coverage.width - 1) * (coverage.height - 1)

/-- Theorem stating that 12 pharmacies are not enough to cover all street segments -/
theorem insufficient_pharmacies
  (grid : StreetGrid)
  (coverage : PharmacyCoverage)
  (h_grid : grid = { north_south := 10, west_east := 10 })
  (h_coverage : coverage = { width := 7, height := 7 })
  (h_pharmacies : Nat := 12) :
  h_pharmacies * intersectionsCovered coverage < streetSegments grid := by
  sorry

end NUMINAMATH_CALUDE_insufficient_pharmacies_l1926_192612


namespace NUMINAMATH_CALUDE_max_running_speed_l1926_192670

/-- The maximum speed at which a person can run to catch a train, given specific conditions -/
theorem max_running_speed (x : ℝ) (h : x > 0) : 
  let v := (30 : ℝ) / 3
  let train_speed := (30 : ℝ)
  let distance_fraction := (1 : ℝ) / 3
  (distance_fraction * x) / v = x / train_speed ∧ 
  ((1 - distance_fraction) * x) / v = (x + (distance_fraction * x)) / train_speed →
  v = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_running_speed_l1926_192670


namespace NUMINAMATH_CALUDE_sum_of_H_and_J_l1926_192699

theorem sum_of_H_and_J : ∃ (H J K L : ℕ),
  H ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  J ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  K ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  L ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  H ≠ J ∧ H ≠ K ∧ H ≠ L ∧ J ≠ K ∧ J ≠ L ∧ K ≠ L ∧
  (H : ℚ) / J - (K : ℚ) / L = 5 / 6 →
  H + J = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_H_and_J_l1926_192699


namespace NUMINAMATH_CALUDE_exactly_three_even_dice_probability_l1926_192602

def num_sides : ℕ := 12
def num_dice : ℕ := 4
def num_even_sides : ℕ := 6

def prob_even_on_one_die : ℚ := num_even_sides / num_sides

theorem exactly_three_even_dice_probability :
  (num_dice.choose 3) * (prob_even_on_one_die ^ 3) * ((1 - prob_even_on_one_die) ^ (num_dice - 3)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_even_dice_probability_l1926_192602


namespace NUMINAMATH_CALUDE_five_eighths_decimal_l1926_192623

theorem five_eighths_decimal : (5 : ℚ) / 8 = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_five_eighths_decimal_l1926_192623


namespace NUMINAMATH_CALUDE_marco_has_largest_number_l1926_192631

def ellen_final (start : ℕ) : ℕ :=
  ((start - 2) * 3) + 4

def marco_final (start : ℕ) : ℕ :=
  ((start * 3) - 3) + 5

def lucia_final (start : ℕ) : ℕ :=
  ((start - 3) + 5) * 3

theorem marco_has_largest_number :
  let ellen_start := 12
  let marco_start := 15
  let lucia_start := 13
  marco_final marco_start > ellen_final ellen_start ∧
  marco_final marco_start > lucia_final lucia_start :=
by sorry

end NUMINAMATH_CALUDE_marco_has_largest_number_l1926_192631


namespace NUMINAMATH_CALUDE_ferris_wheel_ticket_cost_l1926_192686

theorem ferris_wheel_ticket_cost 
  (initial_tickets : ℕ) 
  (remaining_tickets : ℕ) 
  (total_spent : ℕ) 
  (h1 : initial_tickets = 6)
  (h2 : remaining_tickets = 3)
  (h3 : total_spent = 27) :
  total_spent / (initial_tickets - remaining_tickets) = 9 :=
by sorry

end NUMINAMATH_CALUDE_ferris_wheel_ticket_cost_l1926_192686


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1926_192697

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (4 - 5 * x) = 10 → x = -19.2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1926_192697


namespace NUMINAMATH_CALUDE_hotdogs_sold_l1926_192663

theorem hotdogs_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 99 → remaining = 97 → sold = initial - remaining → sold = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotdogs_sold_l1926_192663


namespace NUMINAMATH_CALUDE_sequence_difference_sum_l1926_192691

-- Define the arithmetic sequences
def seq1 : List Nat := List.range 93 |>.map (fun i => i + 1981)
def seq2 : List Nat := List.range 93 |>.map (fun i => i + 201)

-- Define the sum of each sequence
def sum1 : Nat := seq1.sum
def sum2 : Nat := seq2.sum

-- Theorem statement
theorem sequence_difference_sum : sum1 - sum2 = 165540 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_sum_l1926_192691


namespace NUMINAMATH_CALUDE_cos_double_angle_for_tan_two_l1926_192607

theorem cos_double_angle_for_tan_two (θ : Real) (h : Real.tan θ = 2) : 
  Real.cos (2 * θ) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_for_tan_two_l1926_192607


namespace NUMINAMATH_CALUDE_divisibility_condition_l1926_192693

def a (n : ℕ) : ℕ := 3 * 4^n

theorem divisibility_condition (n : ℕ) :
  (∀ m : ℕ, 1992 ∣ (m^(a n + 6) - m^(a n + 4) - m^5 + m^3)) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1926_192693


namespace NUMINAMATH_CALUDE_hcf_problem_l1926_192660

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 17820) (h2 : Nat.lcm a b = 1485) :
  Nat.gcd a b = 12 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l1926_192660


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1926_192615

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, if they are parallel, then x = -4 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (-12, x - 4)
  parallel a b → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1926_192615


namespace NUMINAMATH_CALUDE_smallest_max_sum_l1926_192638

theorem smallest_max_sum (a b c d e : ℕ+) 
  (sum_eq : a + b + c + d + e = 3060)
  (ae_lower_bound : a + e ≥ 1300) :
  let M := max (a + b) (max (b + c) (max (c + d) (d + e)))
  ∀ (a' b' c' d' e' : ℕ+),
    a' + b' + c' + d' + e' = 3060 →
    a' + e' ≥ 1300 →
    max (a' + b') (max (b' + c') (max (c' + d') (d' + e'))) ≥ 1174 :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l1926_192638


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l1926_192690

theorem tax_rate_calculation (tax_rate_percent : ℝ) (base_amount : ℝ) :
  tax_rate_percent = 82 ∧ base_amount = 100 →
  tax_rate_percent / 100 * base_amount = 82 := by
sorry

end NUMINAMATH_CALUDE_tax_rate_calculation_l1926_192690


namespace NUMINAMATH_CALUDE_bowling_tournament_orderings_l1926_192681

/-- Represents a tournament with a fixed number of participants and rounds --/
structure Tournament where
  participants : Nat
  rounds : Nat

/-- Calculates the number of possible orderings in a tournament --/
def possibleOrderings (t : Tournament) : Nat :=
  2 ^ t.rounds

/-- The specific tournament described in the problem --/
def bowlingTournament : Tournament :=
  { participants := 6, rounds := 5 }

/-- Theorem stating that the number of possible orderings in the bowling tournament is 32 --/
theorem bowling_tournament_orderings :
  possibleOrderings bowlingTournament = 32 := by
  sorry

#eval possibleOrderings bowlingTournament

end NUMINAMATH_CALUDE_bowling_tournament_orderings_l1926_192681


namespace NUMINAMATH_CALUDE_f_max_value_l1926_192644

/-- The function f(x) = 3x - x^3 -/
def f (x : ℝ) : ℝ := 3 * x - x^3

/-- The theorem stating that the maximum value of f(x) = 3x - x^3 is 2 for 0 ≤ x ≤ √3 -/
theorem f_max_value :
  ∃ (M : ℝ), M = 2 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 3 → f x ≤ M) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 3 ∧ f x = M) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l1926_192644


namespace NUMINAMATH_CALUDE_least_sum_m_n_l1926_192622

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m.val + n.val) 330 = 1) ∧ 
  (∃ (k : ℕ), m.val ^ m.val = k * (n.val ^ n.val)) ∧ 
  (∀ (l : ℕ), m.val ≠ l * n.val) ∧
  (m.val + n.val = 390) ∧
  (∀ (p q : ℕ+), 
    (Nat.gcd (p.val + q.val) 330 = 1) → 
    (∃ (k : ℕ), p.val ^ p.val = k * (q.val ^ q.val)) → 
    (∀ (l : ℕ), p.val ≠ l * q.val) → 
    (p.val + q.val ≥ 390)) := by
  sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l1926_192622


namespace NUMINAMATH_CALUDE_max_ab_value_l1926_192609

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃! x, x^2 + Real.sqrt a * x - b + 1/4 = 0) → 
  ∀ c, a * b ≤ c → c ≤ 1/16 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l1926_192609


namespace NUMINAMATH_CALUDE_number_of_green_balls_l1926_192653

/-- Given a bag with blue and green balls, prove the number of green balls -/
theorem number_of_green_balls
  (blue_balls : ℕ)
  (total_balls : ℕ)
  (h_blue_balls : blue_balls = 9)
  (h_prob_blue : blue_balls / total_balls = 3 / 10)
  (h_total : total_balls = blue_balls + green_balls)
  (green_balls : ℕ) :
  green_balls = 21 := by
  sorry

#check number_of_green_balls

end NUMINAMATH_CALUDE_number_of_green_balls_l1926_192653


namespace NUMINAMATH_CALUDE_correct_calculation_l1926_192661

theorem correct_calculation (x : ℤ) : 
  x - 749 = 280 → x + 479 = 1508 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1926_192661


namespace NUMINAMATH_CALUDE_power_mod_seven_l1926_192696

theorem power_mod_seven : 3^87 + 5 ≡ 4 [ZMOD 7] := by sorry

end NUMINAMATH_CALUDE_power_mod_seven_l1926_192696


namespace NUMINAMATH_CALUDE_sticker_distribution_count_l1926_192633

/-- The number of ways to partition n identical objects into k or fewer parts -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 9

/-- The number of sheets -/
def num_sheets : ℕ := 3

theorem sticker_distribution_count : 
  partition_count num_stickers num_sheets = 12 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_count_l1926_192633


namespace NUMINAMATH_CALUDE_bus_car_length_ratio_l1926_192632

theorem bus_car_length_ratio : 
  ∀ (red_bus_length orange_car_length yellow_bus_length : ℝ),
  red_bus_length = 4 * orange_car_length →
  red_bus_length = 48 →
  yellow_bus_length = red_bus_length - 6 →
  yellow_bus_length / orange_car_length = 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_bus_car_length_ratio_l1926_192632


namespace NUMINAMATH_CALUDE_rational_equation_solutions_l1926_192656

theorem rational_equation_solutions (a b : ℚ) :
  (∃ x y : ℚ, a * x^2 + b * y^2 = 1) →
  (∀ n : ℕ, ∃ (x₁ y₁ : ℚ) (x₂ y₂ : ℚ), 
    (a * x₁^2 + b * y₁^2 = 1) ∧ 
    (a * x₂^2 + b * y₂^2 = 1) ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solutions_l1926_192656


namespace NUMINAMATH_CALUDE_triangle_side_length_l1926_192649

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 45 * π / 180 →  -- Convert 45° to radians
  C = 105 * π / 180 →  -- Convert 105° to radians
  b = Real.sqrt 2 →
  A + B + C = π →  -- Sum of angles in a triangle
  a * Real.sin B = b * Real.sin A →  -- Law of sines
  c * Real.sin B = b * Real.sin C →  -- Law of sines
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1926_192649


namespace NUMINAMATH_CALUDE_increasing_continuous_function_intermediate_values_l1926_192668

theorem increasing_continuous_function_intermediate_values 
  (f : ℝ → ℝ) (M N : ℝ) :
  (∀ x y, x ∈ Set.Icc 0 2 → y ∈ Set.Icc 0 2 → x < y → f x < f y) →
  ContinuousOn f (Set.Icc 0 2) →
  f 0 = M →
  f 2 = N →
  M > 0 →
  N > 0 →
  (∃ x₁ ∈ Set.Icc 0 2, f x₁ = (M + N) / 2) ∧
  (∃ x₂ ∈ Set.Icc 0 2, f x₂ = Real.sqrt (M * N)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_continuous_function_intermediate_values_l1926_192668


namespace NUMINAMATH_CALUDE_base7_product_l1926_192606

/-- Converts a base 7 number to base 10 -/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a base 10 number to base 7 -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- The problem statement -/
theorem base7_product :
  let a := [1, 2, 3]  -- 321 in base 7 (least significant digit first)
  let b := [3, 1]     -- 13 in base 7 (least significant digit first)
  toBase7 (toBase10 a * toBase10 b) = [3, 0, 5, 4] := by
  sorry

end NUMINAMATH_CALUDE_base7_product_l1926_192606


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1926_192679

theorem sufficient_but_not_necessary (p q : Prop) :
  (p ∧ q → ¬(¬p)) ∧ ¬(¬(¬p) → p ∧ q) := by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1926_192679


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l1926_192636

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_about_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetric_points_difference (a b : ℝ) :
  symmetric_about_x_axis (1, a) (b, 2) → a - b = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l1926_192636


namespace NUMINAMATH_CALUDE_rectangle_length_percentage_l1926_192672

theorem rectangle_length_percentage (area : ℝ) (breadth : ℝ) (length : ℝ) : 
  area = 460 →
  breadth = 20 →
  area = length * breadth →
  (length - breadth) / breadth * 100 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_percentage_l1926_192672


namespace NUMINAMATH_CALUDE_square_sum_value_l1926_192611

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) :
  x^2 + y^2 = 29 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l1926_192611


namespace NUMINAMATH_CALUDE_sequence_length_l1926_192665

theorem sequence_length (m : ℕ+) (a : ℕ → ℝ) 
  (h0 : a 0 = 37)
  (h1 : a 1 = 72)
  (hm : a m = 0)
  (h_rec : ∀ k ∈ Finset.range (m - 1), a (k + 2) = a k - 3 / a (k + 1)) :
  m = 889 := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_l1926_192665


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1926_192685

theorem geometric_progression_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 2^(1/2 : ℝ)) 
  (h₂ : a₂ = 2^(1/4 : ℝ)) 
  (h₃ : a₃ = 2^(1/8 : ℝ)) 
  (h_geo : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) :
  ∃ a₄ : ℝ, a₄ = 2^(1/16 : ℝ) ∧ a₄ = a₃ * (a₃ / a₂) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1926_192685


namespace NUMINAMATH_CALUDE_coke_calories_is_215_l1926_192654

/-- Represents the calorie content of various food items and meals --/
structure CalorieContent where
  cake : ℕ
  chips : ℕ
  breakfast : ℕ
  lunch : ℕ
  dailyLimit : ℕ
  remainingAfterCoke : ℕ

/-- Calculates the calorie content of the coke --/
def cokeCalories (c : CalorieContent) : ℕ :=
  c.dailyLimit - (c.cake + c.chips + c.breakfast + c.lunch) - c.remainingAfterCoke

/-- Theorem stating that the coke has 215 calories --/
theorem coke_calories_is_215 (c : CalorieContent) 
  (h1 : c.cake = 110)
  (h2 : c.chips = 310)
  (h3 : c.breakfast = 560)
  (h4 : c.lunch = 780)
  (h5 : c.dailyLimit = 2500)
  (h6 : c.remainingAfterCoke = 525) :
  cokeCalories c = 215 := by
  sorry

#eval cokeCalories { cake := 110, chips := 310, breakfast := 560, lunch := 780, dailyLimit := 2500, remainingAfterCoke := 525 }

end NUMINAMATH_CALUDE_coke_calories_is_215_l1926_192654


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l1926_192648

/-- Given a hyperbola C₁ and a parabola C₂, prove that the focus of C₂ has coordinates (0, 3/2) -/
theorem parabola_focus_coordinates 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (C₁ : ℝ → ℝ → Prop) 
  (C₂ : ℝ → ℝ → Prop) 
  (h_C₁ : ∀ x y, C₁ x y ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (h_C₂ : ∀ x y, C₂ x y ↔ x^2 = 2 * p * y)
  (h_eccentricity : a^2 + b^2 = 2 * a^2)  -- Eccentricity of C₁ is √2
  (P : ℝ × ℝ) 
  (h_P_on_C₂ : C₂ P.1 P.2)
  (h_tangent_parallel : ∃ (m : ℝ), m = 1 ∨ m = -1 ∧ 
    ∀ x y, C₂ x y → (y - P.2) = m * (x - P.1))
  (F : ℝ × ℝ)
  (h_F_focus : F.1 = 0 ∧ F.2 = p / 2)
  (h_PF_distance : (P.1 - F.1)^2 + (P.2 - F.2)^2 = 9)  -- |PF| = 3
  : F = (0, 3/2) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l1926_192648


namespace NUMINAMATH_CALUDE_triangle_properties_l1926_192664

open Real

theorem triangle_properties (A B C : ℝ) (a b : ℝ) :
  let D := (A + B) / 2
  2 * sin A * cos B + b * sin (2 * A) + 2 * sqrt 3 * a * cos C = 0 →
  2 = 2 →
  sqrt 3 = sqrt 3 →
  C = 2 * π / 3 ∧
  (1/2) * (1/2) * a * 2 * sin C = sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1926_192664


namespace NUMINAMATH_CALUDE_range_of_k_l1926_192610

-- Define set A
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}

-- Define set B
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < k + 1}

-- Define the complement of A in ℝ
def C_R_A : Set ℝ := {x | ¬(x ∈ A)}

-- Theorem statement
theorem range_of_k (k : ℝ) : 
  (C_R_A ∩ B k).Nonempty → 0 < k ∧ k < 3 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_k_l1926_192610


namespace NUMINAMATH_CALUDE_permutations_with_fixed_front_five_people_one_fixed_front_l1926_192659

/-- The number of ways to arrange n people in a line. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line with one specific person always at the front. -/
def permutationsWithFixed (n : ℕ) : ℕ := permutations (n - 1)

theorem permutations_with_fixed_front (n : ℕ) (h : n > 1) :
  permutationsWithFixed n = Nat.factorial (n - 1) := by
  sorry

/-- There are 5 people, and we want to arrange them with one specific person at the front. -/
theorem five_people_one_fixed_front :
  permutationsWithFixed 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_permutations_with_fixed_front_five_people_one_fixed_front_l1926_192659


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_l1926_192625

theorem rationalize_sqrt_five : 
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = 11/4 + (5/4) * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_l1926_192625


namespace NUMINAMATH_CALUDE_cubic_sum_ratio_l1926_192692

theorem cubic_sum_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^6 + y^6 + z^6) / (x*y*z * (x*y + x*z + y*z)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_ratio_l1926_192692


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l1926_192635

/-- The vertex coordinates of the parabola y = 2 - (2x + 1)^2 are (-1/2, 2) -/
theorem parabola_vertex_coordinates :
  let f (x : ℝ) := 2 - (2*x + 1)^2
  ∃ (a b : ℝ), (∀ x, f x ≤ f a) ∧ f a = b ∧ a = -1/2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l1926_192635


namespace NUMINAMATH_CALUDE_max_correct_answers_l1926_192651

/-- Represents a test score --/
structure TestScore where
  total_questions : ℕ
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  score : ℤ

/-- Checks if a TestScore is valid according to the given conditions --/
def is_valid_score (ts : TestScore) : Prop :=
  ts.total_questions = 30 ∧
  ts.correct + ts.incorrect + ts.unanswered = ts.total_questions ∧
  ts.score = 4 * ts.correct - ts.incorrect

/-- Theorem stating the maximum number of correct answers --/
theorem max_correct_answers (ts : TestScore) (h : is_valid_score ts) (score_70 : ts.score = 70) :
  ts.correct ≤ 20 ∧ ∃ (ts' : TestScore), is_valid_score ts' ∧ ts'.score = 70 ∧ ts'.correct = 20 :=
sorry

end NUMINAMATH_CALUDE_max_correct_answers_l1926_192651


namespace NUMINAMATH_CALUDE_triangle_angle_B_l1926_192620

theorem triangle_angle_B (a b : ℝ) (A : ℝ) (h1 : a = 2) (h2 : b = 2 * Real.sqrt 3) (h3 : A = π / 6) :
  ∃ B : ℝ, (B = π / 3 ∨ B = 2 * π / 3) ∧
    a / Real.sin A = b / Real.sin B :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l1926_192620


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1926_192671

theorem sin_2alpha_value (α : Real) (h : Real.sin α + Real.cos α = 1/3) :
  Real.sin (2 * α) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1926_192671


namespace NUMINAMATH_CALUDE_sally_plums_l1926_192682

def plums_picked (melanie dan sally total : ℕ) : Prop :=
  melanie + dan + sally = total

theorem sally_plums :
  ∃ (sally : ℕ), plums_picked 4 9 sally 16 ∧ sally = 3 := by
  sorry

end NUMINAMATH_CALUDE_sally_plums_l1926_192682


namespace NUMINAMATH_CALUDE_computer_table_cost_calculation_l1926_192619

/-- The cost price of the computer table -/
def computer_table_cost : ℝ := 4813.58

/-- The cost price of the office chair -/
def office_chair_cost : ℝ := 5000

/-- The markup percentage -/
def markup_percentage : ℝ := 0.24

/-- The discount percentage -/
def discount_percentage : ℝ := 0.05

/-- The total amount paid by the customer -/
def total_paid : ℝ := 11560

theorem computer_table_cost_calculation :
  let total_before_discount := (1 + markup_percentage) * (computer_table_cost + office_chair_cost)
  (1 - discount_percentage) * total_before_discount = total_paid := by
  sorry

#eval computer_table_cost

end NUMINAMATH_CALUDE_computer_table_cost_calculation_l1926_192619


namespace NUMINAMATH_CALUDE_shoes_price_proof_l1926_192600

theorem shoes_price_proof (total_cost jersey_count shoe_count : ℕ) 
  (jersey_price_ratio : ℚ) (h1 : total_cost = 560) (h2 : jersey_count = 4) 
  (h3 : shoe_count = 6) (h4 : jersey_price_ratio = 1/4) : 
  shoe_count * (total_cost / (shoe_count + jersey_count * jersey_price_ratio)) = 480 := by
  sorry

end NUMINAMATH_CALUDE_shoes_price_proof_l1926_192600


namespace NUMINAMATH_CALUDE_car_speed_proof_l1926_192616

/-- Proves that a car's speed is 112.5 km/h if it takes 2 seconds longer to travel 1 km compared to 120 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v - 1 / 120) * 3600 = 2 ↔ v = 112.5 := by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l1926_192616


namespace NUMINAMATH_CALUDE_domain_of_f_x_squared_l1926_192637

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-1) 3

-- State the theorem
theorem domain_of_f_x_squared 
  (h : ∀ x, x ∈ domain_f_x_plus_1 ↔ f (x + 1) ∈ Set.range f) : 
  (∀ x, f (x^2) ∈ Set.range f ↔ x ∈ Set.Icc (-2) 2) := by
  sorry

end NUMINAMATH_CALUDE_domain_of_f_x_squared_l1926_192637


namespace NUMINAMATH_CALUDE_subset_coloring_existence_l1926_192614

theorem subset_coloring_existence (S : Type) [Fintype S] (h : Fintype.card S = 2002) (N : ℕ) (hN : N ≤ 2^2002) :
  ∃ f : Set S → Bool,
    (∀ A B : Set S, f A = true → f B = true → f (A ∪ B) = true) ∧
    (∀ A B : Set S, f A = false → f B = false → f (A ∪ B) = false) ∧
    (Fintype.card {A : Set S | f A = true} = N) :=
by sorry

end NUMINAMATH_CALUDE_subset_coloring_existence_l1926_192614


namespace NUMINAMATH_CALUDE_kevin_food_spending_l1926_192601

def total_budget : ℕ := 20
def samuel_ticket : ℕ := 14
def samuel_total : ℕ := 20
def kevin_ticket : ℕ := 14
def kevin_drinks : ℕ := 2

theorem kevin_food_spending :
  ∃ (kevin_food : ℕ),
    kevin_food = total_budget - (kevin_ticket + kevin_drinks) ∧
    kevin_food = 4 :=
by sorry

end NUMINAMATH_CALUDE_kevin_food_spending_l1926_192601


namespace NUMINAMATH_CALUDE_joe_caught_23_times_l1926_192629

/-- The number of times Joe caught the ball -/
def joe_catches : ℕ := 23

/-- The number of times Derek caught the ball -/
def derek_catches (j : ℕ) : ℕ := 2 * j - 4

/-- The number of times Tammy caught the ball -/
def tammy_catches (d : ℕ) : ℕ := d / 3 + 16

theorem joe_caught_23_times :
  joe_catches = 23 ∧
  derek_catches joe_catches = 2 * joe_catches - 4 ∧
  tammy_catches (derek_catches joe_catches) = 30 :=
sorry

end NUMINAMATH_CALUDE_joe_caught_23_times_l1926_192629


namespace NUMINAMATH_CALUDE_proposition_2_l1926_192645

-- Define the basic types
variable (Point : Type)
variable (Line : Type)
variable (Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the proposition we want to prove
theorem proposition_2 
  (m n : Line) (α : Plane) :
  perpendicular_plane m α → parallel m n → perpendicular_plane n α :=
sorry

end NUMINAMATH_CALUDE_proposition_2_l1926_192645


namespace NUMINAMATH_CALUDE_square_difference_l1926_192624

theorem square_difference (n m : ℕ+) (h : n * (4 * n + 1) = m * (5 * m + 1)) :
  ∃ k : ℕ+, n - m = k^2 := by sorry

end NUMINAMATH_CALUDE_square_difference_l1926_192624


namespace NUMINAMATH_CALUDE_ice_cream_arrangements_l1926_192695

theorem ice_cream_arrangements (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangements_l1926_192695


namespace NUMINAMATH_CALUDE_stone_breaking_loss_l1926_192680

/-- Represents the properties of a precious stone -/
structure Stone where
  weight : ℝ
  price : ℝ
  k : ℝ

/-- Calculates the price of a stone given its weight and constant k -/
def calculatePrice (weight : ℝ) (k : ℝ) : ℝ := k * weight^3

/-- Calculates the loss when a stone breaks -/
def calculateLoss (original : Stone) (piece1 : Stone) (piece2 : Stone) : ℝ :=
  original.price - (piece1.price + piece2.price)

theorem stone_breaking_loss (original : Stone) (piece1 : Stone) (piece2 : Stone) :
  original.weight = 28 ∧ 
  original.price = 60000 ∧ 
  original.price = calculatePrice original.weight original.k ∧
  piece1.weight = (17 / 28) * original.weight ∧
  piece2.weight = (11 / 28) * original.weight ∧
  piece1.k = original.k ∧
  piece2.k = original.k ∧
  piece1.price = calculatePrice piece1.weight piece1.k ∧
  piece2.price = calculatePrice piece2.weight piece2.k →
  abs (calculateLoss original piece1 piece2 - 42933.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_stone_breaking_loss_l1926_192680


namespace NUMINAMATH_CALUDE_boxes_with_neither_l1926_192677

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : markers = 8)
  (h3 : erasers = 5)
  (h4 : both = 4)
  : total - (markers + erasers - both) = 6 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l1926_192677


namespace NUMINAMATH_CALUDE_angle_of_inclination_of_line_l1926_192675

theorem angle_of_inclination_of_line (x y : ℝ) :
  x + Real.sqrt 3 * y - 1 = 0 →
  ∃ α : ℝ, α = 5 * π / 6 ∧ Real.tan α = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_of_line_l1926_192675


namespace NUMINAMATH_CALUDE_a_1_greater_than_one_l1926_192608

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem a_1_greater_than_one (seq : ArithmeticSequence)
  (sum_condition : seq.a 1 + seq.a 2 + seq.a 5 > 13)
  (geometric_condition : seq.a 2 ^ 2 = seq.a 1 * seq.a 5) :
  seq.a 1 > 1 := by
  sorry

end NUMINAMATH_CALUDE_a_1_greater_than_one_l1926_192608


namespace NUMINAMATH_CALUDE_operation_twice_equals_twenty_l1926_192687

theorem operation_twice_equals_twenty (v : ℝ) : 
  (v - v / 3) - ((v - v / 3) / 3) = 20 → v = 45 := by
sorry

end NUMINAMATH_CALUDE_operation_twice_equals_twenty_l1926_192687


namespace NUMINAMATH_CALUDE_parallel_tangents_imply_a_value_l1926_192658

/-- Given two curves C₁ and C₂, where C₁ is defined by y = ax³ - 6x² + 12x and C₂ is defined by y = e^x,
    if their tangent lines at x = 1 are parallel, then a = e/3. -/
theorem parallel_tangents_imply_a_value (a : ℝ) : 
  (∀ x : ℝ, (3 * a * x^2 - 12 * x + 12) = Real.exp x) → a = Real.exp 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_imply_a_value_l1926_192658


namespace NUMINAMATH_CALUDE_exists_all_intersecting_segment_l1926_192626

/-- A segment on a line -/
structure Segment where
  left : ℝ
  right : ℝ
  h : left < right

/-- A configuration of segments on a line -/
structure SegmentConfiguration where
  n : ℕ
  segments : Finset Segment
  total_count : segments.card = 2 * n + 1
  intersection_condition : ∀ s ∈ segments, (segments.filter (λ t => s.left < t.right ∧ t.left < s.right)).card ≥ n

/-- There exists a segment that intersects all others -/
theorem exists_all_intersecting_segment (config : SegmentConfiguration) :
  ∃ s ∈ config.segments, ∀ t ∈ config.segments, t ≠ s → s.left < t.right ∧ t.left < s.right :=
sorry

end NUMINAMATH_CALUDE_exists_all_intersecting_segment_l1926_192626


namespace NUMINAMATH_CALUDE_vector_BC_l1926_192613

/-- Given vectors BA and CA in 2D space, prove that vector BC is their difference. -/
theorem vector_BC (BA CA : Fin 2 → ℝ) (h1 : BA = ![1, 2]) (h2 : CA = ![4, 5]) :
  BA - CA = ![-3, -3] := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_l1926_192613


namespace NUMINAMATH_CALUDE_triangle_side_length_l1926_192652

theorem triangle_side_length (side2 side3 perimeter : ℝ) 
  (h1 : side2 = 10)
  (h2 : side3 = 15)
  (h3 : perimeter = 32) :
  ∃ side1 : ℝ, side1 + side2 + side3 = perimeter ∧ side1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1926_192652


namespace NUMINAMATH_CALUDE_base9_addition_l1926_192667

-- Define a function to convert a base 9 number to base 10
def base9ToBase10 (n : List Nat) : Nat :=
  n.foldr (fun digit acc => acc * 9 + digit) 0

-- Define a function to convert a base 10 number to base 9
def base10ToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

-- Define the numbers in base 9
def a : List Nat := [2, 5, 4]
def b : List Nat := [6, 2, 7]
def c : List Nat := [5, 0, 3]

-- Define the expected result in base 9
def result : List Nat := [1, 4, 8, 5]

theorem base9_addition :
  base10ToBase9 (base9ToBase10 a + base9ToBase10 b + base9ToBase10 c) = result :=
sorry

end NUMINAMATH_CALUDE_base9_addition_l1926_192667


namespace NUMINAMATH_CALUDE_sum_of_roots_inequality_l1926_192655

theorem sum_of_roots_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq_one : a + b + c = 1) :
  Real.sqrt ((1 / a) - 1) * Real.sqrt ((1 / b) - 1) +
  Real.sqrt ((1 / b) - 1) * Real.sqrt ((1 / c) - 1) +
  Real.sqrt ((1 / c) - 1) * Real.sqrt ((1 / a) - 1) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_inequality_l1926_192655


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1926_192669

theorem polynomial_factorization :
  ∀ x : ℂ, x^15 + x^10 + 1 = (x^3 - 1) * (x^12 + x^9 + x^6 + x^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1926_192669
