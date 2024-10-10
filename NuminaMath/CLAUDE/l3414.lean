import Mathlib

namespace lcm_factor_problem_l3414_341481

theorem lcm_factor_problem (A B : ℕ+) (hcf other_factor : ℕ+) :
  hcf = 23 →
  A = 345 →
  Nat.lcm A B = hcf * other_factor * 15 →
  other_factor = 23 := by
  sorry

end lcm_factor_problem_l3414_341481


namespace smallest_integer_a_l3414_341420

theorem smallest_integer_a : ∃ (a : ℕ), (∀ (x y : ℝ), x > 0 → y > 0 → x + Real.sqrt (3 * x * y) ≤ a * (x + y)) ∧ 
  (∀ (b : ℕ), b < a → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + Real.sqrt (3 * x * y) > b * (x + y)) ∧ 
  a = 2 := by
  sorry

end smallest_integer_a_l3414_341420


namespace destination_distance_l3414_341440

/-- The distance to the destination in nautical miles -/
def distance : ℝ := sorry

/-- Theon's ship speed in nautical miles per hour -/
def theon_speed : ℝ := 15

/-- Yara's ship speed in nautical miles per hour -/
def yara_speed : ℝ := 30

/-- The time difference between Yara and Theon's arrivals in hours -/
def time_difference : ℝ := 3

theorem destination_distance : 
  distance = 90 ∧ 
  yara_speed = 2 * theon_speed ∧
  distance / yara_speed + time_difference = distance / theon_speed :=
by sorry

end destination_distance_l3414_341440


namespace min_value_of_f_l3414_341485

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 2

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -1 :=
by sorry

end min_value_of_f_l3414_341485


namespace camping_hike_distance_l3414_341455

theorem camping_hike_distance (total_distance stream_to_meadow meadow_to_campsite : ℝ)
  (h1 : total_distance = 0.7)
  (h2 : stream_to_meadow = 0.4)
  (h3 : meadow_to_campsite = 0.1) :
  total_distance - (stream_to_meadow + meadow_to_campsite) = 0.2 := by
  sorry

end camping_hike_distance_l3414_341455


namespace count_satisfying_pairs_l3414_341416

def satisfies_inequalities (a b : ℤ) : Prop :=
  (a^2 + b^2 < 25) ∧ ((a - 3)^2 + b^2 < 20) ∧ (a^2 + (b - 3)^2 < 20)

theorem count_satisfying_pairs :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ satisfies_inequalities p.1 p.2) ∧
    s.card = 7 :=
sorry

end count_satisfying_pairs_l3414_341416


namespace total_crayons_l3414_341467

theorem total_crayons (people : ℕ) (crayons_per_person : ℕ) (h1 : people = 3) (h2 : crayons_per_person = 8) :
  people * crayons_per_person = 24 := by
  sorry

end total_crayons_l3414_341467


namespace triangle_properties_l3414_341453

/-- Given a triangle ABC with the following properties:
  - a = 2√2
  - sin C = √2 * sin A
  - cos C = √2/4
  Prove that:
  - c = 4
  - The area of the triangle is 2√7
-/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  a = 2 * Real.sqrt 2 →
  Real.sin C = Real.sqrt 2 * Real.sin A →
  Real.cos C = Real.sqrt 2 / 4 →
  c = 4 ∧
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 7 :=
by
  sorry

end triangle_properties_l3414_341453


namespace point_translation_rotation_l3414_341480

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translate (p : Point) (dx : ℝ) : Point :=
  ⟨p.x + dx, p.y⟩

/-- Rotates a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : Point) : Point :=
  ⟨p.y, -p.x⟩

theorem point_translation_rotation (p : Point) :
  p = ⟨-5, 4⟩ →
  (rotate90Clockwise (translate p 8)) = ⟨4, -3⟩ := by
  sorry

end point_translation_rotation_l3414_341480


namespace circle_radius_with_chord_l3414_341476

/-- The radius of a circle given specific conditions --/
theorem circle_radius_with_chord (r : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    -- Line equation
    (A.1 - Real.sqrt 3 * A.2 + 8 = 0) ∧ 
    (B.1 - Real.sqrt 3 * B.2 + 8 = 0) ∧
    -- Circle equation
    (A.1^2 + A.2^2 = r^2) ∧ 
    (B.1^2 + B.2^2 = r^2) ∧
    -- Length of chord AB
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 36)) → 
  r = 5 := by
sorry


end circle_radius_with_chord_l3414_341476


namespace cow_hen_problem_l3414_341491

theorem cow_hen_problem (cows hens : ℕ) : 
  4 * cows + 2 * hens = 2 * (cows + hens) + 8 → cows = 4 := by
  sorry

end cow_hen_problem_l3414_341491


namespace function_root_implies_m_range_l3414_341448

theorem function_root_implies_m_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ 2 * m * x + 4 = 0) → 
  (m ≤ -2 ∨ m ≥ 1) := by
  sorry

end function_root_implies_m_range_l3414_341448


namespace battery_collection_theorem_l3414_341468

/-- Represents the number of batteries collected by students. -/
structure BatteryCollection where
  jiajia : ℕ
  qiqi : ℕ

/-- Represents the state of battery collection before and after the exchange. -/
structure BatteryExchange where
  initial : BatteryCollection
  final : BatteryCollection

/-- Theorem about battery collection and exchange between Jiajia and Qiqi. -/
theorem battery_collection_theorem (m : ℕ) :
  ∃ (exchange : BatteryExchange),
    -- Initial conditions
    exchange.initial.jiajia = m ∧
    exchange.initial.qiqi = 2 * m - 2 ∧
    -- Condition that Qiqi would have twice as many if she collected two more
    exchange.initial.qiqi + 2 = 2 * exchange.initial.jiajia ∧
    -- Final conditions after Qiqi gives two batteries to Jiajia
    exchange.final.jiajia = exchange.initial.jiajia + 2 ∧
    exchange.final.qiqi = exchange.initial.qiqi - 2 ∧
    -- Prove that Qiqi has m - 6 more batteries than Jiajia after the exchange
    exchange.final.qiqi - exchange.final.jiajia = m - 6 :=
by
  sorry

end battery_collection_theorem_l3414_341468


namespace symmetric_point_example_l3414_341425

/-- Given a point P in a Cartesian coordinate system, this function returns its symmetric point with respect to the origin. -/
def symmetricPoint (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

/-- Theorem stating that the symmetric point of (2, -3) with respect to the origin is (-2, 3). -/
theorem symmetric_point_example : symmetricPoint (2, -3) = (-2, 3) := by
  sorry

end symmetric_point_example_l3414_341425


namespace massager_vibration_increase_l3414_341415

/-- Given a massager with a lowest setting of 1600 vibrations per second
    and a highest setting that produces 768,000 vibrations in 5 minutes,
    prove that the percentage increase from lowest to highest setting is 60% -/
theorem massager_vibration_increase (lowest : ℝ) (highest_total : ℝ) (duration : ℝ) :
  lowest = 1600 →
  highest_total = 768000 →
  duration = 5 * 60 →
  (highest_total / duration - lowest) / lowest * 100 = 60 := by
sorry

end massager_vibration_increase_l3414_341415


namespace periodic_function_value_l3414_341454

-- Define a periodic function with period 2
def isPeriodic2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f x

-- Theorem statement
theorem periodic_function_value (f : ℝ → ℝ) (h1 : isPeriodic2 f) (h2 : f 2 = 2) :
  f 4 = 2 := by
  sorry

end periodic_function_value_l3414_341454


namespace remainder_three_to_ninth_mod_five_l3414_341406

theorem remainder_three_to_ninth_mod_five : 3^9 ≡ 3 [MOD 5] := by
  sorry

end remainder_three_to_ninth_mod_five_l3414_341406


namespace cube_volume_from_surface_area_l3414_341458

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 96 → volume = 64 := by
  sorry

end cube_volume_from_surface_area_l3414_341458


namespace students_not_in_chorus_or_band_l3414_341461

theorem students_not_in_chorus_or_band 
  (total : ℕ) (chorus : ℕ) (band : ℕ) (both : ℕ) 
  (h1 : total = 50)
  (h2 : chorus = 18)
  (h3 : band = 26)
  (h4 : both = 2) :
  total - (chorus + band - both) = 8 := by
  sorry

end students_not_in_chorus_or_band_l3414_341461


namespace chord_bisected_by_point_4_2_l3414_341482

/-- The equation of an ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- A point is the midpoint of two other points -/
def is_midpoint (x y x1 y1 x2 y2 : ℝ) : Prop :=
  x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

/-- A point lies on a line -/
def point_on_line (x y : ℝ) : Prop := x + 2*y - 8 = 0

theorem chord_bisected_by_point_4_2 (x1 y1 x2 y2 : ℝ) :
  is_on_ellipse x1 y1 →
  is_on_ellipse x2 y2 →
  is_midpoint 4 2 x1 y1 x2 y2 →
  point_on_line x1 y1 ∧ point_on_line x2 y2 :=
sorry

end chord_bisected_by_point_4_2_l3414_341482


namespace integer_roots_of_cubic_polynomial_l3414_341404

theorem integer_roots_of_cubic_polynomial (a₂ a₁ : ℤ) :
  ∀ r : ℤ, r^3 + a₂ * r^2 + a₁ * r + 24 = 0 → r ∣ 24 := by
sorry

end integer_roots_of_cubic_polynomial_l3414_341404


namespace k_max_is_closest_to_expected_l3414_341407

/-- The probability of rolling a one on a fair die -/
def p : ℚ := 1 / 6

/-- The number of times the die is tossed -/
def n : ℕ := 20

/-- The expected number of ones when tossing a fair die n times -/
def expected_ones : ℚ := n * p

/-- The probability of rolling k ones in n tosses of a fair die -/
noncomputable def P (k : ℕ) : ℝ := sorry

/-- The value of k that maximizes P(k) -/
noncomputable def k_max : ℕ := sorry

/-- Theorem stating that k_max is the integer closest to the expected number of ones -/
theorem k_max_is_closest_to_expected : 
  k_max = round expected_ones := by sorry

end k_max_is_closest_to_expected_l3414_341407


namespace B_power_15_minus_3_power_14_l3414_341459

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 2; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 16384; 0, -8192] := by sorry

end B_power_15_minus_3_power_14_l3414_341459


namespace power_sum_inequality_l3414_341493

theorem power_sum_inequality (k l m : ℕ) :
  2^(k+l) + 2^(k+m) + 2^(l+m) ≤ 2^(k+l+m+1) + 1 := by
  sorry

end power_sum_inequality_l3414_341493


namespace symmetric_lines_line_symmetry_l3414_341447

/-- Given two lines in a plane and a point, this theorem states that these lines are symmetric with respect to the given point. -/
theorem symmetric_lines (l₁ l₂ : ℝ → ℝ) (M : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, l₁ y = x → l₂ (4 - y) = 6 - x

/-- The main theorem proving that y = 3x - 17 is symmetric to y = 3x + 3 with respect to (3, 2) -/
theorem line_symmetry : 
  symmetric_lines (λ y => (y + 17) / 3) (λ y => (y - 3) / 3) (3, 2) := by
  sorry

end symmetric_lines_line_symmetry_l3414_341447


namespace election_winner_percentage_l3414_341449

/-- Given an election with two candidates where:
  - The winner received 1054 votes
  - The winner won by 408 votes
Prove that the percentage of votes the winner received is
(1054 / (1054 + (1054 - 408))) * 100 -/
theorem election_winner_percentage (winner_votes : ℕ) (winning_margin : ℕ) :
  winner_votes = 1054 →
  winning_margin = 408 →
  (winner_votes : ℝ) / (winner_votes + (winner_votes - winning_margin)) * 100 =
    1054 / 1700 * 100 :=
by sorry

end election_winner_percentage_l3414_341449


namespace largest_common_value_l3414_341492

def arithmetic_progression_1 (n : ℕ) : ℕ := 4 + 5 * n
def arithmetic_progression_2 (n : ℕ) : ℕ := 5 + 8 * n

theorem largest_common_value :
  ∃ (k : ℕ), 
    (∃ (n m : ℕ), arithmetic_progression_1 n = arithmetic_progression_2 m ∧ arithmetic_progression_1 n = k) ∧
    k < 1000 ∧
    (∀ (l : ℕ), l < 1000 → 
      (∃ (p q : ℕ), arithmetic_progression_1 p = arithmetic_progression_2 q ∧ arithmetic_progression_1 p = l) →
      l ≤ k) ∧
    k = 989 :=
by sorry

end largest_common_value_l3414_341492


namespace solve_for_q_l3414_341411

theorem solve_for_q (p q : ℝ) (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 1) (h4 : p*q = 9) :
  q = (9 + 3*Real.sqrt 5) / 2 := by
  sorry

end solve_for_q_l3414_341411


namespace not_pythagorean_triple_l3414_341498

/-- Checks if a triple of natural numbers forms a Pythagorean triple --/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem not_pythagorean_triple : 
  ¬(isPythagoreanTriple 15 8 19) ∧ 
  (isPythagoreanTriple 6 8 10) ∧ 
  (isPythagoreanTriple 5 12 13) ∧ 
  (isPythagoreanTriple 3 5 4) := by
  sorry

end not_pythagorean_triple_l3414_341498


namespace job_farm_reserved_land_l3414_341443

/-- Represents the land allocation of a farm in hectares -/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  cattle : ℕ
  crops : ℕ

/-- Calculates the land reserved for future expansion -/
def reserved_land (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.cattle + farm.crops)

/-- Theorem stating that the reserved land for Job's farm is 15 hectares -/
theorem job_farm_reserved_land :
  let job_farm : FarmLand := {
    total := 150,
    house_and_machinery := 25,
    cattle := 40,
    crops := 70
  }
  reserved_land job_farm = 15 := by
  sorry


end job_farm_reserved_land_l3414_341443


namespace stratified_sampling_middle_schools_l3414_341409

theorem stratified_sampling_middle_schools 
  (total_schools : ℕ) 
  (high_schools : ℕ) 
  (middle_schools : ℕ) 
  (elementary_schools : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_schools = high_schools + middle_schools + elementary_schools)
  (h2 : total_schools = 100)
  (h3 : high_schools = 10)
  (h4 : middle_schools = 30)
  (h5 : elementary_schools = 60)
  (h6 : sample_size = 20) :
  (middle_schools : ℚ) * sample_size / total_schools = 6 := by
sorry

end stratified_sampling_middle_schools_l3414_341409


namespace inequality_representation_l3414_341436

theorem inequality_representation (x y : ℝ) : 
  abs x + abs y ≤ Real.sqrt (2 * (x^2 + y^2)) ∧ 
  Real.sqrt (2 * (x^2 + y^2)) ≤ 2 * max (abs x) (abs y) := by
  sorry

end inequality_representation_l3414_341436


namespace negation_of_absolute_value_non_negative_l3414_341410

theorem negation_of_absolute_value_non_negative :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x : ℝ, |x| < 0) := by sorry

end negation_of_absolute_value_non_negative_l3414_341410


namespace circle_center_radius_sum_l3414_341438

def circle_equation (x y : ℝ) : Prop :=
  x^2 + 2*x - 4*y - 7 = -y^2 + 8*x

def center_and_radius_sum (c d s : ℝ) : ℝ :=
  c + d + s

theorem circle_center_radius_sum :
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    center_and_radius_sum c d s = 5 + 2 * Real.sqrt 5 :=
sorry

end circle_center_radius_sum_l3414_341438


namespace steve_nickels_l3414_341469

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The total value of coins in cents -/
def total_value : ℕ := 70

/-- Proves that Steve is holding 2 nickels -/
theorem steve_nickels :
  ∃ (n : ℕ), 
    (n * nickel_value + (n + 4) * dime_value = total_value) ∧
    (n = 2) := by
  sorry

end steve_nickels_l3414_341469


namespace box_number_equation_l3414_341464

theorem box_number_equation (x : ℝ) : 
  (x > 0 ∧ 8 + 7 / x + 3 / 1000 = 8.073) ↔ x = 100 := by
sorry

end box_number_equation_l3414_341464


namespace largest_divisor_of_difference_of_cubes_squared_l3414_341477

theorem largest_divisor_of_difference_of_cubes_squared (k : ℤ) : 
  ∃ (d : ℤ), d = 16 ∧ 
  d ∣ (((2*k+1)^3)^2 - ((2*k-1)^3)^2) ∧ 
  ∀ (n : ℤ), n > d → ¬(∀ (j : ℤ), n ∣ (((2*j+1)^3)^2 - ((2*j-1)^3)^2)) :=
sorry

end largest_divisor_of_difference_of_cubes_squared_l3414_341477


namespace no_rearranged_powers_of_two_l3414_341490

-- Define a function to check if a number is a power of 2
def isPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

-- Define a function to check if two numbers have the same digits
def haveSameDigits (m n : ℕ) : Prop :=
  ∃ (digits : List ℕ) (perm : List ℕ), 
    digits.length > 0 ∧
    perm.isPerm digits ∧
    m = digits.foldl (fun acc d => acc * 10 + d) 0 ∧
    n = perm.foldl (fun acc d => acc * 10 + d) 0 ∧
    perm.head? ≠ some 0

theorem no_rearranged_powers_of_two :
  ¬∃ (m n : ℕ), m ≠ n ∧ m > 0 ∧ n > 0 ∧ 
  isPowerOfTwo m ∧ isPowerOfTwo n ∧ 
  haveSameDigits m n :=
sorry

end no_rearranged_powers_of_two_l3414_341490


namespace smallest_difference_is_six_l3414_341487

/-- The set of available digits --/
def digits : Finset Nat := {0, 1, 2, 6, 9}

/-- A function to create a three-digit number from three digits --/
def makeThreeDigitNumber (x y z : Nat) : Nat := 100 * x + 10 * y + z

/-- A function to create a two-digit number from two digits --/
def makeTwoDigitNumber (u v : Nat) : Nat := 10 * u + v

/-- The theorem statement --/
theorem smallest_difference_is_six :
  ∀ x y z u v : Nat,
    x ∈ digits → y ∈ digits → z ∈ digits → u ∈ digits → v ∈ digits →
    x ≠ y → x ≠ z → x ≠ u → x ≠ v →
    y ≠ z → y ≠ u → y ≠ v →
    z ≠ u → z ≠ v →
    u ≠ v →
    x ≠ 0 → u ≠ 0 →
    makeThreeDigitNumber x y z - makeTwoDigitNumber u v ≥ 6 :=
by
  sorry

end smallest_difference_is_six_l3414_341487


namespace correct_product_l3414_341421

theorem correct_product (a b : ℕ+) 
  (h1 : (a - 6) * b = 255) 
  (h2 : (a + 10) * b = 335) : 
  a * b = 285 := by sorry

end correct_product_l3414_341421


namespace trig_identity_l3414_341430

theorem trig_identity (α : Real) (h : Real.tan α = 2) : 
  7 * (Real.sin α)^2 + 3 * (Real.cos α)^2 = 31/5 := by
  sorry

end trig_identity_l3414_341430


namespace biathlon_bicycle_distance_l3414_341474

/-- Given a biathlon with specified conditions, prove the distance of the bicycle race. -/
theorem biathlon_bicycle_distance 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (run_distance : ℝ) 
  (run_velocity : ℝ) :
  total_distance = 155 →
  total_time = 6 →
  run_distance = 10 →
  run_velocity = 10 →
  total_distance = run_distance + (total_time - run_distance / run_velocity) * 
    ((total_distance - run_distance) / (total_time - run_distance / run_velocity)) →
  total_distance - run_distance = 145 := by
  sorry

#check biathlon_bicycle_distance

end biathlon_bicycle_distance_l3414_341474


namespace triathlete_average_speed_l3414_341462

/-- The average speed of a triathlete for swimming and running events -/
theorem triathlete_average_speed 
  (swim_speed : ℝ) 
  (run_speed : ℝ) 
  (h1 : swim_speed = 1) 
  (h2 : run_speed = 6) : 
  (2 * swim_speed * run_speed) / (swim_speed + run_speed) = 12 / 7 := by
  sorry

end triathlete_average_speed_l3414_341462


namespace square_area_in_circle_l3414_341499

theorem square_area_in_circle (r : ℝ) (h : r = 10) : 
  let s := r * Real.sqrt 2
  let small_square_side := r / Real.sqrt 2
  let center_distance := s / 2
  2 * center_distance^2 = 100 := by sorry

end square_area_in_circle_l3414_341499


namespace fuel_cost_per_liter_l3414_341405

/-- Calculates the cost per liter of fuel given the conditions of the problem -/
theorem fuel_cost_per_liter 
  (service_cost : ℝ) 
  (minivan_count : ℕ) 
  (truck_count : ℕ) 
  (total_cost : ℝ) 
  (minivan_tank : ℝ) 
  (truck_tank_multiplier : ℝ) :
  service_cost = 2.20 →
  minivan_count = 3 →
  truck_count = 2 →
  total_cost = 347.7 →
  minivan_tank = 65 →
  truck_tank_multiplier = 2.2 →
  (total_cost - (service_cost * (minivan_count + truck_count))) / 
  (minivan_count * minivan_tank + truck_count * (minivan_tank * truck_tank_multiplier)) = 0.70 :=
by sorry

end fuel_cost_per_liter_l3414_341405


namespace min_value_sum_l3414_341445

theorem min_value_sum (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y)
  (h : a / x + b / y = 2) : 
  x + y ≥ (a + b) / 2 + Real.sqrt (a * b) :=
by sorry

end min_value_sum_l3414_341445


namespace complex_power_150_deg_40_l3414_341486

-- Define DeMoivre's Theorem
axiom deMoivre (θ : ℝ) (n : ℕ) : (Complex.exp (θ * Complex.I)) ^ n = Complex.exp (n * θ * Complex.I)

-- Define the problem
theorem complex_power_150_deg_40 :
  (Complex.exp (150 * π / 180 * Complex.I)) ^ 40 = -1/2 - Complex.I * (Real.sqrt 3 / 2) :=
sorry

end complex_power_150_deg_40_l3414_341486


namespace min_moves_for_chess_like_coloring_l3414_341412

/-- Represents a cell in the 5x5 grid -/
inductive Cell
| white
| black

/-- Represents the 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Cell

/-- Checks if two cells are neighbors -/
def are_neighbors (a b : Fin 5 × Fin 5) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

/-- Represents a move (changing colors of two neighboring cells) -/
structure Move where
  cell1 : Fin 5 × Fin 5
  cell2 : Fin 5 × Fin 5
  are_neighbors : are_neighbors cell1 cell2

/-- Applies a move to a grid -/
def apply_move (g : Grid) (m : Move) : Grid :=
  sorry

/-- Checks if a grid has a chess-like coloring -/
def is_chess_like (g : Grid) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem min_moves_for_chess_like_coloring :
  ∃ (moves : List Move),
    moves.length = 12 ∧
    (∀ g : Grid, (∀ i j, g i j = Cell.white) →
      is_chess_like (moves.foldl apply_move g)) ∧
    (∀ (moves' : List Move),
      moves'.length < 12 →
      ¬∃ g : Grid, (∀ i j, g i j = Cell.white) ∧
        is_chess_like (moves'.foldl apply_move g)) :=
  sorry

end min_moves_for_chess_like_coloring_l3414_341412


namespace third_circle_radius_l3414_341417

/-- Two externally tangent circles with a third circle tangent to both and their common external tangent -/
structure TangentCircles where
  /-- Center of the first circle -/
  A : ℝ × ℝ
  /-- Center of the second circle -/
  B : ℝ × ℝ
  /-- Radius of the first circle -/
  r1 : ℝ
  /-- Radius of the second circle -/
  r2 : ℝ
  /-- Radius of the third circle -/
  r3 : ℝ
  /-- The first two circles are externally tangent -/
  externally_tangent : dist A B = r1 + r2
  /-- The third circle is tangent to the first circle -/
  tangent_to_first : ∃ P : ℝ × ℝ, dist P A = r1 + r3 ∧ dist P B = r2 + r3
  /-- The third circle is tangent to the second circle -/
  tangent_to_second : ∃ Q : ℝ × ℝ, dist Q A = r1 + r3 ∧ dist Q B = r2 + r3
  /-- The third circle is tangent to the common external tangent of the first two circles -/
  tangent_to_external : ∃ T : ℝ × ℝ, dist T A = r1 ∧ dist T B = r2 ∧ 
    ∃ C : ℝ × ℝ, dist C A = r1 + r3 ∧ dist C B = r2 + r3 ∧ dist C T = r3

/-- The radius of the third circle is 1 -/
theorem third_circle_radius (tc : TangentCircles) (h1 : tc.r1 = 2) (h2 : tc.r2 = 5) : tc.r3 = 1 := by
  sorry

end third_circle_radius_l3414_341417


namespace quadratic_root_relation_l3414_341478

theorem quadratic_root_relation : ∀ x₁ x₂ : ℝ, 
  x₁^2 - 12*x₁ + 5 = 0 → 
  x₂^2 - 12*x₂ + 5 = 0 → 
  x₁ + x₂ - x₁*x₂ = 7 := by
  sorry

end quadratic_root_relation_l3414_341478


namespace haley_magazines_l3414_341496

theorem haley_magazines (boxes : ℕ) (magazines_per_box : ℕ) 
  (h1 : boxes = 7) (h2 : magazines_per_box = 9) : 
  boxes * magazines_per_box = 63 := by
  sorry

end haley_magazines_l3414_341496


namespace circle_symmetry_about_y_axis_l3414_341433

/-- Given two circles in the xy-plane, this theorem states that they are symmetric about the y-axis
    if and only if their equations are identical when x is replaced by -x in one of them. -/
theorem circle_symmetry_about_y_axis (a b : ℝ) :
  (∀ x y, x^2 + y^2 + a*x = 0 ↔ (-x)^2 + y^2 + b*(-x) = 0) ↔
  a = -b :=
sorry

end circle_symmetry_about_y_axis_l3414_341433


namespace elevator_probability_l3414_341408

/-- The number of floors in the building -/
def num_floors : ℕ := 6

/-- The number of floors where people can exit (excluding ground floor) -/
def exit_floors : ℕ := num_floors - 1

/-- The probability of two people leaving the elevator on different floors -/
def prob_different_floors : ℚ := 4/5

theorem elevator_probability :
  prob_different_floors = 1 - (1 : ℚ) / exit_floors :=
by sorry

end elevator_probability_l3414_341408


namespace quadratic_rewrite_ratio_l3414_341466

/-- Given a quadratic expression of the form ak² + bk + d, 
    rewrite it as c(k + p)² + q and return (c, p, q) -/
def rewrite_quadratic (a b d : ℚ) : ℚ × ℚ × ℚ := sorry

theorem quadratic_rewrite_ratio : 
  let (c, p, q) := rewrite_quadratic 8 (-12) 20
  q / p = -62 / 3 := by sorry

end quadratic_rewrite_ratio_l3414_341466


namespace pta_spending_ratio_l3414_341479

theorem pta_spending_ratio (initial_amount : ℚ) (spent_on_supplies : ℚ) (amount_left : ℚ) 
  (h1 : initial_amount = 400)
  (h2 : amount_left = 150)
  (h3 : amount_left = initial_amount - spent_on_supplies - (initial_amount - spent_on_supplies) / 2) :
  spent_on_supplies = 100 ∧ spent_on_supplies / initial_amount = 1 / 4 := by
  sorry

end pta_spending_ratio_l3414_341479


namespace intersection_line_equation_l3414_341473

/-- Given two lines l₁ and l₂ in the plane, and a line l passing through their
    intersection point and the origin, prove that l has the equation x - 10y = 0. -/
theorem intersection_line_equation :
  let l₁ : ℝ × ℝ → Prop := λ p => 2 * p.1 + p.2 = 3
  let l₂ : ℝ × ℝ → Prop := λ p => p.1 + 4 * p.2 = 2
  let P : ℝ × ℝ := (10/7, 1/7)  -- Intersection point of l₁ and l₂
  let l : ℝ × ℝ → Prop := λ p => p.1 - 10 * p.2 = 0
  (l₁ P ∧ l₂ P) →  -- P is the intersection of l₁ and l₂
  (l (0, 0)) →     -- l passes through the origin
  (l P) →          -- l passes through P
  ∀ p : ℝ × ℝ, (l₁ p ∧ l₂ p) → l p  -- For any point on both l₁ and l₂, it's also on l
  :=
by sorry

end intersection_line_equation_l3414_341473


namespace equation_solution_l3414_341432

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ (1/2) → 
  (((x^2 - 5*x + 4) / (x - 1)) + ((2*x^2 + 7*x - 4) / (2*x - 1)) = 4) → 
  x = 2 := by
sorry

end equation_solution_l3414_341432


namespace similar_triangle_perimeter_l3414_341460

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t2.a = k * t1.a ∧
    t2.b = k * t1.b ∧
    t2.c = k * t1.c

theorem similar_triangle_perimeter 
  (t1 : Triangle) 
  (h1 : t1.isIsosceles)
  (h2 : t1.a = 16 ∧ t1.b = 16 ∧ t1.c = 8)
  (t2 : Triangle)
  (h3 : Triangle.isSimilar t1 t2)
  (h4 : min t2.a (min t2.b t2.c) = 40) :
  t2.perimeter = 200 := by
  sorry


end similar_triangle_perimeter_l3414_341460


namespace trig_identity_l3414_341427

theorem trig_identity (α β : Real) 
  (h1 : Real.cos (α + β) = 1/3)
  (h2 : Real.sin α * Real.sin β = 1/4) :
  (Real.cos α * Real.cos β = 7/12) ∧
  (Real.cos (2*α - 2*β) = 7/18) := by
  sorry

end trig_identity_l3414_341427


namespace four_divides_sum_of_squares_l3414_341434

theorem four_divides_sum_of_squares (a b c : ℕ+) :
  4 ∣ (a^2 + b^2 + c^2) ↔ (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) := by
  sorry

end four_divides_sum_of_squares_l3414_341434


namespace students_allowance_l3414_341497

theorem students_allowance (allowance : ℚ) : 
  (allowance > 0) →
  (3 / 5 * allowance + 1 / 3 * (2 / 5 * allowance) + 60 / 100 = allowance) →
  allowance = 225 / 100 := by
sorry

end students_allowance_l3414_341497


namespace harry_apples_l3414_341418

theorem harry_apples (martha_apples : ℕ) (tim_less : ℕ) (harry_ratio : ℕ) :
  martha_apples = 68 →
  tim_less = 30 →
  harry_ratio = 2 →
  (martha_apples - tim_less) / harry_ratio = 19 :=
by
  sorry

end harry_apples_l3414_341418


namespace unoccupied_chair_fraction_l3414_341426

theorem unoccupied_chair_fraction :
  let total_chairs : ℕ := 40
  let chair_capacity : ℕ := 2
  let total_members : ℕ := total_chairs * chair_capacity
  let attending_members : ℕ := 48
  let unoccupied_chairs : ℕ := (total_members - attending_members) / chair_capacity
  (unoccupied_chairs : ℚ) / total_chairs = 2 / 5 :=
by sorry

end unoccupied_chair_fraction_l3414_341426


namespace problem_solution_l3414_341483

theorem problem_solution (x y z : ℝ) 
  (h : y^2 + |x - 2023| + Real.sqrt (z - 4) = 6*y - 9) : 
  (y - z)^x = -1 := by
  sorry

end problem_solution_l3414_341483


namespace gasoline_price_increase_l3414_341450

theorem gasoline_price_increase (original_price original_quantity : ℝ) 
  (h1 : original_price > 0) (h2 : original_quantity > 0) : 
  ∃ (price_increase : ℝ),
    (original_price * (1 + price_increase / 100) * (original_quantity * 0.95) = 
     original_price * original_quantity * 1.14) ∧ 
    (price_increase = 20) := by
  sorry

end gasoline_price_increase_l3414_341450


namespace miami_hurricane_damage_l3414_341494

/-- Calculates the damage amount in Euros given the damage in US dollars and the exchange rate. -/
def damage_in_euros (damage_usd : ℝ) (exchange_rate : ℝ) : ℝ :=
  damage_usd * exchange_rate

/-- Theorem stating that the damage caused by the hurricane in Miami is 40,500,000 Euros. -/
theorem miami_hurricane_damage :
  let damage_usd : ℝ := 45000000
  let exchange_rate : ℝ := 0.9
  damage_in_euros damage_usd exchange_rate = 40500000 := by
  sorry

end miami_hurricane_damage_l3414_341494


namespace grasshopper_cannot_return_l3414_341429

def jump_sequence (n : ℕ) : ℕ := n

theorem grasshopper_cannot_return : 
  ∀ (x₀ y₀ x₂₂₂₂ y₂₂₂₂ : ℤ),
  (x₀ + y₀) % 2 = 0 →
  (∀ n : ℕ, n ≤ 2222 → ∃ xₙ yₙ : ℤ, 
    (xₙ - x₀)^2 + (yₙ - y₀)^2 = (jump_sequence n)^2) →
  (x₂₂₂₂ + y₂₂₂₂) % 2 = 1 :=
by sorry

end grasshopper_cannot_return_l3414_341429


namespace wedge_volume_l3414_341451

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h : ℝ) (α : ℝ) : 
  d = 10 → α = 60 → (d / 2)^2 * h * π / 6 = 250 * π / 6 := by sorry

end wedge_volume_l3414_341451


namespace set_intersection_theorem_l3414_341424

def A : Set ℝ := {x | x - 1 ≥ 0}
def B : Set ℝ := {x | |x| ≤ 2}

theorem set_intersection_theorem : A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end set_intersection_theorem_l3414_341424


namespace rebus_no_solution_l3414_341402

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a four-digit number -/
def FourDigitNumber := Fin 10000

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 100000

/-- Converts a four-digit number to its decimal representation -/
def toDecimal (n : FourDigitNumber) : ℕ := n.val

/-- Converts a five-digit number to its decimal representation -/
def toDecimalFive (n : FiveDigitNumber) : ℕ := n.val

/-- Constructs a four-digit number from individual digits -/
def makeNumber (k u s y : Digit) : FourDigitNumber :=
  ⟨k.val * 1000 + u.val * 100 + s.val * 10 + y.val, by sorry⟩

/-- Constructs a five-digit number from individual digits -/
def makeNumberFive (u k s y u' s' : Digit) : FiveDigitNumber :=
  ⟨u.val * 10000 + k.val * 1000 + s.val * 100 + y.val * 10 + u'.val, by sorry⟩

/-- The main theorem stating that the rebus has no solution -/
theorem rebus_no_solution :
  ¬∃ (k u s y : Digit),
    k ≠ u ∧ k ≠ s ∧ k ≠ y ∧ u ≠ s ∧ u ≠ y ∧ s ≠ y ∧
    toDecimal (makeNumber k u s y) + toDecimal (makeNumber u k s y) =
    toDecimalFive (makeNumberFive u k s y u s) :=
by sorry

end rebus_no_solution_l3414_341402


namespace polynomial_sum_l3414_341495

variable (x y : ℝ)
variable (P : ℝ → ℝ → ℝ)

theorem polynomial_sum (h : ∀ x y, P x y + (x^2 - y^2) = x^2 + y^2) :
  ∀ x y, P x y = 2 * y^2 := by
sorry

end polynomial_sum_l3414_341495


namespace divisibility_arithmetic_progression_l3414_341401

theorem divisibility_arithmetic_progression (K : ℕ) :
  (∃ n : ℕ, K = 30 * n - 1) ↔ (K^K + 1) % 30 = 0 :=
by sorry

end divisibility_arithmetic_progression_l3414_341401


namespace cube_inequality_iff_l3414_341444

theorem cube_inequality_iff (a b : ℝ) : a < b ↔ a^3 < b^3 := by sorry

end cube_inequality_iff_l3414_341444


namespace circle_equation_l3414_341446

/-- Represents a parabola in the form y^2 = 4x -/
def Parabola := { p : ℝ × ℝ | p.2^2 = 4 * p.1 }

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The directrix of the parabola y^2 = 4x -/
def directrix : ℝ → ℝ := fun x ↦ -1

/-- Represents a circle with center (h, k) and radius r -/
def Circle (h k r : ℝ) := { p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - k)^2 = r^2 }

/-- The theorem stating that the circle with the focus as its center and tangent to the directrix
    has the equation (x - 1)^2 + y^2 = 4 -/
theorem circle_equation : 
  ∃ (c : Set (ℝ × ℝ)), c = Circle focus.1 focus.2 2 ∧ 
  (∀ p ∈ c, p.1 ≠ -1) ∧
  (∃ p ∈ c, p.1 = -1) ∧
  c = { p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 4 } :=
sorry

end circle_equation_l3414_341446


namespace sacks_to_eliminate_l3414_341400

/-- The number of sacks containing at least $65536 -/
def sacks_with_target : ℕ := 6

/-- The total number of sacks -/
def total_sacks : ℕ := 30

/-- The desired probability of selecting a sack with at least $65536 -/
def target_probability : ℚ := 2/5

theorem sacks_to_eliminate :
  ∃ (n : ℕ), n ≥ 15 ∧
  (sacks_with_target : ℚ) / (total_sacks - n : ℚ) ≥ target_probability ∧
  ∀ (m : ℕ), m < n →
    (sacks_with_target : ℚ) / (total_sacks - m : ℚ) < target_probability :=
by sorry

end sacks_to_eliminate_l3414_341400


namespace soda_price_after_increase_l3414_341414

theorem soda_price_after_increase (candy_price_new : ℝ) (candy_increase : ℝ) (soda_increase : ℝ) (combined_price_old : ℝ) 
  (h1 : candy_price_new = 15)
  (h2 : candy_increase = 0.25)
  (h3 : soda_increase = 0.5)
  (h4 : combined_price_old = 16) :
  ∃ (soda_price_new : ℝ), soda_price_new = 6 := by
  sorry

#check soda_price_after_increase

end soda_price_after_increase_l3414_341414


namespace median_line_property_l3414_341437

-- Define the plane α
variable (α : Plane)

-- Define points A, B, and C
variable (A B C : Point)

-- Define the property of being non-collinear
def NonCollinear (A B C : Point) : Prop := sorry

-- Define the property of a point being outside a plane
def OutsidePlane (P : Point) (π : Plane) : Prop := sorry

-- Define the property of a point being equidistant from a plane
def EquidistantFromPlane (P : Point) (π : Plane) : Prop := sorry

-- Define a median line of a triangle
def MedianLine (M : Line) (A B C : Point) : Prop := sorry

-- Define the property of a line being parallel to a plane
def ParallelToPlane (L : Line) (π : Plane) : Prop := sorry

-- Define the property of a line lying within a plane
def LiesWithinPlane (L : Line) (π : Plane) : Prop := sorry

-- The theorem statement
theorem median_line_property 
  (h1 : NonCollinear A B C)
  (h2 : OutsidePlane A α ∧ OutsidePlane B α ∧ OutsidePlane C α)
  (h3 : EquidistantFromPlane A α ∧ EquidistantFromPlane B α ∧ EquidistantFromPlane C α) :
  ∃ (M : Line), MedianLine M A B C ∧ (ParallelToPlane M α ∨ LiesWithinPlane M α) :=
sorry

end median_line_property_l3414_341437


namespace volleyball_team_size_l3414_341423

/-- The number of people on each team in a volleyball game -/
def peoplePerTeam (managers : ℕ) (employees : ℕ) (teams : ℕ) : ℕ :=
  (managers + employees) / teams

/-- Theorem: In a volleyball game with 3 managers, 3 employees, and 3 teams, there are 2 people per team -/
theorem volleyball_team_size : peoplePerTeam 3 3 3 = 2 := by
  sorry

end volleyball_team_size_l3414_341423


namespace custard_pie_price_per_slice_l3414_341413

/-- The price per slice of custard pie given the number of pies, slices per pie, and total earnings -/
def price_per_slice (num_pies : ℕ) (slices_per_pie : ℕ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (num_pies * slices_per_pie)

/-- Theorem stating that the price per slice of custard pie is $3 under given conditions -/
theorem custard_pie_price_per_slice :
  let num_pies : ℕ := 6
  let slices_per_pie : ℕ := 10
  let total_earnings : ℚ := 180
  price_per_slice num_pies slices_per_pie total_earnings = 3 := by
  sorry

end custard_pie_price_per_slice_l3414_341413


namespace complex_number_quadrant_l3414_341457

theorem complex_number_quadrant : ∃ (a b : ℝ), a < 0 ∧ b < 0 ∧ (1 - I) / (1 + 2*I) = a + b*I := by
  sorry

end complex_number_quadrant_l3414_341457


namespace continued_fraction_solution_l3414_341441

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 6 / (2 + 6 / y) ∧ y = 2 + Real.sqrt 7 := by
  sorry

end continued_fraction_solution_l3414_341441


namespace parabola_vertex_l3414_341439

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x - 2)^2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 0)

/-- Theorem: The vertex coordinates of the parabola y = (x-2)² are (2, 0) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end parabola_vertex_l3414_341439


namespace half_to_fourth_power_l3414_341475

theorem half_to_fourth_power : (1/2 : ℚ)^4 = 1/16 := by
  sorry

end half_to_fourth_power_l3414_341475


namespace unique_A_value_l3414_341435

-- Define the ♣ operation
def clubsuit (A B : ℝ) : ℝ := 3 * A + 2 * B^2 + 5

-- Theorem statement
theorem unique_A_value : ∃! A : ℝ, clubsuit A 3 = 73 ∧ A = 50/3 := by
  sorry

end unique_A_value_l3414_341435


namespace line_intersection_area_ratio_l3414_341470

theorem line_intersection_area_ratio (c : ℝ) (h1 : 0 < c) (h2 : c < 6) : 
  let P : ℝ × ℝ := (0, c)
  let Q : ℝ × ℝ := (c, 0)
  let S : ℝ × ℝ := (6, c - 6)
  let area_QRS := (1/2) * (6 - c) * (c - 6)
  let area_QOP := (1/2) * c * c
  area_QRS / area_QOP = 4/25 → c = 30/7 := by
sorry

end line_intersection_area_ratio_l3414_341470


namespace polynomial_equality_l3414_341442

theorem polynomial_equality (a b c d e : ℝ) : 
  (∀ x : ℝ, (3*x + 1)^4 = a*x^4 + b*x^3 + c*x^2 + d*x + e) → 
  a - b + c - d + e = 16 := by
sorry

end polynomial_equality_l3414_341442


namespace S_lower_bound_l3414_341465

/-- The least positive integer S(n) such that S(n) ≡ n (mod 2), S(n) ≥ n, 
    and there are no positive integers k, x₁, x₂, ..., xₖ such that 
    n = x₁ + x₂ + ... + xₖ and S(n) = x₁² + x₂² + ... + xₖ² -/
noncomputable def S (n : ℕ) : ℕ := sorry

/-- S(n) grows at least as fast as c * n^(3/2) for some constant c > 0 
    and for all sufficiently large n -/
theorem S_lower_bound :
  ∃ (c : ℝ) (n₀ : ℕ), c > 0 ∧ ∀ n ≥ n₀, (S n : ℝ) ≥ c * n^(3/2) := by sorry

end S_lower_bound_l3414_341465


namespace triangle_rotation_path_length_l3414_341419

/-- The length of the path traversed by a vertex of an equilateral triangle rotating inside a square -/
theorem triangle_rotation_path_length 
  (triangle_side : ℝ) 
  (square_side : ℝ) 
  (rotations_per_corner : ℕ) 
  (num_corners : ℕ) 
  (h1 : triangle_side = 3) 
  (h2 : square_side = 6) 
  (h3 : rotations_per_corner = 2) 
  (h4 : num_corners = 4) : 
  (rotations_per_corner * num_corners * triangle_side * (2 * Real.pi / 3)) = 16 * Real.pi :=
sorry

end triangle_rotation_path_length_l3414_341419


namespace spiral_grid_third_row_sum_l3414_341484

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the spiral grid -/
def SpiralGrid :=
  Position → ℕ

/-- The size of the grid -/
def gridSize : ℕ := 17

/-- The center position of the grid -/
def centerPos : Position :=
  { row := 9, col := 9 }

/-- Creates a spiral grid with the given properties -/
def createSpiralGrid : SpiralGrid :=
  sorry

/-- Checks if a position is in the third row from the top -/
def isInThirdRow (p : Position) : Prop :=
  p.row = 3

/-- Finds the greatest number in the third row -/
def greatestInThirdRow (grid : SpiralGrid) : ℕ :=
  sorry

/-- Finds the least number in the third row -/
def leastInThirdRow (grid : SpiralGrid) : ℕ :=
  sorry

theorem spiral_grid_third_row_sum :
  let grid := createSpiralGrid
  greatestInThirdRow grid + leastInThirdRow grid = 528 := by
  sorry

end spiral_grid_third_row_sum_l3414_341484


namespace inequality_system_solution_set_l3414_341452

theorem inequality_system_solution_set :
  let S := {x : ℝ | x - 3 < 0 ∧ x + 1 ≥ 0}
  S = {x : ℝ | -1 ≤ x ∧ x < 3} := by
  sorry

end inequality_system_solution_set_l3414_341452


namespace new_pyramid_volume_l3414_341428

/-- Represents the volume change of a pyramid -/
def pyramid_volume_change (initial_volume : ℝ) (length_scale : ℝ) (width_scale : ℝ) (height_scale : ℝ) : ℝ :=
  initial_volume * length_scale * width_scale * height_scale

/-- Theorem: New volume of the pyramid after scaling -/
theorem new_pyramid_volume :
  let initial_volume : ℝ := 100
  let length_scale : ℝ := 3
  let width_scale : ℝ := 2
  let height_scale : ℝ := 1.2
  pyramid_volume_change initial_volume length_scale width_scale height_scale = 720 := by
  sorry


end new_pyramid_volume_l3414_341428


namespace elevation_change_proof_l3414_341422

def initial_elevation : ℝ := 400

def stage1_rate : ℝ := 10
def stage1_time : ℝ := 5

def stage2_rate : ℝ := 15
def stage2_time : ℝ := 3

def stage3_rate : ℝ := 12
def stage3_time : ℝ := 6

def stage4_rate : ℝ := 8
def stage4_time : ℝ := 4

def stage5_rate : ℝ := 5
def stage5_time : ℝ := 2

def final_elevation : ℝ := initial_elevation - 
  (stage1_rate * stage1_time + 
   stage2_rate * stage2_time + 
   stage3_rate * stage3_time - 
   stage4_rate * stage4_time + 
   stage5_rate * stage5_time)

theorem elevation_change_proof : final_elevation = 255 := by sorry

end elevation_change_proof_l3414_341422


namespace nigella_commission_rate_l3414_341488

/-- Represents a realtor's earnings and house sales --/
structure RealtorSales where
  baseSalary : ℕ
  totalEarnings : ℕ
  houseACost : ℕ
  houseBCost : ℕ
  houseCCost : ℕ

/-- Calculates the commission rate for a realtor given their sales data --/
def commissionRate (sales : RealtorSales) : ℚ :=
  let totalHouseCost := sales.houseACost + sales.houseBCost + sales.houseCCost
  let commission := sales.totalEarnings - sales.baseSalary
  (commission : ℚ) / totalHouseCost

/-- Theorem stating that given the conditions from the problem, the commission rate is 2% --/
theorem nigella_commission_rate :
  let sales : RealtorSales := {
    baseSalary := 3000,
    totalEarnings := 8000,
    houseACost := 60000,
    houseBCost := 3 * 60000,
    houseCCost := 2 * 60000 - 110000
  }
  commissionRate sales = 1/50 := by sorry

end nigella_commission_rate_l3414_341488


namespace grid_has_ten_rows_l3414_341403

/-- Represents a grid of colored squares. -/
structure ColoredGrid where
  squares_per_row : ℕ
  red_squares : ℕ
  blue_squares : ℕ
  green_squares : ℕ

/-- Calculates the number of rows in the grid. -/
def number_of_rows (grid : ColoredGrid) : ℕ :=
  (grid.red_squares + grid.blue_squares + grid.green_squares) / grid.squares_per_row

/-- Theorem stating that a grid with the given properties has 10 rows. -/
theorem grid_has_ten_rows (grid : ColoredGrid) 
  (h1 : grid.squares_per_row = 15)
  (h2 : grid.red_squares = 24)
  (h3 : grid.blue_squares = 60)
  (h4 : grid.green_squares = 66) : 
  number_of_rows grid = 10 := by
  sorry

end grid_has_ten_rows_l3414_341403


namespace computer_upgrade_cost_l3414_341489

/-- Calculates the total money spent on a computer after replacing a video card -/
def totalSpent (initialCost oldCardSale newCardPrice : ℕ) : ℕ :=
  initialCost + (newCardPrice - oldCardSale)

theorem computer_upgrade_cost :
  totalSpent 1200 300 500 = 1400 := by
  sorry

end computer_upgrade_cost_l3414_341489


namespace ellipse_condition_l3414_341431

/-- Represents a curve defined by the equation ax^2 + by^2 = 1 -/
structure Curve where
  a : ℝ
  b : ℝ

/-- Predicate to check if a curve is an ellipse -/
def is_ellipse (c : Curve) : Prop :=
  c.a > 0 ∧ c.b > 0 ∧ c.a ≠ c.b

theorem ellipse_condition (c : Curve) :
  (is_ellipse c → c.a > 0 ∧ c.b > 0) ∧
  (∃ c : Curve, c.a > 0 ∧ c.b > 0 ∧ ¬is_ellipse c) :=
sorry

end ellipse_condition_l3414_341431


namespace arithmetic_sequence_ratio_sum_l3414_341456

/-- Given two arithmetic sequences {a_n} and {b_n} with the sum of their first n terms
    denoted as (A_n, B_n), where A_n / B_n = (5n + 12) / (2n + 3) for all n,
    prove that a_5 / b_5 + a_7 / b_12 = 30/7. -/
theorem arithmetic_sequence_ratio_sum (a b : ℕ → ℚ) (A B : ℕ → ℚ) :
  (∀ n, A n / B n = (5 * n + 12) / (2 * n + 3)) →
  (∀ n, A n = n * (a 1 + a n) / 2) →
  (∀ n, B n = n * (b 1 + b n) / 2) →
  a 5 / b 5 + a 7 / b 12 = 30 / 7 := by
  sorry

end arithmetic_sequence_ratio_sum_l3414_341456


namespace randy_blocks_theorem_l3414_341471

/-- The number of blocks Randy used for the tower -/
def blocks_used : ℕ := 19

/-- The number of blocks Randy has left -/
def blocks_left : ℕ := 59

/-- The initial number of blocks Randy had -/
def initial_blocks : ℕ := blocks_used + blocks_left

theorem randy_blocks_theorem : initial_blocks = 78 := by
  sorry

end randy_blocks_theorem_l3414_341471


namespace cube_diagonal_pairs_l3414_341472

/-- The number of diagonals on the faces of a cube -/
def num_diagonals : ℕ := 12

/-- The total number of pairs of diagonals -/
def total_pairs : ℕ := num_diagonals.choose 2

/-- The number of pairs of diagonals that do not form a 60° angle -/
def non_60_degree_pairs : ℕ := 18

/-- The number of pairs of diagonals that form a 60° angle -/
def pairs_60_degree : ℕ := total_pairs - non_60_degree_pairs

theorem cube_diagonal_pairs :
  pairs_60_degree = 48 := by sorry

end cube_diagonal_pairs_l3414_341472


namespace birds_and_storks_on_fence_l3414_341463

theorem birds_and_storks_on_fence (initial_birds initial_storks additional_birds final_total : ℕ) :
  initial_birds = 3 →
  additional_birds = 5 →
  final_total = 10 →
  initial_birds + initial_storks + additional_birds = final_total →
  initial_storks = 2 := by
  sorry

end birds_and_storks_on_fence_l3414_341463
