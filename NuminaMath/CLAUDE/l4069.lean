import Mathlib

namespace NUMINAMATH_CALUDE_cafeteria_tile_problem_l4069_406977

theorem cafeteria_tile_problem :
  let current_tiles : ℕ := 630
  let current_area : ℕ := 18
  let new_tile_side : ℕ := 6
  let new_tiles : ℕ := 315
  (current_tiles * current_area = new_tiles * new_tile_side * new_tile_side) :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_tile_problem_l4069_406977


namespace NUMINAMATH_CALUDE_min_tablets_to_extract_l4069_406912

/-- Represents the number of tablets for each medicine type in the box -/
structure TabletCount where
  a : Nat
  b : Nat
  c : Nat

/-- Calculates the minimum number of tablets to extract to guarantee at least two of each type -/
def minTablets (count : TabletCount) : Nat :=
  (count.a - 1) + (count.b - 1) + 2

/-- Theorem stating the minimum number of tablets to extract for the given problem -/
theorem min_tablets_to_extract (box : TabletCount) 
  (ha : box.a = 25) (hb : box.b = 30) (hc : box.c = 20) : 
  minTablets box = 55 := by
  sorry

#eval minTablets { a := 25, b := 30, c := 20 }

end NUMINAMATH_CALUDE_min_tablets_to_extract_l4069_406912


namespace NUMINAMATH_CALUDE_max_probability_at_twenty_l4069_406901

-- Define the total number of bulbs
def total_bulbs : ℕ := 100

-- Define the number of bulbs picked
def bulbs_picked : ℕ := 10

-- Define the number of defective bulbs in the picked sample
def defective_in_sample : ℕ := 2

-- Define the probability function f(n)
def f (n : ℕ) : ℚ :=
  (Nat.choose n defective_in_sample * Nat.choose (total_bulbs - n) (bulbs_picked - defective_in_sample)) /
  Nat.choose total_bulbs bulbs_picked

-- State the theorem
theorem max_probability_at_twenty {n : ℕ} (h1 : 2 ≤ n) (h2 : n ≤ 92) :
  ∀ m : ℕ, 2 ≤ m ∧ m ≤ 92 → f n ≤ f 20 :=
sorry

end NUMINAMATH_CALUDE_max_probability_at_twenty_l4069_406901


namespace NUMINAMATH_CALUDE_ancient_market_prices_l4069_406922

/-- The cost of animals in an ancient market --/
theorem ancient_market_prices :
  -- Define the costs of animals
  ∀ (camel_cost horse_cost ox_cost elephant_cost : ℚ),
  -- Conditions from the problem
  (10 * camel_cost = 24 * horse_cost) →
  (16 * horse_cost = 4 * ox_cost) →
  (6 * ox_cost = 4 * elephant_cost) →
  (10 * elephant_cost = 110000) →
  -- Conclusion: the cost of one camel is 4400
  camel_cost = 4400 := by
  sorry

end NUMINAMATH_CALUDE_ancient_market_prices_l4069_406922


namespace NUMINAMATH_CALUDE_inequality_proof_l4069_406946

theorem inequality_proof (α β : Real) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : 0 < β) (h4 : β < π/2) : 
  1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4069_406946


namespace NUMINAMATH_CALUDE_max_value_theorem_l4069_406996

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b * Real.sqrt 3 + 2 * b * c ≤ 2 ∧ ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 = 1 ∧ 2 * a * b * Real.sqrt 3 + 2 * b * c = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4069_406996


namespace NUMINAMATH_CALUDE_max_value_sum_l4069_406956

theorem max_value_sum (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) :
  ∃ (M : ℝ), M = Real.sqrt 11 ∧ a + b + c ≤ M ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = M := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_l4069_406956


namespace NUMINAMATH_CALUDE_maine_coon_difference_l4069_406949

/-- Represents the number of cats of a specific breed owned by a person -/
structure CatOwnership where
  persian : Nat
  maine_coon : Nat

/-- Represents the total cat ownership of all three people -/
structure TotalCatOwnership where
  jamie : CatOwnership
  gordon : CatOwnership
  hawkeye : CatOwnership

/-- The theorem stating the difference in Maine Coons between Gordon and Jamie -/
theorem maine_coon_difference (total : TotalCatOwnership) : 
  total.jamie.persian = 4 →
  total.jamie.maine_coon = 2 →
  total.gordon.persian = total.jamie.persian / 2 →
  total.hawkeye.persian = 0 →
  total.hawkeye.maine_coon = total.gordon.maine_coon - 1 →
  total.jamie.persian + total.jamie.maine_coon + 
  total.gordon.persian + total.gordon.maine_coon + 
  total.hawkeye.persian + total.hawkeye.maine_coon = 13 →
  total.gordon.maine_coon - total.jamie.maine_coon = 1 := by
  sorry

end NUMINAMATH_CALUDE_maine_coon_difference_l4069_406949


namespace NUMINAMATH_CALUDE_ride_to_total_ratio_l4069_406917

def total_money : ℚ := 30
def dessert_cost : ℚ := 5
def money_left : ℚ := 10

theorem ride_to_total_ratio : 
  (total_money - dessert_cost - money_left) / total_money = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ride_to_total_ratio_l4069_406917


namespace NUMINAMATH_CALUDE_jerry_syrup_time_l4069_406963

/-- Represents the time it takes Jerry to make cherry syrup -/
def make_cherry_syrup (cherries_per_quart : ℕ) (picking_time : ℕ) (picking_amount : ℕ) (syrup_making_time : ℕ) (quarts : ℕ) : ℕ :=
  let picking_rate : ℚ := picking_amount / picking_time
  let total_cherries : ℕ := cherries_per_quart * quarts
  let total_picking_time : ℕ := (total_cherries / picking_rate).ceil.toNat
  total_picking_time + syrup_making_time

/-- Proves that it takes Jerry 33 hours to make 9 quarts of cherry syrup -/
theorem jerry_syrup_time :
  make_cherry_syrup 500 2 300 3 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_jerry_syrup_time_l4069_406963


namespace NUMINAMATH_CALUDE_barium_oxide_moles_l4069_406910

-- Define the chemical reaction
structure Reaction where
  bao : ℝ    -- moles of Barium oxide
  h2o : ℝ    -- moles of Water
  baoh2 : ℝ  -- moles of Barium hydroxide

-- Define the reaction conditions
def reaction_conditions (r : Reaction) : Prop :=
  r.h2o = 1 ∧ r.baoh2 = r.bao

-- Theorem statement
theorem barium_oxide_moles (e : ℝ) :
  ∀ r : Reaction, reaction_conditions r → r.baoh2 = e → r.bao = e :=
by
  sorry

end NUMINAMATH_CALUDE_barium_oxide_moles_l4069_406910


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4069_406972

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = Real.sqrt 3)
  (θ : ℝ) (h4 : Real.tan θ = Real.sqrt 21 / 2)
  (P Q : ℝ × ℝ) (F2 : ℝ × ℝ) (h5 : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (h6 : Q.1 = 0) (h7 : (Q.2 - F2.2) / (Q.1 - F2.1) = Real.tan θ)
  (h8 : dist P Q / dist P F2 = 1/2) :
  ∃ (k : ℝ), ∀ (x y : ℝ), 3 * x^2 - y^2 = k :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4069_406972


namespace NUMINAMATH_CALUDE_total_movies_equals_sum_watched_and_to_watch_l4069_406945

/-- The 'crazy silly school' series -/
structure CrazySillySchool where
  total_books : ℕ
  total_movies : ℕ
  books_read : ℕ
  movies_watched : ℕ
  movies_to_watch : ℕ

/-- Theorem: The total number of movies in the series is equal to the sum of movies watched and movies left to watch -/
theorem total_movies_equals_sum_watched_and_to_watch (css : CrazySillySchool) 
  (h1 : css.total_books = 4)
  (h2 : css.books_read = 19)
  (h3 : css.movies_watched = 7)
  (h4 : css.movies_to_watch = 10) :
  css.total_movies = css.movies_watched + css.movies_to_watch :=
by
  sorry

#eval 7 + 10  -- Expected output: 17

end NUMINAMATH_CALUDE_total_movies_equals_sum_watched_and_to_watch_l4069_406945


namespace NUMINAMATH_CALUDE_symmetric_point_tanDoubleAngle_l4069_406923

/-- Given a line l in the Cartesian plane defined by the equation 2x*tan(α) + y - 1 = 0,
    and that the symmetric point of the origin (0,0) with respect to l is (1,1),
    prove that tan(2α) = 4/3. -/
theorem symmetric_point_tanDoubleAngle (α : ℝ) : 
  (∀ x y : ℝ, 2 * x * Real.tan α + y - 1 = 0 → 
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) → 
  Real.tan (2 * α) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_tanDoubleAngle_l4069_406923


namespace NUMINAMATH_CALUDE_last_term_is_123_l4069_406965

/-- A sequence of natural numbers -/
def Sequence : Type := ℕ → ℕ

/-- The specific sequence from the problem -/
def s : Sequence :=
  fun n =>
    match n with
    | 1 => 2
    | 2 => 3
    | 3 => 6
    | 4 => 15
    | 5 => 33
    | 6 => 123
    | _ => 0  -- For completeness, though we only care about the first 6 terms

/-- The theorem stating that the last (6th) term of the sequence is 123 -/
theorem last_term_is_123 : s 6 = 123 := by
  sorry


end NUMINAMATH_CALUDE_last_term_is_123_l4069_406965


namespace NUMINAMATH_CALUDE_fifteen_points_densified_thrice_equals_113_original_points_must_be_fifteen_l4069_406924

/-- Calculates the number of points after one densification -/
def densify (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the process of densification repeated k times -/
def densify_k_times (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => densify (densify_k_times n k)

/-- The theorem stating that 3 densifications of 15 points results in 113 points -/
theorem fifteen_points_densified_thrice_equals_113 :
  densify_k_times 15 3 = 113 :=
by sorry

/-- The main theorem proving that starting with 15 points and applying 3 densifications
    is the only way to end up with 113 points -/
theorem original_points_must_be_fifteen (n : ℕ) :
  densify_k_times n 3 = 113 → n = 15 :=
by sorry

end NUMINAMATH_CALUDE_fifteen_points_densified_thrice_equals_113_original_points_must_be_fifteen_l4069_406924


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l4069_406902

theorem wire_ratio_proof (total_length shorter_length : ℕ) 
  (h1 : total_length = 49)
  (h2 : shorter_length = 14)
  (h3 : shorter_length < total_length) :
  let longer_length := total_length - shorter_length
  (shorter_length : ℚ) / longer_length = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l4069_406902


namespace NUMINAMATH_CALUDE_sara_savings_l4069_406903

def quarters_to_cents (quarters : ℕ) (cents_per_quarter : ℕ) : ℕ :=
  quarters * cents_per_quarter

theorem sara_savings : quarters_to_cents 11 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_sara_savings_l4069_406903


namespace NUMINAMATH_CALUDE_red_peaches_count_l4069_406986

/-- Represents a basket of peaches -/
structure Basket :=
  (total : ℕ)
  (green : ℕ)
  (h_green_le_total : green ≤ total)

/-- Calculates the number of red peaches in a basket -/
def red_peaches (b : Basket) : ℕ := b.total - b.green

/-- Theorem: The number of red peaches in a basket with 10 total peaches and 3 green peaches is 7 -/
theorem red_peaches_count (b : Basket) (h_total : b.total = 10) (h_green : b.green = 3) : 
  red_peaches b = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l4069_406986


namespace NUMINAMATH_CALUDE_distance_washington_to_idaho_l4069_406976

/-- The distance from Washington to Idaho in miles -/
def distance_WI : ℝ := 640

/-- The distance from Idaho to Nevada in miles -/
def distance_IN : ℝ := 550

/-- The speed from Washington to Idaho in miles per hour -/
def speed_WI : ℝ := 80

/-- The speed from Idaho to Nevada in miles per hour -/
def speed_IN : ℝ := 50

/-- The total travel time in hours -/
def total_time : ℝ := 19

/-- Theorem stating that the distance from Washington to Idaho is 640 miles -/
theorem distance_washington_to_idaho : 
  distance_WI = 640 ∧ 
  distance_WI / speed_WI + distance_IN / speed_IN = total_time := by
  sorry


end NUMINAMATH_CALUDE_distance_washington_to_idaho_l4069_406976


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l4069_406909

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem parallel_vectors_m_value :
  ∀ m : ℝ, (∃ k : ℝ, vector_a = k • (vector_b m)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l4069_406909


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l4069_406978

theorem degree_to_radian_conversion (deg : ℝ) (rad : ℝ) : 
  (180 : ℝ) = π → 240 = (4 / 3 : ℝ) * π := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l4069_406978


namespace NUMINAMATH_CALUDE_unique_valid_config_l4069_406968

/-- Represents a fence configuration --/
structure FenceConfig where
  max_length : Nat
  num_max : Nat
  num_minus_one : Nat
  num_minus_two : Nat
  num_minus_three : Nat

/-- Checks if a fence configuration is valid --/
def is_valid_config (config : FenceConfig) : Prop :=
  config.num_max + config.num_minus_one + config.num_minus_two + config.num_minus_three = 16 ∧
  config.num_max * config.max_length +
  config.num_minus_one * (config.max_length - 1) +
  config.num_minus_two * (config.max_length - 2) +
  config.num_minus_three * (config.max_length - 3) = 297 ∧
  config.num_max = 8

/-- The unique valid fence configuration --/
def unique_config : FenceConfig :=
  { max_length := 20
  , num_max := 8
  , num_minus_one := 0
  , num_minus_two := 7
  , num_minus_three := 1
  }

/-- Theorem stating that the unique_config is the only valid configuration --/
theorem unique_valid_config :
  is_valid_config unique_config ∧
  (∀ config : FenceConfig, is_valid_config config → config = unique_config) := by
  sorry


end NUMINAMATH_CALUDE_unique_valid_config_l4069_406968


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l4069_406907

theorem lcm_gcd_product (a b : ℕ) (ha : a = 9) (hb : b = 15) :
  Nat.lcm a b * Nat.gcd a b = 135 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l4069_406907


namespace NUMINAMATH_CALUDE_total_cost_is_540_l4069_406948

def cherry_price : ℝ := 5
def olive_price : ℝ := 7
def bag_count : ℕ := 50
def discount_rate : ℝ := 0.1

def discounted_price (original_price : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

def total_cost : ℝ :=
  bag_count * (discounted_price cherry_price + discounted_price olive_price)

theorem total_cost_is_540 : total_cost = 540 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_540_l4069_406948


namespace NUMINAMATH_CALUDE_sum_in_base5_l4069_406947

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_in_base5 : toBase5 (45 + 78) = [4, 4, 3] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base5_l4069_406947


namespace NUMINAMATH_CALUDE_binomial_10_choose_6_l4069_406920

theorem binomial_10_choose_6 : Nat.choose 10 6 = 210 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_6_l4069_406920


namespace NUMINAMATH_CALUDE_five_arithmetic_operations_l4069_406937

theorem five_arithmetic_operations :
  -- 1. 5555 = 7
  (5 + 5 + 5) / 5 = 3 ∧
  (5 + 5) / 5 + 5 = 7 ∧
  -- 2. 5555 = 55
  (5 + 5) * 5 + 5 = 55 ∧
  -- 3. 5,5,5,5 = 4
  (5 * 5 - 5) / 5 = 4 ∧
  -- 4. 5,5,5,5 = 26
  5 * 5 + (5 / 5) = 26 ∧
  -- 5. 5,5,5,5 = 120
  5 * 5 * 5 - 5 = 120 ∧
  -- 6. 5,5,5,5 = 5
  (5 - 5) * 5 + 5 = 5 ∧
  -- 7. 5555 = 30
  (5 / 5 + 5) * 5 = 30 ∧
  -- 8. 5,5,5,5 = 130
  5 * 5 * 5 + 5 = 130 ∧
  -- 9. 5555 = 6
  (5 * 5 + 5) / 5 = 6 ∧
  -- 10. 5555 = 50
  5 * 5 + 5 * 5 = 50 ∧
  -- 11. 5555 = 625
  5 * 5 * 5 * 5 = 625 := by
  sorry

#check five_arithmetic_operations

end NUMINAMATH_CALUDE_five_arithmetic_operations_l4069_406937


namespace NUMINAMATH_CALUDE_parabolas_intersection_circle_l4069_406950

/-- The parabolas y = (x - 2)^2 and x - 5 = (y + 1)^2 intersect on a circle with radius r -/
theorem parabolas_intersection_circle (x y : ℝ) : 
  y = (x - 2)^2 ∧ x - 5 = (y + 1)^2 → (x - 3/2)^2 + (y + 1)^2 = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_circle_l4069_406950


namespace NUMINAMATH_CALUDE_overall_average_percentage_l4069_406927

theorem overall_average_percentage (students_A students_B students_C students_D : ℕ)
  (average_A average_B average_C average_D : ℚ) :
  students_A = 15 →
  students_B = 10 →
  students_C = 20 →
  students_D = 5 →
  average_A = 75 / 100 →
  average_B = 90 / 100 →
  average_C = 80 / 100 →
  average_D = 65 / 100 →
  let total_students := students_A + students_B + students_C + students_D
  let total_percentage := students_A * average_A + students_B * average_B +
                          students_C * average_C + students_D * average_D
  total_percentage / total_students = 79 / 100 := by
  sorry

end NUMINAMATH_CALUDE_overall_average_percentage_l4069_406927


namespace NUMINAMATH_CALUDE_inequality_proof_l4069_406929

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) 
  (h5 : c + d ≤ a) (h6 : c + d ≤ b) : 
  a * d + b * c ≤ a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4069_406929


namespace NUMINAMATH_CALUDE_smallest_sticker_collection_l4069_406971

theorem smallest_sticker_collection : ∃ (S : ℕ), 
  S > 2 ∧
  S % 4 = 2 ∧
  S % 6 = 2 ∧
  S % 9 = 2 ∧
  S % 10 = 2 ∧
  (∀ (T : ℕ), T > 2 → T % 4 = 2 → T % 6 = 2 → T % 9 = 2 → T % 10 = 2 → S ≤ T) ∧
  S = 182 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sticker_collection_l4069_406971


namespace NUMINAMATH_CALUDE_square_root_divided_by_six_l4069_406915

theorem square_root_divided_by_six (x : ℝ) : x > 0 ∧ Real.sqrt x / 6 = 1 ↔ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_root_divided_by_six_l4069_406915


namespace NUMINAMATH_CALUDE_joey_study_time_l4069_406985

/-- Calculates the total study time for Joey's SAT exam preparation --/
def total_study_time (weekday_hours_per_night : ℕ) (weekday_nights : ℕ) 
  (weekend_hours_per_day : ℕ) (weekend_days : ℕ) (weeks_until_exam : ℕ) : ℕ :=
  ((weekday_hours_per_night * weekday_nights + weekend_hours_per_day * weekend_days) * weeks_until_exam)

/-- Proves that Joey will spend 96 hours studying for his SAT exam --/
theorem joey_study_time : 
  total_study_time 2 5 3 2 6 = 96 := by
  sorry

end NUMINAMATH_CALUDE_joey_study_time_l4069_406985


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_in_sequence_l4069_406973

theorem infinitely_many_perfect_squares_in_sequence :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ ∃ m : ℕ, ⌊n * Real.sqrt 2⌋ = m^2 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_in_sequence_l4069_406973


namespace NUMINAMATH_CALUDE_f_symmetry_l4069_406981

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2 + 3 * Real.sin x + 2

theorem f_symmetry (a : ℝ) (h : f a = 1) : f (-a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l4069_406981


namespace NUMINAMATH_CALUDE_parallelepiped_surface_area_l4069_406959

/-- Given a rectangular parallelepiped with dimensions x, y, and z, 
    if the volume and surface area conditions are met, 
    then the total surface area of the original parallelepiped is 22 -/
theorem parallelepiped_surface_area 
  (x y z : ℝ) 
  (h1 : (x + 1) * (y + 1) * (z + 1) = x * y * z + 18) 
  (h2 : 2 * ((x + 1) * (y + 1) + (y + 1) * (z + 1) + (z + 1) * (x + 1)) = 2 * (x * y + x * z + y * z) + 30) : 
  2 * (x * y + x * z + y * z) = 22 := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_surface_area_l4069_406959


namespace NUMINAMATH_CALUDE_min_z_in_triangle_ABC_l4069_406999

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (1, 0)

-- Define the function z
def z (p : ℝ × ℝ) : ℝ := p.1 - p.2

-- Define the set of points inside or on the boundary of triangle ABC
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
    p = (a * A.1 + b * B.1 + c * C.1, a * A.2 + b * B.2 + c * C.2)}

-- Theorem statement
theorem min_z_in_triangle_ABC :
  ∃ (p : ℝ × ℝ), p ∈ triangle_ABC ∧ ∀ (q : ℝ × ℝ), q ∈ triangle_ABC → z p ≤ z q ∧ z p = -3 :=
sorry

end NUMINAMATH_CALUDE_min_z_in_triangle_ABC_l4069_406999


namespace NUMINAMATH_CALUDE_last_student_age_l4069_406943

theorem last_student_age 
  (total_students : ℕ) 
  (avg_age_all : ℝ) 
  (group1_size : ℕ) 
  (avg_age_group1 : ℝ) 
  (group2_size : ℕ) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 5)
  (h4 : avg_age_group1 = 13)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  ∃ (last_student_age : ℝ), 
    last_student_age = total_students * avg_age_all - 
      (group1_size * avg_age_group1 + group2_size * avg_age_group2) ∧
    last_student_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_last_student_age_l4069_406943


namespace NUMINAMATH_CALUDE_amp_four_two_l4069_406964

-- Define the & operation
def amp (a b : ℝ) : ℝ := ((a + b) * (a - b))^2

-- Theorem statement
theorem amp_four_two : amp 4 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_amp_four_two_l4069_406964


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4069_406928

theorem quadratic_inequality_solution_set (c : ℝ) (h : c > 1) :
  {x : ℝ | x^2 - (c + 1/c)*x + 1 > 0} = {x : ℝ | x < 1/c ∨ x > c} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4069_406928


namespace NUMINAMATH_CALUDE_max_books_buyable_l4069_406955

def total_money : ℚ := 24.41
def book_price : ℚ := 2.75

theorem max_books_buyable : 
  ∀ n : ℕ, n * book_price ≤ total_money ∧ 
  (n + 1) * book_price > total_money → n = 8 := by
sorry

end NUMINAMATH_CALUDE_max_books_buyable_l4069_406955


namespace NUMINAMATH_CALUDE_ingrid_income_calculation_l4069_406908

def john_income : ℝ := 57000
def john_tax_rate : ℝ := 0.30
def ingrid_tax_rate : ℝ := 0.40
def combined_tax_rate : ℝ := 0.35581395348837205

theorem ingrid_income_calculation (ingrid_income : ℝ) : 
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = combined_tax_rate →
  ingrid_income = 72000 := by
sorry

end NUMINAMATH_CALUDE_ingrid_income_calculation_l4069_406908


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l4069_406994

theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 3 * x₁ - 1 = 0) → 
  (2 * x₂^2 + 3 * x₂ - 1 = 0) → 
  (x₁ ≠ x₂) →
  (1 / x₁ + 1 / x₂ = 3) := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l4069_406994


namespace NUMINAMATH_CALUDE_cone_surface_area_l4069_406905

theorem cone_surface_area (r h : ℝ) (hr : r = 1) (hh : h = 2 * Real.sqrt 2) :
  let l := Real.sqrt (r^2 + h^2)
  π * r^2 + π * r * l = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l4069_406905


namespace NUMINAMATH_CALUDE_walts_investment_rate_l4069_406979

/-- Given Walt's investment scenario, prove that the unknown interest rate is 9% -/
theorem walts_investment_rate : 
  ∀ (total_extra : ℝ) (total_interest : ℝ) (known_amount : ℝ) (known_rate : ℝ),
  total_extra = 9000 →
  total_interest = 770 →
  known_amount = 4000 →
  known_rate = 0.08 →
  ∃ (unknown_rate : ℝ),
    unknown_rate = 0.09 ∧
    total_interest = known_amount * known_rate + (total_extra - known_amount) * unknown_rate :=
by
  sorry

#check walts_investment_rate

end NUMINAMATH_CALUDE_walts_investment_rate_l4069_406979


namespace NUMINAMATH_CALUDE_figure_sides_l4069_406942

/-- A figure with a perimeter of 49 cm and a side length of 7 cm has 7 sides. -/
theorem figure_sides (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 49) (h2 : side_length = 7) :
  perimeter / side_length = 7 := by
  sorry

end NUMINAMATH_CALUDE_figure_sides_l4069_406942


namespace NUMINAMATH_CALUDE_sqrt_neg_five_squared_l4069_406939

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_five_squared_l4069_406939


namespace NUMINAMATH_CALUDE_parabola_vertex_l4069_406954

/-- Given a quadratic function f(x) = -x^2 + cx + d where the solution to f(x) ≤ 0
    is (-∞, -5] ∪ [1, ∞), the vertex of the parabola defined by f(x) is (3, 4). -/
theorem parabola_vertex (c d : ℝ) :
  let f : ℝ → ℝ := λ x => -x^2 + c*x + d
  (∀ x, f x ≤ 0 ↔ x ≤ -5 ∨ x ≥ 1) →
  (∃! p : ℝ × ℝ, p.1 = 3 ∧ p.2 = 4 ∧ ∀ x, f x ≤ f p.1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l4069_406954


namespace NUMINAMATH_CALUDE_odd_function_extension_l4069_406967

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_extension :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is an odd function
  (∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x) →  -- f(x) = x^2 - 2x when x ≥ 0
  (∀ x : ℝ, x < 0 → f x = -x^2 + 2*x) :=  -- f(x) = -x^2 + 2x when x < 0
by sorry

end NUMINAMATH_CALUDE_odd_function_extension_l4069_406967


namespace NUMINAMATH_CALUDE_scholarship_theorem_l4069_406953

def scholarship_problem (wendy_last_year : ℝ) : Prop :=
  let kelly_last_year := 2 * wendy_last_year
  let nina_last_year := kelly_last_year - 8000
  let jason_last_year := 3/4 * kelly_last_year
  let wendy_this_year := wendy_last_year * 1.1
  let kelly_this_year := kelly_last_year * 1.08
  let nina_this_year := nina_last_year * 1.15
  let jason_this_year := jason_last_year * 1.12
  let total_this_year := wendy_this_year + kelly_this_year + nina_this_year + jason_this_year
  wendy_last_year = 20000 → total_this_year = 135600

theorem scholarship_theorem : scholarship_problem 20000 := by
  sorry

end NUMINAMATH_CALUDE_scholarship_theorem_l4069_406953


namespace NUMINAMATH_CALUDE_probability_even_sum_le_8_l4069_406936

def dice_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 12

theorem probability_even_sum_le_8 : 
  (favorable_outcomes : ℚ) / dice_outcomes = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_even_sum_le_8_l4069_406936


namespace NUMINAMATH_CALUDE_geometric_sequence_a12_l4069_406932

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a12 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 6 * a 10 = 16 →
  a 4 = 1 →
  a 12 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a12_l4069_406932


namespace NUMINAMATH_CALUDE_triangle_area_is_24_l4069_406970

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (0, 8)

-- Define the equation
def satisfies_equation (p : ℝ × ℝ) : Prop :=
  |4 * p.1| + |3 * p.2| + |24 - 4 * p.1 - 3 * p.2| = 24

-- Theorem statement
theorem triangle_area_is_24 :
  satisfies_equation A ∧ satisfies_equation B ∧ satisfies_equation C →
  (1/2 : ℝ) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)| = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_24_l4069_406970


namespace NUMINAMATH_CALUDE_last_integer_in_sequence_l4069_406904

def sequence_term (n : ℕ) : ℚ :=
  800000 / 2^n

theorem last_integer_in_sequence :
  ∀ k : ℕ, (sequence_term k).isInt → sequence_term k ≥ 3125 :=
sorry

end NUMINAMATH_CALUDE_last_integer_in_sequence_l4069_406904


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l4069_406921

theorem power_mod_seventeen : 7^2023 % 17 = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l4069_406921


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l4069_406989

theorem geometric_sequence_product (a r : ℝ) (n : ℕ) (h_even : Even n) :
  let S := a * (1 - r^n) / (1 - r)
  let S' := (1 / (2*a)) * (r^n - 1) / (r - 1) * r^(1-n)
  let P := (2*a)^n * r^(n*(n-1)/2)
  P = (S * S')^(n/2) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l4069_406989


namespace NUMINAMATH_CALUDE_right_triangle_to_square_l4069_406935

theorem right_triangle_to_square (longer_leg : ℝ) (shorter_leg : ℝ) (square_side : ℝ) : 
  longer_leg = 10 →
  longer_leg = 2 * square_side →
  shorter_leg = square_side →
  shorter_leg = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_to_square_l4069_406935


namespace NUMINAMATH_CALUDE_odd_numbers_with_difference_16_are_coprime_l4069_406987

theorem odd_numbers_with_difference_16_are_coprime 
  (a b : ℤ) 
  (ha : Odd a) 
  (hb : Odd b) 
  (hdiff : |a - b| = 16) : 
  Int.gcd a b = 1 := by
sorry

end NUMINAMATH_CALUDE_odd_numbers_with_difference_16_are_coprime_l4069_406987


namespace NUMINAMATH_CALUDE_triangle_side_length_triangle_angle_relation_l4069_406961

-- Define the triangle ABC
structure Triangle where
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C

-- Theorem 1
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 3 * t.c) 
  (h2 : t.b = Real.sqrt 2) 
  (h3 : Real.cos t.B = 2/3) : 
  t.c = Real.sqrt 3 / 3 := by
  sorry

-- Theorem 2
theorem triangle_angle_relation (t : Triangle) 
  (h : Real.sin t.A / t.a = Real.cos t.B / (2 * t.b)) : 
  Real.sin (t.B + π/2) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_triangle_angle_relation_l4069_406961


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l4069_406906

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l4069_406906


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l4069_406900

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 1

theorem tangent_line_at_one :
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
  (∀ x : ℝ, x ≠ 0 → HasDerivAt f (1 / x + 2) x) ∧
  (m * 1 - f 1 + b = 0) ∧
  (m = 3) ∧ (b = -2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l4069_406900


namespace NUMINAMATH_CALUDE_box_volume_theorem_l4069_406913

/-- Represents the possible volumes of the box -/
def PossibleVolumes : Set ℕ := {80, 100, 120, 150, 200}

/-- Theorem: Given a rectangular box with integer side lengths in the ratio 1:2:5,
    the only possible volume from the set of possible volumes is 80 -/
theorem box_volume_theorem (x : ℕ) (hx : x > 0) :
  (∃ (v : ℕ), v ∈ PossibleVolumes ∧ v = x * (2 * x) * (5 * x)) ↔ (x * (2 * x) * (5 * x) = 80) :=
by sorry

end NUMINAMATH_CALUDE_box_volume_theorem_l4069_406913


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l4069_406991

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : S ∩ (U \ T) = {1, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l4069_406991


namespace NUMINAMATH_CALUDE_range_of_a_l4069_406995

-- Define the function f
def f (x : ℝ) : ℝ := -2*x^5 - x^3 - 7*x + 2

-- State the theorem
theorem range_of_a (a : ℝ) : f (a^2) + f (a-2) > 4 → a ∈ Set.Ioo (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4069_406995


namespace NUMINAMATH_CALUDE_power_sum_equals_zero_l4069_406960

theorem power_sum_equals_zero : (-1 : ℤ) ^ (5^2) + (1 : ℤ) ^ (2^5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_zero_l4069_406960


namespace NUMINAMATH_CALUDE_monday_rainfall_calculation_l4069_406966

def total_rainfall : ℝ := 0.67
def tuesday_rainfall : ℝ := 0.42
def wednesday_rainfall : ℝ := 0.08

theorem monday_rainfall_calculation :
  ∃ (monday_rainfall : ℝ),
    monday_rainfall + tuesday_rainfall + wednesday_rainfall = total_rainfall ∧
    monday_rainfall = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_monday_rainfall_calculation_l4069_406966


namespace NUMINAMATH_CALUDE_profit_is_333_l4069_406984

/-- Represents the candy bar sales scenario -/
structure CandyBarSales where
  totalBars : ℕ
  firstBatchCost : ℚ
  secondBatchCost : ℚ
  firstBatchSell : ℚ
  secondBatchSell : ℚ

/-- Calculates the profit from candy bar sales -/
def calculateProfit (sales : CandyBarSales) : ℚ :=
  let costPrice := (800 / 3) + 100
  let sellingPrice := 300 + (600 * 2 / 3)
  sellingPrice - costPrice

/-- Theorem stating that the profit is $333 -/
theorem profit_is_333 (sales : CandyBarSales) 
    (h1 : sales.totalBars = 1200)
    (h2 : sales.firstBatchCost = 1/3)
    (h3 : sales.secondBatchCost = 1/4)
    (h4 : sales.firstBatchSell = 1/2)
    (h5 : sales.secondBatchSell = 2/3) :
  Int.floor (calculateProfit sales) = 333 := by
  sorry

#eval Int.floor (calculateProfit { 
  totalBars := 1200, 
  firstBatchCost := 1/3, 
  secondBatchCost := 1/4, 
  firstBatchSell := 1/2, 
  secondBatchSell := 2/3
})

end NUMINAMATH_CALUDE_profit_is_333_l4069_406984


namespace NUMINAMATH_CALUDE_circle_condition_l4069_406918

theorem circle_condition (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*x + a = 0) →
  (∃ (h k r : ℝ), ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2) →
  a < 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l4069_406918


namespace NUMINAMATH_CALUDE_function_inequality_l4069_406951

/-- Given functions f and g, prove that a ≤ 1 -/
theorem function_inequality (f g : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = x + 4 / x) →
  (∀ x, g x = 2^x + a) →
  (∀ x₁ ∈ Set.Icc (1/2) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂) →
  a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l4069_406951


namespace NUMINAMATH_CALUDE_blocks_between_39_and_40_l4069_406997

/-- Represents the number of blocks in the original tower -/
def original_tower_size : ℕ := 90

/-- Represents the number of blocks taken at a time to build the new tower -/
def blocks_per_group : ℕ := 3

/-- Calculates the group number for a given block number in the original tower -/
def group_number (block : ℕ) : ℕ :=
  (original_tower_size - block) / blocks_per_group + 1

/-- Calculates the position of a block within its group in the new tower -/
def position_in_group (block : ℕ) : ℕ :=
  (original_tower_size - block) % blocks_per_group + 1

/-- Theorem stating that there are 4 blocks between blocks 39 and 40 in the new tower -/
theorem blocks_between_39_and_40 :
  ∃ (a b c d : ℕ),
    group_number 39 = group_number a ∧
    group_number 39 = group_number b ∧
    group_number 40 = group_number c ∧
    group_number 40 = group_number d ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧
    position_in_group 39 < position_in_group a ∧
    position_in_group a < position_in_group b ∧
    position_in_group b < position_in_group c ∧
    position_in_group c < position_in_group d ∧
    position_in_group d < position_in_group 40 :=
by
  sorry

end NUMINAMATH_CALUDE_blocks_between_39_and_40_l4069_406997


namespace NUMINAMATH_CALUDE_area_formula_l4069_406990

/-- Triangle with sides a, b, c and angle A -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Theorem: Area formula for triangles with angle A = 60° or 120° -/
theorem area_formula (t : Triangle) :
  (t.angleA = 60 → area t = (Real.sqrt 3 / 4) * (t.a^2 - (t.b - t.c)^2)) ∧
  (t.angleA = 120 → area t = (Real.sqrt 3 / 12) * (t.a^2 - (t.b - t.c)^2)) := by
  sorry

end NUMINAMATH_CALUDE_area_formula_l4069_406990


namespace NUMINAMATH_CALUDE_train_crossing_time_l4069_406975

/-- Proves that a train with given length and speed takes the calculated time to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 360 →
  train_speed_kmh = 216 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 6 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l4069_406975


namespace NUMINAMATH_CALUDE_four_color_plane_exists_l4069_406919

-- Define the color type
inductive Color
| Red | Blue | Green | Yellow | Purple

-- Define the space as a type of points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the condition that each color appears at least once
axiom all_colors_present : ∀ c : Color, ∃ p : Point, coloring p = c

-- Define a plane
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to check if a point is on a plane
def on_plane (plane : Plane) (point : Point) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

-- Define a function to count distinct colors on a plane
def count_colors_on_plane (plane : Plane) : ℕ := sorry

-- The main theorem
theorem four_color_plane_exists :
  ∃ plane : Plane, count_colors_on_plane plane ≥ 4 := sorry

end NUMINAMATH_CALUDE_four_color_plane_exists_l4069_406919


namespace NUMINAMATH_CALUDE_textbook_ratio_l4069_406934

theorem textbook_ratio (initial : ℚ) (remaining : ℚ) 
  (h1 : initial = 960)
  (h2 : remaining = 360)
  (h3 : ∃ textbook_cost : ℚ, initial - textbook_cost - (1/4) * (initial - textbook_cost) = remaining) :
  ∃ textbook_cost : ℚ, textbook_cost / initial = 1/2 := by
sorry

end NUMINAMATH_CALUDE_textbook_ratio_l4069_406934


namespace NUMINAMATH_CALUDE_octagon_diagonals_l4069_406980

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l4069_406980


namespace NUMINAMATH_CALUDE_correct_proposition_l4069_406944

theorem correct_proposition :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b c : ℝ, a > b ∧ ¬(a*c^2 > b*c^2)) ∧
  (∃ a b : ℝ, a > b ∧ ¬(a^2 > b^2)) ∧
  (∃ a b : ℝ, |a| > b ∧ ¬(a^2 > b^2)) :=
by sorry

end NUMINAMATH_CALUDE_correct_proposition_l4069_406944


namespace NUMINAMATH_CALUDE_first_rectangle_width_first_rectangle_width_proof_l4069_406993

/-- Given two rectangles, where the second has width 3 and height 6,
    and the first has height 5 and area 2 square inches more than the second,
    prove that the width of the first rectangle is 4 inches. -/
theorem first_rectangle_width : ℝ → Prop :=
  fun w : ℝ =>
    let first_height : ℝ := 5
    let second_width : ℝ := 3
    let second_height : ℝ := 6
    let first_area : ℝ := w * first_height
    let second_area : ℝ := second_width * second_height
    first_area = second_area + 2 → w = 4

/-- Proof of the theorem -/
theorem first_rectangle_width_proof : first_rectangle_width 4 := by
  sorry

end NUMINAMATH_CALUDE_first_rectangle_width_first_rectangle_width_proof_l4069_406993


namespace NUMINAMATH_CALUDE_rent_calculation_l4069_406941

def monthly_budget (rent : ℚ) : Prop :=
  let food := (3/5) * rent
  let mortgage := 3 * food
  let savings := 2000
  let taxes := (2/5) * savings
  rent + food + mortgage + savings + taxes = 4840

theorem rent_calculation :
  ∃ (rent : ℚ), monthly_budget rent ∧ rent = 600 := by
sorry

end NUMINAMATH_CALUDE_rent_calculation_l4069_406941


namespace NUMINAMATH_CALUDE_binomial_coefficient_is_integer_l4069_406962

theorem binomial_coefficient_is_integer (m n : ℕ) (h : m > n) :
  ∃ k : ℕ, (m.choose n) = k := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_is_integer_l4069_406962


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l4069_406988

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) : 
  (2 * r) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l4069_406988


namespace NUMINAMATH_CALUDE_alonzo_unsold_tomatoes_l4069_406925

/-- Calculates the amount of unsold tomatoes given the total harvest and amounts sold to two buyers. -/
def unsold_tomatoes (total_harvest : ℝ) (sold_to_maxwell : ℝ) (sold_to_wilson : ℝ) : ℝ :=
  total_harvest - (sold_to_maxwell + sold_to_wilson)

/-- Proves that given the specific amounts in Mr. Alonzo's tomato sales, the unsold amount is 42 kg. -/
theorem alonzo_unsold_tomatoes :
  unsold_tomatoes 245.5 125.5 78 = 42 := by
  sorry

end NUMINAMATH_CALUDE_alonzo_unsold_tomatoes_l4069_406925


namespace NUMINAMATH_CALUDE_plot_area_in_acres_l4069_406974

-- Define the triangle dimensions
def leg1 : ℝ := 8
def leg2 : ℝ := 6

-- Define scale and conversion factors
def scale : ℝ := 3  -- 1 cm = 3 miles
def acres_per_square_mile : ℝ := 640

-- Define the theorem
theorem plot_area_in_acres :
  let triangle_area := (1/2) * leg1 * leg2
  let scaled_area := triangle_area * scale * scale
  let area_in_acres := scaled_area * acres_per_square_mile
  area_in_acres = 138240 := by sorry

end NUMINAMATH_CALUDE_plot_area_in_acres_l4069_406974


namespace NUMINAMATH_CALUDE_last_three_digits_of_11_pow_30_l4069_406992

theorem last_three_digits_of_11_pow_30 : 11^30 ≡ 801 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_11_pow_30_l4069_406992


namespace NUMINAMATH_CALUDE_ellipse_equation_l4069_406983

/-- An ellipse with center at the origin, coordinate axes as axes of symmetry,
    and passing through points (√6, 1) and (-√3, -√2) has the equation x²/9 + y²/3 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧
    x^2 / m + y^2 / n = 1 ∧
    6 / m + 1 / n = 1 ∧
    3 / m + 2 / n = 1) →
  x^2 / 9 + y^2 / 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4069_406983


namespace NUMINAMATH_CALUDE_range_of_m_l4069_406938

def p (x : ℝ) : Prop := (x - 1) / x ≤ 0

def q (x m : ℝ) : Prop := (x - m) * (x - m + 2) ≤ 0

theorem range_of_m (m : ℝ) :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) → 1 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l4069_406938


namespace NUMINAMATH_CALUDE_restaurant_tip_percentage_l4069_406911

/-- Calculates the tip percentage given the cost of an appetizer, number of entrees,
    cost per entree, and total amount spent at a restaurant. -/
theorem restaurant_tip_percentage
  (appetizer_cost : ℚ)
  (num_entrees : ℕ)
  (entree_cost : ℚ)
  (total_spent : ℚ)
  (h1 : appetizer_cost = 10)
  (h2 : num_entrees = 4)
  (h3 : entree_cost = 20)
  (h4 : total_spent = 108) :
  (total_spent - (appetizer_cost + num_entrees * entree_cost)) / (appetizer_cost + num_entrees * entree_cost) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_tip_percentage_l4069_406911


namespace NUMINAMATH_CALUDE_largest_of_six_consecutive_odds_l4069_406982

theorem largest_of_six_consecutive_odds (a : ℕ) (h1 : a > 0) 
  (h2 : a % 2 = 1) 
  (h3 : (a * (a + 2) * (a + 4) * (a + 6) * (a + 8) * (a + 10) = 135135)) : 
  a + 10 = 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_six_consecutive_odds_l4069_406982


namespace NUMINAMATH_CALUDE_trapezoid_area_coefficient_l4069_406933

-- Define the triangle
def triangle_side_1 : ℝ := 15
def triangle_side_2 : ℝ := 39
def triangle_side_3 : ℝ := 36

-- Define the area formula for the trapezoid
def trapezoid_area (γ δ ω : ℝ) : ℝ := γ * ω - δ * ω^2

-- State the theorem
theorem trapezoid_area_coefficient :
  ∃ (γ : ℝ), 
    (trapezoid_area γ (60/169) triangle_side_2 = 0) ∧
    (trapezoid_area γ (60/169) (triangle_side_2/2) = 
      (1/2) * Real.sqrt (
        (triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 *
        ((triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 - triangle_side_1) *
        ((triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 - triangle_side_2) *
        ((triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 - triangle_side_3)
      )) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_coefficient_l4069_406933


namespace NUMINAMATH_CALUDE_bulbs_chosen_l4069_406958

theorem bulbs_chosen (total_bulbs : ℕ) (defective_bulbs : ℕ) (prob_at_least_one_defective : ℝ) :
  total_bulbs = 21 →
  defective_bulbs = 4 →
  prob_at_least_one_defective = 0.35238095238095235 →
  ∃ n : ℕ, n = 2 ∧ (1 - (total_bulbs - defective_bulbs : ℝ) / total_bulbs ^ n) = prob_at_least_one_defective :=
by sorry

end NUMINAMATH_CALUDE_bulbs_chosen_l4069_406958


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l4069_406998

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  2 * x^2 * y - 5 * x * y - 4 * x * y^2 + x * y + 4 * x^2 * y - 7 * x * y^2 =
  6 * x^2 * y - 4 * x * y - 11 * x * y^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) :
  (5 * a^2 + 2 * a - 1) - 4 * (2 * a^2 - 3 * a) =
  -3 * a^2 + 14 * a - 1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l4069_406998


namespace NUMINAMATH_CALUDE_rational_inequality_solution_set_l4069_406957

theorem rational_inequality_solution_set (x : ℝ) : 
  (((2 * x - 1) / (x + 1) < 0) ↔ (-1 < x ∧ x < 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_set_l4069_406957


namespace NUMINAMATH_CALUDE_determinant_equality_l4069_406969

theorem determinant_equality (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 5 →
  Matrix.det ![![a - c, b - d], ![c, d]] = 5 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l4069_406969


namespace NUMINAMATH_CALUDE_total_interest_earned_l4069_406916

def initial_investment : ℝ := 1500
def annual_interest_rate : ℝ := 0.12
def time_period : ℕ := 4

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem total_interest_earned :
  let final_amount := compound_interest initial_investment annual_interest_rate time_period
  final_amount - initial_investment = 862.2 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_earned_l4069_406916


namespace NUMINAMATH_CALUDE_identity_function_proof_l4069_406931

theorem identity_function_proof (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inverse : ∀ x, f (f x) = x) : 
  ∀ x, f x = x := by
sorry

end NUMINAMATH_CALUDE_identity_function_proof_l4069_406931


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l4069_406930

/-- The number of magical herbs available --/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available --/
def num_crystals : ℕ := 6

/-- The number of herbs that react negatively with one crystal --/
def incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir --/
def valid_combinations : ℕ := num_herbs * num_crystals - incompatible_herbs

/-- Theorem stating that the number of valid combinations is 21 --/
theorem wizard_elixir_combinations :
  valid_combinations = 21 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l4069_406930


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l4069_406914

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 80 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = 65 * ((n1 : ℚ) + (n2 : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l4069_406914


namespace NUMINAMATH_CALUDE_domain_shift_l4069_406926

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 1 4

-- State the theorem
theorem domain_shift :
  (∀ x, f x ≠ 0 → x ∈ domain_f) →
  (∀ x, f (x - 1) ≠ 0 → x ∈ Set.Icc 2 5) :=
sorry

end NUMINAMATH_CALUDE_domain_shift_l4069_406926


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l4069_406952

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 7*x + 12

-- Define the points A, B, and C
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (0, 12)

-- Theorem statement
theorem area_of_triangle_ABC : 
  let triangle_area := (1/2) * |A.1 - B.1| * C.2
  (f A.1 = 0) ∧ (f B.1 = 0) ∧ (f 0 = C.2) → triangle_area = 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l4069_406952


namespace NUMINAMATH_CALUDE_exists_diameter_points_l4069_406940

/-- A circle divided into 3k arcs by 3k points -/
structure CircleDivision (k : ℕ) where
  points : Fin (3 * k) → ℝ × ℝ
  is_on_circle : ∀ i, (points i).1^2 + (points i).2^2 = 1
  arc_lengths : Fin (3 * k) → ℝ
  unit_arcs : ∃ (S : Finset (Fin (3 * k))), S.card = k ∧ ∀ i ∈ S, arc_lengths i = 1
  double_unit_arcs : ∃ (S : Finset (Fin (3 * k))), S.card = k ∧ ∀ i ∈ S, arc_lengths i = 2
  triple_unit_arcs : ∃ (S : Finset (Fin (3 * k))), S.card = k ∧ ∀ i ∈ S, arc_lengths i = 3
  total_length : (Finset.univ.sum arc_lengths) = 2 * Real.pi

/-- Two points determine a diameter if they are opposite each other on the circle -/
def is_diameter {k : ℕ} (cd : CircleDivision k) (i j : Fin (3 * k)) : Prop :=
  (cd.points i).1 = -(cd.points j).1 ∧ (cd.points i).2 = -(cd.points j).2

/-- There exist two division points that determine a diameter -/
theorem exists_diameter_points {k : ℕ} (cd : CircleDivision k) :
  ∃ (i j : Fin (3 * k)), is_diameter cd i j :=
sorry

end NUMINAMATH_CALUDE_exists_diameter_points_l4069_406940
