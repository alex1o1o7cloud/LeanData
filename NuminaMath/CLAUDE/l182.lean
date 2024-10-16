import Mathlib

namespace NUMINAMATH_CALUDE_derivative_at_one_l182_18224

/-- Given a function f(x) = (x-1)^3 + 3(x-1), prove that its derivative at x=1 is 3. -/
theorem derivative_at_one (f : ℝ → ℝ) (h : f = λ x ↦ (x - 1)^3 + 3*(x - 1)) :
  deriv f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l182_18224


namespace NUMINAMATH_CALUDE_max_value_when_m_neg_four_range_of_m_for_condition_l182_18231

-- Define the function f
def f (x m : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

-- Theorem for part I
theorem max_value_when_m_neg_four :
  ∃ (x_max : ℝ), ∀ (x : ℝ), f x (-4) ≤ f x_max (-4) ∧ f x_max (-4) = 2 :=
sorry

-- Theorem for part II
theorem range_of_m_for_condition (m : ℝ) :
  (∃ (x₀ : ℝ), f x₀ m ≥ 1 / m - 4) ↔ m ∈ Set.Ioi 0 ∪ {1} :=
sorry

end NUMINAMATH_CALUDE_max_value_when_m_neg_four_range_of_m_for_condition_l182_18231


namespace NUMINAMATH_CALUDE_min_value_of_z_l182_18294

theorem min_value_of_z (x y : ℝ) :
  2 * x^2 + 3 * y^2 + 8 * x - 6 * y + 35 ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_z_l182_18294


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l182_18234

theorem divisibility_by_seven (a b : ℕ) (h : 7 ∣ (a + b)) : 7 ∣ (101 * a + 10 * b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l182_18234


namespace NUMINAMATH_CALUDE_efficiency_increase_l182_18219

theorem efficiency_increase (days_sakshi days_tanya : ℝ) 
  (h1 : days_sakshi = 20) 
  (h2 : days_tanya = 16) : 
  (1 / days_tanya - 1 / days_sakshi) / (1 / days_sakshi) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_efficiency_increase_l182_18219


namespace NUMINAMATH_CALUDE_triangle_side_from_median_l182_18250

theorem triangle_side_from_median (a b k : ℝ) (ha : 0 < a) (hb : 0 < b) (hk : 0 < k) :
  ∃ c : ℝ, c > 0 ∧ c = Real.sqrt ((2 * (a^2 + b^2 - 2 * k^2)) / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_from_median_l182_18250


namespace NUMINAMATH_CALUDE_quadratic_roots_sign_l182_18295

theorem quadratic_roots_sign (a : ℝ) (h1 : a > 0) (h2 : a ≠ 0) :
  ¬∃ (c : ℝ → Prop), ∀ (x y : ℝ),
    (c a → (a * x^2 + 2*x + 1 = 0 ∧ a * y^2 + 2*y + 1 = 0 ∧ x ≠ y ∧ x > 0 ∧ y < 0)) ∧
    (¬c a → ¬(a * x^2 + 2*x + 1 = 0 ∧ a * y^2 + 2*y + 1 = 0 ∧ x ≠ y ∧ x > 0 ∧ y < 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sign_l182_18295


namespace NUMINAMATH_CALUDE_min_k_inequality_k_lower_bound_l182_18222

theorem min_k_inequality (x y z : ℝ) :
  (16/9 : ℝ) * (x^2 - x + 1) * (y^2 - y + 1) * (z^2 - z + 1) ≥ (x*y*z)^2 - x*y*z + 1 :=
by sorry

theorem k_lower_bound (k : ℝ) 
  (h : ∀ x y z : ℝ, k * (x^2 - x + 1) * (y^2 - y + 1) * (z^2 - z + 1) ≥ (x*y*z)^2 - x*y*z + 1) :
  k ≥ 16/9 :=
by sorry

end NUMINAMATH_CALUDE_min_k_inequality_k_lower_bound_l182_18222


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l182_18290

open Set

def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B : A ∩ (Bᶜ) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l182_18290


namespace NUMINAMATH_CALUDE_pigs_count_l182_18297

/-- The number of pigs initially in the barn -/
def initial_pigs : ℕ := 64

/-- The number of pigs that joined -/
def joined_pigs : ℕ := 22

/-- The total number of pigs after joining -/
def total_pigs : ℕ := 86

/-- Theorem stating that the initial number of pigs plus the joined pigs equals the total pigs -/
theorem pigs_count : initial_pigs + joined_pigs = total_pigs := by
  sorry

end NUMINAMATH_CALUDE_pigs_count_l182_18297


namespace NUMINAMATH_CALUDE_digit_sum_problem_l182_18288

/-- Given six unique digits from 2 to 7, prove that if their sums along specific lines total 66, then B must be 4. -/
theorem digit_sum_problem (A B C D E F : ℕ) : 
  A ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  B ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  C ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  D ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  E ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  F ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F →
  (A + B + C) + (A + B + E + F) + (C + D + E) + (B + D + F) + (C + F) = 66 →
  B = 4 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l182_18288


namespace NUMINAMATH_CALUDE_exam_score_calculation_l182_18252

theorem exam_score_calculation 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (total_score : ℕ) 
  (wrong_answer_penalty : ℕ) :
  total_questions = 75 →
  correct_answers = 40 →
  total_score = 125 →
  wrong_answer_penalty = 1 →
  ∃ (score_per_correct : ℕ),
    score_per_correct * correct_answers - 
    wrong_answer_penalty * (total_questions - correct_answers) = total_score ∧
    score_per_correct = 4 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l182_18252


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l182_18217

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  sum_formula : ∀ n : ℕ+, S n = (n : ℝ) * (a 1 + a n) / 2
  arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_properties
  (seq : ArithmeticSequence)
  (h : seq.S 5 > seq.S 6 ∧ seq.S 6 > seq.S 4) :
  common_difference seq < 0 ∧ seq.S 10 > 0 ∧ seq.S 11 < 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l182_18217


namespace NUMINAMATH_CALUDE_restaurant_group_size_l182_18244

theorem restaurant_group_size :
  let adult_meal_cost : ℕ := 3
  let kids_eat_free : Bool := true
  let num_kids : ℕ := 7
  let total_cost : ℕ := 15
  let num_adults : ℕ := total_cost / adult_meal_cost
  let total_people : ℕ := num_adults + num_kids
  total_people = 12 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_size_l182_18244


namespace NUMINAMATH_CALUDE_sixty_first_sample_number_l182_18268

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (numGroups : ℕ) (firstItem : ℕ) (groupIndex : ℕ) : ℕ :=
  firstItem + (groupIndex - 1) * (totalItems / numGroups)

/-- Theorem stating the result of the 61st sample in the given conditions -/
theorem sixty_first_sample_number
  (totalItems : ℕ) (numGroups : ℕ) (firstItem : ℕ) (groupIndex : ℕ)
  (h1 : totalItems = 3000)
  (h2 : numGroups = 150)
  (h3 : firstItem = 11)
  (h4 : groupIndex = 61) :
  systematicSample totalItems numGroups firstItem groupIndex = 1211 := by
  sorry

#eval systematicSample 3000 150 11 61

end NUMINAMATH_CALUDE_sixty_first_sample_number_l182_18268


namespace NUMINAMATH_CALUDE_soda_can_ounces_l182_18283

/-- Represents the daily soda consumption in cans -/
def daily_soda_cans : ℕ := 5

/-- Represents the daily water consumption in ounces -/
def daily_water_oz : ℕ := 64

/-- Represents the weekly total fluid consumption in ounces -/
def weekly_total_oz : ℕ := 868

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Calculates the number of ounces in each can of soda -/
def ounces_per_soda_can : ℚ :=
  (weekly_total_oz - daily_water_oz * days_in_week) / (daily_soda_cans * days_in_week)

theorem soda_can_ounces :
  ounces_per_soda_can = 12 := by sorry

end NUMINAMATH_CALUDE_soda_can_ounces_l182_18283


namespace NUMINAMATH_CALUDE_anthonys_pets_l182_18277

theorem anthonys_pets (initial_pets : ℕ) (lost_pets : ℕ) (final_pets : ℕ) :
  initial_pets = 16 →
  lost_pets = 6 →
  final_pets = 8 →
  (initial_pets - lost_pets - final_pets : ℚ) / (initial_pets - lost_pets) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_anthonys_pets_l182_18277


namespace NUMINAMATH_CALUDE_pigeonhole_socks_l182_18242

theorem pigeonhole_socks (red blue : ℕ) (h1 : red = 10) (h2 : blue = 10) :
  ∃ n : ℕ, n = 3 ∧ 
  (∀ m : ℕ, m < n → ∃ f : Fin m → Bool, Function.Injective f) ∧
  (∀ f : Fin n → Bool, ¬Function.Injective f) :=
sorry

end NUMINAMATH_CALUDE_pigeonhole_socks_l182_18242


namespace NUMINAMATH_CALUDE_smaller_number_l182_18204

theorem smaller_number (L S : ℕ) (hL : L > S) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : S = 270 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_l182_18204


namespace NUMINAMATH_CALUDE_oranges_eaten_l182_18298

/-- Given 96 oranges, with half of them ripe, and 1/4 of ripe oranges and 1/8 of unripe oranges eaten, 
    the total number of eaten oranges is 18. -/
theorem oranges_eaten (total : ℕ) (ripe_fraction : ℚ) (ripe_eaten_fraction : ℚ) (unripe_eaten_fraction : ℚ) :
  total = 96 →
  ripe_fraction = 1/2 →
  ripe_eaten_fraction = 1/4 →
  unripe_eaten_fraction = 1/8 →
  (ripe_fraction * total * ripe_eaten_fraction + (1 - ripe_fraction) * total * unripe_eaten_fraction : ℚ) = 18 := by
sorry

end NUMINAMATH_CALUDE_oranges_eaten_l182_18298


namespace NUMINAMATH_CALUDE_triangle_side_length_l182_18235

theorem triangle_side_length 
  (a b c : ℝ) 
  (B : ℝ) 
  (h1 : b = 3) 
  (h2 : c = Real.sqrt 6) 
  (h3 : B = π / 3) 
  (h4 : b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B) : 
  a = (Real.sqrt 6 + 3 * Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l182_18235


namespace NUMINAMATH_CALUDE_johnson_carter_tie_l182_18270

/-- Represents the months of the baseball season --/
inductive Month
| March
| April
| May
| June
| July
| August

/-- Represents a player's home run data --/
structure PlayerData where
  monthly_hrs : Month → ℕ

def johnson_data : PlayerData :=
  ⟨λ m => match m with
    | Month.March => 2
    | Month.April => 12
    | Month.May => 18
    | Month.June => 0
    | Month.July => 0
    | Month.August => 12⟩

def carter_data : PlayerData :=
  ⟨λ m => match m with
    | Month.March => 0
    | Month.April => 4
    | Month.May => 8
    | Month.June => 22
    | Month.July => 10
    | Month.August => 0⟩

def total_hrs (player : PlayerData) : ℕ :=
  (player.monthly_hrs Month.March) +
  (player.monthly_hrs Month.April) +
  (player.monthly_hrs Month.May) +
  (player.monthly_hrs Month.June) +
  (player.monthly_hrs Month.July) +
  (player.monthly_hrs Month.August)

theorem johnson_carter_tie :
  total_hrs johnson_data = total_hrs carter_data :=
by sorry

end NUMINAMATH_CALUDE_johnson_carter_tie_l182_18270


namespace NUMINAMATH_CALUDE_constant_value_l182_18209

-- Define the function [[x]]
def bracket (x : ℝ) (c : ℝ) : ℝ := x^2 + 2*x + c

-- State the theorem
theorem constant_value :
  ∃ c : ℝ, (∀ x : ℝ, bracket x c = x^2 + 2*x + c) ∧ bracket 2 c = 12 → c = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_value_l182_18209


namespace NUMINAMATH_CALUDE_sum_of_numerator_and_denominator_l182_18214

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

/-- The repeating decimal 0.4̅5̅ -/
def decimal : ℚ := RepeatingDecimal 4 5

/-- The fraction representation of 0.4̅5̅ in lowest terms -/
def fraction : ℚ := 5 / 11

theorem sum_of_numerator_and_denominator : 
  decimal = fraction ∧ fraction.num + fraction.den = 16 := by sorry

end NUMINAMATH_CALUDE_sum_of_numerator_and_denominator_l182_18214


namespace NUMINAMATH_CALUDE_six_digit_number_puzzle_l182_18253

theorem six_digit_number_puzzle : ∃! n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧
  n % 10 = 7 ∧
  7 * 100000 + n / 10 = 5 * n ∧
  n = 142857 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_puzzle_l182_18253


namespace NUMINAMATH_CALUDE_bridge_length_l182_18278

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_sec : ℝ) :
  train_length = 145 →
  train_speed_kmh = 45 →
  crossing_time_sec = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 230 ∧
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time_sec) - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l182_18278


namespace NUMINAMATH_CALUDE_closet_area_l182_18211

theorem closet_area (diagonal : ℝ) (shorter_side : ℝ)
  (h1 : diagonal = 7)
  (h2 : shorter_side = 4) :
  ∃ (area : ℝ), area = 4 * Real.sqrt 33 ∧ 
  area = shorter_side * Real.sqrt (diagonal^2 - shorter_side^2) :=
by sorry

end NUMINAMATH_CALUDE_closet_area_l182_18211


namespace NUMINAMATH_CALUDE_half_radius_circle_y_l182_18279

-- Define the circles
def circle_x : Real → Prop := λ r => r > 0
def circle_y : Real → Prop := λ r => r > 0

-- Define the theorem
theorem half_radius_circle_y 
  (h_area : ∀ (rx ry : Real), circle_x rx → circle_y ry → π * rx^2 = π * ry^2)
  (h_circum : ∀ (rx : Real), circle_x rx → 2 * π * rx = 10 * π) :
  ∃ (ry : Real), circle_y ry ∧ ry / 2 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_half_radius_circle_y_l182_18279


namespace NUMINAMATH_CALUDE_existence_of_special_integer_l182_18220

theorem existence_of_special_integer (P : Finset Nat) (h_prime : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : Nat, x > 0 ∧ (∀ p : Nat, Nat.Prime p →
    (p ∈ P ↔ ∃ a b : Nat, a > 0 ∧ b > 0 ∧ x = a ^ p + b ^ p)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integer_l182_18220


namespace NUMINAMATH_CALUDE_second_part_speed_l182_18282

/-- Proves that given a trip of 70 kilometers, where the first 35 kilometers are traveled at 48 km/h
    and the average speed of the entire trip is 32 km/h, the speed of the second part of the trip is 24 km/h. -/
theorem second_part_speed (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (average_speed : ℝ)
    (h1 : total_distance = 70)
    (h2 : first_part_distance = 35)
    (h3 : first_part_speed = 48)
    (h4 : average_speed = 32) :
    let second_part_distance := total_distance - first_part_distance
    let total_time := total_distance / average_speed
    let first_part_time := first_part_distance / first_part_speed
    let second_part_time := total_time - first_part_time
    let second_part_speed := second_part_distance / second_part_time
    second_part_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_second_part_speed_l182_18282


namespace NUMINAMATH_CALUDE_mult_inverse_mod_million_mult_inverse_specific_l182_18293

/-- The multiplicative inverse of (A * B) modulo 1,000,000 is 466390 -/
theorem mult_inverse_mod_million : Int → Int → Prop :=
  fun A B => (A * B * 466390) % 1000000 = 1

/-- The theorem holds for A = 123456 and B = 162037 -/
theorem mult_inverse_specific : mult_inverse_mod_million 123456 162037 := by
  sorry

end NUMINAMATH_CALUDE_mult_inverse_mod_million_mult_inverse_specific_l182_18293


namespace NUMINAMATH_CALUDE_sine_characterization_l182_18291

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def IsSymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem sine_characterization (f : ℝ → ℝ) 
  (h1 : IsPeriodic f π)
  (h2 : IsSymmetricAbout f (π/3))
  (h3 : IsIncreasingOn f (-π/6) (π/3)) :
  ∀ x, f x = Real.sin (2*x - π/6) := by
sorry

end NUMINAMATH_CALUDE_sine_characterization_l182_18291


namespace NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l182_18263

theorem max_value_of_4x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 16*x + 8*y + 20) : 
  4*x + 3*y ≤ 40 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l182_18263


namespace NUMINAMATH_CALUDE_tenth_pebble_count_l182_18201

def pebble_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 5
  | 2 => 12
  | 3 => 22
  | n + 4 => pebble_sequence (n + 3) + (3 * (n + 4) - 2)

theorem tenth_pebble_count : pebble_sequence 9 = 145 := by
  sorry

end NUMINAMATH_CALUDE_tenth_pebble_count_l182_18201


namespace NUMINAMATH_CALUDE_photo_album_distribution_l182_18286

/-- Represents the distribution of photos in an album --/
structure PhotoAlbum where
  total_photos : ℕ
  total_pages : ℕ
  photos_per_page_set1 : ℕ
  photos_per_page_set2 : ℕ
  photos_per_page_remaining : ℕ

/-- Theorem stating the correct distribution of pages for the given photo album --/
theorem photo_album_distribution (album : PhotoAlbum) 
  (h1 : album.total_photos = 100)
  (h2 : album.total_pages = 30)
  (h3 : album.photos_per_page_set1 = 3)
  (h4 : album.photos_per_page_set2 = 4)
  (h5 : album.photos_per_page_remaining = 3) :
  ∃ (pages_set1 pages_set2 pages_remaining : ℕ),
    pages_set1 = 0 ∧ 
    pages_set2 = 10 ∧
    pages_remaining = 20 ∧
    pages_set1 + pages_set2 + pages_remaining = album.total_pages ∧
    album.photos_per_page_set1 * pages_set1 + 
    album.photos_per_page_set2 * pages_set2 + 
    album.photos_per_page_remaining * pages_remaining = album.total_photos :=
by
  sorry

end NUMINAMATH_CALUDE_photo_album_distribution_l182_18286


namespace NUMINAMATH_CALUDE_clothing_discount_problem_l182_18236

theorem clothing_discount_problem (discount_rate : ℝ) (savings : ℝ) 
  (h1 : discount_rate = 0.2)
  (h2 : savings = 10)
  (h3 : ∀ x, (1 - discount_rate) * (x + savings) = x) :
  ∃ x, x = 40 ∧ (1 - discount_rate) * (x + savings) = x := by
sorry

end NUMINAMATH_CALUDE_clothing_discount_problem_l182_18236


namespace NUMINAMATH_CALUDE_smallest_white_buttons_l182_18243

theorem smallest_white_buttons (n : ℕ) (h1 : n % 10 = 0) : 
  (n / 2 : ℚ) + (n / 5 : ℚ) + 8 ≤ n → 
  (∃ m : ℕ, m ≥ 1 ∧ (n : ℚ) - ((n / 2 : ℚ) + (n / 5 : ℚ) + 8) = m) →
  (∃ k : ℕ, k ≥ 1 ∧ (30 : ℚ) - ((30 / 2 : ℚ) + (30 / 5 : ℚ) + 8) = k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_white_buttons_l182_18243


namespace NUMINAMATH_CALUDE_one_by_one_square_position_l182_18218

/-- A square on a grid --/
structure Square where
  size : ℕ
  row : ℕ
  col : ℕ

/-- A decomposition of a large square into smaller squares --/
structure Decomposition where
  grid_size : ℕ
  squares : List Square

/-- Predicate to check if a decomposition is valid --/
def is_valid_decomposition (d : Decomposition) : Prop :=
  d.grid_size = 23 ∧
  (∀ s ∈ d.squares, s.size ∈ [1, 2, 3]) ∧
  (d.squares.filter (λ s => s.size = 1)).length = 1

/-- Predicate to check if a position is valid for the 1x1 square --/
def is_valid_position (row col : ℕ) : Prop :=
  row % 6 = 0 ∧ col % 6 = 0 ∧ row ≤ 18 ∧ col ≤ 18

theorem one_by_one_square_position (d : Decomposition) 
  (h : is_valid_decomposition d) :
  ∃ s ∈ d.squares, s.size = 1 ∧ is_valid_position s.row s.col :=
sorry

end NUMINAMATH_CALUDE_one_by_one_square_position_l182_18218


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l182_18208

theorem quadratic_form_ratio (a d : ℝ) : 
  (∀ x, x^2 + 500*x + 2500 = (x + a)^2 + d) →
  d / a = -240 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l182_18208


namespace NUMINAMATH_CALUDE_tan_roots_sum_l182_18225

theorem tan_roots_sum (α β : Real) : 
  (∃ x y : Real, x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ∧ y^2 + 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
   x = Real.tan α ∧ y = Real.tan β) →
  α > -π/2 ∧ α < π/2 ∧ β > -π/2 ∧ β < π/2 →
  α + β = π/3 ∨ α + β = -2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_roots_sum_l182_18225


namespace NUMINAMATH_CALUDE_total_equipment_cost_l182_18210

/-- The number of players on the team -/
def num_players : ℕ := 25

/-- The cost of a jersey in dollars -/
def jersey_cost : ℚ := 25

/-- The cost of shorts in dollars -/
def shorts_cost : ℚ := 15.20

/-- The cost of socks in dollars -/
def socks_cost : ℚ := 6.80

/-- The cost of cleats in dollars -/
def cleats_cost : ℚ := 40

/-- The cost of a water bottle in dollars -/
def water_bottle_cost : ℚ := 12

/-- The total cost of equipment for all players on the team -/
theorem total_equipment_cost : 
  num_players * (jersey_cost + shorts_cost + socks_cost + cleats_cost + water_bottle_cost) = 2475 := by
  sorry

end NUMINAMATH_CALUDE_total_equipment_cost_l182_18210


namespace NUMINAMATH_CALUDE_problem_solution_l182_18223

open Real

theorem problem_solution :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 →
    (∀ (a' b' : ℝ), a' > 0 ∧ b' > 0 → 1/a' + 4/b' ≥ 9/2) ∧
    (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1/a₀ + 4/b₀ = 9/2) ∧
    (∀ (x : ℝ), (∀ (a' b' : ℝ), a' > 0 ∧ b' > 0 → 1/a' + 4/b' ≥ abs (2*x - 1) - abs (x + 1)) →
      -5/2 ≤ x ∧ x ≤ 13/2) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l182_18223


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l182_18247

/-- Given vectors a and b in R², and c defined as a linear combination of a and b,
    prove that if a is perpendicular to c, then the scalar k in the linear combination
    has a specific value. -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (k : ℝ) :
  a = (3, 1) →
  b = (1, 0) →
  let c := a + k • b
  (a.1 * c.1 + a.2 * c.2 = 0) →
  k = -10/3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l182_18247


namespace NUMINAMATH_CALUDE_polynomial_roots_product_l182_18230

theorem polynomial_roots_product (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℤ, x^3 + a*x^2 + b*x + 6*a = (x - r)^2 * (x - s)) ∧ 
   r ≠ s) → 
  |a * b| = 546 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_product_l182_18230


namespace NUMINAMATH_CALUDE_seunghye_number_l182_18233

theorem seunghye_number (x : ℝ) : 10 * x - x = 37.35 → x = 4.15 := by
  sorry

end NUMINAMATH_CALUDE_seunghye_number_l182_18233


namespace NUMINAMATH_CALUDE_three_layer_runner_area_l182_18212

/-- Given three table runners and a table, calculate the area covered by three layers of runner -/
theorem three_layer_runner_area 
  (total_runner_area : ℝ) 
  (table_area : ℝ) 
  (coverage_percent : ℝ) 
  (two_layer_area : ℝ) 
  (h1 : total_runner_area = 224) 
  (h2 : table_area = 175) 
  (h3 : coverage_percent = 0.8) 
  (h4 : two_layer_area = 24) : 
  ∃ (three_layer_area : ℝ), 
    three_layer_area = 12 ∧ 
    total_runner_area = coverage_percent * table_area + two_layer_area + 2 * three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_three_layer_runner_area_l182_18212


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l182_18255

def quadratic_equation (m n x : ℝ) : ℝ := 9 * x^2 - 2 * m * x + n

def has_two_real_roots (m n : ℤ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m n x = 0 ∧ quadratic_equation m n y = 0

def roots_in_interval (m n : ℤ) : Prop :=
  ∀ x : ℝ, quadratic_equation m n x = 0 → 0 < x ∧ x < 1

theorem quadratic_roots_theorem :
  ∀ m n : ℤ, has_two_real_roots m n ∧ roots_in_interval m n ↔ (m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l182_18255


namespace NUMINAMATH_CALUDE_belle_collected_97_stickers_l182_18285

def belle_stickers (carolyn_stickers : ℕ) (difference : ℕ) : ℕ :=
  carolyn_stickers + difference

theorem belle_collected_97_stickers 
  (h1 : belle_stickers 79 18 = 97) : belle_stickers 79 18 = 97 := by
  sorry

end NUMINAMATH_CALUDE_belle_collected_97_stickers_l182_18285


namespace NUMINAMATH_CALUDE_no_three_integer_solutions_l182_18259

theorem no_three_integer_solutions (b : ℤ) : 
  ¬(∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x^2 + b*x + 5 ≤ 0) ∧ (y^2 + b*y + 5 ≤ 0) ∧ (z^2 + b*z + 5 ≤ 0) ∧
    (∀ (w : ℤ), w ≠ x ∧ w ≠ y ∧ w ≠ z → w^2 + b*w + 5 > 0)) :=
by sorry

#check no_three_integer_solutions

end NUMINAMATH_CALUDE_no_three_integer_solutions_l182_18259


namespace NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l182_18289

/-- Given a point P with coordinates (-1, 2), its symmetric point about the x-axis has coordinates (-1, -2) -/
theorem symmetric_point_about_x_axis :
  let P : ℝ × ℝ := (-1, 2)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point P = (-1, -2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l182_18289


namespace NUMINAMATH_CALUDE_square_brush_ratio_l182_18254

/-- A square with side length s and a brush of width w -/
structure SquareAndBrush where
  s : ℝ
  w : ℝ

/-- The painted area is one-third of the square's area -/
def paintedAreaIsOneThird (sb : SquareAndBrush) : Prop :=
  (1/2 * sb.w^2 + (sb.s - sb.w)^2 / 2) = sb.s^2 / 3

/-- The theorem to be proved -/
theorem square_brush_ratio (sb : SquareAndBrush) 
    (h : paintedAreaIsOneThird sb) : 
    sb.s / sb.w = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_brush_ratio_l182_18254


namespace NUMINAMATH_CALUDE_problem_statement_l182_18202

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 3) :
  x * y ≤ 9/8 ∧ 
  4^x + 2^y ≥ 4 * Real.sqrt 2 ∧ 
  x / y + 1 / x ≥ 2/3 + 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l182_18202


namespace NUMINAMATH_CALUDE_f_properties_l182_18245

def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

theorem f_properties :
  (∀ x, f 0 x = f 0 (-x)) ∧
  (∀ a, a > 1/2 → ∀ x, f a x ≥ a + 3/4) ∧
  (∀ a, a ≤ -1/2 → ∀ x, f a x ≥ -a + 3/4) ∧
  (∀ a, -1/2 < a ∧ a ≤ 1/2 → ∀ x, f a x ≥ a^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l182_18245


namespace NUMINAMATH_CALUDE_player_field_time_l182_18262

/-- Proves that each player in a 10-player team plays 36 minutes in a 45-minute match with 8 players always on field --/
theorem player_field_time (team_size : ℕ) (field_players : ℕ) (match_duration : ℕ) 
  (h1 : team_size = 10)
  (h2 : field_players = 8)
  (h3 : match_duration = 45) :
  (field_players * match_duration) / team_size = 36 := by
  sorry

end NUMINAMATH_CALUDE_player_field_time_l182_18262


namespace NUMINAMATH_CALUDE_sector_central_angle_l182_18200

theorem sector_central_angle (radius : ℝ) (area : ℝ) (angle : ℝ) :
  radius = 1 →
  area = 1 →
  area = (1 / 2) * angle * radius^2 →
  angle = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l182_18200


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_diff_sum_of_solutions_eq_eight_l182_18213

theorem sum_of_solutions_squared_diff (a c : ℝ) :
  (∀ x : ℝ, (x - a)^2 = c) → 
  (∃ x₁ x₂ : ℝ, (x₁ - a)^2 = c ∧ (x₂ - a)^2 = c ∧ x₁ + x₂ = 2 * a) :=
by sorry

-- The specific problem
theorem sum_of_solutions_eq_eight :
  (∃ x₁ x₂ : ℝ, (x₁ - 4)^2 = 49 ∧ (x₂ - 4)^2 = 49 ∧ x₁ + x₂ = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_diff_sum_of_solutions_eq_eight_l182_18213


namespace NUMINAMATH_CALUDE_grapes_purchased_l182_18261

/-- Given the cost of grapes, amount of mangoes, cost of mangoes, and total paid,
    calculate the amount of grapes purchased. -/
theorem grapes_purchased 
  (grape_cost : ℕ) 
  (mango_amount : ℕ) 
  (mango_cost : ℕ) 
  (total_paid : ℕ) : 
  grape_cost = 70 →
  mango_amount = 9 →
  mango_cost = 50 →
  total_paid = 1010 →
  (total_paid - mango_amount * mango_cost) / grape_cost = 8 :=
by
  sorry

#check grapes_purchased

end NUMINAMATH_CALUDE_grapes_purchased_l182_18261


namespace NUMINAMATH_CALUDE_square_plus_one_positive_l182_18258

theorem square_plus_one_positive (a : ℝ) : a^2 + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_positive_l182_18258


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l182_18292

def f (x : ℤ) : ℤ := x^3 - 4*x^2 - 7*x + 10

theorem integer_roots_of_cubic :
  {x : ℤ | f x = 0} = {1, -2, 5} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l182_18292


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l182_18264

theorem smallest_n_for_candy_purchase : ∃ (n : ℕ+), 
  (∀ (k : ℕ+), 24 * k = Nat.lcm (Nat.lcm 10 15) 18 → n ≤ k) ∧ 
  24 * n = Nat.lcm (Nat.lcm 10 15) 18 ∧ 
  n = 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l182_18264


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l182_18271

theorem existence_of_special_sequence :
  ∃ (a : ℕ → ℕ),
    (∀ k, a k < a (k + 1)) ∧
    (∀ n : ℤ, ∃ N : ℕ, ∀ k > N, ¬ Prime (a k + n)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l182_18271


namespace NUMINAMATH_CALUDE_incorrect_guess_is_20th_bear_prove_incorrect_guess_is_20th_bear_l182_18266

/-- Represents the color of a bear -/
inductive BearColor
| White
| Brown
| Black

/-- Represents a row of 1000 bears -/
def BearRow := Fin 1000 → BearColor

/-- Predicate to check if three consecutive bears have all three colors -/
def hasAllColors (row : BearRow) (i : Fin 998) : Prop :=
  ∃ (c1 c2 c3 : BearColor), 
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    row i = c1 ∧ row (i + 1) = c2 ∧ row (i + 2) = c3

/-- The main theorem stating that the 20th bear's color must be the incorrect guess -/
theorem incorrect_guess_is_20th_bear (row : BearRow) : Prop :=
  (∀ i : Fin 998, hasAllColors row i) →
  (row 1 = BearColor.White) →
  (row 399 = BearColor.Black) →
  (row 599 = BearColor.Brown) →
  (row 799 = BearColor.White) →
  (row 19 ≠ BearColor.Brown)

-- The proof of the theorem
theorem prove_incorrect_guess_is_20th_bear :
  ∃ (row : BearRow), incorrect_guess_is_20th_bear row :=
sorry

end NUMINAMATH_CALUDE_incorrect_guess_is_20th_bear_prove_incorrect_guess_is_20th_bear_l182_18266


namespace NUMINAMATH_CALUDE_least_divisible_by_first_eight_l182_18215

theorem least_divisible_by_first_eight : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, k > 0 ∧ k ≤ 8 → k ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, k > 0 ∧ k ≤ 8 → k ∣ m) → n ≤ m) ∧
  n = 840 :=
by sorry

end NUMINAMATH_CALUDE_least_divisible_by_first_eight_l182_18215


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l182_18257

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ),
    x^2 + a*x + b = 0 ↔ x = (2 - Complex.I) ∨ x = (2 + Complex.I) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l182_18257


namespace NUMINAMATH_CALUDE_david_recreation_spending_l182_18207

theorem david_recreation_spending :
  ∀ (last_week_wages : ℝ) (last_week_percent : ℝ),
    last_week_percent > 0 →
    (0.7 * last_week_wages * 0.2) = (0.7 * (last_week_percent / 100) * last_week_wages) →
    last_week_percent = 20 := by
  sorry

end NUMINAMATH_CALUDE_david_recreation_spending_l182_18207


namespace NUMINAMATH_CALUDE_correct_assignment_calculation_symbol_l182_18251

/-- Enum representing different flowchart symbols -/
inductive FlowchartSymbol
  | StartEnd
  | Decision
  | AssignmentCalculation
  | InputOutput

/-- Function that returns the correct symbol for assignment and calculation -/
def assignmentCalculationSymbol : FlowchartSymbol := FlowchartSymbol.AssignmentCalculation

/-- Theorem stating that the assignment and calculation symbol is correct -/
theorem correct_assignment_calculation_symbol :
  assignmentCalculationSymbol = FlowchartSymbol.AssignmentCalculation := by
  sorry

#check correct_assignment_calculation_symbol

end NUMINAMATH_CALUDE_correct_assignment_calculation_symbol_l182_18251


namespace NUMINAMATH_CALUDE_pascal_parallelogram_sum_l182_18256

/-- Represents a position in Pascal's triangle -/
structure Position :=
  (row : ℕ)
  (col : ℕ)
  (h : col ≤ row)

/-- The value at a given position in Pascal's triangle -/
def pascal_value (pos : Position) : ℕ := sorry

/-- The parallelogram bounded by right and left diagonals intersecting at a given position -/
def parallelogram (pos : Position) : Set Position := sorry

/-- The sum of all numbers in the parallelogram -/
def parallelogram_sum (pos : Position) : ℕ := sorry

/-- Theorem: For any number a in Pascal's triangle, a - 1 equals the sum of all numbers
    in the parallelogram bounded by the right and left diagonals intersecting at a,
    excluding the diagonals themselves -/
theorem pascal_parallelogram_sum (pos : Position) :
  pascal_value pos - 1 = parallelogram_sum pos := by sorry

end NUMINAMATH_CALUDE_pascal_parallelogram_sum_l182_18256


namespace NUMINAMATH_CALUDE_order_of_exponentials_l182_18240

theorem order_of_exponentials :
  let a : ℝ := Real.rpow 0.6 4.2
  let b : ℝ := Real.rpow 0.7 4.2
  let c : ℝ := Real.rpow 0.6 5.1
  b > a ∧ a > c :=
by sorry

end NUMINAMATH_CALUDE_order_of_exponentials_l182_18240


namespace NUMINAMATH_CALUDE_min_value_expression_l182_18265

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  3 * a^2 + 3 * b^2 + 1 / (a + b)^2 + 4 / (a^2 * b^2) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l182_18265


namespace NUMINAMATH_CALUDE_walking_scenario_solution_l182_18275

/-- Represents the walking scenario between two people --/
structure WalkingScenario where
  speed_A : ℝ  -- Speed of person A in meters per minute
  time_A_start : ℝ  -- Time person A has been walking when B starts
  speed_diff : ℝ  -- Speed difference between person B and person A
  time_diff_AC_CB : ℝ  -- Time difference between A to C and C to B for person A
  time_diff_CA_BC : ℝ  -- Time difference between C to A and B to C for person B

/-- The main theorem about the walking scenario --/
theorem walking_scenario_solution (w : WalkingScenario) 
  (h_speed_diff : w.speed_diff = 30)
  (h_time_A_start : w.time_A_start = 5.5)
  (h_time_diff_AC_CB : w.time_diff_AC_CB = 4)
  (h_time_diff_CA_BC : w.time_diff_CA_BC = 3) :
  ∃ (time_A_to_C : ℝ) (distance_AB : ℝ),
    time_A_to_C = 10 ∧ distance_AB = 1440 := by
  sorry


end NUMINAMATH_CALUDE_walking_scenario_solution_l182_18275


namespace NUMINAMATH_CALUDE_pot_filling_time_l182_18284

-- Define the constants
def drops_per_minute : ℕ := 3
def ml_per_drop : ℕ := 20
def pot_capacity_liters : ℕ := 3

-- Define the theorem
theorem pot_filling_time :
  (pot_capacity_liters * 1000) / (drops_per_minute * ml_per_drop) = 50 := by
  sorry

end NUMINAMATH_CALUDE_pot_filling_time_l182_18284


namespace NUMINAMATH_CALUDE_roulette_sectors_l182_18205

def roulette_wheel (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 10 ∧ 
  (1 - (5 / n)^2 : ℚ) = 3/4

theorem roulette_sectors : ∃ (n : ℕ), roulette_wheel n ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_roulette_sectors_l182_18205


namespace NUMINAMATH_CALUDE_max_of_roots_l182_18273

theorem max_of_roots (α β γ : ℝ) 
  (sum_eq : α + β + γ = 14)
  (sum_squares_eq : α^2 + β^2 + γ^2 = 84)
  (sum_cubes_eq : α^3 + β^3 + γ^3 = 584) :
  max α (max β γ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_of_roots_l182_18273


namespace NUMINAMATH_CALUDE_z_modulus_range_l182_18299

-- Define the complex number z
def z (a : ℝ) : ℂ := Complex.mk (a - 2) (a + 1)

-- Define the condition for z to be in the second quadrant
def second_quadrant (a : ℝ) : Prop := a - 2 < 0 ∧ a + 1 > 0

-- State the theorem
theorem z_modulus_range :
  ∃ (min max : ℝ), min = 3 * Real.sqrt 2 / 2 ∧ max = 3 ∧
  ∀ a : ℝ, second_quadrant a →
    Complex.abs (z a) ≥ min ∧ Complex.abs (z a) ≤ max ∧
    (∃ a₁ a₂ : ℝ, second_quadrant a₁ ∧ second_quadrant a₂ ∧
      Complex.abs (z a₁) = min ∧ Complex.abs (z a₂) = max) :=
by sorry

end NUMINAMATH_CALUDE_z_modulus_range_l182_18299


namespace NUMINAMATH_CALUDE_fermat_primes_totient_divisor_641_l182_18274

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

theorem fermat_primes_totient (k : ℕ) : 
  (phi (sigma (2^k)) = 2^k) ↔ k ∈ ({1, 3, 7, 15, 31} : Set ℕ) := by
  sorry

/-- 641 is a divisor of 2^32 + 1 -/
theorem divisor_641 : ∃ m : ℕ, 2^32 + 1 = 641 * m := by
  sorry

end NUMINAMATH_CALUDE_fermat_primes_totient_divisor_641_l182_18274


namespace NUMINAMATH_CALUDE_sum_of_four_solution_values_l182_18269

-- Define the polynomial function f
noncomputable def f (x : ℝ) : ℝ := 
  (x - 5) * (x - 3) * (x - 1) * (x + 1) * (x + 3) * (x + 5) / 315 - 3.4

-- Define the property of having exactly 4 solutions
def has_four_solutions (c : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (f x₁ = c ∧ f x₂ = c ∧ f x₃ = c ∧ f x₄ = c) ∧
    (∀ x, f x = c → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

-- Theorem statement
theorem sum_of_four_solution_values :
  ∃ (a b : ℤ), has_four_solutions (a : ℝ) ∧ has_four_solutions (b : ℝ) ∧ a + b = -7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_solution_values_l182_18269


namespace NUMINAMATH_CALUDE_rectangle_area_l182_18228

theorem rectangle_area (square_area : ℝ) (rect_length rect_width : ℝ) : 
  square_area = 36 →
  4 * square_area.sqrt = 2 * (rect_length + rect_width) →
  rect_length = 3 * rect_width →
  rect_length * rect_width = 27 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l182_18228


namespace NUMINAMATH_CALUDE_sqrt_cos_squared_660_l182_18276

theorem sqrt_cos_squared_660 : Real.sqrt (Real.cos (660 * π / 180) ^ 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cos_squared_660_l182_18276


namespace NUMINAMATH_CALUDE_absolute_value_zero_implies_negative_three_l182_18203

theorem absolute_value_zero_implies_negative_three (a : ℝ) :
  |a + 3| = 0 → a = -3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_zero_implies_negative_three_l182_18203


namespace NUMINAMATH_CALUDE_train_length_proof_l182_18260

/-- Given a train and a platform with equal length, if the train crosses the platform
    in 60 seconds at a speed of 30 m/s, then the length of the train is 900 meters. -/
theorem train_length_proof (train_length platform_length : ℝ) 
  (speed : ℝ) (time : ℝ) (h1 : train_length = platform_length) 
  (h2 : speed = 30) (h3 : time = 60) :
  train_length = 900 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l182_18260


namespace NUMINAMATH_CALUDE_circular_film_radius_l182_18216

/-- The radius of a circular film formed by a liquid poured from a rectangular container onto a water surface -/
theorem circular_film_radius 
  (length width height thickness : ℝ) 
  (h_length : length = 10)
  (h_width : width = 4)
  (h_height : height = 8)
  (h_thickness : thickness = 0.15)
  (h_positive : length > 0 ∧ width > 0 ∧ height > 0 ∧ thickness > 0) :
  let volume := length * width * height
  let radius := Real.sqrt (volume / (thickness * Real.pi))
  radius = Real.sqrt (2133.33 / Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_circular_film_radius_l182_18216


namespace NUMINAMATH_CALUDE_cube_base_diagonal_l182_18238

/-- Given a cube with space diagonal length of 5 units, 
    the diagonal of its base has length 5 * sqrt(2/3) units. -/
theorem cube_base_diagonal (c : Real) (h : c > 0) 
  (space_diagonal : c * Real.sqrt 3 = 5) : 
  c * Real.sqrt 2 = 5 * Real.sqrt (2/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_base_diagonal_l182_18238


namespace NUMINAMATH_CALUDE_vertical_angles_congruence_equivalence_l182_18248

-- Define what it means for angles to be vertical
def are_vertical_angles (a b : Angle) : Prop := sorry

-- Define what it means for angles to be congruent
def are_congruent (a b : Angle) : Prop := sorry

-- The theorem to prove
theorem vertical_angles_congruence_equivalence :
  (∀ a b : Angle, are_vertical_angles a b → are_congruent a b) ↔
  (∀ a b : Angle, are_vertical_angles a b → are_congruent a b) :=
sorry

end NUMINAMATH_CALUDE_vertical_angles_congruence_equivalence_l182_18248


namespace NUMINAMATH_CALUDE_zoe_average_speed_l182_18246

/-- Represents the hiking scenario with Chantal and Zoe -/
structure HikingScenario where
  d : ℝ  -- Represents one-third of the total distance
  chantal_speed1 : ℝ  -- Chantal's speed for the first third
  chantal_speed2 : ℝ  -- Chantal's speed for the rocky part
  chantal_speed3 : ℝ  -- Chantal's speed for descent on rocky part

/-- The theorem stating Zoe's average speed -/
theorem zoe_average_speed (h : HikingScenario) 
  (h_chantal_speed1 : h.chantal_speed1 = 5)
  (h_chantal_speed2 : h.chantal_speed2 = 3)
  (h_chantal_speed3 : h.chantal_speed3 = 4) :
  let total_time := h.d / h.chantal_speed1 + h.d / h.chantal_speed2 + h.d / h.chantal_speed2 + h.d / h.chantal_speed3
  (h.d / total_time) = 60 / 47 := by
  sorry

#check zoe_average_speed

end NUMINAMATH_CALUDE_zoe_average_speed_l182_18246


namespace NUMINAMATH_CALUDE_set_union_problem_l182_18267

theorem set_union_problem (a b : ℕ) :
  let M : Set ℕ := {3, 2^a}
  let N : Set ℕ := {a, b}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l182_18267


namespace NUMINAMATH_CALUDE_koi_count_after_three_weeks_l182_18232

/-- The number of koi fish after three weeks of adding fish to a tank -/
def koiAfterThreeWeeks (initialTotal : ℕ) (daysAdding : ℕ) (koiAddedPerDay : ℕ) (goldfishAddedPerDay : ℕ) (finalGoldfish : ℕ) : ℕ :=
  let initialGoldfish := finalGoldfish - daysAdding * goldfishAddedPerDay
  let initialKoi := initialTotal - initialGoldfish
  initialKoi + daysAdding * koiAddedPerDay

/-- Theorem stating the number of koi fish after three weeks -/
theorem koi_count_after_three_weeks :
  koiAfterThreeWeeks 280 21 2 5 200 = 227 := by
  sorry

end NUMINAMATH_CALUDE_koi_count_after_three_weeks_l182_18232


namespace NUMINAMATH_CALUDE_unique_terminating_decimal_pair_one_fourth_is_terminating_one_fifth_is_terminating_l182_18221

/-- A number is a terminating decimal if it can be expressed as a fraction with denominator of the form 2^a * 5^b -/
def IsTerminatingDecimal (x : ℚ) : Prop :=
  ∃ (a b : ℕ), ∃ (m : ℤ), x = m / (2^a * 5^b)

/-- The main theorem stating that 4 is the only natural number n > 1 such that
    both 1/n and 1/(n+1) are terminating decimals -/
theorem unique_terminating_decimal_pair :
  ∀ n : ℕ, n > 1 →
    (IsTerminatingDecimal (1 / n) ∧ IsTerminatingDecimal (1 / (n + 1))) ↔ n = 4 := by
  sorry

/-- Verifies that 1/4 is a terminating decimal -/
theorem one_fourth_is_terminating : IsTerminatingDecimal (1 / 4) := by
  sorry

/-- Verifies that 1/5 is a terminating decimal -/
theorem one_fifth_is_terminating : IsTerminatingDecimal (1 / 5) := by
  sorry

end NUMINAMATH_CALUDE_unique_terminating_decimal_pair_one_fourth_is_terminating_one_fifth_is_terminating_l182_18221


namespace NUMINAMATH_CALUDE_expand_and_simplify_l182_18237

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 8) = x^2 + 5*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l182_18237


namespace NUMINAMATH_CALUDE_no_reversed_arithmetic_progression_l182_18280

/-- Function that returns the odd positive integer obtained by reversing the binary representation of n -/
def r (n : Nat) : Nat :=
  sorry

/-- Predicate to check if a sequence is an arithmetic progression -/
def isArithmeticProgression (s : List Nat) : Prop :=
  sorry

theorem no_reversed_arithmetic_progression :
  ¬∃ (a : Fin 8 → Nat),
    (∀ i : Fin 8, Odd (a i)) ∧
    (∀ i j : Fin 8, i < j → a i < a j) ∧
    isArithmeticProgression (List.ofFn a) ∧
    isArithmeticProgression (List.map r (List.ofFn a)) :=
  sorry

end NUMINAMATH_CALUDE_no_reversed_arithmetic_progression_l182_18280


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l182_18206

/-- The expansion of (1 - 1/(2x))^6 in terms of 1/x -/
def expansion (x : ℝ) (a : Fin 7 → ℝ) : Prop :=
  (1 - 1/(2*x))^6 = a 0 + a 1 * (1/x) + a 2 * (1/x)^2 + a 3 * (1/x)^3 + 
                    a 4 * (1/x)^4 + a 5 * (1/x)^5 + a 6 * (1/x)^6

/-- The sum of the coefficients a_3 and a_4 is equal to -25/16 -/
theorem sum_of_coefficients (x : ℝ) (a : Fin 7 → ℝ) 
  (h : expansion x a) : a 3 + a 4 = -25/16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l182_18206


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l182_18227

theorem circular_seating_arrangement (n : ℕ) (π : Fin (2*n) → Fin (2*n)) 
  (hπ : Function.Bijective π) : 
  ∃ (i j : Fin (2*n)), i ≠ j ∧ (π i - π j) % (2*n) = (i - j) % (2*n) := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l182_18227


namespace NUMINAMATH_CALUDE_packaging_combinations_l182_18239

/-- The number of varieties of wrapping paper -/
def wrapping_paper : ℕ := 10

/-- The number of colors of ribbon -/
def ribbon : ℕ := 5

/-- The number of types of gift cards -/
def gift_cards : ℕ := 4

/-- The number of options for decorative stickers -/
def stickers : ℕ := 2

/-- The total number of packaging combinations -/
def total_combinations : ℕ := wrapping_paper * ribbon * gift_cards * stickers

theorem packaging_combinations : total_combinations = 400 := by
  sorry

end NUMINAMATH_CALUDE_packaging_combinations_l182_18239


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l182_18226

def problem (b : ℝ × ℝ) : Prop :=
  let a : ℝ × ℝ := (2, 1)
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = 4^2 ∧ 
  a.1 * b.1 + a.2 * b.2 = 1 →
  b.1^2 + b.2^2 = 3^2

theorem vector_magnitude_problem : ∀ b : ℝ × ℝ, problem b :=
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l182_18226


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l182_18229

theorem no_positive_integer_solution :
  ¬ ∃ (n : ℕ+) (p : ℕ), Nat.Prime p ∧ n.val^2 - 45*n.val + 520 = p := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l182_18229


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l182_18272

theorem opposite_of_negative_five : 
  (-(- 5 : ℤ)) = (5 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l182_18272


namespace NUMINAMATH_CALUDE_penny_whale_species_l182_18296

theorem penny_whale_species (shark_species eel_species total_species : ℕ) 
  (h1 : shark_species = 35)
  (h2 : eel_species = 15)
  (h3 : total_species = 55) :
  total_species - (shark_species + eel_species) = 5 := by
sorry

end NUMINAMATH_CALUDE_penny_whale_species_l182_18296


namespace NUMINAMATH_CALUDE_integer_power_sum_l182_18241

theorem integer_power_sum (x : ℝ) (h : ∃ (k : ℤ), x + 1/x = k) :
  ∀ (n : ℕ), ∃ (m : ℤ), x^n + 1/(x^n) = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l182_18241


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l182_18287

theorem power_mod_seventeen : 2^2023 % 17 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l182_18287


namespace NUMINAMATH_CALUDE_ship_speed_calculation_l182_18281

theorem ship_speed_calculation (total_distance : ℝ) (travel_time : ℝ) (backward_distance : ℝ) :
  travel_time = 20 ∧
  backward_distance = 200 ∧
  total_distance / 2 - total_distance / 3 = backward_distance →
  (total_distance / 2) / travel_time = 30 := by
sorry

end NUMINAMATH_CALUDE_ship_speed_calculation_l182_18281


namespace NUMINAMATH_CALUDE_proposition_evaluation_l182_18249

open Real

-- Define proposition p
def p : Prop := ∃ x₀ : ℝ, x₀ - 2 > log x₀

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- Theorem statement
theorem proposition_evaluation :
  (p ∧ q) ∧ ¬(p ∧ ¬q) ∧ (¬p ∨ q) ∧ (p ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_proposition_evaluation_l182_18249
