import Mathlib

namespace NUMINAMATH_CALUDE_neg_two_plus_one_eq_neg_one_l2613_261399

theorem neg_two_plus_one_eq_neg_one : (-2) + 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_neg_two_plus_one_eq_neg_one_l2613_261399


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9689_l2613_261304

theorem largest_prime_factor_of_9689 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 9689 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 9689 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9689_l2613_261304


namespace NUMINAMATH_CALUDE_triangle_property_l2613_261350

open Real

theorem triangle_property (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) 
  (hABC : A + B + C = π) (hSin : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * π / 3 ∧ 
  (∃ (a b c : ℝ), a = 3 ∧ 
    sin A / a = sin B / b ∧ 
    sin A / a = sin C / c ∧ 
    a + b + c ≤ 3 + 2 * Real.sqrt 3) := by
  sorry

#check triangle_property

end NUMINAMATH_CALUDE_triangle_property_l2613_261350


namespace NUMINAMATH_CALUDE_sum_fifth_sixth_row20_l2613_261364

def pascal_triangle (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem sum_fifth_sixth_row20 : 
  pascal_triangle 20 4 + pascal_triangle 20 5 = 20349 := by sorry

end NUMINAMATH_CALUDE_sum_fifth_sixth_row20_l2613_261364


namespace NUMINAMATH_CALUDE_two_lines_forming_angle_with_skew_lines_l2613_261330

/-- Represents a line in 3D space -/
structure Line3D where
  -- We'll use a simplified representation of a line
  -- More details could be added if needed

/-- Represents a point in 3D space -/
structure Point3D where
  -- We'll use a simplified representation of a point
  -- More details could be added if needed

/-- The angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- Whether two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Whether a line passes through a point -/
def line_passes_through (l : Line3D) (p : Point3D) : Prop :=
  sorry

/-- The main theorem -/
theorem two_lines_forming_angle_with_skew_lines 
  (a b : Line3D) (P : Point3D) 
  (h_skew : are_skew a b) 
  (h_angle : angle_between_lines a b = 50) : 
  ∃! (s : Finset Line3D), 
    s.card = 2 ∧ 
    ∀ l ∈ s, line_passes_through l P ∧ 
              angle_between_lines l a = 30 ∧ 
              angle_between_lines l b = 30 :=
sorry

end NUMINAMATH_CALUDE_two_lines_forming_angle_with_skew_lines_l2613_261330


namespace NUMINAMATH_CALUDE_negative_sqrt_geq_a_plus_sqrt_neg_two_l2613_261303

theorem negative_sqrt_geq_a_plus_sqrt_neg_two (a : ℝ) (h : a > 0) :
  -Real.sqrt a ≥ a + Real.sqrt (-2) :=
by sorry

end NUMINAMATH_CALUDE_negative_sqrt_geq_a_plus_sqrt_neg_two_l2613_261303


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2613_261355

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2613_261355


namespace NUMINAMATH_CALUDE_least_x_value_l2613_261346

theorem least_x_value (x y : ℤ) (h : x * y + 6 * x + 8 * y = -4) :
  ∀ z : ℤ, z ≥ -52 ∨ ¬∃ w : ℤ, z * w + 6 * z + 8 * w = -4 :=
sorry

end NUMINAMATH_CALUDE_least_x_value_l2613_261346


namespace NUMINAMATH_CALUDE_initial_boarders_count_prove_initial_boarders_count_l2613_261398

/-- Proves that the initial number of boarders is 120 given the conditions of the problem -/
theorem initial_boarders_count : ℕ → ℕ → Prop :=
  fun initial_boarders initial_day_students =>
    -- Initial ratio of boarders to day students is 2:5
    (initial_boarders : ℚ) / initial_day_students = 2 / 5 →
    -- After 30 new boarders join, the ratio becomes 1:2
    ((initial_boarders : ℚ) + 30) / initial_day_students = 1 / 2 →
    -- The initial number of boarders is 120
    initial_boarders = 120

-- The proof of the theorem
theorem prove_initial_boarders_count : ∃ (initial_boarders initial_day_students : ℕ),
  initial_boarders_count initial_boarders initial_day_students :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_initial_boarders_count_prove_initial_boarders_count_l2613_261398


namespace NUMINAMATH_CALUDE_units_digit_of_n_l2613_261388

/-- Given two natural numbers m and n, returns true if m has units digit 9 -/
def has_units_digit_9 (m : ℕ) : Prop :=
  m % 10 = 9

/-- Given two natural numbers m and n, returns true if their product equals 31^6 -/
def product_equals_31_pow_6 (m n : ℕ) : Prop :=
  m * n = 31^6

/-- Theorem stating that if m has units digit 9 and m * n = 31^6, then n has units digit 9 -/
theorem units_digit_of_n (m n : ℕ) 
  (h1 : has_units_digit_9 m) 
  (h2 : product_equals_31_pow_6 m n) : 
  n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l2613_261388


namespace NUMINAMATH_CALUDE_largest_class_size_l2613_261327

theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (diff : ℕ) : 
  total_students = 140 → num_classes = 5 → diff = 2 →
  ∃ x : ℕ, x = 32 ∧ 
    (x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff) = total_students) :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l2613_261327


namespace NUMINAMATH_CALUDE_free_younger_son_time_l2613_261353

/-- The time required to cut all strands of duct tape -/
def cut_time (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℚ :=
  total_strands / (hannah_rate + son_rate)

/-- Theorem stating that it takes 2 minutes to cut 22 strands of duct tape -/
theorem free_younger_son_time :
  cut_time 22 8 3 = 2 := by sorry

end NUMINAMATH_CALUDE_free_younger_son_time_l2613_261353


namespace NUMINAMATH_CALUDE_total_graduation_messages_l2613_261349

def number_of_students : ℕ := 40

theorem total_graduation_messages :
  (number_of_students * (number_of_students - 1)) / 2 = 1560 :=
by sorry

end NUMINAMATH_CALUDE_total_graduation_messages_l2613_261349


namespace NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l2613_261362

theorem one_fourth_in_one_eighth : (1 / 8 : ℚ) / (1 / 4 : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l2613_261362


namespace NUMINAMATH_CALUDE_min_power_congruence_l2613_261380

theorem min_power_congruence :
  ∃ (m n : ℕ), 
    n > m ∧ 
    m ≥ 1 ∧ 
    42^n % 100 = 42^m % 100 ∧
    (∀ (m' n' : ℕ), n' > m' ∧ m' ≥ 1 ∧ 42^n' % 100 = 42^m' % 100 → m + n ≤ m' + n') ∧
    m = 2 ∧
    n = 22 :=
by sorry

end NUMINAMATH_CALUDE_min_power_congruence_l2613_261380


namespace NUMINAMATH_CALUDE_elisa_lap_time_improvement_l2613_261385

/-- Calculates the improvement in lap time given current and previous swimming performance -/
def lap_time_improvement (current_laps : ℕ) (current_time : ℕ) (previous_laps : ℕ) (previous_time : ℕ) : ℚ :=
  (previous_time : ℚ) / (previous_laps : ℚ) - (current_time : ℚ) / (current_laps : ℚ)

/-- Proves that Elisa's lap time improvement is 0.5 minutes per lap -/
theorem elisa_lap_time_improvement :
  lap_time_improvement 15 30 20 50 = 1/2 := by sorry

end NUMINAMATH_CALUDE_elisa_lap_time_improvement_l2613_261385


namespace NUMINAMATH_CALUDE_workshop_average_age_l2613_261365

theorem workshop_average_age (total_members : ℕ) (overall_avg : ℝ) 
  (num_women num_men num_speakers : ℕ) (women_avg men_avg : ℝ) : 
  total_members = 50 →
  overall_avg = 22 →
  num_women = 25 →
  num_men = 20 →
  num_speakers = 5 →
  women_avg = 20 →
  men_avg = 25 →
  (total_members : ℝ) * overall_avg = 
    (num_women : ℝ) * women_avg + (num_men : ℝ) * men_avg + (num_speakers : ℝ) * ((total_members : ℝ) * overall_avg - (num_women : ℝ) * women_avg - (num_men : ℝ) * men_avg) / (num_speakers : ℝ) →
  ((total_members : ℝ) * overall_avg - (num_women : ℝ) * women_avg - (num_men : ℝ) * men_avg) / (num_speakers : ℝ) = 20 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_age_l2613_261365


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l2613_261386

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_part_of_product : Complex.im ((1 - 2*i) * i) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l2613_261386


namespace NUMINAMATH_CALUDE_decimal_parts_fraction_decimal_parts_fraction_proof_l2613_261367

theorem decimal_parts_fraction : ℝ → Prop := 
  fun x => let a : ℤ := ⌊2 + Real.sqrt 2⌋
           let b : ℝ := 2 + Real.sqrt 2 - a
           let c : ℤ := ⌊4 - Real.sqrt 2⌋
           let d : ℝ := 4 - Real.sqrt 2 - c
           (b + d) / (a * c) = 1/6

theorem decimal_parts_fraction_proof : decimal_parts_fraction 2 := by
  sorry

end NUMINAMATH_CALUDE_decimal_parts_fraction_decimal_parts_fraction_proof_l2613_261367


namespace NUMINAMATH_CALUDE_farm_bird_difference_l2613_261373

/-- Given a farm with chickens, ducks, and geese, calculate the difference between
    the combined number of chickens and geese and the number of ducks. -/
theorem farm_bird_difference (chickens ducks geese : ℕ) : 
  chickens = 42 →
  ducks = 48 →
  geese = chickens →
  chickens + geese - ducks = 36 := by
sorry

end NUMINAMATH_CALUDE_farm_bird_difference_l2613_261373


namespace NUMINAMATH_CALUDE_colin_speed_l2613_261368

/-- Given the relationships between the speeds of Colin, Brandon, Tony, Bruce, and Daniel,
    prove that Colin's speed is 8 miles per hour when Bruce's speed is 1 mile per hour. -/
theorem colin_speed (bruce tony brandon colin daniel : ℝ) : 
  bruce = 1 →
  tony = 2 * bruce →
  brandon = (1/3) * tony^2 →
  colin = 6 * brandon →
  daniel = (1/4) * colin →
  colin = 8 := by
sorry

end NUMINAMATH_CALUDE_colin_speed_l2613_261368


namespace NUMINAMATH_CALUDE_sandy_clothes_cost_l2613_261317

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℚ := 13.99

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℚ := 12.14

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℚ := 7.43

/-- The total amount Sandy spent on clothes -/
def total_cost : ℚ := shorts_cost + shirt_cost + jacket_cost

theorem sandy_clothes_cost : total_cost = 33.56 := by sorry

end NUMINAMATH_CALUDE_sandy_clothes_cost_l2613_261317


namespace NUMINAMATH_CALUDE_max_cakes_l2613_261345

/-- Represents the configuration of cuts on a rectangular cake -/
structure CakeCut where
  rows : Nat
  columns : Nat

/-- Calculates the total number of cake pieces after cutting -/
def totalPieces (cut : CakeCut) : Nat :=
  (cut.rows + 1) * (cut.columns + 1)

/-- Calculates the number of interior pieces -/
def interiorPieces (cut : CakeCut) : Nat :=
  (cut.rows - 1) * (cut.columns - 1)

/-- Calculates the number of perimeter pieces -/
def perimeterPieces (cut : CakeCut) : Nat :=
  2 * (cut.rows + cut.columns)

/-- Checks if the cutting configuration satisfies the given condition -/
def isValidCut (cut : CakeCut) : Prop :=
  interiorPieces cut = perimeterPieces cut + 1

/-- The main theorem stating the maximum number of cakes -/
theorem max_cakes : ∃ (cut : CakeCut), isValidCut cut ∧ 
  totalPieces cut = 65 ∧ 
  (∀ (other : CakeCut), isValidCut other → totalPieces other ≤ 65) :=
sorry

end NUMINAMATH_CALUDE_max_cakes_l2613_261345


namespace NUMINAMATH_CALUDE_bakers_friend_cakes_prove_bakers_friend_cakes_l2613_261316

/-- Given that Baker initially made 169 cakes and has 32 cakes left,
    prove that the number of cakes bought by Baker's friend is 137. -/
theorem bakers_friend_cakes : ℕ → ℕ → ℕ → Prop :=
  fun initial_cakes remaining_cakes cakes_bought =>
    initial_cakes = 169 →
    remaining_cakes = 32 →
    cakes_bought = initial_cakes - remaining_cakes →
    cakes_bought = 137

/-- Proof of the theorem -/
theorem prove_bakers_friend_cakes :
  bakers_friend_cakes 169 32 137 := by
  sorry

end NUMINAMATH_CALUDE_bakers_friend_cakes_prove_bakers_friend_cakes_l2613_261316


namespace NUMINAMATH_CALUDE_greatest_power_of_five_l2613_261312

/-- The number of divisors function -/
noncomputable def num_divisors (n : ℕ) : ℕ := sorry

theorem greatest_power_of_five (n : ℕ) 
  (h1 : n > 0)
  (h2 : num_divisors n = 72)
  (h3 : num_divisors (5 * n) = 90) :
  ∃ (k : ℕ) (m : ℕ), n = 5^k * m ∧ m % 5 ≠ 0 ∧ k = 3 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_five_l2613_261312


namespace NUMINAMATH_CALUDE_equal_segments_imply_equal_x_y_l2613_261383

/-- Given two pairs of equal lengths (a₁, a₂) and (b₁, b₂), prove that x = y. -/
theorem equal_segments_imply_equal_x_y (a₁ a₂ b₁ b₂ x y : ℝ) 
  (h1 : a₁ = a₂) (h2 : b₁ = b₂) : x = y := by
  sorry

end NUMINAMATH_CALUDE_equal_segments_imply_equal_x_y_l2613_261383


namespace NUMINAMATH_CALUDE_parabola_directrix_m_l2613_261319

/-- Given a parabola with equation y = mx² and directrix y = 1/8, prove that m = -2 -/
theorem parabola_directrix_m (m : ℝ) : 
  (∀ x y : ℝ, y = m * x^2) →  -- Parabola equation
  (∃ k : ℝ, k = 1/8 ∧ ∀ x : ℝ, k = -(1 / (4 * m))) →  -- Directrix equation
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_m_l2613_261319


namespace NUMINAMATH_CALUDE_unknown_number_value_l2613_261372

theorem unknown_number_value : 
  ∃ (unknown_number : ℝ), 
    (∀ x : ℝ, (3 + 2 * x)^5 = (unknown_number + 3 * x)^4) ∧
    ((3 + 2 * 1.5)^5 = (unknown_number + 3 * 1.5)^4) →
    unknown_number = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_unknown_number_value_l2613_261372


namespace NUMINAMATH_CALUDE_opposite_of_negative_half_l2613_261391

theorem opposite_of_negative_half : -(-(1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_half_l2613_261391


namespace NUMINAMATH_CALUDE_congruence_solution_solution_properties_sum_of_solution_l2613_261351

theorem congruence_solution : ∃ (y : ℤ), (10 * y + 3) % 18 = 7 % 18 ∧ y % 9 = 4 % 9 := by
  sorry

theorem solution_properties : 4 < 9 ∧ 9 ≥ 2 := by
  sorry

theorem sum_of_solution : ∃ (a m : ℤ), (10 * a + 3) % 18 = 7 % 18 ∧ a % m = a ∧ a < m ∧ m ≥ 2 ∧ a + m = 13 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_solution_properties_sum_of_solution_l2613_261351


namespace NUMINAMATH_CALUDE_abc_mod_9_l2613_261338

theorem abc_mod_9 (a b c : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9)
  (h1 : (a + 3*b + 2*c) % 9 = 0)
  (h2 : (2*a + 2*b + 3*c) % 9 = 3)
  (h3 : (3*a + b + 2*c) % 9 = 6) :
  (a * b * c) % 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_mod_9_l2613_261338


namespace NUMINAMATH_CALUDE_ryan_spits_18_inches_l2613_261378

/-- The distance Billy can spit a watermelon seed in inches -/
def billy_distance : ℝ := 30

/-- The percentage farther Madison can spit compared to Billy -/
def madison_percentage : ℝ := 0.2

/-- The percentage shorter Ryan can spit compared to Madison -/
def ryan_percentage : ℝ := 0.5

/-- The distance Madison can spit a watermelon seed in inches -/
def madison_distance : ℝ := billy_distance * (1 + madison_percentage)

/-- The distance Ryan can spit a watermelon seed in inches -/
def ryan_distance : ℝ := madison_distance * (1 - ryan_percentage)

/-- Theorem stating that Ryan can spit a watermelon seed 18 inches -/
theorem ryan_spits_18_inches : ryan_distance = 18 := by sorry

end NUMINAMATH_CALUDE_ryan_spits_18_inches_l2613_261378


namespace NUMINAMATH_CALUDE_cycling_speed_problem_l2613_261311

/-- Proves that given the conditions of the problem, B's cycling speed is 20 kmph -/
theorem cycling_speed_problem (a_speed b_speed : ℝ) (delay meeting_distance : ℝ) : 
  a_speed = 10 →
  delay = 7 →
  meeting_distance = 140 →
  b_speed * delay = meeting_distance →
  b_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_cycling_speed_problem_l2613_261311


namespace NUMINAMATH_CALUDE_largest_valid_sample_size_l2613_261343

def population : ℕ := 36

def is_valid_sample_size (X : ℕ) : Prop :=
  (population % X = 0) ∧ (population % (X + 1) ≠ 0)

theorem largest_valid_sample_size :
  ∃ (X : ℕ), is_valid_sample_size X ∧ ∀ (Y : ℕ), Y > X → ¬is_valid_sample_size Y :=
by
  sorry

end NUMINAMATH_CALUDE_largest_valid_sample_size_l2613_261343


namespace NUMINAMATH_CALUDE_symmetric_points_nm_value_l2613_261363

/-- Given two points P and Q symmetric with respect to the y-axis, prove that n^m = 1/2 -/
theorem symmetric_points_nm_value (m n : ℝ) : 
  (m - 1 = -2) → (4 = n + 2) → n^m = 1/2 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_nm_value_l2613_261363


namespace NUMINAMATH_CALUDE_total_money_collected_l2613_261301

/-- Calculates the total money collected from ticket sales given the prices and attendance. -/
theorem total_money_collected 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (total_attendance : ℕ) 
  (children_attendance : ℕ) 
  (h1 : adult_price = 60 / 100) 
  (h2 : child_price = 25 / 100) 
  (h3 : total_attendance = 280) 
  (h4 : children_attendance = 80) :
  (total_attendance - children_attendance) * adult_price + children_attendance * child_price = 140 / 100 := by
sorry

end NUMINAMATH_CALUDE_total_money_collected_l2613_261301


namespace NUMINAMATH_CALUDE_x_equals_one_sufficient_but_not_necessary_for_x_squared_equals_one_l2613_261336

-- Define the conditions
def condition1 (x : ℝ) : Prop := x = 1 → x^2 = 1
def condition2 (x : ℝ) : Prop := x^2 = 1 → (x = 1 ∨ x = -1)

-- Define sufficient but not necessary
def sufficient_but_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

-- Theorem statement
theorem x_equals_one_sufficient_but_not_necessary_for_x_squared_equals_one :
  (∀ x : ℝ, condition1 x) →
  (∀ x : ℝ, condition2 x) →
  sufficient_but_not_necessary (∃ x : ℝ, x = 1) (∃ x : ℝ, x^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_x_equals_one_sufficient_but_not_necessary_for_x_squared_equals_one_l2613_261336


namespace NUMINAMATH_CALUDE_prime_ones_and_seven_l2613_261305

/-- Represents a number with n-1 digits 1 and one digit 7 -/
def A (n : ℕ) (k : ℕ) : ℕ := (10^n + 54 * 10^k - 1) / 9

/-- Checks if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- All numbers with n-1 digits 1 and one digit 7 are prime -/
def all_prime (n : ℕ) : Prop := ∀ k : ℕ, k < n → is_prime (A n k)

theorem prime_ones_and_seven :
  ∀ n : ℕ, (all_prime n ↔ n = 1 ∨ n = 2) :=
sorry

end NUMINAMATH_CALUDE_prime_ones_and_seven_l2613_261305


namespace NUMINAMATH_CALUDE_expected_value_of_segments_expected_value_is_1037_l2613_261310

/-- The number of points in the plane -/
def n : ℕ := 100

/-- The number of pairs connected by line segments -/
def connected_pairs : ℕ := 4026

/-- A function to calculate binomial coefficient -/
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/-- The theorem to prove -/
theorem expected_value_of_segments (no_three_collinear : True) 
  (all_points_unique : True) : ℝ :=
  let total_pairs := choose n 2
  let diff_50_pairs := choose 51 2
  let prob_segment := connected_pairs / total_pairs
  prob_segment * diff_50_pairs

/-- The main theorem stating the expected value is 1037 -/
theorem expected_value_is_1037 (no_three_collinear : True) 
  (all_points_unique : True) :
  expected_value_of_segments no_three_collinear all_points_unique = 1037 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_segments_expected_value_is_1037_l2613_261310


namespace NUMINAMATH_CALUDE_volunteer_members_count_l2613_261354

/-- The number of sheets of cookies baked by each member -/
def sheets_per_member : ℕ := 10

/-- The number of cookies on each sheet -/
def cookies_per_sheet : ℕ := 16

/-- The total number of cookies baked -/
def total_cookies : ℕ := 16000

/-- The number of members who volunteered to bake cookies -/
def num_members : ℕ := total_cookies / (sheets_per_member * cookies_per_sheet)

theorem volunteer_members_count : num_members = 100 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_members_count_l2613_261354


namespace NUMINAMATH_CALUDE_mayor_approval_probability_l2613_261325

/-- The probability of a voter approving the mayor's work -/
def p_approve : ℝ := 0.6

/-- The number of voters selected -/
def n_voters : ℕ := 4

/-- The number of approvals we're interested in -/
def k_approvals : ℕ := 2

/-- The probability of exactly k_approvals in n_voters independent trials -/
def prob_k_approvals (p : ℝ) (n k : ℕ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem mayor_approval_probability :
  prob_k_approvals p_approve n_voters k_approvals = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_mayor_approval_probability_l2613_261325


namespace NUMINAMATH_CALUDE_inequality_proof_l2613_261393

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (b + c) * (c + a) * (a + b) ≥ 4 * ((a + b + c) * ((a + b + c) / 3) ^ (1/8) - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2613_261393


namespace NUMINAMATH_CALUDE_line_l_theorem_circle_M_theorem_l2613_261371

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 5 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, -1)

-- Define point Q
def point_Q : ℝ × ℝ := (0, 1)

-- Define the line l
def line_l (x y : ℝ) : Prop := x = -2 ∨ y = (15/8)*x + 11/4

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y - 7 = 0

-- Theorem for line l
theorem line_l_theorem : 
  ∃ (A B : ℝ × ℝ), 
    (∀ (x y : ℝ), line_l x y ↔ (∃ t : ℝ, (x, y) = (1-t) • point_P + t • A ∨ (x, y) = (1-t) • point_P + t • B)) ∧
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16 :=
sorry

-- Theorem for circle M
theorem circle_M_theorem :
  (∀ x y : ℝ, circle_M x y → (x = point_P.1 ∧ y = point_P.2) ∨ (x = point_Q.1 ∧ y = point_Q.2)) ∧
  (∃ t : ℝ, ∀ x y : ℝ, circle_M x y → circle_C x y ∨ (x, y) = (1-t) • point_Q + t • point_P) :=
sorry

end NUMINAMATH_CALUDE_line_l_theorem_circle_M_theorem_l2613_261371


namespace NUMINAMATH_CALUDE_motorcycle_vs_car_profit_difference_l2613_261389

/-- Represents the production and sales data for a vehicle type -/
structure VehicleData where
  materialCost : ℕ
  quantity : ℕ
  price : ℕ

/-- Calculates the profit for a given vehicle type -/
def profit (data : VehicleData) : ℤ :=
  (data.quantity * data.price) - data.materialCost

/-- Theorem: The difference in profit between motorcycle and car production is $50 -/
theorem motorcycle_vs_car_profit_difference :
  let carData : VehicleData := ⟨100, 4, 50⟩
  let motorcycleData : VehicleData := ⟨250, 8, 50⟩
  profit motorcycleData - profit carData = 50 := by
  sorry

#eval profit ⟨250, 8, 50⟩ - profit ⟨100, 4, 50⟩

end NUMINAMATH_CALUDE_motorcycle_vs_car_profit_difference_l2613_261389


namespace NUMINAMATH_CALUDE_remainder_when_divided_by_nine_l2613_261356

/-- The smallest positive integer satisfying the given conditions -/
def smallest_n : ℕ := sorry

/-- The first condition: n mod 8 = 6 -/
axiom cond1 : smallest_n % 8 = 6

/-- The second condition: n mod 7 = 5 -/
axiom cond2 : smallest_n % 7 = 5

/-- The smallest_n is indeed the smallest positive integer satisfying both conditions -/
axiom smallest : ∀ m : ℕ, m > 0 → m % 8 = 6 → m % 7 = 5 → m ≥ smallest_n

/-- Theorem: The smallest n satisfying both conditions leaves a remainder of 1 when divided by 9 -/
theorem remainder_when_divided_by_nine : smallest_n % 9 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_when_divided_by_nine_l2613_261356


namespace NUMINAMATH_CALUDE_percentage_of_360_is_120_l2613_261381

theorem percentage_of_360_is_120 : 
  (120 : ℝ) / 360 * 100 = 100 / 3 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_360_is_120_l2613_261381


namespace NUMINAMATH_CALUDE_units_digit_3_pow_20_l2613_261341

def units_digit_pattern : ℕ → ℕ
| 0 => 3
| 1 => 9
| 2 => 7
| 3 => 1
| n + 4 => units_digit_pattern n

theorem units_digit_3_pow_20 :
  units_digit_pattern 19 = 1 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_3_pow_20_l2613_261341


namespace NUMINAMATH_CALUDE_nancy_bead_purchase_cost_l2613_261370

/-- The total cost of Nancy's purchase given the prices of crystal and metal beads and the quantities she buys. -/
theorem nancy_bead_purchase_cost (crystal_price metal_price : ℕ) (crystal_qty metal_qty : ℕ) : 
  crystal_price = 9 → metal_price = 10 → crystal_qty = 1 → metal_qty = 2 →
  crystal_price * crystal_qty + metal_price * metal_qty = 29 := by
sorry

end NUMINAMATH_CALUDE_nancy_bead_purchase_cost_l2613_261370


namespace NUMINAMATH_CALUDE_tangent_parallel_condition_extreme_values_and_intersections_l2613_261322

/-- The cubic function f(x) = x^3 - ax^2 + bx + c -/
def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 + b*x + c

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem tangent_parallel_condition (a b c : ℝ) :
  (∃ x₀ : ℝ, f' a b x₀ = 0) → a^2 ≥ 3*b :=
sorry

theorem extreme_values_and_intersections (c : ℝ) :
  (f' 3 (-9) (-1) = 0 ∧ f' 3 (-9) 3 = 0) →
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f 3 (-9) c x₁ = 0 ∧ f 3 (-9) c x₂ = 0 ∧ f 3 (-9) c x₃ = 0 ∧
    (∀ x : ℝ, f 3 (-9) c x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  -5 < c ∧ c < 27 :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_condition_extreme_values_and_intersections_l2613_261322


namespace NUMINAMATH_CALUDE_rectangle_area_12_l2613_261369

def rectangle_area (w l : ℕ+) : ℕ := w.val * l.val

def valid_rectangles : Set (ℕ+ × ℕ+) :=
  {p | rectangle_area p.1 p.2 = 12}

theorem rectangle_area_12 :
  valid_rectangles = {(1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1)} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_12_l2613_261369


namespace NUMINAMATH_CALUDE_paint_house_time_l2613_261352

/-- Given that five people can paint a house in seven hours and everyone works at the same rate,
    proves that two people would take 17.5 hours to paint the same house. -/
theorem paint_house_time (people_rate : ℝ → ℝ → ℝ) :
  (people_rate 5 7 = 1) →  -- Five people can paint the house in seven hours
  (∀ n t, people_rate n t = people_rate 1 1 * n * t) →  -- Everyone works at the same rate
  (people_rate 2 17.5 = 1) :=  -- Two people take 17.5 hours
by sorry

end NUMINAMATH_CALUDE_paint_house_time_l2613_261352


namespace NUMINAMATH_CALUDE_circle_M_properties_l2613_261347

-- Define the circle M
def circle_M : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ (x + 3)^2 + (y - 4)^2 = 25}

-- Define the points
def point_A : ℝ × ℝ := (-3, -1)
def point_B : ℝ × ℝ := (-6, 8)
def point_C : ℝ × ℝ := (1, 1)
def point_P : ℝ × ℝ := (2, 3)

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := x = 2
def tangent_line_2 (x y : ℝ) : Prop := 12 * x - 5 * y - 9 = 0

theorem circle_M_properties :
  (point_A ∈ circle_M) ∧
  (point_B ∈ circle_M) ∧
  (point_C ∈ circle_M) ∧
  (∀ (x y : ℝ), tangent_line_1 x y → (x, y) ∈ circle_M → (x, y) = point_P) ∧
  (∀ (x y : ℝ), tangent_line_2 x y → (x, y) ∈ circle_M → (x, y) = point_P) :=
sorry

end NUMINAMATH_CALUDE_circle_M_properties_l2613_261347


namespace NUMINAMATH_CALUDE_min_students_all_correct_l2613_261395

theorem min_students_all_correct (total_students : ℕ) 
  (q1_correct q2_correct q3_correct q4_correct : ℕ) 
  (h1 : total_students = 45)
  (h2 : q1_correct = 35)
  (h3 : q2_correct = 27)
  (h4 : q3_correct = 41)
  (h5 : q4_correct = 38) :
  total_students - (total_students - q1_correct) - 
  (total_students - q2_correct) - (total_students - q3_correct) - 
  (total_students - q4_correct) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_students_all_correct_l2613_261395


namespace NUMINAMATH_CALUDE_books_in_wrong_place_l2613_261313

theorem books_in_wrong_place
  (initial_books : ℕ)
  (books_left : ℕ)
  (history_books : ℕ)
  (fiction_books : ℕ)
  (children_books : ℕ)
  (h1 : initial_books = 51)
  (h2 : books_left = 16)
  (h3 : history_books = 12)
  (h4 : fiction_books = 19)
  (h5 : children_books = 8) :
  history_books + fiction_books + children_books - (initial_books - books_left) = 4 :=
by sorry

end NUMINAMATH_CALUDE_books_in_wrong_place_l2613_261313


namespace NUMINAMATH_CALUDE_jamie_alex_payment_difference_l2613_261382

-- Define the problem parameters
def total_slices : ℕ := 10
def plain_pizza_cost : ℚ := 10
def spicy_topping_cost : ℚ := 3
def spicy_fraction : ℚ := 1/3

-- Define the number of slices each person ate
def jamie_spicy_slices : ℕ := (spicy_fraction * total_slices).num.toNat
def jamie_plain_slices : ℕ := 2
def alex_plain_slices : ℕ := total_slices - jamie_spicy_slices - jamie_plain_slices

-- Define the theorem
theorem jamie_alex_payment_difference :
  let total_cost : ℚ := plain_pizza_cost + spicy_topping_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let jamie_payment : ℚ := cost_per_slice * (jamie_spicy_slices + jamie_plain_slices)
  let alex_payment : ℚ := cost_per_slice * alex_plain_slices
  jamie_payment - alex_payment = 0 :=
sorry

end NUMINAMATH_CALUDE_jamie_alex_payment_difference_l2613_261382


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2613_261376

theorem solution_set_of_inequality (x : ℝ) :
  (x - 1) * (2 - x) > 0 ↔ x ∈ Set.Ioo 1 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2613_261376


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2613_261315

theorem diophantine_equation_solutions :
  (∀ k : ℤ, (101 * (4 + 13 * k) - 13 * (31 + 101 * k) = 1)) ∧
  (∀ k : ℤ, (79 * (-6 + 19 * k) - 19 * (-25 + 79 * k) = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2613_261315


namespace NUMINAMATH_CALUDE_license_plate_theorem_l2613_261387

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels (including Y) -/
def vowel_count : ℕ := 6

/-- The number of consonants -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of possible three-character license plates with two consonants followed by a vowel -/
def license_plate_count : ℕ := consonant_count * consonant_count * vowel_count

theorem license_plate_theorem : license_plate_count = 2400 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l2613_261387


namespace NUMINAMATH_CALUDE_andrew_winning_strategy_l2613_261302

/-- Represents the state of the game with two heaps of pebbles -/
structure GameState where
  a : ℕ
  b : ℕ

/-- Predicate to check if a number is of the form 2^x + 1 -/
def isPowerOfTwoPlusOne (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 2^x + 1

/-- Predicate to check if Andrew has a winning strategy -/
def andrewWins (state : GameState) : Prop :=
  state.a = 1 ∨ state.b = 1 ∨
  isPowerOfTwoPlusOne (state.a + state.b) ∨
  (isPowerOfTwoPlusOne state.a ∧ state.b < state.a) ∨
  (isPowerOfTwoPlusOne state.b ∧ state.a < state.b)

/-- The main theorem stating the winning condition for Andrew -/
theorem andrew_winning_strategy (state : GameState) :
  andrewWins state ↔ ∃ (strategy : GameState → ℕ → GameState),
    ∀ (move : ℕ), andrewWins (strategy state move) :=
  sorry

end NUMINAMATH_CALUDE_andrew_winning_strategy_l2613_261302


namespace NUMINAMATH_CALUDE_kindergarten_total_l2613_261374

/-- Represents the number of children in a kindergarten with different pet ownership patterns -/
structure KindergartenPets where
  dogs_only : ℕ
  both : ℕ
  cats_total : ℕ

/-- Calculates the total number of children in the kindergarten -/
def total_children (k : KindergartenPets) : ℕ :=
  k.dogs_only + k.both + (k.cats_total - k.both)

/-- Theorem stating the total number of children in the kindergarten -/
theorem kindergarten_total (k : KindergartenPets) 
  (h1 : k.dogs_only = 18)
  (h2 : k.both = 6)
  (h3 : k.cats_total = 12) :
  total_children k = 30 := by
  sorry

#check kindergarten_total

end NUMINAMATH_CALUDE_kindergarten_total_l2613_261374


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2613_261397

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | |x - 1| ≤ 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2613_261397


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l2613_261384

theorem consecutive_integers_sum_of_squares (n : ℕ) : 
  (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2 = 770 → n + 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l2613_261384


namespace NUMINAMATH_CALUDE_find_k_l2613_261318

theorem find_k (k : ℝ) (h : 64 / k = 8) : k = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2613_261318


namespace NUMINAMATH_CALUDE_smallest_third_side_l2613_261375

theorem smallest_third_side (a b : ℝ) (ha : a = 7.5) (hb : b = 11.5) :
  ∃ (s : ℕ), s = 5 ∧ 
  (a + s > b) ∧ (a + b > s) ∧ (b + s > a) ∧
  (∀ (t : ℕ), t < s → ¬(a + t > b ∧ a + b > t ∧ b + t > a)) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_third_side_l2613_261375


namespace NUMINAMATH_CALUDE_subtraction_amount_l2613_261344

theorem subtraction_amount (N : ℕ) (A : ℕ) : N = 32 → (N - A) / 13 = 2 → A = 6 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_amount_l2613_261344


namespace NUMINAMATH_CALUDE_poem_lines_proof_l2613_261357

/-- The number of lines added to the poem each month -/
def lines_per_month : ℕ := 3

/-- The number of months after which the poem will have 90 lines -/
def months : ℕ := 22

/-- The total number of lines in the poem after 22 months -/
def total_lines : ℕ := 90

/-- The current number of lines in the poem -/
def current_lines : ℕ := total_lines - (lines_per_month * months)

theorem poem_lines_proof : current_lines = 24 := by
  sorry

end NUMINAMATH_CALUDE_poem_lines_proof_l2613_261357


namespace NUMINAMATH_CALUDE_line_equation_with_opposite_intercepts_l2613_261340

/-- A line passing through a given point with opposite intercepts on the coordinate axes -/
structure LineWithOppositeIntercepts where
  -- The x-coordinate of the point
  x : ℝ
  -- The y-coordinate of the point
  y : ℝ
  -- The equation of the line in the form ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through the point (x, y)
  point_on_line : a * x + b * y + c = 0
  -- The intercepts are opposite in value
  opposite_intercepts : a * c = -b * c ∨ a = 0 ∧ b = 0 ∧ c = 0

/-- The equation of a line with opposite intercepts passing through (3, -2) -/
theorem line_equation_with_opposite_intercepts :
  ∀ (l : LineWithOppositeIntercepts),
  l.x = 3 ∧ l.y = -2 →
  (l.a = 2 ∧ l.b = 3 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = -1 ∧ l.c = -5) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_with_opposite_intercepts_l2613_261340


namespace NUMINAMATH_CALUDE_bankers_discount_calculation_l2613_261328

/-- Calculates the banker's discount for a given period and rate -/
def bankers_discount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem: The banker's discount for the given conditions is 18900 -/
theorem bankers_discount_calculation (principal : ℝ) 
  (rate1 rate2 rate3 : ℝ) (time1 time2 time3 : ℝ) :
  principal = 180000 ∧ 
  rate1 = 0.12 ∧ rate2 = 0.14 ∧ rate3 = 0.16 ∧
  time1 = 0.25 ∧ time2 = 0.25 ∧ time3 = 0.25 →
  bankers_discount principal rate1 time1 + 
  bankers_discount principal rate2 time2 + 
  bankers_discount principal rate3 time3 = 18900 := by
  sorry

#eval bankers_discount 180000 0.12 0.25 + 
      bankers_discount 180000 0.14 0.25 + 
      bankers_discount 180000 0.16 0.25

end NUMINAMATH_CALUDE_bankers_discount_calculation_l2613_261328


namespace NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l2613_261358

theorem no_integer_satisfies_conditions : ¬ ∃ m : ℤ, m % 9 = 2 ∧ m % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l2613_261358


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2613_261334

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧ 
  ((1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b)) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2613_261334


namespace NUMINAMATH_CALUDE_conference_handshakes_l2613_261360

theorem conference_handshakes (n : ℕ) (h : n = 7) : 
  (2 * n) * ((2 * n - 1) - 2) / 2 = 77 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l2613_261360


namespace NUMINAMATH_CALUDE_smallest_stairs_l2613_261321

theorem smallest_stairs (n : ℕ) : 
  (n > 15) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (∀ m : ℕ, m > 15 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_stairs_l2613_261321


namespace NUMINAMATH_CALUDE_quadrupled_exponent_base_l2613_261394

theorem quadrupled_exponent_base (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) 
  (h : (4 * c)^(4 * d) = c^d * y^d) : y = 256 * c^3 := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_exponent_base_l2613_261394


namespace NUMINAMATH_CALUDE_prism_coloring_iff_divisible_by_three_l2613_261331

/-- Represents a prism with an n-gon base -/
structure Prism :=
  (n : ℕ)

/-- Represents a coloring of the prism vertices -/
def Coloring (p : Prism) := Fin (2 * p.n) → Fin 3

/-- Predicate to check if a coloring is valid -/
def is_valid_coloring (p : Prism) (c : Coloring p) : Prop :=
  ∀ v : Fin (2 * p.n), 
    ∃ (c1 c2 : Fin 3), c1 ≠ c v ∧ c2 ≠ c v ∧ c1 ≠ c2 ∧
    ∃ (v1 v2 : Fin (2 * p.n)), v1 ≠ v ∧ v2 ≠ v ∧ v1 ≠ v2 ∧
    c v1 = c1 ∧ c v2 = c2

theorem prism_coloring_iff_divisible_by_three (p : Prism) :
  (∃ c : Coloring p, is_valid_coloring p c) ↔ p.n % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_prism_coloring_iff_divisible_by_three_l2613_261331


namespace NUMINAMATH_CALUDE_second_platform_length_l2613_261361

/-- Given a train and two platforms, calculate the length of the second platform. -/
theorem second_platform_length
  (train_length : ℝ)
  (first_platform_length : ℝ)
  (first_crossing_time : ℝ)
  (second_crossing_time : ℝ)
  (h1 : train_length = 30)
  (h2 : first_platform_length = 90)
  (h3 : first_crossing_time = 12)
  (h4 : second_crossing_time = 15)
  (h5 : train_length > 0)
  (h6 : first_platform_length > 0)
  (h7 : first_crossing_time > 0)
  (h8 : second_crossing_time > 0) :
  let speed := (train_length + first_platform_length) / first_crossing_time
  let second_platform_length := speed * second_crossing_time - train_length
  second_platform_length = 120 := by
sorry


end NUMINAMATH_CALUDE_second_platform_length_l2613_261361


namespace NUMINAMATH_CALUDE_exactly_three_tangent_lines_l2613_261308

/-- A line passing through (0, 1) that intersects the parabola y^2 = 4x at only one point -/
structure TangentLine where
  slope : ℝ
  intersects_once : ∃! (x y : ℝ), y^2 = 4*x ∧ y = slope * x + 1

/-- The number of lines passing through (0, 1) that intersect y^2 = 4x at only one point -/
def num_tangent_lines : ℕ := sorry

/-- Theorem stating that there are exactly 3 such lines -/
theorem exactly_three_tangent_lines : num_tangent_lines = 3 := by sorry

end NUMINAMATH_CALUDE_exactly_three_tangent_lines_l2613_261308


namespace NUMINAMATH_CALUDE_scientific_notation_78922_l2613_261366

theorem scientific_notation_78922 : 
  78922 = 7.8922 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_78922_l2613_261366


namespace NUMINAMATH_CALUDE_intersection_point_of_perpendicular_lines_l2613_261307

/-- Given a line l: 2x + y = 10 and a point (-10, 0), this theorem proves that the 
    intersection point of l and the line l' passing through (-10, 0) and perpendicular 
    to l is (2, 6). -/
theorem intersection_point_of_perpendicular_lines 
  (l : Set (ℝ × ℝ)) 
  (h_l : l = {(x, y) | 2 * x + y = 10}) 
  (p : ℝ × ℝ) 
  (h_p : p = (-10, 0)) :
  ∃ (q : ℝ × ℝ), q ∈ l ∧ 
    (∃ (l' : Set (ℝ × ℝ)), p ∈ l' ∧ 
      (∀ (x y : ℝ), (x, y) ∈ l' ↔ (x - p.1) * 2 + (y - p.2) = 0) ∧
      q ∈ l' ∧
      q = (2, 6)) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_of_perpendicular_lines_l2613_261307


namespace NUMINAMATH_CALUDE_parallel_implies_a_eq_2_perpendicular_implies_a_eq_neg3_or_0_l2613_261324

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - y + 2 = 0
def l₂ (a x y : ℝ) : Prop := (a + 2) * x - a * y - 2 = 0

-- Define parallel and perpendicular relations
def parallel (a : ℝ) : Prop := ∀ x y, l₁ a x y ↔ l₂ a x y
def perpendicular (a : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, l₁ a x₁ y₁ → l₂ a x₂ y₂ → 
  (x₂ - x₁) * (y₂ - y₁) = 0

-- Theorem statements
theorem parallel_implies_a_eq_2 : ∀ a : ℝ, parallel a → a = 2 := by sorry

theorem perpendicular_implies_a_eq_neg3_or_0 : ∀ a : ℝ, perpendicular a → a = -3 ∨ a = 0 := by sorry

end NUMINAMATH_CALUDE_parallel_implies_a_eq_2_perpendicular_implies_a_eq_neg3_or_0_l2613_261324


namespace NUMINAMATH_CALUDE_suki_coffee_bags_suki_coffee_bags_proof_l2613_261390

theorem suki_coffee_bags (suki_bag_weight jimmy_bag_weight container_weight : ℕ)
                         (jimmy_bags : ℚ)
                         (num_containers : ℕ)
                         (suki_bags : ℕ) : Prop :=
  suki_bag_weight = 22 →
  jimmy_bag_weight = 18 →
  jimmy_bags = 4.5 →
  container_weight = 8 →
  num_containers = 28 →
  (↑suki_bags * suki_bag_weight + jimmy_bags * jimmy_bag_weight : ℚ) = ↑(num_containers * container_weight) →
  suki_bags = 6

theorem suki_coffee_bags_proof : suki_coffee_bags 22 18 8 (4.5 : ℚ) 28 6 := by
  sorry

end NUMINAMATH_CALUDE_suki_coffee_bags_suki_coffee_bags_proof_l2613_261390


namespace NUMINAMATH_CALUDE_tangent_line_passes_through_fixed_point_l2613_261348

/-- The parabola Γ -/
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- The point P from which tangents are drawn -/
def P (m : ℝ) : ℝ × ℝ := (m, -4)

/-- The fixed point through which AB always passes -/
def fixedPoint : ℝ × ℝ := (0, 4)

/-- Theorem stating that AB always passes through the fixed point -/
theorem tangent_line_passes_through_fixed_point (m : ℝ) :
  ∀ A B : ℝ × ℝ,
  A ∈ Γ → B ∈ Γ →
  (∃ t : ℝ, A = (1 - t) • P m + t • B) →
  (∃ s : ℝ, B = (1 - s) • P m + s • A) →
  ∃ r : ℝ, fixedPoint = (1 - r) • A + r • B :=
sorry

end NUMINAMATH_CALUDE_tangent_line_passes_through_fixed_point_l2613_261348


namespace NUMINAMATH_CALUDE_tan_1000_degrees_l2613_261377

theorem tan_1000_degrees (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * Real.pi / 180) = Real.tan (1000 * Real.pi / 180) → n = -80 :=
by sorry

end NUMINAMATH_CALUDE_tan_1000_degrees_l2613_261377


namespace NUMINAMATH_CALUDE_paradise_park_large_seats_l2613_261300

/-- Represents a Ferris wheel with small and large seats. -/
structure FerrisWheel where
  smallSeats : Nat
  largeSeats : Nat
  smallSeatCapacity : Nat
  largeSeatCapacity : Nat
  totalLargeSeatCapacity : Nat

/-- The Ferris wheel in paradise park -/
def paradiseParkFerrisWheel : FerrisWheel := {
  smallSeats := 3
  largeSeats := 0  -- We don't know this value yet
  smallSeatCapacity := 16
  largeSeatCapacity := 12
  totalLargeSeatCapacity := 84
}

/-- Theorem: The number of large seats on the paradise park Ferris wheel is 7 -/
theorem paradise_park_large_seats : 
  paradiseParkFerrisWheel.largeSeats = 
    paradiseParkFerrisWheel.totalLargeSeatCapacity / paradiseParkFerrisWheel.largeSeatCapacity := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_large_seats_l2613_261300


namespace NUMINAMATH_CALUDE_vector_to_point_coordinates_l2613_261306

/-- Given a vector AB = (-2, 4), if point A is at the origin (0, 0), 
    then the coordinates of point B are (-2, 4). -/
theorem vector_to_point_coordinates (A B : ℝ × ℝ) : 
  (A.1 - B.1 = 2 ∧ A.2 - B.2 = -4) → 
  (A = (0, 0) → B = (-2, 4)) := by
  sorry

end NUMINAMATH_CALUDE_vector_to_point_coordinates_l2613_261306


namespace NUMINAMATH_CALUDE_stratified_sampling_total_components_l2613_261342

theorem stratified_sampling_total_components :
  let total_sample_size : ℕ := 45
  let sample_size_A : ℕ := 20
  let sample_size_C : ℕ := 10
  let num_B : ℕ := 300
  let num_C : ℕ := 200
  let num_A : ℕ := (total_sample_size * (num_B + num_C)) / (total_sample_size - sample_size_A - sample_size_C)
  num_A + num_B + num_C = 900 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_total_components_l2613_261342


namespace NUMINAMATH_CALUDE_square_sum_from_means_l2613_261335

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = 10) : 
  x^2 + y^2 = 1400 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l2613_261335


namespace NUMINAMATH_CALUDE_train_length_l2613_261359

/-- Given a train that crosses a platform in a certain time and a signal pole in another time,
    this theorem proves the length of the train. -/
theorem train_length
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : platform_crossing_time = 39)
  (h2 : pole_crossing_time = 24)
  (h3 : platform_length = 187.5) :
  (platform_crossing_time * platform_length) / (platform_crossing_time - pole_crossing_time) = 300 :=
by sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2613_261359


namespace NUMINAMATH_CALUDE_ice_cream_theorem_l2613_261326

theorem ice_cream_theorem (n : ℕ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_l2613_261326


namespace NUMINAMATH_CALUDE_second_class_size_l2613_261379

theorem second_class_size (first_class_size : ℕ) (first_class_avg : ℝ) 
  (second_class_avg : ℝ) (total_avg : ℝ) :
  first_class_size = 30 →
  first_class_avg = 40 →
  second_class_avg = 70 →
  total_avg = 58.75 →
  ∃ (second_class_size : ℕ), 
    (first_class_size * first_class_avg + second_class_size * second_class_avg) / 
    (first_class_size + second_class_size) = total_avg ∧
    second_class_size = 50 :=
by
  sorry

#check second_class_size

end NUMINAMATH_CALUDE_second_class_size_l2613_261379


namespace NUMINAMATH_CALUDE_polynomial_has_real_root_l2613_261329

/-- The polynomial in question -/
def polynomial (b x : ℝ) : ℝ := x^4 + b*x^3 - 2*x^2 + b*x + 2

/-- Theorem stating the condition for the polynomial to have at least one real root -/
theorem polynomial_has_real_root (b : ℝ) :
  (∃ x : ℝ, polynomial b x = 0) ↔ b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_has_real_root_l2613_261329


namespace NUMINAMATH_CALUDE_containers_used_l2613_261396

def initial_balls : ℕ := 100
def balls_per_container : ℕ := 10

theorem containers_used :
  let remaining_balls := initial_balls / 2
  remaining_balls / balls_per_container = 5 := by
  sorry

end NUMINAMATH_CALUDE_containers_used_l2613_261396


namespace NUMINAMATH_CALUDE_lollipop_theorem_l2613_261337

/-- Represents the sequence of lollipops eaten over six days -/
def LollipopSequence := Fin 6 → ℝ

/-- The condition that each day after the first, 5 more lollipops are eaten -/
def ArithmeticCondition (seq : LollipopSequence) : Prop :=
  ∀ i : Fin 5, seq (i.succ) = seq i + 5

/-- The total number of lollipops eaten over six days is 150 -/
def SumCondition (seq : LollipopSequence) : Prop :=
  (Finset.univ.sum seq) = 150

theorem lollipop_theorem (seq : LollipopSequence) 
  (h_arith : ArithmeticCondition seq) (h_sum : SumCondition seq) : 
  seq 3 = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_theorem_l2613_261337


namespace NUMINAMATH_CALUDE_sqrt_square_negative_l2613_261333

theorem sqrt_square_negative : Real.sqrt ((-2023)^2) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_negative_l2613_261333


namespace NUMINAMATH_CALUDE_provolone_needed_l2613_261309

def cheese_blend (m r p : ℝ) : Prop :=
  m / r = 2 ∧ p / r = 2

theorem provolone_needed (m r : ℝ) (hm : m = 20) (hr : r = 10) :
  ∃ p : ℝ, cheese_blend m r p ∧ p = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_provolone_needed_l2613_261309


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2613_261332

/-- Given a geometric sequence {a_n} where a_1 + a_2 = 324 and a_3 + a_4 = 36,
    prove that a_5 + a_6 = 4 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →
  a 1 + a 2 = 324 →
  a 3 + a 4 = 36 →
  a 5 + a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2613_261332


namespace NUMINAMATH_CALUDE_min_value_x_sqrt_9_minus_x_squared_l2613_261314

theorem min_value_x_sqrt_9_minus_x_squared :
  ∃ (x : ℝ), -3 < x ∧ x < 0 ∧
  x * Real.sqrt (9 - x^2) = -9/2 ∧
  ∀ (y : ℝ), -3 < y ∧ y < 0 →
  y * Real.sqrt (9 - y^2) ≥ -9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_sqrt_9_minus_x_squared_l2613_261314


namespace NUMINAMATH_CALUDE_ratio_second_part_l2613_261320

/-- Given a ratio where the first part is 10 and the ratio expressed as a percent is 50,
    prove that the second part of the ratio is also 10. -/
theorem ratio_second_part (first_part : ℕ) (ratio_percent : ℕ) (second_part : ℕ) : 
  first_part = 10 → ratio_percent = 50 → second_part = 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_second_part_l2613_261320


namespace NUMINAMATH_CALUDE_boat_upstream_time_l2613_261339

theorem boat_upstream_time (B C : ℝ) (h1 : B = 4 * C) (h2 : B > 0) (h3 : C > 0) : 
  (10 : ℝ) * (B + C) / (B - C) = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_time_l2613_261339


namespace NUMINAMATH_CALUDE_age_ratio_after_five_years_l2613_261392

/-- Theorem: Ratio of parent's age to son's age after 5 years -/
theorem age_ratio_after_five_years
  (parent_age : ℕ)
  (son_age : ℕ)
  (h1 : parent_age = 45)
  (h2 : son_age = 15) :
  (parent_age + 5) / (son_age + 5) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_after_five_years_l2613_261392


namespace NUMINAMATH_CALUDE_original_amount_l2613_261323

theorem original_amount (X : ℝ) : (0.1 * (0.5 * X) = 25) → X = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_amount_l2613_261323
