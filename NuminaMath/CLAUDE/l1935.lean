import Mathlib

namespace NUMINAMATH_CALUDE_non_obtuse_triangle_perimeter_gt_four_circumradius_l1935_193532

/-- A triangle with vertices A, B, and C in the real plane. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The perimeter of a triangle. -/
def perimeter (t : Triangle) : ℝ := sorry

/-- The radius of the circumcircle of a triangle. -/
def circumradius (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is non-obtuse. -/
def is_non_obtuse (t : Triangle) : Prop := sorry

/-- Theorem: For any non-obtuse triangle, its perimeter is greater than
    four times the radius of its circumcircle. -/
theorem non_obtuse_triangle_perimeter_gt_four_circumradius (t : Triangle) :
  is_non_obtuse t → perimeter t > 4 * circumradius t := by sorry

end NUMINAMATH_CALUDE_non_obtuse_triangle_perimeter_gt_four_circumradius_l1935_193532


namespace NUMINAMATH_CALUDE_teacher_age_survey_is_comprehensive_l1935_193562

-- Define the survey types
inductive SurveyType
  | TelevisionLifespan
  | CityIncome
  | StudentMyopia
  | TeacherAge

-- Define a function to determine if a survey is suitable for comprehensive method
def isSuitableForComprehensiveSurvey (survey : SurveyType) : Prop :=
  match survey with
  | .TelevisionLifespan => false  -- Involves destructiveness, must be sampled
  | .CityIncome => false          -- Large number of people, suitable for sampling
  | .StudentMyopia => false       -- Large number of people, suitable for sampling
  | .TeacherAge => true           -- Small number of people, easy to survey comprehensively

-- Theorem statement
theorem teacher_age_survey_is_comprehensive :
  isSuitableForComprehensiveSurvey SurveyType.TeacherAge = true := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_survey_is_comprehensive_l1935_193562


namespace NUMINAMATH_CALUDE_inequality_proof_l1935_193598

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 / b + b^2 / c + c^2 / a ≥ a + b + c + 4 * (a - b)^2 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1935_193598


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l1935_193577

theorem no_simultaneous_squares : ¬ ∃ (x y : ℕ), 
  ∃ (a b : ℕ), (x^2 + 2*y = a^2) ∧ (y^2 + 2*x = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l1935_193577


namespace NUMINAMATH_CALUDE_sentence_has_32_letters_l1935_193570

def original_sentence : String := "В ЭТОМ ПРЕДЛОЖЕНИИ ... БУКВ"
def filled_word : String := "ТРИДЦАТЬ ДВЕ"
def full_sentence : String := "В ЭТОМ ПРЕДЛОЖЕНИИ ТРИДЦАТЬ ДВЕ БУКВЫ"

def is_cyrillic_letter (c : Char) : Bool :=
  (c.toNat ≥ 1040 ∧ c.toNat ≤ 1103) ∨ (c = 'Ё' ∨ c = 'ё')

def count_cyrillic_letters (s : String) : Nat :=
  s.toList.filter is_cyrillic_letter |>.length

theorem sentence_has_32_letters : count_cyrillic_letters full_sentence = 32 := by
  sorry

end NUMINAMATH_CALUDE_sentence_has_32_letters_l1935_193570


namespace NUMINAMATH_CALUDE_sunflower_germination_rate_l1935_193550

theorem sunflower_germination_rate 
  (daisy_seeds : ℕ) 
  (sunflower_seeds : ℕ) 
  (daisy_germination_rate : ℚ) 
  (flower_production_rate : ℚ) 
  (total_flowering_plants : ℕ) :
  daisy_seeds = 25 →
  sunflower_seeds = 25 →
  daisy_germination_rate = 3/5 →
  flower_production_rate = 4/5 →
  total_flowering_plants = 28 →
  (daisy_seeds : ℚ) * daisy_germination_rate * flower_production_rate +
  (sunflower_seeds : ℚ) * (4/5) * flower_production_rate = total_flowering_plants →
  (4/5) = 20 / sunflower_seeds :=
by sorry

end NUMINAMATH_CALUDE_sunflower_germination_rate_l1935_193550


namespace NUMINAMATH_CALUDE_sequence_comparison_l1935_193578

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ+ → ℝ :=
  fun n => a₁ + (n.val - 1) * d

def geometric_sequence (b₁ : ℝ) (q : ℝ) : ℕ+ → ℝ :=
  fun n => b₁ * q^(n.val - 1)

theorem sequence_comparison (a : ℕ+ → ℝ) (b : ℕ+ → ℝ) :
  (a 1 = 2) →
  (b 1 = 2) →
  (a 2 = 4) →
  (b 2 = 4) →
  (∀ n : ℕ+, a n = 2 * n.val) →
  (∀ n : ℕ+, b n = 2^n.val) →
  (∀ n : ℕ+, n ≥ 3 → a n < b n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_comparison_l1935_193578


namespace NUMINAMATH_CALUDE_complex_equality_l1935_193563

theorem complex_equality (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b + Complex.I →
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_l1935_193563


namespace NUMINAMATH_CALUDE_kim_shirts_fraction_l1935_193581

theorem kim_shirts_fraction (initial_shirts : ℕ) (remaining_shirts : ℕ) :
  initial_shirts = 4 * 12 →
  remaining_shirts = 32 →
  (initial_shirts - remaining_shirts : ℚ) / initial_shirts = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_kim_shirts_fraction_l1935_193581


namespace NUMINAMATH_CALUDE_range_of_a_l1935_193518

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 4^x - (a+3)*2^x + 1 = 0) → a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1935_193518


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1935_193580

theorem fraction_to_decimal : (3 : ℚ) / 60 = (5 : ℚ) / 100 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1935_193580


namespace NUMINAMATH_CALUDE_inequality_of_distinct_positives_l1935_193586

theorem inequality_of_distinct_positives (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_distinct_positives_l1935_193586


namespace NUMINAMATH_CALUDE_nonagon_non_adjacent_segments_l1935_193561

theorem nonagon_non_adjacent_segments (n : ℕ) (h : n = 9) : 
  (n * (n - 1)) / 2 - n = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_non_adjacent_segments_l1935_193561


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1935_193575

theorem inequality_solution_set :
  {x : ℝ | (x + 1) * (x - 2) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1935_193575


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1935_193513

def expression (x : ℝ) : ℝ :=
  5 * (x^2 - 2*x^3 + x) + 2 * (x + 3*x^3 - 4*x^2 + 2*x^5 + 2*x^3) - 7 * (2 + x - 5*x^3 - 2*x^2)

theorem coefficient_of_x_cubed (x : ℝ) :
  ∃ (a b c d : ℝ), expression x = a*x^5 + b*x^4 + 35*x^3 + c*x^2 + d*x + (5*1 - 7*2) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1935_193513


namespace NUMINAMATH_CALUDE_A_solution_l1935_193593

noncomputable def A (x y : ℝ) : ℝ := 
  (Real.sqrt (4 * (x - Real.sqrt y) + y / x) * 
   Real.sqrt (9 * x^2 + 6 * (2 * y * x^3)^(1/3) + (4 * y^2)^(1/3))) / 
  (6 * x^2 + 2 * (2 * y * x^3)^(1/3) - 3 * Real.sqrt (y * x^2) - (4 * y^5)^(1/6)) / 2.343

theorem A_solution (x y : ℝ) (hx : x > 0) (hy : y ≥ 0) :
  A x y = if y > 4 * x^2 then -1 / Real.sqrt x else 1 / Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_A_solution_l1935_193593


namespace NUMINAMATH_CALUDE_sandra_amount_sandra_gets_100_l1935_193546

def share_money (sandra_ratio : ℕ) (amy_ratio : ℕ) (ruth_ratio : ℕ) (amy_amount : ℕ) : ℕ → ℕ → ℕ → Prop :=
  λ sandra_amount ruth_amount total_amount =>
    sandra_amount * amy_ratio = amy_amount * sandra_ratio ∧
    ruth_amount * amy_ratio = amy_amount * ruth_ratio ∧
    total_amount = sandra_amount + amy_amount + ruth_amount

theorem sandra_amount (amy_amount : ℕ) :
  share_money 2 1 3 amy_amount (2 * amy_amount) (3 * amy_amount) (6 * amy_amount) :=
by sorry

theorem sandra_gets_100 :
  share_money 2 1 3 50 100 150 300 :=
by sorry

end NUMINAMATH_CALUDE_sandra_amount_sandra_gets_100_l1935_193546


namespace NUMINAMATH_CALUDE_balloon_count_l1935_193568

theorem balloon_count (colors : Nat) (yellow_taken : Nat) : 
  colors = 4 → yellow_taken = 84 → colors * yellow_taken * 2 = 672 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l1935_193568


namespace NUMINAMATH_CALUDE_sum_remainder_mod_20_l1935_193522

theorem sum_remainder_mod_20 : (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_20_l1935_193522


namespace NUMINAMATH_CALUDE_angle_A_measure_l1935_193531

/-- Given a complex geometric figure with the following properties:
    - Angle B is 120°
    - Angle B forms a linear pair with another angle
    - A triangle adjacent to this setup contains an angle of 50°
    - A small triangle connected to one vertex of the larger triangle has an angle of 45°
    - This small triangle shares a vertex with angle A
    Prove that the measure of angle A is 65° -/
theorem angle_A_measure (B : Real) (adjacent_angle : Real) (large_triangle_angle : Real) (small_triangle_angle : Real) (A : Real) :
  B = 120 →
  B + adjacent_angle = 180 →
  large_triangle_angle = 50 →
  small_triangle_angle = 45 →
  A + small_triangle_angle + (180 - B - large_triangle_angle) = 180 →
  A = 65 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l1935_193531


namespace NUMINAMATH_CALUDE_rich_walk_distance_l1935_193557

-- Define the walking pattern
def house_to_sidewalk : ℕ := 20
def sidewalk_to_road_end : ℕ := 200
def left_turn_multiplier : ℕ := 2
def final_stretch_divisor : ℕ := 2

-- Define the total distance walked
def total_distance : ℕ :=
  let initial_distance := house_to_sidewalk + sidewalk_to_road_end
  let after_left_turn := initial_distance + left_turn_multiplier * initial_distance
  let to_end_point := after_left_turn + after_left_turn / final_stretch_divisor
  2 * to_end_point

-- Theorem statement
theorem rich_walk_distance : total_distance = 1980 := by sorry

end NUMINAMATH_CALUDE_rich_walk_distance_l1935_193557


namespace NUMINAMATH_CALUDE_max_piece_length_and_total_pieces_l1935_193583

-- Define the lengths of the two pipes
def pipe1_length : ℕ := 42
def pipe2_length : ℕ := 63

-- Define the theorem
theorem max_piece_length_and_total_pieces :
  ∃ (max_length : ℕ) (total_pieces : ℕ),
    max_length = Nat.gcd pipe1_length pipe2_length ∧
    max_length = 21 ∧
    total_pieces = pipe1_length / max_length + pipe2_length / max_length ∧
    total_pieces = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_piece_length_and_total_pieces_l1935_193583


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1935_193509

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 3 + 127 / 999) ∧ (x = 3124 / 999) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1935_193509


namespace NUMINAMATH_CALUDE_dropped_student_score_l1935_193503

theorem dropped_student_score 
  (initial_students : ℕ) 
  (initial_average : ℝ) 
  (remaining_students : ℕ) 
  (new_average : ℝ) : ℝ :=
  by
  have h1 : initial_students = 16 := by sorry
  have h2 : initial_average = 60.5 := by sorry
  have h3 : remaining_students = 15 := by sorry
  have h4 : new_average = 64 := by sorry

  -- The score of the dropped student
  let dropped_score := initial_students * initial_average - remaining_students * new_average

  -- Prove that the dropped score is 8
  have h5 : dropped_score = 8 := by sorry

  exact dropped_score

end NUMINAMATH_CALUDE_dropped_student_score_l1935_193503


namespace NUMINAMATH_CALUDE_dumpling_storage_temp_l1935_193529

def storage_temp_range (x : ℝ) : Prop := -20 ≤ x ∧ x ≤ -16

theorem dumpling_storage_temp :
  (storage_temp_range (-17)) ∧
  (storage_temp_range (-18)) ∧
  (storage_temp_range (-19)) ∧
  (¬ storage_temp_range (-22)) :=
by sorry

end NUMINAMATH_CALUDE_dumpling_storage_temp_l1935_193529


namespace NUMINAMATH_CALUDE_pool_filling_rate_l1935_193502

theorem pool_filling_rate (jim_rate sue_rate tony_rate : ℚ) 
  (h_jim : jim_rate = 1 / 30)
  (h_sue : sue_rate = 1 / 45)
  (h_tony : tony_rate = 1 / 90) :
  jim_rate + sue_rate + tony_rate = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_rate_l1935_193502


namespace NUMINAMATH_CALUDE_triangle_inequality_range_l1935_193505

/-- A right-angled triangle with sides a, b, and hypotenuse c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  pythagorean : a^2 + b^2 = c^2

/-- The theorem stating the range of t for which the given inequality holds -/
theorem triangle_inequality_range (tri : RightTriangle) :
  (∀ t : ℝ, 1 / tri.a^2 + 4 / tri.b^2 + t / tri.c^2 ≥ 0) ↔ 
  (∀ t : ℝ, t ≥ -9 ∧ t ∈ Set.Ici (-9)) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_range_l1935_193505


namespace NUMINAMATH_CALUDE_plant_species_numbering_not_unique_l1935_193584

theorem plant_species_numbering_not_unique : ∃ a b : ℕ, 
  2 ≤ a ∧ a < b ∧ b ≤ 20000 ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 20000 → Nat.gcd a k = Nat.gcd b k) :=
sorry

end NUMINAMATH_CALUDE_plant_species_numbering_not_unique_l1935_193584


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l1935_193564

def f (x : ℝ) : ℝ := -x^2 + 5

theorem parabola_point_ordering :
  ∀ (y₁ y₂ y₃ : ℝ),
  f (-4) = y₁ ∧ f (-1) = y₂ ∧ f 2 = y₃ →
  y₂ > y₃ ∧ y₃ > y₁ :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l1935_193564


namespace NUMINAMATH_CALUDE_smallest_divisible_by_four_and_five_l1935_193517

/-- A function that checks if a number contains the digits 1, 2, 3, 4, and 5 exactly once -/
def containsDigitsOnce (n : ℕ) : Prop := sorry

/-- A function that returns the set of all five-digit numbers containing 1, 2, 3, 4, and 5 exactly once -/
def fiveDigitSet : Set ℕ := {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧ containsDigitsOnce n}

theorem smallest_divisible_by_four_and_five :
  ∃ (n : ℕ), n ∈ fiveDigitSet ∧ n % 4 = 0 ∧ n % 5 = 0 ∧
  ∀ (m : ℕ), m ∈ fiveDigitSet → m % 4 = 0 → m % 5 = 0 → n ≤ m ∧
  n = 14532 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_four_and_five_l1935_193517


namespace NUMINAMATH_CALUDE_movie_ticket_sales_l1935_193594

theorem movie_ticket_sales (adult_price student_price total_revenue : ℚ)
  (student_tickets : ℕ) (h1 : adult_price = 4)
  (h2 : student_price = 5 / 2) (h3 : total_revenue = 445 / 2)
  (h4 : student_tickets = 9) :
  ∃ (adult_tickets : ℕ),
    adult_price * adult_tickets + student_price * student_tickets = total_revenue ∧
    adult_tickets + student_tickets = 59 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_sales_l1935_193594


namespace NUMINAMATH_CALUDE_circle_radius_is_six_l1935_193573

theorem circle_radius_is_six (r : ℝ) (h : r > 0) :
  2 * 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2) → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_six_l1935_193573


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1935_193565

theorem sqrt_equation_solution (t : ℝ) : 
  Real.sqrt (3 * Real.sqrt (2 * t - 1)) = (12 - 2 * t) ^ (1/4) → t = 21/20 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1935_193565


namespace NUMINAMATH_CALUDE_bronze_status_donation_bound_l1935_193567

/-- Represents the fundraising status of the school --/
structure FundraisingStatus where
  goal : ℕ
  remaining : ℕ
  bronzeFamilies : ℕ
  silverFamilies : ℕ
  goldFamilies : ℕ

/-- Represents the donation tiers --/
structure DonationTiers where
  bronze : ℕ
  silver : ℕ
  gold : ℕ

/-- The Bronze Status donation is less than or equal to the remaining amount needed --/
theorem bronze_status_donation_bound (status : FundraisingStatus) (tiers : DonationTiers) :
  status.goal = 750 ∧
  status.remaining = 50 ∧
  status.bronzeFamilies = 10 ∧
  status.silverFamilies = 7 ∧
  status.goldFamilies = 1 ∧
  tiers.bronze ≤ tiers.silver ∧
  tiers.silver ≤ tiers.gold →
  tiers.bronze ≤ status.remaining :=
by sorry

end NUMINAMATH_CALUDE_bronze_status_donation_bound_l1935_193567


namespace NUMINAMATH_CALUDE_circle_radius_with_area_four_l1935_193538

theorem circle_radius_with_area_four (r : ℝ) :
  r > 0 → π * r^2 = 4 → r = 2 / Real.sqrt π := by sorry

end NUMINAMATH_CALUDE_circle_radius_with_area_four_l1935_193538


namespace NUMINAMATH_CALUDE_circle_equation_problem1_circle_equation_problem2_l1935_193507

-- Problem 1
theorem circle_equation_problem1 (x y : ℝ) :
  (∃ (h : ℝ), x - 2*y - 2 = 0 ∧ 
    (x - 0)^2 + (y - 4)^2 = (x - 4)^2 + (y - 6)^2) →
  (x - 4)^2 + (y - 1)^2 = 25 :=
sorry

-- Problem 2
theorem circle_equation_problem2 (x y : ℝ) :
  (2*2 + 3*2 - 10 = 0 ∧
    ((x - 2)^2 + (y - 2)^2 = 13 ∧
     (y - 2)/(x - 2) * (-2/3) = -1)) →
  ((x - 4)^2 + (y - 5)^2 = 13 ∨ x^2 + (y + 1)^2 = 13) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_problem1_circle_equation_problem2_l1935_193507


namespace NUMINAMATH_CALUDE_x_value_proof_l1935_193555

theorem x_value_proof (x : ℝ) (h : 9 / (x^2) = x / 81) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1935_193555


namespace NUMINAMATH_CALUDE_true_propositions_l1935_193589

-- Define the propositions
variable (p₁ p₂ p₃ p₄ : Prop)

-- Define the truth values of the propositions
axiom p₁_true : p₁
axiom p₂_false : ¬p₂
axiom p₃_false : ¬p₃
axiom p₄_true : p₄

-- Define the compound propositions
def prop1 := p₁ ∧ p₄
def prop2 := p₁ ∧ p₂
def prop3 := ¬p₂ ∨ p₃
def prop4 := ¬p₃ ∨ ¬p₄

-- Theorem to prove
theorem true_propositions : 
  prop1 p₁ p₄ ∧ prop3 p₂ p₃ ∧ prop4 p₃ p₄ ∧ ¬(prop2 p₁ p₂) :=
sorry

end NUMINAMATH_CALUDE_true_propositions_l1935_193589


namespace NUMINAMATH_CALUDE_sqrt_two_function_value_l1935_193534

/-- Given a function f where f(x-1) = x^2 - 2x for all real x, prove that f(√2) = 1 -/
theorem sqrt_two_function_value (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x - 1) = x^2 - 2*x) : 
  f (Real.sqrt 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_function_value_l1935_193534


namespace NUMINAMATH_CALUDE_system_solution_l1935_193558

theorem system_solution :
  let x : ℚ := -49/23
  let y : ℚ := 136/69
  (7 * x = -9 - 3 * y) ∧ (4 * x = 5 * y - 34) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1935_193558


namespace NUMINAMATH_CALUDE_large_jar_capacity_l1935_193549

/-- Given a shelf of jars with the following properties:
  * There are 100 total jars
  * Small jars hold 3 liters each
  * The total capacity of all jars is 376 liters
  * There are 62 small jars
  This theorem proves that each large jar holds 5 liters. -/
theorem large_jar_capacity (total_jars : ℕ) (small_jar_capacity : ℕ) (total_capacity : ℕ) (small_jars : ℕ)
  (h1 : total_jars = 100)
  (h2 : small_jar_capacity = 3)
  (h3 : total_capacity = 376)
  (h4 : small_jars = 62) :
  (total_capacity - small_jars * small_jar_capacity) / (total_jars - small_jars) = 5 := by
  sorry

end NUMINAMATH_CALUDE_large_jar_capacity_l1935_193549


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1935_193521

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3^x ∧ x ≤ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1935_193521


namespace NUMINAMATH_CALUDE_count_four_digit_integers_l1935_193511

theorem count_four_digit_integers (y : ℕ) : 
  (∃ (n : ℕ), 1000 ≤ y ∧ y < 10000 ∧ (5678 * y + 123) % 29 = 890 % 29) →
  (Finset.filter (λ y => 1000 ≤ y ∧ y < 10000 ∧ (5678 * y + 123) % 29 = 890 % 29) (Finset.range 10000)).card = 310 :=
by sorry

end NUMINAMATH_CALUDE_count_four_digit_integers_l1935_193511


namespace NUMINAMATH_CALUDE_horse_journey_l1935_193560

/-- Given a geometric sequence with common ratio 1/2 and sum of first 7 terms equal to 700,
    the sum of the first 14 terms is 22575/32 -/
theorem horse_journey (a : ℝ) (S : ℕ → ℝ) : 
  (∀ n, S (n + 1) = S n + a * (1/2)^n) → 
  S 0 = 0 →
  S 7 = 700 →
  S 14 = 22575/32 := by
sorry

end NUMINAMATH_CALUDE_horse_journey_l1935_193560


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l1935_193536

/-- A tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  /-- The dihedral angle between faces ABC and BCD in radians -/
  dihedral_angle : ℝ
  /-- The area of triangle ABC -/
  area_ABC : ℝ
  /-- The area of triangle BCD -/
  area_BCD : ℝ
  /-- The length of edge BC -/
  length_BC : ℝ

/-- The volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume :
  ∃ t : Tetrahedron,
    t.dihedral_angle = 30 * (π / 180) ∧
    t.area_ABC = 120 ∧
    t.area_BCD = 80 ∧
    t.length_BC = 10 ∧
    volume t = 320 :=
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l1935_193536


namespace NUMINAMATH_CALUDE_no_valid_sequence_for_certain_n_l1935_193548

/-- A sequence where each number from 1 to n appears twice, 
    and the second occurrence of each number r is r positions after its first occurrence -/
def ValidSequence (n : ℕ) (seq : List ℕ) : Prop :=
  (seq.length = 2 * n) ∧
  (∀ r ∈ Finset.range n, 
    ∃ i j, seq.nthLe i (by sorry) = r + 1 ∧ 
           seq.nthLe j (by sorry) = r + 1 ∧ 
           j = i + (r + 1))

theorem no_valid_sequence_for_certain_n (n : ℕ) :
  (∃ seq : List ℕ, ValidSequence n seq) → 
  (n % 4 ≠ 2 ∧ n % 4 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_sequence_for_certain_n_l1935_193548


namespace NUMINAMATH_CALUDE_jimmy_cards_l1935_193535

/-- 
Given:
- Jimmy gives 3 cards to Bob
- Jimmy gives twice as many cards to Mary as he gave to Bob
- Jimmy has 9 cards left after giving away cards

Prove that Jimmy initially had 18 cards.
-/
theorem jimmy_cards : 
  ∀ (cards_to_bob cards_to_mary cards_left initial_cards : ℕ),
  cards_to_bob = 3 →
  cards_to_mary = 2 * cards_to_bob →
  cards_left = 9 →
  initial_cards = cards_to_bob + cards_to_mary + cards_left →
  initial_cards = 18 := by
sorry


end NUMINAMATH_CALUDE_jimmy_cards_l1935_193535


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l1935_193516

/-- The range of b for which the line y = x + b intersects the circle (x-2)^2 + (y-3)^2 = 4
    within the constraints 0 ≤ x ≤ 4 and 1 ≤ y ≤ 3 -/
theorem line_circle_intersection_range :
  ∀ b : ℝ,
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 3 ∧
   y = x + b ∧ (x - 2)^2 + (y - 3)^2 = 4) ↔
  (1 - 2 * Real.sqrt 2 ≤ b ∧ b ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l1935_193516


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1935_193512

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (m^2 - 4*m - 1 = 0) → 
  (n^2 - 4*n - 1 = 0) → 
  (m + n = 4) → 
  (m * n = -1) → 
  m + n - m * n = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1935_193512


namespace NUMINAMATH_CALUDE_jack_sugar_final_amount_l1935_193556

/-- Calculates the final amount of sugar Jack has after a series of transactions -/
def final_sugar_amount (initial : ℤ) (use_day2 borrow_day2 buy_day3 buy_day4 use_day5 return_day5 : ℤ) : ℤ :=
  initial - use_day2 - borrow_day2 + buy_day3 + buy_day4 - use_day5 + return_day5

/-- Theorem stating that Jack's final sugar amount is 85 pounds -/
theorem jack_sugar_final_amount :
  final_sugar_amount 65 18 5 30 20 10 3 = 85 := by
  sorry

end NUMINAMATH_CALUDE_jack_sugar_final_amount_l1935_193556


namespace NUMINAMATH_CALUDE_smallest_y_for_divisibility_by_11_l1935_193547

/-- Given a number in the form 7y86038 where y is a single digit (0-9),
    2 is the smallest whole number for y that makes the number divisible by 11. -/
theorem smallest_y_for_divisibility_by_11 :
  ∃ (y : ℕ), y ≤ 9 ∧ 
  (7 * 10^6 + y * 10^5 + 8 * 10^4 + 6 * 10^3 + 0 * 10^2 + 3 * 10 + 8) % 11 = 0 ∧
  ∀ (z : ℕ), z < y → (7 * 10^6 + z * 10^5 + 8 * 10^4 + 6 * 10^3 + 0 * 10^2 + 3 * 10 + 8) % 11 ≠ 0 ∧
  y = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_for_divisibility_by_11_l1935_193547


namespace NUMINAMATH_CALUDE_roots_polynomial_d_values_l1935_193553

theorem roots_polynomial_d_values (u v c d : ℝ) : 
  (∃ w : ℝ, {u, v, w} = {x | x^3 + c*x + d = 0}) ∧
  (∃ w : ℝ, {u+3, v-2, w} = {x | x^3 + c*x + (d+120) = 0}) →
  d = 84 ∨ d = -25 := by
sorry

end NUMINAMATH_CALUDE_roots_polynomial_d_values_l1935_193553


namespace NUMINAMATH_CALUDE_janes_age_l1935_193526

theorem janes_age (agnes_age : ℕ) (future_years : ℕ) (jane_age : ℕ) : 
  agnes_age = 25 → 
  future_years = 13 → 
  agnes_age + future_years = 2 * (jane_age + future_years) → 
  jane_age = 6 := by
sorry

end NUMINAMATH_CALUDE_janes_age_l1935_193526


namespace NUMINAMATH_CALUDE_extra_fee_is_fifteen_l1935_193541

/-- Represents the data plan charges and fees -/
structure DataPlan where
  normalMonthlyCharge : ℝ
  promotionalRate : ℝ
  totalPaid : ℝ
  extraFee : ℝ

/-- Calculates the extra fee for going over the data limit -/
def calculateExtraFee (plan : DataPlan) : Prop :=
  let firstMonthCharge := plan.normalMonthlyCharge * plan.promotionalRate
  let regularMonthsCharge := plan.normalMonthlyCharge * 5
  let totalWithoutExtra := firstMonthCharge + regularMonthsCharge
  plan.extraFee = plan.totalPaid - totalWithoutExtra

/-- Theorem stating the extra fee is $15 given the problem conditions -/
theorem extra_fee_is_fifteen :
  ∃ (plan : DataPlan),
    plan.normalMonthlyCharge = 30 ∧
    plan.promotionalRate = 1/3 ∧
    plan.totalPaid = 175 ∧
    calculateExtraFee plan ∧
    plan.extraFee = 15 := by
  sorry

end NUMINAMATH_CALUDE_extra_fee_is_fifteen_l1935_193541


namespace NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l1935_193551

theorem max_value_of_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 5) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2 * x + y ≤ z → z ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l1935_193551


namespace NUMINAMATH_CALUDE_impossible_all_positive_l1935_193501

/-- Represents a 4x4 grid of integers -/
def Grid := Fin 4 → Fin 4 → Int

/-- The initial grid configuration -/
def initial_grid : Grid :=
  fun i j => if i = 2 ∧ j = 3 then -1 else 1

/-- Represents an operation on the grid -/
inductive Operation
  | row (i : Fin 4)
  | col (j : Fin 4)
  | diag (d : Fin 7)

/-- Applies an operation to a grid -/
def apply_operation (g : Grid) (op : Operation) : Grid :=
  match op with
  | Operation.row i => fun x y => if x = i then -g x y else g x y
  | Operation.col j => fun x y => if y = j then -g x y else g x y
  | Operation.diag d => fun x y => if x + y = d then -g x y else g x y

/-- Applies a sequence of operations to a grid -/
def apply_operations (g : Grid) (ops : List Operation) : Grid :=
  ops.foldl apply_operation g

/-- Predicate to check if all cells in a grid are positive -/
def all_positive (g : Grid) : Prop :=
  ∀ i j, g i j > 0

/-- The main theorem -/
theorem impossible_all_positive (ops : List Operation) :
  ¬(all_positive (apply_operations initial_grid ops)) :=
sorry

end NUMINAMATH_CALUDE_impossible_all_positive_l1935_193501


namespace NUMINAMATH_CALUDE_replacement_cost_theorem_l1935_193597

/-- The cost to replace all cardio machines in a chain of gyms -/
def total_replacement_cost (num_gyms : ℕ) (bikes_per_gym treadmills_per_gym ellipticals_per_gym : ℕ)
  (bike_cost : ℝ) : ℝ :=
  let treadmill_cost := 1.5 * bike_cost
  let elliptical_cost := 2 * treadmill_cost
  let total_bikes := num_gyms * bikes_per_gym
  let total_treadmills := num_gyms * treadmills_per_gym
  let total_ellipticals := num_gyms * ellipticals_per_gym
  total_bikes * bike_cost + total_treadmills * treadmill_cost + total_ellipticals * elliptical_cost

/-- Theorem stating the total cost to replace all cardio machines -/
theorem replacement_cost_theorem :
  total_replacement_cost 20 10 5 5 700 = 455000 := by
  sorry


end NUMINAMATH_CALUDE_replacement_cost_theorem_l1935_193597


namespace NUMINAMATH_CALUDE_exists_double_area_quadrilateral_l1935_193579

/-- The area of a quadrilateral given by four points in the plane -/
noncomputable def quadrilateralArea (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the existence of points A, B, C, and D such that 
    the area of ABCD is twice the area of ADBC -/
theorem exists_double_area_quadrilateral :
  ∃ (A B C D : ℝ × ℝ), quadrilateralArea A B C D = 2 * quadrilateralArea A D B C := by
  sorry

end NUMINAMATH_CALUDE_exists_double_area_quadrilateral_l1935_193579


namespace NUMINAMATH_CALUDE_probability_factor_less_than_7_l1935_193545

def factors_of_72 : Finset ℕ := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}

def factors_less_than_7 : Finset ℕ := {1, 2, 3, 4, 6}

theorem probability_factor_less_than_7 :
  (factors_less_than_7.card : ℚ) / (factors_of_72.card : ℚ) = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_7_l1935_193545


namespace NUMINAMATH_CALUDE_oranges_from_ann_l1935_193524

theorem oranges_from_ann (initial_oranges final_oranges : ℕ) 
  (h1 : initial_oranges = 9)
  (h2 : final_oranges = 38) :
  final_oranges - initial_oranges = 29 := by
  sorry

end NUMINAMATH_CALUDE_oranges_from_ann_l1935_193524


namespace NUMINAMATH_CALUDE_bad_shape_cards_l1935_193510

/-- Calculates the number of baseball cards in bad shape given the initial conditions and distributions --/
theorem bad_shape_cards (initial : ℕ) (from_father : ℕ) (from_ebay : ℕ) (to_dexter : ℕ) (kept : ℕ) : 
  initial + from_father + from_ebay - (to_dexter + kept) = 4 :=
by
  sorry

#check bad_shape_cards 4 13 36 29 20

end NUMINAMATH_CALUDE_bad_shape_cards_l1935_193510


namespace NUMINAMATH_CALUDE_sequence_inequality_range_l1935_193588

/-- Given a sequence a_n with sum S_n, prove the range of t -/
theorem sequence_inequality_range (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) : 
  (∀ n : ℕ, 2 * S n = (n + 1) * a n) →  -- Condition: 2S_n = (n+1)a_n
  (a 1 = 1) →  -- Condition: a_1 = 1
  (∀ n : ℕ, n ≥ 2 → a n = n) →  -- Derived from conditions
  (t > 0) →  -- Condition: t > 0
  (∃! n : ℕ, n > 0 ∧ a n^2 - t * a n - 2 * t^2 < 0) →  -- Condition: unique positive n satisfying inequality
  t ∈ Set.Ioo (1/2 : ℝ) 1 :=  -- Conclusion: t is in the open interval (1/2, 1]
sorry

end NUMINAMATH_CALUDE_sequence_inequality_range_l1935_193588


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1935_193552

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) 
  (h1 : total_students = 466)
  (h2 : boys = 127)
  (h3 : boys < total_students - boys) :
  total_students - boys - boys = 212 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1935_193552


namespace NUMINAMATH_CALUDE_divisible_by_four_or_seven_l1935_193523

theorem divisible_by_four_or_seven : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 60 ∧ (4 ∣ n ∨ 7 ∣ n)) ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 60 ∧ (4 ∣ n ∨ 7 ∣ n) → n ∈ S) ∧
  Finset.card S = 21 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_four_or_seven_l1935_193523


namespace NUMINAMATH_CALUDE_pictures_from_phone_l1935_193544

-- Define the problem parameters
def num_albums : ℕ := 3
def pics_per_album : ℕ := 2
def camera_pics : ℕ := 4

-- Define the total number of pictures
def total_pics : ℕ := num_albums * pics_per_album

-- Define the number of pictures from the phone
def phone_pics : ℕ := total_pics - camera_pics

-- Theorem statement
theorem pictures_from_phone : phone_pics = 2 := by
  sorry

end NUMINAMATH_CALUDE_pictures_from_phone_l1935_193544


namespace NUMINAMATH_CALUDE_book_purchasing_problem_l1935_193576

/-- Represents a book purchasing plan. -/
structure BookPlan where
  classics : ℕ
  comics : ℕ

/-- Checks if a book plan is valid according to the given conditions. -/
def isValidPlan (p : BookPlan) (classicPrice comicPrice : ℕ) : Prop :=
  p.comics = p.classics + 20 ∧
  p.classics + p.comics ≥ 72 ∧
  classicPrice * p.classics + comicPrice * p.comics ≤ 2000

theorem book_purchasing_problem :
  ∃ (classicPrice comicPrice : ℕ),
    -- Given conditions
    20 * classicPrice + 40 * comicPrice = 1520 ∧
    20 * classicPrice - 20 * comicPrice = 440 ∧
    -- Prove the following
    classicPrice = 40 ∧
    comicPrice = 18 ∧
    (∀ p : BookPlan, isValidPlan p classicPrice comicPrice →
      (p.classics = 26 ∧ p.comics = 46) ∨
      (p.classics = 27 ∧ p.comics = 47) ∨
      (p.classics = 28 ∧ p.comics = 48)) ∧
    (∀ c : ℕ, c ∈ [26, 27, 28] →
      isValidPlan ⟨c, c + 20⟩ classicPrice comicPrice) :=
by sorry

end NUMINAMATH_CALUDE_book_purchasing_problem_l1935_193576


namespace NUMINAMATH_CALUDE_g_composition_of_three_l1935_193520

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- State the theorem
theorem g_composition_of_three : g (g (g 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l1935_193520


namespace NUMINAMATH_CALUDE_science_club_committee_formation_l1935_193592

theorem science_club_committee_formation (total_members : ℕ) 
                                         (new_members : ℕ) 
                                         (committee_size : ℕ) 
                                         (h1 : total_members = 20) 
                                         (h2 : new_members = 10) 
                                         (h3 : committee_size = 4) :
  (Nat.choose total_members committee_size) - 
  (Nat.choose new_members committee_size) = 4635 :=
sorry

end NUMINAMATH_CALUDE_science_club_committee_formation_l1935_193592


namespace NUMINAMATH_CALUDE_diamond_calculation_l1935_193585

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation :
  let x := diamond (diamond 1 3) 2
  let y := diamond 1 (diamond 3 2)
  x - y = -13/30 := by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l1935_193585


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1935_193537

theorem trigonometric_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1935_193537


namespace NUMINAMATH_CALUDE_min_draw_for_20_balls_l1935_193596

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  white : Nat
  blue : Nat

/-- The minimum number of balls to draw to ensure at least 20 of a single color -/
def minDrawToEnsure20 (counts : BallCounts) : Nat :=
  sorry

/-- The theorem stating the minimum number of balls to draw -/
theorem min_draw_for_20_balls (counts : BallCounts) 
  (h1 : counts.red = 23)
  (h2 : counts.green = 24)
  (h3 : counts.white = 12)
  (h4 : counts.blue = 21) :
  minDrawToEnsure20 counts = 70 :=
sorry

end NUMINAMATH_CALUDE_min_draw_for_20_balls_l1935_193596


namespace NUMINAMATH_CALUDE_digit_150_is_5_l1935_193506

-- Define the fraction
def fraction : ℚ := 5 / 37

-- Define the length of the repeating cycle
def cycle_length : ℕ := 3

-- Define the position we're interested in
def target_position : ℕ := 150

-- Define the function to get the nth digit after the decimal point
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem digit_150_is_5 : nth_digit target_position = 5 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_5_l1935_193506


namespace NUMINAMATH_CALUDE_inverse_p_is_true_l1935_193515

-- Define the original proposition
def p (x : ℝ) : Prop := x < -3 → x^2 - 2*x - 8 > 0

-- Define the inverse of the proposition
def p_inverse (x : ℝ) : Prop := ¬(x < -3) → ¬(x^2 - 2*x - 8 > 0)

-- Theorem stating that the inverse of p is true
theorem inverse_p_is_true : ∀ x : ℝ, p_inverse x :=
  sorry

end NUMINAMATH_CALUDE_inverse_p_is_true_l1935_193515


namespace NUMINAMATH_CALUDE_line_relationship_exclusive_line_relationship_unique_l1935_193569

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define the relationship between two lines
inductive LineRelationship
  | Parallel
  | Skew
  | Intersecting

-- Define a function to determine the relationship between two lines
def determineRelationship (l1 l2 : Line3D) : LineRelationship :=
  sorry

-- Theorem: Two lines must have exactly one of the three relationships
theorem line_relationship_exclusive (l1 l2 : Line3D) :
  (determineRelationship l1 l2 = LineRelationship.Parallel) ∨
  (determineRelationship l1 l2 = LineRelationship.Skew) ∨
  (determineRelationship l1 l2 = LineRelationship.Intersecting) :=
  sorry

-- Theorem: The relationship between two lines is unique
theorem line_relationship_unique (l1 l2 : Line3D) :
  ¬((determineRelationship l1 l2 = LineRelationship.Parallel) ∧
    (determineRelationship l1 l2 = LineRelationship.Skew)) ∧
  ¬((determineRelationship l1 l2 = LineRelationship.Parallel) ∧
    (determineRelationship l1 l2 = LineRelationship.Intersecting)) ∧
  ¬((determineRelationship l1 l2 = LineRelationship.Skew) ∧
    (determineRelationship l1 l2 = LineRelationship.Intersecting)) :=
  sorry

end NUMINAMATH_CALUDE_line_relationship_exclusive_line_relationship_unique_l1935_193569


namespace NUMINAMATH_CALUDE_compare_data_fluctuation_l1935_193500

def group_mean (g : String) : ℝ :=
  match g with
  | "A" => 80
  | "B" => 90
  | _ => 0

def group_variance (g : String) : ℝ :=
  match g with
  | "A" => 10
  | "B" => 5
  | _ => 0

def less_fluctuation (g1 g2 : String) : Prop :=
  group_variance g1 < group_variance g2

theorem compare_data_fluctuation (g1 g2 : String) :
  less_fluctuation g1 g2 → group_variance g1 < group_variance g2 :=
by sorry

end NUMINAMATH_CALUDE_compare_data_fluctuation_l1935_193500


namespace NUMINAMATH_CALUDE_circle_equation_l1935_193574

/-- Prove that the equation (x-1)^2 + (y-1)^2 = 2 represents the circle with center (1,1) passing through the point (2,2). -/
theorem circle_equation (x y : ℝ) : 
  (∀ (x₀ y₀ : ℝ), (x₀ - 1)^2 + (y₀ - 1)^2 = 2 ↔ 
    ((x₀ - 1)^2 + (y₀ - 1)^2 = (x - 1)^2 + (y - 1)^2 ∧ (x - 1)^2 + (y - 1)^2 = 1)) ∧
  (2 - 1)^2 + (2 - 1)^2 = 2 := by
sorry


end NUMINAMATH_CALUDE_circle_equation_l1935_193574


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l1935_193508

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the operations and relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Set Point)
variable (planes_parallel : Plane → Plane → Prop)

-- Define the given lines and planes
variable (m n l₁ l₂ : Line)
variable (α β : Plane)
variable (M : Point)

theorem parallel_planes_condition
  (h1 : subset m α)
  (h2 : subset n α)
  (h3 : subset l₁ β)
  (h4 : subset l₂ β)
  (h5 : intersect l₁ l₂ = {M})
  (h6 : parallel m l₁)
  (h7 : parallel n l₂) :
  planes_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l1935_193508


namespace NUMINAMATH_CALUDE_only_B_is_true_l1935_193559

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Proposition A
def propA (P₀ : Point2D) (l : Line2D) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ y - P₀.y = k * (x - P₀.x)

-- Proposition B
def propB (P₁ P₂ : Point2D) (l : Line2D) : Prop :=
  P₁ ≠ P₂ → ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ 
    (y - P₁.y) * (P₂.x - P₁.x) = (x - P₁.x) * (P₂.y - P₁.y)

-- Proposition C
def propC (l : Line2D) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ x / a + y / b = 1

-- Proposition D
def propD (b : ℝ) (l : Line2D) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ y = k * x + b

theorem only_B_is_true :
  (∃ P₀ : Point2D, ∀ l : Line2D, propA P₀ l) = false ∧
  (∀ P₁ P₂ : Point2D, ∀ l : Line2D, propB P₁ P₂ l) = true ∧
  (∀ l : Line2D, propC l) = false ∧
  (∃ b : ℝ, ∀ l : Line2D, propD b l) = false :=
sorry

end NUMINAMATH_CALUDE_only_B_is_true_l1935_193559


namespace NUMINAMATH_CALUDE_machinery_spending_l1935_193530

/-- Represents the financial breakdown of Kanul's spending --/
structure KanulSpending where
  total : ℝ
  rawMaterials : ℝ
  cash : ℝ
  machinery : ℝ

/-- Theorem stating the amount spent on machinery --/
theorem machinery_spending (k : KanulSpending) 
  (h1 : k.total = 1000)
  (h2 : k.rawMaterials = 500)
  (h3 : k.cash = 0.1 * k.total)
  (h4 : k.total = k.rawMaterials + k.cash + k.machinery) :
  k.machinery = 400 := by
  sorry

end NUMINAMATH_CALUDE_machinery_spending_l1935_193530


namespace NUMINAMATH_CALUDE_amount_subtracted_l1935_193543

theorem amount_subtracted (number : ℝ) (subtracted_amount : ℝ) : 
  number = 70 →
  (number / 2) - subtracted_amount = 25 →
  subtracted_amount = 10 := by
sorry

end NUMINAMATH_CALUDE_amount_subtracted_l1935_193543


namespace NUMINAMATH_CALUDE_batman_game_cost_l1935_193540

def football_cost : ℚ := 14.02
def strategy_cost : ℚ := 9.46
def total_spent : ℚ := 35.52

theorem batman_game_cost :
  ∃ (batman_cost : ℚ),
    batman_cost = total_spent - football_cost - strategy_cost ∧
    batman_cost = 12.04 :=
by sorry

end NUMINAMATH_CALUDE_batman_game_cost_l1935_193540


namespace NUMINAMATH_CALUDE_goods_train_length_l1935_193587

/-- Calculates the length of a goods train given the speeds of two trains
    traveling in opposite directions and the time taken for the goods train
    to pass a stationary observer in the other train. -/
theorem goods_train_length
  (speed_train : ℝ)
  (speed_goods : ℝ)
  (pass_time : ℝ)
  (h1 : speed_train = 15)
  (h2 : speed_goods = 97)
  (h3 : pass_time = 9)
  : ∃ (length : ℝ), abs (length - 279.99) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_goods_train_length_l1935_193587


namespace NUMINAMATH_CALUDE_horner_v4_value_l1935_193572

/-- The polynomial f(x) = 12 + 35x - 8x^2 + 79x^3 + 6x^4 + 5x^5 + 3x^6 -/
def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

/-- The fourth intermediate value in Horner's method for polynomial f -/
def v4 (x : ℝ) : ℝ := (((3*x + 5)*x + 6)*x + 79)*x - 8

/-- Theorem: The value of v4 for f(x) at x = -4 is 220 -/
theorem horner_v4_value : v4 (-4) = 220 := by sorry

end NUMINAMATH_CALUDE_horner_v4_value_l1935_193572


namespace NUMINAMATH_CALUDE_total_chewing_gums_l1935_193542

theorem total_chewing_gums (mary sam sue : ℕ) : 
  mary = 5 → sam = 10 → sue = 15 → mary + sam + sue = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_chewing_gums_l1935_193542


namespace NUMINAMATH_CALUDE_marbles_after_2000_steps_l1935_193566

/-- Represents the state of baskets with marbles -/
def BasketState := List Nat

/-- Converts a natural number to its base-6 representation -/
def toBase6 (n : Nat) : List Nat :=
  sorry

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List Nat) : Nat :=
  sorry

/-- Simulates the marble placement process for a given number of steps -/
def simulateMarblePlacement (steps : Nat) : BasketState :=
  sorry

/-- Counts the total number of marbles in a given basket state -/
def countMarbles (state : BasketState) : Nat :=
  sorry

/-- Theorem stating that the number of marbles after 2000 steps
    is equal to the sum of digits in the base-6 representation of 2000 -/
theorem marbles_after_2000_steps :
  countMarbles (simulateMarblePlacement 2000) = sumDigits (toBase6 2000) :=
by sorry

end NUMINAMATH_CALUDE_marbles_after_2000_steps_l1935_193566


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1935_193571

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (2 * x - a < 0 ∧ 1 - 2 * x ≥ 7) ↔ x ≤ -3) → 
  a > -6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1935_193571


namespace NUMINAMATH_CALUDE_map_scale_conversion_l1935_193595

/-- Given a map scale where 10 cm represents 50 km, 
    prove that a 23 cm length on the map represents 115 km. -/
theorem map_scale_conversion (scale : ℝ → ℝ) : 
  (scale 10 = 50) → (scale 23 = 115) :=
by
  sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l1935_193595


namespace NUMINAMATH_CALUDE_large_circle_radius_l1935_193599

theorem large_circle_radius (C₁ C₂ C₃ C₄ O : ℝ × ℝ) (r : ℝ) :
  -- Four unit circles externally tangent in square formation
  r = 1 ∧
  dist C₁ C₂ = 2 ∧ dist C₂ C₃ = 2 ∧ dist C₃ C₄ = 2 ∧ dist C₄ C₁ = 2 ∧
  -- Large circle internally tangent to the four unit circles
  dist O C₁ = dist O C₂ ∧ dist O C₂ = dist O C₃ ∧ dist O C₃ = dist O C₄ ∧
  dist O C₁ = dist C₁ C₃ / 2 + r →
  -- Radius of the large circle
  dist O C₁ + r = Real.sqrt 2 + 2 := by
sorry


end NUMINAMATH_CALUDE_large_circle_radius_l1935_193599


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1935_193539

/-- Given a circle with equation 2x^2 = -2y^2 + 16x - 8y + 40, 
    the area of a square inscribed around it with one pair of sides 
    parallel to the x-axis is 160 square units. -/
theorem inscribed_square_area (x y : ℝ) : 
  2 * x^2 = -2 * y^2 + 16 * x - 8 * y + 40 → 
  ∃ (s : ℝ), s > 0 ∧ s^2 = 160 ∧ 
  ∃ (cx cy : ℝ), (x - cx)^2 + (y - cy)^2 ≤ (s/2)^2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1935_193539


namespace NUMINAMATH_CALUDE_max_students_is_18_l1935_193591

/-- Represents the structure of Ms. Gregory's class -/
structure ClassStructure where
  boys : ℕ
  girls : ℕ
  science_club : ℕ
  math_club : ℕ

/-- Checks if the given class structure satisfies all conditions -/
def is_valid_structure (c : ClassStructure) : Prop :=
  3 * c.boys = 4 * c.science_club ∧ 
  2 * c.girls = 3 * c.science_club ∧ 
  c.math_club = 2 * c.science_club ∧
  c.boys + c.girls = c.science_club + c.math_club

/-- The maximum number of students in Ms. Gregory's class -/
def max_students : ℕ := 18

/-- Theorem stating that the maximum number of students is 18 -/
theorem max_students_is_18 : 
  ∀ c : ClassStructure, is_valid_structure c → c.boys + c.girls ≤ max_students :=
by
  sorry

#check max_students_is_18

end NUMINAMATH_CALUDE_max_students_is_18_l1935_193591


namespace NUMINAMATH_CALUDE_hotel_assignment_problem_l1935_193514

/-- The number of ways to assign friends to rooms -/
def assignFriendsToRooms (numFriends numRooms maxPerRoom : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of assignments for the given problem -/
theorem hotel_assignment_problem :
  assignFriendsToRooms 6 5 2 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_hotel_assignment_problem_l1935_193514


namespace NUMINAMATH_CALUDE_unique_real_root_of_polynomial_l1935_193528

theorem unique_real_root_of_polynomial (x : ℝ) :
  x^4 - 4*x^3 + 5*x^2 - 2*x + 2 = 0 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_real_root_of_polynomial_l1935_193528


namespace NUMINAMATH_CALUDE_circle_C_equation_max_y_over_x_l1935_193590

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the line that intersects with x-axis to form the center of circle C
def center_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the line tangent to circle C
def tangent_line (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the circle for the second part of the problem
def circle_P (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

-- Theorem for the first part of the problem
theorem circle_C_equation :
  ∀ x y : ℝ, 
  (∃ x₀, center_line x₀ 0 ∧ (∀ x y, circle_C x y → (x - x₀)^2 + y^2 = 2)) →
  (∃ d : ℝ, d > 0 ∧ ∀ x y, circle_C x y → d = |x + y + 3| / Real.sqrt 2) →
  circle_C x y ↔ (x + 1)^2 + y^2 = 2 :=
sorry

-- Theorem for the second part of the problem
theorem max_y_over_x :
  (∃ k : ℝ, k = Real.sqrt 3 / 3 ∧ 
   ∀ x y : ℝ, circle_P x y → |y / x| ≤ k ∧ 
   ∃ x₀ y₀ : ℝ, circle_P x₀ y₀ ∧ |y₀ / x₀| = k) :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_max_y_over_x_l1935_193590


namespace NUMINAMATH_CALUDE_other_colors_correct_l1935_193525

/-- Represents a school with its student data -/
structure School where
  total_students : ℕ
  blue_percent : ℚ
  red_percent : ℚ
  green_percent : ℚ
  blue_red_percent : ℚ
  blue_green_percent : ℚ
  red_green_percent : ℚ

/-- Calculates the number of students wearing other colors -/
def other_colors (s : School) : ℕ :=
  s.total_students - (s.total_students * (s.blue_percent + s.red_percent + s.green_percent - 
    s.blue_red_percent - s.blue_green_percent - s.red_green_percent)).ceil.toNat

/-- The first school's data -/
def school1 : School := {
  total_students := 800,
  blue_percent := 30/100,
  red_percent := 20/100,
  green_percent := 10/100,
  blue_red_percent := 5/100,
  blue_green_percent := 3/100,
  red_green_percent := 2/100
}

/-- The second school's data -/
def school2 : School := {
  total_students := 700,
  blue_percent := 25/100,
  red_percent := 25/100,
  green_percent := 20/100,
  blue_red_percent := 10/100,
  blue_green_percent := 5/100,
  red_green_percent := 3/100
}

/-- The third school's data -/
def school3 : School := {
  total_students := 500,
  blue_percent := 1/100,
  red_percent := 1/100,
  green_percent := 1/100,
  blue_red_percent := 1/2/100,
  blue_green_percent := 1/2/100,
  red_green_percent := 1/2/100
}

/-- Theorem stating the correct number of students wearing other colors in each school -/
theorem other_colors_correct :
  other_colors school1 = 400 ∧
  other_colors school2 = 336 ∧
  other_colors school3 = 475 := by
  sorry

end NUMINAMATH_CALUDE_other_colors_correct_l1935_193525


namespace NUMINAMATH_CALUDE_total_wheels_is_150_l1935_193527

/-- The total number of wheels Naomi saw at the park -/
def total_wheels : ℕ :=
  let regular_bikes := 7
  let children_bikes := 11
  let tandem_bikes_4 := 5
  let tandem_bikes_6 := 3
  let unicycles := 4
  let tricycles := 6
  let training_wheel_bikes := 8

  let regular_bike_wheels := 2
  let children_bike_wheels := 4
  let tandem_bike_4_wheels := 4
  let tandem_bike_6_wheels := 6
  let unicycle_wheels := 1
  let tricycle_wheels := 3
  let training_wheel_bike_wheels := 4

  regular_bikes * regular_bike_wheels +
  children_bikes * children_bike_wheels +
  tandem_bikes_4 * tandem_bike_4_wheels +
  tandem_bikes_6 * tandem_bike_6_wheels +
  unicycles * unicycle_wheels +
  tricycles * tricycle_wheels +
  training_wheel_bikes * training_wheel_bike_wheels

theorem total_wheels_is_150 : total_wheels = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_150_l1935_193527


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l1935_193582

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 20

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 25

/-- The total number of chocolate bars in the large box -/
def total_bars : ℕ := num_small_boxes * bars_per_small_box

theorem chocolate_bars_count : total_bars = 500 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l1935_193582


namespace NUMINAMATH_CALUDE_algebraic_inequalities_l1935_193533

theorem algebraic_inequalities :
  (∀ a : ℝ, a^2 + 2 > 2*a) ∧
  (∀ x : ℝ, (x+5)*(x+7) < (x+6)^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_inequalities_l1935_193533


namespace NUMINAMATH_CALUDE_value_of_5_minus_c_l1935_193504

theorem value_of_5_minus_c (c d : ℤ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 7 + d = 10 + c) : 
  5 - c = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_5_minus_c_l1935_193504


namespace NUMINAMATH_CALUDE_statement_b_incorrect_l1935_193554

/-- A predicate representing the conditions for a point to be on a locus -/
def LocusCondition (α : Type*) := α → Prop

/-- A predicate representing the geometric locus itself -/
def GeometricLocus (α : Type*) := α → Prop

/-- Statement B: If a point is on the locus, then it satisfies the conditions;
    however, there may be points not on the locus that also satisfy these conditions. -/
def StatementB (α : Type*) (locus : GeometricLocus α) (condition : LocusCondition α) : Prop :=
  (∀ x : α, locus x → condition x) ∧
  ∃ y : α, condition y ∧ ¬locus y

/-- Theorem stating that Statement B is an incorrect method for defining a geometric locus -/
theorem statement_b_incorrect (α : Type*) :
  ¬∀ (locus : GeometricLocus α) (condition : LocusCondition α),
    StatementB α locus condition ↔ (∀ x : α, locus x ↔ condition x) :=
sorry

end NUMINAMATH_CALUDE_statement_b_incorrect_l1935_193554


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l1935_193519

theorem shaded_fraction_of_rectangle (length width : ℝ) (h1 : length = 10) (h2 : width = 15) :
  let total_area := length * width
  let third_area := total_area / 3
  let shaded_area := third_area / 2
  shaded_area / total_area = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l1935_193519
