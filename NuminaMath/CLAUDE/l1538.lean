import Mathlib

namespace NUMINAMATH_CALUDE_english_chinese_difference_l1538_153884

def hours_english : ℕ := 6
def hours_chinese : ℕ := 3

theorem english_chinese_difference : hours_english - hours_chinese = 3 := by
  sorry

end NUMINAMATH_CALUDE_english_chinese_difference_l1538_153884


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1538_153846

/-- Given a quadratic equation of the form (x^2 - bx + b^2) / (ax^2 - c) = (m-1) / (m+1),
    if the roots are numerically equal but of opposite signs, and c = b^2,
    then m = (a-1) / (a+1) -/
theorem quadratic_equation_roots (a b m : ℝ) :
  (∃ x y : ℝ, x = -y ∧ x ≠ 0 ∧
    (x^2 - b*x + b^2) / (a*x^2 - b^2) = (m-1) / (m+1) ∧
    (y^2 - b*y + b^2) / (a*y^2 - b^2) = (m-1) / (m+1)) →
  m = (a-1) / (a+1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1538_153846


namespace NUMINAMATH_CALUDE_total_boxes_sold_l1538_153849

def boxes_sold (friday saturday sunday monday : ℕ) : ℕ :=
  friday + saturday + sunday + monday

theorem total_boxes_sold :
  ∀ (friday saturday sunday monday : ℕ),
    friday = 40 →
    saturday = 2 * friday - 10 →
    sunday = saturday / 2 →
    monday = sunday + (sunday / 4 + 1) →
    boxes_sold friday saturday sunday monday = 189 :=
by sorry

end NUMINAMATH_CALUDE_total_boxes_sold_l1538_153849


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l1538_153851

theorem no_solution_fractional_equation :
  ∀ y : ℝ, y ≠ 3 → (y - 2) / (y - 3) ≠ 2 - 1 / (3 - y) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l1538_153851


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l1538_153853

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 2016) :
  (∃ k : ℕ, k = 334 ∧ 
   (∀ m : ℕ, n^m ∣ n! ↔ m ≤ k)) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l1538_153853


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1538_153885

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1538_153885


namespace NUMINAMATH_CALUDE_mem_not_veen_l1538_153817

-- Define the sets
variable (U : Type) -- Universe set
variable (Mem En Veen : Set U)

-- Define the hypotheses
variable (h1 : Mem ⊆ En)
variable (h2 : En ∩ Veen = ∅)

-- Theorem to prove
theorem mem_not_veen :
  (∀ x, x ∈ Mem → x ∉ Veen) ∧
  (Mem ∩ Veen = ∅) :=
sorry

end NUMINAMATH_CALUDE_mem_not_veen_l1538_153817


namespace NUMINAMATH_CALUDE_cube_sum_root_l1538_153870

theorem cube_sum_root : Real.sqrt (3^3 + 3^3 + 3^3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_root_l1538_153870


namespace NUMINAMATH_CALUDE_sum_of_fifty_eights_l1538_153835

theorem sum_of_fifty_eights : (List.replicate 50 8).sum = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifty_eights_l1538_153835


namespace NUMINAMATH_CALUDE_g_value_at_10_l1538_153842

theorem g_value_at_10 (g : ℕ → ℝ) 
  (h1 : g 1 = 1)
  (h2 : ∀ (m n : ℕ), m ≥ n → g (m + n) + g (m - n) = (g (2*m) + g (2*n))/2 + 2) :
  g 10 = 102 := by
sorry

end NUMINAMATH_CALUDE_g_value_at_10_l1538_153842


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_315_l1538_153801

theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_315_l1538_153801


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1538_153883

theorem sphere_surface_area (c : Real) (h : c = 2 * Real.pi) :
  ∃ (r : Real), 
    c = 2 * Real.pi * r ∧ 
    4 * Real.pi * r^2 = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1538_153883


namespace NUMINAMATH_CALUDE_equation_pattern_l1538_153802

theorem equation_pattern (n : ℕ) (hn : n > 0) : 9 * (n - 1) + n = 10 * (n - 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_pattern_l1538_153802


namespace NUMINAMATH_CALUDE_line_cartesian_to_polar_l1538_153881

/-- Given a line in Cartesian coordinates x cos α + y sin α = 0,
    its equivalent polar coordinate equation is θ = α - π/2 --/
theorem line_cartesian_to_polar (α : Real) :
  ∀ x y r θ : Real,
  (x * Real.cos α + y * Real.sin α = 0) →
  (x = r * Real.cos θ) →
  (y = r * Real.sin θ) →
  (θ = α - π/2) := by
  sorry

end NUMINAMATH_CALUDE_line_cartesian_to_polar_l1538_153881


namespace NUMINAMATH_CALUDE_otimes_inequality_l1538_153892

/-- Custom binary operation ⊗ on ℝ -/
def otimes (x y : ℝ) : ℝ := (1 - x) * (1 + y)

/-- Theorem: If (x-a) ⊗ (x+a) < 1 holds for any real x, then -2 < a < 0 -/
theorem otimes_inequality (a : ℝ) : 
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -2 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_otimes_inequality_l1538_153892


namespace NUMINAMATH_CALUDE_trigonometric_fraction_equals_one_l1538_153872

theorem trigonometric_fraction_equals_one : 
  (Real.sin (22 * π / 180) * Real.cos (8 * π / 180) + 
   Real.cos (158 * π / 180) * Real.cos (98 * π / 180)) / 
  (Real.sin (23 * π / 180) * Real.cos (7 * π / 180) + 
   Real.cos (157 * π / 180) * Real.cos (97 * π / 180)) = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_fraction_equals_one_l1538_153872


namespace NUMINAMATH_CALUDE_division_not_imply_multiple_and_factor_l1538_153828

theorem division_not_imply_multiple_and_factor :
  ¬ (∀ a b : ℝ, a / b = 5 → (∃ k : ℤ, a = b * k) ∧ (∃ k : ℤ, b * k = a)) := by
  sorry

end NUMINAMATH_CALUDE_division_not_imply_multiple_and_factor_l1538_153828


namespace NUMINAMATH_CALUDE_total_area_is_62_l1538_153839

/-- The area of a figure composed of three rectangles -/
def figure_area (area1 area2 area3 : ℕ) : ℕ := area1 + area2 + area3

/-- Theorem: The total area of the figure is 62 square units -/
theorem total_area_is_62 (area1 area2 area3 : ℕ) 
  (h1 : area1 = 30) 
  (h2 : area2 = 12) 
  (h3 : area3 = 20) : 
  figure_area area1 area2 area3 = 62 := by
  sorry

#eval figure_area 30 12 20

end NUMINAMATH_CALUDE_total_area_is_62_l1538_153839


namespace NUMINAMATH_CALUDE_pipe_fill_time_l1538_153888

/-- Given two pipes A and B that can fill a tank, this theorem proves
    the time it takes for pipe B to fill the tank alone, given certain conditions. -/
theorem pipe_fill_time (fill_time_A fill_time_B : ℝ) : 
  fill_time_A = 24 →
  8 * (1 / fill_time_A + 1 / fill_time_B) + 10 * (1 / fill_time_A) = 1 →
  fill_time_B = 32 := by
  sorry


end NUMINAMATH_CALUDE_pipe_fill_time_l1538_153888


namespace NUMINAMATH_CALUDE_number_calculation_l1538_153843

theorem number_calculation (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 30) : 
  (40/100) * N = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l1538_153843


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1538_153836

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → a 1 + a 2 = -1 → a 3 = 4 → a 4 + a 5 = 17 := by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1538_153836


namespace NUMINAMATH_CALUDE_bryden_payment_is_correct_l1538_153847

/-- The face value of a state quarter in dollars -/
def quarter_value : ℝ := 0.25

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 6

/-- The percentage of face value the collector offers, expressed as a decimal -/
def collector_offer_percentage : ℝ := 16

/-- The discount percentage applied to the total payment, expressed as a decimal -/
def discount_percentage : ℝ := 0.1

/-- The amount Bryden receives for his state quarters -/
def bryden_payment : ℝ :=
  (bryden_quarters : ℝ) * quarter_value * collector_offer_percentage * (1 - discount_percentage)

theorem bryden_payment_is_correct :
  bryden_payment = 21.6 := by sorry

end NUMINAMATH_CALUDE_bryden_payment_is_correct_l1538_153847


namespace NUMINAMATH_CALUDE_barrel_capacity_l1538_153854

def number_of_barrels : ℕ := 4
def flow_rate : ℚ := 7/2
def fill_time : ℕ := 8

theorem barrel_capacity : 
  (flow_rate * fill_time) / number_of_barrels = 7 := by sorry

end NUMINAMATH_CALUDE_barrel_capacity_l1538_153854


namespace NUMINAMATH_CALUDE_custom_mul_solution_l1538_153804

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 2*a - b^2

/-- Theorem stating that if a * 3 = 3 under the custom multiplication, then a = 6 -/
theorem custom_mul_solution :
  ∀ a : ℝ, custom_mul a 3 = 3 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_solution_l1538_153804


namespace NUMINAMATH_CALUDE_square_root_of_nine_l1538_153816

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l1538_153816


namespace NUMINAMATH_CALUDE_fraction_comparison_l1538_153841

theorem fraction_comparison : 
  let original := -15 / 12
  let a := -30 / 24
  let b := -1 - 3 / 12
  let c := -1 - 9 / 36
  let d := -1 - 5 / 15
  let e := -1 - 25 / 100
  (a = original ∧ b = original ∧ c = original ∧ e = original) ∧ d ≠ original :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1538_153841


namespace NUMINAMATH_CALUDE_reliability_comparison_l1538_153812

/-- Probability of a 3-member system making a correct decision -/
def prob_3_correct (p : ℝ) : ℝ := 3 * p^2 * (1 - p) + p^3

/-- Probability of a 5-member system making a correct decision -/
def prob_5_correct (p : ℝ) : ℝ := 10 * p^3 * (1 - p)^2 + 5 * p^4 * (1 - p) + p^5

/-- A 5-member system is more reliable than a 3-member system -/
def more_reliable (p : ℝ) : Prop := prob_5_correct p > prob_3_correct p

theorem reliability_comparison (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  more_reliable p ↔ p > (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_reliability_comparison_l1538_153812


namespace NUMINAMATH_CALUDE_product_is_square_l1538_153858

theorem product_is_square (g : ℕ) (h : g = 14) : ∃ n : ℕ, 3150 * g = n^2 := by
  sorry

end NUMINAMATH_CALUDE_product_is_square_l1538_153858


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1538_153889

theorem sum_of_fractions : (3 : ℚ) / 10 + (3 : ℚ) / 1000 = 303 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1538_153889


namespace NUMINAMATH_CALUDE_dry_mixed_fruits_weight_is_188_l1538_153808

/-- Calculates the weight of dry mixed fruits after dehydration -/
def dryMixedFruitsWeight (freshGrapesWater freshApplesWater : ℝ)
                         (freshGrapesWeight freshApplesWeight : ℝ) : ℝ :=
  (1 - freshGrapesWater) * freshGrapesWeight + (1 - freshApplesWater) * freshApplesWeight

/-- Theorem: The weight of dry mixed fruits after dehydration is 188 kg -/
theorem dry_mixed_fruits_weight_is_188 :
  dryMixedFruitsWeight 0.65 0.84 400 300 = 188 := by
  sorry

end NUMINAMATH_CALUDE_dry_mixed_fruits_weight_is_188_l1538_153808


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l1538_153861

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_s

/-- Proof that a train's length is approximately 119.97 meters -/
theorem train_length_proof (speed_kmh : ℝ) (time_s : ℝ) 
  (h1 : speed_kmh = 48) 
  (h2 : time_s = 9) : 
  ∃ ε > 0, |train_length speed_kmh time_s - 119.97| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l1538_153861


namespace NUMINAMATH_CALUDE_abc_fraction_value_l1538_153822

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 2)
  (h2 : b * c / (b + c) = 5)
  (h3 : c * a / (c + a) = 9) :
  a * b * c / (a * b + b * c + c * a) = 90 / 73 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l1538_153822


namespace NUMINAMATH_CALUDE_ginger_garden_work_hours_l1538_153874

/-- Calculates the number of hours Ginger worked in her garden --/
def hours_worked (bottle_capacity : ℕ) (bottles_for_plants : ℕ) (total_water_used : ℕ) : ℕ :=
  (total_water_used - bottles_for_plants * bottle_capacity) / bottle_capacity

/-- Proves that Ginger worked 8 hours in her garden given the problem conditions --/
theorem ginger_garden_work_hours :
  let bottle_capacity : ℕ := 2
  let bottles_for_plants : ℕ := 5
  let total_water_used : ℕ := 26
  hours_worked bottle_capacity bottles_for_plants total_water_used = 8 := by
  sorry


end NUMINAMATH_CALUDE_ginger_garden_work_hours_l1538_153874


namespace NUMINAMATH_CALUDE_negative_numbers_roots_l1538_153887

theorem negative_numbers_roots :
  (∀ x : ℝ, x < 0 → ¬∃ y : ℝ, y ^ 2 = x) ∧
  (∀ x : ℝ, x < 0 → ∃ y : ℝ, y ^ 3 = x) :=
by sorry

end NUMINAMATH_CALUDE_negative_numbers_roots_l1538_153887


namespace NUMINAMATH_CALUDE_stating_boat_speed_with_stream_l1538_153825

/-- Represents the speed of a boat in different conditions. -/
structure BoatSpeed where
  stillWater : ℝ
  againstStream : ℝ
  withStream : ℝ

/-- 
Theorem stating that given a man's rowing speed in still water is 6 km/h 
and his speed against the stream is 10 km/h, his speed with the stream is 10 km/h.
-/
theorem boat_speed_with_stream 
  (speed : BoatSpeed) 
  (h1 : speed.stillWater = 6) 
  (h2 : speed.againstStream = 10) : 
  speed.withStream = 10 := by
  sorry

#check boat_speed_with_stream

end NUMINAMATH_CALUDE_stating_boat_speed_with_stream_l1538_153825


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1538_153814

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1538_153814


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l1538_153823

theorem sqrt_x_plus_one_real (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l1538_153823


namespace NUMINAMATH_CALUDE_power_of_two_sum_l1538_153809

theorem power_of_two_sum (x : ℕ) : 2^x + 2^x + 2^x + 2^x + 2^x = 2048 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l1538_153809


namespace NUMINAMATH_CALUDE_base_2_representation_of_125_l1538_153857

theorem base_2_representation_of_125 :
  ∃ (b : List Bool), 
    (b.length = 7) ∧
    (b.foldl (λ acc x => 2 * acc + if x then 1 else 0) 0 = 125) ∧
    (b = [true, true, true, true, true, false, true]) := by
  sorry

end NUMINAMATH_CALUDE_base_2_representation_of_125_l1538_153857


namespace NUMINAMATH_CALUDE_tan_equation_solutions_l1538_153869

theorem tan_equation_solutions (x : ℝ) :
  -π < x ∧ x ≤ π ∧ 2 * Real.tan x - Real.sqrt 3 = 0 ↔ 
  x = Real.arctan (Real.sqrt 3 / 2) ∨ x = Real.arctan (Real.sqrt 3 / 2) - π :=
by sorry

end NUMINAMATH_CALUDE_tan_equation_solutions_l1538_153869


namespace NUMINAMATH_CALUDE_max_boxes_fit_l1538_153856

def large_box_length : ℕ := 8
def large_box_width : ℕ := 7
def large_box_height : ℕ := 6

def small_box_length : ℕ := 4
def small_box_width : ℕ := 7
def small_box_height : ℕ := 6

def cm_per_meter : ℕ := 100

theorem max_boxes_fit (large_box_volume small_box_volume max_boxes : ℕ) : 
  large_box_volume = (large_box_length * cm_per_meter) * (large_box_width * cm_per_meter) * (large_box_height * cm_per_meter) →
  small_box_volume = small_box_length * small_box_width * small_box_height →
  max_boxes = large_box_volume / small_box_volume →
  max_boxes = 2000000 := by
  sorry

#check max_boxes_fit

end NUMINAMATH_CALUDE_max_boxes_fit_l1538_153856


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_36_degrees_l1538_153897

def angle_measure : ℝ := 36

def complement (x : ℝ) : ℝ := 90 - x

def supplement (x : ℝ) : ℝ := 180 - x

theorem supplement_of_complement_of_36_degrees : 
  supplement (complement angle_measure) = 126 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_36_degrees_l1538_153897


namespace NUMINAMATH_CALUDE_sequence_properties_l1538_153845

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

theorem sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a : 2 * a 5 - a 3 = 3)
  (h_b2 : b 2 = 1)
  (h_b4 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (∃ q : ℝ, (q = 2 ∨ q = -2) ∧ ∀ n : ℕ, b (n + 1) = b n * q) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1538_153845


namespace NUMINAMATH_CALUDE_words_per_page_l1538_153834

theorem words_per_page (total_pages : Nat) (max_words_per_page : Nat) (total_words_mod : Nat) :
  total_pages = 136 →
  max_words_per_page = 100 →
  total_words_mod = 184 →
  ∃ (words_per_page : Nat),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 203 = total_words_mod ∧
    words_per_page = 73 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l1538_153834


namespace NUMINAMATH_CALUDE_total_arrangements_l1538_153831

/-- Represents the number of people in the group -/
def total_people : Nat := 6

/-- Represents the number of people who must sit together -/
def group_size : Nat := 3

/-- Calculates the number of ways to arrange the group -/
def arrange_group (n : Nat) : Nat :=
  Nat.factorial n

/-- Calculates the number of ways to choose people for the group -/
def choose_group (n : Nat) : Nat :=
  n

/-- Calculates the number of ways to insert the group -/
def insert_group (n : Nat) : Nat :=
  n * (n - 1)

/-- The main theorem stating the total number of arrangements -/
theorem total_arrangements :
  arrange_group (total_people - group_size) *
  choose_group group_size *
  insert_group (total_people - group_size + 1) = 216 :=
sorry

end NUMINAMATH_CALUDE_total_arrangements_l1538_153831


namespace NUMINAMATH_CALUDE_pirate_coins_l1538_153819

theorem pirate_coins (x : ℚ) : 
  (3/7 * x + 0.51 * (4/7 * x) = (2.04/7) * x) →
  ((2.04/7) * x - (1.96/7) * x = 8) →
  x = 700 :=
by sorry

end NUMINAMATH_CALUDE_pirate_coins_l1538_153819


namespace NUMINAMATH_CALUDE_no_snow_probability_l1538_153868

theorem no_snow_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l1538_153868


namespace NUMINAMATH_CALUDE_complex_product_zero_l1538_153800

theorem complex_product_zero (z : ℂ) (h : z^2 + 1 = 0) :
  (z^4 + Complex.I) * (z^4 - Complex.I) = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_product_zero_l1538_153800


namespace NUMINAMATH_CALUDE_function_domain_range_nonempty_function_range_determined_single_element_domain_range_l1538_153832

-- Define a function type
def Function (α β : Type) := α → β

-- Statement 1: The domain and range of a function are both non-empty sets
theorem function_domain_range_nonempty {α β : Type} (f : Function α β) :
  Nonempty α ∧ Nonempty β :=
sorry

-- Statement 2: Once the domain and the rule of correspondence are determined,
-- the range of the function is also determined
theorem function_range_determined {α β : Type} (f g : Function α β) :
  (∀ x : α, f x = g x) → Set.range f = Set.range g :=
sorry

-- Statement 3: If there is only one element in the domain of a function,
-- then there is also only one element in its range
theorem single_element_domain_range {α β : Type} (f : Function α β) :
  (∃! x : α, True) → (∃! y : β, ∃ x : α, f x = y) :=
sorry

end NUMINAMATH_CALUDE_function_domain_range_nonempty_function_range_determined_single_element_domain_range_l1538_153832


namespace NUMINAMATH_CALUDE_opposite_numbers_properties_l1538_153844

theorem opposite_numbers_properties :
  (∀ a b : ℝ, a = -b → a + b = 0) ∧
  (∀ a b : ℝ, a + b = 0 → a = -b) ∧
  (∀ a b : ℝ, b ≠ 0 → (a / b = -1 → a = -b)) :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_properties_l1538_153844


namespace NUMINAMATH_CALUDE_thumbtack_solution_l1538_153815

/-- Represents the problem of calculating remaining thumbtacks --/
structure ThumbTackProblem where
  total_cans : Nat
  total_tacks : Nat
  boards_tested : Nat
  tacks_per_board : Nat

/-- Calculates the number of remaining thumbtacks in each can --/
def remaining_tacks (problem : ThumbTackProblem) : Nat :=
  (problem.total_tacks / problem.total_cans) - (problem.boards_tested * problem.tacks_per_board)

/-- Theorem stating the solution to the specific problem --/
theorem thumbtack_solution :
  let problem : ThumbTackProblem := {
    total_cans := 3,
    total_tacks := 450,
    boards_tested := 120,
    tacks_per_board := 1
  }
  remaining_tacks problem = 30 := by sorry


end NUMINAMATH_CALUDE_thumbtack_solution_l1538_153815


namespace NUMINAMATH_CALUDE_intersection_projection_distance_l1538_153891

/-- Given a line and a circle intersecting at two points, 
    prove that the distance between the projections of these points on the x-axis is 4. -/
theorem intersection_projection_distance (A B C D : ℝ × ℝ) : 
  -- Line equation
  (∀ (x y : ℝ), (x, y) ∈ {(x, y) | x - Real.sqrt 3 * y + 6 = 0} → 
    (A.1 - Real.sqrt 3 * A.2 + 6 = 0 ∧ B.1 - Real.sqrt 3 * B.2 + 6 = 0)) →
  -- Circle equation
  (A.1^2 + A.2^2 = 12 ∧ B.1^2 + B.2^2 = 12) →
  -- A and B are distinct points
  A ≠ B →
  -- C and D are projections of A and B on x-axis
  (C = (A.1, 0) ∧ D = (B.1, 0)) →
  -- Distance between C and D is 4
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 4 :=
by sorry


end NUMINAMATH_CALUDE_intersection_projection_distance_l1538_153891


namespace NUMINAMATH_CALUDE_special_function_property_l1538_153803

/-- A function f: ℝ → ℝ satisfying certain properties -/
structure SpecialFunction (f : ℝ → ℝ) : Prop where
  odd : ∀ x, f (-x) = -f x
  steep : ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 1

theorem special_function_property (f : ℝ → ℝ) (h : SpecialFunction f) :
  ∀ m, f m > m → m > 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l1538_153803


namespace NUMINAMATH_CALUDE_unique_solution_iff_m_eq_49_div_12_l1538_153827

/-- For a quadratic equation ax^2 + bx + c = 0 to have exactly one solution,
    its discriminant (b^2 - 4ac) must be zero. -/
def has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The quadratic equation 3x^2 - 7x + m = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  3*x^2 - 7*x + m = 0

/-- Theorem: The quadratic equation 3x^2 - 7x + m = 0 has exactly one solution
    if and only if m = 49/12 -/
theorem unique_solution_iff_m_eq_49_div_12 :
  (∃! x, quadratic_equation x m) ↔ m = 49/12 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_m_eq_49_div_12_l1538_153827


namespace NUMINAMATH_CALUDE_frac_less_one_necessary_not_sufficient_l1538_153894

theorem frac_less_one_necessary_not_sufficient (a : ℝ) :
  (∀ a, a > 1 → 1/a < 1) ∧ 
  (∃ a, 1/a < 1 ∧ ¬(a > 1)) :=
sorry

end NUMINAMATH_CALUDE_frac_less_one_necessary_not_sufficient_l1538_153894


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l1538_153886

theorem triangle_max_perimeter :
  ∀ (x : ℕ),
  x > 0 →
  x + 2*x > 15 →
  x + 15 > 2*x →
  2*x + 15 > x →
  (∀ y : ℕ, y > 0 → y + 2*y > 15 → y + 15 > 2*y → 2*y + 15 > y → x + 2*x + 15 ≥ y + 2*y + 15) →
  x + 2*x + 15 = 57 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l1538_153886


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l1538_153865

def line1 (t : ℚ) : ℚ × ℚ := (2 + 3*t, 2 - 4*t)
def line2 (u : ℚ) : ℚ × ℚ := (4 + 5*u, -8 + 3*u)

def intersection_point : ℚ × ℚ := (-123/141, 454/141)

theorem lines_intersect_at_point :
  ∃ (t u : ℚ), line1 t = line2 u ∧ line1 t = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l1538_153865


namespace NUMINAMATH_CALUDE_probability_black_ball_l1538_153852

def total_balls : ℕ := 2 + 3

def black_balls : ℕ := 2

theorem probability_black_ball :
  (black_balls : ℚ) / total_balls = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_black_ball_l1538_153852


namespace NUMINAMATH_CALUDE_sam_total_wins_l1538_153860

theorem sam_total_wins (first_period : Nat) (second_period : Nat)
  (first_win_rate : Rat) (second_win_rate : Rat) :
  first_period = 100 →
  second_period = 100 →
  first_win_rate = 1/2 →
  second_win_rate = 3/5 →
  (first_period * first_win_rate + second_period * second_win_rate : Rat) = 110 := by
  sorry

end NUMINAMATH_CALUDE_sam_total_wins_l1538_153860


namespace NUMINAMATH_CALUDE_max_distance_complex_l1538_153898

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  (⨆ (z : ℂ), Complex.abs ((1 + 2*Complex.I)*z^4 - z^6)) = 81 * (9 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_max_distance_complex_l1538_153898


namespace NUMINAMATH_CALUDE_johnson_family_seating_l1538_153830

/-- The number of ways to arrange 5 sons and 3 daughters in a row of 8 chairs
    such that at least 2 sons are next to each other -/
def seating_arrangements (num_sons : Nat) (num_daughters : Nat) : Nat :=
  Nat.factorial (num_sons + num_daughters) - 
  (Nat.factorial num_daughters * Nat.factorial num_sons)

theorem johnson_family_seating :
  seating_arrangements 5 3 = 39600 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l1538_153830


namespace NUMINAMATH_CALUDE_jenny_lasagna_profit_l1538_153837

/-- Calculates Jenny's profit from selling lasagna pans -/
def jennys_profit (cost_per_pan : ℝ) (num_pans : ℕ) (price_per_pan : ℝ) : ℝ :=
  (price_per_pan * num_pans) - (cost_per_pan * num_pans)

theorem jenny_lasagna_profit :
  let cost_per_pan : ℝ := 10
  let num_pans : ℕ := 20
  let price_per_pan : ℝ := 25
  jennys_profit cost_per_pan num_pans price_per_pan = 300 := by
  sorry

end NUMINAMATH_CALUDE_jenny_lasagna_profit_l1538_153837


namespace NUMINAMATH_CALUDE_snack_machine_quarters_l1538_153867

def candy_bar_cost : ℕ := 25
def chocolate_cost : ℕ := 75
def juice_cost : ℕ := 50
def quarter_value : ℕ := 25

def total_cost (candy_bars chocolate juice : ℕ) : ℕ :=
  candy_bars * candy_bar_cost + chocolate * chocolate_cost + juice * juice_cost

def quarters_needed (total : ℕ) : ℕ :=
  (total + quarter_value - 1) / quarter_value

theorem snack_machine_quarters : quarters_needed (total_cost 3 2 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_snack_machine_quarters_l1538_153867


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1538_153833

theorem arithmetic_expression_equality : 5 * 7 - 6 * 8 + 9 * 2 + 7 * 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1538_153833


namespace NUMINAMATH_CALUDE_sam_travel_distance_l1538_153875

/-- Given that Marguerite drove 150 miles in 3 hours, and Sam increased his speed by 20% and drove for 4 hours, prove that Sam traveled 240 miles. -/
theorem sam_travel_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) 
  (sam_speed_increase : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 →
  marguerite_time = 3 →
  sam_speed_increase = 0.2 →
  sam_time = 4 →
  (marguerite_distance / marguerite_time) * (1 + sam_speed_increase) * sam_time = 240 := by
  sorry

end NUMINAMATH_CALUDE_sam_travel_distance_l1538_153875


namespace NUMINAMATH_CALUDE_mixed_committee_probability_l1538_153873

def total_members : ℕ := 30
def num_boys : ℕ := 13
def num_girls : ℕ := 17
def committee_size : ℕ := 6

def probability_mixed_committee : ℚ :=
  1 - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size) / Nat.choose total_members committee_size

theorem mixed_committee_probability :
  probability_mixed_committee = 579683 / 593775 :=
by sorry

end NUMINAMATH_CALUDE_mixed_committee_probability_l1538_153873


namespace NUMINAMATH_CALUDE_sum_of_digits_R50_div_R8_l1538_153811

def R (k : ℕ) : ℕ := (10^k - 1) / 9

theorem sum_of_digits_R50_div_R8 : ∃ (q : ℕ), R 50 = q * R 8 ∧ (q.digits 10).sum = 6 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_R50_div_R8_l1538_153811


namespace NUMINAMATH_CALUDE_sum_of_powers_l1538_153855

theorem sum_of_powers (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) :
  x^2 / (x - 1) + x^4 / (x^2 - 1) + x^6 / (x^3 - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1538_153855


namespace NUMINAMATH_CALUDE_cube_minus_square_equals_zero_l1538_153850

theorem cube_minus_square_equals_zero : 4^3 - 8^2 = 0 :=
by
  -- Given conditions (not used in the proof, but included for completeness)
  have h1 : 2^3 - 7^2 = 1 := by sorry
  have h2 : 3^3 - 6^2 = 9 := by sorry
  have h3 : 5^3 - 9^2 = 16 := by sorry
  
  -- Proof
  sorry

end NUMINAMATH_CALUDE_cube_minus_square_equals_zero_l1538_153850


namespace NUMINAMATH_CALUDE_range_of_a_l1538_153818

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → x^2 + 2*x - a > 0) → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1538_153818


namespace NUMINAMATH_CALUDE_vendor_apples_thrown_away_l1538_153820

/-- Calculates the percentage of apples thrown away given the initial quantity and selling/discarding percentages --/
def apples_thrown_away (initial_quantity : ℕ) (sell_day1 sell_day2 discard_day1 : ℚ) : ℚ :=
  let remaining_after_sell1 := initial_quantity * (1 - sell_day1)
  let discarded_day1 := remaining_after_sell1 * discard_day1
  let remaining_after_discard1 := remaining_after_sell1 - discarded_day1
  let remaining_after_sell2 := remaining_after_discard1 * (1 - sell_day2)
  (discarded_day1 + remaining_after_sell2) / initial_quantity * 100

theorem vendor_apples_thrown_away :
  apples_thrown_away 100 (30/100) (50/100) (20/100) = 42 :=
by sorry

end NUMINAMATH_CALUDE_vendor_apples_thrown_away_l1538_153820


namespace NUMINAMATH_CALUDE_exponent_division_l1538_153821

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1538_153821


namespace NUMINAMATH_CALUDE_merchant_profit_l1538_153805

theorem merchant_profit (cost_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  markup_percent = 40 →
  discount_percent = 10 →
  let marked_price := cost_price * (1 + markup_percent / 100)
  let selling_price := marked_price * (1 - discount_percent / 100)
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  profit_percent = 26 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_l1538_153805


namespace NUMINAMATH_CALUDE_james_weekly_pistachio_expense_l1538_153882

/-- Represents the cost of pistachios in dollars per can. -/
def cost_per_can : ℝ := 10

/-- Represents the amount of pistachios in ounces per can. -/
def ounces_per_can : ℝ := 5

/-- Represents the amount of pistachios James eats in ounces every 5 days. -/
def ounces_per_five_days : ℝ := 30

/-- Represents the number of days in a week. -/
def days_in_week : ℝ := 7

/-- Proves that James spends $84 per week on pistachios. -/
theorem james_weekly_pistachio_expense : 
  (cost_per_can / ounces_per_can) * (ounces_per_five_days / 5) * days_in_week = 84 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_pistachio_expense_l1538_153882


namespace NUMINAMATH_CALUDE_p_iff_q_l1538_153806

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a*y - 2 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, f x₁ y₁ ∧ f x₂ y₂ ∧ g x₁ y₁ ∧ g x₂ y₂ →
    (y₂ - y₁) / (x₂ - x₁) = (y₂ - y₁) / (x₂ - x₁)

-- Define the propositions p and q
def p (a : ℝ) : Prop := parallel (l₁) (l₂ a)
def q (a : ℝ) : Prop := a = -1

-- State the theorem
theorem p_iff_q : ∀ a : ℝ, p a ↔ q a := by sorry

end NUMINAMATH_CALUDE_p_iff_q_l1538_153806


namespace NUMINAMATH_CALUDE_log_inequality_equiv_range_l1538_153890

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_inequality_equiv_range (x : ℝ) :
  lg (x + 1) < lg (3 - x) ↔ -1 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_equiv_range_l1538_153890


namespace NUMINAMATH_CALUDE_max_attendance_difference_l1538_153840

-- Define the estimates and error margins
def chloe_estimate : ℝ := 40000
def derek_estimate : ℝ := 55000
def emma_estimate : ℝ := 75000

def chloe_error : ℝ := 0.05
def derek_error : ℝ := 0.15
def emma_error : ℝ := 0.10

-- Define the ranges for actual attendances
def chicago_range : Set ℝ := {x | chloe_estimate * (1 - chloe_error) ≤ x ∧ x ≤ chloe_estimate * (1 + chloe_error)}
def denver_range : Set ℝ := {x | derek_estimate / (1 + derek_error) ≤ x ∧ x ≤ derek_estimate / (1 - derek_error)}
def miami_range : Set ℝ := {x | emma_estimate * (1 - emma_error) ≤ x ∧ x ≤ emma_estimate * (1 + emma_error)}

-- Define the theorem
theorem max_attendance_difference :
  ∃ (c d m : ℝ),
    c ∈ chicago_range ∧
    d ∈ denver_range ∧
    m ∈ miami_range ∧
    (⌊(max c (max d m) - min c (min d m) + 500) / 1000⌋ * 1000 = 45000) :=
sorry

end NUMINAMATH_CALUDE_max_attendance_difference_l1538_153840


namespace NUMINAMATH_CALUDE_completing_square_equiv_l1538_153880

theorem completing_square_equiv (x : ℝ) : 
  x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 := by sorry

end NUMINAMATH_CALUDE_completing_square_equiv_l1538_153880


namespace NUMINAMATH_CALUDE_equal_sides_from_tangent_sum_l1538_153862

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively
  (sum_angles : A + B + C = π)  -- Sum of angles in a triangle is π
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)  -- Sides are positive

-- State the theorem
theorem equal_sides_from_tangent_sum (t : Triangle) :
  t.a * Real.tan t.A + t.b * Real.tan t.B = (t.a + t.b) * Real.tan ((t.A + t.B) / 2) →
  t.a = t.b :=
by sorry

end NUMINAMATH_CALUDE_equal_sides_from_tangent_sum_l1538_153862


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1538_153807

theorem quadratic_equal_roots (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 = 0 ∧ (∀ y : ℝ, y^2 - a*y + 1 = 0 → y = x)) →
  a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1538_153807


namespace NUMINAMATH_CALUDE_base3_addition_l1538_153859

-- Define a type for base-3 numbers
def Base3 := ℕ

-- Function to convert a base-3 number to its decimal representation
def to_decimal (n : Base3) : ℕ := sorry

-- Function to convert a decimal number to its base-3 representation
def to_base3 (n : ℕ) : Base3 := sorry

-- Define the given numbers in base 3
def a : Base3 := to_base3 1
def b : Base3 := to_base3 22
def c : Base3 := to_base3 212
def d : Base3 := to_base3 1001

-- Define the result in base 3
def result : Base3 := to_base3 210

-- Theorem statement
theorem base3_addition :
  to_decimal a - to_decimal b + to_decimal c - to_decimal d = to_decimal result := by
  sorry

end NUMINAMATH_CALUDE_base3_addition_l1538_153859


namespace NUMINAMATH_CALUDE_rolling_cube_dot_path_length_l1538_153824

/-- The path length of a dot on a rolling cube -/
theorem rolling_cube_dot_path_length :
  let cube_side : ℝ := 2
  let dot_distance : ℝ := 2 / 3
  let path_length : ℝ := (4 * Real.pi * Real.sqrt 10) / 3
  cube_side > 0 ∧ 0 < dot_distance ∧ dot_distance < cube_side →
  path_length = 4 * (Real.pi * Real.sqrt (dot_distance^2 + cube_side^2)) / 2 :=
by sorry


end NUMINAMATH_CALUDE_rolling_cube_dot_path_length_l1538_153824


namespace NUMINAMATH_CALUDE_correct_operations_l1538_153877

theorem correct_operations (x y : ℝ) (h : x ≠ y) :
  ((-3 * x * y) ^ 2 = 9 * x^2 * y^2) ∧
  ((x - y) / (2 * x * y - x^2 - y^2) = 1 / (y - x)) := by
  sorry

end NUMINAMATH_CALUDE_correct_operations_l1538_153877


namespace NUMINAMATH_CALUDE_probability_at_least_two_correct_l1538_153810

/-- The probability of getting at least two correct answers out of five questions
    with four choices each, when guessing randomly. -/
theorem probability_at_least_two_correct : ℝ := by
  -- Define the number of questions and choices
  let n : ℕ := 5
  let choices : ℕ := 4

  -- Define the probability of a correct guess
  let p : ℝ := 1 / choices

  -- Define the binomial probability function
  let binomial_prob (k : ℕ) : ℝ := (n.choose k) * p^k * (1 - p)^(n - k)

  -- Calculate the probability of getting 0 or 1 correct
  let prob_zero_or_one : ℝ := binomial_prob 0 + binomial_prob 1

  -- The probability of at least two correct is 1 minus the probability of 0 or 1 correct
  let prob_at_least_two : ℝ := 1 - prob_zero_or_one

  -- Prove that this probability is equal to 47/128
  sorry

#eval (47 : ℚ) / 128

end NUMINAMATH_CALUDE_probability_at_least_two_correct_l1538_153810


namespace NUMINAMATH_CALUDE_technicians_count_l1538_153895

/-- Represents the workshop scenario with workers and salaries -/
structure Workshop where
  total_workers : ℕ
  avg_salary : ℚ
  technician_salary : ℚ
  other_salary : ℚ

/-- Calculates the number of technicians in the workshop -/
def num_technicians (w : Workshop) : ℚ :=
  ((w.avg_salary - w.other_salary) * w.total_workers) / (w.technician_salary - w.other_salary)

/-- The given workshop scenario -/
def given_workshop : Workshop :=
  { total_workers := 22
    avg_salary := 850
    technician_salary := 1000
    other_salary := 780 }

/-- Theorem stating that the number of technicians in the given workshop is 7 -/
theorem technicians_count :
  num_technicians given_workshop = 7 := by
  sorry


end NUMINAMATH_CALUDE_technicians_count_l1538_153895


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l1538_153864

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∃ (r s : ℤ), x^2 + b*x + 2008 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬∃ (r s : ℤ), x^2 + b'*x + 2008 = (x + r) * (x + s)) ∧
  b = 259 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l1538_153864


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1538_153826

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specific speeds, 
    the man's speed against the current is 8.6 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 15 3.2 = 8.6 := by
  sorry


end NUMINAMATH_CALUDE_mans_speed_against_current_l1538_153826


namespace NUMINAMATH_CALUDE_count_integers_satisfying_equation_l1538_153879

def count_satisfying_integers (lower upper : ℕ) : ℕ :=
  (upper - lower + 1) / 4 + 1

theorem count_integers_satisfying_equation : 
  count_satisfying_integers 1 2002 = 501 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_equation_l1538_153879


namespace NUMINAMATH_CALUDE_geometric_sequence_quadratic_one_root_l1538_153878

/-- If real numbers a, b, c form a geometric sequence, then the function f(x) = ax^2 + 2bx + c has exactly one real root. -/
theorem geometric_sequence_quadratic_one_root
  (a b c : ℝ) (h_geometric : b^2 = a*c) :
  ∃! x, a*x^2 + 2*b*x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_quadratic_one_root_l1538_153878


namespace NUMINAMATH_CALUDE_intersection_M_N_l1538_153863

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1538_153863


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_theorem_l1538_153893

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

theorem line_plane_perpendicular_theorem 
  (a b : Line) (α : Plane) :
  perpendicular_lines a b → 
  perpendicular_line_plane a α → 
  parallel_line_plane b α ∨ subset_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_theorem_l1538_153893


namespace NUMINAMATH_CALUDE_beta_values_l1538_153866

theorem beta_values (β : ℂ) (h1 : β ≠ 1) 
  (h2 : Complex.abs (β^3 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^6 - 1) = 6 * Complex.abs (β - 1)) :
  β = Complex.I * 2 * Real.sqrt 2 ∨ β = Complex.I * (-2) * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_beta_values_l1538_153866


namespace NUMINAMATH_CALUDE_sarah_toad_count_l1538_153896

/-- The number of toads each person has -/
structure ToadCount where
  tim : ℕ
  jim : ℕ
  sarah : ℕ

/-- Given conditions about toad counts -/
def toad_conditions (tc : ToadCount) : Prop :=
  tc.tim = 30 ∧ 
  tc.jim = tc.tim + 20 ∧ 
  tc.sarah = 2 * tc.jim

/-- Theorem stating Sarah has 100 toads under given conditions -/
theorem sarah_toad_count (tc : ToadCount) (h : toad_conditions tc) : tc.sarah = 100 := by
  sorry

end NUMINAMATH_CALUDE_sarah_toad_count_l1538_153896


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l1538_153829

theorem largest_n_satisfying_conditions : ∃ (n : ℕ), n = 313 ∧ 
  (∃ (m : ℤ), n^2 = (m+1)^3 - m^3) ∧ 
  (∃ (a : ℕ), 5*n + 103 = a^2) ∧
  (∀ (k : ℕ), k > n → ¬(∃ (m : ℤ), k^2 = (m+1)^3 - m^3) ∨ ¬(∃ (a : ℕ), 5*k + 103 = a^2)) :=
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l1538_153829


namespace NUMINAMATH_CALUDE_complex_power_difference_abs_l1538_153876

def i : ℂ := Complex.I

theorem complex_power_difference_abs : 
  Complex.abs ((2 + i)^18 - (2 - i)^18) = 19531250 := by sorry

end NUMINAMATH_CALUDE_complex_power_difference_abs_l1538_153876


namespace NUMINAMATH_CALUDE_triangle_cosine_relation_l1538_153848

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, if S + a² = (b + c)², then cos A = -15/17 -/
theorem triangle_cosine_relation (a b c S : ℝ) (A : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < S →  -- positive area
  S = (1/2) * b * c * Real.sin A →  -- area formula
  a^2 + b^2 - 2 * a * b * Real.cos A = c^2 →  -- cosine law
  S + a^2 = (b + c)^2 →  -- given condition
  Real.cos A = -15/17 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_relation_l1538_153848


namespace NUMINAMATH_CALUDE_age_difference_proof_l1538_153838

/-- Represents the age difference between Petra's mother and twice Petra's age --/
def age_difference (petra_age : ℕ) (mother_age : ℕ) : ℕ :=
  mother_age - 2 * petra_age

/-- Theorem stating the age difference between Petra's mother and twice Petra's age --/
theorem age_difference_proof :
  let petra_age : ℕ := 11
  let mother_age : ℕ := 36
  age_difference petra_age mother_age = 14 ∧
  petra_age + mother_age = 47 ∧
  ∃ (n : ℕ), mother_age = 2 * petra_age + n :=
by sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1538_153838


namespace NUMINAMATH_CALUDE_greatest_rational_root_of_quadratic_l1538_153813

theorem greatest_rational_root_of_quadratic (a b c : ℕ) 
  (ha : a ≤ 100) (hb : b ≤ 100) (hc : c ≤ 100) (ha_pos : a > 0) :
  ∃ (x : ℚ), x = -1/99 ∧ 
    (∀ (y : ℚ), y ≠ x → a * y^2 + b * y + c = 0 → y < x) ∧
    a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_rational_root_of_quadratic_l1538_153813


namespace NUMINAMATH_CALUDE_decimal_51_to_binary_l1538_153871

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Checks if a list of booleans represents the binary form of a given natural number -/
def is_binary_of (bits : List Bool) (n : ℕ) : Prop :=
  to_binary n = bits.reverse

theorem decimal_51_to_binary :
  is_binary_of [true, true, false, false, true, true] 51 := by
  sorry

end NUMINAMATH_CALUDE_decimal_51_to_binary_l1538_153871


namespace NUMINAMATH_CALUDE_twenty_percent_greater_than_88_l1538_153899

theorem twenty_percent_greater_than_88 (x : ℝ) : x = 88 * 1.2 → x = 105.6 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_greater_than_88_l1538_153899
