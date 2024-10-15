import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1163_116397

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 2 * a 3 = 5 →
  a 7 * a 8 * a 9 = 10 →
  a 4 * a 5 * a 6 = 10 / (a 1)^3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1163_116397


namespace NUMINAMATH_CALUDE_vector_equality_sufficient_not_necessary_l1163_116334

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_equality_sufficient_not_necessary :
  (∀ a b : V, a = b → (‖a‖ = ‖b‖ ∧ parallel a b)) ∧
  (∃ a b : V, ‖a‖ = ‖b‖ ∧ parallel a b ∧ a ≠ b) :=
sorry

end NUMINAMATH_CALUDE_vector_equality_sufficient_not_necessary_l1163_116334


namespace NUMINAMATH_CALUDE_total_toys_l1163_116369

/-- The number of toys each person has -/
structure ToyCount where
  jaxon : ℕ
  gabriel : ℕ
  jerry : ℕ

/-- The conditions of the problem -/
def toy_conditions (t : ToyCount) : Prop :=
  t.jaxon = 15 ∧ 
  t.gabriel = 2 * t.jaxon ∧ 
  t.jerry = t.gabriel + 8

/-- The theorem stating the total number of toys -/
theorem total_toys (t : ToyCount) (h : toy_conditions t) : 
  t.jaxon + t.gabriel + t.jerry = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_l1163_116369


namespace NUMINAMATH_CALUDE_selection_methods_eq_51_l1163_116363

/-- The number of ways to select k elements from n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 9

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of specific students (A, B, C) -/
def specific_students : ℕ := 3

/-- The number of ways to select 4 students from 9, where at least two of three specific students must be selected -/
def selection_methods : ℕ :=
  choose specific_students 2 * choose (total_students - specific_students) (selected_students - 2) +
  choose specific_students 3 * choose (total_students - specific_students) (selected_students - 3)

theorem selection_methods_eq_51 : selection_methods = 51 := by sorry

end NUMINAMATH_CALUDE_selection_methods_eq_51_l1163_116363


namespace NUMINAMATH_CALUDE_percentage_problem_l1163_116326

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  x = 780 →
  (25 / 100) * x = (p / 100) * 1500 - 30 →
  p = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1163_116326


namespace NUMINAMATH_CALUDE_garden_area_not_covered_by_flower_beds_l1163_116393

def garden_side_length : ℝ := 16
def flower_bed_radius : ℝ := 8

theorem garden_area_not_covered_by_flower_beds :
  let total_area := garden_side_length ^ 2
  let flower_bed_area := 4 * (π * flower_bed_radius ^ 2) / 4
  total_area - flower_bed_area = 256 - 64 * π := by sorry

end NUMINAMATH_CALUDE_garden_area_not_covered_by_flower_beds_l1163_116393


namespace NUMINAMATH_CALUDE_representative_selection_count_l1163_116386

def male_count : ℕ := 5
def female_count : ℕ := 4
def total_representatives : ℕ := 4
def min_male : ℕ := 2
def min_female : ℕ := 1

theorem representative_selection_count : 
  (Nat.choose male_count 2 * Nat.choose female_count 2) + 
  (Nat.choose male_count 3 * Nat.choose female_count 1) = 100 := by
  sorry

end NUMINAMATH_CALUDE_representative_selection_count_l1163_116386


namespace NUMINAMATH_CALUDE_songs_storable_jeff_l1163_116364

/-- Calculates the number of songs that can be stored on a phone given the total storage, used storage, and size of each song. -/
def songs_storable (total_storage : ℕ) (used_storage : ℕ) (song_size : ℕ) : ℕ :=
  ((total_storage - used_storage) * 1000) / song_size

/-- Theorem stating that given the specific conditions, 400 songs can be stored. -/
theorem songs_storable_jeff : songs_storable 16 4 30 = 400 := by
  sorry

#eval songs_storable 16 4 30

end NUMINAMATH_CALUDE_songs_storable_jeff_l1163_116364


namespace NUMINAMATH_CALUDE_carol_trivia_score_l1163_116339

/-- Represents Carol's trivia game scores -/
structure TriviaGame where
  round1 : Int
  round2 : Int
  round3 : Int

/-- Calculates the total score of a trivia game -/
def totalScore (game : TriviaGame) : Int :=
  game.round1 + game.round2 + game.round3

/-- Theorem: Carol's total score at the end of the game is 7 points -/
theorem carol_trivia_score :
  ∃ (game : TriviaGame), game.round1 = 17 ∧ game.round2 = 6 ∧ game.round3 = -16 ∧ totalScore game = 7 := by
  sorry

end NUMINAMATH_CALUDE_carol_trivia_score_l1163_116339


namespace NUMINAMATH_CALUDE_not_always_geometric_sequence_l1163_116313

def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ+, a (n + 1) = q * a n

theorem not_always_geometric_sequence :
  ¬ (∀ a : ℕ+ → ℝ, (∃ q : ℝ, ∀ n : ℕ+, a (n + 1) = q * a n) → is_geometric_sequence a) :=
by
  sorry

end NUMINAMATH_CALUDE_not_always_geometric_sequence_l1163_116313


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1163_116395

/-- Given a line L1 with equation x - 2y + 1 = 0, prove that the line L2 with equation 2x + y + 1 = 0
    passes through the point (-2, 3) and is perpendicular to L1. -/
theorem perpendicular_line_through_point :
  let L1 : ℝ → ℝ → Prop := λ x y => x - 2*y + 1 = 0
  let L2 : ℝ → ℝ → Prop := λ x y => 2*x + y + 1 = 0
  let point : ℝ × ℝ := (-2, 3)
  (L2 point.1 point.2) ∧
  (∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L1 x2 y2 → L2 x1 y1 → L2 x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) *
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) =
    ((x2 - x1) * (y2 - y1) - (y2 - y1) * (x2 - x1)) *
    ((x2 - x1) * (y2 - y1) - (y2 - y1) * (x2 - x1))) :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1163_116395


namespace NUMINAMATH_CALUDE_g_increasing_iff_a_in_range_l1163_116389

-- Define the piecewise function g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ -1 then -a / (x - 1) else (3 - 3*a) * x + 1

-- State the theorem
theorem g_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → g a x < g a y) ↔ (4/5 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_g_increasing_iff_a_in_range_l1163_116389


namespace NUMINAMATH_CALUDE_unique_remainder_modulo_8_and_13_l1163_116307

theorem unique_remainder_modulo_8_and_13 : 
  ∃! n : ℕ, n < 180 ∧ n % 8 = 2 ∧ n % 13 = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_remainder_modulo_8_and_13_l1163_116307


namespace NUMINAMATH_CALUDE_bookshelf_selections_l1163_116335

/-- Represents a bookshelf with three layers -/
structure Bookshelf :=
  (layer1 : ℕ)
  (layer2 : ℕ)
  (layer3 : ℕ)

/-- The total number of books in the bookshelf -/
def total_books (b : Bookshelf) : ℕ :=
  b.layer1 + b.layer2 + b.layer3

/-- The number of ways to select one book from each layer -/
def ways_to_select_from_each_layer (b : Bookshelf) : ℕ :=
  b.layer1 * b.layer2 * b.layer3

/-- Our specific bookshelf instance -/
def our_bookshelf : Bookshelf :=
  ⟨6, 5, 4⟩

theorem bookshelf_selections (b : Bookshelf) :
  (total_books b = 15) ∧
  (ways_to_select_from_each_layer b = 120) :=
sorry

end NUMINAMATH_CALUDE_bookshelf_selections_l1163_116335


namespace NUMINAMATH_CALUDE_max_k_for_no_real_roots_l1163_116321

theorem max_k_for_no_real_roots (k : ℤ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_no_real_roots_l1163_116321


namespace NUMINAMATH_CALUDE_lillian_candy_count_l1163_116383

/-- The number of candies Lillian has after receiving candies from her father -/
def total_candies (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Lillian has 93 candies after receiving candies from her father -/
theorem lillian_candy_count :
  total_candies 88 5 = 93 := by
  sorry

end NUMINAMATH_CALUDE_lillian_candy_count_l1163_116383


namespace NUMINAMATH_CALUDE_playstation_value_l1163_116327

theorem playstation_value (computer_cost accessories_cost out_of_pocket : ℝ) 
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : out_of_pocket = 580) : 
  ∃ (playstation_value : ℝ), 
    playstation_value = 400 ∧ 
    computer_cost + accessories_cost = out_of_pocket + playstation_value * 0.8 := by
  sorry

end NUMINAMATH_CALUDE_playstation_value_l1163_116327


namespace NUMINAMATH_CALUDE_custom_mult_four_three_l1163_116396

/-- Custom multiplication operation -/
def customMult (x y : ℝ) : ℝ := x^2 - x*y + y^2

/-- Theorem stating that 4 * 3 = 13 under the custom multiplication -/
theorem custom_mult_four_three : customMult 4 3 = 13 := by sorry

end NUMINAMATH_CALUDE_custom_mult_four_three_l1163_116396


namespace NUMINAMATH_CALUDE_point_on_graph_l1163_116300

def f (x : ℝ) : ℝ := x + 1

theorem point_on_graph :
  f 0 = 1 ∧ 
  f 1 ≠ 1 ∧
  f 2 ≠ 0 ∧
  f (-1) ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_graph_l1163_116300


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1163_116357

def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {1, 2}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1163_116357


namespace NUMINAMATH_CALUDE_greatest_number_of_bouquets_l1163_116356

/-- Represents the number of tulips of each color --/
structure TulipCount where
  white : Nat
  red : Nat
  blue : Nat
  yellow : Nat

/-- Represents the ratio of tulips in each bouquet --/
structure BouquetRatio where
  white : Nat
  red : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of bouquets that can be made with given tulips and ratio --/
def calculateBouquets (tulips : TulipCount) (ratio : BouquetRatio) : Nat :=
  min (tulips.white / ratio.white)
      (min (tulips.red / ratio.red)
           (min (tulips.blue / ratio.blue)
                (tulips.yellow / ratio.yellow)))

/-- Calculates the total number of flowers in a bouquet --/
def flowersPerBouquet (ratio : BouquetRatio) : Nat :=
  ratio.white + ratio.red + ratio.blue + ratio.yellow

theorem greatest_number_of_bouquets
  (tulips : TulipCount)
  (ratio : BouquetRatio)
  (h1 : tulips = ⟨21, 91, 37, 67⟩)
  (h2 : ratio = ⟨3, 7, 5, 9⟩)
  (h3 : flowersPerBouquet ratio ≥ 24)
  (h4 : flowersPerBouquet ratio ≤ 50)
  : calculateBouquets tulips ratio = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_of_bouquets_l1163_116356


namespace NUMINAMATH_CALUDE_condition_2_not_implies_right_triangle_l1163_116380

/-- A triangle ABC --/
structure Triangle :=
  (A B C : ℝ)
  (sum_angles : A + B + C = 180)

/-- Definition of a right triangle --/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

/-- The condition ∠A = ∠B - ∠C --/
def condition_2 (t : Triangle) : Prop :=
  t.A = t.B - t.C

/-- Theorem: The condition ∠A = ∠B - ∠C does not necessarily imply a right triangle --/
theorem condition_2_not_implies_right_triangle :
  ∃ t : Triangle, condition_2 t ∧ ¬is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_condition_2_not_implies_right_triangle_l1163_116380


namespace NUMINAMATH_CALUDE_gcd_272_595_l1163_116387

theorem gcd_272_595 : Nat.gcd 272 595 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_272_595_l1163_116387


namespace NUMINAMATH_CALUDE_jesses_room_difference_l1163_116399

/-- Jesse's room dimensions and length-width difference --/
theorem jesses_room_difference (length width : ℕ) (h1 : length = 20) (h2 : width = 19) :
  length - width = 1 := by
  sorry

end NUMINAMATH_CALUDE_jesses_room_difference_l1163_116399


namespace NUMINAMATH_CALUDE_always_odd_l1163_116373

theorem always_odd (p m : ℤ) (h : Odd p) : Odd (p^2 + 2*m*p) := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l1163_116373


namespace NUMINAMATH_CALUDE_students_taking_german_german_count_l1163_116348

theorem students_taking_german (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  let students_taking_language := total - neither
  let only_french := french - both
  let only_german := students_taking_language - only_french - both
  only_german + both
  
theorem german_count :
  students_taking_german 94 41 9 40 = 22 := by sorry

end NUMINAMATH_CALUDE_students_taking_german_german_count_l1163_116348


namespace NUMINAMATH_CALUDE_deductive_reasoning_validity_l1163_116354

/-- Represents a deductive reasoning argument -/
structure DeductiveArgument where
  major_premise : Prop
  minor_premise : Prop
  form_of_reasoning : Prop
  conclusion : Prop

/-- States that if all components of a deductive argument are correct, the conclusion must be correct -/
theorem deductive_reasoning_validity (arg : DeductiveArgument) :
  arg.major_premise → arg.minor_premise → arg.form_of_reasoning → arg.conclusion :=
by sorry

end NUMINAMATH_CALUDE_deductive_reasoning_validity_l1163_116354


namespace NUMINAMATH_CALUDE_percentage_decrease_l1163_116379

theorem percentage_decrease (original : ℝ) (increase_percent : ℝ) (difference : ℝ) :
  original = 80 →
  increase_percent = 12.5 →
  difference = 30 →
  let increased_value := original * (1 + increase_percent / 100)
  let decrease_percent := (increased_value - original - difference) / original * 100
  decrease_percent = 25 := by sorry

end NUMINAMATH_CALUDE_percentage_decrease_l1163_116379


namespace NUMINAMATH_CALUDE_initial_plus_rainfall_equals_final_l1163_116351

/-- Represents the rainfall data for a single day -/
structure RainfallData where
  rate1 : Real  -- rainfall rate from 2pm to 4pm in inches per hour
  duration1 : Real  -- duration of rainfall from 2pm to 4pm in hours
  rate2 : Real  -- rainfall rate from 4pm to 7pm in inches per hour
  duration2 : Real  -- duration of rainfall from 4pm to 7pm in hours
  rate3 : Real  -- rainfall rate from 7pm to 9pm in inches per hour
  duration3 : Real  -- duration of rainfall from 7pm to 9pm in hours
  final_amount : Real  -- amount of water in the gauge at 9pm in inches

/-- Calculates the total rainfall during the day -/
def total_rainfall (data : RainfallData) : Real :=
  data.rate1 * data.duration1 + data.rate2 * data.duration2 + data.rate3 * data.duration3

/-- Theorem stating that the initial amount plus total rainfall equals the final amount -/
theorem initial_plus_rainfall_equals_final (data : RainfallData) 
    (h1 : data.rate1 = 4) (h2 : data.duration1 = 2)
    (h3 : data.rate2 = 3) (h4 : data.duration2 = 3)
    (h5 : data.rate3 = 0.5) (h6 : data.duration3 = 2)
    (h7 : data.final_amount = 20) :
    ∃ initial_amount : Real, initial_amount + total_rainfall data = data.final_amount := by
  sorry

end NUMINAMATH_CALUDE_initial_plus_rainfall_equals_final_l1163_116351


namespace NUMINAMATH_CALUDE_nina_money_problem_l1163_116349

theorem nina_money_problem (W : ℝ) (M : ℝ) :
  (10 * W = M) →
  (14 * (W - 1.75) = M) →
  M = 61.25 := by
sorry

end NUMINAMATH_CALUDE_nina_money_problem_l1163_116349


namespace NUMINAMATH_CALUDE_proportion_equality_false_l1163_116391

theorem proportion_equality_false : 
  ¬(∀ (A B C : ℚ), (A / B = C / 4 ∧ A = 4) → B = C) :=
by sorry

end NUMINAMATH_CALUDE_proportion_equality_false_l1163_116391


namespace NUMINAMATH_CALUDE_equal_expressions_condition_l1163_116309

theorem equal_expressions_condition (a b c : ℝ) :
  a + b + c = (a + b) * (a + c) ↔ a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_condition_l1163_116309


namespace NUMINAMATH_CALUDE_slope_of_line_l1163_116371

theorem slope_of_line (x y : ℝ) :
  3 * y = 4 * x - 9 → (∃ m b : ℝ, y = m * x + b ∧ m = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l1163_116371


namespace NUMINAMATH_CALUDE_cube_root_approximation_l1163_116302

-- Define k as the real cube root of 2
noncomputable def k : ℝ := Real.rpow 2 (1/3)

-- Define the inequality function
def inequality (A B C a b c : ℤ) (x : ℚ) : Prop :=
  x ≥ 0 → |((A * x^2 + B * x + C) / (a * x^2 + b * x + c) : ℝ) - k| < |x - k|

-- State the theorem
theorem cube_root_approximation :
  ∀ x : ℚ, inequality 2 2 2 1 2 2 x := by sorry

end NUMINAMATH_CALUDE_cube_root_approximation_l1163_116302


namespace NUMINAMATH_CALUDE_platform_length_l1163_116325

/-- Given a train with speed 72 km/hr and length 220 m, crossing a platform in 26 seconds,
    the length of the platform is 300 m. -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 →
  train_length = 220 →
  crossing_time = 26 →
  (train_speed * (5/18) * crossing_time) - train_length = 300 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l1163_116325


namespace NUMINAMATH_CALUDE_uncle_james_height_difference_l1163_116365

theorem uncle_james_height_difference :
  ∀ (james_original_height james_new_height uncle_height : ℝ),
  uncle_height = 72 →
  james_original_height = (2/3) * uncle_height →
  james_new_height = james_original_height + 10 →
  uncle_height - james_new_height = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_uncle_james_height_difference_l1163_116365


namespace NUMINAMATH_CALUDE_det_special_matrix_l1163_116370

def matrix (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![x + 2, x, x;
     x, x + 2, x;
     x, x, x + 2]

theorem det_special_matrix (x : ℝ) :
  Matrix.det (matrix x) = 8 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l1163_116370


namespace NUMINAMATH_CALUDE_dollar_sum_squared_zero_l1163_116323

/-- The dollar operation for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem: For real numbers x and y, (x + y)²$(y + x)² = 0 -/
theorem dollar_sum_squared_zero (x y : ℝ) : dollar ((x + y)^2) ((y + x)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dollar_sum_squared_zero_l1163_116323


namespace NUMINAMATH_CALUDE_eight_monkeys_eat_fortyeight_bananas_l1163_116328

/-- Given a rate at which monkeys eat bananas, calculate the number of monkeys needed to eat a certain number of bananas -/
def monkeys_needed (initial_monkeys initial_bananas target_bananas : ℕ) : ℕ :=
  initial_monkeys

/-- Theorem: Given that 8 monkeys take 8 minutes to eat 8 bananas, 8 monkeys are needed to eat 48 bananas -/
theorem eight_monkeys_eat_fortyeight_bananas :
  monkeys_needed 8 8 48 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_monkeys_eat_fortyeight_bananas_l1163_116328


namespace NUMINAMATH_CALUDE_semicircle_area_shaded_area_calculation_l1163_116336

theorem semicircle_area (r : ℝ) (h : r = 2.5) : 
  (π * r^2) / 2 = 3.125 * π := by
  sorry

/- Definitions based on problem conditions -/
def semicircle_ADB_radius : ℝ := 2
def semicircle_BEC_radius : ℝ := 1
def point_D : ℝ × ℝ := (1, 2)  -- midpoint of arc ADB
def point_E : ℝ × ℝ := (3, 1)  -- midpoint of arc BEC
def point_F : ℝ × ℝ := (3, 2.5)  -- midpoint of arc DFE

/- Main theorem -/
theorem shaded_area_calculation : 
  let r : ℝ := semicircle_ADB_radius + semicircle_BEC_radius / 2
  (π * r^2) / 2 = 3.125 * π := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_shaded_area_calculation_l1163_116336


namespace NUMINAMATH_CALUDE_area_triangle_ABC_l1163_116319

/-- Linear function f(x) = ax + b -/
def linear_function (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

/-- The y-intercept of a linear function f(x) = ax + b -/
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

theorem area_triangle_ABC : 
  ∀ (m n : ℝ),
  let f := linear_function (3/2) m
  let g := linear_function (-1/2) n
  (f (-4) = 0) →
  (g (-4) = 0) →
  let B := (0, y_intercept f)
  let C := (0, y_intercept g)
  let A := (-4, 0)
  (1/2 * |A.1| * |B.2 - C.2| = 16) :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_ABC_l1163_116319


namespace NUMINAMATH_CALUDE_odd_function_sum_zero_l1163_116358

/-- A function v is odd if v(-x) = -v(x) for all x in its domain -/
def IsOdd (v : ℝ → ℝ) : Prop := ∀ x, v (-x) = -v x

/-- The sum of v(-3.14), v(-1.57), v(1.57), and v(3.14) is zero for any odd function v -/
theorem odd_function_sum_zero (v : ℝ → ℝ) (h : IsOdd v) :
  v (-3.14) + v (-1.57) + v 1.57 + v 3.14 = 0 :=
sorry

end NUMINAMATH_CALUDE_odd_function_sum_zero_l1163_116358


namespace NUMINAMATH_CALUDE_parabola_equation_c_value_l1163_116361

/-- A parabola with vertex at (5, 1) passing through (2, 3) has equation x = ay^2 + by + c where c = 17/4 -/
theorem parabola_equation_c_value (a b c : ℝ) : 
  (∀ y : ℝ, 5 = a * 1^2 + b * 1 + c) →  -- vertex at (5, 1)
  (2 = a * 3^2 + b * 3 + c) →           -- passes through (2, 3)
  (∀ x y : ℝ, x = a * y^2 + b * y + c) →  -- equation of the form x = ay^2 + by + c
  c = 17/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_c_value_l1163_116361


namespace NUMINAMATH_CALUDE_prime_sum_of_powers_l1163_116345

theorem prime_sum_of_powers (a b p : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime p → p = a^b + b^a → p = 17 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_of_powers_l1163_116345


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l1163_116316

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 130) (h2 : Nat.gcd a c = 770) :
  ∃ (b' c' : ℕ+), Nat.gcd a b' = 130 ∧ Nat.gcd a c' = 770 ∧ 
  Nat.gcd b' c' = 10 ∧ ∀ (b'' c'' : ℕ+), Nat.gcd a b'' = 130 → Nat.gcd a c'' = 770 → 
  Nat.gcd b'' c'' ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l1163_116316


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1163_116352

/-- A geometric sequence with a_3 = 8 * a_6 has S_4 / S_2 = 5/4 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- geometric sequence condition
  (∀ n, S n = (a 1 * (1 - (a 2 / a 1)^n)) / (1 - (a 2 / a 1))) →  -- sum formula
  a 3 = 8 * a 6 →  -- given condition
  S 4 / S 2 = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1163_116352


namespace NUMINAMATH_CALUDE_student_number_problem_l1163_116346

theorem student_number_problem (x : ℝ) : 2 * x - 148 = 110 → x = 129 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1163_116346


namespace NUMINAMATH_CALUDE_dunkers_lineups_l1163_116390

/-- The number of players in the team -/
def total_players : ℕ := 15

/-- The number of players who can't play together -/
def special_players : ℕ := 3

/-- The number of players in a starting lineup -/
def lineup_size : ℕ := 5

/-- The number of possible starting lineups -/
def possible_lineups : ℕ := 2277

/-- Function to calculate binomial coefficient -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem dunkers_lineups :
  choose (total_players - special_players) lineup_size +
  special_players * choose (total_players - special_players) (lineup_size - 1) =
  possible_lineups :=
sorry

end NUMINAMATH_CALUDE_dunkers_lineups_l1163_116390


namespace NUMINAMATH_CALUDE_largest_non_square_sum_diff_smallest_n_with_square_digit_sum_n_66_has_square_digit_sum_l1163_116377

def is_sum_of_squares (x : ℕ) : Prop := ∃ a b : ℕ, x = a^2 + b^2

def is_diff_of_squares (x : ℕ) : Prop := ∃ a b : ℕ, x = a^2 - b^2

def sum_of_digit_squares (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.map (λ d => d^2)).sum

def largest_n_digit_number (n : ℕ) : ℕ := 10^n - 1

theorem largest_non_square_sum_diff (n : ℕ) (h : n > 2) :
  ∀ k : ℕ, k ≤ largest_n_digit_number n →
    (¬ is_sum_of_squares k ∧ ¬ is_diff_of_squares k) →
    k ≤ 10^n - 2 :=
sorry

theorem smallest_n_with_square_digit_sum :
  ∀ n : ℕ, n < 66 → ¬ ∃ k : ℕ, sum_of_digit_squares n = k^2 :=
sorry

theorem n_66_has_square_digit_sum :
  ∃ k : ℕ, sum_of_digit_squares 66 = k^2 :=
sorry

end NUMINAMATH_CALUDE_largest_non_square_sum_diff_smallest_n_with_square_digit_sum_n_66_has_square_digit_sum_l1163_116377


namespace NUMINAMATH_CALUDE_order_of_abc_l1163_116366

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l1163_116366


namespace NUMINAMATH_CALUDE_last_three_digits_of_square_l1163_116368

theorem last_three_digits_of_square (n : ℕ) : ∃ n, n^2 % 1000 = 689 ∧ ¬∃ m, m^2 % 1000 = 759 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_square_l1163_116368


namespace NUMINAMATH_CALUDE_paper_count_l1163_116332

theorem paper_count (initial_math initial_science used_math used_science received_math given_science : ℕ) :
  initial_math = 220 →
  initial_science = 150 →
  used_math = 95 →
  used_science = 68 →
  received_math = 30 →
  given_science = 15 →
  (initial_math - used_math + received_math) + (initial_science - used_science - given_science) = 222 := by
  sorry

end NUMINAMATH_CALUDE_paper_count_l1163_116332


namespace NUMINAMATH_CALUDE_jacob_future_age_l1163_116360

-- Define Jacob's current age
def jacob_age : ℕ := sorry

-- Define Michael's current age
def michael_age : ℕ := sorry

-- Define the number of years from now
variable (X : ℕ)

-- Jacob is 14 years younger than Michael
axiom age_difference : jacob_age = michael_age - 14

-- In 9 years, Michael will be twice as old as Jacob
axiom future_age_relation : michael_age + 9 = 2 * (jacob_age + 9)

-- Theorem: Jacob's age in X years from now is 5 + X
theorem jacob_future_age : jacob_age + X = 5 + X := by sorry

end NUMINAMATH_CALUDE_jacob_future_age_l1163_116360


namespace NUMINAMATH_CALUDE_insect_jump_coordinates_l1163_116311

/-- A point in a 2D plane represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a jump to the right by a certain distance -/
def jumpRight (p : Point) (distance : ℝ) : Point :=
  ⟨p.x + distance, p.y⟩

theorem insect_jump_coordinates :
  let A : Point := ⟨-2, 1⟩
  let B : Point := jumpRight A 4
  B.x = 2 ∧ B.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_insect_jump_coordinates_l1163_116311


namespace NUMINAMATH_CALUDE_election_votes_l1163_116318

theorem election_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) :
  total_votes > 0 →
  winner_percentage = 62 / 100 →
  vote_difference = 300 →
  ⌊total_votes * winner_percentage⌋ - ⌊total_votes * (1 - winner_percentage)⌋ = vote_difference →
  ⌊total_votes * winner_percentage⌋ = 775 :=
by
  sorry

end NUMINAMATH_CALUDE_election_votes_l1163_116318


namespace NUMINAMATH_CALUDE_problem_solution_l1163_116303

theorem problem_solution : ∃! x : ℝ, 3 ∈ ({x + 2, x^2 + 2*x} : Set ℝ) ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1163_116303


namespace NUMINAMATH_CALUDE_abs_4y_minus_7_not_positive_l1163_116312

theorem abs_4y_minus_7_not_positive (y : ℚ) :
  (|4 * y - 7| ≤ 0) ↔ (y = 7 / 4) := by sorry

end NUMINAMATH_CALUDE_abs_4y_minus_7_not_positive_l1163_116312


namespace NUMINAMATH_CALUDE_cosine_shift_l1163_116333

theorem cosine_shift (x : ℝ) :
  let f (x : ℝ) := 3 * Real.cos (1/2 * x - π/3)
  let period := 4 * π
  let shift := period / 8
  let g (x : ℝ) := f (x + shift)
  g x = 3 * Real.cos (1/2 * x - π/12) := by
  sorry

end NUMINAMATH_CALUDE_cosine_shift_l1163_116333


namespace NUMINAMATH_CALUDE_delta_takes_five_hours_prove_delta_time_l1163_116338

/-- Represents the time taken by Delta and Epsilon to complete a landscaping job -/
structure LandscapingJob where
  delta : ℝ  -- Time taken by Delta alone
  epsilon : ℝ  -- Time taken by Epsilon alone
  together : ℝ  -- Time taken by both working together

/-- The conditions of the landscaping job -/
def job_conditions (job : LandscapingJob) : Prop :=
  job.together = job.delta - 3 ∧
  job.together = job.epsilon - 4 ∧
  job.together = 2

/-- The theorem stating that Delta takes 5 hours to complete the job alone -/
theorem delta_takes_five_hours (job : LandscapingJob) 
  (h : job_conditions job) : job.delta = 5 := by
  sorry

/-- The main theorem proving Delta's time based on the given conditions -/
theorem prove_delta_time : ∃ (job : LandscapingJob), job_conditions job ∧ job.delta = 5 := by
  sorry

end NUMINAMATH_CALUDE_delta_takes_five_hours_prove_delta_time_l1163_116338


namespace NUMINAMATH_CALUDE_cubic_polynomial_coefficients_l1163_116340

/-- Given a cubic polynomial with real coefficients that has 2 - 3i as a root,
    prove that its coefficients are a = -3, b = 1, and c = -39. -/
theorem cubic_polynomial_coefficients 
  (p : ℂ → ℂ) 
  (h1 : ∀ x, p x = x^3 + a*x^2 + b*x - c) 
  (h2 : p (2 - 3*I) = 0) 
  (a b c : ℝ) :
  a = -3 ∧ b = 1 ∧ c = -39 :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_coefficients_l1163_116340


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1163_116337

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + 2 * Complex.I) / (1 + 2 * Complex.I) = b * Complex.I) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1163_116337


namespace NUMINAMATH_CALUDE_table_height_is_five_l1163_116305

/-- Represents the configuration of blocks and table -/
structure Configuration where
  total_length : ℝ

/-- Represents the table and blocks setup -/
structure TableSetup where
  block_length : ℝ
  block_width : ℝ
  table_height : ℝ
  config1 : Configuration
  config2 : Configuration

/-- The theorem stating the height of the table given the configurations -/
theorem table_height_is_five (setup : TableSetup)
  (h1 : setup.config1.total_length = setup.block_length + setup.table_height + setup.block_width)
  (h2 : setup.config2.total_length = 2 * setup.block_width + setup.table_height)
  (h3 : setup.config1.total_length = 45)
  (h4 : setup.config2.total_length = 40) :
  setup.table_height = 5 := by
  sorry

#check table_height_is_five

end NUMINAMATH_CALUDE_table_height_is_five_l1163_116305


namespace NUMINAMATH_CALUDE_annulus_area_dead_grass_area_l1163_116341

/-- The area of an annulus formed by two concentric circles -/
theorem annulus_area (r₁ r₂ : ℝ) (h : 0 < r₁ ∧ r₁ < r₂) : 
  π * (r₂^2 - r₁^2) = π * (r₂ + r₁) * (r₂ - r₁) :=
by sorry

/-- The area of dead grass caused by a walking man with a sombrero -/
theorem dead_grass_area (r_walk r_sombrero : ℝ) 
  (h_walk : r_walk = 5)
  (h_sombrero : r_sombrero = 3) : 
  π * ((r_walk + r_sombrero)^2 - (r_walk - r_sombrero)^2) = 60 * π :=
by sorry

end NUMINAMATH_CALUDE_annulus_area_dead_grass_area_l1163_116341


namespace NUMINAMATH_CALUDE_sunny_lead_in_second_race_l1163_116398

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race scenario -/
structure RaceScenario where
  sunny : Runner
  windy : Runner
  race_distance : ℝ
  sunny_lead : ℝ

/-- Calculates the distance covered by a runner in a given time -/
def distance_covered (runner : Runner) (time : ℝ) : ℝ :=
  runner.speed * time

/-- Represents the conditions of the problem -/
def race_conditions : RaceScenario :=
  { sunny := { speed := 8 },
    windy := { speed := 7 },
    race_distance := 400,
    sunny_lead := 50 }

/-- Theorem statement -/
theorem sunny_lead_in_second_race :
  let first_race := race_conditions
  let second_race := { first_race with
    sunny := { speed := first_race.sunny.speed * 0.9 },
    sunny_lead := -50 }
  let sunny_time := (second_race.race_distance - second_race.sunny_lead) / second_race.sunny.speed
  let windy_distance := distance_covered second_race.windy sunny_time
  (second_race.race_distance - second_race.sunny_lead) - windy_distance = 12.5 := by
  sorry


end NUMINAMATH_CALUDE_sunny_lead_in_second_race_l1163_116398


namespace NUMINAMATH_CALUDE_triangle_side_length_l1163_116378

theorem triangle_side_length (A B C : ℝ × ℝ) (tanB : ℝ) (AB : ℝ) :
  tanB = 4 / 3 →
  AB = 3 →
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB^2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (AB * tanB)^2 →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1163_116378


namespace NUMINAMATH_CALUDE_complex_equation_modulus_l1163_116362

theorem complex_equation_modulus : ∀ (x y : ℝ), 
  (Complex.I : ℂ) * x + 2 * (Complex.I : ℂ) * x = (2 : ℂ) + (Complex.I : ℂ) * y → 
  Complex.abs (x + (Complex.I : ℂ) * y) = 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_modulus_l1163_116362


namespace NUMINAMATH_CALUDE_complex_symmetry_division_l1163_116381

/-- Two complex numbers are symmetric about the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_about_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- The main theorem: If z₁ and z₂ are symmetric about the imaginary axis and z₁ = -1 + i, then z₁ / z₂ = i -/
theorem complex_symmetry_division (z₁ z₂ : ℂ) 
  (h_sym : symmetric_about_imaginary_axis z₁ z₂) 
  (h_z₁ : z₁ = -1 + Complex.I) : 
  z₁ / z₂ = Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_symmetry_division_l1163_116381


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_one_l1163_116322

theorem expression_simplification_and_evaluation (x : ℝ) (h : x ≠ 3) :
  (x^2 - 5) / (x - 3) - 4 / (x - 3) = x + 3 :=
by sorry

theorem expression_evaluation_at_one :
  ((1 : ℝ)^2 - 5) / (1 - 3) - 4 / (1 - 3) = 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_one_l1163_116322


namespace NUMINAMATH_CALUDE_rectangle_width_25_l1163_116384

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  area : ℝ
  perimeter : ℝ

/-- The width of a rectangle -/
def width (r : Rectangle) : ℝ :=
  sorry

/-- The length of a rectangle -/
def length (r : Rectangle) : ℝ :=
  sorry

theorem rectangle_width_25 (r : Rectangle) 
  (h_area : r.area = 750)
  (h_perimeter : r.perimeter = 110) :
  width r = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_25_l1163_116384


namespace NUMINAMATH_CALUDE_opposite_face_of_B_l1163_116372

/-- Represents a square on the 3x3 grid --/
inductive Square
| A | B | C | D | E | F | G | H | I

/-- Represents the cube formed by folding the grid --/
structure Cube where
  open_face : Square
  opposite_pairs : List (Square × Square)

/-- Defines the folding of the 3x3 grid into a cube --/
def fold_grid (open_face : Square) : Cube :=
  { open_face := open_face,
    opposite_pairs := sorry }  -- The actual folding logic would go here

/-- The main theorem to prove --/
theorem opposite_face_of_B (c : Cube) :
  c.open_face = Square.F → (Square.B, Square.I) ∈ c.opposite_pairs :=
sorry

end NUMINAMATH_CALUDE_opposite_face_of_B_l1163_116372


namespace NUMINAMATH_CALUDE_no_collision_after_jumps_l1163_116315

/-- Represents the position of a grasshopper -/
structure Position where
  x : Int
  y : Int

/-- Represents the state of the system with four grasshoppers -/
structure GrasshopperSystem where
  positions : Fin 4 → Position

/-- Performs a symmetric jump for one grasshopper -/
def symmetricJump (system : GrasshopperSystem) (jumper : Fin 4) : GrasshopperSystem :=
  sorry

/-- Checks if any two grasshoppers occupy the same position -/
def hasCollision (system : GrasshopperSystem) : Bool :=
  sorry

/-- Initial configuration of the grasshoppers on a square -/
def initialSquare : GrasshopperSystem :=
  { positions := λ i => match i with
    | 0 => ⟨0, 0⟩
    | 1 => ⟨0, 1⟩
    | 2 => ⟨1, 1⟩
    | 3 => ⟨1, 0⟩ }

theorem no_collision_after_jumps :
  ∀ (jumps : List (Fin 4)), ¬(hasCollision (jumps.foldl symmetricJump initialSquare)) :=
  sorry

end NUMINAMATH_CALUDE_no_collision_after_jumps_l1163_116315


namespace NUMINAMATH_CALUDE_x_yz_equals_12_l1163_116367

theorem x_yz_equals_12 (x y z : ℝ) (h : x * (x + y + z) = x^2 + 12) : x * (y + z) = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_yz_equals_12_l1163_116367


namespace NUMINAMATH_CALUDE_tan_150_degrees_l1163_116394

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l1163_116394


namespace NUMINAMATH_CALUDE_complex_modulus_l1163_116329

theorem complex_modulus (z : ℂ) (h : z * (1 - Complex.I) = -Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l1163_116329


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1163_116347

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ascending order
  b = 10 →  -- Median is 10
  (a + b + c) / 3 = a + 20 →  -- Mean is 20 more than least
  (a + b + c) / 3 = c - 25 →  -- Mean is 25 less than greatest
  a + b + c = 45 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1163_116347


namespace NUMINAMATH_CALUDE_baker_flour_remaining_l1163_116310

/-- A baker's recipe requires 3 eggs for every 2 cups of flour. -/
def recipe_ratio : ℚ := 3 / 2

/-- The number of eggs needed to use up all remaining flour. -/
def eggs_needed : ℕ := 9

/-- Calculates the number of cups of flour remaining given the recipe ratio and eggs needed. -/
def flour_remaining (ratio : ℚ) (eggs : ℕ) : ℚ := (eggs : ℚ) / ratio

theorem baker_flour_remaining :
  flour_remaining recipe_ratio eggs_needed = 6 := by
  sorry

end NUMINAMATH_CALUDE_baker_flour_remaining_l1163_116310


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1163_116314

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|x| - 2 = 1) ↔ (x = 3 ∨ x = -3) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1163_116314


namespace NUMINAMATH_CALUDE_phone_sale_problem_l1163_116385

theorem phone_sale_problem (total : ℕ) (defective : ℕ) (customer_a : ℕ) (customer_b : ℕ) 
  (h_total : total = 20)
  (h_defective : defective = 5)
  (h_customer_a : customer_a = 3)
  (h_customer_b : customer_b = 5)
  (h_all_sold : total - defective = customer_a + customer_b + (total - defective - customer_a - customer_b)) :
  total - defective - customer_a - customer_b = 7 := by
  sorry

end NUMINAMATH_CALUDE_phone_sale_problem_l1163_116385


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1163_116353

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 3 = 3 →
  a 4 + a 6 = 6 →
  a 1 * a 3 + a 2 * a 4 + a 3 * a 5 + a 4 * a 6 + a 5 * a 7 = 62 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1163_116353


namespace NUMINAMATH_CALUDE_second_number_value_l1163_116343

theorem second_number_value (x y : ℝ) 
  (h1 : (1/5) * x = (5/8) * y) 
  (h2 : x + 35 = 4 * y) : 
  y = 40 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1163_116343


namespace NUMINAMATH_CALUDE_saree_sale_price_l1163_116304

/-- Applies a discount to a given price -/
def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

/-- Calculates the final price after applying multiple discounts -/
def final_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount original_price

theorem saree_sale_price :
  let original_price : ℝ := 175
  let discounts : List ℝ := [0.30, 0.25, 0.15, 0.10]
  let result := final_price original_price discounts
  ∃ ε > 0, |result - 70.28| < ε :=
sorry

end NUMINAMATH_CALUDE_saree_sale_price_l1163_116304


namespace NUMINAMATH_CALUDE_fixed_equidistant_point_l1163_116350

-- Define the circles
variable (k₁ k₂ : Set (ℝ × ℝ))

-- Define the intersection point A
variable (A : ℝ × ℝ)

-- Define the particles P₁ and P₂ as functions of time
variable (P₁ P₂ : ℝ → ℝ × ℝ)

-- Define the constant angular speeds
variable (ω₁ ω₂ : ℝ)

-- Axioms
axiom circles_intersect : A ∈ k₁ ∩ k₂

axiom P₁_on_k₁ : ∀ t, P₁ t ∈ k₁
axiom P₂_on_k₂ : ∀ t, P₂ t ∈ k₂

axiom start_at_A : P₁ 0 = A ∧ P₂ 0 = A

axiom constant_speed : ∀ t, ‖(P₁ t).fst - (P₁ 0).fst‖ = ω₁ * t
                    ∧ ‖(P₂ t).fst - (P₂ 0).fst‖ = ω₂ * t

axiom same_direction : ω₁ * ω₂ > 0

axiom simultaneous_arrival : ∃ T > 0, P₁ T = A ∧ P₂ T = A

-- Theorem
theorem fixed_equidistant_point :
  ∃ P : ℝ × ℝ, ∀ t, ‖P - P₁ t‖ = ‖P - P₂ t‖ :=
sorry

end NUMINAMATH_CALUDE_fixed_equidistant_point_l1163_116350


namespace NUMINAMATH_CALUDE_problem_1_l1163_116374

theorem problem_1 : -2.8 + (-3.6) + 3 - (-3.6) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1163_116374


namespace NUMINAMATH_CALUDE_chord_length_inequality_l1163_116359

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define the line y = kx + 1
def line1 (k x y : ℝ) : Prop := y = k*x + 1

-- Define the line kx + y - 2 = 0
def line2 (k x y : ℝ) : Prop := k*x + y - 2 = 0

-- Define a function to calculate the chord length
noncomputable def chordLength (k : ℝ) (line : ℝ → ℝ → ℝ → Prop) : ℝ :=
  sorry -- Actual calculation of chord length would go here

-- Theorem statement
theorem chord_length_inequality (k : ℝ) :
  chordLength k line1 ≠ chordLength k line2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_inequality_l1163_116359


namespace NUMINAMATH_CALUDE_moore_law_gpu_transistors_l1163_116388

def initial_year : Nat := 1992
def final_year : Nat := 2011
def initial_transistors : Nat := 500000
def doubling_period : Nat := 3

def moore_law_prediction (initial : Nat) (years : Nat) (period : Nat) : Nat :=
  initial * (2 ^ (years / period))

theorem moore_law_gpu_transistors :
  moore_law_prediction initial_transistors (final_year - initial_year) doubling_period = 32000000 := by
  sorry

end NUMINAMATH_CALUDE_moore_law_gpu_transistors_l1163_116388


namespace NUMINAMATH_CALUDE_gustran_facial_cost_l1163_116324

/-- Represents the prices of services at a salon -/
structure SalonPrices where
  haircut : ℕ
  nails : ℕ
  facial : ℕ

/-- Calculates the total cost of services at a salon -/
def totalCost (prices : SalonPrices) : ℕ :=
  prices.haircut + prices.nails + prices.facial

theorem gustran_facial_cost (gustran : SalonPrices) (barbara : SalonPrices) (fancy : SalonPrices)
  (h1 : gustran.haircut = 45)
  (h2 : gustran.nails = 30)
  (h3 : barbara.haircut = 30)
  (h4 : barbara.nails = 40)
  (h5 : barbara.facial = 28)
  (h6 : fancy.haircut = 34)
  (h7 : fancy.nails = 20)
  (h8 : fancy.facial = 30)
  (h9 : totalCost fancy = 84)
  (h10 : totalCost fancy ≤ totalCost barbara)
  (h11 : totalCost fancy ≤ totalCost gustran) :
  gustran.facial = 9 := by
  sorry

end NUMINAMATH_CALUDE_gustran_facial_cost_l1163_116324


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l1163_116376

/-- The surface area of a hemisphere given its base area -/
theorem hemisphere_surface_area (base_area : ℝ) (h : base_area = 3) :
  let r : ℝ := Real.sqrt (base_area / Real.pi)
  2 * Real.pi * r^2 + base_area = 9 := by
  sorry


end NUMINAMATH_CALUDE_hemisphere_surface_area_l1163_116376


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1163_116375

/-- A positive geometric sequence with specific sum conditions has a common ratio of 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- The sequence is positive
  (∃ q > 0, ∀ n, a (n + 1) = a n * q) →  -- The sequence is geometric with positive ratio
  a 3 + a 5 = 5 →  -- First condition
  a 5 + a 7 = 20 →  -- Second condition
  (∃ q > 0, ∀ n, a (n + 1) = a n * q ∧ q = 2) :=  -- The common ratio is 2
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1163_116375


namespace NUMINAMATH_CALUDE_sum_of_first_and_last_l1163_116344

/-- A sequence of eight terms -/
structure EightTermSequence where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  U : ℝ
  V : ℝ
  W : ℝ

/-- The sum of any four consecutive terms is 40 -/
def consecutive_sum_40 (seq : EightTermSequence) : Prop :=
  seq.P + seq.Q + seq.R + seq.S = 40 ∧
  seq.Q + seq.R + seq.S + seq.T = 40 ∧
  seq.R + seq.S + seq.T + seq.U = 40 ∧
  seq.S + seq.T + seq.U + seq.V = 40 ∧
  seq.T + seq.U + seq.V + seq.W = 40

theorem sum_of_first_and_last (seq : EightTermSequence) 
  (h1 : seq.S = 10)
  (h2 : consecutive_sum_40 seq) : 
  seq.P + seq.W = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_and_last_l1163_116344


namespace NUMINAMATH_CALUDE_min_value_and_inequality_range_l1163_116382

theorem min_value_and_inequality_range (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (|2*a + b| + |2*a - b|) / |a| ≥ 4) ∧
  (∀ x : ℝ, |2*a + b| + |2*a - b| ≥ |a| * (|2 + x| + |2 - x|) ↔ -2 ≤ x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_range_l1163_116382


namespace NUMINAMATH_CALUDE_sin_alpha_plus_5pi_12_l1163_116317

theorem sin_alpha_plus_5pi_12 (α : Real) (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : Real.cos α - Real.sin α = 2 * Real.sqrt 2 / 3) :
  Real.sin (α + 5 * π / 12) = (2 + Real.sqrt 15) / 6 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_5pi_12_l1163_116317


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1163_116308

theorem ceiling_sum_sqrt : ⌈Real.sqrt 20⌉ + ⌈Real.sqrt 200⌉ + ⌈Real.sqrt 2000⌉ = 65 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1163_116308


namespace NUMINAMATH_CALUDE_real_solutions_of_equation_l1163_116320

theorem real_solutions_of_equation (x : ℝ) : 
  x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by
sorry

end NUMINAMATH_CALUDE_real_solutions_of_equation_l1163_116320


namespace NUMINAMATH_CALUDE_max_value_cubic_function_l1163_116342

theorem max_value_cubic_function :
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → x^3 - 3*x^2 + 2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cubic_function_l1163_116342


namespace NUMINAMATH_CALUDE_meet_five_times_l1163_116355

/-- Represents the meeting problem between Michael and the garbage truck --/
structure MeetingProblem where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (problem : MeetingProblem) : ℕ :=
  sorry

/-- The theorem stating that Michael and the truck meet exactly 5 times --/
theorem meet_five_times (problem : MeetingProblem) 
  (h1 : problem.michael_speed = 5)
  (h2 : problem.truck_speed = 10)
  (h3 : problem.pail_distance = 200)
  (h4 : problem.truck_stop_time = 30) :
  number_of_meetings problem = 5 :=
  sorry

end NUMINAMATH_CALUDE_meet_five_times_l1163_116355


namespace NUMINAMATH_CALUDE_percent_value_in_quarters_l1163_116331

def num_dimes : ℕ := 75
def num_quarters : ℕ := 30
def value_dime : ℕ := 10
def value_quarter : ℕ := 25

def total_value : ℕ := num_dimes * value_dime + num_quarters * value_quarter
def value_in_quarters : ℕ := num_quarters * value_quarter

theorem percent_value_in_quarters :
  (value_in_quarters : ℚ) / total_value * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_percent_value_in_quarters_l1163_116331


namespace NUMINAMATH_CALUDE_eighth_term_is_22_l1163_116301

/-- An arithmetic sequence with a_1 = 1 and sum of first 5 terms = 35 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 1 + a 2 + a 3 + a 4 + a 5 = 35)

/-- The 8th term of the sequence is 22 -/
theorem eighth_term_is_22 (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  a 8 = 22 := by
  sorry


end NUMINAMATH_CALUDE_eighth_term_is_22_l1163_116301


namespace NUMINAMATH_CALUDE_least_k_for_subset_sum_l1163_116330

theorem least_k_for_subset_sum (n : ℕ) :
  let k := if n % 2 = 1 then 2 * n else n + 1
  ∀ (A : Finset ℕ), A.card ≥ k →
    ∃ (S : Finset ℕ), S ⊆ A ∧ S.card % 2 = 0 ∧ (S.sum id) % n = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_k_for_subset_sum_l1163_116330


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l1163_116392

theorem sum_of_solutions_squared_equation (x : ℝ) :
  (∀ x, (x - 4)^2 = 16 → x = 0 ∨ x = 8) →
  (∃ a b, (a - 4)^2 = 16 ∧ (b - 4)^2 = 16 ∧ a + b = 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l1163_116392


namespace NUMINAMATH_CALUDE_light_travel_distance_l1163_116306

/-- The distance light travels in one year in a vacuum (in miles) -/
def light_speed_vacuum : ℝ := 5870000000000

/-- The factor by which light speed is reduced in the medium -/
def speed_reduction_factor : ℝ := 2

/-- The number of years we're considering -/
def years : ℝ := 1000

/-- The theorem stating the distance light travels in the given conditions -/
theorem light_travel_distance :
  (light_speed_vacuum / speed_reduction_factor) * years = 2935 * (10 ^ 12) := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l1163_116306
