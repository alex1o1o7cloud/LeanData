import Mathlib

namespace triangle_side_lengths_l1363_136390

theorem triangle_side_lengths (x y z k : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (k_ge_2 : k ≥ 2) 
  (prod_cond : x * y * z ≤ 2) 
  (sum_cond : 1 / x^2 + 1 / y^2 + 1 / z^2 < k) :
  (∃ a b c : ℝ, a = x ∧ b = y ∧ c = z ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  (2 ≤ k ∧ k ≤ 9/4) :=
sorry

end triangle_side_lengths_l1363_136390


namespace dealership_sales_prediction_l1363_136389

/-- Represents the sales ratio of different car types -/
structure SalesRatio where
  sports : ℕ
  sedans : ℕ
  suvs : ℕ

/-- Represents the expected sales of different car types -/
structure ExpectedSales where
  sports : ℕ
  sedans : ℕ
  suvs : ℕ

/-- Given a sales ratio and expected sports car sales, calculates the expected sales of all car types -/
def calculateExpectedSales (ratio : SalesRatio) (expectedSports : ℕ) : ExpectedSales :=
  { sports := expectedSports,
    sedans := expectedSports * ratio.sedans / ratio.sports,
    suvs := expectedSports * ratio.suvs / ratio.sports }

theorem dealership_sales_prediction 
  (ratio : SalesRatio)
  (expectedSports : ℕ)
  (h1 : ratio.sports = 5)
  (h2 : ratio.sedans = 8)
  (h3 : ratio.suvs = 3)
  (h4 : expectedSports = 35) :
  let expected := calculateExpectedSales ratio expectedSports
  expected.sedans = 56 ∧ expected.suvs = 21 := by
  sorry

end dealership_sales_prediction_l1363_136389


namespace complement_union_theorem_l1363_136397

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_union_theorem :
  (A ∪ B)ᶜ = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end complement_union_theorem_l1363_136397


namespace cuboid_surface_area_example_l1363_136346

/-- The surface area of a cuboid with given dimensions -/
def cuboidSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

/-- Theorem: The surface area of a cuboid with edges 4 cm, 5 cm, and 6 cm is 148 cm² -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 4 5 6 = 148 := by
  sorry

end cuboid_surface_area_example_l1363_136346


namespace count_valid_numbers_l1363_136334

def is_odd (n : ℕ) : Prop := n % 2 = 1

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def valid_number (n : ℕ) : Prop :=
  n > 0 ∧ n < 1000000 ∧ is_odd n ∧ is_odd (digit_sum n) ∧ is_odd (digit_product n)

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, valid_number n) ∧ S.card = 39 ∧
  ∀ m, valid_number m → m ∈ S :=
sorry

end count_valid_numbers_l1363_136334


namespace find_unknown_number_l1363_136358

theorem find_unknown_number : ∃ x : ℝ, (213 * 16 = 3408) ∧ (1.6 * x = 3.408) → x = 2.13 := by
  sorry

end find_unknown_number_l1363_136358


namespace percentage_of_B_grades_l1363_136338

def scores : List Nat := [91, 68, 58, 99, 82, 94, 88, 76, 79, 62, 87, 81, 65, 85, 89, 73, 77, 84, 59, 72]

def is_grade_B (score : Nat) : Bool :=
  85 ≤ score ∧ score ≤ 94

def count_grade_B (scores : List Nat) : Nat :=
  scores.filter is_grade_B |>.length

theorem percentage_of_B_grades :
  (count_grade_B scores : ℚ) / scores.length * 100 = 25 := by
  sorry

end percentage_of_B_grades_l1363_136338


namespace pentagon_coverage_theorem_l1363_136306

/-- Represents the tiling of a plane with squares and pentagons -/
structure PlaneTiling where
  /-- The number of smaller squares in each large square -/
  total_squares : ℕ
  /-- The number of smaller squares used to form pentagons in each large square -/
  pentagon_squares : ℕ

/-- Calculates the percentage of the plane covered by pentagons -/
def pentagon_coverage_percentage (tiling : PlaneTiling) : ℚ :=
  (tiling.pentagon_squares : ℚ) / (tiling.total_squares : ℚ) * 100

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest_integer (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- Theorem stating that the percentage of the plane covered by pentagons
    in the given tiling is 56% when rounded to the nearest integer -/
theorem pentagon_coverage_theorem (tiling : PlaneTiling) 
  (h1 : tiling.total_squares = 9)
  (h2 : tiling.pentagon_squares = 5) : 
  round_to_nearest_integer (pentagon_coverage_percentage tiling) = 56 := by
  sorry

end pentagon_coverage_theorem_l1363_136306


namespace distinct_real_pairs_l1363_136367

theorem distinct_real_pairs (x y : ℝ) (hxy : x ≠ y) :
  x^100 - y^100 = 2^99 * (x - y) ∧
  x^200 - y^200 = 2^199 * (x - y) →
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) := by
  sorry

end distinct_real_pairs_l1363_136367


namespace boxes_in_marker_carton_l1363_136363

def pencil_cartons : ℕ := 20
def pencil_boxes_per_carton : ℕ := 10
def pencil_box_cost : ℕ := 2
def marker_cartons : ℕ := 10
def marker_carton_cost : ℕ := 4
def total_spent : ℕ := 600

theorem boxes_in_marker_carton :
  ∃ (x : ℕ), 
    x * marker_carton_cost * marker_cartons + 
    pencil_cartons * pencil_boxes_per_carton * pencil_box_cost = 
    total_spent ∧ 
    x = 5 := by sorry

end boxes_in_marker_carton_l1363_136363


namespace xy_minus_two_equals_negative_one_l1363_136332

theorem xy_minus_two_equals_negative_one 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : (Real.sqrt (x^2 + 1) - x + 1) * (Real.sqrt (y^2 + 1) - y + 1) = 2) : 
  x * y - 2 = -1 := by
sorry

end xy_minus_two_equals_negative_one_l1363_136332


namespace probability_of_sum_15_l1363_136392

/-- Represents a standard playing card --/
inductive Card
| Number (n : Nat)
| Face

/-- A standard 52-card deck --/
def Deck : Finset Card :=
  sorry

/-- Predicate for a card being a number card (2 through 10) --/
def isNumberCard (c : Card) : Prop :=
  match c with
  | Card.Number n => 2 ≤ n ∧ n ≤ 10
  | Card.Face => False

/-- Predicate for two cards summing to 15 --/
def sumsTo15 (c1 c2 : Card) : Prop :=
  match c1, c2 with
  | Card.Number n1, Card.Number n2 => n1 + n2 = 15
  | _, _ => False

/-- The probability of selecting two number cards that sum to 15 --/
def probabilityOfSum15 : ℚ :=
  sorry

theorem probability_of_sum_15 :
  probabilityOfSum15 = 8 / 442 := by
  sorry

end probability_of_sum_15_l1363_136392


namespace parallel_condition_not_sufficient_nor_necessary_l1363_136343

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Two lines are parallel -/
def parallel_lines (l m : Line3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

theorem parallel_condition_not_sufficient_nor_necessary 
  (l m : Line3D) (α : Plane3D) 
  (h_diff : l ≠ m) (h_parallel : parallel_lines l m) : 
  (¬ (∀ α, parallel_line_plane l α → parallel_line_plane m α)) ∧ 
  (¬ (∀ α, parallel_line_plane m α → parallel_line_plane l α)) :=
by sorry

end parallel_condition_not_sufficient_nor_necessary_l1363_136343


namespace product_equality_l1363_136325

theorem product_equality : 250 * 24.98 * 2.498 * 1250 = 19484012.5 := by
  sorry

end product_equality_l1363_136325


namespace john_index_cards_purchase_l1363_136337

/-- Calculates the total number of index card packs bought for all students -/
def total_packs_bought (num_classes : ℕ) (students_per_class : ℕ) (packs_per_student : ℕ) : ℕ :=
  num_classes * students_per_class * packs_per_student

/-- Proves that given 6 classes with 30 students each, and 2 packs per student, the total packs bought is 360 -/
theorem john_index_cards_purchase :
  total_packs_bought 6 30 2 = 360 := by
  sorry

end john_index_cards_purchase_l1363_136337


namespace solutions_count_l1363_136333

/-- The number of solutions to the Diophantine equation 3x + 5y = 805 where x and y are positive integers -/
def num_solutions : ℕ :=
  (Finset.filter (fun t : ℕ => 265 - 5 * t > 0 ∧ 2 + 3 * t > 0) (Finset.range 53)).card

theorem solutions_count : num_solutions = 53 := by
  sorry

end solutions_count_l1363_136333


namespace watson_class_composition_l1363_136344

/-- The number of kindergartners in Ms. Watson's class -/
def num_kindergartners : ℕ := 42 - (24 + 4)

theorem watson_class_composition :
  num_kindergartners = 14 :=
by sorry

end watson_class_composition_l1363_136344


namespace trigonometric_identity_l1363_136393

theorem trigonometric_identity (α : ℝ) 
  (h : Real.cos (π / 4 + α) = Real.sqrt 2 / 3) : 
  Real.sin (2 * α) / (1 - Real.sin α + Real.cos α) = 1 / 3 := by
  sorry

end trigonometric_identity_l1363_136393


namespace probability_a_speaks_truth_l1363_136387

theorem probability_a_speaks_truth 
  (prob_b : ℝ)
  (prob_a_and_b : ℝ)
  (h1 : prob_b = 0.60)
  (h2 : prob_a_and_b = 0.51)
  (h3 : ∃ (prob_a : ℝ), prob_a_and_b = prob_a * prob_b) :
  ∃ (prob_a : ℝ), prob_a = 0.85 := by
sorry

end probability_a_speaks_truth_l1363_136387


namespace identifier_count_l1363_136396

/-- The number of English letters -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible characters for the second and third positions -/
def num_chars : ℕ := num_letters + num_digits

/-- The total number of possible identifiers -/
def total_identifiers : ℕ := num_letters + (num_letters * num_chars) + (num_letters * num_chars * num_chars)

theorem identifier_count : total_identifiers = 34658 := by
  sorry

end identifier_count_l1363_136396


namespace geometric_sequence_problem_l1363_136335

/-- Given a geometric sequence {a_n} where a_5 = -16 and a_8 = 8, prove that a_11 = -4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n)^m) →  -- geometric sequence property
  a 5 = -16 →
  a 8 = 8 →
  a 11 = -4 := by
sorry

end geometric_sequence_problem_l1363_136335


namespace trig_equation_solution_l1363_136357

theorem trig_equation_solution (x : Real) : 
  (6 * Real.sin x ^ 2 + Real.sin x * Real.cos x - Real.cos x ^ 2 = 2) ↔ 
  (∃ k : Int, x = -π/4 + π * k) ∨ 
  (∃ n : Int, x = Real.arctan (3/4) + π * n) := by sorry

end trig_equation_solution_l1363_136357


namespace intercept_sum_modulo_40_l1363_136395

/-- 
Given the congruence 5x ≡ 3y - 2 (mod 40), this theorem proves that 
the sum of the x-intercept and y-intercept is 38, where both intercepts 
are non-negative integers less than 40.
-/
theorem intercept_sum_modulo_40 : ∃ (x₀ y₀ : ℕ), 
  x₀ < 40 ∧ y₀ < 40 ∧ 
  (5 * x₀) % 40 = (3 * 0 - 2) % 40 ∧
  (5 * 0) % 40 = (3 * y₀ - 2) % 40 ∧
  x₀ + y₀ = 38 := by
  sorry

end intercept_sum_modulo_40_l1363_136395


namespace m_properties_l1363_136385

/-- The smallest positive integer with both 5 and 6 as digits, each appearing at least once, and divisible by both 3 and 7 -/
def m : ℕ := 5665665660

/-- Checks if a natural number contains both 5 and 6 as digits -/
def has_five_and_six (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * 10 + 5 ∧ ∃ (c d : ℕ), n = c * 10 + 6

/-- Returns the last four digits of a natural number -/
def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem m_properties :
  has_five_and_six m ∧ 
  m % 3 = 0 ∧ 
  m % 7 = 0 ∧ 
  ∀ k < m, ¬(has_five_and_six k ∧ k % 3 = 0 ∧ k % 7 = 0) ∧
  last_four_digits m = 5660 :=
sorry

end m_properties_l1363_136385


namespace tina_fruit_difference_l1363_136360

/-- Calculates the difference between remaining tangerines and oranges in Tina's bag --/
def remaining_difference (initial_oranges initial_tangerines removed_oranges removed_tangerines : ℕ) : ℕ :=
  (initial_tangerines - removed_tangerines) - (initial_oranges - removed_oranges)

/-- Proves that the difference between remaining tangerines and oranges is 4 --/
theorem tina_fruit_difference :
  remaining_difference 5 17 2 10 = 4 := by
  sorry

end tina_fruit_difference_l1363_136360


namespace mabel_katrina_marble_ratio_l1363_136359

/-- Prove that Mabel has 5 times as many marbles as Katrina -/
theorem mabel_katrina_marble_ratio : 
  ∀ (amanda katrina mabel : ℕ),
  amanda + 12 = 2 * katrina →
  mabel = 85 →
  mabel = amanda + 63 →
  mabel / katrina = 5 := by
sorry

end mabel_katrina_marble_ratio_l1363_136359


namespace largest_integer_x_l1363_136322

theorem largest_integer_x : ∀ x : ℤ, (2 * x : ℚ) / 7 + 3 / 4 < 8 / 7 ↔ x ≤ 1 := by
  sorry

end largest_integer_x_l1363_136322


namespace stock_price_decrease_l1363_136379

theorem stock_price_decrease (F : ℝ) (h1 : F > 0) : 
  let J := 0.9 * F
  let M := 0.8 * J
  (F - M) / F * 100 = 28 := by
sorry

end stock_price_decrease_l1363_136379


namespace power_of_product_squared_l1363_136347

theorem power_of_product_squared (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := by
  sorry

end power_of_product_squared_l1363_136347


namespace sphere_surface_area_l1363_136391

theorem sphere_surface_area (triangle_side_length : ℝ) (center_to_plane_distance : ℝ) : 
  triangle_side_length = 3 →
  center_to_plane_distance = Real.sqrt 7 →
  ∃ (sphere_radius : ℝ),
    sphere_radius ^ 2 = triangle_side_length ^ 2 / 3 + center_to_plane_distance ^ 2 ∧
    4 * Real.pi * sphere_radius ^ 2 = 40 * Real.pi := by
  sorry

end sphere_surface_area_l1363_136391


namespace probability_one_third_implies_five_l1363_136365

def integer_list : List ℕ := [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

def count (n : ℕ) (l : List ℕ) : ℕ := (l.filter (· = n)).length

theorem probability_one_third_implies_five :
  ∀ n : ℕ, 
  (count n integer_list : ℚ) / (integer_list.length : ℚ) = 1 / 3 →
  n = 5 := by
sorry

end probability_one_third_implies_five_l1363_136365


namespace non_red_cubes_count_total_small_cubes_correct_l1363_136326

/-- Represents the number of small cubes without red faces in a 6x6x6 cube with three faces painted red -/
def non_red_cubes : Set ℕ :=
  {n : ℕ | n = 120 ∨ n = 125}

/-- The main theorem stating that the number of non-red cubes is either 120 or 125 -/
theorem non_red_cubes_count :
  ∀ n : ℕ, n ∈ non_red_cubes ↔ (n = 120 ∨ n = 125) :=
by
  sorry

/-- The cube is 6x6x6 -/
def cube_size : ℕ := 6

/-- The number of small cubes the large cube is cut into -/
def total_small_cubes : ℕ := 216

/-- The number of faces painted red -/
def painted_faces : ℕ := 3

/-- The size of each small cube -/
def small_cube_size : ℕ := 1

/-- Theorem stating that the total number of small cubes is correct -/
theorem total_small_cubes_correct :
  cube_size ^ 3 = total_small_cubes :=
by
  sorry

end non_red_cubes_count_total_small_cubes_correct_l1363_136326


namespace bees_in_hive_l1363_136310

/-- The total number of bees in a hive after more bees fly in -/
def total_bees (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: The total number of bees is 24 when there are initially 16 bees and 8 more fly in -/
theorem bees_in_hive : total_bees 16 8 = 24 := by
  sorry

end bees_in_hive_l1363_136310


namespace some_employees_not_managers_l1363_136316

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Employee : U → Prop)
variable (Manager : U → Prop)
variable (Punctual : U → Prop)
variable (Shareholder : U → Prop)

-- State the theorem
theorem some_employees_not_managers
  (h1 : ∃ x, Employee x ∧ ¬Punctual x)
  (h2 : ∀ x, Manager x → Punctual x)
  (h3 : ∃ x, Manager x ∧ Shareholder x) :
  ∃ x, Employee x ∧ ¬Manager x :=
sorry

end some_employees_not_managers_l1363_136316


namespace arithmetic_mean_example_l1363_136320

theorem arithmetic_mean_example : 
  let numbers : List ℕ := [12, 24, 36, 48]
  (numbers.sum / numbers.length : ℚ) = 30 := by
sorry

end arithmetic_mean_example_l1363_136320


namespace platform_length_l1363_136311

/-- Given a train of length 450 meters that takes 39 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is 525 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 450 →
  time_platform = 39 →
  time_pole = 18 →
  (train_length / time_pole) * time_platform - train_length = 525 :=
by sorry

end platform_length_l1363_136311


namespace fraction_equality_l1363_136355

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) 
  (h2 : (2 * a) / (3 * b) + (a + 12 * b) / (3 * b + 12 * a) = 5 / 3) : 
  a / b = -93 / 49 := by
  sorry

end fraction_equality_l1363_136355


namespace fraction_to_decimal_l1363_136350

theorem fraction_to_decimal : (17 : ℚ) / 625 = 0.0272 := by sorry

end fraction_to_decimal_l1363_136350


namespace sum_of_cubes_difference_l1363_136394

theorem sum_of_cubes_difference (a b c : ℕ+) :
  (a + b + c : ℕ)^3 - a^3 - b^3 - c^3 = 180 → a + b + c = 4 := by
sorry

end sum_of_cubes_difference_l1363_136394


namespace difference_sum_of_T_l1363_136399

def T : Finset ℕ := Finset.range 9

def difference_sum (s : Finset ℕ) : ℕ :=
  s.sum (fun i => s.sum (fun j => if 3^j > 3^i then 3^j - 3^i else 0))

theorem difference_sum_of_T : difference_sum T = 69022 := by
  sorry

end difference_sum_of_T_l1363_136399


namespace urn_problem_l1363_136302

def urn1_green : ℚ := 5
def urn1_blue : ℚ := 7
def urn2_green : ℚ := 20
def urn1_total : ℚ := urn1_green + urn1_blue
def same_color_prob : ℚ := 62/100

theorem urn_problem (M : ℚ) :
  (urn1_green / urn1_total) * (urn2_green / (urn2_green + M)) +
  (urn1_blue / urn1_total) * (M / (urn2_green + M)) = same_color_prob →
  M = 610/1657 := by
sorry

end urn_problem_l1363_136302


namespace cos_neg_nineteen_pi_sixths_l1363_136315

theorem cos_neg_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_neg_nineteen_pi_sixths_l1363_136315


namespace quadratic_value_at_2_l1363_136307

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The properties of the quadratic function -/
structure QuadraticProperties (a b c : ℝ) : Prop where
  max_value : ∃ (y : ℝ), ∀ (x : ℝ), f a b c x ≤ y ∧ f a b c (-2) = y
  max_is_10 : f a b c (-2) = 10
  passes_through : f a b c 0 = -6

theorem quadratic_value_at_2 {a b c : ℝ} (h : QuadraticProperties a b c) : 
  f a b c 2 = -54 := by
  sorry

#check quadratic_value_at_2

end quadratic_value_at_2_l1363_136307


namespace absolute_value_equation_l1363_136314

theorem absolute_value_equation : 
  {x : ℤ | |(-5 + x)| = 11} = {16, -6} := by sorry

end absolute_value_equation_l1363_136314


namespace sqrt_defined_iff_l1363_136376

theorem sqrt_defined_iff (x : ℝ) : Real.sqrt (5 - 3 * x) ≥ 0 ↔ x ≤ 5 / 3 := by
  sorry

end sqrt_defined_iff_l1363_136376


namespace y_minus_x_values_l1363_136382

theorem y_minus_x_values (x y : ℝ) 
  (h1 : |x + 1| = 3)
  (h2 : |y| = 5)
  (h3 : -y/x > 0) :
  y - x = -7 ∨ y - x = 9 := by
sorry

end y_minus_x_values_l1363_136382


namespace davids_english_marks_l1363_136383

def marks_mathematics : ℕ := 85
def marks_physics : ℕ := 92
def marks_chemistry : ℕ := 87
def marks_biology : ℕ := 95
def average_marks : ℕ := 89
def number_of_subjects : ℕ := 5

theorem davids_english_marks :
  ∃ (marks_english : ℕ),
    marks_english +
    marks_mathematics +
    marks_physics +
    marks_chemistry +
    marks_biology =
    average_marks * number_of_subjects ∧
    marks_english = 86 := by
  sorry

end davids_english_marks_l1363_136383


namespace initial_marbles_l1363_136309

theorem initial_marbles (M : ℚ) : 
  (2 / 5 : ℚ) * M = 30 →
  (1 / 2 : ℚ) * ((2 / 5 : ℚ) * M) = 15 →
  M = 75 := by
  sorry

end initial_marbles_l1363_136309


namespace negation_of_proposition_l1363_136308

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a^2 + b^2 = 4 → a ≥ 2*b) ↔
  (∃ (a b : ℝ), a^2 + b^2 = 4 ∧ a < 2*b) :=
by sorry

end negation_of_proposition_l1363_136308


namespace intersection_complement_A_with_B_l1363_136381

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 4)}
def B : Set ℝ := {x | -1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 0}

theorem intersection_complement_A_with_B :
  (Set.univ \ A) ∩ B = Set.Icc (0 : ℝ) (1/2) := by sorry

end intersection_complement_A_with_B_l1363_136381


namespace square_roots_problem_l1363_136312

theorem square_roots_problem (x : ℝ) :
  (x + 1 > 0) ∧ (4 - 2*x > 0) ∧ (x + 1)^2 = (4 - 2*x)^2 →
  (x + 1)^2 = 36 :=
by sorry

end square_roots_problem_l1363_136312


namespace root_sum_squares_plus_product_l1363_136349

theorem root_sum_squares_plus_product (a b : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + b*x₁ + b^2 + a = 0) → 
  (x₂^2 + b*x₂ + b^2 + a = 0) → 
  x₁^2 + x₁*x₂ + x₂^2 + a = 0 := by
sorry

end root_sum_squares_plus_product_l1363_136349


namespace limit_sequence_equals_e_to_four_thirds_l1363_136366

open Real

theorem limit_sequence_equals_e_to_four_thirds :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |((3 * n + 1) / (3 * n - 1)) ^ (2 * n + 3) - Real.exp (4 / 3)| < ε :=
by
  sorry

end limit_sequence_equals_e_to_four_thirds_l1363_136366


namespace perfect_square_trinomial_m_value_l1363_136304

theorem perfect_square_trinomial_m_value (m : ℝ) :
  (∃ (a b : ℝ), ∀ y : ℝ, y^2 - m*y + 9 = (a*y + b)^2) →
  m = 6 ∨ m = -6 := by
  sorry

end perfect_square_trinomial_m_value_l1363_136304


namespace hyperbola_C_properties_l1363_136317

/-- Hyperbola C with distance √2 from focus to asymptote -/
structure HyperbolaC where
  b : ℝ
  b_pos : b > 0
  focus_to_asymptote : ∃ (c : ℝ), b * c / Real.sqrt (b^2 + 2) = Real.sqrt 2

/-- Intersection points of line l with hyperbola C -/
structure IntersectionPoints (h : HyperbolaC) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  l_passes_through_2_0 : ∃ (m : ℝ), A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2
  A_on_C : A.1^2 - A.2^2 = 2
  B_on_C : B.1^2 - B.2^2 = 2
  A_B_right_branch : A.1 > 0 ∧ B.1 > 0

/-- Main theorem -/
theorem hyperbola_C_properties (h : HyperbolaC) :
  (∀ (x y : ℝ), x^2/2 - y^2/h.b^2 = 1 ↔ x^2 - y^2 = 2) ∧
  (∀ (i : IntersectionPoints h),
    ∃ (N : ℝ × ℝ), N = (1, 0) ∧
      (i.A.1 - N.1) * (i.B.1 - N.1) + (i.A.2 - N.2) * (i.B.2 - N.2) = -1) :=
sorry

end hyperbola_C_properties_l1363_136317


namespace student_ticket_price_l1363_136328

theorem student_ticket_price 
  (total_sales : ℝ)
  (student_ticket_surplus : ℕ)
  (nonstudent_tickets : ℕ)
  (nonstudent_price : ℝ)
  (h1 : total_sales = 10500)
  (h2 : student_ticket_surplus = 250)
  (h3 : nonstudent_tickets = 850)
  (h4 : nonstudent_price = 9) :
  ∃ (student_price : ℝ), 
    student_price = 2.59 ∧ 
    (nonstudent_tickets : ℝ) * nonstudent_price + 
    ((nonstudent_tickets : ℝ) + (student_ticket_surplus : ℝ)) * student_price = total_sales :=
by sorry

end student_ticket_price_l1363_136328


namespace max_pairs_sum_l1363_136330

theorem max_pairs_sum (n : ℕ) (h : n = 3009) :
  let S := Finset.range n
  ∃ (k : ℕ) (f : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ f → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ f → q ∈ f → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ f → q ∈ f → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ f → p.1 + p.2 ≤ n + 1) ∧
    f.card = k ∧
    k = 1203 ∧
    (∀ (m : ℕ) (g : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ g → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) →
      (∀ (p q : ℕ × ℕ), p ∈ g → q ∈ g → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p q : ℕ × ℕ), p ∈ g → q ∈ g → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      (∀ (p : ℕ × ℕ), p ∈ g → p.1 + p.2 ≤ n + 1) →
      g.card = m →
      m ≤ k) :=
by
  sorry

end max_pairs_sum_l1363_136330


namespace roots_imply_k_range_l1363_136370

/-- The quadratic function f(x) = 2x^2 - kx + k - 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - k * x + k - 3

theorem roots_imply_k_range :
  ∀ k : ℝ,
  (∃ x₁ x₂ : ℝ, 
    f k x₁ = 0 ∧ 0 < x₁ ∧ x₁ < 1 ∧
    f k x₂ = 0 ∧ 1 < x₂ ∧ x₂ < 2) →
  3 < k ∧ k < 5 :=
by sorry

end roots_imply_k_range_l1363_136370


namespace stratified_sampling_proportion_l1363_136362

/-- Represents the number of students to be selected in a stratified sampling -/
def total_sample : ℕ := 45

/-- Represents the total number of male students -/
def male_population : ℕ := 500

/-- Represents the total number of female students -/
def female_population : ℕ := 400

/-- Represents the number of male students selected in the sample -/
def male_sample : ℕ := 25

/-- Calculates the number of female students to be selected in the sample -/
def female_sample : ℕ := (male_sample * female_population) / male_population

/-- Proves that the calculated female sample size maintains the stratified sampling proportion -/
theorem stratified_sampling_proportion :
  female_sample = 20 ∧
  (male_sample : ℚ) / male_population = (female_sample : ℚ) / female_population ∧
  male_sample + female_sample = total_sample :=
sorry

end stratified_sampling_proportion_l1363_136362


namespace ellipse_area_l1363_136345

/-- The area of an ellipse defined by the equation 9x^2 + 16y^2 = 144 is 12π. -/
theorem ellipse_area (x y : ℝ) : 
  (9 * x^2 + 16 * y^2 = 144) → (π * 4 * 3 : ℝ) = 12 * π := by
  sorry

end ellipse_area_l1363_136345


namespace some_number_is_four_l1363_136327

theorem some_number_is_four : ∃ n : ℚ, (27 / n) * 12 - 18 = 3 * 12 + 27 ∧ n = 4 := by sorry

end some_number_is_four_l1363_136327


namespace integer_solutions_system_l1363_136356

theorem integer_solutions_system :
  ∀ x y z : ℤ,
  (x + y = 1 - z ∧ x^3 + y^3 = 1 - z^2) ↔
  ((∃ k : ℤ, x = k ∧ y = -k ∧ z = 1) ∨
   (x = 0 ∧ y = 1 ∧ z = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 0) ∨
   (x = 0 ∧ y = -2 ∧ z = 3) ∨
   (x = -2 ∧ y = 0 ∧ z = 3) ∨
   (x = -2 ∧ y = -3 ∧ z = 6) ∨
   (x = -3 ∧ y = -2 ∧ z = 6)) :=
by sorry

end integer_solutions_system_l1363_136356


namespace m_eq_n_necessary_not_sufficient_l1363_136329

/-- Defines a circle in R^2 --/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b r : ℝ), r > 0 ∧ ∀ (x y : ℝ), f x y = 0 ↔ (x - a)^2 + (y - b)^2 = r^2

/-- The equation mx^2 + ny^2 = 3 --/
def equation (m n : ℝ) (x y : ℝ) : ℝ := m * x^2 + n * y^2 - 3

theorem m_eq_n_necessary_not_sufficient :
  (∀ m n : ℝ, is_circle (equation m n) → m = n) ∧
  (∃ m n : ℝ, m = n ∧ ¬is_circle (equation m n)) :=
sorry

end m_eq_n_necessary_not_sufficient_l1363_136329


namespace M_intersect_N_l1363_136386

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}

theorem M_intersect_N : M ∩ N = {1, 3} := by
  sorry

end M_intersect_N_l1363_136386


namespace max_side_length_11_l1363_136313

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter_24 : a + b + c = 24
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The maximum length of any side in a triangle with integer side lengths and perimeter 24 is 11 -/
theorem max_side_length_11 (t : Triangle) : t.a ≤ 11 ∧ t.b ≤ 11 ∧ t.c ≤ 11 := by
  sorry

end max_side_length_11_l1363_136313


namespace log_sum_lower_bound_l1363_136372

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_lower_bound :
  (∀ a : ℝ, a > 1 → log 2 a + log a 2 ≥ 2) ∧
  (∃ m : ℝ, m < 2 ∧ ∀ a : ℝ, a > 1 → log 2 a + log a 2 ≥ m) :=
by sorry

end log_sum_lower_bound_l1363_136372


namespace ball_drawing_game_l1363_136373

/-- Represents the probability that the last ball is white in the ball-drawing game. -/
def lastBallWhiteProbability (p : ℕ) : ℚ :=
  if p % 2 = 0 then 0 else 1

/-- The ball-drawing game theorem. -/
theorem ball_drawing_game (p q : ℕ) :
  ∀ (pile : ℕ), lastBallWhiteProbability p = if p % 2 = 0 then 0 else 1 := by
  sorry

#check ball_drawing_game

end ball_drawing_game_l1363_136373


namespace only_one_divides_power_minus_one_l1363_136377

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1 ↔ n = 1) := by
  sorry

end only_one_divides_power_minus_one_l1363_136377


namespace consecutive_integers_product_l1363_136339

theorem consecutive_integers_product (a : ℤ) (h : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7 = 20) :
  (a + 6) * a = 391 := by
  sorry

end consecutive_integers_product_l1363_136339


namespace number_equation_solution_l1363_136361

theorem number_equation_solution :
  ∃ (x : ℝ), 7 * x = 3 * x + 12 ∧ x = 3 := by
  sorry

end number_equation_solution_l1363_136361


namespace hall_volume_l1363_136378

/-- A rectangular hall with specific dimensions and area properties -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  height : ℝ
  area_equality : 2 * (length * width) = 2 * (length * height + width * height)

/-- The volume of a rectangular hall is 900 cubic meters -/
theorem hall_volume (hall : RectangularHall) 
  (h_length : hall.length = 15)
  (h_width : hall.width = 10) : 
  hall.length * hall.width * hall.height = 900 := by
  sorry

#check hall_volume

end hall_volume_l1363_136378


namespace anne_cleaning_time_l1363_136384

-- Define the cleaning rates
def bruce_rate : ℝ := sorry
def anne_rate : ℝ := sorry

-- Define the conditions
axiom combined_rate : bruce_rate + anne_rate = 1 / 4
axiom doubled_anne_rate : bruce_rate + 2 * anne_rate = 1 / 3

-- Theorem to prove
theorem anne_cleaning_time :
  1 / anne_rate = 12 := by sorry

end anne_cleaning_time_l1363_136384


namespace angle_C_sides_a_b_max_area_l1363_136323

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 2 ∧ Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A

-- Theorem 1: Angle C
theorem angle_C (t : Triangle) (h : triangle_conditions t) : t.C = π/3 := by
  sorry

-- Theorem 2: Sides a and b
theorem sides_a_b (t : Triangle) (h : triangle_conditions t) 
  (area : (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) : 
  t.a = 2 ∧ t.b = 2 := by
  sorry

-- Theorem 3: Maximum area
theorem max_area (t : Triangle) (h : triangle_conditions t) :
  ∀ (s : Triangle), triangle_conditions s → 
    (1/2) * s.a * s.b * Real.sin s.C ≤ Real.sqrt 3 := by
  sorry

end angle_C_sides_a_b_max_area_l1363_136323


namespace mean_of_three_numbers_l1363_136374

theorem mean_of_three_numbers (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 92 →
  d = 120 →
  b = 60 →
  (a + b + c) / 3 = 82 + 2/3 :=
by sorry

end mean_of_three_numbers_l1363_136374


namespace no_primes_in_range_l1363_136305

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ k ∈ Set.Ioo (n! + 2) (n! + n + 1), ¬ Nat.Prime k := by
  sorry

end no_primes_in_range_l1363_136305


namespace triangle_area_inequality_l1363_136371

/-- The area of a triangle given its side lengths -/
noncomputable def S (a b c : ℝ) : ℝ := sorry

/-- Triangle inequality -/
def is_triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_area_inequality 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : is_triangle a₁ b₁ c₁) 
  (h₂ : is_triangle a₂ b₂ c₂) : 
  Real.sqrt (S a₁ b₁ c₁) + Real.sqrt (S a₂ b₂ c₂) ≤ Real.sqrt (S (a₁ + a₂) (b₁ + b₂) (c₁ + c₂)) :=
sorry

end triangle_area_inequality_l1363_136371


namespace smallest_root_of_unity_order_l1363_136352

open Complex

theorem smallest_root_of_unity_order : ∃ (n : ℕ), n > 0 ∧ 
  (∀ z : ℂ, z^5 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^5 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 16 := by
  sorry

end smallest_root_of_unity_order_l1363_136352


namespace orange_harvest_theorem_l1363_136321

theorem orange_harvest_theorem (daily_harvest : ℕ) (harvest_days : ℕ) 
  (h1 : daily_harvest = 76) (h2 : harvest_days = 63) :
  daily_harvest * harvest_days = 4788 := by
  sorry

end orange_harvest_theorem_l1363_136321


namespace intersection_perpendicular_line_l1363_136342

-- Define the lines l1, l2, and l3
def l1 (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def l2 (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l3 (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (x, y) where
  x := -2
  y := 2

-- Define perpendicularity of lines
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem intersection_perpendicular_line :
  ∃ (m b : ℝ), 
    (l1 P.1 P.2) ∧ 
    (l2 P.1 P.2) ∧ 
    (perpendicular m ((1 : ℝ) / 2)) ∧ 
    (∀ (x y : ℝ), m * x + y + b = 0 ↔ 2 * x + y + 2 = 0) := by
  sorry

end intersection_perpendicular_line_l1363_136342


namespace probability_three_hearts_is_correct_l1363_136351

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of hearts in a standard deck -/
def heartsCount : ℕ := 13

/-- Calculates the probability of drawing three hearts in a row from a standard deck without replacement -/
def probabilityThreeHearts : ℚ :=
  (heartsCount : ℚ) / deckSize *
  ((heartsCount - 1) : ℚ) / (deckSize - 1) *
  ((heartsCount - 2) : ℚ) / (deckSize - 2)

/-- Theorem stating that the probability of drawing three hearts in a row is 26/2025 -/
theorem probability_three_hearts_is_correct :
  probabilityThreeHearts = 26 / 2025 := by
  sorry

end probability_three_hearts_is_correct_l1363_136351


namespace cubic_root_equation_solutions_l1363_136354

theorem cubic_root_equation_solutions :
  let f (x : ℝ) := Real.rpow (18 * x - 2) (1/3) + Real.rpow (16 * x + 2) (1/3) - 6 * Real.rpow x (1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = -1/12 ∨ x = 3/4 := by
  sorry

end cubic_root_equation_solutions_l1363_136354


namespace octagon_diagonals_l1363_136398

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: An octagon has 20 internal diagonals -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l1363_136398


namespace simplify_fraction_l1363_136341

theorem simplify_fraction : (88 : ℚ) / 7744 = 1 / 88 := by
  sorry

end simplify_fraction_l1363_136341


namespace smallest_n_for_m_disjoint_monochromatic_edges_l1363_136300

/-- A two-coloring of a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- Predicate for m pairwise disjoint edges of the same color -/
def HasMDisjointMonochromaticEdges (n m : ℕ) (coloring : TwoColoring n) : Prop :=
  ∃ (edges : Fin m → Fin n × Fin n),
    (∀ i : Fin m, (edges i).1 ≠ (edges i).2) ∧
    (∀ i j : Fin m, i ≠ j → (edges i).1 ≠ (edges j).1 ∧ (edges i).1 ≠ (edges j).2 ∧
                            (edges i).2 ≠ (edges j).1 ∧ (edges i).2 ≠ (edges j).2) ∧
    (∃ c : Fin 2, ∀ i : Fin m, coloring (edges i).1 (edges i).2 = c)

/-- The main theorem -/
theorem smallest_n_for_m_disjoint_monochromatic_edges (m : ℕ) (hm : m > 0) :
  (∀ n : ℕ, n ≥ 3 * m - 1 → ∀ coloring : TwoColoring n, HasMDisjointMonochromaticEdges n m coloring) ∧
  (∀ n : ℕ, n < 3 * m - 1 → ∃ coloring : TwoColoring n, ¬HasMDisjointMonochromaticEdges n m coloring) :=
sorry

end smallest_n_for_m_disjoint_monochromatic_edges_l1363_136300


namespace sqrt_equation_solutions_l1363_136303

theorem sqrt_equation_solutions (x : ℝ) : 
  Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6 ↔ x = 2 ∨ x = -2 := by
  sorry

end sqrt_equation_solutions_l1363_136303


namespace hyperbola_transverse_axis_length_l1363_136388

/-- Given a hyperbola with equation x²/a² - y² = 1 where a > 0 and eccentricity = 2,
    prove that the length of its transverse axis is 2√3/3 -/
theorem hyperbola_transverse_axis_length (a : ℝ) (h1 : a > 0) :
  let e := 2  -- eccentricity
  let c := Real.sqrt (a^2 + 1)  -- focal distance
  e = c / a →  -- definition of eccentricity
  2 * a = 2 * Real.sqrt 3 / 3 :=
by sorry

end hyperbola_transverse_axis_length_l1363_136388


namespace olivers_score_l1363_136364

theorem olivers_score (n : ℕ) (avg_24 : ℚ) (avg_25 : ℚ) (oliver_score : ℚ) :
  n = 25 →
  avg_24 = 76 →
  avg_25 = 78 →
  (n - 1) * avg_24 + oliver_score = n * avg_25 →
  oliver_score = 126 := by
  sorry

end olivers_score_l1363_136364


namespace tagged_fish_in_second_catch_l1363_136348

-- Define the parameters of the problem
def initial_tagged : ℕ := 30
def second_catch : ℕ := 50
def total_fish : ℕ := 750

-- Define the theorem
theorem tagged_fish_in_second_catch :
  ∃ (T : ℕ), (T : ℚ) / second_catch = initial_tagged / total_fish ∧ T = 2 := by
  sorry

end tagged_fish_in_second_catch_l1363_136348


namespace tangent_line_parallel_point_l1363_136331

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^4 - x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 4*x^3 - 1

-- Theorem statement
theorem tangent_line_parallel_point (P : ℝ × ℝ) :
  f' P.1 = 3 →  -- The slope of the tangent line at P is 3
  f P.1 = P.2 → -- P lies on the curve f(x)
  P = (1, 0) := by sorry

end tangent_line_parallel_point_l1363_136331


namespace rons_height_l1363_136340

/-- Proves that Ron's height is 13 feet given the water depth and its relation to Ron's height -/
theorem rons_height (water_depth : ℝ) (h1 : water_depth = 208) 
  (h2 : ∃ (rons_height : ℝ), water_depth = 16 * rons_height) : 
  ∃ (rons_height : ℝ), rons_height = 13 := by
  sorry

end rons_height_l1363_136340


namespace usb_cost_problem_l1363_136368

/-- Given that three identical USBs cost $45, prove that seven such USBs cost $105. -/
theorem usb_cost_problem (cost_of_three : ℝ) (h1 : cost_of_three = 45) : 
  (7 / 3) * cost_of_three = 105 := by
  sorry

end usb_cost_problem_l1363_136368


namespace blue_car_fraction_l1363_136319

theorem blue_car_fraction (total : ℕ) (black : ℕ) : 
  total = 516 →
  black = 86 →
  (total - (total / 2 + black)) / total = 1 / 3 := by
  sorry

end blue_car_fraction_l1363_136319


namespace average_running_distance_l1363_136318

def monday_distance : ℝ := 4.2
def tuesday_distance : ℝ := 3.8
def wednesday_distance : ℝ := 3.6
def thursday_distance : ℝ := 4.4
def number_of_days : ℕ := 4

def total_distance : ℝ := monday_distance + tuesday_distance + wednesday_distance + thursday_distance

theorem average_running_distance :
  total_distance / number_of_days = 4 := by sorry

end average_running_distance_l1363_136318


namespace only_zero_function_satisfies_l1363_136369

/-- A function satisfying the given inequality for all non-zero real x and all real y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x ≠ 0 → f (x^2 + y) ≥ (1/x + 1) * f y

/-- The main theorem stating that the only function satisfying the inequality is the zero function -/
theorem only_zero_function_satisfies :
  ∀ f : ℝ → ℝ, SatisfiesInequality f ↔ ∀ x, f x = 0 := by sorry

end only_zero_function_satisfies_l1363_136369


namespace train_speed_l1363_136336

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 120) (h2 : time = 16) :
  length / time = 7.5 := by
  sorry

end train_speed_l1363_136336


namespace diophantine_equation_solution_l1363_136324

theorem diophantine_equation_solution (a b : ℕ+) 
  (h1 : (b ^ 619 : ℕ) ∣ (a ^ 1000 : ℕ) + 1)
  (h2 : (a ^ 619 : ℕ) ∣ (b ^ 1000 : ℕ) + 1) :
  a = 1 ∧ b = 1 := by
  sorry

end diophantine_equation_solution_l1363_136324


namespace allan_initial_balloons_l1363_136380

/-- The number of balloons Allan initially brought to the park -/
def initial_balloons : ℕ := sorry

/-- The number of balloons Allan bought at the park -/
def bought_balloons : ℕ := 3

/-- The total number of balloons Allan had after buying more -/
def total_balloons : ℕ := 8

/-- Theorem stating that Allan initially brought 5 balloons to the park -/
theorem allan_initial_balloons : 
  initial_balloons = total_balloons - bought_balloons := by sorry

end allan_initial_balloons_l1363_136380


namespace system_solution_conditions_l1363_136301

/-- Given a system of equations:
    a x + b y = c z
    a √(1 - x²) + b √(1 - y²) = c √(1 - z²)
    where x, y, z are real variables,
    prove that for a real solution to exist:
    1. a, b, c must satisfy the triangle inequalities
    2. At least one of a or b must have the same sign as c -/
theorem system_solution_conditions (a b c : ℝ) : 
  (∃ x y z : ℝ, a * x + b * y = c * z ∧ 
   a * Real.sqrt (1 - x^2) + b * Real.sqrt (1 - y^2) = c * Real.sqrt (1 - z^2)) →
  (abs a ≤ abs b + abs c ∧ abs b ≤ abs a + abs c ∧ abs c ≤ abs a + abs b) ∧
  (a * c ≥ 0 ∨ b * c ≥ 0) := by
sorry

end system_solution_conditions_l1363_136301


namespace parabola_chord_length_l1363_136353

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Theorem statement
theorem parabola_chord_length :
  ∀ p : ℝ,
  p > 0 →
  (∃ x y : ℝ, parabola p x y ∧ x = 1 ∧ y = 0) →  -- Focus at (1, 0)
  (∃ A B : ℝ × ℝ,
    parabola p A.1 A.2 ∧ 
    parabola p B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A ≠ B) →
  (∀ x y : ℝ, parabola p x y ↔ y^2 = 2*x) ∧     -- Standard equation
  (∃ A B : ℝ × ℝ,
    parabola p A.1 A.2 ∧ 
    parabola p B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4) -- Chord length
  := by sorry

end parabola_chord_length_l1363_136353


namespace inequality_proof_l1363_136375

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  Real.sqrt (a^3 / (1 + b * c)) + Real.sqrt (b^3 / (1 + a * c)) + Real.sqrt (c^3 / (1 + a * b)) > 2 := by
  sorry

end inequality_proof_l1363_136375
