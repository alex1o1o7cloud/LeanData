import Mathlib

namespace previous_salary_calculation_l724_72451

/-- Represents the salary and commission structure of Tom's new job -/
structure NewJob where
  base_salary : ℝ
  commission_rate : ℝ
  sale_value : ℝ

/-- Calculates the total earnings from the new job given a number of sales -/
def earnings_new_job (job : NewJob) (num_sales : ℝ) : ℝ :=
  job.base_salary + job.commission_rate * job.sale_value * num_sales

/-- Theorem stating that if Tom needs to make at least 266.67 sales to not lose money,
    then his previous job salary was $75,000 -/
theorem previous_salary_calculation (job : NewJob) 
    (h1 : job.base_salary = 45000)
    (h2 : job.commission_rate = 0.15)
    (h3 : job.sale_value = 750)
    (h4 : earnings_new_job job 266.67 ≥ earnings_new_job job 266.66) :
    earnings_new_job job 266.67 = 75000 := by
  sorry

#check previous_salary_calculation

end previous_salary_calculation_l724_72451


namespace max_visible_cubes_l724_72433

/-- Represents a transparent cube made of unit cubes --/
structure TransparentCube where
  size : Nat
  deriving Repr

/-- Calculates the number of visible unit cubes from a single point --/
def visibleUnitCubes (cube : TransparentCube) : Nat :=
  let fullFace := cube.size * cube.size
  let surfaceFaces := 2 * (cube.size * cube.size - (cube.size - 2) * (cube.size - 2))
  let sharedEdges := 3 * cube.size
  fullFace + surfaceFaces - sharedEdges + 1

/-- Theorem stating that the maximum number of visible unit cubes is 181 for a 12x12x12 cube --/
theorem max_visible_cubes (cube : TransparentCube) (h : cube.size = 12) :
  visibleUnitCubes cube = 181 := by
  sorry

#eval visibleUnitCubes { size := 12 }

end max_visible_cubes_l724_72433


namespace distance_from_origin_l724_72491

theorem distance_from_origin (x y : ℝ) (h1 : x > 2) (h2 : x = 15) 
  (h3 : (x - 2)^2 + (y - 7)^2 = 13^2) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt 274 := by
  sorry

end distance_from_origin_l724_72491


namespace rectangle_exists_in_octagon_decomposition_l724_72476

/-- A regular octagon -/
structure RegularOctagon where
  -- Add necessary fields

/-- A parallelogram -/
structure Parallelogram where
  -- Add necessary fields

/-- A decomposition of a regular octagon into parallelograms -/
structure OctagonDecomposition where
  octagon : RegularOctagon
  parallelograms : Finset Parallelogram
  is_valid : Bool  -- Predicate to check if the decomposition is valid

/-- Predicate to check if a parallelogram is a rectangle -/
def is_rectangle (p : Parallelogram) : Prop :=
  sorry

/-- Main theorem: In any valid decomposition of a regular octagon into parallelograms,
    there exists at least one rectangle among the parallelograms -/
theorem rectangle_exists_in_octagon_decomposition (d : OctagonDecomposition) 
    (h : d.is_valid) : ∃ p ∈ d.parallelograms, is_rectangle p :=
  sorry

end rectangle_exists_in_octagon_decomposition_l724_72476


namespace parabola_shift_parabola_transformation_l724_72405

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x - 1)^2 + 2

-- Theorem stating the equivalence of the original parabola after transformation and the shifted parabola
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 2 :=
by
  sorry

-- Theorem stating that the shifted parabola is the result of the described transformations
theorem parabola_transformation :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 2 :=
by
  sorry

end parabola_shift_parabola_transformation_l724_72405


namespace unique_subset_existence_l724_72406

theorem unique_subset_existence : 
  ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (pair : ℤ × ℤ), 
    pair.1 ∈ X ∧ pair.2 ∈ X ∧ pair.1 + 2 * pair.2 = n := by
  sorry

end unique_subset_existence_l724_72406


namespace project_time_ratio_l724_72482

/-- Proves that the ratio of time charged by Pat to Kate is 2:1 given the problem conditions -/
theorem project_time_ratio : 
  ∀ (p k m : ℕ) (r : ℚ),
  p + k + m = 153 →
  p = r * k →
  p = m / 3 →
  m = k + 85 →
  r = 2 := by
sorry

end project_time_ratio_l724_72482


namespace correct_calculation_l724_72465

theorem correct_calculation (x y : ℝ) : 3 * x - (-2 * y + 4) = 3 * x + 2 * y - 4 := by
  sorry

end correct_calculation_l724_72465


namespace x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l724_72407

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x ≤ 1 ∧ x^2 > 1) := by
  sorry

end x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l724_72407


namespace lemonade_percentage_in_solution1_l724_72466

/-- Represents a solution mixture of lemonade and carbonated water -/
structure Solution :=
  (lemonade : ℝ)
  (carbonated_water : ℝ)
  (h_sum : lemonade + carbonated_water = 100)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (proportion1 : ℝ)
  (proportion2 : ℝ)
  (h_prop_sum : proportion1 + proportion2 = 100)

theorem lemonade_percentage_in_solution1
  (s1 : Solution)
  (s2 : Solution)
  (mix : Mixture)
  (h1 : s2.lemonade = 45)
  (h2 : s2.carbonated_water = 55)
  (h3 : mix.solution1 = s1)
  (h4 : mix.solution2 = s2)
  (h5 : mix.proportion1 = 40)
  (h6 : mix.proportion2 = 60)
  (h7 : mix.proportion1 / 100 * s1.carbonated_water + mix.proportion2 / 100 * s2.carbonated_water = 65) :
  s1.lemonade = 20 := by
sorry

end lemonade_percentage_in_solution1_l724_72466


namespace square_plus_abs_zero_implies_both_zero_l724_72420

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) : a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_plus_abs_zero_implies_both_zero_l724_72420


namespace number_plus_two_equals_six_l724_72430

theorem number_plus_two_equals_six :
  ∃ x : ℝ, (2 + x = 6) ∧ (x = 4) := by
  sorry

end number_plus_two_equals_six_l724_72430


namespace ryan_sandwich_slices_l724_72411

/-- The number of sandwiches Ryan wants to make -/
def num_sandwiches : ℕ := 5

/-- The number of slices of bread needed for one sandwich -/
def slices_per_sandwich : ℕ := 3

/-- The total number of slices needed for all sandwiches -/
def total_slices : ℕ := num_sandwiches * slices_per_sandwich

theorem ryan_sandwich_slices : total_slices = 15 := by
  sorry

end ryan_sandwich_slices_l724_72411


namespace smallest_divisors_sum_of_powers_l724_72497

theorem smallest_divisors_sum_of_powers (n a b : ℕ) : 
  (a > 1) →
  (∀ k, 1 < k → k < a → ¬(k ∣ n)) →
  (a ∣ n) →
  (b > a) →
  (b ∣ n) →
  (∀ k, a < k → k < b → ¬(k ∣ n)) →
  (n = a^a + b^b) →
  (n = 260) :=
by sorry

end smallest_divisors_sum_of_powers_l724_72497


namespace x_value_l724_72404

theorem x_value : ∃ x : ℚ, (3 * x + 4) / 5 = 15 ∧ x = 71 / 3 := by sorry

end x_value_l724_72404


namespace bottles_purchased_l724_72412

/-- The number of large bottles purchased -/
def large_bottles : ℕ := 1380

/-- The cost of a large bottle in dollars -/
def large_bottle_cost : ℚ := 175/100

/-- The number of small bottles purchased -/
def small_bottles : ℕ := 690

/-- The cost of a small bottle in dollars -/
def small_bottle_cost : ℚ := 135/100

/-- The average price per bottle in dollars -/
def average_price : ℚ := 16163438256658595/10000000000000000

theorem bottles_purchased :
  (large_bottles * large_bottle_cost + small_bottles * small_bottle_cost) / 
  (large_bottles + small_bottles : ℚ) = average_price := by
  sorry

end bottles_purchased_l724_72412


namespace smallest_three_digit_multiple_of_17_l724_72471

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end smallest_three_digit_multiple_of_17_l724_72471


namespace claire_pets_male_hamster_fraction_l724_72452

theorem claire_pets_male_hamster_fraction :
  ∀ (total_pets gerbils hamsters male_pets male_gerbils male_hamsters : ℕ),
    total_pets = 90 →
    gerbils = 66 →
    total_pets = gerbils + hamsters →
    male_pets = 25 →
    male_gerbils = 16 →
    male_pets = male_gerbils + male_hamsters →
    (male_hamsters : ℚ) / (hamsters : ℚ) = 3/8 :=
by
  sorry

end claire_pets_male_hamster_fraction_l724_72452


namespace gcd_lcm_sum_l724_72458

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 24 18 = 87 := by
  sorry

end gcd_lcm_sum_l724_72458


namespace distance_to_left_focus_l724_72429

/-- A hyperbola with real axis length m and a point P on it -/
structure Hyperbola (m : ℝ) where
  /-- The distance from P to the right focus is m -/
  dist_right_focus : ℝ
  /-- The distance from P to the right focus equals m -/
  dist_right_focus_eq : dist_right_focus = m

/-- The theorem stating that the distance from P to the left focus is 2m -/
theorem distance_to_left_focus (m : ℝ) (h : Hyperbola m) : 
  ∃ (dist_left_focus : ℝ), dist_left_focus = 2 * m := by
  sorry

end distance_to_left_focus_l724_72429


namespace sequence_properties_l724_72495

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := 33 * n - n^2

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℤ := 34 - 2 * n

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) ∧
  (a 1 = 32) ∧
  (∀ n : ℕ, a (n+1) - a n = -2) := by
  sorry

end sequence_properties_l724_72495


namespace quadratic_solution_l724_72490

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9) - 36 = 0) → b = 5 := by
sorry

end quadratic_solution_l724_72490


namespace banquet_food_consumption_l724_72479

/-- The total food consumed at a banquet is at least the product of the minimum number of guests and the maximum food consumed per guest. -/
theorem banquet_food_consumption 
  (max_food_per_guest : ℝ) 
  (min_guests : ℕ) 
  (h1 : max_food_per_guest = 2) 
  (h2 : min_guests = 162) : 
  ℝ := by
  sorry

#eval (2 : ℝ) * 162  -- Expected output: 324

end banquet_food_consumption_l724_72479


namespace range_of_a_l724_72474

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3*a) ↔ -1 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l724_72474


namespace carolyn_silverware_knife_percentage_l724_72436

/-- Represents the composition of a silverware set -/
structure Silverware :=
  (knives : ℕ)
  (forks : ℕ)
  (spoons : ℕ)

/-- Calculates the total number of pieces in a silverware set -/
def Silverware.total (s : Silverware) : ℕ :=
  s.knives + s.forks + s.spoons

/-- Represents a trade of silverware pieces -/
structure Trade :=
  (knives_gained : ℕ)
  (spoons_lost : ℕ)

/-- Applies a trade to a silverware set -/
def Silverware.apply_trade (s : Silverware) (t : Trade) : Silverware :=
  { knives := s.knives + t.knives_gained,
    forks := s.forks,
    spoons := s.spoons - t.spoons_lost }

/-- Calculates the percentage of knives in a silverware set -/
def Silverware.knife_percentage (s : Silverware) : ℚ :=
  (s.knives : ℚ) / (s.total : ℚ) * 100

theorem carolyn_silverware_knife_percentage :
  let initial_set : Silverware := { knives := 6, forks := 12, spoons := 6 * 3 }
  let trade : Trade := { knives_gained := 10, spoons_lost := 6 }
  let final_set := initial_set.apply_trade trade
  final_set.knife_percentage = 40 := by
  sorry

end carolyn_silverware_knife_percentage_l724_72436


namespace beth_candy_counts_l724_72493

def possible_candy_counts (total : ℕ) (anne_min : ℕ) (beth_min : ℕ) (chris_min : ℕ) (chris_max : ℕ) : Set ℕ :=
  {b | ∃ (a c : ℕ), 
    a + b + c = total ∧ 
    a ≥ anne_min ∧ 
    b ≥ beth_min ∧ 
    c ≥ chris_min ∧ 
    c ≤ chris_max}

theorem beth_candy_counts : 
  possible_candy_counts 10 3 2 2 3 = {2, 3, 4, 5} := by
  sorry

end beth_candy_counts_l724_72493


namespace division_expression_equality_l724_72494

theorem division_expression_equality : 
  (1 : ℚ) / 12 / ((1 : ℚ) / 3 - (1 : ℚ) / 4 - (5 : ℚ) / 12) = -(1 : ℚ) / 4 := by
  sorry

end division_expression_equality_l724_72494


namespace shape_count_l724_72437

theorem shape_count (total_shapes : ℕ) (total_edges : ℕ) 
  (h1 : total_shapes = 13) 
  (h2 : total_edges = 47) : 
  ∃ (triangles squares : ℕ),
    triangles + squares = total_shapes ∧ 
    3 * triangles + 4 * squares = total_edges ∧
    triangles = 5 ∧ 
    squares = 8 := by
  sorry

end shape_count_l724_72437


namespace video_game_map_area_l724_72435

-- Define the map dimensions
def map_width : ℝ := 10
def map_length : ℝ := 2

-- Define the area of a rectangle
def rectangle_area (width length : ℝ) : ℝ := width * length

-- Theorem statement
theorem video_game_map_area : rectangle_area map_width map_length = 20 := by
  sorry

end video_game_map_area_l724_72435


namespace peanut_butter_jar_servings_l724_72428

/-- The number of servings in a jar of peanut butter -/
def peanut_butter_servings (jar_contents : ℚ) (serving_size : ℚ) : ℚ :=
  jar_contents / serving_size

theorem peanut_butter_jar_servings :
  let jar_contents : ℚ := 35 + 4/5
  let serving_size : ℚ := 2 + 1/3
  peanut_butter_servings jar_contents serving_size = 15 + 17/35 := by
  sorry

end peanut_butter_jar_servings_l724_72428


namespace investment_percentage_rate_l724_72424

/-- Given an investment scenario, prove the percentage rate of the remaining investment --/
theorem investment_percentage_rate
  (total_investment : ℝ)
  (investment_at_five_percent : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 18000)
  (h2 : investment_at_five_percent = 6000)
  (h3 : total_interest = 660)
  : (total_interest - investment_at_five_percent * 0.05) / (total_investment - investment_at_five_percent) * 100 = 3 := by
  sorry

end investment_percentage_rate_l724_72424


namespace total_dogs_l724_72415

theorem total_dogs (brown : ℕ) (white : ℕ) (black : ℕ)
  (h1 : brown = 20)
  (h2 : white = 10)
  (h3 : black = 15) :
  brown + white + black = 45 := by
  sorry

end total_dogs_l724_72415


namespace inverse_g_at_113_l724_72449

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_g_at_113 : g⁻¹ 113 = 3 := by sorry

end inverse_g_at_113_l724_72449


namespace haley_flash_drive_files_l724_72403

/-- Calculates the number of files remaining on a flash drive after compression and deletion. -/
def files_remaining (music_files : ℕ) (video_files : ℕ) (document_files : ℕ) 
                    (music_compression : ℕ) (video_compression : ℕ) 
                    (deleted_files : ℕ) : ℕ :=
  music_files * music_compression + video_files * video_compression + document_files - deleted_files

/-- Theorem stating the number of files remaining on Haley's flash drive -/
theorem haley_flash_drive_files : 
  files_remaining 27 42 12 2 3 11 = 181 := by
  sorry

end haley_flash_drive_files_l724_72403


namespace prime_sum_problem_l724_72469

theorem prime_sum_problem (m n : ℕ) (hm : Nat.Prime m) (hn : Nat.Prime n) 
  (h : 5 * m + 7 * n = 129) : m + n = 19 ∨ m + n = 25 := by
  sorry

end prime_sum_problem_l724_72469


namespace abs_equation_one_l724_72468

theorem abs_equation_one (x : ℝ) : |3*x - 5| + 4 = 8 ↔ x = 3 ∨ x = 1/3 := by
  sorry

end abs_equation_one_l724_72468


namespace vegetable_ghee_weight_l724_72447

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 395

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 950

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def mixture_ratio : ℚ := 3 / 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3640

theorem vegetable_ghee_weight : 
  weight_a * (mixture_ratio * total_volume / (1 + mixture_ratio)) + 
  weight_b * (total_volume / (1 + mixture_ratio)) = total_weight := by
  sorry

#check vegetable_ghee_weight

end vegetable_ghee_weight_l724_72447


namespace binary_1101001_is_105_and_odd_l724_72402

-- Define the binary number as a list of bits
def binary_number : List Nat := [1, 1, 0, 1, 0, 0, 1]

-- Function to convert binary to decimal
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Theorem statement
theorem binary_1101001_is_105_and_odd :
  (binary_to_decimal binary_number = 105) ∧ (105 % 2 = 1) := by
  sorry

#eval binary_to_decimal binary_number
#eval 105 % 2

end binary_1101001_is_105_and_odd_l724_72402


namespace student_sampling_interval_l724_72498

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The systematic sampling interval for 1000 students with a sample size of 40 is 25 -/
theorem student_sampling_interval :
  systematicSamplingInterval 1000 40 = 25 := by
  sorry

end student_sampling_interval_l724_72498


namespace max_sum_of_squares_l724_72487

theorem max_sum_of_squares (a b c : ℤ) : 
  a + b + c = 3 → a^3 + b^3 + c^3 = 3 → a^2 + b^2 + c^2 ≤ 57 := by
  sorry

end max_sum_of_squares_l724_72487


namespace unused_ribbon_theorem_l724_72400

/-- Represents the pattern of ribbon pieces -/
inductive RibbonPiece
  | two
  | four
  | six
  | eight
  | ten

/-- Returns the length of a ribbon piece in meters -/
def piece_length (p : RibbonPiece) : ℕ :=
  match p with
  | .two => 2
  | .four => 4
  | .six => 6
  | .eight => 8
  | .ten => 10

/-- Represents the pattern of ribbon usage -/
def ribbon_pattern : List RibbonPiece :=
  [.two, .two, .two, .four, .four, .six, .six, .six, .six, .eight, .ten, .ten]

/-- Calculates the unused ribbon length after following the pattern once -/
def unused_ribbon (total_length : ℕ) (pattern : List RibbonPiece) : ℕ :=
  let used := pattern.foldl (fun acc p => acc + piece_length p) 0
  total_length - (used % total_length)

theorem unused_ribbon_theorem :
  unused_ribbon 30 ribbon_pattern = 4 := by sorry

#eval unused_ribbon 30 ribbon_pattern

end unused_ribbon_theorem_l724_72400


namespace quadratic_function_properties_l724_72453

/-- Given a linear function y = cx + 2c, prove that the quadratic function
    y = 0.5c(x + 2)^2 passes through the points (0, 2c) and (-2, 0) -/
theorem quadratic_function_properties (c : ℝ) :
  let f (x : ℝ) := 0.5 * c * (x + 2)^2
  (f 0 = 2 * c) ∧ (f (-2) = 0) := by
sorry

end quadratic_function_properties_l724_72453


namespace ten_factorial_mod_thirteen_l724_72427

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem ten_factorial_mod_thirteen : 
  factorial 10 % 13 = 7 := by
  sorry

end ten_factorial_mod_thirteen_l724_72427


namespace range_of_r_l724_72467

noncomputable def r (x : ℝ) : ℝ := 1 / (1 - x)^3

theorem range_of_r :
  Set.range r = {y : ℝ | y < 0 ∨ y > 0} :=
by sorry

end range_of_r_l724_72467


namespace markup_percentage_l724_72483

theorem markup_percentage (cost selling_price markup : ℝ) : 
  markup = selling_price - cost →
  markup = 0.0909090909090909 * selling_price →
  markup = 0.1 * cost := by
  sorry

end markup_percentage_l724_72483


namespace function_property_l724_72439

theorem function_property (a : ℝ) : 
  (∀ x ∈ Set.Icc (0 : ℝ) 1, 
    ∀ y ∈ Set.Icc (0 : ℝ) 1, 
    ∀ z ∈ Set.Icc (0 : ℝ) 1, 
    (1/2) * a * x^2 - (x - 1) * Real.exp x + 
    (1/2) * a * y^2 - (y - 1) * Real.exp y ≥ 
    (1/2) * a * z^2 - (z - 1) * Real.exp z) →
  a ∈ Set.Icc 1 4 := by
sorry

end function_property_l724_72439


namespace polar_curve_is_line_and_circle_l724_72488

/-- The curve represented by the polar equation ρsin(θ) = sin(2θ) -/
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ * Real.sin θ = Real.sin (2 * θ)

/-- The line part of the curve -/
def line_part (x y : ℝ) : Prop :=
  y = 0

/-- The circle part of the curve -/
def circle_part (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

/-- Theorem stating that the polar curve consists of a line and a circle -/
theorem polar_curve_is_line_and_circle :
  ∀ ρ θ x y : ℝ, polar_curve ρ θ → 
  (∃ ρ' θ', x = ρ' * Real.cos θ' ∧ y = ρ' * Real.sin θ') →
  (line_part x y ∨ circle_part x y) :=
sorry

end polar_curve_is_line_and_circle_l724_72488


namespace triangle_sum_proof_l724_72441

/-- Triangle operation: a + b - 2c --/
def triangle_op (a b c : ℤ) : ℤ := a + b - 2*c

theorem triangle_sum_proof :
  let t1 := triangle_op 3 4 5
  let t2 := triangle_op 6 8 2
  2 * t1 + 3 * t2 = 24 := by
  sorry

end triangle_sum_proof_l724_72441


namespace tim_marbles_l724_72408

/-- Given that Fred has 110 blue marbles and 22 times more blue marbles than Tim,
    prove that Tim has 5 blue marbles. -/
theorem tim_marbles (fred_marbles : ℕ) (ratio : ℕ) (h1 : fred_marbles = 110) (h2 : ratio = 22) :
  fred_marbles / ratio = 5 := by
  sorry

end tim_marbles_l724_72408


namespace modular_exponentiation_l724_72409

theorem modular_exponentiation (m : ℕ) : 
  0 ≤ m ∧ m < 29 ∧ (4 * m) % 29 = 1 → (5^m)^4 % 29 - 3 = 13 := by
  sorry

end modular_exponentiation_l724_72409


namespace jaymee_shara_age_difference_l724_72456

theorem jaymee_shara_age_difference (shara_age jaymee_age : ℕ) 
  (h1 : shara_age = 10) 
  (h2 : jaymee_age = 22) : 
  jaymee_age - 2 * shara_age = 2 := by
  sorry

end jaymee_shara_age_difference_l724_72456


namespace parallelogram_angle_l724_72489

/-- 
Given a parallelogram with the following properties:
- One angle exceeds the other by 40 degrees
- An inscribed circle touches the extended line of the smaller angle
- This touch point forms a triangle exterior to the parallelogram
- The angle at this point is 60 degrees less than double the smaller angle

Prove that the smaller angle of the parallelogram is 70 degrees.
-/
theorem parallelogram_angle (x : ℝ) : 
  x > 0 ∧ 
  x + 40 > x ∧
  x + (x + 40) = 180 ∧
  2 * x - 60 > 0 → 
  x = 70 := by sorry

end parallelogram_angle_l724_72489


namespace power_mod_seven_l724_72417

theorem power_mod_seven : 2^2004 % 7 = 1 := by sorry

end power_mod_seven_l724_72417


namespace probability_of_white_ball_l724_72481

/-- Given a box with white and black balls, calculate the probability of drawing a white ball -/
theorem probability_of_white_ball (white_balls black_balls : ℕ) : 
  white_balls = 5 → black_balls = 6 → 
  (white_balls : ℚ) / (white_balls + black_balls : ℚ) = 5 / 11 := by
  sorry

end probability_of_white_ball_l724_72481


namespace three_fourths_to_fifth_power_l724_72496

theorem three_fourths_to_fifth_power : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end three_fourths_to_fifth_power_l724_72496


namespace triangle_abc_properties_l724_72462

/-- Triangle ABC with vertices A(0,2), B(2,0), and C(-2,-1) -/
structure Triangle where
  A : Prod ℝ ℝ := (0, 2)
  B : Prod ℝ ℝ := (2, 0)
  C : Prod ℝ ℝ := (-2, -1)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about the properties of triangle ABC -/
theorem triangle_abc_properties (t : Triangle) :
  ∃ (l : LineEquation) (area : ℝ),
    -- The line equation of height AH
    (l.a = 4 ∧ l.b = 1 ∧ l.c = -2) ∧
    -- The area of triangle ABC
    area = 5 := by
  sorry

end triangle_abc_properties_l724_72462


namespace log_10_2_bounds_l724_72401

theorem log_10_2_bounds :
  let log_10 (x : ℝ) := Real.log x / Real.log 10
  10^3 = 1000 ∧ 10^4 = 10000 ∧ 2^9 = 512 ∧ 2^14 = 16384 →
  2/7 < log_10 2 ∧ log_10 2 < 1/3 := by sorry

end log_10_2_bounds_l724_72401


namespace first_number_in_sum_l724_72413

theorem first_number_in_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 3.622) 
  (b_eq : b = 0.014) 
  (c_eq : c = 0.458) : 
  a = 3.15 := by
sorry

end first_number_in_sum_l724_72413


namespace triangle_area_l724_72432

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 15√3/4 when b = 7, c = 5, and B = 2π/3 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 7 → c = 5 → B = 2 * π / 3 → 
  (1/2) * b * c * Real.sin B = 15 * Real.sqrt 3 / 4 := by
  sorry

end triangle_area_l724_72432


namespace fish_tagging_problem_l724_72492

/-- The number of fish initially tagged in a pond -/
def initially_tagged (total_fish : ℕ) (later_catch : ℕ) (tagged_in_catch : ℕ) : ℕ :=
  (tagged_in_catch * total_fish) / later_catch

theorem fish_tagging_problem (total_fish : ℕ) (later_catch : ℕ) (tagged_in_catch : ℕ)
  (h1 : total_fish = 1800)
  (h2 : later_catch = 60)
  (h3 : tagged_in_catch = 2)
  (h4 : initially_tagged total_fish later_catch tagged_in_catch = (tagged_in_catch * total_fish) / later_catch) :
  initially_tagged total_fish later_catch tagged_in_catch = 60 :=
by sorry

end fish_tagging_problem_l724_72492


namespace min_value_of_sum_min_value_is_four_fifths_min_value_equality_l724_72461

theorem min_value_of_sum (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b = 2 → 
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 2 → 
  1/(1+a) + 1/(2+2*b) ≤ 1/(1+x) + 1/(2+2*y) :=
by
  sorry

theorem min_value_is_four_fifths (a b : ℝ) :
  a > 0 → b > 0 → a + 2*b = 2 → 
  1/(1+a) + 1/(2+2*b) ≥ 4/5 :=
by
  sorry

theorem min_value_equality (a b : ℝ) :
  a > 0 → b > 0 → a + 2*b = 2 → 
  (1/(1+a) + 1/(2+2*b) = 4/5) ↔ (a = 3/2 ∧ b = 1/4) :=
by
  sorry

end min_value_of_sum_min_value_is_four_fifths_min_value_equality_l724_72461


namespace smallest_odd_digit_multiple_of_9_l724_72446

/-- A function that checks if a number has only odd digits -/
def hasOnlyOddDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

/-- The smallest positive integer less than 10,000 with only odd digits that is a multiple of 9 -/
def smallestOddDigitMultipleOf9 : ℕ := 1117

theorem smallest_odd_digit_multiple_of_9 :
  smallestOddDigitMultipleOf9 < 10000 ∧
  hasOnlyOddDigits smallestOddDigitMultipleOf9 ∧
  smallestOddDigitMultipleOf9 % 9 = 0 ∧
  ∀ n : ℕ, n < 10000 → hasOnlyOddDigits n → n % 9 = 0 → smallestOddDigitMultipleOf9 ≤ n :=
by sorry

#eval smallestOddDigitMultipleOf9

end smallest_odd_digit_multiple_of_9_l724_72446


namespace taxi_charge_theorem_l724_72422

/-- A taxi service with a given initial fee and per-distance charge -/
structure TaxiService where
  initialFee : ℚ
  chargePerIncrement : ℚ
  incrementDistance : ℚ

/-- Calculate the total charge for a given trip distance -/
def totalCharge (service : TaxiService) (distance : ℚ) : ℚ :=
  service.initialFee + service.chargePerIncrement * (distance / service.incrementDistance)

/-- Theorem: The total charge for a 3.6-mile trip with the given taxi service is $5.20 -/
theorem taxi_charge_theorem :
  let service : TaxiService := ⟨2.05, 0.35, 2/5⟩
  totalCharge service (36/10) = 26/5 := by
  sorry


end taxi_charge_theorem_l724_72422


namespace pencils_per_row_l724_72473

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  total_pencils = 25 → num_rows = 5 → total_pencils = num_rows * pencils_per_row → pencils_per_row = 5 := by
  sorry

end pencils_per_row_l724_72473


namespace cubic_factorization_l724_72472

theorem cubic_factorization (x y z : ℝ) :
  x^3 + y^3 + z^3 - 3*x*y*z = (x + y + z) * (x^2 + y^2 + z^2 - x*y - y*z - z*x) := by
  sorry

end cubic_factorization_l724_72472


namespace tangent_line_sum_l724_72421

/-- Given a function f: ℝ → ℝ with a tangent line y = -x + 8 at x = 5,
    prove that f(5) + f'(5) = 2 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : (fun x => -x + 8) = fun x => f 5 + (deriv f 5) * (x - 5)) : 
    f 5 + deriv f 5 = 2 := by
  sorry

end tangent_line_sum_l724_72421


namespace min_value_theorem_inequality_theorem_l724_72499

variable (a b c : ℝ)

-- Define the conditions
def sum_condition (a b c : ℝ) : Prop := a + 2 * b + 3 * c = 6

-- Define the non-zero condition
def non_zero (x : ℝ) : Prop := x ≠ 0

-- Theorem for the first part
theorem min_value_theorem (ha : non_zero a) (hb : non_zero b) (hc : non_zero c) 
  (h_sum : sum_condition a b c) : 
  a^2 + 2 * b^2 + 3 * c^2 ≥ 6 := by sorry

-- Theorem for the second part
theorem inequality_theorem (ha : non_zero a) (hb : non_zero b) (hc : non_zero c) 
  (h_sum : sum_condition a b c) : 
  a^2 / (1 + a) + 2 * b^2 / (3 + b) + 3 * c^2 / (5 + c) ≥ 9/7 := by sorry

end min_value_theorem_inequality_theorem_l724_72499


namespace power_calculation_l724_72419

theorem power_calculation : 
  (27 : ℝ)^3 * 9^2 / 3^17 = 1/81 :=
by
  have h1 : (27 : ℝ) = 3^3 := by sorry
  have h2 : (9 : ℝ) = 3^2 := by sorry
  sorry

end power_calculation_l724_72419


namespace max_b_for_integer_solution_l724_72454

theorem max_b_for_integer_solution : ∃ (b : ℤ), b = 9599 ∧
  (∀ (b' : ℤ), (∃ (x : ℤ), x^2 + b'*x - 9600 = 0 ∧ 10 ∣ x ∧ 12 ∣ x) → b' ≤ b) ∧
  (∃ (x : ℤ), x^2 + b*x - 9600 = 0 ∧ 10 ∣ x ∧ 12 ∣ x) := by
  sorry

end max_b_for_integer_solution_l724_72454


namespace exponential_graph_not_in_second_quadrant_l724_72477

/-- Given a > 1 and b < -1, the graph of y = a^x + b does not intersect the second quadrant -/
theorem exponential_graph_not_in_second_quadrant 
  (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  ∀ x y : ℝ, y = a^x + b → ¬(x < 0 ∧ y > 0) :=
by sorry

end exponential_graph_not_in_second_quadrant_l724_72477


namespace jills_hair_braiding_l724_72410

/-- Given the conditions of Jill's hair braiding for the dance team, 
    prove that each dancer has 5 braids. -/
theorem jills_hair_braiding 
  (num_dancers : ℕ) 
  (time_per_braid : ℕ) 
  (total_time_minutes : ℕ) 
  (h1 : num_dancers = 8)
  (h2 : time_per_braid = 30)
  (h3 : total_time_minutes = 20) :
  (total_time_minutes * 60) / (time_per_braid * num_dancers) = 5 :=
sorry

end jills_hair_braiding_l724_72410


namespace second_meeting_time_l724_72434

/-- The time in seconds for Racing Magic to complete one lap -/
def racing_magic_lap_time : ℕ := 150

/-- The number of laps Charging Bull completes in one hour -/
def charging_bull_laps_per_hour : ℕ := 40

/-- The time in minutes when both vehicles meet at the starting point for the second time -/
def meeting_time : ℕ := 15

/-- Theorem stating that the vehicles meet at the starting point for the second time after 15 minutes -/
theorem second_meeting_time :
  let racing_magic_lap_time_min : ℚ := racing_magic_lap_time / 60
  let charging_bull_lap_time_min : ℚ := 60 / charging_bull_laps_per_hour
  Nat.lcm (Nat.ceil (racing_magic_lap_time_min * 2)) (Nat.ceil (charging_bull_lap_time_min * 2)) / 2 = meeting_time :=
sorry

end second_meeting_time_l724_72434


namespace eighth_grade_gpa_l724_72463

/-- Proves that the average GPA for 8th graders is 91 given the specified conditions -/
theorem eighth_grade_gpa (sixth_grade_gpa seventh_grade_gpa eighth_grade_gpa school_avg_gpa : ℝ) :
  sixth_grade_gpa = 93 →
  seventh_grade_gpa = sixth_grade_gpa + 2 →
  school_avg_gpa = 93 →
  school_avg_gpa = (sixth_grade_gpa + seventh_grade_gpa + eighth_grade_gpa) / 3 →
  eighth_grade_gpa = 91 := by
  sorry

end eighth_grade_gpa_l724_72463


namespace unique_intersection_l724_72442

/-- Parabola C defined by x²=4y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

/-- Line MH passing through points M(t,0) and H(2t,t²) -/
def line_MH (t x y : ℝ) : Prop := y = t*(x - t)

/-- Point H on parabola C -/
def point_H (t : ℝ) : ℝ × ℝ := (2*t, t^2)

theorem unique_intersection (t : ℝ) (h : t ≠ 0) :
  ∀ x y : ℝ, parabola_C x y ∧ line_MH t x y → (x, y) = point_H t :=
sorry

end unique_intersection_l724_72442


namespace fraction_value_l724_72450

theorem fraction_value : (5 * 7) / 10 = 3.5 := by
  sorry

end fraction_value_l724_72450


namespace inverse_101_mod_102_l724_72459

theorem inverse_101_mod_102 : (101⁻¹ : ZMod 102) = 101 := by sorry

end inverse_101_mod_102_l724_72459


namespace quadratic_function_properties_l724_72431

/-- A quadratic function with vertex at (-1, 4) passing through (2, -5) -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 4

theorem quadratic_function_properties :
  (∀ x, f x = -x^2 - 2*x + 3) ∧
  (f (-1/2) = 11/4) ∧
  (∀ x, f x = 3 ↔ x = 0 ∨ x = -2) := by
  sorry


end quadratic_function_properties_l724_72431


namespace download_speed_calculation_l724_72480

theorem download_speed_calculation (file_size : ℝ) (speed_ratio : ℝ) (time_diff : ℝ) :
  file_size = 600 ∧ speed_ratio = 15 ∧ time_diff = 140 →
  ∃ (speed_4g : ℝ) (speed_5g : ℝ),
    speed_5g = speed_ratio * speed_4g ∧
    file_size / speed_4g - file_size / speed_5g = time_diff ∧
    speed_4g = 4 ∧ speed_5g = 60 := by
  sorry

end download_speed_calculation_l724_72480


namespace jason_oranges_l724_72455

theorem jason_oranges (mary_oranges total_oranges : ℕ)
  (h1 : mary_oranges = 14)
  (h2 : total_oranges = 55) :
  total_oranges - mary_oranges = 41 := by
  sorry

end jason_oranges_l724_72455


namespace consecutive_binomial_coefficients_l724_72444

theorem consecutive_binomial_coefficients (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 2 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 3 / 4 →
  n + k = 47 := by
  sorry

end consecutive_binomial_coefficients_l724_72444


namespace simplest_quadratic_radical_l724_72485

def is_simplest_quadratic_radical (x : ℝ → ℝ) (others : List (ℝ → ℝ)) : Prop :=
  ∀ y ∈ others, ∃ k : ℝ, k ≠ 0 ∧ ∀ a : ℝ, (x a) = k * (y a) → k = 1

theorem simplest_quadratic_radical :
  let x : ℝ → ℝ := λ a => Real.sqrt (a^2 + 1)
  let y₁ : ℝ → ℝ := λ _ => Real.sqrt 8
  let y₂ : ℝ → ℝ := λ _ => 1 / Real.sqrt 3
  let y₃ : ℝ → ℝ := λ _ => Real.sqrt 0.5
  is_simplest_quadratic_radical x [y₁, y₂, y₃] :=
sorry

end simplest_quadratic_radical_l724_72485


namespace max_large_chips_l724_72414

theorem max_large_chips (total : ℕ) (is_prime : ℕ → Prop) : 
  total = 54 →
  ∃ (small large prime : ℕ), 
    is_prime prime ∧
    small + large = total ∧
    small = large + prime ∧
    ∀ (l : ℕ), (∃ (s p : ℕ), is_prime p ∧ s + l = total ∧ s = l + p) → l ≤ 26 := by
  sorry

end max_large_chips_l724_72414


namespace positive_Y_value_l724_72440

-- Define the ∆ relation
def triangle (X Y : ℝ) : ℝ := X^2 + 3*Y^2

-- Theorem statement
theorem positive_Y_value :
  ∃ Y : ℝ, Y > 0 ∧ triangle 9 Y = 360 ∧ Y = Real.sqrt 93 := by
  sorry

end positive_Y_value_l724_72440


namespace geometric_product_and_quotient_l724_72443

/-- A sequence is geometric if the ratio of consecutive terms is constant. -/
def IsGeometric (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_product_and_quotient
  (a b : ℕ → ℝ)
  (ha : IsGeometric a)
  (hb : IsGeometric b)
  (hb_nonzero : ∀ n, b n ≠ 0) :
  IsGeometric (fun n ↦ a n * b n) ∧
  IsGeometric (fun n ↦ a n / b n) :=
sorry

end geometric_product_and_quotient_l724_72443


namespace polynomial_simplification_l724_72464

/-- Simplification of polynomial expression -/
theorem polynomial_simplification (x : ℝ) :
  (3 * x^10 + 5 * x^9 + 2 * x^8) + (7 * x^12 - x^10 + 4 * x^9 + x^7 + 6 * x^4 + 9) =
  7 * x^12 + 2 * x^10 + 9 * x^9 + 2 * x^8 + x^7 + 6 * x^4 + 9 := by
  sorry

end polynomial_simplification_l724_72464


namespace rectangle_area_l724_72426

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 120) : l * b = 675 := by
  sorry

end rectangle_area_l724_72426


namespace system_solution_l724_72486

theorem system_solution :
  ∃ (x y z : ℝ),
    (1 / x + 2 / y - 3 / z = 3) ∧
    (4 / x - 1 / y - 2 / z = 5) ∧
    (3 / x + 4 / y + 1 / z = 23) ∧
    (x = 1 / 3) ∧ (y = 1 / 3) ∧ (z = 1 / 2) :=
by
  use 1/3, 1/3, 1/2
  sorry

#check system_solution

end system_solution_l724_72486


namespace base_seven_54321_equals_13539_l724_72470

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_54321_equals_13539 :
  base_seven_to_ten [1, 2, 3, 4, 5] = 13539 := by
  sorry

end base_seven_54321_equals_13539_l724_72470


namespace exchange_result_l724_72484

/-- Represents the number of exchanges between Xiao Zhang and Xiao Li -/
def exchanges : ℕ := 4

/-- Xiao Zhang's initial number of pencils -/
def zhang_initial_pencils : ℕ := 200

/-- Xiao Li's initial number of pens -/
def li_initial_pens : ℕ := 20

/-- Number of pencils Xiao Zhang gives in each exchange -/
def pencils_per_exchange : ℕ := 6

/-- Number of pens Xiao Li gives in each exchange -/
def pens_per_exchange : ℕ := 1

/-- Xiao Zhang's pencils after exchanges -/
def zhang_final_pencils : ℕ := zhang_initial_pencils - exchanges * pencils_per_exchange

/-- Xiao Li's pens after exchanges -/
def li_final_pens : ℕ := li_initial_pens - exchanges * pens_per_exchange

theorem exchange_result : zhang_final_pencils = 11 * li_final_pens := by
  sorry

end exchange_result_l724_72484


namespace simplify_polynomial_l724_72448

theorem simplify_polynomial (x : ℝ) : (3*x)^4 + (3*x)*(x^3) + 2*x^5 = 84*x^4 + 2*x^5 := by
  sorry

end simplify_polynomial_l724_72448


namespace beaus_sons_age_l724_72445

theorem beaus_sons_age (beau_age : ℕ) (sons_age : ℕ) : 
  beau_age = 42 →
  3 * (sons_age - 3) = beau_age - 3 →
  sons_age = 16 := by
sorry

end beaus_sons_age_l724_72445


namespace coefficient_of_y_in_equation3_l724_72475

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := 6*x - 5*y + 3*z = 22
def equation2 (x y z : ℝ) : Prop := 4*x + 8*y - 11*z = 7
def equation3 (x y z : ℝ) : Prop := 5*x - y + 2*z = 12/6

-- Define the sum condition
def sum_condition (x y z : ℝ) : Prop := x + y + z = 10

-- Theorem statement
theorem coefficient_of_y_in_equation3 (x y z : ℝ) 
  (eq1 : equation1 x y z) 
  (eq2 : equation2 x y z) 
  (eq3 : equation3 x y z) 
  (sum : sum_condition x y z) : 
  ∃ (a b c : ℝ), equation3 x y z ↔ a*x + (-1)*y + c*z = b :=
sorry

end coefficient_of_y_in_equation3_l724_72475


namespace daps_equivalent_to_dips_l724_72416

/-- Represents the conversion rate between daps and dops -/
def daps_to_dops : ℚ := 5 / 4

/-- Represents the conversion rate between dops and dips -/
def dops_to_dips : ℚ := 3 / 9

/-- The number of dips we want to convert -/
def target_dips : ℚ := 54

theorem daps_equivalent_to_dips :
  (daps_to_dops * (1 / dops_to_dips) * target_dips : ℚ) = 22.5 := by
  sorry

end daps_equivalent_to_dips_l724_72416


namespace thread_needed_proof_l724_72460

def thread_per_keychain : ℕ := 12
def friends_from_classes : ℕ := 6
def friends_from_clubs : ℕ := friends_from_classes / 2

def total_friends : ℕ := friends_from_classes + friends_from_clubs

theorem thread_needed_proof : 
  thread_per_keychain * total_friends = 108 := by
  sorry

end thread_needed_proof_l724_72460


namespace hyperbola_quadrilateral_area_ratio_max_l724_72418

theorem hyperbola_quadrilateral_area_ratio_max (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x : ℝ), x = a * b / (a^2 + b^2) ∧ ∀ (y : ℝ), y = a * b / (a^2 + b^2) → x ≥ y) →
  a * b / (a^2 + b^2) ≤ 1/2 :=
by sorry

end hyperbola_quadrilateral_area_ratio_max_l724_72418


namespace isosceles_triangle_parallel_lines_l724_72423

theorem isosceles_triangle_parallel_lines (base : ℝ) (line1 line2 : ℝ) : 
  base = 20 →
  line2 > line1 →
  line1 * line1 = (1/3) * base * base →
  line2 * line2 = (2/3) * base * base →
  line2 - line1 = (20 * (Real.sqrt 6 - Real.sqrt 3)) / 3 := by
  sorry

end isosceles_triangle_parallel_lines_l724_72423


namespace triangle_perimeter_l724_72438

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem triangle_perimeter (a b c : ℕ) :
  a = 2 → b = 5 → is_odd c → a + b > c → b + c > a → c + a > b →
  a + b + c = 12 := by
  sorry

end triangle_perimeter_l724_72438


namespace students_liking_both_desserts_l724_72478

theorem students_liking_both_desserts
  (total : ℕ)
  (like_brownies : ℕ)
  (like_ice_cream : ℕ)
  (like_neither : ℕ)
  (h1 : total = 45)
  (h2 : like_brownies = 22)
  (h3 : like_ice_cream = 17)
  (h4 : like_neither = 13) :
  (like_brownies + like_ice_cream) - (total - like_neither) = 7 := by
  sorry

end students_liking_both_desserts_l724_72478


namespace meteorologist_more_reliable_l724_72457

/-- Probability of a clear day -/
def p_clear : ℝ := 0.74

/-- Accuracy of a senator's forecast -/
def p_senator_accuracy : ℝ := sorry

/-- Accuracy of the meteorologist's forecast -/
def p_meteorologist_accuracy : ℝ := 1.5 * p_senator_accuracy

/-- Event that the day is clear -/
def G : Prop := sorry

/-- Event that the first senator predicts a clear day -/
def M₁ : Prop := sorry

/-- Event that the second senator predicts a clear day -/
def M₂ : Prop := sorry

/-- Event that the meteorologist predicts a rainy day -/
def S : Prop := sorry

/-- Probability of an event -/
noncomputable def P : Prop → ℝ := sorry

/-- Conditional probability -/
noncomputable def P_cond (A B : Prop) : ℝ := P (A ∧ B) / P B

theorem meteorologist_more_reliable :
  P_cond (¬G) (S ∧ M₁ ∧ M₂) > P_cond G (S ∧ M₁ ∧ M₂) :=
sorry

end meteorologist_more_reliable_l724_72457


namespace total_spent_calculation_l724_72425

-- Define the prices and quantities
def shirt_price : ℝ := 15.00
def shirt_quantity : ℕ := 4
def pants_price : ℝ := 40.00
def pants_quantity : ℕ := 2
def suit_price : ℝ := 150.00
def suit_quantity : ℕ := 1
def sweater_price : ℝ := 30.00
def sweater_quantity : ℕ := 2
def tie_price : ℝ := 20.00
def tie_quantity : ℕ := 3
def shoes_price : ℝ := 80.00
def shoes_quantity : ℕ := 1

-- Define the discount rates
def shirt_discount : ℝ := 0.20
def pants_discount : ℝ := 0.30
def suit_discount : ℝ := 0.15
def coupon_discount : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Define the theorem
theorem total_spent_calculation :
  let initial_total := shirt_price * shirt_quantity + pants_price * pants_quantity + 
                       suit_price * suit_quantity + sweater_price * sweater_quantity + 
                       tie_price * tie_quantity + shoes_price * shoes_quantity
  let discounted_shirts := shirt_price * shirt_quantity * (1 - shirt_discount)
  let discounted_pants := pants_price * pants_quantity * (1 - pants_discount)
  let discounted_suit := suit_price * suit_quantity * (1 - suit_discount)
  let discounted_total := discounted_shirts + discounted_pants + discounted_suit + 
                          sweater_price * sweater_quantity + tie_price * tie_quantity + 
                          shoes_price * shoes_quantity
  let coupon_applied := discounted_total * (1 - coupon_discount)
  let final_total := coupon_applied * (1 + sales_tax_rate)
  final_total = 407.77 := by
sorry

end total_spent_calculation_l724_72425
