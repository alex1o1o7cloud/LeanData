import Mathlib

namespace NUMINAMATH_CALUDE_room_occupancy_l1176_117645

theorem room_occupancy (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (2 : ℚ) / 3 * chairs ∧ 
  chairs - (2 : ℚ) / 3 * chairs = 8 →
  people = 27 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l1176_117645


namespace NUMINAMATH_CALUDE_boxes_with_neither_l1176_117609

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ)
  (h1 : total = 15)
  (h2 : markers = 10)
  (h3 : crayons = 8)
  (h4 : both = 4) :
  total - (markers + crayons - both) = 1 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l1176_117609


namespace NUMINAMATH_CALUDE_daisy_taller_than_reese_l1176_117612

/-- The heights of three people and their relationships -/
structure Heights where
  daisy : ℝ
  parker : ℝ
  reese : ℝ
  parker_shorter : parker = daisy - 4
  reese_height : reese = 60
  average_height : (daisy + parker + reese) / 3 = 64

/-- Daisy is 8 inches taller than Reese -/
theorem daisy_taller_than_reese (h : Heights) : h.daisy - h.reese = 8 := by
  sorry

end NUMINAMATH_CALUDE_daisy_taller_than_reese_l1176_117612


namespace NUMINAMATH_CALUDE_horner_method_equals_f_at_2_l1176_117601

-- Define the polynomial function
def f (x : ℝ) : ℝ := 8 * x^7 + 5 * x^6 + 3 * x^4 + 2 * x + 1

-- Define Horner's method for this specific polynomial
def horner_method (x : ℝ) : ℝ :=
  ((((((8 * x + 5) * x + 0) * x + 3) * x + 0) * x + 0) * x + 2) * x + 1

-- Theorem statement
theorem horner_method_equals_f_at_2 : 
  horner_method 2 = f 2 ∧ horner_method 2 = 1397 := by sorry

end NUMINAMATH_CALUDE_horner_method_equals_f_at_2_l1176_117601


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l1176_117690

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 7
  let selected_republicans : ℕ := 4
  let selected_democrats : ℕ := 3
  (Nat.choose total_republicans selected_republicans) *
  (Nat.choose total_democrats selected_democrats) = 7350 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l1176_117690


namespace NUMINAMATH_CALUDE_books_returned_percentage_l1176_117679

/-- Calculates the percentage of loaned books that were returned -/
def percentage_returned (initial_books : ℕ) (loaned_books : ℕ) (final_books : ℕ) : ℚ :=
  ((final_books - (initial_books - loaned_books)) : ℚ) / (loaned_books : ℚ) * 100

/-- Theorem stating that the percentage of returned books is 65% -/
theorem books_returned_percentage 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : loaned_books = 20)
  (h3 : final_books = 68) : 
  percentage_returned initial_books loaned_books final_books = 65 := by
  sorry

#eval percentage_returned 75 20 68

end NUMINAMATH_CALUDE_books_returned_percentage_l1176_117679


namespace NUMINAMATH_CALUDE_xiaoxiang_age_problem_l1176_117630

theorem xiaoxiang_age_problem :
  let xiaoxiang_age : ℕ := 5
  let father_age : ℕ := 48
  let mother_age : ℕ := 42
  let years_passed : ℕ := 15
  (father_age + years_passed) + (mother_age + years_passed) = 6 * (xiaoxiang_age + years_passed) :=
by
  sorry

end NUMINAMATH_CALUDE_xiaoxiang_age_problem_l1176_117630


namespace NUMINAMATH_CALUDE_min_distance_and_slope_l1176_117648

-- Define the circle F
def circle_F (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the curve W (trajectory)
def curve_W (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l passing through F(1,0)
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the intersection points
def point_A (k : ℝ) : ℝ × ℝ := sorry
def point_D (k : ℝ) : ℝ × ℝ := sorry
def point_B (k : ℝ) : ℝ × ℝ := sorry
def point_C (k : ℝ) : ℝ × ℝ := sorry

-- Define the distances
def dist_AB (k : ℝ) : ℝ := sorry
def dist_CD (k : ℝ) : ℝ := sorry

-- State the theorem
theorem min_distance_and_slope :
  ∃ (k : ℝ), 
    (∀ (k' : ℝ), dist_AB k + 4 * dist_CD k ≤ dist_AB k' + 4 * dist_CD k') ∧
    dist_AB k + 4 * dist_CD k = 4 ∧
    (k = 2 * Real.sqrt 2 ∨ k = -2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_and_slope_l1176_117648


namespace NUMINAMATH_CALUDE_max_colored_cells_1000_cube_l1176_117675

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  sideLength : n > 0

/-- Represents the maximum number of cells that can be colored on a cube's surface -/
def maxColoredCells (c : Cube n) : ℕ :=
  6 * n^2 - 2 * n^2

theorem max_colored_cells_1000_cube :
  ∀ (c : Cube 1000), maxColoredCells c = 2998000 :=
sorry

end NUMINAMATH_CALUDE_max_colored_cells_1000_cube_l1176_117675


namespace NUMINAMATH_CALUDE_five_dice_not_same_probability_l1176_117604

theorem five_dice_not_same_probability :
  let n : ℕ := 6  -- number of sides on each die
  let k : ℕ := 5  -- number of dice rolled
  let total_outcomes : ℕ := n^k
  let same_number_outcomes : ℕ := n
  let not_same_number_probability : ℚ := 1 - (same_number_outcomes : ℚ) / total_outcomes
  not_same_number_probability = 1295 / 1296 :=
by sorry

end NUMINAMATH_CALUDE_five_dice_not_same_probability_l1176_117604


namespace NUMINAMATH_CALUDE_triangle_existence_l1176_117643

theorem triangle_existence (n : ℕ) (points : Finset (ℝ × ℝ × ℝ)) (segments : Finset ((ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ))) :
  points.card = 2 * n →
  segments.card = n^2 + 1 →
  ∃ (a b c : ℝ × ℝ × ℝ), 
    a ∈ points ∧ b ∈ points ∧ c ∈ points ∧
    (a, b) ∈ segments ∧ (b, c) ∈ segments ∧ (a, c) ∈ segments :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_l1176_117643


namespace NUMINAMATH_CALUDE_min_value_sum_l1176_117615

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b^2)) + (b / (4 * c^3)) + (c / (5 * a^4)) ≥ 1 ∧
  ((a / (3 * b^2)) + (b / (4 * c^3)) + (c / (5 * a^4)) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l1176_117615


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_inning_l1176_117683

/-- Represents a batsman's performance --/
structure BatsmanPerformance where
  innings : ℕ
  runsInLastInning : ℕ
  averageIncrease : ℕ
  boundaries : ℕ
  strikeRate : ℚ

/-- Calculates the average after the last inning --/
def averageAfterLastInning (b : BatsmanPerformance) : ℚ :=
  (b.innings * b.averageIncrease + b.runsInLastInning) / b.innings

/-- Theorem stating the batsman's average after the 12th inning --/
theorem batsman_average_after_12th_inning (b : BatsmanPerformance)
  (h1 : b.innings = 12)
  (h2 : b.runsInLastInning = 60)
  (h3 : b.averageIncrease = 4)
  (h4 : b.boundaries ≥ 8)
  (h5 : b.strikeRate ≥ 130) :
  averageAfterLastInning b = 16 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_inning_l1176_117683


namespace NUMINAMATH_CALUDE_gray_eyed_black_haired_count_l1176_117684

/-- Represents the characteristics of models in a modeling agency. -/
structure ModelAgency where
  total : Nat
  redHaired : Nat
  blackHaired : Nat
  greenEyed : Nat
  grayEyed : Nat
  greenEyedRedHaired : Nat

/-- Conditions for the modeling agency problem. -/
def modelingAgencyConditions : ModelAgency :=
  { total := 60
  , redHaired := 24  -- Derived from total - blackHaired
  , blackHaired := 36
  , greenEyed := 36  -- Derived from total - grayEyed
  , grayEyed := 24
  , greenEyedRedHaired := 22 }

/-- Theorem stating that the number of gray-eyed black-haired models is 10. -/
theorem gray_eyed_black_haired_count (agency : ModelAgency) 
  (h1 : agency = modelingAgencyConditions) : 
  agency.blackHaired + agency.greenEyedRedHaired - agency.greenEyed = 10 := by
  sorry

#check gray_eyed_black_haired_count

end NUMINAMATH_CALUDE_gray_eyed_black_haired_count_l1176_117684


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l1176_117694

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l1176_117694


namespace NUMINAMATH_CALUDE_expression_change_l1176_117673

/-- The change in the expression 2x^2 + 5 when x changes by ±b -/
theorem expression_change (x b : ℝ) (h : b > 0) :
  let f : ℝ → ℝ := λ t => 2 * t^2 + 5
  abs (f (x + b) - f x) = 2 * b * (2 * x + b) ∧
  abs (f (x - b) - f x) = 2 * b * (2 * x + b) :=
by sorry

end NUMINAMATH_CALUDE_expression_change_l1176_117673


namespace NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l1176_117698

theorem ceiling_squared_negative_fraction :
  ⌈(-7/4)^2⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l1176_117698


namespace NUMINAMATH_CALUDE_smallest_n_after_tax_l1176_117613

theorem smallest_n_after_tax : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), m > 0 ∧ (104 * m = 100 * 100 * n)) ∧ 
  (∀ (k : ℕ), k > 0 → k < n → ¬∃ (j : ℕ), j > 0 ∧ (104 * j = 100 * 100 * k)) ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_after_tax_l1176_117613


namespace NUMINAMATH_CALUDE_valid_outfit_count_l1176_117639

/-- The number of colors available for each item -/
def num_colors : ℕ := 6

/-- The number of different types of clothing items -/
def num_items : ℕ := 4

/-- Calculates the total number of outfit combinations without restrictions -/
def total_combinations : ℕ := num_colors ^ num_items

/-- Calculates the number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Calculates the number of outfits where shoes don't match any other item color -/
def valid_shoe_combinations : ℕ := num_colors * num_colors * num_colors * (num_colors - 1)

/-- Calculates the number of outfits where shirt, pants, and hat are the same color, but shoes are different -/
def same_color_except_shoes : ℕ := num_colors * (num_colors - 1) - num_colors

/-- The main theorem stating the number of valid outfit combinations -/
theorem valid_outfit_count : 
  total_combinations - same_color_outfits - valid_shoe_combinations - same_color_except_shoes = 1104 := by
  sorry

end NUMINAMATH_CALUDE_valid_outfit_count_l1176_117639


namespace NUMINAMATH_CALUDE_even_decreasing_function_inequality_l1176_117665

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on (0, +∞) if for all x, y ∈ (0, +∞),
    x < y implies f(x) > f(y) -/
def IsDecreasingOnPositiveReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x > f y

theorem even_decreasing_function_inequality
  (f : ℝ → ℝ)
  (heven : IsEven f)
  (hdecr : IsDecreasingOnPositiveReals f) :
  f (-5) < f (-4) ∧ f (-4) < f 3 :=
sorry

end NUMINAMATH_CALUDE_even_decreasing_function_inequality_l1176_117665


namespace NUMINAMATH_CALUDE_age_sum_problem_l1176_117674

theorem age_sum_problem (a b c : ℕ+) (h1 : a = b) (h2 : a > c) (h3 : a * b * c = 144) :
  a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_problem_l1176_117674


namespace NUMINAMATH_CALUDE_cos_double_angle_special_l1176_117627

/-- Given an angle θ formed by the positive x-axis and a line passing through
    the origin and the point (-3, 4), prove that cos(2θ) = -7/25 -/
theorem cos_double_angle_special (θ : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos θ = -3 ∧ r * Real.sin θ = 4) → 
  Real.cos (2 * θ) = -7/25 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_l1176_117627


namespace NUMINAMATH_CALUDE_solution_system_equations_l1176_117663

theorem solution_system_equations :
  ∀ x y : ℝ,
    x > 0 ∧ y > 0 →
    x - 3 * Real.sqrt (x * y) - 2 * Real.sqrt (x / y) = 0 →
    x^2 * y^2 + x^4 = 82 →
    ((x = 3 ∧ y = 1/3) ∨ (x = Real.rpow 66 (1/4) ∧ y = 4 / Real.rpow 66 (1/4))) :=
by
  sorry

#check solution_system_equations

end NUMINAMATH_CALUDE_solution_system_equations_l1176_117663


namespace NUMINAMATH_CALUDE_people_in_line_l1176_117692

theorem people_in_line (total : ℕ) (left : ℕ) (right : ℕ) : 
  total = 11 → left = 5 → right = total - left - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_people_in_line_l1176_117692


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1176_117670

/-- The speed of a boat in still water, given that the time taken to row upstream
    is twice the time taken to row downstream, and the speed of the stream is 12 kmph. -/
theorem boat_speed_in_still_water : ∃ (V_b : ℝ),
  (∀ (t : ℝ), t > 0 → (V_b + 12) * t = (V_b - 12) * (2 * t)) ∧ V_b = 36 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1176_117670


namespace NUMINAMATH_CALUDE_candy_mixture_problem_l1176_117699

/-- Candy mixture problem -/
theorem candy_mixture_problem (x : ℝ) :
  (64 * 2 + x * 3 = (64 + x) * 2.2) →
  (64 + x = 80) := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_problem_l1176_117699


namespace NUMINAMATH_CALUDE_arithmetic_sequence_y_value_l1176_117625

/-- Given an arithmetic sequence with first three terms y + 1, 3y - 2, and 9 - 2y, prove that y = 2 -/
theorem arithmetic_sequence_y_value (y : ℝ) : 
  (∃ d : ℝ, (3*y - 2) - (y + 1) = d ∧ (9 - 2*y) - (3*y - 2) = d) → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_y_value_l1176_117625


namespace NUMINAMATH_CALUDE_tree_height_calculation_l1176_117636

/-- Given a flagpole and a tree, calculate the height of the tree using similar triangles -/
theorem tree_height_calculation (flagpole_height flagpole_shadow tree_shadow : ℝ) 
  (h1 : flagpole_height = 4)
  (h2 : flagpole_shadow = 6)
  (h3 : tree_shadow = 12) :
  (flagpole_height / flagpole_shadow) * tree_shadow = 8 :=
by sorry

end NUMINAMATH_CALUDE_tree_height_calculation_l1176_117636


namespace NUMINAMATH_CALUDE_triangle_sin_a_l1176_117646

theorem triangle_sin_a (A B C : ℝ) (a b c : ℝ) (h : ℝ) : 
  B = π / 4 →
  h = c / 3 →
  (1/2) * a * h = (1/2) * a * c * Real.sin B →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  Real.sin A = a * Real.sin B / b →
  Real.sin A = 3 * Real.sqrt 10 / 10 :=
sorry

end NUMINAMATH_CALUDE_triangle_sin_a_l1176_117646


namespace NUMINAMATH_CALUDE_fraction_decomposition_l1176_117616

theorem fraction_decomposition (n : ℕ) (hn : n > 0) :
  (∃ (a b : ℕ), a ≠ b ∧ 3 / (5 * n) = 1 / a + 1 / b) ∧
  ((∃ (x : ℤ), 3 / (5 * n) = 1 / x + 1 / x) ↔ ∃ (k : ℕ), n = 3 * k) ∧
  (n > 1 → ∃ (c d : ℕ), 3 / (5 * n) = 1 / c - 1 / d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l1176_117616


namespace NUMINAMATH_CALUDE_yellow_candy_probability_l1176_117672

theorem yellow_candy_probability (p_red p_orange p_yellow : ℝ) : 
  p_red = 0.25 →
  p_orange = 0.35 →
  p_red + p_orange + p_yellow = 1 →
  p_yellow = 0.4 := by
sorry

end NUMINAMATH_CALUDE_yellow_candy_probability_l1176_117672


namespace NUMINAMATH_CALUDE_consecutive_numbers_divisibility_l1176_117662

theorem consecutive_numbers_divisibility (n : ℕ) :
  n ≥ 4 ∧
  n ∣ ((n - 3) * (n - 2) * (n - 1)) →
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_divisibility_l1176_117662


namespace NUMINAMATH_CALUDE_closest_fraction_l1176_117605

def medals_won : ℚ := 35 / 225

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (x : ℚ), x ∈ options ∧ 
  ∀ (y : ℚ), y ∈ options → |x - medals_won| ≤ |y - medals_won| ∧
  x = 1/6 :=
sorry

end NUMINAMATH_CALUDE_closest_fraction_l1176_117605


namespace NUMINAMATH_CALUDE_murtha_pebbles_after_20_days_l1176_117626

def pebbles_collected (n : ℕ) : ℕ := n + 1

def pebbles_given_away (n : ℕ) : ℕ := if n % 5 = 0 then 3 else 0

def total_pebbles (days : ℕ) : ℕ :=
  2 + (Finset.range (days - 1)).sum pebbles_collected - (Finset.range days).sum pebbles_given_away

theorem murtha_pebbles_after_20_days :
  total_pebbles 20 = 218 := by sorry

end NUMINAMATH_CALUDE_murtha_pebbles_after_20_days_l1176_117626


namespace NUMINAMATH_CALUDE_min_value_of_complex_l1176_117617

open Complex

theorem min_value_of_complex (z : ℂ) (h : abs (z + I) + abs (z - I) = 2) :
  (∀ w : ℂ, abs (w + I) + abs (w - I) = 2 → abs (z + I + 1) ≤ abs (w + I + 1)) ∧
  (∃ z₀ : ℂ, abs (z₀ + I) + abs (z₀ - I) = 2 ∧ abs (z₀ + I + 1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_complex_l1176_117617


namespace NUMINAMATH_CALUDE_investment_calculation_l1176_117610

/-- Given two investors p and q with an investment ratio of 4:5, 
    where q invests 65000, prove that p's investment is 52000. -/
theorem investment_calculation (p q : ℕ) : 
  (p : ℚ) / q = 4 / 5 → q = 65000 → p = 52000 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l1176_117610


namespace NUMINAMATH_CALUDE_range_of_a_l1176_117660

-- Define the propositions p and q
def p (x : ℝ) : Prop := 1/2 ≤ x ∧ x ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

-- Define the condition about the relationship between p and q
def condition (a : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x, ¬(p x) ∧ q x a)

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, condition a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1176_117660


namespace NUMINAMATH_CALUDE_batsman_sixes_l1176_117624

def total_runs : ℕ := 120
def boundaries : ℕ := 3
def boundary_value : ℕ := 4
def six_value : ℕ := 6

theorem batsman_sixes :
  ∃ (sixes : ℕ),
    sixes * six_value + boundaries * boundary_value + (total_runs / 2) = total_runs ∧
    sixes = 8 := by
  sorry

end NUMINAMATH_CALUDE_batsman_sixes_l1176_117624


namespace NUMINAMATH_CALUDE_min_value_collinear_points_l1176_117628

/-- Given points A(3,-1), B(x,y), and C(0,1) are collinear, and x > 0, y > 0, 
    the minimum value of (3/x + 2/y) is 8 -/
theorem min_value_collinear_points (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (y + 1) / (x - 3) = 2 / (-3)) : 
  (∀ a b : ℝ, a > 0 → b > 0 → (y + 1) / (x - 3) = 2 / (-3) → 3 / x + 2 / y ≤ 3 / a + 2 / b) → 
  3 / x + 2 / y = 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_collinear_points_l1176_117628


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l1176_117685

def numbers : List Nat := [18, 24, 36]

theorem gcf_lcm_sum (C D : Nat) (hC : C = Nat.gcd 18 (Nat.gcd 24 36)) 
  (hD : D = Nat.lcm 18 (Nat.lcm 24 36)) : C + D = 78 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l1176_117685


namespace NUMINAMATH_CALUDE_polygon_division_existence_l1176_117619

/-- A polygon represented by a list of points in 2D space -/
def Polygon : Type := List (ℝ × ℝ)

/-- A line segment represented by its two endpoints -/
def LineSegment : Type := (ℝ × ℝ) × (ℝ × ℝ)

/-- Function to check if a line segment divides a polygon into two equal-area parts -/
def divides_equally (p : Polygon) (l : LineSegment) : Prop := sorry

/-- Function to check if a line segment bisects a side of a polygon -/
def bisects_side (p : Polygon) (l : LineSegment) : Prop := sorry

/-- Function to check if a line segment divides a side of a polygon in 1:2 ratio -/
def divides_side_in_ratio (p : Polygon) (l : LineSegment) : Prop := sorry

/-- Function to check if a polygon is convex -/
def is_convex (p : Polygon) : Prop := sorry

theorem polygon_division_existence :
  ∃ (p : Polygon) (l : LineSegment), 
    divides_equally p l ∧ 
    bisects_side p l ∧ 
    divides_side_in_ratio p l ∧
    is_convex p :=
sorry

end NUMINAMATH_CALUDE_polygon_division_existence_l1176_117619


namespace NUMINAMATH_CALUDE_unchanged_temperature_count_is_219_l1176_117677

/-- The count of integer Fahrenheit temperatures between 32 and 2000 (inclusive) 
    that remain unchanged after the specified conversion process -/
def unchangedTemperatureCount : ℕ :=
  let minTemp := 32
  let maxTemp := 2000
  (maxTemp - minTemp) / 9 + 1

theorem unchanged_temperature_count_is_219 : 
  unchangedTemperatureCount = 219 := by
  sorry

end NUMINAMATH_CALUDE_unchanged_temperature_count_is_219_l1176_117677


namespace NUMINAMATH_CALUDE_two_numbers_with_sum_and_gcd_lcm_sum_l1176_117656

theorem two_numbers_with_sum_and_gcd_lcm_sum (a b : ℕ) : 
  a + b = 60 ∧ 
  Nat.gcd a b + Nat.lcm a b = 84 → 
  (a = 24 ∧ b = 36) ∨ (a = 36 ∧ b = 24) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_with_sum_and_gcd_lcm_sum_l1176_117656


namespace NUMINAMATH_CALUDE_coin_toss_sequences_coin_toss_theorem_l1176_117608

/-- The number of ways to place n indistinguishable balls into k distinguishable urns -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) n

/-- The number of different sequences of 20 coin tosses with specific subsequence counts -/
theorem coin_toss_sequences : ℕ := 
  let hh_placements := stars_and_bars 3 4
  let tt_placements := stars_and_bars 7 5
  hh_placements * tt_placements

/-- The main theorem stating the number of valid sequences -/
theorem coin_toss_theorem : coin_toss_sequences = 6600 := by sorry

end NUMINAMATH_CALUDE_coin_toss_sequences_coin_toss_theorem_l1176_117608


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1176_117621

theorem algebraic_expression_equality (x y : ℝ) :
  2 * x - y + 1 = 3 → 4 * x - 2 * y + 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1176_117621


namespace NUMINAMATH_CALUDE_price_after_discounts_l1176_117682

-- Define the discount rates
def discount1 : ℚ := 20 / 100
def discount2 : ℚ := 10 / 100
def discount3 : ℚ := 5 / 100

-- Define the original and final prices
def originalPrice : ℚ := 10000
def finalPrice : ℚ := 6800

-- Theorem statement
theorem price_after_discounts :
  originalPrice * (1 - discount1) * (1 - discount2) * (1 - discount3) = finalPrice := by
  sorry

end NUMINAMATH_CALUDE_price_after_discounts_l1176_117682


namespace NUMINAMATH_CALUDE_circle_radius_condition_l1176_117614

theorem circle_radius_condition (x y c : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 - 6*y + c = 0 → (x + 4)^2 + (y - 3)^2 = 25) → c = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_condition_l1176_117614


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1176_117651

theorem diophantine_equation_solutions :
  ∀ x y : ℕ, 2 * y^2 - x * y - x^2 + 2 * y + 7 * x - 84 = 0 ↔ (x = 1 ∧ y = 6) ∨ (x = 14 ∧ y = 13) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1176_117651


namespace NUMINAMATH_CALUDE_simplify_expression_l1176_117622

theorem simplify_expression : 
  1 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 1 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1176_117622


namespace NUMINAMATH_CALUDE_m_range_theorem_l1176_117632

/-- Proposition P: The equation (x^2)/(m^2) + (y^2)/(2m+8) = 1 represents an ellipse with foci on the x-axis -/
def P (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / m^2 + y^2 / (2*m + 8) = 1 ∧ m^2 > 2*m + 8 ∧ 2*m + 8 > 0

/-- Proposition Q: The curve y = x^2 + (2m-3)x + 1/4 intersects the x-axis at two distinct points -/
def Q (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (2*m - 3)*x₁ + 1/4 = 0 ∧ x₂^2 + (2*m - 3)*x₂ + 1/4 = 0

/-- The range of m given the conditions -/
def range_of_m (m : ℝ) : Prop :=
  m ≤ -4 ∨ (-2 ≤ m ∧ m < 1) ∨ (2 < m ∧ m ≤ 4)

theorem m_range_theorem (m : ℝ) :
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → range_of_m m := by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1176_117632


namespace NUMINAMATH_CALUDE_sequence_nth_term_l1176_117607

/-- Given a sequence {a_n} where the differences between successive terms form
    a geometric sequence with first term 1 and common ratio r, 
    prove that the nth term of the sequence is (1-r^(n-1))/(1-r) -/
theorem sequence_nth_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n : ℕ, a (n+1) - a n = r^(n-1)) :
  ∀ n : ℕ, a n = (1 - r^(n-1)) / (1 - r) :=
sorry

end NUMINAMATH_CALUDE_sequence_nth_term_l1176_117607


namespace NUMINAMATH_CALUDE_square_side_lengths_average_l1176_117642

theorem square_side_lengths_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 36) (h₃ : a₃ = 64) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 19 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_lengths_average_l1176_117642


namespace NUMINAMATH_CALUDE_article_cost_l1176_117688

theorem article_cost (selling_price1 selling_price2 : ℝ) (percentage_diff : ℝ) :
  selling_price1 = 350 →
  selling_price2 = 340 →
  percentage_diff = 0.05 →
  (selling_price1 - selling_price2) / (selling_price2 - (selling_price1 - selling_price2) / percentage_diff) = percentage_diff →
  selling_price1 - (selling_price1 - selling_price2) / percentage_diff = 140 :=
by sorry

end NUMINAMATH_CALUDE_article_cost_l1176_117688


namespace NUMINAMATH_CALUDE_business_investment_l1176_117637

theorem business_investment (p q : ℕ) (h1 : q = 15000) (h2 : p / q = 4) : p = 60000 := by
  sorry

end NUMINAMATH_CALUDE_business_investment_l1176_117637


namespace NUMINAMATH_CALUDE_fraction_equals_one_l1176_117676

theorem fraction_equals_one (x : ℝ) : x ≠ 3 → ((2 * x - 7) / (x - 3) = 1 ↔ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_one_l1176_117676


namespace NUMINAMATH_CALUDE_marks_wage_proof_l1176_117634

/-- Mark's hourly wage before the raise -/
def pre_raise_wage : ℝ := 40

/-- Mark's weekly work hours -/
def weekly_hours : ℝ := 40

/-- Mark's raise percentage -/
def raise_percentage : ℝ := 0.05

/-- Mark's weekly expenses -/
def weekly_expenses : ℝ := 700

/-- Mark's leftover money per week -/
def weekly_leftover : ℝ := 980

theorem marks_wage_proof :
  pre_raise_wage * weekly_hours * (1 + raise_percentage) = weekly_expenses + weekly_leftover :=
by sorry

end NUMINAMATH_CALUDE_marks_wage_proof_l1176_117634


namespace NUMINAMATH_CALUDE_house_painting_time_l1176_117640

/-- Given that 12 women can paint a house in 6 days, prove that 18 women 
    working at the same rate can paint the same house in 4 days. -/
theorem house_painting_time 
  (women_rate : ℝ → ℝ → ℝ) -- Function that takes number of women and days, returns houses painted
  (h1 : women_rate 12 6 = 1) -- 12 women paint 1 house in 6 days
  (h2 : ∀ w d, women_rate w d = w * d * (women_rate 1 1)) -- Linear relationship
  : women_rate 18 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_house_painting_time_l1176_117640


namespace NUMINAMATH_CALUDE_johns_age_l1176_117631

theorem johns_age (john dad : ℕ) : 
  john = dad - 30 →
  john + dad = 80 →
  john = 25 := by sorry

end NUMINAMATH_CALUDE_johns_age_l1176_117631


namespace NUMINAMATH_CALUDE_greatest_common_remainder_l1176_117635

theorem greatest_common_remainder (a b c : ℕ) (h1 : a = 41) (h2 : b = 71) (h3 : c = 113) :
  ∃ (d : ℕ), d > 0 ∧ 
  (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
  (∀ (k : ℕ), k > 0 → (∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s) → k ≤ d) ∧
  d = Nat.gcd (b - a) (Nat.gcd (c - b) (c - a)) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_remainder_l1176_117635


namespace NUMINAMATH_CALUDE_curve_translation_l1176_117668

-- Define a function representing the original curve
variable (f : ℝ → ℝ)

-- Define the translation
def translate (curve : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ :=
  fun x ↦ curve (x - h) + k

-- Theorem statement
theorem curve_translation (f : ℝ → ℝ) :
  ∃ h k : ℝ, 
    (translate f h k 2 = 3) ∧ 
    (translate f h k = fun x ↦ f (x - 1) + 2) ∧
    (h = 1) ∧ (k = 2) := by
  sorry


end NUMINAMATH_CALUDE_curve_translation_l1176_117668


namespace NUMINAMATH_CALUDE_other_x_intercept_l1176_117697

/-- Given a quadratic function with vertex (5, -3) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = -3 + a * (x - 5)^2) →  -- vertex form
  (a * 1^2 + b * 1 + c = 0) →                        -- x-intercept at (1, 0)
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9       -- other x-intercept at 9
  := by sorry

end NUMINAMATH_CALUDE_other_x_intercept_l1176_117697


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1176_117671

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0) ↔ (1 ≤ m ∧ m < 9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1176_117671


namespace NUMINAMATH_CALUDE_pencil_price_l1176_117657

theorem pencil_price (num_pens num_pencils total_spent pen_avg_price : ℝ) 
  (h1 : num_pens = 30)
  (h2 : num_pencils = 75)
  (h3 : total_spent = 450)
  (h4 : pen_avg_price = 10) :
  (total_spent - num_pens * pen_avg_price) / num_pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_l1176_117657


namespace NUMINAMATH_CALUDE_rectangle_area_l1176_117602

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 6 → 
  ratio = 3 → 
  (2 * r) * (ratio * 2 * r) = 432 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1176_117602


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l1176_117641

theorem base_2_representation_of_123 :
  ∃ (a b c d e f g : ℕ),
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1) ∧
    123 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l1176_117641


namespace NUMINAMATH_CALUDE_BA_equals_AB_l1176_117649

variable {α : Type*} [CommRing α]

def matrix_eq (A B : Matrix (Fin 2) (Fin 2) α) : Prop :=
  ∀ i j, A i j = B i j

theorem BA_equals_AB (A B : Matrix (Fin 2) (Fin 2) α) 
  (h1 : A + B = A * B)
  (h2 : matrix_eq (A * B) !![5, 2; -2, 4]) :
  matrix_eq (B * A) !![5, 2; -2, 4] := by
  sorry

end NUMINAMATH_CALUDE_BA_equals_AB_l1176_117649


namespace NUMINAMATH_CALUDE_joan_seashells_l1176_117695

/-- The number of seashells Joan has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proof that Joan has 16 seashells after giving 63 away from her initial 79 -/
theorem joan_seashells : remaining_seashells 79 63 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l1176_117695


namespace NUMINAMATH_CALUDE_feifei_arrival_time_l1176_117623

/-- Represents the speed of an entity -/
structure Speed :=
  (value : ℝ)

/-- Represents a distance -/
structure Distance :=
  (value : ℝ)

/-- Represents a time duration in minutes -/
structure Duration :=
  (minutes : ℝ)

/-- Represents the scenario of Feifei walking to school -/
structure WalkToSchool :=
  (feifei_speed : Speed)
  (dog_speed : Speed)
  (first_catchup : Distance)
  (second_catchup : Distance)
  (total_distance : Distance)
  (dog_start_delay : Duration)

/-- The theorem stating that Feifei arrives at school 18 minutes after starting -/
theorem feifei_arrival_time (scenario : WalkToSchool) 
  (h1 : scenario.dog_speed.value = 3 * scenario.feifei_speed.value)
  (h2 : scenario.first_catchup.value = 200)
  (h3 : scenario.second_catchup.value = 400)
  (h4 : scenario.total_distance.value = 800)
  (h5 : scenario.dog_start_delay.minutes = 3) :
  ∃ (arrival_time : Duration), arrival_time.minutes = 18 :=
sorry

end NUMINAMATH_CALUDE_feifei_arrival_time_l1176_117623


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_slope_product_l1176_117666

/-- An ellipse passing through (2,0) with eccentricity √3/2 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_eq : a^2 = 4
  h_ecc : (a^2 - b^2) / a^2 = 3/4

/-- A line passing through (1,0) with non-zero slope -/
structure Line where
  k : ℝ
  h_k_nonzero : k ≠ 0

/-- The theorem statement -/
theorem ellipse_line_intersection_slope_product (C : Ellipse) (l : Line) :
  ∃ k' : ℝ, l.k * k' = -1/4 := by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_slope_product_l1176_117666


namespace NUMINAMATH_CALUDE_definite_integral_exp_minus_2x_l1176_117686

theorem definite_integral_exp_minus_2x : 
  ∫ x in (0: ℝ)..1, (Real.exp x - 2 * x) = Real.exp 1 - 2 := by sorry

end NUMINAMATH_CALUDE_definite_integral_exp_minus_2x_l1176_117686


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_11_l1176_117606

theorem greatest_two_digit_multiple_of_11 : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 11 ∣ n → n ≤ 99 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_11_l1176_117606


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l1176_117664

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- Theorem stating that f satisfies all the given conditions
theorem f_satisfies_conditions :
  -- Condition 1: The domain of f is ℝ (implicit in the definition of f)
  -- Condition 2: For any x₁, x₂ ∈ ℝ, if x₁ + x₂ ≠ 0, then f(x₁) + f(x₂) = 0
  (∀ x₁ x₂ : ℝ, x₁ + x₂ ≠ 0 → f x₁ + f x₂ = 0) ∧
  -- Condition 3: For any x ∈ ℝ, if t > 0, then f(x + t) > f(x)
  (∀ x t : ℝ, t > 0 → f (x + t) > f x) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l1176_117664


namespace NUMINAMATH_CALUDE_round_trip_speed_calculation_l1176_117678

/-- Proves that given a round trip of 240 miles with a total travel time of 5.4 hours,
    where the return trip speed is 50 miles per hour, the outbound trip speed is 40 miles per hour. -/
theorem round_trip_speed_calculation (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) :
  total_distance = 240 →
  total_time = 5.4 →
  return_speed = 50 →
  ∃ (outbound_speed : ℝ),
    outbound_speed = 40 ∧
    total_time = (total_distance / 2) / outbound_speed + (total_distance / 2) / return_speed :=
by sorry

end NUMINAMATH_CALUDE_round_trip_speed_calculation_l1176_117678


namespace NUMINAMATH_CALUDE_pyramid_circumscribed_sphere_area_l1176_117654

theorem pyramid_circumscribed_sphere_area :
  ∀ (a b c : ℝ),
    a = 1 →
    b = Real.sqrt 6 →
    c = 3 →
    (∃ (r : ℝ), r * r = (a * a + b * b + c * c) / 4 ∧
      4 * Real.pi * r * r = 16 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_circumscribed_sphere_area_l1176_117654


namespace NUMINAMATH_CALUDE_min_sum_at_6_l1176_117680

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = -14
  sum_of_5th_6th : a 5 + a 6 = -4
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_of_first_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: The sum of the first n terms takes its minimum value when n = 6 -/
theorem min_sum_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≠ 0 → sum_of_first_n_terms seq 6 ≤ sum_of_first_n_terms seq n := by
  sorry

end NUMINAMATH_CALUDE_min_sum_at_6_l1176_117680


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l1176_117669

-- Define the concept of an angle being in a specific quadrant
def in_second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 90 < α ∧ α < k * 360 + 180

def in_first_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 < α ∧ α < n * 360 + 90

def in_third_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 + 180 < α ∧ α < n * 360 + 270

-- State the theorem
theorem half_angle_quadrant (α : Real) :
  in_second_quadrant α → (in_first_quadrant (α/2) ∨ in_third_quadrant (α/2)) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l1176_117669


namespace NUMINAMATH_CALUDE_max_cars_quotient_l1176_117693

/-- Represents the maximum number of cars that can pass a sensor in one hour -/
def N : ℕ := 4000

/-- The length of a car in meters -/
def car_length : ℝ := 5

/-- The safety rule factor: number of car lengths per 20 km/h of speed -/
def safety_factor : ℝ := 2

/-- Theorem stating that the maximum number of cars passing the sensor in one hour, 
    divided by 15, is equal to 266 -/
theorem max_cars_quotient : N / 15 = 266 := by sorry

end NUMINAMATH_CALUDE_max_cars_quotient_l1176_117693


namespace NUMINAMATH_CALUDE_exactly_two_roots_l1176_117687

def equation (x k : ℂ) : Prop :=
  x / (x + 1) + x / (x + 3) = k * x

theorem exactly_two_roots :
  ∃! k : ℂ, (∃ x y : ℂ, x ≠ y ∧ 
    (∀ z : ℂ, equation z k ↔ z = x ∨ z = y)) ↔ 
  k = (4 : ℂ) / 3 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_roots_l1176_117687


namespace NUMINAMATH_CALUDE_sum_reciprocals_l1176_117618

theorem sum_reciprocals (x y z : ℝ) (ω : ℂ) 
  (hx : x ≠ -1) (hy : y ≠ -1) (hz : z ≠ -1)
  (hω1 : ω^3 = 1) (hω2 : ω ≠ 1)
  (h : 1/(x + ω) + 1/(y + ω) + 1/(z + ω) = ω) :
  1/(x + 1) + 1/(y + 1) + 1/(z + 1) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l1176_117618


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l1176_117655

theorem sqrt_difference_equality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (1 / Real.sqrt (2011 + Real.sqrt (2011^2 - 1)) : ℝ) = Real.sqrt m - Real.sqrt n →
  m + n = 2011 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l1176_117655


namespace NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l1176_117653

/-- Given that:
  1) Eighteen years ago, the father was 3 times as old as his son.
  2) Now, the father is twice as old as his son.
  Prove that the sum of their current ages is 108 years. -/
theorem father_son_age_sum : ℕ → ℕ → Prop :=
  fun (son_age father_age : ℕ) =>
    (father_age - 18 = 3 * (son_age - 18)) →
    (father_age = 2 * son_age) →
    (son_age + father_age = 108)

/-- Proof of the theorem -/
theorem father_son_age_sum_proof : ∃ (son_age father_age : ℕ),
  father_son_age_sum son_age father_age :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l1176_117653


namespace NUMINAMATH_CALUDE_knights_and_liars_l1176_117652

-- Define the inhabitants
inductive Inhabitant : Type
| A
| B
| C

-- Define the possible types of inhabitants
inductive InhabitantType : Type
| Knight
| Liar

-- Define a function to determine if an inhabitant is a knight or liar
def isKnight : Inhabitant → Bool
| Inhabitant.A => true  -- We assume A is a knight based on the solution
| Inhabitant.B => true  -- To be proved
| Inhabitant.C => false -- To be proved

-- Define what B and C claim about A's statement
def B_claim : Prop := isKnight Inhabitant.A = true
def C_claim : Prop := isKnight Inhabitant.A = false

-- The main theorem to prove
theorem knights_and_liars :
  (B_claim ∧ ¬C_claim) →
  (isKnight Inhabitant.B = true ∧ isKnight Inhabitant.C = false) := by
  sorry


end NUMINAMATH_CALUDE_knights_and_liars_l1176_117652


namespace NUMINAMATH_CALUDE_friends_ratio_l1176_117603

theorem friends_ratio (james_friends : ℕ) (shared_friends : ℕ) (combined_list : ℕ) :
  james_friends = 75 →
  shared_friends = 25 →
  combined_list = 275 →
  ∃ (john_friends : ℕ),
    john_friends = combined_list - james_friends →
    (john_friends : ℚ) / james_friends = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_friends_ratio_l1176_117603


namespace NUMINAMATH_CALUDE_angle_range_from_cosine_bounds_l1176_117650

theorem angle_range_from_cosine_bounds (A : Real) (h_acute : 0 < A ∧ A < Real.pi / 2) 
  (h_cos_bounds : 1 / 2 < Real.cos A ∧ Real.cos A < Real.sqrt 3 / 2) : 
  Real.pi / 6 < A ∧ A < Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_angle_range_from_cosine_bounds_l1176_117650


namespace NUMINAMATH_CALUDE_train_journey_distance_l1176_117620

/-- Represents the train's journey with an accident -/
structure TrainJourney where
  initialSpeed : ℝ
  totalDistance : ℝ
  accidentDelay : ℝ
  speedReductionFactor : ℝ
  totalDelay : ℝ
  alternateLaterAccidentDistance : ℝ
  alternateTotalDelay : ℝ

/-- The train journey satisfies the given conditions -/
def satisfiesConditions (j : TrainJourney) : Prop :=
  j.accidentDelay = 0.5 ∧
  j.speedReductionFactor = 3/4 ∧
  j.totalDelay = 3.5 ∧
  j.alternateLaterAccidentDistance = 90 ∧
  j.alternateTotalDelay = 3

/-- The theorem stating that the journey distance is 600 miles -/
theorem train_journey_distance (j : TrainJourney) 
  (h : satisfiesConditions j) : j.totalDistance = 600 :=
sorry

#check train_journey_distance

end NUMINAMATH_CALUDE_train_journey_distance_l1176_117620


namespace NUMINAMATH_CALUDE_sum_of_real_roots_of_quartic_l1176_117600

theorem sum_of_real_roots_of_quartic (x : ℝ) :
  let f : ℝ → ℝ := λ x => x^4 - 4*x - 1
  ∃ (r₁ r₂ : ℝ), (f r₁ = 0 ∧ f r₂ = 0) ∧ (∀ r : ℝ, f r = 0 → r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_real_roots_of_quartic_l1176_117600


namespace NUMINAMATH_CALUDE_tangent_line_at_neg_one_max_value_on_interval_min_value_on_interval_l1176_117658

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 1

-- Define the interval [0, 4]
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

-- Theorem for the tangent line equation at x = -1
theorem tangent_line_at_neg_one :
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ 4 * x - y + 4 = 0 :=
sorry

-- Theorem for the maximum value of f(x) on the interval [0, 4]
theorem max_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 45 :=
sorry

-- Theorem for the minimum value of f(x) on the interval [0, 4]
theorem min_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_neg_one_max_value_on_interval_min_value_on_interval_l1176_117658


namespace NUMINAMATH_CALUDE_height_of_tallest_person_l1176_117659

/-- Given four people with heights satisfying certain conditions, 
    prove that the tallest person is 84 inches tall. -/
theorem height_of_tallest_person 
  (h₁ h₂ h₃ h₄ : ℝ) 
  (height_order : h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄)
  (diff_1_2 : h₂ - h₁ = 2)
  (diff_2_3 : h₃ - h₂ = 2)
  (diff_3_4 : h₄ - h₃ = 6)
  (average_height : (h₁ + h₂ + h₃ + h₄) / 4 = 78) :
  h₄ = 84 := by
sorry

end NUMINAMATH_CALUDE_height_of_tallest_person_l1176_117659


namespace NUMINAMATH_CALUDE_inequality_solution_f_less_than_one_l1176_117681

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1|

-- Theorem 1
theorem inequality_solution (x : ℝ) : f x > x + 5 ↔ x > 4 ∨ x < -2 := by sorry

-- Theorem 2
theorem f_less_than_one (x y : ℝ) (h1 : |x - 3*y - 1| < 1/4) (h2 : |2*y + 1| < 1/6) : f x < 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_f_less_than_one_l1176_117681


namespace NUMINAMATH_CALUDE_total_food_for_three_months_l1176_117633

-- Define the number of days in each month
def december_days : ℕ := 31
def january_days : ℕ := 31
def february_days : ℕ := 28

-- Define the amount of food per feeding
def food_per_feeding : ℚ := 1/2

-- Define the number of feedings per day
def feedings_per_day : ℕ := 2

-- Theorem statement
theorem total_food_for_three_months :
  let total_days := december_days + january_days + february_days
  let daily_food := food_per_feeding * feedings_per_day
  total_days * daily_food = 90 := by sorry

end NUMINAMATH_CALUDE_total_food_for_three_months_l1176_117633


namespace NUMINAMATH_CALUDE_town_population_problem_l1176_117691

theorem town_population_problem (original : ℕ) : 
  (((original + 1500) * 85 / 100) : ℕ) = original - 45 → original = 8800 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l1176_117691


namespace NUMINAMATH_CALUDE_tangent_sum_half_pi_l1176_117696

theorem tangent_sum_half_pi (α β γ : Real) 
  (h_acute : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2) 
  (h_sum : α + β + γ = π/2) : 
  Real.tan α * Real.tan β + Real.tan α * Real.tan γ + Real.tan β * Real.tan γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_half_pi_l1176_117696


namespace NUMINAMATH_CALUDE_thirtieth_term_is_119_l1176_117667

/-- A function representing an arithmetic sequence with a given first term and common difference -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem stating that the 30th term of the specified arithmetic sequence is 119 -/
theorem thirtieth_term_is_119 :
  let a₁ := 3
  let a₂ := 7
  let d := a₂ - a₁
  arithmeticSequence a₁ d 30 = 119 := by
sorry


end NUMINAMATH_CALUDE_thirtieth_term_is_119_l1176_117667


namespace NUMINAMATH_CALUDE_exercise_book_distribution_l1176_117689

theorem exercise_book_distribution (students : ℕ) (total_books : ℕ) : 
  (3 * students + 7 = total_books) ∧ (5 * students = total_books + 9) →
  students = 8 ∧ total_books = 31 := by
sorry

end NUMINAMATH_CALUDE_exercise_book_distribution_l1176_117689


namespace NUMINAMATH_CALUDE_inequality_proof_l1176_117611

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * b + b * c + c * a ≤ 3 * a * b * c) :
  Real.sqrt ((a^2 + b^2) / (a + b)) + Real.sqrt ((b^2 + c^2) / (b + c)) +
  Real.sqrt ((c^2 + a^2) / (c + a)) + 3 ≤
  Real.sqrt 2 * (Real.sqrt (a + b) + Real.sqrt (b + c) + Real.sqrt (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1176_117611


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l1176_117661

/-- The number of people that can ride the Ferris wheel at the same time -/
def total_riders : ℕ := 4

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := 2

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := total_riders / people_per_seat

theorem ferris_wheel_seats : num_seats = 2 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l1176_117661


namespace NUMINAMATH_CALUDE_carla_marbles_l1176_117629

/-- The number of marbles Carla has after buying more -/
def total_marbles (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating the total number of marbles Carla has -/
theorem carla_marbles :
  total_marbles 2289 489 = 2778 := by
  sorry

end NUMINAMATH_CALUDE_carla_marbles_l1176_117629


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1176_117644

theorem rationalize_denominator : 7 / Real.sqrt 200 = (7 * Real.sqrt 2) / 20 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1176_117644


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l1176_117638

/-- Given a bag with white and red balls, prove that the probability of drawing a white ball equals 2/6 -/
theorem probability_of_white_ball (b : ℕ) : 
  let white_balls := b - 4
  let red_balls := b + 46
  let total_balls := white_balls + red_balls
  let prob_white := white_balls / total_balls
  prob_white = 2 / 6 := by
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l1176_117638


namespace NUMINAMATH_CALUDE_lindas_lunchbox_total_cost_l1176_117647

/-- The cost of a sandwich at Linda's Lunchbox -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda at Linda's Lunchbox -/
def soda_cost : ℕ := 2

/-- The cost of a cookie at Linda's Lunchbox -/
def cookie_cost : ℕ := 1

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 7

/-- The number of sodas purchased -/
def num_sodas : ℕ := 6

/-- The number of cookies purchased -/
def num_cookies : ℕ := 4

/-- The total cost of the purchase at Linda's Lunchbox -/
def total_cost : ℕ := num_sandwiches * sandwich_cost + num_sodas * soda_cost + num_cookies * cookie_cost

theorem lindas_lunchbox_total_cost : total_cost = 44 := by
  sorry

end NUMINAMATH_CALUDE_lindas_lunchbox_total_cost_l1176_117647
