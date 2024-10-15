import Mathlib

namespace NUMINAMATH_CALUDE_total_games_won_l1738_173837

/-- The number of games won by the Chicago Bulls -/
def bulls_wins : ℕ := 70

/-- The number of games won by the Miami Heat -/
def heat_wins : ℕ := bulls_wins + 5

/-- The number of games won by the New York Knicks -/
def knicks_wins : ℕ := 2 * heat_wins

/-- The number of games won by the Los Angeles Lakers -/
def lakers_wins : ℕ := (3 * (bulls_wins + knicks_wins)) / 2

/-- The total number of games won by all four teams -/
def total_wins : ℕ := bulls_wins + heat_wins + knicks_wins + lakers_wins

theorem total_games_won : total_wins = 625 := by
  sorry

end NUMINAMATH_CALUDE_total_games_won_l1738_173837


namespace NUMINAMATH_CALUDE_ab9_equals_459_implies_a_equals_4_l1738_173880

/-- Represents a three-digit number with 9 as the last digit -/
structure ThreeDigitNumber9 where
  hundreds : Nat
  tens : Nat
  inv_hundreds : hundreds < 10
  inv_tens : tens < 10

/-- Converts a ThreeDigitNumber9 to its numerical value -/
def ThreeDigitNumber9.toNat (n : ThreeDigitNumber9) : Nat :=
  100 * n.hundreds + 10 * n.tens + 9

theorem ab9_equals_459_implies_a_equals_4 (ab9 : ThreeDigitNumber9) 
  (h : ab9.toNat = 459) : ab9.hundreds = 4 := by
  sorry

end NUMINAMATH_CALUDE_ab9_equals_459_implies_a_equals_4_l1738_173880


namespace NUMINAMATH_CALUDE_function_range_l1738_173883

theorem function_range : 
  ∃ (min max : ℝ), min = -1 ∧ max = 3 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → min ≤ x^2 - 2*x ∧ x^2 - 2*x ≤ max) ∧
  (∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 3 ∧ 0 ≤ x₂ ∧ x₂ ≤ 3 ∧ 
    x₁^2 - 2*x₁ = min ∧ x₂^2 - 2*x₂ = max) := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1738_173883


namespace NUMINAMATH_CALUDE_complete_square_l1738_173820

theorem complete_square (x : ℝ) : x^2 - 8*x + 15 = 0 ↔ (x - 4)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_l1738_173820


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_l1738_173818

theorem mean_equality_implies_x_value : 
  ∃ x : ℝ, (6 + 9 + 18) / 3 = (x + 15) / 2 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_l1738_173818


namespace NUMINAMATH_CALUDE_fourth_sample_number_l1738_173868

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Checks if a number is in the systematic sample -/
def in_sample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

theorem fourth_sample_number 
  (s : SystematicSample)
  (h_pop : s.population_size = 56)
  (h_sample : s.sample_size = 4)
  (h_6 : in_sample s 6)
  (h_34 : in_sample s 34)
  (h_48 : in_sample s 48) :
  in_sample s 20 :=
sorry

end NUMINAMATH_CALUDE_fourth_sample_number_l1738_173868


namespace NUMINAMATH_CALUDE_min_segment_length_for_cyclists_l1738_173866

/-- Represents a cyclist on a circular track -/
structure Cyclist where
  speed : ℝ
  position : ℝ

/-- The circular track -/
def trackLength : ℝ := 300

/-- Theorem stating the minimum length of track segment where all cyclists will eventually appear -/
theorem min_segment_length_for_cyclists (c1 c2 c3 : Cyclist) 
  (h1 : c1.speed ≠ c2.speed)
  (h2 : c2.speed ≠ c3.speed)
  (h3 : c1.speed ≠ c3.speed)
  (h4 : c1.speed > 0 ∧ c2.speed > 0 ∧ c3.speed > 0) :
  ∃ (d : ℝ), d = 75 ∧ 
  (∀ (t : ℝ), ∃ (t' : ℝ), t' ≥ t ∧ 
    (((c1.position + c1.speed * t') % trackLength - 
      (c2.position + c2.speed * t') % trackLength + trackLength) % trackLength ≤ d ∧
     ((c2.position + c2.speed * t') % trackLength - 
      (c3.position + c3.speed * t') % trackLength + trackLength) % trackLength ≤ d ∧
     ((c1.position + c1.speed * t') % trackLength - 
      (c3.position + c3.speed * t') % trackLength + trackLength) % trackLength ≤ d)) :=
sorry

end NUMINAMATH_CALUDE_min_segment_length_for_cyclists_l1738_173866


namespace NUMINAMATH_CALUDE_f_2022_is_zero_l1738_173843

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem f_2022_is_zero (f : ℝ → ℝ) 
  (h1 : is_even_function (fun x ↦ f (2*x + 1)))
  (h2 : is_odd_function (fun x ↦ f (x + 2))) :
  f 2022 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2022_is_zero_l1738_173843


namespace NUMINAMATH_CALUDE_school_bus_capacity_l1738_173825

/-- Calculates the total number of students that can be seated on a bus --/
def bus_capacity (rows : ℕ) (sections_per_row : ℕ) (students_per_section : ℕ) : ℕ :=
  rows * sections_per_row * students_per_section

/-- Theorem: A bus with 13 rows, 2 sections per row, and 2 students per section can seat 52 students --/
theorem school_bus_capacity : bus_capacity 13 2 2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_school_bus_capacity_l1738_173825


namespace NUMINAMATH_CALUDE_circle_through_origin_equation_l1738_173848

/-- Defines a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem circle_through_origin_equation 
  (c : Circle) 
  (h1 : c.center = (3, 4)) 
  (h2 : onCircle c (0, 0)) : 
  ∀ (x y : ℝ), onCircle c (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_circle_through_origin_equation_l1738_173848


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1738_173852

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- The main theorem stating that any function satisfying the functional equation
    must be of the form f(x) = kx for some constant k -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1738_173852


namespace NUMINAMATH_CALUDE_gumdrop_cost_l1738_173849

/-- Given a total amount of 224 cents and the ability to buy 28 gumdrops,
    prove that the cost of each gumdrop is 8 cents. -/
theorem gumdrop_cost (total : ℕ) (quantity : ℕ) (h1 : total = 224) (h2 : quantity = 28) :
  total / quantity = 8 := by
  sorry

end NUMINAMATH_CALUDE_gumdrop_cost_l1738_173849


namespace NUMINAMATH_CALUDE_short_stack_pancakes_l1738_173828

/-- The number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- The number of customers who ordered short stack -/
def short_stack_orders : ℕ := 9

/-- The number of customers who ordered big stack -/
def big_stack_orders : ℕ := 6

/-- The total number of pancakes needed -/
def total_pancakes : ℕ := 57

/-- The number of pancakes in a short stack -/
def short_stack : ℕ := 3

theorem short_stack_pancakes :
  short_stack * short_stack_orders + big_stack * big_stack_orders = total_pancakes :=
by sorry

end NUMINAMATH_CALUDE_short_stack_pancakes_l1738_173828


namespace NUMINAMATH_CALUDE_sum_last_two_digits_lfs_l1738_173804

/-- Lucas Factorial Series function -/
def lucasFactorialSeries : ℕ → ℕ
| 0 => 2
| 1 => 1
| 2 => 3
| 3 => 4
| 4 => 7
| 5 => 11
| _ => 0

/-- Calculate factorial -/
def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

/-- Get last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ :=
  n % 100

/-- Sum of last two digits of factorials in Lucas Factorial Series -/
def sumLastTwoDigitsLFS : ℕ :=
  let series := List.range 6
  series.foldl (fun acc i => acc + lastTwoDigits (factorial (lucasFactorialSeries i))) 0

/-- Main theorem -/
theorem sum_last_two_digits_lfs :
  sumLastTwoDigitsLFS = 73 := by
  sorry


end NUMINAMATH_CALUDE_sum_last_two_digits_lfs_l1738_173804


namespace NUMINAMATH_CALUDE_compound_weight_l1738_173824

/-- Given a compound with a molecular weight of 1188, prove that the total weight of 4 moles is 4752,
    while the molecular weight remains constant. -/
theorem compound_weight (molecular_weight : ℕ) (num_moles : ℕ) :
  molecular_weight = 1188 → num_moles = 4 →
  (num_moles * molecular_weight = 4752) ∧ (molecular_weight = 1188) := by
  sorry

#check compound_weight

end NUMINAMATH_CALUDE_compound_weight_l1738_173824


namespace NUMINAMATH_CALUDE_checkerboard_swap_iff_div_three_l1738_173806

/-- Represents the color of a cell -/
inductive Color
  | White
  | Black
  | Green

/-- Represents a grid of size n × n -/
def Grid (n : ℕ) := Fin n → Fin n → Color

/-- Initial checkerboard coloring with at least one corner black -/
def initialGrid (n : ℕ) : Grid n := 
  λ i j => if (i.val + j.val) % 2 = 0 then Color.Black else Color.White

/-- Recoloring rule for a 2×2 subgrid -/
def recolorSubgrid (g : Grid n) (i j : Fin n) : Grid n :=
  λ x y => if (x.val ≥ i.val && x.val < i.val + 2 && y.val ≥ j.val && y.val < j.val + 2)
    then match g x y with
      | Color.White => Color.Black
      | Color.Black => Color.Green
      | Color.Green => Color.White
    else g x y

/-- Check if the grid is in a swapped checkerboard pattern -/
def isSwappedCheckerboard (g : Grid n) : Prop :=
  ∀ i j, g i j = if (i.val + j.val) % 2 = 0 then Color.White else Color.Black

/-- Main theorem: The checkerboard color swap is possible iff n is divisible by 3 -/
theorem checkerboard_swap_iff_div_three (n : ℕ) :
  (∃ (moves : List (Fin n × Fin n)), 
    isSwappedCheckerboard (moves.foldl (λ g (i, j) => recolorSubgrid g i j) (initialGrid n))) 
  ↔ 
  3 ∣ n := by sorry

end NUMINAMATH_CALUDE_checkerboard_swap_iff_div_three_l1738_173806


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1738_173846

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1738_173846


namespace NUMINAMATH_CALUDE_prize_distribution_methods_l1738_173835

-- Define the number of prizes
def num_prizes : ℕ := 6

-- Define the number of people
def num_people : ℕ := 5

-- Define a function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define a function to calculate permutations
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem prize_distribution_methods :
  (combination num_prizes 2) * (permutation num_people num_people) =
  (number_of_distribution_methods : ℕ) :=
sorry

end NUMINAMATH_CALUDE_prize_distribution_methods_l1738_173835


namespace NUMINAMATH_CALUDE_pink_shells_count_l1738_173841

theorem pink_shells_count (total : ℕ) (purple yellow blue orange : ℕ) 
  (h1 : total = 65)
  (h2 : purple = 13)
  (h3 : yellow = 18)
  (h4 : blue = 12)
  (h5 : orange = 14) :
  total - (purple + yellow + blue + orange) = 8 := by
  sorry

end NUMINAMATH_CALUDE_pink_shells_count_l1738_173841


namespace NUMINAMATH_CALUDE_cousins_initial_money_l1738_173897

/-- Represents the money distribution problem with Carmela and her cousins -/
def money_distribution (carmela_initial : ℕ) (cousin_count : ℕ) (give_amount : ℕ) (cousin_initial : ℕ) : Prop :=
  let carmela_final := carmela_initial - (cousin_count * give_amount)
  let cousin_final := cousin_initial + give_amount
  carmela_final = cousin_final

/-- Proves that given the conditions, each cousin must have had $2 initially -/
theorem cousins_initial_money :
  money_distribution 7 4 1 2 := by
  sorry

end NUMINAMATH_CALUDE_cousins_initial_money_l1738_173897


namespace NUMINAMATH_CALUDE_existence_of_b₁_b₂_l1738_173811

theorem existence_of_b₁_b₂ (a₁ a₂ : ℝ) 
  (h₁ : a₁ ≥ 0) (h₂ : a₂ ≥ 0) (h₃ : a₁ + a₂ = 1) : 
  ∃ b₁ b₂ : ℝ, b₁ ≥ 0 ∧ b₂ ≥ 0 ∧ b₁ + b₂ = 1 ∧ 
  (5/4 - a₁) * b₁ + 3 * (5/4 - a₂) * b₂ > 1 := by
sorry

end NUMINAMATH_CALUDE_existence_of_b₁_b₂_l1738_173811


namespace NUMINAMATH_CALUDE_baker_leftover_cupcakes_l1738_173831

/-- Represents the cupcake distribution problem --/
def cupcake_distribution (total_cupcakes nutty_cupcakes gluten_free_cupcakes num_children : ℕ)
  (num_nut_allergic num_gluten_only : ℕ) : ℕ :=
  let regular_cupcakes := total_cupcakes - nutty_cupcakes - gluten_free_cupcakes
  let nutty_per_child := nutty_cupcakes / (num_children - num_nut_allergic)
  let nutty_distributed := nutty_per_child * (num_children - num_nut_allergic)
  let regular_per_child := regular_cupcakes / num_children
  let regular_distributed := regular_per_child * num_children
  let leftover_nutty := nutty_cupcakes - nutty_distributed
  let leftover_regular := regular_cupcakes - regular_distributed
  leftover_nutty + leftover_regular

/-- Theorem stating that given the specific conditions, Ms. Baker will have 5 cupcakes left over --/
theorem baker_leftover_cupcakes :
  cupcake_distribution 84 18 25 7 2 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_baker_leftover_cupcakes_l1738_173831


namespace NUMINAMATH_CALUDE_candy_cost_in_dollars_l1738_173816

/-- The cost of a single piece of candy in cents -/
def candy_cost : ℕ := 2

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of candy pieces we're calculating the cost for -/
def candy_pieces : ℕ := 500

theorem candy_cost_in_dollars : 
  (candy_pieces * candy_cost) / cents_per_dollar = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_in_dollars_l1738_173816


namespace NUMINAMATH_CALUDE_percentage_difference_l1738_173854

theorem percentage_difference (x y : ℝ) (h : x = 4 * y) :
  (x - y) / x * 100 = 75 :=
by sorry

end NUMINAMATH_CALUDE_percentage_difference_l1738_173854


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2310_l1738_173805

theorem smallest_prime_factor_of_2310 : Nat.minFac 2310 = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2310_l1738_173805


namespace NUMINAMATH_CALUDE_probability_both_in_photo_l1738_173876

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℝ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the photography setup -/
structure PhotoSetup where
  trackCoverage : ℝ  -- fraction of track covered by the photo
  minTime : ℝ        -- minimum time after start for taking the photo (in seconds)
  maxTime : ℝ        -- maximum time after start for taking the photo (in seconds)

/-- Calculate the probability of both runners being in the photo -/
def probabilityBothInPhoto (ann : Runner) (ben : Runner) (setup : PhotoSetup) : ℝ :=
  sorry

/-- Theorem statement for the probability problem -/
theorem probability_both_in_photo 
  (ann : Runner) 
  (ben : Runner) 
  (setup : PhotoSetup) 
  (h1 : ann.name = "Ann" ∧ ann.lapTime = 75 ∧ ann.direction = true)
  (h2 : ben.name = "Ben" ∧ ben.lapTime = 60 ∧ ben.direction = false)
  (h3 : setup.trackCoverage = 1/6)
  (h4 : setup.minTime = 12 * 60)
  (h5 : setup.maxTime = 15 * 60) :
  probabilityBothInPhoto ann ben setup = 1/6 :=
sorry

end NUMINAMATH_CALUDE_probability_both_in_photo_l1738_173876


namespace NUMINAMATH_CALUDE_flash_catch_up_distance_l1738_173830

theorem flash_catch_up_distance 
  (v : ℝ) -- Ace's speed
  (z : ℝ) -- Flash's speed multiplier
  (k : ℝ) -- Ace's head start distance
  (t₀ : ℝ) -- Time Ace runs before Flash starts
  (h₁ : v > 0) -- Ace's speed is positive
  (h₂ : z > 1) -- Flash is faster than Ace
  (h₃ : k ≥ 0) -- Head start is non-negative
  (h₄ : t₀ ≥ 0) -- Time before Flash starts is non-negative
  : 
  ∃ (t : ℝ), t > 0 ∧ z * v * t = v * (t + t₀) + k ∧
  z * v * t = z * (t₀ * v + k) / (z - 1) :=
sorry

end NUMINAMATH_CALUDE_flash_catch_up_distance_l1738_173830


namespace NUMINAMATH_CALUDE_min_value_theorem_l1738_173823

theorem min_value_theorem (x y : ℝ) (h1 : x + y = 1) (h2 : x > 0) (h3 : y > 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 
    (1 / (2 * x) + x / (y + 1)) ≤ (1 / (2 * a) + a / (b + 1))) ∧
  (1 / (2 * x) + x / (y + 1) = 5/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1738_173823


namespace NUMINAMATH_CALUDE_linoleum_cut_theorem_l1738_173802

/-- Represents a square on the linoleum piece -/
inductive Square
| White
| Black

/-- Represents the modified 8x8 grid with two additional white squares -/
def ModifiedGrid := Array (Array Square)

/-- Represents a cut on the grid -/
structure Cut where
  start_row : Nat
  start_col : Nat
  end_row : Nat
  end_col : Nat

/-- Represents a transformation (rotation and translation) -/
structure Transform where
  rotation : Nat  -- 0, 1, 2, or 3 for 0, 90, 180, 270 degrees
  translation_row : Int
  translation_col : Int

/-- Checks if a grid is a proper 8x8 chessboard -/
def is_proper_chessboard (grid : Array (Array Square)) : Bool :=
  sorry

/-- Applies a cut to the grid, returning two pieces -/
def apply_cut (grid : ModifiedGrid) (cut : Cut) : (ModifiedGrid × ModifiedGrid) :=
  sorry

/-- Applies a transformation to a grid piece -/
def apply_transform (piece : ModifiedGrid) (transform : Transform) : ModifiedGrid :=
  sorry

/-- Combines two grid pieces -/
def combine_pieces (piece1 piece2 : ModifiedGrid) : ModifiedGrid :=
  sorry

theorem linoleum_cut_theorem (original_grid : ModifiedGrid) :
  ∃ (cut : Cut) (transform : Transform),
    let (piece1, piece2) := apply_cut original_grid cut
    let transformed_piece := apply_transform piece1 transform
    let result := combine_pieces transformed_piece piece2
    is_proper_chessboard result :=
  sorry

end NUMINAMATH_CALUDE_linoleum_cut_theorem_l1738_173802


namespace NUMINAMATH_CALUDE_total_letters_in_names_l1738_173899

theorem total_letters_in_names (jonathan_first : ℕ) (jonathan_surname : ℕ) 
  (sister_first : ℕ) (sister_surname : ℕ) 
  (h1 : jonathan_first = 8) (h2 : jonathan_surname = 10) 
  (h3 : sister_first = 5) (h4 : sister_surname = 10) : 
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_letters_in_names_l1738_173899


namespace NUMINAMATH_CALUDE_trajectory_is_ray_l1738_173872

/-- The set of complex numbers z satisfying |z+1| - |z-1| = 2 forms a ray in the complex plane -/
theorem trajectory_is_ray : 
  {z : ℂ | Complex.abs (z + 1) - Complex.abs (z - 1) = 2} = 
  {z : ℂ | ∃ t : ℝ, t ≥ 0 ∧ z = 1 + t} := by sorry

end NUMINAMATH_CALUDE_trajectory_is_ray_l1738_173872


namespace NUMINAMATH_CALUDE_expression_equality_l1738_173836

theorem expression_equality : 
  -21 * (2/3) + 3 * (1/4) - (-2/3) - (1/4) = -18 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1738_173836


namespace NUMINAMATH_CALUDE_correct_miscopied_value_l1738_173867

/-- Given a set of values with an incorrect mean due to one miscopied value,
    calculate the correct value that should have been recorded. -/
theorem correct_miscopied_value
  (n : ℕ) -- Total number of values
  (initial_mean : ℚ) -- Initial (incorrect) mean
  (wrong_value : ℚ) -- Value that was incorrectly recorded
  (correct_mean : ℚ) -- Correct mean after fixing the error
  (h1 : n = 30) -- There are 30 values
  (h2 : initial_mean = 150) -- The initial mean was 150
  (h3 : wrong_value = 135) -- The value was incorrectly recorded as 135
  (h4 : correct_mean = 151) -- The correct mean is 151
  : ℚ := -- The theorem returns a rational number
by
  -- The proof goes here
  sorry

#check correct_miscopied_value

end NUMINAMATH_CALUDE_correct_miscopied_value_l1738_173867


namespace NUMINAMATH_CALUDE_line_equation_condition1_line_equation_condition2_line_equation_condition3_l1738_173856

-- Define the line l
def line_l (a b c : ℝ) : Prop := ∀ x y : ℝ, a * x + b * y + c = 0

-- Define the point (1, -2) that the line passes through
def point_condition (a b c : ℝ) : Prop := a * 1 + b * (-2) + c = 0

-- Theorem for condition 1
theorem line_equation_condition1 (a b c : ℝ) :
  point_condition a b c →
  (∃ k : ℝ, k = 1 - π / 12 ∧ b / a = -k) →
  line_l a b c ↔ line_l 1 (-Real.sqrt 3) (-2 * Real.sqrt 3 - 1) :=
sorry

-- Theorem for condition 2
theorem line_equation_condition2 (a b c : ℝ) :
  point_condition a b c →
  (b / a = 1) →
  line_l a b c ↔ line_l 1 (-1) (-3) :=
sorry

-- Theorem for condition 3
theorem line_equation_condition3 (a b c : ℝ) :
  point_condition a b c →
  (c / b = -1) →
  line_l a b c ↔ line_l 1 1 1 :=
sorry

end NUMINAMATH_CALUDE_line_equation_condition1_line_equation_condition2_line_equation_condition3_l1738_173856


namespace NUMINAMATH_CALUDE_vann_teeth_cleaning_l1738_173812

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of teeth a pig has -/
def pig_teeth : ℕ := 28

/-- The number of dogs Vann will clean -/
def num_dogs : ℕ := 5

/-- The number of cats Vann will clean -/
def num_cats : ℕ := 10

/-- The number of pigs Vann will clean -/
def num_pigs : ℕ := 7

/-- The total number of teeth Vann will clean -/
def total_teeth : ℕ := dog_teeth * num_dogs + cat_teeth * num_cats + pig_teeth * num_pigs

theorem vann_teeth_cleaning :
  total_teeth = 706 := by
  sorry

end NUMINAMATH_CALUDE_vann_teeth_cleaning_l1738_173812


namespace NUMINAMATH_CALUDE_remainder_1999_11_mod_8_l1738_173838

theorem remainder_1999_11_mod_8 : 1999^11 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1999_11_mod_8_l1738_173838


namespace NUMINAMATH_CALUDE_product_xy_equals_nine_sqrt_three_l1738_173813

-- Define the variables
variable (x y a b : ℝ)

-- State the theorem
theorem product_xy_equals_nine_sqrt_three
  (h1 : x = b^(3/2))
  (h2 : y = a)
  (h3 : a + a = b^2)
  (h4 : y = b)
  (h5 : a + a = b^(3/2))
  (h6 : b = 3) :
  x * y = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_equals_nine_sqrt_three_l1738_173813


namespace NUMINAMATH_CALUDE_abc_mod_seven_l1738_173827

theorem abc_mod_seven (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
  (h1 : (a + 3*b + 2*c) % 7 = 2)
  (h2 : (2*a + b + 3*c) % 7 = 3)
  (h3 : (3*a + 2*b + c) % 7 = 5) :
  (a * b * c) % 7 = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_mod_seven_l1738_173827


namespace NUMINAMATH_CALUDE_grade_change_impossible_l1738_173845

theorem grade_change_impossible : ∀ (n1 n2 n3 n4 : ℤ),
  2 * n1 + n2 - 2 * n3 - n4 = 27 ∧
  -n1 + 2 * n2 + n3 - 2 * n4 = -27 →
  False :=
by
  sorry

end NUMINAMATH_CALUDE_grade_change_impossible_l1738_173845


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l1738_173885

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l1738_173885


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1738_173864

theorem smallest_k_no_real_roots : 
  ∀ k : ℤ, (∀ x : ℝ, 2 * x * (k * x - 5) - x^2 + 12 ≠ 0) → k ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1738_173864


namespace NUMINAMATH_CALUDE_bridge_units_correct_l1738_173888

-- Define the units
inductive LengthUnit
| Kilometers

inductive LoadUnit
| Tons

-- Define the bridge properties
structure Bridge where
  length : ℕ
  loadCapacity : ℕ

-- Define the function to assign units
def assignUnits (b : Bridge) : (LengthUnit × LoadUnit) :=
  (LengthUnit.Kilometers, LoadUnit.Tons)

-- Theorem statement
theorem bridge_units_correct (b : Bridge) (h1 : b.length = 1) (h2 : b.loadCapacity = 50) :
  assignUnits b = (LengthUnit.Kilometers, LoadUnit.Tons) := by
  sorry

#check bridge_units_correct

end NUMINAMATH_CALUDE_bridge_units_correct_l1738_173888


namespace NUMINAMATH_CALUDE_average_annual_reduction_l1738_173879

theorem average_annual_reduction (total_reduction : ℝ) (years : ℕ) (average_reduction : ℝ) : 
  total_reduction = 0.19 → years = 2 → (1 - average_reduction) ^ years = 1 - total_reduction → average_reduction = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_average_annual_reduction_l1738_173879


namespace NUMINAMATH_CALUDE_problem_solution_l1738_173842

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  3 * (x^4 + 2*y^2) / 9 = 113/3 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1738_173842


namespace NUMINAMATH_CALUDE_roots_of_quadratic_and_quartic_l1738_173886

theorem roots_of_quadratic_and_quartic (α β p q : ℝ) : 
  (α^2 - 3*α + 1 = 0) ∧ 
  (β^2 - 3*β + 1 = 0) ∧ 
  (α^4 - p*α^2 + q = 0) ∧ 
  (β^4 - p*β^2 + q = 0) →
  p = 7 ∧ q = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_and_quartic_l1738_173886


namespace NUMINAMATH_CALUDE_divisor_power_result_l1738_173859

theorem divisor_power_result (k : ℕ) : 
  21^k ∣ 435961 → 7^k - k^7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_power_result_l1738_173859


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l1738_173819

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 365 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 365 := by sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l1738_173819


namespace NUMINAMATH_CALUDE_batsman_score_difference_l1738_173858

theorem batsman_score_difference (total_innings : ℕ) (total_average : ℚ) (reduced_average : ℚ) (highest_score : ℕ) :
  total_innings = 46 →
  total_average = 61 →
  reduced_average = 58 →
  highest_score = 202 →
  ∃ (lowest_score : ℕ),
    (total_average * total_innings : ℚ) = 
      (reduced_average * (total_innings - 2) + (highest_score + lowest_score) : ℚ) ∧
    highest_score - lowest_score = 150 :=
by sorry

end NUMINAMATH_CALUDE_batsman_score_difference_l1738_173858


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1738_173877

theorem cubic_root_sum (p q r : ℝ) : 
  (p^3 - 3*p + 1 = 0) → 
  (q^3 - 3*q + 1 = 0) → 
  (r^3 - 3*r + 1 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1738_173877


namespace NUMINAMATH_CALUDE_first_half_speed_l1738_173839

/-- Proves that given a 60-mile trip where the speed increases by 16 mph halfway through,
    and the average speed for the entire trip is 30 mph,
    the average speed during the first half of the trip is 24 mph. -/
theorem first_half_speed (v : ℝ) : 
  (60 : ℝ) / ((30 / v) + (30 / (v + 16))) = 30 → v = 24 := by
  sorry

end NUMINAMATH_CALUDE_first_half_speed_l1738_173839


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1738_173853

theorem inequality_solution_set (x : ℝ) : 2 * x - 3 > 7 - x ↔ x > 10 / 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1738_173853


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1738_173807

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1738_173807


namespace NUMINAMATH_CALUDE_electricity_scientific_notation_equality_l1738_173801

/-- The amount of electricity generated by a wind power station per day -/
def electricity_per_day : ℝ := 74850000

/-- The scientific notation representation of the electricity generated per day -/
def scientific_notation : ℝ := 7.485 * (10^7)

/-- Theorem stating that the electricity_per_day is equal to its scientific notation representation -/
theorem electricity_scientific_notation_equality :
  electricity_per_day = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_electricity_scientific_notation_equality_l1738_173801


namespace NUMINAMATH_CALUDE_handshake_arrangements_mod_1000_l1738_173832

/-- The number of ways 10 people can shake hands, where each person shakes hands with exactly two others -/
def handshake_arrangements : ℕ := sorry

/-- Theorem stating that the number of handshake arrangements is congruent to 688 modulo 1000 -/
theorem handshake_arrangements_mod_1000 : 
  handshake_arrangements ≡ 688 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_handshake_arrangements_mod_1000_l1738_173832


namespace NUMINAMATH_CALUDE_total_questions_in_contest_l1738_173840

/-- Represents a participant in the spelling contest -/
structure Participant where
  name : String
  round1_correct : Nat
  round1_wrong : Nat
  round2_correct : Nat
  round2_wrong : Nat
  round3_correct : Nat
  round3_wrong : Nat

/-- Calculates the total number of questions for a participant in all rounds -/
def totalQuestions (p : Participant) : Nat :=
  p.round1_correct + p.round1_wrong +
  p.round2_correct + p.round2_wrong +
  p.round3_correct + p.round3_wrong

/-- Represents the spelling contest -/
structure SpellingContest where
  drew : Participant
  carla : Participant
  blake : Participant

/-- Theorem stating the total number of questions in the spelling contest -/
theorem total_questions_in_contest (contest : SpellingContest)
  (h1 : contest.drew.round1_correct = 20)
  (h2 : contest.drew.round1_wrong = 6)
  (h3 : contest.carla.round1_correct = 14)
  (h4 : contest.carla.round1_wrong = 2 * contest.drew.round1_wrong)
  (h5 : contest.drew.round2_correct = 24)
  (h6 : contest.drew.round2_wrong = 9)
  (h7 : contest.carla.round2_correct = 21)
  (h8 : contest.carla.round2_wrong = 8)
  (h9 : contest.blake.round2_correct = 18)
  (h10 : contest.blake.round2_wrong = 11)
  (h11 : contest.drew.round3_correct = 28)
  (h12 : contest.drew.round3_wrong = 14)
  (h13 : contest.carla.round3_correct = 22)
  (h14 : contest.carla.round3_wrong = 10)
  (h15 : contest.blake.round3_correct = 15)
  (h16 : contest.blake.round3_wrong = 16)
  : totalQuestions contest.drew + totalQuestions contest.carla + totalQuestions contest.blake = 248 := by
  sorry


end NUMINAMATH_CALUDE_total_questions_in_contest_l1738_173840


namespace NUMINAMATH_CALUDE_steve_book_earnings_l1738_173821

/-- Calculates an author's net earnings from book sales -/
def authorNetEarnings (copies : ℕ) (earningsPerCopy : ℚ) (agentPercentage : ℚ) : ℚ :=
  let totalEarnings := copies * earningsPerCopy
  let agentCommission := totalEarnings * agentPercentage
  totalEarnings - agentCommission

/-- Proves that given the specified conditions, the author's net earnings are $1,800,000 -/
theorem steve_book_earnings :
  authorNetEarnings 1000000 2 (1/10) = 1800000 := by
  sorry

#eval authorNetEarnings 1000000 2 (1/10)

end NUMINAMATH_CALUDE_steve_book_earnings_l1738_173821


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1738_173800

theorem cubic_equation_solution (x : ℝ) (h : x^3 + 1/x^3 = 110) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1738_173800


namespace NUMINAMATH_CALUDE_alyssa_pears_l1738_173809

theorem alyssa_pears (total_pears nancy_pears : ℕ) 
  (h1 : total_pears = 59) 
  (h2 : nancy_pears = 17) : 
  total_pears - nancy_pears = 42 := by
sorry

end NUMINAMATH_CALUDE_alyssa_pears_l1738_173809


namespace NUMINAMATH_CALUDE_right_triangle_with_acute_angle_greater_than_epsilon_l1738_173873

theorem right_triangle_with_acute_angle_greater_than_epsilon :
  ∀ ε : Real, 0 < ε → ε < π / 4 →
  ∃ a b c : ℕ, 
    a * a + b * b = c * c ∧ 
    Real.arctan (min (a / b) (b / a)) > ε :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_acute_angle_greater_than_epsilon_l1738_173873


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l1738_173815

theorem modular_arithmetic_problem :
  ∃ (a b c : ℤ),
    (7 * a) % 60 = 1 ∧
    (13 * b) % 60 = 1 ∧
    (17 * c) % 60 = 1 ∧
    (4 * a + 12 * b - 6 * c) % 60 = 58 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l1738_173815


namespace NUMINAMATH_CALUDE_megan_seashells_l1738_173871

def current_seashells : ℕ := 19
def needed_seashells : ℕ := 6
def target_seashells : ℕ := 25

theorem megan_seashells : current_seashells + needed_seashells = target_seashells := by
  sorry

end NUMINAMATH_CALUDE_megan_seashells_l1738_173871


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l1738_173862

theorem sugar_solution_percentage (x : ℝ) : 
  x > 0 ∧ x < 100 →
  (3/4 * x + 1/4 * 34) / 100 = 16 / 100 →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l1738_173862


namespace NUMINAMATH_CALUDE_equation_solution_l1738_173881

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 1 + Real.sqrt 3 ∧ x₂ = 1 - Real.sqrt 3) ∧
  (x₁^2 = (4*x₁ - 2)/(x₁ - 2) ∧ x₂^2 = (4*x₂ - 2)/(x₂ - 2)) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l1738_173881


namespace NUMINAMATH_CALUDE_duck_race_charity_amount_l1738_173826

/-- The amount of money raised for charity in the annual rubber duck race -/
def charity_money_raised (regular_price : ℝ) (large_price : ℝ) (regular_sold : ℕ) (large_sold : ℕ) : ℝ :=
  regular_price * (regular_sold : ℝ) + large_price * (large_sold : ℝ)

/-- Theorem stating the amount of money raised for charity in the given scenario -/
theorem duck_race_charity_amount :
  charity_money_raised 3 5 221 185 = 1588 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_race_charity_amount_l1738_173826


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1738_173894

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmeticSequence a →
  a 4 + a 6 + a 8 + a 10 + a 12 = 120 →
  a 7 - (1/3) * a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1738_173894


namespace NUMINAMATH_CALUDE_equation_describes_ellipse_l1738_173891

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

-- Define the property of being an ellipse
def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (F₁ F₂ : ℝ × ℝ) (a : ℝ),
    a > Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) / 2 ∧
    ∀ (x y : ℝ), f x y ↔ 
      Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) +
      Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = 2 * a

-- Theorem statement
theorem equation_describes_ellipse : is_ellipse equation := by
  sorry

end NUMINAMATH_CALUDE_equation_describes_ellipse_l1738_173891


namespace NUMINAMATH_CALUDE_overlap_area_of_circles_l1738_173855

/-- The area of overlap between two circles with given properties -/
theorem overlap_area_of_circles (r : ℝ) (overlap_percentage : ℝ) : 
  r = 10 ∧ overlap_percentage = 0.25 → 
  2 * (25 * Real.pi - 50) = 
    2 * ((overlap_percentage * 2 * Real.pi * r^2 / 4) - (r^2 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_of_circles_l1738_173855


namespace NUMINAMATH_CALUDE_sqrt_eighteen_minus_sqrt_two_l1738_173865

theorem sqrt_eighteen_minus_sqrt_two : Real.sqrt 18 - Real.sqrt 2 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_minus_sqrt_two_l1738_173865


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1738_173896

/-- Given two circles X and Y where an arc of 60° on X has the same length as an arc of 40° on Y,
    the ratio of the area of circle X to the area of circle Y is 9/4. -/
theorem circle_area_ratio (R_X R_Y : ℝ) (h : R_X > 0 ∧ R_Y > 0) :
  (60 / 360 * (2 * Real.pi * R_X) = 40 / 360 * (2 * Real.pi * R_Y)) →
  (Real.pi * R_X^2) / (Real.pi * R_Y^2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1738_173896


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_is_16_l1738_173860

/-- The speed of a boat in still water, given downstream travel information and stream speed. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 60)
  (h3 : downstream_time = 3)
  : ℝ :=
  let downstream_speed := downstream_distance / downstream_time
  let boat_speed := downstream_speed - stream_speed
  16

/-- Proof that the boat's speed in still water is 16 km/hr -/
theorem boat_speed_is_16
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 60)
  (h3 : downstream_time = 3)
  : boat_speed_in_still_water stream_speed downstream_distance downstream_time h1 h2 h3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_is_16_l1738_173860


namespace NUMINAMATH_CALUDE_inequalities_proof_l1738_173847

theorem inequalities_proof (a b c : ℝ) (h1 : a > 0) (h2 : a > b) (h3 : b > c) : 
  (a * b > b * c) ∧ (a * c > b * c) ∧ (a * b > a * c) ∧ (a + b > b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1738_173847


namespace NUMINAMATH_CALUDE_f_negative_three_l1738_173895

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- For any positive number x, f(2+x) = -2f(2-x)
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f (2 + x) = -2 * f (2 - x)

-- Main theorem
theorem f_negative_three (h1 : is_even f) (h2 : satisfies_condition f) (h3 : f (-1) = 4) :
  f (-3) = -8 := by
  sorry


end NUMINAMATH_CALUDE_f_negative_three_l1738_173895


namespace NUMINAMATH_CALUDE_unsold_bag_weights_l1738_173898

def bag_weights : List Nat := [3, 7, 12, 15, 17, 28, 30]

def total_weight : Nat := bag_weights.sum

structure SalesDistribution where
  day1 : Nat
  day2 : Nat
  day3 : Nat
  unsold : Nat

def is_valid_distribution (d : SalesDistribution) : Prop :=
  d.day1 + d.day2 + d.day3 + d.unsold = total_weight ∧
  d.day2 = 2 * d.day1 ∧
  d.day3 = 2 * d.day2 ∧
  d.unsold ∈ bag_weights

theorem unsold_bag_weights :
  ∀ d : SalesDistribution, is_valid_distribution d → d.unsold = 7 ∨ d.unsold = 28 :=
by sorry

end NUMINAMATH_CALUDE_unsold_bag_weights_l1738_173898


namespace NUMINAMATH_CALUDE_circular_fields_area_difference_l1738_173850

theorem circular_fields_area_difference (r₁ r₂ : ℝ) (h : r₁ / r₂ = 3 / 10) :
  1 - (π * r₁^2) / (π * r₂^2) = 91 / 100 := by
  sorry

end NUMINAMATH_CALUDE_circular_fields_area_difference_l1738_173850


namespace NUMINAMATH_CALUDE_intersection_points_with_constraints_l1738_173803

/-- The number of intersection points of n lines -/
def intersectionPoints (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of lines -/
def numLines : ℕ := 10

/-- The number of parallel line pairs -/
def numParallelPairs : ℕ := 1

/-- The number of lines intersecting at a single point -/
def numConcurrentLines : ℕ := 3

theorem intersection_points_with_constraints :
  intersectionPoints numLines - numParallelPairs - (numConcurrentLines.choose 2 - 1) = 42 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_with_constraints_l1738_173803


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1738_173884

/-- An arithmetic sequence with common difference d and first term a_1 -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : d ≠ 0)
  (h3 : a 1 ≠ 0)
  (h4 : geometric_sequence (a 2) (a 4) (a 8)) :
  (a 1 + a 5 + a 9) / (a 2 + a 3) = 3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1738_173884


namespace NUMINAMATH_CALUDE_polynomial_coefficient_problem_l1738_173887

theorem polynomial_coefficient_problem (a b : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + b) * (2*x^2 - 3*x - 1) = 
    2*x^4 + (-5)*x^3 + (-6)*x^2 + ((-3*b - a)*x - b)) → 
  a = -1 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_problem_l1738_173887


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l1738_173834

theorem cube_volume_ratio (edge_ratio : ℝ) (small_volume : ℝ) :
  edge_ratio = 4.999999999999999 →
  small_volume = 1 →
  (edge_ratio ^ 3) * small_volume = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l1738_173834


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l1738_173851

def U : Finset Int := {-1, 0, 1, 2, 3, 4}
def A : Finset Int := {2, 3}
def B : Finset Int := {1, 2, 3, 4} \ A

theorem intersection_complement_equals : B ∩ (U \ A) = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l1738_173851


namespace NUMINAMATH_CALUDE_skee_ball_tickets_proof_l1738_173892

/-- The number of tickets Tom won from 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- The number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- The number of tickets Tom has left -/
def remaining_tickets : ℕ := 50

/-- The number of tickets Tom won from 'skee ball' -/
def skee_ball_tickets : ℕ := (remaining_tickets + spent_tickets) - whack_a_mole_tickets

theorem skee_ball_tickets_proof : skee_ball_tickets = 25 := by
  sorry

end NUMINAMATH_CALUDE_skee_ball_tickets_proof_l1738_173892


namespace NUMINAMATH_CALUDE_log_9_729_l1738_173814

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_9_729 : log 9 729 = 3 := by sorry

end NUMINAMATH_CALUDE_log_9_729_l1738_173814


namespace NUMINAMATH_CALUDE_even_function_domain_l1738_173857

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = ax^2 + bx + 3a + b -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + 3 * a + b

theorem even_function_domain (a b : ℝ) :
  (IsEven (f a b)) ∧ (Set.Icc (2 * a) (a - 1)).Nonempty → a + b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_domain_l1738_173857


namespace NUMINAMATH_CALUDE_square_sum_value_l1738_173875

theorem square_sum_value (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 2) : a^2 + b^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l1738_173875


namespace NUMINAMATH_CALUDE_not_coplanar_implies_not_intersect_exists_not_intersect_but_coplanar_l1738_173878

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Check if two lines intersect -/
def linesIntersect (l1 l2 : Line3D) : Prop := sorry

theorem not_coplanar_implies_not_intersect 
  (E F G H : Point3D) 
  (EF : Line3D) 
  (GH : Line3D) 
  (h1 : EF = Line3D.mk E F) 
  (h2 : GH = Line3D.mk G H) : 
  ¬(areCoplanar E F G H) → ¬(linesIntersect EF GH) := 
sorry

theorem exists_not_intersect_but_coplanar :
  ∃ (E F G H : Point3D) (EF GH : Line3D),
    EF = Line3D.mk E F ∧ 
    GH = Line3D.mk G H ∧ 
    ¬(linesIntersect EF GH) ∧ 
    areCoplanar E F G H :=
sorry

end NUMINAMATH_CALUDE_not_coplanar_implies_not_intersect_exists_not_intersect_but_coplanar_l1738_173878


namespace NUMINAMATH_CALUDE_max_sum_of_digits_24hour_format_l1738_173890

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits in a Time24 -/
def sumOfDigitsTime24 (t : Time24) : ℕ :=
  sumOfDigits t.hours.val + sumOfDigits t.minutes.val

/-- The theorem stating the maximum sum of digits in a 24-hour format display -/
theorem max_sum_of_digits_24hour_format :
  ∃ (max : ℕ), ∀ (t : Time24), sumOfDigitsTime24 t ≤ max ∧
  ∃ (t' : Time24), sumOfDigitsTime24 t' = max ∧ max = 24 := by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_24hour_format_l1738_173890


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l1738_173869

/-- The cost of movie tickets for a family --/
theorem movie_ticket_cost (C : ℝ) : 
  (∃ (A : ℝ), 
    A = C + 3.25 ∧ 
    2 * A + 4 * C - 2 = 30) → 
  C = 4.25 := by
sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l1738_173869


namespace NUMINAMATH_CALUDE_r_value_when_n_is_2_l1738_173874

theorem r_value_when_n_is_2 (n : ℕ) (s r : ℕ) 
  (h1 : s = 3^(n^2) + 1) 
  (h2 : r = 5^s - s) 
  (h3 : n = 2) : 
  r = 5^82 - 82 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_2_l1738_173874


namespace NUMINAMATH_CALUDE_min_xy_m_range_l1738_173882

-- Define the conditions
def condition (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 1/x + 3/y = 2

-- Theorem for the minimum value of xy
theorem min_xy (x y : ℝ) (h : condition x y) : 
  ∀ a b : ℝ, condition a b → x * y ≤ a * b ∧ x * y ≥ 3 :=
sorry

-- Theorem for the range of m
theorem m_range (x y : ℝ) (h : condition x y) :
  ∀ m : ℝ, (∀ a b : ℝ, condition a b → 3*a + b ≥ m^2 - m) → 
  -2 ≤ m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_min_xy_m_range_l1738_173882


namespace NUMINAMATH_CALUDE_lindas_savings_l1738_173870

theorem lindas_savings (savings : ℝ) (tv_cost : ℝ) : 
  tv_cost = 240 →
  (1 / 4 : ℝ) * savings = tv_cost →
  savings = 960 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l1738_173870


namespace NUMINAMATH_CALUDE_total_apples_eaten_l1738_173822

def simone_daily_consumption : ℚ := 1/2
def simone_days : ℕ := 16
def lauri_daily_consumption : ℚ := 1/3
def lauri_days : ℕ := 15

theorem total_apples_eaten :
  (simone_daily_consumption * simone_days + lauri_daily_consumption * lauri_days : ℚ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_eaten_l1738_173822


namespace NUMINAMATH_CALUDE_smallest_m_for_nth_roots_in_T_l1738_173861

def T : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_roots_in_T : 
  ∃ m : ℕ+, (∀ n : ℕ+, n ≥ m → ∃ z ∈ T, z^(n:ℕ) = 1) ∧ 
  (∀ k : ℕ+, k < m → ∃ n : ℕ+, n ≥ k ∧ ∀ z ∈ T, z^(n:ℕ) ≠ 1) ∧
  m = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_nth_roots_in_T_l1738_173861


namespace NUMINAMATH_CALUDE_rate_increase_factor_l1738_173893

/-- Reaction rate equation -/
def reaction_rate (k : ℝ) (C_CO : ℝ) (C_O2 : ℝ) : ℝ :=
  k * C_CO^2 * C_O2

/-- Theorem: When concentrations triple, rate increases by factor of 27 -/
theorem rate_increase_factor (k : ℝ) (C_CO : ℝ) (C_O2 : ℝ) :
  reaction_rate k (3 * C_CO) (3 * C_O2) = 27 * reaction_rate k C_CO C_O2 := by
  sorry


end NUMINAMATH_CALUDE_rate_increase_factor_l1738_173893


namespace NUMINAMATH_CALUDE_c_oxen_count_l1738_173889

/-- Represents the number of oxen and months for each person --/
structure GrazingData where
  oxen : ℕ
  months : ℕ

/-- Calculates the total oxen-months for a given GrazingData --/
def oxenMonths (data : GrazingData) : ℕ := data.oxen * data.months

/-- Theorem: Given the conditions, c put 15 oxen for grazing --/
theorem c_oxen_count (total_rent : ℚ) (a b : GrazingData) (c_months : ℕ) (c_rent : ℚ) :
  total_rent = 210 →
  a = { oxen := 10, months := 7 } →
  b = { oxen := 12, months := 5 } →
  c_months = 3 →
  c_rent = 54 →
  ∃ (c_oxen : ℕ), 
    let c : GrazingData := { oxen := c_oxen, months := c_months }
    (c_rent / total_rent) * (oxenMonths a + oxenMonths b + oxenMonths c) = oxenMonths c ∧
    c_oxen = 15 := by
  sorry


end NUMINAMATH_CALUDE_c_oxen_count_l1738_173889


namespace NUMINAMATH_CALUDE_f_10_eq_3_div_5_l1738_173829

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (x : ℝ) (h : x > 0) : f x = 2 * f (1/x) * Real.log x + 1

theorem f_10_eq_3_div_5 : f 10 = 3/5 := by sorry

end NUMINAMATH_CALUDE_f_10_eq_3_div_5_l1738_173829


namespace NUMINAMATH_CALUDE_trigonometric_inequalities_l1738_173808

theorem trigonometric_inequalities (α β γ : ℝ) (h : α + β + γ = 0) :
  (|Real.cos (α + β)| ≤ |Real.cos α| + |Real.sin β|) ∧
  (|Real.sin (α + β)| ≤ |Real.cos α| + |Real.cos β|) ∧
  (|Real.cos α| + |Real.cos β| + |Real.cos γ| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequalities_l1738_173808


namespace NUMINAMATH_CALUDE_brother_age_problem_l1738_173810

theorem brother_age_problem (younger_age older_age : ℕ) : 
  younger_age + older_age = 26 → 
  older_age = younger_age + 2 → 
  older_age = 14 := by
sorry

end NUMINAMATH_CALUDE_brother_age_problem_l1738_173810


namespace NUMINAMATH_CALUDE_product_of_polynomials_l1738_173833

/-- Given two polynomials A(d) and B(d) whose product is C(d), prove that k + m = -4 --/
theorem product_of_polynomials (k m : ℚ) : 
  (∀ d : ℚ, (5*d^2 - 2*d + k) * (4*d^2 + m*d - 9) = 20*d^4 - 28*d^3 + 13*d^2 - m*d - 18) → 
  k + m = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l1738_173833


namespace NUMINAMATH_CALUDE_gemma_pizza_order_l1738_173844

/-- The number of pizzas Gemma ordered -/
def number_of_pizzas : ℕ := 4

/-- The cost of each pizza in dollars -/
def pizza_cost : ℕ := 10

/-- The tip amount in dollars -/
def tip_amount : ℕ := 5

/-- The amount Gemma paid with in dollars -/
def payment_amount : ℕ := 50

/-- The change Gemma received in dollars -/
def change_amount : ℕ := 5

theorem gemma_pizza_order :
  number_of_pizzas * pizza_cost + tip_amount = payment_amount - change_amount :=
sorry

end NUMINAMATH_CALUDE_gemma_pizza_order_l1738_173844


namespace NUMINAMATH_CALUDE_exponent_comparison_l1738_173817

theorem exponent_comparison : 65^1000 - 8^2001 > 0 := by
  sorry

end NUMINAMATH_CALUDE_exponent_comparison_l1738_173817


namespace NUMINAMATH_CALUDE_circle_equation_with_tangent_conditions_l1738_173863

/-- The standard equation of a circle with center on y = (1/2)x^2 and tangent to y = 0 and x = 0 -/
theorem circle_equation_with_tangent_conditions (t : ℝ) :
  (∃ (r : ℝ), r > 0 ∧
    (∀ (x y : ℝ), (x - t)^2 + (y - (1/2) * t^2)^2 = r^2 ↔
      ((x = 0 ∨ y = 0) → (x - t)^2 + (y - (1/2) * t^2)^2 = r^2))) →
  (∃ (s : ℝ), s = 1 ∨ s = -1) ∧
    (∀ (x y : ℝ), (x - s)^2 + (y - (1/2))^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_with_tangent_conditions_l1738_173863
