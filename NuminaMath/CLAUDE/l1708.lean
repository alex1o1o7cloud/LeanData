import Mathlib

namespace NUMINAMATH_CALUDE_cindys_homework_l1708_170809

theorem cindys_homework (x : ℝ) : (x - 7) * 4 = 48 → (x * 4) - 7 = 69 := by
  sorry

end NUMINAMATH_CALUDE_cindys_homework_l1708_170809


namespace NUMINAMATH_CALUDE_inverse_65_mod_66_l1708_170884

theorem inverse_65_mod_66 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 65 ∧ (65 * x) % 66 = 1 :=
by
  use 65
  sorry

end NUMINAMATH_CALUDE_inverse_65_mod_66_l1708_170884


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_equals_three_sqrt_two_l1708_170893

theorem sqrt_six_times_sqrt_three_equals_three_sqrt_two :
  Real.sqrt 6 * Real.sqrt 3 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_equals_three_sqrt_two_l1708_170893


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1708_170885

theorem necessary_not_sufficient (a b : ℝ) : 
  (a > b → a > b - 1) ∧ ¬(a > b - 1 → a > b) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1708_170885


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1708_170878

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (5 * x₁^2 - 7 * x₁ + 2 = 0) → 
  (5 * x₂^2 - 7 * x₂ + 2 = 0) → 
  (x₁^2 + x₂^2 = 29/25) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1708_170878


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1708_170828

/-- Given a triangle with inradius 2.5 cm and area 25 cm², prove its perimeter is 20 cm -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) 
  (h1 : r = 2.5)
  (h2 : A = 25)
  (h3 : A = r * (p / 2)) :
  p = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1708_170828


namespace NUMINAMATH_CALUDE_weight_increase_percentage_shyam_weight_increase_percentage_l1708_170834

theorem weight_increase_percentage (ram_ratio : ℝ) (shyam_ratio : ℝ) 
  (ram_increase : ℝ) (total_weight : ℝ) (total_increase : ℝ) : ℝ :=
  let original_total := total_weight / (1 + total_increase / 100)
  let x := original_total / (ram_ratio + shyam_ratio)
  let ram_original := ram_ratio * x
  let shyam_original := shyam_ratio * x
  let ram_new := ram_original * (1 + ram_increase / 100)
  let shyam_new := total_weight - ram_new
  (shyam_new - shyam_original) / shyam_original * 100

/-- Given the weights of Ram and Shyam in a 7:5 ratio, Ram's weight increased by 10%,
    and the total weight after increase is 82.8 kg with a 15% total increase,
    prove that Shyam's weight increase percentage is 22%. -/
theorem shyam_weight_increase_percentage :
  weight_increase_percentage 7 5 10 82.8 15 = 22 := by
  sorry

end NUMINAMATH_CALUDE_weight_increase_percentage_shyam_weight_increase_percentage_l1708_170834


namespace NUMINAMATH_CALUDE_infinite_greater_than_index_l1708_170870

/-- A sequence of integers -/
def IntegerSequence := ℕ → ℤ

/-- Property: all elements in the sequence are pairwise distinct -/
def PairwiseDistinct (a : IntegerSequence) : Prop :=
  ∀ i j, i ≠ j → a i ≠ a j

/-- Property: all elements in the sequence are greater than 1 -/
def AllGreaterThanOne (a : IntegerSequence) : Prop :=
  ∀ k, a k > 1

/-- The main theorem -/
theorem infinite_greater_than_index
  (a : IntegerSequence)
  (h_distinct : PairwiseDistinct a)
  (h_greater : AllGreaterThanOne a) :
  ∃ S : Set ℕ, (Set.Infinite S) ∧ (∀ k ∈ S, a k > k) :=
sorry

end NUMINAMATH_CALUDE_infinite_greater_than_index_l1708_170870


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l1708_170857

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≤ 0}

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3*x^2 + 1}

-- State the theorem
theorem intersection_M_complement_N : 
  M ∩ (U \ N) = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l1708_170857


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1708_170836

def geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) = 2 * a n

theorem geometric_sequence_sum (a : ℕ+ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 1 + a 3 = 2) : 
  a 5 + a 7 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1708_170836


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l1708_170858

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 87 → books_sold = 33 → books_per_shelf = 6 → 
  (initial_stock - books_sold) / books_per_shelf = 9 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l1708_170858


namespace NUMINAMATH_CALUDE_rational_root_iff_k_eq_neg_two_or_zero_l1708_170886

/-- The polynomial X^2017 - X^2016 + X^2 + kX + 1 has a rational root if and only if k = -2 or k = 0 -/
theorem rational_root_iff_k_eq_neg_two_or_zero (k : ℚ) :
  (∃ x : ℚ, x^2017 - x^2016 + x^2 + k*x + 1 = 0) ↔ (k = -2 ∨ k = 0) := by
  sorry

end NUMINAMATH_CALUDE_rational_root_iff_k_eq_neg_two_or_zero_l1708_170886


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1708_170826

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|x| - 2 - |-1| = 2) ↔ (x = 5 ∨ x = -5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1708_170826


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1708_170800

/-- The speed of a boat in still water given its travel distances with and against a stream. -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ) 
  (h_along : along_stream = 11) 
  (h_against : against_stream = 3) : 
  (along_stream + against_stream) / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1708_170800


namespace NUMINAMATH_CALUDE_certain_number_calculation_l1708_170806

theorem certain_number_calculation (x y : ℝ) : 
  0.12 / x * 2 = y → x = 0.1 → y = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l1708_170806


namespace NUMINAMATH_CALUDE_area_at_stage_8_l1708_170841

/-- Represents the width of a rectangle at a given stage -/
def width (stage : ℕ) : ℕ :=
  if stage ≤ 4 then 4 else 2 * stage - 6

/-- Represents the area of a rectangle at a given stage -/
def area (stage : ℕ) : ℕ := 4 * width stage

/-- The total area of the figure at Stage 8 -/
def totalArea : ℕ := (List.range 8).map (fun i => area (i + 1)) |>.sum

theorem area_at_stage_8 : totalArea = 176 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l1708_170841


namespace NUMINAMATH_CALUDE_sunflower_contest_total_l1708_170859

/-- The total number of seeds eaten in a three-player sunflower eating contest -/
def total_seeds (player1_seeds player2_seeds extra_seeds : ℕ) : ℕ :=
  player1_seeds + player2_seeds + (player2_seeds + extra_seeds)

/-- Theorem stating the total number of seeds eaten in the specific contest scenario -/
theorem sunflower_contest_total : 
  total_seeds 78 53 30 = 214 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_contest_total_l1708_170859


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l1708_170849

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_condition
  (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (a 3 * a 5 = 16 → a 4 = 4) ∧ 
  ¬(a 4 = 4 → a 3 * a 5 = 16) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l1708_170849


namespace NUMINAMATH_CALUDE_max_towns_is_four_l1708_170829

/-- Represents the type of link between two towns -/
inductive LinkType
| Air
| Bus
| Train

/-- Represents a town -/
structure Town where
  id : Nat

/-- Represents a link between two towns -/
structure Link where
  town1 : Town
  town2 : Town
  linkType : LinkType

/-- A network of towns and their connections -/
structure TownNetwork where
  towns : List Town
  links : List Link

/-- Checks if a given network satisfies all the required conditions -/
def isValidNetwork (network : TownNetwork) : Prop :=
  -- Each pair of towns is linked by exactly one type of link
  (∀ t1 t2 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t1 ≠ t2 →
    ∃! link : Link, link ∈ network.links ∧ 
    ((link.town1 = t1 ∧ link.town2 = t2) ∨ (link.town1 = t2 ∧ link.town2 = t1))) ∧
  -- At least one pair is linked by each type
  (∃ link : Link, link ∈ network.links ∧ link.linkType = LinkType.Air) ∧
  (∃ link : Link, link ∈ network.links ∧ link.linkType = LinkType.Bus) ∧
  (∃ link : Link, link ∈ network.links ∧ link.linkType = LinkType.Train) ∧
  -- No town has all three types of links
  (∀ t : Town, t ∈ network.towns →
    ¬(∃ l1 l2 l3 : Link, l1 ∈ network.links ∧ l2 ∈ network.links ∧ l3 ∈ network.links ∧
      (l1.town1 = t ∨ l1.town2 = t) ∧ (l2.town1 = t ∨ l2.town2 = t) ∧ (l3.town1 = t ∨ l3.town2 = t) ∧
      l1.linkType = LinkType.Air ∧ l2.linkType = LinkType.Bus ∧ l3.linkType = LinkType.Train)) ∧
  -- No three towns form a triangle with all sides of the same type
  (∀ t1 t2 t3 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t3 ∈ network.towns →
    t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 →
    ¬(∃ l1 l2 l3 : Link, l1 ∈ network.links ∧ l2 ∈ network.links ∧ l3 ∈ network.links ∧
      ((l1.town1 = t1 ∧ l1.town2 = t2) ∨ (l1.town1 = t2 ∧ l1.town2 = t1)) ∧
      ((l2.town1 = t2 ∧ l2.town2 = t3) ∨ (l2.town1 = t3 ∧ l2.town2 = t2)) ∧
      ((l3.town1 = t3 ∧ l3.town2 = t1) ∨ (l3.town1 = t1 ∧ l3.town2 = t3)) ∧
      l1.linkType = l2.linkType ∧ l2.linkType = l3.linkType))

/-- The theorem stating that the maximum number of towns in a valid network is 4 -/
theorem max_towns_is_four :
  (∃ network : TownNetwork, isValidNetwork network ∧ network.towns.length = 4) ∧
  (∀ network : TownNetwork, isValidNetwork network → network.towns.length ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_towns_is_four_l1708_170829


namespace NUMINAMATH_CALUDE_michael_singles_percentage_l1708_170892

/-- Calculates the percentage of singles in a player's hits -/
def percentage_singles (total_hits : ℕ) (home_runs triples doubles : ℕ) : ℚ :=
  let non_singles := home_runs + triples + doubles
  let singles := total_hits - non_singles
  (singles : ℚ) / (total_hits : ℚ) * 100

theorem michael_singles_percentage :
  percentage_singles 50 2 3 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_michael_singles_percentage_l1708_170892


namespace NUMINAMATH_CALUDE_only_translation_preserves_pattern_l1708_170824

/-- Represents the types of figures in the pattern -/
inductive Figure
| Triangle
| Square

/-- Represents a point on the line ℓ -/
structure PointOnLine where
  position : ℝ

/-- Represents the infinite pattern on line ℓ -/
def Pattern := ℕ → Figure

/-- Represents the possible rigid motion transformations -/
inductive RigidMotion
| Rotation (center : PointOnLine) (angle : ℝ)
| Translation (distance : ℝ)
| ReflectionAcrossL
| ReflectionPerpendicular (point : PointOnLine)

/-- Defines the alternating pattern of triangles and squares -/
def alternatingPattern : Pattern :=
  fun n => if n % 2 = 0 then Figure.Triangle else Figure.Square

/-- Checks if a rigid motion preserves the pattern -/
def preservesPattern (motion : RigidMotion) (pattern : Pattern) : Prop :=
  ∀ n, pattern n = pattern (n + 1)  -- This is a simplification; actual preservation would be more complex

/-- The main theorem stating that only translation preserves the pattern -/
theorem only_translation_preserves_pattern :
  ∀ motion : RigidMotion,
    preservesPattern motion alternatingPattern ↔ ∃ d, motion = RigidMotion.Translation d :=
sorry

end NUMINAMATH_CALUDE_only_translation_preserves_pattern_l1708_170824


namespace NUMINAMATH_CALUDE_book_pages_l1708_170842

theorem book_pages : 
  ∀ (P : ℕ), 
  (7 : ℚ) / 13 * P + (5 : ℚ) / 9 * ((6 : ℚ) / 13 * P) + 96 = P → 
  P = 468 :=
by
  sorry

end NUMINAMATH_CALUDE_book_pages_l1708_170842


namespace NUMINAMATH_CALUDE_total_points_is_238_l1708_170827

/-- Represents a player's statistics in the basketball game -/
structure PlayerStats :=
  (two_pointers : ℕ)
  (three_pointers : ℕ)
  (free_throws : ℕ)
  (steals : ℕ)
  (rebounds : ℕ)
  (fouls : ℕ)

/-- Calculates the total points for a player given their stats -/
def calculate_points (stats : PlayerStats) : ℤ :=
  2 * stats.two_pointers + 3 * stats.three_pointers + stats.free_throws +
  stats.steals + 2 * stats.rebounds - 5 * stats.fouls

/-- The main theorem to prove -/
theorem total_points_is_238 :
  let sam := PlayerStats.mk 20 10 5 4 6 2
  let alex := PlayerStats.mk 15 8 5 6 3 3
  let jake := PlayerStats.mk 10 6 3 7 5 4
  let lily := PlayerStats.mk 16 4 7 3 7 1
  calculate_points sam + calculate_points alex + calculate_points jake + calculate_points lily = 238 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_238_l1708_170827


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l1708_170821

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l1708_170821


namespace NUMINAMATH_CALUDE_product_sum_relation_l1708_170875

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 14 → b = 8 → b - a = 3 := by sorry

end NUMINAMATH_CALUDE_product_sum_relation_l1708_170875


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1708_170820

/-- A geometric sequence with a_2 = 2 and a_10 = 8 has a_6 = 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 2 = 2 →
  a 10 = 8 →
  a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1708_170820


namespace NUMINAMATH_CALUDE_three_power_plus_one_not_divisible_l1708_170830

theorem three_power_plus_one_not_divisible (n : ℕ) 
  (h_odd : Odd n) (h_gt_one : n > 1) : ¬(n ∣ 3^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_three_power_plus_one_not_divisible_l1708_170830


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l1708_170852

theorem product_from_lcm_gcd : 
  ∀ (a b : ℕ+), 
    Nat.lcm a b = 72 → 
    Nat.gcd a b = 8 → 
    (a : ℕ) * b = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l1708_170852


namespace NUMINAMATH_CALUDE_shoe_picking_probability_l1708_170822

def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def blue_pairs : ℕ := 4
def green_pairs : ℕ := 3

def total_shoes : ℕ := 2 * total_pairs

theorem shoe_picking_probability :
  let black_shoes : ℕ := 2 * black_pairs
  let blue_shoes : ℕ := 2 * blue_pairs
  let green_shoes : ℕ := 2 * green_pairs
  let prob_black := (black_shoes : ℚ) / total_shoes * (black_pairs : ℚ) / (total_shoes - 1)
  let prob_blue := (blue_shoes : ℚ) / total_shoes * (blue_pairs : ℚ) / (total_shoes - 1)
  let prob_green := (green_shoes : ℚ) / total_shoes * (green_pairs : ℚ) / (total_shoes - 1)
  prob_black + prob_blue + prob_green = 89 / 435 :=
by sorry

end NUMINAMATH_CALUDE_shoe_picking_probability_l1708_170822


namespace NUMINAMATH_CALUDE_min_blocks_for_cube_l1708_170844

/-- The length of the rectangular block -/
def block_length : ℕ := 5

/-- The width of the rectangular block -/
def block_width : ℕ := 4

/-- The height of the rectangular block -/
def block_height : ℕ := 3

/-- The side length of the cube formed by the blocks -/
def cube_side : ℕ := Nat.lcm (Nat.lcm block_length block_width) block_height

/-- The volume of the cube -/
def cube_volume : ℕ := cube_side ^ 3

/-- The volume of a single block -/
def block_volume : ℕ := block_length * block_width * block_height

/-- The number of blocks needed to form the cube -/
def blocks_needed : ℕ := cube_volume / block_volume

theorem min_blocks_for_cube : blocks_needed = 3600 := by
  sorry

end NUMINAMATH_CALUDE_min_blocks_for_cube_l1708_170844


namespace NUMINAMATH_CALUDE_anne_bottle_caps_l1708_170816

/-- Anne's initial bottle cap count -/
def initial_count : ℕ := 10

/-- Number of bottle caps Anne finds -/
def found_count : ℕ := 5

/-- Anne's final bottle cap count -/
def final_count : ℕ := initial_count + found_count

theorem anne_bottle_caps : final_count = 15 := by
  sorry

end NUMINAMATH_CALUDE_anne_bottle_caps_l1708_170816


namespace NUMINAMATH_CALUDE_min_value_problem_l1708_170883

/-- Given positive real numbers a, b, c, and a function f with minimum value 4,
    prove that a + b + c = 4 and the minimum value of (1/4)a² + (1/9)b² + c² is 8/7 -/
theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 →
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1708_170883


namespace NUMINAMATH_CALUDE_min_shift_for_sine_overlap_l1708_170805

theorem min_shift_for_sine_overlap (f g : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + π / 3)) →
  (∀ x, g x = Real.sin (2 * x)) →
  (∀ x, f x = g (x + π / 6)) →
  (∀ x, f x = g (x + φ)) →
  φ > 0 →
  φ ≥ π / 6 :=
sorry

end NUMINAMATH_CALUDE_min_shift_for_sine_overlap_l1708_170805


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1708_170868

theorem quadratic_inequality_solution (x : ℝ) : 3 * x^2 - 2 * x + 1 > 7 ↔ x < -2/3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1708_170868


namespace NUMINAMATH_CALUDE_richard_remaining_distance_l1708_170819

/-- Calculates the remaining distance Richard has to walk to reach New York City. -/
def remaining_distance (total_distance day1_distance day2_fraction day2_reduction day3_distance : ℝ) : ℝ :=
  let day2_distance := day1_distance * day2_fraction - day2_reduction
  let distance_walked := day1_distance + day2_distance + day3_distance
  total_distance - distance_walked

/-- Theorem stating that Richard has 36 miles left to walk to reach New York City. -/
theorem richard_remaining_distance :
  remaining_distance 70 20 (1/2) 6 10 = 36 := by
  sorry

end NUMINAMATH_CALUDE_richard_remaining_distance_l1708_170819


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l1708_170846

theorem quadratic_negative_root (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ x^2 + m*x - 4 = 0) ↔ m > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l1708_170846


namespace NUMINAMATH_CALUDE_ab_value_l1708_170890

theorem ab_value (a b c d : ℝ) 
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 25)
  (h3 : a = 2*c + Real.sqrt d) :
  a * b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1708_170890


namespace NUMINAMATH_CALUDE_parabola_sum_l1708_170897

/-- A parabola with equation y = px^2 + qx + r -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.yCoord (para : Parabola) (x : ℝ) : ℝ :=
  para.p * x^2 + para.q * x + para.r

theorem parabola_sum (para : Parabola) :
  para.yCoord 3 = 2 →   -- Vertex at (3, 2)
  para.yCoord 1 = 6 →   -- Passes through (1, 6)
  para.p + para.q + para.r = 6 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l1708_170897


namespace NUMINAMATH_CALUDE_triangle_inequality_tangent_l1708_170847

theorem triangle_inequality_tangent (a b c α β : ℝ) 
  (h : a + b < 3 * c) : 
  Real.tan (α / 2) * Real.tan (β / 2) < 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_tangent_l1708_170847


namespace NUMINAMATH_CALUDE_negative_roots_equation_reciprocal_roots_equation_l1708_170812

-- Part 1
theorem negative_roots_equation (r1 r2 : ℝ) :
  r1^2 + 3*r1 - 2 = 0 ∧ r2^2 + 3*r2 - 2 = 0 →
  (-r1)^2 - 3*(-r1) - 2 = 0 ∧ (-r2)^2 - 3*(-r2) - 2 = 0 := by sorry

-- Part 2
theorem reciprocal_roots_equation (a b c r1 r2 : ℝ) :
  a ≠ 0 ∧ r1 ≠ r2 ∧ r1 ≠ 0 ∧ r2 ≠ 0 ∧
  a*r1^2 - b*r1 + c = 0 ∧ a*r2^2 - b*r2 + c = 0 →
  c*(1/r1)^2 - b*(1/r1) + a = 0 ∧ c*(1/r2)^2 - b*(1/r2) + a = 0 := by sorry

end NUMINAMATH_CALUDE_negative_roots_equation_reciprocal_roots_equation_l1708_170812


namespace NUMINAMATH_CALUDE_direction_525_to_527_l1708_170823

/-- Represents the directions of movement -/
inductive Direction
| Right
| Up
| Left
| Down
| Diagonal

/-- Defines the cyclic pattern of directions -/
def directionPattern : Fin 5 → Direction
| 0 => Direction.Right
| 1 => Direction.Up
| 2 => Direction.Left
| 3 => Direction.Down
| 4 => Direction.Diagonal

/-- Returns the direction for a given point number -/
def directionAtPoint (n : Nat) : Direction :=
  directionPattern (n % 5)

/-- Theorem: The sequence of directions from point 525 to 527 is Right, Up -/
theorem direction_525_to_527 :
  (directionAtPoint 525, directionAtPoint 526) = (Direction.Right, Direction.Up) := by
  sorry

#check direction_525_to_527

end NUMINAMATH_CALUDE_direction_525_to_527_l1708_170823


namespace NUMINAMATH_CALUDE_brick_length_calculation_l1708_170854

/-- Calculates the length of a brick given wall dimensions, partial brick dimensions, and number of bricks --/
theorem brick_length_calculation (wall_length wall_width wall_height brick_width brick_height num_bricks : ℝ) :
  wall_length = 800 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_width = 11.25 →
  brick_height = 6 →
  num_bricks = 3200 →
  (wall_length * wall_width * wall_height) / (num_bricks * brick_width * brick_height) = 50 := by
  sorry

#check brick_length_calculation

end NUMINAMATH_CALUDE_brick_length_calculation_l1708_170854


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l1708_170898

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l1708_170898


namespace NUMINAMATH_CALUDE_correct_categorization_l1708_170835

-- Define the teams
def IntegerTeam : Set ℝ := {0, -8}
def FractionTeam : Set ℝ := {1/7, 0.505}
def IrrationalTeam : Set ℝ := {Real.sqrt 13, Real.pi}

-- Define the properties for each team
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n
def isFraction (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b
def isIrrational (x : ℝ) : Prop := ¬(isInteger x ∨ isFraction x)

-- Theorem to prove the correct categorization
theorem correct_categorization :
  (∀ x ∈ IntegerTeam, isInteger x) ∧
  (∀ x ∈ FractionTeam, isFraction x) ∧
  (∀ x ∈ IrrationalTeam, isIrrational x) :=
  sorry

end NUMINAMATH_CALUDE_correct_categorization_l1708_170835


namespace NUMINAMATH_CALUDE_puzzle_cost_calculation_l1708_170808

def puzzle_cost (initial_money savings comic_cost final_money : ℕ) : ℕ :=
  initial_money + savings - comic_cost - final_money

theorem puzzle_cost_calculation :
  puzzle_cost 8 13 2 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_cost_calculation_l1708_170808


namespace NUMINAMATH_CALUDE_towels_folded_in_one_hour_l1708_170882

/-- Represents the number of towels a person can fold in one hour --/
def towels_per_hour (
  jane_rate : ℕ → ℕ
) (
  kyla_rate : ℕ → ℕ
) (
  anthony_rate : ℕ → ℕ
) (
  david_rate : ℕ → ℕ
) : ℕ :=
  jane_rate 60 + kyla_rate 60 + anthony_rate 60 + david_rate 60

/-- Jane's folding rate: 5 towels in 5 minutes, 3-minute break after every 5 minutes --/
def jane_rate (minutes : ℕ) : ℕ :=
  (minutes / 8) * 5

/-- Kyla's folding rate: 12 towels in 10 minutes for first 30 minutes, then 6 towels in 10 minutes --/
def kyla_rate (minutes : ℕ) : ℕ :=
  min 36 (minutes / 10 * 12) + max 0 ((minutes - 30) / 10 * 6)

/-- Anthony's folding rate: 14 towels in 20 minutes, 10-minute break after 40 minutes --/
def anthony_rate (minutes : ℕ) : ℕ :=
  (min minutes 40) / 20 * 14

/-- David's folding rate: 4 towels in 15 minutes, speed increases by 1 towel per 15 minutes for every 3 sets --/
def david_rate (minutes : ℕ) : ℕ :=
  (minutes / 15) * 4 + (minutes / 45)

theorem towels_folded_in_one_hour :
  towels_per_hour jane_rate kyla_rate anthony_rate david_rate = 134 := by
  sorry

end NUMINAMATH_CALUDE_towels_folded_in_one_hour_l1708_170882


namespace NUMINAMATH_CALUDE_parabola_focus_theorem_l1708_170899

/-- Parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- Point on the parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * c.p * x

/-- Circle tangent to y-axis and intersecting MF -/
structure TangentCircle (c : Parabola) (m : PointOnParabola c) where
  a : ℝ × ℝ  -- Point A
  tangent_to_y_axis : sorry
  intersects_mf : sorry

/-- Theorem: Given the conditions, p = 2 -/
theorem parabola_focus_theorem (c : Parabola) 
    (m : PointOnParabola c)
    (h_m : m.y = 2 * Real.sqrt 2)
    (circle : TangentCircle c m)
    (h_ratio : (Real.sqrt ((m.x - circle.a.1)^2 + (m.y - circle.a.2)^2)) / 
               (Real.sqrt ((c.p - circle.a.1)^2 + circle.a.2^2)) = 2) :
  c.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_theorem_l1708_170899


namespace NUMINAMATH_CALUDE_same_solution_value_of_c_l1708_170839

theorem same_solution_value_of_c : 
  ∀ x c : ℚ, (3 * x + 4 = 2 ∧ c * x - 15 = 0) → c = -45/2 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_value_of_c_l1708_170839


namespace NUMINAMATH_CALUDE_leahs_coin_value_l1708_170803

/-- Represents the number and value of coins --/
structure CoinCollection where
  pennies : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents --/
def totalValue (coins : CoinCollection) : ℕ :=
  coins.pennies + 25 * coins.quarters

/-- Theorem stating the value of Leah's coin collection --/
theorem leahs_coin_value :
  ∀ (coins : CoinCollection),
    coins.pennies + coins.quarters = 15 →
    coins.pennies = 2 * (coins.quarters + 1) →
    totalValue coins = 110 := by
  sorry


end NUMINAMATH_CALUDE_leahs_coin_value_l1708_170803


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l1708_170848

theorem opposite_of_fraction (n : ℕ) (h : n ≠ 0) : 
  -(1 : ℚ) / n = -(1 / n) := by sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l1708_170848


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l1708_170862

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ x - y + 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ x - y + 3 = 0
  ∀ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ → l₂ x₂ y₂ →
  (∃ (k : ℝ), ∀ (x y : ℝ), l₁ x y ↔ l₂ (x + k) (y + k)) →
  Real.sqrt 2 = |x₂ - x₁| :=
by sorry

end NUMINAMATH_CALUDE_distance_between_parallel_lines_l1708_170862


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l1708_170865

/-- The trajectory of point P given the symmetry of points A and B and the product of slopes condition -/
theorem trajectory_of_point_P (x y : ℝ) : 
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (1, -1)
  let P : ℝ × ℝ := (x, y)
  let slope_AP := (y - A.2) / (x - A.1)
  let slope_BP := (y - B.2) / (x - B.1)
  x ≠ 1 ∧ x ≠ -1 →
  slope_AP * slope_BP = 1/3 →
  3 * y^2 - x^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l1708_170865


namespace NUMINAMATH_CALUDE_square_difference_pattern_l1708_170811

theorem square_difference_pattern (n : ℕ) : (n + 1)^2 - n^2 = 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l1708_170811


namespace NUMINAMATH_CALUDE_segment_length_l1708_170843

/-- Given points P, Q, and R on line segment AB, prove that AB has length 48 -/
theorem segment_length (A B P Q R : ℝ) : 
  (0 < A) → (A < P) → (P < Q) → (Q < R) → (R < B) →  -- Points lie on AB in order
  (P - A) / (B - P) = 3 / 5 →                        -- P divides AB in ratio 3:5
  (Q - A) / (B - Q) = 5 / 7 →                        -- Q divides AB in ratio 5:7
  R - Q = 3 →                                        -- QR = 3
  R - P = 5 →                                        -- PR = 5
  B - A = 48 := by sorry

end NUMINAMATH_CALUDE_segment_length_l1708_170843


namespace NUMINAMATH_CALUDE_A_power_50_l1708_170817

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  !![5, 2; -16, -6]

theorem A_power_50 : A^50 = !![301, 100; -800, -249] := by
  sorry

end NUMINAMATH_CALUDE_A_power_50_l1708_170817


namespace NUMINAMATH_CALUDE_pythagorean_theorem_l1708_170877

-- Define a right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleBAC_is_right : angleBAC = 90

-- State the theorem
theorem pythagorean_theorem (t : RightTriangle) : t.b^2 + t.c^2 = t.a^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_l1708_170877


namespace NUMINAMATH_CALUDE_problem_solution_l1708_170863

theorem problem_solution (x y : ℝ) (h1 : x^2 + 4 = y - 2) (h2 : x = 6) : y = 42 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1708_170863


namespace NUMINAMATH_CALUDE_pamela_sugar_amount_l1708_170807

/-- The amount of sugar Pamela spilled in ounces -/
def sugar_spilled : ℝ := 5.2

/-- The amount of sugar Pamela has left in ounces -/
def sugar_left : ℝ := 4.6

/-- The initial amount of sugar Pamela bought in ounces -/
def initial_sugar : ℝ := sugar_spilled + sugar_left

theorem pamela_sugar_amount : initial_sugar = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_pamela_sugar_amount_l1708_170807


namespace NUMINAMATH_CALUDE_power_function_property_l1708_170895

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) (h2 : f 4 = 2) : f 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l1708_170895


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_three_l1708_170845

theorem least_positive_integer_to_multiple_of_three : 
  ∃ (n : ℕ), n > 0 ∧ (575 + n) % 3 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (575 + m) % 3 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_three_l1708_170845


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1708_170814

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 →
  y ≠ 0 →
  d ≠ 0 →
  8 * x - 6 * y = c →
  10 * y - 15 * x = d →
  c / d = -2 / 5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1708_170814


namespace NUMINAMATH_CALUDE_inverse_of_M_l1708_170869

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 0; 0, -1]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![4, 1; 2, 3]
def M : Matrix (Fin 2) (Fin 2) ℚ := B * A

theorem inverse_of_M :
  M⁻¹ = !![3/10, -1/10; 1/5, -2/5] := by sorry

end NUMINAMATH_CALUDE_inverse_of_M_l1708_170869


namespace NUMINAMATH_CALUDE_eva_age_is_six_l1708_170860

-- Define the set of ages
def ages : Finset ℕ := {2, 4, 6, 8, 10}

-- Define the condition for park visit
def park_visit (a b : ℕ) : Prop := a + b = 12 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

-- Define the condition for concert visit
def concert_visit : Prop := 2 ∈ ages ∧ 10 ∈ ages

-- Define the condition for staying home
def stay_home (eva_age : ℕ) : Prop := eva_age ∈ ages ∧ 4 ∈ ages

-- Theorem statement
theorem eva_age_is_six :
  ∃ (a b : ℕ), park_visit a b ∧ concert_visit ∧ stay_home 6 →
  ∃! (eva_age : ℕ), eva_age ∈ ages ∧ eva_age ≠ 2 ∧ eva_age ≠ 4 ∧ eva_age ≠ 8 ∧ eva_age ≠ 10 :=
by sorry

end NUMINAMATH_CALUDE_eva_age_is_six_l1708_170860


namespace NUMINAMATH_CALUDE_digit_placement_l1708_170866

theorem digit_placement (n : ℕ) (h : n < 10) :
  100 + 10 * n + 1 = 101 + 10 * n := by
  sorry

end NUMINAMATH_CALUDE_digit_placement_l1708_170866


namespace NUMINAMATH_CALUDE_entree_dessert_cost_difference_l1708_170850

/-- Given Hannah's restaurant bill, prove the cost difference between entree and dessert -/
theorem entree_dessert_cost_difference 
  (total_cost : ℕ) 
  (entree_cost : ℕ) 
  (h1 : total_cost = 23)
  (h2 : entree_cost = 14) :
  entree_cost - (total_cost - entree_cost) = 5 := by
  sorry

#check entree_dessert_cost_difference

end NUMINAMATH_CALUDE_entree_dessert_cost_difference_l1708_170850


namespace NUMINAMATH_CALUDE_sum_squares_units_digit_3003_l1708_170896

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def square (n : ℕ) : ℕ := n * n

def units_digit (n : ℕ) : ℕ := n % 10

theorem sum_squares_units_digit_3003 :
  units_digit (List.sum (List.map square (first_n_odd_integers 3003))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_units_digit_3003_l1708_170896


namespace NUMINAMATH_CALUDE_square_root_equation_l1708_170837

theorem square_root_equation (t s : ℝ) : t = 15 * s^2 ∧ t = 3.75 → s = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l1708_170837


namespace NUMINAMATH_CALUDE_triangle_least_perimeter_l1708_170856

theorem triangle_least_perimeter (a b x : ℕ) : 
  a = 15 → b = 24 → x > 0 → 
  a + x > b → b + x > a → a + b > x → 
  (∀ y : ℕ, y > 0 → y + a > b → b + y > a → a + b > y → a + b + y ≥ a + b + x) →
  a + b + x = 49 :=
sorry

end NUMINAMATH_CALUDE_triangle_least_perimeter_l1708_170856


namespace NUMINAMATH_CALUDE_min_value_E_l1708_170864

theorem min_value_E (E : ℝ) : 
  (∃ x : ℝ, ∀ y : ℝ, |E| + |y + 7| + |y - 5| ≥ |E| + |x + 7| + |x - 5| ∧ |E| + |x + 7| + |x - 5| = 12) →
  |E| ≥ 0 ∧ ∀ δ > 0, ∃ x : ℝ, |E| + |x + 7| + |x - 5| < 12 + δ :=
by sorry

end NUMINAMATH_CALUDE_min_value_E_l1708_170864


namespace NUMINAMATH_CALUDE_chess_match_average_time_l1708_170876

/-- Proves that in a chess match with given conditions, one player's average move time is 28 seconds -/
theorem chess_match_average_time (total_moves : ℕ) (opponent_avg_time : ℕ) (match_duration : ℕ) :
  total_moves = 30 →
  opponent_avg_time = 40 →
  match_duration = 17 * 60 →
  ∃ (player_avg_time : ℕ), player_avg_time = 28 ∧ 
    (total_moves / 2) * (player_avg_time + opponent_avg_time) = match_duration := by
  sorry

end NUMINAMATH_CALUDE_chess_match_average_time_l1708_170876


namespace NUMINAMATH_CALUDE_impossibility_of_simultaneous_inequalities_l1708_170831

theorem impossibility_of_simultaneous_inequalities 
  (a b c : Real) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  ¬(a * (1 - b) > 1/4 ∧ b * (1 - c) > 1/4 ∧ c * (1 - a) > 1/4) := by
sorry

end NUMINAMATH_CALUDE_impossibility_of_simultaneous_inequalities_l1708_170831


namespace NUMINAMATH_CALUDE_candy_distribution_l1708_170810

/-- Represents the number of positions moved for the k-th candy distribution -/
def a (k : ℕ) : ℕ := k * (k + 1) / 2

/-- Checks if all students in a circle of size n receive a candy -/
def all_receive_candy (n : ℕ) : Prop :=
  ∀ m : ℕ, m < n → ∃ k : ℕ, a k % n = m

/-- Main theorem: All students receive a candy iff n is a power of 2 -/
theorem candy_distribution (n : ℕ) :
  all_receive_candy n ↔ ∃ m : ℕ, n = 2^m :=
sorry

/-- Helper lemma: If n is not a power of 2, not all students receive a candy -/
lemma not_power_of_two_not_all_receive (n : ℕ) :
  (¬ ∃ m : ℕ, n = 2^m) → ¬ all_receive_candy n :=
sorry

/-- Helper lemma: If n is a power of 2, all students receive a candy -/
lemma power_of_two_all_receive (m : ℕ) :
  all_receive_candy (2^m) :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l1708_170810


namespace NUMINAMATH_CALUDE_cubic_polynomial_inequality_l1708_170867

/-- 
A cubic polynomial with real coefficients that has three real roots 
satisfies the inequality 6a^3 + 10(a^2 - 2b)^(3/2) - 12ab ≥ 27c, 
with equality if and only if b = 0, c = -4/27 * a^3, and a ≤ 0.
-/
theorem cubic_polynomial_inequality (a b c : ℝ) : 
  (∃ x y z : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ 
               y^3 + a*y^2 + b*y + c = 0 ∧ 
               z^3 + a*z^2 + b*z + c = 0 ∧ 
               x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  6*a^3 + 10*(a^2 - 2*b)^(3/2) - 12*a*b ≥ 27*c ∧
  (6*a^3 + 10*(a^2 - 2*b)^(3/2) - 12*a*b = 27*c ↔ b = 0 ∧ c = -4/27 * a^3 ∧ a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_inequality_l1708_170867


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l1708_170887

theorem power_tower_mod_500 : 5^(5^(5^5)) ≡ 125 [ZMOD 500] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l1708_170887


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1708_170894

theorem gcd_polynomial_and_multiple (x : ℤ) : 
  36000 ∣ x → 
  Nat.gcd ((5*x + 3)*(11*x + 2)*(6*x + 7)*(3*x + 8) : ℤ).natAbs x.natAbs = 144 := by
sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1708_170894


namespace NUMINAMATH_CALUDE_division_simplification_l1708_170838

theorem division_simplification : (180 : ℚ) / (12 + 9 * 3 - 4) = 36 / 7 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l1708_170838


namespace NUMINAMATH_CALUDE_decimal_representation_of_fraction_l1708_170853

theorem decimal_representation_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 16 / 50 → (n : ℚ) / d = 0.32 := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_of_fraction_l1708_170853


namespace NUMINAMATH_CALUDE_tangent_slope_at_pi_l1708_170815

theorem tangent_slope_at_pi (f : ℝ → ℝ) (h : f = λ x => 2*x + Real.sin x) :
  HasDerivAt f 1 π := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_pi_l1708_170815


namespace NUMINAMATH_CALUDE_complex_distance_range_l1708_170871

theorem complex_distance_range (z : ℂ) (h : Complex.abs z = 1) :
  0 ≤ Complex.abs (z - (1 - Complex.I * Real.sqrt 3)) ∧
  Complex.abs (z - (1 - Complex.I * Real.sqrt 3)) ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_complex_distance_range_l1708_170871


namespace NUMINAMATH_CALUDE_slower_train_speed_l1708_170855

/-- Prove that given two trains moving in the same direction, with the faster train
    traveling at 50 km/hr, taking 15 seconds to pass a man in the slower train,
    and having a length of 75 meters, the speed of the slower train is 32 km/hr. -/
theorem slower_train_speed
  (faster_train_speed : ℝ)
  (passing_time : ℝ)
  (faster_train_length : ℝ)
  (h1 : faster_train_speed = 50)
  (h2 : passing_time = 15)
  (h3 : faster_train_length = 75) :
  ∃ (slower_train_speed : ℝ),
    slower_train_speed = 32 ∧
    (faster_train_speed - slower_train_speed) * 1000 / 3600 = faster_train_length / passing_time :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_l1708_170855


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_of_3_12_l1708_170833

theorem largest_consecutive_sum_of_3_12 :
  (∃ (k : ℕ), k > 486 ∧ 
    (∃ (n : ℕ), 3^12 = (Finset.range k).sum (λ i => n + i + 1))) →
  False :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_of_3_12_l1708_170833


namespace NUMINAMATH_CALUDE_number_equal_nine_l1708_170818

theorem number_equal_nine : ∃ x : ℝ, x^6 = 3^12 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_equal_nine_l1708_170818


namespace NUMINAMATH_CALUDE_sin_3phi_from_exponential_l1708_170825

theorem sin_3phi_from_exponential (φ : ℝ) :
  Complex.exp (Complex.I * φ) = (1 + Complex.I * Real.sqrt 8) / 3 →
  Real.sin (3 * φ) = -5 * Real.sqrt 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_3phi_from_exponential_l1708_170825


namespace NUMINAMATH_CALUDE_juggling_balls_average_l1708_170880

/-- Represents a juggling sequence -/
def JugglingSequence (n : ℕ) := Fin n → ℕ

/-- The number of balls in a juggling sequence -/
def numberOfBalls (n : ℕ) (j : JugglingSequence n) : ℚ :=
  (Finset.sum Finset.univ (fun i => j i)) / n

theorem juggling_balls_average (n : ℕ) (j : JugglingSequence n) :
  numberOfBalls n j = (Finset.sum Finset.univ (fun i => j i)) / n :=
by sorry

end NUMINAMATH_CALUDE_juggling_balls_average_l1708_170880


namespace NUMINAMATH_CALUDE_sin_special_angle_l1708_170889

/-- Given a function f(x) = sin(x/2 + π/4), prove that f(π/2) = 1 -/
theorem sin_special_angle (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (x / 2 + π / 4)) :
  f (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_special_angle_l1708_170889


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1708_170861

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x + 1) < 1) ↔ (x ∈ Set.Iio (-1) ∪ Set.Ioi 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1708_170861


namespace NUMINAMATH_CALUDE_divisibility_properties_l1708_170874

theorem divisibility_properties (n : ℕ) :
  (∃ k : ℤ, 2^n - 1 = 7 * k) ↔ (∃ m : ℕ, n = 3 * m) ∧
  ¬(∃ k : ℤ, 2^n + 1 = 7 * k) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l1708_170874


namespace NUMINAMATH_CALUDE_book_pages_count_l1708_170891

/-- The number of pages Cora read on Monday -/
def monday_pages : ℕ := 23

/-- The number of pages Cora read on Tuesday -/
def tuesday_pages : ℕ := 38

/-- The number of pages Cora read on Wednesday -/
def wednesday_pages : ℕ := 61

/-- The number of pages Cora will read on Thursday -/
def thursday_pages : ℕ := 12

/-- The number of pages Cora will read on Friday -/
def friday_pages : ℕ := 2 * thursday_pages

/-- The total number of pages in the book -/
def total_pages : ℕ := monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages

theorem book_pages_count : total_pages = 158 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l1708_170891


namespace NUMINAMATH_CALUDE_train_length_calculation_l1708_170802

theorem train_length_calculation (crossing_time : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) :
  crossing_time = 25.997920166386688 →
  bridge_length = 160 →
  train_speed_kmph = 36 →
  let train_speed_mps := train_speed_kmph * (5/18)
  let total_distance := train_speed_mps * crossing_time
  let train_length := total_distance - bridge_length
  train_length = 99.97920166386688 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1708_170802


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l1708_170840

theorem profit_percentage_previous_year 
  (revenue_prev : ℝ) 
  (profit_prev : ℝ) 
  (revenue_1999 : ℝ) 
  (profit_1999 : ℝ) 
  (h1 : revenue_1999 = 0.7 * revenue_prev) 
  (h2 : profit_1999 = 0.15 * revenue_1999) 
  (h3 : profit_1999 = 1.0499999999999999 * profit_prev) : 
  profit_prev / revenue_prev = 0.1 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l1708_170840


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1708_170801

def sequence_v (n : ℕ) : ℚ :=
  sorry

theorem sum_of_coefficients :
  (∃ a b c : ℚ, ∀ n : ℕ, sequence_v n = a * n^2 + b * n + c) →
  (sequence_v 1 = 7) →
  (∀ n : ℕ, sequence_v (n + 1) - sequence_v n = 5 + 6 * (n - 1)) →
  (∃ a b c : ℚ, (∀ n : ℕ, sequence_v n = a * n^2 + b * n + c) ∧ a + b + c = 7) :=
by sorry


end NUMINAMATH_CALUDE_sum_of_coefficients_l1708_170801


namespace NUMINAMATH_CALUDE_first_month_bill_is_50_l1708_170851

/-- Represents Elvin's monthly telephone bill --/
structure PhoneBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- The total bill is the sum of call charge and internet charge --/
def PhoneBill.total (bill : PhoneBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem first_month_bill_is_50 
  (firstMonth secondMonth : PhoneBill)
  (h1 : firstMonth.total = 50)
  (h2 : secondMonth.total = 76)
  (h3 : secondMonth.callCharge = 2 * firstMonth.callCharge)
  (h4 : firstMonth.internetCharge = secondMonth.internetCharge) :
  firstMonth.total = 50 := by
  sorry

#check first_month_bill_is_50

end NUMINAMATH_CALUDE_first_month_bill_is_50_l1708_170851


namespace NUMINAMATH_CALUDE_vector_sum_max_min_l1708_170873

/-- Given plane vectors a, b, and c satisfying certain conditions, 
    prove that the sum of the maximum and minimum values of |c| is √7 -/
theorem vector_sum_max_min (a b c : ℝ × ℝ) : 
  (‖a‖ = 1) → 
  (‖b‖ = 1) → 
  (a • (a - 2 • b) = 0) → 
  ((c - 2 • a) • (c - b) = 0) →
  (Real.sqrt ((max (‖c‖) (‖c‖)) ^ 2 + (min (‖c‖) (‖c‖)) ^ 2) = Real.sqrt 7) := by
  sorry

#check vector_sum_max_min

end NUMINAMATH_CALUDE_vector_sum_max_min_l1708_170873


namespace NUMINAMATH_CALUDE_smallest_advantageous_discount_l1708_170879

theorem smallest_advantageous_discount : ∃ (n : ℕ), n = 29 ∧ 
  (∀ (x : ℝ), x > 0 → 
    (1 - n / 100) * x < (1 - 0.12) * (1 - 0.18) * x ∧
    (1 - n / 100) * x < (1 - 0.08) * (1 - 0.08) * (1 - 0.08) * x ∧
    (1 - n / 100) * x < (1 - 0.20) * (1 - 0.10) * x) ∧
  (∀ (m : ℕ), m < n → 
    ∃ (x : ℝ), x > 0 ∧
      ((1 - m / 100) * x ≥ (1 - 0.12) * (1 - 0.18) * x ∨
       (1 - m / 100) * x ≥ (1 - 0.08) * (1 - 0.08) * (1 - 0.08) * x ∨
       (1 - m / 100) * x ≥ (1 - 0.20) * (1 - 0.10) * x)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_advantageous_discount_l1708_170879


namespace NUMINAMATH_CALUDE_exactly_one_absent_probability_l1708_170813

theorem exactly_one_absent_probability (p_absent : ℝ) (h1 : p_absent = 1 / 20) :
  let p_present := 1 - p_absent
  2 * p_absent * p_present = 19 / 200 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_absent_probability_l1708_170813


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1708_170888

theorem sqrt_product_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (10 * p) * Real.sqrt (5 * p^2) * Real.sqrt (6 * p^4) = 10 * p^3 * Real.sqrt (3 * p) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1708_170888


namespace NUMINAMATH_CALUDE_natalia_comics_count_l1708_170881

/-- The number of novels Natalia has -/
def novels : ℕ := 145

/-- The number of documentaries Natalia has -/
def documentaries : ℕ := 419

/-- The number of albums Natalia has -/
def albums : ℕ := 209

/-- The number of items each crate can hold -/
def items_per_crate : ℕ := 9

/-- The number of crates Natalia will use -/
def num_crates : ℕ := 116

/-- The number of comics Natalia has -/
def comics : ℕ := 271

theorem natalia_comics_count : 
  novels + documentaries + albums + comics = num_crates * items_per_crate := by
  sorry

end NUMINAMATH_CALUDE_natalia_comics_count_l1708_170881


namespace NUMINAMATH_CALUDE_drawer_is_translation_l1708_170804

-- Define the possible transformations
inductive Transformation
  | DrawerMovement
  | MagnifyingGlassEffect
  | ClockHandMovement
  | MirrorReflection

-- Define the properties of a translation
def isTranslation (t : Transformation) : Prop :=
  match t with
  | Transformation.DrawerMovement => true
  | _ => false

-- Theorem statement
theorem drawer_is_translation :
  ∀ t : Transformation, isTranslation t ↔ t = Transformation.DrawerMovement :=
by sorry

end NUMINAMATH_CALUDE_drawer_is_translation_l1708_170804


namespace NUMINAMATH_CALUDE_rearrangements_without_substring_l1708_170872

def word : String := "HMMTHMMT"

def total_permutations : ℕ := 420

def permutations_with_substring : ℕ := 60

theorem rearrangements_without_substring :
  (total_permutations - permutations_with_substring + 1 : ℕ) = 361 := by sorry

end NUMINAMATH_CALUDE_rearrangements_without_substring_l1708_170872


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l1708_170832

theorem max_digits_product_5_4 : 
  ∃ (a b : ℕ), 
    10000 ≤ a ∧ a < 100000 ∧
    1000 ≤ b ∧ b < 10000 ∧
    (∀ (x y : ℕ), 
      10000 ≤ x ∧ x < 100000 ∧ 
      1000 ≤ y ∧ y < 10000 → 
      x * y < 1000000000) ∧
    999999999 < a * b :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l1708_170832
