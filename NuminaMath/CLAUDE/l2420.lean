import Mathlib

namespace NUMINAMATH_CALUDE_percent_asian_in_west_1990_l2420_242008

/-- Represents the population of Asians in millions for each region in the U.S. in 1990 -/
structure AsianPopulation where
  northeast : ℕ
  midwest : ℕ
  south : ℕ
  west : ℕ

/-- Calculates the percentage of Asians living in the West to the nearest percent -/
def percentInWest (pop : AsianPopulation) : ℕ :=
  let total := pop.northeast + pop.midwest + pop.south + pop.west
  let westPercentage := (pop.west * 100) / total
  -- Round to nearest percent
  if westPercentage % 10 ≥ 5 then
    (westPercentage / 10 + 1) * 10
  else
    (westPercentage / 10) * 10

/-- The given population data for Asians in 1990 -/
def population1990 : AsianPopulation :=
  { northeast := 2
  , midwest := 2
  , south := 2
  , west := 6 }

/-- Theorem stating that the percentage of Asians living in the West in 1990 is 50% -/
theorem percent_asian_in_west_1990 : percentInWest population1990 = 50 := by
  sorry


end NUMINAMATH_CALUDE_percent_asian_in_west_1990_l2420_242008


namespace NUMINAMATH_CALUDE_eggs_per_hen_l2420_242013

/-- Given 303.0 eggs collected from 28.0 hens, prove that the number of eggs
    laid by each hen, when rounded to the nearest whole number, is 11. -/
theorem eggs_per_hen (total_eggs : ℝ) (num_hens : ℝ) 
    (h1 : total_eggs = 303.0) (h2 : num_hens = 28.0) :
  round (total_eggs / num_hens) = 11 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_hen_l2420_242013


namespace NUMINAMATH_CALUDE_factorization_equality_l2420_242030

theorem factorization_equality (a b : ℝ) : a^2 - 4*a*b + 4*b^2 = (a - 2*b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2420_242030


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2420_242069

-- Define the function f(x) = -x^2 + 2x + 1
def f (x : ℝ) : ℝ := -x^2 + 2*x + 1

-- Define the inequality
def has_solution (m : ℝ) : Prop :=
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ x^2 - 2*x - 1 + m ≤ 0

-- Theorem statement
theorem inequality_solution_range :
  ∀ m : ℝ, has_solution m ↔ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2420_242069


namespace NUMINAMATH_CALUDE_consecutive_digits_pattern_l2420_242098

def consecutive_digits (n : Nat) : Nat :=
  if n = 0 then 0 else
  let rec aux (k : Nat) (acc : Nat) : Nat :=
    if k = 0 then acc else aux (k - 1) (acc * 10 + k)
  aux n 0

def reverse_consecutive_digits (n : Nat) : Nat :=
  if n = 0 then 0 else
  let rec aux (k : Nat) (acc : Nat) : Nat :=
    if k = 0 then acc else aux (k - 1) (acc * 10 + (10 - k))
  aux n 0

theorem consecutive_digits_pattern (n : Nat) (h : n > 0 ∧ n ≤ 9) :
  consecutive_digits n * 8 + n = reverse_consecutive_digits n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_digits_pattern_l2420_242098


namespace NUMINAMATH_CALUDE_circle_center_polar_coordinates_l2420_242060

theorem circle_center_polar_coordinates :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 1}
  let center := (1, 1)
  let r := Real.sqrt 2
  let θ := Real.pi / 4
  (center ∈ circle) ∧
  (r * Real.cos θ = center.1 - 0) ∧
  (r * Real.sin θ = center.2 - 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_polar_coordinates_l2420_242060


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subsequence_l2420_242001

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (x y z : ℝ) : Prop :=
  y / x = z / y

theorem arithmetic_sequence_with_geometric_subsequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 1)
  (h_geom : is_geometric_sequence (a 1) (a 3) (a 9)) :
  (∀ n : ℕ, a n = 1) ∨ (∀ n : ℕ, a n = n) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subsequence_l2420_242001


namespace NUMINAMATH_CALUDE_prob_one_defective_in_two_l2420_242074

/-- Given a set of 4 items with 3 genuine and 1 defective, the probability
of selecting exactly one defective item when randomly choosing 2 items is 1/2. -/
theorem prob_one_defective_in_two (n : ℕ) (k : ℕ) (d : ℕ) :
  n = 4 →
  k = 2 →
  d = 1 →
  (n.choose k) = 6 →
  (d * (n - d).choose (k - 1)) = 3 →
  (d * (n - d).choose (k - 1)) / (n.choose k) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_prob_one_defective_in_two_l2420_242074


namespace NUMINAMATH_CALUDE_sarees_with_six_shirts_l2420_242054

/-- The price of a saree in dollars -/
def saree_price : ℕ := sorry

/-- The price of a shirt in dollars -/
def shirt_price : ℕ := sorry

/-- The number of sarees bought with 6 shirts -/
def num_sarees : ℕ := sorry

theorem sarees_with_six_shirts :
  (2 * saree_price + 4 * shirt_price = 1600) →
  (12 * shirt_price = 2400) →
  (num_sarees * saree_price + 6 * shirt_price = 1600) →
  num_sarees = 1 := by
  sorry

end NUMINAMATH_CALUDE_sarees_with_six_shirts_l2420_242054


namespace NUMINAMATH_CALUDE_rhombus_area_l2420_242004

/-- The area of a rhombus with side length 3 cm and an interior angle of 45 degrees is 9 square centimeters. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 3) (h2 : θ = π/4) :
  s * s * Real.sin θ = 9 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2420_242004


namespace NUMINAMATH_CALUDE_michelle_crayon_boxes_l2420_242018

/-- Given Michelle has 35 crayons in total and each box holds 5 crayons, 
    prove that the number of boxes Michelle has is 7. -/
theorem michelle_crayon_boxes 
  (total_crayons : ℕ) 
  (crayons_per_box : ℕ) 
  (h1 : total_crayons = 35)
  (h2 : crayons_per_box = 5) : 
  total_crayons / crayons_per_box = 7 := by
  sorry

#check michelle_crayon_boxes

end NUMINAMATH_CALUDE_michelle_crayon_boxes_l2420_242018


namespace NUMINAMATH_CALUDE_specific_plot_fencing_cost_l2420_242005

/-- Represents a rectangular plot with given dimensions and fencing cost -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencingCostPerMeter : ℝ

/-- Calculates the total cost of fencing a rectangular plot -/
def totalFencingCost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencingCostPerMeter

/-- Theorem stating the total fencing cost for a specific rectangular plot -/
theorem specific_plot_fencing_cost :
  let plot : RectangularPlot := {
    length := 65,
    breadth := 35,
    fencingCostPerMeter := 26.5
  }
  totalFencingCost plot = 5300 := by sorry

end NUMINAMATH_CALUDE_specific_plot_fencing_cost_l2420_242005


namespace NUMINAMATH_CALUDE_problem_solution_l2420_242049

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 3 = y ^ 2) 
  (h3 : x / 6 = 3 * y) : 
  x = 108 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2420_242049


namespace NUMINAMATH_CALUDE_count_integer_pairs_l2420_242017

theorem count_integer_pairs : 
  let w_count := (Finset.range 450).filter (fun w => w % 23 = 5) |>.card
  let n_count := (Finset.range 450).filter (fun n => n % 17 = 7) |>.card
  w_count * n_count = 540 := by
sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l2420_242017


namespace NUMINAMATH_CALUDE_march_total_distance_l2420_242046

/-- Represents Emberly's walking distance for a single day -/
structure DailyWalk where
  day : Nat
  distance : Float

/-- Emberly's walking pattern for March -/
def marchWalks : List DailyWalk := [
  ⟨1, 4⟩, ⟨2, 3⟩, ⟨3, 4⟩, ⟨4, 3⟩, ⟨5, 4⟩, ⟨6, 0⟩, ⟨7, 0⟩,
  ⟨8, 5⟩, ⟨9, 2.5⟩, ⟨10, 5⟩, ⟨11, 5⟩, ⟨12, 2.5⟩, ⟨13, 2.5⟩, ⟨14, 0⟩,
  ⟨15, 6⟩, ⟨16, 6⟩, ⟨17, 0⟩, ⟨18, 0⟩, ⟨19, 0⟩, ⟨20, 4⟩, ⟨21, 4⟩, ⟨22, 3.5⟩,
  ⟨23, 4.5⟩, ⟨24, 0⟩, ⟨25, 4.5⟩, ⟨26, 0⟩, ⟨27, 4.5⟩, ⟨28, 0⟩, ⟨29, 4.5⟩, ⟨30, 0⟩, ⟨31, 0⟩
]

/-- Calculate the total distance walked in March -/
def totalDistance (walks : List DailyWalk) : Float :=
  walks.foldl (fun acc walk => acc + walk.distance) 0

/-- Theorem: The total distance Emberly walked in March is 82 miles -/
theorem march_total_distance : totalDistance marchWalks = 82 := by
  sorry


end NUMINAMATH_CALUDE_march_total_distance_l2420_242046


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2420_242040

theorem simplify_square_roots : 
  Real.sqrt 726 / Real.sqrt 242 + Real.sqrt 484 / Real.sqrt 121 = Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2420_242040


namespace NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l2420_242082

/-- The volume of a cylinder formed by rotating a square around one of its sides. -/
theorem volume_cylinder_from_square_rotation (side_length : Real) (volume : Real) : 
  side_length = 2 → volume = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l2420_242082


namespace NUMINAMATH_CALUDE_smallest_number_l2420_242056

/-- Converts a number from base 6 to decimal -/
def base6ToDecimal (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 4 to decimal -/
def base4ToDecimal (n : Nat) : Nat :=
  (n / 1000) * 64 + ((n / 100) % 10) * 16 + ((n / 10) % 10) * 4 + (n % 10)

/-- Converts a number from base 2 to decimal -/
def base2ToDecimal (n : Nat) : Nat :=
  (n / 100000) * 32 + ((n / 10000) % 10) * 16 + ((n / 1000) % 10) * 8 +
  ((n / 100) % 10) * 4 + ((n / 10) % 10) * 2 + (n % 10)

theorem smallest_number (n1 n2 n3 : Nat) 
  (h1 : n1 = 210)
  (h2 : n2 = 1000)
  (h3 : n3 = 111111) :
  base2ToDecimal n3 < base6ToDecimal n1 ∧ base2ToDecimal n3 < base4ToDecimal n2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2420_242056


namespace NUMINAMATH_CALUDE_total_interest_calculation_l2420_242027

/-- Calculate simple interest -/
def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

theorem total_interest_calculation (rate : ℕ) : 
  rate = 10 → 
  simple_interest 5000 rate 2 + simple_interest 3000 rate 4 = 2200 := by
  sorry

#check total_interest_calculation

end NUMINAMATH_CALUDE_total_interest_calculation_l2420_242027


namespace NUMINAMATH_CALUDE_power_of_two_equality_l2420_242032

theorem power_of_two_equality : ∃ x : ℕ, 8^12 + 8^12 + 8^12 = 2^x ∧ x = 38 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l2420_242032


namespace NUMINAMATH_CALUDE_specific_pentagon_area_l2420_242037

/-- A pentagon with specific side lengths that can be divided into a right triangle and a trapezoid -/
structure SpecificPentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  triangle_base : ℝ
  triangle_height : ℝ
  trapezoid_base1 : ℝ
  trapezoid_base2 : ℝ
  trapezoid_height : ℝ
  side1_eq : side1 = 15
  side2_eq : side2 = 20
  side3_eq : side3 = 27
  side4_eq : side4 = 24
  side5_eq : side5 = 20
  triangle_base_eq : triangle_base = 15
  triangle_height_eq : triangle_height = 20
  trapezoid_base1_eq : trapezoid_base1 = 20
  trapezoid_base2_eq : trapezoid_base2 = 27
  trapezoid_height_eq : trapezoid_height = 24

/-- The area of the specific pentagon is 714 square units -/
theorem specific_pentagon_area (p : SpecificPentagon) : 
  (1/2 * p.triangle_base * p.triangle_height) + 
  (1/2 * (p.trapezoid_base1 + p.trapezoid_base2) * p.trapezoid_height) = 714 := by
  sorry

end NUMINAMATH_CALUDE_specific_pentagon_area_l2420_242037


namespace NUMINAMATH_CALUDE_first_expression_value_l2420_242044

theorem first_expression_value (E a : ℝ) (h1 : (E + (3 * a - 8)) / 2 = 89) (h2 : a = 34) : E = 84 := by
  sorry

end NUMINAMATH_CALUDE_first_expression_value_l2420_242044


namespace NUMINAMATH_CALUDE_geometric_progression_sum_ratio_l2420_242096

theorem geometric_progression_sum_ratio (a : ℝ) (n : ℕ) : 
  let r : ℝ := 3
  let S_n := a * (1 - r^n) / (1 - r)
  let S_3 := a * (1 - r^3) / (1 - r)
  S_n / S_3 = 28 → n = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_ratio_l2420_242096


namespace NUMINAMATH_CALUDE_range_of_H_l2420_242062

-- Define the function H
def H (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H :
  Set.range H = Set.Iic 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_H_l2420_242062


namespace NUMINAMATH_CALUDE_melted_ice_cream_height_l2420_242078

/-- The height of a cylindrical region formed by a melted spherical ice cream scoop -/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 3)
    (h_cylinder : r_cylinder = 9) : 
    (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * (4 / 9) := by
  sorry

#check melted_ice_cream_height

end NUMINAMATH_CALUDE_melted_ice_cream_height_l2420_242078


namespace NUMINAMATH_CALUDE_sand_container_problem_l2420_242025

theorem sand_container_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (a * Real.exp (-8 * b) = a / 2) →
  (∃ t : ℝ, a * Real.exp (-b * t) = a / 8 ∧ t > 0) →
  (∃ t : ℝ, a * Real.exp (-b * t) = a / 8 ∧ t = 24) :=
by sorry

end NUMINAMATH_CALUDE_sand_container_problem_l2420_242025


namespace NUMINAMATH_CALUDE_michael_twice_jacob_age_l2420_242091

theorem michael_twice_jacob_age (jacob_current_age : ℕ) (michael_current_age : ℕ) : 
  jacob_current_age = 11 - 4 →
  michael_current_age = jacob_current_age + 12 →
  ∃ x : ℕ, michael_current_age + x = 2 * (jacob_current_age + x) ∧ x = 5 :=
by sorry

end NUMINAMATH_CALUDE_michael_twice_jacob_age_l2420_242091


namespace NUMINAMATH_CALUDE_decreasing_function_characterization_l2420_242059

open Set

/-- A function f is decreasing on an open interval (a, b) if for all x₁, x₂ in (a, b),
    x₁ < x₂ implies f(x₁) > f(x₂) -/
def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ Ioo a b → x₂ ∈ Ioo a b → x₁ < x₂ → f x₁ > f x₂

theorem decreasing_function_characterization
  (f : ℝ → ℝ) (a b : ℝ) (h : a < b)
  (h_domain : ∀ x, x ∈ Ioo a b → f x ∈ range f)
  (h_inequality : ∀ x₁ x₂, x₁ ∈ Ioo a b → x₂ ∈ Ioo a b →
    (x₁ - x₂) * (f x₁ - f x₂) < 0) :
  DecreasingOn f a b := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_characterization_l2420_242059


namespace NUMINAMATH_CALUDE_circle_symmetry_l2420_242035

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 - x + 2*y = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x + 2)^2 + (y - 3/2)^2 = 5/4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y x' y' : ℝ),
  original_circle x y →
  symmetry_line ((x + x') / 2) ((y + y') / 2) →
  symmetric_circle x' y' :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2420_242035


namespace NUMINAMATH_CALUDE_total_tomatoes_l2420_242011

def number_of_rows : ℕ := 30
def plants_per_row : ℕ := 10
def tomatoes_per_plant : ℕ := 20

theorem total_tomatoes : 
  number_of_rows * plants_per_row * tomatoes_per_plant = 6000 := by
  sorry

end NUMINAMATH_CALUDE_total_tomatoes_l2420_242011


namespace NUMINAMATH_CALUDE_power_of_two_l2420_242081

theorem power_of_two (n : ℕ) : 32 * (1/2)^2 = 2^n → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_l2420_242081


namespace NUMINAMATH_CALUDE_hexagon_sequence_theorem_l2420_242087

/-- Represents the number of dots in the nth hexagon of the sequence -/
def hexagon_dots (n : ℕ) : ℕ :=
  if n = 0 then 0
  else 1 + 3 * n * (n - 1)

/-- The theorem stating the number of dots in the first four hexagons -/
theorem hexagon_sequence_theorem :
  hexagon_dots 1 = 1 ∧
  hexagon_dots 2 = 7 ∧
  hexagon_dots 3 = 19 ∧
  hexagon_dots 4 = 37 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_sequence_theorem_l2420_242087


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2420_242003

/-- Given a rectangle with area 540 square centimeters, if its length is decreased by 20%
    and its width is increased by 15%, then its new area is 496.8 square centimeters. -/
theorem rectangle_area_change (l w : ℝ) (h1 : l * w = 540) : 
  (0.8 * l) * (1.15 * w) = 496.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2420_242003


namespace NUMINAMATH_CALUDE_acid_solution_volume_l2420_242019

/-- Given a solution with 1.6 litres of pure acid and a concentration of 20%,
    prove that the total volume of the solution is 8 litres. -/
theorem acid_solution_volume (pure_acid : ℝ) (concentration : ℝ) (total_volume : ℝ) 
    (h1 : pure_acid = 1.6)
    (h2 : concentration = 0.2)
    (h3 : pure_acid = concentration * total_volume) : 
  total_volume = 8 := by
sorry

end NUMINAMATH_CALUDE_acid_solution_volume_l2420_242019


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l2420_242036

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) : 
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 1/5 →
  germination_rate2 = 7/20 →
  (((seeds_plot1 : ℚ) * germination_rate1 + (seeds_plot2 : ℚ) * germination_rate2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) = 13/50 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l2420_242036


namespace NUMINAMATH_CALUDE_K_factorization_l2420_242070

theorem K_factorization (x y z : ℝ) : 
  (x + 2*y + 3*z) * (2*x - y - z) * (y + 2*z + 3*x) +
  (y + 2*z + 3*x) * (2*y - z - x) * (z + 2*x + 3*y) +
  (z + 2*x + 3*y) * (2*z - x - y) * (x + 2*y + 3*z) =
  (y + z - 2*x) * (z + x - 2*y) * (x + y - 2*z) := by
sorry

end NUMINAMATH_CALUDE_K_factorization_l2420_242070


namespace NUMINAMATH_CALUDE_bird_percentage_l2420_242068

/-- The percentage of birds that are not hawks, paddyfield-warblers, kingfishers, or blackbirds in Goshawk-Eurasian Nature Reserve -/
theorem bird_percentage (total : ℝ) (hawks paddyfield_warblers kingfishers blackbirds : ℝ)
  (h1 : hawks = 0.3 * total)
  (h2 : paddyfield_warblers = 0.4 * (total - hawks))
  (h3 : kingfishers = 0.25 * paddyfield_warblers)
  (h4 : blackbirds = 0.15 * (hawks + paddyfield_warblers))
  (h5 : total > 0) :
  (total - (hawks + paddyfield_warblers + kingfishers + blackbirds)) / total = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_bird_percentage_l2420_242068


namespace NUMINAMATH_CALUDE_complete_square_sum_l2420_242065

/-- 
Given a quadratic equation x^2 - 6x + 5 = 0, when rewritten in the form (x + d)^2 = e 
where d and e are integers, prove that d + e = 1
-/
theorem complete_square_sum (d e : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + d)^2 = e) → d + e = 1 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l2420_242065


namespace NUMINAMATH_CALUDE_triangle_side_length_l2420_242024

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  b + c = 2 * Real.sqrt 3 →
  A = π / 3 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  a = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2420_242024


namespace NUMINAMATH_CALUDE_grisha_has_winning_strategy_l2420_242086

/-- Represents the state of the game board -/
def GameBoard := List Nat

/-- Represents a player's move -/
inductive Move
| Square : Nat → Move  -- Square the number at a given index
| Increment : Nat → Move  -- Increment the number at a given index

/-- Represents a player -/
inductive Player
| Grisha
| Gleb

/-- Applies a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  match move with
  | Move.Square i => sorry
  | Move.Increment i => sorry

/-- Checks if any number on the board is divisible by 2023 -/
def hasDivisibleBy2023 (board : GameBoard) : Bool :=
  sorry

/-- Represents a game strategy -/
def Strategy := GameBoard → Move

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (player : Player) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem stating that Grisha has a winning strategy -/
theorem grisha_has_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy Player.Grisha strategy :=
sorry

end NUMINAMATH_CALUDE_grisha_has_winning_strategy_l2420_242086


namespace NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l2420_242000

/-- The ratio of the area of an isosceles right triangle to the area of a pentagon formed by the triangle and a rectangle -/
theorem triangle_to_pentagon_area_ratio :
  let triangle_leg : ℝ := 2
  let triangle_hypotenuse : ℝ := triangle_leg * Real.sqrt 2
  let rectangle_width : ℝ := triangle_hypotenuse
  let rectangle_height : ℝ := 2 * triangle_leg
  let triangle_area : ℝ := (1 / 2) * triangle_leg ^ 2
  let rectangle_area : ℝ := rectangle_width * rectangle_height
  let pentagon_area : ℝ := triangle_area + rectangle_area
  triangle_area / pentagon_area = 2 / (2 + 8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l2420_242000


namespace NUMINAMATH_CALUDE_problem_solution_l2420_242080

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x + 1) / (2^(x+1) - a)

theorem problem_solution :
  ∃ (a : ℝ),
    (∀ (x : ℝ), f a x = (2^x + 1) / (2^(x+1) - a)) ∧
    (a = 2) ∧
    (∀ (x y : ℝ), 0 < x → 0 < y → x < y → f a x > f a y) ∧
    (∀ (k : ℝ), (∃ (x : ℝ), 0 < x ∧ x ≤ 1 ∧ k * f a x = 2) → 0 < k ∧ k ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2420_242080


namespace NUMINAMATH_CALUDE_sales_equation_solution_l2420_242053

/-- Given the sales equation and conditions, prove the value of p. -/
theorem sales_equation_solution (f w p : ℂ) (h1 : f * p - w = 15000) 
  (h2 : f = 10) (h3 : w = 10 + 250 * Complex.I) : p = 1501 + 25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_sales_equation_solution_l2420_242053


namespace NUMINAMATH_CALUDE_sum_of_coefficients_after_shift_l2420_242048

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_quadratic (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a,
    b := 2 * f.a * shift + f.b,
    c := f.a * shift^2 + f.b * shift + f.c }

/-- The original quadratic function y = 3x^2 - 2x + 6 -/
def original_function : QuadraticFunction :=
  { a := 3, b := -2, c := 6 }

/-- The amount of left shift -/
def left_shift : ℝ := 5

theorem sum_of_coefficients_after_shift :
  let shifted := shift_quadratic original_function left_shift
  shifted.a + shifted.b + shifted.c = 102 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_after_shift_l2420_242048


namespace NUMINAMATH_CALUDE_quadratic_function_min_value_l2420_242066

theorem quadratic_function_min_value 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^2 + x + a) 
  (h2 : ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, f y ≤ f x) 
  (h3 : ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 2) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, f x ≤ f y ∧ f x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_min_value_l2420_242066


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2420_242023

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (1, 2) (x, 4) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2420_242023


namespace NUMINAMATH_CALUDE_technician_round_trip_l2420_242009

/-- Represents the percentage of a round-trip journey completed -/
def round_trip_percentage (outbound_percent : ℝ) (return_percent : ℝ) : ℝ :=
  (outbound_percent + return_percent * outbound_percent) * 50

/-- Theorem stating that completing the outbound journey and 10% of the return journey
    results in 55% of the round-trip being completed -/
theorem technician_round_trip :
  round_trip_percentage 100 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_technician_round_trip_l2420_242009


namespace NUMINAMATH_CALUDE_return_speed_calculation_l2420_242052

/-- Proves that given a round trip with specified conditions, the return speed is 160 km/h -/
theorem return_speed_calculation (total_time : ℝ) (outbound_time_minutes : ℝ) (outbound_speed : ℝ) :
  total_time = 5 →
  outbound_time_minutes = 192 →
  outbound_speed = 90 →
  let outbound_time_hours : ℝ := outbound_time_minutes / 60
  let distance : ℝ := outbound_speed * outbound_time_hours
  let return_time : ℝ := total_time - outbound_time_hours
  let return_speed : ℝ := distance / return_time
  return_speed = 160 := by
  sorry

#check return_speed_calculation

end NUMINAMATH_CALUDE_return_speed_calculation_l2420_242052


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l2420_242063

theorem max_value_on_ellipse :
  ∀ x y : ℝ, (x^2 / 6 + y^2 / 4 = 1) →
  ∃ (max : ℝ), (∀ x' y' : ℝ, (x'^2 / 6 + y'^2 / 4 = 1) → x' + 2*y' ≤ max) ∧
  max = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l2420_242063


namespace NUMINAMATH_CALUDE_blue_shirt_percentage_l2420_242033

/-- Proves that the percentage of students wearing blue shirts is 44% -/
theorem blue_shirt_percentage
  (total_students : ℕ)
  (red_shirt_percentage : ℚ)
  (green_shirt_percentage : ℚ)
  (other_colors_count : ℕ)
  (h_total : total_students = 900)
  (h_red : red_shirt_percentage = 28/100)
  (h_green : green_shirt_percentage = 10/100)
  (h_other : other_colors_count = 162) :
  (total_students : ℚ) - (red_shirt_percentage + green_shirt_percentage + (other_colors_count : ℚ) / total_students) * total_students = 44/100 * total_students :=
by sorry

end NUMINAMATH_CALUDE_blue_shirt_percentage_l2420_242033


namespace NUMINAMATH_CALUDE_abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l2420_242012

theorem abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two :
  (∀ x : ℝ, x < -2 → |x| > 2) ∧
  (∃ x : ℝ, |x| > 2 ∧ ¬(x < -2)) :=
sorry

end NUMINAMATH_CALUDE_abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l2420_242012


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l2420_242039

/-- The intersection point of two lines with given angles of inclination -/
theorem intersection_point_of_lines (m n : ℝ → ℝ) (k₁ k₂ : ℝ) :
  (∀ x, m x = k₁ * x + 2) →
  (∀ x, n x = k₂ * x + Real.sqrt 3 + 1) →
  k₁ = Real.tan (π / 4) →
  k₂ = Real.tan (π / 3) →
  ∃ x y, m x = n x ∧ m x = y ∧ x = -1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l2420_242039


namespace NUMINAMATH_CALUDE_integer_pair_theorem_l2420_242055

/-- Given positive integers a and b where a > b, prove that a²b - ab² = 30 
    if and only if (a, b) is one of (5, 2), (5, 3), (6, 1), or (6, 5) -/
theorem integer_pair_theorem (a b : ℕ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) :
  a^2 * b - a * b^2 = 30 ↔ 
  ((a = 5 ∧ b = 2) ∨ (a = 5 ∧ b = 3) ∨ (a = 6 ∧ b = 1) ∨ (a = 6 ∧ b = 5)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_theorem_l2420_242055


namespace NUMINAMATH_CALUDE_min_value_theorem_l2420_242007

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min : ℝ), min = 2 + 2 * Real.sqrt 2 ∧ ∀ z, z = (2 / x) + (x / y) → z ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2420_242007


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2420_242083

theorem sum_of_squares_of_roots (r₁ r₂ r₃ r₄ : ℂ) : 
  (r₁^4 + 6*r₁^3 + 11*r₁^2 + 6*r₁ + 1 = 0) →
  (r₂^4 + 6*r₂^3 + 11*r₂^2 + 6*r₂ + 1 = 0) →
  (r₃^4 + 6*r₃^3 + 11*r₃^2 + 6*r₃ + 1 = 0) →
  (r₄^4 + 6*r₄^3 + 11*r₄^2 + 6*r₄ + 1 = 0) →
  r₁^2 + r₂^2 + r₃^2 + r₄^2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2420_242083


namespace NUMINAMATH_CALUDE_unique_expansion_terms_l2420_242058

def expansion_terms (N : ℕ) : ℕ := Nat.choose N 5

theorem unique_expansion_terms : 
  ∃! N : ℕ, N > 0 ∧ expansion_terms N = 231 :=
by sorry

end NUMINAMATH_CALUDE_unique_expansion_terms_l2420_242058


namespace NUMINAMATH_CALUDE_millet_exceeds_half_on_thursday_l2420_242057

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : ℕ
  millet : ℚ
  other : ℚ

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := state.millet / 2 + 2 / 5,
    other := 3 / 5 }

/-- Checks if millet proportion exceeds 50% -/
def milletExceedsHalf (state : FeederState) : Prop :=
  state.millet > (state.millet + state.other) / 2

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1, millet := 2 / 5, other := 3 / 5 }

theorem millet_exceeds_half_on_thursday :
  let thursday := nextDay (nextDay (nextDay initialState))
  milletExceedsHalf thursday ∧
  ∀ (prevDay : FeederState), prevDay.day < thursday.day →
    ¬ milletExceedsHalf prevDay := by
  sorry

end NUMINAMATH_CALUDE_millet_exceeds_half_on_thursday_l2420_242057


namespace NUMINAMATH_CALUDE_overlapping_area_is_64_l2420_242064

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side_length : ℝ)

/-- Represents the rotation of a sheet -/
inductive Rotation
  | NoRotation
  | Rotate45
  | Rotate90

/-- Represents the configuration of three sheets -/
structure SheetConfiguration :=
  (bottom : Sheet)
  (middle : Sheet)
  (top : Sheet)
  (middle_rotation : Rotation)
  (top_rotation : Rotation)

/-- Calculates the area of the overlapping polygon -/
def overlapping_area (config : SheetConfiguration) : ℝ :=
  sorry

/-- Theorem stating that the overlapping area is 64 for the given configuration -/
theorem overlapping_area_is_64 :
  ∀ (config : SheetConfiguration),
    config.bottom.side_length = 8 ∧
    config.middle.side_length = 8 ∧
    config.top.side_length = 8 ∧
    config.middle_rotation = Rotation.Rotate45 ∧
    config.top_rotation = Rotation.Rotate90 →
    overlapping_area config = 64 :=
  sorry

end NUMINAMATH_CALUDE_overlapping_area_is_64_l2420_242064


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2420_242038

theorem rectangular_prism_volume
  (side_area front_area bottom_area : ℝ)
  (h₁ : side_area = 12)
  (h₂ : front_area = 8)
  (h₃ : bottom_area = 6)
  : ∃ (length width height : ℝ),
    length * width = front_area ∧
    width * height = side_area ∧
    length * height = bottom_area ∧
    length * width * height = 24 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2420_242038


namespace NUMINAMATH_CALUDE_second_number_proof_l2420_242071

theorem second_number_proof (x y z : ℚ) 
  (sum_eq : x + y + z = 125)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 7 / 6) :
  y = 3500 / 73 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l2420_242071


namespace NUMINAMATH_CALUDE_train_speed_l2420_242020

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 45) :
  (train_length + bridge_length) / crossing_time = 13.33333333333333 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l2420_242020


namespace NUMINAMATH_CALUDE_first_group_size_l2420_242031

/-- Represents the work done by a group of workers -/
structure Work where
  workers : ℕ
  days : ℕ
  hectares : ℕ

/-- The work principle: workers * days is proportional to hectares for any two Work instances -/
axiom work_principle {w1 w2 : Work} : 
  w1.workers * w1.days * w2.hectares = w2.workers * w2.days * w1.hectares

/-- The first group's work -/
def first_group : Work := { workers := 0, days := 24, hectares := 80 }

/-- The second group's work -/
def second_group : Work := { workers := 36, days := 30, hectares := 400 }

/-- The theorem to prove -/
theorem first_group_size : first_group.workers = 9 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l2420_242031


namespace NUMINAMATH_CALUDE_min_sixth_graders_l2420_242072

theorem min_sixth_graders (x : ℕ) (hx : x > 0) : 
  let girls := x / 3
  let boys := x - girls
  let sixth_grade_girls := girls / 2
  let non_sixth_grade_boys := (boys * 5) / 7
  let sixth_grade_boys := boys - non_sixth_grade_boys
  let total_sixth_graders := sixth_grade_girls + sixth_grade_boys
  x % 3 = 0 ∧ girls % 2 = 0 ∧ boys % 7 = 0 →
  ∀ y : ℕ, y > 0 ∧ y < x ∧ 
    (let girls_y := y / 3
     let boys_y := y - girls_y
     let sixth_grade_girls_y := girls_y / 2
     let non_sixth_grade_boys_y := (boys_y * 5) / 7
     let sixth_grade_boys_y := boys_y - non_sixth_grade_boys_y
     let total_sixth_graders_y := sixth_grade_girls_y + sixth_grade_boys_y
     y % 3 = 0 ∧ girls_y % 2 = 0 ∧ boys_y % 7 = 0) →
    total_sixth_graders_y < total_sixth_graders →
  total_sixth_graders = 15 := by
sorry

end NUMINAMATH_CALUDE_min_sixth_graders_l2420_242072


namespace NUMINAMATH_CALUDE_geometric_mean_of_one_and_four_l2420_242002

theorem geometric_mean_of_one_and_four :
  ∀ x : ℝ, x^2 = 1 * 4 → x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_one_and_four_l2420_242002


namespace NUMINAMATH_CALUDE_triangle_area_l2420_242090

/-- Given a triangle with perimeter 48 cm and inradius 2.5 cm, its area is 60 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 48 → inradius = 2.5 → area = perimeter / 2 * inradius → area = 60 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2420_242090


namespace NUMINAMATH_CALUDE_number_ratio_l2420_242076

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 69) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l2420_242076


namespace NUMINAMATH_CALUDE_farmers_market_total_sales_l2420_242021

/-- Calculates the total sales from a farmers' market given specific conditions --/
theorem farmers_market_total_sales :
  let broccoli_sales : ℕ := 57
  let carrot_sales : ℕ := 2 * broccoli_sales
  let spinach_sales : ℕ := carrot_sales / 2 + 16
  let cauliflower_sales : ℕ := 136
  broccoli_sales + carrot_sales + spinach_sales + cauliflower_sales = 380 :=
by
  sorry


end NUMINAMATH_CALUDE_farmers_market_total_sales_l2420_242021


namespace NUMINAMATH_CALUDE_exponential_sum_rule_l2420_242093

theorem exponential_sum_rule (a : ℝ) (x₁ x₂ : ℝ) (ha : 0 < a) :
  a^(x₁ + x₂) = a^x₁ * a^x₂ := by
  sorry

end NUMINAMATH_CALUDE_exponential_sum_rule_l2420_242093


namespace NUMINAMATH_CALUDE_irrational_sqrt_6_l2420_242034

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

-- Theorem statement
theorem irrational_sqrt_6 :
  IsIrrational (Real.sqrt 6) ∧ 
  IsRational 3.14 ∧
  IsRational (-1/3) ∧
  IsRational (22/7) :=
sorry

end NUMINAMATH_CALUDE_irrational_sqrt_6_l2420_242034


namespace NUMINAMATH_CALUDE_root_transformation_l2420_242022

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 9 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 9 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 9 = 0) →
  ((3*r₁)^3 - 9*(3*r₁)^2 + 243 = 0) ∧
  ((3*r₂)^3 - 9*(3*r₂)^2 + 243 = 0) ∧
  ((3*r₃)^3 - 9*(3*r₃)^2 + 243 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l2420_242022


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_range_f_below_g_implies_a_range_l2420_242042

-- Define the function f
def f (a x : ℝ) : ℝ := x * |x - a| + 3 * x

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 1

-- Theorem 1
theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → -3 ≤ a ∧ a ≤ 3 :=
sorry

-- Theorem 2
theorem f_below_g_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a x < g x) → 3/2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_range_f_below_g_implies_a_range_l2420_242042


namespace NUMINAMATH_CALUDE_fraction_denominator_l2420_242028

theorem fraction_denominator (n : ℕ) (d : ℕ) (h1 : n = 325) (h2 : (n : ℚ) / d = 1 / 8) 
  (h3 : ∃ (seq : ℕ → ℕ), (∀ k, seq k < 10) ∧ 
    (∀ k, ((n : ℚ) / d - (n : ℚ) / d).floor + (seq k) / 10^(k+1) = ((n : ℚ) / d * 10^(k+1)).floor / 10^(k+1)) ∧ 
    seq 80 = 5) : 
  d = 8 := by sorry

end NUMINAMATH_CALUDE_fraction_denominator_l2420_242028


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2016th_term_l2420_242073

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_2016th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_incr : ∀ n : ℕ, a (n + 1) > a n) 
  (h_first : a 1 = 1) 
  (h_geom : (a 4)^2 = a 2 * a 8) :
  a 2016 = 2016 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2016th_term_l2420_242073


namespace NUMINAMATH_CALUDE_three_white_marbles_possible_l2420_242041

/-- Represents the state of the urn -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Represents the possible operations on the urn -/
inductive Operation
  | op1 | op2 | op3 | op4 | op5

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.op1 => UrnState.mk state.white (state.black - 2)
  | Operation.op2 => UrnState.mk (state.white - 1) (state.black - 2)
  | Operation.op3 => UrnState.mk state.white (state.black - 1)
  | Operation.op4 => UrnState.mk state.white (state.black - 1)
  | Operation.op5 => UrnState.mk (state.white - 3) (state.black + 2)

/-- Applies a sequence of operations to the urn state -/
def applyOperations (initial : UrnState) (ops : List Operation) : UrnState :=
  ops.foldl applyOperation initial

/-- The theorem to be proved -/
theorem three_white_marbles_possible :
  ∃ (ops : List Operation),
    let final := applyOperations (UrnState.mk 150 50) ops
    final.white = 3 ∧ final.black ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_three_white_marbles_possible_l2420_242041


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l2420_242084

/-- The length of the longest side of a triangle with vertices at (3,3), (8,9), and (9,3) is √61 units. -/
theorem longest_side_of_triangle : ∃ (a b c : ℝ × ℝ),
  a = (3, 3) ∧ b = (8, 9) ∧ c = (9, 3) ∧
  (max (dist a b) (max (dist b c) (dist c a)))^2 = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l2420_242084


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l2420_242067

theorem hot_dogs_remainder : 25197643 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l2420_242067


namespace NUMINAMATH_CALUDE_prob_not_all_same_l2420_242010

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability that not all dice show the same number when rolling 
    five fair 6-sided dice -/
theorem prob_not_all_same : 
  (1 - (numSides : ℚ) / (numSides ^ numDice)) = 1295 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_all_same_l2420_242010


namespace NUMINAMATH_CALUDE_simplify_exponents_l2420_242026

theorem simplify_exponents (t : ℝ) (h : t ≠ 0) : (t^5 * t^3) / t^2 = t^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponents_l2420_242026


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2420_242045

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 2 * I / (1 + I)
  Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2420_242045


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_d_value_l2420_242079

theorem polynomial_factor_implies_d_value (c d : ℤ) : 
  (∃ k : ℤ, (X^3 - 2*X^2 - X + 2) * (c*X + k) = c*X^4 + d*X^3 - 2*X^2 + 2) → 
  d = -1 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_d_value_l2420_242079


namespace NUMINAMATH_CALUDE_coupon1_best_for_given_prices_coupon1_not_best_for_lower_prices_l2420_242050

-- Define the discount functions for each coupon
def coupon1_discount (price : ℝ) : ℝ := 0.12 * price

def coupon2_discount (price : ℝ) : ℝ := 30

def coupon3_discount (price : ℝ) : ℝ := 0.15 * (price - 150)

def coupon4_discount (price : ℝ) : ℝ := 25 + 0.05 * (price - 25)

-- Define a function to check if Coupon 1 gives the best discount
def coupon1_is_best (price : ℝ) : Prop :=
  coupon1_discount price > coupon2_discount price ∧
  coupon1_discount price > coupon3_discount price ∧
  coupon1_discount price > coupon4_discount price

-- Theorem stating that Coupon 1 is best for $300, $350, and $400
theorem coupon1_best_for_given_prices :
  coupon1_is_best 300 ∧ coupon1_is_best 350 ∧ coupon1_is_best 400 :=
by sorry

-- Additional theorem to show Coupon 1 is not best for $200 and $250
theorem coupon1_not_best_for_lower_prices :
  ¬(coupon1_is_best 200) ∧ ¬(coupon1_is_best 250) :=
by sorry

end NUMINAMATH_CALUDE_coupon1_best_for_given_prices_coupon1_not_best_for_lower_prices_l2420_242050


namespace NUMINAMATH_CALUDE_plain_cookie_price_l2420_242006

theorem plain_cookie_price 
  (chocolate_chip_price : ℝ) 
  (total_boxes : ℝ) 
  (total_value : ℝ) 
  (plain_boxes : ℝ) 
  (h1 : chocolate_chip_price = 1.25)
  (h2 : total_boxes = 1585)
  (h3 : total_value = 1586.25)
  (h4 : plain_boxes = 793.125) : 
  (total_value - (total_boxes - plain_boxes) * chocolate_chip_price) / plain_boxes = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_plain_cookie_price_l2420_242006


namespace NUMINAMATH_CALUDE_train_problem_l2420_242085

/-- 
Given two trains departing simultaneously from points A and B towards each other,
this theorem proves the speeds of the trains and the distance between A and B.
-/
theorem train_problem (p q t : ℝ) (hp : p > 0) (hq : q > 0) (ht : t > 0) :
  ∃ (x y z : ℝ),
    x > 0 ∧ y > 0 ∧ z > 0 ∧  -- Speeds and distance are positive
    (p / y = (z - p) / x) ∧  -- Trains meet at distance p from B
    (t * y = q + z - p) ∧   -- Second train's position after t hours
    (t * (x + y) = 2 * z) ∧ -- Total distance traveled by both trains after t hours
    (x = (4 * p - 2 * q) / t) ∧ -- Speed of first train
    (y = 2 * p / t) ∧           -- Speed of second train
    (z = 3 * p - q)             -- Distance between A and B
  := by sorry

end NUMINAMATH_CALUDE_train_problem_l2420_242085


namespace NUMINAMATH_CALUDE_sandy_molly_age_difference_l2420_242089

theorem sandy_molly_age_difference :
  ∀ (sandy_age molly_age : ℕ),
    sandy_age = 70 →
    sandy_age * 9 = molly_age * 7 →
    molly_age - sandy_age = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_molly_age_difference_l2420_242089


namespace NUMINAMATH_CALUDE_three_painters_three_rooms_l2420_242088

/-- Represents the time taken for painters to complete rooms -/
def time_to_complete (painters : ℕ) (rooms : ℕ) : ℝ := sorry

/-- The work rate is proportional to the number of painters -/
axiom work_rate_proportional (p1 p2 r1 r2 : ℕ) (t : ℝ) :
  time_to_complete p1 r1 = t → time_to_complete p2 r2 = t * (r2 * p1 : ℝ) / (r1 * p2 : ℝ)

theorem three_painters_three_rooms : 
  time_to_complete 9 27 = 9 → time_to_complete 3 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_painters_three_rooms_l2420_242088


namespace NUMINAMATH_CALUDE_sector_radius_l2420_242092

/-- Given a sector with arc length and area, calculate its radius -/
theorem sector_radius (arc_length : ℝ) (area : ℝ) (radius : ℝ) : 
  arc_length = 2 → area = 2 → (1/2) * arc_length * radius = area → radius = 2 := by
  sorry

#check sector_radius

end NUMINAMATH_CALUDE_sector_radius_l2420_242092


namespace NUMINAMATH_CALUDE_complex_number_equation_l2420_242095

theorem complex_number_equation : ∃ z : ℂ, z / (1 + Complex.I) = Complex.I ^ 2015 + Complex.I ^ 2016 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equation_l2420_242095


namespace NUMINAMATH_CALUDE_workers_wage_increase_l2420_242077

/-- If a worker's daily wage is increased by 40% resulting in a new wage of $35 per day, 
    then the original daily wage was $25. -/
theorem workers_wage_increase (original_wage : ℝ) 
  (h1 : original_wage * 1.4 = 35) : original_wage = 25 := by
  sorry

end NUMINAMATH_CALUDE_workers_wage_increase_l2420_242077


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l2420_242015

theorem intersection_in_fourth_quadrant (k : ℝ) :
  let line1 : ℝ → ℝ := λ x => -2 * x + 3 * k + 14
  let line2 : ℝ → ℝ := λ y => (3 * k + 2 + 4 * y) / 1
  let x := k + 6
  let y := k + 2
  (∀ x', line1 x' = line2 x') →
  (x > 0 ∧ y < 0) →
  -6 < k ∧ k < -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l2420_242015


namespace NUMINAMATH_CALUDE_unique_solution_system_l2420_242029

theorem unique_solution_system (a b c : ℝ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  a^4 + b^2 * c^2 = 16 * a ∧
  b^4 + c^2 * a^2 = 16 * b ∧
  c^4 + a^2 * b^2 = 16 * c →
  a = 2 ∧ b = 2 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2420_242029


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l2420_242043

theorem adult_tickets_sold (children_tickets : ℕ) (children_price : ℕ) (adult_price : ℕ) (total_earnings : ℕ) : 
  children_tickets = 210 →
  children_price = 25 →
  adult_price = 50 →
  total_earnings = 5950 →
  (total_earnings - children_tickets * children_price) / adult_price = 14 :=
by sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l2420_242043


namespace NUMINAMATH_CALUDE_min_dot_product_l2420_242051

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the center O and left focus F
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-3, 0)

-- Define a point P on the right branch of the hyperbola
def P : ℝ × ℝ → Prop := λ p => hyperbola p.1 p.2 ∧ p.1 ≥ 2

-- Define the dot product of OP and FP
def dot_product (p : ℝ × ℝ) : ℝ := p.1 * (p.1 + 3) + p.2 * p.2

-- Theorem statement
theorem min_dot_product :
  ∀ p : ℝ × ℝ, P p → ∀ q : ℝ × ℝ, P q → dot_product p ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_l2420_242051


namespace NUMINAMATH_CALUDE_equal_diagonal_polygon_is_quadrilateral_or_pentagon_l2420_242047

/-- A convex polygon with n sides and all diagonals equal -/
structure EqualDiagonalPolygon where
  n : ℕ
  sides : n ≥ 4
  convex : Bool
  all_diagonals_equal : Bool

/-- The set of quadrilaterals -/
def Quadrilaterals : Set EqualDiagonalPolygon :=
  {p : EqualDiagonalPolygon | p.n = 4}

/-- The set of pentagons -/
def Pentagons : Set EqualDiagonalPolygon :=
  {p : EqualDiagonalPolygon | p.n = 5}

theorem equal_diagonal_polygon_is_quadrilateral_or_pentagon 
  (F : EqualDiagonalPolygon) (h_convex : F.convex = true) 
  (h_diag : F.all_diagonals_equal = true) :
  F ∈ Quadrilaterals ∪ Pentagons :=
sorry

end NUMINAMATH_CALUDE_equal_diagonal_polygon_is_quadrilateral_or_pentagon_l2420_242047


namespace NUMINAMATH_CALUDE_pet_store_dogs_l2420_242099

theorem pet_store_dogs (cat_count : ℕ) (cat_ratio dog_ratio : ℕ) : 
  cat_count = 18 → cat_ratio = 3 → dog_ratio = 4 → 
  (cat_count * dog_ratio) / cat_ratio = 24 := by
sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l2420_242099


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l2420_242097

/-- The perimeter of a semi-circle with radius 14 cm is 14π + 28 cm -/
theorem semicircle_perimeter :
  let r : ℝ := 14
  let diameter : ℝ := 2 * r
  let half_circumference : ℝ := π * r
  let perimeter : ℝ := half_circumference + diameter
  perimeter = 14 * π + 28 := by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l2420_242097


namespace NUMINAMATH_CALUDE_men_work_hours_per_day_l2420_242094

-- Define the number of men, women, and days
def num_men : ℕ := 15
def num_women : ℕ := 21
def days_men : ℕ := 21
def days_women : ℕ := 20
def hours_women : ℕ := 9

-- Define the ratio of work done by women to men
def women_to_men_ratio : ℚ := 2 / 3

-- Define the function to calculate total work hours
def total_work_hours (num_workers : ℕ) (num_days : ℕ) (hours_per_day : ℕ) : ℕ :=
  num_workers * num_days * hours_per_day

-- Theorem statement
theorem men_work_hours_per_day :
  ∃ (hours_men : ℕ),
    (total_work_hours num_men days_men hours_men : ℚ) * women_to_men_ratio =
    (total_work_hours num_women days_women hours_women : ℚ) ∧
    hours_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_men_work_hours_per_day_l2420_242094


namespace NUMINAMATH_CALUDE_point_transformation_l2420_242075

def initial_point : ℝ × ℝ × ℝ := (2, 3, -1)

def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  reflect_yz (reflect_xz (rotate_z_90 p))

theorem point_transformation :
  transform initial_point = (3, -2, -1) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l2420_242075


namespace NUMINAMATH_CALUDE_union_equals_A_l2420_242061

def A : Set ℤ := {-1, 0, 1}
def B (a : ℤ) : Set ℤ := {a, a^2}

theorem union_equals_A (a : ℤ) : A ∪ B a = A ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l2420_242061


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2420_242014

/-- Given a rectangle with area 300 square meters, if its length is doubled and
    its width is tripled, the area of the new rectangle will be 1800 square meters. -/
theorem rectangle_area_change (length width : ℝ) 
    (h_area : length * width = 300) : 
    (2 * length) * (3 * width) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2420_242014


namespace NUMINAMATH_CALUDE_empty_seats_arrangements_l2420_242016

/-- The number of chairs in a row -/
def num_chairs : ℕ := 8

/-- The number of students taking seats -/
def num_students : ℕ := 4

/-- The number of empty seats -/
def num_empty_seats : ℕ := num_chairs - num_students

/-- Calculates the number of ways to arrange all empty seats adjacent to each other -/
def adjacent_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial (n - k + 1)

/-- Calculates the number of ways to arrange all empty seats not adjacent to each other -/
def non_adjacent_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial k * Nat.choose (n - k + 1) k

theorem empty_seats_arrangements :
  adjacent_arrangements num_chairs num_empty_seats = 120 ∧
  non_adjacent_arrangements num_chairs num_empty_seats = 120 := by
  sorry

end NUMINAMATH_CALUDE_empty_seats_arrangements_l2420_242016
