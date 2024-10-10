import Mathlib

namespace unique_solution_5a_7b_plus_4_eq_3c_l1523_152323

theorem unique_solution_5a_7b_plus_4_eq_3c :
  ∀ a b c : ℕ, 5^a * 7^b + 4 = 3^c → a = 1 ∧ b = 0 ∧ c = 2 :=
by sorry

end unique_solution_5a_7b_plus_4_eq_3c_l1523_152323


namespace similar_polygons_area_sum_l1523_152333

/-- Given two similar polygons with corresponding sides a and b, 
    we can construct a third similar polygon with side c -/
theorem similar_polygons_area_sum 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (similar : ∃ (k : ℝ), k > 0 ∧ b = k * a) :
  ∃ (c : ℝ), 
    c > 0 ∧ 
    c^2 = a^2 + b^2 ∧ 
    ∃ (k : ℝ), k > 0 ∧ c = k * a ∧
    c^2 / a^2 = (a^2 + b^2) / a^2 :=
by sorry

end similar_polygons_area_sum_l1523_152333


namespace ending_number_divisible_by_eleven_l1523_152324

theorem ending_number_divisible_by_eleven (start : Nat) (count : Nat) : 
  start ≥ 29 →
  start % 11 = 0 →
  count = 5 →
  ∀ k, k ∈ Finset.range count → (start + k * 11) % 11 = 0 →
  start + (count - 1) * 11 = 77 :=
by sorry

end ending_number_divisible_by_eleven_l1523_152324


namespace billion_product_without_zeros_l1523_152340

theorem billion_product_without_zeros :
  ∃ (a b : ℕ), 
    a * b = 1000000000 ∧ 
    (∀ d : ℕ, d > 0 → d ≤ 9 → (a / 10^d) % 10 ≠ 0) ∧
    (∀ d : ℕ, d > 0 → d ≤ 9 → (b / 10^d) % 10 ≠ 0) :=
by sorry

end billion_product_without_zeros_l1523_152340


namespace smallest_alpha_inequality_half_satisfies_inequality_smallest_alpha_is_half_l1523_152361

theorem smallest_alpha_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∀ α : ℝ, α > 0 → α < 1/2 →
    ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + y) / 2 < α * Real.sqrt (x * y) + (1 - α) * Real.sqrt ((x^2 + y^2) / 2) :=
by sorry

theorem half_satisfies_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / 2 ≥ (1/2) * Real.sqrt (x * y) + (1/2) * Real.sqrt ((x^2 + y^2) / 2) :=
by sorry

theorem smallest_alpha_is_half :
  ∀ α : ℝ, α > 0 →
    (∀ x y : ℝ, x > 0 → y > 0 →
      (x + y) / 2 ≥ α * Real.sqrt (x * y) + (1 - α) * Real.sqrt ((x^2 + y^2) / 2)) →
    α ≥ 1/2 :=
by sorry

end smallest_alpha_inequality_half_satisfies_inequality_smallest_alpha_is_half_l1523_152361


namespace boundary_slopes_sum_l1523_152344

/-- Parabola P with equation y = x^2 + 4x + 4 -/
def P : ℝ → ℝ := λ x => x^2 + 4*x + 4

/-- Point Q -/
def Q : ℝ × ℝ := (10, 16)

/-- Function to determine if a line with slope m through Q intersects P -/
def intersects (m : ℝ) : Prop :=
  ∃ x : ℝ, P x = Q.2 + m * (x - Q.1)

/-- The lower boundary slope -/
noncomputable def r : ℝ := -24 - 16 * Real.sqrt 2

/-- The upper boundary slope -/
noncomputable def s : ℝ := -24 + 16 * Real.sqrt 2

/-- Theorem stating that r + s = -48 -/
theorem boundary_slopes_sum : r + s = -48 := by sorry

end boundary_slopes_sum_l1523_152344


namespace bus_passengers_after_three_stops_l1523_152338

theorem bus_passengers_after_three_stops : 
  let initial_passengers := 0
  let first_stop_on := 7
  let second_stop_off := 3
  let second_stop_on := 5
  let third_stop_off := 2
  let third_stop_on := 4
  
  let after_first_stop := initial_passengers + first_stop_on
  let after_second_stop := after_first_stop - second_stop_off + second_stop_on
  let after_third_stop := after_second_stop - third_stop_off + third_stop_on
  
  after_third_stop = 11 := by sorry

end bus_passengers_after_three_stops_l1523_152338


namespace rectangle_sides_theorem_l1523_152391

/-- A pair of positive integers representing the sides of a rectangle --/
structure RectangleSides where
  x : ℕ+
  y : ℕ+

/-- The set of rectangle sides that satisfy the perimeter-area equality condition --/
def validRectangleSides : Set RectangleSides :=
  { sides | (2 * sides.x.val + 2 * sides.y.val : ℕ) = sides.x.val * sides.y.val }

/-- The theorem stating that only three specific pairs of sides satisfy the conditions --/
theorem rectangle_sides_theorem :
  validRectangleSides = {⟨3, 6⟩, ⟨6, 3⟩, ⟨4, 4⟩} := by
  sorry

end rectangle_sides_theorem_l1523_152391


namespace imaginary_part_of_complex_fraction_l1523_152325

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (2 + 3*I) / (-3 + 2*I)
  Complex.im z = -1 := by sorry

end imaginary_part_of_complex_fraction_l1523_152325


namespace fort_blocks_count_l1523_152396

/-- Represents the dimensions of a rectangular structure -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular structure given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the specifications of the fort -/
structure FortSpecs where
  outerDimensions : Dimensions
  wallThickness : ℕ
  floorThickness : ℕ

/-- Calculates the inner dimensions of the fort given its specifications -/
def innerDimensions (specs : FortSpecs) : Dimensions :=
  { length := specs.outerDimensions.length - 2 * specs.wallThickness,
    width := specs.outerDimensions.width - 2 * specs.wallThickness,
    height := specs.outerDimensions.height - specs.floorThickness }

/-- Calculates the number of blocks needed for the fort -/
def blocksNeeded (specs : FortSpecs) : ℕ :=
  volume specs.outerDimensions - volume (innerDimensions specs)

theorem fort_blocks_count : 
  let fortSpecs : FortSpecs := 
    { outerDimensions := { length := 20, width := 15, height := 8 },
      wallThickness := 2,
      floorThickness := 1 }
  blocksNeeded fortSpecs = 1168 := by sorry

end fort_blocks_count_l1523_152396


namespace problem_solution_l1523_152350

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m|

-- State the theorem
theorem problem_solution :
  -- Given conditions
  (∀ x : ℝ, f x 2 ≤ 3 ↔ x ∈ Set.Icc (-1) 5) →
  -- Part I: m = 2
  (∃ m : ℝ, ∀ x : ℝ, f x m ≤ 3 ↔ x ∈ Set.Icc (-1) 5) ∧ 
  -- Part II: Minimum value of a² + b² + c² is 2/3
  (∀ a b c : ℝ, a - 2*b + c = 2 → a^2 + b^2 + c^2 ≥ 2/3) ∧
  (∃ a b c : ℝ, a - 2*b + c = 2 ∧ a^2 + b^2 + c^2 = 2/3) :=
by sorry


end problem_solution_l1523_152350


namespace inequality_solution_l1523_152354

def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 ∨ m = -12 then
    {x | x ≠ m / 6}
  else if m < -12 ∨ m > 0 then
    {x | x < (m - Real.sqrt (m^2 + 12*m)) / 6 ∨ x > (m + Real.sqrt (m^2 + 12*m)) / 6}
  else
    Set.univ

theorem inequality_solution (m : ℝ) :
  {x : ℝ | 3 * x^2 - m * x - m > 0} = solution_set m :=
by sorry

end inequality_solution_l1523_152354


namespace brand_z_fraction_fraction_to_percentage_l1523_152385

/-- Represents the state of the fuel tank -/
structure TankState where
  z : ℚ  -- Amount of brand Z gasoline
  y : ℚ  -- Amount of brand Y gasoline

/-- Fills the tank with brand Z gasoline when empty -/
def initial_fill : TankState :=
  { z := 1, y := 0 }

/-- Fills the tank with brand Y when 1/4 empty -/
def first_refill (s : TankState) : TankState :=
  { z := 3/4 * s.z, y := 1/4 + 3/4 * s.y }

/-- Fills the tank with brand Z when half empty -/
def second_refill (s : TankState) : TankState :=
  { z := 1/2 + 1/2 * s.z, y := 1/2 * s.y }

/-- Fills the tank with brand Y when half empty -/
def third_refill (s : TankState) : TankState :=
  { z := 1/2 * s.z, y := 1/2 + 1/2 * s.y }

/-- The final state of the tank after all refills -/
def final_state : TankState :=
  third_refill (second_refill (first_refill initial_fill))

/-- Theorem stating that the fraction of brand Z gasoline in the final state is 7/16 -/
theorem brand_z_fraction :
  final_state.z / (final_state.z + final_state.y) = 7/16 := by
  sorry

/-- Theorem stating that 7/16 is equivalent to 43.75% -/
theorem fraction_to_percentage :
  (7/16 : ℚ) = 43.75/100 := by
  sorry

end brand_z_fraction_fraction_to_percentage_l1523_152385


namespace tangent_sum_l1523_152389

-- Define the tangent and cotangent functions
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- State the theorem
theorem tangent_sum (A B : ℝ) 
  (h1 : tg A + tg B = 2) 
  (h2 : ctg A + ctg B = 3) : 
  tg (A + B) = 6 := by sorry

end tangent_sum_l1523_152389


namespace negative_two_x_squared_cubed_l1523_152326

theorem negative_two_x_squared_cubed (x : ℝ) : (-2 * x^2)^3 = -8 * x^6 := by
  sorry

end negative_two_x_squared_cubed_l1523_152326


namespace arithmetic_progression_fifth_term_l1523_152355

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def isArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_progression_fifth_term
  (a : ℕ → ℝ)
  (h_ap : isArithmeticProgression a)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 45) :
  a 5 = 9 := by
  sorry

end arithmetic_progression_fifth_term_l1523_152355


namespace parabola_focus_distance_l1523_152351

/-- Theorem: For a parabola x^2 = 2py (p > 0) with a point A(m, 4) on it,
    if the distance from A to its focus is 17/4, then p = 1/2 and m = ±2. -/
theorem parabola_focus_distance (p m : ℝ) : 
  p > 0 →  -- p is positive
  m^2 = 2*p*4 →  -- A(m, 4) is on the parabola
  (m^2 + (4 - p/2)^2)^(1/2) = 17/4 →  -- Distance from A to focus is 17/4
  p = 1/2 ∧ (m = 2 ∨ m = -2) := by
sorry

end parabola_focus_distance_l1523_152351


namespace min_value_expression_l1523_152364

theorem min_value_expression (x y z : ℝ) (h : 2 * x * y + y * z > 0) :
  (x^2 + y^2 + z^2) / (2 * x * y + y * z) ≥ 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), 2 * x₀ * y₀ + y₀ * z₀ > 0 ∧
    (x₀^2 + y₀^2 + z₀^2) / (2 * x₀ * y₀ + y₀ * z₀) = 3 :=
by sorry

end min_value_expression_l1523_152364


namespace cube_root_1600_l1523_152387

theorem cube_root_1600 (c d : ℕ+) (h1 : (1600 : ℝ)^(1/3) = c * d^(1/3)) 
  (h2 : ∀ (c' d' : ℕ+), (1600 : ℝ)^(1/3) = c' * d'^(1/3) → d ≤ d') : 
  c + d = 29 := by
sorry

end cube_root_1600_l1523_152387


namespace digit_selection_theorem_l1523_152330

/-- The number of digits available for selection -/
def n : ℕ := 10

/-- The number of digits to be selected -/
def k : ℕ := 4

/-- Function to calculate the number of permutations without repetition -/
def permutations_without_repetition (n k : ℕ) : ℕ := sorry

/-- Function to calculate the number of four-digit numbers without repetition -/
def four_digit_numbers_without_repetition (n k : ℕ) : ℕ := sorry

/-- Function to calculate the number of even four-digit numbers greater than 3000 without repetition -/
def even_four_digit_numbers_gt_3000_without_repetition (n k : ℕ) : ℕ := sorry

theorem digit_selection_theorem :
  permutations_without_repetition n k = 5040 ∧
  four_digit_numbers_without_repetition n k = 4356 ∧
  even_four_digit_numbers_gt_3000_without_repetition n k = 1792 := by
  sorry

end digit_selection_theorem_l1523_152330


namespace divisibility_property_l1523_152302

theorem divisibility_property (a b : ℤ) : (7 ∣ a^2 + b^2) → (7 ∣ a) ∧ (7 ∣ b) := by
  sorry

end divisibility_property_l1523_152302


namespace max_subsets_exists_444_subsets_l1523_152307

/-- A structure representing a collection of 3-element subsets of a 1000-element set. -/
structure SubsetCollection where
  /-- The underlying 1000-element set -/
  base : Finset (Fin 1000)
  /-- The collection of 3-element subsets -/
  subsets : Finset (Finset (Fin 1000))
  /-- Each subset has exactly 3 elements -/
  three_element : ∀ s ∈ subsets, Finset.card s = 3
  /-- Each subset is a subset of the base set -/
  subset_of_base : ∀ s ∈ subsets, s ⊆ base
  /-- The union of any 5 subsets has at least 12 elements -/
  union_property : ∀ (five_subsets : Finset (Finset (Fin 1000))), 
    five_subsets ⊆ subsets → Finset.card five_subsets = 5 → 
    Finset.card (Finset.biUnion five_subsets id) ≥ 12

/-- The maximum number of three-element subsets satisfying the given conditions is 444. -/
theorem max_subsets (sc : SubsetCollection) : Finset.card sc.subsets ≤ 444 := by
  sorry

/-- There exists a collection of 444 three-element subsets satisfying the given conditions. -/
theorem exists_444_subsets : ∃ sc : SubsetCollection, Finset.card sc.subsets = 444 := by
  sorry

end max_subsets_exists_444_subsets_l1523_152307


namespace smallest_union_size_l1523_152309

theorem smallest_union_size (A B : Finset ℕ) 
  (hA : A.card = 30)
  (hB : B.card = 20)
  (hInter : (A ∩ B).card ≥ 10) :
  (A ∪ B).card ≥ 40 ∧ ∃ (C D : Finset ℕ), C.card = 30 ∧ D.card = 20 ∧ (C ∩ D).card ≥ 10 ∧ (C ∪ D).card = 40 :=
by
  sorry

end smallest_union_size_l1523_152309


namespace inequality_proof_l1523_152358

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (1 - x) * (1 - y) * (1 - z) > 8 * x * y * z := by
  sorry

end inequality_proof_l1523_152358


namespace box_height_l1523_152320

/-- Calculates the height of a box with given specifications -/
theorem box_height (internal_volume : ℕ) (external_side_length : ℕ) : 
  internal_volume = 6912 ∧ 
  external_side_length = 26 → 
  (external_side_length - 2)^2 * 12 = internal_volume := by
  sorry

#check box_height

end box_height_l1523_152320


namespace line_ellipse_intersection_range_l1523_152379

/-- The range of values for m where the line y = x + m intersects the ellipse x^2/4 + y^2/3 = 1 -/
theorem line_ellipse_intersection_range :
  let line (x m : ℝ) := x + m
  let ellipse (x y : ℝ) := x^2/4 + y^2/3 = 1
  let intersects (m : ℝ) := ∃ x, ellipse x (line x m)
  ∀ m, intersects m ↔ m ∈ Set.Icc (-Real.sqrt 7) (Real.sqrt 7) :=
by sorry


end line_ellipse_intersection_range_l1523_152379


namespace vincent_sticker_packs_l1523_152349

/-- The number of packs Vincent bought yesterday -/
def yesterday_packs : ℕ := sorry

/-- The number of packs Vincent bought today -/
def today_packs : ℕ := yesterday_packs + 10

/-- The total number of packs Vincent has -/
def total_packs : ℕ := 40

theorem vincent_sticker_packs : yesterday_packs = 15 := by
  sorry

end vincent_sticker_packs_l1523_152349


namespace quadratic_two_distinct_roots_l1523_152381

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 6*x₁ + 9*k = 0 ∧ x₂^2 - 6*x₂ + 9*k = 0) ↔ k < 1 :=
sorry

end quadratic_two_distinct_roots_l1523_152381


namespace inequality_proof_l1523_152341

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 3*b^3) / (5*a + b) + (b^3 + 3*c^3) / (5*b + c) + (c^3 + 3*a^3) / (5*c + a) ≥ 2/3 * (a^2 + b^2 + c^2) := by
  sorry

end inequality_proof_l1523_152341


namespace milk_production_theorem_l1523_152372

/-- Represents the milk production scenario -/
structure MilkProduction where
  initial_cows : ℕ
  initial_days : ℕ
  initial_gallons : ℕ
  max_daily_per_cow : ℕ
  available_cows : ℕ
  target_days : ℕ

/-- Calculates the total milk production given the scenario -/
def total_milk_production (mp : MilkProduction) : ℕ :=
  let daily_rate_per_cow := mp.initial_gallons / (mp.initial_cows * mp.initial_days)
  let actual_rate := min daily_rate_per_cow mp.max_daily_per_cow
  mp.available_cows * actual_rate * mp.target_days

/-- Theorem stating that the total milk production is 96 gallons -/
theorem milk_production_theorem (mp : MilkProduction) 
  (h1 : mp.initial_cows = 10)
  (h2 : mp.initial_days = 5)
  (h3 : mp.initial_gallons = 40)
  (h4 : mp.max_daily_per_cow = 2)
  (h5 : mp.available_cows = 15)
  (h6 : mp.target_days = 8) :
  total_milk_production mp = 96 := by
  sorry

end milk_production_theorem_l1523_152372


namespace quadratic_roots_range_l1523_152386

theorem quadratic_roots_range (h1 : ∃ x : ℝ, Real.log x < 0)
  (h2 : ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0))
  (m : ℝ) :
  -2 ≤ m ∧ m ≤ 2 := by
sorry

end quadratic_roots_range_l1523_152386


namespace charlie_prob_different_colors_l1523_152366

/-- Represents the number of marbles of each color -/
def num_marbles : ℕ := 3

/-- Represents the total number of marbles -/
def total_marbles : ℕ := 3 * num_marbles

/-- Represents the number of marbles each person takes -/
def marbles_per_person : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the total number of ways the marbles can be drawn -/
def total_ways : ℕ := 
  (choose total_marbles marbles_per_person) * 
  (choose (total_marbles - marbles_per_person) marbles_per_person) * 
  (choose marbles_per_person marbles_per_person)

/-- Calculates the number of favorable outcomes for Charlie -/
def favorable_outcomes : ℕ := 
  (choose num_marbles 2) * (choose num_marbles 2) * (choose num_marbles 2)

/-- The probability of Charlie drawing three different colored marbles -/
theorem charlie_prob_different_colors : 
  (favorable_outcomes : ℚ) / total_ways = 5 / 8 := by sorry

end charlie_prob_different_colors_l1523_152366


namespace event_probability_l1523_152313

theorem event_probability (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (1 - (1 - p)^3 = 63/64) →
  (3 * p * (1 - p)^2 = 9/64) :=
by
  sorry

end event_probability_l1523_152313


namespace surface_area_of_problem_solid_l1523_152376

/-- Represents a solid formed by unit cubes -/
structure CubeSolid where
  base_length : ℕ
  top_cube_position : ℕ
  total_cubes : ℕ

/-- Calculate the surface area of the cube solid -/
def surface_area (solid : CubeSolid) : ℕ :=
  -- Front and back
  2 * solid.base_length +
  -- Left and right sides
  (solid.base_length - 1) + (solid.top_cube_position + 3) +
  -- Top surface
  solid.base_length + 1

/-- The specific cube solid described in the problem -/
def problem_solid : CubeSolid :=
  { base_length := 7
  , top_cube_position := 2
  , total_cubes := 8 }

theorem surface_area_of_problem_solid :
  surface_area problem_solid = 34 := by
  sorry

end surface_area_of_problem_solid_l1523_152376


namespace shells_given_to_brother_l1523_152393

def shells_per_day : ℕ := 10
def days_collecting : ℕ := 6
def shells_remaining : ℕ := 58

theorem shells_given_to_brother :
  shells_per_day * days_collecting - shells_remaining = 2 := by
  sorry

end shells_given_to_brother_l1523_152393


namespace negation_of_forall_gt_one_negation_of_inequality_l1523_152374

theorem negation_of_forall_gt_one (P : ℝ → Prop) :
  (¬ ∀ x > 1, P x) ↔ (∃ x > 1, ¬ P x) := by sorry

theorem negation_of_inequality :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by sorry

end negation_of_forall_gt_one_negation_of_inequality_l1523_152374


namespace linear_system_fraction_l1523_152328

theorem linear_system_fraction (x y m n : ℚ) 
  (eq1 : x + 2*y = 5)
  (eq2 : x + y = 7)
  (eq3 : x = -m)
  (eq4 : y = -n) :
  (3*m + 2*n) / (5*m - n) = 11/14 := by
sorry

end linear_system_fraction_l1523_152328


namespace mobile_phone_costs_and_schemes_l1523_152315

/-- Given the cost equations for mobile phones, this theorem proves the costs of each type
    and the number of valid purchasing schemes. -/
theorem mobile_phone_costs_and_schemes :
  ∃ (cost_A cost_B : ℕ) (num_schemes : ℕ),
    -- Cost equations
    (2 * cost_A + 3 * cost_B = 7400) ∧
    (3 * cost_A + 5 * cost_B = 11700) ∧
    -- Costs of phones
    (cost_A = 1900) ∧
    (cost_B = 1200) ∧
    -- Number of valid purchasing schemes
    (num_schemes = 9) ∧
    -- Definition of valid purchasing schemes
    (∀ m : ℕ, 
      (12 ≤ m ∧ m ≤ 20) ↔ 
      (44400 ≤ 1900*m + 1200*(30-m) ∧ 1900*m + 1200*(30-m) ≤ 50000)) := by
  sorry


end mobile_phone_costs_and_schemes_l1523_152315


namespace determinant_scaling_l1523_152357

theorem determinant_scaling (x y z a b c p q r : ℝ) :
  Matrix.det !![x, y, z; a, b, c; p, q, r] = 2 →
  Matrix.det !![3*x, 3*y, 3*z; 3*a, 3*b, 3*c; 3*p, 3*q, 3*r] = 54 := by
  sorry

end determinant_scaling_l1523_152357


namespace power_and_division_simplification_l1523_152356

theorem power_and_division_simplification : 1^567 + 3^5 / 3^3 - 2 = 8 := by
  sorry

end power_and_division_simplification_l1523_152356


namespace second_bottle_capacity_l1523_152375

theorem second_bottle_capacity
  (total_milk : ℝ)
  (first_bottle_capacity : ℝ)
  (second_bottle_milk : ℝ)
  (h1 : total_milk = 8)
  (h2 : first_bottle_capacity = 4)
  (h3 : second_bottle_milk = 16 / 3)
  (h4 : ∃ (f : ℝ), f * first_bottle_capacity + second_bottle_milk = total_milk ∧
                   f * first_bottle_capacity ≤ first_bottle_capacity ∧
                   second_bottle_milk ≤ f * (total_milk - first_bottle_capacity * f)) :
  total_milk - first_bottle_capacity * (total_milk - second_bottle_milk) / first_bottle_capacity = 8 :=
by
  sorry

end second_bottle_capacity_l1523_152375


namespace smallest_integer_with_remainder_two_l1523_152346

theorem smallest_integer_with_remainder_two (n : ℕ) : 
  (n > 1) →
  (n % 3 = 2) →
  (n % 5 = 2) →
  (n % 7 = 2) →
  (∀ m : ℕ, m > 1 ∧ m % 3 = 2 ∧ m % 5 = 2 ∧ m % 7 = 2 → m ≥ n) →
  n = 107 :=
by
  sorry

#check smallest_integer_with_remainder_two

end smallest_integer_with_remainder_two_l1523_152346


namespace cuboid_volume_l1523_152343

theorem cuboid_volume (a b c : ℝ) : 
  (a^2 + b^2 + c^2 = 16) →  -- space diagonal length is 4
  (a / 4 = 1/2) →           -- edge a forms 60° angle with diagonal
  (b / 4 = 1/2) →           -- edge b forms 60° angle with diagonal
  (c / 4 = 1/2) →           -- edge c forms 60° angle with diagonal
  (a * b * c = 8) :=        -- volume is 8
by sorry

end cuboid_volume_l1523_152343


namespace outdoor_section_length_l1523_152383

/-- Given a rectangular outdoor section with area 35 square feet and width 7 feet, 
    the length of the section is 5 feet. -/
theorem outdoor_section_length : 
  ∀ (area width length : ℝ), 
    area = 35 → 
    width = 7 → 
    area = width * length → 
    length = 5 := by
sorry

end outdoor_section_length_l1523_152383


namespace average_first_n_odd_numbers_l1523_152382

/-- The nth odd number -/
def nthOddNumber (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of the first n odd numbers -/
def sumFirstNOddNumbers (n : ℕ) : ℕ := n ^ 2

/-- The average of the first n odd numbers -/
def averageFirstNOddNumbers (n : ℕ) : ℕ := sumFirstNOddNumbers n / n

theorem average_first_n_odd_numbers (n : ℕ) (h : n > 0) :
  averageFirstNOddNumbers n = nthOddNumber n := by
  sorry

end average_first_n_odd_numbers_l1523_152382


namespace sum_of_last_two_digits_fibonacci_factorial_series_l1523_152322

def last_two_digits (n : ℕ) : ℕ := n % 100

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 144]

def factorial_ends_in_zeros (n : ℕ) : Prop := n > 10 → last_two_digits (n.factorial) = 0

theorem sum_of_last_two_digits_fibonacci_factorial_series :
  factorial_ends_in_zeros 11 →
  (fibonacci_factorial_series.map (λ n => last_two_digits n.factorial)).sum = 50 := by
  sorry

end sum_of_last_two_digits_fibonacci_factorial_series_l1523_152322


namespace halfDollarProbabilityIs3_16_l1523_152342

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | HalfDollar
  | Quarter

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Dime => 10
  | Coin.HalfDollar => 50
  | Coin.Quarter => 25

/-- The total value of each coin type in cents -/
def totalValue : Coin → ℕ
  | Coin.Dime => 2000
  | Coin.HalfDollar => 3000
  | Coin.Quarter => 1500

/-- The number of coins of each type -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Dime + coinCount Coin.HalfDollar + coinCount Coin.Quarter

/-- The probability of selecting a half-dollar -/
def halfDollarProbability : ℚ := coinCount Coin.HalfDollar / totalCoins

theorem halfDollarProbabilityIs3_16 : halfDollarProbability = 3 / 16 := by
  sorry

end halfDollarProbabilityIs3_16_l1523_152342


namespace final_selling_price_approx_1949_l1523_152399

/-- Calculate the final selling price of a cycle, helmet, and safety lights --/
def calculate_final_selling_price (cycle_cost helmet_cost safety_light_cost : ℚ) 
  (num_safety_lights : ℕ) (cycle_discount tax_rate cycle_loss helmet_profit transaction_fee : ℚ) : ℚ :=
  let cycle_discounted := cycle_cost * (1 - cycle_discount)
  let total_cost := cycle_discounted + helmet_cost + (safety_light_cost * num_safety_lights)
  let total_with_tax := total_cost * (1 + tax_rate)
  let cycle_selling := cycle_discounted * (1 - cycle_loss)
  let helmet_selling := helmet_cost * (1 + helmet_profit)
  let safety_lights_selling := safety_light_cost * num_safety_lights
  let total_selling := cycle_selling + helmet_selling + safety_lights_selling
  let final_price := total_selling * (1 - transaction_fee)
  final_price

/-- Theorem stating that the final selling price is approximately 1949 --/
theorem final_selling_price_approx_1949 : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_final_selling_price 1400 400 200 2 (10/100) (5/100) (12/100) (25/100) (3/100) - 1949| < ε :=
sorry

end final_selling_price_approx_1949_l1523_152399


namespace square_difference_263_257_l1523_152319

theorem square_difference_263_257 : 263^2 - 257^2 = 3120 := by
  sorry

end square_difference_263_257_l1523_152319


namespace time_to_find_worm_l1523_152321

/-- Given Kevin's toad feeding scenario, prove the time to find each worm. -/
theorem time_to_find_worm (num_toads : ℕ) (worms_per_toad : ℕ) (total_hours : ℕ) :
  num_toads = 8 →
  worms_per_toad = 3 →
  total_hours = 6 →
  (total_hours * 60) / (num_toads * worms_per_toad) = 15 :=
by sorry

end time_to_find_worm_l1523_152321


namespace unique_hyperbolas_l1523_152373

/-- Binomial coefficient function -/
def binomial (m n : ℕ) : ℕ := Nat.choose m n

/-- The set of binomial coefficients for 1 ≤ n ≤ m ≤ 5 -/
def binomial_set : Finset ℕ :=
  Finset.filter (λ x => x > 1) $
    Finset.image (λ (m, n) => binomial m n) $
      Finset.filter (λ (m, n) => 1 ≤ n ∧ n ≤ m ∧ m ≤ 5) $
        Finset.product (Finset.range 6) (Finset.range 6)

theorem unique_hyperbolas : Finset.card binomial_set = 6 := by
  sorry

end unique_hyperbolas_l1523_152373


namespace min_hours_theorem_min_hours_sufficient_less_hours_insufficient_l1523_152397

/-- Represents the minimum number of hours required for all friends to know all news -/
def min_hours (N : ℕ) : ℕ :=
  if N = 64 then 6
  else if N = 55 then 7
  else if N = 100 then 7
  else 0  -- undefined for other values of N

/-- The theorem stating the minimum number of hours for specific N values -/
theorem min_hours_theorem :
  (min_hours 64 = 6) ∧ (min_hours 55 = 7) ∧ (min_hours 100 = 7) := by
  sorry

/-- Helper function to calculate the maximum number of friends who can know a piece of news after h hours -/
def max_friends_knowing (h : ℕ) : ℕ := 2^h

/-- Theorem stating that the minimum hours is sufficient for all friends to know all news -/
theorem min_hours_sufficient (N : ℕ) (h : ℕ) (h_eq : h = min_hours N) :
  max_friends_knowing h ≥ N := by
  sorry

/-- Theorem stating that one less hour is insufficient for all friends to know all news -/
theorem less_hours_insufficient (N : ℕ) (h : ℕ) (h_eq : h = min_hours N) :
  max_friends_knowing (h - 1) < N := by
  sorry

end min_hours_theorem_min_hours_sufficient_less_hours_insufficient_l1523_152397


namespace list_number_property_l1523_152345

theorem list_number_property (L : List ℝ) (n : ℝ) :
  L.length = 21 →
  L.Nodup →
  n ∈ L →
  n = 0.2 * L.sum →
  n = 5 * ((L.sum - n) / 20) :=
by sorry

end list_number_property_l1523_152345


namespace car_speed_l1523_152300

/-- Given a car that travels 325 miles in 5 hours, its speed is 65 miles per hour -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 325) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : speed = 65 := by
  sorry

end car_speed_l1523_152300


namespace max_gcd_sum_780_l1523_152359

theorem max_gcd_sum_780 :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 780 ∧
  ∀ (a b : ℕ), a > 0 → b > 0 → a + b = 780 → Nat.gcd a b ≤ Nat.gcd x y ∧
  Nat.gcd x y = 390 :=
sorry

end max_gcd_sum_780_l1523_152359


namespace antonio_age_is_51_months_l1523_152312

/- Define Isabella's current age in months -/
def isabella_age_months : ℕ := 10 * 12 - 18

/- Define the relationship between Isabella's and Antonio's ages -/
def antonio_age_months : ℕ := isabella_age_months / 2

/- Theorem stating Antonio's age in months -/
theorem antonio_age_is_51_months : antonio_age_months = 51 := by
  sorry

end antonio_age_is_51_months_l1523_152312


namespace skating_minutes_needed_for_average_l1523_152394

def skating_schedule (days : Nat) (hours_per_day : Nat) : Nat :=
  days * hours_per_day * 60

def total_minutes_8_days : Nat :=
  skating_schedule 6 1 + skating_schedule 2 2

def average_minutes_per_day : Nat := 100

def total_days : Nat := 10

theorem skating_minutes_needed_for_average :
  skating_schedule 6 1 + skating_schedule 2 2 + 400 = total_days * average_minutes_per_day := by
  sorry

end skating_minutes_needed_for_average_l1523_152394


namespace roses_per_girl_l1523_152363

theorem roses_per_girl (total_students : Nat) (total_plants : Nat) (total_birches : Nat) 
  (h1 : total_students = 24)
  (h2 : total_plants = 24)
  (h3 : total_birches = 6)
  (h4 : total_birches * 3 ≤ total_students) :
  ∃ (roses_per_girl : Nat), 
    roses_per_girl * (total_students - total_birches * 3) = total_plants - total_birches ∧ 
    roses_per_girl = 3 := by
  sorry

end roses_per_girl_l1523_152363


namespace bicycle_speed_l1523_152301

/-- Given a journey with two modes of transport (on foot and by bicycle), 
    calculate the speed of the bicycle. -/
theorem bicycle_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (foot_distance : ℝ) 
  (foot_speed : ℝ) 
  (h1 : total_distance = 80) 
  (h2 : total_time = 7) 
  (h3 : foot_distance = 32) 
  (h4 : foot_speed = 8) :
  (total_distance - foot_distance) / (total_time - foot_distance / foot_speed) = 16 := by
  sorry


end bicycle_speed_l1523_152301


namespace diagonal_sum_lower_bound_l1523_152395

/-- Given a convex quadrilateral ABCD with sides a, b, c, d and diagonals x, y,
    where a is the smallest side, prove that x + y ≥ (1 + √3)a -/
theorem diagonal_sum_lower_bound (a b c d x y : ℝ) :
  a > 0 →
  b ≥ a →
  c ≥ a →
  d ≥ a →
  x ≥ a →
  y ≥ a →
  x + y ≥ (1 + Real.sqrt 3) * a :=
by sorry

end diagonal_sum_lower_bound_l1523_152395


namespace parabola_vertices_distance_l1523_152316

/-- Given an equation representing portions of two parabolas, 
    this theorem states the distance between their vertices. -/
theorem parabola_vertices_distance : 
  ∃ (f g : ℝ → ℝ),
    (∀ x y : ℝ, (Real.sqrt (x^2 + y^2) + |y + 2| = 4) ↔ 
      ((y ≥ -2 ∧ y = f x) ∨ (y < -2 ∧ y = g x))) →
    ∃ (v1 v2 : ℝ × ℝ),
      (v1.1 = 0 ∧ v1.2 = f 0) ∧
      (v2.1 = 0 ∧ v2.2 = g 0) ∧
      Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 58 / 11 :=
sorry

end parabola_vertices_distance_l1523_152316


namespace box_height_is_nine_l1523_152377

/-- A rectangular box with dimensions 6 × 6 × h -/
structure Box (h : ℝ) where
  length : ℝ := 6
  width : ℝ := 6
  height : ℝ := h

/-- A sphere with a given radius -/
structure Sphere (r : ℝ) where
  radius : ℝ := r

/-- Predicate to check if a sphere is tangent to three sides of a box -/
def tangent_to_three_sides (s : Sphere r) (b : Box h) : Prop :=
  sorry

/-- Predicate to check if two spheres are tangent -/
def spheres_tangent (s1 : Sphere r1) (s2 : Sphere r2) : Prop :=
  sorry

/-- The main theorem -/
theorem box_height_is_nine :
  ∀ (h : ℝ) (b : Box h) (large_sphere : Sphere 3) (small_spheres : Fin 8 → Sphere 1.5),
    (∀ i, tangent_to_three_sides (small_spheres i) b) →
    (∀ i, spheres_tangent large_sphere (small_spheres i)) →
    h = 9 :=
by sorry

end box_height_is_nine_l1523_152377


namespace game_wheel_probability_l1523_152370

theorem game_wheel_probability (pX pY pZ pW : ℚ) : 
  pX = 1/4 → pY = 1/3 → pW = 1/6 → pX + pY + pZ + pW = 1 → pZ = 1/4 := by
  sorry

end game_wheel_probability_l1523_152370


namespace products_produced_is_twenty_l1523_152352

/-- Calculates the number of products produced given fixed cost, marginal cost, and total cost. -/
def products_produced (fixed_cost marginal_cost total_cost : ℚ) : ℚ :=
  (total_cost - fixed_cost) / marginal_cost

/-- Theorem stating that the number of products produced is 20 given the specified costs. -/
theorem products_produced_is_twenty :
  products_produced 12000 200 16000 = 20 := by
  sorry

#eval products_produced 12000 200 16000

end products_produced_is_twenty_l1523_152352


namespace edward_pipe_usage_l1523_152365

/- Define the problem parameters -/
def total_washers : ℕ := 20
def remaining_washers : ℕ := 4
def washers_per_bolt : ℕ := 2
def feet_per_bolt : ℕ := 5

/- Define the function to calculate feet of pipe used -/
def feet_of_pipe_used (total_washers remaining_washers washers_per_bolt feet_per_bolt : ℕ) : ℕ :=
  let washers_used := total_washers - remaining_washers
  let bolts_used := washers_used / washers_per_bolt
  bolts_used * feet_per_bolt

/- Theorem statement -/
theorem edward_pipe_usage :
  feet_of_pipe_used total_washers remaining_washers washers_per_bolt feet_per_bolt = 40 := by
  sorry

end edward_pipe_usage_l1523_152365


namespace quadratic_roots_relation_l1523_152329

theorem quadratic_roots_relation (a b c d : ℝ) (h : a ≠ 0 ∧ c ≠ 0) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ c * (x/2007)^2 + d * (x/2007) + a = 0) →
  b^2 = d^2 := by
sorry

end quadratic_roots_relation_l1523_152329


namespace least_positive_integer_multiple_of_53_l1523_152347

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧
  (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) ∧
  x = 4 :=
by sorry

end least_positive_integer_multiple_of_53_l1523_152347


namespace total_teachers_is_210_l1523_152378

/-- Represents the number of teachers in each category and the sample size -/
structure TeacherData where
  senior : ℕ
  intermediate : ℕ
  sample_size : ℕ
  other_sampled : ℕ

/-- Calculates the total number of teachers given the data -/
def totalTeachers (data : TeacherData) : ℕ :=
  sorry

/-- Theorem stating that given the conditions, the total number of teachers is 210 -/
theorem total_teachers_is_210 (data : TeacherData) 
  (h1 : data.senior = 104)
  (h2 : data.intermediate = 46)
  (h3 : data.sample_size = 42)
  (h4 : data.other_sampled = 12)
  (h5 : ∀ (category : ℕ), (category : ℚ) / (totalTeachers data : ℚ) = (data.sample_size : ℚ) / (totalTeachers data : ℚ)) :
  totalTeachers data = 210 :=
sorry

end total_teachers_is_210_l1523_152378


namespace f_strictly_decreasing_a_range_l1523_152337

/-- The piecewise function f(x) defined by a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

/-- The theorem stating the range of a for which f is strictly decreasing -/
theorem f_strictly_decreasing_a_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, (f a x₁ - f a x₂) * (x₁ - x₂) < 0) ↔ 0 < a ∧ a ≤ 1/4 :=
sorry

end f_strictly_decreasing_a_range_l1523_152337


namespace molecular_weight_of_BaSO4_l1523_152332

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.327

/-- The atomic weight of Sulfur in g/mol -/
def atomic_weight_S : ℝ := 32.065

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The number of moles of BaSO4 -/
def moles_BaSO4 : ℝ := 3

/-- The molecular weight of BaSO4 in g/mol -/
def molecular_weight_BaSO4 : ℝ := atomic_weight_Ba + atomic_weight_S + 4 * atomic_weight_O

/-- The total weight of the given moles of BaSO4 in grams -/
def total_weight_BaSO4 : ℝ := moles_BaSO4 * molecular_weight_BaSO4

theorem molecular_weight_of_BaSO4 :
  total_weight_BaSO4 = 700.164 := by sorry

end molecular_weight_of_BaSO4_l1523_152332


namespace b_profit_l1523_152369

-- Define the basic variables
def total_profit : ℕ := 21000

-- Define the investment ratio
def investment_ratio : ℕ := 3

-- Define the time ratio
def time_ratio : ℕ := 2

-- Define the profit sharing ratio
def profit_sharing_ratio : ℕ := investment_ratio * time_ratio

-- Theorem to prove
theorem b_profit (a_investment b_investment : ℕ) (a_time b_time : ℕ) :
  a_investment = investment_ratio * b_investment →
  a_time = time_ratio * b_time →
  (profit_sharing_ratio * b_investment * b_time + b_investment * b_time) * 3000 = total_profit * b_investment * b_time :=
by sorry


end b_profit_l1523_152369


namespace triangle_third_side_l1523_152368

theorem triangle_third_side (a b c : ℝ) (θ : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : θ = Real.pi / 3) :
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos θ → c = Real.sqrt 57 := by
sorry

end triangle_third_side_l1523_152368


namespace morning_campers_l1523_152390

theorem morning_campers (total : ℕ) (afternoon : ℕ) (morning : ℕ) : 
  total = 60 → afternoon = 7 → morning = total - afternoon → morning = 53 := by
sorry

end morning_campers_l1523_152390


namespace quadratic_function_uniqueness_l1523_152311

/-- A quadratic function of the form f(x) = x^2 + c*x + d -/
def f (c d : ℝ) (x : ℝ) : ℝ := x^2 + c*x + d

/-- The theorem stating the uniqueness of c and d for the given condition -/
theorem quadratic_function_uniqueness :
  ∀ c d : ℝ,
  (∀ x : ℝ, (f c d (f c d x + 2*x)) / (f c d x) = 2*x^2 + 1984*x + 2024) →
  c = 1982 ∧ d = 21 := by
  sorry

#check quadratic_function_uniqueness

end quadratic_function_uniqueness_l1523_152311


namespace expand_and_simplify_l1523_152303

theorem expand_and_simplify (x y : ℝ) : (x + 2*y) * (2*x - 3*y) = 2*x^2 + x*y - 6*y^2 := by
  sorry

end expand_and_simplify_l1523_152303


namespace max_value_of_sum_l1523_152334

theorem max_value_of_sum (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) :
  x + y ≤ 11 / 4 ∧ ∃ (x₀ y₀ : ℝ), 5 * x₀ + 3 * y₀ ≤ 10 ∧ 3 * x₀ + 6 * y₀ ≤ 12 ∧ x₀ + y₀ = 11 / 4 :=
by sorry

end max_value_of_sum_l1523_152334


namespace sum_of_variables_l1523_152353

/-- Given a system of equations, prove that 2x + 2y + 2z = 8 -/
theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 20 - 4*x)
  (eq2 : x + z = -10 - 4*y)
  (eq3 : x + y = 14 - 4*z) :
  2*x + 2*y + 2*z = 8 := by
  sorry

end sum_of_variables_l1523_152353


namespace linear_function_positive_sum_product_inequality_l1523_152304

-- Define the linear function
def f (k h : ℝ) (x : ℝ) : ℝ := k * x + h

-- Theorem for the first part
theorem linear_function_positive (k h m n : ℝ) (hk : k ≠ 0) (hmn : m < n) 
  (hfm : f k h m > 0) (hfn : f k h n > 0) :
  ∀ x, m < x ∧ x < n → f k h x > 0 := by sorry

-- Theorem for the second part
theorem sum_product_inequality (a b c : ℝ) 
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) :
  a * b + b * c + c * a > -1 := by sorry

end linear_function_positive_sum_product_inequality_l1523_152304


namespace simple_interest_problem_l1523_152327

theorem simple_interest_problem (simple_interest rate time : ℝ) 
  (h1 : simple_interest = 100)
  (h2 : rate = 5)
  (h3 : time = 4) :
  simple_interest = (500 * rate * time) / 100 := by
sorry

end simple_interest_problem_l1523_152327


namespace dartboard_angles_l1523_152339

theorem dartboard_angles (p₁ p₂ : ℝ) (θ₁ θ₂ : ℝ) :
  p₁ = 1/8 →
  p₂ = 2 * p₁ →
  p₁ = θ₁ / 360 →
  p₂ = θ₂ / 360 →
  θ₁ = 45 ∧ θ₂ = 90 :=
by sorry

end dartboard_angles_l1523_152339


namespace sum_of_digits_1948_base9_l1523_152308

/-- Converts a natural number from base 10 to base 9 -/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the sum of a list of natural numbers -/
def sum (l : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_1948_base9 :
  sum (toBase9 1948) = 12 :=
sorry

end sum_of_digits_1948_base9_l1523_152308


namespace largest_mersenne_prime_under_500_l1523_152306

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def mersenne_prime (p : ℕ) : Prop := is_prime p ∧ ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_under_500 :
  ∃ p : ℕ, mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, mersenne_prime q → q < 500 → q ≤ p :=
by sorry

end largest_mersenne_prime_under_500_l1523_152306


namespace inequality_proof_l1523_152384

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (ha1 : a ≥ 1) (hb1 : b ≥ 1) (hc1 : c ≥ 1)
  (habcd : a * b * c * d = 1) :
  1 / (a^2 - a + 1)^2 + 1 / (b^2 - b + 1)^2 + 
  1 / (c^2 - c + 1)^2 + 1 / (d^2 - d + 1)^2 ≤ 4 := by
sorry

end inequality_proof_l1523_152384


namespace infinite_points_in_region_l1523_152398

theorem infinite_points_in_region : 
  ∃ (S : Set (ℚ × ℚ)), 
    (∀ (p : ℚ × ℚ), p ∈ S ↔ 
      (0 < p.1 ∧ 0 < p.2) ∧ 
      (p.1^2 + p.2^2 ≤ 16) ∧ 
      (p.1 ≤ 3 ∧ p.2 ≤ 3)) ∧ 
    Set.Infinite S :=
by sorry

end infinite_points_in_region_l1523_152398


namespace pentagon_area_l1523_152360

/-- The area of a specific pentagon -/
theorem pentagon_area : 
  ∀ (pentagon_sides : List ℝ) 
    (trapezoid_bases : List ℝ) 
    (trapezoid_height : ℝ) 
    (triangle_base : ℝ) 
    (triangle_height : ℝ),
  pentagon_sides = [18, 25, 30, 28, 25] →
  trapezoid_bases = [25, 28] →
  trapezoid_height = 30 →
  triangle_base = 18 →
  triangle_height = 24 →
  (1/2 * (trapezoid_bases.sum) * trapezoid_height) + (1/2 * triangle_base * triangle_height) = 1011 := by
sorry


end pentagon_area_l1523_152360


namespace four_Z_one_equals_five_l1523_152388

-- Define the Z operation
def Z (a b : ℝ) : ℝ := a^2 - 3*a*b + b^2

-- Theorem statement
theorem four_Z_one_equals_five : Z 4 1 = 5 := by
  sorry

end four_Z_one_equals_five_l1523_152388


namespace brenda_bracelets_l1523_152367

/-- Given the number of bracelets and total number of stones, 
    calculate the number of stones per bracelet -/
def stones_per_bracelet (num_bracelets : ℕ) (total_stones : ℕ) : ℕ :=
  total_stones / num_bracelets

/-- Theorem: Given 3 bracelets and 36 total stones, 
    there will be 12 stones per bracelet -/
theorem brenda_bracelets : stones_per_bracelet 3 36 = 12 := by
  sorry

end brenda_bracelets_l1523_152367


namespace betty_oranges_l1523_152317

/-- The number of boxes Betty has -/
def num_boxes : ℕ := 3

/-- The number of oranges in each box -/
def oranges_per_box : ℕ := 8

/-- The total number of oranges Betty has -/
def total_oranges : ℕ := num_boxes * oranges_per_box

theorem betty_oranges : total_oranges = 24 := by
  sorry

end betty_oranges_l1523_152317


namespace parabola_vertex_l1523_152305

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 2)^2 - 5

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, -5)

/-- Theorem: The vertex of the parabola y = -2(x-2)^2 - 5 is at the point (2, -5) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end parabola_vertex_l1523_152305


namespace circle_triangle_area_l1523_152392

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  -- We'll assume a simple representation for this problem
  -- More complex representations might be needed for general use
  point : Point
  direction : Point

def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.x - c2.center.x) ^ 2 + (c1.center.y - c2.center.y) ^ 2 = (c1.radius + c2.radius) ^ 2

def internally_tangent_to_line (c : Circle) (l : Line) : Prop :=
  -- This is a simplification; in reality, we'd need more complex calculations
  c.center.y = l.point.y + c.radius

def externally_tangent_to_line (c : Circle) (l : Line) : Prop :=
  -- This is a simplification; in reality, we'd need more complex calculations
  c.center.y = l.point.y - c.radius

def between_points_on_line (p q r : Point) (l : Line) : Prop :=
  -- This is a simplification; in reality, we'd need more complex calculations
  p.x < q.x ∧ q.x < r.x

def triangle_area (p q r : Point) : ℝ :=
  0.5 * |p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y)|

theorem circle_triangle_area :
  ∀ (P Q R : Circle) (l : Line) (P' Q' R' : Point),
    P.radius = 1 →
    Q.radius = 3 →
    R.radius = 5 →
    internally_tangent_to_line P l →
    externally_tangent_to_line Q l →
    externally_tangent_to_line R l →
    between_points_on_line P' Q' R' l →
    externally_tangent P Q →
    externally_tangent Q R →
    triangle_area P.center Q.center R.center = 16 := by
  sorry

end circle_triangle_area_l1523_152392


namespace ball_attendance_l1523_152314

theorem ball_attendance :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by sorry

end ball_attendance_l1523_152314


namespace opposite_of_neg_one_half_l1523_152331

/-- The opposite of a rational number -/
def opposite (x : ℚ) : ℚ := -x

/-- The property that defines the opposite of a number -/
def is_opposite (x y : ℚ) : Prop := x + y = 0

theorem opposite_of_neg_one_half :
  is_opposite (-1/2 : ℚ) (1/2 : ℚ) :=
by sorry

end opposite_of_neg_one_half_l1523_152331


namespace race_start_theorem_l1523_152318

/-- Represents the start distance one runner can give another in a kilometer race -/
def start_distance (runner1 runner2 : ℕ) : ℝ := sorry

/-- The race distance in meters -/
def race_distance : ℝ := 1000

theorem race_start_theorem (A B C : ℕ) :
  start_distance A B = 50 →
  start_distance B C = 52.63157894736844 →
  start_distance A C = 100 := by
  sorry

end race_start_theorem_l1523_152318


namespace six_eight_ten_pythagorean_l1523_152362

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- Theorem: (6, 8, 10) is a Pythagorean triple -/
theorem six_eight_ten_pythagorean : is_pythagorean_triple 6 8 10 := by
  sorry

end six_eight_ten_pythagorean_l1523_152362


namespace polynomial_exists_for_non_squares_l1523_152348

-- Define the polynomial P(x,y,z)
def P (x y z : ℕ) : ℤ :=
  (1 - 2013 * (z - 1) * (z - 2)) * ((x + y - 1)^2 + 2*y - 2 + z)

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

-- State the theorem
theorem polynomial_exists_for_non_squares :
  ∀ n : ℕ, n > 0 →
    (¬ is_perfect_square n ↔ ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ P x y z = n) :=
by sorry

end polynomial_exists_for_non_squares_l1523_152348


namespace absolute_value_inequality_l1523_152336

theorem absolute_value_inequality (x : ℝ) : 
  |x + 1| > 3 ↔ x ∈ Set.Iio (-4) ∪ Set.Ioi 2 := by sorry

end absolute_value_inequality_l1523_152336


namespace expression_value_l1523_152310

theorem expression_value : 
  (11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10) = 1 := by
sorry

end expression_value_l1523_152310


namespace handshakes_eight_couples_l1523_152380

/-- Represents the number of handshakes in a group of couples with one injured person --/
def handshakes_in_couples_group (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let shaking_people := total_people - 1
  let handshakes_per_person := total_people - 3
  (shaking_people * handshakes_per_person) / 2

/-- Theorem stating that in a group of 8 married couples where everyone shakes hands
    with each other except their spouse and one person doesn't shake hands at all,
    the total number of handshakes is 90 --/
theorem handshakes_eight_couples :
  handshakes_in_couples_group 8 = 90 := by
  sorry

end handshakes_eight_couples_l1523_152380


namespace ball_color_probability_l1523_152371

/-- The number of balls -/
def n : ℕ := 8

/-- The probability of a ball being painted black or white -/
def p : ℚ := 1/2

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of exactly k successes in n independent trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial n k : ℚ) * p^k * (1 - p)^(n - k)

theorem ball_color_probability :
  binomial_probability n (n/2) p = 35/128 := by
  sorry

end ball_color_probability_l1523_152371


namespace ratio_composition_l1523_152335

theorem ratio_composition (a b c : ℚ) 
  (h1 : a / b = 11 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 11 / 15 := by
sorry

end ratio_composition_l1523_152335
