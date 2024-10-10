import Mathlib

namespace smallest_n_for_monochromatic_isosceles_trapezoid_l3218_321891

/-- A coloring of vertices with three colors -/
def Coloring (n : ℕ) := Fin n → Fin 3

/-- Check if four vertices form an isosceles trapezoid in an n-gon -/
def IsIsoscelesTrapezoid (n : ℕ) (v1 v2 v3 v4 : Fin n) : Prop := sorry

/-- Check if a coloring contains four vertices of the same color forming an isosceles trapezoid -/
def HasMonochromaticIsoscelesTrapezoid (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (v1 v2 v3 v4 : Fin n), 
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4 ∧
    c v1 = c v2 ∧ c v1 = c v3 ∧ c v1 = c v4 ∧
    IsIsoscelesTrapezoid n v1 v2 v3 v4

/-- The main theorem -/
theorem smallest_n_for_monochromatic_isosceles_trapezoid :
  (∀ (c : Coloring 17), HasMonochromaticIsoscelesTrapezoid 17 c) ∧
  (∀ (n : ℕ), n < 17 → ∃ (c : Coloring n), ¬HasMonochromaticIsoscelesTrapezoid n c) :=
sorry

end smallest_n_for_monochromatic_isosceles_trapezoid_l3218_321891


namespace rectangular_box_surface_area_l3218_321829

/-- The surface area of a rectangular box -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * h + l * w + w * h)

/-- Theorem: The surface area of a rectangular box with length l, width w, and height h
    is equal to 2(lh + lw + wh) -/
theorem rectangular_box_surface_area (l w h : ℝ) :
  surface_area l w h = 2 * (l * h + l * w + w * h) := by
  sorry

end rectangular_box_surface_area_l3218_321829


namespace cat_toy_cost_l3218_321879

theorem cat_toy_cost (total_payment change cage_cost : ℚ) 
  (h1 : total_payment = 20)
  (h2 : change = 0.26)
  (h3 : cage_cost = 10.97) :
  total_payment - change - cage_cost = 8.77 := by
  sorry

end cat_toy_cost_l3218_321879


namespace calculation_proof_l3218_321851

theorem calculation_proof : 0.54 - (1/8 : ℚ) + 0.46 - (7/8 : ℚ) = 0 := by
  sorry

end calculation_proof_l3218_321851


namespace common_roots_sum_l3218_321894

theorem common_roots_sum (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + b*x + c = 0) →
  (∃ y : ℝ, y^2 + y + a = 0 ∧ y^2 + c*y + b = 0) →
  a + b + c = -3 := by
sorry

end common_roots_sum_l3218_321894


namespace fixed_point_exponential_l3218_321830

/-- The fixed point of the function f(x) = a^(x-2) + 1 -/
theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) + 1
  f 2 = 2 := by sorry

end fixed_point_exponential_l3218_321830


namespace work_completion_time_l3218_321895

/-- Represents the number of days it takes for a worker to complete the work alone -/
structure Worker where
  days : ℝ

/-- Represents the work scenario -/
structure WorkScenario where
  a : Worker
  b : Worker
  c : Worker
  cLeaveDays : ℝ

/-- Calculates the time taken to complete the work given a work scenario -/
def completionTime (scenario : WorkScenario) : ℝ :=
  sorry

/-- The specific work scenario from the problem -/
def problemScenario : WorkScenario :=
  { a := ⟨30⟩
  , b := ⟨30⟩
  , c := ⟨40⟩
  , cLeaveDays := 4 }

/-- Theorem stating that the work is completed in approximately 15 days -/
theorem work_completion_time :
  ⌈completionTime problemScenario⌉ = 15 :=
  sorry

end work_completion_time_l3218_321895


namespace square_difference_cubed_l3218_321814

theorem square_difference_cubed : (5^2 - 4^2)^3 = 729 := by
  sorry

end square_difference_cubed_l3218_321814


namespace point_in_second_quadrant_l3218_321853

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (4 - m, 2)

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem point_in_second_quadrant (m : ℝ) :
  in_second_quadrant (P m) → m = 5 :=
by
  sorry


end point_in_second_quadrant_l3218_321853


namespace sum_areas_halving_circles_l3218_321865

/-- The sum of areas of an infinite series of circles with halving radii -/
theorem sum_areas_halving_circles (π : ℝ) (h : π > 0) : 
  let r₀ : ℝ := 2  -- radius of the first circle
  let seriesSum : ℝ := ∑' n, π * (r₀ * (1/2)^n)^2  -- sum of areas
  seriesSum = 16 * π / 3 :=
by sorry

end sum_areas_halving_circles_l3218_321865


namespace rational_difference_l3218_321803

theorem rational_difference (x y : ℚ) (h : (1 + y) / (x - y) = x) : y = x - 1 := by
  sorry

end rational_difference_l3218_321803


namespace lost_shoes_count_l3218_321815

/-- Given an initial number of shoe pairs and a remaining number of matching pairs,
    calculate the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * remaining_pairs

/-- Theorem stating that with 20 initial pairs and 15 remaining pairs,
    10 individual shoes are lost. -/
theorem lost_shoes_count : shoes_lost 20 15 = 10 := by
  sorry

end lost_shoes_count_l3218_321815


namespace alyssa_cherries_cost_l3218_321883

/-- The amount Alyssa paid for cherries -/
def cherries_cost (total_spent grapes_cost : ℚ) : ℚ :=
  total_spent - grapes_cost

/-- Proof that Alyssa paid $9.85 for cherries -/
theorem alyssa_cherries_cost :
  let total_spent : ℚ := 21.93
  let grapes_cost : ℚ := 12.08
  cherries_cost total_spent grapes_cost = 9.85 := by
  sorry

#eval cherries_cost 21.93 12.08

end alyssa_cherries_cost_l3218_321883


namespace product_1_to_30_trailing_zeros_l3218_321882

/-- The number of trailing zeros in the product of integers from 1 to n -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The product of integers from 1 to 30 has 7 trailing zeros -/
theorem product_1_to_30_trailing_zeros :
  trailingZeros 30 = 7 := by
sorry


end product_1_to_30_trailing_zeros_l3218_321882


namespace three_digit_number_problem_l3218_321848

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

theorem three_digit_number_problem (n : ThreeDigitNumber) 
  (sum_18 : n.hundreds + n.tens + n.units = 18)
  (hundreds_tens_relation : n.hundreds = n.tens + 1)
  (units_tens_relation : n.units = n.tens + 2) :
  n.toNat = 657 := by
  sorry

end three_digit_number_problem_l3218_321848


namespace product_of_sums_evaluate_specific_product_l3218_321824

theorem product_of_sums (a b : ℕ) : (a + 1) * (a^2 + 1^2) * (a^4 + 1^4) = ((a^2 - 1^2) * (a^2 + 1^2) * (a^4 - 1^4) * (a^4 + 1^4)) / (a - 1) / 2 := by
  sorry

theorem evaluate_specific_product : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end product_of_sums_evaluate_specific_product_l3218_321824


namespace valid_arrangement_exists_l3218_321804

/-- A chessboard is represented as a function from (Fin 8 × Fin 8) to Option (Fin 2),
    where Some 0 represents a white piece, Some 1 represents a black piece,
    and None represents an empty square. -/
def Chessboard := Fin 8 → Fin 8 → Option (Fin 2)

/-- Count the number of neighbors of a given color for a piece at position (i, j) -/
def countNeighbors (board : Chessboard) (i j : Fin 8) (color : Fin 2) : Nat :=
  sorry

/-- Check if a given arrangement satisfies the condition that each piece
    has an equal number of white and black neighbors -/
def isValidArrangement (board : Chessboard) : Prop :=
  sorry

/-- Count the total number of pieces of a given color on the board -/
def countPieces (board : Chessboard) (color : Fin 2) : Nat :=
  sorry

/-- The main theorem stating that a valid arrangement exists -/
theorem valid_arrangement_exists : ∃ (board : Chessboard),
  (countPieces board 0 = 16) ∧
  (countPieces board 1 = 16) ∧
  isValidArrangement board :=
sorry

end valid_arrangement_exists_l3218_321804


namespace simplify_and_evaluate_expression_l3218_321899

theorem simplify_and_evaluate_expression (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) (h3 : a ≠ -2) (h4 : a ≠ -1) (h5 : a = 1) :
  1 - (a - 2) / a / ((a^2 - 4) / (a^2 + a)) = 1 / 3 := by
  sorry

end simplify_and_evaluate_expression_l3218_321899


namespace smallest_n_divisible_l3218_321866

theorem smallest_n_divisible : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m^2 % 24 = 0 ∧ m^3 % 450 = 0 → n ≤ m) ∧
  n^2 % 24 = 0 ∧ n^3 % 450 = 0 := by
  use 60
  sorry

end smallest_n_divisible_l3218_321866


namespace ben_pea_picking_time_l3218_321884

/-- Given Ben's rate of picking sugar snap peas, calculate the time needed to pick a different amount -/
theorem ben_pea_picking_time (initial_peas initial_time target_peas : ℕ) : 
  initial_peas > 0 → initial_time > 0 → target_peas > 0 →
  (target_peas * initial_time) / initial_peas = 9 :=
by
  sorry

#check ben_pea_picking_time 56 7 72

end ben_pea_picking_time_l3218_321884


namespace exists_all_berries_l3218_321834

/-- A binary vector of length 7 -/
def BinaryVector := Fin 7 → Bool

/-- The set of 16 vectors representing the work schedule -/
def WorkSchedule := Fin 16 → BinaryVector

/-- The condition that the first vector is all zeros -/
def firstDayAllMine (schedule : WorkSchedule) : Prop :=
  ∀ i : Fin 7, schedule 0 i = false

/-- The condition that any two vectors differ in at least 3 positions -/
def atLeastThreeDifferences (schedule : WorkSchedule) : Prop :=
  ∀ d1 d2 : Fin 16, d1 ≠ d2 →
    (Finset.filter (fun i => schedule d1 i ≠ schedule d2 i) Finset.univ).card ≥ 3

/-- The theorem to be proved -/
theorem exists_all_berries (schedule : WorkSchedule)
  (h1 : firstDayAllMine schedule)
  (h2 : atLeastThreeDifferences schedule) :
  ∃ d : Fin 16, ∀ i : Fin 7, schedule d i = true := by
  sorry

end exists_all_berries_l3218_321834


namespace prism_volume_l3218_321855

-- Define a right rectangular prism
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define the volume of a rectangular prism
def volume (p : RectangularPrism) : ℝ := p.length * p.width * p.height

-- Define the areas of the faces
def faceArea1 (p : RectangularPrism) : ℝ := p.length * p.width
def faceArea2 (p : RectangularPrism) : ℝ := p.width * p.height
def faceArea3 (p : RectangularPrism) : ℝ := p.length * p.height

-- State the theorem
theorem prism_volume (p : RectangularPrism)
  (h1 : faceArea1 p = 60)
  (h2 : faceArea2 p = 72)
  (h3 : faceArea3 p = 90) :
  volume p = 4320 := by
  sorry

end prism_volume_l3218_321855


namespace trapezoid_segment_length_l3218_321854

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  area_ratio : ℝ
  sum_of_parallel_sides : ℝ
  area_ratio_condition : area_ratio = 5 / 3
  sum_condition : AB + CD = sum_of_parallel_sides

/-- Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC to the area of triangle ADC
    is 5:3, and AB + CD = 160 cm, then AB = 100 cm -/
theorem trapezoid_segment_length (t : Trapezoid) (h : t.sum_of_parallel_sides = 160) : t.AB = 100 := by
  sorry


end trapezoid_segment_length_l3218_321854


namespace acrobat_count_range_l3218_321832

/-- Represents the count of animals in the zoo --/
structure AnimalCount where
  elephants : ℕ
  monkeys : ℕ
  acrobats : ℕ

/-- Checks if the animal count satisfies the given conditions --/
def isValidCount (count : AnimalCount) : Prop :=
  count.elephants * 4 + count.monkeys * 2 + count.acrobats * 2 = 50 ∧
  count.elephants + count.monkeys + count.acrobats = 18

/-- The main theorem stating the range of possible acrobat counts --/
theorem acrobat_count_range :
  ∀ n : ℕ, 0 ≤ n ∧ n ≤ 11 →
  ∃ (count : AnimalCount), isValidCount count ∧ count.acrobats = n :=
by sorry

end acrobat_count_range_l3218_321832


namespace sum_digits_first_1998_even_l3218_321874

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The sum of digits used to write all even integers from 2 to n -/
def sum_digits_even (n : ℕ) : ℕ := sorry

/-- The 1998th positive even integer -/
def n_1998 : ℕ := 3996

theorem sum_digits_first_1998_even : sum_digits_even n_1998 = 7440 := by sorry

end sum_digits_first_1998_even_l3218_321874


namespace prob_same_color_is_89_169_l3218_321836

def num_blue_balls : ℕ := 8
def num_yellow_balls : ℕ := 5
def total_balls : ℕ := num_blue_balls + num_yellow_balls

def prob_same_color : ℚ :=
  (num_blue_balls / total_balls) ^ 2 + (num_yellow_balls / total_balls) ^ 2

theorem prob_same_color_is_89_169 :
  prob_same_color = 89 / 169 := by
  sorry

end prob_same_color_is_89_169_l3218_321836


namespace cloth_sold_meters_l3218_321831

/-- Proves that the number of meters of cloth sold is 80 -/
theorem cloth_sold_meters (total_selling_price : ℝ) (profit_per_meter : ℝ) (cost_price_per_meter : ℝ)
  (h1 : total_selling_price = 6900)
  (h2 : profit_per_meter = 20)
  (h3 : cost_price_per_meter = 66.25) :
  (total_selling_price / (cost_price_per_meter + profit_per_meter)) = 80 := by
  sorry

end cloth_sold_meters_l3218_321831


namespace rhombus_side_length_l3218_321858

/-- A rhombus with a perimeter of 60 centimeters has a side length of 15 centimeters. -/
theorem rhombus_side_length (perimeter : ℝ) (h1 : perimeter = 60) : 
  perimeter / 4 = 15 := by
  sorry

end rhombus_side_length_l3218_321858


namespace sequence_is_cubic_polynomial_l3218_321889

def fourth_difference (u : ℕ → ℝ) : ℕ → ℝ :=
  λ n => u (n + 4) - 4 * u (n + 3) + 6 * u (n + 2) - 4 * u (n + 1) + u n

theorem sequence_is_cubic_polynomial 
  (u : ℕ → ℝ) 
  (h : ∀ n, fourth_difference u n = 0) : 
  ∃ a b c d : ℝ, ∀ n, u n = a * n^3 + b * n^2 + c * n + d :=
sorry

end sequence_is_cubic_polynomial_l3218_321889


namespace prob_same_student_given_same_look_l3218_321867

/-- Represents a group of identical students -/
structure IdenticalGroup where
  size : Nat
  count : Nat

/-- Represents the Multiples Obfuscation Program -/
def MultiplesObfuscationProgram : List IdenticalGroup :=
  [⟨1, 1⟩, ⟨2, 1⟩, ⟨3, 1⟩, ⟨4, 1⟩, ⟨5, 1⟩, ⟨6, 1⟩, ⟨7, 1⟩, ⟨8, 1⟩]

/-- Total number of students in the program -/
def totalStudents : Nat :=
  MultiplesObfuscationProgram.foldr (fun g acc => g.size * g.count + acc) 0

/-- Number of pairs where students look the same -/
def sameLookPairs : Nat :=
  MultiplesObfuscationProgram.foldr (fun g acc => g.size * g.size * g.count + acc) 0

/-- Probability of encountering the same student twice -/
def probSameStudent : Rat :=
  totalStudents / (totalStudents * totalStudents)

/-- Probability of encountering students that look the same -/
def probSameLook : Rat :=
  sameLookPairs / (totalStudents * totalStudents)

theorem prob_same_student_given_same_look :
  probSameStudent / probSameLook = 3 / 17 := by sorry

end prob_same_student_given_same_look_l3218_321867


namespace subtraction_preserves_inequality_l3218_321826

theorem subtraction_preserves_inequality (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end subtraction_preserves_inequality_l3218_321826


namespace bouncy_ball_difference_l3218_321880

/-- Proves that the difference between red and yellow bouncy balls is 18 -/
theorem bouncy_ball_difference :
  ∀ (red_packs yellow_packs balls_per_pack : ℕ),
  red_packs = 5 →
  yellow_packs = 4 →
  balls_per_pack = 18 →
  red_packs * balls_per_pack - yellow_packs * balls_per_pack = 18 :=
by
  sorry

#check bouncy_ball_difference

end bouncy_ball_difference_l3218_321880


namespace solution_implies_sum_l3218_321869

/-- The function f(x) = |x+1| + |x-3| -/
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

/-- The function g(x) = a - |x-2| -/
def g (a x : ℝ) : ℝ := a - |x - 2|

/-- The theorem stating that if the solution set of f(x) < g(x) is (b, 7/2), then a + b = 6 -/
theorem solution_implies_sum (a b : ℝ) :
  (∀ x, f x < g a x ↔ b < x ∧ x < 7/2) →
  a + b = 6 := by
  sorry

end solution_implies_sum_l3218_321869


namespace square_condition_l3218_321845

def a_n (n : ℕ+) : ℕ := (10^n.val - 1) / 9

theorem square_condition (n : ℕ+) (b : ℕ) : 
  0 < b ∧ b < 10 →
  (∃ k : ℕ, a_n (2*n) - b * a_n n = k^2) ↔ (b = 2 ∨ (b = 7 ∧ n = 1)) :=
by sorry

end square_condition_l3218_321845


namespace intersection_of_P_and_Q_l3218_321820

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}
def Q : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 < x ∧ x ≤ 3} := by sorry

end intersection_of_P_and_Q_l3218_321820


namespace difference_of_squares_301_297_l3218_321801

theorem difference_of_squares_301_297 : 301^2 - 297^2 = 2392 := by
  sorry

end difference_of_squares_301_297_l3218_321801


namespace second_green_probability_l3218_321841

-- Define the contents of each bag
def bag1 : Finset ℕ := {0, 0, 0, 1}  -- 0 represents green, 1 represents red
def bag2 : Finset ℕ := {0, 0, 1, 1}
def bag3 : Finset ℕ := {0, 1, 1, 1}

-- Define the probability of selecting each bag
def bagProb : ℕ → ℚ
  | 1 => 1/3
  | 2 => 1/3
  | 3 => 1/3
  | _ => 0

-- Define the probability of selecting a green candy from a bag
def greenProb : Finset ℕ → ℚ
  | s => (s.filter (· = 0)).card / s.card

-- Define the probability of selecting a red candy from a bag
def redProb : Finset ℕ → ℚ
  | s => (s.filter (· = 1)).card / s.card

-- Define the probability of selecting a green candy as the second candy
def secondGreenProb : ℚ := sorry

theorem second_green_probability : secondGreenProb = 73/144 := by sorry

end second_green_probability_l3218_321841


namespace max_value_on_unit_circle_l3218_321877

/-- The maximum value of f(z) = |z^3 - z + 2| on the unit circle -/
theorem max_value_on_unit_circle :
  ∃ (M : ℝ), M = Real.sqrt 13 ∧
  (∀ z : ℂ, Complex.abs z = 1 →
    Complex.abs (z^3 - z + 2) ≤ M) ∧
  (∃ z : ℂ, Complex.abs z = 1 ∧
    Complex.abs (z^3 - z + 2) = M) := by
  sorry

end max_value_on_unit_circle_l3218_321877


namespace probability_sum_20_l3218_321878

def total_balls : ℕ := 5
def balls_labeled_5 : ℕ := 3
def balls_labeled_10 : ℕ := 2
def balls_drawn : ℕ := 3
def target_sum : ℕ := 20

theorem probability_sum_20 : 
  (Nat.choose balls_labeled_5 2 * Nat.choose balls_labeled_10 1) / 
  Nat.choose total_balls balls_drawn = 3 / 5 := by sorry

end probability_sum_20_l3218_321878


namespace average_permutation_sum_l3218_321842

def permutation_sum (b : Fin 8 → Fin 8) : ℕ :=
  |b 0 - b 1| + |b 2 - b 3| + |b 4 - b 5| + |b 6 - b 7|

def all_permutations : Finset (Fin 8 → Fin 8) :=
  Finset.univ.filter (fun f => Function.Injective f)

theorem average_permutation_sum :
  (Finset.sum all_permutations permutation_sum) / all_permutations.card = 12 := by
  sorry

end average_permutation_sum_l3218_321842


namespace rod_mass_is_one_fourth_l3218_321897

/-- The linear density function of the rod -/
def ρ : ℝ → ℝ := fun x ↦ x^3

/-- The length of the rod -/
def rod_length : ℝ := 1

/-- The mass of the rod -/
noncomputable def rod_mass : ℝ := ∫ x in (0)..(rod_length), ρ x

/-- Theorem: The mass of the rod is equal to 1/4 -/
theorem rod_mass_is_one_fourth : rod_mass = 1/4 := by
  sorry

end rod_mass_is_one_fourth_l3218_321897


namespace factor_condition_l3218_321852

theorem factor_condition (a b c m l : ℝ) : 
  ((b + c) * (c + a) * (a + b) + a * b * c = 
   (m * (a^2 + b^2 + c^2) + l * (a * b + a * c + b * c)) * k) →
  (m = 0 ∧ l = a + b + c) :=
by sorry

end factor_condition_l3218_321852


namespace sum_of_reciprocals_squared_l3218_321861

/-- Given the definitions of p, q, r, and s, prove that (1/p + 1/q + 1/r + 1/s)² = 560/151321 -/
theorem sum_of_reciprocals_squared (p q r s : ℝ) 
  (hp : p = Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35)
  (hq : q = -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35)
  (hr : r = Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35)
  (hs : s = -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35) :
  (1/p + 1/q + 1/r + 1/s)^2 = 560/151321 := by
  sorry

end sum_of_reciprocals_squared_l3218_321861


namespace angle_with_same_terminal_side_as_negative_35_l3218_321846

-- Define a function that represents angles with the same terminal side as a given angle
def sameTerminalSide (angle : ℝ) : ℤ → ℝ := fun k => k * 360 + angle

-- Theorem statement
theorem angle_with_same_terminal_side_as_negative_35 :
  ∃ (x : ℝ), 0 ≤ x ∧ x < 360 ∧ ∃ (k : ℤ), x = sameTerminalSide (-35) k ∧ x = 325 := by
  sorry

end angle_with_same_terminal_side_as_negative_35_l3218_321846


namespace lunch_group_probability_l3218_321890

theorem lunch_group_probability (total_students : ℕ) (num_groups : ℕ) (friends : ℕ) 
  (h1 : total_students = 800)
  (h2 : num_groups = 4)
  (h3 : friends = 4)
  (h4 : total_students % num_groups = 0) :
  (1 : ℚ) / (num_groups ^ (friends - 1)) = 1 / 64 :=
sorry

end lunch_group_probability_l3218_321890


namespace problem_solution_l3218_321837

theorem problem_solution :
  (∀ a : ℝ, 2*a + 3*a - 4*a = a) ∧
  (-1^2022 + 27/4 * (-1/3 - 1) / (-3)^2 + |-1| = -1) :=
by sorry

end problem_solution_l3218_321837


namespace exists_multicolor_triangle_l3218_321887

/-- Represents the three possible colors for vertices -/
inductive Color
| Red
| Blue
| Yellow

/-- Represents a vertex in the triangle -/
structure Vertex where
  x : ℝ
  y : ℝ
  color : Color

/-- Represents a small equilateral triangle -/
structure SmallTriangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

/-- Represents the large equilateral triangle ABC -/
structure LargeTriangle where
  n : ℕ
  smallTriangles : Array SmallTriangle

/-- Predicate to check if a vertex is on side BC -/
def onSideBC (v : Vertex) : Prop := sorry

/-- Predicate to check if a vertex is on side CA -/
def onSideCA (v : Vertex) : Prop := sorry

/-- Predicate to check if a vertex is on side AB -/
def onSideAB (v : Vertex) : Prop := sorry

/-- The main theorem to be proved -/
theorem exists_multicolor_triangle (ABC : LargeTriangle) : 
  (∀ v : Vertex, onSideBC v → v.color ≠ Color.Red) →
  (∀ v : Vertex, onSideCA v → v.color ≠ Color.Blue) →
  (∀ v : Vertex, onSideAB v → v.color ≠ Color.Yellow) →
  ∃ t : SmallTriangle, t ∈ ABC.smallTriangles ∧ 
    t.v1.color ≠ t.v2.color ∧ t.v2.color ≠ t.v3.color ∧ t.v1.color ≠ t.v3.color :=
sorry

end exists_multicolor_triangle_l3218_321887


namespace students_not_playing_l3218_321818

theorem students_not_playing (total : ℕ) (basketball : ℕ) (volleyball : ℕ) (both : ℕ) : 
  total = 20 ∧ 
  basketball = total / 2 ∧ 
  volleyball = total * 2 / 5 ∧ 
  both = total / 10 → 
  total - (basketball + volleyball - both) = 4 := by
sorry

end students_not_playing_l3218_321818


namespace max_product_value_l3218_321810

-- Define the functions h and k on ℝ
variable (h k : ℝ → ℝ)

-- Define the ranges of h and k
variable (h_range : Set.range h = Set.Icc (-3) 5)
variable (k_range : Set.range k = Set.Icc (-1) 3)

-- Theorem statement
theorem max_product_value :
  ∃ (x : ℝ), h x * k x = 15 ∧ ∀ (y : ℝ), h y * k y ≤ 15 := by
  sorry

end max_product_value_l3218_321810


namespace museum_ticket_cost_class_trip_cost_l3218_321856

/-- Calculates the total cost of museum tickets for a class, including a group discount -/
theorem museum_ticket_cost (num_students num_teachers : ℕ) 
  (student_price teacher_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_people := num_students + num_teachers
  let regular_cost := num_students * student_price + num_teachers * teacher_price
  let discount := if total_people ≥ 25 then discount_rate * regular_cost else 0
  regular_cost - discount

/-- Proves that the total cost for the class trip is $230.40 -/
theorem class_trip_cost : 
  museum_ticket_cost 30 4 8 12 (20/100) = 230.4 := by
  sorry

end museum_ticket_cost_class_trip_cost_l3218_321856


namespace tan_squared_fixed_point_l3218_321857

noncomputable def f (x : ℝ) : ℝ := 1 / ((x + 1) / x)

theorem tan_squared_fixed_point (t : ℝ) (h : 0 ≤ t ∧ t ≤ π / 2) :
  f (Real.tan t ^ 2) = Real.tan t ^ 2 := by
  sorry

end tan_squared_fixed_point_l3218_321857


namespace three_digit_number_divisible_by_45_l3218_321821

/-- Reverses a three-digit number -/
def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem three_digit_number_divisible_by_45 (n : ℕ) :
  is_three_digit n →
  n % 45 = 0 →
  n - reverse_number n = 297 →
  n = 360 ∨ n = 855 := by
sorry

end three_digit_number_divisible_by_45_l3218_321821


namespace triangle_angle_measure_l3218_321871

/-- Given a triangle ABC with sides a = 7, b = 5, and c = 3, 
    the measure of angle A is 120 degrees. -/
theorem triangle_angle_measure (A B C : EuclideanSpace ℝ (Fin 2)) :
  let a := ‖B - C‖
  let b := ‖A - C‖
  let c := ‖A - B‖
  a = 7 ∧ b = 5 ∧ c = 3 →
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) * (180 / Real.pi) = 120 := by
  sorry


end triangle_angle_measure_l3218_321871


namespace expression_evaluation_l3218_321827

theorem expression_evaluation : 2 + 3 * 4 - 5 * 6 + 7 = -9 := by
  sorry

end expression_evaluation_l3218_321827


namespace expected_adjacent_red_pairs_l3218_321850

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCardCount : ℕ := 26

/-- The probability that a card adjacent to a red card is also red -/
def probAdjacentRed : ℚ := 25 / 51

/-- The expected number of pairs of adjacent red cards in a standard 52-card deck
    dealt in a circle -/
theorem expected_adjacent_red_pairs :
  (redCardCount : ℚ) * probAdjacentRed = 650 / 51 := by sorry

end expected_adjacent_red_pairs_l3218_321850


namespace prob_white_then_yellow_is_two_thirds_l3218_321833

/-- The probability of drawing a white ball first, followed by a yellow ball, 
    from a bag containing 6 yellow and 4 white ping pong balls, 
    when drawing two balls without replacement. -/
def prob_white_then_yellow : ℚ :=
  let total_balls : ℕ := 10
  let yellow_balls : ℕ := 6
  let white_balls : ℕ := 4
  let prob_white_first : ℚ := white_balls / total_balls
  let prob_yellow_second : ℚ := yellow_balls / (total_balls - 1)
  prob_white_first * prob_yellow_second

theorem prob_white_then_yellow_is_two_thirds :
  prob_white_then_yellow = 2/3 := by
  sorry

end prob_white_then_yellow_is_two_thirds_l3218_321833


namespace special_function_ratio_bounds_l3218_321892

open Real

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioi 0
  pos : ∀ x ∈ domain, f x > 0
  deriv_bound : ∀ x ∈ domain, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x

/-- The main theorem -/
theorem special_function_ratio_bounds (sf : SpecialFunction) :
    1/8 < sf.f 1 / sf.f 2 ∧ sf.f 1 / sf.f 2 < 1/4 := by
  sorry

end special_function_ratio_bounds_l3218_321892


namespace second_train_length_correct_l3218_321862

/-- The length of the second train given the conditions of the problem -/
def second_train_length : ℝ := 119.98240140788738

/-- The speed of the first train in km/h -/
def first_train_speed : ℝ := 42

/-- The speed of the second train in km/h -/
def second_train_speed : ℝ := 30

/-- The length of the first train in meters -/
def first_train_length : ℝ := 100

/-- The time taken for the trains to clear each other in seconds -/
def clearing_time : ℝ := 10.999120070394369

/-- Theorem stating that the calculated length of the second train is correct given the problem conditions -/
theorem second_train_length_correct :
  second_train_length = 
    (first_train_speed + second_train_speed) * (1000 / 3600) * clearing_time - first_train_length :=
by
  sorry


end second_train_length_correct_l3218_321862


namespace waiter_tips_ratio_l3218_321847

theorem waiter_tips_ratio (salary tips : ℝ) 
  (h : tips / (salary + tips) = 0.7142857142857143) : 
  tips / salary = 2.5 := by
  sorry

end waiter_tips_ratio_l3218_321847


namespace complex_modulus_l3218_321802

theorem complex_modulus (z : ℂ) : (1 + Complex.I * Real.sqrt 3) * z = 1 + Complex.I →
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_l3218_321802


namespace f_13_equals_neg_2_l3218_321881

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_13_equals_neg_2 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_period : has_period f 4) 
  (h_f_neg_1 : f (-1) = 2) : 
  f 13 = -2 := by
sorry

end f_13_equals_neg_2_l3218_321881


namespace quadratic_properties_quadratic_max_conditions_l3218_321885

-- Define the quadratic function
def quadratic_function (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- Theorem for part 1
theorem quadratic_properties :
  let f := quadratic_function 4 3
  ∃ (vertex_x vertex_y : ℝ),
    (∀ x, f x ≤ f vertex_x) ∧
    vertex_x = 2 ∧
    vertex_y = 7 ∧
    (∀ x, -1 ≤ x ∧ x ≤ 3 → -2 ≤ f x ∧ f x ≤ 7) :=
sorry

-- Theorem for part 2
theorem quadratic_max_conditions :
  ∃ (b c : ℝ),
    (∀ x ≤ 0, quadratic_function b c x ≤ 2) ∧
    (∀ x > 0, quadratic_function b c x ≤ 3) ∧
    (∃ x ≤ 0, quadratic_function b c x = 2) ∧
    (∃ x > 0, quadratic_function b c x = 3) ∧
    b = 2 ∧
    c = 2 :=
sorry

end quadratic_properties_quadratic_max_conditions_l3218_321885


namespace birds_on_fence_l3218_321898

/-- Given that there are initially 12 birds on a fence and after more birds land
    there are a total of 20 birds, prove that 8 birds landed on the fence. -/
theorem birds_on_fence (initial_birds : ℕ) (total_birds : ℕ) (h1 : initial_birds = 12) (h2 : total_birds = 20) :
  total_birds - initial_birds = 8 := by
  sorry

end birds_on_fence_l3218_321898


namespace candy_calculation_l3218_321868

/-- 
Given the initial amount of candy, the amount eaten, and the amount received,
prove that the final amount of candy is equal to the initial amount minus
the eaten amount plus the received amount.
-/
theorem candy_calculation (initial eaten received : ℕ) :
  initial - eaten + received = (initial - eaten) + received := by
  sorry

end candy_calculation_l3218_321868


namespace unique_k_divisibility_l3218_321839

theorem unique_k_divisibility (a b l : ℕ) (ha : a > 1) (hb : b > 1) 
  (hodd_a : Odd a) (hodd_b : Odd b) (hsum : a + b = 2^l) :
  ∀ k : ℕ, k > 0 → (k^2 ∣ a^k + b^k) → k = 1 := by
  sorry

end unique_k_divisibility_l3218_321839


namespace complex_modulus_equal_parts_l3218_321822

theorem complex_modulus_equal_parts (b : ℝ) :
  let z : ℂ := (3 - b * Complex.I) / Complex.I
  (z.re = z.im) → Complex.abs z = 3 * Real.sqrt 2 := by
sorry

end complex_modulus_equal_parts_l3218_321822


namespace expansion_gameplay_hours_l3218_321859

/-- Calculates the hours of gameplay added by an expansion given the total gameplay hours,
    percentage of boring gameplay, and total enjoyable gameplay hours. -/
theorem expansion_gameplay_hours
  (total_hours : ℝ)
  (boring_percentage : ℝ)
  (total_enjoyable_hours : ℝ)
  (h1 : total_hours = 100)
  (h2 : boring_percentage = 0.8)
  (h3 : total_enjoyable_hours = 50) :
  total_enjoyable_hours - (1 - boring_percentage) * total_hours = 30 :=
by sorry

end expansion_gameplay_hours_l3218_321859


namespace largest_red_socks_proof_l3218_321817

/-- The largest number of red socks satisfying the given conditions -/
def largest_red_socks : ℕ := 1164

/-- The total number of socks -/
def total_socks : ℕ := 1936

/-- Probability of selecting two socks of the same color -/
def same_color_prob : ℚ := 3/5

theorem largest_red_socks_proof :
  (total_socks ≤ 2500) ∧
  (largest_red_socks > (total_socks - largest_red_socks)) ∧
  (largest_red_socks * (largest_red_socks - 1) + 
   (total_socks - largest_red_socks) * (total_socks - largest_red_socks - 1)) / 
   (total_socks * (total_socks - 1)) = same_color_prob ∧
  (∀ r : ℕ, r > largest_red_socks → 
    (r ≤ total_socks ∧ r > (total_socks - r) ∧
     (r * (r - 1) + (total_socks - r) * (total_socks - r - 1)) / 
     (total_socks * (total_socks - 1)) = same_color_prob) → false) :=
by sorry

end largest_red_socks_proof_l3218_321817


namespace rectangle_placement_l3218_321819

theorem rectangle_placement (a b c d : ℝ) 
  (h1 : a < c) (h2 : c ≤ d) (h3 : d < b) (h4 : a * b < c * d) :
  (∃ (α : ℝ), a * (Real.cos α) + b * (Real.sin α) ≤ c ∧ 
              a * (Real.sin α) + b * (Real.cos α) ≤ d) ↔ 
  (b^2 - a^2)^2 ≤ (b*d - a*c)^2 + (b*c - a*d)^2 := by sorry

end rectangle_placement_l3218_321819


namespace water_in_bucket_l3218_321806

/-- 
Given a bucket with an initial amount of water and an additional amount added,
calculate the total amount of water in the bucket.
-/
theorem water_in_bucket (initial : ℝ) (added : ℝ) :
  initial = 3 → added = 6.8 → initial + added = 9.8 := by
  sorry

end water_in_bucket_l3218_321806


namespace isosceles_trapezoid_theorem_l3218_321805

/-- Represents an isosceles trapezoid with inscribed and circumscribed circles. -/
structure IsoscelesTrapezoid where
  r : ℝ  -- radius of inscribed circle
  R : ℝ  -- radius of circumscribed circle
  k : ℝ  -- ratio of R to r
  h_k_def : k = R / r
  h_k_pos : k > 0

/-- The angles and permissible k values for an isosceles trapezoid. -/
def trapezoid_properties (t : IsoscelesTrapezoid) : Prop :=
  let angle := Real.arcsin (1 / t.k * Real.sqrt ((1 + Real.sqrt (1 + 4 * t.k ^ 2)) / 2))
  (∀ θ, θ = angle ∨ θ = Real.pi - angle → 
    θ.cos * t.r = t.r ∧ θ.sin * t.R = t.R / 2) ∧ 
  t.k > Real.sqrt 2

/-- Main theorem about isosceles trapezoid properties. -/
theorem isosceles_trapezoid_theorem (t : IsoscelesTrapezoid) : 
  trapezoid_properties t := by sorry

end isosceles_trapezoid_theorem_l3218_321805


namespace age_determination_l3218_321863

/-- Represents a triple of positive integers -/
structure AgeTriple where
  a : Nat
  b : Nat
  c : Nat
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

/-- The product of the three ages is 2450 -/
def product_is_2450 (t : AgeTriple) : Prop :=
  t.a * t.b * t.c = 2450

/-- The sum of the three ages is even -/
def sum_is_even (t : AgeTriple) : Prop :=
  ∃ k : Nat, t.a + t.b + t.c = 2 * k

/-- The smallest age is unique -/
def smallest_is_unique (t : AgeTriple) : Prop :=
  (t.a < t.b ∧ t.a < t.c) ∨ (t.b < t.a ∧ t.b < t.c) ∨ (t.c < t.a ∧ t.c < t.b)

theorem age_determination :
  ∃! (t1 t2 : AgeTriple),
    product_is_2450 t1 ∧
    product_is_2450 t2 ∧
    sum_is_even t1 ∧
    sum_is_even t2 ∧
    t1 ≠ t2 ∧
    (∀ t : AgeTriple, product_is_2450 t ∧ sum_is_even t → t = t1 ∨ t = t2) ∧
    ∃! (t : AgeTriple),
      product_is_2450 t ∧
      sum_is_even t ∧
      smallest_is_unique t ∧
      (t = t1 ∨ t = t2) :=
by
  sorry

#check age_determination

end age_determination_l3218_321863


namespace distance_to_origin_of_complex_fraction_l3218_321849

theorem distance_to_origin_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end distance_to_origin_of_complex_fraction_l3218_321849


namespace b_age_is_eighteen_l3218_321873

/-- Given three people a, b, and c, where:
    - a is two years older than b
    - b is twice as old as c
    - The total of their ages is 47
    Prove that b is 18 years old. -/
theorem b_age_is_eighteen (a b c : ℕ) 
    (h1 : a = b + 2) 
    (h2 : b = 2 * c) 
    (h3 : a + b + c = 47) : 
  b = 18 := by
  sorry

end b_age_is_eighteen_l3218_321873


namespace first_term_of_arithmetic_sequence_l3218_321843

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_incr : ∀ n : ℕ, a n < a (n + 1))
  (h_sum : a 1 + a 2 + a 3 = 12)
  (h_prod : a 1 * a 2 * a 3 = 48) :
  a 1 = 2 := by
sorry

end first_term_of_arithmetic_sequence_l3218_321843


namespace circle_center_l3218_321823

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 15 = 0

/-- The center of a circle given by its coordinates -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem stating that the center of the circle with the given equation is (3, -1) -/
theorem circle_center : 
  ∃ (center : CircleCenter), center.x = 3 ∧ center.y = -1 ∧
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - center.x)^2 + (y - center.y)^2 = 25 :=
sorry

end circle_center_l3218_321823


namespace gcd_of_three_numbers_l3218_321875

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end gcd_of_three_numbers_l3218_321875


namespace coefficient_x5_in_expansion_l3218_321816

theorem coefficient_x5_in_expansion :
  let n : ℕ := 36
  let k : ℕ := 5
  let coeff : ℤ := (n.choose k) * (-2 : ℤ) ^ (n - k)
  coeff = -8105545721856 :=
by sorry

end coefficient_x5_in_expansion_l3218_321816


namespace x_cube_plus_reciprocal_l3218_321811

theorem x_cube_plus_reciprocal (φ : Real) (x : Real) 
  (h1 : 0 < φ) (h2 : φ < π) (h3 : x + 1/x = 2 * Real.cos (2 * φ)) : 
  x^3 + 1/x^3 = 2 * Real.cos (6 * φ) := by
  sorry

end x_cube_plus_reciprocal_l3218_321811


namespace dollar_hash_composition_l3218_321844

def dollar (N : ℝ) : ℝ := 2 * (N + 1)

def hash (N : ℝ) : ℝ := 0.5 * N + 1

theorem dollar_hash_composition : hash (dollar (dollar (dollar 5))) = 28 := by
  sorry

end dollar_hash_composition_l3218_321844


namespace probability_square_or_triangle_l3218_321872

-- Define the total number of figures
def total_figures : ℕ := 10

-- Define the number of triangles
def num_triangles : ℕ := 3

-- Define the number of squares
def num_squares : ℕ := 4

-- Define the number of circles
def num_circles : ℕ := 3

-- Theorem statement
theorem probability_square_or_triangle :
  (num_triangles + num_squares : ℚ) / total_figures = 4 / 5 := by
  sorry

end probability_square_or_triangle_l3218_321872


namespace license_plate_count_l3218_321800

/-- The number of possible letters in each letter position of the license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in each digit position of the license plate. -/
def num_digits : ℕ := 10

/-- The number of letter positions in the license plate. -/
def num_letter_positions : ℕ := 3

/-- The number of digit positions in the license plate. -/
def num_digit_positions : ℕ := 4

/-- The total number of possible license plates in Eldorado. -/
def total_license_plates : ℕ := num_letters ^ num_letter_positions * num_digits ^ num_digit_positions

theorem license_plate_count :
  total_license_plates = 175760000 :=
by sorry

end license_plate_count_l3218_321800


namespace solution_set_of_trigonometric_system_l3218_321896

theorem solution_set_of_trigonometric_system :
  let S := {(x, y) | 
    2 * (Real.cos x)^2 + 2 * Real.sqrt 2 * Real.cos x * (Real.cos (4*x))^2 + (Real.cos (4*x))^2 = 0 ∧
    Real.sin x = Real.cos y}
  S = {(x, y) | 
    (∃ k n : ℤ, x = 3 * Real.pi / 4 + 2 * Real.pi * ↑k ∧ (y = Real.pi / 4 + 2 * Real.pi * ↑n ∨ y = -Real.pi / 4 + 2 * Real.pi * ↑n)) ∨
    (∃ k n : ℤ, x = -3 * Real.pi / 4 + 2 * Real.pi * ↑k ∧ (y = 3 * Real.pi / 4 + 2 * Real.pi * ↑n ∨ y = -3 * Real.pi / 4 + 2 * Real.pi * ↑n))} :=
by sorry

end solution_set_of_trigonometric_system_l3218_321896


namespace mean_of_remaining_numbers_l3218_321828

def numbers : List ℕ := [1867, 1993, 2019, 2025, 2109, 2121]

theorem mean_of_remaining_numbers :
  ∀ (four_nums : List ℕ),
    four_nums.length = 4 →
    four_nums.all (· ∈ numbers) →
    (four_nums.sum : ℚ) / 4 = 2008 →
    let remaining_nums := numbers.filter (· ∉ four_nums)
    (remaining_nums.sum : ℚ) / 2 = 2051 := by
  sorry

end mean_of_remaining_numbers_l3218_321828


namespace track_circumference_jogging_track_circumference_l3218_321808

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (speed1 speed2 : ℝ) (meeting_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let time_in_hours := meeting_time / 60
  let circumference := relative_speed * time_in_hours
  circumference

/-- The actual problem statement -/
theorem jogging_track_circumference : 
  ∃ (c : ℝ), abs (c - track_circumference 20 17 37) < 0.0001 :=
sorry

end track_circumference_jogging_track_circumference_l3218_321808


namespace tangent_identities_l3218_321840

theorem tangent_identities :
  (∃ (f : ℝ → ℝ), (∀ x, f x = Real.tan x) ∧
    (f (π / 7) * f (2 * π / 7) * f (3 * π / 7) = Real.sqrt 7) ∧
    (f (π / 7)^2 + f (2 * π / 7)^2 + f (3 * π / 7)^2 = 21)) :=
by
  sorry

end tangent_identities_l3218_321840


namespace perfume_fundraising_l3218_321870

/-- The amount of additional money needed to buy a perfume --/
def additional_money_needed (perfume_cost initial_christian initial_sue yards_mowed yard_price dogs_walked dog_price : ℚ) : ℚ :=
  perfume_cost - (initial_christian + initial_sue + yards_mowed * yard_price + dogs_walked * dog_price)

/-- Theorem stating the additional money needed is $6.00 --/
theorem perfume_fundraising :
  additional_money_needed 50 5 7 4 5 6 2 = 6 := by
  sorry

end perfume_fundraising_l3218_321870


namespace scientific_notation_103000000_l3218_321888

theorem scientific_notation_103000000 : ∃ (a : ℝ) (n : ℤ), 
  1 ≤ a ∧ a < 10 ∧ 103000000 = a * (10 : ℝ) ^ n ∧ a = 1.03 ∧ n = 8 := by
  sorry

end scientific_notation_103000000_l3218_321888


namespace min_real_roots_l3218_321860

/-- A polynomial of degree 10 with real coefficients -/
structure Polynomial10 where
  coeffs : Fin 11 → ℝ
  lead_coeff_nonzero : coeffs 10 ≠ 0

/-- The roots of a polynomial -/
def roots (p : Polynomial10) : Multiset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinct_abs_values (p : Polynomial10) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def num_real_roots (p : Polynomial10) : ℕ := sorry

/-- If a polynomial of degree 10 with real coefficients has exactly 6 distinct absolute values
    among its roots, then it has at least 3 real roots -/
theorem min_real_roots (p : Polynomial10) :
  distinct_abs_values p = 6 → num_real_roots p ≥ 3 := by sorry

end min_real_roots_l3218_321860


namespace option_a_correct_option_b_correct_option_c_correct_option_d_incorrect_l3218_321809

-- Define variables
variable (a b c : ℝ)

-- Theorem for Option A
theorem option_a_correct : a = b → a + 6 = b + 6 := by sorry

-- Theorem for Option B
theorem option_b_correct : a = b → a / 9 = b / 9 := by sorry

-- Theorem for Option C
theorem option_c_correct (h : c ≠ 0) : a / c = b / c → a = b := by sorry

-- Theorem for Option D (incorrect transformation)
theorem option_d_incorrect : ∃ a b : ℝ, -2 * a = -2 * b ∧ a ≠ -b := by sorry

end option_a_correct_option_b_correct_option_c_correct_option_d_incorrect_l3218_321809


namespace second_question_percentage_l3218_321886

theorem second_question_percentage 
  (first_correct : ℝ) 
  (neither_correct : ℝ) 
  (both_correct : ℝ) 
  (h1 : first_correct = 63) 
  (h2 : neither_correct = 20) 
  (h3 : both_correct = 33) : 
  ∃ second_correct : ℝ, 
    second_correct = 50 ∧ 
    first_correct + second_correct - both_correct = 100 - neither_correct :=
by sorry

end second_question_percentage_l3218_321886


namespace cube_of_cube_root_fourth_smallest_prime_l3218_321864

-- Define the fourth smallest prime number
def fourth_smallest_prime : ℕ := 7

-- State the theorem
theorem cube_of_cube_root_fourth_smallest_prime :
  (fourth_smallest_prime : ℝ) = ((fourth_smallest_prime : ℝ) ^ (1/3 : ℝ)) ^ 3 :=
sorry

end cube_of_cube_root_fourth_smallest_prime_l3218_321864


namespace distribute_five_four_l3218_321893

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of partitions of n into at most k parts -/
def partitions (n k : ℕ) : ℕ := sorry

theorem distribute_five_four : distribute 5 4 = 6 := by sorry

end distribute_five_four_l3218_321893


namespace angle_ratio_l3218_321838

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- BP and BQ trisect ∠ABC
axiom trisect : angle A B P = angle B P Q ∧ angle B P Q = angle P B Q

-- BM bisects ∠ABP
axiom bisect : angle A B M = (1/2) * angle A B P

-- Theorem statement
theorem angle_ratio : 
  (angle M B Q) / (angle A B Q) = 3/4 := by sorry

end angle_ratio_l3218_321838


namespace find_b_value_l3218_321813

theorem find_b_value (a b : ℝ) (h1 : 2 * a + 1 = 1) (h2 : b - a = 1) : b = 1 := by
  sorry

end find_b_value_l3218_321813


namespace power_3_2048_mod_11_l3218_321812

theorem power_3_2048_mod_11 : 3^2048 ≡ 5 [ZMOD 11] := by
  sorry

end power_3_2048_mod_11_l3218_321812


namespace exists_prime_pair_solution_l3218_321876

/-- A pair of prime numbers (p, q) is a solution if the quadratic equation
    px^2 - qx + p = 0 has rational roots. -/
def is_solution (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧
  ∃ (x y : ℚ), p * x^2 - q * x + p = 0 ∧ p * y^2 - q * y + p = 0 ∧ x ≠ y

/-- There exists a pair of prime numbers (p, q) that is a solution. -/
theorem exists_prime_pair_solution : ∃ (p q : ℕ), is_solution p q :=
sorry

end exists_prime_pair_solution_l3218_321876


namespace range_of_function_l3218_321807

theorem range_of_function (k : ℝ) (h : k > 0) :
  let f : ℝ → ℝ := fun x ↦ 3 * x^k
  Set.range (fun x ↦ f x) = Set.Ici (3 * 2^k) := by
  sorry

end range_of_function_l3218_321807


namespace points_five_units_away_l3218_321825

theorem points_five_units_away (x : ℝ) : 
  (|x - 2| = 5) ↔ (x = 7 ∨ x = -3) := by sorry

end points_five_units_away_l3218_321825


namespace sergio_fruit_sales_l3218_321835

/-- Calculates the total amount of money earned from fruit sales given the production of mangoes -/
def totalFruitSales (mangoProduction : ℕ) : ℕ :=
  let appleProduction := 2 * mangoProduction
  let orangeProduction := mangoProduction + 200
  let totalProduction := appleProduction + mangoProduction + orangeProduction
  totalProduction * 50

/-- Theorem stating that given the conditions, Mr. Sergio's total sales amount to $90,000 -/
theorem sergio_fruit_sales : totalFruitSales 400 = 90000 := by
  sorry

end sergio_fruit_sales_l3218_321835
