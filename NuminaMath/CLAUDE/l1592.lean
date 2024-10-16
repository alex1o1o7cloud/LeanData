import Mathlib

namespace NUMINAMATH_CALUDE_ryan_chinese_time_l1592_159222

/-- The time Ryan spends on learning English and Chinese daily -/
def total_time : ℝ := 3

/-- The time Ryan spends on learning English daily -/
def english_time : ℝ := 2

/-- The time Ryan spends on learning Chinese daily -/
def chinese_time : ℝ := total_time - english_time

theorem ryan_chinese_time : chinese_time = 1 := by sorry

end NUMINAMATH_CALUDE_ryan_chinese_time_l1592_159222


namespace NUMINAMATH_CALUDE_l₁_passes_through_point_distance_when_parallel_l1592_159242

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a + 2) * x + y + a + 1 = 0
def l₂ (a x y : ℝ) : Prop := 3 * x + a * y - 2 * a = 0

-- Statement 1: l₁ always passes through (-1, 1)
theorem l₁_passes_through_point (a : ℝ) : l₁ a (-1) 1 := by sorry

-- Helper function to check if lines are parallel
def parallel (a : ℝ) : Prop := a + 2 = 3 / a

-- Statement 2: When l₁ and l₂ are parallel, their distance is 2√10/5
theorem distance_when_parallel (a : ℝ) (h : parallel a) :
  ∃ d : ℝ, d = (2 * Real.sqrt 10) / 5 ∧ 
  (∀ x y : ℝ, l₁ a x y ↔ l₂ a (x + d * 3 / 5) (y - d * 4 / 5)) := by sorry

end NUMINAMATH_CALUDE_l₁_passes_through_point_distance_when_parallel_l1592_159242


namespace NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l1592_159253

theorem negation_of_exists_greater_than_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l1592_159253


namespace NUMINAMATH_CALUDE_cylinder_radius_l1592_159285

/-- The original radius of a cylinder satisfying specific conditions -/
theorem cylinder_radius : ∃ (r : ℝ), r > 0 ∧ 
  (∀ (y : ℝ), 
    (2 * π * ((r + 6)^2 - r^2) = y) ∧ 
    (6 * π * r^2 = y)) → 
  r = 6 := by sorry

end NUMINAMATH_CALUDE_cylinder_radius_l1592_159285


namespace NUMINAMATH_CALUDE_calculation_proof_l1592_159221

theorem calculation_proof : 2^2 - Real.tan (60 * π / 180) + |Real.sqrt 3 - 1| - (3 - Real.pi)^0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1592_159221


namespace NUMINAMATH_CALUDE_count_valid_paths_l1592_159200

/-- The number of paths from (0,1) to (n-1,n) that stay strictly above y=x -/
def validPaths (n : ℕ) : ℚ :=
  (1 : ℚ) / n * (Nat.choose (2*n - 2) (n - 1))

/-- Theorem stating the number of valid paths -/
theorem count_valid_paths (n : ℕ) (h : n > 0) :
  validPaths n = (1 : ℚ) / n * (Nat.choose (2*n - 2) (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_count_valid_paths_l1592_159200


namespace NUMINAMATH_CALUDE_shaded_area_is_30_l1592_159215

/-- An isosceles right triangle with legs of length 10 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  is_leg_length_10 : leg_length = 10

/-- A partition of the triangle into 25 congruent smaller triangles -/
structure Partition (t : IsoscelesRightTriangle) where
  num_small_triangles : ℕ
  is_25_triangles : num_small_triangles = 25

/-- The shaded region covering 15 of the smaller triangles -/
structure ShadedRegion (p : Partition t) where
  num_shaded_triangles : ℕ
  is_15_triangles : num_shaded_triangles = 15

/-- The theorem stating that the area of the shaded region is 30 -/
theorem shaded_area_is_30 (t : IsoscelesRightTriangle) (p : Partition t) (s : ShadedRegion p) :
  (t.leg_length ^ 2 / 2) * (s.num_shaded_triangles / p.num_small_triangles) = 30 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_is_30_l1592_159215


namespace NUMINAMATH_CALUDE_erasers_per_box_l1592_159234

/-- Given that Jacqueline has 4 boxes of erasers and a total of 40 erasers,
    prove that there are 10 erasers in each box. -/
theorem erasers_per_box (total_erasers : ℕ) (num_boxes : ℕ) (h1 : total_erasers = 40) (h2 : num_boxes = 4) :
  total_erasers / num_boxes = 10 := by
  sorry

#check erasers_per_box

end NUMINAMATH_CALUDE_erasers_per_box_l1592_159234


namespace NUMINAMATH_CALUDE_coffee_ratio_is_two_to_one_l1592_159298

/-- Represents the amount of coffee used for different strengths -/
structure CoffeeAmount where
  weak : ℕ
  strong : ℕ

/-- Calculates the ratio of strong to weak coffee -/
def coffeeRatio (amount : CoffeeAmount) : ℚ :=
  amount.strong / amount.weak

/-- Theorem stating the ratio of strong to weak coffee is 2:1 -/
theorem coffee_ratio_is_two_to_one :
  ∃ (amount : CoffeeAmount),
    amount.weak + amount.strong = 36 ∧
    amount.weak = 12 ∧
    coffeeRatio amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_coffee_ratio_is_two_to_one_l1592_159298


namespace NUMINAMATH_CALUDE_class_average_height_l1592_159229

def average_height_problem (total_students : ℕ) (group1_count : ℕ) (group1_avg : ℝ) (group2_avg : ℝ) : Prop :=
  let group2_count : ℕ := total_students - group1_count
  let total_height : ℝ := group1_count * group1_avg + group2_count * group2_avg
  let class_avg : ℝ := total_height / total_students
  class_avg = 168.6

theorem class_average_height :
  average_height_problem 50 40 169 167 := by
  sorry

end NUMINAMATH_CALUDE_class_average_height_l1592_159229


namespace NUMINAMATH_CALUDE_cart_distance_theorem_l1592_159272

/-- Represents a cart with two wheels -/
structure Cart where
  front_wheel_circumference : ℝ
  back_wheel_circumference : ℝ

/-- Calculates the distance traveled by the cart -/
def distance_traveled (c : Cart) (back_revolutions : ℝ) : ℝ :=
  back_revolutions * c.back_wheel_circumference

/-- Theorem stating the distance traveled by the cart -/
theorem cart_distance_theorem (c : Cart) 
    (h1 : c.front_wheel_circumference = 30)
    (h2 : c.back_wheel_circumference = 32)
    (h3 : ∃ (r : ℝ), r * c.back_wheel_circumference = (r + 5) * c.front_wheel_circumference) :
  ∃ (r : ℝ), distance_traveled c r = 2400 := by
  sorry

#check cart_distance_theorem

end NUMINAMATH_CALUDE_cart_distance_theorem_l1592_159272


namespace NUMINAMATH_CALUDE_system_one_solution_l1592_159257

theorem system_one_solution (x y : ℝ) : 
  2 * x + y = 4 ∧ x + 2 * y = 5 → x = 1 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_system_one_solution_l1592_159257


namespace NUMINAMATH_CALUDE_cube_surface_area_l1592_159216

/-- Given a cube made up of 6 squares, each with a perimeter of 24 cm,
    prove that its surface area is 216 cm². -/
theorem cube_surface_area (cube_side_length : ℝ) (square_perimeter : ℝ) : 
  square_perimeter = 24 →
  cube_side_length = square_perimeter / 4 →
  6 * cube_side_length ^ 2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1592_159216


namespace NUMINAMATH_CALUDE_wire_pieces_lengths_l1592_159299

/-- Represents the lengths of four pieces of wire --/
structure WirePieces where
  piece1 : ℝ
  piece2 : ℝ
  piece3 : ℝ
  piece4 : ℝ

/-- The total length of the wire is 72 feet --/
def totalLength : ℝ := 72

/-- Theorem stating the correct lengths of the wire pieces --/
theorem wire_pieces_lengths : ∃ (w : WirePieces),
  w.piece1 = 14.75 ∧
  w.piece2 = 11.75 ∧
  w.piece3 = 21.5 ∧
  w.piece4 = 24 ∧
  w.piece1 = w.piece2 + 3 ∧
  w.piece3 = 2 * w.piece2 - 2 ∧
  w.piece4 = (w.piece1 + w.piece2 + w.piece3) / 2 ∧
  w.piece1 + w.piece2 + w.piece3 + w.piece4 = totalLength := by
  sorry

end NUMINAMATH_CALUDE_wire_pieces_lengths_l1592_159299


namespace NUMINAMATH_CALUDE_prob_A_is_70_percent_l1592_159251

/-- The probability that person A speaks the truth -/
def prob_A : ℝ := sorry

/-- The probability that person B speaks the truth -/
def prob_B : ℝ := 0.6

/-- The probability that both A and B speak the truth simultaneously -/
def prob_AB : ℝ := 0.42

/-- Theorem stating that the probability of A speaking the truth is 70% -/
theorem prob_A_is_70_percent :
  (prob_AB = prob_A * prob_B) → prob_A = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_is_70_percent_l1592_159251


namespace NUMINAMATH_CALUDE_product_of_invertible_labels_l1592_159264

-- Define the function types
inductive FunctionType
| Quadratic
| ScatterPlot
| Sine
| Reciprocal

-- Define the structure for a function
structure Function where
  label : Nat
  type : FunctionType
  invertible : Bool

-- Define the problem setup
def problemSetup : List Function := [
  { label := 2, type := FunctionType.Quadratic, invertible := false },
  { label := 3, type := FunctionType.ScatterPlot, invertible := true },
  { label := 4, type := FunctionType.Sine, invertible := true },
  { label := 5, type := FunctionType.Reciprocal, invertible := true }
]

-- Theorem statement
theorem product_of_invertible_labels :
  (problemSetup.filter (λ f => f.invertible)).foldl (λ acc f => acc * f.label) 1 = 60 := by
  sorry

end NUMINAMATH_CALUDE_product_of_invertible_labels_l1592_159264


namespace NUMINAMATH_CALUDE_volleyball_tournament_l1592_159263

theorem volleyball_tournament (n : ℕ) : n > 0 → 2 * (n.choose 2) = 56 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_l1592_159263


namespace NUMINAMATH_CALUDE_prob_different_colors_is_four_fifths_l1592_159252

/-- Represents the colors of the balls in the bag -/
inductive BallColor
  | Red
  | Black
  | White

/-- Represents the contents of the bag -/
def bag : Multiset BallColor :=
  2 • {BallColor.Red} + 2 • {BallColor.Black} + 1 • {BallColor.White}

/-- The total number of balls in the bag -/
def totalBalls : ℕ := 5

/-- The number of ways to choose 2 balls from the bag -/
def totalChoices : ℕ := Nat.choose totalBalls 2

/-- The number of ways to choose 2 balls of the same color -/
def sameColorChoices : ℕ := Nat.choose 2 2 + Nat.choose 2 2

/-- The probability of drawing two balls of different colors -/
def probDifferentColors : ℚ :=
  1 - (sameColorChoices : ℚ) / totalChoices

theorem prob_different_colors_is_four_fifths :
  probDifferentColors = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_is_four_fifths_l1592_159252


namespace NUMINAMATH_CALUDE_average_pen_price_l1592_159284

/-- Given the purchase of pens and pencils, prove the average price of a pen. -/
theorem average_pen_price 
  (total_cost : ℝ)
  (num_pens : ℕ)
  (num_pencils : ℕ)
  (avg_pencil_price : ℝ)
  (h1 : total_cost = 570)
  (h2 : num_pens = 30)
  (h3 : num_pencils = 75)
  (h4 : avg_pencil_price = 2) :
  (total_cost - num_pencils * avg_pencil_price) / num_pens = 14 := by
  sorry

end NUMINAMATH_CALUDE_average_pen_price_l1592_159284


namespace NUMINAMATH_CALUDE_base_8_4513_equals_2379_l1592_159214

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4513_equals_2379 :
  base_8_to_10 [3, 1, 5, 4] = 2379 := by
  sorry

end NUMINAMATH_CALUDE_base_8_4513_equals_2379_l1592_159214


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_on_interval_l1592_159238

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem f_monotone_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ ≤ x₂ → x₂ ≤ 1 → f x₁ ≥ f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_on_interval_l1592_159238


namespace NUMINAMATH_CALUDE_one_third_1206_percent_of_200_l1592_159288

theorem one_third_1206_percent_of_200 : (1206 / 3) / 200 * 100 = 201 := by
  sorry

end NUMINAMATH_CALUDE_one_third_1206_percent_of_200_l1592_159288


namespace NUMINAMATH_CALUDE_isabel_homework_problem_l1592_159283

/-- The total number of homework problems Isabel had -/
def total_problems (finished : ℕ) (pages_left : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished + pages_left * problems_per_page

/-- Theorem stating that Isabel had 72 homework problems in total -/
theorem isabel_homework_problem :
  total_problems 32 5 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_problem_l1592_159283


namespace NUMINAMATH_CALUDE_nina_running_distance_l1592_159246

theorem nina_running_distance : 
  0.08333333333333333 + 0.08333333333333333 + 0.6666666666666666 = 0.8333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_nina_running_distance_l1592_159246


namespace NUMINAMATH_CALUDE_intersection_point_on_line_and_plane_l1592_159206

/-- The line passing through the point (5, 2, -4) in the direction <-2, 0, -1> -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (5 - 2*t, 2, -4 - t)

/-- The plane 2x - 5y + 4z + 24 = 0 -/
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  2*x - 5*y + 4*z + 24 = 0

/-- The intersection point of the line and the plane -/
def intersection_point : ℝ × ℝ × ℝ := (3, 2, -5)

theorem intersection_point_on_line_and_plane :
  ∃ t : ℝ, line t = intersection_point ∧ plane intersection_point := by
  sorry


end NUMINAMATH_CALUDE_intersection_point_on_line_and_plane_l1592_159206


namespace NUMINAMATH_CALUDE_inscribed_squares_equal_area_l1592_159203

/-- 
Given an isosceles right triangle with an inscribed square parallel to the legs,
prove that if this square has an area of 625, then a square inscribed with sides
parallel and perpendicular to the hypotenuse also has an area of 625.
-/
theorem inscribed_squares_equal_area (side : ℝ) (h_area : side^2 = 625) :
  let hypotenuse := side * Real.sqrt 2
  let side_hyp_square := hypotenuse / 2
  side_hyp_square^2 = 625 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_equal_area_l1592_159203


namespace NUMINAMATH_CALUDE_cows_for_96_days_l1592_159204

/-- Represents the number of cows that can eat all the grass in a given number of days -/
structure GrazingScenario where
  cows : ℕ
  days : ℕ

/-- Represents the meadow with growing grass -/
structure Meadow where
  scenario1 : GrazingScenario
  scenario2 : GrazingScenario
  growth_rate : ℚ

/-- Calculate the number of cows that can eat all the grass in 96 days -/
def calculate_cows (m : Meadow) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem cows_for_96_days (m : Meadow) : 
  m.scenario1 = ⟨70, 24⟩ → 
  m.scenario2 = ⟨30, 60⟩ → 
  calculate_cows m = 20 := by
  sorry

end NUMINAMATH_CALUDE_cows_for_96_days_l1592_159204


namespace NUMINAMATH_CALUDE_lot_width_calculation_l1592_159245

/-- Given a rectangular lot with length 40 m, height 2 m, and volume 1600 m³, 
    the width of the lot is 20 m. -/
theorem lot_width_calculation (length height volume width : ℝ) 
  (h_length : length = 40)
  (h_height : height = 2)
  (h_volume : volume = 1600)
  (h_relation : volume = length * width * height) : 
  width = 20 := by
  sorry

end NUMINAMATH_CALUDE_lot_width_calculation_l1592_159245


namespace NUMINAMATH_CALUDE_mass_percentage_H_in_C9H14N3O5_l1592_159239

/-- Molar mass of carbon in g/mol -/
def molar_mass_C : ℝ := 12.01

/-- Molar mass of hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- Molar mass of nitrogen in g/mol -/
def molar_mass_N : ℝ := 14.01

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Calculate the mass percentage of hydrogen in C9H14N3O5 -/
theorem mass_percentage_H_in_C9H14N3O5 :
  let total_mass := 9 * molar_mass_C + 14 * molar_mass_H + 3 * molar_mass_N + 5 * molar_mass_O
  let mass_H := 14 * molar_mass_H
  let percentage := (mass_H / total_mass) * 100
  ∃ ε > 0, |percentage - 5.79| < ε :=
sorry

end NUMINAMATH_CALUDE_mass_percentage_H_in_C9H14N3O5_l1592_159239


namespace NUMINAMATH_CALUDE_point_on_line_with_vector_relation_l1592_159210

/-- Given points A and B, if point P is on line AB and vector AB is twice vector AP, 
    then P has specific coordinates -/
theorem point_on_line_with_vector_relation (A B P : ℝ × ℝ) : 
  A = (2, 0) → 
  B = (4, 2) → 
  (∃ t : ℝ, P = (1 - t) • A + t • B) →  -- P is on line AB
  B - A = 2 • (P - A) → 
  P = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_with_vector_relation_l1592_159210


namespace NUMINAMATH_CALUDE_probability_is_two_fifths_l1592_159211

/-- A diagram with five triangles, two of which are shaded. -/
structure Diagram where
  triangles : Finset (Fin 5)
  shaded : Finset (Fin 5)
  total_triangles : triangles.card = 5
  shaded_triangles : shaded.card = 2
  shaded_subset : shaded ⊆ triangles

/-- The probability of selecting a shaded triangle from the diagram. -/
def probability_shaded (d : Diagram) : ℚ :=
  d.shaded.card / d.triangles.card

/-- Theorem stating that the probability of selecting a shaded triangle is 2/5. -/
theorem probability_is_two_fifths (d : Diagram) :
  probability_shaded d = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_two_fifths_l1592_159211


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1592_159240

theorem arithmetic_calculations : 
  ((-8) + 10 - 2 + (-1) = -1) ∧ 
  (12 - 7 * (-4) + 8 / (-2) = 36) ∧ 
  ((1/2 + 1/3 - 1/6) / (-1/18) = -12) ∧ 
  (-1^4 - (1 + 0.5) * (1/3) * (-4)^2 = -33/32) := by
  sorry

#eval (-8) + 10 - 2 + (-1)
#eval 12 - 7 * (-4) + 8 / (-2)
#eval (1/2 + 1/3 - 1/6) / (-1/18)
#eval -1^4 - (1 + 0.5) * (1/3) * (-4)^2

end NUMINAMATH_CALUDE_arithmetic_calculations_l1592_159240


namespace NUMINAMATH_CALUDE_election_invalid_votes_percentage_l1592_159277

theorem election_invalid_votes_percentage 
  (total_votes : ℕ) 
  (votes_B : ℕ) 
  (h_total : total_votes = 9720)
  (h_B : votes_B = 3159)
  (h_difference : ∃ (votes_A : ℕ), votes_A = votes_B + (15 * total_votes) / 100) :
  (total_votes - (votes_B + (votes_B + (15 * total_votes) / 100))) * 100 / total_votes = 20 := by
sorry

end NUMINAMATH_CALUDE_election_invalid_votes_percentage_l1592_159277


namespace NUMINAMATH_CALUDE_stock_price_change_l1592_159217

theorem stock_price_change (total_stocks : ℕ) (higher_percentage : ℚ) : 
  total_stocks = 4200 →
  higher_percentage = 35/100 →
  ∃ (higher lower : ℕ),
    higher + lower = total_stocks ∧
    higher = (1 + higher_percentage) * lower ∧
    higher = 2412 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l1592_159217


namespace NUMINAMATH_CALUDE_digit_129_in_n_or_3n_l1592_159275

/-- Given a natural number, returns true if it contains the digit 1, 2, or 9 in its base-ten representation -/
def containsDigit129 (n : ℕ) : Prop :=
  ∃ d, d ∈ [1, 2, 9] ∧ ∃ k m, n = k * 10 + d + m * 10

theorem digit_129_in_n_or_3n (n : ℕ+) : containsDigit129 n.val ∨ containsDigit129 (3 * n.val) := by
  sorry

end NUMINAMATH_CALUDE_digit_129_in_n_or_3n_l1592_159275


namespace NUMINAMATH_CALUDE_cylinder_volume_l1592_159232

/-- The volume of a solid cylinder in a cubic container --/
theorem cylinder_volume (container_side : ℝ) (exposed_height : ℝ) (base_area_ratio : ℝ) :
  container_side = 20 →
  exposed_height = 8 →
  base_area_ratio = 1/8 →
  (container_side - exposed_height) * (container_side * container_side * base_area_ratio) = 650 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1592_159232


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1592_159268

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 2) : 
  (a^2 - 4*a + 4) / a / (a - 4/a) = 1 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1592_159268


namespace NUMINAMATH_CALUDE_spoiled_apple_probability_l1592_159231

/-- The probability of selecting a spoiled apple from a basket -/
def prob_spoiled_apple (total : ℕ) (spoiled : ℕ) (selected : ℕ) : ℚ :=
  (selected : ℚ) / total

/-- The number of ways to choose k items from n items -/
def combinations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem spoiled_apple_probability :
  let total := 7
  let spoiled := 1
  let selected := 2
  prob_spoiled_apple total spoiled selected = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_spoiled_apple_probability_l1592_159231


namespace NUMINAMATH_CALUDE_shifted_sine_function_l1592_159291

/-- Given a function f and its right-shifted version g, prove that g has the correct form -/
theorem shifted_sine_function 
  (f g : ℝ → ℝ) 
  (h₁ : ∀ x, f x = 3 * Real.sin (2 * x))
  (h₂ : ∀ x, g x = f (x - π/8)) :
  ∀ x, g x = 3 * Real.sin (2 * x - π/4) := by
  sorry


end NUMINAMATH_CALUDE_shifted_sine_function_l1592_159291


namespace NUMINAMATH_CALUDE_librarian_took_books_oliver_book_problem_l1592_159271

theorem librarian_took_books (total_books : ℕ) (books_per_shelf : ℕ) (shelves_needed : ℕ) : ℕ :=
  let remaining_books := shelves_needed * books_per_shelf
  total_books - remaining_books

theorem oliver_book_problem :
  librarian_took_books 46 4 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_librarian_took_books_oliver_book_problem_l1592_159271


namespace NUMINAMATH_CALUDE_sum_of_valid_a_l1592_159226

theorem sum_of_valid_a : ∃ (S : Finset Int), 
  (∀ a ∈ S, (∃ x : Int, x ≤ 2 ∧ x > a + 2) ∧ 
             (∃ x y : Nat, a * x + 2 * y = -4 ∧ x + y = 4)) ∧
  (∀ a : Int, (∃ x : Int, x ≤ 2 ∧ x > a + 2) ∧ 
              (∃ x y : Nat, a * x + 2 * y = -4 ∧ x + y = 4) → a ∈ S) ∧
  (S.sum id = -16) := by
sorry

end NUMINAMATH_CALUDE_sum_of_valid_a_l1592_159226


namespace NUMINAMATH_CALUDE_no_real_solutions_log_equation_l1592_159290

theorem no_real_solutions_log_equation :
  ¬ ∃ (x : ℝ), Real.log (x^2 - 3*x + 9) = 1 := by sorry

end NUMINAMATH_CALUDE_no_real_solutions_log_equation_l1592_159290


namespace NUMINAMATH_CALUDE_kiyana_grapes_l1592_159202

/-- Proves that if Kiyana has 24 grapes and gives away half of them, the number of grapes she gives away is 12. -/
theorem kiyana_grapes : 
  let total_grapes : ℕ := 24
  let grapes_given_away : ℕ := total_grapes / 2
  grapes_given_away = 12 := by
  sorry

end NUMINAMATH_CALUDE_kiyana_grapes_l1592_159202


namespace NUMINAMATH_CALUDE_equation_solution_l1592_159237

theorem equation_solution :
  let x : ℚ := 32
  let n : ℚ := -5/6
  35 - (23 - (15 - x)) = 12 * n / (1 / 2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1592_159237


namespace NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_is_30_degrees_l1592_159247

theorem angle_with_complement_40percent_of_supplement_is_30_degrees :
  ∀ x : ℝ,
  (x > 0) →
  (x < 90) →
  (90 - x = (2/5) * (180 - x)) →
  x = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_is_30_degrees_l1592_159247


namespace NUMINAMATH_CALUDE_scooter_profit_percentage_l1592_159236

/-- Calculates the profit percentage for a scooter sale given specific conditions -/
theorem scooter_profit_percentage 
  (initial_price : ℝ)
  (initial_repair_rate : ℝ)
  (additional_maintenance : ℝ)
  (safety_upgrade_rate : ℝ)
  (sales_tax_rate : ℝ)
  (selling_price : ℝ)
  (h1 : initial_price = 4700)
  (h2 : initial_repair_rate = 0.1)
  (h3 : additional_maintenance = 500)
  (h4 : safety_upgrade_rate = 0.05)
  (h5 : sales_tax_rate = 0.12)
  (h6 : selling_price = 5800) :
  let initial_repair := initial_price * initial_repair_rate
  let total_repair := initial_repair + additional_maintenance
  let safety_upgrade := total_repair * safety_upgrade_rate
  let total_cost := initial_price + total_repair + safety_upgrade
  let sales_tax := selling_price * sales_tax_rate
  let total_selling_price := selling_price + sales_tax
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  ∃ ε > 0, abs (profit_percentage - 13.60) < ε :=
by sorry

end NUMINAMATH_CALUDE_scooter_profit_percentage_l1592_159236


namespace NUMINAMATH_CALUDE_golden_silk_button_optimal_price_reduction_l1592_159230

/-- Represents the problem of finding the optimal price reduction for Golden Silk Button --/
theorem golden_silk_button_optimal_price_reduction 
  (initial_cost : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (sales_increase_rate : ℝ) 
  (target_profit : ℝ) 
  (price_reduction : ℝ) : 
  initial_cost = 24 → 
  initial_price = 40 → 
  initial_sales = 20 → 
  sales_increase_rate = 2 → 
  target_profit = 330 → 
  price_reduction = 5 → 
  (initial_price - price_reduction - initial_cost) * (initial_sales + sales_increase_rate * price_reduction) = target_profit :=
by sorry

end NUMINAMATH_CALUDE_golden_silk_button_optimal_price_reduction_l1592_159230


namespace NUMINAMATH_CALUDE_sweater_price_after_discounts_l1592_159287

/-- Calculates the final price of an item after two successive discounts -/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  originalPrice * (1 - discount1) * (1 - discount2)

/-- Theorem: The final price of a $240 sweater after 60% and 30% discounts is $67.20 -/
theorem sweater_price_after_discounts :
  finalPrice 240 0.6 0.3 = 67.2 := by
  sorry

end NUMINAMATH_CALUDE_sweater_price_after_discounts_l1592_159287


namespace NUMINAMATH_CALUDE_cinema_systematic_sampling_l1592_159255

/-- Represents a sampling method --/
inductive SamplingMethod
  | LotteryMethod
  | RandomNumberMethod
  | StratifiedSampling
  | SystematicSampling

/-- Represents a cinema with rows and seats --/
structure Cinema where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a selection of audience members --/
structure AudienceSelection where
  seatNumber : Nat
  count : Nat

/-- Determines the sampling method based on the cinema layout and audience selection --/
def determineSamplingMethod (c : Cinema) (a : AudienceSelection) : SamplingMethod :=
  sorry

/-- Theorem stating that the given scenario results in systematic sampling --/
theorem cinema_systematic_sampling (c : Cinema) (a : AudienceSelection) :
  c.rows = 30 ∧ c.seatsPerRow = 25 ∧ a.seatNumber = 18 ∧ a.count = 30 →
  determineSamplingMethod c a = SamplingMethod.SystematicSampling :=
  sorry

end NUMINAMATH_CALUDE_cinema_systematic_sampling_l1592_159255


namespace NUMINAMATH_CALUDE_total_blocks_l1592_159296

theorem total_blocks (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : red = 18)
  (h2 : yellow = red + 7)
  (h3 : blue = red + 14) :
  red + yellow + blue = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_blocks_l1592_159296


namespace NUMINAMATH_CALUDE_max_stores_visited_l1592_159243

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 12) (h2 : total_visits = 45) 
  (h3 : total_shoppers = 22) (h4 : double_visitors = 14) 
  (h5 : double_visitors ≤ total_shoppers) 
  (h6 : 2 * double_visitors ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ total_stores ∧ 
    (∀ (person_visits : ℕ), person_visits ≤ max_visits) ∧ 
    max_visits = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l1592_159243


namespace NUMINAMATH_CALUDE_circle_center_parabola_focus_l1592_159278

/-- The value of p for which the center of the circle x^2 + y^2 - 6x = 0 
    is exactly the focus of the parabola y^2 = 2px (p > 0) -/
theorem circle_center_parabola_focus (p : ℝ) : p > 0 → 
  (∃ (x y : ℝ), x^2 + y^2 - 6*x = 0 ∧ y^2 = 2*p*x) →
  (∀ (x y : ℝ), x^2 + y^2 - 6*x = 0 → x = 3 ∧ y = 0) →
  (∀ (x y : ℝ), y^2 = 2*p*x → x = p/2 ∧ y = 0) →
  p = 6 := by sorry

end NUMINAMATH_CALUDE_circle_center_parabola_focus_l1592_159278


namespace NUMINAMATH_CALUDE_tank_width_is_four_feet_l1592_159266

/-- Proves that the width of a rectangular tank is 4 feet given specific conditions. -/
theorem tank_width_is_four_feet 
  (fill_rate : ℝ) 
  (length depth time_to_fill : ℝ) 
  (h_fill_rate : fill_rate = 4)
  (h_length : length = 6)
  (h_depth : depth = 3)
  (h_time_to_fill : time_to_fill = 18)
  : (fill_rate * time_to_fill) / (length * depth) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tank_width_is_four_feet_l1592_159266


namespace NUMINAMATH_CALUDE_square_root_range_l1592_159261

theorem square_root_range (x : ℝ) : x - 2 ≥ 0 ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_range_l1592_159261


namespace NUMINAMATH_CALUDE_work_left_after_three_days_l1592_159258

/-- The fraction of work left after two workers collaborate -/
def work_left (days_a : ℕ) (days_b : ℕ) (days_together : ℕ) : ℚ :=
  1 - (days_together : ℚ) * (1 / days_a + 1 / days_b)

/-- Theorem stating the fraction of work left after 3 days of collaboration -/
theorem work_left_after_three_days :
  work_left 15 20 3 = 13 / 20 := by
  sorry

end NUMINAMATH_CALUDE_work_left_after_three_days_l1592_159258


namespace NUMINAMATH_CALUDE_product_of_roots_l1592_159254

theorem product_of_roots (x : ℝ) : (x + 4) * (x - 5) = 22 → ∃ y : ℝ, (x + 4) * (x - 5) = 22 ∧ (x * y = -42) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1592_159254


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l1592_159280

theorem empty_solution_set_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 3 > a^2 - 2*a - 1) ↔ (-1 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l1592_159280


namespace NUMINAMATH_CALUDE_girls_count_l1592_159294

theorem girls_count (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 600 → 
  difference = 30 → 
  girls + (girls - difference) = total → 
  girls = 315 := by
sorry

end NUMINAMATH_CALUDE_girls_count_l1592_159294


namespace NUMINAMATH_CALUDE_c_range_l1592_159295

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem c_range (c : ℝ) : c > 0 →
  (is_decreasing (fun x ↦ c^x) ∨ (∀ x ∈ Set.Icc 0 2, x + c > 2)) ∧
  ¬(is_decreasing (fun x ↦ c^x) ∧ (∀ x ∈ Set.Icc 0 2, x + c > 2)) →
  (0 < c ∧ c < 1) ∨ c > 2 :=
by sorry

end NUMINAMATH_CALUDE_c_range_l1592_159295


namespace NUMINAMATH_CALUDE_factorization_problems_l1592_159225

theorem factorization_problems :
  (∀ x y : ℝ, 2*x^2*y - 8*x*y + 8*y = 2*y*(x-2)^2) ∧
  (∀ a : ℝ, 18*a^2 - 50 = 2*(3*a+5)*(3*a-5)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1592_159225


namespace NUMINAMATH_CALUDE_alice_current_age_l1592_159235

/-- Alice's current age -/
def alice_age : ℕ := 30

/-- Beatrice's current age -/
def beatrice_age : ℕ := 11

/-- In 8 years, Alice will be twice as old as Beatrice -/
axiom future_age_relation : alice_age + 8 = 2 * (beatrice_age + 8)

/-- Ten years ago, the sum of their ages was 21 -/
axiom past_age_sum : (alice_age - 10) + (beatrice_age - 10) = 21

theorem alice_current_age : alice_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_alice_current_age_l1592_159235


namespace NUMINAMATH_CALUDE_f_not_prime_l1592_159219

def f (n : ℕ+) : ℤ := n.val^4 - 380 * n.val^2 + 841

theorem f_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (f n)) := by
  sorry

end NUMINAMATH_CALUDE_f_not_prime_l1592_159219


namespace NUMINAMATH_CALUDE_smallest_screw_count_screw_packs_problem_l1592_159213

theorem smallest_screw_count : ℕ → Prop :=
  fun k => (∃ x y : ℕ, x ≠ y ∧ k = 10 * x ∧ k = 12 * y) ∧
           (∀ m : ℕ, m < k → ¬(∃ a b : ℕ, a ≠ b ∧ m = 10 * a ∧ m = 12 * b))

theorem screw_packs_problem : smallest_screw_count 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_screw_count_screw_packs_problem_l1592_159213


namespace NUMINAMATH_CALUDE_sin_theta_value_l1592_159244

theorem sin_theta_value (θ : Real) (h1 : 5 * Real.tan θ = 2 * Real.cos θ) (h2 : 0 < θ) (h3 : θ < Real.pi) :
  Real.sin θ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_value_l1592_159244


namespace NUMINAMATH_CALUDE_speed_above_limit_l1592_159249

def distance : ℝ := 150
def time : ℝ := 2
def speed_limit : ℝ := 60

theorem speed_above_limit : (distance / time) - speed_limit = 15 := by
  sorry

end NUMINAMATH_CALUDE_speed_above_limit_l1592_159249


namespace NUMINAMATH_CALUDE_jeff_average_skips_l1592_159260

def jeff_skips (sam_skips : ℕ) (rounds : ℕ) : List ℕ :=
  let round1 := sam_skips - 1
  let round2 := sam_skips - 3
  let round3 := sam_skips + 4
  let round4 := sam_skips / 2
  let round5 := round4 + (sam_skips - round4 + 2)
  [round1, round2, round3, round4, round5]

def average_skips (skips : List ℕ) : ℚ :=
  (skips.sum : ℚ) / skips.length

theorem jeff_average_skips (sam_skips : ℕ) (rounds : ℕ) :
  sam_skips = 16 ∧ rounds = 5 →
  average_skips (jeff_skips sam_skips rounds) = 74/5 :=
by sorry

end NUMINAMATH_CALUDE_jeff_average_skips_l1592_159260


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1592_159267

theorem sin_product_equals_one_sixteenth :
  Real.sin (18 * π / 180) * Real.sin (42 * π / 180) *
  Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1592_159267


namespace NUMINAMATH_CALUDE_xyz_value_l1592_159274

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1592_159274


namespace NUMINAMATH_CALUDE_max_value_range_l1592_159256

/-- Given a function f with derivative f' and a real number a, 
    prove that if f'(x) = a(x-1)(x-a) and f attains a maximum at x = a, 
    then 0 < a < 1 -/
theorem max_value_range (f f' : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f' x = a * (x - 1) * (x - a))
  (h2 : IsLocalMax f a) : 0 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_max_value_range_l1592_159256


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1592_159293

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : α > Real.pi / 2) 
  (h2 : α < Real.pi) 
  (h3 : Real.sin α = 5 / 13) : 
  Real.tan (α + Real.pi / 4) = 7 / 17 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1592_159293


namespace NUMINAMATH_CALUDE_missing_number_last_two_digits_l1592_159233

def last_two_digits (n : ℕ) : ℕ := n % 100

def product_last_two_digits (nums : List ℕ) : ℕ :=
  last_two_digits (nums.foldl (λ acc x => last_two_digits (acc * last_two_digits x)) 1)

theorem missing_number_last_two_digits
  (h : product_last_two_digits [122, 123, 125, 129, x] = 50) :
  last_two_digits x = 1 :=
sorry

end NUMINAMATH_CALUDE_missing_number_last_two_digits_l1592_159233


namespace NUMINAMATH_CALUDE_largest_fraction_l1592_159227

theorem largest_fraction : 
  (101 : ℚ) / 199 > 5 / 11 ∧
  (101 : ℚ) / 199 > 6 / 13 ∧
  (101 : ℚ) / 199 > 19 / 39 ∧
  (101 : ℚ) / 199 > 159 / 319 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l1592_159227


namespace NUMINAMATH_CALUDE_f_sum_equals_half_point_five_l1592_159289

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f(x+1) = -f(x) for all x -/
axiom f_period (x : ℝ) : f (x + 1) = -f x

/-- f(x) = x for x in (-1, 1) -/
axiom f_identity (x : ℝ) (h : x > -1 ∧ x < 1) : f x = x

/-- The main theorem to prove -/
theorem f_sum_equals_half_point_five : f 3 + f (-7.5) = 0.5 := by sorry

end NUMINAMATH_CALUDE_f_sum_equals_half_point_five_l1592_159289


namespace NUMINAMATH_CALUDE_no_four_naturals_exist_l1592_159224

theorem no_four_naturals_exist : ¬∃ (a b c d : ℕ), 
  a + b + c + d = 2^100 ∧ a * b * c * d = 17^100 := by
  sorry

end NUMINAMATH_CALUDE_no_four_naturals_exist_l1592_159224


namespace NUMINAMATH_CALUDE_lino_shell_collection_l1592_159259

/-- The number of shells Lino picked up in the morning -/
def morning_shells : ℕ := 292

/-- The number of shells Lino picked up in the afternoon -/
def afternoon_shells : ℕ := 324

/-- The total number of shells Lino picked up -/
def total_shells : ℕ := morning_shells + afternoon_shells

/-- Theorem stating that the total number of shells Lino picked up is 616 -/
theorem lino_shell_collection : total_shells = 616 := by
  sorry

end NUMINAMATH_CALUDE_lino_shell_collection_l1592_159259


namespace NUMINAMATH_CALUDE_fourth_dog_weight_l1592_159207

theorem fourth_dog_weight (y : ℝ) :
  let dog1 : ℝ := 25
  let dog2 : ℝ := 31
  let dog3 : ℝ := 35
  let dog4 : ℝ := x
  let dog5 : ℝ := y
  (dog1 + dog2 + dog3 + dog4) / 4 = (dog1 + dog2 + dog3 + dog4 + dog5) / 5 →
  x = -91 - 5 * y :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_dog_weight_l1592_159207


namespace NUMINAMATH_CALUDE_gcf_of_75_and_105_l1592_159282

theorem gcf_of_75_and_105 : Nat.gcd 75 105 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_105_l1592_159282


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l1592_159292

theorem simultaneous_equations_solution :
  ∃! (x y : ℚ), 3 * x - 4 * y = 11 ∧ 9 * x + 6 * y = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l1592_159292


namespace NUMINAMATH_CALUDE_thirty_day_month_equal_tuesdays_thursdays_l1592_159201

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- A function that checks if a given starting day results in equal Tuesdays and Thursdays in a 30-day month -/
def equalTuesdaysThursdays (startDay : DayOfWeek) : Bool :=
  sorry

/-- The number of days that can be the first day of a 30-day month with equal Tuesdays and Thursdays -/
def validStartDays : Nat :=
  sorry

theorem thirty_day_month_equal_tuesdays_thursdays :
  validStartDays = 4 :=
sorry

end NUMINAMATH_CALUDE_thirty_day_month_equal_tuesdays_thursdays_l1592_159201


namespace NUMINAMATH_CALUDE_rajesh_savings_l1592_159228

def monthly_salary : ℕ := 15000
def food_percentage : ℚ := 40 / 100
def medicine_percentage : ℚ := 20 / 100
def savings_percentage : ℚ := 60 / 100

theorem rajesh_savings : 
  let remaining := monthly_salary - (monthly_salary * food_percentage + monthly_salary * medicine_percentage)
  ↑(remaining * savings_percentage) = 3600 := by sorry

end NUMINAMATH_CALUDE_rajesh_savings_l1592_159228


namespace NUMINAMATH_CALUDE_tan_sum_range_l1592_159279

theorem tan_sum_range (m : ℝ) (α β : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ 
    m * x^2 - 2 * x * Real.sqrt (7 * m - 3) + 2 * m = 0 ∧
    m * y^2 - 2 * y * Real.sqrt (7 * m - 3) + 2 * m = 0 ∧
    x = Real.tan α ∧ y = Real.tan β) →
  ∃ (l u : ℝ), l = -(7 * Real.sqrt 3) / 3 ∧ u = -2 * Real.sqrt 2 ∧
    Real.tan (α + β) ∈ Set.Icc l u :=
sorry

end NUMINAMATH_CALUDE_tan_sum_range_l1592_159279


namespace NUMINAMATH_CALUDE_science_club_committee_selection_l1592_159269

theorem science_club_committee_selection (total_candidates : Nat) 
  (previously_served : Nat) (committee_size : Nat) 
  (h1 : total_candidates = 20) (h2 : previously_served = 8) 
  (h3 : committee_size = 4) :
  Nat.choose total_candidates committee_size - 
  Nat.choose (total_candidates - previously_served) committee_size = 4350 :=
by
  sorry

end NUMINAMATH_CALUDE_science_club_committee_selection_l1592_159269


namespace NUMINAMATH_CALUDE_total_paving_cost_l1592_159205

/-- Represents a section of a room with its dimensions and slab cost -/
structure Section where
  length : ℝ
  width : ℝ
  slabCost : ℝ

/-- Calculates the cost of paving a section -/
def sectionCost (s : Section) : ℝ :=
  s.length * s.width * s.slabCost

/-- The three sections of the room -/
def sectionA : Section := { length := 8, width := 4.75, slabCost := 900 }
def sectionB : Section := { length := 6, width := 3.25, slabCost := 800 }
def sectionC : Section := { length := 5, width := 2.5, slabCost := 1000 }

/-- Theorem stating the total cost of paving the floor for the entire room -/
theorem total_paving_cost :
  sectionCost sectionA + sectionCost sectionB + sectionCost sectionC = 62300 := by
  sorry


end NUMINAMATH_CALUDE_total_paving_cost_l1592_159205


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1592_159281

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1592_159281


namespace NUMINAMATH_CALUDE_expression_range_l1592_159270

theorem expression_range (x a b c : ℝ) (h : a^2 + b^2 + c^2 ≠ 0) :
  ∃ y ∈ Set.Icc (-Real.sqrt 5) (Real.sqrt 5),
    y = (a * Real.cos x - b * Real.sin x + 2 * c) / Real.sqrt (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_range_l1592_159270


namespace NUMINAMATH_CALUDE_system_solution_transformation_l1592_159208

theorem system_solution_transformation 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h : ∃ (x y : ℝ), x = 3 ∧ y = 4 ∧ a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂) :
  ∃ (x y : ℝ), x = 5 ∧ y = 5 ∧ 3 * a₁ * x + 4 * b₁ * y = 5 * c₁ ∧ 3 * a₂ * x + 4 * b₂ * y = 5 * c₂ :=
by sorry

end NUMINAMATH_CALUDE_system_solution_transformation_l1592_159208


namespace NUMINAMATH_CALUDE_division_and_addition_l1592_159223

theorem division_and_addition : -4 + 6 / (-2) = -7 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l1592_159223


namespace NUMINAMATH_CALUDE_intersection_equality_l1592_159248

def A (m : ℝ) : Set ℝ := {-1, 3, m}
def B : Set ℝ := {3, 4}

theorem intersection_equality (m : ℝ) : B ∩ A m = B → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l1592_159248


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1592_159276

/-- Given a geometric sequence {a_n} where a₂a₆ + a₄² = π, prove that a₃a₅ = π/2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_eq : a 2 * a 6 + a 4 * a 4 = Real.pi) : a 3 * a 5 = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1592_159276


namespace NUMINAMATH_CALUDE_john_average_speed_l1592_159286

/-- John's average speed in miles per hour -/
def john_speed : ℝ := 30

/-- Carla's average speed in miles per hour -/
def carla_speed : ℝ := 35

/-- Time Carla needs to catch up to John in hours -/
def catch_up_time : ℝ := 3

/-- Time difference between John's and Carla's departure in hours -/
def departure_time_difference : ℝ := 0.5

theorem john_average_speed :
  john_speed = 30 ∧
  carla_speed * catch_up_time = john_speed * (catch_up_time + departure_time_difference) :=
sorry

end NUMINAMATH_CALUDE_john_average_speed_l1592_159286


namespace NUMINAMATH_CALUDE_era_burger_left_l1592_159212

/-- Represents the problem of Era's burger distribution --/
def era_burger_problem (total_burgers : ℕ) (num_friends : ℕ) (slices_per_burger : ℕ) 
  (friend1_slices : ℕ) (friend2_slices : ℕ) (friend3_slices : ℕ) (friend4_slices : ℕ) : Prop :=
  total_burgers = 5 ∧
  num_friends = 4 ∧
  slices_per_burger = 2 ∧
  friend1_slices = 1 ∧
  friend2_slices = 2 ∧
  friend3_slices = 3 ∧
  friend4_slices = 3

/-- Theorem stating that Era has 1 slice of burger left --/
theorem era_burger_left (total_burgers num_friends slices_per_burger 
  friend1_slices friend2_slices friend3_slices friend4_slices : ℕ) :
  era_burger_problem total_burgers num_friends slices_per_burger 
    friend1_slices friend2_slices friend3_slices friend4_slices →
  total_burgers * slices_per_burger - (friend1_slices + friend2_slices + friend3_slices + friend4_slices) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_era_burger_left_l1592_159212


namespace NUMINAMATH_CALUDE_right_angled_constructions_l1592_159220

/-- Represents a triangle with angles in degrees -/
structure Triangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)

/-- Checks if a triangle is right-angled -/
def is_right_angled (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

/-- The basic triangle obtained from dividing a regular hexagon into 12 parts -/
def basic_triangle : Triangle :=
  { angle1 := 30, angle2 := 60, angle3 := 90 }

/-- Represents the number of basic triangles used to form a larger triangle -/
inductive TriangleComposition
  | One
  | Three
  | Four
  | Nine

/-- Function to construct a triangle from a given number of basic triangles -/
def construct_triangle (n : TriangleComposition) : Triangle :=
  sorry

/-- Theorem stating that right-angled triangles can be formed using 1, 3, 4, or 9 basic triangles -/
theorem right_angled_constructions :
  ∀ n : TriangleComposition, is_right_angled (construct_triangle n) :=
sorry

end NUMINAMATH_CALUDE_right_angled_constructions_l1592_159220


namespace NUMINAMATH_CALUDE_rectangular_block_height_l1592_159297

/-- The height of a rectangular block with given volume and base area -/
theorem rectangular_block_height (volume : ℝ) (base_area : ℝ) (height : ℝ) : 
  volume = 120 → base_area = 24 → volume = base_area * height → height = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_block_height_l1592_159297


namespace NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_geometric_progression_l1592_159265

theorem quadratic_discriminant_zero_implies_geometric_progression
  (k a b c : ℝ) (h1 : k ≠ 0) :
  4 * k^2 * (b^2 - a*c) = 0 →
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r :=
by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_geometric_progression_l1592_159265


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1592_159241

theorem right_triangle_shorter_leg : 
  ∀ (a b c : ℕ), 
    a^2 + b^2 = c^2 →  -- Pythagorean theorem
    c = 65 →  -- hypotenuse length
    a ≤ b →  -- a is the shorter leg
    a = 16 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1592_159241


namespace NUMINAMATH_CALUDE_quadrupled_base_exponent_l1592_159218

theorem quadrupled_base_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4*a)^(4*b) = a^b * x^(2*b) → x = 16 * a^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_base_exponent_l1592_159218


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1592_159273

theorem divisibility_theorem (a b c : ℕ) 
  (h1 : b ∣ a^3) 
  (h2 : c ∣ b^3) 
  (h3 : a ∣ c^3) : 
  a * b * c ∣ (a + b + c)^13 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1592_159273


namespace NUMINAMATH_CALUDE_sum_in_base5_l1592_159250

/-- Converts a number from base 4 to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 --/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Represents a number in base 5 --/
structure Base5 (n : ℕ) where
  value : ℕ
  isBase5 : value < 5^n

theorem sum_in_base5 :
  let a := base4ToBase10 203
  let b := base4ToBase10 112
  let c := base4ToBase10 321
  let sum := a + b + c
  base10ToBase5 sum = 2222 :=
sorry

end NUMINAMATH_CALUDE_sum_in_base5_l1592_159250


namespace NUMINAMATH_CALUDE_k_negative_sufficient_not_necessary_l1592_159209

-- Define the condition for the equation to represent a hyperbola
def is_hyperbola (k : ℝ) : Prop := k * (k - 1) > 0

-- State the theorem
theorem k_negative_sufficient_not_necessary :
  (∀ k : ℝ, k < 0 → is_hyperbola k) ∧
  (∃ k : ℝ, ¬(k < 0) ∧ is_hyperbola k) :=
sorry

end NUMINAMATH_CALUDE_k_negative_sufficient_not_necessary_l1592_159209


namespace NUMINAMATH_CALUDE_calculate_fraction_product_l1592_159262

theorem calculate_fraction_product : 
  let mixed_number : ℚ := 3 + 3/4
  let decimal_one : ℚ := 0.2
  let whole_number : ℕ := 135
  let decimal_two : ℚ := 5.4
  ((mixed_number * decimal_one) / whole_number) * decimal_two = 0.03 := by
sorry

end NUMINAMATH_CALUDE_calculate_fraction_product_l1592_159262
