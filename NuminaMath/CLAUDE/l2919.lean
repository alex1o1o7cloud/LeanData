import Mathlib

namespace right_triangle_perimeter_area_l2919_291966

theorem right_triangle_perimeter_area (a c : ℝ) :
  a > 0 ∧ c > 0 ∧  -- Positive sides
  c > a ∧  -- Hypotenuse is longest side
  Real.sqrt (c - 5) + 2 * Real.sqrt (10 - 2*c) = a - 4 →  -- Given equation
  ∃ b : ℝ, 
    b > 0 ∧  -- Positive side
    a^2 + b^2 = c^2 ∧  -- Pythagorean theorem
    a + b + c = 12 ∧  -- Perimeter
    (1/2) * a * b = 6  -- Area
  := by sorry

end right_triangle_perimeter_area_l2919_291966


namespace soda_lasts_40_days_l2919_291996

/-- The number of days soda bottles last given the initial quantity and daily consumption rate -/
def soda_duration (total_bottles : ℕ) (daily_consumption : ℕ) : ℕ :=
  total_bottles / daily_consumption

theorem soda_lasts_40_days :
  soda_duration 360 9 = 40 := by
  sorry

end soda_lasts_40_days_l2919_291996


namespace unique_solution_for_equation_l2919_291974

theorem unique_solution_for_equation : ∃! (m n : ℕ), m^m + (m*n)^n = 1984 ∧ m = 4 ∧ n = 3 := by
  sorry

end unique_solution_for_equation_l2919_291974


namespace range_of_a_l2919_291947

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}
def B : Set ℝ := {x | x^2 - 6*x + 5 < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (A a ∩ B = ∅) → (a ≥ 6 ∨ a ≤ 0) := by
  sorry

end range_of_a_l2919_291947


namespace perimeter_difference_l2919_291942

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : Real
  width : Real

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : Real :=
  2 * (r.length + r.width)

/-- Represents a cutting configuration for the plywood -/
structure CuttingConfig where
  piece : Rectangle
  num_pieces : Nat

/-- The original plywood dimensions -/
def plywood : Rectangle :=
  { length := 10, width := 6 }

/-- The number of pieces to cut the plywood into -/
def num_pieces : Nat := 6

/-- Checks if a cutting configuration is valid for the given plywood -/
def is_valid_config (config : CuttingConfig) : Prop :=
  config.num_pieces = num_pieces ∧
  config.piece.length * config.piece.width * config.num_pieces = plywood.length * plywood.width

/-- Theorem stating the difference between max and min perimeter -/
theorem perimeter_difference :
  ∃ (max_config min_config : CuttingConfig),
    is_valid_config max_config ∧
    is_valid_config min_config ∧
    (∀ c : CuttingConfig, is_valid_config c →
      perimeter c.piece ≤ perimeter max_config.piece ∧
      perimeter c.piece ≥ perimeter min_config.piece) ∧
    perimeter max_config.piece - perimeter min_config.piece = 11.34 := by
  sorry

end perimeter_difference_l2919_291942


namespace arithmetic_sequence_properties_l2919_291975

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_properties (a₁ : ℝ) (d : ℝ) (h : d > 0) :
  (∀ n m : ℕ, n < m → arithmetic_sequence a₁ d n < arithmetic_sequence a₁ d m) ∧
  (∀ n m : ℕ, n < m → arithmetic_sequence a₁ d n + 3 * n * d < arithmetic_sequence a₁ d m + 3 * m * d) ∧
  (∃ a₁ d : ℝ, d > 0 ∧ ∃ n m : ℕ, n < m ∧ n * (arithmetic_sequence a₁ d n) ≥ m * (arithmetic_sequence a₁ d m)) ∧
  (∃ a₁ d : ℝ, d > 0 ∧ ∃ n m : ℕ, n < m ∧ arithmetic_sequence a₁ d n / n ≤ arithmetic_sequence a₁ d m / m) :=
by sorry

end arithmetic_sequence_properties_l2919_291975


namespace zeros_in_square_of_nines_zeros_count_in_2019_nines_squared_l2919_291951

theorem zeros_in_square_of_nines (n : ℕ) : 
  (10^n - 1)^2 % 10^(n-1) = 1 ∧ (10^n - 1)^2 % 10^n ≠ 0 := by
  sorry

theorem zeros_count_in_2019_nines_squared : 
  ∃ k : ℕ, (10^2019 - 1)^2 = k * 10^2018 + 1 ∧ k % 10 ≠ 0 := by
  sorry

end zeros_in_square_of_nines_zeros_count_in_2019_nines_squared_l2919_291951


namespace triangle_angle_and_perimeter_l2919_291908

def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_angle_and_perimeter 
  (a b c : ℝ) (A B C : ℝ) :
  triangle a b c →
  a > 2 →
  b - c = 1 →
  Real.sqrt 3 * a * Real.cos C = c * Real.sin A →
  (C = Real.pi / 3 ∧
   ∃ (p : ℝ), p = a + b + c ∧ p ≥ 9 + 6 * Real.sqrt 2 ∧
   ∀ (q : ℝ), q = a + b + c → q ≥ p) :=
by sorry

end triangle_angle_and_perimeter_l2919_291908


namespace cylinder_surface_area_l2919_291925

/-- The surface area of a cylinder with height 2 and base circumference 2π is 6π -/
theorem cylinder_surface_area :
  ∀ (h : ℝ) (c : ℝ),
  h = 2 →
  c = 2 * Real.pi →
  2 * Real.pi * (c / (2 * Real.pi)) * (c / (2 * Real.pi)) + c * h = 6 * Real.pi :=
by sorry

end cylinder_surface_area_l2919_291925


namespace audio_cassette_count_audio_cassette_count_proof_l2919_291905

/-- Proves that the number of audio cassettes in the first set is 7 given the problem conditions --/
theorem audio_cassette_count : ℕ :=
  let video_cost : ℕ := 300
  let some_audio_and_3_video_cost : ℕ := 1110
  let five_audio_and_4_video_cost : ℕ := 1350
  7

theorem audio_cassette_count_proof :
  let video_cost : ℕ := 300
  let some_audio_and_3_video_cost : ℕ := 1110
  let five_audio_and_4_video_cost : ℕ := 1350
  ∃ (audio_cost : ℕ) (first_set_count : ℕ),
    first_set_count * audio_cost + 3 * video_cost = some_audio_and_3_video_cost ∧
    5 * audio_cost + 4 * video_cost = five_audio_and_4_video_cost ∧
    first_set_count = audio_cassette_count :=
by
  sorry

end audio_cassette_count_audio_cassette_count_proof_l2919_291905


namespace smallest_value_w3_plus_z3_l2919_291998

theorem smallest_value_w3_plus_z3 (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 18) :
  Complex.abs (w^3 + z^3) ≥ 50 := by
  sorry

end smallest_value_w3_plus_z3_l2919_291998


namespace average_temperature_calculation_l2919_291921

/-- Given the average temperature for four consecutive days and the temperatures of the first and last days, calculate the average temperature for the last four days. -/
theorem average_temperature_calculation 
  (temp_mon : ℝ) 
  (temp_fri : ℝ) 
  (avg_mon_to_thu : ℝ) 
  (h1 : temp_mon = 41)
  (h2 : temp_fri = 33)
  (h3 : avg_mon_to_thu = 48) :
  (4 * avg_mon_to_thu - temp_mon + temp_fri) / 4 = 46 := by
  sorry

end average_temperature_calculation_l2919_291921


namespace anil_tomato_production_l2919_291917

/-- Represents the number of tomatoes in a square backyard -/
def TomatoCount (side : ℕ) : ℕ := side * side

/-- Proves that given the conditions of Anil's tomato garden, he produced 4356 tomatoes this year -/
theorem anil_tomato_production : 
  ∃ (last_year current_year : ℕ),
    TomatoCount current_year = TomatoCount last_year + 131 ∧
    current_year > last_year ∧
    TomatoCount current_year = 4356 := by
  sorry


end anil_tomato_production_l2919_291917


namespace equation_solution_l2919_291980

theorem equation_solution : 
  let f (x : ℝ) := (x^2 - 3*x + 2) * (x^2 + 3*x - 2)
  let g (x : ℝ) := x^2 * (x + 3) * (x - 3)
  f (1/3) = g (1/3) := by sorry

end equation_solution_l2919_291980


namespace element_in_set_l2919_291946

def M : Set (ℕ × ℕ) := {(1, 2)}

theorem element_in_set : (1, 2) ∈ M := by
  sorry

end element_in_set_l2919_291946


namespace annas_remaining_money_l2919_291972

/-- Given Anna's initial amount and her purchases, calculate the amount left --/
theorem annas_remaining_money (initial_amount : ℚ) 
  (gum_price choc_price cane_price : ℚ)
  (gum_quantity choc_quantity cane_quantity : ℕ) : 
  initial_amount = 10 →
  gum_price = 1 →
  choc_price = 1 →
  cane_price = 1/2 →
  gum_quantity = 3 →
  choc_quantity = 5 →
  cane_quantity = 2 →
  initial_amount - (gum_price * gum_quantity + choc_price * choc_quantity + cane_price * cane_quantity) = 1 := by
  sorry

end annas_remaining_money_l2919_291972


namespace wage_productivity_relationship_l2919_291997

/-- Represents the regression line equation for worker's wage and labor productivity -/
def regression_line (x : ℝ) : ℝ := 50 + 80 * x

/-- Theorem stating the relationship between changes in labor productivity and worker's wage -/
theorem wage_productivity_relationship :
  ∀ x : ℝ, regression_line (x + 1) - regression_line x = 80 := by
  sorry

end wage_productivity_relationship_l2919_291997


namespace operation_laws_l2919_291983

theorem operation_laws (a b : ℝ) :
  ((25 * b) * 8 = b * (25 * 8)) ∧
  (a * 6 + 6 * 15 = 6 * (a + 15)) ∧
  (1280 / 16 / 8 = 1280 / (16 * 8)) := by
  sorry

end operation_laws_l2919_291983


namespace slab_rate_per_square_meter_l2919_291960

theorem slab_rate_per_square_meter 
  (length : ℝ) 
  (width : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : total_cost = 14437.5) : 
  total_cost / (length * width) = 700 :=
by sorry

end slab_rate_per_square_meter_l2919_291960


namespace factor_expression_l2919_291941

theorem factor_expression (x : ℝ) : 5*x*(x-2) + 9*(x-2) - 4*(x-2) = 5*(x-2)*(x+1) := by
  sorry

end factor_expression_l2919_291941


namespace johnson_work_completion_l2919_291913

/-- Johnson and Vincent's Work Completion Problem -/
theorem johnson_work_completion (vincent_days : ℕ) (together_days : ℕ) (johnson_days : ℕ) : 
  vincent_days = 40 → together_days = 8 → johnson_days = 10 →
  (1 : ℚ) / johnson_days + (1 : ℚ) / vincent_days = (1 : ℚ) / together_days := by
  sorry

#check johnson_work_completion

end johnson_work_completion_l2919_291913


namespace unique_solution_condition_l2919_291973

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = b * x + 4) ↔ b ≠ 4 := by sorry

end unique_solution_condition_l2919_291973


namespace point_coordinates_sum_l2919_291902

/-- Given two points C and D, where C is at the origin and D is on the line y = 6,
    if the slope of CD is 3/4, then the sum of D's coordinates is 14. -/
theorem point_coordinates_sum (x : ℝ) : 
  let C : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (x, 6)
  (6 - 0) / (x - 0) = 3 / 4 →
  x + 6 = 14 := by
sorry

end point_coordinates_sum_l2919_291902


namespace work_completion_theorem_l2919_291964

/-- Given that 36 men can complete a piece of work in 18 days,
    and a smaller group can complete the same work in 72 days,
    prove that the smaller group consists of 9 men. -/
theorem work_completion_theorem :
  ∀ (total_work : ℕ) (smaller_group : ℕ),
  total_work = 36 * 18 →
  total_work = smaller_group * 72 →
  smaller_group = 9 :=
by
  sorry

end work_completion_theorem_l2919_291964


namespace range_of_a_l2919_291953

/-- Given real numbers a, b, c satisfying a system of equations, 
    prove that the range of values for a is [1, 9]. -/
theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  a ∈ Set.Icc 1 9 := by
  sorry


end range_of_a_l2919_291953


namespace correct_factorization_l2919_291936

theorem correct_factorization (a b : ℝ) : a * (a - b) - b * (b - a) = (a - b) * (a + b) := by
  sorry

end correct_factorization_l2919_291936


namespace altitudes_intersect_at_one_point_l2919_291954

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of an acute triangle --/
def isAcute (t : Triangle) : Prop := sorry

/-- Definition of an altitude of a triangle --/
def altitude (t : Triangle) (v : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The orthocenter of a triangle --/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem: The three altitudes of an acute triangle intersect at one point --/
theorem altitudes_intersect_at_one_point (t : Triangle) (h : isAcute t) :
  ∃! p : ℝ × ℝ, p ∈ altitude t t.A ∩ altitude t t.B ∩ altitude t t.C :=
sorry

end altitudes_intersect_at_one_point_l2919_291954


namespace function_difference_l2919_291912

theorem function_difference (f : ℝ → ℝ) (h : ∀ x, f x = 8^x) :
  ∀ x, f (x + 1) - f x = 7 * f x := by
sorry

end function_difference_l2919_291912


namespace race_outcomes_count_l2919_291982

-- Define the number of participants
def num_participants : ℕ := 7

-- Define a function to calculate the number of permutations
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

-- Define a function to calculate the number of combinations
def combinations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem race_outcomes_count :
  (3 * combinations (num_participants - 1) 2 * permutations 2 2) = 90 := by
  sorry

end race_outcomes_count_l2919_291982


namespace box_dimensions_l2919_291986

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  h₁ : 0 < a₁
  h₂ : 0 < a₂
  h₃ : 0 < a₃
  h₄ : a₁ ≤ a₂
  h₅ : a₂ ≤ a₃

/-- The volume of a cube -/
def cubeVolume : ℝ := 2

/-- The proportion of the box filled by cubes -/
def fillProportion : ℝ := 0.4

/-- Checks if the given dimensions satisfy the cube-filling condition -/
def satisfiesCubeFilling (d : BoxDimensions) : Prop :=
  ∃ (n : ℕ), n * cubeVolume = fillProportion * (d.a₁ * d.a₂ * d.a₃)

/-- The theorem stating the possible box dimensions -/
theorem box_dimensions : 
  ∀ d : BoxDimensions, satisfiesCubeFilling d → 
    (d.a₁ = 2 ∧ d.a₂ = 3 ∧ d.a₃ = 5) ∨ (d.a₁ = 2 ∧ d.a₂ = 5 ∧ d.a₃ = 6) := by
  sorry

end box_dimensions_l2919_291986


namespace subtraction_of_negatives_l2919_291979

theorem subtraction_of_negatives : -14 - (-26) = 12 := by
  sorry

end subtraction_of_negatives_l2919_291979


namespace distinct_tower_heights_94_bricks_l2919_291900

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of distinct tower heights possible -/
def distinctTowerHeights (brickCount : ℕ) (dimensions : BrickDimensions) : ℕ :=
  let maxY := 4
  List.range (maxY + 1)
    |> List.map (fun y => brickCount - y + 1)
    |> List.sum

/-- Theorem stating the number of distinct tower heights -/
theorem distinct_tower_heights_94_bricks :
  let brickDimensions : BrickDimensions := ⟨4, 10, 19⟩
  distinctTowerHeights 94 brickDimensions = 465 := by
  sorry

#eval distinctTowerHeights 94 ⟨4, 10, 19⟩

end distinct_tower_heights_94_bricks_l2919_291900


namespace equation_solution_l2919_291924

theorem equation_solution (y : ℚ) : (4 * y - 2) / (5 * y - 5) = 3 / 4 → y = -7 := by
  sorry

end equation_solution_l2919_291924


namespace diophantine_equation_solution_l2919_291916

theorem diophantine_equation_solution :
  ∀ x y z : ℕ, x^5 + x^4 + 1 = 3^y * 7^z ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 2) := by
  sorry

end diophantine_equation_solution_l2919_291916


namespace equation_solution_l2919_291939

theorem equation_solution :
  ∃ x : ℝ, x - 15 = 30 ∧ x = 45 := by
sorry

end equation_solution_l2919_291939


namespace valid_n_characterization_l2919_291999

def is_valid_n (n : ℕ) : Prop :=
  ∃ (k : ℤ), (37.5^n + 26.5^n : ℝ) = k ∧ k > 0

theorem valid_n_characterization :
  ∀ n : ℕ, is_valid_n n ↔ n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 :=
by sorry

end valid_n_characterization_l2919_291999


namespace inscribed_polyhedron_volume_relation_l2919_291957

/-- A polyhedron with an inscribed sphere -/
structure InscribedPolyhedron where
  -- The volume of the polyhedron
  volume : ℝ
  -- The radius of the inscribed sphere
  sphereRadius : ℝ
  -- The surface area of the polyhedron
  surfaceArea : ℝ
  -- Assumption that the sphere is inscribed in the polyhedron
  isInscribed : Prop
  -- Assumption that the polyhedron can be decomposed into pyramids
  canDecompose : Prop
  -- Assumption that each pyramid has a face as base and sphere center as apex
  pyramidProperty : Prop

/-- Theorem stating the volume relation for a polyhedron with an inscribed sphere -/
theorem inscribed_polyhedron_volume_relation (p : InscribedPolyhedron) :
  p.volume = (1 / 3) * p.sphereRadius * p.surfaceArea := by
  sorry

end inscribed_polyhedron_volume_relation_l2919_291957


namespace angle_implies_x_min_value_of_f_l2919_291978

noncomputable section

def a : ℝ × ℝ := (Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

def angle_between (u v : ℝ × ℝ) : ℝ := Real.arccos ((u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2)))

def f (x : ℝ) : ℝ := a.1 * (b x).1 + a.2 * (b x).2

theorem angle_implies_x (x : ℝ) (h : x ∈ Set.Ioo 0 (Real.pi / 2)) :
  angle_between a (b x) = Real.pi / 3 → x = 5 * Real.pi / 12 := by sorry

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (Real.pi / 2) ∧ ∀ (y : ℝ), y ∈ Set.Ioo 0 (Real.pi / 2) → f x ≤ f y ∧ f x = -Real.sqrt 2 / 2 := by sorry

end angle_implies_x_min_value_of_f_l2919_291978


namespace flyers_left_proof_l2919_291956

/-- The number of flyers left after Jack and Rose hand out some flyers -/
def flyers_left (initial : ℕ) (jack_handed : ℕ) (rose_handed : ℕ) : ℕ :=
  initial - (jack_handed + rose_handed)

/-- Proof that given 1,236 initial flyers, with Jack handing out 120 flyers
    and Rose handing out 320 flyers, the number of flyers left is 796 -/
theorem flyers_left_proof :
  flyers_left 1236 120 320 = 796 := by
  sorry

end flyers_left_proof_l2919_291956


namespace tessa_apples_for_pie_l2919_291934

def apples_needed_for_pie (initial_apples : ℕ) (received_apples : ℕ) (required_apples : ℕ) : ℕ :=
  max (required_apples - (initial_apples + received_apples)) 0

theorem tessa_apples_for_pie :
  apples_needed_for_pie 4 5 10 = 1 := by
  sorry

end tessa_apples_for_pie_l2919_291934


namespace tangent_line_intersection_l2919_291943

open Real

theorem tangent_line_intersection (x₀ : ℝ) : 
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ 
    (1/2 * x₀^2 - log m = x₀ * (x₀ - m)) ∧ 
    (1/(2*m) = x₀)) →
  (Real.sqrt 3 < x₀ ∧ x₀ < 2) :=
sorry

end tangent_line_intersection_l2919_291943


namespace paper_cranes_count_l2919_291930

theorem paper_cranes_count (T : ℕ) : 
  (T / 2 : ℚ) - (T / 2 : ℚ) / 5 = 400 → T = 1000 := by
  sorry

end paper_cranes_count_l2919_291930


namespace unique_sum_of_squares_l2919_291948

def is_sum_of_squares (n : ℕ) (k : ℕ) : Prop :=
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℕ), n = a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 ∧ (a₁ ≠ 0 → k ≥ 1) ∧
    (a₂ ≠ 0 → k ≥ 2) ∧ (a₃ ≠ 0 → k ≥ 3) ∧ (a₄ ≠ 0 → k ≥ 4) ∧ (a₅ ≠ 0 → k = 5)

def has_unique_representation (n : ℕ) : Prop :=
  ∃! (k : ℕ) (a₁ a₂ a₃ a₄ a₅ : ℕ), k ≤ 5 ∧ is_sum_of_squares n k ∧
    n = a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2

theorem unique_sum_of_squares :
  {n : ℕ | has_unique_representation n} = {1, 2, 3, 6, 7, 15} := by sorry

end unique_sum_of_squares_l2919_291948


namespace position_relationships_complete_l2919_291901

-- Define the type for position relationships
inductive PositionRelationship
  | Intersection
  | Parallel
  | Skew

-- Define a type for straight lines in 3D space
structure Line3D where
  -- We don't need to specify the internal structure of Line3D for this statement

-- Define the function that determines the position relationship between two lines
noncomputable def positionRelationship (l1 l2 : Line3D) : PositionRelationship :=
  sorry

-- Theorem statement
theorem position_relationships_complete (l1 l2 : Line3D) :
  ∃ (r : PositionRelationship), positionRelationship l1 l2 = r :=
sorry

end position_relationships_complete_l2919_291901


namespace mechanic_work_hours_l2919_291949

/-- Proves that a mechanic works 8 hours a day given the specified conditions -/
theorem mechanic_work_hours 
  (hourly_rate : ℕ) 
  (days_worked : ℕ) 
  (parts_cost : ℕ) 
  (total_paid : ℕ) 
  (h : hourly_rate = 60)
  (d : days_worked = 14)
  (p : parts_cost = 2500)
  (t : total_paid = 9220) :
  ∃ (hours_per_day : ℕ), 
    hours_per_day = 8 ∧ 
    hourly_rate * hours_per_day * days_worked + parts_cost = total_paid :=
by sorry

end mechanic_work_hours_l2919_291949


namespace mower_blades_cost_l2919_291918

def total_earned : ℕ := 104
def num_games : ℕ := 7
def game_price : ℕ := 9

theorem mower_blades_cost (remaining : ℕ) 
  (h1 : remaining = num_games * game_price) 
  (h2 : remaining + (total_earned - remaining) = total_earned) : 
  total_earned - remaining = 41 := by
  sorry

end mower_blades_cost_l2919_291918


namespace age_problem_l2919_291920

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 27 → 
  b = 10 := by
sorry

end age_problem_l2919_291920


namespace souvenir_discount_equation_l2919_291963

/-- Proves that for a souvenir with given original and final prices after two consecutive discounts, 
    the equation relating these prices and the discount percentage is correct. -/
theorem souvenir_discount_equation (a : ℝ) : 
  let original_price : ℝ := 168
  let final_price : ℝ := 128
  original_price * (1 - a / 100)^2 = final_price := by
  sorry

end souvenir_discount_equation_l2919_291963


namespace sum_difference_even_odd_l2919_291990

/-- Sum of the first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- Sum of the first n positive odd integers -/
def sumFirstOddIntegers (n : ℕ) : ℕ := n * n

/-- The positive difference between the sum of the first 25 positive even integers
    and the sum of the first 20 positive odd integers is 250 -/
theorem sum_difference_even_odd : 
  (sumFirstEvenIntegers 25 : ℤ) - (sumFirstOddIntegers 20 : ℤ) = 250 := by sorry

end sum_difference_even_odd_l2919_291990


namespace project_duration_is_four_days_l2919_291988

/-- Calculates the number of days taken to finish a project given the number of naps, hours per nap, and working hours. -/
def projectDuration (numNaps : ℕ) (hoursPerNap : ℕ) (workingHours : ℕ) : ℚ :=
  let totalHours := numNaps * hoursPerNap + workingHours
  totalHours / 24

/-- Theorem stating that under the given conditions, the project duration is 4 days. -/
theorem project_duration_is_four_days :
  projectDuration 6 7 54 = 4 := by
  sorry

#eval projectDuration 6 7 54

end project_duration_is_four_days_l2919_291988


namespace merry_lambs_l2919_291984

theorem merry_lambs (merry_lambs : ℕ) (brother_lambs : ℕ) : 
  brother_lambs = merry_lambs + 3 →
  merry_lambs + brother_lambs = 23 →
  merry_lambs = 10 := by
sorry

end merry_lambs_l2919_291984


namespace multiplication_problem_l2919_291923

theorem multiplication_problem (x : ℝ) (n : ℝ) (h1 : x = 13) (h2 : x * n = (36 - x) + 16) : n = 3 := by
  sorry

end multiplication_problem_l2919_291923


namespace range_of_a_l2919_291903

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 4 := by sorry

end range_of_a_l2919_291903


namespace simplify_fraction_l2919_291950

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) : (x^2 / (x - 2)) - (2*x / (x - 2)) = x := by
  sorry

end simplify_fraction_l2919_291950


namespace smallest_prime_divisor_of_sum_l2919_291940

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : Nat), Prime p ∧ p ∣ (2^12 + 3^14 + 7^4) ∧ ∀ (q : Nat), Prime q → q ∣ (2^12 + 3^14 + 7^4) → p ≤ q :=
by sorry

end smallest_prime_divisor_of_sum_l2919_291940


namespace circle_passes_through_points_l2919_291931

def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 4*x - 6*y

theorem circle_passes_through_points :
  (circle_equation 0 0 = 0) ∧
  (circle_equation 4 0 = 0) ∧
  (circle_equation (-1) 1 = 0) :=
by sorry

#check circle_passes_through_points

end circle_passes_through_points_l2919_291931


namespace triangle_third_side_l2919_291965

theorem triangle_third_side (a b c : ℕ) : 
  (a - b = 7 ∨ b - a = 7) →  -- difference between two sides is 7
  (a + b + c) % 2 = 1 →      -- perimeter is odd
  c = 8                      -- third side is 8
:= by sorry

end triangle_third_side_l2919_291965


namespace point_coordinates_l2919_291969

theorem point_coordinates (x y : ℝ) : 
  (|y| = (1/2) * |x|) → -- distance from x-axis is half the distance from y-axis
  (|x| = 10) →          -- point is 10 units from y-axis
  (y = 5 ∨ y = -5) :=   -- y-coordinate is either 5 or -5
by
  sorry

end point_coordinates_l2919_291969


namespace quadratic_symmetry_axis_l2919_291989

theorem quadratic_symmetry_axis 
  (a b c : ℝ) 
  (ha : a ≠ 0)
  (h1 : a * (0 + 4)^2 + b * (0 + 4) + c = 0)
  (h2 : a * (0 - 1)^2 + b * (0 - 1) + c = 0) :
  -b / (2 * a) = 1.5 := by sorry

end quadratic_symmetry_axis_l2919_291989


namespace f_inv_is_inverse_unique_solution_is_three_l2919_291967

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 9

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

-- Theorem stating that f_inv is indeed the inverse of f
theorem f_inv_is_inverse : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x := by sorry

-- Main theorem
theorem unique_solution_is_three :
  ∃! x : ℝ, f x = f_inv x := by sorry

end f_inv_is_inverse_unique_solution_is_three_l2919_291967


namespace abs_x_eq_x_necessary_not_sufficient_l2919_291971

theorem abs_x_eq_x_necessary_not_sufficient :
  (∀ x : ℝ, |x| = x → x^2 ≥ -x) ∧
  ¬(∀ x : ℝ, x^2 ≥ -x → |x| = x) := by
  sorry

end abs_x_eq_x_necessary_not_sufficient_l2919_291971


namespace tan_Y_in_right_triangle_l2919_291944

theorem tan_Y_in_right_triangle (Y : Real) (opposite hypotenuse : ℝ) 
  (h1 : opposite = 8)
  (h2 : hypotenuse = 17)
  (h3 : 0 < opposite)
  (h4 : opposite < hypotenuse) :
  Real.tan Y = 8 / 15 := by
  sorry

end tan_Y_in_right_triangle_l2919_291944


namespace blue_pill_cost_proof_l2919_291994

def total_cost : ℚ := 430
def days : ℕ := 10
def blue_red_diff : ℚ := 3

def blue_pill_cost : ℚ := 23

theorem blue_pill_cost_proof :
  (blue_pill_cost * days + (blue_pill_cost - blue_red_diff) * days = total_cost) ∧
  (blue_pill_cost > 0) ∧
  (blue_pill_cost - blue_red_diff > 0) :=
by sorry

end blue_pill_cost_proof_l2919_291994


namespace mayoral_election_vote_ratio_l2919_291958

theorem mayoral_election_vote_ratio :
  let votes_Z : ℕ := 25000
  let votes_X : ℕ := 22500
  let votes_Y : ℕ := (3 * votes_Z) / 5
  (votes_X - votes_Y) * 2 = votes_Y := by
  sorry

end mayoral_election_vote_ratio_l2919_291958


namespace elijah_card_count_l2919_291910

/-- The number of cards in a standard deck of playing cards -/
def cards_per_deck : ℕ := 52

/-- The number of decks Elijah has -/
def number_of_decks : ℕ := 6

/-- The total number of cards Elijah has -/
def total_cards : ℕ := number_of_decks * cards_per_deck

theorem elijah_card_count : total_cards = 312 := by
  sorry

end elijah_card_count_l2919_291910


namespace or_proposition_true_l2919_291985

theorem or_proposition_true : 
  let p : Prop := 3 > 4
  let q : Prop := 3 < 4
  p ∨ q := by
  sorry

end or_proposition_true_l2919_291985


namespace a_b_reciprocal_l2919_291976

theorem a_b_reciprocal (a b : ℚ) : 
  (-7/8) / (7/4 - 7/8 - 7/12) = a →
  (7/4 - 7/8 - 7/12) / (-7/8) = b →
  a = -1 / b :=
by
  sorry

end a_b_reciprocal_l2919_291976


namespace decagon_diagonal_intersection_probability_l2919_291938

/-- A regular decagon -/
structure RegularDecagon where
  -- Add necessary properties here

/-- The number of diagonals in a regular decagon -/
def num_diagonals (d : RegularDecagon) : ℕ := 35

/-- The number of ways to choose 3 diagonals from a regular decagon -/
def num_diagonal_trios (d : RegularDecagon) : ℕ := 6545

/-- The number of ways to choose 5 points from a regular decagon such that no three points are consecutive -/
def num_valid_point_sets (d : RegularDecagon) : ℕ := 252

/-- The probability that three randomly chosen diagonals of a regular decagon intersect inside the decagon -/
def intersection_probability (d : RegularDecagon) : ℚ :=
  num_valid_point_sets d / num_diagonal_trios d

theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  intersection_probability d = 252 / 6545 := by
  sorry


end decagon_diagonal_intersection_probability_l2919_291938


namespace hyperbola_sum_l2919_291907

theorem hyperbola_sum (h k a b c : ℝ) : 
  (h = 0 ∧ k = 0) →  -- center at (0,0)
  c = 8 →            -- focus at (0,8)
  a = 4 →            -- vertex at (0,-4)
  c^2 = a^2 + b^2 →  -- relationship between a, b, and c
  h + k + a + b = 4 + 4 * Real.sqrt 3 := by
sorry

end hyperbola_sum_l2919_291907


namespace additional_grazing_area_l2919_291927

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 23^2 - π * 9^2 = 448 * π := by
  sorry

end additional_grazing_area_l2919_291927


namespace right_triangle_leg_length_l2919_291959

theorem right_triangle_leg_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a = 8 → c = 17 →
  b = 15 := by
sorry

end right_triangle_leg_length_l2919_291959


namespace earliest_82_degrees_l2919_291995

/-- The temperature function modeling the temperature in Denver, CO -/
def temperature (t : ℝ) : ℝ := -2 * t^2 + 16 * t + 40

/-- Theorem stating that the earliest non-negative time when the temperature reaches 82 degrees is 3 hours past noon -/
theorem earliest_82_degrees :
  ∀ t : ℝ, t ≥ 0 → temperature t = 82 → t ≥ 3 := by
  sorry

end earliest_82_degrees_l2919_291995


namespace range_of_m_l2919_291904

-- Define the set A as the solution set of |x+m| ≤ 4
def A (m : ℝ) : Set ℝ := {x : ℝ | |x + m| ≤ 4}

-- Define the theorem
theorem range_of_m :
  (∀ m : ℝ, A m ⊆ {x : ℝ | -2 ≤ x ∧ x ≤ 8}) →
  {m : ℝ | ∃ x : ℝ, x ∈ A m} = {m : ℝ | -4 ≤ m ∧ m ≤ -2} :=
by sorry

end range_of_m_l2919_291904


namespace books_shop1_is_65_l2919_291919

-- Define the problem parameters
def total_spent_shop1 : ℕ := 6500
def books_shop2 : ℕ := 35
def total_spent_shop2 : ℕ := 2000
def avg_price : ℕ := 85

-- Define the function to calculate the number of books from the first shop
def books_shop1 : ℕ := 
  (total_spent_shop1 + total_spent_shop2) / avg_price - books_shop2

-- Theorem to prove
theorem books_shop1_is_65 : books_shop1 = 65 := by
  sorry

end books_shop1_is_65_l2919_291919


namespace unique_solution_iff_prime_l2919_291952

theorem unique_solution_iff_prime (n : ℕ+) :
  (∃! a : ℕ, a < n.val.factorial ∧ (n.val.factorial ∣ a^n.val + 1)) ↔ Nat.Prime n.val := by
  sorry

end unique_solution_iff_prime_l2919_291952


namespace problem_statement_l2919_291955

theorem problem_statement (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : c > 0) :
  (a^2 - b*c > b^2 - a*c) ∧ (a^3 > b^2) ∧ (a + 1/a > b + 1/b) := by
  sorry

end problem_statement_l2919_291955


namespace swimming_laps_per_day_l2919_291932

/-- Proves that swimming 300 laps in 5 weeks, 5 days per week, results in 12 laps per day -/
theorem swimming_laps_per_day 
  (total_laps : ℕ) 
  (weeks : ℕ) 
  (days_per_week : ℕ) 
  (h1 : total_laps = 300) 
  (h2 : weeks = 5) 
  (h3 : days_per_week = 5) : 
  total_laps / (weeks * days_per_week) = 12 := by
  sorry

end swimming_laps_per_day_l2919_291932


namespace positive_expression_l2919_291970

theorem positive_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + 3 * b^2 > 0 := by
  sorry

end positive_expression_l2919_291970


namespace ratio_problem_l2919_291922

/-- Given two numbers in a 15:1 ratio where the first number is 150, prove that the second number is 10. -/
theorem ratio_problem (a b : ℝ) (h1 : a / b = 15) (h2 : a = 150) : b = 10 := by
  sorry

end ratio_problem_l2919_291922


namespace garrison_provision_days_l2919_291929

/-- Calculates the initial number of days provisions were supposed to last for a garrison -/
def initialProvisionDays (initialGarrison : ℕ) (reinforcement : ℕ) (daysBeforeReinforcement : ℕ) (daysAfterReinforcement : ℕ) : ℕ :=
  ((initialGarrison + reinforcement) * daysAfterReinforcement + initialGarrison * daysBeforeReinforcement) / initialGarrison

theorem garrison_provision_days :
  initialProvisionDays 1000 1250 15 20 = 60 := by
  sorry

end garrison_provision_days_l2919_291929


namespace vision_data_median_l2919_291981

/-- Represents the vision data for a class of students -/
def VisionData : List (Float × Nat) := [
  (4.0, 1), (4.1, 2), (4.2, 6), (4.3, 3), (4.4, 3),
  (4.5, 4), (4.6, 1), (4.7, 2), (4.8, 5), (4.9, 7), (5.0, 5)
]

/-- The total number of students -/
def totalStudents : Nat := 39

/-- Calculates the median of the vision data -/
def median (data : List (Float × Nat)) (total : Nat) : Float :=
  sorry

/-- Theorem stating that the median of the given vision data is 4.6 -/
theorem vision_data_median : median VisionData totalStudents = 4.6 := by
  sorry

end vision_data_median_l2919_291981


namespace therapy_hours_is_five_l2919_291933

/-- Represents the cost structure and billing for a psychologist's therapy sessions. -/
structure TherapyCost where
  firstHourCost : ℕ
  additionalHourCost : ℕ
  firstHourPremium : ℕ
  twoHourTotal : ℕ
  someHoursTotal : ℕ

/-- Calculates the number of therapy hours given the cost structure and total charge. -/
def calculateTherapyHours (cost : TherapyCost) : ℕ :=
  sorry

/-- Theorem stating that given the specific cost structure, the calculated therapy hours is 5. -/
theorem therapy_hours_is_five (cost : TherapyCost)
  (h1 : cost.firstHourCost = cost.additionalHourCost + cost.firstHourPremium)
  (h2 : cost.firstHourPremium = 25)
  (h3 : cost.twoHourTotal = 115)
  (h4 : cost.someHoursTotal = 250) :
  calculateTherapyHours cost = 5 :=
sorry

end therapy_hours_is_five_l2919_291933


namespace keychain_arrangements_l2919_291915

/-- The number of keys on the keychain -/
def total_keys : ℕ := 6

/-- The number of keys that must be adjacent -/
def adjacent_keys : ℕ := 3

/-- The number of distinct arrangements of the adjacent keys -/
def adjacent_arrangements : ℕ := Nat.factorial adjacent_keys

/-- The number of distinct arrangements of the remaining groups (adjacent group + other keys) -/
def group_arrangements : ℕ := Nat.factorial (total_keys - adjacent_keys + 1 - 1)

/-- The total number of distinct arrangements -/
def total_arrangements : ℕ := adjacent_arrangements * group_arrangements

theorem keychain_arrangements :
  total_arrangements = 36 :=
sorry

end keychain_arrangements_l2919_291915


namespace unique_solution_system_l2919_291945

/-- Given positive real numbers a, b, c satisfying √a + √b + √c = √π/2,
    prove that there exists a unique triple (x, y, z) of real numbers
    satisfying the system of equations:
    √(y-a) + √(z-a) = 1
    √(z-b) + √(x-b) = 1
    √(x-c) + √(y-c) = 1 -/
theorem unique_solution_system (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = Real.sqrt (π / 2)) :
  ∃! x y z : ℝ,
    Real.sqrt (y - a) + Real.sqrt (z - a) = 1 ∧
    Real.sqrt (z - b) + Real.sqrt (x - b) = 1 ∧
    Real.sqrt (x - c) + Real.sqrt (y - c) = 1 :=
by sorry


end unique_solution_system_l2919_291945


namespace wooden_easel_cost_l2919_291935

theorem wooden_easel_cost (paintbrush_cost paint_cost albert_has additional_needed : ℚ)
  (h1 : paintbrush_cost = 1.5)
  (h2 : paint_cost = 4.35)
  (h3 : albert_has = 6.5)
  (h4 : additional_needed = 12) :
  let total_cost := albert_has + additional_needed
  let other_items_cost := paintbrush_cost + paint_cost
  let easel_cost := total_cost - other_items_cost
  easel_cost = 12.65 := by sorry

end wooden_easel_cost_l2919_291935


namespace minimum_properties_l2919_291961

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (exp x - 1) + x

theorem minimum_properties {x₀ : ℝ} (h₀ : x₀ > 0) 
  (h₁ : ∀ x > 0, f x ≥ f x₀) : 
  f x₀ = x₀ + 1 ∧ f x₀ < 3 := by
  sorry

end minimum_properties_l2919_291961


namespace odd_square_minus_one_l2919_291937

theorem odd_square_minus_one (n : ℕ) : (2*n + 1)^2 - 1 = 4*n*(n + 1) := by
  sorry

end odd_square_minus_one_l2919_291937


namespace b_investment_is_4000_l2919_291987

/-- Represents the investment and profit distribution in a partnership --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- Theorem stating that under given conditions, B's investment is 4000 --/
theorem b_investment_is_4000 (p : Partnership)
  (h1 : p.a_investment = 8000)
  (h2 : p.c_investment = 2000)
  (h3 : p.total_profit = 252000)
  (h4 : p.c_profit_share = 36000)
  (h5 : p.c_investment * p.total_profit = p.c_profit_share * (p.a_investment + p.b_investment + p.c_investment)) :
  p.b_investment = 4000 := by
  sorry


end b_investment_is_4000_l2919_291987


namespace new_rectangle_area_l2919_291991

/-- Given a rectangle with sides 3 and 4, prove that a new rectangle
    formed with one side equal to the diagonal of the original rectangle
    and the other side equal to the sum of the original sides has an area of 35. -/
theorem new_rectangle_area (a b : ℝ) (ha : a = 3) (hb : b = 4) :
  let d := Real.sqrt (a^2 + b^2)
  let new_side_sum := a + b
  d * new_side_sum = 35 := by sorry

end new_rectangle_area_l2919_291991


namespace probability_of_cooking_l2919_291926

/-- The set of courses Xiao Ming is interested in -/
inductive Course
| Planting
| Cooking
| Pottery
| Woodworking

/-- The probability of selecting a specific course from the set of courses -/
def probability_of_course (c : Course) : ℚ :=
  1 / 4

/-- Theorem stating that the probability of selecting "Cooking" is 1/4 -/
theorem probability_of_cooking :
  probability_of_course Course.Cooking = 1 / 4 := by
  sorry

end probability_of_cooking_l2919_291926


namespace driver_net_rate_of_pay_l2919_291962

/-- Calculate the net rate of pay for a driver given specific conditions --/
theorem driver_net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 60)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_rate = 0.60)
  (h5 : gas_price = 2.50)
  : ∃ (net_rate : ℝ), net_rate = 30 :=
by
  sorry


end driver_net_rate_of_pay_l2919_291962


namespace smallest_third_term_of_geometric_progression_l2919_291928

-- Define the arithmetic progression
def arithmetic_progression (a d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmetic_progression a d n + d

-- Define the geometric progression
def geometric_progression (g_1 g_2 g_3 : ℝ) : Prop :=
  g_2 ^ 2 = g_1 * g_3

-- Theorem statement
theorem smallest_third_term_of_geometric_progression :
  ∀ d : ℝ,
  let a := arithmetic_progression 9 d
  let g_1 := 9
  let g_2 := a 1 + 5
  let g_3 := a 2 + 30
  geometric_progression g_1 g_2 g_3 →
  ∃ min_g_3 : ℝ, min_g_3 = 29 - 20 * Real.sqrt 2 ∧
  ∀ other_g_3 : ℝ, geometric_progression g_1 g_2 other_g_3 → min_g_3 ≤ other_g_3 :=
sorry

end smallest_third_term_of_geometric_progression_l2919_291928


namespace train_length_calculation_l2919_291977

/-- Calculates the length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60 →
  person_speed = 6 →
  passing_time = 13.090909090909092 →
  ∃ (train_length : ℝ), abs (train_length - 240) < 0.01 :=
by sorry

end train_length_calculation_l2919_291977


namespace factor_expression_l2919_291906

theorem factor_expression (b : ℝ) : 53 * b^2 + 159 * b = 53 * b * (b + 3) := by
  sorry

end factor_expression_l2919_291906


namespace imaginary_part_of_complex_fraction_l2919_291911

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (3 - 2 * Complex.I^3) / (1 + Complex.I)
  Complex.im z = -1/2 := by
  sorry

end imaginary_part_of_complex_fraction_l2919_291911


namespace multiple_of_a_share_l2919_291993

def total_sum : ℚ := 427
def c_share : ℚ := 84

theorem multiple_of_a_share : ∃ (a b : ℚ) (x : ℚ), 
  a + b + c_share = total_sum ∧ 
  x * a = 4 * b ∧ 
  x * a = 7 * c_share ∧
  x = 3 := by sorry

end multiple_of_a_share_l2919_291993


namespace marias_number_problem_l2919_291992

theorem marias_number_problem (n : ℚ) : 
  (((n + 3) * 3 - 2) / 3 = 10) → (n = 23 / 3) := by
  sorry

end marias_number_problem_l2919_291992


namespace lcm_gcd_problem_l2919_291909

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 21 → 
  a = 210 → 
  b = 462 := by
sorry

end lcm_gcd_problem_l2919_291909


namespace power_comparison_l2919_291968

theorem power_comparison : 2^1000 < 5^500 ∧ 5^500 < 3^750 := by
  sorry

end power_comparison_l2919_291968


namespace dumbbell_distribution_impossible_l2919_291914

def dumbbell_weights : List ℕ := [4, 5, 6, 9, 10, 11, 14, 19, 23, 24]

theorem dumbbell_distribution_impossible :
  ¬ ∃ (rack1 rack2 rack3 : List ℕ),
    (rack1 ++ rack2 ++ rack3).toFinset = dumbbell_weights.toFinset ∧
    (rack1.sum : ℚ) * 2 = rack2.sum ∧
    (rack2.sum : ℚ) * 2 = rack3.sum :=
by sorry

end dumbbell_distribution_impossible_l2919_291914
