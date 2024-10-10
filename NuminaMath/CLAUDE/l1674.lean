import Mathlib

namespace opponent_total_score_l1674_167470

def TeamScores : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def LostGames (scores : List ℕ) : List ℕ := 
  scores.filter (λ x => x % 2 = 1 ∧ x ≤ 13)

def WonGames (scores : List ℕ) (lostGames : List ℕ) : List ℕ :=
  scores.filter (λ x => x ∉ lostGames)

def OpponentScoresInLostGames (lostGames : List ℕ) : List ℕ :=
  lostGames.map (λ x => x + 1)

def OpponentScoresInWonGames (wonGames : List ℕ) : List ℕ :=
  wonGames.map (λ x => x / 2)

theorem opponent_total_score :
  let lostGames := LostGames TeamScores
  let wonGames := WonGames TeamScores lostGames
  let opponentLostScores := OpponentScoresInLostGames lostGames
  let opponentWonScores := OpponentScoresInWonGames wonGames
  (opponentLostScores.sum + opponentWonScores.sum) = 75 :=
sorry

end opponent_total_score_l1674_167470


namespace alyssas_turnips_l1674_167456

theorem alyssas_turnips (keith_turnips total_turnips : ℕ) 
  (h1 : keith_turnips = 6)
  (h2 : total_turnips = 15) :
  total_turnips - keith_turnips = 9 :=
by sorry

end alyssas_turnips_l1674_167456


namespace range_of_a_l1674_167446

-- Define the statements p and q
def p (a : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁^2 + (k*x₁ + 1)^2/a = 1) ∧ 
    (x₂^2 + (k*x₂ + 1)^2/a = 1)

def q (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, 4^x₀ - 2^x₀ - a ≤ 0

-- Theorem statement
theorem range_of_a :
  (∀ a : ℝ, ¬(p a ∧ q a)) ∧ (∀ a : ℝ, p a ∨ q a) →
  ∀ a : ℝ, -1/4 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l1674_167446


namespace last_three_digits_of_7_power_10000_l1674_167479

theorem last_three_digits_of_7_power_10000 (h : 7^500 ≡ 1 [ZMOD 1250]) :
  7^10000 ≡ 1 [ZMOD 1250] := by
  sorry

end last_three_digits_of_7_power_10000_l1674_167479


namespace equation_roots_l1674_167457

theorem equation_roots : 
  let S := {x : ℝ | 0 < x ∧ x < 1 ∧ 8 * x * (2 * x^2 - 1) * (8 * x^4 - 8 * x^2 + 1) = 1}
  S = {Real.cos (π / 9), Real.cos (π / 3), Real.cos (2 * π / 7)} := by sorry

end equation_roots_l1674_167457


namespace land_price_per_acre_l1674_167443

theorem land_price_per_acre (total_acres : ℕ) (num_lots : ℕ) (price_per_lot : ℕ) : 
  total_acres = 4 →
  num_lots = 9 →
  price_per_lot = 828 →
  (num_lots * price_per_lot) / total_acres = 1863 := by
sorry

end land_price_per_acre_l1674_167443


namespace probability_above_parabola_l1674_167463

/-- A single-digit positive integer -/
def SingleDigit := { n : ℕ | 1 ≤ n ∧ n ≤ 9 }

/-- The total number of possible (a, b) combinations -/
def TotalCombinations : ℕ := 81

/-- The number of valid (a, b) combinations where (a, b) lies above y = ax^2 + bx -/
def ValidCombinations : ℕ := 72

/-- The probability that a randomly chosen point (a, b) lies above y = ax^2 + bx -/
def ProbabilityAboveParabola : ℚ := ValidCombinations / TotalCombinations

theorem probability_above_parabola :
  ProbabilityAboveParabola = 8 / 9 := by sorry

end probability_above_parabola_l1674_167463


namespace inequality_system_solution_l1674_167429

theorem inequality_system_solution (m : ℝ) :
  (∀ x : ℝ, (3 * x - 9 > 0 ∧ x > m) ↔ x > 3) →
  m ≤ 3 :=
by sorry

end inequality_system_solution_l1674_167429


namespace focus_of_given_parabola_l1674_167461

/-- A parabola is defined by the equation y = ax^2 + bx + c where a ≠ 0 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- The given parabola y = 4x^2 + 8x - 5 -/
def given_parabola : Parabola :=
  { a := 4
    b := 8
    c := -5
    a_nonzero := by norm_num }

theorem focus_of_given_parabola :
  focus given_parabola = (-1, -143/16) := by sorry

end focus_of_given_parabola_l1674_167461


namespace part1_part2_l1674_167452

-- Define the given condition
def condition (x y : ℝ) : Prop :=
  |x - 4 - 2 * Real.sqrt 2| + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0

-- Define a rhombus
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

-- Theorem for part 1
theorem part1 {x y : ℝ} (h : condition x y) :
  x * y^2 - x^2 * y = -32 * Real.sqrt 2 := by
  sorry

-- Theorem for part 2
theorem part2 {x y : ℝ} (h : condition x y) :
  let r : Rhombus := ⟨x, y⟩
  (r.diagonal1 * r.diagonal2 / 2 = 4) ∧
  (r.diagonal1 * r.diagonal2 / (4 * Real.sqrt 3) = 2 * Real.sqrt 3 / 3) := by
  sorry

end part1_part2_l1674_167452


namespace max_profit_and_optimal_price_l1674_167441

/-- Represents the profit function for a product with given initial conditions -/
def profit (x : ℝ) : ℝ :=
  (500 - 10 * x) * ((50 + x) - 40)

/-- Theorem stating the maximum profit and optimal selling price -/
theorem max_profit_and_optimal_price :
  ∃ (max_profit : ℝ) (optimal_price : ℝ),
    (∀ x : ℝ, profit x ≤ max_profit) ∧
    (profit (optimal_price - 50) = max_profit) ∧
    max_profit = 9000 ∧
    optimal_price = 70 := by
  sorry

#check max_profit_and_optimal_price

end max_profit_and_optimal_price_l1674_167441


namespace sum_parity_l1674_167465

theorem sum_parity (a b : ℤ) (h : a + b = 1998) : 
  ∃ k : ℤ, 7 * a + 3 * b = 2 * k ∧ 7 * a + 3 * b ≠ 6799 := by
sorry

end sum_parity_l1674_167465


namespace find_divisor_l1674_167451

def is_divisor (n : ℕ) (d : ℕ) : Prop :=
  (n / d : ℚ) + 8 = 61

theorem find_divisor :
  ∃ (d : ℕ), is_divisor 265 d ∧ d = 5 := by
  sorry

end find_divisor_l1674_167451


namespace f_period_three_f_applied_95_times_main_result_l1674_167401

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x^3)^(1/3)

theorem f_period_three (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) : f (f (f x)) = x :=
sorry

theorem f_applied_95_times (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) : 
  (f^[95]) x = f (f x) :=
sorry

theorem main_result : (f^[95]) 19 = (1 - 1/19^3)^(1/3) :=
sorry

end f_period_three_f_applied_95_times_main_result_l1674_167401


namespace ellipse_and_line_theorem_l1674_167402

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The standard form equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- A line passing through a point (x₀, y₀) with slope m -/
structure Line where
  x₀ : ℝ
  y₀ : ℝ
  m : ℝ

/-- The equation of a line -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.m * (x - l.x₀) + l.y₀

theorem ellipse_and_line_theorem (e : Ellipse) 
  (h_triangle : e.a = 2 * Real.sqrt ((e.a^2 - e.b^2) / 4))
  (h_minor_axis : e.b = Real.sqrt 3) :
  (∃ (l : Line), l.x₀ = 0 ∧ l.y₀ = 2 ∧ 
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      e.equation x₁ y₁ ∧ 
      e.equation x₂ y₂ ∧
      l.equation x₁ y₁ ∧ 
      l.equation x₂ y₂ ∧
      x₁ ≠ x₂ ∧
      x₁ * x₂ + y₁ * y₂ = 2) ∧
    (l.m = Real.sqrt 2 / 2 ∨ l.m = -Real.sqrt 2 / 2)) ∧
  e.a = 2 ∧
  e.b = Real.sqrt 3 :=
sorry

end ellipse_and_line_theorem_l1674_167402


namespace max_small_boxes_in_large_box_l1674_167469

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the large box -/
def largeBox : BoxDimensions :=
  { length := 12, width := 14, height := 16 }

/-- The dimensions of the small box -/
def smallBox : BoxDimensions :=
  { length := 3, width := 7, height := 2 }

/-- Theorem stating the maximum number of small boxes that fit into the large box -/
theorem max_small_boxes_in_large_box :
  boxVolume largeBox / boxVolume smallBox = 64 := by
  sorry

end max_small_boxes_in_large_box_l1674_167469


namespace two_cos_sixty_degrees_equals_one_l1674_167416

theorem two_cos_sixty_degrees_equals_one : 2 * Real.cos (π / 3) = 1 := by
  sorry

end two_cos_sixty_degrees_equals_one_l1674_167416


namespace mono_increasing_sufficient_not_necessary_l1674_167459

open Set
open Function

-- Define a monotonically increasing function
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Statement B
def StatementB (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ < x₂ ∧ f x₁ < f x₂

-- Theorem to prove
theorem mono_increasing_sufficient_not_necessary :
  (∀ f : ℝ → ℝ, MonoIncreasing f → StatementB f) ∧
  (∃ g : ℝ → ℝ, ¬MonoIncreasing g ∧ StatementB g) :=
by sorry

end mono_increasing_sufficient_not_necessary_l1674_167459


namespace smallest_multiple_of_seven_greater_than_500_l1674_167493

theorem smallest_multiple_of_seven_greater_than_500 :
  ∃ (n : ℕ), n * 7 = 504 ∧ 
  504 > 500 ∧
  ∀ (m : ℕ), m * 7 > 500 → m * 7 ≥ 504 :=
by
  sorry

end smallest_multiple_of_seven_greater_than_500_l1674_167493


namespace total_puppies_adopted_l1674_167494

def puppies_week1 : ℕ := 20

def puppies_week2 : ℕ := (2 * puppies_week1) / 5

def puppies_week3 : ℕ := (3 * puppies_week2) / 8

def puppies_week4 : ℕ := 2 * puppies_week2

def puppies_week5 : ℕ := puppies_week1 + 10

def puppies_week6 : ℕ := 2 * puppies_week3 - 5

def puppies_week7 : ℕ := 2 * puppies_week6

def puppies_week8 : ℕ := (7 * puppies_week6) / 4

def puppies_week9 : ℕ := (3 * puppies_week8) / 2

def puppies_week10 : ℕ := (9 * puppies_week1) / 4

def puppies_week11 : ℕ := (5 * puppies_week10) / 6

theorem total_puppies_adopted : 
  puppies_week1 + puppies_week2 + puppies_week3 + puppies_week4 + 
  puppies_week5 + puppies_week6 + puppies_week7 + puppies_week8 + 
  puppies_week9 + puppies_week10 + puppies_week11 = 164 := by
  sorry

end total_puppies_adopted_l1674_167494


namespace distance_difference_l1674_167408

/-- Represents the distance traveled by a biker given their speed and time. -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents Camila's constant speed in miles per hour. -/
def camila_speed : ℝ := 15

/-- Represents Daniel's initial speed in miles per hour. -/
def daniel_initial_speed : ℝ := 15

/-- Represents Daniel's reduced speed in miles per hour. -/
def daniel_reduced_speed : ℝ := 10

/-- Represents the total time of the bike ride in hours. -/
def total_time : ℝ := 6

/-- Represents the time at which Daniel's speed changes in hours. -/
def speed_change_time : ℝ := 3

/-- Calculates the distance Camila travels in 6 hours. -/
def camila_distance : ℝ := distance camila_speed total_time

/-- Calculates the distance Daniel travels in 6 hours. -/
def daniel_distance : ℝ := 
  distance daniel_initial_speed speed_change_time + 
  distance daniel_reduced_speed (total_time - speed_change_time)

theorem distance_difference : camila_distance - daniel_distance = 15 := by
  sorry

end distance_difference_l1674_167408


namespace haploid_corn_triploid_watermelon_heritable_variation_l1674_167404

-- Define the sources of heritable variations
inductive HeritableVariationSource
  | GeneMutation
  | ChromosomalVariation
  | GeneRecombination

-- Define a structure for crop variations
structure CropVariation where
  name : String
  isChromosomalVariation : Bool

-- Define the property of being a heritable variation
def isHeritableVariation (source : HeritableVariationSource) : Prop :=
  match source with
  | HeritableVariationSource.GeneMutation => True
  | HeritableVariationSource.ChromosomalVariation => True
  | HeritableVariationSource.GeneRecombination => True

-- Theorem statement
theorem haploid_corn_triploid_watermelon_heritable_variation 
  (haploidCorn triploidWatermelon : CropVariation)
  (haploidCornChromosomal : haploidCorn.isChromosomalVariation = true)
  (triploidWatermelonChromosomal : triploidWatermelon.isChromosomalVariation = true) :
  isHeritableVariation HeritableVariationSource.ChromosomalVariation := by
  sorry


end haploid_corn_triploid_watermelon_heritable_variation_l1674_167404


namespace equation_root_range_l1674_167474

theorem equation_root_range (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ 
   Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x) = k + 1) 
  → k ∈ Set.Icc (-2) 1 := by
sorry

end equation_root_range_l1674_167474


namespace blue_cube_faces_l1674_167489

theorem blue_cube_faces (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end blue_cube_faces_l1674_167489


namespace strawberries_left_l1674_167480

/-- Theorem: If Adam picked 35 strawberries and ate 2 strawberries, then he has 33 strawberries left. -/
theorem strawberries_left (initial : ℕ) (eaten : ℕ) (h1 : initial = 35) (h2 : eaten = 2) :
  initial - eaten = 33 := by
  sorry

end strawberries_left_l1674_167480


namespace tangent_line_length_l1674_167472

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define point P
def P : ℝ × ℝ := (-2, 5)

-- Define the tangent line (abstractly, as we don't know its equation)
def tangent_line (Q : ℝ × ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ (x y : ℝ), y = m*x + b ∧ circle_equation x y → (x, y) = Q

-- Theorem statement
theorem tangent_line_length :
  ∃ (Q : ℝ × ℝ), circle_equation Q.1 Q.2 ∧ tangent_line Q →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 6 :=
sorry

end tangent_line_length_l1674_167472


namespace one_in_set_zero_one_l1674_167473

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by sorry

end one_in_set_zero_one_l1674_167473


namespace missing_number_proof_l1674_167413

theorem missing_number_proof (x : ℝ) : x * 240 = 173 * 240 → x = 173 := by
  sorry

end missing_number_proof_l1674_167413


namespace student_guinea_pig_difference_l1674_167425

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 22

/-- The number of guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- Theorem stating the difference between total students and total guinea pigs -/
theorem student_guinea_pig_difference :
  num_classrooms * students_per_classroom - num_classrooms * guinea_pigs_per_classroom = 95 :=
by sorry

end student_guinea_pig_difference_l1674_167425


namespace abs_sum_simplification_l1674_167458

theorem abs_sum_simplification (m x : ℝ) (h1 : 0 < m) (h2 : m < 10) (h3 : m ≤ x) (h4 : x ≤ 10) :
  |x - m| + |x - 10| + |x - m - 10| = 20 - x := by
  sorry

end abs_sum_simplification_l1674_167458


namespace square_difference_fourth_power_l1674_167486

theorem square_difference_fourth_power : (7^2 - 5^2)^4 = 331776 := by
  sorry

end square_difference_fourth_power_l1674_167486


namespace batting_average_calculation_l1674_167431

/-- Calculates the batting average given the total innings, highest score, score difference, and average excluding extremes -/
def batting_average (total_innings : ℕ) (highest_score : ℕ) (score_difference : ℕ) (avg_excluding_extremes : ℚ) : ℚ :=
  let lowest_score := highest_score - score_difference
  let runs_excluding_extremes := avg_excluding_extremes * (total_innings - 2)
  let total_runs := runs_excluding_extremes + highest_score + lowest_score
  total_runs / total_innings

theorem batting_average_calculation :
  batting_average 46 179 150 58 = 60 := by
  sorry

end batting_average_calculation_l1674_167431


namespace smallest_constant_inequality_l1674_167406

theorem smallest_constant_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ p : ℝ, ∀ a b : ℝ, 0 < a → 0 < b →
    Real.sqrt (a * b) - (2 * a * b) / (a + b) ≤ p * ((a + b) / 2 - Real.sqrt (a * b))) ∧
  (∀ p : ℝ, (∀ a b : ℝ, 0 < a → 0 < b →
    Real.sqrt (a * b) - (2 * a * b) / (a + b) ≤ p * ((a + b) / 2 - Real.sqrt (a * b))) →
    1 ≤ p) ∧
  (∀ a b : ℝ, 0 < a → 0 < b →
    Real.sqrt (a * b) - (2 * a * b) / (a + b) ≤ (a + b) / 2 - Real.sqrt (a * b)) :=
by sorry

end smallest_constant_inequality_l1674_167406


namespace hyperbola_equation_l1674_167462

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    ∃ t : ℝ, (b*x + a*y = 0 ∨ b*x - a*y = 0) ∧ 
    x^2 + y^2 - 6*x + 5 = t^2) →
  (3 : ℝ)^2 = a^2 - b^2 →
  a^2 = 5 ∧ b^2 = 4 := by sorry

end hyperbola_equation_l1674_167462


namespace total_courses_is_200_l1674_167400

/-- The number of college courses attended by Max -/
def max_courses : ℕ := 40

/-- The number of college courses attended by Sid -/
def sid_courses : ℕ := 4 * max_courses

/-- The total number of college courses attended by Max and Sid -/
def total_courses : ℕ := max_courses + sid_courses

/-- Theorem stating that the total number of courses attended by Max and Sid is 200 -/
theorem total_courses_is_200 : total_courses = 200 := by
  sorry

end total_courses_is_200_l1674_167400


namespace function_inequality_l1674_167437

open Real

theorem function_inequality (f : ℝ → ℝ) (f_deriv : Differentiable ℝ f) 
  (h1 : ∀ x, f x + deriv f x > 1) (h2 : f 0 = 4) :
  ∀ x, f x > 3 / exp x + 1 ↔ x > 0 := by
  sorry

end function_inequality_l1674_167437


namespace find_x_in_ratio_l1674_167466

/-- Given t = 5, prove that the positive integer x satisfying 2 : m : t = m : 32 : x is 20 -/
theorem find_x_in_ratio (t : ℕ) (h_t : t = 5) :
  ∃ (m : ℤ) (x : ℕ), 2 * 32 * t = m * m * x ∧ x = 20 := by
  sorry

end find_x_in_ratio_l1674_167466


namespace interval_intersection_l1674_167492

theorem interval_intersection (x : ℝ) : (|5 - x| < 5 ∧ x^2 < 25) ↔ (0 < x ∧ x < 5) := by
  sorry

end interval_intersection_l1674_167492


namespace parabola_intersection_l1674_167453

-- Define the parabola
def parabola (k x : ℝ) : ℝ := x^2 - (k-1)*x - 3*k - 2

-- Define the intersection points
def α (k : ℝ) : ℝ := sorry
def β (k : ℝ) : ℝ := sorry

-- Theorem statement
theorem parabola_intersection (k : ℝ) : 
  (parabola k (α k) = 0) ∧ 
  (parabola k (β k) = 0) ∧ 
  ((α k)^2 + (β k)^2 = 17) → 
  k = 2 := by sorry

end parabola_intersection_l1674_167453


namespace cone_volume_contradiction_l1674_167436

theorem cone_volume_contradiction (base_area height volume : ℝ) : 
  base_area = 9 → height = 5 → volume = 45 → (1/3) * base_area * height ≠ volume :=
by
  sorry

end cone_volume_contradiction_l1674_167436


namespace min_packages_correct_min_packages_value_l1674_167468

/-- The number of t-shirts in each package -/
def package_size : ℕ := 6

/-- The number of t-shirts Mom wants to buy -/
def desired_shirts : ℕ := 71

/-- The minimum number of packages needed to buy at least the desired number of shirts -/
def min_packages : ℕ := (desired_shirts + package_size - 1) / package_size

theorem min_packages_correct : 
  min_packages * package_size ≥ desired_shirts ∧ 
  ∀ k : ℕ, k * package_size ≥ desired_shirts → k ≥ min_packages :=
by sorry

theorem min_packages_value : min_packages = 12 :=
by sorry

end min_packages_correct_min_packages_value_l1674_167468


namespace polynomial_coefficient_b_l1674_167498

theorem polynomial_coefficient_b (a b c : ℚ) : 
  (∀ x, (5*x^2 - 3*x + 7/3) * (a*x^2 + b*x + c) = 
        15*x^4 - 14*x^3 + 20*x^2 - 25/3*x + 14/3) →
  b = -1 := by
sorry

end polynomial_coefficient_b_l1674_167498


namespace expression_simplification_l1674_167417

theorem expression_simplification (m : ℝ) (h : m^2 - m - 1 = 0) :
  (m - 1) / (m^2 - 2*m) / (m + 1/(m - 2)) = 1 := by
  sorry

end expression_simplification_l1674_167417


namespace problem_solution_l1674_167481

theorem problem_solution (n : ℕ+) 
  (x : ℝ) (hx : x = (Real.sqrt (n + 2) - Real.sqrt n) / (Real.sqrt (n + 2) + Real.sqrt n))
  (y : ℝ) (hy : y = (Real.sqrt (n + 2) + Real.sqrt n) / (Real.sqrt (n + 2) - Real.sqrt n))
  (h_eq : 14 * x^2 + 26 * x * y + 14 * y^2 = 2014) :
  n = 5 := by
sorry

end problem_solution_l1674_167481


namespace piggy_bank_theorem_l1674_167430

/-- The value of a piggy bank containing dimes and quarters -/
def piggy_bank_value (num_dimes num_quarters : ℕ) (dime_value quarter_value : ℚ) : ℚ :=
  (num_dimes : ℚ) * dime_value + (num_quarters : ℚ) * quarter_value

/-- Theorem: The value of a piggy bank with 35 dimes and 65 quarters is $19.75 -/
theorem piggy_bank_theorem :
  piggy_bank_value 35 65 (10 / 100) (25 / 100) = 1975 / 100 := by
  sorry

#eval piggy_bank_value 35 65 (10 / 100) (25 / 100)

end piggy_bank_theorem_l1674_167430


namespace point_outside_circle_l1674_167471

/-- A circle with a given radius -/
structure Circle where
  radius : ℝ

/-- A point with a given distance from the center of a circle -/
structure Point where
  distanceFromCenter : ℝ

/-- Determines if a point is outside a circle -/
def isOutside (c : Circle) (p : Point) : Prop :=
  p.distanceFromCenter > c.radius

/-- Theorem: If the radius of a circle is 3 and the distance from a point to the center is 4,
    then the point is outside the circle -/
theorem point_outside_circle (c : Circle) (p : Point)
    (h1 : c.radius = 3)
    (h2 : p.distanceFromCenter = 4) :
    isOutside c p := by
  sorry

end point_outside_circle_l1674_167471


namespace five_by_five_uncoverable_l1674_167405

/-- Represents a rectangular board -/
structure Board where
  rows : ℕ
  cols : ℕ

/-- Represents a domino -/
structure Domino where
  width : ℕ
  height : ℕ

/-- Checks if a board can be completely covered by a given domino -/
def is_coverable (b : Board) (d : Domino) : Prop :=
  (b.rows * b.cols) % (d.width * d.height) = 0

/-- Theorem stating that a 5x5 board cannot be covered by 1x2 dominoes -/
theorem five_by_five_uncoverable :
  ¬ is_coverable (Board.mk 5 5) (Domino.mk 2 1) := by
  sorry

end five_by_five_uncoverable_l1674_167405


namespace inverse_of_5_mod_31_l1674_167491

theorem inverse_of_5_mod_31 : ∃ x : ℕ, x ≤ 30 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  sorry

end inverse_of_5_mod_31_l1674_167491


namespace cube_edge_color_probability_l1674_167435

theorem cube_edge_color_probability :
  let num_edges : ℕ := 12
  let num_colors : ℕ := 2
  let num_visible_faces : ℕ := 4
  let prob_same_color_face : ℝ := 2 / 2^4

  (1 : ℝ) / 256 = prob_same_color_face^num_visible_faces := by sorry

end cube_edge_color_probability_l1674_167435


namespace sum_of_reciprocals_negative_l1674_167499

theorem sum_of_reciprocals_negative (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_eight : a * b * c = 8) : 
  1/a + 1/b + 1/c < 0 := by
  sorry

end sum_of_reciprocals_negative_l1674_167499


namespace increase_in_average_goals_is_point_two_l1674_167497

/-- Calculates the increase in average goals score after the fifth match -/
def increase_in_average_goals (total_matches : ℕ) (total_goals : ℕ) (goals_in_fifth_match : ℕ) : ℚ :=
  let goals_before_fifth := total_goals - goals_in_fifth_match
  let matches_before_fifth := total_matches - 1
  let average_before := goals_before_fifth / matches_before_fifth
  let average_after := total_goals / total_matches
  average_after - average_before

/-- The increase in average goals score after the fifth match is 0.2 -/
theorem increase_in_average_goals_is_point_two :
  increase_in_average_goals 5 21 5 = 1/5 := by
  sorry

end increase_in_average_goals_is_point_two_l1674_167497


namespace scientific_notation_equality_l1674_167476

-- Define the original number
def original_number : ℝ := 850000

-- Define the scientific notation components
def coefficient : ℝ := 8.5
def exponent : ℤ := 5

-- Theorem statement
theorem scientific_notation_equality :
  original_number = coefficient * (10 : ℝ) ^ exponent := by
  sorry

end scientific_notation_equality_l1674_167476


namespace divisor_sum_condition_l1674_167496

def d (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_sum_condition (n : ℕ) : n ≥ 3 → (d (n - 1) + d n + d (n + 1) ≤ 8 ↔ n = 3 ∨ n = 4 ∨ n = 6) := by
  sorry

end divisor_sum_condition_l1674_167496


namespace smallest_divisible_by_1_to_12_l1674_167440

theorem smallest_divisible_by_1_to_12 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → n ≤ m) ∧ n = 27720 := by
  sorry

end smallest_divisible_by_1_to_12_l1674_167440


namespace washer_cost_l1674_167422

/-- Given a washer-dryer combination costing $1,200, where the washer costs $220 more than the dryer,
    prove that the cost of the washer is $710. -/
theorem washer_cost (total_cost dryer_cost washer_cost : ℕ) : 
  total_cost = 1200 →
  washer_cost = dryer_cost + 220 →
  total_cost = washer_cost + dryer_cost →
  washer_cost = 710 := by
sorry

end washer_cost_l1674_167422


namespace addition_to_reach_91_l1674_167403

theorem addition_to_reach_91 : ∃ x : ℚ, (5 * 12) / (180 / 3) + x = 91 :=
by
  sorry

end addition_to_reach_91_l1674_167403


namespace indians_invented_arabic_numerals_l1674_167415

/-- Represents a numerical system -/
structure NumericalSystem where
  digits : Set Nat
  name : String
  isUniversal : Bool

/-- The civilization that invented a numerical system -/
inductive Civilization
  | Indians
  | Chinese
  | Babylonians
  | Arabs

/-- Arabic numerals as defined in the problem -/
def arabicNumerals : NumericalSystem :=
  { digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    name := "Arabic numerals",
    isUniversal := true }

/-- The theorem stating that ancient Indians invented Arabic numerals -/
theorem indians_invented_arabic_numerals :
  ∃ (inventor : Civilization), inventor = Civilization.Indians ∧
  (arabicNumerals.digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
   arabicNumerals.name = "Arabic numerals" ∧
   arabicNumerals.isUniversal = true) :=
by sorry

end indians_invented_arabic_numerals_l1674_167415


namespace inequality_proof_l1674_167412

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end inequality_proof_l1674_167412


namespace base7_5463_equals_1956_l1674_167426

def base7ToBase10 (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem base7_5463_equals_1956 : base7ToBase10 5 4 6 3 = 1956 := by
  sorry

end base7_5463_equals_1956_l1674_167426


namespace next_square_property_l1674_167409

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_square_property (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_perfect_square ((n / 100) * (n % 100))

theorem next_square_property : 
  ∀ n : ℕ, n > 1818 → has_square_property n → n ≥ 1832 :=
sorry

end next_square_property_l1674_167409


namespace triangle_shape_l1674_167460

theorem triangle_shape (a b : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → 
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.cos A = b * Real.cos B →
  (A = B ∨ A + B = π / 2) :=
sorry

end triangle_shape_l1674_167460


namespace expression_evaluation_l1674_167495

theorem expression_evaluation : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end expression_evaluation_l1674_167495


namespace inequality_always_holds_l1674_167427

theorem inequality_always_holds (m : ℝ) :
  (∀ x : ℝ, 2 * m * x^2 + m * x - 3/4 < 0) → -6 < m ∧ m ≤ 0 := by
  sorry

end inequality_always_holds_l1674_167427


namespace copper_percentage_second_alloy_l1674_167428

/-- Calculates the percentage of copper in the second alloy -/
theorem copper_percentage_second_alloy 
  (desired_percentage : Real) 
  (first_alloy_percentage : Real)
  (first_alloy_weight : Real)
  (total_weight : Real) :
  let second_alloy_weight := total_weight - first_alloy_weight
  let desired_copper := desired_percentage * total_weight / 100
  let first_alloy_copper := first_alloy_percentage * first_alloy_weight / 100
  let second_alloy_copper := desired_copper - first_alloy_copper
  second_alloy_copper / second_alloy_weight * 100 = 21 :=
by
  sorry

#check copper_percentage_second_alloy 19.75 18 45 108

end copper_percentage_second_alloy_l1674_167428


namespace sons_age_l1674_167482

/-- Proves that the son's age is 30 given the conditions of the problem -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 5 = 2 * (son_age + 5) →
  son_age = 30 := by
  sorry

end sons_age_l1674_167482


namespace aunt_marge_candy_distribution_l1674_167433

theorem aunt_marge_candy_distribution (total_candy : ℕ) 
  (kate_candy : ℕ) (robert_candy : ℕ) (mary_candy : ℕ) (bill_candy : ℕ) : 
  total_candy = 20 ∧ 
  robert_candy = kate_candy + 2 ∧
  bill_candy = mary_candy - 6 ∧
  mary_candy = robert_candy + 2 ∧
  kate_candy = bill_candy + 2 ∧
  total_candy = kate_candy + robert_candy + mary_candy + bill_candy →
  kate_candy = 4 := by
sorry

end aunt_marge_candy_distribution_l1674_167433


namespace time_to_reach_B_after_second_meeting_l1674_167423

-- Define the variables
variable (S : ℝ) -- Total distance between A and B
variable (v_A v_B : ℝ) -- Speeds of A and B
variable (t : ℝ) -- Time taken by B to catch up with A

-- Define the theorem
theorem time_to_reach_B_after_second_meeting : 
  -- A starts 48 minutes (4/5 hours) before B
  v_A * (t + 4/5) = 2/3 * S →
  -- B catches up with A when A has traveled 2/3 of the distance
  v_B * t = 2/3 * S →
  -- They meet again 6 minutes (1/10 hour) after B leaves B
  v_A * (t + 4/5 + 1/2 * t + 1/10) + 1/10 * v_B = S →
  -- The time it takes for A to reach B after meeting B again is 12 minutes (1/5 hour)
  1/5 = S / v_A - (t + 4/5 + 1/2 * t + 1/10) := by
  sorry

end time_to_reach_B_after_second_meeting_l1674_167423


namespace min_value_expression_l1674_167432

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (4 * z) / (2 * x + y) + (4 * x) / (y + 2 * z) + y / (x + z) ≥ 3 ∧
  ((4 * z) / (2 * x + y) + (4 * x) / (y + 2 * z) + y / (x + z) = 3 ↔ 2 * x = y ∧ y = 2 * z) := by
sorry

end min_value_expression_l1674_167432


namespace inequality_proof_l1674_167454

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end inequality_proof_l1674_167454


namespace cube_volume_problem_l1674_167488

theorem cube_volume_problem (a : ℕ) : 
  (a - 2) * a * (a + 2) = a^3 - 14 → a^3 = 27 := by
  sorry

end cube_volume_problem_l1674_167488


namespace bakery_earnings_l1674_167464

/-- Represents the daily production and prices of baked goods in a bakery --/
structure BakeryData where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  daily_cupcakes : ℕ
  daily_cookies : ℕ
  daily_biscuits : ℕ

/-- Calculates the total earnings for a given number of days --/
def total_earnings (data : BakeryData) (days : ℕ) : ℝ :=
  (data.cupcake_price * data.daily_cupcakes +
   data.cookie_price * data.daily_cookies +
   data.biscuit_price * data.daily_biscuits) * days

/-- Theorem stating that the total earnings for 5 days is $350 --/
theorem bakery_earnings (data : BakeryData) 
  (h1 : data.cupcake_price = 1.5)
  (h2 : data.cookie_price = 2)
  (h3 : data.biscuit_price = 1)
  (h4 : data.daily_cupcakes = 20)
  (h5 : data.daily_cookies = 10)
  (h6 : data.daily_biscuits = 20) :
  total_earnings data 5 = 350 := by
  sorry

end bakery_earnings_l1674_167464


namespace salt_bag_weight_l1674_167439

/-- Given a bag of sugar weighing 16 kg and the fact that removing 4 kg from the combined
    weight of sugar and salt bags results in 42 kg, prove that the salt bag weighs 30 kg. -/
theorem salt_bag_weight (sugar_weight : ℕ) (combined_minus_four : ℕ) :
  sugar_weight = 16 ∧ combined_minus_four = 42 →
  ∃ (salt_weight : ℕ), salt_weight = 30 ∧ sugar_weight + salt_weight = combined_minus_four + 4 :=
by sorry

end salt_bag_weight_l1674_167439


namespace max_food_per_guest_l1674_167421

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ) (h1 : total_food = 323) (h2 : min_guests = 162) :
  (total_food / min_guests : ℕ) = 1 := by
  sorry

end max_food_per_guest_l1674_167421


namespace circle_area_above_line_l1674_167487

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 18*y + 61 = 0

-- Define the line equation
def line_equation (y : ℝ) : Prop :=
  y = 4

-- Theorem statement
theorem circle_area_above_line : 
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    (center_y > 4) ∧
    (center_y - radius > 4) ∧
    (radius = 1) ∧
    (Real.pi * radius^2 = Real.pi) :=
sorry

end circle_area_above_line_l1674_167487


namespace natasha_hill_climbing_l1674_167483

/-- Natasha's hill climbing problem -/
theorem natasha_hill_climbing
  (time_up : ℝ)
  (time_down : ℝ)
  (avg_speed_total : ℝ)
  (h_time_up : time_up = 4)
  (h_time_down : time_down = 2)
  (h_avg_speed_total : avg_speed_total = 2) :
  let total_time := time_up + time_down
  let total_distance := avg_speed_total * total_time
  let distance_up := total_distance / 2
  let avg_speed_up := distance_up / time_up
  avg_speed_up = 1.5 := by
sorry

end natasha_hill_climbing_l1674_167483


namespace function_inequality_solutions_l1674_167438

noncomputable def f (x : ℝ) : ℝ := (Real.log (2 * x)) / x

theorem function_inequality_solutions (a : ℝ) :
  (∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x : ℤ, (x : ℝ) > 0 → (x ∈ s ↔ f x ^ 2 + a * f x > 0)) ↔
  a ∈ Set.Ioo (-Real.log 2) (-1/3 * Real.log 6) ∪ {-1/3 * Real.log 6} :=
sorry

end function_inequality_solutions_l1674_167438


namespace jacks_kids_l1674_167419

theorem jacks_kids (shirts_per_kid : ℕ) (buttons_per_shirt : ℕ) (total_buttons : ℕ) : 
  shirts_per_kid = 3 → buttons_per_shirt = 7 → total_buttons = 63 →
  ∃ (num_kids : ℕ), num_kids * shirts_per_kid * buttons_per_shirt = total_buttons ∧ num_kids = 3 :=
by
  sorry

end jacks_kids_l1674_167419


namespace cubic_difference_l1674_167424

theorem cubic_difference (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) :
  x^3 - y^3 = -448 := by
sorry

end cubic_difference_l1674_167424


namespace addition_of_integers_l1674_167410

theorem addition_of_integers : -10 + 3 = -7 := by
  sorry

end addition_of_integers_l1674_167410


namespace average_study_time_difference_l1674_167445

/-- The daily differences in study time (in minutes) between Mira and Clara over a week -/
def study_time_differences : List Int := [15, 0, -15, 25, 5, -5, 10]

/-- The number of days in the week -/
def days_in_week : Nat := 7

/-- Theorem stating that the average difference in daily study time is 5 minutes -/
theorem average_study_time_difference :
  (study_time_differences.sum : ℚ) / days_in_week = 5 := by
  sorry

end average_study_time_difference_l1674_167445


namespace range_of_x_for_inequality_l1674_167444

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem range_of_x_for_inequality (x : ℝ) :
  (∀ m : ℝ, m ∈ Set.Icc (-2) 2 → f (m*x - 2) + f x < 0) →
  x ∈ Set.Ioo (-2) (2/3) :=
sorry

end range_of_x_for_inequality_l1674_167444


namespace unique_odd_pair_divisibility_l1674_167414

theorem unique_odd_pair_divisibility : 
  ∀ (a b : ℤ), 
    Odd a → Odd b →
    (∃ (c : ℕ), ∀ (n : ℕ), ∃ (k : ℤ), (c^n + 1 : ℤ) = k * (2^n * a + b)) →
    a = 1 ∧ b = 1 := by
  sorry

end unique_odd_pair_divisibility_l1674_167414


namespace puzzle_solution_l1674_167418

/-- Represents the possible values in a cell of the grid -/
inductive CellValue
  | Two
  | Zero
  | One
  | Five
  | Empty

/-- Represents a 5x6 grid -/
def Grid := Matrix (Fin 5) (Fin 6) CellValue

/-- Checks if a given grid satisfies the puzzle constraints -/
def is_valid_grid (g : Grid) : Prop :=
  -- Each row contains each digit exactly once
  (∀ i, ∃! j, g i j = CellValue.Two) ∧
  (∀ i, ∃! j, g i j = CellValue.Zero) ∧
  (∀ i, ∃! j, g i j = CellValue.One) ∧
  (∀ i, ∃! j, g i j = CellValue.Five) ∧
  -- Each column contains each digit exactly once
  (∀ j, ∃! i, g i j = CellValue.Two) ∧
  (∀ j, ∃! i, g i j = CellValue.Zero) ∧
  (∀ j, ∃! i, g i j = CellValue.One) ∧
  (∀ j, ∃! i, g i j = CellValue.Five) ∧
  -- Same digits are not adjacent diagonally
  (∀ i j, i < 4 → j < 5 → g i j ≠ g (i+1) (j+1)) ∧
  (∀ i j, i < 4 → j > 0 → g i j ≠ g (i+1) (j-1))

/-- The theorem stating the solution to the puzzle -/
theorem puzzle_solution (g : Grid) (h : is_valid_grid g) :
  g 4 0 = CellValue.One ∧
  g 4 1 = CellValue.Five ∧
  g 4 2 = CellValue.Empty ∧
  g 4 3 = CellValue.Empty ∧
  g 4 4 = CellValue.Two :=
sorry

end puzzle_solution_l1674_167418


namespace repeating_decimal_equals_fraction_l1674_167477

/-- The repeating decimal 0.567567567... expressed as a rational number -/
def repeating_decimal : ℚ := 567 / 999

theorem repeating_decimal_equals_fraction : repeating_decimal = 21 / 37 := by
  sorry

end repeating_decimal_equals_fraction_l1674_167477


namespace parabola_line_intersection_length_no_isosceles_right_triangle_on_parabola_l1674_167467

/-- Given a parabola y^2 = 2px where p > 0, and a line y = k(x - p/2) intersecting 
    the parabola at points A and B, the length of AB is (2p(k^2 + 1)) / k^2 -/
theorem parabola_line_intersection_length (p k : ℝ) (hp : p > 0) :
  let f : ℝ → ℝ := λ k => (2 * p * (k^2 + 1)) / k^2
  let parabola : ℝ × ℝ → Prop := λ (x, y) => y^2 = 2 * p * x
  let line : ℝ → ℝ := λ x => k * (x - p / 2)
  let A := (x₁, line x₁)
  let B := (x₂, line x₂)
  parabola A ∧ parabola B → abs (x₂ - x₁) = f k :=
by sorry

/-- There does not exist a point C on the parabola y^2 = 2px such that 
    triangle ABC is an isosceles right triangle with C as the vertex of the right angle -/
theorem no_isosceles_right_triangle_on_parabola (p : ℝ) (hp : p > 0) :
  let parabola : ℝ × ℝ → Prop := λ (x, y) => y^2 = 2 * p * x
  ¬ ∃ (A B C : ℝ × ℝ), parabola A ∧ parabola B ∧ parabola C ∧
    (C.1 < (A.1 + B.1) / 2) ∧
    (abs (A.1 - C.1) = abs (B.1 - C.1)) ∧
    ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0) :=
by sorry

end parabola_line_intersection_length_no_isosceles_right_triangle_on_parabola_l1674_167467


namespace ratio_of_divisor_sums_l1674_167434

def N : ℕ := 64 * 45 * 91 * 49

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 126 := by sorry

end ratio_of_divisor_sums_l1674_167434


namespace fixed_point_of_exponential_function_l1674_167450

/-- The function f(x) = a^(x-1) - 2 passes through the point (1, -1) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f := fun x : ℝ => a^(x - 1) - 2
  f 1 = -1 := by
  sorry

end fixed_point_of_exponential_function_l1674_167450


namespace apollonius_circle_locus_l1674_167475

/-- Given two points A and B in a 2D plane, and a positive real number n,
    the Apollonius circle is the locus of points P such that PA = n * PB -/
theorem apollonius_circle_locus 
  (A B : EuclideanSpace ℝ (Fin 2))  -- Two given points in 2D space
  (n : ℝ) 
  (hn : n > 0) :  -- n is positive
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ),
    ∀ P : EuclideanSpace ℝ (Fin 2), 
      dist P A = n * dist P B ↔ 
      dist P center = radius :=
sorry

end apollonius_circle_locus_l1674_167475


namespace pyramid_volume_l1674_167448

-- Define the rectangular parallelepiped
structure Parallelepiped where
  AB : ℝ
  BC : ℝ
  CG : ℝ

-- Define the rectangular pyramid
structure Pyramid where
  base : ℝ -- Area of the base BDFE
  height : ℝ -- Height of the pyramid (XM)

-- Define the problem
theorem pyramid_volume (p : Parallelepiped) (pyr : Pyramid) : 
  p.AB = 4 → 
  p.BC = 2 → 
  p.CG = 5 → 
  pyr.base = p.AB * p.BC → 
  pyr.height = p.CG → 
  (1/3 : ℝ) * pyr.base * pyr.height = 40/3 := by
  sorry

end pyramid_volume_l1674_167448


namespace quadratic_equation_properties_l1674_167478

theorem quadratic_equation_properties (m : ℝ) :
  m < 4 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + m = 0 ∧ x₂^2 - 4*x₂ + m = 0) ∧
  ((-1)^2 - 4*(-1) + m = 0 → m = -5 ∧ 5^2 - 4*5 + m = 0) :=
by sorry

end quadratic_equation_properties_l1674_167478


namespace range_of_b_given_false_proposition_l1674_167447

theorem range_of_b_given_false_proposition :
  (¬ ∃ a : ℝ, a < 0 ∧ a + 1/a > b) →
  ∀ b : ℝ, b ≥ -2 ↔ b ∈ Set.Ici (-2 : ℝ) := by
  sorry

end range_of_b_given_false_proposition_l1674_167447


namespace range_of_m_l1674_167485

theorem range_of_m (m : ℝ) : 
  (m + 4)^(-1/2 : ℝ) < (3 - 2*m)^(-1/2 : ℝ) → 
  -1/3 < m ∧ m < 3/2 :=
by
  sorry

end range_of_m_l1674_167485


namespace binomial_coeff_not_coprime_l1674_167490

theorem binomial_coeff_not_coprime (k m n : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) :
  ¬(Nat.gcd (Nat.choose n k) (Nat.choose n m) = 1) :=
sorry

end binomial_coeff_not_coprime_l1674_167490


namespace bernoulli_expectation_and_variance_l1674_167411

/-- A random variable with Bernoulli distribution -/
structure BernoulliRV where
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Probability mass function for Bernoulli distribution -/
def prob (ξ : BernoulliRV) (k : ℕ) : ℝ :=
  if k = 0 then 1 - ξ.p
  else if k = 1 then ξ.p
  else 0

/-- Expected value of a Bernoulli random variable -/
def expectation (ξ : BernoulliRV) : ℝ := ξ.p

/-- Variance of a Bernoulli random variable -/
def variance (ξ : BernoulliRV) : ℝ := (1 - ξ.p) * ξ.p

/-- Theorem: The expected value and variance of a Bernoulli random variable -/
theorem bernoulli_expectation_and_variance (ξ : BernoulliRV) :
  expectation ξ = ξ.p ∧ variance ξ = (1 - ξ.p) * ξ.p := by sorry

end bernoulli_expectation_and_variance_l1674_167411


namespace total_distance_of_trip_l1674_167442

-- Define the triangle XYZ
def Triangle (XY YZ ZX : ℝ) : Prop :=
  XY > 0 ∧ YZ > 0 ∧ ZX > 0 ∧ XY^2 = YZ^2 + ZX^2

-- Theorem statement
theorem total_distance_of_trip (XY YZ ZX : ℝ) 
  (h1 : Triangle XY YZ ZX) (h2 : XY = 5000) (h3 : ZX = 4000) : 
  XY + YZ + ZX = 12000 := by
  sorry

end total_distance_of_trip_l1674_167442


namespace abs_neg_eleven_l1674_167455

theorem abs_neg_eleven : |(-11 : ℤ)| = 11 := by sorry

end abs_neg_eleven_l1674_167455


namespace orange_weight_change_l1674_167484

theorem orange_weight_change (initial_weight : ℝ) (initial_water_percent : ℝ) (water_decrease : ℝ) : 
  initial_weight = 5 →
  initial_water_percent = 95 →
  water_decrease = 5 →
  let non_water_weight := initial_weight * (100 - initial_water_percent) / 100
  let new_water_percent := initial_water_percent - water_decrease
  let new_total_weight := non_water_weight / ((100 - new_water_percent) / 100)
  new_total_weight = 2.5 := by
  sorry

end orange_weight_change_l1674_167484


namespace triangle_line_equations_l1674_167420

/-- Triangle with vertices A(4, 0), B(6, 7), and C(0, 3) -/
structure Triangle where
  A : ℝ × ℝ := (4, 0)
  B : ℝ × ℝ := (6, 7)
  C : ℝ × ℝ := (0, 3)

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The line passing through the midpoints of sides BC and AB -/
def midpointLine (t : Triangle) : LineEquation :=
  { a := 3, b := 4, c := -29 }

/-- The perpendicular bisector of side BC -/
def perpendicularBisector (t : Triangle) : LineEquation :=
  { a := 3, b := 2, c := -19 }

theorem triangle_line_equations (t : Triangle) :
  (midpointLine t = { a := 3, b := 4, c := -29 }) ∧
  (perpendicularBisector t = { a := 3, b := 2, c := -19 }) := by
  sorry

end triangle_line_equations_l1674_167420


namespace perfect_square_trinomial_condition_l1674_167449

/-- If 9x^2 + mxy + 16y^2 is a perfect square trinomial, then m = ±24 -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  (∃ (a b : ℝ), ∀ (x y : ℝ), 9*x^2 + m*x*y + 16*y^2 = (a*x + b*y)^2) →
  (m = 24 ∨ m = -24) :=
by sorry

end perfect_square_trinomial_condition_l1674_167449


namespace circle_and_line_properties_l1674_167407

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

-- Define the lines l
def line_l₁ (x y : ℝ) : Prop :=
  4*x + 3*y + 3 = 0

def line_l₂ (x y : ℝ) : Prop :=
  4*x + 3*y - 7 = 0

-- Theorem statement
theorem circle_and_line_properties :
  -- Circle C passes through O(0,0), A(-2,4), and B(1,1)
  circle_C 0 0 ∧ circle_C (-2) 4 ∧ circle_C 1 1 ∧
  -- Line l has slope -4/3
  (∀ x y : ℝ, (line_l₁ x y ∨ line_l₂ x y) → (y - 2) = -4/3 * (x + 1)) ∧
  -- The chord intercepted by circle C on line l has a length of 4
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    ((line_l₁ x₁ y₁ ∧ line_l₁ x₂ y₂) ∨ (line_l₂ x₁ y₁ ∧ line_l₂ x₂ y₂)) ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16) →
  -- The equation of circle C is correct
  (∀ x y : ℝ, circle_C x y ↔ x^2 + y^2 + 2*x - 4*y = 0) ∧
  -- The equation of line l is one of the two given equations
  (∀ x y : ℝ, (4*x + 3*y + 3 = 0 ∨ 4*x + 3*y - 7 = 0) ↔ (line_l₁ x y ∨ line_l₂ x y)) :=
sorry

end circle_and_line_properties_l1674_167407
