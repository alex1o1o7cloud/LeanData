import Mathlib

namespace NUMINAMATH_CALUDE_tetrahedron_smallest_faces_l2479_247971

/-- Represents the number of faces in a geometric shape. -/
def faces (shape : String) : ℕ :=
  match shape with
  | "Tetrahedron" => 4
  | "Quadrangular pyramid" => 5
  | "Triangular prism" => 5
  | "Triangular pyramid" => 4
  | _ => 0

/-- The list of shapes we're considering. -/
def shapes : List String :=
  ["Tetrahedron", "Quadrangular pyramid", "Triangular prism", "Triangular pyramid"]

/-- Theorem stating that the tetrahedron has the smallest number of faces among the given shapes. -/
theorem tetrahedron_smallest_faces :
    ∀ shape ∈ shapes, faces "Tetrahedron" ≤ faces shape := by
  sorry

#check tetrahedron_smallest_faces

end NUMINAMATH_CALUDE_tetrahedron_smallest_faces_l2479_247971


namespace NUMINAMATH_CALUDE_total_rope_inches_is_264_l2479_247932

/-- Represents the length of rope in feet for each week -/
def rope_length : Fin 4 → ℕ
  | 0 => 6  -- Week 1
  | 1 => 2 * rope_length 0  -- Week 2
  | 2 => rope_length 1 - 4  -- Week 3
  | 3 => rope_length 2 / 2  -- Week 4

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the total length of rope in feet at the end of the month -/
def total_rope_length : ℕ :=
  rope_length 0 + rope_length 1 + rope_length 2 - rope_length 3

/-- Theorem stating the total length of rope in inches at the end of the month -/
theorem total_rope_inches_is_264 : feet_to_inches total_rope_length = 264 := by
  sorry

end NUMINAMATH_CALUDE_total_rope_inches_is_264_l2479_247932


namespace NUMINAMATH_CALUDE_gretchen_walking_time_l2479_247991

/-- The number of minutes Gretchen should walk for every 90 minutes of sitting -/
def walking_time_per_90_min : ℕ := 10

/-- The number of minutes in 90 minutes -/
def sitting_time_per_break : ℕ := 90

/-- The number of hours Gretchen spends working at her desk -/
def work_hours : ℕ := 6

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the total walking time for Gretchen based on her work hours -/
def total_walking_time : ℕ :=
  (work_hours * minutes_per_hour / sitting_time_per_break) * walking_time_per_90_min

theorem gretchen_walking_time :
  total_walking_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_walking_time_l2479_247991


namespace NUMINAMATH_CALUDE_average_yield_is_100_l2479_247927

/-- Calculates the average yield per tree given the number of trees and their yields. -/
def averageYield (x : ℕ) : ℚ :=
  let trees1 := x + 2
  let trees2 := x
  let trees3 := x - 2
  let yield1 := 30
  let yield2 := 120
  let yield3 := 180
  let totalTrees := trees1 + trees2 + trees3
  let totalNuts := trees1 * yield1 + trees2 * yield2 + trees3 * yield3
  totalNuts / totalTrees

/-- Theorem stating that the average yield per tree is 100 when x = 10. -/
theorem average_yield_is_100 : averageYield 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_average_yield_is_100_l2479_247927


namespace NUMINAMATH_CALUDE_symmetry_of_P_l2479_247962

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis. -/
def symmetry_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The original point P. -/
def P : Point :=
  ⟨-2, -1⟩

theorem symmetry_of_P :
  symmetry_x_axis P = Point.mk (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_P_l2479_247962


namespace NUMINAMATH_CALUDE_smallest_fitting_polygon_l2479_247968

/-- A regular polygon with n sides that can fit perfectly when rotated by 40° or 60° -/
def FittingPolygon (n : ℕ) : Prop :=
  n > 0 ∧ (40 * n) % 360 = 0 ∧ (60 * n) % 360 = 0

/-- The smallest number of sides for a fitting polygon is 18 -/
theorem smallest_fitting_polygon : ∃ (n : ℕ), FittingPolygon n ∧ ∀ m, FittingPolygon m → n ≤ m :=
  sorry

end NUMINAMATH_CALUDE_smallest_fitting_polygon_l2479_247968


namespace NUMINAMATH_CALUDE_polynomial_equality_l2479_247987

theorem polynomial_equality (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 9 * p^8 * q = 36 * p^7 * q^2 → p = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2479_247987


namespace NUMINAMATH_CALUDE_john_task_completion_time_l2479_247976

-- Define a custom type for time
structure Time where
  hours : Nat
  minutes : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

-- Define the problem statement
theorem john_task_completion_time 
  (start_time : Time)
  (two_tasks_end_time : Time)
  (h1 : start_time = { hours := 8, minutes := 30 })
  (h2 : two_tasks_end_time = { hours := 11, minutes := 10 })
  (h3 : ∃ (task_duration : Nat), 
        addMinutes start_time (2 * task_duration) = two_tasks_end_time) :
  addMinutes two_tasks_end_time 
    ((two_tasks_end_time.hours * 60 + two_tasks_end_time.minutes - 
      start_time.hours * 60 - start_time.minutes) / 2) = 
    { hours := 12, minutes := 30 } :=
by sorry

end NUMINAMATH_CALUDE_john_task_completion_time_l2479_247976


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2479_247977

theorem sum_of_four_numbers : 2345 + 3452 + 4523 + 5234 = 15554 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2479_247977


namespace NUMINAMATH_CALUDE_stratified_sample_distribution_l2479_247983

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the total number of students -/
def totalStudents (d : GradeDistribution) : ℕ :=
  d.grade10 + d.grade11 + d.grade12

/-- Represents the sample size for each grade -/
structure SampleDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the total sample size -/
def totalSample (s : SampleDistribution) : ℕ :=
  s.grade10 + s.grade11 + s.grade12

theorem stratified_sample_distribution 
  (population : GradeDistribution)
  (sample : SampleDistribution) :
  totalStudents population = 4000 →
  population.grade10 = 32 * k →
  population.grade11 = 33 * k →
  population.grade12 = 35 * k →
  totalSample sample = 200 →
  sample.grade10 = 64 ∧ sample.grade11 = 66 ∧ sample.grade12 = 70 :=
by sorry


end NUMINAMATH_CALUDE_stratified_sample_distribution_l2479_247983


namespace NUMINAMATH_CALUDE_proportion_problem_l2479_247920

-- Define the proportion relation
def in_proportion (a b c d : ℝ) : Prop := a * d = b * c

-- State the theorem
theorem proportion_problem :
  ∀ (a b c d : ℝ),
  in_proportion a b c d →
  a = 2 →
  b = 3 →
  c = 6 →
  d = 9 := by
sorry

end NUMINAMATH_CALUDE_proportion_problem_l2479_247920


namespace NUMINAMATH_CALUDE_gcd_of_36_and_60_l2479_247914

theorem gcd_of_36_and_60 : Nat.gcd 36 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_36_and_60_l2479_247914


namespace NUMINAMATH_CALUDE_tensor_product_result_l2479_247953

-- Define the sets P and Q
def P : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def Q : Set ℝ := {x | x > 1}

-- Define the ⊗ operation
def tensorProduct (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

-- Theorem statement
theorem tensor_product_result :
  tensorProduct P Q = {x | (0 ≤ x ∧ x ≤ 1) ∨ (x > 2)} := by
  sorry

end NUMINAMATH_CALUDE_tensor_product_result_l2479_247953


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2479_247969

/-- Given a rectangular field with one side uncovered and three sides fenced, 
    calculate its area. -/
theorem rectangular_field_area 
  (L : ℝ) -- Length of the uncovered side
  (total_fencing : ℝ) -- Total length of fencing used
  (h1 : L = 20) -- The uncovered side is 20 feet long
  (h2 : total_fencing = 64) -- The total fencing required is 64 feet
  : L * ((total_fencing - L) / 2) = 440 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2479_247969


namespace NUMINAMATH_CALUDE_sales_tax_difference_l2479_247988

/-- The price of the item before tax -/
def item_price : ℝ := 50

/-- The first sales tax rate -/
def tax_rate_1 : ℝ := 0.075

/-- The second sales tax rate -/
def tax_rate_2 : ℝ := 0.0625

/-- Theorem: The difference between the sales taxes is $0.625 -/
theorem sales_tax_difference : 
  item_price * tax_rate_1 - item_price * tax_rate_2 = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l2479_247988


namespace NUMINAMATH_CALUDE_factorial_expression_equals_1584_l2479_247972

theorem factorial_expression_equals_1584 :
  (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_factorial_expression_equals_1584_l2479_247972


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2479_247993

theorem triangle_angle_calculation (α β γ δ : ℝ) 
  (h1 : α = 120)
  (h2 : β = 30)
  (h3 : γ = 21)
  (h4 : α + (180 - α) = 180) : 
  180 - ((180 - α) + β + γ) = 69 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2479_247993


namespace NUMINAMATH_CALUDE_cube_volume_from_side_area_l2479_247916

theorem cube_volume_from_side_area :
  ∀ (side_area : ℝ) (volume : ℝ),
    side_area = 64 →
    volume = (side_area.sqrt) ^ 3 →
    volume = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_side_area_l2479_247916


namespace NUMINAMATH_CALUDE_even_iff_divisible_by_72_l2479_247926

theorem even_iff_divisible_by_72 (n : ℕ) : 
  Even n ↔ 72 ∣ (3^n + 63) := by sorry

end NUMINAMATH_CALUDE_even_iff_divisible_by_72_l2479_247926


namespace NUMINAMATH_CALUDE_expression_evaluation_l2479_247998

theorem expression_evaluation (a b : ℚ) (h1 : a = 2) (h2 : b = 1/2) :
  (a^3 + b^2)^2 - (a^3 - b^2)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2479_247998


namespace NUMINAMATH_CALUDE_pigeon_count_theorem_l2479_247942

theorem pigeon_count_theorem :
  ∃! n : ℕ,
    300 < n ∧ n < 900 ∧
    n % 2 = 1 ∧
    n % 3 = 2 ∧
    n % 4 = 3 ∧
    n % 5 = 4 ∧
    n % 6 = 5 ∧
    n % 7 = 0 ∧
    n = 539 := by
  sorry

end NUMINAMATH_CALUDE_pigeon_count_theorem_l2479_247942


namespace NUMINAMATH_CALUDE_largest_common_divisor_17_30_l2479_247999

theorem largest_common_divisor_17_30 : 
  ∃ (n : ℕ), n > 0 ∧ n = 13 ∧ 
  (∀ (m : ℕ), m > 0 → 17 % m = 30 % m → m ≤ n) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_17_30_l2479_247999


namespace NUMINAMATH_CALUDE_final_output_is_127_l2479_247997

def flowchart_output : ℕ → ℕ
| 0 => 0
| (n + 1) => let a := flowchart_output n; if a < 100 then 2 * a + 1 else a

theorem final_output_is_127 : flowchart_output 7 = 127 := by
  sorry

end NUMINAMATH_CALUDE_final_output_is_127_l2479_247997


namespace NUMINAMATH_CALUDE_cow_sheep_value_l2479_247986

/-- The value of cows and sheep in taels of gold -/
theorem cow_sheep_value (x y : ℚ) 
  (h1 : 5 * x + 2 * y = 10) 
  (h2 : 2 * x + 5 * y = 8) : 
  x + y = 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cow_sheep_value_l2479_247986


namespace NUMINAMATH_CALUDE_inequality_proof_l2479_247925

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (1 / (2 * x)) + (1 / (2 * y)) + (1 / (2 * z)) > (1 / (y + z)) + (1 / (z + x)) + (1 / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2479_247925


namespace NUMINAMATH_CALUDE_fifteen_factorial_largest_square_exponent_sum_l2479_247967

def largest_perfect_square_exponent_sum (n : ℕ) : ℕ :=
  let prime_factors := Nat.factors n
  let max_square_exponents := prime_factors.map (fun p => (Nat.factorization n p) / 2)
  max_square_exponents.sum

theorem fifteen_factorial_largest_square_exponent_sum :
  largest_perfect_square_exponent_sum (Nat.factorial 15) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_largest_square_exponent_sum_l2479_247967


namespace NUMINAMATH_CALUDE_moe_eating_time_l2479_247970

/-- The time taken for Moe to eat a certain number of cuttlebone pieces -/
theorem moe_eating_time (X : ℝ) : 
  (200 : ℝ) / 800 * X = X / 4 := by sorry

end NUMINAMATH_CALUDE_moe_eating_time_l2479_247970


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2479_247961

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : Real) (time : Real) : Real :=
  let length_km := speed * (time / 3600)
  let length_m := length_km * 1000
  length_m

/-- Proves that a train with speed 60 km/hr crossing a pole in 15 seconds has a length of 250 meters -/
theorem train_length_proof :
  train_length 60 15 = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2479_247961


namespace NUMINAMATH_CALUDE_fish_pond_population_l2479_247959

theorem fish_pond_population (initial_tagged : Nat) (second_catch : Nat) (tagged_in_second : Nat) :
  initial_tagged = 60 →
  second_catch = 60 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = initial_tagged / (1800 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_population_l2479_247959


namespace NUMINAMATH_CALUDE_max_distance_between_spheres_max_distance_achieved_l2479_247950

def sphere1_center : ℝ × ℝ × ℝ := (-4, -10, 5)
def sphere1_radius : ℝ := 20

def sphere2_center : ℝ × ℝ × ℝ := (10, 7, -16)
def sphere2_radius : ℝ := 90

theorem max_distance_between_spheres :
  ∀ (p1 p2 : ℝ × ℝ × ℝ),
  ‖p1 - sphere1_center‖ = sphere1_radius →
  ‖p2 - sphere2_center‖ = sphere2_radius →
  ‖p1 - p2‖ ≤ 140.433 :=
by sorry

theorem max_distance_achieved :
  ∃ (p1 p2 : ℝ × ℝ × ℝ),
  ‖p1 - sphere1_center‖ = sphere1_radius ∧
  ‖p2 - sphere2_center‖ = sphere2_radius ∧
  ‖p1 - p2‖ = 140.433 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_spheres_max_distance_achieved_l2479_247950


namespace NUMINAMATH_CALUDE_tournament_rankings_count_l2479_247933

/-- Represents a player in the tournament -/
inductive Player : Type
| P : Player
| Q : Player
| R : Player
| S : Player

/-- Represents a match between two players -/
structure Match :=
  (player1 : Player)
  (player2 : Player)

/-- Represents the tournament structure -/
structure Tournament :=
  (saturday_match1 : Match)
  (saturday_match2 : Match)
  (sunday_championship : Match)
  (sunday_consolation : Match)

/-- Represents a ranking of players -/
def Ranking := List Player

/-- Returns all possible rankings for a given tournament -/
def possibleRankings (t : Tournament) : List Ranking :=
  sorry

theorem tournament_rankings_count :
  ∀ t : Tournament,
  (t.saturday_match1.player1 = Player.P ∧ t.saturday_match1.player2 = Player.Q) →
  (t.saturday_match2.player1 = Player.R ∧ t.saturday_match2.player2 = Player.S) →
  (List.length (possibleRankings t) = 16) :=
by sorry

end NUMINAMATH_CALUDE_tournament_rankings_count_l2479_247933


namespace NUMINAMATH_CALUDE_probability_of_winning_pair_l2479_247990

/-- Represents the color of a card -/
inductive Color
| Red
| Green

/-- Represents the label of a card -/
inductive Label
| A | B | C | D | E

/-- Represents a card in the deck -/
structure Card where
  color : Color
  label : Label

/-- The deck of cards -/
def deck : Finset Card := sorry

/-- Predicate for a winning pair of cards -/
def is_winning_pair (c1 c2 : Card) : Prop :=
  c1.color = c2.color ∨ c1.label = c2.label

/-- The number of cards in the deck -/
def deck_size : ℕ := sorry

/-- The number of winning pairs -/
def winning_pairs : ℕ := sorry

theorem probability_of_winning_pair :
  (winning_pairs : ℚ) / (deck_size.choose 2 : ℚ) = 51 / 91 := by sorry

end NUMINAMATH_CALUDE_probability_of_winning_pair_l2479_247990


namespace NUMINAMATH_CALUDE_inequality_solutions_l2479_247975

theorem inequality_solutions (a : ℝ) (h1 : a < 0) (h2 : a ≤ -Real.rpow 2 (1/3)) :
  ∃ (w x y z : ℤ), w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
  (a^2 * |a + (w : ℝ)/a^2| + |1 + w| ≤ 1 - a^3) ∧
  (a^2 * |a + (x : ℝ)/a^2| + |1 + x| ≤ 1 - a^3) ∧
  (a^2 * |a + (y : ℝ)/a^2| + |1 + y| ≤ 1 - a^3) ∧
  (a^2 * |a + (z : ℝ)/a^2| + |1 + z| ≤ 1 - a^3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solutions_l2479_247975


namespace NUMINAMATH_CALUDE_final_water_fraction_l2479_247984

def container_volume : ℚ := 20
def replacement_volume : ℚ := 5
def num_replacements : ℕ := 5

def water_fraction_after_replacements : ℚ := (3/4) ^ num_replacements

theorem final_water_fraction :
  water_fraction_after_replacements = 243/1024 :=
by sorry

end NUMINAMATH_CALUDE_final_water_fraction_l2479_247984


namespace NUMINAMATH_CALUDE_smallest_factor_for_cube_l2479_247957

theorem smallest_factor_for_cube (n : ℕ) : n > 0 ∧ n * 49 = (7 : ℕ)^3 ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬∃ k : ℕ, m * 49 = k^3 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_for_cube_l2479_247957


namespace NUMINAMATH_CALUDE_reflection_result_l2479_247913

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 2)
  let reflected := (-p'.2, -p'.1)
  (reflected.1, reflected.2 + 2)

def D : ℝ × ℝ := (5, 0)

theorem reflection_result :
  let D' := reflect_x D
  let D'' := reflect_line D'
  D'' = (2, -3) := by sorry

end NUMINAMATH_CALUDE_reflection_result_l2479_247913


namespace NUMINAMATH_CALUDE_oranges_per_box_l2479_247966

/-- Given a fruit farm that packs oranges, prove that each box contains 10 oranges. -/
theorem oranges_per_box (total_oranges : ℕ) (total_boxes : ℝ) 
  (h1 : total_oranges = 26500) 
  (h2 : total_boxes = 2650.0) : 
  (total_oranges : ℝ) / total_boxes = 10 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l2479_247966


namespace NUMINAMATH_CALUDE_negation_of_existence_square_positive_negation_l2479_247974

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, x < 0 ∧ p x) ↔ (∀ x, x < 0 → ¬ p x) := by sorry

theorem square_positive_negation :
  (¬ ∃ x : ℝ, x < 0 ∧ x^2 > 0) ↔ (∀ x : ℝ, x < 0 → x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_positive_negation_l2479_247974


namespace NUMINAMATH_CALUDE_candy_count_l2479_247960

/-- The number of candy pieces caught by Tabitha and her friends -/
def total_candy (tabitha stan julie carlos veronica benjamin kelly : ℕ) : ℕ :=
  tabitha + stan + julie + carlos + veronica + benjamin + kelly

/-- Theorem stating the total number of candy pieces caught by the friends -/
theorem candy_count : ∃ (tabitha stan julie carlos veronica benjamin kelly : ℕ),
  tabitha = 22 ∧
  stan = tabitha / 3 + 4 ∧
  julie = tabitha / 2 ∧
  carlos = 2 * stan ∧
  veronica = julie + stan - 5 ∧
  benjamin = (tabitha + carlos) / 2 + 9 ∧
  kelly = stan * julie / tabitha ∧
  total_candy tabitha stan julie carlos veronica benjamin kelly = 119 := by
  sorry

#check candy_count

end NUMINAMATH_CALUDE_candy_count_l2479_247960


namespace NUMINAMATH_CALUDE_max_diff_six_digit_even_numbers_l2479_247954

/-- A function that checks if a natural number has only even digits -/
def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d % 2 = 0

/-- A function that checks if a natural number has at least one odd digit -/
def has_odd_digit (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ n.digits 10 ∧ d % 2 ≠ 0

/-- The theorem stating the maximum difference between two 6-digit numbers with the given conditions -/
theorem max_diff_six_digit_even_numbers :
  ∃ (a b : ℕ),
    (100000 ≤ a ∧ a < 1000000) ∧
    (100000 ≤ b ∧ b < 1000000) ∧
    has_only_even_digits a ∧
    has_only_even_digits b ∧
    (∀ n : ℕ, a < n ∧ n < b → has_odd_digit n) ∧
    b - a = 111112 ∧
    (∀ a' b' : ℕ,
      (100000 ≤ a' ∧ a' < 1000000) →
      (100000 ≤ b' ∧ b' < 1000000) →
      has_only_even_digits a' →
      has_only_even_digits b' →
      (∀ n : ℕ, a' < n ∧ n < b' → has_odd_digit n) →
      b' - a' ≤ 111112) :=
by sorry

end NUMINAMATH_CALUDE_max_diff_six_digit_even_numbers_l2479_247954


namespace NUMINAMATH_CALUDE_no_common_points_necessary_not_sufficient_for_parallel_l2479_247951

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a placeholder and should be properly defined

/-- Two lines have no common points -/
def no_common_points (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines having no common points
  sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines being parallel
  sorry

/-- Skew lines: lines that are not parallel and do not intersect -/
def skew (l1 l2 : Line3D) : Prop :=
  no_common_points l1 l2 ∧ ¬parallel l1 l2

theorem no_common_points_necessary_not_sufficient_for_parallel :
  (∀ l1 l2 : Line3D, parallel l1 l2 → no_common_points l1 l2) ∧
  (∃ l1 l2 : Line3D, no_common_points l1 l2 ∧ ¬parallel l1 l2) :=
by
  sorry

#check no_common_points_necessary_not_sufficient_for_parallel

end NUMINAMATH_CALUDE_no_common_points_necessary_not_sufficient_for_parallel_l2479_247951


namespace NUMINAMATH_CALUDE_diamond_operation_result_l2479_247911

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the diamond operation
def diamond : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.three
  | Element.one, Element.four => Element.two
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.two
  | Element.two, Element.four => Element.four
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.two
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.one
  | Element.four, Element.one => Element.two
  | Element.four, Element.two => Element.four
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.three

theorem diamond_operation_result :
  diamond (diamond Element.three Element.one) (diamond Element.four Element.two) = Element.one := by
  sorry

end NUMINAMATH_CALUDE_diamond_operation_result_l2479_247911


namespace NUMINAMATH_CALUDE_union_M_N_intersection_M_complement_N_l2479_247978

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 2)}
def N : Set ℝ := {x | x < 1 ∨ x > 3}

-- Theorem for M ∪ N
theorem union_M_N : M ∪ N = {x | x < 1 ∨ x ≥ 2} := by sorry

-- Theorem for M ∩ (U \ N)
theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_M_N_intersection_M_complement_N_l2479_247978


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2479_247992

theorem quadratic_distinct_roots (a₁ a₂ a₃ a₄ : ℝ) (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄) :
  let discriminant := (a₁ + a₂ + a₃ + a₄)^2 - 4*(a₁*a₃ + a₂*a₄)
  discriminant > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2479_247992


namespace NUMINAMATH_CALUDE_incorrect_statement_l2479_247908

theorem incorrect_statement : ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l2479_247908


namespace NUMINAMATH_CALUDE_descent_time_is_two_hours_l2479_247940

/-- Proves that the time taken to descend a hill is 2 hours, given specific conditions. -/
theorem descent_time_is_two_hours 
  (time_to_top : ℝ) 
  (avg_speed_total : ℝ) 
  (avg_speed_up : ℝ) 
  (time_to_top_is_four : time_to_top = 4)
  (avg_speed_total_is_three : avg_speed_total = 3)
  (avg_speed_up_is_two_point_two_five : avg_speed_up = 2.25) :
  let distance_to_top : ℝ := avg_speed_up * time_to_top
  let total_distance : ℝ := 2 * distance_to_top
  let total_time : ℝ := total_distance / avg_speed_total
  time_to_top - (total_time - time_to_top) = 2 := by
  sorry

#check descent_time_is_two_hours

end NUMINAMATH_CALUDE_descent_time_is_two_hours_l2479_247940


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_nineteen_fourths_l2479_247995

theorem floor_plus_self_eq_nineteen_fourths :
  ∃! (x : ℚ), ⌊x⌋ + x = 19/4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_nineteen_fourths_l2479_247995


namespace NUMINAMATH_CALUDE_monthly_income_of_P_l2479_247981

/-- Given the average monthly incomes of three individuals P, Q, and R,
    prove that the monthly income of P is 4000. -/
theorem monthly_income_of_P (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  (P + R) / 2 = 5200 →
  P = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_of_P_l2479_247981


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_negative_five_squared_l2479_247949

theorem arithmetic_square_root_of_negative_five_squared (x : ℝ) : 
  x = 5 ∧ x * x = (-5)^2 ∧ x ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_negative_five_squared_l2479_247949


namespace NUMINAMATH_CALUDE_arrangements_count_l2479_247947

/-- The number of different arrangements of 5 students (2 girls and 3 boys) 
    where the two girls are not next to each other -/
def num_arrangements : ℕ := 72

/-- The number of ways to arrange 3 boys -/
def boy_arrangements : ℕ := 6

/-- The number of ways to insert 2 girls into 4 possible spaces -/
def girl_insertions : ℕ := 12

theorem arrangements_count : 
  num_arrangements = boy_arrangements * girl_insertions :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l2479_247947


namespace NUMINAMATH_CALUDE_joe_egg_hunt_l2479_247937

theorem joe_egg_hunt (park_eggs : ℕ) (town_hall_eggs : ℕ) (total_eggs : ℕ) 
  (h1 : park_eggs = 5)
  (h2 : town_hall_eggs = 3)
  (h3 : total_eggs = 20) :
  total_eggs - park_eggs - town_hall_eggs = 12 := by
  sorry

end NUMINAMATH_CALUDE_joe_egg_hunt_l2479_247937


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l2479_247903

theorem solve_quadratic_equation (x : ℝ) (h1 : 3 * x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l2479_247903


namespace NUMINAMATH_CALUDE_f_properties_l2479_247973

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_properties :
  (∃ (x_min : ℝ), f 1 x_min = 2 ∧ ∀ x, f 1 x ≥ f 1 x_min) ∧
  (∀ a ≤ 0, ∀ x y, x < y → f a x > f a y) ∧
  (∀ a > 0, ∀ x y, x < y → 
    ((x < -Real.log a ∧ y < -Real.log a) → f a x > f a y) ∧
    ((x > -Real.log a ∧ y > -Real.log a) → f a x < f a y)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2479_247973


namespace NUMINAMATH_CALUDE_factorize_ax_minus_ay_l2479_247900

theorem factorize_ax_minus_ay (a x y : ℝ) : a * x - a * y = a * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorize_ax_minus_ay_l2479_247900


namespace NUMINAMATH_CALUDE_lcm_24_90_l2479_247980

theorem lcm_24_90 : Nat.lcm 24 90 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_90_l2479_247980


namespace NUMINAMATH_CALUDE_cookies_in_class_l2479_247919

/-- The number of cookies brought by Mona, Jasmine, and Rachel -/
def total_cookies (mona jasmine rachel : ℕ) : ℕ := mona + jasmine + rachel

/-- Theorem stating the total number of cookies brought to class -/
theorem cookies_in_class : ∃ (jasmine rachel : ℕ),
  jasmine = 20 - 5 ∧ 
  rachel = jasmine + 10 ∧
  total_cookies 20 jasmine rachel = 60 := by
sorry

end NUMINAMATH_CALUDE_cookies_in_class_l2479_247919


namespace NUMINAMATH_CALUDE_meaningful_expression_l2479_247944

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x + 2)) ↔ x > -2 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2479_247944


namespace NUMINAMATH_CALUDE_chess_tournament_theorem_l2479_247928

/-- Represents a player's score sequence in the chess tournament -/
structure PlayerScore where
  round1 : ℕ
  round2 : ℕ
  round3 : ℕ
  round4 : ℕ

/-- Checks if a sequence is quadratic -/
def isQuadraticSequence (s : PlayerScore) : Prop :=
  ∃ a t r : ℕ, 
    s.round1 = a ∧
    s.round2 = a + t + r ∧
    s.round3 = a + 2*t + 4*r ∧
    s.round4 = a + 3*t + 9*r

/-- Checks if a sequence is arithmetic -/
def isArithmeticSequence (s : PlayerScore) : Prop :=
  ∃ b d : ℕ, 
    s.round1 = b ∧
    s.round2 = b + d ∧
    s.round3 = b + 2*d ∧
    s.round4 = b + 3*d

/-- Calculates the total score of a player -/
def totalScore (s : PlayerScore) : ℕ :=
  s.round1 + s.round2 + s.round3 + s.round4

/-- The main theorem -/
theorem chess_tournament_theorem 
  (playerA playerB : PlayerScore)
  (h1 : isQuadraticSequence playerA)
  (h2 : isArithmeticSequence playerB)
  (h3 : totalScore playerA = totalScore playerB)
  (h4 : totalScore playerA ≤ 25)
  (h5 : totalScore playerB ≤ 25) :
  playerA.round1 + playerA.round2 + playerB.round1 + playerB.round2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_theorem_l2479_247928


namespace NUMINAMATH_CALUDE_complement_of_M_l2479_247935

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 3, 5}

theorem complement_of_M : (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2479_247935


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2479_247934

theorem complex_magnitude_problem (z : ℂ) (h : Complex.I * z = 2 + 4 * Complex.I) :
  Complex.abs z = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2479_247934


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l2479_247904

-- Define the operation ⊗
noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 5 y = 15 ∧ y = 90 := by sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l2479_247904


namespace NUMINAMATH_CALUDE_function_change_proof_l2479_247963

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The initial x value -/
def x_initial : ℝ := 2

/-- The final x value -/
def x_final : ℝ := 2.5

/-- The change in x -/
def delta_x : ℝ := x_final - x_initial

theorem function_change_proof :
  f x_final - f x_initial = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_function_change_proof_l2479_247963


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l2479_247923

/-- Calculates the total amount a buyer pays for a cycle, given the initial cost,
    loss percentage, and sales tax percentage. -/
def totalCyclePrice (initialCost : ℚ) (lossPercentage : ℚ) (salesTaxPercentage : ℚ) : ℚ :=
  let sellingPrice := initialCost * (1 - lossPercentage / 100)
  let salesTax := sellingPrice * (salesTaxPercentage / 100)
  sellingPrice + salesTax

/-- Theorem stating that for a cycle bought at Rs. 1400, sold at 20% loss,
    with 5% sales tax, the total price is Rs. 1176. -/
theorem cycle_price_calculation :
  totalCyclePrice 1400 20 5 = 1176 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l2479_247923


namespace NUMINAMATH_CALUDE_smallest_angle_of_quadrilateral_with_ratio_l2479_247956

/-- 
Given a quadrilateral with interior angles in a 4:5:6:7 ratio,
prove that the smallest interior angle measures 720/11 degrees.
-/
theorem smallest_angle_of_quadrilateral_with_ratio (a b c d : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- All angles are positive
  b = (5/4) * a ∧ c = (6/4) * a ∧ d = (7/4) * a →  -- Angles are in 4:5:6:7 ratio
  a + b + c + d = 360 →  -- Sum of angles in a quadrilateral is 360°
  a = 720 / 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_of_quadrilateral_with_ratio_l2479_247956


namespace NUMINAMATH_CALUDE_evaluate_expression_l2479_247982

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 2) :
  4 * x^(y + 1) + 5 * y^(x + 1) = 188 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2479_247982


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2479_247906

/-- Proves that the weight of the replaced person is 55 kg given the conditions of the problem. -/
theorem replaced_person_weight
  (initial_count : Nat)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : initial_count = 8)
  (h2 : weight_increase = 2.5)
  (h3 : new_person_weight = 75) :
  (new_person_weight - initial_count * weight_increase) = 55 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2479_247906


namespace NUMINAMATH_CALUDE_M_equals_P_l2479_247996

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def P : Set ℝ := {a | ∃ x : ℝ, a = x^2 - 1}

theorem M_equals_P : M = P := by sorry

end NUMINAMATH_CALUDE_M_equals_P_l2479_247996


namespace NUMINAMATH_CALUDE_intersection_of_specific_lines_l2479_247955

/-- Two lines in a plane -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- The point of intersection of two lines -/
def intersection (l1 l2 : Line) : ℚ × ℚ :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  (x, y)

/-- Theorem: The intersection of y = -3x + 1 and y + 1 = 15x is (1/9, 2/3) -/
theorem intersection_of_specific_lines :
  let line1 : Line := { slope := -3, intercept := 1 }
  let line2 : Line := { slope := 15, intercept := -1 }
  intersection line1 line2 = (1/9, 2/3) := by
sorry

end NUMINAMATH_CALUDE_intersection_of_specific_lines_l2479_247955


namespace NUMINAMATH_CALUDE_abc_sum_and_squares_l2479_247939

theorem abc_sum_and_squares (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  (a*b + b*c + c*a = -1/2) ∧ (a^4 + b^4 + c^4 = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_and_squares_l2479_247939


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2479_247946

theorem complex_fraction_simplification :
  ((-1 : ℂ) + 3*Complex.I) / (1 + Complex.I) = 1 + 2*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2479_247946


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2479_247964

/-- The line ax-y+2a=0 (a∈R) intersects the circle x^2+y^2=5 -/
theorem line_intersects_circle (a : ℝ) : 
  ∃ (x y : ℝ), (a * x - y + 2 * a = 0) ∧ (x^2 + y^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2479_247964


namespace NUMINAMATH_CALUDE_absolute_value_equation_implies_zero_product_l2479_247924

theorem absolute_value_equation_implies_zero_product (x y : ℝ) (hy : y > 0) :
  |x - Real.log (y^2)| = x + Real.log (y^2) → x * (y - 1)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_implies_zero_product_l2479_247924


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_cos_2alpha_l2479_247917

theorem sin_2alpha_minus_cos_2alpha (α : Real) (h : Real.tan α = 3) :
  Real.sin (2 * α) - Real.cos (2 * α) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_cos_2alpha_l2479_247917


namespace NUMINAMATH_CALUDE_edmund_earns_64_dollars_l2479_247945

/-- Calculates the amount Edmund earns for extra chores over two weeks -/
def edmunds_earnings (normal_chores_per_week : ℕ) (chores_per_day : ℕ) 
  (days : ℕ) (payment_per_extra_chore : ℕ) : ℕ :=
  let total_chores := chores_per_day * days
  let normal_total_chores := normal_chores_per_week * (days / 7)
  let extra_chores := total_chores - normal_total_chores
  extra_chores * payment_per_extra_chore

/-- Theorem stating that Edmund earns $64 for extra chores over two weeks -/
theorem edmund_earns_64_dollars :
  edmunds_earnings 12 4 14 2 = 64 := by
  sorry


end NUMINAMATH_CALUDE_edmund_earns_64_dollars_l2479_247945


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l2479_247958

theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = Real.sqrt 13 ∧
  ∀ (x y : ℝ), x^2 / 9 + y^2 / 4 = 1 →
  x + y ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l2479_247958


namespace NUMINAMATH_CALUDE_envelope_printing_equation_l2479_247994

/-- The equation for two envelope-printing machines to print 500 envelopes in 2 minutes -/
theorem envelope_printing_equation (x : ℝ) : x > 0 → 500 / 8 + 500 / x = 500 / 2 := by
  sorry

end NUMINAMATH_CALUDE_envelope_printing_equation_l2479_247994


namespace NUMINAMATH_CALUDE_chess_board_numbering_specific_cell_number_l2479_247915

/-- Represents the numbering system of an infinite chessboard where each cell
    is assigned the smallest possible number not yet used for numbering any
    preceding cells in the same row or column. -/
noncomputable def chessBoardNumber (row : Nat) (col : Nat) : Nat :=
  sorry

/-- The number assigned to a cell is equal to the XOR of (row - 1) and (col - 1) -/
theorem chess_board_numbering (row col : Nat) :
  chessBoardNumber row col = Nat.xor (row - 1) (col - 1) :=
sorry

/-- The cell at the intersection of the 100th row and the 1000th column
    receives the number 921 -/
theorem specific_cell_number :
  chessBoardNumber 100 1000 = 921 :=
sorry

end NUMINAMATH_CALUDE_chess_board_numbering_specific_cell_number_l2479_247915


namespace NUMINAMATH_CALUDE_det_matrix_2x2_l2479_247912

theorem det_matrix_2x2 (A : Matrix (Fin 2) (Fin 2) ℝ) : 
  A = ![![6, -2], ![-5, 3]] → Matrix.det A = 8 := by
  sorry

end NUMINAMATH_CALUDE_det_matrix_2x2_l2479_247912


namespace NUMINAMATH_CALUDE_cookie_orange_cost_ratio_l2479_247901

/-- Cost of items in Susie and Calvin's purchases -/
structure ItemCosts where
  orange : ℚ
  muffin : ℚ
  cookie : ℚ

/-- Susie's purchase -/
def susie_purchase (costs : ItemCosts) : ℚ :=
  3 * costs.muffin + 5 * costs.orange

/-- Calvin's purchase -/
def calvin_purchase (costs : ItemCosts) : ℚ :=
  5 * costs.muffin + 10 * costs.orange + 4 * costs.cookie

theorem cookie_orange_cost_ratio :
  ∀ (costs : ItemCosts),
  costs.muffin = 2 * costs.orange →
  calvin_purchase costs = 3 * susie_purchase costs →
  costs.cookie = (13/4) * costs.orange :=
by sorry

end NUMINAMATH_CALUDE_cookie_orange_cost_ratio_l2479_247901


namespace NUMINAMATH_CALUDE_two_thousand_eighth_number_without_two_l2479_247922

/-- A function that checks if a positive integer contains the digit 2 -/
def containsTwo (n : ℕ) : Bool :=
  String.contains (toString n) '2'

/-- A function that generates the sequence of numbers without 2 -/
def sequenceWithoutTwo : ℕ → ℕ
  | 0 => 0
  | n + 1 => if containsTwo (sequenceWithoutTwo n + 1)
              then sequenceWithoutTwo n + 2
              else sequenceWithoutTwo n + 1

theorem two_thousand_eighth_number_without_two :
  sequenceWithoutTwo 2008 = 3781 := by
  sorry

end NUMINAMATH_CALUDE_two_thousand_eighth_number_without_two_l2479_247922


namespace NUMINAMATH_CALUDE_min_value_of_u_l2479_247943

theorem min_value_of_u (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (x + 1/x) * (y + 1/(4*y)) ≥ 25/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_u_l2479_247943


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2479_247948

theorem fraction_sum_equality : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2479_247948


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l2479_247902

theorem average_of_four_numbers (x y z w : ℝ) 
  (h : (5 / 2) * (x + y + z + w) = 25) : 
  (x + y + z + w) / 4 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l2479_247902


namespace NUMINAMATH_CALUDE_monotone_decreasing_range_a_l2479_247936

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4*a*x + 3 else (2 - 3*a)*x + 1

theorem monotone_decreasing_range_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → a ∈ Set.Icc (1/2) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_range_a_l2479_247936


namespace NUMINAMATH_CALUDE_semicircle_radius_l2479_247909

/-- The radius of a semi-circle given its perimeter -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 113) : 
  ∃ (radius : ℝ), radius = perimeter / (Real.pi + 2) := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l2479_247909


namespace NUMINAMATH_CALUDE_runner_ends_at_start_l2479_247979

/-- A runner on a circular track -/
structure Runner where
  start_position : ℝ  -- Position on the track (0 ≤ position < track_length)
  distance_run : ℝ    -- Total distance run
  track_length : ℝ    -- Length of the circular track

/-- Theorem: A runner who completes an integer number of laps ends at the starting position -/
theorem runner_ends_at_start (runner : Runner) (h : runner.track_length > 0) :
  runner.distance_run % runner.track_length = 0 →
  (runner.start_position + runner.distance_run) % runner.track_length = runner.start_position :=
by sorry

end NUMINAMATH_CALUDE_runner_ends_at_start_l2479_247979


namespace NUMINAMATH_CALUDE_product_remainder_l2479_247929

theorem product_remainder (a b c : ℕ) (ha : a = 2457) (hb : b = 7623) (hc : c = 91309) : 
  (a * b * c) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2479_247929


namespace NUMINAMATH_CALUDE_prove_a_equals_3x_l2479_247905

theorem prove_a_equals_3x (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27*x^3) 
  (h3 : a - b = 3*x) : 
  a = 3*x := by sorry

end NUMINAMATH_CALUDE_prove_a_equals_3x_l2479_247905


namespace NUMINAMATH_CALUDE_percentage_of_employees_6_years_or_more_l2479_247989

/-- Represents the distribution of employees' duration of service at the Fermat Company -/
structure EmployeeDistribution :=
  (less_than_1_year : ℕ)
  (from_1_to_1_5_years : ℕ)
  (from_1_5_to_2_5_years : ℕ)
  (from_2_5_to_3_5_years : ℕ)
  (from_3_5_to_4_5_years : ℕ)
  (from_4_5_to_5_5_years : ℕ)
  (from_5_5_to_6_5_years : ℕ)
  (from_6_5_to_7_5_years : ℕ)
  (from_7_5_to_8_5_years : ℕ)
  (from_8_5_to_10_years : ℕ)

/-- Calculates the total number of employees -/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1_year + d.from_1_to_1_5_years + d.from_1_5_to_2_5_years +
  d.from_2_5_to_3_5_years + d.from_3_5_to_4_5_years + d.from_4_5_to_5_5_years +
  d.from_5_5_to_6_5_years + d.from_6_5_to_7_5_years + d.from_7_5_to_8_5_years +
  d.from_8_5_to_10_years

/-- Calculates the number of employees who have worked for 6 years or more -/
def employees_6_years_or_more (d : EmployeeDistribution) : ℕ :=
  d.from_5_5_to_6_5_years + d.from_6_5_to_7_5_years + d.from_7_5_to_8_5_years +
  d.from_8_5_to_10_years

/-- The theorem to be proved -/
theorem percentage_of_employees_6_years_or_more
  (d : EmployeeDistribution)
  (h1 : d.less_than_1_year = 4)
  (h2 : d.from_1_to_1_5_years = 6)
  (h3 : d.from_1_5_to_2_5_years = 7)
  (h4 : d.from_2_5_to_3_5_years = 4)
  (h5 : d.from_3_5_to_4_5_years = 3)
  (h6 : d.from_4_5_to_5_5_years = 3)
  (h7 : d.from_5_5_to_6_5_years = 2)
  (h8 : d.from_6_5_to_7_5_years = 1)
  (h9 : d.from_7_5_to_8_5_years = 1)
  (h10 : d.from_8_5_to_10_years = 1) :
  (employees_6_years_or_more d : ℚ) / (total_employees d : ℚ) = 5 / 32 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_employees_6_years_or_more_l2479_247989


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l2479_247985

/-- 
Given a quadratic equation (m-2)x^2 + 2x + 1 = 0, this theorem states that 
for the equation to have two distinct real roots, m must be less than 3 and not equal to 2.
-/
theorem quadratic_distinct_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (m - 2) * x^2 + 2 * x + 1 = 0 ∧ 
   (m - 2) * y^2 + 2 * y + 1 = 0) ↔ 
  (m < 3 ∧ m ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l2479_247985


namespace NUMINAMATH_CALUDE_simplify_expression_l2479_247918

theorem simplify_expression (a : ℝ) (h : a ≠ 1) :
  1 - 1 / (1 + (a + 2) / (1 - a)) = (2 + a) / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2479_247918


namespace NUMINAMATH_CALUDE_mass_ratio_simplification_l2479_247965

-- Define the units
def kg : ℚ → ℚ := id
def ton : ℚ → ℚ := (· * 1000)

-- Define the ratio
def ratio (a b : ℚ) : ℚ × ℚ := (a, b)

-- Define the problem
theorem mass_ratio_simplification :
  let mass1 := kg 250
  let mass2 := ton 0.5
  let simplified_ratio := ratio 1 2
  let decimal_value := 0.5
  (mass1 / mass2 = decimal_value) ∧
  (ratio (mass1 / gcd mass1 mass2) (mass2 / gcd mass1 mass2) = simplified_ratio) := by
  sorry


end NUMINAMATH_CALUDE_mass_ratio_simplification_l2479_247965


namespace NUMINAMATH_CALUDE_equation_solutions_l2479_247931

theorem equation_solutions : 
  ∀ m : ℝ, 9 * m^2 - (2*m + 1)^2 = 0 ↔ m = 1 ∨ m = -1/5 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2479_247931


namespace NUMINAMATH_CALUDE_lisa_max_below_a_l2479_247938

/-- Represents Lisa's quiz performance and goal --/
structure QuizPerformance where
  total_quizzes : ℕ
  goal_percentage : ℚ
  completed_quizzes : ℕ
  completed_as : ℕ

/-- Calculates the maximum number of remaining quizzes where Lisa can score below 'A' --/
def max_below_a (qp : QuizPerformance) : ℕ :=
  let total_as_needed : ℕ := (qp.goal_percentage * qp.total_quizzes).ceil.toNat
  let remaining_quizzes : ℕ := qp.total_quizzes - qp.completed_quizzes
  remaining_quizzes - (total_as_needed - qp.completed_as)

/-- Theorem stating that given Lisa's quiz performance, the maximum number of remaining quizzes where she can score below 'A' is 7 --/
theorem lisa_max_below_a :
  let qp : QuizPerformance := {
    total_quizzes := 60,
    goal_percentage := 3/4,
    completed_quizzes := 30,
    completed_as := 22
  }
  max_below_a qp = 7 := by sorry

end NUMINAMATH_CALUDE_lisa_max_below_a_l2479_247938


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2479_247941

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2479_247941


namespace NUMINAMATH_CALUDE_completing_square_result_l2479_247910

theorem completing_square_result (x : ℝ) : x^2 + 4*x - 1 = 0 → (x + 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l2479_247910


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2479_247930

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, b.1 = t * a.1 ∧ b.2 = t * a.2

theorem parallel_vectors_k_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (k, 4)
  are_parallel a b → k = -2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2479_247930


namespace NUMINAMATH_CALUDE_function_value_at_half_l2479_247921

def real_function_property (f : ℝ → ℝ) : Prop :=
  f 1 = -1 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

theorem function_value_at_half (f : ℝ → ℝ) (h : real_function_property f) : f (1/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_half_l2479_247921


namespace NUMINAMATH_CALUDE_condition_relationship_l2479_247907

/-- Given propositions p, q, and r, we define what it means for one proposition
    to be a sufficient but not necessary condition for another. -/
def sufficient_not_necessary (a b : Prop) : Prop :=
  (a → b) ∧ ¬(b → a)

/-- Given propositions p, q, and r, we define what it means for one proposition
    to be a necessary but not sufficient condition for another. -/
def necessary_not_sufficient (a b : Prop) : Prop :=
  (b → a) ∧ ¬(a → b)

/-- Theorem stating the relationship between p, q, and r based on their conditional properties. -/
theorem condition_relationship (p q r : Prop) 
  (h1 : sufficient_not_necessary p q) 
  (h2 : sufficient_not_necessary q r) : 
  necessary_not_sufficient r p :=
sorry

end NUMINAMATH_CALUDE_condition_relationship_l2479_247907


namespace NUMINAMATH_CALUDE_line_symmetry_l2479_247952

/-- The equation of a line in the Cartesian plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry of a point about another point -/
def symmetric_point (p q : Point) : Point :=
  ⟨2 * q.x - p.x, 2 * q.y - p.y⟩

/-- Two lines are symmetric about a point if for any point on one line,
    its symmetric point about the given point lies on the other line -/
def symmetric_lines (l₁ l₂ : Line) (p : Point) : Prop :=
  ∀ x y : ℝ, l₁.a * x + l₁.b * y + l₁.c = 0 →
    let sym := symmetric_point ⟨x, y⟩ p
    l₂.a * sym.x + l₂.b * sym.y + l₂.c = 0

theorem line_symmetry :
  let l₁ : Line := ⟨3, -1, 2, sorry⟩
  let l₂ : Line := ⟨3, -1, -6, sorry⟩
  let p : Point := ⟨1, 1⟩
  symmetric_lines l₁ l₂ p := by sorry

end NUMINAMATH_CALUDE_line_symmetry_l2479_247952
