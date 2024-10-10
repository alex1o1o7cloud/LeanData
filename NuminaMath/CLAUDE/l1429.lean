import Mathlib

namespace fraction_multiplication_addition_l1429_142988

theorem fraction_multiplication_addition : (2 : ℚ) / 9 * 5 / 8 + 1 / 4 = 7 / 18 := by
  sorry

end fraction_multiplication_addition_l1429_142988


namespace polygon_sides_l1429_142987

/-- Theorem: For a polygon with n sides, if the sum of its interior angles is 180° less than three times the sum of its exterior angles, then n = 7. -/
theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 := by
  sorry

end polygon_sides_l1429_142987


namespace age_puzzle_l1429_142913

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 18) (h2 : 3 * (A + x) - 3 * (A - 3) = A) : x = 3 := by
  sorry

end age_puzzle_l1429_142913


namespace pet_store_puppies_l1429_142951

theorem pet_store_puppies (sold : ℕ) (cages : ℕ) (puppies_per_cage : ℕ) 
  (h1 : sold = 24)
  (h2 : cages = 8)
  (h3 : puppies_per_cage = 4) :
  sold + cages * puppies_per_cage = 56 := by
  sorry

end pet_store_puppies_l1429_142951


namespace parallelogram_side_length_l1429_142994

theorem parallelogram_side_length 
  (s : ℝ) 
  (side1 : ℝ) 
  (side2 : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h : side1 = 3 * s) 
  (h' : side2 = s) 
  (h'' : angle = π / 3) 
  (h''' : area = 9 * Real.sqrt 3) 
  (h'''' : area = side2 * side1 * Real.sin angle) : 
  s = Real.sqrt 6 := by
sorry

end parallelogram_side_length_l1429_142994


namespace limit_at_one_l1429_142978

-- Define the function f
def f (x : ℝ) : ℝ := x

-- State the theorem
theorem limit_at_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 1| < ε :=
by
  sorry

end limit_at_one_l1429_142978


namespace sparrow_seeds_count_l1429_142954

theorem sparrow_seeds_count : ∃ n : ℕ+, 
  (9 * n < 1001) ∧ 
  (10 * n > 1100) ∧ 
  (n = 111) := by
sorry

end sparrow_seeds_count_l1429_142954


namespace cos_45_sin_15_minus_sin_45_cos_15_l1429_142945

theorem cos_45_sin_15_minus_sin_45_cos_15 :
  Real.cos (45 * π / 180) * Real.sin (15 * π / 180) - 
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) = -1/2 := by
  sorry

end cos_45_sin_15_minus_sin_45_cos_15_l1429_142945


namespace bigger_number_problem_l1429_142968

theorem bigger_number_problem (x y : ℝ) 
  (sum_eq : x + y = 77)
  (ratio_eq : 5 * x = 6 * y)
  (x_geq_y : x ≥ y) : 
  x = 42 := by
  sorry

end bigger_number_problem_l1429_142968


namespace problem_solution_l1429_142998

def P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}
def Q (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem problem_solution :
  (∀ x, x ∉ P ↔ (x < -2 ∨ x > 10)) ∧
  (∀ m, P ⊆ Q m ↔ m ≥ 9) ∧
  (∀ m, P ∩ Q m = Q m ↔ m ≤ 9) := by sorry

end problem_solution_l1429_142998


namespace base_conversion_theorem_l1429_142935

theorem base_conversion_theorem (n A B : ℕ) : 
  (0 < n) →
  (0 ≤ A) ∧ (A < 8) →
  (0 ≤ B) ∧ (B < 6) →
  (n = 8 * A + B) →
  (n = 6 * B + A) →
  n = 47 :=
by sorry

end base_conversion_theorem_l1429_142935


namespace remainder_7n_mod_4_l1429_142983

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l1429_142983


namespace fundraiser_total_l1429_142975

def fundraiser (sasha_muffins : ℕ) (sasha_price : ℕ)
               (melissa_multiplier : ℕ) (melissa_price : ℕ)
               (tiffany_price : ℕ)
               (sarah_muffins : ℕ) (sarah_price : ℕ)
               (damien_dozens : ℕ) (damien_price : ℕ) : ℕ :=
  let melissa_muffins := melissa_multiplier * sasha_muffins
  let tiffany_muffins := (sasha_muffins + melissa_muffins) / 2
  let damien_muffins := damien_dozens * 12
  (sasha_muffins * sasha_price) +
  (melissa_muffins * melissa_price) +
  (tiffany_muffins * tiffany_price) +
  (sarah_muffins * sarah_price) +
  (damien_muffins * damien_price)

theorem fundraiser_total :
  fundraiser 30 4 4 3 5 50 2 2 6 = 1099 := by
  sorry

end fundraiser_total_l1429_142975


namespace cube_root_equation_solution_l1429_142932

theorem cube_root_equation_solution :
  ∀ x : ℝ, (5 + x / 3) ^ (1/3 : ℝ) = 2 → x = 9 := by
  sorry

end cube_root_equation_solution_l1429_142932


namespace quadratic_root_l1429_142982

theorem quadratic_root (a b c : ℝ) (h : a ≠ 0 ∧ b + c ≠ 0) :
  let f : ℝ → ℝ := λ x => a * (b + c) * x^2 - b * (c + a) * x - c * (a + b)
  (f (-1) = 0) → (f (c * (a + b) / (a * (b + c))) = 0) :=
by sorry

end quadratic_root_l1429_142982


namespace x_value_from_fraction_equality_l1429_142910

theorem x_value_from_fraction_equality (x y : ℝ) :
  x / (x - 1) = (y^2 + 2*y + 3) / (y^2 + 2*y - 2) →
  x = (y^2 + 2*y + 3) / 5 := by
sorry

end x_value_from_fraction_equality_l1429_142910


namespace election_abstention_percentage_l1429_142927

theorem election_abstention_percentage 
  (total_members : ℕ) 
  (votes_cast : ℕ) 
  (candidate_a_percentage : ℚ) 
  (candidate_b_percentage : ℚ) 
  (candidate_c_percentage : ℚ) 
  (candidate_d_percentage : ℚ) 
  (h1 : total_members = 1600) 
  (h2 : votes_cast = 900) 
  (h3 : candidate_a_percentage = 45/100) 
  (h4 : candidate_b_percentage = 35/100) 
  (h5 : candidate_c_percentage = 15/100) 
  (h6 : candidate_d_percentage = 5/100) 
  (h7 : candidate_a_percentage + candidate_b_percentage + candidate_c_percentage + candidate_d_percentage = 1) :
  (total_members - votes_cast : ℚ) / total_members * 100 = 43.75 := by
sorry

end election_abstention_percentage_l1429_142927


namespace modulus_of_complex_number_l1429_142976

theorem modulus_of_complex_number : 
  let z : ℂ := (1 + 3 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 5 := by sorry

end modulus_of_complex_number_l1429_142976


namespace battle_station_staffing_l1429_142997

theorem battle_station_staffing (total_resumes : ℕ) (suitable_fraction : ℚ) 
  (job_openings : ℕ) (h1 : total_resumes = 30) (h2 : suitable_fraction = 2/3) 
  (h3 : job_openings = 5) :
  (total_resumes : ℚ) * suitable_fraction * 
  (total_resumes : ℚ) * suitable_fraction - 1 * 
  (total_resumes : ℚ) * suitable_fraction - 2 * 
  (total_resumes : ℚ) * suitable_fraction - 3 * 
  (total_resumes : ℚ) * suitable_fraction - 4 = 930240 := by
  sorry

end battle_station_staffing_l1429_142997


namespace max_slope_product_l1429_142937

theorem max_slope_product (m₁ m₂ : ℝ) : 
  (m₁ = 5 * m₂) →                    -- One slope is 5 times the other
  (|((m₂ - m₁) / (1 + m₁ * m₂))| = 1) →  -- Lines intersect at 45° angle
  (∀ n₁ n₂ : ℝ, (n₁ = 5 * n₂) → (|((n₂ - n₁) / (1 + n₁ * n₂))| = 1) → m₁ * m₂ ≥ n₁ * n₂) →
  m₁ * m₂ = 1.8 :=
by sorry

end max_slope_product_l1429_142937


namespace greatest_integer_quadratic_inequality_l1429_142955

theorem greatest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 13*n + 30 < 0 → n ≤ 9 :=
by sorry

end greatest_integer_quadratic_inequality_l1429_142955


namespace hoseok_position_l1429_142962

theorem hoseok_position (n : Nat) (h : n = 9) :
  ∀ (position_tallest : Nat), position_tallest = 5 →
    n + 1 - position_tallest = 5 :=
by sorry

end hoseok_position_l1429_142962


namespace condition_analysis_l1429_142972

theorem condition_analysis (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 1 → a + Real.log b > b + Real.log a) ∧
  (∃ a b : ℝ, a + Real.log b > b + Real.log a ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end condition_analysis_l1429_142972


namespace quadrilateral_diagonal_intersection_l1429_142979

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_intersection 
  (ABCD : Quadrilateral) 
  (hConvex : isConvex ABCD) 
  (hAB : distance ABCD.A ABCD.B = 10)
  (hCD : distance ABCD.C ABCD.D = 15)
  (hAC : distance ABCD.A ABCD.C = 17)
  (E : Point)
  (hE : E = lineIntersection ABCD.A ABCD.C ABCD.B ABCD.D)
  (hAreas : triangleArea ABCD.A E ABCD.D = triangleArea ABCD.B E ABCD.C) :
  distance ABCD.A E = 17 / 2 := by
  sorry

end quadrilateral_diagonal_intersection_l1429_142979


namespace street_running_distances_l1429_142923

/-- Represents the distance run around a square block -/
def run_distance (block_side : ℝ) (street_width : ℝ) (position : ℕ) : ℝ :=
  match position with
  | 0 => 4 * (block_side - 2 * street_width) -- inner side
  | 1 => 4 * block_side -- block side
  | 2 => 4 * (block_side + 2 * street_width) -- outer side
  | _ => 0 -- invalid position

theorem street_running_distances 
  (block_side : ℝ) (street_width : ℝ) 
  (h1 : block_side = 500) 
  (h2 : street_width = 25) : 
  run_distance block_side street_width 2 - run_distance block_side street_width 1 = 200 ∧
  run_distance block_side street_width 1 - run_distance block_side street_width 0 = 200 :=
by
  sorry

end street_running_distances_l1429_142923


namespace expression_evaluation_l1429_142957

theorem expression_evaluation : 
  let a := (1/4 + 1/12 - 7/18 - 1/36 : ℚ)
  let part1 := (1/36 : ℚ) / a
  let part2 := a / (1/36 : ℚ)
  part1 * part2 = 1 → part1 + part2 = -10/3 := by
sorry

end expression_evaluation_l1429_142957


namespace polar_to_rectangular_conversion_l1429_142948

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3
  let θ : ℝ := 3 * π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (-3 / Real.sqrt 2, 3 / Real.sqrt 2) := by sorry

end polar_to_rectangular_conversion_l1429_142948


namespace system_solution_l1429_142916

theorem system_solution : 
  {(x, y) : ℝ × ℝ | x^2 + y^2 + x + y = 50 ∧ x * y = 20} = 
  {(5, 4), (4, 5), (-5 + Real.sqrt 5, -5 - Real.sqrt 5), (-5 - Real.sqrt 5, -5 + Real.sqrt 5)} := by
  sorry

end system_solution_l1429_142916


namespace domino_less_than_trimino_l1429_142938

/-- A domino tiling of a 2n × 2n grid -/
def DominoTiling (n : ℕ) := Fin (2*n) → Fin (2*n) → Bool

/-- A trimino tiling of a 3n × 3n grid -/
def TriminoTiling (n : ℕ) := Fin (3*n) → Fin (3*n) → Bool

/-- The number of domino tilings of a 2n × 2n grid -/
def numDominoTilings (n : ℕ) : ℕ := sorry

/-- The number of trimino tilings of a 3n × 3n grid -/
def numTriminoTilings (n : ℕ) : ℕ := sorry

/-- Theorem: The number of domino tilings of a 2n × 2n grid is less than
    the number of trimino tilings of a 3n × 3n grid for all positive n -/
theorem domino_less_than_trimino (n : ℕ) (h : n > 0) : 
  numDominoTilings n < numTriminoTilings n := by
  sorry

end domino_less_than_trimino_l1429_142938


namespace gauss_family_mean_age_l1429_142936

def gauss_family_ages : List ℕ := [8, 8, 8, 8, 16, 17]

theorem gauss_family_mean_age : 
  (gauss_family_ages.sum : ℚ) / gauss_family_ages.length = 65 / 6 := by
  sorry

end gauss_family_mean_age_l1429_142936


namespace circle_area_l1429_142949

theorem circle_area (x y : ℝ) : 
  (4 * x^2 + 4 * y^2 - 8 * x + 24 * y + 60 = 0) → 
  (∃ (center_x center_y radius : ℝ), 
    ((x - center_x)^2 + (y - center_y)^2 = radius^2) ∧ 
    (π * radius^2 = 5 * π)) := by
  sorry

end circle_area_l1429_142949


namespace f_properties_l1429_142961

def f (a x : ℝ) : ℝ := |1 - x - a| + |2 * a - x|

theorem f_properties (a x : ℝ) :
  (f a 1 < 3 ↔ a > -2/3 ∧ a < 4/3) ∧
  (a ≥ 2/3 → f a x ≥ 1) :=
by sorry

end f_properties_l1429_142961


namespace halfway_between_fractions_l1429_142912

theorem halfway_between_fractions :
  (3 / 4 + 5 / 6) / 2 = 19 / 24 := by sorry

end halfway_between_fractions_l1429_142912


namespace greatest_prime_factor_of_expression_l1429_142963

theorem greatest_prime_factor_of_expression : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (3^8 + 6^7) ∧ ∀ q : ℕ, q.Prime → q ∣ (3^8 + 6^7) → q ≤ p ∧ p = 131 :=
by sorry

end greatest_prime_factor_of_expression_l1429_142963


namespace sand_per_lorry_l1429_142908

/-- Calculates the number of tons of sand per lorry given the following conditions:
  * 500 bags of cement are provided
  * Cement costs $10 per bag
  * 20 lorries of sand are received
  * Sand costs $40 per ton
  * Total cost for all materials is $13000
-/
theorem sand_per_lorry (cement_bags : ℕ) (cement_cost : ℚ) (lorries : ℕ) (sand_cost : ℚ) (total_cost : ℚ) :
  cement_bags = 500 →
  cement_cost = 10 →
  lorries = 20 →
  sand_cost = 40 →
  total_cost = 13000 →
  (total_cost - cement_bags * cement_cost) / sand_cost / lorries = 10 := by
  sorry

#check sand_per_lorry

end sand_per_lorry_l1429_142908


namespace allocation_methods_six_individuals_l1429_142970

/-- The number of ways to allocate 6 individuals into 2 rooms -/
def allocation_methods (n : ℕ) : ℕ → ℕ
  | 1 => Nat.choose n 3  -- Exactly 3 per room
  | 2 => Nat.choose n 1 * Nat.choose (n-1) (n-1) +  -- 1 in first room
         Nat.choose n 2 * Nat.choose (n-2) (n-2) +  -- 2 in first room
         Nat.choose n 3 * Nat.choose (n-3) (n-3) +  -- 3 in first room
         Nat.choose n 4 * Nat.choose (n-4) (n-4) +  -- 4 in first room
         Nat.choose n 5 * Nat.choose (n-5) (n-5)    -- 5 in first room
  | _ => 0  -- For any other input

theorem allocation_methods_six_individuals :
  allocation_methods 6 1 = 20 ∧ allocation_methods 6 2 = 62 := by
  sorry

#eval allocation_methods 6 1  -- Should output 20
#eval allocation_methods 6 2  -- Should output 62

end allocation_methods_six_individuals_l1429_142970


namespace red_to_blue_ratio_l1429_142943

/-- Represents the number of beads of each color in Michelle's necklace. -/
structure Necklace where
  total : ℕ
  blue : ℕ
  red : ℕ
  white : ℕ
  silver : ℕ

/-- The conditions of Michelle's necklace. -/
def michelle_necklace : Necklace where
  total := 40
  blue := 5
  red := 10  -- This is derived, not given directly
  white := 15 -- This is derived, not given directly
  silver := 10

/-- The ratio of red beads to blue beads is 2:1. -/
theorem red_to_blue_ratio (n : Necklace) (h1 : n = michelle_necklace) 
    (h2 : n.white = n.blue + n.red) 
    (h3 : n.total = n.blue + n.red + n.white + n.silver) : 
  n.red / n.blue = 2 := by
  sorry

#check red_to_blue_ratio

end red_to_blue_ratio_l1429_142943


namespace kibble_remaining_is_seven_l1429_142964

/-- The amount of kibble remaining in Luna's bag after one day of feeding. -/
def kibble_remaining (initial_amount : ℕ) (mary_morning : ℕ) (mary_evening : ℕ) (frank_afternoon : ℕ) : ℕ :=
  initial_amount - (mary_morning + mary_evening + frank_afternoon + 2 * frank_afternoon)

/-- Theorem stating that the amount of kibble remaining is 7 cups. -/
theorem kibble_remaining_is_seven :
  kibble_remaining 12 1 1 1 = 7 := by
  sorry

end kibble_remaining_is_seven_l1429_142964


namespace min_distance_complex_l1429_142966

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ w, Complex.abs (z + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

end min_distance_complex_l1429_142966


namespace two_white_marbles_probability_l1429_142953

/-- The probability of drawing two white marbles consecutively without replacement from a bag containing 5 red marbles and 7 white marbles is 7/22. -/
theorem two_white_marbles_probability :
  let red_marbles : ℕ := 5
  let white_marbles : ℕ := 7
  let total_marbles : ℕ := red_marbles + white_marbles
  let prob_first_white : ℚ := white_marbles / total_marbles
  let prob_second_white : ℚ := (white_marbles - 1) / (total_marbles - 1)
  prob_first_white * prob_second_white = 7 / 22 :=
by sorry

end two_white_marbles_probability_l1429_142953


namespace smallest_distance_between_circles_l1429_142944

open Complex

theorem smallest_distance_between_circles (z w : ℂ) : 
  abs (z - (2 + 4*I)) = 2 →
  abs (w - (5 + 2*I)) = 4 →
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), abs (z' - (2 + 4*I)) = 2 → abs (w' - (5 + 2*I)) = 4 → abs (z' - w') ≥ min_dist) ∧
    min_dist = 6 - Real.sqrt 13 :=
sorry

end smallest_distance_between_circles_l1429_142944


namespace sum_of_quadratic_solutions_l1429_142900

theorem sum_of_quadratic_solutions : 
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 5 - (2*x - 8)
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 8 :=
by sorry

end sum_of_quadratic_solutions_l1429_142900


namespace circle_area_equals_rectangle_area_l1429_142992

theorem circle_area_equals_rectangle_area (R : ℝ) (h : R = 4) :
  π * R^2 = (2 * π * R) * (R / 2) := by
  sorry

end circle_area_equals_rectangle_area_l1429_142992


namespace cubic_root_sum_l1429_142917

/-- Given p, q, and r are the roots of x^3 - 8x^2 + 6x - 3 = 0,
    prove that p/(qr - 1) + q/(pr - 1) + r/(pq - 1) = 21.75 -/
theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 6*p - 3 = 0 → 
  q^3 - 8*q^2 + 6*q - 3 = 0 → 
  r^3 - 8*r^2 + 6*r - 3 = 0 → 
  p/(q*r - 1) + q/(p*r - 1) + r/(p*q - 1) = 21.75 := by
sorry

end cubic_root_sum_l1429_142917


namespace equation_solution_l1429_142984

theorem equation_solution : ∃ X : ℝ, 
  (1.5 * ((3.6 * 0.48 * X) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002) ∧ 
  (abs (X - 2.5) < 0.0000000000000005) := by
  sorry

end equation_solution_l1429_142984


namespace total_pears_picked_l1429_142942

theorem total_pears_picked (alyssa nancy michael : ℕ) 
  (h1 : alyssa = 42)
  (h2 : nancy = 17)
  (h3 : michael = 31) :
  alyssa + nancy + michael = 90 := by
  sorry

end total_pears_picked_l1429_142942


namespace probability_same_color_girls_marbles_l1429_142918

/-- The probability of all 4 girls selecting the same colored marble -/
def probability_same_color (total_marbles : ℕ) (white_marbles : ℕ) (black_marbles : ℕ) (num_girls : ℕ) : ℚ :=
  let prob_all_white := (white_marbles.factorial * (total_marbles - num_girls).factorial) / 
                        (total_marbles.factorial * (white_marbles - num_girls).factorial)
  let prob_all_black := (black_marbles.factorial * (total_marbles - num_girls).factorial) / 
                        (total_marbles.factorial * (black_marbles - num_girls).factorial)
  prob_all_white + prob_all_black

/-- The theorem stating the probability of all 4 girls selecting the same colored marble -/
theorem probability_same_color_girls_marbles : 
  probability_same_color 8 4 4 4 = 1 / 35 := by
  sorry

end probability_same_color_girls_marbles_l1429_142918


namespace determinant_scaling_l1429_142907

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 3 →
  Matrix.det !![3*x, 3*y; 6*z, 6*w] = 54 := by
  sorry

end determinant_scaling_l1429_142907


namespace overlap_area_is_one_l1429_142996

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  h_x : x < 3
  h_y : y < 3

/-- Represents a triangle on the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The two specific triangles on the grid -/
def triangle1 : GridTriangle := {
  p1 := ⟨0, 0, by norm_num, by norm_num⟩,
  p2 := ⟨2, 1, by norm_num, by norm_num⟩,
  p3 := ⟨1, 2, by norm_num, by norm_num⟩
}

def triangle2 : GridTriangle := {
  p1 := ⟨2, 2, by norm_num, by norm_num⟩,
  p2 := ⟨0, 1, by norm_num, by norm_num⟩,
  p3 := ⟨1, 0, by norm_num, by norm_num⟩
}

/-- Calculates the area of the overlapping region of two triangles -/
def overlapArea (t1 t2 : GridTriangle) : ℝ := sorry

/-- Theorem stating that the overlap area of the specific triangles is 1 -/
theorem overlap_area_is_one : overlapArea triangle1 triangle2 = 1 := by sorry

end overlap_area_is_one_l1429_142996


namespace alice_shopping_cost_l1429_142904

/-- Represents the shopping list and discounts --/
structure ShoppingTrip where
  apple_price : ℕ
  apple_quantity : ℕ
  bread_price : ℕ
  bread_quantity : ℕ
  cereal_price : ℕ
  cereal_quantity : ℕ
  cake_price : ℕ
  cheese_price : ℕ
  cereal_discount : ℕ
  bread_discount : Bool
  coupon_threshold : ℕ
  coupon_value : ℕ

/-- Calculates the total cost of the shopping trip --/
def calculate_total (trip : ShoppingTrip) : ℕ :=
  let apple_cost := trip.apple_price * trip.apple_quantity
  let bread_cost := if trip.bread_discount then trip.bread_price else trip.bread_price * trip.bread_quantity
  let cereal_cost := (trip.cereal_price - trip.cereal_discount) * trip.cereal_quantity
  let total := apple_cost + bread_cost + cereal_cost + trip.cake_price + trip.cheese_price
  if total ≥ trip.coupon_threshold then total - trip.coupon_value else total

/-- Theorem stating that Alice's shopping trip costs $38 --/
theorem alice_shopping_cost : 
  let trip : ShoppingTrip := {
    apple_price := 2,
    apple_quantity := 4,
    bread_price := 4,
    bread_quantity := 2,
    cereal_price := 5,
    cereal_quantity := 3,
    cake_price := 8,
    cheese_price := 6,
    cereal_discount := 1,
    bread_discount := true,
    coupon_threshold := 40,
    coupon_value := 10
  }
  calculate_total trip = 38 := by
sorry

end alice_shopping_cost_l1429_142904


namespace line_projections_parallel_implies_parallel_or_skew_l1429_142941

/-- Two lines in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- Projection of a line onto a plane -/
def project_line (l : Line3D) (p : Plane3D) : Line3D :=
  sorry

/-- Predicate for parallel lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for skew lines -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

theorem line_projections_parallel_implies_parallel_or_skew 
  (a b : Line3D) (α : Plane3D) :
  parallel (project_line a α) (project_line b α) →
  parallel a b ∨ skew a b :=
sorry

end line_projections_parallel_implies_parallel_or_skew_l1429_142941


namespace equation_solution_l1429_142980

theorem equation_solution :
  ∃ x : ℚ, x + 5/8 = 2 + 3/16 - 2/3 ∧ x = 43/48 := by
  sorry

end equation_solution_l1429_142980


namespace rectangle_perimeter_l1429_142928

theorem rectangle_perimeter (a b : ℤ) : 
  a ≠ b →  -- non-square condition
  a > 0 →  -- positive dimension
  b > 0 →  -- positive dimension
  a * b + 9 = 2 * a + 2 * b + 9 →  -- area plus 9 equals perimeter plus 9
  2 * (a + b) = 18 :=  -- perimeter equals 18
by sorry

end rectangle_perimeter_l1429_142928


namespace sum_of_digits_of_large_number_l1429_142995

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_large_number : sum_of_digits (10^95 - 95 - 2) = 840 := by sorry

end sum_of_digits_of_large_number_l1429_142995


namespace fermat_like_theorem_l1429_142919

theorem fermat_like_theorem : ∀ (x y z k : ℕ), x < k → y < k → x^k + y^k ≠ z^k := by
  sorry

end fermat_like_theorem_l1429_142919


namespace angle_measure_in_triangle_l1429_142989

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

-- Define the angle measure function
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_measure_in_triangle (A B C P : ℝ × ℝ) :
  Triangle A B C →
  angle_measure A B C = 40 →
  angle_measure A C B = 40 →
  angle_measure P A C = 20 →
  angle_measure P C B = 30 →
  angle_measure P B C = 20 := by
  sorry

end angle_measure_in_triangle_l1429_142989


namespace bonus_implication_l1429_142971

-- Define the universe of discourse
variable (Employee : Type)

-- Define the predicates
variable (completes_all_projects : Employee → Prop)
variable (receives_bonus : Employee → Prop)

-- Mr. Thompson's statement
variable (thompson_statement : ∀ (e : Employee), completes_all_projects e → receives_bonus e)

-- Theorem to prove
theorem bonus_implication :
  ∀ (e : Employee), ¬(receives_bonus e) → ¬(completes_all_projects e) := by
  sorry

end bonus_implication_l1429_142971


namespace marble_difference_l1429_142986

/-- Represents a jar of marbles -/
structure Jar :=
  (blue : ℕ)
  (green : ℕ)

/-- The problem statement -/
theorem marble_difference (jar1 jar2 : Jar) : 
  jar1.blue + jar1.green = jar2.blue + jar2.green →
  7 * jar1.green = 3 * jar1.blue →
  9 * jar2.green = jar2.blue →
  jar1.green + jar2.green = 80 →
  jar2.blue - jar1.blue = 40 := by
  sorry

end marble_difference_l1429_142986


namespace base_conversion_sum_l1429_142991

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_conversion_sum : 
  let base9 := toBase10 [1, 2, 3] 9
  let base8 := toBase10 [6, 5, 2] 8
  let base7 := toBase10 [4, 3, 1] 7
  base9 - base8 + base7 = 162 := by
  sorry

end base_conversion_sum_l1429_142991


namespace medicine_price_reduction_l1429_142925

theorem medicine_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 25)
  (h2 : final_price = 16)
  (h3 : final_price = original_price * (1 - x)^2)
  (h4 : 0 < x ∧ x < 1) : 
  x = 0.2 := by sorry

end medicine_price_reduction_l1429_142925


namespace inverse_composition_equality_l1429_142950

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the condition f⁻¹ ∘ g = λ x, 2*x - 4
variable (h : ∀ x, f⁻¹ (g x) = 2 * x - 4)

-- State the theorem
theorem inverse_composition_equality : g⁻¹ (f (-3)) = 1/2 := by
  sorry

end inverse_composition_equality_l1429_142950


namespace inverse_and_negation_of_union_subset_inverse_of_divisibility_negation_and_contrapositive_of_inequality_inverse_of_quadratic_inequality_l1429_142915

-- Define the sets A and B
variable (A B : Set α)

-- Define the divisibility relation
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

-- 1. Inverse and negation of "If x ∈ (A ∪ B), then x ∈ B"
theorem inverse_and_negation_of_union_subset (x : α) :
  (x ∈ B → x ∈ A ∪ B) ∧ (x ∉ A ∪ B → x ∉ B) := by sorry

-- 2. Inverse of "If a natural number is divisible by 6, then it is divisible by 2"
theorem inverse_of_divisibility :
  ¬(∀ n : ℕ, divides 2 n → divides 6 n) := by sorry

-- 3. Negation and contrapositive of "If 0 < x < 5, then |x-2| < 3"
theorem negation_and_contrapositive_of_inequality (x : ℝ) :
  ¬(¬(0 < x ∧ x < 5) → |x - 2| ≥ 3) ∧
  (|x - 2| ≥ 3 → ¬(0 < x ∧ x < 5)) := by sorry

-- 4. Inverse of "If (a-2)x^2 + 2(a-2)x - 4 < 0 holds for all x ∈ ℝ, then a ∈ (-2, 2)"
theorem inverse_of_quadratic_inequality (a : ℝ) :
  a ∈ Set.Ioo (-2) 2 →
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) := by sorry

end inverse_and_negation_of_union_subset_inverse_of_divisibility_negation_and_contrapositive_of_inequality_inverse_of_quadratic_inequality_l1429_142915


namespace isosceles_triangle_sides_l1429_142921

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

-- Define the properties of the triangle
def triangle_properties (t : IsoscelesTriangle) (area1 area2 : ℝ) : Prop :=
  area1 = 6 * 6 / 11 ∧ 
  area2 = 5 * 5 / 11 ∧ 
  area1 + area2 = 1 / 2 * t.base * (t.leg ^ 2 - (t.base / 2) ^ 2).sqrt

-- Theorem statement
theorem isosceles_triangle_sides 
  (t : IsoscelesTriangle) 
  (area1 area2 : ℝ) 
  (h : triangle_properties t area1 area2) : 
  t.base = 6 ∧ t.leg = 5 := by
  sorry


end isosceles_triangle_sides_l1429_142921


namespace white_pairs_coincide_l1429_142967

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  blue_white : ℕ

/-- Given the initial triangle counts and the number of coinciding pairs of various types,
    calculates the number of white-white pairs that coincide when the figure is folded -/
def calculate_white_pairs (counts : TriangleCounts) (pairs : CoincidingPairs) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, 5 white pairs coincide -/
theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) 
  (h1 : counts.red = 5)
  (h2 : counts.blue = 6)
  (h3 : counts.white = 9)
  (h4 : pairs.red_red = 3)
  (h5 : pairs.blue_blue = 2)
  (h6 : pairs.red_white = 3)
  (h7 : pairs.blue_white = 1) :
  calculate_white_pairs counts pairs = 5 :=
by sorry

end white_pairs_coincide_l1429_142967


namespace inequality_range_l1429_142931

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 :=
sorry

end inequality_range_l1429_142931


namespace smallest_distance_between_circles_l1429_142969

theorem smallest_distance_between_circles (z w : ℂ) :
  Complex.abs (z - (2 + 4*I)) = 2 →
  Complex.abs (w - (5 + 6*I)) = 4 →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 13 - 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 + 4*I)) = 2 →
      Complex.abs (w' - (5 + 6*I)) = 4 →
      Complex.abs (z' - w') ≥ min_dist :=
by sorry

end smallest_distance_between_circles_l1429_142969


namespace lindsay_squat_weight_l1429_142911

/-- The total weight Lindsey will squat -/
def total_weight (num_bands : ℕ) (resistance_per_band : ℕ) (dumbbell_weight : ℕ) : ℕ :=
  num_bands * resistance_per_band + dumbbell_weight

/-- Theorem stating the total weight Lindsey will squat -/
theorem lindsay_squat_weight :
  let num_bands : ℕ := 2
  let resistance_per_band : ℕ := 5
  let dumbbell_weight : ℕ := 10
  total_weight num_bands resistance_per_band dumbbell_weight = 20 :=
by
  sorry

end lindsay_squat_weight_l1429_142911


namespace max_value_expression_l1429_142958

theorem max_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 ∧
  (x^2 * y^2 * (x^2 + y^2) = 2 ↔ x = 1 ∧ y = 1) :=
by sorry

end max_value_expression_l1429_142958


namespace fish_tank_problem_l1429_142952

theorem fish_tank_problem (tank1_size : ℚ) (tank2_size : ℚ) (tank1_water : ℚ) 
  (fish2_length : ℚ) (fish_diff : ℕ) :
  tank1_size = 2 * tank2_size →
  tank1_water = 48 →
  fish2_length = 2 →
  fish_diff = 3 →
  ∃ (fish1_length : ℚ),
    fish1_length = 3 ∧
    (tank1_water / fish1_length - 1 = tank2_size / fish2_length + fish_diff) :=
by sorry

end fish_tank_problem_l1429_142952


namespace no_discount_possible_l1429_142946

theorem no_discount_possible (purchase_price : ℝ) (marked_price_each : ℝ) 
  (h1 : purchase_price = 50)
  (h2 : marked_price_each = 22.5) :
  2 * marked_price_each < purchase_price := by
  sorry

#eval 2 * 22.5 -- This will output 45.0, confirming the contradiction

end no_discount_possible_l1429_142946


namespace tom_initial_investment_l1429_142906

/-- Represents the business partnership between Tom and Jose -/
structure Partnership where
  tom_investment : ℕ
  jose_investment : ℕ
  tom_join_time : ℕ
  jose_join_time : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Tom's initial investment given the partnership details -/
def calculate_tom_investment (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that Tom's initial investment is 3000 given the problem conditions -/
theorem tom_initial_investment :
  let p : Partnership := {
    tom_investment := 0,  -- We don't know this value yet
    jose_investment := 45000,
    tom_join_time := 0,  -- Tom joined at the start
    jose_join_time := 2,
    total_profit := 54000,
    jose_profit := 30000
  }
  calculate_tom_investment p = 3000 := by
  sorry

end tom_initial_investment_l1429_142906


namespace philip_orange_collection_l1429_142973

/-- The number of oranges in Philip's collection -/
def num_oranges : ℕ := 178 * 2

/-- The number of groups of oranges -/
def orange_groups : ℕ := 178

/-- The number of oranges in each group -/
def oranges_per_group : ℕ := 2

/-- Theorem stating that the number of oranges in Philip's collection is 356 -/
theorem philip_orange_collection : num_oranges = 356 := by
  sorry

#eval num_oranges -- This will output 356

end philip_orange_collection_l1429_142973


namespace equilateral_triangle_between_poles_l1429_142926

theorem equilateral_triangle_between_poles (pole1 pole2 : ℝ) (h1 : pole1 = 11) (h2 : pole2 = 13) :
  let a := 8 * Real.sqrt 3
  (a ^ 2 = pole1 ^ 2 + 2 ^ 2) ∧
  (a ^ 2 = pole2 ^ 2 + 2 ^ 2) ∧
  (Real.sqrt (a ^ 2 - pole1 ^ 2) + Real.sqrt (a ^ 2 - pole2 ^ 2) = 2) :=
by sorry

end equilateral_triangle_between_poles_l1429_142926


namespace least_with_eight_factors_l1429_142902

/-- A function that returns the number of distinct positive factors of a natural number. -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly eight distinct positive factors. -/
def has_eight_factors (n : ℕ) : Prop := num_factors n = 8

/-- The theorem stating that 54 is the least positive integer with exactly eight distinct positive factors. -/
theorem least_with_eight_factors : 
  has_eight_factors 54 ∧ ∀ m : ℕ, m < 54 → ¬(has_eight_factors m) := by sorry

end least_with_eight_factors_l1429_142902


namespace negative_sum_l1429_142993

theorem negative_sum (a b c : ℝ) 
  (ha : -2 < a ∧ a < -1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : -1 < c ∧ c < 0) : 
  b + c < 0 := by
  sorry

end negative_sum_l1429_142993


namespace min_value_fraction_l1429_142905

theorem min_value_fraction (x : ℝ) (h : x > 12) :
  x^2 / (x - 12) ≥ 48 ∧ (x^2 / (x - 12) = 48 ↔ x = 24) := by
  sorry

end min_value_fraction_l1429_142905


namespace find_k_value_l1429_142959

theorem find_k_value (k : ℝ) (h1 : k ≠ 0) 
  (h2 : ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 - k * (x^2 + x + 3)) : 
  k = 3 := by
  sorry

end find_k_value_l1429_142959


namespace no_valid_tetrahedron_labeling_l1429_142999

/-- Represents a labeling of a tetrahedron's vertices -/
def TetrahedronLabeling := Fin 4 → Fin 4

/-- Checks if a labeling uses each number exactly once -/
def is_valid_labeling (l : TetrahedronLabeling) : Prop :=
  ∀ i : Fin 4, ∃! j : Fin 4, l j = i

/-- Calculates the sum of labels on a face -/
def face_sum (l : TetrahedronLabeling) (face : Fin 4 → Fin 3) : ℕ :=
  (face 0).val + (face 1).val + (face 2).val

/-- Checks if all face sums are equal -/
def all_face_sums_equal (l : TetrahedronLabeling) (faces : Fin 4 → (Fin 4 → Fin 3)) : Prop :=
  ∀ i j : Fin 4, face_sum l (faces i) = face_sum l (faces j)

/-- The main theorem stating that no valid labeling exists -/
theorem no_valid_tetrahedron_labeling (faces : Fin 4 → (Fin 4 → Fin 3)) :
  ¬∃ l : TetrahedronLabeling, is_valid_labeling l ∧ all_face_sums_equal l faces :=
sorry

end no_valid_tetrahedron_labeling_l1429_142999


namespace special_function_values_l1429_142909

/-- A function satisfying f(x + y) = 2 f(x) f(y) for all real x and y -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = 2 * f x * f y

/-- Theorem stating the possible values of f(1) for a SpecialFunction -/
theorem special_function_values (f : ℝ → ℝ) (hf : SpecialFunction f) :
  f 1 = 0 ∨ ∃ r : ℝ, f 1 = r :=
by sorry

end special_function_values_l1429_142909


namespace distance_difference_around_block_l1429_142974

/-- The difference in distance run by two people around a square block -/
def distanceDifference (blockSideLength : ℝ) (streetWidth : ℝ) : ℝ :=
  4 * (2 * streetWidth)

theorem distance_difference_around_block :
  let blockSideLength : ℝ := 400
  let streetWidth : ℝ := 20
  distanceDifference blockSideLength streetWidth = 160 := by sorry

end distance_difference_around_block_l1429_142974


namespace jake_and_sister_weight_l1429_142990

/-- Jake's current weight in pounds -/
def jakes_weight : ℕ := 156

/-- Jake's weight after losing 20 pounds -/
def jakes_reduced_weight : ℕ := jakes_weight - 20

/-- Jake's sister's weight in pounds -/
def sisters_weight : ℕ := jakes_reduced_weight / 2

/-- The combined weight of Jake and his sister -/
def combined_weight : ℕ := jakes_weight + sisters_weight

theorem jake_and_sister_weight : combined_weight = 224 := by
  sorry

end jake_and_sister_weight_l1429_142990


namespace correct_categorization_l1429_142960

def given_numbers : List ℚ := [-13.5, 5, 0, -10, 3.14, 27, -4/5, -15/100, 21/3]

def is_negative (x : ℚ) : Prop := x < 0
def is_non_negative (x : ℚ) : Prop := x ≥ 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ ¬(is_integer x)

def negative_numbers : List ℚ := [-13.5, -10, -4/5, -15/100]
def non_negative_numbers : List ℚ := [5, 0, 3.14, 27, 21/3]
def integers : List ℚ := [5, 0, -10, 27]
def negative_fractions : List ℚ := [-13.5, -4/5, -15/100]

theorem correct_categorization :
  (∀ x ∈ negative_numbers, is_negative x) ∧
  (∀ x ∈ non_negative_numbers, is_non_negative x) ∧
  (∀ x ∈ integers, is_integer x) ∧
  (∀ x ∈ negative_fractions, is_negative_fraction x) ∧
  (∀ x ∈ given_numbers, 
    (x ∈ negative_numbers ∨ x ∈ non_negative_numbers) ∧
    (x ∈ integers ∨ x ∈ negative_fractions ∨ (is_non_negative x ∧ ¬(is_integer x)))) := by
  sorry

end correct_categorization_l1429_142960


namespace two_percent_of_one_l1429_142920

theorem two_percent_of_one : (2 : ℚ) / 100 = (2 : ℚ) / 100 * 1 := by sorry

end two_percent_of_one_l1429_142920


namespace current_year_is_2021_l1429_142939

-- Define the given conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := kelsey_birth_year - 3
def sister_current_age : ℕ := 50

-- Define the theorem
theorem current_year_is_2021 :
  sister_birth_year + sister_current_age = 2021 :=
sorry

end current_year_is_2021_l1429_142939


namespace simplify_expression_l1429_142930

theorem simplify_expression : 18 * (8 / 12) * (1 / 6) = 2 := by
  sorry

end simplify_expression_l1429_142930


namespace dividend_calculation_l1429_142924

theorem dividend_calculation (k : ℕ) (quotient : ℕ) (h1 : k = 8) (h2 : quotient = 8) :
  k * quotient = 64 := by
  sorry

end dividend_calculation_l1429_142924


namespace min_value_of_f_l1429_142934

def f (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 7

theorem min_value_of_f :
  ∃ (y_min : ℝ), ∀ (y : ℝ), f y ≥ f y_min ∧ y_min = 3/2 := by
  sorry

end min_value_of_f_l1429_142934


namespace horner_V₁_eq_22_l1429_142929

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 4x^5 + 2x^4 + 3.5x^3 - 2.6x^2 + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- Coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-0.8, 1.7, -2.6, 3.5, 2, 4]

/-- V₁ in Horner's method for f(5) -/
def V₁ : ℝ := 4 * 5 + 2

theorem horner_V₁_eq_22 : V₁ = 22 := by
  sorry

#eval V₁  -- Should output 22

end horner_V₁_eq_22_l1429_142929


namespace expected_weekly_rain_l1429_142956

/-- The number of days in the week --/
def days : ℕ := 7

/-- The probability of sun (0 inches of rain) --/
def probSun : ℝ := 0.3

/-- The probability of 5 inches of rain --/
def probRain5 : ℝ := 0.4

/-- The probability of 12 inches of rain --/
def probRain12 : ℝ := 0.3

/-- The amount of rain on a sunny day --/
def rainSun : ℝ := 0

/-- The amount of rain on a day with 5 inches --/
def rain5 : ℝ := 5

/-- The amount of rain on a day with 12 inches --/
def rain12 : ℝ := 12

/-- The expected value of rainfall for one day --/
def expectedDailyRain : ℝ := probSun * rainSun + probRain5 * rain5 + probRain12 * rain12

/-- Theorem: The expected value of total rainfall for the week is 39.2 inches --/
theorem expected_weekly_rain : days * expectedDailyRain = 39.2 := by
  sorry

end expected_weekly_rain_l1429_142956


namespace ms_cole_students_l1429_142933

/-- Represents the number of students in each math level class taught by Ms. Cole -/
structure MathClasses where
  sixth_level : ℕ
  fourth_level : ℕ
  seventh_level : ℕ

/-- Calculates the total number of students Ms. Cole teaches -/
def total_students (classes : MathClasses) : ℕ :=
  classes.sixth_level + classes.fourth_level + classes.seventh_level

/-- Theorem stating the total number of students Ms. Cole teaches -/
theorem ms_cole_students : ∃ (classes : MathClasses), 
  classes.sixth_level = 40 ∧ 
  classes.fourth_level = 4 * classes.sixth_level ∧
  classes.seventh_level = 2 * classes.fourth_level ∧
  total_students classes = 520 := by
  sorry

end ms_cole_students_l1429_142933


namespace john_card_expenditure_l1429_142965

/-- The number of thank you cards John sent for Christmas gifts -/
def christmas_cards : ℕ := 20

/-- The number of thank you cards John sent for birthday gifts -/
def birthday_cards : ℕ := 15

/-- The cost of each thank you card in dollars -/
def card_cost : ℕ := 2

/-- The total cost of all thank you cards John bought -/
def total_cost : ℕ := (christmas_cards + birthday_cards) * card_cost

theorem john_card_expenditure :
  total_cost = 70 := by sorry

end john_card_expenditure_l1429_142965


namespace probability_of_a_l1429_142914

theorem probability_of_a (a b : Set α) (p : Set α → ℝ) 
  (h1 : p b = 2/5)
  (h2 : p (a ∩ b) = p a * p b)
  (h3 : p (a ∩ b) = 0.16000000000000003) :
  p a = 0.4 := by
sorry

end probability_of_a_l1429_142914


namespace min_cups_in_boxes_min_cups_for_100_boxes_l1429_142940

theorem min_cups_in_boxes : ℕ → ℕ
  | n => (n * (n + 1)) / 2

theorem min_cups_for_100_boxes :
  min_cups_in_boxes 100 = 5050 := by sorry

end min_cups_in_boxes_min_cups_for_100_boxes_l1429_142940


namespace hyperbola_standard_equation_l1429_142985

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  e : ℝ

-- Define the conditions
def hyperbola_conditions (h : Hyperbola) : Prop :=
  h.e = Real.sqrt 3 ∧
  h.c = Real.sqrt 3 * h.a ∧
  h.b = Real.sqrt 2 * h.a ∧
  8 / h.a^2 - 1 / h.a^2 = 1

-- Define the standard equation
def standard_equation (h : Hyperbola) : Prop :=
  ∀ (x y : ℝ), x^2 / 7 - y^2 / 14 = 1 ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1

-- Theorem statement
theorem hyperbola_standard_equation (h : Hyperbola) :
  hyperbola_conditions h → standard_equation h :=
sorry

end hyperbola_standard_equation_l1429_142985


namespace sum_of_coefficients_l1429_142922

theorem sum_of_coefficients (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-4 * x^2 + 14 * x + 38) / (x - 3)) →
  A + B = 46 := by
sorry

end sum_of_coefficients_l1429_142922


namespace log_inequality_l1429_142947

theorem log_inequality : 
  let a := Real.log 2 / Real.log (1/3)
  let b := (1/3)^2
  let c := 2^(1/3)
  a < b ∧ b < c := by sorry

end log_inequality_l1429_142947


namespace expected_correct_answers_l1429_142981

theorem expected_correct_answers 
  (total_problems : ℕ) 
  (katya_probability : ℚ) 
  (pen_probability : ℚ) 
  (katya_problems : ℕ) :
  total_problems = 20 →
  katya_probability = 4/5 →
  pen_probability = 1/2 →
  katya_problems ≥ 10 →
  katya_problems ≤ total_problems →
  (katya_problems : ℚ) * katya_probability + 
  (total_problems - katya_problems : ℚ) * pen_probability ≥ 13 := by
sorry

end expected_correct_answers_l1429_142981


namespace right_triangle_conditions_l1429_142977

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides

-- Define what it means for a triangle to be right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Define the conditions
def condition1 (t : Triangle) : Prop := t.A + t.B = t.C
def condition2 (t : Triangle) : Prop := ∃ (k : Real), t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k
def condition3 (t : Triangle) : Prop := t.A = 90 - t.B

-- Theorem statement
theorem right_triangle_conditions (t : Triangle) :
  (condition1 t ∨ condition2 t ∨ condition3 t) → isRightTriangle t :=
by sorry

end right_triangle_conditions_l1429_142977


namespace cube_surface_area_l1429_142901

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) :
  volume = 8 →
  volume = side^3 →
  surface_area = 6 * side^2 →
  surface_area = 24 := by
sorry

end cube_surface_area_l1429_142901


namespace square_difference_sum_l1429_142903

theorem square_difference_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 = 240 := by
  sorry

end square_difference_sum_l1429_142903
